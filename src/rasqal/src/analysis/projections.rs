// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2024 Oxford Quantum Circuits Ltd

use crate::analysis::solver::{QuantumSolver, SolverConfig, SolverResult};
use crate::config::RasqalConfig;
use crate::execution::RuntimeCollection;
use crate::features::QuantumFeatures;
use crate::graphs::AnalysisGraph;
use crate::hardware::Qubit;
use crate::runtime::{ActiveTracers, TracingModule};
use crate::smart_pointers::Ptr;
use crate::{with_mutable, with_mutable_self};
use log::{log, Level};
use ndarray::{array, Array2};
use num::range;
use num::traits::FloatConst;
use num_complex::{Complex, Complex64, ComplexFloat};
use std::cmp::Ordering;
use std::collections::hash_map::Keys;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::fmt::{Display, Formatter, Write};
use std::iter::zip;
use std::ops::{Deref, Mul, MulAssign};
use std::time::Instant;

/// A projected value that is either concretized and has a result, or in analysis mode and can be
/// queried LIKE it was a result, but we haven't actually executed on the QPU yet.
pub struct QuantumProjection {
  trace_module: Ptr<TracingModule>,
  engines: Ptr<RuntimeCollection>,
  instructions: Vec<Ptr<AnalysisOperation>>,
  cached_result: Option<AnalysisResult>,
  cached_filtered: HashMap<String, AnalysisResult>,
  solver_config: SolverConfig
}

/// A for-now list of linear gates and hardware operations that we can store and send to our
/// Python runtimes. In time these will be removed, and we'll reconstruct gates from
/// our other analysis structures.
enum AnalysisOperation {
  Initialize(),
  Reset(Vec<Qubit>),
  X(Qubit, f64),
  Y(Qubit, f64),
  Z(Qubit, f64),
  CX(Vec<Qubit>, Qubit, f64),
  CZ(Vec<Qubit>, Qubit, f64),
  CY(Vec<Qubit>, Qubit, f64),
  Measure(Vec<Qubit>)
}

impl AnalysisOperation {
  /// This only returns directly-attached qubits. So it does not return the controllers of a
  /// controlled operation, but it does return the target.
  pub fn associated_qubits(&self) -> Vec<&Qubit> {
    match self {
      AnalysisOperation::Initialize() => vec![],
      AnalysisOperation::Reset(qbs) => qbs.iter().collect(),
      AnalysisOperation::X(qb, _)
      | AnalysisOperation::Y(qb, _)
      | AnalysisOperation::Z(qb, _)
      | AnalysisOperation::CX(_, qb, _)
      | AnalysisOperation::CZ(_, qb, _)
      | AnalysisOperation::CY(_, qb, _) => vec![qb],
      AnalysisOperation::Measure(qbs) => qbs.iter().collect()
    }
  }
}

impl Display for AnalysisOperation {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    f.write_str(
      match self {
        AnalysisOperation::Initialize() => "init".to_string(),
        AnalysisOperation::Reset(qb) => format!(
          "Reset {}",
          qb.iter()
            .map(|val| val.to_string())
            .collect::<Vec<_>>()
            .join(", ")
        ),
        AnalysisOperation::X(qb, theta) => format!("X[{qb}] {theta}"),
        AnalysisOperation::Y(qb, theta) => format!("Y[{qb}] {theta}"),
        AnalysisOperation::Z(qb, theta) => format!("Z[{qb}] {theta}"),
        AnalysisOperation::CX(controlled, target, theta) => format!(
          "CX[{}->{}] {}",
          controlled
            .iter()
            .map(|val| val.to_string())
            .collect::<Vec<_>>()
            .join(","),
          target,
          theta
        ),
        AnalysisOperation::CZ(controlled, target, theta) => format!(
          "CZ[{}->{}] {}",
          controlled
            .iter()
            .map(|val| val.to_string())
            .collect::<Vec<_>>()
            .join(","),
          target,
          theta
        ),
        AnalysisOperation::CY(controlled, target, theta) => format!(
          "CY[{}->{}] {}",
          controlled
            .iter()
            .map(|val| val.to_string())
            .collect::<Vec<_>>()
            .join(","),
          target,
          theta
        ),
        AnalysisOperation::Measure(qb) => format!(
          "Measure {}",
          qb.iter()
            .map(|val| val.to_string())
            .collect::<Vec<_>>()
            .join(",")
        )
      }
      .as_str()
    )
  }
}

/// A projection is the umbrella for each individual quantum execution. When a qubit is first
/// activated a projection is created that will then take care of execution and analysis.
///
/// For a full description see the paper, but otherwise consider this a quantum analysis
/// structure and circuit synthesizer.
impl QuantumProjection {
  pub fn new(engines: &Ptr<RuntimeCollection>) -> QuantumProjection {
    QuantumProjection {
      engines: engines.clone(),
      instructions: Vec::new(),
      trace_module: Ptr::from(TracingModule::new()),
      cached_result: None,
      cached_filtered: HashMap::new(),
      solver_config: SolverConfig::off()
    }
  }

  pub fn with_tracer_and_config(
    engines: &Ptr<RuntimeCollection>, tracing_module: &Ptr<TracingModule>,
    config: &Ptr<RasqalConfig>
  ) -> QuantumProjection {
    QuantumProjection {
      engines: engines.clone(),
      instructions: Vec::new(),
      trace_module: tracing_module.clone(),
      cached_result: None,
      cached_filtered: HashMap::new(),
      solver_config: SolverConfig::with_config(config)
    }
  }

  /// Quick helper module as right now there's no sub-definition for projections.
  fn is_tracing(&self) -> bool { self.trace_module.has(ActiveTracers::Projections) }

  pub fn Init(&mut self) { self.add(AnalysisOperation::Initialize()); }

  pub fn Reset(&mut self, vec: Vec<Qubit>) { self.add(AnalysisOperation::Reset(vec)); }

  pub fn U(&mut self, qb: Qubit, theta: f64, phi: f64, lambda: f64) {
    self.add(AnalysisOperation::Z(qb.clone(), lambda));
    self.add(AnalysisOperation::Y(qb.clone(), phi));
    self.add(AnalysisOperation::Z(qb, theta));
  }

  pub fn X(&mut self, qb: Qubit, radian: f64) { self.add(AnalysisOperation::X(qb, radian)); }

  pub fn Y(&mut self, qb: Qubit, radian: f64) { self.add(AnalysisOperation::Y(qb, radian)); }

  pub fn Z(&mut self, qb: Qubit, radian: f64) { self.add(AnalysisOperation::Z(qb, radian)); }

  pub fn CX(&mut self, controls: Vec<Qubit>, target: Qubit, radian: f64) {
    self.add(AnalysisOperation::CX(controls, target, radian));
  }

  pub fn CZ(&mut self, controls: Vec<Qubit>, target: Qubit, radian: f64) {
    self.add(AnalysisOperation::CZ(controls, target, radian));
  }

  pub fn CY(&mut self, controls: Vec<Qubit>, target: Qubit, radian: f64) {
    self.add(AnalysisOperation::CY(controls, target, radian));
  }

  pub fn Measure(&mut self, qbs: Vec<Qubit>) { self.add(AnalysisOperation::Measure(qbs)); }

  /// Adds this operation to the projection.
  fn add(&mut self, inst: AnalysisOperation) {
    // Clear any pre-computed results upon a change to the state.
    if self.cached_result.is_some() {
      self.cached_result = None;
      self.cached_filtered.clear();
    }
    self.instructions.push(Ptr::from(inst));
  }

  /// Equality across projections for specific qubit.
  pub fn is_equal_for(&self, other: &Self, qbs: Option<&Vec<i64>>) -> bool {
    // TODO: Needs far more nuanced equality check, as we want to check on predicted values and
    //  more tightened boundaries.

    // If we're full comparison, do a quick short-circuit.
    if qbs.is_none() && self.instructions.len() != other.instructions.len() {
      return false;
    }

    let index_set: Option<HashSet<&i64>> = qbs.map(|val| HashSet::from_iter(val));
    for (ours, theirs) in zip(&self.instructions, &other.instructions) {
      if let Some(qubits) = index_set.as_ref() {
        let ours_match = ours
          .associated_qubits()
          .iter()
          .map(|val| val.index)
          .any(|val| qubits.contains(&val));
        let theirs_match = theirs
          .associated_qubits()
          .iter()
          .map(|val| val.index)
          .any(|val| qubits.contains(&val));

        // Skip comparison of instructions which don't have anything to do with our filter,
        // return false if we have one which does relate but the other does not.
        if !ours_match && !theirs_match {
          continue;
        } else if ours_match != theirs_match {
          return false;
        }
      }

      // TODO: These instructions shouldn't live long, so just do string compare. Inefficient but
      //  convenient.
      if ours.to_string() != theirs.to_string() {
        return false;
      }
    }

    true
  }

  /// Perform quantum circuit prediction and return acceptable results.
  fn solve(&mut self) -> AnalysisResult {
    if !self.solver_config.active {
      return AnalysisResult::empty();
    }

    let start = Instant::now();
    let qsolver = QuantumSolver::with_trace(self.trace_module.clone());
    for inst in self.instructions.iter() {
      match inst.deref() {
        AnalysisOperation::Initialize() => {}
        AnalysisOperation::Reset(qbs) => {
          for qubit in qbs {
            qsolver.reset(qubit);
          }
        }
        AnalysisOperation::X(qb, radians) => {
          qsolver.X(qb, radians);
        }
        AnalysisOperation::Y(qb, radians) => {
          qsolver.Y(qb, radians);
        }
        AnalysisOperation::Z(qb, radians) => {
          qsolver.Z(qb, radians);
        }
        AnalysisOperation::CX(controls, targets, radians) => {
          qsolver.CX(controls, targets, radians);
        }
        AnalysisOperation::CZ(controls, targets, radians) => {
          qsolver.CZ(controls, targets, radians);
        }
        AnalysisOperation::CY(controls, targets, radians) => {
          qsolver.CY(controls, targets, radians);
        }
        AnalysisOperation::Measure(qbs) => {
          for qb in qbs {
            qsolver.measure(qb);
          }
        }
      }
    }

    let took = start.elapsed();
    let ms_duration = took.as_millis();
    if ms_duration > 1000 {
      log!(Level::Info, "Building solver took {}ms", ms_duration);
    } else {
      log!(Level::Info, "Building solver took {}s", took.as_secs());
    }

    // For the projections, for now we only accept fully quantified results. Strip all
    // unknown values.
    let mut solver_results = Vec::new();
    for result in qsolver.solve() {
      if !result.bitstring.contains("X") {
        solver_results.push(result);
      }
    }

    AnalysisResult::from_solver_result(solver_results)
  }

  /// Get results for this entire state.
  pub fn results(&mut self) -> AnalysisResult { self.concretize().clone() }

  /// Extracts the results for this particular set of qubits from the results of running this
  /// projection.
  ///
  /// For example if your results are 01: 150, 00: 50 and ask for qubit 0 you'll get the
  /// result 0: 200. It needs to be pointed out that this needs to be viewed as a window into the
  /// overall result, not something that can be viewed by itself, because you lose all the nuance
  /// around the overall state.
  ///
  /// It's great for asking more brute-force questions like 'is this qubit overwhelmingly 1 in the
  /// results' and things of that sort though. This is also used for implicit conditional
  /// evaluations.
  pub fn results_for(&mut self, qb: &Vec<Qubit>) -> AnalysisResult {
    if qb.is_empty() {
      return AnalysisResult::empty();
    }

    // Check if we have a cached value, if so, return.
    let positions = qb.iter().map(|val| val.index as usize).collect::<Vec<_>>();
    let cache_key = positions
      .iter()
      .map(|val| val.to_string())
      .collect::<Vec<_>>()
      .join(",");
    if let Some(cached) = self.cached_filtered.get(&cache_key) {
      return cached.clone();
    }

    let results = self.concretize();

    // Strip out set qubits from the results. So if you have 01010: 50 and 01011: 7
    let mut new_distribution: HashMap<String, i64> = HashMap::new();
    for (key, value) in results.distribution.iter() {
      // -1 for zero-indexing.
      let key_length = key.len() - 1;
      let mut new_key = String::new();
      for index in positions.iter() {
        if let Some(nth_value) = key.chars().nth(key_length - index) {
          new_key.push(nth_value);
        }
      }

      if !new_key.is_empty() {
        let existing = if let Some(existing) = new_distribution.get(new_key.as_str()) {
          existing
        } else {
          &0
        };

        new_distribution.insert(new_key.clone(), value + existing);
      }
    }

    let new_results = AnalysisResult::new(new_distribution);
    self.cached_filtered.insert(cache_key, new_results.clone());
    if self.is_tracing() {
      log!(
        Level::Info,
        "Results for [{}]: {}",
        qb.iter()
          .map(|val| val.to_string())
          .collect::<Vec<_>>()
          .join(", "),
        new_results.to_string()
      );
    }

    new_results
  }

  /// Take the projection so far, build up a backend execution and then execute against an
  /// available QPU.
  pub fn concretize(&mut self) -> &AnalysisResult {
    if self.cached_result.is_some() {
      return self.cached_result.as_ref().unwrap();
    }

    let mut query_result = self.solve();
    if query_result.is_empty() {
      let start = Instant::now();
      let features = QuantumFeatures::default();
      let runtime = self.engines.find_capable_QPU(&features).unwrap_or_else(|| {
        panic!(
          "Cannot find QPU with these features available: [{}]",
          features
        )
      });

      let builder = runtime.create_builder();
      for inst in self.instructions.iter() {
        match inst.deref() {
          AnalysisOperation::Initialize() => {}
          AnalysisOperation::Reset(qbs) => {
            for qubit in qbs {
              builder.reset(qubit);
            }
          }
          AnalysisOperation::X(qb, radians) => {
            builder.x(qb, *radians);
          }
          AnalysisOperation::Y(qb, radians) => {
            builder.y(qb, *radians);
          }
          AnalysisOperation::Z(qb, radians) => {
            builder.z(qb, *radians);
          }
          AnalysisOperation::CX(controls, targets, radians) => {
            builder.cx(controls, targets, *radians);
          }
          AnalysisOperation::CZ(controls, targets, radians) => {
            builder.cz(controls, targets, *radians);
          }
          AnalysisOperation::CY(controls, targets, radians) => {
            builder.cy(controls, targets, *radians);
          }
          AnalysisOperation::Measure(qbs) => {
            for qb in qbs {
              builder.measure(qb);
            }
          }
        }
      }

      query_result = runtime.execute(&builder);
      let took = start.elapsed();
      log!(Level::Info, "QPU execution took {}ms.", took.as_millis());
    }

    self.cached_result = Some(query_result);

    if self.is_tracing() {
      log!(Level::Info, "Executed circuit:");
      for inst in self.instructions.iter() {
        log!(Level::Info, "{}", inst.to_string());
      }

      // Order results so you can easily compare two side-by-side.
      let mut result_values = self
        .cached_result
        .as_ref()
        .unwrap()
        .distribution
        .iter()
        .collect::<Vec<_>>();
      result_values.sort_by(|(left_key, _), (right_key, _)| left_key.cmp(right_key));
      if result_values.is_empty() {
        log!(Level::Info, "No results from this execution.");
      } else {
        log!(Level::Info, "Projection results:");
      }

      for (key, value) in result_values.iter() {
        log!(Level::Info, "  \"{}\": {}", key, value);
      }
    }

    self.cached_result.as_ref().unwrap()
  }
}

impl Clone for QuantumProjection {
  fn clone(&self) -> Self {
    QuantumProjection {
      trace_module: self.trace_module.clone(),
      engines: self.engines.clone(),
      instructions: self.instructions.clone(),
      cached_result: self.cached_result.clone(),
      cached_filtered: self.cached_filtered.clone(),
      solver_config: self.solver_config.clone()
    }
  }
}

impl Display for QuantumProjection {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { f.write_str("q-projection") }
}

impl PartialEq<Self> for QuantumProjection {
  fn eq(&self, other: &Self) -> bool { self.is_equal_for(other, None) }
}

impl PartialOrd for QuantumProjection {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    // TODO
    Some(Ordering::Equal)
  }
}

impl Eq for QuantumProjection {}

/// Non-deferred result distribution from a QPU execution.
pub struct AnalysisResult {
  pub distribution: HashMap<String, i64>
}

impl AnalysisResult {
  pub fn new(distribution: HashMap<String, i64>) -> AnalysisResult {
    AnalysisResult { distribution }
  }

  pub fn from_solver_result(res: Vec<SolverResult>) -> AnalysisResult {
    let mut distribution = HashMap::new();
    for result in res.iter() {
      distribution.insert(
        result.bitstring.clone(),
        (result.probability * 100.0) as i64
      );
    }
    AnalysisResult::new(distribution)
  }

  pub fn is_empty(&self) -> bool { self.size() == 0 }

  /// Return size of the results register in qubits.
  pub fn size(&self) -> usize { self.distribution.keys().next().map_or(0, |val| val.len()) }

  pub fn one() -> AnalysisResult { AnalysisResult::new(HashMap::from([("1".to_string(), 100)])) }

  pub fn zero() -> AnalysisResult { AnalysisResult::new(HashMap::from([("0".to_string(), 100)])) }

  pub fn empty() -> AnalysisResult { AnalysisResult::default() }

  /// Check if this distribution can be considered true/false.
  ///
  /// This is done by counting the instances of 0/1 in a particular bitstring, and if one
  /// is more than the other adding its count to a rolling 1/2 total. If the bitstring has equal
  /// numbers, such as 0011, it is discarded as neither.
  ///
  /// If the counts of both at the end are equal it will default to false.
  pub fn is_one(&self) -> bool {
    let mut zeros = 0;
    let mut ones = 0;
    for (key, val) in self.distribution.iter() {
      let length = key.len();
      let boundary = length / 2;
      let zero_count = key.matches("0").count();
      let one_count = length - zero_count;

      // If there are more zeros, add to zero count, if equal, ignore, otherwise one.
      if zero_count > boundary {
        zeros += val;
      } else if one_count > boundary {
        ones += val;
      }
    }

    // Default to zero if equals.
    ones > zeros
  }

  pub fn is_zero(&self) -> bool { !self.is_one() }
}

impl PartialEq for AnalysisResult {
  fn eq(&self, other: &Self) -> bool {
    // TODO: decide whether to do proper distribution analysis
    let self_is_one = self.is_one();
    let other_is_one = self.is_one();
    self_is_one == other_is_one
  }
}

impl Default for AnalysisResult {
  fn default() -> Self { AnalysisResult::new(HashMap::new()) }
}

impl Eq for AnalysisResult {}

impl Clone for AnalysisResult {
  fn clone(&self) -> Self { AnalysisResult::new(self.distribution.clone()) }
}

impl Display for AnalysisResult {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    f.debug_map().entries(self.distribution.iter()).finish()
  }
}
