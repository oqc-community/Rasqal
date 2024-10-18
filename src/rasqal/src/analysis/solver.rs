// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2024 Oxford Quantum Circuits Ltd

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
use num::traits::float::TotalOrder;
use num::traits::real::Real;
use num::traits::FloatConst;
use num::{range, Zero};
use num_complex::{Complex, Complex64, ComplexFloat};
use std::any::Any;
use std::cmp::Ordering;
use std::collections::hash_map::Keys;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::fmt::{Display, Formatter, Write};
use std::iter::zip;
use std::ops::{Deref, Index, Mul, MulAssign};
use std::rc::Rc;
use std::time::Instant;

/// More succinct way to initialize complex numbers in the matrix'.
macro_rules! C {
  ($real:expr, $img:expr) => {
    Complex::new($real, $img)
  };
}

/// An is-near check up to 4 dp. In a perfect world if the boundaries are constants it should
/// compile away the operations.
macro_rules! is_near {
  ($numb:expr, $bound:expr) => {
    $numb < $bound + 0.0001 && $numb > $bound - 0.0001
  };
}

/// Construct which holds the entanglement information between two qubits, shared between both.
#[derive(Clone)]
pub struct Tangle {
  left: Ptr<AnalysisQubit>,
  state: Ptr<EntangledFragment>,
  right: Ptr<AnalysisQubit>
}

impl Tangle {
  pub fn from_qubits(left: &Ptr<AnalysisQubit>, right: &Ptr<AnalysisQubit>) -> Tangle {
    Tangle {
      left: left.clone(),
      state: Ptr::from(left.state.entangle_with(&right.state)),
      right: right.clone()
    }
  }

  /// Helper method to just return a qubit of a certain index. Returns None if neither match.
  pub fn with_index(&self, index: &i64) -> Option<&Ptr<AnalysisQubit>> {
    if self.left.index == *index {
      Some(&self.left)
    } else if self.right.index == *index {
      Some(&self.right)
    } else {
      None
    }
  }

  /// Are we currently entangled?
  pub fn is_entangled(&self) -> bool {
    // I would prefer this be static but can't without shared pointer workarounds, which at
    // that point value is questionable.
    let czero: Complex<f64> = Complex::zero();
    self.state.get((2, 0)) != &czero
      || self.state.get((2, 1)) != &czero
      || self.state.get((3, 0)) != &czero
      || self.state.get((3, 1)) != &czero
  }
}

/// Solved entanglement metadata between qubits. Holds teh ratio of entanglement and with what
/// qubit, optionally if the resultant entanglement is inferred by another entanglement.
///
/// This means if you have Q0~Q1~Q2 any entanglement information for Q0 about Q2 will be via Q1.
#[derive(Clone)]
pub struct EntanglementMetadata {
  qubit: i64,

  /// Entanglement is inferred via this qubit.
  via: Option<i64>,

  /// Entanglement ratio with this particular qubit.
  ratio: f64
}

impl EntanglementMetadata {
  pub fn new(qubit: i64, ratio: f64) -> EntanglementMetadata {
    EntanglementMetadata {
      qubit,
      via: None,
      ratio
    }
  }

  pub fn with_via(qubit: i64, via: i64, ratio: f64) -> EntanglementMetadata {
    EntanglementMetadata {
      qubit,
      via: Some(via),
      ratio
    }
  }
}

#[derive(Clone)]
pub struct MeasureAnalysis {
  qubit: i64,
  probability: f64,
  entangled_with: Vec<EntanglementMetadata>
}

impl MeasureAnalysis {
  pub fn new(
    qubit: i64, result: f64, entangled_with: Vec<EntanglementMetadata>
  ) -> MeasureAnalysis {
    let result = if result < 0.0 { -result } else { result };
    MeasureAnalysis {
      qubit,
      probability: result,
      entangled_with
    }
  }

  pub fn qubit(qubit: i64, result: f64) -> MeasureAnalysis {
    MeasureAnalysis::new(qubit, result, Vec::new())
  }

  pub fn entangled_qubit(
    qubit: i64, result: f64, entangled_with: Vec<EntanglementMetadata>
  ) -> MeasureAnalysis {
    MeasureAnalysis::new(qubit, result, entangled_with)
  }
}

impl Display for MeasureAnalysis {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    let tangles = self
      .entangled_with
      .iter()
      .map(|val| format!("Q{}~{:.2}", val.qubit, val.ratio))
      .collect::<Vec<_>>();
    let mut additions = String::new();
    if !tangles.is_empty() {
      additions = format!(" with [{}]", tangles.join(", "));
    }

    f.write_str(&format!("{:.2}%{}", self.probability * 100., additions))
  }
}

// TODO: Split entangled/unentangled qubits, the former are only represented by their
//  tangles anyway so there's a clean distinction between them.
#[derive(Clone)]
pub struct AnalysisQubit {
  index: i64,

  /// Record of the raw qubit sans entanglement.
  state: Ptr<QubitFragment>,

  /// All the tangles between this qubit and others. Key is the index of the other qubit, along
  /// with a 4x4 density matrix.
  tangles: Ptr<HashMap<i64, Ptr<Tangle>>>,
  trace_module: Ptr<TracingModule>
}

impl AnalysisQubit {
  pub fn new(
    index: i64, state: Ptr<QubitFragment>, tangles: HashMap<i64, Ptr<Tangle>>,
    tracer: Ptr<TracingModule>
  ) -> AnalysisQubit {
    AnalysisQubit {
      index,
      state,
      tangles: Ptr::from(tangles),
      trace_module: tracer
    }
  }

  fn is_tracing(&self) -> bool { self.trace_module.has(ActiveTracers::Solver) }

  pub fn with_index(index: i64, tracer: Ptr<TracingModule>) -> AnalysisQubit {
    AnalysisQubit::new(
      index,
      Ptr::from(QubitFragment::DefaultQubit()),
      HashMap::default(),
      tracer
    )
  }

  pub fn entangled_with(&self) -> Keys<'_, i64, Ptr<Tangle>> { self.tangles.keys() }

  pub fn is_entangled_with(&self, index: &i64) -> bool { self.tangles.contains_key(index) }

  pub fn is_entangled(&self) -> bool { !self.tangles.is_empty() }

  pub fn entangle(&self, other: &Ptr<AnalysisQubit>) {
    if self.is_entangled_with(&other.index) {
      return;
    }

    let tangle = Ptr::from(Tangle::from_qubits(&Ptr::from(self), other));
    with_mutable_self!(self.tangles.insert(other.index, tangle.clone()));
    with_mutable!(other.tangles.insert(self.index, tangle));
  }

  /// Retrieve the measurement information about this qubit but _don't_ sympathetically snap other
  /// qubits. This is important for when you're gathering measure information of qubits measured
  /// simultaneously or otherwise peeking at a qubit in isolation.
  pub fn measure(&self) -> MeasureAnalysis {
    if self.tangles.is_empty() {
      MeasureAnalysis::qubit(self.index, self.state.get((1, 1)).re)
    } else {
      fn recurse_chains(
        current_qubit: &i64, tangles: &Ptr<HashMap<i64, Ptr<Tangle>>>,
        results: &mut Vec<EntanglementMetadata>, guard: &mut HashSet<i64>
      ) {
        guard.insert(*current_qubit);
        for (key, tangle) in tangles.iter() {
          if guard.contains(key) {
            continue;
          }

          // TODO: Only checks for 11 / 00 entanglement, not reversed, need to see how that
          //  plays out.
          let entangled = tangle.state.get((3, 0)).re;
          let mut max = tangle.state.get((0, 0)).re;
          let mut next = tangle.state.get((1, 1)).re;
          if next >= max {
            max = next;
          }

          next = tangle.state.get((2, 2)).re;
          if next >= max {
            max = next;
          }

          next = tangle.state.get((3, 3)).re;
          if next >= max {
            max = next;
          }

          results.push(EntanglementMetadata::with_via(
            *key,
            current_qubit.clone(),
            max / entangled
          ));
          recurse_chains(
            key,
            &tangle.with_index(key).unwrap().tangles,
            results,
            guard
          );
        }
      }

      // Collect full entanglement metadata across every chain at this moment in time.
      let mut entanglement_meta = Vec::new();
      let mut guard = HashSet::new();
      recurse_chains(
        &self.index,
        &self.tangles,
        &mut entanglement_meta,
        &mut guard
      );

      let mut percentage = 0.0;
      for (key, tangle) in self.tangles.iter() {
        // Depending which qubit we're looking at, our 'is one' check on a cell slightly changes.
        percentage = percentage
          + (if tangle.left.index == self.index {
            tangle.state.get((1, 1)).re + tangle.state.get((3, 3)).re
          } else {
            tangle.state.get((2, 2)).re + tangle.state.get((3, 3)).re
          });
      }

      percentage = percentage / self.tangles.len() as f64;
      MeasureAnalysis::entangled_qubit(self.index, percentage, entanglement_meta)
    }
  }

  /// Measures this qubit then snaps entanglement.
  pub fn measure_then_snap(&self) -> MeasureAnalysis {
    let results = self.measure();
    self.snap_entanglement();
    results
  }

  /// Sympathetically snaps entangled qubits to their appropriate values then removes the
  /// tangles from this qubit.
  pub fn snap_entanglement(&self) {
    if !self.tangles.is_empty() {
      for (key, tangle) in self.tangles.iter() {
        // After retrieving our metadata, snap the other qubit and slice our entanglement linkage.
        let other_qubit = tangle
          .with_index(key)
          .expect("Should be entangled with qubit.");
        with_mutable!(other_qubit.tangles.remove(&self.index));
        other_qubit.sympathize();
      }

      with_mutable_self!(self.tangles.clear());
    }
  }

  /// Sympathetically snap to a result after an entangled qubit has been measured. Separate from
  /// snap() as that is the qubit whose entanglement is forcibly snapped.
  pub fn sympathize(&self) {
    // TODO: Double-check what happens when we sympathetically snap entanglement to unmeasured
    //  qubits. Precision is the name of the game.
  }

  /// Applies this gate to this qubit and all tangles.
  pub fn apply(&self, gate: &GateFragment) {
    // To reduce verbosity we only trace multi-qubit gates. Applications of normal gates going
    // wrong can be visibly seen by other tracing methods.
    let is_tracing = self.is_tracing() && gate.affected_qubits == 2;

    let mut tracer = Vec::new();
    if is_tracing {
      tracer.push(format!(
        "\nQ{}:\n{}",
        self.index,
        self.state.stringify_matrix().join("\n")
      ));
    }

    with_mutable_self!(self.state.apply(gate));

    // If we're a single-qubit gate, expand.
    let expanded_gate = if gate.affected_qubits == 2 {
      gate
    } else {
      &gate.tensor(&MatrixFragment::id())
    };

    let inverted_gate = expanded_gate.invert();
    let mut unentangled = Vec::new();
    for tangle in self.tangles.values() {
      // If our actual target is inverted, invert the matrix too.
      let applied_gate = if tangle.right.index == self.index {
        &inverted_gate
      } else {
        expanded_gate
      };

      let mut before = None;
      if is_tracing {
        before = Some(tangle.state.stringify_matrix());
      }

      if let Some(error) = with_mutable!(tangle.state.apply(&applied_gate)) {
        panic!("{}", error);
      }

      if is_tracing {
        let mut before = before.unwrap();
        let mut stringified_gate = applied_gate.stringify_matrix();
        let after = tangle.state.stringify_matrix();

        let mut composite = Vec::new();
        for id in 0..4 {
          composite.push(format!(
            "{} x {} > {}",
            stringified_gate.index(id),
            before.index(id),
            after.index(id)
          ));
        }

        tracer.push(format!(
          "\n<{}~{}>:\n{}",
          tangle.left.index,
          tangle.right.index,
          composite.join("\n")
        ));
      }

      // If our rotation has removed entanglement, drop the tangle entirely.
      if !tangle.is_entangled() {
        unentangled.push(tangle);
      }
    }

    if is_tracing {
      log!(Level::Info, "{}\n", tracer.join("\n"));
    }

    // If we're no longer entangled remove it from both qubits.
    for tangle in unentangled.iter() {
      with_mutable!(tangle.left.tangles.remove(&tangle.right.index));
      with_mutable!(tangle.right.tangles.remove(&tangle.left.index));
    }
  }

  pub fn X(&self, radians: &f64) { self.apply(&GateFragment::X(radians)); }

  pub fn Y(&self, radians: &f64) { self.apply(&GateFragment::Y(radians)); }

  pub fn Z(&self, radians: &f64) { self.apply(&GateFragment::Z(radians)) }

  pub fn CX(&self, control: &i64, radians: &f64) { self.apply(&GateFragment::CX(radians)) }

  pub fn CZ(&self, control: &i64, radians: &f64) { self.apply(&GateFragment::CZ(radians)) }

  pub fn CY(&self, control: &i64, radians: &f64) { self.apply(&GateFragment::CY(radians)) }

  fn stringify(&self, indent_level: i32) -> Vec<String> {
    let mut result = Vec::new();
    let mut base_indent = String::new();
    for multiplier in 0..indent_level {
      base_indent = format!("{}    ", base_indent);
    }
    let indent = format!("{}    ", base_indent);

    result.push(format!("{}{{\n", base_indent));
    result.push(format!(
      "{}Q{}: {:.2}%\n",
      indent,
      self.index,
      self.state.get((1, 1)).re
    ));
    for matrix_fragment in self.state.stringify_matrix() {
      result.push(format!("{}{}\n", indent, matrix_fragment));
    }

    if !self.tangles.is_empty() {
      let mut tangles = self.tangles.iter().collect::<Vec<_>>();
      tangles.sort_by_key(|val| val.0);
      for (index, state) in tangles {
        result.push(format!("{}\n", indent));
        result.push(format!(
          "{}<{}~{}>:\n",
          indent, state.left.index, state.right.index
        ));
        for matrix_fragment in &state.state.stringify_matrix() {
          result.push(format!("{}{}\n", indent, matrix_fragment));
        }
      }
    }

    result.push(format!("{}}},\n", base_indent));
    result
  }
}

impl Display for AnalysisQubit {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    for line in self.stringify(0) {
      f.write_str(&line);
    }
    f.write_str("")
  }
}

/// A cluster of entangled states that should be treated as an individual cohesive state.
#[derive(Clone)]
pub struct EntanglementCluster {
  qubits: Ptr<HashMap<i64, Ptr<AnalysisQubit>>>,
  trace_module: Ptr<TracingModule>,
  solver: Ptr<QuantumSolver>
}

impl EntanglementCluster {
  pub fn new(solver: Ptr<QuantumSolver>, trace_module: &Ptr<TracingModule>) -> EntanglementCluster {
    EntanglementCluster {
      qubits: Ptr::from(HashMap::default()),
      trace_module: trace_module.clone(),
      solver
    }
  }

  fn is_tracing(&self) -> bool { self.trace_module.has(ActiveTracers::Solver) }

  pub fn spans(&self) -> Keys<'_, i64, Ptr<AnalysisQubit>> { self.qubits.keys() }

  /// Gets the qubit at this index crom this cluster. Assumes existence.
  pub fn qubit_for(&self, index: &i64) -> &Ptr<AnalysisQubit> { self.qubits.get(&index).unwrap() }

  pub fn merge(&self, other: &Ptr<EntanglementCluster>) {
    // No need to check for existence since if a qubit is related it will already be in a
    // cluster together.
    for (index, qubit) in other.qubits.iter() {
      with_mutable_self!(self.qubits.insert(index.clone(), qubit.clone()));
    }
  }

  pub fn add_then_entangle(&self, qubit_one: &Ptr<AnalysisQubit>, qubit_two: &i64) {
    self.add(qubit_one);
    self.entangle(&qubit_one.index, qubit_two);
  }

  /// Adds this qubit to the cluster.
  pub fn add(&self, qubit: &Ptr<AnalysisQubit>) {
    if !self.qubits.contains_key(&qubit.index) {
      with_mutable_self!(self.qubits.insert(qubit.index, qubit.clone()));
    }
  }

  /// Remove this qubit from the cluster, including all entanglement information.
  pub fn remove(&self, index: &i64) {
    // Copy because we're modifying the underlying map.
    if let Some(qubit) = self.qubits.get(&index) {
      if self.is_tracing() {
        log!(Level::Info, "Removing {} from cluster.", index)
      }

      // We'll remove ourselves once we've deleted our final tangle, because we're always on
      // one side of the equation.
      for val in qubit.tangles.values().collect::<Vec<_>>() {
        with_mutable!(val.left.tangles.remove(&val.right.index));
        if val.left.tangles.is_empty() {
          with_mutable_self!(self.qubits.remove(&val.left.index));
          with_mutable_self!(self.solver.clusters.remove(&val.left.index));
        }

        with_mutable!(val.right.tangles.remove(&val.left.index));
        if val.right.tangles.is_empty() {
          with_mutable_self!(self.qubits.remove(&val.right.index));
          with_mutable_self!(self.solver.clusters.remove(&val.right.index));
        }
      }
    }
  }

  /// Removes this qubit from the cluster if it's unentangled.
  pub fn remove_if_unentangled(&self, index: &i64) {
    if let Some(qubit) = self.qubits.get(&index) {
      if qubit.tangles.is_empty() {
        if self.is_tracing() {
          log!(
            Level::Info,
            "Qubit {} has been unentangled, now removing.",
            index
          )
        }

        with_mutable_self!(self.qubits.remove(index));
        with_mutable_self!(self.solver.clusters.remove(index));
      }
    }
  }

  /// Entangles these two qubits if they exist. Does not entangle if not.
  pub fn entangle(&self, left: &i64, right: &i64) {
    if let Some(rqubit) = self.qubits.get(right) {
      if let Some(lqubit) = self.qubits.get(left) {
        rqubit.entangle(lqubit);
      }
    }
  }

  pub fn contains(&self, qubit: &i64) -> bool { self.qubits.contains_key(qubit) }

  pub fn measure(&self, index: &i64) -> MeasureAnalysis {
    let qubit = with_mutable_self!(self.qubits.get(&index).expect(&format!(
      "Measure performed on qubit {index} not in the cluster: {}",
      self
    )));
    qubit.measure()
  }

  pub fn measure_then_snap(&self, index: &i64) -> MeasureAnalysis {
    let qubit = with_mutable_self!(self.qubits.get(&index).expect(&format!(
      "Measure performed on qubit {index} not in the cluster: {}",
      self
    )));
    let tangles = qubit.tangles.keys().collect::<Vec<_>>();
    let results = qubit.measure();

    // If the measure made our further qubits unentangle themselves, remove.
    self.remove(index);
    for tangle_index in tangles {
      self.remove_if_unentangled(tangle_index);
    }

    results
  }

  pub fn X(&self, qubit: &i64, radians: &f64) {
    self
      .qubits
      .get(qubit)
      .expect(&format!(
        "Attempted X on qubit {qubit} which doesn't exist in cluster: {}",
        self
      ))
      .X(radians);
  }

  pub fn Y(&self, qubit: &i64, radians: &f64) {
    self
      .qubits
      .get(qubit)
      .expect(&format!(
        "Attempted Y on qubit {qubit} which doesn't exist in cluster: {}",
        self
      ))
      .Y(radians);
  }

  pub fn Z(&self, qubit: &i64, radians: &f64) {
    self
      .qubits
      .get(qubit)
      .expect(&format!(
        "Attempted Z on qubit {qubit} which doesn't exist in cluster: {}",
        self
      ))
      .Z(radians);
  }

  pub fn CX(&self, control: &i64, target: &i64, radians: &f64) {
    let qubit = self.qubits.get(target).expect(&format!(
      "Attempted CX on qubit {target} which doesn't exist in cluster: {}",
      self
    ));
    qubit.CX(control, radians);

    self.remove_if_unentangled(target);
    self.remove_if_unentangled(control);
  }

  pub fn CZ(&self, control: &i64, target: &i64, radians: &f64) {
    let qubit = self.qubits.get(target).expect(&format!(
      "Attempted CZ on qubit {target} which doesn't exist in cluster: {}",
      self
    ));
    qubit.CZ(control, radians);

    self.remove_if_unentangled(target);
    self.remove_if_unentangled(control);
  }

  pub fn CY(&self, control: &i64, target: &i64, radians: &f64) {
    let qubit = self.qubits.get(target).expect(&format!(
      "Attempted CY on qubit {target} which doesn't exist in cluster: {}",
      self
    ));

    qubit.CY(control, radians);
    self.remove_if_unentangled(target);
    self.remove_if_unentangled(control);
  }

  /// Swaps the two qubits in the clusters internal structures, dosen't change anything else.
  pub fn SWAP(&mut self, left: &i64, right: &i64) {
    // The actual swap is taken care of at a higher level, we just need to re-associate the
    // qubit indexes when they come in here.
    let left_qubit = self.qubits.remove(left);
    let right_qubit = self.qubits.remove(right);

    if let Some(qb) = left_qubit {
      self.qubits.insert(*right, qb);
    }

    if let Some(qb) = right_qubit {
      self.qubits.insert(*left, qb);
    }
  }
}

impl Display for EntanglementCluster {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    if self.qubits.is_empty() {
      f.write_str("Empty.")
    } else {
      f.write_str("((\n");

      let mut sorted_qubits = self.qubits.values().collect::<Vec<_>>();

      sorted_qubits.sort_by_key(|val| val.index);
      f.write_str(
        &sorted_qubits
          .iter()
          .map(|val| val.stringify(1).join(""))
          .collect::<Vec<_>>()
          .join("")
      );
      f.write_str(")),\n")
    }
  }
}

type GateFragment = MatrixFragment;

/// Matrix which can be applied to a state fragment.
#[derive(Clone)]
pub struct MatrixFragment {
  matrix: Array2<Complex<f64>>,
  affected_qubits: i32
}

impl MatrixFragment {
  pub fn new(matrix: Array2<Complex<f64>>) -> MatrixFragment {
    let affected_qubits = if matrix.len() == 4 { 1 } else { 2 };

    MatrixFragment {
      matrix,
      affected_qubits
    }
  }

  #[rustfmt::skip]
  pub fn id() -> MatrixFragment {
    MatrixFragment::new(
      array![
        [C!(1.0, 0.), C!(0.0, 0.)],
        [C!(0.0, 0.), C!(1.0, 0.)]
      ])
  }

  pub fn get(&self, key: (usize, usize)) -> &Complex<f64> { self.matrix.get(key).unwrap() }

  pub fn tensor(&self, other: &MatrixFragment) -> MatrixFragment {
    // We don't expand past 2 qubits.
    if self.affected_qubits == 2 {
      panic!("Attempted to tensor a 4x4 matrix.")
    }

    let one = self.get((0, 0));
    let two = self.get((0, 1));
    let three = self.get((1, 0));
    let four = self.get((1, 1));

    let other_one = other.get((0, 0));
    let other_two = other.get((0, 1));
    let other_three = other.get((1, 0));
    let other_four = other.get((1, 1));

    Self::new(array![
      [
        one * other_one,
        one * other_two,
        two * other_one,
        two * other_two
      ],
      [
        one * other_three,
        one * other_four,
        two * other_three,
        two * other_four
      ],
      [
        three * other_one,
        three * other_two,
        four * other_one,
        four * other_two
      ],
      [
        three * other_three,
        three * other_four,
        four * other_three,
        four * other_four
      ],
    ])
  }

  pub fn transpose_conjugate(&self) -> MatrixFragment {
    // TODO: Cache and re-use familiar transformations.
    if self.affected_qubits == 1 {
      Self::new(array![
        [self.get((0, 0)).conj(), self.get((1, 0)).conj()],
        [self.get((0, 1)).conj(), self.get((1, 1)).conj()],
      ])
    } else if self.affected_qubits == 2 {
      Self::new(array![
        [
          self.get((0, 0)).conj(),
          self.get((1, 0)).conj(),
          self.get((2, 0)).conj(),
          self.get((3, 0)).conj()
        ],
        [
          self.get((0, 1)).conj(),
          self.get((1, 1)).conj(),
          self.get((2, 1)).conj(),
          self.get((3, 1)).conj()
        ],
        [
          self.get((0, 2)).conj(),
          self.get((1, 2)).conj(),
          self.get((2, 2)).conj(),
          self.get((3, 2)).conj()
        ],
        [
          self.get((0, 3)).conj(),
          self.get((1, 3)).conj(),
          self.get((2, 3)).conj(),
          self.get((3, 3)).conj()
        ],
      ])
    } else {
      panic!(
        "Can't transpose a matrix covering {} qubits.",
        self.affected_qubits
      );
    }
  }

  /// Flips the matrix reversing the value or which qubit the operation gets applied too.
  /// TODO: Check the latter.
  pub fn invert(&self) -> MatrixFragment {
    // TODO: Cache and re-use familiar transformations.
    if self.affected_qubits == 1 {
      Self::new(array![[*self.get((1, 1)), *self.get((1, 0))], [
        *self.get((0, 1)),
        *self.get((0, 0))
      ],])
    } else if self.affected_qubits == 2 {
      Self::new(array![
        [
          *self.get((3, 3)),
          *self.get((2, 3)),
          *self.get((1, 3)),
          *self.get((0, 3))
        ],
        [
          *self.get((3, 2)),
          *self.get((2, 2)),
          *self.get((1, 2)),
          *self.get((0, 2))
        ],
        [
          *self.get((3, 1)),
          *self.get((2, 1)),
          *self.get((1, 1)),
          *self.get((0, 1))
        ],
        [
          *self.get((3, 0)),
          *self.get((2, 0)),
          *self.get((1, 0)),
          *self.get((0, 0))
        ],
      ])
    } else {
      panic!(
        "Can't transpose a matrix covering {} qubits.",
        self.affected_qubits
      );
    }
  }

  #[rustfmt::skip]
  pub fn X(radians: &f64) -> MatrixFragment {
    if radians == &PI {
      MatrixFragment::new(
        array![
          [C!(0.0, 0.), C!(1.0, 0.)],
          [C!(1.0, 0.), C!(0.0, 0.)]
        ])
    } else {
      let radians_halved = radians/2.;
      MatrixFragment::new(array![
        [C!(radians_halved.cos(), 0.), C!(0.0, -radians_halved.sin())],
        [C!(0.0, -radians_halved.sin()), C!(radians_halved.cos(), 0.)]
      ])
    }
  }

  #[rustfmt::skip]
  pub fn Y(radians: &f64) -> MatrixFragment {
    if radians == &PI {
      MatrixFragment::new(array![
          [C!(0.0, 0.), C!(-1.0_f64.sqrt(), 0.)],
          [C!(1.0_f64.sqrt(), 0.), C!(0.0, 0.)]
        ])
    } else {
      let radians_halved = radians/2.;
      MatrixFragment::new(array![
          [C!(radians_halved.cos(), 0.), C!(-radians_halved.sin(), 0.)],
          [C!(radians_halved.sin(), 0.), C!(radians_halved.cos(), 0.)]
        ])
    }
  }

  #[rustfmt::skip]
  pub fn Z(radians: &f64) -> MatrixFragment {
    if radians == &PI {
      MatrixFragment::new(array![
          [C!(1.0, 0.), C!(0.0, 0.)],
          [C!(0.0, 0.), C!(-1.0, 0.)]
        ])
    } else {
      let radians_halved = radians/2.;
      MatrixFragment::new(array![
          [f64::E().powc(C!(0., -radians_halved)), C!(0., 0.)],
          [C!(0., 0.), f64::E().powc(C!(0., radians_halved))]
        ])
    }
  }

  #[rustfmt::skip]
  pub fn Had() -> MatrixFragment {
    let one_sq2 = C!(1. / 2.0f64.sqrt(), 0.);
    MatrixFragment::new(array![
        [one_sq2 * C!(1.0, 0.), one_sq2 * C!(1.0, 0.)],
        [one_sq2 * C!(1.0, 0.), one_sq2 * C!(-1.0, 0.)]
      ])
  }

  // TODO: Fix all controlled rotations to allow variable rotations.

  #[rustfmt::skip]
  pub fn CX(radians: &f64) -> MatrixFragment {
    MatrixFragment::new(array![
        [C!(1.0, 0.), C!(0.0, 0.), C!(0.0, 0.), C!(0.0, 0.)],
        [C!(0.0, 0.), C!(1.0, 0.), C!(0.0, 0.), C!(0.0, 0.)],
        [C!(0.0, 0.), C!(0.0, 0.), C!(0.0, 0.), C!(1., 0.)],
        [C!(0.0, 0.), C!(0.0, 0.), C!(1., 0.), C!(0.0, 0.)]
      ])
  }

  #[rustfmt::skip]
  pub fn CZ(radians: &f64) -> MatrixFragment {
    MatrixFragment::new(array![
        [C!(1.0, 0.), C!(0.0, 0.), C!(0.0, 0.), C!(0.0, 0.)],
        [C!(0.0, 0.), C!(1.0, 0.), C!(0.0, 0.), C!(0.0, 0.)],
        [C!(0.0, 0.), C!(0.0, 0.), C!(1., 0.), C!(0.0, 0.)],
        [C!(0.0, 0.), C!(0.0, 0.), C!(0.0, 0.), C!(-1., 0.)]
      ])
  }

  #[rustfmt::skip]
  pub fn CY(radians: &f64) -> MatrixFragment {
    MatrixFragment::new(array![
        [C!(1.0, 0.), C!(0.0, 0.), C!(0.0, 0.), C!(0.0, 0.)],
        [C!(0.0, 0.), C!(1.0, 0.), C!(0.0, 0.), C!(0.0, 0.)],
        [C!(0.0, 0.), C!(0.0, 0.), C!(0.0, 0.), C!(-1., 0.)],
        [C!(0.0, 0.), C!(0.0, 0.), C!(1., 0.), C!(0.0, 0.)]
      ])
  }

  #[rustfmt::skip]
  pub fn SWAP() -> MatrixFragment {
    MatrixFragment::new(array![
        [C!(1.0, 0.), C!(0.0, 0.), C!(0.0, 0.), C!(0.0, 0.)],
        [C!(0.0, 0.), C!(0.0, 0.), C!(1.0, 0.), C!(0.0, 0.)],
        [C!(0.0, 0.), C!(1.0, 0.), C!(0.0, 0.), C!(0.0, 0.)],
        [C!(0.0, 0.), C!(0.0, 0.), C!(0.0, 0.), C!(1.0, 0.)]
      ])
  }

  /// Returns this matrix in a nicely-formatted way for human readability and logging.
  fn stringify_matrix(&self) -> Vec<String> {
    let mut result = Vec::new();
    let matrix = &self.matrix;
    let dimensions = matrix.dim();

    fn strip(string: &String) -> String {
      string
        .replace(".00", "")
        .replace("-0+0i", "0")
        .replace("0+0i", "0")
    }

    // Restrict precision, but trim off any which are fully zero to make the output less verbose.
    if dimensions == (2, 2) {
      let mut first_row = vec![
        strip(&format!("{:.2}", matrix.get((0, 0)).unwrap())),
        strip(&format!("{:.2}", matrix.get((1, 0)).unwrap())),
      ];

      let max_length = first_row.iter().map(|val| val.len()).max().unwrap();
      first_row = first_row
        .iter()
        .map(|val| format!("{val: >width$}", width = max_length + 1 - val.len()))
        .collect::<Vec<_>>();

      let mut second_row = vec![
        strip(&format!("{:.2}", matrix.get((0, 1)).unwrap())),
        strip(&format!("{:.2}", matrix.get((1, 1)).unwrap())),
      ];

      let max_length = second_row.iter().map(|val| val.len()).max().unwrap();
      second_row = second_row
        .iter()
        .map(|val| format!("{val: >width$}", width = max_length + 1 - val.len()))
        .collect::<Vec<_>>();

      result.push(format!(
        "[{}, {}]",
        first_row.get(0).unwrap(),
        second_row.get(0).unwrap()
      ));

      result.push(format!(
        "[{}, {}]",
        first_row.get(1).unwrap(),
        second_row.get(1).unwrap()
      ));
    } else if dimensions == (4, 4) {
      let mut first_row = vec![
        strip(&format!("{:.2}", matrix.get((0, 0)).unwrap())),
        strip(&format!("{:.2}", matrix.get((1, 0)).unwrap())),
        strip(&format!("{:.2}", matrix.get((2, 0)).unwrap())),
        strip(&format!("{:.2}", matrix.get((3, 0)).unwrap())),
      ];

      let max_length = first_row.iter().map(|val| val.len()).max().unwrap();
      first_row = first_row
        .iter()
        .map(|val| format!("{: >width$}{val}", "", width = max_length - val.len()))
        .collect::<Vec<_>>();

      let mut second_row = vec![
        strip(&format!("{:.2}", matrix.get((0, 1)).unwrap())),
        strip(&format!("{:.2}", matrix.get((1, 1)).unwrap())),
        strip(&format!("{:.2}", matrix.get((2, 1)).unwrap())),
        strip(&format!("{:.2}", matrix.get((3, 1)).unwrap())),
      ];

      let max_length = second_row.iter().map(|val| val.len()).max().unwrap();
      second_row = second_row
        .iter()
        .map(|val| format!("{: >width$}{val}", "", width = max_length - val.len()))
        .collect::<Vec<_>>();

      let mut third_row = vec![
        strip(&format!("{:.2}", matrix.get((0, 2)).unwrap())),
        strip(&format!("{:.2}", matrix.get((1, 2)).unwrap())),
        strip(&format!("{:.2}", matrix.get((2, 2)).unwrap())),
        strip(&format!("{:.2}", matrix.get((3, 2)).unwrap())),
      ];

      let max_length = third_row.iter().map(|val| val.len()).max().unwrap();
      third_row = third_row
        .iter()
        .map(|val| format!("{: >width$}{val}", "", width = max_length - val.len()))
        .collect::<Vec<_>>();

      let mut fourth_row = vec![
        strip(&format!("{:.2}", matrix.get((0, 3)).unwrap())),
        strip(&format!("{:.2}", matrix.get((1, 3)).unwrap())),
        strip(&format!("{:.2}", matrix.get((2, 3)).unwrap())),
        strip(&format!("{:.2}", matrix.get((3, 3)).unwrap())),
      ];

      let max_length = fourth_row.iter().map(|val| val.len()).max().unwrap();
      fourth_row = fourth_row
        .iter()
        .map(|val| format!("{: >width$}{val}", "", width = max_length - val.len()))
        .collect::<Vec<_>>();

      result.push(format!(
        "[{}, {}, {}, {}]",
        first_row.get(0).unwrap(),
        second_row.get(0).unwrap(),
        third_row.get(0).unwrap(),
        fourth_row.get(0).unwrap()
      ));

      result.push(format!(
        "[{}, {}, {}, {}]",
        first_row.get(1).unwrap(),
        second_row.get(1).unwrap(),
        third_row.get(1).unwrap(),
        fourth_row.get(1).unwrap()
      ));

      result.push(format!(
        "[{}, {}, {}, {}]",
        first_row.get(2).unwrap(),
        second_row.get(2).unwrap(),
        third_row.get(2).unwrap(),
        fourth_row.get(2).unwrap()
      ));

      result.push(format!(
        "[{}, {}, {}, {}]",
        first_row.get(3).unwrap(),
        second_row.get(3).unwrap(),
        third_row.get(3).unwrap(),
        fourth_row.get(3).unwrap()
      ));
    } else {
      panic!("Attempted to print matrix of irregular dimensions.")
    }

    // Reduce verbosity of the output where we don't need to know about it.
    result
  }
}

impl Display for MatrixFragment {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    f.write_str(&self.stringify_matrix().join("\n"))
  }
}

/// Multiply both matrix fragments together. Dimensions are expected to be equal.
fn multiply(left: &MatrixFragment, right: &MatrixFragment) -> MatrixFragment {
  let mult =
    |left_index, right_index| -> Complex64 { left.get(left_index) * right.get(right_index) };

  // TODO: Find specialized MM library or algorithm, current one not seemingly working correctly.
  //  This is just a brute-force implementation for testing.
  if left.affected_qubits == 1 {
    MatrixFragment::new(array![
      [
        mult((0, 0), (0, 0)) + mult((0, 1), (1, 0)),
        mult((0, 0), (1, 0)) + mult((0, 1), (1, 1))
      ],
      [
        mult((1, 0), (0, 0)) + mult((1, 1), (1, 0)),
        mult((1, 0), (1, 0)) + mult((1, 1), (1, 1))
      ]
    ])
  } else if left.affected_qubits == 2 {
    MatrixFragment::new(array![
      [
        mult((0, 0), (0, 0)) + mult((0, 1), (1, 0)) + mult((0, 2), (2, 0)) + mult((0, 3), (3, 0)),
        mult((0, 0), (0, 1)) + mult((0, 1), (1, 1)) + mult((0, 2), (2, 1)) + mult((0, 3), (3, 1)),
        mult((0, 0), (0, 2)) + mult((0, 1), (1, 2)) + mult((0, 2), (2, 2)) + mult((0, 3), (3, 2)),
        mult((0, 0), (0, 3)) + mult((0, 1), (1, 3)) + mult((0, 2), (2, 3)) + mult((0, 3), (3, 3))
      ],
      [
        mult((1, 0), (0, 0)) + mult((1, 1), (1, 0)) + mult((1, 2), (2, 0)) + mult((1, 3), (3, 0)),
        mult((1, 0), (0, 1)) + mult((1, 1), (1, 1)) + mult((1, 2), (2, 1)) + mult((1, 3), (3, 1)),
        mult((1, 0), (0, 2)) + mult((1, 1), (1, 2)) + mult((1, 2), (2, 2)) + mult((1, 3), (3, 2)),
        mult((1, 0), (0, 3)) + mult((1, 1), (1, 3)) + mult((1, 2), (2, 3)) + mult((1, 3), (3, 3))
      ],
      [
        mult((2, 0), (0, 0)) + mult((2, 1), (1, 0)) + mult((2, 2), (2, 0)) + mult((2, 3), (3, 0)),
        mult((2, 0), (0, 1)) + mult((2, 1), (1, 1)) + mult((2, 2), (2, 1)) + mult((2, 3), (3, 1)),
        mult((2, 0), (0, 2)) + mult((2, 1), (1, 2)) + mult((2, 2), (2, 2)) + mult((2, 3), (3, 2)),
        mult((2, 0), (0, 3)) + mult((2, 1), (1, 3)) + mult((2, 2), (2, 3)) + mult((2, 3), (3, 3))
      ],
      [
        mult((3, 0), (0, 0)) + mult((3, 1), (1, 0)) + mult((3, 2), (2, 0)) + mult((3, 3), (3, 0)),
        mult((3, 0), (0, 1)) + mult((3, 1), (1, 1)) + mult((3, 2), (2, 1)) + mult((3, 3), (3, 1)),
        mult((3, 0), (0, 2)) + mult((3, 1), (1, 2)) + mult((3, 2), (2, 2)) + mult((3, 3), (3, 2)),
        mult((3, 0), (0, 3)) + mult((3, 1), (1, 3)) + mult((3, 2), (2, 3)) + mult((3, 3), (3, 3))
      ],
    ])
  } else {
    panic!("Attempted multiplication on irregular size of matrix.")
  }
}

impl Mul for MatrixFragment {
  type Output = MatrixFragment;

  fn mul(self, rhs: Self) -> Self::Output { multiply(&self, &rhs) }
}

impl Mul for &MatrixFragment {
  type Output = MatrixFragment;

  fn mul(self, rhs: Self) -> Self::Output { multiply(&self, &rhs) }
}

impl Mul for &mut MatrixFragment {
  type Output = MatrixFragment;

  fn mul(self, rhs: Self) -> Self::Output { MatrixFragment::new(&self.matrix * &rhs.matrix) }
}

// While there is no distinction it's better to define what the types mean, even if the generic
// structures are the same.
type QubitFragment = StateFragment;
type EntangledFragment = StateFragment;

/// Composite enum for matrix operations to be able to automatically expand when used against
/// smaller ones.
#[derive(Clone)]
pub struct StateFragment {
  matrix_fragment: MatrixFragment
}

impl StateFragment {
  pub fn new(matrix_fragment: MatrixFragment) -> StateFragment { StateFragment { matrix_fragment } }

  #[rustfmt::skip]
  pub fn DefaultQubit() -> QubitFragment {
    StateFragment {
      matrix_fragment:
        MatrixFragment::new(array![
          [C!(1.0, 0.0), C!(0.0, 0.0)],
          [C!(0.0, 0.0), C!(0.0, 0.0)]
        ])
    }
  }

  /// Creates a state fragment to represent entanglement between 2 qubits. Needs setup with
  /// initial qubit values.
  #[rustfmt::skip]
  pub fn DefaultEntangled() -> EntangledFragment {
    StateFragment {
      matrix_fragment:
        MatrixFragment::new(array![
        [C!(1.0, 0.0), C!(0.0, 0.0), C!(0.0, 0.0), C!(0.0, 0.0)],
        [C!(0.0, 0.0), C!(0.0, 0.0), C!(0.0, 0.0), C!(0.0, 0.0)],
        [C!(0.0, 0.0), C!(0.0, 0.0), C!(0.0, 0.0), C!(0.0, 0.0)],
        [C!(0.0, 0.0), C!(0.0, 0.0), C!(0.0, 0.0), C!(0.0, 0.0)]
      ])
    }
  }

  pub fn get(&self, key: (usize, usize)) -> &Complex<f64> { self.matrix_fragment.get(key) }

  pub fn represented_qubits(&self) -> i32 { self.matrix_fragment.affected_qubits }

  pub fn entangle_with(&self, right: &Ptr<QubitFragment>) -> EntangledFragment {
    EntangledFragment::new(self.matrix_fragment.tensor(&right.matrix_fragment))
  }

  pub fn apply(&mut self, gate: &MatrixFragment) -> Option<String> {
    if self.represented_qubits() != gate.affected_qubits {
      return Some(String::from("Can't apply to fragments of differing sizes."));
    }

    let mut result = gate * &self.matrix_fragment;
    self.matrix_fragment = result * gate.transpose_conjugate();
    None
  }

  pub fn stringify_matrix(&self) -> Vec<String> { self.matrix_fragment.stringify_matrix() }
}

impl Display for StateFragment {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    f.write_str(&self.matrix_fragment.to_string())
  }
}

impl Mul for StateFragment {
  type Output = StateFragment;

  fn mul(self, rhs: Self) -> Self::Output {
    StateFragment::new(self.matrix_fragment * rhs.matrix_fragment)
  }
}

#[derive(Clone)]
pub struct SolverConfig {
  pub active: bool
}

impl SolverConfig {
  pub fn new(active: bool) -> SolverConfig { SolverConfig { active } }

  pub fn off() -> SolverConfig { SolverConfig::new(false) }

  pub fn on() -> SolverConfig { SolverConfig::new(true) }

  pub fn with_config(config: &Ptr<RasqalConfig>) -> SolverConfig {
    SolverConfig::new(config.solver_active)
  }
}

#[derive(Clone)]
pub struct SolverResult {
  pub bitstring: String,
  pub probability: f64
}

impl SolverResult {
  pub fn from_result_fragment(fragment: &ResultFragment) -> SolverResult {
    SolverResult {
      bitstring: fragment.as_bitstring(),
      probability: fragment.rolling_probability
    }
  }
}

impl Display for SolverResult {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    f.write_str(&format!(
      "{} @ {:.2}%",
      self.bitstring,
      self.probability * 100.
    ))
  }
}

#[derive(Clone)]
pub struct ResultFragment {
  /// Rolling probability of this whole fragment being applicable. Used for filtering.
  rolling_probability: f64,
  fragment: HashMap<i64, i16>,

  /// Max number of qubits that are actually measured. Indexes do not need to be sequential, so
  /// you can measure qubits 5, 75, 240 and 700 and this will simply be 4. Used to pad out
  /// unknowable  values if a qubit is early in the analysis chain.
  measureable_qubits: Ptr<HashSet<i64>>
}

impl ResultFragment {
  pub fn new(
    index: i64, result: i16, probability: f64, measureable_qubits: Ptr<HashSet<i64>>
  ) -> ResultFragment {
    let mut fragment = ResultFragment {
      fragment: HashMap::default(),
      rolling_probability: probability,
      measureable_qubits
    };
    fragment.fragment.insert(index, result);
    fragment
  }

  /// Flips the bitstring results and reverses the probability.
  /// So 11 @ 30% becomes 00 @ 70%. Used to mirror initial results so we have both sides of the
  /// binary calculation.
  pub fn with_flipped(result: &ResultFragment) -> ResultFragment {
    let mut flipped_fragments = HashMap::default();
    for (key, value) in result.fragment.iter() {
      let flipped = if *value == 0 { 1 } else { 0 };
      flipped_fragments.insert(*key, flipped);
    }

    ResultFragment {
      rolling_probability: 1.0 - result.rolling_probability,
      fragment: flipped_fragments,
      measureable_qubits: result.measureable_qubits.clone()
    }
  }

  pub fn add(&mut self, qubit: i64, result: i16, probability: f64) {
    self.rolling_probability = self.rolling_probability * probability;
    self.fragment.insert(qubit, result);
  }

  /// If the passed-in fragment can be overlaid this one it then is. Note that this changes the
  /// fragment itself.
  ///
  /// For an example:
  ///
  /// XX1X10 & 0XXX10 = 0X1X10
  /// 100XXX & 111XXX = N/A
  pub fn overlay(&mut self, other: &ResultFragment) {
    // TODO: In time re-evaluate if we want to allow partial overlays such as X001 & XX01.
    //  Makes probabilities very complicated to calculate, but would give potentially greater
    //  precision.
    //  The argument against this though is that overlays should cover the same entanglement
    //  clusters, which should cover the same qubits so the above would never happen unless you
    //  take a previously overlaid fragment. We'll need to ascertain if that is correct.

    // For now, only overlay if there is no collisions.
    let mut insertions = Vec::new();
    for (key, value) in other.fragment.iter() {
      if self.fragment.contains_key(key) {
        return;
      } else {
        insertions.push((key, value));
      }
    }

    for (key, value) in insertions {
      self.fragment.insert(*key, *value);
    }

    self.rolling_probability = self.rolling_probability * other.rolling_probability
  }

  /// Fills out all unmeasured qubits in the bitstring with zeros, up to `register_count`.
  pub fn fill_empty(&mut self, register_count: i64) {
    for i in 0..=register_count {
      if !self.measureable_qubits.contains(&i) {
        self.fragment.insert(i, 0);
      }
    }
  }

  /// Generates a human-readable bitstring from this fragment. Replaces all unknown bits with X.
  pub fn as_bitstring(&self) -> String {
    let mut result = String::new();
    for i in self.measureable_qubits.iter() {
      if let Some(value) = self.fragment.get(&i) {
        result.push_str(&value.to_string())
      } else {
        result.push('X')
      }
    }
    result
  }

  /// Is this fragment actually fully resolved with results in every slot?
  pub fn is_solved(&self) -> bool { self.fragment.len() >= self.measureable_qubits.len() }
}

impl Display for ResultFragment {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    f.write_str(&format!(
      "{} @ {:.2}%",
      self.as_bitstring(),
      self.rolling_probability * 100.
    ))
  }
}

pub struct ResultsSynthsizer {
  /// The range a probability must be within the highest to not be dropped from template creation.
  /// So if our highest probability is 55%, and this value is 20, it will drop any combinations
  /// whose probability is under %35.
  probability_range: f64,

  /// In the case where a large amount of entanglements all come out at around the same
  /// probability the max amount we should add in total. The dropped ones will be the last
  /// added, which will be the longest result chains.
  max_entangles_per_add: usize,
  fragments: Vec<ResultFragment>,

  /// Set of measured qubit indexes so we know which indexes are used, or not, and to default all
  /// unmeasured to zero after we've synthesized a result.
  measured_qubits: Ptr<HashSet<i64>>,

  register_size: i64
}

impl ResultsSynthsizer {
  pub fn new(
    probability_range: f64, max_entangles_per_add: usize, measured_qubits: HashSet<i64>,
    register_size: i64
  ) -> ResultsSynthsizer {
    ResultsSynthsizer {
      probability_range,
      max_entangles_per_add,
      fragments: Vec::default(),
      measured_qubits: Ptr::from(measured_qubits),
      register_size
    }
  }

  pub fn add(&mut self, measure: &MeasureAnalysis) {
    // Build starter fragment with just our qubit.
    let qubit_result = 1;
    let mut starter = ResultFragment::new(
      measure.qubit,
      qubit_result,
      measure.probability,
      self.measured_qubits.clone()
    );

    // Any full entanglement gets appended to the default fragment as there is no chance they
    // will deviate. This can drastically reduce complexity depending upon how reliant the
    // algorithm is on fully entangled values for their results.
    let mut removals = HashSet::new();

    // Sort our entanglements by coupling strength.
    let mut entanglements = measure.entangled_with.iter().collect::<Vec<_>>();
    entanglements.sort_by(|a, b| b.ratio.total_cmp(&a.ratio));

    for ent_meta in entanglements.iter() {
      // Since we're sorted on this, soon as we see a non-one hundred value we know there
      // will be no others. Ratio is 0-100.
      if !is_near!(ent_meta.ratio, 1.0) {
        break;
      }

      starter.add(ent_meta.qubit, qubit_result, ent_meta.ratio);
      removals.insert(ent_meta.qubit);
    }

    let number_of_entanglements = entanglements.len();
    let mut results = Vec::new();
    results.push(ResultFragment::with_flipped(&starter));
    results.push(starter);

    // Any entangled values will at best be equal, not lower.
    let highest_probability = results
      .iter()
      .map(|val| val.rolling_probability)
      .reduce(f64::max)
      .unwrap();
    let lowest_bound = highest_probability - self.probability_range;

    for ent_meta in entanglements.iter() {
      if results.len() >= self.max_entangles_per_add {
        break;
      }

      // Note: This will scale better than copy / remove with large lists but we need to
      //  make sure qubits are unique. Otherwise, a composite identifier will be needed.
      if removals.contains(&ent_meta.qubit) {
        continue;
      }

      // Merge all templates starting with the highest probability to happen, breaking out
      // of the loop when we've breached our constraints.
      let mut temp = Vec::new();
      for fragment in results.iter() {
        if (fragment.rolling_probability * ent_meta.ratio) < lowest_bound {
          continue;
        }

        let mut new_fragment = fragment.clone();

        // With entangled values they are all the same, whether 1 or 0, so we just take the first
        // value and use that as our result.
        let universal_result = fragment.fragment.values().take(1).last().unwrap();
        new_fragment.add(
          ent_meta.qubit,
          *universal_result,
          fragment.rolling_probability
        );
        temp.push(new_fragment);

        if temp.len() + results.len() >= self.max_entangles_per_add {
          break;
        }
      }

      results.append(&mut temp);
    }

    self.fragments.append(&mut results);
  }

  /// Combine all the fragments together into usable bitstrings. Resolves whether we know the
  /// value at a certain qubit point, or not.
  pub fn synthesize(&self) -> Vec<SolverResult> {
    let mut results = Vec::new();

    // Evaluate the lower bound now we have every current potential result.
    let lower_bound = self
      .fragments
      .iter()
      .map(|val| val.rolling_probability)
      .reduce(f64::max)
      .unwrap()
      - self.probability_range;

    // Filter out duplicated values from results.
    // TODO: We need a more efficient way to filter out duplicates, preferably finding them earlier.
    //  But for now this should be OK. Duplicates should only come from 100% entangled qubits and
    //  aren't affected by boundary filters at the point they are generated.
    //
    // TODO: At best we'll just have fully_entangled_qubits * max_entanglement duplicates in a state
    //  which is fully entangled with each other. While not optimal, size of circuits right now are
    //  not large enough for it to make a significant difference (probably), and certainly not by
    //  comparison to other bottlenecks.
    let mut duplicated = HashSet::new();

    // Filter out the ones which don't breach the boundary.
    let mut filtered_fragments = self
      .fragments
      .iter()
      .filter(|val| val.rolling_probability >= lower_bound)
      .collect::<Vec<_>>();
    for fragment in filtered_fragments.iter() {
      let mut composite = (*fragment).clone();
      for overlay_fragment in filtered_fragments.iter() {
        if std::ptr::eq(fragment, overlay_fragment) {
          continue;
        }

        // Break if we get to a point where this fragment will drop out of prediction range.
        if fragment.rolling_probability * overlay_fragment.rolling_probability < lower_bound {
          break;
        }

        composite.overlay(overlay_fragment);
      }

      let composite_key = composite.to_string();
      if composite.rolling_probability >= lower_bound && !duplicated.contains(&composite_key) {
        composite.fill_empty(self.register_size);
        results.push(SolverResult::from_result_fragment(&composite));
        duplicated.insert(composite_key);
      }
    }

    // Sort by probability since list shouldn't be large at this point, and it's what we'll want.
    results.sort_by(|left, right| right.probability.total_cmp(&left.probability));
    results
  }
}

impl Display for ResultsSynthsizer {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    f.write_str(&format!(
      "\n{}",
      &self
        .fragments
        .iter()
        .map(|val| val.to_string())
        .collect::<Vec<_>>()
        .join("\n")
    ))
  }
}

/// Acts as a pseudo-state that allows for partial circuit solving and value introspection.
pub struct QuantumSolver {
  qubits: Ptr<HashMap<i64, Ptr<AnalysisQubit>>>,
  clusters: Ptr<HashMap<i64, Ptr<EntanglementCluster>>>,
  measures: Ptr<HashMap<i64, MeasureAnalysis>>,
  trace_module: Ptr<TracingModule>,
  probability_range: f64,
  max_entanglements: usize
}

impl QuantumSolver {
  pub fn new() -> QuantumSolver {
    QuantumSolver {
      qubits: Ptr::from(HashMap::default()),
      clusters: Ptr::from(HashMap::default()),
      measures: Ptr::from(HashMap::default()),
      trace_module: Ptr::from(TracingModule::default()),
      probability_range: 0.25,
      max_entanglements: 20
    }
  }

  fn is_tracing(&self) -> bool { self.trace_module.has(ActiveTracers::Solver) }

  pub fn with_trace(trace_module: Ptr<TracingModule>) -> QuantumSolver {
    QuantumSolver {
      qubits: Ptr::from(HashMap::default()),
      measures: Ptr::from(HashMap::default()),
      clusters: Ptr::from(HashMap::default()),
      trace_module,
      probability_range: 0.25,
      max_entanglements: 20
    }
  }

  /// Gets a qubit, or adds a default one at this index if it doesn't exist.
  fn qubit_for(&self, index: &i64) -> &Ptr<AnalysisQubit> {
    if let Some(qubit) = self.qubits.get(index) {
      qubit
    } else {
      with_mutable_self!(self.qubits.insert(
        index.clone(),
        Ptr::from(AnalysisQubit::with_index(*index, self.trace_module.clone()))
      ));
      self.qubits.get(&index).unwrap()
    }
  }

  /// Gets the cluster for this index. Inserts the qubit into both solver and cluster, creating a
  /// new cluster if required. Don't use this if you only want to fetch a cluster without modifying
  /// it.
  fn cluster_for(&self, index: &i64) -> &Ptr<EntanglementCluster> {
    let cluster = if let Some(cluster) = with_mutable_self!(self.clusters.get_mut(index)) {
      cluster
    } else {
      with_mutable_self!(self.clusters.insert(
        index.clone(),
        Ptr::from(EntanglementCluster::new(
          Ptr::from(self),
          &self.trace_module
        ))
      ));
      self.clusters.get(&index).unwrap()
    };

    if !cluster.contains(index) {
      cluster.add(self.qubit_for(index));
    }

    cluster
  }

  fn merge_clusters(&self, merger: Vec<&i64>, mergee: &i64) -> &Ptr<EntanglementCluster> {
    let target_cluster = self.cluster_for(&mergee);
    for index in merger {
      if let Some(cluster) = self.clusters.get(index) {
        // If clusters are different, merge, entangle our two qubits, then replace reference.
        if !cluster.contains(&mergee) {
          target_cluster.merge(cluster);
          target_cluster.entangle(index, mergee);

          // Remove the previous cluster, it's no longer needed, replace with new merged one.
          with_mutable_self!(self.clusters.insert(index.clone(), target_cluster.clone()));
        }
      } else {
        // If we don't have a cluster, just add our qubit and assign it a cluster.
        let qubit = self.qubit_for(index);
        target_cluster.add_then_entangle(qubit, mergee);
        with_mutable_self!(self.clusters.insert(index.clone(), target_cluster.clone()));
      }
    }
    target_cluster
  }

  /// Reset this qubit to its default state, including removing all entanglement information.
  pub fn reset(&self, qb: &Qubit) {
    if self.is_tracing() {
      log!(Level::Info, "Reset[{}]", qb.index)
    }

    if let Some(cluster) = self.clusters.get(&qb.index) {
      cluster.remove(&qb.index);
      with_mutable_self!(self.clusters.remove(&qb.index));
    }

    let qubit = self.qubit_for(&qb.index);
    with_mutable!(qubit.state = Ptr::from(StateFragment::DefaultQubit()));
  }

  pub fn measure_all(&self, qbs: &Vec<&Qubit>) {
    for qb in qbs {
      self.measure(qb);
    }
  }

  pub fn measure(&self, qb: &Qubit) {
    let mut tracing_message = None;
    if self.is_tracing() {
      let addendum = if let Some(cluster) = self.clusters.get(&qb.index) {
        let clustered_with = cluster
          .spans()
          .filter(|val| *val != &qb.index)
          .map(|val| val.to_string())
          .collect::<Vec<_>>()
          .join(",");
        if !clustered_with.is_empty() {
          format!(", clustered with [{}]", clustered_with)
        } else {
          String::new()
        }
      } else {
        String::new()
      };
      tracing_message = Some(format!(
        "\nMeasuring Q{}{}:\n{}",
        qb.index,
        addendum,
        self.qubit_for(&qb.index)
      ));
    }

    let result = if let Some(cluster) = self.clusters.get(&qb.index) {
      cluster.measure(&qb.index)
    } else {
      self.qubit_for(&qb.index).measure()
    };

    if self.is_tracing() {
      log!(
        Level::Info,
        "{}Result: {}\n",
        tracing_message.unwrap(),
        result
      );
    }

    // We only record the last measure on a qubit as the valid one.
    with_mutable_self!(self.measures.insert(qb.index, result));
  }

  pub fn X(&self, qb: &Qubit, radians: &f64) {
    let mut pre = None;
    if self.is_tracing() {
      pre = Some(self.qubit_for(&qb.index).measure());
    }

    if let Some(cluster) = self.clusters.get(&qb.index) {
      cluster.X(&qb.index, radians);
    } else {
      self.qubit_for(&qb.index).X(radians)
    }

    if self.is_tracing() {
      self.trace_gate("X", qb.index.to_string(), &pre.unwrap(), radians)
    }
  }

  pub fn Y(&self, qb: &Qubit, radians: &f64) {
    let mut pre = None;
    if self.is_tracing() {
      pre = Some(self.qubit_for(&qb.index).measure());
    }

    if let Some(cluster) = self.clusters.get(&qb.index) {
      cluster.Y(&qb.index, radians);
    } else {
      self.qubit_for(&qb.index).Y(radians)
    }

    if self.is_tracing() {
      self.trace_gate("Y", qb.index.to_string(), &pre.unwrap(), radians)
    }
  }

  pub fn Z(&self, qb: &Qubit, radians: &f64) {
    let mut pre = None;
    if self.is_tracing() {
      pre = Some(self.qubit_for(&qb.index).measure());
    }

    if let Some(cluster) = self.clusters.get(&qb.index) {
      cluster.Z(&qb.index, radians);
    } else {
      self.qubit_for(&qb.index).Z(radians)
    }

    if self.is_tracing() {
      self.trace_gate("Z", qb.index.to_string(), &pre.unwrap(), radians)
    }
  }

  pub fn Had(&self, qb: &Qubit) {
    self.Z(qb, &PI);
    self.Y(qb, &(PI / 2.0))
  }

  pub fn CX(&self, controls: &Vec<Qubit>, target: &Qubit, radians: &f64) {
    let mut pre = None;
    if self.is_tracing() {
      pre = Some(self.qubit_for(&target.index).measure());
    }

    let target_cluster = self.merge_clusters(
      controls.iter().map(|val| &val.index).collect::<Vec<_>>(),
      &target.index
    );

    for qb in controls {
      // If a single iteration drops entanglement we need to re-add the qubit for analysis.
      // TODO: Rethink.
      if !target_cluster.contains(&target.index) {
        target_cluster.add(self.qubit_for(&target.index));
      }

      target_cluster.CX(&qb.index, &target.index, radians);
    }

    if self.is_tracing() {
      self.trace_gate(
        "CX",
        format!(
          "{}->{}",
          controls
            .iter()
            .map(|val| val.index.to_string())
            .collect::<Vec<_>>()
            .join(","),
          target.index
        ),
        &pre.unwrap(),
        radians
      )
    }
  }

  pub fn CY(&self, controls: &Vec<Qubit>, target: &Qubit, radians: &f64) {
    let mut pre = None;
    if self.is_tracing() {
      pre = Some(self.qubit_for(&target.index).measure());
    }

    let target_cluster = self.merge_clusters(
      controls.iter().map(|val| &val.index).collect::<Vec<_>>(),
      &target.index
    );

    for qb in controls {
      // If a single iteration drops entanglement we need to re-add the qubit for analysis.
      // TODO: Rethink.
      if !target_cluster.contains(&target.index) {
        target_cluster.add(self.qubit_for(&target.index));
      }

      target_cluster.CY(&qb.index, &target.index, radians);
    }

    if self.is_tracing() {
      self.trace_gate(
        "CY",
        format!(
          "{}->{}",
          controls
            .iter()
            .map(|val| val.index.to_string())
            .collect::<Vec<_>>()
            .join(","),
          target.index
        ),
        &pre.unwrap(),
        radians
      )
    }
  }

  pub fn CZ(&self, controls: &Vec<Qubit>, target: &Qubit, radians: &f64) {
    let mut pre = None;
    if self.is_tracing() {
      pre = Some(self.qubit_for(&target.index).measure());
    }

    let target_cluster = self.merge_clusters(
      controls.iter().map(|val| &val.index).collect::<Vec<_>>(),
      &target.index
    );

    for qb in controls {
      // If a single iteration drops entanglement we need to re-add the qubit for analysis.
      // TODO: Rethink.
      if !target_cluster.contains(&target.index) {
        target_cluster.add(self.qubit_for(&target.index));
      }

      target_cluster.CZ(&qb.index, &target.index, radians);
    }

    if self.is_tracing() {
      self.trace_gate(
        "CZ",
        format!(
          "{}->{}",
          controls
            .iter()
            .map(|val| val.index.to_string())
            .collect::<Vec<_>>()
            .join(","),
          target.index
        ),
        &pre.unwrap(),
        radians
      )
    }
  }

  pub fn SWAP(&mut self, left: &i64, right: &i64) {
    if self.qubits.contains_key(left) && self.qubits.contains_key(right) {
      if self.is_tracing() {
        log!(Level::Info, "Swapping {} and {}", left, right);
      }

      let mut qubit_one = self.qubits.remove(left).unwrap();
      let mut qubit_two = self.qubits.remove(left).unwrap();
      let first_index = qubit_one.index;
      let second_index = qubit_two.index;

      // Just merge indexes so we can deal with the point where entangled qubits reference both
      // our swapped qubits.
      let mut entanglements = qubit_one.entangled_with().collect::<HashSet<&i64>>();
      for index in qubit_two.entangled_with() {
        entanglements.insert(index);
      }

      // Go through both qubits entangled threads and swap them around.
      for index in entanglements {
        let target_qubit = self
          .qubits
          .get(index)
          .expect("Entangled qubit {index} has to exist in solver.");

        let first_tangle = with_mutable!(target_qubit.tangles.remove(&first_index));
        let second_tangle = with_mutable!(target_qubit.tangles.remove(&second_index));

        if let Some(tangle) = first_tangle {
          with_mutable!(target_qubit.tangles.insert(second_index, tangle));
        }

        if let Some(tangle) = second_tangle {
          with_mutable!(target_qubit.tangles.insert(first_index, tangle));
        }
      }

      // Then just swap the designation around.
      qubit_one.index = second_index;
      qubit_two.index = first_index;
      self.qubits.insert(qubit_one.index, qubit_one);
      self.qubits.insert(qubit_two.index, qubit_two);

      if let Some(cluster) = self.clusters.get_mut(&first_index) {
        cluster.SWAP(left, right)
      }

      if let Some(cluster) = self.clusters.get_mut(&second_index) {
        cluster.SWAP(left, right)
      }
    }
  }

  pub fn solve(&self) -> Vec<SolverResult> {
    // We don't worry about printing if we're utterly empty.
    if self.is_tracing() {
      if self.qubits.is_empty() {
        log!(Level::Info, "Nothing to solve.");
      } else {
        log!(
          Level::Info,
          "Solving with {} probability range, {} max entanglements.",
          self.probability_range,
          self.max_entanglements
        );
        log!(Level::Info, "Current state:{}", self.to_string());
      }
    }

    let start = Instant::now();
    let measurable_indexes = self
      .measures
      .keys()
      .map(|val| val.clone())
      .collect::<HashSet<i64>>();

    let mut synth = ResultsSynthsizer::new(
      self.probability_range,
      self.max_entanglements,
      measurable_indexes,
      *self.qubits.keys().max().unwrap()
    );
    for meas in self.measures.values() {
      synth.add(meas);
    }

    if self.is_tracing() {
      log!(Level::Info, "Unfiltered results: {}\n", synth);
    }

    let results = synth.synthesize();
    if self.is_tracing() {
      log!(
        Level::Info,
        "Solver results:\n{}\n",
        results
          .iter()
          .map(|val| val.to_string())
          .collect::<Vec<_>>()
          .join("\n")
      );
    }

    let took = start.elapsed();
    log!(Level::Info, "Solving took {}ms", took.as_millis());
    results
  }

  /// Tracing method for printing a simplified difference between measures before/after a
  /// gate application.
  fn trace_gate(
    &self, gate: &str, associated_qubits: String, pre: &MeasureAnalysis, radians: &f64
  ) {
    let mut post = self.qubit_for(&pre.qubit).measure();
    let mut differences = Vec::new();
    if pre.probability != post.probability {
      differences.push(format!("from {:.2}%", pre.probability * 100.))
    }

    let preq = pre
      .entangled_with
      .iter()
      .map(|val| (val.qubit, val))
      .collect::<HashMap<_, _>>();
    let postq = post
      .entangled_with
      .iter()
      .map(|val| (val.qubit, val))
      .collect::<HashMap<_, _>>();
    for qb in pre
      .entangled_with
      .iter()
      .filter(|val| !postq.contains_key(&val.qubit))
    {
      differences.push(format!("rem Q{}", qb.qubit))
    }

    for (index, ent) in postq.iter() {
      if !preq.contains_key(index) {
        differences.push(format!("add Q{}~{}", ent.qubit, ent.ratio))
      } else {
        let prev = preq.get(index).unwrap();
        if ent.ratio != prev.ratio {
          differences.push(format!("Q{}~{}", prev.qubit, prev.ratio))
        }
      }
    }

    let mut diff = String::new();
    if !differences.is_empty() {
      diff = format!(" # {}", differences.join(","));
    }

    log!(
      Level::Info,
      "{}[{}] {:.4} @ {}{}",
      gate,
      associated_qubits,
      radians,
      post,
      diff
    )
  }
}

impl Display for QuantumSolver {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    f.write_str("\nSolver:\n");

    let mut covered_qubits = HashSet::new();
    let mut qubits = self.qubits.iter().collect::<Vec<_>>();
    qubits.sort_by_key(|val| val.0);
    for (key, value) in qubits {
      if covered_qubits.contains(key) {
        continue;
      }

      if let Some(cluster) = self.clusters.get(key) {
        for index in cluster.spans() {
          covered_qubits.insert(index);
        }

        if cluster.qubits.len() > 0 {
          f.write_str("[Cluster]\n");
          f.write_str(&cluster.to_string());
        }
      } else {
        covered_qubits.insert(key);
        f.write_str("[Qubit]\n");
        f.write_str(&value.to_string());
      }
    }

    f.write_str("Measures:\n");
    let mut ordered_measures = self.measures.iter().collect::<Vec<_>>();
    ordered_measures.sort_by(|left, right| left.0.cmp(right.0));
    for (index, result) in ordered_measures.iter() {
      f.write_fmt(format_args!("{} -> {}\n", index, result));
    }

    f.write_str("")
  }
}

#[cfg(test)]
mod tests {
  use crate::analysis::solver::{GateFragment, QuantumSolver, QubitFragment};
  use crate::hardware::Qubit;
  use crate::runtime::{ActiveTracers, TracingModule};
  use crate::smart_pointers::Ptr;
  use std::borrow::Borrow;
  use std::f64::consts::PI;
  use std::fmt::Display;

  #[test]
  fn bell_test() {
    let solver = QuantumSolver::with_trace(Ptr::from(TracingModule::with(ActiveTracers::all())));
    let (q0, q1) = (Qubit::new(0), Qubit::new(1));
    solver.Had(&q0);
    solver.CX(&vec![q1.clone()], &q0, &PI);
    solver.measure(&q0);
    solver.measure(&q1);
    let result = solver.solve();

    let results = result
      .iter()
      .filter(|val| val.bitstring == "11" || val.bitstring == "00")
      .collect::<Vec<_>>();
    assert_eq!(results.len(), 2);
    assert!(results[0].probability >= 0.49 && results[0].probability <= 0.51);
    assert!(results[1].probability >= 0.49 && results[1].probability <= 0.51);
  }

  #[test]
  fn X() {
    let mut qubit = QubitFragment::DefaultQubit();
    let result = qubit.apply(&GateFragment::X(&(PI / 2.)));
    assert!(result.is_none());

    let zero = qubit.get((0, 0)).re;
    assert!(zero >= 0.48 && zero <= 0.52);

    assert_eq!(qubit.get((0, 1)).re, 0.);
    assert_eq!(qubit.get((1, 0)).re, 0.);

    let one = qubit.get((1, 1)).re;
    assert!(one >= 0.48 && one <= 0.52);
  }

  #[test]
  fn Z() {
    let mut qubit = QubitFragment::DefaultQubit();
    let result = qubit.apply(&GateFragment::Z(&(PI / 2.)));
    assert!(result.is_none());

    assert!(qubit.get((0, 0)).re >= 0.99);
    assert_eq!(qubit.get((0, 1)).re, 0.);
    assert_eq!(qubit.get((1, 0)).re, 0.);
    assert_eq!(qubit.get((1, 1)).re, 0.);
  }

  #[test]
  fn Y() {
    let mut qubit = QubitFragment::DefaultQubit();
    let result = qubit.apply(&GateFragment::Y(&PI));
    assert!(result.is_none());

    let stuff = qubit.to_string();
    let stuff = stuff;

    assert_eq!(qubit.get((0, 0)).re, 0.);
    assert_eq!(qubit.get((0, 1)).re, 0.);
    assert_eq!(qubit.get((1, 0)).re, 0.);
    assert!(qubit.get((1, 1)).re <= -0.99);
  }

  #[test]
  fn Had() {
    let mut qubit = QubitFragment::DefaultQubit();
    let result = qubit.apply(&GateFragment::Had());
    assert!(result.is_none());

    assert!(qubit.get((0, 0)).re > 0.22);
    assert!(qubit.get((0, 1)).re > 0.22);
    assert!(qubit.get((1, 0)).re > 0.22);
    assert!(qubit.get((1, 1)).re > 0.22);
  }
}
