// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2024 Oxford Quantum Circuits Ltd

#![deny(clippy::all, clippy::pedantic)]

use crate::builders::IntegrationRuntime;
use crate::evaluator::QIREvaluator;
use crate::features::QuantumFeatures;
use crate::graphs::ExecutableAnalysisGraph;
use crate::instructions::Value;
use crate::runtime::QuantumRuntime;
use crate::smart_pointers::Ptr;
use crate::with_mutable;
use inkwell::attributes::AttributeLoc;
use inkwell::values::FunctionValue;
use inkwell::{
  context::Context,
  memory_buffer::MemoryBuffer,
  module::Module,
  passes::{PassManager, PassManagerBuilder},
  targets::{InitializationConfig, Target},
  OptimizationLevel
};

use crate::config::RasqalConfig;
use crate::exceptions::catch_panics;
use log::{log, Level};
use std::{ffi::OsStr, path::Path};

/// Executes the file.
pub fn run_file(
  path: impl AsRef<Path>, args: &Vec<Value>, runtimes: &Ptr<RuntimeCollection>,
  entry_point: Option<&str>, config: &Ptr<RasqalConfig>
) -> Result<Option<Ptr<Value>>, String> {
  catch_panics(|| run_graph(&parse_file(path, entry_point)?, args, runtimes, config))
}

/// Parses the .ll/.bc file and builds an [`ExecutableAnalysisGraph`] for it.
pub fn parse_file(
  path: impl AsRef<Path>, entry_point: Option<&str>
) -> Result<Ptr<ExecutableAnalysisGraph>, String> {
  log!(
    Level::Info,
    "Parsing from {}.",
    path.as_ref().to_str().unwrap()
  );
  let context = Context::create();
  let module = file_to_module(path, &context)?;
  catch_panics(|| build_graph_from_module(&module, entry_point))
}

/// Transforms an LLVM file into an LLVM module.
pub fn file_to_module(path: impl AsRef<Path>, context: &Context) -> Result<Module, String> {
  let path = path.as_ref();
  let extension = path.extension().and_then(OsStr::to_str);

  match extension {
    Some("ll") => MemoryBuffer::create_from_file(path)
      .and_then(|buffer| context.create_module_from_ir(buffer))
      .map_err(|e| e.to_string()),
    Some("bc") => Module::parse_bitcode_from_path(path, context).map_err(|e| e.to_string()),
    _ => Err(format!("Unsupported file extension '{extension:?}'."))
  }
}

/// Builds a graph from a QIR module.
pub fn build_graph_from_module(
  module: &Module, entry_point: Option<&str>
) -> Result<Ptr<ExecutableAnalysisGraph>, String> {
  catch_panics(|| {
    module
      .verify()
      .map_err(|e| format!("Failed to verify module: {}", e.to_string()))?;

    let pass_manager_builder = PassManagerBuilder::create();
    pass_manager_builder.set_optimization_level(OptimizationLevel::None);

    let fpm = PassManager::create(());
    fpm.add_global_dce_pass();
    fpm.add_strip_dead_prototypes_pass();
    pass_manager_builder.populate_module_pass_manager(&fpm);
    fpm.run_on(module);

    Target::initialize_native(&InitializationConfig::default())?;
    inkwell::support::load_library_permanently(Path::new(""));

    let entry_point = choose_entry_point(module_functions(module), entry_point)?;

    log!(
      Level::Info,
      "{} is the entry-point.",
      entry_point.get_name().to_str().unwrap()
    );
    let evaluator = QIREvaluator::new();
    evaluator.evaluate(&entry_point, &Ptr::from(module))
  })
}

/// Executes a graph with the current runtimes and context.
pub fn run_graph(
  graph: &Ptr<ExecutableAnalysisGraph>, arguments: &Vec<Value>, runtimes: &Ptr<RuntimeCollection>,
  config: &Ptr<RasqalConfig>
) -> Result<Option<Ptr<Value>>, String> {
  let mut runtime = QuantumRuntime::new(runtimes, config);
  catch_panics(|| runtime.execute(graph, arguments))
}

/// Top-level collection item that holds information about target runtimes and engines for graphs.
pub struct RuntimeCollection {
  QPU_runtimes: Vec<Ptr<IntegrationRuntime>>
}

impl RuntimeCollection {
  pub fn new(engines: Vec<Ptr<IntegrationRuntime>>) -> RuntimeCollection {
    RuntimeCollection {
      QPU_runtimes: engines
    }
  }

  pub fn add(&mut self, python_engine: &Ptr<IntegrationRuntime>) {
    self.QPU_runtimes.push(python_engine.clone());
  }

  pub fn from(python_engine: &Ptr<IntegrationRuntime>) -> RuntimeCollection {
    RuntimeCollection::new(vec![python_engine.clone()])
  }

  /// Fetches the first available QPU which has these features.
  pub fn find_capable_QPU(&self, features: &QuantumFeatures) -> Option<Ptr<IntegrationRuntime>> {
    for engine in self.QPU_runtimes.iter() {
      if with_mutable!(engine.is_valid()) && engine.has_features(features) {
        return Some(engine.clone());
      }
    }

    None
  }
}

impl Default for RuntimeCollection {
  fn default() -> Self {
    RuntimeCollection {
      QPU_runtimes: Vec::default()
    }
  }
}

/// Returns all functions from a module.
pub fn module_functions<'ctx>(module: &Module<'ctx>) -> impl Iterator<Item = FunctionValue<'ctx>> {
  struct FunctionValueIter<'ctx>(Option<FunctionValue<'ctx>>);

  impl<'ctx> Iterator for FunctionValueIter<'ctx> {
    type Item = FunctionValue<'ctx>;

    fn next(&mut self) -> Option<Self::Item> {
      let function = self.0;
      self.0 = function.and_then(FunctionValue::get_next_function);
      function
    }
  }

  FunctionValueIter(module.get_first_function())
}

/// Checks if this function is a QIR entry-point.
pub fn is_entry_point(function: FunctionValue) -> bool {
  function
    .get_string_attribute(AttributeLoc::Function, "entry_point")
    .is_some()
    || function
      .get_string_attribute(AttributeLoc::Function, "EntryPoint")
      .is_some()
}

/// Looks through the entry-points available and either picks the method that matches the name
/// passed-in.
pub fn choose_entry_point<'ctx>(
  functions: impl Iterator<Item = FunctionValue<'ctx>>, name: Option<&str>
) -> Result<FunctionValue<'ctx>, String> {
  if let Some(func_name) = name {
    functions
      .filter(|f| func_name == f.get_name().to_str().unwrap())
      .next()
      .ok_or("Can't find a method with this nane.".to_string())
  } else {
    let eps: Vec<FunctionValue> = functions.filter(|f| is_entry_point(*f)).collect();
    if eps.is_empty() {
      return Err("Can't find any entry-points.".to_string());
    }

    if eps.len() > 1 {
      return Err(
        "No specified method and more than one entry-point. Can't auto-detect.".to_string()
      );
    }
    Ok(*(eps.first().unwrap()))
  }
}

#[cfg(test)]
mod tests {
  use crate::builders::IntegrationRuntime;
  use crate::config::RasqalConfig;
  use crate::execution::{run_file, RuntimeCollection};
  use crate::instructions::Value;
  use crate::smart_pointers::Ptr;
  use std::borrow::Borrow;
  use std::fs::canonicalize;

  /// Just run a QIR file to make sure it parses and returns the value.
  fn run(path: &str) -> Option<Ptr<Value>> { run_with_config(path, RasqalConfig::default()) }

  fn run_with_args(path: &str, args: &Vec<Value>) -> Option<Ptr<Value>> {
    run_with_args_and_config(path, args, RasqalConfig::default()).expect("Execution failed.")
  }

  fn run_with_config(path: &str, config: RasqalConfig) -> Option<Ptr<Value>> {
    run_with_args_and_config(path, &Vec::new(), config).expect("Execution failed.")
  }

  fn fail(path: &str) -> Option<String> { fail_with_config(path, RasqalConfig::default()) }

  fn fail_with_args(path: &str, args: &Vec<Value>) -> Option<String> {
    run_with_args_and_config(path, args, RasqalConfig::default()).err()
  }

  fn fail_with_config(path: &str, config: RasqalConfig) -> Option<String> {
    run_with_args_and_config(path, &Vec::new(), config).err()
  }

  fn run_with_args_and_config(
    path: &str, args: &Vec<Value>, config: RasqalConfig
  ) -> Result<Option<Ptr<Value>>, String> {
    let relative_path = canonicalize(path).unwrap();
    let path = relative_path.to_str().unwrap();

    let runtimes = Ptr::from(RuntimeCollection::from(&Ptr::from(
      IntegrationRuntime::default()
    )));

    run_file(path, args, runtimes.borrow(), None, &Ptr::from(config))
  }

  // #[test]
  // fn execute_500qb_125k() {
  //   let config = RasqalConfig::default().with_activate_solver();
  //   run_with_config("../tests/files/qir/large/500qb125k.ll", config);
  // }

  // #[test]
  // fn execute_500qb_20k() {
  //   let config = RasqalConfig::default().with_activate_solver();
  //   run_with_config("../tests/files/qir/large/500qb20k.ll", config);
  // }

  #[test]
  fn execute_500qb_4k() {
    let config = RasqalConfig::default().with_activate_solver();
    run_with_config("../tests/files/qir/large/500qb4k.ll", config);
  }

  #[test]
  fn execute_qaoa_solver() {
    let config = RasqalConfig::default().with_activate_solver();
    run_with_config("../tests/qsharp/qaoa/qir/qaoa.ll", config);
  }

  #[test]
  fn execute_qaoa() { run("../tests/qsharp/qaoa/qir/qaoa.ll"); }

  #[test]
  fn execute_simplified_oracle_generator() {
    run("../tests/qsharp/simplified-oracle-generator/qir/simplified-oracle-generator.ll");
  }

  #[test]
  fn execute_oracle_generator() { run("../tests/qsharp/oracle-generator/qir/oracle-generator.ll"); }

  #[test]
  fn execute_minified_oracle_generator() {
    run_with_args(
      "../tests/qsharp/minified-oracle-generator/qir/minified-oracle-generator.ll",
      &vec![Value::from(true)]
    );
  }

  #[test]
  fn execute_unrestricted_bell() { run("../tests/files/qir/unrestricted_bell.ll"); }

  #[test]
  fn test_step_count() {
    let mut config = RasqalConfig::default();
    config.step_count_limit(2);
    let results = fail_with_config("../tests/files/qir/unrestricted_bell.ll", config);
    assert!(results.is_some())
  }

  #[test]
  fn execute_bell_int_return() { run(&"../tests/files/qir/bell_int_return.ll"); }

  #[test]
  fn execute_bell_psi_minus() { run(&"../tests/files/qir/bell_psi_minus.ll"); }

  #[test]
  fn execute_bell_psi_plus() { run(&"../tests/files/qir/bell_psi_plus.ll"); }

  #[test]
  fn execute_bell_theta_plus() { run(&"../tests/files/qir/bell_theta_plus.ll"); }

  #[test]
  fn execute_bell_theta_minus() { run(&"../tests/files/qir/bell_theta_minus.ll"); }

  #[test]
  fn execute_basic_cudaq() {
    let config = RasqalConfig::default().with_trace_projections();
    run_with_config(&"../tests/files/qir/basic_cudaq.ll", config);
  }
}
