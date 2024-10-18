// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2024 Oxford Quantum Circuits Ltd

use crate::graphs::{
  AnalysisGraph, AnalysisGraphBuilder, CallableAnalysisGraph, ExecutableAnalysisGraph, Node
};
use crate::hardware::Qubit;
use crate::instructions::{
  Condition, Equalities, Expression, Instruction, LambdaModifier, Operator, Pauli, Value
};
use crate::runtime::RuntimeContext;
use crate::smart_pointers::Ptr;
use crate::with_mutable;
use inkwell::basic_block::BasicBlock;
use inkwell::module::Module;
use inkwell::types::AnyTypeEnum;
use inkwell::values::{
  AnyValue, AnyValueEnum, AsValueRef, BasicValue, BasicValueEnum, FunctionValue, InstructionOpcode,
  InstructionValue
};
use inkwell::{FloatPredicate, IntPredicate};
use lazy_static::lazy_static;
use llvm_sys::core::{
  LLVMConstIntGetSExtValue, LLVMGetElementType, LLVMGetNumOperands, LLVMGetOperand,
  LLVMGetTypeKind, LLVMPrintTypeToString, LLVMPrintValueToString, LLVMTypeOf
};
use llvm_sys::prelude::LLVMValueRef;
use llvm_sys::LLVMTypeKind;
use log::{log, warn, Level};
use regex::Regex;
use std::borrow::{Borrow, BorrowMut};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::ffi::{c_uint, CStr};
use std::ops::Deref;
use std::time::Instant;

macro_rules! operand_to_value {
  ($target:ident, $index:expr) => {
    $target
      .get_operand($index)
      .expect("Can't resolve operand.")
      .left()
      .expect("Operand isn't a value.")
      .as_any_value_enum()
      .borrow_mut()
  };
}

macro_rules! operand_to_instruction {
  ($target:ident, $index:expr) => {
    $target
      .get_operand($index)
      .expect("Can't resolve operand.")
      .left()
      .expect("Operand isn't a value.")
  };
}

macro_rules! operand_to_bb {
  ($target:ident, $index:expr) => {
    $target
      .get_operand($index)
      .expect("Can't resolve operand.")
      .right()
      .expect("Operand isn't a basic block.")
  };
}

// TODO: Since Inkwell doesn't expose things properly try and use the llvm-sys objects to find the
//  data. We want to remove all the string fetching/matching as it's inefficient.

/// Fetches the assignment variable &{value} = ... from a stringified LLVM instruction.
pub fn get_ref_id_from_instruction(inst: &InstructionValue) -> String {
  let inst_str = inst
    .to_string()
    .trim_end_matches('"')
    .trim_start_matches('"')
    .trim()
    .to_string();
  parse_ref_id_from_instruction(inst).expect("Can't find ref-id from instruction")
}

/// Same as [`get_ref_id_from_instruction`] which doesn't panic.
pub fn parse_ref_id_from_instruction(inst: &InstructionValue) -> Option<String> {
  let inst_str = inst
    .to_string()
    .trim_end_matches('"')
    .trim_start_matches('"')
    .trim()
    .to_string();
  parse_ref_id_from_instruction_str(&inst_str)
}

/// See [`get_ref_id_from_instruction`].
pub fn parse_ref_id_from_instruction_str(inst_str: &str) -> Option<String> {
  lazy_static! {
    static ref llvm_var_finder: Regex = Regex::new("([%@][^ ]*) =").unwrap();
  }
  llvm_var_finder.captures(inst_str).map_or_else(
    || parse_ref_id_from_value(inst_str).or(None),
    |capture_groups| Some(capture_groups.get(1).unwrap().as_str().to_string())
  )
}

// TODO: Need a proper way to get the variables from a general state, while this works it's not
//     entirely bulletproof and needs tweaking as issues come up. And issues caused from it are not
//     immediately obvious.
/// Attempts to get a variable assignment from an argument value: some_call(%Array* %register, ...).
pub fn parse_ref_id_from_value(ptr_string: &str) -> Option<String> {
  let ptr_string = ptr_string.trim_matches('"').trim();
  lazy_static! {
    static ref local_variable_finder: Regex = Regex::new("^.*\\s(%[\\w0-9\\-]+)$").unwrap();
  }
  let capture_groups = local_variable_finder.captures(ptr_string);
  let mut ref_id = capture_groups.map(|val| val.get(1).unwrap().as_str().to_string());

  // If we can't find a local variable, look globally.
  ref_id = ref_id.or_else(|| {
    lazy_static! {
      static ref global_variable_finder: Regex = Regex::new("^.*\\s(@[\\w0-9\\-]+)$").unwrap();
    }
    let capture_groups = global_variable_finder.captures(ptr_string);
    capture_groups.and_then(|val| {
      let val = val.get(1).unwrap().as_str().to_string();
      if val.trim().is_empty() {
        None
      } else {
        Some(val)
      }
    })
  });

  // Finally check if we're a global instruction target.
  ref_id.or_else(|| {
    lazy_static! {
      static ref global_instruction_finder: Regex = Regex::new("^@[^\\s*]+(\\s|$)").unwrap();
    }
    let capture_groups = global_instruction_finder.captures(ptr_string);
    capture_groups.and_then(|value| {
      let mut value = value.get(0).unwrap().as_str();
      value = value.trim();
      if value.is_empty() {
        None
      } else {
        Some(value.to_string())
      }
    })
  })
}

/// See [`parse_ref_id_from_value`]
pub fn get_ref_id_from_value(ptr_string: &str) -> String {
  parse_ref_id_from_value(ptr_string).expect("Can't parse ref-id from value.")
}

/// Parsing context, molds all state required by the evalautor to run.
pub struct EvaluationContext<'ctx> {
  pub module: Ptr<Module<'ctx>>,
  pub global_variables: Ptr<HashMap<String, Ptr<Value>>>,

  /// Basic-block anchor nodes, allows us to reference the start/end of a basic block
  /// without it having an explicit node.
  pub anchors: HashMap<String, Ptr<Node>>,

  /// The graphs currently built, or in the process of building, key'd by their name.
  pub method_graphs: Ptr<HashMap<String, Ptr<AnalysisGraph>>>,

  /// Hack for QIR implementations that don't implement variables at all and assume
  /// measures/returns by magic means. If not empty all returns will instead return
  /// the values in the list.
  pub is_base_profile: Ptr<bool>,

  /// Incremental counter for throwaway variables.
  pub throwaway_variables: Ptr<i64>
}

impl<'a> EvaluationContext<'a> {
  pub fn new(module: &Ptr<Module<'a>>) -> EvaluationContext<'a> {
    EvaluationContext {
      module: module.clone(),
      global_variables: Ptr::from(HashMap::new()),
      anchors: HashMap::new(),
      method_graphs: Ptr::from(HashMap::new()),
      is_base_profile: Ptr::from(false),
      throwaway_variables: Ptr::from(0)
    }
  }

  /// Creates a subcontext for individual methods. All method-scoped
  /// variables are reset while persisting global values.
  pub fn create_subcontext(parent: &Ptr<EvaluationContext<'a>>) -> EvaluationContext<'a> {
    EvaluationContext {
      module: parent.module.clone(),
      global_variables: parent.global_variables.clone(),
      anchors: HashMap::new(),
      method_graphs: parent.method_graphs.clone(),
      is_base_profile: parent.is_base_profile.clone(),
      throwaway_variables: parent.throwaway_variables.clone()
    }
  }

  /// Gets the next throwaway variable for assignment.
  pub fn next_throwaway(&self) -> String {
    let var = format!("_eph_{}", self.throwaway_variables);
    unsafe {
      self
        .throwaway_variables
        .as_ptr()
        .replace(*self.throwaway_variables + 1);
    }
    var
  }
}

/// Parser for turning QIR LLVM Modules into an [`AnalysisGraph`].
pub struct QIREvaluator {}

impl QIREvaluator {
  pub fn new() -> QIREvaluator { QIREvaluator {} }

  /// Evaluates a module and entry-point for execution. Returns the execution graph to then be
  /// run in an interpreter.
  pub fn evaluate(
    &self, entry_point: &FunctionValue, module: &Ptr<Module>
  ) -> Result<Ptr<ExecutableAnalysisGraph>, String> {
    let mut context = Ptr::from(EvaluationContext::new(module));
    let mut target_global = module.get_first_global();
    while target_global.is_some() {
      let global = target_global.unwrap();
      let maybe_initializer = global.get_initializer();
      if let Some(init) = maybe_initializer {
        // TODO: Remove graph requirement from as_value.
        if let Some(value) =
          self.as_value(&init.as_any_value_enum(), &Ptr::default(), context.borrow())
        {
          // Some globals seem invalid here, in that case don't add them.
          if let Some(ref_id) = parse_ref_id_from_instruction_str(&global.to_string()) {
            context.global_variables.insert(ref_id, Ptr::from(value));
          }
        }
      }

      target_global = global.get_next_global();
    }

    let start = Instant::now();
    let builder = self.walk_function(entry_point, context.borrow());
    let took = start.elapsed();
    log!(Level::Info, "Evaluation took {}ms.", took.as_millis());

    // Create a callable graph with its arguments, but the values set as empty (validly).
    let mut callable = Ptr::from(CallableAnalysisGraph::new(&builder.graph));
    for param in entry_point.get_params().iter() {
      let param_ref_id = get_ref_id_from_value(&param.to_string());
      callable
        .argument_mappings
        .insert(param_ref_id, Ptr::from(Value::Empty));
    }

    let exe_graph = ExecutableAnalysisGraph::with_context(
      &callable,
      &Ptr::from(RuntimeContext::from_evaluation(&context))
    );

    Ok(Ptr::from(exe_graph))
  }

  /// For-now method to retrieve the name of a Call target. I'm sure it's in the
  /// instruction somewhere but not obvious how to retrieve it via this API.
  fn get_method_name(&self, inst: &InstructionValue) -> Option<String> {
    if inst.get_opcode() != InstructionOpcode::Call {
      return None;
    }

    let mut operation_name = inst.print_to_string().to_string();
    let start = operation_name.find('@')?;
    let end = operation_name.find('(')?;
    let mut call_name = operation_name.split_off(start + 1);
    call_name.truncate(end - start - 1);
    Some(call_name)
  }

  /// Note: on the exit of a basic block there should be no auto-attach point. Each BB is isolated
  /// and the various jumps are dealt with by attaching to the anchors and distinct operations.
  fn walk_basic_block(
    &self, bb: &BasicBlock, graph: &Ptr<AnalysisGraphBuilder>, context: &Ptr<EvaluationContext>
  ) {
    // Attach our starting anchor as the default attach point to get started.
    let bb_name = bb.get_name().to_str().unwrap();
    let starting_point = with_mutable!(context
      .anchors
      .get_mut(bb_name)
      .expect("Anchor needs to exist."));
    with_mutable!(graph.reattach(starting_point));

    let mut next_inst = bb.get_first_instruction();
    while next_inst.is_some() {
      let inst = Ptr::from(next_inst.unwrap());
      self.walk_instruction(inst.borrow(), graph, context);
      next_inst = inst.get_next_instruction();
    }

    with_mutable!(graph.unattach());
  }

  fn walk_function(
    &self, func: &FunctionValue, context: &Ptr<EvaluationContext>
  ) -> Ptr<AnalysisGraphBuilder> {
    let method_name = func.get_name().to_str().unwrap().to_string();
    if let Some(existing) = context.method_graphs.get(method_name.as_str()) {
      return Ptr::from(AnalysisGraphBuilder::new(existing));
    }

    let mut subcontext = Ptr::from(EvaluationContext::create_subcontext(context));

    let graph = Ptr::from(AnalysisGraph::new(method_name.clone()));
    with_mutable!(context.method_graphs.insert(method_name, graph.clone()));

    // Build up anchor labels/nodes so we can associate them at the start and end.
    for bb in func.get_basic_blocks() {
      let bb_name = bb.get_name().to_str().unwrap().to_string();
      let anchor_node = with_mutable!(graph.add_loose(Instruction::Label(bb_name.clone())));
      subcontext.anchors.insert(bb_name, anchor_node.clone());
    }

    let builder = Ptr::from(AnalysisGraphBuilder::new(graph.borrow()));
    for bb in func.get_basic_blocks().iter() {
      self.walk_basic_block(bb, &builder, subcontext.borrow());
    }

    builder
  }

  /// Hacked-together method to centralize GEP extraction. Done using llvm-sys objects because
  /// Inkwell doesn't have any way to access operands when a GEP is an argument.
  fn extract_gep(
    &self, any_val: &AnyValueEnum, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) -> Option<Value> {
    unsafe {
      let expr = any_val.as_value_ref();
      let first_op = LLVMGetOperand(expr, 0);

      // For now assume getelementptr only works on tuples/vectors/structs and we have an
      // element kind. May not always hold.
      let type_ref = LLVMTypeOf(first_op);
      let type_string = CStr::from_ptr(LLVMPrintTypeToString(type_ref))
        .to_str()
        .unwrap()
        .to_string();
      let ele_type = LLVMGetElementType(type_ref);
      let ele_kind = LLVMGetTypeKind(ele_type);

      let llvm_string = CStr::from_ptr(LLVMPrintValueToString(first_op));
      let ref_id = parse_ref_id_from_instruction_str(&llvm_string.to_string_lossy().to_string())
        .expect("Need ref-id from instruction");
      let mut prev_throwaway = context.next_throwaway();
      graph.Assign(prev_throwaway.clone(), Value::Ref(ref_id.clone(), None));

      let op_num = LLVMGetNumOperands(expr);

      // The first operand is the type/pointer, second is walking through the pointer so is
      // always 0, we ignore this.
      let mut starting_operand = 2;

      // If we're an array we skip another indexer as GEP sets the address at the beginning
      // of an array for iteration.
      // TODO: Likely not accurate for every type, especially jagged arrays.
      if ele_kind == LLVMTypeKind::LLVMArrayTypeKind {
        starting_operand += 1;
      }

      while starting_operand < op_num {
        let next_throwaway = context.next_throwaway();
        let op_value = LLVMGetOperand(expr, starting_operand as c_uint);
        let actual_index: i64 = LLVMConstIntGetSExtValue(op_value);
        graph.Assign(
          next_throwaway.clone(),
          Value::Ref(
            prev_throwaway.clone(),
            Some(Ptr::from(Value::Int(actual_index)))
          )
        );
        prev_throwaway = next_throwaway;
        starting_operand += 1;
      }

      Some(Value::Ref(prev_throwaway.clone(), None))
    }
  }

  /// Fixed up const_extract_value from Inkwell.
  pub fn const_extract_value(&self, array: LLVMValueRef, index: u32) -> BasicValueEnum {
    use llvm_sys::core::LLVMGetAggregateElement;

    unsafe { BasicValueEnum::new(LLVMGetAggregateElement(array, index as c_uint)) }
  }

  /// Recursive call of [`as_value`]. Call the non-recursive version for most use-cases.
  fn _as_value_recursive(
    &self, graph: &Ptr<AnalysisGraphBuilder>, type_enum: &AnyTypeEnum, val_enum: &AnyValueEnum,
    context: &Ptr<EvaluationContext>
  ) -> Option<Value> {
    let ephemeral = val_enum.to_string();
    let stringified_value = ephemeral.trim_matches('"').trim();
    let ref_id = parse_ref_id_from_value(&stringified_value);
    if let Some(ref_id_value) = ref_id {
      return Some(Value::Ref(ref_id_value, None));
    }

    match type_enum {
      AnyTypeEnum::ArrayType(t) => {
        // Arrays are either strings or constant arrays of values, so redirect based
        // on which one it is.
        let vec = val_enum.into_array_value();
        if vec.is_const_string() {
          Some(Value::String(
            vec
              .get_string_constant()
              .unwrap()
              .to_str()
              .unwrap()
              .to_string()
          ))
        } else if vec.is_const() {
          let mut result = Vec::new();
          for int in 0..(t.len()) {
            result.push(
              self
                .as_value_ptr(
                  &self
                    .const_extract_value(vec.as_value_ref(), int)
                    .as_any_value_enum(),
                  graph,
                  context
                )
                .expect("Can't resolve array element.")
            );
          }

          Some(Value::Array(result))
        } else {
          Some(Value::Empty)
        }
      }
      AnyTypeEnum::FloatType(t) => {
        let llvm_context = t.get_context();

        // Second value is about losiness of floats, which right now we discard.
        let numeric = val_enum
          .into_float_value()
          .get_constant()
          .expect("Float parsing has failed.")
          .0;

        // TODO: Find a way to make f128 work, if it needs it.
        if t == llvm_context.f16_type().borrow_mut() {
          Some(Value::Float(numeric))
        } else if t == llvm_context.f32_type().borrow_mut() {
          Some(Value::Float(numeric))
        } else if t == llvm_context.f64_type().borrow_mut() {
          Some(Value::Float(numeric))
        } else if t == llvm_context.f128_type().borrow_mut() {
          Some(Value::Float(numeric))
        } else {
          Some(Value::Float(numeric))
        }
      }
      AnyTypeEnum::IntType(t) => {
        let llvm_context = t.get_context();
        let numeric = val_enum
          .into_int_value()
          .get_sign_extended_constant()
          .expect("Int parsing has failed.");

        // TODO: Doesn't _really_ deal with longs.
        return if t == llvm_context.bool_type().borrow_mut() {
          // Bools come in as -1 = true, 0 = false.
          Some(Value::Bool(numeric == -1))
        } else if t == llvm_context.i8_type().borrow_mut() {
          Some(Value::Short(numeric as i16))
        } else if t == llvm_context.i16_type().borrow_mut() {
          Some(Value::Short(numeric as i16))
        } else if t == llvm_context.i32_type().borrow_mut() {
          Some(Value::Int(numeric))
        } else if t == llvm_context.i64_type().borrow_mut() {
          Some(Value::Int(numeric))
        } else if t == llvm_context.i128_type().borrow_mut() {
          Some(Value::Long(numeric as i128))
        } else {
          Some(Value::Int(numeric))
        };
      }
      AnyTypeEnum::PointerType(t) => {
        // TODO: GEP analysis and fixing should be cleaned up as soon as Inkwell
        //  supports it.
        if stringified_value.contains("getelementptr") {
          self.extract_gep(val_enum, graph, context)
        } else {
          let pointer_val = val_enum.into_pointer_value();

          // This is purely for base profile support since its syntax is invalid.
          lazy_static! {
            static ref base_profile_finder: Regex =
              Regex::new("^%(Qubit|Result)\\* ((inttoptr \\(i64 ([0-9]+))|(null))").unwrap();
          }

          let capture_groups = base_profile_finder.captures(&stringified_value);
          if let Some(groupings) = capture_groups {
            let name = groupings.get(1).unwrap().as_str();
            let mut value = if let Some(matched) = groupings.get(4) {
              matched.as_str()
            } else {
              groupings
                .get(5)
                .expect(
                  format!(
                    "Unable to find base profile value. Instruction: {}",
                    stringified_value
                  )
                  .as_str()
                )
                .as_str()
            };

            // nulls get coerced into 0 if processed correctly, but we have to do it manually here.
            if value == "null" {
              value = "0";
            }

            return match name {
              "Qubit" => Some(Value::Qubit(Qubit::new(value.parse().unwrap()))),
              "Result" => Some(Value::Int(value.parse().unwrap())),
              _ => panic!("Attempted specific match on non-base-profile pointer. Instruction: {}, name: {}, value: {}", stringified_value, name, value)
            };
          }

          // Structs, especially opaque ones, have their own rules and even if they're
          // null it may mean something very different.
          if pointer_val.is_null() || pointer_val.is_undef() {
            return Some(Value::Empty);
          }

          panic!("Unable to resolve pointer value.");
        }
      }
      AnyTypeEnum::StructType(t) => {
        // Opaques need us to give them a type...
        if t.is_opaque() {
          let struct_name = t
            .get_name()
            .unwrap_or_default()
            .to_str()
            .unwrap_or_default();
          let ptr_val = val_enum.into_pointer_value();
          let index = if ptr_val.is_null() || ptr_val.is_undef() {
            0
          } else {
            ptr_val
              .const_to_int(context.module.get_context().i64_type())
              .get_sign_extended_constant()
              .unwrap_or_default()
          };

          // TODO: Make custom results object, probably re-use projection results.
          match struct_name {
            "Qubit" => Some(Value::Qubit(Qubit::new(index))),
            "Result" => Some(Value::Int(index)),
            val => {
              // Qubit and results are special in that their nulls = 0, everything
              // else null is implied as empty.
              if ptr_val.is_null() {
                return Some(Value::Empty);
              }

              unimplemented!()
            }
          }

        // Where-as pure composites already have a structure, it's just freeform.
        // For now just consider them arrays (as access is the same).

        // TODO: Decide whether arrays/composites need to be distinguished.
        } else {
          let struct_val = val_enum.into_struct_value();
          let mut result = Vec::new();
          for int in 0..(t.count_fields()) {
            result.push(
              self
                .as_value_ptr(
                  &self
                    .const_extract_value(struct_val.as_value_ref(), int)
                    .as_any_value_enum(),
                  graph,
                  context
                )
                .expect("Can't resolve struct element.")
            );
          }

          Some(Value::Array(result))
        }
      }
      AnyTypeEnum::VectorType(_) => {
        unimplemented!()
      }
      AnyTypeEnum::VoidType(_) => {
        unimplemented!()
      }
      AnyTypeEnum::FunctionType(_) => {
        unimplemented!()
      }
    }
  }

  /// See [`as_value`] but returns value wrapped in a flexi-pointer.
  fn as_value_ptr(
    &self, any_val: &AnyValueEnum, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) -> Option<Ptr<Value>> {
    let result = self.as_value(any_val, graph, context);
    result.map(|val| Ptr::from(val))
  }

  /// Evaluates the instruction and returns its results as a value.
  fn as_value(
    &self, any_val: &AnyValueEnum, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) -> Option<Value> {
    let function_val = match any_val {
      AnyValueEnum::FunctionValue(fv) => Some(Value::Ref(
        fv.get_name().to_str().unwrap().to_string(),
        None
      )),
      _ => None
    };

    // If we're a function value don't continue, as their to_string is the entire function.
    if function_val.is_some() {
      return function_val;
    }

    let instruction_value = match any_val {
      AnyValueEnum::ArrayValue(av) => av.as_instruction(),
      AnyValueEnum::FloatValue(av) => av.as_instruction(),
      AnyValueEnum::IntValue(av) => av.as_instruction(),
      AnyValueEnum::PointerValue(av) => av.as_instruction(),
      AnyValueEnum::StructValue(av) => av.as_instruction(),
      AnyValueEnum::VectorValue(av) => av.as_instruction(),
      _ => None
    };

    // If we're an instruction get the value assignment for the result of it instead.
    if instruction_value.is_some() {
      let ref_id = get_ref_id_from_instruction(instruction_value.unwrap().borrow());
      return Some(Value::Ref(ref_id, None));
    }

    // Recursive method for looping through pointers to get to values.
    self._as_value_recursive(graph, any_val.get_type().borrow(), any_val, context)
  }

  /// Evaluates a specific LLVM IR instruction and adds it to the graph.
  fn walk_instruction(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let op_code = inst.get_opcode();
    match op_code {
      InstructionOpcode::Call => {
        self.eval_call(inst, graph, context);
      }
      InstructionOpcode::Return => {
        self.eval_ret(inst, graph, context);
      }
      InstructionOpcode::Br => {
        self.eval_branch(inst, graph, context);
      }
      InstructionOpcode::Switch | InstructionOpcode::IndirectBr | InstructionOpcode::Invoke => {
        todo!("{}", inst.print_to_string().to_string())
      }
      InstructionOpcode::FNeg => {
        self.eval_fneg(inst, graph, context);
      }
      InstructionOpcode::Add => {
        self.eval_add(inst, graph, context);
      }
      InstructionOpcode::FAdd => {
        self.eval_add(inst, graph, context);
      }
      InstructionOpcode::Sub => {
        self.eval_sub(inst, graph, context);
      }
      InstructionOpcode::FSub => {
        self.eval_sub(inst, graph, context);
      }
      InstructionOpcode::Mul => {
        self.eval_mul(inst, graph, context);
      }
      InstructionOpcode::FMul => {
        self.eval_mul(inst, graph, context);
      }
      InstructionOpcode::UDiv => {
        self.eval_div(inst, graph, context);
      }
      InstructionOpcode::SDiv => {
        self.eval_div(inst, graph, context);
      }
      InstructionOpcode::FDiv => {
        self.eval_div(inst, graph, context);
      }
      InstructionOpcode::URem
      | InstructionOpcode::SRem
      | InstructionOpcode::FRem
      | InstructionOpcode::Shl
      | InstructionOpcode::LShr
      | InstructionOpcode::AShr => {
        todo!("{}", inst.print_to_string().to_string())
      }
      InstructionOpcode::And => {
        self.eval_or(inst, graph, context);
      }
      InstructionOpcode::Or => {
        self.eval_or(inst, graph, context);
      }
      InstructionOpcode::Xor => {
        self.eval_xor(inst, graph, context);
      }
      InstructionOpcode::ExtractElement
      | InstructionOpcode::InsertElement
      | InstructionOpcode::ShuffleVector => {
        todo!("{}", inst.print_to_string().to_string())
      }
      InstructionOpcode::ExtractValue => {
        self.eval_extractvalue(inst, graph, context);
      }
      InstructionOpcode::InsertValue => {
        self.eval_insertvalue(inst, graph, context);
      }
      InstructionOpcode::Load => {
        self.eval_load(inst, graph, context);
      }
      InstructionOpcode::Store => {
        self.eval_store(inst, graph, context);
      }
      InstructionOpcode::Fence
      | InstructionOpcode::AtomicCmpXchg
      | InstructionOpcode::AtomicRMW => {
        todo!("{}", inst.print_to_string().to_string())
      }
      InstructionOpcode::GetElementPtr => {
        self.eval_getelementptr(inst, graph, context);
      }
      InstructionOpcode::Trunc => {
        self.eval_trunc(inst, graph, context);
      }
      InstructionOpcode::FPTrunc => {
        self.eval_trunc(inst, graph, context);
      }
      InstructionOpcode::ZExt | InstructionOpcode::FPExt | InstructionOpcode::SExt => {
        self.eval_ext(inst, graph, context);
      }
      InstructionOpcode::FPToUI => {
        self.eval_numeric_cast(inst, graph, context);
      }
      InstructionOpcode::UIToFP => {
        self.eval_numeric_cast(inst, graph, context);
      }
      InstructionOpcode::FPToSI => {
        self.eval_numeric_cast(inst, graph, context);
      }
      InstructionOpcode::SIToFP => {
        self.eval_numeric_cast(inst, graph, context);
      }
      InstructionOpcode::PtrToInt => {
        todo!("{}", inst.print_to_string().to_string())
      }
      InstructionOpcode::IntToPtr => {
        self.eval_int_to_ptr(inst, graph, context);
      }
      InstructionOpcode::BitCast => {
        self.eval_bitcast(inst, graph, context);
      }
      InstructionOpcode::AddrSpaceCast => {
        todo!("{}", inst.print_to_string().to_string())
      }
      InstructionOpcode::ICmp => {
        self.eval_icmp(inst, graph, context);
      }
      InstructionOpcode::FCmp => {
        self.eval_icmp(inst, graph, context);
      }
      InstructionOpcode::Phi => {
        // All a phi's logic is taken care of by the associated branches, so the phi itself
        // doesn't need to be processed.
      }
      InstructionOpcode::Select => {
        self.eval_select(inst, graph, context);
      }
      InstructionOpcode::Alloca
      | InstructionOpcode::Resume
      | InstructionOpcode::Freeze
      | InstructionOpcode::VAArg
      | InstructionOpcode::LandingPad
      | InstructionOpcode::CatchPad
      | InstructionOpcode::CleanupPad
      | InstructionOpcode::Unreachable => {
        // Instructions we likely won't need for quite some time, if ever.
      }
      _ => panic!(
        "Unknown instruction type: {}! Can't verify program will execute correctly.",
        inst.print_to_string().to_string()
      )
    }
  }

  /// Evaluates the method intrinsics (aka without any body). This covers a whole range of
  /// things from system calls, custom compiler intrinsics to QIR intrinsics.
  ///
  /// We only acknowledge QIR intrinsics.
  fn eval_intrinsic(
    &self, name: String, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) -> Option<Value> {
    let parse_as_value = |inst: &Ptr<InstructionValue>, index: u32| -> Option<Value> {
      let op = inst
        .get_operand(index)
        .unwrap_or_else(|| panic!("Operand at {} doesn't exist", 0));
      let qb_value = op.left().expect("Operand isn't a value.");
      self.as_value(qb_value.as_any_value_enum().borrow(), graph, context)
    };

    let parse_qubit = |inst: &Ptr<InstructionValue>, index: u32| -> Value {
      parse_as_value(inst, index).expect("Can't find a qubit variable.")
    };

    // Parse the lambda array, evaluate all potential methods and return the first one to
    // use as an anchor.
    let parse_default_callable = |global_name: &String| -> Option<Ptr<AnalysisGraph>> {
      context
        .global_variables
        .get(global_name)
        .and_then(|callable_array| {
          let mut first = None;
          for val in callable_array.as_array() {
            if let Some((method_name, _)) = val.try_as_reference() {
              if let Some(llvm_method) = context.module.get_function(method_name.as_str()) {
                let first_eval = !context.method_graphs.contains_key(method_name.as_str());
                let mut builder = self.walk_function(&llvm_method, context);
                if first_eval {
                  for exit in builder.exit_points() {
                    let ret_node = Ptr::from(Instruction::Return(Ptr::from(Value::Ref(
                      "%result-tuple".to_string(),
                      None
                    ))));
                    builder.add_with_edge(ret_node.borrow(), exit.borrow(), None, None);
                  }
                }

                if first.is_none() {
                  first = Some(builder.graph.clone());
                }
              }
            }
          }

          first
        })
    };

    // X is mapped as 1 instead of -1 in the test files we have. Fix-up for now.
    let fix_pauli = |mut pauli: Value| -> Value {
      if let Value::Int(mut i) = pauli {
        if i == 1 {
          i = -1;
        }

        pauli = Value::Pauli(Pauli::from_num(&(i as i8)));
      }

      pauli
    };

    // Expands the qis__ctrl argument tuples out.
    let expand_arg_tuple = |tuple_index: u32| -> (String, String, String) {
      // The arguments are in a tuple: controllers, (pauli, target, rotation).
      // Extract and expand.
      let target_tuple = parse_as_value(inst, tuple_index).expect("Need tuple to flatten.");
      let tuple_var = context.next_throwaway();
      graph.Assign(tuple_var.clone(), target_tuple);
      let pauli = context.next_throwaway();
      graph.Assign(
        pauli.clone(),
        Value::Ref(tuple_var.clone(), Some(Ptr::from(Value::Int(0))))
      );
      let rotation = context.next_throwaway();
      graph.Assign(
        rotation.clone(),
        Value::Ref(tuple_var.clone(), Some(Ptr::from(Value::Int(1))))
      );
      let target = context.next_throwaway();
      graph.Assign(
        target.clone(),
        Value::Ref(tuple_var, Some(Ptr::from(Value::Int(2))))
      );
      (pauli, target, rotation)
    };

    match name.as_str() {
      // Rotations
      "__quantum__qis__r__body" => {
        let mut pauli = parse_as_value(inst, 0).expect("Can't find a pauli.");
        pauli = fix_pauli(pauli);

        let rotation = parse_as_value(inst, 1).expect("Can't find a rotation.");
        let qubit = parse_as_value(inst, 2).expect("Can't find a qubit.");
        graph.R(pauli, qubit, rotation);
      }
      "__quantum__qis__r__ctl" => {
        let control = parse_qubit(inst, 0);
        let (pauli, target, rotation) = expand_arg_tuple(1);
        graph.CR(
          Value::Ref(pauli, None),
          control,
          Value::Ref(target, None),
          Value::Ref(rotation, None)
        );
      }
      "__quantum__qis__r__adj" => {
        let controls = parse_as_value(inst, 0).expect("Can't find controls.");
        let qubit = parse_as_value(inst, 1).expect("Can't find a qubit.");
        let rotation = parse_as_value(inst, 2).expect("Can't find a rotation.");
        graph.CR(Value::Pauli(Pauli::Z), controls, qubit, rotation);
      }
      "__quantum__qis__r__ctladj" => {
        let controls = parse_as_value(inst, 0).expect("Can't find controls.");
        let (pauli, target, rotation) = expand_arg_tuple(1);
        let throwaway = context.next_throwaway();

        graph.Expression(
          Expression::NegateSign(Value::Ref(rotation, None)),
          Some(throwaway.clone())
        );
        graph.CR(
          Value::Pauli(Pauli::Z),
          controls,
          Value::Ref(target, None),
          Value::Ref(throwaway, None)
        );
      }
      "__quantum__qis__h__body" => {
        let qubit = parse_qubit(inst, 0);
        graph.Z(qubit.clone(), PI);
        graph.Y(qubit, PI / 2.0);
      }
      "__quantum__qis__h__ctl" => {
        let controllers = parse_as_value(inst, 0).expect("Couldn't resolve control qubits.");
        let target = parse_qubit(inst, 1);
        graph.CZ(controllers.clone(), target.clone(), PI);
        graph.CY(controllers, target, PI / 2.0);
      }
      "__quantum__qis__s__body" => {
        let qb = parse_qubit(inst, 0);
        graph.Z(qb, PI / 2.0);
      }
      "__quantum__qis__s__adj" => {
        let qb = parse_qubit(inst, 0);
        graph.Z(qb, -(PI / 2.0));
      }
      "__quantum__qis__s__ctl" => {
        let controllers = parse_as_value(inst, 0).expect("Need control qubits.");
        let qb = parse_qubit(inst, 1);
        graph.CZ(controllers, qb, PI / 2.0);
      }
      "__quantum__qis__s__ctladj" => {
        let controllers = parse_as_value(inst, 0).expect("Need control qubits.");
        let qb = parse_qubit(inst, 1);
        graph.CZ(controllers, qb, -PI / 2.0);
      }
      "__quantum__qis__t__body" => {
        let qb = parse_qubit(inst, 0);
        graph.Z(qb, PI / 4.0);
      }
      "__quantum__qis__t__adj" => {
        let qb = parse_qubit(inst, 0);
        graph.Z(qb, -(PI / 4.0));
      }
      "__quantum__qis__t__ctl" => {
        let controllers = parse_as_value(inst, 0).expect("Need control qubits.");
        let qb = parse_qubit(inst, 0);
        graph.CZ(controllers, qb, PI / 4.0);
      }
      "__quantum__qis__t__ctladj" => {
        let controllers = parse_as_value(inst, 0).expect("Need control qubits.");
        let qb = parse_qubit(inst, 0);
        graph.CZ(controllers, qb, -PI / 4.0);
      }
      "__quantum__qis__x__body" => {
        let qb = parse_qubit(inst, 0);
        graph.X(qb, PI);
      }
      "__quantum__qis__x__adj" => {
        let qb = parse_qubit(inst, 0);
        graph.X(qb, -PI);
      }
      "__quantum__qis__x__ctl" => {
        let control = parse_as_value(inst, 0).expect("Need control qubits.");
        let target = parse_qubit(inst, 1);
        graph.CX(control, target, PI);
      }
      "__quantum__qis__y__body" => {
        let qb = parse_qubit(inst, 0);
        graph.Y(qb, PI);
      }
      "__quantum__qis__y__adj" => {
        let qb = parse_qubit(inst, 0);
        graph.Y(qb, -PI);
      }
      "__quantum__qis__y__ctl" => {
        let control = parse_as_value(inst, 0).expect("Need control qubits.");
        let target = parse_qubit(inst, 1);
        graph.CY(control, target, PI);
      }
      "__quantum__qis__z__body" => {
        let qb = parse_qubit(inst, 0);
        graph.Z(qb, PI);
      }
      "__quantum__qis__z__adj" => {
        let qb = parse_qubit(inst, 0);
        graph.Z(qb, -PI);
      }
      "__quantum__qis__z__ctl" => {
        let control = parse_as_value(inst, 0).expect("Need control qubits.");
        let target = parse_qubit(inst, 1);
        graph.CZ(control, target, PI);
      }
      "__quantum__qis__cnot__body" => {
        let control = parse_qubit(inst, 0);
        let target = parse_qubit(inst, 1);
        graph.CX(control, target, PI);
      }
      "__quantum__qis__rx__body" => {
        let qubit = parse_as_value(inst, 1).expect("Can't find a qubit.");
        let rotation = parse_as_value(inst, 0).expect("Can't find a rotation.");
        graph.R(Value::Pauli(Pauli::X), qubit, rotation);
      }
      "__quantum__qis__rx__adj" => {
        let qubit = parse_as_value(inst, 0).expect("Can't find a qubit.");
        let rotation = parse_as_value(inst, 1).expect("Can't find a rotation.");
        let throwaway = context.next_throwaway();

        graph.Expression(Expression::NegateSign(rotation), Some(throwaway.clone()));
        graph.R(Value::Pauli(Pauli::X), qubit, Value::Ref(throwaway, None));
      }
      "__quantum__qis__rx__ctl" => {
        let controls = parse_as_value(inst, 0).expect("Can't find controls.");
        let (pauli, target, rotation) = expand_arg_tuple(1);
        graph.CR(
          Value::Pauli(Pauli::X),
          controls,
          Value::Ref(target, None),
          Value::Ref(rotation, None)
        );
      }
      "__quantum__qis__rx__ctladj" => {
        let controls = parse_as_value(inst, 0).expect("Can't find controls.");
        let (pauli, target, rotation) = expand_arg_tuple(1);
        let throwaway = context.next_throwaway();

        graph.Expression(
          Expression::NegateSign(Value::Ref(rotation, None)),
          Some(throwaway.clone())
        );
        graph.CR(
          Value::Pauli(Pauli::X),
          controls,
          Value::Ref(target, None),
          Value::Ref(throwaway, None)
        );
      }
      "__quantum__qis__ry__body" => {
        let qubit = parse_as_value(inst, 1).expect("Can't find a qubit.");
        let rotation = parse_as_value(inst, 0).expect("Can't find a rotation.");
        graph.R(Value::Pauli(Pauli::Y), qubit, rotation);
      }
      "__quantum__qis__ry__adj" => {
        let qubit = parse_as_value(inst, 0).expect("Can't find a qubit.");
        let rotation = parse_as_value(inst, 1).expect("Can't find a rotation.");
        let throwaway = context.next_throwaway();

        graph.Expression(Expression::NegateSign(rotation), Some(throwaway.clone()));
        graph.R(Value::Pauli(Pauli::Y), qubit, Value::Ref(throwaway, None));
      }
      "__quantum__qis__ry__ctl" => {
        let controls = parse_as_value(inst, 0).expect("Can't find controls.");
        let (pauli, target, rotation) = expand_arg_tuple(1);
        graph.CR(
          Value::Pauli(Pauli::Y),
          controls,
          Value::Ref(target, None),
          Value::Ref(rotation, None)
        );
      }
      "__quantum__qis__ry__ctladj" => {
        let controls = parse_as_value(inst, 0).expect("Can't find controls.");
        let (pauli, target, rotation) = expand_arg_tuple(1);
        let throwaway = context.next_throwaway();

        graph.Expression(
          Expression::NegateSign(Value::Ref(rotation, None)),
          Some(throwaway.clone())
        );
        graph.CR(
          Value::Pauli(Pauli::Y),
          controls,
          Value::Ref(target, None),
          Value::Ref(throwaway, None)
        );
      }
      "__quantum__qis__rz__body" => {
        let qubit = parse_as_value(inst, 1).expect("Can't find a qubit.");
        let rotation = parse_as_value(inst, 0).expect("Can't find a rotation.");
        graph.R(Value::Pauli(Pauli::Z), qubit, rotation);
      }
      "__quantum__qis__rz__adj" => {
        let qubit = parse_as_value(inst, 0).expect("Can't find a qubit.");
        let rotation = parse_as_value(inst, 1).expect("Can't find a rotation.");
        let throwaway = context.next_throwaway();

        graph.Expression(Expression::NegateSign(rotation), Some(throwaway.clone()));
        graph.R(Value::Pauli(Pauli::Z), qubit, Value::Ref(throwaway, None));
      }
      "__quantum__qis__rz__ctl" => {
        let controls = parse_as_value(inst, 0).expect("Can't find controls.");
        let (pauli, target, rotation) = expand_arg_tuple(1);
        graph.CR(
          Value::Pauli(Pauli::Z),
          controls,
          Value::Ref(target, None),
          Value::Ref(rotation, None)
        );
      }
      "__quantum__qis__rz__ctladj" => {
        let controls = parse_as_value(inst, 0).expect("Can't find controls.");
        let (pauli, target, rotation) = expand_arg_tuple(1);
        let throwaway = context.next_throwaway();

        graph.Expression(
          Expression::NegateSign(Value::Ref(rotation, None)),
          Some(throwaway.clone())
        );
        graph.CR(
          Value::Pauli(Pauli::Z),
          controls,
          Value::Ref(target, None),
          Value::Ref(throwaway, None)
        );
      }
      "__quantum__qis__measure__body" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        let bases = parse_as_value(inst, 0).expect("Can't resolve measure basis.");
        let qubits = parse_as_value(inst, 1).expect("Can't resolve measure qubits.");

        graph.Measure(bases, qubits, Value::String(ref_id));
      }
      "__quantum__qis__m__body" | "__quantum__qis__mz__body" => {
        let target_value = if let Some(val) = parse_ref_id_from_instruction(inst.borrow()) {
          Value::String(val)
        } else {
          parse_as_value(inst, 1).expect("Can't find result register.")
        };

        let qb = parse_qubit(inst, 0);
        graph.Measure(Value::Pauli(Pauli::Z), qb, target_value);
      }

      // Slightly non-official QIR, measure then reset.
      "__quantum__qis__mresetz__body" => {
        let target_value = if let Some(val) = parse_ref_id_from_instruction(inst.borrow()) {
          Value::String(val)
        } else {
          parse_as_value(inst, 1).expect("Can't find result register.")
        };

        let qb = parse_qubit(inst, 0);
        graph.Measure(Value::Pauli(Pauli::Z), qb.clone(), target_value);
        graph.Reset(qb);
      }

      "__quantum__qis__cx__body" => {
        let control = parse_qubit(inst, 0);
        let target = parse_qubit(inst, 1);
        graph.CR(Value::Pauli(Pauli::X), control, target, Value::Float(PI));
      }
      "__quantum__qis__cz__body" => {
        let control = parse_qubit(inst, 0);
        let target = parse_qubit(inst, 1);
        graph.CR(Value::Pauli(Pauli::Z), control, target, Value::Float(PI));
      }
      "__quantum__qis__ccx__body" => {
        let control_one = parse_qubit(inst, 0);
        let control_two = parse_qubit(inst, 1);
        let target = parse_qubit(inst, 2);
        graph.CR(
          Value::Pauli(Pauli::Z),
          Value::Array(vec![Ptr::from(control_one), Ptr::from(control_two)]),
          target,
          Value::Float(PI)
        );
      }

      // Results/initialize
      "__quantum__rt__initialize" => {
        graph.Initialize();
      }
      "__quantum__rt__fail" => {
        let message = parse_as_value(inst, 0);
        graph.Throw(message);
      }
      "__quantum__rt__result_record_output" => {
        // Base profiles only have one method, so don't need to care about child
        // contexts.
        if !context.is_base_profile.deref() {
          with_mutable!(context.is_base_profile.expand_into(&Ptr::from(true)));
        }
      }
      "__quantum__rt__string_equal" | "__quantum__rt__result_equal" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        let left = parse_as_value(inst, 0).expect("Left comparison result unresolvable.");
        let right = parse_as_value(inst, 1).expect("Right comparison result unresolvable.");

        graph.Condition(ref_id, left, Equalities::Equals, right);
      }
      "__quantum__rt__result_get_one" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        graph.Assign(ref_id, Value::Int(1));
      }
      "__quantum__rt__result_get_zero" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        graph.Assign(ref_id, Value::Int(0));
      }
      "__quantum__rt__result_to_string" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        let val = parse_as_value(inst, 0).expect("Can't resolve value.");
        graph.Expression(Expression::Stringify(val), Some(ref_id));
      }

      // Qubit operations
      "__quantum__rt__qubit_allocate" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        graph.ActivateQubit(ref_id, None);
      }
      "__quantum__rt__qubit_allocate_array" => {
        let qubit_numbers = parse_as_value(inst, 0).expect("Qubit array count unresolved.");
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        graph.ActivateQubit(ref_id, Some(qubit_numbers));
      }
      "__quantum__rt__qubit_release" | "__quantum__rt__qubit_release_array" => {
        let deactivated_qubit = parse_qubit(inst, 0);
        graph.DeactivateQubit(deactivated_qubit);
      }

      // General utilities
      "__quantum__rt__message" => {
        let log_message = parse_as_value(inst, 0).expect("Can't find message value.");
        graph.Log(log_message);
      }

      // Array operators. Hope these all go at some point.
      "__quantum__rt__array_copy" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        let copy_target = parse_as_value(inst, 0).expect("Should be a reference.");
        graph.Expression(Expression::Clone(copy_target), Some(ref_id));
      }
      // __quantum__rt__memory_allocate is an undocumented method that just assigns memory for a tuple, seemingly.
      // As it's undocumented and rarely used we'll just treat it as an array assignment for now.
      "__quantum__rt__array_create"
      | "__quantum__rt__array_create_1d"
      | "__quantum__rt__memory_allocate" => {
        // We don't care about sizes, we dynamically allocate them anyway.
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        graph.Assign(ref_id, Value::Array(Vec::new()));
      }
      "__quantum__rt__array_get_element_ptr" | "__quantum__rt__array_get_element_ptr_1d" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        let new_result: Vec<Ptr<Value>> = Vec::new();

        let target = parse_as_value(inst, 0)
          .expect("Target of array access unresolvable.")
          .as_reference();
        let index = parse_as_value(inst, 1).expect("Index unresolvable.");

        graph.Assign(ref_id.clone(), Value::Ref(target.0, Some(Ptr::from(index))));
      }
      "__quantum__rt__array_get_size" | "__quantum__rt__array_get_size_1d" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        let length_target = parse_as_value(inst, 0).expect("Should be a reference.");
        graph.Expression(Expression::Length(length_target), Some(ref_id));
      }
      "__quantum__rt__callable_copy" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        let val = parse_as_value(inst, 0).expect("Can't resolve value.");

        // No need to copy anything, we don't assign state.
        graph.Assign(ref_id, val);
      }
      "__quantum__rt__callable_create" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        let method = parse_as_value(inst, 0).expect("Can't find global callable array.");
        let callable_lambda = if let Value::Ref(call, _) = method.borrow() {
          parse_default_callable(call)
        } else {
          None
        };

        if let Some(lambda) = callable_lambda {
          let stored = parse_as_value(inst, 2).expect("Can't find stored value.");
          let mut subgraph = CallableAnalysisGraph::new(&lambda);
          subgraph
            .argument_mappings
            .insert("%capture-tuple".to_string(), Ptr::from(stored));
          subgraph.argument_mappings.insert(
            "%result-tuple".to_string(),
            Ptr::from(Value::Array(Vec::new()))
          );
          graph.Assign(ref_id, Value::Callable(Ptr::from(subgraph)));
        } else {
          panic!("Unable to resolve callable initialization.");
        }
      }
      "__quantum__rt__callable_invoke" => {
        let method = parse_as_value(inst, 0).expect("Can't find callable.");
        let args = parse_as_value(inst, 1).expect("Can't find argument.");
        let results = parse_as_value(inst, 2).expect("Can't find results.");

        let results = match results {
          Value::Ref(ref_, _) => Some(ref_.clone()),
          _ => None
        };

        graph.Expression(
          Expression::ArgInjection(
            method.clone(),
            if args == Value::Empty {
              None
            } else {
              Some(args)
            }
          ),
          None
        );

        // Call a subgraph with our dynamic callable.
        graph.Subgraph(method, results);
      }
      "__quantum__rt__callable_make_adjoint" => {
        let method = parse_as_value(inst, 0).expect("Can't find callable.");
        graph.Expression(Expression::MakeCtrlAdj(method, LambdaModifier::Adj), None);
      }
      "__quantum__rt__callable_make_controlled" => {
        let method = parse_as_value(inst, 0).expect("Can't find callable.");
        graph.Expression(Expression::MakeCtrlAdj(method, LambdaModifier::Ctl), None);
      }

      // We ignore alias counts, no need.
      "__quantum__rt__callable_update_alias_count"
      | "__quantum__rt__callable_update_reference_count"
      | "__quantum__rt__capture_update_alias_count"
      | "__quantum__rt__capture_update_reference_count"
      | "__quantum__rt__array_update_alias_count"
      | "__quantum__rt__array_update_reference_count"
      | "__quantum__rt__result_update_reference_count"
      | "__quantum__rt__string_update_reference_count"
      | "__quantum__rt__tuple_update_alias_count"
      | "__quantum__rt__bigint_update_reference_count"
      | "__quantum__rt__tuple_update_reference_count" => {}

      // All to-string operations are the same for us, just stringify the value.
      "__quantum__rt__bool_to_string"
      | "__quantum__rt__bigint_to_string"
      | "__quantum__rt__double_to_string"
      | "__quantum__rt__int_to_string"
      | "__quantum__rt__pauli_to_string"
      | "__quantum__rt__qubit_to_string"
      | "__quantum__rt__range_to_string" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        let val = parse_as_value(inst, 0).expect("Can't resolve value.");
        graph.Expression(Expression::Stringify(val), Some(ref_id));
      }
      "__quantum__rt__string_concatenate" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        let left_string = parse_as_value(inst, 0).expect("Can't resolve string value.");
        let right_string = parse_as_value(inst, 1).expect("Can't resolve string value.");

        graph.Arithmatic(ref_id, left_string, Operator::Add, right_string);
      }
      "__quantum__rt__string_create" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        let created_string =
          parse_as_value(inst, 0).expect("Can't resolve string creation target.");

        graph.Assign(ref_id, created_string);
      }
      "__quantum__rt__tuple_create" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        graph.Assign(ref_id, Value::Array(Vec::new()));
      }

      "llvm.powi.f64.i32" | "llvm.powi.f64.i64" => {
        let ref_id = get_ref_id_from_instruction(inst.borrow());
        let value = parse_as_value(inst, 0).expect("Can't resolve string value.");
        let power_multiplier = parse_as_value(inst, 1).expect("Can't resolve string value.");
        graph.Arithmatic(ref_id, value, Operator::PowerOf, power_multiplier);
      }

      // Output recording doesn't matter for us.
      "__quantum__rt__tuple_record_output" | "__quantum__rt__array_record_output" => {}

      // Bigint support that hopefully we'll just be able to ignore.
      "__quantum__rt__bigint_add"
      | "__quantum__rt__bigint_bitand"
      | "__quantum__rt__bigint_bitnot"
      | "__quantum__rt__bigint_bitor"
      | "__quantum__rt__bigint_bitxor"
      | "__quantum__rt__bigint_create_array"
      | "__quantum__rt__bigint_create_i64"
      | "__quantum__rt__bigint_divide"
      | "__quantum__rt__bigint_equal"
      | "__quantum__rt__bigint_get_data"
      | "__quantum__rt__bigint_get_length"
      | "__quantum__rt__bigint_greater"
      | "__quantum__rt__bigint_greater_eq"
      | "__quantum__rt__bigint_modulus"
      | "__quantum__rt__bigint_multiply"
      | "__quantum__rt__bigint_negate"
      | "__quantum__rt__bigint_power"
      | "__quantum__rt__bigint_shiftleft"
      | "__quantum__rt__bigint_shiftright"
      | "__quantum__rt__bigint_subtract"
      | "__quantum__rt__array_project"
      | "__quantum__rt__array_slice"
      | "__quantum__rt__array_slice_1d"
      | "__quantum__rt__array_get_dim"
      | "__quantum__rt__array_concatenate"
      | "__quantum__rt__string_get_data"
      | "__quantum__rt__string_get_length"
      | "__quantum__rt__tuple_copy"
      | _ => {
        warn!("Attempted to process unknown intrinsic {}.", name);
      }
    }

    None
  }

  fn eval_call(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let method_name = self
      .get_method_name(inst.borrow())
      .expect("Can't resolve method name of call operation.");
    let called_func = context.module.get_function(method_name.as_str());
    if called_func.is_none() || called_func.unwrap().get_basic_blocks().is_empty() {
      self.eval_intrinsic(method_name, inst, graph, context);
    } else {
      let func = called_func.unwrap();

      let mut args = HashMap::new();
      let mut index = 0;
      let loops = inst.get_num_operands() - 1;
      while index < loops {
        let param = func.get_nth_param(index).unwrap().to_string();
        let param_ref_id = get_ref_id_from_value(&param);
        let value = self
          .as_value_ptr(operand_to_value!(inst, index), graph, context)
          .expect("Unable to resolve value.");
        args.insert(param_ref_id, value);
        index += 1;
      }

      let builder = self.walk_function(func.borrow(), context);
      let mut subgraph = Ptr::from(CallableAnalysisGraph::new(&builder.graph));

      // Add specific args to this particular call.
      subgraph.argument_mappings = args;

      let target_var = parse_ref_id_from_instruction(inst);
      graph.Subgraph(Value::Callable(subgraph), target_var);
    }
  }

  fn eval_int_to_ptr(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let val = self
      .as_value(inst.as_any_value_enum().borrow_mut(), graph, context)
      .expect("Int to pointer unresolvable.");
    let ref_id = get_ref_id_from_instruction(inst.borrow());
    graph.Assign(ref_id, val);
  }

  /// We implicitly convert types on use, so as long as they aren't wildly different no need for static casts.
  fn eval_numeric_cast(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let val = self
      .as_value(operand_to_value!(inst, 0), graph, context)
      .expect("Int to pointer unresolvable.");
    let ref_id = get_ref_id_from_instruction(inst.borrow());
    graph.Assign(ref_id, val);
  }

  fn eval_ext(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let val = self
      .as_value(operand_to_value!(inst, 0), graph, context)
      .expect("ext value unresolvable.");
    let ref_id = get_ref_id_from_instruction(inst.borrow());
    graph.Assign(ref_id, val);
  }

  fn eval_bitcast(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let val = self
      .as_value(operand_to_value!(inst, 0), graph, context)
      .expect("Bitcast value unresolvable.");
    let ref_id = get_ref_id_from_instruction(inst.borrow());
    graph.Assign(ref_id, val);
  }

  fn eval_trunc(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let val = self
      .as_value(operand_to_value!(inst, 0), graph, context)
      .expect("Truncate value unresolvable.");
    let ref_id = get_ref_id_from_instruction(inst.borrow());
    graph.Assign(ref_id, val);
  }

  /// Load is meaningless for us, as is alignment and memory metadata. Just treat it as an assign.
  fn eval_load(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let val = self
      .as_value(operand_to_value!(inst, 0), graph, context)
      .expect("Load value unresolvable.");
    let ref_id = get_ref_id_from_instruction(inst.borrow());
    graph.Assign(ref_id, val);
  }

  fn eval_store(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let value = self
      .as_value(operand_to_value!(inst, 0), graph, context)
      .expect("Store value unresolvable.");

    // The argument resolves to another variable which we want to just directly assign too.
    let target_variable_str = get_ref_id_from_instruction(
      inst
        .get_operand(1)
        .unwrap()
        .left()
        .unwrap()
        .as_instruction_value()
        .expect("Has to be storing in another variable.")
        .borrow()
    );
    graph.Assign(target_variable_str, value);
  }

  fn eval_branch(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let op_count = inst.get_num_operands();
    let branch_basic_block = inst
      .get_parent()
      .unwrap()
      .get_name()
      .to_str()
      .unwrap()
      .to_string();
    let last_node = with_mutable!(graph.auto_attach_target.borrow_mut());

    // Walks the branches outwards-going edge to work out what values the phi node would have
    // assigned, and then turn those into our own edge assignments instead.
    //
    // This means phi nodes don't really have any evaluation, because all branches will be
    // dealing with the conditional themselves.
    let get_assignment = |bb: &BasicBlock| -> Option<Vec<(String, Value)>> {
      let mut results = Vec::new();
      let mut potential_phi = bb.get_first_instruction();
      let mut is_phi = true;
      while is_phi {
        if let Some(phi) = potential_phi {
          match phi.get_opcode() {
            InstructionOpcode::Phi => {
              let inst_string = phi.to_string();

              // Do a dirty match to find the basic block names.
              lazy_static! {
                static ref bb_finder: Regex = Regex::new(", %([^]]+?)]+").unwrap();
              }
              let capture_groups: Vec<String> = bb_finder
                .captures_iter(inst_string.as_str())
                .map(|val| val.get(1).unwrap().as_str().trim().to_string())
                .collect();

              // The value in the operand is the instruction linking to the value that gets
              // assigned if we're coming from a particular basic-block. So find the assignment
              // that is for the branch we're currently looking at and return it.
              let ref_id = get_ref_id_from_instruction(phi.borrow());
              let operands = phi.get_num_operands();
              let mut i = 0;
              while i < operands {
                let basic_block = capture_groups
                  .get(i as usize)
                  .expect("Can't find the name of the basic block.")
                  .clone();
                if basic_block == branch_basic_block {
                  let val = self
                    .as_value(operand_to_value!(phi, i), graph, context)
                    .expect("Can't resolve phi node references.");
                  results.push((ref_id, val.clone()));
                  break;
                }
                i += 1;
              }
            }
            _ => {
              is_phi = false;
            }
          };

          potential_phi = phi.get_next_instruction();
        } else {
          is_phi = false;
        }
      }

      if results.is_empty() {
        None
      } else {
        Some(results)
      }
    };

    // Unconditional.
    if op_count == 1 {
      let basic_block = operand_to_bb!(inst, 0);
      let target = basic_block.get_name().to_str().unwrap().to_string();
      let target = with_mutable!(context
        .anchors
        .get_mut(target.as_str())
        .expect("Node should exist."));
      let assignments = get_assignment(basic_block.borrow());
      with_mutable!(graph.add_edge(last_node.borrow_mut(), target, assignments, None));
    } else {
      // Conditions 'seem' to always be a reference to another result, this just casts it to a bool.
      // But can't discount just having a flat true/false value.
      let condition = self
        .as_value(operand_to_value!(inst, 0), graph, context)
        .expect("Conditional unable to be evaluated for branch.");
      let false_block = operand_to_bb!(inst, 1);
      let true_block = operand_to_bb!(inst, 2);

      let true_name = true_block.get_name().to_str().unwrap();
      let false_name = false_block.get_name().to_str().unwrap();

      let true_branch = with_mutable!(context.anchors.get_mut(true_name).expect("Should exist."));
      let false_branch = with_mutable!(context.anchors.get_mut(false_name).expect("Should exist."));

      let true_assignments = get_assignment(true_block.borrow());
      let false_assignments = get_assignment(false_block.borrow());

      // We model branches as a conditional outwards edge if the condition is true, otherwise unconditional out.
      // All edge conditions should be evaluated before the unconditional, as it acts as a fall-back.
      with_mutable!(graph.add_edge(
        last_node.borrow_mut(),
        true_branch.borrow_mut(),
        true_assignments,
        Some(Condition::new(
          condition,
          Equalities::Equals,
          Value::Bool(true)
        ))
      ));
      with_mutable!(graph.add_edge(
        last_node.borrow_mut(),
        false_branch.borrow_mut(),
        false_assignments,
        None
      ));
    }
  }

  fn eval_icmp(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let operator = if let Some(pred) = inst.get_fcmp_predicate() {
      match pred {
        FloatPredicate::OEQ => Equalities::Equals,
        FloatPredicate::ONE => Equalities::NotEquals,
        FloatPredicate::UGT => Equalities::GreaterThan,
        FloatPredicate::UGE => Equalities::GreaterOrEqualThan,
        FloatPredicate::ULT => Equalities::LessThan,
        FloatPredicate::ULE => Equalities::LessOrEqualThan,
        FloatPredicate::OGT => Equalities::GreaterThan,
        FloatPredicate::OGE => Equalities::GreaterOrEqualThan,
        FloatPredicate::OLT => Equalities::LessThan,
        FloatPredicate::OLE => Equalities::LessOrEqualThan,
        _ => panic!("Untranslatable fcompare.")
      }
    } else if let Some(pred) = inst.get_icmp_predicate() {
      match pred {
        IntPredicate::EQ => Equalities::Equals,
        IntPredicate::NE => Equalities::NotEquals,
        IntPredicate::UGT => Equalities::GreaterThan,
        IntPredicate::UGE => Equalities::GreaterOrEqualThan,
        IntPredicate::ULT => Equalities::LessThan,
        IntPredicate::ULE => Equalities::LessOrEqualThan,
        IntPredicate::SGT => Equalities::GreaterThan,
        IntPredicate::SGE => Equalities::GreaterOrEqualThan,
        IntPredicate::SLT => Equalities::LessThan,
        IntPredicate::SLE => Equalities::LessOrEqualThan
      }
    } else {
      panic!("Comparison operator that looks strange.")
    };

    let left = self
      .as_value(operand_to_value!(inst, 0), graph, context)
      .expect("Can't resolve left side of icmp.");

    let right = self
      .as_value(operand_to_value!(inst, 1), graph, context)
      .expect("Can't resolve right side of icmp.");

    let ref_id = get_ref_id_from_instruction(inst.borrow());
    graph.Condition(ref_id, left, operator, right);
  }

  fn add_arithmatic_op(
    &self, op: Operator, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let lhs = self
      .as_value(operand_to_value!(inst, 0), graph, context)
      .unwrap_or_else(|| panic!("Can't resolve left side of {op}."));
    let lhs_as_int = match lhs {
      Value::Int(i) => Some(i),
      _ => None
    };

    let rhs = self
      .as_value(operand_to_value!(inst, 1), graph, context)
      .unwrap_or_else(|| panic!("Can't resolve right side of {op}."));
    let rhs_as_int = match rhs {
      Value::Int(i) => Some(i),
      _ => None
    };

    let ref_id = get_ref_id_from_instruction(inst.borrow());
    graph.Arithmatic(ref_id, lhs, op, rhs);
  }

  fn eval_mul(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    self.add_arithmatic_op(Operator::Multiply, inst, graph, context);
  }

  fn eval_div(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    self.add_arithmatic_op(Operator::Divide, inst, graph, context);
  }

  fn eval_sub(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    self.add_arithmatic_op(Operator::Subtract, inst, graph, context);
  }

  fn eval_add(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    self.add_arithmatic_op(Operator::Add, inst, graph, context);
  }

  fn eval_insertvalue(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    // TODO: Don't double-up stringification from get_ref_x.
    let inst_str = inst.to_string();
    let inst_str = inst_str.trim_matches('"').trim();

    let mut target_ref = get_ref_id_from_instruction(inst);
    let target_composite = self
      .as_value(operand_to_value!(inst, 0), graph, context)
      .expect("Can't resolve composite to insert into.");

    let insert_value = self
      .as_value(operand_to_value!(inst, 1), graph, context)
      .expect("Can't resolve value to insert.");

    // TODO: Have to extract indexers via regex/string comparison because they're not exposed
    //  as an operand for some reason.
    let mut index_values = Vec::new();
    for indexer in inst_str.split(',').rev() {
      let indexer = indexer.trim();
      lazy_static! {
        static ref numeric_matcher: Regex = Regex::new("^[0-9]+$").unwrap();
      }
      if numeric_matcher.is_match(indexer) {
        index_values.push(
          indexer
            .parse::<i64>()
            .unwrap_or_else(|_| panic!("Unable to parse {indexer} as an int"))
        );
      }
    }
    index_values.reverse();

    let mut throwaway_var = context.next_throwaway();

    // Assign our referenced/new object to the target variable.
    graph.Assign(target_ref.clone(), target_composite);

    // Pull out the element we want to change with an indexer reference.
    for index in index_values {
      graph.Assign(
        throwaway_var.clone(),
        Value::Ref(target_ref, Some(Ptr::from(Value::Int(index))))
      );
      target_ref = throwaway_var;
      throwaway_var = context.next_throwaway();
    }

    // Directly change that element with the value we want to insert.
    graph.Assign(target_ref, insert_value);
  }

  fn eval_extractvalue(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    // TODO: Don't double-up stringification from get_ref_x.
    let inst_str = inst.to_string();
    let inst_str = inst_str.trim_matches('"').trim();

    let target_ref = get_ref_id_from_instruction(inst);
    let target_composite = self
      .as_value(operand_to_value!(inst, 0), graph, context)
      .expect("Can't resolve composite to extract from.");

    // TODO: Same as insertvalue, find way around this.
    let mut index_values = Vec::new();
    for indexer in inst_str.split(',').rev() {
      let indexer = indexer.trim();
      lazy_static! {
        static ref numeric_matcher: Regex = Regex::new("^[0-9]+$").unwrap();
      }
      if numeric_matcher.is_match(indexer) {
        index_values.push(
          indexer
            .parse::<i64>()
            .unwrap_or_else(|_| panic!("Unable to parse {indexer} as an int"))
        );
      }
    }
    index_values.reverse();

    let mut throwaway_var = context.next_throwaway();
    graph.Assign(throwaway_var.clone(), target_composite);

    // Pull out the element we want to change with an indexer reference.
    for index in index_values {
      let next_throwaway = context.next_throwaway();
      graph.Assign(
        next_throwaway.clone(),
        Value::Ref(throwaway_var, Some(Ptr::from(Value::Int(index))))
      );
      throwaway_var = next_throwaway;
    }

    // Directly extract from our composite the object we want.
    graph.Assign(target_ref.clone(), Value::Ref(throwaway_var, None));
  }

  /// The GEP instruction is special in that it only deals with pointer addresses, nothing more.
  /// This becomes very simple for us because address == the object itself in our model of the
  /// world, so we just chain indexer operations repeatedly on the same object and let the
  /// runtime resolve the type nuances.
  fn eval_getelementptr(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let target_ref = get_ref_id_from_instruction(inst);
    let extracted_ref = self
      .extract_gep(inst.as_any_value_enum().borrow(), graph, context)
      .expect("Couldn't extract getelementptr instruction.");

    graph.Assign(target_ref.clone(), extracted_ref);
  }

  fn eval_select(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let target_ref = get_ref_id_from_instruction(inst);
    let condition = self
      .as_value(operand_to_value!(inst, 0), graph, context)
      .expect("Couldn't get target.");

    let true_value = self
      .as_value(operand_to_value!(inst, 1), graph, context)
      .expect("Couldn't get true select value.");

    let false_value = self
      .as_value(operand_to_value!(inst, 2), graph, context)
      .expect("Couldn't get false select value.");

    // We just use edge-assignments to simulate the select. So we have two edges that connect these two nodes - one which has a
    // condition, the other which is the default path, assigning correctly along each.
    let mut last_node = graph.next_auto_attach().clone();
    let attach_node = Ptr::from(Instruction::NoOp);
    let mut added_node = with_mutable!(graph.add_with_edge(
      &attach_node,
      &mut last_node,
      Some(vec![(target_ref.clone(), false_value)]),
      None
    ));

    with_mutable!(graph.add_edge(
      &mut last_node,
      &mut added_node,
      Some(vec![(target_ref, true_value)]),
      Some(Condition::new(
        condition,
        Equalities::Equals,
        Value::Bool(true)
      ))
    ));
  }

  fn eval_bitwise(
    &self, op: Operator, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let target_ref = get_ref_id_from_instruction(inst);
    let lhs = self
      .as_value(operand_to_value!(inst, 0), graph, context)
      .expect("Couldn't get true select value.");

    let rhs = self
      .as_value(operand_to_value!(inst, 1), graph, context)
      .expect("Couldn't get false select value.");

    graph.Arithmatic(target_ref, lhs, op, rhs);
  }

  fn eval_or(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    self.eval_bitwise(Operator::Or, inst, graph, context);
  }

  fn eval_and(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    self.eval_bitwise(Operator::And, inst, graph, context);
  }

  fn eval_xor(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    self.eval_bitwise(Operator::Xor, inst, graph, context);
  }

  fn eval_ret(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    if inst.get_num_operands() == 1 {
      let results = self
        .as_value(operand_to_value!(inst, 0), graph, context)
        .expect("Can't resolve result.");

      graph.Return(results);
    }
  }

  fn eval_fneg(
    &self, inst: &Ptr<InstructionValue>, graph: &Ptr<AnalysisGraphBuilder>,
    context: &Ptr<EvaluationContext>
  ) {
    let target = get_ref_id_from_instruction(inst);
    let value = self
      .as_value(operand_to_value!(inst, 0), graph, context)
      .expect("Can't resolve float for sign-flip.");

    graph.Expression(Expression::NegateSign(value), Some(target));
  }
}
