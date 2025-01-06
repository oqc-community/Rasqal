import random
import os

from rasqal.runtime import RasqalRunner
from rasqal.simulators import fetch_qasm_runner

from utils import TracingRuntime

directory_path = os.path.dirname(os.path.abspath(__file__))


def read_qir_file(file_name):
    """ Reads a QIR file from our examples' folder. """
    with open(os.path.join(directory_path, "qir", file_name), "r") as f:
        return f.read()


def fetch_runner(qubit_count, tracing_runner=False):
    """
    Allows you to return a tracing runner or our QASM simulated backend.

    The tracer will print out precisely what circuits are being executed by the simulator so you can see
    precisely what's going on. It's only used for visualization though, so will be a little slower.
    """
    if tracing_runner:
        return RasqalRunner(TracingRuntime(qubit_count))
    else:
        return fetch_qasm_runner(qubit_count)


def base_profile_bell():
    """
    Runs base-profile compliant QIR equivalent to:

    operation RunBell() : Bool {
        use (a, b) = (Qubit(), Qubit());
        H(a);
        CNOT(a, b);
        let result = MeasureAllZ([a, b]);
        return IsResultOne(result);
    }
    """
    print(f"Running base-profile-bell.")
    runner = fetch_runner(4)
    results = runner.run_ll(read_qir_file("base-profile-bell.ll"))

    assert len(results) == 2
    assert results['00'] > 450
    assert results['11'] > 450


def full_bell():
    """
    Runs QIR equivalent to:

    operation RunBell() : Bool {
        use (a, b) = (Qubit(), Qubit());
        H(a);
        CNOT(a, b);
        let result = MeasureAllZ([a, b]);
        return IsResultOne(result);
    }
    """
    print(f"Running full-bell.")
    runner = fetch_runner(4)
    results = runner.run_ll(read_qir_file("full-bell.ll"))

    # IsResultOne on a qubit register is interpreted as asking which value - 0 or 1 - is
    # overwhelmingly represented in the results.
    assert results in [0, 1]


def basic_branching():
    """
    Runs QIR equivalent to:

    @EntryPoint()
    operation Run() : (Bool, Bool, Bool) {
      use reg = Qubit[3];
      ApplyToEach(H, reg);

      // This would be lowered into the backend as a quantum-level branching instruction.
      // But right now it's evaluated in-line and executed.
      if ResultAsBool(M(reg[1])) {
        H(reg[0]);
        H(reg[2]);
      }

      let result = ResultArrayAsBoolArray(MultiM(reg));
      return (result[0], result[1], result[2]);
    }
    """
    print(f"Running basic-branching.")
    runner = fetch_runner(4)
    results = runner.run_ll(read_qir_file("basic-branching.ll"))
    print(f"Returned {results}")


def basic_loops():
    """
    Runs QIR equivalent to:

    @EntryPoint()
    operation Run(numSegments: Int) : Int {
      mutable incrementing_result = 0;
      for index in 0..numSegments {
        use reg = Qubit[3] {
          ApplyToEach(H, reg);

          // Would use modulo here but need to add it to allowed instructions.
          let is_even = index == 2 || index == 4 || index == 6 || index == 8 || index == 10;
          if is_even {
            H(reg[0]);
            H(reg[2]);
          }

          let result = ResultArrayAsBoolArray(MultiM(reg));
          mutable appending = 0;
          if result[0] { set appending = appending + 1; }
          if result[1] { set appending = appending + 1; }
          if result[2] { set appending = appending + 1; }

          set incrementing_result = incrementing_result + appending;
        }
      }

      return incrementing_result * 5;
    }
    """
    print(f"Running basic-loops.")
    runner = fetch_runner(4)
    results = runner.run_ll(read_qir_file("basic-loops.ll"), [5])
    print(f"Returned {results}")


def nested_calls():
    """
    Runs QIR equivalent to:

    operation LikelyTrue() : Bool {
      use (a, b) = (Qubit(), Qubit());
      H(a);
      H(b);
      return ResultAsBool(M(a)) || ResultAsBool(M(b));
    }

    operation FakeWhile(result: Double, a: Qubit, b: Qubit, c: Qubit) : Double {
      if !LikelyTrue() {
        return result;
      }

      mutable another_result = result + 13.0;
      Rx(another_result, a);
      if LikelyTrue() { H(a); }

      Rx(another_result * another_result, b);
      if LikelyTrue() { H(b); }

      Rx(another_result * another_result/2.0 + 5.0, c);
      if LikelyTrue() { H(c); }

      return another_result;
    }

    @EntryPoint()
    operation Run() : (Double, Bool, Bool, Bool) {
      mutable result = 0.0;
      use (a, b, c) = (Qubit(), Qubit(), Qubit());

      // This would be a while LikelyTrue() but Q# doesn't allow it.
      for i in 0..20 {
        set result = FakeWhile(result, a, b, c);
      }

      return (result, ResultAsBool(M(a)), ResultAsBool(M(b)), ResultAsBool(M(c)));
    }
    """
    # Note: With quantum/expression folding active the above would execute very differently than it does currently.
    print(f"Running nested-calls.")
    runner = fetch_runner(4)
    results = runner.run_ll(read_qir_file("nested-calls.ll"))
    print(f"Returned {results}")


def qaoa():
    """
    Look at src/tests/qsharp/qaoa for the full source.

    This is one of our more complex examples and exercises almost every sort of instruction currently available.

    Note: while this was picked up as a comprehensive stress-test its use as an actual QAOA algorithm hasn't been verified.
    It likely can work as one but the default arguments which it uses do not provide a good result, so it would need
    to evolve them.
    """
    print("Running qaoa.")
    runner = fetch_runner(30)
    runner.run_ll(read_qir_file("qaoa.ll"))


base_profile_bell()
full_bell()
basic_branching()
basic_loops()
nested_calls()
qaoa()
