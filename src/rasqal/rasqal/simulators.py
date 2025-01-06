# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

from typing import Dict

from qiskit import QiskitError, QuantumCircuit, transpile
from qiskit_aer.backends import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration

from .runtime import RasqalRunner
from .adaptors import BuilderAdaptor, RuntimeAdaptor


def fetch_qasm_runner(qubit_count=30):
    """
    Returns a runner which uses a QASM simulator as a runtime.
    """
    return RasqalRunner(QASMRuntime(qubit_count))


class QASMBuilder(BuilderAdaptor):
    """
    Builder which builds Qiskit quantum circuits.
    """

    def __init__(self, qubit_count: int):
        super().__init__()
        self.circuit = QuantumCircuit(qubit_count, qubit_count)
        self.shot_count = 1024
        self.bit_count = 0

    def cx(self, controls, target, theta):
        self.circuit.crx(theta, controls, target)

    def cz(self, controls, target, theta):
        self.circuit.crx(theta, controls, target)

    def cy(self, controls, target, theta):
        self.circuit.cry(theta, controls, target)

    def x(self, qubit, theta):
        self.circuit.rx(theta, qubit)

    def y(self, qubit, theta):
        self.circuit.ry(theta, qubit)

    def z(self, qubit, theta):
        self.circuit.rz(theta, qubit)

    def swap(self, qubit1, qubit2):
        self.circuit.swap(qubit1, qubit2)
        return self

    def reset(self, qubit):
        self.circuit.reset(qubit)

    def measure(self, qubit):
        self.circuit.measure(qubit, self.bit_count)
        self.bit_count = self.bit_count + 1
        return self


class QASMRuntime(RuntimeAdaptor):
    """
    Qiskit-backed runtime.
    Builds and runs a pure QASM simulation as a backend.
    """

    def __init__(self, qubit_count=30):
        self.qubit_count = qubit_count

    def execute(self, builder: QASMBuilder) -> Dict[str, int]:
        config = AerBackendConfiguration.from_dict(AerSimulator._DEFAULT_CONFIGURATION)
        config.n_qubits = self.qubit_count
        qasm_sim = AerSimulator(configuration=config)

        circuit = builder.circuit
        try:
            job = qasm_sim.run(transpile(circuit, qasm_sim), shots=builder.shot_count)
            results = job.result()
            distribution = results.get_counts()  # Used to pass in circuit, check.
        except QiskitError as e:
            raise ValueError(f"Error while attempting to build/run circuit: {str(e)}")

        removals = builder.circuit.num_qubits - builder.bit_count

        # Because qiskit needs all values up-front we just provide a maximal classical register then strim off
        # the values we aren't going to use.
        return {key[removals:]: value for key, value in distribution.items()}

    def create_builder(self) -> BuilderAdaptor:
        return QASMBuilder(self.qubit_count)

    def has_features(self, required_features):
        return True
