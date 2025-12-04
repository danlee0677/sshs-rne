from qiskit import transpile
from qiskit_aer import AerSimulator

def report_gate_stats(qc, basis_gates=None, opt=1):
    backend = AerSimulator()
    tqc = transpile(qc, backend, optimization_level=opt, basis_gates=basis_gates)
    return {
        "depth": tqc.depth(),
        "count_ops": dict(tqc.count_ops()),
        "num_qubits": tqc.num_qubits,
    }
