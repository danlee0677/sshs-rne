import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator

def phase_flip_on_all_zeros(qc: QuantumCircuit, qubits: list[int]) -> None:
    """
    Apply a phase of -1 only to |0...0> over 'qubits'. Standard construction:
      X on all, multi-controlled Z on last, X on all.
    """
    if len(qubits) == 0:
        return
    if len(qubits) == 1:
        qc.z(qubits[0])
        return

    for q in qubits:
        qc.x(q)

    target = qubits[-1]
    ctrls = qubits[:-1]
    qc.h(target)
    qc.mcx(ctrls, target, mode="noancilla")
    qc.h(target)

    for q in qubits:
        qc.x(q)

def phase_flip_on_anc0_subspace(qc: QuantumCircuit, anc_qubit: int) -> None:
    """
    Mark good states as anc=0 with a -1 phase: diag(-1, +1) = -Z (global phase ignored).
    Implement as X-Z-X on anc.
    """
    qc.x(anc_qubit)
    qc.z(anc_qubit)
    qc.x(anc_qubit)

def build_U_total(statevec: np.ndarray, qsvt_circ: QuantumCircuit, sys_qubits: list[int]) -> QuantumCircuit:
    """
    U_total = PREP(|b>) on sys_qubits, then QSVT on all qubits.
    Must be invertible => use StatePreparation (unitary) rather than initialize.
    """
    n = qsvt_circ.num_qubits
    qc = QuantumCircuit(n)

    b = np.asarray(statevec, dtype=np.complex128).reshape(-1)
    b = b / np.linalg.norm(b)
    prep = StatePreparation(b)
    qc.append(prep, sys_qubits)

    qc.compose(qsvt_circ, inplace=True)
    return qc

def grover_iterate(U_total: QuantumCircuit, anc_qubit: int) -> QuantumCircuit:
    """
    Q = - U S0 U† S_good
    where:
      S_good: phase flip on anc=0 subspace
      S0    : phase flip on |0...0> (all qubits)
    """
    n = U_total.num_qubits
    qc = QuantumCircuit(n)

    U_gate = U_total.to_gate(label="Utot")
    U_dag = U_total.inverse().to_gate(label="Utot†")

    # S_good
    phase_flip_on_anc0_subspace(qc, anc_qubit)

    # U†
    qc.append(U_dag, list(range(n)))

    # S0
    phase_flip_on_all_zeros(qc, list(range(n)))

    # U
    qc.append(U_gate, list(range(n)))

    return qc

def build_amplitude_amplified(U_total: QuantumCircuit, anc_qubit: int, rounds: int) -> QuantumCircuit:
    """
    Start from |0..0>, apply (Grover iterate)^rounds, then apply U_total once at end
    (common AA layout: apply U first, then k Grover iterates; we do that form here)
    """
    n = U_total.num_qubits
    qc = QuantumCircuit(n)

    U_gate = U_total.to_gate(label="Utot")
    qc.append(U_gate, list(range(n)))

    Q = grover_iterate(U_total, anc_qubit).to_gate(label="Q")
    for _ in range(rounds):
        qc.append(Q, list(range(n)))

    return qc

def success_prob_anc0(circ: QuantumCircuit, anc_qubit: int) -> float:
    """
    Statevector-based success probability P(anc=0).
    """
    backend = AerSimulator(method="statevector")
    tqc = transpile(circ, backend, optimization_level=1)
    tqc.save_statevector()
    sv = np.array(backend.run(tqc).result().get_statevector(tqc), dtype=np.complex128)

    n = circ.num_qubits
    # Qiskit statevector ordering: little-endian indexing.
    # anc_qubit is an index in circuit qubits; we compute probability by masking basis indices.
    p = 0.0
    for idx, amp in enumerate(sv):
        bit = (idx >> anc_qubit) & 1
        if bit == 0:
            p += (amp.conjugate() * amp).real
    return float(p)
