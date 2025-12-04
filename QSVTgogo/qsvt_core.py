import numpy as np
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from pyqsp.poly import PolyTaylorSeries

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

def target_pseudoinverse_bounded(x: np.ndarray, kappa: float) -> np.ndarray:
    """
    Bounded odd pseudoinverse-like function on [-1,1]:
      g(x)= 0                      if |x| < 1/kappa
           = 1/(kappa*x)           otherwise
    This ensures |g(x)| <= 1.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    thr = 1.0 / float(kappa)
    mask = np.abs(x) >= thr
    out[mask] = 1.0 / (float(kappa) * x[mask])
    return out

def qsp_phases_for_pinv(kappa: float, degree: int = 31, max_scale: float = 0.95) -> np.ndarray:
    def g(x):
        return target_pseudoinverse_bounded(x, kappa)

    poly = PolyTaylorSeries().taylor_series(
        func=g,
        degree=degree,
        max_scale=max_scale,
        chebyshev_basis=True,
    )

    phases, _, _ = QuantumSignalProcessingPhases(
        poly,
        method="sym_qsp",
        chebyshev_basis=True,
    )
    return np.array(phases, dtype=float)

def build_qubiterate(U_block: np.ndarray, anc: int) -> QuantumCircuit:
    """
    Standard qubiterate W = R U, where R = (2Π - I).
    For a single Halmos ancilla qubit, Π projects onto anc=0, so (2Π-I) = Z on that ancilla.
    Circuit-wise: apply U, then apply Z(anc) => W = Z_anc * U.
    """
    dim = U_block.shape[0]
    n_qubits = int(np.log2(dim))
    assert 2**n_qubits == dim

    qc = QuantumCircuit(n_qubits)
    qc.append(UnitaryGate(U_block, label="U_A"), list(range(n_qubits)))
    qc.z(anc)
    return qc

def build_qsvt_from_phases(U_block: np.ndarray, phases: np.ndarray, anc: int) -> QuantumCircuit:
    """
    Canonical QSP-on-qubiterate form:
      V = Rz(-2φ0) W Rz(-2φ1) W ... Rz(-2φ_{L-2}) W Rz(-2φ_{L-1})
    where W = (2Π-I)U is the qubiterate.
    """
    W = build_qubiterate(U_block, anc)
    W_gate = W.to_gate(label="W")

    n = W.num_qubits
    qc = QuantumCircuit(n)

    for k, phi in enumerate(phases[:-1]):
        qc.rz(-2.0 * float(phi), anc)
        qc.append(W_gate, list(range(n)))

    qc.rz(-2.0 * float(phases[-1]), anc)
    return qc
