import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

from a7_matrices import make_A7_full, make_consistent_b, embed_linear_system
from utils import pad_to_pow2_square, pad_vec_to_len, estimate_kappa_from_eigs
from block_encoding import halmos_block_encode
from qsvt_core import qsp_phases_for_pinv, build_qsvt_from_phases
from amplitude_amplification import build_U_total, build_amplitude_amplified, success_prob_anc0
from analyze import report_gate_stats


def run_statevector_and_postselect(qc: QuantumCircuit, anc_qubit: int, out_dim: int):
    backend = AerSimulator(method="statevector")
    tqc = transpile(qc, backend, optimization_level=1)
    tqc.save_statevector()
    sv = np.array(backend.run(tqc).result().get_statevector(tqc), dtype=np.complex128)

    n = qc.num_qubits
    total_dim = 2**n

    # anc=0 브랜치의 이론적 길이 = 2**(n-1)
    branch_dim = 2**(n - 1)

    anc0 = np.zeros((branch_dim,), dtype=np.complex128)
    j = 0
    for idx in range(total_dim):
        bit = (idx >> anc_qubit) & 1
        if bit == 0:
            anc0[j] = sv[idx]
            j += 1

    p = float(np.vdot(anc0, anc0).real)

    # out_dim 이 branch_dim보다 크면 안 되니까 min()으로 방어
    out_len = min(out_dim, branch_dim)
    return anc0[:out_len], p




def main():
    # ===== user-level knobs =====
    P = 8               # try 8, 12, 16
    degree = 31           # QSP polynomial degree
    max_scale = 0.95
    aa_rounds = 1         # try 0,1,2
    seed = 0
    # ===========================

    rng = np.random.default_rng(seed)
    coeff_c = rng.normal(size=P - 1)
    coeff_s = rng.normal(size=P - 1)

    A = make_A7_full(P, coeff_c, coeff_s)              # shape: m x n
    b, x_true = make_consistent_b(A, rng)              # ensure Ax=b is consistent

    # Hermitian embedding: solve Hy = [b;0]
    H, bH = embed_linear_system(A, b)                  # H is (m+n)x(m+n)
    H = (H + H.conj().T) / 2                           # numeric Hermitian

    # pad to power-of-2 dimension for qubits
    Hp = pad_to_pow2_square(H)
    N = Hp.shape[0]
    bHp = pad_vec_to_len(bH, N)
    bHp = bHp / np.linalg.norm(bHp)

    # kappa estimate on Hp
    kappa = estimate_kappa_from_eigs(Hp, tol=1e-10) * 1.05
    kappa = max(kappa, 1.0)

    # Halmos block-encoding for Hp
    alpha, U_block = halmos_block_encode(Hp, alpha=None, reunitarize=True)

    # QSVT phases for bounded pseudoinverse on scaled spectrum x = λ/alpha
    # (QSVT will approximate g(x)=1/(kappa x) for |x|>=1/kappa and 0 otherwise)
    print('hihi')
    phases = qsp_phases_for_pinv(kappa=kappa, degree=degree, max_scale=max_scale)

    # Build QSVT over qubiterate W = (2Π-I)U with Π=|0><0| on Halmos ancilla
    # U_block dimension is 2N, so total qubits = log2(2N) = log2(N)+1
    dimU = U_block.shape[0]
    nq = int(np.log2(dimU))
    assert 2**nq == dimU

    # We choose Halmos ancilla as the top/most-significant logical qubit, but in our circuit it's just a qubit index.
    # For Halmos dilation, ancilla corresponds to the extra doubling dimension. We place it as the last qubit index for convenience.
    # Because we use UnitaryGate on the full matrix, the mapping is consistent with that index.
    anc = nq - 1

    qsvt = build_qsvt_from_phases(U_block=U_block, phases=phases, anc=anc)

    # U_total = state-prep(|bH>) on system subspace + QSVT
    # System qubits are all except anc (we store bHp in the bottom block inside Hp’s space; Halmos anc starts at |0>)
    sys_qubits = list(range(nq - 1))
    U_total = build_U_total(statevec=bHp, qsvt_circ=qsvt, sys_qubits=sys_qubits)

    # success prob before AA
    p0 = success_prob_anc0(U_total, anc_qubit=anc)
    print('hihi')
    

    # add amplitude amplification (optional)
    if aa_rounds > 0:
        U_amp = build_amplitude_amplified(U_total, anc_qubit=anc, rounds=aa_rounds)
    else:
        U_amp = U_total

    p1 = success_prob_anc0(U_amp, anc_qubit=anc)

    # postselect anc=0 and recover y (in Hp-space). Then extract x from y’s lower block
    y_vec, p_post = run_statevector_and_postselect(U_amp, anc_qubit=anc, out_dim=Hp.shape[0])

    # Classical reference: least squares on Ax=b
    x_ls = np.linalg.lstsq(A, b, rcond=None)[0]
    print('hihi')

    # From embedding, true y should be approx [0; x]
    m = A.shape[0]
    n = A.shape[1]
    y_target = np.concatenate([np.zeros(m, dtype=np.complex128), x_ls], axis=0)
    y_target = pad_vec_to_len(y_target, N)

    # Best-fit scaling (QSVT gives scaled inverse; this removes global scale ambiguity)
    scale = (np.vdot(y_vec, y_target) / (np.vdot(y_vec, y_vec) + 1e-18))
    y_hat = scale * y_vec

    x_hat = y_hat[m : m + n]
    print('hihi')

    rel_err = float(np.linalg.norm(x_hat - x_ls) / (np.linalg.norm(x_ls) + 1e-18))
    resid = float(np.linalg.norm(A @ x_hat - b) / (np.linalg.norm(b) + 1e-18))
    print('hihi')

    # gate stats
    stats_qsvt = report_gate_stats(U_total, basis_gates=["rz", "sx", "cx"], opt=1)
    stats_amp = report_gate_stats(U_amp, basis_gates=["rz", "sx", "cx"], opt=1)
    print('hihi')


    print("=== Dimensions ===")
    print("A shape:", A.shape, "H shape:", H.shape, "Hp dim:", N, "U_block dim:", dimU, "qubits:", nq)
    print("alpha =", alpha, "kappa~ =", kappa, "degree =", degree)
    print()

    print("=== Success prob ===")
    print("P(anc=0) before AA =", p0)
    print("P(anc=0) after  AA =", p1, f"(rounds={aa_rounds})")
    print()

    print("=== Solution quality ===")
    print("rel_err(x_hat vs x_ls) =", rel_err)
    print("residual ||Ax-b||/||b|| =", resid)
    print()

    print("=== Gate stats (rz,sx,cx basis, opt=1) ===")
    print("[U_total] qubits =", stats_qsvt["num_qubits"], "depth =", stats_qsvt["depth"], "ops =", stats_qsvt["count_ops"])
    print("[U_amp ] qubits =", stats_amp["num_qubits"], "depth =", stats_amp["depth"], "ops =", stats_amp["count_ops"])


if __name__ == "__main__":
    main()
