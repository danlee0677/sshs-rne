import numpy as np

def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def pad_to_pow2_square(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.complex128)
    n = M.shape[0]
    N = next_pow2(n)
    if N == n:
        return M
    Mp = np.zeros((N, N), dtype=np.complex128)
    Mp[:n, :n] = M
    return Mp

def pad_vec_to_len(v: np.ndarray, N: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.complex128).reshape(-1)
    out = np.zeros((N,), dtype=np.complex128)
    out[: len(v)] = v
    return out

def estimate_kappa_from_eigs(H: np.ndarray, tol: float = 1e-10) -> float:
    # H is Hermitian (or numerically close)
    w = np.linalg.eigvalsh((H + H.conj().T) / 2)
    aw = np.sort(np.abs(w))
    aw = aw[aw > tol]
    if len(aw) == 0:
        return 1.0
    return float(aw[-1] / aw[0])
