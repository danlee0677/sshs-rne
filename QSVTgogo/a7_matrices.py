import numpy as np

def make_A7_full(P: int, coeff_c: np.ndarray, coeff_s: np.ndarray) -> np.ndarray:
    coeff_c = np.asarray(coeff_c, dtype=float).reshape(-1)
    coeff_s = np.asarray(coeff_s, dtype=float).reshape(-1)
    assert coeff_c.shape[0] == P - 1
    assert coeff_s.shape[0] == P - 1

    A_cos = np.zeros((P - 1, P + 1), dtype=float)
    A_sin = np.zeros((P - 1, P + 1), dtype=float)

    for i in range(P - 1):
        A_cos[i, 0] = coeff_c[i]
        A_cos[i, i + 1] = -coeff_c[i]
        A_cos[i, P] = -coeff_c[i]

        A_sin[i, 0] = coeff_s[i]
        A_sin[i, i + 1] = -coeff_s[i]
        A_sin[i, P] = -coeff_s[i]

    return np.vstack([A_cos, A_sin])

def hermitian_embedding(A: np.ndarray) -> np.ndarray:
    """
    Rectangular A (m x n) -> Hermitian H (m+n x m+n):
        H = [[0, A],
             [A†, 0]]
    """
    A = np.asarray(A, dtype=np.complex128)
    m, n = A.shape
    Zm = np.zeros((m, m), dtype=np.complex128)
    Zn = np.zeros((n, n), dtype=np.complex128)
    top = np.hstack([Zm, A])
    bot = np.hstack([A.conj().T, Zn])
    return np.vstack([top, bot])

def make_consistent_b(A: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Make a consistent system Ax=b by sampling x_true and setting b=A x_true.
    """
    A = np.asarray(A, dtype=np.complex128)
    n = A.shape[1]
    x_true = rng.normal(size=n) + 0j
    b = A @ x_true
    return b, x_true

def embed_linear_system(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve Ax=b via Hy = [b;0], where H=[[0,A],[A†,0]].
    If exact consistency holds and solution is in column space, y should have form [0; x].
    """
    A = np.asarray(A, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128).reshape(-1)
    m, n = A.shape
    H = hermitian_embedding(A)
    bH = np.concatenate([b, np.zeros(n, dtype=np.complex128)], axis=0)
    return H, bH
