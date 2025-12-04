import numpy as np

def psd_sqrt(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.complex128)
    M = (M + M.conj().T) / 2
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return V @ np.diag(np.sqrt(w)) @ V.conj().T

def unitarize_polar(M: np.ndarray) -> np.ndarray:
    U, _, Vh = np.linalg.svd(M, full_matrices=False)
    return U @ Vh

def halmos_block_encode(A: np.ndarray, alpha: float | None = None, reunitarize: bool = True) -> tuple[float, np.ndarray]:
    """
    Given square A, produce unitary U of size 2n such that top-left block is A/alpha:
        U = [[A/alpha, sqrt(I - A A† / alpha^2)],
             [sqrt(I - A† A / alpha^2), -A†/alpha]]
    """
    A = np.asarray(A, dtype=np.complex128)
    n = A.shape[0]
    assert A.shape[0] == A.shape[1]

    if alpha is None:
        alpha = float(np.linalg.norm(A, 2))
        if alpha == 0:
            alpha = 1.0

    Atil = A / alpha
    I = np.eye(n, dtype=np.complex128)

    B = psd_sqrt(I - Atil @ Atil.conj().T)
    C = psd_sqrt(I - Atil.conj().T @ Atil)

    U = np.block([[Atil, B],
                  [C, -Atil.conj().T]])

    if reunitarize:
        U = unitarize_polar(U)

    return float(alpha), U
