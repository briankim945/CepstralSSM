import jax
import jax.numpy as jnp
from flax import linen as nn

# --- 1. Math Primitives: The Log-Semiring ---

def complex_lse(x, y):
    """
    Log-Sum-Exp for Complex Numbers (GOOM Addition).
    Computes log(exp(x) + exp(y)) stably.
    """
    m = jax.lax.stop_gradient(jnp.maximum(x.real, y.real))
    w = jnp.exp(x - m) + jnp.exp(y - m)
    return m + jnp.log(w)

def log_mat_mul_2x2(A_log, B_log):
    """
    Matrix Multiplication in Log-Space for 2x2 Matrices.
    C = A @ B  =>  C_ij = LSE_k(A_ik + B_kj)
    """
    # Unroll 2x2 manually for speed
    a00, a01 = A_log[..., 0, 0], A_log[..., 0, 1]
    a10, a11 = A_log[..., 1, 0], A_log[..., 1, 1]
    b00, b01 = B_log[..., 0, 0], B_log[..., 0, 1]
    b10, b11 = B_log[..., 1, 0], B_log[..., 1, 1]

    c00 = complex_lse(a00 + b00, a01 + b10)
    c01 = complex_lse(a00 + b01, a01 + b11)
    c10 = complex_lse(a10 + b00, a11 + b10)
    c11 = complex_lse(a10 + b01, a11 + b11)

    # Stack back to (..., 2, 2)
    row0 = jnp.stack([c00, c01], axis=-1)
    row1 = jnp.stack([c10, c11], axis=-1)
    return jnp.stack([row0, row1], axis=-2)

def log_mat_vec_mul_2(A_log, v_log):
    """
    Matrix-Vector Multiplication in Log-Space.
    y = A @ x => y_i = LSE_j(A_ij + v_j)
    """
    a00, a01 = A_log[..., 0, 0], A_log[..., 0, 1]
    a10, a11 = A_log[..., 1, 0], A_log[..., 1, 1]
    v0, v1   = v_log[..., 0], v_log[..., 1]

    y0 = complex_lse(a00 + v0, a01 + v1)
    y1 = complex_lse(a10 + v0, a11 + v1)
    
    return jnp.stack([y0, y1], axis=-1)

def cssm_coupled_scan_op(carry_i, carry_j):
    """
    Associative Scan Operator for SSM: (K, u)
    (K_j, u_j) o (K_i, u_i) = (K_j @ K_i,  K_j @ u_i + u_j)
    """
    k_i, u_i = carry_i
    k_j, u_j = carry_j
    
    # 1. Compose Transitions
    k_new = log_mat_mul_2x2(k_j, k_i)
    
    # 2. Update State Input
    # u_new = LSE( (K_j @ u_i), u_j )
    transformed_u = log_mat_vec_mul_2(k_j, u_i)
    u_new = complex_lse(transformed_u, u_j)
    
    return k_new, u_new
