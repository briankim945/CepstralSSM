import jax
import jax.numpy as jnp

def complex_lse(x , y) :
    """
    Log - Sum - Exp for complex numbers ( GOOM addition ) .
    Computes : log ( exp ( x ) + exp ( y ) )

    Stabilized by factoring out the largest real component :
    log ( e ^ x + e ^ y ) = m + log ( e ^( x - m ) + e ^( y - m ) )
    """
    # 1. Select max real part for stability
    # Use stop_gradient to prevent gradients flowing through the max selection
    m = jax.lax.stop_gradient(jnp.maximum(x.real, y.real))

    # 2. Compute the stable sum in exponentiated space
    # ( x - m ) ensures arguments are <= 0 , preventing overflow
    w = jnp.exp(x - m) + jnp.exp(y - m)

    # 3. Return to log - space
    return m + jnp.log(w)

def cssm_scan_operator(carry_i , carry_j) :
    """
    Associative binary operator for the Cepstral Monoid .
    Args :
    carry_i : Tuple ( log_k_i , log_u_i ) from time step t
    carry_j : Tuple ( log_k_j , log_u_j ) from time step t +1
    """
    k_i , u_i = carry_i
    k_j , u_j = carry_j

    # 1. Kernel Composition : Multiplication becomes Addition
    # K_new = K_j * K_i
    k_new = k_j + k_i

    # 2. State Update : Affine transform
    # u_new = K_j * u_i + u_j
    # log ( u_new ) = LSE ( k_j + u_i , u_j )
    u_new = complex_lse(k_j + u_i, u_j)

    return k_new, u_new

def cssm_forward(u_seq , k_seq) :
    """
    Args :
    u_seq : Input sequence (T , H , W , C ) [ Real ]
    k_seq : Transition kernels (T , H , W , C ) [ Real , Spatial domain ]
    Returns :
    y_seq : Output sequence (T , H , W , C ) [ Real ]
    """
    # 1. Real Spectral Transform ( RFFT )
    # Diagonalize spatial mixing .
    # Output shape : (T , H , W /2 + 1 , C ) [ Complex ]
    U_hat = jnp.fft.rfft2(u_seq, axes =(1 , 2))
    K_hat = jnp.fft.rfft2(k_seq, axes =(1 , 2))

    # 2. Cepstral Lift ( GOOM Transform )
    # Map to complex log - space for stable additive dynamics
    U_log = jnp.log(U_hat)
    K_log = jnp.log(K_hat)

    # 3. Parallel Associative Scan
    # Solves the recurrence in O ( log T ) span
    _ , X_log = jax.lax.associative_scan(
        cssm_scan_operator,
        (K_log, U_log)
    )

    # 4. Inverse Transform ( IRFFT )
    # Map back to linear space and spatial domain
    X_hat = jnp.exp(X_log)
    X_rec = jnp.fft.irfft2(X_hat, axes =(1 , 2))

    return X_rec
