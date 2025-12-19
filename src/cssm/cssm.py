import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import checkify

from typing import Optional
# from flax import linen as nn
from .ops import cssm_coupled_scan_op

class hCSSM(nnx.Module):
    def __init__(self, kernel_size: int, channels: int = 3, *, rngs: nnx.Rngs):
        rng = jax.random.PRNGKey(0)
        self.channels = channels
        self.kernel_size = kernel_size
        self.control_conv = nnx.Conv(in_features=channels, out_features=32,
            kernel_size=(3,3), strides=(2,2), rngs=rngs
        )
        self.alpha_dense = nnx.Linear(32,channels,rngs=rngs)
        self.delta_dense = nnx.Linear(32,channels,rngs=rngs)
        self.mu_dense = nnx.Linear(32,channels,rngs=rngs)
        self.gamma_dense = nnx.Linear(32,channels,rngs=rngs)
        norm_initializer = nnx.initializers.normal()
        self.k_exc = nnx.Param(norm_initializer(rng, (self.kernel_size, self.kernel_size, channels)))
        self.k_inh = nnx.Param(norm_initializer(rng, (self.kernel_size, self.kernel_size, channels)))
    
    def __call__(self, input_seq, rng: Optional[jax.Array] = None):
        # MAKE THE stats CONTROLLER AN INITIALIZED OBJ?
        # And all the linear layers
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        # input_seq: (B, T, H, W, C) [Real]
        B, T, H, W, C = input_seq.shape
        
        # --- A. The Controller (Biological Gates) ---
        # "Fast Weights": Predict dynamics from input
        # features 32
        # kernel_size 3,3
        # strides 2,2
        # checkify.check(jnp.array_equal(jnp.reshape(jnp.reshape(input_seq, (B * T, H, W, C)), (B, T, H, W, C)), input_seq),
        #                "The reshapes aren't equal")
        stats = self.control_conv(jnp.reshape(input_seq, (B * T, H, W, C))) # Downsample
        stats = jnp.reshape(stats, (B, T, -1, 32)) # New shape -> (B, T, downsampled_h * downsampled_w, 32)
        jax.debug.print("stats.shape: {x}", x=stats.shape)
        stats = jax.nn.gelu(stats)
        ctx = jnp.mean(stats, axis=2) # Global Pool -> (B, T, 32)
        
        # 1. Decay Gates (Diagonals)
        # alpha: Shunting Inh (X decay), delta: Refractory (Y decay)
        alpha = jax.nn.sigmoid(self.alpha_dense(ctx)) 
        delta = jax.nn.sigmoid(self.delta_dense(ctx))
        
        # 2. Coupling Gates (Off-Diagonals)
        # mu: Subtractive Inh strength, gamma: Additive Exc strength
        mu    = jax.nn.softplus(self.mu_dense(ctx))
        gamma = jax.nn.softplus(self.gamma_dense(ctx))
        
        # --- B. Spatial Kernels ---
        # Learnable Near (Exc) and Far (Inh) kernels
        
        # k_exc = self.param('k_exc', nn.initializers.normal(), 
        #                    (self.kernel_size, self.kernel_size, C))
        # k_inh = self.param('k_inh', nn.initializers.normal(), 
        #                    (self.kernel_size, self.kernel_size, C))
        
        # FFT to Spectral Domain
        pad_h, pad_w = (H - self.kernel_size)//2, (W - self.kernel_size)//2
        # Pad function ...
        K_E_spec = jnp.fft.rfft2(jnp.pad(self.k_exc, 
                                         ((pad_h, pad_h),
                                          (pad_w, pad_w), (0,0))), axes=(0,1)) 
        K_I_spec = jnp.fft.rfft2(jnp.pad(self.k_inh, 
                                         ((pad_h, pad_h),
                                          (pad_w, pad_w), (0,0))), axes=(0,1))
        
        # --- C. Build Transition Matrix K_t ---
        # We build the 2x2 matrix for every T, Freq, Channel
        
        # 1. Diagonal Terms (Decay)
        # Simple complex decay 0.9 + 0j
        decay_base = 0.9 * (1.0 + 0j)
        A_xx = decay_base * alpha[:, :, None, None] # (B, T, 1, 1, C)
        A_yy = decay_base * delta[:, :, None, None] # (B, T, 1, 1, C)
        # tile?
        
        # 2. Off-Diagonal Terms (Coupling)
        # Y -> X (Inhibition): -1.0 * K_I * mu
        # Negative sign is critical for opponent dynamics!
        jax.debug.print("{k} {m}", k=K_I_spec.shape, m=mu.shape)
        A_xy = -1.0 * K_I_spec[None, ...] * mu[:, :, None, None] # (B, T, H_f, W_f, C)
        
        # X -> Y (Excitation): +1.0 * K_E * gamma
        A_yx = K_E_spec[None, ...] * gamma[:, :, None, None] # (B, T, H_f, W_f, C)

        # Reshape Diagonal Terms
        # _, _, H_f, W_f, _ = A_xy.shape
        A_xx = jnp.tile(A_xx, (1,1,H,W,1))
        A_yy = jnp.tile(A_yy, (1,1,H,W,1))
        
        # Stack into (B, T, H, W_f, C, 2, 2)
        row0 = jnp.stack([A_xx, A_xy], axis=-1)
        row1 = jnp.stack([A_yx, A_yy], axis=-1)
        K_hat = jnp.stack([row0, row1], axis=-2)
        
        # --- D. Build Input Vector U_t ---
        U_hat = jnp.fft.rfft2(input_seq, axes=(2, 3))
        # Input drives X (index 0), Y (index 1) gets 0 input
        U_vec = jnp.stack([U_hat, jnp.zeros_like(U_hat)], axis=-1)
        
        # --- E. Cepstral Scan ---
        # 1. Lift to GOOM Space
        K_log = jnp.log(K_hat)
        U_log = jnp.log(U_vec)
        
        # 2. Parallel Scan
        _, State_log = jax.lax.associative_scan(
            cssm_coupled_scan_op, (K_log, U_log), axis=1
        )
        
        # 3. Project Output (Firing Rate Y)
        Y_log = State_log[..., 1] # Index 1 is Y
        Y_rec = jnp.fft.irfft2(jnp.exp(Y_log), axes=(2, 3))
        
        return Y_rec
