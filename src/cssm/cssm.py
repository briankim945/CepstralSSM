import jax
import jax.numpy as jnp
from flax import nnx
from flax import linen as nn
from ops import cssm_coupled_scan_op

class hCSSM(nn.Module):
    channels: int
    kernel_size: int = 15
    
    @nn.compact
    def __call__(self, input_seq):
        # input_seq: (B, T, H, W, C) [Real]
        B, T, H, W, C = input_seq.shape
        
        # --- A. The Controller (Biological Gates) ---
        # "Fast Weights": Predict dynamics from input
        stats = nn.Conv(32, (3,3), strides=(2,2))(input_seq) # Downsample
        stats = nn.gelu(stats)
        ctx = jnp.mean(stats, axis=(1, 2)) # Global Pool -> (B, T, 32)
        
        # 1. Decay Gates (Diagonals)
        # alpha: Shunting Inh (X decay), delta: Refractory (Y decay)
        alpha = nn.sigmoid(nn.Dense(C)(ctx)) 
        delta = nn.sigmoid(nn.Dense(C)(ctx))
        
        # 2. Coupling Gates (Off-Diagonals)
        # mu: Subtractive Inh strength, gamma: Additive Exc strength
        mu    = nn.softplus(nn.Dense(C)(ctx))
        gamma = nn.softplus(nn.Dense(C)(ctx))
        
        # --- B. Spatial Kernels ---
        # Learnable Near (Exc) and Far (Inh) kernels
        k_exc = self.param('k_exc', nn.initializers.normal(), 
                           (self.kernel_size, self.kernel_size, C))
        k_inh = self.param('k_inh', nn.initializers.normal(), 
                           (self.kernel_size, self.kernel_size, C))
        
        # FFT to Spectral Domain
        pad_h, pad_w = (H - self.kernel_size)//2, (W - self.kernel_size)//2
        # Pad function ...
        K_E_spec = jnp.fft.rfft2(jnp.pad(k_exc, (pad_h, pad_h), (pad_w, pad_w)), axes=(0,1)) 
        K_I_spec = jnp.fft.rfft2(jnp.pad(k_inh, (pad_h, pad_h), (pad_w, pad_w)), axes=(0,1))
        
        # --- C. Build Transition Matrix K_t ---
        # We build the 2x2 matrix for every T, Freq, Channel
        
        # 1. Diagonal Terms (Decay)
        # Simple complex decay 0.9 + 0j
        decay_base = 0.9 * (1.0 + 0j)
        A_xx = decay_base * alpha[:, None, None, :]
        A_yy = decay_base * delta[:, None, None, :]
        
        # 2. Off-Diagonal Terms (Coupling)
        # Y -> X (Inhibition): -1.0 * K_I * mu
        # Negative sign is critical for opponent dynamics!
        A_xy = -1.0 * K_I_spec[None, ...] * mu[:, None, None, :]
        
        # X -> Y (Excitation): +1.0 * K_E * gamma
        A_yx = K_E_spec[None, ...] * gamma[:, None, None, :]
        
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
