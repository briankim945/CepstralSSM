import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.signal import convolve
import flax.linen as nn


class SSMConv(nn.Module):
    """
    SSM layer with convolutional mode for efficient parallel training.
    
    This implementation can compute the entire sequence in parallel using
    convolutions rather than sequential recurrence, making training much faster.
    """
    d_model: int
    d_state: int = 64
    kernel_size: int = 64  # Length of SSM convolution kernel
    dt_min: float = 0.001
    dt_max: float = 0.1
    
    def setup(self):
        # SSM parameters (similar to before)
        self.A = self.param('A', self.init_A, (self.d_state,))
        self.B = self.param('B', nn.initializers.normal(stddev=0.01), 
                           (self.d_state, self.d_model))
        self.C = self.param('C', nn.initializers.normal(stddev=0.01), 
                           (self.d_model, self.d_state))
        self.D = self.param('D', nn.initializers.ones, (self.d_model,))
        
        self.log_dt = self.param('log_dt', 
                                 lambda rng, shape: jnp.log(
                                     random.uniform(rng, shape, 
                                                   minval=self.dt_min, 
                                                   maxval=self.dt_max)),
                                 (self.d_model,))
    
    def init_A(self, rng, shape):
        """Initialize A with HiPPO-like initialization."""
        n = jnp.arange(shape[0])
        A = -(n + 1)  # Diagonal elements for stability
        return A
    
    def ssm_kernel(self, L):
        """
        Compute the SSM convolution kernel.
        
        Args:
            L: Length of the kernel
            
        Returns:
            K: Convolution kernel of shape (L, d_model)
        """
        dt = jnp.exp(self.log_dt)  # (d_model,)
        
        # Discretize A and B
        # A_bar = exp(A * dt), B_bar = (A^{-1})(exp(A*dt) - I)B
        A = self.A[:, None]  # (d_state, 1)
        dt_expanded = dt[None, :]  # (1, d_model)
        
        # Compute discretized matrices
        A_bar = jnp.exp(A * dt_expanded)  # (d_state, d_model)
        
        # For each position in the kernel
        powers = jnp.arange(L)[:, None, None]  # (L, 1, 1)
        A_powers = A_bar[None, :, :] ** powers  # (L, d_state, d_model)
        
        # K_l = C @ A_bar^l @ B for l in 0..L-1
        # C: (d_model, d_state)
        # A_powers: (L, d_state, d_model)
        # B: (d_state, d_model)
        
        K = jnp.einsum('md,ldk,dk->lk', self.C, A_powers, self.B)
        
        return K
    
    def __call__(self, u, use_conv=True):
        """
        Forward pass with option for convolutional or recurrent mode.
        
        Args:
            u: Input of shape (batch, length, d_model)
            use_conv: If True, use parallel convolution; else use recurrence
            
        Returns:
            y: Output of shape (batch, length, d_model)
        """
        batch_size, seq_len, _ = u.shape
        
        if use_conv:
            # Convolutional mode (parallel, efficient for training)
            K = self.ssm_kernel(min(seq_len, self.kernel_size))  # (kernel_size, d_model)
            
            # Apply convolution for each channel
            y = jnp.zeros_like(u)
            
            for i in range(self.d_model):
                # Convolve each batch and channel
                for b in range(batch_size):
                    y_conv = jnp.convolve(u[b, :, i], K[:, i], mode='same')
                    y = y.at[b, :, i].set(y_conv)
            
            # Add skip connection
            y = y + self.D[None, None, :] * u
            
        else:
            # Recurrent mode (sequential, for inference or long sequences)
            dt = jnp.exp(self.log_dt)
            A_bar = jnp.exp(self.A[:, None] * dt[None, :])
            
            x = jnp.zeros((batch_size, self.d_state, self.d_model))
            outputs = []
            
            for t in range(seq_len):
                u_t = u[:, t, :]
                
                # Update state
                x = A_bar[None, :, :] * x + \
                    self.B[:, :, None] * u_t[:, None, :]
                
                # Compute output
                y_t = jnp.einsum('md,bdm->bm', self.C, x) + self.D * u_t
                outputs.append(y_t)
            
            y = jnp.stack(outputs, axis=1)
        
        return y


class MambaBlock(nn.Module):
    """
    Mamba-style block combining SSM with gating and projections.
    Based on the Mamba architecture.
    """
    d_model: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    
    @nn.compact
    def __call__(self, x):
        batch, length, d = x.shape
        d_inner = self.d_model * self.expand
        
        # Input projection
        x_norm = nn.LayerNorm()(x)
        x_proj = nn.Dense(d_inner * 2, use_bias=False)(x_norm)
        
        # Split into main and gate paths
        x_main, x_gate = jnp.split(x_proj, 2, axis=-1)
        
        # 1D Convolution on main path
        x_conv = nn.Conv(features=d_inner, 
                         kernel_size=(self.d_conv,),
                         feature_group_count=d_inner,  # Depthwise
                         padding='SAME')(x_main)
        x_conv = nn.silu(x_conv)
        
        # SSM
        ssm = SSMConv(d_model=d_inner, d_state=self.d_state)
        x_ssm = ssm(x_conv)
        
        # Gating
        x_gated = x_ssm * nn.silu(x_gate)
        
        # Output projection
        x_out = nn.Dense(self.d_model, use_bias=False)(x_gated)
        
        # Residual
        return x + x_out


# Example usage
if __name__ == "__main__":
    # Test the convolutional SSM
    batch_size = 4
    seq_len = 128
    d_model = 256
    
    rng = random.PRNGKey(42)
    x = random.normal(rng, (batch_size, seq_len, d_model))
    
    # Test SSMConv
    print("Testing SSMConv...")
    model = SSMConv(d_model=d_model, d_state=64)
    variables = model.init(rng, x)
    
    # Compare conv vs recurrent mode
    y_conv = model.apply(variables, x, use_conv=True)
    y_rec = model.apply(variables, x, use_conv=False)
    
    print(f"Conv output shape: {y_conv.shape}")
    print(f"Recurrent output shape: {y_rec.shape}")
    print(f"Difference: {jnp.abs(y_conv - y_rec).mean()}")
    
    # Test MambaBlock
    print("\nTesting MambaBlock...")
    mamba = MambaBlock(d_model=d_model, d_state=16)
    variables_mamba = mamba.init(rng, x)
    y_mamba = mamba.apply(variables_mamba, x)
    
    print(f"Mamba output shape: {y_mamba.shape}")
    print(f"Num parameters: {sum(p.size for p in jax.tree_util.tree_leaves(variables_mamba))}")