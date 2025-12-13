import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple
import flax.linen as nn


class SSMLayer(nn.Module):
    """
    Basic State Space Model (SSM) layer implementation in JAX/Flax.
    
    The SSM is defined by the continuous-time equations:
        x'(t) = Ax(t) + Bu(t)
        y(t) = Cx(t) + Du(t)
    
    Which we discretize for use in deep learning.
    """
    d_model: int  # Model dimension (input/output size)
    d_state: int = 64  # State dimension (N)
    dt_min: float = 0.001
    dt_max: float = 0.1
    
    def setup(self):
        # Initialize SSM parameters
        # A matrix: (d_state, d_state) - state transition
        self.A = self.param('A', self.init_A, (self.d_state, self.d_state))
        
        # B matrix: (d_state, d_model) - input to state
        self.B = self.param('B', nn.initializers.normal(stddev=0.01), 
                           (self.d_state, self.d_model))
        
        # C matrix: (d_model, d_state) - state to output
        self.C = self.param('C', nn.initializers.normal(stddev=0.01), 
                           (self.d_model, self.d_state))
        
        # D matrix: (d_model,) - skip connection
        self.D = self.param('D', nn.initializers.ones, (self.d_model,))
        
        # Learnable timescale parameter
        self.log_dt = self.param('log_dt', 
                                 lambda rng, shape: jnp.log(
                                     random.uniform(rng, shape, 
                                                   minval=self.dt_min, 
                                                   maxval=self.dt_max)),
                                 (self.d_model,))
    
    def init_A(self, rng, shape):
        """Initialize A matrix with diagonal structure for stability."""
        # Use HiPPO initialization or diagonal structure
        # Here we use a simple diagonal negative real part for stability
        A = -0.5 * jnp.ones(shape)
        A = A * jnp.eye(shape[0])
        return A
    
    def discretize(self, dt):
        """
        Discretize continuous SSM using zero-order hold (ZOH).
        
        Converts continuous (A, B) to discrete (A_bar, B_bar).
        """
        # Expand dt to match d_state dimension
        dt = dt[None, :]  # (1, d_model)
        
        # Simple Euler discretization (you can use more sophisticated methods)
        # For better accuracy, use matrix exponential
        A_bar = jnp.eye(self.d_state)[:, :, None] + self.A[:, :, None] * dt
        B_bar = self.B[:, :, None] * dt
        
        return A_bar, B_bar
    
    def __call__(self, u):
        """
        Forward pass through SSM.
        
        Args:
            u: Input sequence of shape (batch, length, d_model)
            
        Returns:
            y: Output sequence of shape (batch, length, d_model)
        """
        batch_size, seq_len, _ = u.shape
        
        # Get timescale
        dt = jnp.exp(self.log_dt)  # (d_model,)
        
        # Discretize the SSM
        A_bar, B_bar = self.discretize(dt)
        
        # Initialize state
        x = jnp.zeros((batch_size, self.d_state, self.d_model))
        
        # Prepare output
        outputs = []
        
        # Recurrent computation
        for t in range(seq_len):
            u_t = u[:, t, :]  # (batch, d_model)
            
            # State update: x_{t+1} = A_bar @ x_t + B_bar @ u_t
            # A_bar: (d_state, d_state, d_model)
            # x: (batch, d_state, d_model)
            # B_bar: (d_state, d_model, 1)
            
            # Vectorized state update across d_model dimensions
            x_new = jnp.einsum('ijk,bjk->bik', A_bar, x) + \
                    jnp.einsum('ijk,bk->bij', B_bar, u_t)
            
            # Output: y_t = C @ x_t + D @ u_t
            y_t = jnp.einsum('ij,bij->bi', self.C, x_new) + self.D * u_t
            
            outputs.append(y_t)
            x = x_new
        
        # Stack outputs
        y = jnp.stack(outputs, axis=1)  # (batch, length, d_model)
        
        return y


class SSMBlock(nn.Module):
    """
    Full SSM block with normalization and projection layers.
    """
    d_model: int
    d_state: int = 64
    expand_factor: int = 2
    
    @nn.compact
    def __call__(self, x):
        # Layer norm
        x_norm = nn.LayerNorm()(x)
        
        # Expand dimension
        d_inner = self.d_model * self.expand_factor
        x_proj = nn.Dense(d_inner)(x_norm)
        
        # Split for gating (optional, similar to Mamba)
        x_ssm = nn.silu(x_proj)
        
        # Apply SSM
        x_ssm = SSMLayer(d_model=d_inner, d_state=self.d_state)(x_ssm)
        
        # Project back
        x_out = nn.Dense(self.d_model)(x_ssm)
        
        # Residual connection
        return x + x_out


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    batch_size = 2
    seq_len = 10
    d_model = 64
    d_state = 32
    
    # Initialize model
    model = SSMBlock(d_model=d_model, d_state=d_state)
    
    # Create random input
    rng = random.PRNGKey(0)
    x = random.normal(rng, (batch_size, seq_len, d_model))
    
    # Initialize parameters
    variables = model.init(rng, x)
    
    # Forward pass
    output = model.apply(variables, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.size for p in jax.tree_util.tree_leaves(variables))}")