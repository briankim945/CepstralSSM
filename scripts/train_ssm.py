import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
from flax.training import train_state, checkpoints
from flax import struct
import time
from typing import Any, Dict, Callable
from pathlib import Path

# Import our SSM models
from ssm_conv import MambaBlock, SSMConv


class TrainState(train_state.TrainState):
    """
    Extended TrainState with additional metrics tracking.
    """
    batch_stats: Any = None  # For batch norm if needed
    metrics: Dict[str, float] = struct.field(default_factory=dict)


def create_train_state(rng, model, learning_rate, input_shape):
    """
    Initialize training state with model parameters and optimizer.
    
    Args:
        rng: Random key
        model: Flax model
        learning_rate: Learning rate for optimizer
        input_shape: Shape of input data (batch, seq_len, d_model)
    
    Returns:
        TrainState object
    """
    # Initialize model parameters
    dummy_input = jnp.ones(input_shape)
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    # Create optimizer
    # Using AdamW with weight decay and learning rate schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=learning_rate,
        warmup_steps=1000,
        decay_steps=10000,
        end_value=1e-5
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(learning_rate=schedule, weight_decay=0.01)
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        metrics={}
    )


def compute_metrics(logits, labels, mask=None):
    """
    Compute training metrics.
    
    Args:
        logits: Model predictions (batch, seq_len, vocab_size) or (batch, seq_len, d_model)
        labels: Ground truth labels
        mask: Optional mask for padding tokens
    
    Returns:
        Dictionary of metrics
    """
    if mask is None:
        mask = jnp.ones(labels.shape[:2])
    
    # Example: MSE loss for regression or cross-entropy for classification
    # Here we'll use MSE as a simple example
    loss = jnp.mean((logits - labels) ** 2 * mask[..., None])
    
    # Mean absolute error
    mae = jnp.mean(jnp.abs(logits - labels) * mask[..., None])
    
    return {
        'loss': loss,
        'mae': mae,
    }


@jit
def train_step(state, batch, labels, mask=None):
    """
    Single training step with JIT compilation.
    
    Args:
        state: Current TrainState
        batch: Input batch (batch_size, seq_len, d_model)
        labels: Target labels
        mask: Optional mask for variable-length sequences
    
    Returns:
        Updated state and metrics
    """
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch)
        metrics = compute_metrics(logits, labels, mask)
        return metrics['loss'], metrics
    
    # Compute gradients
    (loss, metrics), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    # Update metrics
    state = state.replace(metrics=metrics)
    
    return state, metrics


@jit
def eval_step(state, batch, labels, mask=None):
    """
    Single evaluation step.
    
    Args:
        state: Current TrainState
        batch: Input batch
        labels: Target labels
        mask: Optional mask
    
    Returns:
        Metrics dictionary
    """
    logits = state.apply_fn({'params': state.params}, batch)
    metrics = compute_metrics(logits, labels, mask)
    return metrics


def train_epoch(state, train_loader, epoch):
    """
    Train for one epoch.
    
    Args:
        state: TrainState
        train_loader: Iterator over training batches
        epoch: Current epoch number
    
    Returns:
        Updated state and aggregated metrics
    """
    batch_metrics = []
    
    for batch_idx, (inputs, labels, mask) in enumerate(train_loader):
        state, metrics = train_step(state, inputs, labels, mask)
        batch_metrics.append(metrics)
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            current_loss = metrics['loss']
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {current_loss:.4f}")
    
    # Aggregate metrics
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics = {
        k: jnp.mean(jnp.array([m[k] for m in batch_metrics_np]))
        for k in batch_metrics_np[0].keys()
    }
    
    return state, epoch_metrics


def evaluate(state, eval_loader):
    """
    Evaluate model on validation/test set.
    
    Args:
        state: TrainState
        eval_loader: Iterator over evaluation batches
    
    Returns:
        Aggregated metrics
    """
    batch_metrics = []
    
    for inputs, labels, mask in eval_loader:
        metrics = eval_step(state, inputs, labels, mask)
        batch_metrics.append(metrics)
    
    # Aggregate metrics
    batch_metrics_np = jax.device_get(batch_metrics)
    eval_metrics = {
        k: jnp.mean(jnp.array([m[k] for m in batch_metrics_np]))
        for k in batch_metrics_np[0].keys()
    }
    
    return eval_metrics


def create_learning_rate_schedule(base_lr, warmup_steps, total_steps):
    """
    Create a learning rate schedule with warmup and cosine decay.
    """
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=base_lr * 0.1
    )


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    input_shape,
    checkpoint_dir='checkpoints',
    seed=42
):
    """
    Complete training loop.
    
    Args:
        model: Flax model to train
        train_loader: Training data iterator
        val_loader: Validation data iterator
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        input_shape: Shape of input (batch, seq_len, d_model)
        checkpoint_dir: Directory to save checkpoints
        seed: Random seed
    
    Returns:
        Trained state and training history
    """
    # Initialize
    rng = random.PRNGKey(seed)
    rng, init_rng = random.split(rng)
    
    state = create_train_state(init_rng, model, learning_rate, input_shape)
    
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
    }
    
    best_val_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Model has {sum(x.size for x in jax.tree_util.tree_leaves(state.params))} parameters")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training
        state, train_metrics = train_epoch(state, train_loader, epoch)
        
        # Validation
        val_metrics = evaluate(state, val_loader)
        
        # Record history
        history['train_loss'].append(float(train_metrics['loss']))
        history['train_mae'].append(float(train_metrics['mae']))
        history['val_loss'].append(float(val_metrics['loss']))
        history['val_mae'].append(float(val_metrics['mae']))
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.2f}s")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train MAE: {train_metrics['mae']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_path,
                target=state,
                step=epoch,
                prefix='best_',
                overwrite=True
            )
            print(f"Saved new best model with val_loss: {best_val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_path,
                target=state,
                step=epoch,
                prefix='checkpoint_',
                keep=3  # Keep only last 3 checkpoints
            )
    
    print("\nTraining complete!")
    return state, history


def load_checkpoint(checkpoint_dir, model, input_shape, step=None):
    """
    Load a saved checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model: Model architecture
        input_shape: Input shape for initialization
        step: Specific step to load (None for latest)
    
    Returns:
        Restored TrainState
    """
    rng = random.PRNGKey(0)
    state = create_train_state(rng, model, 1e-3, input_shape)
    
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=state,
        step=step
    )
    
    return restored_state


# Example data generator for demonstration
def create_dummy_data_loader(rng, batch_size, seq_len, d_model, num_batches):
    """
    Create a dummy data loader for testing.
    In practice, replace this with your actual data loading logic.
    """
    for i in range(num_batches):
        rng, input_rng, label_rng = random.split(rng, 3)
        
        inputs = random.normal(input_rng, (batch_size, seq_len, d_model))
        # For this example, labels are just noisy versions of inputs
        labels = inputs + 0.1 * random.normal(label_rng, inputs.shape)
        mask = jnp.ones((batch_size, seq_len))
        
        yield inputs, labels, mask


if __name__ == "__main__":
    # Configuration
    config = {
        'batch_size': 32,
        'seq_len': 64,
        'd_model': 128,
        'num_epochs': 10,
        'learning_rate': 1e-3,
        'num_train_batches': 100,
        'num_val_batches': 20,
    }
    
    # Create model
    model = MambaBlock(d_model=config['d_model'], d_state=16)
    
    # Create dummy data loaders
    rng = random.PRNGKey(42)
    train_rng, val_rng = random.split(rng)
    
    train_loader = create_dummy_data_loader(
        train_rng,
        config['batch_size'],
        config['seq_len'],
        config['d_model'],
        config['num_train_batches']
    )
    
    val_loader = create_dummy_data_loader(
        val_rng,
        config['batch_size'],
        config['seq_len'],
        config['d_model'],
        config['num_val_batches']
    )
    
    # Train model
    input_shape = (config['batch_size'], config['seq_len'], config['d_model'])
    
    final_state, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        input_shape=input_shape,
        checkpoint_dir='ssm_checkpoints'
    )
    
    print("\nTraining history:")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")