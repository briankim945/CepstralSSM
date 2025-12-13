"""
Complete SSM Training Example
==============================

This script demonstrates a full training pipeline for SSM models in JAX:
- Data preparation
- Model initialization
- Training with validation
- Checkpointing
- Evaluation and visualization
"""

import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state, checkpoints
import optax
from pathlib import Path
from typing import Tuple, Iterator
import time

from ssm_conv import MambaBlock, SSMConv
from training_utils import (
    TrainingConfig, 
    Logger, 
    compute_batch_metrics,
    create_optimizer,
    EarlyStopping,
    visualize_predictions
)


class SSMDataset:
    """
    Example dataset for SSM training.
    Generates synthetic time series data.
    """
    
    def __init__(self, num_samples: int, seq_len: int, d_model: int, task: str = 'regression'):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.d_model = d_model
        self.task = task
    
    def generate_sine_waves(self, rng, batch_size: int):
        """Generate synthetic sine wave data."""
        rng, freq_rng, phase_rng, noise_rng = random.split(rng, 4)
        
        # Random frequencies and phases
        frequencies = random.uniform(freq_rng, (batch_size, self.d_model), minval=0.5, maxval=2.0)
        phases = random.uniform(phase_rng, (batch_size, self.d_model), minval=0, maxval=2 * jnp.pi)
        
        # Time steps
        t = jnp.linspace(0, 4 * jnp.pi, self.seq_len)
        
        # Generate sine waves
        inputs = jnp.sin(frequencies[:, None, :] * t[None, :, None] + phases[:, None, :])
        
        # Add noise
        noise = 0.1 * random.normal(noise_rng, inputs.shape)
        inputs = inputs + noise
        
        # Labels: predict next step (or same step for denoising)
        if self.task == 'regression':
            # Denoising task: clean sine waves
            labels = jnp.sin(frequencies[:, None, :] * t[None, :, None] + phases[:, None, :])
        else:
            # Next-step prediction
            labels = jnp.concatenate([inputs[:, 1:, :], inputs[:, -1:, :]], axis=1)
        
        mask = jnp.ones((batch_size, self.seq_len))
        
        return inputs, labels, mask
    
    def data_loader(self, rng, batch_size: int, num_batches: int) -> Iterator:
        """Create a data loader iterator."""
        for _ in range(num_batches):
            rng, batch_rng = random.split(rng)
            yield self.generate_sine_waves(batch_rng, batch_size)


def create_model(config: TrainingConfig):
    """Create SSM model based on config."""
    return MambaBlock(
        d_model=config.d_model,
        d_state=config.d_state,
        expand=2
    )


def train_step(state, batch, labels, mask, task_type):
    """Single training step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch)
        metrics = compute_batch_metrics(logits, labels, mask, task_type)
        return metrics['loss'], metrics
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, metrics


def eval_step(state, batch, labels, mask, task_type):
    """Single evaluation step."""
    logits = state.apply_fn({'params': state.params}, batch)
    metrics = compute_batch_metrics(logits, labels, mask, task_type)
    return metrics


def train_and_evaluate(
    config: TrainingConfig,
    train_dataset: SSMDataset,
    val_dataset: SSMDataset,
    test_dataset: SSMDataset
):
    """
    Complete training and evaluation pipeline.
    """
    # Setup
    rng = random.PRNGKey(42)
    model = create_model(config)
    logger = Logger(config.log_dir)
    early_stopping = EarlyStopping(patience=10)
    
    # Create directories
    Path(config.checkpoint_dir).mkdir(exist_ok=True, parents=True)
    Path(config.log_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize model
    print("Initializing model...")
    dummy_input = jnp.ones((config.batch_size, config.seq_len, config.d_model))
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model has {num_params:,} parameters")
    
    # Create optimizer
    steps_per_epoch = train_dataset.num_samples // config.batch_size
    total_steps = steps_per_epoch * config.num_epochs
    tx = create_optimizer(config, total_steps)
    
    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    # JIT compile training and eval steps
    train_step_jit = jax.jit(lambda s, b, l, m: train_step(s, b, l, m, config.task))
    eval_step_jit = jax.jit(lambda s, b, l, m: eval_step(s, b, l, m, config.task))
    
    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Training
        train_rng = random.fold_in(rng, epoch)
        train_loader = train_dataset.data_loader(
            train_rng, 
            config.batch_size, 
            steps_per_epoch
        )
        
        train_metrics_list = []
        
        for batch_idx, (inputs, labels, mask) in enumerate(train_loader):
            state, metrics = train_step_jit(state, inputs, labels, mask)
            train_metrics_list.append(metrics)
            global_step += 1
            
            # Log training metrics
            if global_step % config.log_every == 0:
                avg_metrics = {
                    k: jnp.mean(jnp.array([m[k] for m in train_metrics_list[-config.log_every:]]))
                    for k in train_metrics_list[0].keys()
                }
                logger.log_metrics(global_step, jax.device_get(avg_metrics), prefix='train_')
            
            # Validation
            if global_step % config.eval_every == 0:
                val_rng = random.fold_in(rng, epoch + 1000)
                val_loader = val_dataset.data_loader(
                    val_rng,
                    config.batch_size,
                    val_dataset.num_samples // config.batch_size
                )
                
                val_metrics_list = []
                for val_inputs, val_labels, val_mask in val_loader:
                    val_metrics = eval_step_jit(state, val_inputs, val_labels, val_mask)
                    val_metrics_list.append(val_metrics)
                
                avg_val_metrics = {
                    k: jnp.mean(jnp.array([m[k] for m in val_metrics_list]))
                    for k in val_metrics_list[0].keys()
                }
                logger.log_metrics(global_step, jax.device_get(avg_val_metrics), prefix='val_')
                
                # Save best model
                val_loss = float(avg_val_metrics['loss'])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoints.save_checkpoint(
                        ckpt_dir=config.checkpoint_dir,
                        target=state,
                        step=global_step,
                        prefix='best_',
                        overwrite=True
                    )
                    print(f"  → New best model saved (val_loss: {val_loss:.4f})")
                
                # Early stopping check
                if early_stopping.should_stop(val_loss):
                    print(f"Early stopping triggered at step {global_step}")
                    break
            
            # Checkpoint
            if global_step % config.checkpoint_every == 0:
                checkpoints.save_checkpoint(
                    ckpt_dir=config.checkpoint_dir,
                    target=state,
                    step=global_step,
                    prefix='checkpoint_',
                    keep=3
                )
        
        epoch_time = time.time() - epoch_start
        
        # Epoch summary
        epoch_metrics = {
            k: jnp.mean(jnp.array([m[k] for m in train_metrics_list]))
            for k in train_metrics_list[0].keys()
        }
        print(f"\nEpoch {epoch + 1}/{config.num_epochs} ({epoch_time:.2f}s)")
        print(f"  Train loss: {epoch_metrics['loss']:.4f}")
        
        if early_stopping.should_stop(best_val_loss):
            break
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_rng = random.PRNGKey(999)
    test_loader = test_dataset.data_loader(
        test_rng,
        config.batch_size,
        test_dataset.num_samples // config.batch_size
    )
    
    test_metrics_list = []
    test_batch_saved = None
    test_labels_saved = None
    
    for test_inputs, test_labels, test_mask in test_loader:
        if test_batch_saved is None:
            test_batch_saved = test_inputs
            test_labels_saved = test_labels
        
        test_metrics = eval_step_jit(state, test_inputs, test_labels, test_mask)
        test_metrics_list.append(test_metrics)
    
    avg_test_metrics = {
        k: jnp.mean(jnp.array([m[k] for m in test_metrics_list]))
        for k in test_metrics_list[0].keys()
    }
    
    print("\nTest Results:")
    for k, v in avg_test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    logger.save_metrics()
    logger.plot_training_curves(save_path=Path(config.log_dir) / 'training_curves.png')
    
    if test_batch_saved is not None:
        visualize_predictions(
            state,
            test_batch_saved,
            test_labels_saved,
            config.task,
            num_samples=5,
            save_path=Path(config.log_dir) / 'predictions.png'
        )
    
    print(f"\nTraining complete! Results saved to {config.log_dir}/")
    
    return state, logger


if __name__ == "__main__":
    # Configuration
    config = TrainingConfig(
        d_model=64,
        d_state=32,
        batch_size=32,
        seq_len=128,
        num_epochs=20,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=500,
        gradient_clip=1.0,
        log_every=50,
        eval_every=200,
        checkpoint_every=500,
        checkpoint_dir='ssm_checkpoints',
        log_dir='ssm_logs',
        task='regression'
    )
    
    # Create datasets
    train_dataset = SSMDataset(
        num_samples=10000,
        seq_len=config.seq_len,
        d_model=config.d_model,
        task=config.task
    )
    
    val_dataset = SSMDataset(
        num_samples=2000,
        seq_len=config.seq_len,
        d_model=config.d_model,
        task=config.task
    )
    
    test_dataset = SSMDataset(
        num_samples=2000,
        seq_len=config.seq_len,
        d_model=config.d_model,
        task=config.task
    )
    
    # Train
    final_state, logger = train_and_evaluate(
        config,
        train_dataset,
        val_dataset,
        test_dataset
    )
    
    print("\n✓ Training pipeline complete!")
    print(f"  - Checkpoints: {config.checkpoint_dir}/")
    print(f"  - Logs: {config.log_dir}/")
    print(f"  - Visualizations: {config.log_dir}/training_curves.png")