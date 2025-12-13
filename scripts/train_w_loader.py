"""
Training Script with File-Based DataLoaders
============================================

Complete example of training SSM models using data from local files.
"""

import jax
import jax.numpy as jnp
from jax import random
from pathlib import Path
import numpy as np
from typing import Optional

from ssm_conv import MambaBlock
from training_utils import TrainingConfig, Logger, EarlyStopping, visualize_predictions
from dataloader import (
    NumpyDataLoader,
    ImageDataLoader,
    InMemoryDataLoader,
    load_dataset_from_directory
)
from train_ssm import create_train_state, train_step, eval_step


def prepare_synthetic_dataset(
    output_dir: str = "data",
    num_train: int = 1000,
    num_val: int = 200,
    num_test: int = 200,
    seq_len: int = 128,
    d_model: int = 64
):
    """
    Create synthetic dataset and save to disk.
    Replace this with your actual data preparation logic.
    """
    print(f"Preparing synthetic dataset in {output_dir}/...")
    
    # Create directories
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    test_dir = Path(output_dir) / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_and_save(num_samples, save_dir, split_name):
        """Generate data and save as .npz files."""
        for i in range(num_samples):
            # Generate synthetic sine wave data
            frequencies = np.random.uniform(0.5, 2.0, size=d_model)
            phases = np.random.uniform(0, 2 * np.pi, size=d_model)
            t = np.linspace(0, 4 * np.pi, seq_len)
            
            # Input: noisy sine waves
            clean_signal = np.sin(frequencies[None, :] * t[:, None] + phases[None, :])
            noise = 0.1 * np.random.randn(seq_len, d_model)
            input_data = clean_signal + noise
            
            # Label: clean sine waves (denoising task)
            label_data = clean_signal
            
            # Save as .npz with both input and label
            np.savez(
                save_dir / f"{split_name}_{i:04d}.npz",
                data=input_data.astype(np.float32),
                label=label_data.astype(np.float32)
            )
        
        print(f"  Saved {num_samples} samples to {save_dir}")
    
    # Generate splits
    generate_and_save(num_train, train_dir, "train")
    generate_and_save(num_val, val_dir, "val")
    generate_and_save(num_test, test_dir, "test")
    
    print(f"✓ Dataset preparation complete!")
    return train_dir, val_dir, test_dir


def train_with_file_dataloader(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    config: TrainingConfig,
    model_save_dir: str = "checkpoints",
    log_dir: str = "logs"
):
    """
    Train SSM model using file-based dataloaders.
    
    Args:
        train_dir: Directory with training data
        val_dir: Directory with validation data
        test_dir: Directory with test data
        config: Training configuration
        model_save_dir: Where to save checkpoints
        log_dir: Where to save logs
    """
    print("\n" + "=" * 70)
    print("Training SSM with File-Based DataLoaders")
    print("=" * 70)
    
    # Create directories
    Path(model_save_dir).mkdir(exist_ok=True, parents=True)
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize logger and early stopping
    logger = Logger(log_dir)
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    
    # Create model
    print("\n1. Initializing model...")
    model = MambaBlock(d_model=config.d_model, d_state=config.d_state)
    
    # Initialize model parameters
    rng = random.PRNGKey(42)
    dummy_input = jnp.ones((config.batch_size, config.seq_len, config.d_model))
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"   Model has {num_params:,} parameters")
    
    # Create optimizer
    import optax
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=10000,
        end_value=config.learning_rate * 0.01
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    )
    
    from flax.training import train_state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    # JIT compile train and eval steps
    print("\n2. JIT compiling training functions...")
    
    @jax.jit
    def train_step_jit(state, batch, labels, mask):
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, batch)
            loss = jnp.mean((logits - labels) ** 2 * mask[..., None])
            mae = jnp.mean(jnp.abs(logits - labels) * mask[..., None])
            return loss, {'loss': loss, 'mae': mae}
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics
    
    @jax.jit
    def eval_step_jit(state, batch, labels, mask):
        logits = state.apply_fn({'params': state.params}, batch)
        loss = jnp.mean((logits - labels) ** 2 * mask[..., None])
        mae = jnp.mean(jnp.abs(logits - labels) * mask[..., None])
        return {'loss': loss, 'mae': mae}
    
    # Training loop
    print("\n3. Starting training...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 70)
        
        # Create train dataloader for this epoch
        train_loader = NumpyDataLoader(
            data_dir=train_dir,
            batch_size=config.batch_size,
            shuffle=True,
            label_key='label',
            num_epochs=1,
            drop_last=True
        )
        
        # Training
        train_metrics_list = []
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Create mask (all ones for synthetic data)
            mask = jnp.ones((inputs.shape[0], inputs.shape[1]))
            
            # Train step
            state, metrics = train_step_jit(state, inputs, labels, mask)
            train_metrics_list.append(metrics)
            global_step += 1
            
            # Log periodically
            if global_step % config.log_every == 0:
                avg_metrics = {
                    k: jnp.mean(jnp.array([m[k] for m in train_metrics_list[-config.log_every:]]))
                    for k in train_metrics_list[0].keys()
                }
                logger.log_metrics(global_step, jax.device_get(avg_metrics), prefix='train_')
        
        # Epoch training summary
        epoch_train_metrics = {
            k: jnp.mean(jnp.array([m[k] for m in train_metrics_list]))
            for k in train_metrics_list[0].keys()
        }
        print(f"Train - Loss: {epoch_train_metrics['loss']:.4f}, MAE: {epoch_train_metrics['mae']:.4f}")
        
        # Validation
        val_loader = NumpyDataLoader(
            data_dir=val_dir,
            batch_size=config.batch_size,
            shuffle=False,
            label_key='label',
            num_epochs=1
        )
        
        val_metrics_list = []
        for val_inputs, val_labels in val_loader:
            mask = jnp.ones((val_inputs.shape[0], val_inputs.shape[1]))
            val_metrics = eval_step_jit(state, val_inputs, val_labels, mask)
            val_metrics_list.append(val_metrics)
        
        avg_val_metrics = {
            k: jnp.mean(jnp.array([m[k] for m in val_metrics_list]))
            for k in val_metrics_list[0].keys()
        }
        logger.log_metrics(global_step, jax.device_get(avg_val_metrics), prefix='val_')
        
        print(f"Val   - Loss: {avg_val_metrics['loss']:.4f}, MAE: {avg_val_metrics['mae']:.4f}")
        
        # Save best model
        val_loss = float(avg_val_metrics['loss'])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            from flax.training import checkpoints
            checkpoints.save_checkpoint(
                ckpt_dir=model_save_dir,
                target=state,
                step=global_step,
                prefix='best_',
                overwrite=True
            )
            print(f"  → Saved new best model (val_loss: {val_loss:.4f})")
        
        # Early stopping
        if early_stopping.should_stop(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    # Test evaluation
    print("\n" + "=" * 70)
    print("4. Final evaluation on test set...")
    print("-" * 70)
    
    test_loader = NumpyDataLoader(
        data_dir=test_dir,
        batch_size=config.batch_size,
        shuffle=False,
        label_key='label',
        num_epochs=1
    )
    
    test_metrics_list = []
    test_batch_saved = None
    test_labels_saved = None
    
    for test_inputs, test_labels in test_loader:
        if test_batch_saved is None:
            test_batch_saved = test_inputs[:5]  # Save first 5 for visualization
            test_labels_saved = test_labels[:5]
        
        mask = jnp.ones((test_inputs.shape[0], test_inputs.shape[1]))
        test_metrics = eval_step_jit(state, test_inputs, test_labels, mask)
        test_metrics_list.append(test_metrics)
    
    avg_test_metrics = {
        k: jnp.mean(jnp.array([m[k] for m in test_metrics_list]))
        for k in test_metrics_list[0].keys()
    }
    
    print("\nTest Results:")
    for k, v in avg_test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save logs and visualizations
    print("\n5. Saving results...")
    logger.save_metrics()
    logger.plot_training_curves(save_path=Path(log_dir) / 'training_curves.png')
    
    if test_batch_saved is not None:
        visualize_predictions(
            state,
            test_batch_saved,
            test_labels_saved,
            config.task,
            num_samples=5,
            save_path=Path(log_dir) / 'predictions.png'
        )
    
    print(f"\n✓ Training complete!")
    print(f"  Checkpoints: {model_save_dir}/")
    print(f"  Logs: {log_dir}/")
    
    return state, logger


if __name__ == "__main__":
    # Configuration
    config = TrainingConfig(
        d_model=64,
        d_state=32,
        batch_size=32,
        seq_len=128,
        num_epochs=15,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=500,
        gradient_clip=1.0,
        log_every=50,
        eval_every=200,
        checkpoint_dir='file_checkpoints',
        log_dir='file_logs',
        task='regression'
    )
    
    print("\n" + "=" * 70)
    print("SSM Training with File-Based DataLoaders")
    print("=" * 70)
    
    # Step 1: Prepare dataset (or use your existing data)
    print("\nStep 1: Preparing dataset...")
    train_dir, val_dir, test_dir = prepare_synthetic_dataset(
        output_dir="synthetic_data",
        num_train=1000,
        num_val=200,
        num_test=200,
        seq_len=config.seq_len,
        d_model=config.d_model
    )
    
    # Step 2: Train model
    print("\nStep 2: Training model...")
    final_state, logger = train_with_file_dataloader(
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        test_dir=str(test_dir),
        config=config,
        model_save_dir='file_checkpoints',
        log_dir='file_logs'
    )
    
    print("\n" + "=" * 70)
    print("All done! 🎉")
    print("=" * 70)
    print("\nTo use with your own data:")
    print("1. Organize your data in train/val/test directories")
    print("2. Save as .npy, .npz, .csv, or image files")
    print("3. Use the appropriate dataloader (NumpyDataLoader, ImageDataLoader, etc.)")
    print("4. Run this script!")