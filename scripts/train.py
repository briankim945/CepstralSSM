import jax
import jax.numpy as jnp
from jax import random
from pathlib import Path
import numpy as np
from typing import Optional
import argparse
import os.path as osp

from src.convnext import ConvNeXt
from .training_utils import TrainingConfig, Logger, EarlyStopping, visualize_predictions
from data.dataloader import load_dataset_from_directory


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
    print("Training with File-Based DataLoaders")
    print("=" * 70)
    
    # Create directories
    Path(model_save_dir).mkdir(exist_ok=True, parents=True)
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize logger and early stopping
    logger = Logger(log_dir)
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    
    # Create model
    print("\n1. Initializing model...")
    model = ConvNeXt()
    
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
        train_loader = load_dataset_from_directory(
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
        val_loader = load_dataset_from_directory(
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
    
    test_loader = load_dataset_from_directory(
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='./data_dir')
    parser.add_argument('-o', '--output_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

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
        checkpoint_dir=args.output_dir,
        log_dir='logs',
        task='regression'
    )
    
    print("\n" + "=" * 70)
    print("Training with File-Based DataLoaders")
    print("=" * 70)
    
    # Train model
    print("\nTraining model...")
    final_state, logger = train_with_file_dataloader(
        train_dir=str(osp.join(args.data_dir, "train")),
        val_dir=str(osp.join(args.data_dir, "val")),
        test_dir=str(osp.join(args.data_dir, "test")),
        config=config,
    )
