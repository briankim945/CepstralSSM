import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.training import train_state
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Optional
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model config
    d_model: int = 256
    d_state: int = 16
    
    # Training config
    batch_size: int = 32
    seq_len: int = 128
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Optimization
    gradient_clip: float = 1.0
    grad_accum_steps: int = 1
    
    # Logging
    log_every: int = 100
    eval_every: int = 500
    checkpoint_every: int = 1000
    
    # Paths
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    
    # Task type
    task: str = 'regression'  # 'regression', 'classification', 'sequence_modeling'
    num_classes: int = 10  # For classification


class LossRegistry:
    """Registry of different loss functions for different tasks."""
    
    @staticmethod
    def mse_loss(logits, labels, mask=None):
        """Mean squared error for regression."""
        if mask is None:
            mask = jnp.ones(labels.shape[:2])
        
        loss = jnp.sum((logits - labels) ** 2 * mask[..., None])
        loss = loss / jnp.sum(mask)
        
        return loss
    
    @staticmethod
    def cross_entropy_loss(logits, labels, mask=None):
        """Cross entropy for classification."""
        if mask is None:
            mask = jnp.ones(labels.shape[0])
        
        # Assuming labels are integer class indices
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(
            jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1) * mask
        )
        loss = loss / jnp.sum(mask)
        
        return loss
    
    @staticmethod
    def sequence_cross_entropy(logits, labels, mask=None):
        """Cross entropy for sequence prediction (e.g., language modeling)."""
        if mask is None:
            mask = jnp.ones(labels.shape)
        
        # logits: (batch, seq_len, vocab_size)
        # labels: (batch, seq_len) with integer indices
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        # Get log probability of correct class at each position
        labels_one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        loss = -jnp.sum(log_probs * labels_one_hot * mask[..., None])
        loss = loss / jnp.sum(mask)
        
        return loss
    
    @staticmethod
    def huber_loss(logits, labels, delta=1.0, mask=None):
        """Huber loss for robust regression."""
        if mask is None:
            mask = jnp.ones(labels.shape[:2])
        
        errors = logits - labels
        abs_errors = jnp.abs(errors)
        
        quadratic = jnp.minimum(abs_errors, delta)
        linear = abs_errors - quadratic
        
        loss = 0.5 * quadratic ** 2 + delta * linear
        loss = jnp.sum(loss * mask[..., None]) / jnp.sum(mask)
        
        return loss


class MetricsTracker:
    """Track and compute various metrics."""
    
    @staticmethod
    def accuracy(logits, labels, mask=None):
        """Classification accuracy."""
        if mask is None:
            mask = jnp.ones(labels.shape[0])
        
        predictions = jnp.argmax(logits, axis=-1)
        correct = (predictions == labels) * mask
        
        return jnp.sum(correct) / jnp.sum(mask)
    
    @staticmethod
    def sequence_accuracy(logits, labels, mask=None):
        """Token-level accuracy for sequences."""
        if mask is None:
            mask = jnp.ones(labels.shape)
        
        predictions = jnp.argmax(logits, axis=-1)
        correct = (predictions == labels) * mask
        
        return jnp.sum(correct) / jnp.sum(mask)
    
    @staticmethod
    def mae(logits, labels, mask=None):
        """Mean absolute error."""
        if mask is None:
            mask = jnp.ones(labels.shape[:2])
        
        errors = jnp.abs(logits - labels)
        return jnp.sum(errors * mask[..., None]) / jnp.sum(mask)
    
    @staticmethod
    def rmse(logits, labels, mask=None):
        """Root mean squared error."""
        if mask is None:
            mask = jnp.ones(labels.shape[:2])
        
        squared_errors = (logits - labels) ** 2
        mse = jnp.sum(squared_errors * mask[..., None]) / jnp.sum(mask)
        
        return jnp.sqrt(mse)


class Logger:
    """Simple logging utility."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.metrics_history = []
    
    def log_metrics(self, step: int, metrics: Dict[str, float], prefix: str = ''):
        """Log metrics for a step."""
        log_entry = {'step': step, **{f'{prefix}{k}': v for k, v in metrics.items()}}
        self.metrics_history.append(log_entry)
        
        # Print to console
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        print(f"Step {step} | {prefix}{metrics_str}")
    
    def save_metrics(self):
        """Save metrics history to JSON."""
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training and validation curves."""
        if not self.metrics_history:
            return
        
        # Extract metrics
        steps = [entry['step'] for entry in self.metrics_history]
        train_losses = [entry.get('train_loss', None) for entry in self.metrics_history]
        val_losses = [entry.get('val_loss', None) for entry in self.metrics_history]
        
        # Filter out None values
        train_data = [(s, l) for s, l in zip(steps, train_losses) if l is not None]
        val_data = [(s, l) for s, l in zip(steps, val_losses) if l is not None]
        
        if train_data or val_data:
            plt.figure(figsize=(10, 6))
            
            if train_data:
                train_steps, train_vals = zip(*train_data)
                plt.plot(train_steps, train_vals, label='Train Loss', alpha=0.7)
            
            if val_data:
                val_steps, val_vals = zip(*val_data)
                plt.plot(val_steps, val_vals, label='Val Loss', alpha=0.7)
            
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            else:
                plt.savefig(self.log_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
            
            plt.close()


class GradientAccumulator:
    """Accumulate gradients over multiple steps."""
    
    def __init__(self, accum_steps: int):
        self.accum_steps = accum_steps
        self.accumulated_grads = None
        self.step_count = 0
    
    def accumulate(self, grads):
        """Add gradients to accumulator."""
        if self.accumulated_grads is None:
            self.accumulated_grads = grads
        else:
            self.accumulated_grads = jax.tree_map(
                lambda acc, g: acc + g, 
                self.accumulated_grads, 
                grads
            )
        
        self.step_count += 1
    
    def should_update(self):
        """Check if we should apply accumulated gradients."""
        return self.step_count >= self.accum_steps
    
    def get_and_reset(self):
        """Get accumulated gradients and reset."""
        # Average the gradients
        grads = jax.tree_map(
            lambda g: g / self.accum_steps, 
            self.accumulated_grads
        )
        
        self.accumulated_grads = None
        self.step_count = 0
        
        return grads


def create_optimizer(config: TrainingConfig, total_steps: int):
    """Create optimizer with learning rate schedule."""
    # Learning rate schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=total_steps,
        end_value=config.learning_rate * 0.01
    )
    
    # Optimizer chain
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.scale_by_adam(),
        optax.add_decayed_weights(config.weight_decay),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0)
    )
    
    return optimizer


def compute_batch_metrics(logits, labels, mask, task_type: str):
    """Compute metrics based on task type."""
    metrics = {}
    
    if task_type == 'regression':
        metrics['loss'] = LossRegistry.mse_loss(logits, labels, mask)
        metrics['mae'] = MetricsTracker.mae(logits, labels, mask)
        metrics['rmse'] = MetricsTracker.rmse(logits, labels, mask)
    
    elif task_type == 'classification':
        metrics['loss'] = LossRegistry.cross_entropy_loss(logits, labels, mask)
        metrics['accuracy'] = MetricsTracker.accuracy(logits, labels, mask)
    
    elif task_type == 'sequence_modeling':
        metrics['loss'] = LossRegistry.sequence_cross_entropy(logits, labels, mask)
        metrics['accuracy'] = MetricsTracker.sequence_accuracy(logits, labels, mask)
    
    return metrics


class EarlyStopping:
    """Early stopping based on validation metrics."""
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def should_stop(self, val_loss: float) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def visualize_predictions(
    state,
    test_batch,
    test_labels,
    task_type: str,
    num_samples: int = 5,
    save_path: Optional[str] = None
):
    """Visualize model predictions."""
    # Get predictions
    predictions = state.apply_fn({'params': state.params}, test_batch)
    predictions = jax.device_get(predictions)
    test_labels = jax.device_get(test_labels)
    test_batch = jax.device_get(test_batch)
    
    if task_type == 'regression':
        # Plot predictions vs ground truth for sequence data
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2 * num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes[:num_samples]):
            # Plot first dimension of output
            ax.plot(test_labels[i, :, 0], label='Ground Truth', alpha=0.7)
            ax.plot(predictions[i, :, 0], label='Prediction', alpha=0.7, linestyle='--')
            ax.set_title(f'Sample {i + 1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    
    plt.close()


if __name__ == "__main__":
    # Example: Create a logger and plot some dummy metrics
    logger = Logger('example_logs')
    
    # Simulate training
    for step in range(100):
        train_metrics = {'loss': 1.0 / (step + 1), 'mae': 0.5 / (step + 1)}
        logger.log_metrics(step, train_metrics, prefix='train_')
        
        if step % 10 == 0:
            val_metrics = {'loss': 1.2 / (step + 1)}
            logger.log_metrics(step, val_metrics, prefix='val_')
    
    # Save and plot
    logger.save_metrics()
    logger.plot_training_curves()
    
    print("Logger example complete! Check 'example_logs' directory.")