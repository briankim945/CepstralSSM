# from datasets import load_dataset
# import grain.python as grain
# # import matplotlib.pyplot as plt

import jax
from flax import nnx
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import optax
import numpy as np
from jax import numpy as jnp
from src.convnext import ConvNeXt
import tqdm
import sys
import argparse


bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"


seed = 12
train_batch_size = 32
val_batch_size = 2 * train_batch_size

# Select first 20 classes to reduce the dataset size and the training time.
train_size = 20 * 750
val_size = 20 * 250

# Transformations (ensure you define standard ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to convert PyTorch tensors to JAX arrays
def torch_to_jax(batch):
    # Convert to NumPy array first (zero-copy if on CPU)
    numpy_batch = {k: v.numpy() for k, v in batch.items()} if isinstance(batch, dict) else batch.numpy()
    # Convert NumPy array to JAX array
    return jax.numpy.array(numpy_batch)

def compute_losses_and_logits(model: nnx.Module, images: jax.Array, labels: jax.Array):
    logits = model(images)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    return loss, logits

@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, batch: dict[str, np.ndarray]
):
    # Convert np.ndarray to jax.Array on GPU
    images = jnp.array(batch["image"])
    labels = jnp.array(batch["label"], dtype=jnp.int32)

    grad_fn = nnx.value_and_grad(compute_losses_and_logits, has_aux=True)
    (loss, logits), grads = grad_fn(model, images, labels)

    optimizer.update(grads)  # In-place updates.

    return loss


@nnx.jit
def eval_step(
    model: nnx.Module, batch: dict[str, np.ndarray], eval_metrics: nnx.MultiMetric
):
    # Convert np.ndarray to jax.Array on GPU
    images = jnp.array(batch["image"])
    labels = jnp.array(batch["label"], dtype=jnp.int32)
    loss, logits = compute_losses_and_logits(model, images, labels)

    eval_metrics.update(
        loss=loss,
        logits=logits,
        labels=labels,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='./data_dir')
    args = parser.parse_args()

    # Load the dataset (point root to where ImageNet is downloaded)
    # You need your ImageNet data in the correct folder structure
    imagenet_data = datasets.ImageFolder(root=args.data_dir, transform=transform)
    print("Loaded in dataset...")
    train_dataset, val_dataset = torch.utils.data.random_split(imagenet_data, [0.8, 0.2])
    # print(imagenet_data[0], imagenet_data[0][0].shape)

    # Use PyTorch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)
    print("Loaded in dataloader...")

    train_features, train_labels = next(iter(train_loader))
    val_features, val_labels = next(iter(val_loader))

    print("Training batch info:", train_features.shape, train_features.dtype, train_labels.shape, train_labels.dtype)
    print("Validation batch info:", val_features.shape, val_features.dtype, val_labels.shape, val_labels.dtype)

    num_epochs = 10000
    learning_rate = 0.001
    momentum = 0.8
    total_steps = len(train_dataset) // train_batch_size

    lr_schedule = optax.linear_schedule(learning_rate, 0.0, num_epochs * total_steps)

    # CREATE MODEL
    rngs = nnx.Rngs(42)
    ### TODO: Make this modifiable
    model = ConvNeXt(
        num_classes=1000,
        dims=(224, 384, 768, 1536),
        rngs=rngs
    )

    optimizer = nnx.ModelAndOptimizer(model, optax.sgd(lr_schedule, momentum, nesterov=True))

    eval_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
        accuracy=nnx.metrics.Accuracy(),
    )


    train_metrics_history = {
        "train_loss": [],
    }

    eval_metrics_history = {
        "val_loss": [],
        "val_accuracy": [],
    }

    def train_one_epoch(epoch):
        model.train()  # Set model to the training mode: e.g. update batch statistics
        with tqdm.tqdm(
            desc=f"[train] epoch: {epoch}/{num_epochs}, ",
            total=total_steps,
            bar_format=bar_format,
            leave=True,
        ) as pbar:
            for batch in train_loader:
                loss = train_step(model, optimizer, batch)
                train_metrics_history["train_loss"].append(loss.item())
                pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)

    def evaluate_model(epoch):
        # Computes the metrics on the training and test sets after each training epoch.
        model.eval()  # Sets model to evaluation model: e.g. use stored batch statistics.

        eval_metrics.reset()  # Reset the eval metrics
        for val_batch in val_loader:
            eval_step(model, val_batch, eval_metrics)

        for metric, value in eval_metrics.compute().items():
            eval_metrics_history[f'val_{metric}'].append(value)

        print(f"[val] epoch: {epoch + 1}/{num_epochs}")
        print(f"- total loss: {eval_metrics_history['val_loss'][-1]:0.4f}")
        print(f"- Accuracy: {eval_metrics_history['val_accuracy'][-1]:0.4f}")

    # %%time

    for epoch in range(num_epochs):
        train_one_epoch(epoch)
        evaluate_model(epoch)
