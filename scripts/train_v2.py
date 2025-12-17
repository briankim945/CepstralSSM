# from datasets import load_dataset
# import grain.python as grain
# # import matplotlib.pyplot as plt

import jax
from flax import nnx
import torch
import optax
import numpy as np
from jax import numpy as jnp

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os
from os import path as osp
import tqdm
import wandb
import sys
import argparse
import time

from src.convnext import ConvNeXt
from src.convnext_3d import ConvNeXt3D
from src.conv_cssm import CepstralConvNeXt
from scripts.training_utils import FlatImageFolderDataset


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
    transforms.Resize(192),
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
    model: nnx.Module, optimizer: nnx.ModelAndOptimizer, batch: list,
):
    # Convert np.ndarray to jax.Array on GPU
    images = batch[0]
    labels = batch[1]

    grad_fn = nnx.value_and_grad(compute_losses_and_logits, has_aux=True)
    (loss, logits), grads = grad_fn(model, images, labels)

    optimizer.update(grads)  # In-place updates.

    return loss, logits


@nnx.jit
def eval_step(
    model: nnx.Module, batch: list, eval_metrics: nnx.MultiMetric
):
    # Convert np.ndarray to jax.Array on GPU
    images = batch[0]
    labels = batch[1]
    loss, logits = compute_losses_and_logits(model, images, labels)

    eval_metrics.update(
        loss=loss,
        logits=logits,
        labels=labels,
    )


def main():
    if is_master_process:
        root_dir = "logs"
        save_dir = osp.join(root_dir, args.output_dir)
        os.makedirs(save_dir, exist_ok=True)

        wandb.init(
            project='conv_ssm',
            # config=config,
            dir=save_dir,
            # id=config.run_id,
            id=args.output_dir,
            resume='allow'
        )
        wandb.run.name = args.output_dir
        wandb.run.save(save_dir)

    print("Finding jax devices...")
    print(jax.devices())

    imagenet_data = datasets.ImageFolder(root=osp.join(args.data_dir, 'train'), transform=transform)
    print("Loaded in dataset...")
    train_dataset, val_dataset = torch.utils.data.random_split(imagenet_data, [0.8, 0.2])
    # print(imagenet_data[0], imagenet_data[0][0].shape)

    # Use PyTorch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=1)
    print("Loaded in dataloader...")

    train_features, train_labels = next(iter(train_loader))
    val_features, val_labels = next(iter(val_loader))

    print("Training batch info:", train_features.shape, train_features.dtype, train_labels.shape, train_labels.dtype)
    print("Validation batch info:", val_features.shape, val_features.dtype, val_labels.shape, val_labels.dtype)

    num_epochs = config['epochs'] #100
    learning_rate = 0.001
    momentum = 0.8
    total_steps = len(train_dataset) // train_batch_size
    total_inputs = len(train_dataset)

    lr_schedule = optax.linear_schedule(learning_rate, 0.0, num_epochs * total_steps)

    # CREATE MODEL
    rngs = nnx.Rngs(42)
    ### TODO: Make this modifiable
    if args.type == 'video':
        model = ConvNeXt3D(
            num_classes=1000,
            dims=(192, 192, 384, 768),
            rngs=rngs
        )
    elif args.type == 'cepstral':
        model = CepstralConvNeXt(rngs=rngs)
    else:
        model = ConvNeXt(
            num_classes=1000,
            rngs=rngs
        )

    optimizer = nnx.ModelAndOptimizer(model, optax.sgd(lr_schedule, momentum, nesterov=True), wrt=nnx.Param)

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
            for iteration, batch in enumerate(train_loader):
                if args.type == 'video' or args.type == 'cepstral':
                    batch = [
                        jnp.repeat(
                            jnp.expand_dims(jnp.permute_dims(jnp.asarray(batch[0]), (0,2,3,1)), 1),
                            config["timesteps"], axis=1),
                        jnp.asarray(batch[1], dtype=jnp.int32)
                    ]
                else:
                    batch = [
                        jnp.permute_dims(jnp.asarray(batch[0]), (0,2,3,1)),
                        jnp.asarray(batch[1], dtype=jnp.int32)
                    ]
                loss, logits = train_step(model, optimizer, batch)
                train_metrics_history["train_loss"].append(loss.item())

                if is_master_process and iteration % config['batch_log_interval'] == 0:
                    wandb.log({'train/lr': lr_schedule(epoch)}, step=epoch)
                    wandb.log({'train/loss': loss.item()}, step=epoch)
                    pbar.set_postfix({"loss": loss.item()})
                    pbar.update(config['batch_log_interval']) #pbar.update(1)

    def evaluate_model(epoch):
        # Computes the metrics on the training and test sets after each training epoch.
        model.eval()  # Sets model to evaluation model: e.g. use stored batch statistics.

        eval_metrics.reset()  # Reset the eval metrics
        for val_batch in val_loader:
            if args.type == 'video' or args.type == 'cepstral':
                val_batch = [
                    jnp.repeat(
                                jnp.expand_dims(jnp.permute_dims(jnp.asarray(val_batch[0]), (0,2,3,1)), 1),
                                config["timesteps"], axis=1),
                    jnp.asarray(val_batch[1], dtype=jnp.int32)
                ]
            else:
                val_batch = [
                    jnp.permute_dims(jnp.asarray(val_batch[0]), (0,2,3,1)),
                    jnp.asarray(val_batch[1], dtype=jnp.int32)
                ]
            eval_step(model, val_batch, eval_metrics)

        for metric, value in eval_metrics.compute().items():
            eval_metrics_history[f'val_{metric}'].append(value)

        if is_master_process:
            wandb.log({**{f'eval/{metric}': val
                        for metric, val in eval_metrics.compute().items()}
                    }, step=epoch)

    # %%time

    for epoch in range(num_epochs):
        start = time.time()
        train_one_epoch(epoch)
        train_end = time.time()
        train_time = train_end - start
        wandb.log({
            "epoch_train_time": train_time,
            "train_time_per_image": train_time / total_inputs,
            "throughput/train_samples_per_second": total_inputs / train_time,
        }, step=epoch)
        start = time.time()
        evaluate_model(epoch)
        eval_end = time.time()
        eval_time = eval_end - start
        wandb.log({
            "epoch_eval_time": eval_time,
            "eval_time_per_image": eval_time / total_inputs,
            "throughput/eval_samples_per_second": total_inputs / eval_time,
        }, step=epoch)
    
    print("Completed training...")
    print("Now testing...")

    test_batch_size = 64

    test_dataset = FlatImageFolderDataset(root_dir=osp.join(args.data_dir, 'train'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=1)
    preds = []

    for batch in test_loader:
        images = jnp.permute_dims(jnp.asarray(batch[0]), (0,2,3,1))
        logits = model(images)
        preds.append(logits)

    preds = jnp.stack(preds, axis=0)
    jnp.save(args.output_dir, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='./data_dir')
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('--type', choices=['image', 'video', 'cepstral'], default='image')
    args = parser.parse_args()

    print(jax.process_count())
    is_master_process = jax.process_index() == 0

    config = {
        "epochs": 11,
        "batch_log_interval": 1000,
        "timesteps": 16,
    }

    main()
