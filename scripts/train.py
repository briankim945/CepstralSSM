# from datasets import load_dataset
# import grain.python as grain
# # import matplotlib.pyplot as plt

import jax
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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

if __name__ == "__main__":
    # train_dataset = load_dataset("/gpfs/data/shared/imagenet/ILSVRC2012")
    # val_dataset = load_dataset("/gpfs/data/shared/imagenet/ILSVRC2012") 
    # test_dataset = load_dataset("/gpfs/data/shared/imagenet/ILSVRC2012") 

    # print("Training dataset size:", len(train_dataset))
    # print("Validation dataset size:", len(val_dataset))
    # print("Test dataset size:", len(test_dataset))

    # # Create an `grain.IndexSampler` with no sharding for single-device computations.
    # train_sampler = grain.IndexSampler(
    #     len(train_dataset),  # The total number of samples in the data source.
    #     shuffle=True,            # Shuffle the data to randomize the order.of samples
    #     seed=seed,               # Set a seed for reproducibility.
    #     shard_options=grain.NoSharding(),  # No sharding since this is a single-device setup.
    #     num_epochs=1,            # Iterate over the dataset for one epoch.
    # )

    # # val_sampler = grain.IndexSampler(
    # #     len(val_dataset),  # The total number of samples in the data source.
    # #     shuffle=False,         # Do not shuffle the data.
    # #     seed=seed,             # Set a seed for reproducibility.
    # #     shard_options=grain.NoSharding(),  # No sharding since this is a single-device setup.
    # #     num_epochs=1,          # Iterate over the dataset for one epoch.
    # # )


    # train_loader = grain.DataLoader(
    #     data_source=train_dataset,
    #     sampler=train_sampler,                 # A sampler to determine how to access the data.
    #     worker_count=4,                        # Number of child processes launched to parallelize the transformations among.
    #     worker_buffer_size=2,                  # Count of output batches to produce in advance per worker.
    #     operations=[
    #         grain.Batch(train_batch_size, drop_remainder=True),
    #     ]
    # )

    # # # Test (validation) dataset `grain.DataLoader`.
    # # val_loader = grain.DataLoader(
    # #     data_source=val_dataset,
    # #     sampler=val_sampler,                   # A sampler to determine how to access the data.
    # #     worker_count=4,                        # Number of child processes launched to parallelize the transformations among.
    # #     worker_buffer_size=2,
    # #     operations=[
    # #         grain.Batch(val_batch_size),
    # #     ]
    # # )

    # Load the dataset (point root to where ImageNet is downloaded)
    # You need your ImageNet data in the correct folder structure
    imagenet_data = datasets.ImageNet(root='/gpfs/data/shared/imagenet/ILSVRC2012', split='train', transform=transform)

    # Use PyTorch DataLoader
    train_loader = DataLoader(imagenet_data, batch_size=64, shuffle=True, num_workers=4)

    train_batch = torch_to_jax(next(iter(train_loader)))
    # val_batch = next(iter(val_loader))

    print("Training batch info:", train_batch["image"].shape, train_batch["image"].dtype, train_batch["label"].shape, train_batch["label"].dtype)
    # print("Validation batch info:", val_batch["image"].shape, val_batch["image"].dtype, val_batch["label"].shape, val_batch["label"].dtype)
