import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, List, Optional, Callable
import json
import pickle
from PIL import Image


class FileDataLoader:
    """
    Base dataloader for reading files from a directory.
    
    Supports shuffling, batching, and prefetching.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        shuffle: bool = True,
        file_pattern: str = "*",
        num_epochs: Optional[int] = None,
        prefetch_size: int = 2,
        drop_last: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing data files
            batch_size: Batch size
            shuffle: Whether to shuffle data
            file_pattern: Glob pattern to match files (e.g., "*.npy", "*.jpg")
            num_epochs: Number of epochs to iterate (None = infinite)
            prefetch_size: Number of batches to prefetch
            drop_last: Drop last incomplete batch
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_pattern = file_pattern
        self.num_epochs = num_epochs
        self.prefetch_size = prefetch_size
        self.drop_last = drop_last
        
        # Find all matching files
        self.files = sorted(self.data_dir.glob(file_pattern))
        if not self.files:
            raise ValueError(f"No files found matching {file_pattern} in {data_dir}")
        
        print(f"Found {len(self.files)} files in {data_dir}")
    
    def load_file(self, filepath: Path):
        """Override this method to define how to load individual files."""
        raise NotImplementedError("Subclasses must implement load_file()")
    
    def __iter__(self) -> Iterator:
        """Iterate over batches."""
        epoch = 0
        
        while self.num_epochs is None or epoch < self.num_epochs:
            # Shuffle files if requested
            files = self.files.copy()
            if self.shuffle:
                np.random.shuffle(files)
            
            # Load and batch
            batch_data = []
            
            for filepath in files:
                # Load single file
                data = self.load_file(filepath)
                batch_data.append(data)
                
                # Yield batch when full
                if len(batch_data) == self.batch_size:
                    yield self._collate_batch(batch_data)
                    batch_data = []
            
            # Handle remaining data
            if batch_data and not self.drop_last:
                yield self._collate_batch(batch_data)
            
            epoch += 1
    
    def _collate_batch(self, batch_data: List) -> Tuple:
        """
        Collate list of data into batched arrays.
        Override for custom collation.
        """
        # Simple case: assume each item is a tuple (input, label)
        if isinstance(batch_data[0], tuple):
            inputs = jnp.stack([item[0] for item in batch_data])
            labels = jnp.stack([item[1] for item in batch_data])
            return inputs, labels
        else:
            return jnp.stack(batch_data)


class NumpyDataLoader(FileDataLoader):
    """DataLoader for .npy or .npz files."""
    
    def __init__(self, *args, label_key: Optional[str] = None, **kwargs):
        """
        Args:
            label_key: For .npz files, key to use for labels (None = no labels)
        """
        super().__init__(*args, file_pattern="*.np[yz]", **kwargs)
        self.label_key = label_key
    
    def load_file(self, filepath: Path):
        """Load a single .npy or .npz file."""
        if filepath.suffix == ".npy":
            data = np.load(filepath)
            return jnp.array(data)
        
        elif filepath.suffix == ".npz":
            data = np.load(filepath)
            if self.label_key:
                # Return (input, label) tuple
                input_data = jnp.array(data['data'])  # Assume 'data' key for input
                label_data = jnp.array(data[self.label_key])
                return input_data, label_data
            else:
                # Return just the first array
                return jnp.array(data[data.files[0]])


class ImageDataLoader(FileDataLoader):
    """DataLoader for image files (.jpg, .png, etc.)."""
    
    def __init__(
        self,
        *args,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        label_from_dirname: bool = False,
        class_to_idx: Optional[dict] = None,
        **kwargs
    ):
        """
        Args:
            image_size: Resize images to this size (H, W)
            normalize: Normalize to [0, 1]
            label_from_dirname: Use directory name as label
            class_to_idx: Mapping from class names to indices
        """
        super().__init__(*args, file_pattern="*.jpg", **kwargs)
        self.image_size = image_size
        self.normalize = normalize
        self.label_from_dirname = label_from_dirname
        self.class_to_idx = class_to_idx
        
        # Also include png, jpeg
        png_files = list(self.data_dir.glob("*.png"))
        jpeg_files = list(self.data_dir.glob("*.jpeg"))
        self.files.extend(png_files + jpeg_files)
        self.files = sorted(self.files)
        
        if label_from_dirname and class_to_idx is None:
            # Auto-create class mapping
            class_names = sorted(set(f.parent.name for f in self.files))
            self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            print(f"Found {len(self.class_to_idx)} classes: {list(self.class_to_idx.keys())}")
    
    def load_file(self, filepath: Path):
        """Load and preprocess an image."""
        # Load image
        img = Image.open(filepath).convert('RGB')
        
        # Resize
        img = img.resize(self.image_size)
        
        # Convert to array
        img_array = np.array(img)
        
        # Normalize
        if self.normalize:
            img_array = img_array.astype(np.float32) / 255.0
        
        img_jax = jnp.array(img_array)
        
        # Get label if using directory structure
        if self.label_from_dirname:
            class_name = filepath.parent.name
            label = self.class_to_idx[class_name]
            return img_jax, jnp.array(label)
        
        return img_jax


class SequenceDataLoader(FileDataLoader):
    """
    DataLoader for sequence data (e.g., time series, text).
    Expects files containing sequences that may need padding.
    """
    
    def __init__(
        self,
        *args,
        max_seq_len: Optional[int] = None,
        pad_value: float = 0.0,
        return_mask: bool = True,
        **kwargs
    ):
        """
        Args:
            max_seq_len: Maximum sequence length (pad/truncate to this)
            pad_value: Value to use for padding
            return_mask: Return attention mask
        """
        super().__init__(*args, **kwargs)
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.return_mask = return_mask
    
    def load_file(self, filepath: Path):
        """Load sequence data."""
        # Assume .npy files containing sequences
        seq = np.load(filepath)
        
        # Handle padding/truncation
        if self.max_seq_len:
            original_len = len(seq)
            
            if len(seq) > self.max_seq_len:
                seq = seq[:self.max_seq_len]
                mask = jnp.ones(self.max_seq_len)
            else:
                # Pad
                pad_len = self.max_seq_len - len(seq)
                seq = np.pad(seq, ((0, pad_len), (0, 0)) if seq.ndim > 1 else (0, pad_len),
                           constant_values=self.pad_value)
                # Create mask
                mask = jnp.concatenate([
                    jnp.ones(original_len),
                    jnp.zeros(pad_len)
                ])
        else:
            mask = jnp.ones(len(seq))
        
        seq_jax = jnp.array(seq)
        
        if self.return_mask:
            return seq_jax, mask
        return seq_jax
    
    def _collate_batch(self, batch_data: List):
        """Collate sequences with masks."""
        if self.return_mask:
            sequences, masks = zip(*batch_data)
            return jnp.stack(sequences), jnp.stack(masks)
        else:
            return jnp.stack(batch_data)


class InMemoryDataLoader:
    """
    Efficient dataloader that loads all data into memory first.
    Good for datasets that fit in RAM.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        rng_key: Optional[jax.Array] = None,
    ):
        """
        Args:
            data: Input data array (N, ...)
            labels: Label array (N, ...), optional
            batch_size: Batch size
            shuffle: Whether to shuffle
            rng_key: JAX random key for shuffling
        """
        self.data = jnp.array(data)
        self.labels = jnp.array(labels) if labels is not None else None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng_key = rng_key if rng_key is not None else random.PRNGKey(0)
        
        self.num_samples = len(self.data)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
    
    def __iter__(self) -> Iterator:
        """Iterate over batches."""
        # Shuffle indices if requested
        if self.shuffle:
            self.rng_key, subkey = random.split(self.rng_key)
            indices = random.permutation(subkey, self.num_samples)
            data = self.data[indices]
            labels = self.labels[indices] if self.labels is not None else None
        else:
            data = self.data
            labels = self.labels
        
        # Yield batches
        for i in range(0, self.num_samples, self.batch_size):
            batch_data = data[i:i + self.batch_size]
            
            if labels is not None:
                batch_labels = labels[i:i + self.batch_size]
                yield batch_data, batch_labels
            else:
                yield batch_data
    
    def __len__(self):
        return self.num_batches


def load_dataset_from_directory(
    data_dir: str,
    batch_size: int = 32,
    file_type: str = 'auto',
    **kwargs
) -> FileDataLoader:
    """
    Convenience function to automatically create the right dataloader.
    
    Args:
        data_dir: Directory path
        batch_size: Batch size
        file_type: 'auto', 'numpy', 'image', 'csv', 'sequence'
        **kwargs: Additional arguments for specific loaders
    
    Returns:
        Appropriate DataLoader instance
    """
    data_path = Path(data_dir)
    
    if file_type == 'auto':
        # Auto-detect based on files in directory
        files = list(data_path.glob('*'))
        if not files:
            raise ValueError(f"No files found in {data_dir}")
        
        first_file = files[0]
        if first_file.suffix in ['.npy', '.npz']:
            file_type = 'numpy'
        elif first_file.suffix in ['.jpg', '.png', '.jpeg']:
            file_type = 'image'
        elif first_file.suffix == '.csv':
            file_type = 'csv'
        else:
            raise ValueError(f"Cannot auto-detect file type from {first_file.suffix}")
    
    # Create appropriate loader
    if file_type == 'numpy':
        return NumpyDataLoader(data_dir, batch_size, **kwargs)
    elif file_type == 'image':
        return ImageDataLoader(data_dir, batch_size, **kwargs)
    elif file_type == 'csv':
        return CSVDataLoader(data_dir, batch_size, **kwargs)
    elif file_type == 'sequence':
        return SequenceDataLoader(data_dir, batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown file_type: {file_type}")


# Example usage demonstrations
if __name__ == "__main__":
    print("JAX DataLoader Examples\n" + "=" * 50)
    
    # Example 1: Create some dummy data
    print("\n1. Creating dummy numpy data...")
    dummy_dir = Path("dummy_data")
    dummy_dir.mkdir(exist_ok=True)
    
    for i in range(20):
        data = np.random.randn(64, 10)  # (seq_len, features)
        np.save(dummy_dir / f"sample_{i:03d}.npy", data)
    
    print(f"   Created 20 .npy files in {dummy_dir}")
    
    # Example 2: Use NumpyDataLoader
    print("\n2. Using NumpyDataLoader...")
    loader = NumpyDataLoader(
        data_dir=dummy_dir,
        batch_size=4,
        shuffle=True,
        num_epochs=1
    )
    
    for batch_idx, batch in enumerate(loader):
        print(f"   Batch {batch_idx}: shape = {batch.shape}")
        if batch_idx >= 2:  # Just show first 3 batches
            break
    
    # Example 3: InMemoryDataLoader
    print("\n3. Using InMemoryDataLoader...")
    data = np.random.randn(100, 32, 10)
    labels = np.random.randint(0, 5, size=100)
    
    mem_loader = InMemoryDataLoader(
        data=data,
        labels=labels,
        batch_size=16,
        shuffle=True,
        rng_key=random.PRNGKey(42)
    )
    
    for batch_idx, (batch_data, batch_labels) in enumerate(mem_loader):
        print(f"   Batch {batch_idx}: data={batch_data.shape}, labels={batch_labels.shape}")
        if batch_idx >= 2:
            break
    
    print("\n✓ DataLoader examples complete!")
    print(f"  Clean up with: rm -rf {dummy_dir}")