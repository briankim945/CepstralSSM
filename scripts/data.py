from typing import Union, Path
import os

from torchvision import datasets


# Imagenette class folder names (10 classes)
IMAGENETTE_CLASSES = [
    'n01440764',  # tench
    'n02102040',  # English springer
    'n02979186',  # cassette player
    'n03000684',  # chain saw
    'n03028079',  # church
    'n03394916',  # French horn
    'n03417042',  # garbage truck
    'n03425413',  # gas pump
    'n03445777',  # golf ball
    'n03888257',  # parachute
]

# Human-readable class names
CLASS_NAMES = [
    'tench', 'English springer', 'cassette player', 'chain saw',
    'church', 'French horn', 'garbage truck', 'gas pump',
    'golf ball', 'parachute'
]


class ImageNetteFolder(datasets.ImageFolder):

    def find_classes(directory: Union[str, Path]) -> tuple[list[str], dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes) if cls_name in IMAGENETTE_CLASSES}
        return classes, class_to_idx
