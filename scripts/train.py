from datasets import load_dataset
# import matplotlib.pyplot as plt


# def display_datapoints(*datapoints, tag="", names_map=None):
#     num_samples = len(datapoints)

#     fig, axs = plt.subplots(1, num_samples, figsize=(20, 10))
#     for i, datapoint in enumerate(datapoints):
#         if isinstance(datapoint, dict):
#             img, label = datapoint["image"], datapoint["label"]
#         else:
#             img, label = datapoint

#         if hasattr(img, "dtype") and img.dtype in (np.float32, ):
#             img = ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)

#         label_str = f" ({names_map[label]})" if names_map is not None else ""
#         axs[i].set_title(f"{tag}Label: {label}{label_str}")
#         axs[i].imshow(img)


# Select first 20 classes to reduce the dataset size and the training time.
train_size = 20 * 750
val_size = 20 * 250

train_dataset = load_dataset("/gpfs/data/shared/imagenet/ILSVRC2012/train") #load_dataset("food101", split=f"train[:{train_size}]")
val_dataset = load_dataset("/gpfs/data/shared/imagenet/ILSVRC2012/val") #load_dataset("food101", split=f"validation[:{val_size}]")

# Create labels mapping where we map current labels between 0 and 19.
labels_mapping = {}
index = 0
for i in range(0, len(val_dataset), 250):
    label = val_dataset[i]["label"]
    if label not in labels_mapping:
        labels_mapping[label] = index
        index += 1

inv_labels_mapping = {v: k for k, v in labels_mapping.items()}

print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))

# display_datapoints(
#     train_dataset[0], train_dataset[1000], train_dataset[2000], train_dataset[3000],
#     tag="(Training) ",
#     names_map=train_dataset.features["label"].names
# )

# display_datapoints(
#     val_dataset[0], val_dataset[1000], val_dataset[2000], val_dataset[-1],
#     tag="(Validation) ",
#     names_map=val_dataset.features["label"].names
# )
