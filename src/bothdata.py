import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from colorama import Fore
from matplotlib import pyplot as plt
from utils.boxes import rescale_bboxes, stacker
from utils.setup import get_classes
from utils.logger import get_logger
from utils.rich_handlers import DataLoaderHandler


class DETRData(Dataset):
    def __init__(self, path, train=True):
        super().__init__()
        self.path = path
        self.labels_path = os.path.join(self.path, 'labels')
        self.images_path = os.path.join(self.path, 'images')
        self.label_files = os.listdir(self.labels_path)
        self.labels = list(filter(lambda x: x.endswith('.txt'), self.label_files))
        self.train = train

        # Initialize logger
        self.logger = get_logger("data_loader")
        self.data_handler = DataLoaderHandler()

        # Consistency check
        self._check_files()

        dataset_info = {
            "Dataset Path": self.path,
            "Mode": "Training" if train else "Testing",
            "Total Samples": len(self.labels),
            "Images Path": self.images_path,
            "Labels Path": self.labels_path
        }
        self.data_handler.log_dataset_stats(dataset_info)

        transform_list = [
            "Resize to 224x224",
            "Horizontal Flip p=0.5 (training only)",
            "Color Jitter (training only)",
            "Normalize (ImageNet stats)",
            "Convert to Tensor"
        ]
        self.data_handler.log_transform_info(transform_list)

        # Build class mapping dynamically (id -> name)
        self.id2name = self._infer_id_to_name()
        if self.id2name:
            max_id = max(self.id2name.keys())
            self.class_names = [self.id2name.get(i, f"class_{i}") for i in range(max_id + 1)]
        else:
            self.class_names = []

    def _label_to_image_stem(self, label_fname: str) -> str:
        """
        Handles both naming conventions:
        1. Friend's format: 'abcd1234....txt' -> 'abcd1234....jpg'
        2. Your format: 'uuid-again-xxxx.txt' -> 'again-xxxx.jpg'
        """
        parts = label_fname.split("-")
        if len(parts[0]) == 8:  # UUID prefix exists → your dataset
            return "-".join(parts[1:]).replace(".txt", "")
        else:  # Friend’s dataset (direct name match)
            return label_fname.replace(".txt", "")

    def _infer_id_to_name(self) -> dict:
        """Infer {class_id: class_name} from filenames and labels."""
        id2name = {}
        for lf in self.labels:
            img_stem = self._label_to_image_stem(lf)
            class_name_from_fname = img_stem.split("-")[0]
            with open(os.path.join(self.labels_path, lf), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    if cid not in id2name:
                        id2name[cid] = class_name_from_fname
                    elif id2name[cid] != class_name_from_fname:
                        self.logger.warning(
                            f"Class id {cid} maps to both '{id2name[cid]}' and "
                            f"'{class_name_from_fname}' (from {lf})."
                        )
                    break
        return id2name

    def _check_files(self):
        """Check if all labels have matching images and vice versa."""
        image_files = set([f.replace(".jpg", "") for f in os.listdir(self.images_path) if f.endswith(".jpg")])
        label_files = set([self._label_to_image_stem(f) for f in self.labels])

        missing_images = label_files - image_files
        missing_labels = image_files - label_files

        if missing_images:
            self.logger.warning(f"{len(missing_images)} label(s) without matching image(s): {list(missing_images)[:5]}...")
        if missing_labels:
            self.logger.warning(f"{len(missing_labels)} image(s) without matching label(s): {list(missing_labels)[:5]}...")

    def safe_transform(self, image, bboxes, labels, max_attempts=50):
        transform = A.Compose(
            [
                A.Resize(224, 224),
                *([A.HorizontalFlip(p=0.5)] if self.train else []),
                *([A.ColorJitter(brightness=0.5, contrast=0.5,
                                 saturation=0.5, hue=0.5, p=0.5)] if self.train else []),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )

        for attempt in range(max_attempts):
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
                if len(transformed['bboxes']) > 0:
                    return transformed
            except Exception:
                continue
        return {'image': image, 'bboxes': bboxes, 'class_labels': labels}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_path = os.path.join(self.labels_path, self.labels[idx])
        image_name = self._label_to_image_stem(self.labels[idx])
        image_path = os.path.join(self.images_path, f"{image_name}.jpg")

        # Check if the image file exists
        if not os.path.exists(image_path):
            self.logger.warning(f"Image file not found: {image_path}. Skipping this sample.")
            return self.__getitem__((idx + 1) % len(self))  # Skip to the next sample

        img = Image.open(image_path).convert("RGB")
        with open(label_path, "r") as f:
            annotations = f.readlines()

        class_labels = []
        bounding_boxes = []
        for annotation in annotations:
            parts = annotation.strip().split()
            class_labels.append(parts[0])
            bounding_boxes.append(parts[1:])
        class_labels = np.array(class_labels).astype(int)
        bounding_boxes = np.array(bounding_boxes).astype(float)

        augmented = self.safe_transform(
            image=np.array(img), bboxes=bounding_boxes, labels=class_labels
        )
        augmented_img_tensor = augmented["image"]
        augmented_bounding_boxes = np.array(augmented["bboxes"])
        augmented_classes = augmented["class_labels"]

        labels = torch.tensor(augmented_classes, dtype=torch.long)
        boxes = torch.tensor(augmented_bounding_boxes, dtype=torch.float32)
        return augmented_img_tensor, {"labels": labels, "boxes": boxes}


if __name__ == "__main__":
    dataset = DETRData("data/train", train=True)
    dataloader = DataLoader(dataset, collate_fn=stacker, batch_size=4, drop_last=True)

    X, y = next(iter(dataloader))
    print(Fore.LIGHTCYAN_EX + str(y) + Fore.RESET)

    CLASSES = dataset.class_names
    print(CLASSES)

    fig, ax = plt.subplots(2, 2)
    axs = ax.flatten()
    for idx, (img, annotations, ax) in enumerate(zip(X, y, axs)):
        ax.imshow(img.permute(1, 2, 0))
        box_classes = annotations["labels"]
        boxes = rescale_bboxes(annotations["boxes"], (224, 224))
        for box_class, bbox in zip(box_classes, boxes):
            if box_class != 3:
                xmin, ymin, xmax, ymax = bbox.detach().numpy()
                ax.add_patch(
                    plt.Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        fill=False,
                        color=(0.000, 0.447, 0.741),
                        linewidth=3,
                    )
                )
                text = f"{CLASSES[box_class]}"
                ax.text(
                    xmin,
                    ymin,
                    text,
                    fontsize=15,
                    bbox=dict(facecolor="yellow", alpha=0.5),
                )

    fig.tight_layout()
    plt.show()