import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset 
import os 
from PIL import Image 
import albumentations as A
import numpy as np
from colorama import Fore 
from matplotlib import pyplot as plt 
from utils.boxes import rescale_bboxes, stacker
from utils.setup import get_classes
from utils.logger import get_logger
from utils.rich_handlers import DataLoaderHandler
import sys 


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
        
        # Log dataset initialization
        dataset_info = {
            "Dataset Path": self.path,
            "Mode": "Training" if train else "Testing",
            "Total Samples": len(self.labels),
            "Images Path": self.images_path,
            "Labels Path": self.labels_path,
            "First Label File": self.labels[0] if self.labels else "No labels found",
            "First Image Name": os.path.splitext(self.labels[0])[0] + ".jpg" if self.labels else "No images found"
        }
        self.data_handler.log_dataset_stats(dataset_info)
        
        # Verify paths exist
        if not os.path.exists(self.images_path):
            self.logger.error(f"Images directory not found: {self.images_path}")
        if not os.path.exists(self.labels_path):
            self.logger.error(f"Labels directory not found: {self.labels_path}")
        
        # Log transforms information
        transform_list = [
            "Resize to 500x500",
            "Random Crop 224x224 (training only)",
            "Final Resize to 224x224",
            "Horizontal Flip p=0.5 (training only)",
            "Color Jitter (training only)",
            "Normalize (ImageNet stats)",
            "Convert to Tensor"
        ]
        self.data_handler.log_transform_info(transform_list)             

    def safe_transform(self, image, bboxes, labels, max_attempts=50):
        self.transform = A.Compose(
            [   
                A.Resize(500,500),
                *([A.RandomCrop(width=224, height=224, p=0.33)] if self.train else []), # Example random crop
                A.Resize(224,224),
                *([A.HorizontalFlip(p=0.5)] if self.train else []),
                *([A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5)] if self.train else []),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
        
        for attempt in range(max_attempts):
            try:
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
                # Check if we still have bboxes after transformation
                if len(transformed['bboxes']) > 0:
                    return transformed
            except:
                continue
        
        return {'image': image, 'bboxes': bboxes, 'class_labels': labels}

    def __len__(self): 
        return len(self.labels) 

    def __getitem__(self, idx): 
        self.label_path = os.path.join(self.labels_path, self.labels[idx]) 
        label_file = self.labels[idx]
        
        # Handle different naming conventions
        if 'train_dataset' in self.path:
            # For train_dataset, use base name directly
            self.image_name = os.path.splitext(label_file)[0]
        else:
            # For original data structure (data/train and data/test)
            # Remove the UUID prefix if it exists
            parts = label_file.split('-', 1)
            self.image_name = parts[1].split('.')[0] if len(parts) > 1 else parts[0].split('.')[0]
            
        self.image_path = os.path.join(self.images_path, f'{self.image_name}.jpg')
        
        if not os.path.exists(self.image_path):
            self.logger.error(f"Image not found: {self.image_path}")
            self.logger.error(f"Label file: {label_file}")
            raise FileNotFoundError(f"Image not found: {self.image_path}")
            
        img = Image.open(self.image_path)
        with open(self.label_path, 'r') as f: 
            annotations = f.readlines()
        class_labels = []
        bounding_boxes = []
        for annotation in annotations: 
            annotation = annotation.split('\n')[:-1][0].split(' ')
            class_labels.append(annotation[0]) 
            bounding_boxes.append(annotation[1:])
        class_labels = np.array(class_labels).astype(int) 
        bounding_boxes = np.array(bounding_boxes).astype(float) 

        augmented = self.safe_transform(image=np.array(img), bboxes=bounding_boxes, labels=class_labels)
        augmented_img_tensor = augmented['image']
        augmented_bounding_boxes = np.array(augmented['bboxes'])
        augmented_classes = augmented['class_labels']

        labels = torch.tensor(augmented_classes, dtype=torch.long)  
        boxes = torch.tensor(augmented_bounding_boxes, dtype=torch.float32)
        return augmented_img_tensor, {'labels': labels, 'boxes': boxes}

if __name__ == '__main__':
    dataset = DETRData('data/train_dataset', train=False) 
    dataloader = DataLoader(dataset, collate_fn=stacker, batch_size=4, drop_last=True)

    X, y = next(iter(dataloader))
    print(Fore.LIGHTCYAN_EX + str(y) + Fore.RESET) 
    CLASSES = get_classes() 
    fig, ax = plt.subplots(2,2) 
    axs = ax.flatten()
    for idx, (img, annotations, ax) in enumerate(zip(X, y, axs)): 
        ax.imshow(img.permute(1,2,0))
        box_classes = annotations['labels'] 
        boxes = rescale_bboxes(annotations['boxes'], (224,224))
        for box_class, bbox in zip(box_classes, boxes): 
            if box_class != 3: 
                xmin, ymin, xmax, ymax = bbox.detach().numpy()
                print(xmin, ymin, xmax, ymax) 
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=(0.000, 0.447, 0.741), linewidth=3))
                text = f'{CLASSES[box_class]}'
                ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    fig.tight_layout() 
    plt.show()     