import os
from collections import defaultdict

def create_mappings(images_dir, labels_dir):
    # Collect image files and label files
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    label_files = [f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))]

    # Dictionary: code -> list of images / labels
    images_map = defaultdict(list)
    labels_map = defaultdict(list)

    # Group by first 8 characters
    for img in image_files:
        code = img[:8]
        images_map[code].append(img)

    for lbl in label_files:
        code = lbl[:8]
        labels_map[code].append(lbl)

    # Create final mapping: code -> {images, labels}
    mappings = {}
    for code in set(images_map.keys()).union(labels_map.keys()):
        mappings[code] = {
            "images": images_map.get(code, []),
            "labels": labels_map.get(code, [])
        }

    return mappings


if __name__ == "__main__":
    images_dir = "images"   # 🔹 change to your images directory
    labels_dir = "labels"   # 🔹 change to your labels directory

    mappings = create_mappings(images_dir, labels_dir)

    # Print mapping
    count=0
    for code, files in mappings.items():
        print(f"\nCode: {code}")
        print(f"  Images: {files['images']}")
        print(f"  Labels: {files['labels']}")
        count=count+1
        print(count)