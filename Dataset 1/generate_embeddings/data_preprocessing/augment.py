import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path
import torch

input_dir = Path('../dataset1/images')
augmented_dir = Path('../dataset1/augmented_images')
augmented_dir.mkdir(parents=True, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Perspective(scale=(0.05, 0.01), p=1),
    A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT, p=0.5),
    A.Resize(576, 1024),
    ToTensorV2()
])

def augment_and_save(image_path, save_dir):
    if not image_path.exists():
        print(f"Skipping {image_path}, file not found.")
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping {image_path}, unable to read file.")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)['image']

    if isinstance(augmented, torch.Tensor):
        augmented = augmented.permute(1, 2, 0).numpy()

    # cv2.imwrite expects BGR
    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

    augmented_path = save_dir / f"{image_path.stem}_aug.jpg"
    cv2.imwrite(str(augmented_path), augmented)
    print(f"Saved: {augmented_path}")
    return str(augmented_path)

image_extensions = {".jpg", ".jpeg", ".png"}
for image_path in input_dir.iterdir():
    if image_path.suffix.lower() in image_extensions:
        augment_and_save(image_path, augmented_dir)

print("\nAugmentation complete!")
