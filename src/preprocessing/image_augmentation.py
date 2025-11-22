"""Image augmentation utilities for OCR training"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class ImageAugmentation:
    """Image augmentation for OCR training data"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize image augmentation.

        Args:
            config: Augmentation configuration dictionary
        """
        self.config = config or {}
        self.augmentation_pipeline = self._build_pipeline()

    def _build_pipeline(self) -> A.Compose:
        """Build augmentation pipeline from configuration"""
        transforms = []

        # Rotation
        if self.config.get("rotation_range"):
            rotation_range = self.config["rotation_range"]
            if isinstance(rotation_range, list) and len(rotation_range) == 2:
                transforms.append(
                    A.Rotate(limit=rotation_range, border_mode=0, value=0, p=0.5)
                )

        # Brightness
        if self.config.get("brightness_range"):
            brightness_range = self.config["brightness_range"]
            if isinstance(brightness_range, list) and len(brightness_range) == 2:
                transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=(brightness_range[0] - 1, brightness_range[1] - 1),
                        contrast_limit=0.2,
                        p=0.5,
                    )
                )

        # Blur
        if self.config.get("blur_probability", 0) > 0:
            transforms.append(A.GaussianBlur(blur_limit=3, p=self.config["blur_probability"]))

        # Noise
        if self.config.get("noise_probability", 0) > 0:
            transforms.append(
                A.GaussNoise(var_limit=(10.0, 50.0), p=self.config["noise_probability"])
            )

        # Perspective/affine transformations
        transforms.append(A.Perspective(scale=(0.05, 0.1), p=0.3))

        # Elastic transform (simulates document warping)
        transforms.append(A.ElasticTransform(alpha=1, sigma=50, p=0.3))

        return A.Compose(transforms)

    def augment_image(self, image: np.ndarray, bboxes: Optional[List[List[float]]] = None) -> Tuple[np.ndarray, Optional[List[List[float]]]]:
        """
        Augment a single image.

        Args:
            image: Input image as numpy array (H, W, C)
            bboxes: Optional list of bounding boxes in format [x1, y1, x2, y2]

        Returns:
            Tuple of (augmented_image, transformed_bboxes)
        """
        if bboxes:
            # Convert bboxes to albumentations format: [[x1, y1, x2, y2], ...]
            # Albumentations expects normalized coordinates for some transforms
            height, width = image.shape[:2]
            normalized_bboxes = []
            for bbox in bboxes:
                normalized_bboxes.append(
                    [
                        bbox[0] / width,  # x1
                        bbox[1] / height,  # y1
                        bbox[2] / width,  # x2
                        bbox[3] / height,  # y2
                    ]
                )

            transformed = self.augmentation_pipeline(
                image=image, bboxes=normalized_bboxes, bbox_params=A.BboxParams(format="pascal_voc")
            )

            # Convert back to pixel coordinates
            transformed_bboxes = []
            for bbox in transformed["bboxes"]:
                transformed_bboxes.append(
                    [
                        bbox[0] * width,
                        bbox[1] * height,
                        bbox[2] * width,
                        bbox[3] * height,
                    ]
                )

            return transformed["image"], transformed_bboxes
        else:
            transformed = self.augmentation_pipeline(image=image)
            return transformed["image"], None

    def augment_image_file(
        self, image_path: str, output_path: str, bboxes: Optional[List[List[float]]] = None
    ):
        """
        Augment an image file and save to output path.

        Args:
            image_path: Path to input image
            output_path: Path to save augmented image
            bboxes: Optional list of bounding boxes
        """
        image = np.array(Image.open(image_path).convert("RGB"))
        augmented_image, transformed_bboxes = self.augment_image(image, bboxes)

        Image.fromarray(augmented_image).save(output_path)
        return transformed_bboxes

