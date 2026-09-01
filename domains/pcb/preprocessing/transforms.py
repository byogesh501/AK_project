"""
Preprocessing and augmentation transforms for PCB defect detection.

Handles paired defect/template images and preserves bounding box annotations.
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Tuple, List, Dict, Optional
import random
from PIL import Image


class PairedToTensor:
    """Convert paired PIL Images to tensors."""

    def __call__(self, defect_img: Image.Image, template_img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert both images to tensors [C, H, W] with values in [0, 1].
        """
        return TF.to_tensor(defect_img), TF.to_tensor(template_img)


class PairedNormalize:
    """Normalize paired images with same mean/std."""

    def __init__(self, mean: List[float] = [0.485, 0.456, 0.406], std: List[float] = [0.229, 0.224, 0.225]):
        """
        Args:
            mean: Channel means for normalization
            std: Channel standard deviations for normalization
        """
        self.mean = mean
        self.std = std

    def __call__(self, defect_tensor: torch.Tensor, template_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize both tensors."""
        defect_norm = TF.normalize(defect_tensor, self.mean, self.std)
        template_norm = TF.normalize(template_tensor, self.mean, self.std)
        return defect_norm, template_norm


class PairedRandomHorizontalFlip:
    """Randomly flip paired images horizontally with bounding box adjustment."""

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of flipping
        """
        self.p = p

    def __call__(
        self,
        defect_img: Image.Image,
        template_img: Image.Image,
        annotations: List[Dict]
    ) -> Tuple[Image.Image, Image.Image, List[Dict]]:
        """
        Flip both images and adjust bounding boxes.

        Returns:
            Tuple of (defect_img, template_img, updated_annotations)
        """
        if random.random() < self.p:
            defect_img = TF.hflip(defect_img)
            template_img = TF.hflip(template_img)

            # Update annotations: x_center = 1 - x_center for horizontal flip
            flipped_annotations = []
            for anno in annotations:
                flipped_anno = anno.copy()
                flipped_anno['x_center'] = 1.0 - anno['x_center']
                flipped_annotations.append(flipped_anno)
            return defect_img, template_img, flipped_annotations

        return defect_img, template_img, annotations


class PairedRandomVerticalFlip:
    """Randomly flip paired images vertically with bounding box adjustment."""

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of flipping
        """
        self.p = p

    def __call__(
        self,
        defect_img: Image.Image,
        template_img: Image.Image,
        annotations: List[Dict]
    ) -> Tuple[Image.Image, Image.Image, List[Dict]]:
        """
        Flip both images vertically and adjust bounding boxes.

        Returns:
            Tuple of (defect_img, template_img, updated_annotations)
        """
        if random.random() < self.p:
            defect_img = TF.vflip(defect_img)
            template_img = TF.vflip(template_img)

            # Update annotations: y_center = 1 - y_center for vertical flip
            flipped_annotations = []
            for anno in annotations:
                flipped_anno = anno.copy()
                flipped_anno['y_center'] = 1.0 - anno['y_center']
                flipped_annotations.append(flipped_anno)
            return defect_img, template_img, flipped_annotations

        return defect_img, template_img, annotations


class PairedRandomRotation:
    """Randomly rotate paired images by 90-degree increments with bbox adjustment."""

    def __init__(self, angles: List[int] = [0, 90, 180, 270]):
        """
        Args:
            angles: List of rotation angles (degrees) to choose from
        """
        self.angles = angles

    def __call__(
        self,
        defect_img: Image.Image,
        template_img: Image.Image,
        annotations: List[Dict]
    ) -> Tuple[Image.Image, Image.Image, List[Dict]]:
        """
        Rotate both images and adjust bounding boxes.

        Returns:
            Tuple of (defect_img, template_img, updated_annotations)
        """
        angle = random.choice(self.angles)

        if angle == 0:
            return defect_img, template_img, annotations

        # Rotate images
        defect_img = TF.rotate(defect_img, angle, expand=False)
        template_img = TF.rotate(template_img, angle, expand=False)

        # Update annotations for 90-degree rotations
        rotated_annotations = []
        for anno in annotations:
            rotated_anno = anno.copy()
            x_c, y_c = anno['x_center'], anno['y_center']
            w, h = anno['width'], anno['height']

            if angle == 90:
                # 90° clockwise: (x, y) -> (y, 1-x), swap width/height
                rotated_anno['x_center'] = y_c
                rotated_anno['y_center'] = 1.0 - x_c
                rotated_anno['width'] = h
                rotated_anno['height'] = w
            elif angle == 180:
                # 180°: (x, y) -> (1-x, 1-y), width/height unchanged
                rotated_anno['x_center'] = 1.0 - x_c
                rotated_anno['y_center'] = 1.0 - y_c
            elif angle == 270:
                # 270° clockwise: (x, y) -> (1-y, x), swap width/height
                rotated_anno['x_center'] = 1.0 - y_c
                rotated_anno['y_center'] = x_c
                rotated_anno['width'] = h
                rotated_anno['height'] = w

            rotated_annotations.append(rotated_anno)

        return defect_img, template_img, rotated_annotations


class PairedColorJitter:
    """Apply random color jittering to paired images (preserves spatial sync)."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.1, hue: float = 0.05):
        """
        Args:
            brightness: How much to jitter brightness
            contrast: How much to jitter contrast
            saturation: How much to jitter saturation
            hue: How much to jitter hue
        """
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(
        self,
        defect_img: Image.Image,
        template_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply same color jitter to both images."""
        # Get random jitter params
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.transform.get_params(
                self.transform.brightness,
                self.transform.contrast,
                self.transform.saturation,
                self.transform.hue
            )

        # Apply same transform to both images
        defect_img = TF.adjust_brightness(defect_img, brightness_factor)
        defect_img = TF.adjust_contrast(defect_img, contrast_factor)
        defect_img = TF.adjust_saturation(defect_img, saturation_factor)
        defect_img = TF.adjust_hue(defect_img, hue_factor)

        template_img = TF.adjust_brightness(template_img, brightness_factor)
        template_img = TF.adjust_contrast(template_img, contrast_factor)
        template_img = TF.adjust_saturation(template_img, saturation_factor)
        template_img = TF.adjust_hue(template_img, hue_factor)

        return defect_img, template_img


class TrainTransform:
    """
    Training augmentation pipeline for paired PCB images.
    Applies random spatial and color augmentations while preserving bounding boxes.
    """

    def __init__(
        self,
        hflip_p: float = 0.5,
        vflip_p: float = 0.5,
        rotation_angles: List[int] = [0, 90, 180, 270],
        color_jitter: bool = True,
        normalize: bool = True
    ):
        self.hflip = PairedRandomHorizontalFlip(p=hflip_p)
        self.vflip = PairedRandomVerticalFlip(p=vflip_p)
        self.rotation = PairedRandomRotation(angles=rotation_angles)
        self.color_jitter = PairedColorJitter() if color_jitter else None
        self.to_tensor = PairedToTensor()
        self.normalize = PairedNormalize() if normalize else None

    def __call__(self, defect_img: Image.Image, template_img: Image.Image, annotations: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Apply training augmentations.

        Returns:
            Tuple of (defect_tensor, template_tensor, updated_annotations)
        """
        # Spatial augmentations (affect bounding boxes)
        defect_img, template_img, annotations = self.hflip(defect_img, template_img, annotations)
        defect_img, template_img, annotations = self.vflip(defect_img, template_img, annotations)
        defect_img, template_img, annotations = self.rotation(defect_img, template_img, annotations)

        # Color augmentations (don't affect bounding boxes)
        if self.color_jitter:
            defect_img, template_img = self.color_jitter(defect_img, template_img)

        # Convert to tensors
        defect_tensor, template_tensor = self.to_tensor(defect_img, template_img)

        # Normalize
        if self.normalize:
            defect_tensor, template_tensor = self.normalize(defect_tensor, template_tensor)

        return defect_tensor, template_tensor, annotations


class ValTestTransform:
    """
    Deterministic preprocessing for validation/test.
    No augmentation, only normalization.
    """

    def __init__(self, normalize: bool = True):
        self.to_tensor = PairedToTensor()
        self.normalize = PairedNormalize() if normalize else None

    def __call__(self, defect_img: Image.Image, template_img: Image.Image, annotations: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Apply deterministic preprocessing.

        Returns:
            Tuple of (defect_tensor, template_tensor, annotations)
        """
        # Convert to tensors
        defect_tensor, template_tensor = self.to_tensor(defect_img, template_img)

        # Normalize
        if self.normalize:
            defect_tensor, template_tensor = self.normalize(defect_tensor, template_tensor)

        # Return annotations unchanged
        return defect_tensor, template_tensor, annotations
