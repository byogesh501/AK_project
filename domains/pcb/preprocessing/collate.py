"""
Custom collate function for DeepPCB dataset batching.

Handles variable-length annotations by padding to the maximum count
in each batch and providing a mask tensor.
"""

from typing import Dict, List

import torch


def pcb_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate a list of DeepPCBDataset samples into a batched dict.

    Image tensors are stacked normally.  Annotations, which vary in length
    per image, are padded to the maximum count in the batch and accompanied
    by a boolean mask (True = real annotation, False = padding).

    Expected input per sample (after transform):
        - defect_image:   Tensor [C, H, W]
        - template_image: Tensor [C, H, W]
        - annotations:    list of dicts with float values
        - pair_id:        str
        - group:          str
        - classes:        set of str

    Returns:
        Dict with keys:
            - defect_images:   Tensor [B, C, H, W]
            - template_images: Tensor [B, C, H, W]
            - annotations:     Tensor [B, max_annos, 5]
                               columns: class_id, x_center, y_center, width, height
            - anno_mask:       BoolTensor [B, max_annos]
            - pair_ids:        list[str] of length B
            - groups:          list[str] of length B
            - classes:         list[set[str]] of length B
    """
    defect_images = torch.stack([s['defect_image'] for s in batch])
    template_images = torch.stack([s['template_image'] for s in batch])

    pair_ids = [s['pair_id'] for s in batch]
    groups = [s['group'] for s in batch]
    classes = [s['classes'] for s in batch]

    # Pad annotations to max count in this batch
    anno_lists = [s['annotations'] for s in batch]
    max_annos = max(len(a) for a in anno_lists) if anno_lists else 0
    # Guarantee at least 1 slot so the tensor has a valid shape
    max_annos = max(max_annos, 1)

    batch_size = len(batch)
    annotations = torch.zeros(batch_size, max_annos, 5)
    anno_mask = torch.zeros(batch_size, max_annos, dtype=torch.bool)

    for i, annos in enumerate(anno_lists):
        for j, anno in enumerate(annos):
            annotations[i, j, 0] = anno['class_id']
            annotations[i, j, 1] = anno['x_center']
            annotations[i, j, 2] = anno['y_center']
            annotations[i, j, 3] = anno['width']
            annotations[i, j, 4] = anno['height']
            anno_mask[i, j] = True

    return {
        'defect_images': defect_images,
        'template_images': template_images,
        'annotations': annotations,
        'anno_mask': anno_mask,
        'pair_ids': pair_ids,
        'groups': groups,
        'classes': classes,
    }
