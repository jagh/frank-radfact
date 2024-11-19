
"""
 Simplified implementation of the box_metrics logic, including its own version of:
  + NormalizedBox 
  + and related utilities.

Key Features:

    NormalizedBox: Ensures box coordinates are normalized between 0 and 1.
    get_mask_from_boxes: Converts box coordinates into a binary mask of a given size.
    compute_box_metrics: Calculates Intersection over Union (IoU), precision, and recall using binary masks.

Example Usage:

pred_boxes = [
    NormalizedBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5),
    NormalizedBox(x_min=0.6, y_min=0.6, x_max=0.9, y_max=0.9),
]
true_boxes = [
    NormalizedBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6),
]

# Compute Metrics
metrics = compute_box_metrics(pred_boxes, true_boxes)
print("Metrics:", metrics)
"""





import numpy as np
from dataclasses import dataclass

IOU = "iou"
PRECISION = "precision"
RECALL = "recall"


@dataclass
class NormalizedBox:
    """Bounding box normalized to the image size, with coordinates in the range [0, 1]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self):
        if not (0 <= self.x_min <= 1 and 0 <= self.y_min <= 1 and self.x_max <= 1 and self.y_max <= 1):
            raise ValueError(f"Box coordinates must be in range [0, 1]: {self}")


def get_mask_from_boxes(boxes: list[NormalizedBox], mask_size: int = 224) -> np.ndarray:
    """
    Create a pixel mask from a list of normalized boxes.

    :param boxes: A list of NormalizedBox objects.
    :param mask_size: The size of the mask (default 224x224).
    :return: A binary mask with shape (mask_size, mask_size).
    """
    mask = np.zeros((mask_size, mask_size), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = (
            int(box.x_min * mask_size),
            int(box.y_min * mask_size),
            int(box.x_max * mask_size),
            int(box.y_max * mask_size),
        )
        mask[y1:y2, x1:x2] = True
    return mask


def compute_box_metrics(pred_boxes: list[NormalizedBox], true_boxes: list[NormalizedBox], mask_size: int = 224) -> dict:
    """
    Compute IOU, precision, and recall between predicted and ground truth boxes.

    :param pred_boxes: A list of predicted NormalizedBox objects.
    :param true_boxes: A list of ground truth NormalizedBox objects.
    :param mask_size: The size of the masks.
    :return: A dictionary with IOU, precision, and recall values.
    """
    pred_mask = get_mask_from_boxes(pred_boxes, mask_size)
    true_mask = get_mask_from_boxes(true_boxes, mask_size)

    pred_area = pred_mask.sum()
    true_area = true_mask.sum()

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    iou = intersection / union if union > 0 else 0.0
    precision = intersection / pred_area if pred_area > 0 else 0.0
    recall = intersection / true_area if true_area > 0 else 0.0

    return {IOU: iou, PRECISION: precision, RECALL: recall}



# # Example Normalized Boxes
# pred_boxes = [
#     NormalizedBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5),
#     NormalizedBox(x_min=0.6, y_min=0.6, x_max=0.9, y_max=0.9),
# ]
# true_boxes = [
#     NormalizedBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6),
# ]

# # Compute Metrics
# metrics = compute_box_metrics(pred_boxes, true_boxes)
# print("Metrics:", metrics)