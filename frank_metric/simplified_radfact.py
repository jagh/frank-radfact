"""
Simplified RadFact logic using the new lightweight versions of schema.py and box_metrics.py.

Explanation

    Lightweight Dependencies:
        Uses the GroundedPhrase, NormalizedBox, and EvidencedPhrase classes from the simplified schema.py.
        Uses compute_box_metrics from the simplified box_metrics.py.

    Metric Computation:
        Implements logical, spatial, and grounding metrics using lightweight logic.

    Flexible Input:
        Supports evaluation for multiple examples, where candidates and references are provided as dictionaries.

"""

from frank_metric.schema import GroundedPhraseEvidenced, NormalizedBox, OneWayNLIFractions, RadFactScore
from frank_metric.box_metrics import compute_box_metrics, PRECISION, RECALL
from typing import List, Dict
import numpy as np


class SimplifiedRadFactMetric:
    def __init__(self, image_size: int = 224, box_precision_threshold: float = 0.5):
        """
        Simplified RadFact metric implementation.
        :param image_size: The size of the image (default is 224x224).
        :param box_precision_threshold: Threshold for precision in box entailment.
        """
        self.image_size = image_size
        self.box_precision_threshold = box_precision_threshold

    def _are_boxes_entailed(self, boxes: List[NormalizedBox], evidence_boxes: List[NormalizedBox]) -> bool:
        """
        Check if prediction boxes are sufficiently entailed by evidence boxes.
        """
        if not boxes or not evidence_boxes:
            return False
        metrics = compute_box_metrics(boxes, evidence_boxes, self.image_size)
        return metrics[PRECISION] > self.box_precision_threshold

    def compute_oneway_metrics(self, phrases: List[GroundedPhraseEvidenced], evidence_boxes: List[NormalizedBox]) -> OneWayNLIFractions:
        """
        Compute one-way metrics (e.g., candidate -> reference or vice versa).
        """
        num_phrases = len(phrases)
        entailed_phrases = [p for p in phrases if hasattr(p, "status") and p.status == "entailed"]
        entailed_fraction = len(entailed_phrases) / num_phrases if num_phrases else np.nan

        boxed_phrases = [p for p in phrases if p.boxes]
        entailed_boxes = [p for p in boxed_phrases if self._are_boxes_entailed(p.boxes, evidence_boxes)]
        spatial_entailment_boxes = [p for p in entailed_boxes if self._are_boxes_entailed(p.boxes, evidence_boxes)]

        entailed_box_fraction = len(spatial_entailment_boxes) / len(entailed_boxes) if entailed_boxes else np.nan
        full_box_fraction = len(spatial_entailment_boxes) / len(boxed_phrases) if boxed_phrases else np.nan

        return OneWayNLIFractions(
            entailed_fraction=entailed_fraction,
            full_box_fraction=full_box_fraction,
            entailed_box_fraction=entailed_box_fraction,
            num_phrases=num_phrases,
            num_phrases_with_boxes=len(boxed_phrases),
        )

    def compute_scores(self, candidates: List[GroundedPhraseEvidenced], references: List[GroundedPhraseEvidenced]) -> RadFactScore:
        """
        Compute metrics for candidates and references.
        """
        candidate_boxes = [box for r in references if r.boxes for box in r.boxes]
        reference_boxes = [box for c in candidates if c.boxes for box in c.boxes]

        candidate_metrics = self.compute_oneway_metrics(candidates, reference_boxes)
        reference_metrics = self.compute_oneway_metrics(references, candidate_boxes)

        return RadFactScore.from_candidate_and_reference_fractions(candidate_metrics, reference_metrics)

    def evaluate(self, candidates: Dict[int, List[GroundedPhraseEvidenced]], references: Dict[int, List[GroundedPhraseEvidenced]]) -> Dict[int, RadFactScore]:
        """
        Evaluate RadFact metrics for multiple examples.
        """
        results = {}
        for study_id in candidates:
            results[study_id] = self.compute_scores(candidates[study_id], references[study_id])
        return results
