
"""
Simplified implementation of the schema logic, including the requested versions of 
    + GroundedPhrase, 
    + NormalizedBox, 
    + EvidencedPhrase, 
    + find_best_match, 
    + and normalise_text_for_comparison.

Key Features:

    NormalizedBox:
        Validates bounding box coordinates to ensure they are within [0, 1].

    GroundedPhrase:
        Represents a phrase optionally associated with bounding boxes.
        Includes a from_dict method to create instances from a dictionary.

    EvidencedPhrase:
        Encapsulates a phrase, its evidence, and its logical status.
        Provides a method to convert the status to binary.

    find_best_match:
        Finds the best match for a string in a list of candidates using exact match or fuzzy matching with the longest common substring.

    normalise_text_for_comparison:
        Normalizes text by removing extra spaces, punctuation, and converting to lowercase.


Example Usage:

    # Create a GroundedPhrase
    phrase = GroundedPhrase("example phrase", [NormalizedBox(0.1, 0.2, 0.3, 0.4)])# Example GroundedPhrase
    phrase_data = {"text": "The heart is normal.", "boxes": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}]}
    phrase = GroundedPhrase.from_dict(phrase_data)
    print(phrase)

    # Example EvidencedPhrase
    evidence_phrase = EvidencedPhrase(phrase="The heart is normal.", evidence=["Normal heart size observed"], status="entailed")
    print(evidence_phrase.convert_to_binary())

    # Example Matching
    text = "Heart size is normal."
    candidates = ["Normal heart size", "Enlarged heart", "Heart size is normal."]
    best_index, best_match = find_best_match(text, candidates)
    print(f"Best Match: {best_match} at index {best_index}")

"""

from dataclasses import dataclass
from enum import Enum

from typing import List, Optional, Dict, Any
import re
from difflib import SequenceMatcher


@dataclass
class OneWayNLIFractions:
    """Represents metrics for one-way entailment."""
    entailed_fraction: float
    full_box_fraction: float
    entailed_box_fraction: float
    num_phrases: int
    num_phrases_with_boxes: int

@dataclass
class RadFactScore:
    logical_precision: float
    logical_recall: float
    spatial_precision: float
    spatial_recall: float
    grounding_precision: float
    grounding_recall: float
    num_candidate_phrases: int | float
    num_reference_phrases: int | float
    num_candidate_phrases_with_boxes: int | float
    num_reference_phrases_with_boxes: int | float

    @staticmethod
    def _compute_f1_score(precision: float, recall: float) -> float:
        return 2 * (precision * recall) / (precision + recall)

    @property
    def logical_f1(self) -> float:
        return self._compute_f1_score(self.logical_precision, self.logical_recall)

    @property
    def spatial_f1(self) -> float:
        return self._compute_f1_score(self.spatial_precision, self.spatial_recall)

    @property
    def grounding_f1(self) -> float:
        return self._compute_f1_score(self.grounding_precision, self.grounding_recall)

    @classmethod
    def from_candidate_and_reference_fractions(
        cls, candidate: OneWayNLIFractions, reference: OneWayNLIFractions
    ) -> "RadFactScore":
        """Create a score from the candidate and reference fractions."""
        return cls(
            logical_precision=candidate.entailed_fraction,
            logical_recall=reference.entailed_fraction,
            spatial_precision=candidate.full_box_fraction,
            spatial_recall=reference.full_box_fraction,
            grounding_precision=candidate.entailed_box_fraction,
            grounding_recall=reference.entailed_box_fraction,
            num_candidate_phrases=candidate.num_phrases,
            num_reference_phrases=reference.num_phrases,
            num_candidate_phrases_with_boxes=candidate.num_phrases_with_boxes,
            num_reference_phrases_with_boxes=reference.num_phrases_with_boxes,
        )

    @classmethod
    def from_aggregate(cls, scores: list["RadFactScore"], only_factual_scores: bool = False) -> "RadFactScore":
        """Aggregate the scores from a list of samples. If only_factual_scores is True, we only aggregate the logical
        scores. The spatial and grounding scores are set to 0.0.
        """

        def _nanmean(values: list[float | int]) -> float:
            """
            Compute the mean of the values, ignoring NaNs.
            This is mostly for mypy convenience.
            """
            return float(np.nanmean(values))

        n = len(scores)
        if n == 0:
            return cls(
                logical_precision=0.0,
                logical_recall=0.0,
                spatial_precision=0.0,
                spatial_recall=0.0,
                grounding_precision=0.0,
                grounding_recall=0.0,
                num_candidate_phrases=0.0,
                num_reference_phrases=0.0,
                num_candidate_phrases_with_boxes=0.0,
                num_reference_phrases_with_boxes=0.0,
            )
        return cls(
            # If no predicted or reference phrases, these can be NaN
            logical_precision=_nanmean([x.logical_precision for x in scores]),
            logical_recall=_nanmean([x.logical_recall for x in scores]),
            # Box metrics can be NaN if there are no boxes, either direction
            spatial_precision=0.0 if only_factual_scores else _nanmean([x.spatial_precision for x in scores]),
            spatial_recall=0.0 if only_factual_scores else _nanmean([x.spatial_recall for x in scores]),
            grounding_precision=0.0 if only_factual_scores else _nanmean([x.grounding_precision for x in scores]),
            grounding_recall=0.0 if only_factual_scores else _nanmean([x.grounding_recall for x in scores]),
            # Numbers of phrases etc. should never have NaN
            num_candidate_phrases=sum(x.num_candidate_phrases for x in scores) / n,
            num_reference_phrases=sum(x.num_reference_phrases for x in scores) / n,
            # These can be nan if we are running the metric on data without boxes so we set it to 0.0 when
            # only_factual_scores is True
            num_candidate_phrases_with_boxes=(
                0.0 if only_factual_scores else _nanmean([x.num_candidate_phrases_with_boxes for x in scores])
            ),
            num_reference_phrases_with_boxes=(
                0.0 if only_factual_scores else _nanmean([x.num_reference_phrases_with_boxes for x in scores])
            ),
        )

class SpatialEntailmentStatus(str, Enum):
    NO_BOXES = "no_boxes"
    SPATIAL_ENTAILMENT = "spatial_entailment"
    NO_SPATIAL_ENTAILMENT = "no_spatial_entailment"


@dataclass
class NormalizedBox:
    """Bounding box normalized to the image size, with coordinates in the range [0, 1]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self):
        if not (0 <= self.x_min <= 1 and 0 <= self.y_min <= 1 and 0 <= self.x_max <= 1 and 0 <= self.y_max <= 1):
            raise ValueError(f"Box coordinates must be in range [0, 1]: {self}")


@dataclass
class GroundedPhrase:
    """A grounded phrase consists of a string with an optional list of normalized bounding boxes."""
    text: str
    boxes: Optional[List[NormalizedBox]] = None

    def __post_init__(self):
        if self.boxes is not None and len(self.boxes) == 0:
            raise ValueError(f"Empty boxes for grounded text: {self}, this should be set to None")

    @classmethod
    def from_dict(cls, grounded_phrase_dict: Dict[str, Any]) -> "GroundedPhrase":
        text = grounded_phrase_dict["text"]
        if not isinstance(text, str):
            raise ValueError(f"text is not a string: {text}")
        box_list = grounded_phrase_dict.get("boxes")
        if box_list is None:
            return cls(text=text, boxes=None)
        if isinstance(box_list, list):
            return cls(text=text, boxes=[NormalizedBox(**box) for box in box_list])
        else:
            raise ValueError(f"boxes is not a list: {box_list}")

@dataclass
class EvidencedPhrase:
    """A phrase with its associated evidence and logical status."""
    phrase: str
    evidence: List[str]
    status: str

    def convert_to_binary(self) -> "EvidencedPhrase":
        """Convert the status to binary by mapping it to a simple 'entailed' or 'not-entailed'."""
        if self.status.lower() in {"entailed", "entailment"}:
            return self
        return EvidencedPhrase(phrase=self.phrase, evidence=self.evidence, status="not-entailed")


@dataclass(kw_only=True)
class GroundedPhraseEvidenced(GroundedPhrase):
    status: str
    evidence: List[GroundedPhrase]
    spatial_entailment_status: SpatialEntailmentStatus | None = None
    evidence_indices: List[int] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.evidence_indices is not None:
            assert len(self.evidence_indices) == len(self.evidence)

    def get_all_evidence_boxes(self) -> List[NormalizedBox]:
        all_evidence_boxes = []
        for premise in self.evidence:
            if premise.boxes is not None:
                all_evidence_boxes.extend(premise.boxes)
        return all_evidence_boxes

def find_best_match(text: str, candidate_texts: List[str]) -> (int, str):
    """Find the best matching text from a list of candidate texts."""
    candidates_normalized = [normalise_text_for_comparison(candidate) for candidate in candidate_texts]
    text_normalized = normalise_text_for_comparison(text)

    for i, candidate in enumerate(candidates_normalized):
        if candidate == text_normalized:
            return i, candidate_texts[i]

    best_match = ""
    best_index = -1
    best_length = 0
    for i, candidate in enumerate(candidates_normalized):
        match = SequenceMatcher(None, text_normalized, candidate).find_longest_match(0, len(text_normalized), 0, len(candidate))
        substring = text[match.a:match.a + match.size]
        if len(substring) > best_length:
            best_match = candidate_texts[i]
            best_index = i
            best_length = len(substring)

    if best_index == -1:
        raise ValueError(f"No match found for text: {text}")
    return best_index, best_match


def normalise_text_for_comparison(text: str) -> str:
    """Normalize a string for text comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces
    text = re.sub(r"[\-–—]", "", text)  # Remove dashes
    text = re.sub(r"[\.\,\:\;\!\?\']", "", text)  # Remove punctuation
    return text


# # Example GroundedPhrase
# phrase_data = {"text": "The heart is normal.", "boxes": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}]}
# phrase = GroundedPhrase.from_dict(phrase_data)
# print(phrase)

# # Example EvidencedPhrase
# evidence_phrase = EvidencedPhrase(phrase="The heart is normal.", evidence=["Normal heart size observed"], status="entailed")
# print(evidence_phrase.convert_to_binary())

# # Example Matching
# text = "Heart size is normal."
# candidates = ["Normal heart size", "Enlarged heart", "Heart size is normal."]
# best_index, best_match = find_best_match(text, candidates)
# print(f"Best Match: {best_match} at index {best_index}")