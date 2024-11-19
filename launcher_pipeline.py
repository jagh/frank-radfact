import json
from src.radfact_frank_metrics.simplified_radfact import SimplifiedRadFactMetric
from src.radfact_frank_metrics.schema import GroundedPhraseEvidenced, NormalizedBox


def load_data(file_path: str):
    """Load grounded reporting data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def parse_phrases(phrases: list):
    """Convert a list of phrases with boxes into GroundedPhraseEvidenced objects."""
    return [
        GroundedPhraseEvidenced(
            text=p["text"],
            boxes=[NormalizedBox(**box) for box in p["boxes"]] if p["boxes"] else None,
            status="entailed",  # Assumes all phrases are entailed for this example
            evidence=[]
        )
        for p in phrases
    ]


def main():
    # Load the data
    json_file = "examples/radfact-reports/grounded_reporting_examples.json"  # Replace with your file path
    data = load_data(json_file)

    candidates = {}
    references = {}

    # Parse data
    for example in data:
        example_id = example["example_id"]
        candidates[example_id] = parse_phrases(example["prediction"])
        references[example_id] = parse_phrases(example["target"])

    # Initialize and compute metrics
    radfact_metric = SimplifiedRadFactMetric()
    results = radfact_metric.evaluate(candidates, references)

    # Print results
    for study_id, score in results.items():
        print(f"Study ID: {study_id}")
        print(f"Logical F1: {score.logical_f1}")
        print(f"Spatial F1: {score.spatial_f1}")
        print(f"Grounding F1: {score.grounding_f1}")
        print("-" * 40)


if __name__ == "__main__":
    main()
