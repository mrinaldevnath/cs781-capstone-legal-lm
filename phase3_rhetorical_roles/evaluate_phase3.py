#!/usr/bin/env python3
"""Local evaluator for Phase 3.

The competition scorer evaluates two gold/reference JSON files together and
one participant prediction.json. This local evaluator mirrors that setup
using the student's own held-out splits.

Usage:
  python evaluate_phase3.py \
      --truth-cl cl_test_truth.json \
      --truth-it it_test_truth.json \
      --prediction prediction.json \
      --labels labels.json

Each truth file must be a JSON object mapping document IDs to ordered lists of
sentence labels. prediction.json must have the same structure and contain
predictions for documents from both reference files.

labels.json may be either:
  ["Label1", "Label2", ...]
or a label->id / id->label mapping.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import argparse
import json

from sklearn.metrics import f1_score, precision_score, recall_score

EXPECTED_LABEL_COUNT = 13


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_id(doc_id):
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("Document IDs must be non-empty strings.")
    return doc_id[:-4] if doc_id.endswith(".txt") else doc_id


def load_vocab(data):
    if isinstance(data, list):
        labels = data
    elif isinstance(data, dict) and data:
        values = list(data.values())
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            # label -> integer ID
            labels = list(data.keys())
        elif all(isinstance(v, str) for v in values):
            # integer/string ID -> label
            labels = values
        else:
            raise ValueError("Unsupported labels.json format.")
    else:
        raise ValueError("labels.json must be a non-empty list or object.")

    if len(labels) != EXPECTED_LABEL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LABEL_COUNT} labels, found {len(labels)}."
        )
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Labels must be unique non-empty strings.")
    if len(set(labels)) != len(labels):
        raise ValueError("Labels must be unique non-empty strings.")
    return labels


def normalize_reference(data, source_name):
    if not isinstance(data, dict):
        raise ValueError(f"{source_name}: reference data must be a JSON object.")

    normalized = {}
    for raw_id, labels in data.items():
        doc_id = normalize_id(raw_id)
        if doc_id in normalized:
            raise ValueError(
                f"{source_name}: duplicate document ID after .txt normalization: {doc_id}"
            )
        normalized[doc_id] = labels
    return normalized


def normalize_predictions(data):
    if not isinstance(data, dict):
        raise ValueError("prediction.json must be a JSON object mapping document IDs to label lists.")

    normalized = {}
    for raw_id, labels in data.items():
        doc_id = normalize_id(raw_id)
        if doc_id in normalized:
            raise ValueError(
                f"prediction.json: duplicate document ID after .txt normalization: {doc_id}"
            )
        normalized[doc_id] = labels
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Phase 3 sentence-label predictions on two local held-out "
            "reference splits, mirroring the competition's combined scoring setup."
        )
    )
    parser.add_argument(
        "--truth-cl",
        required=True,
        help="Local held-out reference JSON for the CL split.",
    )
    parser.add_argument(
        "--truth-it",
        required=True,
        help="Local held-out reference JSON for the IT split.",
    )
    parser.add_argument(
        "--prediction",
        required=True,
        help="Participant prediction JSON covering documents from both splits.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Public Phase 3 label vocabulary JSON.",
    )
    args = parser.parse_args()

    truth_cl = normalize_reference(load_json(args.truth_cl), "--truth-cl")
    truth_it = normalize_reference(load_json(args.truth_it), "--truth-it")
    predictions = normalize_predictions(load_json(args.prediction))
    labels = load_vocab(load_json(args.labels))
    allowed = set(labels)

    overlap = set(truth_cl) & set(truth_it)
    if overlap:
        sample = sorted(overlap)[:5]
        suffix = "..." if len(overlap) > 5 else ""
        raise ValueError(
            "The two reference splits contain overlapping document IDs: "
            + ", ".join(sample)
            + suffix
        )

    truth = {}
    truth.update(truth_cl)
    truth.update(truth_it)

    if set(truth) != set(predictions):
        missing = sorted(set(truth) - set(predictions))
        extra = sorted(set(predictions) - set(truth))
        parts = [
            f"missing={len(missing)}",
            f"extra={len(extra)}",
        ]
        if missing:
            parts.append(f"example missing ID(s)={missing[:5]}")
        if extra:
            parts.append(f"example extra ID(s)={extra[:5]}")
        raise ValueError("ID mismatch: " + "; ".join(parts))

    y_true = []
    y_pred = []

    for doc_id in truth:
        gold = truth[doc_id]
        pred = predictions[doc_id]

        if not isinstance(gold, list) or not isinstance(pred, list):
            raise ValueError(f"{doc_id}: labels must be JSON lists.")
        if len(gold) != len(pred):
            raise ValueError(
                f"{doc_id}: sentence-count mismatch ({len(gold)} vs {len(pred)})."
            )
        if any(label not in allowed for label in gold + pred):
            raise ValueError(f"{doc_id}: unknown label encountered.")

        y_true.extend(gold)
        y_pred.extend(pred)

    score = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    print(f"CL documents: {len(truth_cl)}")
    print(f"IT documents: {len(truth_it)}")
    print(f"Total documents: {len(truth)}")
    print(f"Total sentences: {len(y_true)}")
    print(f"Macro-F1:        {score:.6f}")
    print(f"Macro-Precision: {macro_precision:.6f}")
    print(f"Macro-Recall:    {macro_recall:.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"EVALUATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
