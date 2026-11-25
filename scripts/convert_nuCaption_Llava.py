#!/usr/bin/env python3
"""Convert NuCaption QA annotations to sample-grouped LLaVA conversations."""

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any


BEV_FEATURE_ROOT = Path("/home/renault/repo/dataset/nuscenes")
VIEW_PREFIX = re.compile(
    r"^This is the car's (?P<view>front left|front right|back left|back right|front|back) view\.\s*",
    re.IGNORECASE,
)


def load_annotations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    if not isinstance(annotations, list):
        raise ValueError(f"{path}: expected a JSON list of annotations")
    return annotations


def rewrite_question(question: str, input_path: Path, index: int) -> str:
    question = question.strip()
    match = VIEW_PREFIX.match(question)
    if match is None:
        raise ValueError(
            f"{input_path}: annotation {index} question does not start with "
            '"This is the car\'s <view> view."'
        )

    view = match.group("view").lower()
    remaining_question = question[match.end():]
    return (
        "The point-cloud features cover the full scene. "
        f"Answer only about the {view} region. {remaining_question}"
    ).strip()


def convert_split(
    input_path: Path,
    split: str,
    output_path: Path,
    version: str,
) -> tuple[int, int]:
    if version not in {"plain", "qwen2"}:
        raise ValueError(f"Unsupported version: {version}")

    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    converted: list[dict[str, Any]] = []
    annotations = load_annotations(input_path)
    feature_dir = BEV_FEATURE_ROOT / f"unilion_bev_feats_{split}"

    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise ValueError(f"{input_path}: annotation {index} is not a JSON object")

        token = annotation.get("sample_token")
        question = annotation.get("question")
        answer = annotation.get("answer_lidar")
        annotation_split = annotation.get("split")

        if not isinstance(token, str) or not token:
            raise ValueError(f"{input_path}: annotation {index} has no valid sample_token")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{input_path}: annotation {index} has no valid question")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{input_path}: annotation {index} has no valid answer_lidar")
        if annotation_split is not None and annotation_split != split:
            raise ValueError(
                f"{input_path}: annotation {index} declares split "
                f"{annotation_split!r}, expected {split!r}"
            )

        rewritten_question = rewrite_question(question, input_path, index)
        human_value = f"<image>\n{rewritten_question}"
        if version == "plain":
            converted.append(
                {
                    "image": [str(feature_dir / f"{token}.pt")],
                    "history": {"token": token},
                    "conversations": [
                        {"from": "human", "value": human_value},
                        {"from": "gpt", "value": answer.strip()},
                    ],
                }
            )
        else:
            conversations = grouped.setdefault(token, [])
            if not conversations:
                conversations.append({"from": "human", "value": human_value})
            else:
                conversations.append({"from": "human", "value": rewritten_question})
            conversations.append({"from": "gpt", "value": answer.strip()})

    if version == "qwen2":
        converted = [
            {
                "image": [str(feature_dir / f"{token}.pt")],
                "history": {"token": token},
                "conversations": conversations,
            }
            for token, conversations in grouped.items()
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(converted, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return len(annotations), len(converted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Group NuCaption QA entries by sample_token and create LLaVA-style "
            "train and validation JSON files."
        )
    )
    parser.add_argument("train_json", type=Path, help="Path to the NuCaption train.json")
    parser.add_argument("val_json", type=Path, help="Path to the NuCaption val.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for train_llava.json and val_llava.json (default: current directory)",
    )
    parser.add_argument(
        "--version",
        choices=["plain", "qwen2"],
        required=True,
        help="plain creates one QA example per record; qwen2 groups QA turns by sample token",
    )
    args = parser.parse_args()

    for split, input_path in (("train", args.train_json), ("val", args.val_json)):
        output_path = args.output_dir / f"{split}_{args.version}_llava.json"
        annotation_count, sample_count = convert_split(
            input_path, split, output_path, args.version
        )
        print(
            f"{split}: converted {annotation_count} QA annotations into "
            f"{sample_count} samples -> {output_path}"
        )


if __name__ == "__main__":
    main()