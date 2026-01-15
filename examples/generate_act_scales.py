import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

from datasets import load_dataset
from smoothquant.calibration import get_act_scales, get_act_scales_from_dataset


def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="facebook/opt-1.3b", help="model name"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="act_scales/opt-1.3b.pt",
        help="where to save the act scales",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/val.jsonl.zst",
        help="location of the calibration dataset, we use the validation set of the Pile dataset",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="",
        help="optional Hugging Face dataset name (e.g. wikitext)",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default="",
        help="optional Hugging Face dataset config (e.g. wikitext-2-raw-v1)",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="",
        help="optional Hugging Face dataset split (e.g. test)",
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name)

    if args.hf_dataset:
        dataset = load_dataset(
            args.hf_dataset,
            args.hf_config if args.hf_config else None,
            split=args.hf_split if args.hf_split else None,
        )
        act_scales = get_act_scales_from_dataset(
            model, tokenizer, dataset, args.num_samples, args.seq_len
        )
    else:
        if not os.path.exists(args.dataset_path):
            print(f"Cannot find the dataset at {args.dataset_path}")
            print("Please download the Pile dataset and put the validation set at the path")
            print(
                "You can download the validation dataset of the Pile at https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst"
            )
            raise FileNotFoundError

        act_scales = get_act_scales(
            model, tokenizer, args.dataset_path, args.num_samples, args.seq_len
        )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == "__main__":
    main()
