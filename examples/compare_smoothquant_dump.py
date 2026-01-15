import argparse
import json
from typing import Dict, List

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
import tqdm

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--act_scales_path", type=str, required=True)
    parser.add_argument("--layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31")
    parser.add_argument("--module_name", type=str, default="self_attn.q_proj")
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="0-99",
        help="Comma-separated indices or ranges (e.g. 0,5,10-20).",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--act_quant", type=str, default="per_token")
    parser.add_argument("--quantize_bmm_input", action="store_true")
    parser.add_argument("--output", type=str, default="smoothquant_compare.pt")
    return parser.parse_args()


def build_targets(layers: List[int], module_name: str) -> List[str]:
    targets = []
    for layer in layers:
        targets.append(f"model.layers.{layer}.{module_name}")
    return targets

def get_sample(tokenizer, sample_idx: int, max_length: int) -> Dict[str, torch.Tensor]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = dataset[sample_idx]["text"]
    tokens = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )
    return {"text": text, "input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}


def capture_inputs(model, target_names: List[str], input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    captured: Dict[str, torch.Tensor] = {}
    hooks = []

    def hook_fn(name):
        def _inner(_, inputs, __):
            if name not in captured:
                captured[name] = inputs[0].detach().cpu()
        return _inner

    for name, module in model.named_modules():
        if name in target_names:
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        model(input_ids.to(model.device))

    for h in hooks:
        h.remove()
    return captured


def extract_weights(model, target_names: List[str]) -> Dict[str, torch.Tensor]:
    weights = {}
    for name, module in model.named_modules():
        if name in target_names and hasattr(module, "weight"):
            weights[name] = module.weight.detach().cpu()
    return weights


def capture_quant_inputs(model, target_names: List[str], input_ids: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
    captured: Dict[str, Dict[str, torch.Tensor]] = {}
    hooks = []

    def hook_fn(name, module):
        def _inner(_, inputs, __):
            x = inputs[0].detach().cpu()
            qx = module.act_quant(inputs[0].detach().clone()).cpu()
            captured[name] = {"input": x, "input_quant": qx}
        return _inner

    for name, module in model.named_modules():
        if name in target_names:
            hooks.append(module.register_forward_hook(hook_fn(name, module)))

    with torch.no_grad():
        model(input_ids.to(model.device))

    for h in hooks:
        h.remove()
    return captured


def parse_int_list(value: str) -> List[int]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    result: List[int] = []
    for item in items:
        if "-" in item:
            start_str, end_str = item.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid range: {item}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(item))
    return result


def tensor_max(tensor: torch.Tensor) -> float:
    return tensor.abs().detach().float().max().item()


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    module_name = args.module_name
    target_names = build_targets(layers, module_name)
    sample_indices = parse_int_list(args.sample_indices)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator = Evaluator(dataset, tokenizer, "cuda")

    # Original fp16 model
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto"
    )
    # fp16_inputs = capture_inputs(model_fp16, target_names, sample["input_ids"])
    fp16_weights = extract_weights(model_fp16, target_names)

    # Smooth model
    model_smooth = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto"
    )
    act_scales = torch.load(args.act_scales_path)
    smooth_lm(model_smooth, act_scales, args.alpha)
    smooth_weights = extract_weights(model_smooth, target_names)

    # Quantized model
    model_quant = quantize_model(
        model_smooth,
        weight_quant="per_channel",
        act_quant=args.act_quant,
        quantize_bmm_input=args.quantize_bmm_input,
    )
    quant_weights = extract_weights(model_quant, target_names)
    fp16_inputs = {}
    quant_inputs = {}
    texts = {}
    max_stats = {
        "fp16_input": {name: {} for name in target_names},
        "quant_input_fp": {name: {} for name in target_names},
        "quant_input_q": {name: {} for name in target_names},
        "fp16_weight": {name: {} for name in target_names},
        "smooth_weight": {name: {} for name in target_names},
        "quant_weight": {name: {} for name in target_names},
    }

    for sample_idx in sample_indices:
        sample = get_sample(tokenizer, sample_idx, args.max_length)
        texts[sample_idx] = sample["text"]
        fp16_inputs[sample_idx] = capture_inputs(
            model_fp16, target_names, sample["input_ids"]
        )
        quant_inputs[sample_idx] = capture_quant_inputs(
            model_quant, target_names, sample["input_ids"]
        )

        for name in target_names:
            max_stats["fp16_input"][name][sample_idx] = tensor_max(
                fp16_inputs[sample_idx][name]
            )
            max_stats["quant_input_fp"][name][sample_idx] = tensor_max(
                quant_inputs[sample_idx][name]["input"]
            )
            max_stats["quant_input_q"][name][sample_idx] = tensor_max(
                quant_inputs[sample_idx][name]["input_quant"]
            )

    for name in target_names:
        max_stats["fp16_weight"][name] = tensor_max(fp16_weights[name])
        max_stats["smooth_weight"][name] = tensor_max(smooth_weights[name])
        max_stats["quant_weight"][name] = tensor_max(quant_weights[name])
    ppl = evaluator.evaluate(model_quant)
    print(f"Perplexity: {ppl}")
    output = {
        "meta": {
            "model_path": args.model_path,
            "act_scales_path": args.act_scales_path,
            "alpha": args.alpha,
            "act_quant": args.act_quant,
            "quantize_bmm_input": args.quantize_bmm_input,
            "layers": layers,
            "sample_indices": sample_indices,
        },
        "fp16_inputs": fp16_inputs,
        "fp16_weights": fp16_weights,
        "smooth_weights": smooth_weights,
        "quant_weights": quant_weights,
        "quant_inputs": quant_inputs,
        "texts": texts,
        "max_stats": max_stats,
    }

    torch.save(output, args.output)
    print(json.dumps({"output": args.output, "targets": target_names}, indent=2))


if __name__ == "__main__":
    main()
