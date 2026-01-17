import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
from smoothquant.sparse import apply_activation_sparsity_to_model, remove_activation_sparsity_hooks
import tqdm

from datasets import load_dataset
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument(
    "--act_scales_path",
    type=str,
    default="act_scales/llama-2-7b.pt",
)
parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--quantize", action="store_true")
parser.add_argument(
    "--act_sparsity",
    type=str,
    default="",
    help="Enable activation N:M sparsity, format '2:4'. Empty disables.",
)
parser.add_argument(
    "--w_bits",
    type=int,
    default=8,
    help="Weight quantization bit width (default: 8)",
)
parser.add_argument(
    "--a_bits",
    type=int,
    default=8,
    help="Activation quantization bit width (default: 8)",
)
parser.add_argument(
    "--target_modules",
    type=str,
    default=None,
    help="Target modules for sparsity (e.g., 'q_proj,k_proj,v_proj')",
)
parser.add_argument(
    "--weight_scoring",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable weight scoring for sparsity scaling (default: True)",
)
parser.add_argument(
    "--act_sparsity_location",
    type=str,
    default="pre_quant",
    choices=["pre_quant", "post_quant"],
    help="Where to apply activation sparsity when quantizing (default: pre_quant)",
)


args = parser.parse_args()
alpha = args.alpha
model_path = args.model_path
act_scales_path = args.act_scales_path
n_samples = args.n_samples
act_sparsity = args.act_sparsity
w_bits = args.w_bits
a_bits = args.a_bits
target_modules = args.target_modules.split(",") if args.target_modules else None
weight_scoring = args.weight_scoring
act_sparsity_location = args.act_sparsity_location
print(f"W{w_bits}A{a_bits} Quantization.")
act_sparsity_n = 0
act_sparsity_m = 0
if act_sparsity:
    act_sparsity_n, act_sparsity_m = map(int, act_sparsity.split(":"))
print(f"N:M Sparsity: {act_sparsity_n}: {act_sparsity_m}.")


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


tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=n_samples)

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)

if args.smooth:
    print("smooth...")
    act_scales = torch.load(act_scales_path)
    smooth_lm(model, act_scales, alpha)

sparsity_hooks = None

if args.quantize:
    print("quantize...")
    print(f"target_modules: {target_modules}")
    print(f"N:M: {act_sparsity_n}:{act_sparsity_m}")
    print(f"act_sparsity_location: {act_sparsity_location}")
    model = quantize_model(
        model,
        weight_quant="per_channel",
        w_bits=w_bits,
        a_bits=a_bits,
        act_quant="per_token",
        quantize_bmm_input=True,
        act_sparsity_n=act_sparsity_n,
        act_sparsity_m=act_sparsity_m,
        target_modules=target_modules,
        weight_scoring=weight_scoring,
        act_sparsity_location=act_sparsity_location,
    )
elif act_sparsity_n and act_sparsity_m:
    print(f"\nApplying activation sparsity ({act_sparsity_n}:{act_sparsity_m}) without quantization...")
    if target_modules:
        print(f"Target modules: {target_modules}")
    sparsity_hooks = apply_activation_sparsity_to_model(
        model,
        act_sparsity_n=act_sparsity_n,
        act_sparsity_m=act_sparsity_m,
        target_modules=target_modules,
        weight_scoring=weight_scoring,
    )
    print(f"Registered {sparsity_hooks['num_hooks']} sparsity hooks")

ppl = evaluator.evaluate(model)
print(f"Perplexity: {ppl}")

# Clean up sparsity hooks if they were applied
if sparsity_hooks is not None:
    remove_activation_sparsity_hooks(sparsity_hooks)
