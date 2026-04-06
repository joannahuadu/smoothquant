from __future__ import annotations

import random
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets, load_dataset


def _load_c4_from_cache(split: str):
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "allenai___c4"
    if split == "train":
        base = cache_root / "default-b04fc8a0b8562884" / "0.0.0" / "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
        arrow_files = sorted(base.glob("c4-train-*.arrow"))
    elif split == "validation":
        base = cache_root / "default-c7bc8b0aefc5e48f" / "0.0.0" / "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
        arrow_files = sorted(base.glob("c4-validation*.arrow"))
    else:
        raise ValueError(f"unsupported split: {split}")
    if not arrow_files:
        raise FileNotFoundError(f"no cached C4 arrow files found for split={split}")
    datasets = [Dataset.from_file(str(path)) for path in arrow_files]
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def get_calibration_batches(dataset_name, tokenizer, nsamples, seed, seqlen):
    dataset_name = str(dataset_name).lower()
    if dataset_name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        rng = random.Random(seed)
        batches = []
        for _ in range(nsamples):
            i = rng.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            batches.append((inp, tar))
        return batches

    if dataset_name == "c4":
        try:
            traindata = load_dataset(
                "allenai/c4",
                data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                split="train",
            )
        except Exception:
            traindata = _load_c4_from_cache("train")
        rng = random.Random(seed)
        batches = []
        for _ in range(nsamples):
            while True:
                i = rng.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            start = rng.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            end = start + seqlen
            inp = trainenc.input_ids[:, start:end]
            tar = inp.clone()
            tar[:, :-1] = -100
            batches.append((inp, tar))
        return batches

    raise ValueError(f"unsupported dataset: {dataset_name}")
