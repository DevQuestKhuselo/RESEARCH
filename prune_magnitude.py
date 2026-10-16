"""
Unstructured magnitude pruning of LLaMAX3-8B-Alpaca at a target sparsity
(proposal: 30% and 50%), saved as a normal fp16 checkpoint so the existing
translation / evaluation pipeline runs on it unchanged.

What is pruned
--------------
The weight matrices of every nn.Linear inside the transformer blocks
(q/k/v/o attention projections and gate/up/down MLP projections).
Token embeddings and the LM head are NOT pruned, the standard choice in LLM
pruning work: they are lookup/readout tables whose rows map to individual
tokens, so zeroing entries removes specific tokens' representations rather
than redundant capacity. Biases and norms are untouched (Llama has no
linear biases).

Scope
-----
--scope global (default, as in the proposal): ONE magnitude threshold over all
    prunable weights. Layers with smaller-magnitude weights lose more than
    50% at 50% sparsity overall; the per-module sparsity actually achieved is
    recorded in the metadata so this can be reported.
--scope layer: each weight matrix pruned to exactly the target sparsity.
    Useful as a sensitivity check if global pruning collapses the model.

The threshold is found by bisection on |w| (a few dozen fast counting
passes), which avoids sorting ~7B values. Ties at the threshold can push the
achieved sparsity marginally above target; the exact achieved value is saved.

Efficiency caveat for the write-up
----------------------------------
The checkpoint stays dense fp16 (zeros are stored), so file size, GPU memory
and latency do NOT shrink without sparse kernels or a sparse storage format.
Report pruning's efficiency in terms of non-zero parameters; the measured
size/latency should be ~unchanged from the baseline.

Requirements (research env):  torch, transformers, accelerate
Usage:
    python prune_magnitude.py --sparsity 0.30 --out_dir models/llamax3-8b-mag30
    python prune_magnitude.py --sparsity 0.50 --out_dir models/llamax3-8b-mag50
"""

import argparse
import json
import os
import time
from collections import defaultdict

MODEL_ID = "LLaMAX/LLaMAX3-8B-Alpaca"


def prunable_linears(model):
    """(name, module) for every nn.Linear inside the transformer blocks."""
    import torch.nn as nn
    lm_head = model.get_output_embeddings()
    out = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module is not lm_head and ".layers." in f".{name}":
            out.append((name, module))
    if not out:
        raise RuntimeError("No prunable linear layers found -- unexpected model structure")
    return out


def count_le(weights, t):
    return sum(int((w.abs() <= t).sum()) for w in weights)


def find_threshold(weights, sparsity, iters=64):
    """Smallest t such that at least `sparsity` of the weights have |w| <= t."""
    total = sum(w.numel() for w in weights)
    target = int(round(sparsity * total))
    if target <= 0:
        return -1.0
    lo = 0.0
    hi = max(float(w.abs().max()) for w in weights)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if count_le(weights, mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * max(hi, 1e-30):
            break
    return hi


def module_type(name):
    return name.rsplit(".", 1)[-1]  # e.g. q_proj, down_proj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sparsity", type=float, required=True, help="Fraction of weights to zero, e.g. 0.3")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scope", choices=["global", "layer"], default="global")
    p.add_argument("--model_id", default=MODEL_ID, help="Override only for testing")
    args = p.parse_args()
    if not 0.0 < args.sparsity < 1.0:
        p.error("--sparsity must be between 0 and 1, e.g. 0.3")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} (fp16) on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16, device_map=device)
    model.eval()

    layers = prunable_linears(model)
    weights = [m.weight.data for _, m in layers]
    n_prunable = sum(w.numel() for w in weights)
    n_total = sum(p_.numel() for p_ in model.parameters())
    print(f"{len(layers)} prunable matrices, {n_prunable / 1e9:.3f}B of {n_total / 1e9:.3f}B params")

    t0 = time.time()
    with torch.no_grad():
        if args.scope == "global":
            t = find_threshold(weights, args.sparsity)
            print(f"Global threshold |w| <= {t:.6g}")
            for w in weights:
                w.mul_(w.abs() > t)
            thresholds = {"global": t}
        else:
            thresholds = {}
            for (name, _), w in zip(layers, weights):
                t = find_threshold([w], args.sparsity)
                w.mul_(w.abs() > t)
                thresholds[name] = t
    prune_seconds = time.time() - t0

    # Measure what was actually achieved
    zeros_by_type, numel_by_type = defaultdict(int), defaultdict(int)
    per_layer = {}
    for name, w in zip((n for n, _ in layers), weights):
        z = int((w == 0).sum())
        zeros_by_type[module_type(name)] += z
        numel_by_type[module_type(name)] += w.numel()
        per_layer[name] = round(z / w.numel(), 5)
    zeros = sum(zeros_by_type.values())
    achieved = zeros / n_prunable
    print(f"Achieved sparsity on prunable weights: {achieved:.4%} "
          f"({zeros / n_total:.4%} of all params) in {prune_seconds:.1f}s")
    for k in sorted(numel_by_type):
        print(f"  {k:<10} {zeros_by_type[k] / numel_by_type[k]:.2%}")

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    meta = {
        "base_model": args.model_id,
        "method": "unstructured magnitude pruning",
        "scope": args.scope,
        "target_sparsity": args.sparsity,
        "achieved_sparsity_prunable": round(achieved, 6),
        "achieved_sparsity_all_params": round(zeros / n_total, 6),
        "prunable_params": n_prunable,
        "total_params": n_total,
        "nonzero_params_after": n_total - zeros,
        "excluded": "token embeddings, LM head, norms",
        "sparsity_by_module_type": {k: round(zeros_by_type[k] / numel_by_type[k], 5) for k in sorted(numel_by_type)},
        "sparsity_by_layer": per_layer,
        "threshold": thresholds if args.scope == "global" else "per-matrix (see sparsity_by_layer)",
        "prune_seconds": round(prune_seconds, 1),
        "dtype": "float16 (dense storage; zeros stored explicitly)",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch_version": torch.__version__,
    }
    with open(os.path.join(args.out_dir, "pruning_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved pruned model to {args.out_dir}")


if __name__ == "__main__":
    main()
