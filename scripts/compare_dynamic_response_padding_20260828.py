#!/usr/bin/env python3
"""Compare fixed-cap and batch-dynamic response padding on one replay pack.

The replay pack contains the dynamic-width representation produced by the
runtime.  The fixed representation is reconstructed by right-padding those
same tensors to the diagnostic hard cap.  Only padding is changed; all
non-padding response content, masks, scores, rewards, and group identifiers
must remain identical.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


WIDTH_FIELDS = {
    "responses",
    "response_mask",
    "token_level_scores",
    "token_level_rewards",
    "advantages",
    "returns",
    "old_log_probs",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", type=Path)
    parser.add_argument("--hard-cap", type=int, default=512)
    args = parser.parse_args()
    if args.hard_cap <= 0:
        raise SystemExit("--hard-cap must be positive")

    pack = torch.load(args.pack, map_location="cpu", weights_only=False)
    tensors = pack["tensor_fields"]
    dynamic = tensors["responses"]
    batch_size, dynamic_width = dynamic.shape
    if dynamic_width > args.hard_cap:
        raise SystemExit("dynamic response width exceeds hard cap")

    # Reconstruct the fixed-cap tensors from exactly the captured dynamic data.
    fixed = {}
    for name in WIDTH_FIELDS:
        value = tensors.get(name)
        if value is None or value.ndim != 2 or value.shape[1] != dynamic_width:
            continue
        fixed[name] = F.pad(value, (0, args.hard_cap - dynamic_width))

    mask = tensors["response_mask"]
    nonpad = mask.bool()
    checks = {}
    for name, value in fixed.items():
        original = tensors[name]
        checks[name] = bool(torch.equal(value[:, :dynamic_width], original))
    checks["nonpad_response_ids"] = bool(
        torch.equal(dynamic[nonpad], fixed["responses"][:, :dynamic_width][nonpad])
    )
    checks["response_mask"] = bool(
        torch.equal(mask, fixed["response_mask"][:, :dynamic_width])
    )
    for name in ("token_level_rewards", "advantages", "returns", "old_log_probs"):
        if name in tensors:
            checks[f"nonpad_{name}"] = bool(
                torch.equal(
                    tensors[name][nonpad],
                    fixed[name][:, :dynamic_width][nonpad],
                )
            )

    if not all(checks.values()):
        raise SystemExit(json.dumps({"status": "failed", "checks": checks}))

    dynamic_elements = batch_size * dynamic_width
    fixed_elements = batch_size * args.hard_cap
    saved_elements = fixed_elements - dynamic_elements
    aligned_fields = {}
    saved_bytes = 0
    for name, value in fixed.items():
        field_saved = saved_elements * value.element_size()
        aligned_fields[name] = {
            "dtype": str(value.dtype),
            "element_size_bytes": value.element_size(),
            "saved_bytes": field_saved,
        }
        saved_bytes += field_saved

    result = {
        "status": "ok",
        "pack": str(args.pack),
        "batch_size": batch_size,
        "hard_cap": args.hard_cap,
        "dynamic_width": dynamic_width,
        "fixed_response_elements": fixed_elements,
        "dynamic_response_elements": dynamic_elements,
        "padding_elements_saved": saved_elements,
        "response_tensor_savings_ratio": saved_elements / fixed_elements,
        "aligned_field_saved_bytes": aligned_fields,
        "estimated_aligned_tensor_bytes_saved": saved_bytes,
        "checks": checks,
        "uid_unchanged": pack["non_tensor_batch"].get("uid") is not None,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
