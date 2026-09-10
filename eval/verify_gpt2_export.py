"""Check the hash-pinned local ONNX export against a pinned GPT-2 checkpoint.

Optional provenance check, separate from timed evaluation. Requires torch,
transformers and the cached checkpoint; downloads are not performed here.
"""
import argparse
import importlib.metadata
from pathlib import Path

import numpy as np

from gpt2_evaluation import MODEL_REVISION, RECORDED_MODEL_SHA256, session, sha256, write_json

CHECKPOINT_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
CONFIG_SHA256 = "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("eval/models/gpt2.onnx"))
    parser.add_argument("--model-sha256", default=RECORDED_MODEL_SHA256)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path.home() / ".cache/huggingface/hub/models--gpt2/snapshots" / MODEL_REVISION)
    parser.add_argument("--tokens", type=Path, default=Path("target/gpt2-evaluation/tokens.npy"))
    parser.add_argument("--output", type=Path, default=Path("target/gpt2-evaluation/reference.json"))
    args = parser.parse_args()
    if sha256(args.model) != args.model_sha256 or sha256(args.checkpoint_dir / "model.safetensors") != CHECKPOINT_SHA256:
        raise ValueError("ONNX or reference checkpoint checksum mismatch")
    if sha256(args.checkpoint_dir / "config.json") != CONFIG_SHA256:
        raise ValueError("Reference configuration checksum mismatch")
    import torch
    from transformers import GPT2LMHeadModel
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = GPT2LMHeadModel.from_pretrained(str(args.checkpoint_dir), local_files_only=True, attn_implementation="eager").eval()
    runtime, input_name = session(args.model, 1, optimize=False)
    ids = np.load(args.tokens, allow_pickle=False)
    if len(ids) < 256:
        raise ValueError("Reference validation needs at least 256 tokens")
    errors = {}
    for length in (8, 31, 256):
        tokens = ids[:length][None]
        with torch.no_grad():
            expected = reference(torch.from_numpy(tokens.copy()), use_cache=False).logits.numpy()
        actual = runtime.run(None, {input_name: tokens})[0]
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError("Reference outputs must be finite")
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-4)
        errors[str(length)] = float(np.max(np.abs(actual - expected)))
        print(f"[reference] {length} tokens: max absolute logit error {errors[str(length)]:.8f}", flush=True)
    write_json(args.output, dict(passed=True, checkpoint="openai-community/gpt2", revision=MODEL_REVISION,
                                checkpoint_sha256=CHECKPOINT_SHA256, config_sha256=sha256(args.checkpoint_dir / "config.json"),
                                model_sha256=args.model_sha256, tokens_sha256=sha256(args.tokens),
                                max_abs_logit_error_by_sequence_length=errors, atol=1e-3, rtol=1e-4,
                                packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "onnxruntime")},
                                scope="three fixed test-prefix probes; not an exhaustive export equivalence proof"))


if __name__ == "__main__":
    main()
