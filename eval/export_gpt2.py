"""Optional pinned GPT-2 legacy ONNX export helper used by benchmark_gpt2.py."""
import sys
from pathlib import Path


def export_gpt2(model_dir: Path, download: bool = False) -> Path:
    """Export GPT-2 small from HuggingFace to ONNX."""
    try:
        import torch
        from transformers import GPT2LMHeadModel
    except ImportError:
        sys.exit("Export requires: pip install torch transformers")

    out_path = model_dir / "gpt2.onnx"
    if out_path.exists():
        size_mb = out_path.stat().st_size / 1_048_576
        print(f"[export] {out_path} already exists ({size_mb:.1f} MB) — skipping.")
        return out_path

    model_dir.mkdir(parents=True, exist_ok=True)
    print("[export] Downloading GPT-2 small from HuggingFace ...")
    model = GPT2LMHeadModel.from_pretrained("gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        attn_implementation="eager", local_files_only=not download,
        **({"cache_dir": str(model_dir / "hf-cache")} if download else {}))
    model.eval()

    # Wrap to disable KV cache and return a plain logits tensor.
    # This keeps the ONNX graph simple and avoids DynamicCache in the output.
    class _GPT2Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids):
            return self.inner(input_ids, use_cache=False).logits

    wrapper = _GPT2Wrapper(model)
    dummy   = torch.ones(1, 8, dtype=torch.long)

    # -----------------------------------------------------------------------
    # Two ONNX compatibility patches for transformers 5.x + opset 14:
    #
    # PATCH 1 — torch.diff → ONNX-safe implementation
    #   masking_utils.py:731: torch.diff(position_ids, prepend=..., dim=-1)
    #   aten::diff has no ONNX symbolic.  We temporarily replace torch.diff
    #   with a version using torch.cat + slice + sub (all have symbolics).
    #   torch.onnx.export internally re-runs the model via _get_trace_graph;
    #   that re-run calls our patched function and records supported ops.
    #
    # PATCH 2 — aten::cumsum ONNX symbolic handles bool input
    #   masking_utils.py:732: (position_diff != 1).cumsum(-1)
    #   position_diff != 1 produces a bool tensor.  PyTorch auto-promotes
    #   bool→int64 for cumsum, but the ONNX symbolic doesn't insert a Cast,
    #   so OnnxRuntime rejects the graph ("Type 'tensor(bool)' ... invalid").
    #   We register a custom symbolic that adds Cast(bool→INT64) when needed.
    # -----------------------------------------------------------------------
    import warnings
    from torch.onnx import register_custom_op_symbolic, symbolic_helper

    # -- Patch 2 (global, permanent — safe to register multiple times) ------
    def _cumsum_cast(g, input, dim, dtype):
        if input.type().scalarType() == 'Bool':
            input = g.op("Cast", input, to_i=7)          # TensorProto.INT64
        elif not symbolic_helper._is_none(dtype):
            dv = symbolic_helper._get_const(dtype, "i", "dtype")
            if dv is not None:
                input = g.op("Cast", input, to_i=dv)
        dim_val = symbolic_helper._get_const(dim, "i", "dim")
        axis = g.op("Constant",
                    value_t=torch.tensor(dim_val if dim_val is not None else 0,
                                         dtype=torch.int64))
        return g.op("CumSum", input, axis)

    register_custom_op_symbolic("aten::cumsum", _cumsum_cast, 11)
    register_custom_op_symbolic("aten::cumsum", _cumsum_cast, 14)

    # -- Patch 1 (scoped to export call) ------------------------------------
    _orig_diff = torch.diff

    def _onnx_safe_diff(input, n=1, dim=-1, prepend=None, append=None):
        result = input
        parts = []
        if prepend is not None:
            parts.append(prepend)
        parts.append(result)
        if append is not None:
            parts.append(append)
        if len(parts) > 1:
            result = torch.cat(parts, dim=dim)
        for _ in range(n):
            ndim = result.dim()
            d = dim if dim >= 0 else ndim + dim
            s_tail = [slice(None)] * ndim
            s_head = [slice(None)] * ndim
            s_tail[d] = slice(1, None)
            s_head[d] = slice(None, -1)
            result = result[tuple(s_tail)] - result[tuple(s_head)]
        return result

    torch.diff = _onnx_safe_diff
    print(f"[export] Exporting to {out_path} ...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapper,
                (dummy,),
                str(out_path),
                opset_version=14,
                dynamo=False,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "seq_len"},
                    "logits":    {0: "batch_size", 1: "seq_len"},
                },
                do_constant_folding=True,
                verbose=False,
            )
    finally:
        torch.diff = _orig_diff  # always restore

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"[export] Done — {size_mb:.1f} MB written to {out_path}")
    return out_path


