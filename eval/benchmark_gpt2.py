#!/usr/bin/env python3
"""
GPT-2 quantization quality benchmark for quantize-rs.

Three-step workflow
-------------------
Step 1 — Export GPT-2 small from HuggingFace to ONNX:

    python eval/benchmark_gpt2.py --export

Step 2 — Quantize with quantize-rs (INT8 or INT4):

    python eval/benchmark_gpt2.py --quantize --bits 8
    python eval/benchmark_gpt2.py --quantize --bits 4 --min-elements 8192

Step 3 — Compare perplexity + text generation:

    python eval/benchmark_gpt2.py --benchmark --bits 8
    python eval/benchmark_gpt2.py --benchmark --bits 4

All steps in one go:

    python eval/benchmark_gpt2.py --all --bits 8
    python eval/benchmark_gpt2.py --all --bits 4 --min-elements 8192

Requirements
------------
    pip install onnxruntime transformers numpy
    pip install torch                 # only needed for --export
    pip install datasets              # optional: real WikiText-2 perplexity

Run from the quantize-rs project root so the binary path resolves correctly.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Fallback perplexity texts (public-domain, Wikipedia-style)
# Used when the `datasets` package is not installed.
# ---------------------------------------------------------------------------
FALLBACK_TEXTS = [
    "The tower is 324 metres tall, about the same height as an 81-storey building, "
    "and the tallest structure in Paris. Its base is square, measuring 125 metres on each side. "
    "During its construction, the Eiffel Tower surpassed the Washington Monument to become "
    "the tallest man-made structure in the world.",
    "Homarus americanus, the American lobster, is a species of lobster found on the "
    "Atlantic coast of North America. It is also known as the Atlantic lobster or the "
    "Canadian lobster, and is often called the true lobster.",
    "The Python programming language was created by Guido van Rossum and first released "
    "in 1991. It emphasises code readability and uses significant indentation. Python is "
    "dynamically typed and garbage-collected. It supports multiple programming paradigms.",
    "The French Revolution was a period of radical political and societal transformation "
    "in France that began with the Estates General of 1789 and ended with the formation "
    "of the French Consulate in November 1799.",
    "In mathematics, a prime number is a natural number greater than 1 that is not a "
    "product of two smaller natural numbers. A natural number greater than 1 that is not "
    "prime is called a composite number.",
    "The Amazon rainforest covers over five and a half million square kilometres and "
    "represents over half of the planet's remaining rainforests. It comprises the largest "
    "and most biodiverse tract of tropical rainforest in the world.",
    "The speed of light in a vacuum is exactly 299,792,458 metres per second. It is "
    "denoted by the letter c in physics. According to special relativity, c is the upper "
    "limit for the speed at which conventional matter or energy can travel through space.",
    "William Shakespeare was an English playwright, poet, and actor. He is widely regarded "
    "as the greatest writer in the English language and the world's greatest dramatist. "
    "He is often called England's national poet and the Bard of Avon.",
    "The human brain contains approximately 86 billion neurons, each connected to thousands "
    "of others through synapses, forming a complex network. It is protected by the skull "
    "and surrounded by cerebrospinal fluid.",
    "The Great Wall of China is a series of fortifications built across the historical "
    "northern borders of ancient Chinese states and Imperial China to protect against "
    "nomadic invasions from the Eurasian Steppe.",
    "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide "
    "to produce oxygen and energy in the form of glucose. It is one of the most important "
    "biological processes on Earth.",
    "The International Space Station is a modular space station in low Earth orbit. "
    "It is a multinational collaborative project between five space agencies: NASA, Roscosmos, "
    "JAXA, ESA, and CSA. The station serves as a microgravity laboratory.",
    "Alan Turing was a British mathematician and computer scientist who formalised the "
    "concepts of algorithm and computation. He played a crucial role in breaking the "
    "Enigma cipher during World War II.",
    "The Sahara is the world's largest hot desert, covering an area of about 9.2 million "
    "square kilometres across North Africa. It is surpassed in size only by the "
    "Antarctic and Arctic deserts.",
    "Quantum mechanics is a fundamental theory in physics that provides a description "
    "of the physical properties of nature at the scale of atoms and subatomic particles. "
    "It was developed in the early twentieth century.",
    "The Roman Empire at its greatest extent covered approximately five million square "
    "kilometres and had an estimated population of 50 to 90 million people. It was one "
    "of the largest empires in ancient history.",
    "DNA, or deoxyribonucleic acid, is the hereditary material in humans and almost all "
    "other organisms. It carries the genetic information used in the growth, development, "
    "functioning, and reproduction of all known organisms and many viruses.",
    "The Milky Way is the galaxy that includes the Solar System, with an estimated "
    "100 to 400 billion stars and more than 100 billion planets. The Solar System is "
    "located at a radius of about 27,000 light-years from the Galactic Center.",
    "Ludwig van Beethoven was a German composer and pianist. He is one of the most "
    "admired composers in the history of Western music; his works rank among the most "
    "performed of the classical music repertoire.",
    "The Industrial Revolution, which began in Britain around 1760, was the transition "
    "to new manufacturing processes in Europe and the United States. It preceded the "
    "widespread use of steam power and the growth of factory systems.",
]

GENERATION_PROMPTS = [
    "The history of artificial intelligence began",
    "In the field of quantum computing, researchers have discovered",
    "The most important invention of the twentieth century was",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def fmt_bytes(n: int) -> str:
    if n >= 1_073_741_824:
        return f"{n / 1_073_741_824:.2f} GB"
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    return f"{n / 1024:.1f} KB"


def print_wrapped(text: str, indent: str = "    ", width: int = 72) -> None:
    words = text.split()
    line = indent
    for word in words:
        candidate = (line + " " + word) if line != indent else (indent + word)
        if len(candidate) > width:
            print(line)
            line = indent + word
        else:
            line = candidate
    if line.strip():
        print(line)


def binary_path(project_root: Path) -> Path:
    ext = ".exe" if platform.system() == "Windows" else ""
    return project_root / "target" / "release" / "examples" / f"validate_real_model{ext}"


# ---------------------------------------------------------------------------
# Step 1 — Export
# ---------------------------------------------------------------------------

def export_gpt2(model_dir: Path) -> Path:
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
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
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


# ---------------------------------------------------------------------------
# Step 2 — Quantize
# ---------------------------------------------------------------------------

def quantize(
    model_dir: Path,
    project_root: Path,
    bits: int,
    per_channel: bool,
    min_elements: int,
) -> Path:
    """Run validate_real_model to quantize gpt2.onnx."""
    src = model_dir / "gpt2.onnx"
    if not src.exists():
        sys.exit(f"[quantize] {src} not found — run --export first.")

    dst = model_dir / f"gpt2_int{bits}.onnx"
    binary = binary_path(project_root)
    if not binary.exists():
        sys.exit(
            f"[quantize] Binary not found: {binary}\n"
            f"           Run: cargo build --release --example validate_real_model"
        )

    cmd = [
        str(binary),
        str(src),
        "--bits", str(bits),
        "--min-elements", str(min_elements),
        "--output", str(dst),
    ]
    if per_channel:
        cmd.append("--per-channel")

    print(f"\n[quantize] {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    return dst


# ---------------------------------------------------------------------------
# Step 3 — Benchmark
# ---------------------------------------------------------------------------

def load_tokenizer():
    try:
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        return tok
    except ImportError:
        sys.exit("Benchmark requires: pip install transformers")


def load_texts(n: int = 20) -> list[str]:
    """Load WikiText-2 test sentences; fall back to built-in texts."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t.strip() for t in ds["text"] if len(t.strip()) > 120][:n]
        if texts:
            print(f"[benchmark] Loaded {len(texts)} sentences from WikiText-2.")
            return texts
    except Exception:
        pass
    print("[benchmark] Using built-in texts (install `datasets` for WikiText-2).")
    return FALLBACK_TEXTS[:n]


def load_session(path: Path):
    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("Benchmark requires: pip install onnxruntime")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    print(f"[benchmark] Loading {path.name} ...")
    sess = ort.InferenceSession(str(path), sess_options=opts)

    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print(f"[benchmark]   input='{in_name}'  output='{out_name}'")
    return sess, in_name, out_name


def compute_perplexity(
    session,
    in_name: str,
    out_name: str,
    tokenizer,
    texts: list[str],
    max_length: int = 256,
) -> tuple[float, float]:
    """
    Slide over each text, compute negative log-likelihood, return
    (perplexity, avg_ms_per_forward_pass).
    """
    total_nll = 0.0
    total_tokens = 0
    total_ms = 0.0
    n_passes = 0

    for text in texts:
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            continue
        ids = ids[:max_length]
        input_ids = np.array([ids], dtype=np.int64)

        t0 = time.perf_counter()
        logits = session.run([out_name], {in_name: input_ids})[0]  # [1, seq, vocab]
        t1 = time.perf_counter()

        total_ms += (t1 - t0) * 1000
        n_passes += 1

        # logits[i] predicts token[i+1]
        shift_logits = logits[0, :-1]           # [seq-1, vocab]
        shift_labels = np.array(ids[1:])        # [seq-1]

        lp  = log_softmax(shift_logits, axis=-1)
        nll = -lp[np.arange(len(shift_labels)), shift_labels].sum()

        total_nll    += float(nll)
        total_tokens += len(shift_labels)

    ppl    = float(np.exp(total_nll / total_tokens)) if total_tokens > 0 else float("inf")
    avg_ms = total_ms / n_passes if n_passes > 0 else 0.0
    return ppl, avg_ms


def generate_text(
    session,
    in_name: str,
    out_name: str,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
) -> tuple[str, float]:
    """Greedy generation. Returns (full_text, ms_per_token)."""
    ids = tokenizer.encode(prompt)
    input_ids = np.array([ids], dtype=np.int64)

    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        logits    = session.run([out_name], {in_name: input_ids})[0]
        next_tok  = int(np.argmax(logits[0, -1]))
        input_ids = np.concatenate([input_ids, [[next_tok]]], axis=1)
        if next_tok == tokenizer.eos_token_id:
            break
    t1 = time.perf_counter()

    n_new       = input_ids.shape[1] - len(ids)
    ms_per_tok  = (t1 - t0) * 1000 / max(n_new, 1)
    return tokenizer.decode(input_ids[0]), ms_per_tok


def run_benchmark(model_dir: Path, bits: int) -> None:
    orig_path = model_dir / "gpt2.onnx"
    q_path    = model_dir / f"gpt2_int{bits}.onnx"

    for p in [orig_path, q_path]:
        if not p.exists():
            sys.exit(f"[benchmark] {p} not found — run --export and --quantize first.")

    tokenizer = load_tokenizer()
    texts     = load_texts(20)

    sess_orig, in_orig, out_orig = load_session(orig_path)
    sess_q,    in_q,    out_q    = load_session(q_path)

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------
    print(f"\n[benchmark] Computing perplexity on {len(texts)} texts "
          f"(this takes ~1 min on CPU) ...")

    ppl_orig, ms_orig = compute_perplexity(sess_orig, in_orig, out_orig, tokenizer, texts)
    print(f"            FP32  done — ppl={ppl_orig:.2f}")

    ppl_q, ms_q = compute_perplexity(sess_q, in_q, out_q, tokenizer, texts)
    print(f"            INT{bits}  done — ppl={ppl_q:.2f}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    size_orig = orig_path.stat().st_size
    size_q    = q_path.stat().st_size

    delta_size = (size_q - size_orig) / size_orig * 100
    delta_ppl  = (ppl_q  - ppl_orig)  / ppl_orig  * 100
    delta_ms   = (ms_q   - ms_orig)   / ms_orig   * 100

    W = 68
    print(f"\n{'═' * W}")
    print(f"  RESULTS — GPT-2 small   FP32  →  INT{bits}")
    print(f"{'═' * W}")
    print(f"  {'Metric':<32}  {'FP32':>10}  {f'INT{bits}':>10}  {'Change':>8}")
    print(f"  {'─' * (W - 2)}")
    print(f"  {'File size':<32}  {fmt_bytes(size_orig):>10}  {fmt_bytes(size_q):>10}  {delta_size:>+7.1f}%")
    print(f"  {'Perplexity  (↓ better)':<32}  {ppl_orig:>10.2f}  {ppl_q:>10.2f}  {delta_ppl:>+7.1f}%")
    print(f"  {'Avg ms / fwd pass':<32}  {ms_orig:>9.1f}ms  {ms_q:>9.1f}ms  {delta_ms:>+7.1f}%")
    print(f"  {'─' * (W - 2)}")

    # Verdict
    if abs(delta_ppl) < 1.0:
        verdict = "Negligible quality loss"
        icon    = "[OK]"
    elif delta_ppl < 3.0:
        verdict = "Minor quality loss"
        icon    = "[~]"
    elif delta_ppl < 8.0:
        verdict = "Moderate quality loss"
        icon    = "[~]"
    else:
        verdict = "Significant quality loss"
        icon    = "[!!]"

    print(f"\n  {icon} {verdict}  (perplexity {delta_ppl:+.2f}%)")
    print(f"{'═' * W}\n")

    # ------------------------------------------------------------------
    # Generation comparison
    # ------------------------------------------------------------------
    print(f"{'─' * W}")
    print(f"  TEXT GENERATION — greedy, up to 80 new tokens")
    print(f"{'─' * W}")

    for prompt in GENERATION_PROMPTS:
        print(f'\n  Prompt: "{prompt}"')

        text_orig, ms_tok_orig = generate_text(sess_orig, in_orig, out_orig, tokenizer, prompt)
        text_q,    ms_tok_q    = generate_text(sess_q,    in_q,    out_q,    tokenizer, prompt)

        print(f"\n  FP32  ({ms_tok_orig:.0f} ms/tok):")
        print_wrapped(text_orig)

        print(f"\n  INT{bits}  ({ms_tok_q:.0f} ms/tok):")
        print_wrapped(text_q)

        print()

    print(f"{'─' * W}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT-2 quantization quality benchmark for quantize-rs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--export",    action="store_true", help="Export GPT-2 to ONNX (needs torch + transformers)")
    parser.add_argument("--quantize",  action="store_true", help="Quantize with quantize-rs")
    parser.add_argument("--benchmark", action="store_true", help="Run perplexity + generation benchmark")
    parser.add_argument("--all",       action="store_true", help="Run all three steps in sequence")

    parser.add_argument("--bits",         type=int,  default=8,    help="Bit width: 4 or 8  (default: 8)")
    parser.add_argument("--no-per-channel", action="store_true",   help="Disable per-channel quantization")
    parser.add_argument("--min-elements", type=int,  default=128,  help="Min elements to quantize  (default: 128; use 8192 for INT4)")

    parser.add_argument("--model-dir",    type=Path, default=Path("eval/models"), help="Directory for ONNX files  (default: eval/models)")
    parser.add_argument("--project-root", type=Path, default=Path("."),           help="quantize-rs project root  (default: .)")

    args = parser.parse_args()

    if not any([args.export, args.quantize, args.benchmark, args.all]):
        parser.print_help()
        sys.exit(0)

    per_channel = not args.no_per_channel

    if args.all or args.export:
        export_gpt2(args.model_dir)

    if args.all or args.quantize:
        quantize(
            args.model_dir,
            args.project_root,
            args.bits,
            per_channel,
            args.min_elements,
        )

    if args.all or args.benchmark:
        run_benchmark(args.model_dir, args.bits)


if __name__ == "__main__":
    main()
