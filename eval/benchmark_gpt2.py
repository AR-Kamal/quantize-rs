#!/usr/bin/env python3
"""GPT-2 coverage and INT8 quality evaluation.

    python eval/benchmark_gpt2.py --audit
    python eval/benchmark_gpt2.py --evaluate --download

See eval/GPT2_EVALUATION.md. Uses checksum-pinned WikiText-2 test data, with
no substitute texts. The old generation demo and mixed INT4 benchmark have
been replaced by a bounded symmetric per-channel INT8 evaluation.
"""
from gpt2_evaluation import main


if __name__ == "__main__":
    main()
