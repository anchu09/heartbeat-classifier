"""CLI wrapper for the evaluation pipeline.

Delegates to ``heartbeat_classifier.evaluation.evaluator.main``.

Usage
-----
::

    python scripts/evaluate.py \\
        --model results/run_01/model.keras \\
        --data-root . \\
        --output-dir results/eval_01
"""

from heartbeat_classifier.evaluation.evaluator import main

if __name__ == "__main__":
    main()
