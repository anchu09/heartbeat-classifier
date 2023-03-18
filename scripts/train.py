"""CLI wrapper for the training pipeline.

Delegates to ``heartbeat_classifier.training.trainer.main``.

Usage
-----
::

    python scripts/train.py --data-root . --output-dir results/run_01
"""

from heartbeat_classifier.training.trainer import main

if __name__ == "__main__":
    main()
