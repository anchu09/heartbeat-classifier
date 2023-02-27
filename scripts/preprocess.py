"""CLI: Run the full ECG preprocessing pipeline.

Usage::

    python scripts/preprocess.py --data-root .
"""

from heartbeat_classifier.preprocessing.signal_processor import main

if __name__ == "__main__":
    main()
