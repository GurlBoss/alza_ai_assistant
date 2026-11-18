"""Like run_once, but in recovery mode: an unread email that FAILED without any
reply (e.g. sending failed) is processed again.

Run from the repository root: python -m scripts.run_once_recovery
Stops after the first email it answers, like run_once.
"""

from scripts.run_once import main

if __name__ == "__main__":
    main(recovery=True)
