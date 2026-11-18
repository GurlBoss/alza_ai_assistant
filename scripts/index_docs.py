"""Full (re)index of the knowledge base bucket.

Run from the repository root: python -m scripts.index_docs [--force]
Unchanged files are skipped by their hash unless --force is given; files that
are no longer in the bucket are removed from the index.
"""

import argparse
import logging

from app.config import Settings
from app.knowledge_base.rag import KnowledgeBase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="reindex unchanged files")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    knowledge = KnowledgeBase.from_settings(Settings())
    if knowledge is None:
        raise SystemExit(
            "Set KB_BUCKET, VS_INDEX_ID, VS_ENDPOINT_ID, VS_DEPLOYED_INDEX_ID"
        )
    results = knowledge.index_all(force=args.force)
    for name, result in results.items():
        print(f"{result:10} {name}")
    if not results:
        print("No files in docs/")


if __name__ == "__main__":
    main()
