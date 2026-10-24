#!/usr/bin/env python3
"""Fetch the four independently hashed corpora concurrently, then pinned vision assets."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fetch_data import FILES, fetch
from fetch_models import MODELS, fetch_model


def main():
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = [executor.submit(fetch, spec) for spec in FILES.values()]
        for job in jobs:
            job.result()
    fetch_model(MODELS["siglip2"])


if __name__ == "__main__":
    main()
