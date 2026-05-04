#!/usr/bin/env python
"""Wrapper script for ingestion pipeline."""
from src.pipelines.ingest_pipeline import run

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--max", type=int, default=50)
    p.add_argument("--no-kg", action="store_true")
    args = p.parse_args()
    run(args.query, args.max, build_kg=not args.no_kg)
