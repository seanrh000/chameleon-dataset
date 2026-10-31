"""Offline JSON-lines-free batch bridge for the experimental v0.5 engine."""
import argparse
import contextlib
import json
import sys
from .runtime import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    jobs = json.load(sys.stdin)
    if not isinstance(jobs, list): raise ValueError('Expected a list of resize tasks')
    with contextlib.redirect_stdout(sys.stderr):
        predictor = Predictor(args.checkpoint, device=args.device)
        results = []
        for job in jobs:
            design, report = predictor.resize(job['source'], job['target_canvas'])
            results.append({'design':design, 'report':report})
    json.dump(results, sys.stdout, ensure_ascii=False, allow_nan=False)


if __name__ == '__main__': main()
