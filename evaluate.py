"""Run from repository root: PYTHONPATH=src python evaluate.py."""
import json
from pathlib import Path
import pandas as pd
from title_verifier import TitleVerifier
from title_verifier.evaluation import make_pairs, evaluate

if __name__ == '__main__':
    registry = pd.read_csv('data/registry.csv')
    metrics, predictions = evaluate(TitleVerifier(registry), make_pairs(registry))
    Path('reports').mkdir(exist_ok=True)
    Path('reports/metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    predictions.to_csv('reports/test_predictions.csv', index=False)
    print(json.dumps(metrics, indent=2))
