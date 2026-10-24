"""Reproducible family-held-out benchmark with validation-only threshold tuning."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .core import features


def make_pairs(registry):
    families = registry.family.unique().copy()
    np.random.default_rng(42).shuffle(families)
    partitions = {"train":families[:14], "validation":families[14:19], "test":families[19:]}
    rows = []
    for split, groups in partitions.items():
        subset = registry[registry.family.isin(groups)].reset_index(drop=True)
        for i, row in subset.iterrows():
            title = row.title
            words = title.split()
            variants = [title.upper()+'!', ' '.join(words[1:]), ' '.join(words[::-1]), title[:-2]+title[-1]+title[-2]]
            for query in variants:
                rows.append([query, title, 1, row.family, split, row.title_id])
            # Hard negatives share the subject; easy negatives share the format.
            hard = subset[(subset.family == row.family) & (subset.title_id != row.title_id)].iloc[0]
            easy = subset[(subset.family != row.family) & (subset.category == row.category)].iloc[0]
            for negative in [hard, easy]:
                rows.append([title, negative.title, 0, row.family, split, negative.title_id])
    return pd.DataFrame(rows, columns=['query','candidate','label','family','split','candidate_id'])


def evaluate(verifier, pairs):
    train, val, test = [pairs[pairs.split == s].copy() for s in ['train','validation','test']]
    verifier.fit_pairs(train)
    scores = verifier.pair_scores(val)
    grid = np.linspace(.05, .95, 91)
    # Tie break toward a higher threshold. Validation data only.
    threshold = max(grid, key=lambda t: (f1_score(val.label, scores >= t, zero_division=0), t))
    verifier.threshold = float(threshold)
    pred = verifier.pair_scores(test) >= threshold
    metrics = {'threshold':float(threshold), 'test_pairs':len(test),
               'accuracy':accuracy_score(test.label,pred), 'precision':precision_score(test.label,pred,zero_division=0),
               'recall':recall_score(test.label,pred,zero_division=0), 'f1':f1_score(test.label,pred,zero_division=0),
               'confusion_matrix':confusion_matrix(test.label,pred,labels=[0,1]).tolist()}
    # Retrieval evaluated on positive test queries, against the full registry.
    positives = test[test.label == 1]
    ranks = []
    for row in positives.itertuples():
        found = verifier.search(row.query, top_k=5).title_id.tolist()
        ranks.append(found.index(row.candidate_id)+1 if row.candidate_id in found else 0)
    metrics['recall_at_5'] = float(np.mean(np.array(ranks)>0))
    metrics['mrr_at_5'] = float(np.mean([1/x if x else 0 for x in ranks]))
    test['pair_score'] = verifier.pair_scores(test)
    test['prediction'] = pred.astype(int)
    return metrics, test
