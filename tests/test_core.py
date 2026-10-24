import unittest
import pandas as pd
from title_verifier import TitleVerifier, normalize, features
from title_verifier.evaluation import make_pairs, evaluate

class TestVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = pd.read_csv('data/registry.csv')
        cls.v = TitleVerifier(cls.registry)
        cls.pairs = make_pairs(cls.registry)
        cls.metrics, _ = evaluate(cls.v, cls.pairs)

    def test_exact_normalization(self):
        result = self.v.search('  THE river CHRONICLE!!! ')
        self.assertEqual(result.iloc[0].decision, 'EXACT MATCH')
        self.assertEqual(result.iloc[0].title_id, 'T001')

    def test_invalid_inputs(self):
        for value in ['', '!!!', None, float('nan'), 'a'*501]:
            with self.assertRaises(ValueError): self.v.search(value)
        with self.assertRaises(ValueError): self.v.search('hello', 0)

    def test_duplicate_and_missing_registry(self):
        for data in [pd.DataFrame(), pd.concat([self.registry,self.registry.iloc[:1]])]:
            with self.assertRaises(ValueError): TitleVerifier(data)

    def test_features_symmetric(self):
        self.assertEqual(features('City Herald','Herald City'),features('Herald City','City Herald'))

    def test_split_isolation(self):
        sets = [set(self.pairs[self.pairs.split == s].family) for s in ['train','validation','test']]
        self.assertFalse(sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])
        for row in self.pairs.itertuples():
            candidate_family = self.registry.set_index('title_id').loc[row.candidate_id,'family']
            self.assertIn(candidate_family, sets[['train','validation','test'].index(row.split)])

    def test_results_and_retrieval(self):
        self.assertGreater(self.metrics['test_pairs'], 0)
        self.assertGreaterEqual(self.metrics['recall_at_5'], .8)
        self.assertTrue(self.v.search('river chroncile').similarity.between(0,1.000001).all())

if __name__ == '__main__': unittest.main()
