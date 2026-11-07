import json
import tempfile
from pathlib import Path
import unittest
from mini_vla.evaluation_standard_v1 import ROOT,STANDARD,load_cases

class EvaluationStandardTests(unittest.TestCase):
    def test_fast_set_is_exact_historical_subset_of_development(self):
        fast=load_cases('fast50');dev=load_cases('development200')
        expected={(t,i) for t in range(10) for i in [2,12,22,32,42]}
        self.assertEqual({(x['task_id'],x['initial_state_index']) for x in fast},expected)
        self.assertTrue({x['pair_id'] for x in fast}<={x['pair_id'] for x in dev})
        for task in range(10):self.assertEqual(sum(x['task_id']==task for x in dev),20)

    def test_old_development_and_confirmation_partition_is_unchanged(self):
        standard=json.loads((ROOT/STANDARD).read_text());manifest=json.loads((ROOT/standard['case_manifest']).read_text())
        old=json.loads((ROOT/manifest['source_split']).read_text())
        for name in ['development','confirmation']:
            self.assertEqual({x['pair_id'] for x in old[name]}, {x['pair_id'] for x in manifest['cases'] if x['partition']==name})
        self.assertEqual(len(manifest['cases']),500)

    def test_confirmation_and_full_coverage_remain_sealed(self):
        for level in ['confirmation300','coverage500']:
            with self.assertRaises(PermissionError):load_cases(level)

    def test_unknown_level_is_rejected(self):
        with self.assertRaises(ValueError):load_cases('custom_best_cases')

    def test_a_release_without_frozen_execution_settings_is_rejected(self):
        standard=json.loads((ROOT/STANDARD).read_text())
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='eval_standard_test_') as tmp:
            p=Path(tmp)/'release.json'
            p.write_text(json.dumps({'status':'FROZEN','method_selection_closed':True,
                'case_manifest_sha256':standard['case_manifest_sha256'],'checkpoints':[{'path':'placeholder','sha256':'placeholder'}]}))
            with self.assertRaises(PermissionError):load_cases('confirmation300',p)

if __name__=='__main__':unittest.main()
