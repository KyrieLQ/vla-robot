"""No GPU, process launch or message delivery in these admission guard tests."""
import copy
import unittest
from unittest.mock import patch
import mini_vla_dispatcher_v1 as d


class AdmissionTests(unittest.TestCase):
    def gpu(self, used=0, pids=None):
        return dict(uuid='test', index=5, compute_pids=pids or [], used_mib=used,
                    total_mib=24000, free_mib=24000-used, utilization=0)

    def check_samples(self, a, b):
        with patch.object(d, 'sample', side_effect=[a, b]), patch.object(d.time, 'sleep'):
            return d.ready_device({'resources': {'allowed_physical_gpus': [5]}})

    def test_idle_twice(self):
        self.assertEqual(self.check_samples([self.gpu()], [self.gpu()]), [5])

    def test_busy_then_idle_rejected(self):
        self.assertEqual(self.check_samples([self.gpu(600)], [self.gpu()]), [])

    def test_compute_process_rejected(self):
        self.assertEqual(self.check_samples([self.gpu(pids=[123])], [self.gpu()]), [])

    def test_became_busy_rejected(self):
        self.assertEqual(self.check_samples([self.gpu()], [self.gpu(600)]), [])

    def test_queued_hash_and_thread(self):
        entry=dict(plan='configs/mini_vla_m3_20261019_v2.json', launcher='scripts/start_mini_vla_m3_v2.sh',
                   run_name='mini_vla_test_only')
        for key in ['plan', 'launcher']:
            entry[key+'_sha256']=d.digest(d.ROOT/entry[key])
        d.validate(entry, 'PUBLIC_REPRODUCTION')
        with self.assertRaises(ValueError): d.validate(entry, 'wrong-thread')
        changed=copy.deepcopy(entry); changed['plan_sha256']='wrong'
        with self.assertRaises(ValueError): d.validate(changed, 'PUBLIC_REPRODUCTION')


if __name__ == '__main__':
    unittest.main()
