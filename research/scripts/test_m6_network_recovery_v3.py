"""Deterministic CPU-only queue failure tests; no external network."""
import unittest
from collections import Counter
from unittest.mock import patch
from mini_vla import prepare_m6_v3 as m

class RecoveryTests(unittest.TestCase):
    def test_transient_file_is_deferred_without_repeating_success(self):
        calls=Counter();events=[]
        def worker(row,*args):
            name=row['rfilename'];calls[name]+=1
            if name=='b' and calls[name]==1:raise m.TransportUnavailable('TLS')
            return {'path':name}
        with patch.object(m.time,'sleep'):
            rows,history=m.collect_files([{'rfilename':x} for x in 'abc'],None,'fixed',lambda v,e:events.append(e),backoff_seconds=10,worker=worker)
        self.assertEqual({r['path'] for r in rows},set('abc'));self.assertEqual(calls,Counter(a=1,b=2,c=1));self.assertEqual(len(history),1)
        self.assertTrue(any(e['stage']=='NETWORK_BACKOFF' for e in events))

    def test_integrity_failure_is_not_network_retry(self):
        def worker(row,*args):raise ValueError('SHA256 mismatch')
        with self.assertRaises(ValueError):m.collect_files([{'rfilename':'a'}],None,'fixed',lambda *x:None,worker=worker)

    def test_permanent_outage_bounded_and_incomplete(self):
        calls=Counter();events=[]
        def worker(row,*args):calls[row['rfilename']]+=1;raise m.TransportUnavailable('TLS')
        with patch.object(m.time,'sleep') as sleep:
            with self.assertRaises(m.TransportUnavailable):
                m.collect_files([{'rfilename':str(i)} for i in range(30)],None,'fixed',lambda v,e:events.append(e),max_rounds=3,backoff_seconds=10,worker=worker)
        self.assertEqual(sleep.call_count,2);self.assertLessEqual(sum(calls.values()),15)
        self.assertEqual(events[-1]['stage'],'INCOMPLETE_DOWNLOAD');self.assertEqual(len(events[-1]['missing_paths']),30)

if __name__=='__main__':unittest.main()
