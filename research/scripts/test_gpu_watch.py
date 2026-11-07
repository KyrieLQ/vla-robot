#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import gpu_watch as watch

class MonitorTests(unittest.TestCase):
    def gpu(self, **changes):
        g=dict(index=0,uuid='GPU-test',compute_pids=[],used_mib=1,free_mib=24251,total_mib=24576,utilization=0)
        g.update(changes)
        return g

    def test_two_samples_required(self):
        self.assertEqual(watch.confirmed_idle([self.gpu(compute_pids=[1])],[self.gpu()]),[])
        self.assertEqual(watch.confirmed_idle([self.gpu()],[self.gpu(used_mib=16000)]),[])
        self.assertEqual(len(watch.confirmed_idle([self.gpu()],[self.gpu()])),1)

    def test_uuid_identity_not_position(self):
        self.assertEqual(watch.confirmed_idle([self.gpu()],[self.gpu(uuid='GPU-other')]),[])

    def test_cooldown(self):
        self.assertEqual(watch.eligible_alerts([self.gpu()],{'GPU-test':100},200,3600),[])
        self.assertEqual(len(watch.eligible_alerts([self.gpu()],{'GPU-test':100},3700,3600)),1)

    def test_message_prohibits_training(self):
        text=watch.notification_text('idle',Path('/example/event.json'),Path('/example/config.json'),[self.gpu()])
        self.assertIn('禁止自动训练',text)
        self.assertIn('不要因先前实验授权自动启动',text)

    def test_event_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'events').mkdir()
            cfg={'monitor_id':'test','thread':'test'}
            with patch.object(watch,'deliver',side_effect=lambda e,c:dict(e,delivery='accepted')) as sender:
                one=watch.make_event(p,p/'config.json',cfg,'idle',1,[self.gpu()])
                two=watch.make_event(p,p/'config.json',cfg,'idle',1,[self.gpu()])
            self.assertEqual(one['event_id'],two['event_id'])
            self.assertEqual(sender.call_count,1)

    def test_periodic_checks_expire_without_gpu_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);cfg={'monitor_id':'expiry-test','thread':'test','deadline_unix':time.time()+.18,
                            'interval_seconds':.05,'confirmation_seconds':0,'notification_cooldown_seconds':3600,
                            'max_idle_events':12}
            config=p/'config.json';config.write_text(json.dumps(cfg))
            with patch.object(watch,'sample',return_value=[self.gpu(compute_pids=[123])]), \
                 patch.object(watch,'deliver',side_effect=lambda e,c:dict(e,delivery='accepted')) as sender, \
                 patch.object(watch.signal,'signal'):
                watch.watch(config)
            state=json.loads((p/'state.json').read_text())
            self.assertEqual(state['status'],'EXPIRED')
            self.assertGreaterEqual(state['checks'],2)
            self.assertEqual(sender.call_count,1)
            self.assertEqual(sender.call_args[0][0]['kind'],'expired')

if __name__=='__main__':unittest.main()
