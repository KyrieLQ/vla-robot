#!/usr/bin/env python3
"""Safety boundary tests for selection, without touching a GPU."""
import unittest
from gpu_run import idle

class IdleSelectionTests(unittest.TestCase):
    def setUp(self):self.g=dict(compute_pids=[],used_mib=1,free_mib=24251,total_mib=24576,utilization=0)
    def test_idle_passes(self):self.assertTrue(idle(self.g))
    def test_zero_util_with_process_is_busy(self):self.g['compute_pids']=[123];self.assertFalse(idle(self.g))
    def test_zero_util_with_memory_is_busy(self):self.g.update(used_mib=14000,free_mib=10252);self.assertFalse(idle(self.g))
    def test_utilization_is_busy(self):self.g['utilization']=80;self.assertFalse(idle(self.g))
    def test_insufficient_free_is_busy(self):self.g['free_mib']=20000;self.assertFalse(idle(self.g))

if __name__=='__main__':unittest.main()
