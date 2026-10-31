"""Admission checks use only text fixtures; no nuclear calculation."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from dpl_rhf.rhfb.execution_limits import (
    ResourceSafetyError, require_bounded_cgroup, require_small_coordinate_space,
)


class ExecutionLimitsTests(unittest.TestCase):
    def _check(self, memory, swap='0', record='0::/job\n'):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'proc/self').mkdir(parents=True)
            (root/'cg/job').mkdir(parents=True)
            (root/'proc/self/cgroup').write_text(record)
            (root/'cg/job/memory.max').write_text(memory)
            (root/'cg/job/memory.swap.max').write_text(swap)
            return require_bounded_cgroup(proc_root=root/'proc', cgroup_root=root/'cg')

    def test_finite_kernel_limit(self):
        evidence = self._check(str(64*1024**2))
        self.assertEqual(evidence['memory_max_bytes'], 64*1024**2)
        self.assertTrue(evidence['enforced'])

    def test_refuse_missing_unlimited_and_excessive_limits(self):
        for memory in ('max', str(5*1024**3), '0', 'invalid'):
            with self.subTest(memory=memory), self.assertRaises(ResourceSafetyError):
                self._check(memory)
        with self.assertRaises(ResourceSafetyError):
            self._check('67108864', record='1:memory:/job\n')

    def test_refuse_swap_or_path_escape(self):
        for swap in ('max', '1048576'):
            with self.assertRaises(ResourceSafetyError):
                self._check('67108864', swap)
        with self.assertRaises(ResourceSafetyError):
            self._check('67108864', record='0::/../job\n')

    def test_large_coordinate_space_is_not_silently_reduced(self):
        for points, dimension in ((882, 432), (193, 96), (192, 97)):
            requested = SimpleNamespace(point_count=points, dimension=dimension)
            with self.assertRaises(ResourceSafetyError):
                require_small_coordinate_space(requested)
            self.assertEqual((requested.point_count, requested.dimension), (points, dimension))
        require_small_coordinate_space(SimpleNamespace(point_count=192, dimension=96))


if __name__ == '__main__':
    unittest.main()
