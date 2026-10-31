"""Check rejection before scientific allocation, using sentinel basis methods."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dpl_rhf.rhfb.coordinate_normal_functional import CoordinateNormalFunctional
from dpl_rhf.rhfb.execution_limits import ResourceSafetyError


class CoordinateAdmissionTests(unittest.TestCase):
    def test_large_space_rejected_before_density_or_basis_allocation(self):
        basis = SimpleNamespace(point_count=882, dimension=432)
        functional = CoordinateNormalFunctional(basis)
        states = {s: SimpleNamespace(q=SimpleNamespace(device='cpu')) for s in ('n', 'p')}
        with patch('dpl_rhf.rhfb.coordinate_normal_functional.torch.eye',
                   side_effect=AssertionError('allocation before admission')) as eye:
            for method in (functional.energy_only, functional.evaluate):
                with self.assertRaises(ResourceSafetyError):
                    method(states)
            eye.assert_not_called()
        self.assertEqual((basis.point_count, basis.dimension), (882, 432))


if __name__ == '__main__':
    unittest.main()
