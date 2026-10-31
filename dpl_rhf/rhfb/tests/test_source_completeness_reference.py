"""Reference-only branch truncation check; no eigensolve enters NN training."""
import json
import os
from pathlib import Path

import pytest

from dpl_rhf.rhfb.source_completeness_reference import completeness_reference


def test_omitted_qp_terms_are_not_an_exact_pure_density_when_pairing_couples_sectors():
    result = completeness_reference()
    uncoupled, coupled = result["cases"]
    for case in result["cases"]:
        full = case["full_mixed_qp_sum"]
        assert full["chi_skew_error"] < 1.e-11
        assert full["generalized_purity_error"] < 1.e-11
    assert uncoupled["positive_fermi_qp_sum_only"]["generalized_purity_error"] < 1.e-11
    assert uncoupled["positive_fermi_qp_sum_only"]["chi_skew_error"] < 1.e-11
    omitted = coupled["positive_fermi_qp_sum_only"]
    assert coupled["omitted_sea_v_squared"] > 1.e-9
    assert omitted["chi_skew_error"] > 1.e-6
    assert omitted["generalized_purity_error"] > 1.e-6
    assert omitted["purity_error_after_chi_skew_symmetrization"] > 1.e-9
    assert result["physical_result_accepted"] is False
    output = os.environ.get("RHFB_COMPLETENESS_JSON")
    if output:
        Path(output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')


def test_reference_eigensolve_refuses_the_training_path():
    with pytest.raises(RuntimeError,match="forbidden"):
        completeness_reference(training_path=True)
