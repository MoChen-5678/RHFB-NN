from dpl_rhf.legacy.test_fieldspace_rmf import *  # noqa: F401,F403

if __name__ == "__main__":
    from dpl_rhf.legacy.test_fieldspace_rmf import (
        test_autograd_directional_derivative,
        test_matrix_is_hermitian_and_states_are_normalized,
        test_second_order_derivative_is_consistent_inside_domain,
        test_zhang_adf_stencils,
    )

    test_matrix_is_hermitian_and_states_are_normalized()
    print("PASS hermiticity, orbital normalization, and particle numbers")
    test_second_order_derivative_is_consistent_inside_domain()
    print("PASS second-order noncentral derivative consistency")
    test_zhang_adf_stencils()
    print("PASS Zhang-style asymmetric finite-difference stencils")
    test_autograd_directional_derivative()
    print("PASS autograd directional derivative")
