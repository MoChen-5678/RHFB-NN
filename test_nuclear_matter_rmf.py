from dpl_rhf.legacy.test_nuclear_matter_rmf import *  # noqa: F401,F403

if __name__ == "__main__":
    from dpl_rhf.legacy.test_nuclear_matter_rmf import (
        test_gauss_legendre_polynomial_exactness,
        test_pkdd_symmetric_matter_variational_and_thermodynamic_consistency,
    )

    test_gauss_legendre_polynomial_exactness()
    print("PASS Gauss-Legendre polynomial exactness")
    test_pkdd_symmetric_matter_variational_and_thermodynamic_consistency()
    print("PASS PKDD field equation and thermodynamic consistency")
