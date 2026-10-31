from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class RHFParameters:
    """Density-dependent RHF parameters in Core-1204 conventions."""

    model: str = "PKO2"
    hbar_c: float = 197.328284
    alpha_inv: float = 137.03602
    rho_sat: float = 0.151021
    mass_n: float = 938.9
    mass_p: float = 938.9
    m_sigma: float = 534.461766
    m_omega: float = 783.0
    m_rho: float = 769.0
    m_pi: float = 138.0
    g_sigma: float = 8.920597
    g_omega: float = 10.550553
    g_rho: float = 4.068299
    a_sigma: float = 1.375772
    b_sigma: float = 2.064391
    c_sigma: float = 3.052417
    d_sigma: float = 0.330459
    a_omega: float = 1.451420
    b_omega: float = 3.574373
    c_omega: float = 5.478373
    d_omega: float = 0.246668
    a_rho: float = 0.631605
    f_pi: float = 0.0
    a_pi: float = 0.0
    # Dimensionless f_rho/g_rho ratio in Define.f90.  ``rho_tensor_vertex``
    # applies the 1/(M_n+M_p) conversion performed by DDRHF.f90.
    f_rho: float = 0.0
    a_rho_tensor: float = 0.0

    @property
    def rho_tensor_vertex(self) -> float:
        return self.g_rho * self.f_rho / (
            self.mass_n / self.hbar_c + self.mass_p / self.hbar_c
        )

PKO2Parameters = RHFParameters


def pko2_parameters() -> RHFParameters:
    return RHFParameters()


def pka1_parameters() -> RHFParameters:
    """PKA1 parameters from Core-1204/Define.f90 lines 56-62."""

    return RHFParameters(
        model="PKA1",
        rho_sat=0.159996,
        mass_n=938.9,
        mass_p=938.9,
        m_sigma=488.227904,
        m_omega=783.0,
        m_rho=769.0,
        m_pi=138.0,
        g_sigma=8.372672,
        g_omega=11.270457,
        g_rho=3.649857,
        a_sigma=1.103589,
        b_sigma=16.490109,
        c_sigma=18.278714,
        d_sigma=0.135041,
        a_omega=1.126166,
        b_omega=0.108010,
        c_omega=0.141251,
        d_omega=1.536183,
        a_rho=0.544017,
        f_pi=1.030722,
        a_pi=1.2,
        f_rho=3.199491,
        a_rho_tensor=0.820583,
    )


def rhf_parameters(model: str) -> RHFParameters:
    name = model.upper()
    if name == "PKO2":
        return pko2_parameters()
    if name == "PKA1":
        return pka1_parameters()
    raise ValueError(f"unsupported RHF parameter set {model}")
