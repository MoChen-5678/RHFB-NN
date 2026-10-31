"""End-to-end synthetic NN orchestration; these are not nuclear calculations."""
from types import SimpleNamespace
import json

import pytest
import torch
from torch import nn

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.bogoliubov_blocking import bogoliubov_frame
from dpl_rhf.rhfb.manifold import slater_reference, generalized_density_matrix, GeneralizedDensity
from dpl_rhf.rhfb.network import GeneralizedDensityNet
from dpl_rhf.rhfb.selfconsistent_neural_training import train_joint


class ZeroDelta:
    def apply(self, vectors):
        return torch.zeros_like(vectors)


def networks():
    torch.manual_seed(20261015)
    basis = AngularRadialBasis(radius_fm=1.4, spacing_fm=.2, kappa_max=2,
                               derivative_order=5)
    q, labels = basis.build_initial_q(2, minimum=4, empty_states=2)
    reference = slater_reference(torch.zeros(q.shape[1], dtype=torch.float64))
    models = {}
    for s in ("n", "p"):
        model = GeneralizedDensityNet(basis, q, labels, reference, hidden=8,
                                      depth=1, radial_basis_size=4, pairing_seed=0.)
        m = model.active_dimension
        u = torch.eye(m, dtype=torch.complex128)
        v = torch.zeros_like(u)
        for i in (0, 2):
            u[i:i+2,i:i+2] *= torch.cos(torch.tensor(.4, dtype=torch.float64))
            v[i,i+1] = torch.sin(torch.tensor(.4, dtype=torch.float64))
            v[i+1,i] = -v[i,i+1]
        with torch.no_grad():
            model.canonical_number_transport.copy_(bogoliubov_frame(u,v))
            model._number_transport_active = True
        models[s] = model
    return basis, models


def quadratic_evaluator(basis, states, mus, targets, **kwargs):
    del mus, targets, kwargs
    # A prescribed one-body Hamiltonian, not a PKA1 nuclear calculation.
    h = torch.diag(torch.linspace(-2., 2., basis.dimension,
                                   dtype=torch.float64)).to(torch.complex128)
    energy = sum(torch.trace(h @ (state.q @ state.rho_active @ state.q.mH)).real
                 for state in states.values())
    return SimpleNamespace(loss=energy, total_energy=energy, particle_hole_energy=energy,
        pairing_energy=energy.new_zeros(()), normal_fields={s:h for s in states},
        active_pairing_gaps={s:torch.zeros_like(state.chi_active) for s,state in states.items()},
        delta_actions={s:ZeroDelta() for s in states})


def options():
    return dict(train_chemical_potentials=False, learning_rate=1.e-3,
                number_weight=0., correct_mean_numbers=True,
                stationarity_evaluator=quadratic_evaluator)


def test_main_trainer_retracts_real_network_and_saves_the_corrected_state(tmp_path):
    basis, models = networks()
    report = train_joint(models,basis,{"n":2,"p":2},1,tmp_path/"one",**options())
    assert report["accepted_updates"] == 1, report
    state = torch.load(tmp_path/"one/joint_step_0001.pt",weights_only=False)
    for s in ("n","p"):
        _, density = models[s]()
        assert abs(float(torch.trace(density.rho).real)-2) < 1.e-10
        assert state["mean_number_corrections"][s]["q_unchanged"]
        assert state["mean_number_corrections"][s]["converged"]
        torch.testing.assert_close(state["model_states"][s]["canonical_number_transport"],
                                   models[s].canonical_number_transport,atol=0.,rtol=0.)
    assert report["basis_refreshes"] == 0 and not report["eigensolver_used"]
    assert not report["physical_result_accepted"]


def test_field_failure_rolls_back_weights_and_number_transport(tmp_path):
    basis, models = networks()
    calls = 0
    def failing_evaluator(*args,**kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("synthetic current-field failure after trial retraction")
        return quadratic_evaluator(*args,**kwargs)
    args = options()
    args["stationarity_evaluator"] = failing_evaluator
    report = train_joint(models,basis,{"n":2,"p":2},1,tmp_path/"rollback",**args)
    assert report["status"] == "incomplete"
    assert report["accepted_updates"] == 0
    saved = torch.load(tmp_path/"rollback/joint_step_0000.pt",weights_only=False)
    for s in ("n","p"):
        for key,value in saved["model_states"][s].items():
            actual = models[s].state_dict()[key]
            if isinstance(value,torch.Tensor):
                assert torch.equal(value,actual),key
            else:
                assert value == actual


def test_number_transport_resume_matches_an_uninterrupted_run(tmp_path):
    basis, full = networks()
    report = train_joint(full,basis,{"n":2,"p":2},2,tmp_path/"full",**options())
    assert report["accepted_updates"] == 2, report
    _, split = networks()
    first = train_joint(split,basis,{"n":2,"p":2},1,tmp_path/"split",**options())
    assert first["accepted_updates"] == 1
    _, resumed = networks()
    second = train_joint(resumed,basis,{"n":2,"p":2},2,tmp_path/"split",
                         resume_from=tmp_path/"split/joint_step_0001.pt",**options())
    assert second["accepted_updates"] == 2, second
    for s in ("n","p"):
        for key,value in full[s].state_dict().items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(resumed[s].state_dict()[key],value,atol=1.e-12,rtol=0.)


class RotatedSlaterNet(nn.Module):
    reference_parity = 1
    blocked_index = None
    def __init__(self):
        super().__init__()
        self.angle = nn.Parameter(torch.tensor(.2,dtype=torch.float64))
    def forward(self):
        a = torch.zeros((4,4),dtype=torch.complex128)
        a[1,2],a[2,1] = self.angle,-self.angle
        q = torch.matrix_exp(a)
        rho = torch.diag(torch.tensor([1.,1.,0.,0.],dtype=torch.complex128))
        chi = torch.zeros_like(rho)
        return q,GeneralizedDensity(rho,chi,generalized_density_matrix(rho,chi),1)


def selfconsistent_toy(_basis,states,mus,targets,**kwargs):
    del mus,targets,kwargs
    t = torch.diag(torch.tensor([-1.,-.8,-20.,-22.],dtype=torch.complex128))
    rhos={s:state.q @ state.rho_active @ state.q.mH for s,state in states.items()}
    energy=sum((torch.trace(t@rho)+.025*torch.trace(rho@rho)).real for rho in rhos.values())
    return SimpleNamespace(loss=energy,total_energy=energy,particle_hole_energy=energy,
        pairing_energy=energy.new_zeros(()),normal_fields={s:t+.05*rho for s,rho in rhos.items()},
        active_pairing_gaps={s:torch.zeros((4,4),dtype=torch.complex128) for s in states},
        delta_actions={s:ZeroDelta() for s in states})


def branch_policy():
    from dpl_rhf.rhfb.bdg_branch_filter import BandEnclosureCertificate,BandBoundSource
    from dpl_rhf.rhfb.mixed_nucleon_sea_branch import (
        MixedNambuBandEvidence,EvidenceScope,MixedNucleonSeaBranchPolicy)
    # Weyl bounds: 0<=.05*rho<=.05 I for every Slater rho in this family.
    cert=BandEnclosureCertificate(.7,1.2,19.8,22.2,1,-1,.7,22.2,
        True,True,True,True,True,True,True,BandBoundSource.ANALYTIC_TOY,
        "Weyl-bound-t-plus-0.05rho-0<=rho<=I")
    ev=MixedNambuBandEvidence(5.,cert,EvidenceScope.SELFCONSISTENT_FAMILY,
        "analytic-Weyl-enclosure",field_family_id="four-mode-Hartree-toy")
    return MixedNucleonSeaBranchPolicy({"n":ev,"p":ev},blocked_species=(),
                                       quadrature_order=161,log_step=.35)


def test_actual_branch_policy_runs_inside_the_joint_nn_objective(tmp_path):
    models={s:RotatedSlaterNet() for s in ("n","p")}
    basis=SimpleNamespace(dimension=4,radius_fm=1.,spacing_fm=.2,kappa_max=0,point_count=1)
    policy=branch_policy()
    report=train_joint(models,basis,{"n":2,"p":2},1,tmp_path/"branch",
        branch_fixed_point_objective=policy,stationarity_evaluator=selfconsistent_toy,
        train_chemical_potentials=False,learning_rate=1.e-2,number_weight=0.)
    assert report["accepted_updates"] == 1,report
    assert report["optimization_target"] == "full_nambu_mixed_branch_fixed_point_plus_number_penalty"
    history=json.loads((tmp_path/"branch/history.json").read_text())
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["branch_diagnostics"]["species"]["n"]["evidence_fingerprint"]
    assert not report["no_sea_branch_certified"]
    saved=torch.load(tmp_path/"branch/joint_step_0001.pt",weights_only=False)
    assert saved["metadata"]["branch_fixed_point_objective"] == policy.checkpoint_metadata()


def test_even_branch_policy_cannot_be_silently_used_for_odd_target(tmp_path):
    with pytest.raises(ValueError,match="blocking must match"):
        train_joint({s:RotatedSlaterNet() for s in ("n","p")},
                    SimpleNamespace(dimension=4),{"n":1,"p":2},0,tmp_path/"bad-branch",
                    branch_fixed_point_objective=branch_policy(),
                    stationarity_evaluator=selfconsistent_toy)


def test_branch_evidence_is_rechecked_after_the_network_changes_the_field(tmp_path):
    from dataclasses import replace
    from dpl_rhf.rhfb.full_branch_stationarity import nambu_hamiltonian
    from dpl_rhf.rhfb.mixed_nucleon_sea_branch import (
        EvidenceScope, MixedNucleonSeaBranchPolicy, hamiltonian_sha256)
    from dpl_rhf.rhfb.operators import DensityFactors
    models={s:RotatedSlaterNet() for s in ("n","p")}
    basis=SimpleNamespace(dimension=4,radius_fm=1.,spacing_fm=.2,kappa_max=0,point_count=1)
    policy=branch_policy()
    with torch.no_grad():
        states={}
        for s,model in models.items():
            q,gd=model()
            states[s]=DensityFactors(q,gd.rho,gd.chi)
        evaluation=selfconsistent_toy(basis,states,None,None)
        evidence={s:replace(policy.evidence_by_species[s],scope=EvidenceScope.EXACT_HAMILTONIAN,
            field_family_id=None,hamiltonian_sha256=hamiltonian_sha256(
                nambu_hamiltonian(evaluation.normal_fields[s],torch.zeros((4,4),dtype=torch.complex128),0.)))
            for s in ("n","p")}
    exact_policy=MixedNucleonSeaBranchPolicy(evidence,blocked_species=())
    report=train_joint(models,basis,{"n":2,"p":2},1,tmp_path/"stale-proof",
        branch_fixed_point_objective=exact_policy,stationarity_evaluator=selfconsistent_toy,
        train_chemical_potentials=False,learning_rate=1.e-2,number_weight=0.)
    assert report["status"] == "incomplete",report
    assert report["accepted_updates"] == 0
    assert "different matrix" in report["error"]
    saved=torch.load(tmp_path/"stale-proof/joint_step_0000.pt",weights_only=False)
    for s in ("n","p"):
        assert torch.equal(models[s].angle,saved["model_states"][s]["angle"])


def test_com_observer_reports_post_variation_without_changing_training(tmp_path):
    import numpy as np
    from dpl_rhf.rhfb.cm_momentum_adf import (
        BasisMomentumOperators,PostVariationCoMObserver,_basis_signature)
    basis,plain=networks()
    # Synthetic independent operators test the reporting/optimization boundary;
    # the separate ADF tests establish actual coordinate matrix elements.
    d=basis.dimension
    p=np.zeros((3,d,d),dtype=np.complex128)
    p2=np.eye(d,dtype=np.complex128)
    ops=BasisMomentumOperators(p,p2,p.copy(),p.copy(),_basis_signature(basis),
        "synthetic-independent-operators",basis.hbar_c_mev_fm,{})
    observer=PostVariationCoMObserver(basis,ops,mass_number=4,nucleon_mass_mev=938.9)
    plain_report=train_joint(plain,basis,{"n":2,"p":2},1,tmp_path/"plain",**options())
    _,observed=networks()
    observed_report=train_joint(observed,basis,{"n":2,"p":2},1,tmp_path/"observed",
                                 post_variation_observer=observer,**options())
    assert plain_report["accepted_updates"] == observed_report["accepted_updates"] == 1
    for s in ("n","p"):
        for key,value in plain[s].state_dict().items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(observed[s].state_dict()[key],value,atol=0.,rtol=0.)
    rows=json.loads((tmp_path/"observed/history.json").read_text())
    cm=rows[-1]["post_variation_observables"]
    assert cm["energy_correction_mev"] < 0
    assert cm["included_in_variation"] is False
    assert cm["physical_result_accepted"] is False
    assert observed_report["post_variation_observer"] == observer.checkpoint_metadata()


def test_odd_neural_fixed_point_uses_current_blocked_branch_without_spectral_teacher(tmp_path):
    from dataclasses import replace
    from dpl_rhf.rhfb.mixed_nucleon_sea_branch import (
        BlockedModeEvidence,EvidenceScope,MixedNucleonSeaBranchPolicy)

    class OddOrEmptyNet(nn.Module):
        def __init__(self,empty=False):
            super().__init__()
            self.angle=nn.Parameter(torch.tensor(.2,dtype=torch.float64))
            self.empty=empty
            self.reference_parity=1 if empty else -1
            self.blocked_index=None if empty else 0
        def forward(self):
            a=torch.zeros((4,4),dtype=torch.complex128)
            a[0,1],a[1,0]=self.angle,-self.angle
            q=torch.matrix_exp(a)
            rho=torch.zeros((4,4),dtype=torch.complex128)
            if not self.empty:
                rho[0,0]=1.
            chi=torch.zeros_like(rho)
            return q,GeneralizedDensity(rho,chi,generalized_density_matrix(rho,chi),self.reference_parity)
        def forward_with_blocked_mode(self):
            q,gd=self()
            mode=None if self.empty else torch.cat((q[:,0],torch.zeros(4,dtype=q.dtype)))
            return q,gd,mode

    def diagonal_hartree(_basis,states,mus,targets,**kwargs):
        del mus,targets,kwargs
        t=torch.diag(torch.tensor([1.,2.,-20.,-22.],dtype=torch.complex128))
        rhos={s:state.q@state.rho_active@state.q.mH for s,state in states.items()}
        energy=sum((torch.trace(t@rho)+.025*torch.diag(rho).square().sum()).real
                   for rho in rhos.values())
        return SimpleNamespace(loss=energy,total_energy=energy,particle_hole_energy=energy,
            pairing_energy=energy.new_zeros(()),
            normal_fields={s:t+.05*torch.diag(torch.diag(rho)) for s,rho in rhos.items()},
            active_pairing_gaps={s:torch.zeros((4,4),dtype=torch.complex128) for s in states},
            delta_actions={s:ZeroDelta() for s in states})

    class SymmetryKnownMode:
        # The field family is exactly diagonal: e_1 is known analytically and
        # has zero response. No instantaneous eigenvectors are extracted.
        def checkpoint_metadata(self):
            return {"mode":"fixed-diagonal-symmetry-e1","spectral_teacher":False}
        def __call__(self,species,h,base):
            del species,base
            mode=torch.zeros(8,dtype=torch.complex128)
            mode[0]=1.
            return BlockedModeEvidence(mode=mode+0.*h[0,0],
                scope=EvidenceScope.SELFCONSISTENT_FAMILY,
                proof_id="diagonal-symbolic-eigenmode-zero-response",
                field_family_id="four-mode-diagonal-Hartree")

    original=branch_policy().evidence_by_species["n"]
    cert=replace(original.certificate,particle_abs_upper=2.2)
    evidence=replace(original,certificate=cert,field_family_id="four-mode-diagonal-Hartree")
    policy=MixedNucleonSeaBranchPolicy({s:evidence for s in ("n","p")},
        blocked_species=("n",),blocked_mode_sources={"n":SymmetryKnownMode()})
    models={"n":OddOrEmptyNet(),"p":OddOrEmptyNet(empty=True)}
    basis=SimpleNamespace(dimension=4,radius_fm=1.,spacing_fm=.2,kappa_max=0,point_count=1)
    report=train_joint(models,basis,{"n":1,"p":0},1,tmp_path/"odd-branch",
        branch_fixed_point_objective=policy,stationarity_evaluator=diagonal_hartree,
        train_chemical_potentials=False,learning_rate=1.e-2,number_weight=0.)
    assert report["accepted_updates"] == 1,report
    rows=json.loads((tmp_path/"odd-branch/history.json").read_text())
    assert rows[-1]["species"]["n"]["number_parity"] == -1
    assert rows[-1]["species"]["p"]["number_parity"] == 1
    assert rows[-1]["loss"] < rows[0]["loss"]
    assert not report["eigensolver_used"] and report["basis_refreshes"] == 0
    assert not report["physical_identity_certified"]
