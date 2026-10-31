!*************************************************************************************************!
!
Module Define
!
!*************************************************************************************************!
!
Implicit None

!--- Definitions of common entry used in the program
!---- Number of D-meshpoints
!---- MSD    NF    ND   ECUT: 400~500MeV
!     201    28    12
!     251    36    12
!     301    48    12
    INTEGER, PARAMETER :: MSD = 251

!---- Maximal values of kappa, use different values of kappa limits for saving space
!--- KMX(1): j = l + 1/2; KMX(2): j = l - 1/2
    INTEGER, PARAMETER, DIMENSION(2) :: KMX = (/10,  9/)

!---- Number of j-blocks
    INTEGER, PARAMETER :: NBX = KMX(1) + KMX(2)

!---- Number of total levels
    INTEGER, PARAMETER :: NMX = 16              !--- When MSD is increased, NMX shall be increase, too.
    INTEGER, PARAMETER :: NTX = (NMX*2 + 1 - KMX(1)/2)*KMX(1)/2*2 - NMX + (NMX - KMX(1)/2)*2*MOD(KMX(1),2)

!---- Maximal number for GFV
    INTEGER, PARAMETER :: igfv =100

!---- Number of processors
    integer, parameter :: NoP = 10

!--- Hartree (IE = 1) or Hartree-Fock (IE = 2) approaches
    INTEGER :: IE
    DATA IE/2/
 
!--- The choice of pairing force: LFG = 1, (Gogny force), LFG = 0, (Delta force)
    Integer :: LFG
    Data LFG/1/

!--- Nucleus information: mass number (amas, nama), element name (nucnam), nucleon numbers (npr)
    type nuclide
        integer :: nama, npr(2)
        double precision :: amas
        character, dimension(2) :: nucnam
    end type nuclide
    type (nuclide) :: nuc

!--- Self-consistent iteration control
    logical :: lpc                          !--- for convergence
    integer :: ii, inxt, iaut, maxi, iwt
    character (len=3) :: cii
    double precision :: si, siold, xmix, xmix0, xmax
    double precision, parameter :: epsi = 0.1d-4
    data si/1.0d0/, iaut/0/, inxt/1/, xmix0/1.0d0/, xmix/1.0d0/, xmax/1.0d0/, maxi/200/
    data iwt/1/, lpc/.false./

!--- In-Out streams
    integer :: l6, lin, lou, lwin, lwou, lplo
    data l6/10/, lin/30/, lou/0/, lwin/31/, lwou/2/, lplo/11/

!--- Isospin projection of nucleon
    double precision, dimension(2) :: tauz
    data tauz/1.0d0, -1.0d0/

!--- Initial parameters for poentials and levels
    integer :: inin, levin
    data inin/1/, levin/1/

!--- Blocking information
    INTEGER, PARAMETER :: KBX = 5
    TYPE BLOCKING
        INTEGER :: nb
        INTEGER, DIMENSION(KBX) :: kp, il, ib
    End TYPE BLOCKING
    TYPE (BLOCKING), DIMENSION(2) :: itbl
!    integer, dimension(2) :: itbl, inbl, nbl

!--- Definition of the effective interactions
    type effective
        character(len=8) :: txtfor                        !--- Interction name
        integer :: len                                    !--- Length of name
        double precision :: rvs                           !--- Saturation density determined by the interaction
        double precision, dimension(2) :: amu             !--- Nucleon masses
        double precision :: amsig, amome, amrho, ampio    !--- Meson masses
        double precision :: gsig, gome, grho, grtn, fpio  !--- Coupling constants in all channels
        double precision :: arho, artn, apio              !--- Density dependent parameters in isovector channels
        double precision :: asig, bsig, csig, dsig        !--- Density dependent parameters for sigma-scalar
        double precision :: aome, bome, come, dome        !--- Density dependent parameters for omega-vector
        
        integer :: inl, IE                                !--- INL = 0 (density-dependent), 1 (s), 2 (s&o) and 3 (s,o, & r)
        double precision :: gs2, gs3, go2, gr2
    end type effective

    type (effective) :: pset                              !--- Variable used through the whole routine
    type (effective) :: PKO1, PKO2, PKO3, PKA1            !--- Effective interactions in DDRHF
    type (effective) :: PKDD, DDME1, DDME2, TW99          !--- Effective interactions in DDRMF
    type (effective) :: PK1, PK1r, NL3, NLSH, TM1, NL1, NL2, NLZ, NLZ2, TM2, NLSV    !--- Non-linear self-couplings

!--- Density dependent parameters and their derivatives with respect to the density
    type couplings
        double precision, dimension(MSD) :: gsig, gome, grho, fpio, grtn
        double precision, dimension(MSD) :: dsig, dome, drho, dpio, drtn
    end type couplings
    type (couplings) :: cct

!--- PKA1: Physcical Reviews C 76 (2007) 034314
    data PKA1%txtfor/'PKA1'/, PKA1%rvs/0.159996/, PKA1%amu/938.9, 938.9/, PKA1%len/4/, PKA1%inl/0/, PKA1%IE/1/
    data PKA1%amsig/488.227904/, PKA1%amome/783.000000/, PKA1%amrho/769.000000/, PKA1%ampio/138.000000/
    data PKA1%gsig /  8.372672/, PKA1%gome / 11.270457/, PKA1%grho /  3.649857/, PKA1%fpio/  1.030722/
    data PKA1%asig /  1.103589/, PKA1%bsig / 16.490109/, PKA1%csig / 18.278714/, PKA1%dsig/  0.135041/
    data PKA1%aome /  1.126166/, PKA1%bome /  0.108010/, PKA1%come /  0.141251/, PKA1%dome/  1.536183/
    data PKA1%arho /  0.544017/, PKA1%apio /  1.200000/, PKA1%grtn /  3.199491/, PKA1%artn/  0.820583/

!--- PKO1: Physics Letters B 640 (2006) 150¨C154
    data PKO1%txtfor/'PKO1'/, PKO1%rvs/0.151989/, PKO1%amu/938.9, 938.9/, PKO1%len/4/, PKO1%inl/0/, PKO1%IE/1/
    data PKO1%amsig/525.769084/, PKO1%amome/783.000000/, PKO1%amrho/769.000000/, PKO1%ampio/138.000000/
    data PKO1%gsig /  8.833239/, PKO1%gome / 10.729933/, PKO1%grho /  2.629000/, PKO1%fpio/  1.000000/
    data PKO1%asig /  1.384494/, PKO1%bsig /  1.513190/, PKO1%csig /  2.296615/, PKO1%dsig/  0.380974/
    data PKO1%aome /  1.403347/, PKO1%bome /  2.008719/, PKO1%come /  3.046686/, PKO1%dome/  0.330770/ 
    data PKO1%arho /  0.076760/, PKO1%apio /  1.231976/, PKO1%grtn /  0.000000/, PKO1%artn/  0.000000/

!--- PKO2: EPL, 82 (2008) 12001
    data PKO2%txtfor/'PKO2'/, PKO2%rvs/0.151021/, PKO2%amu/938.9, 938.9/, PKO2%len/4/, PKO2%inl/0/, PKO2%IE/1/
    data PKO2%amsig/534.461766/, PKO2%amome/783.000000/, PKO2%amrho/769.000000/, PKO2%ampio/138.000000/
    data PKO2%gsig /  8.920597/, PKO2%gome / 10.550553/, PKO2%grho /  4.068299/, PKO2%fpio/  0.000000/
    data PKO2%asig /  1.375772/, PKO2%bsig /  2.064391/, PKO2%csig /  3.052417/, PKO2%dsig/  0.330459/
    data PKO2%aome /  1.451420/, PKO2%bome /  3.574373/, PKO2%come /  5.478373/, PKO2%dome/  0.246668/
    data PKO2%arho /  0.631605/, PKO2%apio /  0.000000/, PKO2%grtn /  0.000000/, PKO2%artn/  0.000000/

!--- PKO3: EPL, 82 (2008) 12001
    data PKO3%txtfor/'PKO3'/, PKO3%rvs/0.153006/, PKO3%amu/938.9, 938.9/, PKO3%len/4/, PKO3%inl/0/, PKO3%IE/1/
    data PKO3%amsig/525.667686/, PKO3%amome/783.000000/, PKO3%amrho/769.000000/, PKO3%ampio/138.000000/
    data PKO3%gsig /  8.895635/, PKO3%gome / 10.802690/, PKO3%grho /  3.832480/, PKO3%fpio/  1.000000/
    data PKO3%asig /  1.244635/, PKO3%bsig /  1.566659/, PKO3%csig /  2.074581/, PKO3%dsig/  0.400843/
    data PKO3%aome /  1.245714/, PKO3%bome /  1.645754/, PKO3%come /  2.177077/, PKO3%dome/  0.391293/ 
    data PKO3%arho /  0.635336/, PKO3%apio /  0.934122/, PKO3%grtn /  0.000000/, PKO3%artn/  0.000000/

!--- TW99: Nuclear Physics A 656 (1999) 331
    data TW99%txtfor/'TW99'/, TW99%amu/939.0, 939.0/, TW99%rvs/0.153004/, TW99%len/4/, TW99%inl/0/, TW99%IE/0/
    data TW99%amsig/550.0     /, TW99%amome/783.0     /, TW99%amrho/763.0     /, TW99%ampio/138.0     /
    data TW99%gsig / 10.72854 /, TW99%gome / 13.29015 /, TW99%grho /  6.127157/, TW99%fpio/  0.0     /
    data TW99%asig /  1.365469/, TW99%bsig /  0.226061/, TW99%csig /  0.409704/, TW99%dsig/  0.901995/
    data TW99%aome /  1.402488/, TW99%bome /  0.172577/, TW99%come /  0.344293/, TW99%dome/  0.983955/
    data TW99%arho /  0.515   /, TW99%apio /  0.000000/, TW99%grtn /  0.000000/, TW99%artn/  0.000000/

!--- DDME1: Phys. Rev. C66, 024306(2002)
    data DDME1%txtfor/'DD-ME1'/, DDME1%amu/938.5, 938.5/, DDME1%rvs/0.151962/, DDME1%len/6/, DDME1%inl/0/, DDME1%IE/0/
    data DDME1%amsig/549.5255/, DDME1%amome/783.0   /, DDME1%amrho/763.0   /, DDME1%ampio/138.0   /
    data DDME1%gsig / 10.4434/, DDME1%gome / 12.8939/, DDME1%grho /  6.2789/, DDME1%fpio/  0.0   /
    data DDME1%asig /  1.3854/, DDME1%bsig /  0.9781/, DDME1%csig /  1.5342/, DDME1%dsig/  0.4661/
    data DDME1%aome /  1.3879/, DDME1%bome /  0.8525/, DDME1%come /  1.3566/, DDME1%dome/  0.4957/
    data DDME1%arho /  0.5008/, DDME1%apio /  0.0   /, DDME1%grtn /  0.0   /, DDME1%artn/  0.0   /

!--- DDME2: Phys. Rev. C71, 024312(2005)
    data DDME2%txtfor/'DD-ME2'/, DDME2%amu/938.9, 938.9/, DDME2%rvs/0.152/, DDME2%len/6/, DDME2%inl/0/, DDME2%IE/0/
    data DDME2%amsig/550.1238/, DDME2%amome/783.0   /, DDME2%amrho/763.0   /, DDME2%ampio/138.0   /
    data DDME2%gsig / 10.5396/, DDME2%gome / 13.0189/, DDME2%grho /  6.4792/, DDME2%fpio/  0.0   /
    data DDME2%asig /  1.3881/, DDME2%bsig /  1.0943/, DDME2%csig /  1.7057/, DDME2%dsig/  0.4421/
    data DDME2%aome /  1.3892/, DDME2%bome /  0.9240/, DDME2%come /  1.4620/, DDME2%dome/  0.4775/
    data DDME2%arho /  0.5647/, DDME2%apio /  0.0   /, DDME2%grtn /  0.0   /, DDME2%artn/  0.0   /

!--- PKDD: Phys. Rev. C69, 0034319(2004)
    data PKDD%txtfor/'PKDD'/, PKDD%amu/939.5731, 938.2796/, PKDD%rvs/0.149552/, PKDD%len/4/, PKDD%inl/0/, PKDD%IE/0/
    data PKDD%amsig/555.511236  /, PKDD%amome/783.0       /, PKDD%amrho/763.0       /, PKDD%ampio/138.0       /
    data PKDD%gsig / 10.738508  /, PKDD%gome / 13.147623  /, PKDD%grho /  5.164857  /, PKDD%fpio/  0.0       /
    data PKDD%asig /  1.32742274/, PKDD%bsig /  0.43512557/, PKDD%csig /  0.69166629/, PKDD%dsig/  0.69421032/
    data PKDD%aome /  1.34217027/, PKDD%bome /  0.37116653/, PKDD%come /  0.61139691/, PKDD%dome/  0.73837631/
    data PKDD%arho /  0.18330476/, PKDD%apio /  0.0       /, PKDD%grtn /  0.0       /, PKDD%artn/  0.0       /

!--- NL1: PRL 57-2916
    data NL1%txtfor/'NL1'/, NL1%amu(1:2)/938.0d0,938.0/, NL1%rvs/0.153/, NL1%len/3/, NL1%inl/1/, NL1%IE/0/
    data NL1%amsig/492.25d0/, NL1%amome/795.359d0/, NL1%amrho/763.0d0/, NL1%ampio/138.0/
    data NL1%gsig/10.1377d0/, NL1%gome/13.2846d0/, NL1%grho/4.975725d0/, NL1%fpio/0.0/, NL1%grtn/0.0/
    data NL1%gs2/-12.1724d0/, NL1%gs3/-36.2646d0/, NL1%go2/0.0/, NL1%gr2/0.0/

!--- NL2: PRL 57-2916
    data NL2%txtfor/'NL2'/, NL2%amu/938.0, 938.0/, NL2%rvs/0.153/, NL2%len/3/, NL2%inl/1/, NL2%IE/0/
	data NL2%amsig/504.89/, NL2%amome/780.0/, NL2%amrho/763.0/, NL2%ampio/138.0/
	data NL2%gsig/9.111/, NL2%gome/11.493/, NL2%grho/5.507/, NL2%fpio/0.0/, NL2%grtn/0.0/
	data NL2%gs2/-2.304/, NL2%gs3/13.783/, NL2%go2/0.0/, NL2%gr2/.0/

!--- NL3: Phys.Rev. C55 (1997) 540-543
    data NL3%txtfor/'NL3'/, NL3%amu(1:2)/939.0, 939.0/, NL3%rvs/0.148/, NL3%len/3/, NL3%inl/1/, NL3%IE/0/
    data NL3%amsig/508.194/, NL3%amome/782.5010/, NL3%amrho/763.0/, NL3%ampio/138.0/
    data NL3%gsig/10.2169/, NL3%gome/12.8675/, NL3%grho/4.4744/, NL3%fpio/0.0/, NL3%grtn/0.0/
    data NL3%gs2/-10.4307/, NL3%gs3/-28.8851/, NL3%go2/0.0/, NL3%gr2/0.0/

!--- NLZ:
    data NLZ%txtfor/'NL-Z'/, NLZ%amu/938.9, 938.9/, NLZ%rvs/0.145/, NLZ%len/4/, NLZ%inl/1/, NLZ%IE/0/
    data NLZ%amsig/488.67/, NLZ%amome/780./, NLZ%amrho/763./, NLZ%ampio/138.0/
    data NLZ%gsig/10.0553/, NLZ%gome/12.9086/, NLZ%grho/4.84944/, NLZ%fpio/0.0/, NLZ%grtn/0.0/
    data NLZ%gs2/-13.5072/, NLZ%gs3/-40.2243/, NLZ%go2/.0/, NLZ%gr2/.0/

!--- NLZ2:
    data NLZ2%txtfor/'NL-Z2'/, NLZ2%amu/938.9, 938.9/, NLZ2%rvs/0.145/, NLZ2%len/5/, NLZ2%inl/1/, NLZ2%IE/0/
    data NLZ2%amsig/493.150/, NLZ2%amome/780./, NLZ2%amrho/763./, NLZ2%ampio/138.0/
    data NLZ2%gsig/10.1369/, NLZ2%gome/12.9084/, NLZ2%grho/4.55627/, NLZ2%fpio/0.0/, NLZ2%grtn/0.0/
    data NLZ2%gs2/-13.7561/, NLZ2%gs3/-41.4013/, NLZ2%go2/.0/, NLZ2%gr2/.0/

! NL-SH: PLB312-377
    data NLSH%txtfor/'NL-SH'/, NLSH%amu(1:2)/939.0,939.0/, NLSH%rvs/0.146/, NLSH%len/5/, NLSH%inl/1/, NLSH%IE/0/
    data NLSH%amsig/526.059/, NLSH%amome/783.0/, NLSH%amrho/763.0/, NLSH%ampio/138.0/
    data NLSH%gsig/10.444/, NLSH%gome/12.945/, NLSH%grho/4.383/, NLSH%fpio/0.0/, NLSH%grtn/0.0/
    data NLSH%gs2/-6.9099/, NLSH%gs3/-15.8337/, NLSH%go2/0.d0/, NLSH%gr2/0.0/

!--- NL-SV1:
    data NLSV%txtfor/'NL-SV1'/, NLSV%amu/939.0,939.0/, NLSV%rvs/0.146/, NLSV%len/5/, NLSV%inl/2/, NLSV%IE/0/
    data NLSV%amsig/510.03488d0/,NLSV%amome/783.0d0/,NLSV%amrho/763.0d0/, NLSV%ampio/138.0/
    data NLSV%gsig/10.12479d0/,NLSV%gome/12.72661d0/,NLSV%grho/4.49197d0/, NLSV%fpio/0.0/, NLSV%grtn/0.0/
    data NLSV%gs2/-9.24058d0/,NLSV%gs3/-15.388d0/,NLSV%go2/41.01023/, NLSV%gr2/0.0/

!  TM1: NPA 579-557
    data TM1%txtfor/'TM1'/, TM1%amu(1:2)/938.0,938.0/, TM1%rvs/0.145/, TM1%len/3/, TM1%inl/2/, TM1%IE/0/
    data TM1%amsig/511.198d0/, TM1%amome/783.0d0/, TM1%amrho/770.0d0/, TM1%ampio/138.0/
    data TM1%gsig/10.0289d0/, TM1%gome/12.6139d0/, TM1%grho/4.6322d0/, TM1%fpio/0.0/, TM1%grtn/0.0/
    data TM1%gs2/-7.2325d0/, TM1%gs3/0.6183d0/, TM1%go2/71.3075d0/, TM1%gr2/0.0/

!--- TM2: NPA 579-557
    data TM2%txtfor/'TM2'/, TM2%amu/938.0, 938.0/, TM2%rvs/0.145/, TM2%len/3/, TM2%inl/2/, TM2%IE/0/
    data TM2%amsig/526.443d0/, TM2%amome/783.0d0/, TM2%amrho/770.0d0/, TM2%ampio/138.0/
    data TM2%gsig/11.4694d0/, TM2%gome/14.6377d0/, TM2%grho/4.6783d0/, TM2%fpio/0.0/, TM2%grtn/0.0/
    data TM2%gs2/-4.4440d0/, TM2%gs3/4.6076d0/, TM2%go2/84.5318d0/, TM2%gr2/.0/

!--- PK1: PRC 69-034319
    data PK1%txtfor/'PK1'/, PK1%amu(1:2)/939.5731,938.2796/, PK1%rvs/0.148/, PK1%len/3/, PK1%inl/2/, PK1%IE/0/
    data PK1%amsig/514.089144/, PK1%amome/784.254028/, PK1%amrho/763.0/, PK1%ampio/138.0/
    data PK1%gsig/10.322199/, PK1%gome/13.013093/, PK1%grho/4.529662/, PK1%fpio/0.0/, PK1%grtn/0.0/
    data PK1%gs2/-8.168840/, PK1%gs3/-9.997649/, PK1%go2/55.635980/, PK1%gr2/0.0/

!--- PK1R: PRC 69-034319
    data PK1R%txtfor/'PK1R'/, PK1R%amu(1:2)/939.5731,938.2796/, PK1R%rvs/0.148/, PK1R%len/4/, PK1R%inl/3/, PK1R%IE/0/
    data PK1R%amsig/514.087264/, PK1R%amome/784.222254/, PK1R%amrho/763.0/, PK1R%ampio/138.0/
    data PK1R%gsig/10.321864/, PK1R%gome/13.013354/, PK1R%grho/4.549978/, PK1R%fpio/0.0/, PK1R%grtn/0.0/
    data PK1R%gs2/-8.156193/, PK1R%gs3/-10.198444/, PK1R%go2/54.445920/, PK1R%gr2/350.000000/         

!--- Math and physics constants    
    double precision :: one, two, three, four, half, third, zero, pi, hbc, alphi, r0
    data one/1.0d0/, two/2.0d0/, half/0.5d0/, zero/0.0d0/, three/3.0d0/, four/4.0d0/
    data third/0.3333333333333333333333333333333333333333333333333333333333d0/
    data pi/3.141592653589793d0/, hbc/197.328284d0/, alphi/137.03602d0/, r0/1.2d0/

!--- Information for Fermi levels
    type fermis
        double precision, dimension(2) :: ala, tz, del
    end type fermis
    type (fermis) :: fermi
    data fermi%ala/-7.0,-7.0/

!--- Wave functions, the first and second differentials
    type waves
        double precision, dimension(MSD) ::  ug,  uf,  vg,  vf    !--- UV components of quasi-particle spinors
        double precision, dimension(MSD) :: dug, duf, dvg, dvf    !--- First order differential
        double precision, dimension(MSD) :: sug, suf, svg, svf    !--- Second order differential
    end type waves
    type (waves), dimension(NTX, 2) :: wav

!--- Canonical wave function
    type CANWAVE
        double precision :: vv, E, cdlt, r2
        double precision, dimension(MSD) :: f, g, den, df, dg
    end type CANWAVE
    type (CANWAVE), dimension(NTX, 2) :: cwav


!--- Kappa-block information
    type blocks
        integer  :: nb
        integer, dimension(NBX) :: kbl, id, ia, la, lap
		logical, dimension(NBX) :: lpb
        double precision, dimension(MSD, NBX) :: UD, VD
        character(len=5), dimension(NBX) :: tb
    end type blocks
    type (blocks), dimension(2) :: block

!--- Single particle information
    type level
        integer :: nt
        character(len=8), dimension(NTX) :: tb
        integer, dimension(NTX) :: nk, nr, mu, ioc, matc, la, lap
        double precision, dimension(NTX) :: ee, vv, del, spk, ec, vc
		logical, dimension(NTX) :: lpb
    end type level
    type (level), dimension(2) :: lev

!--- Direct part of the potentials 
    type dpotel
        double precision, dimension(MSD) ::  vps,  vms,  vt
        double precision, dimension(MSD) :: dvps, dvms, dvt
    end type dpotel
    type (dpotel), dimension(2) :: dpotl

!--- Exchange part of the potentials
    type epotel
        double precision, dimension(MSD, MSD) :: XG, XF, YG, YF
    end type epotel
    type (epotel), dimension(NBX, 2) :: epotl, opotl


!--- The direct part of the mean fields: sigma, omega, rho, and coulomb fields
    type dselfs
        double precision, dimension(MSD) :: sig, ome, rho, cou, rtn, rvt, rtv
    end type dselfs
    type (dselfs) :: dself

!--- The exchange part of the mean fields: sigma, omega, rho, pion and coulomb fields
    type eselfs
        double precision, dimension(MSD, MSD) :: sigxg, sigyg, sigxf, sigyf
        double precision, dimension(MSD, MSD) :: omexg, omeyg, omexf, omeyf
        double precision, dimension(MSD, MSD) :: rhoxg, rhoyg, rhoxf, rhoyf
        double precision, dimension(MSD, MSD) :: couxg, couyg, couxf, couyf
        double precision, dimension(MSD, MSD) :: pioxg, pioyg, pioxf, pioyf
        double precision, dimension(MSD, MSD) :: rtnxg, rtnyg, rtnxf, rtnyf
        double precision, dimension(MSD, MSD) :: rvtxg, rvtyg, rvtxf, rvtyf 
        double precision, dimension(MSD, MSD) :: rtvxg, rtvyg, rtvxf, rtvyf
    end type eselfs
    type (eselfs), dimension(NBX, 2) :: eself, tself, oself

!--- Rearrangement term
    double precision, dimension(MSD) :: SigR

!--- Contact part of pion and rho-tensor coupling
    type Contact
        double precision, dimension(MSD) :: pxg, pyf, rxg, rxf, ryg, ryf
    end type Contact
    type (Contact), dimension(2) :: CONT

!--- Spherical box and radial mesh informations
    type mesh
        integer :: npt
        double precision :: h, rmax
        double precision, dimension(MSD) :: r
    end type
    type (mesh) :: well
    data well%rmax/20.0d0/, well%h/0.10/

!--- Notations for Physics quantities
    type text
        character, dimension(:) :: tp(2), tis(2), tit(2)
        character, dimension(:) :: tl(0:20)
    end type text

    type (text) :: tex
    data tex%tp/'+', '-'/, tex%tis/'n', 'p'/, tex%tit/'N', 'P'/
    data tex%tl/'s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'P', 'q', 'r', &
               &'S', 't', 'u', 'v', 'w'/

!--- File name information for output
    character*8 :: name
    integer :: IFN


!--- For halo structure
    Integer, parameter :: NRX = MSD/10
    type Halos
        double precision, dimension(NRX) :: xnm, xno
    end type Halos
    type (Halos), dimension(2) :: Halo


End Module Define
