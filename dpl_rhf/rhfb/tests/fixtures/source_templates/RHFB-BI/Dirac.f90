!*************************************************************************************************!
!
MODULE DIRACE
!
!*************************************************************************************************!
!
USE Define
USE BASE, only: BASIS, BASK, NBS, NBF, NBD
Use RHFBlib
Use GOGNY
Use PairD
Use omp_lib
Implicit none

!--- Expansion coefficient on the woods-saxon basis
    type WSEXP
        double precision, dimension(NBS) :: ua, va
    end type WSEXP

    type (WSEXP), dimension(NTX, 2) :: wavc

!--- The stiff matrix
    double precision, dimension(2*NBS, 2*NBS) :: AX

    type StiffM
        double precision, dimension(NBS, NBS) :: AB, AH
    end type StiffM
    type (StiffM), dimension(NBX, 2) :: STM, STM0
    double precision, dimension(NBS, NBS), private :: svh, svb

!--- V component of Bogoliubov quasi-particle wavefunctions
    type Bogoliubov_V
        double precision, dimension(MSD, NBS) :: vg, vf
        double precision, dimension(NBS, NBS) :: va
    end type Bogoliubov_V
    
    type (Bogoliubov_V), dimension(NBX, 2) :: QPWV

!--- Eigenvalues of the single-particle Hamiltonian
    type EigenSPH
        double precision :: eig, ek
        double precision, dimension(MSD) :: G, F
        double precision :: dsig, dome, drho, drtn, drvt, dcou, erea
        double precision :: esig, eome, erho, ertn, ervt, ecou, epio
    end type EigenSPH
    type (EigenSPH), dimension(NTX, 2) :: ESPH

!--- Computation time for calculating the stiff matrix
    double precision :: stiff_time

!--- Parameters for Woods-Saxon potential
    type woods
        double precision :: V0, akv
        double precision, dimension(2) :: vso, r0v, av, r0s, as
    end type woods

    type (woods), private :: WS

    data WS%V0/-71.28/, WS%akv/0.4616/
    data WS%vso/11.1175, 8.9698/, WS%r0v/1.2334,1.2496/, WS%av/0.615, 0.6124/
    data WS%r0s/1.1443, 1.1401/, WS%as/0.6476, 0.6469/

Contains
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine Initial(lpr)
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!     This subroutine prepares initial mean field potential with a woods-saxon type potential 
!
!-------------------------------------------------------------------------------------------------!

    logical, intent(in) :: lpr
    integer :: i, npt, it, ita, i0
    double precision :: rav, ras, vp, vls, u, w, r, rh, c, emcc(2), fsig, fome, frho, d, xr0
    double precision, dimension(MSD) :: argv, args

    if(inin.eq.0) return
    if(lpr) write(l6,*) ' ****************** BEGIN Intialization *********************************'

!--- Initialize the self-energies with Woods-Saxon Basis
    emcc        = pset%amu/hbc*two
    npt         = well%npt

    do it = 1, 2
        ita = 3 - it
        rav = WS%r0v(it)*nuc%amas**third
        ras = WS%r0s(it)*nuc%amas**third
        vp  = WS%V0*(one - WS%akv*(nuc%npr(it)-nuc%npr(ita))/nuc%amas)/hbc
        vls = vp*WS%vso(it)

        argv(1:npt)             =  (well%r(1:npt) - rav              )/WS%av(it)
        args(1:npt)             =  (well%r(1:npt) - rav + well%h*half)/WS%av(it)
        dpotl(it)%vps (1:npt)   =  vp /( one + dexp(argv(1:npt)) )

        argv(1:npt)             =  (well%r(1:npt) - ras              )/WS%as(it)
        args(1:npt)             =  (well%r(1:npt) - ras + well%h*half)/WS%as(it)
        dpotl(it)%vms (1:npt)   = -vls/( one + dexp(argv(1:npt)) ) - emcc(it)
    end do

!--- Coulomb potentials
    rav                 = WS%r0v(2)*nuc%amas**third
    i                   = rav/well%h + 1
    argv(1:i)           = half*(3.0/rav - well%r(1:i)**2/rav**3)
    argv(i+1:npt)       = one/well%r(i+1:npt)
    args(1:npt)         = argv(1:npt)*nuc%npr(2)/alphi

    dpotl(2)%vps(1:npt) = args(1:npt) + dpotl(2)%vps(1:npt)
    dpotl(2)%vms(1:npt) = args(1:npt) + dpotl(2)%vms(1:npt)

!--- Initialization of the exchange potentials and pairing potentials
    do it = 1, 2
        dpotl(it)%vt    = zero
        do i0 = 1, block(it)%nb
        !--- Exchange potential    
            epotl(i0,it)%XG = zero;         epotl(i0,it)%XF = zero
            epotl(i0,it)%YG = zero;         epotl(i0,it)%YF = zero

        !--- Pairing potential
            d           = 2.5/hbc
            xr0         = 1.2*nuc%amas**third
            if(LFG.eq.1) then
                do i = 1, npt
                    gkap(i0, it)%deg(1:npt, i)  = zero
                    gkap(i0, it)%deg(i, i)      = d*dexp( -2.5d0*( (well%r(i) - xr0)/xr0 )**2 )
                    gkap(i0, it)%def(1:npt, i)  = zero
                    gkap(i0, it)%def(i, i)      = d*dexp( -2.5d0*( (well%r(i) - xr0)/xr0 )**2 )
                end do
            else
                dkap(i0,it)%del = d*dexp( -2.5d0*( (well%r - xr0)/xr0 )**2 )
            end if
            
        end do
    end do

!-- printout of meson potentials
    if (lpr) then
        write(l6,101)
    101 format(/,6x,'x (fm)', 6x,'V+S(n)',7x,'V+S(p)',7x,'V-S(n)',7x,'V-S(p)')
        do i = 1, well%npt
            write(l6,'(5f12.4)') well%r(i), (dpotl(it)%vps(i)*hbc, dpotl(it)%vms(i)*hbc+2*pset%amu(it), it = 1, 2)
        end do
        write(l6,*) ' ************************************ END Intialization ***********************************'
    end if
    return

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
End Subroutine Initial
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine Stiff
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!--- This subroutine is used to calculate the stiff matrix
!
!-------------------------------------------------------------------------------------------------!
!
    integer :: ib, it
    double precision, dimension(MSD) :: fun
    double precision :: rel, sv
    
    integer :: i1, i2, i, nd, kab, ka

    nd      = well%npt
    STM0    = STM
    do it = 1, 2
        do ib = 1, block(it)%nb

            kab     = block(it)%kbl(ib)
            ka      = kab;                          if(kab.lt.0) ka = KMX(2) - kab

        !--- Calculate the direct part
        !$OMP PARALLEL PRIVATE(fun, rel, i1, i2, i)
        !$OMP DO SCHEDULE(STATIC)
            do i1 = 1, NBS
                do i2 = i1, NBS
                    fun = BASIS(i1, ka, it)%G*BASIS(i2, ka, it)%G*dpotl(it)%vps + &
                        & BASIS(i1, ka, it)%F*BASIS(i2, ka, it)%F*dpotl(it)%vms + &
                        & BASIS(i1, ka, it)%G*BASIS(i2, ka, it)%F*dpotl(it)%vt  + &
                        & BASIS(i1, ka, it)%F*BASIS(i2, ka, it)%G*dpotl(it)%vt

                    Call simps(fun, nd, well%h, rel)

                    STM(ib,it)%AH(i1, i2)   = BASK(ka, it)%EKT(i1, i2) + rel
                    if(IE.eq.1) STM(ib,it)%AH(i2, i1) = STM(ib,it)%AH(i1, i2)
                end do
            end do
        !$OMP END DO 

        !--- Calculate the exchange part
        !$OMP DO SCHEDULE(STATIC)
            do i1 = 1, NBS
                if(IE.eq.1) cycle
                do i2 = i1, NBS
                    fun     = zero
                    do i = 1, nd
                        fun(1:nd)   = BASIS(i1,ka,it)%G(1:nd)*BASIS(i2,ka,it)%G(i)*epotl(ib,it)%YG(1:nd,i) + &
                                    & BASIS(i1,ka,it)%G(1:nd)*BASIS(i2,ka,it)%F(i)*epotl(ib,it)%YF(1:nd,i) + &
                                    & BASIS(i1,ka,it)%F(1:nd)*BASIS(i2,ka,it)%G(i)*epotl(ib,it)%XG(1:nd,i) + &
                                    & BASIS(i1,ka,it)%F(1:nd)*BASIS(i2,ka,it)%F(i)*epotl(ib,it)%XF(1:nd,i) + fun(1:nd)
                    end do

                    Call simps(fun, nd, well%h, rel)

                    STM(ib,it)%AH(i1, i2)   = STM(ib,it)%AH(i1, i2) + rel
                    STM(ib,it)%AH(i2, i1)   = STM(ib,it)%AH(i1, i2)
                end do
            end do
        !$OMP END DO

        !--- Calculate Pair-field
            if(LFG.eq.1) then  !--- Gogny pairing
            !$OMP DO SCHEDULE(STATIC)
                do i1 = 1, NBS
                    do i2 = i1, NBS
                        fun = zero
                        do i = 1, nd
                            fun(1:nd)   = BASIS(i1,ka,it)%G(1:nd)*BASIS(i2,ka,it)%G(i)*gkap(ib,it)%deg(1:nd,i) +  &
                                        & BASIS(i1,ka,it)%F(1:nd)*BASIS(i2,ka,it)%F(i)*gkap(ib,it)%def(1:nd,i) +  fun(1:nd)
                        end do

                        Call simps(fun, nd, well%h, rel)
                
                        STM(ib,it)%AB(i1, i2)  = rel
                        STM(ib,it)%AB(i2, i1)  = rel
                    end do
                end do
            !$OMP END DO
            else  ! --- Delta pairing
            !$OMP DO SCHEDULE(STATIC)
                do i1 = 1, NBS
                    do i2 = i1, NBS
                        fun = BASIS(i1, ka, it)%G*BASIS(i2, ka, it)%G*dkap(ib, it)%del + &
                            & BASIS(i1, ka, it)%F*BASIS(i2, ka, it)%F*dkap(ib, it)%del

                        Call simps(fun, nd, well%h, rel)
                
                        STM(ib,it)%AB(i1, i2)  = rel
                        STM(ib,it)%AB(i2, i1)  = rel
                    end do
                end do
            !$OMP END DO
            end if
        !$OMP END PARALLEL
            
        !--- Convergence check
            svb = STM0(ib,it)%AB - STM(ib,it)%AB
            svh = STM0(ib,it)%AH - STM(ib,it)%AH
            
            sv  = maxval(dabs(svb));        si  = max(si, sv)
            sv  = maxval(dabs(svh));        si  = max(si, sv)
            
            if(ii.eq.1) cycle
            STM(ib,it)%AB   = STM0(ib,it)%AB - xmix*svb
            STM(ib,it)%AH   = STM0(ib,it)%AH - xmix*svh       
        end do
    end do

    return

End Subroutine Stiff
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine Dirac(lpr)
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!--- Calculate the eigenvalues and wave functions
!
!-------------------------------------------------------------------------------------------------!
!
    logical, intent(in) :: lpr
    double precision, dimension(2*NBS, 2*NBS) :: xa
    double precision, dimension(2*NBS) :: eig, eia, tmp

    integer :: it, ib, i0, n, id1, id2, iw, ka, NBB, i1, lit, maxl

    double precision, dimension(MSD) :: fun
    double precision :: rel, sn, snold, al, alold, xh, xl, dl, su, sv, dd, dn, epsl, alafm, sl
    double precision :: time1, time2
    character :: bbb

!--- test
    double precision, dimension(2*NBS) :: uu, vv

    data maxl/500/, epsl/.1d-5/, dl/20.0d0/

    NBB         = 2*NBS
    fermi%tz    = nuc%npr

!---    Calculate the Stiff matrix
!   Call cpu_time(time1)
    time1   = omp_get_wtime ( )
    Call Stiff
    time2   = omp_get_wtime ( )
!   Call cpu_time(time2)
    stiff_time  = time2 - time1
    write( 6, 200) 'Time used in calculating the stiff matrix:         ', stiff_time, 's'
    write(l6, 200) 'Time used in calculating the stiff matrix:         ', stiff_time, 's'
200 format(2x, a, f12.6, a, f12.6, a)
    
    do it = 1, 2

        xh      = fermi%ala(it) + dl
        xl      = fermi%ala(it) - dl
        
        lit     = 1
        sl      = one
        do while(lit.lt.maxl .and. sl.gt.epsl)            
            sn      = zero
            alafm   = fermi%ala(it)/hbc

            do ib = 1, block(it)%nb
                if(block(it)%id(ib).gt.NBF) Stop 'Not enough eigen states in Dirac!'
            
            !--- Calculate the Stiff matrix
                do i1 = 1, NBS
                    AX(    1:  NBS,     i1) =  STM(ib,it)%AH(1:NBS, i1)
                    AX(    1:  NBS, NBS+i1) =  STM(ib,it)%AB(1:NBS, i1)
                    AX(NBS+1:2*NBS,     i1) =  STM(ib,it)%AB(1:NBS, i1)
                    AX(NBS+1:2*NBS, NBS+i1) = -STM(ib,it)%AH(1:NBS, i1)
                end do
            
            !--- Chemical potential for particle conservation
                do i1 = 1, NBS
                    AX(    i1,     i1)  = AX(    i1,     i1) - alafm
                    AX(NBS+i1, NBS+i1)  = AX(NBS+i1, NBS+i1) + alafm
                end do

            !--- Sovle the eigen system: eig is the set of eigenvalues and xa is the set of eigenvectors
                Call sdiag(NBB, NBB, AX, eig, xa, eia, 1)

                ka  = block(it)%kbl(ib);        if(ka.lt.0) ka  = KMX(2) - ka
            !--- Store the expansion coefficient for quasi-particle states
                do n = 1, NBD
                    QPWV(ib, it)%va(1:NBS,     n)   = xa(NBS+1:NBB,     n)
                end do
                do n = 1, NBF
                    QPWV(ib, it)%va(1:NBS, NBD+n)   = xa(NBS+1:NBB, NBS+n)
                end do
                
                QPWV(ib, it)%vg = zero;         QPWV(ib, it)%vf = zero
                do n = 1, NBS
                    do iw = 1, NBS
                        QPWV(ib, it)%vg(1:MSD, n)   = BASIS(iw, ka, it)%G(1:MSD)*QPWV(ib,it)%va(iw, n) + &
                                                    & QPWV(ib, it)%vg(1:MSD, n)
                        QPWV(ib, it)%vf(1:MSD, n)   = BASIS(iw, ka, it)%F(1:MSD)*QPWV(ib,it)%va(iw, n) + &
                                                    & QPWV(ib, it)%vf(1:MSD, n)
                    end do
                end do

            !--- Eigenstates from NBS + 1 to NBS + NBF
                do n = 1, block(it)%id(ib)
                    i0                      = block(it)%ia(ib) + n
                    lev(it)%ee(i0)          = eig(NBS+n)*hbc
                    wavc(i0,it)%ua(1:NBS)   = xa(    1:NBS, NBS+n)
                    wavc(i0,it)%va(1:NBS)   = xa(NBS+1:NBB, NBS+n)

                    su                      = zero
                    sv                      = zero
                    do  iw = 1, NBS
                        su  = su + wavc(i0,it)%ua(iw)**2
                        sv  = sv + wavc(i0,it)%va(iw)**2
                    end do 

                !---- For the blocking case
                    if( lev(it)%lpb(i0) )then
                        sv          = sv*(lev(it)%mu(i0)-one)/lev(it)%mu(i0) + su/lev(it)%mu(i0)
                        su          = one - sv
                    end if
                    lev(it)%vv(i0)          = sv
                    sn                      = sn + lev(it)%vv(i0)*lev(it)%mu(i0)

                !--- Calculate the wave functions and the first, second direvatives
                    wav(i0,it)%ug           = zero;             wav(i0,it)%uf           = zero
                    wav(i0,it)%vg           = zero;             wav(i0,it)%vf           = zero

                    wav(i0,it)%dug          = zero;             wav(i0,it)%duf          = zero
                    wav(i0,it)%dvg          = zero;             wav(i0,it)%dvf          = zero

                    wav(i0,it)%sug          = zero;             wav(i0,it)%suf          = zero
                    wav(i0,it)%svg          = zero;             wav(i0,it)%svf          = zero
                    do iw = 1, NBS
                        wav(i0,it)%ug   = BASIS(iw, ka, it)%G *wavc(i0,it)%ua(iw) + wav(i0,it)%ug
                        wav(i0,it)%uf   = BASIS(iw, ka, it)%F *wavc(i0,it)%ua(iw) + wav(i0,it)%uf
                        wav(i0,it)%vg   = BASIS(iw, ka, it)%G *wavc(i0,it)%va(iw) + wav(i0,it)%vg
                        wav(i0,it)%vf   = BASIS(iw, ka, it)%F *wavc(i0,it)%va(iw) + wav(i0,it)%vf

                        wav(i0,it)%dug  = BASIS(iw, ka, it)%DG*wavc(i0,it)%ua(iw) + wav(i0,it)%dug
                        wav(i0,it)%duf  = BASIS(iw, ka, it)%DF*wavc(i0,it)%ua(iw) + wav(i0,it)%duf
                        wav(i0,it)%dvg  = BASIS(iw, ka, it)%DG*wavc(i0,it)%va(iw) + wav(i0,it)%dvg
                        wav(i0,it)%dvf  = BASIS(iw, ka, it)%DF*wavc(i0,it)%va(iw) + wav(i0,it)%dvf

                        wav(i0,it)%sug  = BASIS(iw, ka, it)%SG*wavc(i0,it)%ua(iw) + wav(i0,it)%sug
                        wav(i0,it)%suf  = BASIS(iw, ka, it)%SF*wavc(i0,it)%ua(iw) + wav(i0,it)%suf
                        wav(i0,it)%svg  = BASIS(iw, ka, it)%SG*wavc(i0,it)%va(iw) + wav(i0,it)%svg
                        wav(i0,it)%svf  = BASIS(iw, ka, it)%SF*wavc(i0,it)%va(iw) + wav(i0,it)%svf
                    end do
                end do
            end do

            if(lit.gt.1) dd = (sn - snold)/(fermi%ala(it) - alold)

            snold   = sn
            dn      = sn - nuc%npr(it)

        !--- Calculating the new lambda values
            alold   = fermi%ala(it)
            if(dn.lt.zero) then
                xl  = fermi%ala(it)
            else
                xh  = fermi%ala(it)
            end if

            if(lit.eq.1) then
                fermi%ala(it)   = fermi%ala(it) - 1.0d-1*dn
            else
                if(dd.eq.zero) dd = three
                fermi%ala(it)   = fermi%ala(it) - dn/dd

            !--- Bisection
                if(fermi%ala(it).lt.xl .or. fermi%ala(it).gt.xh) then
                    bbb             = 'B'
                    fermi%ala(it)   = half*(xh + xl)
                end if
            end if

        !--- Convergence over the chemical potential
            sl  = dabs(fermi%ala(it) - alold)
            lit = lit + 1
        end do

        if(lit.ge.maxl) Stop ' Failed in the Lambda-iteraction!'

        if(lpr .or. sn.ne.zero) then
            write(l6,140) lit,'. Lambda-Iter successful:', it, fermi%ala(it), dn, sn
            write( 6,140) lit,'. Lambda-Iter successful:', it, fermi%ala(it), dn, sn
  140       format(i10,a,i4,3f13.8)
        end if

    end do
    
    return

End Subroutine Dirac
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine bcslmd(it, lpr)
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!     This subroutine solves the BCS equations for new lambda .
!
!-------------------------------------------------------------------------------------------------!
!
    logical, intent(in) :: lpr
    integer :: ib, nmax, n0, kap, n, LTX, lit, it
    double precision :: xl, xh, sn1, sv, sev, eig, alold, snold, eps, epsl, dd, dltmax
    double precision, dimension(NTX, 2) :: se, sdlt

    double precision :: al, dn, sn
    character*1 :: bbb
    character(len=8) :: lab

    data eps/1.d-12/,epsl/1.d-14/, LTX/100/

    if (lpr) write(l6,*) '****** BEGIN BCSLMD ******************************'

    lit     = 0
    al      = fermi%ala(it)
    xl      = al - 50.d0
    xh      = al + 50.d0

    dltmax  = zero
    do lit = 1, LTX
        sn      = zero

    !---- loop over blocks
        do ib = 1, block(it)%nb

    !------- Set up the quantum numbers for the n'th state
            kap     = block(it)%kbl(ib)
            n0      = block(it)%ia(ib)
            nmax    = block(it)%id(ib)

    !------- Find the auxiliary parameter : single particle energy and delta 
            sn1     = zero
            do n = 1, nmax
                if ( lit.eq.1 ) then
                    eig             = lev(it)%ee(n0+n)
                    sv              = lev(it)%vv(n0+n)
                    se(n0+n,it)     = al + ( 1.d0 - 2.d0*sv )*eig
                    sdlt(n0+n,it)   = sqrt(abs(eig**2 - (se(n0+n, it) - al)**2))
                    sev             = se(n0+n, it)
                    sv              = half*( one - (sev-al)/eig )
                    if( abs(dltmax).lt.abs(sdlt(n0+n,it)) ) dltmax = abs( sdlt(n0+n,it) )
                else
                    sev             = se(n0+n,it)
                    eig             = sqrt(sdlt(n0+n,it)**2 + (sev-al)**2)
                    sv              = half*( one - (sev-al)/eig )
                endif
                lab     = lev(it)%tb(n0+n)
                sn1     = sn1 + sv*lev(it)%mu(n0+n)

                if(mod(lit,111).eq.0) write( 6,'(a,2i5,4f12.4)') lab, n, lev(it)%mu(n0+n), 1.d0-sv, sv, sdlt(n0+n,it), se(n0+n,it)
                if(mod(lit,111).eq.0) write(l6,'(a,2i5,4f12.4)') lab, n, lev(it)%mu(n0+n), 1.d0-sv, sv, sdlt(n0+n,it), se(n0+n,it)

    !------- end loop over n
            end do
        
            sn = sn + sn1
    !---- end loop over blocks
        end do

    !---- calculation of a new lambda-value
        if (abs(al-alold).lt.epsl .and. lit.gt.1 ) then 
!            Pause 'al-alold lt epsl'
            exit
        end if
        dn      = sn - fermi%tz(it)

        if( abs(dltmax).lt. 5.d-1 .and. abs(dn).lt.1d-4 ) then
!            write(*,*) dltmax
!            Pause 'Pairing too weak'
            exit
        end if

        if (dn.lt.zero) then
            xl  = al
        else
            xh  = al
        end if

        if (abs(dn).lt.eps) then 
!            Pause 'particle number conserved'
            exit
        end if

        if (lit.eq.1) then
            alold   = al
            snold   = sn
            al      = alold - 1.d-2*dn
        else
            dd      = (sn - snold)/(al - alold)
            if (dd.eq.zero) dd = 1.d-10
            alold   = al
            al      = al - dn/dd
            snold   = sn
        end if

        bbb  = 'S'
    !--- Bisection
        if (al.lt.xl.or.al.gt.xh)  then
            al  = half*(xl+xh)
            bbb = 'B'
        end if

        if (lit.gt.30)  then
            al  = ( xl + xh ) / two
            bbb = 'H'
        end if

        if (lpr) then
            write(l6,20)  lit, '. L-Iteration: ', bbb, alold, dn, al    
            write( 6,20)  lit, '. L-Iteration: ', bbb, alold, dn, al
      20    format(i4, a, a1, 3f13.8)
        end if
    end do

    if(lit.gt.LTX) Stop ' Cannot converge in BCSLMD'

    if (lpr) then
        write(l6,40) lit,'. Lambda-Iter successful:',it,al,dn,sn
        write( 6,40) lit,'. Lambda-Iter successful:',it,al,dn,sn
  40    format(i8,a,i4,3f13.8)
    end if

    fermi%ala(it)   = al
     write(l6,40) lit,'. Lambda-Iter successful:',it,al,dn,sn
     write( 6,40) lit,'. Lambda-Iter successful:',it,al,dn,sn
    if (lpr) write(l6,*) '****** END BCSLMD ********************************'

    return

End Subroutine bcslmd

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine EstimateH
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!--- This subroutine is used to diagonalize the single-particle Hamiltonian
!
!-------------------------------------------------------------------------------------------------!
!
    double precision, dimension(NBS, NBS) :: xa, AH
    double precision, dimension(NBS) :: eig, eia, tmp

    integer :: ib, it, i0, n, ka, iw
    
    do it = 1, 2

        open (33, file = 'LEV/'//name(1:IFN)//'-levH_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown')
        write(33, '(a, 14(12x,a))') '  State    Node  2*j+1  k          E              Ekin',  &
        & 'Dsig', 'Dome', 'Drho', 'Drtn', 'Drvt', 'Dcou', 'Erea', 'Esig', 'Eome', 'Erho', 'Ertn',&
        & 'Ervt', 'Ecou', 'Epio'

        do ib = 1, block(it)%nb
            AH  = STM(ib, it)%AH
            Call sdiag(NBS, NBS, AH, eig, xa, eia, 1)
            
            ka  = block(it)%kbl(ib);        if(ka.lt.0) ka  = KMX(2) - ka
            do n = 1, block(it)%id(ib)
                i0                  = block(it)%ia(ib) + n
                ESPH(i0, it)%eig    = eig(NBD+n)*hbc
                
                ESPH(i0, it)%G      = zero
                ESPH(i0, it)%F      = zero
                do iw = 1, NBS
                    ESPH(i0, it)%G  = ESPH(i0, it)%G + BASIS(iw, ka, it)%G*xa(iw, NBD+n)
                    ESPH(i0, it)%F  = ESPH(i0, it)%F + BASIS(iw, ka, it)%F*xa(iw, NBD+n)
                end do
                Call ENGHDIS(ib, i0, it)

                write(33,'(a, 3i6, 16E16.6)')  lev(it)%tb(i0), n, lev(it)%mu(i0), lev(it)%nk(i0), &
                & ESPH(i0,it)%eig, ESPH(i0,it)%ek, ESPH(i0,it)%dsig, ESPH(i0,it)%dome, ESPH(i0,it)%drho, &
                & ESPH(i0,it)%drtn, ESPH(i0,it)%drvt, ESPH(i0,it)%dcou, ESPH(i0,it)%erea, ESPH(i0,it)%esig, &
                & ESPH(i0,it)%eome, ESPH(i0,it)%erho, ESPH(i0,it)%ertn, ESPH(i0,it)%ervt, ESPH(i0,it)%ecou, &
                & ESPH(i0,it)%epio
            end do
        end do
        
        close(33)
        
    end do
    return
    
End Subroutine EstimateH
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine ENGHDIS(ib, i0, it)
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!--- This subroutine calculate the single particle energy contributions from different channels
!
    double precision, dimension(MSD) :: dg, df, fun, sig, rv, rs, rt
    double precision, dimension(MSD) :: ome, rho, cou, rtn, rvt, rtv
    double precision, dimension(MSD, MSD) :: gg, gf, fg, ff

    integer :: ka, it, i0, nd, ib, n, i
    double precision :: h, tc(2)
		
    data tc/0.0d0, 1.0d0/
		
    nd	= well%npt;             h   = well%h
    
    sig = cct%gsig*dself%sig;   ome = cct%gome*dself%ome;       rho = cct%grho*dself%rho
    rtn = cct%grtn*dself%rtn;   rtv = cct%grtn*dself%rtv;       rvt = cct%grho*dself%rvt
        
    cou = dself%cou

    rv  = (ESPH(i0,it)%G**2 + ESPH(i0,it)%F**2)
    rs  = (ESPH(i0,it)%G**2 - ESPH(i0,it)%F**2)
    rt  = ESPH(i0,it)%G*ESPH(i0,it)%F*two
            
   !--- Kinetic energy
    ka  = lev(it)%nk(i0)
    Call deriv5(ESPH(i0,it)%g, dg, nd, h);      Call deriv5(ESPH(i0,it)%f, df, nd, h)
            
    fun = ESPH(i0,it)%f*dg - ESPH(i0,it)%g*df + two*ka*ESPH(i0,it)%g*ESPH(i0,it)%f/well%r + pset%amu(it)*rs/hbc
    Call simps(fun(2), nd-1, h, ESPH(i0,it)%ek);      ESPH(i0,it)%ek   = ESPH(i0,it)%ek*hbc - pset%amu(it)
            
   !--- Direct terms
    fun = sig*rs;               Call simps(fun, nd, h, ESPH(i0,it)%dsig);    ESPH(i0,it)%dsig = ESPH(i0,it)%dsig*hbc
    fun = ome*rv;               Call simps(fun, nd, h, ESPH(i0,it)%dome);    ESPH(i0,it)%dome = ESPH(i0,it)%dome*hbc
    fun = SigR*rv;              Call simps(fun, nd, h, ESPH(i0,it)%erea);    ESPH(i0,it)%erea = ESPH(i0,it)%erea*hbc
    fun = rho*rv;               Call simps(fun, nd, h, ESPH(i0,it)%drho);    ESPH(i0,it)%drho = ESPH(i0,it)%drho*hbc*tauz(it)
    fun = rtn*rt;               Call simps(fun, nd, h, ESPH(i0,it)%drtn);    ESPH(i0,it)%drtn = ESPH(i0,it)%drtn*hbc*tauz(it)
    fun = rtv*rt + rvt*rv;      Call simps(fun, nd, h, ESPH(i0,it)%drvt);    ESPH(i0,it)%drvt = ESPH(i0,it)%drvt*hbc*tauz(it)
    fun = cou*rv;               Call simps(fun, nd, h, ESPH(i0,it)%dcou);    ESPH(i0,it)%dcou = ESPH(i0,it)%dcou*hbc*tc(it)

        
   !--- Exchange terms
    do i = 1, nd
        gg(1:nd, i) = ESPH(i0,it)%G(1:nd)*ESPH(i0,it)%G(i);    gf(1:nd, i) = ESPH(i0,it)%G(1:nd)*ESPH(i0,it)%F(i)
        fg(1:nd, i) = ESPH(i0,it)%F(1:nd)*ESPH(i0,it)%G(i);    ff(1:nd, i) = ESPH(i0,it)%F(1:nd)*ESPH(i0,it)%F(i)
    end do
   
    sig = zero;         ome = zero;         rho = zero;         cou = zero
    rtn = zero;         rvt = zero;         rtv = zero;         fun = zero
    do i = 1, nd
        sig(1:nd)   = sig(1:nd) + gg(1:nd, i)*eself(ib,it)%sigyg(1:nd, i) + gf(1:nd, i)*eself(ib,it)%sigyf(1:nd, i) &
                    &           + fg(1:nd, i)*eself(ib,it)%sigxg(1:nd, i) + ff(1:nd, i)*eself(ib,it)%sigxf(1:nd, i)
        ome(1:nd)   = ome(1:nd) + gg(1:nd, i)*eself(ib,it)%omeyg(1:nd, i) + gf(1:nd, i)*eself(ib,it)%omeyf(1:nd, i) &
                    &           + fg(1:nd, i)*eself(ib,it)%omexg(1:nd, i) + ff(1:nd, i)*eself(ib,it)%omexf(1:nd, i)
        rho(1:nd)   = rho(1:nd) + gg(1:nd, i)*eself(ib,it)%rhoyg(1:nd, i) + gf(1:nd, i)*eself(ib,it)%rhoyf(1:nd, i) &
                    &           + fg(1:nd, i)*eself(ib,it)%rhoxg(1:nd, i) + ff(1:nd, i)*eself(ib,it)%rhoxf(1:nd, i)
        rtn(1:nd)   = rtn(1:nd) + gg(1:nd, i)*eself(ib,it)%rtnyg(1:nd, i) + gf(1:nd, i)*eself(ib,it)%rtnyf(1:nd, i) &
                    &           + fg(1:nd, i)*eself(ib,it)%rtnxg(1:nd, i) + ff(1:nd, i)*eself(ib,it)%rtnxf(1:nd, i)
        rvt(1:nd)   = rvt(1:nd) + gg(1:nd, i)*eself(ib,it)%rvtyg(1:nd, i) + gf(1:nd, i)*eself(ib,it)%rvtyf(1:nd, i) &
                    &           + fg(1:nd, i)*eself(ib,it)%rvtxg(1:nd, i) + ff(1:nd, i)*eself(ib,it)%rvtxf(1:nd, i)
        rtv(1:nd)   = rtv(1:nd) + gg(1:nd, i)*eself(ib,it)%rtvyg(1:nd, i) + gf(1:nd, i)*eself(ib,it)%rtvyf(1:nd, i) &
                    &           + fg(1:nd, i)*eself(ib,it)%rtvxg(1:nd, i) + ff(1:nd, i)*eself(ib,it)%rtvxf(1:nd, i)
        cou(1:nd)   = cou(1:nd) + gg(1:nd, i)*eself(ib,it)%couyg(1:nd, i) + gf(1:nd, i)*eself(ib,it)%couyf(1:nd, i) &
                    &           + fg(1:nd, i)*eself(ib,it)%couxg(1:nd, i) + ff(1:nd, i)*eself(ib,it)%couxf(1:nd, i)
        fun(1:nd)   = fun(1:nd) + gg(1:nd, i)*eself(ib,it)%pioyg(1:nd, i) + gf(1:nd, i)*eself(ib,it)%pioyf(1:nd, i) &
                    &           + fg(1:nd, i)*eself(ib,it)%pioxg(1:nd, i) + ff(1:nd, i)*eself(ib,it)%pioxf(1:nd, i)
    end do
    sig(1:nd)   = cct%gsig(1:nd)*sig(1:nd)*hbc;         Call simps(sig, nd, h, ESPH(i0,it)%esig)
    ome(1:nd)   = cct%gome(1:nd)*ome(1:nd)*hbc;         Call simps(ome, nd, h, ESPH(i0,it)%eome)
    rho(1:nd)   = cct%grho(1:nd)*rho(1:nd)*hbc;         Call simps(rho, nd, h, ESPH(i0,it)%erho)
    cou(1:nd)   =                cou(1:nd)*hbc;         Call simps(cou, nd, h, ESPH(i0,it)%ecou)
    rtn(1:nd)   = cct%grtn(1:nd)*rtn(1:nd)*hbc;         Call simps(rtn, nd, h, ESPH(i0,it)%ertn)
    fun(1:nd)   = cct%fpio(1:nd)*fun(1:nd)*hbc;         Call simps(fun, nd, h, ESPH(i0,it)%epio)
    
    rvt(1:nd)   = cct%grtn(1:nd)*rtv(1:nd)*hbc + cct%grho(1:nd)*rvt(1:nd)*hbc
                                                        Call simps(rvt, nd, h, ESPH(i0,it)%ervt)

    return    
End Subroutine ENGHDIS
End Module DIRACE
