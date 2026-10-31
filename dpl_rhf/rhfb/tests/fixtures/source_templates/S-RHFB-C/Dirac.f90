!*************************************************************************************************!
!
MODULE DIRACE
!
!*************************************************************************************************!
!
USE Define
USE BASE, only: BASIS 
Use RHFBlib
Use Eigenvalues
Use GOGNY
Use PairD
Use Broyden
Use omp_lib
Implicit none

!--- Expansion coefficient on the woods-saxon basis
    type WSEXP
        double precision, dimension(NBS, NBF) :: ua, va
        double precision, dimension(NBS, NBD) :: xa
        double precision, dimension(     NBD) :: ee
    end type WSEXP
    type (WSEXP), dimension(NBX, 2) :: wavc
    
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
        do i0 = 1, cell(it)%nb
        !--- Exchange potential    
            epotl(i0,it)%XG = zero;         epotl(i0,it)%XF = zero
            epotl(i0,it)%YG = zero;         epotl(i0,it)%YF = zero

        !--- Pairing potential
            d           = 2.5/hbc
            xr0         = 1.2*nuc%amas**third
            if(LFG.eq.1) then
                do i = 1, npt
                    gkap(i0, it)%dgg(1:npt, i)  = zero
                    gkap(i0, it)%dgg(i, i)      = d*dexp( -2.5d0*( (well%r(i) - xr0)/xr0 )**2 )
                    gkap(i0, it)%dff(1:npt, i)  = zero
                    gkap(i0, it)%dff(i, i)      = d*dexp( -2.5d0*( (well%r(i) - xr0)/xr0 )**2 )
                end do
                gkap(i0,it)%dgf = zero;     gkap(i0,it)%dfg = zero
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
    
    integer :: i1, i2, i, nd, kab, ka, i12, i1b, i2b
    double precision, allocatable, dimension(:) :: AH

    if( ii.eq.0 .and. inin.eq.0 ) return

    nd  = well%npt
    do it = 1, 2
        do ib = 1, cell(it)%nb
            ST0(ib,it)%AH   = STM(ib,it)%AH
            ST0(ib,it)%AB   = STM(ib,it)%AB

            kab = cell(it)%kbl(ib)
            ka  = kab;                          if(kab.lt.0) ka = KMX(2) - kab

        !--- Calculate the direct part
        !$OMP PARALLEL PRIVATE(fun, rel, i12, i1, i2, i1b, i2b, i)
        !$OMP DO SCHEDULE(STATIC)
            do i12 = 1, cell(it)%NDS(ib)
                
                i1  = cell(it)%i1(i12,ib);              i2  = cell(it)%i2(i12,ib)
                i1b = cell(it)%ii(i1, ib);              i2b = cell(it)%ii(i2, ib)
                
                fun(:)  = BASIS(ka,it)%XG(:,i1b)*dpotl(it)%vps(:)*BASIS(ka,it)%XG(:,i2b) + &
                        & BASIS(ka,it)%XF(:,i1b)*dpotl(it)%vms(:)*BASIS(ka,it)%XF(:,i2b) + &
                        & BASIS(ka,it)%XG(:,i1b)*dpotl(it)%vt (:)*BASIS(ka,it)%XF(:,i2b) + &
                        & BASIS(ka,it)%XF(:,i1b)*dpotl(it)%vt (:)*BASIS(ka,it)%XG(:,i2b)

                Call simps(fun, nd, well%h, rel)

                STM(ib,it)%AH(i12)  = BASIS(ka,it)%XKT(i1b,i2b) + rel
                
            !--- Wilson terms
            !    if(i1b.eq.i2b) STM(ib,it)%AH(i12) = STM(ib,it)%AH(i12) + BASIS(ka,it)%XWT(i1b)
            end do
        !$OMP END DO 

        !--- Calculate the exchange part
            if(IE.eq.0) go to 100
        !$OMP DO SCHEDULE(STATIC)
            do i12 = 1, cell(it)%NDS(ib)
                
                i1  = cell(it)%i1(i12,ib);              i2  = cell(it)%i2(i12,ib)
                i1b = cell(it)%ii(i1, ib);              i2b = cell(it)%ii(i2, ib)
                
                fun = zero
                do i = 1, nd
                    fun(:)  = BASIS(ka,it)%XG(:,i1b)*epotl(ib,it)%YG(:,i)*BASIS(ka,it)%XG(i,i2b) + &
                            & BASIS(ka,it)%XG(:,i1b)*epotl(ib,it)%YF(:,i)*BASIS(ka,it)%XF(i,i2b) + &
                            & BASIS(ka,it)%XF(:,i1b)*epotl(ib,it)%XG(:,i)*BASIS(ka,it)%XG(i,i2b) + &
                            & BASIS(ka,it)%XF(:,i1b)*epotl(ib,it)%XF(:,i)*BASIS(ka,it)%XF(i,i2b) + fun(:)
                end do

                Call simps(fun, nd, well%h, rel)

                STM(ib,it)%AH(i12)  = STM(ib,it)%AH(i12) + rel
            end do
        !$OMP END DO

        100 Continue
        !--- Calculate Pair-field
            if(LFG.eq.1) then  !--- Gogny pairing
            !$OMP DO SCHEDULE(STATIC)
                do i12 = 1, cell(it)%NDS(ib)
                    
                    i1  = cell(it)%i1(i12,ib);              i2  = cell(it)%i2(i12,ib)
                    i1b = cell(it)%ii(i1, ib);              i2b = cell(it)%ii(i2, ib)
                    
                    fun = zero
                    do i = 1, nd
                        fun(:)  = BASIS(ka,it)%XG(:,i1b)*gkap(ib,it)%dgg(:,i)*BASIS(ka,it)%XG(i,i2b) + &
                                & BASIS(ka,it)%XG(:,i1b)*gkap(ib,it)%dgf(:,i)*BASIS(ka,it)%XF(i,i2b) + &
                                & BASIS(ka,it)%XF(:,i1b)*gkap(ib,it)%dfg(:,i)*BASIS(ka,it)%XG(i,i2b) + &
                                & BASIS(ka,it)%XF(:,i1b)*gkap(ib,it)%dff(:,i)*BASIS(ka,it)%XF(i,i2b) + fun(:)
                    end do

                    Call simps(fun, nd, well%h, rel)
                
                    STM(ib,it)%AB(i12)  = rel
                end do
            !$OMP END DO
            else  ! --- Delta pairing
            !$OMP DO SCHEDULE(STATIC)
                do i12 = 1, cell(it)%NDS(ib)
                    i1  = cell(it)%i1(i12,ib);              i2  = cell(it)%i2(i12,ib)
                    i1b = cell(it)%ii(i1, ib);              i2b = cell(it)%ii(i2, ib)
                    
                    fun(:)  = BASIS(ka,it)%XG(:,i1b)*dkap(ib,it)%del(:)*BASIS(ka,it)%XG(:,i2b) + &
                            & BASIS(ka,it)%XF(:,i1b)*dkap(ib,it)%del(:)*BASIS(ka,it)%XF(:,i2b)

                    Call simps(fun, nd, well%h, rel)
                
                    STM(ib,it)%AB(i12)  = rel
                end do
            !$OMP END DO
            end if
        !$OMP END PARALLEL 
        end do
    end do
            
    !--- Convergence check
    if(ii.ge.1) then
        Call Convergence(nmix)
    end if
    
    return

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
End Subroutine Stiff
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
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
    double precision, allocatable, dimension(:,:) :: xa, AX
    double precision, allocatable, dimension(:) :: eig, eia

    integer :: it, ib, i0, n, id1, id2, iw, ka, NBB, i1, lit, maxl, i2, i12, is

    double precision, dimension(MSD) :: fun
    double precision :: rel, sn, snold, al, alold, xh, xl, dl, su, sv, dd, dn, epsl, alafm, sl
    double precision :: time1, time2
    character :: bbb

!--- test
    double precision, dimension(2*NBS) :: uu, vv

    data maxl/500/, epsl/.1d-5/, dl/20.0d0/

    fermi%tz(1:IBX) = nuc%npr(1:IBX)

!---    Calculate the Stiff matrix
    time1   = omp_get_wtime ( )
    Call Stiff
    time2   = omp_get_wtime ( )
    stiff_time  = time2 - time1
    write( 6, 200) 'Time used in calculating the stiff matrix:         ', stiff_time, 's'
    write(l6, 200) 'Time used in calculating the stiff matrix:         ', stiff_time, 's'
200 format(2x, a, f12.6, a, f12.6, a)
    
    do it = 1, IBX

    !--- Self-consistent procedure to determine the chemical potential
        xh  = fermi%ala(it) + dl;           xl  = fermi%ala(it) - dl
        lit = 1;                            sl  = one

        do while(lit.lt.maxl .and. sl.gt.epsl)            
            sn      = zero
            alafm   = fermi%ala(it)/hbc

        !--- loop over blocks
            do ib = 1, cell(it)%nb
            
                NBB = cell(it)%NDF(ib)
                if( .not.allocated( AX  ) ) allocate( AX (2*NBB, 2*NBB) )
                if( .not.allocated( xa  ) ) allocate( xa (2*NBB, 2*NBB) )
                if( .not.allocated( eig ) ) allocate( eig(2*NBB       ) )
                if( .not.allocated( eia ) ) allocate( eia(2*NBB       ) )
                
                
            !--- Prepare the Stiff matrix
                do i12 = 1, NBB**2
                    i1  = cell(it)%i1(i12,ib);         i2  = cell(it)%i2(i12,ib)
                    
                    AX(    i1,     i2)  =  STM(ib,it)%AH(i12)
                    AX(    i1, NBB+i2)  =  STM(ib,it)%AB(i12)
                    AX(NBB+i1,     i2)  =  STM(ib,it)%AB(i12)
                    AX(NBB+i1, NBB+i2)  = -STM(ib,it)%AH(i12)
                end do
            
            !--- Chemical potential for particle conservation
                do i1 = 1, NBB
                    AX(    i1,     i1)  = AX(    i1,     i1) - alafm
                    AX(NBB+i1, NBB+i1)  = AX(NBB+i1, NBB+i1) + alafm
                end do

            !--- Sovle the eigen system: eig is the set of eigenvalues and xa is the set of eigenvectors
            !   Call sdiag(2*NBB, 2*NBB, AX, eig, xa, eia, 1)
                Call Cdiag(2*NBB, 2*NBB, AX, eig, xa, is)

            !--- negative energy states
                do n = 1, cell(it)%ND(ib)
                    wavc(ib,it)%xa(1:NBB,n) = xa(1:NBB, cell(it)%ND(ib)+1-n)
                    wavc(ib,it)%ee(n)       = eig(      cell(it)%ND(ib)+1-n)*hbc
                end do

            !--- Eigenstates from NBS + 1 to NBS + NBF: positive energy states
                do n = 1, cell(it)%id(ib)
                    i0                      = cell(it)%ia(ib) + n
                    lev(it)%ee(i0)          = eig(NBB+n)*hbc
                    lev(it)%vv(i0)          = zero 
                    
                !--- Expansion coefficient of the quasi-particle orbits
                    wavc(ib,it)%ua(1:NBB,n) = xa(    1:  NBB, NBB+n)
                    wavc(ib,it)%va(1:NBB,n) = xa(NBB+1:2*NBB, NBB+n)

                    if( lev(it)%ee(i0).gt.XPT(it) .and. LFG.ne.1) cycle
                !--- Calculate the occupation probability of quasi-particle orbits
                    su  = zero;             sv  = zero
                    do  iw = 1, NBB
                        su  = su + wavc(ib,it)%ua(iw,n)**2
                        sv  = sv + wavc(ib,it)%va(iw,n)**2
                    end do 

                !---- For the blocking case
                    if( lev(it)%lpb(i0) )then
                        sv  = sv*(lev(it)%mu(i0)-one)/lev(it)%mu(i0) + su/lev(it)%mu(i0)
                        su  = one - sv
                    end if
                    lev(it)%vv(i0)  = sv
                    
                !--- For checking particle number conservation
                    sn  = sn + lev(it)%vv(i0)*lev(it)%mu(i0)
                end do          !--- end loop within block
                
                if( allocated( AX  ) ) deallocate( AX  )
                if( allocated( xa  ) ) deallocate( xa  )
                if( allocated( eig ) ) deallocate( eig )
                if( allocated( eia ) ) deallocate( eia )
            end do          !--- end loop over blocks

        !--- To determine the new lambda value
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

    !--- Calculate the wave functions and the first, second direvatives
        do ib = 1, cell(it)%nb
            
            ka  = cell(it)%kbl(ib);         if(ka.lt.0) ka  = KMX(2) - ka
            do n = 1, cell(it)%id(ib)
                i0  = cell(it)%ia(ib) + n

                wav(i0,it)%ug   = zero;                 wav(i0,it)%uf   = zero
                wav(i0,it)%vg   = zero;                 wav(i0,it)%vf   = zero

                wav(i0,it)%dug  = zero;                 wav(i0,it)%duf  = zero
                wav(i0,it)%dvg  = zero;                 wav(i0,it)%dvf  = zero

                wav(i0,it)%sug  = zero;                 wav(i0,it)%suf  = zero
                wav(i0,it)%svg  = zero;                 wav(i0,it)%svf  = zero
                
                dn  = wavc(ib,it)%ua(1,n)
                do i1 = 2, cell(it)%NDF(ib)
                    if( dabs(dn) .lt. dabs(wavc(ib,it)%ua(i1,n)) ) dn = wavc(ib,it)%ua(i1,n)
                    if( dabs(dn) .lt. dabs(wavc(ib,it)%va(i1,n)) ) dn = wavc(ib,it)%va(i1,n)
                end do
                wavc(ib,it)%ua(:,n) = wavc(ib,it)%ua(:,n)*sign(one, dn)
                wavc(ib,it)%va(:,n) = wavc(ib,it)%va(:,n)*sign(one, dn)
                    
                do i1 = 1, cell(it)%NDF(ib)
                    iw  = cell(it)%ii(i1,ib)
                    
                    wav(i0,it)%ug (:)   = BASIS(ka, it)%XG(:,iw)*wavc(ib,it)%ua(i1,n) + wav(i0,it)%ug (:)
                    wav(i0,it)%uf (:)   = BASIS(ka, it)%XF(:,iw)*wavc(ib,it)%ua(i1,n) + wav(i0,it)%uf (:)
                    wav(i0,it)%vg (:)   = BASIS(ka, it)%XG(:,iw)*wavc(ib,it)%va(i1,n) + wav(i0,it)%vg (:)
                    wav(i0,it)%vf (:)   = BASIS(ka, it)%XF(:,iw)*wavc(ib,it)%va(i1,n) + wav(i0,it)%vf (:)
                    
                    wav(i0,it)%dug(:)   = BASIS(ka, it)%DG(:,iw)*wavc(ib,it)%ua(i1,n) + wav(i0,it)%dug(:)
                    wav(i0,it)%duf(:)   = BASIS(ka, it)%DF(:,iw)*wavc(ib,it)%ua(i1,n) + wav(i0,it)%duf(:)
                    wav(i0,it)%dvg(:)   = BASIS(ka, it)%DG(:,iw)*wavc(ib,it)%va(i1,n) + wav(i0,it)%dvg(:)
                    wav(i0,it)%dvf(:)   = BASIS(ka, it)%DF(:,iw)*wavc(ib,it)%va(i1,n) + wav(i0,it)%dvf(:)
                    
                    wav(i0,it)%sug(:)   = BASIS(ka, it)%SG(:,iw)*wavc(ib,it)%ua(i1,n) + wav(i0,it)%sug(:)
                    wav(i0,it)%suf(:)   = BASIS(ka, it)%SF(:,iw)*wavc(ib,it)%ua(i1,n) + wav(i0,it)%suf(:)
                    wav(i0,it)%svg(:)   = BASIS(ka, it)%SG(:,iw)*wavc(ib,it)%va(i1,n) + wav(i0,it)%svg(:)
                    wav(i0,it)%svf(:)   = BASIS(ka, it)%SF(:,iw)*wavc(ib,it)%va(i1,n) + wav(i0,it)%svf(:)
                end do
            end do
        end do

    end do
    
    return

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
End Subroutine Dirac
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!


!*************************************************************************************************!
!
End Module DIRACE
!
!*************************************************************************************************!
