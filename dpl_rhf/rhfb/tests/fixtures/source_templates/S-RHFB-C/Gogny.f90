!*************************************************************************************************!
!
MODULE GOGNY
!
!*************************************************************************************************!
!
Use Define
Use RHFBlib
Use Jsymbols
Use DBESIK
Use Self, only: c3s, c3j, c3l, KLP
implicit none

!---- Gogny force D1
    type GognyF
        character(len=10) :: txtgog
        double precision, dimension(2) :: gr, gw, gb, gh, gm
        double precision :: w0, t3, vg
    end type GognyF

    type (GognyF) :: D1, D1S, GPF, D1N, D1M

    data D1%txtgog/'Gogny D1'/
    data D1%gr/   0.7d0,  1.20d0/, D1%gw/-402.40d0, -21.30d0/, D1%gb/-100.d0,-11.77d0/ 
    data D1%gh/-496.2d0, 37.27d0/, D1%gm/ -23.56d0, -68.81d0/
    data D1%w0/-115.0d0/, D1%t3/1350.d0/, D1%vg/1.0d0/

    data D1S%txtgog/'Gogny D1S'/
    data D1S%gr/0.7d0,1.2d0/, D1S%gw/-1720.30d0, 103.639d0/, D1S%gb/ 1300.00d0,-163.483d0/
    data D1S%gh/-1813.53d0, 162.812d0/, D1S%gm/ 1397.60d0,-223.934d0/
    data D1S%w0/-130.d0/, D1S%t3/1350.d0/, D1S%vg/1.0d0/

    data D1N%txtgog/'Gogny D1N'/
    data D1N%gr/0.8d0,1.2d0/, D1N%gw/-2047.61d0, 293.02d0/, D1N%gb/ 1700.00d0,-300.78d0/
    data D1N%gh/-2414.93d0, 414.59d0/, D1N%gm/ 1519.35d0,-316.84d0/
    data D1N%w0/-115.d0/, D1N%t3/1609.46d0/, D1N%vg/1.0d0/

    data D1M%txtgog/'Gogny D1M'/
    data D1M%gr/0.5d0,1.0d0/, D1M%gw/-12797.57d0, 490.95d0/, D1M%gb/ 14048.85d0,-752.27d0/
    data D1M%gh/-15144.43d0, 675.12d0/, D1M%gm/ 11963.89d0,-693.57d0/
    data D1M%w0/-115.36d0/, D1M%t3/1562.22d0/, D1M%vg/1.0d0/

!--- Matrix element of Gogny pairing force
    INTEGER, PARAMETER :: KMLu = KMX(1), KMLd = KMX(2) 
    TYPE GognyP
        double precision, dimension(MSD, MSD) :: gg
    END TYPE GognyP
    type (GognyP), dimension(0:NBX+1, 2) ::VGL
    
    type GognyForce 
        double precision, dimension(0:NBX+1, 2) :: xx, xy, yy, yx
        double precision, dimension(MSD, MSD) :: gg, gf, fg, ff
    end type GognyForce
    type (GognyForce), dimension(NBX, NBX) :: GoFC

!--- Pairing potential Delta(r,r) for Gogny force
    type Gogkap
        double precision, dimension(MSD, MSD) :: deg, def, dkag, dkaf
        double precision, dimension(MSD, MSD) :: dgg, dgf, dfg, dff
        double precision, dimension(MSD, MSD) :: xgg, xgf, xfg, xff
        double precision, dimension(MSD) :: xkag, xkaf
    end type Gogkap
    type (Gogkap), dimension(NBX, 2) :: gkap, gka0

    integer, private :: nit
!--- Temporary array
    type tempw
        double precision, dimension(2*MSD-1) :: wk
    end type tempw
    type (tempw), dimension(0:NBX+1), private :: wtk  
    
    double precision, dimension(MSD, MSD) :: pag, paf

Contains
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine LambdaG(lpr)
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!     The determination of lambda ala(it) for given proton/neutron number
!          2
!         k
!        --- = lambda - V (r)  , N / Z = Inte [ density,dr^3 ]
!        2 m
!                   3       2                            3/2      2
!        density = k / (3 pi ) = [ 2 m ( lambda - V(r) )]  / (3 pi )
!
!----------------------------------------------------------------------!
    logical, intent(in) :: lpr
    double precision :: almda, rala, ralb
    integer :: i

    if (lpr) write(l6,*) ' ****** BEGIN lambda ***************************'

!-- Find the up & below point for lambda then solve the equation
    do nit =1, 2
        fermi%ala(nit)  =  zero
        almda           =  100.d0

        almda           = almda - one
        rala            = dnnz(almda)
        ralb            = dnnz(almda-one)
        i               = 1
        do while(rala*ralb .gt. zero .and. i.le.500)
            almda       = almda - one
            rala        = dnnz(almda)
            ralb        = dnnz(almda-one)
            i           = i + 1
        end do
        if(i.gt.500) Stop 'In Lambda: failed in finding ala '
        fermi%ala(nit)  = rtbrent(dnnz, almda, almda-one, rala, ralb, epsi)
    end do

    if (lpr) then
        write(l6,'(5f12.4)') ( fermi%ala(i), i=1,2 )
        write(l6,*) ' ****** END LAMBDA ****************************'
    endif

!    return

!--  O 16
    fermi%ala(1)    =  -9.44479523;     fermi%ala(2)    =  -5.74106763

!-- Sn 114
    fermi%ala(1)    = -22.61114373;     fermi%ala(2)    =  -6.19315460

!-- Sn100
    fermi%ala(1)    = -13.54118423;     fermi%ala(2)    =  -0.04167044

!-- Pb208
    fermi%ala(1)    =  -7.55878081;     fermi%ala(2)    = -11.20400849
    return

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
End Subroutine LambdaG
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
FUNCTION dnnz(almda)
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
    double precision, intent(in) :: almda
    integer :: n, i
    double precision :: dnnz, res, rhoc(MSD)
    

    n       = 0
    do i = 1, well%npt
        res     = almda/hbc - dpotl(nit)%vps(i)
        if( res .lt. zero) exit
        n       = n + 1
        rhoc(n) = ( two*(pset%amu(nit)/hbc)*(almda/hbc - dpotl(nit)%vps(i)) )**1.5d0*(4/3*pi)
    end do
    call simps(rhoc, n, well%h, res)
    dnnz    = res - nuc%npr(nit)

    RETURN
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
End Function dnnz
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine gofac(lpr)
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!     calculates the pairing matrixelements with Gogny force
!
!     <n1,l1,j1; n2,l2,j2 | Vpp | n3,l3,j3; n4,l4,j4 >
!
!     the result are essentially matrix elements coupled to angular
!     momentum 0, i.e. the are nonvanishing only for
!     j1 = j2 = j  and j3 = j4 = j'
!
!     there is only one small modification:
!     the factor sqrt(2j+1)*sqrt(2j'+1), which arises from the normali-
!     zation of the Clebsch-Gordan coeffitients is taken out.
!
!=======================================================================
!
!     The radial part of Gogny's force : the modified Bessel  
!     function have been calculated with :  
!          ki = 1 . numerical integration with Legendre function 
!               2 . modified Bessel function 
!               3 . modified Bessel function times exponential factor 
!----------------------------------------------------------------------!

    logical, intent(in) :: lpr

    double precision :: vg, x, ALPH, e1, e2, yc, x1, x2, ax, h6, hu1, hu2, hd1, hd2, gg, ff, gf, fg
    double precision :: Y(NBX+2), pgt(MSD), rh2(2*MSD-1), sign
    integer :: i, k1, j1, lu1, ld1, k2, j2, lu2, ld2, lam, is, npt, KM, KODE, NZ, ir1, ir2, LI, LM
    integer :: kp1, kp2, L
    double precision, dimension(2) :: GA, GD

    if(lpr) write(l6,*) ' ****** BEGIN GOFAC ********************************'

    vg      = GPF%vg/hbc
    
    do i = 1,2
        GA(i)   = vg*( GPF%gw(i) - GPF%gh(i) )
        GD(i)   = vg*( GPF%gb(i) - GPF%gm(i) )
    end do
    
    !---- V_l =  (2l+1) /2 * exp( -(r1/mu)**2-(r2/mu)**2 )  ------- (1)  ki
!            Int_{-1}^1  P_l(x) exp( 2 r1 r2 x )
!         =  (2l+1) /2 * exp( -(r1/mu)**2-(r2/mu)**2 )  ------- (2)  ki
!            [2 Pi/  2 r1 r2]^1/2 I_l+1/2 ( 2 r1 r2 )
!
!--- Calculate V_lambda
    KM  = NBX + 2;           ALPH   = half;         KODE    = 2
    npt = well%npt
    h6  = well%h/6.0d0

    do ir2 = 2, 2*npt-1
        rh2(ir2)    = (ir2 - 1)*half*well%h
    end do
    rh2(1)  = well%r(1)

    do i = 1, 2
        do ir1 = 1, npt
            x1  = well%r(ir1)/GPF%gr(i)
            
            do ir2 = 1, 2*npt-1
                x2  = rh2(ir2)/GPF%gr(i)
                e1  = (x1 - x2)**2;                 e2  = dexp(-e1) 
 
                ax  = two*x1*x2
                yc  = e2*dsqrt(two*pi/ax)
                
                CALL DBESI(ax, ALPH, KODE, KM, Y, NZ);          if(NZ.gt.0) Stop ' DBESI: NZ > 0'
                do lam = 0, NBX+1
                    wtk(lam)%wk(ir2) = yc*Y(lam+1)
                end do
            end do
            do lam = 0, NBX+1
                Call Propagator( wtk(lam)%wk, npt, pgt)
                VGL(lam, i)%gg(ir1, 1:npt) = pgt(1:npt)*h6
            end do

        end do
    end do
    
!$OMP PARALLEL Do private(k1,lu1,ld1,k2,lu2,ld2,LI,LM,L,i,gf,gg,ff)
    do k1 = 1, NBX
        lu1 = KLP(k1)%lu;       ld1 = KLP(k1)%ld
        
        do k2 = 1, NBX
            lu2 = KLP(k2)%lu;       ld2 = KLP(k2)%ld
            
            !--- Initialize
            GoFC(k1,k2)%gg  = zero;             GoFC(k1,k2)%fg  = zero
            GoFC(k1,k2)%gf  = zero;             GoFC(k1,k2)%ff  = zero
            
            LI  = min( abs(lu1-lu2), abs(ld1-ld2) )
            LM  = max(     lu1+lu2,      ld1+ld2  )
            
            do L = LI, LM, 2
                do i = 1, 2
                    gf  = (two*L + one)*c3j(L,k1,k2)*( GA(i) + GD(i) )*half
                    gg  = gf - (two*L + one)*c3l(L,k1, k2)*GD(i)*half
                    ff  = gf - (two*L + one)*c3s(L,k1, k2)*GD(i)*half
                    
                    GoFC(k1, k2)%gg  = VGL(L,i)%gg*gg + GoFC(k1, k2)%gg
                    GoFC(k1, k2)%gf  = VGL(L,i)%gg*gf + GoFC(k1, k2)%gf
                    GoFC(k1, k2)%fg  = VGL(L,i)%gg*gf + GoFC(k1, k2)%fg
                    GoFC(k1, k2)%ff  = VGL(L,i)%gg*ff + GoFC(k1, k2)%ff
                end do
            end do
        end do
    end do
!$OMP END PARALLEL Do

    if(lpr) write(l6,*) ' ****** END GOFAC **********************************'
    return

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
End Subroutine gofac
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine KappaG(lpr)
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!     Calculates the Pairing Tensor Kappa(r,r')  
!
!----------------------------------------------------------------------!
    logical, intent(in) :: lpr

    integer :: it, ib, na, ne, i1, i2, npt, n
    double precision :: xmu

    if (lpr) write(l6,*) ' ****** BEGIN KAPPA ********************************'

!---- loop over neutrons and protons
    npt     = well%npt

    do it = 1, IBX

!---- loop over different j,l-blocks
    !$OMP PARALLEL PRIVATE(ib, na, ne, n, i2, xmu)
    !$OMP DO SCHEDULE(STATIC)
        do ib = 1, cell(it)%nb
            na  = cell(it)%ia(ib) + 1
            ne  = cell(it)%ia(ib) + cell(it)%id(ib)

            gkap(ib,it)%xkag    = zero;             gkap(ib,it)%xkaf    = zero
            gkap(ib,it)%xgg     = zero;             gkap(ib,it)%xff     = zero
            gkap(ib,it)%xgf     = zero;             gkap(ib,it)%xfg     = zero
            do n = na, ne

        !--- If the level is blocking
                xmu = lev(it)%mu(n);            if( lev(it)%lpb(n) ) xmu = xmu - two 

            !--- Considering the temperature effects
                do i2 = 1, npt
                    gkap(ib,it)%xgg(:,i2)   = ( wav(n,it)%ug(:)*wav(n,it)%vg(i2) + &
                                            &   wav(n,it)%vg(:)*wav(n,it)%ug(i2) )*half*xmu + gkap(ib,it)%xgg(:,i2)
                    gkap(ib,it)%xgf(:,i2)   = ( wav(n,it)%ug(:)*wav(n,it)%vf(i2) + &
                                            &   wav(n,it)%vg(:)*wav(n,it)%uf(i2) )*half*xmu + gkap(ib,it)%xgf(:,i2)
                    gkap(ib,it)%xfg(:,i2)   = ( wav(n,it)%uf(:)*wav(n,it)%vg(i2) + &
                                            &   wav(n,it)%vf(:)*wav(n,it)%ug(i2) )*half*xmu + gkap(ib,it)%xfg(:,i2)
                    gkap(ib,it)%xff(:,i2)   = ( wav(n,it)%uf(:)*wav(n,it)%vf(i2) + &
                                            &   wav(n,it)%vf(:)*wav(n,it)%uf(i2) )*half*xmu + gkap(ib,it)%xff(:,i2)
                end do
                gkap(ib,it)%xkag    = gkap(ib,it)%xkag + wav(n,it)%ug*wav(n,it)%vg*xmu
                gkap(ib,it)%xkaf    = gkap(ib,it)%xkaf + wav(n,it)%uf*wav(n,it)%vf*xmu
            end do
            
    !---- end loop ib over j,l-blocks
        end do
    !$OMP END DO
    !$OMP END PARALLEL
        
!---- end loop it over neutron and protons
    end do
!
    if (lpr) write(l6,*) ' ****** END KAPPA **********************************'
    return

!-end-KAPPA
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
End Subroutine KappaG
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine DeltaG(lpr)
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!     Calculates the Pairing Potential   Delta(r,r')  for Gogny Force
!
!     stores it in:  del(r,r',ib,it) = Delta(r,r') * r * r'
!
!----------------------------------------------------------------------!

    logical, intent(in) :: lpr
    integer :: it, ib1, k1, i2, ib2, k2, lam, npt, i1, lu1, lu2, ld1, ld2, LI, LM
    double precision :: svp 
    
    if (lpr) write(l6,*) ' ****** BEGIN DELTA ********************************'
    
!--- Calcuate the kappa matrix
    Call KappaG(.false.)

!---- loop over neutrons and protons
    gka0    = gkap
    do it = 1, IBX

!---- loop over different kappa-blocks
    !$OMP PARALLEL PRIVATE(ib1, k1, k2, ib2)
    !$OMP DO SCHEDULE(STATIC)
        do ib1 = 1, cell(it)%nb
            k1  = cell(it)%kbl(ib1);       if(k1.lt.0) k1 = KMX(2) - k1

            gkap(ib1,it)%dgg = zero;        gkap(ib1,it)%dgf = zero
            gkap(ib1,it)%dff = zero;        gkap(ib1,it)%dfg = zero

            do ib2  = 1, cell(it)%nb
                k2  = cell(it)%kbl(ib2);       if(k2.lt.0) k2  = KMX(2) - k2

                gkap(ib1,it)%dgg = gkap(ib1,it)%dgg - GoFC(k1,k2)%gg*gkap(ib2,it)%xgg
                gkap(ib1,it)%dgf = gkap(ib1,it)%dgf - GoFC(k1,k2)%gf*gkap(ib2,it)%xgf
                gkap(ib1,it)%dfg = gkap(ib1,it)%dfg - GoFC(k1,k2)%fg*gkap(ib2,it)%xfg
                gkap(ib1,it)%dff = gkap(ib1,it)%dff - GoFC(k1,k2)%ff*gkap(ib2,it)%xff
            end do            

    !---- end loop ib1 over kappa-blocks
        end do
    !$OMP END DO
    !$OMP END PARALLEL
                    
!---- end loop it over neutron and protons
    enddo


    if (lpr) write(l6,*) ' ****** END DELTA **********************************'
    return
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
End Subroutine DeltaG
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

!
!*************************************************************************************************!
!
End Module GOGNY
!
!*************************************************************************************************!
