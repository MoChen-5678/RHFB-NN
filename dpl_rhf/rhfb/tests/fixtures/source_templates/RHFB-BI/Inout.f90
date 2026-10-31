!*************************************************************************************************!
!
Module INOUTS
!
!*************************************************************************************************!
Use Define
Use RHFBlib
Use Densities
Use BASE
Use DIRACE
Implicit None

!--- for output
    double precision, dimension(NTX,MSD), private :: xde, xgc, xfc 
    character*8, dimension(NTX), private :: tab

!--- Contains reader, inout, kinout, output
Contains
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine reader
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!     This subroutine reads the input parameters from the file hfb.dat
!
!----------------------------------------------------------------------!

    character*10 fname
    integer :: ipa, it

!--- Open the input file
    if (lin.ne.0) open(lin,file='rhfb.dat',status='old')
    read(lin,100) l6

!    if (lou.ne.0) open(l6,file='rhfb.fg',status='unknown')

!-- Parameters for the Mesh:
    read(lin,101) well%rmax
    read(lin,101) well%h
    well%npt    = (well%rmax + 0.0000000001)/well%h + 1
    If(well%npt .gt.MSD) Stop 'MSD is too small!'
    do ipa  = 1, well%npt
        well%r(ipa) = (ipa - 1)*well%h
    end do
    well%r(1)   = well%h*1.0d-2

!-- Parameter for the iteration:
    read(lin,100) maxi
    read(lin,101) xmix
    xmix0 = xmix

!-- Initialization of wavefunctions:
    read(lin,100) inin

!-- Nucleus under consideration
    read(lin,'(2a1,3i4)') nuc%nucnam, nuc%nama, nuc%npr
    nuc%amas    = nuc%nama

!-- Blocvking neutron or proton block
    read(lin,*) itbl(1)%nb, itbl(2)%nb
    do it = 1, 2
        if(itbl(it)%nb.eq.0) then
            read(lin,*)
        else
            if( itbl(it)%nb.gt.KBX ) STOP 'In reader: Number of blocking too large!'
            read(lin,*) itbl(it)%kp(1:itbl(it)%nb)
        end if
    end do
    
100 format(10x,i5)
101 format(10x,2f10.3)
    close(lin)
    
    return

End Subroutine reader
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine output
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

    double precision :: csig, come, crho
    double precision :: pwi0, hom
    INTEGER :: IT

!--- OUTPUT file
    if(nuc%nama.lt.10) then
        if(nuc%nucnam(1).eq.' ') then
            write(name,'(a1,i1)') nuc%nucnam(2), nuc%nama
            IFN = 2
        else
            write(name,'(2a1,i1)') nuc%nucnam, nuc%nama
            IFN = 3
        end if
    else if(nuc%nama.lt.100) then
        if(nuc%nucnam(1).eq.' ') then
            write(name,'(a1,i2)') nuc%nucnam(2), nuc%nama
            IFN = 3
        else
            write(name,'(2a1,i2)') nuc%nucnam, nuc%nama
            IFN = 4
        end if
    else 
        if(nuc%nucnam(1).eq.' ') then
            write(name,'(a1,i3)') nuc%nucnam(2), nuc%nama
            IFN = 4
        else
            write(name,'(2a1,i3)') nuc%nucnam, nuc%nama
            IFN = 5
        end if
    end if

    if (l6.ne.6)  open(l6,file=name(1:IFN)//'.'//pset%txtfor,status='unknown')
    write(l6,*) ' *************** BEGIN READER ***************************'
    write(l6,201) ' Box-size  Rmax and r0 : ', well%rmax, r0
    write(l6,201) ' Step-size: ', well%h
    write(l6,200) ' Number of Mesh-points: ', well%npt
    write(l6,200) ' Maximal numb. of iterations: ',maxi
    write(l6,201) ' Mixing parameter     : ',xmix
    write(l6,200) ' inin: initial wavefunctions:  ', inin
    write(l6,200) ' levin: initialize levels   :  ', levin
    write(l6,'(a,20x,2a1,3i4)') ' Nucleus:  ',nuc%nucnam, nuc%nama, nuc%npr

    write(l6,*)
    If(LFG.eq.1) then
        write(l6,'(a, 5x, a)') ' The Gogny Pairing Force: ', GPF%txtgog
        write(l6,'(a, 5x, f12.5)') ' Gogny force factor:      ', GPF%vg
    else
        write(l6,'(a, 5x, 2f12.6)') ' The Delta Pairing Force: ', delPF%V0, delPF%rho0
    end if
    if(itbl(1)%nb.ge.1)write(l6,'(a,5i5)') ' Blocking n block - Probability ', itbl(1)%kp(1:itbl(1)%nb)
    if(itbl(2)%nb.ge.1)write(l6,'(a,5i5)') ' Blocking p block - Probability ', itbl(2)%kp(1:itbl(2)%nb)

!-- printout of force:
    csig        = pset%gsig*(pset%amu(1) + pset%amu(2))/pset%amsig*half
    come        = pset%gome*(pset%amu(1) + pset%amu(2))/pset%amome*half
    crho        = pset%grho*(pset%amu(1) + pset%amu(2))/pset%amrho*half
    write(l6,*)
    write(l6,'(/,a,a)') ' Meson-Parameters: ', pset%txtfor
    write(l6,'(a,2f12.6)') ' Nucleon mass                  : ', pset%amu
    write(l6,'(a,f10.4,a,f10.4,a,f10.4)') ' msig  = ', pset%amsig, '  gsig = ', pset%gsig, '  Csig', csig
    write(l6,'(a,f10.4,a,f10.4,a,f10.4)') ' mome  = ', pset%amome, '  gome = ', pset%gome, '  Come', come
    write(l6,'(a,f10.4,a,f10.4,a,f10.4)') ' mrho  = ', pset%amrho, '  grho = ', pset%grho, '  Crho', crho

!--- BASE information
    WRITE(l6,*)
    WRITE(l6,'(A,3I4)') ' RADIAL MESH SPLIT:', NBH
    WRITE(l6,'(A,3I4)') ' NUMBER OF THE STATES INVOLVED IN THE RHFB-DWS CALCULATIONS:', NBF, NBD, NBS
    WRITE( 6,'(A,3I4)') ' RADIAL MESH SPLIT:', NBH
    WRITE( 6,'(A,3I4)') ' NUMBER OF THE STATES INVOLVED IN THE RHFB-DWS CALCULATIONS:', NBF, NBD, NBS
    DO IT = 1, 2
        WRITE(l6,'(A,I4, 4X, 4F14.4)') ' THE ENERGY CUT-OFF OF THE BASIS:', IT, TOP(it), BOT(IT),&
                                  & TOP(IT)+pset%amu(it), BOT(IT)+pset%amu(it)
        WRITE( 6,'(A,I4, 4X, 4F14.4)') ' THE ENERGY CUT-OFF OF THE BASIS:', IT, TOP(it), BOT(IT),&
                                  & TOP(IT)+pset%amu(it), BOT(IT)+pset%amu(it)
    END DO

!-- size of the common blocks
    write( 6,'(/, A, a,i4, a, i4, a)') 'Kappa Maximum (spin-up, spin-down): ', '(', KMX(1), ', ', KMX(2), ')'
    write(l6,'(/, A, a,i4, a, i4, a)') 'Kappa Maximum (spin-up, spin-down): ', '(', KMX(1), ', ', KMX(2), ')'
    write( 6,'(/, 15h COMMON DELKAP:,4i4,i12)') MSD, MSD, NBX, 2, MSD*NBX*2
    write(l6,'(/, 15h COMMON DELKAP:,4i4,i12)') MSD, MSD, NBX, 2, MSD*NBX*2
    write( 6,'(15h COMMON GOGGOG:,4i4,i12)') (NBX+1), (NBX+1), NBX, 2, (NBX+1)**2 * NBX * 2
    write(l6,'(15h COMMON GOGGOG:,4i4,i12)') (NBX+1), (NBX+1), NBX, 2, (NBX+1)**2 * NBX * 2

    write( 6,'(15h COMMON WAVEFG:,4i4,i12)') MSD, NTX, 2, 4, MSD*NTX*2*4
    write(l6,'(15h COMMON WAVEFG:,4i4,i12)') MSD, NTX, 2, 4, MSD*NTX*2*4

    write(l6,*) ' ****** END READER *********************************'
    return

200 format(a,i5)
201 format(a,3f10.3)


End Subroutine output

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine inout(is, lpr)
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!
!     is = 1: reads meson-fields from tape
!          2: writes meson-fields  to tape
!
!----------------------------------------------------------------------!

    logical, intent(in) :: lpr
    Integer, intent(in) :: is

    character, dimension(2) :: tit
    integer :: it, ka, i0, i
    data tit/'N', 'P'/

!    if(lpr) write(l6,*) ' ****** BEGIN INOUT ********************************'

!-- reading of meson fields from tape:
    if (is.eq.1) then
        if(inin.ne.0) return
        
        do it = 1, 2
            open(lwin, file = name(1:IFN)//'.UV-'//tit(it)//'.'//pset%txtfor, status = 'old')
            read(lwin, '(E22.12)') fermi%ala(it)
            do i0 = 1, lev(it)%nt

                read(lwin, 100) wavc(i0,it)%ua(1:NBS), wavc(i0,it)%va(1:NBS)
                ka  = lev(it)%nk(i0);           if(ka.lt.0) ka  = KMX(2) - ka

            !--- Calculate the wave functions
                wav(i0,it)%ug           = zero;             wav(i0,it)%uf           = zero
                wav(i0,it)%vg           = zero;             wav(i0,it)%vf           = zero

                wav(i0,it)%dug          = zero;             wav(i0,it)%duf          = zero
                wav(i0,it)%dvg          = zero;             wav(i0,it)%dvf          = zero

                wav(i0,it)%sug          = zero;             wav(i0,it)%suf          = zero
                wav(i0,it)%svg          = zero;             wav(i0,it)%svf          = zero
                do i = 1, NBS
                    wav(i0,it)%ug   = BASIS(i, ka, it)%G *wavc(i0,it)%ua(i) + wav(i0,it)%ug
                    wav(i0,it)%uf   = BASIS(i, ka, it)%F *wavc(i0,it)%ua(i) + wav(i0,it)%uf
                    wav(i0,it)%vg   = BASIS(i, ka, it)%G *wavc(i0,it)%va(i) + wav(i0,it)%vg
                    wav(i0,it)%vf   = BASIS(i, ka, it)%F *wavc(i0,it)%va(i) + wav(i0,it)%vf

                    wav(i0,it)%dug  = BASIS(i, ka, it)%DG*wavc(i0,it)%ua(i) + wav(i0,it)%dug
                    wav(i0,it)%duf  = BASIS(i, ka, it)%DF*wavc(i0,it)%ua(i) + wav(i0,it)%duf
                    wav(i0,it)%dvg  = BASIS(i, ka, it)%DG*wavc(i0,it)%va(i) + wav(i0,it)%dvg
                    wav(i0,it)%dvf  = BASIS(i, ka, it)%DF*wavc(i0,it)%va(i) + wav(i0,it)%dvf

                    wav(i0,it)%sug  = BASIS(i, ka, it)%SG*wavc(i0,it)%ua(i) + wav(i0,it)%sug
                    wav(i0,it)%suf  = BASIS(i, ka, it)%SF*wavc(i0,it)%ua(i) + wav(i0,it)%suf
                    wav(i0,it)%svg  = BASIS(i, ka, it)%SG*wavc(i0,it)%va(i) + wav(i0,it)%svg
                    wav(i0,it)%svf  = BASIS(i, ka, it)%SF*wavc(i0,it)%va(i) + wav(i0,it)%svf
                end do
            end do
            close(lwin)
        end do
100     format(120E22.12)

    else
!-- writing of potentials to tape:
        do it = 1, 2
            open(lwou, file = name(1:IFN)//'.UV-'//tit(it)//'.'//pset%txtfor, status = 'unknown')

            write(lwou, '(E22.12)') fermi%ala(it)
            do i0 = 1, lev(it)%nt
                write(lwou, 100) wavc(i0,it)%ua(1:NBS), wavc(i0,it)%va(1:NBS)
            end do
            close(lwou)

        end do
    endif

!    If(lpr) write(l6,*) ' ****** END INOUT **********************************'

    return

End Subroutine inout

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine Export
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!--- Export the results during the iteration
!

    Integer :: it, n, i0, ib, lmx, i, j

!--- Create the directory for saving the exports
    if(lpc) then
        Call SYSTEM("mkdir DEN")
        Call SYSTEM("mkdir LEV")
        Call SYSTEM("mkdir WAV")
!        Call SYSTEM("mkdir LMF")
!        Call SYSTEM("mkdir NMF")
    else if(.not.lpc .and. ii.eq.1) then
        Call SYSTEM("mkdir ITER/LEV")
        Call SYSTEM("mkdir ITER/DEN")
        Call SYSTEM("mkdir ITER/WAV")
!        Call SYSTEM("mkdir ITER/LMF")
!        Call SYSTEM("mkdir ITER/NMF")
    end if
    
!--- OUTPUT the canonical single-particle information
    do it = 1, 2
    !--- select substantial orbits for output
        lmx = 0
        do i0 = 1, lev(it)%nt
            if(cwav(i0,it)%vv.gt.1.0e-8) then
                lmx         = lmx + 1
                xde(lmx,:)  = cwav(i0,it)%den(:)/cwav(i0,it)%vv
                xgc(lmx,:)  = cwav(i0,it)%g(:)/( well%r(:)*dsqrt(cwav(i0,it)%vv) )
                xfc(lmx,:)  = cwav(i0,it)%f(:)/( well%r(:)*dsqrt(cwav(i0,it)%vv) )
                tab(lmx)    = lev(it)%tb(i0)
            end if
        end do

    !--- Open the file for the output of the canonical single particle levels
        if(lpc) then

        !--- OUTPUT the probability densities with substantial contribution
            open (33, file = 'DEN/'//name(1:IFN)//'-DEN_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown')
            write(33, 100) 'r', ( tab(i0), i0 = 1, lmx )
            do i = 1, well%npt
                write(33, 103) well%r(i), (xde(i0, i), i0 = 1, lmx)
            end do
            close(33)
            
        !--- OUTPUT the single-particle information for pairing collapse, and the quasi-particle energy
            open(33, file = 'LEV/'//name(1:IFN)//'-levM_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown')
            open(34, file = 'LEV/'//name(1:IFN)//'.levQ_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown')
            write(33, 101) 'State', 'Node', '2*j+1', 'vc', 'EC', ' vv ', 'EQ'
            write(34, 101) 'State', 'Node', '2*j+1', 'vv', 'EQ'
            do ib = 1, block(it)%nb
                do n = 1, block(it)%id(ib)
                    i0 = block(it)%ia(ib) + n
                    
                    write(33,102) lev(it)%tb(i0), n, lev(it)%mu(i0), lev(it)%vc(i0), lev(it)%ec(i0), &
                        &                                            lev(it)%vv(i0), lev(it)%ee(i0)
                    
                    write(34,102) lev(it)%tb(i0), lev(it)%nk(i0), lev(it)%mu(i0), lev(it)%vv(i0), lev(it)%ee(i0)
                end do
            end do
            close(33); close(34)
            
        !--- OUTPUT canonical single particle information
            open (33, file = 'LEV/'//name(1:IFN)//'-lev_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown')
            open (34, file = 'DEN/'//name(1:IFN)//'-XCG_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown')
            open (35, file = 'DEN/'//name(1:IFN)//'-XCF_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown')
            
        !--- OUTPUT quasi-particle wave functions
            open (41, file = 'WAV/'//name(1:IFN)//'-QVG_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown')
            open (42, file = 'WAV/'//name(1:IFN)//'-QVF_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown')
            open (43, file = 'WAV/'//name(1:IFN)//'-QUG_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown')
            open (44, file = 'WAV/'//name(1:IFN)//'-QUF_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown')
        else
        !--- OUTPUT canonical single particle information for each step
            open (33, file = 'ITER/LEV/'//name(1:IFN)//'-lev_'//tex%tit(it)//'.'//cii//'.'//pset%txtfor, status = 'unknown')
            open (34, file = 'ITER/DEN/'//name(1:IFN)//'-XCG_'//tex%tit(it)//'.'//cii//'.'//pset%txtfor, status = 'unknown')
            open (35, file = 'ITER/DEN/'//name(1:IFN)//'-XCF_'//tex%tit(it)//'.'//cii//'.'//pset%txtfor, status = 'unknown')
        
        !--- OUTPUT quasi-particle wave functions
            open (41, file = 'ITER/WAV/'//name(1:IFN)//'-QVG_'//tex%tit(it)//'.'//cii//'.'//pset%txtfor, status = 'unknown')
            open (42, file = 'ITER/WAV/'//name(1:IFN)//'-QVF_'//tex%tit(it)//'.'//cii//'.'//pset%txtfor, status = 'unknown')
            open (43, file = 'ITER/WAV/'//name(1:IFN)//'-QUG_'//tex%tit(it)//'.'//cii//'.'//pset%txtfor, status = 'unknown')
            open (44, file = 'ITER/WAV/'//name(1:IFN)//'-QUF_'//tex%tit(it)//'.'//cii//'.'//pset%txtfor, status = 'unknown')
        end if

        write(33, 101) 'State', 'Node', '2*j+1', 'vv', 'EC', 'cdlt', 'r2'
        write(34, 100) 'r', ( tab(i0), i0 = 1, lmx )
        write(35, 100) 'r', ( tab(i0), i0 = 1, lmx )
        
        write(41, 100) 'r', ( lev(it)%tb(i0), i0 = 1, lev(it)%nt )
        write(42, 100) 'r', ( lev(it)%tb(i0), i0 = 1, lev(it)%nt )
        write(43, 100) 'r', ( lev(it)%tb(i0), i0 = 1, lev(it)%nt )
        write(44, 100) 'r', ( lev(it)%tb(i0), i0 = 1, lev(it)%nt )

    !--- OUTPUT the canonical single-particle information at each step as well as the converged one
        do ib = 1, block(it)%nb
        
            do n = 1, block(it)%id(ib)

                i0  = block(it)%ia(ib) + n

                write(33, 102)  lev(it)%tb(i0), n, lev(it)%mu(i0), cwav(i0,it)%vv, cwav(i0,it)%E, cwav(i0,it)%cdlt, cwav(i0,it)%r2
                    
            end do
        end do
                        
        do i = 1, well%npt
            write(34, 103) well%r(i), (xgc(i0, i), i0 = 1, lmx)
            write(35, 103) well%r(i), (xfc(i0, i), i0 = 1, lmx)
        end do

        close(33);          close(34);          close(35)
        
        do i = 1, well%npt            
            write(41, 103) well%r(i), (wav(i0,it)%vg(i), i0 = 1, lev(it)%nt)      
            write(42, 103) well%r(i), (wav(i0,it)%vf(i), i0 = 1, lev(it)%nt)      
            write(43, 103) well%r(i), (wav(i0,it)%ug(i), i0 = 1, lev(it)%nt)      
            write(44, 103) well%r(i), (wav(i0,it)%uf(i), i0 = 1, lev(it)%nt)
        end do
        
        close(41);          close(42);          close(43);          close(44)
    end do

100 format (3x, a, 4x, 300(4x, a8, 4x))
101 format(2x,a, 4x,a, 2x, a, 7x, a, 2(14x, a), 2(9x, a))
102 format(a,2i6, 4E16.6)
103 format(f8.3, 300e16.7)

!--- OUTPUT the densities
    if(lpc) then
    !--- OUTPUT the halo information when the calculation reaches convergence
        open(33, file = 'DEN/'//name(1:IFN)//'-halo.'//pset%txtfor, status = 'unknown')
        
        write(33, 201)  'r', 'xnum(n)', 'xnum(p)', 'xnuo(n)', 'xnuo(p)'
        
        do j = 1, NRX
            i = 10*j + 1
            write(33, 202) well%r(i), Halo(1:2)%xnm(j), Halo(1:2)%xno(j)
        end do
        close(33)

    !--- OUTPUT the local local densities when the calculation reaches convergence
        open(33, file = 'DEN/'//name(1:IFN)//'-den'//'.'//pset%txtfor, status = 'unknown')
    else
    !--- OUTPUT the local local densities at each step
        open(33, file = 'ITER/DEN/'//name(1:IFN)//'-den.'//cii//'.'//pset%txtfor, status = 'unknown')
    end if

    write(33, 201) 'r', 'rhos(n)', 'rhos(p)', 'rhob(n)', 'rhob(p)', 'rhot(n)', 'rhot(p)', 'rhoc'

    do i = 1, well%npt
        write(33, 202) well%r(i), den(1:2)%rs(i), den(1:2)%rv(i), den(1:2)%rt(i), dens%rc(i) 
    end do
    close(33)

201 format(4x,a,10x,a, 6(10x,a))
202 format(f8.3, 9E17.8)
     
    return
    n   = well%npt
!--- Output the local potentials
    if(lpc) then        !--- when the calculation reaches convergence 
        open (14, file = 'LMF/'//name(1:IFN)//'-VS.'//pset%txtfor, status = 'unknown')
    else                !--- at each step
        open (14, file = 'ITER/LMF/'//name(1:IFN)//'-VS.'//cii//'.'//pset%txtfor, status = 'unknown') 
    end if
    write(14,301) 'r(fm)','V+S(n)','V+S(p)','V-S(n)','V-S(p)',' VT(n)', ' VT(p)'
    do i = 1, n
        write(14, 302) well%r(i), dpotl(1:2)%vps(i), dpotl(1:2)%vms(i) + two*pset%amu(1:2)/hbc, dpotl(1:2)%vt(i)
    end do
    close(14)
301 format(2x, a, 7x,6(a, 10x))
302 format(f8.3, 6e16.7)
        
    if(IE.eq.1) return

!--- Output the non-local potentials  
    do it = 1, 2

        do ib = 1, block(it)%nb
                
            if(lpc) then                    !--- Output the converged results
                open (14, file = 'NMF/'//name(1:IFN)//'-XG('//block(it)%tb(ib)//').'//pset%txtfor, status = 'unknown')
                open (15, file = 'NMF/'//name(1:IFN)//'-XF('//block(it)%tb(ib)//').'//pset%txtfor, status = 'unknown') 
                open (16, file = 'NMF/'//name(1:IFN)//'-YG('//block(it)%tb(ib)//').'//pset%txtfor, status = 'unknown')
                open (17, file = 'NMF/'//name(1:IFN)//'-YF('//block(it)%tb(ib)//').'//pset%txtfor, status = 'unknown')
            else                            !--- output each step
                open (14, file = 'ITER/NMF/'//name(1:IFN)//'-XG('//block(it)%tb(ib)//').'//cii//'.'//pset%txtfor, status = 'unknown') 
                open (15, file = 'ITER/NMF/'//name(1:IFN)//'-XF('//block(it)%tb(ib)//').'//cii//'.'//pset%txtfor, status = 'unknown') 
                open (16, file = 'ITER/NMF/'//name(1:IFN)//'-YG('//block(it)%tb(ib)//').'//cii//'.'//pset%txtfor, status = 'unknown')
                open (17, file = 'ITER/NMF/'//name(1:IFN)//'-YF('//block(it)%tb(ib)//').'//cii//'.'//pset%txtfor, status = 'unknown') 
            end if
                
            write(14, 401) well%r(1:n)
            write(15, 401) well%r(1:n)
            write(16, 401) well%r(1:n)
            write(17, 401) well%r(1:n)
                
            do i = 1, n
                write(14, 402) well%r(i), opotl(ib,it)%XG(i, 1:n)
                write(15, 402) well%r(i), opotl(ib,it)%XF(i, 1:n)
                write(16, 402) well%r(i), opotl(ib,it)%YG(i, 1:n)
                write(17, 402) well%r(i), opotl(ib,it)%YF(i, 1:n)
            end do
            close(14);          close(15);          close(16);          close(17)
        end do
            
    end do
401 format(8x,   251e16.7)
402 format(f8.3, 251e16.7)
           
    return

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
End Subroutine Export 
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!


End Module INOUTS
