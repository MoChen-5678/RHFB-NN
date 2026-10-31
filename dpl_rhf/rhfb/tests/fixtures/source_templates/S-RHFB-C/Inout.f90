!*************************************************************************************************!
!
Module INOUTS
!
!*************************************************************************************************!
Use Define
Use RHFBlib
Use Densities
Use BASE, only: BASIS, NBH, TOP, BOT
Use DIRACE
Use IMEbase, only: EMX
Use Bogoliubov, only: cwav
Implicit None

!--- for output
    double precision, dimension(NTX,MSD), private :: xde, xgc, xfc 
    character*8, dimension(NTX), private :: tab
    
!--- for isotopic and isotonic calculations
    Integer :: NSTEP, Nnum, IDT

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

    type (effective), dimension(0:20) :: PARA !--geng2020.10.1    


!--- Effective interaction and pairing force
    PARA(0)     = PKA1
    PARA(1)     = PKO1;         PARA(2)     = PKO2;         PARA(3)     = PKO3

    PARA(4)     = DDME1;        PARA(5)     = DDME2;        PARA(6)     = PKDD;         PARA(7)     = TW99

    PARA(8)     = NL1;          PARA(9)     = NL2;          PARA(10)    = NL3;          PARA(11)    = NLZ
    PARA(12)    = NLZ2;         PARA(13)    = NLSH;         PARA(14)    = NLSV;         PARA(15)    = TM1
    PARA(16)    = TM2;          PARA(17)    = PK1;          PARA(18)    = PK1R
    PARA(19)    = DDLZ1;        PARA(20)    = DDLZ2     !--geng2020.10.1    

    
!--- Open the input file
    if (lin.ne.0) open(lin,file='rhf.dat',status='old')

!-- Nucleus under consideration
    read(lin,'(2a1,6i4)') nuc%nucnam, nuc%npr(0:IBX), IDT, ipa, NSTEP
    nuc%amas    = nuc%npr(0)
    Call NUCLEI(1, nuc%npr(2), nuc%nucnam)

!--- For isotonic and isotopic evolution
    if ( IDT.gt.0 .and. IDT.le.IBX ) then
        Nnum    = abs( (nuc%npr(IDT) - ipa)/NSTEP )
    else
        Nnum    = 0
        IDT     = 1
    end if

!--- Index of the output files
    read(lin,100) l6

!--- Index of the output files
    read(lin,100) ipa
    pset    = PARA(ipa)
    IE      = pset%IE

!--- Coupling constants for rho-tensor couplings !--geng2020.10.1    
    if(ipa.eq.0) then
        pset%grtn   = pset%grho*pset%grtn/(pset%amu(1)/hbc + pset%amu(2)/hbc)
    else 
        pset%grtn   = pset%grtn/(pset%amu(1)/hbc + pset%amu(2)/hbc)
        pset%gotn   = pset%gotn/(pset%amu(1)/hbc + pset%amu(2)/hbc)
    end if

!-- Parameters of mesh information: update one
    well%npt    = MSD    
    read(lin,101) well%h
    well%rmax   = well%h*(MSD - 1)

!--- radial mesh points
    do ipa  = 1, well%npt
        well%r(ipa) = (ipa - 1)*well%h
    end do
    well%r(1)   = well%h*1.0d-2

!-- Parameter for the iteration:
    read(lin,100) maxi
    read(lin,100) nmix
    read(lin,101) xmix(1);      xmix(2) = xmix(1)
    xmix0 = xmix

!-- Initialization of wavefunctions:
    read(lin,100) inin
    
!--- Input the pairing force:
    read(lin,100) LFG
    IF(LFG.eq.1) then
        GPF = D1S           ! D1N, D1, D1M
        read(lin, 101) GPF%vg
    else if(LFG.eq.0) then
        read(lin, 101) delPF%V0
        delPF%rho0  = pset%rvs
    else
        Stop 'Input of LFG is incorrect!!'
    end if

    read(lin, '(8x, f6.1, f8.1)')  XCT          !--- Energy Cutoff

    read(lin, '(8x, f6.1, f8.1)')  XPT          !--- Pairing windows

!--- If drop the tensor force components: iwt = 0
    read(lin,100) iwt

!--- If drop the tensor force components: iwt = 0
    read(lin,100) ime
    if( ime.eq.2 ) then
        read(lin,101) EMX
    else
        read(lin,  *)
    end if

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
    
100 format(8x,i10)
101 format(8x,f10.3)
    close(lin)
    
    return

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
End Subroutine reader
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
Subroutine output
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

    double precision :: csig, come, crho
    double precision :: pwi0, hom
    INTEGER :: IT

!--- OUTPUT file
    if( nuc%npr(0).lt.10 ) then
        if(nuc%nucnam(1).eq.' ') then
            write(name,'(a1,i1)') nuc%nucnam(2), nuc%npr(0)
            IFN = 2
        else
            write(name,'(2a1,i1)') nuc%nucnam, nuc%npr(0)
            IFN = 3
        end if
    else if( nuc%npr(0).lt.100 ) then
        if(nuc%nucnam(1).eq.' ') then
            write(name,'(a1,i2)') nuc%nucnam(2), nuc%npr(0)
            IFN = 3
        else
            write(name,'(2a1,i2)') nuc%nucnam, nuc%npr(0)
            IFN = 4
        end if
    else 
        if(nuc%nucnam(1).eq.' ') then
            write(name,'(a1,i3)') nuc%nucnam(2), nuc%npr(0)
            IFN = 4
        else
            write(name,'(2a1,i3)') nuc%nucnam, nuc%npr(0)
            IFN = 5
        end if
    end if

    if (l6.ne.6)  open(l6,file=name(1:IFN)//'.'//pset%txtfor,status='unknown')
    write(l6,*) ' *************** BEGIN READER ***************************'
    write(l6,201) ' Box-size  Rmax and r0 :       ', well%rmax, r0
    write(l6,201) ' Step-size:                    ', well%h
    write(l6,200) ' Number of Mesh-points:        ', well%npt
    write(l6,200) ' Maximal numb. of iterations:  ', maxi
    write(l6,201) ' Mixing parameter:             ', xmix
    write(l6,200) ' inin: initial wavefunctions:  ', inin
    write(l6,200) ' levin: initialize levels:     ', levin
    write(l6,'(a,20x,2a1,3i4)') &
                & ' Nucleus:                      ', nuc%nucnam, nuc%npr

    write(l6,*)
    If(LFG.eq.1) then
        write(l6,'(a, 5x, a)') &
                & ' The Gogny Pairing Force:      ', GPF%txtgog
        write(l6,'(a, 5x, f12.5)') &
                & ' Gogny force factor:           ', GPF%vg
    else
        write(l6,'(a, 5x, 2f12.6)') &
                & ' The Delta Pairing Force:      ', delPF%V0, delPF%rho0
    end if
    if(itbl(1)%nb.ge.1) write(l6,'(a,5i5)') &
                & ' Blocking n block - Probability', itbl(1)%kp(1:itbl(1)%nb)
    if(itbl(2)%nb.ge.1) write(l6,'(a,5i5)') &
                & ' Blocking p block - Probability', itbl(2)%kp(1:itbl(2)%nb)

    write(l6,201) 'Energy cuttoff in DWS base     ', XCT
    write(l6,201) 'Energy cuttoff in Pairings     ', XPT

    If(iwt.eq.0) then
        write(l6, '(a, i4)') &
                & ' The tensor force are dropped: ', iwt
    end if
    
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


!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
End Subroutine output
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

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
    integer :: it, ka, i0, i, ib, nn
    data tit/'N', 'P'/

!    if(lpr) write(l6,*) ' ****** BEGIN INOUT ********************************'

!-- reading of meson fields from tape:
    if (is.eq.1) then
        if(inin.ne.0) return
        
        do it = 1, 2
            open(21, file = 'ABH/'//name(1:IFN)//'.AH-'//tit(it)//'.'//pset%txtfor, status = 'old')
            open(22, file = 'ABH/'//name(1:IFN)//'.BH-'//tit(it)//'.'//pset%txtfor, status = 'old')

            read(22, '(E22.12)') fermi%ala(it)
            
            do ib = 1, cell(it)%nb
                read(21, *)  
                read(21, 100) STM(ib,it)%AH
                read(21,*)
                read(22, *)  
                read(22, 100) STM(ib,it)%AB
                read(22,*) 
            end do
            close(21);      close(22)
        end do
100     format(20E22.12)

    else
!-- writing of potentials to tape:
        Call SYSTEM("mkdir ABH")
        do it = 1, 2
            open(21, file = 'ABH/'//name(1:IFN)//'.AH-'//tit(it)//'.'//pset%txtfor, status = 'unknown')
            open(22, file = 'ABH/'//name(1:IFN)//'.BH-'//tit(it)//'.'//pset%txtfor, status = 'unknown')

            write(22, '(E22.12)') fermi%ala(it)
            do ib = 1, cell(it)%nb
                write(21, *) cell(it)%tb(ib)
                write(21, 100) STM(ib,it)%AH
                write(21,*)
                write(22, *) cell(it)%tb(ib)
                write(22, 100) STM(ib,it)%AB
                write(22,*) 
            end do
            close(21);      close(22)
        end do
    endif

!    If(lpr) write(l6,*) ' ****** END INOUT **********************************'

    return

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
End Subroutine inout
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

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
    Call SYSTEM("mkdir DEN")
    Call SYSTEM("mkdir LEV")
    
!--- OUTPUT the canonical single-particle information
    do it = 1, 2

    !--- Open the file for the output of the canonical single particle levels

    !--- OUTPUT the probability densities with substantial contribution
        open (33, file = 'DEN/'//name(1:IFN)//'-DEN_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown')
        write(33, 100) 'r', ( tab(i0), i0 = 1, lmx )
        do i = 1, well%npt
            write(33, 103) well%r(i), (xde(i0, i), i0 = 1, lmx)
        end do
        close(33)
            
    !--- OUTPUT the single-particle information for pairing collapse, and the quasi-particle energy 
        open(34, file = 'LEV/'//name(1:IFN)//'.levQ_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown') 
        write(34, 101) 'State', 'Node', '2*j+1', 'vv', 'EQ'
        do ib = 1, cell(it)%nb
            do n = 1, cell(it)%id(ib)
                i0 = cell(it)%ia(ib) + n
                write(34,102) lev(it)%tb(i0), lev(it)%nk(i0), lev(it)%mu(i0), lev(it)%vv(i0), lev(it)%ee(i0)
            end do
        end do
        close(34)
            
    !--- OUTPUT canonical single particle information
        open (33, file = 'LEV/'//name(1:IFN)//'-lev_'//tex%tit(it)//'.'//pset%txtfor, status = 'unknown')
        write(33, 101) 'State', 'Node', '2*j+1', 'vv', 'EC', 'del', 'EC', 'del', 'r2', '2r', 'rd'
        do ib = 1, cell(it)%nb
        
            do n = 1, cwav(ib,it)%nd
                i0  = cell(it)%ia(ib) + n
                write(33, 102)  lev(it)%tb(i0), n, lev(it)%mu(i0), cwav(ib,it)%vv(n), cwav(ib,it)%ee(n), cwav(ib,it)%de(n),  &
                        &                                          cwav(ib,it)%ec(n), cwav(ib,it)%dc(n), cwav(ib,it)%r2(n), cwav(ib,it)%r(n), cwav(ib,it)%dr(n)
                    
            end do
        end do      
        close(33)
        
    end do

100 format (3x, a, 4x, 300(4x, a8, 4x))
101 format(2x,a, 4x,a, 2x, a, 7x, a, 2(14x, a), 2(14x, a), 2(9x, a), 2(9x, a,4x))
102 format(a,2i6, 8E16.6)
103 format(f8.3, 300e16.7)

!--- OUTPUT the densities

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
    write(33, 201) 'r', 'rhos(n)', 'rhos(p)', 'rhob(n)', 'rhob(p)', 'rhot(n)', 'rhot(p)', 'rhoc'
    do i = 1, well%npt
        write(33, 202) well%r(i), den(1:2)%rs(i), den(1:2)%rv(i), den(1:2)%rt(i), dens%rc(i) 
    end do
    close(33)

201 format(4x,a,10x,a, 6(10x,a))
202 format(f8.3, 9E17.8)
     
!--- OUTPUT the canonical wave functions
    Call SYSTEM("mkdir WAC")
    do it = 1, IBX
        open(33, file = 'WAC/'//name(1:IFN)//'.WAV_G-'//tex%tit(it)//'.'//pset%txtfor, status='unknown')
        open(34, file = 'WAC/'//name(1:IFN)//'.WAV_F-'//tex%tit(it)//'.'//pset%txtfor, status='unknown')
        do ib = 1, cell(it)%nb
            n   = cell(it)%ia(ib)
            write(33, '(a)') cell(it)%tb(ib)
            write(34, '(a)') cell(it)%tb(ib)

            write(33, 301) '   ee  ', ( cwav(ib,it)%ee(i0), i0 = 1, cwav(ib,it)%nd )
            write(34, 301) '   ee  ', ( cwav(ib,it)%ee(i0), i0 = 1, cwav(ib,it)%nd )

            write(33, 301) '   vv  ', ( cwav(ib,it)%vv(i0), i0 = 1, cwav(ib,it)%nd )
            write(34, 301) '   vv  ', ( cwav(ib,it)%vv(i0), i0 = 1, cwav(ib,it)%nd )

            write(33, 302) '   nc  ', ( cwav(ib,it)%nc(i0), i0 = 1, cwav(ib,it)%nd )
            write(34, 302) '   nf  ', ( cwav(ib,it)%nf(i0), i0 = 1, cwav(ib,it)%nd )

            write(33, 303) '   r   ', ( lev(it)%tb(i0+n), i0 = 1, cwav(ib,it)%nd )
            write(34, 303) '   r   ', ( lev(it)%tb(i0+n), i0 = 1, cwav(ib,it)%nd )
            do i = 1, well%npt
                write(33, 304) well%r(i), ( cwav(ib,it)%XG(i,i0), i0 = 1, cwav(ib,it)%nd )
                write(34, 304) well%r(i), ( cwav(ib,it)%XF(i,i0), i0 = 1, cwav(ib,it)%nd )
            end do 
            write(33,*)
            write(34,*)
        end do
        close(33);  close(34)
    end do
301 format(a, 40e14.6)
302 format(a, 6x, 40(i2, 12x))
303 format(a, 3x, 40(a, 6x))
304 format(f7.2, 40e14.6)
    return


!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
End Subroutine Export 
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!
!*************************************************************************************************!
!
End Module INOUTS
!
!*************************************************************************************************!
