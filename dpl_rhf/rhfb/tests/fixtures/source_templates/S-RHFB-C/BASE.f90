!*************************************************************************************************!
!
MODULE BASE
!
!*************************************************************************************************!
USE RHFBLIB
USE DEFINE
IMPLICIT NONE

!--- THIS MODULE IS USED TO BUILD A BASIS ON THE WOOD-SAXON POTENTIAL

!--- SIZE OF THE BASIS
!    NBF: NUMBER OF THE POSITIVE ENERGY STATES
!    NBD: NUMBER OF THE NEGATIVE ENERGY STATES
    INTEGER, PARAMETER, PRIVATE :: NB = 2, NBR = NB*(MSD-1) + 1
    INTEGER, PARAMETER :: NBH = NB
    
!--- STRUCTURE FOR THE BASIS
    TYPE BASES
        Integer :: KAP, L, LP, ND, NF, NDF
        DOUBLE PRECISION, DIMENSION(NBS) :: EIG, XWT
        DOUBLE PRECISION, DIMENSION(NBS, NBS) :: XKT
        
    !--- Radial wave functions
        DOUBLE PRECISION, DIMENSION(MSD, NBS) :: XG, XF, DG, DF, SG, SF, WG, WF
    END TYPE BASES

    TYPE (BASES), DIMENSION(NBX, IBX) :: BASIS

!--- POTENTIALS
    DOUBLE PRECISION, DIMENSION(NBR), PRIVATE :: RXM
    INTEGER, PRIVATE :: ITB, KAB, MATB, IEN, NPS
    DOUBLE PRECISION, PRIVATE :: HB

    DOUBLE PRECISION, DIMENSION(NBR, 2), PRIVATE :: VPS, VPSH, VMS, VMSH

!--- PARAMETERS FOR WOODS-SAXON POTENTIAL
    DOUBLE PRECISION, PRIVATE :: V0, AKV
    DOUBLE PRECISION, DIMENSION(IBX), PRIVATE :: VSO, R0V, R0S, AV, AS

    DATA V0/-71.28/, AKV/0.4616/, VSO/11.1175, 8.9698/
    DATA R0V/1.2334,1.2496/, R0S/1.1443, 1.1400/
    DATA AV /0.615, 0.6124/, AS /0.6476, 0.6469/


!--- MATCHING MATRIX
    DOUBLE PRECISION, DIMENSION(2,2), PRIVATE :: GFM
    DOUBLE PRECISION, DIMENSION(NBR), PRIVATE :: SG, SF

!--- WAVE FUNCTIONS
    TYPE WAVEB
        DOUBLE PRECISION, DIMENSION(NBR) :: G, F
    END TYPE WAVEB

!--- ENERGY CUT-OFF
    DOUBLE PRECISION, DIMENSION(IBX) :: TOP, BOT
    
!--- Wilson term
    Type Wilson_term
        Integer :: m
        double precision :: R
    End Type Wilson_term
    Type (Wilson_term) :: XWT
    DATA XWT%R/0.001/, XWT%M/3/

CONTAINS
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
SUBROUTINE WOODS
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
!--- THIS SUBROUTINE IS USED TO THE BASIS POTENTIALS
!

    INTEGER :: I, NPT, IT, ITA, I0
    DOUBLE PRECISION :: RAV, RAS, VP, VLS, ARGV, ARGS, U, W, R, RH, C, EMCC(2)

!--- BUILD UP THE BASIS
    EMCC        = pset%amu/hbc*two
    NPT         = NPS

!--- INITIALIZE THE SELF-ENERGIES WITH WOODS-SAXON BASIS
    DO IT = 1, 2

        ITA = 3 - IT
        RAV = R0V(IT)*nuc%amas**third
        RAS = R0S(IT)*nuc%amas**third
        VP  = V0*(one - AKV*(nuc%npr(IT)-nuc%npr(ITA))/nuc%amas)/hbc
        VLS = VP*VSO(IT)

        DO I = 1, NPT
            R       = RXM(I)
            RH      = R + HALF*HB

            ARGV    = (R - RAV)/AV(IT);                 ARGS    = (RH- RAV)/AV(IT)
            IF(ARGV.LE.65.D0) THEN
                VPS (I, IT) =  VP/(one + DEXP(ARGV));       VPSH(I, IT) =  VP/(one + DEXP(ARGS))
            ELSE
                VPS (I, IT) =  zero;                        VPSH(I, IT) =  zero
            END IF

            ARGV    = (R - RAS)/AS(IT);                 ARGS    = (RH- RAS)/AS(IT)
            IF(ARGV.LE.65.D0) THEN
                VMS (I, IT) = -VLS/(one + DEXP(ARGV));      VMSH(I, IT) = -VLS/(one + DEXP(ARGS))
            ELSE
                VMS (I, IT) =  zero;                        VMSH(I, IT) =  zero
            END IF
            VMS (I, IT) = VMS (I, IT) - EMCC(IT);       VMSH(I, IT) = VMSH(I, IT) - EMCC(IT)

!--- THE COULOMB FIELD         
            IF(IT.EQ.2) THEN
                IF(R.LT.RAV) THEN
                    C   = half*(3./RAV - R**2/RAV**3)
                ELSE
                    C   = one/R
                END IF
                VPS(I, IT)  =  C*NUC%NPR(2)/alphi + VPS(I, IT)
                VMS(I, IT)  =  C*NUC%NPR(2)/alphi + VMS(I, IT)

                IF(RH.LT.RAV) THEN
                    C   = half*(3./RAV - RH**2/RAV**3)
                ELSE
                    C   = one/RH
                END IF
                VPSH(I, IT) =  C*NUC%NPR(2)/alphi + VPSH(I, IT)
                VMSH(I, IT) =  C*NUC%NPR(2)/alphi + VMSH(I, IT)
            END IF
        END DO
    END DO
    
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
END SUBROUTINE WOODS
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
FUNCTION DETB(EIG)
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
    DOUBLE PRECISION, INTENT(IN) :: EIG
    DOUBLE PRECISION :: DETB, EIGFM, H, H2, GMESH, FMESH, ORIGIN, ALPH, BETA

    DOUBLE PRECISION, DIMENSION(4) :: AG, AF
    DOUBLE PRECISION, DIMENSION(2, 2) :: AA
    DOUBLE PRECISION :: X, X1, X2, X4, SG1, SF1, SG2, SF2, U1G, U1F, U2G, U2F
    INTEGER :: I0, KA, I, MAT, JTEM, J, JK, ITEM, NPT, IT

!--- INITIALIZATION OF STATE'S QUANTITIES
    KA      = KAB;              MAT     = MATB;             H       = HB
    H2      = HB*half;          NPT     = NPS;              IT      = ITB

    EIGFM   = EIG/hbc;          SG(1)   = zero;             SF(1)   = zero

!--- DETERMINE THE WAVE FUNCTIONS ON THE ORIGINAL AND INFINITY POSITION
    ORIGIN  = 1.0D-5
    IF(KA.GT.0) THEN
        ALPH    = (two*KA + one)/H*ORIGIN;          BETA    = EIGFM - VMS(2, IT)
    ELSE
        ALPH    = (EIGFM - VPS(2, IT))*ORIGIN;      BETA    = (two*KA - one)/H 
    END IF
    SG(2)   = ORIGIN;                   SF(2)   = ALPH/BETA

    IF(IEN.EQ.1) THEN
        SG(NPT) = zero;                     SF(NPT) = one
    ELSE IF(IEN.EQ.-1) THEN
        SG(NPT) = one;                      SF(NPT) = zero
    END IF
!       SG(NPT) = ZERO;                     SF(NPT) = ONE

!--- INTEGRATE INWARD FROM (NPT - 1) *MESH
    JTEM      = NPT - MAT
    DO J = 1, JTEM
        I   = NPT - J + 1
        X   = RXM(I)
        X1  = KA/ X;                    X2  = KA/(X - H2);                  X4  = KA/(X - H)

        SG1 = SG(I);                                    SF1 = SF(I)
        DO  JK = 1, 4
            GO TO (36,37,37,38) , JK
 36         SG2     = SG1;                                  SF2     = SF1
            U1G     = X1;                                   U1F     = EIGFM - VMS(I, IT)
            U2F     = X1;                                   U2G     = EIGFM - VPS(I, IT)
            GO TO 35

 37         SG2     = SG1 + AG(JK-1);                       SF2     = SF1 + AF(JK-1)
            U1G     = X2;                                   U1F     = EIGFM - VMSH(I-1, IT)
            U2F     = X2;                                   U2G     = EIGFM - VPSH(I-1, IT)
            GO TO 35

 38         SG2     = SG1 + two*AG(3);                      SF2     = SF1 + two*AF(3)
            U1G     = X4;                                   U1F     = EIGFM - VMS(I-1, IT)
            U2F     = X4;                                   U2G     = EIGFM - VPS(I-1, IT)
         
 35         AG(JK)  = - H2*(-U1G*SG2 + U1F*SF2);            AF(JK)  = - H2*( U2F*SF2 - U2G*SG2)
        END DO
        SG2     = (AG(1) + two*(AG(2)+AG(3)) + AG(4))*third
        SF2     = (AF(1) + two*(AF(2)+AF(3)) + AF(4))*third

        SG(I-1) = SG1 + SG2;                            SF(I-1) = SF1 + SF2
    END DO
    GFM(1,1)    = SG(MAT);                          GFM(2,1)    = SF(MAT)

!--- INTEGRATE OUT TO MATCH POINT
    ITEM    = MAT - 1
    DO I  = 2, ITEM
        X   = RXM(I)
        X1  = KA/ X;                    X2  = KA/(X + H2);                  X4  = KA/(X + H)
      
        SG1 = SG(I);                                    SF1 = SF(I)
     
        DO  JK = 1, 4
            GO TO (46,47,47,48) , JK
 46         SG2     = SG1;                                  SF2     = SF1
            U1G     = X1;                                   U1F     = EIGFM - VMS(I, IT)
            U2F     = X1;                                   U2G     = EIGFM - VPS(I, IT)
            GO TO 45

 47         SG2     = SG1 + AG(JK-1);                       SF2     = SF1 + AF(JK-1)
            U1G     = X2;                                   U1F     = EIGFM - VMSH(I, IT)
            U2F     = X2;                                   U2G     = EIGFM - VPSH(I, IT)
            GO TO 45

 48         SG2     = SG1 + two*AG(3);                      SF2     = SF1 + two*AF(3)
            U1G     = X4;                                   U1F     = EIGFM - VMS(I+1, IT)
            U2F     = X4;                                   U2G     = EIGFM - VPS(I+1, IT)

 45         AG(JK)  = H2*(-U1G*SG2 + U1F*SF2);              AF(JK)  = H2*( U2F*SF2 - U2G*SG2)
        END DO
        SG2     = (AG(1) + two*(AG(2)+AG(3)) + AG(4))*third
        SF2     = (AF(1) + two*(AF(2)+AF(3)) + AF(4))*third

        SG(I+1) = SG1 + SG2;                            SF(I+1) = SF1 + SF2
    END DO ! END ITERATION OUTWARD TO MATCH POINT.
    GFM(1,2)    = SG(MAT);                          GFM(2,2)    = SF(MAT)

!--- RETURN THE VALUE OF MATCHING DETERMINANT
    AA      = GFM
    DETB    = DET(2, 2, AA)

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
END FUNCTION DETB
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
SUBROUTINE MATCH(WAVE, NC, LPRX)
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
    TYPE (WAVEB), INTENT(OUT) :: WAVE
    INTEGER, INTENT(OUT) :: NC
    LOGICAL, INTENT(OUT) :: LPRX

    INTEGER :: I1, I2, INODE, NPT, MAT, I, J
    DOUBLE PRECISION :: S, S1, XSTEP

    DOUBLE PRECISION, DIMENSION(2, 2) :: AA, BB
    DOUBLE PRECISION, DIMENSION(2) :: EEM, AE, ZZ, G
    DOUBLE PRECISION, DIMENSION(NBR) :: FUN

    XSTEP   = HB;                   NPT     = NPS;                  MAT     = MATB

!--- NORMALIZE THE MATCHING MATRIX AND RESCALE THE WAVE FUNCTION
    DO I = 1, 2
        S   = zero
        DO J = 1, 2
            S           = S + GFM(J, I)**2
        END DO
        S   = one/DSQRT(S);                 GFM(1:2, I) = S*GFM(1:2, I)

        IF(I.EQ.1) THEN
            I1          = MAT + 1;          I2          = NPT
        ELSE
            I1          = 2;                I2          = MAT
        END IF
        SG(I1:I2)   = S*SG(I1:I2);          SF(I1:I2)   = S*SF(I1:I2)
    END DO

!--- CALCULATE MT*M
    DO I = 1, 2
        DO J = 1, 2
            S   = zero
            DO I1 = 1, 2
                S   = S + GFM(I1, I)*GFM(I1, J)
            END DO
            AA(I, J)    = S
        END DO
    END DO

!--- THE EIGENVALUES AND EIGENVECTORS FOR AA
    CALL SDIAG(2, 2, AA, EEM, AA, ZZ, 1)

!--- RESCALE THE WAVE FUNCTION WITH THE EIGENVECTORS OF AA
    DO I = MAT + 1 , NPT
        WAVE%G(I)   = -SG(I)*AA(1, 1);      WAVE%F(I)   = -SF(I)*AA(1, 1)
    END DO
    DO I = 1, MAT
        WAVE%G(I)   =  SG(I)*AA(2, 1);      WAVE%F(I)   =  SF(I)*AA(2, 1)
    END DO

!--- NORMALIZATION
    FUN(1:NPT)      = WAVE%G(1:NPT)**2 + WAVE%F(1:NPT)**2

    CALL simps(FUN, NPT, XSTEP, S);             S               = one/DSQRT(S)
    WAVE%G(1:NPT)   = WAVE%G(1:NPT)*S;          WAVE%F(1:NPT)   = WAVE%F(1:NPT)*S


!--- NODE CHECK
    INODE           = 0;                        NC              = 0
    LPRX            = .FALSE.
    DO I = 3, NPT-1
        IF(IEN.EQ.1)    S = WAVE%G(I)*WAVE%G(I-1)
        IF(IEN.EQ.-1)   S = WAVE%F(I)*WAVE%F(I-1)
        IF(S .LT. zero) THEN
            INODE     = INODE + 1 
            IF(ABS(I-MAT).LE.1) LPRX    = .TRUE.
        END IF
    END DO
    NC              = INODE 

    RETURN
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
END SUBROUTINE MATCH
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
SUBROUTINE DBASE
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!

    INTEGER :: IB, N, IT, I, NC, KA, LA, LB, IA, LP, ND, NF
    DOUBLE PRECISION :: EIG, F1, F2, EIG1, EIG2, ESTEP, EPOINT, ES, FF, REL, ESI
    TYPE (WAVEB) :: WAVE, WAV0
    
    DOUBLE PRECISION :: EBP = -70.D0, ETP = 700.0D0, ETN = -1160.D0, EBN = -1878.0D0
    DOUBLE PRECISION, PARAMETER :: TOL = 1.D-130
    DOUBLE PRECISION, DIMENSION(NBR) :: FUN, DG, DF
    DOUBLE PRECISION, DIMENSION(MSD) :: FUC
    LOGICAL :: LPRX
    DATA ES/5.0D0/, ESI/0.1D-3/

    ESTEP   = ES
    NPS     = NB*(well%npt - 1) + 1
    HB      = well%h/NB

    DO I = 1, NPS
        RXM(I)  = (I-1)*HB
    END DO
    RXM(1)  = HB*1.D-2

!--- DETERMINE THE POTENTIALS OF THE BASIS
    CALL WOODS
 
!--- LOOP OVER NUCLEONS
    TOP = 0.0D0;        BOT = 0.0D0
    DO IT = 1, 2
        EBP     = VPS(2, IT)*hbc;       ETP     = 1800.0D0
        ETN     = VMS(2, IT)*hbc;       EBN     = ETN - pset%amu(IT)*two
        ITB     = IT
!--- LOOP OVER THE SINGLE PARTICLE LEVELS
        DO KA = 1, NBX

        !--- KAPPA
            KAB = KA
            LA  = KAB
            IF(KA.GT.KMX(2)) KAB   = KMX(2) - KA
            IF(KA.GT.KMX(2)) LA    = KA - KMX(2) - 1
            
            LP  = 2*iabs(KAB) - 1 - LA;             BASIS(KA,IT)%LP = LP;           BASIS(KA,IT)%L  = LA

        !--- DETERMINE THE STATES IN FERMI SEA
            IEN = 1
            EIG = EBP + .5D0;                   EPOINT  = EIG
            DO N = 1, NBF

                IB      = NBD + N

                MATB    = 1.1*nuc%amas**third/HB + 1

            !--- FIND THE WAVEFUNCTIONS AND EIGENVALUES FOR THE I'TH STATE
 21             EIG     = EPOINT

                EIG1    = EIG +  ESI;                   EIG2    = EIG1 + ESTEP
                F1      = DETB(EIG1);                   F2      = DETB(EIG2)
                DO WHILE(F1*F2.GT.zero.AND.EIG2.LT.ETP) 
                    EIG1    = EIG2;                         EIG2    = EIG1 + ESTEP
                    F1      = F2;                           F2      = DETB(EIG2)
                END DO

                IF(EIG2.GT.ETP) THEN
                    ESTEP   = ESTEP*half
                    IF(ESTEP.LT.ESI) STOP ' EXCEED THE TOP OF THE SERACHING REGION!'
                    GO TO 21
                END IF

                EIG     = RTBRENT(DETB, EIG1, EIG2, F1, F2, TOL)
                FF      = DETB(EIG)
                CALL MATCH(WAVE, NC, LPRX)
!                WRITE(*,'(2I3,2I3, 7F12.6)')IT, KAB, N-1, NC, EIG, FF, EIG1, F1, EIG2, F2, ESTEP

            !--- NODE CHECK
                IF(NC.NE.N-1) THEN
                    ESTEP   = ESTEP*half
                    IF(ESTEP.LT.ESI)  STOP ' SEARCHING STEP IS TOO SMALL!'
                    GO TO 21
                END IF

                IF(LPRX) THEN
                    MATB    = MATB + 0.2/HB 
!                   WRITE(*,*)  'THE MATCH POINT IS CHANGED TO', MATB 
                    IF((NPS - MATB) .LT. (1.0/HB) ) STOP ' IN DBASE: MATB IS TOO BIG!'
                    GO TO 21
                END IF

            !--- RESET THE SERACHING STEP AND STARTING POINT
                ESTEP                   =  ES
                EPOINT                  =  EIG

            !--- TYPE (BASES), DIMENSION(NBX, 2) :: BASIS
            !--- Determine the cuttoff in negative energy states
                IF( EIG .le. XCT(1) ) BASIS(KA, IT)%NF   = N
                
                BASIS(KA, IT)%EIG(IB)   =  EIG
                IF(LA.eq.0) WAVE%G(1)   = 3.0d0*( WAVE%G(2) - WAVE%G(3) ) + WAVE%G(4)                        
                IF(LP.eq.0) WAVE%F(1)   = 3.0d0*( WAVE%F(2) - WAVE%F(3) ) + WAVE%F(4)

                CALL  deriv5(WAVE%G, DG, NPS, HB);       CALL  deriv5(WAVE%F, DF, NPS, HB)
                CALL sderiv5(WAVE%G, SG, NPS, HB);       CALL sderiv5(WAVE%F, SF, NPS, HB)
                DO I = 1, well%npt
                    BASIS(KA, IT)%XG(I,IB)  =  WAVE%G( NB*(I-1)+1 )
                    BASIS(KA, IT)%XF(I,IB)  =  WAVE%F( NB*(I-1)+1 )
                    BASIS(KA, IT)%DG(I,IB)  =      DG( NB*(I-1)+1 )
                    BASIS(KA, IT)%DF(I,IB)  =      DF( NB*(I-1)+1 )
                    BASIS(KA, IT)%SG(I,IB)  =      SG( NB*(I-1)+1 )
                    BASIS(KA, IT)%SF(I,IB)  =      SF( NB*(I-1)+1 )
                END DO
                CALL Xderiv5(WAVE%G, DG, NPS, HB, XWT%m);       CALL Xderiv5(WAVE%F, DF, NPS, HB, XWT%m)
                CALL Xderiv5(    DG, SG, NPS, HB, XWT%m);       CALL Xderiv5(    DF, SF, NPS, HB, XWT%m)
                DO I = 1, well%npt
                    BASIS(KA, IT)%WG(I,IB)  =      SG( NB*(I-1)+1 )
                    BASIS(KA, IT)%WF(I,IB)  =      SF( NB*(I-1)+1 )
                END DO
 !               WRITE(8, '(4I3, 2F16.6)') IB, KAB, LA, IT, BASIS(IB, KA, IT)%EIG, BASIS(IB, KA, IT)%EKT
            END DO
            
            TOP(IT) = MAX(TOP(IT), DABS( BASIS(KA,IT)%EIG(NBS) ) )

        !--- DETERMINE THE STATES IN DIRAC SEA
            IEN     = -1
            EIG     = ETN - .5D0;                  EPOINT  = EIG
            DO N = 1, NBD
                IB      = N
                MATB    = 1.0*nuc%amas**third/HB + 1

            !--- FIND THE WAVEFUNCTIONS AND EIGENVALUES FOR THE I'TH STATE
 31             EIG     = EPOINT

                EIG1    = EIG -  ESI;                   EIG2    = EIG1 - ESTEP
                F1      = DETB(EIG1);                   F2      = DETB(EIG2)
                DO WHILE(F1*F2.GT.zero.AND.EIG2.GT.EBN) 
                    EIG1    = EIG2;                         EIG2    = EIG1 - ESTEP
                    F1      = F2;                           F2      = DETB(EIG2)
                END DO

                IF(EIG2.LT.EBN) THEN
                    ESTEP   = ESTEP*half
                    IF(ESTEP.LT.ESI) STOP ' EXCEED THE RANGE OF THE SERACHING REGION!'
                    GO TO 31
                END IF

                EIG     = RTBRENT(DETB, EIG1, EIG2, F1, F2, TOL)
                FF      = DETB(EIG)
                CALL MATCH(WAVE, NC, LPRX)
!                WRITE(*,'(2I3,2I3, 7F12.6)')IT, KAB, N-1, NC, EIG, FF, EIG1, F1, EIG2, F2, ESTEP

            !--- NODE CHECK
                IF(NC.NE.N-1) THEN
                    ESTEP   = ESTEP*half
                    IF(ESTEP.LT.ESI)  STOP ' SEARCHING STEP IS TOO SMALL!'
                    GO TO 31
                END IF

                IF(LPRX) THEN
                    MATB    = MATB + 0.2/HB
!                   WRITE(*,*)  'THE MATCH POINT IS CHANGED TO', MATB                  
                    IF((NPS - MATB) .LT. (1.0/HB) ) STOP ' IN DBASE: MATB IS TOO BIG!'
                    GO TO 31
                END IF

            !--- RESET THE SERACHING STEP AND STARTING POINT
                ESTEP                   =  ES
                EPOINT                  =  EIG

            !    TYPE (BASES), DIMENSION(NBS, NBX, 2) :: BASIS
                if( (EIG+two*pset%amu(it)) .ge. XCT(2) )  BASIS(KA, IT)%ND = N
                BASIS(KA, IT)%EIG(IB)   =  EIG
                IF(LA.eq.0) WAVE%G(1)   = 3.0d0*( WAVE%G(2) - WAVE%G(3) ) + WAVE%G(4)                        
                IF(LP.eq.0) WAVE%F(1)   = 3.0d0*( WAVE%F(2) - WAVE%F(3) ) + WAVE%F(4)

                CALL  deriv5(WAVE%G, DG, NPS, HB);       CALL  deriv5(WAVE%F, DF, NPS, HB)
                CALL sderiv5(WAVE%G, SG, NPS, HB);       CALL sderiv5(WAVE%F, SF, NPS, HB)
                DO I = 1, well%npt
                    BASIS(KA, IT)%XG(I,IB)  =  WAVE%G( NB*(I-1)+1 )
                    BASIS(KA, IT)%XF(I,IB)  =  WAVE%F( NB*(I-1)+1 )
                    BASIS(KA, IT)%DG(I,IB)  =      DG( NB*(I-1)+1 )
                    BASIS(KA, IT)%DF(I,IB)  =      DF( NB*(I-1)+1 )
                    BASIS(KA, IT)%SG(I,IB)  =      SG( NB*(I-1)+1 )
                    BASIS(KA, IT)%SF(I,IB)  =      SF( NB*(I-1)+1 )
                END DO
                CALL Xderiv5(WAVE%G, DG, NPS, HB, XWT%m);       CALL Xderiv5(WAVE%F, DF, NPS, HB, XWT%m)
                CALL Xderiv5(    DG, SG, NPS, HB, XWT%m);       CALL Xderiv5(    DF, SF, NPS, HB, XWT%m)
                DO I = 1, well%npt
                    BASIS(KA, IT)%WG(I,IB)  =      SG( NB*(I-1)+1 )
                    BASIS(KA, IT)%WF(I,IB)  =      SF( NB*(I-1)+1 )
                END DO
!                WRITE(8, '(4I3, 2F16.6)') IB, KAB, LA, IT, BASIS(IB, KA, IT)%EIG, BASIS(IB, KA, IT)%EKT
            END DO

            BOT(IT) = MIN(BOT(IT), BASIS(KA,IT)%EIG(NBD))

            !--- normalization check
                
        !--- NDF: total number of states involved in the expansions of Dirac spinors
            BASIS(KA,IT)%NDF    = BASIS(KA,IT)%ND + BASIS(KA,IT)%NF
            DO IA = 1, NBS
                FUC(:)  = BASIS(KA,IT)%XG(:,IA)**2 + BASIS(KA,IT)%XF(:,IA)**2
                CALL simps(FUC, WELL%NPT, WELL%H, REL);         REL = ONE/DSQRT(REL)
                BASIS(KA,IT)%XG(:,IA)   = BASIS(KA,IT)%XG(:,IA)*REL
                BASIS(KA,IT)%XF(:,IA)   = BASIS(KA,IT)%XF(:,IA)*REL
                    
                BASIS(KA,IT)%DG(:,IA)   = BASIS(KA,IT)%DG(:,IA)*REL
                BASIS(KA,IT)%DF(:,IA)   = BASIS(KA,IT)%DF(:,IA)*REL
                    
                BASIS(KA,IT)%SG(:,IA)   = BASIS(KA,IT)%SG(:,IA)*REL
                BASIS(KA,IT)%SF(:,IA)   = BASIS(KA,IT)%SF(:,IA)*REL
                    
                BASIS(KA,IT)%WG(:,IA)   = BASIS(KA,IT)%WG(:,IA)*REL
                BASIS(KA,IT)%WF(:,IA)   = BASIS(KA,IT)%WF(:,IA)*REL
            END DO
            ND  = BASIS(KA,IT)%ND;          NF  = BASIS(KA,IT)%NF
            WRITE( *, '(3I4, 2E20.10)')  KAB, ND, NF, BASIS(KA, IT)%EIG(ND)+two*pset%amu(it), BASIS(KA, IT)%EIG(NBD+NF)

        !--- CALCULATE THE KINETIC ENERGY
            DO IA = 1, NBS
                DO IB = 1, NBS

                    FUC(:)  = BASIS(KA,IT)%XG(:,IA)*( -BASIS(KA, IT)%DF(:,IB) + KAB/WELL%R(:)*BASIS(KA, IT)%XF(:,IB) ) + &
                            & BASIS(KA,IT)%XF(:,IA)*(  BASIS(KA, IT)%DG(:,IB) + KAB/WELL%R(:)*BASIS(KA, IT)%XG(:,IB) )

                    CALL simps(FUC, WELL%NPT, WELL%H, REL)
                    BASIS(KA, IT)%XKT(IA, IB)   = REL
                END DO
                
            !--- Wilson terms
                FUC(:)  = BASIS(KA,IT)%XG(:,IA)*BASIS(KA,IT)%WG(:,IA) - BASIS(KA,IT)%XF(:,IA)*BASIS(KA,IT)%WF(:,IA)
                CALL simps(FUC, WELL%NPT, WELL%H, REL)
                BASIS(KA,IT)%XWT(IA)    = REL*(-1.0)**XWT%m*WELL%H**(XWT%M*2-1)
            END DO

        END DO
    END DO
    RETURN
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
!
END SUBROUTINE DBASE
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!

!*************************************************************************************************!
!
END MODULE BASE
!
!*************************************************************************************************!

