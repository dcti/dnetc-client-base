
        SECTION rc5core_020_030,CODE
	OPT	O+,W-

	; Copyright distributed.net 1997-2003 - All Rights Reserved
	; For use in distributed.net projects only.
	; Any other distribution or use of this source violates copyright.

        ; $VER: MC68020/030 re-entrant RC5 core 19-Dec-2002
	; $Id: r72-020_030-mh.s,v 1.1.2.1 2003/01/07 15:22:21 oliver Exp $

        ;
        ; MC680x0 RC5 key checking function
        ; for distributed.net RC5-72 clients.
        ;
        ; Dual key, rolled loops, re-entrant
        ;
        ; Written by John Girvin <girv@girvnet.org.uk>
        ; Adapted to 72-bit by Malcolm Howell <coreblimey@rottingscorpion.com>
        ;
        ; TODO: further optimisation probably possible
        ;

;--------------------

        INCLUDE r72-0x0-common-mh.i

        XREF    _rc5_check64

        XDEF    _rc5_72_unit_func_020_030_mh_2

;--------------------

        ;
        ; MC680x0 RC5 key checking function
        ; Dual key, rolled loops, re-entrant
        ;
        ; Entry: a0=rc5unitwork structure:
        ;        0(a0) = plain.hi  - plaintext
        ;        4(a0) = plain.lo
        ;        8(a0) = cypher.hi - cyphertext
        ;       12(a0) = cypher.lo
        ;       16(a0) = L0.hi     - key
        ;       20(a0) = L0.mid
        ;       24(a0) = L0.lo
        ;        a1=pointer to number of keys to check
        ;
        ;       (see ccoreio.h for detailed interface description)
        ;

        CNOP    0,8
_rc5_72_unit_func_020_030_mh_2:

        movem.l d2-7/a2-6,-(a7) ;P  [11]

        ;Note that Sa[25] is never written to memory, so reserved storage
        ;is 4 bytes smaller than one might expect
        lea     -204(a7),a7     ;p  [ ] save space for Sx[] storage
        move.l  #$2B4C3474,d1   ;   [0] d1=P+25Q
        move.l  (a1),d0         ; Get iteration count
        move.l  #Q,d2   ;p  [ ] d2=Q
        moveq   #26-4-1,d3      ;   [0] d3=loop counter
.ruf_initp3q_loop:
        move.l  d1,-(a7)        ;p  [ ] initialise P+nQ lookup
        sub.l   d2,d1
        dbf     d3,.ruf_initp3q_loop

        lea     -RUFV_SIZE(a7),a7

        lsr.l   #1,d0           ; Initial loop counter (correct instruction?)
        move.l  a1,RUFV_ILC(a7) ; Save initial loop counter pointer
        addq.l  #1,d0           ; Fiddle loop counter
        move.l  #P0QR3,a5       ; a5=handy constant
        move.l  d0,RUFV_LC(a7)  ; Set loop counter

        lea     RUFV_L1P2QR(a7),a1     ; a1=vars for loop
        lea     RUFV_SIZE+88+12(a7),a3 ; a3=&Sb[03]
        lea     104(a3),a2             ; a2=&Sa[03]

        bra     .ruf_l0chg

        RUF_ALIGN       d1,_rc5_72_unit_func_020_030_mh_2 ;Align to 8 bytes and pOEP

        ; Upon reaching/jumping to mainloop, a1-a3 must be set up with right
        ; addresses: a2=&Sa[03], a3=&Sb[03], a1=RUFV_L1P2QR (First constant)

.ruf_mainloop:
        ;A : d0 d5
        ;L0: d1 d6
        ;L1: d2   \- d7,a6; swapped often
        ;L2: d3   /

        ; Pipelines take turns to use d4 scratch register

        move.l  L0_hi(a0),d3        ;d3=L2a=key_hi
        move.l  d3,d5               ;Use d5 as L2b for now

        move.l  (a1)+,d4    ;Use d4 as handy constant L1P2QR
        addq.b  #1,d5       ;Next key (no carry?)

        add.l   d4,d3       ; Perform calculation of L2 after iteration 2
        add.l   d4,d5
        rol.l   d4,d3
        rol.l   d4,d5

        move.l  (a1)+,d0    ;d0=P32QR
        move.l  d5,a6       ;Preserve L2b in a6
        move.l  (a1)+,d2    ;Get L1X in d2
        add.l   d0,d5       ;Perform calculation of Sb[3] in d5
        move.l  (a1)+,d6    ;L0X=Value of L0 going into iteration 3
        rol.l   #3,d5
        ;a1 now points to P+4*Q lookup

        add.l   d3,d0       ;Perform calculation of Sa[3] in d0
        move.l  d5,(a3)+
        rol.l   #3,d0
        move.l  d3,d4   ; Manually do second half of iter 3 on pipeline 1

        move.l  d0,(a2)+
        add.l   d0,d4

        move.l  d6,d1       ;Copy L0X into d1
        move.l  d5,a4       ;Preserve Ab so we can use d5 as loop count
        add.l   d4,d1
        move.l  d2,d7       ;Copy L1X into d7
        rol.l   d4,d1
        move.l  a6,d4       ;Set up scratch reg with L2b for first macro
        moveq   #6-1,d5     ;Loop counter

        ; Repeated round 1 even-odd-even-odd rounds

        ; This macro does a complete iteration on pipeline 1 simultaneously
        ; with the end of the preceding iteration and first half of the next
        ; on pipeline 2

        ; args: 1 is L[j-1] reg on pipeline 1
        ;       2 is L[j]   reg on pipeline 1
        ;       3 is L      reg on pipeline 2
        ;
        ;Upon entry, the scratch register is in use by pipeline 2

ROUND1:	MACRO

        add.l   \1,d0
        add.l   d5,d4       ; Assume d4 already contains prev Lb[n]

        add.l   (a1),d0
        add.l   d4,\3

        rol.l   #3,d0
        rol.l   d4,\3
        ; Pipeline 2 now finished with d4

        move.l  d0,(a2)+

        ; Pipeline 1 claims d4
        move.l  \1,d4

        add.l   \3,d5
        add.l   d0,d4
        add.l   (a1)+,d5    ;Use and advance P+nQ lookup
        add.l   d4,\2
        rol.l   #3,d5
        rol.l   d4,\2
        ;Pipeline 1 finished with d4

        move.l  d5,(a3)+
        move.l  \3,d4       ;Set up scratch for next macro use
        ENDM

        ;Now the iterations
        ;Pipeline 1 does one "whole" iteration in each macro,
        ;pipeline 2 is always half an iteration behind

        ;Would like to align but no scratch registers!

        ; CNOP 0,8
.r1_loop:
        exg     d5,a4       ;a4=Loop count, d5=Ab
        ROUND1  d1,d2,d6    ;Iterations 4,7,... 19
        ROUND1  d2,d3,d7    ;
        exg     d7,a6       ;d6=L0b,a6=L1b,d7=L2b
        ROUND1  d3,d1,d7    ;
        exg     d5,a4       ;d5=Loop count, a4=Ab
        exg     d7,a6       ;d7=L1b
        dbf     d5,.r1_loop

        move.l  a4,d5       ;Get back Ab

        ;The following macros are part of the loop - maybe it will be
        ;quicker to have them inside the loop and do the extra exg d7,a6
        ;afterwards.

        ROUND1  d1,d2,d6    ;22
        ROUND1  d2,d3,d7    ;23
        exg     d7,a6       ;d6=L0b,a6=L1b,d7=L2b
        ROUND1  d3,d1,d7    ;24

        ;Finish round 1

        ;Round 1, Iteration 25 on pipeline 1
        add.l   d1,d0
        ;Iteration 24, second half on pipeline 2
        add.l   d5,d4

        add.l   (a1),d0
        add.l   d4,d6

        rol.l   #3,d0
        rol.l   d4,d6   ;End of pipeline 2 iter24

        move.l  d0,a4   ;Store Sa[25] in a4
        add.l   d6,d5   ;Begin pipe2 iter25

        add.l   (a1)+,d5    ;a1=&Sb[00]
        move.l  d1,d4   ;Pipeline 1 claims d4
        rol.l   #3,d5
        add.l   d0,d4
        move.l  d5,(a3)+    ;Store Sb[25], a3=&Sa[00]
        add.l   d4,d2
        exg     d6,a6       ;a6=L0b,d6=L1b,d7=L2b
        rol.l   d4,d2   ;Pipeline 1 finished with d4, round 1 ended

        move.l  a6,d4   ;Pipe 2 claims d4
        add.l   d2,d0   ;Start of round2 on pipe 1
        add.l   d5,d4
        add.l   a5,d0   ;a5=P0QR3=S[00]
        add.l   d4,d6
        rol.l   #3,d0
        rol.l   d4,d6   ;End of round1 on pipe 2 - finishes with d4

        ;Now a3=Sa[], a1=Sb[] - no point moving pointers around now!

        lea     RUFV_AX(a7),a2      ;Get a2 ready with pointer to constants


        move.l  d0,(a3)+    ;Sa[00]=A
        move.l  d2,d4       ;pipe 1 claims d4

        add.l   d6,d5       ;Start of round2 on pipe 2
        add.l   d0,d4
        add.l   a5,d5
        add.l   d4,d3
        rol.l   #3,d5
        rol.l   d4,d3       ;pipe 1 finishes with d4

        move.l  d5,(a1)+
        move.l  d6,d4       ;pipe2 claims d4

        add.l   (a2),d0     ;pipe1 iter1, S[01]=RUFV_AX
        add.l   d5,d4
        add.l   d3,d0

        add.l   d4,d7
        rol.l   #3,d0
        rol.l   d4,d7       ;pipe2 finishes with d4

        move.l  d0,(a3)+
        move.l  d3,d4       ;pipe1 claims d4
        add.l   d7,d5
        add.l   d0,d4
        add.l   (a2)+,d5    ;Worth caching (a2) = AX in a register?
        add.l   d4,d1
        rol.l   #3,d5
        rol.l   d4,d1       ;pipe1 finished with d4

        move.l  d5,(a1)+
        move.l  d7,d4       ;pipe2 claims d4

        add.l   d1,d0       ;pipe1 iter2
        exg     d7,a6       ;d7=L0b,d6=L1b,a6=L2b
        add.l   (a2),d0     ;S[02]=AXP2QR
        add.l   d5,d4
        rol.l   #3,d0
        add.l   d4,d7
        move.l  d0,(a3)+
        rol.l   d4,d7       ;pipe2 finished with d4

        ;Iteration 2 (only half of it for pipeline 2)

        add.l   (a2),d5     ;pipe2 iter2
        move.l  d1,d4       ;pipe1 claims d4
        add.l   d7,d5
        add.l   d0,d4

        rol.l   #3,d5
        add.l   d4,d2
        move.l  d5,(a1)+
        rol.l   d4,d2       ;pipe1 finishes with d4

        move.l  d7,d4   ; Ready scratch register for macro repetitions
        move.l  d5,a5   ;Overwrite handy constant a5 with Ab

        moveq   #6-1,d5

        ;ROUND2 macro works the same as ROUND1, except table pointers differ
ROUND2:	MACRO

        add.l   (a3),d0
        add.l   d5,d4       ; Assume d4 already contains prev Lb[n]
        add.l   \1,d0
        add.l   d4,\3
        rol.l   #3,d0
        rol.l   d4,\3       ; Pipeline 2 now finished with d4

        move.l  d0,(a3)+        
        move.l  \1,d4   ;Pipeline 1 claims d4
        add.l   (a1),d5
        add.l   d0,d4
        add.l   \3,d5
        add.l   d4,\2
        rol.l   #3,d5
        rol.l   d4,\2   ;Pipeline 1 finished with d4

        move.l  d5,(a1)+
        move.l  \3,d4       ;Set up scratch for next macro use

        ENDM

        CNOP    0,8
.r2_loop:
        exg     d5,a5       ;a5=Loop count, d5=Ab
        ROUND2  d2,d3,d6    ;3
        exg     d6,a6       ;a6=L1b
        ROUND2  d3,d1,d6    ;4
        ROUND2  d1,d2,d7    ;5
        exg     d6,a6       ;d6=L1b
        exg     d5,a5
        dbf     d5,.r2_loop

        move.l  a5,d5

        ;Same story for this tail end - do we include it in the loop,
        ;and mess around with redundant exg afterwards?

        ROUND2  d2,d3,d6    ;21
        exg     d6,a6       ;a6=L1b
        ROUND2  d3,d1,d6    ;22
        ROUND2  d1,d2,d7    ;23
        exg     d7,a6       ;a6=L0b,d7=L1b,d6=L2b
        ROUND2  d2,d3,d7    ;24

        add.l   a4,d0       ;Sa[25] is stored in a4
        add.l   d5,d4       ; Finish 24 manually on pipeline 2
        add.l   d3,d0
        add.l   d4,d6
        rol.l   #3,d0       ;Don't need to store Sa[25]
        rol.l   d4,d6

        move.l  d3,d4       ;pipe1 claims d4
        add.l   d6,d5
        add.l   d0,d4
        add.l   (a1)+,d5    ;a1=&Sa[00]
        add.l   d4,d1
        rol.l   #3,d5       ;Don't need to store Sb[25]
        rol.l   d4,d1       ;pipe1 finished with d4
                        ;Maybe can skip following exg by waiting until round 3
        exg     d6,a6       ;d6=L0b,d7=L1b,a6=L2b

        move.l  a6,d4
        move.l  d7,a3   ;Store pipe2 state in address registers
        add.l   d5,d4
        add.l   d1,d0   ;Begin round3/encryption on pipe1
        add.l   d4,d6
        add.l   (a1)+,d0
        rol.l   d4,d6
        rol.l   #3,d0
 
        ;---- Combined round 3 of key expansion and encryption round ----

        ;Pipeline 1 only

        move.l  d6,a2   ;Store pipe2 state: a2=L0b,a3=L1b,a6=L2b
        move.l  d1,d4
        ;Now d6=eAa, d7=eBa

        ;Iteration 0
        move.l  plain_lo(a0),d6
        add.l   d0,d4
        add.l   d4,d2
        add.l   d0,d6       ;eA=plain_lo + S[00]
        add.l   (a1)+,d0
        rol.l   d4,d2

        ;Iteration 1
        move.l  plain_hi(a0),d7
        add.l   d2,d0
        move.l  d2,d4
        rol.l   #3,d0
        move.l  d5,a5       ;Store Ab so we can use d5 as loop counter
        add.l   d0,d4
        add.l   d0,d7
        add.l   d4,d3
        moveq   #4-1,d5

        bra     .r3a_loop_entry

ROUND3: MACRO
        rol.l   d4,\1       ;L[j-1] <<= scr
        add.l   (a1)+,d0    ;A+=S[i]
        eor.l   \4,\3       ;eA^=eB (or vice versa)
        add.l   \1,d0       ;A+=L[j-1]
        move.l  \1,d4       ;scr <- L[j-1]
        rol.l   #3,d0       ;A <<= 3
        rol.l   \4,\3       ;eA <<= eB (or vv)
        add.l   d0,d4       ;scr += A
        add.l   d0,\3       ;eA += A
        add.l   d4,\2       ;L[j] += scr
        ENDM

        CNOP    0,8
.r3a_loop:
        ROUND3  d1,d2,d6,d7
        ROUND3  d2,d3,d7,d6
.r3a_loop_entry:
        ROUND3  d3,d1,d6,d7      ;Iterations 2-23 inclusive
        ROUND3  d1,d2,d7,d6
        ROUND3  d2,d3,d6,d7
        ROUND3  d3,d1,d7,d6
        dbf     d5,.r3a_loop

        rol.l   d4,d1
        add.l   (a1),d0     ;Iteration 24 (as much of it as needed)

        eor.l   d7,d6
        add.l   d1,d0

        rol.l   d7,d6
        rol.l   #3,d0
        move.l  a5,d5       ;Restore Ab
        add.l   d0,d6

        cmp.l   cypher_lo(a0),d6       ;eAa == cypher.lo?
        bne.s   .ruf_notfounda

        ;Fill in "check" portion of RC5_72UnitWork
        addq.l  #1,check_count(a0)
        move.l  L0_hi(a0),check_hi(a0)
        move.l  L0_mid(a0),check_mid(a0)
        move.l  L0_lo(a0),check_lo(a0)

        ;-- Low 32 bits match! (1 in 2^32 keys!) --
        ; Need to completely re-check key but
        ; generating high 32 bits as well...

        moveq   #0,d0
        jsr     _rc5_check64
        bne.s   .ruf_notfounda

        ;---------------------------------------
        ; 'Interesting' key found on pipeline 1

        move.l  RUFV_LC(a7),d0        ;d0=loop count
        move.l  RUFV_ILC(a7),a1       ;a1=initial loop count address
        add.l   d0,d0
        move.l  (a1),d1
        sub.l   d0,d1   ;Find number of keys checked
        move.l  d1,(a1) ;Return keys checked in in/out parameter

        lea     RUFV_SIZE+88+204(a7),a7

        movem.l (a7)+,d2-7/a2-6
        moveq   #2,d0       ;RESULT_FOUND
        rts

        CNOP    0,8
.ruf_notfounda:
        ;-- Perform round 3 for 'b' key --
        ;Now d6=eAa, d7=eBa
        ;d1=L0b,d2=L1b,d3=L2b and d5=Ab
        ;Retrieve stored state as we go - a2=L0,a3=L1,a6=L2

        add.l   a2,d5
        lea     RUFV_SIZE+88(a7),a1     ;a1=&Sb[00]

        add.l   (a1)+,d5
        move.l  a2,d4

        move.l  plain_lo(a0),d6
        rol.l   #3,d5

        add.l   d5,d6       ;eA=plain_lo + S[00]
        move.l  a3,d2

        add.l   d5,d4
        move.l  a6,d3

        add.l   d4,d2
        move.l  a2,d1

        rol.l   d4,d2

        ;Iteration 1
        add.l   (a1)+,d5

        move.l  plain_hi(a0),d7
        add.l   d2,d5
        move.l  d2,d4
        rol.l   #3,d5

        add.l   d5,d4
        add.l   d5,d7
        moveq   #4-1,d0     ;loop counter
        add.l   d4,d3

        bra     .r3b_loop_entry

ROUND3b:	MACRO
        rol.l   d4,\1
        add.l   (a1)+,d5
        eor.l   \4,\3
        add.l   \1,d5
        move.l  \1,d4
        rol.l   #3,d5
        rol.l   \4,\3
        add.l   d5,d4
        add.l   d5,\3
        add.l   d4,\2
        ENDM

        CNOP    0,8
.r3b_loop:
        ROUND3b d1,d2,d6,d7
        ROUND3b d2,d3,d7,d6
.r3b_loop_entry:
        ROUND3b d3,d1,d6,d7      ;Iterations 2-23 inclusive
        ROUND3b d1,d2,d7,d6
        ROUND3b d2,d3,d6,d7
        ROUND3b d3,d1,d7,d6
        dbf     d0,.r3b_loop

        ;Iteration 24 (as much of it as needed)
        rol.l   d4,d1
        add.l   (a1),d5
        add.l   d1,d5
        eor.l   d7,d6
        rol.l   #3,d5
        rol.l   d7,d6
        add.l   d5,d6
        lea     RUFV_SIZE+88+12(a7),a3  ;a3=&Sb[03]

        cmp.l   cypher_lo(a0),d6       ;eAb == cypher.lo?
        bne.s   .ruf_notfoundb

        ;Fill in "check" portion of RC5_72UnitWork
        addq.l  #1,check_count(a0)
        move.l  L0_hi(a0),d0
        addq.l  #1,d0           ;Pipeline 2 key is one higher
        move.l  d0,check_hi(a0)
        move.l  L0_mid(a0),check_mid(a0)
        move.l  L0_lo(a0),check_lo(a0)

        ;-- Low 32 bits match! (1 in 2^32 keys!) --
        ; Need to completely re-check key but
        ; generating high 32 bits as well...

        moveq   #1,d0
        jsr     _rc5_check64
        bne.s   .ruf_notfoundb

        ;---------------------------------------
        ; 'Interesting' key found on pipeline 2

        move.l  RUFV_LC(a7),d0        ;d0=loop count
        move.l  RUFV_ILC(a7),a1       ;a1=initial key count address
        add.l   d0,d0
        move.l  (a1),d1
        sub.l   d0,d1   ;Find number of keys checked
        addq.l  #1,d1   ;Add one for second pipeline
        move.l  d1,(a1) ;Return keys checked in in/out parameter

        lea     RUFV_SIZE+88+204(a7),a7

        movem.l (a7)+,d2-7/a2-6
        moveq   #2,d0       ;RESULT_FOUND
        rts

        CNOP    0,8
.ruf_notfoundb:
        lea     104(a3),a2             ;a2=&Sa[03]
        move.l  #P0QR3,a5   ;Regain handy constant

        ;Mangle-increment current key

        addq.b  #2,19(a0)       ;Increment L2 by pipeline count
        lea     RUFV_L1P2QR(a7),a1 ; First constant in main loop
        bcc.w   .ruf_midone
        addq.b  #1,20(a0)       ;Every 256^1 (256)
        bcc.s   .ruf_l1chg
        addq.b  #1,21(a0)       ;Every 256^2 (65536)
        bcc.s   .ruf_l1chg
        addq.b  #1,22(a0)       ;Every 256^3 (16777216)
        bcc.s   .ruf_l1chg
        addq.b  #1,23(a0)       ;Every 256^4 (4294967296)
        bcc.s   .ruf_l1chg

        addq.b  #1,24(a0)       ;Every 256^5 (1099511627776)
        bcc.s   .ruf_l0chg
        addq.b  #1,25(a0)       ;Every 256^6 (281474976710656)
        bcc.s   .ruf_l0chg
        addq.b  #1,26(a0)       ;Every 256^7 (72057594037927936)
        bcc.s   .ruf_l0chg
        addq.b  #1,27(a0)       ;Every 256^8

        ; Need to do anything special wrapping 0xff..f -> 0x00..0 ?

.ruf_l0chg:     ; L0 has changed so recalculate "constants"
                ; Only called every 256^5 keys so not worth optimising
        move.l  L0_lo(a0),d1
        add.l   a5,d1   ;d1=L0=L0+S[00]
        ror.l   #3,d1   ;d1=L0=(L0+A)>>>3 = (L0+A)<<<P0QR3
        move.l  d1,RUFV_L0X(a7) ;Set L0x

        add.l   #PR3Q,d1        ;d1=A+P+Q
        rol.l   #3,d1   ;d1=(A+P+Q)<<<3
        move.l  d1,RUFV_AX(a7)  ;Set Ax

        add.l   RUFV_L0X(a7),d1 ;d1=A+L0
        move.l  d1,RUFV_L0XAX(a7)       ;Set A+L0

        move.l  RUFV_AX(a7),d1  ;d1=A   ; Use different dn, work in parallel?
        add.l   #P2Q,d1 ;d1=A+P+2Q
        move.l  d1,RUFV_AXP2Q(a7)       ;Set A+P+2Q

.ruf_l1chg:     ; L1 change requires some recalculation
                ; Every 256 keys = 128 loops, so worth making it quick

        move.l  RUFV_L0XAX(a7),d2
        move.l  L0_mid(a0),d3
        add.l   d2,d3   ;d3=L1+L0x+Ax
        move.l  RUFV_AXP2Q(a7),d0
        rol.l   d2,d3   ;d3=L1x=(L1+L0x+Ax)<<<(L0x+Ax)
        move.l  d3,RUFV_L1X(a7)
        add.l   d3,d0
        rol.l   #3,d0
        move.l  d0,RUFV_AXP2QR(a7)
        move.l  d0,d4   ;Copy AxP2QR into d4
        add.l   d3,d4   ;Add L1x
        add.l   #P3Q,d0
        move.l  d4,RUFV_L1P2QR(a7)
        move.l  d0,RUFV_P32QR(a7)

        RUF_ALIGN       d1,_rc5_72_unit_func_020_030_mh_2 ;Align to 8 bytes and pOEP
.ruf_midone:    subq.l  #1,RUFV_LC(a7) ;Loop back for next key
        bne     .ruf_mainloop

        ;---------------------------------
        ; Key not found on either pipeline
        ; Return iterations*pipeline_count

        lea     RUFV_SIZE+88+204(a7),a7
        movem.l (a7)+,d2-7/a2-6
        moveq   #1,d0       ;RESULT_NOTHING
        rts

;--------------------
