;-------------------------------------------------------------------
; ARM optimised RC5-64 core
;
; Steve Lee, Chris Berry, Tim Dobson 1997,1998
;

        AREA    fastrc5area, CODE


        EXPORT  rc5_unit_func_arm

R       *       12
T       *       2*(R+1)

P32     *       &B7E15163
Q32     *       &9E3779B9
P32_ROR29 *     (P32 :SHL: 3) :OR: (P32 :SHR: 29)

rA      RN      10
rB      RN      11
rC      RN      12

        GBLS    regpool
        GBLS    pool2
        GBLS    regtodo
        GBLS    regload
        GBLS    lastreg
        GBLS    thisreg
        GBLA    rcount
        GBLA    overlap


; This macro fills as many registers as needed in $list
; from the memory pointed to by $reg.
        MACRO
        LoadRegs        $inc, $reg, $list
        ASSERT  rcount>0
 [ rcount >= (1+:LEN:$list)/3
rcount  SETA    rcount-(1+:LEN:$list)/3
regload SETS    $list
 |
regload SETS    $list:LEFT:(rcount*3-1)
rcount  SETA    0
 ]
        LDM$inc $reg,{$regload}
        MEND


; When the current register pool is completely processed, stores
; it out.
        MACRO
        StoreRegs       $reg
 [ :LEN:regload > 1 :LAND: :LEN:regtodo = 0
        STMIA   $reg, {$regload}
 ]
        MEND


; Sets thisreg to the next available register from the loaded pool.
; if none are available then loads some more. This version works left
; to right.
        MACRO
        CycleRegsR      $reg
lastreg SETS    thisreg
 [ :LEN:regtodo = 0
        ASSERT  :LEN:pool2 = 0 :LOR: :LEN:pool2 = 5
  [ :LEN:pool2 = 5
; we have a pool2. This is a pool of the last two available
; registers. only one of these is filled each time round so the
; last one from one filling is available after the next filling
regtodo SETS    regpool:CC:",":CC:(pool2:LEFT:2)
pool2   SETS    (pool2:RIGHT:2):CC:",":CC:(pool2:LEFT:2)
  |
regtodo SETS    regpool
  ]
        LoadRegs        IA, $reg, regtodo
regtodo SETS    regload
 ]

thisreg SETS    regtodo:LEFT:2
 [ :LEN:regtodo > 3
regtodo SETS    regtodo:RIGHT:(:LEN:regtodo-3)
 |
regtodo SETS    ""
 ]
        MEND


; this macro does not use pool2 or set lastreg
        MACRO
        CycleRegsL      $reg
 [ :LEN:regtodo = 0
        LoadRegs        DB, $reg, regpool
regtodo SETS    regload
 ]
thisreg SETS    regtodo:RIGHT:2
 [ :LEN:regtodo > 3
regtodo SETS    regtodo:LEFT:(:LEN:regtodo-3)
 |
regtodo SETS    ""
 ]
        MEND


        MACRO
        OnePass $outer, $inner

 [ Inner = 0

  [ Outer = 0
        ADR     r1, pqtable
regpool SETS    "r4,r5,r6,r7,r8,r9"
pool2   SETS    "rA,rB"
  |
        SUB     r12,r12,#T*4
regpool SETS    "r1,r4,r5,r6,r7,r8,r9,rA,rB"
pool2   SETS    ""
  ]
  [ Outer = 2
overlap SETA    8
rcount  SETA    T-overlap
  |
rcount  SETA    T
  ]
 ]
 [ Outer = 0

        CycleRegsR      r1!
  [ Inner = 0

        ADD     r3, r3, $thisreg, ROR #29
        MOV     r3, r3, ROR #(31 :AND: -P32_ROR29)
  |
        ADD     r0, $thisreg, $lastreg, ROR #29
        ADD     $thisreg, r0, r2
        StoreRegs       r12!

        ADD     r0, r2, $thisreg, ROR #29
        ADD     r3, r3, r0
        RSB     r0, r0, #0
        MOV     r3, r3, ROR r0
  ]

        CycleRegsR      r1!
        ADD     r0, $thisreg, $lastreg, ROR #29
        ADD     $thisreg, r0, r3
        StoreRegs       r12!

        ADD     r0, r3, $thisreg, ROR #29
        ADD     r2, r2, r0
        RSB     r0, r0, #0
        MOV     r2, r2, ROR r0
 |
  [ Outer = 1
        ADD     r0, r2, $thisreg, ROR #29
        CycleRegsR      r12
        ADD     $thisreg, r0, $thisreg, ROR #29
        ADD     r0, r2, $thisreg, ROR #29
        StoreRegs       r12!

        ADD     r3, r3, r0
        RSB     r0, r0, #0
        MOV     r3, r3, ROR r0

        ADD     r0, r3, $thisreg, ROR #29
        CycleRegsR      r12
        ADD     $thisreg, r0, $thisreg, ROR #29

        ADD     r0, r3, $thisreg, ROR #29
        StoreRegs       r12!

        ADD     r2, r2, r0
        RSB     r0, r0, #0
        MOV     r2, r2, ROR r0
  |

   [ rcount = 0
rcount  SETA    overlap
regpool SETS    "r4,r5,r6,r7,r8,r9,rA,rB"
   ]
   [ Inner = 0
        ADD     r0, r2, $thisreg, ROR #29
   |
        ADD     r0, r2, $thisreg
   ]
        CycleRegsR      r12
        ADD     r0, r0, $thisreg, ROR #29
        MOV     $thisreg, r0, ROR #29
        ADD     r0, r2, $thisreg
        StoreRegs       r12!

        ADD     r3, r3, r0
        RSB     r0, r0, #0
        MOV     r3, r3, ROR r0

        ADD     r0, r3, $thisreg
        CycleRegsR      r12
        ADD     r0, r0, $thisreg, ROR #29
        MOV     $thisreg, r0, ROR #29

   [ Inner <> T/2-1
        ADD     r0, r3, $thisreg
        StoreRegs       r12!

        ADD     r2, r2, r0
        RSB     r0, r0, #0
        MOV     r2, r2, ROR r0
   ]
  ]
 ]
        MEND


        GBLA    TMP
        GBLA    Outer
        GBLA    Inner
        GBLA    CNT

CNT     SETA    0
TMP     SETA    P32

pqtable
        WHILE   CNT < T
        &       TMP
TMP     SETA    TMP + Q32
CNT     SETA    CNT + 1
        WEND


rc5_unit_func_arm
        STMFD   r13!, {r4-r11,r14}

        mov     r14, r1

        LDMIA   r0!,{r4-r7}
        LDMIA   r0, {r2-r3}
        STMFD   r13!, {r4-r7}
        SUB     r13,r13,#T*4
        MOV     r12,r13
        STMFD   r13!,{r0,r2-r3}
timingloop
        str     r2,[r12,#-8]

Outer   SETA    0
        WHILE   Outer < 3
Inner   SETA    0
        WHILE   Inner < T/2
        OnePass Outer, Inner
Inner   SETA    Inner + 1
        WEND
Outer   SETA    Outer + 1
        WEND

        ADD     r0,r12,#overlap*4
        LDMIA   r0, {r0-r3}

; r12 now points overlap words from the end of the s table

regtodo SETS    regload
rcount  SETA    T-overlap


TMP     SETA    0
        WHILE   TMP < R


        CycleRegsL      r12!
        sub     r2,r2,$thisreg
        eor     r2,r3,r2,ror r3

        CycleRegsL      r12!
        sub     r3,r3,$thisreg
        eor     r3,r2,r3,ror r2

TMP     SETA    TMP + 1
        WEND

        CycleRegsL      r12!
        sub     r2,r2,$thisreg
        CycleRegsL      r12!

; check r1, r3
        TEQ     r0,r2
        beq     check_r1
missed
        subs    r14,r14,#1
        beq     inc_then_end


; increments 32107654
        ldmdb   r12,{r2,r3}
inc_1st
        adds    r2,r2,#&01000000
        bcc     timingloop

carry_1st
        add     r2,r2,#&00010000
        tst     r2,   #&00ff0000
        bne     timingloop
        sub     r2,r2,#&01000000

        add     r2,r2,#&00000100
        tst     r2,   #&0000ff00
        bne     timingloop
        sub     r2,r2,#&00010000

        add     r2,r2,#&00000001
        ands    r2,r2,#&000000ff
        bne     timingloop

; not likely to happen very often...
        adds    r3,r3,#&01000000
        strcc   r3,[r12,#-4]
        bcc     timingloop

carry_1st_again
        add     r3,r3,#&00010000
        tst     r3,   #&00ff0000
        strne   r3,[r12,#-4]
        bne     timingloop
        sub     r3,r3,#&01000000
        add     r3,r3,#&00000100
        tst     r3,   #&0000ff00
        strne   r3,[r12,#-4]
        bne     timingloop
        sub     r3,r3,#&00010000
        add     r3,r3,#&00000001
        and     r3,r3,#&000000ff
        str     r3,[r12,#-4]
        b       timingloop




; increments 32107654 before leaving
inc_then_end
        ldmfd   r13!,{r1-r3}
        adds    r2,r2,#&01000000
        bcc     the_end

        add     r2,r2,#&00010000
        tst     r2,   #&00ff0000
        bne     the_end
        sub     r2,r2,#&01000000

        add     r2,r2,#&00000100
        tst     r2,   #&0000ff00
        bne     the_end
        sub     r2,r2,#&00010000

        add     r2,r2,#&00000001
        ands    r2,r2,#&000000ff
        bne     the_end

; not likely to happen very often...
        adds    r3,r3,#&01000000
        bcc     the_end

        add     r3,r3,#&00010000
        tst     r3,   #&00ff0000
        bne     the_end
        sub     r3,r3,#&01000000
        add     r3,r3,#&00000100
        tst     r3,   #&0000ff00
        bne     the_end
        sub     r3,r3,#&00010000
        add     r3,r3,#&00000001
        and     r3,r3,#&000000ff
        b       the_end

check_r1
        sub     r3,r3,$thisreg

        TEQ     r1, r3
        bne     missed
        ldmfd   r13!,{r1-r3}
the_end
        mov     r0,r14
        stmia   r1, {r2-r3}
        ADD     r13,r13,#(T+4)*4
        LDMIA   r13!,{r4-r11, pc}^


        END

