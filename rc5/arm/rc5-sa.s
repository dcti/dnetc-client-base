;-------------------------------------------------------------------
; StrongARM optimised RC5-64 core
;
; Steve Lee, Chris Berry, Tim Dobson 1997,1998
;

        AREA    fastrc5area, CODE


        EXPORT  rc5_unit_func_strongarm

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
        GBLS    last1
        GBLS    last2
        GBLS    this1
        GBLS    this2
        GBLA    rcount
        GBLA    overlap


; This macro fills as many registers as needed in $list
; from the memory pointed to by $reg.
        MACRO
        LoadRegs        $inc, $reg, $list, $alt
        ASSERT  rcount>0
 [ rcount >= (1+:LEN:$list)/3
rcount  SETA    rcount-(1+:LEN:$list)/3
regload SETS    $list
 |
regload SETS    $list:LEFT:(rcount*3-1)
rcount  SETA    0
 ]
 [ $alt = 0
        LDM$inc $reg,{$regload}
 |
        LCLS    allreg
        LCLS    altload
        ASSERT  (1+:LEN:regload)/3 :MOD: 2 = 0
allreg  SETS    regload
altload SETS    ""
        WHILE   :LEN:allreg>6
altload SETS    altload:CC:(allreg:RIGHT:3)
allreg  SETS    allreg:LEFT:(:LEN:allreg - 6)
        WEND
altload SETS    (allreg:RIGHT:2):CC:altload
        LDM$inc $reg,{$altload}
 ]
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
        CycleRegsR      $reg, $alt
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
        LoadRegs        IA, $reg, regtodo, $alt
regtodo SETS    regload
 ]

thisreg SETS    regtodo:LEFT:2
 [ :LEN:regtodo > 3
regtodo SETS    regtodo:RIGHT:(:LEN:regtodo-3)
 |
regtodo SETS    ""
 ]
        MEND


; this macro must only be called when the total regpool (with pool2
; halved) is even. last1 and last2 should only be used if alternate
; loading is set and a pool2 is available.
; when alternate loading is used, only this2 contains valid input data.
        MACRO
        CycleRegsR2     $reg, $alt
last1   SETS    this1
last2   SETS    this2
        CycleRegsR      $reg, $alt
this1   SETS    thisreg
        CycleRegsR      $reg, $alt
this2   SETS    thisreg
        MEND


; this macro does not use pool2 or set lastreg
        MACRO
        CycleRegsL      $reg
 [ :LEN:regtodo = 0
        LoadRegs        DB, $reg, regpool, 0
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
        CycleRegsL2     $reg
        CycleRegsL      $reg
this2   SETS    thisreg
        CycleRegsL      $reg
this1   SETS    thisreg
        MEND


        MACRO
        OnePass $outer, $inner

 [ Inner = 0

  [ Outer = 0
        ADR     r6, pqtable
regpool SETS    "r7,r8,r9"  ; odd number of registers
pool2   SETS    "rA,rB"   ; plus one of these makes it even
  |
        SUB     r12,r12,#T*4*2
regpool SETS    "r6,r7,r8,r9,rA,rB"  ; even number of registers
pool2   SETS    ""
  ]
  [ Outer = 2
overlap SETA    6
rcount  SETA    T*2-overlap
  |
rcount  SETA    T*2
  ]
 ]
 [ Outer = 0

        CycleRegsR2     r6!, 1
  [ Inner = 0
        MOV     $this1,$this2

        ADD     r3, r3, $this1, ROR #29
        ADD     r5, r5, $this2, ROR #29
        MOV     r3, r3, ROR #(31 :AND: -P32_ROR29)
        MOV     r5, r5, ROR #(31 :AND: -P32_ROR29)
  |
        ADD     r0, $this2, $last1, ROR #29
        ADD     r1, $this2, $last2, ROR #29
        ADD     $this1, r0, r2
        ADD     $this2, r1, r4
        StoreRegs       r12!

        ADD     r0, r2, $this1, ROR #29
        ADD     r1, r4, $this2, ROR #29
        ADD     r3, r3, r0
        ADD     r5, r5, r1
        RSB     r0, r0, #0
        RSB     r1, r1, #0
        MOV     r3, r3, ROR r0
        MOV     r5, r5, ROR r1
  ]

        CycleRegsR2     r6!, 1
        ADD     r0, $this2, $last1, ROR #29
        ADD     r1, $this2, $last2, ROR #29
        ADD     $this1, r0, r3
        ADD     $this2, r1, r5
        StoreRegs       r12!

        ADD     r0, r3, $this1, ROR #29
        ADD     r1, r5, $this2, ROR #29
        ADD     r2, r2, r0
        ADD     r4, r4, r1
        RSB     r0, r0, #0
        RSB     r1, r1, #0
        MOV     r2, r2, ROR r0
        MOV     r4, r4, ROR r1
 |
  [ Outer = 1
        ADD     r0, r2, $this1, ROR #29
        ADD     r1, r4, $this2, ROR #29
        CycleRegsR2     r12, 0
        ADD     $this1, r0, $this1, ROR #29
        ADD     $this2, r1, $this2, ROR #29
        ADD     r0, r2, $this1, ROR #29
        ADD     r1, r4, $this2, ROR #29
        StoreRegs       r12!

        ADD     r3, r3, r0
        ADD     r5, r5, r1
        RSB     r0, r0, #0
        RSB     r1, r1, #0
        MOV     r3, r3, ROR r0
        MOV     r5, r5, ROR r1

        ADD     r0, r3, $this1, ROR #29
        ADD     r1, r5, $this2, ROR #29
        CycleRegsR2     r12, 0
        ADD     $this1, r0, $this1, ROR #29
        ADD     $this2, r1, $this2, ROR #29

        ADD     r0, r3, $this1, ROR #29
        ADD     r1, r5, $this2, ROR #29
        StoreRegs       r12!

        ADD     r2, r2, r0
        ADD     r4, r4, r1
        RSB     r0, r0, #0
        RSB     r1, r1, #0
        MOV     r2, r2, ROR r0
        MOV     r4, r4, ROR r1
  |

   [ rcount = 0
rcount  SETA    overlap
regpool SETS    "r6,r7,r8,r9,rA,rB"  ; even number of registers
        ASSERT  overlap = (1+:LEN:regpool)/3
   ]
   [ Inner = 0
        ADD     r0, r2, $this1, ROR #29
        ADD     r1, r4, $this2, ROR #29
   |
        ADD     r0, r2, $this1
        ADD     r1, r4, $this2
   ]
        CycleRegsR2     r12, 0
        ADD     r0, r0, $this1, ROR #29
        ADD     r1, r1, $this2, ROR #29
        MOV     $this1, r0, ROR #29
        MOV     $this2, r1, ROR #29
        ADD     r0, r2, $this1
        ADD     r1, r4, $this2
        StoreRegs       r12!

        ADD     r3, r3, r0
        ADD     r5, r5, r1
        RSB     r0, r0, #0
        RSB     r1, r1, #0
        MOV     r3, r3, ROR r0
        MOV     r5, r5, ROR r1

        ADD     r0, r3, $this1
        ADD     r1, r5, $this2
        CycleRegsR2     r12, 0
        ADD     r0, r0, $this1, ROR #29
        ADD     r1, r1, $this2, ROR #29
        MOV     $this1, r0, ROR #29
        MOV     $this2, r1, ROR #29

   [ Inner <> T/2-1
        ADD     r0, r3, $this1
        ADD     r1, r5, $this2
        StoreRegs       r12!

        ADD     r2, r2, r0
        ADD     r4, r4, r1
        RSB     r0, r0, #0
        RSB     r1, r1, #0
        MOV     r2, r2, ROR r0
        MOV     r4, r4, ROR r1
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


        IMPORT  rc5_unit_func_arm
        IMPORT  __rt_stkovf_split_big
rc5_unit_func_strongarm
        mov     r12,r13
        STMFD   r13!, {r4-r12,r14,pc}
        sub     r11,r12,#4
        sub     r12,r13,#&200
        cmps    r12,r10
        bllt    __rt_stkovf_split_big
        str     r11,[r13,#-4]!

        tst     r1,#1
        bne     odd
even
; now we have an even number, so we can process them 2 at a time.
        mov     r14, r1
now_its_even

        LDMIA   r0!,{r4-r7}
        LDMIA   r0, {r2-r3}
        STMFD   r13!, {r4-r7}
        SUB     r13,r13,#T*4*2 +20
        STR     r0,[r13]
        ADD     r12,r13,#20
timingloop
        adds    r4,r2,#&01000000
        bcs     carry_2nd
done_2nd
        mov     r5,r3
done_2nd_again
        stmdb   r12,{r2-r5}

;key expansion
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
rcount  SETA    T*2-overlap


TMP     SETA    0
        WHILE   TMP < R


        CycleRegsL2     r12!
 [ TMP = 0
        sub     r4,r2,$this2
        sub     r2,r2,$this1
        eor     r4,r3,r4,ror r3
        eor     r2,r3,r2,ror r3

        CycleRegsL2     r12!
        sub     r5,r3,$this2
        sub     r3,r3,$this1
 |
        sub     r2,r2,$this1
        sub     r4,r4,$this2
        eor     r2,r3,r2,ror r3
        eor     r4,r5,r4,ror r5

        CycleRegsL2     r12!
        sub     r3,r3,$this1
        sub     r5,r5,$this2
 ]
        eor     r3,r2,r3,ror r2
        eor     r5,r4,r5,ror r4

TMP     SETA    TMP + 1
        WEND

        CycleRegsL2     r12!
        sub     r2,r2,$this1
        sub     r4,r4,$this2
        CycleRegsL2     r12!

; check
check_r2
        teq     r0,r2
        beq     check_r3
check_r4
        TEQ     r0,r4
        beq     check_r5
missed
        ldmdb   r12,{r2,r3}
        subs    r14,r14,#2
        beq     the_end


; increments 32107654
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
        bcc     timingloop

carry_1st_again
        add     r3,r3,#&00010000
        tst     r3,   #&00ff0000
        bne     timingloop
        sub     r3,r3,#&01000000
        add     r3,r3,#&00000100
        tst     r3,   #&0000ff00
        bne     timingloop
        sub     r3,r3,#&00010000
        add     r3,r3,#&00000001
        and     r3,r3,#&000000ff
        b       timingloop

carry_2nd
        add     r4,r4,#&00010000
        tst     r4,   #&00ff0000
        bne     done_2nd
        sub     r4,r4,#&01000000
        add     r4,r4,#&00000100
        tst     r4,   #&0000ff00
        bne     done_2nd
        sub     r4,r4,#&00010000
        add     r4,r4,#&00000001
        ands    r4,r4,#&000000ff
        bne     done_2nd
        adds    r5,r3,#&01000000
        bcc     done_2nd_again
        add     r5,r5,#&00010000
        tst     r5,   #&00ff0000
        bne     done_2nd_again
        sub     r5,r5,#&01000000
        add     r5,r5,#&00000100
        tst     r5,   #&0000ff00
        bne     done_2nd_again
        sub     r5,r5,#&00010000
        add     r5,r5,#&00000001
        and     r5,r5,#&000000ff
        b       done_2nd_again

        ;; end of new thing


the_end
        mov     r0,#0

; increments 32107654 before leaving
inc_then_end
        ldmdb   r12,{r2,r3}
        adds    r2,r2,#&01000000
        bcc     function_exit

        add     r2,r2,#&00010000
        tst     r2,   #&00ff0000
        bne     function_exit
        sub     r2,r2,#&01000000

        add     r2,r2,#&00000100
        tst     r2,   #&0000ff00
        bne     function_exit
        sub     r2,r2,#&00010000

        add     r2,r2,#&00000001
        ands    r2,r2,#&000000ff
        bne     function_exit

; not likely to happen very often...
        adds    r3,r3,#&01000000
        bcc     function_exit

        add     r3,r3,#&00010000
        tst     r3,   #&00ff0000
        bne     function_exit
        sub     r3,r3,#&01000000
        add     r3,r3,#&00000100
        tst     r3,   #&0000ff00
        bne     function_exit
        sub     r3,r3,#&00010000
        add     r3,r3,#&00000001
        and     r3,r3,#&000000ff


function_exit
        ldr     r1,[r13]
        stmia   r1,{r2,r3}
        ldr     r11,[r12,#(T*2+4)*4]
        ldmdb   r11, {r4-r11,r13, pc}


check_r5
        sub     r5,r5,$this2
        TEQ     r1, r5
        bne     missed

;it's r4,r5!
        ldmdb   r12,{r2-r5}
        sub     r0,r14,#1
        b       function_exit


check_r3
        sub     r3,r3,$this1
        TEQ     r1, r3
        bne     check_r4

; it's r2,r3!
        ldmdb   r12,{r2-r3}
        mov     r0,r14
        b       function_exit



odd
        mov     r10, r0
        mov     r11,r1

        mov     r1,#1
        bl      rc5_unit_func_arm

        cmp     r0,#0
        bne     found_on_single
        bics    r14,r11,#1
        movne   r0,r10
        bne     now_its_even  ; were we asked to do exactly one key?

found_on_single
        movne   r0,r11
        ldr     r11,[r13]
        ldmdb   r11, {r4-r11, r13, pc}


        DCD  &DEAD

OS_EnterOS              *       &16
XOS_Module              *       &2001E
XOS_Upcall              *       &20033
OS_ReadMonotonicTime    *       &42
XTaskWindow_TaskInfo    *       &63380

read_monotonic_time
        EXPORT  read_monotonic_time
        swi     OS_ReadMonotonicTime
        mov     pc,lr


find_core
        EXPORT  find_core
        mov     r1,lr
        swi     OS_EnterOS
        mrc     p15, 0, r0, c0, c0, 0
        and     r0, r0, #0xf000
        teq     r0, #0xa000             ; StrongARM = 'axxx'
        moveq   r0,#1
        movne   r0,#0
        movs    pc,r1

riscos_check_taskwindow
        EXPORT  riscos_check_taskwindow
        mov     r1,r0
        mov     r0,#0
        swi     XTaskWindow_TaskInfo
        movvs   r0,#0
        str     r0,[r1]
        movs    pc,lr

riscos_upcall_6
        EXPORT  riscos_upcall_6
        LDR     ip,adcon_nonzero
        MOV     r0,#6
        LDR     r1,[ip]
        TEQS    r1,#0
        BEQ     allocate_nonzero
        SWI     XOS_Upcall
        MOVS    pc,lr

        IMPORT  atexit
allocate_nonzero
        STMFD   sp!,{v1,lr}
        MOV     r0,#6
        MOV     r3,#4
        SWI     XOS_Module
        LDMVSFD sp!,{v1,pc}^
        STR     r2,[ip]
        STR     r2,[r2]                 ; a non-zero value...
        MOV     v1,r2
        ADR     a1,deallocate_nonzero
        BL      atexit
        MOV     r0,#6
        MOV     r1,v1
        SWI     XOS_Upcall
        LDMFD   sp!,{v1,pc}^

deallocate_nonzero
        LDR     r2,adcon_nonzero
        MOV     r0,#7
        LDR     r2,[r2]
        SWI     XOS_Module
        MOVS    pc,lr

adcon_nonzero
        DCD     nonzero

        AREA    fastrc5zidata, DATA, NOINIT
nonzero %       4

        END


