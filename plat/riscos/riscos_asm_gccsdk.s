;-------------------------------------------------------------------
; RISC OS assembler support functions
;

        AREA    |C$$code|, CODE, READONLY

        DCB     "@(#)$Id: riscos_asm_gccsdk.s,v 1.1.2.1 2001/05/06 11:08:34 teichp Exp $", 0
        ALIGN

OS_EnterOS              *       &16
XOS_Module              *       &2001E
XOS_Upcall              *       &20033
OS_ReadMonotonicTime    *       &42
XTaskWindow_TaskInfo    *       &63380

a1	RN	0
a2	RN	1
a3	RN	2
a4	RN	3
v1	RN	4
v2	RN	5
v3	RN	6
v4	RN	7
v5	RN	8
v6	RN	9
v7	RN	10
fp	RN	11
ip	RN	12
sp	RN	13
lr	RN	14
pc	RN	15

r0	RN	0
r1	RN	1
r2	RN	2
r3	RN	3

c0	CN	0

        EXPORT  read_monotonic_time
read_monotonic_time
        SWI     OS_ReadMonotonicTime
        MOVS    pc,lr

;
; Read the processor ID.
;
        EXPORT  ARMident
ARMident
        MOV     a2,lr
        SWI     OS_EnterOS
        DCD	&EE100F10	;MRC     cp15,0,a1,c0,c0
        MOVS    pc,a2

IOMD_Base       *       &03200000
IOMD_ID0        *       &94
IOMD_ID1        *       &98

        EXPORT  IOMDident
IOMDident
        MOV     a2,lr
        SWI     OS_EnterOS
        MOV     a3,#IOMD_Base
        LDRB    a1,[a3,#IOMD_ID0]
        LDRB    a4,[a3,#IOMD_ID1]
        ADD     a1,a1,a4,LSL #8
        MOVS    pc,a2

        EXPORT  riscos_upcall_6
riscos_upcall_6
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
