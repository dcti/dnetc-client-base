; Copyright distributed.net 1997-2002 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;-------------------------------------------------------------------
; RISC OS assembler support functions
;

        AREA    |C$$code|, CODE, READONLY

        DCB     "@(#)$Id: riscos_asm_gccsdk.s,v 1.4 2003/11/01 15:07:10 mweiser Exp $", 0
        ALIGN

OS_Byte			*	&06
OS_EnterOS              *       &16
OS_LeaveOS		*	&7C
OS_IntOn		*	&13
OS_IntOff		*	&14
XOS_Module              *       &2001E
XOS_Upcall              *       &20033
OS_ReadMonotonicTime    *       &42
;XTaskWindow_TaskInfo    *       &63380

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

cp15	CP	15

        EXPORT  read_monotonic_time
read_monotonic_time
        SWI     OS_ReadMonotonicTime
        MOV     pc,lr

;
; Read the processor ID.
;
        EXPORT  ARMident
ARMident
	STMDB	sp!,{lr}

	TEQ	r0,r0
	TEQ	pc,pc
	BEQ	ident_32bit

	MOV	r0,#129
	MOV	r1,#0
	MOV	r2,#255
	SWI	OS_Byte
	CMP	r1,#&A5
	BGE	ident_new

	SWI	OS_IntOff
	MOV	r0,#4
	LDR	r1,[r0]
	STMDB	sp!,{r1}
	LDR	r1,=&E1A0F00E
	STR	r1,[r0]

	LDR	r0,=&41560200
	LDR	r1,=&41560250
	STMDB	sp!,{r1}
	SWP	r0,r0,[r13]

	SWI	OS_EnterOS
	MRC	cp15,0,r0,c0,c0
	TEQP	pc,#0
	MOV	r0,r0

	MOV	r1,#4
	ADD	sp,sp,#4
	LDMIA	sp!,{r2}
	STR	r2,[r1]
	SWI	OS_IntOn

	LDMIA	sp!,{pc}

ident_32bit
        SWI     OS_EnterOS
        MRC	cp15,0,r0,c0,c0
	SWI	OS_LeaveOS
	LDMIA	sp!,{pc}

ident_new
        SWI     OS_EnterOS
        MRC	cp15,0,r0,c0,c0
	TEQP	pc,#0
	MOV	r0,r0
	LDR	r1,=&ff00fff0
	AND	r1,r0,r1
	LDR	r2,=&41007100
	CMP	r1,r2
	BEQ	iomd		;ARM7500/FE report as ARM710, identify by integral IOMD id
        LDMIA	sp!,{pc}

iomd
	SWI	OS_EnterOS
	LDR	r1,=&03200098	;IOMD_ID1
	LDRB	r1,[r1]
	TEQP	pc,#0
	MOV	r0,r0
	CMP	r1,#&5b
	LDREQ	r0,=&41007500
	CMP	r1,#&aa
	LDREQ	r0,=&410F7500
	LDMIA	sp!,{pc}

        EXPORT  riscos_upcall_6
riscos_upcall_6
        LDR     ip,adcon_nonzero
        MOV     r0,#6
        LDR     r1,[ip]
        TEQS    r1,#0
        BEQ     allocate_nonzero
        SWI     XOS_Upcall
        MOV     pc,lr

        IMPORT  atexit
allocate_nonzero
        STMFD   sp!,{v1,lr}
        MOV     r0,#6
        MOV     r3,#4
        SWI     XOS_Module
        LDMVSFD sp!,{v1,pc}
        STR     r2,[ip]
        STR     r2,[r2]                 ; a non-zero value...
        MOV     v1,r2
        ADR     a1,deallocate_nonzero
        BL      atexit
        MOV     r0,#6
        MOV     r1,v1
        SWI     XOS_Upcall
        LDMFD   sp!,{v1,pc}

deallocate_nonzero
        LDR     r2,adcon_nonzero
        MOV     r0,#7
        LDR     r2,[r2]
        SWI     XOS_Module
        MOV     pc,lr

adcon_nonzero
        DCD     nonzero

        AREA    fastrc5zidata, DATA, NOINIT
nonzero %       4

        END
