; Copyright distributed.net 1997-2008 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; Author: Peter Teichmann <teichp@distributed.net>
; $Id: ogrng-arm1-gccsdk.s,v 1.2 2008/12/30 20:58:43 andreasb Exp $
;
; Stack:
; 0   int *pnodes
; 4   u16 *pchoose
;
; Register:
; r0  oState
; r1  lev
; r2
; r3
; r4
; r5
; r6  depth
; r7  maxlen_m1
; r8  nodes
; r9  comp0
; r10 dist0
; r11 newbit
; r12 limit
; r14 mark

r0      RN      0
r1      RN      1
r2      RN      2
r3      RN      3
r4      RN      4
r5      RN      5
r6      RN      6
r7      RN      7
r8      RN      8
r9      RN      9
r10     RN      10
r11     RN      11
r12     RN      12
r13     RN      13
r14     RN      14
r15     RN      15
sp      RN      13
lr      RN      14
pc      RN      15

; Stack
O_pnodes	EQU	0
O_pchoose	EQU	4

; OgrState
O_max 		EQU	0
O_maxdepth	EQU	4
O_maxdepthm1	EQU	8
O_half_depth	EQU	12
O_half_depth2	EQU	16
O_startdepth	EQU	20
O_stopdepth	EQU	24
O_depth		EQU	28
O_Levels	EQU	32

; OgrLevel
O_list		EQU	0
O_dist		EQU	32
O_comp		EQU	64
O_mark		EQU	96
O_limit		EQU	100
S_lev		EQU	104

CORE_S_OK	EQU	0
CORE_S_CONTINUE	EQU	1
CORE_S_SUCCESS	EQU	2

	AREA	|C$$CODE|, CODE, READONLY
	ALIGN	32
firstblank
	DCB	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	DCB	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	DCB	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	DCB	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	DCB	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	DCB	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	DCB	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	DCB	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	DCB	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
	DCB	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
	DCB	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
	DCB	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
	DCB	3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
	DCB	3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
	DCB	4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
	DCB	5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9

	ALIGN	32
	EXPORT	ogr_cycle_256_arm1
ogr_cycle_256_arm1
	stmdb	sp!, {r4, r5, r6, r7, r8, r9, r10, r11, r12, r14}
	sub	sp, sp, #2*4

	str	r1, [sp, #O_pnodes]	; save pnodes
	str	r2, [sp, #O_pchoose]	; save pchoose
	
	ldr	r8, [r1]		; nodes = *pnodes
	
	ldr	r6, [r0, #O_depth]	; depth = oState->depth
	
	mov	r2, #S_lev
	mul	r2, r6, r2
	add	r1, r0, #O_Levels
	add	r1, r1, r2		; lev = &oState->Levels[depth]
	
	ldr	r7, [r0, #O_max]
	sub	r7, r7, #1		; maxlen_m1 = oState->max - 1
	
; Start SETUP_TOP_STATE(lev)
	
	ldr	r9, [r1, #O_comp]	; comp0 = lev->comp[0];
	
	ldr	r11, [r0, #O_maxdepthm1]
	sub	r11, r6, r11
	mov	r11, r11, lsr#31	; newbit = (depth < oState->maxdepthm1) ? 1 : 0
	
; End SETUP_TOP_STATE(lev)
	
do_while_loop
	ldr	r12, [r1, #O_limit]	; limit = lev->limit
	ldr	r14, [r1, #O_mark]	; mark = lev->mark

for_loop
for_continue
	cmp	r9, #0xfffffffe		; if (comp0 < (SCALAR)~1)
	bhs	comp0_hs

comp0_lo
	mov	r3, r9
	adr	r4, firstblank
	mov	r5, #0
	cmp	r3, #0xffff0000
	movcs	r3, r3, lsl#16
	addcs	r5, r5, #16
	cmp	r3, #0xff000000
	movcs	r3, r3, lsl#8
	ldrb	r4, [r4, r3, lsr#24]
	addcs	r5, r5, #8
	add	r5, r5, r4		; s=LOOKUP_FIRSTBLANK(comp0)
	
	add	r14, r14, r5
	cmp	r14, r12		; if ((mark += s) > limit)
	bgt	for_break
	
; Start COMP_LEFT_LIST_RIGHT(lev, s)
	ldr	r2, [r1, #O_list+0*4]	; equal for all cases!
	add	pc, pc, r5, lsl#8
	%	1*4

; COMP_LEFT_LIST_RIGHT macro, must be always 64 instructions=256 bytes
; because of the method to jump to them!

	MACRO
	COMP_LEFT_LIST_RIGHT $N
	
	mov	r3, r11, lsl #32-$N
	orr	r3, r3, r2, lsr #$N
	str	r3, [r1, #O_list+0*4]	; list[0] = list[0]>>N | newbit<<32-N
	
	ldr	r3, [r1, #O_list+1*4]
	mov	r2, r2, lsl #32-$N
	orr	r2, r2, r3, lsr #$N
	str	r2, [r1, #O_list+1*4]	; list[1] = list[1]>>N | list[0]<<32-N
	
	ldr	r2, [r1, #O_list+2*4]
	mov	r3, r3, lsl #32-$N
	orr	r3, r3, r2, lsr #$N
	str	r3, [r1, #O_list+2*4]	; list[2] = list[2]>>N | list[1]<<32-N
	
	ldr	r3, [r1, #O_list+3*4]
	mov	r2, r2, lsl #32-$N
	orr	r2, r2, r3, lsr #$N
	str	r2, [r1, #O_list+3*4]	; list[3] = list[3]>>N | list[2]<<32-N
	
	ldr	r2, [r1, #O_list+4*4]
	mov	r3, r3, lsl #32-$N
	orr	r3, r3, r2, lsr #$N
	str	r3, [r1, #O_list+4*4]	; list[4] = list[4]>>N | list[3]<<32-N
	
	ldr	r3, [r1, #O_list+5*4]
	mov	r2, r2, lsl #32-$N
	orr	r2, r2, r3, lsr #$N
	str	r2, [r1, #O_list+5*4]	; list[5] = list[5]>>N | list[4]<<32-N
	
	ldr	r2, [r1, #O_list+6*4]
	mov	r3, r3, lsl #32-$N
	orr	r3, r3, r2, lsr #$N
	str	r3, [r1, #O_list+6*4]	; list[6] = list[6]>>N | list[5]<<32-N
	
	ldr	r3, [r1, #O_list+7*4]
	mov	r2, r2, lsl #32-$N
	orr	r2, r2, r3, lsr #$N
	str	r2, [r1, #O_list+7*4]	; list[7] = list[7]>>N | list[6]<<32-N
	
	ldr	r2, [r1, #O_comp+0*4]
	ldr	r3, [r1, #O_comp+1*4]
	mov	r2, r2, lsl #$N
	orr	r9, r2, r3, lsr #32-$N  ; comp0 = comp[0]<<N | comp[1]>>32-N
	str	r9, [r1, #O_comp+0*4]	; comp[0] = comp0
	
	ldr	r2, [r1, #O_comp+2*4]
	mov	r3, r3, lsl #$N
	orr	r3, r3, r2, lsr #32-$N
	str	r3, [r1, #O_comp+1*4]	; comp[1] = comp[1]<<N | comp[2]>>32-N
	
	ldr	r3, [r1, #O_comp+3*4]
	mov	r2, r2, lsl #$N
	orr	r2, r2, r3, lsr #32-$N
	str	r2, [r1, #O_comp+2*4]	; comp[2] = comp[2]<<N | comp[3]>>32-N
	
	ldr	r2, [r1, #O_comp+4*4]
	mov	r3, r3, lsl #$N
	orr	r3, r3, r2, lsr #32-$N
	str	r3, [r1, #O_comp+3*4]	; comp[3] = comp[3]<<N | comp[4]>>32-N
	
	ldr	r3, [r1, #O_comp+5*4]
	mov	r2, r2, lsl #$N
	orr	r2, r2, r3, lsr #32-$N
	str	r2, [r1, #O_comp+4*4]	; comp[4] = comp[4]<<N | comp[5]>>32-N
	
	ldr	r2, [r1, #O_comp+6*4]
	mov	r3, r3, lsl #$N
	orr	r3, r3, r2, lsr #32-$N
	str	r3, [r1, #O_comp+5*4]	; comp[5] = comp[5]<<N | comp[6]>>32-N
	
	ldr	r3, [r1, #O_comp+7*4]
	mov	r2, r2, lsl #$N
	orr	r2, r2, r3, lsr #32-$N
	str	r2, [r1, #O_comp+6*4]	; comp[6] = comp[6]<<N | comp[7]>>32-N
	
	mov	r3, r3, lsl #$N
	str	r3, [r1, #O_comp+7*4]	; comp[7] = comp[7]<<N
	
	mov	r11, #0			; newbit = 0

	b	comp0_fi
	
	MEND

	%	64*4
	COMP_LEFT_LIST_RIGHT 1
	COMP_LEFT_LIST_RIGHT 2
	COMP_LEFT_LIST_RIGHT 3
	COMP_LEFT_LIST_RIGHT 4
	COMP_LEFT_LIST_RIGHT 5
	COMP_LEFT_LIST_RIGHT 6
	COMP_LEFT_LIST_RIGHT 7
	COMP_LEFT_LIST_RIGHT 8
	COMP_LEFT_LIST_RIGHT 9
	COMP_LEFT_LIST_RIGHT 10
	COMP_LEFT_LIST_RIGHT 11
	COMP_LEFT_LIST_RIGHT 12
	COMP_LEFT_LIST_RIGHT 13
	COMP_LEFT_LIST_RIGHT 14
	COMP_LEFT_LIST_RIGHT 15
	COMP_LEFT_LIST_RIGHT 16
	COMP_LEFT_LIST_RIGHT 17
	COMP_LEFT_LIST_RIGHT 18
	COMP_LEFT_LIST_RIGHT 19
	COMP_LEFT_LIST_RIGHT 20
	COMP_LEFT_LIST_RIGHT 21
	COMP_LEFT_LIST_RIGHT 22
	COMP_LEFT_LIST_RIGHT 23
	COMP_LEFT_LIST_RIGHT 24
	COMP_LEFT_LIST_RIGHT 25
	COMP_LEFT_LIST_RIGHT 26
	COMP_LEFT_LIST_RIGHT 27
	COMP_LEFT_LIST_RIGHT 28
	COMP_LEFT_LIST_RIGHT 29
	COMP_LEFT_LIST_RIGHT 30
	COMP_LEFT_LIST_RIGHT 31
	%	9*4
	
;End COMP_LEFT_LIST_RIGHT(lev, s)

comp0_hs
	add	r14, r14, #32		; 
	cmp	r14, r12		; if ((mark += SCALAR_BITS) > limit)
	bgt	for_break
	
	cmp	r9, #0xffffffff
	
;Start COMP_LEFT_LIST_RIGHT_WORD(lev)

	ldr	r2, [r1, #O_list+0*4]
	str	r11, [r1, #O_list+0*4]	; list[0] = newbit
	ldr	r3, [r1, #O_list+1*4]
	str	r2, [r1, #O_list+1*4]	; list[1] = list[0]
	ldr	r2, [r1, #O_list+2*4]
	str	r3, [r1, #O_list+2*4]	; list[2] = list[1]
	ldr	r3, [r1, #O_list+3*4]
	str	r2, [r1, #O_list+3*4]	; list[3] = list[2]
	ldr	r2, [r1, #O_list+4*4]
	str	r3, [r1, #O_list+4*4]	; list[4] = list[3]
	ldr	r3, [r1, #O_list+5*4]
	str	r2, [r1, #O_list+5*4]	; list[5] = list[4]
	ldr	r2, [r1, #O_list+6*4]
	str	r3, [r1, #O_list+6*4]	; list[6] = list[5]
	str	r2, [r1, #O_list+7*4]	; list[7] = list[6]
	
	ldr	r9, [r1, #O_comp+1*4]	; comp0 = comp[1]
	ldr	r2, [r1, #O_comp+2*4]
	str	r9, [r1, #O_comp+0*4]	; comp[0] = comp0
	ldr	r3, [r1, #O_comp+3*4]
	str	r2, [r1, #O_comp+1*4]	; comp[1] = comp[2]
	ldr	r2, [r1, #O_comp+4*4]
	str	r3, [r1, #O_comp+2*4]	; comp[2] = comp[3]
	ldr	r3, [r1, #O_comp+5*4]
	str	r2, [r1, #O_comp+3*4]	; comp[3] = comp[4]
	ldr	r2, [r1, #O_comp+6*4]
	str	r3, [r1, #O_comp+4*4]	; comp[4] = comp[5]
	ldr	r3, [r1, #O_comp+7*4]
	str	r2, [r1, #O_comp+5*4]	; comp[5] = comp[6]
	mov	r11, #0			; newbit = 0
	str	r3, [r1, #O_comp+6*4]	; comp[6] = comp[7]
	str	r11, [r1, #O_comp+7*4]	; comp[7] = 0
	
;End COMP_LEFT_LIST_RIGHT_WORD(lev)
	
	beq	for_continue

comp0_fi
	ldr	r2, [r0, #O_maxdepthm1]
	str	r14, [r1, #O_mark]	; lev->mark = mark
	cmp	r6, r2
	beq	exit
	
;Start PUSH_LEVEL_UPDATE_STATE(lev)
	
	MACRO
	PUSH_PART $I
	
	ldr	r2, [r1, #O_list+$I*4]
	ldr	r3, [r1, #O_dist+$I*4]
	ldr	r4, [r1, #O_comp+$I*4]
	str	r2, [r1, #O_list+S_lev+$I*4]	; list[lev+1] = list[lev]
	orr	r2, r2, r3
	str	r2, [r1, #O_dist+S_lev+$I*4]	; dist[lev+1] = dist[lev] | list[lev+1]
	orr	r2, r2, r4
	str	r2, [r1, #O_comp+S_lev+$I*4]	; comp[lev+1] = comp[lev] | dist[lev+1]
	
	MEND
	
	ldr	r2, [r1, #O_list+0*4]		; PUSH_PART 0 special
	ldr	r3, [r1, #O_dist+0*4]
	ldr	r4, [r1, #O_comp+0*4]
	str	r2, [r1, #O_list+S_lev+0*4]	; list[lev+1] = list[lev]
	orr	r10, r2, r3			; dist0
	str	r10, [r1, #O_dist+S_lev+0*4]	; dist[lev+1] = dist[lev] | list[lev+1]
	orr	r9, r10, r4			; comp0
	str	r9, [r1, #O_comp+S_lev+0*4]	; comp[lev+1] = comp[lev] | dist[lev+1]
	
	PUSH_PART 1
	PUSH_PART 2
	PUSH_PART 3
	PUSH_PART 4
	PUSH_PART 5
	PUSH_PART 6
	PUSH_PART 7
	mov	r11, #1			; newbit = 1
	
;End PUSH_LEVEL_UPDATE_STATE(lev)

	add	r1, r1, #S_lev		; ++lev
	add	r6, r6, #1		; ++depth
	
	ldr	r2, [sp, #O_pchoose]
	mov	r3, r10, lsr #16
	add	r3, r6, r3, lsl#5	; (dist0>>16)*32+depth
	add	r2, r2, r3, lsl#1	; pointer to halfword
	ldrb	r4, [r2, #0]
	ldrb	r12, [r2, #1]
	
	ldr	r2, [r0, #O_half_depth]
	ldr	r3, [r0, #O_half_depth2]
	orr	r12, r4, r12, lsl#8	; limit = choose(dist0, depth);
	cmp	r6, r3			; if (depth <= oState->half_depth2
	bgt	not_between
	cmp	r6, r2			; && depth > oState->half_depth)
	ble	not_between
	
	mov	r4, #S_lev
	mul	r2, r4, r2
	add	r4, r0, #O_Levels+O_mark
	ldr	r2, [r4, r2]
	
	cmp	r6, r3
	sub	r2, r7, r2	; temp = maxlen_m1 - oState->Levels[oState->half_depth].mark
	bge	not_smaller	; if (depth < oState->half_depth2)
	
	mov	r3, r10
	adr	r4, firstblank
	cmp	r3, #0xffff0000
	movcs	r3, r3, lsl#16
	subcs	r2, r2, #16
	cmp	r3, #0xff000000
	movcs	r3, r3, lsl#8
	ldrb	r4, [r4, r3, lsr#24]
	subcs	r2, r2, #8
	sub	r2, r2, r4		; temp -= LOOKUP_FIRSTBLANK(dist0)
	
not_smaller
	cmp	r12, r2			; if (limit > temp)
	movgt	r12, r2			; limit = temp

not_between
	str	r12, [r1, #O_limit]	; lev->limit = limit
	
	subs	r8, r8, #1		; if (--nodes <= 0)
	bgt	for_loop
	
	str	r14, [r1, #O_mark]	; lev->mark = mark
	b	exit

for_break
	sub	r1, r1, #S_lev		; --lev
	sub	r6, r6, #1		; --depth

	ldr	r2, [r0, #O_stopdepth]

; Start POP_LEVEL(lev)

	ldr	r9, [r1, #O_comp+0*4]	; comp0 = lev->comp[0]
	ldr	r10, [r1, #O_dist+0*4]	; dist0 = lev->dist[0]
	mov	r11, #0			; newbit = 0

; End POP_LEVEL(lev)

	cmp	r6, r2			; while (depth > oState->stopdepth)
	bgt	do_while_loop

exit

; Start SAVE_FINAL_STATE(lev)

; End SAVE_FINAL_STATE(lev)

	ldr	r2, [sp, #O_pnodes]
	ldr	r3, [r2]
	sub	r3, r3, r8
	str	r3, [r2]		; *pnodes -= nodes
	
	mov	r0, r6			; return depth
	
	add	sp, sp, #2*4
	ldmia	sp!, {r4, r5, r6, r7, r8, r9, r10, r11, r12, pc}


	END
