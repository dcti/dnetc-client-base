; Copyright distributed.net 1997-2008 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; Author: Peter Teichmann <teichp@distributed.net>
; $Id: ogrng-arm2-gccsdk.s,v 1.2 2009/01/04 22:00:30 teichp Exp $
;
; ARMv5 variant, optimized for XScale:
; * Intel really did a extremly bad job. They had better just taken the
;   StrongARM core and scaled to 0.18u. Consequences are:
; * 3 cycle more for ldm/stm than sequence of ldr/str (SA 1)
; * 5 cycle delay for mispredicted branches and data
;   processing with pc destination. Even unconditional branches need to
;   be predicted! Jesus! (SA 2/3)
; * 8 cycle delay for load to pc (SA 4)
; * predicted branch 1 cycle, mispredicted branch 5 cycles (SA taken 2/not
;   taken 1)
; * avoid branch target buffer conflicts! The branch target buffer is direct
;   mapped and contains 128 entries. (problem does not exist on SA)
; * 3 cycle result delay for ldr (SA 2)
; * 2 cycle delay for data processing if result is used for
;   shift-by-immediate in next instruction (SA 1, as normal)
;
; Stack:
; 0   int *pnodes
;
; Register:
; r0  oState
; r1  lev
; r2  pchoose
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

	AREA	|C$$CODE|, CODE, READONLY

	ALIGN	32
	EXPORT	ogr_cycle_256_arm2
ogr_cycle_256_arm2
	stmdb	sp!, {r4, r5, r6, r7, r8, r9, r10, r11, r12, r14}
	sub	sp, sp, #2*4

	str	r1, [sp, #O_pnodes]	; save pnodes	
	ldr	r8, [r1]		; nodes = *pnodes
	
	ldr	r6, [r0, #O_depth]	; depth = oState->depth
	
	mov	r5, #S_lev
	mul	r5, r6, r5
	add	r1, r0, #O_Levels
	add	r1, r1, r5		; lev = &oState->Levels[depth]
	
	ldr	r7, [r0, #O_max]
	sub	r7, r7, #1		; maxlen_m1 = oState->max - 1
	
; Start SETUP_TOP_STATE(lev)
	
	ldr	r9, [r1, #O_comp]	; comp0 = lev->comp[0]
	
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
	mvn	r5, r9
	clz	r5, r5
	add	r5, r5, #1		; s=LOOKUP_FIRSTBLANK(comp0)
	
	add	r14, r14, r5
	cmp	r14, r12		; if ((mark += s) > limit)
	bgt	for_break
	
; Start COMP_LEFT_LIST_RIGHT(lev, s)
	ldr	r3, [r1, #O_list+0*4]	; equal for all cases!
	ldr	r4, [r1, #O_list+1*4]	; equal for all cases!

	add	pc, pc, r5, lsl#2	; this is better for XScale because
	%	2*4			; it avoids the BTB conflict for
	b	cllr_1			; b comp0_fi
	b	cllr_2
	b	cllr_3
	b	cllr_4
	b	cllr_5
	b	cllr_6
	b	cllr_7
	b	cllr_8
	b	cllr_9
	b	cllr_10
	b	cllr_11
	b	cllr_12
	b	cllr_13
	b	cllr_14
	b	cllr_15
	b	cllr_16
	b	cllr_17
	b	cllr_18
	b	cllr_19
	b	cllr_20
	b	cllr_21
	b	cllr_22
	b	cllr_23
	b	cllr_24
	b	cllr_25
	b	cllr_26
	b	cllr_27
	b	cllr_28
	b	cllr_29
	b	cllr_30

; COMP_LEFT_LIST_RIGHT macro, should be 2^N+/-1 instructions to avoid
; BTB conflicts (presently 127)

	MACRO
	COMP_LEFT_LIST_RIGHT $N
	
	mov	r11, r11, lsl #32-$N
	ldr	r5, [r1, #O_list+2*4]
	orr	r11, r11, r3, lsr #$N
	str	r11, [r1, #O_list+0*4]	; list[0] = list[0]>>N | newbit<<32-N
	
	mov	r11, r3, lsl #32-$N
	ldr	r3, [r1, #O_list+3*4]
	orr	r11, r11, r4, lsr #$N
	str	r11, [r1, #O_list+1*4]	; list[1] = list[1]>>N | list[0]<<32-N
	
	mov	r11, r4, lsl #32-$N
	ldr	r4, [r1, #O_list+4*4]
	orr	r11, r11, r5, lsr #$N
	str	r11, [r1, #O_list+2*4]	; list[2] = list[2]>>N | list[1]<<32-N
	
	mov	r11, r5, lsl #32-$N
	ldr	r5, [r1, #O_list+5*4]
	orr	r11, r11, r3, lsr #$N
	str	r11, [r1, #O_list+3*4]	; list[3] = list[3]>>N | list[2]<<32-N
	
	mov	r11, r3, lsl #32-$N
	ldr	r3, [r1, #O_list+6*4]
	orr	r11, r11, r4, lsr #$N
	str	r11, [r1, #O_list+4*4]	; list[4] = list[4]>>N | list[3]<<32-N
	
	mov	r11, r4, lsl #32-$N
	ldr	r4, [r1, #O_list+7*4]
	orr	r11, r11, r5, lsr #$N
	str	r11, [r1, #O_list+5*4]	; list[5] = list[5]>>N | list[4]<<32-N
	
	mov	r11, r5, lsl #32-$N
	orr	r11, r11, r3, lsr #$N
	str	r11, [r1, #O_list+6*4]	; list[6] = list[6]>>N | list[5]<<32-N
	
	mov	r11, r3, lsl #32-$N
	orr	r11, r11, r4, lsr #$N
	str	r11, [r1, #O_list+7*4]	; list[7] = list[7]>>N | list[6]<<32-N
	
	ldr	r3, [r1, #O_comp+0*4]
	ldr	r4, [r1, #O_comp+1*4]
	ldr	r5, [r1, #O_comp+2*4]
	mov	r3, r3, lsl #$N
	orr	r9, r3, r4, lsr #32-$N  ; comp0 = comp[0]<<N | comp[1]>>32-N
	str	r9, [r1, #O_comp+0*4]	; comp[0] = comp0
	
	ldr	r3, [r1, #O_comp+3*4]
	mov	r4, r4, lsl #$N
	orr	r4, r4, r5, lsr #32-$N
	str	r4, [r1, #O_comp+1*4]	; comp[1] = comp[1]<<N | comp[2]>>32-N
	
	ldr	r4, [r1, #O_comp+4*4]
	mov	r5, r5, lsl #$N
	orr	r5, r5, r3, lsr #32-$N
	str	r5, [r1, #O_comp+2*4]	; comp[2] = comp[2]<<N | comp[3]>>32-N
	
	ldr	r5, [r1, #O_comp+5*4]
	mov	r3, r3, lsl #$N
	orr	r3, r3, r4, lsr #32-$N
	str	r3, [r1, #O_comp+3*4]	; comp[3] = comp[3]<<N | comp[4]>>32-N
	
	ldr	r3, [r1, #O_comp+6*4]
	mov	r4, r4, lsl #$N
	orr	r4, r4, r5, lsr #32-$N
	str	r4, [r1, #O_comp+4*4]	; comp[4] = comp[4]<<N | comp[5]>>32-N
	
	ldr	r4, [r1, #O_comp+7*4]
	mov	r5, r5, lsl #$N
	orr	r5, r5, r3, lsr #32-$N
	str	r5, [r1, #O_comp+5*4]	; comp[5] = comp[5]<<N | comp[6]>>32-N
	
	mov	r3, r3, lsl #$N
	orr	r3, r3, r4, lsr #32-$N
	str	r3, [r1, #O_comp+6*4]	; comp[6] = comp[6]<<N | comp[7]>>32-N
	
	mov	r4, r4, lsl #$N
	str	r4, [r1, #O_comp+7*4]	; comp[7] = comp[7]<<N
	
	mov	r11, #0			; newbit = 0

	b	comp0_fi

	MEND
	
	%	0*4
cllr_31
	COMP_LEFT_LIST_RIGHT 31
	%	64*4
cllr_1
	COMP_LEFT_LIST_RIGHT 1
	%	64*4
cllr_2
	COMP_LEFT_LIST_RIGHT 2
	%	64*4
cllr_3
	COMP_LEFT_LIST_RIGHT 3
	%	64*4
cllr_4
	COMP_LEFT_LIST_RIGHT 4
	%	64*4
cllr_5
	COMP_LEFT_LIST_RIGHT 5
	%	64*4
cllr_6
	COMP_LEFT_LIST_RIGHT 6
	%	64*4
cllr_7
	COMP_LEFT_LIST_RIGHT 7
	%	64*4
cllr_8
	COMP_LEFT_LIST_RIGHT 8
	%	64*4
cllr_9
	COMP_LEFT_LIST_RIGHT 9
	%	64*4
cllr_10
	COMP_LEFT_LIST_RIGHT 10
	%	64*4
cllr_11
	COMP_LEFT_LIST_RIGHT 11
	%	64*4
cllr_12
	COMP_LEFT_LIST_RIGHT 12
	%	64*4
cllr_13
	COMP_LEFT_LIST_RIGHT 13
	%	64*4
cllr_14
	COMP_LEFT_LIST_RIGHT 14
	%	64*4
cllr_15
	COMP_LEFT_LIST_RIGHT 15
	%	64*4
cllr_16
	COMP_LEFT_LIST_RIGHT 16
	%	64*4
cllr_17
	COMP_LEFT_LIST_RIGHT 17
	%	64*4
cllr_18
	COMP_LEFT_LIST_RIGHT 18
	%	64*4
cllr_19
	COMP_LEFT_LIST_RIGHT 19
	%	64*4
cllr_20
	COMP_LEFT_LIST_RIGHT 20
	%	64*4
cllr_21
	COMP_LEFT_LIST_RIGHT 21
	%	64*4
cllr_22
	COMP_LEFT_LIST_RIGHT 22
	%	64*4
cllr_23
	COMP_LEFT_LIST_RIGHT 23
	%	64*4
cllr_24
	COMP_LEFT_LIST_RIGHT 24
	%	64*4
cllr_25
	COMP_LEFT_LIST_RIGHT 25
	%	64*4
cllr_26
	COMP_LEFT_LIST_RIGHT 26
	%	64*4
cllr_27
	COMP_LEFT_LIST_RIGHT 27
	%	64*4
cllr_28
	COMP_LEFT_LIST_RIGHT 28
	%	64*4
cllr_29
	COMP_LEFT_LIST_RIGHT 29
	%	64*4
cllr_30
	COMP_LEFT_LIST_RIGHT 30
	%	44*4
	
; End COMP_LEFT_LIST_RIGHT(lev, s)

comp0_hs
	add	r14, r14, #32		; 
	cmp	r14, r12		; if ((mark += SCALAR_BITS) > limit)
	bgt	for_break
	
	cmp	r9, #0xffffffff
	
; Start COMP_LEFT_LIST_RIGHT_WORD(lev)

	ldr	r3, [r1, #O_list+0*4]
	ldr	r4, [r1, #O_list+1*4]
	str	r11, [r1, #O_list+0*4]	; list[0] = newbit
	ldr	r5, [r1, #O_list+2*4]
	str	r3, [r1, #O_list+1*4]	; list[1] = list[0]
	ldr	r3, [r1, #O_list+3*4]
	str	r4, [r1, #O_list+2*4]	; list[2] = list[1]
	ldr	r4, [r1, #O_list+4*4]
	str	r5, [r1, #O_list+3*4]	; list[3] = list[2]
	ldr	r5, [r1, #O_list+5*4]
	str	r3, [r1, #O_list+4*4]	; list[4] = list[3]
	ldr	r3, [r1, #O_list+6*4]
	str	r4, [r1, #O_list+5*4]	; list[5] = list[4]

	ldr	r9, [r1, #O_comp+1*4]	; comp0 = comp[1]
	str	r5, [r1, #O_list+6*4]	; list[6] = list[5]
	ldr	r4, [r1, #O_comp+2*4]
	str	r3, [r1, #O_list+7*4]	; list[7] = list[6]
	
	ldr	r5, [r1, #O_comp+3*4]
	str	r9, [r1, #O_comp+0*4]	; comp[0] = comp0
	ldr	r3, [r1, #O_comp+4*4]
	str	r4, [r1, #O_comp+1*4]	; comp[1] = comp[2]
	ldr	r4, [r1, #O_comp+5*4]
	str	r5, [r1, #O_comp+2*4]	; comp[2] = comp[3]
	ldr	r5, [r1, #O_comp+6*4]
	str	r3, [r1, #O_comp+3*4]	; comp[3] = comp[4]
	ldr	r3, [r1, #O_comp+7*4]
	str	r4, [r1, #O_comp+4*4]	; comp[4] = comp[5]
	mov	r11, #0			; newbit = 0
	str	r5, [r1, #O_comp+5*4]	; comp[5] = comp[6]
	str	r3, [r1, #O_comp+6*4]	; comp[6] = comp[7]
	str	r11, [r1, #O_comp+7*4]	; comp[7] = 0
	
; End COMP_LEFT_LIST_RIGHT_WORD(lev)
	
	beq	for_continue

comp0_fi
	ldr	r5, [r0, #O_maxdepthm1]
	str	r14, [r1, #O_mark]	; lev->mark = mark
	cmp	r6, r5			; if (depth == oState->maxdepthm1)
	beq	exit
	
; Start PUSH_LEVEL_UPDATE_STATE(lev)
	
	MACRO
	PUSH_PART $I
	
	ldr	r3, [r1, #O_list+$I*4]
	ldr	r4, [r1, #O_dist+$I*4]
	ldr	r5, [r1, #O_comp+$I*4]
	str	r3, [r1, #O_list+S_lev+$I*4]	; list[lev+1] = list[lev]
	orr	r3, r3, r4
	str	r3, [r1, #O_dist+S_lev+$I*4]	; dist[lev+1] = dist[lev] | list[lev+1]
	orr	r3, r3, r5
	str	r3, [r1, #O_comp+S_lev+$I*4]	; comp[lev+1] = comp[lev] | dist[lev+1]
	
	MEND
	
	ldr	r3, [r1, #O_list+0*4]		; PUSH_PART 0 special
	ldr	r4, [r1, #O_dist+0*4]
	ldr	r5, [r1, #O_comp+0*4]
	str	r3, [r1, #O_list+S_lev+0*4]	; list[lev+1] = list[lev]
	orr	r10, r3, r4			; dist0
	str	r10, [r1, #O_dist+S_lev+0*4]	; dist[lev+1] = dist[lev] | list[lev+1]
	orr	r9, r10, r5			; comp0
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
	
	ldr	r4, [r0, #O_half_depth2]
	ldr	r3, [r0, #O_half_depth]
	
	mov	r5, r10, lsr #16
	add	r5, r6, r5, lsl #5	; (dist0>>16)*32+depth
	add	r5, r2, r5, lsl #1	; pointer to halfword
	ldrh	r12, [r5]		; limit = choose(dist0, depth)

	cmp	r6, r4			; if (depth <= oState->half_depth2
	bgt	not_between
	cmp	r6, r3			; && depth > oState->half_depth)
	ble	not_between
	
	add	r5, r3, r3, lsl #2
	add	r5, r5, r3, lsl #3
	add	r5, r0, r5, lsl #3		; r5 = r0 + r3 * 104
	ldr	r3, [r5, #O_Levels+O_mark]	; oState->Levels[oState->half_depth].mark
	
	cmp	r6, r4
	mvn	r5, r10		; belongs before clz
	sub	r3, r7, r3	; temp = maxlen_m1 - oState->Levels[oState->half_depth].mark
	bge	not_smaller	; if (depth < oState->half_depth2)

	clz	r5, r5
	add	r5, r5, #1
	sub	r3, r3, r5		; temp -= LOOKUP_FIRSTBLANK(dist0)
	
not_smaller
	cmp	r12, r3			; if (limit > temp)
	movgt	r12, r3			; limit = temp

not_between
	str	r12, [r1, #O_limit]	; lev->limit = limit
	
	subs	r8, r8, #1		; if (--nodes <= 0)
	bgt	for_loop
	
	str	r14, [r1, #O_mark]	; lev->mark = mark
	b	exit

for_break
	sub	r1, r1, #S_lev		; --lev
	sub	r6, r6, #1		; --depth

	ldr	r5, [r0, #O_stopdepth]

; Start POP_LEVEL(lev)

	ldr	r9, [r1, #O_comp+0*4]	; comp0 = lev->comp[0]
	ldr	r10, [r1, #O_dist+0*4]	; dist0 = lev->dist[0]
	mov	r11, #0			; newbit = 0

; End POP_LEVEL(lev)

	cmp	r6, r5			; while (depth > oState->stopdepth)
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
