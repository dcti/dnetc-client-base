@ Copyright distributed.net 1997-2002 - All Rights Reserved
@ For use in distributed.net projects only.
@ Any other distribution or use of this source violates copyright.
@
@ Author: Peter Teichmann <teichp@distributed.net>
@ $Id: ogrng-arm2-gnu.S,v 1.1 2009/01/02 21:03:24 teichp Exp $
@
@ Optimized for XScale:
@ * Intel really did a extremly bad job. They had better just taken the
@   StrongARM core and scaled to 0.18u. Consequences are:
@ * 3 cycle more for ldm/stm than sequence of ldr/str (SA 1)
@ * 5 cycle delay for mispredicted branches and data
@   processing with pc destination. Even unconditional branches need to
@   be predicted! Jesus! (SA 2/3)
@ * 8 cycle delay for load to pc (SA 4)
@ * predicted branch 1 cycle, mispredicted branch 5 cycles (SA taken 2/not
@   taken 1)
@ * avoid branch target buffer conflicts! The branch target buffer is direct
@   mapped and contains 128 entries. (problem does not exist on SA)
@ * 3 cycle result delay for ldr (SA 2)
@ * 2 cycle delay for data processing if result is used for
@   shift-by-immediate in next instruction (SA 1, as normal)
@
@ Stack:
@ 0   int *pnodes
@ 4   u16 *pchoose
@
@ Register:
@ r0  oState
@ r1  lev
@ r2
@ r3
@ r4
@ r5
@ r6  depth
@ r7  maxlen_m1
@ r8  nodes
@ r9  comp0
@ r10 dist0
@ r11 newbit
@ r12 limit
@ r14 mark

@ Stack
#define	O_pnodes	0
#define	O_pchoose	4

@ OgrState
#define	O_max 		0
#define	O_maxdepth	4
#define	O_maxdepthm1	8
#define	O_half_depth	12
#define	O_half_depth2	16
#define	O_startdepth	20
#define	O_stopdepth	24
#define	O_depth		28
#define	O_Levels	32

@ OgrLevel
#define	O_list		0
#define	O_dist		32
#define	O_comp		64
#define	O_mark		96
#define	O_limit		100
#define	S_lev		104

.text
	.align	5
	.global	ogr_cycle_256_arm2
ogr_cycle_256_arm2:
	stmdb	sp!, {r4, r5, r6, r7, r8, r9, r10, r11, r12, r14}
	sub	sp, sp, #2*4

	str	r1, [sp, #O_pnodes]	@ save pnodes
	str	r2, [sp, #O_pchoose]	@ save pchoose
	
	ldr	r8, [r1]		@ nodes = *pnodes
	
	ldr	r6, [r0, #O_depth]	@ depth = oState->depth
	
	mov	r2, #S_lev
	mul	r2, r6, r2
	add	r1, r0, #O_Levels
	add	r1, r1, r2		@ lev = &oState->Levels[depth]
	
	ldr	r7, [r0, #O_max]
	sub	r7, r7, #1		@ maxlen_m1 = oState->max - 1
	
@ Start SETUP_TOP_STATE(lev)
	
	ldr	r9, [r1, #O_comp]	@ comp0 = lev->comp[0]
	
	ldr	r11, [r0, #O_maxdepthm1]
	sub	r11, r6, r11
	mov	r11, r11, lsr#31	@ newbit = (depth < oState->maxdepthm1) ? 1 : 0
	
@ End SETUP_TOP_STATE(lev)
	
do_while_loop:
	ldr	r12, [r1, #O_limit]	@ limit = lev->limit
	ldr	r14, [r1, #O_mark]	@ mark = lev->mark

for_loop:
for_continue:
	cmp	r9, #0xfffffffe		@ if (comp0 < (SCALAR)~1)
	bhs	comp0_hs

comp0_lo:
	mvn	r5, r9
	.word	0xe16f5f15		@ clz	r5, r5
	add	r5, r5, #1		@ s=LOOKUP_FIRSTBLANK(comp0)
	
	add	r14, r14, r5
	cmp	r14, r12		@ if ((mark += s) > limit)
	bgt	for_break
	
@ Start COMP_LEFT_LIST_RIGHT(lev, s)
	add	pc, pc, r5, lsl#2	@ this is better for XScale because
	.space	2*4			@ it avoids the BTB conflict for
	b	cllr_1			@ b comp0_fi
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

@ COMP_LEFT_LIST_RIGHT macro, should be 2^N+/-1 instructions to avoid
@ BTB conflicts (presently 65)

.MACRO	COMP_LEFT_LIST_RIGHT N
	
	ldr	r2, [r1, #O_list+0*4]	@ equal for all cases!
	ldr	r3, [r1, #O_list+1*4]
	ldr	r4, [r1, #O_list+2*4]
	mov	r5, r11, lsl #32-\N
	orr	r5, r5, r2, lsr #\N
	str	r5, [r1, #O_list+0*4]	@ list[0] = list[0]>>N | newbit<<32-N
	
	ldr	r5, [r1, #O_list+3*4]
	mov	r2, r2, lsl #32-\N
	orr	r2, r2, r3, lsr #\N
	str	r2, [r1, #O_list+1*4]	@ list[1] = list[1]>>N | list[0]<<32-N
	
	ldr	r2, [r1, #O_list+4*4]
	mov	r3, r3, lsl #32-\N
	orr	r3, r3, r4, lsr #\N
	str	r3, [r1, #O_list+2*4]	@ list[2] = list[2]>>N | list[1]<<32-N
	
	ldr	r3, [r1, #O_list+5*4]
	mov	r4, r4, lsl #32-\N
	orr	r4, r4, r5, lsr #\N
	str	r4, [r1, #O_list+3*4]	@ list[3] = list[3]>>N | list[2]<<32-N
	
	ldr	r4, [r1, #O_list+6*4]
	mov	r5, r5, lsl #32-\N
	orr	r5, r5, r2, lsr #\N
	str	r5, [r1, #O_list+4*4]	@ list[4] = list[4]>>N | list[3]<<32-N
	
	ldr	r5, [r1, #O_list+7*4]
	mov	r2, r2, lsl #32-\N
	orr	r2, r2, r3, lsr #\N
	str	r2, [r1, #O_list+5*4]	@ list[5] = list[5]>>N | list[4]<<32-N
	
	mov	r3, r3, lsl #32-\N
	orr	r3, r3, r4, lsr #\N
	str	r3, [r1, #O_list+6*4]	@ list[6] = list[6]>>N | list[5]<<32-N
	
	mov	r4, r4, lsl #32-\N
	orr	r4, r4, r5, lsr #\N
	str	r4, [r1, #O_list+7*4]	@ list[7] = list[7]>>N | list[6]<<32-N
	
	ldr	r2, [r1, #O_comp+0*4]
	ldr	r3, [r1, #O_comp+1*4]
	ldr	r4, [r1, #O_comp+2*4]
	ldr	r5, [r1, #O_comp+3*4]
	mov	r2, r2, lsl #\N
	orr	r9, r2, r3, lsr #32-\N  @ comp0 = comp[0]<<N | comp[1]>>32-N
	str	r9, [r1, #O_comp+0*4]	@ comp[0] = comp0
	
	ldr	r2, [r1, #O_comp+4*4]
	mov	r3, r3, lsl #\N
	orr	r3, r3, r4, lsr #32-\N
	str	r3, [r1, #O_comp+1*4]	@ comp[1] = comp[1]<<N | comp[2]>>32-N
	
	ldr	r3, [r1, #O_comp+5*4]
	mov	r4, r4, lsl #\N
	orr	r4, r4, r5, lsr #32-\N
	str	r4, [r1, #O_comp+2*4]	@ comp[2] = comp[2]<<N | comp[3]>>32-N
	
	ldr	r4, [r1, #O_comp+6*4]
	mov	r5, r5, lsl #\N
	orr	r5, r5, r2, lsr #32-\N
	str	r5, [r1, #O_comp+3*4]	@ comp[3] = comp[3]<<N | comp[4]>>32-N
	
	ldr	r5, [r1, #O_comp+7*4]
	mov	r2, r2, lsl #\N
	orr	r2, r2, r3, lsr #32-\N
	str	r2, [r1, #O_comp+4*4]	@ comp[4] = comp[4]<<N | comp[5]>>32-N
	
	mov	r3, r3, lsl #\N
	orr	r3, r3, r4, lsr #32-\N
	str	r3, [r1, #O_comp+5*4]	@ comp[5] = comp[5]<<N | comp[6]>>32-N
	
	mov	r4, r4, lsl #\N
	orr	r4, r4, r5, lsr #32-\N
	str	r4, [r1, #O_comp+6*4]	@ comp[6] = comp[6]<<N | comp[7]>>32-N
	
	mov	r5, r5, lsl #\N
	str	r5, [r1, #O_comp+7*4]	@ comp[7] = comp[7]<<N
	
	mov	r11, #0			@ newbit = 0

	b	comp0_fi
	
.ENDM

cllr_31:
	COMP_LEFT_LIST_RIGHT 31
cllr_1:
	COMP_LEFT_LIST_RIGHT 1
cllr_2:
	COMP_LEFT_LIST_RIGHT 2
cllr_3:
	COMP_LEFT_LIST_RIGHT 3
cllr_4:
	COMP_LEFT_LIST_RIGHT 4
cllr_5:
	COMP_LEFT_LIST_RIGHT 5
cllr_6:
	COMP_LEFT_LIST_RIGHT 6
cllr_7:
	COMP_LEFT_LIST_RIGHT 7
cllr_8:
	COMP_LEFT_LIST_RIGHT 8
cllr_9:
	COMP_LEFT_LIST_RIGHT 9
cllr_10:
	COMP_LEFT_LIST_RIGHT 10
cllr_11:
	COMP_LEFT_LIST_RIGHT 11
cllr_12:
	COMP_LEFT_LIST_RIGHT 12
cllr_13:
	COMP_LEFT_LIST_RIGHT 13
cllr_14:
	COMP_LEFT_LIST_RIGHT 14
cllr_15:
	COMP_LEFT_LIST_RIGHT 15
cllr_16:
	COMP_LEFT_LIST_RIGHT 16
cllr_17:
	COMP_LEFT_LIST_RIGHT 17
cllr_18:
	COMP_LEFT_LIST_RIGHT 18
cllr_19:
	COMP_LEFT_LIST_RIGHT 19
cllr_20:
	COMP_LEFT_LIST_RIGHT 20
cllr_21:
	COMP_LEFT_LIST_RIGHT 21
cllr_22:
	COMP_LEFT_LIST_RIGHT 22
cllr_23:
	COMP_LEFT_LIST_RIGHT 23
cllr_24:
	COMP_LEFT_LIST_RIGHT 24
cllr_25:
	COMP_LEFT_LIST_RIGHT 25
cllr_26:
	COMP_LEFT_LIST_RIGHT 26
cllr_27:
	COMP_LEFT_LIST_RIGHT 27
cllr_28:
	COMP_LEFT_LIST_RIGHT 28
cllr_29:
	COMP_LEFT_LIST_RIGHT 29
cllr_30:
	COMP_LEFT_LIST_RIGHT 30
	.space	0*4
	
@ End COMP_LEFT_LIST_RIGHT(lev, s)

comp0_hs:
	add	r14, r14, #32		@ 
	cmp	r14, r12		@ if ((mark += SCALAR_BITS) > limit)
	bgt	for_break
	
	cmp	r9, #0xffffffff
	
@ Start COMP_LEFT_LIST_RIGHT_WORD(lev)

	ldr	r2, [r1, #O_list+0*4]
	str	r11, [r1, #O_list+0*4]	@ list[0] = newbit
	ldr	r3, [r1, #O_list+1*4]
	str	r2, [r1, #O_list+1*4]	@ list[1] = list[0]
	ldr	r2, [r1, #O_list+2*4]
	str	r3, [r1, #O_list+2*4]	@ list[2] = list[1]
	ldr	r3, [r1, #O_list+3*4]
	str	r2, [r1, #O_list+3*4]	@ list[3] = list[2]
	ldr	r2, [r1, #O_list+4*4]
	str	r3, [r1, #O_list+4*4]	@ list[4] = list[3]
	ldr	r3, [r1, #O_list+5*4]
	str	r2, [r1, #O_list+5*4]	@ list[5] = list[4]
	ldr	r2, [r1, #O_list+6*4]
	str	r3, [r1, #O_list+6*4]	@ list[6] = list[5]

	ldr	r9, [r1, #O_comp+1*4]	@ comp0 = comp[1]
	str	r2, [r1, #O_list+7*4]	@ list[7] = list[6]
	
	ldr	r2, [r1, #O_comp+2*4]
	ldr	r3, [r1, #O_comp+3*4]
	str	r9, [r1, #O_comp+0*4]	@ comp[0] = comp0
	ldr	r4, [r1, #O_comp+4*4]
	str	r2, [r1, #O_comp+1*4]	@ comp[1] = comp[2]
	ldr	r2, [r1, #O_comp+5*4]
	str	r3, [r1, #O_comp+2*4]	@ comp[2] = comp[3]
	ldr	r3, [r1, #O_comp+6*4]
	str	r4, [r1, #O_comp+3*4]	@ comp[3] = comp[4]
	ldr	r4, [r1, #O_comp+7*4]
	str	r2, [r1, #O_comp+4*4]	@ comp[4] = comp[5]
	mov	r11, #0			@ newbit = 0
	str	r3, [r1, #O_comp+5*4]	@ comp[5] = comp[6]
	str	r4, [r1, #O_comp+6*4]	@ comp[6] = comp[7]
	str	r11, [r1, #O_comp+7*4]	@ comp[7] = 0
	
@ End COMP_LEFT_LIST_RIGHT_WORD(lev)
	
	beq	for_continue

comp0_fi:
	ldr	r2, [r0, #O_maxdepthm1]
	str	r14, [r1, #O_mark]	@ lev->mark = mark
	cmp	r6, r2
	beq	exit
	
@ Start PUSH_LEVEL_UPDATE_STATE(lev)
	
.MACRO	PUSH_PART I
	
	ldr	r2, [r1, #O_list+\I*4]
	ldr	r3, [r1, #O_dist+\I*4]
	ldr	r4, [r1, #O_comp+\I*4]
	str	r2, [r1, #O_list+S_lev+\I*4]	@ list[lev+1] = list[lev]
	orr	r2, r2, r3
	str	r2, [r1, #O_dist+S_lev+\I*4]	@ dist[lev+1] = dist[lev] | list[lev+1]
	orr	r2, r2, r4
	str	r2, [r1, #O_comp+S_lev+\I*4]	@ comp[lev+1] = comp[lev] | dist[lev+1]
	
.ENDM
	
	ldr	r2, [r1, #O_list+0*4]		@ PUSH_PART 0 special
	ldr	r3, [r1, #O_dist+0*4]
	ldr	r4, [r1, #O_comp+0*4]
	str	r2, [r1, #O_list+S_lev+0*4]	@ list[lev+1] = list[lev]
	orr	r10, r2, r3			@ dist0
	str	r10, [r1, #O_dist+S_lev+0*4]	@ dist[lev+1] = dist[lev] | list[lev+1]
	orr	r9, r10, r4			@ comp0
	str	r9, [r1, #O_comp+S_lev+0*4]	@ comp[lev+1] = comp[lev] | dist[lev+1]
	
	PUSH_PART 1
	PUSH_PART 2
	PUSH_PART 3
	PUSH_PART 4
	PUSH_PART 5
	PUSH_PART 6
	PUSH_PART 7
	mov	r11, #1			@ newbit = 1
	
@End PUSH_LEVEL_UPDATE_STATE(lev)

	add	r1, r1, #S_lev		@ ++lev
	add	r6, r6, #1		@ ++depth
	
	ldr	r2, [sp, #O_pchoose]
	mov	r3, r10, lsr #16
	add	r3, r6, r3, lsl#5	@ (dist0>>16)*32+depth
	add	r2, r2, r3, lsl#1	@ pointer to halfword
	ldrb	r4, [r2, #0]
	ldrb	r12, [r2, #1]
	
	ldr	r3, [r0, #O_half_depth2]
	ldr	r2, [r0, #O_half_depth]
	orr	r12, r4, r12, lsl#8	@ limit = choose(dist0, depth)
	cmp	r6, r3			@ if (depth <= oState->half_depth2
	bgt	not_between
	cmp	r6, r2			@ && depth > oState->half_depth)
	ble	not_between
	
	mov	r4, #S_lev
	mul	r2, r4, r2
	add	r4, r0, #O_Levels+O_mark
	ldr	r2, [r4, r2]
	
	cmp	r6, r3
	sub	r2, r7, r2	@ temp = maxlen_m1 - oState->Levels[oState->half_depth].mark
	bge	not_smaller	@ if (depth < oState->half_depth2)

	mvn	r5, r10
	.word	0xe16f5f15		@ clz	r5, r5
	add	r5, r5, #1
	sub	r2, r2, r5		@ temp -= LOOKUP_FIRSTBLANK(dist0)
	
not_smaller:
	cmp	r12, r2			@ if (limit > temp)
	movgt	r12, r2			@ limit = temp

not_between:
	str	r12, [r1, #O_limit]	@ lev->limit = limit
	
	subs	r8, r8, #1		@ if (--nodes <= 0)
	bgt	for_loop
	
	str	r14, [r1, #O_mark]	@ lev->mark = mark
	b	exit

for_break:
	sub	r1, r1, #S_lev		@ --lev
	sub	r6, r6, #1		@ --depth

	ldr	r2, [r0, #O_stopdepth]

@ Start POP_LEVEL(lev)

	ldr	r9, [r1, #O_comp+0*4]	@ comp0 = lev->comp[0]
	ldr	r10, [r1, #O_dist+0*4]	@ dist0 = lev->dist[0]
	mov	r11, #0			@ newbit = 0

@ End POP_LEVEL(lev)

	cmp	r6, r2			@ while (depth > oState->stopdepth)
	bgt	do_while_loop

exit:

@ Start SAVE_FINAL_STATE(lev)

@ End SAVE_FINAL_STATE(lev)

	ldr	r2, [sp, #O_pnodes]
	ldr	r3, [r2]
	sub	r3, r3, r8
	str	r3, [r2]		@ *pnodes -= nodes
	
	mov	r0, r6			@ return depth
	
	add	sp, sp, #2*4
	ldmia	sp!, {r4, r5, r6, r7, r8, r9, r10, r11, r12, pc}
