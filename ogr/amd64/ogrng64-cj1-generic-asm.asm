;
; Assembly core for OGR-NG, 64bit generic asm version. Based on SSE-K8 assembly core (ogrng-cj1-sse-k8-asm.asm).
; $Id: ogrng64-cj1-generic-asm.asm,v 1.1 2009/08/08 18:26:19 snikkel Exp $
;
; Created by Craig Johnston (craig.johnston@dolby.com)
;
; 2009-04-26: Initial Generic 64 bit asm version.
;

%ifdef __NASM_VER__
	cpu	686	; Generic i686 instructions only - cmov is 686
%else
	cpu	686
	BITS	64
%endif

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
	[SECTION _DATA USE32 ALIGN=16 CLASS=DATA]
%else
	[SECTION .data]
%endif

%ifdef __OMF__ ; Borland and Watcom compilers/linkers
	[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
	[SECTION .text]
%endif

	global	_ogrng64_cycle_256_cj1_generic, ogrng64_cycle_256_cj1_generic

	%define CHOOSE_DIST_BITS	16     ; /* number of bits to take into account  */

_ogrng64_cycle_256_cj1_generic:
ogrng64_cycle_256_cj1_generic:

	%define worksize	50h

	%define work_depth					rsp+00h
	%define work_halfdepth				rsp+04h
	%define work_halfdepth2				rsp+08h
	%define work_nodes					rsp+0Ch
	%define work_maxlen_m1				rsp+10h
	%define work_maxdepth_m1			rsp+14h
	%define work_stopdepth				rsp+18h

	; 64 bit work elements
	%define work_half_depth_mark_addr	rsp+20h
	%define work_oState					rsp+28h
	%define work_pnodes					rsp+30h
	%define work_pchoose				rsp+38h
	%define work_oldrsp					rsp+40h

	; State Offsets
	%define oState_max			00h
	%define oState_maxdepthm1	08h
	%define oState_half_depth	0Ch
	%define oState_half_depth2	10h
	%define oState_stopdepth	18h
	%define oState_depth		1Ch
	%define oState_Levels		20h

; It's possible to put rbp (current level) a little forward and reference
; elements as 'rbp-nn' and 'rbp+nn' (with signed byte offsets) to avoid
; long command 'rbp+nnnnnnnn' (dword offset).
	%define rbp_shift		128	; may be up to 128
	%define sizeof_level	112	; (32*3+8+8)
	%define level_list		00h
	%define level_dist		20h
	%define level_comp		40h
	%define level_mark		60h
	%define level_limit		64h

%define cur(el, index)   [rbp+level_ %+ el + ((index)*8) - rbp_shift]
%define next(el, index)  [rbp+sizeof_level+level_ %+ el + ((index)*8) - rbp_shift]

	push	rsi
	push	rdi
	push	rbx
	push	rbp
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, worksize

%ifdef _WINDOWS
	; Switch to linux calling convention
	mov	rdi, rcx
	mov	rsi, rdx
	mov	rdx, r8
%endif

	; Align stack to 16 bytes
	mov	rcx, rsp
	and	rsp, 0xFFFFFFF0
	mov	[work_oldrsp], rcx

	; write the paramters in the aligned work space
	mov	[work_oState], rdi
	mov	[work_pnodes], rsi
	mov	[work_pchoose], rdx

	mov	ecx, [rdi+oState_depth]
	mov	[work_depth], ecx	; depth = oState->depth (cached for SETUP_TOP_STATE)

	imul	eax, ecx, sizeof_level
	lea	rbp, [rax+rdi+oState_Levels+rbp_shift]	; lev = &oState->Levels[oState->depth]
	mov	eax, [rsi]
	mov	[work_nodes], eax	; nodes = *pnodes

	mov	eax, [rdi+oState_half_depth]
	mov	[work_halfdepth], eax	; halfdepth = oState->half_depth

	; get address of oState->Levels[oState->half_depth].mark
	; value of this var can be changed during crunching, but addr is const
	imul	eax, sizeof_level
	lea	rax, [rax+rdi+oState_Levels+level_mark]
	mov	[work_half_depth_mark_addr], rax

	mov	eax, [rdi+oState_half_depth2]
	mov	[work_halfdepth2], eax	; halfdepth2 = oState->half_depth2

	mov	eax, [rdi+oState_max]
	dec	eax
	mov	[work_maxlen_m1], eax	; maxlen_m1 = oState->max - 1

%define tempa	r8
%define tempb	r9
%define tempc	r10
%define tempd	r11

%define comp0	r12
%define comp1	r13
%define comp2	r14
%define comp3	r15

%define newbit	rsi

	; SETUP_TOP_STATE(lev);
	; !!!
	mov	comp0, cur(comp, 0)	; SCALAR comp0 = lev->comp[0];
	mov	comp1, cur(comp, 1)
	mov	comp2, cur(comp, 2)
	mov	comp3, cur(comp, 3)

	; int newbit = (depth < oState->maxdepthm1) ? 1 : 0;
	xor	eax, eax
	cmp	ecx, [rdi+oState_maxdepthm1]	; depth still in ecx
	setl	al
	mov	newbit, rax

	; split loop header
	mov	ebx, [rbp+level_mark-rbp_shift]	; mark  = lev->mark;
	mov	edi, [rbp+level_limit-rbp_shift]

; Jump probabilies calculated from summary statistic of 'dnetc -test'.
; Main loop was entered 0x0B5ACBE2 times. For each jump, we measured
; number of times this jump was reached (percents from main loop
; means importance of this code path) and probability of this jump to
; be taken (for branch prediction of important jumps).

	align	16
do_loop_split:
for_loop:

	; REGISTER - end
	; eax = inverse shift amount (location of 0)
	; ecx = shift amount (ecx - eax)

	;      if (comp0 < (SCALAR)~1) {

	mov	rcx, comp0
	cmp	rcx, -2
	jnb	comp0_ge_fffe		; ENTERED: 0x????????(??%), taken: ??%
	not	rcx
	bsr	rax, rcx
	mov	ecx, 64
	sub	ecx, eax		; s = ecx-bsr

found_shift:
	; REGISTER - start
	; eax = inverse shift amount (location of 0)
	; ecx = shift amount (64 - eax)

	;        if ((mark += s) > limit) {
	;          break;
	;        }
	add	ebx, ecx
	cmp	ebx, edi ; limit (==lev->limit)
	jg	break_for		; ENTERED: 0x07A5D5F7(67.35%), taken 30-34%

	;        COMP_LEFT_LIST_RIGHT(lev, s);
	; !!!

	mov	edx, ecx
	; newbit + list goes right

	mov	tempa, cur(list, 0)
	mov	tempc, cur(list, 1)
	mov	tempb, tempa	; list[0] cached in tempb
	mov	tempd, tempc	; list[1] cached in tempd

;	mov	ecx, edx
	shr	tempa, cl
	shr	tempc, cl
	mov	ecx, eax
	shl	newbit, cl	; insert newbit
	shl	tempb, cl

	or	newbit, tempa
	or	tempb, tempc
	mov	cur(list, 0), newbit
	mov	cur(list, 1), tempb


	mov	newbit, cur(list, 2)
	mov	tempb, cur(list, 3)
	mov	tempa, newbit	; list[2] cached in tempa

;	mov	ecx, eax
	shl	tempd, cl
	shl	tempa, cl
	mov	ecx, edx
	shr	newbit, cl
	shr	tempb, cl

	or	tempd, newbit
	or	tempa, tempb
	mov	cur(list, 2), tempd
	mov	cur(list, 3), tempa

	; comp goes left

	mov	tempa, comp1
	mov	tempb, comp2
	mov	tempc, comp3

;	mov	ecx, edx
	shl	comp0, cl
	shl	comp1, cl
	shl	comp2, cl
	shl	comp3, cl

	mov	ecx, eax
	shr	tempa, cl
	shr	tempb, cl
	shr	tempc, cl

	or	comp0, tempa
	or	comp1, tempb
	or	comp2, tempc

after_if:
	; ebx = mark

	;      if (depth == oState->maxdepthm1) {
	;        goto exit;         /* Ruler found */
	;      }
	mov	rax, [work_oState]
	mov	eax, [rax+oState_maxdepthm1]
	cmp	eax, [work_depth]
	je	exit			; ENTERED: 0x0513FD1B(44.72%), taken: 0%

	;      PUSH_LEVEL_UPDATE_STATE(lev);
	; !!!
	; **   LIST[lev+1] = LIST[lev]
	; **   DIST[lev+1] = (DIST[lev] | LIST[lev+1])
	; **   COMP[lev+1] = (COMP[lev] | DIST[lev+1])
	; **   newbit = 1;

	mov	rcx, cur(dist, 0)
	mov	rax, cur(dist, 1)
	mov	rsi, cur(dist, 2)	; Safe to use rsi since newbit is going to be set later
	mov	rdx, cur(dist, 3)

	mov	tempa, cur(list, 0)
	mov	next(list, 0), tempa
	mov	cur(comp, 0), comp0
	or	rcx, tempa
	mov	next(dist, 0), rcx	; dist0 in rcx, used later
	or	comp0, rcx

	mov	tempb, cur(list, 1)
	mov	next(list, 1), tempb
	mov	cur(comp, 1), comp1
	or	rax, tempb
	mov	next(dist, 1), rax
	or	comp1, rax

	mov	tempc, cur(list, 2)
	mov	next(list, 2), tempc
	mov	cur(comp, 2), comp2
	or	rsi, tempc
	mov	next(dist, 2), rsi
	or	comp2, rsi

	mov	tempd, cur(list, 3)
	mov	next(list, 3), tempd
	mov	cur(comp, 3), comp3
	or	rdx, tempd
	mov	next(dist, 3), rdx
	or	comp3, rdx

;	!! delay init !!
;	newbit = 1

	;      lev->mark = mark;
	mov	[rbp+level_mark-rbp_shift], ebx
	mov	[rbp+level_limit-rbp_shift], edi

	;      lev++;
	add	rbp, sizeof_level

	;      depth++;
	mov	edx, [work_depth]
	add	edx, 1
	mov	[work_depth], edx

	; /* Compute the maximum position for the next level */
	; #define choose(dist,seg) pchoose[(dist >> (SCALAR_BITS-CHOOSE_DIST_BITS)) * 32 + (seg)]
	; limit = choose(dist0, depth);

	mov	rsi, [work_pchoose]	; Safe to use rsi since newbit is going to be set later
	mov	rax, rcx ; dist 0
	shr	rax, 64 - CHOOSE_DIST_BITS
	shl	eax, 5
	add	eax, edx		; depth
	movzx	edi, word [rsi+rax*2]

	;      if (depth > oState->half_depth && depth <= oState->half_depth2) {
	;;;      if (depth > halfdepth && depth <= halfdepth2) {
	mov	eax, [work_depth]
	cmp	eax, [work_halfdepth2]
	jle	continue_if_depth	; ENTERED: 0x0513FD14(44.72%), NOT taken 97.02%

skip_if_depth:

	mov	newbit, 1

	;      if (--nodes <= 0) {

	dec	dword [work_nodes]
	jg	for_loop	; ENTERED: 0x0513FD14(44.72%), taken 99.99%

	;        goto exit;
	jmp	exit

	align	16
continue_if_depth:
	cmp	eax, [work_halfdepth]
	jle	skip_if_depth		; ENTERED: 0x????????(??.??%), taken 0.5%

;        int temp = maxlen_m1 - oState->Levels[oState->half_depth].mark;
;;        int temp = oState->max - 1 - oState->Levels[halfdepth].mark;

	mov	rax, [work_half_depth_mark_addr]
	mov	edx, [work_maxlen_m1]
	sub	edx, [rax]

;        if (depth < oState->half_depth2) {
	mov	eax, [work_depth]
	cmp	eax, [work_halfdepth2]
	jge	update_limit_temp	; ENTERED: 0x00267D08(1.32%), taken 78.38%

;          temp -= LOOKUP_FIRSTBLANK(dist0); // "33" version
;;;        temp -= LOOKUP_FIRSTBLANK(dist0 & -((SCALAR)1 << 32));
;;;        (we already have upper part of dist0 in ecx)

	not	rcx
	mov	eax, -1
	bsr	rcx, rcx
	cmovz	ecx, eax

	add	edx, ecx
	sub	edx, 64

update_limit_temp:
;        if (limit > temp) {
;          limit = temp;
;        }

	cmp	edi, edx
	cmovg	edi, edx
	jmp	skip_if_depth

	align	16
comp0_ge_fffe:
	;      else {         /* s >= SCALAR_BITS */

	;        if ((mark += SCALAR_BITS) > limit) {
	;          break;
	;        }
	add	ebx, 64
	cmp	ebx, edi ; limit (==lev->limit)
	jg	break_for		; ENTERED: 0x03B4F5EB(32.64%), taken 66.70%

	;      COMP_LEFT_LIST_RIGHT_WORD(lev);
	;      continue;

	; COMP_LEFT_LIST_RIGHT_WORD(lev);
	; !!!

	; newbit + list goes right
	mov	tempa, cur(list, 0)
	mov	tempb, cur(list, 1)
	mov	tempc, cur(list, 2)
	mov	cur(list, 0), newbit
	mov	cur(list, 1), tempa
	mov	cur(list, 2), tempb
	mov	cur(list, 3), tempc

	xor	newbit, newbit		; newbit = 0

	; comp goes left			; 01 23 45 67 --
	mov	comp0, comp1
	mov	comp1, comp2
	mov	comp2, comp3
	xor	comp3, comp3

	cmp	rcx, -1
	je	for_loop
	jmp	after_if		; ENTERED: 0x013BFDCE(10.87%), taken 97.10%

	align	16
break_for:

	mov	rax, [work_oState]

	;    lev--;
	sub	rbp, sizeof_level

	;    depth--;
	dec	dword [work_depth]

	;    POP_LEVEL(lev);
	; !!!
	mov	comp0, cur(comp, 0)	; SCALAR comp0 = lev->comp[0];
	mov	comp1, cur(comp, 1)
	mov	comp2, cur(comp, 2)
	mov	comp3, cur(comp, 3)

	xor	newbit, newbit	;      newbit = 0;

	;  } while (depth > oState->stopdepth);
	mov	eax, [rax+oState_stopdepth]

	; split loop header
	mov	ebx, [rbp+level_mark-rbp_shift]	; mark  = lev->mark;
	mov	edi, [rbp+level_limit-rbp_shift]

	cmp	eax, [work_depth]
	jl	do_loop_split		; ENTERED: 0x0513FCC2(44.72%), taken 99.99%

exit:
	;  SAVE_FINAL_STATE(lev);
	; !!!
	mov	cur(comp, 0), comp0
	mov	cur(comp, 1), comp1
	mov	cur(comp, 2), comp2
	mov	cur(comp, 3), comp3

	;      lev->mark = mark;
	mov	[rbp+level_mark-rbp_shift], ebx
	mov	[rbp+level_limit-rbp_shift], edi

	mov	rbx, [work_pnodes]	; *pnodes -= nodes;
	mov	eax, [work_nodes]
	sub	[rbx], eax

	mov	eax, [work_depth]	; return depth;

	mov	rsp, [work_oldrsp]
	add	rsp, worksize
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbp
	pop	rbx
	pop	rdi
	pop	rsi
	ret
