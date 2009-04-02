;
; Assembly core for OGR-NG, SSE2 version. Based on MMX assembly core (ogrng-b-asm-rt.asm).
; $Id: ogrng-cj1-sse2-asm.asm,v 1.1 2009/04/02 12:14:33 stream Exp $
;
; NOTE: Requires *ALL* DIST, LIST and COMP bitmaps to start on 16 byte boundaries.
;
; Created by Craig Johnston (craig.johnston@dolby.com)
;
; 2009-03-30: Initial SSE2 version. Converted bitmap operations from MMX to SSE2
;             Optimised register usage to keep mark, depth and limit in general registers
;             Precomputed some of the memory offset calculations being done in the main loop
;             General optimisation of the math to reduce instructions
;
; Possible improvements:
; * Align stack to 16 byte boundary
; * Vary choose array cache size
; * Arrange the choose array for better cache hits, depth as most significant index perhaps?

%ifdef __NASM_VER__
	cpu	p4	; NASM doesnt know feature flags but accepts sse2 in p4
%else
	cpu	586 mmx sse2	; mmx is not part of i586
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

	global	_ogr_cycle_256_cj1_sse2, ogr_cycle_256_cj1_sse2

	%define CHOOSE_DIST_BITS	16     ; /* number of bits to take into account  */

_ogr_cycle_256_cj1_sse2:
ogr_cycle_256_cj1_sse2:

	%define regsavesize	10h	; 4 registers saved
	%define worksize	30h

	%define work_halfdepth				esp+00h
	%define work_halfdepth2				esp+04h
	%define work_nodes					esp+08h
	%define work_maxlen_m1				esp+0Ch
	%define work_half_depth_mark_addr	esp+10h
	%define work_maxdepth_m1			esp+14h
	%define work_stopdepth				esp+18h

	%define param_oState	esp+regsavesize+worksize+04h
	%define param_pnodes	esp+regsavesize+worksize+08h
	%define param_pchoose	esp+regsavesize+worksize+0Ch

	%define oState_max		00h

	%define oState_maxdepthm1	08h
	%define oState_half_depth	0Ch
	%define oState_half_depth2	10h

	%define oState_stopdepth	18h
	%define oState_depth		1Ch
	%define oState_Levels		20h

; It's possible to put ebp (current level) a little forward and reference
; elements as 'ebp-nn' and 'ebp+nn' (with signed byte offsets) to avoid
; long command 'ebp+nnnnnnnn' (dword offset).
	%define ebp_shift		128	; may be up to 128
	%define sizeof_level	112	; (32*3+8 + 8)
	%define level_list		00h
	%define level_dist		20h
	%define level_comp		40h
	%define level_mark		60h
	%define level_limit		64h

%define cur(el, index)   [ebp+level_ %+ el + ((index)*8) - ebp_shift]
%define next(el, index)  [ebp+sizeof_level+level_ %+ el + ((index)*8) - ebp_shift]

	push	ebx
	push	esi
	push	edi
	push	ebp			; note: regsavesize bytes pushed
	sub	esp, worksize

	mov	ebx, [param_oState]
	mov	edx, [ebx+oState_depth]
	imul	eax, edx, sizeof_level
	lea	ebp, [eax+ebx+oState_Levels+ebp_shift]	; lev = &oState->Levels[oState->depth]
	mov	eax, [param_pnodes]
	mov	eax, [eax]
	mov	[work_nodes], eax	; nodes = *pnodes

	mov	eax, [ebx+oState_half_depth]
	mov	[work_halfdepth], eax	; halfdepth = oState->half_depth
	; get address of oState->Levels[oState->half_depth].mark
	; value of this var can be changed during crunching, but addr is const
	imul	eax, sizeof_level
	lea	eax, [eax+ebx+oState_Levels+level_mark]
	mov	[work_half_depth_mark_addr], eax

	mov	eax, [ebx+oState_half_depth2]
	mov	[work_halfdepth2], eax	; halfdepth2 = oState->half_depth2

	mov	eax, [ebx+oState_max]
	sub	eax, 1
	mov	[work_maxlen_m1], eax	; maxlen_m1 = oState->max - 1

	mov	eax, [param_oState]
	mov	eax, [eax+oState_maxdepthm1]
	mov	[work_maxdepth_m1], eax

	mov	eax, [param_oState]
	mov	eax, [eax+oState_stopdepth]
	mov	[work_stopdepth], eax

%define mm_0		mm0
%define mm_1		mm1
%define mm_2		mm2
%define mm_newbit	mm3
%define mm_4		mm4
%define mm_5		mm5
%define mm_6		mm6
%define mm_7		mm7

%define xmm_temp_s	xmm0	; Shared by newbit
%define xmm_temp_ss	xmm1
%define xmm_list0	xmm2
%define xmm_list2	xmm3
%define xmm_comp0	xmm4
%define xmm_comp2	xmm5
%define xmm_temp_a	xmm6	; holds dist0 after PUSH_LEVEL_UPDATE_STATE
%define xmm_temp_b	xmm7

	; SETUP_TOP_STATE(lev);
	; !!!
	movdqa	xmm_comp0, cur(comp, 0)
	movdqa	xmm_comp2, cur(comp, 2)

	; int newbit = (depth < oState->maxdepthm1) ? 1 : 0;
	xor	eax, eax
	cmp	edx, [ebx+oState_maxdepthm1]
	setl	al
	movd	mm_newbit, eax

	; mm0..mm3 = comp
	; mm4 = newbit

	; split loop header
	mov	ebx, [ebp+level_mark-ebp_shift]	; mark  = lev->mark;
	mov	edi, [ebp+level_limit-ebp_shift]

	align	16

; Jump probabilies calculated from summary statistic of 'dnetc -test'.
; Main loop was entered 0x0B5ACBE2 times. For each jump, we measured
; number of times this jump was reached (percents from main loop
; means importance of this code path) and probability of this jump to
; be taken (for branch prediction of important jumps).

	; REGISTER - globals
	; ebx = mark
	; edi = limit
	; edx = work depth
	; ebp = stack location

do_loop_split:
	movdqa	xmm_list0, cur(list, 0)
	movdqa	xmm_list2, cur(list, 2)

for_loop:
	; REGISTER - end
	; esi = comp0 low/high
	; eax = inverse shift amount (location of 0)
	; ecx = shift amount (ecx - eax)

	pshuflw	xmm_temp_b, xmm_comp0, 11101110b

	;      if (comp0 < (SCALAR)~1) {

	mov	ecx, 32
	movd	esi, xmm_temp_b
	xor	esi, 0FFFFFFFFh		; implies 'not esi'
	je	use_high_word		; ENTERED: 0x????????(??%), taken: ??%
	bsr	eax, esi
	sub	ecx, eax		; s = ecx-bsr
	add	eax, 32	; = ss
	jmp found_shift

use_high_word:
	movd	esi, xmm_comp0
	cmp	esi, 0FFFFFFFEh
	jnb	comp0_ge_fffe		; ENTERED: 0x????????(??%), taken: ??%
	not	esi
	mov	ecx, 64			; add ecx, 32 ??? shorter but no gain
	bsr	eax, esi
	sub	ecx, eax		; s = ecx-bsr
	jmp found_shift

	align 16
found_shift:
	; REGISTER - start
	; eax = inverse shift amount (location of 0)
	; ecx = shift amount (ecx - eax)

	;        if ((mark += s) > limit) {
	;          break;
	;        }
	add	ebx, ecx
	cmp	ebx, edi	; limit (==lev->limit)
	jg	break_for		; ENTERED: 0x07A5D5F7(67.35%), taken 30-34%

	;        COMP_LEFT_LIST_RIGHT(lev, s);
	; !!!

	movd	xmm_temp_s, ecx

	; newbit + list goes right and comp goes left

	; copy of list for shifting left
	movdqa	xmm_temp_a, xmm_list0
	psrlq	xmm_list0, xmm_temp_s
	movdqa	xmm_temp_b, xmm_list2

	movd	xmm_temp_ss, eax

	psrlq	xmm_list2, xmm_temp_s

	; WARNING xmm_temp_s now contains newbit
	movq2dq	xmm_temp_s, mm_newbit

	psllq	xmm_temp_a, xmm_temp_ss
	psllq	xmm_temp_b, xmm_temp_ss
	psllq	xmm_temp_s, xmm_temp_ss

	shufpd	xmm_temp_s, xmm_temp_a, 0	; select Low(a), Low(s)
	shufpd	xmm_temp_a, xmm_temp_b, 1	; select Low(b), High(a)

	por	xmm_list0, xmm_temp_s
	por	xmm_list2, xmm_temp_a

	; comp goes left

	; xmm_temp_s restored to normal purpose
	movd	xmm_temp_s, ecx
	movdqa	xmm_temp_a, xmm_comp0
	movdqa	xmm_temp_b, xmm_comp2

	psllq	xmm_comp0, xmm_temp_s
	psllq	xmm_comp2, xmm_temp_s
	xorpd	xmm_temp_s, xmm_temp_s	;using xmm_temp_s as a source of 0's

	psrlq	xmm_temp_a, xmm_temp_ss
	psrlq	xmm_temp_b, xmm_temp_ss

	shufpd	xmm_temp_a, xmm_temp_b, 1	; select Low(b), High(a)
	shufpd	xmm_temp_b, xmm_temp_s, 1	; select Low(s), High(b)

	por	xmm_comp0, xmm_temp_a
	por	xmm_comp2, xmm_temp_b

	pxor	mm_newbit, mm_newbit		; newbit = 0

after_if:
	; REGISTER - usage
	; eax = temp

	;      if (depth == oState->maxdepthm1) {
	;        goto exit;         /* Ruler found */
	;      }
	mov	eax, [work_maxdepth_m1]
	cmp	eax, edx
	je	exit			; ENTERED: 0x0513FD1B(44.72%), taken: 0%

	;      PUSH_LEVEL_UPDATE_STATE(lev);
	; !!!
	; **   LIST[lev+1] = LIST[lev]
	; **   DIST[lev+1] = (DIST[lev] | LIST[lev+1])
	; **   COMP[lev+1] = (COMP[lev] | DIST[lev+1])
	; **   newbit = 1;

	; Save our loaded values
	movdqa	xmm_temp_a, cur(dist, 0)
	movdqa	xmm_temp_b, cur(dist, 2)
	movdqa	cur(list, 0), xmm_list0
	movdqa	cur(list, 2), xmm_list2

	; **   LIST[lev+1] = LIST[lev]	; No need as we keep list in registers
;	movdqa	next(list, 0), xmm_list0
;	movdqa	next(list, 2), xmm_list2

	; **   DIST[lev+1] = (DIST[lev] | LIST[lev+1])
	movdqa	cur(comp, 0), xmm_comp0
	por	xmm_temp_a, xmm_list0
	movdqa	cur(comp, 2), xmm_comp2
	por	xmm_temp_b, xmm_list2
	movdqa	next(dist, 0), xmm_temp_a
	por	xmm_comp0, xmm_temp_a
	movdqa	next(dist, 2), xmm_temp_b

	; **   COMP[lev+1] = (COMP[lev] | DIST[lev+1])
	por	xmm_comp2, xmm_temp_b
;	movdqa	next(comp, 0), xmm_comp0	; No need as we keep comp in registers
;	movdqa	next(comp, 2), xmm_comp2

	pshuflw	xmm_temp_a, xmm_temp_a, 11101110b	; Fill xmm_temp_a with dist0

;	!! delay init !!
;	newbit = 1

	;      lev->mark = mark;
	mov	[ebp+level_mark-ebp_shift], ebx
	mov	[ebp+level_limit-ebp_shift], edi

	;      lev++;
	add	ebp, sizeof_level

	;      depth++;
	add	edx, 1

	; /* Compute the maximum position for the next level */
	; #define choose(dist,seg) pchoose[(dist >> (SCALAR_BITS-CHOOSE_DIST_BITS)) * 32 + (seg)]
	; limit = choose(dist0, depth);

	; REGISTER - usage
	; eax = temp
	; esi = choose array mem location

	movd	eax, xmm_temp_a ; dist 0
	mov	esi, [param_pchoose]
	shr	eax, CHOOSE_DIST_BITS
	shl	eax, 5
	add	eax, edx		; depth
	movzx	edi, word [esi+eax*2]

	;      if (depth > oState->half_depth && depth <= oState->half_depth2) {
	;;;      if (depth > halfdepth && depth <= halfdepth2) {

	cmp	edx, [work_halfdepth2]
	jle	continue_if_depth	; ENTERED: 0x0513FD14(44.72%), NOT taken 97.02%

skip_if_depth:
	; REGISTER - usage
	; eax = temp

	mov	eax, 1		; newbit = 1 (delayed)
	movd	mm_newbit, eax

	;      if (--nodes <= 0) {

	sub	dword [work_nodes], 1
	jg	for_loop	; ENTERED: 0x0513FD14(44.72%), taken 99.99%

	;        goto exit;
	jmp	exit

	align	16

continue_if_depth:
	; REGISTER - usage
	; ecx = dist0
	; esi = temp
	; eax = temp

	cmp	edx, [work_halfdepth]
	jle	skip_if_depth		; ENTERED: 0x????????(??.??%), taken 0.5%

;        int temp = maxlen_m1 - oState->Levels[oState->half_depth].mark;
;;        int temp = oState->max - 1 - oState->Levels[halfdepth].mark;

	mov	esi, [work_maxlen_m1]
	mov	eax, [work_half_depth_mark_addr]
	sub	esi, [eax]

;        if (depth < oState->half_depth2) {
	mov	eax, edx
	cmp	eax, [work_halfdepth2]
	jge	update_limit_temp	; ENTERED: 0x00267D08(1.32%), taken 78.38%

;          temp -= LOOKUP_FIRSTBLANK(dist0); // "33" version
;;;        temp -= LOOKUP_FIRSTBLANK(dist0 & -((SCALAR)1 << 32));

	movd	ecx, xmm_temp_a		; move dist0 in ecx
	xor	ecx, -1		; "not ecx" does not set flags!
	mov	eax, -1
	je	skip_bsr	; ENTERED: 0x00085254(0.29%), taken 0.00%
	bsr	eax, ecx
skip_bsr:
	add	esi, eax
	sub	esi, 32

update_limit_temp:
;        if (limit > temp) {
;          limit = temp;
;        }

	cmp	edi, esi
	jl	limitok		; ENTERED: 0x00267D08(1.32%), taken 17.35%
	mov	edi, esi
limitok:
	jmp	skip_if_depth

	align	16

comp0_ge_fffe:
	; REGISTER - start
	; esi = comp0 (high)

	;      else {         /* s >= SCALAR_BITS */

	;        if ((mark += SCALAR_BITS) > limit) {
	;          break;
	;        }
	add	ebx, 64
	cmp	ebx, edi ; limit (==lev->limit)
	jg	break_for		; ENTERED: 0x03B4F5EB(32.64%), taken 66.70%

	;        if (comp0 == ~0u) {
	;          COMP_LEFT_LIST_RIGHT_WORD(lev);
	;          continue;
	;        }
	;        COMP_LEFT_LIST_RIGHT_WORD(lev);

	cmp	esi,0ffffffffH		; if (comp0 == ~0u)

	; COMP_LEFT_LIST_RIGHT_WORD(lev);
	; !!!

	shufpd	xmm_list2, xmm_list2, 1	; swaps high and low
	movq2dq	xmm_temp_s, mm_newbit
	movhlps	xmm_list2, xmm_list0

	; list0 = Low(List0), newbit
	punpcklqdq	xmm_list0, xmm_temp_s	; shift newbit into top half
	shufpd	xmm_list0, xmm_list0, 1	; swaps high and low

	; comp left
	movlhps	xmm_temp_b, xmm_comp2	; Shift Low(comp2) into High(b)
	pxor	xmm_temp_a, xmm_temp_a
	punpckhqdq	xmm_comp0, xmm_temp_b
	pxor	mm_newbit, mm_newbit	; Clear newbit
	punpckhqdq	xmm_comp2, xmm_temp_a

	je	for_loop		; ENTERED: 0x013BFDCE(10.87%), taken 97.10%
	jmp	after_if

	align	16
break_for:

	;    lev--;
	sub	ebp, sizeof_level
	;    depth--;
	sub	edx, 1
	;    POP_LEVEL(lev);
	; !!!

	movdqa	xmm_comp0, cur(comp, 0)
	movdqa	xmm_comp2, cur(comp, 2)

	pxor	mm_newbit, mm_newbit	;      newbit = 0;

	;  } while (depth > oState->stopdepth);
	mov	eax, [work_stopdepth]
	cmp	eax, edx

	; split loop header
	mov	ebx, [ebp+level_mark-ebp_shift]	; mark  = lev->mark;
	mov	edi, [ebp+level_limit-ebp_shift]

	jl	do_loop_split		; ENTERED: 0x0513FCC2(44.72%), taken 99.99%

exit:
	;  SAVE_FINAL_STATE(lev);
	; !!!

	movdqa	cur(list, 0), xmm_list0
	movdqa	cur(list, 2), xmm_list2
	movdqa	cur(comp, 0), xmm_comp0
	movdqa	cur(comp, 2), xmm_comp2

	;      lev->mark = mark;
	mov	[ebp+level_mark-ebp_shift], ebx
	mov	[ebp+level_limit-ebp_shift], edi

	mov	ebx, [param_pnodes]	; *pnodes -= nodes;
	mov	eax, [work_nodes]
	sub	[ebx], eax

	mov	eax, edx	; return depth;

	add	esp, worksize
	pop	ebp
	pop	edi
	pop	esi
	pop	ebx
	emms
	ret
