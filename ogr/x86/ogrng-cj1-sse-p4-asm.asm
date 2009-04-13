;
; Assembly core for OGR-NG, SSE version tuned for P4. Based on MMX assembly core (ogrng-b-asm-rt.asm).
; $Id: ogrng-cj1-sse-p4-asm.asm,v 1.1 2009/04/13 03:58:21 stream Exp $
;
; Created by Craig Johnston (craig.johnston@dolby.com)
;
; 2009-04-12: Initial SSE version. Tuned for Pentium 4.
;             Using 16 byte aligned bitmaps
;             Rewrote lookup of first zero bit
;             Changed mmx movq and shuffle into a sse shuffle and moved calculation to before the jump into "for_loop"
;             Minor general optimisations, removal of redundant instructions
;             Changed some branches into conditional moves
;

%ifdef __NASM_VER__
	cpu	p3	; NASM doesnt know feature flags but accepts mmx and sse in p3
%else
	cpu	586 mmx sse; mmx and sse are not part of i586
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

	global	_ogr_cycle_256_cj1_sse_p4, ogr_cycle_256_cj1_sse_p4

	%define CHOOSE_DIST_BITS	16     ; /* number of bits to take into account  */

_ogr_cycle_256_cj1_sse_p4:
ogr_cycle_256_cj1_sse_p4:

	%define regsavesize	10h	; 4 registers saved
	%define worksize	20h

	%define work_depth					esp+00h
	%define work_halfdepth				esp+04h
	%define work_halfdepth2				esp+08h
	%define work_nodes					esp+0Ch
	%define work_maxlen_m1				esp+10h
	%define work_half_depth_mark_addr	esp+14h

	%define	param_oState	esp+regsavesize+worksize+04h
	%define param_pnodes	esp+regsavesize+worksize+08h
	%define param_pchoose	esp+regsavesize+worksize+0Ch

	%define oState_max		00h

	%define oState_maxdepthm1	08h
	%define oState_half_depth	0Ch
	%define oState_half_depth2	10h

	%define oState_stopdepth	18h
	%define	oState_depth		1Ch
	%define oState_Levels		20h

; It's possible to put ebp (current level) a little forward and reference
; elements as 'ebp-nn' and 'ebp+nn' (with signed byte offsets) to avoid
; long command 'ebp+nnnnnnnn' (dword offset).
	%define ebp_shift		128	; may be up to 128
	%define sizeof_level	112	; (32*3+8+8)
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
	mov	ecx, [ebx+oState_depth]
	mov	[work_depth], ecx	; depth = oState->depth (cached for SETUP_TOP_STATE)
	imul	eax, ecx, sizeof_level
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
	dec	eax
	mov	[work_maxlen_m1], eax	; maxlen_m1 = oState->max - 1

%define mm_comp0	mm0
%define mm_comp1	mm1
%define mm_comp2	mm2
%define mm_newbit	mm3
%define mm_temp_a	mm4
%define mm_temp_b	mm5
%define mm_temp_s	mm6
%define mm_temp_ss	mm7

	; SETUP_TOP_STATE(lev);
	; !!!
	movq	mm_comp0, cur(comp, 0)	; SCALAR comp0 = lev->comp[0];
	movq	mm_comp1, cur(comp, 1)
	movq	mm_comp2, cur(comp, 2)
	; movq	mm3, cur(comp, 3)	; comp[3] not cached!
	; int newbit = (depth < oState->maxdepthm1) ? 1 : 0;
	xor	eax, eax
	cmp	ecx, [ebx+oState_maxdepthm1]	; depth still in ecx
	setl	al
	movd	mm_newbit, eax

	; mm0..mm3 = comp
	; mm4 = newbit

	; Prepare ecx, esi for find leading zero
	pshufw	mm_temp_a, mm_comp0, 11101110b
	movd	ecx, mm_temp_a
	movd	esi, mm_comp0

	; split loop header
	mov	ebx, [ebp+level_mark-ebp_shift]	; mark  = lev->mark;
	mov	edi, [ebp+level_limit-ebp_shift]

	align	16

; Jump probabilies calculated from summary statistic of 'dnetc -test'.
; Main loop was entered 0x0B5ACBE2 times. For each jump, we measured
; number of times this jump was reached (percents from main loop
; means importance of this code path) and probability of this jump to
; be taken (for branch prediction of important jumps).

do_loop_split:
for_loop:

	; REGISTER - end
	; eax = inverse shift amount (location of 0)
	; ecx = shift amount (ecx - eax)

	;      if (comp0 == (SCALAR)~0) {

	xor	ecx, 0FFFFFFFFh		; implies 'not ecx'
	bsr	eax, ecx
	and	ecx, ecx
	je	use_high_word		; ENTERED: 0x????????(??%), taken: ??%
	mov	ecx, 32
	sub	ecx, eax		; s = ecx-bsr
	add	eax, 32	; = ss
	jmp	found_shift

	align	16
use_high_word:
	xor	esi, 0FFFFFFFFh		; implies 'not esi'
	bsr	eax, esi
	and	esi, esi
	je	full_shift		; ENTERED: 0x????????(??%), taken: ??%
	mov	ecx, 64
	sub	ecx, eax		; s = ecx-bsr

found_shift:
	; REGISTER - start
	; eax = inverse shift amount (location of 0)
	; ecx = shift amount (ecx - eax)

	;        if ((mark += s) > limit) {
	;          break;
	;        }
	add	ebx, ecx
	cmp	ebx, edi ; limit (==lev->limit)
	jg	break_for		; ENTERED: 0x07A5D5F7(67.35%), taken 30-34%

	;        COMP_LEFT_LIST_RIGHT(lev, s);
	; !!!

	movd	mm_temp_ss, eax		; mm7 = ss
	movd	mm_temp_s, ecx		; mm6 = s

	; newbit + list goes right

	psllq	mm_newbit, mm_temp_ss	; insert newbit
	movq	mm_temp_a, cur(list, 0)
	movq	mm_temp_b, mm_temp_a	; list[0] cached in mm_temp_b
	psrlq	mm_temp_a, mm_temp_s
	por	mm_newbit, mm_temp_a
	movq	cur(list, 0), mm_newbit

	movq	mm_temp_a, cur(list, 1)
	movq	mm_newbit, mm_temp_a	; list[1] cached in mm_newbit
	psllq	mm_temp_b, mm_temp_ss
	psrlq	mm_temp_a, mm_temp_s
	por	mm_temp_b, mm_temp_a
	movq	cur(list, 1), mm_temp_b

	movq	mm_temp_a, cur(list, 2)
	movq	mm_temp_b, mm_temp_a	; list[2] cached in mm_tempb
	psllq	mm_newbit, mm_temp_ss
	psrlq	mm_temp_a, mm_temp_s
	por	mm_newbit, mm_temp_a
	movq	cur(list, 2), mm_newbit

	movq	mm_temp_a, cur(list, 3)
	psllq	mm_temp_b, mm_temp_ss
	psrlq	mm_temp_a, mm_temp_s
	por	mm_temp_b, mm_temp_a
	movq	cur(list, 3), mm_temp_b

	; comp goes left

	movq	mm_temp_a, mm_comp1
	psllq	mm_comp0, mm_temp_s
	movq	mm_temp_b, mm_comp2
	psrlq	mm_temp_a, mm_temp_ss
	movq	mm_newbit, cur(comp, 3)
	por	mm_comp0, mm_temp_a
	movq	mm_temp_a, mm_newbit
	psllq	mm_comp1, mm_temp_s
	psrlq	mm_temp_b, mm_temp_ss
	por	mm_comp1, mm_temp_b
	psllq	mm_comp2, mm_temp_s
	psrlq	mm_newbit, mm_temp_ss
	por	mm_comp2, mm_newbit
	psllq	mm_temp_a, mm_temp_s
	movq	cur(comp, 3), mm_temp_a

after_if:
	; ebx = mark

	;      if (depth == oState->maxdepthm1) {
	;        goto exit;         /* Ruler found */
	;      }
	mov	eax, [param_oState]
	mov	eax, [eax+oState_maxdepthm1]
	cmp	eax, [work_depth]
	je	exit			; ENTERED: 0x0513FD1B(44.72%), taken: 0%

	;      PUSH_LEVEL_UPDATE_STATE(lev);
	; !!!
	; **   LIST[lev+1] = LIST[lev]
	; **   DIST[lev+1] = (DIST[lev] | LIST[lev+1])
	; **   COMP[lev+1] = (COMP[lev] | DIST[lev+1])
	; **   newbit = 1;

	movq	cur(comp, 0), mm_comp0
	movq	cur(comp, 1), mm_comp1
	movq	cur(comp, 2), mm_comp2

	movq	mm_temp_b, cur(list, 0)
	movq	next(list, 0), mm_temp_b
	por	mm_temp_b, cur(dist, 0)		; dist0 ready in mm_temp_b
	movq	next(dist, 0), mm_temp_b
	por	mm_comp0, mm_temp_b

	movq	mm_temp_a, cur(list, 1)
	movq	next(list, 1), mm_temp_a
	por	mm_temp_a, cur(dist, 1)
	movq	next(dist, 1), mm_temp_a
	por	mm_comp1, mm_temp_a

	movq	mm_temp_a, cur(list, 2)
	movq	next(list, 2), mm_temp_a
	por	mm_temp_a, cur(dist, 2)
	movq	next(dist, 2), mm_temp_a
	por	mm_comp2, mm_temp_a

	movq	mm_temp_a, cur(list, 3)
	movq	next(list, 3), mm_temp_a
	por	mm_temp_a, cur(dist, 3)
	movq	next(dist, 3), mm_temp_a
	por	mm_temp_a, cur(comp, 3)
	movq	next(comp, 3), mm_temp_a

;	!! delay init !!
;	newbit = 1

	; Prepare ecx, esi for find leading zero (Stage 1)
	pshufw	mm_temp_a, mm_comp0, 11101110b

	;      lev->mark = mark;
	mov	[ebp+level_mark-ebp_shift], ebx
	mov	[ebp+level_limit-ebp_shift], edi

	;      lev++;
	add	ebp, sizeof_level

	punpckhdq mm_temp_b, mm_temp_b	; dist0 >> 32

	;      depth++;
	mov	edx, [work_depth]
	add	edx, 1
	mov	[work_depth], edx

	; /* Compute the maximum position for the next level */
	; #define choose(dist,seg) pchoose[(dist >> (SCALAR_BITS-CHOOSE_DIST_BITS)) * 32 + (seg)]
	; limit = choose(dist0, depth);

	movd	ecx, mm_temp_b		; dist0 in ecx

	mov	eax, ecx ; dist 0
	shr	eax, CHOOSE_DIST_BITS
	shl	eax, 5
	add	eax, edx		; depth
	mov	edx, [param_pchoose]
	movzx	edi, word [edx+eax*2]

	;      if (depth > oState->half_depth && depth <= oState->half_depth2) {
	;;;      if (depth > halfdepth && depth <= halfdepth2) {
	mov	eax, [work_depth]
	cmp	eax, [work_halfdepth2]
	jle	continue_if_depth	; ENTERED: 0x0513FD14(44.72%), NOT taken 97.02%

skip_if_depth:
	; Prepare ecx, esi for find leading zero (Stage 2)
	movd	ecx, mm_temp_a
	movd	esi, mm_comp0

	mov	edx, 1		; newbit = 1 (delayed)
	movd	mm_newbit, edx

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

	mov	esi, [work_maxlen_m1]
	mov	eax, [work_half_depth_mark_addr]
	sub	esi, [eax]

;        if (depth < oState->half_depth2) {
	mov	eax, [work_depth]
	cmp	eax, [work_halfdepth2]
	jge	update_limit_temp	; ENTERED: 0x00267D08(1.32%), taken 78.38%

;          temp -= LOOKUP_FIRSTBLANK(dist0); // "33" version
;;;        temp -= LOOKUP_FIRSTBLANK(dist0 & -((SCALAR)1 << 32));
;;;        (we already have upper part of dist0 in ecx)

	not	ecx
	mov	edx, -1
	bsr	eax, ecx
	cmovz	eax, edx

	add	esi, eax
	sub	esi, 32

update_limit_temp:
;        if (limit > temp) {
;          limit = temp;
;        }

	cmp	edi, esi
	cmovg	edi, esi
	jmp	skip_if_depth

	align	16

full_shift:
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
	movq	mm_temp_a,  cur(list, 0)
	movq	mm_temp_b,  cur(list, 1)
	movq	mm_temp_s,  cur(list, 2)
	movq	cur(list, 0), mm_newbit
	movq	cur(list, 1), mm_temp_a
	movq	cur(list, 2), mm_temp_b
	movq	cur(list, 3), mm_temp_s
	pxor	mm_newbit, mm_newbit		; newbit = 0

	; Prepare ecx, esi for find leading zero
	pshufw	mm_temp_a, mm_comp1, 11101110b
	movd	ecx, mm_temp_a
	movd	esi, mm_comp1

	; comp goes left			; 01 23 45 67 --
	movq	mm_comp0, mm_comp1
	movq	mm_comp1, mm_comp2
	movq	mm_comp2, cur(comp, 3)
	movq	cur(comp, 3), mm_newbit		; newbit is zero

	jmp	for_loop		; ENTERED: 0x013BFDCE(10.87%), taken 97.10%

	align	16
break_for:

	;    lev--;
	sub	ebp, sizeof_level
	;    depth--;
	dec	dword [work_depth]
	;    POP_LEVEL(lev);
	; !!!
	movq	mm_comp0, cur(comp, 0)	; SCALAR comp0 = lev->comp[0];
	movq	mm_comp1, cur(comp, 1)
	movq	mm_comp2, cur(comp, 2)

	pxor	mm_newbit, mm_newbit	;      newbit = 0;

	; Prepare ecx, esi for find leading zero
	pshufw	mm_temp_a, mm_comp0, 11101110b
	movd	ecx, mm_temp_a
	movd	esi, mm_comp0

	;  } while (depth > oState->stopdepth);
	mov	edx, [param_oState]
	mov	edx, [edx+oState_stopdepth]
	cmp	edx, [work_depth]

	; split loop header
	mov	ebx, [ebp+level_mark-ebp_shift]	; mark  = lev->mark;
	mov	edi, [ebp+level_limit-ebp_shift]

	jl	do_loop_split		; ENTERED: 0x0513FCC2(44.72%), taken 99.99%

exit:
	;  SAVE_FINAL_STATE(lev);
	; !!!
	movq	cur(comp, 0), mm_comp0
	movq	cur(comp, 1), mm_comp1
	movq	cur(comp, 2), mm_comp2

	;      lev->mark = mark;
	mov	[ebp+level_mark-ebp_shift], ebx
	mov	[ebp+level_limit-ebp_shift], edi

	mov	ebx, [param_pnodes]	; *pnodes -= nodes;
	mov	eax, [work_nodes]
	sub	[ebx], eax

	mov	eax, [work_depth]	; return depth;

	add	esp, worksize
	pop	ebp
	pop	edi
	pop	esi
	pop	ebx
	emms
	ret
