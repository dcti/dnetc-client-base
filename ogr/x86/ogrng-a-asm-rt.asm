cpu	386

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

	global	_ogr_cycle_256_rt1, ogr_cycle_256_rt1

_ogr_cycle_256_rt1:
ogr_cycle_256_rt1:

	%define regsavesize	10h	; 4 registers saved
	%define worksize	20h

	%define	work_depth	esp+00h
	%define work_halfdepth	esp+04h
	%define work_halfdepth2	esp+08h
	%define work_nodes	esp+0Ch

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
; long command 'ebp+nnnnnnnn' (dword offset). But I get no speed gain.
	%define ebp_shift		0	; may be 128
	%define sizeof_level		104	; (32*3+8)
	%define level_list		00h
	%define level_dist		20h
	%define level_comp		40h
	%define level_mark		60h
	%define level_limit		64h

%define cur(el, index)   [ebp+level_ %+ el + ((index)*4) - ebp_shift]
%define next(el, index)  [ebp+sizeof_level+level_ %+ el + ((index)*4) - ebp_shift]

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
	mov	eax, [ebx+oState_half_depth2]
	mov	[work_halfdepth2], eax	; halfdepth2 = oState->half_depth2

	; SETUP_TOP_STATE(lev);
	; !!!
	mov	esi, cur(comp, 0)	; SCALAR comp0 = lev->comp[0];
	; int newbit = (depth < oState->maxdepthm1) ? 1 : 0;
	cmp	ecx, [ebx+oState_maxdepthm1]	; depth still in ecx
	setl	al
	movzx	edi, al

	; esi = comp0
	; edi = newbit

	; split loop header
	mov	ebx, [ebp+level_mark-ebp_shift]	; mark  = lev->mark;

	align	16

; Jump probabilies calculated from summary statistic of 'dnetc -test'.
; Main loop was entered 0x0B5ACBE2 times. For each jump, we measured
; number of times this jump was reached (percents from main loop
; means importance of this code path) and probability of this jump to
; be taken (for branch prediction of important jumps).

do_loop_split:
for_loop:

	; esi = comp0
	; edi = newbit
	; ebx = mark

	;      if (comp0 < 0xFFFFFFFE) {
	cmp	esi, 0FFFFFFFEh
	jnb	comp0_ge_fffe		; ENTERED: 0x0B5ACBE2(100%), taken: 33-35%
	;        int s = LOOKUP_FIRSTBLANK_SAFE(comp0);
	not	esi
	mov	ecx, 20H
	bsr	edx, esi
	sub	ecx, edx			; s
	;        if ((mark += s) > limit) {
	;          break;
	;        }
	add	ebx, ecx
	cmp	ebx, [ebp+level_limit-ebp_shift] ; limit (==lev->limit)
	jg	break_for		; ENTERED: 0x07A5D5F7(67.35%), taken 30-34%

	;        COMP_LEFT_LIST_RIGHT(lev, s);
	; !!!

	mov	edx, cur(list, 6)
	mov	eax, cur(list, 5)
	shrd	dword cur(list, 7), edx, cl	; do/store 1C+18
	shrd	edx, eax, cl			; do 18+14
	mov	esi, cur(list, 4)
	mov	cur(list, 6), edx		; store 18
	shrd	eax, esi, cl			; do 14+10
	mov	edx, cur(list, 3)
	mov	cur(list, 5), eax		; store 14
	shrd	esi, edx, cl			; do 10+0C
	mov	eax, cur(list, 2)
	mov	cur(list, 4), esi		; store 10
	shrd	edx, eax, cl			; do 0C+08
	mov	esi, cur(list, 1)
	mov	cur(list, 3), edx		; store 0C
	shrd	eax, esi, cl			; do 08+04
	mov	edx, cur(list, 0)
	mov	cur(list, 2), eax		; store 08
	shrd	esi, edx, cl			; do 04+00
	shrd	edx, edi, cl			; do 00+newbit
	mov	cur(list, 0), edx		; store 00
	mov	cur(list, 1), esi		; store 04

	mov	eax, cur(comp, 1)
	mov	edx, cur(comp, 2)
	shld	dword cur(comp, 0), eax, cl
	shld	eax, edx, cl
	mov	cur(comp, 1), eax
	mov	eax, cur(comp, 3)
	shld	edx, eax, cl
	mov	cur(comp, 2), edx
	mov	edx, cur(comp, 4)
	shld	eax, edx, cl
	mov	cur(comp, 3), eax
	mov	eax, cur(comp, 5)
	shld	edx, eax, cl
	mov	cur(comp, 4), edx
	mov	edx, cur(comp, 6)
	shld	eax, edx, cl
	mov	cur(comp, 5), eax
	mov	eax, cur(comp, 7)
	shld	edx, eax, cl
	mov	cur(comp, 6), edx
	shl	eax, cl
	mov	cur(comp, 7), eax

	mov	esi, cur(comp, 0)		; comp0 = ...
	xor	edi, edi			; newbit = 0

after_if:
	; esi = comp0
	; edi = newbit
	; ebx = mark

	;      lev->mark = mark;
	mov	[ebp+level_mark-ebp_shift], ebx

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

	mov	ecx, cur(list, 0)
	mov	next(list, 0), ecx
	mov	edx, cur(list, 1)
	mov	next(list, 1), edx
	or	ecx, cur(dist, 0)
	or	edx, cur(dist, 1)
	mov	next(dist, 0), ecx		; dist0 ready in ecx
	mov	next(dist, 1), edx
	or	esi, ecx			; comp0 (updated cached)
	mov	next(comp, 0), esi
	or	edx, cur(comp, 1)
	mov	next(comp, 1), edx

%assign	_i	2
%rep	6
	mov	eax, cur(list, _i)
	mov	next(list, _i), eax
	or	eax, cur(dist, _i)
	mov	next(dist, _i), eax
	or	eax, cur(comp, _i)
	mov	next(comp, _i), eax
%assign	_i	_i + 1
%endrep
;	!! delay init !!
;	mov	edi, 1		; newbit = 1

	;      lev++;
	add	ebp, sizeof_level
	;      depth++;
	inc	dword [work_depth]

	%define CHOOSE_DIST_BITS	16     ; /* number of bits to take into account  */

	; /* Compute the maximum position for the next level */
	; #define choose(dist,seg) pchoose[(dist >> (SCALAR_BITS-CHOOSE_DIST_BITS)) * 32 + (seg)]
	; limit = choose(dist0, depth);

	mov	eax, ecx ; dist 0
	shr	eax, 16
	shl	eax, 5
	add	eax, [work_depth]
	mov	edx, [param_pchoose]
	movzx	edi, word [edx+eax*2]

	;      if (depth > oState->half_depth && depth <= oState->half_depth2) {
	;;;      if (depth > halfdepth && depth <= halfdepth2) {
	;
	; Check second condition first, it's very rare
	mov	eax, [work_depth]
	cmp	eax, [work_halfdepth2]
	jle	continue_if_depth	; ENTERED: 0x0513FD14(44.72%), NOT taken 97.02%

skip_if_depth:
	;
	; returning here with edi=new limit
	;
	mov	[ebp+level_limit-ebp_shift], edi

	mov	edi, 1		; newbit = 1 (delayed)

	;      if (--nodes <= 0) {

	dec	dword [work_nodes]
	jg	for_loop	; ENTERED: 0x0513FD14(44.72%), taken 99.99%

	;        lev->mark = mark;
	;        goto exit;
	mov	[ebp+level_mark-ebp_shift], ebx
	jmp	exit

	align	16

continue_if_depth:
	cmp	eax, [work_halfdepth]
	jle	skip_if_depth		; ENTERED: 0x********(**.**%), taken 0.5%

;        int temp = maxlen_m1 - oState->Levels[oState->half_depth].mark;
;;        int temp = oState->max - 1 - oState->Levels[halfdepth].mark;
;	Note: "-1" is not counted here. Instead, condition below is altered
;	and one is substracted, if required, during copy (in "lea").

	mov	edx, [param_oState]
	mov	eax, [work_halfdepth]
	imul	eax, sizeof_level
	mov	eax, [eax+edx+oState_Levels+level_mark]
	mov	edx, [edx+oState_max]
	sub	edx, eax

;        if (depth < oState->half_depth2) {
	mov	eax, [work_depth]
	cmp	eax, [work_halfdepth2]
	jge	update_limit_temp	; ENTERED: 0x00267D08(1.32%), taken 78.38%

;          temp -= LOOKUP_FIRSTBLANK(dist0); // "33" version

	xor	ecx, -1		; "not ecx" does not set flags!
	mov	eax, -1
	je	skip_bsr	; ENTERED: 0x00085254(0.29%), taken 0.00%
	bsr	eax, ecx
skip_bsr:
	add	edx, eax
	sub	edx, 32

update_limit_temp:
;        if (limit > temp) {
;          limit = temp;
;        }

	cmp	edi, edx
	jl	limitok		; ENTERED: 0x00267D08(1.32%), taken 17.35%
	lea	edi, [edx-1]
limitok:
	jmp	skip_if_depth

	align	16

comp0_ge_fffe:
	;      else {         /* s >= 32 */

	;        if ((mark += 32) > limit) {
	;          break;
	;        }
	add	ebx,32
	cmp	ebx, [ebp+level_limit-ebp_shift] ; limit (==lev->limit)
	jg	break_for		; ENTERED: 0x03B4F5EB(32.64%), taken 66.70%

	;        if (comp0 == ~0u) {
	;          COMP_LEFT_LIST_RIGHT_32(lev);
	;          continue;
	;        }
	;        COMP_LEFT_LIST_RIGHT_32(lev);

	cmp	esi,0ffffffffH		; if (comp0 == ~0u)

	; COMP_LEFT_LIST_RIGHT_32(lev);
	; !!!

%assign	_i	6
%rep	7
	mov	eax, cur(list, _i)
	mov	cur(list, _i+1), eax
%assign	_i	_i-1
%endrep
	mov	cur(list, 0), edi	; lev->list[0] = newbit;
	mov	edi, 0			; newbit = 0 (keep flags!)
	mov	esi, cur(comp, 1)	; comp0 cached
	mov	cur(comp, 0), esi
%assign _i 1
%rep	6
	mov	eax, cur(comp, _i+1)	; lev->comp[1] = lev->comp[2] etc.
	mov	cur(comp, _i), eax
%assign	_i	_i+1
%endrep
	mov	cur(comp, 7), edi	; lev->comp[7] = 0

	je	for_loop		; ENTERED: 0x013BFDCE(10.87%), taken 97.10%
	jmp	after_if

	align	16
break_for:

	;    lev--;
	sub	ebp, sizeof_level
	;    depth--;
	dec	dword [work_depth]
	;    POP_LEVEL(lev);
	; !!!
	mov	esi, cur(comp, 0)	;      comp0 = lev->comp[0];
	; unused here; mov	e??, [ebp+level_dist+0]	;      dist0 = lev->dist[0];
	xor	edi, edi		;      newbit = 0;

	;  } while (depth > oState->stopdepth);
	mov	eax, [param_oState]
	mov	eax, [eax+oState_stopdepth]
	cmp	eax, [work_depth]

	; split loop header
	mov	ebx, [ebp+level_mark-ebp_shift]	; mark  = lev->mark;

	jl	do_loop_split		; ENTERED: 0x0513FCC2(44.72%), taken 99.99%

exit:
	;  SAVE_FINAL_STATE(lev);
	; !!!
	; /* nothing for 32-bits */

	mov	ebx, [param_pnodes]	; *pnodes -= nodes;
	mov	eax, [work_nodes]
	sub	[ebx], eax

	mov	eax, [work_depth]	; return depth;

	add	esp, worksize
	pop	ebp
	pop	edi
	pop	esi
	pop	ebx
	ret
