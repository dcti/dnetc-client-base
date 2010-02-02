;
; Base assembly core for OGR-NG, 64bit. Based on generic 64 bit assembly core (ogrng64-cj1-generic-asm.asm) and SSE2 core (ogrng-cj1-sse2-asm.asm).
; $Id: ogrng64-cj1-base.asm,v 1.1 2010/02/02 05:35:17 stream Exp $
;
; Created by Craig Johnston (craig.johnston@dolby.com)
;
; 2009-10-25: Added LZCNT and PEXTRW support
;
; 2009-04-06: Initial 64 bit SSE2 version.
;
; Possible improvements:
; * Try and use alternating registers in chain for comp, list and dist0
; * Try alternate bsr/lzcnt version with no jump for full shift

	%define CHOOSE_DIST_BITS	16     ; /* number of bits to take into account  */

	; Register renames
	%define mm_one		mm0

	%define xmm_newbit	xmm15
	%define xmm_temp_s	xmm0
	%define xmm_temp_ss	xmm1
	%define xmm_list0	xmm2
	%define xmm_list2	xmm3
	%define xmm_comp0	xmm4
	%define xmm_comp2	xmm5
	%define xmm_dist0	xmm6
	%define xmm_dist2	xmm7
	%define xmm_temp_a	xmm8
	%define xmm_temp_b	xmm9

	; REGISTER - globals
	; ebx = mark
	; edi = limit
	; edx = work depth
	; ebp = stack location
	; r12 = half mark addr
	; r13 = pchoose
	; r14d = max_depth_m1
	; r15d = nodes

	%define worksize	30h

	%define work_halfdepth		rsp+00h
	%define work_halfdepth2		rsp+04h
	%define work_maxlen_m1		rsp+08h
	%define work_stopdepth		rsp+0Ch

	; 64 bit work elements
	%define work_oState			rsp+10h
	%define work_pnodes			rsp+18h
	%define work_oldrsp			rsp+20h

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


	; Macro defining the whole body of the function
	; Parameter 1 = The Name of this block
	; Parameter 2 = The Name of the block to jump to when pushing
	; Parameter 3 = The Name of the block to jump to when popping
%macro func 3

	align	16
do_loop_split%1:
	movdqa	xmm_list0, cur(list, 0)
	movdqa	xmm_list2, cur(list, 2)
	movdqa	xmm_dist0, cur(dist, 0)
	movdqa	xmm_dist2, cur(dist, 2)

for_loop%1:

	; REGISTER - end
	; eax = inverse shift amount (location of 0)
	; ecx = shift amount (ecx - eax)

	;      if (comp0 == (SCALAR)~1) {

	movq	rcx, xmm_comp0
	xor	rcx, -1
	je	full_shift%1

%ifdef use_lzcnt
	lzcnt	rcx, rcx
	mov	eax, 63
	sub	eax, ecx
	add	ecx, 1
%else
	bsr	rax, rcx
	mov	ecx, 64
	sub	ecx, eax		; s = ecx-bsr
%endif

found_shift%1:
	; REGISTER - start
	; eax = inverse shift amount (location of 0)
	; ecx = shift amount (64 - eax)

	;        if ((mark += s) > limit) {
	;          break;
	;        }
	add	ebx, ecx
	cmp	ebx, edi ; limit (==lev->limit)
	ja	break_for%1

	;        COMP_LEFT_LIST_RIGHT(lev, s);
	; !!!

	movd	xmm_temp_ss, eax
	movd	xmm_temp_s, ecx

	; newbit + list goes right and comp goes left

	; copy of list for shifting left
	movdqa	xmm_temp_a, xmm_list0
	movdqa	xmm_temp_b, xmm_list2

	movdqa	cur(dist, 0), xmm_dist0
	movdqa	cur(dist, 2), xmm_dist2

	psllq	xmm_temp_a, xmm_temp_ss
	psllq	xmm_temp_b, xmm_temp_ss
	psllq	xmm_newbit, xmm_temp_ss

	psrlq	xmm_list0, xmm_temp_s
	psrlq	xmm_list2, xmm_temp_s

	shufpd	xmm_newbit, xmm_temp_a, 0	; select Low(a), Low(s)
	shufpd	xmm_temp_a, xmm_temp_b, 1	; select Low(b), High(a)

	movdqa	xmm_temp_b, xmm_comp0
	por	xmm_list0, xmm_newbit
	movdqa	xmm_newbit, xmm_comp2	; xmm_newbit used as temp space
	por	xmm_list2, xmm_temp_a

	; comp goes left

	psrlq	xmm_temp_b, xmm_temp_ss
	psrlq	xmm_newbit, xmm_temp_ss

	psllq	xmm_comp0, xmm_temp_s
	psllq	xmm_comp2, xmm_temp_s

	shufpd	xmm_temp_b, xmm_newbit, 1	; select Low(n), High(b)
	psrldq	xmm_newbit, 8

after_if%1:
	; ebx = mark

	;      if (depth == oState->maxdepthm1) {
	;        goto exit;         /* Ruler found */
	;      }
	cmp	r14d, edx
	je	ruler_found%1

	por	xmm_comp0, xmm_temp_b
	por	xmm_comp2, xmm_newbit

	;      PUSH_LEVEL_UPDATE_STATE(lev);
	; !!!
	; **   LIST[lev+1] = LIST[lev]
	; **   DIST[lev+1] = (DIST[lev] | LIST[lev+1])
	; **   COMP[lev+1] = (COMP[lev] | DIST[lev+1])
	; **   newbit = 1;

	; Save our loaded values
	movdqa	cur(list, 0), xmm_list0
	movdqa	cur(list, 2), xmm_list2

	; **   LIST[lev+1] = LIST[lev]	; No need as we keep list in registers
	; **   DIST[lev+1] = (DIST[lev] | LIST[lev+1])
	; **   COMP[lev+1] = (COMP[lev] | DIST[lev+1])

	movdqa	cur(comp, 0), xmm_comp0
	por	xmm_dist0, xmm_list0
	por	xmm_dist2, xmm_list2
	movdqa	cur(comp, 2), xmm_comp2
	por	xmm_comp0, xmm_dist0
	por	xmm_comp2, xmm_dist2

;	!! delay init !!
;	newbit = 1

	;      lev->mark = mark;
	mov	[rbp+level_mark-rbp_shift], ebx
	mov	[rbp+level_limit-rbp_shift], edi

	;      lev++;
	add	rbp, sizeof_level

	;      depth++;
	inc	edx

	; /* Compute the maximum position for the next level */
	; #define choose(dist,seg) pchoose[(dist >> (SCALAR_BITS-CHOOSE_DIST_BITS)) * 32 + (seg)]
	; limit = choose(dist0, depth);

%ifdef use_pextrw
	pextrw	eax, xmm_dist0, 11b
%else
	movq	rax, xmm_dist0 ; dist 0
	shr	rax, 64 - CHOOSE_DIST_BITS
%endif
	shl	eax, 5
	add	eax, edx
	movzx	edi, word [r13+rax*2]

	;      if (depth > oState->half_depth && depth <= oState->half_depth2) {
	;;;      if (depth > halfdepth && depth <= halfdepth2) {
	cmp	edx, [work_halfdepth2]
	jbe	continue_if_depth%1

skip_if_depth%1:

	movq2dq	xmm_newbit, mm_one

	;      if (--nodes <= 0) {

	sub	r15d, 1
	jg	for_loop%2

	;        goto exit;
	jmp	exit

	align	16
continue_if_depth%1:
	cmp	edx, [work_halfdepth]
	jbe	skip_if_depth%1

;        int temp = maxlen_m1 - oState->Levels[oState->half_depth].mark;
;;        int temp = oState->max - 1 - oState->Levels[halfdepth].mark;

	mov	esi, [work_maxlen_m1]
	sub	esi, [r12]

;        if (depth < oState->half_depth2) {
	cmp	edx, [work_halfdepth2]
	jae	update_limit_temp%1

;          temp -= LOOKUP_FIRSTBLANK(dist0); // "33" version
;;;        temp -= LOOKUP_FIRSTBLANK(dist0 & -((SCALAR)1 << 32));
;;;        (we already have upper part of dist0 in ecx)

	movq	rcx, xmm_dist0	; dist0
	not	rcx
%ifdef use_lzcnt
	sub	esi, 1
	lzcnt	rcx, rcx
	sub	esi, ecx
%else
	mov	eax, -1
	bsr	rcx, rcx
	cmovz	ecx, eax
	add	esi, ecx
	sub	esi, 64
%endif

update_limit_temp%1:
;        if (limit > temp) {
;          limit = temp;
;        }

	cmp	edi, esi
	cmovg	edi, esi
	jmp	skip_if_depth%1

	align	16
full_shift%1:
	;      else {         /* s >= SCALAR_BITS */

	;        if ((mark += SCALAR_BITS) > limit) {
	;          break;
	;        }
	add	ebx, 64
	cmp	ebx, edi ; limit (==lev->limit)
	ja	break_for%1

	;      COMP_LEFT_LIST_RIGHT_WORD(lev);
	;      continue;

	; COMP_LEFT_LIST_RIGHT_WORD(lev);
	; !!!

	pslldq	xmm_list2, 8
	movhlps	xmm_list2, xmm_list0

	pslldq	xmm_list0, 8
	por	xmm_list0, xmm_newbit

	psrldq	xmm_comp0, 8
	pxor	xmm_newbit, xmm_newbit	; Clear newbit
	punpcklqdq	xmm_comp0, xmm_comp2

	psrldq	xmm_comp2, 8

	jmp	for_loop%1

	align	16
break_for%1:

	;    lev--;
	sub	rbp, sizeof_level

	;    depth--;
	dec	edx

	;    POP_LEVEL(lev);
	; !!!
	movdqa	xmm_comp0, cur(comp, 0)
	movdqa	xmm_comp2, cur(comp, 2)

	pxor	xmm_newbit, xmm_newbit	;      newbit = 0;

	;  } while (depth > oState->stopdepth);
	mov	eax, [work_stopdepth]

	; split loop header
	mov	ebx, [rbp+level_mark-rbp_shift]	; mark  = lev->mark;
	mov	edi, [rbp+level_limit-rbp_shift]

	cmp	eax, edx
	jb	do_loop_split%3

	movdqa	xmm_list0, cur(list, 0)
	movdqa	xmm_list2, cur(list, 2)
	movdqa	xmm_dist0, cur(dist, 0)
	movdqa	xmm_dist2, cur(dist, 2)
	jmp	exit

ruler_found%1:
	por	xmm_comp0, xmm_temp_b
	por	xmm_comp2, xmm_newbit
	jmp	exit
%endmacro

%macro header 0
%ifdef _WINDOWS
	push	rsi
	push	rdi
	push	rbx
	push	rbp
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, worksize

	; Switch to linux calling convention
	mov	rdi, rcx
	mov	rsi, rdx
	mov	rdx, r8

	; Align stack to 32 bytes
	mov	rcx, rsp
	and	rsp, 0xFFFFFFE0
	mov	[work_oldrsp], rcx

	sub	rsp, 0xA0
	movdqa	[rsp+0x00], xmm6
	movdqa	[rsp+0x10], xmm7
	movdqa	[rsp+0x20], xmm8
	movdqa	[rsp+0x30], xmm9
	movdqa	[rsp+0x40], xmm10
	movdqa	[rsp+0x50], xmm11
	movdqa	[rsp+0x60], xmm12
	movdqa	[rsp+0x70], xmm13
	movdqa	[rsp+0x80], xmm14
	movdqa	[rsp+0x90], xmm15
%else
	push	rbx
	push	rbp
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, worksize

	; Align stack to 32 bytes
	mov	rcx, rsp
	and	rsp, 0xFFFFFFE0
	mov	[work_oldrsp], rcx
%endif

start:
	; write the paramters in the aligned work space
	mov	[work_oState], rdi
	mov	[work_pnodes], rsi
	mov	r13, rdx

	mov	edx, [rdi+oState_depth]

	imul	eax, edx, sizeof_level
	lea	rbp, [rax+rdi+oState_Levels+rbp_shift]	; lev = &oState->Levels[oState->depth]
	mov	r15d, [rsi]	; nodes = *pnodes

	mov	eax, [rdi+oState_half_depth]
	mov	[work_halfdepth], eax	; halfdepth = oState->half_depth

	; get address of oState->Levels[oState->half_depth].mark
	; value of this var can be changed during crunching, but addr is const
	imul	eax, sizeof_level
	lea	r12, [rax+rdi+oState_Levels+level_mark]

	mov	eax, [rdi+oState_half_depth2]
	mov	[work_halfdepth2], eax	; halfdepth2 = oState->half_depth2

	mov	eax, [rdi+oState_max]
	dec	eax
	mov	[work_maxlen_m1], eax	; maxlen_m1 = oState->max - 1

	mov	r14d, [rdi+oState_maxdepthm1]

	mov	eax, [rdi+oState_stopdepth]
	mov	[work_stopdepth], eax

	; SETUP_TOP_STATE(lev);
	; !!!
	movdqa	xmm_comp0, cur(comp, 0)
	movdqa	xmm_comp2, cur(comp, 2)

	; int newbit = (depth < oState->maxdepthm1) ? 1 : 0;
	xor	eax, eax
	cmp	edx, r14d
	setl	al
	movq	xmm_newbit, rax

	mov	eax, 1
	movq	mm_one, rax

	; mm0..mm3 = comp
	; mm4 = newbit

	; split loop header
	mov	ebx, [rbp+level_mark-rbp_shift]	; mark  = lev->mark;
	mov	edi, [rbp+level_limit-rbp_shift]
%endmacro

%macro footer 0
exit:
	;  SAVE_FINAL_STATE(lev);
	; !!!
	movdqa	cur(list, 0), xmm_list0
	movdqa	cur(list, 2), xmm_list2
	movdqa	cur(dist, 0), xmm_dist0
	movdqa	cur(dist, 2), xmm_dist2
	movdqa	cur(comp, 0), xmm_comp0
	movdqa	cur(comp, 2), xmm_comp2

	;      lev->mark = mark;
	mov	[rbp+level_mark-rbp_shift], ebx
	mov	[rbp+level_limit-rbp_shift], edi

	mov	rbx, [work_pnodes]	; *pnodes -= nodes;
	sub	[rbx], r15d

	mov	eax, edx	; return depth;

%ifdef _WINDOWS
	movdqa	xmm6, [rsp+0x00]
	movdqa	xmm7, [rsp+0x10]
	movdqa	xmm8, [rsp+0x20]
	movdqa	xmm9, [rsp+0x30]
	movdqa	xmm10, [rsp+0x40]
	movdqa	xmm11, [rsp+0x50]
	movdqa	xmm12, [rsp+0x60]
	movdqa	xmm13, [rsp+0x70]
	movdqa	xmm14, [rsp+0x80]
	movdqa	xmm15, [rsp+0x90]
	add	rsp, 0xA0

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
%else
	mov	rsp, [work_oldrsp]
	add	rsp, worksize
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbp
	pop	rbx
%endif
	emms
	ret
%endmacro

%macro body 1
	%assign max_id %1
	%assign id 1
	%rep %1
		%assign next_id id + 1
		%if next_id > max_id
			%assign next_id max_id
		%endif

		%assign prev_id id - 1
		%if prev_id < 1
			%assign prev_id 1
		%endif

		func id, next_id, prev_id
		%assign id id + 1
	%endrep
%endmacro
