;
; Assembly core for OGR-NG, 64bit with AVX2. Based on SSE2 core (ogrng-cj1-sse2-asm.asm).
; $Id: ogrng64-cj1-avx2-asm.asm,v 1.0 2013/06/28 05:35:17 stream Exp $
;
; Created by Craig Johnston (craig.johnston@dolby.com)
;
; 2017-04-17: Initial AVX2 version
;

%ifdef __NASM_VER__
	cpu	686
%else
	cpu	p3 mmx sse sse2 sse41 avx avx2 lzcnt
	BITS	64
%endif

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
	[SECTION _DATA USE32 ALIGN=16 CLASS=DATA]
	[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
	[SECTION .data]
	[SECTION .text]
%endif

	%define CHOOSE_DIST_BITS	16     ; /* number of bits to take into account  */

	; Register renames
	%define xmm_newbit	xmm0
	%define ymm_newbit	ymm0	; Used only when blending
	%define ymm_list	ymm1
	%define ymm_comp	ymm2
	%define xmm_comp	xmm2	; Used for when only the lowest 128 bits of comp is requred
	%define ymm_dist	ymm3
	%define xmm_dist	xmm3	; Used for when only the lowest 128 bits of dist is requred

	%define xmm_temp_s	xmm4
	%define xmm_temp_ss	xmm5

	%define ymm_temp_A	ymm6
	%define ymm_temp_B	ymm7

	%define xmm_zero	xmm14
	%define ymm_zero	ymm14
	%define xmm_one	xmm15


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
	%define sizeof_level	128	; (32*3+8+8*6)
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
	vmovdqa	ymm_list, cur(list, 0)
	vmovdqa	ymm_dist, cur(dist, 0)

for_loop%1:

	; REGISTER - end
	; eax = inverse shift amount (location of 0)
	; ecx = shift amount (ecx - eax)

	xor	rax, -1
	jz	full_shift%1

%ifdef use_lzcnt
	lzcnt	rcx, rax
	mov	eax, 63
	sub	eax, ecx
	add	ecx, 1
%else
	bsr	rax, rax
	mov	ecx, 64
	sub	ecx, eax		; s = ecx-bsr
%endif

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

	; Input
	; comp [D C B A]
	; list [D C B A]
	; newb [0 0 0 N]
	;
	; Output
	; comp >>[0 D C B] (temp_B) | <<[D C B A] (comp)
	; list >>[D C B A] (list) | <<[C B A N] (temp_A)
	; newb [X X X X]

	vmovq	xmm_temp_ss, rax
	vmovq	xmm_temp_s, rcx
	vmovdqa	cur(dist, 0), ymm_dist

	; newbit + list goes right and comp goes left

	vpsllq	ymm_temp_A, ymm_list, xmm_temp_ss
	vpsrlq	ymm_temp_B, ymm_comp, xmm_temp_ss
	vpsllq	xmm_newbit, xmm_newbit, xmm_temp_ss
	vpsllq	ymm_comp, ymm_comp, xmm_temp_s
	vpermpd	ymm_temp_A, ymm_temp_A, 90h	; Reorder temp to be [C B A D]
	vpblendd	ymm_temp_B, ymm_temp_B, ymm_zero, 3	; overwrite lowest quadword with 0
	vpsrlq	ymm_list, ymm_list, xmm_temp_s
	vpermpd	ymm_temp_B, ymm_temp_B, 39h	; Reorder temp to be [0 D C B]

	; ebx = mark

	;      if (depth == oState->maxdepthm1) {
	;        goto exit;         /* Ruler found */
	;      }
	cmp	r14d, edx
	je	ruler_found%1

	vpblendd	ymm_temp_A, ymm_temp_A, ymm_newbit, 3	; overwrite lowest quadword with N
	vpor	ymm_list, ymm_temp_A
	vpor	ymm_comp, ymm_temp_B

	;      PUSH_LEVEL_UPDATE_STATE(lev);
	; !!!
	; **   LIST[lev+1] = LIST[lev]
	; **   DIST[lev+1] = (DIST[lev] | LIST[lev+1])
	; **   COMP[lev+1] = (COMP[lev] | DIST[lev+1])
	; **   newbit = 1;

	; Save our loaded values
	vmovdqa	cur(list, 0), ymm_list

	; **   LIST[lev+1] = LIST[lev]	; No need as we keep list in registers
	; **   DIST[lev+1] = (DIST[lev] | LIST[lev+1])
	; **   COMP[lev+1] = (COMP[lev] | DIST[lev+1])

	vmovdqa	cur(comp, 0), ymm_comp
	vpor	ymm_dist, ymm_list
	vpor	ymm_comp, ymm_dist

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

	vpextrw	rax, xmm_dist, 3	; Extract the first 16 bits from dist
	shl	eax, 5
	add	eax, edx
	movzx	edi, word [r13+rax*2]

	;      if (depth > oState->half_depth && depth <= oState->half_depth2) {
	;;;      if (depth > halfdepth && depth <= halfdepth2) {
	cmp	edx, [work_halfdepth2]
	jbe	continue_if_depth%1

skip_if_depth%1:

	vpextrq	rax, xmm_comp, 0
	vmovq	xmm_newbit, xmm_one;

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

	vpextrq	rcx, xmm_dist, 0	; move upper part of dist into rcx
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

	; Input
	; comp [D C B A]
	; list [D C B A]
	; newb [0 0 0 N]
	;
	; Output
	; comp >>[0 D C B]
	; list [C B A N]
	; newb [0 0 0 0]

	vpermpd	ymm_comp, ymm_comp, 39h	; Reorder to be [A D C B]
	vpermpd ymm_list, ymm_list, 90h	; Reorder to be [C B A D]

	vpblendd	ymm_comp, ymm_comp, ymm_zero, 192	; overwrite highest quadword with 0
	vpblendd	ymm_list, ymm_list, ymm_newbit, 3	; overwrite lowest quadword with N
	vmovq	xmm_newbit, xmm_zero	; Clear newbit

	vpextrq	rax, xmm_comp, 0
	jmp	for_loop%1

	align	16
break_for%1:

	;    lev--;
	sub	rbp, sizeof_level

	;    depth--;
	dec	edx

	;    POP_LEVEL(lev);
	; !!!
	vmovdqa	ymm_comp, cur(comp, 0)
	vmovq	xmm_newbit, xmm_zero	;      newbit = 0;

	;  } while (depth > oState->stopdepth);
	mov	ecx, [work_stopdepth]

	vpextrq	rax, xmm_comp, 0

	; split loop header
	mov	ebx, [rbp+level_mark-rbp_shift]	; mark  = lev->mark;
	mov	edi, [rbp+level_limit-rbp_shift]

	cmp	ecx, edx
	jb	do_loop_split%3

	vmovdqa	ymm_list, cur(list, 0)
	vmovdqa	ymm_dist, cur(dist, 0)
	jmp	exit

ruler_found%1:
	vpblendd	ymm_temp_A, ymm_temp_A, ymm_newbit, 3	; overwrite lowest quadword with N
	vpor	ymm_list, ymm_temp_A
	vpor	ymm_comp, ymm_temp_B
	jmp	exit
%endmacro

%macro header 1
	; Although Linux requires less registers to save, common code
	; is simpler to manage. So save maximum amount required to work with all OS'es.
	push	rsi	; Windows
	push	rdi	; Windows
	push	rbx
	push	rbp
	push	r12
	push	r13
	push	r14
	push	r15
	; According to x64 ABI, stack must be aligned by 16 before call =>
	; it'll be xxxxxxx8 after call. We've pushed EVEN number of registers above =>
	; stack is still at xxxxxxx8. Subtracting ***8 will make it aligned to 16,
	; so we can save XMM registers (required for Windows only, but see above).
	sub	rsp, 0xA8
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

%ifdef _WINDOWS
	; Switch to linux calling convention
	mov	rdi, rcx
	mov	rsi, rdx
	mov	rdx, r8
%endif

	; Create work area and align it to 32 bytes
	mov	rcx, rsp
	sub	rsp, worksize
	and	rsp, -32
	mov	[work_oldrsp], rcx

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

	; Zero all vector registers
	vzeroall

	mov eax, 1
	vmovq xmm_one, rax

	; SETUP_TOP_STATE(lev);
	; !!!
	vmovdqa	ymm_comp, cur(comp, 0)

	; int newbit = (depth < oState->maxdepthm1) ? 1 : 0;
	xor	eax, eax
	cmp	edx, r14d
	setl	al
	vmovq	xmm_newbit, rax

	; mm0..mm3 = comp
	; mm4 = newbit

	; split loop header
	mov	ebx, [rbp+level_mark-rbp_shift]	; mark  = lev->mark;
	mov	edi, [rbp+level_limit-rbp_shift]

	vpextrq	rax, xmm_comp, 0

	jmp do_loop_split%1
%endmacro

%macro footer 0
exit:
	;  SAVE_FINAL_STATE(lev);
	; !!!
	vmovdqa	cur(list, 0), ymm_list
	vmovdqa	cur(dist, 0), ymm_dist
	vmovdqa	cur(comp, 0), ymm_comp

	;      lev->mark = mark;
	mov	[rbp+level_mark-rbp_shift], ebx
	mov	[rbp+level_limit-rbp_shift], edi

	mov	rbx, [work_pnodes]	; *pnodes -= nodes;
	sub	[rbx], r15d

	mov	eax, edx	; return depth;

	mov	rsp, [work_oldrsp]
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
	add	rsp, 0xA8
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbp
	pop	rbx
	pop	rdi
	pop	rsi
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

global	_ogrng64_cycle_256_cj1_avx2
global	ogrng64_cycle_256_cj1_avx2
_ogrng64_cycle_256_cj1_avx2:
ogrng64_cycle_256_cj1_avx2:

	header 5
	body 30
	footer
