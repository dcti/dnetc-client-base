;
; Assembly core for OGR, MMX version
;
; Created by Roman Trunov (proxyma@tula.net)
;
; Based on disassembled output of Watcom compiler which suddenly generated
; code about 15% faster than prior Windows clients.
;
; Watcom's output was optimized a little more manually (alas, this compiler
; cannot align loops), but pipeline optimization can be far away from complete.
; This code was developed on and tuned for PII-Celeron CPU, on other processors
; its performance can be comparable or even less then current cores.
;
; Anyway, on my system this code works about 30% faster than Windows client.
;
; Additional improvements can be achieved by better optimization of pipelines
; and usage of MMX instruction/registers.
;
; 2005-06-18: Removed local variable "limit", it's allocated on stack and
;             contains only copy of "lev->limit". I cannot allocate it in
;             register and there no difference how to reference it in
;             memory - as [esp+xx] or [ebp+xx]. Gained 2% in speed.

cpu	586

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
	[SECTION _DATA USE32 ALIGN=16 CLASS=DATA]
%else
	[SECTION .data]
%endif

;
; Better keep local copy of this table due to differences in naming
; of external symbols (even cdecl) in different compilers.
;
_OGR:	dd   0,   1,   3,   6,  11,  17,  25,  34,  44,  55	; /*  1 */
	dd  72,  85, 106, 127, 151, 177, 199, 216, 246, 283	; /* 11 */
	dd 333, 356, 372, 425, 480, 492, 553, 585, 623		; /* 21 */
;
; Address of found_one can be kept on stack (as extra parameter to ogr_cycle)
; but both cases are Ok and this one is simpler.
;
_found_one_cdecl_ptr:
	dd   0

%ifdef __OMF__ ; Borland and Watcom compilers/linkers
	[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
	[SECTION .text]
%endif

%macro  _xalign 2
	%assign __distance %1-(($-$$) & (%1-1))
	%if     __distance <= %2
		align	%1
	%endif
%endmacro

%macro	_natural_align 0
	%if (($-$$) & 15) <> 0
		%error "Assembly failed, this location must be 16-bytes aligned"
	%endif
%endmacro

global	_ogr_watcom_rt1_mmx64_asm, ogr_watcom_rt1_mmx64_asm

	%define	sizeof_level 50h		; sizeof(level)
	%define	lev_list0  00H			; fields in level
	%define lev_list1  08H
	%define	lev_list2  10H
	%define	lev_dist0  18H
	%define	lev_dist1  20H
	%define	lev_dist2  28H
	%define	lev_comp0  30H
	%define	lev_comp1  38H
	%define	lev_comp2  40H
	%define lev_mark   48H
	%define lev_limit  4cH

;
; Registers assigned:
;
; ecx = comp0 (32 bit)
; mm0 = comp1
; mm1 = comp2
; mm2 = list0 (32 bit) in lower part, newbit in high part
; mm3 = list1
; mm4 = list2
;
; mm5-mm7 are used for calculations
;
ogr_cycle_:
	push	ecx
	push	esi
	push	edi
	push	ebp
	sub	esp,30H
	add	ecx,3
	mov	dword [esp+1cH],ecx	; ogr_choose_dat+3
	mov	dword [esp+0cH],edx	; pnodes
	mov	edx,[edx]
	mov	dword [esp+2cH],edx	; *pnodes
	mov	dword [esp+18H],eax	; oState
	mov	edx,dword [eax+1cH]
	inc	edx
	mov	dword [esp+24H],edx	; int depth = oState->depth + 1;
	imul	edx,sizeof_level
	lea	ebp,[eax+edx+20H]	; lev = &oState->Levels[depth]
	and	dword [esp+14H],0	; int nodes = 0;
	mov	ecx,dword [eax+10H]
	imul	ecx,sizeof_level
	lea	ebx,[eax+ecx+20H]
	mov	dword [esp+4],ebx	; levHalfDepth = &oState->Levels[oState->half_depth]
	mov	edx,dword [eax]
	mov	dword [esp+10H],edx	; maxlength  = oState->max;
	mov	edx,dword [eax+8]
	mov	ecx,dword [esp+24H]
	sub	edx,ecx
	mov	dword [esp+20H],edx	; remdepth   = oState->maxdepthm1 - depth;
	mov	edx,dword [eax+10H]
	mov	ebx,dword [eax+18H]
	sub	edx,ebx
	mov	dword [esp],edx		; halfdepth  = oState->half_depth - oState->startdepth;
	mov	edx,dword [eax+14H]
	sub	edx,ebx
	mov	dword [esp+8],edx	; halfdepth2 = oState->half_depth2 - oState->startdepth;
	sub	ecx,ebx
	mov	dword [esp+24H],ecx	; depth -= oState->startdepth;

; stack allocation ([SP+##h]):
;	00H	halfdepth
;	04H	levHalfDepth
;	08H	halfdepth2
;	0CH	pnodes
;	10H	maxlength
;	14H	nodes
;	18H	oState
;	1CH	offsetof(level->comp) == [ebp+40] => KILLED => Now ogr_choose_dat+3
;	20H	remdepth
;	24H	depth
;	28H	limit => KILLED.
;	2cH	mark => KILLED. Now contains *pnodes for fast compare.

	mov	ebx,dword [ebp+lev_mark]	;   int mark = lev->mark;

	;    #define SETUP_TOP_STATE(lev)            \
	;      U comp1, comp2;                       \
	;      U list0, list1, list2;                \
	;      U dist1, dist2;                       \
	;      u32 comp0 = (u32) lev->comp[0];       \
	;      u32 dist0 = (u32) lev->dist[0];       \
	;      comp1 = lev->comp[1];                 \
	;      comp2 = lev->comp[2];                 \
	;      list0 = lev->list[0] | ((U)1 << 32);  \
	;      list1 = lev->list[1];                 \
	;      list2 = lev->list[2];                 \
	;      dist1 = lev->dist[1];                 \
	;      dist2 = lev->dist[2];

	mov	eax,1
	movd	mm7,eax			; special constant H=0000 L=0001

	mov	edi,dword [ebp+lev_dist0]	; (u32) dist0
	mov	ecx,dword [ebp+lev_comp0]	; (u32) lev->comp[0]
	movq	mm0,[ebp+lev_comp1]		;	lev->comp[1]
	movq	mm1,[ebp+lev_comp2]		;	lev->comp[2]
	movd	mm2,[ebp+lev_list0]		; (u32) lev->list[0]...
	punpckldq mm2,mm7			;	  ... | 1 << 32
	movq	mm3,[ebp+lev_list1]		;	lev->list[1]
	movq	mm4,[ebp+lev_list2]		;	lev->list[2]

	jmp	outerloop

%if ($-$$) <> 95h
	%error	"Assembly of jumps and constant must be optimized, add -O5 to NASM options"
%endif

	align	16
checklimit:
	; we enter here with:
	; 	eax = depth
	;	edx = limit
	; we must return back to store_limit with:
	;	edx = limit
	; Don't forget to store limit when leaving globally!
	;      if (depth <= halfdepth) {
	cmp	eax,dword [esp]
	jg	L$56
; This check unnecessary - we're always working with_time_constraints
;	;        if (nodes >= *pnodes) {
;	mov	eax,dword [esp+14H]	; nodes
;	cmp	eax,dword [esp+2cH]	; *pnodes
;	jge	L$54
	;        limit = maxlength - OGR[remdepth];
	mov	eax,dword [esp+20H]	; remdepth
	mov	edx,dword [esp+10H]	; maxlength
	mov	edi,dword [_OGR+eax*4]
	sub	edx,edi			; limit (new)
	;        if (limit > oState->half_length)
	mov	edi,dword [esp+18H]	; oState
	mov	eax,dword [edi+0cH]	; oState->half_length
	cmp	edx,eax
	jle	store_limit
	;          limit = oState->half_length;
	mov	edx,eax
	jmp	store_limit

	align	16
L$56:
	;      else if (limit >= maxlength - levHalfDepth->mark) {
	mov	eax,dword [esp+10H]	; maxlength
	mov	edi,dword [esp+4]	; levHalfDepth
	sub	eax,dword [edi+lev_mark]; levHalfDepth->mark
	cmp	eax,edx
	jg	store_limit
	;        limit = maxlength - levHalfDepth->mark - 1;
	lea	edx,[eax-1]
	jmp	store_limit

	align	16
outerloop:
					; edi must be loaded with dist0
	;    limit = maxlength - choose(dist0 >> ttmDISTBITS, remdepth);
	shr	edi,14H			; dist0
	imul	edi,0cH
	mov	edx,dword [esp+20H]	; remdepth
	xor	eax,eax
	add	edi,dword [esp+1cH]	; ogr_choose_dat+3
	mov	al,byte [edi+edx]	; mov al,byte [_ogr_choose_dat+edi+edx+3]
	mov	edx,dword [esp+10H]	; maxlength
	sub	edx,eax
	;    if (depth <= halfdepth2) {
	mov	eax,dword [esp+24H]	; depth
	cmp	eax,dword [esp+8]	; halfdepth2
	jle	checklimit
store_limit:
	; limit (save on stack) => KILLED
	;    lev->limit = limit;
	mov	dword [ebp+lev_limit],edx
	test	esi, 80000000h		; for align
	;    nodes++;
	inc	dword [esp+14H]

	_natural_align

stay:   ; Most important internal loop
	;
	; entry: ebx = mark   (keep and update!)
	;	 ecx = comp0  (reloaded immediately after shift)
	;	 ebp = level
	;	 esi =
	;	 edi will be dist0  (reloaded later, don't care)
	; usable: eax, edx, esi, edi
	; usable with care: ecx
	;
	;    if (comp0 < 0xfffffffe) {
	cmp	ecx,0fffffffeH
	jnb	L$57_
	;      int s = LOOKUP_FIRSTBLANK( comp0 );
	not	ecx
	mov	eax,20H
	bsr	edx,ecx
	sub	eax,edx			; s
	;      if ((mark += s) > limit) goto up;   /* no spaces left */
	add	ebx,eax
	cmp	ebx,dword [ebp+lev_limit]	; limit (==lev->limit)
	jg	up
	;      COMP_LEFT_LIST_RIGHT(lev, s);

	;    #define COMP_LEFT_LIST_RIGHT(lev, s) {  \
	;      U temp1, temp2;                       \
	;      register int ss = 64 - (s);           \
	;      temp2 = list0 << ss;                  \
	;      list0 = list0 >> (s);                 \
	;      temp1 = list1 << ss;                  \
	;      list1 = (list1 >> (s)) | temp2;       \
	;      temp2 = comp1 >> ss;                  \
	;      list2 = (list2 >> (s)) | temp1;       \
	;      temp1 = comp2 >> ss;                  \
	;      comp0 = (u32) ((comp0 << (s)) | temp2);       \
	;      comp1 = (comp1 << (s)) | temp1;       \
	;      comp2 = (comp2 << (s));               \
	;    }

					; eax=s, ecx = not comp0
	mov	edx,ecx			; copy "not comp0"
	mov	ecx,eax			; ecx=s (for shift below)
	mov	eax,64
	sub	eax,ecx			; eax=ss (64-s)
	not	edx			; recover comp0
	movd	mm6,ecx			; mm6 = s
	movd	mm7,eax			; mm7 = ss

	psrlq	mm4,mm6
	movq	mm5,mm3
	psllq	mm5,mm7
	por	mm4,mm5
	psrlq	mm3,mm6
	movq	mm5,mm2
	psllq	mm5,mm7
	por	mm3,mm5
	psrlq	mm2,mm6

	movq	mm5,mm0
	punpckhdq mm5,mm5
	movd	eax,mm5
	shld	edx,eax,cl
	psllq	mm0,mm6
	movq	mm5,mm1
	psrlq	mm5,mm7
	por	mm0,mm5
	psllq	mm1,mm6
	mov	ecx,edx

L$58:
					; comp0 must be in ecx
	;    lev->mark = mark;
	;    if (remdepth == 0) {                  /* New ruler ? (last mark placed) */
	mov	dword [ebp+lev_mark],ebx	; lev->mark
	cmp	dword [esp+20H],0		; remdepth
	je	L$61
	;    PUSH_LEVEL_UPDATE_STATE(lev);         /* Go deeper */

	;    #define PUSH_LEVEL_UPDATE_STATE(lev)    \
	;      lev->list[0] = list0; dist0 |= list0; \
	;      lev->list[1] = list1; dist1 |= list1; \
	;      lev->list[2] = list2; dist2 |= list2; \
	;      lev->comp[0] = comp0; comp0 |= dist0; \
	;      lev->comp[1] = comp1; comp1 |= dist1; \
	;      lev->comp[2] = comp2; comp2 |= dist2; \
	;      list0 |= ((U)1 << 32);
	;
	; Since dist is not cached, it will be updated in memory similar
	; to "low-register" code:
	;
	; lev2->dist[n] = lev->dist[n] | lev->list[n];
	;
	; Note! new comp0, dist0 must be loaded to ecx, edi

	movd	edi,mm2				; list0
	mov	[ebp+lev_list0],edi		; lev->list[0] = list0
	movq	[ebp+lev_list1],mm3		; lev->list[1] = list1
	movq	[ebp+lev_list2],mm4		; lev->list[2] = list2
	mov	[ebp+lev_comp0],ecx		; lev->comp[0] = comp0
	movq	[ebp+lev_comp1],mm0		; lev->comp[1] = comp1
	movq	[ebp+lev_comp2],mm1		; lev->comp[2] = comp2
	or	edi,[ebp+lev_dist0]		; dist0 |= list0
	mov	[ebp+sizeof_level+lev_dist0],edi ; lev2->dist0
	or	ecx,edi				; comp0 |= dist0
	movq	mm6,[ebp+lev_dist1]		; lev->dist[1]
	por	mm6,mm3				; dist1 |= list1
	movq	[ebp+sizeof_level+lev_dist1],mm6 ; lev2->dist1 = dist1
	por	mm0,mm6				; comp1 |= dist1
	movq	mm6,[ebp+lev_dist2]		; lev->dist[2]
	por	mm6,mm4				; dist2 |= list2
	movq	[ebp+sizeof_level+lev_dist2],mm6 ; lev2->dist2 = dist2
	por	mm1,mm6				; comp2 |= dist2
	mov	eax,1
	movd	mm7,eax
	punpckldq mm2,mm7			; list0 |= ((U)1 << 32);

	;    lev++;
	add	ebp,sizeof_level
	;    depth++;
	inc	dword [esp+24H]
	;    remdepth--;
	dec	dword [esp+20H]
	;    continue;

; When core processed at least given number of nodes (or a little more)
; it must return to main client for rescheduling. It's required for
; timeslicing in non-preemptive OS'es.
; The trick is that there is no difference where to check count of nodes
; we've processed, in the beginning or in the end of loop. +/- 1 iteration
; of the core means nothing for client's scheduler.
; Checking node count here help us do not break perfectly aligned
; command sequence from "outerloop" to "stay".

	mov	eax,dword [esp+14H]	; nodes
	cmp	eax,dword [esp+2cH]	; *pnodes
	jl	outerloop
	xor	eax,eax
	inc	eax		; mov	eax,1 but shorter (bad align below)
	jmp	L$53_exit

	align	16

	;    else {  /* s >= 32 */
	;      if ((mark += 32) > limit) goto up;
L$57_:
	add	ebx,20H
	cmp	ebx,dword [ebp+lev_limit]	; limit (==lev->limit)
	jg	up

	cmp	ecx,0ffffffffH

	; small optimize. in both cases (ecx == -1 and not) we perform
	; COMP_LEFT_LIST_RIGHT_32, then just jump to different labels.

	;    #define COMP_LEFT_LIST_RIGHT_32(lev)      \
	;      list2 = (list2 >> 32) | (list1 << 32);  \
	;      list1 = (list1 >> 32) | (list0 << 32);  \
	;      list0 >>= 32;                           \
	;      comp0 = (u32) (comp1 >> 32);            \
	;      comp1 = (comp1 << 32) | (comp2 >> 32);  \
	;      comp2 <<= 32;

					;  mm2  mm3  mm4     mm2  mm3  mm4
	punpckhdq mm4,mm4		;           6789 =>           6767
	punpckldq mm4,mm3		;      2345 6767 =>      2345 4567
	punpckhdq mm3,mm3		;      2345 4567 =>      2323 4567
	punpckldq mm3,mm2		; NN01 2323 4567 => NN01 0123 4567
	punpckhdq mm2,mm7		; NN01 0123 4567 => ZZNN 0123 4567

	movq	  mm6,mm0		; comp1 [0123] => mm6 (copy)
	punpckhdq mm6,mm6		; [0123] => [0101]
	movd	  ecx,mm6		; comp0 => reloaded to ecx
					;  mm0  mm1  mm6
					; 0123 4567 ----
	psllq	  mm0,32		; 2300 4567 ----
	movq	  mm6,mm1		; 2300 4567 4567
	psrlq	  mm6,32		; 2300 4567 0045
	por	  mm0,mm6		; 2345 4567 ----
	psllq	  mm1,32		; 2345 67ZZ ----

	; Good that MMX do not set ALU flags

	jne	L$58	; ecx != -1
	jmp	stay	; ecx == -1

	align	16
up:
	;    lev--;
	sub	ebp,sizeof_level

	;    #define POP_LEVEL(lev)    \
	;      list0 = lev->list[0];   \
	;      list1 = lev->list[1];   \
	;      list2 = lev->list[2];   \
	;      dist0 &= ~list0;        \
	;      dist1 &= ~list1;        \
	;      dist2 &= ~list2;        \
	;      comp0 = (u32) lev->comp[0];   \
	;      comp1 = lev->comp[1];   \
	;      comp2 = lev->comp[2];

	mov	ecx,dword [ebp+lev_comp0]	; (u32) lev->comp[0]
	movq	mm0,[ebp+lev_comp1]		;	lev->comp[1]
	movq	mm1,[ebp+lev_comp2]		;	lev->comp[2]
	movd	mm2,[ebp+lev_list0]		; (u32) lev->list[0], newbit = 0
	movq	mm3,[ebp+lev_list1]		;	lev->list[1]
	movq	mm4,[ebp+lev_list2]		;	lev->list[2]
	; dist not loaded because it's not cached

	;    limit = lev->limit; KILLED
	;    mark = lev->mark;
	mov	ebx,dword [ebp+lev_mark]
	;    remdepth++;
	inc	dword [esp+20H]
	;    depth--;
	;    if (depth <= 0) {
	dec	dword [esp+24H]
	jg	stay
	xor	eax,eax
L$53_exit:
	;
	; ecx must be loaded with comp0!
	;
	;    #define SAVE_FINAL_STATE(lev)   \
	;      lev->list[0] = list0;         \
	;      lev->list[1] = list1;         \
	;      lev->list[2] = list2;         \
	;      lev->dist[0] = dist0;         \
	;      lev->dist[1] = dist1;         \
	;      lev->dist[2] = dist2;         \
	;      lev->comp[0] = comp0;         \
	;      lev->comp[1] = comp1;         \
	;      lev->comp[2] = comp2;

	movd	[ebp+lev_list0],mm2		; lev->list[0]
	movq	[ebp+lev_list1],mm3
	movq	[ebp+lev_list2],mm4
	mov	[ebp+lev_comp0],ecx		; lev->comp[0] = comp0
	movq	[ebp+lev_comp1],mm0		; lev->comp[1]
	movq	[ebp+lev_comp2],mm1

	mov	dword [ebp+lev_mark],ebx	; mark
	mov	edx,dword [esp+24H]
	dec	edx
	mov	ecx,dword [esp+18H]
	mov	ecx,dword [ecx+18H]
	add	edx,ecx
	mov	ecx,dword [esp+18H]
	mov	dword [ecx+1cH],edx
	mov	edx,dword [esp+14H]
	mov	ecx,dword [esp+0cH]
	mov	dword [ecx],edx
	add	esp,30H
	pop	ebp
	pop	edi
	pop	esi
	pop	ecx
	emms
	ret

L$61:
	mov	eax,dword [esp+18H]
	push	ebx			; preserve regs clobbered by cdecl
	push	ecx
	push	edx
	push	eax			; parameter
	call	[_found_one_cdecl_ptr]
	add	esp, 4
	pop	edx
	pop	ecx
	pop	ebx			; restore clobbered regs
	cmp	eax,1
	jne	L$53_exit
	jmp	stay

_ogr_watcom_rt1_mmx64_asm:
ogr_watcom_rt1_mmx64_asm:
	push	ebx
	mov	eax, [esp+24]		; found_one cdecl procedure
	mov	[_found_one_cdecl_ptr], eax
	mov	eax, [esp+8]		; state
	mov	edx, [esp+12]		; pnodes
	mov	ebx, [esp+16]		; with_time_constraints (ignored, always true)
	mov	ecx, [esp+20]		; address of ogr_choose_dat table
	call	ogr_cycle_
	pop	ebx
	ret
