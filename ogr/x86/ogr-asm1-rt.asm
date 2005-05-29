;
; Assembly core for OGR
;
; Based on disassembled output of Watcom compiler which suddenly generated
; code about 15% faster then current Windows client.
;
; Watcom's output was optimizied a little more manually (alas, this compiler
; cannot align loops), but pipeline optimization can be far away from complete.
; This code was developed on and tuned for PII-Celeron CPU, on other processors
; it's performance can be comparable or even less then current cores.
;
; Anyway, on my system this code works about 30% faster then Windows client.
;
; Addintional improvements can be achevied by better optimization of pipelines
; and usage of MMX instruction/registers.
;

cpu	386

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

global	_ogr_watcom_rt1_asm, ogr_watcom_rt1_asm

	%define	sizeof_level 44h		; sizeof(level)

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
;	28H	limit
;	2cH	mark => KILLED. Now contains *pnodes for fast compare.

	mov	ebx,dword [ebp+3cH]	;   int mark = lev->mark;

;    #define SETUP_TOP_STATE(lev)  \
;      U comp0 = lev->comp[0];     \
;      U dist0 = lev->dist[0];     \
;      int newbit = 1;

	mov	ecx,dword [ebp+28H]
	mov	edi,dword [ebp+14H]
	mov	esi,1

	jmp	outerloop

%if ($-$$) <> 7Bh
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
	sub	eax,dword [edi+3cH]	; levHalfDepth->mark
	cmp	eax,edx
	jg	store_limit
	;        limit = maxlength - levHalfDepth->mark - 1;
	lea	edx,[eax-1]
	jmp	store_limit

; see about with_time_constraints above
;L$54:
;	;          retval = CORE_S_CONTINUE;
;	;          break;
; don't forget to store limit when leaving block!
; happily limit not used in return sequence and can be ignored
;	mov	eax,1
;	jmp	L$53

	align	16
outerloop:
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
	mov	dword [esp+28H],edx	; limit (save on stack)
	;    lev->limit = limit;
	mov	dword [ebp+40H],edx
	;    nodes++;
	mov	eax, eax		; for align
	inc	dword [esp+14H]

	_natural_align

stay:   ; Most important internal loop
	;
	; entry: ebx = mark   (keep and update!)
	;	 ecx = comp0  (reloaded immediately after shift)
	;	 ebp = level
	;	 esi = newbit (reloaded immediately after shift)
	;	 edi will be dist0  (reloaded later, don't care)
	; usable: eax, edx, edi
	; usable with care: ecx, esi
	;
	;    if (comp0 < 0xfffffffe) {
	cmp	ecx,0fffffffeH
	jnb	L$57_
	;      int s = LOOKUP_FIRSTBLANK( comp0 );
	not	ecx
	mov	eax,20H
	bsr	ecx,ecx
	sub	eax,ecx			; s
	;      if ((mark += s) > limit) goto up;   /* no spaces left */
	add	ebx,eax
	cmp	ebx,dword [esp+28H]	; limit
	mov	ecx,eax
	jg	up
	;      COMP_LEFT_LIST_RIGHT(lev, s);
%define comp_offset 40			; offsetof(struct Level, comp)
%macro  COMP_LEFT_LIST_RIGHT_cl  2
	; %1 - register with level
	; %2 - register with newbit

	mov	edx,dword [%1+0cH]
	mov	eax,dword [%1+8]
	shrd	dword [%1+10H],edx,cl
	shrd	edx,eax,cl
	mov	edi,dword [%1+4]
	mov	dword [%1+0cH],edx
	shrd	eax,edi,cl
	mov	edx,dword [%1]
	mov	dword [%1+8],eax
	shrd	edi,edx,cl
	shrd	edx,%2,cl		; newbit
	mov	dword [%1],edx
	mov	dword [%1+4],edi

	mov	eax,dword [%1+comp_offset+4]
	mov	edx,dword [%1+comp_offset+8]
	shld	dword [%1+comp_offset],eax,cl
	shld	eax,edx,cl
	mov	dword [%1+comp_offset+4],eax
	mov	eax,dword [%1+comp_offset+0cH]
	shld	edx,eax,cl
	mov	dword [%1+comp_offset+8],edx
	mov	edx,dword [%1+comp_offset+10H]
	shld	eax,edx,cl
	mov	dword [%1+comp_offset+0cH],eax
	shl	edx,cl
	mov	dword [%1+comp_offset+10H],edx

%endmacro
	COMP_LEFT_LIST_RIGHT_cl  ebp, esi	; level in ebp, newbit in esi

	mov	ecx,dword [ebp+28H]	; comp0  = ...
	xor	esi,esi			; newbit = 0
;	and	esi,0			; to align next label

L$58:
	;    lev->mark = mark;
	;    if (remdepth == 0) {                  /* New ruler ? (last mark placed) */
	mov	dword [ebp+3cH],ebx	; lev->mark
	cmp	dword [esp+20H],0	; remdepth
	je	L$61
	;    PUSH_LEVEL_UPDATE_STATE(lev);         /* Go deeper */
	;    #define PUSH_LEVEL_UPDATE_STATE(lev) {            \
	;      U temp1, temp2;                                 \
	;      struct Level *lev2 = lev + 1;                   \
	;      dist0 = (lev2->list[0] = lev->list[0]);         \
	;      temp2 = (lev2->list[1] = lev->list[1]);         \
	;      dist0 = (lev2->dist[0] = lev->dist[0] | dist0); \
	;      temp2 = (lev2->dist[1] = lev->dist[1] | temp2); \
	;      comp0 = lev->comp[0] | dist0;                   \
	;      lev2->comp[0] = comp0;                          \
	;      lev2->comp[1] = lev->comp[1] | temp2;           \
	;      temp1 = (lev2->list[2] = lev->list[2]);         \
	;      temp2 = (lev2->list[3] = lev->list[3]);         \
	;      temp1 = (lev2->dist[2] = lev->dist[2] | temp1); \
	;      temp2 = (lev2->dist[3] = lev->dist[3] | temp2); \
	;      lev2->comp[2] = lev->comp[2] | temp1;           \
	;      lev2->comp[3] = lev->comp[3] | temp2;           \
	;      temp1 = (lev2->list[4] = lev->list[4]);         \
	;      temp1 = (lev2->dist[4] = lev->dist[4] | temp1); \
	;      lev2->comp[4] = lev->comp[4] | temp1;           \
	;      newbit = 1;
	;
	; Note! comp0, dist0 and newbit must be loaded to ecx, edi, esi
	; eax = temp1, edx = temp2

	mov	edi,dword [ebp]			; lev->list[0]
	mov	dword [ebp+sizeof_level],edi		; dist0, lev2->list[0]
	mov	edx,dword [ebp+4]		; lev->list[1]
	mov	dword [ebp+sizeof_level+4],edx	; temp2, lev2->list[0]
	or	edi,dword [ebp+14H]
	mov	dword [ebp+sizeof_level+14H],edi	; dist0 =
	or	edx,dword [ebp+18H]
	mov	dword [ebp+sizeof_level+18H],edx	; temp2 =
	mov	ecx,dword [ebp+28H]
	or	ecx,edi
	mov	dword [ebp+sizeof_level+28H],ecx	; comp0 =
	or	edx,dword [ebp+2cH]
	mov	dword [ebp+sizeof_level+2cH],edx	; temp2
	mov	eax,dword [ebp+8]
	mov	dword [ebp+sizeof_level+8],eax	; temp1 =
	mov	edx,dword [ebp+0cH]
	mov	dword [ebp+sizeof_level+0cH],edx	; temp2 =
	or	eax,dword [ebp+1cH]
	mov	dword [ebp+sizeof_level+1cH],eax	; temp1 =
	or	edx,dword [ebp+20H]
	mov	dword [ebp+sizeof_level+20H],edx	; temp2 =
	or	eax,dword [ebp+30H]
	mov	dword [ebp+sizeof_level+30H],eax	; temp1
	or	edx,dword [ebp+34H]
	mov	dword [ebp+sizeof_level+34H],edx	; temp2
	mov	eax,dword [ebp+10H]
	mov	dword [ebp+sizeof_level+10H],eax	; temp1
	or	eax,dword [ebp+24H]
	mov	dword [ebp+sizeof_level+24H],eax	; temp1
	or	eax,dword [ebp+38H]
	mov	esi,1				; newbit
	mov	dword [ebp+sizeof_level+38H],eax	; temp1
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
; we've processed, in the begining or in the end of loop. +/- 1 iteration
; of the core means nothing for cleint's scheduler.
; Checking nodecount here help us do not break perfecly aligned
; command sequence from "outerloop" to "stay".

	mov	eax,dword [esp+14H]	; nodes
	cmp	eax,dword [esp+2cH]	; *pnodes
	jl	outerloop
	mov	eax,1
	jmp	L$53

	align	16

	;    else {  /* s >= 32 */
	;      if ((mark += 32) > limit) goto up;
L$57_:
	add	ebx,20H
	cmp	ebx,dword [esp+28H]	; limit
	jg	up

	cmp	ecx,0ffffffffH
	; small optimize. in both cases (ecx == -1 and not) we perform
	; COMP_LEFT_LIST_RIGHT_32, then just jump to diferent labels.
	mov	eax,dword [ebp+0cH]
	mov	dword [ebp+10H],eax
	mov	eax,dword [ebp+8]
	mov	dword [ebp+0cH],eax
	mov	eax,dword [ebp+4]
	mov	dword [ebp+8],eax
	mov	eax,dword [ebp]
	mov	dword [ebp+4],eax
	mov	dword [ebp],esi
	mov	esi,0			; newbit. use mov, don't break flags!
	mov	ecx,dword [ebp+2cH]	; note: comp0 is reloaded here to ecx
	mov	dword [ebp+28H],ecx
	mov	eax,dword [ebp+30H]
	mov	dword [ebp+2cH],eax
	mov	eax,dword [ebp+34H]
	mov	dword [ebp+30H],eax
	mov	eax,dword [ebp+38H]
	mov	dword [ebp+34H],eax
	mov	dword [ebp+38H],esi

	jne	L$58	; ecx != -1
	jmp	stay	; ecx == -1

	align	16
up:
	;    lev--;
	sub	ebp,sizeof_level
	;    #define POP_LEVEL(lev)  \
	;      comp0 = lev->comp[0]; \
	;      newbit = 0;
	mov	ecx,dword [ebp+28H]	; comp0  to ecx
	;    limit = lev->limit;
	mov	eax,dword [ebp+40H]
	mov	dword [esp+28H],eax
	;    mark = lev->mark;
	mov	ebx,dword [ebp+3cH]
	;    remdepth++;
	xor	esi,esi			; newbit to esi
	inc	dword [esp+20H]
	;    depth--;
	;    if (depth <= 0) {
	dec	dword [esp+24H]
	jg	stay
	xor	eax,eax
L$53:
	mov	dword [ebp+3cH],ebx	; mark
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
	jne	L$53
	jmp	stay

_ogr_watcom_rt1_asm:
ogr_watcom_rt1_asm:
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
