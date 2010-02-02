;
; Assembly core for OGR-NG, 64bit SSE2 with LZCNT asm version.
; $Id: ogrng64-cj1-sse2-lzcnt-asm.asm,v 1.1 2010/02/02 05:35:23 stream Exp $
;
; Created by Craig Johnston (craig.johnston@dolby.com)
;

%ifdef __NASM_VER__
	cpu	686
%else
	cpu	p3 mmx sse sse2 amd
	BITS	64
%endif

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
	[SECTION _DATA USE32 ALIGN=16 CLASS=DATA]
	[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
	[SECTION .data]
	[SECTION .text]
%endif

%define use_lzcnt
%include "ogrng64-cj1-base.asm"

global	_ogrng64_cycle_256_cj1_sse2_lzcnt
global	ogrng64_cycle_256_cj1_sse2_lzcnt
_ogrng64_cycle_256_cj1_sse2_lzcnt:
ogrng64_cycle_256_cj1_sse2_lzcnt:

	header
	body 13
	footer
