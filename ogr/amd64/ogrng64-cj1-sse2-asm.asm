;
; Assembly core for OGR-NG, 64bit SSE2 asm version.
;
; Created by Craig Johnston (craig.johnston@dolby.com)
;

%ifdef __NASM_VER__
	cpu	686
%else
	cpu	p3 mmx sse sse2
	BITS	64
%endif

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
	[SECTION _DATA USE32 ALIGN=16 CLASS=DATA]
	[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
	[SECTION .data]
	[SECTION .text]
%endif

%include "ogrng64-cj1-base.asm"

global	_ogrng64_cycle_256_cj1_sse2
global	ogrng64_cycle_256_cj1_sse2
_ogrng64_cycle_256_cj1_sse2:
ogrng64_cycle_256_cj1_sse2:

	header
	body 13
	footer
