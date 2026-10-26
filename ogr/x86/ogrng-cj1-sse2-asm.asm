;
; Assembly core for OGR-NG, SSE2 version.
; $Id: ogrng-cj1-sse2-asm.asm,v 1.8 2010/02/02 05:35:29 stream Exp $
;
; Created by Craig Johnston (craig.johnston@dolby.com)
;

cpu	p4

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
	[SECTION _DATA USE32 ALIGN=16 CLASS=DATA]
	[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
	[SECTION .data]
	[SECTION .text]
%endif

%include "ogrng-cj1-sse2-base.asm"

global	_ogr_cycle_256_cj1_sse2
global	ogr_cycle_256_cj1_sse2
_ogr_cycle_256_cj1_sse2:
ogr_cycle_256_cj1_sse2:

	header
	body 1
	footer
