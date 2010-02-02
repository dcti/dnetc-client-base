;
; Assembly core for OGR-NG, SSE2 with LZCNT version.
; $Id: ogrng-cj1-sse2-lzcnt-asm.asm,v 1.1 2010/02/02 05:35:36 stream Exp $
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

%define use_lzcnt
%include "ogrng-cj1-sse2-base.asm"

global	_ogr_cycle_256_cj1_sse2_lzcnt
global	ogr_cycle_256_cj1_sse2_lzcnt
_ogr_cycle_256_cj1_sse2_lzcnt:
ogr_cycle_256_cj1_sse2_lzcnt:

	header
	body 1
	footer
