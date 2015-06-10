;
; Assembly core for OGR-NG, SSE2 version. Using PEXTRD.
; $Id: ogrng-cj1-sse41-asm.asm,v 1.1 2010/02/02 05:35:41 stream Exp $
;
; Created by Craig Johnston (craig.johnston@dolby.com)
;

%ifdef __NASM_VER__
        cpu     p4
%else
        cpu     p4 mmx sse sse2 sse4.1
%endif

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
	[SECTION _DATA USE32 ALIGN=16 CLASS=DATA]
	[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
	[SECTION .data]
	[SECTION .text]
%endif

%define use_pextrd
%include "ogrng-cj1-sse2-base.asm"

global	_ogr_cycle_256_cj1_sse41
global	ogr_cycle_256_cj1_sse41
_ogr_cycle_256_cj1_sse41:
ogr_cycle_256_cj1_sse41:

	header
	body 1
	footer
