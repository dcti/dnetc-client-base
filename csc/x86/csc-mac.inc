; Copyright distributed.net 1997 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; @(#)$Id: csc-mac.inc,v 1.1.2.3 1999/11/07 19:48:55 cyp Exp $
;

%ifdef __OMF__
[SECTION _DATA CLASS=DATA USE32 FLAT PUBLIC ALIGN=16] 
[SECTION _TEXT CLASS=CODE USE32 FLAT PUBLIC ALIGN=16] ;8,16,256,512,...
%define __DATASECT__ [SECTION _DATA]
%define __CODESECT__ [SECTION _TEXT]
%elifdef __ELF__
[SECTION .data align=16]
[SECTION .text align=32]
%define __DATASECT__ [SECTION .data]
%define __CODESECT__ [SECTION .text]
%else
%define __DATASECT__ [SECTION .data]
%define __CODESECT__ [SECTION .text]
%endif

%macro lje 1
       jne %%skipne
       jmp %1
%%skipne:
%endmacro       
%macro ljne 1
       je %%skipe
       jmp %1
%%skipe:       
%endmacro       
%macro ljl 1
       jge %%skipl
       jmp %1
%%skipl:
%endmacro       
%macro ljge 1
       jl %%skipge
       jmp %1
%%skipge:
%endmacro       

%define offset
%define ptr
