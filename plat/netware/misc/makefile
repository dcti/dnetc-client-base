#
# Watcom makefile for watcom clib stub-out
#
# $Id: makefile,v 1.1.2.1 2001/01/21 15:10:31 cyp Exp $
#

objlst = nwwatemu.obj nwpre3x.obj
target = nwwatemu.lib #clib3s.lib

wcc386 = wcc386
wpp386 = wpp386

!define p_wcc386opt /ms /wx /we /s /zp1 /5s /zl /zld /fpi87 /os
!define p_asm386opt /ms /we /3s /fpi87 

cc_compiler_cmd = $(wcc386) $(p_wcc386opt) /dN_PLAT_NLM /dN_IAPX386 $[*.c 
pp_compiler_cmd = $(wpp386) $(p_wcc386opt) /dN_PLAT_NLM /dN_IAPX386 $[*.cpp
assembler_cmd = wasm.exe $(p_asm386opt) $[*.asm

.BEFORE
     @set inc386=$(inc_386)
     @set include=$(inc_386)
     @set lib386=$(lib_386)
     @set wcg386=$(code_386)

rebuild: .symbolic
        #always rebuild the whole library
        @for %i in ($(objlst)) do @if exist %i @del %i
        %make $(target)

.asm.obj: 
        $(assembler_cmd)
        @wlib.exe -b -q -t $(target) -+ $[*.obj

.c.obj:
        $(cc_compiler_cmd)
        @wlib.exe -b -q -t $(target) -+ $[*.obj

.cpp.obj 
        $(pp_compiler_cmd)
        @wlib.exe -b -q -t $(target) -+ $[*.obj

$(target) : $(objlst) 
        @for %i in ($(objlst)) do @if exist %i @del %i
