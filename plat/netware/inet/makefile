#
# Watcom makefile for tcpip.nlm and netdb.nlm replacements
# (ie, things that should have been in clib/resolv, but aren't)
#
# $Id: makefile,v 1.1.2.1 2001/01/21 15:10:30 cyp Exp $
#

objlst = nhtohnls.obj in_aton.obj in_ntoa.obj in_makea.obj &
         in_netof.obj in_lnaof.obj hostname.obj hbyname.obj
target = inetlib.lib

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
