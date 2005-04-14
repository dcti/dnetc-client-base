;
; Copyright distributed.net 2005 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; x86 Processor frequency identification for distributed.net effort
; returns output of rdtsc if available
;
; $Id: x86rdtsc.asm,v 1.1.2.3 2005/04/14 22:01:06 jlawson Exp $
;
; return u64

%ifndef __OMF__
  %ifdef OS2
    %define __OMF__
  %endif
%endif
                      
%ifdef __OMF__   ; Watcom+OS/2 or Borland+Win32
[SECTION _DATA CLASS=DATA USE32 PUBLIC ALIGN=16]
[SECTION _TEXT CLASS=CODE USE32 PUBLIC ALIGN=16]
%define __DATASECT__ [SECTION _DATA]
%define __CODESECT__ [SECTION _TEXT]
%else
%define __DATASECT__ [SECTION .data]
%define __CODESECT__ [SECTION .text]
%endif

global          x86rdtsc,_x86rdtsc

__CODESECT__
_x86rdtsc:
x86rdtsc: 

  push ebx
  push ecx
  push edx
  push esi
  push edi

  ; See if CPUID instruction is supported ...
  ; ... Get copies of EFLAGS into eax and ecx
  pushf
  pop eax
  mov ecx, eax

  ; ... Toggle the ID bit in one copy and store
  ;     to the EFLAGS reg
  xor eax, 200000h
  push eax
  popf

  ; ... Get the (hopefully modified) EFLAGS
  pushf
  pop eax

  ; ... Compare and test result
  cmp ecx, eax
  je near NotSupported

Standard:
  xor eax, eax
  cpuid
  cmp eax, 1            ; See if CPUID code 1 is supported
  jb near NotSupported

  mov eax,1
  cpuid
  test edx,00000010h    	; See if Time-Stamp Counter is supported
  jz near NotSupported
	
  pop edi 
  pop esi
  pop edx   
  pop ecx
  pop ebx

  rdtsc        			; result is returned in edx:eax
  ret

NotSupported:
  pop edi
  pop esi
  pop edx
  pop ecx
  pop ebx
  xor eax,eax
  xor edx,edx
  ret
