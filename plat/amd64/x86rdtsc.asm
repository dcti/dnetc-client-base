;
; Copyright distributed.net 2005 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; x86 Processor frequency identification for distributed.net effort
; returns output of rdtsc if available
;
; $Id: x86rdtsc.asm,v 1.2 2007/10/22 16:48:30 jlawson Exp $
;
; return u64

BITS 64

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

  push rsi
  push rdi
  push rbp
  push rbx
  push r12
  push r13
  push r14
  push r15

  ; See if CPUID instruction is supported ...
  ; ... Get copies of EFLAGS into eax and ecx
  pushf
  pop rax 
  mov ecx, eax

  ; ... Toggle the ID bit in one copy and store
  ;     to the EFLAGS reg
  xor eax, 200000h
  push rax 
  popf

  ; ... Get the (hopefully modified) EFLAGS
  pushf
  pop rax 

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

  pop r15         
  pop r14         
  pop r13         
  pop r12         
  pop rbx         
  pop rbp         
  pop rdi         
  pop rsi 
	
  rdtsc        			; result is returned in edx:eax
  and rdx, 00000000FFFFFFFFh 
  shl rdx, 32
  and rax, 00000000FFFFFFFFh
  or rax, rdx

  ret

NotSupported:
  pop r15
  pop r14
  pop r13
  pop r12
  pop rbx
  pop rbp
  pop rdi
  pop rsi
  xor rax,rax
  ret
