; Copyright distributed.net 1997 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; $Log: convcsc.asm,v $
; Revision 1.1.2.1  1999/11/06 00:26:13  cyp
; they're here! (see also bench.res for 'ideal' combination)
;
;

global          csc_bit_order
global          convert_key_from_csc_to_inc,_convert_key_from_csc_to_inc
global          convert_key_from_inc_to_csc,_convert_key_from_inc_to_csc

%include "csc-mac.inc"

__DATASECT__
    db  "@(#)$Id: convcsc.asm,v 1.1.2.1 1999/11/06 00:26:13 cyp Exp $",0

__DATASECT__
    align 16
csc_bit_order:
    db  0x16, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00
    db  0x26, 0x00, 0x00, 0x00, 0x2e, 0x00, 0x00, 0x00
    db  0x36, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0x00
    db  0x08, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00
    db  0x0a, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00
    db  0x0c, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00
    db  0x0e, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00
    db  0x10, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00
    db  0x12, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00
    db  0x14, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00
    db  0x17, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00
    db  0x19, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00
    db  0x1b, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00
    db  0x1d, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00
    db  0x20, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00
    db  0x22, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00
    db  0x24, 0x00, 0x00, 0x00, 0x25, 0x00, 0x00, 0x00
    db  0x27, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00
    db  0x29, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00
    db  0x2b, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00
    db  0x2d, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00
    db  0x30, 0x00, 0x00, 0x00, 0x31, 0x00, 0x00, 0x00
    db  0x32, 0x00, 0x00, 0x00, 0x33, 0x00, 0x00, 0x00
    db  0x34, 0x00, 0x00, 0x00, 0x35, 0x00, 0x00, 0x00
    db  0x37, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00
    db  0x39, 0x00, 0x00, 0x00, 0x3a, 0x00, 0x00, 0x00
    db  0x3b, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00
    db  0x3d, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00
    db  0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00
    db  0x04, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00
    db  0x06, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00

__CODESECT__
    align 16
convert_key_from_csc_to_inc:
_convert_key_from_csc_to_inc:
    push      ebp
    mov       ebp,esp
    push      ecx
    push      ecx
    mov       edx,dword ptr [ebp+0xc]
    push      ebx
    push      esi
    mov       esi,dword ptr [ebp+0x8]
    xor       ebx,ebx
    push      edi
    mov       dword ptr [ebp-0x8],ebx
    xor       edi,edi
    mov       dword ptr [ebp-0x4],offset csc_bit_order
X$1:
    mov       eax,dword ptr [ebp-0x4]
    mov       ecx,dword ptr [eax]
    cmp       ecx,0x0000001f
    jg        X$2
    mov       eax,dword ptr [edx]
    jmp       X$3
X$2:
    mov       eax,dword ptr [esi]
    add       ecx,0xffffffe0
X$3:
    shr       eax,cl
    and       eax,0x00000001
    cmp       dword ptr [ebp-0x4],offset csc_bit_order+0x7c
    jg        X$4
    mov       ecx,edi
    shl       eax,cl
    or        dword ptr [ebp-0x8],eax
    jmp       X$5
X$4:
    lea       ecx,[edi-0x20]
    shl       eax,cl
    or        ebx,eax
X$5:
    add       dword ptr [ebp-0x4],0x00000004
    inc       edi
    cmp       dword ptr [ebp-0x4],offset csc_bit_order+0x100
    jl        X$1
    mov       eax,dword ptr [ebp-0x8]
    mov       dword ptr [esi],ebx
    pop       edi
    pop       esi
    mov       dword ptr [edx],eax
    pop       ebx
    leave     
    ret       

__CODESECT__
    align 16
convert_key_from_inc_to_csc:
_convert_key_from_inc_to_csc:
    push      ebp
    mov       ebp,esp
    push      ecx
    push      ebx
    push      esi
    mov       esi,dword ptr [ebp+0xc]
    xor       edx,edx
    push      edi
    mov       edi,dword ptr [ebp+0x8]
    mov       dword ptr [ebp-0x4],edx
    xor       ebx,ebx
X$6:
    xor       ecx,ecx
    mov       eax,offset csc_bit_order
X$7:
    cmp       dword ptr [eax],edx
    je        X$8
    add       eax,0x00000004
    inc       ecx
    cmp       eax,offset csc_bit_order+0x100
    jl        X$7
X$8:
    cmp       ecx,0x0000001f
    jg        X$9
    mov       eax,dword ptr [esi]
    jmp       X$10
X$9:
    mov       eax,dword ptr [edi]
    add       ecx,0xffffffe0
X$10:
    shr       eax,cl
    and       eax,0x00000001
    cmp       edx,0x0000001f
    jg        X$11
    mov       ecx,edx
    shl       eax,cl
    or        dword ptr [ebp-0x4],eax
    jmp       X$12
X$11:
    lea       ecx,[edx-0x20]
    shl       eax,cl
    or        ebx,eax
X$12:
    inc       edx
    cmp       edx,0x00000040
    jl        X$6
    mov       eax,dword ptr [ebp-0x4]
    mov       dword ptr [edi],ebx
    mov       dword ptr [esi],eax
    pop       edi
    pop       esi
    pop       ebx
    leave     
    ret       

__CODESECT__
    align 16
