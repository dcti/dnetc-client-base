; Copyright distributed.net 1997 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; $Log: csc-6b-i.asm,v $
; Revision 1.1.2.3  1999/11/06 05:41:54  gregh
; Remove 'near' specifier on some jumps for compatibility with nasm 0.97.
; It appears that nasm 0.98 ignores superfluous near specifiers.
;
; Revision 1.1.2.2  1999/11/06 05:21:04  daa
; fix typo in global declatation of _csc_unit_func_6b_i
;
; Revision 1.1.2.1  1999/11/06 00:26:14  cyp
; they're here! (see also bench.res for 'ideal' combination)
;
;

global          csc_unit_func_6b_i,_csc_unit_func_6b_i

extern          csc_tabc, csc_tabp, csc_tabe, csc_bit_order
extern          convert_key_from_inc_to_csc, convert_key_from_csc_to_inc

%include "csc-mac.inc"

__DATASECT__
    db  "@(#)$Id: csc-6b-i.asm,v 1.1.2.3 1999/11/06 05:41:54 gregh Exp $",0

__CODESECT__
    align 16
cscipher_bitslicer_6b_i:
    sub       esp,0x000001e0
    mov       eax,dword ptr [esp+0x1e4]
    push      ebx
    push      ebp
    mov       ebp,dword ptr [esp+0x1fc]
    push      esi
    mov       edx,ebp
    push      edi
    add       ebp,0x00000b00
    lea       esi,[eax+0x100]
    mov       ecx,0x00000040
    mov       edi,edx
    mov       dword ptr [esp+0x204],ebp
    repe movsd 
    add       ebp,0x000002c0
    lea       edi,[edx+0x100]
    mov       dword ptr [esp+0x1e8],ebp
    mov       ecx,0x00000040
    mov       esi,eax
    mov       dword ptr [esp+0x1e8],edi
    repe movsd 
    xor       eax,eax
    or        ecx,0xffffffff
    mov       dword ptr [esp+0x1dc],eax
    mov       dword ptr [edx+0x260],eax
    mov       dword ptr [edx+0x280],eax
    mov       dword ptr [edx+0x2a0],eax
    mov       dword ptr [edx+0x2c0],eax
    mov       dword ptr [edx+0x2e0],eax
    lea       eax,[edx+0x200]
    lea       esi,[edx+0x128]
    mov       dword ptr [esp+0x1e4],eax
    mov       dword ptr [esp+0x1e0],edx
    mov       dword ptr [eax],ecx
    lea       eax,[edx+0x204]
    mov       dword ptr [esp+0x18],eax
    mov       eax,csc_tabc+0x28
    lea       ebx,[ebp+0x80]
    mov       dword ptr [edx+0x220],ecx
    mov       dword ptr [edx+0x240],ecx
    mov       dword ptr [esp+0x14],eax
    mov       dword ptr [esp+0x40],esi
    mov       dword ptr [esp+0x28],0x00000007
X$1:
    mov       ecx,dword ptr [esi-0x8]
    mov       edx,dword ptr [eax-0x8]
    mov       edi,dword ptr [eax-0x4]
    xor       ecx,edx
    mov       edx,dword ptr [esi-0x4]
    mov       dword ptr [esp+0x30],ecx
    xor       edx,edi
    mov       edi,dword ptr [eax]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esi]
    xor       edx,edi
    mov       edi,dword ptr [eax+0x4]
    mov       dword ptr [esp+0x24],edx
    mov       edx,dword ptr [esi+0x4]
    xor       edx,edi
    mov       edi,edx
    mov       dword ptr [esp+0x1c],edx
    not       edi
    mov       dword ptr [esp+0x10],edi
    or        ecx,edi
    mov       edi,dword ptr [esi+0x8]
    mov       esi,dword ptr [esi+0x14]
    xor       ecx,edi
    mov       edi,dword ptr [eax+0x8]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x24]
    or        edi,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edi,edx
    mov       edx,dword ptr [eax+0x14]
    xor       esi,edx
    mov       edx,dword ptr [esp+0x78]
    xor       esi,edi
    xor       edx,edi
    mov       edi,dword ptr [esp+0x24]
    or        edx,edi
    mov       edi,dword ptr [esp+0x40]
    mov       edi,dword ptr [edi+0x10]
    xor       edi,dword ptr [eax+0x10]
    xor       edi,edx
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0x30]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x78]
    or        edi,edx
    mov       edx,dword ptr [esp+0x40]
    xor       edi,dword ptr [edx+0xc]
    mov       edx,dword ptr [eax+0xc]
    mov       eax,ecx
    xor       edi,edx
    mov       edx,ecx
    and       eax,edi
    or        edx,esi
    xor       eax,edx
    mov       edx,dword ptr [esp+0x24]
    xor       edx,eax
    mov       dword ptr [esp+0x10],eax
    mov       dword ptr [esp+0x24],edx
    mov       edx,dword ptr [esp+0x20]
    mov       eax,ecx
    mov       dword ptr [esp+0x94],edi
    or        eax,edx
    mov       edx,dword ptr [esp+0x10]
    xor       eax,esi
    mov       dword ptr [esp+0x38],eax
    or        eax,edx
    mov       edx,dword ptr [esp+0x30]
    xor       eax,ecx
    xor       edx,eax
    mov       dword ptr [esp+0x2c],eax
    mov       dword ptr [esp+0x30],edx
    mov       edx,ecx
    or        edx,edi
    mov       edi,dword ptr [esp+0x20]
    mov       eax,ecx
    and       eax,edi
    mov       edi,dword ptr [esp+0x1c]
    xor       edx,eax
    mov       eax,edx
    not       eax
    xor       edi,eax
    mov       eax,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x1c],edi
    or        edx,eax
    mov       eax,dword ptr [esp+0x2c]
    xor       edx,eax
    mov       eax,dword ptr [esp+0x38]
    xor       edx,eax
    mov       eax,dword ptr [esp+0x78]
    not       edx
    xor       eax,edx
    mov       edx,edi
    mov       dword ptr [esp+0x78],eax
    mov       eax,dword ptr [esp+0x24]
    not       edx
    or        eax,edi
    mov       edi,dword ptr [esp+0x78]
    xor       eax,edx
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0x24]
    xor       edi,eax
    xor       esi,eax
    mov       eax,dword ptr [esp+0x20]
    or        edi,edx
    mov       edx,dword ptr [esp+0x18]
    xor       eax,edi
    mov       dword ptr [edx+0xc0],eax
    mov       eax,dword ptr [esp+0x30]
    mov       dword ptr [edx+0xe0],esi
    mov       esi,eax
    xor       esi,edi
    mov       dword ptr [edx],eax
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0x78]
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [edx+0x20],esi
    or        edi,esi
    mov       esi,dword ptr [esp+0x40]
    xor       edi,dword ptr [esp+0x94]
    add       esi,0x00000020
    mov       dword ptr [esp+0x40],esi
    mov       dword ptr [edx+0xa0],edi
    mov       edi,eax
    or        edi,dword ptr [esp+0x2c]
    mov       eax,dword ptr [esp+0x14]
    add       eax,0x00000020
    add       edx,0x00000004
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x1c]
    mov       dword ptr [edx+0x5c],ecx
    mov       ecx,dword ptr [esp+0x24]
    mov       dword ptr [edx+0x3c],ecx
    mov       ecx,dword ptr [esp+0x28]
    mov       dword ptr [edx+0x7c],edi
    dec       ecx
    mov       dword ptr [esp+0x14],eax
    mov       dword ptr [esp+0x18],edx
    mov       dword ptr [esp+0x28],ecx
    ljne       X$1
    mov       esi,dword ptr [esp+0x204]
    mov       edx,0x00000002
    mov       ecx,0x000000e8
    mov       dword ptr [esp+0x10],edx
    lea       eax,[esi+0xe8]
    mov       dword ptr [esp+0x14],0x0000003a
    mov       dword ptr [esp+0x1ec],eax
    mov       dword ptr [esp+0x8c],eax
    mov       eax,dword ptr [esp+0x1e0]
    mov       dword ptr [esp+0x30],csc_tabp+0x5
    mov       dword ptr [esp+0x18],ecx
    lea       edi,[eax+0x208]
    add       eax,0x00000158
    mov       dword ptr [esp+0x94],eax
    mov       eax,dword ptr [esp+0x1f8]
    add       eax,0x00000005
    mov       dword ptr [esp+0x38],edi
    mov       dword ptr [esp+0x40],eax
X$2:
    mov       eax,dword ptr [esp+0x30]
    mov       edi,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x34],0x00000001
    mov       al,byte ptr [eax]
    xor       al,byte ptr [edi]
    mov       byte ptr [esp+0x20],al
    mov       eax,dword ptr [esp+0x20]
    and       eax,0x000000ff
    mov       edi,eax
    mov       al,byte ptr [eax+csc_tabp]
    xor       edi,0x00000040
    xor       al,byte ptr [edi+csc_tabp]
    mov       byte ptr [esp+0x20],al
    mov       eax,dword ptr [esp+0x94]
    mov       dword ptr [ecx+esi-0xe8],eax
    mov       eax,dword ptr [esp+0x20]
    mov       esi,dword ptr [esp+0x38]
    and       eax,0x000000ff
    mov       dword ptr [esp+0x28],eax
    mov       eax,dword ptr [esp+0x8c]
    xor       ecx,ecx
    add       eax,0xffffff1c
    mov       dword ptr [esp+0x1c],esi
X$3:
    mov       edi,dword ptr [esp+0x28]
    mov       esi,0x00000001
    shl       esi,cl
    test      edi,esi
    lje        X$6
    mov       esi,dword ptr [esp+0x1c]
    mov       edi,dword ptr [esp+0x34]
    mov       dword ptr [eax],esi
    inc       edi
    mov       esi,edx
    mov       dword ptr [esp+0x34],edi
    shr       esi,0x00000004
    mov       edi,edx
    add       eax,0x00000004
    shl       esi,0x00000003
    and       edi,0x00000007
    add       eax,0x00000004
    add       edi,esi
    lea       edi,[ebp+edi*4]
    mov       dword ptr [eax-0x4],edi
    mov       edi,dword ptr [esp+0x34]
    inc       edi
    mov       dword ptr [esp+0x34],edi
    mov       edi,edx
    and       edi,0x0000000f
    cmp       edi,0x00000007
    ja        X$4
    add       edi,esi
    lea       esi,[ebx+edi*4]
    mov       dword ptr [eax],esi
    jmp       X$5
X$4:
    lea       edi,[edx+0x1]
    add       eax,0x00000004
    and       edi,0x00000007
    add       edi,esi
    shl       edi,0x00000002
    lea       esi,[edi+ebx]
    mov       dword ptr [eax-0x4],esi
    mov       esi,dword ptr [esp+0x34]
    inc       esi
    test      dl,0x01
    mov       dword ptr [esp+0x34],esi
    je        X$6
    add       edi,ebp
    mov       dword ptr [eax],edi
X$5:
    mov       edi,dword ptr [esp+0x34]
    inc       edi
    add       eax,0x00000004
    mov       dword ptr [esp+0x34],edi
X$6:
    mov       esi,dword ptr [esp+0x1c]
    inc       ecx
    add       esi,0x00000020
    add       edx,0x00000008
    cmp       ecx,0x00000008
    mov       dword ptr [esp+0x1c],esi
    ljl        X$3
    mov       ecx,dword ptr [esp+0x34]
    mov       eax,dword ptr [esp+0x14]
    mov       esi,dword ptr [esp+0x204]
    mov       edi,dword ptr [esp+0x38]
    lea       edx,[eax+ecx-0x3a]
    mov       ecx,dword ptr [esp+0x18]
    add       eax,0x0000001d
    add       edi,0x00000004
    mov       dword ptr [esi+edx*4],0x00000000
    mov       edx,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x14],eax
    mov       eax,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x38],edi
    mov       edi,dword ptr [esp+0x40]
    inc       edx
    add       ecx,0x00000074
    dec       eax
    dec       edi
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esp+0x94]
    mov       dword ptr [esp+0x40],edi
    mov       edi,dword ptr [esp+0x8c]
    add       eax,0x00000020
    add       edi,0x00000074
    cmp       ecx,0x000003a0
    mov       dword ptr [esp+0x10],edx
    mov       dword ptr [esp+0x18],ecx
    mov       dword ptr [esp+0x94],eax
    mov       dword ptr [esp+0x8c],edi
    ljl        X$2
    mov       eax,dword ptr [esp+0x1e4]
    lea       ecx,[ebx+0x8]
    mov       dword ptr [esp+0x10],ecx
    mov       ecx,dword ptr [esp+0x1fc]
    mov       edx,ebx
    add       ecx,0x00000024
    lea       esi,[ebp+0x18]
    sub       edx,ebp
    mov       dword ptr [esp+0x24],esi
    mov       dword ptr [esp+0x18],edx
    mov       dword ptr [esp+0x14],0x00000004
X$7:
    mov       edx,dword ptr [ecx-0x4]
    mov       edi,dword ptr [eax+0x20]
    xor       edx,edi
    mov       edi,dword ptr [ecx]
    mov       dword ptr [esp+0x5c],edx
    mov       edx,dword ptr [eax+0x24]
    xor       edx,edi
    mov       edi,dword ptr [eax+0x28]
    mov       dword ptr [esp+0x80],edx
    mov       edx,dword ptr [ecx+0x4]
    xor       edx,edi
    mov       edi,dword ptr [eax+0x2c]
    mov       dword ptr [esp+0x48],edx
    mov       edx,dword ptr [ecx+0x8]
    xor       edx,edi
    mov       edi,dword ptr [eax+0x30]
    mov       dword ptr [esp+0x34],edx
    mov       edx,dword ptr [ecx+0xc]
    xor       edx,edi
    mov       edi,dword ptr [eax+0x34]
    mov       dword ptr [esp+0x4c],edx
    mov       edx,dword ptr [ecx+0x10]
    xor       edx,edi
    mov       edi,dword ptr [eax+0x38]
    mov       dword ptr [esp+0x50],edx
    mov       edx,dword ptr [ecx+0x14]
    xor       edx,edi
    mov       edi,dword ptr [eax+0x3c]
    mov       dword ptr [esp+0x44],edx
    mov       edx,dword ptr [ecx+0x18]
    xor       edx,edi
    mov       edi,dword ptr [ecx-0x8]
    xor       edi,dword ptr [eax+0x1c]
    mov       dword ptr [esp+0x88],edx
    mov       dword ptr [esp+0x54],edi
    xor       edi,edx
    mov       dword ptr [esi+0x4],edi
    mov       edx,dword ptr [ecx-0xc]
    mov       edi,dword ptr [eax+0x18]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x50]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x18]
    mov       dword ptr [edi+esi],edx
    mov       edi,dword ptr [esp+0x44]
    xor       edx,edi
    mov       dword ptr [esi],edx
    mov       edx,dword ptr [ecx-0x10]
    mov       edi,dword ptr [eax+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x50]
    mov       dword ptr [esp+0x58],edx
    xor       edx,edi
    mov       dword ptr [esi-0x4],edx
    mov       esi,dword ptr [ecx-0x14]
    mov       edi,dword ptr [eax+0x10]
    mov       edx,dword ptr [esp+0x34]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       esi,edx
    mov       edx,dword ptr [esp+0x10]
    mov       dword ptr [edx+0x8],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [edi-0x8],esi
    mov       esi,dword ptr [ecx-0x18]
    mov       edi,dword ptr [eax+0xc]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x34]
    mov       dword ptr [esp+0x7c],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [edi-0xc],esi
    mov       esi,dword ptr [ecx-0x1c]
    mov       edi,dword ptr [eax+0x8]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x80]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [edx],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x24]
    add       edx,0x00000020
    mov       dword ptr [esp+0x10],edx
    add       eax,0x00000040
    mov       dword ptr [edi-0x10],esi
    mov       esi,dword ptr [ecx-0x20]
    mov       edi,dword ptr [eax-0x3c]
    add       ecx,0x00000040
    xor       esi,edi
    mov       edi,dword ptr [esp+0x80]
    mov       dword ptr [esp+0x70],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [edi-0x14],esi
    mov       esi,dword ptr [ecx-0x64]
    mov       edi,dword ptr [eax-0x40]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x88]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x5c]
    mov       dword ptr [edx-0x28],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [edi-0x18],esi
    mov       esi,dword ptr [esp+0x54]
    mov       edi,dword ptr [esp+0x44]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    mov       dword ptr [edx-0xc],esi
    mov       esi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [edx-0x14],esi
    mov       esi,dword ptr [esp+0x7c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x5c]
    mov       dword ptr [edx-0x1c],esi
    mov       esi,dword ptr [esp+0x70]
    xor       esi,edi
    mov       dword ptr [edx-0x24],esi
    mov       esi,dword ptr [esp+0x24]
    mov       edx,dword ptr [esp+0x14]
    add       esi,0x00000020
    dec       edx
    mov       dword ptr [esp+0x24],esi
    mov       dword ptr [esp+0x14],edx
    ljne       X$7
X$8:
    mov       esi,dword ptr [esp+0x1fc]
    mov       edx,dword ptr [ebp+0x4]
    mov       ecx,0x00000040
    lea       edi,[esp+0xdc]
    repe movsd 
    mov       edi,dword ptr [ebp+0xc]
    mov       ecx,dword ptr [ebp]
    mov       esi,dword ptr [ebp+0x8]
    mov       dword ptr [esp+0x38],edi
    not       edi
    mov       eax,ecx
    mov       dword ptr [esp+0xa4],edx
    mov       edx,dword ptr [ebp+0x10]
    or        eax,edi
    xor       eax,edx
    mov       edx,esi
    or        edx,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x94],esi
    xor       edx,edi
    mov       edi,dword ptr [ebp+0x1c]
    xor       edi,edx
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0xa4]
    xor       edi,edx
    mov       edx,dword ptr [ebp+0x18]
    or        edi,esi
    mov       esi,ecx
    xor       edx,edi
    xor       esi,edi
    mov       edi,dword ptr [esp+0xa4]
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [ebp+0x14]
    or        esi,edi
    mov       edi,dword ptr [esp+0x20]
    xor       esi,edx
    mov       edx,eax
    mov       dword ptr [esp+0x28],esi
    or        edx,edi
    mov       edi,eax
    and       edi,esi
    xor       edx,edi
    mov       edi,dword ptr [esp+0x94]
    xor       edi,edx
    mov       dword ptr [esp+0x10],edx
    mov       dword ptr [esp+0x94],edi
    mov       edi,dword ptr [esp+0x1c]
    mov       edx,eax
    or        edx,edi
    mov       edi,dword ptr [esp+0x20]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x18],edx
    or        edx,edi
    mov       edi,dword ptr [esp+0x1c]
    xor       edx,eax
    mov       dword ptr [esp+0x14],edx
    xor       ecx,edx
    mov       edx,eax
    and       edx,edi
    mov       edi,eax
    or        edi,esi
    xor       edx,edi
    mov       edi,dword ptr [esp+0x38]
    mov       esi,edx
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x10]
    or        edx,esi
    mov       esi,dword ptr [esp+0x14]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x18]
    xor       edx,esi
    mov       esi,dword ptr [esp+0xa4]
    not       edx
    xor       esi,edx
    mov       edx,dword ptr [esp+0x94]
    mov       dword ptr [esp+0xa4],esi
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x38],edi
    mov       dword ptr [esp+0x10],esi
    or        edx,edi
    mov       edi,dword ptr [esp+0x94]
    xor       edx,esi
    mov       esi,dword ptr [esp+0xa4]
    xor       esi,edx
    or        esi,edi
    mov       edi,dword ptr [esp+0x20]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x1c]
    xor       edx,esi
    mov       dword ptr [esp+0x118],edi
    mov       edi,dword ptr [esp+0xa4]
    mov       dword ptr [esp+0x114],edx
    mov       edx,ecx
    xor       edx,esi
    mov       esi,dword ptr [esp+0x28]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,esi
    mov       esi,dword ptr [ebx+0x8]
    mov       dword ptr [esp+0x110],edx
    mov       edx,ecx
    or        edx,edi
    xor       edx,eax
    mov       eax,dword ptr [ebx]
    mov       dword ptr [esp+0x10c],edx
    mov       edx,dword ptr [ebx+0x4]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [ebx+0xc]
    mov       dword ptr [esp+0x6c],edx
    mov       edi,eax
    not       edx
    or        edi,edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [ebx+0x10]
    xor       edi,edx
    mov       edx,esi
    mov       dword ptr [esp+0x78],edi
    mov       edi,dword ptr [esp+0x6c]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [ebx+0x1c]
    xor       edi,edx
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0x70]
    xor       edi,edx
    mov       edx,dword ptr [ebx+0x18]
    or        edi,esi
    xor       edx,edi
    mov       dword ptr [esp+0x1c],edx
    mov       edx,eax
    xor       edx,edi
    mov       edi,dword ptr [esp+0x70]
    or        edx,edi
    mov       edi,dword ptr [ebx+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x78]
    and       edi,edx
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x10],edi
    mov       edi,dword ptr [esp+0x78]
    or        edi,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x10],edx
    xor       esi,edx
    mov       edx,dword ptr [esp+0x78]
    or        edx,edi
    mov       edi,dword ptr [esp+0x20]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x2c],edx
    or        edx,edi
    mov       edi,dword ptr [esp+0x78]
    xor       edx,edi
    mov       dword ptr [esp+0x28],edx
    xor       eax,edx
    mov       edx,edi
    mov       edi,dword ptr [esp+0x14]
    or        edx,edi
    mov       edi,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [esp+0x1c]
    and       edi,edx
    mov       edx,dword ptr [esp+0x18]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x18],edx
    not       edx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x6c],edi
    mov       edi,dword ptr [esp+0x10]
    or        edx,edi
    mov       edi,dword ptr [esp+0x28]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x2c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x70]
    not       edx
    xor       edi,edx
    mov       dword ptr [esp+0x70],edi
    mov       edi,dword ptr [esp+0x6c]
    mov       edx,edi
    not       edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,esi
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x70]
    xor       edi,edx
    or        edi,esi
    mov       dword ptr [esp+0x18],edi
    mov       edi,dword ptr [esp+0x20]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0xf8],edi
    mov       edi,dword ptr [esp+0x18]
    xor       edx,edi
    mov       dword ptr [esp+0xf4],edx
    mov       edx,eax
    xor       edx,edi
    mov       edi,dword ptr [esp+0x70]
    or        edx,edi
    mov       edi,dword ptr [esp+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0xf0],edx
    mov       edx,eax
    or        edx,edi
    mov       edi,dword ptr [esp+0x78]
    xor       edx,edi
    mov       edi,dword ptr [ebp+0x20]
    mov       dword ptr [esp+0xec],edx
    mov       edx,dword ptr [ebp+0x24]
    mov       dword ptr [esp+0xb0],edx
    mov       edx,dword ptr [ebp+0x28]
    mov       dword ptr [esp+0xbc],edx
    mov       edx,dword ptr [ebp+0x2c]
    mov       dword ptr [esp+0x8c],edx
    mov       dword ptr [esp+0xa8],edi
    not       edx
    mov       dword ptr [esp+0x10],edx
    or        edi,edx
    mov       edx,dword ptr [ebp+0x30]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xbc]
    mov       dword ptr [esp+0x24],edi
    mov       edi,dword ptr [esp+0x8c]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [ebp+0x3c]
    xor       edi,edx
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0xb0]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xbc]
    or        edi,edx
    mov       edx,dword ptr [ebp+0x38]
    xor       edx,edi
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [esp+0xa8]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xb0]
    or        edx,edi
    mov       edi,dword ptr [ebp+0x34]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x24]
    and       edi,edx
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x10],edi
    mov       edi,dword ptr [esp+0x24]
    or        edi,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xbc]
    xor       edi,edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0xbc],edi
    mov       edi,dword ptr [esp+0x1c]
    or        edx,edi
    mov       edi,dword ptr [esp+0x20]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x2c],edx
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xa8]
    xor       edi,edx
    mov       dword ptr [esp+0x28],edx
    mov       edx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0xa8],edi
    mov       edi,dword ptr [esp+0x1c]
    and       edx,edi
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [esp+0x14]
    or        edi,edx
    mov       edx,dword ptr [esp+0x18]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x18],edx
    not       edx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x8c],edi
    mov       edi,dword ptr [esp+0x10]
    or        edx,edi
    mov       edi,dword ptr [esp+0x28]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x2c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xb0]
    not       edx
    xor       edi,edx
    mov       dword ptr [esp+0xb0],edi
    mov       edi,dword ptr [esp+0x8c]
    mov       edx,edi
    not       edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0xbc]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xb0]
    mov       dword ptr [esp+0x18],edx
    xor       edi,edx
    mov       edx,dword ptr [esp+0xbc]
    or        edi,edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x28],edi
    mov       edi,dword ptr [esp+0x18]
    xor       edx,edi
    mov       dword ptr [esp+0x158],edx
    mov       edx,dword ptr [esp+0x1c]
    mov       edi,dword ptr [esp+0x28]
    xor       edx,edi
    mov       dword ptr [esp+0x154],edx
    mov       edx,dword ptr [esp+0xa8]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xb0]
    or        edx,edi
    mov       edi,dword ptr [esp+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x150],edx
    mov       edx,dword ptr [esp+0xa8]
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edx,edi
    mov       edi,dword ptr [ebx+0x20]
    mov       dword ptr [esp+0x14c],edx
    mov       edx,dword ptr [ebx+0x24]
    mov       dword ptr [esp+0x34],edx
    mov       edx,dword ptr [ebx+0x28]
    mov       dword ptr [esp+0x80],edx
    mov       edx,dword ptr [ebx+0x2c]
    mov       dword ptr [esp+0xa0],edx
    mov       dword ptr [esp+0x78],edi
    not       edx
    or        edi,edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [ebx+0x30]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x80]
    mov       dword ptr [esp+0x24],edi
    mov       edi,dword ptr [esp+0xa0]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [ebx+0x3c]
    xor       edi,edx
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0x34]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x80]
    or        edi,edx
    mov       edx,dword ptr [ebx+0x38]
    xor       edx,edi
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [esp+0x78]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x34]
    or        edx,edi
    mov       edi,dword ptr [ebx+0x34]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x24]
    and       edi,edx
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x10],edi
    mov       edi,dword ptr [esp+0x24]
    or        edi,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x80]
    xor       edi,edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x80],edi
    mov       edi,dword ptr [esp+0x1c]
    or        edx,edi
    mov       edi,dword ptr [esp+0x20]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x2c],edx
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x78]
    xor       edi,edx
    mov       dword ptr [esp+0x28],edx
    mov       edx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x78],edi
    or        edx,dword ptr [esp+0x14]
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [esp+0x1c]
    and       edi,edx
    mov       edx,dword ptr [esp+0x18]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xa0]
    mov       dword ptr [esp+0x18],edx
    not       edx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x18]
    mov       dword ptr [esp+0xa0],edi
    mov       edi,dword ptr [esp+0x10]
    or        edx,edi
    mov       edi,dword ptr [esp+0x28]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x2c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x34]
    not       edx
    xor       edi,edx
    mov       dword ptr [esp+0x34],edi
    mov       edi,dword ptr [esp+0xa0]
    mov       edx,edi
    not       edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x80]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x34]
    xor       edi,edx
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [esp+0x80]
    or        edi,edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x28],edi
    mov       edi,dword ptr [esp+0x18]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x138],edx
    mov       edx,dword ptr [esp+0x1c]
    xor       edx,edi
    mov       dword ptr [esp+0x134],edx
    mov       edx,dword ptr [esp+0x78]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x34]
    or        edx,edi
    mov       edi,dword ptr [esp+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x130],edx
    mov       edx,dword ptr [esp+0x78]
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edx,edi
    mov       edi,dword ptr [ebp+0x40]
    mov       dword ptr [esp+0x12c],edx
    mov       edx,dword ptr [ebp+0x44]
    mov       dword ptr [esp+0x98],edx
    mov       edx,dword ptr [ebp+0x48]
    mov       dword ptr [esp+0x74],edx
    mov       edx,dword ptr [ebp+0x4c]
    mov       dword ptr [esp+0x30],edx
    mov       dword ptr [esp+0x3c],edi
    not       edx
    mov       dword ptr [esp+0x10],edx
    or        edi,edx
    mov       edx,dword ptr [ebp+0x50]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x24],edi
    mov       edi,dword ptr [esp+0x30]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [ebp+0x5c]
    xor       edi,edx
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0x98]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x74]
    or        edi,edx
    mov       edx,dword ptr [ebp+0x58]
    xor       edx,edi
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [esp+0x3c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x98]
    or        edx,edi
    mov       edi,dword ptr [ebp+0x54]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x40],edx
    mov       edx,dword ptr [esp+0x24]
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x40]
    and       edi,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x74]
    xor       edi,edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x74],edi
    mov       edi,dword ptr [esp+0x1c]
    or        edx,edi
    mov       edi,dword ptr [esp+0x20]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x28],edx
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x3c]
    xor       edi,edx
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x3c],edi
    mov       edi,dword ptr [esp+0x1c]
    and       edx,edi
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0x40]
    or        edi,edx
    mov       edx,dword ptr [esp+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x14],edx
    not       edx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x30],edi
    mov       edi,dword ptr [esp+0x10]
    or        edx,edi
    mov       edi,dword ptr [esp+0x18]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x28]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x98]
    not       edx
    xor       edi,edx
    mov       dword ptr [esp+0x98],edi
    mov       edi,dword ptr [esp+0x30]
    mov       edx,edi
    not       edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x74]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x98]
    mov       dword ptr [esp+0x14],edx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x74]
    or        edi,edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x18],edi
    mov       edi,dword ptr [esp+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x198],edx
    mov       edx,dword ptr [esp+0x1c]
    xor       edx,edi
    mov       dword ptr [esp+0x194],edx
    mov       edx,dword ptr [esp+0x3c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x98]
    or        edx,edi
    mov       edi,dword ptr [esp+0x40]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x190],edx
    mov       edx,dword ptr [esp+0x3c]
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edx,edi
    mov       edi,dword ptr [ebx+0x40]
    mov       dword ptr [esp+0x18c],edx
    mov       edx,dword ptr [ebx+0x44]
    mov       dword ptr [esp+0xc0],edx
    mov       edx,dword ptr [ebx+0x48]
    mov       dword ptr [esp+0xc4],edx
    mov       edx,dword ptr [ebx+0x4c]
    mov       dword ptr [esp+0xb4],edx
    mov       dword ptr [esp+0xac],edi
    not       edx
    or        edi,edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [ebx+0x50]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xc4]
    mov       dword ptr [esp+0x24],edi
    mov       edi,dword ptr [esp+0xb4]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [ebx+0x5c]
    xor       edi,edx
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0xc0]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xc4]
    or        edi,edx
    mov       edx,dword ptr [ebx+0x58]
    xor       edx,edi
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [esp+0xac]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xc0]
    or        edx,edi
    mov       edi,dword ptr [ebx+0x54]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x40],edx
    mov       edx,dword ptr [esp+0x24]
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x40]
    and       edi,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xc4]
    xor       edi,edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0xc4],edi
    mov       edi,dword ptr [esp+0x1c]
    or        edx,edi
    mov       edi,dword ptr [esp+0x20]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x28],edx
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xac]
    xor       edi,edx
    mov       dword ptr [esp+0x18],edx
    mov       dword ptr [esp+0xac],edi
    mov       edx,dword ptr [esp+0x24]
    mov       edi,dword ptr [esp+0x1c]
    and       edx,edi
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0x40]
    or        edi,edx
    mov       edx,dword ptr [esp+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xb4]
    mov       dword ptr [esp+0x14],edx
    not       edx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x14]
    mov       dword ptr [esp+0xb4],edi
    mov       edi,dword ptr [esp+0x10]
    or        edx,edi
    mov       edi,dword ptr [esp+0x18]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x28]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xc0]
    not       edx
    xor       edi,edx
    mov       dword ptr [esp+0xc0],edi
    mov       edi,dword ptr [esp+0xb4]
    mov       edx,edi
    not       edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0xc4]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xc0]
    xor       edi,edx
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0xc4]
    or        edi,edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x18],edi
    mov       edi,dword ptr [esp+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x178],edx
    mov       edx,dword ptr [esp+0x1c]
    xor       edx,edi
    mov       dword ptr [esp+0x174],edx
    mov       edx,dword ptr [esp+0xac]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xc0]
    or        edx,edi
    mov       edi,dword ptr [esp+0x40]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x170],edx
    mov       edx,dword ptr [esp+0xac]
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edx,edi
    mov       edi,dword ptr [ebp+0x60]
    mov       dword ptr [esp+0x16c],edx
    mov       edx,dword ptr [ebp+0x64]
    mov       dword ptr [esp+0xc8],edx
    mov       edx,dword ptr [ebp+0x68]
    mov       dword ptr [esp+0x9c],edx
    mov       edx,dword ptr [ebp+0x6c]
    mov       dword ptr [esp+0x40],edx
    mov       dword ptr [esp+0xd8],edi
    not       edx
    mov       dword ptr [esp+0x10],edx
    or        edi,edx
    mov       edx,dword ptr [ebp+0x70]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x9c]
    mov       dword ptr [esp+0x24],edi
    mov       edi,dword ptr [esp+0x40]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [ebp+0x7c]
    xor       edi,edx
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0xc8]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x9c]
    or        edi,edx
    mov       edx,dword ptr [ebp+0x78]
    xor       edx,edi
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [esp+0xd8]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xc8]
    or        edx,edi
    mov       edi,dword ptr [ebp+0x74]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x20]
    mov       dword ptr [esp+0xd4],edx
    mov       edx,dword ptr [esp+0x24]
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0xd4]
    and       edi,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x9c]
    xor       edi,edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x9c],edi
    mov       edi,dword ptr [esp+0x1c]
    or        edx,edi
    mov       edi,dword ptr [esp+0x20]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x28],edx
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xd8]
    xor       edi,edx
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0xd8],edi
    mov       edi,dword ptr [esp+0x1c]
    and       edx,edi
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0xd4]
    or        edi,edx
    mov       edx,dword ptr [esp+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x14],edx
    not       edx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x40],edi
    mov       edi,dword ptr [esp+0x10]
    or        edx,edi
    mov       edi,dword ptr [esp+0x18]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x28]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xc8]
    not       edx
    xor       edi,edx
    mov       dword ptr [esp+0xc8],edi
    mov       edi,dword ptr [esp+0x40]
    mov       edx,edi
    not       edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x9c]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xc8]
    mov       dword ptr [esp+0x14],edx
    xor       edi,edx
    or        edi,dword ptr [esp+0x9c]
    mov       dword ptr [esp+0x18],edi
    mov       edx,dword ptr [esp+0x20]
    mov       edi,dword ptr [esp+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x1d8],edx
    mov       edx,dword ptr [esp+0x1c]
    xor       edx,edi
    mov       dword ptr [esp+0x1d4],edx
    mov       edx,dword ptr [esp+0xd8]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xc8]
    or        edx,edi
    mov       edi,dword ptr [esp+0xd4]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x1d0],edx
    mov       edx,dword ptr [esp+0xd8]
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edx,edi
    mov       edi,dword ptr [ebx+0x60]
    mov       dword ptr [esp+0x1cc],edx
    mov       edx,dword ptr [ebx+0x64]
    mov       dword ptr [esp+0xd0],edx
    mov       edx,dword ptr [ebx+0x68]
    mov       dword ptr [esp+0xb8],edx
    mov       edx,dword ptr [ebx+0x6c]
    mov       dword ptr [esp+0xd4],edx
    mov       dword ptr [esp+0x90],edi
    not       edx
    or        edi,edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [ebx+0x70]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xb8]
    mov       dword ptr [esp+0x24],edi
    mov       edi,dword ptr [esp+0xd4]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [ebx+0x7c]
    xor       edi,edx
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0xd0]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xb8]
    or        edi,edx
    mov       edx,dword ptr [ebx+0x78]
    xor       edx,edi
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [esp+0x90]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xd0]
    or        edx,edi
    mov       edi,dword ptr [ebx+0x74]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x24]
    and       edi,edx
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x10],edi
    mov       edi,dword ptr [esp+0x24]
    or        edi,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xb8]
    xor       edi,edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0xb8],edi
    mov       edi,dword ptr [esp+0x1c]
    or        edx,edi
    mov       edi,dword ptr [esp+0x20]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x2c],edx
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x90]
    mov       dword ptr [esp+0x28],edx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x90],edi
    mov       edi,dword ptr [esp+0x14]
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [esp+0x1c]
    and       edi,edx
    mov       edx,dword ptr [esp+0x18]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xd4]
    mov       dword ptr [esp+0x18],edx
    not       edx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x18]
    mov       dword ptr [esp+0xd4],edi
    mov       edi,dword ptr [esp+0x10]
    or        edx,edi
    mov       edi,dword ptr [esp+0x28]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x2c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xd0]
    not       edx
    xor       edi,edx
    mov       dword ptr [esp+0xd0],edi
    mov       edi,dword ptr [esp+0xd4]
    mov       edx,edi
    not       edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0xb8]
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xd0]
    xor       edi,edx
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [esp+0xb8]
    or        edi,edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x28],edi
    mov       edi,dword ptr [esp+0x18]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x1b8],edx
    mov       edx,dword ptr [esp+0x1c]
    xor       edx,edi
    mov       dword ptr [esp+0x1b4],edx
    mov       edx,dword ptr [esp+0x90]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xd0]
    or        edx,edi
    mov       edi,dword ptr [esp+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x1b0],edx
    mov       edx,dword ptr [esp+0x90]
    or        edx,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x20]
    mov       dword ptr [esp+0x1ac],edx
    mov       edx,dword ptr [esp+0x78]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x24]
    mov       dword ptr [esp+0x5c],edx
    mov       edx,dword ptr [esp+0x34]
    xor       edx,edi
    mov       edi,edx
    mov       edx,dword ptr [esp+0x80]
    xor       edx,dword ptr [csc_tabe+0x28]
    mov       dword ptr [esp+0x48],edx
    mov       edx,dword ptr [esp+0xa0]
    xor       edx,dword ptr [csc_tabe+0x2c]
    mov       dword ptr [esp+0x34],edx
    mov       edx,dword ptr [esp+0x12c]
    xor       edx,dword ptr [csc_tabe+0x30]
    mov       dword ptr [esp+0x4c],edx
    mov       edx,dword ptr [esp+0x130]
    xor       edx,dword ptr [csc_tabe+0x34]
    mov       dword ptr [esp+0x50],edx
    mov       edx,dword ptr [esp+0x134]
    xor       edx,dword ptr [csc_tabe+0x38]
    xor       esi,edi
    mov       dword ptr [esp+0x44],edx
    mov       edx,dword ptr [esp+0x138]
    xor       edx,dword ptr [csc_tabe+0x3c]
    mov       dword ptr [esp+0x88],edx
    xor       eax,edx
    xor       eax,dword ptr [csc_tabe]
    mov       dword ptr [esp+0x84],eax
    mov       edx,eax
    mov       eax,dword ptr [esp+0x5c]
    xor       edx,eax
    mov       eax,dword ptr [esp+0x70]
    xor       eax,dword ptr [csc_tabe+0x4]
    mov       dword ptr [esp+0x70],eax
    xor       eax,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0xcc],eax
    xor       esi,dword ptr [csc_tabe+0x8]
    mov       eax,esi
    mov       dword ptr [esp+0x60],esi
    mov       esi,dword ptr [csc_tabe+0xc]
    xor       eax,edi
    mov       edi,dword ptr [esp+0xec]
    mov       dword ptr [esp+0xa0],eax
    mov       eax,dword ptr [esp+0x6c]
    xor       eax,esi
    mov       dword ptr [esp+0x7c],eax
    mov       esi,eax
    mov       eax,dword ptr [esp+0x34]
    xor       esi,eax
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x10]
    mov       dword ptr [esp+0x20],esi
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x14]
    mov       dword ptr [esp+0x64],eax
    mov       eax,dword ptr [esp+0xf0]
    xor       eax,edi
    mov       edi,dword ptr [esp+0xf4]
    mov       dword ptr [esp+0x58],eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x18]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x1c]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esp+0xf8]
    xor       eax,edi
    mov       edi,esi
    mov       dword ptr [esp+0x54],eax
    mov       eax,edx
    not       edi
    mov       dword ptr [esp+0x10],edi
    or        eax,edi
    mov       edi,dword ptr [esp+0x64]
    xor       eax,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       eax,edi
    mov       edi,dword ptr [esp+0xa0]
    or        edi,esi
    mov       esi,dword ptr [esp+0x10]
    xor       edi,esi
    mov       esi,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x10],edi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x88]
    xor       edi,esi
    mov       esi,dword ptr [esp+0xcc]
    mov       dword ptr [esp+0x1c],edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xa0]
    or        esi,edi
    mov       edi,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x10],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x44]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x78],esi
    mov       esi,edx
    xor       esi,edi
    mov       edi,dword ptr [esp+0xcc]
    or        esi,edi
    mov       edi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x50]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x6c],esi
    mov       esi,eax
    or        esi,edi
    mov       edi,eax
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0x6c]
    and       edi,esi
    mov       esi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xa0]
    xor       edi,esi
    mov       dword ptr [esp+0x10],esi
    mov       dword ptr [esp+0xa0],edi
    mov       edi,dword ptr [esp+0x78]
    mov       esi,eax
    or        esi,edi
    mov       edi,dword ptr [esp+0x1c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x28],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x78]
    xor       esi,eax
    mov       dword ptr [esp+0x18],esi
    xor       edx,esi
    mov       esi,eax
    and       esi,edi
    mov       edi,eax
    mov       dword ptr [esp+0x14],esi
    mov       esi,dword ptr [esp+0x6c]
    or        edi,esi
    mov       esi,dword ptr [esp+0x14]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x14],esi
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0x10]
    or        esi,edi
    mov       edi,dword ptr [esp+0x18]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x28]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xcc]
    not       esi
    xor       edi,esi
    mov       dword ptr [esp+0xcc],edi
    mov       edi,dword ptr [esp+0x20]
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0xa0]
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xcc]
    mov       dword ptr [esp+0x14],esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0xa0]
    or        edi,esi
    mov       esi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x18],edi
    mov       edi,dword ptr [esp+0x14]
    xor       esi,edi
    mov       dword ptr [esp+0x138],esi
    mov       esi,dword ptr [esp+0x78]
    mov       edi,dword ptr [esp+0x18]
    xor       esi,edi
    mov       dword ptr [esp+0x134],esi
    mov       esi,edx
    xor       esi,edi
    mov       edi,dword ptr [esp+0xcc]
    or        esi,edi
    mov       edi,dword ptr [esp+0x6c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x130],esi
    mov       esi,edx
    or        esi,edi
    mov       edi,dword ptr [esp+0x7c]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x70]
    mov       dword ptr [esp+0x12c],esi
    mov       esi,dword ptr [esp+0x5c]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x6c],eax
    mov       eax,dword ptr [esp+0x48]
    xor       edi,eax
    mov       eax,edi
    mov       dword ptr [esp+0x1c],edi
    not       eax
    mov       dword ptr [esp+0x10],eax
    or        eax,esi
    mov       esi,dword ptr [esp+0x64]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x60]
    or        edi,esi
    mov       esi,dword ptr [esp+0x10]
    xor       edi,esi
    mov       esi,edi
    xor       esi,dword ptr [esp+0x54]
    xor       esi,dword ptr [esp+0x44]
    mov       dword ptr [esp+0x78],esi
    mov       esi,dword ptr [esp+0x6c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    or        esi,edi
    mov       edi,esi
    xor       edi,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x24],edi
    mov       edi,dword ptr [esp+0x84]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x6c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       esi,edi
    mov       edi,eax
    and       edi,esi
    mov       dword ptr [esp+0x28],esi
    mov       esi,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x10],edi
    mov       edi,eax
    or        edi,esi
    mov       esi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x10],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x14],esi
    mov       esi,eax
    or        esi,edi
    mov       edi,dword ptr [esp+0x78]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x80],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x84]
    xor       esi,eax
    mov       dword ptr [esp+0x34],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x18],esi
    mov       esi,eax
    or        esi,edi
    mov       edi,eax
    mov       dword ptr [esp+0x2c],esi
    mov       esi,dword ptr [esp+0x24]
    and       edi,esi
    mov       esi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x2c],esi
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x2c]
    mov       dword ptr [esp+0x1c],edi
    mov       edi,dword ptr [esp+0x10]
    or        esi,edi
    mov       edi,dword ptr [esp+0x34]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x80]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x6c]
    not       esi
    xor       edi,esi
    mov       dword ptr [esp+0x6c],edi
    mov       edi,dword ptr [esp+0x1c]
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0x14]
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x2c],esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x14]
    or        edi,esi
    mov       esi,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x34],edi
    mov       edi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x34]
    mov       dword ptr [esp+0xf8],esi
    mov       esi,dword ptr [esp+0x24]
    xor       esi,edi
    mov       dword ptr [esp+0xf4],esi
    mov       esi,dword ptr [esp+0x18]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x6c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x28]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0xf0],esi
    mov       esi,dword ptr [esp+0x18]
    or        esi,edi
    mov       edi,dword ptr [csc_tabe+0x64]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x90]
    mov       dword ptr [esp+0xec],esi
    mov       esi,dword ptr [csc_tabe+0x60]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0x68]
    mov       dword ptr [esp+0x5c],eax
    mov       eax,dword ptr [esp+0xd0]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x70]
    mov       dword ptr [esp+0x80],eax
    mov       eax,dword ptr [esp+0xb8]
    xor       eax,esi
    mov       esi,dword ptr [esp+0xd4]
    mov       dword ptr [esp+0x48],eax
    mov       eax,dword ptr [csc_tabe+0x6c]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x1ac]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x74]
    mov       dword ptr [esp+0x4c],eax
    mov       eax,dword ptr [esp+0x1b0]
    xor       eax,edi
    mov       dword ptr [esp+0x50],eax
    mov       eax,dword ptr [esp+0x1b4]
    mov       edi,dword ptr [csc_tabe+0x78]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x7c]
    mov       dword ptr [esp+0x44],eax
    mov       eax,dword ptr [esp+0x1b8]
    xor       eax,edi
    mov       edi,eax
    mov       eax,dword ptr [esp+0xac]
    xor       eax,edi
    mov       dword ptr [esp+0x88],edi
    mov       edi,dword ptr [csc_tabe+0x40]
    xor       eax,edi
    mov       edi,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x84],eax
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x44]
    mov       dword ptr [esp+0xb8],eax
    mov       eax,dword ptr [esp+0xc0]
    xor       eax,edi
    mov       edi,dword ptr [esp+0x80]
    mov       dword ptr [esp+0x70],eax
    xor       eax,edi
    mov       dword ptr [esp+0xc0],eax
    mov       eax,dword ptr [esp+0xc4]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x48]
    xor       eax,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x60],eax
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x4c]
    mov       dword ptr [esp+0xd0],eax
    mov       eax,dword ptr [esp+0xb4]
    xor       eax,edi
    mov       dword ptr [esp+0x7c],eax
    mov       edi,eax
    mov       eax,dword ptr [esp+0x16c]
    xor       edi,esi
    xor       esi,eax
    mov       eax,dword ptr [csc_tabe+0x50]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x170]
    mov       dword ptr [esp+0x64],esi
    mov       esi,dword ptr [csc_tabe+0x54]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x174]
    mov       dword ptr [esp+0x58],eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0x58]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0x5c]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esp+0x178]
    xor       eax,esi
    mov       esi,edi
    mov       dword ptr [esp+0x54],eax
    mov       eax,dword ptr [esp+0xb8]
    not       esi
    mov       dword ptr [esp+0x10],esi
    or        eax,esi
    mov       esi,dword ptr [esp+0x64]
    mov       dword ptr [esp+0xd4],edi
    xor       eax,esi
    mov       esi,dword ptr [esp+0x4c]
    xor       eax,esi
    mov       esi,dword ptr [esp+0xd0]
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x10],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x88]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0xb4],esi
    mov       esi,dword ptr [esp+0xc0]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xd0]
    or        esi,edi
    mov       edi,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x10],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x44]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0xac],esi
    mov       esi,dword ptr [esp+0xb8]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xc0]
    or        esi,edi
    mov       edi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x50]
    xor       esi,edi
    mov       edi,eax
    and       edi,esi
    mov       dword ptr [esp+0x28],esi
    mov       esi,dword ptr [esp+0xb4]
    mov       dword ptr [esp+0x10],edi
    mov       edi,eax
    or        edi,esi
    mov       esi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xd0]
    xor       edi,esi
    mov       dword ptr [esp+0x10],esi
    mov       dword ptr [esp+0xd0],edi
    mov       edi,dword ptr [esp+0xac]
    mov       esi,eax
    or        esi,edi
    mov       edi,dword ptr [esp+0xb4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x90],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0xb8]
    xor       esi,eax
    xor       edi,esi
    mov       dword ptr [esp+0x78],esi
    mov       dword ptr [esp+0xb8],edi
    mov       edi,dword ptr [esp+0xac]
    mov       esi,eax
    and       esi,edi
    mov       edi,eax
    mov       dword ptr [esp+0x2c],esi
    mov       esi,dword ptr [esp+0x28]
    or        edi,esi
    mov       esi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xd4]
    mov       dword ptr [esp+0x2c],esi
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x2c]
    mov       dword ptr [esp+0xd4],edi
    mov       edi,dword ptr [esp+0x10]
    or        esi,edi
    mov       edi,dword ptr [esp+0x78]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x90]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xc0]
    not       esi
    xor       edi,esi
    mov       dword ptr [esp+0xc0],edi
    mov       edi,dword ptr [esp+0xd4]
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0xd0]
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xc0]
    mov       dword ptr [esp+0x2c],esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0xd0]
    or        edi,esi
    mov       esi,dword ptr [esp+0xb4]
    mov       dword ptr [esp+0x78],edi
    mov       edi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x1b8],esi
    mov       esi,dword ptr [esp+0xac]
    xor       esi,edi
    mov       dword ptr [esp+0x1b4],esi
    mov       esi,dword ptr [esp+0xb8]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xc0]
    or        esi,edi
    mov       edi,dword ptr [esp+0x28]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x1b0],esi
    mov       esi,dword ptr [esp+0xb8]
    or        esi,edi
    mov       edi,dword ptr [esp+0x7c]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x70]
    mov       dword ptr [esp+0x1ac],esi
    mov       esi,dword ptr [esp+0x5c]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0xc4],eax
    mov       eax,dword ptr [esp+0x48]
    xor       edi,eax
    mov       eax,edi
    mov       dword ptr [esp+0xb4],edi
    not       eax
    mov       dword ptr [esp+0x10],eax
    or        eax,esi
    mov       esi,dword ptr [esp+0x64]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x60]
    or        edi,esi
    mov       esi,dword ptr [esp+0x10]
    xor       edi,esi
    mov       esi,edi
    xor       esi,dword ptr [esp+0x54]
    xor       esi,dword ptr [esp+0x44]
    mov       dword ptr [esp+0xac],esi
    mov       esi,dword ptr [esp+0xc4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    or        esi,edi
    mov       edi,esi
    xor       edi,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x78],edi
    mov       edi,dword ptr [esp+0x84]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xc4]
    or        esi,edi
    mov       edi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       esi,edi
    mov       edi,eax
    and       edi,esi
    mov       dword ptr [esp+0x90],esi
    mov       esi,dword ptr [esp+0xac]
    mov       dword ptr [esp+0x10],edi
    mov       edi,eax
    or        edi,esi
    mov       esi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x10],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x28],esi
    mov       esi,eax
    or        esi,edi
    mov       edi,dword ptr [esp+0xac]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x80],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x84]
    xor       esi,eax
    mov       dword ptr [esp+0x34],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x90]
    mov       dword ptr [esp+0x2c],esi
    mov       esi,eax
    or        esi,edi
    mov       edi,eax
    mov       dword ptr [esp+0x24],esi
    mov       esi,dword ptr [esp+0x78]
    and       edi,esi
    mov       esi,dword ptr [esp+0x24]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xb4]
    mov       dword ptr [esp+0x24],esi
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x24]
    mov       dword ptr [esp+0xb4],edi
    mov       edi,dword ptr [esp+0x10]
    or        esi,edi
    mov       edi,dword ptr [esp+0x34]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x80]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xc4]
    not       esi
    xor       edi,esi
    mov       dword ptr [esp+0xc4],edi
    mov       edi,dword ptr [esp+0xb4]
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0x28]
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xc4]
    mov       dword ptr [esp+0x24],esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x28]
    or        edi,esi
    mov       esi,dword ptr [esp+0xac]
    mov       dword ptr [esp+0x34],edi
    mov       edi,dword ptr [esp+0x24]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x34]
    mov       dword ptr [esp+0x178],esi
    mov       esi,dword ptr [esp+0x78]
    xor       esi,edi
    mov       dword ptr [esp+0x174],esi
    mov       esi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xc4]
    or        esi,edi
    mov       edi,dword ptr [esp+0x90]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x170],esi
    mov       esi,dword ptr [esp+0x2c]
    or        esi,edi
    mov       edi,dword ptr [csc_tabe+0xa4]
    xor       esi,eax
    mov       eax,dword ptr [esp+0xa8]
    mov       dword ptr [esp+0x16c],esi
    mov       esi,dword ptr [csc_tabe+0xa0]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0xa8]
    mov       dword ptr [esp+0x5c],eax
    mov       eax,dword ptr [esp+0xb0]
    xor       eax,edi
    mov       edi,eax
    mov       eax,dword ptr [esp+0xbc]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x48],eax
    mov       eax,dword ptr [csc_tabe+0xac]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x14c]
    xor       eax,dword ptr [csc_tabe+0xb0]
    mov       dword ptr [esp+0x4c],eax
    mov       eax,dword ptr [esp+0x150]
    xor       eax,dword ptr [csc_tabe+0xb4]
    mov       dword ptr [esp+0x50],eax
    mov       eax,dword ptr [esp+0x154]
    xor       eax,dword ptr [csc_tabe+0xb8]
    mov       dword ptr [esp+0x44],eax
    mov       eax,dword ptr [esp+0x158]
    xor       eax,dword ptr [csc_tabe+0xbc]
    mov       dword ptr [esp+0x88],eax
    xor       ecx,eax
    xor       ecx,dword ptr [csc_tabe+0x80]
    mov       dword ptr [esp+0x84],ecx
    mov       eax,ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xa4]
    xor       ecx,dword ptr [csc_tabe+0x84]
    mov       dword ptr [esp+0x70],ecx
    xor       ecx,edi
    mov       dword ptr [esp+0x24],ecx
    mov       ecx,dword ptr [esp+0x94]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x88]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x60],ecx
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x38]
    xor       edi,dword ptr [csc_tabe+0x8c]
    mov       dword ptr [esp+0x7c],edi
    xor       edi,esi
    mov       dword ptr [esp+0x78],edi
    mov       edi,dword ptr [esp+0x10c]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x90]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x94]
    mov       dword ptr [esp+0x64],esi
    mov       esi,dword ptr [esp+0x110]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x114]
    mov       dword ptr [esp+0x58],esi
    mov       esi,dword ptr [esp+0x50]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x98]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x9c]
    mov       dword ptr [esp+0x68],esi
    mov       esi,dword ptr [esp+0x118]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x54],esi
    mov       esi,eax
    not       edi
    mov       dword ptr [esp+0x10],edi
    or        esi,edi
    mov       edi,dword ptr [esp+0x64]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x78]
    mov       dword ptr [esp+0xa8],esi
    mov       esi,ecx
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x10],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x88]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x38],esi
    mov       esi,dword ptr [esp+0x24]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x68]
    or        esi,ecx
    mov       dword ptr [esp+0x10],esi
    xor       esi,edi
    xor       esi,dword ptr [esp+0x44]
    mov       dword ptr [esp+0x94],esi
    mov       edi,dword ptr [esp+0x10]
    mov       esi,eax
    xor       esi,edi
    mov       edi,dword ptr [esp+0x24]
    or        esi,edi
    mov       edi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x50]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x8c],esi
    mov       esi,dword ptr [esp+0xa8]
    or        esi,edi
    mov       edi,dword ptr [esp+0xa8]
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0x8c]
    and       edi,esi
    mov       esi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x94]
    mov       dword ptr [esp+0x10],esi
    xor       ecx,esi
    mov       esi,dword ptr [esp+0xa8]
    or        esi,edi
    mov       edi,dword ptr [esp+0x38]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0xa4],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0xa8]
    xor       esi,edi
    mov       dword ptr [esp+0x90],esi
    xor       eax,esi
    mov       esi,edi
    mov       edi,dword ptr [esp+0x94]
    and       esi,edi
    mov       edi,dword ptr [esp+0xa8]
    mov       dword ptr [esp+0xac],esi
    mov       esi,dword ptr [esp+0x8c]
    or        edi,esi
    mov       esi,dword ptr [esp+0xac]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x78]
    mov       dword ptr [esp+0xac],esi
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0xac]
    mov       dword ptr [esp+0x78],edi
    mov       edi,dword ptr [esp+0x10]
    or        esi,edi
    mov       edi,dword ptr [esp+0x90]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xa4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x24]
    not       esi
    xor       edi,esi
    mov       dword ptr [esp+0x24],edi
    mov       edi,dword ptr [esp+0x78]
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x10],esi
    mov       esi,ecx
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edi,esi
    or        edi,ecx
    mov       dword ptr [esp+0xac],edi
    mov       edi,dword ptr [esp+0x38]
    xor       edi,esi
    mov       esi,dword ptr [esp+0x94]
    mov       dword ptr [esp+0x158],edi
    mov       edi,dword ptr [esp+0xac]
    xor       esi,edi
    mov       dword ptr [esp+0x154],esi
    mov       esi,eax
    xor       esi,edi
    mov       edi,dword ptr [esp+0x24]
    or        esi,edi
    mov       edi,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x144],ecx
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x150],esi
    mov       esi,eax
    or        esi,edi
    mov       edi,dword ptr [esp+0xa8]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x14c],esi
    mov       esi,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x148],esi
    mov       esi,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x140],esi
    mov       esi,dword ptr [esp+0x70]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0xbc],esi
    mov       esi,dword ptr [esp+0x7c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x94],esi
    mov       dword ptr [esp+0x13c],eax
    not       esi
    mov       dword ptr [esp+0x10],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x64]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0xa4],esi
    mov       esi,dword ptr [esp+0x94]
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x10],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x44]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x38],esi
    mov       esi,dword ptr [esp+0xbc]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    or        esi,edi
    mov       edi,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x10],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x8c],esi
    mov       esi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xbc]
    or        esi,edi
    mov       edi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x38]
    mov       dword ptr [esp+0xa8],esi
    mov       esi,dword ptr [esp+0xa4]
    or        esi,edi
    mov       edi,dword ptr [esp+0xa4]
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0xa8]
    and       edi,esi
    mov       esi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x90],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0xa4]
    or        esi,edi
    mov       edi,dword ptr [esp+0x38]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x90]
    mov       dword ptr [esp+0x80],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0xa4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x34],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0xac],esi
    mov       esi,dword ptr [esp+0xa4]
    and       esi,edi
    mov       edi,dword ptr [esp+0xa4]
    mov       dword ptr [esp+0xb0],esi
    mov       esi,dword ptr [esp+0xa8]
    or        edi,esi
    mov       esi,dword ptr [esp+0xb0]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x94]
    mov       dword ptr [esp+0xb0],esi
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0xb0]
    mov       dword ptr [esp+0x94],edi
    mov       edi,dword ptr [esp+0x90]
    or        esi,edi
    mov       edi,dword ptr [esp+0x34]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x80]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xbc]
    not       esi
    xor       edi,esi
    mov       dword ptr [esp+0xbc],edi
    mov       edi,dword ptr [esp+0x94]
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x90],esi
    mov       esi,dword ptr [esp+0x10]
    or        esi,edi
    mov       edi,dword ptr [esp+0x90]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xbc]
    mov       dword ptr [esp+0xb0],esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x10]
    or        edi,esi
    mov       esi,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x34],edi
    mov       edi,dword ptr [esp+0xb0]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x34]
    mov       dword ptr [esp+0x118],esi
    mov       esi,dword ptr [esp+0x8c]
    xor       esi,edi
    mov       dword ptr [esp+0x114],esi
    mov       esi,dword ptr [esp+0xac]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xbc]
    or        esi,edi
    mov       edi,dword ptr [esp+0xa8]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x90]
    mov       dword ptr [esp+0x110],esi
    mov       esi,dword ptr [esp+0xac]
    or        esi,edi
    mov       edi,dword ptr [esp+0xa4]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xe0]
    mov       dword ptr [esp+0x10c],esi
    mov       esi,dword ptr [esp+0xd8]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xe4]
    mov       dword ptr [esp+0x5c],esi
    mov       esi,dword ptr [esp+0xc8]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xe8]
    mov       dword ptr [esp+0x80],esi
    mov       esi,dword ptr [esp+0x9c]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xec]
    mov       dword ptr [esp+0x48],esi
    mov       esi,dword ptr [esp+0x40]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xf0]
    mov       dword ptr [esp+0x34],esi
    mov       esi,dword ptr [esp+0x1cc]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xf4]
    mov       dword ptr [esp+0x4c],esi
    mov       esi,dword ptr [esp+0x1d0]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xf8]
    mov       dword ptr [esp+0x50],esi
    mov       esi,dword ptr [esp+0x1d4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x1d8]
    mov       dword ptr [esp+0x44],esi
    mov       esi,dword ptr [csc_tabe+0xfc]
    xor       edi,esi
    mov       esi,dword ptr [esp+0x3c]
    xor       esi,edi
    mov       dword ptr [esp+0x88],edi
    mov       edi,dword ptr [csc_tabe+0xc0]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x84],esi
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xc4]
    mov       dword ptr [esp+0xa4],esi
    mov       esi,dword ptr [esp+0x98]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x80]
    mov       dword ptr [esp+0x70],esi
    xor       esi,edi
    mov       dword ptr [esp+0xb0],esi
    mov       esi,dword ptr [esp+0x74]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xc8]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x60],esi
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xcc]
    mov       dword ptr [esp+0x90],esi
    mov       esi,dword ptr [esp+0x30]
    xor       esi,edi
    mov       dword ptr [esp+0x7c],esi
    mov       edi,esi
    mov       esi,dword ptr [esp+0x34]
    xor       edi,esi
    mov       dword ptr [esp+0xd8],edi
    mov       edi,dword ptr [esp+0x18c]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xd0]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xd4]
    mov       dword ptr [esp+0x64],esi
    mov       esi,dword ptr [esp+0x190]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x194]
    mov       dword ptr [esp+0x58],esi
    mov       esi,dword ptr [esp+0x50]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xd8]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xdc]
    mov       dword ptr [esp+0x68],esi
    mov       esi,dword ptr [esp+0x198]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xd8]
    mov       dword ptr [esp+0x54],esi
    mov       esi,dword ptr [esp+0xa4]
    not       edi
    mov       dword ptr [esp+0x38],edi
    or        esi,edi
    mov       edi,dword ptr [esp+0x64]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xd8]
    mov       dword ptr [esp+0x74],esi
    mov       esi,dword ptr [esp+0x90]
    or        esi,edi
    mov       edi,dword ptr [esp+0x38]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x30],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x88]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x38],esi
    mov       esi,dword ptr [esp+0xb0]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x90]
    or        esi,edi
    mov       edi,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x40],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x44]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x30],esi
    mov       esi,dword ptr [esp+0xa4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xb0]
    or        esi,edi
    mov       edi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x50]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x40],esi
    mov       esi,dword ptr [esp+0x74]
    or        esi,edi
    mov       edi,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x8c],esi
    mov       esi,dword ptr [esp+0x40]
    and       edi,esi
    mov       esi,dword ptr [esp+0x8c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x90]
    xor       edi,esi
    mov       dword ptr [esp+0x8c],esi
    mov       esi,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x90],edi
    mov       edi,dword ptr [esp+0x30]
    or        esi,edi
    mov       edi,dword ptr [esp+0x38]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x98],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x74]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xa4]
    xor       edi,esi
    mov       dword ptr [esp+0xa8],esi
    mov       esi,dword ptr [esp+0x74]
    mov       dword ptr [esp+0xa4],edi
    mov       edi,dword ptr [esp+0x30]
    and       esi,edi
    mov       edi,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x3c],esi
    mov       esi,dword ptr [esp+0x40]
    or        edi,esi
    mov       esi,dword ptr [esp+0x3c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xd8]
    mov       dword ptr [esp+0x3c],esi
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0xd8],edi
    mov       edi,dword ptr [esp+0x8c]
    or        esi,edi
    mov       edi,dword ptr [esp+0xa8]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xb0]
    not       esi
    xor       edi,esi
    mov       dword ptr [esp+0xb0],edi
    mov       edi,dword ptr [esp+0xd8]
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x8c],esi
    mov       esi,dword ptr [esp+0x90]
    or        esi,edi
    mov       edi,dword ptr [esp+0x8c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xb0]
    mov       dword ptr [esp+0x3c],esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x90]
    or        edi,esi
    mov       esi,dword ptr [esp+0x38]
    mov       dword ptr [esp+0xa8],edi
    mov       edi,dword ptr [esp+0x3c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xa8]
    mov       dword ptr [esp+0x1d8],esi
    mov       esi,dword ptr [esp+0x30]
    xor       esi,edi
    mov       dword ptr [esp+0x1d4],esi
    mov       esi,dword ptr [esp+0xa4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xb0]
    or        esi,edi
    mov       edi,dword ptr [esp+0x40]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x1d0],esi
    mov       esi,dword ptr [esp+0xa4]
    or        esi,edi
    mov       edi,dword ptr [esp+0x74]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x1cc],esi
    mov       esi,dword ptr [esp+0x70]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0xc8],esi
    mov       esi,dword ptr [esp+0x7c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x40],esi
    not       esi
    mov       dword ptr [esp+0x38],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x64]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x74],esi
    mov       esi,dword ptr [esp+0x40]
    or        esi,edi
    mov       edi,dword ptr [esp+0x38]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x30],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x44]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x38],esi
    mov       esi,dword ptr [esp+0xc8]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    or        esi,edi
    mov       edi,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x8c],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x30],esi
    mov       esi,dword ptr [esp+0x8c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xc8]
    or        esi,edi
    mov       edi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x98],esi
    and       edi,esi
    mov       esi,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x8c],edi
    mov       edi,dword ptr [esp+0x74]
    or        edi,esi
    mov       esi,dword ptr [esp+0x8c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x3c],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x30]
    mov       dword ptr [esp+0xa8],esi
    mov       esi,dword ptr [esp+0x74]
    or        esi,edi
    mov       edi,dword ptr [esp+0x38]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x80],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x74]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x34],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    mov       dword ptr [esp+0x8c],esi
    mov       esi,dword ptr [esp+0x74]
    or        esi,edi
    mov       edi,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x9c],esi
    mov       esi,dword ptr [esp+0x30]
    and       edi,esi
    mov       esi,dword ptr [esp+0x9c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x9c],esi
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x9c]
    mov       dword ptr [esp+0x40],edi
    mov       edi,dword ptr [esp+0x3c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x34]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x80]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xc8]
    not       esi
    xor       edi,esi
    mov       dword ptr [esp+0xc8],edi
    mov       edi,dword ptr [esp+0x40]
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x3c],esi
    mov       esi,dword ptr [esp+0xa8]
    or        esi,edi
    mov       edi,dword ptr [esp+0x3c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xc8]
    mov       dword ptr [esp+0x9c],esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0xa8]
    or        edi,esi
    mov       esi,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x34],edi
    mov       edi,dword ptr [esp+0x9c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x34]
    mov       dword ptr [esp+0x198],esi
    mov       esi,dword ptr [esp+0x30]
    xor       esi,edi
    mov       dword ptr [esp+0x194],esi
    mov       esi,dword ptr [esp+0x8c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xc8]
    or        esi,edi
    mov       edi,dword ptr [esp+0x98]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x190],esi
    mov       esi,dword ptr [esp+0x8c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x74]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x120]
    mov       dword ptr [esp+0x18c],esi
    mov       esi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x124]
    mov       dword ptr [esp+0x5c],esi
    mov       esi,dword ptr [esp+0xc4]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x128]
    mov       dword ptr [esp+0x80],esi
    mov       esi,dword ptr [esp+0x28]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x12c]
    mov       dword ptr [esp+0x48],esi
    mov       esi,dword ptr [esp+0xb4]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x130]
    mov       dword ptr [esp+0x34],esi
    mov       esi,dword ptr [esp+0x16c]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x134]
    mov       dword ptr [esp+0x4c],esi
    mov       esi,dword ptr [esp+0x170]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x138]
    mov       dword ptr [esp+0x50],esi
    mov       esi,dword ptr [esp+0x174]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x13c]
    mov       dword ptr [esp+0x44],esi
    mov       esi,dword ptr [esp+0x178]
    xor       esi,edi
    mov       edi,esi
    mov       esi,dword ptr [esp+0x18]
    xor       esi,edi
    mov       dword ptr [esp+0x88],edi
    mov       edi,dword ptr [csc_tabe+0x100]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x84],esi
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x104]
    mov       dword ptr [esp+0x9c],esi
    mov       esi,dword ptr [esp+0x6c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x80]
    mov       dword ptr [esp+0x70],esi
    xor       esi,edi
    mov       dword ptr [esp+0x98],esi
    mov       esi,dword ptr [esp+0x14]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x108]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x60],esi
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x10c]
    mov       dword ptr [esp+0x74],esi
    mov       esi,dword ptr [esp+0x1c]
    xor       esi,edi
    mov       dword ptr [esp+0x7c],esi
    mov       edi,esi
    mov       esi,dword ptr [esp+0x34]
    xor       edi,esi
    mov       dword ptr [esp+0x3c],edi
    mov       edi,dword ptr [esp+0xec]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x110]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x114]
    mov       dword ptr [esp+0x64],esi
    mov       esi,dword ptr [esp+0xf0]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xf4]
    mov       dword ptr [esp+0x58],esi
    mov       esi,dword ptr [esp+0x50]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x118]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x11c]
    mov       dword ptr [esp+0x68],esi
    mov       esi,dword ptr [esp+0xf8]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x54],esi
    mov       esi,dword ptr [esp+0x9c]
    not       edi
    or        esi,edi
    mov       dword ptr [esp+0x14],edi
    mov       edi,dword ptr [esp+0x64]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x6c],esi
    mov       esi,dword ptr [esp+0x74]
    or        esi,edi
    mov       edi,dword ptr [esp+0x14]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x18],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x88]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x14],esi
    mov       esi,dword ptr [esp+0x98]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x74]
    or        esi,edi
    mov       edi,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x28],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x44]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x18],esi
    mov       esi,dword ptr [esp+0x9c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    or        esi,edi
    mov       edi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x50]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x28],esi
    mov       esi,dword ptr [esp+0x6c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x2c],esi
    mov       esi,dword ptr [esp+0x28]
    and       edi,esi
    mov       esi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x74]
    xor       edi,esi
    mov       dword ptr [esp+0x2c],esi
    mov       esi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x74],edi
    mov       edi,dword ptr [esp+0x18]
    or        esi,edi
    mov       edi,dword ptr [esp+0x14]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x2c]
    mov       dword ptr [esp+0x30],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x6c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x9c]
    xor       edi,esi
    mov       dword ptr [esp+0x38],esi
    mov       esi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x9c],edi
    mov       edi,dword ptr [esp+0x18]
    and       esi,edi
    mov       edi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x1c],esi
    mov       esi,dword ptr [esp+0x28]
    or        edi,esi
    mov       esi,dword ptr [esp+0x1c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x1c],esi
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x3c],edi
    mov       edi,dword ptr [esp+0x2c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x38]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x30]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    not       esi
    xor       edi,esi
    mov       dword ptr [esp+0x98],edi
    mov       edi,dword ptr [esp+0x3c]
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x2c],esi
    mov       esi,dword ptr [esp+0x74]
    or        esi,edi
    mov       edi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    xor       edi,esi
    mov       dword ptr [esp+0x1c],esi
    mov       esi,dword ptr [esp+0x74]
    or        edi,esi
    mov       esi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x38],edi
    mov       edi,dword ptr [esp+0x1c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x178],esi
    mov       esi,dword ptr [esp+0x18]
    xor       esi,edi
    mov       dword ptr [esp+0x174],esi
    mov       esi,dword ptr [esp+0x9c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    or        esi,edi
    mov       edi,dword ptr [esp+0x28]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x2c]
    mov       dword ptr [esp+0x170],esi
    mov       esi,dword ptr [esp+0x9c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x6c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x16c],esi
    mov       esi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x168],esi
    mov       esi,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x164],esi
    mov       esi,dword ptr [esp+0x98]
    mov       dword ptr [esp+0x160],esi
    mov       esi,dword ptr [esp+0x9c]
    mov       dword ptr [esp+0x15c],esi
    mov       esi,dword ptr [esp+0x70]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x74],esi
    mov       esi,dword ptr [esp+0x7c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x30],esi
    not       esi
    mov       dword ptr [esp+0x14],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x64]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x6c],esi
    mov       esi,dword ptr [esp+0x30]
    or        esi,edi
    mov       edi,dword ptr [esp+0x14]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x18],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x44]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x14],esi
    mov       esi,dword ptr [esp+0x74]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    or        esi,edi
    mov       edi,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x28],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x18],esi
    mov       esi,dword ptr [esp+0x28]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x74]
    or        esi,edi
    mov       edi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x6c]
    and       edi,esi
    mov       dword ptr [esp+0x38],esi
    mov       esi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x28],edi
    mov       edi,dword ptr [esp+0x6c]
    or        edi,esi
    mov       esi,dword ptr [esp+0x28]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x1c],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x28],esi
    mov       esi,dword ptr [esp+0x6c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x14]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x98],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x6c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x3c],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x2c],esi
    mov       esi,dword ptr [esp+0x6c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0xb4],esi
    mov       esi,dword ptr [esp+0x18]
    and       edi,esi
    mov       esi,dword ptr [esp+0xb4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x30]
    mov       dword ptr [esp+0xb4],esi
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0xb4]
    mov       dword ptr [esp+0x30],edi
    mov       edi,dword ptr [esp+0x1c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x3c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x74]
    not       esi
    xor       edi,esi
    mov       dword ptr [esp+0x74],edi
    mov       edi,dword ptr [esp+0x30]
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x1c],esi
    mov       esi,dword ptr [esp+0x28]
    or        esi,edi
    mov       edi,dword ptr [esp+0x1c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x74]
    mov       dword ptr [esp+0xb4],esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x28]
    or        edi,esi
    mov       esi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x3c],edi
    mov       edi,dword ptr [esp+0xb4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0xf8],esi
    mov       esi,dword ptr [esp+0x18]
    xor       esi,edi
    mov       dword ptr [esp+0xf4],esi
    mov       esi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x74]
    or        esi,edi
    mov       edi,dword ptr [esp+0x38]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0xf0],esi
    mov       esi,dword ptr [esp+0x2c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x6c]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x160]
    mov       dword ptr [esp+0xec],esi
    mov       esi,dword ptr [esp+0x30]
    mov       dword ptr [esp+0xe8],esi
    mov       esi,dword ptr [esp+0x28]
    mov       dword ptr [esp+0xe4],esi
    mov       esi,dword ptr [esp+0x74]
    mov       dword ptr [esp+0xe0],esi
    mov       esi,dword ptr [esp+0x2c]
    mov       dword ptr [esp+0xdc],esi
    mov       esi,dword ptr [esp+0x8c]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x164]
    mov       dword ptr [esp+0x5c],esi
    mov       esi,dword ptr [esp+0xc8]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x168]
    mov       dword ptr [esp+0x80],esi
    mov       esi,dword ptr [esp+0xa8]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x16c]
    mov       dword ptr [esp+0x48],esi
    mov       esi,dword ptr [esp+0x40]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x170]
    mov       dword ptr [esp+0x34],esi
    mov       esi,dword ptr [esp+0x18c]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x174]
    mov       dword ptr [esp+0x4c],esi
    mov       esi,dword ptr [esp+0x190]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x178]
    mov       dword ptr [esp+0x50],esi
    mov       esi,dword ptr [esp+0x194]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x17c]
    mov       dword ptr [esp+0x44],esi
    mov       esi,dword ptr [esp+0x198]
    xor       esi,edi
    mov       edi,esi
    mov       esi,dword ptr [esp+0xac]
    mov       dword ptr [esp+0x88],edi
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x140]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x84],esi
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x144]
    mov       dword ptr [esp+0x9c],esi
    mov       esi,dword ptr [esp+0xbc]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x80]
    mov       dword ptr [esp+0x70],esi
    xor       esi,edi
    mov       dword ptr [esp+0x98],esi
    mov       esi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x148]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x60],esi
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x14c]
    mov       dword ptr [esp+0x74],esi
    mov       esi,dword ptr [esp+0x94]
    xor       esi,edi
    mov       dword ptr [esp+0x7c],esi
    mov       edi,esi
    mov       esi,dword ptr [esp+0x34]
    xor       edi,esi
    mov       dword ptr [esp+0x3c],edi
    mov       edi,dword ptr [esp+0x10c]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x150]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x154]
    mov       dword ptr [esp+0x64],esi
    mov       esi,dword ptr [esp+0x110]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x114]
    mov       dword ptr [esp+0x58],esi
    mov       esi,dword ptr [esp+0x50]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x158]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x15c]
    mov       dword ptr [esp+0x68],esi
    mov       esi,dword ptr [esp+0x118]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x54],esi
    mov       esi,dword ptr [esp+0x9c]
    not       edi
    mov       dword ptr [esp+0x10],edi
    or        esi,edi
    mov       edi,dword ptr [esp+0x64]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x6c],esi
    mov       esi,dword ptr [esp+0x74]
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x14],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x88]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0x98]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x74]
    or        esi,edi
    mov       edi,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x18],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x44]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x14],esi
    mov       esi,dword ptr [esp+0x9c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    or        esi,edi
    mov       edi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x50]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x28],esi
    and       edi,esi
    mov       esi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x18],edi
    mov       edi,dword ptr [esp+0x6c]
    or        edi,esi
    mov       esi,dword ptr [esp+0x18]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x74]
    xor       edi,esi
    mov       dword ptr [esp+0x18],esi
    mov       esi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x74],edi
    mov       edi,dword ptr [esp+0x14]
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x38],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x6c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x9c]
    xor       edi,esi
    mov       dword ptr [esp+0x1c],esi
    mov       esi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x9c],edi
    mov       edi,dword ptr [esp+0x14]
    and       esi,edi
    mov       edi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x2c],esi
    mov       esi,dword ptr [esp+0x28]
    or        edi,esi
    mov       esi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x2c],esi
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x2c]
    mov       dword ptr [esp+0x3c],edi
    mov       edi,dword ptr [esp+0x18]
    or        esi,edi
    mov       edi,dword ptr [esp+0x1c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x38]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    not       esi
    xor       edi,esi
    mov       dword ptr [esp+0x98],edi
    mov       edi,dword ptr [esp+0x3c]
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x18],esi
    mov       esi,dword ptr [esp+0x74]
    or        esi,edi
    mov       edi,dword ptr [esp+0x18]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    mov       dword ptr [esp+0x2c],esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x74]
    or        edi,esi
    mov       esi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x1c],edi
    mov       edi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x198],esi
    mov       esi,dword ptr [esp+0x14]
    xor       esi,edi
    mov       dword ptr [esp+0x194],esi
    mov       esi,dword ptr [esp+0x9c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    or        esi,edi
    mov       edi,dword ptr [esp+0x28]
    xor       esi,edi
    mov       dword ptr [esp+0x190],esi
    mov       esi,dword ptr [esp+0x9c]
    mov       edi,dword ptr [esp+0x18]
    or        esi,edi
    mov       edi,dword ptr [esp+0x6c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x18c],esi
    mov       esi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x188],esi
    mov       esi,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x184],esi
    mov       esi,dword ptr [esp+0x98]
    mov       dword ptr [esp+0x180],esi
    mov       esi,dword ptr [esp+0x9c]
    mov       dword ptr [esp+0x17c],esi
    mov       esi,dword ptr [esp+0x70]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x74],esi
    mov       esi,dword ptr [esp+0x7c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x30],esi
    not       esi
    mov       dword ptr [esp+0x10],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x64]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x6c],esi
    mov       esi,dword ptr [esp+0x30]
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x14],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x44]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0x74]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    or        esi,edi
    mov       edi,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x18],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x14],esi
    mov       esi,dword ptr [esp+0x18]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x74]
    or        esi,edi
    mov       edi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x6c]
    and       edi,esi
    mov       dword ptr [esp+0x1c],esi
    mov       esi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x18],edi
    mov       edi,dword ptr [esp+0x6c]
    or        edi,esi
    mov       esi,dword ptr [esp+0x18]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x2c],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x18],esi
    mov       esi,dword ptr [esp+0x6c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x2c]
    mov       dword ptr [esp+0x94],esi
    or        esi,edi
    xor       esi,dword ptr [esp+0x6c]
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x40],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x28],esi
    mov       esi,dword ptr [esp+0x6c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x38],esi
    mov       esi,dword ptr [esp+0x14]
    and       edi,esi
    mov       esi,dword ptr [esp+0x38]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x38],esi
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x30],edi
    mov       edi,dword ptr [esp+0x2c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x40]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x94]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x74]
    not       esi
    xor       edi,esi
    mov       dword ptr [esp+0x74],edi
    mov       edi,dword ptr [esp+0x30]
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x2c],esi
    mov       esi,dword ptr [esp+0x18]
    or        esi,edi
    mov       edi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x38],esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x18]
    or        edi,esi
    mov       esi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x40],edi
    mov       edi,dword ptr [esp+0x38]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x118],esi
    mov       esi,dword ptr [esp+0x14]
    xor       esi,edi
    mov       dword ptr [esp+0x114],esi
    mov       esi,dword ptr [esp+0x28]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x74]
    or        esi,edi
    mov       edi,dword ptr [esp+0x1c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x2c]
    mov       dword ptr [esp+0x110],esi
    mov       esi,dword ptr [esp+0x28]
    or        esi,edi
    mov       edi,dword ptr [esp+0x6c]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x1a0]
    mov       dword ptr [esp+0x10c],esi
    mov       esi,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x108],esi
    mov       esi,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x104],esi
    mov       esi,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x100],esi
    mov       esi,dword ptr [esp+0x28]
    mov       dword ptr [esp+0xfc],esi
    mov       esi,dword ptr [esp+0xb8]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x1a4]
    mov       dword ptr [esp+0x5c],esi
    mov       esi,dword ptr [esp+0xc0]
    xor       esi,edi
    mov       dword ptr [esp+0x80],esi
    mov       esi,dword ptr [esp+0xd0]
    mov       edi,dword ptr [csc_tabe+0x1a8]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x1ac]
    mov       dword ptr [esp+0x48],esi
    mov       esi,dword ptr [esp+0xd4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x1ac]
    xor       edi,dword ptr [csc_tabe+0x1b0]
    mov       dword ptr [esp+0x4c],edi
    mov       edi,dword ptr [esp+0x1b0]
    xor       edi,dword ptr [csc_tabe+0x1b4]
    mov       dword ptr [esp+0x50],edi
    mov       edi,dword ptr [esp+0x1b4]
    xor       edi,dword ptr [csc_tabe+0x1b8]
    mov       dword ptr [esp+0x44],edi
    mov       edi,dword ptr [esp+0x1b8]
    xor       edi,dword ptr [csc_tabe+0x1bc]
    xor       edx,edi
    mov       dword ptr [esp+0x88],edi
    mov       edi,dword ptr [csc_tabe+0x180]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x84],edx
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x184]
    mov       dword ptr [esp+0x6c],edx
    mov       edx,dword ptr [esp+0xcc]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x80]
    mov       dword ptr [esp+0x70],edx
    xor       edx,edi
    mov       dword ptr [esp+0xcc],edx
    mov       edx,dword ptr [esp+0xa0]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x188]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x60],edx
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x18c]
    mov       dword ptr [esp+0xa0],edx
    mov       edx,dword ptr [esp+0x20]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x12c]
    mov       dword ptr [esp+0x7c],edx
    xor       edx,esi
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x190]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x194]
    mov       dword ptr [esp+0x64],esi
    mov       esi,dword ptr [esp+0x130]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x134]
    mov       dword ptr [esp+0x58],esi
    mov       esi,dword ptr [esp+0x50]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x198]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x19c]
    mov       dword ptr [esp+0x68],esi
    mov       esi,dword ptr [esp+0x138]
    xor       esi,edi
    mov       edi,edx
    mov       dword ptr [esp+0x54],esi
    mov       esi,dword ptr [esp+0x6c]
    not       edi
    mov       dword ptr [esp+0x10],edi
    or        esi,edi
    mov       edi,dword ptr [esp+0x64]
    mov       dword ptr [esp+0x20],edx
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xa0]
    or        edi,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x14],edi
    xor       edi,edx
    mov       edx,dword ptr [esp+0x88]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xcc]
    mov       dword ptr [esp+0x10],edi
    mov       edi,dword ptr [esp+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xa0]
    or        edx,edi
    mov       edi,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x18],edx
    xor       edx,edi
    mov       edi,dword ptr [esp+0x44]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0x6c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xcc]
    or        edx,edi
    mov       edi,dword ptr [esp+0x58]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x50]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x18],edx
    mov       edx,esi
    or        edx,edi
    mov       edi,esi
    mov       dword ptr [esp+0x28],edx
    mov       edx,dword ptr [esp+0x18]
    and       edi,edx
    mov       edx,dword ptr [esp+0x28]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xa0]
    xor       edi,edx
    mov       dword ptr [esp+0x28],edx
    mov       dword ptr [esp+0xa0],edi
    mov       edi,dword ptr [esp+0x14]
    mov       edx,esi
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x38],edx
    or        edx,edi
    mov       edi,dword ptr [esp+0x6c]
    xor       edx,esi
    xor       edi,edx
    mov       dword ptr [esp+0x1c],edx
    mov       dword ptr [esp+0x6c],edi
    mov       edi,dword ptr [esp+0x14]
    mov       edx,esi
    and       edx,edi
    mov       edi,esi
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0x18]
    or        edi,edx
    mov       edx,dword ptr [esp+0x2c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x2c],edx
    not       edx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x2c]
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0x28]
    or        edx,edi
    mov       edi,dword ptr [esp+0x1c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x38]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xcc]
    not       edx
    xor       edi,edx
    mov       dword ptr [esp+0xcc],edi
    mov       edi,dword ptr [esp+0x20]
    mov       edx,edi
    not       edx
    mov       dword ptr [esp+0x28],edx
    mov       edx,dword ptr [esp+0xa0]
    or        edx,edi
    mov       edi,dword ptr [esp+0x28]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xcc]
    xor       edi,edx
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0xa0]
    or        edi,edx
    mov       edx,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x1c],edi
    mov       edi,dword ptr [esp+0x2c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x1b8],edx
    mov       edx,dword ptr [esp+0x14]
    xor       edx,edi
    mov       dword ptr [esp+0x1b4],edx
    mov       edx,dword ptr [esp+0x6c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xcc]
    or        edx,edi
    mov       edi,dword ptr [esp+0x18]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x1b0],edx
    mov       edx,dword ptr [esp+0x6c]
    or        edx,edi
    mov       edi,dword ptr [esp+0x7c]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x1ac],edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x1a8],edx
    mov       edx,dword ptr [esp+0xa0]
    mov       dword ptr [esp+0x1a4],edx
    mov       edx,dword ptr [esp+0xcc]
    mov       dword ptr [esp+0x1a0],edx
    mov       edx,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x19c],edx
    mov       edx,dword ptr [esp+0x70]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0xa0],edx
    mov       edx,dword ptr [esp+0x48]
    xor       edi,edx
    mov       edx,edi
    mov       dword ptr [esp+0x20],edi
    not       edx
    mov       dword ptr [esp+0x10],edx
    or        edx,esi
    mov       esi,dword ptr [esp+0x64]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x60]
    or        edi,esi
    mov       esi,dword ptr [esp+0x10]
    xor       edi,esi
    mov       esi,edi
    xor       esi,dword ptr [esp+0x54]
    xor       esi,dword ptr [esp+0x44]
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0xa0]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    or        esi,edi
    mov       edi,esi
    xor       edi,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x14],edi
    mov       edi,dword ptr [esp+0x84]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xa0]
    or        esi,edi
    mov       edi,dword ptr [esp+0x58]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x18],esi
    mov       esi,edx
    or        esi,edi
    mov       edi,edx
    mov       dword ptr [esp+0x28],esi
    mov       esi,dword ptr [esp+0x18]
    and       edi,esi
    mov       esi,dword ptr [esp+0x28]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x1c],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x28],esi
    mov       esi,edx
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x40],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x84]
    xor       esi,edx
    mov       dword ptr [esp+0x30],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x2c],esi
    mov       esi,edx
    and       esi,edi
    mov       edi,edx
    mov       dword ptr [esp+0x38],esi
    mov       esi,dword ptr [esp+0x18]
    or        edi,esi
    mov       esi,dword ptr [esp+0x38]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x38],esi
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0x1c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x30]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x40]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xa0]
    not       esi
    xor       edi,esi
    mov       dword ptr [esp+0xa0],edi
    mov       edi,dword ptr [esp+0x20]
    mov       esi,edi
    not       esi
    mov       dword ptr [esp+0x1c],esi
    mov       esi,dword ptr [esp+0x28]
    or        esi,edi
    mov       edi,dword ptr [esp+0x1c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xa0]
    mov       dword ptr [esp+0x38],esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x28]
    or        edi,esi
    mov       esi,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x30],edi
    mov       edi,dword ptr [esp+0x38]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x138],esi
    mov       esi,dword ptr [esp+0x14]
    xor       esi,edi
    mov       dword ptr [esp+0x134],esi
    mov       esi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xa0]
    or        esi,edi
    mov       edi,dword ptr [esp+0x18]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x130],esi
    mov       esi,dword ptr [esp+0x2c]
    or        esi,edi
    xor       esi,edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x12c],esi
    mov       dword ptr [esp+0x128],edx
    mov       edx,dword ptr [esp+0x28]
    mov       esi,dword ptr [csc_tabe+0x1e0]
    mov       dword ptr [esp+0x124],edx
    mov       edx,dword ptr [esp+0xa0]
    mov       dword ptr [esp+0x120],edx
    mov       edx,dword ptr [esp+0x2c]
    mov       edi,dword ptr [csc_tabe+0x1e4]
    mov       dword ptr [esp+0x11c],edx
    mov       edx,dword ptr [esp+0xa4]
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x1e8]
    mov       dword ptr [esp+0x5c],edx
    mov       edx,dword ptr [esp+0xb0]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x1ec]
    mov       dword ptr [esp+0x80],edx
    mov       edx,dword ptr [esp+0x90]
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x1f0]
    mov       dword ptr [esp+0x48],edx
    mov       edx,dword ptr [esp+0xd8]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x1cc]
    xor       edi,esi
    mov       esi,dword ptr [csc_tabe+0x1f4]
    mov       dword ptr [esp+0x4c],edi
    mov       edi,dword ptr [esp+0x1d0]
    xor       edi,esi
    mov       esi,edi
    mov       edi,dword ptr [esp+0x1d4]
    xor       edi,dword ptr [csc_tabe+0x1f8]
    mov       dword ptr [esp+0x50],esi
    mov       dword ptr [esp+0x44],edi
    mov       edi,dword ptr [esp+0x1d8]
    xor       edi,dword ptr [csc_tabe+0x1fc]
    mov       dword ptr [esp+0x88],edi
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x1c0]
    xor       eax,edi
    mov       edi,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x84],eax
    xor       eax,edi
    mov       edi,dword ptr [esp+0x24]
    xor       edi,dword ptr [csc_tabe+0x1c4]
    mov       dword ptr [esp+0x70],edi
    xor       edi,dword ptr [esp+0x80]
    mov       dword ptr [esp+0x3c],edi
    mov       edi,dword ptr [esp+0x80]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1c8]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x60],ecx
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1cc]
    mov       dword ptr [esp+0x90],ecx
    mov       ecx,dword ptr [esp+0x78]
    xor       ecx,edi
    mov       dword ptr [esp+0x7c],ecx
    mov       edi,ecx
    mov       ecx,dword ptr [esp+0x14c]
    xor       edi,edx
    xor       edx,ecx
    mov       ecx,dword ptr [csc_tabe+0x1d0]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x150]
    mov       dword ptr [esp+0x64],edx
    mov       edx,dword ptr [csc_tabe+0x1d4]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x1d8]
    mov       dword ptr [esp+0x58],ecx
    mov       ecx,esi
    mov       esi,dword ptr [esp+0x154]
    mov       dword ptr [esp+0x18],edi
    xor       ecx,esi
    mov       esi,dword ptr [csc_tabe+0x1dc]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x158]
    xor       edx,esi
    mov       dword ptr [esp+0x68],ecx
    mov       dword ptr [esp+0x54],edx
    mov       esi,edi
    mov       edx,eax
    not       esi
    mov       dword ptr [esp+0x10],esi
    or        edx,esi
    mov       esi,dword ptr [esp+0x64]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x4c]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x90]
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x14],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x88]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0x3c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x90]
    or        esi,edi
    mov       edi,esi
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x44]
    xor       edi,ecx
    mov       ecx,eax
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x14],edi
    mov       edi,dword ptr [esp+0x58]
    or        ecx,esi
    mov       esi,dword ptr [esp+0x50]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       ecx,esi
    mov       esi,edx
    or        esi,edi
    mov       edi,edx
    and       edi,ecx
    mov       dword ptr [esp+0x1c],ecx
    xor       esi,edi
    mov       edi,dword ptr [esp+0x90]
    xor       edi,esi
    mov       dword ptr [esp+0x28],esi
    mov       dword ptr [esp+0x90],edi
    mov       edi,dword ptr [esp+0x14]
    mov       esi,edx
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x20],esi
    or        esi,edi
    mov       edi,dword ptr [esp+0x14]
    xor       esi,edx
    mov       dword ptr [esp+0x2c],esi
    xor       eax,esi
    mov       esi,edx
    and       esi,edi
    mov       edi,edx
    or        edi,ecx
    xor       esi,edi
    mov       edi,dword ptr [esp+0x18]
    mov       ecx,esi
    not       ecx
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x28]
    or        esi,ecx
    mov       ecx,dword ptr [esp+0x2c]
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x20]
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x3c]
    not       esi
    xor       ecx,esi
    mov       dword ptr [esp+0x18],edi
    mov       dword ptr [esp+0x3c],ecx
    mov       esi,edi
    mov       ecx,dword ptr [esp+0x90]
    mov       dword ptr [esp+0x1bc],eax
    not       esi
    or        ecx,edi
    mov       edi,dword ptr [esp+0x90]
    xor       ecx,esi
    mov       dword ptr [esp+0x28],esi
    mov       esi,dword ptr [esp+0x3c]
    xor       esi,ecx
    or        esi,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x14]
    xor       ecx,esi
    mov       dword ptr [esp+0x1d8],edi
    mov       edi,eax
    mov       dword ptr [esp+0x1d4],ecx
    mov       ecx,dword ptr [esp+0x3c]
    xor       edi,esi
    mov       esi,dword ptr [esp+0x1c]
    or        edi,ecx
    xor       edi,esi
    mov       esi,eax
    mov       eax,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x1d0],edi
    mov       edi,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x1c0],ecx
    or        esi,edi
    mov       edi,dword ptr [esp+0x70]
    xor       esi,edx
    mov       edx,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x1cc],esi
    mov       esi,dword ptr [esp+0x48]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x7c]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x1c8],edx
    mov       edx,dword ptr [esp+0x90]
    mov       ecx,eax
    mov       dword ptr [esp+0x1c4],edx
    not       eax
    mov       edx,eax
    mov       dword ptr [esp+0x1c],ecx
    or        edx,esi
    mov       esi,dword ptr [esp+0x64]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x60]
    or        ecx,esi
    mov       esi,dword ptr [esp+0x54]
    xor       ecx,eax
    mov       dword ptr [esp+0x18],edx
    mov       eax,ecx
    xor       eax,esi
    mov       esi,dword ptr [esp+0x44]
    xor       eax,esi
    mov       esi,edi
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x60]
    or        esi,ecx
    mov       dword ptr [esp+0x10],eax
    mov       ecx,esi
    xor       ecx,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x20],ecx
    mov       ecx,dword ptr [esp+0x84]
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x58]
    or        esi,edi
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x4c]
    xor       esi,ecx
    mov       ecx,edx
    mov       dword ptr [esp+0x14],esi
    and       ecx,esi
    mov       esi,edx
    or        esi,eax
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x60]
    mov       eax,ecx
    xor       eax,esi
    mov       dword ptr [esp+0x28],eax
    mov       esi,dword ptr [esp+0x20]
    mov       eax,edx
    or        eax,esi
    mov       esi,dword ptr [esp+0x10]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x30],eax
    or        eax,ecx
    xor       eax,edx
    mov       dword ptr [esp+0x38],eax
    xor       eax,esi
    mov       esi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x2c],eax
    mov       eax,edx
    or        eax,esi
    mov       esi,dword ptr [esp+0x20]
    and       edx,esi
    mov       esi,dword ptr [esp+0x1c]
    xor       eax,edx
    mov       edx,eax
    or        eax,ecx
    mov       ecx,dword ptr [esp+0x38]
    not       edx
    xor       esi,edx
    mov       edx,dword ptr [esp+0x30]
    xor       eax,ecx
    lea       ecx,[esp+0x14]
    xor       eax,edx
    lea       edx,[esp+0x20]
    not       eax
    xor       edi,eax
    lea       eax,[esp+0x18]
    push      eax
    push      ecx
    mov       ecx,dword ptr [esp+0x34]
    lea       eax,[esp+0x18]
    push      edx
    mov       edx,dword ptr [esp+0x34]
    push      eax
    push      ecx
    push      edi
    push      edx
    push      esi
    call      _csc_transF
    mov       eax,dword ptr [esp+0x30]
    mov       ecx,dword ptr [esp+0x40]
    mov       edx,dword ptr [esp+0x34]
    mov       dword ptr [esp+0x178],eax
    mov       eax,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x168],esi
    mov       dword ptr [esp+0x16c],eax
    mov       eax,dword ptr [esp+0x200]
    mov       dword ptr [esp+0x174],ecx
    mov       ecx,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x170],edx
    mov       edx,dword ptr [esp+0x4c]
    lea       esi,[eax+0x300]
    mov       eax,dword ptr [esp+0x204]
    mov       dword ptr [esp+0x160],edi
    add       esp,0x00000020
    mov       dword ptr [esp+0x144],ecx
    mov       dword ptr [esp+0x13c],edx
    mov       dword ptr [esp+0x78],eax
    mov       edi,csc_tabc+0x100
    mov       dword ptr [esp+0x8c],0x00000007
    mov       dword ptr [esp+0x18],0x00000008
    jmp       X$11
X$9:
    mov       edi,dword ptr [esp+0x28]
    mov       eax,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x18],0x00000008
    jmp       X$11
X$10:
    mov       eax,dword ptr [esp+0x78]
X$11:
    mov       ecx,dword ptr [eax]
    mov       edx,dword ptr [edi]
    xor       ecx,edx
    mov       edx,dword ptr [edi+0x4]
    xor       edx,dword ptr [eax+0x4]
    mov       dword ptr [esp+0x40],ecx
    mov       dword ptr [esp+0x30],edx
    mov       edx,dword ptr [edi+0x8]
    xor       edx,dword ptr [eax+0x8]
    mov       dword ptr [esp+0x3c],edx
    mov       edx,dword ptr [edi+0xc]
    xor       edx,dword ptr [eax+0xc]
    mov       dword ptr [esp+0x94],edx
    not       edx
    or        ecx,edx
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [edi+0x10]
    xor       ecx,edx
    mov       edx,dword ptr [eax+0x10]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x10],ecx
    mov       ecx,dword ptr [esp+0x94]
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x14]
    xor       edx,ecx
    mov       ecx,dword ptr [edi+0x1c]
    xor       ecx,dword ptr [eax+0x1c]
    xor       ecx,edx
    mov       dword ptr [esp+0x20],ecx
    mov       ecx,dword ptr [esp+0x30]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x3c]
    or        ecx,edx
    mov       edx,dword ptr [eax+0x18]
    mov       dword ptr [esp+0x14],ecx
    mov       ecx,dword ptr [edi+0x18]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x14]
    xor       ecx,edx
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0x40]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    or        ecx,edx
    mov       edx,dword ptr [edi+0x14]
    xor       ecx,edx
    mov       edx,dword ptr [eax+0x14]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x38],ecx
    mov       ecx,dword ptr [esp+0x10]
    mov       eax,ecx
    or        eax,edx
    mov       edx,ecx
    mov       dword ptr [esp+0x14],eax
    mov       eax,dword ptr [esp+0x38]
    and       edx,eax
    mov       eax,dword ptr [esp+0x14]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x3c]
    xor       edx,eax
    mov       dword ptr [esp+0x14],eax
    mov       dword ptr [esp+0x3c],edx
    mov       edx,dword ptr [esp+0x1c]
    mov       eax,ecx
    or        eax,edx
    mov       edx,dword ptr [esp+0x20]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x2c],eax
    or        eax,edx
    mov       edx,dword ptr [esp+0x40]
    xor       eax,ecx
    xor       edx,eax
    mov       dword ptr [esp+0x28],eax
    mov       dword ptr [esp+0x40],edx
    mov       edx,dword ptr [esp+0x1c]
    mov       eax,ecx
    and       eax,edx
    mov       edx,dword ptr [esp+0x38]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x94]
    xor       eax,ecx
    mov       ecx,eax
    not       ecx
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x14]
    or        eax,ecx
    mov       ecx,dword ptr [esp+0x28]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x2c]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x30]
    not       eax
    xor       ecx,eax
    lea       eax,[esp+0x10]
    push      eax
    lea       eax,[esp+0x3c]
    push      eax
    lea       eax,[esp+0x24]
    push      eax
    lea       eax,[esp+0x2c]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       dword ptr [esp+0x44],ecx
    push      ecx
    mov       ecx,dword ptr [esp+0x54]
    push      ecx
    push      edx
    mov       dword ptr [esp+0xb4],edx
    call      _csc_transF
    mov       edx,dword ptr [esp+0x40]
    mov       eax,dword ptr [esp+0x3c]
    mov       ecx,dword ptr [esp+0x58]
    mov       dword ptr [esi+0xe0],edx
    mov       edx,dword ptr [esp+0x30]
    mov       dword ptr [esi+0xc0],eax
    mov       eax,dword ptr [esp+0xb4]
    mov       dword ptr [esi+0x80],edx
    mov       edx,dword ptr [esp+0x50]
    mov       dword ptr [esi+0x60],eax
    mov       eax,dword ptr [esp+0x60]
    mov       dword ptr [esi+0xa0],ecx
    mov       ecx,dword ptr [esp+0x5c]
    mov       dword ptr [esi+0x20],edx
    mov       edx,dword ptr [esp+0x98]
    mov       dword ptr [esi],eax
    mov       eax,dword ptr [esp+0x38]
    mov       dword ptr [esi+0x40],ecx
    add       esp,0x00000020
    add       edi,0x00000020
    add       edx,0x00000020
    add       esi,0x00000004
    dec       eax
    mov       dword ptr [esp+0x78],edx
    mov       dword ptr [esp+0x18],eax
    ljne       X$10
    mov       ecx,dword ptr [esi-0x200]
    mov       edx,dword ptr [esi]
    sub       esi,0x00000020
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0xfc]
    mov       dword ptr [esp+0x28],edi
    mov       edi,dword ptr [esi+0x24]
    mov       dword ptr [esi+0x20],edx
    mov       eax,edx
    mov       edx,dword ptr [esi-0x1dc]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x100]
    xor       edi,edx
    mov       dword ptr [esp+0x5c],eax
    mov       eax,edi
    mov       edx,dword ptr [esp+0x104]
    xor       eax,ecx
    mov       dword ptr [esi+0x24],edi
    mov       edi,dword ptr [esi+0x28]
    mov       ecx,eax
    mov       eax,dword ptr [esi-0x1d8]
    xor       edi,eax
    mov       eax,edi
    mov       dword ptr [esi+0x28],edi
    mov       edi,dword ptr [esi+0x2c]
    xor       eax,edx
    mov       edx,dword ptr [esi-0x1d4]
    mov       dword ptr [esp+0x48],eax
    xor       edi,edx
    mov       edx,dword ptr [esp+0x108]
    mov       eax,edi
    mov       dword ptr [esi+0x2c],edi
    mov       edi,dword ptr [esi+0x30]
    xor       eax,edx
    mov       edx,eax
    mov       eax,dword ptr [esi-0x1d0]
    xor       edi,eax
    mov       dword ptr [esi+0x30],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x10c]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x34]
    mov       dword ptr [esp+0x4c],eax
    mov       eax,dword ptr [esi-0x1cc]
    xor       edi,eax
    mov       dword ptr [esi+0x34],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x110]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x38]
    mov       dword ptr [esp+0x50],eax
    mov       eax,dword ptr [esi-0x1c8]
    xor       edi,eax
    mov       dword ptr [esi+0x38],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x114]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x3c]
    mov       dword ptr [esp+0x44],eax
    mov       eax,dword ptr [esi-0x1c4]
    xor       edi,eax
    mov       dword ptr [esi+0x3c],edi
    mov       eax,edi
    xor       eax,dword ptr [esp+0x118]
    mov       edi,eax
    mov       eax,dword ptr [esi-0x200]
    xor       dword ptr [esi],eax
    mov       eax,dword ptr [esi]
    mov       dword ptr [esp+0x88],edi
    xor       eax,edi
    xor       eax,dword ptr [esp+0xdc]
    mov       dword ptr [esp+0x84],eax
    mov       edi,eax
    mov       eax,dword ptr [esp+0x5c]
    xor       edi,eax
    mov       eax,dword ptr [esi-0x1fc]
    xor       dword ptr [esi+0x4],eax
    mov       eax,dword ptr [esi+0x4]
    xor       eax,dword ptr [esp+0xe0]
    mov       dword ptr [esp+0x70],eax
    xor       eax,ecx
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esi-0x1f8]
    xor       dword ptr [esi+0x8],eax
    mov       eax,dword ptr [esi+0x8]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xe4]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x60],eax
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1f4]
    mov       dword ptr [esp+0x3c],eax
    mov       eax,dword ptr [esi+0xc]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xe8]
    mov       dword ptr [esi+0xc],eax
    xor       eax,ecx
    mov       dword ptr [esp+0x7c],eax
    mov       ecx,eax
    mov       eax,dword ptr [esi-0x1f0]
    xor       ecx,edx
    xor       dword ptr [esi+0x10],eax
    mov       eax,dword ptr [esi+0x10]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xec]
    xor       eax,edx
    mov       edx,dword ptr [esi-0x1ec]
    mov       dword ptr [esp+0x64],eax
    mov       eax,dword ptr [esi+0x14]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xf0]
    mov       dword ptr [esi+0x14],eax
    xor       eax,edx
    mov       edx,dword ptr [esi+0x18]
    mov       dword ptr [esp+0x58],eax
    mov       eax,dword ptr [esi-0x1e8]
    mov       dword ptr [esp+0x14],ecx
    xor       edx,eax
    mov       dword ptr [esi+0x18],edx
    mov       eax,edx
    mov       edx,dword ptr [esp+0x50]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xf4]
    xor       eax,edx
    mov       edx,dword ptr [esi-0x1e4]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esi+0x1c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xf8]
    mov       dword ptr [esi+0x1c],eax
    xor       eax,edx
    mov       edx,ecx
    mov       dword ptr [esp+0x54],eax
    not       edx
    mov       eax,edi
    mov       dword ptr [esp+0x10],edx
    or        eax,edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x4c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x3c]
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x10]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x10],edx
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    mov       dword ptr [esp+0x2c],eax
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x20],edx
    mov       edx,dword ptr [esp+0x10]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x3c]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x10],ecx
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x44]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x58]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x50]
    xor       ecx,edx
    mov       edx,eax
    and       edx,ecx
    mov       dword ptr [esp+0x10],ecx
    mov       ecx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x18],edx
    mov       edx,eax
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x18]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x3c]
    xor       edx,ecx
    mov       dword ptr [esp+0x18],ecx
    mov       dword ptr [esp+0x3c],edx
    mov       edx,dword ptr [esp+0x1c]
    mov       ecx,eax
    or        ecx,edx
    mov       edx,dword ptr [esp+0x20]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x40],ecx
    or        ecx,edx
    mov       edx,dword ptr [esp+0x10]
    xor       ecx,eax
    mov       dword ptr [esp+0x38],ecx
    xor       edi,ecx
    mov       ecx,eax
    or        ecx,edx
    mov       edx,dword ptr [esp+0x1c]
    and       eax,edx
    mov       edx,dword ptr [esp+0x14]
    xor       ecx,eax
    mov       eax,ecx
    not       eax
    xor       edx,eax
    mov       eax,dword ptr [esp+0x18]
    or        ecx,eax
    mov       eax,dword ptr [esp+0x38]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x40]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x30]
    not       ecx
    xor       eax,ecx
    lea       ecx,[esp+0x2c]
    push      ecx
    lea       ecx,[esp+0x14]
    push      ecx
    lea       ecx,[esp+0x24]
    push      ecx
    lea       ecx,[esp+0x2c]
    push      ecx
    push      edi
    mov       dword ptr [esp+0x44],eax
    push      eax
    mov       eax,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x2c],edx
    push      eax
    push      edx
    call      _csc_transF
    mov       ecx,dword ptr [esp+0x40]
    mov       edx,dword ptr [esp+0x3c]
    mov       eax,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x138],ecx
    mov       ecx,dword ptr [esp+0x4c]
    add       esp,0x00000020
    mov       dword ptr [esp+0x114],edx
    mov       dword ptr [esp+0x110],eax
    mov       dword ptr [esp+0x10c],ecx
    mov       eax,dword ptr [esp+0x3c]
    mov       edx,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x104],eax
    mov       eax,dword ptr [esp+0x70]
    mov       ecx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0xfc],edi
    mov       edi,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x108],edx
    mov       edx,dword ptr [esp+0x48]
    xor       eax,edi
    mov       edi,dword ptr [esp+0x7c]
    mov       dword ptr [esp+0x100],ecx
    xor       edi,edx
    mov       ecx,eax
    mov       dword ptr [esp+0x18],edi
    mov       edx,edi
    or        edi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x2c],ecx
    not       edx
    xor       edi,edx
    mov       eax,edx
    mov       edx,edi
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x60]
    or        eax,dword ptr [esp+0x84]
    or        ecx,edi
    xor       edx,dword ptr [esp+0x54]
    mov       edi,ecx
    xor       eax,dword ptr [esp+0x64]
    xor       edi,dword ptr [esp+0x68]
    xor       edx,dword ptr [esp+0x44]
    mov       dword ptr [esp+0x1c],eax
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0x84]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x2c]
    or        ecx,edi
    mov       edi,dword ptr [esp+0x58]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       ecx,edi
    mov       dword ptr [esp+0x10],edx
    mov       edi,ecx
    mov       ecx,eax
    mov       dword ptr [esp+0x14],edi
    and       ecx,edi
    mov       edi,eax
    or        edi,edx
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x60]
    mov       edx,ecx
    xor       edx,edi
    mov       edi,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x38],edx
    mov       edx,eax
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x94],edx
    or        edx,ecx
    xor       edx,eax
    mov       dword ptr [esp+0x40],edx
    xor       edx,edi
    mov       edi,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x30],edx
    mov       edx,eax
    and       edx,edi
    mov       edi,dword ptr [esp+0x14]
    or        eax,edi
    xor       edx,eax
    mov       eax,dword ptr [esp+0x18]
    mov       edi,edx
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x94]
    not       edi
    xor       eax,edi
    mov       edi,dword ptr [esp+0x40]
    xor       edx,edi
    mov       dword ptr [esp+0x18],eax
    xor       edx,ecx
    mov       edi,dword ptr [esp+0x2c]
    lea       ecx,[esp+0x14]
    not       edx
    xor       edi,edx
    lea       edx,[esp+0x1c]
    push      edx
    lea       edx,[esp+0x24]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x3c]
    lea       ecx,[esp+0x1c]
    push      ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    push      edi
    push      ecx
    push      eax
    call      _csc_transF
    mov       eax,dword ptr [esp+0x40]
    mov       ecx,dword ptr [esp+0x34]
    mov       edx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x114],eax
    mov       eax,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x110],ecx
    mov       ecx,dword ptr [esp+0x58]
    add       esi,0x00000040
    mov       dword ptr [esp+0x118],edx
    mov       edx,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x108],eax
    mov       eax,dword ptr [esi-0x1e0]
    mov       dword ptr [esp+0x104],ecx
    mov       ecx,dword ptr [esi+0x20]
    mov       dword ptr [esp+0x10c],edx
    mov       edx,dword ptr [esp+0x50]
    xor       ecx,eax
    mov       dword ptr [esp+0x100],edi
    mov       edi,dword ptr [esp+0x15c]
    mov       dword ptr [esp+0xfc],edx
    mov       edx,dword ptr [esi+0x24]
    mov       dword ptr [esi+0x20],ecx
    mov       eax,ecx
    mov       ecx,dword ptr [esi-0x1dc]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x28]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x160]
    mov       dword ptr [esp+0x7c],eax
    mov       dword ptr [esi+0x24],edx
    mov       eax,edx
    mov       edx,dword ptr [esi-0x1d8]
    xor       eax,ecx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x164]
    mov       ecx,eax
    mov       eax,edi
    mov       dword ptr [esi+0x28],edi
    mov       edi,dword ptr [esi+0x2c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x168]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esi-0x1d4]
    add       esp,0x00000020
    xor       edi,eax
    mov       eax,edi
    mov       dword ptr [esi+0x2c],edi
    mov       edi,dword ptr [esi+0x30]
    xor       eax,edx
    mov       edx,eax
    mov       eax,dword ptr [esi-0x1d0]
    xor       edi,eax
    mov       dword ptr [esi+0x30],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x14c]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x34]
    mov       dword ptr [esp+0x4c],eax
    mov       eax,dword ptr [esi-0x1cc]
    xor       edi,eax
    mov       dword ptr [esi+0x34],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x150]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x38]
    mov       dword ptr [esp+0x50],eax
    mov       eax,dword ptr [esi-0x1c8]
    xor       edi,eax
    mov       dword ptr [esi+0x38],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x154]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x3c]
    mov       dword ptr [esp+0x44],eax
    mov       eax,dword ptr [esi-0x1c4]
    xor       edi,eax
    mov       dword ptr [esi+0x3c],edi
    mov       eax,edi
    xor       eax,dword ptr [esp+0x158]
    mov       edi,eax
    mov       eax,dword ptr [esi-0x200]
    xor       dword ptr [esi],eax
    mov       eax,dword ptr [esi]
    mov       dword ptr [esp+0x88],edi
    xor       eax,edi
    xor       eax,dword ptr [esp+0x11c]
    mov       dword ptr [esp+0x84],eax
    mov       edi,eax
    mov       eax,dword ptr [esp+0x5c]
    xor       edi,eax
    mov       eax,dword ptr [esi-0x1fc]
    xor       dword ptr [esi+0x4],eax
    mov       eax,dword ptr [esi+0x4]
    xor       eax,dword ptr [esp+0x120]
    mov       dword ptr [esp+0x70],eax
    xor       eax,ecx
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esi-0x1f8]
    xor       dword ptr [esi+0x8],eax
    mov       eax,dword ptr [esi+0x8]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x124]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x60],eax
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1f4]
    mov       dword ptr [esp+0x3c],eax
    mov       eax,dword ptr [esi+0xc]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x128]
    mov       dword ptr [esi+0xc],eax
    xor       eax,ecx
    mov       dword ptr [esp+0x7c],eax
    mov       ecx,eax
    mov       eax,dword ptr [esi-0x1f0]
    xor       ecx,edx
    xor       dword ptr [esi+0x10],eax
    mov       eax,dword ptr [esi+0x10]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x12c]
    xor       eax,edx
    mov       edx,dword ptr [esi-0x1ec]
    mov       dword ptr [esp+0x64],eax
    mov       eax,dword ptr [esi+0x14]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x130]
    mov       dword ptr [esi+0x14],eax
    xor       eax,edx
    mov       edx,dword ptr [esi+0x18]
    mov       dword ptr [esp+0x58],eax
    mov       eax,dword ptr [esi-0x1e8]
    mov       dword ptr [esp+0x10],ecx
    xor       edx,eax
    mov       dword ptr [esi+0x18],edx
    mov       eax,edx
    mov       edx,dword ptr [esp+0x50]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x134]
    xor       eax,edx
    mov       edx,dword ptr [esi-0x1e4]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esi+0x1c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x138]
    mov       dword ptr [esi+0x1c],eax
    xor       eax,edx
    mov       edx,ecx
    mov       dword ptr [esp+0x54],eax
    not       edx
    mov       eax,edi
    mov       dword ptr [esp+0x14],edx
    or        eax,edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x4c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x3c]
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x14]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x14],edx
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    mov       dword ptr [esp+0x18],eax
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x20],edx
    mov       edx,dword ptr [esp+0x14]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x3c]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x14],ecx
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x44]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x58]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x50]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x38],ecx
    mov       ecx,eax
    or        ecx,edx
    mov       edx,eax
    mov       dword ptr [esp+0x14],ecx
    mov       ecx,dword ptr [esp+0x38]
    and       edx,ecx
    mov       ecx,dword ptr [esp+0x14]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x3c]
    xor       edx,ecx
    mov       dword ptr [esp+0x14],ecx
    mov       dword ptr [esp+0x3c],edx
    mov       edx,dword ptr [esp+0x1c]
    mov       ecx,eax
    or        ecx,edx
    mov       edx,dword ptr [esp+0x20]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x40],ecx
    or        ecx,edx
    mov       edx,dword ptr [esp+0x1c]
    xor       ecx,eax
    mov       dword ptr [esp+0x2c],ecx
    xor       edi,ecx
    mov       ecx,eax
    and       ecx,edx
    mov       edx,dword ptr [esp+0x38]
    or        eax,edx
    mov       edx,dword ptr [esp+0x10]
    xor       ecx,eax
    mov       eax,ecx
    not       eax
    xor       edx,eax
    mov       eax,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x10],edx
    or        ecx,eax
    mov       eax,dword ptr [esp+0x2c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x40]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x30]
    not       ecx
    xor       eax,ecx
    lea       ecx,[esp+0x18]
    push      ecx
    lea       ecx,[esp+0x3c]
    push      ecx
    lea       ecx,[esp+0x24]
    push      ecx
    lea       ecx,[esp+0x2c]
    push      ecx
    push      edi
    mov       dword ptr [esp+0x44],eax
    push      eax
    mov       eax,dword ptr [esp+0x54]
    push      eax
    push      edx
    call      _csc_transF
    mov       eax,dword ptr [esp+0x58]
    mov       edx,dword ptr [esp+0x3c]
    mov       ecx,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x170],eax
    mov       eax,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x174],edx
    mov       edx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x164],eax
    mov       eax,dword ptr [esp+0x90]
    mov       dword ptr [esp+0x15c],edi
    mov       edi,dword ptr [esp+0x7c]
    mov       dword ptr [esp+0x168],edx
    mov       edx,dword ptr [esp+0x68]
    xor       eax,edi
    mov       edi,dword ptr [esp+0x9c]
    mov       dword ptr [esp+0x178],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       edi,edx
    mov       dword ptr [esp+0x16c],ecx
    mov       ecx,dword ptr [esp+0x50]
    mov       dword ptr [esp+0x38],edi
    mov       edx,edi
    or        edi,dword ptr [esp+0x80]
    mov       dword ptr [esp+0x160],ecx
    not       edx
    mov       ecx,eax
    xor       edi,edx
    mov       dword ptr [esp+0x4c],ecx
    mov       eax,edx
    mov       edx,edi
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x80]
    or        eax,dword ptr [esp+0xa4]
    or        ecx,edi
    xor       edx,dword ptr [esp+0x74]
    mov       edi,ecx
    xor       eax,dword ptr [esp+0x84]
    xor       edi,dword ptr [esp+0x88]
    xor       edx,dword ptr [esp+0x64]
    add       esp,0x00000020
    mov       dword ptr [esp+0x1c],eax
    mov       dword ptr [esp+0x20],edi
    mov       edi,dword ptr [esp+0x84]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x2c]
    or        ecx,edi
    mov       edi,dword ptr [esp+0x58]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x4c]
    xor       ecx,edi
    mov       dword ptr [esp+0x10],edx
    mov       edi,ecx
    mov       ecx,eax
    or        ecx,edx
    mov       dword ptr [esp+0x14],edi
    mov       edx,eax
    and       edx,edi
    mov       edi,dword ptr [esp+0x60]
    xor       ecx,edx
    mov       edx,ecx
    xor       edx,edi
    mov       edi,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x38],edx
    mov       edx,eax
    or        edx,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x94],edx
    or        edx,ecx
    xor       edx,eax
    mov       dword ptr [esp+0x40],edx
    xor       edx,edi
    mov       edi,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x30],edx
    mov       edx,eax
    and       edx,edi
    mov       edi,dword ptr [esp+0x14]
    or        eax,edi
    xor       edx,eax
    mov       eax,dword ptr [esp+0x18]
    mov       edi,edx
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x94]
    not       edi
    xor       eax,edi
    mov       edi,dword ptr [esp+0x40]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x2c]
    xor       edx,ecx
    lea       ecx,[esp+0x1c]
    not       edx
    xor       edi,edx
    push      ecx
    lea       edx,[esp+0x18]
    lea       ecx,[esp+0x24]
    push      edx
    push      ecx
    mov       ecx,dword ptr [esp+0x3c]
    lea       edx,[esp+0x1c]
    push      edx
    mov       edx,dword ptr [esp+0x48]
    push      ecx
    push      edi
    push      edx
    push      eax
    mov       dword ptr [esp+0x38],eax
    call      _csc_transF
    mov       ecx,dword ptr [esp+0x40]
    mov       edx,dword ptr [esp+0x34]
    mov       eax,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x154],ecx
    mov       ecx,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x150],edx
    mov       edx,dword ptr [esp+0x58]
    mov       dword ptr [esp+0x158],eax
    mov       eax,dword ptr [esp+0x3c]
    add       esi,0x00000040
    mov       dword ptr [esp+0x148],ecx
    mov       dword ptr [esp+0x144],edx
    mov       ecx,dword ptr [esi-0x1e0]
    mov       edx,dword ptr [esi+0x20]
    mov       dword ptr [esp+0x14c],eax
    mov       eax,dword ptr [esp+0x50]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x19c]
    mov       dword ptr [esp+0x13c],eax
    mov       dword ptr [esp+0x140],edi
    mov       edi,dword ptr [esi+0x24]
    mov       dword ptr [esi+0x20],edx
    mov       eax,edx
    mov       edx,dword ptr [esi-0x1dc]
    add       esp,0x00000020
    xor       eax,ecx
    xor       edi,edx
    mov       dword ptr [esp+0x5c],eax
    mov       dword ptr [esi+0x24],edi
    mov       ecx,dword ptr [esp+0x180]
    mov       eax,edi
    mov       edi,dword ptr [esi+0x28]
    mov       edx,dword ptr [esp+0x184]
    xor       eax,ecx
    mov       ecx,eax
    mov       eax,dword ptr [esi-0x1d8]
    xor       edi,eax
    mov       eax,edi
    mov       dword ptr [esi+0x28],edi
    mov       edi,dword ptr [esi+0x2c]
    xor       eax,edx
    mov       edx,dword ptr [esi-0x1d4]
    mov       dword ptr [esp+0x48],eax
    xor       edi,edx
    mov       edx,dword ptr [esp+0x188]
    mov       eax,edi
    mov       dword ptr [esi+0x2c],edi
    mov       edi,dword ptr [esi+0x30]
    xor       eax,edx
    mov       edx,eax
    mov       eax,dword ptr [esi-0x1d0]
    xor       edi,eax
    mov       dword ptr [esi+0x30],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x18c]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x34]
    mov       dword ptr [esp+0x4c],eax
    mov       eax,dword ptr [esi-0x1cc]
    xor       edi,eax
    mov       dword ptr [esi+0x34],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x190]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x38]
    mov       dword ptr [esp+0x50],eax
    mov       eax,dword ptr [esi-0x1c8]
    xor       edi,eax
    mov       dword ptr [esi+0x38],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x194]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x3c]
    mov       dword ptr [esp+0x44],eax
    mov       eax,dword ptr [esi-0x1c4]
    xor       edi,eax
    mov       dword ptr [esi+0x3c],edi
    mov       eax,edi
    xor       eax,dword ptr [esp+0x198]
    mov       edi,eax
    mov       eax,dword ptr [esi-0x200]
    xor       dword ptr [esi],eax
    mov       eax,dword ptr [esi]
    mov       dword ptr [esp+0x88],edi
    xor       eax,edi
    xor       eax,dword ptr [esp+0x15c]
    mov       dword ptr [esp+0x84],eax
    mov       edi,eax
    mov       eax,dword ptr [esp+0x5c]
    xor       edi,eax
    mov       eax,dword ptr [esi-0x1fc]
    xor       dword ptr [esi+0x4],eax
    mov       eax,dword ptr [esi+0x4]
    xor       eax,dword ptr [esp+0x160]
    mov       dword ptr [esp+0x70],eax
    xor       eax,ecx
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esi-0x1f8]
    xor       dword ptr [esi+0x8],eax
    mov       eax,dword ptr [esi+0x8]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x164]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x60],eax
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1f4]
    mov       dword ptr [esp+0x3c],eax
    mov       eax,dword ptr [esi+0xc]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x168]
    mov       dword ptr [esi+0xc],eax
    xor       eax,ecx
    mov       dword ptr [esp+0x7c],eax
    mov       ecx,eax
    mov       eax,dword ptr [esi-0x1f0]
    xor       ecx,edx
    xor       dword ptr [esi+0x10],eax
    mov       eax,dword ptr [esi+0x10]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x16c]
    xor       eax,edx
    mov       edx,dword ptr [esi-0x1ec]
    mov       dword ptr [esp+0x64],eax
    mov       eax,dword ptr [esi+0x14]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x170]
    mov       dword ptr [esi+0x14],eax
    xor       eax,edx
    mov       edx,dword ptr [esi+0x18]
    mov       dword ptr [esp+0x58],eax
    mov       eax,dword ptr [esi-0x1e8]
    mov       dword ptr [esp+0x14],ecx
    xor       edx,eax
    mov       dword ptr [esi+0x18],edx
    mov       eax,edx
    mov       edx,dword ptr [esp+0x50]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x174]
    xor       eax,edx
    mov       edx,dword ptr [esi-0x1e4]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esi+0x1c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x178]
    mov       dword ptr [esi+0x1c],eax
    xor       eax,edx
    mov       edx,ecx
    mov       dword ptr [esp+0x54],eax
    not       edx
    mov       eax,edi
    mov       dword ptr [esp+0x10],edx
    or        eax,edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x4c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x3c]
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x10]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x10],edx
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    mov       dword ptr [esp+0x2c],eax
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x20],edx
    mov       edx,dword ptr [esp+0x10]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x3c]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x10],ecx
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x44]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x58]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x50]
    xor       ecx,edx
    mov       dword ptr [esp+0x10],ecx
    mov       edx,eax
    and       edx,ecx
    mov       ecx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x18],edx
    mov       edx,eax
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x18]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x3c]
    xor       edx,ecx
    mov       dword ptr [esp+0x18],ecx
    mov       dword ptr [esp+0x3c],edx
    mov       edx,dword ptr [esp+0x1c]
    mov       ecx,eax
    or        ecx,edx
    mov       edx,dword ptr [esp+0x20]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x40],ecx
    or        ecx,edx
    mov       edx,dword ptr [esp+0x10]
    xor       ecx,eax
    mov       dword ptr [esp+0x38],ecx
    xor       edi,ecx
    mov       ecx,eax
    or        ecx,edx
    mov       edx,dword ptr [esp+0x1c]
    and       eax,edx
    mov       edx,dword ptr [esp+0x14]
    xor       ecx,eax
    mov       eax,ecx
    not       eax
    xor       edx,eax
    mov       eax,dword ptr [esp+0x18]
    or        ecx,eax
    mov       eax,dword ptr [esp+0x38]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x40]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x30]
    not       ecx
    xor       eax,ecx
    lea       ecx,[esp+0x2c]
    push      ecx
    lea       ecx,[esp+0x14]
    push      ecx
    lea       ecx,[esp+0x24]
    push      ecx
    lea       ecx,[esp+0x2c]
    push      ecx
    push      edi
    mov       dword ptr [esp+0x44],eax
    push      eax
    mov       eax,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x2c],edx
    push      eax
    push      edx
    call      _csc_transF
    mov       ecx,dword ptr [esp+0x40]
    mov       eax,dword ptr [esp+0x30]
    mov       edx,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x1b8],ecx
    mov       ecx,dword ptr [esp+0x4c]
    mov       dword ptr [esp+0x1b0],eax
    mov       eax,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x1ac],ecx
    mov       ecx,dword ptr [esp+0x50]
    mov       dword ptr [esp+0x1a4],eax
    mov       eax,dword ptr [esp+0x90]
    mov       dword ptr [esp+0x1a0],ecx
    mov       ecx,dword ptr [esp+0x7c]
    add       esp,0x00000020
    mov       dword ptr [esp+0x194],edx
    mov       edx,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x17c],edi
    mov       edi,dword ptr [esp+0x84]
    xor       eax,ecx
    mov       dword ptr [esp+0x188],edx
    mov       dword ptr [esp+0x40],eax
    mov       eax,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x18],edi
    mov       ecx,dword ptr [esp+0x7c]
    mov       edx,dword ptr [esp+0x48]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x64]
    mov       dword ptr [esp+0x30],edx
    mov       edx,dword ptr [esp+0x58]
    xor       edx,dword ptr [esp+0x4c]
    mov       dword ptr [esp+0x14],eax
    mov       dword ptr [esp+0x10],ecx
    mov       dword ptr [esp+0x38],edx
    mov       edx,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [esp+0x54]
    xor       edx,dword ptr [esp+0x44]
    mov       dword ptr [esp+0x20],edx
    lea       edx,[esp+0x30]
    push      edx
    lea       edx,[esp+0x3c]
    push      edx
    lea       edx,[esp+0x24]
    push      edx
    lea       edx,[esp+0x2c]
    push      edx
    mov       edx,dword ptr [esp+0x50]
    push      edi
    push      edx
    push      eax
    push      ecx
    call      _csc_transF
    add       esp,0x00000020
    lea       eax,[esp+0x18]
    lea       ecx,[esp+0x40]
    lea       edx,[esp+0x14]
    push      eax
    push      ecx
    mov       ecx,dword ptr [esp+0x38]
    lea       eax,[esp+0x18]
    push      edx
    mov       edx,dword ptr [esp+0x44]
    push      eax
    mov       eax,dword ptr [esp+0x2c]
    push      ecx
    mov       ecx,dword ptr [esp+0x34]
    push      edx
    push      eax
    push      ecx
    call      _csc_transG
    add       esp,0x00000020
    lea       edx,[esp+0x30]
    lea       eax,[esp+0x38]
    lea       ecx,[esp+0x1c]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x20]
    lea       edx,[esp+0x28]
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    push      edx
    mov       edx,dword ptr [esp+0x24]
    push      eax
    mov       eax,dword ptr [esp+0x24]
    push      ecx
    push      edx
    push      eax
    call      _csc_transF
    mov       ecx,dword ptr [esp+0x40]
    mov       edx,dword ptr [esp+0x3c]
    mov       eax,dword ptr [esp+0x58]
    mov       dword ptr [esp+0x198],ecx
    mov       ecx,dword ptr [esp+0x50]
    mov       dword ptr [esp+0x194],edx
    mov       edx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x190],eax
    mov       eax,dword ptr [esp+0x34]
    mov       dword ptr [esp+0x18c],ecx
    mov       ecx,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x188],edx
    mov       edx,dword ptr [esp+0x38]
    add       esp,0x00000020
    mov       dword ptr [esp+0x164],eax
    mov       dword ptr [esp+0x160],ecx
    mov       eax,dword ptr [esi-0x1a0]
    mov       ecx,dword ptr [esi+0x60]
    mov       edi,dword ptr [esp+0x1bc]
    add       esi,0x00000040
    xor       ecx,eax
    mov       dword ptr [esp+0x15c],edx
    mov       edx,dword ptr [esi+0x24]
    mov       dword ptr [esi+0x20],ecx
    mov       eax,ecx
    mov       ecx,dword ptr [esi-0x1dc]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x28]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x1c0]
    mov       dword ptr [esp+0x5c],eax
    mov       eax,edx
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x1c4]
    mov       dword ptr [esi+0x24],edx
    mov       edx,eax
    mov       eax,dword ptr [esi-0x1d8]
    xor       edi,eax
    mov       eax,edi
    mov       dword ptr [esi+0x28],edi
    mov       edi,dword ptr [esi+0x2c]
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1d4]
    mov       dword ptr [esp+0x48],eax
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x1c8]
    mov       eax,edi
    mov       dword ptr [esi+0x2c],edi
    mov       edi,dword ptr [esi+0x30]
    xor       eax,ecx
    mov       ecx,eax
    mov       eax,dword ptr [esi-0x1d0]
    xor       edi,eax
    mov       dword ptr [esp+0x34],ecx
    mov       dword ptr [esi+0x30],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x1cc]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x34]
    mov       dword ptr [esp+0x4c],eax
    mov       eax,dword ptr [esi-0x1cc]
    xor       edi,eax
    mov       dword ptr [esi+0x34],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x1d0]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x38]
    mov       dword ptr [esp+0x50],eax
    mov       eax,dword ptr [esi-0x1c8]
    xor       edi,eax
    mov       dword ptr [esi+0x38],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x1d4]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x3c]
    mov       dword ptr [esp+0x44],eax
    mov       eax,dword ptr [esi-0x1c4]
    xor       edi,eax
    mov       dword ptr [esi+0x3c],edi
    mov       eax,edi
    xor       eax,dword ptr [esp+0x1d8]
    mov       edi,eax
    mov       eax,dword ptr [esi-0x200]
    xor       dword ptr [esi],eax
    mov       eax,dword ptr [esi]
    mov       dword ptr [esp+0x88],edi
    xor       eax,edi
    mov       edi,dword ptr [esp+0x19c]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x4]
    mov       dword ptr [esp+0x84],eax
    mov       eax,dword ptr [esi-0x1fc]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x1a0]
    mov       dword ptr [esi+0x4],edi
    xor       edi,eax
    mov       eax,dword ptr [esi-0x1f8]
    xor       dword ptr [esi+0x8],eax
    mov       eax,dword ptr [esi+0x8]
    xor       eax,edx
    xor       eax,dword ptr [esp+0x1a4]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esi-0x1f4]
    xor       dword ptr [esi+0xc],eax
    mov       eax,dword ptr [esi+0xc]
    xor       eax,dword ptr [esp+0x1a8]
    mov       dword ptr [esp+0x7c],eax
    mov       eax,dword ptr [esi-0x1f0]
    xor       dword ptr [esi+0x10],eax
    mov       eax,dword ptr [esi+0x10]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x1ac]
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1ec]
    mov       dword ptr [esp+0x64],eax
    mov       eax,dword ptr [esi+0x14]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x1b0]
    mov       dword ptr [esi+0x14],eax
    xor       eax,ecx
    mov       ecx,dword ptr [esi+0x18]
    mov       dword ptr [esp+0x58],eax
    mov       eax,dword ptr [esi-0x1e8]
    xor       ecx,eax
    mov       dword ptr [esi+0x18],ecx
    mov       eax,ecx
    mov       ecx,dword ptr [esp+0x50]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x1b4]
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1e4]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esi+0x1c]
    xor       eax,ecx
    mov       dword ptr [esi+0x1c],eax
    mov       ecx,eax
    mov       eax,dword ptr [esp+0x1b8]
    xor       ecx,eax
    lea       eax,[esp+0x1bc]
    mov       dword ptr [esp+0x54],ecx
    lea       ecx,[esp+0x1c0]
    push      eax
    push      ecx
    lea       eax,[esp+0x1cc]
    lea       ecx,[esp+0x1d0]
    push      eax
    push      ecx
    lea       eax,[esp+0x1dc]
    lea       ecx,[esp+0x1e0]
    push      eax
    push      ecx
    lea       eax,[esp+0x1ec]
    lea       ecx,[esp+0x1f0]
    push      eax
    mov       eax,dword ptr [esp+0xa0]
    push      ecx
    mov       ecx,dword ptr [esp+0x7c]
    xor       eax,ecx
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x80]
    push      eax
    mov       eax,dword ptr [esp+0xa0]
    push      ecx
    mov       ecx,dword ptr [esp+0x70]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x8c]
    push      edx
    mov       edx,dword ptr [esp+0x60]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x84]
    push      eax
    mov       eax,dword ptr [esp+0x7c]
    xor       ecx,eax
    push      ecx
    mov       ecx,dword ptr [esp+0x84]
    xor       edx,ecx
    mov       eax,dword ptr [esp+0x9c]
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0xc4]
    xor       ecx,eax
    push      ecx
    call      _csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x19c]
    lea       eax,[esp+0x1a0]
    lea       ecx,[esp+0x1a4]
    push      edx
    push      eax
    lea       edx,[esp+0x1b0]
    push      ecx
    lea       eax,[esp+0x1b8]
    push      edx
    lea       ecx,[esp+0x1c0]
    push      eax
    lea       edx,[esp+0x1c8]
    push      ecx
    mov       ecx,dword ptr [esp+0x9c]
    lea       eax,[esp+0x1d0]
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    push      eax
    mov       eax,dword ptr [esp+0x9c]
    push      ecx
    mov       ecx,dword ptr [esp+0x80]
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x8c]
    push      edi
    push      edx
    mov       edx,dword ptr [esp+0x74]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x78]
    push      eax
    mov       eax,dword ptr [esp+0x94]
    push      eax
    mov       eax,dword ptr [esp+0x8c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x78]
    push      eax
    mov       eax,dword ptr [esp+0x8c]
    xor       eax,edx
    push      ecx
    push      eax
    call      _csc_transP
    mov       eax,dword ptr [esp+0x15c]
    mov       edx,dword ptr [csc_tabe+0x20]
    mov       ecx,dword ptr [csc_tabe+0x24]
    mov       edi,dword ptr [csc_tabe+0x28]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x2c]
    mov       dword ptr [esp+0x9c],eax
    mov       eax,dword ptr [esp+0x160]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x164]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x30]
    mov       dword ptr [esp+0x88],ecx
    mov       ecx,dword ptr [esp+0x168]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x16c]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x34]
    mov       dword ptr [esp+0x8c],edx
    mov       edx,dword ptr [esp+0x170]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x38]
    mov       dword ptr [esp+0x90],edx
    mov       edx,dword ptr [esp+0x174]
    add       esp,0x00000040
    add       esi,0x00000040
    xor       edx,edi
    mov       dword ptr [esp+0x80],eax
    mov       dword ptr [esp+0x34],ecx
    mov       dword ptr [esp+0x44],edx
    mov       edi,dword ptr [esp+0x138]
    mov       edx,dword ptr [csc_tabe+0x3c]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xdc]
    mov       dword ptr [esp+0x88],edi
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xe0]
    xor       edx,dword ptr [csc_tabe+0x4]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0xe4]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xe8]
    xor       edx,dword ptr [csc_tabe+0xc]
    mov       dword ptr [esp+0x60],eax
    mov       dword ptr [esp+0x7c],edx
    mov       edx,dword ptr [esp+0xec]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x10]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x14]
    mov       dword ptr [esp+0x64],ecx
    mov       ecx,dword ptr [esp+0xf0]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xf4]
    mov       dword ptr [esp+0x58],ecx
    mov       ecx,dword ptr [esp+0x50]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x18]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x1c]
    mov       dword ptr [esp+0x68],ecx
    mov       ecx,dword ptr [esp+0xf8]
    xor       ecx,edx
    lea       edx,[esp+0x11c]
    mov       dword ptr [esp+0x54],ecx
    lea       ecx,[esp+0x120]
    push      edx
    push      ecx
    lea       edx,[esp+0x12c]
    lea       ecx,[esp+0x130]
    push      edx
    push      ecx
    lea       edx,[esp+0x13c]
    lea       ecx,[esp+0x140]
    push      edx
    push      ecx
    lea       edx,[esp+0x14c]
    lea       ecx,[esp+0x150]
    push      edx
    push      ecx
    mov       ecx,dword ptr [esp+0x7c]
    mov       edx,edi
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x90]
    push      edx
    mov       edx,dword ptr [esp+0xa4]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xa0]
    push      ecx
    mov       ecx,dword ptr [esp+0x70]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x80]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x98]
    push      eax
    mov       eax,dword ptr [esp+0x84]
    xor       ecx,eax
    push      ecx
    mov       ecx,dword ptr [esp+0x7c]
    xor       edx,ecx
    push      edx
    mov       eax,dword ptr [esp+0x90]
    mov       edx,dword ptr [esp+0xc4]
    xor       eax,edx
    push      eax
    call      _csc_transP
    add       esp,0x00000040
    lea       ecx,[esp+0xdc]
    lea       edx,[esp+0xe0]
    lea       eax,[esp+0xe4]
    push      ecx
    push      edx
    lea       ecx,[esp+0xf0]
    push      eax
    lea       edx,[esp+0xf8]
    push      ecx
    lea       eax,[esp+0x100]
    push      edx
    lea       ecx,[esp+0x108]
    push      eax
    mov       eax,dword ptr [esp+0x88]
    lea       edx,[esp+0x110]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    push      edi
    xor       eax,edx
    mov       edx,dword ptr [esp+0x6c]
    push      eax
    mov       eax,dword ptr [esp+0x88]
    mov       ecx,dword ptr [esp+0x8c]
    push      eax
    mov       eax,dword ptr [esp+0xa8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x78]
    push      eax
    mov       eax,dword ptr [esp+0x88]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x98]
    push      ecx
    push      eax
    mov       eax,dword ptr [esp+0x8c]
    push      edx
    xor       eax,dword ptr [esp+0x80]
    push      eax
    call      _csc_transP
    mov       eax,dword ptr [esp+0x1dc]
    mov       edx,dword ptr [csc_tabe+0x60]
    mov       ecx,dword ptr [csc_tabe+0x64]
    mov       edi,dword ptr [csc_tabe+0x68]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x6c]
    mov       dword ptr [esp+0x9c],eax
    mov       eax,dword ptr [esp+0x1e0]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x1e4]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x70]
    mov       dword ptr [esp+0x88],ecx
    mov       ecx,dword ptr [esp+0x1e8]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x1ec]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x74]
    mov       dword ptr [esp+0x8c],edx
    mov       edx,dword ptr [esp+0x1f0]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x78]
    mov       dword ptr [esp+0x90],edx
    mov       edx,dword ptr [esp+0x1f4]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x1f8]
    mov       dword ptr [esp+0x84],edx
    mov       edx,dword ptr [csc_tabe+0x7c]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x19c]
    add       esp,0x00000040
    mov       dword ptr [esp+0x88],edi
    mov       dword ptr [esp+0x80],eax
    mov       dword ptr [esp+0x34],ecx
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe+0x40]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x160]
    xor       edx,dword ptr [csc_tabe+0x44]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x164]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x48]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x168]
    xor       edx,dword ptr [csc_tabe+0x4c]
    mov       dword ptr [esp+0x60],eax
    mov       dword ptr [esp+0x7c],edx
    mov       edx,dword ptr [esp+0x16c]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x50]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x54]
    mov       dword ptr [esp+0x64],ecx
    mov       ecx,dword ptr [esp+0x170]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x174]
    mov       dword ptr [esp+0x58],ecx
    mov       ecx,dword ptr [esp+0x50]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x58]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x5c]
    mov       dword ptr [esp+0x68],ecx
    mov       ecx,dword ptr [esp+0x178]
    xor       ecx,edx
    lea       edx,[esp+0x1a0]
    mov       dword ptr [esp+0x54],ecx
    lea       ecx,[esp+0x19c]
    push      ecx
    push      edx
    lea       ecx,[esp+0x1ac]
    lea       edx,[esp+0x1b0]
    push      ecx
    push      edx
    lea       ecx,[esp+0x1bc]
    lea       edx,[esp+0x1c0]
    push      ecx
    push      edx
    lea       ecx,[esp+0x1cc]
    lea       edx,[esp+0x1d0]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x90]
    push      ecx
    mov       ecx,dword ptr [esp+0xa4]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x70]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0xa8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x84]
    push      eax
    mov       eax,dword ptr [esp+0x7c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x98]
    push      ecx
    mov       ecx,dword ptr [esp+0x84]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0xc4]
    xor       ecx,eax
    push      ecx
    call      _csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x15c]
    lea       eax,[esp+0x160]
    push      edx
    lea       ecx,[esp+0x168]
    push      eax
    lea       edx,[esp+0x170]
    push      ecx
    lea       eax,[esp+0x178]
    push      edx
    lea       ecx,[esp+0x180]
    push      eax
    lea       edx,[esp+0x188]
    push      ecx
    lea       eax,[esp+0x190]
    push      edx
    mov       edx,dword ptr [esp+0x78]
    mov       ecx,dword ptr [esp+0x7c]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    push      edi
    xor       eax,edx
    mov       edx,dword ptr [esp+0x6c]
    push      eax
    mov       eax,dword ptr [esp+0xa4]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x8c]
    push      ecx
    push      eax
    mov       eax,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x80]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x78]
    push      eax
    mov       eax,dword ptr [esp+0xa0]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    xor       eax,edx
    push      eax
    call      _csc_transP
    mov       eax,dword ptr [esp+0x17c]
    mov       edx,dword ptr [csc_tabe+0xa0]
    mov       ecx,dword ptr [csc_tabe+0xa4]
    mov       edi,dword ptr [csc_tabe+0xa8]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0xac]
    mov       dword ptr [esp+0x9c],eax
    mov       eax,dword ptr [esp+0x180]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x184]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0xb0]
    mov       dword ptr [esp+0x88],ecx
    mov       ecx,dword ptr [esp+0x188]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x18c]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0xb4]
    mov       dword ptr [esp+0x8c],edx
    mov       edx,dword ptr [esp+0x190]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0xb8]
    mov       dword ptr [esp+0x90],edx
    mov       edx,dword ptr [esp+0x194]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x198]
    mov       dword ptr [esp+0x84],edx
    mov       edx,dword ptr [csc_tabe+0xbc]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x13c]
    mov       dword ptr [esp+0xc8],edi
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe+0x80]
    add       esp,0x00000040
    xor       edi,edx
    mov       edx,dword ptr [esp+0x100]
    xor       edx,dword ptr [csc_tabe+0x84]
    mov       dword ptr [esp+0x80],eax
    mov       dword ptr [esp+0x34],ecx
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x104]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x88]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x108]
    xor       edx,dword ptr [csc_tabe+0x8c]
    mov       dword ptr [esp+0x60],eax
    mov       dword ptr [esp+0x7c],edx
    mov       edx,dword ptr [esp+0x10c]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x90]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x94]
    mov       dword ptr [esp+0x64],ecx
    mov       ecx,dword ptr [esp+0x110]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x114]
    mov       dword ptr [esp+0x58],ecx
    mov       ecx,dword ptr [esp+0x50]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x98]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x9c]
    mov       dword ptr [esp+0x68],ecx
    mov       ecx,dword ptr [esp+0x118]
    xor       ecx,edx
    lea       edx,[esp+0x140]
    mov       dword ptr [esp+0x54],ecx
    lea       ecx,[esp+0x13c]
    push      ecx
    push      edx
    lea       ecx,[esp+0x14c]
    lea       edx,[esp+0x150]
    push      ecx
    push      edx
    lea       ecx,[esp+0x15c]
    lea       edx,[esp+0x160]
    push      ecx
    push      edx
    lea       ecx,[esp+0x16c]
    lea       edx,[esp+0x170]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x90]
    push      ecx
    mov       ecx,dword ptr [esp+0xa4]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x70]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0xa8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x84]
    push      eax
    mov       eax,dword ptr [esp+0x7c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x98]
    push      ecx
    mov       ecx,dword ptr [esp+0x84]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0xc4]
    xor       ecx,eax
    push      ecx
    call      _csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0xfc]
    lea       eax,[esp+0x100]
    lea       ecx,[esp+0x104]
    push      edx
    push      eax
    push      ecx
    lea       edx,[esp+0x114]
    lea       eax,[esp+0x118]
    push      edx
    lea       ecx,[esp+0x120]
    push      eax
    lea       edx,[esp+0x128]
    push      ecx
    lea       eax,[esp+0x130]
    push      edx
    mov       edx,dword ptr [esp+0x78]
    mov       ecx,dword ptr [esp+0x7c]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    push      edi
    xor       eax,edx
    mov       edx,dword ptr [esp+0x6c]
    push      eax
    mov       eax,dword ptr [esp+0xa4]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x8c]
    push      ecx
    push      eax
    mov       eax,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x80]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x78]
    push      eax
    mov       eax,dword ptr [esp+0xa0]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    xor       eax,edx
    push      eax
    call      _csc_transP
    mov       eax,dword ptr [esp+0x1fc]
    mov       edx,dword ptr [csc_tabe+0xe0]
    mov       ecx,dword ptr [csc_tabe+0xe4]
    mov       edi,dword ptr [csc_tabe+0xe8]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0xec]
    mov       dword ptr [esp+0x9c],eax
    mov       eax,dword ptr [esp+0x200]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x204]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0xf0]
    mov       dword ptr [esp+0x88],ecx
    mov       ecx,dword ptr [esp+0x208]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x20c]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0xf4]
    mov       dword ptr [esp+0x8c],edx
    mov       edx,dword ptr [esp+0x210]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0xf8]
    mov       dword ptr [esp+0x90],edx
    mov       edx,dword ptr [esp+0x214]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x218]
    mov       dword ptr [esp+0x84],edx
    mov       edx,dword ptr [csc_tabe+0xfc]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x1bc]
    mov       dword ptr [esp+0xc8],edi
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe+0xc0]
    mov       dword ptr [esp+0xc0],eax
    xor       edi,edx
    mov       edx,dword ptr [esp+0x1c0]
    xor       edx,dword ptr [csc_tabe+0xc4]
    add       esp,0x00000040
    mov       dword ptr [esp+0x34],ecx
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x184]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0xc8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x188]
    xor       edx,dword ptr [csc_tabe+0xcc]
    mov       dword ptr [esp+0x60],eax
    mov       dword ptr [esp+0x7c],edx
    mov       edx,dword ptr [esp+0x18c]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0xd0]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0xd4]
    mov       dword ptr [esp+0x64],ecx
    mov       ecx,dword ptr [esp+0x190]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x194]
    mov       dword ptr [esp+0x58],ecx
    mov       ecx,dword ptr [esp+0x50]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0xd8]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0xdc]
    mov       dword ptr [esp+0x68],ecx
    mov       ecx,dword ptr [esp+0x198]
    xor       ecx,edx
    lea       edx,[esp+0x1c0]
    mov       dword ptr [esp+0x54],ecx
    lea       ecx,[esp+0x1bc]
    push      ecx
    push      edx
    lea       ecx,[esp+0x1cc]
    lea       edx,[esp+0x1d0]
    push      ecx
    push      edx
    lea       ecx,[esp+0x1dc]
    lea       edx,[esp+0x1e0]
    push      ecx
    push      edx
    lea       ecx,[esp+0x1ec]
    lea       edx,[esp+0x1f0]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x90]
    push      ecx
    mov       ecx,dword ptr [esp+0xa4]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x70]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0xa8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x84]
    push      eax
    mov       eax,dword ptr [esp+0x7c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x98]
    push      ecx
    mov       ecx,dword ptr [esp+0x84]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0xc4]
    xor       ecx,eax
    push      ecx
    call      _csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x17c]
    lea       eax,[esp+0x180]
    lea       ecx,[esp+0x184]
    push      edx
    push      eax
    lea       edx,[esp+0x190]
    push      ecx
    lea       eax,[esp+0x198]
    push      edx
    lea       ecx,[esp+0x1a0]
    push      eax
    push      ecx
    lea       edx,[esp+0x1ac]
    lea       eax,[esp+0x1b0]
    push      edx
    mov       edx,dword ptr [esp+0x78]
    mov       ecx,dword ptr [esp+0x7c]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    push      edi
    xor       eax,edx
    mov       edx,dword ptr [esp+0x6c]
    push      eax
    mov       eax,dword ptr [esp+0xa4]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x8c]
    push      ecx
    push      eax
    mov       eax,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x80]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x78]
    push      eax
    mov       eax,dword ptr [esp+0xa0]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    xor       eax,edx
    push      eax
    call      _csc_transP
    mov       eax,dword ptr [esp+0x19c]
    mov       edx,dword ptr [csc_tabe+0x120]
    mov       ecx,dword ptr [csc_tabe+0x124]
    mov       edi,dword ptr [csc_tabe+0x128]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x12c]
    mov       dword ptr [esp+0x9c],eax
    mov       eax,dword ptr [esp+0x1a0]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x1a4]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x130]
    mov       dword ptr [esp+0x88],ecx
    mov       ecx,dword ptr [esp+0x1a8]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x1ac]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x134]
    mov       dword ptr [esp+0x8c],edx
    mov       edx,dword ptr [esp+0x1b0]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x138]
    mov       dword ptr [esp+0x90],edx
    mov       edx,dword ptr [esp+0x1b4]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x1b8]
    mov       dword ptr [esp+0x84],edx
    mov       edx,dword ptr [csc_tabe+0x13c]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x11c]
    mov       dword ptr [esp+0xc8],edi
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe+0x100]
    mov       dword ptr [esp+0xc0],eax
    xor       edi,edx
    mov       edx,dword ptr [esp+0x120]
    xor       edx,dword ptr [csc_tabe+0x104]
    mov       dword ptr [esp+0x74],ecx
    add       esp,0x00000040
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0xe4]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x108]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xe8]
    xor       edx,dword ptr [csc_tabe+0x10c]
    mov       dword ptr [esp+0x60],eax
    mov       dword ptr [esp+0x7c],edx
    mov       edx,dword ptr [esp+0xec]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x110]
    xor       ecx,edx
    mov       dword ptr [esp+0x64],ecx
    mov       ecx,dword ptr [esp+0xf0]
    mov       edx,dword ptr [csc_tabe+0x114]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xf4]
    mov       dword ptr [esp+0x58],ecx
    mov       ecx,dword ptr [esp+0x50]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x118]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x11c]
    mov       dword ptr [esp+0x68],ecx
    mov       ecx,dword ptr [esp+0xf8]
    xor       ecx,edx
    lea       edx,[esp+0x160]
    mov       dword ptr [esp+0x54],ecx
    lea       ecx,[esp+0x15c]
    push      ecx
    push      edx
    lea       ecx,[esp+0x16c]
    lea       edx,[esp+0x170]
    push      ecx
    push      edx
    lea       ecx,[esp+0x17c]
    lea       edx,[esp+0x180]
    push      ecx
    push      edx
    lea       ecx,[esp+0x18c]
    lea       edx,[esp+0x190]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x90]
    push      ecx
    mov       ecx,dword ptr [esp+0xa4]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x70]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0xa8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x84]
    push      eax
    mov       eax,dword ptr [esp+0x7c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x98]
    push      ecx
    mov       ecx,dword ptr [esp+0x84]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0xc4]
    xor       ecx,eax
    push      ecx
    call      _csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0xdc]
    lea       eax,[esp+0xe0]
    lea       ecx,[esp+0xe4]
    push      edx
    push      eax
    lea       edx,[esp+0xf0]
    push      ecx
    lea       eax,[esp+0xf8]
    push      edx
    lea       ecx,[esp+0x100]
    push      eax
    lea       edx,[esp+0x108]
    push      ecx
    lea       eax,[esp+0x110]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x90]
    push      edi
    mov       edx,dword ptr [esp+0x80]
    mov       ecx,dword ptr [esp+0x84]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x6c]
    push      eax
    mov       eax,dword ptr [esp+0xa4]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x8c]
    push      ecx
    push      eax
    mov       eax,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x80]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x78]
    push      eax
    mov       eax,dword ptr [esp+0xa0]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    xor       eax,edx
    push      eax
    call      _csc_transP
    mov       eax,dword ptr [esp+0x1bc]
    mov       edx,dword ptr [csc_tabe+0x160]
    mov       ecx,dword ptr [csc_tabe+0x164]
    mov       edi,dword ptr [csc_tabe+0x168]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x16c]
    mov       dword ptr [esp+0x9c],eax
    mov       eax,dword ptr [esp+0x1c0]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x1c4]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x170]
    mov       dword ptr [esp+0x88],ecx
    mov       ecx,dword ptr [esp+0x1c8]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x1cc]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x174]
    mov       dword ptr [esp+0x8c],edx
    mov       edx,dword ptr [esp+0x1d0]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x178]
    mov       dword ptr [esp+0x90],edx
    mov       edx,dword ptr [esp+0x1d4]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x1d8]
    mov       dword ptr [esp+0x84],edx
    mov       edx,dword ptr [csc_tabe+0x17c]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x13c]
    mov       dword ptr [esp+0xc8],edi
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe+0x140]
    mov       dword ptr [esp+0xc0],eax
    xor       edi,edx
    mov       edx,dword ptr [esp+0x140]
    xor       edx,dword ptr [csc_tabe+0x144]
    mov       dword ptr [esp+0x74],ecx
    add       esp,0x00000040
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x104]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x148]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x108]
    xor       edx,dword ptr [csc_tabe+0x14c]
    mov       dword ptr [esp+0x60],eax
    mov       dword ptr [esp+0x7c],edx
    mov       edx,dword ptr [esp+0x10c]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x150]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x154]
    mov       dword ptr [esp+0x64],ecx
    mov       ecx,dword ptr [esp+0x110]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x114]
    mov       dword ptr [esp+0x58],ecx
    mov       ecx,dword ptr [esp+0x50]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x158]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x15c]
    mov       dword ptr [esp+0x68],ecx
    mov       ecx,dword ptr [esp+0x118]
    xor       ecx,edx
    lea       edx,[esp+0x180]
    mov       dword ptr [esp+0x54],ecx
    lea       ecx,[esp+0x17c]
    push      ecx
    push      edx
    lea       ecx,[esp+0x18c]
    lea       edx,[esp+0x190]
    push      ecx
    push      edx
    lea       ecx,[esp+0x19c]
    lea       edx,[esp+0x1a0]
    push      ecx
    push      edx
    lea       ecx,[esp+0x1ac]
    lea       edx,[esp+0x1b0]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x90]
    push      ecx
    mov       ecx,dword ptr [esp+0xa4]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x70]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0xa8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x84]
    push      eax
    mov       eax,dword ptr [esp+0x7c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x98]
    push      ecx
    mov       ecx,dword ptr [esp+0x84]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0xc4]
    xor       ecx,eax
    push      ecx
    call      _csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0xfc]
    lea       eax,[esp+0x100]
    lea       ecx,[esp+0x104]
    push      edx
    push      eax
    lea       edx,[esp+0x110]
    push      ecx
    lea       eax,[esp+0x118]
    push      edx
    lea       ecx,[esp+0x120]
    push      eax
    lea       edx,[esp+0x128]
    push      ecx
    mov       ecx,dword ptr [esp+0x78]
    lea       eax,[esp+0x130]
    push      edx
    mov       edx,dword ptr [esp+0x78]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    xor       eax,edx
    push      edi
    push      eax
    mov       eax,dword ptr [esp+0xa4]
    push      ecx
    mov       edx,dword ptr [esp+0x74]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x90]
    push      eax
    mov       eax,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x80]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x78]
    push      eax
    mov       eax,dword ptr [esp+0xa0]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    xor       eax,edx
    push      eax
    call      _csc_transP
    mov       eax,dword ptr [esp+0x1dc]
    mov       edx,dword ptr [csc_tabe+0x1a0]
    mov       ecx,dword ptr [csc_tabe+0x1a4]
    mov       edi,dword ptr [csc_tabe+0x1a8]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x1ac]
    mov       dword ptr [esp+0x9c],eax
    mov       eax,dword ptr [esp+0x1e0]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x1e4]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1b0]
    mov       dword ptr [esp+0x88],ecx
    mov       ecx,dword ptr [esp+0x1e8]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x1ec]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x1b4]
    mov       dword ptr [esp+0x8c],edx
    mov       edx,dword ptr [esp+0x1f0]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x1b8]
    mov       dword ptr [esp+0x90],edx
    mov       edx,dword ptr [esp+0x1f4]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x1f8]
    mov       dword ptr [esp+0x84],edx
    mov       edx,dword ptr [csc_tabe+0x1bc]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x15c]
    mov       dword ptr [esp+0xc8],edi
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe+0x180]
    mov       dword ptr [esp+0xc0],eax
    xor       edi,edx
    mov       edx,dword ptr [esp+0x160]
    xor       edx,dword ptr [csc_tabe+0x184]
    mov       dword ptr [esp+0x74],ecx
    add       esp,0x00000040
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x124]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x188]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x128]
    xor       edx,dword ptr [csc_tabe+0x18c]
    mov       dword ptr [esp+0x60],eax
    mov       dword ptr [esp+0x7c],edx
    mov       edx,dword ptr [esp+0x12c]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x190]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x194]
    mov       dword ptr [esp+0x64],ecx
    mov       ecx,dword ptr [esp+0x130]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x134]
    mov       dword ptr [esp+0x58],ecx
    mov       ecx,dword ptr [esp+0x50]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x198]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x19c]
    mov       dword ptr [esp+0x68],ecx
    mov       ecx,dword ptr [esp+0x138]
    xor       ecx,edx
    lea       edx,[esp+0x1a0]
    mov       dword ptr [esp+0x54],ecx
    lea       ecx,[esp+0x19c]
    push      ecx
    push      edx
    lea       ecx,[esp+0x1ac]
    lea       edx,[esp+0x1b0]
    push      ecx
    push      edx
    lea       ecx,[esp+0x1bc]
    lea       edx,[esp+0x1c0]
    push      ecx
    push      edx
    lea       ecx,[esp+0x1cc]
    lea       edx,[esp+0x1d0]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x90]
    push      ecx
    mov       ecx,dword ptr [esp+0xa4]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x70]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0xa8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x84]
    push      eax
    mov       eax,dword ptr [esp+0x7c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x98]
    push      ecx
    mov       ecx,dword ptr [esp+0x84]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0xc4]
    xor       ecx,eax
    push      ecx
    call      _csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x11c]
    lea       eax,[esp+0x120]
    lea       ecx,[esp+0x124]
    push      edx
    push      eax
    lea       edx,[esp+0x130]
    push      ecx
    lea       eax,[esp+0x138]
    push      edx
    lea       ecx,[esp+0x140]
    push      eax
    lea       edx,[esp+0x148]
    push      ecx
    lea       eax,[esp+0x150]
    push      edx
    mov       edx,dword ptr [esp+0x78]
    mov       ecx,dword ptr [esp+0x7c]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      edi
    push      eax
    mov       eax,dword ptr [esp+0xa4]
    push      ecx
    xor       eax,edx
    mov       edx,dword ptr [esp+0x90]
    push      eax
    mov       eax,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x80]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x78]
    push      eax
    mov       eax,dword ptr [esp+0xa0]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    xor       eax,edx
    push      eax
    call      _csc_transP
    mov       edx,dword ptr [esp+0x1fc]
    mov       edi,dword ptr [csc_tabe+0x1e0]
    mov       eax,dword ptr [esp+0x200]
    mov       ecx,dword ptr [csc_tabe+0x1e4]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x204]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x1e8]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1ec]
    mov       dword ptr [esp+0x88],ecx
    mov       ecx,dword ptr [esp+0x208]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1f0]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0x20c]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1f4]
    mov       dword ptr [esp+0x8c],ecx
    mov       ecx,dword ptr [esp+0x210]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1f8]
    mov       dword ptr [esp+0x90],ecx
    mov       ecx,dword ptr [esp+0x214]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x218]
    mov       dword ptr [esp+0x84],ecx
    mov       ecx,dword ptr [csc_tabe+0x1fc]
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x17c]
    mov       dword ptr [esp+0xc8],edi
    xor       edi,ecx
    mov       ecx,dword ptr [csc_tabe+0x1c0]
    mov       dword ptr [esp+0xc0],eax
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x180]
    xor       ecx,dword ptr [csc_tabe+0x1c4]
    add       esp,0x00000040
    mov       dword ptr [esp+0x5c],edx
    mov       dword ptr [esp+0x70],ecx
    mov       ecx,dword ptr [esp+0x144]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x1c8]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x148]
    xor       ecx,dword ptr [csc_tabe+0x1cc]
    mov       dword ptr [esp+0x60],eax
    mov       dword ptr [esp+0x7c],ecx
    mov       ecx,dword ptr [esp+0x34]
    xor       ecx,dword ptr [esp+0x14c]
    xor       ecx,dword ptr [csc_tabe+0x1d0]
    mov       dword ptr [esp+0x64],ecx
    mov       ecx,dword ptr [esp+0x150]
    xor       ecx,dword ptr [csc_tabe+0x1d4]
    mov       dword ptr [esp+0x58],ecx
    mov       ecx,dword ptr [esp+0x50]
    xor       ecx,dword ptr [esp+0x154]
    xor       ecx,dword ptr [csc_tabe+0x1d8]
    mov       dword ptr [esp+0x68],ecx
    mov       ecx,dword ptr [esp+0x158]
    xor       ecx,dword ptr [csc_tabe+0x1dc]
    mov       dword ptr [esp+0x54],ecx
    lea       ecx,[esp+0x1bc]
    push      ecx
    lea       ecx,[esp+0x1c4]
    push      ecx
    lea       ecx,[esp+0x1cc]
    push      ecx
    lea       ecx,[esp+0x1d4]
    push      ecx
    lea       ecx,[esp+0x1dc]
    push      ecx
    lea       ecx,[esp+0x1e4]
    push      ecx
    lea       ecx,[esp+0x1ec]
    push      ecx
    lea       ecx,[esp+0x1f4]
    push      ecx
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x90]
    push      ecx
    mov       ecx,dword ptr [esp+0xa4]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x70]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0xa8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x84]
    push      eax
    mov       eax,dword ptr [esp+0x7c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x98]
    push      ecx
    mov       ecx,dword ptr [esp+0x84]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x7c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0xc4]
    xor       ecx,eax
    push      ecx
    call      _csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x13c]
    lea       eax,[esp+0x140]
    lea       ecx,[esp+0x144]
    push      edx
    push      eax
    lea       edx,[esp+0x150]
    push      ecx
    lea       eax,[esp+0x158]
    push      edx
    lea       ecx,[esp+0x160]
    push      eax
    lea       edx,[esp+0x168]
    push      ecx
    lea       eax,[esp+0x170]
    push      edx
    mov       edx,dword ptr [esp+0x78]
    mov       ecx,dword ptr [esp+0x7c]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      edi
    mov       edi,dword ptr [esp+0x70]
    push      eax
    mov       eax,dword ptr [esp+0xa4]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x8c]
    push      ecx
    push      eax
    mov       eax,dword ptr [esp+0x88]
    push      edx
    xor       eax,edi
    mov       edi,dword ptr [esp+0x78]
    push      eax
    mov       eax,dword ptr [esp+0xa0]
    push      eax
    mov       eax,dword ptr [esp+0x90]
    xor       eax,edi
    push      eax
    call      _csc_transP
    mov       eax,dword ptr [esp+0xcc]
    add       esp,0x00000040
    dec       eax
    mov       dword ptr [esp+0x8c],eax
    ljne       X$9
    lea       ecx,[esp+0x15c]
    lea       eax,[esi+0xc0]
    mov       dword ptr [esp+0x24],ecx
    mov       ecx,dword ptr [esp+0x200]
    mov       dword ptr [esp+0x34],0xffffffff
    mov       dword ptr [esp+0x10],0x00000000
    lea       edx,[ecx+0xc0]
    mov       dword ptr [esp+0x20],eax
    mov       dword ptr [esp+0x1c],edx
    lea       edx,[esp+0xfc]
    sub       edx,ecx
    mov       dword ptr [esp+0x38],edx
    lea       edx,[esp+0xdc]
    sub       edx,ecx
    mov       dword ptr [esp+0x40],edx
X$12:
    cmp       dword ptr [esp+0x10],0x00000008
    ljge       X$26
    lea       edi,[eax-0x60]
    lea       ecx,[eax-0xa0]
    mov       dword ptr [esp+0x2c],edi
    lea       edi,[eax-0x40]
    push      esi
    mov       dword ptr [esp+0x90],ecx
    lea       edx,[eax-0x80]
    mov       dword ptr [esp+0x1c],edi
    push      ecx
    mov       ecx,dword ptr [esp+0x34]
    lea       edi,[eax-0x20]
    mov       dword ptr [esp+0x9c],edx
    push      edx
    mov       edx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x20],edi
    push      ecx
    mov       ecx,dword ptr [esp+0x24]
    push      edx
    lea       edi,[eax+0x20]
    push      ecx
    push      eax
    mov       eax,dword ptr [esp+0x94]
    mov       dword ptr [esp+0x4c],edi
    push      edi
    mov       edi,dword ptr [esp+0x48]
    mov       edx,dword ptr [eax]
    mov       ecx,dword ptr [edi]
    xor       edx,ecx
    mov       ecx,dword ptr [edi+0x4]
    push      edx
    mov       edx,dword ptr [eax+0x4]
    xor       ecx,edx
    mov       edx,dword ptr [edi+0x8]
    push      ecx
    mov       ecx,dword ptr [eax+0x8]
    xor       edx,ecx
    mov       ecx,dword ptr [edi+0xc]
    push      edx
    mov       edx,dword ptr [eax+0xc]
    xor       ecx,edx
    mov       edx,dword ptr [edi+0x10]
    push      ecx
    mov       ecx,dword ptr [eax+0x10]
    xor       edx,ecx
    mov       ecx,dword ptr [edi+0x14]
    push      edx
    mov       edx,dword ptr [eax+0x14]
    xor       ecx,edx
    mov       edx,dword ptr [edi+0x18]
    push      ecx
    mov       ecx,dword ptr [eax+0x18]
    xor       edx,ecx
    mov       ecx,dword ptr [edi+0x1c]
    push      edx
    mov       edx,dword ptr [eax+0x1c]
    xor       ecx,edx
    push      ecx
    call      _csc_transP
    mov       edx,dword ptr [esp+0x78]
    mov       eax,dword ptr [esp+0x5c]
    add       esp,0x00000040
    mov       ecx,dword ptr [edx+eax]
    mov       edx,dword ptr [eax+0x20]
    xor       ecx,edx
    mov       edx,dword ptr [esi-0x120]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    xor       ecx,dword ptr [edx]
    mov       edx,dword ptr [esp+0x34]
    not       ecx
    and       edx,ecx
    mov       dword ptr [esp+0x34],edx
    lje        X$13
    mov       ecx,dword ptr [esp+0x40]
    mov       edx,dword ptr [ecx+eax]
    mov       ecx,dword ptr [esi-0x140]
    xor       edx,ecx
    mov       ecx,dword ptr [eax]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x20]
    xor       edx,dword ptr [ecx]
    mov       ecx,dword ptr [esp+0x34]
    not       edx
    and       ecx,edx
    mov       dword ptr [esp+0x34],ecx
    lje        X$13
    mov       ecx,dword ptr [esp+0x24]
    mov       edx,dword ptr [eax-0x20]
    xor       edx,dword ptr [ecx+0x20]
    mov       ecx,dword ptr [esi-0x160]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x14]
    xor       edx,dword ptr [ecx]
    mov       ecx,dword ptr [esp+0x34]
    not       edx
    and       ecx,edx
    mov       dword ptr [esp+0x34],ecx
    lje        X$13
    mov       edx,dword ptr [eax-0x40]
    mov       ecx,dword ptr [esi-0x180]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x24]
    xor       edx,dword ptr [ecx]
    mov       ecx,dword ptr [esp+0x18]
    xor       edx,dword ptr [ecx]
    mov       ecx,dword ptr [esp+0x34]
    not       edx
    and       ecx,edx
    mov       dword ptr [esp+0x34],ecx
    lje        X$13
    mov       ecx,dword ptr [esp+0x24]
    mov       edx,dword ptr [eax-0x60]
    xor       edx,dword ptr [ecx-0x20]
    mov       ecx,dword ptr [esi-0x1a0]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x2c]
    xor       edx,dword ptr [ecx]
    mov       ecx,dword ptr [esp+0x34]
    not       edx
    and       ecx,edx
    mov       dword ptr [esp+0x34],ecx
    lje        X$13
    mov       edx,dword ptr [esp+0x24]
    mov       ecx,dword ptr [edx-0x40]
    mov       edx,dword ptr [eax-0x80]
    xor       ecx,edx
    mov       edx,dword ptr [esi-0x1c0]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x94]
    xor       ecx,dword ptr [edx]
    mov       edx,dword ptr [esp+0x34]
    not       ecx
    and       edx,ecx
    mov       dword ptr [esp+0x34],edx
    lje        X$13
    mov       ecx,dword ptr [esp+0x24]
    mov       edx,dword ptr [ecx-0x60]
    mov       ecx,dword ptr [eax-0xa0]
    xor       edx,ecx
    mov       ecx,dword ptr [esi-0x1e0]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x8c]
    xor       edx,dword ptr [ecx]
    mov       ecx,dword ptr [esp+0x34]
    not       edx
    and       ecx,edx
    mov       dword ptr [esp+0x34],ecx
    je        X$13
    mov       edx,dword ptr [esp+0x24]
    mov       ecx,dword ptr [edx-0x80]
    mov       edx,dword ptr [eax-0xc0]
    xor       ecx,edx
    mov       edx,dword ptr [esi-0x200]
    xor       ecx,edx
    mov       edx,dword ptr [esi]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x34]
    not       ecx
    and       edx,ecx
    mov       dword ptr [esp+0x34],edx
    je        X$13
    mov       ecx,dword ptr [esp+0x10]
    mov       edx,dword ptr [esp+0x24]
    inc       ecx
    add       edi,0x00000020
    mov       dword ptr [esp+0x10],ecx
    mov       ecx,0x00000004
    add       eax,ecx
    mov       dword ptr [esp+0x28],edi
    mov       edi,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x1c],eax
    mov       eax,dword ptr [esp+0x20]
    add       edx,ecx
    add       edi,0x00000020
    add       esi,ecx
    add       eax,ecx
    mov       dword ptr [esp+0x24],edx
    mov       dword ptr [esp+0x78],edi
    mov       dword ptr [esp+0x20],eax
    jmp       X$12
X$13:
    mov       eax,dword ptr [esp+0x1dc]
    inc       eax
    test      al,0x01
    mov       dword ptr [esp+0x1dc],eax
    je        X$15
    mov       ecx,dword ptr [esp+0x204]
    mov       eax,dword ptr [ecx]
    test      eax,eax
    lje        X$8
X$14:
    mov       edx,dword ptr [eax]
    add       ecx,0x00000004
    not       edx
    mov       dword ptr [eax],edx
    mov       eax,dword ptr [ecx]
    test      eax,eax
    jne       X$14
    jmp       X$8
X$15:
    test      al,0x02
    je        X$17
    mov       eax,dword ptr [esp+0x204]
    lea       ecx,[eax+0x74]
    mov       eax,dword ptr [eax+0x74]
    test      eax,eax
    lje        X$8
X$16:
    mov       edx,dword ptr [eax]
    add       ecx,0x00000004
    not       edx
    mov       dword ptr [eax],edx
    mov       eax,dword ptr [ecx]
    test      eax,eax
    jne       X$16
    jmp       X$8
X$17:
    test      al,0x04
    je        X$19
    mov       ecx,dword ptr [esp+0x1ec]
    mov       eax,dword ptr [ecx]
    test      eax,eax
    lje        X$8
X$18:
    mov       edx,dword ptr [eax]
    add       ecx,0x00000004
    not       edx
    mov       dword ptr [eax],edx
    mov       eax,dword ptr [ecx]
    test      eax,eax
    jne       X$18
    jmp       X$8
X$19:
    test      al,0x08
    je        X$21
    mov       eax,dword ptr [esp+0x204]
    lea       ecx,[eax+0x15c]
    mov       eax,dword ptr [eax+0x15c]
    test      eax,eax
    lje        X$8
X$20:
    mov       edx,dword ptr [eax]
    add       ecx,0x00000004
    not       edx
    mov       dword ptr [eax],edx
    mov       eax,dword ptr [ecx]
    test      eax,eax
    jne       X$20
    jmp       X$8
X$21:
    test      al,0x10
    je        X$23
    mov       eax,dword ptr [esp+0x204]
    lea       ecx,[eax+0x1d0]
    mov       eax,dword ptr [eax+0x1d0]
    test      eax,eax
    lje        X$8
X$22:
    mov       edx,dword ptr [eax]
    add       ecx,0x00000004
    not       edx
    mov       dword ptr [eax],edx
    mov       eax,dword ptr [ecx]
    test      eax,eax
    jne       X$22
    jmp       X$8
X$23:
    test      al,0x20
    je        X$25
    mov       eax,dword ptr [esp+0x204]
    lea       ecx,[eax+0x244]
    mov       eax,dword ptr [eax+0x244]
    test      eax,eax
    lje        X$8
X$24:
    mov       edx,dword ptr [eax]
    add       ecx,0x00000004
    not       edx
    mov       dword ptr [eax],edx
    mov       eax,dword ptr [ecx]
    test      eax,eax
    jne       X$24
    jmp       X$8
X$25:
    xor       eax,eax
    pop       edi
    pop       esi
    pop       ebp
    pop       ebx
    add       esp,0x000001e0
    ret       
X$26:
    mov       esi,dword ptr [esp+0x1e8]
    mov       edi,dword ptr [esp+0x1f4]
    mov       eax,dword ptr [esp+0x34]
    mov       ecx,0x00000040
    repe movsd 
    pop       edi
    pop       esi
    pop       ebp
    pop       ebx
    add       esp,0x000001e0
    ret       

__CODESECT__
    align 16
_csc_transF:
    mov       edx,dword ptr [esp+0x4]
    mov       eax,dword ptr [esp+0x20]
    mov       ecx,edx
    push      ebx
    mov       ebx,dword ptr [eax]
    push      esi
    mov       esi,dword ptr [esp+0x18]
    push      edi
    not       ecx
    mov       edi,ecx
    or        edi,esi
    xor       ebx,edi
    mov       edi,dword ptr [esp+0x14]
    mov       dword ptr [eax],ebx
    mov       eax,edi
    or        eax,edx
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x20]
    mov       edx,dword ptr [ecx]
    xor       edx,eax
    mov       dword ptr [ecx],edx
    mov       ecx,dword ptr [esp+0x18]
    mov       edx,dword ptr [esp+0x24]
    xor       eax,ecx
    or        eax,edi
    mov       edi,dword ptr [edx]
    xor       edi,eax
    xor       eax,esi
    or        eax,ecx
    mov       ecx,dword ptr [esp+0x28]
    mov       dword ptr [edx],edi
    pop       edi
    mov       edx,dword ptr [ecx]
    pop       esi
    xor       edx,eax
    pop       ebx
    mov       dword ptr [ecx],edx
    ret       

__CODESECT__
    align 16
_csc_transG:
    mov       ecx,dword ptr [esp+0x10]
    mov       eax,dword ptr [esp+0x4]
    push      ebx
    mov       ebx,dword ptr [esp+0x10]
    push      ebp
    push      esi
    mov       edx,ecx
    mov       esi,ecx
    or        edx,eax
    and       esi,ebx
    xor       edx,esi
    mov       esi,dword ptr [esp+0x24]
    push      edi
    mov       ebp,dword ptr [esi]
    xor       ebp,edx
    mov       dword ptr [esi],ebp
    mov       ebp,dword ptr [esp+0x18]
    mov       esi,ecx
    or        esi,ebp
    xor       esi,eax
    mov       eax,dword ptr [esp+0x30]
    mov       edi,esi
    or        edi,edx
    xor       edi,ecx
    xor       dword ptr [eax],edi
    mov       eax,ecx
    and       eax,ebp
    or        ecx,ebx
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x24]
    mov       ebx,eax
    or        eax,edx
    mov       ebp,dword ptr [ecx]
    xor       eax,edi
    not       ebx
    xor       ebp,ebx
    xor       eax,esi
    mov       dword ptr [ecx],ebp
    mov       ecx,dword ptr [esp+0x2c]
    not       eax
    mov       edx,dword ptr [ecx]
    pop       edi
    xor       edx,eax
    pop       esi
    pop       ebp
    mov       dword ptr [ecx],edx
    pop       ebx
    ret       

__CODESECT__
    align 16
_csc_transP:
    push      ebx
    push      ebp
    push      esi
    mov       ebx,dword ptr [esp+0x2c]
    push      edi
    mov       edi,dword ptr [esp+0x24]
    mov       ecx,dword ptr [esp+0x20]
    mov       esi,dword ptr [esp+0x28]
    mov       edx,edi
    mov       ebp,dword ptr [esp+0x18]
    not       edx
    mov       eax,edx
    or        eax,ebx
    xor       ecx,eax
    mov       eax,esi
    or        eax,edi
    xor       eax,edx
    mov       edx,dword ptr [esp+0x14]
    xor       edx,eax
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0x2c]
    xor       eax,edx
    or        eax,esi
    xor       ebp,eax
    xor       eax,ebx
    or        eax,edx
    mov       edx,dword ptr [esp+0x1c]
    xor       edx,eax
    mov       dword ptr [esp+0x18],ebp
    mov       dword ptr [esp+0x1c],edx
    mov       edx,ecx
    mov       eax,dword ptr [esp+0x1c]
    and       edx,eax
    mov       eax,ecx
    or        eax,dword ptr [esp+0x14]
    xor       edx,eax
    mov       eax,ecx
    xor       esi,edx
    or        eax,ebp
    mov       dword ptr [esp+0x28],esi
    mov       esi,dword ptr [esp+0x14]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x30],eax
    or        eax,edx
    xor       eax,ecx
    mov       dword ptr [esp+0x24],eax
    xor       ebx,eax
    mov       eax,ecx
    or        eax,esi
    mov       esi,ecx
    and       esi,ebp
    mov       ebp,dword ptr [esp+0x30]
    xor       eax,esi
    mov       esi,eax
    or        eax,edx
    mov       edx,dword ptr [esp+0x24]
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x28]
    xor       eax,edx
    mov       edx,edi
    xor       eax,ebp
    mov       ebp,dword ptr [esp+0x2c]
    not       edx
    or        esi,edi
    not       eax
    xor       esi,edx
    xor       ebp,eax
    mov       eax,esi
    mov       dword ptr [esp+0x2c],ebp
    xor       eax,ebp
    mov       ebp,dword ptr [esp+0x28]
    or        eax,ebp
    mov       ebp,dword ptr [esp+0x14]
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0x34]
    mov       dword ptr [ebp],esi
    mov       ebp,dword ptr [esp+0x18]
    mov       esi,eax
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0x38]
    xor       eax,ebx
    or        edx,ebx
    mov       dword ptr [ebp],esi
    mov       esi,dword ptr [esp+0x2c]
    mov       ebp,dword ptr [esp+0x1c]
    or        eax,esi
    xor       eax,ebp
    mov       ebp,dword ptr [esp+0x3c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    mov       dword ptr [ebp],eax
    mov       eax,dword ptr [esp+0x48]
    mov       dword ptr [ecx],edx
    mov       edx,dword ptr [esp+0x44]
    mov       ecx,dword ptr [esp+0x28]
    mov       dword ptr [edx],edi
    mov       edx,dword ptr [esp+0x4c]
    mov       dword ptr [eax],ecx
    mov       eax,dword ptr [esp+0x50]
    mov       dword ptr [edx],esi
    pop       edi
    pop       esi
    mov       dword ptr [eax],ebx
    pop       ebp
    pop       ebx
    ret       

__CODESECT__
    align 16
csc_unit_func_6b_i:
_csc_unit_func_6b_i:
    mov       eax,dword ptr [esp+0xc]
    sub       esp,0x00000028
    test      al,0x0f
    push      ebx
    push      ebp
    push      esi
    push      edi
    je        X$27
    add       eax,0x0000000f
    and       al,0xf0
X$27:
    mov       edi,dword ptr [esp+0x3c]
    mov       esi,eax
    add       eax,0x00000200
    lea       edx,[esp+0x44]
    mov       ebp,eax
    add       eax,0x00000100
    mov       ebx,eax
    mov       ecx,dword ptr [edi+0x14]
    add       eax,0x00000100
    push      edx
    mov       dword ptr [esp+0x2c],eax
    mov       eax,dword ptr [edi+0x10]
    mov       dword ptr [esp+0x14],eax
    lea       eax,[esp+0x14]
    push      eax
    mov       dword ptr [esp+0x34],ebp
    mov       dword ptr [esp+0x24],ebx
    mov       dword ptr [esp+0x4c],ecx
    call      convert_key_from_inc_to_csc
    mov       ecx,dword ptr [esp+0x4c]
    mov       edx,dword ptr [edi+0x4]
    mov       edi,dword ptr [edi+0xc]
    sub       ebp,ebx
    mov       dword ptr [esp+0x28],ebp
    mov       ebp,esi
    add       esp,0x00000008
    sub       ebp,ebx
    mov       dword ptr [esp+0x14],ecx
    mov       eax,0x00000001
    mov       ecx,ebx
    mov       dword ptr [esp+0x24],ebp
    mov       dword ptr [esp+0x18],0x00000040
X$28:
    mov       ebp,dword ptr [esp+0x20]
    mov       ebx,eax
    and       ebx,edx
    neg       ebx
    sbb       ebx,ebx
    mov       dword ptr [ebp+ecx],ebx
    mov       ebx,eax
    mov       ebp,dword ptr [esp+0x14]
    and       ebx,edi
    neg       ebx
    sbb       ebx,ebx
    mov       dword ptr [ecx],ebx
    mov       ebx,eax
    and       ebx,ebp
    mov       ebp,dword ptr [esp+0x24]
    neg       ebx
    sbb       ebx,ebx
    shl       eax,0x00000001
    mov       dword ptr [ebp+ecx],ebx
    jne       X$29
    mov       eax,dword ptr [esp+0x3c]
    mov       edx,dword ptr [eax]
    mov       edi,dword ptr [eax+0x8]
    mov       eax,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x14],eax
    mov       eax,0x00000001
X$29:
    mov       ebx,dword ptr [esp+0x18]
    add       ecx,0x00000004
    dec       ebx
    mov       dword ptr [esp+0x18],ebx
    jne       X$28
    mov       ecx,0x00000040
    xor       eax,eax
    lea       edi,[esi+0x100]
    mov       byte ptr [esp+0x37],0x00
    repe stosd 
    mov       eax,dword ptr [esp+0x10]
    mov       ebp,0x0000000b
    mov       ecx,eax
    mov       edx,eax
    shr       ecx,0x00000018
    mov       byte ptr [esp+0x30],cl
    mov       ecx,eax
    shr       edx,0x00000010
    mov       byte ptr [esp+0x33],al
    mov       eax,dword ptr [esp+0x44]
    mov       byte ptr [esp+0x31],dl
    mov       edx,eax
    shr       ecx,0x00000008
    mov       byte ptr [esp+0x32],cl
    mov       ecx,eax
    shr       edx,0x00000018
    mov       byte ptr [esp+0x34],dl
    mov       edx,dword ptr [csc_bit_order+0x18]
    shr       ecx,0x00000010
    shr       eax,0x00000008
    mov       byte ptr [esp+0x35],cl
    mov       byte ptr [esp+0x36],al
    mov       dword ptr [esi+edx*4],0xaaaaaaaa
    mov       eax,dword ptr [csc_bit_order+0x1c]
    mov       dword ptr [esp+0x18],ebp
    mov       dword ptr [esi+eax*4],0xcccccccc
    mov       ecx,dword ptr [csc_bit_order+0x20]
    mov       dword ptr [esi+ecx*4],0xf0f0f0f0
    mov       edx,dword ptr [csc_bit_order+0x24]
    mov       ecx,dword ptr [esp+0x40]
    mov       dword ptr [esi+edx*4],0xff00ff00
    mov       eax,dword ptr [csc_bit_order+0x28]
    mov       dword ptr [esi+eax*4],0xffff0000
    mov       eax,dword ptr [ecx]
    cmp       eax,0x00000800
    jbe       X$31
X$30:
    inc       ebp
    mov       edx,0x00000001
    mov       ecx,ebp
    shl       edx,cl
    cmp       eax,edx
    ja        X$30
    mov       dword ptr [esp+0x18],ebp
X$31:
    mov       ebx,csc_bit_order
X$32:
    mov       ecx,dword ptr [ebx]
    lea       edi,[esp+0x37]
    mov       eax,ecx
    add       ebx,0x00000004
    cdq       
    and       edx,0x00000007
    mov       dword ptr [esi+ecx*4],0x00000000
    add       eax,edx
    sar       eax,0x00000003
    sub       edi,eax
    mov       eax,ecx
    cdq       
    xor       eax,edx
    sub       eax,edx
    and       eax,0x00000007
    xor       eax,edx
    mov       ecx,eax
    mov       al,0x01
    sub       ecx,edx
    mov       dl,byte ptr [edi]
    shl       al,cl
    not       al
    and       dl,al
    cmp       ebx,csc_bit_order+0x18
    mov       byte ptr [edi],dl
    jl        X$32
    add       ebp,0xfffffff5
    test      ebp,ebp
    mov       dword ptr [esp+0x24],ebp
    jbe       X$34
    mov       ebx,csc_bit_order+0x2c
X$33:
    mov       ecx,dword ptr [ebx]
    lea       edi,[esp+0x37]
    mov       eax,ecx
    add       ebx,0x00000004
    cdq       
    and       edx,0x00000007
    mov       dword ptr [esi+ecx*4],0x00000000
    add       eax,edx
    sar       eax,0x00000003
    sub       edi,eax
    mov       eax,ecx
    cdq       
    xor       eax,edx
    sub       eax,edx
    and       eax,0x00000007
    xor       eax,edx
    mov       ecx,eax
    mov       al,byte ptr [edi]
    sub       ecx,edx
    mov       dl,0x01
    shl       dl,cl
    not       dl
    and       al,dl
    dec       ebp
    mov       byte ptr [edi],al
    jne       X$33
X$34:
    mov       edi,dword ptr [esp+0x28]
    mov       eax,dword ptr [esp+0x1c]
    mov       ebp,dword ptr [esp+0x2c]
    push      edi
    push      eax
    lea       ecx,[esp+0x38]
    push      ebp
    push      ecx
    push      esi
    xor       ebx,ebx
    call      cscipher_bitslicer_6b_i
    add       esp,0x00000014
    test      eax,eax
    jne       X$39
    mov       ecx,dword ptr [esp+0x24]
    mov       edx,0x00000001
    shl       edx,cl
    mov       dword ptr [esp+0x2c],edx
    jmp       X$36
X$35:
    mov       edx,dword ptr [esp+0x2c]
X$36:
    inc       ebx
    cmp       ebx,edx
    jae       X$39
    xor       ecx,ecx
    test      bl,0x01
    jne       X$38
X$37:
    inc       ecx
    mov       edx,0x00000001
    shl       edx,cl
    test      ebx,edx
    je        X$37
X$38:
    mov       ecx,dword ptr [ecx*4+csc_bit_order+0x2c]
    push      edi
    mov       edx,ecx
    mov       eax,dword ptr [esi+ecx*4]
    not       eax
    mov       dword ptr [esi+ecx*4],eax
    lea       eax,[esp+0x3b]
    shr       edx,0x00000003
    sub       eax,edx
    and       ecx,0x00000007
    mov       dl,0x01
    shl       dl,cl
    mov       cl,byte ptr [eax]
    xor       cl,dl
    mov       byte ptr [eax],cl
    mov       eax,dword ptr [esp+0x20]
    push      eax
    lea       ecx,[esp+0x38]
    push      ebp
    push      ecx
    push      esi
    call      cscipher_bitslicer_6b_i
    add       esp,0x00000014
    test      eax,eax
    je        X$35
X$39:
    xor       ecx,ecx
    cmp       eax,ecx
    lje        X$46
    xor       edx,edx
    cmp       eax,0x00000001
    je        X$41
X$40:
    shr       eax,0x00000001
    inc       edx
    cmp       eax,0x00000001
    jne       X$40
X$41:
    mov       dword ptr [esp+0x10],ecx
    mov       dword ptr [esp+0x44],ecx
    mov       eax,0x00000008
    add       esi,0x00000020
X$42:
    mov       edi,dword ptr [esi]
    cmp       eax,0x00000020
    mov       ecx,edx
    jge       X$43
    shr       edi,cl
    mov       ecx,eax
    and       edi,0x00000001
    shl       edi,cl
    or        dword ptr [esp+0x44],edi
    jmp       X$44
X$43:
    shr       edi,cl
    lea       ecx,[eax-0x20]
    and       edi,0x00000001
    shl       edi,cl
    or        dword ptr [esp+0x10],edi
X$44:
    inc       eax
    add       esi,0x00000004
    cmp       eax,0x00000040
    jl        X$42
    lea       edx,[esp+0x44]
    lea       eax,[esp+0x10]
    push      edx
    push      eax
    call      convert_key_from_csc_to_inc
    mov       edx,dword ptr [esp+0x44]
    mov       ecx,dword ptr [esp+0x4c]
    add       esp,0x00000008
    mov       eax,dword ptr [edx+0x14]
    cmp       ecx,eax
    jae       X$45
    mov       esi,dword ptr [esp+0x40]
    sub       eax,ecx
    mov       dword ptr [esi],eax
    mov       dword ptr [edx+0x14],ecx
    mov       ecx,dword ptr [esp+0x10]
    mov       eax,0x00000002
    mov       dword ptr [edx+0x10],ecx
    pop       edi
    pop       esi
    pop       ebp
    pop       ebx
    add       esp,0x00000028
    ret       
X$45:
    mov       esi,ecx
    sub       esi,eax
    mov       eax,dword ptr [esp+0x40]
    mov       dword ptr [eax],esi
    mov       dword ptr [edx+0x14],ecx
    mov       ecx,dword ptr [esp+0x10]
    mov       eax,0x00000002
    mov       dword ptr [edx+0x10],ecx
    pop       edi
    pop       esi
    pop       ebp
    pop       ebx
    add       esp,0x00000028
    ret       
X$46:
    mov       ecx,dword ptr [esp+0x18]
    mov       edx,dword ptr [esp+0x40]
    mov       eax,0x00000001
    shl       eax,cl
    mov       ecx,dword ptr [esp+0x3c]
    mov       dword ptr [edx],eax
    mov       esi,dword ptr [ecx+0x14]
    add       esi,eax
    mov       edx,esi
    mov       dword ptr [ecx+0x14],esi
    cmp       edx,eax
    jae       X$47
    inc       dword ptr [ecx+0x10]
X$47:
    pop       edi
    pop       esi
    pop       ebp
    mov       eax,0x00000001
    pop       ebx
    add       esp,0x00000028
    ret       

__CODESECT__
    align 16

