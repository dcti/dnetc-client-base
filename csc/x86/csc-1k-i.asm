; Copyright distributed.net 1997 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; $Log: csc-1k-i.asm,v $
; Revision 1.1.2.2  1999/11/07 01:31:16  remi
; Increased code alignment.
;
; Revision 1.1.2.1  1999/11/06 00:26:13  cyp
; they're here! (see also bench.res for 'ideal' combination)
;
;

global          csc_unit_func_1k_i,_csc_unit_func_1k_i

extern          csc_tabc,csc_tabe,csc_bit_order
extern          convert_key_from_inc_to_csc,convert_key_from_csc_to_inc

%include "csc-mac.inc"

__DATASECT__
    db  "@(#)$Id: csc-1k-i.asm,v 1.1.2.2 1999/11/07 01:31:16 remi Exp $",0

__CODESECT__
    align 32
cscipher_bitslicer_1k_i:
    sub       esp,0x000002c0
    mov       eax,dword ptr [esp+0x2c4]
    push      ebx
    mov       ebx,dword ptr [esp+0x2d4]
    push      ebp
    push      esi
    push      edi
    lea       esi,[eax+0x100]
    mov       ecx,0x00000040
    mov       edi,ebx
    lea       edx,[ebx+0x100]
    repe movsd 
    mov       ecx,0x00000040
    mov       esi,eax
    mov       edi,edx
    mov       ebp,offset csc_tabc
    repe movsd 
    mov       esi,dword ptr [esp+0x2d8]
    mov       ecx,0x00000040
    lea       edi,[esp+0x1a4]
    mov       dword ptr [esp+0x10c],ebp
    add       ebx,0x00000200
    mov       dword ptr [esp+0x128],edx
    repe movsd 
    mov       dword ptr [esp+0x184],0x00000008
    mov       dword ptr [esp+0x7c],0x00000008
    jmp       X$3
X$1:
    mov       ebp,dword ptr [esp+0x10c]
    mov       edx,dword ptr [esp+0x128]
    mov       dword ptr [esp+0x7c],0x00000008
    jmp       X$3
X$2:
    mov       edx,dword ptr [esp+0x128]
X$3:
    mov       edi,dword ptr [ebp]
    mov       eax,dword ptr [edx]
    mov       esi,dword ptr [ebp+0x4]
    mov       ecx,dword ptr [ebp+0x8]
    xor       edi,eax
    mov       eax,dword ptr [edx+0x4]
    xor       eax,esi
    mov       esi,dword ptr [edx+0x8]
    mov       dword ptr [esp+0x80],eax
    mov       eax,dword ptr [ebp+0xc]
    xor       esi,ecx
    mov       ecx,dword ptr [edx+0xc]
    xor       ecx,eax
    mov       dword ptr [esp+0xa4],esi
    mov       eax,ecx
    mov       dword ptr [esp+0x9c],ecx
    not       eax
    mov       dword ptr [esp+0x24],eax
    mov       eax,edi
    or        eax,dword ptr [esp+0x24]
    or        esi,ecx
    mov       ecx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x84],edi
    xor       eax,dword ptr [edx+0x10]
    mov       edx,dword ptr [edx+0x1c]
    xor       esi,ecx
    mov       ecx,dword ptr [ebp+0x1c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x80]
    xor       edx,esi
    xor       ecx,esi
    mov       esi,dword ptr [esp+0xa4]
    xor       eax,dword ptr [ebp+0x10]
    or        ecx,esi
    mov       esi,dword ptr [esp+0x128]
    xor       edi,ecx
    mov       esi,dword ptr [esi+0x18]
    xor       esi,dword ptr [ebp+0x18]
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x80]
    or        edi,ecx
    mov       ecx,dword ptr [esp+0x128]
    xor       edi,dword ptr [ecx+0x14]
    mov       ecx,dword ptr [ebp+0x14]
    mov       ebp,eax
    xor       edi,ecx
    mov       ecx,eax
    and       ecx,edi
    or        ebp,edx
    xor       ecx,ebp
    mov       ebp,dword ptr [esp+0xa4]
    xor       ebp,ecx
    mov       dword ptr [esp+0x10],ecx
    mov       dword ptr [esp+0xa4],ebp
    mov       ebp,eax
    or        ebp,esi
    mov       dword ptr [esp+0xe0],edi
    xor       ebp,edx
    mov       dword ptr [esp+0x3c],ebp
    or        ebp,ecx
    mov       ecx,dword ptr [esp+0x84]
    xor       ebp,eax
    xor       ecx,ebp
    mov       dword ptr [esp+0x20],ebp
    mov       dword ptr [esp+0x84],ecx
    mov       ecx,eax
    mov       ebp,eax
    and       ecx,esi
    or        ebp,edi
    mov       edi,dword ptr [esp+0x9c]
    xor       ecx,ebp
    mov       ebp,ecx
    not       ebp
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0x10]
    or        ecx,ebp
    mov       ebp,dword ptr [esp+0x20]
    xor       ecx,ebp
    mov       ebp,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x9c],edi
    xor       ecx,ebp
    mov       ebp,dword ptr [esp+0x80]
    add       ebx,0x00000004
    not       ecx
    xor       ebp,ecx
    mov       ecx,dword ptr [esp+0xa4]
    mov       dword ptr [esp+0x80],ebp
    mov       ebp,edi
    not       ebp
    or        ecx,edi
    mov       edi,dword ptr [esp+0x80]
    xor       ecx,ebp
    xor       edi,ecx
    xor       edx,ecx
    or        edi,dword ptr [esp+0xa4]
    mov       ecx,dword ptr [esp+0x84]
    mov       dword ptr [ebx+0xdc],edx
    mov       edx,dword ptr [esp+0x80]
    xor       esi,edi
    mov       dword ptr [ebx+0x1c],edx
    mov       dword ptr [ebx+0xbc],esi
    mov       esi,ecx
    xor       esi,edi
    mov       edi,dword ptr [esp+0xe0]
    or        esi,edx
    mov       edx,dword ptr [esp+0x128]
    xor       esi,edi
    mov       dword ptr [ebx-0x4],ecx
    mov       dword ptr [ebx+0x9c],esi
    mov       esi,ecx
    or        esi,ebp
    mov       ebp,dword ptr [esp+0x10c]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x9c]
    mov       dword ptr [ebx+0x5c],eax
    mov       eax,dword ptr [esp+0xa4]
    mov       dword ptr [ebx+0x3c],eax
    mov       eax,dword ptr [esp+0x7c]
    mov       dword ptr [ebx+0x7c],esi
    add       ebp,0x00000020
    add       edx,0x00000020
    dec       eax
    mov       dword ptr [esp+0x10c],ebp
    mov       dword ptr [esp+0x128],edx
    mov       dword ptr [esp+0x7c],eax
    ljne       X$2
    mov       ecx,dword ptr [ebx-0x200]
    mov       edx,dword ptr [ebx]
    sub       ebx,0x00000020
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x1c4]
    mov       eax,edx
    mov       ebp,dword ptr [esp+0x1c8]
    xor       eax,ecx
    mov       dword ptr [ebx+0x20],edx
    mov       edx,dword ptr [ebx-0x1dc]
    mov       ecx,eax
    mov       eax,dword ptr [ebx+0x24]
    mov       edi,dword ptr [ebx+0x28]
    xor       eax,edx
    mov       dword ptr [ebx+0x24],eax
    mov       edx,dword ptr [esp+0x1cc]
    xor       eax,ebp
    mov       ebp,dword ptr [esp+0x1d0]
    mov       esi,eax
    mov       eax,dword ptr [ebx-0x1d8]
    xor       edi,eax
    mov       dword ptr [esp+0x4c],ecx
    mov       eax,edi
    mov       dword ptr [ebx+0x28],edi
    xor       eax,edx
    mov       edx,dword ptr [ebx-0x1d4]
    mov       edi,eax
    mov       eax,dword ptr [ebx+0x2c]
    xor       eax,edx
    mov       edx,dword ptr [ebx+0x30]
    mov       dword ptr [ebx+0x2c],eax
    xor       eax,ebp
    mov       ebp,eax
    mov       eax,dword ptr [ebx-0x1d0]
    xor       edx,eax
    mov       dword ptr [esp+0x40],edi
    mov       dword ptr [ebx+0x30],edx
    mov       eax,edx
    xor       eax,dword ptr [esp+0x1d4]
    mov       edx,eax
    mov       eax,dword ptr [ebx-0x1cc]
    xor       dword ptr [ebx+0x34],eax
    mov       eax,dword ptr [ebx+0x34]
    xor       eax,dword ptr [esp+0x1d8]
    mov       dword ptr [esp+0x44],edx
    mov       dword ptr [esp+0x34],eax
    mov       eax,dword ptr [ebx-0x1c8]
    xor       dword ptr [ebx+0x38],eax
    mov       eax,dword ptr [ebx+0x38]
    xor       eax,dword ptr [esp+0x1dc]
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [ebx-0x1c4]
    xor       dword ptr [ebx+0x3c],eax
    mov       eax,dword ptr [ebx+0x3c]
    xor       eax,dword ptr [esp+0x1e0]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [ebx-0x200]
    xor       dword ptr [ebx],eax
    mov       eax,dword ptr [ebx]
    xor       eax,dword ptr [esp+0x1a4]
    xor       eax,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x14],eax
    xor       eax,ecx
    mov       ecx,dword ptr [ebx-0x1fc]
    xor       dword ptr [ebx+0x4],ecx
    mov       ecx,dword ptr [ebx+0x4]
    xor       ecx,dword ptr [esp+0x1a8]
    mov       dword ptr [esp+0x6c],ecx
    xor       ecx,esi
    mov       dword ptr [esp+0xfc],ecx
    mov       ecx,dword ptr [ebx-0x1f8]
    xor       dword ptr [ebx+0x8],ecx
    mov       ecx,dword ptr [ebx+0x8]
    xor       ecx,dword ptr [esp+0x1ac]
    xor       ecx,esi
    mov       esi,ecx
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [ebx-0x1f4]
    xor       esi,edi
    mov       edi,dword ptr [ebx+0xc]
    xor       edi,ecx
    mov       dword ptr [ebx+0xc],edi
    mov       ecx,edi
    xor       ecx,dword ptr [esp+0x1b0]
    mov       dword ptr [esp+0x38],ecx
    mov       edi,ecx
    mov       ecx,dword ptr [ebx-0x1f0]
    xor       edi,ebp
    xor       dword ptr [ebx+0x10],ecx
    mov       ecx,dword ptr [ebx+0x10]
    xor       ecx,dword ptr [esp+0x1b4]
    xor       ecx,ebp
    mov       ebp,dword ptr [ebx+0x14]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [ebx-0x1ec]
    xor       ebp,ecx
    mov       dword ptr [ebx+0x14],ebp
    mov       ecx,ebp
    mov       ebp,dword ptr [esp+0x1b8]
    xor       ecx,ebp
    mov       ebp,dword ptr [ebx+0x18]
    mov       dword ptr [esp+0x2c],ecx
    mov       ecx,dword ptr [ebx-0x1e8]
    xor       ebp,ecx
    mov       dword ptr [ebx+0x18],ebp
    mov       ecx,ebp
    mov       ebp,dword ptr [esp+0x1bc]
    xor       ecx,ebp
    mov       ebp,dword ptr [esp+0x34]
    xor       ecx,ebp
    mov       ebp,dword ptr [ebx+0x1c]
    mov       dword ptr [esp+0x48],ecx
    mov       ecx,dword ptr [ebx-0x1e4]
    xor       ebp,ecx
    mov       dword ptr [ebx+0x1c],ebp
    mov       ecx,ebp
    mov       ebp,dword ptr [esp+0x1c0]
    xor       ecx,ebp
    mov       ebp,edi
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,eax
    not       ebp
    or        ecx,ebp
    xor       ecx,dword ptr [esp+0x18]
    xor       ecx,edx
    mov       edx,esi
    or        edx,edi
    mov       dword ptr [esp+0x2b4],ecx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x10],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x60]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x134],edx
    mov       edx,dword ptr [esp+0xfc]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x48]
    or        edx,esi
    mov       dword ptr [esp+0x10],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x30]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x124],edx
    mov       edx,eax
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xfc]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x2c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x34]
    xor       edx,ebp
    mov       ebp,ecx
    and       ebp,edx
    mov       dword ptr [esp+0xc4],edx
    mov       edx,dword ptr [esp+0x134]
    mov       dword ptr [esp+0x10],ebp
    mov       ebp,ecx
    or        ebp,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x124]
    mov       dword ptr [esp+0x24],edx
    xor       esi,edx
    mov       edx,ecx
    mov       dword ptr [esp+0x2b8],esi
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x134]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x20],edx
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xc4]
    xor       edx,ecx
    mov       dword ptr [esp+0x10],edx
    xor       eax,edx
    mov       edx,ecx
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x124]
    and       ecx,ebp
    mov       ebp,dword ptr [esp+0x10]
    xor       edx,ecx
    mov       ecx,edx
    not       ecx
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x24]
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x20]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xfc]
    xor       edx,ecx
    mov       ecx,esi
    not       edx
    xor       ebp,edx
    mov       edx,edi
    not       edx
    or        ecx,edi
    mov       dword ptr [esp+0x2ac],edx
    xor       ecx,edx
    mov       edx,ebp
    xor       edx,ecx
    mov       dword ptr [esp+0xfc],ebp
    mov       ebp,dword ptr [esp+0x38]
    or        edx,esi
    mov       esi,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x2a8],edi
    mov       edi,dword ptr [esp+0x4c]
    mov       dword ptr [esp+0xac],ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       ebp,esi
    mov       esi,ebp
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x180],edx
    not       esi
    mov       edx,ecx
    mov       ecx,esi
    or        ecx,edi
    mov       edi,dword ptr [esp+0x18]
    xor       ecx,edi
    mov       edi,ebp
    or        edi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x8c],edx
    mov       dword ptr [esp+0xbc],ecx
    xor       edi,esi
    mov       esi,edi
    xor       edx,edi
    mov       edi,dword ptr [esp+0x1c]
    xor       esi,dword ptr [esp+0x28]
    or        edx,edi
    xor       esi,dword ptr [esp+0x30]
    mov       edi,edx
    xor       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x158],esi
    mov       dword ptr [esp+0xd8],edi
    mov       edi,dword ptr [esp+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x8c]
    or        edx,edi
    mov       edi,dword ptr [esp+0x2c]
    add       ebx,0x00000040
    xor       edx,edi
    mov       edi,dword ptr [esp+0x44]
    xor       edx,edi
    mov       edi,ecx
    or        edi,esi
    mov       esi,ecx
    and       esi,edx
    mov       dword ptr [esp+0x164],edx
    xor       edi,esi
    mov       esi,dword ptr [esp+0x1c]
    mov       edx,edi
    xor       edx,esi
    mov       esi,dword ptr [esp+0xd8]
    mov       dword ptr [esp+0x104],edx
    mov       edx,ecx
    or        edx,esi
    mov       esi,dword ptr [esp+0x158]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x20],edx
    or        edx,edi
    xor       edx,ecx
    mov       dword ptr [esp+0x10],edx
    xor       edx,esi
    mov       dword ptr [esp+0xe4],edx
    mov       edx,dword ptr [esp+0xd8]
    mov       esi,ecx
    and       esi,edx
    mov       edx,dword ptr [esp+0x164]
    or        ecx,edx
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x20]
    mov       edx,esi
    or        esi,edi
    not       edx
    mov       edi,dword ptr [esp+0x8c]
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x64],ebp
    xor       esi,edx
    mov       edx,ebp
    xor       esi,ecx
    not       esi
    xor       edi,esi
    mov       esi,dword ptr [esp+0x104]
    mov       ecx,esi
    mov       dword ptr [esp+0x8c],edi
    not       edx
    or        ecx,ebp
    mov       ebp,dword ptr [ebx+0x20]
    mov       dword ptr [esp+0xa4],edx
    xor       ecx,edx
    mov       edx,edi
    mov       edi,dword ptr [esp+0x204]
    mov       dword ptr [esp+0xc0],ecx
    xor       edx,ecx
    mov       ecx,dword ptr [ebx-0x1e0]
    or        edx,esi
    mov       esi,dword ptr [ebx+0x24]
    xor       ebp,ecx
    mov       ecx,ebp
    mov       dword ptr [esp+0x150],edx
    xor       ecx,edi
    mov       edi,dword ptr [ebx+0x28]
    mov       edx,ecx
    mov       ecx,dword ptr [ebx-0x1dc]
    xor       esi,ecx
    mov       dword ptr [ebx+0x20],ebp
    mov       ebp,dword ptr [esp+0x208]
    mov       ecx,esi
    xor       ecx,ebp
    mov       ebp,dword ptr [esp+0x20c]
    mov       dword ptr [ebx+0x24],esi
    mov       esi,ecx
    mov       ecx,dword ptr [ebx-0x1d8]
    mov       dword ptr [esp+0x4c],edx
    xor       edi,ecx
    mov       dword ptr [ebx+0x28],edi
    mov       ecx,edi
    mov       edi,dword ptr [ebx+0x2c]
    xor       ecx,ebp
    mov       ebp,dword ptr [esp+0x210]
    mov       dword ptr [esp+0x40],ecx
    mov       ecx,dword ptr [ebx-0x1d4]
    xor       edi,ecx
    mov       ecx,edi
    mov       dword ptr [ebx+0x2c],edi
    mov       edi,dword ptr [ebx+0x30]
    xor       ecx,ebp
    mov       ebp,ecx
    mov       ecx,dword ptr [ebx-0x1d0]
    xor       edi,ecx
    mov       dword ptr [ebx+0x30],edi
    mov       ecx,edi
    mov       edi,dword ptr [esp+0x214]
    xor       ecx,edi
    mov       edi,dword ptr [ebx+0x34]
    mov       dword ptr [esp+0x44],ecx
    mov       ecx,dword ptr [ebx-0x1cc]
    xor       edi,ecx
    mov       dword ptr [ebx+0x34],edi
    mov       ecx,edi
    mov       edi,dword ptr [esp+0x218]
    xor       ecx,edi
    mov       edi,dword ptr [ebx+0x38]
    mov       dword ptr [esp+0x34],ecx
    mov       ecx,dword ptr [ebx-0x1c8]
    xor       edi,ecx
    mov       dword ptr [ebx+0x38],edi
    mov       ecx,edi
    mov       edi,dword ptr [esp+0x21c]
    xor       ecx,edi
    mov       edi,dword ptr [ebx+0x3c]
    mov       dword ptr [esp+0x30],ecx
    mov       ecx,dword ptr [ebx-0x1c4]
    xor       edi,ecx
    mov       dword ptr [ebx+0x3c],edi
    mov       ecx,edi
    xor       ecx,dword ptr [esp+0x220]
    mov       edi,ecx
    mov       ecx,dword ptr [ebx-0x200]
    xor       dword ptr [ebx],ecx
    mov       ecx,dword ptr [ebx]
    xor       ecx,dword ptr [esp+0x1e4]
    mov       dword ptr [esp+0x60],edi
    xor       ecx,edi
    mov       edi,dword ptr [ebx+0x4]
    mov       dword ptr [esp+0x14],ecx
    xor       ecx,edx
    mov       edx,dword ptr [ebx-0x1fc]
    xor       edi,edx
    mov       dword ptr [ebx+0x4],edi
    mov       edx,edi
    mov       edi,dword ptr [esp+0x1e8]
    xor       edx,edi
    mov       edi,dword ptr [ebx+0x8]
    mov       dword ptr [esp+0x6c],edx
    xor       edx,esi
    mov       dword ptr [esp+0xb8],edx
    mov       edx,dword ptr [ebx-0x1f8]
    xor       edi,edx
    mov       dword ptr [ebx+0x8],edi
    mov       edx,edi
    xor       edx,dword ptr [esp+0x1ec]
    xor       edx,esi
    mov       esi,dword ptr [ebx+0xc]
    mov       dword ptr [esp+0x1c],edx
    mov       edi,edx
    mov       edx,dword ptr [esp+0x40]
    xor       edi,edx
    mov       edx,dword ptr [ebx-0x1f4]
    xor       esi,edx
    mov       dword ptr [ebx+0xc],esi
    mov       edx,esi
    xor       edx,dword ptr [esp+0x1f0]
    mov       dword ptr [esp+0x38],edx
    mov       esi,edx
    mov       edx,dword ptr [ebx-0x1f0]
    xor       esi,ebp
    xor       dword ptr [ebx+0x10],edx
    mov       edx,dword ptr [ebx+0x10]
    xor       edx,dword ptr [esp+0x1f4]
    mov       dword ptr [esp+0x19c],esi
    xor       edx,ebp
    mov       ebp,dword ptr [ebx+0x14]
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [ebx-0x1ec]
    xor       ebp,edx
    mov       dword ptr [ebx+0x14],ebp
    mov       edx,ebp
    mov       ebp,dword ptr [esp+0x1f8]
    xor       edx,ebp
    mov       ebp,dword ptr [ebx+0x18]
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [ebx-0x1e8]
    xor       ebp,edx
    mov       dword ptr [ebx+0x18],ebp
    mov       edx,ebp
    mov       ebp,dword ptr [esp+0x1fc]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x34]
    xor       edx,ebp
    mov       ebp,dword ptr [ebx+0x1c]
    mov       dword ptr [esp+0x48],edx
    mov       edx,dword ptr [ebx-0x1e4]
    xor       ebp,edx
    mov       dword ptr [ebx+0x1c],ebp
    mov       edx,ebp
    mov       ebp,dword ptr [esp+0x200]
    xor       edx,ebp
    mov       ebp,esi
    mov       dword ptr [esp+0x28],edx
    mov       edx,ecx
    not       ebp
    mov       dword ptr [esp+0x10],ebp
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x44]
    xor       edx,ebp
    mov       ebp,edi
    or        ebp,esi
    mov       esi,dword ptr [esp+0x10]
    xor       ebp,esi
    mov       esi,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x10],ebp
    xor       ebp,esi
    mov       esi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x2c4],edx
    xor       ebp,esi
    mov       esi,dword ptr [esp+0xb8]
    mov       dword ptr [esp+0x148],ebp
    mov       ebp,dword ptr [esp+0x10]
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0x48]
    or        esi,edi
    mov       dword ptr [esp+0x10],esi
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0x30]
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x15c],esi
    mov       esi,ecx
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0xb8]
    or        esi,ebp
    mov       ebp,dword ptr [esp+0x2c]
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0x34]
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0x148]
    mov       dword ptr [esp+0x1a0],esi
    mov       esi,edx
    or        esi,ebp
    mov       ebp,edx
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0x1a0]
    and       ebp,esi
    mov       esi,dword ptr [esp+0x10]
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0x15c]
    mov       dword ptr [esp+0x24],esi
    xor       edi,esi
    mov       esi,edx
    mov       dword ptr [esp+0x2b0],edi
    or        esi,ebp
    mov       ebp,dword ptr [esp+0x148]
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x20],esi
    or        esi,ebp
    mov       ebp,dword ptr [esp+0x15c]
    xor       esi,edx
    mov       dword ptr [esp+0x10],esi
    xor       ecx,esi
    mov       esi,edx
    and       esi,ebp
    mov       ebp,dword ptr [esp+0x1a0]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x19c]
    xor       esi,edx
    mov       edx,esi
    not       edx
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x24]
    or        esi,edx
    mov       edx,dword ptr [esp+0x10]
    xor       esi,edx
    mov       edx,dword ptr [esp+0x20]
    xor       esi,edx
    mov       edx,dword ptr [esp+0xb8]
    not       esi
    xor       edx,esi
    mov       esi,ebp
    mov       dword ptr [esp+0xb8],edx
    mov       edx,edi
    not       esi
    or        edx,ebp
    mov       dword ptr [esp+0x2bc],esi
    xor       edx,esi
    mov       esi,dword ptr [esp+0xb8]
    mov       dword ptr [esp+0x19c],ebp
    mov       ebp,dword ptr [esp+0x4c]
    mov       dword ptr [esp+0x2a4],edx
    xor       esi,edx
    mov       edx,dword ptr [esp+0x6c]
    or        esi,edi
    mov       edi,dword ptr [esp+0x40]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x2c8],esi
    xor       ebp,edi
    mov       esi,edx
    mov       dword ptr [esp+0x9c],ebp
    mov       edi,ebp
    or        ebp,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x80],esi
    not       edi
    xor       ebp,edi
    mov       edx,edi
    mov       edi,ebp
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0x1c]
    or        edx,dword ptr [esp+0x14]
    or        esi,ebp
    xor       edi,dword ptr [esp+0x28]
    mov       ebp,esi
    xor       edx,dword ptr [esp+0x18]
    xor       ebp,dword ptr [esp+0x48]
    xor       edi,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x98],edx
    mov       dword ptr [esp+0xe8],ebp
    mov       ebp,dword ptr [esp+0x14]
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0x80]
    or        esi,ebp
    mov       ebp,dword ptr [esp+0x2c]
    mov       dword ptr [esp+0x94],edi
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0x44]
    add       ebx,0x00000040
    xor       esi,ebp
    mov       ebp,esi
    mov       esi,edx
    or        esi,edi
    mov       edi,edx
    and       edi,ebp
    mov       dword ptr [esp+0x160],ebp
    mov       ebp,dword ptr [esp+0x1c]
    xor       esi,edi
    mov       edi,esi
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0xe8]
    mov       dword ptr [esp+0x7c],edi
    mov       edi,edx
    or        edi,ebp
    mov       ebp,dword ptr [esp+0x94]
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x20],edi
    or        edi,esi
    xor       edi,edx
    mov       dword ptr [esp+0x10],edi
    xor       edi,ebp
    mov       dword ptr [esp+0x68],edi
    mov       edi,dword ptr [esp+0xe8]
    mov       ebp,edx
    and       ebp,edi
    mov       edi,dword ptr [esp+0x160]
    or        edx,edi
    mov       edi,dword ptr [esp+0x9c]
    xor       ebp,edx
    mov       edx,ebp
    or        ebp,esi
    mov       esi,dword ptr [esp+0x10]
    not       edx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x20]
    xor       ebp,esi
    mov       esi,edi
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x80]
    not       ebp
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x7c]
    mov       dword ptr [esp+0x80],edx
    mov       edx,ebp
    not       esi
    or        edx,edi
    mov       dword ptr [esp+0xe0],esi
    xor       edx,esi
    mov       esi,dword ptr [esp+0x80]
    mov       dword ptr [esp+0x9c],edi
    mov       edi,dword ptr [ebx+0x20]
    mov       dword ptr [esp+0x5c],edx
    xor       esi,edx
    mov       edx,dword ptr [ebx-0x1e0]
    or        esi,ebp
    mov       ebp,dword ptr [ebx+0x24]
    xor       edi,edx
    mov       dword ptr [esp+0x168],esi
    mov       esi,dword ptr [esp+0x244]
    mov       edx,edi
    mov       dword ptr [ebx+0x20],edi
    mov       edi,dword ptr [esp+0x248]
    xor       edx,esi
    mov       dword ptr [esp+0x4c],edx
    mov       edx,dword ptr [ebx-0x1dc]
    mov       esi,dword ptr [ebx+0x28]
    xor       ebp,edx
    mov       edx,ebp
    mov       dword ptr [ebx+0x24],ebp
    mov       ebp,dword ptr [esp+0x24c]
    xor       edx,edi
    mov       edi,edx
    mov       edx,dword ptr [ebx-0x1d8]
    xor       esi,edx
    mov       edx,esi
    mov       dword ptr [ebx+0x28],esi
    xor       edx,ebp
    mov       esi,dword ptr [ebx+0x2c]
    mov       ebp,dword ptr [esp+0x250]
    mov       dword ptr [esp+0x40],edx
    mov       edx,dword ptr [ebx-0x1d4]
    xor       esi,edx
    mov       edx,esi
    mov       dword ptr [ebx+0x2c],esi
    mov       esi,dword ptr [ebx+0x30]
    xor       edx,ebp
    mov       ebp,edx
    mov       edx,dword ptr [ebx-0x1d0]
    xor       esi,edx
    mov       dword ptr [ebx+0x30],esi
    mov       edx,esi
    mov       esi,dword ptr [esp+0x254]
    xor       edx,esi
    mov       esi,dword ptr [ebx+0x34]
    mov       dword ptr [esp+0x44],edx
    mov       edx,dword ptr [ebx-0x1cc]
    xor       esi,edx
    mov       dword ptr [ebx+0x34],esi
    mov       edx,esi
    mov       esi,dword ptr [esp+0x258]
    xor       edx,esi
    mov       esi,dword ptr [ebx+0x38]
    mov       dword ptr [esp+0x34],edx
    mov       edx,dword ptr [ebx-0x1c8]
    xor       esi,edx
    mov       dword ptr [ebx+0x38],esi
    mov       edx,esi
    mov       esi,dword ptr [esp+0x25c]
    xor       edx,esi
    mov       esi,dword ptr [ebx+0x3c]
    mov       dword ptr [esp+0x30],edx
    mov       edx,dword ptr [ebx-0x1c4]
    xor       esi,edx
    mov       dword ptr [ebx+0x3c],esi
    mov       edx,esi
    xor       edx,dword ptr [esp+0x260]
    mov       esi,edx
    mov       edx,dword ptr [ebx-0x200]
    xor       dword ptr [ebx],edx
    mov       edx,dword ptr [ebx]
    xor       edx,dword ptr [esp+0x224]
    mov       dword ptr [esp+0x60],esi
    xor       edx,esi
    mov       dword ptr [esp+0x14],edx
    mov       esi,edx
    mov       edx,dword ptr [esp+0x4c]
    xor       esi,edx
    mov       edx,dword ptr [ebx-0x1fc]
    xor       dword ptr [ebx+0x4],edx
    mov       edx,dword ptr [ebx+0x4]
    xor       edx,dword ptr [esp+0x228]
    mov       dword ptr [esp+0x6c],edx
    xor       edx,edi
    mov       dword ptr [esp+0xf0],edx
    mov       edx,dword ptr [ebx-0x1f8]
    xor       dword ptr [ebx+0x8],edx
    mov       edx,dword ptr [ebx+0x8]
    xor       edx,dword ptr [esp+0x22c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x1c],edx
    xor       edx,edi
    mov       edi,dword ptr [ebx+0xc]
    mov       dword ptr [esp+0xdc],edx
    mov       edx,dword ptr [ebx-0x1f4]
    xor       edi,edx
    mov       dword ptr [ebx+0xc],edi
    mov       edx,edi
    xor       edx,dword ptr [esp+0x230]
    mov       dword ptr [esp+0x38],edx
    mov       edi,edx
    mov       edx,dword ptr [ebx-0x1f0]
    xor       edi,ebp
    xor       dword ptr [ebx+0x10],edx
    mov       edx,dword ptr [ebx+0x10]
    mov       dword ptr [esp+0xc8],edi
    xor       edx,dword ptr [esp+0x234]
    xor       edx,ebp
    mov       ebp,dword ptr [ebx+0x14]
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [ebx-0x1ec]
    xor       ebp,edx
    mov       edx,ebp
    mov       dword ptr [ebx+0x14],ebp
    mov       ebp,dword ptr [esp+0x238]
    xor       edx,ebp
    mov       ebp,dword ptr [ebx+0x18]
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [ebx-0x1e8]
    xor       ebp,edx
    mov       dword ptr [ebx+0x18],ebp
    mov       edx,ebp
    mov       ebp,dword ptr [esp+0x23c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x34]
    xor       edx,ebp
    mov       ebp,dword ptr [ebx+0x1c]
    mov       dword ptr [esp+0x48],edx
    mov       edx,dword ptr [ebx-0x1e4]
    xor       ebp,edx
    mov       dword ptr [ebx+0x1c],ebp
    mov       edx,ebp
    mov       ebp,dword ptr [esp+0x240]
    xor       edx,ebp
    mov       ebp,edi
    mov       dword ptr [esp+0x28],edx
    mov       edx,esi
    not       ebp
    mov       dword ptr [esp+0x10],ebp
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x44]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xdc]
    or        ebp,edi
    mov       edi,dword ptr [esp+0x10]
    xor       ebp,edi
    mov       edi,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x10],ebp
    xor       ebp,edi
    mov       edi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x198],edx
    xor       ebp,edi
    mov       edi,dword ptr [esp+0xf0]
    mov       dword ptr [esp+0x140],ebp
    mov       ebp,dword ptr [esp+0x10]
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0xdc]
    or        edi,ebp
    mov       ebp,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x10],edi
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0x30]
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x13c],edi
    mov       edi,esi
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0xf0]
    or        edi,ebp
    mov       ebp,dword ptr [esp+0x2c]
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0x34]
    xor       edi,ebp
    mov       ebp,edx
    and       ebp,edi
    mov       dword ptr [esp+0x114],edi
    mov       edi,dword ptr [esp+0x140]
    mov       dword ptr [esp+0x10],ebp
    mov       ebp,edx
    or        ebp,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0xdc]
    mov       dword ptr [esp+0x24],edi
    xor       ebp,edi
    mov       dword ptr [esp+0xdc],ebp
    mov       ebp,dword ptr [esp+0x13c]
    mov       edi,edx
    or        edi,ebp
    mov       ebp,dword ptr [esp+0x140]
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x20],edi
    or        edi,ebp
    mov       ebp,dword ptr [esp+0x114]
    xor       edi,edx
    mov       dword ptr [esp+0x10],edi
    xor       esi,edi
    mov       edi,edx
    or        edi,ebp
    mov       ebp,dword ptr [esp+0x13c]
    and       edx,ebp
    xor       edi,edx
    mov       edx,dword ptr [esp+0xc8]
    mov       ebp,edi
    not       ebp
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x24]
    or        edi,ebp
    mov       ebp,dword ptr [esp+0x10]
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0x20]
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0xf0]
    not       edi
    xor       ebp,edi
    mov       dword ptr [esp+0xc8],edx
    mov       dword ptr [esp+0xf0],ebp
    mov       ebp,dword ptr [esp+0xdc]
    mov       edi,edx
    mov       edx,ebp
    or        edx,dword ptr [esp+0xc8]
    not       edi
    mov       dword ptr [esp+0x2cc],edi
    xor       edx,edi
    mov       edi,dword ptr [esp+0xf0]
    mov       dword ptr [esp+0x12c],edx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x6c]
    or        edi,ebp
    mov       ebp,dword ptr [esp+0x4c]
    xor       edx,ebp
    mov       dword ptr [esp+0xa0],edi
    mov       edi,dword ptr [esp+0x40]
    mov       dword ptr [esp+0xb0],edx
    mov       edx,dword ptr [esp+0x38]
    mov       ebp,dword ptr [esp+0x14]
    xor       edx,edi
    mov       edi,edx
    mov       dword ptr [esp+0x74],edx
    not       edi
    mov       dword ptr [esp+0x10],edi
    or        edi,ebp
    mov       ebp,dword ptr [esp+0x18]
    xor       edi,ebp
    mov       ebp,edx
    mov       edx,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x78],edi
    or        ebp,edx
    mov       edx,dword ptr [esp+0x10]
    xor       ebp,edx
    mov       edx,ebp
    xor       edx,dword ptr [esp+0x28]
    xor       edx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x11c],edx
    mov       edx,dword ptr [esp+0xb0]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x1c]
    or        edx,ebp
    mov       ebp,edx
    xor       ebp,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x138],ebp
    mov       ebp,dword ptr [esp+0x14]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xb0]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x2c]
    add       ebx,0x00000040
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x44]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x11c]
    mov       dword ptr [esp+0x118],edx
    mov       edx,edi
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x118]
    and       edi,ebp
    mov       ebp,dword ptr [esp+0x1c]
    xor       edx,edi
    mov       edi,edx
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0x138]
    mov       dword ptr [esp+0x174],edi
    mov       edi,dword ptr [esp+0x78]
    or        edi,ebp
    mov       ebp,dword ptr [esp+0x11c]
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x3c],edi
    or        edi,edx
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x20],edi
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0x118]
    mov       dword ptr [esp+0xd4],edi
    mov       edi,dword ptr [esp+0x78]
    or        edi,ebp
    mov       ebp,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x10],edi
    mov       edi,dword ptr [esp+0x138]
    and       ebp,edi
    mov       edi,dword ptr [esp+0x10]
    xor       edi,ebp
    mov       ebp,edi
    mov       dword ptr [esp+0x10],edi
    mov       edi,dword ptr [esp+0x74]
    not       ebp
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0x10]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x20]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x3c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xb0]
    not       edx
    xor       ebp,edx
    mov       dword ptr [esp+0x74],edi
    mov       dword ptr [esp+0xb0],ebp
    mov       ebp,dword ptr [esp+0x174]
    mov       edx,ebp
    or        edx,dword ptr [esp+0x74]
    not       edi
    mov       dword ptr [esp+0x130],edi
    xor       edx,edi
    mov       edi,dword ptr [esp+0xb0]
    mov       dword ptr [esp+0x50],edx
    xor       edi,edx
    mov       edx,dword ptr [ebx-0x1e0]
    or        edi,ebp
    mov       ebp,dword ptr [esp+0x284]
    mov       dword ptr [esp+0x100],edi
    mov       edi,dword ptr [ebx+0x20]
    xor       edi,edx
    mov       edx,edi
    mov       dword ptr [ebx+0x20],edi
    mov       edi,dword ptr [ebx+0x24]
    xor       edx,ebp
    mov       ebp,edx
    mov       edx,dword ptr [ebx-0x1dc]
    xor       edi,edx
    mov       dword ptr [esp+0x4c],ebp
    mov       dword ptr [ebx+0x24],edi
    mov       edx,edi
    xor       edx,dword ptr [esp+0x288]
    mov       edi,dword ptr [ebx+0x28]
    mov       dword ptr [esp+0x38],edx
    mov       edx,dword ptr [ebx-0x1d8]
    xor       edi,edx
    mov       dword ptr [ebx+0x28],edi
    mov       edx,edi
    mov       edi,dword ptr [esp+0x28c]
    xor       edx,edi
    mov       edi,dword ptr [ebx+0x2c]
    mov       dword ptr [esp+0x40],edx
    mov       edx,dword ptr [ebx-0x1d4]
    xor       edi,edx
    mov       dword ptr [ebx+0x2c],edi
    mov       edx,edi
    mov       edi,dword ptr [esp+0x290]
    xor       edx,edi
    mov       edi,dword ptr [ebx+0x30]
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [ebx-0x1d0]
    xor       edi,edx
    mov       dword ptr [ebx+0x30],edi
    mov       edx,edi
    mov       edi,dword ptr [esp+0x294]
    xor       edx,edi
    mov       edi,dword ptr [ebx+0x34]
    mov       dword ptr [esp+0x44],edx
    mov       edx,dword ptr [ebx-0x1cc]
    xor       edi,edx
    mov       dword ptr [ebx+0x34],edi
    mov       edx,edi
    mov       edi,dword ptr [esp+0x298]
    xor       edx,edi
    mov       edi,dword ptr [ebx+0x38]
    mov       dword ptr [esp+0x34],edx
    mov       edx,dword ptr [ebx-0x1c8]
    xor       edi,edx
    mov       dword ptr [ebx+0x38],edi
    mov       edx,edi
    mov       edi,dword ptr [esp+0x29c]
    xor       edx,edi
    mov       edi,dword ptr [ebx+0x3c]
    mov       dword ptr [esp+0x30],edx
    mov       edx,dword ptr [ebx-0x1c4]
    xor       edi,edx
    mov       dword ptr [ebx+0x3c],edi
    mov       edx,edi
    xor       edx,dword ptr [esp+0x2a0]
    mov       edi,edx
    mov       edx,dword ptr [ebx-0x200]
    xor       dword ptr [ebx],edx
    mov       edx,dword ptr [ebx]
    xor       edx,dword ptr [esp+0x264]
    mov       dword ptr [esp+0x60],edi
    xor       edx,edi
    mov       edi,edx
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [ebx-0x1fc]
    xor       edi,ebp
    mov       ebp,dword ptr [ebx+0x4]
    xor       ebp,edx
    mov       dword ptr [ebx+0x4],ebp
    mov       edx,ebp
    mov       ebp,dword ptr [esp+0x268]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x6c],edx
    xor       edx,ebp
    mov       ebp,dword ptr [ebx+0x8]
    mov       dword ptr [esp+0xd0],edx
    mov       edx,dword ptr [ebx-0x1f8]
    xor       ebp,edx
    mov       dword ptr [ebx+0x8],ebp
    mov       edx,ebp
    mov       ebp,dword ptr [esp+0x26c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x38]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x1c],edx
    xor       edx,ebp
    mov       ebp,dword ptr [ebx+0xc]
    mov       dword ptr [esp+0xec],edx
    mov       edx,dword ptr [ebx-0x1f4]
    xor       ebp,edx
    mov       dword ptr [ebx+0xc],ebp
    mov       edx,ebp
    mov       ebp,dword ptr [esp+0x270]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x38],edx
    xor       edx,ebp
    mov       ebp,dword ptr [ebx+0x10]
    mov       dword ptr [esp+0xf8],edx
    mov       edx,dword ptr [ebx-0x1f0]
    xor       ebp,edx
    mov       dword ptr [ebx+0x10],ebp
    mov       edx,ebp
    mov       ebp,dword ptr [esp+0x274]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    xor       edx,ebp
    mov       ebp,dword ptr [ebx+0x14]
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [ebx-0x1ec]
    xor       ebp,edx
    mov       dword ptr [ebx+0x14],ebp
    mov       edx,ebp
    mov       ebp,dword ptr [esp+0x278]
    xor       edx,ebp
    mov       ebp,dword ptr [ebx+0x18]
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [ebx-0x1e8]
    xor       ebp,edx
    mov       dword ptr [ebx+0x18],ebp
    mov       edx,ebp
    mov       ebp,dword ptr [esp+0x27c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x34]
    xor       edx,ebp
    mov       ebp,dword ptr [ebx+0x1c]
    mov       dword ptr [esp+0x48],edx
    mov       edx,dword ptr [ebx-0x1e4]
    xor       ebp,edx
    mov       dword ptr [ebx+0x1c],ebp
    mov       edx,ebp
    mov       ebp,dword ptr [esp+0x280]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xf8]
    mov       dword ptr [esp+0x28],edx
    mov       edx,edi
    not       ebp
    mov       dword ptr [esp+0x10],ebp
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x44]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xf8]
    mov       dword ptr [esp+0xf4],edx
    mov       edx,dword ptr [esp+0xec]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x10]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x10],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x60]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x84],edx
    mov       edx,dword ptr [esp+0xd0]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xec]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x10],edx
    xor       edx,ebp
    xor       edx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x154],edx
    mov       edx,edi
    mov       ebp,dword ptr [esp+0x10]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xd0]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x2c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x34]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xf4]
    and       ebp,edx
    mov       dword ptr [esp+0x170],edx
    mov       edx,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x10],ebp
    mov       ebp,dword ptr [esp+0xf4]
    or        ebp,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xec]
    xor       ebp,edx
    mov       dword ptr [esp+0x24],edx
    mov       edx,dword ptr [esp+0xf4]
    mov       dword ptr [esp+0xec],ebp
    mov       ebp,dword ptr [esp+0x154]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x84]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x3c],edx
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xf4]
    xor       edx,ebp
    mov       dword ptr [esp+0x20],edx
    xor       edi,edx
    mov       edx,ebp
    mov       ebp,dword ptr [esp+0x170]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xf4]
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x154]
    and       ebp,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xf8]
    mov       dword ptr [esp+0x10],edx
    not       edx
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x10]
    mov       dword ptr [esp+0xf8],ebp
    mov       ebp,dword ptr [esp+0x24]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x20]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x3c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xd0]
    not       edx
    xor       ebp,edx
    mov       dword ptr [esp+0xd0],ebp
    mov       ebp,dword ptr [esp+0xf8]
    mov       edx,ebp
    not       edx
    mov       dword ptr [esp+0x178],edx
    mov       edx,dword ptr [esp+0xec]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x178]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xd0]
    mov       dword ptr [esp+0x10],edx
    xor       ebp,edx
    mov       edx,dword ptr [esp+0xec]
    or        ebp,edx
    mov       edx,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x188],ebp
    mov       ebp,dword ptr [esp+0x10]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x4c]
    mov       dword ptr [esp+0x2a0],edx
    mov       edx,dword ptr [esp+0x6c]
    xor       edx,ebp
    mov       dword ptr [esp+0xb4],edx
    mov       edx,dword ptr [esp+0x38]
    mov       ebp,dword ptr [esp+0x40]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x54],edx
    not       edx
    mov       dword ptr [esp+0x10],edx
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0xa8],edx
    mov       edx,dword ptr [esp+0x54]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x10]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x10],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x30]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x90],edx
    mov       edx,dword ptr [esp+0xb4]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x1c]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x10],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xb4]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x2c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x44]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xa8]
    and       ebp,edx
    mov       dword ptr [esp+0x108],edx
    mov       edx,dword ptr [esp+0x90]
    mov       dword ptr [esp+0x10],ebp
    mov       ebp,dword ptr [esp+0xa8]
    or        ebp,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x24],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x70]
    mov       dword ptr [esp+0x84],edx
    mov       edx,dword ptr [esp+0xa8]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x90]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x3c],edx
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xa8]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x20],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x70]
    mov       dword ptr [esp+0xcc],edx
    mov       edx,dword ptr [esp+0xa8]
    and       edx,ebp
    mov       ebp,dword ptr [esp+0xa8]
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x108]
    or        ebp,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x10],edx
    not       edx
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x54],ebp
    mov       ebp,dword ptr [esp+0x24]
    add       ebx,0x00000040
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x20]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x3c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xb4]
    not       edx
    xor       ebp,edx
    mov       dword ptr [esp+0xb4],ebp
    mov       ebp,dword ptr [esp+0x54]
    mov       edx,ebp
    not       edx
    mov       dword ptr [esp+0x16c],edx
    mov       edx,dword ptr [esp+0x84]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x16c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xb4]
    xor       ebp,edx
    mov       dword ptr [esp+0x18c],edx
    mov       edx,dword ptr [esp+0x84]
    or        ebp,edx
    mov       edx,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x190],ebp
    mov       ebp,edx
    xor       ebp,dword ptr [csc_tabe+0x20]
    mov       dword ptr [esp+0x4c],ebp
    mov       ebp,dword ptr [esp+0x80]
    xor       ebp,dword ptr [csc_tabe+0x24]
    mov       dword ptr [esp+0x38],ebp
    mov       ebp,dword ptr [esp+0x7c]
    xor       ebp,dword ptr [csc_tabe+0x28]
    mov       dword ptr [esp+0x40],ebp
    mov       ebp,dword ptr [esp+0x9c]
    xor       ebp,dword ptr [csc_tabe+0x2c]
    mov       dword ptr [esp+0x18],ebp
    mov       ebp,dword ptr [esp+0xe0]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x98]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x30]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x168]
    mov       dword ptr [esp+0x44],edx
    mov       edx,dword ptr [esp+0x68]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x80]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x160]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x34]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x168]
    mov       dword ptr [esp+0x34],edx
    mov       edx,dword ptr [esp+0xe8]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x38]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x30],edx
    mov       edx,dword ptr [esp+0x94]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x3c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xe4]
    mov       dword ptr [esp+0x60],edx
    xor       ebp,edx
    mov       edx,dword ptr [csc_tabe]
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x4c]
    mov       dword ptr [esp+0x14],ebp
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x5c],ebp
    mov       ebp,dword ptr [csc_tabe+0x4]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x6c],edx
    xor       edx,ebp
    mov       dword ptr [esp+0x80],edx
    mov       edx,dword ptr [esp+0x104]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x8]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x1c],edx
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0xc]
    mov       dword ptr [esp+0x120],edx
    mov       edx,dword ptr [esp+0x64]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x38],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xa4]
    mov       dword ptr [esp+0x14c],edx
    mov       edx,dword ptr [esp+0xe4]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xbc]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x10]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x150]
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [esp+0xe4]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x8c]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x164]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x14]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x150]
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0xd8]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x34]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x18]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xc0]
    mov       dword ptr [esp+0x48],edx
    mov       edx,dword ptr [esp+0x158]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x1c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x14c]
    mov       dword ptr [esp+0x28],edx
    mov       edx,dword ptr [esp+0x5c]
    not       ebp
    mov       dword ptr [esp+0x10],ebp
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x44]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x14c]
    mov       dword ptr [esp+0xa4],edx
    mov       edx,dword ptr [esp+0x120]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x10]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x10],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x60]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x24],edx
    mov       edx,dword ptr [esp+0x80]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x120]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x10],edx
    xor       edx,ebp
    xor       edx,dword ptr [esp+0x30]
    mov       ebp,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x7c],edx
    mov       edx,dword ptr [esp+0x5c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x80]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x2c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x34]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xa4]
    and       ebp,edx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x24]
    mov       dword ptr [esp+0x20],ebp
    mov       ebp,dword ptr [esp+0xa4]
    or        ebp,edx
    mov       edx,dword ptr [esp+0x20]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x120]
    xor       ebp,edx
    mov       dword ptr [esp+0x68],edx
    mov       edx,dword ptr [esp+0xa4]
    mov       dword ptr [esp+0x120],ebp
    mov       ebp,dword ptr [esp+0x7c]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x24]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x68]
    mov       dword ptr [esp+0xe0],edx
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xa4]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x5c]
    xor       ebp,edx
    mov       dword ptr [esp+0x3c],edx
    mov       edx,dword ptr [esp+0xa4]
    mov       dword ptr [esp+0x5c],ebp
    mov       ebp,dword ptr [esp+0x10]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xa4]
    mov       dword ptr [esp+0x20],edx
    mov       edx,dword ptr [esp+0x7c]
    and       ebp,edx
    mov       edx,dword ptr [esp+0x20]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x14c]
    mov       dword ptr [esp+0x20],edx
    not       edx
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x14c],ebp
    mov       ebp,dword ptr [esp+0x68]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x3c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xe0]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x80]
    not       edx
    xor       ebp,edx
    mov       dword ptr [esp+0x80],ebp
    mov       ebp,dword ptr [esp+0x14c]
    mov       edx,ebp
    not       edx
    mov       dword ptr [esp+0x20],edx
    mov       edx,dword ptr [esp+0x120]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x20]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x80]
    mov       dword ptr [esp+0x88],edx
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x120]
    or        ebp,edx
    mov       edx,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x144],ebp
    mov       ebp,dword ptr [esp+0x4c]
    xor       edx,ebp
    mov       dword ptr [esp+0xbc],edx
    mov       edx,dword ptr [esp+0x38]
    mov       ebp,dword ptr [esp+0x40]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x64],edx
    not       edx
    mov       dword ptr [esp+0x3c],edx
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0xc0],edx
    mov       edx,dword ptr [esp+0x64]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x3c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x3c],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x30]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x94],edx
    mov       edx,dword ptr [esp+0xbc]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x1c]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x3c],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x104],edx
    mov       edx,dword ptr [esp+0x3c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xbc]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x2c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x44]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x94]
    mov       dword ptr [esp+0x164],edx
    mov       edx,dword ptr [esp+0xc0]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xc0]
    mov       dword ptr [esp+0x3c],edx
    mov       edx,dword ptr [esp+0x164]
    and       ebp,edx
    mov       edx,dword ptr [esp+0x3c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x68],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x104]
    mov       dword ptr [esp+0x150],edx
    mov       edx,dword ptr [esp+0xc0]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x94]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x98],edx
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xc0]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x14]
    mov       dword ptr [esp+0xe0],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x104]
    mov       dword ptr [esp+0x17c],edx
    mov       edx,dword ptr [esp+0xc0]
    and       edx,ebp
    mov       ebp,dword ptr [esp+0xc0]
    mov       dword ptr [esp+0x3c],edx
    mov       edx,dword ptr [esp+0x164]
    or        ebp,edx
    mov       edx,dword ptr [esp+0x3c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x64]
    mov       dword ptr [esp+0x3c],edx
    not       edx
    xor       ebp,edx
    mov       dword ptr [esp+0x64],ebp
    mov       edx,dword ptr [esp+0x3c]
    mov       ebp,dword ptr [esp+0x68]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xe0]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x98]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xbc]
    not       edx
    xor       ebp,edx
    mov       dword ptr [esp+0xbc],ebp
    mov       ebp,dword ptr [esp+0x64]
    mov       edx,ebp
    not       edx
    mov       dword ptr [esp+0x98],edx
    mov       edx,dword ptr [esp+0x150]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x98]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xbc]
    xor       ebp,edx
    mov       dword ptr [esp+0x58],edx
    mov       edx,dword ptr [esp+0x150]
    or        ebp,edx
    mov       edx,dword ptr [esp+0xcc]
    mov       dword ptr [esp+0x110],ebp
    mov       ebp,edx
    xor       ebp,dword ptr [csc_tabe+0x60]
    mov       dword ptr [esp+0x4c],ebp
    mov       ebp,dword ptr [esp+0xb4]
    xor       ebp,dword ptr [csc_tabe+0x64]
    mov       dword ptr [esp+0x38],ebp
    mov       ebp,dword ptr [esp+0x84]
    xor       ebp,dword ptr [csc_tabe+0x68]
    mov       dword ptr [esp+0x40],ebp
    mov       ebp,dword ptr [esp+0x54]
    xor       ebp,dword ptr [csc_tabe+0x6c]
    mov       dword ptr [esp+0x18],ebp
    mov       ebp,dword ptr [esp+0x16c]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xa8]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x70]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x190]
    mov       dword ptr [esp+0x44],edx
    mov       edx,dword ptr [esp+0xcc]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xb4]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x108]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x74]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x190]
    mov       dword ptr [esp+0x34],edx
    mov       edx,dword ptr [esp+0x70]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x78]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x18c]
    mov       dword ptr [esp+0x30],edx
    mov       edx,dword ptr [esp+0x90]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x7c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xd4]
    mov       dword ptr [esp+0x60],edx
    xor       ebp,edx
    mov       edx,dword ptr [csc_tabe+0x40]
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x4c]
    mov       dword ptr [esp+0x14],ebp
    xor       ebp,edx
    mov       edx,dword ptr [esp+0xb0]
    mov       dword ptr [esp+0xcc],ebp
    mov       ebp,dword ptr [csc_tabe+0x44]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x6c],edx
    xor       edx,ebp
    mov       dword ptr [esp+0x108],edx
    mov       edx,dword ptr [esp+0x174]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x48]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x1c],edx
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x4c]
    mov       dword ptr [esp+0xb4],edx
    mov       edx,dword ptr [esp+0x74]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x38],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x130]
    mov       dword ptr [esp+0x8c],edx
    mov       edx,dword ptr [esp+0xd4]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x78]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x50]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x100]
    mov       dword ptr [esp+0x18],edx
    mov       edx,dword ptr [esp+0xd4]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xb0]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x118]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x54]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x100]
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0x138]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x34]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x58]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x50]
    mov       dword ptr [esp+0x48],edx
    mov       edx,dword ptr [esp+0x11c]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x5c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x28],edx
    mov       edx,dword ptr [esp+0xcc]
    not       ebp
    mov       dword ptr [esp+0x50],ebp
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x44]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0xa8],edx
    mov       edx,dword ptr [esp+0xb4]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x50]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x50],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x60]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x50]
    mov       dword ptr [esp+0x68],edx
    mov       edx,dword ptr [esp+0x108]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xb4]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x50],edx
    xor       edx,ebp
    xor       edx,dword ptr [esp+0x30]
    mov       ebp,dword ptr [esp+0x50]
    mov       dword ptr [esp+0x160],edx
    mov       edx,dword ptr [esp+0xcc]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x108]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x2c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x34]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x168],edx
    mov       edx,dword ptr [esp+0xa8]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xa8]
    mov       dword ptr [esp+0x50],edx
    mov       edx,dword ptr [esp+0x168]
    and       ebp,edx
    mov       edx,dword ptr [esp+0x50]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xb4]
    xor       ebp,edx
    mov       dword ptr [esp+0x100],edx
    mov       edx,dword ptr [esp+0xa8]
    mov       dword ptr [esp+0xb4],ebp
    mov       ebp,dword ptr [esp+0x160]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x68]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x100]
    mov       dword ptr [esp+0x18c],edx
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xa8]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xcc]
    xor       ebp,edx
    mov       dword ptr [esp+0x130],edx
    mov       edx,dword ptr [esp+0xa8]
    mov       dword ptr [esp+0xcc],ebp
    mov       ebp,dword ptr [esp+0x160]
    and       edx,ebp
    mov       ebp,dword ptr [esp+0xa8]
    mov       dword ptr [esp+0x50],edx
    mov       edx,dword ptr [esp+0x168]
    or        ebp,edx
    mov       edx,dword ptr [esp+0x50]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x50],edx
    not       edx
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x50]
    mov       dword ptr [esp+0x8c],ebp
    mov       ebp,dword ptr [esp+0x100]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x130]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x18c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x108]
    not       edx
    xor       ebp,edx
    mov       dword ptr [esp+0x108],ebp
    mov       ebp,dword ptr [esp+0x8c]
    mov       edx,ebp
    not       edx
    mov       dword ptr [esp+0xe0],edx
    mov       edx,dword ptr [esp+0xb4]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xe0]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x108]
    mov       dword ptr [esp+0x194],edx
    xor       ebp,edx
    mov       edx,dword ptr [esp+0xb4]
    or        ebp,edx
    mov       edx,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x3c],ebp
    mov       ebp,dword ptr [esp+0x4c]
    xor       edx,ebp
    mov       dword ptr [esp+0x54],edx
    mov       edx,dword ptr [esp+0x38]
    mov       ebp,dword ptr [esp+0x40]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x78],edx
    not       edx
    mov       dword ptr [esp+0x50],edx
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x18]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0xb0],edx
    mov       edx,dword ptr [esp+0x78]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x50]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x50],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x30]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x50]
    mov       dword ptr [esp+0x158],edx
    mov       edx,dword ptr [esp+0x54]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x1c]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x50],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x16c],edx
    mov       edx,dword ptr [esp+0x50]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x54]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x2c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x44]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x158]
    mov       dword ptr [esp+0x138],edx
    mov       edx,dword ptr [esp+0xb0]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xb0]
    mov       dword ptr [esp+0x50],edx
    mov       edx,dword ptr [esp+0x138]
    and       ebp,edx
    mov       edx,dword ptr [esp+0x50]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x100],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x16c]
    mov       dword ptr [esp+0x11c],edx
    mov       edx,dword ptr [esp+0xb0]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x158]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x100]
    mov       dword ptr [esp+0x18c],edx
    or        edx,ebp
    mov       ebp,dword ptr [esp+0xb0]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x130],edx
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x16c]
    mov       dword ptr [esp+0x2c0],edx
    mov       edx,dword ptr [esp+0xb0]
    and       edx,ebp
    mov       ebp,dword ptr [esp+0xb0]
    mov       dword ptr [esp+0x50],edx
    mov       edx,dword ptr [esp+0x138]
    or        ebp,edx
    mov       edx,dword ptr [esp+0x50]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x50],edx
    not       edx
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x50]
    mov       dword ptr [esp+0x78],ebp
    mov       ebp,dword ptr [esp+0x100]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x130]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x18c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x54]
    not       edx
    xor       ebp,edx
    mov       dword ptr [esp+0x54],ebp
    mov       ebp,dword ptr [esp+0x78]
    mov       edx,ebp
    not       edx
    mov       dword ptr [esp+0x100],edx
    mov       edx,dword ptr [esp+0x11c]
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x100]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x54]
    xor       ebp,edx
    mov       dword ptr [esp+0x130],edx
    mov       edx,dword ptr [esp+0x11c]
    or        ebp,edx
    mov       edx,ecx
    mov       dword ptr [esp+0x50],ebp
    mov       ebp,dword ptr [csc_tabe+0xa0]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0xa4]
    mov       dword ptr [esp+0x4c],edx
    mov       edx,dword ptr [esp+0xb8]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0xa8]
    mov       dword ptr [esp+0x38],edx
    mov       edx,dword ptr [esp+0x2b0]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0xac]
    mov       dword ptr [esp+0x40],edx
    mov       edx,dword ptr [esp+0x19c]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x2bc]
    mov       dword ptr [esp+0x18],edx
    mov       edx,ecx
    or        edx,ebp
    mov       ebp,dword ptr [esp+0x2c4]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0xb0]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xb8]
    mov       dword ptr [esp+0x44],edx
    mov       edx,dword ptr [esp+0x2c8]
    xor       ecx,edx
    or        ecx,ebp
    mov       ebp,dword ptr [esp+0x1a0]
    xor       ecx,ebp
    mov       ebp,dword ptr [csc_tabe+0xb4]
    xor       ecx,ebp
    mov       ebp,dword ptr [csc_tabe+0xb8]
    mov       dword ptr [esp+0x34],ecx
    mov       ecx,dword ptr [esp+0x15c]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x2a4]
    xor       ecx,ebp
    mov       ebp,dword ptr [csc_tabe+0xbc]
    mov       dword ptr [esp+0x30],ecx
    mov       ecx,dword ptr [esp+0x148]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x80]
    xor       ecx,ebp
    mov       ebp,eax
    xor       ebp,ecx
    mov       dword ptr [esp+0x60],ecx
    mov       ecx,dword ptr [esp+0x4c]
    xor       ebp,edx
    mov       edx,dword ptr [csc_tabe+0x84]
    mov       dword ptr [esp+0x14],ebp
    xor       ebp,ecx
    mov       ecx,dword ptr [esp+0xfc]
    xor       ecx,edx
    mov       dword ptr [esp+0x6c],ecx
    mov       edx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0x2b8]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x88]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x1c],ecx
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x8c]
    mov       dword ptr [esp+0xb8],ecx
    mov       ecx,dword ptr [esp+0x2a8]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x38],ecx
    xor       ecx,edx
    mov       dword ptr [esp+0xe8],ecx
    mov       ecx,eax
    or        ecx,dword ptr [esp+0x2ac]
    xor       ecx,dword ptr [esp+0x2b4]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x90]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x180]
    xor       eax,edx
    mov       dword ptr [esp+0x18],ecx
    or        eax,dword ptr [esp+0xfc]
    xor       eax,dword ptr [esp+0xc4]
    xor       eax,dword ptr [csc_tabe+0x94]
    mov       dword ptr [esp+0x2c],eax
    mov       eax,dword ptr [esp+0x124]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x34]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x98]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xac]
    mov       dword ptr [esp+0x48],eax
    mov       eax,dword ptr [esp+0x134]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x9c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xe8]
    mov       dword ptr [esp+0x28],eax
    mov       eax,edx
    not       eax
    mov       dword ptr [esp+0xc4],eax
    mov       eax,ebp
    or        eax,dword ptr [esp+0xc4]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x44]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xb8]
    or        ecx,edx
    mov       edx,dword ptr [esp+0xc4]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x28]
    mov       dword ptr [esp+0xac],ecx
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x15c],eax
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xac]
    mov       dword ptr [esp+0x84],ecx
    mov       ecx,dword ptr [esp+0x74]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xb8]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x48]
    mov       dword ptr [esp+0xac],ecx
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xac]
    mov       dword ptr [esp+0x9c],ecx
    mov       ecx,ebp
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x74]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x2c]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x34]
    xor       ecx,edx
    mov       edx,eax
    and       edx,ecx
    mov       dword ptr [esp+0x124],ecx
    mov       ecx,dword ptr [esp+0x84]
    mov       dword ptr [esp+0xac],edx
    mov       edx,eax
    or        edx,ecx
    mov       ecx,dword ptr [esp+0xac]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xb8]
    xor       edx,ecx
    mov       dword ptr [esp+0xc4],ecx
    mov       dword ptr [esp+0xb8],edx
    mov       edx,dword ptr [esp+0x9c]
    mov       ecx,eax
    or        ecx,edx
    mov       edx,dword ptr [esp+0x84]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xc4]
    mov       dword ptr [esp+0x180],ecx
    or        ecx,edx
    mov       edx,dword ptr [esp+0x124]
    xor       ecx,eax
    mov       dword ptr [esp+0xac],ecx
    xor       ebp,ecx
    mov       ecx,eax
    or        ecx,edx
    mov       edx,dword ptr [esp+0x9c]
    and       eax,edx
    mov       edx,dword ptr [esp+0xe8]
    xor       ecx,eax
    mov       eax,ecx
    not       eax
    xor       edx,eax
    mov       eax,dword ptr [esp+0xc4]
    or        ecx,eax
    mov       eax,dword ptr [esp+0xac]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x180]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x74]
    not       ecx
    xor       eax,ecx
    lea       ecx,[esp+0x15c]
    push      ecx
    lea       ecx,[esp+0x128]
    push      ecx
    lea       ecx,[esp+0xa4]
    push      ecx
    lea       ecx,[esp+0x90]
    push      ecx
    push      ebp
    mov       dword ptr [esp+0x88],eax
    push      eax
    mov       eax,dword ptr [esp+0xd0]
    mov       dword ptr [esp+0x100],edx
    push      eax
    push      edx
    call      csc_transF
    mov       eax,dword ptr [esp+0x8c]
    mov       ecx,dword ptr [esp+0x6c]
    mov       edx,dword ptr [esp+0x58]
    xor       eax,ecx
    mov       dword ptr [esp+0x11c],eax
    mov       eax,dword ptr [esp+0x60]
    mov       ecx,dword ptr [esp+0x34]
    xor       edx,eax
    mov       eax,edx
    add       esp,0x00000020
    not       eax
    mov       dword ptr [esp+0xac],eax
    or        eax,ecx
    mov       ecx,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x134],edx
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x148],eax
    or        edx,ecx
    xor       edx,dword ptr [esp+0xac]
    mov       ecx,edx
    xor       ecx,dword ptr [esp+0x28]
    xor       ecx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x70],ecx
    mov       ecx,dword ptr [esp+0xfc]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x1c]
    or        ecx,edx
    mov       edx,ecx
    xor       edx,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x90],edx
    mov       edx,dword ptr [esp+0x14]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xfc]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x2c]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x44]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x70]
    mov       dword ptr [esp+0xd4],ecx
    mov       ecx,eax
    or        ecx,edx
    mov       edx,eax
    mov       dword ptr [esp+0xac],ecx
    mov       ecx,dword ptr [esp+0xd4]
    and       edx,ecx
    mov       ecx,dword ptr [esp+0xac]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0xc4],ecx
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x90]
    mov       dword ptr [esp+0x190],ecx
    mov       ecx,eax
    or        ecx,edx
    mov       edx,dword ptr [esp+0x70]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xc4]
    mov       dword ptr [esp+0x180],ecx
    or        ecx,edx
    mov       edx,dword ptr [esp+0x14]
    xor       ecx,eax
    mov       dword ptr [esp+0xac],ecx
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x90]
    mov       dword ptr [esp+0x174],ecx
    mov       ecx,eax
    and       ecx,edx
    mov       edx,dword ptr [esp+0xd4]
    or        eax,edx
    mov       edx,dword ptr [esp+0x134]
    xor       ecx,eax
    mov       eax,ecx
    not       eax
    xor       edx,eax
    mov       eax,dword ptr [esp+0xc4]
    or        ecx,eax
    mov       eax,dword ptr [esp+0xac]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x180]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0xfc]
    not       ecx
    xor       eax,ecx
    lea       ecx,[esp+0x148]
    push      ecx
    lea       ecx,[esp+0xd8]
    push      ecx
    lea       ecx,[esp+0x98]
    push      ecx
    lea       ecx,[esp+0x7c]
    push      ecx
    mov       ecx,dword ptr [esp+0x184]
    push      ecx
    mov       dword ptr [esp+0x110],eax
    push      eax
    mov       eax,dword ptr [esp+0x1a8]
    mov       dword ptr [esp+0x14c],edx
    push      eax
    push      edx
    call      csc_transF
    mov       ecx,dword ptr [csc_tabe+0xe0]
    mov       eax,edi
    mov       edx,dword ptr [csc_tabe+0xe8]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xf0]
    mov       dword ptr [esp+0x6c],eax
    mov       eax,dword ptr [csc_tabe+0xe4]
    add       esp,0x00000020
    xor       ecx,eax
    mov       eax,dword ptr [esp+0xec]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0xec]
    mov       dword ptr [esp+0x40],eax
    mov       eax,dword ptr [esp+0xf8]
    xor       eax,edx
    mov       edx,edi
    mov       dword ptr [esp+0x18],eax
    mov       eax,dword ptr [esp+0x178]
    or        edx,eax
    mov       eax,dword ptr [esp+0xf4]
    xor       edx,eax
    mov       eax,dword ptr [csc_tabe+0xf0]
    xor       edx,eax
    mov       eax,dword ptr [esp+0x188]
    xor       edi,eax
    mov       eax,dword ptr [esp+0xd0]
    or        edi,eax
    mov       eax,dword ptr [esp+0x170]
    xor       edi,eax
    mov       eax,dword ptr [csc_tabe+0xf4]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x154]
    mov       dword ptr [esp+0x34],edi
    mov       edi,dword ptr [esp+0x188]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0xf8]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0xfc]
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esp+0x2a0]
    xor       eax,edi
    mov       edi,esi
    mov       dword ptr [esp+0x60],eax
    xor       edi,eax
    mov       eax,dword ptr [csc_tabe+0xc0]
    mov       dword ptr [esp+0x44],edx
    xor       edi,eax
    mov       eax,dword ptr [esp+0x4c]
    mov       dword ptr [esp+0x14],edi
    xor       edi,eax
    mov       eax,dword ptr [esp+0xf0]
    xor       eax,dword ptr [csc_tabe+0xc4]
    mov       dword ptr [esp+0x6c],eax
    xor       eax,ecx
    mov       dword ptr [esp+0xf4],eax
    mov       eax,dword ptr [esp+0xdc]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0xc8]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x1c],eax
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0xcc]
    mov       dword ptr [esp+0xf8],eax
    mov       eax,dword ptr [esp+0xc8]
    xor       eax,ecx
    mov       dword ptr [esp+0x38],eax
    mov       ecx,eax
    mov       eax,dword ptr [esp+0x18]
    xor       ecx,eax
    mov       dword ptr [esp+0xd0],ecx
    mov       ecx,esi
    or        ecx,dword ptr [esp+0x2cc]
    xor       ecx,dword ptr [esp+0x198]
    xor       ecx,eax
    mov       eax,dword ptr [csc_tabe+0xd0]
    xor       ecx,eax
    mov       dword ptr [esp+0x18],ecx
    mov       eax,dword ptr [esp+0xa0]
    xor       esi,eax
    mov       eax,dword ptr [esp+0xf0]
    or        esi,eax
    mov       eax,dword ptr [esp+0x114]
    xor       esi,eax
    mov       eax,dword ptr [csc_tabe+0xd4]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x13c]
    mov       dword ptr [esp+0x2c],esi
    mov       esi,dword ptr [esp+0xa0]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x34]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0xd8]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x12c]
    mov       dword ptr [esp+0x48],eax
    mov       eax,dword ptr [esp+0x140]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0xdc]
    xor       eax,esi
    mov       esi,dword ptr [esp+0xd0]
    mov       dword ptr [esp+0x28],eax
    mov       eax,edi
    not       esi
    or        eax,esi
    xor       eax,ecx
    xor       eax,edx
    mov       edx,dword ptr [esp+0xf8]
    mov       ecx,edx
    mov       dword ptr [esp+0xc4],eax
    or        ecx,dword ptr [esp+0xd0]
    xor       ecx,esi
    mov       esi,ecx
    xor       esi,dword ptr [esp+0x28]
    xor       esi,dword ptr [esp+0x60]
    mov       dword ptr [esp+0xe4],esi
    mov       esi,dword ptr [esp+0xf4]
    xor       esi,ecx
    or        esi,edx
    mov       edx,dword ptr [esp+0x48]
    mov       ecx,esi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    xor       ecx,edx
    mov       edx,edi
    mov       dword ptr [esp+0xd8],ecx
    mov       ecx,dword ptr [esp+0xf4]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x2c]
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x34]
    xor       edx,esi
    mov       esi,eax
    xor       edx,ecx
    mov       ecx,eax
    and       ecx,edx
    mov       dword ptr [esp+0x154],edx
    mov       dword ptr [esp+0x12c],ecx
    mov       ecx,dword ptr [esp+0xe4]
    or        esi,ecx
    mov       ecx,dword ptr [esp+0x12c]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0xf8]
    xor       esi,ecx
    mov       dword ptr [esp+0xa0],ecx
    mov       dword ptr [esp+0xf8],esi
    mov       esi,dword ptr [esp+0xd8]
    mov       ecx,eax
    or        ecx,esi
    mov       esi,dword ptr [esp+0xe4]
    xor       ecx,esi
    mov       dword ptr [esp+0x12c],ecx
    mov       esi,ecx
    mov       ecx,dword ptr [esp+0xa0]
    or        esi,ecx
    mov       ecx,eax
    xor       esi,eax
    xor       edi,esi
    or        ecx,edx
    mov       edx,dword ptr [esp+0xd8]
    and       eax,edx
    mov       edx,dword ptr [esp+0xd0]
    xor       ecx,eax
    mov       eax,ecx
    not       eax
    xor       edx,eax
    mov       eax,dword ptr [esp+0xa0]
    or        ecx,eax
    mov       eax,dword ptr [esp+0xf4]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x12c]
    xor       ecx,esi
    mov       dword ptr [esp+0xd0],edx
    not       ecx
    xor       eax,ecx
    lea       ecx,[esp+0xc4]
    push      ecx
    lea       ecx,[esp+0x158]
    push      ecx
    lea       ecx,[esp+0xe0]
    push      ecx
    lea       ecx,[esp+0xf0]
    push      ecx
    push      edi
    mov       dword ptr [esp+0x108],eax
    push      eax
    mov       eax,dword ptr [esp+0x110]
    push      eax
    push      edx
    call      csc_transF
    mov       eax,dword ptr [esp+0x8c]
    mov       edx,dword ptr [esp+0x6c]
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,edx
    mov       esi,eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,ecx
    mov       dword ptr [esp+0x110],esi
    mov       edx,eax
    add       esp,0x00000020
    mov       dword ptr [esp+0x140],edx
    mov       ecx,edx
    or        edx,dword ptr [esp+0x1c]
    not       ecx
    xor       edx,ecx
    mov       eax,ecx
    mov       ecx,edx
    or        eax,dword ptr [esp+0x14]
    xor       ecx,dword ptr [esp+0x28]
    xor       eax,dword ptr [esp+0x18]
    xor       ecx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x178],eax
    mov       dword ptr [esp+0x118],ecx
    mov       ecx,esi
    mov       esi,dword ptr [esp+0x48]
    xor       ecx,edx
    or        ecx,dword ptr [esp+0x1c]
    mov       edx,ecx
    xor       edx,esi
    mov       esi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0xec],edx
    mov       edx,dword ptr [esp+0xf0]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x2c]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x44]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x118]
    xor       ecx,edx
    mov       edx,ecx
    mov       ecx,eax
    mov       dword ptr [esp+0x13c],edx
    and       ecx,edx
    mov       edx,eax
    or        edx,esi
    mov       esi,dword ptr [esp+0x1c]
    xor       ecx,edx
    mov       edx,ecx
    xor       edx,esi
    mov       esi,dword ptr [esp+0xec]
    mov       dword ptr [esp+0x170],edx
    mov       edx,eax
    or        edx,esi
    mov       esi,dword ptr [esp+0x118]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x198],edx
    or        edx,ecx
    xor       edx,eax
    mov       dword ptr [esp+0x12c],edx
    xor       edx,esi
    mov       esi,dword ptr [esp+0xec]
    mov       dword ptr [esp+0x188],edx
    mov       edx,eax
    and       edx,esi
    mov       esi,dword ptr [esp+0x13c]
    or        eax,esi
    xor       edx,eax
    mov       eax,dword ptr [esp+0x140]
    mov       esi,edx
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x12c]
    not       esi
    xor       eax,esi
    mov       esi,dword ptr [esp+0x198]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0xf0]
    xor       edx,esi
    mov       dword ptr [esp+0x140],eax
    not       edx
    xor       ecx,edx
    lea       edx,[esp+0x178]
    push      edx
    lea       edx,[esp+0x140]
    push      edx
    lea       edx,[esp+0xf4]
    push      edx
    lea       edx,[esp+0x124]
    push      edx
    mov       edx,dword ptr [esp+0x198]
    push      edx
    mov       dword ptr [esp+0x104],ecx
    push      ecx
    mov       ecx,dword ptr [esp+0x188]
    push      ecx
    push      eax
    call      csc_transF
    mov       eax,dword ptr [esp+0x2e0]
    mov       edx,dword ptr [csc_tabe+0x120]
    mov       esi,dword ptr [csc_tabe+0x124]
    mov       ecx,eax
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x6c],ecx
    mov       ecx,edx
    xor       ecx,esi
    mov       esi,dword ptr [csc_tabe+0x128]
    mov       dword ptr [esp+0x58],ecx
    mov       ecx,dword ptr [esp+0x13c]
    xor       ecx,esi
    mov       esi,dword ptr [csc_tabe+0x12c]
    mov       dword ptr [esp+0x60],ecx
    mov       ecx,dword ptr [esp+0x98]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x120]
    mov       dword ptr [esp+0x38],ecx
    mov       ecx,eax
    or        ecx,esi
    mov       esi,dword ptr [esp+0xd0]
    xor       ecx,esi
    mov       esi,dword ptr [csc_tabe+0x130]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x70]
    xor       eax,esi
    add       esp,0x00000020
    or        eax,edx
    mov       edx,dword ptr [esp+0x138]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x134]
    mov       dword ptr [esp+0x44],ecx
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x138]
    mov       dword ptr [esp+0x34],eax
    mov       eax,dword ptr [esp+0x16c]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x130]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x13c]
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esp+0x158]
    xor       eax,esi
    xor       eax,edx
    mov       esi,eax
    mov       eax,dword ptr [esp+0x17c]
    mov       edx,eax
    mov       dword ptr [esp+0x60],esi
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x100]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x4c]
    mov       dword ptr [esp+0x14],edx
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x104]
    mov       dword ptr [esp+0xdc],edx
    mov       edx,dword ptr [esp+0xbc]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x6c],edx
    xor       edx,esi
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0x150]
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x108]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x1c],edx
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x10c]
    mov       dword ptr [esp+0x54],edx
    mov       edx,dword ptr [esp+0x64]
    xor       edx,esi
    mov       dword ptr [esp+0x38],edx
    mov       esi,edx
    mov       edx,dword ptr [esp+0x18]
    xor       esi,edx
    mov       dword ptr [esp+0x64],esi
    mov       esi,eax
    or        esi,dword ptr [esp+0x98]
    xor       esi,dword ptr [esp+0xc0]
    xor       esi,edx
    mov       edx,dword ptr [csc_tabe+0x110]
    xor       esi,edx
    mov       edx,dword ptr [esp+0x110]
    xor       eax,edx
    or        eax,dword ptr [esp+0xbc]
    xor       eax,dword ptr [esp+0x164]
    xor       eax,dword ptr [csc_tabe+0x114]
    mov       dword ptr [esp+0x2c],eax
    mov       eax,dword ptr [esp+0x104]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x34]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x118]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x58]
    mov       dword ptr [esp+0x48],eax
    mov       eax,dword ptr [esp+0x94]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x11c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x64]
    mov       dword ptr [esp+0x28],eax
    mov       eax,dword ptr [esp+0xdc]
    not       edx
    or        eax,edx
    mov       dword ptr [esp+0x58],edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,esi
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x54]
    mov       dword ptr [esp+0xa0],eax
    or        ecx,edx
    mov       edx,dword ptr [esp+0x58]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x58],ecx
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x60]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x58]
    mov       dword ptr [esp+0xc8],ecx
    mov       ecx,dword ptr [esp+0x78]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x54]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x58],ecx
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x58]
    mov       dword ptr [esp+0xbc],ecx
    mov       ecx,dword ptr [esp+0xdc]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x78]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x2c]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x34]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xc8]
    mov       dword ptr [esp+0xc0],ecx
    mov       ecx,eax
    or        ecx,edx
    mov       edx,eax
    mov       dword ptr [esp+0x58],ecx
    mov       ecx,dword ptr [esp+0xc0]
    and       edx,ecx
    mov       ecx,dword ptr [esp+0x58]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x54]
    xor       edx,ecx
    mov       dword ptr [esp+0x98],ecx
    mov       dword ptr [esp+0x54],edx
    mov       edx,dword ptr [esp+0xbc]
    mov       ecx,eax
    or        ecx,edx
    mov       edx,dword ptr [esp+0xc8]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x98]
    mov       dword ptr [esp+0x110],ecx
    or        ecx,edx
    mov       edx,dword ptr [esp+0xdc]
    xor       ecx,eax
    xor       edx,ecx
    mov       dword ptr [esp+0x58],ecx
    mov       dword ptr [esp+0xdc],edx
    mov       edx,dword ptr [esp+0xbc]
    mov       ecx,eax
    and       ecx,edx
    mov       edx,dword ptr [esp+0xc0]
    or        eax,edx
    mov       edx,dword ptr [esp+0x64]
    xor       ecx,eax
    mov       eax,ecx
    not       eax
    xor       edx,eax
    mov       eax,dword ptr [esp+0x98]
    or        ecx,eax
    mov       eax,dword ptr [esp+0x58]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x110]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x78]
    not       ecx
    xor       eax,ecx
    lea       ecx,[esp+0xa0]
    push      ecx
    lea       ecx,[esp+0xc4]
    mov       dword ptr [esp+0x68],edx
    mov       dword ptr [esp+0x7c],eax
    push      ecx
    lea       ecx,[esp+0xc4]
    push      ecx
    lea       ecx,[esp+0xd4]
    push      ecx
    mov       ecx,dword ptr [esp+0xec]
    push      ecx
    push      eax
    mov       eax,dword ptr [esp+0x6c]
    push      eax
    push      edx
    call      csc_transF
    mov       edx,dword ptr [esp+0xdc]
    mov       ecx,dword ptr [esp+0xe8]
    mov       eax,dword ptr [esp+0xe0]
    mov       dword ptr [esp+0x25c],edx
    mov       edx,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x260],ecx
    mov       ecx,dword ptr [esp+0xc0]
    mov       dword ptr [esp+0x258],eax
    mov       eax,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x250],edx
    mov       edx,dword ptr [esp+0xfc]
    mov       dword ptr [esp+0x254],ecx
    mov       ecx,dword ptr [esp+0x98]
    mov       dword ptr [esp+0x24c],eax
    mov       eax,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x244],edx
    mov       edx,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x248],ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x58]
    mov       dword ptr [esp+0x84],eax
    xor       edx,ecx
    add       esp,0x00000020
    mov       ecx,edx
    mov       dword ptr [esp+0x58],edx
    not       ecx
    mov       eax,ecx
    or        eax,dword ptr [esp+0x14]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x1c]
    or        edx,esi
    mov       esi,dword ptr [esp+0x28]
    xor       edx,ecx
    mov       dword ptr [esp+0x98],eax
    mov       ecx,edx
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x30]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0xc8],ecx
    mov       ecx,dword ptr [esp+0x64]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x1c]
    or        ecx,edx
    mov       edx,ecx
    xor       edx,esi
    mov       esi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0x64]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x2c]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x44]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0xc8]
    xor       ecx,edx
    mov       edx,ecx
    mov       ecx,eax
    or        ecx,esi
    mov       esi,eax
    and       esi,edx
    mov       dword ptr [esp+0x94],edx
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x1c]
    mov       edx,ecx
    xor       edx,esi
    mov       esi,dword ptr [esp+0x78]
    mov       dword ptr [esp+0xa0],edx
    mov       edx,eax
    or        edx,esi
    mov       esi,dword ptr [esp+0xc8]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x17c],edx
    or        edx,ecx
    xor       edx,eax
    mov       dword ptr [esp+0x110],edx
    xor       edx,esi
    mov       esi,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x114],edx
    mov       edx,eax
    and       edx,esi
    mov       esi,dword ptr [esp+0x94]
    or        eax,esi
    mov       esi,dword ptr [esp+0x58]
    xor       edx,eax
    mov       eax,edx
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x17c]
    not       eax
    xor       esi,eax
    mov       eax,dword ptr [esp+0x110]
    xor       edx,eax
    mov       eax,dword ptr [esp+0x64]
    xor       edx,ecx
    lea       ecx,[esp+0x98]
    not       edx
    xor       eax,edx
    push      ecx
    lea       edx,[esp+0x98]
    lea       ecx,[esp+0x7c]
    push      edx
    push      ecx
    mov       ecx,dword ptr [esp+0x120]
    lea       edx,[esp+0xd4]
    push      edx
    mov       edx,dword ptr [esp+0xb0]
    push      ecx
    push      eax
    push      edx
    push      esi
    mov       dword ptr [esp+0x84],eax
    call      csc_transF
    mov       eax,dword ptr [esp+0xe8]
    mov       edx,dword ptr [esp+0xb4]
    mov       dword ptr [esp+0x1e0],eax
    mov       eax,dword ptr [esp+0xb8]
    mov       ecx,dword ptr [esp+0x98]
    mov       dword ptr [esp+0x1d4],eax
    mov       eax,dword ptr [esp+0x134]
    mov       dword ptr [esp+0x1d0],esi
    mov       esi,dword ptr [csc_tabe+0x160]
    mov       dword ptr [esp+0x1c4],eax
    mov       eax,dword ptr [esp+0x1a8]
    mov       dword ptr [esp+0x1d8],edx
    mov       edx,dword ptr [esp+0x84]
    xor       eax,esi
    mov       dword ptr [esp+0x1c8],edx
    mov       edx,dword ptr [csc_tabe+0x164]
    mov       dword ptr [esp+0x6c],eax
    mov       eax,dword ptr [esp+0x110]
    mov       dword ptr [esp+0x1dc],ecx
    mov       ecx,dword ptr [esp+0xc0]
    xor       eax,edx
    mov       esi,dword ptr [csc_tabe+0x16c]
    mov       dword ptr [esp+0x1cc],ecx
    mov       ecx,dword ptr [csc_tabe+0x168]
    mov       edx,eax
    mov       eax,dword ptr [esp+0x190]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x170]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0x160]
    xor       eax,esi
    add       esp,0x00000020
    mov       dword ptr [esp+0x18],eax
    mov       eax,dword ptr [esp+0x178]
    xor       eax,ecx
    mov       dword ptr [esp+0x44],eax
    mov       eax,dword ptr [esp+0x13c]
    mov       esi,dword ptr [csc_tabe+0x174]
    mov       ecx,dword ptr [csc_tabe+0x178]
    xor       eax,esi
    mov       dword ptr [esp+0x34],eax
    mov       eax,dword ptr [esp+0xec]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x118]
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [csc_tabe+0x17c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x174]
    xor       eax,ecx
    mov       dword ptr [esp+0x60],ecx
    mov       ecx,dword ptr [csc_tabe+0x140]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x144]
    mov       dword ptr [esp+0x14],eax
    mov       esi,eax
    mov       eax,dword ptr [esp+0x4c]
    xor       esi,eax
    mov       eax,dword ptr [esp+0xfc]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x148]
    mov       dword ptr [esp+0x6c],eax
    xor       eax,edx
    mov       dword ptr [esp+0x64],eax
    mov       eax,dword ptr [esp+0x190]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x40]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x14c]
    mov       dword ptr [esp+0x1c],eax
    xor       eax,edx
    mov       edx,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x54],eax
    mov       eax,dword ptr [esp+0x134]
    xor       eax,ecx
    mov       dword ptr [esp+0x38],eax
    mov       ecx,eax
    mov       eax,dword ptr [esp+0x148]
    xor       ecx,edx
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x150]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x154]
    mov       dword ptr [esp+0x18],eax
    mov       eax,dword ptr [esp+0xd4]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x34]
    mov       dword ptr [esp+0x2c],eax
    mov       eax,dword ptr [esp+0x90]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x158]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x15c]
    mov       dword ptr [esp+0x48],eax
    mov       eax,dword ptr [esp+0x70]
    xor       eax,edx
    mov       edx,ecx
    mov       dword ptr [esp+0x28],eax
    mov       eax,esi
    not       edx
    mov       dword ptr [esp+0x58],edx
    or        eax,edx
    mov       edx,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x104],ecx
    xor       eax,edx
    mov       edx,dword ptr [esp+0x44]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x58]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x58],edx
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x60]
    mov       dword ptr [esp+0xa0],eax
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x64]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x58]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x54]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x58],ecx
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x58]
    mov       dword ptr [esp+0x90],ecx
    mov       ecx,esi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x64]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x2c]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x34]
    xor       ecx,edx
    mov       edx,eax
    and       edx,ecx
    mov       dword ptr [esp+0x94],ecx
    mov       ecx,dword ptr [esp+0x70]
    mov       dword ptr [esp+0x58],edx
    mov       edx,eax
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x58]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x54]
    xor       edx,ecx
    mov       dword ptr [esp+0x98],ecx
    mov       dword ptr [esp+0x54],edx
    mov       edx,dword ptr [esp+0x90]
    mov       ecx,eax
    or        ecx,edx
    mov       edx,dword ptr [esp+0x70]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x98]
    mov       dword ptr [esp+0x110],ecx
    or        ecx,edx
    mov       edx,dword ptr [esp+0x94]
    xor       ecx,eax
    mov       dword ptr [esp+0x58],ecx
    xor       esi,ecx
    mov       ecx,eax
    or        ecx,edx
    mov       edx,dword ptr [esp+0x90]
    and       eax,edx
    mov       edx,dword ptr [esp+0x104]
    xor       ecx,eax
    mov       eax,ecx
    not       eax
    xor       edx,eax
    mov       eax,dword ptr [esp+0x98]
    or        ecx,eax
    mov       eax,dword ptr [esp+0x58]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x110]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x64]
    not       ecx
    xor       eax,ecx
    lea       ecx,[esp+0xa0]
    push      ecx
    lea       ecx,[esp+0x98]
    push      ecx
    lea       ecx,[esp+0x98]
    push      ecx
    lea       ecx,[esp+0x7c]
    push      ecx
    push      esi
    mov       dword ptr [esp+0x78],eax
    push      eax
    mov       eax,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x11c],edx
    push      eax
    push      edx
    call      csc_transF
    mov       ecx,dword ptr [esp+0x90]
    add       esp,0x00000020
    mov       dword ptr [esp+0x260],ecx
    mov       edx,dword ptr [esp+0x90]
    mov       eax,dword ptr [esp+0x94]
    mov       ecx,dword ptr [esp+0xa0]
    mov       dword ptr [esp+0x25c],edx
    mov       edx,dword ptr [esp+0x104]
    mov       dword ptr [esp+0x258],eax
    mov       eax,dword ptr [esp+0x54]
    mov       dword ptr [esp+0x254],ecx
    mov       ecx,dword ptr [esp+0x64]
    mov       dword ptr [esp+0x250],edx
    mov       edx,dword ptr [esp+0x4c]
    mov       dword ptr [esp+0x24c],eax
    mov       eax,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x248],ecx
    mov       ecx,dword ptr [esp+0x40]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x244],esi
    xor       edx,ecx
    mov       esi,eax
    mov       dword ptr [esp+0x58],edx
    mov       ecx,edx
    or        edx,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0xd4],esi
    not       ecx
    xor       edx,ecx
    mov       eax,ecx
    mov       ecx,edx
    or        eax,dword ptr [esp+0x14]
    xor       ecx,dword ptr [esp+0x28]
    xor       eax,dword ptr [esp+0x18]
    xor       ecx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x98],eax
    mov       dword ptr [esp+0x70],ecx
    mov       ecx,esi
    mov       esi,dword ptr [esp+0x48]
    xor       ecx,edx
    or        ecx,dword ptr [esp+0x1c]
    mov       edx,ecx
    xor       edx,esi
    mov       esi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x90],edx
    mov       edx,dword ptr [esp+0xd4]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x2c]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x44]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x70]
    xor       ecx,edx
    mov       edx,ecx
    mov       ecx,eax
    or        ecx,esi
    mov       esi,eax
    and       esi,edx
    mov       dword ptr [esp+0x94],edx
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x1c]
    mov       edx,ecx
    xor       edx,esi
    mov       esi,dword ptr [esp+0x90]
    mov       dword ptr [esp+0xa0],edx
    mov       edx,eax
    or        edx,esi
    mov       esi,dword ptr [esp+0x70]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x14]
    mov       dword ptr [esp+0x17c],edx
    or        edx,ecx
    xor       edx,eax
    mov       dword ptr [esp+0x110],edx
    xor       edx,esi
    mov       esi,dword ptr [esp+0x90]
    mov       dword ptr [esp+0x114],edx
    mov       edx,eax
    and       edx,esi
    mov       esi,dword ptr [esp+0x94]
    or        eax,esi
    xor       edx,eax
    mov       eax,edx
    mov       esi,dword ptr [esp+0x58]
    or        edx,ecx
    mov       ecx,dword ptr [esp+0x17c]
    not       eax
    xor       esi,eax
    mov       eax,dword ptr [esp+0x110]
    xor       edx,eax
    mov       eax,dword ptr [esp+0xd4]
    xor       edx,ecx
    lea       ecx,[esp+0x98]
    not       edx
    xor       eax,edx
    push      ecx
    lea       edx,[esp+0x98]
    lea       ecx,[esp+0x94]
    push      edx
    push      ecx
    mov       ecx,dword ptr [esp+0x120]
    lea       edx,[esp+0x7c]
    push      edx
    mov       edx,dword ptr [esp+0xb0]
    push      ecx
    push      eax
    push      edx
    push      esi
    mov       dword ptr [esp+0xf4],eax
    call      csc_transF
    mov       eax,dword ptr [esp+0x90]
    mov       edx,dword ptr [esp+0xb4]
    mov       ecx,dword ptr [esp+0xb0]
    mov       dword ptr [esp+0x200],eax
    mov       eax,dword ptr [esp+0xb8]
    mov       dword ptr [esp+0x1f8],edx
    mov       edx,dword ptr [esp+0xf4]
    mov       dword ptr [esp+0x1f4],eax
    mov       eax,dword ptr [esp+0x134]
    mov       dword ptr [esp+0x1fc],ecx
    mov       ecx,dword ptr [esp+0xc0]
    mov       dword ptr [esp+0x1e4],eax
    mov       eax,dword ptr [esp+0xec]
    mov       dword ptr [esp+0x1f0],esi
    mov       esi,dword ptr [csc_tabe+0x1a0]
    mov       dword ptr [esp+0x1e8],edx
    mov       edx,eax
    mov       dword ptr [esp+0x1ec],ecx
    mov       ecx,dword ptr [csc_tabe+0x1a4]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x128]
    add       esp,0x00000020
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0xb4]
    xor       ecx,dword ptr [csc_tabe+0x1a8]
    mov       dword ptr [esp+0x4c],edx
    mov       dword ptr [esp+0x40],ecx
    mov       ecx,dword ptr [esp+0x8c]
    xor       ecx,dword ptr [csc_tabe+0x1ac]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,eax
    or        ecx,dword ptr [esp+0xe0]
    xor       ecx,dword ptr [esp+0xa8]
    xor       ecx,dword ptr [csc_tabe+0x1b0]
    mov       dword ptr [esp+0x44],ecx
    mov       ecx,dword ptr [esp+0x3c]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x108]
    or        eax,ecx
    mov       ecx,dword ptr [esp+0x168]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x1b4]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x34],eax
    mov       eax,dword ptr [esp+0x160]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x1b8]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x194]
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esp+0x68]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x1bc]
    xor       eax,ecx
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0x5c]
    mov       ecx,eax
    xor       ecx,dword ptr [esp+0x60]
    xor       ecx,dword ptr [csc_tabe+0x180]
    mov       dword ptr [esp+0x14],ecx
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x80]
    mov       dword ptr [esp+0x3c],ecx
    xor       edx,dword ptr [csc_tabe+0x184]
    mov       dword ptr [esp+0x6c],edx
    xor       edx,esi
    mov       dword ptr [esp+0x5c],edx
    mov       edx,dword ptr [esp+0x120]
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x188]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x1c],edx
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x18c]
    mov       dword ptr [esp+0x8c],edx
    mov       edx,dword ptr [esp+0x14c]
    xor       edx,esi
    mov       dword ptr [esp+0x38],edx
    mov       esi,edx
    mov       edx,dword ptr [esp+0x18]
    xor       esi,edx
    mov       dword ptr [esp+0xcc],esi
    mov       esi,eax
    or        esi,dword ptr [esp+0x20]
    xor       esi,dword ptr [esp+0xa4]
    xor       esi,edx
    mov       edx,dword ptr [csc_tabe+0x190]
    xor       esi,edx
    mov       edx,dword ptr [esp+0x144]
    xor       eax,edx
    or        eax,dword ptr [esp+0x80]
    xor       eax,dword ptr [esp+0x10]
    xor       eax,dword ptr [csc_tabe+0x194]
    mov       dword ptr [esp+0x2c],eax
    mov       eax,dword ptr [esp+0x7c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x34]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x198]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x88]
    mov       dword ptr [esp+0x48],eax
    mov       eax,dword ptr [esp+0x24]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x19c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xcc]
    mov       dword ptr [esp+0x28],eax
    mov       eax,ecx
    not       edx
    or        eax,edx
    mov       dword ptr [esp+0x88],edx
    mov       edx,dword ptr [esp+0x44]
    xor       eax,esi
    xor       eax,edx
    mov       edx,dword ptr [esp+0xcc]
    mov       dword ptr [esp+0x20],eax
    mov       eax,dword ptr [esp+0x8c]
    or        eax,edx
    mov       edx,dword ptr [esp+0x88]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x88],eax
    xor       eax,edx
    mov       edx,dword ptr [esp+0x60]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x88]
    mov       dword ptr [esp+0x10],eax
    mov       eax,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x8c]
    or        eax,edx
    mov       dword ptr [esp+0x88],eax
    mov       edx,dword ptr [esp+0x48]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x30]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x88]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x5c]
    or        ecx,edx
    mov       edx,dword ptr [esp+0x2c]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x34]
    xor       ecx,edx
    lea       edx,[esp+0x3c]
    push      edx
    lea       edx,[esp+0x60]
    push      edx
    lea       edx,[esp+0x94]
    push      edx
    lea       edx,[esp+0xd8]
    push      edx
    mov       edx,dword ptr [esp+0x30]
    push      edx
    push      ecx
    mov       dword ptr [esp+0x15c],eax
    push      eax
    mov       eax,dword ptr [esp+0x2c]
    mov       dword ptr [esp+0xa4],ecx
    push      eax
    call      csc_transG
    mov       edx,dword ptr [esp+0xec]
    mov       eax,dword ptr [esp+0xac]
    mov       ecx,edx
    or        eax,edx
    mov       edx,dword ptr [esp+0xac]
    add       esp,0x00000020
    not       ecx
    mov       dword ptr [esp+0x194],ecx
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       ecx,eax
    or        ecx,edx
    mov       edx,dword ptr [esp+0x10]
    xor       edx,eax
    mov       eax,dword ptr [esp+0x144]
    xor       eax,ecx
    mov       dword ptr [esp+0x280],edx
    mov       dword ptr [esp+0x27c],eax
    mov       eax,dword ptr [esp+0x3c]
    mov       edx,eax
    mov       dword ptr [esp+0x264],eax
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x5c]
    or        edx,ecx
    mov       dword ptr [esp+0x268],ecx
    xor       edx,dword ptr [esp+0x88]
    mov       ecx,dword ptr [esp+0x4c]
    mov       dword ptr [esp+0x278],edx
    mov       edx,eax
    or        edx,dword ptr [esp+0x194]
    mov       eax,dword ptr [esp+0x14]
    xor       edx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x20],eax
    mov       eax,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x274],edx
    mov       edx,dword ptr [esp+0xcc]
    mov       dword ptr [esp+0x270],edx
    mov       edx,dword ptr [esp+0x8c]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x26c],edx
    mov       edx,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x5c],eax
    mov       eax,dword ptr [esp+0x38]
    xor       eax,edx
    mov       dword ptr [esp+0x70],ecx
    mov       edx,eax
    mov       dword ptr [esp+0x24],eax
    not       edx
    mov       ecx,edx
    or        ecx,dword ptr [esp+0x14]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x1c]
    mov       dword ptr [esp+0x194],ecx
    or        eax,esi
    xor       eax,edx
    mov       edx,eax
    mov       eax,dword ptr [esp+0x28]
    mov       esi,edx
    xor       esi,eax
    mov       eax,dword ptr [esp+0x30]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x1c]
    or        eax,edx
    mov       edx,eax
    xor       edx,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x14]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x5c]
    or        eax,edx
    mov       edx,dword ptr [esp+0x2c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x44]
    xor       eax,edx
    lea       edx,[esp+0x20]
    push      edx
    lea       edx,[esp+0x60]
    push      edx
    lea       edx,[esp+0x78]
    push      edx
    lea       edx,[esp+0x30]
    push      edx
    push      ecx
    mov       dword ptr [esp+0x158],eax
    push      eax
    mov       eax,dword ptr [esp+0x28]
    push      eax
    push      esi
    call      csc_transG
    mov       ecx,dword ptr [esp+0x44]
    mov       eax,dword ptr [esp+0x90]
    mov       edx,ecx
    or        eax,ecx
    mov       ecx,dword ptr [esp+0x7c]
    add       esp,0x00000020
    not       edx
    xor       eax,edx
    mov       dword ptr [esp+0x88],eax
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x70]
    or        ecx,eax
    mov       eax,dword ptr [esp+0x88]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x10]
    xor       eax,ecx
    mov       dword ptr [esp+0x200],esi
    mov       dword ptr [esp+0x1fc],eax
    mov       eax,dword ptr [esp+0x20]
    mov       esi,eax
    mov       dword ptr [esp+0x1e4],eax
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x5c]
    or        esi,ecx
    mov       dword ptr [esp+0x1e8],ecx
    xor       esi,dword ptr [esp+0x144]
    mov       dword ptr [esp+0x1f8],esi
    mov       esi,eax
    mov       eax,dword ptr [csc_tabe+0x1e0]
    or        esi,edx
    mov       edx,dword ptr [esp+0x194]
    xor       esi,edx
    mov       edx,dword ptr [esp+0x24]
    xor       edi,eax
    mov       eax,dword ptr [esp+0xf4]
    mov       dword ptr [esp+0x1f0],edx
    mov       edx,dword ptr [esp+0x70]
    mov       ecx,edi
    mov       dword ptr [esp+0x1f4],esi
    mov       dword ptr [esp+0x1ec],edx
    mov       dword ptr [esp+0x4c],ecx
    mov       edi,dword ptr [csc_tabe+0x1e4]
    mov       esi,dword ptr [csc_tabe+0x1e8]
    xor       eax,edi
    mov       edx,dword ptr [csc_tabe+0x1ec]
    mov       edi,eax
    mov       eax,dword ptr [esp+0xf8]
    xor       eax,esi
    mov       esi,eax
    mov       eax,dword ptr [esp+0xd0]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x1f0]
    mov       dword ptr [esp+0x18],eax
    mov       eax,dword ptr [esp+0xc4]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x154]
    mov       dword ptr [esp+0x44],eax
    mov       eax,dword ptr [csc_tabe+0x1f4]
    xor       edx,eax
    mov       eax,dword ptr [esp+0xd8]
    xor       eax,dword ptr [csc_tabe+0x1f8]
    mov       dword ptr [esp+0x40],esi
    mov       dword ptr [esp+0x34],edx
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esp+0xe4]
    xor       eax,dword ptr [csc_tabe+0x1fc]
    mov       dword ptr [esp+0x60],eax
    xor       ebp,eax
    xor       ebp,dword ptr [csc_tabe+0x1c0]
    mov       eax,ebp
    mov       dword ptr [esp+0x14],ebp
    mov       ebp,dword ptr [csc_tabe+0x1c4]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x68],eax
    xor       ecx,ebp
    mov       ebp,dword ptr [csc_tabe+0x1c8]
    mov       dword ptr [esp+0x6c],ecx
    xor       ecx,edi
    mov       dword ptr [esp+0x5c],ecx
    mov       ecx,dword ptr [esp+0xb8]
    xor       ecx,edi
    xor       ecx,ebp
    mov       dword ptr [esp+0x1c],ecx
    xor       ecx,esi
    mov       esi,dword ptr [csc_tabe+0x1cc]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0xe8]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x15c]
    mov       dword ptr [esp+0x38],ecx
    mov       edi,ecx
    mov       ecx,dword ptr [esp+0x18]
    mov       ebp,ecx
    xor       edi,ecx
    mov       ecx,dword ptr [csc_tabe+0x1d0]
    xor       esi,ebp
    mov       ebp,dword ptr [csc_tabe+0x1d4]
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x124]
    mov       dword ptr [esp+0x7c],edi
    xor       ecx,ebp
    mov       ebp,dword ptr [esp+0x9c]
    mov       dword ptr [esp+0x2c],ecx
    mov       ecx,dword ptr [csc_tabe+0x1d8]
    xor       ebp,edx
    mov       edx,dword ptr [csc_tabe+0x1dc]
    xor       ebp,ecx
    mov       ecx,dword ptr [esp+0x84]
    xor       ecx,edx
    mov       edx,edi
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,eax
    not       edx
    or        ecx,edx
    mov       dword ptr [esp+0x88],edx
    mov       edx,dword ptr [esp+0x44]
    xor       ecx,esi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x3c],ecx
    or        edx,edi
    mov       edi,dword ptr [esp+0x88]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x28]
    mov       dword ptr [esp+0x88],edx
    xor       edx,edi
    mov       edi,dword ptr [esp+0x60]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x88]
    mov       dword ptr [esp+0x24],edx
    mov       edx,dword ptr [esp+0x5c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x74]
    or        edx,edi
    mov       edi,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x88],edx
    xor       edx,ebp
    xor       edx,edi
    mov       edi,dword ptr [esp+0x88]
    xor       eax,edi
    mov       edi,dword ptr [esp+0x5c]
    or        eax,edi
    mov       edi,dword ptr [esp+0x2c]
    xor       eax,edi
    mov       edi,dword ptr [esp+0x34]
    xor       eax,edi
    lea       edi,[esp+0x68]
    push      edi
    lea       edi,[esp+0x60]
    push      edi
    lea       edi,[esp+0x7c]
    push      edi
    lea       edi,[esp+0x88]
    push      edi
    push      ecx
    mov       dword ptr [esp+0x34],eax
    push      eax
    mov       eax,dword ptr [esp+0x3c]
    push      edx
    push      eax
    mov       dword ptr [esp+0x30],edx
    call      csc_transG
    add       esp,0x00000020
    lea       ecx,[esp+0x3c]
    lea       edx,[esp+0x20]
    lea       eax,[esp+0x10]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x70]
    lea       ecx,[esp+0x2c]
    push      eax
    mov       eax,dword ptr [esp+0x68]
    push      ecx
    mov       ecx,dword ptr [esp+0x84]
    push      edx
    mov       edx,dword ptr [esp+0x90]
    push      eax
    push      ecx
    push      edx
    call      csc_transF
    mov       eax,dword ptr [esp+0x44]
    mov       edx,dword ptr [esp+0x40]
    mov       ecx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x2c0],eax
    mov       eax,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x2b8],edx
    mov       edx,dword ptr [esp+0x94]
    mov       dword ptr [esp+0x2bc],ecx
    mov       ecx,dword ptr [esp+0x9c]
    mov       dword ptr [esp+0x2b4],eax
    mov       eax,dword ptr [esp+0x7c]
    mov       dword ptr [esp+0x2ac],edx
    mov       edx,dword ptr [esp+0x34]
    mov       dword ptr [esp+0x2b0],ecx
    mov       ecx,dword ptr [esp+0x88]
    mov       dword ptr [esp+0x2a8],eax
    mov       eax,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x88],edx
    mov       edx,dword ptr [esp+0x6c]
    add       esp,0x00000020
    mov       dword ptr [esp+0x284],ecx
    mov       ecx,dword ptr [esp+0x40]
    xor       eax,edx
    mov       dword ptr [esp+0x74],eax
    mov       eax,dword ptr [esp+0x38]
    mov       edi,dword ptr [esp+0x1c]
    xor       eax,ecx
    mov       ecx,eax
    mov       dword ptr [esp+0x24],eax
    not       ecx
    mov       edx,ecx
    or        eax,edi
    or        edx,dword ptr [esp+0x14]
    xor       eax,ecx
    mov       dword ptr [esp+0x7c],edi
    xor       edx,esi
    mov       esi,eax
    mov       eax,dword ptr [esp+0x28]
    mov       ecx,esi
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x30]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x74]
    xor       eax,esi
    mov       dword ptr [esp+0xe0],edx
    or        eax,edi
    mov       edi,dword ptr [esp+0x74]
    mov       esi,eax
    mov       dword ptr [esp+0x10],ecx
    xor       esi,ebp
    mov       ebp,dword ptr [esp+0x14]
    xor       eax,ebp
    mov       ebp,dword ptr [esp+0x2c]
    or        eax,edi
    mov       edi,dword ptr [esp+0x44]
    xor       eax,ebp
    mov       dword ptr [esp+0x20],esi
    xor       eax,edi
    lea       edi,[esp+0x68]
    push      edi
    lea       edi,[esp+0x78]
    push      edi
    lea       edi,[esp+0x84]
    push      edi
    lea       edi,[esp+0x30]
    push      edi
    push      edx
    push      eax
    push      esi
    push      ecx
    mov       dword ptr [esp+0x5c],eax
    call      csc_transG
    add       esp,0x00000020
    lea       eax,[esp+0xe0]
    lea       ecx,[esp+0x3c]
    lea       edx,[esp+0x20]
    push      eax
    push      ecx
    mov       ecx,dword ptr [esp+0x70]
    lea       eax,[esp+0x18]
    push      edx
    mov       edx,dword ptr [esp+0x80]
    push      eax
    mov       eax,dword ptr [esp+0x8c]
    push      ecx
    mov       ecx,dword ptr [esp+0x38]
    push      edx
    push      eax
    push      ecx
    call      csc_transF
    mov       edx,dword ptr [esp+0x30]
    mov       eax,dword ptr [esp+0x40]
    mov       ecx,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x240],edx
    mov       edx,dword ptr [esp+0x100]
    mov       dword ptr [esp+0x23c],eax
    mov       eax,dword ptr [esp+0x44]
    mov       dword ptr [esp+0x238],ecx
    mov       ecx,dword ptr [esp+0x9c]
    add       esp,0x00000020
    mov       dword ptr [esp+0x214],edx
    mov       dword ptr [esp+0x210],eax
    mov       eax,dword ptr [esp+0x68]
    mov       edx,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x204],eax
    mov       eax,dword ptr [esp+0x184]
    dec       eax
    mov       dword ptr [esp+0x20c],ecx
    mov       dword ptr [esp+0x208],edx
    mov       dword ptr [esp+0x184],eax
    ljne       X$1
    mov       eax,dword ptr [esp+0x2dc]
    lea       ecx,[esp+0x224]
    mov       dword ptr [esp+0x10],ecx
    lea       ecx,[esp+0x1c4]
    mov       edi,dword ptr [esp+0x128]
    mov       esi,dword ptr [esp+0x10c]
    sub       ecx,eax
    mov       dword ptr [esp+0x68],0xffffffff
    mov       dword ptr [esp+0x88],ecx
    lea       ecx,[esp+0x1a4]
    sub       ecx,eax
    mov       dword ptr [esp+0x184],0x00000000
    lea       ebp,[eax+0xc0]
    mov       dword ptr [esp+0x144],ecx
X$4:
    mov       eax,dword ptr [esi]
    mov       ecx,dword ptr [edi]
    mov       edx,dword ptr [esi+0x4]
    xor       eax,ecx
    mov       ecx,dword ptr [edi+0x4]
    mov       dword ptr [esp+0x7c],eax
    xor       ecx,edx
    mov       edx,dword ptr [edi+0x8]
    xor       edx,dword ptr [esi+0x8]
    mov       dword ptr [esp+0x24],ecx
    mov       dword ptr [esp+0xd8],edx
    mov       edx,dword ptr [edi+0xc]
    xor       edx,dword ptr [esi+0xc]
    mov       dword ptr [esp+0xe4],edx
    mov       edx,dword ptr [edi+0x10]
    xor       edx,dword ptr [esi+0x10]
    mov       dword ptr [esp+0xe8],edx
    mov       edx,dword ptr [edi+0x14]
    xor       edx,dword ptr [esi+0x14]
    mov       dword ptr [esp+0x9c],edx
    mov       edx,dword ptr [edi+0x18]
    xor       edx,dword ptr [esi+0x18]
    mov       dword ptr [esp+0x84],edx
    mov       edx,dword ptr [edi+0x1c]
    xor       edx,dword ptr [esi+0x1c]
    mov       dword ptr [esp+0x10c],edx
    lea       edx,[esp+0xe8]
    push      edx
    lea       edx,[esp+0xa0]
    push      edx
    lea       edx,[esp+0x8c]
    push      edx
    lea       edx,[esp+0x118]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0xec]
    push      ecx
    mov       ecx,dword ptr [esp+0xfc]
    push      eax
    push      ecx
    call      csc_transF
    add       esp,0x00000020
    lea       edx,[esp+0x7c]
    lea       eax,[esp+0x24]
    lea       ecx,[esp+0xd8]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0xf0]
    lea       edx,[esp+0xec]
    push      ecx
    mov       ecx,dword ptr [esp+0xa8]
    push      edx
    mov       edx,dword ptr [esp+0x94]
    push      eax
    mov       eax,dword ptr [esp+0x120]
    push      ecx
    push      edx
    push      eax
    call      csc_transG
    add       esp,0x00000020
    lea       ecx,[esp+0xe8]
    lea       edx,[esp+0x9c]
    lea       eax,[esp+0x84]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x84]
    lea       ecx,[esp+0x114]
    push      eax
    mov       eax,dword ptr [esp+0x30]
    push      ecx
    mov       ecx,dword ptr [esp+0xe8]
    push      edx
    mov       edx,dword ptr [esp+0xf8]
    push      eax
    push      ecx
    push      edx
    call      csc_transF
    mov       eax,dword ptr [esp+0x12c]
    mov       ecx,dword ptr [esp+0xa4]
    add       esp,0x00000020
    mov       dword ptr [ebx+0xe0],eax
    mov       edx,dword ptr [esp+0x9c]
    mov       eax,dword ptr [esp+0xe8]
    mov       dword ptr [ebx+0xc0],ecx
    mov       ecx,dword ptr [esp+0xe4]
    mov       dword ptr [ebx+0xa0],edx
    mov       edx,dword ptr [esp+0xd8]
    mov       dword ptr [ebx+0x60],ecx
    mov       ecx,dword ptr [esp+0x7c]
    mov       dword ptr [ebx+0x40],edx
    mov       edx,dword ptr [esp+0x88]
    mov       dword ptr [ebx+0x80],eax
    mov       eax,dword ptr [esp+0x24]
    mov       dword ptr [ebx],ecx
    mov       ecx,dword ptr [edx+ebp]
    mov       dword ptr [ebx+0x20],eax
    mov       edx,dword ptr [ebp+0x20]
    mov       eax,dword ptr [ebx-0x120]
    xor       ecx,edx
    mov       edx,dword ptr [ebx+0xe0]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x68]
    xor       ecx,edx
    not       ecx
    and       eax,ecx
    lje        X$5
    mov       edx,dword ptr [esp+0x144]
    mov       ecx,dword ptr [edx+ebp]
    mov       edx,dword ptr [ebx-0x140]
    xor       ecx,edx
    mov       edx,dword ptr [ebp]
    xor       ecx,edx
    mov       edx,dword ptr [ebx+0xc0]
    xor       ecx,edx
    not       ecx
    and       eax,ecx
    lje        X$5
    mov       ecx,dword ptr [esp+0x10]
    mov       edx,dword ptr [ebp-0x20]
    xor       edx,dword ptr [ecx+0x20]
    xor       edx,dword ptr [ebx-0x160]
    xor       edx,dword ptr [ebx+0xa0]
    not       edx
    and       eax,edx
    lje        X$5
    mov       edx,dword ptr [ebp-0x40]
    xor       edx,dword ptr [ebx-0x180]
    xor       edx,dword ptr [ecx]
    xor       edx,dword ptr [ebx+0x80]
    not       edx
    and       eax,edx
    lje        X$5
    mov       edx,dword ptr [ebp-0x60]
    xor       edx,dword ptr [ecx-0x20]
    xor       edx,dword ptr [ebx-0x1a0]
    xor       edx,dword ptr [ebx+0x60]
    not       edx
    and       eax,edx
    lje        X$5
    mov       edx,dword ptr [ecx-0x40]
    xor       edx,dword ptr [ebp-0x80]
    xor       edx,dword ptr [ebx-0x1c0]
    xor       edx,dword ptr [ebx+0x40]
    not       edx
    and       eax,edx
    je        X$5
    mov       edx,dword ptr [ecx-0x60]
    xor       edx,dword ptr [ebp-0xa0]
    xor       edx,dword ptr [ebx-0x1e0]
    xor       edx,dword ptr [ebx+0x20]
    not       edx
    and       eax,edx
    je        X$5
    mov       edx,dword ptr [ecx-0x80]
    xor       edx,dword ptr [ebp-0xc0]
    xor       edx,dword ptr [ebx-0x200]
    xor       edx,dword ptr [ebx]
    not       edx
    and       eax,edx
    mov       dword ptr [esp+0x68],eax
    je        X$5
    mov       eax,dword ptr [esp+0x184]
    add       ebp,0x00000004
    inc       eax
    add       ecx,0x00000004
    add       esi,0x00000020
    add       edi,0x00000020
    add       ebx,0x00000004
    cmp       eax,0x00000008
    mov       dword ptr [esp+0x184],eax
    mov       dword ptr [esp+0x10],ecx
    ljl        X$4
    mov       eax,dword ptr [esp+0x68]
X$5:
    pop       edi
    pop       esi
    pop       ebp
    pop       ebx
    add       esp,0x000002c0
    ret       

__CODESECT__
    align 32
csc_transF:
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
    align 32
csc_transG:
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
    align 32
csc_unit_func_1k_i:
_csc_unit_func_1k_i:
    mov       eax,dword ptr [esp+0xc]
    sub       esp,0x00000020
    test      al,0x01
    push      ebx
    push      ebp
    push      esi
    push      edi
    je        X$6
    add       eax,0x0000000f
    and       al,0xf0
X$6:
    mov       edi,dword ptr [esp+0x34]
    mov       esi,eax
    add       eax,0x00000200
    lea       edx,[esp+0x3c]
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
    mov       dword ptr [esp+0x44],ecx
    call      convert_key_from_inc_to_csc
    mov       ecx,dword ptr [esp+0x44]
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
X$7:
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
    jne       X$8
    mov       eax,dword ptr [esp+0x34]
    mov       edx,dword ptr [eax]
    mov       edi,dword ptr [eax+0x8]
    mov       eax,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x14],eax
    mov       eax,0x00000001
X$8:
    mov       ebx,dword ptr [esp+0x18]
    add       ecx,0x00000004
    dec       ebx
    mov       dword ptr [esp+0x18],ebx
    jne       X$7
    mov       ecx,0x00000040
    xor       eax,eax
    lea       edi,[esi+0x100]
    repe stosd 
    mov       ecx,dword ptr [csc_bit_order]
    mov       dword ptr [esi+ecx*4],0xaaaaaaaa
    mov       edx,dword ptr [csc_bit_order+0x4]
    mov       dword ptr [esi+edx*4],0xcccccccc
    mov       eax,dword ptr [csc_bit_order+0x8]
    mov       dword ptr [esi+eax*4],0xf0f0f0f0
    mov       ecx,dword ptr [csc_bit_order+0xc]
    mov       eax,dword ptr [esp+0x38]
    mov       dword ptr [esi+ecx*4],0xff00ff00
    mov       edx,dword ptr [csc_bit_order+0x10]
    mov       ecx,0x00000005
    mov       dword ptr [esi+edx*4],0xffff0000
    mov       eax,dword ptr [eax]
    cmp       eax,0x00000020
    mov       dword ptr [esp+0x18],ecx
    jbe       X$10
X$9:
    inc       ecx
    mov       edx,0x00000001
    shl       edx,cl
    cmp       eax,edx
    ja        X$9
    mov       dword ptr [esp+0x18],ecx
X$10:
    add       ecx,0xfffffffb
    test      ecx,ecx
    mov       dword ptr [esp+0x24],ecx
    jbe       X$12
    mov       eax,offset csc_bit_order+0x14
X$11:
    mov       edx,dword ptr [eax]
    add       eax,0x00000004
    dec       ecx
    mov       dword ptr [esi+edx*4],0x00000000
    jne       X$11
X$12:
    mov       edi,dword ptr [esp+0x28]
    mov       eax,dword ptr [esp+0x1c]
    mov       ebp,dword ptr [esp+0x2c]
    push      edi
    push      eax
    push      ebp
    push      esi
    xor       ebx,ebx
    call      cscipher_bitslicer_1k_i
    add       esp,0x00000010
    test      eax,eax
    jne       X$17
    mov       ecx,dword ptr [esp+0x24]
    mov       edx,0x00000001
    shl       edx,cl
    mov       dword ptr [esp+0x2c],edx
    jmp       X$14
X$13:
    mov       edx,dword ptr [esp+0x2c]
X$14:
    inc       ebx
    cmp       ebx,edx
    jae       X$17
    xor       ecx,ecx
    test      bl,0x01
    jne       X$16
X$15:
    inc       ecx
    mov       edx,0x00000001
    shl       edx,cl
    test      ebx,edx
    je        X$15
X$16:
    mov       eax,dword ptr [ecx*4+csc_bit_order+0x14]
    mov       edx,dword ptr [esp+0x1c]
    mov       ecx,dword ptr [esi+eax*4]
    push      edi
    lea       eax,[esi+eax*4]
    push      edx
    not       ecx
    push      ebp
    push      esi
    mov       dword ptr [eax],ecx
    call      cscipher_bitslicer_1k_i
    add       esp,0x00000010
    test      eax,eax
    je        X$13
X$17:
    xor       ecx,ecx
    cmp       eax,ecx
    lje        X$24
    xor       edx,edx
    cmp       eax,0x00000001
    je        X$19
X$18:
    shr       eax,0x00000001
    inc       edx
    cmp       eax,0x00000001
    jne       X$18
X$19:
    mov       dword ptr [esp+0x10],ecx
    mov       dword ptr [esp+0x3c],ecx
    mov       eax,0x00000008
    add       esi,0x00000020
X$20:
    mov       edi,dword ptr [esi]
    cmp       eax,0x00000020
    mov       ecx,edx
    jge       X$21
    shr       edi,cl
    mov       ecx,eax
    and       edi,0x00000001
    shl       edi,cl
    or        dword ptr [esp+0x3c],edi
    jmp       X$22
X$21:
    shr       edi,cl
    lea       ecx,[eax-0x20]
    and       edi,0x00000001
    shl       edi,cl
    or        dword ptr [esp+0x10],edi
X$22:
    inc       eax
    add       esi,0x00000004
    cmp       eax,0x00000040
    jl        X$20
    lea       eax,[esp+0x3c]
    lea       ecx,[esp+0x10]
    push      eax
    push      ecx
    call      convert_key_from_csc_to_inc
    mov       edx,dword ptr [esp+0x3c]
    mov       ecx,dword ptr [esp+0x44]
    add       esp,0x00000008
    mov       eax,dword ptr [edx+0x14]
    cmp       ecx,eax
    jae       X$23
    mov       esi,dword ptr [esp+0x38]
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
    add       esp,0x00000020
    ret       
X$23:
    mov       esi,ecx
    sub       esi,eax
    mov       eax,dword ptr [esp+0x38]
    mov       dword ptr [eax],esi
    mov       dword ptr [edx+0x14],ecx
    mov       ecx,dword ptr [esp+0x10]
    mov       eax,0x00000002
    mov       dword ptr [edx+0x10],ecx
    pop       edi
    pop       esi
    pop       ebp
    pop       ebx
    add       esp,0x00000020
    ret       
X$24:
    mov       ecx,dword ptr [esp+0x18]
    mov       edx,dword ptr [esp+0x38]
    mov       eax,0x00000001
    shl       eax,cl
    mov       ecx,dword ptr [esp+0x34]
    mov       dword ptr [edx],eax
    mov       esi,dword ptr [ecx+0x14]
    add       esi,eax
    mov       edx,esi
    mov       dword ptr [ecx+0x14],esi
    cmp       edx,eax
    jae       X$25
    inc       dword ptr [ecx+0x10]
X$25:
    pop       edi
    pop       esi
    pop       ebp
    mov       eax,0x00000001
    pop       ebx
    add       esp,0x00000020
    ret       

__CODESECT__
    align 32

