; Copyright distributed.net 1997 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; $Log: csc-1k.asm,v $
; Revision 1.1.2.2  1999/11/07 01:31:17  remi
; Increased code alignment.
;
; Revision 1.1.2.1  1999/11/06 00:26:14  cyp
; they're here! (see also bench.res for 'ideal' combination)
;
;

global         csc_unit_func_1k,_csc_unit_func_1k

extern         csc_tabc,csc_tabe,csc_bit_order,csc_transP
extern         convert_key_from_inc_to_csc, convert_key_from_csc_to_inc

%include "csc-mac.inc"

__DATASECT__
    db  "@(#)$Id: csc-1k.asm,v 1.1.2.2 1999/11/07 01:31:17 remi Exp $",0

__CODESECT__
   align 32
cscipher_bitslicer_1k:
    sub       esp,0x00000140
    mov       edx,dword ptr [esp+0x144]
    push      ebx
    mov       ebx,dword ptr [esp+0x154]
    push      ebp
    push      esi
    push      edi
    lea       esi,[edx+0x100]
    mov       ecx,0x00000040
    mov       edi,ebx
    lea       eax,[ebx+0x100]
    repe movsd 
    mov       ecx,0x00000040
    mov       esi,edx
    mov       edi,eax
    add       ebx,0x00000200
    repe movsd 
    mov       esi,dword ptr [esp+0x158]
    mov       ecx,0x00000040
    lea       edi,[esp+0x48]
    mov       ebp,offset csc_tabc
    repe movsd 
    mov       dword ptr [esp+0x44],eax
    mov       dword ptr [esp+0x148],0x00000008
    lea       esi,[ebx+0x40]
    mov       edi,0x00000008
    jmp       X$3
X$1:
    mov       ebp,dword ptr [esp+0x14c]
    mov       eax,dword ptr [esp+0x44]
    lea       esi,[ebx+0x40]
    mov       edi,0x00000008
    jmp       X$3
X$2:
    mov       eax,dword ptr [esp+0x44]
X$3:
    lea       ecx,[esi-0x20]
    push      ebx
    push      ecx
    lea       edx,[esi+0x20]
    push      esi
    push      edx
    lea       ecx,[esi+0x40]
    lea       edx,[esi+0x60]
    push      ecx
    push      edx
    lea       ecx,[esi+0x80]
    lea       edx,[esi+0xa0]
    push      ecx
    mov       ecx,dword ptr [ebp]
    push      edx
    mov       edx,dword ptr [eax]
    xor       ecx,edx
    mov       edx,dword ptr [ebp+0x4]
    push      ecx
    mov       ecx,dword ptr [eax+0x4]
    xor       edx,ecx
    mov       ecx,dword ptr [eax+0x8]
    push      edx
    mov       edx,dword ptr [ebp+0x8]
    xor       ecx,edx
    mov       edx,dword ptr [eax+0xc]
    push      ecx
    mov       ecx,dword ptr [ebp+0xc]
    xor       edx,ecx
    mov       ecx,dword ptr [ebp+0x10]
    push      edx
    mov       edx,dword ptr [eax+0x10]
    xor       ecx,edx
    mov       edx,dword ptr [ebp+0x14]
    push      ecx
    mov       ecx,dword ptr [eax+0x14]
    xor       edx,ecx
    mov       ecx,dword ptr [eax+0x18]
    push      edx
    mov       edx,dword ptr [ebp+0x18]
    xor       ecx,edx
    mov       edx,dword ptr [eax+0x1c]
    push      ecx
    mov       ecx,dword ptr [ebp+0x1c]
    xor       edx,ecx
    push      edx
    call      csc_transP
    mov       eax,dword ptr [esp+0x84]
    add       esp,0x00000040
    add       ebp,0x00000020
    add       eax,0x00000020
    add       ebx,0x00000004
    add       esi,0x00000004
    dec       edi
    mov       dword ptr [esp+0x44],eax
    ljne       X$2
    mov       eax,dword ptr [ebx-0x200]
    mov       edx,dword ptr [ebx]
    mov       ecx,dword ptr [esp+0x68]
    sub       ebx,0x00000020
    xor       edx,eax
    mov       edi,dword ptr [esp+0x6c]
    mov       eax,edx
    mov       dword ptr [esp+0x14c],ebp
    mov       ebp,dword ptr [ebx+0x24]
    xor       eax,ecx
    mov       ecx,dword ptr [ebx-0x1dc]
    mov       esi,dword ptr [ebx+0x28]
    xor       ebp,ecx
    mov       dword ptr [esp+0x28],eax
    mov       eax,ebp
    mov       ecx,dword ptr [esp+0x70]
    xor       eax,edi
    mov       dword ptr [ebx+0x20],edx
    mov       edx,eax
    mov       eax,dword ptr [ebx-0x1d8]
    xor       esi,eax
    mov       edi,dword ptr [esp+0x74]
    mov       eax,esi
    mov       dword ptr [ebx+0x24],ebp
    mov       ebp,dword ptr [ebx+0x2c]
    xor       eax,ecx
    mov       ecx,dword ptr [ebx-0x1d4]
    mov       dword ptr [esp+0x24],eax
    xor       ebp,ecx
    mov       dword ptr [ebx+0x28],esi
    mov       esi,dword ptr [ebx+0x30]
    mov       eax,ebp
    xor       eax,edi
    mov       edi,dword ptr [ebx+0x34]
    mov       ecx,eax
    mov       eax,dword ptr [ebx-0x1d0]
    xor       esi,eax
    mov       dword ptr [ebx+0x2c],ebp
    mov       ebp,dword ptr [esp+0x78]
    mov       eax,esi
    xor       eax,ebp
    mov       ebp,dword ptr [ebx+0x38]
    mov       dword ptr [esp+0x2c],eax
    mov       eax,dword ptr [ebx-0x1cc]
    xor       edi,eax
    mov       dword ptr [ebx+0x30],esi
    mov       esi,dword ptr [esp+0x7c]
    mov       eax,edi
    xor       eax,esi
    mov       esi,dword ptr [ebx+0x3c]
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [ebx-0x1c8]
    xor       ebp,eax
    mov       dword ptr [ebx+0x34],edi
    mov       edi,dword ptr [esp+0x80]
    mov       eax,ebp
    xor       eax,edi
    mov       edi,dword ptr [ebx]
    mov       dword ptr [esp+0x20],eax
    mov       eax,dword ptr [ebx-0x1c4]
    xor       esi,eax
    mov       dword ptr [ebx+0x38],ebp
    mov       ebp,dword ptr [esp+0x84]
    mov       eax,esi
    xor       eax,ebp
    mov       ebp,dword ptr [esp+0x48]
    mov       dword ptr [ebx+0x3c],esi
    mov       esi,eax
    mov       eax,dword ptr [ebx-0x200]
    mov       dword ptr [esp+0x10],esi
    xor       edi,eax
    mov       dword ptr [esp+0x3c],ecx
    mov       eax,edi
    mov       dword ptr [ebx],edi
    xor       eax,ebp
    xor       eax,esi
    mov       esi,dword ptr [ebx+0x4]
    mov       dword ptr [esp+0x34],eax
    mov       eax,dword ptr [ebx-0x1fc]
    xor       esi,eax
    mov       dword ptr [ebx+0x4],esi
    mov       eax,dword ptr [esp+0x4c]
    mov       ebp,dword ptr [ebx+0x8]
    mov       edi,dword ptr [esp+0x50]
    xor       esi,eax
    mov       eax,dword ptr [ebx-0x1f8]
    xor       ebp,eax
    mov       eax,ebp
    mov       dword ptr [ebx+0x8],ebp
    xor       eax,edi
    mov       edi,dword ptr [ebx+0xc]
    xor       eax,edx
    mov       ebp,dword ptr [ebx+0x10]
    mov       dword ptr [esp+0x38],eax
    mov       eax,dword ptr [ebx-0x1f4]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x54]
    mov       dword ptr [ebx+0xc],edi
    xor       edi,eax
    mov       eax,dword ptr [ebx-0x1f0]
    xor       ebp,eax
    mov       dword ptr [ebx+0x10],ebp
    mov       eax,ebp
    mov       ebp,dword ptr [esp+0x58]
    xor       eax,ebp
    mov       ebp,dword ptr [ebx+0x14]
    xor       eax,ecx
    mov       ecx,dword ptr [ebx-0x1ec]
    mov       dword ptr [esp+0x1c],eax
    mov       eax,dword ptr [esp+0x5c]
    xor       ebp,ecx
    mov       ecx,dword ptr [ebx+0x18]
    mov       dword ptr [ebx+0x14],ebp
    xor       ebp,eax
    mov       eax,dword ptr [ebx-0x1e8]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x60]
    mov       dword ptr [ebx+0x18],ecx
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x30]
    xor       ecx,eax
    mov       eax,dword ptr [ebx+0x1c]
    mov       dword ptr [esp+0x14],ecx
    mov       ecx,dword ptr [ebx-0x1e4]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x64]
    mov       dword ptr [ebx+0x1c],eax
    xor       eax,ecx
    mov       dword ptr [esp+0x18],eax
    lea       eax,[esp+0x68]
    lea       ecx,[esp+0x6c]
    push      eax
    push      ecx
    lea       eax,[esp+0x78]
    lea       ecx,[esp+0x7c]
    push      eax
    push      ecx
    lea       eax,[esp+0x88]
    lea       ecx,[esp+0x8c]
    push      eax
    push      ecx
    lea       eax,[esp+0x98]
    lea       ecx,[esp+0x9c]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      ecx
    mov       ecx,dword ptr [esp+0x48]
    xor       eax,ecx
    mov       ecx,esi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    mov       eax,edi
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x68]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x5c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x44]
    push      ecx
    mov       ecx,dword ptr [esp+0x64]
    mov       edx,ebp
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x4c]
    push      edx
    mov       edx,dword ptr [esp+0x58]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x48]
    lea       eax,[esp+0x4c]
    lea       ecx,[esp+0x50]
    push      edx
    push      eax
    lea       edx,[esp+0x5c]
    push      ecx
    lea       eax,[esp+0x64]
    push      edx
    lea       ecx,[esp+0x6c]
    push      eax
    lea       edx,[esp+0x74]
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    push      edx
    mov       edx,dword ptr [esp+0x44]
    lea       eax,[esp+0x80]
    push      eax
    mov       eax,dword ptr [esp+0x3c]
    xor       esi,edx
    mov       edx,dword ptr [esp+0x58]
    push      ecx
    push      esi
    mov       esi,dword ptr [esp+0x4c]
    mov       ecx,dword ptr [esp+0x3c]
    xor       edi,esi
    push      edx
    mov       edx,dword ptr [esp+0x4c]
    push      edi
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x4c]
    push      ebp
    xor       eax,edx
    push      ecx
    push      eax
    call      csc_transP
    mov       edx,dword ptr [ebx-0x1a0]
    mov       esi,dword ptr [ebx+0x60]
    mov       ecx,dword ptr [esp+0xe8]
    add       ebx,0x00000040
    xor       esi,edx
    mov       edi,dword ptr [esp+0xec]
    mov       ebp,dword ptr [ebx+0x24]
    mov       eax,esi
    xor       eax,ecx
    mov       ecx,dword ptr [ebx-0x1d8]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [ebx-0x1dc]
    xor       ebp,eax
    mov       dword ptr [ebx+0x20],esi
    mov       esi,dword ptr [ebx+0x28]
    mov       eax,ebp
    xor       eax,edi
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0xf0]
    mov       edx,eax
    mov       eax,esi
    add       esp,0x00000040
    xor       eax,ecx
    mov       dword ptr [ebx+0x24],ebp
    mov       dword ptr [esp+0x24],eax
    mov       eax,dword ptr [ebx-0x1d4]
    mov       dword ptr [ebx+0x28],esi
    mov       ebp,dword ptr [ebx+0x2c]
    mov       edi,dword ptr [esp+0xb4]
    mov       esi,dword ptr [ebx+0x30]
    xor       ebp,eax
    mov       eax,ebp
    mov       dword ptr [ebx+0x2c],ebp
    mov       ebp,dword ptr [esp+0xb8]
    xor       eax,edi
    mov       ecx,eax
    mov       eax,dword ptr [ebx-0x1d0]
    mov       edi,dword ptr [ebx+0x34]
    xor       esi,eax
    mov       eax,esi
    mov       dword ptr [ebx+0x30],esi
    mov       esi,dword ptr [esp+0xbc]
    xor       eax,ebp
    mov       dword ptr [esp+0x2c],eax
    mov       eax,dword ptr [ebx-0x1cc]
    mov       ebp,dword ptr [ebx+0x38]
    xor       edi,eax
    mov       eax,edi
    mov       dword ptr [ebx+0x34],edi
    mov       edi,dword ptr [esp+0xc0]
    xor       eax,esi
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [ebx-0x1c8]
    mov       esi,dword ptr [ebx+0x3c]
    xor       ebp,eax
    mov       eax,ebp
    mov       dword ptr [ebx+0x38],ebp
    mov       ebp,dword ptr [esp+0xc4]
    xor       eax,edi
    mov       dword ptr [esp+0x20],eax
    mov       eax,dword ptr [ebx-0x1c4]
    mov       edi,dword ptr [ebx]
    xor       esi,eax
    mov       eax,esi
    mov       dword ptr [ebx+0x3c],esi
    xor       eax,ebp
    mov       ebp,dword ptr [esp+0x88]
    mov       esi,eax
    mov       eax,dword ptr [ebx-0x200]
    xor       edi,eax
    mov       dword ptr [esp+0x10],esi
    mov       eax,edi
    mov       dword ptr [ebx],edi
    xor       eax,ebp
    mov       ebp,dword ptr [ebx+0x8]
    xor       eax,esi
    mov       esi,dword ptr [ebx+0x4]
    mov       dword ptr [esp+0x34],eax
    mov       eax,dword ptr [ebx-0x1fc]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x8c]
    mov       edi,dword ptr [esp+0x90]
    mov       dword ptr [ebx+0x4],esi
    xor       esi,eax
    mov       eax,dword ptr [ebx-0x1f8]
    xor       ebp,eax
    mov       dword ptr [esp+0x3c],ecx
    mov       eax,ebp
    mov       dword ptr [ebx+0x8],ebp
    xor       eax,edi
    mov       edi,dword ptr [ebx+0xc]
    xor       eax,edx
    mov       ebp,dword ptr [ebx+0x10]
    mov       dword ptr [esp+0x38],eax
    mov       eax,dword ptr [ebx-0x1f4]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x94]
    mov       dword ptr [ebx+0xc],edi
    xor       edi,eax
    mov       eax,dword ptr [ebx-0x1f0]
    xor       ebp,eax
    mov       dword ptr [ebx+0x10],ebp
    mov       eax,ebp
    xor       eax,dword ptr [esp+0x98]
    xor       eax,ecx
    mov       ecx,dword ptr [ebx-0x1ec]
    mov       dword ptr [esp+0x1c],eax
    mov       ebp,dword ptr [ebx+0x14]
    mov       eax,dword ptr [esp+0x9c]
    xor       ebp,ecx
    mov       ecx,dword ptr [ebx+0x18]
    mov       dword ptr [ebx+0x14],ebp
    xor       ebp,eax
    mov       eax,dword ptr [ebx-0x1e8]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0xa0]
    mov       dword ptr [ebx+0x18],ecx
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x30]
    xor       ecx,eax
    mov       eax,dword ptr [ebx+0x1c]
    mov       dword ptr [esp+0x14],ecx
    mov       ecx,dword ptr [ebx-0x1e4]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xa4]
    mov       dword ptr [ebx+0x1c],eax
    xor       eax,ecx
    mov       dword ptr [esp+0x18],eax
    lea       eax,[esp+0xa8]
    lea       ecx,[esp+0xac]
    push      eax
    push      ecx
    lea       eax,[esp+0xb8]
    lea       ecx,[esp+0xbc]
    push      eax
    push      ecx
    lea       eax,[esp+0xc8]
    lea       ecx,[esp+0xcc]
    push      eax
    push      ecx
    lea       eax,[esp+0xd8]
    lea       ecx,[esp+0xdc]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      ecx
    mov       ecx,dword ptr [esp+0x48]
    xor       eax,ecx
    mov       ecx,esi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    mov       eax,edi
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x68]
    xor       eax,edx
    mov       edx,ebp
    push      eax
    mov       eax,dword ptr [esp+0x5c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x44]
    push      ecx
    mov       ecx,dword ptr [esp+0x64]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x4c]
    push      edx
    mov       edx,dword ptr [esp+0x58]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x88]
    lea       eax,[esp+0x8c]
    lea       ecx,[esp+0x90]
    push      edx
    push      eax
    lea       edx,[esp+0x9c]
    push      ecx
    lea       eax,[esp+0xa4]
    push      edx
    push      eax
    lea       ecx,[esp+0xb0]
    lea       edx,[esp+0xb4]
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    push      edx
    mov       edx,dword ptr [esp+0x44]
    lea       eax,[esp+0xc0]
    push      eax
    mov       eax,dword ptr [esp+0x3c]
    xor       esi,edx
    mov       edx,dword ptr [esp+0x58]
    push      ecx
    push      esi
    mov       esi,dword ptr [esp+0x4c]
    mov       ecx,dword ptr [esp+0x3c]
    xor       edi,esi
    push      edx
    mov       edx,dword ptr [esp+0x4c]
    push      edi
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x4c]
    push      ebp
    xor       eax,edx
    push      ecx
    push      eax
    call      csc_transP
    mov       edx,dword ptr [ebx-0x1a0]
    mov       esi,dword ptr [ebx+0x60]
    mov       ecx,dword ptr [esp+0x128]
    add       ebx,0x00000040
    xor       esi,edx
    mov       edi,dword ptr [esp+0x12c]
    mov       ebp,dword ptr [ebx+0x24]
    mov       eax,esi
    xor       eax,ecx
    mov       ecx,dword ptr [ebx-0x1d8]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [ebx-0x1dc]
    xor       ebp,eax
    mov       dword ptr [ebx+0x20],esi
    mov       esi,dword ptr [ebx+0x28]
    mov       eax,ebp
    xor       eax,edi
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x130]
    mov       edx,eax
    mov       eax,esi
    mov       edi,dword ptr [esp+0x134]
    xor       eax,ecx
    mov       dword ptr [ebx+0x24],ebp
    mov       ebp,dword ptr [ebx+0x2c]
    mov       dword ptr [esp+0x64],eax
    mov       eax,dword ptr [ebx-0x1d4]
    mov       dword ptr [ebx+0x28],esi
    mov       esi,dword ptr [ebx+0x30]
    xor       ebp,eax
    mov       eax,ebp
    mov       dword ptr [ebx+0x2c],ebp
    mov       ebp,dword ptr [esp+0x138]
    xor       eax,edi
    mov       ecx,eax
    mov       eax,dword ptr [ebx-0x1d0]
    mov       edi,dword ptr [ebx+0x34]
    xor       esi,eax
    mov       eax,esi
    mov       dword ptr [ebx+0x30],esi
    mov       esi,dword ptr [esp+0x13c]
    xor       eax,ebp
    mov       dword ptr [esp+0x6c],eax
    mov       eax,dword ptr [ebx-0x1cc]
    mov       ebp,dword ptr [ebx+0x38]
    xor       edi,eax
    mov       eax,edi
    add       esp,0x00000040
    xor       eax,esi
    mov       dword ptr [esp+0x3c],ecx
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [ebx-0x1c8]
    mov       dword ptr [ebx+0x34],edi
    xor       ebp,eax
    mov       edi,dword ptr [esp+0x100]
    mov       esi,dword ptr [ebx+0x3c]
    mov       eax,ebp
    mov       dword ptr [ebx+0x38],ebp
    mov       ebp,dword ptr [esp+0x104]
    xor       eax,edi
    mov       dword ptr [esp+0x20],eax
    mov       eax,dword ptr [ebx-0x1c4]
    mov       edi,dword ptr [ebx]
    xor       esi,eax
    mov       eax,esi
    mov       dword ptr [ebx+0x3c],esi
    xor       eax,ebp
    mov       ebp,dword ptr [esp+0xc8]
    mov       esi,eax
    mov       eax,dword ptr [ebx-0x200]
    xor       edi,eax
    mov       dword ptr [esp+0x10],esi
    mov       eax,edi
    mov       dword ptr [ebx],edi
    xor       eax,ebp
    mov       ebp,dword ptr [ebx+0x8]
    xor       eax,esi
    mov       esi,dword ptr [ebx+0x4]
    mov       dword ptr [esp+0x34],eax
    mov       eax,dword ptr [ebx-0x1fc]
    xor       esi,eax
    mov       eax,dword ptr [esp+0xcc]
    mov       edi,dword ptr [esp+0xd0]
    mov       dword ptr [ebx+0x4],esi
    xor       esi,eax
    mov       eax,dword ptr [ebx-0x1f8]
    xor       ebp,eax
    mov       eax,ebp
    mov       dword ptr [ebx+0x8],ebp
    xor       eax,edi
    mov       edi,dword ptr [ebx+0xc]
    xor       eax,edx
    mov       ebp,dword ptr [ebx+0x10]
    mov       dword ptr [esp+0x38],eax
    mov       eax,dword ptr [ebx-0x1f4]
    xor       edi,eax
    mov       eax,dword ptr [esp+0xd4]
    mov       dword ptr [ebx+0xc],edi
    xor       edi,eax
    mov       eax,dword ptr [ebx-0x1f0]
    xor       ebp,eax
    mov       dword ptr [ebx+0x10],ebp
    mov       eax,ebp
    mov       ebp,dword ptr [esp+0xd8]
    xor       eax,ebp
    mov       ebp,dword ptr [ebx+0x14]
    xor       eax,ecx
    mov       ecx,dword ptr [ebx-0x1ec]
    mov       dword ptr [esp+0x1c],eax
    mov       eax,dword ptr [esp+0xdc]
    xor       ebp,ecx
    mov       ecx,dword ptr [ebx+0x18]
    mov       dword ptr [ebx+0x14],ebp
    xor       ebp,eax
    mov       eax,dword ptr [ebx-0x1e8]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0xe0]
    mov       dword ptr [ebx+0x18],ecx
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x30]
    xor       ecx,eax
    mov       eax,dword ptr [ebx+0x1c]
    mov       dword ptr [esp+0x14],ecx
    mov       ecx,dword ptr [ebx-0x1e4]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xe4]
    mov       dword ptr [ebx+0x1c],eax
    xor       eax,ecx
    mov       dword ptr [esp+0x18],eax
    lea       eax,[esp+0xe8]
    push      eax
    lea       ecx,[esp+0xf0]
    lea       eax,[esp+0xf4]
    push      ecx
    push      eax
    lea       ecx,[esp+0x100]
    lea       eax,[esp+0x104]
    push      ecx
    lea       ecx,[esp+0x10c]
    push      eax
    push      ecx
    lea       eax,[esp+0x118]
    lea       ecx,[esp+0x11c]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      ecx
    mov       ecx,dword ptr [esp+0x48]
    xor       eax,ecx
    mov       ecx,esi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    mov       eax,edi
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x68]
    xor       eax,edx
    mov       edx,ebp
    push      eax
    mov       eax,dword ptr [esp+0x5c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x44]
    push      ecx
    mov       ecx,dword ptr [esp+0x64]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x4c]
    push      edx
    mov       edx,dword ptr [esp+0x58]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0xc8]
    lea       eax,[esp+0xcc]
    lea       ecx,[esp+0xd0]
    push      edx
    push      eax
    lea       edx,[esp+0xdc]
    push      ecx
    lea       eax,[esp+0xe4]
    push      edx
    lea       ecx,[esp+0xec]
    push      eax
    lea       edx,[esp+0xf4]
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    push      edx
    mov       edx,dword ptr [esp+0x44]
    lea       eax,[esp+0x100]
    push      eax
    mov       eax,dword ptr [esp+0x3c]
    xor       esi,edx
    mov       edx,dword ptr [esp+0x58]
    push      ecx
    push      esi
    mov       esi,dword ptr [esp+0x4c]
    mov       ecx,dword ptr [esp+0x3c]
    xor       edi,esi
    push      edx
    mov       edx,dword ptr [esp+0x4c]
    push      edi
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x4c]
    push      ebp
    xor       eax,edx
    push      ecx
    push      eax
    call      csc_transP
    mov       edx,dword ptr [ebx-0x1a0]
    mov       esi,dword ptr [ebx+0x60]
    mov       ecx,dword ptr [esp+0x168]
    add       ebx,0x00000040
    xor       esi,edx
    mov       edi,dword ptr [esp+0x16c]
    mov       ebp,dword ptr [ebx+0x24]
    mov       eax,esi
    xor       eax,ecx
    mov       ecx,dword ptr [ebx-0x1d8]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [ebx-0x1dc]
    xor       ebp,eax
    mov       dword ptr [ebx+0x20],esi
    mov       esi,dword ptr [ebx+0x28]
    mov       eax,ebp
    xor       eax,edi
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x170]
    mov       edx,eax
    mov       eax,esi
    mov       edi,dword ptr [esp+0x174]
    xor       eax,ecx
    mov       dword ptr [ebx+0x24],ebp
    mov       ebp,dword ptr [ebx+0x2c]
    mov       dword ptr [esp+0x64],eax
    mov       eax,dword ptr [ebx-0x1d4]
    mov       dword ptr [ebx+0x28],esi
    mov       esi,dword ptr [ebx+0x30]
    xor       ebp,eax
    mov       eax,ebp
    mov       dword ptr [ebx+0x2c],ebp
    mov       ebp,dword ptr [esp+0x178]
    xor       eax,edi
    mov       ecx,eax
    mov       eax,dword ptr [ebx-0x1d0]
    mov       edi,dword ptr [ebx+0x34]
    xor       esi,eax
    mov       eax,esi
    mov       dword ptr [ebx+0x30],esi
    mov       esi,dword ptr [esp+0x17c]
    xor       eax,ebp
    mov       dword ptr [esp+0x6c],eax
    mov       eax,dword ptr [ebx-0x1cc]
    mov       ebp,dword ptr [ebx+0x38]
    xor       edi,eax
    mov       eax,edi
    mov       dword ptr [ebx+0x34],edi
    mov       edi,dword ptr [esp+0x180]
    xor       eax,esi
    mov       dword ptr [esp+0x70],eax
    mov       eax,dword ptr [ebx-0x1c8]
    mov       esi,dword ptr [ebx+0x3c]
    xor       ebp,eax
    mov       eax,ebp
    mov       dword ptr [ebx+0x38],ebp
    mov       ebp,dword ptr [esp+0x184]
    xor       eax,edi
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [ebx-0x1c4]
    mov       edi,dword ptr [ebx]
    xor       esi,eax
    mov       eax,esi
    mov       dword ptr [ebx+0x3c],esi
    xor       eax,ebp
    mov       ebp,dword ptr [esp+0x148]
    mov       esi,eax
    mov       eax,dword ptr [ebx-0x200]
    xor       edi,eax
    mov       dword ptr [esp+0x50],esi
    mov       eax,edi
    add       esp,0x00000040
    xor       eax,ebp
    mov       dword ptr [esp+0x3c],ecx
    xor       eax,esi
    mov       esi,dword ptr [ebx+0x4]
    mov       dword ptr [esp+0x34],eax
    mov       eax,dword ptr [ebx-0x1fc]
    xor       esi,eax
    mov       dword ptr [ebx],edi
    mov       dword ptr [ebx+0x4],esi
    mov       eax,dword ptr [esp+0x10c]
    mov       ebp,dword ptr [ebx+0x8]
    mov       edi,dword ptr [esp+0x110]
    xor       esi,eax
    mov       eax,dword ptr [ebx-0x1f8]
    xor       ebp,eax
    mov       eax,ebp
    mov       dword ptr [ebx+0x8],ebp
    xor       eax,edi
    mov       edi,dword ptr [ebx+0xc]
    xor       eax,edx
    mov       ebp,dword ptr [ebx+0x10]
    mov       dword ptr [esp+0x38],eax
    mov       eax,dword ptr [ebx-0x1f4]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x114]
    mov       dword ptr [ebx+0xc],edi
    xor       edi,eax
    mov       eax,dword ptr [ebx-0x1f0]
    xor       ebp,eax
    mov       dword ptr [ebx+0x10],ebp
    mov       eax,ebp
    mov       ebp,dword ptr [esp+0x118]
    xor       eax,ebp
    mov       ebp,dword ptr [ebx+0x14]
    xor       eax,ecx
    mov       ecx,dword ptr [ebx-0x1ec]
    mov       dword ptr [esp+0x1c],eax
    mov       eax,dword ptr [esp+0x11c]
    xor       ebp,ecx
    mov       ecx,dword ptr [ebx+0x18]
    mov       dword ptr [ebx+0x14],ebp
    xor       ebp,eax
    mov       eax,dword ptr [ebx-0x1e8]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x120]
    mov       dword ptr [ebx+0x18],ecx
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x30]
    xor       ecx,eax
    mov       eax,dword ptr [ebx+0x1c]
    mov       dword ptr [esp+0x14],ecx
    mov       ecx,dword ptr [ebx-0x1e4]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x124]
    mov       dword ptr [ebx+0x1c],eax
    xor       eax,ecx
    mov       dword ptr [esp+0x18],eax
    lea       eax,[esp+0x128]
    lea       ecx,[esp+0x12c]
    push      eax
    push      ecx
    lea       eax,[esp+0x138]
    lea       ecx,[esp+0x13c]
    push      eax
    push      ecx
    lea       eax,[esp+0x148]
    lea       ecx,[esp+0x14c]
    push      eax
    push      ecx
    lea       eax,[esp+0x158]
    lea       ecx,[esp+0x15c]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      ecx
    mov       ecx,dword ptr [esp+0x48]
    xor       eax,ecx
    mov       ecx,esi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    mov       eax,edi
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x68]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x5c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x44]
    push      ecx
    mov       ecx,dword ptr [esp+0x64]
    mov       edx,ebp
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x4c]
    push      edx
    mov       edx,dword ptr [esp+0x58]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x108]
    lea       eax,[esp+0x10c]
    lea       ecx,[esp+0x110]
    push      edx
    push      eax
    lea       edx,[esp+0x11c]
    push      ecx
    lea       eax,[esp+0x124]
    push      edx
    lea       ecx,[esp+0x12c]
    push      eax
    lea       edx,[esp+0x134]
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    push      edx
    mov       edx,dword ptr [esp+0x44]
    lea       eax,[esp+0x140]
    push      eax
    mov       eax,dword ptr [esp+0x3c]
    xor       esi,edx
    mov       edx,dword ptr [esp+0x58]
    push      ecx
    push      esi
    mov       esi,dword ptr [esp+0x4c]
    mov       ecx,dword ptr [esp+0x3c]
    xor       edi,esi
    push      edx
    mov       edx,dword ptr [esp+0x4c]
    push      edi
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x4c]
    push      ebp
    xor       eax,edx
    push      ecx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0xc8]
    mov       edi,dword ptr [csc_tabe+0x20]
    mov       ecx,dword ptr [esp+0xd0]
    mov       edx,dword ptr [csc_tabe+0x28]
    mov       esi,dword ptr [csc_tabe+0x24]
    mov       ebp,dword ptr [csc_tabe+0x2c]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x30]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xd8]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esp+0xcc]
    xor       edx,edi
    mov       dword ptr [esp+0x64],ecx
    mov       ecx,dword ptr [esp+0xd4]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0x34]
    mov       dword ptr [esp+0x6c],edx
    mov       edx,dword ptr [esp+0xdc]
    add       esp,0x00000040
    add       ebx,0x00000040
    xor       ecx,ebp
    mov       ebp,dword ptr [csc_tabe+0x38]
    xor       edx,esi
    mov       esi,dword ptr [esp+0xa0]
    mov       dword ptr [esp+0x3c],ecx
    mov       dword ptr [esp+0x30],edx
    mov       edi,dword ptr [csc_tabe+0x3c]
    xor       esi,ebp
    mov       ebp,dword ptr [csc_tabe]
    mov       dword ptr [esp+0x20],esi
    mov       esi,dword ptr [esp+0xa4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x48]
    xor       edi,esi
    mov       dword ptr [esp+0x10],esi
    mov       esi,dword ptr [esp+0x4c]
    xor       edi,ebp
    mov       ebp,dword ptr [csc_tabe+0x8]
    mov       dword ptr [esp+0x34],edi
    mov       edi,dword ptr [csc_tabe+0x4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x50]
    xor       edi,eax
    xor       edi,ebp
    mov       ebp,dword ptr [csc_tabe+0xc]
    mov       dword ptr [esp+0x38],edi
    mov       edi,dword ptr [esp+0x54]
    xor       edi,ebp
    mov       ebp,dword ptr [esp+0x58]
    xor       ebp,ecx
    mov       ecx,dword ptr [csc_tabe+0x10]
    xor       ebp,ecx
    mov       ecx,dword ptr [csc_tabe+0x14]
    mov       dword ptr [esp+0x1c],ebp
    mov       ebp,dword ptr [esp+0x5c]
    xor       ebp,ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x18]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x1c]
    mov       dword ptr [esp+0x14],ecx
    mov       ecx,dword ptr [esp+0x64]
    xor       ecx,edx
    lea       edx,[esp+0x88]
    mov       dword ptr [esp+0x18],ecx
    lea       ecx,[esp+0x8c]
    push      edx
    push      ecx
    lea       edx,[esp+0x98]
    lea       ecx,[esp+0x9c]
    push      edx
    push      ecx
    lea       edx,[esp+0xa8]
    lea       ecx,[esp+0xac]
    push      edx
    push      ecx
    lea       edx,[esp+0xb8]
    lea       ecx,[esp+0xbc]
    push      edx
    mov       edx,dword ptr [esp+0x50]
    push      ecx
    mov       ecx,dword ptr [esp+0x48]
    xor       edx,ecx
    mov       ecx,esi
    xor       ecx,eax
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    mov       eax,edi
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x68]
    xor       eax,edx
    mov       edx,ebp
    push      eax
    mov       eax,dword ptr [esp+0x5c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x44]
    push      ecx
    mov       ecx,dword ptr [esp+0x64]
    xor       edx,ecx
    push      edx
    mov       edx,dword ptr [esp+0x58]
    xor       eax,edx
    mov       ecx,dword ptr [esp+0x50]
    push      eax
    xor       ecx,dword ptr [esp+0x4c]
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x48]
    lea       eax,[esp+0x4c]
    lea       ecx,[esp+0x50]
    push      edx
    push      eax
    lea       edx,[esp+0x5c]
    push      ecx
    lea       eax,[esp+0x64]
    push      edx
    lea       ecx,[esp+0x6c]
    push      eax
    lea       edx,[esp+0x74]
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    push      edx
    mov       edx,dword ptr [esp+0x44]
    lea       eax,[esp+0x80]
    push      eax
    mov       eax,dword ptr [esp+0x3c]
    xor       esi,edx
    mov       edx,dword ptr [esp+0x58]
    push      ecx
    push      esi
    mov       esi,dword ptr [esp+0x4c]
    mov       ecx,dword ptr [esp+0x3c]
    xor       edi,esi
    push      edx
    mov       edx,dword ptr [esp+0x4c]
    push      edi
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x4c]
    push      ebp
    xor       eax,edx
    push      ecx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x148]
    mov       edi,dword ptr [csc_tabe+0x60]
    mov       ecx,dword ptr [csc_tabe+0x68]
    xor       eax,edi
    mov       ebp,dword ptr [esp+0x154]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esp+0x150]
    mov       edi,dword ptr [csc_tabe+0x70]
    mov       edx,dword ptr [esp+0x14c]
    mov       esi,dword ptr [csc_tabe+0x64]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x78]
    mov       dword ptr [esp+0x64],eax
    mov       eax,dword ptr [csc_tabe+0x6c]
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x158]
    xor       eax,edi
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x74]
    mov       dword ptr [esp+0x6c],eax
    mov       eax,dword ptr [esp+0x15c]
    add       esp,0x00000040
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0x40]
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esp+0x120]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x124]
    mov       dword ptr [esp+0x20],eax
    mov       eax,dword ptr [csc_tabe+0x7c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0xc8]
    mov       dword ptr [esp+0x40],edx
    mov       dword ptr [esp+0x3c],ebp
    mov       dword ptr [esp+0x10],ecx
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x44]
    mov       edi,dword ptr [csc_tabe+0x48]
    xor       eax,esi
    mov       esi,dword ptr [esp+0xcc]
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0xd0]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x4c]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0xd4]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xd8]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x50]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xdc]
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [csc_tabe+0x54]
    xor       ebp,edx
    mov       edx,dword ptr [esp+0xe0]
    xor       edx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x34],eax
    mov       dword ptr [esp+0x38],ecx
    xor       edx,dword ptr [csc_tabe+0x58]
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0xe4]
    xor       edx,dword ptr [csc_tabe+0x5c]
    mov       dword ptr [esp+0x18],edx
    lea       edx,[esp+0x108]
    push      edx
    lea       edx,[esp+0x110]
    push      edx
    lea       edx,[esp+0x118]
    push      edx
    lea       edx,[esp+0x120]
    push      edx
    lea       edx,[esp+0x128]
    push      edx
    lea       edx,[esp+0x130]
    push      edx
    lea       edx,[esp+0x138]
    push      edx
    lea       edx,[esp+0x140]
    push      edx
    mov       edx,dword ptr [esp+0x48]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x60]
    push      eax
    mov       eax,esi
    xor       eax,edx
    mov       edx,dword ptr [esp+0x40]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x64]
    push      ecx
    mov       ecx,edi
    xor       ecx,eax
    mov       eax,ebp
    push      ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       ecx,eax
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    xor       edx,ecx
    push      edx
    call      csc_transP
    add       esp,0x00000040
    lea       eax,[esp+0xc8]
    lea       ecx,[esp+0xcc]
    lea       edx,[esp+0xd0]
    push      eax
    push      ecx
    lea       eax,[esp+0xdc]
    push      edx
    lea       ecx,[esp+0xe4]
    push      eax
    lea       edx,[esp+0xec]
    push      ecx
    lea       eax,[esp+0xf4]
    push      edx
    mov       edx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x44]
    lea       ecx,[esp+0x100]
    push      ecx
    xor       esi,eax
    mov       eax,dword ptr [esp+0x58]
    mov       ecx,dword ptr [esp+0x3c]
    push      edx
    push      esi
    mov       edx,dword ptr [esp+0x3c]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x58]
    push      edi
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    push      ebp
    push      edx
    xor       eax,dword ptr [esp+0x5c]
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0xe8]
    mov       edi,dword ptr [csc_tabe+0xa0]
    mov       ecx,dword ptr [csc_tabe+0xa8]
    xor       eax,edi
    mov       ebp,dword ptr [esp+0xf4]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esp+0xf0]
    mov       edi,dword ptr [csc_tabe+0xb0]
    mov       edx,dword ptr [esp+0xec]
    mov       esi,dword ptr [csc_tabe+0xa4]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0xb8]
    mov       dword ptr [esp+0x64],eax
    mov       eax,dword ptr [csc_tabe+0xac]
    xor       ebp,eax
    mov       eax,dword ptr [esp+0xf8]
    xor       eax,edi
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0xb4]
    mov       dword ptr [esp+0x6c],eax
    mov       eax,dword ptr [esp+0xfc]
    mov       edi,dword ptr [csc_tabe+0x88]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0x80]
    mov       dword ptr [esp+0x70],eax
    mov       eax,dword ptr [esp+0x100]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x104]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [csc_tabe+0xbc]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0xa8]
    xor       eax,ecx
    mov       dword ptr [esp+0x50],ecx
    mov       ecx,dword ptr [csc_tabe+0x84]
    xor       eax,esi
    mov       esi,dword ptr [esp+0xac]
    add       esp,0x00000040
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x70]
    xor       ecx,edx
    mov       dword ptr [esp+0x40],edx
    mov       edx,dword ptr [csc_tabe+0x8c]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x74]
    mov       dword ptr [esp+0x3c],ebp
    mov       dword ptr [esp+0x34],eax
    mov       dword ptr [esp+0x38],ecx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x78]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x90]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x7c]
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [csc_tabe+0x94]
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x80]
    xor       edx,dword ptr [esp+0x30]
    xor       edx,dword ptr [csc_tabe+0x98]
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0x84]
    xor       edx,dword ptr [csc_tabe+0x9c]
    mov       dword ptr [esp+0x18],edx
    lea       edx,[esp+0xa8]
    push      edx
    lea       edx,[esp+0xb0]
    push      edx
    lea       edx,[esp+0xb8]
    push      edx
    lea       edx,[esp+0xc0]
    push      edx
    lea       edx,[esp+0xc8]
    push      edx
    lea       edx,[esp+0xd0]
    push      edx
    lea       edx,[esp+0xd8]
    push      edx
    lea       edx,[esp+0xe0]
    push      edx
    mov       edx,dword ptr [esp+0x48]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x60]
    push      eax
    mov       eax,esi
    xor       eax,edx
    mov       edx,dword ptr [esp+0x40]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x64]
    push      ecx
    mov       ecx,edi
    xor       ecx,eax
    mov       eax,ebp
    push      ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       ecx,eax
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    xor       edx,ecx
    push      edx
    call      csc_transP
    add       esp,0x00000040
    lea       eax,[esp+0x68]
    lea       ecx,[esp+0x6c]
    lea       edx,[esp+0x70]
    push      eax
    push      ecx
    lea       eax,[esp+0x7c]
    push      edx
    lea       ecx,[esp+0x84]
    push      eax
    lea       edx,[esp+0x8c]
    push      ecx
    lea       eax,[esp+0x94]
    push      edx
    mov       edx,dword ptr [esp+0x4c]
    lea       ecx,[esp+0x9c]
    push      eax
    push      ecx
    mov       eax,dword ptr [esp+0x48]
    mov       ecx,dword ptr [esp+0x3c]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x58]
    push      edx
    push      esi
    mov       edx,dword ptr [esp+0x3c]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x58]
    push      edi
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    push      ebp
    push      edx
    xor       eax,dword ptr [esp+0x5c]
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x168]
    mov       edi,dword ptr [csc_tabe+0xe0]
    mov       ecx,dword ptr [csc_tabe+0xe8]
    xor       eax,edi
    mov       ebp,dword ptr [esp+0x174]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esp+0x170]
    mov       edi,dword ptr [csc_tabe+0xf0]
    mov       edx,dword ptr [esp+0x16c]
    mov       esi,dword ptr [csc_tabe+0xe4]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0xf8]
    mov       dword ptr [esp+0x64],eax
    mov       eax,dword ptr [csc_tabe+0xec]
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x178]
    xor       eax,edi
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0xf4]
    mov       dword ptr [esp+0x6c],eax
    mov       eax,dword ptr [esp+0x17c]
    mov       edi,dword ptr [csc_tabe+0xc8]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0xc0]
    mov       dword ptr [esp+0x70],eax
    mov       eax,dword ptr [esp+0x180]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x184]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [csc_tabe+0xfc]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x128]
    xor       eax,ecx
    mov       dword ptr [esp+0x50],ecx
    mov       ecx,dword ptr [csc_tabe+0xc4]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x12c]
    mov       dword ptr [esp+0x80],edx
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x130]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0xcc]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x134]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x138]
    mov       dword ptr [esp+0x7c],ebp
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0xd0]
    add       esp,0x00000040
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xfc]
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [csc_tabe+0xd4]
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x100]
    xor       edx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x34],eax
    mov       dword ptr [esp+0x38],ecx
    xor       edx,dword ptr [csc_tabe+0xd8]
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0x104]
    xor       edx,dword ptr [csc_tabe+0xdc]
    mov       dword ptr [esp+0x18],edx
    lea       edx,[esp+0x128]
    push      edx
    lea       edx,[esp+0x130]
    push      edx
    lea       edx,[esp+0x138]
    push      edx
    lea       edx,[esp+0x140]
    push      edx
    lea       edx,[esp+0x148]
    push      edx
    lea       edx,[esp+0x150]
    push      edx
    lea       edx,[esp+0x158]
    push      edx
    lea       edx,[esp+0x160]
    push      edx
    mov       edx,dword ptr [esp+0x48]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x60]
    push      eax
    mov       eax,esi
    xor       eax,edx
    mov       edx,dword ptr [esp+0x40]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x64]
    push      ecx
    mov       ecx,edi
    xor       ecx,eax
    mov       eax,ebp
    push      ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       ecx,eax
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    xor       edx,ecx
    push      edx
    call      csc_transP
    add       esp,0x00000040
    lea       eax,[esp+0xe8]
    lea       ecx,[esp+0xec]
    lea       edx,[esp+0xf0]
    push      eax
    push      ecx
    lea       eax,[esp+0xfc]
    push      edx
    lea       ecx,[esp+0x104]
    push      eax
    lea       edx,[esp+0x10c]
    push      ecx
    lea       eax,[esp+0x114]
    push      edx
    mov       edx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x44]
    lea       ecx,[esp+0x120]
    push      ecx
    mov       ecx,dword ptr [esp+0x3c]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x58]
    push      edx
    push      esi
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x58]
    push      edi
    push      ecx
    mov       edx,dword ptr [esp+0x48]
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x4c]
    push      ebp
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x108]
    mov       edi,dword ptr [csc_tabe+0x120]
    mov       ecx,dword ptr [csc_tabe+0x128]
    xor       eax,edi
    mov       ebp,dword ptr [esp+0x114]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esp+0x110]
    mov       edi,dword ptr [csc_tabe+0x130]
    mov       edx,dword ptr [esp+0x10c]
    mov       esi,dword ptr [csc_tabe+0x124]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x138]
    mov       dword ptr [esp+0x64],eax
    mov       eax,dword ptr [csc_tabe+0x12c]
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x118]
    xor       eax,edi
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x134]
    mov       dword ptr [esp+0x6c],eax
    mov       eax,dword ptr [esp+0x11c]
    mov       edi,dword ptr [csc_tabe+0x108]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0x100]
    mov       dword ptr [esp+0x70],eax
    mov       eax,dword ptr [esp+0x120]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x124]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [csc_tabe+0x13c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x88]
    xor       eax,ecx
    mov       dword ptr [esp+0x50],ecx
    mov       ecx,dword ptr [csc_tabe+0x104]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x80],edx
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x90]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x10c]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x94]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x98]
    mov       dword ptr [esp+0x7c],ebp
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x110]
    add       esp,0x00000040
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [csc_tabe+0x114]
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x60]
    xor       edx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x34],eax
    mov       dword ptr [esp+0x38],ecx
    xor       edx,dword ptr [csc_tabe+0x118]
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0x64]
    xor       edx,dword ptr [csc_tabe+0x11c]
    mov       dword ptr [esp+0x18],edx
    lea       edx,[esp+0xc8]
    push      edx
    lea       edx,[esp+0xd0]
    push      edx
    lea       edx,[esp+0xd8]
    push      edx
    lea       edx,[esp+0xe0]
    push      edx
    lea       edx,[esp+0xe8]
    push      edx
    lea       edx,[esp+0xf0]
    push      edx
    lea       edx,[esp+0xf8]
    push      edx
    lea       edx,[esp+0x100]
    push      edx
    mov       edx,dword ptr [esp+0x48]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x60]
    push      eax
    mov       eax,esi
    xor       eax,edx
    mov       edx,dword ptr [esp+0x40]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x64]
    push      ecx
    mov       ecx,edi
    xor       ecx,eax
    mov       eax,ebp
    push      ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       ecx,eax
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    xor       edx,ecx
    push      edx
    call      csc_transP
    add       esp,0x00000040
    lea       eax,[esp+0x48]
    lea       ecx,[esp+0x4c]
    lea       edx,[esp+0x50]
    push      eax
    push      ecx
    lea       eax,[esp+0x5c]
    push      edx
    lea       ecx,[esp+0x64]
    push      eax
    lea       edx,[esp+0x6c]
    push      ecx
    lea       eax,[esp+0x74]
    push      edx
    mov       edx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x44]
    lea       ecx,[esp+0x80]
    push      ecx
    xor       esi,eax
    mov       eax,dword ptr [esp+0x58]
    mov       ecx,dword ptr [esp+0x3c]
    push      edx
    push      esi
    mov       edx,dword ptr [esp+0x3c]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x58]
    push      edi
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    push      ebp
    push      edx
    xor       eax,dword ptr [esp+0x5c]
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x128]
    mov       edi,dword ptr [csc_tabe+0x160]
    add       esp,0x00000040
    mov       ecx,dword ptr [csc_tabe+0x168]
    xor       eax,edi
    mov       ebp,dword ptr [esp+0xf4]
    mov       dword ptr [esp+0x28],eax
    mov       eax,dword ptr [esp+0xf0]
    mov       edi,dword ptr [csc_tabe+0x170]
    mov       edx,dword ptr [esp+0xec]
    mov       esi,dword ptr [csc_tabe+0x164]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x178]
    mov       dword ptr [esp+0x24],eax
    mov       eax,dword ptr [csc_tabe+0x16c]
    xor       ebp,eax
    mov       eax,dword ptr [esp+0xf8]
    xor       eax,edi
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x174]
    mov       dword ptr [esp+0x2c],eax
    mov       eax,dword ptr [esp+0xfc]
    mov       edi,dword ptr [csc_tabe+0x148]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0x140]
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esp+0x100]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x104]
    mov       dword ptr [esp+0x20],eax
    mov       eax,dword ptr [csc_tabe+0x17c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x68]
    xor       eax,ecx
    mov       dword ptr [esp+0x10],ecx
    mov       ecx,dword ptr [csc_tabe+0x144]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x40],edx
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x70]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x14c]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x74]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x3c],ebp
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x150]
    mov       dword ptr [esp+0x34],eax
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x7c]
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [csc_tabe+0x154]
    xor       ebp,edx
    mov       edx,dword ptr [esp+0x80]
    xor       edx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x38],ecx
    xor       edx,dword ptr [csc_tabe+0x158]
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0x84]
    xor       edx,dword ptr [csc_tabe+0x15c]
    mov       dword ptr [esp+0x18],edx
    lea       edx,[esp+0xe8]
    push      edx
    lea       edx,[esp+0xf0]
    push      edx
    lea       edx,[esp+0xf8]
    push      edx
    lea       edx,[esp+0x100]
    push      edx
    lea       edx,[esp+0x108]
    push      edx
    lea       edx,[esp+0x110]
    push      edx
    lea       edx,[esp+0x118]
    push      edx
    lea       edx,[esp+0x120]
    push      edx
    mov       edx,dword ptr [esp+0x48]
    xor       eax,edx
    push      eax
    mov       eax,esi
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x40]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x64]
    push      ecx
    mov       ecx,edi
    xor       ecx,eax
    mov       eax,ebp
    push      ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       ecx,eax
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    xor       edx,ecx
    push      edx
    call      csc_transP
    add       esp,0x00000040
    lea       eax,[esp+0x68]
    lea       ecx,[esp+0x6c]
    lea       edx,[esp+0x70]
    push      eax
    push      ecx
    lea       eax,[esp+0x7c]
    push      edx
    lea       ecx,[esp+0x84]
    push      eax
    lea       edx,[esp+0x8c]
    push      ecx
    lea       eax,[esp+0x94]
    push      edx
    mov       edx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x44]
    lea       ecx,[esp+0xa0]
    push      ecx
    xor       esi,eax
    mov       eax,dword ptr [esp+0x58]
    mov       ecx,dword ptr [esp+0x3c]
    push      edx
    push      esi
    mov       edx,dword ptr [esp+0x3c]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x58]
    push      edi
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    push      ebp
    push      edx
    xor       eax,dword ptr [esp+0x5c]
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x148]
    mov       edi,dword ptr [csc_tabe+0x1a0]
    mov       edx,dword ptr [esp+0x14c]
    mov       esi,dword ptr [csc_tabe+0x1a4]
    mov       ecx,dword ptr [csc_tabe+0x1a8]
    xor       eax,edi
    mov       ebp,dword ptr [esp+0x154]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esp+0x150]
    add       esp,0x00000040
    xor       edx,esi
    xor       eax,ecx
    mov       dword ptr [esp+0x24],eax
    mov       eax,dword ptr [csc_tabe+0x1ac]
    mov       dword ptr [esp+0x40],edx
    mov       edi,dword ptr [csc_tabe+0x1b0]
    mov       esi,dword ptr [csc_tabe+0x1b4]
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x118]
    xor       eax,edi
    mov       ecx,dword ptr [csc_tabe+0x1b8]
    mov       dword ptr [esp+0x2c],eax
    mov       eax,dword ptr [esp+0x11c]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0x180]
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esp+0x120]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x124]
    mov       dword ptr [esp+0x20],eax
    mov       eax,dword ptr [csc_tabe+0x1bc]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x88]
    xor       eax,ecx
    mov       edi,dword ptr [csc_tabe+0x188]
    mov       dword ptr [esp+0x10],ecx
    mov       ecx,dword ptr [csc_tabe+0x184]
    xor       eax,esi
    mov       esi,dword ptr [esp+0x8c]
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0x90]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x18c]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x94]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x98]
    mov       dword ptr [esp+0x3c],ebp
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x190]
    mov       dword ptr [esp+0x34],eax
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0x9c]
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [csc_tabe+0x194]
    xor       ebp,edx
    mov       edx,dword ptr [esp+0xa0]
    xor       edx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x38],ecx
    xor       edx,dword ptr [csc_tabe+0x198]
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0xa4]
    xor       edx,dword ptr [csc_tabe+0x19c]
    mov       dword ptr [esp+0x18],edx
    lea       edx,[esp+0x108]
    push      edx
    lea       edx,[esp+0x110]
    push      edx
    lea       edx,[esp+0x118]
    push      edx
    lea       edx,[esp+0x120]
    push      edx
    lea       edx,[esp+0x128]
    push      edx
    lea       edx,[esp+0x130]
    push      edx
    lea       edx,[esp+0x138]
    push      edx
    lea       edx,[esp+0x140]
    push      edx
    mov       edx,dword ptr [esp+0x48]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x60]
    push      eax
    mov       eax,esi
    xor       eax,edx
    mov       edx,dword ptr [esp+0x40]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x64]
    push      ecx
    mov       ecx,edi
    xor       ecx,eax
    push      ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    mov       eax,ebp
    xor       eax,edx
    mov       edx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       ecx,eax
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    xor       edx,ecx
    push      edx
    call      csc_transP
    add       esp,0x00000040
    lea       eax,[esp+0x88]
    lea       ecx,[esp+0x8c]
    lea       edx,[esp+0x90]
    push      eax
    push      ecx
    lea       eax,[esp+0x9c]
    push      edx
    lea       ecx,[esp+0xa4]
    push      eax
    lea       edx,[esp+0xac]
    push      ecx
    lea       eax,[esp+0xb4]
    push      edx
    mov       edx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x44]
    lea       ecx,[esp+0xc0]
    push      ecx
    xor       esi,eax
    mov       eax,dword ptr [esp+0x58]
    mov       ecx,dword ptr [esp+0x3c]
    push      edx
    push      esi
    mov       edx,dword ptr [esp+0x3c]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x58]
    push      edi
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    push      ebp
    push      edx
    xor       eax,dword ptr [esp+0x5c]
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x168]
    mov       edi,dword ptr [csc_tabe+0x1e0]
    mov       ecx,dword ptr [csc_tabe+0x1e8]
    xor       eax,edi
    mov       ebp,dword ptr [esp+0x174]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [esp+0x170]
    mov       edi,dword ptr [csc_tabe+0x1f0]
    mov       edx,dword ptr [esp+0x16c]
    mov       esi,dword ptr [csc_tabe+0x1e4]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x1f8]
    mov       dword ptr [esp+0x64],eax
    mov       eax,dword ptr [csc_tabe+0x1ec]
    xor       ebp,eax
    mov       eax,dword ptr [esp+0x178]
    xor       eax,edi
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x1f4]
    mov       dword ptr [esp+0x6c],eax
    mov       eax,dword ptr [esp+0x17c]
    add       esp,0x00000040
    xor       eax,esi
    mov       dword ptr [esp+0x40],edx
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esp+0x140]
    mov       dword ptr [esp+0x3c],ebp
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x144]
    mov       esi,dword ptr [csc_tabe+0x1c0]
    mov       dword ptr [esp+0x20],eax
    mov       eax,dword ptr [csc_tabe+0x1fc]
    mov       edi,dword ptr [csc_tabe+0x1c8]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0xa8]
    xor       eax,ecx
    mov       dword ptr [esp+0x10],ecx
    mov       ecx,dword ptr [csc_tabe+0x1c4]
    xor       eax,esi
    mov       esi,dword ptr [esp+0xac]
    mov       dword ptr [esp+0x34],eax
    xor       esi,ecx
    mov       ecx,dword ptr [esp+0xb0]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x1cc]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0xb4]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xb8]
    xor       edx,ebp
    mov       ebp,dword ptr [csc_tabe+0x1d0]
    xor       edx,ebp
    mov       ebp,dword ptr [esp+0xbc]
    mov       dword ptr [esp+0x1c],edx
    mov       edx,dword ptr [csc_tabe+0x1d4]
    xor       ebp,edx
    mov       edx,dword ptr [esp+0xc0]
    xor       edx,dword ptr [esp+0x30]
    mov       dword ptr [esp+0x38],ecx
    xor       edx,dword ptr [csc_tabe+0x1d8]
    mov       dword ptr [esp+0x14],edx
    mov       edx,dword ptr [esp+0xc4]
    xor       edx,dword ptr [csc_tabe+0x1dc]
    mov       dword ptr [esp+0x18],edx
    lea       edx,[esp+0x128]
    push      edx
    lea       edx,[esp+0x130]
    push      edx
    lea       edx,[esp+0x138]
    push      edx
    lea       edx,[esp+0x140]
    push      edx
    lea       edx,[esp+0x148]
    push      edx
    lea       edx,[esp+0x150]
    push      edx
    lea       edx,[esp+0x158]
    push      edx
    lea       edx,[esp+0x160]
    push      edx
    mov       edx,dword ptr [esp+0x48]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x60]
    push      eax
    mov       eax,esi
    xor       eax,edx
    mov       edx,dword ptr [esp+0x40]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x64]
    push      ecx
    mov       ecx,edi
    xor       ecx,eax
    mov       eax,ebp
    push      ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       ecx,eax
    push      ecx
    mov       ecx,dword ptr [esp+0x4c]
    xor       edx,ecx
    push      edx
    call      csc_transP
    add       esp,0x00000040
    lea       eax,[esp+0xa8]
    lea       ecx,[esp+0xac]
    lea       edx,[esp+0xb0]
    push      eax
    push      ecx
    lea       eax,[esp+0xbc]
    push      edx
    lea       ecx,[esp+0xc4]
    push      eax
    lea       edx,[esp+0xcc]
    push      ecx
    lea       eax,[esp+0xd4]
    push      edx
    mov       edx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x44]
    lea       ecx,[esp+0xe0]
    push      ecx
    mov       ecx,dword ptr [esp+0x3c]
    xor       esi,eax
    mov       eax,dword ptr [esp+0x58]
    push      edx
    push      esi
    push      eax
    mov       eax,dword ptr [esp+0x50]
    mov       esi,dword ptr [esp+0x4c]
    mov       edx,dword ptr [esp+0x40]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x44]
    push      edi
    push      ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,esi
    xor       ebp,ecx
    push      ebp
    push      edx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x188]
    add       esp,0x00000040
    dec       eax
    mov       dword ptr [esp+0x148],eax
    ljne       X$1
    lea       eax,[esp+0xc8]
    mov       dword ptr [esp+0x10],0xffffffff
    mov       dword ptr [esp+0x40],eax
    mov       eax,dword ptr [esp+0x15c]
    mov       dword ptr [esp+0x148],0x00000000
    lea       ebp,[ebx+0xc0]
    lea       ecx,[eax+0xc0]
    mov       dword ptr [esp+0x3c],ecx
    lea       ecx,[esp+0x68]
    sub       ecx,eax
    mov       dword ptr [esp+0x38],ecx
    lea       ecx,[esp+0x48]
    sub       ecx,eax
    mov       dword ptr [esp+0x34],ecx
X$4:
    lea       eax,[ebp-0xa0]
    lea       ecx,[ebp-0x80]
    push      ebx
    lea       edx,[ebp-0x60]
    push      eax
    lea       esi,[ebp-0x40]
    push      ecx
    push      edx
    lea       eax,[ebp-0x20]
    push      esi
    mov       esi,dword ptr [esp+0x160]
    lea       edi,[ebp+0x20]
    push      eax
    push      ebp
    mov       edx,dword ptr [esi]
    push      edi
    mov       edi,dword ptr [esp+0x64]
    mov       eax,dword ptr [esi+0x4]
    mov       ecx,dword ptr [edi]
    xor       edx,ecx
    mov       ecx,dword ptr [edi+0x8]
    push      edx
    mov       edx,dword ptr [edi+0x4]
    xor       eax,edx
    mov       edx,dword ptr [edi+0xc]
    push      eax
    mov       eax,dword ptr [esi+0x8]
    xor       ecx,eax
    mov       eax,dword ptr [esi+0x10]
    push      ecx
    mov       ecx,dword ptr [esi+0xc]
    xor       edx,ecx
    mov       ecx,dword ptr [esi+0x14]
    push      edx
    mov       edx,dword ptr [edi+0x10]
    xor       eax,edx
    mov       edx,dword ptr [edi+0x18]
    push      eax
    mov       eax,dword ptr [edi+0x14]
    xor       ecx,eax
    mov       eax,dword ptr [edi+0x1c]
    push      ecx
    mov       ecx,dword ptr [esi+0x18]
    xor       edx,ecx
    push      edx
    mov       edx,dword ptr [esi+0x1c]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       edx,dword ptr [esp+0x78]
    mov       ecx,dword ptr [esp+0x7c]
    add       esp,0x00000040
    mov       eax,dword ptr [edx+ecx]
    mov       edx,dword ptr [ecx+0x20]
    xor       eax,edx
    mov       edx,dword ptr [ebx-0x120]
    xor       eax,edx
    mov       edx,dword ptr [ebp+0x20]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x10]
    not       eax
    and       edx,eax
    mov       dword ptr [esp+0x10],edx
    lje        X$5
    mov       edx,dword ptr [esp+0x34]
    mov       eax,dword ptr [edx+ecx]
    mov       edx,dword ptr [ebx-0x140]
    xor       eax,edx
    mov       edx,dword ptr [ecx]
    xor       eax,edx
    mov       edx,dword ptr [ebp]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x10]
    not       eax
    and       edx,eax
    mov       dword ptr [esp+0x10],edx
    lje        X$5
    mov       edx,dword ptr [esp+0x40]
    mov       eax,dword ptr [edx+0x20]
    mov       edx,dword ptr [ecx-0x20]
    xor       eax,edx
    mov       edx,dword ptr [ebx-0x160]
    xor       eax,edx
    mov       edx,dword ptr [ebp-0x20]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x10]
    not       eax
    and       edx,eax
    mov       dword ptr [esp+0x10],edx
    lje        X$5
    mov       edx,dword ptr [ecx-0x40]
    mov       eax,dword ptr [ebx-0x180]
    xor       edx,eax
    mov       eax,dword ptr [ebp-0x40]
    xor       edx,eax
    mov       eax,dword ptr [esp+0x40]
    xor       edx,dword ptr [eax]
    mov       eax,dword ptr [esp+0x10]
    not       edx
    and       eax,edx
    mov       dword ptr [esp+0x10],eax
    lje        X$5
    mov       edx,dword ptr [esp+0x40]
    mov       eax,dword ptr [edx-0x20]
    mov       edx,dword ptr [ecx-0x60]
    xor       eax,edx
    mov       edx,dword ptr [ebx-0x1a0]
    xor       eax,edx
    mov       edx,dword ptr [ebp-0x60]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x10]
    not       eax
    and       edx,eax
    mov       dword ptr [esp+0x10],edx
    lje        X$5
    mov       edx,dword ptr [esp+0x40]
    mov       eax,dword ptr [edx-0x40]
    mov       edx,dword ptr [ecx-0x80]
    xor       eax,edx
    mov       edx,dword ptr [ebx-0x1c0]
    xor       eax,edx
    mov       edx,dword ptr [ebp-0x80]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x10]
    not       eax
    and       edx,eax
    mov       dword ptr [esp+0x10],edx
    lje        X$5
    mov       edx,dword ptr [esp+0x40]
    mov       eax,dword ptr [edx-0x60]
    mov       edx,dword ptr [ecx-0xa0]
    xor       eax,edx
    mov       edx,dword ptr [ebx-0x1e0]
    xor       eax,edx
    mov       edx,dword ptr [ebp-0xa0]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x10]
    not       eax
    and       edx,eax
    mov       dword ptr [esp+0x10],edx
    je        X$5
    mov       edx,dword ptr [esp+0x40]
    mov       eax,dword ptr [edx-0x80]
    mov       edx,dword ptr [ecx-0xc0]
    xor       eax,edx
    mov       edx,dword ptr [ebx-0x200]
    xor       eax,edx
    mov       edx,dword ptr [ebx]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x10]
    not       eax
    and       edx,eax
    mov       dword ptr [esp+0x10],edx
    je        X$5
    mov       eax,dword ptr [esp+0x148]
    add       ecx,0x00000004
    mov       dword ptr [esp+0x3c],ecx
    mov       ecx,dword ptr [esp+0x40]
    inc       eax
    add       ecx,0x00000004
    add       esi,0x00000020
    add       edi,0x00000020
    add       ebx,0x00000004
    add       ebp,0x00000004
    cmp       eax,0x00000008
    mov       dword ptr [esp+0x148],eax
    mov       dword ptr [esp+0x40],ecx
    mov       dword ptr [esp+0x14c],esi
    mov       dword ptr [esp+0x44],edi
    ljl        X$4
X$5:
    mov       eax,dword ptr [esp+0x10]
    pop       edi
    pop       esi
    pop       ebp
    pop       ebx
    add       esp,0x00000140
    ret       

__CODESECT__
   align 32
csc_unit_func_1k:
_csc_unit_func_1k:
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
    call      cscipher_bitslicer_1k
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
    call      cscipher_bitslicer_1k
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

