; Copyright distributed.net 1997 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; $Log: csc-6b.asm,v $
; Revision 1.1.2.3  1999/11/07 01:31:19  remi
; Increased code alignment.
;
; Revision 1.1.2.2  1999/11/06 05:41:56  gregh
; Remove 'near' specifier on some jumps for compatibility with nasm 0.97.
; It appears that nasm 0.98 ignores superfluous near specifiers.
;
; Revision 1.1.2.1  1999/11/06 00:26:16  cyp
; they're here! (see also bench.res for 'ideal' combination)
;
;

global         csc_unit_func_6b,_csc_unit_func_6b

extern         csc_tabc,csc_tabp,csc_tabe,csc_bit_order,csc_transP
extern         convert_key_from_inc_to_csc,convert_key_from_csc_to_inc

%include "csc-mac.inc"

__DATASECT__
    db  "@(#)$Id: csc-6b.asm,v 1.1.2.3 1999/11/07 01:31:19 remi Exp $",0

__CODESECT__
    align 32
cscipher_bitslicer_6b:
    sub       esp,0x0000015c
    mov       eax,dword ptr [esp+0x160]
    push      ebx
    push      ebp
    mov       ebp,dword ptr [esp+0x178]
    push      esi
    mov       edx,ebp
    push      edi
    add       ebp,0x00000b00
    lea       esi,[eax+0x100]
    mov       ecx,0x00000040
    mov       edi,edx
    mov       dword ptr [esp+0x180],ebp
    repe movsd 
    add       ebp,0x000002c0
    lea       edi,[edx+0x100]
    mov       dword ptr [esp+0x164],ebp
    mov       ecx,0x00000040
    mov       esi,eax
    mov       dword ptr [esp+0x164],edi
    repe movsd 
    xor       eax,eax
    or        ecx,0xffffffff
    mov       dword ptr [esp+0x15c],eax
    mov       dword ptr [edx+0x260],eax
    mov       dword ptr [edx+0x280],eax
    mov       dword ptr [edx+0x2a0],eax
    mov       dword ptr [edx+0x2c0],eax
    mov       dword ptr [edx+0x2e0],eax
    lea       eax,[edx+0x200]
    mov       dword ptr [edx+0x220],ecx
    mov       dword ptr [edx+0x240],ecx
    mov       dword ptr [esp+0x160],eax
    mov       dword ptr [eax],ecx
    lea       ecx,[edx+0x204]
    mov       dword ptr [esp+0x158],edx
    lea       ebx,[ebp+0x80]
    lea       eax,[ecx+0x40]
    mov       dword ptr [esp+0x4c],ecx
    mov       esi,offset csc_tabc+0x28
    mov       dword ptr [esp+0x10],eax
    lea       edi,[edx+0x128]
    mov       dword ptr [esp+0x50],0x00000007
    jmp       X$2
X$1:
    mov       eax,dword ptr [esp+0x10]
    mov       ecx,dword ptr [esp+0x4c]
X$2:
    push      ecx
    lea       ecx,[eax-0x20]
    push      ecx
    lea       edx,[eax+0x20]
    push      eax
    lea       ecx,[eax+0x40]
    push      edx
    push      ecx
    lea       edx,[eax+0x60]
    lea       ecx,[eax+0x80]
    push      edx
    mov       edx,dword ptr [edi-0x8]
    push      ecx
    mov       ecx,dword ptr [esi-0x8]
    add       eax,0x000000a0
    xor       edx,ecx
    mov       ecx,dword ptr [edi]
    push      eax
    mov       eax,dword ptr [edi-0x4]
    push      edx
    mov       edx,dword ptr [esi-0x4]
    xor       eax,edx
    mov       edx,dword ptr [edi+0x4]
    push      eax
    mov       eax,dword ptr [esi]
    xor       ecx,eax
    mov       eax,dword ptr [edi+0x8]
    push      ecx
    mov       ecx,dword ptr [esi+0x4]
    xor       edx,ecx
    mov       ecx,dword ptr [edi+0xc]
    push      edx
    mov       edx,dword ptr [esi+0x8]
    xor       eax,edx
    mov       edx,dword ptr [edi+0x10]
    push      eax
    mov       eax,dword ptr [esi+0xc]
    xor       ecx,eax
    mov       eax,dword ptr [edi+0x14]
    push      ecx
    mov       ecx,dword ptr [esi+0x10]
    xor       edx,ecx
    push      edx
    mov       edx,dword ptr [esi+0x14]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       edx,dword ptr [esp+0x8c]
    mov       ecx,dword ptr [esp+0x50]
    mov       eax,0x00000004
    add       esp,0x00000040
    add       edx,eax
    add       ecx,eax
    mov       eax,dword ptr [esp+0x50]
    add       esi,0x00000020
    add       edi,0x00000020
    dec       eax
    mov       dword ptr [esp+0x4c],edx
    mov       dword ptr [esp+0x10],ecx
    mov       dword ptr [esp+0x50],eax
    ljne       X$1
    mov       edi,dword ptr [esp+0x180]
    mov       edx,0x00000002
    mov       esi,0x000000e8
    mov       dword ptr [esp+0x50],edx
    lea       eax,[edi+0xe8]
    mov       dword ptr [esp+0x20],0x0000003a
    mov       dword ptr [esp+0x168],eax
    mov       dword ptr [esp+0x1c],eax
    mov       eax,dword ptr [esp+0x158]
    mov       dword ptr [esp+0x3c],offset csc_tabp+0x5
    mov       dword ptr [esp+0x14],esi
    lea       ecx,[eax+0x208]
    add       eax,0x00000158
    mov       dword ptr [esp+0x18],eax
    mov       eax,dword ptr [esp+0x174]
    add       eax,0x00000005
    mov       dword ptr [esp+0x44],ecx
    mov       dword ptr [esp+0x40],eax
X$3:
    mov       ecx,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x10],0x00000001
    mov       al,byte ptr [ecx]
    mov       ecx,dword ptr [esp+0x40]
    xor       al,byte ptr [ecx]
    mov       byte ptr [esp+0x4c],al
    mov       eax,dword ptr [esp+0x4c]
    and       eax,0x000000ff
    mov       ecx,eax
    xor       ecx,0x00000040
    mov       cl,byte ptr [ecx+csc_tabp]
    xor       cl,byte ptr [eax+csc_tabp]
    mov       eax,dword ptr [esp+0x18]
    mov       dword ptr [esi+edi-0xe8],eax
    mov       esi,dword ptr [esp+0x44]
    mov       byte ptr [esp+0x4c],cl
    xor       ecx,ecx
    mov       eax,dword ptr [esp+0x4c]
    mov       dword ptr [esp+0x48],esi
    and       eax,0x000000ff
    mov       dword ptr [esp+0x154],eax
    mov       eax,dword ptr [esp+0x1c]
    add       eax,0xffffff1c
X$4:
    mov       edi,dword ptr [esp+0x154]
    mov       esi,0x00000001
    shl       esi,cl
    test      edi,esi
    lje        X$7
    mov       esi,dword ptr [esp+0x48]
    mov       edi,dword ptr [esp+0x10]
    mov       dword ptr [eax],esi
    inc       edi
    mov       esi,edx
    mov       dword ptr [esp+0x10],edi
    shr       esi,0x00000004
    mov       edi,edx
    add       eax,0x00000004
    shl       esi,0x00000003
    and       edi,0x00000007
    add       eax,0x00000004
    add       edi,esi
    lea       edi,[ebp+edi*4]
    mov       dword ptr [eax-0x4],edi
    mov       edi,dword ptr [esp+0x10]
    inc       edi
    mov       dword ptr [esp+0x10],edi
    mov       edi,edx
    and       edi,0x0000000f
    cmp       edi,0x00000007
    ja        X$5
    add       edi,esi
    lea       esi,[ebx+edi*4]
    mov       dword ptr [eax],esi
    jmp       X$6
X$5:
    lea       edi,[edx+0x1]
    add       eax,0x00000004
    and       edi,0x00000007
    add       edi,esi
    shl       edi,0x00000002
    lea       esi,[edi+ebx]
    mov       dword ptr [eax-0x4],esi
    mov       esi,dword ptr [esp+0x10]
    inc       esi
    test      dl,0x01
    mov       dword ptr [esp+0x10],esi
    je        X$7
    add       edi,ebp
    mov       dword ptr [eax],edi
X$6:
    mov       edi,dword ptr [esp+0x10]
    inc       edi
    add       eax,0x00000004
    mov       dword ptr [esp+0x10],edi
X$7:
    mov       esi,dword ptr [esp+0x48]
    inc       ecx
    add       esi,0x00000020
    add       edx,0x00000008
    cmp       ecx,0x00000008
    mov       dword ptr [esp+0x48],esi
    ljl        X$4
    mov       ecx,dword ptr [esp+0x10]
    mov       eax,dword ptr [esp+0x20]
    mov       edi,dword ptr [esp+0x180]
    mov       esi,dword ptr [esp+0x14]
    lea       edx,[eax+ecx-0x3a]
    mov       ecx,dword ptr [esp+0x44]
    add       eax,0x0000001d
    add       ecx,0x00000004
    mov       dword ptr [edi+edx*4],0x00000000
    mov       edx,dword ptr [esp+0x50]
    mov       dword ptr [esp+0x20],eax
    mov       eax,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0x44],ecx
    mov       ecx,dword ptr [esp+0x40]
    inc       edx
    add       esi,0x00000074
    dec       eax
    dec       ecx
    mov       dword ptr [esp+0x3c],eax
    mov       eax,dword ptr [esp+0x18]
    mov       dword ptr [esp+0x40],ecx
    mov       ecx,dword ptr [esp+0x1c]
    add       eax,0x00000020
    add       ecx,0x00000074
    cmp       esi,0x000003a0
    mov       dword ptr [esp+0x50],edx
    mov       dword ptr [esp+0x14],esi
    mov       dword ptr [esp+0x18],eax
    mov       dword ptr [esp+0x1c],ecx
    ljl        X$3
    mov       eax,dword ptr [esp+0x160]
    lea       ecx,[ebx+0x8]
    mov       dword ptr [esp+0x50],ecx
    mov       ecx,dword ptr [esp+0x178]
    mov       edx,ebx
    add       ecx,0x00000024
    lea       esi,[ebp+0x18]
    sub       edx,ebp
    mov       dword ptr [esp+0x3c],esi
    mov       dword ptr [esp+0x154],edx
    mov       dword ptr [esp+0x4c],0x00000004
X$8:
    mov       edx,dword ptr [ecx-0x4]
    mov       edi,dword ptr [eax+0x20]
    xor       edx,edi
    mov       edi,dword ptr [ecx]
    mov       dword ptr [esp+0x20],edx
    mov       edx,dword ptr [eax+0x24]
    xor       edx,edi
    mov       edi,dword ptr [eax+0x28]
    mov       dword ptr [esp+0x48],edx
    mov       edx,dword ptr [ecx+0x4]
    xor       edx,edi
    mov       edi,dword ptr [eax+0x2c]
    mov       dword ptr [esp+0x34],edx
    mov       edx,dword ptr [ecx+0x8]
    xor       edx,edi
    mov       edi,dword ptr [eax+0x30]
    mov       dword ptr [esp+0x40],edx
    mov       edx,dword ptr [ecx+0xc]
    xor       edx,edi
    mov       edi,dword ptr [eax+0x34]
    mov       dword ptr [esp+0x30],edx
    mov       edx,dword ptr [ecx+0x10]
    xor       edx,edi
    mov       edi,dword ptr [eax+0x38]
    mov       dword ptr [esp+0x38],edx
    mov       edx,dword ptr [ecx+0x14]
    xor       edx,edi
    mov       edi,dword ptr [eax+0x3c]
    mov       dword ptr [esp+0x24],edx
    mov       edx,dword ptr [ecx+0x18]
    xor       edx,edi
    mov       edi,dword ptr [ecx-0x8]
    xor       edi,dword ptr [eax+0x1c]
    mov       dword ptr [esp+0x44],edx
    mov       dword ptr [esp+0x14],edi
    xor       edi,edx
    mov       dword ptr [esi+0x4],edi
    mov       edx,dword ptr [ecx-0xc]
    mov       edi,dword ptr [eax+0x18]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x38]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x154]
    mov       dword ptr [edi+esi],edx
    mov       edi,dword ptr [esp+0x24]
    xor       edx,edi
    mov       dword ptr [esi],edx
    mov       edx,dword ptr [ecx-0x10]
    mov       edi,dword ptr [eax+0x14]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x38]
    mov       dword ptr [esp+0x28],edx
    xor       edx,edi
    mov       dword ptr [esi-0x4],edx
    mov       esi,dword ptr [ecx-0x14]
    mov       edi,dword ptr [eax+0x10]
    mov       edx,dword ptr [esp+0x40]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x30]
    xor       esi,edx
    mov       edx,dword ptr [esp+0x50]
    mov       dword ptr [edx+0x8],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    mov       dword ptr [edi-0x8],esi
    mov       esi,dword ptr [ecx-0x18]
    mov       edi,dword ptr [eax+0xc]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x2c],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    mov       dword ptr [edi-0xc],esi
    mov       esi,dword ptr [ecx-0x1c]
    mov       edi,dword ptr [eax+0x8]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x48]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x34]
    mov       dword ptr [edx],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    add       edx,0x00000020
    mov       dword ptr [esp+0x50],edx
    add       eax,0x00000040
    mov       dword ptr [edi-0x10],esi
    mov       esi,dword ptr [ecx-0x20]
    mov       edi,dword ptr [eax-0x3c]
    add       ecx,0x00000040
    xor       esi,edi
    mov       edi,dword ptr [esp+0x48]
    mov       dword ptr [esp+0x10],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    mov       dword ptr [edi-0x14],esi
    mov       esi,dword ptr [ecx-0x64]
    mov       edi,dword ptr [eax-0x40]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x44]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x20]
    mov       dword ptr [edx-0x28],esi
    xor       esi,edi
    mov       edi,dword ptr [esp+0x3c]
    mov       dword ptr [edi-0x18],esi
    mov       esi,dword ptr [esp+0x14]
    mov       edi,dword ptr [esp+0x24]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x30]
    mov       dword ptr [edx-0xc],esi
    mov       esi,dword ptr [esp+0x28]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x34]
    mov       dword ptr [edx-0x14],esi
    mov       esi,dword ptr [esp+0x2c]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x20]
    mov       dword ptr [edx-0x1c],esi
    mov       esi,dword ptr [esp+0x10]
    xor       esi,edi
    mov       dword ptr [edx-0x24],esi
    mov       esi,dword ptr [esp+0x3c]
    mov       edx,dword ptr [esp+0x4c]
    add       esi,0x00000020
    dec       edx
    mov       dword ptr [esp+0x3c],esi
    mov       dword ptr [esp+0x4c],edx
    ljne       X$8
X$9:
    mov       esi,dword ptr [esp+0x178]
    mov       ecx,0x00000040
    lea       edi,[esp+0x54]
    lea       edx,[esp+0x74]
    repe movsd 
    lea       eax,[esp+0x78]
    push      edx
    lea       ecx,[esp+0x80]
    push      eax
    lea       edx,[esp+0x88]
    push      ecx
    lea       eax,[esp+0x90]
    push      edx
    lea       ecx,[esp+0x98]
    push      eax
    lea       edx,[esp+0xa0]
    push      ecx
    mov       ecx,dword ptr [ebp]
    lea       eax,[esp+0xa8]
    push      edx
    mov       edx,dword ptr [ebp+0x4]
    push      eax
    mov       eax,dword ptr [ebp+0x8]
    push      ecx
    mov       ecx,dword ptr [ebp+0xc]
    push      edx
    mov       edx,dword ptr [ebp+0x10]
    push      eax
    mov       eax,dword ptr [ebp+0x14]
    push      ecx
    mov       ecx,dword ptr [ebp+0x18]
    push      edx
    mov       edx,dword ptr [ebp+0x1c]
    push      eax
    push      ecx
    push      edx
    call      csc_transP
    add       esp,0x00000040
    lea       eax,[esp+0x54]
    lea       ecx,[esp+0x58]
    lea       edx,[esp+0x5c]
    push      eax
    push      ecx
    lea       eax,[esp+0x68]
    push      edx
    lea       ecx,[esp+0x70]
    push      eax
    lea       edx,[esp+0x78]
    push      ecx
    lea       eax,[esp+0x80]
    push      edx
    mov       edx,dword ptr [ebx]
    lea       ecx,[esp+0x88]
    push      eax
    mov       eax,dword ptr [ebx+0x4]
    push      ecx
    mov       ecx,dword ptr [ebx+0x8]
    push      edx
    mov       edx,dword ptr [ebx+0xc]
    push      eax
    mov       eax,dword ptr [ebx+0x10]
    push      ecx
    mov       ecx,dword ptr [ebx+0x14]
    push      edx
    mov       edx,dword ptr [ebx+0x18]
    push      eax
    mov       eax,dword ptr [ebx+0x1c]
    push      ecx
    push      edx
    push      eax
    call      csc_transP
    add       esp,0x00000040
    lea       ecx,[esp+0xb4]
    lea       edx,[esp+0xb8]
    lea       eax,[esp+0xbc]
    push      ecx
    push      edx
    lea       ecx,[esp+0xc8]
    push      eax
    push      ecx
    lea       edx,[esp+0xd4]
    lea       eax,[esp+0xd8]
    push      edx
    lea       ecx,[esp+0xe0]
    push      eax
    mov       eax,dword ptr [ebp+0x20]
    lea       edx,[esp+0xe8]
    push      ecx
    mov       ecx,dword ptr [ebp+0x24]
    push      edx
    mov       edx,dword ptr [ebp+0x28]
    push      eax
    mov       eax,dword ptr [ebp+0x2c]
    push      ecx
    mov       ecx,dword ptr [ebp+0x30]
    push      edx
    mov       edx,dword ptr [ebp+0x34]
    push      eax
    mov       eax,dword ptr [ebp+0x38]
    push      ecx
    mov       ecx,dword ptr [ebp+0x3c]
    push      edx
    push      eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x94]
    lea       eax,[esp+0x98]
    lea       ecx,[esp+0x9c]
    push      edx
    push      eax
    lea       edx,[esp+0xa8]
    push      ecx
    lea       eax,[esp+0xb0]
    push      edx
    lea       ecx,[esp+0xb8]
    push      eax
    lea       edx,[esp+0xc0]
    push      ecx
    mov       ecx,dword ptr [ebx+0x20]
    lea       eax,[esp+0xc8]
    push      edx
    mov       edx,dword ptr [ebx+0x24]
    push      eax
    mov       eax,dword ptr [ebx+0x28]
    push      ecx
    mov       ecx,dword ptr [ebx+0x2c]
    push      edx
    mov       edx,dword ptr [ebx+0x30]
    push      eax
    mov       eax,dword ptr [ebx+0x34]
    push      ecx
    mov       ecx,dword ptr [ebx+0x38]
    push      edx
    mov       edx,dword ptr [ebx+0x3c]
    push      eax
    push      ecx
    push      edx
    call      csc_transP
    add       esp,0x00000040
    lea       eax,[esp+0xf4]
    lea       ecx,[esp+0xf8]
    lea       edx,[esp+0xfc]
    push      eax
    push      ecx
    lea       eax,[esp+0x108]
    push      edx
    lea       ecx,[esp+0x110]
    push      eax
    lea       edx,[esp+0x118]
    push      ecx
    lea       eax,[esp+0x120]
    push      edx
    mov       edx,dword ptr [ebp+0x40]
    lea       ecx,[esp+0x128]
    push      eax
    mov       eax,dword ptr [ebp+0x44]
    push      ecx
    mov       ecx,dword ptr [ebp+0x48]
    push      edx
    push      eax
    push      ecx
    mov       edx,dword ptr [ebp+0x4c]
    mov       eax,dword ptr [ebp+0x50]
    mov       ecx,dword ptr [ebp+0x54]
    push      edx
    mov       edx,dword ptr [ebp+0x58]
    push      eax
    mov       eax,dword ptr [ebp+0x5c]
    push      ecx
    push      edx
    push      eax
    call      csc_transP
    add       esp,0x00000040
    lea       ecx,[esp+0xd4]
    lea       edx,[esp+0xd8]
    lea       eax,[esp+0xdc]
    push      ecx
    push      edx
    lea       ecx,[esp+0xe8]
    push      eax
    lea       edx,[esp+0xf0]
    push      ecx
    lea       eax,[esp+0xf8]
    push      edx
    lea       ecx,[esp+0x100]
    push      eax
    mov       eax,dword ptr [ebx+0x40]
    lea       edx,[esp+0x108]
    push      ecx
    mov       ecx,dword ptr [ebx+0x44]
    push      edx
    mov       edx,dword ptr [ebx+0x48]
    push      eax
    mov       eax,dword ptr [ebx+0x4c]
    push      ecx
    mov       ecx,dword ptr [ebx+0x50]
    push      edx
    mov       edx,dword ptr [ebx+0x54]
    push      eax
    mov       eax,dword ptr [ebx+0x58]
    push      ecx
    mov       ecx,dword ptr [ebx+0x5c]
    push      edx
    push      eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x134]
    lea       eax,[esp+0x138]
    lea       ecx,[esp+0x13c]
    push      edx
    push      eax
    lea       edx,[esp+0x148]
    push      ecx
    lea       eax,[esp+0x150]
    push      edx
    lea       ecx,[esp+0x158]
    push      eax
    lea       edx,[esp+0x160]
    push      ecx
    mov       ecx,dword ptr [ebp+0x60]
    lea       eax,[esp+0x168]
    push      edx
    mov       edx,dword ptr [ebp+0x64]
    push      eax
    mov       eax,dword ptr [ebp+0x68]
    push      ecx
    mov       ecx,dword ptr [ebp+0x6c]
    push      edx
    mov       edx,dword ptr [ebp+0x70]
    push      eax
    mov       eax,dword ptr [ebp+0x74]
    push      ecx
    mov       ecx,dword ptr [ebp+0x78]
    push      edx
    mov       edx,dword ptr [ebp+0x7c]
    push      eax
    push      ecx
    push      edx
    call      csc_transP
    add       esp,0x00000040
    lea       eax,[esp+0x114]
    lea       ecx,[esp+0x118]
    push      eax
    lea       edx,[esp+0x120]
    push      ecx
    lea       eax,[esp+0x128]
    push      edx
    lea       ecx,[esp+0x130]
    push      eax
    lea       edx,[esp+0x138]
    push      ecx
    lea       eax,[esp+0x140]
    push      edx
    mov       edx,dword ptr [ebx+0x60]
    lea       ecx,[esp+0x148]
    push      eax
    mov       eax,dword ptr [ebx+0x64]
    push      ecx
    mov       ecx,dword ptr [ebx+0x68]
    push      edx
    mov       edx,dword ptr [ebx+0x6c]
    push      eax
    mov       eax,dword ptr [ebx+0x70]
    push      ecx
    mov       ecx,dword ptr [ebx+0x74]
    push      edx
    mov       edx,dword ptr [ebx+0x78]
    push      eax
    mov       eax,dword ptr [ebx+0x7c]
    push      ecx
    push      edx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0xd4]
    mov       edi,dword ptr [csc_tabe+0x20]
    mov       esi,dword ptr [csc_tabe+0x24]
    mov       ecx,dword ptr [esp+0xdc]
    mov       edx,dword ptr [csc_tabe+0x28]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x2c]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0xd8]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xe4]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0x30]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0xe0]
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x38]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x34]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0xe8]
    mov       dword ptr [esp+0x88],eax
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x3c]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0xec]
    xor       edx,esi
    mov       esi,dword ptr [esp+0xf0]
    mov       dword ptr [esp+0x64],edx
    mov       edx,dword ptr [esp+0x94]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe]
    mov       dword ptr [esp+0x84],esi
    xor       esi,edx
    mov       edx,dword ptr [csc_tabe+0x4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    add       esp,0x00000040
    xor       edi,edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x60]
    xor       edx,dword ptr [csc_tabe+0xc]
    mov       dword ptr [esp+0x40],ecx
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0x64]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x10]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x14]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0x68]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x18]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x1c]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0x70]
    xor       ecx,edx
    lea       edx,[esp+0x98]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0x94]
    push      ecx
    push      edx
    lea       ecx,[esp+0xa4]
    lea       edx,[esp+0xa8]
    push      ecx
    push      edx
    lea       ecx,[esp+0xb4]
    lea       edx,[esp+0xb8]
    push      ecx
    push      edx
    lea       ecx,[esp+0xc4]
    lea       edx,[esp+0xc8]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x40]
    mov       ecx,esi
    xor       ecx,edx
    mov       edx,edi
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x54]
    lea       eax,[esp+0x58]
    lea       ecx,[esp+0x5c]
    push      edx
    push      eax
    lea       edx,[esp+0x68]
    push      ecx
    lea       eax,[esp+0x70]
    push      edx
    lea       ecx,[esp+0x78]
    push      eax
    lea       edx,[esp+0x80]
    push      ecx
    push      edx
    mov       ecx,dword ptr [esp+0x58]
    lea       eax,[esp+0x8c]
    push      eax
    mov       eax,dword ptr [esp+0x40]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x4c]
    push      esi
    push      edi
    mov       edx,dword ptr [esp+0x44]
    push      ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,ecx
    push      edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x154]
    mov       edi,dword ptr [csc_tabe+0x60]
    mov       esi,dword ptr [csc_tabe+0x64]
    mov       ecx,dword ptr [esp+0x15c]
    mov       edx,dword ptr [csc_tabe+0x68]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x6c]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0x158]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x164]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0x70]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0x160]
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x78]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x74]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x168]
    mov       dword ptr [esp+0x88],eax
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x7c]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0x16c]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x170]
    mov       dword ptr [esp+0x64],edx
    mov       edx,dword ptr [esp+0x114]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x40]
    mov       dword ptr [esp+0x84],esi
    xor       esi,edx
    mov       edx,dword ptr [csc_tabe+0x44]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x118]
    mov       dword ptr [esp+0x80],ecx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x11c]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x48]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x120]
    xor       edx,dword ptr [csc_tabe+0x4c]
    add       esp,0x00000040
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0xe4]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x50]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x54]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0xe8]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xec]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x58]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x5c]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0xf0]
    xor       ecx,edx
    lea       edx,[esp+0x118]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0x114]
    push      ecx
    push      edx
    lea       ecx,[esp+0x124]
    lea       edx,[esp+0x128]
    push      ecx
    push      edx
    lea       ecx,[esp+0x134]
    lea       edx,[esp+0x138]
    push      ecx
    push      edx
    lea       ecx,[esp+0x144]
    lea       edx,[esp+0x148]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x40]
    mov       ecx,esi
    xor       ecx,edx
    mov       edx,edi
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0xd4]
    lea       eax,[esp+0xd8]
    lea       ecx,[esp+0xdc]
    push      edx
    push      eax
    lea       edx,[esp+0xe8]
    push      ecx
    lea       eax,[esp+0xf0]
    push      edx
    lea       ecx,[esp+0xf8]
    push      eax
    lea       edx,[esp+0x100]
    push      ecx
    mov       ecx,dword ptr [esp+0x54]
    lea       eax,[esp+0x108]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x40]
    push      esi
    xor       edi,eax
    push      edi
    mov       eax,dword ptr [esp+0x54]
    mov       edx,dword ptr [esp+0x44]
    push      ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,ecx
    push      edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0xf4]
    mov       edi,dword ptr [csc_tabe+0xa0]
    mov       esi,dword ptr [csc_tabe+0xa4]
    mov       ecx,dword ptr [esp+0xfc]
    mov       edx,dword ptr [csc_tabe+0xa8]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0xac]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0xf8]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x104]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0xb0]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0x100]
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0xb8]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0xb4]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x108]
    mov       dword ptr [esp+0x88],eax
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0xbc]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0x10c]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x110]
    mov       dword ptr [esp+0x64],edx
    mov       edx,dword ptr [esp+0xb4]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x80]
    mov       dword ptr [esp+0x84],esi
    xor       esi,edx
    mov       edx,dword ptr [csc_tabe+0x84]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xb8]
    mov       dword ptr [esp+0x80],ecx
    xor       edi,edx
    mov       edx,dword ptr [esp+0xbc]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x88]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xc0]
    xor       edx,dword ptr [csc_tabe+0x8c]
    add       esp,0x00000040
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0x84]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x90]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x94]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0x88]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x98]
    xor       ecx,edx
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0x90]
    mov       edx,dword ptr [csc_tabe+0x9c]
    xor       ecx,edx
    lea       edx,[esp+0xb8]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0xb4]
    push      ecx
    push      edx
    lea       ecx,[esp+0xc4]
    lea       edx,[esp+0xc8]
    push      ecx
    push      edx
    lea       ecx,[esp+0xd4]
    lea       edx,[esp+0xd8]
    push      ecx
    push      edx
    lea       ecx,[esp+0xe4]
    lea       edx,[esp+0xe8]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x40]
    mov       ecx,esi
    xor       ecx,edx
    mov       edx,edi
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x74]
    lea       eax,[esp+0x78]
    lea       ecx,[esp+0x7c]
    push      edx
    push      eax
    lea       edx,[esp+0x88]
    push      ecx
    lea       eax,[esp+0x90]
    push      edx
    lea       ecx,[esp+0x98]
    push      eax
    lea       edx,[esp+0xa0]
    push      ecx
    mov       ecx,dword ptr [esp+0x54]
    lea       eax,[esp+0xa8]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x40]
    mov       edx,dword ptr [esp+0x3c]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x4c]
    push      esi
    push      edi
    push      ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,ecx
    push      eax
    push      edx
    mov       eax,dword ptr [esp+0x5c]
    mov       ecx,dword ptr [esp+0x64]
    mov       edx,dword ptr [esp+0x58]
    xor       eax,ecx
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x174]
    mov       edi,dword ptr [csc_tabe+0xe0]
    mov       esi,dword ptr [csc_tabe+0xe4]
    mov       ecx,dword ptr [esp+0x17c]
    mov       edx,dword ptr [csc_tabe+0xe8]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0xec]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0x178]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x184]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0xf0]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0x180]
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0xf8]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0xf4]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x188]
    mov       dword ptr [esp+0x88],eax
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0xfc]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0x18c]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x190]
    mov       dword ptr [esp+0x64],edx
    mov       edx,dword ptr [esp+0x134]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0xc0]
    mov       dword ptr [esp+0x84],esi
    xor       esi,edx
    mov       edx,dword ptr [csc_tabe+0xc4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x138]
    mov       dword ptr [esp+0x80],ecx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x13c]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0xc8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x140]
    xor       edx,dword ptr [csc_tabe+0xcc]
    add       esp,0x00000040
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0x104]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0xd0]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0xd4]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0x108]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x10c]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0xd8]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0xdc]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0x110]
    xor       ecx,edx
    lea       edx,[esp+0x138]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0x134]
    push      ecx
    push      edx
    lea       ecx,[esp+0x144]
    lea       edx,[esp+0x148]
    push      ecx
    push      edx
    lea       ecx,[esp+0x154]
    lea       edx,[esp+0x158]
    push      ecx
    push      edx
    lea       ecx,[esp+0x164]
    lea       edx,[esp+0x168]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x40]
    mov       ecx,esi
    xor       ecx,edx
    mov       edx,edi
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0xf4]
    lea       eax,[esp+0xf8]
    lea       ecx,[esp+0xfc]
    push      edx
    push      eax
    lea       edx,[esp+0x108]
    push      ecx
    lea       eax,[esp+0x110]
    push      edx
    lea       ecx,[esp+0x118]
    push      eax
    lea       edx,[esp+0x120]
    push      ecx
    mov       ecx,dword ptr [esp+0x54]
    lea       eax,[esp+0x128]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x40]
    push      esi
    xor       edi,eax
    mov       eax,dword ptr [esp+0x50]
    push      edi
    mov       edx,dword ptr [esp+0x44]
    push      ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,ecx
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,dword ptr [esp+0x60]
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x114]
    mov       edi,dword ptr [csc_tabe+0x120]
    mov       esi,dword ptr [csc_tabe+0x124]
    mov       ecx,dword ptr [esp+0x11c]
    mov       edx,dword ptr [csc_tabe+0x128]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x12c]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0x118]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x124]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0x130]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0x120]
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x138]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x134]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x128]
    mov       dword ptr [esp+0x88],eax
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x13c]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0x12c]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x130]
    mov       dword ptr [esp+0x64],edx
    mov       edx,dword ptr [esp+0x94]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x100]
    mov       dword ptr [esp+0x84],esi
    xor       esi,edx
    mov       edx,dword ptr [csc_tabe+0x104]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    mov       dword ptr [esp+0x80],ecx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x9c]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x108]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xa0]
    xor       edx,dword ptr [csc_tabe+0x10c]
    add       esp,0x00000040
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0x64]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x110]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x114]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0x68]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x118]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x11c]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0x70]
    xor       ecx,edx
    lea       edx,[esp+0xd8]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0xd4]
    push      ecx
    lea       ecx,[esp+0xe0]
    push      edx
    push      ecx
    lea       edx,[esp+0xec]
    lea       ecx,[esp+0xf0]
    push      edx
    push      ecx
    lea       edx,[esp+0xfc]
    lea       ecx,[esp+0x100]
    push      edx
    lea       edx,[esp+0x108]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x40]
    mov       ecx,esi
    xor       ecx,edx
    mov       edx,edi
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x54]
    lea       eax,[esp+0x58]
    lea       ecx,[esp+0x5c]
    push      edx
    push      eax
    lea       edx,[esp+0x68]
    push      ecx
    lea       eax,[esp+0x70]
    push      edx
    lea       ecx,[esp+0x78]
    push      eax
    lea       edx,[esp+0x80]
    push      ecx
    mov       ecx,dword ptr [esp+0x54]
    lea       eax,[esp+0x88]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x40]
    push      esi
    xor       edi,eax
    mov       eax,dword ptr [esp+0x50]
    push      edi
    mov       edx,dword ptr [esp+0x44]
    push      ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,ecx
    push      edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x134]
    mov       edi,dword ptr [csc_tabe+0x160]
    add       esp,0x00000040
    mov       esi,dword ptr [csc_tabe+0x164]
    mov       ecx,dword ptr [esp+0xfc]
    mov       edx,dword ptr [csc_tabe+0x168]
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x16c]
    mov       dword ptr [esp+0x20],eax
    mov       eax,dword ptr [esp+0xf8]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x104]
    xor       eax,esi
    mov       esi,dword ptr [csc_tabe+0x170]
    mov       dword ptr [esp+0x34],ecx
    mov       ecx,dword ptr [esp+0x100]
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x178]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x174]
    mov       dword ptr [esp+0x30],edx
    mov       edx,dword ptr [esp+0x108]
    mov       dword ptr [esp+0x48],eax
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x17c]
    mov       dword ptr [esp+0x38],edx
    mov       edx,dword ptr [esp+0x10c]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x110]
    mov       dword ptr [esp+0x24],edx
    mov       edx,dword ptr [esp+0x74]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x140]
    mov       dword ptr [esp+0x44],esi
    xor       esi,edx
    mov       edx,dword ptr [csc_tabe+0x144]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x40],ecx
    xor       edi,edx
    mov       edx,dword ptr [esp+0x7c]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x148]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x80]
    xor       edx,dword ptr [csc_tabe+0x14c]
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0x84]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x150]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x154]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0x88]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x158]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x15c]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0x90]
    xor       ecx,edx
    lea       edx,[esp+0xf8]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0xf4]
    push      ecx
    push      edx
    lea       ecx,[esp+0x104]
    lea       edx,[esp+0x108]
    push      ecx
    push      edx
    lea       ecx,[esp+0x114]
    lea       edx,[esp+0x118]
    push      ecx
    push      edx
    lea       ecx,[esp+0x124]
    lea       edx,[esp+0x128]
    push      ecx
    push      edx
    mov       ecx,esi
    mov       edx,dword ptr [esp+0x40]
    xor       ecx,edx
    mov       edx,edi
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x74]
    lea       eax,[esp+0x78]
    lea       ecx,[esp+0x7c]
    push      edx
    push      eax
    lea       edx,[esp+0x88]
    push      ecx
    lea       eax,[esp+0x90]
    push      edx
    lea       ecx,[esp+0x98]
    push      eax
    lea       edx,[esp+0xa0]
    push      ecx
    mov       ecx,dword ptr [esp+0x54]
    lea       eax,[esp+0xa8]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x40]
    push      esi
    xor       edi,eax
    mov       eax,dword ptr [esp+0x50]
    push      edi
    mov       edx,dword ptr [esp+0x44]
    push      ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,ecx
    push      edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x154]
    mov       edi,dword ptr [csc_tabe+0x1a0]
    mov       esi,dword ptr [csc_tabe+0x1a4]
    mov       ecx,dword ptr [esp+0x15c]
    xor       eax,edi
    add       esp,0x00000040
    mov       dword ptr [esp+0x20],eax
    mov       eax,dword ptr [esp+0x118]
    xor       eax,esi
    mov       dword ptr [esp+0x48],eax
    mov       edx,dword ptr [csc_tabe+0x1a8]
    mov       esi,dword ptr [csc_tabe+0x1b0]
    mov       edi,dword ptr [csc_tabe+0x1ac]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x124]
    mov       dword ptr [esp+0x34],ecx
    mov       ecx,dword ptr [esp+0x120]
    xor       edx,esi
    mov       esi,dword ptr [csc_tabe+0x1b8]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1b4]
    mov       dword ptr [esp+0x30],edx
    mov       edx,dword ptr [esp+0x128]
    mov       dword ptr [esp+0x40],ecx
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x1bc]
    mov       dword ptr [esp+0x38],edx
    mov       edx,dword ptr [esp+0x12c]
    xor       edx,esi
    mov       esi,dword ptr [esp+0x130]
    mov       dword ptr [esp+0x24],edx
    mov       edx,dword ptr [esp+0x94]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x180]
    mov       dword ptr [esp+0x44],esi
    xor       esi,edx
    mov       edx,dword ptr [csc_tabe+0x184]
    xor       esi,edi
    mov       edi,dword ptr [esp+0x98]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x9c]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x188]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xa0]
    xor       edx,dword ptr [csc_tabe+0x18c]
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0xa4]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x190]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x194]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0xa8]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xac]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x198]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x19c]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0xb0]
    xor       ecx,edx
    lea       edx,[esp+0x118]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0x114]
    push      ecx
    push      edx
    lea       ecx,[esp+0x124]
    lea       edx,[esp+0x128]
    push      ecx
    push      edx
    lea       ecx,[esp+0x134]
    lea       edx,[esp+0x138]
    push      ecx
    push      edx
    lea       ecx,[esp+0x144]
    lea       edx,[esp+0x148]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x40]
    mov       ecx,esi
    xor       ecx,edx
    mov       edx,edi
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    mov       ecx,dword ptr [esp+0x44]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x94]
    lea       eax,[esp+0x98]
    lea       ecx,[esp+0x9c]
    push      edx
    push      eax
    lea       edx,[esp+0xa8]
    push      ecx
    lea       eax,[esp+0xb0]
    push      edx
    lea       ecx,[esp+0xb8]
    push      eax
    lea       edx,[esp+0xc0]
    push      ecx
    mov       ecx,dword ptr [esp+0x54]
    lea       eax,[esp+0xc8]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x40]
    mov       edx,dword ptr [esp+0x3c]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x4c]
    push      esi
    push      edi
    push      ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,ecx
    mov       esi,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,esi
    push      eax
    call      csc_transP
    mov       edx,dword ptr [esp+0x174]
    mov       eax,dword ptr [csc_tabe+0x1e0]
    mov       ecx,dword ptr [esp+0x17c]
    mov       esi,dword ptr [csc_tabe+0x1e8]
    mov       edi,dword ptr [csc_tabe+0x1e4]
    xor       edx,eax
    mov       eax,dword ptr [esp+0x178]
    xor       ecx,esi
    xor       eax,edi
    mov       edi,dword ptr [csc_tabe+0x1ec]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0x180]
    add       esp,0x00000040
    xor       ecx,edi
    mov       dword ptr [esp+0x20],edx
    mov       dword ptr [esp+0x48],eax
    mov       dword ptr [esp+0x40],ecx
    mov       ecx,dword ptr [esp+0x144]
    mov       esi,dword ptr [csc_tabe+0x1f0]
    mov       edi,dword ptr [csc_tabe+0x1f4]
    xor       ecx,esi
    mov       esi,dword ptr [csc_tabe+0x1f8]
    mov       dword ptr [esp+0x30],ecx
    mov       ecx,dword ptr [esp+0x148]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1fc]
    mov       dword ptr [esp+0x38],ecx
    mov       ecx,dword ptr [esp+0x14c]
    xor       ecx,esi
    mov       esi,dword ptr [esp+0x150]
    mov       dword ptr [esp+0x24],ecx
    mov       ecx,dword ptr [esp+0xb4]
    xor       esi,edi
    mov       edi,dword ptr [csc_tabe+0x1c0]
    mov       dword ptr [esp+0x44],esi
    xor       esi,ecx
    mov       ecx,dword ptr [csc_tabe+0x1c4]
    xor       esi,edi
    mov       edi,dword ptr [esp+0xb8]
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0xbc]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x1c8]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xc0]
    xor       ecx,dword ptr [csc_tabe+0x1cc]
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],ecx
    mov       ecx,dword ptr [esp+0x40]
    xor       ecx,dword ptr [esp+0xc4]
    xor       ecx,dword ptr [csc_tabe+0x1d0]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0xc8]
    xor       ecx,dword ptr [csc_tabe+0x1d4]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,dword ptr [esp+0xcc]
    xor       ecx,dword ptr [csc_tabe+0x1d8]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0xd0]
    xor       ecx,dword ptr [csc_tabe+0x1dc]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0x134]
    push      ecx
    lea       ecx,[esp+0x13c]
    push      ecx
    lea       ecx,[esp+0x144]
    push      ecx
    lea       ecx,[esp+0x14c]
    push      ecx
    lea       ecx,[esp+0x154]
    push      ecx
    lea       ecx,[esp+0x15c]
    push      ecx
    lea       ecx,[esp+0x164]
    push      ecx
    lea       ecx,[esp+0x16c]
    push      ecx
    mov       ecx,esi
    xor       ecx,edx
    mov       edx,edi
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    mov       eax,dword ptr [esp+0x4c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    xor       eax,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0xb4]
    lea       eax,[esp+0xb8]
    lea       ecx,[esp+0xbc]
    push      edx
    push      eax
    lea       edx,[esp+0xc8]
    push      ecx
    lea       eax,[esp+0xd0]
    push      edx
    lea       ecx,[esp+0xd8]
    push      eax
    lea       edx,[esp+0xe0]
    push      ecx
    mov       ecx,dword ptr [esp+0x54]
    lea       eax,[esp+0xe8]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x40]
    push      esi
    xor       edi,eax
    mov       eax,dword ptr [esp+0x50]
    push      edi
    mov       edx,dword ptr [esp+0x44]
    push      ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,ecx
    push      edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       ecx,dword ptr [esp+0x198]
    mov       eax,dword ptr [esp+0x1a0]
    add       esp,0x00000040
    mov       dword ptr [esp+0x4c],eax
    lea       esi,[ecx+0x300]
    mov       edi,offset csc_tabc+0x100
    mov       dword ptr [esp+0x154],0x00000007
    mov       dword ptr [esp+0x50],0x00000008
    lea       ecx,[esi+0x40]
    mov       dword ptr [esp+0x10],ecx
    jmp       X$12
X$10:
    mov       edi,dword ptr [esp+0x50]
    mov       eax,dword ptr [esp+0x4c]
    lea       ecx,[esi+0x40]
    mov       dword ptr [esp+0x50],0x00000008
    mov       dword ptr [esp+0x10],ecx
    jmp       X$12
X$11:
    mov       ecx,dword ptr [esp+0x10]
    mov       eax,dword ptr [esp+0x4c]
X$12:
    lea       edx,[ecx-0x20]
    push      esi
    push      edx
    lea       edx,[ecx+0x20]
    push      ecx
    push      edx
    lea       edx,[ecx+0x40]
    push      edx
    lea       edx,[ecx+0x60]
    push      edx
    lea       edx,[ecx+0x80]
    add       ecx,0x000000a0
    push      edx
    mov       edx,dword ptr [eax]
    push      ecx
    mov       ecx,dword ptr [edi]
    xor       ecx,edx
    mov       edx,dword ptr [eax+0x4]
    push      ecx
    mov       ecx,dword ptr [edi+0x4]
    xor       edx,ecx
    mov       ecx,dword ptr [eax+0x8]
    push      edx
    mov       edx,dword ptr [edi+0x8]
    xor       ecx,edx
    mov       edx,dword ptr [eax+0xc]
    push      ecx
    mov       ecx,dword ptr [edi+0xc]
    xor       edx,ecx
    mov       ecx,dword ptr [eax+0x10]
    push      edx
    mov       edx,dword ptr [edi+0x10]
    xor       ecx,edx
    mov       edx,dword ptr [eax+0x14]
    push      ecx
    mov       ecx,dword ptr [edi+0x14]
    xor       edx,ecx
    mov       ecx,dword ptr [eax+0x18]
    push      edx
    mov       edx,dword ptr [edi+0x18]
    xor       ecx,edx
    mov       edx,dword ptr [eax+0x1c]
    push      ecx
    mov       ecx,dword ptr [edi+0x1c]
    xor       edx,ecx
    push      edx
    call      csc_transP
    mov       eax,dword ptr [esp+0x8c]
    mov       ecx,dword ptr [esp+0x50]
    add       eax,0x00000020
    add       esp,0x00000040
    mov       dword ptr [esp+0x4c],eax
    mov       eax,dword ptr [esp+0x50]
    add       edi,0x00000020
    add       esi,0x00000004
    add       ecx,0x00000004
    dec       eax
    mov       dword ptr [esp+0x10],ecx
    mov       dword ptr [esp+0x50],eax
    ljne       X$11
    mov       eax,dword ptr [esi-0x200]
    mov       ecx,dword ptr [esi]
    sub       esi,0x00000020
    xor       ecx,eax
    mov       dword ptr [esp+0x50],edi
    mov       edi,dword ptr [esp+0x74]
    mov       edx,dword ptr [esi+0x24]
    mov       dword ptr [esi+0x20],ecx
    mov       eax,ecx
    mov       ecx,dword ptr [esi-0x1dc]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x28]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x78]
    mov       dword ptr [esp+0x20],eax
    mov       eax,edx
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x7c]
    mov       dword ptr [esi+0x24],edx
    mov       edx,eax
    mov       eax,dword ptr [esi-0x1d8]
    xor       edi,eax
    mov       eax,edi
    mov       dword ptr [esi+0x28],edi
    mov       edi,dword ptr [esi+0x2c]
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1d4]
    mov       dword ptr [esp+0x34],eax
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x80]
    mov       eax,edi
    mov       dword ptr [esi+0x2c],edi
    mov       edi,dword ptr [esi+0x30]
    xor       eax,ecx
    mov       ecx,eax
    mov       eax,dword ptr [esi-0x1d0]
    xor       edi,eax
    mov       dword ptr [esp+0x40],ecx
    mov       dword ptr [esi+0x30],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x84]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x34]
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esi-0x1cc]
    xor       edi,eax
    mov       dword ptr [esi+0x34],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x88]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x38]
    mov       dword ptr [esp+0x38],eax
    mov       eax,dword ptr [esi-0x1c8]
    xor       edi,eax
    mov       dword ptr [esi+0x38],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x8c]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x3c]
    mov       dword ptr [esp+0x24],eax
    mov       eax,dword ptr [esi-0x1c4]
    xor       edi,eax
    mov       dword ptr [esi+0x3c],edi
    mov       eax,edi
    xor       eax,dword ptr [esp+0x90]
    mov       edi,eax
    mov       eax,dword ptr [esi-0x200]
    xor       dword ptr [esi],eax
    mov       eax,dword ptr [esi]
    mov       dword ptr [esp+0x44],edi
    xor       eax,edi
    mov       edi,dword ptr [esp+0x54]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x4]
    mov       dword ptr [esp+0x10],eax
    mov       eax,dword ptr [esi-0x1fc]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x58]
    mov       dword ptr [esi+0x4],edi
    xor       edi,eax
    mov       eax,dword ptr [esi-0x1f8]
    xor       dword ptr [esi+0x8],eax
    mov       eax,dword ptr [esi+0x8]
    xor       eax,edx
    xor       eax,dword ptr [esp+0x5c]
    mov       dword ptr [esp+0x3c],eax
    mov       eax,dword ptr [esi-0x1f4]
    xor       dword ptr [esi+0xc],eax
    mov       eax,dword ptr [esi+0xc]
    xor       eax,dword ptr [esp+0x60]
    mov       dword ptr [esp+0x2c],eax
    mov       eax,dword ptr [esi-0x1f0]
    xor       dword ptr [esi+0x10],eax
    mov       eax,dword ptr [esi+0x10]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x64]
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1ec]
    mov       dword ptr [esp+0x1c],eax
    mov       eax,dword ptr [esi+0x14]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x68]
    mov       dword ptr [esi+0x14],eax
    xor       eax,ecx
    mov       ecx,dword ptr [esi+0x18]
    mov       dword ptr [esp+0x28],eax
    mov       eax,dword ptr [esi-0x1e8]
    xor       ecx,eax
    mov       dword ptr [esi+0x18],ecx
    mov       eax,ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1e4]
    mov       dword ptr [esp+0x18],eax
    mov       eax,dword ptr [esi+0x1c]
    xor       eax,ecx
    mov       dword ptr [esi+0x1c],eax
    mov       ecx,eax
    mov       eax,dword ptr [esp+0x70]
    xor       ecx,eax
    lea       eax,[esp+0x74]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0x78]
    push      eax
    push      ecx
    lea       eax,[esp+0x84]
    lea       ecx,[esp+0x88]
    push      eax
    push      ecx
    lea       eax,[esp+0x94]
    lea       ecx,[esp+0x98]
    push      eax
    push      ecx
    lea       eax,[esp+0xa4]
    lea       ecx,[esp+0xa8]
    push      eax
    mov       eax,dword ptr [esp+0x2c]
    push      ecx
    mov       ecx,dword ptr [esp+0x40]
    xor       eax,ecx
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x6c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       eax,dword ptr [esp+0x4c]
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x54]
    lea       eax,[esp+0x58]
    lea       ecx,[esp+0x5c]
    push      edx
    push      eax
    lea       edx,[esp+0x68]
    push      ecx
    lea       eax,[esp+0x70]
    push      edx
    lea       ecx,[esp+0x78]
    push      eax
    lea       edx,[esp+0x80]
    push      ecx
    mov       ecx,dword ptr [esp+0x28]
    lea       eax,[esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    push      ecx
    mov       ecx,dword ptr [esp+0x44]
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x58]
    push      edi
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x58]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x5c]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    push      ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,ecx
    push      eax
    call      csc_transP
    mov       edx,dword ptr [esi-0x1a0]
    mov       eax,dword ptr [esi+0x60]
    mov       edi,dword ptr [esp+0xf4]
    add       esi,0x00000040
    xor       eax,edx
    mov       ecx,dword ptr [esp+0xf8]
    mov       edx,dword ptr [esi+0x24]
    mov       dword ptr [esi+0x20],eax
    xor       eax,edi
    mov       edi,dword ptr [esi+0x28]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esi-0x1dc]
    xor       edx,eax
    add       esp,0x00000040
    mov       eax,edx
    mov       dword ptr [esi+0x24],edx
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1d8]
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0xbc]
    mov       edx,eax
    mov       eax,edi
    xor       eax,ecx
    mov       dword ptr [esi+0x28],edi
    mov       edi,dword ptr [esi+0x2c]
    mov       dword ptr [esp+0x34],eax
    mov       eax,dword ptr [esi-0x1d4]
    xor       edi,eax
    mov       dword ptr [esi+0x2c],edi
    mov       eax,edi
    mov       ecx,dword ptr [esp+0xc0]
    mov       edi,dword ptr [esi+0x30]
    xor       eax,ecx
    mov       ecx,eax
    mov       eax,dword ptr [esi-0x1d0]
    xor       edi,eax
    mov       dword ptr [esp+0x40],ecx
    mov       dword ptr [esi+0x30],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0xc4]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x34]
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esi-0x1cc]
    xor       edi,eax
    mov       dword ptr [esi+0x34],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0xc8]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x38]
    mov       dword ptr [esp+0x38],eax
    mov       eax,dword ptr [esi-0x1c8]
    xor       edi,eax
    mov       dword ptr [esi+0x38],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0xcc]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x3c]
    mov       dword ptr [esp+0x24],eax
    mov       eax,dword ptr [esi-0x1c4]
    xor       edi,eax
    mov       dword ptr [esi+0x3c],edi
    mov       eax,edi
    xor       eax,dword ptr [esp+0xd0]
    mov       edi,eax
    mov       eax,dword ptr [esi-0x200]
    xor       dword ptr [esi],eax
    mov       eax,dword ptr [esi]
    mov       dword ptr [esp+0x44],edi
    xor       eax,edi
    mov       edi,dword ptr [esp+0x94]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x4]
    mov       dword ptr [esp+0x10],eax
    mov       eax,dword ptr [esi-0x1fc]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x98]
    mov       dword ptr [esi+0x4],edi
    xor       edi,eax
    mov       eax,dword ptr [esi-0x1f8]
    xor       dword ptr [esi+0x8],eax
    mov       eax,dword ptr [esi+0x8]
    xor       eax,edx
    xor       eax,dword ptr [esp+0x9c]
    mov       dword ptr [esp+0x3c],eax
    mov       eax,dword ptr [esi-0x1f4]
    xor       dword ptr [esi+0xc],eax
    mov       eax,dword ptr [esi+0xc]
    xor       eax,dword ptr [esp+0xa0]
    mov       dword ptr [esp+0x2c],eax
    mov       eax,dword ptr [esi-0x1f0]
    xor       dword ptr [esi+0x10],eax
    mov       eax,dword ptr [esi+0x10]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xa4]
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1ec]
    mov       dword ptr [esp+0x1c],eax
    mov       eax,dword ptr [esi+0x14]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xa8]
    mov       dword ptr [esi+0x14],eax
    xor       eax,ecx
    mov       ecx,dword ptr [esi+0x18]
    mov       dword ptr [esp+0x28],eax
    mov       eax,dword ptr [esi-0x1e8]
    xor       ecx,eax
    mov       dword ptr [esi+0x18],ecx
    mov       eax,ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xac]
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1e4]
    mov       dword ptr [esp+0x18],eax
    mov       eax,dword ptr [esi+0x1c]
    xor       eax,ecx
    mov       dword ptr [esi+0x1c],eax
    mov       ecx,eax
    mov       eax,dword ptr [esp+0xb0]
    xor       ecx,eax
    lea       eax,[esp+0xb4]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0xb8]
    push      eax
    push      ecx
    lea       eax,[esp+0xc4]
    lea       ecx,[esp+0xc8]
    push      eax
    push      ecx
    lea       eax,[esp+0xd4]
    lea       ecx,[esp+0xd8]
    push      eax
    push      ecx
    lea       eax,[esp+0xe4]
    lea       ecx,[esp+0xe8]
    push      eax
    mov       eax,dword ptr [esp+0x2c]
    push      ecx
    mov       ecx,dword ptr [esp+0x40]
    xor       eax,ecx
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x6c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x94]
    lea       eax,[esp+0x98]
    lea       ecx,[esp+0x9c]
    push      edx
    push      eax
    lea       edx,[esp+0xa8]
    push      ecx
    lea       eax,[esp+0xb0]
    push      edx
    lea       ecx,[esp+0xb8]
    push      eax
    lea       edx,[esp+0xc0]
    push      ecx
    mov       ecx,dword ptr [esp+0x28]
    lea       eax,[esp+0xc8]
    push      edx
    push      eax
    push      ecx
    mov       ecx,dword ptr [esp+0x44]
    xor       edi,ecx
    mov       eax,dword ptr [esp+0x50]
    mov       ecx,dword ptr [esp+0x58]
    mov       edx,dword ptr [esp+0x60]
    push      edi
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x58]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x5c]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    push      ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,ecx
    push      eax
    call      csc_transP
    mov       edx,dword ptr [esi-0x1a0]
    mov       eax,dword ptr [esi+0x60]
    mov       edi,dword ptr [esp+0x134]
    add       esi,0x00000040
    xor       eax,edx
    mov       ecx,dword ptr [esp+0x138]
    mov       edx,dword ptr [esi+0x24]
    mov       dword ptr [esi+0x20],eax
    xor       eax,edi
    mov       edi,dword ptr [esi+0x28]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esi-0x1dc]
    xor       edx,eax
    add       esp,0x00000040
    mov       eax,edx
    mov       dword ptr [esi+0x24],edx
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1d8]
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0xfc]
    mov       edx,eax
    mov       eax,edi
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x100]
    mov       dword ptr [esi+0x28],edi
    mov       edi,dword ptr [esi+0x2c]
    mov       dword ptr [esp+0x34],eax
    mov       eax,dword ptr [esi-0x1d4]
    xor       edi,eax
    mov       eax,edi
    mov       dword ptr [esi+0x2c],edi
    mov       edi,dword ptr [esi+0x30]
    xor       eax,ecx
    mov       ecx,eax
    mov       eax,dword ptr [esi-0x1d0]
    xor       edi,eax
    mov       dword ptr [esp+0x40],ecx
    mov       dword ptr [esi+0x30],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x104]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x34]
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esi-0x1cc]
    xor       edi,eax
    mov       dword ptr [esi+0x34],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x108]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x38]
    mov       dword ptr [esp+0x38],eax
    mov       eax,dword ptr [esi-0x1c8]
    xor       edi,eax
    mov       dword ptr [esi+0x38],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x10c]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x3c]
    mov       dword ptr [esp+0x24],eax
    mov       eax,dword ptr [esi-0x1c4]
    xor       edi,eax
    mov       dword ptr [esi+0x3c],edi
    mov       eax,edi
    xor       eax,dword ptr [esp+0x110]
    mov       edi,eax
    mov       eax,dword ptr [esi-0x200]
    xor       dword ptr [esi],eax
    mov       eax,dword ptr [esi]
    xor       eax,edi
    mov       dword ptr [esp+0x44],edi
    mov       edi,dword ptr [esp+0xd4]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x4]
    mov       dword ptr [esp+0x10],eax
    mov       eax,dword ptr [esi-0x1fc]
    xor       edi,eax
    mov       eax,dword ptr [esp+0xd8]
    mov       dword ptr [esi+0x4],edi
    xor       edi,eax
    mov       eax,dword ptr [esi-0x1f8]
    xor       dword ptr [esi+0x8],eax
    mov       eax,dword ptr [esi+0x8]
    xor       eax,edx
    xor       eax,dword ptr [esp+0xdc]
    mov       dword ptr [esp+0x3c],eax
    mov       eax,dword ptr [esi-0x1f4]
    xor       dword ptr [esi+0xc],eax
    mov       eax,dword ptr [esi+0xc]
    xor       eax,dword ptr [esp+0xe0]
    mov       dword ptr [esp+0x2c],eax
    mov       eax,dword ptr [esi-0x1f0]
    xor       dword ptr [esi+0x10],eax
    mov       eax,dword ptr [esi+0x10]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xe4]
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1ec]
    mov       dword ptr [esp+0x1c],eax
    mov       eax,dword ptr [esi+0x14]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xe8]
    mov       dword ptr [esi+0x14],eax
    xor       eax,ecx
    mov       ecx,dword ptr [esi+0x18]
    mov       dword ptr [esp+0x28],eax
    mov       eax,dword ptr [esi-0x1e8]
    xor       ecx,eax
    mov       dword ptr [esi+0x18],ecx
    mov       eax,ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xec]
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1e4]
    mov       dword ptr [esp+0x18],eax
    mov       eax,dword ptr [esi+0x1c]
    xor       eax,ecx
    mov       dword ptr [esi+0x1c],eax
    mov       ecx,eax
    mov       eax,dword ptr [esp+0xf0]
    xor       ecx,eax
    lea       eax,[esp+0xf4]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0xf8]
    push      eax
    push      ecx
    lea       eax,[esp+0x104]
    lea       ecx,[esp+0x108]
    push      eax
    push      ecx
    lea       eax,[esp+0x114]
    lea       ecx,[esp+0x118]
    push      eax
    push      ecx
    lea       eax,[esp+0x124]
    lea       ecx,[esp+0x128]
    push      eax
    mov       eax,dword ptr [esp+0x2c]
    push      ecx
    mov       ecx,dword ptr [esp+0x40]
    xor       eax,ecx
    mov       ecx,edi
    push      eax
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x60]
    mov       eax,dword ptr [esp+0x50]
    push      ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x6c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0xd4]
    lea       eax,[esp+0xd8]
    lea       ecx,[esp+0xdc]
    push      edx
    push      eax
    lea       edx,[esp+0xe8]
    push      ecx
    lea       eax,[esp+0xf0]
    push      edx
    lea       ecx,[esp+0xf8]
    push      eax
    lea       edx,[esp+0x100]
    push      ecx
    mov       ecx,dword ptr [esp+0x28]
    lea       eax,[esp+0x108]
    push      edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    push      ecx
    mov       ecx,dword ptr [esp+0x44]
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x58]
    push      edi
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x58]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x5c]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    push      ecx
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,ecx
    push      eax
    call      csc_transP
    mov       edx,dword ptr [esi-0x1a0]
    mov       eax,dword ptr [esi+0x60]
    mov       edi,dword ptr [esp+0x174]
    add       esi,0x00000040
    xor       eax,edx
    add       esp,0x00000040
    mov       edx,dword ptr [esi+0x24]
    mov       dword ptr [esi+0x20],eax
    xor       eax,edi
    mov       dword ptr [esp+0x20],eax
    mov       eax,dword ptr [esi-0x1dc]
    mov       ecx,dword ptr [esp+0x138]
    mov       edi,dword ptr [esi+0x28]
    xor       edx,eax
    mov       eax,edx
    mov       dword ptr [esi+0x24],edx
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1d8]
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x13c]
    mov       edx,eax
    mov       eax,edi
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x140]
    mov       dword ptr [esi+0x28],edi
    mov       edi,dword ptr [esi+0x2c]
    mov       dword ptr [esp+0x34],eax
    mov       eax,dword ptr [esi-0x1d4]
    xor       edi,eax
    mov       eax,edi
    mov       dword ptr [esi+0x2c],edi
    mov       edi,dword ptr [esi+0x30]
    xor       eax,ecx
    mov       ecx,eax
    mov       eax,dword ptr [esi-0x1d0]
    xor       edi,eax
    mov       dword ptr [esp+0x40],ecx
    mov       dword ptr [esi+0x30],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x144]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x34]
    mov       dword ptr [esp+0x30],eax
    mov       eax,dword ptr [esi-0x1cc]
    xor       edi,eax
    mov       dword ptr [esi+0x34],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x148]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x38]
    mov       dword ptr [esp+0x38],eax
    mov       eax,dword ptr [esi-0x1c8]
    xor       edi,eax
    mov       dword ptr [esi+0x38],edi
    mov       eax,edi
    mov       edi,dword ptr [esp+0x14c]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x3c]
    mov       dword ptr [esp+0x24],eax
    mov       eax,dword ptr [esi-0x1c4]
    xor       edi,eax
    mov       dword ptr [esi+0x3c],edi
    mov       eax,edi
    xor       eax,dword ptr [esp+0x150]
    mov       edi,eax
    mov       eax,dword ptr [esi-0x200]
    xor       dword ptr [esi],eax
    mov       eax,dword ptr [esi]
    mov       dword ptr [esp+0x44],edi
    xor       eax,edi
    mov       edi,dword ptr [esp+0x114]
    xor       eax,edi
    mov       edi,dword ptr [esi+0x4]
    mov       dword ptr [esp+0x10],eax
    mov       eax,dword ptr [esi-0x1fc]
    xor       edi,eax
    mov       eax,dword ptr [esp+0x118]
    mov       dword ptr [esi+0x4],edi
    xor       edi,eax
    mov       eax,dword ptr [esi-0x1f8]
    xor       dword ptr [esi+0x8],eax
    mov       eax,dword ptr [esi+0x8]
    xor       eax,edx
    xor       eax,dword ptr [esp+0x11c]
    mov       dword ptr [esp+0x3c],eax
    mov       eax,dword ptr [esi-0x1f4]
    xor       dword ptr [esi+0xc],eax
    mov       eax,dword ptr [esi+0xc]
    xor       eax,dword ptr [esp+0x120]
    mov       dword ptr [esp+0x2c],eax
    mov       eax,dword ptr [esi-0x1f0]
    xor       dword ptr [esi+0x10],eax
    mov       eax,dword ptr [esi+0x10]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x124]
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1ec]
    mov       dword ptr [esp+0x1c],eax
    mov       eax,dword ptr [esi+0x14]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x128]
    mov       dword ptr [esi+0x14],eax
    xor       eax,ecx
    mov       ecx,dword ptr [esi+0x18]
    mov       dword ptr [esp+0x28],eax
    mov       eax,dword ptr [esi-0x1e8]
    xor       ecx,eax
    mov       dword ptr [esi+0x18],ecx
    mov       eax,ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x12c]
    xor       eax,ecx
    mov       ecx,dword ptr [esi-0x1e4]
    mov       dword ptr [esp+0x18],eax
    mov       eax,dword ptr [esi+0x1c]
    xor       eax,ecx
    mov       dword ptr [esi+0x1c],eax
    mov       ecx,eax
    mov       eax,dword ptr [esp+0x130]
    xor       ecx,eax
    lea       eax,[esp+0x134]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0x138]
    push      eax
    push      ecx
    lea       eax,[esp+0x144]
    lea       ecx,[esp+0x148]
    push      eax
    push      ecx
    lea       eax,[esp+0x154]
    lea       ecx,[esp+0x158]
    push      eax
    push      ecx
    lea       eax,[esp+0x164]
    lea       ecx,[esp+0x168]
    push      eax
    mov       eax,dword ptr [esp+0x2c]
    push      ecx
    mov       ecx,dword ptr [esp+0x40]
    xor       eax,ecx
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x44]
    push      edx
    mov       edx,dword ptr [esp+0x6c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x114]
    lea       eax,[esp+0x118]
    push      edx
    lea       ecx,[esp+0x120]
    push      eax
    lea       edx,[esp+0x128]
    push      ecx
    lea       eax,[esp+0x130]
    push      edx
    lea       ecx,[esp+0x138]
    push      eax
    lea       edx,[esp+0x140]
    push      ecx
    mov       ecx,dword ptr [esp+0x28]
    lea       eax,[esp+0x148]
    push      edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    push      ecx
    mov       ecx,dword ptr [esp+0x44]
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0x3c]
    push      edi
    push      edx
    mov       edx,dword ptr [esp+0x60]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    push      eax
    mov       eax,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    xor       eax,edx
    push      ecx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0xd4]
    mov       edx,dword ptr [csc_tabe+0x20]
    mov       ecx,dword ptr [csc_tabe+0x24]
    mov       edi,dword ptr [csc_tabe+0x28]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x2c]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0xd8]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xdc]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x30]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0xe0]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xe4]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x34]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0xe8]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x38]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0xec]
    xor       edx,edi
    mov       edi,dword ptr [esp+0xf0]
    mov       dword ptr [esp+0x64],edx
    mov       edx,dword ptr [csc_tabe+0x3c]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x94]
    mov       dword ptr [esp+0x84],edi
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe]
    add       esp,0x00000040
    xor       edi,edx
    mov       edx,dword ptr [esp+0x58]
    add       esi,0x00000040
    xor       edx,dword ptr [csc_tabe+0x4]
    mov       dword ptr [esp+0x48],eax
    mov       dword ptr [esp+0x40],ecx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x60]
    xor       edx,dword ptr [csc_tabe+0xc]
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0x64]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x10]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x14]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0x68]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x18]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x1c]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0x70]
    xor       ecx,edx
    lea       edx,[esp+0x94]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0x98]
    push      edx
    push      ecx
    lea       edx,[esp+0xa4]
    lea       ecx,[esp+0xa8]
    push      edx
    push      ecx
    lea       edx,[esp+0xb4]
    lea       ecx,[esp+0xb8]
    push      edx
    push      ecx
    lea       edx,[esp+0xc4]
    lea       ecx,[esp+0xc8]
    push      edx
    push      ecx
    mov       ecx,dword ptr [esp+0x40]
    mov       edx,edi
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x30]
    push      edx
    mov       edx,dword ptr [esp+0x6c]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x50]
    push      ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x68]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x60]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x48]
    push      eax
    mov       eax,dword ptr [esp+0x6c]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x5c]
    xor       edx,ecx
    push      edx
    mov       edx,dword ptr [esp+0x80]
    xor       eax,edx
    push      eax
    call      csc_transP
    add       esp,0x00000040
    lea       ecx,[esp+0x54]
    lea       edx,[esp+0x58]
    lea       eax,[esp+0x5c]
    push      ecx
    push      edx
    push      eax
    lea       ecx,[esp+0x6c]
    lea       edx,[esp+0x70]
    push      ecx
    lea       eax,[esp+0x78]
    push      edx
    lea       ecx,[esp+0x80]
    push      eax
    mov       eax,dword ptr [esp+0x28]
    lea       edx,[esp+0x88]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x40]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      edi
    push      eax
    mov       eax,dword ptr [esp+0x64]
    mov       ecx,dword ptr [esp+0x44]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x5c]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x48]
    push      ecx
    push      eax
    mov       eax,dword ptr [esp+0x4c]
    push      edx
    xor       eax,dword ptr [esp+0x60]
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x154]
    mov       edx,dword ptr [csc_tabe+0x60]
    mov       ecx,dword ptr [csc_tabe+0x64]
    mov       edi,dword ptr [csc_tabe+0x68]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x6c]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0x158]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x15c]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x70]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0x160]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x164]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x74]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x168]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x78]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0x16c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x170]
    mov       dword ptr [esp+0x64],edx
    mov       edx,dword ptr [csc_tabe+0x7c]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x114]
    mov       dword ptr [esp+0x84],edi
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe+0x40]
    mov       dword ptr [esp+0x88],eax
    xor       edi,edx
    mov       edx,dword ptr [esp+0x118]
    xor       edx,dword ptr [csc_tabe+0x44]
    add       esp,0x00000040
    mov       dword ptr [esp+0x40],ecx
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0xdc]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x48]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xe0]
    xor       edx,dword ptr [csc_tabe+0x4c]
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0xe4]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x50]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x54]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0xe8]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xec]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x58]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x5c]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0xf0]
    xor       ecx,edx
    lea       edx,[esp+0x118]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0x114]
    push      ecx
    push      edx
    lea       ecx,[esp+0x124]
    lea       edx,[esp+0x128]
    push      ecx
    push      edx
    lea       ecx,[esp+0x134]
    lea       edx,[esp+0x138]
    push      ecx
    push      edx
    lea       ecx,[esp+0x144]
    lea       edx,[esp+0x148]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x40]
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0xd4]
    lea       eax,[esp+0xd8]
    lea       ecx,[esp+0xdc]
    push      edx
    push      eax
    lea       edx,[esp+0xe8]
    push      ecx
    lea       eax,[esp+0xf0]
    push      edx
    lea       ecx,[esp+0xf8]
    push      eax
    push      ecx
    lea       edx,[esp+0x104]
    lea       eax,[esp+0x108]
    push      edx
    mov       edx,dword ptr [esp+0x3c]
    mov       ecx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x30]
    push      edi
    xor       eax,edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x54]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x44]
    push      ecx
    push      eax
    mov       eax,dword ptr [esp+0x58]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0xf4]
    mov       edx,dword ptr [csc_tabe+0xa0]
    mov       ecx,dword ptr [csc_tabe+0xa4]
    mov       edi,dword ptr [csc_tabe+0xa8]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0xac]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0xf8]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xfc]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0xb0]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0x100]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x104]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0xb4]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x108]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0xb8]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0x10c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x110]
    mov       dword ptr [esp+0x64],edx
    mov       edx,dword ptr [csc_tabe+0xbc]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xb4]
    mov       dword ptr [esp+0x84],edi
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe+0x80]
    mov       dword ptr [esp+0x88],eax
    xor       edi,edx
    mov       edx,dword ptr [esp+0xb8]
    xor       edx,dword ptr [csc_tabe+0x84]
    mov       dword ptr [esp+0x80],ecx
    add       esp,0x00000040
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x7c]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x88]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x80]
    xor       edx,dword ptr [csc_tabe+0x8c]
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0x84]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x90]
    xor       ecx,edx
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0x88]
    mov       edx,dword ptr [csc_tabe+0x94]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x98]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x9c]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0x90]
    xor       ecx,edx
    lea       edx,[esp+0xb8]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0xb4]
    push      ecx
    push      edx
    lea       ecx,[esp+0xc4]
    lea       edx,[esp+0xc8]
    push      ecx
    push      edx
    lea       ecx,[esp+0xd4]
    lea       edx,[esp+0xd8]
    push      ecx
    push      edx
    lea       ecx,[esp+0xe4]
    lea       edx,[esp+0xe8]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x40]
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x74]
    lea       eax,[esp+0x78]
    lea       ecx,[esp+0x7c]
    push      edx
    push      eax
    lea       edx,[esp+0x88]
    push      ecx
    lea       eax,[esp+0x90]
    push      edx
    lea       ecx,[esp+0x98]
    push      eax
    lea       edx,[esp+0xa0]
    push      ecx
    lea       eax,[esp+0xa8]
    push      edx
    push      eax
    mov       eax,dword ptr [esp+0x30]
    push      edi
    mov       edx,dword ptr [esp+0x44]
    mov       ecx,dword ptr [esp+0x60]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x54]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x44]
    push      ecx
    push      eax
    mov       eax,dword ptr [esp+0x58]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x174]
    mov       edx,dword ptr [csc_tabe+0xe0]
    mov       ecx,dword ptr [csc_tabe+0xe4]
    mov       edi,dword ptr [csc_tabe+0xe8]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0xec]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0x178]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x17c]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0xf0]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0x180]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x184]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0xf4]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x188]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0xf8]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0x18c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x190]
    mov       dword ptr [esp+0x64],edx
    mov       edx,dword ptr [csc_tabe+0xfc]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x134]
    mov       dword ptr [esp+0x84],edi
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe+0xc0]
    mov       dword ptr [esp+0x88],eax
    xor       edi,edx
    mov       edx,dword ptr [esp+0x138]
    xor       edx,dword ptr [csc_tabe+0xc4]
    mov       dword ptr [esp+0x80],ecx
    add       esp,0x00000040
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0xfc]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0xc8]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x100]
    xor       edx,dword ptr [csc_tabe+0xcc]
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0x104]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0xd0]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0xd4]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0x108]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x10c]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0xd8]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0xdc]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0x110]
    xor       ecx,edx
    lea       edx,[esp+0x138]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0x134]
    push      ecx
    push      edx
    lea       ecx,[esp+0x144]
    lea       edx,[esp+0x148]
    push      ecx
    push      edx
    lea       ecx,[esp+0x154]
    lea       edx,[esp+0x158]
    push      ecx
    push      edx
    lea       ecx,[esp+0x164]
    lea       edx,[esp+0x168]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x40]
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0xf4]
    lea       eax,[esp+0xf8]
    lea       ecx,[esp+0xfc]
    push      edx
    push      eax
    lea       edx,[esp+0x108]
    push      ecx
    lea       eax,[esp+0x110]
    push      edx
    lea       ecx,[esp+0x118]
    push      eax
    lea       edx,[esp+0x120]
    push      ecx
    mov       ecx,dword ptr [esp+0x54]
    lea       eax,[esp+0x128]
    push      edx
    mov       edx,dword ptr [esp+0x3c]
    push      eax
    mov       eax,dword ptr [esp+0x30]
    xor       eax,edx
    push      edi
    push      eax
    mov       eax,dword ptr [esp+0x54]
    push      ecx
    mov       edx,dword ptr [esp+0x60]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x48]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x114]
    mov       edx,dword ptr [csc_tabe+0x120]
    mov       ecx,dword ptr [csc_tabe+0x124]
    mov       edi,dword ptr [csc_tabe+0x128]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x12c]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0x118]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x11c]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x130]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0x120]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x124]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x134]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x128]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x138]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0x12c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x130]
    mov       dword ptr [esp+0x64],edx
    mov       edx,dword ptr [csc_tabe+0x13c]
    xor       edi,edx
    mov       edx,dword ptr [esp+0x94]
    mov       dword ptr [esp+0x84],edi
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe+0x100]
    mov       dword ptr [esp+0x88],eax
    xor       edi,edx
    mov       edx,dword ptr [esp+0x98]
    xor       edx,dword ptr [csc_tabe+0x104]
    mov       dword ptr [esp+0x80],ecx
    add       esp,0x00000040
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x108]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x60]
    xor       edx,dword ptr [csc_tabe+0x10c]
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0x64]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x110]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x114]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0x68]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x118]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x11c]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0x70]
    xor       ecx,edx
    lea       edx,[esp+0xd8]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0xd4]
    push      ecx
    push      edx
    lea       ecx,[esp+0xe4]
    lea       edx,[esp+0xe8]
    push      ecx
    push      edx
    lea       ecx,[esp+0xf4]
    lea       edx,[esp+0xf8]
    push      ecx
    push      edx
    lea       ecx,[esp+0x104]
    lea       edx,[esp+0x108]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x40]
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x54]
    lea       eax,[esp+0x58]
    lea       ecx,[esp+0x5c]
    push      edx
    push      eax
    lea       edx,[esp+0x68]
    push      ecx
    lea       eax,[esp+0x70]
    push      edx
    lea       ecx,[esp+0x78]
    push      eax
    lea       edx,[esp+0x80]
    push      ecx
    lea       eax,[esp+0x88]
    push      edx
    mov       edx,dword ptr [esp+0x3c]
    mov       ecx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x30]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      edi
    push      eax
    mov       eax,dword ptr [esp+0x54]
    push      ecx
    xor       eax,edx
    mov       edx,dword ptr [esp+0x48]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x134]
    mov       edx,dword ptr [csc_tabe+0x160]
    mov       ecx,dword ptr [csc_tabe+0x164]
    mov       edi,dword ptr [csc_tabe+0x168]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x16c]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0x138]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x13c]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x170]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0x140]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x144]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x174]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x148]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x178]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0x14c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x150]
    mov       dword ptr [esp+0x64],edx
    mov       edx,dword ptr [csc_tabe+0x17c]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xb4]
    mov       dword ptr [esp+0x84],edi
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe+0x140]
    mov       dword ptr [esp+0x88],eax
    xor       edi,edx
    mov       edx,dword ptr [esp+0xb8]
    xor       edx,dword ptr [csc_tabe+0x144]
    mov       dword ptr [esp+0x80],ecx
    add       esp,0x00000040
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x7c]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x148]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x80]
    xor       edx,dword ptr [csc_tabe+0x14c]
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0x84]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x150]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x154]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0x88]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x8c]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x158]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x15c]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0x90]
    xor       ecx,edx
    lea       edx,[esp+0xf8]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0xf4]
    push      ecx
    push      edx
    lea       ecx,[esp+0x104]
    lea       edx,[esp+0x108]
    push      ecx
    push      edx
    lea       ecx,[esp+0x114]
    lea       edx,[esp+0x118]
    push      ecx
    push      edx
    lea       ecx,[esp+0x124]
    lea       edx,[esp+0x128]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x40]
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x74]
    lea       eax,[esp+0x78]
    lea       ecx,[esp+0x7c]
    push      edx
    push      eax
    lea       edx,[esp+0x88]
    push      ecx
    lea       eax,[esp+0x90]
    push      edx
    lea       ecx,[esp+0x98]
    push      eax
    lea       edx,[esp+0xa0]
    push      ecx
    lea       eax,[esp+0xa8]
    push      edx
    mov       edx,dword ptr [esp+0x3c]
    mov       ecx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x30]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      edi
    push      eax
    mov       eax,dword ptr [esp+0x54]
    push      ecx
    xor       eax,edx
    mov       edx,dword ptr [esp+0x48]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,dword ptr [esp+0x60]
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x154]
    mov       edx,dword ptr [csc_tabe+0x1a0]
    mov       ecx,dword ptr [csc_tabe+0x1a4]
    mov       edi,dword ptr [csc_tabe+0x1a8]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x1ac]
    mov       dword ptr [esp+0x60],eax
    mov       eax,dword ptr [esp+0x158]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x15c]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1b0]
    mov       dword ptr [esp+0x74],ecx
    mov       ecx,dword ptr [esp+0x160]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x164]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x1b4]
    mov       dword ptr [esp+0x70],edx
    mov       edx,dword ptr [esp+0x168]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x1b8]
    mov       dword ptr [esp+0x78],edx
    mov       edx,dword ptr [esp+0x16c]
    xor       edx,edi
    mov       edi,dword ptr [esp+0x170]
    mov       dword ptr [esp+0x64],edx
    mov       edx,dword ptr [csc_tabe+0x1bc]
    xor       edi,edx
    mov       edx,dword ptr [esp+0xd4]
    mov       dword ptr [esp+0x84],edi
    xor       edi,edx
    mov       edx,dword ptr [csc_tabe+0x180]
    mov       dword ptr [esp+0x88],eax
    xor       edi,edx
    mov       edx,dword ptr [esp+0xd8]
    xor       edx,dword ptr [csc_tabe+0x184]
    mov       dword ptr [esp+0x80],ecx
    add       esp,0x00000040
    mov       dword ptr [esp+0x10],edx
    mov       edx,dword ptr [esp+0x9c]
    xor       eax,edx
    mov       edx,dword ptr [csc_tabe+0x188]
    xor       eax,edx
    mov       edx,dword ptr [esp+0xa0]
    xor       edx,dword ptr [csc_tabe+0x18c]
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],edx
    mov       edx,dword ptr [esp+0xa4]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x190]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x194]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0xa8]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0xac]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x198]
    xor       ecx,edx
    mov       edx,dword ptr [csc_tabe+0x19c]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0xb0]
    xor       ecx,edx
    lea       edx,[esp+0x118]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0x114]
    push      ecx
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
    lea       edx,[esp+0x148]
    push      ecx
    push      edx
    mov       edx,dword ptr [esp+0x40]
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0x94]
    lea       eax,[esp+0x98]
    lea       ecx,[esp+0x9c]
    push      edx
    push      eax
    lea       edx,[esp+0xa8]
    push      ecx
    lea       eax,[esp+0xb0]
    push      edx
    lea       ecx,[esp+0xb8]
    push      eax
    lea       edx,[esp+0xc0]
    push      ecx
    lea       eax,[esp+0xc8]
    push      edx
    mov       edx,dword ptr [esp+0x3c]
    mov       ecx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x30]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      edi
    push      eax
    mov       eax,dword ptr [esp+0x54]
    push      ecx
    xor       eax,edx
    mov       edx,dword ptr [esp+0x48]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    push      edx
    mov       edx,dword ptr [esp+0x64]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       edx,dword ptr [esp+0x174]
    add       esp,0x00000040
    mov       edi,dword ptr [csc_tabe+0x1e0]
    mov       eax,dword ptr [esp+0x138]
    mov       ecx,dword ptr [csc_tabe+0x1e4]
    xor       edx,edi
    mov       edi,dword ptr [csc_tabe+0x1e8]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0x13c]
    mov       dword ptr [esp+0x48],eax
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1ec]
    mov       dword ptr [esp+0x34],ecx
    mov       ecx,dword ptr [esp+0x140]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1f0]
    mov       dword ptr [esp+0x40],ecx
    mov       ecx,dword ptr [esp+0x144]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1f4]
    mov       dword ptr [esp+0x30],ecx
    mov       ecx,dword ptr [esp+0x148]
    xor       ecx,edi
    mov       edi,dword ptr [csc_tabe+0x1f8]
    mov       dword ptr [esp+0x38],ecx
    mov       ecx,dword ptr [esp+0x14c]
    xor       ecx,edi
    mov       edi,dword ptr [esp+0x150]
    mov       dword ptr [esp+0x24],ecx
    mov       ecx,dword ptr [csc_tabe+0x1fc]
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0xb4]
    mov       dword ptr [esp+0x44],edi
    xor       edi,ecx
    mov       ecx,dword ptr [csc_tabe+0x1c0]
    mov       dword ptr [esp+0x20],edx
    xor       edi,ecx
    mov       ecx,dword ptr [esp+0xb8]
    xor       ecx,dword ptr [csc_tabe+0x1c4]
    mov       dword ptr [esp+0x10],ecx
    mov       ecx,dword ptr [esp+0xbc]
    xor       eax,ecx
    mov       ecx,dword ptr [csc_tabe+0x1c8]
    xor       eax,ecx
    mov       ecx,dword ptr [esp+0xc0]
    xor       ecx,dword ptr [csc_tabe+0x1cc]
    mov       dword ptr [esp+0x3c],eax
    mov       dword ptr [esp+0x2c],ecx
    mov       ecx,dword ptr [esp+0x40]
    xor       ecx,dword ptr [esp+0xc4]
    xor       ecx,dword ptr [csc_tabe+0x1d0]
    mov       dword ptr [esp+0x1c],ecx
    mov       ecx,dword ptr [esp+0xc8]
    xor       ecx,dword ptr [csc_tabe+0x1d4]
    mov       dword ptr [esp+0x28],ecx
    mov       ecx,dword ptr [esp+0x38]
    xor       ecx,dword ptr [esp+0xcc]
    xor       ecx,dword ptr [csc_tabe+0x1d8]
    mov       dword ptr [esp+0x18],ecx
    mov       ecx,dword ptr [esp+0xd0]
    xor       ecx,dword ptr [csc_tabe+0x1dc]
    mov       dword ptr [esp+0x14],ecx
    lea       ecx,[esp+0x134]
    push      ecx
    lea       ecx,[esp+0x13c]
    push      ecx
    lea       ecx,[esp+0x144]
    push      ecx
    lea       ecx,[esp+0x14c]
    push      ecx
    lea       ecx,[esp+0x154]
    push      ecx
    lea       ecx,[esp+0x15c]
    push      ecx
    lea       ecx,[esp+0x164]
    push      ecx
    lea       ecx,[esp+0x16c]
    push      ecx
    mov       ecx,edi
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x30]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x40]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x68]
    push      eax
    mov       eax,dword ptr [esp+0x58]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x60]
    xor       ecx,eax
    mov       eax,dword ptr [esp+0x48]
    push      ecx
    mov       ecx,dword ptr [esp+0x6c]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x48]
    push      edx
    mov       edx,dword ptr [esp+0x5c]
    xor       eax,edx
    push      eax
    mov       eax,dword ptr [esp+0x80]
    xor       ecx,eax
    push      ecx
    call      csc_transP
    add       esp,0x00000040
    lea       edx,[esp+0xb4]
    lea       eax,[esp+0xb8]
    lea       ecx,[esp+0xbc]
    push      edx
    push      eax
    lea       edx,[esp+0xc8]
    push      ecx
    lea       eax,[esp+0xd0]
    push      edx
    lea       ecx,[esp+0xd8]
    push      eax
    lea       edx,[esp+0xe0]
    push      ecx
    lea       eax,[esp+0xe8]
    push      edx
    mov       edx,dword ptr [esp+0x3c]
    mov       ecx,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x30]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x54]
    push      edi
    mov       edi,dword ptr [esp+0x54]
    push      eax
    mov       eax,dword ptr [esp+0x54]
    xor       eax,edx
    mov       edx,dword ptr [esp+0x44]
    push      ecx
    push      eax
    mov       eax,dword ptr [esp+0x58]
    push      edx
    xor       eax,edi
    mov       edi,dword ptr [esp+0x58]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    push      eax
    mov       eax,dword ptr [esp+0x50]
    xor       eax,edi
    push      eax
    call      csc_transP
    mov       eax,dword ptr [esp+0x194]
    add       esp,0x00000040
    dec       eax
    mov       dword ptr [esp+0x154],eax
    ljne       X$10
    lea       ecx,[esp+0xd4]
    lea       eax,[esi+0xc0]
    mov       dword ptr [esp+0x48],ecx
    mov       ecx,dword ptr [esp+0x17c]
    mov       dword ptr [esp+0x10],0xffffffff
    mov       dword ptr [esp+0x3c],0x00000000
    lea       edx,[ecx+0xc0]
    mov       dword ptr [esp+0x44],eax
    mov       dword ptr [esp+0x1c],edx
    lea       edx,[esp+0x74]
    sub       edx,ecx
    mov       dword ptr [esp+0x20],edx
    lea       edx,[esp+0x54]
    sub       edx,ecx
    mov       dword ptr [esp+0x28],edx
X$13:
    cmp       dword ptr [esp+0x3c],0x00000008
    ljge       X$27
    lea       edi,[eax-0x60]
    lea       ecx,[eax-0xa0]
    mov       dword ptr [esp+0x18],edi
    lea       edi,[eax-0x40]
    mov       dword ptr [esp+0x40],edi
    push      esi
    mov       dword ptr [esp+0x28],ecx
    lea       edx,[eax-0x80]
    lea       edi,[eax-0x20]
    push      ecx
    mov       ecx,dword ptr [esp+0x20]
    mov       dword ptr [esp+0x34],edx
    mov       dword ptr [esp+0x15c],edi
    push      edx
    mov       edx,dword ptr [esp+0x4c]
    push      ecx
    mov       ecx,dword ptr [esp+0x164]
    push      edx
    lea       edi,[eax+0x20]
    push      ecx
    push      eax
    mov       eax,dword ptr [esp+0x68]
    mov       dword ptr [esp+0x30],edi
    push      edi
    mov       edi,dword ptr [esp+0x70]
    mov       ecx,dword ptr [eax]
    mov       edx,dword ptr [edi]
    xor       edx,ecx
    mov       ecx,dword ptr [eax+0x4]
    push      edx
    mov       edx,dword ptr [edi+0x4]
    xor       ecx,edx
    mov       edx,dword ptr [eax+0x8]
    push      ecx
    mov       ecx,dword ptr [edi+0x8]
    xor       edx,ecx
    mov       ecx,dword ptr [eax+0xc]
    push      edx
    mov       edx,dword ptr [edi+0xc]
    xor       ecx,edx
    mov       edx,dword ptr [eax+0x10]
    push      ecx
    mov       ecx,dword ptr [edi+0x10]
    xor       edx,ecx
    mov       ecx,dword ptr [eax+0x14]
    push      edx
    mov       edx,dword ptr [edi+0x14]
    xor       ecx,edx
    mov       edx,dword ptr [eax+0x18]
    mov       eax,dword ptr [eax+0x1c]
    push      ecx
    xor       edx,dword ptr [edi+0x18]
    push      edx
    mov       edx,dword ptr [edi+0x1c]
    xor       eax,edx
    push      eax
    call      csc_transP
    mov       ecx,dword ptr [esp+0x60]
    mov       eax,dword ptr [esp+0x5c]
    add       esp,0x00000040
    mov       edx,dword ptr [ecx+eax]
    mov       ecx,dword ptr [eax+0x20]
    xor       edx,ecx
    mov       ecx,dword ptr [esi-0x120]
    xor       edx,ecx
    mov       ecx,dword ptr [esp+0x14]
    xor       edx,dword ptr [ecx]
    mov       ecx,dword ptr [esp+0x10]
    not       edx
    and       ecx,edx
    mov       dword ptr [esp+0x10],ecx
    lje        X$14
    mov       edx,dword ptr [esp+0x28]
    mov       ecx,dword ptr [edx+eax]
    mov       edx,dword ptr [esi-0x140]
    xor       ecx,edx
    mov       edx,dword ptr [eax]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x44]
    xor       ecx,dword ptr [edx]
    mov       edx,dword ptr [esp+0x10]
    not       ecx
    and       edx,ecx
    mov       dword ptr [esp+0x10],edx
    lje        X$14
    mov       edx,dword ptr [esp+0x48]
    mov       ecx,dword ptr [eax-0x20]
    xor       ecx,dword ptr [edx+0x20]
    mov       edx,dword ptr [esi-0x160]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x154]
    xor       ecx,dword ptr [edx]
    mov       edx,dword ptr [esp+0x10]
    not       ecx
    and       edx,ecx
    mov       dword ptr [esp+0x10],edx
    lje        X$14
    mov       ecx,dword ptr [eax-0x40]
    mov       edx,dword ptr [esi-0x180]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x40]
    xor       ecx,dword ptr [edx]
    mov       edx,dword ptr [esp+0x48]
    xor       ecx,dword ptr [edx]
    mov       edx,dword ptr [esp+0x10]
    not       ecx
    and       edx,ecx
    mov       dword ptr [esp+0x10],edx
    lje        X$14
    mov       edx,dword ptr [esp+0x48]
    mov       ecx,dword ptr [eax-0x60]
    xor       ecx,dword ptr [edx-0x20]
    mov       edx,dword ptr [esi-0x1a0]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x18]
    xor       ecx,dword ptr [edx]
    mov       edx,dword ptr [esp+0x10]
    not       ecx
    and       edx,ecx
    mov       dword ptr [esp+0x10],edx
    lje        X$14
    mov       edx,dword ptr [esp+0x48]
    mov       ecx,dword ptr [eax-0x80]
    xor       ecx,dword ptr [edx-0x40]
    mov       edx,dword ptr [esi-0x1c0]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x2c]
    xor       ecx,dword ptr [edx]
    mov       edx,dword ptr [esp+0x10]
    not       ecx
    and       edx,ecx
    mov       dword ptr [esp+0x10],edx
    lje        X$14
    mov       edx,dword ptr [esp+0x48]
    mov       ecx,dword ptr [eax-0xa0]
    xor       ecx,dword ptr [edx-0x60]
    mov       edx,dword ptr [esi-0x1e0]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x24]
    xor       ecx,dword ptr [edx]
    mov       edx,dword ptr [esp+0x10]
    not       ecx
    and       edx,ecx
    mov       dword ptr [esp+0x10],edx
    je        X$14
    mov       edx,dword ptr [esp+0x48]
    mov       ecx,dword ptr [eax-0xc0]
    xor       ecx,dword ptr [edx-0x80]
    mov       edx,dword ptr [esi-0x200]
    xor       ecx,edx
    mov       edx,dword ptr [esi]
    xor       ecx,edx
    mov       edx,dword ptr [esp+0x10]
    not       ecx
    and       edx,ecx
    mov       dword ptr [esp+0x10],edx
    je        X$14
    mov       ecx,dword ptr [esp+0x3c]
    mov       edx,dword ptr [esp+0x48]
    inc       ecx
    add       edi,0x00000020
    mov       dword ptr [esp+0x3c],ecx
    mov       ecx,0x00000004
    add       eax,ecx
    mov       dword ptr [esp+0x50],edi
    mov       edi,dword ptr [esp+0x4c]
    mov       dword ptr [esp+0x1c],eax
    mov       eax,dword ptr [esp+0x44]
    add       edx,ecx
    add       edi,0x00000020
    add       esi,ecx
    add       eax,ecx
    mov       dword ptr [esp+0x48],edx
    mov       dword ptr [esp+0x4c],edi
    mov       dword ptr [esp+0x44],eax
    jmp       ptr X$13
X$14:
    mov       eax,dword ptr [esp+0x15c]
    inc       eax
    test      al,0x01
    mov       dword ptr [esp+0x15c],eax
    je        X$16
    mov       ecx,dword ptr [esp+0x180]
    mov       eax,dword ptr [ecx]
    test      eax,eax
    lje        X$9
X$15:
    mov       edx,dword ptr [eax]
    add       ecx,0x00000004
    not       edx
    mov       dword ptr [eax],edx
    mov       eax,dword ptr [ecx]
    test      eax,eax
    jne       X$15
    jmp       ptr X$9
X$16:
    test      al,0x02
    je        X$18
    mov       eax,dword ptr [esp+0x180]
    lea       ecx,[eax+0x74]
    mov       eax,dword ptr [eax+0x74]
    test      eax,eax
    lje        X$9
X$17:
    mov       edx,dword ptr [eax]
    add       ecx,0x00000004
    not       edx
    mov       dword ptr [eax],edx
    mov       eax,dword ptr [ecx]
    test      eax,eax
    jne       X$17
    jmp       ptr X$9
X$18:
    test      al,0x04
    je        X$20
    mov       ecx,dword ptr [esp+0x168]
    mov       eax,dword ptr [ecx]
    test      eax,eax
    lje        X$9
X$19:
    mov       edx,dword ptr [eax]
    add       ecx,0x00000004
    not       edx
    mov       dword ptr [eax],edx
    mov       eax,dword ptr [ecx]
    test      eax,eax
    jne       X$19
    jmp       ptr X$9
X$20:
    test      al,0x08
    je        X$22
    mov       eax,dword ptr [esp+0x180]
    lea       ecx,[eax+0x15c]
    mov       eax,dword ptr [eax+0x15c]
    test      eax,eax
    lje        X$9
X$21:
    mov       edx,dword ptr [eax]
    add       ecx,0x00000004
    not       edx
    mov       dword ptr [eax],edx
    mov       eax,dword ptr [ecx]
    test      eax,eax
    jne       X$21
    jmp       ptr X$9
X$22:
    test      al,0x10
    je        X$24
    mov       eax,dword ptr [esp+0x180]
    lea       ecx,[eax+0x1d0]
    mov       eax,dword ptr [eax+0x1d0]
    test      eax,eax
    lje        X$9
X$23:
    mov       edx,dword ptr [eax]
    add       ecx,0x00000004
    not       edx
    mov       dword ptr [eax],edx
    mov       eax,dword ptr [ecx]
    test      eax,eax
    jne       X$23
    jmp       ptr X$9
X$24:
    test      al,0x20
    je        X$26
    mov       eax,dword ptr [esp+0x180]
    lea       ecx,[eax+0x244]
    mov       eax,dword ptr [eax+0x244]
    test      eax,eax
    lje        X$9
X$25:
    mov       edx,dword ptr [eax]
    add       ecx,0x00000004
    not       edx
    mov       dword ptr [eax],edx
    mov       eax,dword ptr [ecx]
    test      eax,eax
    jne       X$25
    jmp       ptr X$9
X$26:
    xor       eax,eax
    pop       edi
    pop       esi
    pop       ebp
    pop       ebx
    add       esp,0x0000015c
    ret       
X$27:
    mov       esi,dword ptr [esp+0x164]
    mov       edi,dword ptr [esp+0x170]
    mov       eax,dword ptr [esp+0x10]
    mov       ecx,0x00000040
    repe movsd 
    pop       edi
    pop       esi
    pop       ebp
    pop       ebx
    add       esp,0x0000015c
    ret       

__CODESECT__
    align 32
csc_unit_func_6b:
_csc_unit_func_6b:
    mov       eax,dword ptr [esp+0xc]
    sub       esp,0x00000028
    test      al,0x0f
    push      ebx
    push      ebp
    push      esi
    push      edi
    je        X$28
    add       eax,0x0000000f
    and       al,0xf0
X$28:
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
X$29:
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
    jne       X$30
    mov       eax,dword ptr [esp+0x3c]
    mov       edx,dword ptr [eax]
    mov       edi,dword ptr [eax+0x8]
    mov       eax,dword ptr [esp+0x10]
    mov       dword ptr [esp+0x14],eax
    mov       eax,0x00000001
X$30:
    mov       ebx,dword ptr [esp+0x18]
    add       ecx,0x00000004
    dec       ebx
    mov       dword ptr [esp+0x18],ebx
    jne       X$29
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
    jbe       X$32
X$31:
    inc       ebp
    mov       edx,0x00000001
    mov       ecx,ebp
    shl       edx,cl
    cmp       eax,edx
    ja        X$31
    mov       dword ptr [esp+0x18],ebp
X$32:
    mov       ebx,offset csc_bit_order
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
    mov       al,0x01
    sub       ecx,edx
    mov       dl,byte ptr [edi]
    shl       al,cl
    not       al
    and       dl,al
    cmp       ebx,offset csc_bit_order+0x18
    mov       byte ptr [edi],dl
    jl        X$33
    add       ebp,0xfffffff5
    test      ebp,ebp
    mov       dword ptr [esp+0x24],ebp
    jbe       X$35
    mov       ebx,offset csc_bit_order+0x2c
X$34:
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
    jne       X$34
X$35:
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
    call      cscipher_bitslicer_6b
    add       esp,0x00000014
    test      eax,eax
    jne       X$40
    mov       ecx,dword ptr [esp+0x24]
    mov       edx,0x00000001
    shl       edx,cl
    mov       dword ptr [esp+0x2c],edx
    jmp       X$37
X$36:
    mov       edx,dword ptr [esp+0x2c]
X$37:
    inc       ebx
    cmp       ebx,edx
    jae       X$40
    xor       ecx,ecx
    test      bl,0x01
    jne       X$39
X$38:
    inc       ecx
    mov       edx,0x00000001
    shl       edx,cl
    test      ebx,edx
    je        X$38
X$39:
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
    call      cscipher_bitslicer_6b
    add       esp,0x00000014
    test      eax,eax
    je        X$36
X$40:
    xor       ecx,ecx
    cmp       eax,ecx
    lje        X$47
    xor       edx,edx
    cmp       eax,0x00000001
    je        X$42
X$41:
    shr       eax,0x00000001
    inc       edx
    cmp       eax,0x00000001
    jne       X$41
X$42:
    mov       dword ptr [esp+0x10],ecx
    mov       dword ptr [esp+0x44],ecx
    mov       eax,0x00000008
    add       esi,0x00000020
X$43:
    mov       edi,dword ptr [esi]
    cmp       eax,0x00000020
    mov       ecx,edx
    jge       X$44
    shr       edi,cl
    mov       ecx,eax
    and       edi,0x00000001
    shl       edi,cl
    or        dword ptr [esp+0x44],edi
    jmp       X$45
X$44:
    shr       edi,cl
    lea       ecx,[eax-0x20]
    and       edi,0x00000001
    shl       edi,cl
    or        dword ptr [esp+0x10],edi
X$45:
    inc       eax
    add       esi,0x00000004
    cmp       eax,0x00000040
    jl        X$43
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
    jae       X$46
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
X$46:
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
X$47:
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
    jae       X$48
    inc       dword ptr [ecx+0x10]
X$48:
    pop       edi
    pop       esi
    pop       ebp
    mov       eax,0x00000001
    pop       ebx
    add       esp,0x00000028
    ret       

__CODESECT__
    align 32

