; Copyright distributed.net 1997 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; Callable/inline versions of csc_trans[F|G|P].
;
; @(#)$Id: csc-tran.inc,v 1.3 2000/06/02 06:32:57 jlawson Exp $
;

%macro        _called_csc_transF 0
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
%endmacro

%macro        _inline_csc_transF 0
    push      eax ;sub esp,4
    _called_csc_transF
    pop       eax ;add esp,4
%endmacro       

%macro        _called_csc_transG 0
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
%endmacro       

%macro        _inline_csc_transG 0
    push      eax ;sub esp,4
    _called_csc_transG
    pop       eax ;add esp,4
%endmacro    

%macro        _called_csc_transP 0
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
%endmacro       

%macro        _inline_csc_transP 0
    push      eax ;sub esp,4
    _called_csc_transP
    pop       eax ;add esp,4
%endmacro    

