;
; Copyright distributed.net 1998-2004 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; x86 Processor identification for rc5 distributed.net effort
; Written in a dark and stormy night (Jan 16, 1998) by
; Cyrus Patel <cyp@fb14.uni-mainz.de>
;
; $Id: x86ident.asm,v 1.1.2.3 2004/06/27 20:46:23 jlawson Exp $
;
; correctly identifies almost every 386+ processor with the
; following exceptions:
; a) The code to use Cyrix Device ID registers for differentiate between a 
;    Cyrix 486, 6x86, etc is in place but disabled. Even so, it would not
;    detect correctly if CPUID is enabled _and_ reassigned.
; b) It may not correctly identify a Rise CPU if cpuid is disabled/reassigned
; c) It identifies an Intel 486 as an AMD 486 (chips are identical) and not
;    vice-versa because the Intel:0400 CPUID is actually a real CPUID value.
; d) It _does_ id the Transmeta Crusoe (its Family:5,Model:4) :)
;
; x86ident:  return u32
;            hiword = manufacturer ID (also for 386/486)
;                     (BX from cpuid for simplicity)
;                    [bxBXdxDXcxCX]
;                     GenuineIntel = 'eG' = 0x6547
;                     GenuineTMx86 = 'MT' = 0x4D54 <-- NOTE!
;                     AuthenticAMD = 'uA' = 0x7541
;                     CyrixInstead = 'yC' = 0x7943
;                     NexGenDriven = 'eN' = 0x654E
;                     CentaurHauls = 'eC' = 0x6543
;                     UMC UMC UMC  = 'MU' = 0x4D55
;                     RiseRiseRise = 'iR' = 0x6952
;            loword = bits 12..15 for identification extensions
;                     bits 11...0 for family/model/stepping per CPUID
;                                 or 0x0300 for 386, 0x0400 for 486
;
; 0123456789|ABCDEFGHIJKLMNOPQRSTUVWXYZ|abcdefghijklmnopqrstuvwxyz|
; 3        3|4              5          |6              7          |
; 0123456789|123456789ABCDEF0123456789A|123456789ABCDEF0123456789A|
BITS 64

%define __DATASECT__ [SECTION .data]
%define __CODESECT__ [SECTION .text]

global          x86ident,_x86ident
global          x86ident_haveioperm, _x86ident_haveioperm

__DATASECT__
__savident      dd 0              
_x86ident_haveioperm:             ; do we have permission to do in/out?
x86ident_haveioperm dd 0          ; we do on win9x (not NT), win16, dos

__CODESECT__
_x86ident:
x86ident:       mov     eax,[__savident]
                or      eax, eax
                jz      _ge386
                ret

_ge386:
_ge586:         push    rbx             ; cpuid trashes ebx
                xor     eax, eax        ; cpuid function zero
                cpuid                   ; => eax=maxlevels,ebx:ecx:edx=vendor
                push    rbx             ; save maker code
                mov     eax, 1          ; family/model/stepping/features
                cpuid                   ; => ax=type/family/model/stepping
                and     ax,0fffh        ; drop the type flags
                mov     cx,ax           ; save family/model/stepping bits
                pop     rax             ; restore maker code
		shl	eax,16
                mov     ax,cx           ; add family/model/stepping bits
		pop	rbx

_end:           mov     [__savident],eax; save it for next time
                ret

;----------------------------------------------------------------------

                endp
                end
