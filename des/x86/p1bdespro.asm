; p1BDESPRO.ASM    Pentium Pro

; $Log: p1bdespro.asm,v $
; Revision 1.1.2.1  2001/01/21 16:57:16  cyp
; reorg
;
; Revision 1.4  1998/12/25 03:23:43  cyp
; Bryd now supports upto 4 processors. The third and fourth processor will
; use the two (otherwise idle) cores, ie pro on a p5 machine and vice versa.
; For non-mt builds the second cores (bbryd_des and bbryd_des_pro) are
; aliased to the first two cores (bryd_des and bryd_des_pro) which should
; make #ifdef checks around p2des_unit_func_[p5|pro]() obsolete.
;
; Revision 1.3  1998/06/18 05:46:52  remi
; Added $Id: p1bdespro.asm,v 1.1.2.1 2001/01/21 16:57:16 cyp Exp $. and $Log: p1bdespro.asm,v $
; Added $Id: p1bdespro.asm,v 1.4 1998/12/25 03:23:43 cyp Exp $. and Revision 1.1.2.1  2001/01/21 16:57:16  cyp
; Added $Id: p1bdespro.asm,v 1.4 1998/12/25 03:23:43 cyp Exp $. and reorg
; Added $Id: p1bdespro.asm,v 1.4 1998/12/25 03:23:43 cyp Exp $. and
; Added $Id: p1bdespro.asm,v 1.1.2.1 2001/01/21 16:57:16 cyp Exp $. and Revision 1.4  1998/12/25 03:23:43  cyp
; Added $Id: p1bdespro.asm,v 1.1.2.1 2001/01/21 16:57:16 cyp Exp $. and Bryd now supports upto 4 processors. The third and fourth processor will
; Added $Id: p1bdespro.asm,v 1.1.2.1 2001/01/21 16:57:16 cyp Exp $. and use the two (otherwise idle) cores, ie pro on a p5 machine and vice versa.
; Added $Id: p1bdespro.asm,v 1.1.2.1 2001/01/21 16:57:16 cyp Exp $. and For non-mt builds the second cores (bbryd_des and bbryd_des_pro) are
; Added $Id: p1bdespro.asm,v 1.1.2.1 2001/01/21 16:57:16 cyp Exp $. and aliased to the first two cores (bryd_des and bryd_des_pro) which should
; Added $Id: p1bdespro.asm,v 1.1.2.1 2001/01/21 16:57:16 cyp Exp $. and make #ifdef checks around p2des_unit_func_[p5|pro]() obsolete.
; Added $Id: p1bdespro.asm,v 1.1.2.1 2001/01/21 16:57:16 cyp Exp $. and.
;

; mov reg8,... changed to movzx reg32,...
; Marked Core 2 PRO b

; Supplement to BrydDES Key Search Library version 1.01.
; Date: January 29, 1998.
; Copyright Svend Olaf Mikkelsen, 1995, 1997, 1998.


; Please read README.TXT.

; Written for Microsoft Macro Assembler, version 6.00B.
; Modified, so it also seems to assemble correct with TASM 5.0.
; MASM @@ labels are changed to the form _01:
; TASM Parity? seems buggy, so Parity? not used.

        .386
        .model flat,c

	.data
_id	byte "@(#)$Id: p1bdespro.asm,v 1.1.2.1 2001/01/21 16:57:16 cyp Exp $"
	byte 0
align 4

bswapmacro macro registername
        push eax
        mov eax,registername
        ror ax,8
        ror eax,16
        ror ax,8
        push eax
        mov eax,[esp+4]
        pop registername
        add esp,4
        endm


extern                  bryd_continue_pro   :near
extern                  bryd_key_found_pro  :near

public                  p1bryd_des
;public                  p1desencrypt
;public                  p1desdecrypt
;public                  p1desinit
;public                  p1key_byte_to_hex
;public                  p1c_key_byte_to_hex

;        include bdesmcp.inc      ; macros
; BDESMCP.INC   Macros  Pentium Pro version

; mov reg8,... changed to movzx reg32,...
; Marked Core 2 PRO b

; Suplement to BrydDES Key Search Library version 1.01.
; Date: January 29, 1998.
; Copyright Svend Olaf Mikkelsen, 1995, 1997, 1998.


; Please read README.TXT.

; Written for Microsoft Macro Assembler, version 6.00B.
; Modified, so it also seems to assemble correct with TASM 5.0.



changetwokeydword macro dwordno1,value1,dwordno2,value2

        mov eax,dword ptr keysetup [(dwordno1-1)*4]
        mov edx,dword ptr keysetup [(dwordno2-1)*4]
        xor eax,value1
        xor edx,value2
        mov dword ptr keysetup [(dwordno1-1)*4],eax
        mov dword ptr keysetup [(dwordno2-1)*4],edx

        endm

changethreekeydword macro dwordno1,value1,dwordno2,value2,dwordno3,value3

        mov eax,dword ptr keysetup [(dwordno1-1)*4]
        mov ebp,dword ptr keysetup [(dwordno2-1)*4]
        xor eax,value1
        mov edx,dword ptr keysetup [(dwordno3-1)*4]
        xor ebp,value2
        mov dword ptr keysetup [(dwordno1-1)*4],eax
        xor edx,value3
        mov dword ptr keysetup [(dwordno2-1)*4],ebp
        mov dword ptr keysetup [(dwordno3-1)*4],edx

        endm


round15box3part1 macro
        mov esi,dword ptr cipherpermu
        mov edi,dword ptr cipherpermu [4]
        mov eax,dword ptr keysetup [(16-1)*8]   ; round 16
        xor eax,edi
        mov edx,dword ptr keysetup [(16-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        ;;;mov bl,al
        movzx ebx,al
        rol edx,4
        ;;;mov cl,ah
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        ;;;mov bl,dl
        movzx ebx,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor esi,ebp
        ;;;mov cl,dh
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        xor ecx,20h
        xor esi,ebp
        ;;;mov bl,ah
        movzx  ebx,ah
        mov edi,dword ptr sbox7 [ecx]
        and eax,0FFh
        xor ebp,edi
        mov redo16box7,ebp
        ;;;mov cl,dh
        movzx ecx,dh
        mov edi,dword ptr sbox3 [ecx]
        xor esi,edi
        and edx,0FFh
        mov edi,dword ptr sbox2 [ebx]
        xor ebx,04h
        xor esi,edi
        mov ecx,dword ptr sbox4 [eax]
        xor esi,ecx
        mov ebx,dword ptr sbox2 [ebx]
        mov ecx,dword ptr sbox5 [edx]
        xor ebx,edi
        xor esi,ecx
        mov redo16box2,ebx
        xor ebx,ebx
        xor ecx,ecx
        mov saveesiround16,esi

        ; Complement key part:

        mov esi,dword ptr cipherpermucomp
        mov edi,dword ptr cipherpermucomp [4]

        mov eax,dword ptr keysetup [(16-1)*8]
        xor eax,edi
        mov edx,dword ptr keysetup [(16-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        ;;;mov bl,al
        movzx ebx,al
        rol edx,4
        ;;;mov cl,ah
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        ;;;mov bl,dl
        movzx ebx,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor esi,ebp
        ;;;mov cl,dh
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        xor ecx,20h
        xor esi,ebp
        ;;;mov bl,ah
        movzx ebx,ah
        and eax,0FFh
        mov edi,dword ptr sbox7 [ecx]
        xor ecx,ecx
        xor ebp,edi
        ;;;mov cl,dh
        movzx ecx,dh
        mov edi,dword ptr sbox3[ecx]
        xor esi,edi
        and edx,0FFh
        mov edi,dword ptr sbox2 [ebx]
        xor ebx,04h
        xor esi,edi
        mov ecx,dword ptr sbox4 [eax]
        xor esi,ecx
        mov ebx,dword ptr sbox2 [ebx]
        mov ecx,dword ptr sbox5 [edx]
        xor ebx,edi
        xor esi,ecx
        mov redo16box2comp,ebx
        mov redo16box7comp,ebp
        mov saveesiround16comp,esi
        xor ebx,ebx
        xor ecx,ecx

        endm


round15box3part2 macro
        ; lines marked ;2 belongs to the not complement key part

        mov ebx,redo16box2comp
        mov ebp,redo16box7comp
        mov esi,saveesiround16comp

        ; then we make four versions of box 3 in round 15

        and ebx,00FC00000h   ;  redo16box2comp

        shr ebx,20
        and ebp,00FC00000h   ; redo16box7comp

        shr ebp,20
        mov edx,dword ptr keysetup [(15-1)*8+4]

        xor edx,esi
        and esi,08200401h

        and edx,00FC00000h
        mov round15box5checkcomp,esi

        shr edx,20
        mov ecx,redo16box2                   ; 2

        mov edi,dword ptr cipherpermucompbox3bits;  [4]
        and ecx,00FC00000h                   ; 2

        mov eax,dword ptr sbox3 [edx]
        xor edx,ebx

        shr ecx,20
        xor eax,edi

        mov esi,dword ptr sbox3 [edx]
        xor edx,ebp

        xor esi,edi
        mov round13box3check00comp,eax

        mov eax,dword ptr sbox3 [edx]
        xor edx,ebx

        mov round13box3check01comp,esi
        xor eax,edi

        mov ebx,dword ptr sbox3 [edx]
        mov edx,dword ptr keysetup [(15-1)*8+4]

        mov esi,saveesiround16  ; 2
        xor ebx,edi

        xor edx,esi             ; 2
        mov round13box3check11comp,eax

        and edx,00FC00000h      ;2
        mov round13box3check10comp,ebx


        ; not complement key:

        shr edx,20
        mov eax,esi

        and eax,08200401h
        mov ebp,edx

        xor ebp,ecx
        mov edi,redo16box7

        mov round15box5check,eax
        and edi,00FC00000h

        shr edi,20
        mov eax,dword ptr cipherpermubox3bits

        mov ebx,dword ptr sbox3 [ebp]   ;01
        xor ebp,edi

        xor ebx,eax
        mov esi,dword ptr sbox3 [edx]   ;00

        mov round13box3check01,ebx
        xor esi,eax

        mov ebx,dword ptr sbox3 [ebp]   ;11
        xor ebp,ecx

        xor ebx,eax
        xor ecx,ecx

        mov round13box3check11,ebx
        mov edx,dword ptr sbox3 [ebp]   ;10

        mov round13box3check00,esi
        xor edx,eax

        mov round13box3check10,edx
        xor ebx,ebx

        endm


change46new   macro
;changetwokeydword  4,00100000h, 7,00000800h
changethreekeydword 7,00000800h,10,00020000h,11,00000008h
changetwokeydword 14,00004000h,16,00000040h
changetwokeydword 18,00040000h,19,00000020h
changetwokeydword 21,00000400h,23,00000080h
;changetwokeydword 26,00000800h,27,00000040h
;changetwokeydword 29,00004000h,32,00000200h
endm

change46rest   macro
        mov eax,is14b
        mov edx,dword ptr keysetup[24]
        and edx,00000800h
        .if eax != edx
            xor eax,00000800h
            mov is14b,eax
            changetwokeydword 26,00000800h,27,00000040h
            changetwokeydword 29,00004000h,32,00000200h
        .endif
endm


change50new   macro
;changetwokeydword  4,00400000h, 5,10000000h
changethreekeydword  5,10000000h,8,01000000h, 9,00040000h
changetwokeydword 11,40000000h,14,00800000h
changetwokeydword 16,00000001h,18,08000000h
changetwokeydword 19,08000000h,22,04000000h
changetwokeydword 24,00000004h,25,20000000h  ; round 13 box 2
;changethreekeydword 27,00080000h,30,00000002h,31,04000000h
endm

change50rest   macro
        mov eax,is18b
        mov edx,dword ptr keysetup[16]
        and edx,10000000h
        .if eax != edx
            xor eax,10000000h
            mov is18b,eax
            changethreekeydword 27,00080000h,30,00000002h,31,04000000h
        .endif
endm


change52new   macro
;changetwokeydword    4,00000001h, 7,00100000h
changethreekeydword   7,00100000h, 14,00000008h, 11,00800000h   ;NB changed
changetwokeydword   16,02000000h,18,00000002h
changethreekeydword   21,00400000h,24,40000000h,25,80000000h
;changetwokeydword   25,80000000h,28,80000000h
;changetwokeydword   29,00200000h,32,00800000h
endm
; rest changed before round 14

;NB NB New, january 1998.

change52rest   macro
        mov eax,is52b
        mov edx,dword ptr keysetup[24]
        and edx,00100000h
        .if eax != edx
            xor eax,00100000h
            mov is52b,eax
            changethreekeydword 28,80000000h,29,00200000h,32,00800000h
        .endif
endm



redo2box3macro macro

        mov esi,ediafter2comp
        mov ebp,redo2box3comp
        mov edi,ediafter2
        xor esi,ebp
        mov ebp,redo2box3
        mov ediafter2comp,esi
        xor edi,ebp
        mov esi,esiafter1
        mov ediafter2,edi

        endm


redo2box5macro  macro

        mov esi,ediafter2comp
        mov ebp,redo2box5comp
        mov edi,ediafter2
        xor esi,ebp
        mov ebp,redo2box5
        mov ediafter2comp,esi
        xor edi,ebp
        mov esi,esiafter1
        mov ediafter2,edi

        endm


initialpermumacro     macro  reg1, reg2, reg3
        ; The initial and final permutation code is inspired by the
        ; Eric Young, who again was inspired by others.
        ; See the Libdes library.

        rol reg2,4
        mov reg3,reg1
        xor reg1,reg2
        and reg1,0F0F0F0F0h
        xor reg3,reg1
        xor reg2,reg1
        ror reg3,12
        mov reg1,reg3
        xor reg3,reg2
        and reg3,000FFFF0h
        xor reg1,reg3
        xor reg2,reg3
        rol reg1,14
        mov reg3,reg1
        xor reg1,reg2
        and reg1,033333333h
        xor reg3,reg1
        xor reg2,reg1
        ror reg3,6
        mov reg1,reg3
        xor reg3,reg2
        and reg3,0FF00FF0h
        xor reg1,reg3
        xor reg2,reg3
        rol reg1,7
        mov reg3,reg1
        xor reg1,reg2
        and reg1,055555555h
        xor reg3,reg1
        xor reg2,reg1
        ror reg2,1
        mov reg1,reg3
        endm

finalpermumacro  macro reg1,reg2,reg3

        rol       reg1,1
        mov       reg3,reg2
        xor       reg2,reg1
        and       reg2,55555555h
        xor       reg1,reg2
        xor       reg2,reg3

        ror       reg2,7
        mov       reg3,reg1
        xor       reg1,reg2
        and       reg1,0FF00FF0h
        xor       reg2,reg1
        xor       reg1,reg3

        ror       reg1,6
        mov       reg3,reg2
        xor       reg2,reg1
        and       reg2,0CCCCCCCCh
        xor       reg1,reg2
        xor       reg2,reg3

        rol       reg1,14
        mov       reg3,reg2
        xor       reg2,reg1
        and       reg2,0FFFF000h
        xor       reg1,reg2
        xor       reg2,reg3

        ror       reg1,12
        mov       reg3,reg2
        xor       reg2,reg1
        and       reg2,0F0F0F0Fh
        xor       reg1,reg2
        xor       reg2,reg3

        rol       reg2,4

        endm



desround    macro roundno,reg1,reg2

        mov eax,dword ptr keysetup [(roundno-1)*8]
        xor eax,reg2
        mov edx,dword ptr keysetup [((roundno-1)*8+4)]
        xor edx,reg2
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        ;;;mov bl,al
        movzx ebx,al
        rol edx,4
        ;;;mov cl,ah
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        ;;;mov bl,dl
        movzx ebx,dl
        xor reg1,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor reg1,ebp
        ;;;mov cl,dh
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor reg1,ebp
        mov ebp,dword ptr sbox7 [ecx]
        ;;;mov bl,ah
        movzx ebx,ah
        xor reg1,ebp
        ;;;mov cl,dh
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor reg1,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor reg1,ebp
        mov ebp,dword ptr sbox4 [eax]
        xor reg1,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor reg1,ebp
        endm

desround1    macro
        mov esi,dword ptr plainpermu
        mov edi,dword ptr plainpermu [4]

        mov eax,dword ptr keysetup [(1-1)*8]
        xor eax,edi
        mov edx,dword ptr keysetup [(1-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        ;;;mov bl,al
        movzx ebx,al
        rol edx,4
        ;;;mov cl,ah
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        ;;;mov bl,dl
        movzx ebx,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor esi,ebp
        ;;;mov cl,dh
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]

        xor ebx,10h
        xor esi,ebp

        mov edi,dword ptr sbox7 [ecx]
        ;;;mov cl,dh
        movzx ecx,dh

        xor esi,edi
        mov edi,dword ptr sbox1 [ebx]

        xor edi,ebp
        xor ebx,ebx

        mov redo1box1,edi
        ;;;mov bl,ah
        movzx ebx,ah

        and eax,0FFh
        mov ebp,dword ptr sbox2 [ebx]

        xor esi,ebp
        mov ebp,dword ptr sbox3 [ecx]

        xor esi,ebp
        xor ecx,04h

        and edx,0FFh
        mov edi,dword ptr sbox4 [eax]

        xor esi,edi
        mov edi,dword ptr sbox3 [ecx]

        mov eax,dword ptr sbox5 [edx]
        xor edx,10h

        xor edi,ebp
        xor ecx,ecx

        mov redo1box3,edi
        mov edx,dword ptr sbox5 [edx]

        xor esi,eax
        xor edx,eax

        mov redo1box5,edx

        endm


desround1comp    macro

        mov esi,dword ptr plainpermucomp
        mov edi,dword ptr plainpermucomp [4]

        mov eax,dword ptr keysetup [(1-1)*8]
        xor eax,edi
        mov edx,dword ptr keysetup [(1-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        ;;;mov bl,al
        movzx ebx,al
        rol edx,4
        ;;;mov cl,ah
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        ;;;mov bl,dl
        movzx ebx,dl
        xor esi,ebp

        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor esi,ebp
        ;;;mov cl,dh
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]

        xor ebx,10h
        xor esi,ebp

        mov edi,dword ptr sbox7 [ecx]
        ;;;mov cl,dh
        movzx ecx,dh

        xor esi,edi
        mov edi,dword ptr sbox1 [ebx]

        xor edi,ebp
        xor ebx,ebx

        mov redo1box1comp,edi
        ;;;mov bl,ah
        movzx ebx,ah

        and eax,0FFh
        mov ebp,dword ptr sbox2 [ebx]

        xor esi,ebp
        mov ebp,dword ptr sbox3 [ecx]

        xor esi,ebp
        xor ecx,04h

        and edx,0FFh
        mov edi,dword ptr sbox4 [eax]

        xor esi,edi
        mov edi,dword ptr sbox3 [ecx]

        mov eax,dword ptr sbox5 [edx]
        xor edx,10h

        xor edi,ebp
        xor ecx,ecx

        mov redo1box3comp,edi
        mov edx,dword ptr sbox5 [edx]

        xor esi,eax
        xor edx,eax

        mov redo1box5comp,edx

        endm


desround2comp    macro
        mov eax,dword ptr keysetup [(2-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(2-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        ;;;mov bl,al
        movzx ebx,al
        rol edx,4
        ;;;mov cl,ah
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        ;;;mov bl,dl
        movzx ebx,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        ;;;mov cl,dh
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]

        xor edi,ebp
        mov round2box1ptrcomp,ebx

        mov undo2box1comp,ebp
        mov ebp,dword ptr tbox1 [ebx]

        mov redo2box1comp,ebp
        mov ebp,dword ptr sbox7 [ecx]

        ;;;mov bl,ah
        movzx ebx,ah
        xor edi,ebp
        ;;;mov cl,dh
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor edi,ebp

        mov ebp,dword ptr tbox3 [ecx]
        mov redo2box3comp,ebp

        mov ebp,dword ptr sbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor edi,ebp

        mov ebp,dword ptr tbox5 [edx]
        mov redo2box5comp,ebp

        endm

desround2   macro
        mov eax,dword ptr keysetup [(2-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(2-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        ;;;mov bl,al
        movzx ebx,al
        rol edx,4
        ;;;mov cl,ah
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        ;;;mov bl,dl
        movzx ebx,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        ;;;mov cl,dh
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor edi,ebp

        mov round2box1ptr,ebx
        mov undo2box1,ebp
        mov ebp,dword ptr tbox1 [ebx]
        mov redo2box1,ebp

        mov ebp,dword ptr sbox7 [ecx]
        ;;;mov bl,ah
        movzx ebx,ah
        xor edi,ebp
        ;;;mov cl,dh
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor edi,ebp

        mov ebp,dword ptr tbox3 [ecx]
        mov redo2box3,ebp

        mov ebp,dword ptr sbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor edi,ebp

        mov ebp,dword ptr tbox5 [edx]
        mov redo2box5,ebp

        endm


redo2box1macro macro      ; redo round 2 box 1

        mov ebx,round2box1ptrcomp
        mov edx,round2box1ptr

        xor ebx,40h
        mov eax,undo2box1comp

        xor edx,40h
        mov esi,ediafter2comp

        xor esi,eax
        mov ebp,dword ptr sbox1 [ebx]

        xor esi,ebp
        mov ebp,dword ptr tbox1 [ebx]

        mov edi,ediafter2
        mov eax,undo2box1

        mov redo2box1comp,ebp
        xor edi,eax

        xor ebx,ebx
        mov ebp,dword ptr sbox1 [edx]

        xor edi,ebp
        mov ebp,dword ptr tbox1 [edx]

        mov redo2box1,ebp
        mov ediafter2comp,esi
        mov ediafter2,edi
        mov esi,esiafter1

        endm


desround12part1   macro
        mov eax,dword ptr keysetup [(12-1)*8]

        xor eax,esi
        mov edx,dword ptr keysetup [(12-1)*8+4]

        xor edx,esi
        and eax,0FCFCFCFCh

        and edx,0CFCFCFCFh
        ;;;mov cl,ah
        movzx ecx,ah

        rol edx,4

        mov saveeax,eax
        ;;;mov bl,dl
        movzx ebx,dl

        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]

        xor edi,ebp
        mov ebp,dword ptr sbox1 [ebx]

        xor edi,ebp
        ;;;mov cl,dh
        movzx ecx,dh

        mov saveedx,edx
        ;;;mov bl,ah
        movzx ebx,ah

        mov ebp,dword ptr sbox7 [ecx]
        ;;;mov cl,byte ptr saveedx [2]
        movzx ecx,byte ptr saveedx [2]

        xor edi,ebp
        and eax,0FFh

        mov ebp,dword ptr sbox5 [ecx]
        mov edx,dword ptr keysetup [(12-1+1)*8+4]

        xor edi,ebp
        mov ebp,dword ptr sbox2 [ebx]

        xor edi,ebp
        mov ebp,dword ptr sbox4 [eax]

        xor edi,ebp
        mov ebp,compcontrol

        xor edx,edi
        mov eax,checkoffset

        and edx,0FC00000h
        ;;;mov cl,byte ptr saveedx [3]
        movzx ecx,byte ptr saveedx [3]
        shr edx,20
        mov eax,[eax][ebp]

        mov saveregister,esi
        and esi,20080820h

        xor eax,esi
        mov edx,dword ptr sbox3 [edx]

        cmp edx,eax
        je desround12rest
        ; ebp compcontrol

        endm

desround12part2 macro

desround12rest:
        ;mov cl,byte ptr saveedx [3]
        ;;;mov bl,byte ptr saveeax
        movzx ebx,byte ptr saveeax
        mov ebp,dword ptr sbox3 [ecx]

        xor edi,ebp
        mov ebp,dword ptr sbox8 [ebx]

        xor edi,ebp
        mov esi,saveregister

        endm



desround13   macro
        mov eax,dword ptr keysetup [(13-1)*8]
        xor eax,edi
        mov edx,dword ptr keysetup [(13-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        ;;;mov bl,al
        movzx ebx,al
        rol edx,4
        ;;;mov cl,ah
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        ;;;mov bl,dl
        movzx ebx,dl
        xor esi,ebp
        shr eax,16
        mov saveedx,edx
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]

        xor esi,ebp
        mov ebp,dword ptr sbox6 [ecx]

        ;;;mov bl,ah
        movzx ebx,ah
        and eax,0FFh

        xor esi,ebp
        ;;;mov cl,dh
        movzx ecx,dh

        mov ebp,dword ptr sbox4 [eax]
        mov edx,dword ptr keysetup [(13-1+1)*8+4]

        xor esi,ebp
        mov ebp,dword ptr sbox3 [ecx]

        xor esi,ebp
        mov ebp,dword ptr sbox2 [ebx]

        xor esi,ebp
        mov ebp,compcontrol

        xor edx,esi
        mov saveregister,edi

        shr edx,12
        and edi,08200401h

        mov ebp,round15box5check [ebp]
        and edx,0FCh

        xor ebp,edi
        mov edx,dword ptr sbox5 [edx]     ; agi

        cmp edx,ebp
        je _51

        mov ebp,compcontrol
        jmp test2


_51:    ;;;mov bl,byte ptr saveedx [2]
        movzx ebx,byte ptr saveedx [2]
        ;;;mov cl,byte ptr saveedx [1]
        movzx ecx,byte ptr saveedx [1]

        mov eax,is14b
        mov edx,dword ptr keysetup [24]
        and edx,00000800h
        .if eax != edx
            xor eax,00000800h
            mov is14b,eax
            changetwokeydword 26,00000800h,27,00000040h
            changetwokeydword 29,00004000h,32,00000200h
            xor ecx,80h
        .endif

        mov edi,saveregister
        mov ebp,dword ptr sbox5 [ebx]  ; agi

        xor esi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        xor ecx,ecx

        xor esi,ebp

        endm

desround14   macro
        mov eax,dword ptr keysetup [(14-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(14-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        ;;;mov bl,al
        movzx ebx,al
        rol edx,4
        ;;;mov cl,ah
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        ;;;mov bl,dl
        movzx ebx,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        ;;;mov cl,dh
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]

        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]

        ;;;mov cl,dh
        movzx ecx,dh
        and edx,0FFh

        xor edi,ebp
        mov ebp,compcontrol
        mov ebp,round15box2check [ebp]

        mov saveregister,esi
        mov edx,dword ptr sbox5 [edx]

        xor edi,edx
        mov edx,dword ptr sbox3 [ecx]

        xor edi,edx
        mov edx,dword ptr keysetup [(14-1+1)*8]

        xor edx,edi
        and esi,00420082h

        shr edx,26
        xor ebp,esi

        mov edx,dword ptr sbox2 [edx*4]
        mov esi,1 ; means nothing, will not delete it and change alignment now

        cmp edx,ebp
        je _52
        mov ebp,compcontrol
        jmp test2


_52:    ;;;mov bl,ah
        movzx ebx,ah
        and eax,0FFh
        mov esi,saveregister
        mov ebp,dword ptr sbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr sbox2 [ebx]
        xor edi,ebp

        endm


desmacro310  macro

        ; round 3 to 10
        local roundno
        xor ebx,ebx   ; Might enhance Pentium Pro speed.
            irp roundno,<3,5,7,9>
                desround roundno,esi,edi
                desround (roundno+1),edi,esi
            endm
        endm


ch52round3to12a      macro
        xor ecx,ecx   ; Might enhance Pentium Pro speed.
        mov eax,dword ptr keysetup [(3-1)*8]
        xor eax,edi
        mov edx,dword ptr keysetup [(3-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ebx,al
        rol edx,4
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        movzx ebx,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor esi,ebp
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        movzx ebx,ah
        xor esi,ebp
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr sbox4 [eax]
        xor esi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor esi,ebp

        mov eax,dword ptr keysetup [(4-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(4-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ebx,al
        rol edx,4
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        movzx ebx,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        movzx ebx,ah
        xor edi,ebp
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor edi,ebp
        mov ebp,dword ptr round4box4 [eax]
        xor edi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor edi,ebp

        mov eax,dword ptr keysetup [(5-1)*8]
        xor eax,edi
        mov edx,dword ptr keysetup [(5-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ebx,al
        rol edx,4
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        movzx ebx,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor esi,ebp
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        movzx ebx,ah
        xor esi,ebp
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr sbox4 [eax]
        xor esi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor esi,ebp

        mov eax,dword ptr keysetup [(6-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(6-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ebx,al
        rol edx,4
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        movzx ebx,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        movzx ebx,ah
        xor edi,ebp
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor edi,ebp
        mov ebp,dword ptr round6box4 [eax]
        xor edi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor edi,ebp

        mov eax,dword ptr keysetup [(7-1)*8]
        xor eax,edi
        mov edx,dword ptr keysetup [(7-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ebx,al
        rol edx,4
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        movzx ebx,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor esi,ebp
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr round7box1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        movzx ebx,ah
        xor esi,ebp
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr sbox4 [eax]
        xor esi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor esi,ebp

        mov eax,dword ptr keysetup [(8-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(8-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ebx,al
        rol edx,4
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        movzx ebx,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        movzx ebx,ah
        xor edi,ebp
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr round8box3 [ecx]
        xor edi,ebp
        mov ebp,dword ptr sbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor edi,ebp

        mov eax,dword ptr keysetup [(9-1)*8]
        xor eax,edi
        mov edx,dword ptr keysetup [(9-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ebx,al
        rol edx,4
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        movzx ebx,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor esi,ebp
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr round9box1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        movzx ebx,ah
        xor esi,ebp
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr sbox4 [eax]
        xor esi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor esi,ebp

        mov eax,dword ptr keysetup [(10-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(10-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ebx,al
        rol edx,4
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        movzx ebx,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        movzx ebx,ah
        xor edi,ebp
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor edi,ebp
        mov ebp,dword ptr sbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor edi,ebp

        mov eax,dword ptr keysetup [(11-1)*8]
        xor eax,edi
        mov edx,dword ptr keysetup [(11-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ebx,al
        rol edx,4
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        movzx ebx,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor esi,ebp
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        movzx ebx,ah
        xor esi,ebp
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr round11box4 [eax]
        xor esi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor esi,ebp

        mov eax,dword ptr keysetup [(12-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(12-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ecx,ah
        rol edx,4
        mov saveeax,eax
        movzx ebx,dl
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        mov ebp,dword ptr round12box1 [ebx]
        xor edi,ebp
        movzx ecx,dh
        mov saveedx,edx
        movzx ebx,ah
        mov ebp,dword ptr sbox7 [ecx]
        movzx ecx,byte ptr saveedx [2]
        xor edi,ebp
        and eax,0FFh
        mov ebp,dword ptr sbox5 [ecx]
        mov edx,dword ptr keysetup [(12-1+1)*8+4]
        xor edi,ebp
        mov ebp,dword ptr sbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox4 [eax]
        xor edi,ebp
        mov ebp,compcontrol
        xor edx,edi
        mov eax,checkoffset
        and edx,0FC00000h
        movzx ecx,byte ptr saveedx [3]
        shr edx,20
        mov eax,[eax][ebp]
        mov saveregister,esi
        and esi,20080820h
        xor eax,esi
        mov edx,dword ptr sbox3 [edx]
        cmp edx,eax
        je ch52round12rest
        ; ebp = compcontrol

        endm



ch52round12b  macro

ch52round12rest:

        movzx ebx,byte ptr saveeax
        mov ebp,dword ptr sbox3 [ecx]
        xor edi,ebp
        mov ebp,dword ptr sbox8 [ebx]
        xor edi,ebp
        mov esi,saveregister

        endm


ch52round13   macro
        mov eax,dword ptr keysetup [(13-1)*8]
        xor eax,edi
        mov edx,dword ptr keysetup [(13-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ebx,al
        rol edx,4
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        movzx ebx,dl
        xor esi,ebp
        shr eax,16
        mov saveedx,edx
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox6 [ecx]
        movzx ebx,ah
        and eax,0FFh
        xor esi,ebp
        movzx ecx,dh
        mov ebp,dword ptr sbox4 [eax]
        xor ebx,80h
        xor esi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr sbox2 [ebx]
        mov edx,dword ptr keysetup [(13-1+1)*8+4]
        xor ebx,ebx
        xor esi,ebp
        mov ebp,compcontrol
        xor edx,esi
        mov saveregister,edi
        shr edx,12
        and edi,08200401h
        mov ebp,round15box5check [ebp]
        and edx,0FCh
        xor ebp,edi
        mov edx,dword ptr sbox5 [edx]     ; agi
        cmp edx,ebp
        je _53
        mov ebp,compcontrol
        jmp test52changed2

_53:    movzx ebx,byte ptr saveedx [2]
        movzx ecx,byte ptr saveedx [1]

        mov eax,is14b
        mov edx,dword ptr keysetup[24]
        and edx,00000800h
        .if eax != edx     ; change bit 14 for round 13 to 16
            xor eax,00000800h
            mov is14b,eax
            changetwokeydword 26,00000800h,27,00000040h
            changetwokeydword 29,00004000h,32,00000200h
            xor ecx,80h
        .endif

        mov edi,saveregister
        mov ebp,dword ptr sbox5 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        xor ecx,ecx
        xor esi,ebp

        endm

ch52round14   macro
        mov eax,dword ptr keysetup [(14-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(14-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ebx,al
        rol edx,4
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        movzx ebx,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        movzx ecx,dh
        xor ebx,08h
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor ebx,ebx
        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        movzx ecx,dh
        and edx,0FFh
        xor edi,ebp
        mov ebp,compcontrol
        mov ebp,round15box2check [ebp]
        mov saveregister,esi
        mov edx,dword ptr sbox5 [edx]
        xor edi,edx
        mov edx,dword ptr sbox3 [ecx]
        xor edi,edx
        mov edx,dword ptr keysetup [(14-1+1)*8]
        xor edx,edi
        and esi,00420082h
        shr edx,26
        xor ebp,esi
        mov edx,dword ptr sbox2 [edx*4]
        cmp edx,ebp
        je _54
        mov ebp,compcontrol
        jmp test52changed2

_54:    movzx ebx,ah
        and eax,0FFh
        mov esi,saveregister
        mov ebp,dword ptr sbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr sbox2 [ebx]
        xor edi,ebp

        endm

ch52round15    macro

        mov eax,dword ptr keysetup [(15-1)*8]
        xor eax,edi
        mov edx,dword ptr keysetup [(15-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ebx,al
        rol edx,4
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        xor esi,ebp
        movzx ebx,dl
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor esi,ebp
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        movzx ebx,ah
        xor esi,ebp
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor esi,ebp
        xor eax,20h
        mov ebp,dword ptr sbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr sbox4 [eax]
        xor esi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor esi,ebp
        endm

ch52round16  macro
        mov eax,dword ptr keysetup [(16-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(16-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        movzx ebx,al
        rol edx,4
        movzx ecx,ah
        mov ebp,dword ptr sbox8 [ebx]
        movzx ebx,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        movzx ecx,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        movzx ebx,ah
        xor edi,ebp
        movzx ecx,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor ecx,08h
        xor edi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor ecx,ecx
        xor edi,ebp
        mov ebp,dword ptr sbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor edi,ebp

        endm

testbit52changed macro

        mov edi,ediafter2comp
        mov ebp,redo2box1comp
        mov esi,esiafter1comp
        xor edi,ebp

test52keyfrom3:
        ch52round3to12a

test52changed2:
        ;mov ebp,compcontrol  ;   NB must be set already
        .if ebp == 0
            ret
        .endif

        mov edi,ediafter2
        mov ebp,redo2box1
        mov esi,esiafter1
        xor edi,ebp

        mov compcontrol,0
        jmp test52keyfrom3

;rest of original key
        ch52round12b

        ch52round13
        ;change46rest moved to round 13
        change50rest
        ch52round14
        ch52round15

        mov ebp,compcontrol
        mov edx,dword ptr cipherpermu [4][ebp]
        .if esi == edx
            ch52round16
            mov ebp,compcontrol
            mov eax,dword ptr cipherpermu [ebp]
            .if edi == eax
                call key_from_permu     ;resultat i esi edi
                mov ebp,compcontrol
                .if ebp != 0
                    xor esi,0FFFFFFFFh
                    xor edi,0FFFFFFFFh
                .endif
                xor edi,00100000h  ; this is bit 52 when stored to mem
                call key_found_low
            .endif
        .endif
        xor ebx,ebx
        xor ecx,ecx
        mov ebp,compcontrol
        jmp test52changed2

        endm
; end include of bdesmcp.inc

        .data

;        include bdesdat.inc    ; DES data
; BDESDAT.INC

; Part of BrydDES Key Search Library version 1.01.
; Date: January 17, 1998.
; Copyright Svend Olaf Mikkelsen, 1995, 1997, 1998.


; Please read README.TXT.

; Written for Microsoft Macro Assembler, version 6.00B.
; Modified, so it also seems to assemble correct with TASM 5.0.


sboxdata    byte 0E0h,04Fh,0D7h,014h,02Eh,0F2h,0BDh,081h
            byte 03Ah,0A6h,06Ch,0CBh,059h,095h,003h,078h
            byte 04Fh,01Ch,0E8h,082h,0D4h,069h,021h,0B7h
            byte 0F5h,0CBh,093h,07Eh,03Ah,0A0h,056h,00Dh

            byte 0F3h,01Dh,084h,0E7h,06Fh,0B2h,038h,04Eh
            byte 09Ch,070h,021h,0DAh,0C6h,009h,05Bh,0A5h
            byte 00Dh,0E8h,07Ah,0B1h,0A3h,04Fh,0D4h,012h
            byte 05Bh,086h,0C7h,06Ch,090h,035h,02Eh,0F9h

            byte 0ADh,007h,090h,0E9h,063h,034h,0F6h,05Ah
            byte 012h,0D8h,0C5h,07Eh,0BCh,04Bh,02Fh,081h
            byte 0D1h,06Ah,04Dh,090h,086h,0F9h,038h,007h
            byte 0B4h,01Fh,02Eh,0C3h,05Bh,0A5h,0E2h,07Ch

            byte 07Dh,0D8h,0EBh,035h,006h,06Fh,090h,0A3h
            byte 014h,027h,082h,05Ch,0B1h,0CAh,04Eh,0F9h
            byte 0A3h,06Fh,090h,006h,0CAh,0B1h,07Dh,0D8h
            byte 0F9h,014h,035h,0EBh,05Ch,027h,082h,04Eh

            byte 02Eh,0CBh,042h,01Ch,074h,0A7h,0BDh,061h
            byte 085h,050h,03Fh,0FAh,0D3h,009h,0E8h,096h
            byte 04Bh,028h,01Ch,0B7h,0A1h,0DEh,072h,08Dh
            byte 0F6h,09Fh,0C0h,059h,06Ah,034h,005h,0E3h

            byte 0CAh,01Fh,0A4h,0F2h,097h,02Ch,069h,085h
            byte 006h,0D1h,03Dh,04Eh,0E0h,07Bh,053h,0B8h
            byte 094h,0E3h,0F2h,05Ch,029h,085h,0CFh,03Ah
            byte 07Bh,00Eh,041h,0A7h,016h,0D0h,0B8h,06Dh

            byte 04Dh,0B0h,02Bh,0E7h,0F4h,009h,081h,0DAh
            byte 03Eh,0C3h,095h,07Ch,052h,0AFh,068h,016h
            byte 016h,04Bh,0BDh,0D8h,0C1h,034h,07Ah,0E7h
            byte 0A9h,0F5h,060h,08Fh,00Eh,052h,093h,02Ch

            byte 0D1h,02Fh,08Dh,048h,06Ah,0F3h,0B7h,014h
            byte 0ACh,095h,036h,0EBh,050h,00Eh,0C9h,072h
            byte 072h,0B1h,04Eh,017h,094h,0CAh,0E8h,02Dh
            byte 00Fh,06Ch,0A9h,0D0h,0F3h,035h,056h,08Bh

spermu      byte 09,17,23,31,13,28,02,18
            byte 24,16,30,06,26,20,10,01
            byte 08,14,25,03,04,29,11,19
            byte 32,12,22,07,05,27,15,21

keypermu1   byte 57,49,41,33,25,17,09,01,58,50,42,34,26,18
            byte 10,02,59,51,43,35,27,19,11,03,60,52,44,36
            byte 63,55,47,39,31,23,15,07,62,54,46,38,30,22
            byte 14,06,61,53,45,37,29,21,13,05,28,20,12,04
            byte 00,00,00,00,00,00,00,00

; rotate 2 x 28   1 left
keypermu2   byte 02,03,04,05,06,07,08,09,10,11,12,13,14,15
            byte 16,17,18,19,20,21,22,23,24,25,26,27,28,01
            byte 30,31,32,33,34,35,36,37,38,39,40,41,42,43
            byte 44,45,46,47,48,49,50,51,52,53,54,55,56,29
            byte 00,00,00,00,00,00,00,00

keypermu3   byte 14,17,11,24,01,05,00,00,03,28,15,06,21,10,00,00
            byte 23,19,12,04,26,08,00,00,16,07,27,20,13,02,00,00
            byte 41,52,31,37,47,55,00,00,30,40,51,45,33,48,00,00
            byte 44,49,39,56,34,53,00,00,46,42,50,36,29,32,00,00


; Data for changing bits in key setup:

; Not all of these data are used, i.e. some are not tested.

changebit1 byte 14
byte  1,10 ,  3, 4 ,  6, 8 ,  7,14 ,  9, 2
byte 12, 9 , 14,32 , 17, 5 , 20, 6 , 22,30
byte 23, 3 , 25,13 , 28,31 , 32,10

changebit2 byte 15
byte  1, 3 ,  4,29 ,  6, 7 ,  7, 6 , 10,10
byte 11, 4 , 14, 8 , 15,14 , 18, 1 , 19,11
byte 24, 5 , 25, 5 , 28, 6 , 30,30 , 31, 9

changebit3 byte 12
byte  2, 5 ,  5,12 ,  9, 9 , 12,29 , 14, 7
byte 15, 6 , 19,10 , 22, 2 , 23, 1 , 26, 1
byte 27,11 , 32,32

changebit4 byte 14
byte  2,18 ,  6,14 ,  7,27 ,  9,22 , 11,25
byte 14,21 , 15,26 , 18,13 , 20,23 , 22,16
byte 25,21 , 28,15 , 29,29 , 32,24

changebit5 byte 14
byte  1,19 ,  3,27 ,  5,22 ,  7,25 , 10,21
byte 11,26 , 13,18 , 18,16 , 21,21 , 24,15
byte 25,29 , 28,18 , 30,26 , 32,14

changebit6 byte 14
byte  3,26 ,  5,18 ,  9,28 , 12,25 , 13,30
byte 15,17 , 17,29 , 20,18 , 22,26 , 23,19
byte 26,22 , 28,17 , 29,20 , 32,21

changebit7 byte 13
byte  4,25 ,  5,30 ,  7,17 , 10,24 , 14,14
byte 15,27 , 18,22 , 20,17 , 21,20 , 26,13
byte 28,23 , 30,16 , 31,28

changebit9 byte 14
byte  1, 4 ,  4, 2 ,  5, 1 ,  8, 1 ,  9,11
byte 14, 5 , 15, 5 , 17,12 , 21, 9 , 24,29
byte 26, 7 , 27, 6 , 30,10 , 31,10

changebit10 byte 13
byte  2,29 ,  3,13 ,  6,31 ,  9,10 , 12, 2
byte 13, 1 , 16, 1 , 17, 2 , 20, 9 , 22,32
byte 25,12 , 29, 9 , 31, 3

changebit11 byte 14
byte  3, 5 ,  6, 6 ,  8,30 ,  9, 3 , 11,13
byte 14,31 , 18,10 , 19, 4 , 22, 8 , 23,14
byte 25, 2 , 28, 9 , 30,32 , 32, 5

changebit12 byte 13
byte  4,26 ,  5,19 ,  8,22 , 10,17 , 11,20
byte 16,13 , 17,18 , 21,28 , 24,25 , 25,30
byte 27,17 , 30,24 , 32,18

changebit13 byte 14
byte  1,27 ,  4,22 ,  6,17 ,  7,20 , 12,13
byte 14,23 , 16,16 , 17,28 , 20,25 , 21,30
byte 23,17 , 26,24 , 30,14 , 31,19

changebit14 byte 13
byte  1,26 ,  4,13 ,  6,23 ,  8,16 , 11,21
byte 14,15 , 15,29 , 18,24 , 22,14 , 23,27
byte 25,22 , 27,25 , 30,21

changebit15 byte 14
byte  2,25 ,  3,21 ,  6,15 ,  7,29 , 10,18
byte 12,26 , 13,19 , 16,22 , 17,22 , 19,25
byte 22,21 , 23,26 , 25,18 , 29,28

changebit17 byte 14
byte  2, 2 ,  4, 8 ,  5,14 ,  7, 2 , 10, 9
byte 12,32 , 15,12 , 18, 6 , 20,30 , 21, 3
byte 23,13 , 26,31 , 29,10 , 31, 4

changebit18 byte 15
byte  1,13 ,  4, 7 ,  5, 6 ,  8,10 ,  9, 4
byte 12, 8 , 13,14 , 15, 2 , 17,11 , 22, 5
byte 23, 5 , 26, 6 , 28,30 , 29, 3 , 32,29

changebit19 byte 13
byte  1, 5 ,  3,12 ,  7, 9 , 10,29 , 12, 7
byte 13, 6 , 16,10 , 17,10 , 20, 2 , 21, 1
byte 24, 1 , 25,11 , 30, 5

changebit20 byte 14
byte  2,26 ,  4,14 ,  5,27 ,  7,22 ,  9,25
byte 12,21 , 13,26 , 15,18 , 18,23 , 20,16
byte 23,21 , 26,15 , 27,29 , 30,18

changebit21 byte 14
byte  2,22 ,  3,22 ,  5,25 ,  8,21 ,  9,26
byte 11,18 , 15,28 , 19,21 , 22,15 , 23,29
byte 26,18 , 28,26 , 29,19 , 31,27

changebit22 byte 14
byte  2,13 ,  3,18 ,  7,28 , 10,25 , 11,30
byte 13,17 , 16,24 , 18,18 , 20,26 , 21,19
byte 24,22 , 26,17 , 27,20 , 31,26

changebit23 byte 13
byte  1,21 ,  3,30 ,  5,17 ,  8,24 , 12,14
byte 13,27 , 15,22 , 18,17 , 19,20 , 24,13
byte 26,23 , 28,16 , 32,25

changebit25 byte 14
byte  2, 8 ,  3, 1 ,  6, 1 ,  7,11 , 12, 5
byte 13, 5 , 16, 6 , 19, 9 , 22,29 , 24, 7
byte 25, 6 , 28,10 , 29, 4 , 32, 2

changebit26 byte 13
byte  2, 7 ,  4,31 ,  7,10 , 10, 2 , 11, 1
byte 14, 1 , 15,11 , 18, 9 , 20,32 , 23,12
byte 27, 9 , 30,29 , 31,13

changebit27 byte 14
byte  1,12 ,  4, 6 ,  6,30 ,  7, 3 ,  9,13
byte 12,31 , 15,10 , 17, 4 , 20, 8 , 21,14
byte 23, 2 , 26, 9 , 28,32 , 31, 5

changebit28 byte 13
byte  2,14 ,  3,19 ,  6,22 ,  8,17 ,  9,20
byte 14,13 , 16,23 , 19,28 , 22,25 , 23,30
byte 25,17 , 28,24 , 32,26

changebit29 byte 13
byte  1,22 ,  4,17 ,  5,20 , 10,13 , 12,23
byte 14,16 , 18,25 , 19,30 , 21,17 , 24,24
byte 28,14 , 29,27 , 32,22

changebit30 byte 14
byte  1,18 ,  4,23 ,  6,16 ,  9,21 , 12,15
byte 13,29 , 16,18 , 20,14 , 21,27 , 23,22
byte 25,25 , 28,21 , 29,26 , 32,13

changebit31 byte 15
byte  1,30 ,  4,15 ,  5,29 ,  8,18 , 10,26
byte 11,19 , 14,22 , 16,17 , 17,25 , 20,21
byte 21,26 , 23,18 , 27,28 , 30,25 , 31,21

changebit33 byte 13
byte  1, 1 ,  3,14 ,  5, 2 ,  8, 9 , 10,32
byte 13,12 , 18,30 , 19, 3 , 21,13 , 24,31
byte 27,10 , 30, 2 , 32, 8

changebit34 byte 15
byte  2,31 ,  3, 6 ,  6,10 ,  7, 4 , 10, 8
byte 11,14 , 13, 2 , 16, 9 , 20, 5 , 21, 5
byte 24, 6 , 26,30 , 27, 3 , 29,13 , 32, 7

changebit35 byte 14
byte  2, 6 ,  5, 9 ,  8,29 , 10, 7 , 11, 6
byte 14,10 , 15, 4 , 18, 2 , 19, 1 , 22, 1
byte 23,11 , 28, 5 , 29, 5 , 31,12

changebit36 byte 13
byte  1,11 ,  4, 9 ,  6,32 ,  9,12 , 13, 9
byte 16,29 , 17,13 , 20,31 , 23,10 , 26, 2
byte 27, 1 , 30, 1 , 31, 2

changebit37 byte 15
byte  2,17 ,  3,25 ,  6,21 ,  7,26 ,  9,18
byte 13,28 , 16,25 , 17,21 , 20,15 , 21,29
byte 24,18 , 26,26 , 27,19 , 30,22 , 31,22

changebit38 byte 13
byte  2,23 ,  5,28 ,  8,25 ,  9,30 , 11,17
byte 14,24 , 18,26 , 19,19 , 22,22 , 24,17
byte 25,20 , 30,13 , 31,18

changebit39 byte 13
byte  2,15 ,  3,17 ,  6,24 , 10,14 , 11,27
byte 13,22 , 15,25 , 17,20 , 22,13 , 24,23
byte 26,16 , 29,21 , 31,30

changebit41 byte 15
byte  1,14 ,  4, 1 ,  5,11 , 10, 5 , 11, 5
byte 14, 6 , 16,30 , 17, 9 , 20,29 , 22, 7
byte 23, 6 , 26,10 , 27, 4 , 30, 8 , 31, 1

changebit42 byte 12
byte  1, 6 ,  5,10 ,  8, 2 ,  9, 1 , 12, 1
byte 13,11 , 18,32 , 21,12 , 25, 9 , 28,29
byte 30, 7 , 32,31

changebit43 byte 13
byte  4,30 ,  5, 3 ,  7,13 , 10,31 , 13,10
byte 16, 2 , 18, 8 , 19,14 , 21, 2 , 24, 9
byte 26,32 , 29,12 , 32, 6

changebit44 byte 15
byte  2, 9 ,  6, 5 ,  7, 5 , 10, 6 , 12,30
byte 13, 3 , 15,13 , 18, 7 , 19, 6 , 22,10
byte 23, 4 , 26, 8 , 27,14 , 29, 2 , 31,11

changebit45 byte 13
byte  1,25 ,  3,20 ,  8,13 , 10,23 , 12,16
byte 15,21 , 17,30 , 19,17 , 22,24 , 26,14
byte 27,27 , 29,22 , 32,17

changebit46 byte 14
byte  4,16 ,  7,21 , 10,15 , 11,29 , 14,18
byte 16,26 , 18,14 , 19,27 , 21,22 , 23,25
byte 26,21 , 27,26 , 29,18 , 32,23

changebit47 byte 15
byte  1,17 ,  3,29 ,  6,18 ,  8,26 ,  9,19
byte 12,22 , 14,17 , 15,20 , 18,21 , 19,26
byte 21,18 , 25,28 , 28,25 , 29,30 , 32,15

changebit49 byte 13
byte  2, 1 ,  3, 2 ,  6, 9 ,  8,32 , 11,12
byte 15, 9 , 17, 3 , 19,13 , 22,31 , 25,10
byte 28, 2 , 29, 1 , 31,14

changebit50 byte 15
byte  4,10 ,  5, 4 ,  8, 8 ,  9,14 , 11, 2
byte 14, 9 , 16,32 , 18, 5 , 19, 5 , 22, 6
byte 24,30 , 25, 3 , 27,13 , 30,31 , 31, 6

changebit51 byte 14
byte  2,30 ,  3, 9 ,  6,29 ,  8, 7 ,  9, 6
byte 12,10 , 13, 4 , 16, 8 , 17, 1 , 20, 1
byte 21,11 , 26, 5 , 27, 5 , 30, 6

changebit52 byte 12
byte  4,32 ,  7,12 , 11, 9 , 14,29 , 16, 7
byte 18,31 , 21,10 , 24, 2 , 25, 1 , 28, 1
byte 29,11 , 32, 9

changebit53 byte 15
byte  1,20 ,  4,21 ,  5,26 ,  7,18 , 11,28
byte 14,25 , 15,30 , 18,15 , 19,29 , 22,18
byte 24,26 , 25,19 , 28,22 , 30,17 , 31,25

changebit54 byte 13
byte  2,16 ,  3,28 ,  6,25 ,  7,30 ,  9,17
byte 12,24 , 16,14 , 17,19 , 20,22 , 22,17
byte 23,20 , 28,13 , 30,23

changebit55 byte 13
byte  1,29 ,  4,24 ,  8,14 ,  9,27 , 11,22
byte 13,25 , 16,21 , 20,13 , 22,23 , 24,16
byte 27,21 , 30,15 , 31,17

changebit57 byte 15
byte  1, 2 ,  3,11 ,  8, 5 ,  9, 5 , 12, 6
byte 14,30 , 15, 3 , 18,29 , 20, 7 , 21, 6
byte 24,10 , 25, 4 , 28, 8 , 29,14 , 32, 1

changebit58 byte 12
byte  2,10 ,  3,10 ,  6, 2 ,  7, 1 , 10, 1
byte 11,11 , 16, 5 , 19,12 , 23, 9 , 26,29
byte 28, 7 , 29, 6

changebit59 byte 13
byte  1, 9 ,  3, 3 ,  5,13 ,  8,31 , 11,10
byte 14, 2 , 15, 1 , 17,14 , 19, 2 , 22, 9
byte 24,32 , 27,12 , 32,30

changebit60 byte 15
byte  2,32 ,  4, 5 ,  5, 5 ,  8, 6 , 10,30
byte 11, 3 , 13,13 , 16,31 , 17, 6 , 20,10
byte 21, 4 , 24, 8 , 25,14 , 27, 2 , 30, 9

changebit61 byte 13
byte  2,21 ,  6,13 ,  8,23 , 10,16 , 13,21
byte 16,15 , 17,17 , 20,24 , 24,14 , 25,27
byte 27,22 , 29,25 , 31,20

changebit62 byte 14
byte  1,28 ,  5,21 ,  8,15 ,  9,29 , 12,18
byte 14,26 , 15,19 , 17,27 , 19,22 , 21,25
byte 24,21 , 25,26 , 27,18 , 32,16

changebit63 byte 14
byte  2,24 ,  4,18 ,  6,26 ,  7,19 , 10,22
byte 12,17 , 13,20 , 17,26 , 19,18 , 23,28
byte 26,25 , 27,30 , 29,17 , 31,29



changetable dword 80000000h
            dword 40000000h
            dword 20000000h
            dword 10000000h
            dword 08000000h
            dword 04000000h
            dword 02000000h
            dword 01000000h
            dword 00800000h
            dword 00400000h
            dword 00200000h
            dword 00100000h
            dword 00080000h
            dword 00040000h
            dword 00020000h
            dword 00010000h
            dword 00008000h
            dword 00004000h
            dword 00002000h
            dword 00001000h
            dword 00000800h
            dword 00000400h
            dword 00000200h
            dword 00000100h
            dword 00000080h
            dword 00000040h
            dword 00000020h
            dword 00000010h
            dword 00000008h
            dword 00000004h
            dword 00000002h
            dword 00000001h

keyfromsetupdata dword 1,6,01000000h
dword 2,6,02000000h
dword 3,9,00800000h
dword 4,7,00000020h
dword 5,5,00000400h
dword 6,9,00000010h
dword 7,7,00008000h
dword 9,5,80000000h
dword 10,6,00000002h
dword 11,8,00000004h
dword 12,8,00000400h
dword 13,6,00008000h
dword 14,6,00000200h
dword 15,6,00020000h
dword 17,5,00040000h
dword 18,5,04000000h
dword 19,7,00800000h
dword 20,5,00000020h
dword 21,5,00000080h
dword 22,7,00000010h
dword 23,5,00008000h
dword 25,6,80000000h
dword 26,7,00400000h
dword 27,6,00000004h
dword 28,6,00000400h
dword 29,5,00001000h
dword 30,6,00010000h
dword 31,5,00000008h
dword 33,5,40000000h
dword 34,6,00400000h
dword 35,8,00000008h
dword 36,6,00000001h
dword 37,6,00000800h
dword 38,8,00000080h
dword 39,6,00000100h
dword 41,5,00200000h
dword 42,8,40000000h
dword 43,7,00080000h
dword 44,7,08000000h
dword 45,8,00080000h
dword 46,10,00020000h
dword 47,6,00004000h
dword 49,6,00800000h
dword 50,8,01000000h
dword 51,6,00000008h
dword 52,11,00800000h
dword 53,5,00000040h
dword 54,6,00000080h
dword 55,8,00040000h
dword 57,8,08000000h
dword 58,6,40000000h
dword 59,5,00080000h
dword 60,5,08000000h
dword 61,8,00000200h
dword 62,8,00020000h
dword 63,6,00000040h
             
;end include of bdesdat.inc                                                             
        byte "BrydDES Key Search Library version 1.01. Core 2 PRO b. "
        byte "Copyright Svend Olaf Mikkelsen,1995,1997,1998."

changeoffsets   dword offset changebit1
                dword offset changebit2
                dword offset changebit3
                dword offset changebit4
                dword offset changebit5
                dword offset changebit6
                dword offset changebit7
                dword 0
                dword offset changebit9
                dword offset changebit10
                dword offset changebit11
                dword offset changebit12
                dword offset changebit13
                dword offset changebit14
                dword offset changebit15
                dword 0
                dword offset changebit17
                dword offset changebit18
                dword offset changebit19
                dword offset changebit20
                dword offset changebit21
                dword offset changebit22
                dword offset changebit23
                dword 0
                dword offset changebit25
                dword offset changebit26
                dword offset changebit27
                dword offset changebit28
                dword offset changebit29
                dword offset changebit30
                dword offset changebit31
                dword 0
                dword offset changebit33
                dword offset changebit34
                dword offset changebit35
                dword offset changebit36
                dword offset changebit37
                dword offset changebit38
                dword offset changebit39
                dword 0
                dword offset changebit41
                dword offset changebit42
                dword offset changebit43
                dword offset changebit44
                dword offset changebit45
                dword offset changebit46
                dword offset changebit47
                dword 0
                dword offset changebit49
                dword offset changebit50
                dword offset changebit51
                dword offset changebit52
                dword offset changebit53
                dword offset changebit54
                dword offset changebit55
                dword 0
                dword offset changebit57
                dword offset changebit58
                dword offset changebit59
                dword offset changebit60
                dword offset changebit61
                dword offset changebit62
                dword offset changebit63

        .data?
align 4

datastart               dword ?

keysetup                byte 128 dup (?)
plainaddr               dword ?
plainpermu              byte 8 dup (?)
plainpermucomp          byte 8 dup (?)
cipheraddr              dword ?
ivaddr                  dword ?
foundkeyaddr            dword ?
maskaddr                dword ?

sbox1                   dword 64 dup (?)
sbox2                   dword 64 dup (?)
sbox3                   dword 64 dup (?)
sbox4                   dword 64 dup (?)
sbox5                   dword 64 dup (?)
sbox6                   dword 64 dup (?)
sbox7                   dword 64 dup (?)
sbox8                   dword 64 dup (?)

round4box4              dword 64 dup (?)
round6box4              dword 64 dup (?)
round7box1              dword 64 dup (?)
round8box3              dword 64 dup (?)
round9box1              dword 64 dup (?)
round11box4             dword 64 dup (?)
round12box1             dword 64 dup (?)

tbox1                   dword 64 dup (?)
tbox5                   dword 64 dup (?)
tbox3                   dword 64 dup (?)

                        dword 64 dup (?)  ; not used, speed lost if removed?
                        dword 64 dup (?)  ; not used

keyaddr                 dword ?
tempkey                 byte 8 dup(?)
foundkey                byte 8 dup(?)
                        dword ?  ; saveesi not used
                        dword ?  ; saveesicomp not used


                        dword ?    ; not used
                        dword ?    ; not used
                        dword ?    ; not used
                        dword ?    ; not used

ediafter2comp           dword ?
L1                      dword ?

;theese 10 lines must be in this order
cipherpermu             byte 8 dup (?)
cipherpermucheck        dword ?    ; prepared, but not used
round15box2check        dword ?
round15box5check        dword ?
                        dword ?    ; do not remove this line
cipherpermucomp         byte 8 dup (?)
cipherpermucheckcomp    dword ?    ; prepared, but not used
round15box2checkcomp    dword ?
round15box5checkcomp    dword ?
                        dword ?

                        dword ?    ; not used
saveregister            dword ?

undo2box1               dword ?
undo2box1comp           dword ?
redo2box3               dword ?
redo2box1               dword ?
ediafter2               dword ?
redo2box5               dword ?
redo2box5comp           dword ?
saveeax                 dword ?
redo2box1comp           dword ?
redo2box3comp           dword ?

                        dword ?    ; not used
                        dword ?    ; not used

redo16box2              dword ?
redo16box2comp          dword ?
redo16box7              dword ?
redo16box7comp          dword ?
saveedx                 dword ?

;theese 10 lines must be in this order
round13box3check00      dword ?
round13box3check01      dword ?
round13box3check10      dword ?
round13box3check11      dword ?
checkoffset             dword ?
                        dword ?    ; do not remove this line
round13box3check00comp  dword ?    ; offset 24 from round13box3check00
round13box3check01comp  dword ?
round13box3check10comp  dword ?
round13box3check11comp  dword ?

                        dword ?    ; not used
cipherpermucompbox3bits dword ?
cipherpermubox3bits     dword ?

round2box1ptr           dword ?
round2box1ptrcomp       dword ?
saveesiround16          dword ?
saveesiround16comp      dword ?

esiafter1               dword ?
esiafter1comp           dword ?

redo1box1               dword ?
redo1box3               dword ?
redo1box5               dword ?
redo1box1comp           dword ?
redo1box3comp           dword ?
redo1box5comp           dword ?
saveesp                 dword ?

compcontrol             dword ?
is14b                   dword ?
is18b                   dword ?
keyisfound              dword ?

firstcall               dword ?

table_of_changedata dword 28 dup(?);   numbered from 1

keysetupoffset          dword ?
bit61_62_63_change      dword ?

                        dword ?
add_zero                dword ?
dataend                 dword ?


        .code
        ;nop
        ;nop
        nop
        nop

p1bryd_des:              ; procedure

bryd_des_frame equ 24

; ptr plain    ptr ciphertext    ptr i_vector    ptr key   ptr  mask
; 0    [in]    4        [out]    8       [in]    12 [in]   16   [in]


; calls bryd_continue and eventually key_found_low

; returns   eax 0  key found, interrupted or not
;           eax 1  not interrupted
;           eax 2  interrupted
;           eax 3  mask error

        push esi
        push edi
        push ebp
        push ebx
        push ecx

        ;make uninitialized data the same at each call
        ;to prevent errors
        mov edi,offset datastart
        mov ecx,offset dataend
        sub ecx,edi
        shr ecx,2
        mov eax,0CE6755BEh         ; random number
        rep stosd

        ; count the mask bits set to zero
        mov eax,[esp][bryd_des_frame+16]  ; mask address
        mov edx,dword ptr [eax]
        mov eax,dword ptr [eax+4]
        bswapmacro eax
        bswapmacro edx
        mov ebx,00000000001001000111010001010000b
        test ebx,eax
        .if !zero?
            mov eax,3
            jmp bryd_end
        .endif
        test eax,00000000000000000000000000001110b
        .if zero?
            mov bit61_62_63_change,1
        .else
            mov bit61_62_63_change,0
        .endif
        xor eax,ebx    ; bit 43 46 50 51 52 54 58 60 set to zero
        or eax, 01010101h
        or edx, 01010101h
        mov ebx,0                  ; count of bits to change
        mov ecx,64
_01:    test edx,80000000h
        .if zero?
            add ebx,1
        .endif
        call shiftleft_edx_eax
        loop _01
        .if ebx > 27
            mov eax,3
            jmp bryd_end
        .endif
        mov esi,28
        sub esi,ebx
        mov firstcall,esi

        ;set up table with the bit numbers to be changed
        mov eax,[esp][bryd_des_frame+16]  ; mask address
        mov edx,dword ptr [eax]
        mov eax,dword ptr [eax+4]
        bswapmacro eax
        bswapmacro edx
        xor eax,00000000001001000111010001010000b
        or eax,01010101h          ; parity bits
        or edx,01010101h
        mov ecx,64
        mov ebx,1
_02:    test edx,80000000h
        .if zero?
                mov ebp,changeoffsets [ebx*4-4]
                mov table_of_changedata [esi*4],ebp
                inc esi
        .endif
        call shiftleft_edx_eax
        inc ebx
        loop _02

        mov is14b,0
        mov is18b,0
        mov compcontrol,0
        mov keyisfound,0

        mov saveesp,esp

        mov esi,[esp][bryd_des_frame+0]
        mov plainaddr,esi

        mov esi,[esp][bryd_des_frame+4]
        mov cipheraddr,esi

        mov esi,[esp][bryd_des_frame+8]
        mov ivaddr,esi

        mov esi,[esp][bryd_des_frame+12]
        mov keyaddr,esi

        mov esi,[esp][bryd_des_frame+16]
        mov maskaddr,esi

        ; setup initial key and set up S-boxes.
        mov ebp,keyaddr

        mov eax,dword ptr [ebp]
        mov ebx,maskaddr
        and eax,dword ptr [ebx]
        mov dword ptr [ebp],eax

        mov eax,dword ptr [ebp+4]
        and eax,dword ptr [ebx+4]
        mov dword ptr [ebp+4],eax

        push ebp
        call p1desinit
        add esp,4

        ; round4box4 is used for testing bit 52 changed
        ; whithout changing the key setup.
        mov ecx,64
        mov ebx,0
_03:    xor ebx,10h
        mov eax,dword ptr sbox4 [ebx]
        xor ebx,10h
        mov dword ptr round4box4 [ebx],eax
        add ebx,4
        loop _03

        mov ecx,64
        mov ebx,0
_04:    xor ebx,80h
        mov eax,dword ptr sbox4 [ebx]
        xor ebx,80h
        mov dword ptr round6box4 [ebx],eax
        add ebx,4
        loop _04

        mov ecx,64
        mov ebx,0
_05:    xor ebx,80h
        mov eax,dword ptr sbox1 [ebx]
        xor ebx,80h
        mov dword ptr round7box1 [ebx],eax
        add ebx,4
        loop _05

        mov ecx,64
        mov ebx,0
_06:    xor ebx,20h
        mov eax,dword ptr sbox3 [ebx]
        xor ebx,20h
        mov dword ptr round8box3 [ebx],eax
        add ebx,4
        loop _06

        mov ecx,64
        mov ebx,0
_07:    xor ebx,20h
        mov eax,dword ptr sbox1 [ebx]
        xor ebx,20h
        mov dword ptr round9box1 [ebx],eax
        add ebx,4
        loop _07

        mov ecx,64
        mov ebx,0
_08:    xor ebx,40h
        mov eax,dword ptr sbox4 [ebx]
        xor ebx,40h
        mov dword ptr round11box4 [ebx],eax
        add ebx,4
        loop _08

        mov ecx,64
        mov ebx,0
_09:    xor ebx,04h
        mov eax,dword ptr sbox1 [ebx]
        xor ebx,04h
        mov dword ptr round12box1 [ebx],eax
        add ebx,4
        loop _09

        ;tbox1 is used in round 2 for making data
        ;to calculate output with bit 52 changed.
        mov ecx,64
        mov ebx,0
_10:    mov eax,dword ptr sbox1 [ebx]
        xor ebx,10h
        mov edx,dword ptr sbox1 [ebx]
        xor ebx,10h
        xor eax,edx
        mov dword ptr tbox1 [ebx],eax
        add ebx,4
        loop _10

        ;tbox3 is used in round 2 for making data
        ;to calculate output with bit 50 changed.
        mov ecx,64
        mov ebx,0
_11:    mov eax,dword ptr sbox3 [ebx]
        xor ebx,04h
        mov edx,dword ptr sbox3 [ebx]
        xor ebx,04h
        xor eax,edx
        mov dword ptr tbox3 [ebx],eax
        add ebx,4
        loop _11

        ;tbox5 is used in round 2 for making data
        ;to calculate output with bit 46 changed.
        mov ecx,64
        mov ebx,0
_12:    mov eax,dword ptr sbox5 [ebx]
        xor ebx,10h
        mov edx,dword ptr sbox5 [ebx]
        xor ebx,10h
        xor eax,edx
        mov dword ptr tbox5 [ebx],eax
        add ebx,4
        loop _12


        mov ebx,cipheraddr
        mov esi,dword ptr [ebx]
        mov edi,dword ptr [ebx][4]

        call initial_permu

        mov dword ptr cipherpermu,esi
        mov dword ptr cipherpermu [4],edi

        mov eax,40104100h
        and eax,edi
        mov cipherpermucheck,eax     ; not used in this version
        xor eax,40104100h
        mov cipherpermucheckcomp,eax ; not used in this version

        mov eax,00420082h
        and eax,edi
        mov round15box2check,eax     ; expected output from round 15, sbox 2
        xor eax,00420082h
        mov round15box2checkcomp,eax ; ..comp means used for testing
                                     ; complement key

        xor esi,0FFFFFFFFh
        xor edi,0FFFFFFFFh
        mov dword ptr cipherpermucomp,esi
        mov dword ptr cipherpermucomp [4],edi
        and edi,20080820h
        mov dword ptr cipherpermucompbox3bits,edi

        mov edi,dword ptr cipherpermu [4]
        and edi,20080820h
        mov dword ptr cipherpermubox3bits,edi

        mov ebx,ivaddr
        mov esi,dword ptr [ebx]
        mov edi,dword ptr [ebx][4]

        mov ebx,plainaddr
        xor esi,dword ptr [ebx]
        xor edi,dword ptr [ebx][4]

        call initial_permu

        mov dword ptr plainpermu,esi
        mov dword ptr plainpermu [4],edi
        xor esi,0FFFFFFFFh
        xor edi,0FFFFFFFFh
        mov dword ptr plainpermucomp,esi
        mov dword ptr plainpermucomp [4],edi

        ; Start key testing:
        xor ebx,ebx
        xor ecx,ecx

        ; Rule for ebx and ecx:
        ; Whenever other bits than the bl and cl bits are used
        ; the registers must be zeroed afterwards.
        ; Extra xor ebx,ebx and xor ecx,ecx are however inserted
        ; to enhance Pentium Pro speed.

        .if firstcall == 1
            call bit1
        .elseif firstcall == 2
            call bit2
        .elseif firstcall == 3
            call bit3
        .elseif firstcall == 4
            call bit4
        .elseif firstcall == 5
            call bit5
        .elseif firstcall == 6
            call bit6
        .elseif firstcall == 7
            call bit7
        .elseif firstcall == 8
            call bit8
        .elseif firstcall == 9
            call bit9
        .elseif firstcall == 10
            call bit10
        .elseif firstcall == 11
            call bit11
        .elseif firstcall == 12
            call bit12
        .elseif firstcall == 13
            call bit13
        .elseif firstcall == 14
            call bit14
        .elseif firstcall == 15
            call bit15
        .elseif firstcall == 16
            call bit16
        .elseif firstcall == 17
            call bit17
        .elseif firstcall == 18
            call bit18
        .elseif firstcall == 19
            call bit19
        .elseif firstcall == 20
            call bit20
        .elseif firstcall == 21
            call bit21
        .elseif firstcall == 22
            call bit22
        .elseif firstcall == 23
            call bit23
        .elseif firstcall == 24
            call bit24
        .elseif firstcall == 25
            call bit25
        .elseif firstcall == 26
            call bit26
        .elseif firstcall == 27
            call bit27
        .else
            call bitno51
        .endif

        mov edx,0
        .if keyisfound == 1
            mov eax, 0
        .else
            mov eax,1              ; eax 1, finished, not interrupted
        .endif
        jmp bryd_end

bryd_not_continue:
        .if keyisfound == 1
            mov eax, 0
        .else
            mov eax,2
        .endif
bryd_end:
        mov edx,0
        pop ecx
        pop ebx
        pop ebp
        pop edi
        pop esi

        ret


change:         ; local procedure

        mov ebp, table_of_changedata[ebp*4]

        xor eax,eax
        xor edx,edx
        xor ecx,ecx
        mov cl,byte ptr [ebp]
        mov dl,byte ptr [ebp+2]
        mov al,byte ptr [ebp+1]
        inc ebp

_13:    mov edi,dword ptr changetable [edx*4-4]
        mov esi,dword ptr keysetup [eax*4-4]
        add ebp,2
        xor esi,edi
        mov dl,byte ptr [ebp+1]
        mov dword ptr keysetup [eax*4-4],esi
        mov al,byte ptr [ebp]
        loop _13
        xor ecx,ecx

        ret


;        include bdeschg.inc ; key setup change macros
                            ; used for bits which are often changed

; BDESCHG.INC

; Part of BrydDES Key Search Library version 1.01.
; Date: January 17, 1998.
; Copyright Svend Olaf Mikkelsen, 1995, 1997, 1998.


; Please read README.TXT.

; Written for Microsoft Macro Assembler, version 6.00B.
; Modified, so it also seems to assemble correct with TASM 5.0.


change1   macro
changetwokeydword    1,00400000h, 3,10000000h
changetwokeydword    6,01000000h, 7,00040000h
changetwokeydword    9,40000000h,12,00800000h
changetwokeydword   14,00000001h,17,08000000h
changetwokeydword   20,04000000h,22,00000004h
changetwokeydword   23,20000000h,25,00080000h
changetwokeydword   28,00000002h,32,00400000h
endm

change2   macro
changetwokeydword    1,20000000h, 4,00000008h
changetwokeydword    6,02000000h, 7,04000000h
changetwokeydword   10,00400000h,11,10000000h
changetwokeydword   14,01000000h,15,00040000h
changetwokeydword   18,80000000h,19,00200000h
changetwokeydword   24,08000000h,25,08000000h
changethreekeydword 28,04000000h,30,00000004h,31,00800000h
endm

change3   macro
changetwokeydword    2,08000000h, 5,00100000h
changetwokeydword    9,00800000h,12,00000008h
changetwokeydword   14,02000000h,15,04000000h
changetwokeydword   19,00400000h,22,40000000h
changetwokeydword   23,80000000h,26,80000000h
changetwokeydword   27,00200000h,32,00000001h
endm

change4   macro
changetwokeydword    2,00004000h, 6,00040000h
changetwokeydword    7,00000020h, 9,00000400h
changetwokeydword   11,00000080h,14,00000800h
changetwokeydword   15,00000040h,18,00080000h
changetwokeydword   20,00000200h,22,00010000h
changetwokeydword   25,00000800h,28,00020000h
changetwokeydword   29,00000008h,32,00000100h
endm

change5   macro
changetwokeydword    1,00002000h, 3,00000020h
changetwokeydword    5,00000400h, 7,00000080h
changetwokeydword   10,00000800h,11,00000040h
changetwokeydword   13,00004000h,18,00010000h
changetwokeydword   21,00000800h,24,00020000h
changetwokeydword   25,00000008h,28,00004000h
changetwokeydword   30,00000040h,32,00040000h
endm

change6   macro
changetwokeydword    3,00000040h, 5,00004000h
changetwokeydword    9,00000010h,12,00000080h
changetwokeydword   13,00000004h,15,00008000h
changetwokeydword   17,00000008h,20,00004000h
changetwokeydword   22,00000040h,23,00002000h
changetwokeydword   26,00000400h,28,00008000h
changetwokeydword   29,00001000h,32,00000800h
endm

change7   macro
changetwokeydword    4,00000080h, 5,00000004h
changetwokeydword    7,00008000h,10,00000100h
changetwokeydword   14,00040000h,15,00000020h
changetwokeydword   18,00000400h,20,00008000h
changetwokeydword   21,00001000h,26,00080000h
changethreekeydword 28,00000200h,30,00010000h,31,00000010h
endm

change9   macro
changetwokeydword    1,10000000h, 4,40000000h
changetwokeydword    5,80000000h, 8,80000000h
changetwokeydword    9,00200000h,14,08000000h
changetwokeydword   15,08000000h,17,00100000h
changetwokeydword   21,00800000h,24,00000008h
changetwokeydword   26,02000000h,27,04000000h
changetwokeydword   30,00400000h,31,00400000h
endm

change10   macro
changetwokeydword    2,00000008h, 3,00080000h
changetwokeydword    6,00000002h, 9,00400000h
changetwokeydword   12,40000000h,13,80000000h
changetwokeydword   16,80000000h,17,40000000h
changetwokeydword   20,00800000h,22,00000001h
changethreekeydword 25,00100000h,29,00800000h,31,20000000h
endm

change11   macro
changetwokeydword    3,08000000h, 6,04000000h
changetwokeydword    8,00000004h, 9,20000000h
changetwokeydword   11,00080000h,14,00000002h
changetwokeydword   18,00400000h,19,10000000h
changetwokeydword   22,01000000h,23,00040000h
changetwokeydword   25,40000000h,28,00800000h
changetwokeydword   30,00000001h,32,08000000h
endm

change12   macro
changetwokeydword    4,00000040h, 5,00002000h
changetwokeydword    8,00000400h,10,00008000h
changetwokeydword   11,00001000h,16,00080000h
changetwokeydword   17,00004000h,21,00000010h
changetwokeydword   24,00000080h,25,00000004h
changethreekeydword 27,00008000h,30,00000100h,32,00004000h
endm

change13   macro
changetwokeydword    1,00000020h, 4,00000400h
changetwokeydword    6,00008000h, 7,00001000h
changetwokeydword   12,00080000h,14,00000200h
changetwokeydword   16,00010000h,17,00000010h
changetwokeydword   20,00000080h,21,00000004h
changetwokeydword   23,00008000h,26,00000100h
changetwokeydword   30,00040000h,31,00002000h
endm

change14   macro
changetwokeydword    1,00000040h, 4,00080000h
changetwokeydword    6,00000200h, 8,00010000h
changetwokeydword   11,00000800h,14,00020000h
changetwokeydword   15,00000008h,18,00000100h
changetwokeydword   22,00040000h,23,00000020h
changethreekeydword 25,00000400h,27,00000080h,30,00000800h
endm

change15   macro
changetwokeydword    2,00000080h, 3,00000800h
changetwokeydword    6,00020000h, 7,00000008h
changetwokeydword   10,00004000h,12,00000040h
changetwokeydword   13,00002000h,16,00000400h
changetwokeydword   17,00000400h,19,00000080h
changetwokeydword   22,00000800h,23,00000040h
changetwokeydword   25,00004000h,29,00000010h
endm

change17   macro
changetwokeydword    2,40000000h, 4,01000000h
changetwokeydword    5,00040000h, 7,40000000h
changetwokeydword   10,00800000h,12,00000001h
changetwokeydword   15,00100000h,18,04000000h
changetwokeydword   20,00000004h,21,20000000h
changetwokeydword   23,00080000h,26,00000002h
changetwokeydword   29,00400000h,31,10000000h
endm

change18   macro
changetwokeydword    1,00080000h, 4,02000000h
changetwokeydword    5,04000000h, 8,00400000h
changetwokeydword    9,10000000h,12,01000000h
changetwokeydword   13,00040000h,15,40000000h
changetwokeydword   17,00200000h,22,08000000h
changetwokeydword   23,08000000h,26,04000000h
changethreekeydword 28,00000004h,29,20000000h,32,00000008h
endm

change19   macro
changetwokeydword    1,08000000h, 3,00100000h
changetwokeydword    7,00800000h,10,00000008h
changetwokeydword   12,02000000h,13,04000000h
changetwokeydword   16,00400000h,17,00400000h
changetwokeydword   20,40000000h,21,80000000h
changethreekeydword 24,80000000h,25,00200000h,30,08000000h
endm

change20   macro
changetwokeydword    2,00000040h, 4,00040000h
changetwokeydword    5,00000020h, 7,00000400h
changetwokeydword    9,00000080h,12,00000800h
changetwokeydword   13,00000040h,15,00004000h
changetwokeydword   18,00000200h,20,00010000h
changetwokeydword   23,00000800h,26,00020000h
changetwokeydword   27,00000008h,30,00004000h
endm

change21   macro
changetwokeydword    2,00000400h, 3,00000400h
changetwokeydword    5,00000080h, 8,00000800h
changetwokeydword    9,00000040h,11,00004000h
changetwokeydword   15,00000010h,19,00000800h
changetwokeydword   22,00020000h,23,00000008h
changetwokeydword   26,00004000h,28,00000040h
changetwokeydword   29,00002000h,31,00000020h
endm

change22   macro
changetwokeydword    2,00080000h, 3,00004000h
changetwokeydword    7,00000010h,10,00000080h
changetwokeydword   11,00000004h,13,00008000h
changetwokeydword   16,00000100h,18,00004000h
changetwokeydword   20,00000040h,21,00002000h
changetwokeydword   24,00000400h,26,00008000h
changetwokeydword   27,00001000h,31,00000040h
endm

change23   macro
changetwokeydword    1,00000800h, 3,00000004h
changetwokeydword    5,00008000h, 8,00000100h
changetwokeydword   12,00040000h,13,00000020h
changetwokeydword   15,00000400h,18,00008000h
changetwokeydword   19,00001000h,24,00080000h
changethreekeydword 26,00000200h,28,00010000h,32,00000080h
endm

change25   macro
changetwokeydword    2,01000000h, 3,80000000h
changetwokeydword    6,80000000h, 7,00200000h
changetwokeydword   12,08000000h,13,08000000h
changetwokeydword   16,04000000h,19,00800000h
changetwokeydword   22,00000008h,24,02000000h
changetwokeydword   25,04000000h,28,00400000h
changetwokeydword   29,10000000h,32,40000000h
endm

change26   macro
changetwokeydword    2,02000000h, 4,00000002h
changetwokeydword    7,00400000h,10,40000000h
changetwokeydword   11,80000000h,14,80000000h
changetwokeydword   15,00200000h,18,00800000h
changetwokeydword   20,00000001h,23,00100000h
changethreekeydword 27,00800000h,30,00000008h,31,00080000h
endm

change27   macro
changetwokeydword    1,00100000h, 4,04000000h
changetwokeydword    6,00000004h, 7,20000000h
changetwokeydword    9,00080000h,12,00000002h
changetwokeydword   15,00400000h,17,10000000h
changetwokeydword   20,01000000h,21,00040000h
changetwokeydword   23,40000000h,26,00800000h
changetwokeydword   28,00000001h,31,08000000h
endm

change28   macro
changetwokeydword    2,00040000h, 3,00002000h
changetwokeydword    6,00000400h, 8,00008000h
changetwokeydword    9,00001000h,14,00080000h
changetwokeydword   16,00000200h,19,00000010h
changetwokeydword   22,00000080h,23,00000004h
changethreekeydword 25,00008000h,28,00000100h,32,00000040h
endm

change29   macro
changetwokeydword    1,00000400h, 4,00008000h
changetwokeydword    5,00001000h,10,00080000h
changetwokeydword   12,00000200h,14,00010000h
changetwokeydword   18,00000080h,19,00000004h
changetwokeydword   21,00008000h,24,00000100h
changethreekeydword 28,00040000h,29,00000020h,32,00000400h
endm

change30   macro
changetwokeydword    1,00004000h, 4,00000200h
changetwokeydword    6,00010000h, 9,00000800h
changetwokeydword   12,00020000h,13,00000008h
changetwokeydword   16,00004000h,20,00040000h
changetwokeydword   21,00000020h,23,00000400h
changetwokeydword   25,00000080h,28,00000800h
changetwokeydword   29,00000040h,32,00080000h
endm

change31   macro
changetwokeydword    1,00000004h, 4,00020000h
changetwokeydword    5,00000008h, 8,00004000h
changetwokeydword   10,00000040h,11,00002000h
changetwokeydword   14,00000400h,16,00008000h
changetwokeydword   17,00000080h,20,00000800h
changetwokeydword   21,00000040h,23,00004000h
changethreekeydword 27,00000010h,30,00000080h,31,00000800h
endm

change33   macro
changetwokeydword    1,80000000h, 3,00040000h
changetwokeydword    5,40000000h, 8,00800000h
changetwokeydword   10,00000001h,13,00100000h
changetwokeydword   18,00000004h,19,20000000h
changetwokeydword   21,00080000h,24,00000002h
changethreekeydword 27,00400000h,30,40000000h,32,01000000h
endm

change34   macro
changetwokeydword    2,00000002h, 3,04000000h
changetwokeydword    6,00400000h, 7,10000000h
changetwokeydword   10,01000000h,11,00040000h
changetwokeydword   13,40000000h,16,00800000h
changetwokeydword   20,08000000h,21,08000000h
changetwokeydword   24,04000000h,26,00000004h
changethreekeydword 27,20000000h,29,00080000h,32,02000000h
endm

change35   macro
changetwokeydword    2,04000000h, 5,00800000h
changetwokeydword    8,00000008h,10,02000000h
changetwokeydword   11,04000000h,14,00400000h
changetwokeydword   15,10000000h,18,40000000h
changetwokeydword   19,80000000h,22,80000000h
changetwokeydword   23,00200000h,28,08000000h
changetwokeydword   29,08000000h,31,00100000h
endm

change36   macro
changetwokeydword    1,00200000h, 4,00800000h
changetwokeydword    6,00000001h, 9,00100000h
changetwokeydword   13,00800000h,16,00000008h
changetwokeydword   17,00080000h,20,00000002h
changetwokeydword   23,00400000h,26,40000000h
changethreekeydword 27,80000000h,30,80000000h,31,40000000h
endm

change37   macro
changetwokeydword    2,00008000h, 3,00000080h
changetwokeydword    6,00000800h, 7,00000040h
changetwokeydword    9,00004000h,13,00000010h
changetwokeydword   16,00000080h,17,00000800h
changetwokeydword   20,00020000h,21,00000008h
changetwokeydword   24,00004000h,26,00000040h
changethreekeydword 27,00002000h,30,00000400h,31,00000400h
endm

change38   macro
changetwokeydword    2,00000200h, 5,00000010h
changetwokeydword    8,00000080h, 9,00000004h
changetwokeydword   11,00008000h,14,00000100h
changetwokeydword   18,00000040h,19,00002000h
changetwokeydword   22,00000400h,24,00008000h
changethreekeydword 25,00001000h,30,00080000h,31,00004000h
endm

change39   macro
changetwokeydword    2,00020000h, 3,00008000h
changetwokeydword    6,00000100h,10,00040000h
changetwokeydword   11,00000020h,13,00000400h
changetwokeydword   15,00000080h,17,00001000h
changetwokeydword   22,00080000h,24,00000200h
changethreekeydword 26,00010000h,29,00000800h,31,00000004h
endm

change41   macro
changetwokeydword    1,00040000h, 4,80000000h
changetwokeydword    5,00200000h,10,08000000h
changetwokeydword   11,08000000h,14,04000000h
changetwokeydword   16,00000004h,17,00800000h
changetwokeydword   20,00000008h,22,02000000h
changetwokeydword   23,04000000h,26,00400000h
changethreekeydword 27,10000000h,30,01000000h,31,80000000h
endm

change42   macro
changetwokeydword    1,04000000h, 5,00400000h
changetwokeydword    8,40000000h, 9,80000000h
changetwokeydword   12,80000000h,13,00200000h
changetwokeydword   18,00000001h,21,00100000h
changetwokeydword   25,00800000h,28,00000008h
changetwokeydword   30,02000000h,32,00000002h
endm

change43   macro
changetwokeydword    4,00000004h, 5,20000000h
changetwokeydword    7,00080000h,10,00000002h
changetwokeydword   13,00400000h,16,40000000h
changetwokeydword   18,01000000h,19,00040000h
changetwokeydword   21,40000000h,24,00800000h
changethreekeydword 26,00000001h,29,00100000h,32,04000000h
endm

change44   macro
changetwokeydword    2,00800000h, 6,08000000h
changetwokeydword    7,08000000h,10,04000000h
changetwokeydword   12,00000004h,13,20000000h
changetwokeydword   15,00080000h,18,02000000h
changetwokeydword   19,04000000h,22,00400000h
changetwokeydword   23,10000000h,26,01000000h
changethreekeydword 27,00040000h,29,40000000h,31,00200000h
endm

change45   macro
changetwokeydword    1,00000080h, 3,00001000h
changetwokeydword    8,00080000h,10,00000200h
changetwokeydword   12,00010000h,15,00000800h
changetwokeydword   17,00000004h,19,00008000h
changetwokeydword   22,00000100h,26,00040000h
changethreekeydword 27,00000020h,29,00000400h,32,00008000h
endm

change46   macro
changetwokeydword    4,00010000h, 7,00000800h
changetwokeydword   10,00020000h,11,00000008h
changetwokeydword   14,00004000h,16,00000040h
changetwokeydword   18,00040000h,19,00000020h
changetwokeydword   21,00000400h,23,00000080h
changetwokeydword   26,00000800h,27,00000040h
changetwokeydword   29,00004000h,32,00000200h
endm

change47   macro
changetwokeydword    1,00008000h, 3,00000008h
changetwokeydword    6,00004000h, 8,00000040h
changetwokeydword    9,00002000h,12,00000400h
changetwokeydword   14,00008000h,15,00001000h
changetwokeydword   18,00000800h,19,00000040h
changetwokeydword   21,00004000h,25,00000010h
changethreekeydword 28,00000080h,29,00000004h,32,00020000h
endm

change49   macro
changetwokeydword    2,80000000h, 3,40000000h
changetwokeydword    6,00800000h, 8,00000001h
changetwokeydword   11,00100000h,15,00800000h
changetwokeydword   17,20000000h,19,00080000h
changetwokeydword   22,00000002h,25,00400000h
changethreekeydword 28,40000000h,29,80000000h,31,00040000h
endm

change50   macro
changetwokeydword    4,00400000h, 5,10000000h
changetwokeydword    8,01000000h, 9,00040000h
changetwokeydword   11,40000000h,14,00800000h
changetwokeydword   16,00000001h,18,08000000h
changetwokeydword   19,08000000h,22,04000000h
changetwokeydword   24,00000004h,25,20000000h
changethreekeydword 27,00080000h,30,00000002h,31,04000000h
endm

change51   macro
changetwokeydword    2,00000004h, 3,00800000h
changetwokeydword    6,00000008h, 8,02000000h
changetwokeydword    9,04000000h,12,00400000h
changetwokeydword   13,10000000h,16,01000000h
changetwokeydword   17,80000000h,20,80000000h
changetwokeydword   21,00200000h,26,08000000h
changetwokeydword   27,08000000h,30,04000000h
endm

change52   macro
changetwokeydword    4,00000001h, 7,00100000h
changetwokeydword   11,00800000h,14,00000008h
changetwokeydword   16,02000000h,18,00000002h
changetwokeydword   21,00400000h,24,40000000h
changetwokeydword   25,80000000h,28,80000000h
changetwokeydword   29,00200000h,32,00800000h
endm

change53   macro
changetwokeydword    1,00001000h, 4,00000800h
changetwokeydword    5,00000040h, 7,00004000h
changetwokeydword   11,00000010h,14,00000080h
changetwokeydword   15,00000004h,18,00020000h
changetwokeydword   19,00000008h,22,00004000h
changetwokeydword   24,00000040h,25,00002000h
changethreekeydword 28,00000400h,30,00008000h,31,00000080h
endm

change54   macro
changetwokeydword    2,00010000h, 3,00000010h
changetwokeydword    6,00000080h, 7,00000004h
changetwokeydword    9,00008000h,12,00000100h
changetwokeydword   16,00040000h,17,00002000h
changetwokeydword   20,00000400h,22,00008000h
changethreekeydword 23,00001000h,28,00080000h,30,00000200h
endm

change55   macro
changetwokeydword    1,00000008h, 4,00000100h
changetwokeydword    8,00040000h, 9,00000020h
changetwokeydword   11,00000400h,13,00000080h
changetwokeydword   16,00000800h,20,00080000h
changetwokeydword   22,00000200h,24,00010000h
changethreekeydword 27,00000800h,30,00020000h,31,00008000h
endm

change57   macro
changetwokeydword    1,40000000h, 3,00200000h
changetwokeydword    8,08000000h, 9,08000000h
changetwokeydword   12,04000000h,14,00000004h
changetwokeydword   15,20000000h,18,00000008h
changetwokeydword   20,02000000h,21,04000000h
changetwokeydword   24,00400000h,25,10000000h
changethreekeydword 28,01000000h,29,00040000h,32,80000000h
endm

change58   macro
changetwokeydword    2,00400000h, 3,00400000h
changetwokeydword    6,40000000h, 7,80000000h
changetwokeydword   10,80000000h,11,00200000h
changetwokeydword   16,08000000h,19,00100000h
changetwokeydword   23,00800000h,26,00000008h
changetwokeydword   28,02000000h,29,04000000h
endm

change59   macro
changetwokeydword    1,00800000h, 3,20000000h
changetwokeydword    5,00080000h, 8,00000002h
changetwokeydword   11,00400000h,14,40000000h
changetwokeydword   15,80000000h,17,00040000h
changetwokeydword   19,40000000h,22,00800000h
changethreekeydword 24,00000001h,27,00100000h,32,00000004h
endm

change60   macro
changetwokeydword    2,00000001h, 4,08000000h
changetwokeydword    5,08000000h, 8,04000000h
changetwokeydword   10,00000004h,11,20000000h
changetwokeydword   13,00080000h,16,00000002h
changetwokeydword   17,04000000h,20,00400000h
changetwokeydword   21,10000000h,24,01000000h
changethreekeydword 25,00040000h,27,40000000h,30,00800000h
endm

change61   macro
changetwokeydword    2,00000800h, 6,00080000h
changetwokeydword    8,00000200h,10,00010000h
changetwokeydword   13,00000800h,16,00020000h
changetwokeydword   17,00008000h,20,00000100h
changetwokeydword   24,00040000h,25,00000020h
changethreekeydword 27,00000400h,29,00000080h,31,00001000h
endm

change62   macro
changetwokeydword    1,00000010h, 5,00000800h
changetwokeydword    8,00020000h, 9,00000008h
changetwokeydword   12,00004000h,14,00000040h
changetwokeydword   15,00002000h,17,00000020h
changetwokeydword   19,00000400h,21,00000080h
changetwokeydword   24,00000800h,25,00000040h
changetwokeydword   27,00004000h,32,00010000h
endm

change63   macro
changetwokeydword    2,00000100h, 4,00004000h
changetwokeydword    6,00000040h, 7,00002000h
changetwokeydword   10,00000400h,12,00008000h
changetwokeydword   13,00001000h,17,00000040h
changetwokeydword   19,00004000h,23,00000010h
changetwokeydword   26,00000080h,27,00000004h
changetwokeydword   29,00008000h,31,00000008h
endm
;end include of bdeschg.inc

bit1:   call bit2
        mov ebp,1
        call change

bit2:   call bit3
        mov ebp,2
        call change

bit3:   call bit4
        mov ebp,3
        call change

bit4:   call bit5
        mov ebp,4
        call change

bit5:   call bit6
        mov ebp,5
        call change

bit6:   call bit7
        mov ebp,6
        call change

bit7:   call bit8
        mov ebp,7
        call change

bit8:   call bit9
        mov ebp,8
        call change

bit9:   call bit10
        mov ebp,9
        call change

bit10:  call bit11
        mov ebp,10
        call change

bit11:  call bit12
        mov ebp,11
        call change

bit12:  call bit13
        mov ebp,12
        call change

bit13:  call bit14
        mov ebp,13
        call change

bit14:  call bit15
        mov ebp,14
        call change

bit15:  call bit16
        mov ebp,15
        call change

bit16:  call bit17
        mov ebp,16
        call change

bit17:  call bit18
        mov ebp,17
        call change

bit18:  call bit19
        mov ebp,18
        call change

bit19:  call bit20
        mov ebp,19
        call change

bit20:  call bryd_continue_pro
        .if eax == 0
            mov esp,saveesp
            jmp bryd_not_continue
        .endif
        xor ebx,ebx
        xor ecx,ecx

        call bit21
        mov ebp,20
        call change

bit21:  call bit22
        mov ebp,21
        call change

bit22:  call bit23
        mov ebp,22
        call change

bit23:  call bit24
        mov ebp,23
        call change

bit24:  call bit25
        mov ebp,24
        call change

bit25:  .if bit61_62_63_change == 1
            jmp bit25a
        .endif
        call bit26
        mov ebp,25
        call change

bit26:  call bit27
        mov ebp,26
        call change

bit27:  call bitno51
        mov ebp,27
        call change
        jmp bitno51

bit25a: call bit26a
        change61

bit26a: call bit27a
        change62

bit27a: call bitno51
        change63

; Key bit 51, 60, 54 and 58 are not used in round 16.

bitno51:  ; Compute expected output from round 13 sbox 3,
          ; which equals to output from round 15 sbox 3 in decryption mode.
          ; Part 1.

        change46rest ; update key setup up bit 46, round 13 to 16
        change50rest
        round15box3part1

        call bitno60
        change51

bitno60:  ; Compute round 1 for key and complement key, as well as values for
          ; changing the output, since bit 60, 54 and 58 are used in round 1.

        desround1comp   ; incl. load esi edi  xor ebx,ebx  xor ecx,ecx
        mov esiafter1comp,esi
        desround1
        mov esiafter1,esi
        call bitno54
        change60

        ; Redo round 1, sbox 1.
        mov eax,esiafter1
        mov ebp,redo1box1
        mov edx,esiafter1comp
        xor eax,ebp
        mov ebp,redo1box1comp
        mov esiafter1,eax
        xor edx,ebp
        mov esiafter1comp,edx

bitno54:  ; Compute expected output from round 13 box 3. Part 2.
        round15box3part2
        call bitno58
        change54

        ; Redo round 1, sbox 5.
        mov eax,esiafter1
        mov ebp,redo1box5
        mov edx,esiafter1comp
        xor eax,ebp
        mov ebp,redo1box5comp
        mov esiafter1,eax
        xor edx,ebp
        mov esiafter1comp,edx

bitno58:
        call bitno50
        change58

        ; Redo round 1, sbox 3.
        mov eax,esiafter1
        mov ebp,redo1box3
        mov edx,esiafter1comp
        xor eax,ebp
        mov ebp,redo1box3comp
        mov esiafter1,eax
        xor edx,ebp
        mov esiafter1comp,edx

; Key bit 50, 46, 43 and 52 are not used in round 1.

bitno50:  mov checkoffset,offset round13box3check00
        ; 00 means: bit46 0  bit50 0
        ; We have 4 values for expected output of round 13 box 3
        ; (8 including complement keys)
        ; depending on the state of bit 46 and 50.

testkey:
        xor ebx,ebx
        xor ecx,ecx
        mov esi,esiafter1comp
        mov edi,dword ptr plainpermucomp [4]
        desround2comp  ; compute round 2 and values for changing the output.
        mov ediafter2comp,edi
        mov esi,esiafter1
        mov edi,dword ptr plainpermu [4]
        desround2
        mov ediafter2,edi
        call testkeyfrom3

        mov checkoffset,offset round13box3check10
        call change46testfrom3

        change50new
        mov checkoffset,offset round13box3check11
        redo2box3macro   ; redo round 2, sbox 3   loads esi and edi
        call testkeyfrom3

        mov checkoffset,offset round13box3check01
        call change46testfrom3

        change43
        redo2box1macro  ; redo round 2, sbox 1   loads esi and edi
                        ; also calculates new values for redoing
                        ; box 1, when bit 52 changes
                        ; since box 1 depends on key bit 43 and key bit 52

        call testkeyfrom3

        mov checkoffset,offset round13box3check11
        call change46testfrom3

        change50new
        mov checkoffset,offset round13box3check10
        redo2box3macro
        call testkeyfrom3

        mov checkoffset,offset round13box3check00
        ;call change46testfrom3
        ;ret
; The call and return is commented out, so the test procedure will
; return directly to the key setup change of next bit.

change46testfrom3:
        change46new
        redo2box5macro

testkeyfrom3:
        desmacro310   ; round 3 to 10
        desround 11,esi,edi
        desround12part1

test2:
        ;mov ebp,compcontrol    NB ebp must be set in the DES rounds
        .if ebp != 0
            testbit52changed
        .endif

        mov edi,ediafter2comp
        mov esi,esiafter1comp
        mov compcontrol,24
        jmp testkeyfrom3

;rest of original key
        desround12part2

        desround13
        ;change46rest   moved to round 13
        change50rest
        desround14
        desround 15,esi,edi

        mov ebp,compcontrol
        mov edx,dword ptr cipherpermu [4][ebp]
        .if esi == edx
            desround 16,edi,esi
            mov ebp,compcontrol
            mov eax,dword ptr cipherpermu [ebp]
            .if edi == eax
                call key_from_permu     ;result in esi edi
                mov ebp,compcontrol
                .if ebp != 0
                    xor esi,0FFFFFFFFh
                    xor edi,0FFFFFFFFh
                .endif
                call key_found_low
            .endif
        .endif
        xor ebx,ebx
        xor ecx,ecx
        mov ebp,compcontrol
        jmp test2


permute:               ; local procedure
; registers not preserved
; in:  eax,edx
; out: ebx,ecx
; in:  ebp             ; table address

            xor ebx,ebx
            xor edi,edi
            mov ch,1
            .while ch <= 64
                mov cl,byte ptr ds:[ebp]
                .if cl != 0
                    mov esi,1
                    ror esi,cl
                    .if cl <= 32   ; bit from eax
                        and esi,eax
                    .else
                        and esi,edx
                    .endif
                    .if ! zero?
                        .if ch <= 32
                            add ebx,1
                        .else
                            add edi,1
                        .endif
                    .endif
                .endif
                .if ch < 32
                    add ebx,ebx
                .elseif ch > 32 && ch < 64
                    add edi,edi
                .endif
                inc ebp
                inc ch
            .endw
            mov ecx,edi

            ret

initial_permu:         ; local procedure

        initialpermumacro esi,edi,ecx
        ret

key_from_permu:  ; local procedure

;keyfromsetupdata  dword 1,1,00400000h
;                dword 2,1,20000000h
;                dword 3,2,80000000h

        mov ecx,28
        mov edx,offset keyfromsetupdata
        mov esi,0
        mov edi,0

_14:    mov eax,[edx] ; bit no
        mov ebx,[edx+4] ; key dword no
        mov ebp,[edx+8] ; check data
        and ebp, dword ptr keysetup [ebx*4-4]
        .if !zero?
            or esi,changetable [eax*4-4]
        .endif
        add edx,12
        loop _14
        bswapmacro esi

        mov ecx,28
_15:    mov eax,[edx]   ; bit no
        sub eax,32
        mov ebx,[edx+4] ; key dword no
        mov ebp,[edx+8] ; check data
        and ebp, dword ptr keysetup [ebx*4-4]
        .if !zero?
            or edi,changetable [eax*4-4]
        .endif
        add edx,12
        loop _15
        bswapmacro edi

        ret


key_found_low:         ; local procedure

        mov keyisfound,1

        ; key received in esi edi
        mov dword ptr tempkey,esi
        mov dword ptr tempkey[4],edi

        mov esi,offset tempkey
        mov edi,offset foundkey
        mov ecx,8
_16:    mov al,byte ptr [esi]
        and al,al
        jnp _16a
        ;.if parity?
            xor al,00000001b
        ;.endif
_16a:
        mov byte ptr [edi],al
        add esi,1
        add edi,1
        loop _16
        push offset foundkey
        call bryd_key_found_pro
        add esp,4
        xor ebx,ebx
        xor ecx,ecx

        ret


p1desinit:               ; procedure

desinit_frame equ 24

; ptr key
; 0    [in]

        push esi
        push edi
        push ebp
        push ebx
        push ecx

        mov ebp, dword ptr [esp][desinit_frame]

        ; set up key
        mov ebx, ebp
        mov eax,dword ptr [ebx]
        mov edx,dword ptr [ebx][4]
        bswapmacro eax
        bswapmacro edx

        mov ebp, offset keypermu1
        call permute

        mov eax,ebx
        mov edx,ecx
        mov L1,1
        mov esi,offset keysetup
        .while L1 <= 16
            push esi

            mov ebp,offset keypermu2
            call permute

            .if L1 > 2 && L1 != 9 && L1 != 16
                mov eax,ebx
                mov edx,ecx

                mov ebp, offset keypermu2
                call permute

            .endif
            mov eax,ebx
            mov edx,ecx

            mov ebp, offset keypermu3
            call permute

            pop esi
            ;ebx 0 1 2 3
            ;ecx 4 5 6 7
                          ; ebx       ecx
            xchg bl,bh    ; 0 1 3 2   4 5 6 7
            rol ebx,8     ; 1 3 2 0   4 5 6 7
            xchg bh,cl    ; 1 3 7 0   4 5 6 2
            ror ecx,8     ; 1 3 7 0   2 4 5 6
            rol bx,8      ; 1 3 0 7   2 4 5 6
            ror cx,8      ; 1 3 0 7   2 4 6 5
            xchg bh,cl    ; 1 3 5 7   2 4 6 0

                          ; 1 3 5 7   2 4 6 0

            ;       222222  444444  666666  888888
            ;           333333  555555  777777  111111

            ror ecx,4

            mov dword ptr [esi],ebx
            mov dword ptr [esi][4],ecx

            rol ecx,4  ; not needed I suppose

            add esi,8
            inc L1

        .endw

        ; set up S-boxes
        push ebp
        mov esi,offset sboxdata
        mov edi,0
        mov ebp,0
        .while ebp < 8*4
            mov ah,1
            .while ah <= 64
                mov ebx,0
                test ah,1
                jz _17
                lodsb
    _17:        shl al,1
                .if carry?
                    mov cl,spermu [ebp]
                    mov edx,1
                    ror edx,cl
                    add ebx,edx
                .endif
                inc ebp
                test ebp,00000011b
                jnz _17
                sub ebp,4
                rol ebx,3
                mov dword ptr sbox1 [edi],ebx
                inc ah
                add edi,4
            .endw
            add ebp,4
        .endw
        pop ebp

        pop ecx
        pop ebx
        pop ebp
        pop edi
        pop esi

        ret

shiftleft_edx_eax:     ; local procedure
        shl edx,1
        shl eax,1
        .if carry?
            or edx,1
        .endif
        ret


p1desencrypt:            ; procedure

desencrypt_frame equ 24

; ptr plain    ptr ciphertext
; 0    [in]    4        [out]

        push esi
        push edi
        push ebp
        push ebx
        push ecx

        mov eax,dword ptr [esp][desencrypt_frame]
        mov esi, dword ptr [eax]
        mov edi, dword ptr [eax+4]

        mov keysetupoffset,0
        mov ebp,0

        initialpermumacro esi,edi,ecx

        xor ebx,ebx
        xor ecx,ecx

_18:    mov eax,dword ptr keysetup[ebp]
        xor eax,edi
        mov edx,dword ptr keysetup[ebp+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor esi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        mov bl,ah
        xor esi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr sbox4 [eax]
        xor esi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor esi,ebp
        mov ebp,keysetupoffset
        mov eax,esi
        add ebp,8
        mov keysetupoffset,ebp
        mov esi,edi
        mov edi,eax
        cmp ebp, 128
        jb _18

        finalpermumacro esi,edi,ecx

        mov eax,dword ptr [esp][desencrypt_frame+4]
        mov dword ptr [eax], edi
        mov dword ptr [eax+4] , esi

        pop ecx
        pop ebx
        pop ebp
        pop edi
        pop esi

        ret

p1desdecrypt:            ; procedure

desdecrypt_frame equ 24

; ptr cipher   ptr plain
; 0    [in]    4        [out]

        push esi
        push edi
        push ebp
        push ebx
        push ecx

        mov eax,dword ptr [esp][desdecrypt_frame]
        mov esi, dword ptr [eax]
        mov edi, dword ptr [eax+4]

        mov keysetupoffset,120
        mov ebp,120

        initialpermumacro esi,edi,ecx

        xor ebx,ebx
        xor ecx,ecx

_19:    mov eax,dword ptr keysetup [ebp]
        xor eax,edi
        mov edx,dword ptr keysetup [ebp+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor esi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        mov bl,ah
        xor esi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr sbox2 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr sbox4 [eax]
        xor esi,ebp
        mov ebp,dword ptr sbox5 [edx]
        xor esi,ebp
        mov ebp,keysetupoffset
        mov eax,esi
        sub ebp,8
        mov keysetupoffset,ebp
        mov esi,edi
        mov edi,eax
        cmp ebp,0
        jnl _19

        finalpermumacro esi,edi,ecx

        mov eax,dword ptr [esp][desdecrypt_frame+4]
        mov dword ptr [eax], edi
        mov dword ptr [eax+4] , esi

        pop ecx
        pop ebx
        pop ebp
        pop edi
        pop esi

        ret

p1key_byte_to_hex:       ; procedure
        mov add_zero,0
        jmp key_byte_to_hex_1
p1c_key_byte_to_hex:
        mov add_zero,1
key_byte_to_hex_1:


; ptr key    ptr hexkey    dword no_of_bytes
; 0  [in]    4    [out]    8              in

; returns  eax  0: parity ok    1: parity adjusted

        push esi
        push edi
        push ebp
        push ebx
        push ecx       ; pushes adds 20 to frame

        mov esi,[esp][24+0]
        mov edi,[esp][24+4]
        mov ecx,[esp][24+8]
        mov edx,0
_20:    mov al,byte ptr [esi]
        and al,al
        jnp _20a
        ;.if parity?
            mov edx,1
            xor al,00000001b
        ;.endif
_20a:
        mov ah,al
        and ah,00001111b
        and al,11110000b
        shr al,4
        .if ah > 9
            add ah,55
        .else
            add ah,48
        .endif
        .if al > 9
            add al,55
        .else
            add al,48
        .endif
        mov word ptr [edi],ax
        add esi,1
        add edi,2
        loop _20

        .if add_zero == 1
            mov byte ptr [edi],0
        .endif

        mov eax,edx

        pop ecx
        pop ebx
        pop ebp
        pop edi
        pop esi
        ret
end




