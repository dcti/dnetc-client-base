; BDESMAC.INC

; Part of BrydDES Key Search Library version 1.01.
; Date: January 17, 1998.
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
        xor ecx,20h
        xor esi,ebp
        mov bl,ah
        mov edi,dword ptr sbox7 [ecx]
        and eax,0FFh
        xor ebp,edi
        mov redo16box7,ebp
        mov cl,dh
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
        xor ecx,20h
        xor esi,ebp
        mov bl,ah
        and eax,0FFh
        mov edi,dword ptr sbox7 [ecx]
        xor ecx,ecx
        xor ebp,edi
        mov cl,dh
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
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor reg1,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor reg1,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor reg1,ebp
        mov ebp,dword ptr sbox7 [ecx]
        mov bl,ah
        xor reg1,ebp
        mov cl,dh
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

        xor ebx,10h
        xor esi,ebp

        mov edi,dword ptr sbox7 [ecx]
        mov cl,dh

        xor esi,edi
        mov edi,dword ptr sbox1 [ebx]

        xor edi,ebp
        xor ebx,ebx

        mov redo1box1,edi
        mov bl,ah

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

        xor ebx,10h
        xor esi,ebp

        mov edi,dword ptr sbox7 [ecx]
        mov cl,dh

        xor esi,edi
        mov edi,dword ptr sbox1 [ebx]

        xor edi,ebp
        xor ebx,ebx

        mov redo1box1comp,edi
        mov bl,ah

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
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]

        xor edi,ebp
        mov round2box1ptrcomp,ebx

        mov undo2box1comp,ebp
        mov ebp,dword ptr tbox1 [ebx]

        mov redo2box1comp,ebp
        mov ebp,dword ptr sbox7 [ecx]

        mov bl,ah
        xor edi,ebp
        mov cl,dh
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
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor edi,ebp

        mov round2box1ptr,ebx
        mov undo2box1,ebp
        mov ebp,dword ptr tbox1 [ebx]
        mov redo2box1,ebp

        mov ebp,dword ptr sbox7 [ecx]
        mov bl,ah
        xor edi,ebp
        mov cl,dh
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
        mov cl,ah

        rol edx,4

        mov saveeax,eax
        mov bl,dl

        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]

        xor edi,ebp
        mov ebp,dword ptr sbox1 [ebx]

        xor edi,ebp
        mov cl,dh

        mov saveedx,edx
        mov bl,ah

        mov ebp,dword ptr sbox7 [ecx]
        mov cl,byte ptr saveedx [2]

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
        mov cl,byte ptr saveedx [3]

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
        mov bl,byte ptr saveeax
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
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov saveedx,edx
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]

        xor esi,ebp
        mov ebp,dword ptr sbox6 [ecx]

        mov bl,ah
        and eax,0FFh

        xor esi,ebp
        mov cl,dh

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


_51:    mov bl,byte ptr saveedx [2]
        mov cl,byte ptr saveedx [1]

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
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]

        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]

        mov cl,dh
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


_52:    mov bl,ah
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

        mov eax,dword ptr keysetup [(4-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(4-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        mov bl,ah
        xor edi,ebp
        mov cl,dh
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

        mov eax,dword ptr keysetup [(6-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(6-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        mov bl,ah
        xor edi,ebp
        mov cl,dh
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
        mov ebp,dword ptr round7box1 [ebx]
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

        mov eax,dword ptr keysetup [(8-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(8-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        mov bl,ah
        xor edi,ebp
        mov cl,dh
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
        mov ebp,dword ptr round9box1 [ebx]
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

        mov eax,dword ptr keysetup [(10-1)*8]
        xor eax,esi
        mov edx,dword ptr keysetup [(10-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        mov bl,ah
        xor edi,ebp
        mov cl,dh
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
        mov cl,ah
        rol edx,4
        mov saveeax,eax
        mov bl,dl
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        mov ebp,dword ptr round12box1 [ebx]
        xor edi,ebp
        mov cl,dh
        mov saveedx,edx
        mov bl,ah
        mov ebp,dword ptr sbox7 [ecx]
        mov cl,byte ptr saveedx [2]
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
        mov cl,byte ptr saveedx [3]
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

        mov bl,byte ptr saveeax
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
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov saveedx,edx
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr sbox6 [ecx]
        mov bl,ah
        and eax,0FFh
        xor esi,ebp
        mov cl,dh
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

_53:    mov bl,byte ptr saveedx [2]
        mov cl,byte ptr saveedx [1]

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
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        xor ebx,08h
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor ebx,ebx
        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        mov cl,dh
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

_54:    mov bl,ah
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
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        xor esi,ebp
        mov bl,dl
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
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr sbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr sbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr sbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr sbox7 [ecx]
        mov bl,ah
        xor edi,ebp
        mov cl,dh
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

