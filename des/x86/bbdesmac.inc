; BDESMAC.INC

; Part of BrydDES Key Search Library version 1.01.
; Date: January 17, 1998.
; Copyright Svend Olaf Mikkelsen, 1995, 1997, 1998.


; Please read README.TXT.

; Written for Microsoft Macro Assembler, version 6.00B.
; Modified, so it also seems to assemble correct with TASM 5.0.



Bchangetwokeydword macro dwordno1,value1,dwordno2,value2

        mov eax,dword ptr Bkeysetup [(dwordno1-1)*4]
        mov edx,dword ptr Bkeysetup [(dwordno2-1)*4]
        xor eax,value1
        xor edx,value2
        mov dword ptr Bkeysetup [(dwordno1-1)*4],eax
        mov dword ptr Bkeysetup [(dwordno2-1)*4],edx

        endm

Bchangethreekeydword macro dwordno1,value1,dwordno2,value2,dwordno3,value3

        mov eax,dword ptr Bkeysetup [(dwordno1-1)*4]
        mov ebp,dword ptr Bkeysetup [(dwordno2-1)*4]
        xor eax,value1
        mov edx,dword ptr Bkeysetup [(dwordno3-1)*4]
        xor ebp,value2
        mov dword ptr Bkeysetup [(dwordno1-1)*4],eax
        xor edx,value3
        mov dword ptr Bkeysetup [(dwordno2-1)*4],ebp
        mov dword ptr Bkeysetup [(dwordno3-1)*4],edx

        endm


Bround15box3part1 macro
        mov esi,dword ptr Bcipherpermu
        mov edi,dword ptr Bcipherpermu [4]
        mov eax,dword ptr Bkeysetup [(16-1)*8]   ; round 16
        xor eax,edi
        mov edx,dword ptr Bkeysetup [(16-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor esi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        xor ecx,20h
        xor esi,ebp
        mov bl,ah
        mov edi,dword ptr Bsbox7 [ecx]
        and eax,0FFh
        xor ebp,edi
        mov Bredo16box7,ebp
        mov cl,dh
        mov edi,dword ptr Bsbox3 [ecx]
        xor esi,edi
        and edx,0FFh
        mov edi,dword ptr Bsbox2 [ebx]
        xor ebx,04h
        xor esi,edi
        mov ecx,dword ptr Bsbox4 [eax]
        xor esi,ecx
        mov ebx,dword ptr Bsbox2 [ebx]
        mov ecx,dword ptr Bsbox5 [edx]
        xor ebx,edi
        xor esi,ecx
        mov Bredo16box2,ebx
        xor ebx,ebx
        xor ecx,ecx
        mov Bsaveesiround16,esi

        ; Complement key part:

        mov esi,dword ptr Bcipherpermucomp
        mov edi,dword ptr Bcipherpermucomp [4]

        mov eax,dword ptr Bkeysetup [(16-1)*8]
        xor eax,edi
        mov edx,dword ptr Bkeysetup [(16-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor esi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        xor ecx,20h
        xor esi,ebp
        mov bl,ah
        and eax,0FFh
        mov edi,dword ptr Bsbox7 [ecx]
        xor ecx,ecx
        xor ebp,edi
        mov cl,dh
        mov edi,dword ptr Bsbox3[ecx]
        xor esi,edi
        and edx,0FFh
        mov edi,dword ptr Bsbox2 [ebx]
        xor ebx,04h
        xor esi,edi
        mov ecx,dword ptr Bsbox4 [eax]
        xor esi,ecx
        mov ebx,dword ptr Bsbox2 [ebx]
        mov ecx,dword ptr Bsbox5 [edx]
        xor ebx,edi
        xor esi,ecx
        mov Bredo16box2comp,ebx
        mov Bredo16box7comp,ebp
        mov Bsaveesiround16comp,esi
        xor ebx,ebx
        xor ecx,ecx

        endm


Bround15box3part2 macro
        ; lines marked ;2 belongs to the not complement key part

        mov ebx,Bredo16box2comp
        mov ebp,Bredo16box7comp
        mov esi,Bsaveesiround16comp

        ; then we make four versions of box 3 in round 15

        and ebx,00FC00000h   ;  redo16box2comp

        shr ebx,20
        and ebp,00FC00000h   ; redo16box7comp

        shr ebp,20
        mov edx,dword ptr Bkeysetup [(15-1)*8+4]

        xor edx,esi
        and esi,08200401h

        and edx,00FC00000h
        mov Bround15box5checkcomp,esi

        shr edx,20
        mov ecx,Bredo16box2                   ; 2

        mov edi,dword ptr Bcipherpermucompbox3bits;  [4]
        and ecx,00FC00000h                   ; 2

        mov eax,dword ptr Bsbox3 [edx]
        xor edx,ebx

        shr ecx,20
        xor eax,edi

        mov esi,dword ptr Bsbox3 [edx]
        xor edx,ebp

        xor esi,edi
        mov Bround13box3check00comp,eax

        mov eax,dword ptr Bsbox3 [edx]
        xor edx,ebx

        mov Bround13box3check01comp,esi
        xor eax,edi

        mov ebx,dword ptr Bsbox3 [edx]
        mov edx,dword ptr Bkeysetup [(15-1)*8+4]

        mov esi,Bsaveesiround16  ; 2
        xor ebx,edi

        xor edx,esi             ; 2
        mov Bround13box3check11comp,eax

        and edx,00FC00000h      ;2
        mov Bround13box3check10comp,ebx


        ; not complement key:

        shr edx,20
        mov eax,esi

        and eax,08200401h
        mov ebp,edx

        xor ebp,ecx
        mov edi,Bredo16box7

        mov Bround15box5check,eax
        and edi,00FC00000h

        shr edi,20
        mov eax,dword ptr Bcipherpermubox3bits

        mov ebx,dword ptr Bsbox3 [ebp]   ;01
        xor ebp,edi

        xor ebx,eax
        mov esi,dword ptr Bsbox3 [edx]   ;00

        mov Bround13box3check01,ebx
        xor esi,eax

        mov ebx,dword ptr Bsbox3 [ebp]   ;11
        xor ebp,ecx

        xor ebx,eax
        xor ecx,ecx

        mov Bround13box3check11,ebx
        mov edx,dword ptr Bsbox3 [ebp]   ;10

        mov Bround13box3check00,esi
        xor edx,eax

        mov Bround13box3check10,edx
        xor ebx,ebx

        endm


Bchange46new   macro
;Bchangetwokeydword  4,00100000h, 7,00000800h
Bchangethreekeydword 7,00000800h,10,00020000h,11,00000008h
Bchangetwokeydword 14,00004000h,16,00000040h
Bchangetwokeydword 18,00040000h,19,00000020h
Bchangetwokeydword 21,00000400h,23,00000080h
;Bchangetwokeydword 26,00000800h,27,00000040h
;Bchangetwokeydword 29,00004000h,32,00000200h
endm

Bchange46rest   macro
        mov eax,Bis14b
        mov edx,dword ptr Bkeysetup[24]
        and edx,00000800h
        .if eax != edx
            xor eax,00000800h
            mov Bis14b,eax
            Bchangetwokeydword 26,00000800h,27,00000040h
            Bchangetwokeydword 29,00004000h,32,00000200h
        .endif
endm


Bchange50new   macro
;Bchangetwokeydword  4,00400000h, 5,10000000h
Bchangethreekeydword  5,10000000h,8,01000000h, 9,00040000h
Bchangetwokeydword 11,40000000h,14,00800000h
Bchangetwokeydword 16,00000001h,18,08000000h
Bchangetwokeydword 19,08000000h,22,04000000h
Bchangetwokeydword 24,00000004h,25,20000000h  ; round 13 box 2
;Bchangethreekeydword 27,00080000h,30,00000002h,31,04000000h
endm

Bchange50rest   macro
        mov eax,Bis18b
        mov edx,dword ptr Bkeysetup[16]
        and edx,10000000h
        .if eax != edx
            xor eax,10000000h
            mov Bis18b,eax
            Bchangethreekeydword 27,00080000h,30,00000002h,31,04000000h
        .endif
endm


Bchange52new   macro
;Bchangetwokeydword    4,00000001h, 7,00100000h
Bchangethreekeydword   7,00100000h, 14,00000008h, 11,00800000h   ;NB changed
Bchangetwokeydword   16,02000000h,18,00000002h
Bchangethreekeydword   21,00400000h,24,40000000h,25,80000000h
;Bchangetwokeydword   25,80000000h,28,80000000h
;Bchangetwokeydword   29,00200000h,32,00800000h
endm
; rest changed before round 14

;NB NB New, january 1998.

Bchange52rest   macro
        mov eax,Bis52b
        mov edx,dword ptr Bkeysetup[24]
        and edx,00100000h
        .if eax != edx
            xor eax,00100000h
            mov Bis52b,eax
            Bchangethreekeydword 28,80000000h,29,00200000h,32,00800000h
        .endif
endm



Bredo2box3macro macro

        mov esi,Bediafter2comp
        mov ebp,Bredo2box3comp
        mov edi,Bediafter2
        xor esi,ebp
        mov ebp,Bredo2box3
        mov Bediafter2comp,esi
        xor edi,ebp
        mov esi,Besiafter1
        mov Bediafter2,edi

        endm


Bredo2box5macro  macro

        mov esi,Bediafter2comp
        mov ebp,Bredo2box5comp
        mov edi,Bediafter2
        xor esi,ebp
        mov ebp,Bredo2box5
        mov Bediafter2comp,esi
        xor edi,ebp
        mov esi,Besiafter1
        mov Bediafter2,edi

        endm


Binitialpermumacro     macro  reg1, reg2, reg3
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

Bfinalpermumacro  macro reg1,reg2,reg3

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



Bdesround    macro Broundno,reg1,reg2

        mov eax,dword ptr Bkeysetup [(Broundno-1)*8]
        xor eax,reg2
        mov edx,dword ptr Bkeysetup [((Broundno-1)*8+4)]
        xor edx,reg2
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor reg1,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor reg1,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor reg1,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        mov bl,ah
        xor reg1,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor reg1,ebp
        mov ebp,dword ptr Bsbox3 [ecx]
        xor reg1,ebp
        mov ebp,dword ptr Bsbox4 [eax]
        xor reg1,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor reg1,ebp
        endm

Bdesround1    macro
        mov esi,dword ptr Bplainpermu
        mov edi,dword ptr Bplainpermu [4]

        mov eax,dword ptr Bkeysetup [(1-1)*8]
        xor eax,edi
        mov edx,dword ptr Bkeysetup [(1-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor esi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]

        xor ebx,10h
        xor esi,ebp

        mov edi,dword ptr Bsbox7 [ecx]
        mov cl,dh

        xor esi,edi
        mov edi,dword ptr Bsbox1 [ebx]

        xor edi,ebp
        xor ebx,ebx

        mov Bredo1box1,edi
        mov bl,ah

        and eax,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]

        xor esi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]

        xor esi,ebp
        xor ecx,04h

        and edx,0FFh
        mov edi,dword ptr Bsbox4 [eax]

        xor esi,edi
        mov edi,dword ptr Bsbox3 [ecx]

        mov eax,dword ptr Bsbox5 [edx]
        xor edx,10h

        xor edi,ebp
        xor ecx,ecx

        mov Bredo1box3,edi
        mov edx,dword ptr Bsbox5 [edx]

        xor esi,eax
        xor edx,eax

        mov Bredo1box5,edx

        endm


Bdesround1comp    macro

        mov esi,dword ptr Bplainpermucomp
        mov edi,dword ptr Bplainpermucomp [4]

        mov eax,dword ptr Bkeysetup [(1-1)*8]
        xor eax,edi
        mov edx,dword ptr Bkeysetup [(1-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor esi,ebp

        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor esi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]

        xor ebx,10h
        xor esi,ebp

        mov edi,dword ptr Bsbox7 [ecx]
        mov cl,dh

        xor esi,edi
        mov edi,dword ptr Bsbox1 [ebx]

        xor edi,ebp
        xor ebx,ebx

        mov Bredo1box1comp,edi
        mov bl,ah

        and eax,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]

        xor esi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]

        xor esi,ebp
        xor ecx,04h

        and edx,0FFh
        mov edi,dword ptr Bsbox4 [eax]

        xor esi,edi
        mov edi,dword ptr Bsbox3 [ecx]

        mov eax,dword ptr Bsbox5 [edx]
        xor edx,10h

        xor edi,ebp
        xor ecx,ecx

        mov Bredo1box3comp,edi
        mov edx,dword ptr Bsbox5 [edx]

        xor esi,eax
        xor edx,eax

        mov Bredo1box5comp,edx

        endm


Bdesround2comp    macro
        mov eax,dword ptr Bkeysetup [(2-1)*8]
        xor eax,esi
        mov edx,dword ptr Bkeysetup [(2-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]

        xor edi,ebp
        mov Bround2box1ptrcomp,ebx

        mov Bundo2box1comp,ebp
        mov ebp,dword ptr Btbox1 [ebx]

        mov Bredo2box1comp,ebp
        mov ebp,dword ptr Bsbox7 [ecx]

        mov bl,ah
        xor edi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]
        xor edi,ebp

        mov ebp,dword ptr Btbox3 [ecx]
        mov Bredo2box3comp,ebp

        mov ebp,dword ptr Bsbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor edi,ebp

        mov ebp,dword ptr Btbox5 [edx]
        mov Bredo2box5comp,ebp

        endm

Bdesround2   macro
        mov eax,dword ptr Bkeysetup [(2-1)*8]
        xor eax,esi
        mov edx,dword ptr Bkeysetup [(2-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor edi,ebp

        mov Bround2box1ptr,ebx
        mov Bundo2box1,ebp
        mov ebp,dword ptr Btbox1 [ebx]
        mov Bredo2box1,ebp

        mov ebp,dword ptr Bsbox7 [ecx]
        mov bl,ah
        xor edi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]
        xor edi,ebp

        mov ebp,dword ptr Btbox3 [ecx]
        mov Bredo2box3,ebp

        mov ebp,dword ptr Bsbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor edi,ebp

        mov ebp,dword ptr Btbox5 [edx]
        mov Bredo2box5,ebp

        endm


Bredo2box1macro macro      ; redo round 2 box 1

        mov ebx,Bround2box1ptrcomp
        mov edx,Bround2box1ptr

        xor ebx,40h
        mov eax,Bundo2box1comp

        xor edx,40h
        mov esi,Bediafter2comp

        xor esi,eax
        mov ebp,dword ptr Bsbox1 [ebx]

        xor esi,ebp
        mov ebp,dword ptr Btbox1 [ebx]

        mov edi,Bediafter2
        mov eax,Bundo2box1

        mov Bredo2box1comp,ebp
        xor edi,eax

        xor ebx,ebx
        mov ebp,dword ptr Bsbox1 [edx]

        xor edi,ebp
        mov ebp,dword ptr Btbox1 [edx]

        mov Bredo2box1,ebp
        mov Bediafter2comp,esi
        mov Bediafter2,edi
        mov esi,Besiafter1

        endm


Bdesround12part1   macro
        mov eax,dword ptr Bkeysetup [(12-1)*8]

        xor eax,esi
        mov edx,dword ptr Bkeysetup [(12-1)*8+4]

        xor edx,esi
        and eax,0FCFCFCFCh

        and edx,0CFCFCFCFh
        mov cl,ah

        rol edx,4

        mov Bsaveeax,eax
        mov bl,dl

        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]

        xor edi,ebp
        mov ebp,dword ptr Bsbox1 [ebx]

        xor edi,ebp
        mov cl,dh

        mov Bsaveedx,edx
        mov bl,ah

        mov ebp,dword ptr Bsbox7 [ecx]
        mov cl,byte ptr Bsaveedx [2]

        xor edi,ebp
        and eax,0FFh

        mov ebp,dword ptr Bsbox5 [ecx]
        mov edx,dword ptr Bkeysetup [(12-1+1)*8+4]

        xor edi,ebp
        mov ebp,dword ptr Bsbox2 [ebx]

        xor edi,ebp
        mov ebp,dword ptr Bsbox4 [eax]

        xor edi,ebp
        mov ebp,Bcompcontrol

        xor edx,edi
        mov eax,Bcheckoffset

        and edx,0FC00000h
        mov cl,byte ptr Bsaveedx [3]

        shr edx,20
        mov eax,[eax][ebp]

        mov Bsaveregister,esi
        and esi,20080820h

        xor eax,esi
        mov edx,dword ptr Bsbox3 [edx]

        cmp edx,eax
        je Bdesround12rest
        ; ebp Bcompcontrol

        endm

Bdesround12part2 macro

Bdesround12rest:
        ;mov cl,byte ptr Bsaveedx [3]
        mov bl,byte ptr Bsaveeax
        mov ebp,dword ptr Bsbox3 [ecx]

        xor edi,ebp
        mov ebp,dword ptr Bsbox8 [ebx]

        xor edi,ebp
        mov esi,Bsaveregister

        endm



Bdesround13   macro
        mov eax,dword ptr Bkeysetup [(13-1)*8]
        xor eax,edi
        mov edx,dword ptr Bkeysetup [(13-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov Bsaveedx,edx
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]

        xor esi,ebp
        mov ebp,dword ptr Bsbox6 [ecx]

        mov bl,ah
        and eax,0FFh

        xor esi,ebp
        mov cl,dh

        mov ebp,dword ptr Bsbox4 [eax]
        mov edx,dword ptr Bkeysetup [(13-1+1)*8+4]

        xor esi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]

        xor esi,ebp
        mov ebp,dword ptr Bsbox2 [ebx]

        xor esi,ebp
        mov ebp,Bcompcontrol

        xor edx,esi
        mov Bsaveregister,edi

        shr edx,12
        and edi,08200401h

        mov ebp,Bround15box5check [ebp]
        and edx,0FCh

        xor ebp,edi
        mov edx,dword ptr Bsbox5 [edx]     ; agi

        cmp edx,ebp
        je B_51

        mov ebp,Bcompcontrol
        jmp Btest2


B_51:    mov bl,byte ptr Bsaveedx [2]
        mov cl,byte ptr Bsaveedx [1]

        mov eax,Bis14b
        mov edx,dword ptr Bkeysetup [24]
        and edx,00000800h
        .if eax != edx
            xor eax,00000800h
            mov Bis14b,eax
            Bchangetwokeydword 26,00000800h,27,00000040h
            Bchangetwokeydword 29,00004000h,32,00000200h
            xor ecx,80h
        .endif

        mov edi,Bsaveregister
        mov ebp,dword ptr Bsbox5 [ebx]  ; agi

        xor esi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        xor ecx,ecx

        xor esi,ebp

        endm

Bdesround14   macro
        mov eax,dword ptr Bkeysetup [(14-1)*8]
        xor eax,esi
        mov edx,dword ptr Bkeysetup [(14-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]

        xor edi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]

        mov cl,dh
        and edx,0FFh

        xor edi,ebp
        mov ebp,Bcompcontrol
        mov ebp,Bround15box2check [ebp]

        mov Bsaveregister,esi
        mov edx,dword ptr Bsbox5 [edx]

        xor edi,edx
        mov edx,dword ptr Bsbox3 [ecx]

        xor edi,edx
        mov edx,dword ptr Bkeysetup [(14-1+1)*8]

        xor edx,edi
        and esi,00420082h

        shr edx,26
        xor ebp,esi

        mov edx,dword ptr Bsbox2 [edx*4]
        mov esi,1 ; means nothing, will not delete it and change alignment now

        cmp edx,ebp
        je B_52
        mov ebp,Bcompcontrol
        jmp Btest2


B_52:    mov bl,ah
        and eax,0FFh
        mov esi,Bsaveregister
        mov ebp,dword ptr Bsbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr Bsbox2 [ebx]
        xor edi,ebp

        endm


Bdesmacro310  macro

        ; round 3 to 10
        local Broundno
        xor ebx,ebx   ; Might enhance Pentium Pro speed.
            irp Broundno,<3,5,7,9>
                Bdesround Broundno,esi,edi
                Bdesround (Broundno+1),edi,esi
            endm
        endm


Bch52round3to12a      macro
        xor ecx,ecx   ; Might enhance Pentium Pro speed.
        mov eax,dword ptr Bkeysetup [(3-1)*8]
        xor eax,edi
        mov edx,dword ptr Bkeysetup [(3-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor esi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        mov bl,ah
        xor esi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox4 [eax]
        xor esi,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor esi,ebp

        mov eax,dword ptr Bkeysetup [(4-1)*8]
        xor eax,esi
        mov edx,dword ptr Bkeysetup [(4-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        mov bl,ah
        xor edi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]
        xor edi,ebp
        mov ebp,dword ptr Bround4box4 [eax]
        xor edi,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor edi,ebp

        mov eax,dword ptr Bkeysetup [(5-1)*8]
        xor eax,edi
        mov edx,dword ptr Bkeysetup [(5-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor esi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        mov bl,ah
        xor esi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox4 [eax]
        xor esi,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor esi,ebp

        mov eax,dword ptr Bkeysetup [(6-1)*8]
        xor eax,esi
        mov edx,dword ptr Bkeysetup [(6-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        mov bl,ah
        xor edi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]
        xor edi,ebp
        mov ebp,dword ptr Bround6box4 [eax]
        xor edi,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor edi,ebp

        mov eax,dword ptr Bkeysetup [(7-1)*8]
        xor eax,edi
        mov edx,dword ptr Bkeysetup [(7-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor esi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bround7box1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        mov bl,ah
        xor esi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox4 [eax]
        xor esi,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor esi,ebp

        mov eax,dword ptr Bkeysetup [(8-1)*8]
        xor eax,esi
        mov edx,dword ptr Bkeysetup [(8-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        mov bl,ah
        xor edi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr Bround8box3 [ecx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor edi,ebp

        mov eax,dword ptr Bkeysetup [(9-1)*8]
        xor eax,edi
        mov edx,dword ptr Bkeysetup [(9-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor esi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bround9box1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        mov bl,ah
        xor esi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox4 [eax]
        xor esi,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor esi,ebp

        mov eax,dword ptr Bkeysetup [(10-1)*8]
        xor eax,esi
        mov edx,dword ptr Bkeysetup [(10-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        mov bl,ah
        xor edi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor edi,ebp

        mov eax,dword ptr Bkeysetup [(11-1)*8]
        xor eax,edi
        mov edx,dword ptr Bkeysetup [(11-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor esi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        mov bl,ah
        xor esi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr Bround11box4 [eax]
        xor esi,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor esi,ebp

        mov eax,dword ptr Bkeysetup [(12-1)*8]
        xor eax,esi
        mov edx,dword ptr Bkeysetup [(12-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov cl,ah
        rol edx,4
        mov Bsaveeax,eax
        mov bl,dl
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor edi,ebp
        mov ebp,dword ptr Bround12box1 [ebx]
        xor edi,ebp
        mov cl,dh
        mov Bsaveedx,edx
        mov bl,ah
        mov ebp,dword ptr Bsbox7 [ecx]
        mov cl,byte ptr Bsaveedx [2]
        xor edi,ebp
        and eax,0FFh
        mov ebp,dword ptr Bsbox5 [ecx]
        mov edx,dword ptr Bkeysetup [(12-1+1)*8+4]
        xor edi,ebp
        mov ebp,dword ptr Bsbox2 [ebx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox4 [eax]
        xor edi,ebp
        mov ebp,Bcompcontrol
        xor edx,edi
        mov eax,Bcheckoffset
        and edx,0FC00000h
        mov cl,byte ptr Bsaveedx [3]
        shr edx,20
        mov eax,[eax][ebp]
        mov Bsaveregister,esi
        and esi,20080820h
        xor eax,esi
        mov edx,dword ptr Bsbox3 [edx]
        cmp edx,eax
        je Bch52round12rest
        ; ebp = compcontrol

        endm



Bch52round12b  macro

Bch52round12rest:

        mov bl,byte ptr Bsaveeax
        mov ebp,dword ptr Bsbox3 [ecx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox8 [ebx]
        xor edi,ebp
        mov esi,Bsaveregister

        endm


Bch52round13   macro
        mov eax,dword ptr Bkeysetup [(13-1)*8]
        xor eax,edi
        mov edx,dword ptr Bkeysetup [(13-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor esi,ebp
        shr eax,16
        mov Bsaveedx,edx
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox6 [ecx]
        mov bl,ah
        and eax,0FFh
        xor esi,ebp
        mov cl,dh
        mov ebp,dword ptr Bsbox4 [eax]
        xor ebx,80h
        xor esi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox2 [ebx]
        mov edx,dword ptr Bkeysetup [(13-1+1)*8+4]
        xor ebx,ebx
        xor esi,ebp
        mov ebp,Bcompcontrol
        xor edx,esi
        mov Bsaveregister,edi
        shr edx,12
        and edi,08200401h
        mov ebp,Bround15box5check [ebp]
        and edx,0FCh
        xor ebp,edi
        mov edx,dword ptr Bsbox5 [edx]     ; agi
        cmp edx,ebp
        je B_53
        mov ebp,Bcompcontrol
        jmp Btest52changed2

B_53:    mov bl,byte ptr Bsaveedx [2]
        mov cl,byte ptr Bsaveedx [1]

        mov eax,Bis14b
        mov edx,dword ptr Bkeysetup[24]
        and edx,00000800h
        .if eax != edx     ; change bit 14 for round 13 to 16
            xor eax,00000800h
            mov Bis14b,eax
            Bchangetwokeydword 26,00000800h,27,00000040h
            Bchangetwokeydword 29,00004000h,32,00000200h
            xor ecx,80h
        .endif

        mov edi,Bsaveregister
        mov ebp,dword ptr Bsbox5 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        xor ecx,ecx
        xor esi,ebp

        endm

Bch52round14   macro
        mov eax,dword ptr Bkeysetup [(14-1)*8]
        xor eax,esi
        mov edx,dword ptr Bkeysetup [(14-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        xor ebx,08h
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor ebx,ebx
        xor edi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        mov cl,dh
        and edx,0FFh
        xor edi,ebp
        mov ebp,Bcompcontrol
        mov ebp,Bround15box2check [ebp]
        mov Bsaveregister,esi
        mov edx,dword ptr Bsbox5 [edx]
        xor edi,edx
        mov edx,dword ptr Bsbox3 [ecx]
        xor edi,edx
        mov edx,dword ptr Bkeysetup [(14-1+1)*8]
        xor edx,edi
        and esi,00420082h
        shr edx,26
        xor ebp,esi
        mov edx,dword ptr Bsbox2 [edx*4]
        cmp edx,ebp
        je B_54
        mov ebp,Bcompcontrol
        jmp Btest52changed2

B_54:    mov bl,ah
        and eax,0FFh
        mov esi,Bsaveregister
        mov ebp,dword ptr Bsbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr Bsbox2 [ebx]
        xor edi,ebp

        endm

Bch52round15    macro

        mov eax,dword ptr Bkeysetup [(15-1)*8]
        xor eax,edi
        mov edx,dword ptr Bkeysetup [(15-1)*8+4]
        xor edx,edi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        xor esi,ebp
        mov bl,dl
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor esi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        mov bl,ah
        xor esi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor esi,ebp
        xor eax,20h
        mov ebp,dword ptr Bsbox3 [ecx]
        xor esi,ebp
        mov ebp,dword ptr Bsbox4 [eax]
        xor esi,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor esi,ebp
        endm

Bch52round16  macro
        mov eax,dword ptr Bkeysetup [(16-1)*8]
        xor eax,esi
        mov edx,dword ptr Bkeysetup [(16-1)*8+4]
        xor edx,esi
        and eax,0FCFCFCFCh
        and edx,0CFCFCFCFh
        mov bl,al
        rol edx,4
        mov cl,ah
        mov ebp,dword ptr Bsbox8 [ebx]
        mov bl,dl
        xor edi,ebp
        shr eax,16
        mov ebp,dword ptr Bsbox6 [ecx]
        xor edi,ebp
        mov cl,dh
        shr edx,16
        mov ebp,dword ptr Bsbox1 [ebx]
        xor edi,ebp
        mov ebp,dword ptr Bsbox7 [ecx]
        mov bl,ah
        xor edi,ebp
        mov cl,dh
        and eax,0FFh
        and edx,0FFh
        mov ebp,dword ptr Bsbox2 [ebx]
        xor ecx,08h
        xor edi,ebp
        mov ebp,dword ptr Bsbox3 [ecx]
        xor ecx,ecx
        xor edi,ebp
        mov ebp,dword ptr Bsbox4 [eax]
        xor edi,ebp
        mov ebp,dword ptr Bsbox5 [edx]
        xor edi,ebp

        endm

Btestbit52changed macro

        mov edi,Bediafter2comp
        mov ebp,Bredo2box1comp
        mov esi,Besiafter1comp
        xor edi,ebp

Btest52keyfrom3:
        Bch52round3to12a

Btest52changed2:
        ;mov ebp,Bcompcontrol  ;   NB must be set already
        .if ebp == 0
            ret
        .endif

        mov edi,Bediafter2
        mov ebp,Bredo2box1
        mov esi,Besiafter1
        xor edi,ebp

        mov Bcompcontrol,0
        jmp Btest52keyfrom3

;rest of original key
        Bch52round12b

        Bch52round13
        ;change46rest moved to round 13
        Bchange50rest
        Bch52round14
        Bch52round15

        mov ebp,Bcompcontrol
        mov edx,dword ptr Bcipherpermu [4][ebp]
        .if esi == edx
            Bch52round16
            mov ebp,Bcompcontrol
            mov eax,dword ptr Bcipherpermu [ebp]
            .if edi == eax
                call Bkey_from_permu     ;resultat i esi edi
                mov ebp,Bcompcontrol
                .if ebp != 0
                    xor esi,0FFFFFFFFh
                    xor edi,0FFFFFFFFh
                .endif
                xor edi,00100000h  ; this is bit 52 when stored to mem
                call Bkey_found_low
            .endif
        .endif
        xor ebx,ebx
        xor ecx,ecx
        mov ebp,Bcompcontrol
        jmp Btest52changed2

        endm

