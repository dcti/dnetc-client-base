; BDESLOW.ASM

; $Log: bbdeslow.asm,v $
; Revision 1.1.2.1  2001/01/21 16:57:14  cyp
; reorg
;
; Revision 1.2  1998/06/18 05:46:35  remi
; Added $Id: bbdeslow.asm,v 1.1.2.1 2001/01/21 16:57:14 cyp Exp $. and $Log: bbdeslow.asm,v $
; Added $Id$. and Revision 1.1.2.1  2001/01/21 16:57:14  cyp
; Added $Id$. and reorg
; Added $Id$. and.
;

; Part of BrydDES Key Search Library version 1.01.
; Date: January 17, 1998.
; Copyright Svend Olaf Mikkelsen, 1995, 1997, 1998.


; Please read README.TXT.

; Written for Microsoft Macro Assembler, version 6.00B.
; Modified, so it also seems to assemble correct with TASM 5.0.
; MASM @@ labels are changed to the form _01:
; TASM Parity? seems buggy, so Parity? not used.

        .386
        .model flat,c

	.data
_id	byte "@(#)$Id: bbdeslow.asm,v 1.1.2.1 2001/01/21 16:57:14 cyp Exp $"
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


extern                  bbryd_continue   :near
extern                  bbryd_key_found  :near

public                  bbryd_des
public                  Bdesencrypt
public                  Bdesdecrypt
public                  Bdesinit
public                  Bkey_byte_to_hex
public                  Bc_key_byte_to_hex

        include Bbdesmac.inc      ; macros


        .data

        include Bbdesdat.inc    ; DES data

        byte "BrydDES Key Search Library version 1.01.  Core 2. "
        byte "Copyright Svend Olaf Mikkelsen, 1995, 1997, 1998. "

Bchangeoffsets   dword offset Bchangebit1
                dword offset Bchangebit2
                dword offset Bchangebit3
                dword offset Bchangebit4
                dword offset Bchangebit5
                dword offset Bchangebit6
                dword offset Bchangebit7
                dword 0
                dword offset Bchangebit9
                dword offset Bchangebit10
                dword offset Bchangebit11
                dword offset Bchangebit12
                dword offset Bchangebit13
                dword offset Bchangebit14
                dword offset Bchangebit15
                dword 0
                dword offset Bchangebit17
                dword offset Bchangebit18
                dword offset Bchangebit19
                dword offset Bchangebit20
                dword offset Bchangebit21
                dword offset Bchangebit22
                dword offset Bchangebit23
                dword 0
                dword offset Bchangebit25
                dword offset Bchangebit26
                dword offset Bchangebit27
                dword offset Bchangebit28
                dword offset Bchangebit29
                dword offset Bchangebit30
                dword offset Bchangebit31
                dword 0
                dword offset Bchangebit33
                dword offset Bchangebit34
                dword offset Bchangebit35
                dword offset Bchangebit36
                dword offset Bchangebit37
                dword offset Bchangebit38
                dword offset Bchangebit39
                dword 0
                dword offset Bchangebit41
                dword offset Bchangebit42
                dword offset Bchangebit43
                dword offset Bchangebit44
                dword offset Bchangebit45
                dword offset Bchangebit46
                dword offset Bchangebit47
                dword 0
                dword offset Bchangebit49
                dword offset Bchangebit50
                dword offset Bchangebit51
                dword offset Bchangebit52
                dword offset Bchangebit53
                dword offset Bchangebit54
                dword offset Bchangebit55
                dword 0
                dword offset Bchangebit57
                dword offset Bchangebit58
                dword offset Bchangebit59
                dword offset Bchangebit60
                dword offset Bchangebit61
                dword offset Bchangebit62
                dword offset Bchangebit63

        .data?
align 4

Bdatastart               dword ?

Bkeysetup                byte 128 dup (?)
Bplainaddr               dword ?
Bplainpermu              byte 8 dup (?)
Bplainpermucomp          byte 8 dup (?)
Bcipheraddr              dword ?
Bivaddr                  dword ?
Bfoundkeyaddr            dword ?
Bmaskaddr                dword ?

Bsbox1                   dword 64 dup (?)
Bsbox2                   dword 64 dup (?)
Bsbox3                   dword 64 dup (?)
Bsbox4                   dword 64 dup (?)
Bsbox5                   dword 64 dup (?)
Bsbox6                   dword 64 dup (?)
Bsbox7                   dword 64 dup (?)
Bsbox8                   dword 64 dup (?)

Bround4box4              dword 64 dup (?)
Bround6box4              dword 64 dup (?)
Bround7box1              dword 64 dup (?)
Bround8box3              dword 64 dup (?)
Bround9box1              dword 64 dup (?)
Bround11box4             dword 64 dup (?)
Bround12box1             dword 64 dup (?)

Btbox1                   dword 64 dup (?)
Btbox5                   dword 64 dup (?)
Btbox3                   dword 64 dup (?)

                        dword 64 dup (?)  ; not used, speed lost if removed?
                        dword 64 dup (?)  ; not used

Bkeyaddr                 dword ?
Btempkey                 byte 8 dup(?)
Bfoundkey                byte 8 dup(?)
                        dword ?  ; saveesi not used
                        dword ?  ; saveesicomp not used


                        dword ?    ; not used
                        dword ?    ; not used
                        dword ?    ; not used
                        dword ?    ; not used

Bediafter2comp           dword ?
BL1                      dword ?

;theese 10 lines must be in this order
Bcipherpermu             byte 8 dup (?)
Bcipherpermucheck        dword ?    ; prepared, but not used
Bround15box2check        dword ?
Bround15box5check        dword ?
                        dword ?    ; do not remove this line
Bcipherpermucomp         byte 8 dup (?)
Bcipherpermucheckcomp    dword ?    ; prepared, but not used
Bround15box2checkcomp    dword ?
Bround15box5checkcomp    dword ?
                        dword ?

                        dword ?    ; not used
Bsaveregister            dword ?

Bundo2box1               dword ?
Bundo2box1comp           dword ?
Bredo2box3               dword ?
Bredo2box1               dword ?
Bediafter2               dword ?
Bredo2box5               dword ?
Bredo2box5comp           dword ?
Bsaveeax                 dword ?
Bredo2box1comp           dword ?
Bredo2box3comp           dword ?

                        dword ?    ; not used
                        dword ?    ; not used

Bredo16box2              dword ?
Bredo16box2comp          dword ?
Bredo16box7              dword ?
Bredo16box7comp          dword ?
Bsaveedx                 dword ?

;theese 10 lines must be in this order
Bround13box3check00      dword ?
Bround13box3check01      dword ?
Bround13box3check10      dword ?
Bround13box3check11      dword ?
Bcheckoffset             dword ?
                        dword ?    ; do not remove this line
Bround13box3check00comp  dword ?    ; offset 24 from round13box3check00
Bround13box3check01comp  dword ?
Bround13box3check10comp  dword ?
Bround13box3check11comp  dword ?

                        dword ?    ; not used
Bcipherpermucompbox3bits dword ?
Bcipherpermubox3bits     dword ?

Bround2box1ptr           dword ?
Bround2box1ptrcomp       dword ?
Bsaveesiround16          dword ?
Bsaveesiround16comp      dword ?

Besiafter1               dword ?
Besiafter1comp           dword ?

Bredo1box1               dword ?
Bredo1box3               dword ?
Bredo1box5               dword ?
Bredo1box1comp           dword ?
Bredo1box3comp           dword ?
Bredo1box5comp           dword ?
Bsaveesp                 dword ?

Bcompcontrol             dword ?
Bis14b                   dword ?
Bis18b                   dword ?
Bkeyisfound              dword ?

Bfirstcall               dword ?

Btable_of_changedata dword 28 dup(?);   numbered from 1

Bkeysetupoffset          dword ?
Bbit61_62_63_change      dword ?

                        dword ?
Badd_zero                dword ?
Bdataend                 dword ?


        .code
        nop
        nop
        nop
        nop

bbryd_des:              ; procedure

Bbryd_des_frame equ 24

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
        mov edi,offset Bdatastart
        mov ecx,offset Bdataend
        sub ecx,edi
        shr ecx,2
        mov eax,0CE6755BEh         ; random number
        rep stosd

        ; count the mask bits set to zero
        mov eax,[esp][Bbryd_des_frame+16]  ; mask address
        mov edx,dword ptr [eax]
        mov eax,dword ptr [eax+4]
        bswapmacro eax
        bswapmacro edx
        mov ebx,00000000001001000111010001010000b
        test ebx,eax
        .if !zero?
            mov eax,3
            jmp Bbryd_end
        .endif
        test eax,00000000000000000000000000001110b
        .if zero?
            mov Bbit61_62_63_change,1
        .else
            mov Bbit61_62_63_change,0
        .endif
        xor eax,ebx    ; bit 43 46 50 51 52 54 58 60 set to zero
        or eax, 01010101h
        or edx, 01010101h
        mov ebx,0                  ; count of bits to change
        mov ecx,64
B_01:    test edx,80000000h
        .if zero?
            add ebx,1
        .endif
        call Bshiftleft_edx_eax
        loop B_01
        .if ebx > 27
            mov eax,3
            jmp Bbryd_end
        .endif
        mov esi,28
        sub esi,ebx
        mov Bfirstcall,esi

        ;set up table with the bit numbers to be changed
        mov eax,[esp][Bbryd_des_frame+16]  ; mask address
        mov edx,dword ptr [eax]
        mov eax,dword ptr [eax+4]
        bswapmacro eax
        bswapmacro edx
        xor eax,00000000001001000111010001010000b
        or eax,01010101h          ; parity bits
        or edx,01010101h
        mov ecx,64
        mov ebx,1
B_02:    test edx,80000000h
        .if zero?
                mov ebp,Bchangeoffsets [ebx*4-4]
                mov Btable_of_changedata [esi*4],ebp
                inc esi
        .endif
        call Bshiftleft_edx_eax
        inc ebx
        loop B_02

        mov Bis14b,0
        mov Bis18b,0
        mov Bcompcontrol,0
        mov Bkeyisfound,0

        mov Bsaveesp,esp

        mov esi,[esp][Bbryd_des_frame+0]
        mov Bplainaddr,esi

        mov esi,[esp][Bbryd_des_frame+4]
        mov Bcipheraddr,esi

        mov esi,[esp][Bbryd_des_frame+8]
        mov Bivaddr,esi

        mov esi,[esp][Bbryd_des_frame+12]
        mov Bkeyaddr,esi

        mov esi,[esp][Bbryd_des_frame+16]
        mov Bmaskaddr,esi

        ; setup initial key and set up S-boxes.
        mov ebp,Bkeyaddr

        mov eax,dword ptr [ebp]
        mov ebx,Bmaskaddr
        and eax,dword ptr [ebx]
        mov dword ptr [ebp],eax

        mov eax,dword ptr [ebp+4]
        and eax,dword ptr [ebx+4]
        mov dword ptr [ebp+4],eax

        push ebp
        call Bdesinit
        add esp,4

        ; round4box4 is used for testing bit 52 changed
        ; whithout changing the key setup.
        mov ecx,64
        mov ebx,0
B_03:    xor ebx,10h
        mov eax,dword ptr Bsbox4 [ebx]
        xor ebx,10h
        mov dword ptr Bround4box4 [ebx],eax
        add ebx,4
        loop B_03

        mov ecx,64
        mov ebx,0
B_04:    xor ebx,80h
        mov eax,dword ptr Bsbox4 [ebx]
        xor ebx,80h
        mov dword ptr Bround6box4 [ebx],eax
        add ebx,4
        loop B_04

        mov ecx,64
        mov ebx,0
B_05:    xor ebx,80h
        mov eax,dword ptr Bsbox1 [ebx]
        xor ebx,80h
        mov dword ptr Bround7box1 [ebx],eax
        add ebx,4
        loop B_05

        mov ecx,64
        mov ebx,0
B_06:    xor ebx,20h
        mov eax,dword ptr Bsbox3 [ebx]
        xor ebx,20h
        mov dword ptr Bround8box3 [ebx],eax
        add ebx,4
        loop B_06

        mov ecx,64
        mov ebx,0
B_07:    xor ebx,20h
        mov eax,dword ptr Bsbox1 [ebx]
        xor ebx,20h
        mov dword ptr Bround9box1 [ebx],eax
        add ebx,4
        loop B_07

        mov ecx,64
        mov ebx,0
B_08:    xor ebx,40h
        mov eax,dword ptr Bsbox4 [ebx]
        xor ebx,40h
        mov dword ptr Bround11box4 [ebx],eax
        add ebx,4
        loop B_08

        mov ecx,64
        mov ebx,0
B_09:    xor ebx,04h
        mov eax,dword ptr Bsbox1 [ebx]
        xor ebx,04h
        mov dword ptr Bround12box1 [ebx],eax
        add ebx,4
        loop B_09

        ;tbox1 is used in round 2 for making data
        ;to calculate output with bit 52 changed.
        mov ecx,64
        mov ebx,0
B_10:    mov eax,dword ptr Bsbox1 [ebx]
        xor ebx,10h
        mov edx,dword ptr Bsbox1 [ebx]
        xor ebx,10h
        xor eax,edx
        mov dword ptr Btbox1 [ebx],eax
        add ebx,4
        loop B_10

        ;tbox3 is used in round 2 for making data
        ;to calculate output with bit 50 changed.
        mov ecx,64
        mov ebx,0
B_11:    mov eax,dword ptr Bsbox3 [ebx]
        xor ebx,04h
        mov edx,dword ptr Bsbox3 [ebx]
        xor ebx,04h
        xor eax,edx
        mov dword ptr Btbox3 [ebx],eax
        add ebx,4
        loop B_11

        ;tbox5 is used in round 2 for making data
        ;to calculate output with bit 46 changed.
        mov ecx,64
        mov ebx,0
B_12:    mov eax,dword ptr Bsbox5 [ebx]
        xor ebx,10h
        mov edx,dword ptr Bsbox5 [ebx]
        xor ebx,10h
        xor eax,edx
        mov dword ptr Btbox5 [ebx],eax
        add ebx,4
        loop B_12


        mov ebx,Bcipheraddr
        mov esi,dword ptr [ebx]
        mov edi,dword ptr [ebx][4]

        call Binitial_permu

        mov dword ptr Bcipherpermu,esi
        mov dword ptr Bcipherpermu [4],edi

        mov eax,40104100h
        and eax,edi
        mov Bcipherpermucheck,eax     ; not used in this version
        xor eax,40104100h
        mov Bcipherpermucheckcomp,eax ; not used in this version

        mov eax,00420082h
        and eax,edi
        mov Bround15box2check,eax     ; expected output from round 15, sbox 2
        xor eax,00420082h
        mov Bround15box2checkcomp,eax ; ..comp means used for testing
                                     ; complement key

        xor esi,0FFFFFFFFh
        xor edi,0FFFFFFFFh
        mov dword ptr Bcipherpermucomp,esi
        mov dword ptr Bcipherpermucomp [4],edi
        and edi,20080820h
        mov dword ptr Bcipherpermucompbox3bits,edi

        mov edi,dword ptr Bcipherpermu [4]
        and edi,20080820h
        mov dword ptr Bcipherpermubox3bits,edi

        mov ebx,Bivaddr
        mov esi,dword ptr [ebx]
        mov edi,dword ptr [ebx][4]

        mov ebx,Bplainaddr
        xor esi,dword ptr [ebx]
        xor edi,dword ptr [ebx][4]

        call Binitial_permu

        mov dword ptr Bplainpermu,esi
        mov dword ptr Bplainpermu [4],edi
        xor esi,0FFFFFFFFh
        xor edi,0FFFFFFFFh
        mov dword ptr Bplainpermucomp,esi
        mov dword ptr Bplainpermucomp [4],edi

        ; Start key testing:
        xor ebx,ebx
        xor ecx,ecx

        ; Rule for ebx and ecx:
        ; Whenever other bits than the bl and cl bits are used
        ; the registers must be zeroed afterwards.
        ; Extra xor ebx,ebx and xor ecx,ecx are however inserted
        ; to enhance Pentium Pro speed.

        .if Bfirstcall == 1
            call Bbit1
        .elseif Bfirstcall == 2
            call Bbit2
        .elseif Bfirstcall == 3
            call Bbit3
        .elseif Bfirstcall == 4
            call Bbit4
        .elseif Bfirstcall == 5
            call Bbit5
        .elseif Bfirstcall == 6
            call Bbit6
        .elseif Bfirstcall == 7
            call Bbit7
        .elseif Bfirstcall == 8
            call Bbit8
        .elseif Bfirstcall == 9
            call Bbit9
        .elseif Bfirstcall == 10
            call Bbit10
        .elseif Bfirstcall == 11
            call Bbit11
        .elseif Bfirstcall == 12
            call Bbit12
        .elseif Bfirstcall == 13
            call Bbit13
        .elseif Bfirstcall == 14
            call Bbit14
        .elseif Bfirstcall == 15
            call Bbit15
        .elseif Bfirstcall == 16
            call Bbit16
        .elseif Bfirstcall == 17
            call Bbit17
        .elseif Bfirstcall == 18
            call Bbit18
        .elseif Bfirstcall == 19
            call Bbit19
        .elseif Bfirstcall == 20
            call Bbit20
        .elseif Bfirstcall == 21
            call Bbit21
        .elseif Bfirstcall == 22
            call Bbit22
        .elseif Bfirstcall == 23
            call Bbit23
        .elseif Bfirstcall == 24
            call Bbit24
        .elseif Bfirstcall == 25
            call Bbit25
        .elseif Bfirstcall == 26
            call Bbit26
        .elseif Bfirstcall == 27
            call Bbit27
        .else
            call Bbitno51
        .endif

        mov edx,0
        .if Bkeyisfound == 1
            mov eax, 0
        .else
            mov eax,1              ; eax 1, finished, not interrupted
        .endif
        jmp Bbryd_end

Bbryd_not_continue:
        .if Bkeyisfound == 1
            mov eax, 0
        .else
            mov eax,2
        .endif
Bbryd_end:
        mov edx,0
        pop ecx
        pop ebx
        pop ebp
        pop edi
        pop esi

        ret


Bchange:         ; local procedure

        mov ebp, Btable_of_changedata[ebp*4]

        xor eax,eax
        xor edx,edx
        xor ecx,ecx
        mov cl,byte ptr [ebp]
        mov dl,byte ptr [ebp+2]
        mov al,byte ptr [ebp+1]
        inc ebp

B_13:    mov edi,dword ptr Bchangetable [edx*4-4]
        mov esi,dword ptr Bkeysetup [eax*4-4]
        add ebp,2
        xor esi,edi
        mov dl,byte ptr [ebp+1]
        mov dword ptr Bkeysetup [eax*4-4],esi
        mov al,byte ptr [ebp]
        loop B_13
        xor ecx,ecx

        ret


        include Bbdeschg.inc ; key setup change macros
                            ; used for bits which are often changed

Bbit1:   call Bbit2
        mov ebp,1
        call Bchange

Bbit2:   call Bbit3
        mov ebp,2
        call Bchange

Bbit3:   call Bbit4
        mov ebp,3
        call Bchange

Bbit4:   call Bbit5
        mov ebp,4
        call Bchange

Bbit5:   call Bbit6
        mov ebp,5
        call Bchange

Bbit6:   call Bbit7
        mov ebp,6
        call Bchange

Bbit7:   call Bbit8
        mov ebp,7
        call Bchange

Bbit8:   call Bbit9
        mov ebp,8
        call Bchange

Bbit9:   call Bbit10
        mov ebp,9
        call Bchange

Bbit10:  call Bbit11
        mov ebp,10
        call Bchange

Bbit11:  call Bbit12
        mov ebp,11
        call Bchange

Bbit12:  call Bbit13
        mov ebp,12
        call Bchange

Bbit13:  call Bbit14
        mov ebp,13
        call Bchange

Bbit14:  call Bbit15
        mov ebp,14
        call Bchange

Bbit15:  call Bbit16
        mov ebp,15
        call Bchange

Bbit16:  call Bbit17
        mov ebp,16
        call Bchange

Bbit17:  call Bbit18
        mov ebp,17
        call Bchange

Bbit18:  call Bbit19
        mov ebp,18
        call Bchange

Bbit19:  call Bbit20
        mov ebp,19
        call Bchange

Bbit20:  call bbryd_continue
        .if eax == 0
            mov esp,Bsaveesp
            jmp Bbryd_not_continue
        .endif
        xor ebx,ebx
        xor ecx,ecx

        call Bbit21
        mov ebp,20
        call Bchange

Bbit21:  call Bbit22
        mov ebp,21
        call Bchange

Bbit22:  call Bbit23
        mov ebp,22
        call Bchange

Bbit23:  call Bbit24
        mov ebp,23
        call Bchange

Bbit24:  call Bbit25
        mov ebp,24
        call Bchange

Bbit25:  .if Bbit61_62_63_change == 1
            jmp Bbit25a
        .endif
        call Bbit26
        mov ebp,25
        call Bchange

Bbit26:  call Bbit27
        mov ebp,26
        call Bchange

Bbit27:  call Bbitno51
        mov ebp,27
        call Bchange
        jmp Bbitno51

Bbit25a: call Bbit26a
        Bchange61

Bbit26a: call Bbit27a
        Bchange62

Bbit27a: call Bbitno51
        Bchange63

; Key bit 51, 60, 54 and 58 are not used in round 16.

Bbitno51:  ; Compute expected output from round 13 sbox 3,
          ; which equals to output from round 15 sbox 3 in decryption mode.
          ; Part 1.

        Bchange46rest ; update key setup up bit 46, round 13 to 16
        Bchange50rest
        Bround15box3part1

        call Bbitno60
        Bchange51

Bbitno60:  ; Compute round 1 for key and complement key, as well as values for
          ; changing the output, since bit 60, 54 and 58 are used in round 1.

        Bdesround1comp   ; incl. load esi edi  xor ebx,ebx  xor ecx,ecx
        mov Besiafter1comp,esi
        Bdesround1
        mov Besiafter1,esi
        call Bbitno54
        Bchange60

        ; Redo round 1, sbox 1.
        mov eax,Besiafter1
        mov ebp,Bredo1box1
        mov edx,Besiafter1comp
        xor eax,ebp
        mov ebp,Bredo1box1comp
        mov Besiafter1,eax
        xor edx,ebp
        mov Besiafter1comp,edx

Bbitno54:  ; Compute expected output from round 13 box 3. Part 2.
        Bround15box3part2
        call Bbitno58
        Bchange54

        ; Redo round 1, sbox 5.
        mov eax,Besiafter1
        mov ebp,Bredo1box5
        mov edx,Besiafter1comp
        xor eax,ebp
        mov ebp,Bredo1box5comp
        mov Besiafter1,eax
        xor edx,ebp
        mov Besiafter1comp,edx

Bbitno58:
        call Bbitno50
        Bchange58

        ; Redo round 1, sbox 3.
        mov eax,Besiafter1
        mov ebp,Bredo1box3
        mov edx,Besiafter1comp
        xor eax,ebp
        mov ebp,Bredo1box3comp
        mov Besiafter1,eax
        xor edx,ebp
        mov Besiafter1comp,edx

; Key bit 50, 46, 43 and 52 are not used in round 1.

Bbitno50:  mov Bcheckoffset,offset Bround13box3check00
        ; 00 means: bit46 0  bit50 0
        ; We have 4 values for expected output of round 13 box 3
        ; (8 including complement keys)
        ; depending on the state of bit 46 and 50.

Btestkey:
        xor ebx,ebx
        xor ecx,ecx
        mov esi,Besiafter1comp
        mov edi,dword ptr Bplainpermucomp [4]
        Bdesround2comp  ; compute round 2 and values for changing the output.
        mov Bediafter2comp,edi
        mov esi,Besiafter1
        mov edi,dword ptr Bplainpermu [4]
        Bdesround2
        mov Bediafter2,edi
        call Btestkeyfrom3

        mov Bcheckoffset,offset Bround13box3check10
        call Bchange46testfrom3

        Bchange50new
        mov Bcheckoffset,offset Bround13box3check11
        Bredo2box3macro   ; redo round 2, sbox 3   loads esi and edi
        call Btestkeyfrom3

        mov Bcheckoffset,offset Bround13box3check01
        call Bchange46testfrom3

        Bchange43
        Bredo2box1macro  ; redo round 2, sbox 1   loads esi and edi
                        ; also calculates new values for redoing
                        ; box 1, when bit 52 changes
                        ; since box 1 depends on key bit 43 and key bit 52

        call Btestkeyfrom3

        mov Bcheckoffset,offset Bround13box3check11
        call Bchange46testfrom3

        Bchange50new
        mov Bcheckoffset,offset Bround13box3check10
        Bredo2box3macro
        call Btestkeyfrom3

        mov Bcheckoffset,offset Bround13box3check00
        ;call change46testfrom3
        ;ret
; The call and return is commented out, so the test procedure will
; return directly to the key setup change of next bit.

Bchange46testfrom3:
        Bchange46new
        Bredo2box5macro

Btestkeyfrom3:
        Bdesmacro310   ; round 3 to 10
        Bdesround 11,esi,edi
        Bdesround12part1

Btest2:
        ;mov ebp,compcontrol    NB ebp must be set in the DES rounds
        .if ebp != 0
            Btestbit52changed
        .endif

        mov edi,Bediafter2comp
        mov esi,Besiafter1comp
        mov Bcompcontrol,24
        jmp Btestkeyfrom3

;rest of original key
        Bdesround12part2

        Bdesround13
        ;change46rest   moved to round 13
        Bchange50rest
        Bdesround14
        Bdesround 15,esi,edi

        mov ebp,Bcompcontrol
        mov edx,dword ptr Bcipherpermu [4][ebp]
        .if esi == edx
            Bdesround 16,edi,esi
            mov ebp,Bcompcontrol
            mov eax,dword ptr Bcipherpermu [ebp]
            .if edi == eax
                call Bkey_from_permu     ;result in esi edi
                mov ebp,Bcompcontrol
                .if ebp != 0
                    xor esi,0FFFFFFFFh
                    xor edi,0FFFFFFFFh
                .endif
                call Bkey_found_low
            .endif
        .endif
        xor ebx,ebx
        xor ecx,ecx
        mov ebp,Bcompcontrol
        jmp Btest2


Bpermute:               ; local procedure
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

Binitial_permu:         ; local procedure

        Binitialpermumacro esi,edi,ecx
        ret

Bkey_from_permu:  ; local procedure

;keyfromsetupdata  dword 1,1,00400000h
;                dword 2,1,20000000h
;                dword 3,2,80000000h

        mov ecx,28
        mov edx,offset Bkeyfromsetupdata
        mov esi,0
        mov edi,0

B_14:    mov eax,[edx] ; bit no
        mov ebx,[edx+4] ; key dword no
        mov ebp,[edx+8] ; check data
        and ebp, dword ptr Bkeysetup [ebx*4-4]
        .if !zero?
            or esi,Bchangetable [eax*4-4]
        .endif
        add edx,12
        loop B_14
        bswapmacro esi

        mov ecx,28
_15:    mov eax,[edx]   ; bit no
        sub eax,32
        mov ebx,[edx+4] ; key dword no
        mov ebp,[edx+8] ; check data
        and ebp, dword ptr Bkeysetup [ebx*4-4]
        .if !zero?
            or edi,Bchangetable [eax*4-4]
        .endif
        add edx,12
        loop _15
        bswapmacro edi

        ret


Bkey_found_low:         ; local procedure

        mov Bkeyisfound,1

        ; key received in esi edi
        mov dword ptr Btempkey,esi
        mov dword ptr Btempkey[4],edi

        mov esi,offset Btempkey
        mov edi,offset Bfoundkey
        mov ecx,8
B_16:    mov al,byte ptr [esi]
        and al,al
        jnp B_16a
        ;.if parity?
            xor al,00000001b
        ;.endif
B_16a:
        mov byte ptr [edi],al
        add esi,1
        add edi,1
        loop B_16
        push offset Bfoundkey
        call bbryd_key_found
        add esp,4
        xor ebx,ebx
        xor ecx,ecx

        ret


Bdesinit:               ; procedure

Bdesinit_frame equ 24

; ptr key
; 0    [in]

        push esi
        push edi
        push ebp
        push ebx
        push ecx

        mov ebp, dword ptr [esp][Bdesinit_frame]

        ; set up key
        mov ebx, ebp
        mov eax,dword ptr [ebx]
        mov edx,dword ptr [ebx][4]
        bswapmacro eax
        bswapmacro edx

        mov ebp, offset Bkeypermu1
        call Bpermute

        mov eax,ebx
        mov edx,ecx
        mov BL1,1
        mov esi,offset Bkeysetup
        .while BL1 <= 16
            push esi

            mov ebp,offset Bkeypermu2
            call Bpermute

            .if BL1 > 2 && BL1 != 9 && BL1 != 16
                mov eax,ebx
                mov edx,ecx

                mov ebp, offset Bkeypermu2
                call Bpermute

            .endif
            mov eax,ebx
            mov edx,ecx

            mov ebp, offset Bkeypermu3
            call Bpermute

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
            inc BL1

        .endw

        ; set up S-boxes
        push ebp
        mov esi,offset Bsboxdata
        mov edi,0
        mov ebp,0
        .while ebp < 8*4
            mov ah,1
            .while ah <= 64
                mov ebx,0
                test ah,1
                jz B_17
                lodsb
    B_17:        shl al,1
                .if carry?
                    mov cl,Bspermu [ebp]
                    mov edx,1
                    ror edx,cl
                    add ebx,edx
                .endif
                inc ebp
                test ebp,00000011b
                jnz B_17
                sub ebp,4
                rol ebx,3
                mov dword ptr Bsbox1 [edi],ebx
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

Bshiftleft_edx_eax:     ; local procedure
        shl edx,1
        shl eax,1
        .if carry?
            or edx,1
        .endif
        ret


Bdesencrypt:            ; procedure

Bdesencrypt_frame equ 24

; ptr plain    ptr ciphertext
; 0    [in]    4        [out]

        push esi
        push edi
        push ebp
        push ebx
        push ecx

        mov eax,dword ptr [esp][Bdesencrypt_frame]
        mov esi, dword ptr [eax]
        mov edi, dword ptr [eax+4]

        mov Bkeysetupoffset,0
        mov ebp,0

        Binitialpermumacro esi,edi,ecx

        xor ebx,ebx
        xor ecx,ecx

B_18:    mov eax,dword ptr Bkeysetup[ebp]
        xor eax,edi
        mov edx,dword ptr Bkeysetup[ebp+4]
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
        mov ebp,Bkeysetupoffset
        mov eax,esi
        add ebp,8
        mov Bkeysetupoffset,ebp
        mov esi,edi
        mov edi,eax
        cmp ebp, 128
        jb B_18

        Bfinalpermumacro esi,edi,ecx

        mov eax,dword ptr [esp][Bdesencrypt_frame+4]
        mov dword ptr [eax], edi
        mov dword ptr [eax+4] , esi

        pop ecx
        pop ebx
        pop ebp
        pop edi
        pop esi

        ret

Bdesdecrypt:            ; procedure

Bdesdecrypt_frame equ 24

; ptr cipher   ptr plain
; 0    [in]    4        [out]

        push esi
        push edi
        push ebp
        push ebx
        push ecx

        mov eax,dword ptr [esp][Bdesdecrypt_frame]
        mov esi, dword ptr [eax]
        mov edi, dword ptr [eax+4]

        mov Bkeysetupoffset,120
        mov ebp,120

        Binitialpermumacro esi,edi,ecx

        xor ebx,ebx
        xor ecx,ecx

B_19:    mov eax,dword ptr Bkeysetup [ebp]
        xor eax,edi
        mov edx,dword ptr Bkeysetup [ebp+4]
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
        mov ebp,Bkeysetupoffset
        mov eax,esi
        sub ebp,8
        mov Bkeysetupoffset,ebp
        mov esi,edi
        mov edi,eax
        cmp ebp,0
        jnl B_19

        Bfinalpermumacro esi,edi,ecx

        mov eax,dword ptr [esp][Bdesdecrypt_frame+4]
        mov dword ptr [eax], edi
        mov dword ptr [eax+4] , esi

        pop ecx
        pop ebx
        pop ebp
        pop edi
        pop esi

        ret

Bkey_byte_to_hex:       ; procedure
        mov Badd_zero,0
        jmp Bkey_byte_to_hex_1
Bc_key_byte_to_hex:
        mov Badd_zero,1
Bkey_byte_to_hex_1:


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
B_20:    mov al,byte ptr [esi]
        and al,al
        jnp B_20a
        ;.if parity?
            mov edx,1
            xor al,00000001b
        ;.endif
B_20a:
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
        loop B_20

        .if Badd_zero == 1
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




