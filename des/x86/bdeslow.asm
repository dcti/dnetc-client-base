; BDESLOW.ASM

; $Log: bdeslow.asm,v $
; Revision 1.1.2.1  2001/01/21 16:57:15  cyp
; reorg
;
; Revision 1.2  1998/06/18 05:46:24  remi
; Added $Id: bdeslow.asm,v 1.1.2.1 2001/01/21 16:57:15 cyp Exp $. and $Log: bdeslow.asm,v $
; Added $Id$. and Revision 1.1.2.1  2001/01/21 16:57:15  cyp
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
_id	byte "@(#)$Id: bdeslow.asm,v 1.1.2.1 2001/01/21 16:57:15 cyp Exp $"
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


extern                  bryd_continue   :near
extern                  bryd_key_found  :near

public                  bryd_des
public                  desencrypt
public                  desdecrypt
public                  desinit
public                  key_byte_to_hex
public                  c_key_byte_to_hex

        include bdesmac.inc      ; macros


        .data

        include bdesdat.inc    ; DES data

        byte "BrydDES Key Search Library version 1.01.  Core 2. "
        byte "Copyright Svend Olaf Mikkelsen, 1995, 1997, 1998. "

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
        nop
        nop
        nop
        nop

bryd_des:              ; procedure

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
        call desinit
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


        include bdeschg.inc ; key setup change macros
                            ; used for bits which are often changed

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

bit20:  call bryd_continue
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
        call bryd_key_found
        add esp,4
        xor ebx,ebx
        xor ecx,ecx

        ret


desinit:               ; procedure

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


desencrypt:            ; procedure

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

desdecrypt:            ; procedure

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

key_byte_to_hex:       ; procedure
        mov add_zero,0
        jmp key_byte_to_hex_1
c_key_byte_to_hex:
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




