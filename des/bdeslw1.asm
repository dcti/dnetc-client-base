;
; $Log: bdeslw1.asm,v $
; Revision 1.1  1998/06/17 20:36:02  cyruspatel
; moved generic intel asm bdeslow.asm, bbdeslow.asm files to ./client/des/
; modified model definition directives to work with any (intel asm)
; assembler - model is overridden to flat unless small model was explicitely
; defined on the command line. Don't declare segreg 'assume's: The .model
; directive controls segreg assumptions unless explicitely declared.
; Added $Logs.
;
; Revision 1.3  1998/06/15 02:54:43  jlawson
; updated bbdeslow.asm to be based off the same source that bdeslow.asm is.
; all asm's modified to use a "modelnum" equate to define memory model.
;
; Revision 1.2  1998/06/09 08:54:42  jlawson
; Changes from Cyrus Patel - disassembled tasm flat model objs to generic 
; intel asm for porting ease. Removed flat model specific assembler 
; statements to allow use of mem model overrides from the command line.
;
; Revision 1.1  1998/05/25 21:29:06  bovine
; Import 5/23/98 client tree
;

.386p

ifndef __SMALL__ 
  .model flat 
endif

        EXTRN           _bryd_key_found:NEAR    ; ok for flat or small model
        EXTRN           _bryd_continue:NEAR     ; ok for flat or small model
        PUBLIC          _bryd_des               ; Located at 1:0000h Type = 1
        PUBLIC          _desinit                ; Located at 1:232Ch Type = 1
        PUBLIC          _desencrypt             ; Located at 1:2469h Type = 1
        PUBLIC          _desdecrypt             ; Located at 1:25E5h Type = 1
        PUBLIC          _key_byte_to_hex        ; Located at 1:275Eh Type = 1
        PUBLIC          _c_key_byte_to_hex      ; Located at 1:276Ah Type = 1

        DGROUP GROUP _DATA,_BSS
_TEXT   SEGMENT DWORD PUBLIC USE32 'CODE'
_TEXT   ENDS
_DATA   SEGMENT DWORD PUBLIC USE32 'DATA'
_DATA   ENDS
_BSS    SEGMENT DWORD PUBLIC USE32 'BSS'
_BSS    ENDS

_TEXT   SEGMENT
        ;ifdef __SMALL__                        ; leave commented out to
        ;assume  cs: _TEXT, ds:DGROUP           ; let the assembler figure 
        ;else  ; __FLAT__                       ; out what to assume
        ;assume  cs: FLAT, ds:FLAT, ss:FLAT
        ;endif

_bryd_des:
        push    esi
        push    edi
        push    ebp
        push    ebx
        push    ecx
        mov     edi,offset $S1
        mov     ecx,offset $S2
        sub     ecx,edi
        shr     ecx,02h
        mov     eax,0CE6755BEh
        repz    stosd
        mov     eax,[esp+028h]
        mov     edx,[eax]
        mov     eax,[eax+004h]
        push    eax
        mov     eax,eax
        ror     ax,08h
        ror     eax,10h
        ror     ax,08h
        push    eax
        mov     eax,[esp+004h]
        pop     eax
        add     esp,+004h
        push    eax
        mov     eax,edx
        ror     ax,08h
        ror     eax,10h
        ror     ax,08h
        push    eax
        mov     eax,[esp+004h]
        pop     edx
        add     esp,+004h
        mov     ebx,00247450h
        test    ebx,eax
        jz      short $L3
        mov     eax,00000003h
        jmp     $L4
$L3:    test    eax,0000000Eh
        jnz     short $L5
        mov     dword ptr $S6,00000001h
        jmp     short $L7
$L5:    mov     dword ptr $S6,00000000h
$L7:    xor     eax,ebx
        or      eax,01010101h
        or      edx,01010101h
        mov     ebx,00000000h
        mov     ecx,00000040h
$L10:   test    edx,80000000h
        jnz     short $L8
        add     ebx,+001h
$L8:    call    $L9
        loop    short $L10
        cmp     ebx,+01Bh
        jbe     short $L11
        mov     eax,00000003h
        jmp     $L4
$L11:   mov     esi,0000001Ch
        sub     esi,ebx
        mov     $S12,esi
        mov     eax,[esp+028h]
        mov     edx,[eax]
        mov     eax,[eax+004h]
        push    eax
        mov     eax,eax
        ror     ax,08h
        ror     eax,10h
        ror     ax,08h
        push    eax
        mov     eax,[esp+004h]
        pop     eax
        add     esp,+004h
        push    eax
        mov     eax,edx
        ror     ax,08h
        ror     eax,10h
        ror     ax,08h
        push    eax
        mov     eax,[esp+004h]
        pop     edx
        add     esp,+004h
        xor     eax,00247450h
        or      eax,01010101h
        or      edx,01010101h
        mov     ecx,00000040h
        mov     ebx,00000001h
$L16:   test    edx,80000000h
        jnz     short $L13
        mov     ebp,$S14[ebx*4]
        mov     $S15[esi*4],ebp
        inc     esi
$L13:   call    $L9
        inc     ebx
        loop    short $L16
        mov     dword ptr $S17,00000000h
        mov     dword ptr $S18,00000000h
        mov     dword ptr $S19,00000000h
        mov     dword ptr $S20,00000000h
        mov     dword ptr $S21,00000000h
        mov     dword ptr $S22,00000000h
        mov     $S23,esp
        mov     esi,[esp+018h]
        mov     $S24,esi
        mov     esi,[esp+01Ch]
        mov     $S25,esi
        mov     esi,[esp+020h]
        mov     $S26,esi
        mov     esi,[esp+024h]
        mov     $S27,esi
        mov     esi,[esp+028h]
        mov     $S28,esi
        mov     ebp,$S27
        mov     eax,[ebp]
        mov     ebx,$S28
        and     eax,[ebx]
        mov     [ebp],eax
        mov     eax,[ebp+004h]
        and     eax,[ebx+004h]
        mov     [ebp+004h],eax
        push    ebp
        call    _desinit
        add     esp,+004h
        mov     ecx,00000040h
        mov     ebx,00000000h
$L31:   mov     eax,[ebx+$S29]
        xor     ebx,+010h
        mov     edx,[ebx+$S29]
        xor     ebx,+010h
        xor     eax,edx
        mov     [ebx+$S30],eax
        add     ebx,+004h
        loop    short $L31
        mov     ecx,00000040h
        mov     ebx,00000000h
$L34:   mov     eax,[ebx+$S32]
        xor     ebx,+004h
        mov     edx,[ebx+$S32]
        xor     ebx,+004h
        xor     eax,edx
        mov     [ebx+$S33],eax
        add     ebx,+004h
        loop    short $L34
        mov     ecx,00000040h
        mov     ebx,00000000h
$L37:   mov     eax,[ebx+$S35]
        xor     ebx,+010h
        mov     edx,[ebx+$S35]
        xor     ebx,+010h
        xor     eax,edx
        mov     [ebx+$S36],eax
        add     ebx,+004h
        loop    short $L37
        mov     ebx,$S25
        mov     esi,[ebx]
        mov     edi,[ebx+004h]
        call    $L38
        mov     $S39,esi
        mov     $S40,edi
        mov     eax,40104100h
        and     eax,edi
        mov     $S41,eax
        xor     eax,40104100h
        mov     $S42,eax
        mov     eax,00420082h
        and     eax,edi
        mov     $S43,eax
        xor     eax,00420082h
        mov     $S44,eax
        xor     esi,-001h
        xor     edi,-001h
        mov     $S45,esi
        mov     $S46,edi
        and     edi,20080820h
        mov     $S47,edi
        mov     edi,$S40
        and     edi,20080820h
        mov     $S48,edi
        mov     ebx,$S26
        mov     esi,[ebx]
        mov     edi,[ebx+004h]
        mov     ebx,$S24
        xor     esi,[ebx]
        xor     edi,[ebx+004h]
        call    $L38
        mov     $S49,esi
        mov     $S50,edi
        xor     esi,-001h
        xor     edi,-001h
        mov     $S51,esi
        mov     $S52,edi
        xor     ebx,ebx
        xor     ecx,ecx
        cmp     dword ptr $S12,+001h
        jnz     short $L53
        call    $L54
        jmp     $L55
$L53:   cmp     dword ptr $S12,+002h
        jnz     short $L56
        call    $L57
        jmp     $L55
$L56:   cmp     dword ptr $S12,+003h
        jnz     short $L58
        call    $L59
        jmp     $L55
$L58:   cmp     dword ptr $S12,+004h
        jnz     short $L60
        call    $L61
        jmp     $L55
$L60:   cmp     dword ptr $S12,+005h
        jnz     short $L62
        call    $L63
        jmp     $L55
$L62:   cmp     dword ptr $S12,+006h
        jnz     short $L64
        call    $L65
        jmp     $L55
$L64:   cmp     dword ptr $S12,+007h
        jnz     short $L66
        call    $L67
        jmp     $L55
$L66:   cmp     dword ptr $S12,+008h
        jnz     short $L68
        call    $L69
        jmp     $L55
$L68:   cmp     dword ptr $S12,+009h
        jnz     short $L70
        call    $L71
        jmp     $L55
$L70:   cmp     dword ptr $S12,+00Ah
        jnz     short $L72
        call    $L73
        jmp     $L55
$L72:   cmp     dword ptr $S12,+00Bh
        jnz     short $L74
        call    $L75
        jmp     $L55
$L74:   cmp     dword ptr $S12,+00Ch
        jnz     short $L76
        call    $L77
        jmp     $L55
$L76:   cmp     dword ptr $S12,+00Dh
        jnz     short $L78
        call    $L79
        jmp     $L55

$L78:   cmp     dword ptr $S12,+00Eh
        jnz     short $L80
        call    $L81
        jmp     $L55
$L80:   cmp     dword ptr $S12,+00Fh
        jnz     short $L82
        call    $L83
        jmp     $L55
$L82:   cmp     dword ptr $S12,+010h
        jnz     short $L84
        call    $L85
        jmp     $L55
$L84:   cmp     dword ptr $S12,+011h
        jnz     short $L86
        call    $L87
        jmp     $L55
$L86:   cmp     dword ptr $S12,+012h
        jnz     short $L88
        call    $L89
        jmp     $L55
$L88:   cmp     dword ptr $S12,+013h
        jnz     short $L90
        call    $L91
        jmp     $L55
$L90:   cmp     dword ptr $S12,+014h
        jnz     short $L92
        call    $L93
        jmp     short $L55
$L92:   cmp     dword ptr $S12,+015h
        jnz     short $L94
        call    $L95
        jmp     short $L55
$L94:   cmp     dword ptr $S12,+016h
        jnz     short $L96
        call    $L97
        jmp     short $L55
$L96:   cmp     dword ptr $S12,+017h
        jnz     short $L98
        call    $L99
        jmp     short $L55
$L98:   cmp     dword ptr $S12,+018h
        jnz     short $L100
        call    $L101
        jmp     short $L55
$L100:  cmp     dword ptr $S12,+019h
        jnz     short $L102
        call    $L103
        jmp     short $L55
$L102:  cmp     dword ptr $S12,+01Ah
        jnz     short $L104
        call    $L105
        jmp     short $L55
$L104:  cmp     dword ptr $S12,+01Bh
        jnz     short $L106
        call    $L107
        jmp     short $L55
$L106:  call    $L108
$L55:   mov     edx,00000000h
        cmp     dword ptr $S21,+001h
        jnz     short $L109
        mov     eax,00000000h
        jmp     short $L110
$L109:  mov     eax,00000001h
$L110:  jmp     short $L4
$L116:  cmp     dword ptr $S21,+001h
        jnz     short $L111
        mov     eax,00000000h
        jmp     short $L4
$L111:  mov     eax,00000002h
$L4:    mov     edx,00000000h
        pop     ecx
        pop     ebx
        pop     ebp
        pop     edi
        pop     esi
        ret
$L114:  mov     ebp,$S15[ebp*4]
        xor     eax,eax
        xor     edx,edx
        xor     ecx,ecx
        mov     cl,[ebp]
        mov     dl,[ebp+002h]
        mov     al,[ebp+001h]
        inc     ebp
$L113:  mov     edi,$S112[edx*4]
        mov     esi,$S1[eax*4]
        add     ebp,+002h
        xor     esi,edi
        mov     dl,[ebp+001h]
        mov     $S1[eax*4],esi
        mov     al,[ebp]
        loop    short $L113
        xor     ecx,ecx
        ret
$L54:   call    near ptr $L57
        mov     ebp,00000001h
        call    near ptr $L114
$L57:   call    near ptr $L59
        mov     ebp,00000002h
        call    near ptr $L114
$L59:   call    near ptr $L61
        mov     ebp,00000003h
        call    near ptr $L114
$L61:   call    near ptr $L63
        mov     ebp,00000004h
        call    near ptr $L114
$L63:   call    near ptr $L65
        mov     ebp,00000005h
        call    $L114
$L65:   call    near ptr $L67
        mov     ebp,00000006h
        call    $L114
$L67:   call    near ptr $L69
        mov     ebp,00000007h
        call    $L114
$L69:   call    near ptr $L71
        mov     ebp,00000008h
        call    $L114
$L71:   call    near ptr $L73
        mov     ebp,00000009h
        call    $L114
$L73:   call    near ptr $L75
        mov     ebp,0000000Ah
        call    $L114
$L75:   call    near ptr $L77
        mov     ebp,0000000Bh
        call    $L114
$L77:   call    near ptr $L79
        mov     ebp,0000000Ch
        call    $L114
$L79:   call    near ptr $L81
        mov     ebp,0000000Dh
        call    $L114
$L81:   call    near ptr $L83
        mov     ebp,0000000Eh
        call    $L114
$L83:   call    near ptr $L85
        mov     ebp,0000000Fh
        call    $L114
$L85:   call    near ptr $L87
        mov     ebp,00000010h
        call    $L114
$L87:   call    near ptr $L89
        mov     ebp,00000011h
        call    $L114
$L89:   call    near ptr $L91
        mov     ebp,00000012h
        call    $L114
$L91:   call    near ptr $L93
        mov     ebp,00000013h
        call    $L114
$L93:   call    _bryd_continue
        or      eax,eax
        jnz     short $L115
        mov     esp,$S23
        jmp     $L116
$L115:  xor     ebx,ebx
        xor     ecx,ecx
        call    near ptr $L95
        mov     ebp,00000014h
        call    $L114
$L95:   call    near ptr $L97
        mov     ebp,00000015h
        call    $L114
$L97:   call    near ptr $L99
        mov     ebp,00000016h
        call    $L114
$L99:   call    near ptr $L101
        mov     ebp,00000017h
        call    $L114
$L101:  call    near ptr $L103
        mov     ebp,00000018h
        call    $L114
$L103:  cmp     dword ptr $S6,+001h
        jnz     short $L117
        jmp     short $L118
$L117:  call    near ptr $L105
        mov     ebp,00000019h
        call    $L114
$L105:  call    near ptr $L107
        mov     ebp,0000001Ah
        call    $L114
$L107:  call    $L108
        mov     ebp,0000001Bh
        call    $L114
        jmp     $L108
$L118:  call    $L119
        mov     eax,$S120
        mov     edx,$S121
        xor     eax,00000800h
        xor     edx,00080000h
        mov     $S120,eax
        mov     $S121,edx
        mov     eax,$S122
        mov     edx,$S123
        xor     eax,00000200h
        xor     edx,00010000h
        mov     $S122,eax
        mov     $S123,edx
        mov     eax,$S124
        mov     edx,$S125
        xor     eax,00000800h
        xor     edx,00020000h
        mov     $S124,eax
        mov     $S125,edx
        mov     eax,$S126
        mov     edx,$S127
        xor     eax,00008000h
        xor     edx,00000100h
        mov     $S126,eax
        mov     $S127,edx
        mov     eax,$S128
        mov     edx,$S129
        xor     eax,00040000h
        xor     edx,+020h
        mov     $S128,eax
        mov     $S129,edx
        mov     eax,$S130
        mov     ebp,$S131
        xor     eax,00000400h
        mov     edx,$S132
        xor     ebp,00000080h
        mov     $S130,eax

        xor     edx,00001000h
        mov     $S131,ebp
        mov     $S132,edx
$L119:  call    $L133
        mov     eax,$S134
        mov     edx,$S135
        xor     eax,+010h
        xor     edx,00000800h
        mov     $S134,eax
        mov     $S135,edx
        mov     eax,$S122
        mov     edx,$S136
        xor     eax,00020000h
        xor     edx,+008h
        mov     $S122,eax
        mov     $S136,edx
        mov     eax,$S137
        mov     edx,$S138
        xor     eax,00004000h
        xor     edx,+040h
        mov     $S137,eax
        mov     $S138,edx
        mov     eax,$S139
        mov     edx,$S126
        xor     eax,00002000h
        xor     edx,+020h
        mov     $S139,eax
        mov     $S126,edx
        mov     eax,$S140
        mov     edx,$S141
        xor     eax,00000400h
        xor     edx,00000080h
        mov     $S140,eax
        mov     $S141,edx
        mov     eax,$S128
        mov     edx,$S129
        xor     eax,00000800h
        xor     edx,+040h
        mov     $S128,eax
        mov     $S129,edx
        mov     eax,$S130
        mov     edx,$S142
        xor     eax,00004000h
        xor     edx,00010000h
        mov     $S130,eax
        mov     $S142,edx
$L133:  call    $L108
        mov     eax,$S120
        mov     edx,$S143
        xor     eax,00000100h
        xor     edx,00004000h
        mov     $S120,eax
        mov     $S143,edx
        mov     eax,$S121
        mov     edx,$S144
        xor     eax,+040h
        xor     edx,00002000h
        mov     $S121,eax
        mov     $S144,edx
        mov     eax,$S123
        mov     edx,$S137
        xor     eax,00000400h
        xor     edx,00008000h
        mov     $S123,eax
        mov     $S137,edx
        mov     eax,$S124
        mov     edx,$S126
        xor     eax,00001000h
        xor     edx,+040h
        mov     $S124,eax
        mov     $S126,edx
        mov     eax,$S140
        mov     edx,$S145
        xor     eax,00004000h
        xor     edx,+010h
        mov     $S140,eax
        mov     $S145,edx
        mov     eax,$S146
        mov     edx,$S130
        xor     eax,00000080h
        xor     edx,+004h
        mov     $S146,eax
        mov     $S130,edx
        mov     eax,$S131
        mov     edx,$S132
        xor     eax,00008000h
        xor     edx,+008h
        mov     $S131,eax
        mov     $S132,edx
$L108:  mov     eax,$S17
        mov     edx,$S144
        and     edx,00000800h
        cmp     eax,edx
        jz      short $L147
        xor     eax,00000800h
        mov     $S17,eax
        mov     eax,$S146
        mov     edx,$S130
        xor     eax,00000800h
        xor     edx,+040h
        mov     $S146,eax
        mov     $S130,edx
        mov     eax,$S131
        mov     edx,$S142
        xor     eax,00004000h
        xor     edx,00000200h
        mov     $S131,eax
        mov     $S142,edx
$L147:  mov     eax,$S18
        mov     edx,$S135
        and     edx,10000000h
        cmp     eax,edx
        jz      short $L148
        xor     eax,10000000h
        mov     $S18,eax
        mov     eax,$S130
        mov     ebp,$S149
        xor     eax,00080000h
        mov     edx,$S132
        xor     ebp,+002h
        mov     $S130,eax
        xor     edx,04000000h
        mov     $S149,ebp
        mov     $S132,edx
$L148:  mov     esi,$S39
        mov     edi,$S40
        mov     eax,$S132
        xor     eax,edi
        mov     edx,$S142
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     esi,ebp
        mov     ebp,[ecx+$S152]
        xor     ecx,+020h
        xor     esi,ebp
        mov     bl,ah
        mov     edi,[ecx+$S152]
        and     eax,000000FFh
        xor     ebp,edi
        mov     $S153,ebp
        mov     cl,dh
        mov     edi,[ecx+$S32]
        xor     esi,edi
        and     edx,000000FFh
        mov     edi,[ebx+$S154]
        xor     ebx,+004h
        xor     esi,edi
        mov     ecx,[eax+$S155]
        xor     esi,ecx
        mov     ebx,[ebx+$S154]
        mov     ecx,[edx+$S35]
        xor     ebx,edi
        xor     esi,ecx
        mov     $S156,ebx
        xor     ebx,ebx
        xor     ecx,ecx
        mov     $S157,esi
        mov     esi,$S45
        mov     edi,$S46
        mov     eax,$S132
        xor     eax,edi
        mov     edx,$S142
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     esi,ebp
        mov     ebp,[ecx+$S152]
        xor     ecx,+020h
        xor     esi,ebp
        mov     bl,ah
        and     eax,000000FFh
        mov     edi,[ecx+$S152]
        xor     ecx,ecx
        xor     ebp,edi
        mov     cl,dh
        mov     edi,[ecx+$S32]
        xor     esi,edi
        and     edx,000000FFh
        mov     edi,[ebx+$S154]
        xor     ebx,+004h
        xor     esi,edi
        mov     ecx,[eax+$S155]
        xor     esi,ecx
        mov     ebx,[ebx+$S154]
        mov     ecx,[edx+$S35]
        xor     ebx,edi
        xor     esi,ecx
        mov     $S158,ebx
        mov     $S159,ebp

        mov     $S160,esi
        xor     ebx,ebx
        xor     ecx,ecx
        call    $L161
        mov     eax,$S120
        mov     edx,$S162
        xor     eax,+004h
        xor     edx,00800000h
        mov     $S120,eax
        mov     $S162,edx
        mov     eax,$S121
        mov     edx,$S122
        xor     eax,+008h
        xor     edx,02000000h
        mov     $S121,eax
        mov     $S122,edx
        mov     eax,$S136
        mov     edx,$S137
        xor     eax,04000000h
        xor     edx,00400000h
        mov     $S136,eax
        mov     $S137,edx
        mov     eax,$S124
        mov     edx,$S125
        xor     eax,10000000h
        xor     edx,01000000h
        mov     $S124,eax
        mov     $S125,edx
        mov     eax,$S126
        mov     edx,$S127
        xor     eax,80000000h
        xor     edx,80000000h
        mov     $S126,eax
        mov     $S127,edx
        mov     eax,$S141
        mov     edx,$S146
        xor     eax,00200000h
        xor     edx,08000000h
        mov     $S141,eax
        mov     $S146,edx
        mov     eax,$S130
        mov     edx,$S149
        xor     eax,08000000h
        xor     edx,04000000h
        mov     $S130,eax
        mov     $S149,edx
$L161:  mov     esi,$S51
        mov     edi,$S52
        mov     eax,$S134
        xor     eax,edi
        mov     edx,$S120
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     ebx,+010h
        xor     esi,ebp
        mov     edi,[ecx+$S152]
        mov     cl,dh
        xor     esi,edi
        mov     edi,[ebx+$S29]
        xor     edi,ebp
        xor     ebx,ebx
        mov     $S163,edi
        mov     bl,ah
        and     eax,000000FFh
        mov     ebp,[ebx+$S154]
        xor     esi,ebp
        mov     ebp,[ecx+$S32]
        xor     esi,ebp
        xor     ecx,+004h
        and     edx,000000FFh
        mov     edi,[eax+$S155]
        xor     esi,edi
        mov     edi,[ecx+$S32]
        mov     eax,[edx+$S35]
        xor     edx,+010h
        xor     edi,ebp
        xor     ecx,ecx
        mov     $S164,edi
        mov     edx,[edx+$S35]
        xor     esi,eax
        xor     edx,eax
        mov     $S165,edx
        mov     $S166,esi
        mov     esi,$S49
        mov     edi,$S50
        mov     eax,$S134
        xor     eax,edi
        mov     edx,$S120
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     ebx,+010h
        xor     esi,ebp
        mov     edi,[ecx+$S152]
        mov     cl,dh
        xor     esi,edi
        mov     edi,[ebx+$S29]
        xor     edi,ebp
        xor     ebx,ebx
        mov     $S167,edi
        mov     bl,ah
        and     eax,000000FFh
        mov     ebp,[ebx+$S154]
        xor     esi,ebp
        mov     ebp,[ecx+$S32]
        xor     esi,ebp
        xor     ecx,+004h
        and     edx,000000FFh
        mov     edi,[eax+$S155]
        xor     esi,edi
        mov     edi,[ecx+$S32]
        mov     eax,[edx+$S35]
        xor     edx,+010h
        xor     edi,ebp
        xor     ecx,ecx
        mov     $S168,edi
        mov     edx,[edx+$S35]
        xor     esi,eax
        xor     edx,eax
        mov     $S169,edx
        mov     $S170,esi
        call    $L171
        mov     eax,$S120
        mov     edx,$S143
        xor     eax,+001h
        xor     edx,08000000h
        mov     $S120,eax
        mov     $S143,edx
        mov     eax,$S135
        mov     edx,$S122
        xor     eax,08000000h
        xor     edx,04000000h
        mov     $S135,eax
        mov     $S122,edx
        mov     eax,$S123
        mov     edx,$S172
        xor     eax,+004h
        xor     edx,20000000h
        mov     $S123,eax
        mov     $S172,edx
        mov     eax,$S124
        mov     edx,$S125
        xor     eax,00080000h
        xor     edx,+002h
        mov     $S124,eax
        mov     $S125,edx
        mov     eax,$S126
        mov     edx,$S127
        xor     eax,04000000h
        xor     edx,00400000h
        mov     $S126,eax
        mov     $S127,edx
        mov     eax,$S141
        mov     edx,$S128
        xor     eax,10000000h
        xor     edx,01000000h
        mov     $S141,eax
        mov     $S128,edx
        mov     eax,$S129
        mov     ebp,$S130
        xor     eax,00040000h
        mov     edx,$S149
        xor     ebp,40000000h
        mov     $S129,eax
        xor     edx,00800000h
        mov     $S130,ebp
        mov     $S149,edx
        mov     eax,$S170
        mov     ebp,$S167
        mov     edx,$S166
        xor     eax,ebp
        mov     ebp,$S163
        mov     $S170,eax
        xor     edx,ebp
        mov     $S166,edx
$L171:  mov     ebx,$S158
        mov     ebp,$S159
        mov     esi,$S160
        and     ebx,0FC00000h
        shr     ebx,14h
        and     ebp,0FC00000h
        shr     ebp,14h
        mov     edx,$S149
        xor     edx,esi
        and     esi,08200401h
        and     edx,0FC00000h
        mov     $S173,esi
        shr     edx,14h
        mov     ecx,$S156
        mov     edi,$S47
        and     ecx,0FC00000h
        mov     eax,[edx+$S32]

        xor     edx,ebx
        shr     ecx,14h
        xor     eax,edi
        mov     esi,[edx+$S32]
        xor     edx,ebp
        xor     esi,edi
        mov     $S174,eax
        mov     eax,[edx+$S32]
        xor     edx,ebx
        mov     $S175,esi
        xor     eax,edi
        mov     ebx,[edx+$S32]
        mov     edx,$S149
        mov     esi,$S157
        xor     ebx,edi
        xor     edx,esi
        mov     $S176,eax
        and     edx,0FC00000h
        mov     $S177,ebx
        shr     edx,14h
        mov     eax,esi
        and     eax,08200401h
        mov     ebp,edx
        xor     ebp,ecx
        mov     edi,$S153
        mov     $S178,eax
        and     edi,0FC00000h
        shr     edi,14h
        mov     eax,$S48
        mov     ebx,[ebp+$S32]
        xor     ebp,edi
        xor     ebx,eax
        mov     esi,[edx+$S32]
        mov     $S179,ebx
        xor     esi,eax
        mov     ebx,[ebp+$S32]
        xor     ebp,ecx
        xor     ebx,eax
        xor     ecx,ecx
        mov     $S180,ebx
        mov     edx,[ebp+$S32]
        mov     $S181,esi
        xor     edx,eax
        mov     $S182,edx
        xor     ebx,ebx
        call    $L183
        mov     eax,$S120
        mov     edx,$S162
        xor     eax,00010000h
        xor     edx,+010h
        mov     $S120,eax
        mov     $S162,edx
        mov     eax,$S121
        mov     edx,$S144
        xor     eax,00000080h
        xor     edx,+004h
        mov     $S121,eax
        mov     $S144,edx
        mov     eax,$S136
        mov     edx,$S137
        xor     eax,00008000h
        xor     edx,00000100h
        mov     $S136,eax
        mov     $S137,edx
        mov     eax,$S125
        mov     edx,$S126
        xor     eax,00040000h
        xor     edx,00002000h
        mov     $S125,eax
        mov     $S126,edx
        mov     eax,$S127
        mov     edx,$S184
        xor     eax,00000400h
        xor     edx,00008000h
        mov     $S127,eax
        mov     $S184,edx
        mov     eax,$S145
        mov     ebp,$S185
        xor     eax,00001000h
        mov     edx,$S149
        xor     ebp,00080000h
        mov     $S145,eax
        xor     edx,00000200h
        mov     $S185,ebp
        mov     $S149,edx
        mov     eax,$S170
        mov     ebp,$S169
        mov     edx,$S166
        xor     eax,ebp
        mov     ebp,$S165
        mov     $S170,eax
        xor     edx,ebp
        mov     $S166,edx
$L183:  call    $L186
        mov     eax,$S120
        mov     edx,$S162
        xor     eax,00400000h
        xor     edx,00400000h
        mov     $S120,eax
        mov     $S162,edx
        mov     eax,$S121
        mov     edx,$S144
        xor     eax,40000000h
        xor     edx,80000000h
        mov     $S121,eax
        mov     $S144,edx
        mov     eax,$S123
        mov     edx,$S172
        xor     eax,80000000h
        xor     edx,00200000h
        mov     $S123,eax
        mov     $S172,edx
        mov     eax,$S125
        mov     edx,$S140
        xor     eax,08000000h
        xor     edx,00100000h
        mov     $S125,eax
        mov     $S140,edx
        mov     eax,$S145
        mov     edx,$S146
        xor     eax,00800000h
        xor     edx,+008h
        mov     $S145,eax
        mov     $S146,edx
        mov     eax,$S185
        mov     edx,$S131
        xor     eax,02000000h
        xor     edx,04000000h
        mov     $S185,eax
        mov     $S131,edx
        mov     eax,$S170
        mov     ebp,$S168
        mov     edx,$S166
        xor     eax,ebp
        mov     ebp,$S164
        mov     $S170,eax
        xor     edx,ebp
        mov     $S166,edx
$L186:  mov     dword ptr $S187,offset $S181
        xor     ebx,ebx
        xor     ecx,ecx
        mov     esi,$S166
        mov     edi,$S52
        mov     eax,$S162
        xor     eax,esi
        mov     edx,$S143
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     edi,ebp
        mov     $S188,ebx
        mov     $S189,ebp
        mov     ebp,[ebx+$S30]
        mov     $S190,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     edi,ebp
        mov     ebp,[ecx+$S32]
        xor     edi,ebp
        mov     ebp,[ecx+$S33]
        mov     $S191,ebp
        mov     ebp,[eax+$S155]
        xor     edi,ebp
        mov     ebp,[edx+$S35]
        xor     edi,ebp
        mov     ebp,[edx+$S36]
        mov     $S192,ebp
        mov     $S193,edi
        mov     esi,$S170
        mov     edi,$S50
        mov     eax,$S162
        xor     eax,esi
        mov     edx,$S143
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]

        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     edi,ebp
        mov     $S194,ebx
        mov     $S195,ebp
        mov     ebp,[ebx+$S30]
        mov     $S196,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     edi,ebp
        mov     ebp,[ecx+$S32]
        xor     edi,ebp
        mov     ebp,[ecx+$S33]
        mov     $S197,ebp
        mov     ebp,[eax+$S155]
        xor     edi,ebp
        mov     ebp,[edx+$S35]
        xor     edi,ebp
        mov     ebp,[edx+$S36]
        mov     $S198,ebp
        mov     $S199,edi
        call    $L200
        mov     dword ptr $S187,offset $S182
        call    $L201
        mov     eax,$S135
        mov     ebp,$S122
        xor     eax,10000000h
        mov     edx,$S136
        xor     ebp,01000000h
        mov     $S135,eax
        xor     edx,00040000h
        mov     $S122,ebp
        mov     $S136,edx
        mov     eax,$S172
        mov     edx,$S138
        xor     eax,40000000h
        xor     edx,00800000h
        mov     $S172,eax
        mov     $S138,edx
        mov     eax,$S125
        mov     edx,$S202
        xor     eax,+001h
        xor     edx,08000000h
        mov     $S125,eax
        mov     $S202,edx
        mov     eax,$S140
        mov     edx,$S184
        xor     eax,08000000h
        xor     edx,04000000h
        mov     $S140,eax
        mov     $S184,edx
        mov     eax,$S128
        mov     edx,$S129
        xor     eax,+004h
        xor     edx,20000000h
        mov     $S128,eax
        mov     $S129,edx
        mov     dword ptr $S187,offset $S180
        mov     esi,$S193
        mov     ebp,$S191
        mov     edi,$S199
        xor     esi,ebp
        mov     ebp,$S197
        mov     $S193,esi
        xor     edi,ebp
        mov     esi,$S170
        mov     $S199,edi
        call    $L200
        mov     dword ptr $S187,offset $S179
        call    $L201
        mov     eax,$S143
        mov     edx,$S135
        xor     eax,+004h
        xor     edx,20000000h
        mov     $S143,eax
        mov     $S135,edx
        mov     eax,$S144
        mov     edx,$S123
        xor     eax,00080000h
        xor     edx,+002h
        mov     $S144,eax
        mov     $S123,edx
        mov     eax,$S124
        mov     edx,$S125
        xor     eax,00400000h
        xor     edx,40000000h
        mov     $S124,eax
        mov     $S125,edx
        mov     eax,$S202
        mov     edx,$S140
        xor     eax,01000000h
        xor     edx,00040000h
        mov     $S202,eax
        mov     $S140,edx
        mov     eax,$S141
        mov     edx,$S128
        xor     eax,40000000h
        xor     edx,00800000h
        mov     $S141,eax
        mov     $S128,edx
        mov     eax,$S146
        mov     ebp,$S131
        xor     eax,+001h
        mov     edx,$S142
        xor     ebp,00100000h
        mov     $S146,eax
        xor     edx,04000000h
        mov     $S131,ebp
        mov     $S142,edx
        mov     ebx,$S188
        mov     edx,$S194
        xor     ebx,+040h
        mov     eax,$S189
        xor     edx,+040h
        mov     esi,$S193
        xor     esi,eax
        mov     ebp,[ebx+$S29]
        xor     esi,ebp
        mov     ebp,[ebx+$S30]
        mov     edi,$S199
        mov     eax,$S195
        mov     $S190,ebp
        xor     edi,eax
        xor     ebx,ebx
        mov     ebp,[edx+$S29]
        xor     edi,ebp
        mov     ebp,[edx+$S30]
        mov     $S196,ebp
        mov     $S193,esi
        mov     $S199,edi
        mov     esi,$S170
        call    $L200
        mov     dword ptr $S187,offset $S180
        call    $L201
        mov     eax,$S135
        mov     ebp,$S122
        xor     eax,10000000h
        mov     edx,$S136
        xor     ebp,01000000h
        mov     $S135,eax
        xor     edx,00040000h
        mov     $S122,ebp
        mov     $S136,edx
        mov     eax,$S172
        mov     edx,$S138
        xor     eax,40000000h
        xor     edx,00800000h
        mov     $S172,eax
        mov     $S138,edx
        mov     eax,$S125
        mov     edx,$S202
        xor     eax,+001h
        xor     edx,08000000h
        mov     $S125,eax
        mov     $S202,edx
        mov     eax,$S140
        mov     edx,$S184
        xor     eax,08000000h
        xor     edx,04000000h
        mov     $S140,eax
        mov     $S184,edx

        mov     eax,$S128
        mov     edx,$S129
        xor     eax,+004h
        xor     edx,20000000h
        mov     $S128,eax
        mov     $S129,edx
        mov     dword ptr $S187,offset $S182
        mov     esi,$S193
        mov     ebp,$S191
        mov     edi,$S199
        xor     esi,ebp
        mov     ebp,$S197
        mov     $S193,esi
        xor     edi,ebp
        mov     esi,$S170
        mov     $S199,edi
        call    $L200
        mov     dword ptr $S187,offset $S181
$L201:  mov     eax,$S144
        mov     ebp,$S123
        xor     eax,00000800h
        mov     edx,$S172
        xor     ebp,00020000h
        mov     $S144,eax
        xor     edx,+008h
        mov     $S123,ebp
        mov     $S172,edx
        mov     eax,$S138
        mov     edx,$S125
        xor     eax,00004000h
        xor     edx,+040h
        mov     $S138,eax
        mov     $S125,edx
        mov     eax,$S202
        mov     edx,$S140
        xor     eax,00040000h
        xor     edx,+020h
        mov     $S202,eax
        mov     $S140,edx
        mov     eax,$S141
        mov     edx,$S145
        xor     eax,00000400h
        xor     edx,00000080h
        mov     $S141,eax
        mov     $S145,edx
        mov     esi,$S193
        mov     ebp,$S192
        mov     edi,$S199
        xor     esi,ebp
        mov     ebp,$S198
        mov     $S193,esi
        xor     edi,ebp
        mov     esi,$S170
        mov     $S199,edi
$L200:  call    $L203
        mov     eax,$S144
        mov     ebp,$S138
        xor     eax,00100000h
        mov     edx,$S172
        xor     ebp,+008h
        mov     $S144,eax
        xor     edx,00800000h
        mov     $S138,ebp
        mov     $S172,edx
        mov     eax,$S125
        mov     edx,$S202
        xor     eax,02000000h
        xor     edx,+002h
        mov     $S125,eax
        mov     $S202,edx
        mov     eax,$S141
        mov     ebp,$S128
        xor     eax,00400000h
        mov     edx,$S129
        xor     ebp,40000000h
        mov     $S141,eax
        xor     edx,80000000h
        mov     $S128,ebp
        mov     $S129,edx
        mov     eax,$S193
        mov     ebp,$S190
        mov     esi,$S166
        xor     eax,ebp
        mov     ebp,$S196
        mov     edi,$S199
        mov     $S193,eax
        xor     edi,ebp
        mov     esi,$S170
        mov     $S199,edi
$L203:  xor     ebx,ebx
        mov     eax,$S135
        xor     eax,edi
        mov     edx,$S121
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     esi,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     esi,ebp
        mov     ebp,[ecx+$S32]
        xor     esi,ebp
        mov     ebp,[eax+$S155]
        xor     esi,ebp
        mov     ebp,[edx+$S35]
        xor     esi,ebp
        mov     eax,$S144
        xor     eax,esi
        mov     edx,$S122
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     edi,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     edi,ebp
        mov     ebp,[ecx+$S32]
        xor     edi,ebp
        mov     ebp,[eax+$S155]
        xor     edi,ebp
        mov     ebp,[edx+$S35]
        xor     edi,ebp
        mov     eax,$S136
        xor     eax,edi
        mov     edx,$S123
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     esi,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     esi,ebp
        mov     ebp,[ecx+$S32]
        xor     esi,ebp
        mov     ebp,[eax+$S155]
        xor     esi,ebp
        mov     ebp,[edx+$S35]
        xor     esi,ebp
        mov     eax,$S172
        xor     eax,esi
        mov     edx,$S137
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     edi,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     edi,ebp
        mov     ebp,[ecx+$S32]
        xor     edi,ebp
        mov     ebp,[eax+$S155]
        xor     edi,ebp
        mov     ebp,[edx+$S35]
        xor     edi,ebp
        mov     eax,$S124
        xor     eax,edi
        mov     edx,$S138
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]

        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     esi,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     esi,ebp
        mov     ebp,[ecx+$S32]
        xor     esi,ebp
        mov     ebp,[eax+$S155]
        xor     esi,ebp
        mov     ebp,[edx+$S35]
        xor     esi,ebp
        mov     eax,$S139
        xor     eax,esi
        mov     edx,$S125
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     edi,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     edi,ebp
        mov     ebp,[ecx+$S32]
        xor     edi,ebp
        mov     ebp,[eax+$S155]
        xor     edi,ebp
        mov     ebp,[edx+$S35]
        xor     edi,ebp
        mov     eax,$S126
        xor     eax,edi
        mov     edx,$S202
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     esi,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     esi,ebp
        mov     ebp,[ecx+$S32]
        xor     esi,ebp
        mov     ebp,[eax+$S155]
        xor     esi,ebp
        mov     ebp,[edx+$S35]
        xor     esi,ebp
        mov     eax,$S140
        xor     eax,esi
        mov     edx,$S127
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     edi,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     edi,ebp
        mov     ebp,[ecx+$S32]
        xor     edi,ebp
        mov     ebp,[eax+$S155]
        xor     edi,ebp
        mov     ebp,[edx+$S35]
        xor     edi,ebp
        mov     eax,$S141
        xor     eax,edi
        mov     edx,$S184
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     esi,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     esi,ebp
        mov     ebp,[ecx+$S32]
        xor     esi,ebp
        mov     ebp,[eax+$S155]
        xor     esi,ebp
        mov     ebp,[edx+$S35]
        xor     esi,ebp
        mov     eax,$S145
        xor     eax,esi
        mov     edx,$S128
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     cl,ah
        rol     edx,04h
        mov     $S204,eax
        mov     bl,dl
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     edi,ebp
        mov     ebp,[ebx+$S29]
        xor     edi,ebp
        mov     cl,dh
        mov     $S205,edx
        mov     bl,ah
        mov     ebp,[ecx+$S152]
        mov     cl,byte ptr $S205 + 00002h
        xor     edi,ebp
        and     eax,000000FFh
        mov     ebp,[ecx+$S35]
        mov     edx,$S146
        xor     edi,ebp
        mov     ebp,[ebx+$S154]
        xor     edi,ebp
        mov     ebp,[eax+$S155]
        xor     edi,ebp
        mov     ebp,$S20
        xor     edx,edi
        mov     eax,$S187
        and     edx,0FC00000h
        mov     cl,byte ptr $S205 + 00003h
        shr     edx,14h
        mov     eax,[ebp+eax]
        mov     $S206,esi
        and     esi,20080820h
        xor     eax,esi
        mov     edx,[edx+$S32]
        cmp     edx,eax
        jz      short $L207
$L210:  mov     edi,$S193
        or      ebp,ebp
        jz      short $L208
        mov     dword ptr $S20,00000000h
        ret
$L208:  mov     dword ptr $S20,00000018h
        mov     esi,$S166
        jmp     $L203
$L207:  mov     bl,byte ptr $S204
        mov     ebp,[ecx+$S32]
        xor     edi,ebp
        mov     ebp,[ebx+$S150]
        xor     edi,ebp
        mov     esi,$S206
        mov     eax,$S129
        xor     eax,edi
        mov     edx,$S146
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     $S205,edx
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     esi,ebp
        mov     ebp,[ecx+$S151]
        mov     bl,ah
        and     eax,000000FFh
        xor     esi,ebp
        mov     cl,dh
        mov     ebp,[eax+$S155]
        mov     edx,$S185
        xor     esi,ebp
        mov     ebp,[ecx+$S32]
        xor     esi,ebp
        mov     ebp,[ebx+$S154]
        xor     esi,ebp
        mov     ebp,$S20
        xor     edx,esi
        mov     $S206,edi
        shr     edx,0Ch
        and     edi,08200401h
        mov     ebp,[ebp+$S178]
        and     edx,000000FCh
        xor     ebp,edi
        mov     edx,[edx+$S35]
        cmp     edx,ebp
        jz      short $L209
        mov     ebp,$S20
        jmp     $L210
$L209:  mov     bl,byte ptr $S205 + 00002h
        mov     cl,byte ptr $S205 + 00001h
        mov     eax,$S17
        mov     edx,$S144
        and     edx,00000800h
        cmp     eax,edx
        jz      short $L211
        xor     eax,00000800h
        mov     $S17,eax
        mov     eax,$S146

        mov     edx,$S130
        xor     eax,00000800h
        xor     edx,+040h
        mov     $S146,eax
        mov     $S130,edx
        mov     eax,$S131
        mov     edx,$S142
        xor     eax,00004000h
        xor     edx,00000200h
        mov     $S131,eax
        mov     $S142,edx
        xor     ecx,00000080h
$L211:  mov     edi,$S206
        mov     ebp,[ebx+$S35]
        xor     esi,ebp
        mov     ebp,[ecx+$S152]
        xor     ecx,ecx
        xor     esi,ebp
        mov     eax,$S18
        mov     edx,$S135
        and     edx,10000000h
        cmp     eax,edx
        jz      short $L212
        xor     eax,10000000h
        mov     $S18,eax
        mov     eax,$S130
        mov     ebp,$S149
        xor     eax,00080000h
        mov     edx,$S132
        xor     ebp,+002h
        mov     $S130,eax
        xor     edx,04000000h
        mov     $S149,ebp
        mov     $S132,edx
$L212:  mov     eax,$S19
        mov     edx,$S144
        and     edx,00100000h
        cmp     eax,edx
        jz      short $L213
        xor     eax,00100000h
        mov     $S19,eax
        mov     eax,$S185
        mov     ebp,$S131
        xor     eax,80000000h
        mov     edx,$S142
        xor     ebp,00200000h
        mov     $S185,eax
        xor     edx,00800000h
        mov     $S131,ebp
        mov     $S142,edx
$L213:  mov     eax,$S130
        xor     eax,esi
        mov     edx,$S185
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     edi,ebp
        mov     ebp,[ecx+$S152]
        mov     cl,dh
        and     edx,000000FFh
        xor     edi,ebp
        mov     ebp,$S20
        mov     ebp,[ebp+$S43]
        mov     $S206,esi
        mov     edx,[edx+$S35]
        xor     edi,edx
        mov     edx,[ecx+$S32]
        xor     edi,edx
        mov     edx,$S131
        xor     edx,edi
        and     esi,00420082h
        shr     edx,1Ah
        xor     ebp,esi
        mov     edx,$S154[edx*4]
        mov     esi,00000001h
        cmp     edx,ebp
        jz      short $L214
        mov     ebp,$S20
        jmp     $L210
$L214:  mov     bl,ah
        and     eax,000000FFh
        mov     esi,$S206
        mov     ebp,[eax+$S155]
        xor     edi,ebp
        mov     ebp,[ebx+$S154]
        xor     edi,ebp
        mov     eax,$S131
        xor     eax,edi
        mov     edx,$S149
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     esi,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     esi,ebp
        mov     ebp,[ecx+$S32]
        xor     esi,ebp
        mov     ebp,[eax+$S155]
        xor     esi,ebp
        mov     ebp,[edx+$S35]
        xor     esi,ebp
        mov     ebp,$S20
        mov     edx,[ebp+$S40]
        cmp     esi,edx
        jnz     $L215
        mov     eax,$S132
        xor     eax,esi
        mov     edx,$S142
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     edi,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     edi,ebp
        mov     ebp,[ecx+$S32]
        xor     edi,ebp
        mov     ebp,[eax+$S155]
        xor     edi,ebp
        mov     ebp,[edx+$S35]
        xor     edi,ebp
        mov     ebp,$S20
        mov     eax,[ebp+$S39]
        cmp     edi,eax
        jnz     short $L215
        call    $L216
        mov     ebp,$S20
        or      ebp,ebp
        jz      short $L217
        xor     esi,-001h
        xor     edi,-001h
$L217:  call    $L218
$L215:  xor     ebx,ebx
        xor     ecx,ecx
        mov     ebp,$S20
        jmp     $L210
$L238:  xor     ebx,ebx
        xor     edi,edi
        mov     ch,01h
        jmp     short $L219
$L226:  mov     cl,ds:[ebp]
        or      cl,cl
        jz      short $L220
        mov     esi,00000001h
        ror     esi,cl
        cmp     cl,20h
        jnbe    short $L221
        and     esi,eax
        jmp     short $L222
$L221:  and     esi,edx
$L222:  jz      short $L220
        cmp     ch,20h
        jnbe    short $L223
        add     ebx,+001h
        jmp     short $L220
$L223:  add     edi,+001h
$L220:  cmp     ch,20h
        jnc     short $L224
        add     ebx,ebx
        jmp     short $L225
$L224:  cmp     ch,20h
        jbe     short $L225
        cmp     ch,40h
        jnc     short $L225
        add     edi,edi
$L225:  inc     ebp
        inc     ch
$L219:  cmp     ch,40h
        jbe     short $L226
        mov     ecx,edi
        ret
$L38:   rol     edi,04h
        mov     ecx,esi
        xor     esi,edi
        and     esi,0F0F0F0F0h
        xor     ecx,esi
        xor     edi,esi
        ror     ecx,0Ch
        mov     esi,ecx
        xor     ecx,edi
        and     ecx,000FFFF0h
        xor     esi,ecx
        xor     edi,ecx
        rol     esi,0Eh
        mov     ecx,esi
        xor     esi,edi
        and     esi,33333333h
        xor     ecx,esi
        xor     edi,esi
        ror     ecx,06h
        mov     esi,ecx
        xor     ecx,edi
        and     ecx,0FF00FF0h
        xor     esi,ecx
        xor     edi,ecx
        rol     esi,07h
        mov     ecx,esi
        xor     esi,edi
        and     esi,55555555h
        xor     ecx,esi
        xor     edi,esi
        ror     edi,1
        mov     esi,ecx
        ret
$L216:  mov     ecx,0000001Ch
        mov     edx,offset $S227
        mov     esi,00000000h
        mov     edi,00000000h
$L229:  mov     eax,[edx]
        mov     ebx,[edx+004h]
        mov     ebp,[edx+008h]
        and     ebp,$S1[ebx*4]
        jz      short $L228
        or      esi,$S112[eax*4]
$L228:  add     edx,+00Ch
        loop    short $L229
        push    eax
        mov     eax,esi
        ror     ax,08h
        ror     eax,10h
        ror     ax,08h
        push    eax
        mov     eax,[esp+004h]
        pop     esi
        add     esp,+004h
        mov     ecx,0000001Ch
$L231:  mov     eax,[edx]
        sub     eax,+020h

        mov     ebx,[edx+004h]
        mov     ebp,[edx+008h]
        and     ebp,$S1[ebx*4]
        jz      short $L230
        or      edi,$S112[eax*4]
$L230:  add     edx,+00Ch
        loop    short $L231
        push    eax
        mov     eax,edi
        ror     ax,08h
        ror     eax,10h
        ror     ax,08h
        push    eax
        mov     eax,[esp+004h]
        pop     edi
        add     esp,+004h
        ret
$L218:  mov     dword ptr $S21,00000001h
        mov     $S232,esi
        mov     $S233,edi
        mov     esi,offset $S232
        mov     edi,offset $S234
        mov     ecx,00000008h
$L236:  mov     al,[esi]
        and     al,al
        jnp     short $L235
        xor     al,01h
$L235:  mov     [edi],al
        add     esi,+001h
        add     edi,+001h
        loop    short $L236
        push    offset $S234
        call    _bryd_key_found
        add     esp,+004h
        xor     ebx,ebx
        xor     ecx,ecx
        ret
_desinit:
        push    esi
        push    edi
        push    ebp
        push    ebx
        push    ecx
        mov     ebp,[esp+018h]
        mov     ebx,ebp
        mov     eax,[ebx]
        mov     edx,[ebx+004h]
        push    eax
        mov     eax,eax
        ror     ax,08h
        ror     eax,10h
        ror     ax,08h
        push    eax
        mov     eax,[esp+004h]
        pop     eax
        add     esp,+004h
        push    eax
        mov     eax,edx
        ror     ax,08h
        ror     eax,10h
        ror     ax,08h
        push    eax
        mov     eax,[esp+004h]
        pop     edx
        add     esp,+004h
        mov     ebp,offset $S237
        call    $L238
        mov     eax,ebx
        mov     edx,ecx
        mov     dword ptr $S239,00000001h
        mov     esi,offset $S134
        jmp     short $L240
$L244:  push    esi
        mov     ebp,offset $S241
        call    $L238
        cmp     dword ptr $S239,+002h
        jbe     short $L242
        cmp     dword ptr $S239,+009h
        jz      short $L242
        cmp     dword ptr $S239,+010h
        jz      short $L242
        mov     eax,ebx
        mov     edx,ecx
        mov     ebp,offset $S241
        call    $L238
$L242:  mov     eax,ebx
        mov     edx,ecx
        mov     ebp,offset $S243
        call    $L238
        pop     esi
        xchg    bl,bh
        rol     ebx,08h
        xchg    bh,cl
        ror     ecx,08h
        rol     bx,08h
        ror     cx,08h
        xchg    bh,cl
        ror     ecx,04h
        mov     [esi],ebx
        mov     [esi+004h],ecx
        rol     ecx,04h
        add     esi,+008h
        inc     dword ptr $S239
$L240:  cmp     dword ptr $S239,+010h
        jbe     short $L244
        push    ebp
        mov     esi,offset $S245
        mov     edi,00000000h
        mov     ebp,00000000h
        jmp     short $L246
$L252:  mov     ah,01h
        jmp     short $L247
$L251:  mov     ebx,00000000h
        test    ah,01h
        jz      short $L248
        lodsb
$L248:  shl     al,1
        jnc     short $L249
        mov     cl,[ebp+$S250]
        mov     edx,00000001h
        ror     edx,cl
        add     ebx,edx
$L249:  inc     ebp
        test    ebp,00000003h
        jnz     short $L248
        sub     ebp,+004h
        rol     ebx,03h
        mov     [edi+$S29],ebx
        inc     ah
        add     edi,+004h
$L247:  cmp     ah,40h
        jbe     short $L251
        add     ebp,+004h
$L246:  cmp     ebp,+020h
        jc      short $L252
        pop     ebp
        pop     ecx
        pop     ebx
        pop     ebp
        pop     edi
        pop     esi
        ret
$L9:    shl     edx,1
        shl     eax,1
        jnc     short $L253
        or      edx,+001h
$L253:  ret
_desencrypt:
        push    esi
        push    edi
        push    ebp
        push    ebx
        push    ecx
        mov     eax,[esp+018h]
        mov     esi,[eax]
        mov     edi,[eax+004h]
        mov     dword ptr $S254,00000000h
        mov     ebp,00000000h
        rol     edi,04h
        mov     ecx,esi
        xor     esi,edi
        and     esi,0F0F0F0F0h
        xor     ecx,esi
        xor     edi,esi
        ror     ecx,0Ch
        mov     esi,ecx
        xor     ecx,edi
        and     ecx,000FFFF0h
        xor     esi,ecx
        xor     edi,ecx
        rol     esi,0Eh
        mov     ecx,esi
        xor     esi,edi
        and     esi,33333333h
        xor     ecx,esi
        xor     edi,esi
        ror     ecx,06h
        mov     esi,ecx
        xor     ecx,edi
        and     ecx,0FF00FF0h
        xor     esi,ecx
        xor     edi,ecx
        rol     esi,07h
        mov     ecx,esi
        xor     esi,edi
        and     esi,55555555h
        xor     ecx,esi
        xor     edi,esi
        ror     edi,1
        mov     esi,ecx
        xor     ebx,ebx
        xor     ecx,ecx
$L255:  mov     eax,[ebp+$S134]
        xor     eax,edi
        mov     edx,[ebp+$S120]
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     esi,ebp
        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     esi,ebp
        mov     ebp,[ecx+$S32]
        xor     esi,ebp
        mov     ebp,[eax+$S155]
        xor     esi,ebp
        mov     ebp,[edx+$S35]
        xor     esi,ebp
        mov     ebp,$S254
        mov     eax,esi
        add     ebp,+008h
        mov     $S254,ebp
        mov     esi,edi
        mov     edi,eax
        cmp     ebp,00000080h
        jb      $L255
        rol     esi,1
        mov     ecx,edi
        xor     edi,esi
        and     edi,55555555h
        xor     esi,edi
        xor     edi,ecx
        ror     edi,07h
        mov     ecx,esi
        xor     esi,edi
        and     esi,0FF00FF0h
        xor     edi,esi
        xor     esi,ecx
        ror     esi,06h
        mov     ecx,edi
        xor     edi,esi
        and     edi,0CCCCCCCCh
        xor     esi,edi
        xor     edi,ecx
        rol     esi,0Eh
        mov     ecx,edi
        xor     edi,esi
        and     edi,0FFFF000h
        xor     esi,edi
        xor     edi,ecx
        ror     esi,0Ch
        mov     ecx,edi
        xor     edi,esi
        and     edi,0F0F0F0Fh
        xor     esi,edi
        xor     edi,ecx
        rol     edi,04h
        mov     eax,[esp+01Ch]
        mov     [eax],edi
        mov     [eax+004h],esi
        pop     ecx
        pop     ebx
        pop     ebp
        pop     edi
        pop     esi
        ret
_desdecrypt:
        push    esi
        push    edi
        push    ebp
        push    ebx
        push    ecx
        mov     eax,[esp+018h]
        mov     esi,[eax]
        mov     edi,[eax+004h]
        mov     dword ptr $S254,00000078h
        mov     ebp,00000078h
        rol     edi,04h
        mov     ecx,esi
        xor     esi,edi
        and     esi,0F0F0F0F0h
        xor     ecx,esi
        xor     edi,esi
        ror     ecx,0Ch
        mov     esi,ecx
        xor     ecx,edi
        and     ecx,000FFFF0h
        xor     esi,ecx
        xor     edi,ecx
        rol     esi,0Eh
        mov     ecx,esi
        xor     esi,edi
        and     esi,33333333h
        xor     ecx,esi
        xor     edi,esi
        ror     ecx,06h
        mov     esi,ecx
        xor     ecx,edi
        and     ecx,0FF00FF0h
        xor     esi,ecx
        xor     edi,ecx
        rol     esi,07h
        mov     ecx,esi
        xor     esi,edi
        and     esi,55555555h
        xor     ecx,esi
        xor     edi,esi
        ror     edi,1
        mov     esi,ecx
        xor     ebx,ebx
        xor     ecx,ecx
$L256:  mov     eax,[ebp+$S134]
        xor     eax,edi
        mov     edx,[ebp+$S120]
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S150]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S151]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S29]
        xor     esi,ebp

        mov     ebp,[ecx+$S152]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S154]
        xor     esi,ebp
        mov     ebp,[ecx+$S32]
        xor     esi,ebp
        mov     ebp,[eax+$S155]
        xor     esi,ebp
        mov     ebp,[edx+$S35]
        xor     esi,ebp
        mov     ebp,$S254
        mov     eax,esi
        sub     ebp,+008h
        mov     $S254,ebp
        mov     esi,edi
        mov     edi,eax
        cmp     ebp,+000h
        jnl     $L256
        rol     esi,1
        mov     ecx,edi
        xor     edi,esi
        and     edi,55555555h
        xor     esi,edi
        xor     edi,ecx
        ror     edi,07h
        mov     ecx,esi
        xor     esi,edi
        and     esi,0FF00FF0h
        xor     edi,esi
        xor     esi,ecx
        ror     esi,06h
        mov     ecx,edi
        xor     edi,esi
        and     edi,0CCCCCCCCh
        xor     esi,edi
        xor     edi,ecx
        rol     esi,0Eh
        mov     ecx,edi
        xor     edi,esi
        and     edi,0FFFF000h
        xor     esi,edi
        xor     edi,ecx
        ror     esi,0Ch
        mov     ecx,edi
        xor     edi,esi
        and     edi,0F0F0F0Fh
        xor     esi,edi
        xor     edi,ecx
        rol     edi,04h
        mov     eax,[esp+01Ch]
        mov     [eax],edi
        mov     [eax+004h],esi
        pop     ecx
        pop     ebx
        pop     ebp
        pop     edi
        pop     esi
        ret
_key_byte_to_hex:
        mov     dword ptr $S257,00000000h
        jmp     short $L258
_c_key_byte_to_hex:
        mov     dword ptr $S257,00000001h
$L258:  push    esi
        push    edi
        push    ebp
        push    ebx
        push    ecx
        mov     esi,[esp+018h]
        mov     edi,[esp+01Ch]
        mov     ecx,[esp+020h]
        mov     edx,00000000h
$L264:  mov     al,[esi]
        and     al,al
        jnp     short $L259
        mov     edx,00000001h
        xor     al,01h
$L259:  mov     ah,al
        and     ah,0Fh
        and     al,0F0h
        shr     al,04h
        cmp     ah,09h
        jbe     short $L260
        add     ah,37h
        jmp     short $L261
$L260:  add     ah,30h
$L261:  cmp     al,09h
        jbe     short $L262
        add     al,37h
        jmp     short $L263
$L262:  add     al,30h
$L263:  mov     [edi],ax
        add     esi,+001h
        add     edi,+002h
        loop    short $L264
        cmp     dword ptr $S257,+001h
        jnz     short $L265
        mov     byte ptr [edi],00h
$L265:  mov     eax,edx
        pop     ecx
        pop     ebx
        pop     ebp
        pop     edi
        pop     esi
        ret

_TEXT   ENDS
_DATA   SEGMENT

$S245   db      0E0h
        db      04Fh
        db      0D7h
        db      014h
        db      02Eh
        db      0F2h
        db      0BDh
        db      081h
        db      03Ah
        db      0A6h
        db      06Ch
        db      0CBh
        db      059h
        db      095h
        db      003h
        db      078h
        db      04Fh
        db      01Ch
        db      0E8h
        db      082h
        db      0D4h
        db      069h
        db      021h
        db      0B7h
        db      0F5h
        db      0CBh
        db      093h
        db      07Eh
        db      03Ah
        db      0A0h
        db      056h
        db      00Dh
        db      0F3h
        db      01Dh
        db      084h
        db      0E7h
        db      06Fh
        db      0B2h
        db      038h
        db      04Eh
        db      09Ch
        db      070h
        db      021h
        db      0DAh
        db      0C6h
        db      009h
        db      05Bh
        db      0A5h
        db      00Dh
        db      0E8h
        db      07Ah
        db      0B1h
        db      0A3h
        db      04Fh
        db      0D4h
        db      012h
        db      05Bh
        db      086h
        db      0C7h
        db      06Ch
        db      090h
        db      035h
        db      02Eh
        db      0F9h
        db      0ADh
        db      007h
        db      090h
        db      0E9h
        db      063h
        db      034h
        db      0F6h
        db      05Ah
        db      012h
        db      0D8h
        db      0C5h
        db      07Eh
        db      0BCh
        db      04Bh
        db      02Fh
        db      081h
        db      0D1h
        db      06Ah
        db      04Dh
        db      090h
        db      086h
        db      0F9h
        db      038h
        db      007h
        db      0B4h
        db      01Fh
        db      02Eh
        db      0C3h
        db      05Bh
        db      0A5h
        db      0E2h
        db      07Ch
        db      07Dh
        db      0D8h
        db      0EBh
        db      035h
        db      006h
        db      06Fh
        db      090h
        db      0A3h
        db      014h
        db      027h
        db      082h
        db      05Ch
        db      0B1h
        db      0CAh
        db      04Eh
        db      0F9h
        db      0A3h
        db      06Fh
        db      090h
        db      006h
        db      0CAh
        db      0B1h
        db      07Dh
        db      0D8h
        db      0F9h
        db      014h
        db      035h
        db      0EBh
        db      05Ch
        db      027h
        db      082h
        db      04Eh
        db      02Eh
        db      0CBh
        db      042h
        db      01Ch
        db      074h
        db      0A7h
        db      0BDh
        db      061h
        db      085h
        db      050h
        db      03Fh
        db      0FAh
        db      0D3h
        db      009h
        db      0E8h
        db      096h
        db      04Bh
        db      028h
        db      01Ch
        db      0B7h
        db      0A1h
        db      0DEh
        db      072h
        db      08Dh
        db      0F6h
        db      09Fh
        db      0C0h
        db      'Yj4'
        db      005h
        db      0E3h
        db      0CAh
        db      01Fh
        db      0A4h
        db      0F2h
        db      097h
        db      02Ch
        db      069h
        db      085h
        db      006h
        db      0D1h
        db      03Dh
        db      04Eh
        db      0E0h
        db      07Bh
        db      053h
        db      0B8h
        db      094h
        db      0E3h
        db      0F2h
        db      05Ch
        db      029h
        db      085h
        db      0CFh
        db      03Ah
        db      07Bh
        db      00Eh
        db      041h
        db      0A7h
        db      016h
        db      0D0h
        db      0B8h
        db      06Dh
        db      04Dh
        db      0B0h
        db      02Bh
        db      0E7h
        db      0F4h
        db      009h
        db      081h
        db      0DAh
        db      03Eh
        db      0C3h
        db      095h
        db      07Ch
        db      052h
        db      0AFh
        db      068h
        db      016h
        db      016h
        db      04Bh
        db      0BDh
        db      0D8h
        db      0C1h
        db      034h
        db      07Ah
        db      0E7h
        db      0A9h
        db      0F5h
        db      060h
        db      08Fh
        db      00Eh
        db      052h
        db      093h
        db      02Ch
        db      0D1h
        db      02Fh
        db      08Dh
        db      048h
        db      06Ah
        db      0F3h
        db      0B7h
        db      014h
        db      0ACh
        db      095h
        db      036h
        db      0EBh
        db      050h
        db      00Eh
        db      0C9h
        db      072h
        db      072h
        db      0B1h
        db      04Eh
        db      017h
        db      094h
        db      0CAh
        db      0E8h
        db      02Dh
        db      00Fh
        db      06Ch
        db      0A9h
        db      0D0h
        db      0F3h
        db      035h
        db      056h
        db      08Bh
$S250   db      009h
        db      011h
        db      017h
        db      01Fh
        db      00Dh
        db      01Ch
        db      002h
        db      012h
        db      018h
        db      010h
        db      01Eh
        db      006h
        db      01Ah
        db      014h
        db      00Ah
        db      001h
        db      008h
        db      00Eh
        db      019h
        db      003h
        db      004h
        db      01Dh
        db      00Bh
        db      013h
        db      020h
        db      00Ch
        db      016h
        db      007h
        db      005h
        db      01Bh
        db      00Fh
        db      015h
$S237   db      '91)!'
        db      019h
        db      011h
        db      009h
        db      001h
        db      ':2*"'
        db      01Ah
        db      012h
        db      00Ah
        db      002h
        db      ';3+#',01Bh
        db      013h
        db      00Bh
        db      003h
        db      '<4,$'
        db      '?7/'''
        db      01Fh
        db      017h
        db      00Fh
        db      007h
        db      '>6.&'
        db      01Eh
        db      016h
        db      00Eh
        db      006h
        db      '=5-%'
        db      01Dh
        db      015h
        db      00Dh
        db      005h
        db      01Ch
        db      014h
        db      00Ch
        db      004h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
$S241   db      002h
        db      003h
        db      004h
        db      005h
        db      006h
        db      007h
        db      008h
        db      009h
        db      00Ah
        db      00Bh
        db      00Ch
        db      00Dh
        db      00Eh
        db      00Fh
        db      010h
        db      011h
        db      012h
        db      013h
        db      014h
        db      015h
        db      016h
        db      017h
        db      018h
        db      019h
        db      01Ah
        db      01Bh
        db      01Ch
        db      001h
        db      01Eh
        db      01Fh
        db      ' !"#$'
        db      '%&''()*+,-./012345678'
        db      01Dh
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
$S243   db      00Eh
        db      011h
        db      00Bh
        db      018h
        db      001h
        db      005h
        db      000h
        db      000h
        db      003h
        db      01Ch
        db      00Fh
        db      006h
        db      015h
        db      00Ah,000h
        db      000h
        db      017h
        db      013h
        db      00Ch
        db      004h
        db      01Ah
        db      008h
        db      000h
        db      000h
        db      010h
        db      007h
        db      01Bh
        db      014h
        db      00Dh
        db      002h
        db      000h
        db      000h
        db      ')4'
        db      01Fh
        db      '%/7',000h
        db      000h
        db      01Eh
        db      '(3-!0',000h
        db      000h
        db      ',1''8"5',000h
        db      000h
        db      '.*2$'
        db      01Dh
        db      ' ',000h
        db      000h
$S266   db      00Eh
        db      001h
        db      00Ah
        db      003h
        db      004h
        db      006h
        db      008h
        db      007h
        db      00Eh
        db      009h
        db      002h
        db      00Ch
        db      009h
        db      00Eh
        db      020h
        db      011h
        db      005h
        db      014h
        db      006h
        db      016h
        db      01Eh
        db      017h
        db      003h
        db      019h
        db      00Dh
        db      01Ch
        db      01Fh
        db      020h
        db      00Ah
$S267   db      00Fh
        db      001h
        db      003h
        db      004h
        db      01Dh
        db      006h
        db      007h
        db      007h
        db      006h
        db      00Ah
        db      00Ah
        db      00Bh
        db      004h
        db      00Eh
        db      008h
        db      00Fh
        db      00Eh
        db      012h
        db      001h
        db      013h
        db      00Bh
        db      018h
        db      005h
        db      019h
        db      005h
        db      01Ch
        db      006h
        db      01Eh
        db      01Eh
        db      01Fh
        db      009h
$S268   db      00Ch
        db      002h
        db      005h
        db      005h
        db      00Ch
        db      009h
        db      009h
        db      00Ch
        db      01Dh
        db      00Eh
        db      007h
        db      00Fh
        db      006h
        db      013h
        db      00Ah
        db      016h
        db      002h
        db      017h
        db      001h
        db      01Ah
        db      001h
        db      01Bh
        db      00Bh
        db      020h
        db      020h
$S269   db      00Eh
        db      002h
        db      012h
        db      006h
        db      00Eh
        db      007h
        db      01Bh
        db      009h
        db      016h
        db      00Bh
        db      019h
        db      00Eh
        db      015h
        db      00Fh
        db      01Ah
        db      012h
        db      00Dh
        db      014h
        db      017h
        db      016h
        db      010h
        db      019h
        db      015h
        db      01Ch
        db      00Fh
        db      01Dh
        db      01Dh
        db      020h
        db      018h
$S270   db      00Eh
        db      001h
        db      013h
        db      003h
        db      01Bh
        db      005h
        db      016h
        db      007h
        db      019h
        db      00Ah
        db      015h
        db      00Bh
        db      01Ah
        db      00Dh
        db      012h
        db      012h
        db      010h
        db      015h
        db      015h
        db      018h
        db      00Fh
        db      019h
        db      01Dh
        db      01Ch
        db      012h
        db      01Eh
        db      01Ah
        db      020h
        db      00Eh
$S271   db      00Eh
        db      003h
        db      01Ah
        db      005h
        db      012h
        db      009h
        db      01Ch
        db      00Ch
        db      019h
        db      00Dh
        db      01Eh
        db      00Fh
        db      011h
        db      011h
        db      01Dh
        db      014h
        db      012h
        db      016h
        db      01Ah
        db      017h
        db      013h
        db      01Ah
        db      016h
        db      01Ch
        db      011h
        db      01Dh
        db      014h
        db      020h
        db      015h
$S272   db      00Dh
        db      004h
        db      019h
        db      005h
        db      01Eh
        db      007h
        db      011h
        db      00Ah
        db      018h
        db      00Eh
        db      00Eh
        db      00Fh
        db      01Bh
        db      012h
        db      016h
        db      014h
        db      011h
        db      015h
        db      014h
        db      01Ah
        db      00Dh
        db      01Ch
        db      017h
        db      01Eh
        db      010h
        db      01Fh
        db      01Ch
$S273   db      00Eh
        db      001h
        db      004h
        db      004h
        db      002h
        db      005h
        db      001h
        db      008h
        db      001h
        db      009h
        db      00Bh
        db      00Eh
        db      005h
        db      00Fh
        db      005h
        db      011h
        db      00Ch
        db      015h
        db      009h
        db      018h
        db      01Dh
        db      01Ah
        db      007h
        db      01Bh
        db      006h
        db      01Eh
        db      00Ah
        db      01Fh
        db      00Ah
$S274   db      00Dh
        db      002h
        db      01Dh
        db      003h
        db      00Dh
        db      006h
        db      01Fh
        db      009h
        db      00Ah
        db      00Ch
        db      002h
        db      00Dh
        db      001h
        db      010h
        db      001h
        db      011h
        db      002h
        db      014h
        db      009h
        db      016h
        db      020h
        db      019h
        db      00Ch
        db      01Dh
        db      009h
        db      01Fh
        db      003h
$S275   db      00Eh
        db      003h
        db      005h
        db      006h
        db      006h
        db      008h
        db      01Eh
        db      009h
        db      003h
        db      00Bh
        db      00Dh
        db      00Eh
        db      01Fh
        db      012h
        db      00Ah
        db      013h
        db      004h
        db      016h
        db      008h
        db      017h
        db      00Eh
        db      019h
        db      002h
        db      01Ch
        db      009h
        db      01Eh
        db      020h
        db      020h
        db      005h
$S276   db      00Dh
        db      004h
        db      01Ah
        db      005h
        db      013h
        db      008h
        db      016h
        db      00Ah
        db      011h
        db      00Bh
        db      014h
        db      010h
        db      00Dh
        db      011h
        db      012h
        db      015h
        db      01Ch
        db      018h
        db      019h
        db      019h
        db      01Eh
        db      01Bh
        db      011h
        db      01Eh
        db      018h
        db      020h
        db      012h
$S277   db      00Eh
        db      001h
        db      01Bh
        db      004h
        db      016h
        db      006h
        db      011h
        db      007h
        db      014h
        db      00Ch
        db      00Dh
        db      00Eh
        db      017h
        db      010h
        db      010h
        db      011h
        db      01Ch
        db      014h
        db      019h
        db      015h
        db      01Eh
        db      017h
        db      011h
        db      01Ah
        db      018h
        db      01Eh
        db      00Eh
        db      01Fh
        db      013h
$S278   db      00Dh
        db      001h
        db      01Ah
        db      004h
        db      00Dh
        db      006h
        db      017h
        db      008h
        db      010h
        db      00Bh
        db      015h
        db      00Eh
        db      00Fh
        db      00Fh
        db      01Dh
        db      012h
        db      018h
        db      016h
        db      00Eh
        db      017h
        db      01Bh
        db      019h
        db      016h
        db      01Bh
        db      019h
        db      01Eh
        db      015h
$S279   db      00Eh
        db      002h
        db      019h
        db      003h
        db      015h
        db      006h
        db      00Fh
        db      007h
        db      01Dh
        db      00Ah
        db      012h
        db      00Ch
        db      01Ah
        db      00Dh
        db      013h
        db      010h
        db      016h
        db      011h
        db      016h
        db      013h
        db      019h
        db      016h
        db      015h
        db      017h
        db      01Ah
        db      019h
        db      012h
        db      01Dh
        db      01Ch
$S280   db      00Eh
        db      002h
        db      002h
        db      004h
        db      008h
        db      005h
        db      00Eh
        db      007h
        db      002h
        db      00Ah
        db      009h
        db      00Ch
        db      020h
        db      00Fh
        db      00Ch
        db      012h
        db      006h
        db      014h
        db      01Eh
        db      015h
        db      003h
        db      017h
        db      00Dh
        db      01Ah
        db      01Fh
        db      01Dh
        db      00Ah
        db      01Fh
        db      004h
$S281   db      00Fh
        db      001h
        db      00Dh
        db      004h
        db      007h
        db      005h
        db      006h
        db      008h
        db      00Ah
        db      009h
        db      004h
        db      00Ch
        db      008h
        db      00Dh
        db      00Eh
        db      00Fh
        db      002h
        db      011h
        db      00Bh
        db      016h
        db      005h
        db      017h
        db      005h
        db      01Ah
        db      006h
        db      01Ch
        db      01Eh
        db      01Dh
        db      003h
        db      020h
        db      01Dh
$S282   db      00Dh
        db      001h
        db      005h
        db      003h
        db      00Ch
        db      007h
        db      009h
        db      00Ah
        db      01Dh
        db      00Ch
        db      007h
        db      00Dh
        db      006h
        db      010h
        db      00Ah
        db      011h
        db      00Ah
        db      014h
        db      002h
        db      015h
        db      001h
        db      018h
        db      001h
        db      019h
        db      00Bh
        db      01Eh
        db      005h
$S283   db      00Eh
        db      002h
        db      01Ah
        db      004h
        db      00Eh
        db      005h
        db      01Bh
        db      007h
        db      016h
        db      009h
        db      019h
        db      00Ch
        db      015h
        db      00Dh
        db      01Ah
        db      00Fh
        db      012h
        db      012h
        db      017h
        db      014h
        db      010h
        db      017h
        db      015h
        db      01Ah
        db      00Fh
        db      01Bh
        db      01Dh
        db      01Eh
        db      012h
$S284   db      00Eh
        db      002h
        db      016h
        db      003h
        db      016h
        db      005h
        db      019h
        db      008h
        db      015h
        db      009h
        db      01Ah
        db      00Bh
        db      012h
        db      00Fh
        db      01Ch
        db      013h
        db      015h
        db      016h
        db      00Fh
        db      017h
        db      01Dh
        db      01Ah
        db      012h
        db      01Ch

        db      01Ah
        db      01Dh
        db      013h
        db      01Fh
        db      01Bh
$S285   db      00Eh
        db      002h
        db      00Dh
        db      003h
        db      012h
        db      007h
        db      01Ch
        db      00Ah
        db      019h
        db      00Bh
        db      01Eh
        db      00Dh
        db      011h
        db      010h
        db      018h
        db      012h
        db      012h
        db      014h
        db      01Ah
        db      015h
        db      013h
        db      018h
        db      016h
        db      01Ah
        db      011h
        db      01Bh
        db      014h
        db      01Fh
        db      01Ah
$S286   db      00Dh
        db      001h
        db      015h
        db      003h
        db      01Eh
        db      005h
        db      011h
        db      008h
        db      018h
        db      00Ch
        db      00Eh
        db      00Dh
        db      01Bh
        db      00Fh
        db      016h
        db      012h
        db      011h
        db      013h
        db      014h
        db      018h
        db      00Dh
        db      01Ah
        db      017h
        db      01Ch
        db      010h
        db      020h
        db      019h
$S287   db      00Eh
        db      002h
        db      008h
        db      003h
        db      001h
        db      006h
        db      001h
        db      007h
        db      00Bh
        db      00Ch
        db      005h
        db      00Dh
        db      005h
        db      010h
        db      006h
        db      013h
        db      009h
        db      016h
        db      01Dh
        db      018h
        db      007h
        db      019h
        db      006h
        db      01Ch
        db      00Ah
        db      01Dh
        db      004h
        db      020h
        db      002h
$S288   db      00Dh
        db      002h
        db      007h
        db      004h
        db      01Fh
        db      007h
        db      00Ah
        db      00Ah
        db      002h
        db      00Bh
        db      001h
        db      00Eh
        db      001h
        db      00Fh
        db      00Bh
        db      012h
        db      009h
        db      014h
        db      020h
        db      017h
        db      00Ch
        db      01Bh
        db      009h
        db      01Eh
        db      01Dh
        db      01Fh
        db      00Dh
$S289   db      00Eh
        db      001h
        db      00Ch
        db      004h
        db      006h
        db      006h
        db      01Eh
        db      007h
        db      003h
        db      009h
        db      00Dh
        db      00Ch
        db      01Fh
        db      00Fh
        db      00Ah
        db      011h
        db      004h
        db      014h
        db      008h
        db      015h
        db      00Eh
        db      017h
        db      002h
        db      01Ah
        db      009h
        db      01Ch
        db      020h
        db      01Fh
        db      005h
$S290   db      00Dh
        db      002h
        db      00Eh
        db      003h
        db      013h
        db      006h
        db      016h
        db      008h
        db      011h
        db      009h
        db      014h
        db      00Eh
        db      00Dh
        db      010h
        db      017h
        db      013h
        db      01Ch
        db      016h
        db      019h
        db      017h
        db      01Eh
        db      019h
        db      011h
        db      01Ch
        db      018h
        db      020h
        db      01Ah
$S291   db      00Dh
        db      001h
        db      016h
        db      004h
        db      011h
        db      005h
        db      014h
        db      00Ah
        db      00Dh
        db      00Ch
        db      017h
        db      00Eh
        db      010h
        db      012h
        db      019h
        db      013h
        db      01Eh
        db      015h
        db      011h
        db      018h
        db      018h
        db      01Ch
        db      00Eh
        db      01Dh
        db      01Bh
        db      020h
        db      016h
$S292   db      00Eh
        db      001h
        db      012h
        db      004h
        db      017h
        db      006h
        db      010h
        db      009h
        db      015h
        db      00Ch
        db      00Fh
        db      00Dh
        db      01Dh
        db      010h
        db      012h
        db      014h
        db      00Eh
        db      015h
        db      01Bh
        db      017h
        db      016h
        db      019h
        db      019h
        db      01Ch
        db      015h
        db      01Dh
        db      01Ah
        db      020h
        db      00Dh
$S293   db      00Fh
        db      001h
        db      01Eh
        db      004h
        db      00Fh
        db      005h
        db      01Dh
        db      008h
        db      012h
        db      00Ah
        db      01Ah
        db      00Bh
        db      013h
        db      00Eh
        db      016h
        db      010h
        db      011h
        db      011h
        db      019h
        db      014h
        db      015h
        db      015h
        db      01Ah
        db      017h
        db      012h
        db      01Bh
        db      01Ch
        db      01Eh
        db      019h
        db      01Fh
        db      015h
$S294   db      00Dh
        db      001h
        db      001h
        db      003h
        db      00Eh
        db      005h
        db      002h
        db      008h
        db      009h
        db      00Ah,' ',00Dh
        db      00Ch
        db      012h
        db      01Eh
        db      013h
        db      003h
        db      015h
        db      00Dh
        db      018h
        db      01Fh
        db      01Bh
        db      00Ah
        db      01Eh
        db      002h
        db      020h
        db      008h
$S295   db      00Fh
        db      002h
        db      01Fh
        db      003h
        db      006h
        db      006h
        db      00Ah
        db      007h
        db      004h
        db      00Ah
        db      008h
        db      00Bh
        db      00Eh
        db      00Dh
        db      002h
        db      010h
        db      009h
        db      014h
        db      005h
        db      015h
        db      005h
        db      018h
        db      006h
        db      01Ah
        db      01Eh
        db      01Bh
        db      003h
        db      01Dh
        db      00Dh
        db      020h
        db      007h
$S296   db      00Eh
        db      002h
        db      006h
        db      005h
        db      009h
        db      008h
        db      01Dh
        db      00Ah
        db      007h
        db      00Bh
        db      006h
        db      00Eh
        db      00Ah
        db      00Fh
        db      004h
        db      012h
        db      002h
        db      013h
        db      001h
        db      016h
        db      001h
        db      017h
        db      00Bh
        db      01Ch
        db      005h
        db      01Dh
        db      005h
        db      01Fh
        db      00Ch
$S297   db      00Dh
        db      001h
        db      00Bh
        db      004h
        db      009h
        db      006h
        db      020h
        db      009h
        db      00Ch
        db      00Dh
        db      009h
        db      010h
        db      01Dh
        db      011h
        db      00Dh
        db      014h
        db      01Fh
        db      017h
        db      00Ah
        db      01Ah
        db      002h
        db      01Bh
        db      001h
        db      01Eh
        db      001h
        db      01Fh
        db      002h
$S298   db      00Fh
        db      002h
        db      011h
        db      003h
        db      019h
        db      006h
        db      015h
        db      007h
        db      01Ah
        db      009h
        db      012h
        db      00Dh
        db      01Ch
        db      010h
        db      019h
        db      011h
        db      015h
        db      014h
        db      00Fh
        db      015h
        db      01Dh
        db      018h
        db      012h
        db      01Ah
        db      01Ah
        db      01Bh
        db      013h
        db      01Eh
        db      016h
        db      01Fh
        db      016h
$S299   db      00Dh
        db      002h
        db      017h
        db      005h
        db      01Ch
        db      008h
        db      019h
        db      009h
        db      01Eh
        db      00Bh
        db      011h
        db      00Eh
        db      018h
        db      012h
        db      01Ah
        db      013h
        db      013h
        db      016h
        db      016h
        db      018h
        db      011h
        db      019h
        db      014h
        db      01Eh
        db      00Dh
        db      01Fh
        db      012h
$S300   db      00Dh
        db      002h
        db      00Fh
        db      003h
        db      011h
        db      006h
        db      018h
        db      00Ah
        db      00Eh
        db      00Bh
        db      01Bh
        db      00Dh
        db      016h
        db      00Fh
        db      019h
        db      011h
        db      014h
        db      016h
        db      00Dh
        db      018h
        db      017h
        db      01Ah
        db      010h
        db      01Dh
        db      015h
        db      01Fh
        db      01Eh
$S301   db      00Fh
        db      001h
        db      00Eh
        db      004h
        db      001h
        db      005h
        db      00Bh
        db      00Ah
        db      005h
        db      00Bh
        db      005h
        db      00Eh
        db      006h
        db      010h
        db      01Eh
        db      011h
        db      009h
        db      014h
        db      01Dh
        db      016h
        db      007h
        db      017h
        db      006h
        db      01Ah
        db      00Ah
        db      01Bh
        db      004h
        db      01Eh
        db      008h
        db      01Fh
        db      001h
$S302   db      00Ch
        db      001h
        db      006h
        db      005h
        db      00Ah
        db      008h
        db      002h
        db      009h
        db      001h
        db      00Ch
        db      001h
        db      00Dh
        db      00Bh
        db      012h
        db      020h
        db      015h
        db      00Ch
        db      019h
        db      009h
        db      01Ch
        db      01Dh
        db      01Eh
        db      007h
        db      020h
        db      01Fh
$S303   db      00Dh
        db      004h
        db      01Eh
        db      005h
        db      003h
        db      007h
        db      00Dh
        db      00Ah
        db      01Fh
        db      00Dh
        db      00Ah
        db      010h
        db      002h
        db      012h
        db      008h
        db      013h
        db      00Eh
        db      015h
        db      002h
        db      018h
        db      009h
        db      01Ah
        db      020h
        db      01Dh
        db      00Ch
        db      020h
        db      006h
$S304   db      00Fh
        db      002h
        db      009h
        db      006h
        db      005h
        db      007h
        db      005h
        db      00Ah
        db      006h
        db      00Ch
        db      01Eh
        db      00Dh
        db      003h
        db      00Fh
        db      00Dh
        db      012h
        db      007h
        db      013h
        db      006h
        db      016h
        db      00Ah
        db      017h
        db      004h
        db      01Ah
        db      008h
        db      01Bh
        db      00Eh
        db      01Dh
        db      002h
        db      01Fh
        db      00Bh
$S305   db      00Dh
        db      001h
        db      019h
        db      003h
        db      014h
        db      008h
        db      00Dh
        db      00Ah
        db      017h
        db      00Ch
        db      010h
        db      00Fh
        db      015h
        db      011h
        db      01Eh
        db      013h
        db      011h
        db      016h
        db      018h
        db      01Ah
        db      00Eh
        db      01Bh
        db      01Bh
        db      01Dh
        db      016h
        db      020h
        db      011h
$S306   db      00Eh
        db      004h
        db      010h
        db      007h
        db      015h
        db      00Ah
        db      00Fh
        db      00Bh
        db      01Dh
        db      00Eh
        db      012h
        db      010h
        db      01Ah
        db      012h
        db      00Eh
        db      013h
        db      01Bh
        db      015h
        db      016h
        db      017h
        db      019h
        db      01Ah
        db      015h
        db      01Bh
        db      01Ah
        db      01Dh
        db      012h
        db      020h
        db      017h
$S307   db      00Fh
        db      001h
        db      011h
        db      003h
        db      01Dh
        db      006h
        db      012h
        db      008h
        db      01Ah
        db      009h
        db      013h
        db      00Ch
        db      016h
        db      00Eh
        db      011h
        db      00Fh
        db      014h
        db      012h
        db      015h
        db      013h
        db      01Ah
        db      015h
        db      012h
        db      019h
        db      01Ch
        db      01Ch
        db      019h
        db      01Dh
        db      01Eh
        db      020h
        db      00Fh
$S308   db      00Dh
        db      002h
        db      001h
        db      003h
        db      002h
        db      006h
        db      009h
        db      008h
        db      020h
        db      00Bh
        db      00Ch
        db      00Fh
        db      009h
        db      011h
        db      003h
        db      013h
        db      00Dh
        db      016h
        db      01Fh
        db      019h
        db      00Ah
        db      01Ch
        db      002h
        db      01Dh
        db      001h
        db      01Fh
        db      00Eh
$S309   db      00Fh
        db      004h
        db      00Ah
        db      005h
        db      004h
        db      008h
        db      008h
        db      009h
        db      00Eh
        db      00Bh
        db      002h
        db      00Eh
        db      009h
        db      010h
        db      020h
        db      012h
        db      005h
        db      013h
        db      005h
        db      016h
        db      006h
        db      018h
        db      01Eh
        db      019h
        db      003h
        db      01Bh
        db      00Dh
        db      01Eh
        db      01Fh
        db      01Fh
        db      006h
$S310   db      00Eh
        db      002h
        db      01Eh
        db      003h
        db      009h
        db      006h
        db      01Dh
        db      008h
        db      007h
        db      009h
        db      006h
        db      00Ch
        db      00Ah
        db      00Dh
        db      004h
        db      010h
        db      008h
        db      011h
        db      001h
        db      014h
        db      001h
        db      015h
        db      00Bh
        db      01Ah
        db      005h
        db      01Bh
        db      005h
        db      01Eh
        db      006h
$S311   db      00Ch
        db      004h
        db      020h
        db      007h
        db      00Ch
        db      00Bh
        db      009h
        db      00Eh
        db      01Dh
        db      010h
        db      007h
        db      012h
        db      01Fh
        db      015h
        db      00Ah
        db      018h
        db      002h
        db      019h
        db      001h
        db      01Ch
        db      001h
        db      01Dh
        db      00Bh
        db      020h
        db      009h
$S312   db      00Fh
        db      001h
        db      014h
        db      004h
        db      015h
        db      005h
        db      01Ah
        db      007h
        db      012h
        db      00Bh
        db      01Ch
        db      00Eh
        db      019h
        db      00Fh
        db      01Eh
        db      012h
        db      00Fh
        db      013h
        db      01Dh
        db      016h
        db      012h
        db      018h
        db      01Ah
        db      019h
        db      013h
        db      01Ch
        db      016h
        db      01Eh
        db      011h
        db      01Fh
        db      019h
$S313   db      00Dh
        db      002h
        db      010h
        db      003h
        db      01Ch
        db      006h
        db      019h
        db      007h
        db      01Eh
        db      009h
        db      011h
        db      00Ch
        db      018h
        db      010h
        db      00Eh
        db      011h
        db      013h
        db      014h
        db      016h
        db      016h
        db      011h
        db      017h
        db      014h
        db      01Ch
        db      00Dh
        db      01Eh
        db      017h
$S314   db      00Dh
        db      001h
        db      01Dh
        db      004h
        db      018h
        db      008h
        db      00Eh
        db      009h
        db      01Bh
        db      00Bh
        db      016h
        db      00Dh
        db      019h
        db      010h
        db      015h
        db      014h
        db      00Dh
        db      016h
        db      017h
        db      018h
        db      010h
        db      01Bh
        db      015h
        db      01Eh
        db      00Fh
        db      01Fh
        db      011h
$S315   db      00Fh
        db      001h
        db      002h
        db      003h
        db      00Bh
        db      008h
        db      005h
        db      009h
        db      005h
        db      00Ch
        db      006h
        db      00Eh
        db      01Eh
        db      00Fh
        db      003h
        db      012h
        db      01Dh
        db      014h
        db      007h
        db      015h
        db      006h
        db      018h
        db      00Ah
        db      019h
        db      004h
        db      01Ch
        db      008h
        db      01Dh
        db      00Eh
        db      020h
        db      001h
$S316   db      00Ch
        db      002h
        db      00Ah
        db      003h
        db      00Ah
        db      006h
        db      002h
        db      007h
        db      001h
        db      00Ah
        db      001h
        db      00Bh
        db      00Bh
        db      010h
        db      005h
        db      013h
        db      00Ch
        db      017h
        db      009h
        db      01Ah
        db      01Dh
        db      01Ch
        db      007h
        db      01Dh
        db      006h
$S317   db      00Dh
        db      001h
        db      009h
        db      003h
        db      003h
        db      005h
        db      00Dh
        db      008h
        db      01Fh
        db      00Bh
        db      00Ah
        db      00Eh
        db      002h
        db      00Fh
        db      001h
        db      011h
        db      00Eh
        db      013h
        db      002h
        db      016h
        db      009h
        db      018h
        db      020h
        db      01Bh
        db      00Ch
        db      020h
        db      01Eh
$S318   db      00Fh
        db      002h
        db      020h
        db      004h
        db      005h
        db      005h
        db      005h
        db      008h
        db      006h
        db      00Ah
        db      01Eh
        db      00Bh
        db      003h
        db      00Dh
        db      00Dh
        db      010h
        db      01Fh
        db      011h
        db      006h
        db      014h
        db      00Ah
        db      015h
        db      004h
        db      018h
        db      008h
        db      019h
        db      00Eh
        db      01Bh
        db      002h
        db      01Eh
        db      009h
$S319   db      00Dh
        db      002h
        db      015h
        db      006h
        db      00Dh
        db      008h
        db      017h
        db      00Ah
        db      010h
        db      00Dh
        db      015h
        db      010h
        db      00Fh
        db      011h
        db      011h
        db      014h
        db      018h
        db      018h
        db      00Eh
        db      019h
        db      01Bh
        db      01Bh
        db      016h
        db      01Dh
        db      019h
        db      01Fh
        db      014h
$S320   db      00Eh
        db      001h
        db      01Ch
        db      005h
        db      015h
        db      008h
        db      00Fh
        db      009h
        db      01Dh
        db      00Ch
        db      012h
        db      00Eh
        db      01Ah
        db      00Fh
        db      013h
        db      011h
        db      01Bh
        db      013h

        db      016h
        db      015h
        db      019h
        db      018h
        db      015h
        db      019h
        db      01Ah
        db      01Bh
        db      012h
        db      020h
        db      010h
$S321   db      00Eh
        db      002h
        db      018h
        db      004h
        db      012h
        db      006h
        db      01Ah
        db      007h
        db      013h
        db      00Ah
        db      016h
        db      00Ch
        db      011h
        db      00Dh
        db      014h
        db      011h
        db      01Ah
        db      013h
        db      012h
        db      017h
        db      01Ch
        db      01Ah
        db      019h
        db      01Bh
        db      01Eh
$S112   dd      01D1F111Dh
        db      000h
        db      000h
        db      000h
        db      080h
        db      000h
        db      000h
        db      000h
        db      '@',000h
        db      000h
        db      000h
        db      ' ',000h
        db      000h
        db      000h
        db      010h
        db      000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      004h
        db      000h
        db      000h
        db      000h
        db      002h
        db      000h
        db      000h
        db      000h
        db      001h
        db      000h
        db      000h
        db      080h
        db      000h
        db      000h
        db      000h
        db      '@',000h
        db      000h
        db      000h
        db      ' ',000h
        db      000h
        db      000h
        db      010h
        db      000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      004h
        db      000h
        db      000h
        db      000h
        db      002h
        db      000h
        db      000h
        db      000h
        db      001h
        db      000h
        db      000h
        db      080h
        db      000h
        db      000h
        db      000h
        db      '@',000h
        db      000h
        db      000h
        db      ' ',000h
        db      000h
        db      000h
        db      010h
        db      000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      004h
        db      000h
        db      000h
        db      000h
        db      002h
        db      000h
        db      000h
        db      000h
        db      001h
        db      000h
        db      000h
        db      080h
        db      000h
        db      000h
        db      000h
        db      '@',000h
        db      000h
        db      000h
        db      ' ',000h
        db      000h
        db      000h
        db      010h
        db      000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      004h
        db      000h
        db      000h
        db      000h
        db      002h
        db      000h
        db      000h
        db      000h
        db      001h
        db      000h
        db      000h
        db      000h
$S227   db      001h
        db      000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      001h
        db      002h
        db      000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      002h
        db      003h
        db      000h
        db      000h
        db      000h
        db      009h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      080h
        db      000h
        db      004h
        db      000h
        db      000h
        db      000h
        db      007h
        db      000h
        db      000h
        db      000h
        db      ' ',000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      000h
        db      004h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      009h
        db      000h
        db      000h
        db      000h
        db      010h
        db      000h
        db      000h
        db      000h
        db      007h
        db      000h
        db      000h
        db      000h
        db      007h
        db      000h
        db      000h
        db      000h
        db      000h
        db      080h
        db      000h
        db      000h
        db      009h
        db      000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      080h
        db      00Ah,000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      002h
        db      000h
        db      000h
        db      000h
        db      00Bh
        db      000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      004h
        db      000h
        db      000h
        db      000h
        db      00Ch
        db      000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      000h
        db      004h
        db      000h
        db      000h
        db      00Dh,000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      080h
        db      000h
        db      000h
        db      00Eh
        db      000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      002h
        db      000h
        db      000h
        db      00Fh
        db      000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      002h
        db      000h
        db      011h
        db      000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      004h
        db      000h
        db      012h
        db      000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      004h
        db      013h
        db      000h
        db      000h
        db      000h
        db      007h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      080h
        db      000h
        db      014h
        db      000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      ' ',000h
        db      000h
        db      000h
        db      015h
        db      000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      080h
        db      000h
        db      000h
        db      000h
        db      016h
        db      000h
        db      000h
        db      000h
        db      007h
        db      000h
        db      000h
        db      000h
        db      010h
        db      000h
        db      000h
        db      000h
        db      017h
        db      000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      000h
        db      080h
        db      000h
        db      000h
        db      019h
        db      000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      080h
        db      01Ah
        db      000h
        db      000h
        db      000h
        db      007h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      '@',000h
        db      01Bh,000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      004h
        db      000h
        db      000h
        db      000h
        db      01Ch
        db      000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      004h
        db      000h
        db      000h
        db      01Dh
        db      000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      000h
        db      010h
        db      000h
        db      000h
        db      01Eh
        db      000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      001h
        db      000h
        db      01Fh
        db      000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      '!',000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      '@"',000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      '@',000h
        db      '#',000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      '$'
        db      000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      001h
        db      000h
        db      000h
        db      000h
        db      '%',000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      '&',000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      080h
        db      000h
        db      000h
        db      000h
        db      '''',000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      001h
        db      000h
        db      000h
        db      ')',000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      ' ',000h
        db      '*',000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      '@+',000h
        db      000h
        db      000h
        db      007h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      ',',000h
        db      000h
        db      000h
        db      007h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      008h
        db      '-',000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      '.',000h
        db      000h
        db      000h
        db      00Ah,000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      002h
        db      000h
        db      '/',000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      '@',000h
        db      000h
        db      '1',000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      080h
        db      000h
        db      '2',000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      001h
        db      '3',000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      '4',000h
        db      000h
        db      000h
        db      00Bh
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      080h
        db      000h
        db      '5',000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      '@',000h
        db      000h
        db      000h
        db      '6',000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      080h
        db      000h
        db      000h
        db      000h
        db      '7',000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      004h
        db      000h
        db      '9',000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      008h
        db      ':',000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      '@;',000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      '<',000h
        db      000h
        db      000h
        db      005h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      008h
        db      '=',000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      000h
        db      002h
        db      000h
        db      000h
        db      '>',000h
        db      000h
        db      000h
        db      008h
        db      000h
        db      000h
        db      000h
        db      000h
        db      000h
        db      002h
        db      000h
        db      '?',000h
        db      000h
        db      000h
        db      006h
        db      000h
        db      000h
        db      000h
        db      '@',000h
        db      000h
        db      000h
        db      'BrydDES Key Search Library version 1.01.  Core 1. Copyrigh'
        db      't Svend Olaf Mikkelsen, 1995, 1997, 19'
$S14    dd      0202E3839h
        dd      $S266
        dd      $S267
        dd      $S268
        dd      $S269
        dd      $S270
        dd      $S271
        dd      $S272
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S273
        dd      $S274
        dd      $S275
        dd      $S276
        dd      $S277
        dd      $S278
        dd      $S279
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S280
        dd      $S281
        dd      $S282

        dd      $S283
        dd      $S284
        dd      $S285
        dd      $S286
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S287
        dd      $S288
        dd      $S289
        dd      $S290
        dd      $S291
        dd      $S292
        dd      $S293
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S294
        dd      $S295
        dd      $S296
        dd      $S297
        dd      $S298
        dd      $S299
        dd      $S300
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S301
        dd      $S302
        dd      $S303
        dd      $S304
        dd      $S305
        dd      $S306
        dd      $S307
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S308
        dd      $S309
        dd      $S310
        dd      $S311
        dd      $S312
        dd      $S313
        dd      $S314
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S315
        dd      $S316
        dd      $S317
        dd      $S318
        dd      $S319
        dd      $S320
        dd      $S321

_DATA   ENDS
_BSS    SEGMENT

$S1     dd      1 dup(?)
$S134   dd      1 dup(?)
$S120   dd      1 dup(?)
$S162   dd      1 dup(?)
$S143   dd      1 dup(?)
$S135   dd      1 dup(?)
$S121   dd      1 dup(?)
$S144   dd      1 dup(?)
$S122   dd      1 dup(?)
$S136   dd      1 dup(?)
$S123   dd      1 dup(?)
$S172   dd      1 dup(?)
$S137   dd      1 dup(?)
$S124   dd      1 dup(?)
$S138   dd      1 dup(?)
$S139   dd      1 dup(?)
$S125   dd      1 dup(?)
$S126   dd      1 dup(?)
$S202   dd      1 dup(?)
$S140   dd      1 dup(?)
$S127   dd      1 dup(?)
$S141   dd      1 dup(?)
$S184   dd      1 dup(?)
$S145   dd      1 dup(?)
$S128   dd      1 dup(?)
$S129   dd      1 dup(?)
$S146   dd      1 dup(?)
$S130   dd      1 dup(?)
$S185   dd      1 dup(?)
$S131   dd      1 dup(?)
$S149   dd      1 dup(?)
$S132   dd      1 dup(?)
$S142   dd      1 dup(?)
$S24    dd      1 dup(?)
$S49    dd      1 dup(?)
$S50    dd      1 dup(?)
$S51    dd      1 dup(?)
$S52    dd      1 dup(?)
$S25    dd      1 dup(?)
$S26    dq      1 dup(?)
$S28    dd      1 dup(?)
$S29    db      256 dup(?)
$S154   db      256 dup(?)
$S32    db      256 dup(?)
$S155   db      256 dup(?)
$S35    db      256 dup(?)
$S151   db      256 dup(?)
$S152   db      256 dup(?)
$S150   db      256 dup(?)
$S30    db      256 dup(?)
$S36    db      256 dup(?)
$S33    db      768 dup(?)
$S27    dd      1 dup(?)
$S232   dd      1 dup(?)
$S233   dd      1 dup(?)
$S234   db      28 dup(?)
$S193   dq      1 dup(?)
$S239   dd      1 dup(?)
$S39    dd      1 dup(?)
$S40    dd      1 dup(?)
$S41    dd      1 dup(?)
$S43    dd      1 dup(?)
$S178   dq      1 dup(?)
$S45    dd      1 dup(?)
$S46    dd      1 dup(?)
$S42    dd      1 dup(?)
$S44    dd      1 dup(?)
$S173   db      12 dup(?)
$S206   dd      1 dup(?)
$S195   dd      1 dup(?)
$S189   dd      1 dup(?)
$S197   dd      1 dup(?)
$S196   dd      1 dup(?)
$S199   dd      1 dup(?)
$S198   dd      1 dup(?)
$S192   dd      1 dup(?)
$S204   dd      1 dup(?)
$S190   dd      1 dup(?)
$S191   db      12 dup(?)
$S156   dd      1 dup(?)
$S158   dd      1 dup(?)
$S153   dd      1 dup(?)
$S159   dd      1 dup(?)
$S205   dd      1 dup(?)
$S181   dd      1 dup(?)
$S179   dd      1 dup(?)
$S182   dd      1 dup(?)
$S180   dd      1 dup(?)
$S187   dq      1 dup(?)
$S174   dd      1 dup(?)
$S175   dd      1 dup(?)
$S177   dd      1 dup(?)
$S176   dq      1 dup(?)
$S47    dd      1 dup(?)
$S48    dd      1 dup(?)
$S194   dd      1 dup(?)
$S188   dd      1 dup(?)
$S157   dd      1 dup(?)
$S160   dd      1 dup(?)
$S170   dd      1 dup(?)
$S166   dd      1 dup(?)
$S167   dd      1 dup(?)
$S168   dd      1 dup(?)
$S169   dd      1 dup(?)
$S163   dd      1 dup(?)
$S164   dd      1 dup(?)
$S165   dd      1 dup(?)
$S23    dq      1 dup(?)
$S17    dd      1 dup(?)
$S18    dd      1 dup(?)
$S21    dd      1 dup(?)
$S12    dd      1 dup(?)
$S15    db      112 dup(?)
$S254   dd      1 dup(?)
$S6     dd      1 dup(?)
$S20    dd      1 dup(?)
$S257   dd      1 dup(?)
$S22    dd      1 dup(?)
$S19    dd      1 dup(?)
$S2     dd      1 dup(?)

_BSS    ENDS
        END
