;
; $Log: bbdeslow.asm,v $
; Revision 1.1  1998/06/17 20:24:32  cyruspatel
; moved generic intel asm bdeslow.asm, bbdeslow.asm files to ./client/des/
; modified model definition directives to work with any (intel asm)
; assembler - model is overridden to flat unless small model was explicitely
; defined on the command line. Don't declare segreg 'assume's: The .model
; directive controls segreg assumptions unless explicitely declared.
; Added $Logs.
;
; Revision 1.5  1998/06/15 06:54:46  jlawson
; fixed segment assumes to be controlled by modelnum define.
;
; Revision 1.4  1998/06/15 03:08:54  jlawson
; cs ds es ss assumed flat. fixed wrong bbdes labels
;
; Revision 1.3  1998/06/15 02:54:43  jlawson
; updated bbdeslow.asm to be based off the same source that bdeslow.asm is.
; all asm's modified to use a "modelnum" equate to define memory model.
;
; Revision 1.2  1998/06/09 08:54:42  jlawson
; Changes from Cyrus Patel - disassembled tasm flat model objs to generic 
; intel asm for porting ease. Removed flat model specific assembler 
; directives to allow use of mem model overrides from the command line.
;
; Revision 1.1  1998/05/24 14:27:07  daa
; Import 5/23/98 client tree
;

.386p

ifndef __SMALL__
  .model flat 
endif

        EXTRN           _bbryd_key_found:NEAR    ; ok for flat or small model
        EXTRN           _bbryd_continue:NEAR     ; ok for flat or small model
        PUBLIC          _bbryd_des               ; Located at 1:0004h Type = 1
        PUBLIC          _Bdesinit                ; Located at 1:2BE8h Type = 1
        PUBLIC          _Bdesencrypt             ; Located at 1:2D25h Type = 1
        PUBLIC          _Bdesdecrypt             ; Located at 1:2EA1h Type = 1
        PUBLIC          _Bkey_byte_to_hex        ; Located at 1:301Ah Type = 1
        PUBLIC          _Bc_key_byte_to_hex      ; Located at 1:3026h Type = 1

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

_bbryd_des:
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
        mov     dword ptr $S15[esi*4],ebp
        inc     esi
$L13:   call    $L9
        inc     ebx
        loop    short $L16
        mov     dword ptr $S17,00000000h
        mov     dword ptr $S18,00000000h
        mov     dword ptr $S19,00000000h
        mov     dword ptr $S20,00000000h
        mov     $S21,esp
        mov     esi,[esp+018h]
        mov     $S22,esi
        mov     esi,[esp+01Ch]
        mov     $S23,esi
        mov     esi,[esp+020h]
        mov     dword ptr $S24,esi
        mov     esi,[esp+024h]
        mov     dword ptr $S25,esi
        mov     esi,[esp+028h]
        mov     dword ptr $S26,esi
        mov     ebp,$S25
        mov     eax,[ebp]
        mov     ebx,$S26
        and     eax,[ebx]
        mov     [ebp],eax
        mov     eax,[ebp+004h]
        and     eax,[ebx+004h]
        mov     [ebp+004h],eax
        push    ebp
        call    _Bdesinit
        add     esp,+004h
        mov     ecx,00000040h
        mov     ebx,00000000h
$L29:   xor     ebx,+010h
        mov     eax,[ebx+$S27]
        xor     ebx,+010h
        mov     [ebx+$S28],eax
        add     ebx,+004h
        loop    short $L29
        mov     ecx,00000040h
        mov     ebx,00000000h
$L31:   xor     ebx,00000080h
        mov     eax,[ebx+$S27]
        xor     ebx,00000080h
        mov     [ebx+$S30],eax
        add     ebx,+004h
        loop    short $L31
        mov     ecx,00000040h
        mov     ebx,00000000h
$L34:   xor     ebx,00000080h
        mov     eax,[ebx+$S32]
        xor     ebx,00000080h
        mov     [ebx+$S33],eax
        add     ebx,+004h
        loop    short $L34
        mov     ecx,00000040h
        mov     ebx,00000000h
$L37:   xor     ebx,+020h
        mov     eax,[ebx+$S35]
        xor     ebx,+020h
        mov     [ebx+$S36],eax
        add     ebx,+004h
        loop    short $L37
        mov     ecx,00000040h
        mov     ebx,00000000h
$L39:   xor     ebx,+020h
        mov     eax,[ebx+$S32]
        xor     ebx,+020h
        mov     [ebx+$S38],eax
        add     ebx,+004h
        loop    short $L39
        mov     ecx,00000040h
        mov     ebx,00000000h
$L41:   xor     ebx,+040h
        mov     eax,[ebx+$S27]
        xor     ebx,+040h
        mov     [ebx+$S40],eax
        add     ebx,+004h
        loop    short $L41
        mov     ecx,00000040h
        mov     ebx,00000000h
$L43:   xor     ebx,+004h
        mov     eax,[ebx+$S32]
        xor     ebx,+004h
        mov     [ebx+$S42],eax
        add     ebx,+004h
        loop    short $L43
        mov     ecx,00000040h
        mov     ebx,00000000h
$L45:   mov     eax,[ebx+$S32]
        xor     ebx,+010h
        mov     edx,[ebx+$S32]
        xor     ebx,+010h
        xor     eax,edx
        mov     [ebx+$S44],eax
        add     ebx,+004h
        loop    short $L45
        mov     ecx,00000040h
        mov     ebx,00000000h
$L47:   mov     eax,[ebx+$S35]
        xor     ebx,+004h
        mov     edx,[ebx+$S35]
        xor     ebx,+004h
        xor     eax,edx
        mov     [ebx+$S46],eax
        add     ebx,+004h
        loop    short $L47
        mov     ecx,00000040h
        mov     ebx,00000000h
$L50:   mov     eax,[ebx+$S48]
        xor     ebx,+010h
        mov     edx,[ebx+$S48]
        xor     ebx,+010h
        xor     eax,edx
        mov     [ebx+$S49],eax
        add     ebx,+004h
        loop    short $L50
        mov     ebx,$S23
        mov     esi,[ebx]
        mov     edi,[ebx+004h]
        call    $L51
        mov     $S52,esi
        mov     $S53,edi
        mov     eax,40104100h
        and     eax,edi
        mov     $S54,eax
        xor     eax,40104100h
        mov     $S55,eax
        mov     eax,00420082h
        and     eax,edi
        mov     $S56,eax
        xor     eax,00420082h
        mov     $S57,eax
        xor     esi,-001h
        xor     edi,-001h
        mov     $S58,esi
        mov     $S59,edi
        and     edi,20080820h
        mov     $S60,edi
        mov     edi,$S53
        and     edi,20080820h
        mov     $S61,edi
        mov     ebx,$S24
        mov     esi,[ebx]
        mov     edi,[ebx+004h]
        mov     ebx,$S22
        xor     esi,[ebx]
        xor     edi,[ebx+004h]
        call    $L51
        mov     $S62,esi
        mov     $S63,edi
        xor     esi,-001h
        xor     edi,-001h
        mov     $S64,esi
        mov     $S65,edi
        xor     ebx,ebx
        xor     ecx,ecx
        cmp     dword ptr $S12,+001h
        jnz     short $L66
        call    $L67
        jmp     $L68

$L66:   cmp     dword ptr $S12,+002h
        jnz     short $L69
        call    $L70
        jmp     $L68
$L69:   cmp     dword ptr $S12,+003h
        jnz     short $L71
        call    $L72
        jmp     $L68
$L71:   cmp     dword ptr $S12,+004h
        jnz     short $L73
        call    $L74
        jmp     $L68
$L73:   cmp     dword ptr $S12,+005h
        jnz     short $L75
        call    $L76
        jmp     $L68
$L75:   cmp     dword ptr $S12,+006h
        jnz     short $L77
        call    $L78
        jmp     $L68
$L77:   cmp     dword ptr $S12,+007h
        jnz     short $L79
        call    $L80
        jmp     $L68
$L79:   cmp     dword ptr $S12,+008h
        jnz     short $L81
        call    $L82
        jmp     $L68
$L81:   cmp     dword ptr $S12,+009h
        jnz     short $L83
        call    $L84
        jmp     $L68
$L83:   cmp     dword ptr $S12,+00Ah
        jnz     short $L85
        call    $L86
        jmp     $L68
$L85:   cmp     dword ptr $S12,+00Bh
        jnz     short $L87
        call    $L88
        jmp     $L68
$L87:   cmp     dword ptr $S12,+00Ch
        jnz     short $L89
        call    $L90
        jmp     $L68
$L89:   cmp     dword ptr $S12,+00Dh
        jnz     short $L91
        call    $L92
        jmp     $L68
$L91:   cmp     dword ptr $S12,+00Eh
        jnz     short $L93
        call    $L94
        jmp     $L68
$L93:   cmp     dword ptr $S12,+00Fh
        jnz     short $L95
        call    $L96
        jmp     $L68
$L95:   cmp     dword ptr $S12,+010h
        jnz     short $L97
        call    $L98
        jmp     $L68
$L97:   cmp     dword ptr $S12,+011h
        jnz     short $L99
        call    $L100
        jmp     $L68
$L99:   cmp     dword ptr $S12,+012h
        jnz     short $L101
        call    $L102
        jmp     $L68
$L101:  cmp     dword ptr $S12,+013h
        jnz     short $L103
        call    $L104
        jmp     $L68
$L103:  cmp     dword ptr $S12,+014h
        jnz     short $L105
        call    $L106
        jmp     short $L68
$L105:  cmp     dword ptr $S12,+015h
        jnz     short $L107
        call    $L108
        jmp     short $L68
$L107:  cmp     dword ptr $S12,+016h
        jnz     short $L109
        call    $L110
        jmp     short $L68
$L109:  cmp     dword ptr $S12,+017h
        jnz     short $L111
        call    $L112
        jmp     short $L68
$L111:  cmp     dword ptr $S12,+018h
        jnz     short $L113
        call    $L114
        jmp     short $L68
$L113:  cmp     dword ptr $S12,+019h
        jnz     short $L115
        call    $L116
        jmp     short $L68
$L115:  cmp     dword ptr $S12,+01Ah
        jnz     short $L117
        call    $L118
        jmp     short $L68
$L117:  cmp     dword ptr $S12,+01Bh
        jnz     short $L119
        call    $L120
        jmp     short $L68
$L119:  call    $L121
$L68:   mov     edx,00000000h
        cmp     dword ptr $S20,+001h
        jnz     short $L122
        mov     eax,00000000h
        jmp     short $L123
$L122:  mov     eax,00000001h
$L123:  jmp     short $L4
$L129:  cmp     dword ptr $S20,+001h
        jnz     short $L124
        mov     eax,00000000h
        jmp     short $L4
$L124:  mov     eax,00000002h
$L4:    mov     edx,00000000h
        pop     ecx
        pop     ebx
        pop     ebp
        pop     edi
        pop     esi
        ret
$L127:  mov     ebp,$S15[ebp*4]
        xor     eax,eax
        xor     edx,edx
        xor     ecx,ecx
        mov     cl,[ebp]
        mov     dl,[ebp+002h]
        mov     al,[ebp+001h]
        inc     ebp
$L126:  mov     edi,$S125[edx*4]
        mov     esi,$S1[eax*4]
        add     ebp,+002h
        xor     esi,edi
        mov     dl,[ebp+001h]
        mov     $S1[eax*4],esi
        mov     al,[ebp]
        loop    short $L126
        xor     ecx,ecx
        ret
$L67:   call    near ptr $L70
        mov     ebp,00000001h
        call    near ptr $L127
$L70:   call    near ptr $L72
        mov     ebp,00000002h
        call    near ptr $L127
$L72:   call    near ptr $L74
        mov     ebp,00000003h
        call    near ptr $L127
$L74:   call    near ptr $L76
        mov     ebp,00000004h
        call    near ptr $L127
$L76:   call    near ptr $L78
        mov     ebp,00000005h
        call    $L127
$L78:   call    near ptr $L80
        mov     ebp,00000006h
        call    $L127
$L80:   call    near ptr $L82
        mov     ebp,00000007h
        call    $L127
$L82:   call    near ptr $L84
        mov     ebp,00000008h
        call    $L127
$L84:   call    near ptr $L86
        mov     ebp,00000009h
        call    $L127
$L86:   call    near ptr $L88
        mov     ebp,0000000Ah
        call    $L127
$L88:   call    near ptr $L90
        mov     ebp,0000000Bh
        call    $L127
$L90:   call    near ptr $L92
        mov     ebp,0000000Ch
        call    $L127
$L92:   call    near ptr $L94
        mov     ebp,0000000Dh
        call    $L127
$L94:   call    near ptr $L96
        mov     ebp,0000000Eh
        call    $L127
$L96:   call    near ptr $L98
        mov     ebp,0000000Fh
        call    $L127
$L98:   call    near ptr $L100
        mov     ebp,00000010h
        call    $L127
$L100:  call    near ptr $L102
        mov     ebp,00000011h
        call    $L127
$L102:  call    near ptr $L104
        mov     ebp,00000012h
        call    $L127
$L104:  call    near ptr $L106
        mov     ebp,00000013h
        call    $L127
$L106:  call    _bbryd_continue
        or      eax,eax
        jnz     short $L128
        mov     esp,$S21
        jmp     $L129
$L128:  xor     ebx,ebx
        xor     ecx,ecx
        call    near ptr $L108
        mov     ebp,00000014h
        call    $L127
$L108:  call    near ptr $L110
        mov     ebp,00000015h
        call    $L127
$L110:  call    near ptr $L112
        mov     ebp,00000016h
        call    $L127
$L112:  call    near ptr $L114
        mov     ebp,00000017h
        call    $L127
$L114:  call    near ptr $L116
        mov     ebp,00000018h
        call    $L127
$L116:  cmp     dword ptr $S6,+001h
        jnz     short $L130
        jmp     short $L131
$L130:  call    near ptr $L118
        mov     ebp,00000019h
        call    $L127
$L118:  call    near ptr $L120
        mov     ebp,0000001Ah

        call    $L127
$L120:  call    $L121
        mov     ebp,0000001Bh
        call    $L127
        jmp     $L121
$L131:  call    $L132
        mov     eax,$S133
        mov     edx,$S134
        xor     eax,00000800h
        xor     edx,00080000h
        mov     $S133,eax
        mov     $S134,edx
        mov     eax,$S135
        mov     edx,$S136
        xor     eax,00000200h
        xor     edx,00010000h
        mov     $S135,eax
        mov     $S136,edx
        mov     eax,$S137
        mov     edx,$S138
        xor     eax,00000800h
        xor     edx,00020000h
        mov     $S137,eax
        mov     $S138,edx
        mov     eax,$S139
        mov     edx,$S140
        xor     eax,00008000h
        xor     edx,00000100h
        mov     $S139,eax
        mov     $S140,edx
        mov     eax,$S141
        mov     edx,$S142
        xor     eax,00040000h
        xor     edx,+020h
        mov     $S141,eax
        mov     $S142,edx
        mov     eax,$S143
        mov     ebp,$S144
        xor     eax,00000400h
        mov     edx,$S145
        xor     ebp,00000080h
        mov     $S143,eax
        xor     edx,00001000h
        mov     $S144,ebp
        mov     $S145,edx
$L132:  call    $L146
        mov     eax,$S147
        mov     edx,$S148
        xor     eax,+010h
        xor     edx,00000800h
        mov     $S147,eax
        mov     $S148,edx
        mov     eax,$S135
        mov     edx,$S149
        xor     eax,00020000h
        xor     edx,+008h
        mov     $S135,eax
        mov     $S149,edx
        mov     eax,$S150
        mov     edx,$S151
        xor     eax,00004000h
        xor     edx,+040h
        mov     $S150,eax
        mov     $S151,edx
        mov     eax,$S152
        mov     edx,$S139
        xor     eax,00002000h
        xor     edx,+020h
        mov     $S152,eax
        mov     $S139,edx
        mov     eax,$S153
        mov     edx,$S154
        xor     eax,00000400h
        xor     edx,00000080h
        mov     $S153,eax
        mov     $S154,edx
        mov     eax,$S141
        mov     edx,$S142
        xor     eax,00000800h
        xor     edx,+040h
        mov     $S141,eax
        mov     $S142,edx
        mov     eax,$S143
        mov     edx,$S155
        xor     eax,00004000h
        xor     edx,00010000h
        mov     $S143,eax
        mov     $S155,edx
$L146:  call    $L121
        mov     eax,$S133
        mov     edx,$S156
        xor     eax,00000100h
        xor     edx,00004000h
        mov     $S133,eax
        mov     $S156,edx
        mov     eax,$S134
        mov     edx,$S157
        xor     eax,+040h
        xor     edx,00002000h
        mov     $S134,eax
        mov     $S157,edx
        mov     eax,$S136
        mov     edx,$S150
        xor     eax,00000400h
        xor     edx,00008000h
        mov     $S136,eax
        mov     $S150,edx
        mov     eax,$S137
        mov     edx,$S139
        xor     eax,00001000h
        xor     edx,+040h
        mov     $S137,eax
        mov     $S139,edx
        mov     eax,$S153
        mov     edx,$S158
        xor     eax,00004000h
        xor     edx,+010h
        mov     $S153,eax
        mov     $S158,edx
        mov     eax,$S159
        mov     edx,$S143
        xor     eax,00000080h
        xor     edx,+004h
        mov     $S159,eax
        mov     $S143,edx
        mov     eax,$S144
        mov     edx,$S145
        xor     eax,00008000h
        xor     edx,+008h
        mov     $S144,eax
        mov     $S145,edx
$L121:  mov     eax,$S17
        mov     edx,$S157
        and     edx,00000800h
        cmp     eax,edx
        jz      short $L160
        xor     eax,00000800h
        mov     $S17,eax
        mov     eax,$S159
        mov     edx,$S143
        xor     eax,00000800h
        xor     edx,+040h
        mov     $S159,eax
        mov     $S143,edx
        mov     eax,$S144
        mov     edx,$S155
        xor     eax,00004000h
        xor     edx,00000200h
        mov     $S144,eax
        mov     $S155,edx
$L160:  mov     eax,$S18
        mov     edx,$S148
        and     edx,10000000h
        cmp     eax,edx
        jz      short $L161
        xor     eax,10000000h
        mov     $S18,eax
        mov     eax,$S143
        mov     ebp,$S162
        xor     eax,00080000h
        mov     edx,$S145
        xor     ebp,+002h
        mov     $S143,eax
        xor     edx,04000000h
        mov     $S162,ebp
        mov     $S145,edx
$L161:  mov     esi,$S52
        mov     edi,$S53
        mov     eax,$S145
        xor     eax,edi
        mov     edx,$S155
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        xor     ecx,+020h
        xor     esi,ebp
        mov     bl,ah
        mov     edi,[ecx+$S165]

        and     eax,000000FFh
        xor     ebp,edi
        mov     $S166,ebp
        mov     cl,dh
        mov     edi,[ecx+$S35]
        xor     esi,edi
        and     edx,000000FFh
        mov     edi,[ebx+$S167]
        xor     ebx,+004h
        xor     esi,edi
        mov     ecx,[eax+$S27]
        xor     esi,ecx
        mov     ebx,[ebx+$S167]
        mov     ecx,[edx+$S48]
        xor     ebx,edi
        xor     esi,ecx
        mov     $S168,ebx
        xor     ebx,ebx
        xor     ecx,ecx
        mov     $S169,esi
        mov     esi,$S58
        mov     edi,$S59
        mov     eax,$S145
        xor     eax,edi
        mov     edx,$S155
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        xor     ecx,+020h
        xor     esi,ebp
        mov     bl,ah
        and     eax,000000FFh
        mov     edi,[ecx+$S165]
        xor     ecx,ecx
        xor     ebp,edi
        mov     cl,dh
        mov     edi,[ecx+$S35]
        xor     esi,edi
        and     edx,000000FFh
        mov     edi,[ebx+$S167]
        xor     ebx,+004h
        xor     esi,edi
        mov     ecx,[eax+$S27]
        xor     esi,ecx
        mov     ebx,[ebx+$S167]
        mov     ecx,[edx+$S48]
        xor     ebx,edi
        xor     esi,ecx
        mov     $S170,ebx
        mov     $S171,ebp
        mov     $S172,esi
        xor     ebx,ebx
        xor     ecx,ecx
        call    $L173
        mov     eax,$S133
        mov     edx,$S174
        xor     eax,+004h
        xor     edx,00800000h
        mov     $S133,eax
        mov     $S174,edx
        mov     eax,$S134
        mov     edx,$S135
        xor     eax,+008h
        xor     edx,02000000h
        mov     $S134,eax
        mov     $S135,edx
        mov     eax,$S149
        mov     edx,$S150
        xor     eax,04000000h
        xor     edx,00400000h
        mov     $S149,eax
        mov     $S150,edx
        mov     eax,$S137
        mov     edx,$S138
        xor     eax,10000000h
        xor     edx,01000000h
        mov     $S137,eax
        mov     $S138,edx
        mov     eax,$S139
        mov     edx,$S140
        xor     eax,80000000h
        xor     edx,80000000h
        mov     $S139,eax
        mov     $S140,edx
        mov     eax,$S154
        mov     edx,$S159
        xor     eax,00200000h
        xor     edx,08000000h
        mov     $S154,eax
        mov     $S159,edx
        mov     eax,$S143
        mov     edx,$S162
        xor     eax,08000000h
        xor     edx,04000000h
        mov     $S143,eax
        mov     $S162,edx
$L173:  mov     esi,$S64
        mov     edi,$S65
        mov     eax,$S147
        xor     eax,edi
        mov     edx,$S133
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     ebx,+010h
        xor     esi,ebp
        mov     edi,[ecx+$S165]
        mov     cl,dh
        xor     esi,edi
        mov     edi,[ebx+$S32]
        xor     edi,ebp
        xor     ebx,ebx
        mov     $S175,edi
        mov     bl,ah
        and     eax,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        xor     ecx,+004h
        and     edx,000000FFh
        mov     edi,[eax+$S27]
        xor     esi,edi
        mov     edi,[ecx+$S35]
        mov     eax,[edx+$S48]
        xor     edx,+010h
        xor     edi,ebp
        xor     ecx,ecx
        mov     $S176,edi
        mov     edx,[edx+$S48]
        xor     esi,eax
        xor     edx,eax
        mov     $S177,edx
        mov     $S178,esi
        mov     esi,$S62
        mov     edi,$S63
        mov     eax,$S147
        xor     eax,edi
        mov     edx,$S133
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     ebx,+010h
        xor     esi,ebp
        mov     edi,[ecx+$S165]
        mov     cl,dh
        xor     esi,edi
        mov     edi,[ebx+$S32]
        xor     edi,ebp
        xor     ebx,ebx
        mov     $S179,edi
        mov     bl,ah
        and     eax,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        xor     ecx,+004h
        and     edx,000000FFh
        mov     edi,[eax+$S27]
        xor     esi,edi
        mov     edi,[ecx+$S35]
        mov     eax,[edx+$S48]
        xor     edx,+010h
        xor     edi,ebp
        xor     ecx,ecx
        mov     $S180,edi
        mov     edx,[edx+$S48]
        xor     esi,eax
        xor     edx,eax
        mov     $S181,edx
        mov     $S182,esi
        call    $L183
        mov     eax,$S133
        mov     edx,$S156
        xor     eax,+001h
        xor     edx,08000000h
        mov     $S133,eax
        mov     $S156,edx
        mov     eax,$S148
        mov     edx,$S135
        xor     eax,08000000h
        xor     edx,04000000h
        mov     $S148,eax
        mov     $S135,edx
        mov     eax,$S136
        mov     edx,$S184
        xor     eax,+004h
        xor     edx,20000000h
        mov     $S136,eax
        mov     $S184,edx
        mov     eax,$S137
        mov     edx,$S138
        xor     eax,00080000h
        xor     edx,+002h
        mov     $S137,eax
        mov     $S138,edx

        mov     eax,$S139
        mov     edx,$S140
        xor     eax,04000000h
        xor     edx,00400000h
        mov     $S139,eax
        mov     $S140,edx
        mov     eax,$S154
        mov     edx,$S141
        xor     eax,10000000h
        xor     edx,01000000h
        mov     $S154,eax
        mov     $S141,edx
        mov     eax,$S142
        mov     ebp,$S143
        xor     eax,00040000h
        mov     edx,$S162
        xor     ebp,40000000h
        mov     $S142,eax
        xor     edx,00800000h
        mov     $S143,ebp
        mov     $S162,edx
        mov     eax,$S182
        mov     ebp,$S179
        mov     edx,$S178
        xor     eax,ebp
        mov     ebp,$S175
        mov     $S182,eax
        xor     edx,ebp
        mov     $S178,edx
$L183:  mov     ebx,$S170
        mov     ebp,$S171
        mov     esi,$S172
        and     ebx,0FC00000h
        shr     ebx,14h
        and     ebp,0FC00000h
        shr     ebp,14h
        mov     edx,$S162
        xor     edx,esi
        and     esi,08200401h
        and     edx,0FC00000h
        mov     $S185,esi
        shr     edx,14h
        mov     ecx,$S168
        mov     edi,$S60
        and     ecx,0FC00000h
        mov     eax,[edx+$S35]
        xor     edx,ebx
        shr     ecx,14h
        xor     eax,edi
        mov     esi,[edx+$S35]
        xor     edx,ebp
        xor     esi,edi
        mov     $S186,eax
        mov     eax,[edx+$S35]
        xor     edx,ebx
        mov     $S187,esi
        xor     eax,edi
        mov     ebx,[edx+$S35]
        mov     edx,$S162
        mov     esi,$S169
        xor     ebx,edi
        xor     edx,esi
        mov     $S188,eax
        and     edx,0FC00000h
        mov     $S189,ebx
        shr     edx,14h
        mov     eax,esi
        and     eax,08200401h
        mov     ebp,edx
        xor     ebp,ecx
        mov     edi,$S166
        mov     $S190,eax
        and     edi,0FC00000h
        shr     edi,14h
        mov     eax,$S61
        mov     ebx,[ebp+$S35]
        xor     ebp,edi
        xor     ebx,eax
        mov     esi,[edx+$S35]
        mov     $S191,ebx
        xor     esi,eax
        mov     ebx,[ebp+$S35]
        xor     ebp,ecx
        xor     ebx,eax
        xor     ecx,ecx
        mov     $S192,ebx
        mov     edx,[ebp+$S35]
        mov     $S193,esi
        xor     edx,eax
        mov     $S194,edx
        xor     ebx,ebx
        call    $L195
        mov     eax,$S133
        mov     edx,$S174
        xor     eax,00010000h
        xor     edx,+010h
        mov     $S133,eax
        mov     $S174,edx
        mov     eax,$S134
        mov     edx,$S157
        xor     eax,00000080h
        xor     edx,+004h
        mov     $S134,eax
        mov     $S157,edx
        mov     eax,$S149
        mov     edx,$S150
        xor     eax,00008000h
        xor     edx,00000100h
        mov     $S149,eax
        mov     $S150,edx
        mov     eax,$S138
        mov     edx,$S139
        xor     eax,00040000h
        xor     edx,00002000h
        mov     $S138,eax
        mov     $S139,edx
        mov     eax,$S140
        mov     edx,$S196
        xor     eax,00000400h
        xor     edx,00008000h
        mov     $S140,eax
        mov     $S196,edx
        mov     eax,$S158
        mov     ebp,$S197
        xor     eax,00001000h
        mov     edx,$S162
        xor     ebp,00080000h
        mov     $S158,eax
        xor     edx,00000200h
        mov     $S197,ebp
        mov     $S162,edx
        mov     eax,$S182
        mov     ebp,$S181
        mov     edx,$S178
        xor     eax,ebp
        mov     ebp,$S177
        mov     $S182,eax
        xor     edx,ebp
        mov     $S178,edx
$L195:  call    $L198
        mov     eax,$S133
        mov     edx,$S174
        xor     eax,00400000h
        xor     edx,00400000h
        mov     $S133,eax
        mov     $S174,edx
        mov     eax,$S134
        mov     edx,$S157
        xor     eax,40000000h
        xor     edx,80000000h
        mov     $S134,eax
        mov     $S157,edx
        mov     eax,$S136
        mov     edx,$S184
        xor     eax,80000000h
        xor     edx,00200000h
        mov     $S136,eax
        mov     $S184,edx
        mov     eax,$S138
        mov     edx,$S153
        xor     eax,08000000h
        xor     edx,00100000h
        mov     $S138,eax
        mov     $S153,edx
        mov     eax,$S158
        mov     edx,$S159
        xor     eax,00800000h
        xor     edx,+008h
        mov     $S158,eax
        mov     $S159,edx
        mov     eax,$S197
        mov     edx,$S144
        xor     eax,02000000h
        xor     edx,04000000h
        mov     $S197,eax
        mov     $S144,edx
        mov     eax,$S182
        mov     ebp,$S180
        mov     edx,$S178
        xor     eax,ebp
        mov     ebp,$S176
        mov     $S182,eax
        xor     edx,ebp
        mov     $S178,edx
$L198:  mov     dword ptr $S199,offset $S193

        xor     ebx,ebx
        xor     ecx,ecx
        mov     esi,$S178
        mov     edi,$S65
        mov     eax,$S174
        xor     eax,esi
        mov     edx,$S156
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     $S200,ebx
        mov     $S201,ebp
        mov     ebp,[ebx+$S44]
        mov     $S202,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     ebp,[ecx+$S35]
        xor     edi,ebp
        mov     ebp,[ecx+$S46]
        mov     $S203,ebp
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,[edx+$S48]
        xor     edi,ebp
        mov     ebp,[edx+$S49]
        mov     $S204,ebp
        mov     $S205,edi
        mov     esi,$S182
        mov     edi,$S63
        mov     eax,$S174
        xor     eax,esi
        mov     edx,$S156
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     $S206,ebx
        mov     $S207,ebp
        mov     ebp,[ebx+$S44]
        mov     $S208,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     ebp,[ecx+$S35]
        xor     edi,ebp
        mov     ebp,[ecx+$S46]
        mov     $S209,ebp
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,[edx+$S48]
        xor     edi,ebp
        mov     ebp,[edx+$S49]
        mov     $S210,ebp
        mov     $S211,edi
        call    $L212
        mov     dword ptr $S199,offset $S194
        call    $L213
        mov     eax,$S148
        mov     ebp,$S135
        xor     eax,10000000h
        mov     edx,$S149
        xor     ebp,01000000h
        mov     $S148,eax
        xor     edx,00040000h
        mov     $S135,ebp
        mov     $S149,edx
        mov     eax,$S184
        mov     edx,$S151
        xor     eax,40000000h
        xor     edx,00800000h
        mov     $S184,eax
        mov     $S151,edx
        mov     eax,$S138
        mov     edx,$S214
        xor     eax,+001h
        xor     edx,08000000h
        mov     $S138,eax
        mov     $S214,edx
        mov     eax,$S153
        mov     edx,$S196
        xor     eax,08000000h
        xor     edx,04000000h
        mov     $S153,eax
        mov     $S196,edx
        mov     eax,$S141
        mov     edx,$S142
        xor     eax,+004h
        xor     edx,20000000h
        mov     $S141,eax
        mov     $S142,edx
        mov     dword ptr $S199,offset $S192
        mov     esi,$S205
        mov     ebp,$S203
        mov     edi,$S211
        xor     esi,ebp
        mov     ebp,$S209
        mov     $S205,esi
        xor     edi,ebp
        mov     esi,$S182
        mov     $S211,edi
        call    $L212
        mov     dword ptr $S199,offset $S191
        call    $L213
        mov     eax,$S156
        mov     edx,$S148
        xor     eax,+004h
        xor     edx,20000000h
        mov     $S156,eax
        mov     $S148,edx
        mov     eax,$S157
        mov     edx,$S136
        xor     eax,00080000h
        xor     edx,+002h
        mov     $S157,eax
        mov     $S136,edx
        mov     eax,$S137
        mov     edx,$S138
        xor     eax,00400000h
        xor     edx,40000000h
        mov     $S137,eax
        mov     $S138,edx
        mov     eax,$S214
        mov     edx,$S153
        xor     eax,01000000h
        xor     edx,00040000h
        mov     $S214,eax
        mov     $S153,edx
        mov     eax,$S154
        mov     edx,$S141
        xor     eax,40000000h
        xor     edx,00800000h
        mov     $S154,eax
        mov     $S141,edx
        mov     eax,$S159
        mov     ebp,$S144
        xor     eax,+001h
        mov     edx,$S155
        xor     ebp,00100000h
        mov     $S159,eax
        xor     edx,04000000h
        mov     $S144,ebp
        mov     $S155,edx
        mov     ebx,$S200
        mov     edx,$S206
        xor     ebx,+040h
        mov     eax,$S201
        xor     edx,+040h
        mov     esi,$S205
        xor     esi,eax
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ebx+$S44]
        mov     edi,$S211
        mov     eax,$S207

        mov     $S202,ebp
        xor     edi,eax
        xor     ebx,ebx
        mov     ebp,[edx+$S32]
        xor     edi,ebp
        mov     ebp,[edx+$S44]
        mov     $S208,ebp
        mov     $S205,esi
        mov     $S211,edi
        mov     esi,$S182
        call    $L212
        mov     dword ptr $S199,offset $S192
        call    $L213
        mov     eax,$S148
        mov     ebp,$S135
        xor     eax,10000000h
        mov     edx,$S149
        xor     ebp,01000000h
        mov     $S148,eax
        xor     edx,00040000h
        mov     $S135,ebp
        mov     $S149,edx
        mov     eax,$S184
        mov     edx,$S151
        xor     eax,40000000h
        xor     edx,00800000h
        mov     $S184,eax
        mov     $S151,edx
        mov     eax,$S138
        mov     edx,$S214
        xor     eax,+001h
        xor     edx,08000000h
        mov     $S138,eax
        mov     $S214,edx
        mov     eax,$S153
        mov     edx,$S196
        xor     eax,08000000h
        xor     edx,04000000h
        mov     $S153,eax
        mov     $S196,edx
        mov     eax,$S141
        mov     edx,$S142
        xor     eax,+004h
        xor     edx,20000000h
        mov     $S141,eax
        mov     $S142,edx
        mov     dword ptr $S199,offset $S194
        mov     esi,$S205
        mov     ebp,$S203
        mov     edi,$S211
        xor     esi,ebp
        mov     ebp,$S209
        mov     $S205,esi
        xor     edi,ebp
        mov     esi,$S182
        mov     $S211,edi
        call    $L212
        mov     dword ptr $S199,offset $S193
$L213:  mov     eax,$S157
        mov     ebp,$S136
        xor     eax,00000800h
        mov     edx,$S184
        xor     ebp,00020000h
        mov     $S157,eax
        xor     edx,+008h
        mov     $S136,ebp
        mov     $S184,edx
        mov     eax,$S151
        mov     edx,$S138
        xor     eax,00004000h
        xor     edx,+040h
        mov     $S151,eax
        mov     $S138,edx
        mov     eax,$S214
        mov     edx,$S153
        xor     eax,00040000h
        xor     edx,+020h
        mov     $S214,eax
        mov     $S153,edx
        mov     eax,$S154
        mov     edx,$S158
        xor     eax,00000400h
        xor     edx,00000080h
        mov     $S154,eax
        mov     $S158,edx
        mov     esi,$S205
        mov     ebp,$S204
        mov     edi,$S211
        xor     esi,ebp
        mov     ebp,$S210
        mov     $S205,esi
        xor     edi,ebp
        mov     esi,$S182
        mov     $S211,edi
$L212:  xor     ebx,ebx
        mov     eax,$S148
        xor     eax,edi
        mov     edx,$S134
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S27]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     eax,$S157
        xor     eax,esi
        mov     edx,$S135
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     ebp,[ecx+$S35]
        xor     edi,ebp
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,[edx+$S48]
        xor     edi,ebp
        mov     eax,$S149
        xor     eax,edi
        mov     edx,$S136
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S27]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     eax,$S184
        xor     eax,esi
        mov     edx,$S150
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     ebp,[ecx+$S35]
        xor     edi,ebp
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,[edx+$S48]
        xor     edi,ebp
        mov     eax,$S137
        xor     eax,edi
        mov     edx,$S151
        xor     edx,edi
        and     eax,0FCFCFCFCh

        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S27]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     eax,$S152
        xor     eax,esi
        mov     edx,$S138
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     ebp,[ecx+$S35]
        xor     edi,ebp
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,[edx+$S48]
        xor     edi,ebp
        mov     eax,$S139
        xor     eax,edi
        mov     edx,$S214
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S27]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     eax,$S153
        xor     eax,esi
        mov     edx,$S140
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     ebp,[ecx+$S35]
        xor     edi,ebp
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,[edx+$S48]
        xor     edi,ebp
        mov     eax,$S154
        xor     eax,edi
        mov     edx,$S196
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S27]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     eax,$S158
        xor     eax,esi
        mov     edx,$S141
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     cl,ah
        rol     edx,04h
        mov     $S215,eax
        mov     bl,dl
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     cl,dh
        mov     $S216,edx
        mov     bl,ah
        mov     ebp,[ecx+$S165]
        mov     cl,byte ptr $S216 + 00002h
        xor     edi,ebp
        and     eax,000000FFh
        mov     ebp,[ecx+$S48]
        mov     edx,$S159
        xor     edi,ebp
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,$S19
        xor     edx,edi
        mov     eax,$S199
        and     edx,0FC00000h
        mov     cl,byte ptr $S216 + 00003h
        shr     edx,14h
        mov     eax,[ebp+eax]
        mov     $S217,esi
        and     esi,20080820h
        xor     eax,esi
        mov     edx,[edx+$S35]
        cmp     edx,eax
        jz      $L218
$L233:  or      ebp,ebp
        jz      $L219
        mov     edi,$S205
        mov     ebp,$S202
        mov     esi,$S178
        xor     edi,ebp
$L222:  xor     ecx,ecx
        mov     eax,$S148
        xor     eax,edi
        mov     edx,$S134
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S27]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     eax,$S157
        xor     eax,esi
        mov     edx,$S135
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     edi,ebp

        mov     ebp,[ecx+$S35]
        xor     edi,ebp
        mov     ebp,[eax+$S28]
        xor     edi,ebp
        mov     ebp,[edx+$S48]
        xor     edi,ebp
        mov     eax,$S149
        xor     eax,edi
        mov     edx,$S136
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S27]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     eax,$S184
        xor     eax,esi
        mov     edx,$S150
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     ebp,[ecx+$S35]
        xor     edi,ebp
        mov     ebp,[eax+$S30]
        xor     edi,ebp
        mov     ebp,[edx+$S48]
        xor     edi,ebp
        mov     eax,$S137
        xor     eax,edi
        mov     edx,$S151
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S33]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S27]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     eax,$S152
        xor     eax,esi
        mov     edx,$S138
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     ebp,[ecx+$S36]
        xor     edi,ebp
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,[edx+$S48]
        xor     edi,ebp
        mov     eax,$S139
        xor     eax,edi
        mov     edx,$S214
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S38]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S27]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     eax,$S153
        xor     eax,esi
        mov     edx,$S140
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     ebp,[ecx+$S35]
        xor     edi,ebp
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,[edx+$S48]
        xor     edi,ebp
        mov     eax,$S154
        xor     eax,edi
        mov     edx,$S196
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S40]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     eax,$S158
        xor     eax,esi
        mov     edx,$S141
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     cl,ah
        rol     edx,04h
        mov     $S215,eax
        mov     bl,dl
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     ebp,[ebx+$S42]
        xor     edi,ebp
        mov     cl,dh
        mov     $S216,edx
        mov     bl,ah
        mov     ebp,[ecx+$S165]
        mov     cl,byte ptr $S216 + 00002h
        xor     edi,ebp
        and     eax,000000FFh
        mov     ebp,[ecx+$S48]
        mov     edx,$S159
        xor     edi,ebp
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,$S19
        xor     edx,edi
        mov     eax,$S199
        and     edx,0FC00000h

        mov     cl,byte ptr $S216 + 00003h
        shr     edx,14h
        mov     eax,[ebp+eax]
        mov     $S217,esi
        and     esi,20080820h
        xor     eax,esi
        mov     edx,[edx+$S35]
        cmp     edx,eax
        jz      short $L220
$L224:  or      ebp,ebp
        jnz     short $L221
        ret
$L221:  mov     edi,$S211
        mov     ebp,$S208
        mov     esi,$S182
        xor     edi,ebp
        mov     dword ptr $S19,00000000h
        jmp     $L222
$L220:  mov     bl,byte ptr $S215
        mov     ebp,[ecx+$S35]
        xor     edi,ebp
        mov     ebp,[ebx+$S163]
        xor     edi,ebp
        mov     esi,$S217
        mov     eax,$S142
        xor     eax,edi
        mov     edx,$S159
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     $S216,edx
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S164]
        mov     bl,ah
        and     eax,000000FFh
        xor     esi,ebp
        mov     cl,dh
        mov     ebp,[eax+$S27]
        xor     ebx,00000080h
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[ebx+$S167]
        mov     edx,$S197
        xor     ebx,ebx
        xor     esi,ebp
        mov     ebp,$S19
        xor     edx,esi
        mov     $S217,edi
        shr     edx,0Ch
        and     edi,08200401h
        mov     ebp,[ebp+$S190]
        and     edx,000000FCh
        xor     ebp,edi
        mov     edx,[edx+$S48]
        cmp     edx,ebp
        jz      short $L223
        mov     ebp,$S19
        jmp     $L224
$L223:  mov     bl,byte ptr $S216 + 00002h
        mov     cl,byte ptr $S216 + 00001h
        mov     eax,$S17
        mov     edx,$S157
        and     edx,00000800h
        cmp     eax,edx
        jz      short $L225
        xor     eax,00000800h
        mov     $S17,eax
        mov     eax,$S159
        mov     edx,$S143
        xor     eax,00000800h
        xor     edx,+040h
        mov     $S159,eax
        mov     $S143,edx
        mov     eax,$S144
        mov     edx,$S155
        xor     eax,00004000h
        xor     edx,00000200h
        mov     $S144,eax
        mov     $S155,edx
        xor     ecx,00000080h
$L225:  mov     edi,$S217
        mov     ebp,[ebx+$S48]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        xor     ecx,ecx
        xor     esi,ebp
        mov     eax,$S18
        mov     edx,$S148
        and     edx,10000000h
        cmp     eax,edx
        jz      short $L226
        xor     eax,10000000h
        mov     $S18,eax
        mov     eax,$S143
        mov     ebp,$S162
        xor     eax,00080000h
        mov     edx,$S145
        xor     ebp,+002h
        mov     $S143,eax
        xor     edx,04000000h
        mov     $S162,ebp
        mov     $S145,edx
$L226:  mov     eax,$S143
        xor     eax,esi
        mov     edx,$S197
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        xor     ebx,+008h
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     ebx,ebx
        xor     edi,ebp
        mov     ebp,[ecx+$S165]
        mov     cl,dh
        and     edx,000000FFh
        xor     edi,ebp
        mov     ebp,$S19
        mov     ebp,[ebp+$S56]
        mov     $S217,esi
        mov     edx,[edx+$S48]
        xor     edi,edx
        mov     edx,[ecx+$S35]
        xor     edi,edx
        mov     edx,$S144
        xor     edx,edi
        and     esi,00420082h
        shr     edx,1Ah
        xor     ebp,esi
        mov     edx,$S167[edx*4]
        cmp     edx,ebp
        jz      short $L227
        mov     ebp,$S19
        jmp     $L224
$L227:  mov     bl,ah
        and     eax,000000FFh
        mov     esi,$S217
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     eax,$S144
        xor     eax,edi
        mov     edx,$S162
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        xor     esi,ebp
        mov     bl,dl
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        xor     eax,+020h
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S27]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     ebp,$S19
        mov     edx,[ebp+$S53]
        cmp     esi,edx
        jnz     $L228
        mov     eax,$S145
        xor     eax,esi
        mov     edx,$S155
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     ecx,+008h
        xor     edi,ebp
        mov     ebp,[ecx+$S35]
        xor     ecx,ecx
        xor     edi,ebp
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,[edx+$S48]
        xor     edi,ebp
        mov     ebp,$S19
        mov     eax,[ebp+$S52]
        cmp     edi,eax
        jnz     short $L228
        call    $L229
        mov     ebp,$S19
        or      ebp,ebp
        jz      short $L230
        xor     esi,-001h
        xor     edi,-001h
$L230:  xor     edi,00100000h
        call    $L231
$L228:  xor     ebx,ebx
        xor     ecx,ecx

        mov     ebp,$S19
        jmp     $L224
$L219:  mov     edi,$S205
        mov     esi,$S178
        mov     dword ptr $S19,00000018h
        jmp     $L212
$L218:  mov     bl,byte ptr $S215
        mov     ebp,[ecx+$S35]
        xor     edi,ebp
        mov     ebp,[ebx+$S163]
        xor     edi,ebp
        mov     esi,$S217
        mov     eax,$S142
        xor     eax,edi
        mov     edx,$S159
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     $S216,edx
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S164]
        mov     bl,ah
        and     eax,000000FFh
        xor     esi,ebp
        mov     cl,dh
        mov     ebp,[eax+$S27]
        mov     edx,$S197
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,$S19
        xor     edx,esi
        mov     $S217,edi
        shr     edx,0Ch
        and     edi,08200401h
        mov     ebp,[ebp+$S190]
        and     edx,000000FCh
        xor     ebp,edi
        mov     edx,[edx+$S48]
        cmp     edx,ebp
        jz      short $L232
        mov     ebp,$S19
        jmp     $L233
$L232:  mov     bl,byte ptr $S216 + 00002h
        mov     cl,byte ptr $S216 + 00001h
        mov     eax,$S17
        mov     edx,$S157
        and     edx,00000800h
        cmp     eax,edx
        jz      short $L234
        xor     eax,00000800h
        mov     $S17,eax
        mov     eax,$S159
        mov     edx,$S143
        xor     eax,00000800h
        xor     edx,+040h
        mov     $S159,eax
        mov     $S143,edx
        mov     eax,$S144
        mov     edx,$S155
        xor     eax,00004000h
        xor     edx,00000200h
        mov     $S144,eax
        mov     $S155,edx
        xor     ecx,00000080h
$L234:  mov     edi,$S217
        mov     ebp,[ebx+$S48]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        xor     ecx,ecx
        xor     esi,ebp
        mov     eax,$S18
        mov     edx,$S148
        and     edx,10000000h
        cmp     eax,edx
        jz      short $L235
        xor     eax,10000000h
        mov     $S18,eax
        mov     eax,$S143
        mov     ebp,$S162
        xor     eax,00080000h
        mov     edx,$S145
        xor     ebp,+002h
        mov     $S143,eax
        xor     edx,04000000h
        mov     $S162,ebp
        mov     $S145,edx
$L235:  mov     eax,$S143
        xor     eax,esi
        mov     edx,$S197
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     ebp,[ecx+$S165]
        mov     cl,dh
        and     edx,000000FFh
        xor     edi,ebp
        mov     ebp,$S19
        mov     ebp,[ebp+$S56]
        mov     $S217,esi
        mov     edx,[edx+$S48]
        xor     edi,edx
        mov     edx,[ecx+$S35]
        xor     edi,edx
        mov     edx,$S144
        xor     edx,edi
        and     esi,00420082h
        shr     edx,1Ah
        xor     ebp,esi
        mov     edx,$S167[edx*4]
        mov     esi,00000001h
        cmp     edx,ebp
        jz      short $L236
        mov     ebp,$S19
        jmp     $L233
$L236:  mov     bl,ah
        and     eax,000000FFh
        mov     esi,$S217
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     eax,$S144
        xor     eax,edi
        mov     edx,$S162
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S27]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     ebp,$S19
        mov     edx,[ebp+$S53]
        cmp     esi,edx
        jnz     $L237
        mov     eax,$S145
        xor     eax,esi
        mov     edx,$S155
        xor     edx,esi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     edi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     edi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     edi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     edi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     edi,ebp
        mov     ebp,[ecx+$S35]
        xor     edi,ebp
        mov     ebp,[eax+$S27]
        xor     edi,ebp
        mov     ebp,[edx+$S48]
        xor     edi,ebp
        mov     ebp,$S19
        mov     eax,[ebp+$S52]
        cmp     edi,eax
        jnz     short $L237
        call    $L229
        mov     ebp,$S19
        or      ebp,ebp
        jz      short $L238
        xor     esi,-001h
        xor     edi,-001h
$L238:  call    $L231
$L237:  xor     ebx,ebx
        xor     ecx,ecx
        mov     ebp,$S19
        jmp     $L233
$L258:  xor     ebx,ebx
        xor     edi,edi
        mov     ch,01h
        jmp     short $L239
$L246:  mov     cl,ds:[ebp]
        or      cl,cl
        jz      short $L240
        mov     esi,00000001h
        ror     esi,cl
        cmp     cl,20h
        jnbe    short $L241
        and     esi,eax
        jmp     short $L242
$L241:  and     esi,edx
$L242:  jz      short $L240
        cmp     ch,20h
        jnbe    short $L243
        add     ebx,+001h
        jmp     short $L240
$L243:  add     edi,+001h
$L240:  cmp     ch,20h
        jnc     short $L244

        add     ebx,ebx
        jmp     short $L245
$L244:  cmp     ch,20h
        jbe     short $L245
        cmp     ch,40h
        jnc     short $L245
        add     edi,edi
$L245:  inc     ebp
        inc     ch
$L239:  cmp     ch,40h
        jbe     short $L246
        mov     ecx,edi
        ret
$L51:   rol     edi,04h
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
$L229:  mov     ecx,0000001Ch
        mov     edx,offset $S247
        mov     esi,00000000h
        mov     edi,00000000h
$L249:  mov     eax,[edx]
        mov     ebx,[edx+004h]
        mov     ebp,[edx+008h]
        and     ebp,$S1[ebx*4]
        jz      short $L248
        or      esi,$S125[eax*4]
$L248:  add     edx,+00Ch
        loop    short $L249
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
$L251:  mov     eax,[edx]
        sub     eax,+020h
        mov     ebx,[edx+004h]
        mov     ebp,[edx+008h]
        and     ebp,$S1[ebx*4]
        jz      short $L250
        or      edi,$S125[eax*4]
$L250:  add     edx,+00Ch
        loop    short $L251
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
$L231:  mov     dword ptr $S20,00000001h
        mov     $S252,esi
        mov     $S253,edi
        mov     esi,offset $S252
        mov     edi,offset $S254
        mov     ecx,00000008h
$L256:  mov     al,[esi]
        and     al,al
        jnp     short $L255
        xor     al,01h
$L255:  mov     [edi],al
        add     esi,+001h
        add     edi,+001h
        loop    short $L256
        push    offset $S254
        call    _bbryd_key_found
        add     esp,+004h
        xor     ebx,ebx
        xor     ecx,ecx
        ret
_Bdesinit:
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
        mov     ebp,offset $S257
        call    $L258
        mov     eax,ebx
        mov     edx,ecx
        mov     dword ptr $S259,00000001h
        mov     esi,offset $S147
        jmp     short $L260
$L264:  push    esi
        mov     ebp,offset $S261
        call    $L258
        cmp     dword ptr $S259,+002h
        jbe     short $L262
        cmp     dword ptr $S259,+009h
        jz      short $L262
        cmp     dword ptr $S259,+010h
        jz      short $L262
        mov     eax,ebx
        mov     edx,ecx
        mov     ebp,offset $S261
        call    $L258
$L262:  mov     eax,ebx
        mov     edx,ecx
        mov     ebp,offset $S263
        call    $L258
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
        inc     dword ptr $S259
$L260:  cmp     dword ptr $S259,+010h
        jbe     short $L264
        push    ebp
        mov     esi,offset $S265
        mov     edi,00000000h
        mov     ebp,00000000h
        jmp     short $L266
$L272:  mov     ah,01h
        jmp     short $L267
$L271:  mov     ebx,00000000h
        test    ah,01h
        jz      short $L268
        lodsb
$L268:  shl     al,1
        jnc     short $L269
        mov     cl,[ebp+$S270]
        mov     edx,00000001h
        ror     edx,cl
        add     ebx,edx
$L269:  inc     ebp
        test    ebp,00000003h
        jnz     short $L268
        sub     ebp,+004h
        rol     ebx,03h
        mov     [edi+$S32],ebx
        inc     ah
        add     edi,+004h
$L267:  cmp     ah,40h
        jbe     short $L271
        add     ebp,+004h
$L266:  cmp     ebp,+020h
        jc      short $L272
        pop     ebp
        pop     ecx
        pop     ebx
        pop     ebp
        pop     edi
        pop     esi
        ret
$L9:    shl     edx,1
        shl     eax,1
        jnc     short $L273
        or      edx,+001h
$L273:  ret
_Bdesencrypt:
        push    esi
        push    edi
        push    ebp
        push    ebx
        push    ecx
        mov     eax,[esp+018h]
        mov     esi,[eax]
        mov     edi,[eax+004h]
        mov     dword ptr $S274,00000000h
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
$L275:  mov     eax,[ebp+$S147]
        xor     eax,edi
        mov     edx,[ebp+$S133]
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S27]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     ebp,$S274
        mov     eax,esi
        add     ebp,+008h
        mov     $S274,ebp
        mov     esi,edi
        mov     edi,eax
        cmp     ebp,00000080h
        jb      $L275
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
_Bdesdecrypt:
        push    esi
        push    edi
        push    ebp
        push    ebx
        push    ecx
        mov     eax,[esp+018h]
        mov     esi,[eax]
        mov     edi,[eax+004h]
        mov     dword ptr $S274,00000078h
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
$L276:  mov     eax,[ebp+$S147]
        xor     eax,edi
        mov     edx,[ebp+$S133]
        xor     edx,edi
        and     eax,0FCFCFCFCh
        and     edx,0CFCFCFCFh
        mov     bl,al
        rol     edx,04h
        mov     cl,ah
        mov     ebp,[ebx+$S163]
        mov     bl,dl
        xor     esi,ebp
        shr     eax,10h
        mov     ebp,[ecx+$S164]
        xor     esi,ebp
        mov     cl,dh
        shr     edx,10h
        mov     ebp,[ebx+$S32]
        xor     esi,ebp
        mov     ebp,[ecx+$S165]
        mov     bl,ah
        xor     esi,ebp
        mov     cl,dh
        and     eax,000000FFh
        and     edx,000000FFh
        mov     ebp,[ebx+$S167]
        xor     esi,ebp
        mov     ebp,[ecx+$S35]
        xor     esi,ebp
        mov     ebp,[eax+$S27]
        xor     esi,ebp
        mov     ebp,[edx+$S48]
        xor     esi,ebp
        mov     ebp,$S274
        mov     eax,esi
        sub     ebp,+008h
        mov     $S274,ebp
        mov     esi,edi
        mov     edi,eax
        cmp     ebp,+000h
        jnl     $L276
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
_Bkey_byte_to_hex:
        mov     dword ptr $S277,00000000h
        jmp     short $L278
_Bc_key_byte_to_hex:
        mov     dword ptr $S277,00000001h
$L278:  push    esi
        push    edi
        push    ebp
        push    ebx
        push    ecx
        mov     esi,[esp+018h]
        mov     edi,[esp+01Ch]
        mov     ecx,[esp+020h]
        mov     edx,00000000h
$L284:  mov     al,[esi]
        and     al,al
        jnp     short $L279
        mov     edx,00000001h
        xor     al,01h
$L279:  mov     ah,al
        and     ah,0Fh
        and     al,0F0h
        shr     al,04h
        cmp     ah,09h
        jbe     short $L280
        add     ah,37h
        jmp     short $L281
$L280:  add     ah,30h
$L281:  cmp     al,09h
        jbe     short $L282
        add     al,37h
        jmp     short $L283
$L282:  add     al,30h
$L283:  mov     [edi],ax
        add     esi,+001h
        add     edi,+002h
        loop    short $L284
        cmp     dword ptr $S277,+001h
        jnz     short $L285
        mov     byte ptr [edi],00h
$L285:  mov     eax,edx
        pop     ecx
        pop     ebx
        pop     ebp
        pop     edi
        pop     esi
        ret

_TEXT   ENDS
_DATA   SEGMENT

$S265   db      0E0h
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
$S270   db      009h
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
$S257   db      '91)!'
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
$S261   db      002h
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
$S263   db      00Eh
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
$S286   db      00Eh
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
$S287   db      00Fh
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
$S288   db      00Ch
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
$S289   db      00Eh
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
$S290   db      00Eh
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
$S291   db      00Eh
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
$S292   db      00Dh
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
$S293   db      00Eh
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
$S294   db      00Dh
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
$S295   db      00Eh
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
$S296   db      00Dh
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
$S297   db      00Eh
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
$S298   db      00Dh
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
$S299   db      00Eh
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
$S300   db      00Eh
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
$S301   db      00Fh
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
$S302   db      00Dh
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
$S303   db      00Eh
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
$S304   db      00Eh
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
$S305   db      00Eh
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
$S306   db      00Dh
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
$S307   db      00Eh
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
$S308   db      00Dh
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
$S309   db      00Eh
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
$S310   db      00Dh
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
$S311   db      00Dh
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
$S312   db      00Eh
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
$S313   db      00Fh
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
$S314   db      00Dh
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
$S315   db      00Fh
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
$S316   db      00Eh
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
$S317   db      00Dh
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
$S318   db      00Fh
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
$S319   db      00Dh
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
$S320   db      00Dh
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
$S321   db      00Fh
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
$S322   db      00Ch
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
$S323   db      00Dh
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
$S324   db      00Fh
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
$S325   db      00Dh
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
$S326   db      00Eh
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
$S327   db      00Fh
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
$S328   db      00Dh
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
$S329   db      00Fh
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
$S330   db      00Eh
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
$S331   db      00Ch
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
$S332   db      00Fh
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
$S333   db      00Dh
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
$S334   db      00Dh
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
$S335   db      00Fh
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
$S336   db      00Ch
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
$S337   db      00Dh
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
$S338   db      00Fh
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
$S339   db      00Dh
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
$S340   db      00Eh
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
$S341   db      00Eh
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
$S125   dd      01D1F111Dh
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
$S247   db      001h
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
        db      'BrydDES Key Search Library version 1.01.  Core 2. Copyrigh'
        db      't Svend Olaf Mikkelsen, 1995, 1997, 19'
$S14    dd      0202E3839h
        dd      $S286
        dd      $S287
        dd      $S288
        dd      $S289
        dd      $S290
        dd      $S291
        dd      $S292
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S293
        dd      $S294
        dd      $S295
        dd      $S296
        dd      $S297
        dd      $S298
        dd      $S299
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S300
        dd      $S301
        dd      $S302

        dd      $S303
        dd      $S304
        dd      $S305
        dd      $S306
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S307
        dd      $S308
        dd      $S309
        dd      $S310
        dd      $S311
        dd      $S312
        dd      $S313
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S314
        dd      $S315
        dd      $S316
        dd      $S317
        dd      $S318
        dd      $S319
        dd      $S320
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S321
        dd      $S322
        dd      $S323
        dd      $S324
        dd      $S325
        dd      $S326
        dd      $S327
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S328
        dd      $S329
        dd      $S330
        dd      $S331
        dd      $S332
        dd      $S333
        dd      $S334
        db      000h
        db      000h
        db      000h
        db      000h
        dd      $S335
        dd      $S336
        dd      $S337
        dd      $S338
        dd      $S339
        dd      $S340
        dd      $S341

_DATA   ENDS
_BSS    SEGMENT

$S1     dd      1 dup(?)
$S147   dd      1 dup(?)
$S133   dd      1 dup(?)
$S174   dd      1 dup(?)
$S156   dd      1 dup(?)
$S148   dd      1 dup(?)
$S134   dd      1 dup(?)
$S157   dd      1 dup(?)
$S135   dd      1 dup(?)
$S149   dd      1 dup(?)
$S136   dd      1 dup(?)
$S184   dd      1 dup(?)
$S150   dd      1 dup(?)
$S137   dd      1 dup(?)
$S151   dd      1 dup(?)
$S152   dd      1 dup(?)
$S138   dd      1 dup(?)
$S139   dd      1 dup(?)
$S214   dd      1 dup(?)
$S153   dd      1 dup(?)
$S140   dd      1 dup(?)
$S154   dd      1 dup(?)
$S196   dd      1 dup(?)
$S158   dd      1 dup(?)
$S141   dd      1 dup(?)
$S142   dd      1 dup(?)
$S159   dd      1 dup(?)
$S143   dd      1 dup(?)
$S197   dd      1 dup(?)
$S144   dd      1 dup(?)
$S162   dd      1 dup(?)
$S145   dd      1 dup(?)
$S155   dd      1 dup(?)
$S22    dd      1 dup(?)
$S62    dd      1 dup(?)
$S63    dd      1 dup(?)
$S64    dd      1 dup(?)
$S65    dd      1 dup(?)
$S23    dd      1 dup(?)
$S24    dd      2 dup(?)
$S26    dd      1 dup(?)
$S32    dd      64  dup(?)
$S167   dd      64  dup(?)
$S35    dd      64  dup(?)
$S27    dd      64  dup(?)
$S48    dd      64  dup(?)
$S164   dd      64  dup(?)
$S165   dd      64  dup(?)
$S163   dd      64  dup(?)
$S28    dd      64  dup(?)
$S30    dd      64  dup(?)
$S33    dd      64  dup(?)
$S36    dd      64  dup(?)
$S38    dd      64  dup(?)
$S40    dd      64  dup(?)
$S42    dd      64 dup(?)
$S44    dd      64 dup(?)
$S49    dd      64 dup(?)
$S46    dd      192 dup(?)   ; $S46    db      768 dup(?)
$S25    dd      1 dup(?)
$S252   dd      1 dup(?)
$S253   dd      1 dup(?)
$S254   dd      8 dup(?)     ; db      32 dup(?)
$S205   dd      1 dup(?)
$S259   dd      1 dup(?)
$S52    dd      1 dup(?)
$S53    dd      1 dup(?)
$S54    dd      1 dup(?)
$S56    dd      1 dup(?)
$S190   dd      2 dup(?)
$S58    dd      1 dup(?)
$S59    dd      1 dup(?)
$S55    dd      1 dup(?)
$S57    dd      1 dup(?)
$S185   dd      3 dup(?)
$S217   dd      1 dup(?)
$S207   dd      1 dup(?)
$S201   dd      1 dup(?)
$S209   dd      1 dup(?)
$S208   dd      1 dup(?)
$S211   dd      1 dup(?)
$S210   dd      1 dup(?)
$S204   dd      1 dup(?)
$S215   dd      1 dup(?)
$S202   dd      1 dup(?)
$S203   dd      3 dup(?)
$S168   dd      1 dup(?)
$S170   dd      1 dup(?)
$S166   dd      1 dup(?)
$S171   dd      1 dup(?)
$S216   dd      1 dup(?)
$S193   dd      1 dup(?)
$S191   dd      1 dup(?)
$S194   dd      1 dup(?)
$S192   dd      1 dup(?)
$S199   dd      2 dup(?)
$S186   dd      1 dup(?)
$S187   dd      1 dup(?)
$S189   dd      1 dup(?)
$S188   dd      2 dup(?)
$S60    dd      1 dup(?)
$S61    dd      1 dup(?)
$S206   dd      1 dup(?)
$S200   dd      1 dup(?)
$S169   dd      1 dup(?)
$S172   dd      1 dup(?)
$S182   dd      1 dup(?)
$S178   dd      1 dup(?)
$S179   dd      1 dup(?)
$S180   dd      1 dup(?)
$S181   dd      1 dup(?)
$S175   dd      1 dup(?)
$S176   dd      1 dup(?)
$S177   dd      1 dup(?)
$S21    dd      1 dup(?)
$S19    dd      1 dup(?)
$S17    dd      1 dup(?)
$S18    dd      1 dup(?)
$S20    dd      1 dup(?)
$S12    dd      1 dup(?)
$S15    dd      28 dup(?) ;$S15    db      112 dup(?)
$S274   dd      1 dup(?)
$S6     dd      2 dup(?)
$S277   dd      1 dup(?)
$S2     dd      1 dup(?)

_BSS    ENDS
        END
