;---------------------------------------------------------------------
;  RC5 core using MMX instructions, meant for use by distributed.net
;  I hereby place this code into the public domain; do whatever the
;  heck you want with it, but if you violate ITAR and go to jail it's
;  not my fault. See readme.txt for details of what this mess means
;  and how to use it.
;
;                                               Jason Papadopoulos
;                                               jasonp@glue.umd.edu
;                                                   7/26/98
;---------------------------------------------------------------------
; $Id: jp-mmx.asm,v 1.1.2.1 2001/01/21 17:44:40 cyp Exp $

%include "jp-mmx.mac"

[GLOBAL _rc5_unit_func_p5_mmx]
[GLOBAL rc5_unit_func_p5_mmx]

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
[SECTION _DATA USE32 ALIGN=16]
%else
[SECTION .data]
%endif

   align 8
   InitTable  dd 0xbf0a8b1d, 0xbf0a8b1d
              dd 0x5618cb1c, 0x5618cb1c
              dd 0xf45044d5, 0xf45044d5
              dd 0x9287be8e, 0x9287be8e
              dd 0x30bf3847, 0x30bf3847
              dd 0xcef6b200, 0xcef6b200
              dd 0x6d2e2bb9, 0x6d2e2bb9
              dd 0x0b65a572, 0x0b65a572
              dd 0xa99d1f2b, 0xa99d1f2b
              dd 0x47d498e4, 0x47d498e4
              dd 0xe60c129d, 0xe60c129d
              dd 0x84438c56, 0x84438c56
              dd 0x227b060f, 0x227b060f
              dd 0xc0b27fc8, 0xc0b27fc8
              dd 0x5ee9f981, 0x5ee9f981
              dd 0xfd21733a, 0xfd21733a
              dd 0x9b58ecf3, 0x9b58ecf3
              dd 0x399066ac, 0x399066ac
              dd 0xd7c7e065, 0xd7c7e065
              dd 0x75ff5a1e, 0x75ff5a1e
              dd 0x1436d3d7, 0x1436d3d7
              dd 0xb26e4d90, 0xb26e4d90
              dd 0x50a5c749, 0x50a5c749
              dd 0xeedd4102, 0xeedd4102
              dd 0x8d14babb, 0x8d14babb
              dd 0x2b4c3474, 0x2b4c3474

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
[SECTION _TEXT USE32 ALIGN=16]
%else
[SECTION .text]
%endif

_rc5_unit_func_p5_mmx:
rc5_unit_func_p5_mmx:
   push ebx
   push esi

   push edi
   push ebp

   sub esp, work_size
   mov esi, InitTable

; 32 bit register assigment
; eax - pointer to RC5UnitWork struct
; ebx - ciphertext low
; ecx - key incrementing
; edx - temporary
; esi - ciphertext high and which key matches
; edi - iterations
; ebp - Quadword aligned pointer to work area

   lea ebp, [esp+8]
   mov eax, [RC5UnitWork]

   and ebp, 0xfffffff8        ; ensure quadword alignment
   mov ecx, 52

; Best to copy all data onto the stack
; which we know will be quadword aligned

; Lost alot of clock cycles when this wasn't done
; (even for just ShiftMask)

; The "rep" could be unrolled and pairing used to get
; this done quicker but as it is outside the main loop
; it is not worth it yet.

   lea edi, [Table0]

   pushf
   cld
   rep
   movsd
   popf

   mov esi, 0x0000001f
   mov edi, [timeslice]       ; work.iterations = timeslice

   mov [ShiftMask], esi
   mov [ShiftMask+4], esi

   xor esi, esi


; Load parameters
   mov ebx, [RC5UnitWork_L0lo]          ; ebx = l0 = Llo1
   mov edx, [RC5UnitWork_L0hi]          ; edx = l1 = Lhi1

   mov [work_key_lo], ebx
   mov [work_key_hi], edx

; Save other parameters
   mov ebx, [RC5UnitWork_plainlo]
   mov edx, [RC5UnitWork_plainhi]

   mov [work_P_0], ebx
   mov [work_P_1], edx

   mov [work_P_0+4], ebx
   mov [work_P_1+4], edx

   mov ebx, [RC5UnitWork_cipherlo]
   mov [AllZeros], esi

   mov [AllZeros+4], esi
   mov esi, [RC5UnitWork_cipherhi]

align 8
change_lo:

; Need to add first round of encryption into here eventually

   mov edx, [work_key_lo]

   mov [work_key1_lo], edx
   mov [work_key2_lo], edx

   mov [work_key3_lo], edx

change_hi:
   mov ecx, [work_key_hi]

mainloop:
   lea edx, [ecx+0x01000000]
   add ecx, 0x02000000

   mov [work_key1_hi], edx
   mov [work_key2_hi], ecx
                          ;             KEY SETUP STARTS HERE
                          ;   0     1     2     3     4     5     6     7
                          ;-----------------------------------------------
   movq mm0, [Table0]     ; A1:A0
   pxor mm5, mm5          ;                         A3:A2 B3:B2

   add edx, 0x02000000
   movq mm1, mm0          ; A1:A0 +1:+0

   mov [work_key3_hi], edx
   add ecx, 0x02000000

   movq [Table], mm0      ; A1:A0 +1:+0
   movq mm4, mm0          ;                         A3:A2

   paddd mm1, [Key0]      ; A1:A0 x1:x0 +1:+0
   movq mm2, mm0          ; A1:A0 +1:+0 +1:+0

   pand mm2, [ShiftMask]  ; A1:A0 x1:x0 s1:s0
   movq mm3, mm1          ; A1:A0 x1:x0 s1:s0 x1:x0

   movq [Table+8], mm4    ;                         A3:A2 B3:B2
   punpckldq mm1, mm1     ; A1:A0 x0:x0 s1:s0 x1:x0

   keytable_round_1  Table0+8  , Table+16 , Key0   , Key   , Key0+16
   keytable_round_1  Table0+16 , Table+32 , Key0+16, Key+16, Key
   keytable_round_1  Table0+24 , Table+48 , Key    , Key   , Key+16
   keytable_round_1  Table0+32 , Table+64 , Key+16 , Key+16, Key
   keytable_round_1  Table0+40 , Table+80 , Key    , Key   , Key+16
   keytable_round_1  Table0+48 , Table+96 , Key+16 , Key+16, Key
   keytable_round_1  Table0+56 , Table+112, Key    , Key   , Key+16
   keytable_round_1  Table0+64 , Table+128, Key+16 , Key+16, Key
   keytable_round_1  Table0+72 , Table+144, Key    , Key   , Key+16
   keytable_round_1  Table0+80 , Table+160, Key+16 , Key+16, Key
   keytable_round_1  Table0+88 , Table+176, Key    , Key   , Key+16
   keytable_round_1  Table0+96 , Table+192, Key+16 , Key+16, Key
   keytable_round_1  Table0+104, Table+208, Key    , Key   , Key+16
   keytable_round_1  Table0+112, Table+224, Key+16 , Key+16, Key
   keytable_round_1  Table0+120, Table+240, Key    , Key   , Key+16
   keytable_round_1  Table0+128, Table+256, Key+16 , Key+16, Key
   keytable_round_1  Table0+136, Table+272, Key    , Key   , Key+16
   keytable_round_1  Table0+144, Table+288, Key+16 , Key+16, Key
   keytable_round_1  Table0+152, Table+304, Key    , Key   , Key+16
   keytable_round_1  Table0+160, Table+320, Key+16 , Key+16, Key
   keytable_round_1  Table0+168, Table+336, Key    , Key   , Key+16
   keytable_round_1  Table0+176, Table+352, Key+16 , Key+16, Key
   keytable_round_1  Table0+184, Table+368, Key    , Key   , Key+16
   keytable_round_1  Table0+192, Table+384, Key+16 , Key+16, Key
   keytable_round_1  Table0+200, Table+400, Key    , Key   , Key+16

   keytable_round  Table     , Table    , Key+16 , Key+16, Key
   keytable_round  Table+16  , Table+16 , Key    , Key   , Key+16
   keytable_round  Table+32  , Table+32 , Key+16 , Key+16, Key
   keytable_round  Table+48  , Table+48 , Key    , Key   , Key+16
   keytable_round  Table+64  , Table+64 , Key+16 , Key+16, Key
   keytable_round  Table+80  , Table+80 , Key    , Key   , Key+16
   keytable_round  Table+96  , Table+96 , Key+16 , Key+16, Key
   keytable_round  Table+112 , Table+112, Key    , Key   , Key+16
   keytable_round  Table+128 , Table+128, Key+16 , Key+16, Key
   keytable_round  Table+144 , Table+144, Key    , Key   , Key+16
   keytable_round  Table+160 , Table+160, Key+16 , Key+16, Key
   keytable_round  Table+176 , Table+176, Key    , Key   , Key+16
   keytable_round  Table+192 , Table+192, Key+16 , Key+16, Key
   keytable_round  Table+208 , Table+208, Key    , Key   , Key+16
   keytable_round  Table+224 , Table+224, Key+16 , Key+16, Key
   keytable_round  Table+240 , Table+240, Key    , Key   , Key+16
   keytable_round  Table+256 , Table+256, Key+16 , Key+16, Key
   keytable_round  Table+272 , Table+272, Key    , Key   , Key+16
   keytable_round  Table+288 , Table+288, Key+16 , Key+16, Key
   keytable_round  Table+304 , Table+304, Key    , Key   , Key+16
   keytable_round  Table+320 , Table+320, Key+16 , Key+16, Key
   keytable_round  Table+336 , Table+336, Key    , Key   , Key+16
   keytable_round  Table+352 , Table+352, Key+16 , Key+16, Key
   keytable_round  Table+368 , Table+368, Key    , Key   , Key+16
   keytable_round  Table+384 , Table+384, Key+16 , Key+16, Key
   keytable_round  Table+400 , Table+400, Key    , Key   , Key+16

   keytable_round  Table     , Table    , Key+16 , Key+16, Key
   keytable_round  Table+16  , Table+16 , Key    , Key   , Key+16
   keytable_round  Table+32  , Table+32 , Key+16 , Key+16, Key
   keytable_round  Table+48  , Table+48 , Key    , Key   , Key+16
   keytable_round  Table+64  , Table+64 , Key+16 , Key+16, Key
   keytable_round  Table+80  , Table+80 , Key    , Key   , Key+16
   keytable_round  Table+96  , Table+96 , Key+16 , Key+16, Key
   keytable_round  Table+112 , Table+112, Key    , Key   , Key+16
   keytable_round  Table+128 , Table+128, Key+16 , Key+16, Key
   keytable_round  Table+144 , Table+144, Key    , Key   , Key+16
   keytable_round  Table+160 , Table+160, Key+16 , Key+16, Key
   keytable_round  Table+176 , Table+176, Key    , Key   , Key+16
   keytable_round  Table+192 , Table+192, Key+16 , Key+16, Key
   keytable_round  Table+208 , Table+208, Key    , Key   , Key+16
   keytable_round  Table+224 , Table+224, Key+16 , Key+16, Key
   keytable_round  Table+240 , Table+240, Key    , Key   , Key+16
   keytable_round  Table+256 , Table+256, Key+16 , Key+16, Key
   keytable_round  Table+272 , Table+272, Key    , Key   , Key+16
   keytable_round  Table+288 , Table+288, Key+16 , Key+16, Key
   keytable_round  Table+304 , Table+304, Key    , Key   , Key+16
   keytable_round  Table+320 , Table+320, Key+16 , Key+16, Key
   keytable_round  Table+336 , Table+336, Key    , Key   , Key+16
   keytable_round  Table+352 , Table+352, Key+16 , Key+16, Key
   keytable_round  Table+368 , Table+368, Key    , Key   , Key+16
   keytable_round  Table+384 , Table+384, Key+16 , Key+16, Key

                          ;   0     1     2     3     4     5     6     7
                          ;-----------------------------------------------
                          ; A1:A0 x0:x0 s1:s0 x1:x0 A3:A2 B3:B2

   paddd mm5, mm4         ;                         A3:A2 +3:+2
   punpckhdq mm3, mm3     ; A1:A0 x0:x0 s1:s0 x1:x1
   movd [AllZeros], mm2   ; A1:A0 x0:x0 s1:s0 x1:x0
   psrlq mm2, 32          ; A1:A0 x0:x0 00:s1 x1:x0
   psllq mm1, [AllZeros]  ; A1:A0 r0:?? 00:s1 x1:x1
   movq mm6, mm5          ;                         A3:A2 +3:+2 +3:+2
   paddd mm5, [Key+8]     ;                         A3:A2 x3:x2 +3:+2
   psllq mm3, mm2         ; A1:A0 r0:?? 00:s1 r1:??
   pand mm6, [ShiftMask]  ;                         A3:A2 x3:x2 s3:s2
   punpckhdq mm1, mm3     ; A1:A0 B1:B0
   movq mm7, mm5          ;                         A3:A2 x3:x2 +3:+2 x3:x2
   punpckldq mm5, mm5     ;                         A3:A2 x2:x2 s3:s2 x3:x2
   movd [AllZeros], mm6   ;                         A3:A2 x3:x2 s3:s2 x3:x2
   psrlq mm6, 32          ;                         A3:A2 x3:x2 00:s3 x3:x2
   paddd mm0, [Table+400] ; +1:+0 B1:B0
   punpckhdq mm7, mm7     ;                         A3:A2 x3:x2 00:s3 x3:x3
   psllq mm7, mm6         ;                         A3:A2 x3:x2 00:s3 r3:??
   paddd mm0, mm1         ; x1:x0 B1:B0
   psllq mm5, [AllZeros]  ;                         A3:A2 r2:?? 00:s3 r3:??
   movq mm2, mm0          ; x1:x0 B1:B0 x1:x0
   punpckhdq mm5, mm7     ;                         A3:A2 B3:B2
   pslld mm0, 3           ; f1:f0 B1:B0 x1:x0
   paddd mm4, mm5         ;                         +3:+2 B3:B2
   psrld mm2, 29          ; f1:f0 B1:B0 b1:b0
   paddd mm4, [Table+408] ;                         x3:x2 B3:B2
   por mm0, mm2           ; A1:A0 B1:B0
   movq mm6, mm4          ;                         x3:x2 B3:B2 x3:x2
   pslld mm4, 3           ;                         f3:f2 B3:B2 x3:x2
   movq [Table+400], mm0  ; A1:A0 B1:B0
   psrld mm6, 29          ;                         f3:f2 B3:B2 b3:b2

   movq mm0, [work_P_0]   ; a0:a0
   por mm4, mm6           ;                         A3:A2 B3:B2
   movq mm1, [work_P_1]   ; a0:a0 b0:b0

   movq [Table+408], mm4  ;                         A3:A2 B3:B2   stall
   movq mm5, mm1          ;                               b0:b0

                          ;---------------------ENCRYPTION STARTS HERE
   paddd mm1, [Table+16]  ; A1:A0 B1:B0
   movq mm4, mm0          ;                         a0:a0 b0:b0
   paddd mm0, [Table]     ; A1:A0 b0:b0
   movq mm2, mm1          ; x1:x0 B1:B0 B1:B0
   paddd mm4, [Table+8]   ;                         A3:A2 b0:b0
   pxor mm0, mm1          ; x1:x0 B1:B0
   pand mm2,[ShiftMask]   ; x1:x0 B1:B0 s1:s0
   movq mm3, mm0          ; x1:x0 B1:B0 s1:s0 x1:x0
   paddd mm5, [Table+24]  ;                         A3:A2 B3:B2
   punpckldq mm0, mm0     ; x0:x0 B1:B0 s1:s0 x1:x0
   movd [AllZeros], mm2   ; x0:x0 B1:B0 s1:s0 x1:x0
   psrlq mm2, 32          ; x0:x0 B1:B0 00:s1 x1:x0
   psllq mm0,[AllZeros]   ; r0:?? B1:B0 00:s1 x1:x0
   pxor mm4, mm5          ;                         x3:x2 B3:B2
   punpckhdq mm3, mm3     ; r0:?? B1:B0 00:s1 x1:x1
   movq mm6, mm5          ;                         x3:x2 B3:B2 B3:B2
   psllq mm3, mm2         ; r0:?? B1:B0 00:s1 r1:??
   movq mm7, mm4          ;                         x3:x2 B3:B2 B3:B2 x3:x2

   encryption_round  0, 1, 2, 3, 4, 5, 6, 7,  Table+32
   encryption_round  1, 0, 2, 3, 5, 4, 6, 7,  Table+48
   encryption_round  0, 1, 2, 3, 4, 5, 6, 7,  Table+64
   encryption_round  1, 0, 2, 3, 5, 4, 6, 7,  Table+80
   encryption_round  0, 1, 2, 3, 4, 5, 6, 7,  Table+96
   encryption_round  1, 0, 2, 3, 5, 4, 6, 7,  Table+112
   encryption_round  0, 1, 2, 3, 4, 5, 6, 7,  Table+128
   encryption_round  1, 0, 2, 3, 5, 4, 6, 7,  Table+144
   encryption_round  0, 1, 2, 3, 4, 5, 6, 7,  Table+160
   encryption_round  1, 0, 2, 3, 5, 4, 6, 7,  Table+176
   encryption_round  0, 1, 2, 3, 4, 5, 6, 7,  Table+192
   encryption_round  1, 0, 2, 3, 5, 4, 6, 7,  Table+208
   encryption_round  0, 1, 2, 3, 4, 5, 6, 7,  Table+224
   encryption_round  1, 0, 2, 3, 5, 4, 6, 7,  Table+240
   encryption_round  0, 1, 2, 3, 4, 5, 6, 7,  Table+256
   encryption_round  1, 0, 2, 3, 5, 4, 6, 7,  Table+272
   encryption_round  0, 1, 2, 3, 4, 5, 6, 7,  Table+288
   encryption_round  1, 0, 2, 3, 5, 4, 6, 7,  Table+304
   encryption_round  0, 1, 2, 3, 4, 5, 6, 7,  Table+320
   encryption_round  1, 0, 2, 3, 5, 4, 6, 7,  Table+336
   encryption_round  0, 1, 2, 3, 4, 5, 6, 7,  Table+352
   encryption_round  1, 0, 2, 3, 5, 4, 6, 7,  Table+368
   encryption_round  0, 1, 2, 3, 4, 5, 6, 7,  Table+384

                          ;   0     1     2     3     4     5     6     7
                          ;-----------------------------------------------
                          ; A1:A0 r0:??       r1:?? A3:A2 x3:x2 A3:A2 x3:x2

   pand mm6, [ShiftMask]  ;                   r1:?? A3:A2 x3:x2 s3:s2 x3:x2
   punpckhdq mm1, mm3     ; A1:A0 r1:r0
   paddd mm1, [Table+400] ; A1:A0 B1:B0
   punpckldq mm5, mm5     ;                         A3:A2 x2:x2 s3:s2 x3:x2


; These checks need to be moved higher up
; basically eliminating the final encryption round

   movq [work_C_0], mm0   ; A1:A0 B1:B0             A3:A2 r2:?? 00:s3 x3:x2
   punpckhdq mm7, mm7     ;                         A3:A2 r2:?? 00:s3 x3:x3

   cmp  [work_C_0], ebx   ; Check A0
   jz near check_B0

check_A1:
   cmp  [work_C_0+4], ebx ; Check A1
   jz near check_B1

check_A2:
   movd [AllZeros], mm6   ;                         A3:A2 x2:x2 s3:s2 x3:x3
   psrlq mm6, 32          ;                         A3:A2 x2:x2 00:s3 x3:x3
   psllq mm5, [AllZeros]  ;                         A3:A2 r2:?? 00:s3 x3:x3
   psllq mm7, mm6         ;                         A3:A2 r2:?? 00:s3 r3:??
   movq [work_C_0], mm4
   punpckhdq mm5, mm7     ;                         A3:A2 r3:r2

   paddd mm5, [Table+408] ;                         A3:A2 B3:B2

   cmp  [work_C_0], ebx   ; Check A2
   jz near check_B2

check_A3:
   cmp  [work_C_0+4], ebx ; Check A3
   jz near check_B3

incr_key:
   dec edi
   jz near full_exit

   mov [work_key_hi], ecx

   test ecx, 0xff000000
   jnz near mainloop

   inc byte [work_key_hi2]
   jnz near change_hi
   inc byte [work_key_hi1]
   jnz near change_hi
   inc byte [work_key_hi]
   jnz near change_hi

   inc byte [work_key_lo3]
   jnz near change_lo
   inc byte [work_key_lo2]
   jnz near change_lo
   inc byte [work_key_lo1]
   jnz near change_lo
   inc byte [work_key_lo]
   jmp change_lo


check_B0:
   movq [work_C_1], mm1   ; A1:A0 B1:B0

   cmp  [work_C_1], esi   ; Check B0
   jnz near check_A1

   xor  esi, esi
   jmp  finish


check_B1:
   movq [work_C_1], mm1   ; A1:A0 B1:B0

   cmp  [work_C_1+4], esi ; Check B1
   jnz near check_A2

   mov  esi, 1
   jmp  finish


check_B2:
   movq [work_C_1], mm5   ;                         A3:A2 B3:B2

   cmp  [work_C_1], esi   ; Check B2
   jnz near check_A3

   mov  esi, 2
   jmp  finish

check_B3:
   movq [work_C_1], mm5   ;                         A3:A2 B3:B2

   cmp  [work_C_1+4], esi ; Check B3
   jnz near incr_key

   mov  esi, 3
   jmp  finish


full_exit:
   mov [work_key_hi], ecx
   xor esi, esi

   test ecx, 0xff000000
   jnz key_updated

   inc byte [work_key_hi2]
   jnz key_updated
   inc byte [work_key_hi1]
   jnz key_updated
   inc byte [work_key_hi]
   jnz key_updated
   inc byte [work_key_lo3]
   jnz key_updated
   inc byte [work_key_lo2]
   jnz key_updated
   inc byte [work_key_lo1]
   jnz key_updated
   inc byte [work_key_lo]

key_updated:
   mov ecx, [work_key_hi]
   mov edx, [work_key_lo]

   mov [RC5UnitWork_L0hi], ecx
   mov [RC5UnitWork_L0lo], edx


finish:
   emms

   mov eax, [timeslice]
   add esp, work_size

   sub eax, edi
   pop ebp

   shl eax, 2
   pop edi

   add eax, esi
   pop esi

   pop ebx
   ret


