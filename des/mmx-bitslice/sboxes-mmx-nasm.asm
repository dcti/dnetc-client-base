;
; $Log: sboxes-mmx-nasm.asm,v $
; Revision 1.2  1999/01/28 00:38:20  trevorh
; OS/2 %if added
;
; Revision 1.1  1998/11/16 15:35:34  remi
; Converted sboxes-mmx.cpp to NASM format.
;
;

%ifdef OS2
[SECTION _DATA USE32 align=16]
%else
[SECTION .data]
%endif
idtag:    db "@(#)$Id: sboxes-mmx-nasm.asm,v 1.2 1999/01/28 00:38:20 trevorh Exp $\0"

%ifdef OS2
[SECTION _TEXT USE32 align=16]
%else
[SECTION .text]
%endif

[GLOBAL _sboxes_mmx_asm]
[GLOBAL sboxes_mmx_asm]
_sboxes_mmx_asm:
sboxes_mmx_asm:
mov eax,idtag
ret

[GLOBAL _mmxs1_kwan]
[GLOBAL _mmxs2_kwan]
[GLOBAL _mmxs3_kwan]
[GLOBAL _mmxs4_kwan]
[GLOBAL _mmxs5_kwan]
[GLOBAL _mmxs6_kwan]
[GLOBAL _mmxs7_kwan]
[GLOBAL _mmxs8_kwan]

[GLOBAL mmxs1_kwan]
[GLOBAL mmxs2_kwan]
[GLOBAL mmxs3_kwan]
[GLOBAL mmxs4_kwan]
[GLOBAL mmxs5_kwan]
[GLOBAL mmxs6_kwan]
[GLOBAL mmxs7_kwan]
[GLOBAL mmxs8_kwan]


; ==============================================
align 4
_mmxs1_kwan:
mmxs1_kwan:

movq [ 40+eax],mm0
movq mm6,mm3	; copy a4

pxor mm0,[ 0+eax]	; x2 = ~a1
pxor mm3,mm2	; x3 = a3 ^ a4

pxor mm6,[ 0+eax]	; x1 = ~a4
movq mm7,mm0	; copy x2

movq [ 56+eax],mm4
por mm7,mm2	; x5 = a3 | x2

movq [ 72+eax],mm3
movq mm4,mm5	; copy a6

movq [ 64+eax],mm6
pxor mm3,mm0	; x4 = x2 ^ x3

movq [ 88+eax],mm7
por mm0,mm6	; x9 = x1 | x2

movq [ 48+eax],mm2
pand mm7,mm6	; x6 = x1 & x5

movq [ 80+eax],mm3
por mm2,mm3	; x23 = a3 | x4

pxor mm2,[ 0+eax]	; x24 = ~x23
pand mm4,mm0	; x10 = a6 & x9

movq mm6,mm7	; copy x6
por mm2,mm5	; x25 = a6 | x24

movq [ 96+eax],mm7
por mm6,mm5	; x7 = a6 | x6

pxor mm7,mm2	; x26 = x6 ^ x25
pxor mm3,mm6	; x8 = x4 ^ x7

movq [ 120+eax],mm2
pxor mm6,mm4	; x11 = x7 ^ x10

pand mm4,[ 48+eax]	; x38 = a3 & x10
movq mm2,mm6	; copy x11

pxor mm6,[ 48+eax]	; x53 = a3 ^ x11
por mm2,mm1	; x12 = a2 | x11

pand mm6,[ 88+eax]	; x54 = x5 & x53
pxor mm2,mm3	; x13 = x8 ^ x12

movq [ 136+eax],mm4
pxor mm0,mm2	; x14 = x9 ^ x13

movq [ 128+eax],mm7
movq mm4,mm5	; copy a6

movq [ 104+eax],mm2
por mm4,mm0	; x15 = a6 | x14

movq mm7,[ 64+eax]
por mm6,mm1	; x55 = a2 | x54

movq [ 112+eax],mm0
movq mm2,mm3	; copy x8

pandn mm0,[ 72+eax]	; x18 = x3 & ~x14
pxor mm4,mm7	; x16 = x1 ^ x15

por mm5,[ 80+eax]	; x57 = a6 | x4
por mm0,mm1	; x19 = a2 | x18

pxor mm5,[ 136+eax]	; x58 = x38 ^ x57
pxor mm4,mm0	; x20 = x16 ^ x19

movq mm0,[ 56+eax]
pand mm2,mm7	; x27 = x1 & x8

movq [ 144+eax],mm6
por mm2,mm1	; x28 = a2 | x27

movq mm6,[ 112+eax]
por mm0,mm4	; x21 = a5 | x20

pand mm6,[ 88+eax]	; x32 = x5 & x14
por mm7,mm3	; x30 = x1 | x8

movq [ 152+eax],mm5
pxor mm6,mm3	; x33 = x8 ^ x32

pxor mm7,[ 96+eax]	; x31 = x6 ^ x30
movq mm5,mm1	; copy a2

pxor mm2,[ 128+eax]	; x29 = x26 ^ x28
pand mm5,mm6	; x34 = a2 & x33

pand mm6,[ 48+eax]	; x40 = a3 & x33
pxor mm5,mm7	; x35 = x31 ^ x34

por mm5,[ 56+eax]	; x36 = a5 | x35

movq mm7,[ 40+eax]
pxor mm5,mm2	; x37 = x29 ^ x36

movq mm2,[ 80+eax]
por mm7,mm3	; x46 = a1 | x8

por mm2,[ 136+eax]	; x39 = x4 | x38
pxor mm3,mm6	; x52 = x8 ^ x40

pxor mm6,[ 120+eax]	; x41 = x25 ^ x40
pxor mm7,mm4	; x47 = x46 ^ x20

movq mm4,[ 48+eax]
por mm7,mm1	; x48 = a2 | x47

por mm4,[ 128+eax]	; x44 = a3 | x26
por mm6,mm1	; x42 = a2 | x41

pxor mm4,[ 112+eax]	; x45 = x14 ^ x44
pxor mm6,mm2	; x43 = x39 ^ x42

movq mm2,[ 104+eax]
pxor mm7,mm4	; x49 = x45 ^ x48

pxor mm3,[ 144+eax]	; x56 = x52 ^ x55
pxor mm0,mm2	; x22 = x13 ^ x21

pxor mm5,[ 8+eax]	; out1 ^= x37
pand mm2,mm3	; x59 = x13 & x56

movq mm4,[ 56+eax]
pand mm2,mm1	; x60 = a2 & x59

pxor mm2,[ 152+eax]	; x61 = x58 ^ x60
pand mm7,mm4	; x50 = a5 & x49

pxor mm0,[ 32+eax]	; out4 ^= x22
pand mm2,mm4	; x62 = a5 & x61

pxor mm7,mm6	; x51 = x43 ^ x50
pxor mm2,mm3	; x63 = x56 ^ x62

pxor mm7,[ 16+eax]	; out2 ^= x51

pxor mm2,[ 24+eax]	; out3 ^= x63

 	; 51 clocks for 67 variables


ret



; ==============================================
align 4
_mmxs2_kwan:
mmxs2_kwan:

movq [ 64+eax],mm3
movq mm6,mm4	; copy a5

movq [ 40+eax],mm0
movq mm7,mm4	; copy a5

pxor mm0,[ 0+eax]	; x2 = ~a1
pxor mm6,mm5	; x3 = a5 ^ a6

pxor mm7,[ 0+eax]	; x1 = ~a5
movq mm3,mm0	; copy x2

movq [ 56+eax],mm2
por mm7,mm5	; x6 = a6 | x1

movq [ 72+eax],mm6
por mm3,mm7	; x7 = x2 | x6

pxor mm7,mm4	; x13 = a5 ^ x6
pxor mm6,mm0	; x4 = x2 ^ x3

pand mm3,mm1	; x8 = a2 & x7
por mm2,mm7	; x14 = a3 | x13

movq [ 48+eax],mm1
pxor mm3,mm5	; x9 = a6 ^ x8

movq [ 80+eax],mm6
pxor mm6,mm1	; x5 = a2 ^ x4

movq [ 96+eax],mm7
pand mm1,mm3	; x12 = a2 & x9

pand mm3,[ 56+eax]	; x10 = a3 & x9
pxor mm1,mm2	; x15 = x12 ^ x14

movq mm7,[ 80+eax]
movq mm2,mm1	; copy x15

pand mm2,[ 64+eax]	; x16 = a4 & x15
pxor mm3,mm6	; x11 = x5 ^ x10

movq [ 88+eax],mm6
pxor mm3,mm2	; x17 = x11 ^ x16

movq mm2,[ 40+eax]
por mm7,mm5	; x22 = a6 | x4

por mm1,mm2	; x40 = a1 | x15
pand mm7,mm3	; x23 = x17 & x22

pxor mm3,[ 16+eax]	; out2 ^= x17
por mm2,mm4	; x18 = a1 | a5

por mm7,[ 56+eax]	; x24 = a3 | x23
movq mm6,mm2	; copy x18

pxor mm1,[ 96+eax]	; x41 = x13 ^ x40
por mm6,mm5	; x19 = a6 | x18

movq [ 16+eax],mm3
pand mm4,mm0	; x27 = a5 & x2

movq mm3,[ 96+eax]
por mm5,mm0	; x26 = a6 | x2

movq [ 104+eax],mm2
pxor mm3,mm6	; x20 = x13 ^ x19

movq mm2,[ 48+eax]
pxor mm0,mm6	; x31 = x2 ^ x19

pxor mm3,mm2	; x21 = a2 ^ x20
pand mm0,mm2	; x32 = a2 & x31

pxor mm7,mm3	; x25 = x21 ^ x24
por mm2,mm4	; x28 = a2 | x27

pxor mm4,[ 72+eax]	; x30 = x3 ^ x27
pand mm6,mm3	; x47 = x19 & x21

pxor mm4,mm0	; x33 = x30 ^ x32
pxor mm6,mm5	; x48 = x26 ^ x47

movq [ 112+eax],mm7
pand mm0,mm3	; x38 = x21 & x32

movq mm7,[ 56+eax]
pxor mm5,mm2	; x29 = x26 ^ x28

pxor mm0,[ 88+eax]	; x39 = x5 ^ x38
pand mm7,mm4	; x34 = a3 & x33

pand mm4,[ 48+eax]	; x49 = a2 & x33
pxor mm7,mm5	; x35 = x29 ^ x34

por mm7,[ 64+eax]	; x36 = a4 | x35
movq mm5,mm1	; copy x41

por mm5,[ 56+eax]	; x42 = a3 | x41
por mm1,mm2	; x44 = x28 | x41

pand mm2,[ 104+eax]	; x53 = x18 & x28
pxor mm4,mm3	; x50 = x21 ^ x49

movq mm3,[ 64+eax]
pand mm2,mm4	; x54 = x50 & x53

pand mm4,[ 56+eax]	; x51 = a3 & x50
pxor mm0,mm5	; x43 = x39 ^ x42

pxor mm7,[ 112+eax]	; x37 = x25 ^ x36
pxor mm4,mm6	; x52 = x48 ^ x51

pxor mm7,[ 24+eax]	; out3 ^= x37
pand mm1,mm3	; x45 = a4 & x44

movq mm5,[ 16+eax]
pxor mm1,mm0	; x46 = x43 ^ x45

pxor mm1,[ 8+eax]	; out1 ^= x46
por mm2,mm3	; x55 = a4 | x54

pxor mm2,mm4	; x56 = x52 ^ x55

pxor mm2,[ 32+eax]	; out4 ^= x56

 	; 44 clocks for 60 variables


ret


; ==============================================
align 4
_mmxs3_kwan:
mmxs3_kwan:


 	; mm6 free
 	; mm7 free
movq [ 40+eax],mm0	;  # mm0 free
movq mm6,mm5	; mm6 = a6
pxor mm6,[ 0+eax]	; # mm6(x2) = ~a6
movq mm7,mm4	; mm7 = a5
pxor mm7,mm6	; # mm7(x9) = a5 ^ x2
movq mm0,mm4	; mm0 = a5
movq [ 48+eax],mm6	;  # mm6 free
pand mm0,mm2	; mm0(x3) = a5 & a3
movq [ 56+eax],mm7	;  # mm7 free
pxor mm0,mm5	; mm0(x4) = x3 ^ a6
movq [ 64+eax],mm4	;  # mm4 free
pandn mm4,mm3	; mm4(x5) = a4 & ~a5
movq [ 72+eax],mm0	;  # mm0 free
por mm7,mm3	; mm7(x10) = a4 | x9
movq mm6,[ 64+eax]	; # mm6 = a5
pxor mm0,mm4	; mm0(x6) = x4 ^ x5
movq [ 80+eax],mm5	;  # mm5 free
pandn mm6,mm2	; mm6(x8) = a3 & ~a5
movq [ 88+eax],mm0	;  # mm0 free
pxor mm7,mm6	; mm7(x11) = x8 ^ x10
movq mm5,[ 48+eax]	; # mm5 = x2
pxor mm0,mm1	; mm0(x7) = x6 ^ a2
movq [ 96+eax],mm4	;  # mm4 free
movq mm4,mm7	; mm4 = x11
por mm5,[ 72+eax]	; # mm5(x23) = x1 | x4
pand mm4,mm0	; mm4(x12) = x7 & x11
movq [ 104+eax],mm7	;  # mm7 free
pxor mm6,mm5	; mm6(x24) = x23 ^ x8
pxor mm7,[ 64+eax]	; # mm7(x13) = a5 ^ x11
por mm6,mm1	; mm6(x25) = a2 | x24
movq [ 112+eax],mm4	;  # mm4 free
pand mm4,mm5	; mm4(x54) = x12 & x23
movq [ 120+eax],mm7	;  # mm7 free
por mm7,mm0	; mm7(x14) = x13 | x7
movq [ 128+eax],mm4	;  # mm4 free
movq mm4,mm2	; mm4 = a3
pxor mm4,[ 56+eax]	; # mm4 = a3 ^ x21
pand mm7,mm3	; mm7(x15) = a4 & x14
movq [ 136+eax],mm0	;  # mm0 free
pxor mm4,mm3	; mm4(x22) = a4 ^ a3 ^ x9
pxor mm5,[ 80+eax]	; # mm5(x27) = a6 ^ x23
pxor mm6,mm4	; mm6(x26) = x22 ^ x25
 	; mm4 free
movq [ 144+eax],mm3	;  # mm3 free
por mm3,mm5	; mm3(x28) = x27 | a4
movq [ 64+eax],mm2	;  # mm2 free
pxor mm5,mm3	; mm5(x51) = x27 ^ x28
por mm5,mm1	; # mm5(x52) = x51 | a2
pxor mm2,mm7	; mm2(x29) = a3 ^ x15
pxor mm7,[ 112+eax]	; # mm7(x16) = x12 ^ x15
movq mm4,mm2	; mm4 = x29
por mm2,[ 96+eax]	; # mm2(x30) = x29 | x5
pand mm7,mm1	; mm7(x17) = a2 & x16
por mm4,[ 72+eax]	; # mm4(x37) = x29 | x4
por mm2,mm1	; mm2(x31) = a2 | x30
pxor mm7,[ 104+eax]	; # mm7(x18) = x17 ^ x11
pxor mm2,mm3	; mm2(x32) = x31 ^ x28
 	; mm3 free
movq mm3,[ 40+eax]	; # mm1 = a3
pxor mm4,[ 144+eax]	; # mm4(x38) = x37 ^ a4
pand mm7,mm3	; mm7(x19) = x18 & a1
pxor mm7,[ 136+eax]	; # mm7(x20) = x19 ^ x7
por mm2,mm3	; mm2(x33) = a1 | x32
movq [ 72+eax],mm4	;  # mm4 free
pxor mm2,mm6	; mm2(x34) = x26 ^ x33
 	; mm6 free
pxor mm7,[ 32+eax]	; ### mm7(out4) = out4 ^ x20
por mm4,mm1	; mm4(x39) = a2 | x38
movq mm6,[ 64+eax]	; # mm6 = a3
movq mm3,mm2	; mm3 = x34
pxor mm6,[ 56+eax]	; # mm6(x35) = a3 ^ x9
por mm6,[ 96+eax]	; # mm6(x36) = x5 | x35
pxor mm3,[ 72+eax]	; # mm3(x43) = x34 ^ x38
pxor mm4,mm6	; mm4(x40) = x36 ^ x39
movq mm6,[ 80+eax]	; # mm6 = a6
pand mm6,[ 104+eax]	; # mm6(x41) = a6 & x11
movq mm0,[ 48+eax]	; # mm0 = x2
pxor mm3,mm6	; mm3(x44) = x43 ^ x41
por mm6,[ 88+eax]	; # mm6(x42) = x41 | x6
pand mm3,mm1	; mm3(x45) = x42 & a2
por mm0,[ 72+eax]	; # mm0(x49) = x2 | x38
pxor mm3,mm6	; mm3(x46) = x42 ^ x45
 	; mm6 free
pxor mm0,[ 120+eax]	; # mm0(x50) = x49 ^ x13
movq mm6,mm5	; mm6 = x52
por mm3,[ 40+eax]	; # mm3(x47) = x46 | a1
pxor mm0,mm5	; mm0(x53) = x50 ^ x52
pand mm6,[ 128+eax]	; # mm6(x55) = x52 & x54
pxor mm3,mm4	; mm3(x48) = x40 ^ x47
por mm6,[ 40+eax]	; # mm6(x56) = a1 | x55
pxor mm3,[ 24+eax]	; ### mm3(out3) = out3 ^ x48
pxor mm6,mm0	; mm6(x57) = x53 ^ x56
pxor mm2,[ 8+eax]	; ### mm2(out1) = out1 ^ x34
pxor mm6,[ 16+eax]	; ### mm6(out2) = out2 ^ x57


ret


; ==============================================
align 4
_mmxs4_kwan:
mmxs4_kwan:


movq [ 64+eax],mm5
movq mm6,mm2	; copy a3

movq [ 56+eax],mm3
movq mm7,mm0	; copy a1

movq [ 40+eax],mm1
por mm6,mm0	; x3 = a1 | a3

pand mm7,mm4	; x8 = a1 & a5
movq mm3,mm1	; copy a2

movq [ 48+eax],mm2
movq mm5,mm4	; copy a5

pand mm5,mm6	; x4 = a5 & x3
por mm3,mm2	; x6 = a2 | a3

pxor mm2,[ 0+eax]	; x2 = ~a3
pxor mm0,mm5	; ~x5 = a1 ^ x4

pxor mm0,[ 0+eax]	; x5 = ~(~x5)
pxor mm6,mm7	; x9 = x8 ^ x3

pxor mm3,mm0	; x7 = x5 ^ x6
movq mm7,mm1	; copy a2

pand mm7,mm6	; x10 = a2 & x9
pxor mm5,mm2	; x14 = x2 ^ x4

pxor mm2,mm4	; x18 = a5 ^ x2
pand mm0,mm5	; x17 = x5 & x14

pxor mm4,mm7	; x11 = a5 ^ x10
pand mm5,mm1	; x15 = a2 & x14

por mm2,mm1	; x19 = a2 | x18
pxor mm5,mm6	; x16 = x9 ^ x15

movq mm1,[ 56+eax]	; retrieve a4
movq mm6,mm0	; copy x17

pand mm1,mm4	; x12 = a4 & x11
pxor mm6,mm2	; x20 = x17 ^ x19

por mm6,[ 56+eax]	; x21 = a4 | x20
pxor mm1,mm3	; x13 = x7 ^ x12

pand mm4,[ 40+eax]	; x28 = a2 & x11
pxor mm6,mm5	; x22 = x16 ^ x21

movq mm3,[ 64+eax]	; retrieve a6
pxor mm4,mm0	; x29 = x28 ^ x17

pxor mm7,[ 48+eax]	; x30 = a3 ^ x10
movq mm0,mm3	; copy a6

pxor mm7,mm2	; x31 = x30 ^ x19
pand mm0,mm6	; x23 = a6 & x22

movq mm2,[ 56+eax]	; retrieve a4
por mm6,mm3	; x26 = a6 | x22

pxor mm0,mm1	; x24 = x13 ^ x23
pand mm7,mm2	; x32 = a4 & x31

pxor mm1,[ 0+eax]	; x25 = ~x13
pxor mm4,mm7	; x33 = x29 ^ x32

movq mm5,mm4	; copy x33
pxor mm4,mm1	; x34 = x25 ^ x33

pxor mm1,[ 8+eax]	; out1 ^= x25
por mm2,mm4	; x37 = a4 | x34

pand mm4,[ 40+eax]	; x35 = a2 & x34
pxor mm1,mm6	; out1 ^= x26

pxor mm4,mm0	; x36 = x24 ^ x35

pxor mm0,[ 16+eax]	; out2 ^= x24
pxor mm2,mm4	; x38 = x36 ^ x37

pand mm3,mm2	; x39 = a6 & x38
pxor mm6,mm2	; x41 = x26 ^ x38

pxor mm5,mm3	; x40 = x33 ^ x39

pxor mm6,mm5	; x42 = x41 ^ x40

pxor mm5,[ 32+eax]	; out4 ^= x40

pxor mm6,[ 24+eax]	; out3 ^= x42


ret


; ==============================================
align 4
_mmxs5_kwan:
mmxs5_kwan:


movq [ 48+eax],mm1	;  # mm1 free
movq mm6,mm3	; mm6 = a4
movq mm7,mm2	; # mm7 = a3
pandn mm6,mm2	; mm6(x1) = a3 & ~a4
pandn mm7,mm0	; # mm7(x3) = a1 & ~a3
movq mm1,mm6	; mm1 = x1
movq [ 40+eax],mm0	;  # mm0 free
pxor mm1,mm0	; mm1(x2) = x1 ^ a1
pxor mm0,mm3	; # mm0(x6) = a4 ^ a1
movq [ 64+eax],mm1	;  # mm1 free
por mm6,mm0	; mm6(x7) = x1 | x6
movq [ 56+eax],mm5	;  # mm5 free
por mm5,mm7	; mm5(x4) = a6 | x3
movq [ 96+eax],mm6	;  # mm6 free
pxor mm1,mm5	; mm1(x5) = x2 ^ x4
movq [ 72+eax],mm5	;  # %mm5 free
pand mm6,mm2	; mm6 = a3 & x7
movq mm5,[ 56+eax]	; # mm5 = a6
pxor mm6,mm3	; mm6(x13) = (a3 & x7) ^ a4
pandn mm5,[ 96+eax]	; # mm5(x8) = x7 & ~a6
movq [ 88+eax],mm0	;  # mm0 free
movq mm0,mm7	; mm0 = x3
movq [ 104+eax],mm5	;  # mm5 free
pxor mm5,mm2	; mm5(x9) = a3 ^ x8
movq [ 80+eax],mm1	;  # mm1 free
pxor mm0,mm3	; mm0 = x3 ^ a4
movq [ 112+eax],mm5	;  # mm5 free
pandn mm7,mm6	; mm7 = x13 & ~x3
por mm0,[ 56+eax]	; # mm0(x16) = a6 | (x3 ^ a4)
por mm5,mm4	; mm5 = a5 | x9
movq [ 120+eax],mm6	;  # mm6 free
pxor mm5,mm1	; mm5 = x5 ^ (a5 | x9)
movq [ 128+eax],mm0	;  # mm0 free
pxor mm7,mm0	; mm7(x17) = x16 ^ (x13 & ~x3)
movq mm0,[ 48+eax]	; # mm0 = a2
movq mm1,mm4	; mm1 = a5
movq [ 56+eax],mm7	;  # mm7 free
por mm1,mm7	; mm1 = a5 | x17
pand mm7,[ 80+eax]	; # mm7(x31) = x17 & x5
pxor mm1,mm6	; mm1(x19) = x13 ^ (a5 | x17)
pandn mm0,mm1	; # mm0 = x19 & ~a2
movq mm6,mm7	; mm6 = x31
pandn mm6,[ 96+eax]	; # mm6(x32) = x7 & ~x31
pxor mm5,mm0	; mm5(x21) = x5 ^ (a5 | x9) ^ (x19 & ~a2)
pxor mm7,[ 112+eax]	; # mm7(x38) = x9 ^ x32
movq mm0,mm3	; mm0 = a4
movq [ 96+eax],mm5	;  # mm5 free
movq mm5,mm6	; mm5 = x32
pandn mm0,[ 104+eax]	; # mm0 = x8 & ~a4
pandn mm5,mm1	; mm5(x43) = x19 & ~x32
 	; mm1 free
pxor mm6,[ 24+eax]	; # mm6 = out3 ^ x32
pxor mm0,mm2	; mm0(x34) = (x8 & ~a4) ^ a3
 	; mm2 free (no more references to a3)
movq mm2,[ 40+eax]	; # mm2 = a1  # 'a1' local var free
movq mm1,mm0	; mm1 = x34
pxor mm2,[ 112+eax]	; # mm2(x24) = a1 ^ x9
pand mm1,mm4	; mm1 = x34 & a5
movq [ 112+eax],mm7	;  # mm7 free
pxor mm6,mm1	; mm6 = out3 ^ x32 ^ (x34 & a5)
 	; mm1 free
movq mm1,[ 72+eax]	; # mm1 = x4
movq mm7,mm2	; mm7 = x24
pand mm7,[ 64+eax]	; # mm7 = x2 & x24
pand mm1,mm3	; mm1 = a4 & x4
pxor mm1,[ 56+eax]	; # mm1 = (a4 & x4) ^ x17
pandn mm7,mm4	; mm7 = a5 & ~(x2 & x24)
movq [ 104+eax],mm2	;  # mm2 free
pxor mm1,mm7	; mm1(x27) = (a4 & x4) ^ x17 ^ (x2 & x24)
 	; mm7 free
movq mm7,[ 16+eax]	; # mm7 = out2
por mm3,mm2	; mm3(x28) = a4 | x24
movq mm2,[ 48+eax]	; # mm2 = a2
pxor mm7,mm1	; mm7 = out2 ^ x27
movq [ 56+eax],mm3	;  # mm3 free
pandn mm2,mm3	; mm2 = x28 & ~a2
movq mm3,[ 112+eax]	; # mm3 = x38
pxor mm7,mm2	;## mm7(out2) = out2 ^ x27 ^ (x28 & ~a2)
 	; mm2 free
movq mm2,[ 128+eax]	; # mm2 = x16
por mm3,mm4	; mm3 = x38 | a5
por mm2,[ 120+eax]	; # mm2 = x13 | x16
por mm1,mm5	; mm1 = x27 | x43
pxor mm5,[ 8+eax]	; # mm5 = out1 ^ x43
pxor mm2,mm3	; mm2 = (x13 | x16) ^ (x38 | a5)
 	; mm3 free
por mm2,[ 48+eax]	; # mm2 = a2 | ((x13 | x16) ^ (x38 | a5))
pxor mm1,[ 88+eax]	; # mm1 = (x27 | x43) ^ x6
pxor mm6,mm2	; mm6 = out3 ^ x32 ^ (x34 & a5) ^ (a2 | ((x13 | x16) ^ (x38 | a5)))
 	; mm2 free
pxor mm6,[ 0+eax]	; ### mm6(out3) = out3 ^ x32 ^ (x34 & a5) ^ ~(a2 | ((x13 | x16) ^ (x38 | a5)))
pandn mm1,mm4	; mm1 = a5 & ~((x27 | x43) ^ x6)
movq mm2,[ 112+eax]	; # mm2 = x38
pxor mm1,[ 104+eax]	; # mm1 = x24 ^ (a5 & ~((x27 | x43) ^ x6))
movq mm3,mm2	; mm3 = x38
pxor mm2,[ 96+eax]	; # mm2 = x21 ^ x38
pxor mm5,mm1	; mm5 = out1 ^ x43 ^ x24 ^ (a5 & ~((x27 | x43) ^ x6))
pand mm3,[ 88+eax]	; # mm3 = x6 & x38
pandn mm2,mm4	; mm2 = a5 & ~(x21 ^ x38)
pand mm2,[ 56+eax]	; # mm2 = a5 & x28 & ~(x21 ^ x38)
pxor mm3,mm0	; mm3 = (x6 & x38) ^ x34
movq mm4,[ 96+eax]	; # mm4 = x21
pxor mm3,mm2	; mm3 = (x6 & x38) ^ x34 ^ (a5 & x28 & ~(x21 ^ x38))
por mm3,[ 48+eax]	; # mm3 = a2 | ((x6 & x38) ^ x34 ^ (a5 & x28 & ~(x21 ^ x38)))
pxor mm4,[ 32+eax]	; ### mm4(out4) = out4 ^ x21
pxor mm5,mm3	;## mm5(out1) = ...


ret


; ==============================================
align 4
_mmxs6_kwan:
mmxs6_kwan:


movq [ 56+eax],mm2
movq mm6,mm4	; copy a5

pxor mm6,[ 0+eax]	; x2 = ~a5
movq mm7,mm5	; copy a6

movq [ 48+eax],mm1
movq mm2,mm4	; copy a5

movq [ 64+eax],mm3
pxor mm7,mm1	; x3 = a2 ^ a6

pxor mm1,[ 0+eax]	; x1 = ~a2
pxor mm7,mm6	; x4 = x2 ^ x3

movq [ 80+eax],mm6
pxor mm7,mm0	; x5 = a1 ^ x4

pand mm2,mm5	; x6 = a5 & a6
movq mm6,mm4	; copy a5

movq [ 72+eax],mm1
movq mm3,mm5	; copy a6

pand mm3,[ 48+eax]	; x15 = a2 & a6
pand mm6,mm7	; x8 = a5 & x5

movq [ 40+eax],mm0
por mm1,mm2	; x7 = x1 | x6

movq [ 96+eax],mm2
pand mm0,mm6	; x9 = a1 & x8

movq [ 112+eax],mm3
pxor mm1,mm0	; x10 = x7 ^ x9

movq mm0,[ 64+eax]
movq mm2,mm4	; copy a5

movq [ 104+eax],mm6
pand mm0,mm1	; x11 = a4 & x10

movq [ 88+eax],mm7
pxor mm2,mm3	; x16 = a5 ^ x15

movq mm6,[ 80+eax]
pxor mm0,mm7	; x12 = x5 ^ x11

movq mm7,[ 40+eax]
pxor mm1,mm5	; x13 = a6 ^ x10

movq [ 120+eax],mm2
pand mm2,mm7	; x17 = a1 & x16

movq mm3,[ 64+eax]
pxor mm6,mm2	; x18 = x2 ^ x17

pxor mm2,[ 48+eax]	; x26 = a2 ^ x17
pand mm1,mm7	; x14 = a1 & x13

por mm3,mm6	; x19 = a4 | x18
pxor mm6,mm5	; x23 = a6 ^ x18

pxor mm1,mm3	; x20 = x14 ^ x19
pand mm7,mm6	; x24 = a1 & x23

pand mm1,[ 56+eax]	; x21 = a3 & x20
pand mm6,mm4	; x38 = a5 & x23

movq mm3,[ 96+eax]
pxor mm0,mm1	; x22 = x12 ^ x21

pxor mm0,[ 16+eax]	; out2 ^= x22
por mm3,mm2	; x27 = x6 | x26

pand mm3,[ 64+eax]	; x28 = a4 & x27
pxor mm4,mm7	; x25 = a5 ^ x24

movq mm1,[ 88+eax]
pxor mm4,mm3	; x29 = x25 ^ x28

pxor mm2,[ 0+eax]	; x30 = ~x26
por mm5,mm4	; x31 = a6 | x29

movq [ 16+eax],mm0
movq mm3,mm5	; copy x31

pandn mm3,[ 64+eax]	; x33 = a4 & ~x31
pxor mm1,mm6	; x39 = x5 ^ x38

movq mm0,[ 96+eax]
pxor mm3,mm2	; x34 = x30 ^ x33

por mm1,[ 64+eax]	; x40 = a4 | x39
pxor mm0,mm3	; x37 = x6 ^ x34

pand mm3,[ 56+eax]	; x35 = a3 & x34
pxor mm0,mm1	; x41 = x37 ^ x40

por mm6,[ 88+eax]	; x50 = x5 | x38
movq mm1,mm7	; copy x24

pxor mm7,[ 112+eax]	; x44 = x15 ^ x24
pxor mm4,mm3	; x36 = x29 ^ x35

movq mm3,[ 64+eax]
pxor mm7,mm5	; x45 = x31 ^ x44

pand mm5,[ 104+eax]	; x52 = x8 & x31
por mm7,mm3	; x46 = a4 | x45

pxor mm6,[ 96+eax]	; x51 = x6 ^ x50
por mm5,mm3	; x53 = a4 | x52

por mm1,[ 120+eax]	; x42 = x16 | x24
pxor mm5,mm6	; x54 = x51 ^ x53

pxor mm1,[ 72+eax]	; x43 = x1 ^ x42

movq mm3,[ 56+eax]
pxor mm7,mm1	; x47 = x43 ^ x46

pxor mm4,[ 32+eax]	; out4 ^= x36
por mm7,mm3	; x48 = a3 | x47

pand mm2,mm1	; x55 = x30 & x43
pxor mm0,mm7	; x49 = x41 ^ x48

pxor mm0,[ 8+eax]	; out1 ^= x49
por mm2,mm3	; x56 = a3 | x55

movq mm1,[ 16+eax]
pxor mm2,mm5	; x57 = x54 ^ x56

pxor mm2,[ 24+eax]	; out3 ^= x57


ret


; ==============================================
align 4
_mmxs7_kwan:
mmxs7_kwan:



movq [ 40+eax],mm0
movq mm6,mm1	; copy a2

movq [ 48+eax],mm1
movq mm7,mm3	; copy a4

movq [ 64+eax],mm5
pand mm6,mm3	; x3 = a2 & a4

movq [ 56+eax],mm3
pxor mm6,mm4	; x4 = a5 ^ x3

pxor mm4,[ 0+eax]	; x2 = ~a5
pand mm7,mm6	; x6 = a4 & x4

pand mm3,mm4	; x12 = a4 & x2
movq mm5,mm1	; copy a2

pxor mm6,mm2	; x5 = a3 ^ x4
pxor mm5,mm7	; x7 = a2 ^ x6

movq [ 72+eax],mm7
por mm4,mm1	; x14 = a2 | x2

por mm1,mm3	; x13 = a2 | x12
pxor mm7,mm6	; x25 = x5 ^ x6

movq [ 80+eax],mm5
pand mm4,mm2	; x15 = a3 & x14

pand mm5,mm2	; x8 = a3 & x7
por mm3,mm7	; x26 = x12 | x25

movq [ 104+eax],mm1
pxor mm0,mm5	; x9 = a1 ^ x8

por mm0,[ 64+eax]	; x10 = a6 | x9
pxor mm1,mm4	; x16 = x13 ^ x15

movq [ 112+eax],mm4
pxor mm0,mm6	; x11 = x5 ^ x10

movq [ 88+eax],mm5
movq mm4,mm3	; copy x26

movq mm6,[ 64+eax]
movq mm5,mm0	; copy x11

pxor mm5,[ 72+eax]	; x17 = x6 ^ x11
por mm4,mm6	; x27 = a6 | x26

movq [ 120+eax],mm7
por mm5,mm6	; x18 = a6 | x17

movq mm7,[ 40+eax]
pxor mm5,mm1	; x19 = x16 ^ x18

movq [ 128+eax],mm3
pand mm7,mm5	; x20 = a1 & x19

movq [ 96+eax],mm0
pxor mm7,mm0	; x21 = x11 ^ x20

movq mm3,[ 56+eax]
movq mm0,mm7	; copy x21

por mm0,[ 48+eax]	; x22 = a2 | x21
pand mm1,mm3	; x35 = a4 & x16

pand mm3,[ 104+eax]	; x39 = a4 & x13

por mm2,[ 80+eax]	; x40 = a3 | x7

pxor mm0,[ 72+eax]	; x23 = x6 ^ x22
pxor mm2,mm3	; x41 = x39 ^ x40

movq mm3,[ 48+eax]
movq mm6,mm0	; copy x23

pxor mm3,[ 0+eax]	; x1 = ~a2

pxor mm6,[ 112+eax]	; x24 = x15 ^ x23
por mm1,mm3	; x36 = x1 | x35

pand mm0,[ 128+eax]	; x30 = x23 & x26
pxor mm4,mm6	; x28 = x24 ^ x27

pand mm0,[ 64+eax]	; x31 = a6 & x30
por mm6,mm3	; x42 = x1 | x24

por mm6,[ 64+eax]	; x43 = a6 | x42
pand mm3,mm5	; x29 = x1 & x19

pand mm1,[ 64+eax]	; x37 = a6 & x36
pxor mm0,mm3	; x32 = x29 ^ x31

por mm0,[ 40+eax]	; x33 = a1 | x32
pxor mm2,mm6	; x44 = x41 ^ x48

pxor mm1,[ 96+eax]	; x38 = x11 ^ x37
pxor mm0,mm4	; x34 = x28 ^ x33

movq mm4,[ 40+eax]
pxor mm5,mm2	; x51 = x19 ^ x44

movq mm6,[ 56+eax]
por mm4,mm2	; x45 = a1 | x44

pxor mm6,[ 120+eax]	; x52 = a4 ^ x25
pxor mm1,mm4	; x46 = x38 ^ x45

movq mm4,[ 64+eax]
pand mm6,mm1	; x53 = x46 & x52

movq mm3,[ 72+eax]
pand mm6,mm4	; x54 = a6 & x53

pxor mm3,[ 112+eax]	; x48 = x6 ^ x15
pxor mm6,mm5	; x55 = x51 ^ x54

pxor mm2,[ 88+eax]	; x47 = x8 ^ x44
por mm3,mm4	; x49 = a6 | x48

por mm6,[ 40+eax]	; x56 = a1 | x55
pxor mm3,mm2	; x50 = x47 ^ x49

pxor mm7,[ 8+eax]	; out1 ^= x21
pxor mm3,mm6	; x57 = x50 ^ x56

pxor mm1,[ 16+eax]	; out2 ^= x46

pxor mm3,[ 24+eax]	; out3 ^= x57

pxor mm0,[ 32+eax]	; out4 ^= x34

 	; 48 clocks for 61 variables


ret


; ==============================================
align 4
_mmxs8_kwan:
mmxs8_kwan:


movq [ 40+eax],mm0
movq mm6,mm2	; copy a3

pxor mm0,[ 0+eax]	; x1 = ~a1
movq mm7,mm2	; copy a3

movq [ 56+eax],mm3
por mm7,mm0	; x4 = a3 | x1

pxor mm3,[ 0+eax]	; x2 = ~a4
pxor mm6,mm0	; x3 = a3 ^ x1

movq [ 72+eax],mm5
movq mm5,mm4	; copy a5

movq [ 48+eax],mm1
movq mm1,mm7	; copy x4

movq [ 64+eax],mm4
pxor mm7,mm3	; x5 = x2 ^ x4

por mm5,mm6	; x22 = a5 | x3
por mm0,mm7	; x8 = x1 | x5

pand mm1,mm4	; x26 = a5 & x4
pandn mm2,mm0	; x25 = x8 & ~a3

por mm4,mm7	; x6 = a5 | x5
pxor mm2,mm1	; x27 = x25 ^ x26

movq [ 88+eax],mm5
pand mm5,mm3	; x23 = x2 & x22

por mm2,[ 48+eax]	; x28 = a2 | x27
pxor mm7,mm4	; x32 = x5 ^ x6

pxor mm3,mm0	; x9 = x2 ^ x8
movq mm1,mm4	; copy x6

pxor mm7,[ 88+eax]	; x33 = x22 ^ x32
pxor mm1,mm3	; x14 = x6 ^ x9

pxor mm4,mm6	; x7 = x3 ^ x6
pxor mm2,mm5	; x29 = x23 ^ x28

pxor mm5,[ 40+eax]	; x39 = a1 ^ x23
pand mm6,mm3	; x15 = x3 & x9

movq [ 80+eax],mm1
pand mm5,mm4	; x40 = x7 & x39

movq [ 96+eax],mm7
movq mm1,mm0	; copy x8

pand mm3,[ 64+eax]	; x10 = a5 & x9
movq mm7,mm0	; copy x8

pand mm1,[ 64+eax]	; x16 = a5 & x8
pxor mm7,mm3	; x11 = x8 ^ x10

pand mm7,[ 48+eax]	; x12 = a2 & x11
pxor mm6,mm1	; x17 = x15 ^ x16

movq mm1,[ 72+eax]	; retrieve a6
pxor mm7,mm4	; x13 = x7 ^ x12

por mm6,[ 48+eax]	; x18 = a2 | x17
pandn mm4,mm0	; x48 = x8 & ~x17

pxor mm6,[ 80+eax]	; x19 = x14 ^ x18
pand mm1,mm2	; x30 = a6 & x29

pxor mm3,[ 40+eax]	; x45 = a1 ^ x10
pxor mm2,mm6	; x51 = x19 ^ x29

por mm6,[ 72+eax]	; x20 = a6 | x19
pxor mm1,mm7	; x31 = x13 ^ x30

pxor mm3,[ 88+eax]	; x46 = x22 ^ x45
pxor mm6,mm7	; x21 = x13 ^ x20

por mm4,[ 48+eax]	; x49 = a2 | x48

pand mm5,[ 48+eax]	; x41 = a2 & x40
pxor mm3,mm4	; x50 = x46 ^ x49

movq mm4,[ 40+eax]	; retrieve a1

pand mm4,[ 96+eax]	; x37 = a1 & x33

por mm7,[ 56+eax]	; x34 = a4 | x13
pxor mm0,mm4	; x38 = x8 ^ x37

pand mm7,[ 48+eax]	; x35 = a2 & x34
pxor mm5,mm0	; x42 = x38 ^ x41

movq mm4,[ 72+eax]	; retrieve a6
por mm2,mm0	; x52 = x38 | x51

pxor mm7,[ 96+eax]	; x36 = x33 ^ x35
por mm5,mm4	; x43 = a6 | x42

pxor mm6,[ 8+eax]	; out1 ^= x21
pand mm2,mm4	; x53 = a6 & x52

pxor mm1,[ 32+eax]	; out4 ^= x31
pxor mm5,mm7	; x44 = x36 ^ x43

pxor mm5,[ 24+eax]	; out3 ^= x44
pxor mm2,mm3	; x54 = x50 ^ x53

pxor mm2,[ 16+eax]	; out2 ^= x54

ret
end
