; DES bitslice driver for MMX by Bruce Ford
;
; Based on deseval-meggs3.cpp by Andrew Meggs
;
; Needs work on memory allocation and freeing, extern declarations etc. 
; for various platforms.
; Look for PLATFORM
;
;
; $Log: deseval-mmx.asm,v $
; Revision 1.2  1999/01/12 07:11:18  fordbr
; Debug code removed
;
; Revision 1.1  1999/01/12 03:30:19  fordbr
; DES bitslice driver for MMX
;  73 clocks per key on Pentium MMX
;  85                   AMD K6-2
;  90                   Intel PII
; 118                   AMD K6
;
;-----------------------------------------------

global whack16
global _whack16

%define mmNOT eax

%macro sbox_1 6-10 64,128,176,240
   movq  mm0,[ebx+248]

   movq  mm2,[ebx+8]

   movq  mm3,[ebx+16]

   movq  mm4,[ebx+24]

;   movq  mm5,[ebx+32]
   movq  mm5,mm6

   pxor  mm0,[%1]

   pxor  mm2,[%3]

   pxor  mm3,[%4]

   pxor  mm4,[%5]

   pxor  mm5,[%6]

   movq  [mmNOT+8],mm0
   movq  mm6,mm3              ; copy a4

   pxor  mm0,[mmNOT]          ; x2 = ~a1
   pxor  mm3,mm2              ; x3 = a3 ^ a4

   pxor  mm6,[mmNOT]          ; x1 = ~a4
   movq  mm7,mm0              ; copy x2

   movq  [mmNOT+16],mm4
   por   mm7,mm2              ; x5 = a3 | x2

   movq  [mmNOT+24],mm3
   movq  mm4,mm5              ; copy a6

   movq  [mmNOT+32],mm6
   pxor  mm3,mm0              ; x4 = x2 ^ x3

   movq  [mmNOT+40],mm7
   por   mm0,mm6              ; x9 = x1 | x2

   movq  [mmNOT+48],mm2
   pand  mm7,mm6              ; x6 = x1 & x5

   movq  [mmNOT+56],mm3
   por   mm2,mm3              ; x23 = a3 | x4

   pxor  mm2,[mmNOT]          ; x24 = ~x23
   pand  mm4,mm0              ; x10 = a6 & x9

   movq  mm1,[ebx]
   movq  mm6,mm7              ; copy x6

   pxor  mm1,[%2]
   por   mm2,mm5              ; x25 = a6 | x24

   movq  [mmNOT+64],mm7
   por   mm6,mm5              ; x7 = a6 | x6

   pxor  mm7,mm2              ; x26 = x6 ^ x25
   pxor  mm3,mm6              ; x8 = x4 ^ x7

   movq  [mmNOT+72],mm2
   pxor  mm6,mm4              ; x11 = x7 ^ x10

   pand  mm4,[mmNOT+48]       ; x38 = a3 & x10
   movq  mm2,mm6              ; copy x11

   pxor  mm6,[mmNOT+48]       ; x53 = a3 ^ x11
   por   mm2,mm1              ; x12 = a2 | x11

   pand  mm6,[mmNOT+40]       ; x54 = x5 & x53
   pxor  mm2,mm3              ; x13 = x8 ^ x12

   movq  [mmNOT+80],mm4
   pxor  mm0,mm2              ; x14 = x9 ^ x13

   movq  [mmNOT+88],mm7
   movq  mm4,mm5              ; copy a6

   movq  [mmNOT+96],mm2
   por   mm4,mm0              ; x15 = a6 | x14

   movq  mm7,[mmNOT+32]
   por   mm6,mm1              ; x55 = a2 | x54

   movq  [mmNOT+104],mm0
   movq  mm2,mm3              ; copy x8

   pandn mm0,[mmNOT+24]       ; x18 = x3 & ~x14
   pxor  mm4,mm7              ; x16 = x1 ^ x15

   por   mm5,[mmNOT+56]       ; x57 = a6 | x4
   por   mm0,mm1              ; x19 = a2 | x18

   pxor  mm5,[mmNOT+80]       ; x58 = x38 ^ x57
   pxor  mm4,mm0              ; x20 = x16 ^ x19

   movq  mm0,[mmNOT+16]
   pand  mm2,mm7              ; x27 = x1 & x8

   movq  [mmNOT+112],mm6
   por   mm2,mm1              ; x28 = a2 | x27

   movq  mm6,[mmNOT+104]
   por   mm0,mm4              ; x21 = a5 | x20

   pand  mm6,[mmNOT+40]       ; x32 = x5 & x14
   por   mm7,mm3              ; x30 = x1 | x8

   movq  [mmNOT+120],mm5
   pxor  mm6,mm3              ; x33 = x8 ^ x32

   pxor  mm7,[mmNOT+64]       ; x31 = x6 ^ x30
   movq  mm5,mm1              ; copy a2

   pxor  mm2,[mmNOT+88]       ; x29 = x26 ^ x28
   pand  mm5,mm6              ; x34 = a2 & x33

   pand  mm6,[mmNOT+48]       ; x40 = a3 & x33
   pxor  mm5,mm7              ; x35 = x31 ^ x34

   por   mm5,[mmNOT+16]       ; x36 = a5 | x35

   movq  mm7,[mmNOT+8]
   pxor  mm5,mm2              ; x37 = x29 ^ x36

   movq  mm2,[mmNOT+56]
   por   mm7,mm3              ; x46 = a1 | x8

   por   mm2,[mmNOT+80]       ; x39 = x4 | x38
   pxor  mm3,mm6              ; x52 = x8 ^ x40

   pxor  mm6,[mmNOT+72]       ; x41 = x25 ^ x40
   pxor  mm7,mm4              ; x47 = x46 ^ x20

   movq  mm4,[mmNOT+48]
   por   mm7,mm1              ; x48 = a2 | x47

   por   mm4,[mmNOT+88]       ; x44 = a3 | x26
   por   mm6,mm1              ; x42 = a2 | x41

   pxor  mm4,[mmNOT+104]      ; x45 = x14 ^ x44
   pxor  mm6,mm2              ; x43 = x39 ^ x42

   movq  mm2,[mmNOT+96]
   pxor  mm7,mm4              ; x49 = x45 ^ x48

   pxor  mm3,[mmNOT+112]      ; x56 = x52 ^ x55
   pxor  mm0,mm2              ; x22 = x13 ^ x21

   pxor  mm5,[esi+64]         ; out1 ^= x37
   pand  mm2,mm3              ; x59 = x13 & x56

   movq  mm4,[mmNOT+16]
   pand  mm2,mm1              ; x60 = a2 & x59

   pxor  mm2,[mmNOT+120]      ; x61 = x58 ^ x60
   pand  mm7,mm4              ; x50 = a5 & x49

   pxor  mm0,[esi+240]        ; out4 ^= x22
   pand  mm2,mm4              ; x62 = a5 & x61

   pxor  mm2,[esi+176]        ; out3 ^ x62
   pxor  mm7,mm6              ; x51 = x43 ^ x50

   pxor  mm7,[esi+128]        ; out2 ^= x51
   pxor  mm2,mm3              ; out3 = out3 ^ x62 ^ x56
                              

%if STORE_RESULT
   movq  [edi+%7],mm5         ; store out1
%endif

%if STORE_RESULT
   movq  [edi+%10],mm0        ; store out4
%endif

%if STORE_RESULT
   movq  [edi+%8],mm7         ; store out2
%endif

%if STORE_RESULT
   movq  [edi+%9],mm2         ; store out3
%endif

                              ; 64 clocks for 67 variables
%endmacro


%macro sbox_2 6-10 96,216,8,136
   movq  mm0,[ebx+24]

   movq  mm2,[ebx+40]

   movq  mm3,[ebx+48]

   movq  mm4,[ebx+56]

   movq  mm5,[ebx+64]

   pxor  mm0,[%1]

   pxor  mm2,[%3]

   pxor  mm3,[%4]

   pxor  mm4,[%5]

   pxor  mm5,[%6]

   movq  [mmNOT+8],mm3
   movq  mm6,mm4              ; copy a5

   movq  [mmNOT+16],mm0
   movq  mm7,mm4              ; copy a5

   pxor  mm0,[mmNOT]          ; x2 = ~a1
   pxor  mm6,mm5              ; x3 = a5 ^ a6

   pxor  mm7,[mmNOT]          ; x1 = ~a5
   movq  mm3,mm0              ; copy x2

   movq  [mmNOT+24],mm2
   por   mm7,mm5              ; x6 = a6 | x1

   movq  [mmNOT+32],mm6
   por   mm3,mm7              ; x7 = x2 | x6

   movq  mm1,[ebx+32]
   pxor  mm7,mm4              ; x13 = a5 ^ x6

   pxor  mm1,[%2]
   pxor  mm6,mm0              ; x4 = x2 ^ x3

   pand  mm3,mm1              ; x8 = a2 & x7
   por   mm2,mm7              ; x14 = a3 | x13

   movq  [mmNOT+40],mm1
   pxor  mm3,mm5              ; x9 = a6 ^ x8

   movq  [mmNOT+48],mm6
   pxor  mm6,mm1              ; x5 = a2 ^ x4

   movq  [mmNOT+56],mm7
   pand  mm1,mm3              ; x12 = a2 & x9

   pand  mm3,[mmNOT+24]       ; x10 = a3 & x9
   pxor  mm1,mm2              ; x15 = x12 ^ x14

   movq  mm7,[mmNOT+48]
   movq  mm2,mm1              ; copy x15

   pand  mm2,[mmNOT+8]        ; x16 = a4 & x15
   pxor  mm3,mm6              ; x11 = x5 ^ x10

   movq  [mmNOT+64],mm6
   pxor  mm3,mm2              ; x17 = x11 ^ x16

   movq  mm2,[mmNOT+16]
   por   mm7,mm5              ; x22 = a6 | x4

   por   mm1,mm2              ; x40 = a1 | x15
   pand  mm7,mm3              ; x23 = x17 & x22

   pxor  mm3,[esi+216]        ; out2 ^= x17
   por   mm2,mm4              ; x18 = a1 | a5

   por   mm7,[mmNOT+24]       ; x24 = a3 | x23
   movq  mm6,mm2              ; copy x18

   pxor  mm1,[mmNOT+56]       ; x41 = x13 ^ x40
   por   mm6,mm5              ; x19 = a6 | x18

%if STORE_RESULT
   movq  [edi+%8],mm3         ; store out2
%else
   movq  [mmNOT+88],mm3       ; store out2
%endif
   pand  mm4,mm0              ; x27 = a5 & x2

   movq  mm3,[mmNOT+56]
   por   mm5,mm0              ; x26 = a6 | x2

   movq  [mmNOT+72],mm2
   pxor  mm3,mm6              ; x20 = x13 ^ x19

   movq  mm2,[mmNOT+40]
   pxor  mm0,mm6              ; x31 = x2 ^ x19

   pxor  mm3,mm2              ; x21 = a2 ^ x20
   pand  mm0,mm2              ; x32 = a2 & x31

   pxor  mm7,mm3              ; x25 = x21 ^ x24
   por   mm2,mm4              ; x28 = a2 | x27

   pxor  mm4,[mmNOT+32]       ; x30 = x3 ^ x27
   pand  mm6,mm3              ; x47 = x19 & x21

   pxor  mm4,mm0              ; x33 = x30 ^ x32
   pxor  mm6,mm5              ; x48 = x26 ^ x47

   movq  [mmNOT+80],mm7
   pand  mm0,mm3              ; x38 = x21 & x32

   movq  mm7,[mmNOT+24]
   pxor  mm5,mm2              ; x29 = x26 ^ x28

   pxor  mm0,[mmNOT+64]       ; x39 = x5 ^ x38
   pand  mm7,mm4              ; x34 = a3 & x33

   pand  mm4,[mmNOT+40]       ; x49 = a2 & x33
   pxor  mm7,mm5              ; x35 = x29 ^ x34

   por   mm7,[mmNOT+8]        ; x36 = a4 | x35
   movq  mm5,mm1              ; copy x41

   por   mm5,[mmNOT+24]       ; x42 = a3 | x41
   por   mm1,mm2              ; x44 = x28 | x41

   pand  mm2,[mmNOT+72]       ; x53 = x18 & x28
   pxor  mm4,mm3              ; x50 = x21 ^ x49

   movq  mm3,[mmNOT+8]
   pand  mm2,mm4              ; x54 = x50 & x53

   pand  mm4,[mmNOT+24]       ; x51 = a3 & x50
   pxor  mm0,mm5              ; x43 = x39 ^ x42

   pxor  mm7,[mmNOT+80]       ; x37 = x25 ^ x36
   pxor  mm4,mm6              ; x52 = x48 ^ x51

   pxor  mm7,[esi+8]          ; out3 ^= x37
   pand  mm1,mm3              ; x45 = a4 & x44

   pxor  mm4,[esi+136]        ; out4 ^ x52
   pxor  mm1,mm0              ; x46 = x43 ^ x45

   pxor  mm1,[esi+96]         ; out1 ^= x46
   por   mm2,mm3              ; x55 = a4 | x54

%if STORE_RESULT
   movq  [edi+%9],mm7         ; store out3
%else
   movq  mm3,[mmNOT+88]       ; retrieve out2
%endif
   pxor  mm2,mm4              ; out4 = out4 ^ x52 ^ x55

%if STORE_RESULT
   movq  [edi+%7],mm1         ; store out1
%endif

%if STORE_RESULT
   movq  [edi+%10],mm2        ; store out4
%endif

                              ; 55 clocks for 60 variables
%endmacro


%macro sbox_3 6-10 184,120,232,40
   movq  mm0,[ebx+56]

   movq  mm2,[ebx+72]

   movq  mm3,[ebx+80]

   movq  mm4,[ebx+88]

   movq  mm5,[ebx+96]

   pxor  mm0,[%1]

   pxor  mm2,[%3]

   pxor  mm3,[%4]

   pxor  mm4,[%5]

   pxor  mm5,[%6]
                              ; mm6 free
                              ; mm7 free
   movq  [mmNOT+8],mm0        ;	mm0 free
   movq  mm6,mm5              ; mm6 = a6

   pxor  mm6,[mmNOT]          ;	mm6(x2) = ~a6
   movq  mm7,mm4              ; mm7 = a5

   movq  mm1,[ebx+64]
   pxor  mm7,mm6              ;	mm7(x9) = a5 ^ x2

   pxor  mm1,[%2]
   movq  mm0,mm4              ; mm0 = a5

   movq  [mmNOT+16],mm6       ;	mm6 free
   pand  mm0,mm2              ; mm0(x3) = a5 & a3

   movq  [mmNOT+24],mm7       ;	mm7 free
   pxor  mm0,mm5              ; mm0(x4) = x3 ^ a6

   movq  [mmNOT+32],mm4       ;	mm4 free
   pandn mm4,mm3              ; mm4(x5) = a4 & ~a5

   movq  [mmNOT+40],mm0       ;	mm0 free
   por   mm7,mm3              ; mm7(x10) = a4 | x9

   movq  mm6,[mmNOT+32]       ;	mm6 = a5
   pxor  mm0,mm4              ; mm0(x6) = x4 ^ x5

   movq  [mmNOT+48],mm5       ;	mm5 free
   pandn mm6,mm2              ; mm6(x8) = a3 & ~a5

   movq  [mmNOT+56],mm0       ;	mm0 free
   pxor  mm7,mm6              ; mm7(x11) = x8 ^ x10

   movq  mm5,[mmNOT+16]       ;	mm5 = x2
   pxor  mm0,mm1              ; mm0(x7) = x6 ^ a2

   movq  [mmNOT+64],mm4       ;	mm4 free
   movq  mm4,mm7              ; mm4 = x11

   por   mm5,[mmNOT+40]       ;	mm5(x23) = x1 | x4
   pand  mm4,mm0              ; mm4(x12) = x7 & x11

   movq  [mmNOT+72],mm7       ;	mm7 free
   pxor  mm6,mm5              ; mm6(x24) = x23 ^ x8

   pxor  mm7,[mmNOT+32]       ;	mm7(x13) = a5 ^ x11
   por   mm6,mm1              ; mm6(x25) = a2 | x24

   movq  [mmNOT+80],mm4       ;		# mm4 free
   pand  mm4,mm5              ; mm4(x54) = x12 & x23

   movq  [mmNOT+88],mm7       ;		# mm7 free
   por   mm7,mm0              ; mm7(x14) = x13 | x7

   movq  [mmNOT+96],mm4       ;		# mm4 free
   movq  mm4,mm2              ; mm4 = a3

   pxor  mm4,[mmNOT+24]       ;	# mm4 = a3 ^ x21
   pand  mm7,mm3              ; mm7(x15) = a4 & x14

   movq  [mmNOT+104],mm0      ;		# mm0 free
   pxor  mm4,mm3              ; mm4(x22) = a4 ^ a3 ^ x9

   pxor  mm5,[mmNOT+48]       ;	# mm5(x27) = a6 ^ x23
   pxor  mm6,mm4              ; mm6(x26) = x22 ^ x25
                              ; mm4 free
   movq  [mmNOT+112],mm3      ;		# mm3 free
   por   mm3,mm5              ; mm3(x28) = x27 | a4

   movq  [mmNOT+32],mm2       ;		# mm2 free
   pxor  mm5,mm3              ; mm5(x51) = x27 ^ x28

   por   mm5,mm1              ;	# mm5(x52) = x51 | a2
   pxor  mm2,mm7              ; mm2(x29) = a3 ^ x15

   pxor  mm7,[mmNOT+80]       ;	# mm7(x16) = x12 ^ x15
   movq  mm4,mm2              ; mm4 = x29

   por   mm2,[mmNOT+64]       ;	# mm2(x30) = x29 | x5
   pand  mm7,mm1              ; mm7(x17) = a2 & x16

   por   mm4,[mmNOT+40]       ;	# mm4(x37) = x29 | x4
   por   mm2,mm1              ; mm2(x31) = a2 | x30

   pxor  mm7,[mmNOT+72]       ;	# mm7(x18) = x17 ^ x11
   pxor  mm2,mm3              ; mm2(x32) = x31 ^ x28
                              ; mm3 free
   movq  mm3,[mmNOT+8]        ;	# mm1 = a3

   pxor  mm4,[mmNOT+112]      ;	# mm4(x38) = x37 ^ a4
   pand  mm7,mm3              ; mm7(x19) = x18 & a1

   pxor  mm7,[mmNOT+104]      ;	# mm7(x20) = x19 ^ x7
   por   mm2,mm3              ; mm2(x33) = a1 | x32

   movq  [mmNOT+40],mm4       ;		# mm4 free
   pxor  mm2,mm6              ; mm2(x34) = x26 ^ x33
                              ; mm6 free
   pxor  mm7,[esi+40]         ;	### mm7(out4) = out4 ^ x20
   por   mm4,mm1              ; mm4(x39) = a2 | x38

   movq  mm6,[mmNOT+32]       ;	# mm6 = a3
   movq  mm3,mm2              ; mm3 = x34

   pxor  mm6,[mmNOT+24]       ;	# mm6(x35) = a3 ^ x9

   por   mm6,[mmNOT+64]       ;	# mm6(x36) = x5 | x35

   pxor  mm3,[mmNOT+40]       ;	# mm3(x43) = x34 ^ x38
   pxor  mm4,mm6              ; mm4(x40) = x36 ^ x39

   movq  mm6,[mmNOT+48]       ;	# mm6 = a6

   pand  mm6,[mmNOT+72]       ;	# mm6(x41) = a6 & x11

   movq  mm0,[mmNOT+16]       ;	# mm0 = x2
   pxor  mm3,mm6              ; mm3(x44) = x43 ^ x41

   por   mm6,[mmNOT+56]       ;	# mm6(x42) = x41 | x6
   pand  mm3,mm1              ; mm3(x45) = x42 & a2

   por   mm0,[mmNOT+40]       ;	# mm0(x49) = x2 | x38
   pxor  mm3,mm6              ; mm3(x46) = x42 ^ x45
                              ; mm6 free
   pxor  mm0,[mmNOT+88]       ;	# mm0(x50) = x49 ^ x13
   movq  mm6,mm5              ; mm6 = x52

   por   mm3,[mmNOT+8]        ;	# mm3(x47) = x46 | a1
   pxor  mm0,mm5              ; mm0(x53) = x50 ^ x52

   pand  mm6,[mmNOT+96]       ;	# mm6(x55) = x52 & x54
   pxor  mm3,mm4              ; mm3(x48) = x40 ^ x47

   por   mm6,[mmNOT+8]        ;	# mm6(x56) = a1 | x55

   pxor  mm3,[esi+232]        ;	### mm3(out3) = out3 ^ x48
   pxor  mm6,mm0              ; mm6(x57) = x53 ^ x56

%if STORE_RESULT
   movq  [edi+%10],mm7        ; store out4
%endif

   pxor  mm2,[esi+184]        ;	### mm2(out1) = out1 ^ x34

%if STORE_RESULT
   movq  [edi+%9],mm3         ; store out3
%endif

   pxor  mm6,[esi+120]        ;	### mm6(out2) = out2 ^ x57

%if STORE_RESULT
   movq  [edi+%7],mm2         ; store out1
%endif

%if STORE_RESULT
   movq  [edi+%8],mm6         ; store out2
%endif

                              ; 64 clocks for 61 variables
%endmacro


%macro sbox_4 6-10 200,152,72,0
   movq  mm0,[ebx+88]

   movq  mm1,[ebx+96]

   movq  mm2,[ebx+104]

   movq  mm3,[ebx+112]

   movq  mm5,[ebx+128]

   pxor  mm0,[%1]

   pxor  mm1,[%2]

   pxor  mm2,[%3]

   pxor  mm3,[%4]

   pxor  mm5,[%6]

   movq  [mmNOT+24],mm1
   movq  mm6,mm2              ; copy a3

   movq  mm4,[ebx+120]
   movq  mm7,mm0              ; copy a1

   pxor  mm4,[%5]
   por   mm6,mm0              ; x3 = a1 | a3

   movq  [mmNOT+16],mm3
   pand  mm7,mm4              ; x8 = a1 & a5

   movq  [mmNOT+8],mm5
   movq  mm3,mm1              ; copy a2

   movq  [mmNOT+32],mm2
   movq  mm5,mm4              ; copy a5

   pand  mm5,mm6              ; x4 = a5 & x3
   por   mm3,mm2              ; x6 = a2 | a3

   pxor  mm2,[mmNOT]          ; x2 = ~a3
   pxor  mm0,mm5              ; ~x5 = a1 ^ x4

   pxor  mm0,[mmNOT]          ; x5 = ~(~x5)
   pxor  mm6,mm7              ; x9 = x8 ^ x3

   pxor  mm3,mm0              ; x7 = x5 ^ x6
   movq  mm7,mm1              ; copy a2

   pand  mm7,mm6              ; x10 = a2 & x9
   pxor  mm5,mm2              ; x14 = x2 ^ x4

   pxor  mm2,mm4              ; x18 = a5 ^ x2
   pand  mm0,mm5              ; x17 = x5 & x14

   pxor  mm4,mm7              ; x11 = a5 ^ x10
   pand  mm5,mm1              ; x15 = a2 & x14

   por   mm2,mm1              ; x19 = a2 | x18
   pxor  mm5,mm6              ; x16 = x9 ^ x15

   movq  mm1,[mmNOT+16]       ; retrieve a4
   movq  mm6,mm0              ; copy x17

   pand  mm1,mm4              ; x12 = a4 & x11
   pxor  mm6,mm2              ; x20 = x17 ^ x19

   por   mm6,[mmNOT+16]       ; x21 = a4 | x20
   pxor  mm1,mm3              ; x13 = x7 ^ x12

   pand  mm4,[mmNOT+24]       ; x28 = a2 & x11
   pxor  mm6,mm5              ; x22 = x16 ^ x21

   movq  mm3,[mmNOT+8]        ; retrieve a6
   pxor  mm4,mm0              ; x29 = x28 ^ x17

   pxor  mm7,[mmNOT+32]       ; x30 = a3 ^ x10
   movq  mm0,mm3              ; copy a6

   pxor  mm7,mm2              ; x31 = x30 ^ x19
   pand  mm0,mm6              ; x23 = a6 & x22

   movq  mm2,[mmNOT+16]       ; retrieve a4
   por   mm6,mm3              ; x26 = a6 | x22

   pxor  mm0,mm1              ; x24 = x13 ^ x23
   pand  mm7,mm2              ; x32 = a4 & x31

   pxor  mm1,[mmNOT]          ; x25 = ~x13
   pxor  mm4,mm7              ; x33 = x29 ^ x32

   movq  mm5,mm4              ; copy x33
   pxor  mm4,mm1              ; x34 = x25 ^ x33

   pxor  mm1,[esi+200]        ; out1 ^ x25
   por   mm2,mm4              ; x37 = a4 | x34

   pand  mm4,[mmNOT+24]       ; x35 = a2 & x34
   pxor  mm1,mm6              ; out1 = out1 ^ x25 ^ x26

   pxor  mm6,[esi+72]         ; out3 ^ x26
   pxor  mm4,mm0              ; x36 = x24 ^ x35

   pxor  mm0,[esi+152]        ; out2 ^= x24
   pxor  mm2,mm4              ; x38 = x36 ^ x37

%if STORE_RESULT
   movq  [edi+%7],mm1         ; store out1
%endif
   pand  mm3,mm2              ; x39 = a6 & x38

   pxor  mm6,mm2              ; out3 ^ x41 = out3 ^ x26 ^ x38
   pxor  mm5,mm3              ; x40 = x33 ^ x39

%if STORE_RESULT
   movq  [edi+%8],mm0         ; store out2
%endif
   pxor  mm6,mm5              ; out3 = out3 ^ x41 ^ x40

   pxor  mm5,[esi]            ; out4 ^= x40

%if STORE_RESULT
   movq  [edi+%9],mm6         ; store out3
%endif

%if STORE_RESULT
   movq  [edi+%10],mm5        ; store out4
%endif

                              ; 45 clocks for 46 variables
%endmacro


%macro sbox_5 6-10 56,104,192,16
   movq  mm0,[ebx+120]

   movq  mm1,[ebx+128]

   movq  mm2,[ebx+136]

   movq  mm3,[ebx+144]

   pxor  mm1,[%2]

   pxor  mm2,[%3]

   pxor  mm3,[%4]

   movq  [mmNOT+8],mm1        ;		# mm1 free
   movq  mm6,mm3              ; mm6 = a4

   pxor  mm0,[%1]
   movq  mm7,mm2              ;	# mm7 = a3

   movq  mm5,[ebx+160]
   pandn mm6,mm2              ; mm6(x1) = a3 & ~a4

   movq  mm4,[ebx+152]
   pandn mm7,mm0              ;	# mm7(x3) = a1 & ~a3

   pxor  mm5,[%6]
   movq  mm1,mm6              ; mm1 = x1

   movq  [mmNOT+16],mm0       ;		# mm0 free
   pxor  mm1,mm0              ; mm1(x2) = x1 ^ a1

   pxor  mm4,[%5]
   pxor  mm0,mm3              ;	# mm0(x6) = a4 ^ a1

   movq  [mmNOT+24],mm1       ;		# mm1 free
   por   mm6,mm0              ; mm6(x7) = x1 | x6

   movq  [mmNOT+32],mm5       ;		# mm5 free
   por   mm5,mm7              ; mm5(x4) = a6 | x3

   movq  [mmNOT+40],mm6       ;		# mm6 free
   pxor  mm1,mm5              ; mm1(x5) = x2 ^ x4

   movq  [mmNOT+48],mm5       ;		# %mm5 free
   pand  mm6,mm2              ; mm6 = a3 & x7

   movq  mm5,[mmNOT+32]       ;	# mm5 = a6
   pxor  mm6,mm3              ; mm6(x13) = (a3 & x7) ^ a4

   pandn mm5,[mmNOT+40]       ;	# mm5(x8) = x7 & ~a6

   movq  [mmNOT+56],mm0       ;		# mm0 free
   movq  mm0,mm7              ; mm0 = x3

   movq  [mmNOT+64],mm5       ;		# mm5 free
   pxor  mm5,mm2              ; mm5(x9) = a3 ^ x8

   movq  [mmNOT+72],mm1       ;		# mm1 free
   pxor  mm0,mm3              ; mm0 = x3 ^ a4

   movq  [mmNOT+80],mm5       ;		# mm5 free
   pandn mm7,mm6              ; mm7 = x13 & ~x3

   por   mm0,[mmNOT+32]       ;	# mm0(x16) = a6 | (x3 ^ a4)
   por   mm5,mm4              ; mm5 = a5 | x9

   movq  [mmNOT+88],mm6       ;		# mm6 free
   pxor  mm5,mm1              ; mm5 = x5 ^ (a5 | x9)

   movq  [mmNOT+96],mm0       ;		# mm0 free
   pxor  mm7,mm0              ; mm7(x17) = x16 ^ (x13 & ~x3)

   movq  mm0,[mmNOT+8]        ;	# mm0 = a2
   movq  mm1,mm4              ; mm1 = a5

   movq  [mmNOT+32],mm7       ;		# mm7 free
   por   mm1,mm7              ; mm1 = a5 | x17

   pand  mm7,[mmNOT+72]       ;	# mm7(x31) = x17 & x5
   pxor  mm1,mm6              ; mm1(x19) = x13 ^ (a5 | x17)

   pandn mm0,mm1              ;	# mm0 = x19 & ~a2
   movq  mm6,mm7              ; mm6 = x31

   pandn mm6,[mmNOT+40]       ;	# mm6(x32) = x7 & ~x31
   pxor  mm5,mm0              ; mm5(x21) = x5 ^ (a5 | x9) ^ (x19 & ~a2)

   pxor  mm7,[mmNOT+80]       ;	# mm7(x38) = x9 ^ x32
   movq  mm0,mm3              ; mm0 = a4

   movq  [mmNOT+40],mm5       ;		# mm5 free
   movq  mm5,mm6              ; mm5 = x32

   pandn mm0,[mmNOT+64]       ;	# mm0 = x8 & ~a4
   pandn mm5,mm1              ; mm5(x43) = x19 & ~x32
                              ; mm1 free
   pxor  mm6,[esi+192]        ;	# mm6 = out3 ^ x32
   pxor  mm0,mm2              ; mm0(x34) = (x8 & ~a4) ^ a3
                              ; mm2 free (no more references to a3)
   movq  mm2,[mmNOT+16]       ;	# mm2 = a1		# 'a1' local var free
   movq  mm1,mm0              ; mm1 = x34

   pxor  mm2,[mmNOT+80]       ;	# mm2(x24) = a1 ^ x9
   pand  mm1,mm4              ; mm1 = x34 & a5

   movq  [mmNOT+16],mm7       ;		# mm7 free
   pxor  mm6,mm1              ; mm6 = out3 ^ x32 ^ (x34 & a5)
                              ; mm1 free
   movq  mm1,[mmNOT+48]       ;	# mm1 = x4
   movq  mm7,mm2              ; mm7 = x24

   pand  mm7,[mmNOT+24]       ;	# mm7 = x2 & x24
   pand  mm1,mm3              ; mm1 = a4 & x4

   pxor  mm1,[mmNOT+32]       ;	# mm1 = (a4 & x4) ^ x17
   pandn mm7,mm4              ; mm7 = a5 & ~(x2 & x24)

   movq  [mmNOT+24],mm2       ;		# mm2 free
   pxor  mm1,mm7              ; mm1(x27) = (a4 & x4) ^ x17 ^ (x2 & x24)
                              ; mm7 free
   movq  mm7,[esi+104]        ;	# mm7 = out2
   por   mm3,mm2              ; mm3(x28) = a4 | x24

   movq  mm2,[mmNOT+8]        ;	# mm2 = a2
   pxor  mm7,mm1              ; mm7 = out2 ^ x27

   movq  [mmNOT+48],mm3       ;		# mm3 free
   pandn mm2,mm3              ; mm2 = x28 & ~a2

   movq  mm3,[mmNOT+16]       ;	# mm3 = x38
   pxor  mm7,mm2              ;## mm7(out2) = out2 ^ x27 ^ (x28 & ~a2)
                              ; mm2 free
   movq  mm2,[mmNOT+96]       ;	# mm2 = x16
   por   mm3,mm4              ; mm3 = x38 | a5

   por   mm2,[mmNOT+88]       ;	# mm2 = x13 | x16
   por   mm1,mm5              ; mm1 = x27 | x43

   pxor  mm5,[esi+56]         ;	# mm5 = out1 ^ x43
   pxor  mm2,mm3              ; mm2 = (x13 | x16) ^ (x38 | a5)
                              ; mm3 free
   por   mm2,[mmNOT+8]        ;	# mm2 = a2 | ((x13 | x16) ^ (x38 | a5))

   pxor  mm1,[mmNOT+56]       ;	# mm1 = (x27 | x43) ^ x6
   pxor  mm6,mm2              ; mm6 = out3 ^ x32 ^ (x34 & a5) ^ (a2 | ((x13 | x16) ^ (x38 | a5)))
                              ; mm2 free
   pxor  mm6,[mmNOT]          ;	### mm6(out3) = out3 ^ x32 ^ (x34 & a5) ^ ~(a2 | ((x13 | x16) ^ (x38 | a5)))
   pandn mm1,mm4              ; mm1 = a5 & ~((x27 | x43) ^ x6)

   movq  mm2,[mmNOT+16]       ;	# mm2 = x38

   pxor  mm1,[mmNOT+24]       ;	# mm1 = x24 ^ (a5 & ~((x27 | x43) ^ x6))
   movq  mm3,mm2              ; mm3 = x38

   pxor  mm2,[mmNOT+40]       ;	# mm2 = x21 ^ x38
   pxor  mm5,mm1              ; mm5 = out1 ^ x43 ^ x24 ^ (a5 & ~((x27 | x43) ^ x6))

   pand  mm3,[mmNOT+56]       ;	# mm3 = x6 & x38
   pandn mm2,mm4              ; mm2 = a5 & ~(x21 ^ x38)

   pand  mm2,[mmNOT+48]       ;	# mm2 = a5 & x28 & ~(x21 ^ x38)
   pxor  mm3,mm0              ; mm3 = (x6 & x38) ^ x34

   movq  mm4,[mmNOT+40]       ;	# mm4 = x21
   pxor  mm3,mm2              ; mm3 = (x6 & x38) ^ x34 ^ (a5 & x28 & ~(x21 ^ x38))

   por   mm3,[mmNOT+8]        ;	# mm3 = a2 | ((x6 & x38) ^ x34 ^ (a5 & x28 & ~(x21 ^ x38)))

%if STORE_RESULT
   movq  [edi+%8],mm7         ; store out2
%endif

   pxor  mm4,[esi+16]         ;	### mm4(out4) = out4 ^ x21

%if STORE_RESULT
   movq  [edi+%9],mm6         ; store out3
%endif
   pxor  mm5,mm3              ;## mm5(out1) = ...

%if STORE_RESULT
   movq  [edi+%10],mm4        ; store out4
%endif

%if STORE_RESULT
   movq  [edi+%7],mm5         ; store out1
%endif

                              ; 65 clocks for ?? variables
%endmacro


%macro sbox_6 6-10 24,224,80,144
   movq  mm1,[ebx+160]

   movq  mm2,[ebx+168]

   movq  mm3,[ebx+176]

   movq  mm4,[ebx+184]

   movq  mm5,[ebx+192]

   pxor  mm1,[%2]

   pxor  mm2,[%3]

   pxor  mm3,[%4]

   pxor  mm4,[%5]

   pxor  mm5,[%6]

   movq  [mmNOT+8],mm2
   movq  mm6,mm4              ; copy a5

   pxor  mm6,[mmNOT]          ; x2 = ~a5
   movq  mm7,mm5              ; copy a6

   movq  [mmNOT+16],mm1
   movq  mm2,mm4              ; copy a5

   movq  mm0,[ebx+152]
   pxor  mm7,mm1              ; x3 = a2 ^ a6

   pxor  mm0,[%1]
   pxor  mm7,mm6              ; x4 = x2 ^ x3

   movq  [mmNOT+32],mm6
   pxor  mm7,mm0              ; x5 = a1 ^ x4

   pxor  mm1,[mmNOT]          ; x1 = ~a2
   pand  mm2,mm5              ; x6 = a5 & a6

   movq  [mmNOT+24],mm3
   movq  mm6,mm4              ; copy a5

   movq  [mmNOT+40],mm1
   movq  mm3,mm5              ; copy a6

   pand  mm3,[mmNOT+16]       ; x15 = a2 & a6
   pand  mm6,mm7              ; x8 = a5 & x5

   movq  [mmNOT+48],mm0
   por   mm1,mm2              ; x7 = x1 | x6

   movq  [mmNOT+56],mm2
   pand  mm0,mm6              ; x9 = a1 & x8

   movq  [mmNOT+64],mm3
   pxor  mm1,mm0              ; x10 = x7 ^ x9

   movq  mm0,[mmNOT+24]
   movq  mm2,mm4              ; copy a5

   movq  [mmNOT+72],mm6
   pand  mm0,mm1              ; x11 = a4 & x10

   movq  [mmNOT+80],mm7
   pxor  mm2,mm3              ; x16 = a5 ^ x15

   movq  mm6,[mmNOT+32]
   pxor  mm0,mm7              ; x12 = x5 ^ x11

   movq  mm7,[mmNOT+48]
   pxor  mm1,mm5              ; x13 = a6 ^ x10

   movq  [mmNOT+88],mm2
   pand  mm2,mm7              ; x17 = a1 & x16

   movq  mm3,[mmNOT+24]
   pxor  mm6,mm2              ; x18 = x2 ^ x17

   pxor  mm2,[mmNOT+16]       ; x26 = a2 ^ x17
   pand  mm1,mm7              ; x14 = a1 & x13

   por   mm3,mm6              ; x19 = a4 | x18
   pxor  mm6,mm5              ; x23 = a6 ^ x18

   pxor  mm1,mm3              ; x20 = x14 ^ x19
   pand  mm7,mm6              ; x24 = a1 & x23

   pand  mm1,[mmNOT+8]        ; x21 = a3 & x20
   pand  mm6,mm4              ; x38 = a5 & x23

   movq  mm3,[mmNOT+56]
   pxor  mm0,mm1              ; x22 = x12 ^ x21

   pxor  mm0,[esi+224]        ; out2 ^= x22
   por   mm3,mm2              ; x27 = x6 | x26

   pand  mm3,[mmNOT+24]       ; x28 = a4 & x27
   pxor  mm4,mm7              ; x25 = a5 ^ x24

   movq  mm1,[mmNOT+80]
   pxor  mm4,mm3              ; x29 = x25 ^ x28

   pxor  mm2,[mmNOT]          ; x30 = ~x26
   por   mm5,mm4              ; x31 = a6 | x29

%if STORE_RESULT
   movq  [edi+%8],mm0         ; store out2
%else
   movq  [mmNOT+96],mm0       ; store out2
%endif
   movq  mm3,mm5              ; copy x31

   pandn mm3,[mmNOT+24]       ; x33 = a4 & ~x31
   pxor  mm1,mm6              ; x39 = x5 ^ x38

   movq  mm0,[mmNOT+56]
   pxor  mm3,mm2              ; x34 = x30 ^ x33

   por   mm1,[mmNOT+24]       ; x40 = a4 | x39
   pxor  mm0,mm3              ; x37 = x6 ^ x34

   pand  mm3,[mmNOT+8]        ; x35 = a3 & x34
   pxor  mm0,mm1              ; x41 = x37 ^ x40

   por   mm6,[mmNOT+80]       ; x50 = x5 | x38
   movq  mm1,mm7              ; copy x24

   pxor  mm7,[mmNOT+64]       ; x44 = x15 ^ x24
   pxor  mm4,mm3              ; x36 = x29 ^ x35

   movq  mm3,[mmNOT+24]
   pxor  mm7,mm5              ; x45 = x31 ^ x44

   pand  mm5,[mmNOT+72]       ; x52 = x8 & x31
   por   mm7,mm3              ; x46 = a4 | x45

   pxor  mm6,[mmNOT+56]       ; x51 = x6 ^ x50
   por   mm5,mm3              ; x53 = a4 | x52

   por   mm1,[mmNOT+88]       ; x42 = x16 | x24
   pxor  mm5,mm6              ; x54 = x51 ^ x53

   pxor  mm1,[mmNOT+40]       ; x43 = x1 ^ x42

   movq  mm3,[mmNOT+8]
   pxor  mm7,mm1              ; x47 = x43 ^ x46

   pxor  mm4,[esi+144]        ; out4 ^= x36
   por   mm7,mm3              ; x48 = a3 | x47

   pxor  mm5,[esi+80]         ; out3 ^ x54
   pxor  mm0,mm7              ; x49 = x41 ^ x48

   pxor  mm0,[esi+24]         ; out1 ^= x49
   pand  mm2,mm1              ; x55 = x30 & x43

%if STORE_RESULT
   movq  [edi+%10],mm4        ; store out4
%else
   movq  mm1,[mmNOT+96]       ; retrieve out2
%endif
   por   mm2,mm3              ; x56 = a3 | x55

%if STORE_RESULT
   movq  [edi+%7],mm0         ; store out1
%endif
   pxor  mm2,mm5              ; out3 = out3 ^ x54 ^ x56

%if STORE_RESULT
   ;stall

   movq  [edi+%9],mm2         ; store out3
%endif

                              ; 59 clocks for 61 variables
%endmacro


%macro sbox_7 6-10 248,88,168,48
   movq  mm1,[ebx+192]

   movq  mm3,[ebx+208]

   movq  mm4,[ebx+216]

   movq  mm5,[ebx+224]

   pxor  mm1,[%2]

   pxor  mm3,[%4]

   pxor  mm5,[%6]
   movq  mm6,mm1              ; copy a2

   pxor  mm4,[%5]
   movq  mm7,mm3              ; copy a4

   movq  [mmNOT+24],mm5
   pand  mm6,mm3              ; x3 = a2 & a4

   movq  [mmNOT+32],mm3
   pxor  mm6,mm4              ; x4 = a5 ^ x3

   pxor  mm4,[mmNOT]          ; x2 = ~a5
   pand  mm7,mm6              ; x6 = a4 & x4

   movq  mm2,[ebx+200]
   pand  mm3,mm4              ; x12 = a4 & x2

   movq  mm0,[ebx+184]
   movq  mm5,mm1              ; copy a2

   pxor  mm2,[%3]
   pxor  mm5,mm7              ; x7 = a2 ^ x6

   pxor  mm0,[%1]
   pxor  mm6,mm2              ; x5 = a3 ^ x4

   movq  [mmNOT+40],mm7
   por   mm4,mm1              ; x14 = a2 | x2

   movq  [mmNOT+16],mm1
   pxor  mm7,mm6              ; x25 = x5 ^ x6

   movq  [mmNOT+8],mm0
   por   mm1,mm3              ; x13 = a2 | x12

   movq  [mmNOT+48],mm5
   pand  mm4,mm2              ; x15 = a3 & x14

   pand  mm5,mm2              ; x8 = a3 & x7
   por   mm3,mm7              ; x26 = x12 | x25

   movq  [mmNOT+56],mm1
   pxor  mm0,mm5              ; x9 = a1 ^ x8

   por   mm0,[mmNOT+24]       ; x10 = a6 | x9
   pxor  mm1,mm4              ; x16 = x13 ^ x15

   movq  [mmNOT+64],mm4
   pxor  mm0,mm6              ; x11 = x5 ^ x10

   movq  [mmNOT+72],mm5
   movq  mm4,mm3              ; copy x26

   movq  mm6,[mmNOT+24]
   movq  mm5,mm0              ; copy x11

   pxor  mm5,[mmNOT+40]       ; x17 = x6 ^ x11
   por   mm4,mm6              ; x27 = a6 | x26

   movq  [mmNOT+80],mm7
   por   mm5,mm6              ; x18 = a6 | x17

   movq  mm7,[mmNOT+8]
   pxor  mm5,mm1              ; x19 = x16 ^ x18

   movq  [mmNOT+88],mm3
   pand  mm7,mm5              ; x20 = a1 & x19

   movq  [mmNOT+96],mm0
   pxor  mm7,mm0              ; x21 = x11 ^ x20

   movq  mm3,[mmNOT+32]
   movq  mm0,mm7              ; copy x21

   por   mm0,[mmNOT+16]       ; x22 = a2 | x21
   pand  mm1,mm3              ; x35 = a4 & x16

   pand  mm3,[mmNOT+56]       ; x39 = a4 & x13

   por   mm2,[mmNOT+48]       ; x40 = a3 | x7

   pxor  mm0,[mmNOT+40]       ; x23 = x6 ^ x22
   pxor  mm2,mm3              ; x41 = x39 ^ x40

   movq  mm3,[mmNOT+16]
   movq  mm6,mm0              ; copy x23

   pxor  mm3,[mmNOT]          ; x1 = ~a2

   pxor  mm6,[mmNOT+64]       ; x24 = x15 ^ x23
   por   mm1,mm3              ; x36 = x1 | x35

   pand  mm0,[mmNOT+88]       ; x30 = x23 & x26
   pxor  mm4,mm6              ; x28 = x24 ^ x27

   pand  mm0,[mmNOT+24]       ; x31 = a6 & x30
   por   mm6,mm3              ; x42 = x1 | x24

   por   mm6,[mmNOT+24]       ; x43 = a6 | x42
   pand  mm3,mm5              ; x29 = x1 & x19

   pand  mm1,[mmNOT+24]       ; x37 = a6 & x36
   pxor  mm0,mm3              ; x32 = x29 ^ x31

   por   mm0,[mmNOT+8]        ; x33 = a1 | x32
   pxor  mm2,mm6              ; x44 = x41 ^ x48

   pxor  mm1,[mmNOT+96]       ; x38 = x11 ^ x37
   pxor  mm0,mm4              ; x34 = x28 ^ x33

   movq  mm4,[mmNOT+8]
   pxor  mm5,mm2              ; x51 = x19 ^ x44

   movq  mm6,[mmNOT+32]
   por   mm4,mm2              ; x45 = a1 | x44

   pxor  mm6,[mmNOT+80]       ; x52 = a4 ^ x25
   pxor  mm1,mm4              ; x46 = x38 ^ x45

   movq  mm4,[mmNOT+24]
   pand  mm6,mm1              ; x53 = x46 & x52

   movq  mm3,[mmNOT+40]
   pand  mm6,mm4              ; x54 = a6 & x53

   pxor  mm3,[mmNOT+64]       ; x48 = x6 ^ x15
   pxor  mm6,mm5              ; x55 = x51 ^ x54

   pxor  mm2,[mmNOT+72]       ; x47 = x8 ^ x44
   por   mm3,mm4              ; x49 = a6 | x48

   por   mm6,[mmNOT+8]        ; x56 = a1 | x55
   pxor  mm3,mm2              ; x50 = x47 ^ x49

   pxor  mm7,[esi+248]        ; out1 ^= x21
   pxor  mm3,mm6              ; x57 = x50 ^ x56

   pxor  mm1,[esi+88]         ; out2 ^= x46

%if STORE_RESULT
   movq  [edi+%7],mm7         ; store out1
%endif

   pxor  mm3,[esi+168]        ; out3 ^= x57

%if STORE_RESULT
   movq  [edi+%8],mm1         ; store out2
%endif

   pxor  mm0,[esi+48]         ; out4 ^= x34

%if STORE_RESULT
   movq  [edi+%9],mm3         ; store out3
%endif

%if STORE_RESULT
   movq  [edi+%10],mm0        ; store out4
%endif

                              ; 60 clocks for 61 variables
%endmacro


%macro sbox_8 6
   movq  mm0,[ebx+216]

   movq  mm1,[ebx+224]

   movq  mm2,[ebx+232]

   movq  mm3,[ebx+240]

   movq  mm4,[ebx+248]

   movq  mm5,[ebx]

   pxor  mm0,[%1]

   pxor  mm1,[%2]

   pxor  mm2,[%3]

   pxor  mm3,[%4]

   pxor  mm4,[%5]

   pxor  mm5,[%6]

   movq  [mmNOT+8],mm0
   movq  mm6,mm2              ; copy a3

   pxor  mm0,[mmNOT]          ; x1 = ~a1
   movq  mm7,mm2              ; copy a3

   movq  [mmNOT+16],mm3
   por   mm7,mm0              ; x4 = a3 | x1

   pxor  mm3,[mmNOT]          ; x2 = ~a4
   pxor  mm6,mm0              ; x3 = a3 ^ x1

   movq  [mmNOT+24],mm5
   movq  mm5,mm4              ; copy a5

   movq  [mmNOT+32],mm1
   movq  mm1,mm7              ; copy x4

   movq  [mmNOT+40],mm4
   pxor  mm7,mm3              ; x5 = x2 ^ x4

   por   mm5,mm6              ; x22 = a5 | x3
   por   mm0,mm7              ; x8 = x1 | x5

   pand  mm1,mm4              ; x26 = a5 & x4
   pandn mm2,mm0              ; x25 = x8 & ~a3

   por   mm4,mm7              ; x6 = a5 | x5
   pxor  mm2,mm1              ; x27 = x25 ^ x26

   movq  [mmNOT+48],mm5
   pand  mm5,mm3              ; x23 = x2 & x22

   por   mm2,[mmNOT+32]       ; x28 = a2 | x27
   pxor  mm7,mm4              ; x32 = x5 ^ x6

   pxor  mm3,mm0              ; x9 = x2 ^ x8
   movq  mm1,mm4              ; copy x6

   pxor  mm7,[mmNOT+48]       ; x33 = x22 ^ x32
   pxor  mm1,mm3              ; x14 = x6 ^ x9

   pxor  mm4,mm6              ; x7 = x3 ^ x6
   pxor  mm2,mm5              ; x29 = x23 ^ x28

   pxor  mm5,[mmNOT+8]        ; x39 = a1 ^ x23
   pand  mm6,mm3              ; x15 = x3 & x9

   movq  [mmNOT+56],mm1
   pand  mm5,mm4              ; x40 = x7 & x39

   movq  [mmNOT+64],mm7
   movq  mm1,mm0              ; copy x8

   pand  mm3,[mmNOT+40]       ; x10 = a5 & x9
   movq  mm7,mm0              ; copy x8

   pand  mm1,[mmNOT+40]       ; x16 = a5 & x8
   pxor  mm7,mm3              ; x11 = x8 ^ x10

   pand  mm7,[mmNOT+32]       ; x12 = a2 & x11
   pxor  mm6,mm1              ; x17 = x15 ^ x16

   movq  mm1,[mmNOT+24]       ; retrieve a6
   pxor  mm7,mm4              ; x13 = x7 ^ x12

   por   mm6,[mmNOT+32]       ; x18 = a2 | x17
   pandn mm4,mm0              ; x48 = x8 & ~x17

   pxor  mm6,[mmNOT+56]       ; x19 = x14 ^ x18
   pand  mm1,mm2              ; x30 = a6 & x29

   pxor  mm3,[mmNOT+8]        ; x45 = a1 ^ x10
   pxor  mm2,mm6              ; x51 = x19 ^ x29

   por   mm6,[mmNOT+24]       ; x20 = a6 | x19
   pxor  mm1,mm7              ; x31 = x13 ^ x30

   pxor  mm3,[mmNOT+48]       ; x46 = x22 ^ x45
   pxor  mm6,mm7              ; x21 = x13 ^ x20

   por   mm4,[mmNOT+32]       ; x49 = a2 | x48

   pand  mm5,[mmNOT+32]       ; x41 = a2 & x40
   pxor  mm3,mm4              ; x50 = x46 ^ x49

   movq  mm4,[mmNOT+8]        ; retrieve a1

   pand  mm4,[mmNOT+64]       ; x37 = a1 & x33

   por   mm7,[mmNOT+16]       ; x34 = a4 | x13
   pxor  mm0,mm4              ; x38 = x8 ^ x37

   pand  mm7,[mmNOT+32]       ; x35 = a2 & x34
   pxor  mm5,mm0              ; x42 = x38 ^ x41

   movq  mm4,[mmNOT+24]       ; retrieve a6
   por   mm2,mm0              ; x52 = x38 | x51

   pxor  mm7,[mmNOT+64]       ; x36 = x33 ^ x35
   por   mm5,mm4              ; x43 = a6 | x42

   pxor  mm6,[esi+32]         ; out1 ^= x21
   pand  mm2,mm4              ; x53 = a6 & x52

   pxor  mm1,[esi+160]        ; out4 ^= x31
   pxor  mm5,mm7              ; x44 = x36 ^ x43

   pxor  mm5,[esi+112]        ; out3 ^= x44
   pxor  mm2,mm3              ; x54 = x50 ^ x53

; These five instructions are removed for pairing with 
; integer instructions at the end of full_round
;   pxor  mm2,[esi+208]        ; out2 ^= x54

;   movq  [edi+32],mm6         ; store out1

;   movq  [edi+160],mm1        ; store out4

;   movq  [edi+112],mm5        ; store out3

;   movq  [edi+208],mm2        ; store out2

                              ; 55 clocks for 58 variables
%endmacro

; PLATFORM  text segment definition
%ifdef BCC
SECTION TEXT USE32 ALIGN=16
%else
[SECTION .text]
%endif

; PLATFORM external routines and memory allocation

extern _malloc, _free

%macro getmem 1
; amount given by argument and leaves a pointer in eax
   push  dword %1
   call  _malloc
%endmacro

%macro relmem 0
; expects the pointer returned from getmem(_malloc) to be on the stack
   call  _free
%endmacro

; PLATFORM yielding if required
; yield1ms is approximately every 1 ms on a Pentium MMX 166MHz
; yield10ms is approximately every 10ms

%define yield1ms
%define yield10ms

%include "startup.asm"

%define STORE_RESULT 1

create_tail:
   lea   ebx, [CL]
   lea   esi, [CR]

   lea   edi, [R14]
   lea   edx, [round16]

   add   edi, ebp
   call  full_round

   mov   ebp, edi
   lea   edi, [edi+L13]

   call  full_round

   mov   ebp, edi
   shr   edi, 5

   ; movq  mm6, [ebx+32] comes from sbox_8 in full_round above
   sbox_1 rnd16_kbit19, rnd16_kbit40, rnd16_kbit55, rnd16_kbit32, rnd16_kbit10, rnd16_kbit13, R12_08, R12_16, R12_22, R12_30
   sbox_3 rnd16_kbit25, rnd16_kbit54, rnd15_kbit05, rnd16_kbit06, rnd16_kbit46, rnd16_kbit34, R12_23, R12_15, R12_29, R12_05

   add   ebp, 100h
   ; Set mm0 to all ones, same as mmNOT, but does not require memory access
   pcmpeqd  mm0, mm0

   test  ebp, 100h
   jz    t4_18
   ; Modify key bit 10 in rounds 15 and 16
   pxor  mm0, [rnd16_kbit10]
   ; stall
   movq  [rnd16_kbit10], mm0
   movq  [rnd15_kbit10], mm0
   jmp   create_tail

t4_18:
   test  ebp, 200h
   jz    t4_46
   ; Modify key bit 18 in rounds 15 and 16
   pxor  mm0, [rnd16_kbit18]
   ; stall
   movq  [rnd16_kbit18], mm0
   movq  [rnd15_kbit18], mm0
   jmp   create_tail

t4_46:
   test  ebp, 400h
   jz    t4_49
   ; Modify key bit 46 in rounds 15 and 16
   pxor  mm0, [rnd16_kbit46]
   ; stall
   movq  [rnd16_kbit46], mm0
   movq  [rnd15_kbit46], mm0
   jmp   create_tail

t4_49:
   ; Modify key bit 49 in rounds 15 and 16
   pxor  mm0, [rnd16_kbit49]
   test  ebp, 800h
   ; stall
   ; By jumping after these bits are reset,
   ; they are correct for the start of the next outermost loop
   movq  [rnd16_kbit49], mm0
   movq  [rnd15_kbit49], mm0
 
   jnz   near create_tail

   retn

%define STORE_RESULT 0
fix_r14s1s3:
   movq  mm6, [ebx+32]

   sbox_1 keybit19, keybit40, keybit55, keybit32, keybit10, keybit13

   movq  [ebp+R12_08],mm5     ; bit 8 round 12

   movq  [ebp+R12_30],mm0     ; bit 30 round 12

   movq  [ebp+R12_16],mm7     ; bit 16 round 12

   movq  [ebp+R12_22],mm2     ; bit 22 round 12
   add   edi, 256

   sbox_3 keybit25, keybit54, keybit05, keybit06, keybit46, keybit34

   movq  [ebp+R12_05],mm7     ; bit 5 round 12
   mov   ebx, esi

   movq  [ebp+R12_29],mm3     ; bit 29 round 12
   add   ebx, 256

   movq  [ebp+R12_23],mm2     ; bit 23 round 12
   lea   esi, [CL]

   movq  [ebp+R12_15],mm6     ; bit 15 round 12
   add   ebp, 8

   retn
%define STORE_RESULT 1

partial_round:
   test  ecx, 80h
   jz    near partial_2
   movq  mm6, [ebx+32]
   sbox_1 edx,     edx+8,   edx+16,  edx+24,  edx+32,  edx+40
partial_2:
   test  ecx, 40h
   jz    near partial_3
   sbox_2 edx+48,  edx+56,  edx+64,  edx+72,  edx+80,  edx+88
partial_3:
   test  ecx, 20h
   jz    near partial_5
   sbox_3 edx+96,  edx+104, edx+112, edx+120, edx+128, edx+136
partial_5:
   test  ecx, 08h
   jz    near partial_6
   sbox_5 edx+192, edx+200, edx+208, edx+216, edx+224, edx+232
partial_6:
   test  ecx, 04h
   jz    near partial_7
   sbox_6 edx+240, edx+248, edx+256, edx+264, edx+272, edx+280
partial_7:
   test  ecx, 02h
   jz    near partial_8
   sbox_7 edx+288, edx+296, edx+304, edx+312, edx+320, edx+328
partial_8:
   test  ecx, 01h
   jz    near partial_end
   sbox_8 edx+336, edx+344, edx+352, edx+360, edx+368, edx+376
   pxor  mm2,[esi+208]        ; out2 ^= x54

   movq  [edi+32],mm6         ; store out1

   movq  [edi+160],mm1        ; store out4

   movq  [edi+112],mm5        ; store out3

   movq  [edi+208],mm2        ; store out2
partial_end:
; S-Box 4 is done in all calls to partial_round so we can save on the test
   sbox_4 edx+144, edx+152, edx+160, edx+168, edx+176, edx+184
   retn

full_round:
   movq  mm6, [ebx+32]
   lea   ecx, [round11]

next_round:
   sbox_1 edx,     edx+8,   edx+16,  edx+24,  edx+32,  edx+40
   sbox_2 edx+48,  edx+56,  edx+64,  edx+72,  edx+80,  edx+88
   sbox_3 edx+96,  edx+104, edx+112, edx+120, edx+128, edx+136
   sbox_4 edx+144, edx+152, edx+160, edx+168, edx+176, edx+184
   sbox_5 edx+192, edx+200, edx+208, edx+216, edx+224, edx+232
   sbox_6 edx+240, edx+248, edx+256, edx+264, edx+272, edx+280
   sbox_7 edx+288, edx+296, edx+304, edx+312, edx+320, edx+328
   sbox_8 edx+336, edx+344, edx+352, edx+360, edx+368, edx+376

   pxor  mm2,[esi+208]        ; out2 ^= x54
   mov   esi, ebx

   movq  [edi+32],mm6         ; store out1
   mov   ebx, edi

   movq  [edi+160],mm1        ; store out4
   add   edx, 384

   movq  [edi+112],mm5        ; store out3
   mov   edi, ebp

   movq  [ebx+208],mm2        ; store out2
   mov   ebp, ebx

   cmp   edx, ecx
   jb   near next_round

full_end:
   retn

whack16:
_whack16:
whack16_:
   			  ; esp+32 *K
   			  ; esp+28 *C
   			  ; esp+24 *P
   			  ; esp+20 (retn)

   push  ebp  ; esp+16
   push  ebx  ; esp+12

   push  ecx  ; esp+8
   push  esi  ; esp+4

   push  edi  ; esp
   mov   ebp, [esp+32]

   mov   esi, [esp+28]
   mov   ebx, [esp+24]

   getmem mem_needed

   pop   edx                 ; remove argument from the stack
   push  eax                 ; keep the pointer to allocated memory

   add   eax, 7              ; Align eax on a quadword boundary

   and   eax, 0fffffff8h
   call  startup

   mov   dword [headstep1], 0
   pcmpeqd mm0, mm0

   mov   dword [tailstep], 0
   mov   dword [headstep2], 0

   movq  [mmNOT], mm0
   xor   ebp, ebp

tail_setup:
   call  create_tail

   lea   ebx, [PR]
   lea   esi, [PL]

   lea   edi, [L1]
   lea   edx, [round1]

   call  full_round

   lea   edi, [R2]
   call  full_round

tail_6:
   lea   edi, [L3]
   call  full_round

head_4:
   lea   edi, [R]
   lea   ebp, [L]

   ; Rounds 4 to 10
   lea   edx, [round4]
   call  full_round

   ; Round 11 S-Boxes 1, 2, 4, 5, 6, 7
   mov   ebp, [headstep1]
   mov   ecx, 0deh

   call  partial_round

   mov   esi, ebx
   mov   ebx, edi

%define STORE_RESULT 0
   sbox_3 keybit54, keybit26, keybit34, keybit03, keybit18, keybit06

   movq  [esi+40], mm7
   movq  mm4, mm3

   pxor  mm7, [ebp+R12_05]    ; bit 5
   movq  mm5, mm2

   pxor  mm3, [ebp+R12_29]    ; bit 29
   movq  mm1, mm6

   pxor  mm2, [ebp+R12_23]    ; bit 23
   por   mm7, mm3

   pxor  mm6, [ebp+R12_15]    ; bit 15
   por   mm7, mm2

   movq  [esi+184], mm5
   por   mm7, mm6

   movq  [esi+120], mm1
   movq  mm2, mm7

   movd  ecx, mm7
   punpckhdq mm2, mm2

   movq  [temp1], mm4         ; stored to temporary as bit 29 is possibly needed for round 11 s-box 8
   cmp   ecx, 0ffffffffh

   movd  edx, mm2
   jne   R12_sbox1

   cmp   edx, 0ffffffffh
   je    near step_head

R12_sbox1:
   ; Round 11
   mov   ebx, esi
   mov   esi, edi

   sbox_8 keybit01, keybit28, keybit29, keybit45, keybit23, keybit44

   pxor  mm2, [esi+208]       ; out2 ^= x54

   movq  [edi+32], mm6        ; store out1

   movq  [edi+160], mm1       ; store out4
   mov   esi, ebx

   movq  [edi+112], mm5       ; store out3
   mov   ebx, edi

   movq  [edi+208], mm2       ; store out2

   ; movq  mm6, [ebx+32] comes from sbox_8 above
   sbox_1 keybit48, keybit12, keybit27, keybit04, keybit39, keybit10

   movq  [esi+240], mm0
   movq  mm6, mm5

   pxor  mm0, [ebp+R12_30]    ; bit 30
   movq  mm4, mm7

   pxor  mm5, [ebp+R12_08]    ; bit 8
   movq  mm3, mm2

   pxor  mm7, [ebp+R12_16]    ; bit 16
   por   mm5, mm0

   pxor  mm2, [ebp+R12_22]    ; bit 22
   por   mm5, mm7

   movd  mm0, ecx
   por   mm5, mm2

   movq  [temp2], mm6         ; stored to temporary as bit 8 is possibly needed for round 11 s-box 3
   por   mm0, mm5

   movd  mm1, edx
   punpckhdq mm5, mm5

   movd  ecx, mm0
   por   mm1, mm5

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   R12_sbox2

   cmp   edx, 0ffffffffh
   je    near step_head

R12_sbox2:
   ; last box in round 11
   movq  [R+128], mm4
   mov   ebx, esi

   movq  [R+176], mm3
   mov   esi, edi

   sbox_3 keybit40, keybit12, keybit20, keybit46, keybit04, keybit17

   movq  mm0, [temp1]         ; Recover bit 29
   shl   ebp, 5

   movq  mm1, [temp2]         ; Recover bit 8

   movq  [edi+40],mm7         ; store out4

   movq  [edi+232],mm3        ; store out3

   movq  [edi+184],mm2        ; store out1
   lea   esi, [R14+ebp]

   movq  [edi+120],mm6        ; store out2
   lea   ebx, [L13+ebp]

   movq  [R+232], mm0         ; bit R29 after round 12

   movq  [R+64], mm1          ; bit R8 after round 12

   ; round 14 decrypt
   sbox_2 keybit24, keybit03, keybit26, keybit20, keybit11, keybit48

   movq  [temp3],mm7          ; store out3

   movq  [temp1],mm1          ; store out1
   lea   esi, [R]

   movq  [temp2],mm3          ; store out2
   lea   ebx, [L]

   movq  [temp4],mm2          ; store out4

   ; round 12 encrypt
   sbox_2 keybit53, keybit32, keybit55, keybit17, keybit40, keybit20

   movq  [esi+8], mm7
   movq  mm6, mm1

   pxor  mm7, [temp3]         ; bit 1
   movq  mm5, mm3

   pxor  mm1, [temp1]         ; bit 12
   movq  mm4, mm2

   pxor  mm3, [temp2]         ; bit 27
   por   mm7, mm1

   pxor  mm2, [temp4]         ; bit 17
   por   mm7, mm3

   movd  mm0, ecx
   por   mm7, mm2

   movd  mm1, edx
   por   mm0, mm7

   movq  [esi+96], mm6
   punpckhdq mm7, mm7

   movd  ecx, mm0
   por   mm1, mm7

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   near R12_sbox4

   cmp   edx, 0ffffffffh
   je    near step_head

R12_sbox4:
   movq  [R+216], mm5
   lea   esi, [R14+ebp]

   movq  [R+136], mm4
   lea   ebx, [L13+ebp]

   ; round 14 decrypt
   sbox_4 keybit33, keybit27, keybit53, keybit04, keybit12, keybit17

   movq  [temp1],mm1          ; store out1

   movq  [temp2],mm0          ; store out2
   lea   esi, [R]

   movq  [temp3],mm6          ; store out3
   lea   ebx, [L]

   movq  [temp4],mm5          ; store out4

   ; round 12 encrypt
   sbox_4 keybit05, keybit24, keybit25, keybit33, keybit41, keybit46

   movq  [R+200], mm1
   movq  mm7, mm0

   pxor  mm1, [temp1]         ; bit 25
   movq  mm4, mm6

   pxor  mm0, [temp2]         ; bit 19
   movq  mm3, mm5

   pxor  mm6, [temp3]         ; bit 9
   por   mm1, mm0

   pxor  mm5, [temp4]         ; bit 0
   por   mm6, mm1

   movd  mm0, ecx
   por   mm6, mm5

   movd  mm1, edx
   por   mm0, mm6

   movq  [R+152], mm7
   punpckhdq mm6, mm6

   movd  ecx, mm0
   por   mm1, mm6

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   near R12_sbox5

   cmp   edx, 0ffffffffh
   jne   near R12_sbox5

step_head:
   mov   ebp, [headstep1]
   pcmpeqd  mm0, mm0

   pxor  mm0, [keybit10]      ; This is correct 50% of the time and costs nothing
   add   ebp, 8

   mov   [headstep1], ebp
   pcmpeqd  mm1, mm1

   test  ebp, 8
   jz    near h4_18

   ; Modify key bit 10 in rounds 3 to 11

   movq  [eax+ 296], mm0
   movq  [eax+1040], mm0
   movq  [eax+1280], mm0
   movq  [eax+1776], mm0
   movq  [eax+2064], mm0
   lea   edx, [round3]
   movq  [eax+2968], mm0
   lea   ebx, [L1]

changeR2S1:
   movq  mm6, [L1+32]
   lea   esi, [PR]

   sbox_1 keybit54, keybit18, keybit33, keybit10, keybit20, keybit48

   movq  [R2+64],mm5          ; bit 8 round 2
   mov   ecx, 07dh            ; setup for partial_round

   movq  [R2+240],mm0         ; bit 30 round 2
   mov   esi, ebx

   movq  [R2+128],mm7         ; bit 16 round 2
   lea   ebx, [R2]

   movq  [R2+176],mm2         ; bit 22 round 2
   lea   edi, [L3]

   call  partial_round

   mov   esi, ebx             ; Organize for round 4 to 11
   mov   ebx, edi

   jmp   head_4

h4_18:
   test  ebp, 10h
   jz    h4_46

   ; Modify bit 18 in rounds 3 to 11

   pxor  mm1, [keybit18]
   ; stall
   movq  [eax+ 304], mm1
   movq  [eax+ 528], mm1
   movq  [eax+1432], mm1
   movq  [eax+1704], mm1
   movq  [eax+2168], mm1
   movq  [eax+2616], mm1
   lea   edx, [round3]

   movq  [eax+2872], mm1
   lea   ebx, [L1]

   movq  [eax+4032], mm1      ; Change in round 3 for partial_round
   jmp   changeR2S1

h4_46:
   test  ebp, 20h
   jz    near h4_49

   pxor  mm1, [keybit46]
   ; stall
   movq  [eax+ 136], mm1
   movq  [eax+ 576], mm1
   movq  [eax+1072], mm1
   movq  [eax+1296], mm1
   movq  [eax+2184], mm1
   movq  [eax+2504], mm1
   lea   edx, [round3]

   movq  [eax+2952], mm1
   lea   ebx, [L1]

   movq  [eax+4072], mm1      ; Change in round 3 for partial_round
   lea   esi, [PR]

   sbox_2 keybit34, keybit13, keybit04, keybit55, keybit46, keybit26

   movq  [R2+8],mm7           ; bit 1 round 2
   mov   ecx, 0bbh

   movq  [R2+96],mm1          ; bit 12 round 2
   mov   esi, ebx

   movq  [R2+216],mm3         ; bit 27 round 2
   lea   ebx, [R2]

   movq  [R2+136],mm2         ; bit 17 round 2
   lea   edi, [L3]

   call  partial_round

   mov   esi, ebx
   mov   ebx, edi

   jmp   head_4

h4_49:
   pxor  mm1, [keybit49]
   ; stall
   movq  [eax+ 368], mm1
   movq  [eax+ 824], mm1
   movq  [eax+1480], mm1
   movq  [eax+2016], mm1
   lea   edx, [round3]

   movq  [eax+2344], mm1
   lea   ebx, [L1]

   movq  [eax+2656], mm1
   lea   esi, [PR]

   movq  [eax+3080], mm1
   test  ebp, 40h

   movq  [eax+4344], mm1      ; Change in round 3 for partial_round

   ; By testing after the bits are reset all are correct for the next loop
   ; sbox_7 is redone in the partial_round call for round 2 of each tail bit
   jz    near step_tail

   sbox_7 keybit09, keybit44, keybit29, keybit07, keybit49, keybit45

   movq  [R2+248],mm7         ; bit 31 round 2
   mov   ecx, 0f5h

   movq  [R2+88],mm1          ; bit 11 round 2
   mov   esi, ebx

   movq  [R2+168],mm3         ; bit 21 round 2
   lea   ebx, [R2]

   movq  [R2+48],mm0          ; bit 6 round 2
   lea   edi, [L3]

   call  partial_round

   mov   esi, ebx
   mov   ebx, edi

   jmp   head_4

step_tail:
   mov   ecx, [tailstep]
   mov   ebp, 0

   inc   ecx
   mov   dword [headstep1], 0

   mov   [tailstep], ecx
   pcmpeqd  mm0, mm0          ; create all 1's in mm0 (this should pair as it is mmreg,mmreg)

   test  ecx, 1
   jz    near t6_11

   yield1ms

   pxor  mm0, [keybit03]
   ; stall
   movq  [eax+ 232], mm0
   movq  [eax+ 520], mm0
   movq  [eax+ 960], mm0
   movq  [eax+1456], mm0
   movq  [eax+1680], mm0
   movq  [eax+2136], mm0
   movq  [eax+2568], mm0
   movq  [eax+2888], mm0
   movq  [eax+3680], mm0      ; Round 2 for partial_round
   movq  [eax+4048], mm0      ; Round 3 for full_round
   lea   ebx, [PR]

changeR1S1R15S3:
   movq  mm6, [PR+32]
   lea   esi, [PL]

   sbox_1 keybit47, keybit11, keybit26, keybit03, keybit13, keybit41

   movq  [L1+64],mm5          ; bit 8 round 1
   lea   ecx, [R14+4096]

   movq  [L1+240],mm0         ; bit 30 round 1
   lea   ebx, [R14]

   movq  [L1+128],mm7         ; bit 16 round 1
   lea   esi, [CL]

   movq  [L1+176],mm2         ; bit 22 round 1
   lea   edi, [L13]

%define STORE_RESULT 1
fix_r15s3:
   sbox_3 keybit39, keybit11, keybit19, keybit20, keybit03, keybit48

   add   ebx, 256
   add   edi, 256

   cmp   ebx, ecx
   jb    near fix_r15s3

   lea   edx, [round2]
   lea   ebx, [L1]

   lea   esi, [PR]
   lea   edi, [R2]

   mov   ecx, 7fh
   call  partial_round        ; round 2

   mov   esi, ebx
   mov   ebx, edi

   lea   edx, [round3]
   jmp   tail_6

t6_11:
   test  ecx, 2
   jz    near t6_42

   pxor  mm0, [keybit11]
   ; stall
   movq  [eax+ 240], mm0
   movq  [eax+ 600], mm0
   movq  [eax+1032], mm0
   movq  [eax+1352], mm0
   movq  [eax+1784], mm0
   movq  [eax+2096], mm0
   movq  [eax+2464], mm0
   movq  [eax+2976], mm0
   movq  [eax+3728], mm0      ; Round 2 for partial_round
   lea   ebx, [PR]
   movq  [eax+3968], mm0      ; Round 3 for full_round
   jmp   changeR1S1R15S3

%define STORE_RESULT 0

t6_42:
   test  ecx, 4
   jz    near t6_05

   pxor  mm0, [keybit42]
   ; stall
   movq  [eax+ 496], mm0
   movq  [eax+ 744], mm0
   movq  [eax+1224], mm0
   movq  [eax+1536], mm0
   movq  [eax+1960], mm0
   movq  [eax+2328], mm0
   movq  [eax+2768], mm0
   movq  [eax+3104], mm0
   movq  [eax+3856], mm0      ; Round 2 for partial_round
   lea   ebx, [PR]
   movq  [eax+4176], mm0      ; Round 3 for full_round
   lea   esi, [PL]

   sbox_7 keybit02, keybit37, keybit22, keybit00, keybit42, keybit38

   movq  [L1+248],mm7         ; bit 31 round 1
   lea   ecx, [R14+4096]

   movq  [L1+88],mm1          ; bit 11 round 1
   lea   ebx, [R14]

   movq  [L1+168],mm3         ; bit 21 round 1
   lea   esi, [CL]

   movq  [L1+48],mm0          ; bit 6 round 1
   lea   edi, [L13]

fix_r15s8:
   ; Fix round 15 and round 14
   sbox_8 keybit02, keybit29, keybit30, keybit42, keybit52, keybit14

   pxor  mm2,[esi+208]        ; out2 ^= x54

   movq  [edi+32],mm6         ; bit 4 round 13

   movq  [edi+160],mm1        ; bit 20 round 13
   mov   esi, ebx

   movq  [edi+112],mm5        ; bit 14 round 13
   mov   ebx, edi

   movq  [edi+208],mm2        ; bit 26 round 13
   add   edi, 256

   ; Don't need the movq mm6, [ebx+32] as mm6 has the correct value from sbox_8 above
   sbox_1 keybit19, keybit40, keybit55, keybit32, keybit10, keybit13

   movq  [ebp+R12_08],mm5     ; bit 8 round 12
   mov   ebx, esi

   movq  [ebp+R12_30],mm0     ; bit 30 round 12
   add   ebx, 256

   movq  [ebp+R12_16],mm7     ; bit 16 round 12
   lea   esi, [CL]

   movq  [ebp+R12_22],mm2     ; bit 22 round 12
   add   ebp, 8

   test  ebp, 8
   jz    check_end_42

   movq  mm0, [keybit10]
   pxor  mm0, [mmNOT]
   ; stall
   movq  [keybit10], mm0

check_end_42:
   cmp   ebx, ecx
   jb    near fix_r15s8

   lea   edx, [round2]
   lea   ebx, [L1]

   lea   esi, [PR]
   lea   edi, [R2]

   mov   ecx, 0f7h
   call  partial_round        ; round 2

   mov   esi, ebx
   mov   ebx, edi

   lea   edx, [round3]
   jmp   tail_6

t6_05:
   test  ecx, 8
   jz    near t6_43

   yield10ms

   pxor  mm0, [keybit05]
   ; stall
   movq  [eax+ 176], mm0
   movq  [eax+ 544], mm0
   movq  [eax+1056], mm0
   movq  [eax+1760], mm0
   movq  [eax+2600], mm0
   xor   ebp, ebp
   movq  [eax+3736], mm0
   lea   ebx, [PR]
   movq  [eax+4008], mm0
   lea   esi, [PL]

   sbox_3 keybit53, keybit25, keybit33, keybit34, keybit17, keybit05

   movq  [L1+40], mm7         ; bit 5 round 1

   movq  [L1+232], mm3        ; bit 29 round 1
   lea   ebx, [R14]

   movq  [L1+184], mm2        ; bit 23 round 1
   lea   esi, [CL]

   movq  [L1+120], mm6        ; bit 15 round 1
   lea   edi, [L13]

fix_r15s2:
   ; Fix round 15 and round 14
   sbox_2 keybit13, keybit17, keybit40, keybit34, keybit25, keybit05

   movq  [edi+8], mm7         ; bit 1 round 13

   movq  [edi+96], mm1        ; bit 12 round 13
   mov   esi, ebx

   movq  [edi+216], mm3       ; bit 27 round 13
   mov   ebx, edi

   movq  [edi+136], mm2       ; bit 17 round 13

   call  fix_r14s1s3

   test  ebp, 80h             ; loop termination test
   jnz   fix_t605_r2

   test  ebp, 8
   jz    check_t605_46

   movq  mm0, [keybit10]
   pxor  mm0, [mmNOT]
   ; stall
   movq  [keybit10], mm0
   jmp   fix_r15s2

check_t605_46:
   test  ebp, 10h
   jnz   near fix_r15s2

   test  ebp, 20h
   jz    near fix_r15s2

   movq  mm0, [keybit46]
   pxor  mm0, [mmNOT]
   ; stall
   movq  [keybit46], mm0
   jmp   fix_r15s2

fix_t605_r2:
   lea   edx, [round2]
   lea   ebx, [L1]

   lea   esi, [PR]
   lea   edi, [R2]

   mov   ecx, 05fh
   call  partial_round

   mov   esi, ebx
   mov   ebx, edi

   lea   edx, [round3]
   jmp   tail_6

t6_43:
   test  ecx, 10h
   jz    near t6_08

   pxor  mm0, [keybit43]
   ; stall
   movq  [eax+ 344], mm0
   movq  [eax+1168], mm0
   movq  [eax+1488], mm0
   movq  [eax+2032], mm0
   movq  [eax+2360], mm0
   mov   ecx, 0deh
   movq  [eax+3016], mm0
   xor   ebp, ebp
   movq  [eax+3776], mm0      ; round 2
   lea   ebx, [PR]
   movq  [eax+4272], mm0      ; round 3
   lea   esi, [PL]

   sbox_8 keybit16, keybit43, keybit44, keybit01, keybit07, keybit28

   pxor  mm2,[esi+208]        ; out2 ^= x54

   movq  [L1+32],mm6          ; bit 4 round 1

   movq  [L1+160],mm1         ; bit 20 round 1
   lea   ebx, [R14]

   movq  [L1+112],mm5         ; bit 14 round 1
   lea   esi, [CL]

   movq  [L1+208],mm2         ; bit 26 round 1
   lea   edi, [L13]

fix_r15s7:
   ; Fix round 15 and round 14
   sbox_7 keybit43, keybit23, keybit08, keybit45, keybit28, keybit51

   movq  [edi+248],mm7        ; bit 31 round 13
   mov   esi, ebx

   movq  [edi+88],mm1         ; bit 11 round 13
   mov   ebx, edi

   movq  [edi+168],mm3        ; bit 21 round 13

   movq  [edi+48],mm0         ; bit 6 round 13

   call  fix_r14s1s3

   test  ebp, 80h             ; loop termination test
   jnz   fix_t643_r2

   test  ebp, 8
   jz    check_t643_46

   movq  mm0, [keybit10]
   pxor  mm0, [mmNOT]
   ; stall
   movq  [keybit10], mm0
   jmp   fix_r15s7

check_t643_46:
   test  ebp, 10h
   jnz   near fix_r15s7

   test  ebp, 20h
   jz    near fix_r15s7

   movq  mm0, [keybit46]
   pxor  mm0, [mmNOT]
   ; stall
   movq  [keybit46], mm0
   jmp   fix_r15s7

fix_t643_r2:
   lea   edx, [round2]
   lea   ebx, [L1]

   lea   esi, [PR]
   lea   edi, [R2]

   call  partial_round

   mov   esi, ebx
   mov   ebx, edi

   lea   edx, [round3]
   jmp   tail_6

t6_08:
   pxor  mm0, [keybit08]
   ; stall
   movq  [eax+ 504], mm0
   movq  [eax+ 752], mm0
   movq  [eax+1208], mm0
   movq  [eax+1864], mm0
   movq  [eax+2304], mm0
   test  ecx, 40h             ; tail stepping loop termination test
   movq  [eax+2728], mm0
   mov   ecx, 0f7h
   movq  [eax+3040], mm0
   mov   ebp, 0
   movq  [eax+3944], mm0      ; round 2
   lea   ebx, [PR]
   movq  [eax+4288], mm0      ; round 3
   lea   esi, [PL]
   jnz   near step_rest

   ; Rounds 1, 13, 14 and 15 will be redone so there is no need to fix them
   ; if stepping the rest of the bits

   sbox_5 keybit36, keybit31, keybit21, keybit08, keybit23, keybit52

   movq  [L1+104],mm7         ; bit 13 round 1

   movq  [L1+192],mm6         ; bit 24 round 1
   lea   ebx, [R14]

   movq  [L1+16],mm4          ; bit 2 round 1
   lea   esi, [CL]
   
   movq  [L1+56],mm5          ; bit 7 round 1
   lea   edi, [L13]

   jmp fix_r15s7

step_rest:
   mov   dword [tailstep], 0
   mov   ecx, [headstep2]

   inc   ecx
   pcmpeqd mm0, mm0

   test  ecx, 1
   mov   [headstep2], ecx

   mov   ebp, 0
   jz    rest_15

   ; Modify bit 12
   pxor  mm0, [keybit12]
   ; stall
   movq  [eax+ 248], mm0
   movq  [eax+ 696], mm0
   movq  [eax+ 952], mm0
   movq  [eax+1408], mm0
   movq  [eax+1688], mm0
   movq  [eax+2144], mm0
   movq  [eax+2512], mm0
   movq  [eax+2920], mm0
   movq  [eax+3720], mm0
   movq  [eax+4040], mm0
   movq  [eax+16600], mm0
   movq  [eax+16912], mm0
   jmp   tail_setup

rest_15:
   test  ecx, 2
   jz    rest_45
   pxor  mm0, [keybit15]
   ; stall
   movq  [eax+ 400], mm0
   movq  [eax+ 720], mm0
   movq  [eax+1264], mm0
   movq  [eax+1512], mm0
   movq  [eax+1992], mm0
   movq  [eax+2248], mm0
   movq  [eax+2784], mm0
   movq  [eax+3096], mm0
   movq  [eax+3800], mm0
   movq  [eax+16816], mm0
   movq  [eax+17144], mm0
   jmp   tail_setup

rest_45:
   test  ecx, 4
   jz    rest_50
   pxor  mm0, [keybit45]
   ; stall
   movq  [eax+ 424], mm0
   movq  [eax+ 736], mm0
   movq  [eax+1160], mm0
   movq  [eax+1856], mm0
   movq  [eax+2296], mm0
   movq  [eax+3176], mm0
   movq  [eax+3912], mm0
   movq  [eax+4224], mm0
   movq  [eax+16744], mm0
   movq  [eax+17208], mm0
   jmp   tail_setup

rest_50:
   ; Modify bit 50
   test  ecx, 10h             ; outermost loop termination
   jnz   finish
   pxor  mm0, [keybit50]
   ; stall
   movq  [eax+ 872], mm0
   movq  [eax+1216], mm0
   movq  [eax+1656], mm0
   movq  [eax+1904], mm0
   movq  [eax+2416], mm0
   movq  [eax+2664], mm0
   movq  [eax+3144], mm0
   movq  [eax+3928], mm0
   movq  [eax+4216], mm0
   movq  [eax+16800], mm0
   movq  [eax+17160], mm0
   jmp   tail_setup

finish:
   emms

   ; Uses the eax value pushed after the call to _malloc
   relmem

   pop   edx                 ; remove argument from the stack
   xor   eax, eax
   xor   edx, edx
   pop   edi
   pop   esi
   pop   ecx
   pop   ebx
   pop   ebp
   retn

R12_sbox5:
   movq  [R+72], mm4
   lea   esi, [R14+ebp]

   movq  [R], mm3
   lea   ebx, [L13+ebp]

   ; decrypt from round 13
   sbox_5 keybit08, keybit30, keybit52, keybit35, keybit50, keybit51

   movq  [temp2],mm7          ; store out2

   movq  [temp3],mm6          ; store out3
   lea   esi, [R]

   movq  [temp4],mm4          ; store out4
   lea   ebx, [L]

   movq  [temp1],mm5          ; store out1

   ; encrypt from round 11
   sbox_5 keybit35, keybit02, keybit51, keybit07, keybit22, keybit23

   movq  [R+104], mm7
   movq  mm1, mm6

   pxor  mm7, [temp2]         ; bit 13
   movq  mm2, mm4

   pxor  mm1, [temp3]         ; bit 24
   movq  mm3, mm5

   pxor  mm4, [temp4]         ; bit 2
   por   mm7, mm1

   pxor  mm5, [temp1]         ; bit 7
   por   mm7, mm4

   movd  mm0, ecx
   por   mm7, mm5

   movd  mm1, edx
   por   mm0, mm7

   movq  [R+192], mm6
   punpckhdq mm7, mm7

   movd  ecx, mm0
   por   mm1, mm7

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   R12_sbox6

   cmp   edx, 0ffffffffh
   je    near step_head

R12_sbox6:
   movq  [R+16], mm2
   lea   esi, [R14+ebp]

   movq  [R+56], mm3
   lea   ebx, [L13+ebp]

   ; decrypt from round 13
   sbox_6 keybit45, keybit01, keybit23, keybit36, keybit07, keybit02

   movq  [temp4],mm4          ; store out4

   movq  [temp1],mm0          ; store out1
   lea   esi, [R]

   movq  [temp2],mm1          ; store out2
   lea   ebx, [L]

   movq  [temp3],mm2          ; store out3

   ; encrypt from round 11
   sbox_6 keybit44, keybit28, keybit50, keybit08, keybit38, keybit29

   movq  [R+144], mm4
   movq  mm7, mm0

   pxor  mm4, [temp4]         ; bit 18
   movq  mm6, mm1

   pxor  mm0, [temp1]         ; bit 3
   movq  mm5, mm2

   pxor  mm1, [temp2]         ; bit 28
   por   mm4, mm0

   pxor  mm2, [temp3]         ; bit 10
   por   mm4, mm1

   movd  mm0, ecx
   por   mm4, mm2

   movd  mm1, edx
   por   mm0, mm4

   movq  [R+24], mm7
   punpckhdq mm4, mm4

   movd  ecx, mm0
   por   mm1, mm4

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   R12_sbox7

   cmp   edx, 0ffffffffh
   je    near step_head

R12_sbox7:
   movq  [R+224], mm6
   lea   esi, [R14+ebp]

   movq  [R+80], mm5
   lea   ebx, [L13+ebp]

   ; decrypt from round 13
   sbox_7 keybit29, keybit09, keybit49, keybit31, keybit14, keybit37

   movq  [temp1],mm7          ; store out1

   movq  [temp2],mm1          ; store out2
   lea   esi, [R]

   movq  [temp3],mm3          ; store out3
   lea   ebx, [L]

   movq  [temp4],mm0          ; store out4

   ; encrypt from round 11
   sbox_7 keybit01, keybit36, keybit21, keybit30, keybit45, keybit09

   movq  [R+248], mm7
   movq  mm6, mm1

   pxor  mm7, [temp1]         ; bit 31
   movq  mm5, mm3

   pxor  mm1, [temp2]         ; bit 11
   movq  mm4, mm0

   pxor  mm3, [temp3]         ; bit 21
   por   mm7, mm1

   pxor  mm0, [temp4]         ; bit 6
   por   mm7, mm3

   movd  mm2, ecx
   por   mm7, mm0

   movd  mm1, edx
   por   mm2, mm7

   movq  [R+88], mm6
   punpckhdq mm7, mm7

   movd  ecx, mm2
   por   mm1, mm7

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   R12_sbox8

   cmp   edx, 0ffffffffh
   je    near step_head

R12_sbox8:
   movq  [R+168], mm5
   lea   esi, [R14+ebp]

   movq  [R+48], mm4
   lea   ebx, [L13+ebp]

   ; decrypt from round 13
   sbox_8 keybit43, keybit15, keybit16, keybit28, keybit38, keybit00

   pxor  mm2, [esi+208]       ; out2 ^= x54

   movq  [temp1],mm6          ; store out1

   movq  [temp4],mm1          ; store out4
   lea   esi, [R]

   movq  [temp3],mm5          ; store out3
   lea   ebx, [L]

   movq  [temp2],mm2          ; store out2

   ; encrypt from round 11
   sbox_8 keybit15, keybit42, keybit43, keybit00, keybit37, keybit31

   pxor  mm2, [esi+208]       ; out2 ^= x54

   movq  [R+32], mm6
   movq  mm7, mm1

   pxor  mm6, [temp1]         ; bit 4
   movq  mm4, mm5

   pxor  mm1, [temp4]         ; bit 20
   movq  mm3, mm2

   pxor  mm5, [temp3]         ; bit 14
   por   mm6, mm1

   pxor  mm2, [temp2]         ; bit 26
   por   mm6, mm5

   movd  mm0, ecx
   por   mm6, mm2

   movd  mm1, edx
   por   mm0, mm6

   movq  [R+160], mm7
   punpckhdq mm6, mm6

   movd  ecx, mm0
   por   mm1, mm6

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   R13_sbox1

   cmp   edx, 0ffffffffh
   je    near step_head

R13_sbox1:
   movq  [R+112], mm4
   lea   esi, [L]

   movq  [R+208], mm3
   lea   ebx, [R]

   movq  mm6, [R+32]
   lea   edi, [L13+ebp]

   ; encrypt from round 12
   sbox_1 keybit05, keybit26, keybit41, keybit18, keybit53, keybit24

   pxor  mm5, [edi+64]        ; bit 8

   pxor  mm0, [edi+240]       ; bit 30

   pxor  mm7, [edi+128]       ; bit 16
   por   mm5, mm0

   pxor  mm2, [edi+176]       ; bit 22
   por   mm5, mm7

   movd  mm0, ecx
   por   mm5, mm2

   movd  mm1, edx
   por   mm0, mm5

   punpckhdq mm5, mm5

   movd  ecx, mm0
   por   mm1, mm5

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   R13_sbox2

   cmp   edx, 0ffffffffh
   je    near step_head

R13_sbox2:
   ; encrypt from round 12
   sbox_2 keybit10, keybit46, keybit12, keybit06, keybit54, keybit34

   pxor  mm7, [edi+8]         ; bit 1

   pxor  mm1, [edi+96]        ; bit 12

   pxor  mm3, [edi+216]       ; bit 27
   por   mm7, mm1

   pxor  mm2, [edi+136]       ; bit 17
   por   mm7, mm3

   movd  mm0, ecx
   por   mm7, mm2

   movd  mm1, edx
   por   mm0, mm7

   punpckhdq mm7, mm7

   movd  ecx, mm0
   por   mm1, mm7

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   R13_sbox3

   cmp   edx, 0ffffffffh
   je    near step_head

R13_sbox3:
   ; encrypt from round 12
   sbox_3 keybit11, keybit40, keybit48, keybit17, keybit32, keybit20

   pxor  mm7, [edi+40]        ; bit 5

   pxor  mm3, [edi+232]       ; bit 29

   pxor  mm2, [edi+184]       ; bit 23
   por   mm7, mm3

   pxor  mm6, [edi+120]       ; bit 15
   por   mm7, mm2

   movd  mm0, ecx
   por   mm7, mm6

   movd  mm1, edx
   por   mm0, mm7

   punpckhdq mm7, mm7

   movd  ecx, mm0
   por   mm1, mm7

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   R13_sbox4

   cmp   edx, 0ffffffffh
   je    near step_head

R13_sbox4:
   ; encrypt from round 12
   sbox_4 keybit19, keybit13, keybit39, keybit47, keybit55, keybit03

   pxor  mm1, [edi+200]       ; bit 25

   pxor  mm0, [edi+152]       ; bit 19

   pxor  mm6, [edi+72]        ; bit 9
   por   mm1, mm0

   pxor  mm5, [edi]           ; bit 0
   por   mm1, mm6

   movd  mm0, ecx
   por   mm5, mm1

   movd  mm1, edx
   por   mm0, mm5

   punpckhdq mm5, mm5

   movd  ecx, mm0
   por   mm1, mm5

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   R13_sbox5

   cmp   edx, 0ffffffffh
   je    near step_head

R13_sbox5:
   ; encrypt from round 12
   sbox_5 keybit49, keybit16, keybit38, keybit21, keybit36, keybit37

   pxor  mm7, [edi+104]       ; bit 13

   pxor  mm6, [edi+192]       ; bit 24

   pxor  mm4, [edi+16]        ; bit 2
   por   mm7, mm6

   pxor  mm5, [edi+56]        ; bit 7
   por   mm7, mm4

   movd  mm0, ecx
   por   mm7, mm5

   movd  mm1, edx
   por   mm0, mm7

   punpckhdq mm7, mm7

   movd  ecx, mm0
   por   mm1, mm7

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   R13_sbox6

   cmp   edx, 0ffffffffh
   je    near step_head

R13_sbox6:
   ; encrypt from round 12
   sbox_6 keybit31, keybit42, keybit09, keybit22, keybit52, keybit43

   pxor  mm4, [edi+144]       ; bit 18

   pxor  mm0, [edi+24]        ; bit 3

   pxor  mm1, [edi+224]       ; bit 28
   por   mm4, mm0

   pxor  mm2, [edi+80]        ; bit 10
   por   mm4, mm1

   movd  mm0, ecx
   por   mm4, mm2

   movd  mm1, edx
   por   mm0, mm4

   punpckhdq mm4, mm4

   movd  ecx, mm0
   por   mm1, mm4

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   R13_sbox7

   cmp   edx, 0ffffffffh
   je    near step_head

R13_sbox7:
   ; encrypt from round 12
   sbox_7 keybit15, keybit50, keybit35, keybit44, keybit00, keybit23

   pxor  mm7, [edi+248]       ; bit 31

   pxor  mm1, [edi+88]        ; bit 11

   pxor  mm3, [edi+168]       ; bit 21
   por   mm7, mm1

   pxor  mm0, [edi+48]        ; bit 6
   por   mm7, mm3

   movd  mm2, ecx
   por   mm7, mm0

   movd  mm1, edx
   por   mm2, mm7

   punpckhdq mm7, mm7

   movd  ecx, mm2
   por   mm1, mm7

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   R13_sbox8

   cmp   edx, 0ffffffffh
   je    near step_head

R13_sbox8:
   ; encrypt from round 12
   sbox_8 keybit29, keybit01, keybit02, keybit14, keybit51, keybit45

   pxor  mm2, [esi+208]       ; out2 ^= x54

   pxor  mm6, [edi+32]        ; bit 8

   pxor  mm1, [edi+160]       ; bit 30

   pxor  mm5, [edi+112]       ; bit 14
   por   mm6, mm1

   pxor  mm2, [edi+208]       ; bit 26
   por   mm6, mm5

   movd  mm0, ecx
   por   mm6, mm2

   movd  mm1, edx
   por   mm0, mm6

   punpckhdq mm6, mm6

   movd  ecx, mm0
   por   mm1, mm6

   cmp   ecx, 0ffffffffh

   movd  edx, mm1
   jne   key_found

   cmp   edx, 0ffffffffh
   je    near step_head

key_found:
   ; Flip the bits as the caller expects THE key to bit the 1 bit in a field of 0s
   not   ecx
   not   edx

   mov   ebp, [esp+36]
   ; Collect the eax value pushed after the call to _malloc
   ; and save our result
   pop   eax
   push  edx

   push  ecx
   push  eax

   movq  mm0, [keybit00]
   movq  mm1, [keybit01]
   movq  [ebp], mm0
   movq  mm2, [keybit02]
   movq  [ebp+8], mm1
   movq  mm0, [keybit03]
   movq  [ebp+16], mm2
   movq  mm1, [keybit04]
   movq  [ebp+24], mm0
   movq  mm2, [keybit05]
   movq  [ebp+32], mm1
   movq  mm0, [keybit06]
   movq  [ebp+40], mm2
   movq  mm1, [keybit07]
   movq  [ebp+48], mm0
   movq  mm2, [keybit08]
   movq  [ebp+56], mm1
   movq  mm0, [keybit09]
   movq  [ebp+64], mm2
   movq  mm1, [keybit10]
   movq  [ebp+72], mm0
   movq  mm2, [keybit11]
   movq  [ebp+80], mm1
   movq  mm0, [keybit12]
   movq  [ebp+88], mm2
   movq  mm1, [keybit13]
   movq  [ebp+96], mm0
   movq  mm2, [keybit14]
   movq  [ebp+104], mm1
   movq  mm0, [keybit15]
   movq  [ebp+112], mm2
   movq  mm1, [keybit16]
   movq  [ebp+120], mm0
   movq  mm2, [keybit17]
   movq  [ebp+128], mm1
   movq  mm0, [keybit18]
   movq  [ebp+136], mm2
   movq  mm1, [keybit19]
   movq  [ebp+144], mm0
   movq  mm2, [keybit20]
   movq  [ebp+152], mm1
   movq  mm0, [keybit21]
   movq  [ebp+160], mm2
   movq  mm1, [keybit22]
   movq  [ebp+168], mm0
   movq  mm2, [keybit23]
   movq  [ebp+176], mm1
   movq  mm0, [keybit24]
   movq  [ebp+184], mm2
   movq  mm1, [keybit25]
   movq  [ebp+192], mm0
   movq  mm2, [keybit26]
   movq  [ebp+200], mm1
   movq  mm0, [keybit27]
   movq  [ebp+208], mm2
   movq  mm1, [keybit28]
   movq  [ebp+216], mm0
   movq  mm2, [keybit29]
   movq  [ebp+224], mm1
   movq  mm0, [keybit30]
   movq  [ebp+232], mm2
   movq  mm1, [keybit31]
   movq  [ebp+240], mm0
   movq  mm2, [keybit32]
   movq  [ebp+248], mm1
   movq  mm0, [keybit33]
   movq  [ebp+256], mm2
   movq  mm1, [keybit34]
   movq  [ebp+264], mm0
   movq  mm2, [keybit35]
   movq  [ebp+272], mm1
   movq  mm0, [keybit36]
   movq  [ebp+280], mm2
   movq  mm1, [keybit37]
   movq  [ebp+288], mm0
   movq  mm2, [keybit38]
   movq  [ebp+296], mm1
   movq  mm0, [keybit39]
   movq  [ebp+304], mm2
   movq  mm1, [keybit40]
   movq  [ebp+312], mm0
   movq  mm2, [keybit41]
   movq  [ebp+320], mm1
   movq  mm0, [keybit42]
   movq  [ebp+328], mm2
   movq  mm1, [keybit43]
   movq  [ebp+336], mm0
   movq  mm2, [keybit44]
   movq  [ebp+344], mm1
   movq  mm0, [keybit45]
   movq  [ebp+352], mm2
   movq  mm1, [keybit46]
   movq  [ebp+360], mm0
   movq  mm2, [keybit47]
   movq  [ebp+368], mm1
   movq  mm0, [keybit48]
   movq  [ebp+376], mm2
   movq  mm1, [keybit49]
   movq  [ebp+384], mm0
   movq  mm2, [keybit50]
   movq  [ebp+392], mm1
   movq  mm0, [keybit51]
   movq  [ebp+400], mm2
   movq  mm1, [keybit52]
   movq  [ebp+408], mm0
   movq  mm2, [keybit53]
   movq  [ebp+416], mm1
   movq  mm0, [keybit54]
   movq  [ebp+424], mm2
   movq  mm1, [keybit55]
   movq  [ebp+432], mm0
   movq  [ebp+440], mm1

   emms

   call  _free

   pop   edx                 ; remove argument from the stack
   pop   eax                 ; retrieve the result

   pop   edx
   pop   edi

   pop   esi
   pop   ecx

   pop   ebx
   pop   ebp

   retn
