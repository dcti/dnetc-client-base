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
             

