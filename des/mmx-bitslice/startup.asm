; Startup code for DES bitslice driver for MMX
; Bruce Ford
;
; $Log: startup.asm,v $
; Revision 1.1  1999/01/12 03:25:29  fordbr
; DES bitslice driver code for MMX
;  73 clocks per key on Pentium MMX
;  85 clocks per key on AMD K6-2
;  90 clocks per key on Intel PII
; 118 clocks per key on AMD K6
;
;---------------------------------------------
;
; round 4: 128
; round 5: 512
; round 6: 896
; round 7: 1280
; round 8: 1664
; round 9: 2048
; round 10: 2432
; round 11: 2816
; round 1: 3200
; round 2: 3584
; round 3: 3968
; round 16: 16512
; round 15: 16896

%define L         eax+4352
%define R         eax+4608
%define L1        eax+4864
%define R2        eax+5120
%define L3        eax+5376
%define R12_05    eax+5632
%define R12_08    eax+5760
%define R12_15    eax+5888
%define R12_16    eax+6016
%define R12_22    eax+6144
%define R12_23    eax+6272
%define R12_29    eax+6400
%define R12_30    eax+6528
%define L13       eax+6656
%define R14       eax+10752
%define PR        eax+14848
%define PL        eax+15104
%define CL        eax+15360
%define CR        eax+15616
%define temp1     eax+15872
%define temp2     eax+15880
%define temp3     eax+15888
%define temp4     eax+15896
%define headstep1 eax+15904
%define tailstep  eax+15908
%define headstep2 eax+15912

; There is a gap of 596 bytes between the end of the frequently used data
; and the round 16 key bits.  This avoids clashes with the sbox temporaries
; up to eax+128.  The keybit area for rounds 16 and 15 requires 768 bytes

%define round4       eax+128
%define round11      eax+2816
%define round1       eax+3200
%define round2       eax+3584
%define round3       eax+3968
%define round16      eax+16512

%define rnd16_kbit40 eax+16512
%define rnd16_kbit19 eax+16528
%define rnd16_kbit06 eax+16544
%define rnd16_kbit34 eax+16552
%define rnd16_kbit32 eax+16592
%define rnd16_kbit46 eax+16608
%define rnd16_kbit18 eax+16616
%define rnd16_kbit10 eax+16640
%define rnd16_kbit55 eax+16648
%define rnd16_kbit54 eax+16656
%define rnd16_kbit25 eax+16680
%define rnd16_kbit13 eax+16696
%define rnd16_kbit49 eax+16872
%define rnd15_kbit46 eax+16920
%define rnd15_kbit05 eax+16984
%define rnd15_kbit10 eax+17056
%define rnd15_kbit18 eax+17064
%define rnd15_kbit49 eax+17112

%define mem_needed 17288      ; extra eight bytes for quadword alignment

; Which keybits are where
;
;  0: 384, 808,1120,1544,2392,2680,3512,3816,4296,16880,17136
;  1: 336, 880,1128,1608,1920,2400,2712,3152,3560,4240,16728
;  2: 432, 728,1552,1872,2288,2744,3488,4160,16776,17232
;  3: 232, 520, 960,1456,1680,2136,2568,2888,3224,3680,4048,17024
;  4: 144,1048,1320,1712,2232,2488,2944,3344,3648,4144,16520
;  5: 176, 544,1056,1760,2600,3336,3736,4008,16984
;  6: 224, 592,1000,1288,1728,2048,2544,2904,3256,3744,16544,17080
;  7: 328, 864,1176,1616,1952,2624,3120,3568,3896,16752,17104
;  8: 504, 752,1208,1864,2304,2728,3040,3416,3944,4288,17200
;  9: 376,1256,1600,2040,2256,2800,3048,3464,3872,4312,16848,17120
; 10: 296,1040,1280,1776,2064,2968,3608,16640,17056
; 11: 240, 600,1032,1352,1784,2096,2464,2976,3208,3728,3968,17000
; 12: 248, 696, 952,1408,1688,2144,2512,2920,3720,4040,16600,16912
; 13: 152,1064,1808,2112,2608,2832,3232,3640,4096,16696,16944
; 14: 456, 768,1192,1504,1928,2336,2776,3064,3440,3952,4200,16720,17272
; 15: 400, 720,1264,1512,1992,2248,2784,3096,3800,16816,17144
; 16: 320, 816,1112,1936,2424,2672,3128,3536,3848,16736,17176
; 17: 208, 616, 904,1344,1840,2160,2520,2952,3328,4064,16672,16952
; 18: 304, 528,1432,1704,2168,2616,2872,3592,4032,16616,17064
; 19: 168, 560, 928,1440,2072,2984,3288,4120,16528,17008
; 20: 608, 976,1384,1672,2192,2432,2928,3384,3616,4128,16560,17016
; 21: 712,1248,1560,2000,2312,3008,3408,3824,4280,16888,17168
; 22: 448, 888,1136,1592,2376,2688,3112,3504,4328,16760,17088
; 23: 472, 760,1640,1984,2320,2640,3184,3424,3920,4256,16792,17192
; 24: 680,1424,1664,2224,2448,3360,3712,3992,16568,16928
; 25: 128, 624, 984,1416,1736,2088,2480,2848,3304,4112,16680,16976
; 26: 200, 632,1080,1336,1792,2528,2896,3216,3672,4104,16624,17072
; 27: 256, 536,1448,2056,2496,2992,3248,3768,4024,16632,16936
; 28: 360, 840,1152,1576,1888,2384,2720,3160,3576,3792,4336,16784,17216
; 29: 784,1104,1648,1896,2632,3168,3448,3888,4184,16704,17240
; 30: 704,1200,1496,2368,2808,3056,3480,3808,4232,16808,17248
; 31: 352, 776,1472,1968,2792,3136,3400,3840,4264,16840
; 32: 192, 688, 912,1816,2120,2552,3000,3368,3688,3976,16592
; 33: 280, 552, 944,1312,1824,2176,2456,3312,3600,16688,16896
; 34: 288, 992,1360,1768,2576,2816,3320,3632,4000,16552,16968
; 35: 440,1096,1632,1944,2272,2696,3472,3960,4208,16832
; 36: 488, 832,1272,1520,1976,2280,2760,3072,3392,3832,16856
; 37: 416, 856,1144,2024,2704,3024,3496,3864,4304,16864,17152
; 38: 408, 848,1184,1624,1912,2352,2648,3528,3784,4320,17128
; 39: 272, 512,1008,1368,1800,2200,2472,2864,3280,3752,16992
; 40: 264, 584,1016,1464,1720,2208,2912,3376,3696,4056,16512,16960
; 41: 184, 640, 920,1832,2152,2440,2880,3240,3704,4152,16584,17048
; 42: 496, 744,1224,1536,1960,2328,2768,3104,3520,3856,4176,17256
; 43: 344,1168,1488,2032,2360,3016,3544,3776,4272,17184
; 44: 392,1088,1584,1880,2408,2752,3192,3552,3880,4192,16768,17096
; 45: 424, 736,1160,1856,2296,3176,3912,4224,16744,17208
; 46: 136, 576,1072,1296,2184,2504,2936,3664,4072,16608,16920
; 47: 664, 936,1328,1696,2104,2560,2840,3200,3760,3984,16576,17040
; 48: 160, 672,1376,1744,2216,2960,3272,3624,4016,16664,17032
; 49: 368, 824,1480,2016,2344,2656,3080,3904,4344,16872,17112
; 50: 872,1216,1656,1904,2416,2664,3144,3928,4216,16800,17160
; 51: 464, 800,1240,1528,2264,3088,3456,3936,4248,16712,17224
; 52: 480, 792,1232,1568,2008,2240,2736,3032,3432,4168,16824,17264
; 53: 656, 896,1392,1752,2584,2856,3296,4136,16536
; 54: 216, 648, 968,1400,1848,2080,2592,3264,3584,4080,16656,16904
; 55: 312, 568,1024,1304,2128,2536,2824,3352,3656,4088,16648

%define keybit00     eax+384
%define keybit01     eax+336
%define keybit02     eax+432
%define keybit03     eax+232
%define keybit04     eax+144
%define keybit05     eax+176
%define keybit06     eax+224
%define keybit07     eax+328
%define keybit08     eax+504
%define keybit09     eax+376
%define keybit10     eax+296
%define keybit11     eax+240
%define keybit12     eax+248
%define keybit13     eax+152
%define keybit14     eax+456
%define keybit15     eax+400
%define keybit16     eax+320
%define keybit17     eax+208
%define keybit18     eax+304
%define keybit19     eax+168
%define keybit20     eax+608
%define keybit21     eax+712
%define keybit22     eax+448
%define keybit23     eax+472
%define keybit24     eax+680
%define keybit25     eax+128
%define keybit26     eax+200
%define keybit27     eax+256
%define keybit28     eax+360
%define keybit29     eax+784
%define keybit30     eax+704
%define keybit31     eax+352
%define keybit32     eax+192
%define keybit33     eax+280
%define keybit34     eax+288
%define keybit35     eax+440
%define keybit36     eax+488
%define keybit37     eax+416
%define keybit38     eax+408
%define keybit39     eax+272
%define keybit40     eax+264
%define keybit41     eax+184
%define keybit42     eax+496
%define keybit43     eax+344
%define keybit44     eax+392
%define keybit45     eax+424
%define keybit46     eax+136
%define keybit47     eax+664
%define keybit48     eax+160
%define keybit49     eax+368
%define keybit50     eax+872
%define keybit51     eax+464
%define keybit52     eax+480
%define keybit53     eax+656
%define keybit54     eax+216
%define keybit55     eax+312

startup:
   ; Key bit 0
	movq  mm0, [ebp+0]
	movq  [eax+ 384], mm0
	movq  [eax+ 808], mm0
	movq  [eax+1120], mm0
	movq  [eax+1544], mm0
	movq  [eax+2392], mm0
	movq  [eax+2680], mm0
	movq  [eax+3512], mm0
	movq  [eax+3816], mm0
	movq  [eax+4296], mm0
	movq  [eax+16880], mm0
	movq  [eax+17136], mm0

	; Key bit 1
	movq  mm0, [ebp+8]
	movq  [eax+ 336], mm0
	movq  [eax+ 880], mm0
   movq  [eax+1128], mm0
   movq  [eax+1608], mm0
   movq  [eax+1920], mm0
   movq  [eax+2400], mm0
   movq  [eax+2712], mm0
   movq  [eax+3152], mm0
   movq  [eax+3560], mm0
   movq  [eax+4240], mm0
   movq  [eax+16728], mm0

   ; Key bit 2
   movq  mm0, [ebp+16]
   movq  [eax+ 432], mm0
   movq  [eax+ 728], mm0
   movq  [eax+1552], mm0
   movq  [eax+1872], mm0
	movq  [eax+2288], mm0
   movq  [eax+2744], mm0
   movq  [eax+3488], mm0
   movq  [eax+4160], mm0
   movq  [eax+16776], mm0
   movq  [eax+17232], mm0

   ; Key bit 3
   movq  mm0, [ebp+24]
	movq  [eax+ 232], mm0
   movq  [eax+ 520], mm0
   movq  [eax+ 960], mm0
   movq  [eax+1456], mm0
   movq  [eax+1680], mm0
   movq  [eax+2136], mm0
   movq  [eax+2568], mm0
   movq  [eax+2888], mm0
   movq  [eax+3224], mm0
   movq  [eax+3680], mm0
   movq  [eax+4048], mm0
   movq  [eax+17024], mm0

   ; Key bit 4
   movq  mm0, [ebp+32]
   movq  [eax+ 144], mm0
   movq  [eax+1048], mm0
	movq  [eax+1320], mm0
   movq  [eax+1712], mm0
   movq  [eax+2232], mm0
   movq  [eax+2488], mm0
   movq  [eax+2944], mm0
   movq  [eax+3344], mm0
   movq  [eax+3648], mm0
   movq  [eax+4144], mm0
   movq  [eax+16520], mm0

   ; Key bit 5
   movq  mm0, [ebp+40]
   movq  [eax+ 176], mm0
   movq  [eax+ 544], mm0
   movq  [eax+1056], mm0
   movq  [eax+1760], mm0
   movq  [eax+2600], mm0
   movq  [eax+3336], mm0
   movq  [eax+3736], mm0
   movq  [eax+4008], mm0
   movq  [eax+16984], mm0

   ; Key bit 6
   movq  mm0, [ebp+48]
   movq  [eax+ 224], mm0
   movq  [eax+ 592], mm0
	movq  [eax+1000], mm0
   movq  [eax+1288], mm0
   movq  [eax+1728], mm0
   movq  [eax+2048], mm0
   movq  [eax+2544], mm0
   movq  [eax+2904], mm0
   movq  [eax+3256], mm0
   movq  [eax+3744], mm0
   movq  [eax+16544], mm0
	movq  [eax+17080], mm0

   ; Key bit 7
   movq  mm0, [ebp+56]
   movq  [eax+ 328], mm0
   movq  [eax+ 864], mm0
   movq  [eax+1176], mm0
   movq  [eax+1616], mm0
   movq  [eax+1952], mm0
   movq  [eax+2624], mm0
   movq  [eax+3120], mm0
   movq  [eax+3568], mm0
   movq  [eax+3896], mm0
   movq  [eax+16752], mm0
   movq  [eax+17104], mm0

   ; Key bit 8
	movq  mm0, [ebp+64]
   movq  [eax+ 504], mm0
   movq  [eax+ 752], mm0
   movq  [eax+1208], mm0
   movq  [eax+1864], mm0
   movq  [eax+2304], mm0
   movq  [eax+2728], mm0
   movq  [eax+3040], mm0
   movq  [eax+3416], mm0
	movq  [eax+3944], mm0
   movq  [eax+4288], mm0
   movq  [eax+17200], mm0

   ; Key bit 9
   movq  mm0, [ebp+72]
   movq  [eax+ 376], mm0
   movq  [eax+1256], mm0
   movq  [eax+1600], mm0
   movq  [eax+2040], mm0
   movq  [eax+2256], mm0
   movq  [eax+2800], mm0
   movq  [eax+3048], mm0
   movq  [eax+3464], mm0
   movq  [eax+3872], mm0
   movq  [eax+4312], mm0
   movq  [eax+16848], mm0
	movq  [eax+17120], mm0

   ; Key bit 10
   movq  mm0, [ebp+80]
   movq  [eax+ 296], mm0
   movq  [eax+1040], mm0
   movq  [eax+1280], mm0
   movq  [eax+1776], mm0
   movq  [eax+2064], mm0
	movq  [eax+2968], mm0
   movq  [eax+3608], mm0
   movq  [eax+16640], mm0
   movq  [eax+17056], mm0

   ; Key bit 11
   movq  mm0, [ebp+88]
   movq  [eax+ 240], mm0
   movq  [eax+ 600], mm0
   movq  [eax+1032], mm0
   movq  [eax+1352], mm0
   movq  [eax+1784], mm0
   movq  [eax+2096], mm0
   movq  [eax+2464], mm0
   movq  [eax+2976], mm0
   movq  [eax+3208], mm0
   movq  [eax+3728], mm0
	movq  [eax+3968], mm0
   movq  [eax+17000], mm0

   ; Key bit 12
   movq  mm0, [ebp+96]
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

   ; Key bit 13
   movq  mm0, [ebp+104]
   movq  [eax+ 152], mm0
   movq  [eax+1064], mm0
   movq  [eax+1808], mm0
   movq  [eax+2112], mm0
   movq  [eax+2608], mm0
   movq  [eax+2832], mm0
	movq  [eax+3232], mm0
   movq  [eax+3640], mm0
   movq  [eax+4096], mm0
   movq  [eax+16696], mm0
   movq  [eax+16944], mm0

   ; Key bit 14
   movq  mm0, [ebp+112]
   movq  [eax+ 456], mm0
	movq  [eax+ 768], mm0
   movq  [eax+1192], mm0
   movq  [eax+1504], mm0
   movq  [eax+1928], mm0
   movq  [eax+2336], mm0
   movq  [eax+2776], mm0
   movq  [eax+3064], mm0
   movq  [eax+3440], mm0
   movq  [eax+3952], mm0
   movq  [eax+4200], mm0
   movq  [eax+16720], mm0
   movq  [eax+17272], mm0

   ; Key bit 15
   movq  mm0, [ebp+120]
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

   ; Key bit 16
   movq  mm0, [ebp+128]
   movq  [eax+ 320], mm0
   movq  [eax+ 816], mm0
   movq  [eax+1112], mm0
   movq  [eax+1936], mm0
   movq  [eax+2424], mm0
   movq  [eax+2672], mm0
   movq  [eax+3128], mm0
   movq  [eax+3536], mm0
   movq  [eax+3848], mm0
   movq  [eax+16736], mm0
   movq  [eax+17176], mm0

   ; Key bit 17
   movq  mm0, [ebp+136]
	movq  [eax+ 208], mm0
   movq  [eax+ 616], mm0
   movq  [eax+ 904], mm0
   movq  [eax+1344], mm0
   movq  [eax+1840], mm0
   movq  [eax+2160], mm0
   movq  [eax+2520], mm0
   movq  [eax+2952], mm0
   movq  [eax+3328], mm0
	movq  [eax+4064], mm0
   movq  [eax+16672], mm0
   movq  [eax+16952], mm0

   ; Key bit 18
   movq  mm0, [ebp+144]
   movq  [eax+ 304], mm0
   movq  [eax+ 528], mm0
   movq  [eax+1432], mm0
   movq  [eax+1704], mm0
   movq  [eax+2168], mm0
   movq  [eax+2616], mm0
   movq  [eax+2872], mm0
   movq  [eax+3592], mm0
   movq  [eax+4032], mm0
   movq  [eax+16616], mm0
   movq  [eax+17064], mm0

   ; Key bit 19
   movq  mm0, [ebp+152]
   movq  [eax+ 168], mm0
   movq  [eax+ 560], mm0
   movq  [eax+ 928], mm0
   movq  [eax+1440], mm0
   movq  [eax+2072], mm0
   movq  [eax+2984], mm0
	movq  [eax+3288], mm0
   movq  [eax+4120], mm0
   movq  [eax+16528], mm0
   movq  [eax+17008], mm0

   ; Key bit 20
   movq  mm0, [ebp+160]
   movq  [eax+ 608], mm0
   movq  [eax+ 976], mm0
   movq  [eax+1384], mm0
   movq  [eax+1672], mm0
   movq  [eax+2192], mm0
   movq  [eax+2432], mm0
   movq  [eax+2928], mm0
   movq  [eax+3384], mm0
   movq  [eax+3616], mm0
   movq  [eax+4128], mm0
	movq  [eax+16560], mm0
   movq  [eax+17016], mm0

   ; Key bit 21
   movq  mm0, [ebp+168]
   movq  [eax+ 712], mm0
   movq  [eax+1248], mm0
   movq  [eax+1560], mm0
   movq  [eax+2000], mm0
	movq  [eax+2312], mm0
   movq  [eax+3008], mm0
   movq  [eax+3408], mm0
   movq  [eax+3824], mm0
   movq  [eax+4280], mm0
   movq  [eax+16888], mm0
   movq  [eax+17168], mm0

   ; Key bit 22
   movq  mm0, [ebp+176]
   movq  [eax+ 448], mm0
   movq  [eax+ 888], mm0
   movq  [eax+1136], mm0
   movq  [eax+1592], mm0
   movq  [eax+2376], mm0
   movq  [eax+2688], mm0
   movq  [eax+3112], mm0
	movq  [eax+3504], mm0
   movq  [eax+4328], mm0
   movq  [eax+16760], mm0
   movq  [eax+17088], mm0

   ; Key bit 23
   movq  mm0, [ebp+184]
   movq  [eax+ 472], mm0
   movq  [eax+ 760], mm0
	movq  [eax+1640], mm0
   movq  [eax+1984], mm0
   movq  [eax+2320], mm0
   movq  [eax+2640], mm0
   movq  [eax+3184], mm0
   movq  [eax+3424], mm0
   movq  [eax+3920], mm0
   movq  [eax+4256], mm0
   movq  [eax+16792], mm0
   movq  [eax+17192], mm0

   ; Key bit 24
   movq  mm0, [ebp+192]
   movq  [eax+ 680], mm0
   movq  [eax+1424], mm0
   movq  [eax+1664], mm0
   movq  [eax+2224], mm0
	movq  [eax+2448], mm0
   movq  [eax+3360], mm0
   movq  [eax+3712], mm0
   movq  [eax+3992], mm0
   movq  [eax+16568], mm0
   movq  [eax+16928], mm0

   ; Key bit 25
   movq  mm0, [ebp+200]
	movq  [eax+ 128], mm0
   movq  [eax+ 624], mm0
   movq  [eax+ 984], mm0
   movq  [eax+1416], mm0
   movq  [eax+1736], mm0
   movq  [eax+2088], mm0
   movq  [eax+2480], mm0
   movq  [eax+2848], mm0
   movq  [eax+3304], mm0
   movq  [eax+4112], mm0
   movq  [eax+16680], mm0
   movq  [eax+16976], mm0

   ; Key bit 26
   movq  mm0, [ebp+208]
   movq  [eax+ 200], mm0
   movq  [eax+ 632], mm0
	movq  [eax+1080], mm0
   movq  [eax+1336], mm0
   movq  [eax+1792], mm0
   movq  [eax+2528], mm0
   movq  [eax+2896], mm0
   movq  [eax+3216], mm0
   movq  [eax+3672], mm0
   movq  [eax+4104], mm0
   movq  [eax+16624], mm0
	movq  [eax+17072], mm0

   ; Key bit 27
   movq  mm0, [ebp+216]
   movq  [eax+ 256], mm0
   movq  [eax+ 536], mm0
   movq  [eax+1448], mm0
   movq  [eax+2056], mm0
   movq  [eax+2496], mm0
   movq  [eax+2992], mm0
   movq  [eax+3248], mm0
   movq  [eax+3768], mm0
   movq  [eax+4024], mm0
   movq  [eax+16632], mm0
   movq  [eax+16936], mm0

   ; Key bit 28
	movq  mm0, [ebp+224]
   movq  [eax+ 360], mm0
   movq  [eax+ 840], mm0
   movq  [eax+1152], mm0
   movq  [eax+1576], mm0
   movq  [eax+1888], mm0
   movq  [eax+2384], mm0
   movq  [eax+2720], mm0
   movq  [eax+3160], mm0
	movq  [eax+3576], mm0
   movq  [eax+3792], mm0
   movq  [eax+4336], mm0
   movq  [eax+16784], mm0
   movq  [eax+17216], mm0

   ; Key bit 29
   movq  mm0, [ebp+232]
   movq  [eax+ 784], mm0
   movq  [eax+1104], mm0
   movq  [eax+1648], mm0
   movq  [eax+1896], mm0
   movq  [eax+2632], mm0
   movq  [eax+3168], mm0
   movq  [eax+3448], mm0
   movq  [eax+3888], mm0
   movq  [eax+4184], mm0
	movq  [eax+16704], mm0
   movq  [eax+17240], mm0

   ; Key bit 30
   movq  mm0, [ebp+240]
   movq  [eax+ 704], mm0
   movq  [eax+1200], mm0
   movq  [eax+1496], mm0
   movq  [eax+2368], mm0
	movq  [eax+2808], mm0
   movq  [eax+3056], mm0
   movq  [eax+3480], mm0
   movq  [eax+3808], mm0
   movq  [eax+4232], mm0
   movq  [eax+16808], mm0
   movq  [eax+17248], mm0

   ; Key bit 31
   movq  mm0, [ebp+248]
   movq  [eax+ 352], mm0
   movq  [eax+ 776], mm0
   movq  [eax+1472], mm0
   movq  [eax+1968], mm0
   movq  [eax+2792], mm0
   movq  [eax+3136], mm0
   movq  [eax+3400], mm0
	movq  [eax+3840], mm0
   movq  [eax+4264], mm0
   movq  [eax+16840], mm0

   ; Key bit 32
   movq  mm0, [ebp+256]
   movq  [eax+ 192], mm0
   movq  [eax+ 688], mm0
   movq  [eax+ 912], mm0
	movq  [eax+1816], mm0
   movq  [eax+2120], mm0
   movq  [eax+2552], mm0
   movq  [eax+3000], mm0
   movq  [eax+3368], mm0
   movq  [eax+3688], mm0
   movq  [eax+3976], mm0
   movq  [eax+16592], mm0

   ; Key bit 33
   movq  mm0, [ebp+264]
   movq  [eax+ 280], mm0
   movq  [eax+ 552], mm0
   movq  [eax+ 944], mm0
   movq  [eax+1312], mm0
   movq  [eax+1824], mm0
   movq  [eax+2176], mm0
	movq  [eax+2456], mm0
   movq  [eax+3312], mm0
   movq  [eax+3600], mm0
   movq  [eax+16688], mm0
   movq  [eax+16896], mm0

   ; Key bit 34
   movq  mm0, [ebp+272]
   movq  [eax+ 288], mm0
	movq  [eax+ 992], mm0
   movq  [eax+1360], mm0
   movq  [eax+1768], mm0
   movq  [eax+2576], mm0
   movq  [eax+2816], mm0
   movq  [eax+3320], mm0
   movq  [eax+3632], mm0
   movq  [eax+4000], mm0
   movq  [eax+16552], mm0
   movq  [eax+16968], mm0

   ; Key bit 35
   movq  mm0, [ebp+280]
   movq  [eax+ 440], mm0
   movq  [eax+1096], mm0
   movq  [eax+1632], mm0
   movq  [eax+1944], mm0
	movq  [eax+2272], mm0
   movq  [eax+2696], mm0
   movq  [eax+3472], mm0
   movq  [eax+3960], mm0
   movq  [eax+4208], mm0
   movq  [eax+16832], mm0

   ; Key bit 36
   movq  mm0, [ebp+288]
	movq  [eax+ 488], mm0
   movq  [eax+ 832], mm0
   movq  [eax+1272], mm0
   movq  [eax+1520], mm0
   movq  [eax+1976], mm0
   movq  [eax+2280], mm0
   movq  [eax+2760], mm0
   movq  [eax+3072], mm0
   movq  [eax+3392], mm0
   movq  [eax+3832], mm0
   movq  [eax+16856], mm0

   ; Key bit 37
   movq  mm0, [ebp+296]
   movq  [eax+ 416], mm0
   movq  [eax+ 856], mm0
   movq  [eax+1144], mm0
	movq  [eax+2024], mm0
   movq  [eax+2704], mm0
   movq  [eax+3024], mm0
   movq  [eax+3496], mm0
   movq  [eax+3864], mm0
   movq  [eax+4304], mm0
   movq  [eax+16864], mm0
   movq  [eax+17152], mm0

	; Key bit 38
   movq  mm0, [ebp+304]
   movq  [eax+ 408], mm0
   movq  [eax+ 848], mm0
   movq  [eax+1184], mm0
   movq  [eax+1624], mm0
   movq  [eax+1912], mm0
   movq  [eax+2352], mm0
   movq  [eax+2648], mm0
   movq  [eax+3528], mm0
   movq  [eax+3784], mm0
   movq  [eax+4320], mm0
   movq  [eax+17128], mm0

   ; Key bit 39
   movq  mm0, [ebp+312]
   movq  [eax+ 272], mm0
	movq  [eax+ 512], mm0
   movq  [eax+1008], mm0
   movq  [eax+1368], mm0
   movq  [eax+1800], mm0
   movq  [eax+2200], mm0
   movq  [eax+2472], mm0
   movq  [eax+2864], mm0
   movq  [eax+3280], mm0
   movq  [eax+3752], mm0
	movq  [eax+16992], mm0

   ; Key bit 40
   movq  mm0, [ebp+320]
   movq  [eax+ 264], mm0
   movq  [eax+ 584], mm0
   movq  [eax+1016], mm0
   movq  [eax+1464], mm0
   movq  [eax+1720], mm0
   movq  [eax+2208], mm0
   movq  [eax+2912], mm0
   movq  [eax+3376], mm0
   movq  [eax+3696], mm0
   movq  [eax+4056], mm0
   movq  [eax+16512], mm0
   movq  [eax+16960], mm0

	; Key bit 41
   movq  mm0, [ebp+328]
   movq  [eax+ 184], mm0
   movq  [eax+ 640], mm0
   movq  [eax+ 920], mm0
   movq  [eax+1832], mm0
   movq  [eax+2152], mm0
   movq  [eax+2440], mm0
   movq  [eax+2880], mm0
	movq  [eax+3240], mm0
   movq  [eax+3704], mm0
   movq  [eax+4152], mm0
   movq  [eax+16584], mm0
   movq  [eax+17048], mm0

   ; Key bit 42
   movq  mm0, [ebp+336]
   movq  [eax+ 496], mm0
   movq  [eax+ 744], mm0
   movq  [eax+1224], mm0
   movq  [eax+1536], mm0
   movq  [eax+1960], mm0
   movq  [eax+2328], mm0
   movq  [eax+2768], mm0
   movq  [eax+3104], mm0
   movq  [eax+3520], mm0
	movq  [eax+3856], mm0
   movq  [eax+4176], mm0
   movq  [eax+17256], mm0

   ; Key bit 43
   movq  mm0, [ebp+344]
   movq  [eax+ 344], mm0
   movq  [eax+1168], mm0
   movq  [eax+1488], mm0
	movq  [eax+2032], mm0
   movq  [eax+2360], mm0
   movq  [eax+3016], mm0
   movq  [eax+3544], mm0
   movq  [eax+3776], mm0
   movq  [eax+4272], mm0
   movq  [eax+17184], mm0

   ; Key bit 44
   movq  mm0, [ebp+352]
   movq  [eax+ 392], mm0
   movq  [eax+1088], mm0
   movq  [eax+1584], mm0
   movq  [eax+1880], mm0
   movq  [eax+2408], mm0
   movq  [eax+2752], mm0
   movq  [eax+3192], mm0
	movq  [eax+3552], mm0
   movq  [eax+3880], mm0
   movq  [eax+4192], mm0
   movq  [eax+16768], mm0
   movq  [eax+17096], mm0

   ; Key bit 45
   movq  mm0, [ebp+360]
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

   ; Key bit 46
   movq  mm0, [ebp+368]
   movq  [eax+ 136], mm0
   movq  [eax+ 576], mm0
   movq  [eax+1072], mm0
   movq  [eax+1296], mm0
   movq  [eax+2184], mm0
	movq  [eax+2504], mm0
   movq  [eax+2936], mm0
   movq  [eax+3664], mm0
   movq  [eax+4072], mm0
   movq  [eax+16608], mm0
   movq  [eax+16920], mm0

   ; Key bit 47
   movq  mm0, [ebp+376]
	movq  [eax+ 664], mm0
   movq  [eax+ 936], mm0
   movq  [eax+1328], mm0
   movq  [eax+1696], mm0
   movq  [eax+2104], mm0
   movq  [eax+2560], mm0
   movq  [eax+2840], mm0
   movq  [eax+3200], mm0
   movq  [eax+3760], mm0
   movq  [eax+3984], mm0
   movq  [eax+16576], mm0
   movq  [eax+17040], mm0

   ; Key bit 48
   movq  mm0, [ebp+384]
   movq  [eax+ 160], mm0
   movq  [eax+ 672], mm0
	movq  [eax+1376], mm0
   movq  [eax+1744], mm0
   movq  [eax+2216], mm0
   movq  [eax+2960], mm0
   movq  [eax+3272], mm0
   movq  [eax+3624], mm0
   movq  [eax+4016], mm0
   movq  [eax+16664], mm0
   movq  [eax+17032], mm0

   ; Key bit 49
   movq  mm0, [ebp+392]
   movq  [eax+ 368], mm0
   movq  [eax+ 824], mm0
   movq  [eax+1480], mm0
   movq  [eax+2016], mm0
   movq  [eax+2344], mm0
   movq  [eax+2656], mm0
   movq  [eax+3080], mm0
   movq  [eax+3904], mm0
   movq  [eax+4344], mm0
   movq  [eax+16872], mm0
   movq  [eax+17112], mm0

   ; Key bit 50
   movq  mm0, [ebp+400]
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

   ; Key bit 51
   movq  mm0, [ebp+408]
   movq  [eax+ 464], mm0
   movq  [eax+ 800], mm0
   movq  [eax+1240], mm0
   movq  [eax+1528], mm0
   movq  [eax+2264], mm0
   movq  [eax+3088], mm0
   movq  [eax+3456], mm0
   movq  [eax+3936], mm0
   movq  [eax+4248], mm0
   movq  [eax+16712], mm0
   movq  [eax+17224], mm0

	; Key bit 52
   movq  mm0, [ebp+416]
   movq  [eax+ 480], mm0
   movq  [eax+ 792], mm0
   movq  [eax+1232], mm0
   movq  [eax+1568], mm0
   movq  [eax+2008], mm0
   movq  [eax+2240], mm0
   movq  [eax+2736], mm0
   movq  [eax+3032], mm0
   movq  [eax+3432], mm0
   movq  [eax+4168], mm0
   movq  [eax+16824], mm0
   movq  [eax+17264], mm0

   ; Key bit 53
   movq  mm0, [ebp+424]
   movq  [eax+ 656], mm0
   movq  [eax+ 896], mm0
   movq  [eax+1392], mm0
   movq  [eax+1752], mm0
   movq  [eax+2584], mm0
   movq  [eax+2856], mm0
   movq  [eax+3296], mm0
   movq  [eax+4136], mm0
   movq  [eax+16536], mm0

   ; Key bit 54
   movq  mm0, [ebp+432]
   movq  [eax+ 216], mm0
   movq  [eax+ 648], mm0
   movq  [eax+ 968], mm0
   movq  [eax+1400], mm0
   movq  [eax+1848], mm0
   movq  [eax+2080], mm0
   movq  [eax+2592], mm0
   movq  [eax+3264], mm0
   movq  [eax+3584], mm0
   movq  [eax+4080], mm0
   movq  [eax+16656], mm0
   movq  [eax+16904], mm0

   ; Key bit 55
   movq  mm0, [ebp+440]
   movq  [eax+ 312], mm0
   movq  [eax+ 568], mm0
	movq  [eax+1024], mm0
   movq  [eax+1304], mm0
   movq  [eax+2128], mm0
   movq  [eax+2536], mm0
   movq  [eax+2824], mm0
   movq  [eax+3352], mm0
	movq  [eax+3656], mm0
	movq  [eax+4088], mm0
	movq  [eax+16648], mm0

	; Initialize the plaintext bits, PL and PR
	; This ensures that they are quadword aligned

	movq  mm0, [ebx+48]
	movq  mm1, [ebx+56]
	movq  [PL], mm0
	movq  [PR], mm1

	movq  mm0, [ebx+112]
	movq  mm1, [ebx+120]
	movq  [PL+8], mm0
	movq  [PR+8], mm1

	movq  mm0, [ebx+176]
	movq  mm1, [ebx+184]
	movq  [PL+16], mm0
	movq  [PR+16], mm1

	movq  mm0, [ebx+240]
	movq  mm1, [ebx+248]
	movq  [PL+24], mm0
	movq  [PR+24], mm1

	movq  mm0, [ebx+304]
	movq  mm1, [ebx+312]
	movq  [PL+32], mm0
	movq  [PR+32], mm1

	movq  mm0, [ebx+368]
	movq  mm1, [ebx+376]
	movq  [PL+40], mm0
	movq  [PR+40], mm1

	movq  mm0, [ebx+432]
	movq  mm1, [ebx+440]
	movq  [PL+48], mm0
	movq  [PR+48], mm1

	movq  mm0, [ebx+496]
	movq  mm1, [ebx+504]
	movq  [PL+56], mm0
	movq  [PR+56], mm1

	movq  mm0, [ebx+32]
	movq  mm1, [ebx+40]
	movq  [PL+64], mm0
	movq  [PR+64], mm1

	movq  mm0, [ebx+96]
	movq  mm1, [ebx+104]
	movq  [PL+72], mm0
	movq  [PR+72], mm1

	movq  mm0, [ebx+160]
	movq  mm1, [ebx+168]
	movq  [PL+80], mm0
	movq  [PR+80], mm1

	movq  mm0, [ebx+224]
	movq  mm1, [ebx+232]
	movq  [PL+88], mm0
	movq  [PR+88], mm1

	movq  mm0, [ebx+288]
	movq  mm1, [ebx+296]
	movq  [PL+96], mm0
	movq  [PR+96], mm1

	movq  mm0, [ebx+352]
	movq  mm1, [ebx+360]
	movq  [PL+104], mm0
	movq  [PR+104], mm1

	movq  mm0, [ebx+416]
	movq  mm1, [ebx+424]
	movq  [PL+112], mm0
	movq  [PR+112], mm1

	movq  mm0, [ebx+480]
	movq  mm1, [ebx+488]
	movq  [PL+120], mm0
   movq  [PR+120], mm1

   movq  mm0, [ebx+16]
   movq  mm1, [ebx+24]
   movq  [PL+128], mm0
   movq  [PR+128], mm1

   movq  mm0, [ebx+80]
   movq  mm1, [ebx+88]
   movq  [PL+136], mm0
   movq  [PR+136], mm1

   movq  mm0, [ebx+144]
	movq  mm1, [ebx+152]
   movq  [PL+144], mm0
   movq  [PR+144], mm1

   movq  mm0, [ebx+208]
   movq  mm1, [ebx+216]
   movq  [PL+152], mm0
   movq  [PR+152], mm1

   movq  mm0, [ebx+272]
   movq  mm1, [ebx+280]
	movq  [PL+160], mm0
   movq  [PR+160], mm1

   movq  mm0, [ebx+336]
   movq  mm1, [ebx+344]
   movq  [PL+168], mm0
   movq  [PR+168], mm1

   movq  mm0, [ebx+400]
   movq  mm1, [ebx+408]
   movq  [PL+176], mm0
   movq  [PR+176], mm1

   movq  mm0, [ebx+464]
   movq  mm1, [ebx+472]
	movq  [PL+184], mm0
   movq  [PR+184], mm1

   movq  mm0, [ebx]
   movq  mm1, [ebx+8]
   movq  [PL+192], mm0
   movq  [PR+192], mm1

   movq  mm0, [ebx+64]
   movq  mm1, [ebx+72]
   movq  [PL+200], mm0
	movq  [PR+200], mm1

   movq  mm0, [ebx+128]
   movq  mm1, [ebx+136]
   movq  [PL+208], mm0
   movq  [PR+208], mm1

   movq  mm0, [ebx+192]
   movq  mm1, [ebx+200]
   movq  [PL+216], mm0
   movq  [PR+216], mm1

   movq  mm0, [ebx+256]
   movq  mm1, [ebx+264]
   movq  [PL+224], mm0
	movq  [PR+224], mm1

   movq  mm0, [ebx+320]
   movq  mm1, [ebx+328]
   movq  [PL+232], mm0
   movq  [PR+232], mm1

   movq  mm0, [ebx+384]
   movq  mm1, [ebx+392]
   movq  [PL+240], mm0
   movq  [PR+240], mm1

	movq  mm0, [ebx+448]
	movq  mm1, [ebx+456]
	movq  [PL+248], mm0
	movq  [PR+248], mm1


	; Initialize the ciphertext bits, CL and CR
	; More quadword alignment

   movq  mm0, [esi+40]
   movq  mm1, [esi+32]
   movq  [CL+64], mm0
   movq  [CR+64], mm1

	movq  mm0, [esi+24]
   movq  mm1, [esi+16]
   movq  [CL+128], mm0
   movq  [CR+128], mm1

   movq  mm0, [esi+408]
   movq  mm1, [esi+400]
   movq  [CL+176], mm0
   movq  [CR+176], mm1

   movq  mm0, [esi+392]
   movq  mm1, [esi+384]
   movq  [CL+240], mm0
   movq  [CR+240], mm1

   movq  mm0, [esi+296]
   movq  mm1, [esi+288]
   movq  [CL+96], mm0
   movq  [CR+96], mm1

   movq  mm0, [esi+200]
   movq  mm1, [esi+192]
   movq  [CL+216], mm0
   movq  [CR+216], mm1

   movq  mm0, [esi+120]
	movq  mm1, [esi+112]
   movq  [CL+8], mm0
   movq  [CR+8], mm1

   movq  mm0, [esi+88]
   movq  mm1, [esi+80]
   movq  [CL+136], mm0
   movq  [CR+136], mm1

   movq  mm0, [esi+472]
   movq  mm1, [esi+464]
   movq  [CL+184], mm0
   movq  [CR+184], mm1

   movq  mm0, [esi+488]
   movq  mm1, [esi+480]
   movq  [CL+120], mm0
   movq  [CR+120], mm1

	movq  mm0, [esi+328]
   movq  mm1, [esi+320]
   movq  [CL+232], mm0
   movq  [CR+232], mm1

   movq  mm0, [esi+376]
   movq  mm1, [esi+368]
	movq  [CL+40], mm0
   movq  [CR+40], mm1

   movq  mm0, [esi+72]
   movq  mm1, [esi+64]
   movq  [CL+200], mm0
   movq  [CR+200], mm1

   movq  mm0, [esi+216]
   movq  mm1, [esi+208]
   movq  [CL+152], mm0
   movq  [CR+152], mm1

   movq  mm0, [esi+104]
   movq  mm1, [esi+96]
   movq  [CL+72], mm0
   movq  [CR+72], mm1

   movq  mm0, [esi+56]
	movq  mm1, [esi+48]
   movq  [CL], mm0
   movq  [CR], mm1

   movq  mm0, [esi+504]
   movq  mm1, [esi+496]
   movq  [CL+56], mm0
	movq  [CR+56], mm1

   movq  mm0, [esi+360]
   movq  mm1, [esi+352]
   movq  [CL+104], mm0
   movq  [CR+104], mm1

   movq  mm0, [esi+8]
   movq  mm1, [esi]
   movq  [CL+192], mm0
   movq  [CR+192], mm1

   movq  mm0, [esi+184]
   movq  mm1, [esi+176]
   movq  [CL+16], mm0
   movq  [CR+16], mm1

   movq  mm0, [esi+248]
   movq  mm1, [esi+240]
	movq  [CL+24], mm0
   movq  [CR+24], mm1

   movq  mm0, [esi+264]
   movq  mm1, [esi+256]
   movq  [CL+224], mm0
   movq  [CR+224], mm1

   movq  mm0, [esi+168]
   movq  mm1, [esi+160]
   movq  [CL+80], mm0
   movq  [CR+80], mm1

   movq  mm0, [esi+152]
   movq  mm1, [esi+144]
   movq  [CL+144], mm0
   movq  [CR+144], mm1

   movq  mm0, [esi+456]
   movq  mm1, [esi+448]
   movq  [CL+248], mm0
   movq  [CR+248], mm1

   movq  mm0, [esi+232]
   movq  mm1, [esi+224]
   movq  [CL+88], mm0
	movq  [CR+88], mm1

   movq  mm0, [esi+344]
   movq  mm1, [esi+336]
   movq  [CL+168], mm0
   movq  [CR+168], mm1

	movq  mm0, [esi+440]
   movq  mm1, [esi+432]
   movq  [CL+48], mm0
   movq  [CR+48], mm1

   movq  mm0, [esi+312]
   movq  mm1, [esi+304]
   movq  [CL+32], mm0
   movq  [CR+32], mm1

   movq  mm0, [esi+136]
   movq  mm1, [esi+128]
   movq  [CL+208], mm0
   movq  [CR+208], mm1

   movq  mm0, [esi+424]
   movq  mm1, [esi+416]
   movq  [CL+112], mm0
   movq  [CR+112], mm1

   movq  mm0, [esi+280]
   movq  mm1, [esi+272]
   movq  [CL+160], mm0
   movq  [CR+160], mm1

   retn

