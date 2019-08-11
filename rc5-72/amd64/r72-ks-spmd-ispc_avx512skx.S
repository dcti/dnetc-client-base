	.text
	.file	"r72-ks-spmd-ispc.ispc"
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2               # -- Begin function rc5pipe___vyuvyuvyuvyuvyuvyuun_3C_unu_3E_un_3C_unu_3E_avx512skx
.LCPI0_0:
	.long	3205139229              # 0xbf0a8b1d
.LCPI0_1:
	.long	354637369               # 0x15235639
.LCPI0_2:
	.long	31                      # 0x1f
.LCPI0_3:
	.long	4098901205              # 0xf45044d5
.LCPI0_4:
	.long	2458369678              # 0x9287be8e
.LCPI0_5:
	.long	817838151               # 0x30bf3847
.LCPI0_6:
	.long	3472273920              # 0xcef6b200
.LCPI0_7:
	.long	1831742393              # 0x6d2e2bb9
.LCPI0_8:
	.long	191210866               # 0xb65a572
.LCPI0_9:
	.long	2845646635              # 0xa99d1f2b
.LCPI0_10:
	.long	1205115108              # 0x47d498e4
.LCPI0_11:
	.long	3859550877              # 0xe60c129d
.LCPI0_12:
	.long	2219019350              # 0x84438c56
.LCPI0_13:
	.long	578487823               # 0x227b060f
.LCPI0_14:
	.long	3232923592              # 0xc0b27fc8
.LCPI0_15:
	.long	1592392065              # 0x5ee9f981
.LCPI0_16:
	.long	4246827834              # 0xfd21733a
.LCPI0_17:
	.long	2606296307              # 0x9b58ecf3
.LCPI0_18:
	.long	965764780               # 0x399066ac
.LCPI0_19:
	.long	3620200549              # 0xd7c7e065
.LCPI0_20:
	.long	1979669022              # 0x75ff5a1e
.LCPI0_21:
	.long	339137495               # 0x1436d3d7
.LCPI0_22:
	.long	2993573264              # 0xb26e4d90
.LCPI0_23:
	.long	1353041737              # 0x50a5c749
.LCPI0_24:
	.long	4007477506              # 0xeedd4102
.LCPI0_25:
	.long	2366945979              # 0x8d14babb
.LCPI0_26:
	.long	726414452               # 0x2b4c3474
	.text
	.p2align	4, 0x90
	.type	rc5pipe___vyuvyuvyuvyuvyuvyuun_3C_unu_3E_un_3C_unu_3E_avx512skx,@function
rc5pipe___vyuvyuvyuvyuvyuvyuun_3C_unu_3E_un_3C_unu_3E_avx512skx: # @rc5pipe___vyuvyuvyuvyuvyuvyuun_3C_unu_3E_un_3C_unu_3E_avx512skx
# %bb.0:                                # %allocas
	subq	$56, %rsp
	vmovups	%zmm5, -128(%rsp)       # 64-byte Spill
	vmovups	%zmm4, -64(%rsp)        # 64-byte Spill
	vmovdqa64	%zmm0, %zmm5
	vpaddd	%zmm0, %zmm1, %zmm6
	vpbroadcastd	.LCPI0_0(%rip), %zmm29 # zmm29 = [3205139229,3205139229,3205139229,3205139229,3205139229,3205139229,3205139229,3205139229,3205139229,3205139229,3205139229,3205139229,3205139229,3205139229,3205139229,3205139229]
	vpaddd	%zmm29, %zmm3, %zmm1
	vprold	$29, %zmm1, %zmm7
	vpaddd	.LCPI0_1(%rip){1to16}, %zmm7, %zmm1
	vprold	$3, %zmm1, %zmm3
	vpaddd	%zmm7, %zmm3, %zmm8
	vpaddd	%zmm2, %zmm8, %zmm2
	vpbroadcastd	.LCPI0_2(%rip), %zmm1 # zmm1 = [31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31]
	vpandd	%zmm1, %zmm8, %zmm8
	vprolvd	%zmm8, %zmm2, %zmm8
	vpaddd	%zmm8, %zmm3, %zmm2
	vpaddd	.LCPI0_3(%rip){1to16}, %zmm2, %zmm2
	vprold	$3, %zmm2, %zmm4
	vpaddd	%zmm8, %zmm4, %zmm9
	vpaddd	%zmm9, %zmm6, %zmm6
	vpandd	%zmm1, %zmm9, %zmm9
	vprolvd	%zmm9, %zmm6, %zmm9
	vpaddd	%zmm9, %zmm4, %zmm6
	vpaddd	.LCPI0_4(%rip){1to16}, %zmm6, %zmm6
	vprold	$3, %zmm6, %zmm6
	vpaddd	%zmm9, %zmm6, %zmm10
	vpaddd	%zmm7, %zmm10, %zmm7
	vpandd	%zmm1, %zmm10, %zmm10
	vprolvd	%zmm10, %zmm7, %zmm10
	vpaddd	%zmm10, %zmm6, %zmm7
	vpaddd	.LCPI0_5(%rip){1to16}, %zmm7, %zmm7
	vprold	$3, %zmm7, %zmm7
	vpaddd	%zmm10, %zmm7, %zmm11
	vpaddd	%zmm8, %zmm11, %zmm8
	vpandd	%zmm1, %zmm11, %zmm11
	vprolvd	%zmm11, %zmm8, %zmm11
	vpaddd	%zmm11, %zmm7, %zmm8
	vpaddd	.LCPI0_6(%rip){1to16}, %zmm8, %zmm8
	vprold	$3, %zmm8, %zmm8
	vpaddd	%zmm11, %zmm8, %zmm12
	vpaddd	%zmm9, %zmm12, %zmm9
	vpandd	%zmm1, %zmm12, %zmm12
	vprolvd	%zmm12, %zmm9, %zmm12
	vpaddd	%zmm12, %zmm8, %zmm9
	vpaddd	.LCPI0_7(%rip){1to16}, %zmm9, %zmm9
	vprold	$3, %zmm9, %zmm9
	vpaddd	%zmm12, %zmm9, %zmm13
	vpaddd	%zmm10, %zmm13, %zmm10
	vpandd	%zmm1, %zmm13, %zmm13
	vprolvd	%zmm13, %zmm10, %zmm13
	vpaddd	%zmm13, %zmm9, %zmm10
	vpaddd	.LCPI0_8(%rip){1to16}, %zmm10, %zmm10
	vprold	$3, %zmm10, %zmm10
	vpaddd	%zmm13, %zmm10, %zmm14
	vpaddd	%zmm11, %zmm14, %zmm11
	vpandd	%zmm1, %zmm14, %zmm14
	vprolvd	%zmm14, %zmm11, %zmm14
	vpaddd	%zmm14, %zmm10, %zmm11
	vpaddd	.LCPI0_9(%rip){1to16}, %zmm11, %zmm11
	vprold	$3, %zmm11, %zmm11
	vpaddd	%zmm14, %zmm11, %zmm15
	vpaddd	%zmm12, %zmm15, %zmm12
	vpandd	%zmm1, %zmm15, %zmm15
	vprolvd	%zmm15, %zmm12, %zmm15
	vpaddd	%zmm15, %zmm11, %zmm12
	vpaddd	.LCPI0_10(%rip){1to16}, %zmm12, %zmm12
	vprold	$3, %zmm12, %zmm12
	vpaddd	%zmm15, %zmm12, %zmm16
	vpaddd	%zmm13, %zmm16, %zmm13
	vpandd	%zmm1, %zmm16, %zmm16
	vprolvd	%zmm16, %zmm13, %zmm16
	vpaddd	%zmm16, %zmm12, %zmm13
	vpaddd	.LCPI0_11(%rip){1to16}, %zmm13, %zmm13
	vprold	$3, %zmm13, %zmm13
	vpaddd	%zmm16, %zmm13, %zmm17
	vpaddd	%zmm14, %zmm17, %zmm14
	vpandd	%zmm1, %zmm17, %zmm17
	vprolvd	%zmm17, %zmm14, %zmm17
	vpaddd	%zmm17, %zmm13, %zmm14
	vpaddd	.LCPI0_12(%rip){1to16}, %zmm14, %zmm14
	vprold	$3, %zmm14, %zmm14
	vpaddd	%zmm17, %zmm14, %zmm18
	vpaddd	%zmm15, %zmm18, %zmm15
	vpandd	%zmm1, %zmm18, %zmm18
	vprolvd	%zmm18, %zmm15, %zmm18
	vpaddd	%zmm18, %zmm14, %zmm15
	vpaddd	.LCPI0_13(%rip){1to16}, %zmm15, %zmm15
	vprold	$3, %zmm15, %zmm15
	vpaddd	%zmm18, %zmm15, %zmm19
	vpaddd	%zmm16, %zmm19, %zmm16
	vpandd	%zmm1, %zmm19, %zmm19
	vprolvd	%zmm19, %zmm16, %zmm19
	vpaddd	%zmm19, %zmm15, %zmm16
	vpaddd	.LCPI0_14(%rip){1to16}, %zmm16, %zmm16
	vprold	$3, %zmm16, %zmm16
	vpaddd	%zmm19, %zmm16, %zmm20
	vpaddd	%zmm17, %zmm20, %zmm17
	vpandd	%zmm1, %zmm20, %zmm20
	vprolvd	%zmm20, %zmm17, %zmm20
	vpaddd	%zmm20, %zmm16, %zmm17
	vpaddd	.LCPI0_15(%rip){1to16}, %zmm17, %zmm17
	vprold	$3, %zmm17, %zmm17
	vpaddd	%zmm20, %zmm17, %zmm21
	vpaddd	%zmm18, %zmm21, %zmm18
	vpandd	%zmm1, %zmm21, %zmm21
	vprolvd	%zmm21, %zmm18, %zmm21
	vpaddd	%zmm21, %zmm17, %zmm18
	vpaddd	.LCPI0_16(%rip){1to16}, %zmm18, %zmm18
	vprold	$3, %zmm18, %zmm18
	vpaddd	%zmm21, %zmm18, %zmm22
	vpaddd	%zmm19, %zmm22, %zmm19
	vpandd	%zmm1, %zmm22, %zmm22
	vprolvd	%zmm22, %zmm19, %zmm22
	vpaddd	%zmm22, %zmm18, %zmm19
	vpaddd	.LCPI0_17(%rip){1to16}, %zmm19, %zmm19
	vprold	$3, %zmm19, %zmm19
	vpaddd	%zmm22, %zmm19, %zmm23
	vpaddd	%zmm20, %zmm23, %zmm20
	vpandd	%zmm1, %zmm23, %zmm23
	vprolvd	%zmm23, %zmm20, %zmm23
	vpaddd	%zmm23, %zmm19, %zmm20
	vpaddd	.LCPI0_18(%rip){1to16}, %zmm20, %zmm20
	vprold	$3, %zmm20, %zmm20
	vpaddd	%zmm23, %zmm20, %zmm24
	vpaddd	%zmm21, %zmm24, %zmm21
	vpandd	%zmm1, %zmm24, %zmm24
	vprolvd	%zmm24, %zmm21, %zmm24
	vpaddd	%zmm24, %zmm20, %zmm21
	vpaddd	.LCPI0_19(%rip){1to16}, %zmm21, %zmm21
	vprold	$3, %zmm21, %zmm21
	vpaddd	%zmm24, %zmm21, %zmm25
	vpaddd	%zmm22, %zmm25, %zmm22
	vpandd	%zmm1, %zmm25, %zmm25
	vprolvd	%zmm25, %zmm22, %zmm25
	vpaddd	%zmm25, %zmm21, %zmm22
	vpaddd	.LCPI0_20(%rip){1to16}, %zmm22, %zmm22
	vprold	$3, %zmm22, %zmm22
	vpaddd	%zmm25, %zmm22, %zmm26
	vpaddd	%zmm23, %zmm26, %zmm23
	vpandd	%zmm1, %zmm26, %zmm26
	vprolvd	%zmm26, %zmm23, %zmm26
	vpaddd	%zmm26, %zmm22, %zmm23
	vpaddd	.LCPI0_21(%rip){1to16}, %zmm23, %zmm23
	vprold	$3, %zmm23, %zmm23
	vpaddd	%zmm26, %zmm23, %zmm27
	vpaddd	%zmm24, %zmm27, %zmm24
	vpandd	%zmm1, %zmm27, %zmm27
	vprolvd	%zmm27, %zmm24, %zmm27
	vpaddd	%zmm27, %zmm23, %zmm24
	vpaddd	.LCPI0_22(%rip){1to16}, %zmm24, %zmm24
	vprold	$3, %zmm24, %zmm24
	vpaddd	%zmm27, %zmm24, %zmm28
	vpaddd	%zmm25, %zmm28, %zmm25
	vpandd	%zmm1, %zmm28, %zmm28
	vprolvd	%zmm28, %zmm25, %zmm28
	vpaddd	%zmm28, %zmm24, %zmm25
	vpaddd	.LCPI0_23(%rip){1to16}, %zmm25, %zmm25
	vprold	$3, %zmm25, %zmm25
	vpaddd	%zmm28, %zmm25, %zmm30
	vpaddd	%zmm26, %zmm30, %zmm26
	vpandd	%zmm1, %zmm30, %zmm30
	vprolvd	%zmm30, %zmm26, %zmm30
	vpaddd	%zmm30, %zmm25, %zmm26
	vpaddd	.LCPI0_24(%rip){1to16}, %zmm26, %zmm26
	vprold	$3, %zmm26, %zmm26
	vpaddd	%zmm30, %zmm26, %zmm31
	vpaddd	%zmm27, %zmm31, %zmm27
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm27, %zmm31
	vpaddd	%zmm31, %zmm26, %zmm27
	vpaddd	.LCPI0_25(%rip){1to16}, %zmm27, %zmm27
	vprold	$3, %zmm27, %zmm27
	vpaddd	%zmm31, %zmm27, %zmm0
	vpaddd	%zmm28, %zmm0, %zmm28
	vpandd	%zmm1, %zmm0, %zmm0
	vprolvd	%zmm0, %zmm28, %zmm2
	vpaddd	%zmm2, %zmm27, %zmm28
	vpaddd	.LCPI0_26(%rip){1to16}, %zmm28, %zmm28
	vprold	$3, %zmm28, %zmm28
	vpaddd	%zmm2, %zmm28, %zmm0
	vpaddd	%zmm30, %zmm0, %zmm30
	vpandd	%zmm1, %zmm0, %zmm0
	vprolvd	%zmm0, %zmm30, %zmm0
	vpaddd	%zmm29, %zmm28, %zmm29
	vpaddd	%zmm29, %zmm0, %zmm29
	vprold	$3, %zmm29, %zmm29
	vpaddd	%zmm0, %zmm29, %zmm30
	vpaddd	%zmm31, %zmm30, %zmm31
	vpandd	%zmm1, %zmm30, %zmm30
	vprolvd	%zmm30, %zmm31, %zmm30
	vpaddd	%zmm3, %zmm29, %zmm3
	vpaddd	%zmm30, %zmm3, %zmm3
	vprold	$3, %zmm3, %zmm3
	vpaddd	%zmm30, %zmm3, %zmm31
	vpaddd	%zmm2, %zmm31, %zmm2
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm2, %zmm31
	vpaddd	%zmm4, %zmm3, %zmm2
	vpaddd	%zmm31, %zmm2, %zmm2
	vprold	$3, %zmm2, %zmm2
	vpaddd	%zmm31, %zmm2, %zmm4
	vpaddd	%zmm0, %zmm4, %zmm0
	vpandd	%zmm1, %zmm4, %zmm4
	vprolvd	%zmm4, %zmm0, %zmm0
	vpaddd	%zmm6, %zmm2, %zmm4
	vpaddd	%zmm0, %zmm4, %zmm4
	vprold	$3, %zmm4, %zmm6
	vpaddd	%zmm0, %zmm6, %zmm4
	vpaddd	%zmm30, %zmm4, %zmm30
	vpandd	%zmm1, %zmm4, %zmm4
	vprolvd	%zmm4, %zmm30, %zmm4
	vpaddd	%zmm7, %zmm6, %zmm7
	vpaddd	%zmm4, %zmm7, %zmm7
	vprold	$3, %zmm7, %zmm7
	vpaddd	%zmm4, %zmm7, %zmm30
	vpaddd	%zmm31, %zmm30, %zmm31
	vpandd	%zmm1, %zmm30, %zmm30
	vprolvd	%zmm30, %zmm31, %zmm30
	vpaddd	%zmm8, %zmm7, %zmm8
	vpaddd	%zmm30, %zmm8, %zmm8
	vprold	$3, %zmm8, %zmm8
	vpaddd	%zmm30, %zmm8, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm9, %zmm8, %zmm9
	vpaddd	%zmm0, %zmm9, %zmm9
	vprold	$3, %zmm9, %zmm9
	vpaddd	%zmm0, %zmm9, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm10, %zmm9, %zmm10
	vpaddd	%zmm4, %zmm10, %zmm10
	vprold	$3, %zmm10, %zmm10
	vpaddd	%zmm4, %zmm10, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm30
	vpaddd	%zmm11, %zmm10, %zmm11
	vpaddd	%zmm30, %zmm11, %zmm11
	vprold	$3, %zmm11, %zmm11
	vpaddd	%zmm30, %zmm11, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm12, %zmm11, %zmm12
	vpaddd	%zmm0, %zmm12, %zmm12
	vprold	$3, %zmm12, %zmm12
	vpaddd	%zmm0, %zmm12, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm13, %zmm12, %zmm13
	vpaddd	%zmm4, %zmm13, %zmm13
	vprold	$3, %zmm13, %zmm13
	vpaddd	%zmm4, %zmm13, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm30
	vpaddd	%zmm14, %zmm13, %zmm14
	vpaddd	%zmm30, %zmm14, %zmm14
	vprold	$3, %zmm14, %zmm14
	vpaddd	%zmm30, %zmm14, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm15, %zmm14, %zmm15
	vpaddd	%zmm0, %zmm15, %zmm15
	vprold	$3, %zmm15, %zmm15
	vpaddd	%zmm0, %zmm15, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm16, %zmm15, %zmm16
	vpaddd	%zmm4, %zmm16, %zmm16
	vprold	$3, %zmm16, %zmm16
	vpaddd	%zmm4, %zmm16, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm30
	vpaddd	%zmm17, %zmm16, %zmm17
	vpaddd	%zmm30, %zmm17, %zmm17
	vprold	$3, %zmm17, %zmm17
	vpaddd	%zmm30, %zmm17, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm18, %zmm17, %zmm18
	vpaddd	%zmm0, %zmm18, %zmm18
	vprold	$3, %zmm18, %zmm18
	vpaddd	%zmm0, %zmm18, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm19, %zmm18, %zmm19
	vpaddd	%zmm4, %zmm19, %zmm19
	vprold	$3, %zmm19, %zmm19
	vpaddd	%zmm4, %zmm19, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm30
	vpaddd	%zmm20, %zmm19, %zmm20
	vpaddd	%zmm30, %zmm20, %zmm20
	vprold	$3, %zmm20, %zmm20
	vpaddd	%zmm30, %zmm20, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm21, %zmm20, %zmm21
	vpaddd	%zmm0, %zmm21, %zmm21
	vprold	$3, %zmm21, %zmm21
	vpaddd	%zmm0, %zmm21, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm22, %zmm21, %zmm22
	vpaddd	%zmm4, %zmm22, %zmm22
	vprold	$3, %zmm22, %zmm22
	vpaddd	%zmm4, %zmm22, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm30
	vpaddd	%zmm23, %zmm22, %zmm23
	vpaddd	%zmm30, %zmm23, %zmm23
	vprold	$3, %zmm23, %zmm23
	vpaddd	%zmm30, %zmm23, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm24, %zmm23, %zmm24
	vpaddd	%zmm0, %zmm24, %zmm24
	vprold	$3, %zmm24, %zmm24
	vpaddd	%zmm0, %zmm24, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm25, %zmm24, %zmm25
	vpaddd	%zmm4, %zmm25, %zmm25
	vprold	$3, %zmm25, %zmm25
	vpaddd	%zmm4, %zmm25, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm30
	vpaddd	%zmm26, %zmm25, %zmm26
	vpaddd	%zmm30, %zmm26, %zmm26
	vprold	$3, %zmm26, %zmm26
	vpaddd	%zmm30, %zmm26, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm27, %zmm26, %zmm27
	vpaddd	%zmm0, %zmm27, %zmm27
	vprold	$3, %zmm27, %zmm27
	vpaddd	%zmm0, %zmm27, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm28, %zmm27, %zmm28
	vpaddd	%zmm4, %zmm28, %zmm28
	vprold	$3, %zmm28, %zmm28
	vpaddd	%zmm4, %zmm28, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm30
	vpaddd	%zmm29, %zmm28, %zmm29
	vpaddd	%zmm30, %zmm29, %zmm29
	vprold	$3, %zmm29, %zmm29
	vpaddd	%zmm30, %zmm29, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm3, %zmm29, %zmm3
	vpaddd	%zmm0, %zmm3, %zmm3
	vprold	$3, %zmm3, %zmm3
	vpaddd	%zmm0, %zmm3, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm2, %zmm3, %zmm2
	vpaddd	%zmm4, %zmm2, %zmm2
	vprold	$3, %zmm2, %zmm2
	vpaddd	%zmm4, %zmm2, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm30
	vpaddd	%zmm6, %zmm2, %zmm6
	vpaddd	%zmm30, %zmm6, %zmm6
	vprold	$3, %zmm6, %zmm6
	vpaddd	%zmm30, %zmm6, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm7, %zmm6, %zmm7
	vpaddd	%zmm0, %zmm7, %zmm7
	vprold	$3, %zmm7, %zmm7
	vpaddd	%zmm0, %zmm7, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm8, %zmm7, %zmm8
	vpaddd	%zmm4, %zmm8, %zmm8
	vprold	$3, %zmm8, %zmm8
	vpaddd	%zmm4, %zmm8, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm30
	vpaddd	%zmm9, %zmm8, %zmm9
	vpaddd	%zmm30, %zmm9, %zmm9
	vprold	$3, %zmm9, %zmm9
	vpaddd	%zmm30, %zmm9, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm10, %zmm9, %zmm10
	vpaddd	%zmm0, %zmm10, %zmm10
	vprold	$3, %zmm10, %zmm10
	vpaddd	%zmm0, %zmm10, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm11, %zmm10, %zmm11
	vpaddd	%zmm4, %zmm11, %zmm11
	vprold	$3, %zmm11, %zmm11
	vpaddd	%zmm4, %zmm11, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm30
	vpaddd	%zmm12, %zmm11, %zmm12
	vpaddd	%zmm30, %zmm12, %zmm12
	vprold	$3, %zmm12, %zmm12
	vpaddd	%zmm30, %zmm12, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm13, %zmm12, %zmm13
	vpaddd	%zmm0, %zmm13, %zmm13
	vprold	$3, %zmm13, %zmm13
	vpaddd	%zmm0, %zmm13, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm14, %zmm13, %zmm14
	vpaddd	%zmm4, %zmm14, %zmm14
	vprold	$3, %zmm14, %zmm14
	vpaddd	%zmm4, %zmm14, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm30
	vpaddd	%zmm15, %zmm14, %zmm15
	vpaddd	%zmm30, %zmm15, %zmm15
	vprold	$3, %zmm15, %zmm15
	vpaddd	%zmm30, %zmm15, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm16, %zmm15, %zmm16
	vpaddd	%zmm0, %zmm16, %zmm16
	vprold	$3, %zmm16, %zmm16
	vpaddd	%zmm0, %zmm16, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm17, %zmm16, %zmm17
	vpaddd	%zmm4, %zmm17, %zmm17
	vprold	$3, %zmm17, %zmm17
	vpaddd	%zmm4, %zmm17, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm30
	vpaddd	%zmm18, %zmm17, %zmm18
	vpaddd	%zmm30, %zmm18, %zmm18
	vprold	$3, %zmm18, %zmm18
	vpaddd	%zmm30, %zmm18, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm19, %zmm18, %zmm19
	vpaddd	%zmm0, %zmm19, %zmm19
	vprold	$3, %zmm19, %zmm19
	vpaddd	%zmm0, %zmm19, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm20, %zmm19, %zmm20
	vpaddd	%zmm4, %zmm20, %zmm20
	vprold	$3, %zmm20, %zmm20
	vpaddd	%zmm4, %zmm20, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm30
	vpaddd	%zmm21, %zmm20, %zmm21
	vpaddd	%zmm30, %zmm21, %zmm21
	vprold	$3, %zmm21, %zmm21
	vpaddd	%zmm30, %zmm21, %zmm31
	vpaddd	%zmm0, %zmm31, %zmm0
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm0, %zmm0
	vpaddd	%zmm22, %zmm21, %zmm22
	vpaddd	%zmm0, %zmm22, %zmm22
	vprold	$3, %zmm22, %zmm22
	vpaddd	%zmm0, %zmm22, %zmm31
	vpaddd	%zmm4, %zmm31, %zmm4
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm4, %zmm4
	vpaddd	%zmm23, %zmm22, %zmm23
	vpaddd	%zmm4, %zmm23, %zmm23
	vprold	$3, %zmm23, %zmm23
	vpaddd	%zmm4, %zmm23, %zmm31
	vpaddd	%zmm30, %zmm31, %zmm30
	vpandd	%zmm1, %zmm31, %zmm31
	vprolvd	%zmm31, %zmm30, %zmm31
	vpaddd	%zmm24, %zmm23, %zmm24
	vpaddd	%zmm31, %zmm24, %zmm24
	vprold	$3, %zmm24, %zmm24
	vpaddd	%zmm31, %zmm24, %zmm30
	vpaddd	%zmm0, %zmm30, %zmm0
	vpandd	%zmm1, %zmm30, %zmm30
	vprolvd	%zmm30, %zmm0, %zmm0
	vpaddd	%zmm25, %zmm24, %zmm25
	vpaddd	%zmm0, %zmm25, %zmm25
	vprold	$3, %zmm25, %zmm25
	vpaddd	%zmm0, %zmm25, %zmm30
	vpaddd	%zmm4, %zmm30, %zmm4
	vpandd	%zmm1, %zmm30, %zmm30
	vprolvd	%zmm30, %zmm4, %zmm4
	vpaddd	%zmm26, %zmm25, %zmm26
	vpaddd	%zmm4, %zmm26, %zmm26
	vprold	$3, %zmm26, %zmm30
	vpaddd	%zmm4, %zmm30, %zmm4
	vpaddd	%zmm31, %zmm4, %zmm26
	vpandd	%zmm1, %zmm4, %zmm4
	vprolvd	%zmm4, %zmm26, %zmm4
	vpaddd	%zmm27, %zmm30, %zmm26
	vpaddd	%zmm4, %zmm26, %zmm26
	vprold	$3, %zmm26, %zmm27
	vpaddd	%zmm4, %zmm27, %zmm4
	vpaddd	%zmm0, %zmm4, %zmm0
	vpandd	%zmm1, %zmm4, %zmm4
	vprolvd	%zmm4, %zmm0, %zmm0
	vpaddd	%zmm28, %zmm27, %zmm4
	vpaddd	%zmm0, %zmm4, %zmm26
	vpaddd	-128(%rsp), %zmm29, %zmm0 # 64-byte Folded Reload
	vpaddd	-64(%rsp), %zmm3, %zmm3 # 64-byte Folded Reload
	vpxord	%zmm0, %zmm3, %zmm0
	vpandd	%zmm1, %zmm3, %zmm4
	vprolvd	%zmm4, %zmm0, %zmm0
	vpaddd	%zmm0, %zmm2, %zmm0
	vpxord	%zmm3, %zmm0, %zmm2
	vpandd	%zmm1, %zmm0, %zmm3
	vprolvd	%zmm3, %zmm2, %zmm2
	vpaddd	%zmm2, %zmm6, %zmm2
	vpxord	%zmm0, %zmm2, %zmm0
	vpandd	%zmm1, %zmm2, %zmm3
	vprolvd	%zmm3, %zmm0, %zmm0
	vpaddd	%zmm0, %zmm7, %zmm0
	vpxord	%zmm2, %zmm0, %zmm2
	vpandd	%zmm1, %zmm0, %zmm3
	vprolvd	%zmm3, %zmm2, %zmm2
	vpaddd	%zmm2, %zmm8, %zmm2
	vpxord	%zmm0, %zmm2, %zmm0
	vpandd	%zmm1, %zmm2, %zmm3
	vprolvd	%zmm3, %zmm0, %zmm0
	vpaddd	%zmm0, %zmm9, %zmm0
	vpxord	%zmm2, %zmm0, %zmm2
	vpandd	%zmm1, %zmm0, %zmm3
	vprolvd	%zmm3, %zmm2, %zmm2
	vpaddd	%zmm2, %zmm10, %zmm2
	vpxord	%zmm0, %zmm2, %zmm0
	vpandd	%zmm1, %zmm2, %zmm3
	vprolvd	%zmm3, %zmm0, %zmm0
	vpaddd	%zmm0, %zmm11, %zmm0
	vpxord	%zmm2, %zmm0, %zmm2
	vpandd	%zmm1, %zmm0, %zmm3
	vprolvd	%zmm3, %zmm2, %zmm2
	vpaddd	%zmm2, %zmm12, %zmm2
	vpxord	%zmm0, %zmm2, %zmm0
	vpandd	%zmm1, %zmm2, %zmm3
	vprolvd	%zmm3, %zmm0, %zmm0
	vpaddd	%zmm0, %zmm13, %zmm0
	vpxord	%zmm2, %zmm0, %zmm2
	vpandd	%zmm1, %zmm0, %zmm3
	vprolvd	%zmm3, %zmm2, %zmm2
	vpaddd	%zmm2, %zmm14, %zmm2
	vpxord	%zmm0, %zmm2, %zmm0
	vpandd	%zmm1, %zmm2, %zmm3
	vprolvd	%zmm3, %zmm0, %zmm0
	vpaddd	%zmm0, %zmm15, %zmm0
	vpxord	%zmm2, %zmm0, %zmm2
	vpandd	%zmm1, %zmm0, %zmm3
	vprolvd	%zmm3, %zmm2, %zmm2
	vpaddd	%zmm2, %zmm16, %zmm2
	vpxord	%zmm0, %zmm2, %zmm0
	vpandd	%zmm1, %zmm2, %zmm3
	vprolvd	%zmm3, %zmm0, %zmm0
	vpaddd	%zmm0, %zmm17, %zmm0
	vpxord	%zmm2, %zmm0, %zmm2
	vpandd	%zmm1, %zmm0, %zmm3
	vprolvd	%zmm3, %zmm2, %zmm2
	vpaddd	%zmm2, %zmm18, %zmm2
	vpxord	%zmm0, %zmm2, %zmm0
	vpandd	%zmm1, %zmm2, %zmm3
	vprolvd	%zmm3, %zmm0, %zmm0
	vpaddd	%zmm0, %zmm19, %zmm0
	vpxord	%zmm2, %zmm0, %zmm2
	vpandd	%zmm1, %zmm0, %zmm3
	vprolvd	%zmm3, %zmm2, %zmm2
	vpaddd	%zmm2, %zmm20, %zmm2
	vpxord	%zmm0, %zmm2, %zmm0
	vpandd	%zmm1, %zmm2, %zmm3
	vprolvd	%zmm3, %zmm0, %zmm0
	vpaddd	%zmm0, %zmm21, %zmm0
	vpxord	%zmm2, %zmm0, %zmm2
	vpandd	%zmm1, %zmm0, %zmm3
	vprolvd	%zmm3, %zmm2, %zmm2
	vpaddd	%zmm2, %zmm22, %zmm2
	vpxord	%zmm0, %zmm2, %zmm0
	vpandd	%zmm1, %zmm2, %zmm3
	vprolvd	%zmm3, %zmm0, %zmm0
	vpaddd	%zmm0, %zmm23, %zmm0
	vpxord	%zmm2, %zmm0, %zmm2
	vpandd	%zmm1, %zmm0, %zmm3
	vprolvd	%zmm3, %zmm2, %zmm2
	vpaddd	%zmm2, %zmm24, %zmm2
	vpxord	%zmm0, %zmm2, %zmm0
	vpandd	%zmm1, %zmm2, %zmm3
	vprolvd	%zmm3, %zmm0, %zmm0
	vpaddd	%zmm0, %zmm25, %zmm0
	vpxord	%zmm2, %zmm0, %zmm2
	vpandd	%zmm1, %zmm0, %zmm3
	vprolvd	%zmm3, %zmm2, %zmm2
	vpaddd	%zmm2, %zmm30, %zmm2
	vpxord	%zmm0, %zmm2, %zmm0
	vpandd	%zmm1, %zmm2, %zmm3
	vprolvd	%zmm3, %zmm0, %zmm0
	vpaddd	%zmm0, %zmm27, %zmm0
	vpslld	$2, %zmm5, %zmm3
	kxnorw	%k0, %k0, %k1
	vpscatterdd	%zmm0, (%rdi,%zmm3) {%k1}
	vpxord	%zmm2, %zmm0, %zmm2
	vpandd	%zmm1, %zmm0, %zmm0
	vprolvd	%zmm0, %zmm2, %zmm0
	vprold	$3, %zmm26, %zmm1
	vpaddd	%zmm0, %zmm1, %zmm0
	kxnorw	%k0, %k0, %k1
	vpscatterdd	%zmm0, (%rsi,%zmm3) {%k1}
	addq	$56, %rsp
	vzeroupper
	retq
.Lfunc_end0:
	.size	rc5pipe___vyuvyuvyuvyuvyuvyuun_3C_unu_3E_un_3C_unu_3E_avx512skx, .Lfunc_end0-rc5pipe___vyuvyuvyuvyuvyuvyuun_3C_unu_3E_un_3C_unu_3E_avx512skx
                                        # -- End function
	.section	.rodata,"a",@progbits
	.p2align	6               # -- Begin function rc5_16pipe___unuunuunuunuunuunuun_3C_unu_3E_un_3C_unu_3E_avx512skx
.LCPI1_0:
	.long	0                       # 0x0
	.long	1                       # 0x1
	.long	2                       # 0x2
	.long	3                       # 0x3
	.long	4                       # 0x4
	.long	5                       # 0x5
	.long	6                       # 0x6
	.long	7                       # 0x7
	.long	8                       # 0x8
	.long	9                       # 0x9
	.long	10                      # 0xa
	.long	11                      # 0xb
	.long	12                      # 0xc
	.long	13                      # 0xd
	.long	14                      # 0xe
	.long	15                      # 0xf
	.text
	.globl	rc5_16pipe___unuunuunuunuunuunuun_3C_unu_3E_un_3C_unu_3E_avx512skx
	.p2align	4, 0x90
	.type	rc5_16pipe___unuunuunuunuunuunuun_3C_unu_3E_un_3C_unu_3E_avx512skx,@function
rc5_16pipe___unuunuunuunuunuunuun_3C_unu_3E_un_3C_unu_3E_avx512skx: # @rc5_16pipe___unuunuunuunuunuunuun_3C_unu_3E_un_3C_unu_3E_avx512skx
# %bb.0:                                # %allocas
	movq	8(%rsp), %rax
	movq	16(%rsp), %r9
	vpbroadcastd	%edi, %zmm1
	vpbroadcastd	%esi, %zmm2
	vpbroadcastd	%edx, %zmm3
	vpbroadcastd	%ecx, %zmm4
	vpbroadcastd	%r8d, %zmm5
	vmovaps	.LCPI1_0(%rip), %zmm0   # zmm0 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	movq	%rax, %rdi
	movq	%r9, %rsi
	jmp	rc5pipe___vyuvyuvyuvyuvyuvyuun_3C_unu_3E_un_3C_unu_3E_avx512skx # TAILCALL
.Lfunc_end1:
	.size	rc5_16pipe___unuunuunuunuunuunuun_3C_unu_3E_un_3C_unu_3E_avx512skx, .Lfunc_end1-rc5_16pipe___unuunuunuunuunuunuun_3C_unu_3E_un_3C_unu_3E_avx512skx
                                        # -- End function
	.section	.rodata,"a",@progbits
	.p2align	6               # -- Begin function rc5_16pipe_avx512skx
.LCPI2_0:
	.long	0                       # 0x0
	.long	1                       # 0x1
	.long	2                       # 0x2
	.long	3                       # 0x3
	.long	4                       # 0x4
	.long	5                       # 0x5
	.long	6                       # 0x6
	.long	7                       # 0x7
	.long	8                       # 0x8
	.long	9                       # 0x9
	.long	10                      # 0xa
	.long	11                      # 0xb
	.long	12                      # 0xc
	.long	13                      # 0xd
	.long	14                      # 0xe
	.long	15                      # 0xf
	.text
	.globl	rc5_16pipe_avx512skx
	.p2align	4, 0x90
	.type	rc5_16pipe_avx512skx,@function
rc5_16pipe_avx512skx:                   # @rc5_16pipe_avx512skx
# %bb.0:                                # %allocas
	movq	8(%rsp), %rax
	movq	16(%rsp), %r9
	vpbroadcastd	%edi, %zmm1
	vpbroadcastd	%esi, %zmm2
	vpbroadcastd	%edx, %zmm3
	vpbroadcastd	%ecx, %zmm4
	vpbroadcastd	%r8d, %zmm5
	vmovaps	.LCPI2_0(%rip), %zmm0   # zmm0 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	movq	%rax, %rdi
	movq	%r9, %rsi
	jmp	rc5pipe___vyuvyuvyuvyuvyuvyuun_3C_unu_3E_un_3C_unu_3E_avx512skx # TAILCALL
.Lfunc_end2:
	.size	rc5_16pipe_avx512skx, .Lfunc_end2-rc5_16pipe_avx512skx
                                        # -- End function

	.ident	"clang version 8.0.0 (tags/RELEASE_800/final 358773)"
	.section	".note.GNU-stack","",@progbits
