# GNU C++ 2.7.2
# -O6 -fdefer-pop -fomit-frame-pointer -fcse-follow-jumps -fcse-skip-blocks
# -fexpensive-optimizations -fthread-jumps -fstrength-reduce -funroll-loops
# -fpeephole -fforce-mem -ffunction-cse -finline-functions -finline
# -fcaller-saves -fpcc-struct-return -frerun-cse-after-loop -fschedule-insns
# -fschedule-insns2 -fcommon -fgnu-linker -Wunused -Wswitch -mpowerpc
# -mnew-mnemonics

gcc2_compiled.:
        .align 2
        .globl _rc5_unit_func
        .long 0x4008
_rc5_unit_func:
# offsets into the RC5UnitWork structure
.set plain_hi,0
.set plain_lo,4
.set cypher_hi,8
.set cypher_lo,12
.set L0_hi,16
.set L0_lo,20

  moveml	d2-d7/a2-a6,a7@-
  subl	#4*26,a7

  movel	a7@(152),a6
  movel	a6@(L0_lo),d0
  movel	a6@(L0_hi),d1
pass1:
  addl	#0xbf0a8b1d,d0
  movel	#0x5618cb1c+0xbf0a8b1d,d2
  rorl	#3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(1*4)
  addl	d3,d1
   addl	#0xf45044d5,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(2*4)
  addl	d3,d0
   addl	#0x9287be8e,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(3*4)
  addl	d3,d1
   addl	#0x30bf3847,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(4*4)
  addl	d3,d0
   addl	#0xcef6b200,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(5*4)
  addl	d3,d1
   addl	#0x6d2e2bb9,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(6*4)
  addl	d3,d0
   addl	#0x0b65a572,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(7*4)
  addl	d3,d1
   addl	#0xa99d1f2b,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(8*4)
  addl	d3,d0
   addl	#0x47d498e4,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(9*4)
  addl	d3,d1
   addl	#0xe60c129d,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(10*4)
  addl	d3,d0
   addl	#0x84438c56,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,d6
  addl	d3,d1
   addl	#0x227b060f,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,d7
  addl	d3,d0
   addl	#0xc0b27fc8,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a0
  addl	d3,d1
   addl	#0x5ee9f981,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a1
  addl	d3,d0
   addl	#0xfd21733a,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a2
  addl	d3,d1
   addl	#0x9b58ecf3,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a3
  addl	d3,d0
   addl	#0x399066ac,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a4
  addl	d3,d1
   addl	#0xd7c7e065,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a5
  addl	d3,d0
   addl	#0x75ff5a1e,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(19*4)
  addl	d3,d1
   addl	#0x1436d3d7,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(20*4)
  addl	d3,d0
   addl	#0xb26e4d90,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(21*4)
  addl	d3,d1
   addl	#0x50a5c749,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(22*4)
  addl	d3,d0
   addl	#0xeedd4102,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(23*4)
  addl	d3,d1
   addl	#0x8d14babb,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(24*4)
  addl	d3,d0
   addl	#0x2b4c3474,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(25*4)
  addl	d3,d1
   addl	#0xbf0a8b1d,d2
  roll	d3,d1

pass2:
  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(0*4)
  addl	d3,d0
   addl	a7@(1*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(1*4)
  addl	d3,d1
   addl	a7@(2*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(2*4)
  addl	d3,d0
   addl	a7@(3*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(3*4)
  addl	d3,d1
   addl	a7@(4*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(4*4)
  addl	d3,d0
   addl	a7@(5*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(5*4)
  addl	d3,d1
   addl	a7@(6*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(6*4)
  addl	d3,d0
   addl	a7@(7*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(7*4)
  addl	d3,d1
   addl	a7@(8*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(8*4)
  addl	d3,d0
   addl	a7@(9*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(9*4)
  addl	d3,d1
   addl	a7@(10*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(10*4)
  addl	d3,d0
   addl	d6,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,d6
  addl	d3,d1
   addl	d7,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,d7
  addl	d3,d0
   addl	a0,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a0
  addl	d3,d1
   addl	a1,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a1
  addl	d3,d0
   addl	a2,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a2
  addl	d3,d1
   addl	a3,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a3
  addl	d3,d0
   addl	a4,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a4
  addl	d3,d1
   addl	a5,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a5
  addl	d3,d0
   addl	a7@(19*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(19*4)
  addl	d3,d1
   addl	a7@(20*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(20*4)
  addl	d3,d0
   addl	a7@(21*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(21*4)
  addl	d3,d1
   addl	a7@(22*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(22*4)
  addl	d3,d0
   addl	a7@(23*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(23*4)
  addl	d3,d1
   addl	a7@(24*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  movel	d2,d3
  addl	d1,d3
  movel	d2,a7@(24*4)
  addl	d3,d0
   addl	a7@(25*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	d2,d3
  addl	d0,d3
  movel	d2,a7@(25*4)
  addl	d3,d1
   addl	a7@(0*4),d2
  roll	d3,d1

pass3:
  addl	d1,d2
  roll	#3,d2
  movel	a6@(plain_lo),d4
  movel	d2,d3
  addl	d1,d3
  addl	d2,d4
  addl	d3,d0
   addl	a7@(1*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  movel	a6@(plain_hi),d5
  movel	d2,d3
  addl	d0,d3
  addl	d2,d5
  addl	d3,d1
   addl	a7@(2*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  eorl	d5,d4
  movel	d2,d3
  roll	d5,d4
  addl	d1,d3
  addl	d2,d4
  addl	d3,d0
   addl	a7@(3*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  eorl	d4,d5
  movel	d2,d3
  roll	d4,d5
  addl	d0,d3
  addl	d2,d5
  addl	d3,d1
   addl	a7@(4*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  eorl	d5,d4
  movel	d2,d3
  roll	d5,d4
  addl	d1,d3
  addl	d2,d4
  addl	d3,d0
   addl	a7@(5*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  eorl	d4,d5
  movel	d2,d3
  roll	d4,d5
  addl	d0,d3
  addl	d2,d5
  addl	d3,d1
   addl	a7@(6*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  eorl	d5,d4
  movel	d2,d3
  roll	d5,d4
  addl	d1,d3
  addl	d2,d4
  addl	d3,d0
   addl	a7@(7*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  eorl	d4,d5
  movel	d2,d3
  roll	d4,d5
  addl	d0,d3
  addl	d2,d5
  addl	d3,d1
   addl	a7@(8*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  eorl	d5,d4
  movel	d2,d3
  roll	d5,d4
  addl	d1,d3
  addl	d2,d4
  addl	d3,d0
   addl	a7@(9*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  eorl	d4,d5
  movel	d2,d3
  roll	d4,d5
  addl	d0,d3
  addl	d2,d5
  addl	d3,d1
   addl	a7@(10*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  eorl	d5,d4
  movel	d2,d3
  roll	d5,d4
  addl	d1,d3
  addl	d2,d4
  addl	d3,d0
   addl	d6,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  eorl	d4,d5
  movel	d2,d3
  roll	d4,d5
  addl	d0,d3
  addl	d2,d5
  addl	d3,d1
   addl	d7,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  eorl	d5,d4
  movel	d2,d3
  roll	d5,d4
  addl	d1,d3
  addl	d2,d4
  addl	d3,d0
   addl	a0,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  eorl	d4,d5
  movel	d2,d3
  roll	d4,d5
  addl	d0,d3
  addl	d2,d5
  addl	d3,d1
   addl	a1,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  eorl	d5,d4
  movel	d2,d3
  roll	d5,d4
  addl	d1,d3
  addl	d2,d4
  addl	d3,d0
   addl	a2,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  eorl	d4,d5
  movel	d2,d3
  roll	d4,d5
  addl	d0,d3
  addl	d2,d5
  addl	d3,d1
   addl	a3,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  eorl	d5,d4
  movel	d2,d3
  roll	d5,d4
  addl	d1,d3
  addl	d2,d4
  addl	d3,d0
   addl	a4,d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  eorl	d4,d5
  movel	d2,d3
  roll	d4,d5
  addl	d0,d3
  addl	d2,d5
  addl	d3,d1
   addl	a5,d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  eorl	d5,d4
  movel	d2,d3
  roll	d5,d4
  addl	d1,d3
  addl	d2,d4
  addl	d3,d0
   addl	a7@(19*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  eorl	d4,d5
  movel	d2,d3
  roll	d4,d5
  addl	d0,d3
  addl	d2,d5
  addl	d3,d1
   addl	a7@(20*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  eorl	d5,d4
  movel	d2,d3
  roll	d5,d4
  addl	d1,d3
  addl	d2,d4
  addl	d3,d0
   addl	a7@(21*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  eorl	d4,d5
  movel	d2,d3
  roll	d4,d5
  addl	d0,d3
  addl	d2,d5
  addl	d3,d1
   addl	a7@(22*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  eorl	d5,d4
  movel	d2,d3
  roll	d5,d4
  addl	d1,d3
  addl	d2,d4
  addl	d3,d0
   addl	a7@(23*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  eorl	d4,d5
  movel	d2,d3
  roll	d4,d5
  addl	d0,d3
  addl	d2,d5
  addl	d3,d1
   addl	a7@(24*4),d2
  roll	d3,d1

  addl	d1,d2
  roll	#3,d2
  eorl	d5,d4
  roll	d5,d4
  addl	d2,d4
  cmpl	a6@(cypher_lo),d4
  bne	notfound

  movel	d2,d3
  addl	d1,d3
  addl	d3,d0
   addl	a7@(25*4),d2
  roll	d3,d0

  addl	d0,d2
  roll	#3,d2
  eorl	d4,d5
  movel	d2,d3
  roll	d4,d5
  addl	d0,d3
  addl	d2,d5
  addl	d3,d1
  roll	d3,d1

  moveql	#1,d0
  cmpl	a6@(cypher_hi),d5
  beq	exit

notfound:
  moveql	#0,d0
exit:		
  addl	#4*26,a7
  moveml	a7@+,d2-d7/a2-a6
  rts

