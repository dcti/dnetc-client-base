// MMX implementation of Kwan's sboxes
//
// Bruce Ford <b.ford@qut.edu.au>
// Rémi Guyomarch <rguyom@mail.dotcom.fr>
//

//
// $Log: sboxes-mmx.cpp,v $
// Revision 1.5  1998/11/16 15:39:50  remi
// Deleted older sboxes.
//
// Revision 1.4  1998/09/28 22:09:23  remi
// Cleared 3 warnings.
//
// Revision 1.3  1998/07/12 05:29:16  fordbr
// Replaced sboxes 1, 2 and 7 with Kwan versions
// Now 1876 kkeys/s on a P5-200MMX
//
// Revision 1.2  1998/07/08 23:42:28  remi
// Added support for CliIdentifyModules().
//
// Revision 1.1  1998/07/08 15:43:52  remi
// First integration of the MMX bitslicer.
//
//

#if (!defined(lint) && defined(__showids__))
const char *sboxes_mmx_cpp(void) { 
return "@(#)$Id: sboxes-mmx.cpp,v 1.5 1998/11/16 15:39:50 remi Exp $"; }
#endif

#include "sboxes-mmx.h"

// For all these functions :
// - input :
//	a1 - mm0
//	a2 - mm1
//	a3 - mm2
//	a4 - mm3
//	a5 - mm4
//	a6 - mm5
//      mmxParams* - %eax
//
// - output :
//	depends on the function


/*
// Debugging helper

#include <stdio.h>

static void xprint (char *comm, slice a1, slice a2) {
  printf ("%s : %08X%08X %08X%08X %s\n", comm,
	  (unsigned)a1, (unsigned)(a1 >> 32), (unsigned)a2, (unsigned)(a2 >> 32),
	  a1 == a2 ? "good !" : "bad !" );
}
*/

#define mmNOT " 0(%eax)"
#define out1  " 8(%eax)"
#define out2  "16(%eax)"
#define out3  "24(%eax)"
#define out4  "32(%eax)"


// --------------------------------------------------------
// --------------------------------------------------------
void mmxs1_kwan(stMmxParams *)
{

// On exit:
// out1 - mm5
// out2 - mm7
// out3 - mm2
// out4 - mm0

#define a1    "40(%eax)"
#define a3    "48(%eax)"
#define a5    "56(%eax)"
#define x1    "64(%eax)"
#define x3    "72(%eax)"
#define x4    "80(%eax)"
#define x5    "88(%eax)"
#define x6    "96(%eax)"
#define x13  "104(%eax)"
#define x14  "112(%eax)"
#define x25  "120(%eax)"
#define x26  "128(%eax)"
#define x38  "136(%eax)"
#define x55  "144(%eax)"
#define x58  "152(%eax)"

asm ("

   movq  %mm0, "a1"
   movq  %mm3, %mm6      # copy a4

   pxor  "mmNOT", %mm0   # x2 = ~a1
   pxor  %mm2, %mm3      # x3 = a3 ^ a4

   pxor  "mmNOT", %mm6   # x1 = ~a4
   movq  %mm0, %mm7      # copy x2

   movq  %mm4, "a5"
   por   %mm2, %mm7      # x5 = a3 | x2

   movq  %mm3, "x3"
   movq  %mm5, %mm4      # copy a6

   movq  %mm6, "x1"
   pxor  %mm0, %mm3      # x4 = x2 ^ x3

   movq  %mm7, "x5"
   por   %mm6, %mm0      # x9 = x1 | x2

   movq  %mm2, "a3"
   pand  %mm6, %mm7      # x6 = x1 & x5

   movq  %mm3, "x4"
   por   %mm3, %mm2      # x23 = a3 | x4

   pxor  "mmNOT", %mm2   # x24 = ~x23
   pand  %mm0, %mm4      # x10 = a6 & x9

   movq  %mm7, %mm6      # copy x6
   por   %mm5, %mm2      # x25 = a6 | x24

   movq  %mm7, "x6"
   por   %mm5, %mm6      # x7 = a6 | x6

   pxor  %mm2, %mm7      # x26 = x6 ^ x25
   pxor  %mm6, %mm3      # x8 = x4 ^ x7

   movq  %mm2, "x25"
   pxor  %mm4, %mm6      # x11 = x7 ^ x10

   pand  "a3", %mm4      # x38 = a3 & x10
   movq  %mm6, %mm2      # copy x11

   pxor  "a3", %mm6      # x53 = a3 ^ x11
   por   %mm1, %mm2      # x12 = a2 | x11

   pand  "x5", %mm6      # x54 = x5 & x53
   pxor  %mm3, %mm2      # x13 = x8 ^ x12

   movq  %mm4, "x38"
   pxor  %mm2, %mm0      # x14 = x9 ^ x13

   movq  %mm7, "x26"
   movq  %mm5, %mm4      # copy a6

   movq  %mm2, "x13"
   por   %mm0, %mm4      # x15 = a6 | x14

   movq  "x1", %mm7
   por   %mm1, %mm6      # x55 = a2 | x54

   movq  %mm0, "x14"
   movq  %mm3, %mm2      # copy x8

   pandn "x3", %mm0      # x18 = x3 & ~x14
   pxor  %mm7, %mm4      # x16 = x1 ^ x15

   por   "x4", %mm5      # x57 = a6 | x4
   por   %mm1, %mm0      # x19 = a2 | x18

   pxor  "x38", %mm5     # x58 = x38 ^ x57
   pxor  %mm0, %mm4      # x20 = x16 ^ x19

   movq  "a5", %mm0
   pand  %mm7, %mm2      # x27 = x1 & x8

   movq  %mm6, "x55"
   por   %mm1, %mm2      # x28 = a2 | x27

   movq  "x14", %mm6
   por   %mm4, %mm0      # x21 = a5 | x20

   pand  "x5", %mm6      # x32 = x5 & x14
   por   %mm3, %mm7      # x30 = x1 | x8

   movq  %mm5, "x58"
   pxor  %mm3, %mm6      # x33 = x8 ^ x32

   pxor  "x6", %mm7      # x31 = x6 ^ x30
   movq  %mm1, %mm5      # copy a2

   pxor  "x26", %mm2     # x29 = x26 ^ x28
   pand  %mm6, %mm5      # x34 = a2 & x33

   pand  "a3", %mm6      # x40 = a3 & x33
   pxor  %mm7, %mm5      # x35 = x31 ^ x34

   por   "a5", %mm5      # x36 = a5 | x35

   movq  "a1", %mm7
   pxor  %mm2, %mm5      # x37 = x29 ^ x36

   movq  "x4", %mm2
   por   %mm3, %mm7      # x46 = a1 | x8

   por   "x38", %mm2     # x39 = x4 | x38
   pxor  %mm6, %mm3      # x52 = x8 ^ x40

   pxor  "x25", %mm6     # x41 = x25 ^ x40
   pxor  %mm4, %mm7      # x47 = x46 ^ x20

   movq  "a3", %mm4
   por   %mm1, %mm7      # x48 = a2 | x47

   por   "x26", %mm4     # x44 = a3 | x26
   por   %mm1, %mm6      # x42 = a2 | x41

   pxor  "x14", %mm4     # x45 = x14 ^ x44
   pxor  %mm2, %mm6      # x43 = x39 ^ x42

   movq  "x13", %mm2
   pxor  %mm4, %mm7      # x49 = x45 ^ x48

   pxor  "x55", %mm3     # x56 = x52 ^ x55
   pxor  %mm2, %mm0      # x22 = x13 ^ x21

   pxor  "out1", %mm5    # out1 ^= x37
   pand  %mm3, %mm2      # x59 = x13 & x56

   movq  "a5", %mm4
   pand  %mm1, %mm2      # x60 = a2 & x59

   pxor  "x58", %mm2     # x61 = x58 ^ x60
   pand  %mm4, %mm7      # x50 = a5 & x49

   pxor  "out4", %mm0    # out4 ^= x22
   pand  %mm4, %mm2      # x62 = a5 & x61

   pxor  %mm6, %mm7      # x51 = x43 ^ x50
   pxor  %mm3, %mm2      # x63 = x56 ^ x62

   pxor  "out2", %mm7    # out2 ^= x51

   pxor  "out3", %mm2    # out3 ^= x63

                         # 51 clocks for 67 variables
");
#undef a1
#undef a3
#undef a5
#undef x1
#undef x3
#undef x4
#undef x5
#undef x6
#undef x13
#undef x14
#undef x25
#undef x26
#undef x38
#undef x55
#undef x58
}


// --------------------------------------------------------
// --------------------------------------------------------
void mmxs2_kwan(stMmxParams *)
{

// On exit:
// out1 - mm1
// out2 - mm5
// out3 - mm7
// out4 - mm2

#define a1    "0x28(%eax)"
#define a2    "0x30(%eax)"
#define a3    "0x38(%eax)"
#define a4    "0x40(%eax)"
#define x3    "0x48(%eax)"
#define x4    "0x50(%eax)"
#define x5    "0x58(%eax)"
#define x13   "0x60(%eax)"
#define x18   "0x68(%eax)"
#define x25   "0x70(%eax)"

asm ("

   movq  %mm3, "a4"
   movq  %mm4, %mm6      # copy a5

   movq  %mm0, "a1"
   movq  %mm4, %mm7      # copy a5

   pxor  "mmNOT", %mm0   # x2 = ~a1
   pxor  %mm5, %mm6      # x3 = a5 ^ a6

   pxor  "mmNOT", %mm7   # x1 = ~a5
   movq  %mm0, %mm3      # copy x2

   movq  %mm2, "a3"
   por   %mm5, %mm7      # x6 = a6 | x1

   movq  %mm6, "x3"
   por   %mm7, %mm3      # x7 = x2 | x6

   pxor  %mm4, %mm7      # x13 = a5 ^ x6
   pxor  %mm0, %mm6      # x4 = x2 ^ x3

   pand  %mm1, %mm3      # x8 = a2 & x7
   por   %mm7, %mm2      # x14 = a3 | x13

   movq  %mm1, "a2"
   pxor  %mm5, %mm3      # x9 = a6 ^ x8

   movq  %mm6, "x4"
   pxor  %mm1, %mm6      # x5 = a2 ^ x4

   movq  %mm7, "x13"
   pand  %mm3, %mm1      # x12 = a2 & x9

   pand  "a3", %mm3      # x10 = a3 & x9
   pxor  %mm2, %mm1      # x15 = x12 ^ x14

   movq  "x4", %mm7
   movq  %mm1, %mm2      # copy x15

   pand  "a4", %mm2      # x16 = a4 & x15
   pxor  %mm6, %mm3      # x11 = x5 ^ x10

   movq  %mm6, "x5"
   pxor  %mm2, %mm3      # x17 = x11 ^ x16

   movq  "a1", %mm2
   por   %mm5, %mm7      # x22 = a6 | x4

   por   %mm2, %mm1      # x40 = a1 | x15
   pand  %mm3, %mm7      # x23 = x17 & x22

   pxor  "out2", %mm3    # out2 ^= x17
   por   %mm4, %mm2      # x18 = a1 | a5

   por   "a3", %mm7      # x24 = a3 | x23
   movq  %mm2, %mm6      # copy x18

   pxor  "x13", %mm1     # x41 = x13 ^ x40
   por   %mm5, %mm6      # x19 = a6 | x18

   movq  %mm3, "out2"
   pand  %mm0, %mm4      # x27 = a5 & x2

   movq  "x13", %mm3
   por   %mm0, %mm5      # x26 = a6 | x2

   movq  %mm2, "x18"
   pxor  %mm6, %mm3      # x20 = x13 ^ x19

   movq  "a2", %mm2
   pxor  %mm6, %mm0      # x31 = x2 ^ x19

   pxor  %mm2, %mm3      # x21 = a2 ^ x20
   pand  %mm2, %mm0      # x32 = a2 & x31

   pxor  %mm3, %mm7      # x25 = x21 ^ x24
   por   %mm4, %mm2      # x28 = a2 | x27

   pxor  "x3", %mm4      # x30 = x3 ^ x27
   pand  %mm3, %mm6      # x47 = x19 & x21

   pxor  %mm0, %mm4      # x33 = x30 ^ x32
   pxor  %mm5, %mm6      # x48 = x26 ^ x47

   movq  %mm7, "x25"
   pand  %mm3, %mm0      # x38 = x21 & x32

   movq  "a3", %mm7
   pxor  %mm2, %mm5      # x29 = x26 ^ x28

   pxor  "x5", %mm0      # x39 = x5 ^ x38
   pand  %mm4, %mm7      # x34 = a3 & x33

   pand  "a2", %mm4      # x49 = a2 & x33
   pxor  %mm5, %mm7      # x35 = x29 ^ x34

   por   "a4", %mm7      # x36 = a4 | x35
   movq  %mm1, %mm5      # copy x41

   por   "a3", %mm5      # x42 = a3 | x41
   por   %mm2, %mm1      # x44 = x28 | x41

   pand  "x18", %mm2     # x53 = x18 & x28
   pxor  %mm3, %mm4      # x50 = x21 ^ x49

   movq  "a4", %mm3
   pand  %mm4, %mm2      # x54 = x50 & x53

   pand  "a3", %mm4      # x51 = a3 & x50
   pxor  %mm5, %mm0      # x43 = x39 ^ x42

   pxor  "x25", %mm7     # x37 = x25 ^ x36
   pxor  %mm6, %mm4      # x52 = x48 ^ x51

   pxor  "out3", %mm7    # out3 ^= x37
   pand  %mm3, %mm1      # x45 = a4 & x44

   movq  "out2", %mm5
   pxor  %mm0, %mm1      # x46 = x43 ^ x45

   pxor  "out1", %mm1    # out1 ^= x46
   por   %mm3, %mm2      # x55 = a4 | x54

   pxor  %mm4, %mm2      # x56 = x52 ^ x55

   pxor  "out4", %mm2    # out4 ^= x56

                         # 44 clocks for 60 variables
");
#undef a1
#undef a2
#undef a3
#undef a4
#undef x3
#undef x4
#undef x5
#undef x13
#undef x18
#undef x25
}

// --------------------------------------------------------
// --------------------------------------------------------
void mmxs3_kwan(stMmxParams *)
{

// On exit:
// out1 - mm2
// out2 - mm6
// out3 - mm3
// out4 - mm7

// 49 cycles
// 61 operations   - 1.24 operations per cycle
// 57 variables    - 1.16 variables per cycle
// 90 instructions - 1.84 instructions per cycle

  /*
  slice xa1, xa2, xa3, xa4, xa5, xa6;

  asm ("
      movq %%mm0, %0
      movq %%mm1, %1
      movq %%mm2, %2
      movq %%mm3, %3
      movq %%mm4, %4
      movq %%mm5, %5

  ": : "m" (xa1), "m" (xa2), "m" (xa3), "m" (xa4), "m" (xa5), "m" (xa6));
  */

#define a1    "0x28(%eax)"
#define x2    "0x30(%eax)"
#define x9    "0x38(%eax)"
#define a5    "0x40(%eax)"
#define x4    "0x48(%eax)"
#define a6    "0x50(%eax)"
#define x6    "0x58(%eax)"
#define x5    "0x60(%eax)"
#define x11   "0x68(%eax)"
#define x12   "0x70(%eax)"
#define x13   "0x78(%eax)"
#define x54   "0x80(%eax)"
#define x7    "0x88(%eax)"
#define a4    "0x90(%eax)"
#define a3  a5
#define x38 x4

asm ("
						# mm6 free
						# mm7 free
	movq	%mm0, "a1"	#		# mm0 free
	movq	%mm5, %mm6		# mm6 = a6
	pxor	"mmNOT",%mm6	#	# mm6(x2) = ~a6
	movq	%mm4, %mm7		# mm7 = a5
	pxor	%mm6, %mm7	#	# mm7(x9) = a5 ^ x2
	movq	%mm4, %mm0		# mm0 = a5
	movq	%mm6, "x2"	#		# mm6 free
	pand	%mm2, %mm0		# mm0(x3) = a5 & a3
	movq	%mm7, "x9"	#		# mm7 free
	pxor	%mm5, %mm0		# mm0(x4) = x3 ^ a6
	movq	%mm4, "a5"	#		# mm4 free
	pandn	%mm3, %mm4		# mm4(x5) = a4 & ~a5
	movq	%mm0, "x4"	#		# mm0 free
	por	%mm3, %mm7		# mm7(x10) = a4 | x9
	movq	"a5", %mm6	#	# mm6 = a5
	pxor	%mm4, %mm0		# mm0(x6) = x4 ^ x5
	movq	%mm5, "a6"	#		# mm5 free
	pandn	%mm2, %mm6		# mm6(x8) = a3 & ~a5
	movq	%mm0, "x6"	#		# mm0 free
	pxor	%mm6, %mm7		# mm7(x11) = x8 ^ x10
	movq	"x2", %mm5	#	# mm5 = x2
	pxor	%mm1, %mm0		# mm0(x7) = x6 ^ a2
	movq	%mm4, "x5"	#		# mm4 free
	movq	%mm7, %mm4		# mm4 = x11
	por	"x4", %mm5	#	# mm5(x23) = x1 | x4
	pand	%mm0, %mm4		# mm4(x12) = x7 & x11
	movq	%mm7, "x11"	#		# mm7 free
	pxor	%mm5, %mm6		# mm6(x24) = x23 ^ x8
	pxor	"a5", %mm7	#	# mm7(x13) = a5 ^ x11
	por	%mm1, %mm6		# mm6(x25) = a2 | x24
	movq	%mm4, "x12"	#		# mm4 free
	pand	%mm5, %mm4		# mm4(x54) = x12 & x23
	movq	%mm7, "x13"	#		# mm7 free
	por	%mm0, %mm7		# mm7(x14) = x13 | x7
	movq	%mm4, "x54"	#		# mm4 free
	movq	%mm2, %mm4		# mm4 = a3
	pxor	"x9", %mm4	#	# mm4 = a3 ^ x21
	pand	%mm3, %mm7		# mm7(x15) = a4 & x14
	movq	%mm0, "x7"	#		# mm0 free
	pxor	%mm3, %mm4		# mm4(x22) = a4 ^ a3 ^ x9
	pxor	"a6", %mm5	#	# mm5(x27) = a6 ^ x23
	pxor	%mm4, %mm6		# mm6(x26) = x22 ^ x25
						# mm4 free
	movq	%mm3, "a4"	#		# mm3 free
	por	%mm5, %mm3		# mm3(x28) = x27 | a4
	movq	%mm2, "a3"	#		# mm2 free
	pxor	%mm3, %mm5		# mm5(x51) = x27 ^ x28
	por	%mm1, %mm5	#	# mm5(x52) = x51 | a2
	pxor	%mm7, %mm2		# mm2(x29) = a3 ^ x15
	pxor	"x12", %mm7	#	# mm7(x16) = x12 ^ x15
	movq	%mm2, %mm4		# mm4 = x29
	por	"x5", %mm2	#	# mm2(x30) = x29 | x5
	pand	%mm1, %mm7		# mm7(x17) = a2 & x16
	por	"x4", %mm4	#	# mm4(x37) = x29 | x4
	por	%mm1, %mm2		# mm2(x31) = a2 | x30
	pxor	"x11", %mm7	#	# mm7(x18) = x17 ^ x11
	pxor	%mm3, %mm2		# mm2(x32) = x31 ^ x28
						# mm3 free
	movq	"a1", %mm3	#	# mm1 = a3
	pxor	"a4", %mm4	#	# mm4(x38) = x37 ^ a4
	pand	%mm3, %mm7		# mm7(x19) = x18 & a1
	pxor	"x7", %mm7	#	# mm7(x20) = x19 ^ x7
	por	%mm3, %mm2		# mm2(x33) = a1 | x32
	movq	%mm4, "x38"	#		# mm4 free
	pxor	%mm6, %mm2		# mm2(x34) = x26 ^ x33
						# mm6 free
	pxor	"out4",%mm7	#	### mm7(out4) = out4 ^ x20
	por	%mm1, %mm4		# mm4(x39) = a2 | x38
	movq	"a3", %mm6	#	# mm6 = a3
	movq	%mm2, %mm3		# mm3 = x34
	pxor	"x9", %mm6	#	# mm6(x35) = a3 ^ x9
	por	"x5", %mm6	#	# mm6(x36) = x5 | x35
	pxor	"x38", %mm3	#	# mm3(x43) = x34 ^ x38
	pxor	%mm6, %mm4		# mm4(x40) = x36 ^ x39
	movq	"a6", %mm6	#	# mm6 = a6
	pand	"x11", %mm6	#	# mm6(x41) = a6 & x11
	movq	"x2", %mm0	#	# mm0 = x2
	pxor	%mm6, %mm3		# mm3(x44) = x43 ^ x41
	por	"x6", %mm6	#	# mm6(x42) = x41 | x6
	pand	%mm1, %mm3		# mm3(x45) = x42 & a2
	por	"x38", %mm0	#	# mm0(x49) = x2 | x38
	pxor	%mm6, %mm3		# mm3(x46) = x42 ^ x45
						# mm6 free
	pxor	"x13", %mm0	#	# mm0(x50) = x49 ^ x13
	movq	%mm5, %mm6		# mm6 = x52
	por	"a1", %mm3	#	# mm3(x47) = x46 | a1
	pxor	%mm5, %mm0		# mm0(x53) = x50 ^ x52
	pand	"x54", %mm6	#	# mm6(x55) = x52 & x54
	pxor	%mm4, %mm3		# mm3(x48) = x40 ^ x47
	por	"a1", %mm6	#	# mm6(x56) = a1 | x55
	pxor	"out3", %mm3	#	### mm3(out3) = out3 ^ x48
	pxor	%mm0, %mm6		# mm6(x57) = x53 ^ x56
	pxor	"out1",%mm2	#	### mm2(out1) = out1 ^ x34
	pxor	"out2",%mm6	#	### mm6(out2) = out2 ^ x57
");

#undef a1
#undef x2
#undef x9
#undef a5
#undef x4
#undef a6
#undef x6
#undef x5
#undef x11
#undef x12
#undef x13
#undef x54
#undef x7
#undef a4
#undef a3
#undef x38
 
/*   
  register slice *ebp asm ("%eax");

 {
   slice	 x1,  x2,  x3,  x4,  x5,  x6,  x7,  x8;
   slice	 x9, x10, x11, x12, x13, x14, x15, x16;
   slice	x17, x18, x19, x20, x21, x22, x23, x24;
   slice	x25, x26, x27, x28, x29, x30, x31, x32;
   slice	x33, x34, x35, x36, x37, x38, x39, x40;
   slice	x41, x42, x43, x44, x45, x46, x47, x48;
   slice	x49, x50, x51, x52, x53, x54, x55, x56;
   slice	x57;

   slice *xout1 = ebp + 1;
   slice *xout2 = ebp + 2;
   slice *xout3 = ebp + 3;
   slice *xout4 = ebp + 4;

   slice out1, out2, out3, out4;

#define a1 xa1
#define a2 xa2
#define a3 xa3
#define a4 xa4
#define a5 xa5
#define a6 xa6

   x5 = a4 & ~a5;	x2 = ~a6;		x3 = a5 & a3;		x21 = a3 ^ a4;
   x8 = a3 & ~a5;	x9 = a5 ^ x2;		x4 = x3 ^ a6;
   x23 = x2 | x4;	x10 = a4 | x9;		x6 = x4 ^ x5;		x22 = x21 ^ x9;
   x24 = x23 ^ x8;	x11 = x8 ^ x10;		x7 = x6 ^ a2;		x27 = a6 ^ x23;
   x25 = a2 | x24;	x13 = a5 ^ x11;		x12 = x7 & x11;		x28 = x27 | a4;
   x26 = x22 ^ x25;	x14 = x13 | x7;		x54 = x12 & x23;	x35 = a3 ^ x9;
   x51 = x27 ^ x28;	x15 = a4 & x14;					x36 = x35 | x5;
   x52 = a2 | x51;	x16 = x12 ^ x15;	x29 = a3 ^ x15;		x41 = a6 & x11;
   x37 = x4 | x29;	x17 = a2 & x16;		x30 = x29 | x5;		x42 = x41 | x6;
   x38 = x37 ^ a4;	x18 = x11 ^ x17;	x31 = a2 | x30;
   x39 = a2 | x38;	x19 = a1 & x18;		x32 = x28 ^ x31;	x49 = x2 | x38;
   x40 = x36 ^ x39;	x20 = x7 ^ x19;		x33 = a1 | x32;		x50 = x49 ^ x13;
			out4 = *xout4 ^ x20;	x34 = x26 ^ x33;	x53 = x50 ^ x52;
   x43 = x34 ^ x38;				out1 = *xout1 ^ x34;	x55 = x54 & x52;
   x44 = x43 ^ x41;							x56 = a1 | x55;
   x45 = a2 & x44;							x57 = x53 ^ x56;
   x46 = x42 ^ x45;							out2 = *xout2 ^ x57;
   x47 = a1 | x46;
   x48 = x40 ^ x47;
   out3 = *xout3 ^ x48;
   
   //slice zzout1, zzout3;
   //asm ("movq %%mm2, %0": : "m"(zzout1));
   //asm ("movq %%mm3, %0": : "m"(zzout3));

   asm ("movq %0, %%mm2": : "m" (out1));
   asm ("movq %0, %%mm6": : "m" (out2));   
   asm ("movq %0, %%mm3": : "m" (out3));
   asm ("movq %0, %%mm7": : "m" (out4));

   //   xprint (" a1",  a1, *(ebp+ 5));
 }
*/
}



// --------------------------------------------------------
// --------------------------------------------------------
void mmxs4_kwan(stMmxParams *)
{

// On exit:
// out1 - mm1
// out2 - mm0
// out3 - mm6
// out4 - mm4

#define a2    "40(%eax)"
#define a3    "48(%eax)"
#define a4    "56(%eax)"
#define a6    "64(%eax)"

asm ("
   movq  %mm5, "a6"
   movq  %mm2, %mm6      # copy a3

   movq  %mm3, "a4"
   movq  %mm0, %mm7      # copy a1

   movq  %mm1, "a2"
   por   %mm0, %mm6      # x3 = a1 | a3

   pand  %mm4, %mm7      # x8 = a1 & a5
   movq  %mm1, %mm3      # copy a2

   movq  %mm2, "a3"
   movq  %mm4, %mm5      # copy a5

   pand  %mm6, %mm5      # x4 = a5 & x3
   por   %mm2, %mm3      # x6 = a2 | a3

   pxor  "mmNOT", %mm2   # x2 = ~a3
   pxor  %mm5, %mm0      # ~x5 = a1 ^ x4

   pxor  "mmNOT", %mm0   # x5 = ~(~x5)
   pxor  %mm7, %mm6      # x9 = x8 ^ x3

   pxor  %mm0, %mm3      # x7 = x5 ^ x6
   movq  %mm1, %mm7      # copy a2

   pand  %mm6, %mm7      # x10 = a2 & x9
   pxor  %mm2, %mm5      # x14 = x2 ^ x4

   pxor  %mm4, %mm2      # x18 = a5 ^ x2
   pand  %mm5, %mm0      # x17 = x5 & x14

   pxor  %mm7, %mm4      # x11 = a5 ^ x10
   pand  %mm1, %mm5      # x15 = a2 & x14

   por   %mm1, %mm2      # x19 = a2 | x18
   pxor  %mm6, %mm5      # x16 = x9 ^ x15

   movq  "a4",  %mm1     # retrieve a4
   movq  %mm0, %mm6      # copy x17

   pand  %mm4, %mm1      # x12 = a4 & x11
   pxor  %mm2, %mm6      # x20 = x17 ^ x19

   por   "a4",  %mm6     # x21 = a4 | x20
   pxor  %mm3, %mm1      # x13 = x7 ^ x12

   pand  "a2",  %mm4     # x28 = a2 & x11
   pxor  %mm5, %mm6      # x22 = x16 ^ x21

   movq  "a6",  %mm3     # retrieve a6
   pxor  %mm0, %mm4      # x29 = x28 ^ x17

   pxor  "a3",  %mm7     # x30 = a3 ^ x10
   movq  %mm3, %mm0      # copy a6

   pxor  %mm2, %mm7      # x31 = x30 ^ x19
   pand  %mm6, %mm0      # x23 = a6 & x22

   movq  "a4",  %mm2     # retrieve a4
   por   %mm3, %mm6      # x26 = a6 | x22

   pxor  %mm1, %mm0      # x24 = x13 ^ x23
   pand  %mm2, %mm7      # x32 = a4 & x31

   pxor  "mmNOT", %mm1   # x25 = ~x13
   pxor  %mm7, %mm4      # x33 = x29 ^ x32

   movq  %mm4, %mm5      # copy x33
   pxor  %mm1, %mm4      # x34 = x25 ^ x33

   pxor  "out1", %mm1    # out1 ^= x25
   por   %mm4, %mm2      # x37 = a4 | x34

   pand  "a2",  %mm4     # x35 = a2 & x34
   pxor  %mm6, %mm1      # out1 ^= x26

   pxor  %mm0, %mm4      # x36 = x24 ^ x35

   pxor  "out2", %mm0    # out2 ^= x24
   pxor  %mm4, %mm2      # x38 = x36 ^ x37

   pand  %mm2, %mm3      # x39 = a6 & x38
   pxor  %mm2, %mm6      # x41 = x26 ^ x38

   pxor  %mm3, %mm5      # x40 = x33 ^ x39

   pxor  %mm5, %mm6      # x42 = x41 ^ x40

   pxor  "out4", %mm5    # out4 ^= x40

   pxor  "out3", %mm6    # out3 ^= x42
");

#undef a2
#undef a3
#undef a4
#undef a6
}



// --------------------------------------------------------
// --------------------------------------------------------
void mmxs5_kwan (stMmxParams *)
{

// On exit:
// out1 - mm5
// out2 - mm7
// out3 - mm6
// out4 - mm4

// based on Kwan non-std sbox : 60 operations after some manipulations
// to use more 'pandn'.
// 52 cycles for 60 operations : hmmm...

  /*  slice xa1, xa2, xa3, xa4, xa5, xa6;

   asm ("
      movq %%mm0, %0;
      movq %%mm1, %1;
      movq %%mm2, %2;
      movq %%mm3, %3;
      movq %%mm4, %4;
      movq %%mm5, %5;

   ": : "m" (xa1), "m" (xa2), "m" (xa3), "m" (xa4), "m" (xa5), "m" (xa6));
  */

#define a1    "0x28(%eax)"
#define a2    "0x30(%eax)"
#define a6    "0x38(%eax)"
#define x2    "0x40(%eax)"
#define x4    "0x48(%eax)"
#define x5    "0x50(%eax)"
#define x6    "0x58(%eax)"
#define x7    "0x60(%eax)"
#define x8    "0x68(%eax)"
#define x9    "0x70(%eax)"
#define x13   "0x78(%eax)"
#define x16   "0x80(%eax)"
#define x17   a6  //"0x88(%eax)"
#define x21   x7  //"0x90(%eax)"
#define x24   x8  //"0x98(%eax)"
#define x28   x17 //"0xA0(%eax)"
#define x38   x9  //"0xA8(%eax)"

asm ("
movq	%mm1, "a2"	#		# mm1 free
movq	%mm3, %mm6		# mm6 = a4
movq	%mm2, %mm7	#	# mm7 = a3
pandn	%mm2, %mm6		# mm6(x1) = a3 & ~a4
pandn	%mm0, %mm7	#	# mm7(x3) = a1 & ~a3
movq	%mm6, %mm1		# mm1 = x1
movq	%mm0, "a1"	#		# mm0 free
pxor	%mm0, %mm1		# mm1(x2) = x1 ^ a1
pxor	%mm3, %mm0	#	# mm0(x6) = a4 ^ a1
movq	%mm1, "x2"	#		# mm1 free
por	%mm0, %mm6		# mm6(x7) = x1 | x6
movq	%mm5, "a6"	#		# mm5 free
por	%mm7, %mm5		# mm5(x4) = a6 | x3
movq	%mm6, "x7"	#		# mm6 free
pxor	%mm5, %mm1		# mm1(x5) = x2 ^ x4
movq	%mm5, "x4"	#		# %mm5 free
pand	%mm2, %mm6		# mm6 = a3 & x7
movq	"a6", %mm5	#	# mm5 = a6
pxor	%mm3, %mm6		# mm6(x13) = (a3 & x7) ^ a4
pandn	"x7", %mm5	#	# mm5(x8) = x7 & ~a6
movq	%mm0, "x6"	#		# mm0 free
movq	%mm7, %mm0		# mm0 = x3
movq	%mm5, "x8"	#		# mm5 free
pxor	%mm2, %mm5		# mm5(x9) = a3 ^ x8
movq	%mm1, "x5"	#		# mm1 free
pxor	%mm3, %mm0		# mm0 = x3 ^ a4
movq	%mm5, "x9"	#		# mm5 free
pandn	%mm6, %mm7		# mm7 = x13 & ~x3
por	"a6", %mm0	#	# mm0(x16) = a6 | (x3 ^ a4)
por	%mm4, %mm5		# mm5 = a5 | x9
movq	%mm6, "x13"	#		# mm6 free
pxor	%mm1, %mm5		# mm5 = x5 ^ (a5 | x9)
movq	%mm0, "x16"	#		# mm0 free
pxor	%mm0, %mm7		# mm7(x17) = x16 ^ (x13 & ~x3)
movq	"a2", %mm0	#	# mm0 = a2
movq	%mm4, %mm1		# mm1 = a5
movq	%mm7, "x17"	#		# mm7 free
por	%mm7, %mm1		# mm1 = a5 | x17
pand	"x5", %mm7	#	# mm7(x31) = x17 & x5
pxor	%mm6, %mm1		# mm1(x19) = x13 ^ (a5 | x17)
pandn	%mm1, %mm0	#	# mm0 = x19 & ~a2
movq	%mm7, %mm6		# mm6 = x31
pandn	"x7", %mm6	#	# mm6(x32) = x7 & ~x31
pxor	%mm0, %mm5		# mm5(x21) = x5 ^ (a5 | x9) ^ (x19 & ~a2)
pxor	"x9", %mm7	#	# mm7(x38) = x9 ^ x32
movq	%mm3, %mm0		# mm0 = a4
movq	%mm5, "x21"	#		# mm5 free
movq	%mm6, %mm5		# mm5 = x32
pandn	"x8", %mm0	#	# mm0 = x8 & ~a4
pandn	%mm1, %mm5		# mm5(x43) = x19 & ~x32
					# mm1 free
pxor	"out3",%mm6	#	# mm6 = out3 ^ x32
pxor	%mm2, %mm0		# mm0(x34) = (x8 & ~a4) ^ a3
					# mm2 free (no more references to a3)
movq	"a1", %mm2	#	# mm2 = a1		# 'a1' local var free
movq	%mm0, %mm1		# mm1 = x34
pxor	"x9", %mm2	#	# mm2(x24) = a1 ^ x9
pand	%mm4, %mm1		# mm1 = x34 & a5
movq	%mm7, "x38"	#		# mm7 free
pxor	%mm1, %mm6		# mm6 = out3 ^ x32 ^ (x34 & a5)
					# mm1 free
movq	"x4", %mm1	#	# mm1 = x4
movq	%mm2, %mm7		# mm7 = x24
pand	"x2", %mm7	#	# mm7 = x2 & x24
pand	%mm3, %mm1		# mm1 = a4 & x4
pxor	"x17",%mm1	#	# mm1 = (a4 & x4) ^ x17
pandn	%mm4, %mm7		# mm7 = a5 & ~(x2 & x24)
movq	%mm2, "x24"	#		# mm2 free
pxor	%mm7, %mm1		# mm1(x27) = (a4 & x4) ^ x17 ^ (x2 & x24)
					# mm7 free
movq	"out2",%mm7	#	# mm7 = out2
por	%mm2, %mm3		# mm3(x28) = a4 | x24
movq	"a2", %mm2	#	# mm2 = a2
pxor	%mm1, %mm7		# mm7 = out2 ^ x27
movq	%mm3, "x28"	#		# mm3 free
pandn	%mm3, %mm2		# mm2 = x28 & ~a2
movq	"x38",%mm3	#	# mm3 = x38
pxor	%mm2, %mm7		### mm7(out2) = out2 ^ x27 ^ (x28 & ~a2)
					# mm2 free
movq	"x16",%mm2	#	# mm2 = x16
por	%mm4, %mm3		# mm3 = x38 | a5
por	"x13",%mm2	#	# mm2 = x13 | x16
por	%mm5, %mm1		# mm1 = x27 | x43
pxor	"out1",%mm5	#	# mm5 = out1 ^ x43
pxor	%mm3, %mm2		# mm2 = (x13 | x16) ^ (x38 | a5)
					# mm3 free
por	"a2", %mm2	#	# mm2 = a2 | ((x13 | x16) ^ (x38 | a5))
pxor	"x6", %mm1	#	# mm1 = (x27 | x43) ^ x6
pxor	%mm2, %mm6		# mm6 = out3 ^ x32 ^ (x34 & a5) ^ (a2 | ((x13 | x16) ^ (x38 | a5)))
					# mm2 free
pxor	"mmNOT",%mm6	#	### mm6(out3) = out3 ^ x32 ^ (x34 & a5) ^ ~(a2 | ((x13 | x16) ^ (x38 | a5)))
pandn	%mm4, %mm1		# mm1 = a5 & ~((x27 | x43) ^ x6)
movq	"x38",%mm2	#	# mm2 = x38
pxor	"x24",%mm1	#	# mm1 = x24 ^ (a5 & ~((x27 | x43) ^ x6))
movq	%mm2, %mm3		# mm3 = x38
pxor	"x21",%mm2	#	# mm2 = x21 ^ x38
pxor	%mm1, %mm5		# mm5 = out1 ^ x43 ^ x24 ^ (a5 & ~((x27 | x43) ^ x6))
pand	"x6", %mm3	#	# mm3 = x6 & x38
pandn	%mm4, %mm2		# mm2 = a5 & ~(x21 ^ x38)
pand	"x28",%mm2	#	# mm2 = a5 & x28 & ~(x21 ^ x38)
pxor	%mm0, %mm3		# mm3 = (x6 & x38) ^ x34
movq	"x21",%mm4	#	# mm4 = x21
pxor	%mm2, %mm3		# mm3 = (x6 & x38) ^ x34 ^ (a5 & x28 & ~(x21 ^ x38))
por	"a2", %mm3	#	# mm3 = a2 | ((x6 & x38) ^ x34 ^ (a5 & x28 & ~(x21 ^ x38)))
pxor	"out4",%mm4	#	### mm4(out4) = out4 ^ x21
pxor	%mm3, %mm5		### mm5(out1) = ...
");
	
#undef a1
#undef a2
#undef a6
#undef x2
#undef x4
#undef x5
#undef x6
#undef x7
#undef x8
#undef x9
#undef x13
#undef x16
#undef x17
#undef x21
#undef x24
#undef x28
#undef x38

/*
 register slice *ebp asm ("%eax");
 
// 60 ops -> 52 cycles -> 1.15 op per cycle
// previous :66 cycles

 {
   slice  x1,  x2,  x3,  x4,  x5,  x6,  x7,  x8;
   slice  x9, x10, x11, x12, x13, x14, x15, x16;
   slice x17, x18, x19, x20, x21, x22, x23, x24;
   slice x25, x26, x27, x28, x29, x30, x31, x32;
   slice x33, x34, x35, x36, x37, x38, x39, x40;
   slice x41, x42, x43, x44, x45, x46, x47, x48;
   slice x49, x50, x51, x52, x53, x54, x55, x56;
   slice x57, x58, x59, x60, x61, x62;

   slice *xout1 = ebp + 1;
   slice *xout2 = ebp + 2;
   slice *xout3 = ebp + 3;
   slice *xout4 = ebp + 4;

   slice out1, out2, out3, out4;

   #define a1 xa1
   #define a2 xa2
   #define a3 xa3
   #define a4 xa4
   #define a5 xa5
   #define a6 xa6

   x1 = a3 & ~a4;
   x2 = x1 ^ a1;
   x3 = a1 & ~a3;
   x4 = a6 | x3;
   x5 = x2 ^ x4;
   x6 = a4 ^ a1;
   x7 = x6 | x1;
   x8 = x7 & ~a6;
   x9 = a3 ^ x8;
   x13 = (a3 & x7) ^ a4;
   x16 = a6 | (a4 ^ x3);
   x17 = (x13 & ~x3) ^ x16;
   x19 = x13 ^ (a5 | x17);
   x21 = x5 ^ (a5 | x9) ^ (x19 & ~a2);
   out4 = *xout4 ^ x21;
   
   x24 = a1 ^ x9; 
   x27 = (a4 & x4) ^ x17 ^ (a5 & ~(x2 & x24));
   x28 = a4 | x24;
   out2 = *xout2 ^ x27 ^ (x28 & ~a2);
   
   x31 = x17 & x5;
   x32 = x7 & ~x31;
   x34 = (x8 & ~a4) ^ a3;
   x38 = x9 ^ x31;
   out3 = *xout3 ^ x32 ^ (a5 & x34) ^ ~(a2 | ((x13 | x16) ^ (a5 | x38)));
   
   x43 = x19 & ~x32;
   out1 = *xout1 ^
     x43 ^ x24 ^ (a5 & ~((x27 | x43) ^ x6)) ^ 
     (a2 | ((x6 & x38) ^ x34 ^ (a5 & x28 & ~(x21 ^ x38))));
   
   slice zzout1, zzout2;
   asm ("movq %%mm5, %0": : "m"(zzout1));
   asm ("movq %%mm7, %0": : "m"(zzout2));

   asm ("movq %0, %%mm5": : "m" (out1));
   asm ("movq %0, %%mm7": : "m" (out2));   
   //asm ("movq %0, %%mm6": : "m" (out3));
   //asm ("movq %0, %%mm4": : "m" (out4));

   xprint ("out1", out1, zzout1);
   xprint ("out2", out2, zzout2);
   xprint ("a1  ",   a1, *(ebp+5));
   xprint ("x9  ",   x9, *(ebp+14));
   xprint ("x24 ",  x24, *(ebp+19));
   xprint ("x27 ",  x27, *(ebp+22));
   xprint ("x28 ",  x28, *(ebp+20));
   }*/
}


// --------------------------------------------------------
// --------------------------------------------------------
void mmxs6_kwan(stMmxParams *)
{

// On exit:
// out1 - mm0
// out2 - mm1
// out3 - mm2
// out4 - mm4

// 48 cycles for 61 vars

#define a1    "40(%eax)"
#define a2    "48(%eax)"
#define a3    "56(%eax)"
#define a4    "64(%eax)"
#define x1    "72(%eax)"
#define x2    "80(%eax)"
#define x5    "88(%eax)"
#define x6    "96(%eax)"
#define x8   "104(%eax)"
#define x15  "112(%eax)"
#define x16  "120(%eax)"

asm ("
   movq  %mm2, "a3"
   movq  %mm4, %mm6      # copy a5

   pxor  "mmNOT", %mm6   # x2 = ~a5
   movq  %mm5, %mm7      # copy a6

   movq  %mm1, "a2"
   movq  %mm4, %mm2      # copy a5

   movq  %mm3, "a4"
   pxor  %mm1, %mm7      # x3 = a2 ^ a6

   pxor  "mmNOT", %mm1   # x1 = ~a2
   pxor  %mm6, %mm7      # x4 = x2 ^ x3

   movq  %mm6, "x2"
   pxor  %mm0, %mm7      # x5 = a1 ^ x4

   pand  %mm5, %mm2      # x6 = a5 & a6
   movq  %mm4, %mm6      # copy a5

   movq  %mm1, "x1"
   movq  %mm5, %mm3      # copy a6

   pand  "a2", %mm3      # x15 = a2 & a6
   pand  %mm7, %mm6      # x8 = a5 & x5

   movq  %mm0, "a1"
   por   %mm2, %mm1      # x7 = x1 | x6

   movq  %mm2, "x6"
   pand  %mm6, %mm0      # x9 = a1 & x8

   movq  %mm3, "x15"
   pxor  %mm0, %mm1      # x10 = x7 ^ x9

   movq  "a4", %mm0
   movq  %mm4, %mm2      # copy a5

   movq  %mm6, "x8"
   pand  %mm1, %mm0      # x11 = a4 & x10

   movq  %mm7, "x5"
   pxor  %mm3, %mm2      # x16 = a5 ^ x15

   movq  "x2", %mm6
   pxor  %mm7, %mm0      # x12 = x5 ^ x11

   movq  "a1", %mm7
   pxor  %mm5, %mm1      # x13 = a6 ^ x10

   movq  %mm2, "x16"
   pand  %mm7, %mm2      # x17 = a1 & x16

   movq  "a4", %mm3
   pxor  %mm2, %mm6      # x18 = x2 ^ x17

   pxor  "a2", %mm2      # x26 = a2 ^ x17
   pand  %mm7, %mm1      # x14 = a1 & x13

   por   %mm6, %mm3      # x19 = a4 | x18
   pxor  %mm5, %mm6      # x23 = a6 ^ x18

   pxor  %mm3, %mm1      # x20 = x14 ^ x19
   pand  %mm6, %mm7      # x24 = a1 & x23

   pand  "a3", %mm1      # x21 = a3 & x20
   pand  %mm4, %mm6      # x38 = a5 & x23

   movq  "x6", %mm3
   pxor  %mm1, %mm0      # x22 = x12 ^ x21

   pxor  "out2", %mm0    # out2 ^= x22
   por   %mm2, %mm3      # x27 = x6 | x26
   
   pand  "a4", %mm3      # x28 = a4 & x27
   pxor  %mm7, %mm4      # x25 = a5 ^ x24

   movq  "x5", %mm1
   pxor  %mm3, %mm4      # x29 = x25 ^ x28

   pxor  "mmNOT", %mm2   # x30 = ~x26
   por   %mm4, %mm5      # x31 = a6 | x29

   movq  %mm0, "out2"
   movq  %mm5, %mm3      # copy x31

   pandn "a4", %mm3      # x33 = a4 & ~x31
   pxor  %mm6, %mm1      # x39 = x5 ^ x38

   movq  "x6", %mm0
   pxor  %mm2, %mm3      # x34 = x30 ^ x33

   por   "a4", %mm1      # x40 = a4 | x39
   pxor  %mm3, %mm0      # x37 = x6 ^ x34

   pand  "a3", %mm3      # x35 = a3 & x34
   pxor  %mm1, %mm0      # x41 = x37 ^ x40

   por   "x5", %mm6      # x50 = x5 | x38
   movq  %mm7, %mm1      # copy x24

   pxor  "x15", %mm7     # x44 = x15 ^ x24
   pxor  %mm3, %mm4      # x36 = x29 ^ x35

   movq  "a4", %mm3
   pxor  %mm5, %mm7      # x45 = x31 ^ x44

   pand  "x8", %mm5      # x52 = x8 & x31
   por   %mm3, %mm7      # x46 = a4 | x45

   pxor  "x6", %mm6      # x51 = x6 ^ x50
   por   %mm3, %mm5      # x53 = a4 | x52

   por   "x16", %mm1     # x42 = x16 | x24
   pxor  %mm6, %mm5      # x54 = x51 ^ x53

   pxor  "x1", %mm1      # x43 = x1 ^ x42

   movq  "a3", %mm3
   pxor  %mm1, %mm7      # x47 = x43 ^ x46

   pxor  "out4", %mm4    # out4 ^= x36
   por   %mm3, %mm7      # x48 = a3 | x47

   pand  %mm1, %mm2      # x55 = x30 & x43
   pxor  %mm7, %mm0      # x49 = x41 ^ x48

   pxor  "out1", %mm0    # out1 ^= x49
   por   %mm3, %mm2      # x56 = a3 | x55

   movq  "out2", %mm1
   pxor  %mm5, %mm2      # x57 = x54 ^ x56

   pxor  "out3", %mm2    # out3 ^= x57
");

#undef a1 
#undef a2 
#undef a3 
#undef a4 
#undef x1 
#undef x2 
#undef x5 
#undef x6 
#undef x8 
#undef x15
#undef x16
}


// --------------------------------------------------------
// --------------------------------------------------------
void mmxs7_kwan(stMmxParams *)
{

// On exit:
// out1 - mm7
// out2 - mm1
// out3 - mm3
// out4 - mm0

#define a1    "40(%eax)"
#define a2    "48(%eax)"
#define a4    "56(%eax)"
#define a6    "64(%eax)"
#define x6    "72(%eax)"
#define x7    "80(%eax)"
#define x8    "88(%eax)"
#define x11   "96(%eax)"
#define x13  "104(%eax)"
#define x15  "112(%eax)"
#define x25  "120(%eax)"
#define x26  "128(%eax)"

asm ("

   movq  %mm0, "a1"
   movq  %mm1, %mm6      # copy a2

   movq  %mm1, "a2"
   movq  %mm3, %mm7      # copy a4

   movq  %mm5, "a6"
   pand  %mm3, %mm6      # x3 = a2 & a4

   movq  %mm3, "a4"
   pxor  %mm4, %mm6      # x4 = a5 ^ x3

   pxor  "mmNOT", %mm4   # x2 = ~a5
   pand  %mm6, %mm7      # x6 = a4 & x4

   pand  %mm4, %mm3      # x12 = a4 & x2
   movq  %mm1, %mm5      # copy a2

   pxor  %mm2, %mm6      # x5 = a3 ^ x4
   pxor  %mm7, %mm5      # x7 = a2 ^ x6

   movq  %mm7, "x6"
   por   %mm1, %mm4      # x14 = a2 | x2

   por   %mm3, %mm1      # x13 = a2 | x12
   pxor  %mm6, %mm7      # x25 = x5 ^ x6

   movq  %mm5, "x7"
   pand  %mm2, %mm4      # x15 = a3 & x14

   pand  %mm2, %mm5      # x8 = a3 & x7
   por   %mm7, %mm3      # x26 = x12 | x25

   movq  %mm1, "x13"
   pxor  %mm5, %mm0      # x9 = a1 ^ x8

   por   "a6", %mm0      # x10 = a6 | x9
   pxor  %mm4, %mm1      # x16 = x13 ^ x15

   movq  %mm4, "x15"
   pxor  %mm6, %mm0      # x11 = x5 ^ x10

   movq  %mm5, "x8"
   movq  %mm3, %mm4      # copy x26

   movq  "a6", %mm6
   movq  %mm0, %mm5      # copy x11

   pxor  "x6", %mm5      # x17 = x6 ^ x11
   por   %mm6, %mm4      # x27 = a6 | x26

   movq  %mm7, "x25"
   por   %mm6, %mm5      # x18 = a6 | x17

   movq  "a1", %mm7
   pxor  %mm1, %mm5      # x19 = x16 ^ x18

   movq  %mm3, "x26"
   pand  %mm5, %mm7      # x20 = a1 & x19

   movq  %mm0, "x11"
   pxor  %mm0, %mm7      # x21 = x11 ^ x20

   movq  "a4", %mm3
   movq  %mm7, %mm0      # copy x21

   por   "a2", %mm0      # x22 = a2 | x21
   pand  %mm3, %mm1      # x35 = a4 & x16

   pand  "x13", %mm3     # x39 = a4 & x13

   por   "x7", %mm2      # x40 = a3 | x7

   pxor  "x6", %mm0      # x23 = x6 ^ x22
   pxor  %mm3, %mm2      # x41 = x39 ^ x40

   movq  "a2", %mm3
   movq  %mm0, %mm6      # copy x23

   pxor  "mmNOT", %mm3   # x1 = ~a2

   pxor  "x15", %mm6     # x24 = x15 ^ x23
   por   %mm3, %mm1      # x36 = x1 | x35

   pand  "x26", %mm0     # x30 = x23 & x26
   pxor  %mm6, %mm4      # x28 = x24 ^ x27

   pand  "a6", %mm0      # x31 = a6 & x30
   por   %mm3, %mm6      # x42 = x1 | x24

   por   "a6", %mm6      # x43 = a6 | x42
   pand  %mm5, %mm3      # x29 = x1 & x19

   pand  "a6", %mm1      # x37 = a6 & x36
   pxor  %mm3, %mm0      # x32 = x29 ^ x31

   por   "a1", %mm0      # x33 = a1 | x32
   pxor  %mm6, %mm2      # x44 = x41 ^ x48

   pxor  "x11", %mm1     # x38 = x11 ^ x37
   pxor  %mm4, %mm0      # x34 = x28 ^ x33

   movq  "a1", %mm4
   pxor  %mm2, %mm5      # x51 = x19 ^ x44

   movq  "a4", %mm6
   por   %mm2, %mm4      # x45 = a1 | x44

   pxor  "x25", %mm6     # x52 = a4 ^ x25
   pxor  %mm4, %mm1      # x46 = x38 ^ x45

   movq  "a6", %mm4
   pand  %mm1, %mm6      # x53 = x46 & x52

   movq  "x6", %mm3
   pand  %mm4, %mm6      # x54 = a6 & x53

   pxor  "x15", %mm3     # x48 = x6 ^ x15
   pxor  %mm5, %mm6      # x55 = x51 ^ x54

   pxor  "x8", %mm2      # x47 = x8 ^ x44
   por   %mm4, %mm3      # x49 = a6 | x48

   por   "a1", %mm6      # x56 = a1 | x55
   pxor  %mm2, %mm3      # x50 = x47 ^ x49

   pxor  "out1", %mm7    # out1 ^= x21
   pxor  %mm6, %mm3      # x57 = x50 ^ x56

   pxor  "out2", %mm1    # out2 ^= x46

   pxor  "out3", %mm3    # out3 ^= x57

   pxor  "out4", %mm0    # out4 ^= x34

                         # 48 clocks for 61 variables
");
#undef a1
#undef a2
#undef a4
#undef a6
#undef x6
#undef x7
#undef x8
#undef x11
#undef x13
#undef x15
#undef x25
#undef x26
}


// --------------------------------------------------------
// --------------------------------------------------------
void mmxs8_kwan(stMmxParams *)
{

// On exit:
// out1 - mm6
// out2 - mm2
// out3 - mm5
// out4 - mm1

#define a1    "40(%eax)"
#define a2    "48(%eax)"
#define a4    "56(%eax)"
#define a5    "64(%eax)"
#define a6    "72(%eax)"
#define x14   "80(%eax)"
#define x22   "88(%eax)"
#define x33   "96(%eax)"

asm ("
   movq  %mm0, "a1"
   movq  %mm2, %mm6      # copy a3

   pxor  "mmNOT", %mm0   # x1 = ~a1
   movq  %mm2, %mm7      # copy a3

   movq  %mm3, "a4"
   por   %mm0, %mm7      # x4 = a3 | x1

   pxor  "mmNOT", %mm3   # x2 = ~a4
   pxor  %mm0, %mm6      # x3 = a3 ^ x1

   movq  %mm5, "a6"
   movq  %mm4, %mm5      # copy a5

   movq  %mm1, "a2"
   movq  %mm7, %mm1      # copy x4

   movq  %mm4, "a5"
   pxor  %mm3, %mm7      # x5 = x2 ^ x4

   por   %mm6, %mm5      # x22 = a5 | x3
   por   %mm7, %mm0      # x8 = x1 | x5

   pand  %mm4, %mm1      # x26 = a5 & x4
   pandn %mm0, %mm2      # x25 = x8 & ~a3

   por   %mm7, %mm4      # x6 = a5 | x5
   pxor  %mm1, %mm2      # x27 = x25 ^ x26

   movq  %mm5, "x22"
   pand  %mm3, %mm5      # x23 = x2 & x22

   por   "a2", %mm2      # x28 = a2 | x27
   pxor  %mm4, %mm7      # x32 = x5 ^ x6

   pxor  %mm0, %mm3      # x9 = x2 ^ x8
   movq  %mm4, %mm1      # copy x6

   pxor  "x22", %mm7     # x33 = x22 ^ x32
   pxor  %mm3, %mm1      # x14 = x6 ^ x9

   pxor  %mm6, %mm4      # x7 = x3 ^ x6
   pxor  %mm5, %mm2      # x29 = x23 ^ x28

   pxor  "a1", %mm5      # x39 = a1 ^ x23
   pand  %mm3, %mm6      # x15 = x3 & x9

   movq  %mm1, "x14"
   pand  %mm4, %mm5      # x40 = x7 & x39

   movq  %mm7, "x33"
   movq  %mm0, %mm1      # copy x8

   pand  "a5", %mm3      # x10 = a5 & x9
   movq  %mm0, %mm7      # copy x8

   pand  "a5", %mm1      # x16 = a5 & x8
   pxor  %mm3, %mm7      # x11 = x8 ^ x10

   pand  "a2", %mm7      # x12 = a2 & x11
   pxor  %mm1, %mm6      # x17 = x15 ^ x16

   movq  "a6", %mm1      # retrieve a6
   pxor  %mm4, %mm7      # x13 = x7 ^ x12

   por   "a2", %mm6      # x18 = a2 | x17
   pandn %mm0, %mm4      # x48 = x8 & ~x17

   pxor  "x14", %mm6     # x19 = x14 ^ x18
   pand  %mm2, %mm1      # x30 = a6 & x29

   pxor  "a1", %mm3      # x45 = a1 ^ x10
   pxor  %mm6, %mm2      # x51 = x19 ^ x29

   por   "a6", %mm6      # x20 = a6 | x19
   pxor  %mm7, %mm1      # x31 = x13 ^ x30

   pxor  "x22", %mm3     # x46 = x22 ^ x45
   pxor  %mm7, %mm6      # x21 = x13 ^ x20

   por   "a2", %mm4      # x49 = a2 | x48

   pand  "a2", %mm5      # x41 = a2 & x40
   pxor  %mm4, %mm3      # x50 = x46 ^ x49

   movq  "a1", %mm4      # retrieve a1

   pand  "x33", %mm4     # x37 = a1 & x33

   por   "a4", %mm7      # x34 = a4 | x13
   pxor  %mm4, %mm0      # x38 = x8 ^ x37

   pand  "a2", %mm7      # x35 = a2 & x34
   pxor  %mm0, %mm5      # x42 = x38 ^ x41

   movq  "a6", %mm4      # retrieve a6
   por   %mm0, %mm2      # x52 = x38 | x51

   pxor  "x33", %mm7     # x36 = x33 ^ x35
   por   %mm4, %mm5      # x43 = a6 | x42

   pxor  "out1", %mm6    # out1 ^= x21
   pand  %mm4, %mm2      # x53 = a6 & x52

   pxor  "out4", %mm1    # out4 ^= x31
   pxor  %mm7, %mm5      # x44 = x36 ^ x43

   pxor  "out3", %mm5    # out3 ^= x44
   pxor  %mm3, %mm2      # x54 = x50 ^ x53

   pxor  "out2", %mm2    # out2 ^= x54
");

#undef a1 
#undef a2 
#undef a4 
#undef a5 
#undef a6 
#undef x14
#undef x22
#undef x33

}

#undef mmNOT
#undef out1
#undef out2
#undef out3
#undef out4


