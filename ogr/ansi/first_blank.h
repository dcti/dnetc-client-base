/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: first_blank.h,v 1.3 2008/12/29 14:18:42 kakace Exp $
 */


#ifndef ogr_first_blank_H
#define ogr_first_blank_H


/* Select the implementation for the LOOKUP_FIRSTBLANK function/macro.
   The macro OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM specifies how support is
   provided :
   0 -> No hardware support.
   1 -> Have ASM code, but still need the lookup table. You'll have to supply
        suitable code for the __CNTLZ_ARRAY_BASED(bitmap,bitarray) macro.
   2 -> Full featured ASM code. You'll have to supply code suitable for the
        __CNTLZ(bitmap) macro.
 
   NOTE : These macros shall arrange to return the number of leading 0 of
          "~bitmap" (i.e. the number of leading 1 of the "bitmap" argument) plus
          one. For a 32-bit bitmap argument, the valid range is [1; 33]. Said
          otherwise :
          __CNTLZ(0xFFFFFFFF) == 33
          __CNTLZ(0xFFFFFFFE) == 32
          __CNTLZ(0xFFFFA427) == 18
          __CNTLZ(0x00000000) ==  1
*/


#if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 1) && !defined(__CNTLZ_ARRAY_BASED)
   #warning Macro __CNTLZ_ARRAY_BASED not defined. OGROPT_FFZ reset to 0.
   #undef  OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
   #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 0
#endif

#if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2) && !defined(__CNTLZ)
   #warning Macro __CNTLZ not defined. OGROPT_FFZ reset to 0.
   #undef  OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
   #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0
#endif


#if defined(BYTE_ORDER) && (BYTE_ORDER == LITTLE_ENDIAN)
   #define FP_CLZ_LITTLEEND 1
#endif


#if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2) && defined(__CNTLZ)
   #define LOOKUP_FIRSTBLANK(x) __CNTLZ(x)
#elif (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 0) && defined(FP_CLZ_LITTLEEND) \
   && (SCALAR_BITS == 32)
   /*
   ** using the exponent in floating point double format
   ** Relies upon IEEE format (Little Endian)
   */
   static inline int LOOKUP_FIRSTBLANK(register unsigned int input)
   {
      unsigned int i;
      union {
         double d;
         int i[2];
      } u;
      i = ~input;
      u.d = i;
      return i == 0 ? 33 : 1055 - (u.i[1] >> 20);
   }
#else
   static const char ogr_first_blank_8bit[256] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9
   };

   #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 1) && defined(__CNTLZ_ARRAY_BASED)
      #define LOOKUP_FIRSTBLANK(x) __CNTLZ_ARRAY_BASED((x),ogr_first_blank_8bit)
   #else /* C code, no asm */

#if defined(_MSC_VER)
#pragma warning(disable:4146) // unary minus operator applied to unsigned type
#endif

      static inline int LOOKUP_FIRSTBLANK(register SCALAR input)
      {
         register int result = 0;
		 // The following may product warnings on some compilers because of 
		 // negation of an unsigned type, however they are mathematically okay.
		 // "-value" is equal to "~(value - 1)", but the shortest form was used.
         #if (SCALAR_BITS == 64)
         if (input >= -((SCALAR) 1 << (SCALAR_BITS-32))) {
            input <<= 32;
            result += 32;
         }
         #endif
         if (input >= -((SCALAR) 1 << (SCALAR_BITS-16))) {
            input <<= 16;
            result += 16;
         }
         if (input >= -((SCALAR) 1 << (SCALAR_BITS-8))) {
            input <<= 8;
            result += 8;
         }
         result += ogr_first_blank_8bit[input>>(SCALAR_BITS-8)];
         return result;
      }
   #endif
#endif

#endif	/* ogr_first_blank_H */
