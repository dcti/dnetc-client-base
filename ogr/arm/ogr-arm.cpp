/*
 * Copyright distributed.net 2002-2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 */

const char *ogr_arm_cpp(void) {
return "@(#)$Id: ogr-arm.cpp,v 1.1.4.1 2004/08/14 23:36:58 kakace Exp $"; }

#if defined(ASM_ARM)

  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* 0-2 - partial support */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             1 /* 0-2 - 'yes'           */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0-2 - 'no'  (default) */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std' (default) */

  /*
  ** Note from Kakace :
  ** The following code has been borrowed from ogr.cpp
  ** Since ARM platforms appear to use ASM cores, most of the settings as
  ** well as this inline code only apply to ogr_create/ogr_create_pass2.
  */
  #if defined(__GNUC__)
    #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 1)
      static __inline__ int __CNTLZ__(register unsigned int input,
                                      const char bitarray[])
      {
        register int temp, result;
        __asm__ ("mov     %0,#0\n\t"             \
                 "cmp     %1,#0xffff0000\n\t"    \
                 "movcs   %1,%1,lsl#16\n\t"      \
                 "addcs   %0,%0,#16\n\t"         \
                 "cmp     %1,#0xff000000\n\t"    \
                 "movcs   %1,%1,lsl#8\n\t"       \
                 "ldrb    %1,[%3,%1,lsr#24]\n\t" \
                 "addcs   %0,%0,#8\n\t"          \
                 "add     %0,%0,%1"              \
                 : "=r" (result), "=r" (temp)
                 : "1" (input), "r" ((unsigned int)bitarray));
        return result;
      }
      #define __CNTLZ_ARRAY_BASED(x) __CNTLZ__(x)
    #endif

    #if (OGROPT_ALTERNATE_CYCLE == 0 && OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 1)
      #error fixme: No longer compatible with existing code
      // need to copy "newbit" into "list0"
      #define COMP_LEFT_LIST_RIGHT_32(lev) {  \
        int a1, a2;                           \
        asm ("ldr %0,[%2,#44]\n               \
              ldr %1,[%2,#48]\n               \
              str %0,[%2,#40]\n               \
              ldr %0,[%2,#52]\n               \
              str %1,[%2,#44]\n               \
              ldr %1,[%2,#56]\n               \
              str %0,[%2,#48]\n               \
              ldr %0,[%2,#12]\n               \
              str %1,[%2,#52]\n               \
              ldr %1,[%2,#8]\n                \
              str %0,[%2,#16]\n               \
              ldr %0,[%2,#4]\n                \
              str %1,[%2,#12]\n               \
              ldr %1,[%2,#0]\n                \
              str %0,[%2,#8]\n                \
              mov %0,#0\n                     \
              str %1,[%2,#4]\n                \
              str %0,[%2,#56]\n               \
              str %0,[%2,#0]" :               \
            "=r" (a1), "=r" (a2),             \
            "=r" (lev) : "2" (lev));          \
      }

    #endif  /* OGROPT_ALTERNATE_CYCLE == 0 */
  #endif  /* __GNUC__ */

#endif  /* ASM_ARM */

#include "ansi/ogr.cpp"
