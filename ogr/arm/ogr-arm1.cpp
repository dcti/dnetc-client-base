/*
 * Copyright distributed.net 2002-2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 */

const char *ogr_arm1_cpp(void) {
return "@(#)$Id: ogr-arm1.cpp,v 1.1.2.1 2008/02/12 15:43:22 teichp Exp $"; }

#if defined(ASM_ARM)

  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* 0-2 - partial support */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'            */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0-2 - 'no'  (default) */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std' (default) */

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
      #define __CNTLZ_ARRAY_BASED(x, y) __CNTLZ__(x, y)
    #endif  /* OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM */
  #endif  /* __GNUC__ */
  
  #define OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table_arm1

  #include "ansi/ogr.cpp"
  
#else  /* ASM_ARM */

  #error use this only with arm since it contains arm assembly
  
#endif
