/*
 * Copyright distributed.net 1999-2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __ASM_PPC_H__
#define __ASM_PPC_H__ "@(#)$Id: asm-ppc.h,v 1.4 2008/03/08 20:18:29 kakace Exp $"

#if defined(__GNUC__)     /*================================================*/

  #if defined(ASM_POWER)
    #define __CNTLZ__(i) ({ int x; __asm__ ("cntlz %0,%0" : "=r" (x) : "0" (i)); x;})
  #elif (SCALAR_BITS == 64) && defined(__ppc64__)
    #define __CNTLZ__(i) ({ int x; __asm__ ("cntlzd %0,%1" : "=r" (x) : "0" (i)); x;})
  #elif defined(ASM_PPC) || defined(__PPC__) || defined(__POWERPC__)
    #define __CNTLZ__(i) ({ int x; __asm__ ("cntlzw %0,%1" : "=r" (x) : "0" (i)); x;})
  #endif

  #if defined(__APPLE_CC__)
    #define __BALIGN __asm__ __volatile__ (".align 4"::)
  #elif !defined(_AIX)
    #define __BALIGN __asm__ __volatile__ (" .balignl 32,0x60000000"; nop; nop : : )
  #endif

#elif defined(__xlC__)    /*================================================*/

  #include <builtins.h>
  #define __CNTLZ__(x) __cntlz4(x)

#elif defined(__MWERKS__) /*================================================*/

  #define __CNTLZ__(x) __cntlzw(x)

#endif  /* compiler */

#endif  /* __ASM_PPC_H__ */
