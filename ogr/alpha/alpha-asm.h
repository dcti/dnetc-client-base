// $Id: alpha-asm.h,v 1.1.4.1 2004/08/14 23:36:29 kakace Exp $

#if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2)
  #if defined(__GNUC__)
    static __inline__ int __CNTLZ__(register unsigned int i)
    { 
      register unsigned long j = (unsigned long)i << 32;
      __asm__ ("ctlz %0,%0" : "=r"(j) : "0" (j));
      return (int)j;
    }
  #else
    static inline int __CNTLZ__(register unsigned int i)
    {
      __int64 r = asm("ctlz %a0, %v0;", (unsigned long)i << 32);
      return (int)r;
    } 
  #endif
  #define __CNTLZ(x) (__CNTLZ__(~(x))+1)
#endif

