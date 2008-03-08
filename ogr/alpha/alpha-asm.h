// $Id: alpha-asm.h,v 1.3 2008/03/08 20:18:28 kakace Exp $

#if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2)
  #if (SCALAR_BITS == 32)
    #if defined(__GNUC__)
      static __inline__ int __CNTLZ__(register SCALAR i)
      { 
        register SCALAR j = (SCALAR)i << 32;
        __asm__ ("ctlz %0,%0" : "=r"(j) : "0" (j));
        return (int)j;
      }
    #else
      static inline int __CNTLZ__(register SCALAR i)
      {
        __int64 r = asm("ctlz %a0, %v0;", (SCALAR)i << 32);
        return (int)r;
      } 
    #endif
  #else /* assume SCALAR_BITS == 64 */
    #if defined(__GNUC__)
    static __inline__ int __CNTLZ__(register SCALAR i)
    { 
      __asm__ ("ctlz %0,%0" : "=r"(i) : "0" (i));
      return (int)j;
    }
    #else
    static inline int __CNTLZ__(register SCALAR i)
    {
      __int64 r = asm("ctlz %a0, %v0;", i);
      return (int)r;
    } 
    #endif
  #endif
  #define __CNTLZ(x) (__CNTLZ__(~(x))+1)
#endif

