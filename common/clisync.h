/* Hey, Emacs, this a -*-C++-*- file !
 * 
 * Simple, lightweight synchronization primitives (most similar to 
 * spinlocks), used by the client for lightweight protection of small 
 * and fast critical sections (eg mem copy operations).
 * Compilation begun Jan 2001 by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
*/
#ifndef __CLISYNC_H__
#define __CLISYNC_H__ "@(#)$Id: clisync.h,v 1.1.2.11 2001/03/22 14:52:06 cyp Exp $"

#include "cputypes.h"           /* thread defines */
#include "sleepdef.h"           /* NonPolledUSleep() */

#if !defined(CLIENT_SUPPORTS_SMP) /* non-threaded client */

  typedef struct { long spl; } fastlock_t;
  #define fastlock_INITIALIZER_UNLOCKED {0}
  static inline void fastlock_lock(fastlock_t *m)   { m->spl = 1; }
  static inline void fastlock_unlock(fastlock_t *m) { m->spl = 0; }
  static inline int fastlock_trylock(fastlock_t *m) { m->spl = 1; return +1; }
  /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */

#elif (CLIENT_OS == OS_MACOS)

  #include <Multiprocessing.h>
//  Without "if (MPLibraryIsLoaded())" this could be really nice
//  #define fastlock_t MPCriticalRegionID
//  #define FASTLOCK_INITIALIZER_UNLOCKED 0
//  #define fastlock_lock(x) MPEnterCriticalRegion(*x,kDurationImmediate)
//  #define fastlock_unlock(x) MPExitCriticalRegion(*x)
//  static inline int fastlock_trylock(fastlock_t *m)
//  { return (MPEnterCriticalRegion(*m,kDurationImmediate)?(0):(+1)); }

  typedef struct {MPCriticalRegionID MPregion; long spl;} fastlock_t;
  #define FASTLOCK_INITIALIZER_UNLOCKED {0,0}
  
  static inline void fastlock_lock(fastlock_t *m)
  {
    if (MPLibraryIsLoaded())
      MPEnterCriticalRegion(m->MPregion,kDurationImmediate);
    else
      m->spl = 1;
  }
  
  static inline void fastlock_unlock(fastlock_t *m)
  {
    if (MPLibraryIsLoaded())
      MPExitCriticalRegion(m->MPregion);
    else
      m->spl = 0;
  }
  
  /* Please verify this is working as expected before using it
  static inline int fastlock_trylock(fastlock_t *m)
  {
    if (MPLibraryIsLoaded())
    {
      return (MPEnterCriticalRegion(m->MPregion,kDurationImmediate)?(0):(+1));
    }
    else
    { m->spl = 1; return +1; }
  }
   */

#elif (CLIENT_CPU == CPU_ALPHA) && defined(__GNUC__)

  #error "please check this"

  typedef volatile long int fastlock_t __attribute__ ((__aligned__ (16)));
  #define FASTLOCK_INITIALIZER_UNLOCKED 0L

  static __inline__ void fastlock_unlock(fastlock_t *__lock)
  {
    __asm__ __volatile__ ("mb; stq $31, %0; mb"
                          : "=m" (__lock));
  }
  static __inline__ int fastlock_trylock(fastlock_t *m)
  {
    /* based on __cmpxchg_u64() in 
       http://lxr.linux.no/source/include/asm-alpha/system.h?v=2.4.0 
    */
    unsigned long old = 0, new = 1;
    unsigned long prev, cmp;
    __asm__ __volatile__(
          "1:     ldq_l %0,%5   \n\t"
          "       cmpeq %0,%3,%1\n\t"
          "       beq %1,2f     \n\t"
          "       mov %4,%1     \n\t"
          "       stq_c %1,%2   \n\t"
          "       beq %1,3f     \n\t"
          "       mb            \n\t"
          "2:                   \n\t"
          ".subsection 2        \n\t"
          "3:     br 1b         \n\t"
          ".previous"
          : "=&r"(prev), "=&r"(cmp), "=m"(*m)
          : "r"((long) old), "r"(new), "m"(*m) : "memory");
    if (prev == 0)
      return +1;
    return 0;
  }
  static __inline__ void fastlock_lock(volatile fastlock_t *m)
  {
    while (fastlock_trylock(m) <= 0)
    {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_ALPHA) && defined(_MSC_VER)
   /* MS VC 6 has spinlock intrinsics */

   typedef struct 
   { 
     #pragma pack(8)
     long spl; 
     #pragma pack()
   } fastlock_t;
   #define FASTLOCK_INITIALIZER_UNLOCKED {0}
   extern "C" int _AcquireSpinLockCount(long *, int);
   extern "C" void _ReleaseSpinLock(long *);
   #pragma intrinsic(_AcquireSpinLockCount, _ReleaseSpinLock)
   static inline void fastlock_lock(fastlock_t *m)
   {
     while (!_AcquireSpinLockCount(&(m->spl), 64))
       Sleep(1);
   }
   static inline void fastlock_unlock(fastlock_t *m)
   {
     _ReleaseSpinLock(&(m->spl));
   }
   /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */
   static inline int fastlock_trylock(fastlock_t *m)
   {
     if (!_AcquireSpinLockCount(&(m->spl),1))
       return 0;
     return +1;
   }

#elif (CLIENT_CPU == CPU_X86)

   #pragma pack(4)
   typedef struct { long spl; } fastlock_t;
   #pragma pack()
   #define FASTLOCK_INITIALIZER_UNLOCKED {0}

   /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */
   static inline int fastlock_trylock(fastlock_t *m)
   {
     long *splptr = &(m->spl);
     int lacquired = 0;

     #if defined(__GNUC__)
     /* gcc is sometimes too clever */
     struct __fool_gcc_volatile { unsigned long a[100]; };
     /* note: no 'lock' prefix even on SMP since xchg is always atomic
     */ __asm__ __volatile__(
                "movl $1,%0\n\t"
                "xchgl %0,%1\n\t"
                "xorl $1,%0\n\t"
                : "=r"(lacquired)
                : "m"(*((struct __fool_gcc_volatile *)(splptr)))
                : "memory");
     #elif defined(__BORLANDC__) /* BCC can't do inline assembler in inline functions */
     _EDX = (unsigned long)splptr;
     _EAX = 1;
     __emit__(0x87, 0x02); /* xchg [edx],eax */
     _EAX ^= 1;
     lacquired = _EAX;
     #else
     _asm mov edx, splptr
     _asm mov eax, 1
     _asm xchg eax,[edx]
     _asm xor eax, 1
     _asm mov lacquired,eax
     #endif
     if (lacquired)
       return +1;
     return 0;
   }
   static inline void fastlock_unlock(fastlock_t *m)
   {
     m->spl = 0;
   }
   static inline void fastlock_lock(fastlock_t *m)
   {
     while (fastlock_trylock(m) <= 0)
     {
       #if defined(__unix__)
       NonPolledUSleep(1);
       #elif (CLIENT_OS == OS_NETWARE)
       ThreadSwitchLowPriority();
       #elif (CLIENT_OS == OS_OS2)
       DosSleep(1);       
       #elif (CLIENT_OS == OS_WIN32)
       Sleep(1);
       #else
       #error "What's up Doc?"
       #endif
     }
   }

#elif (CLIENT_CPU == CPU_POWERPC) && defined(__GNUC__)

  typedef struct { volatile int spl; } fastlock_t;
  #define FASTLOCK_INITIALIZER_UNLOCKED {0}
  static __inline__ void fastlock_unlock(fastlock_t *m)
  { 
    int t;
    __asm__ __volatile__( /* atomic decrement */
            "1: lwarx   %0,0,%2\n"         \
            "   addic   %0,%0,-1\n"        \
            "   stwcx.  %0,0,%2\n"         \
            "   bne     1b"                \
            : "=&r" (t), "=m" (m->spl)     \
            : "r" (m), "m" (m->spl)        \
            : "cc");
    return;
  }
  /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */
  static __inline__ int fastlock_trylock(fastlock_t *m)
  {
    int t;
    __asm__ __volatile__(  /* atomic increment */
            "1: lwarx   %0,0,%2\n"        \
            "   addic   %0,%0,1\n"        \
            "   stwcx.  %0,0,%2\n"        \
            "   bne-    1b"               \
            : "=&r" (t), "=m" (m->spl)    \
            : "r" (m), "m" (m->spl)       \
            : "cc");
    if (t != 1) {          /* count is not 1? */
       fastlock_unlock(m); /* undo the increment */
       t = 0;
    }
    return t;
  }
  static __inline__ void fastlock_lock(fastlock_t *m)
  {
    while (fastlock_trylock(m) <= 0)
    {
      #if (CLIENT_OS == OS_AMIGAOS)
      NonPolledUSleep(1);
      #else
      #error "What's up Doc?"
      #endif
    }
  }

#elif (CLIENT_CPU == CPU_68K) && defined(__GNUC__)

  /* IMPORTANT: has to be a char (not an int) since when the destination for
  ** BTST is a memory location, the operation must be a byte operation
  */
  typedef struct { volatile char spl; } fastlock_t;
  #define FASTLOCK_INITIALIZER_UNLOCKED {0}
  static __inline__ void fastlock_unlock(fastlock_t *m)
  { 
    /* m->spl = 0; */
    __asm__  __volatile__ (
             "clr.b %0"         \
             : "=m"  (m->spl)   \
             :  "0"  (m->spl));
  }
  /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */
  static __inline__ int fastlock_trylock(fastlock_t *m)
  {
    char lacquired;
    __asm__  __volatile__ (
             "bset #0,%1\n"      \
             "seq %0"            \
             : "=d" (lacquired)  \
             :  "m" (m->spl));
    if (lacquired)
      return +1;
    return 0;
  }
  static __inline__ void fastlock_lock(fastlock_t *m)
  {
    while (fastlock_trylock(m) <= 0)
    {
      #if (CLIENT_OS == OS_AMIGAOS)
      NonPolledUSleep(1);
      #else
      #error "What's up Doc?"
      #endif
    }
  }

#elif (CLIENT_CPU == CPU_S390) && defined(__GNUC__)
  /* http://lxr.linux.no/source/include/asm-s390/atomic.h */

  #error "please check this"

  typedef struct { volatile int spl; } fastlock_t __attribute__ ((aligned (4)));
  #define FASTLOCK_INITIALIZER_UNLOCKED {0}
  static __inline__ void fastlock_unlock(volatile fastlock_t *v)
  { 
    int i;
    __asm__ __volatile__(  /* atomic decrement */
                         "   l     0,%0\n"
                         "0: lr    %1,0\n"
                         "   ahi   %1,-1\n"
                         "   cs    0,%1,%0\n"
                         "   jl    0b"
                         : "+m" (*v), "=&d" (i) : : "", "cc" );
    return;
  }
  /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */
  static __inline__ int fastlock_trylock(volatile fastlock_t *v)
  {
    /*
      returns 0  if expected_oldval==value in *v ( swap was successful )
      returns 1  if unsuccessful.
    */  
    int expected_oldval = 0, new_val = 1, failed;
    __asm__ __volatile__( /* atomic_compare_and_swap */
                "  cs   %2,%3,%1\n"
                "  ipm  %0\n"
                "  srl  %0,28\n"
                "0:"
                : "=&r" (retval), "+m" (*v)
                : "d" (expected_oldval) , "d" (new_val)
                : "cc");
     if (!failed)
       return +1;
     return 0;
  }
  static __inline__ void fastlock_lock(volatile fastlock_t *m)
  {
    while (fastlock_trylock(m) <= 0)
    {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_PA_RISC) && defined(__GNUC__)

  typedef __volatile int fastlock_t __attribute__ ((__aligned__ (16)));
  #define FASTLOCK_INITIALIZER_UNLOCKED -1 /* note! "unlocked"=-1 */

  static __inline__ void fastlock_unlock(fastlock_t *__lock)
  {
    *__lock = -1;
  }
  static __inline__ int fastlock_trylock(fastlock_t *__lock)
  {
    /* based on gnu libc source in
       sysdeps/mach/hppa/machine-lock.h
    */
    register int __result;
    /* LDCW, the only atomic read-write operation PA-RISC has.  Sigh. */
    __asm__ __volatile__ ("ldcws %0, %1" : "=m" (*__lock), "=r" (__result));
    if (__result != 0) /* __result is non-zero if we locked it */
      return +1;
    return 0;
  }
  static __inline__ void fastlock_lock(fastlock_t *m)
  {
    while (fastlock_trylock(m) <= 0)
    {
      NonPolledUSleep(1);
    }
  }

#elif defined(CLIENT_CPU == CPU_IA64) && defined(__GNUC__)

  #error "please check this"
  typedef __u64 fastlock_t __attribute__ ((__aligned__ (16)));
  #define FASTLOCK_INITIALIZER_UNLOCKED 0

  static __inline__ void fastlock_unlock(fastlock_t *v)
  { 
    *v = 0;
  }  
  /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */
  static __inline__ int fastlock_trylock(fastlock_t *__ptr)
  {
    /* based on __cmpxchg_u64() in 
       http://lxr.linux.no/source/include/asm-ia64/system.h?v=2.4.0 
    */
    __u64 __res, __old = 0, __new = 1;
    /* IA64_SEMFIX is a workaround for Errata 97. (A-step through B1) */
    #define IA64_SEMFIX  "mf;" 
    __asm__ __volatile__ ("mov ar.ccv=%3;;\n\t"
                          IA64_SEMFIX"cmpxchg8.acq %0=[%1],%2,ar.ccv"
                          : "=r"(__res) 
                          : "r"(__ptr), "r"(__new), "rO"(__old) 
                          : "memory");
    if (__res == 0)
      return +1;
    return 0;
  }    
  static __inline__ void fastlock_lock(fastlock_t *m)
  {
    while (fastlock_trylock(m) <= 0)
    {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_SPARC) && defined(__GNUC__)

  #error "please check this"

  /* based on
     http://lxr.linux.no/source/include/asm-sparc[64]/system.h
  */
  #if (ULONG_MAX == 0xFFFFFFFFUL) /* 32bit */
  static __inline__ unsigned long atomic_xchg_ulong(
                    __volatile__ unsigned long *m, unsigned long val)
  {
     __asm__ __volatile__("swap [%2], %0"
                          : "=&r" (val)
                          : "" (val), "r" (m));
     return val;
  }
  #else /* 64 bit */
  static __inline__ unsigned long atomic_xchg_ulong(
                    __volatile__ unsigned long *m, unsigned long val)
  {
    __asm__ __volatile__("
                 mov             %0, %%g5
         1:      ldx             [%2], %%g7
                 casx            [%2], %%g7, %0
                 cmp             %%g7, %0
                 bne,a,pn        %%xcc, 1b
                 mov             %%g5, %0
                 membar          #StoreLoad | #StoreStore
         "       : "=&r" (val)
                 : "" (val), "r" (m)
                 : "g5", "g7", "cc", "memory");
    return val;
  }
  #endif

  typedef struct { __volatile__ unsigned long spl; } fastlock_t;
  #define FASTLOCK_INITIALIZER_UNLOCKED {0L}

  static __inline__ void fastlock_unlock(fastlock_t *v)
  { 
    atomic_xchg_ulong( &(v->spl), 0 );
  }  
  /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */
  static __inline__ int fastlock_trylock(fastlock_t *v)
  { 
    if (atomic_xchg_ulong( &(v->spl), 1 ) == 0)
      return +1;
    return 0;
  }  
  static __inline__ void fastlock_lock(fastlock_t *m)
  {
    while (fastlock_trylock(m) <= 0)
    {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_VAX) && defined(__GNUC__)

  #error "please check this"

  #define fastlock_t int
  #define FASTLOCK_INITIALIZER_UNLOCKED 0
  /*
  * Emulate 'atomic test-and-set' instruction.  Attempt to acquire the lock,
  * but do not wait.  Returns 0 if successful, nonzero if unable
  * to acquire the lock.
  */
  static __inline__ int __tas(volatile int *lock)
  {
    register _res;
    __asm__ __volatile__ ( 
            "movl $1, r0;"
  	    "bbssi $0, (%1), 1f;"
            "clrl r0;"
            "1: movl r0, %0;"
            "=r"(_res)
            "r"(lock)
            "r0");
    return (int) _res;
  }
  static __inline__ void fastlock_unlock(volatile fastlock_t *v)
  {
    *v = 0;
  }
  static __inline__ int fastlock_trylock(volatile fastlock_t *v)
  { 
    if (__tas( v ) == 0)
      return +1;
    return 0;
  }  
  static __inline__ void fastlock_lock(volatile fastlock_t *m)
  {
    while (fastlock_trylock(m) <= 0)
    {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_SH4) && defined(__GNUC__)

  #define fastlock_t int
  #define FASTLOCK_INITIALIZER_UNLOCKED 0
  /*
  * Have 'atomic test-and-set' instruction.  Attempt to acquire the lock,
  * but do not wait.  Returns 0 if successful, nonzero if unable
  * to acquire the lock.
  */
  static __inline__ unsigned long __tas(volatile int *m)
  {
    unsigned long retval;
    __asm__ __volatile__ ("tas.b    @%1\n\t"
                          "movt     %0"
                          : "=r" (retval): "r" (m): "t", "memory");
    return retval;
  }
  static __inline__ void fastlock_unlock(volatile fastlock_t *v)
  {
    *v = 0;
  }
  static __inline__ int fastlock_trylock(volatile fastlock_t *v)
  { 
    if (__tas( v ) == 0)
      return +1;
    return 0;
  }  
  static __inline__ void fastlock_lock(volatile fastlock_t *m)
  {
    while (fastlock_trylock(m) <= 0)
    {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_ARM) && defined(__GNUC__)

  #error "please check this"

  /* from glibc-2.2.2/sysdeps/arm/atomicity.h */
  static __inline__ int __compare_and_swap (volatile long int *p, 
                                        long int oldval, long int newval)
  {
    int result, tmp;
    __asm__ __volatile__ (
           "0:  ldr  %1,[%2]       \n\t"
           "    mov  %0,#0         \n\t"
           "    cmp   %1,%4        \n\t"
           "    bne   1f           \n\t"
           "    swp   %0,%3,[%2]   \n\t"
           "    cmp   %1,%0        \n\t"
           "    swpne   %1,%0,[%2] \n\t"
           "    bne   0b           \n\t"
           "    mov   %0,#1        \n\t"
           "1:                     \n\t"
           : "=&r" (result), "=&r" (tmp)
           : "r" (p), "r" (newval), "r" (oldval)
           : "cc", "memory");
    return result;
  }

  #define fastlock_t long
  #define FASTLOCK_INITIALIZER_UNLOCKED 0L

  static __inline__ void fastlock_unlock(volatile fastlock_t *v)
  {
    *v = 0;
  }
  static __inline__ int fastlock_trylock(volatile fastlock_t *v)
  { 
    if (__compare_and_swap(v, 0, 1) == 0)
      return +1;
    return 0;
  }  
  static __inline__ void fastlock_lock(volatile fastlock_t *m)
  {
    while (fastlock_trylock(m) <= 0)
    {
      NonPolledUSleep(1);
    }
  }

#elif defined(_POSIX_THREADS_SUPPORTED)
  /* put this at the end, so that more people notice/are affected by 
     any potential problems in other code 
  */
  /* heaaaavyweight, but better than nothing */

  #include <pthread.h>
  #define fastlock_t                    pthread_mutex_t
  #define FASTLOCK_INITIALIZER_UNLOCKED PTHREAD_MUTEX_INITIALIZER
  #define fastlock_lock                 pthread_mutex_lock
  #define fastlock_unlock               pthread_mutex_unlock
  #define fastlock_trylock              pthread_mutex_trylock

#elif (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS)
  /* put this at the end, so that more people notice/are affected by 
     any potential problems in other code 
  */
  /* heaaaavyweight, but better than nothing */
                               
  #include <thread.h>
  #include <synch.h>
  #define fastlock_t                    mutex_t
  #define FASTLOCK_INITIALIZER_UNLOCKED DEFAULTMUTEX
  #define fastlock_lock                 mutex_lock
  #define fastlock_unlock               mutex_unlock
  #define fastlock_trylock              mutex_trylock

#elif (CLIENT_CPU == CPU_POWER) || (CLIENT_CPU == CPU_MIPS)

  #error "Whats required here is ..."
  #error ""
  #error "typedef [...fill_this...] fastlock_t;"
  #error "#define FASTLOCK_INITIALIZER_UNLOCKED [...{0} or whatever...]"
  #error ""
  #error "static __inline__ void fastlock_unlock(fastlock_t *v)  { ...fill this... }"
  #error "static __inline__ int fastlock_trylock(fastlock_t * v) { ...fill this... return +1 on success, 0 on failure, (optional -1 on error) }"
  #error "static __inline__ void fastlock_lock(fastlock_t *v)  { while (fastlock_trylock(v) <= 0) { usleep(0}; } }"
  #error ""
  #error "some code to look at/for..."
  #error "atomic test-and-set tas(). [lots of places on the net, like postgres]"
  #error "atomic compare_and_swap eg, http://lxr.linux.no/source/include/asm-XXX/system.h"
  #error "               or gcc source libstdc++-v3/config/cpu/XXX/bits/atomicity.h"
  #error "atomic [inc|dec]rement eg, http://lxr.linux.no/source/include/asm-XXX/atomic.h"

#else

   #error How did you get here?

#endif

#endif /* __CLISYNC_H__ */

