/* Hey, Emacs, this a -*-C++-*- file !
 * 
 * Simple, lightweight synchronization primitives, used by the client 
 * for lightweight protection of small and fast critical sections 
 * (eg mem copy operations from memory in one scope to memory in another), 
 * with low contention (possible accessors are the crunchers and the client's
 * main loop, the latter accessing cruncher data only every second or so).
 * 
 * Compilation begun Jan 2001 by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * The locking scheme implemented by the client may be best described as 
 * fine-grained parallelization with many locking/unlocking actions taken.
 * Fine-grained locking results in a very short time spent while holding a 
 * lock, so there is a low probability of collision (finding a lock busy).
*/
#ifndef __CLISYNC_H__
#define __CLISYNC_H__ "@(#)$Id: clisync.h,v 1.1.2.15 2001/03/29 15:08:38 cyp Exp $"

#include "cputypes.h"           /* thread defines */
#include "sleepdef.h"           /* NonPolledUSleep() */

#if !defined(CLIENT_SUPPORTS_SMP) /* non-threaded client */

  typedef struct { long spl; } fastlock_t;
  #define FASTLOCK_INITIALIZER_UNLOCKED {0}
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

  typedef { volatile unsigned int spl; } fastlock_t __attribute__ ((__aligned__ (32)));
  #define FASTLOCK_INITIALIZER_UNLOCKED ((fastlock_t){0})

  static __inline__ void fastlock_unlock(fastlock_t *v)
  {
    __asm__ __volatile__("mb": : :"memory")
    v->spl = 0;  
  }
  /* http://lxr.linux.no/source/include/asm-alpha/bitops.h?v=2.4.0#L92 */
  extern __inline__ int
  test_and_set_bit(unsigned long nr, volatile void *addr)
  {
        unsigned long oldbit;
        unsigned long temp;
        int *m = ((int *) addr) + (nr >> 5);
 
        __asm__ __volatile__(
        "1:     ldl_l %0,%4   \n\t" \
        "       and %0,%3,%2  \n\t" \
        "       bne %2,2f     \n\t" \
        "       xor %0,%3,%0  \n\t" \
        "       stl_c %0,%1   \n\t" \
        "       beq %0,3f     \n\t" \
        "2:     mb            \n\t" \
        ".subsection 2        \n\t" \
        "3:     br 1b         \n\t" \
        ".previous"
        :"=&r" (temp), "=m" (*m), "=&r" (oldbit)
        :"Ir" (1UL << (nr & 31)), "m" (*m) : "memory");

        return oldbit != 0;
  }
  static __inline__ int fastlock_trylock(fastlock_t *v)
  {
    if (!test_and_set_bit(0, &(v->spl)))
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
                "movl $1,%0    \n\t" \
                "xchgl %0,%1   \n\t" \
                "xorl $1,%0    \n\t"
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

  /* based on
  http://cvsweb.netbsd.org/bsdweb.cgi/syssrc/sys/arch/powerpc/include/lock.h?rev=1.4.2.1
  approved by Dan Oetting
  */
  typedef struct { __volatile int spl; } fastlock_t __attribute__ ((__aligned__ (16)));
  #define FASTLOCK_INITIALIZER_UNLOCKED ((fastlock_t){0})

  static __inline__ void fastlock_unlock(fastlock_t *__alp)
  { 
    __volatile int *alp = &(__alp->spl);
    __asm __volatile ("sync");
    *alp = 0; /*FASTLOCK_INITIALIZER_UNLOCKED*/
  }
  static __inline__ int fastlock_trylock(fastlock_t *__alp)
  {
    int old, dummy;
    __volatile int *alp = &(__alp->spl);
    __asm__ __volatile__ ( /* compare_and_swap */
                   "1:  lwarx  %0,0,%1 \n\t"\
                   "    cmpwi  %0,%2   \n\t"\
                   "    bne    2f      \n\t"\
                   "    stwcx. %3,0,%1 \n\t"\
                   "    bne-   1b      \n\t"\
                   "2:  stwcx. %3,0,%4 \n\t"\
                   "    isync          \n\t"
                   : "=&r"(old)
                   : "r"(alp), 
                     "I"(0 /* unlocked state */), 
                     "r"(1 /* locked state */),
                     "r"(&dummy)
                     : "memory");
    if (old == 0) /* old state was 0? */
      return +1;  /* then success */
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

#elif (CLIENT_CPU == CPU_POWERPC) && (defined(__MWERKS__) || defined(__MRC__))

  /* based on
  http://cvsweb.netbsd.org/bsdweb.cgi/syssrc/sys/arch/powerpc/include/lock.h?rev=1.4.2.1 
  approved by Dan Oetting 
  */ 
  #define __inline__ inline
  #pragma pack(4)
  typedef struct { volatile int spl; } fastlock_t;
  #pragma pack()
  #define FASTLOCK_INITIALIZER_UNLOCKED ((fastlock_t){0})

  static __inline__ void fastlock_unlock(fastlock_t *__alp)
  { 
    volatile int *alp = &(__alp->spl);
    asm { sync }
    *alp = 0; /*FASTLOCK_INITIALIZER_UNLOCKED*/
  }
  static __inline__ int fastlock_trylock(fastlock_t *__alp)
  {
    int dummy;
    register int *dummyp = &dummy;
    register old, locked = 1;
    volatile int *alp = &(__alp->spl);
    asm 
    {
      @1:     lwarx   old,0,alp
              cmpwi   old,0 //FASTLOCK_INITIALIZER_UNLOCKED
              bne     @2
              stwcx.  locked,0,alp
              bne-    @1
      @2:     stwcx.  locked,0,dummyp
              isync
    }
    if (old == 0) /* old state was 0? */
      return +1;  /* then success */
    return 0;
  }
  #if (CLIENT_OS == OS_MACOS)
  #include <Multiprocessing.h>
  #endif
  static __inline__ void fastlock_lock(fastlock_t *m)
  {
    #if (CLIENT_OS == OS_MACOS)
    int need_mp_sleep = -1; /* unknown */
    while (fastlock_trylock(m) <= 0)
    {
      /* the only way we could get here is if we are
      ** running on an MP system and the lock is being
      ** held on another cpu, so we could actually 
      ** busy spin until the lock was released. But 
      ** we'll play nice...
      */
      if (need_mp_sleep == -1) /* first time through */
      {
        need_mp_sleep = 0;
        if (MPLibraryIsLoaded())
        {
          if (MPTaskIsPreemptive(kMPInvalidIDErr /* self */))
            need_mp_sleep = +1;
        }
      }
      if (need_mp_sleep)
        MPYield();
      else
        macosSmartYield(6); /* shouldn't be needed, but doesn't hurt */
    }
    #else
      #error "What's up Doc?"
    #endif
  }

#elif (CLIENT_CPU == CPU_68K) && defined(__GNUC__)

  /* IMPORTANT: has to be a char (not an int) since when the destination for
  ** BTST is a memory location, the operation must be a byte operation
  */
  typedef struct { volatile char spl; } fastlock_t;
  #define FASTLOCK_INITIALIZER_UNLOCKED ((fastlock_t){0})

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

#elif (CLIENT_CPU == CPU_PA_RISC) && defined(__GNUC__)

  typedef struct { __volatile int spl; } fastlock_t __attribute__ ((__aligned__ (16)));
  #define FASTLOCK_INITIALIZER_UNLOCKED ((fastlock_t){-1}) /* note! "unlocked"=-1 */

  static __inline__ void fastlock_unlock(fastlock_t *__alp)
  {
    __volatile int *__lock = &(__alp->spl);
    *__lock = -1;
  }
  static __inline__ int fastlock_trylock(fastlock_t *__alp)
  {
    /* based on gnu libc source in
       sysdeps/mach/hppa/machine-lock.h
    */
    __volatile int *alp = &(__alp->spl);
    register int result;
    /* LDCW, the only atomic read-write operation PA-RISC has.  Sigh. */
    __asm__ __volatile__ ("ldcws %0, %1" : "=m" (*alp), "=r" (result));
    if (result != 0) /* __result is non-zero if we locked it */
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

#elif (CLIENT_CPU == CPU_POWER) /* AIX only */ && defined(__GNUC__)

  /* Using cs(3). Deprecated as of AIX 4.0
  #define fastlock_t int 
  #define FASTLOCK_INITIALIZER_UNLOCKED 0
  #define fastlock_trylock(__flp) ((cs( __flp, 0, 1)) ? (+1) : (0))
  #define fastlock_unlock(__flp) do { *__flp = 0; break; } while (0)
  #define fastlock_lock(__flp) do { if (fastlock_trylock(__flp) > 0) break; \
                                    NonPolledUSleep(1); } while (1)
  */
  #include <sys/atomic_op.h>

  typdef struct { int lock; } fastlock_t __attribute__ ((aligned (8)));  
  #define FASTLOCK_INITIALIZER_UNLOCKED ((fastlock_t){0})

  static __inline__ void fastlock_unlock(fastlock_t *v)
  {
    /* Atomically writes a single word variable, issuing */
    /* an export fence for multiprocessor systems */
    _clear_lock ( &(v->lock), 0)
  } 
  static __inline__ void fastlock_trylock(fastlock_t *v)
  {
    /* Conditionally updates a single word variable atomically, */
    /* issuing an import fence for multiprocessor systems. */
    if (!_check_lock( &(v->lock), 0, 1 ))
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

#elif (CLIENT_CPU == CPU_S390) && defined(__GNUC__)
  /* based on
     http://lxr.linux.no/source/include/asm-s390/spinlock.h?v=2.4.0
  */
  #error "please check this"

  typedef struct { volatile unsigned long lock; } fastlock_t;
  #define FASTLOCK_INITIALIZER_UNLOCKED ((fastlock_t){0})

  static __inline__ void fastlock_unlock(fastlock_t *lp)
  {
    __asm__ __volatile__ ("    xc 0(4,%0),0(%0)\n" \
                          "    bcr 15,0"
                          : /* no output */ : "a" (lp) );
  }
  static __inline__ int fastlock_trylock(fastlock_t *lp)
  {
    unsigned long result;
    __asm__ __volatile("    slr   %1,%1\n" \
                       "    lhi   0,-1\n"  \
                       "0:  cs    %1,0,%0"
                       : "=m" (lp->lock), "=&d" (result)
                       : "" (lp->lock) : "");
    return ((!result) ? (+1) : (0));
  }
  static __inline__ void fastlock_lock(volatile fastlock_t *m)
  {
    while (fastlock_trylock(m) <= 0)
    {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_IA64) && defined(__GNUC__)

  /* based on 
     http://lxr.linux.no/source/include/asm-ia64/spinlock.h?v=2.4.0
  */
  typedef struct { volatile unsigned int lock; } spinlock_t;
  #define FASTLOCK_INITIALIZER_UNLOCKED ((fastlock_t){0})

  static __inline__ void fastlock_unlock(fastlock_t *v)
  { 
    v->lock = 0;
  }  
  static __inline__ int fastlock_trylock(fastlock_t *v)
  {
    register long result;
    /* IA64_SEMFIX is a workaround for Errata 97. (A-step through B1) */
    #define IA64_SEMFIX  "mf;" 
    __asm__ __volatile__ (
              "mov ar.ccv=r0\n" \
              ";;\n"            \
              IA64_SEMFIX"cmpxchg4.acq %0=[%2],%1,ar.ccv\n" 
             : "=r"(result) : "r"(1), "r"(&(x)->lock) : "ar.ccv", "memory");
    return ((result == 0) ? (+1) : (0));
  }
  static __inline__ void fastlock_lock(fastlock_t *m)
  {
    while (fastlock_trylock(m) <= 0)
    {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_SPARC) && defined(__GNUC__)

  /*
    based on
    http://cvsweb.netbsd.org/bsdweb.cgi/syssrc/sys/arch/sparc/include/lock.h?rev=1.6.2.1 
    http://cvsweb.netbsd.org/bsdweb.cgi/syssrc/sys/arch/sparc64/include/lock.h?rev=1.4.2.1
  */
  typedef struct { __volatile int spl; } fastlock_t;
  #define FASTLOCK_INITIALIZER_UNLOCKED ((fastlock_t){0})
  #define FASTLOCK_INITIALIZER_LOCKED   ((fastlock_t)({0xff000000}))

  static __inline__ int fastlock_trylock(fastlock_t *__alp)
  {
    __volatile int *alp = &(__alp->spl);
    #define __ldstub(__addr)                  \
    ({                                        \
       int __v;                               \
       __asm __volatile("ldstub [%1],%0"      \
           : "=r" (__v)                       \
           : "r" (__addr)                     \
           : "memory");                       \
                                              \
       __v;                                   \
    })
    return ((__ldstub(alp) == 0 /*FASTLOCK_INITIALIZER_UNLOCKED*/)?(+1):(0));
  }
  static __inline__ void fastlock_unlock(fastlock_t *__alp)
  {
    __volatile int *alp = &(__alp->spl);
    *alp = 0; /*FASTLOCK_INITIALIZER_UNLOCKED*/
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

  typedef struct { __volatile int spl; } fastlock_t;
  #define FASTLOCK_INITIALIZER_UNLOCKED ((fastlock_t){0})
  /* 
  based on  
  http://cvsweb.netbsd.org/bsdweb.cgi/syssrc/sys/arch/vax/include/lock.h?rev=1.5.2.1
  */
  static __inline__ void fastlock_unlock(fastlock_t *__alp)
  {
    __volatile int *alp = &(__alp->spl);
    *alp = 0;
  }
  static __inline__ int fastlock_trylock(fastlock_t *__alp)
  { 
    __volatile int *alp = &(__alp->spl);
    int ret;
    __asm__ __volatile ("movl $0,%0;bbssi $0,%1,1f;incl %0;1:"
                : "=&r"(ret)
                : "m"(*alp));
    return ((ret) ? (+1) : (0))
  }  
  static __inline__ void fastlock_lock(fastlock_t *m)
  {
    while (fastlock_trylock(m) <= 0)
    {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_SH4) && defined(__GNUC__)

  typedef struct { __volatile int spl; } fastlock_t;
  #define FASTLOCK_INITIALIZER_UNLOCKED ((fastlock_t){0})
  /*
  * Have 'atomic test-and-set' instruction.  Attempt to acquire the lock,
  * but do not wait.  Returns 0 if successful, nonzero if unable
  * to acquire the lock.
  */
  static __inline__ unsigned long __tas(__volatile int *m)
  {
    unsigned long retval;
    __asm__ __volatile__ ("tas.b    @%1\n\t" \
                          "movt     %0"
                          : "=r" (retval): "r" (m): "t", "memory");
    return retval;
  }
  static __inline__ void fastlock_unlock(fastlock_t *__alp)
  {
    __volatile int *alp = &(__alp->spl);
    *alp = 0;
  }
  static __inline__ int fastlock_trylock(fastlock_t *__alp)
  { 
    __volatile int *alp = &(__alp->spl);
    if (__tas( alp ) == 0)
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

#elif (CLIENT_CPU == CPU_ARM) && defined(__GNUC__)

  #error "please check this"

  /* from glibc-2.2.2/sysdeps/arm/atomicity.h */
  static __inline__ int __compare_and_swap (volatile long int *p, 
                                        long int oldval, long int newval)
  {
    int result, tmp;
    __asm__ __volatile__ (
           "0:  ldr  %1,[%2]       \n\t" \
           "    mov  %0,#0         \n\t" \
           "    cmp   %1,%4        \n\t" \
           "    bne   1f           \n\t" \
           "    swp   %0,%3,[%2]   \n\t" \
           "    cmp   %1,%0        \n\t" \
           "    swpne   %1,%0,[%2] \n\t" \
           "    bne   0b           \n\t" \
           "    mov   %0,#1        \n\t" \
           "1:                     \n\t"
           : "=&r" (result), "=&r" (tmp)
           : "r" (p), "r" (newval), "r" (oldval)
           : "cc", "memory");
    return result;
  }

  typedef { volatile long int spl; } fastlock_t;
  #define FASTLOCK_INITIALIZER_UNLOCKED ((fastlock_t){0})

  static __inline__ void fastlock_unlock(fastlock_t *__alp)
  {
    volatile long int *alp = &(__alp->spl);
    *alp = 0;
  }
  static __inline__ int fastlock_trylock(fastlock_t *__alp)
  { 
    volatile long int *alp = &(__alp->spl);
    if (__compare_and_swap( alp, 0, 1) == 0)
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

#else /* CLIENT_CPU == CPU_MIPS */
  /* can't do atomic operations on mips ISA < 2 without kernel support */

  #error "Whats required here is ... (or define SINGLE_CRUNCHER_ONLY)"
  #error ""
  #error "typedef [...fill_this...] fastlock_t;"
  #error "#define FASTLOCK_INITIALIZER_UNLOCKED [...{0} or whatever...]"
  #error ""
  #error "static __inline__ void fastlock_unlock(fastlock_t *v)  { ...fill this... }"
  #error "static __inline__ int fastlock_trylock(fastlock_t * v) { ...fill this... return +1 on success, 0 on failure, (optional -1 on error) }"
  #error "static __inline__ void fastlock_lock(fastlock_t *v)  { while (fastlock_trylock(v) <= 0) { usleep(0}; } }"
  #error ""
  #error "some code to look at/for..."
  #error "the simplest solution: netbsd's include/machine/lock.h (http://cvsweb.netbsd.org/bsdweb.cgi/syssrc/sys/arch/*/include/lock.h)"
  #error "atomic test-and-set tas(). [lots of places on the net, like postgres]"
  #error "atomic compare_and_swap eg, http://lxr.linux.no/source/include/asm-XXX/system.h"
  #error "               or gcc source libstdc++-v3/config/cpu/XXX/bits/atomicity.h"
  #error "atomic [inc|dec]rement eg, http://lxr.linux.no/source/include/asm-XXX/atomic.h"

#endif

#endif /* __CLISYNC_H__ */

