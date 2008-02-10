/* -*-C++-*-
 *
 * Copyright distributed.net 2001-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
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
 *
 * 2003-08-29 Michael Weiser <michael@weiser.dinsnail.net> cleanup:
 *
 * fastlock_t used to be a structure with one volatile element. The
 * element being declared volatile ensured that compiler optimisations
 * didn't change order of operations on the lock and broke things. It
 * was placed inside a structure to be able to copy it using memcpy()
 * for initialisation without compiler warnings/errors. The structure
 * definitions included heavy padding to ensure the locking element
 * came to lie on proper boundaries for the assembler operations doing
 * the actual locking operations. Initialisation of the lock member
 * was done using a define FASTLOCK_INITIALIZER_UNLOCKED which was
 * mostly ((fastlock_t)({0})).
 *
 * Problems with this scheme:
 * - a structure with one volatile element is ugly to look at and work
 *   with
 *
 * - FASTLOCK_INITIALIZER_UNLOCKED breaks with gcc-3.3
 *    common/problem.cpp: In function `Problem* ProblemAlloc()':
 *    common/problem.cpp:267: error: `fastlock_t' has no non-static
 *     data member named `fastlock_t::spl'
 *    common/problem.cpp:267: error: too many initializers for `fastlock_t'
 *    common/problem.cpp:267: warning: missing initializer for member `
 *     fastlock_t::spl'
 *
 * - the padding concept never actually worked because fastlock_t was
 *   put into the packed structure SuperProblem in problem.cpp. There it
 *   could've ended up on an odd byte boundary all the time if memory
 *   allocation routines wouldn't align to something like word
 *   boundaries for performance anyway and the three InternalProblems in
 *   front of fastlock_t hadn't been padded
 *
 * Therefore we go back to a simpler scheme:
 *
 * - fastlock_init() is introduced instead of the macro and
 *   platform-wise implements initialisation for fastlock_t however
 *   it likes
 *
 * - fastlock_t is reverted to a simple volatile typedef. Alignment
 *   attributes shouldn't be necessary because compilers align the
 *   datatype to its natural boundary anyway. When used in packed
 *   structures alignment has to be done on usage not definition anyway.
 */

#ifndef __CLISYNC_H__
#define __CLISYNC_H__ "@(#)$Id: clisync.h,v 1.8 2008/02/10 00:24:30 kakace Exp $"

#include "cputypes.h"           /* thread defines */
#include "sleepdef.h"           /* NonPolledUSleep() */

#if !defined(CLIENT_SUPPORTS_SMP) /* non-threaded client */

  typedef unsigned int fastlock_t;
  static inline void fastlock_init(fastlock_t *l)   { *l = 0; }
  static inline void fastlock_lock(fastlock_t *l)   { *l = 1; }
  static inline void fastlock_unlock(fastlock_t *l) { *l = 0; }

  /* _trylock returns -1 on EINVAL, 0 if could not lock, 1 if could lock */
  static inline int fastlock_trylock(fastlock_t *l) { *l = 1; return 1; }

#elif (CLIENT_CPU == CPU_ALPHA) && defined(__GNUC__)

  typedef volatile unsigned int fastlock_t;

  static inline void fastlock_init(fastlock_t *l) {
    *l = 0;
  }

  static inline void fastlock_unlock(fastlock_t *l) {
    asm volatile("mb": : :"memory");
    *l = 0;
  }

  /* http://lxr.linux.no/source/include/asm-alpha/bitops.h?v=2.4.0#L92 */
  extern inline int
  test_and_set_bit(unsigned long nr, volatile void *addr) {
    unsigned long oldbit;
    unsigned long temp;
    int *m = ((int *) addr) + (nr >> 5);

#if (CLIENT_OS == OS_DEC_UNIX)
     /* quad-word aligned branch targets are optimal, backwards
       branches are predicted as outcome, while forwards branches are
       predicted as fall-through. misprediction can cost 10
       instruction cycles, leading to double branches on infrequent case
       (forwards then backwards) being more optimal */
    asm volatile("1:     ldl_l %0,%4   \n\t" \
                 "       and %0,%3,%2  \n\t" \
                 "       bne %2,2f     \n\t" \
                 "       xor %0,%3,%0  \n\t" \
                 "       stl_c %0,%1   \n\t" \
                 "       beq %0,3f     \n\t" /* begin double branch */ \
                 "       unop          \n\t" /*align branch target below */ \
                 "       unop          \n\t" /*align branch target below */ \
                 "2:     mb            \n\t" \
                 "       br 4f         \n\t" \
                 "       unop          \n\t" /*align branch target below */ \
                 "       unop          \n\t" /*align branch target below */ \
                 "3:     br 1b         \n\t" /* on infrequent case double */ \
                                             /* branch is faster due to */ \
                                             /* i-cache mispredictions */ \
                 "       unop          \n\t" /*align branch target below */ \
                 "       unop          \n\t" /*align branch target below */ \
                 "       unop          \n\t" /*align branch target below */ \
                 "4:     "
#else
    asm volatile("1:     ldl_l %0,%4   \n\t" \
                 "       and %0,%3,%2  \n\t" \
                 "       bne %2,2f     \n\t" \
                 "       xor %0,%3,%0  \n\t" \
                 "       stl_c %0,%1   \n\t" \
                 "       beq %0,3f     \n\t" \
                 "2:     mb            \n\t" \
                 ".subsection 2        \n\t" \
                 "3:     br 1b         \n\t" \
                 ".previous"
#endif
                 :"=&r" (temp), "=m" (*m), "=&r" (oldbit)
                 :"Ir" (1UL << (nr & 31)), "m" (*m) : "memory");

    return oldbit != 0;
  }

  static inline int fastlock_trylock(fastlock_t *l) {
    if (!test_and_set_bit(0, l))
      return 1;

    return 0;
  }

  static inline void fastlock_lock(fastlock_t *l) {
    while (fastlock_trylock(l) <= 0)
      NonPolledUSleep(1);
  }

#elif (CLIENT_CPU == CPU_ALPHA) && defined(_MSC_VER)

  typedef long fastlock_t;

  static inline void fastlock_init(fastlock_t *l) {
    *l = 0;
  }

  /* MS VC 6 has spinlock intrinsics */
  extern "C" int _AcquireSpinLockCount(long *, int);
  extern "C" void _ReleaseSpinLock(long *);
# pragma intrinsic(_AcquireSpinLockCount, _ReleaseSpinLock)

  static inline void fastlock_lock(fastlock_t *l) {
    while (!_AcquireSpinLockCount(l, 64))
      Sleep(1);
  }

  static inline void fastlock_unlock(fastlock_t *l) {
    _ReleaseSpinLock(l);
  }

  /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */
  static inline int fastlock_trylock(fastlock_t *l) {
    if (!_AcquireSpinLockCount(l,1))
      return 0;

    return 1;
  }

#elif (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64)

  typedef volatile long fastlock_t;

  static inline void fastlock_init(fastlock_t *l) {
    *l = 0;
  }

  /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */
  static inline int fastlock_trylock(fastlock_t *l) {
    int lacquired = 0;

# if defined(__GNUC__)
    /* note: no 'lock' prefix even on SMP since xchg is always atomic */
    asm volatile("movl  $1,%0 \n\t" \
                 "xchgl %0,%1 \n\t" \
                 "xorl  $1,%0"
                 : "=r"(lacquired)
                 : "m"(*l)
                 : "memory");
# elif defined(__BORLANDC__) /* BCC can't do inline assembler in inline functions */
    _EDX = (unsigned long)l;
    _EAX = 1;
    __emit__(0x87, 0x02); /* xchg [edx],eax */
    _EAX ^= 1;
    lacquired = _EAX;
# else
    _asm mov edx, l
    _asm mov eax, 1
    _asm xchg eax,[edx]
    _asm xor eax, 1
    _asm mov lacquired,eax
# endif

    if (lacquired)
      return 1;

    return 0;
  }

  static inline void fastlock_unlock(fastlock_t *l) {
    *l = 0;
  }

  static inline void fastlock_lock(fastlock_t *l) {
    while (fastlock_trylock(l) <= 0) {
# if defined(__unix__)
      NonPolledUSleep(1);
# elif (CLIENT_OS == OS_NETWARE6)
      NonPolledUSleep(1);
# elif (CLIENT_OS == OS_NETWARE)
      ThreadSwitchLowPriority();
# elif (CLIENT_OS == OS_OS2)
      DosSleep(1);
# elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
      Sleep(1);
# else
#  error "What's up Doc?"
# endif
    }
  }

#elif (CLIENT_OS == OS_AIX)

  /* there's no assembly for fastlocks on Power and the PowerPC code
  ** has to use local labels which the AIX system as doesn't
  ** support. So (for now) we use the AIX atomic ops on both PowerPC
  ** and Power. As soon as GNU binutils properly support AIX we can
  ** switch back to the PowerPC asm code with local labels. */

# include <sys/atomic_op.h>

  typedef int fastlock_t;

  static inline void fastlock_init(fastlock_t *l) {
    *l = 0;
  }

  static inline void fastlock_unlock(fastlock_t *l) {
    /* Atomically writes a single word variable, issuing an export
    ** fence for multiprocessor systems */
    _clear_lock(l, 0);
  }

  static inline int fastlock_trylock(fastlock_t *l) {
    /* Conditionally updates a single word variable atomically,
    ** issuing an import fence for multiprocessor systems. */
    if (!_check_lock(l, 0, 1 ))
      return 1;

    return 0;
  }

  static inline void fastlock_lock(fastlock_t *l) {
    while (fastlock_trylock(l) <= 0) {
      NonPolledUSleep(1);
    }
  }

#elif ((CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)) && defined(__GNUC__)

  /* based on
  ** http://cvsweb.netbsd.org/bsdweb.cgi/syssrc/sys/arch/powerpc/include/lock.h?rev=1.4.2.1
  ** approved by Dan Oetting */
  typedef volatile int fastlock_t;

  static inline void fastlock_init(fastlock_t *l) {
    *l = 0;
  }

  static inline void fastlock_unlock(fastlock_t *l) {
    asm volatile ("sync");
    *l = 0;
  }

  static inline int fastlock_trylock(fastlock_t *l) {
    int old, dummy;

    asm volatile (/* compare_and_swap */
                  "1:      lwarx  %0,0,%1 \n" \
                  "        cmpwi  %0,%2   \n" \
                  "        bne    2f      \n" \
                  "        stwcx. %3,0,%1 \n" \
                  "        bne-   1b      \n" \
                  "2:      stwcx. %3,0,%4 \n" \
                  "        isync"
                  : "=&r"(old)
                  : "r"(l),
                  "I"(0), /* unlocked state */
                  "r"(1), /* locked state */
                  "r"(&dummy)
                  : "memory");

    if (old == 0) /* old state was 0? */
      return 1;   /* then success */

    return 0;
  }

  static inline void fastlock_lock(fastlock_t *l) {
    while (fastlock_trylock(l) <= 0) {
# if (CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_MORPHOS)
      NonPolledUSleep(1000);
# elif defined(__unix__)
      NonPolledUSleep(1);
# else
#  error "What's up Doc?"
# endif
    }
  }

#elif (CLIENT_CPU == CPU_68K) && defined(__GNUC__)

  /* IMPORTANT: has to be a char (not an int) since when the destination for
  ** BTST is a memory location, the operation must be a byte operation */
  typedef volatile char fastlock_t;

  static inline void fastlock_init(fastlock_t *l) {
    *l = 0;
  }

  static inline void fastlock_unlock(fastlock_t *l) {
    /* l = 0; */
    asm volatile ("clrb %0"
                  : "=m"  (*l)
                  :  "0"  (*l));
  }

  /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */
  static inline int fastlock_trylock(fastlock_t *l) {
    char lacquired;

    asm volatile ("bset #0,%1 \n\t" \
                  "seq %0"
                  : "=d" (lacquired)
                  :  "m" (*l));

    if (lacquired)
      return 1;

    return 0;
  }

  static inline void fastlock_lock(fastlock_t *l) {
    while (fastlock_trylock(l) <= 0) {
# if (CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_MORPHOS)
      NonPolledUSleep(1000);
# elif defined(__unix__)
      NonPolledUSleep(1);
# else
#  error "What's up Doc?"
# endif
    }
  }

#elif (CLIENT_CPU == CPU_PA_RISC) && defined(__GNUC__)

  typedef volatile int fastlock_t;

  static inline void fastlock_init(fastlock_t *l) {
    /* note! "unlocked" == -1 */
    *l = -1;
  }

  static inline void fastlock_unlock(fastlock_t *l) {
    *l = -1;
  }

  /* based on gnu libc source in sysdeps/mach/hppa/machine-lock.h */
  static inline int fastlock_trylock(fastlock_t *l) {
    register int result;

    /* LDCW, the only atomic read-write operation PA-RISC has. Sigh. */
    asm volatile ("ldcws %0, %1" : "=m" (*l), "=r" (result));

    /* __result is non-zero if we locked it */
    if (result != 0)
      return 1;

    return 0;
  }

  static inline void fastlock_lock(fastlock_t *l) {
    while (fastlock_trylock(l) <= 0) {
      NonPolledUSleep(1);
    }
  }


#elif ((CLIENT_CPU == CPU_S390) || (CLIENT_CPU == CPU_S390X)) && \
      defined(__GNUC__)
  /* based on
  ** http://lxr.linux.no/source/include/asm-s390/spinlock.h?a=s390 */

  typedef volatile unsigned long fastlock_t;

  static inline void fastlock_init(fastlock_t *l) {
    *l = 0;
  }

  static inline void fastlock_unlock(fastlock_t *l) {
    asm volatile("xc 0(4,%0),0(%0) \n\t" \
                 "bcr 15,0"
                 : : "a" (l) : "memory", "cc" );
  }

  static inline int fastlock_trylock(fastlock_t *l) {
    unsigned long result, reg;

    asm volatile("        slr   %0,%0 \n" \
                 "        basr  %1,0  \n"  \
                 "0:      cs    %0,%1,0(%2)"
                 : "=&d" (result), "=&d" (reg)
                 : "a" (l) : "cc", "memory" );

    return !result;
  }

  static inline void fastlock_lock(fastlock_t *l) {
    while (fastlock_trylock(l) <= 0) {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_IA64) && defined(__GNUC__)

  /* based on
  ** http://lxr.linux.no/source/include/asm-ia64/spinlock.h?v=2.4.0 */
  typedef volatile unsigned int fastlock_t;

  static inline void fastlock_init(fastlock_t *l) {
    *l = 0;
  }

  static inline void fastlock_unlock(fastlock_t *l) {
    *l = 0;
  }

  static inline int fastlock_trylock(fastlock_t *l) {
    register long result;

    /* IA64_SEMFIX is a workaround for Errata 97. (A-step through B1) */
# define IA64_SEMFIX  "mf;"

    asm volatile ("mov ar.ccv=r0 \n\t" \
                  ";; \n\t"            \
                  IA64_SEMFIX"cmpxchg4.acq %0=[%2],%1,ar.ccv"
                  : "=r"(result)
                  : "r"(1), "r"(l)
                  : "ar.ccv", "memory");

    if (result == 0)
      return 1;

    return 0;
  }

  static inline void fastlock_lock(fastlock_t *l) {
    while (fastlock_trylock(l) <= 0) {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_SPARC) && defined(__GNUC__)

  /* based on
  ** http://cvsweb.netbsd.org/bsdweb.cgi/syssrc/sys/arch/sparc/include/lock.h?rev=1.6.2.1
  ** http://cvsweb.netbsd.org/bsdweb.cgi/syssrc/sys/arch/sparc64/include/lock.h?rev=1.4.2.1 */
  typedef volatile int fastlock_t;

  static inline void fastlock_init(fastlock_t *l) {
    *l = 0;
  }

  static inline int fastlock_trylock(fastlock_t *l) {
    int v;

    __asm volatile("ldstub [%1],%0"
                   : "=r" (v)
                   : "r" (l)
                   : "memory");

    /* old state was unlocked */
    if (v == 0)
      return 1;

    return 0;
  }

  static inline void fastlock_unlock(fastlock_t *l){
    *l = 0;
  }

  static inline void fastlock_lock(fastlock_t *l) {
    while (fastlock_trylock(l) <= 0) {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_VAX) && defined(__GNUC__)

  typedef volatile int fastlock_t;

  static inline void fastlock_init(fastlock_t *l) {
    *l = 0;
  }

  /* based on
  ** http://cvsweb.netbsd.org/bsdweb.cgi/syssrc/sys/arch/vax/include/lock.h?rev=1.5.2.1 */
  static inline void fastlock_unlock(fastlock_t *l) {
    *l = 0;
  }

  static inline int fastlock_trylock(fastlock_t *l) {
    int ret;

    asm volatile ("movl $0,%0;bbssi $0,%1,1f;incl %0;1:"
                  : "=&r"(ret)
                  : "m"(*l));

    if (ret != 0)
      return 1;

    return 0;
  }

  static inline void fastlock_lock(fastlock_t *m) {
    while (fastlock_trylock(m) <= 0) {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_SH4) && defined(__GNUC__)

  typedef volatile int fastlock_t;

  static inline void fastlock_init(fastlock_t *l) {
    *l = 0;
  }

  static inline void fastlock_unlock(fastlock_t *l) {
    *l = 0;
  }

  static inline int fastlock_trylock(fastlock_t *l) {
    unsigned long retval;

    /* Have 'atomic test-and-set' instruction. Attempt to acquire the
    ** lock, but do not wait. Returns 0 if successful, nonzero if
    ** unable to acquire the lock. */
    asm volatile ("tas.b @%1 \n\t" \
                  "movt  %0"
                  : "=r" (retval): "r" (l): "t", "memory");

    /* old state was unlocked */
    if (retval == 0)
      return 1;

    return 0;
  }

  static inline void fastlock_lock(fastlock_t *l) {
    while (fastlock_trylock(l) <= 0) {
      NonPolledUSleep(1);
    }
  }

#elif (CLIENT_CPU == CPU_ARM) && defined(__GNUC__)

  typedef volatile long int fastlock_t;

  static inline void fastlock_init(fastlock_t *l) {
    *l = 0;
  }

  static inline void fastlock_unlock(fastlock_t *l) {
    *l = 0;
  }

  static inline int fastlock_trylock(fastlock_t *l) {
    int result;

    asm volatile ("mov     %0,#1      \n\t" \
                  "swp     %0,%0,[%1] \n\t" \
                  "cmp     %0,#0      \n\t" \
                  "movne   %0,#0      \n\t" \
                  "moveq   %0,#1"
                  : "=&r" (result) /* If it has already been locked */
                  : "r" (l)        /* the 1 can stay there !        */
                  : "cc", "memory");
    return result;
  }

  static inline void fastlock_lock(fastlock_t *l) {
    while (fastlock_trylock(l) <= 0) {
      NonPolledUSleep(1);
    }
  }

#elif defined(_POSIX_THREADS_SUPPORTED)
  /* heaaaavyweight, but better than nothing */

# include <pthread.h>
# define fastlock_t       pthread_mutex_t

  /* pthread_mutex_init always returns zero */
# define fastlock_init(l) pthread_mutex_init(l, NULL)
# define fastlock_lock    pthread_mutex_lock
# define fastlock_unlock  pthread_mutex_unlock
# define fastlock_trylock pthread_mutex_trylock

#elif (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS)
  /* heaaaavyweight, but better than nothing */

# include <thread.h>
# include <synch.h>
# define fastlock_t       mutex_t

# define fastlock_init(l) mutex_init(l, NULL, NULL)
# define fastlock_lock    mutex_lock
# define fastlock_unlock  mutex_unlock
# define fastlock_trylock mutex_trylock

#elif defined(linux)

#define __LINUX_SPINLOCK_TYPES_H
#include <asm/spinlock_types.h>
#undef __LINUX_SPINLOCK_TYPES_H
#include <asm/spinlock.h>

#define fastlock_t            raw_spinlock_t
#define fastlock_init(l)      *(l) = (raw_spinlock_t)__RAW_SPIN_LOCK_UNLOCKED
#define fastlock_lock         __raw_spin_lock
#define fastlock_unlock       __raw_spin_unlock
#define fastlock_trylock      __raw_spin_trylock

#else
  /* can't do atomic operations on mips ISA < 2 without kernel support */

  /* put this at the end, so that more people notice/are affected by
  ** any potential problems in other code */
# error "Whats required here is ... (or define SINGLE_CRUNCHER_ONLY)"
# error ""
# error "typedef [...fill_this...] fastlock_t;"
# error ""
# error "static inline void fastlock_init(fastlock_t *l) { ...fill this... }"
# error "static inline void fastlock_unlock(fastlock_t *l) { ...fill this... }"
# error "static inline int fastlock_trylock(fastlock_t *l) { ...fill this... return 1 on success, 0 on failure, (optional -1 on error) }"
# error "static inline void fastlock_lock(fastlock_t *l) { while (fastlock_trylock(l) <= 0) { usleep(0}; } }"
# error ""
# error "some code to look at/for..."
# error "the simplest solution: netbsd's include/machine/lock.h (http://cvsweb.netbsd.org/bsdweb.cgi/syssrc/sys/arch/*/include/lock.h)"
# error "atomic test-and-set tas(). [lots of places on the net, like postgres]"
# error "atomic compare_and_swap eg, http://lxr.linux.no/source/include/asm-XXX/system.h"
# error "               or gcc source libstdc++-v3/config/cpu/XXX/bits/atomicity.h"
# error "atomic [inc|dec]rement eg, http://lxr.linux.no/source/include/asm-XXX/atomic.h"

#endif

#endif /* __CLISYNC_H__ */

