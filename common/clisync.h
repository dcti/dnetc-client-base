/* Hey, Emacs, this a -*-C++-*- file !
 * 
 * Intra-process synchronization primitives.
 * Written Jan 2001 by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * Although based on the prototypes as used in Solaris/SunOS
 * eg http://hoth.stsci.edu/man/man3T/ , they are used in the client 
 * for lightweight protection of (small and fast) critical sections, 
 * and are therefore intended to behave more like spinlocks than mutexes.
*/
#ifndef __CLISYNC_H__
#define __CLISYNC_H__ "@(#)$Id: clisync.h,v 1.1.2.7 2001/02/09 08:47:16 mfeiri Exp $"

#include "cputypes.h"           /* thread defines */
#include "sleepdef.h"           /* NonPolledUSleep() */

#if !defined(CLIENT_SUPPORTS_SMP) /* non-threaded client */

  typedef struct { long spl; } mutex_t;
  #define DEFAULTMUTEX {0}
  static inline void mutex_lock(mutex_t *m)   { m->spl = 1; }
  static inline void mutex_unlock(mutex_t *m) { m->spl = 0; }
  static inline int mutex_trylock(mutex_t *m) { m->spl = 1; return +1; }
  /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */

#elif defined(_POSIX_THREADS_SUPPORTED)

  #include <pthread.h>
  #define mutex_t pthread_mutex_t
  #define DEFAULTMUTEX PTHREAD_MUTEX_INITIALIZER
  #define mutex_lock pthread_mutex_lock
  #define mutex_unlock pthread_mutex_unlock
  #define mutex_trylock pthread_mutex_trylock

#elif (CLIENT_OS == OS_MACOS)

  #include <Multiprocessing.h>
//  Without "if (MPLibraryIsLoaded())" this could be really nice
//  #define mutex_t MPCriticalRegionID
//  #define DEFAULTMUTEX 0
//  #define mutex_lock(x) MPEnterCriticalRegion(*x,kDurationImmediate)
//  #define mutex_unlock(x) MPExitCriticalRegion(*x)
//  static inline int mutex_trylock(mutex_t *m)
//  { return (MPEnterCriticalRegion(*m,kDurationImmediate)?(0):(+1)); }

  typedef struct {MPCriticalRegionID MPregion; long spl;} mutex_t;
  #define DEFAULTMUTEX {0,0}
  
  static inline void mutex_lock(mutex_t *m)
  {
    if (MPLibraryIsLoaded())
      MPEnterCriticalRegion(m->MPregion,kDurationImmediate);
    else
      m->spl = 1;
  }
  
  static inline void mutex_unlock(mutex_t *m)
  {
    if (MPLibraryIsLoaded())
      MPExitCriticalRegion(m->MPregion);
    else
      m->spl = 0;
  }
  
  /* Please verify this is working as expected before using it
  static inline int mutex_trylock(mutex_t *m)
  {
    if (MPLibraryIsLoaded())
    {
      return (MPEnterCriticalRegion(m->MPregion,kDurationImmediate)?(0):(+1));
    }
    else
    { m->spl = 1; return +1; }
  }
   */

#elif (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS)
                               
  #include <thread.h>
  #include <synch.h>

#elif (CLIENT_OS == OS_WIN32) && (CLIENT_CPU == CPU_ALPHA)

   typedef struct 
   { 
     #pragma pack(8)
     long spl; 
     #pragma pack()
   } mutex_t;
   #define DEFAULTMUTEX {0}
   extern "C" int _AcquireSpinLockCount(long *, int);
   extern "C" void _ReleaseSpinLock(long *);
   #pragma intrinsic(_AcquireSpinLockCount, _ReleaseSpinLock)
   static inline void mutex_lock(mutex_t *m)
   {
     while (!_AcquireSpinLockCount(&(m->spl), 64))
       Sleep(1);
   }
   static inline void mutex_unlock(mutex_t *m)
   {
     _ReleaseSpinLock(&(m->spl));
   }
   /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */
   static inline int mutex_trylock(mutex_t *m)
   {
     if (!_AcquireSpinLockCount(&(m->spl),1))
       return 0;
     return +1;
   }

#elif (CLIENT_CPU == CPU_X86)

   typedef struct
   { 
     #pragma pack(4)
     long spl; 
     #pragma pack()
   } mutex_t;
   #define DEFAULTMUTEX {0}
   /* _trylock returns -1 on EINVAL, 0 if could not lock, +1 if could lock */
   static inline int mutex_trylock(mutex_t *m)
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
   static inline void mutex_unlock(mutex_t *m)
   {
     m->spl = 0;
   }
   static inline void mutex_lock(mutex_t *m)
   {
     while (mutex_trylock(m) <= 0)
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

  typedef struct { volatile int spl; } mutex_t;
  #define DEFAULTMUTEX {0}
  static __inline__ void mutex_unlock(mutex_t *m)
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
  static __inline__ int mutex_try_lock(mutex_t *m)
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
       mutex_unlock(m); /* undo the increment */
       t = 0;
    }
    return t;
  }
  static __inline__ void mutex_lock(mutex_t *m)
  {
    while (mutex_try_lock(m) <= 0)
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
  typedef struct { volatile char spl; } mutex_t;
  #define DEFAULTMUTEX {0}
  static __inline__ void mutex_unlock(mutex_t *m)
  { 
    /* m->spl = 0; */
    __asm__  __volatile__ (
             "clr.b %0"         \
             : "=m"  (m->spl)   \
             :  "0"  (m->spl));
  }
  static __inline__ char mutex_try_lock(mutex_t *m)
  {
    char lacquired;
    __asm__  __volatile__ (
             "bset #0,%1\n"      \
             "seq %0"            \
             : "=d" (lacquired)  \
             :  "m" (m->spl));
    return lacquired;  // -1 = acquired, 0 = not aquired
  }
  static __inline__ void mutex_lock(mutex_t *m)
  {
    do
    {
      if (mutex_try_lock(m) != 0) return;
      #if (CLIENT_OS == OS_AMIGAOS)
      NonPolledUSleep(1);
      #else
      #error "What's up Doc?"
      #endif
    } while(1);
  }

#else
   #error How did you get here?
#endif

#endif /* __CLISYNC_H__ */
