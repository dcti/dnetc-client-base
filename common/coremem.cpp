/*
 * Copyright distributed.net 1997-2001 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Allocation/free of memory used by crunchers. For crunchers that were
 * created with fork(), the memory is shared between client and cruncher.
 *
 * Created March 2001 by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/
const char *probmem_cpp(void) {
return "@(#)$Id: coremem.cpp,v 1.1.2.3 2001/03/20 18:26:27 cyp Exp $"; }

//#define TRACE

#include "cputypes.h" /* HAVE_MULTICRUNCH_VIA_FORK define */
#include "baseincs.h" /* malloc/free/mmap/munmap */
#include "util.h"     /* trace */
#include "coremem.h"  /* ourselves */

#if defined(HAVE_MULTICRUNCH_VIA_FORK) && \
  ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_HPUX))
    /* MAP_ANON|MAP_SHARED is completely unsupported in linux */
    /* <=2.2.2, and flakey in 2.2.3. MAP_SHARED is broken in 2.0 */
    /* MAP_SHARED is broken on HPUX <= 9.0 */
#include <sys/ipc.h>
#include <sys/shm.h>
#define USE_SYSV_SHM
#endif

void *cmem_alloc(unsigned int sz)
{
  void *mem = ((void *)0);
#if defined(HAVE_MULTICRUNCH_VIA_FORK)
  sz += sizeof(void *);
  #if 0 /* may be needed some places */
  {
    unsigned int pgsize = getpagesize();
    if ((sz % pgsize) != 0)
      sz += pgsize - (sz % pgsize);
  }
  #endif
  #if defined(USE_SYSV_SHM)
  {
    int shmid = shmget(IPC_PRIVATE, sz, 0600 );
    if (shmid != -1)
    {
      mem = (void *)shmat(shmid, 0, 0 );
      shmctl( shmid, IPC_RMID, 0);
    } 
    TRACE_OUT((0,"shmat(%d)=>%p\n", shmid, mem));
  }
  #elif defined(MAP_ANON) /* BSD style */
  {
    mem = mmap( 0, sz, PROT_READ|PROT_WRITE, MAP_ANON|MAP_SHARED, -1, 0);
    if (mem == ((void *)-1))
      mem = (void *)0;
    TRACE_OUT((0,"mmap(0, %u, ..., MAP_ANON|MAP_SHARED, -1, 0)=%p\n%s\n",
                  sz, mem, ((mem)?(""):(strerror(errno))) )); 
  }
  #else /* SysV style */
  {
    int fd = open("/dev/zero", O_RDWR);
    if (fd != -1)
    {
      mem = mmap( 0, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
      if (mem == ((void *)-1))
        mem = (void *)0;
      close(fd);
    }
    TRACE_OUT((0, "mmap(0, %u, ..., MAP_SHARED, fd=%d, 0)=%p\n%s\n",
                  sz, fd, mem, ((mem)?(""):(strerror(errno))) )); 
  }
  #endif
  if (mem)
  {
    char *p = (char *)mem;
    memset( mem, 0, sz );
    *((unsigned int *)mem) = sz;
    p += sizeof(void *);
    mem = (void *)p;
  }
#else
  mem = malloc(sz);
#endif      
  return mem;
}    
  
int cmem_free(void *mem)
{    
#if defined(HAVE_MULTICRUNCH_VIA_FORK)
  {
    char *p = (char *)mem;
    p -= sizeof(void *);
    mem = (void *)p;
  }  
  #if defined(USE_SYSV_SHM)
  TRACE_OUT((0,"shmdt(%p)\n", mem));
  return shmdt((char *)mem);
  #else
  TRACE_OUT((0,"munmap(%p,%u)\n", mem, *((unsigned int *)mem)));
  return munmap((void *)mem, *((unsigned int *)mem));
  #endif
#else
  free(mem);
  return 0;
#endif  
}  
