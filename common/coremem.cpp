/*
 * Copyright distributed.net 1997-2011 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Allocation/free of memory used by crunchers. For crunchers that were
 * created with fork(), the memory is shared between client and cruncher.
 *
 * Created March 2001 by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/
const char *coremem_cpp(void) {
  return "@(#)$Id: coremem.cpp,v 1.10 2011/03/31 05:07:28 jlawson Exp $";
}

//#define TRACE

#include "cputypes.h" /* HAVE_MULTICRUNCH_VIA_FORK define */
#include "baseincs.h" /* malloc/free/mmap/munmap */
#include "util.h"     /* trace */
#include "coremem.h"  /* ourselves */


#if defined(HAVE_MULTICRUNCH_VIA_FORK)

#if ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_HPUX))
/* MAP_ANON|MAP_SHARED is completely unsupported in linux */
/* <=2.2.2, and flakey in 2.2.3. MAP_SHARED is broken in 2.0 */
/* MAP_SHARED is broken on HPUX <= 9.0 */
#include <sys/ipc.h>
#include <sys/shm.h>
#define USE_SYSV_SHM
#endif

// 0 = shared memory (default)
// 1 = malloc (only for -numcpu 0)
static int selected_allocator = -1;
#define default_allocator 0

/* you may call this exactly one time before calling cmem_alloc/cmem_free */
void cmem_select_allocator(int which)
{
  if (selected_allocator < 0)
  {
    if (which == 1) /* malloc for -numcpu 0 */
      selected_allocator = 1;
    else
      selected_allocator = 0; /* default */
  }
}

static void *__shm_cmem_alloc(unsigned int sz)
{
  void *mem = ((void *)0);
  sz += sizeof(void *);
  #if 0 /* may be needed some places */
  {
    unsigned int pgsize = getpagesize();
    if ((sz % pgsize) != 0)
      sz += pgsize - (sz % pgsize);
  }
  #endif
  #if (CLIENT_OS == OS_NEXTSTEP)
  if (vm_allocate(task_self(), (vm_address_t *)&mem,
                  sz, TRUE) != KERN_SUCCESS)
    return NULL;

  if (vm_inherit(task_self(), (vm_address_t)mem, sz,
                 VM_INHERIT_SHARE) != KERN_SUCCESS) {
    vm_deallocate(task_self(), (vm_address_t)mem, sz);
    return NULL;
  }
  #elif defined(USE_SYSV_SHM)
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
               sz, mem, ((mem) ? ("") : (strerror(errno))) ));
  }
  #else /* SysV style */
  {
    int fd = open("/dev/zero", O_RDWR);
    if (fd != -1)
    {
      mem = mmap( 0, sz, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
      if (mem == ((void *)-1))
        mem = (void *)0;
      close(fd);
    }
    TRACE_OUT((0, "mmap(0, %u, ..., MAP_SHARED, fd=%d, 0)=%p\n%s\n",
               sz, fd, mem, ((mem) ? ("") : (strerror(errno))) ));
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
  return mem;
}

static int __shm_cmem_free(void *mem)
{
  {
    char *p = (char *)mem;
    p -= sizeof(void *);
    mem = (void *)p;
  }
  #if (CLIENT_OS == OS_NEXTSTEP)
  if (vm_deallocate(task_self(), (vm_address_t)mem,
                    (vm_size_t)*((unsigned int *)mem)) != KERN_SUCCESS)
    return -1;

  return 0;
  #elif defined(USE_SYSV_SHM)
  TRACE_OUT((0,"shmdt(%p)\n", mem));
  return shmdt((char *)mem);
  #else
  TRACE_OUT((0,"munmap(%p,%u)\n", mem, *((unsigned int *)mem)));
  return munmap((void *)mem, *((unsigned int *)mem));
  #endif
}

static void *__malloc_cmem_alloc(unsigned int sz)
{
  void *mem = ((void *)0);
  mem = malloc(sz);
  return mem;
}

static int __malloc_cmem_free(void *mem)
{
  free(mem);
  return 0;
}

void *cmem_alloc(unsigned int sz)
{
  if (selected_allocator < 0)
    cmem_select_allocator(default_allocator);

  if (selected_allocator == 1)
    return __malloc_cmem_alloc(sz);
  else
    return __shm_cmem_alloc(sz);
}

int cmem_free(void *mem)
{
  if (selected_allocator < 0)
    cmem_select_allocator(default_allocator);

  if (selected_allocator == 1)
    return __malloc_cmem_free(mem);
  else
    return __shm_cmem_free(mem);
}

#elif (CLIENT_OS == OS_AMIGAOS)

void *cmem_alloc(unsigned int sz)
{
  void *mem = ((void *)0);
  #if defined(__amigaos4__)
  mem = AllocVec(sz,MEMF_SHARED);
  #else
  mem = malloc(sz);
  #endif
  return mem;
}

int cmem_free(void *mem)
{
  #if defined(__amigaos4__)
  FreeVec(mem);
  #else
  free(mem);
  #endif
  return 0;
}

#else

void *cmem_alloc(unsigned int sz)
{
  void *mem = ((void *)0);
  mem = malloc(sz);
  return mem;
}

int cmem_free(void *mem)
{
  free(mem);
  return 0;
}

#endif /* defined(HAVE_MULTICRUNCH_VIA_FORK) */
