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
return "@(#)$Id: coremem.cpp,v 1.1.2.2 2001/03/20 14:58:51 cyp Exp $"; }

#include "cputypes.h" /* HAVE_MULTICRUNCH_VIA_FORK define */
#include "baseincs.h" /* malloc/free/mmap/munmap */
#include "coremem.h"  /* ourselves */

void *cmem_alloc(unsigned int sz)
{
  void *mem;
#if defined(HAVE_MULTICRUNCH_VIA_FORK)
  #if defined(MAP_ANON) /* BSD style */ \
    && (CLIENT_OS != OS_LINUX) /* MAP_ANON|MAP_SHARED is broken in < 2.2.2 */
  {
    sz += sizeof(void *);
    mem = mmap( 0, sz, PROT_READ|PROT_WRITE, MAP_ANON|MAP_SHARED, -1, 0);
  }
  #else /* Sun style */
  {
    int fd = open("/dev/zero", O_RDWR);
    mem = ((void *)-1);
    sz += sizeof(void *);
    if (fd != -1)
    {
      mem = mmap( 0, sz, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
      close(fd);
    }
  }
  #endif
  if (mem == ((void *)-1))
    mem = (void *)0;
  if (mem)
  {
    char *p = (char *)mem;
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
  char *p = (char *)mem;
  unsigned int sz;
  p -= sizeof(void *);
  sz = *((unsigned int *)p);
  return munmap((void *)p, sz);
#else
  free(mem);
  return 0;
#endif  
}  
