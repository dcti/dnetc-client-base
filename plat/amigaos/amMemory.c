/* Created by Oliver Roberts <oliver@futaura.co.uk>
**
** $Id: amMemory.c,v 1.1.2.1 2001/01/21 15:10:27 cyp Exp $
**
** ----------------------------------------------------------------------
** This file contains replacement stdlib memory allocation routines,
** utilizing system memory pool routines.  Note: the main reason I'm
** using custom routines is because the 68k libnix memory routines were
** found to be causing crashes on some systems - but, besides, using
** memory pools for 68k should be more efficient anyway (decided to add
** WarpOS/PowerUp support too)
** ----------------------------------------------------------------------
*/

#ifndef __POWERUP__
/*
** Use libnix memory routines for PowerUp since there seems to be a bug
** in the PowerUp's semaphore routines (or the memory pool routines) which
** causes stability problems
*/

#include "amiga.h"

static void *MemPool = NULL;

/* Memory pools cannot by accessed by more than one task at the same
** time, so we must use semaphores to protect against such cases
*/
#ifndef __PPC__
static struct SignalSemaphore MemPoolLock;
#elif !defined(__POWERUP__)
static struct SignalSemaphorePPC MemPoolLock;
#else
static void *MemPoolLock;
#endif

BOOL MemInit(VOID)
{
   if (!MemPool) {
      #ifndef __PPC__
      MemPool = LibCreatePool(MEMF_PUBLIC,8192,4096); /* 68K */
      InitSemaphore(&MemPoolLock);
      #elif !defined(__POWERUP__)
      MemPool = CreatePoolPPC(MEMF_PUBLIC,8192,4096); /* WarpOS */
      InitSemaphorePPC(&MemPoolLock);
      #else
      if ((MemPoolLock = PPCCreateSemaphore(NULL))) {
         MemPool = PPCCreatePool(MEMF_PUBLIC,8192,4096); /* PowerUp */
         if (!MemPool) PPCDeleteSemaphore(MemPoolLock);
      }
      #endif
   }

   return (MemPool != NULL);
}

void *malloc(size_t bytes)
{
   ULONG *mem = NULL;

   if (!MemPool) MemInit();

   if (MemPool) {
      #ifndef __PPC__
      bytes += 4;
      ObtainSemaphore(&MemPoolLock);
      if ((mem = (ULONG *)LibAllocPooled(MemPool,bytes))) { /* 68K */
         *mem = bytes;
         mem = &mem[1];
      }
      ReleaseSemaphore(&MemPoolLock);
      #elif !defined(__POWERUP__)
      bytes += 4;
      ObtainSemaphorePPC(&MemPoolLock);
      if ((mem = (ULONG *)AllocPooledPPC(MemPool,bytes))) { /* WarpOS */
         *mem = bytes;
         mem = &mem[1];
      }
      ReleaseSemaphorePPC(&MemPoolLock);
      #else
      PPCObtainSemaphore(MemPoolLock);
      mem = (ULONG *)PPCAllocVecPooled(MemPool,bytes);  /* PowerUp */
      PPCReleaseSemaphore(MemPoolLock);
      #endif

   }

   return(mem);
}

void *calloc(size_t objsize, size_t numobjs)
{
   void *mem;

   if ((mem = malloc(objsize*numobjs))) {
      memset(mem,0,objsize*numobjs);
   }

   return(mem);
}

void free(void *ptr)
{
   if (ptr) {
      #ifndef __PPC__
      ObtainSemaphore(&MemPoolLock);
      LibFreePooled(MemPool,&((ULONG *)ptr)[-1],((ULONG *)ptr)[-1]); /* 68K */
      ReleaseSemaphore(&MemPoolLock);
      #elif !defined(__POWERUP__)
      ObtainSemaphorePPC(&MemPoolLock);
      FreePooledPPC(MemPool,&((ULONG *)ptr)[-1],((ULONG *)ptr)[-1]); /* WarpOS */
      ReleaseSemaphorePPC(&MemPoolLock);
      #else
      PPCObtainSemaphore(MemPoolLock);
      PPCFreeVecPooled(MemPool,ptr); /* PowerUp */
      PPCReleaseSemaphore(MemPoolLock);
      #endif
   }
}

VOID MemDeinit(VOID)
{
   if (MemPool) {
      #ifndef __PPC__
      ObtainSemaphore(&MemPoolLock);
      LibDeletePool(MemPool); /* 68K */
      ReleaseSemaphore(&MemPoolLock);
      #elif !defined(__POWERUP__)
      ObtainSemaphorePPC(&MemPoolLock);
      DeletePoolPPC(MemPool); /* WarpOS */
      ReleaseSemaphorePPC(&MemPoolLock);
      #else
      PPCObtainSemaphore(MemPoolLock);
      PPCDeletePool(MemPool); /* PowerUp */
      PPCReleaseSemaphore(MemPoolLock);
      PPCDeleteSemaphore(MemPoolLock);
      MemPoolLock = NULL;
      #endif
      MemPool = NULL;
   }
}

/* Call MemDeinit at exit, after all other deinit code (which may use free()) */
#ifdef __PPC__
#define ADD2EXIT(a,pri) asm(".section .fini"); \
                        asm(" .long .text+(" #a "-.text); .long 0x4e")
#else
#define ADD2LIST(a,b,c) asm(".stabs \"_" #b "\"," #c ",0,0,_" #a )
#define ADD2EXIT(a,pri) ADD2LIST(a,__EXIT_LIST__,22); \
                        asm(".stabs \"___EXIT_LIST__\",20,0,0," #pri "+128")
#endif

ADD2EXIT(MemDeinit,-50);

#endif /* __POWERUP__ */
