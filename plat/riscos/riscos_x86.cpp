/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * x86 PCCARD support including a crunch wrapper/controller
 * $Id: riscos_x86.cpp,v 1.1.2.1 2001/01/21 15:10:28 cyp Exp $
*/
#ifdef __cplusplus
extern "C" {
#endif
#include <string.h> /* memcpy */
#include <stdlib.h> /* atexit */
#include <swis.h>   /* _swix */
#ifdef __cplusplus
}
#endif

#include "cputypes.h" /* u8, u32 */
#include "problem.h"  /* RC5UnitWork */
#include "riscos_x86.h" /* ourselves */

#define RC5PC 0x523C0
#define RC5PC_Version       RC5PC+0
#define RC5PC_Title         RC5PC+1
#define RC5PC_Status        RC5PC+2 /* does it exist? */
#define RC5PC_On            RC5PC+3 /* turn card on */
#define RC5PC_Off           RC5PC+4 /* turn card off */
#define RC5PC_WhatPCCard    RC5PC+5 /* identify by name */
#define RC5PC_BufferStatus  RC5PC+6 /* ??? */
#define RC5PC_AddBlock      RC5PC+7 /* load a block */
#define RC5PC_RetriveBlock  RC5PC+8 /* get (changed?) previously loaded */

struct RC5PCstruct
{
  u8 internal_status;
  u8 result;
  u8 zero;
  u8 zero1;
  u32 timeslice;
  struct {u32 hi,lo;} key;
  struct {u32 hi,lo;} iv;
  struct {u32 hi,lo;} plain;
  struct {u32 hi,lo;} cypher;
  struct {u32 hi,lo;} keysdone;
  struct {u32 hi,lo;} iterations;
  struct {u32 hi,lo;} zero2;
};

/* ----------------------------------------------------------------- */

const char *riscos_x86_ident(void) /* returns NULL if no x86 found, */ 
{                                  /* else name or "" if no name */
  static const char *ident = (const char *)0;
  #if defined(HAVE_X86_CARD_SUPPORT)
  static int init = -1;

  if (init < 0)
  {
    unsigned int s;
    _kernel_oserror *err = _swix( RC5PC_Status, _OUT(0), &s );

    init = 0; ident = (const char *)0;
    if (!err && (s & 1) != 0)
    {
      char *mfr, *cpu;
      ident = (const char *)"";
      err = _swix(RC5PC_WhatPCCard,_OUTR(2,3),&mfr,&cpu);
      if (!err)
      {
        static char buf[100];
        unsigned int pos = 0;  
        while (*mfr && pos < (sizeof(buf)-1))
          buf[pos++] = *mfr++;
        if (pos < (sizeof(buf)-2) && *cpu)
        {
          if (pos)
            buf[pos++] = ' ';
          while (*cpu && pos < (sizeof(buf)-1))
            buf[pos++] = *cpu++;
        }      
        buf[pos] = '\0';
        ident = (const char *)&buf[0];
      }  
    }    
  }
  #endif /* HAVE_X86_CARD_SUPPORT */
  return ident;
}  

/* ----------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif
s32 rc5_unit_func_x86( RC5UnitWork *work, u32 *iterations, void * );
#ifdef __cplusplus
}
#endif

static void rc5_unit_func_x86_stop(void) /* called from atexit() */
{
  rc5_unit_func_x86( ((RC5UnitWork *)0), ((u32 *)0), ((void *)0) );
} 

s32 rc5_unit_func_x86( RC5UnitWork *work, u32 *iterations, void *memblk )
{
  #if defined(HAVE_X86_CARD_SUPPORT)
  static int isrunning = -2;
  static RC5UnitWork last;

  RC5PCstruct rc5pc;
  _kernel_oserror *err;
  _kernel_swi_regs r;

  if (!work || !iterations) /* request to stop+unload */
  {
    if (isrunning >= 0)
    {
      r.r[0] = 0;
      if (isrunning > 0) /* started and loaded */
      {
        _kernel_swi(RC5PC_RetriveBlock,&r,&r); /* unload to /dev/null */
        isrunning = 0;
      }  
      if (isrunning == 0) /* started not loaded */
      {  
        _kernel_swi(RC5PC_Off,&r,&r);          /* turn it off */
        isrunning = -1; /* not started */
      }  
    }  
    return -1; /* no good result code */     
  }  

  if (isrunning < 0) /* not started */
  {
    if (isrunning == -2 && !riscos_x86_ident()) 
      return -1;  /* no x86 card */
    if (_kernel_swi(RC5PC_On,&r,&r))
      return -1; /* failed */
    if (isrunning == -2)  
      atexit(rc5_unit_func_x86_stop);
    isrunning = 0; /* started, but not loaded */
  }

  if (isrunning > 0
      && (last.L0.hi!=work->L0.hi || last.L0.lo!=work->L0.lo
      ||  last.iv.hi!=work->iv.hi || last.iv.lo!=work->iv.lo
      ||  last.plain.hi!=work->plain.hi || last.plain.lo!=work->plain.lo
      ||  last.cypher.hi!=work->cypher.hi || last.cypher.lo!=work->cypher.lo))
  {                                                /* different block */ 
    r.r[0] = 0;
    _kernel_swi(RC5PC_RetriveBlock,&r,&r); /* unload to /dev/null */
    isrunning = 0; /* started, but not loaded */
  }

  if (isrunning == 0) /* started, but not loaded */
  {
    rc5pc.key.hi = work->key.hi;
    rc5pc.key.lo = work->key.lo;
    rc5pc.iv.hi = work->iv.hi;
    rc5pc.iv.lo = work->iv.lo;
    rc5pc.plain.hi = work->plain.hi;
    rc5pc.plain.lo = work->plain.lo;
    rc5pc.cypher.hi = work->cypher.hi;
    rc5pc.cypher.lo = work->cypher.lo;
    rc5pc.keysdone.hi = work->keysdone.hi;
    rc5pc.keysdone.lo = work->keysdone.lo;
    rc5pc.iterations.hi = work->iterations.hi;
    rc5pc.iterations.lo = work->iterations.lo;
    rc5pc.timeslice = work->iterations.lo;
    if (rc5pc.timeslice == 0)
      rc5pc.timeslice = (1<<20);

    r.r[1] = (int)&rc5pc;
    err = _kernel_swi(RC5PC_AddBlock,&r,&r);
    if ((err) || (r.r[0] == -1))
      return -1; /* error */

    isrunning = 1;  /* started and loaded */
    memcpy( &last, work, sizeof(RC5UnitWork)); /* update last known state */
    *iterations = 0; /* haven't done any iterations yet */
    return RESULT_WORKING;
  }

  if (isrunning > 0) /* started and loaded */
  {
    u32 itersdone_lo, itersdone_hi;
    r.r[1] = (int)&rc5pc;
    err = _kernel_swi(RC5PC_RetriveBlock,&r,&r);
    if ((err) || (r.r[0] == -1))
      return -1; /* error */

    work->key.hi = rc5pc.key.hi;
    work->key.lo = rc5pc.key.lo;
    work->iv.hi = rc5pc.iv.hi;
    work->iv.lo = rc5pc.iv.lo;
    work->plain.hi = rc5pc.plain.hi;
    work->plain.lo = rc5pc.plain.lo;
    work->cypher.hi = rc5pc.cypher.hi;
    work->cypher.lo = rc5pc.cypher.lo;
    work->keysdone.hi = rc5pc.keysdone.hi;
    work->keysdone.lo = rc5pc.keysdone.lo;
    work->iterations.hi = rc5pc.iterations.hi;
    work->iterations.lo = rc5pc.iterations.lo;

    itersdone_lo = (work->keysdone.lo - last.keysdone.lo);
    itersdone_hi = (work->keysdone.hi - last.keysdone.hi);
    if (itersdone_lo < work->keysdone.lo || itersdone_lo < last.keysdone.lo)
      itersdone_hi++; /* should never happen - can't deal with it anyway */
    *iterations = itersdone_lo; /* # of iterations we've done in the meantime*/
    
    memcpy( &last, work, sizeof(RC5UnitWork)); /* update last known state */

    if ((work->keysdone.hi > work->iterations.hi) ||
                    (work->keysdone.hi == work->iterations.hi &&
                     work->keysdone.lo >= work->iterations.lo))
      return RESULT_NOTHING; /* block completed, nothing found */
    /* 
       ok, if we got here, then the cruncher has either
       found a key (RESULT_FOUND) or is still going (RESULT_WORKING).
       Two problems: 
       a) how do we tell RESULT_FOUND?
          for a normal cruncher (synchronous call), result_found is true
          when the number of iterations done is less that the number of 
          iterations we asked it to do. But we can't do that here because
          the cruncher is still working in the 'background'.
       b) can the cruncher "run out" of iterations_to_do?
          if so, we need to unload it, reset the rc5pc.timeslice
          and reload it. (But perhaps there is a better way)
    */      
    #error FIXME
  }
  #endif /* HAVE_X86_CARD_SUPPORT */

  work = work;             /* possibly unused */
  iterations = iterations; /* possibly unused */
  memblk = memblk;         /* unused */
  return -1;               /* error */
}

/* ----------------------------------------------------------------- */
