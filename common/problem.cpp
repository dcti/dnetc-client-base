// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: problem.cpp,v $
// Revision 1.19  1998/06/15 00:12:24  skand
// fix id marker so it won't interfere when another .cpp file is #included here
//
// Revision 1.18  1998/06/14 10:13:43  skand
// use #if 0 (or 1) to turn on some debugging info, rather than // on each line
//
// Revision 1.17  1998/06/14 08:26:54  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.16  1998/06/14 08:13:04  friedbait
// 'Log' keywords added to maintain automatic change history
//
// Revision 1.15  1998/06/14 00:06:07  remi
// Added $Log.
//

static char *id_problem_cpp="@(#)$Id: problem.cpp,v 1.19 1998/06/15 00:12:24 skand Exp $";

#define NEW_STATS_AND_LOGMSG_STUFF

#include "problem.h"
#include "network.h"
#include "client.h"

#ifdef NEW_STATS_AND_LOGMSG_STUFF
#include "clitime.h" //for CliTimer() which gets a timeval of the current time
#endif

#ifndef _CPU_32BIT_
#error "everything assumes a 32bit CPU..."
#endif

#if (CLIENT_CPU == CPU_X86)
  u32 (*rc5_unit_func)( RC5UnitWork * rc5unitwork, u32 timeslice );
#elif (CLIENT_CPU == CPU_POWERPC) && (CLIENT_OS == OS_WIN32)
  // NT PPC doesn't have good assembly
  #include "rc5ansi2-rg.cpp"
#elif (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_VMS)
  #include "rc5ansi2-rg.cpp"
#elif (CLIENT_CPU == CPU_POWER)
  // power, not powerpc
  #include "rc5ansi1-rg.cpp"
#elif (CLIENT_CPU == CPU_POWERPC)
  extern "C" int crunch_allitnil( RC5UnitWork *work, unsigned long iterations );
  extern "C" int crunch_lintilla( RC5UnitWork *work, unsigned long iterations );
  int whichcrunch = 0;
#elif (CLIENT_CPU == CPU_68K)
  extern "C" int rc5_unit_func( RC5UnitWork *work );
#elif (CLIENT_CPU == CPU_ARM)
  u32 (*rc5_unit_func)( RC5UnitWork * rc5unitwork, unsigned long iterations  );
#elif ((CLIENT_OS == OS_SUNOS) && (CLIENT_CPU==CPU_68K))
  extern "C" int gettimeofday(struct timeval *tp, struct timezone *tzp);
#else
  #include "rc5stub.cpp"
#endif

#if (CLIENT_CPU == CPU_ARM)
 u32 (*des_unit_func)( RC5UnitWork * rc5unitwork, u32 timeslice );
#else
extern u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
#endif

#if ((CLIENT_CPU == CPU_X86) || (CLIENT_OS == OS_BEOS))
  extern u32 Bdes_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
#endif

#if (CLIENT_OS == OS_RISCOS)
extern "C" void riscos_upcall_6(void);
extern void CliSignalHandler(int);
#endif

Problem::Problem()
{
  initialized = 0;
  finished = 0;
  started = 0;
}

Problem::~Problem()
{
  // nothing to do.
}

s32 Problem::IsInitialized()
{
  if (initialized) {
    return 1;
  } else {
    return 0;
  }
}

s32 Problem::LoadState( ContestWork * work , u32 contesttype )
{
  contest = contesttype;
//printf("loadstate contest: %d %d \n",contesttype,contest);

#ifdef _CPU_32BIT_
  // copy over the state information
  contestwork.key.hi = ntohl( work->key.hi );
  contestwork.key.lo = ntohl( work->key.lo );
  contestwork.iv.hi = ntohl( work->iv.hi );
  contestwork.iv.lo = ntohl( work->iv.lo );
  contestwork.plain.hi = ntohl( work->plain.hi );
  contestwork.plain.lo = ntohl( work->plain.lo );
  contestwork.cypher.hi = ntohl( work->cypher.hi );
  contestwork.cypher.lo = ntohl( work->cypher.lo );
  contestwork.keysdone.hi = ntohl( work->keysdone.hi );
  contestwork.keysdone.lo = ntohl( work->keysdone.lo );
  contestwork.iterations.hi = ntohl( work->iterations.hi );
  contestwork.iterations.lo = ntohl( work->iterations.lo );
#if 0
  printf("key    hi/lo:  %08x:%08x\n", contestwork.key.hi, contestwork.key.lo);
  printf("iv     hi/lo:  %08x:%08x\n", contestwork.iv.hi, contestwork.iv.lo);
  printf("plain  hi/lo:  %08x:%08x\n",
	 contestwork.plain.hi, contestwork.plain.lo);
  printf("cipher hi/lo:  %08x:%08x\n",
	 contestwork.cypher.hi, contestwork.cypher.lo);
  printf("iter   hi/lo:  %08x:%08x\n",
	 contestwork.iterations.hi, contestwork.iterations.lo);
#endif

  // determine the starting key number
  // (note: doesn't account for carryover to hi or high end of keysdone)
  u64 key;
  key.hi = contestwork.key.hi;
  key.lo = contestwork.key.lo + contestwork.keysdone.lo;


  // set up the unitwork structure
  rc5unitwork.plain.hi = contestwork.plain.hi ^ contestwork.iv.hi;
  rc5unitwork.plain.lo = contestwork.plain.lo ^ contestwork.iv.lo;
  rc5unitwork.cypher.hi = contestwork.cypher.hi;
  rc5unitwork.cypher.lo = contestwork.cypher.lo;
  if (contesttype == 0)
  {
    rc5unitwork.L0.lo = ((key.hi >> 24) & 0x000000FFL) |
        ((key.hi >>  8) & 0x0000FF00L) |
        ((key.hi <<  8) & 0x00FF0000L) |
        ((key.hi << 24) & 0xFF000000L);
    rc5unitwork.L0.hi = ((key.lo >> 24) & 0x000000FFL) |
        ((key.lo >>  8) & 0x0000FF00L) |
        ((key.lo <<  8) & 0x00FF0000L) |
        ((key.lo << 24) & 0xFF000000L);
  } else {
    rc5unitwork.L0.lo = key.lo;
    rc5unitwork.L0.hi = key.hi;
  }

  // set up the current result state
  rc5result.key.hi = contestwork.key.hi;
  rc5result.key.lo = contestwork.key.lo;
  rc5result.keysdone.hi = contestwork.keysdone.hi;
  rc5result.keysdone.lo = contestwork.keysdone.lo;
  rc5result.iterations.hi = contestwork.iterations.hi;
  rc5result.iterations.lo = contestwork.iterations.lo;
  rc5result.result = RESULT_WORKING;

#endif
  startpercent = (u32) ( (double) 100000.0 *
     ( (double) (contestwork.keysdone.lo) /
       (double) (contestwork.iterations.lo) ) );
  percent=0;
  restart = ( work->keysdone.lo > 0 );

  initialized = 1;
  finished = 0;
  started = 0;

  return( 0 );
}

s32 Problem::RetrieveState( ContestWork * work , s32 setflags )
{
#ifdef _CPU_32BIT_
  // store back the state information
  work->key.hi = htonl( contestwork.key.hi );
  work->key.lo = htonl( contestwork.key.lo );
  work->iv.hi = htonl( contestwork.iv.hi );
  work->iv.lo = htonl( contestwork.iv.lo );
  work->plain.hi = htonl( contestwork.plain.hi );
  work->plain.lo = htonl( contestwork.plain.lo );
  work->cypher.hi = htonl( contestwork.cypher.hi );
  work->cypher.lo = htonl( contestwork.cypher.lo );
  work->keysdone.hi = htonl( contestwork.keysdone.hi );
  work->keysdone.lo = htonl( contestwork.keysdone.lo );
  work->iterations.hi = htonl( contestwork.iterations.hi );
  work->iterations.lo = htonl( contestwork.iterations.lo );
#endif
  if (setflags) {
    initialized = 0;
    finished = 0;
  }
//printf("retrievestate contest: %d \n",contest);
  return( contest );
}

s32 Problem::Run( u32 timeslice , u32 threadnum )
{
  struct timeval stop;

  if ( !initialized )
    return ( -1 );

  if ( finished )
    return ( 1 );

  if (!started)
  {
#ifdef NEW_STATS_AND_LOGMSG_STUFF
    CliTimer(&stop);
#else
    struct timezone dummy;
    gettimeofday( &stop, &dummy );
#endif
    timehi = stop.tv_sec;
    timelo = stop.tv_usec;
    started=1;
  }

  // don't allow a too large of a timeslice be used
  // (technically not necessary, but may save some wasted time)
  // note: doesn't account for high end or carry over
  if ( ( contestwork.keysdone.lo + ( timeslice * PIPELINE_COUNT ) ) > contestwork.iterations.lo )
    timeslice = ( contestwork.iterations.lo - contestwork.keysdone.lo +
                PIPELINE_COUNT - 1 ) / PIPELINE_COUNT + 1;

#if (CLIENT_CPU == CPU_POWERPC)
{
  unsigned long kiter = 0;
  if (contest == 0) {
#if ((CLIENT_OS != OS_BEOS) || (CLIENT_OS != OS_AMIGAOS))
    if (whichcrunch == 0)
      kiter = crunch_allitnil( &rc5unitwork, timeslice );
    else
#endif
      do {
        kiter += crunch_lintilla( &rc5unitwork, timeslice - kiter );
        if (kiter < timeslice) {
          if (crunch_allitnil( &rc5unitwork, 1 ) == 0) {
            break;
          }
          kiter++;
        }
      } while (kiter < timeslice);
  }
  else
  {
    // protect the innocent
    timeslice *= PIPELINE_COUNT;
    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits) / PIPELINE_COUNT;
#if (CLIENT_OS == OS_BEOS)
    if (threadnum == 0) {
      kiter = des_unit_func ( &rc5unitwork, nbits );
    } else {
      kiter = Bdes_unit_func ( &rc5unitwork, nbits );
    }
#else
    kiter = des_unit_func ( &rc5unitwork, nbits );
#endif
  }

  contestwork.keysdone.lo += kiter;

  if (kiter < timeslice)
  {
    // found it?
    rc5result.key.hi = contestwork.key.hi;
    rc5result.key.lo = contestwork.key.lo;
    rc5result.keysdone.hi = contestwork.keysdone.hi;
    rc5result.keysdone.lo = contestwork.keysdone.lo;
    rc5result.iterations.hi = contestwork.iterations.hi;
    rc5result.iterations.lo = contestwork.iterations.lo;
    rc5result.result = RESULT_FOUND;
    finished = 1;
    return( 1 );
  }
}
#elif (CLIENT_CPU == CPU_X86)
{
  unsigned long kiter;
  if (contest == 0)
  {
    kiter = rc5_unit_func ( &rc5unitwork, timeslice );
  }
  else
  {
    // protect the innocent
    timeslice *= PIPELINE_COUNT;
    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits) / PIPELINE_COUNT;
#ifdef MULTITHREAD
    if (threadnum == 0) {
      kiter = des_unit_func ( &rc5unitwork, nbits );
    } else {
      kiter = Bdes_unit_func ( &rc5unitwork, nbits );
    }
#else
    kiter = des_unit_func ( &rc5unitwork, nbits );
#endif
  }
  contestwork.keysdone.lo += kiter;
  if ( kiter < timeslice * PIPELINE_COUNT )
  {
    // found it?
    rc5result.key.hi = contestwork.key.hi;
    rc5result.key.lo = contestwork.key.lo;
    rc5result.keysdone.hi = contestwork.keysdone.hi;
    rc5result.keysdone.lo = contestwork.keysdone.lo;
    rc5result.iterations.hi = contestwork.iterations.hi;
    rc5result.iterations.lo = contestwork.iterations.lo;
    rc5result.result = RESULT_FOUND;
    finished = 1;
    return( 1 );
  }
  else if ( kiter != timeslice * PIPELINE_COUNT )
  {
#if !defined(NEEDVIRTUALMETHODS)
    printf("kiter wrong %ld %d\n", kiter, (int)(timeslice*PIPELINE_COUNT));
#endif
  }
}
#elif (CLIENT_CPU == CPU_SPARC) && (ULTRA_CRUNCH == 1)
{
  unsigned long kiter;
  if (contest == 0) {
    kiter = crunch( &rc5unitwork, timeslice );
  } else {
    // protect the innocent
    timeslice *= PIPELINE_COUNT;
    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits) / PIPELINE_COUNT;
    kiter = des_unit_func ( &rc5unitwork, nbits );
  }
  contestwork.keysdone.lo += kiter;
  if (kiter < ( timeslice * PIPELINE_COUNT ) )
  {
    // found it?
    rc5result.key.hi = contestwork.key.hi;
    rc5result.key.lo = contestwork.key.lo;
    rc5result.keysdone.hi = contestwork.keysdone.hi;
    rc5result.keysdone.lo = contestwork.keysdone.lo;
    rc5result.iterations.hi = contestwork.iterations.hi;
    rc5result.iterations.lo = contestwork.iterations.lo;
    rc5result.result = RESULT_FOUND;
    finished = 1;
    return( 1 );
  }
  else if (kiter != ( timeslice * PIPELINE_COUNT ) )
  {
#if !defined(NEEDVIRTUALMETHODS)
    printf("kiter wrong %ld %d\n", (long) kiter, (int) (timeslice*PIPELINE_COUNT));
#endif
  }
}
#elif ((CLIENT_CPU == CPU_MIPS) && (MIPS_CRUNCH == 1))
{
  unsigned long kiter;
  if (contest == 0) {
    kiter = crunch( &rc5unitwork, timeslice );
  } else {
    // protect the innocent
    timeslice *= PIPELINE_COUNT;
    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits) / PIPELINE_COUNT;
    kiter = des_unit_func ( &rc5unitwork, nbits );
  }
  contestwork.keysdone.lo += kiter;
  if (kiter < ( timeslice * PIPELINE_COUNT ) )
  {
    // found it?
    rc5result.key.hi = contestwork.key.hi;
    rc5result.key.lo = contestwork.key.lo;
    rc5result.keysdone.hi = contestwork.keysdone.hi;
    rc5result.keysdone.lo = contestwork.keysdone.lo;
    rc5result.iterations.hi = contestwork.iterations.hi;
    rc5result.iterations.lo = contestwork.iterations.lo;
    rc5result.result = RESULT_FOUND;
    finished = 1;
    return( 1 );
  }
  else if (kiter != (timeslice * PIPELINE_COUNT))
  {
#if !defined(NEEDVIRTUALMETHODS)
    printf("kiter wrong %ld %d\n", kiter, timeslice*PIPELINE_COUNT);
#endif
  }
}
#elif (CLIENT_CPU == CPU_ARM)
{
  unsigned long kiter;
#if (CLIENT_OS == OS_RISCOS)
  if (_kernel_escape_seen())
  {
      CliSignalHandler(SIGINT);
  }
  if (riscos_in_taskwindow)
  {
      riscos_upcall_6();
  }
#endif

  if (contest == 0)
  {
    /*
        returns timeslice - keys processed
    */
    kiter = rc5_unit_func(&rc5unitwork, timeslice);
    contestwork.keysdone.lo += timeslice;
    if (kiter)
    {
      // found it?
      contestwork.keysdone.lo -= kiter;

      rc5result.key.hi = contestwork.key.hi;
      rc5result.key.lo = contestwork.key.lo;
      rc5result.keysdone.hi = contestwork.keysdone.hi;
      rc5result.keysdone.lo = contestwork.keysdone.lo;
      rc5result.iterations.hi = contestwork.iterations.hi;
      rc5result.iterations.lo = contestwork.iterations.lo;
      rc5result.result = RESULT_FOUND;
      finished = 1;
      return( 1 );
    }
  }
  else
  {
    // protect the innocent
    timeslice *= PIPELINE_COUNT;
    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits) / PIPELINE_COUNT;
    kiter = des_unit_func ( &rc5unitwork, nbits );
    contestwork.keysdone.lo += kiter;
    if (kiter < timeslice)
    {
      // found it?
      rc5result.key.hi = contestwork.key.hi;
      rc5result.key.lo = contestwork.key.lo;
      rc5result.keysdone.hi = contestwork.keysdone.hi;
      rc5result.keysdone.lo = contestwork.keysdone.lo;
      rc5result.iterations.hi = contestwork.iterations.hi;
      rc5result.iterations.lo = contestwork.iterations.lo;
      rc5result.result = RESULT_FOUND;
      finished = 1;
      return( 1 );

    }
  }
}
#else
  unsigned long kiter = 0;
  if (contest == 0) {
    while ( timeslice-- ) // timeslice ignores the number of pipelines
    {
      u32 result = rc5_unit_func( &rc5unitwork );
      if ( result )
      {
      // found it?
        rc5result.key.hi = contestwork.key.hi;
        rc5result.key.lo = contestwork.key.lo;
        rc5result.keysdone.hi = contestwork.keysdone.hi;
        rc5result.keysdone.lo = contestwork.keysdone.lo + result - 1;
        rc5result.iterations.hi = contestwork.iterations.hi;
        rc5result.iterations.lo = contestwork.iterations.lo;
        rc5result.result = RESULT_FOUND;
        finished = 1;
        return( 1 );
      }
      else
      {
        // "mangle-increment" the key number by the number of pipelines
        rc5unitwork.L0.hi = (rc5unitwork.L0.hi + (PIPELINE_COUNT << 24)) & 0xFFFFFFFF;
        if (!(rc5unitwork.L0.hi & 0xFF000000)) {

          rc5unitwork.L0.hi = (rc5unitwork.L0.hi + 0x00010000) & 0x00FFFFFF;
          if (!(rc5unitwork.L0.hi & 0x00FF0000)) {

             rc5unitwork.L0.hi = (rc5unitwork.L0.hi + 0x00000100) & 0x0000FFFF;
             if (!(rc5unitwork.L0.hi & 0x0000FF00)) {

                rc5unitwork.L0.hi = (rc5unitwork.L0.hi + 0x00000001) & 0x000000FF;
                if (!(rc5unitwork.L0.hi & 0x000000FF)) {

                  rc5unitwork.L0.hi = 0x00000000;
                  rc5unitwork.L0.lo = rc5unitwork.L0.lo + 0x01000000;
                  if (!(rc5unitwork.L0.lo & 0xFF000000)) {

                    rc5unitwork.L0.lo = (rc5unitwork.L0.lo + 0x00010000) & 0x00FFFFFF;
                    if (!(rc5unitwork.L0.lo & 0x00FF0000)) {

                      rc5unitwork.L0.lo = (rc5unitwork.L0.lo + 0x00000100) & 0x0000FFFF;
                      if (!(rc5unitwork.L0.lo & 0x0000FF00)) {

                        rc5unitwork.L0.lo = (rc5unitwork.L0.lo + 0x00000001) & 0x000000FF;
                      }
                    }
                  }
               }
            }
          }
        }
        // increment the count of keys done
        // note: doesn't account for carry
        contestwork.keysdone.lo += PIPELINE_COUNT;
      }
    }
  }
  else
  {
    timeslice *= PIPELINE_COUNT;
    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits) / PIPELINE_COUNT;
    kiter = des_unit_func ( &rc5unitwork, nbits );
    contestwork.keysdone.lo += kiter;
    if (kiter < ( timeslice * PIPELINE_COUNT ) )
    {
      // found it?
      rc5result.key.hi = contestwork.key.hi;
      rc5result.key.lo = contestwork.key.lo;
      rc5result.keysdone.hi = contestwork.keysdone.hi;
      rc5result.keysdone.lo = contestwork.keysdone.lo;
      rc5result.iterations.hi = contestwork.iterations.hi;
      rc5result.iterations.lo = contestwork.iterations.lo;
      rc5result.result = RESULT_FOUND;
      finished = 1;
      return( 1 );
    }
    else if (kiter != (timeslice * PIPELINE_COUNT))
    {
#if !defined(NEEDVIRTUALMETHODS)
        printf("kiter wrong %ld %ld\n",
               (long) kiter, (long)(timeslice*PIPELINE_COUNT));
#endif
    }
  }
#endif

  if ( ( contestwork.keysdone.hi > contestwork.iterations.hi ) ||
       ( ( contestwork.keysdone.hi == contestwork.iterations.hi ) &&
       ( contestwork.keysdone.lo >= contestwork.iterations.lo ) ) )
  {
    // done with this block and nothing found
    rc5result.key.hi = contestwork.key.hi;
    rc5result.key.lo = contestwork.key.lo;
    rc5result.keysdone.hi = contestwork.keysdone.hi;
    rc5result.keysdone.lo = contestwork.keysdone.lo;
    rc5result.iterations.hi = contestwork.iterations.hi;
    rc5result.iterations.lo = contestwork.iterations.lo;
    rc5result.result = RESULT_NOTHING;
    finished = 1;
    return( 1 );
  }
  else
  {
    // more to do, come back later.
    rc5result.key.hi = contestwork.key.hi;
    rc5result.key.lo = contestwork.key.lo;
    rc5result.keysdone.hi = contestwork.keysdone.hi;
    rc5result.keysdone.lo = contestwork.keysdone.lo;
    rc5result.iterations.hi = contestwork.iterations.hi;
    rc5result.iterations.lo = contestwork.iterations.lo;
    rc5result.result = RESULT_WORKING;
    finished = 0;
    return( 0 );
  }
}


s32 Problem::GetResult( RC5Result * result )
{
  if ( !initialized )
    return ( -1 );

  // note that all but result go back to network byte order at this point.
  result->key.hi = htonl( rc5result.key.hi );
  result->key.lo = htonl( rc5result.key.lo );
  result->keysdone.hi = htonl( rc5result.keysdone.hi );
  result->keysdone.lo = htonl( rc5result.keysdone.lo );
  result->iterations.hi = htonl( rc5result.iterations.hi );
  result->iterations.lo = htonl( rc5result.iterations.lo );
  result->result = rc5result.result;

  return ( contest );
}

u32 Problem::CalcPercent()
{
  return (u32) ( (double) 100.0 *
                  ( ((double) rc5result.keysdone.lo) /
                    ((double) rc5result.iterations.lo) ) );
}


