/* Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * Copyright distributed.net 1997-2001 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * -----------------------------------------------------------------
 * NOTE: this file (the problem loader/saver) knows nothing about
 *       individual contests. It is problem.cpp's job to provide 
 *       all the information it needs to load/save a problem.
 *       KEEP IT THAT WAY!
 * -----------------------------------------------------------------
*/
const char *probfill_cpp(void) {
return "@(#)$Id: probfill.cpp,v 1.58.2.69 2001/04/06 00:44:12 sampo Exp $"; }

//#define TRACE

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "version.h"   // CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "client.h"    // CONTEST_COUNT, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "problem.h"   // Problem class
#include "logstuff.h"  // Log()/LogScreen()
#include "clitime.h"   // CliGetTimeString()
#include "probman.h"   // GetProblemPointerFromIndex()
#include "clicdata.h"  // CliGetContestNameFromID,CliGetContestWorkUnitSpeed
#include "cpucheck.h"  // GetNumberOfDetectedProcessors() [for thresh fxns]
#include "checkpt.h"   // CHECKPOINT_CLOSE define
#include "triggers.h"  // RaiseExitRequestTrigger()
#include "buffupd.h"   // BUFFERUPDATE_FETCH/_FLUSH define
#include "buffbase.h"  // GetBufferCount,Get|PutBufferRecord,etc
#include "modereq.h"   // ModeReqSet() and MODEREQ_[FETCH|FLUSH]
#include "clievent.h"  // ClientEventSyncPost
#include "util.h"      // trace
#include "probfill.h"  // ourselves.

// =======================================================================
// each individual problem load+save generates 4 or more messages lines 
// (+>=3 lines for every load+save cycle), so we suppress/combine individual 
// load/save messages if the 'load_problem_count' exceeds COMBINEMSG_THRESHOLD
// into a single line 'Loaded|Saved n RC5|DES packets from|to filename'.
#define COMBINEMSG_THRESHOLD 8 // anything above this and we don't show 
                               // individual load/save messages
// =======================================================================   

int SetProblemLoaderFlags( const char *loaderflags_map )
{
  unsigned int prob_i = 0;
  Problem *thisprob;
  while ((thisprob = GetProblemPointerFromIndex(prob_i)) != 0)
  {
    if (ProblemIsInitialized(thisprob))
      thisprob->pub_data.loaderflags |= loaderflags_map[thisprob->pub_data.contest];
    prob_i++;
  }
  return ((prob_i == 0)?(-1):((int)prob_i));
}  

/* ----------------------------------------------------------------------- */

static void __get_num_c_and_p(Client *client, int contestid,
                              unsigned int *numcrunchersP, 
                              unsigned int *numprocsP) 
{
  int numcrunchers = ProblemCountLoaded(contestid);
  int numprocs = client->numcpu;
  if (numprocs < 0) /* auto-detect */
    numprocs = GetNumberOfDetectedProcessors();
  if (numprocs < 1) /* forced single cpu (0) or failed auto-detect (-1) */
    numprocs = 1;  
  if (!numcrunchers)
    numcrunchers = numprocs;
  if (numcrunchersP)
    *numcrunchersP = numcrunchers;
  if (numprocsP)
    *numprocsP = numprocs;
}


static unsigned int __get_thresh_secs(Client *client, int contestid,
                                      int force, unsigned int stats_units,
                                      unsigned int *numcrunchersP)
{
  int was_forced; unsigned int sec;

  /* somebody keeps freaking playing with client->numcpu dependancy */
  /* is it so difficult to understand that some users explictely specify */
  /* a number of crunchers not equal to the number of cpus in the machine? */
  /* (not necessarily crunchers less than cpus either) */

  // get the speed - careful!: if CliGetContestWorkUnitSpeed
  // uses benchmark (was_forced) to get the speed, then the speed is 
  // per-cpu, not per-cruncher. Otherwise, its per-cruncher.
  sec = CliGetContestWorkUnitSpeed(contestid, force, &was_forced);

  if (sec != 0) /* we have a rate */
  {
    unsigned int divx, numcrunchers, numprocs;
    __get_num_c_and_p( client, contestid, &numcrunchers, &numprocs );

    if (numcrunchersP)
      *numcrunchersP = numcrunchers;

    divx = 0;
    if (was_forced) /* work unit speed is per-cpu */
      divx = numprocs;
    else            /* work unit speed is per-cruncher */
      divx = numcrunchers;
    
    if (stats_units) /* get projected time to completion */
      sec = (stats_units * sec) / (divx * 100); /* statsunits=workunits*100 */
    else             /* get number of stats units per day */
      sec = (100 * 24 * 3600 * divx) / sec; 
  }
  return sec;
}

/* ----------------------------------------------------------------------- */

/* Buffer counts obtained from ProbfillGetBufferInfo() are for 
** informational use (by frontends etc) only. Don't shortcut 
** any of the common code calls to GetBufferCount()
*/
static struct { 
  struct { 
      long blk;
      long swu; 
  } counts[2];
  long threshold; 
  unsigned int till_completion; 
} buffer_counts[CONTEST_COUNT] = {
  { { { 0, 0 }, { 0, 0 } }, 0, 0 },
  { { { 0, 0 }, { 0, 0 } }, 0, 0 },
  { { { 0, 0 }, { 0, 0 } }, 0, 0 },
  { { { 0, 0 }, { 0, 0 } }, 0, 0 }
  #if (CONTEST_COUNT != 4)
    #error static initializer expects CONTEST_COUNT == 4
  #endif
};

int ProbfillGetBufferCounts( unsigned int contest, int is_out_type,
                             long *threshold, int *thresh_in_swu,
                             long *blk_count, long *swu_count, 
                             unsigned int *till_completion )
{
  int rc = -1;
  if (contest < CONTEST_COUNT)
  {
    if (threshold) 
      *threshold = buffer_counts[contest].threshold;
    if (thresh_in_swu)
      *thresh_in_swu = (contest != OGR);
    if (till_completion)
      *till_completion = buffer_counts[contest].till_completion;
    if (is_out_type)
      is_out_type = 1;
    if (blk_count)
      *blk_count = buffer_counts[contest].counts[is_out_type].blk;
    if (swu_count)
      *swu_count = buffer_counts[contest].counts[is_out_type].swu;
    rc = 0;
  }
  return rc;
}

/* --------------------------------------------------------------------- */

/* called by GetBufferCount() [buffbase.cpp] whenever both
** swu_count and blk_count were determined.
** In-buffer:
**     When fetching and when a block is loaded, and
**     if 'frequent-check'ing is enabled, then every frequent check cycle
** Out-buffer:
**     When flushing and when a block is completed.
*/
int ProbfillCacheBufferCounts( Client *client,
                               unsigned int cont_i, int is_out_type,
                               long blk_count, long swu_count)
{
  if (cont_i < CONTEST_COUNT && blk_count >= 0 && swu_count >= 0)
  {
    if (is_out_type)
      is_out_type = 1;
    buffer_counts[cont_i].counts[is_out_type].blk = blk_count;
    buffer_counts[cont_i].counts[is_out_type].swu = swu_count;
    if (!is_out_type && client)
    {
      buffer_counts[cont_i].till_completion = 
             __get_thresh_secs(client, cont_i, 0, swu_count, 0 );
    }
  }
  return 0;
}

/* --------------------------------------------------------------------- */

unsigned int ClientGetInThreshold(Client *client, 
                                  int contestid, int force )
{
  // If inthreshold is == 0, then use time threshold.
  // If inthreshold is == 0, AND time is 0, then use BUFTHRESHOLD_DEFAULT
  // If inthreshold > 0 AND time > 0, then use MAX(inthreshold, effective_workunits(time))
  unsigned int numcrunchers = 0; /* unknown */
  unsigned int bufthresh = 0; /* unknown */

  // OGR time threshold NYI
  client->timethreshold[OGR] = 0;

  if (contestid >= CONTEST_COUNT)
  {
    return 1000; // XXX return 1000 when in an invalid state.  this should be
                 // changed or documented.
  }
  if (client->inthreshold[contestid] > 0)
  {
    bufthresh = client->inthreshold[contestid] * 100;
  }
  if (client->timethreshold[contestid] > 0) /* use time */
  {
    unsigned int secs, timethresh = 0;
    secs = __get_thresh_secs(client, contestid, force, 0, &numcrunchers );

    if (secs) /* number of stats units per 24 hour period */
      timethresh = 1 + (client->timethreshold[contestid] * secs / 24);
    if (timethresh > bufthresh)
      bufthresh = timethresh;
  }
  if (numcrunchers < 1) /* undetermined */
  {
    __get_num_c_and_p( client, contestid, &numcrunchers, 0 );
  }
  if (bufthresh < 1) /* undetermined */
  {
    #define BUFTHRESHOLD_DEFAULT_PER_CRUNCHER (24*100)  /* in stats units */
    bufthresh = BUFTHRESHOLD_DEFAULT_PER_CRUNCHER * numcrunchers;
    #undef BUFTHRESHOLD_DEFAULT_PER_CRUNCHER
  }
  if (bufthresh < (numcrunchers * 100)) /* ensure at least 1.00 stats */
  {                                     /* units per per cruncher */
    bufthresh = numcrunchers * 100;
  }
  buffer_counts[contestid].threshold = bufthresh;  
  return bufthresh;
}

/* 
   How thresholds affect contest rotation and contest fallover:

   For contest rotation, outthreshold checks (and connectoften) must be 
   disabled, otherwise the client will update before it hits the end of 
   the load_order, resulting in more work becoming available for all 
   projects.
   Inversely, for contest fallover, outthreshold checks (or connectoften) 
   must be enabled.

   Example scenarios:
   1) User wants to run both OGR and RC5, allocating 3 times as much cpu
      time to OGR than to RC5. These would be the required settings:
      load_order=OGR,RC5      (the order in which the client LOOKs for work)
      inthresholds=OGR=6,RC5=2    (OGR thresh is 3 times RC5 thresh)
      outthresholds=OGR=0,RC5=0 (disable outthresh checking)  
      what happens: 
         client looks for work. OGR is available. does OGR.
         (repeat OGR inthresh times)
         client looks for work, no OGR is available, RC5 is. does RC5.
         (repeat RC5 inthresh times)
         client looks for work, no OGR, no RC5 available. 
                fetches&flushes. 
         client looks for work. OGR is available. does OGR.
   2) User wants to run OGR as long as OGR is available, the do RC5 (until 
      OGR is available again).
      load_order=OGR,RC5
      inthresholds=OGR=<something>,RC5=<something>
      outthresholds=OGR=<not zero and less than inthresh>,RC5=<whatever>
      what happens: 
         client looks for work. OGR is available. does OGR.
         (repeat OGR outhresh times) 
         out threshold now crossed. flushes&fetches.
         client looks for work. OGR is available. does OGR.
   3) User wants to run ONLY contest XXX.
      load_order=XXX,<all others>=0
      if contest XXX is NOT RC5, and no work is available, the client
      will exit. if contest XXX is RC5, and no work is available, it will
      do randoms.
*/

static unsigned int ClientGetOutThreshold(Client *client, 
                                   int contestid, int /* force */)
{
  int outthresh = 0;  /* returns zero if outthresholds are not to be checked */
  client = client; /* shaddup compiler. */

  if (contestid < CONTEST_COUNT)
  {
#if (!defined(NO_OUTBUFFER_THRESHOLDS))
    outthresh = client->outthreshold[contestid]; /* never time driven */
    if (outthresh != 0) /* outthresh=0 => outthresh=inthresh => return 0 */
    {
      unsigned int inthres = ClientGetInThreshold(client, contestid, 0);
      if (inthresh > 0) /* no error */
      {
        if (outthresh <= 0) /* relative to inthresh */
        {
          /*
          a) if the outthreshold (as per .ini) is <=0, then outthreshold is 
          to be interpreted as a value relative to the (computed) inthreshold.
          ie, computed_outthreshold = computed_intthreshold + ini_outthreshold.
          [thus an ini_threshold equal to zero implies rule c) below]
          */
          outthresh = inthresh + outthresh;
        }
        if (outthresh >= inthresh)
        {
          /*
          b) if the outthreshold (according to the .ini) was > inthresh
          then inthreshold rules are effective because outthresh can never
          be greater than inthresh (inthresh will have been checked first).
          (The exception is when using shared buffers, and another client 
          fetches but does not flush).
          Consequence: Only inthreshold is effective and outthresh 
          doesn't need to be checked. ClientGetOutThreshold() returns 0.
          c) if the outthreshold (according to the .ini) was equal to inthresh
          there there is usually no point checking outthresh because 
          the result of both checks would be the same. (The exception is 
          when using shared buffers, and another client fetches but does
          not flush).
          Consequence: inthreshold is effective and outthresh 
          doesn't need to be checked. ClientGetOutThreshold() returns 0.
          */
          outthresh = 0;
        }
      }
    }
#endif
  }
  return 100 * ((unsigned int)outthresh);      
}


/* determine if out buffer threshold has been crossed, and if so, set 
   the flush_required flag
*/
static int __check_outbufthresh_limit( Client *client, unsigned int cont_i, 
                                     long packet_count, unsigned long wu_count, 
                                     int *bufupd_pending )
{
  if ((*bufupd_pending & BUFFERUPDATE_FLUSH) == 0)
  {
    unsigned int thresh = ClientGetOutThreshold( client, cont_i, 0 );
    /* ClientGetOutThreshold() returns 0 if thresh doesn't need checking */
    if (thresh > 0) /* threshold _does_ need to be checked. */
    {               
      if (packet_count < 0) /* not determined or error */
      {
        packet_count = GetBufferCount( client, cont_i, 1, &wu_count );
      }
      if (packet_count > 0) /* wu_count is valid */
      {
        if ((unsigned long)(wu_count) >= ((unsigned long)thresh))
        {
          *bufupd_pending |= BUFFERUPDATE_FLUSH;
        }
      }
    }     
  }
  return ((*bufupd_pending & BUFFERUPDATE_FLUSH) != 0);
}

/* ----------------------------------------------------------------------- */

static unsigned int __IndividualProblemSave( Problem *thisprob, 
                unsigned int prob_i, Client *client, int *is_empty, 
                unsigned load_problem_count, unsigned int *contest,
                int *bufupd_pending, int unconditional_unload,
                int abortive_action )
{                    
  unsigned int did_save = 0;
  prob_i = prob_i; /* shaddup compiler. we need this */

  *contest = 0;
  *is_empty = 1; /* assume not initialized */
  if ( ProblemIsInitialized(thisprob) )  
  {
    WorkRecord wrdata;
    int resultcode;
    unsigned int cont_i;
    memset( (void *)&wrdata, 0, sizeof(WorkRecord));
    resultcode = ProblemRetrieveState( thisprob, &wrdata.work, &cont_i, 0, 0 );

    *is_empty = 0; /* assume problem is in use */

    if (resultcode == RESULT_FOUND || resultcode == RESULT_NOTHING ||
       unconditional_unload || resultcode < 0 /* core error */ ||
      (thisprob->pub_data.loaderflags & (PROBLDR_DISCARD|PROBLDR_FORCEUNLOAD)) != 0) 
    { 
      int finito = (resultcode==RESULT_FOUND || resultcode==RESULT_NOTHING);
      const char *action_msg = 0;
      const char *reason_msg = 0;
      int discarded = 0;
      char dcountbuf[64]; /* we use this as scratch space too */
      struct timeval tv;
      ProblemInfo info;

      *contest = cont_i;
      *is_empty = 1; /* will soon be */

      wrdata.contest = (u8)(cont_i);
      wrdata.resultcode = resultcode;
      wrdata.contest    = (u8)cont_i;
      wrdata.resultcode = resultcode;
      wrdata.cpu        = FILEENTRY_CPU(thisprob->pub_data.client_cpu,
                                        thisprob->pub_data.coresel);
      wrdata.os         = FILEENTRY_OS;      //CLIENT_OS
      wrdata.buildhi    = FILEENTRY_BUILDHI; //(CLIENT_BUILDFRAC >> 8)
      wrdata.buildlo    = FILEENTRY_BUILDLO; //CLIENT_BUILDFRAC & 0xff

      if (finito)
      {
        wrdata.os      = CLIENT_OS;
        #if (CLIENT_OS == OS_RISCOS)
        if (prob_i == 1)
          wrdata.cpu   = CPU_X86;
        else
        #endif
        wrdata.cpu     = CLIENT_CPU;
        wrdata.buildhi = CLIENT_CONTEST;
        wrdata.buildlo = CLIENT_BUILD;
        strncpy( wrdata.id, client->id , sizeof(wrdata.id));
        wrdata.id[sizeof(wrdata.id)-1]=0;
        ClientEventSyncPost( CLIEVENT_PROBLEM_FINISHED, &prob_i, sizeof(prob_i) );
      }

      if ((thisprob->pub_data.loaderflags & PROBLDR_DISCARD)!=0)
      {
        action_msg = "Discarded";
        reason_msg = "project disabled/closed";
        discarded = 1;
      }
      else if (resultcode < 0)
      {
        action_msg = "Discarded";
        reason_msg = "core error";
        discarded = 1;
      }
      else if (PutBufferRecord( client, &wrdata ) < 0)
      {
        action_msg = "Discarded";
        reason_msg = "buffer error - unable to save";
        discarded = 1;
      }
      else
      {
        did_save = 1;
        if (client->nodiskbuffers)
          *bufupd_pending |= BUFFERUPDATE_FLUSH;
        if (__check_outbufthresh_limit( client, cont_i, -1, 0,bufupd_pending))
        { /* adjust bufupd_pending if outthresh has been crossed */
          //Log("1. *bufupd_pending |= BUFFERUPDATE_FLUSH;\n");
        }       

        if (load_problem_count > COMBINEMSG_THRESHOLD)
          ; /* nothing */
        else if (thisprob->pub_data.was_truncated)
        {
          action_msg = "Skipped";
          discarded  = 1;
          reason_msg = thisprob->pub_data.was_truncated;
        }
        else if (!finito)
          action_msg = "Saved";
        else
          action_msg = "Completed";
      }
      info.permille_only_if_exact = 1;
      
      if (ProblemGetInfo( thisprob, &info, P_INFO_E_TIME   | P_INFO_SWUCOUNT | P_INFO_C_PERMIL |
                                           P_INFO_SIGBUF   | P_INFO_RATEBUF  | P_INFO_CCOUNT   |
                                           P_INFO_TCOUNT   | P_INFO_DCOUNT) != -1)
      {
        tv.tv_sec = info.elapsed_secs; tv.tv_usec = info.elapsed_usecs;

        if (finito && !discarded && !info.is_test_packet)
          CliAddContestInfoSummaryData(cont_i,info.ccounthi,info.ccountlo,&tv,info.swucount);

        if (action_msg)
        {
          if (reason_msg) /* was discarded */
          {
            //[....] Discarded CSC 12345678:ABCDEF00 4*2^28
            //       (project disabled/closed)
            Log("%s: %s %s%c(%s)\n", CliGetContestNameFromID(cont_i), action_msg,
                                     info.sigbuf, ((strlen(reason_msg)>10)?('\n'):(' ')), reason_msg );
          }
          else
          {
            U64stringify(dcountbuf, sizeof(dcountbuf), info.dcounthi, info.dcountlo, 2, CliGetContestUnitFromID(cont_i));
            if (finito && info.is_test_packet) /* finished test packet */
              strcat( strcpy( dcountbuf,"Test: RESULT_"),
                     ((resultcode==RESULT_NOTHING)?("NOTHING"):("FOUND")) );
            else if (finito) /* finished non-test packet */ 
            {
              char *p = strrchr(info.sigbuf,':'); /* HACK! to supress too long */
              if (p) *p = '\0';            /* crypto "Completed" lines */
              sprintf( dcountbuf, "%u.%02u stats units", info.swucount/100, info.swucount%100);
            }
            else if (info.c_permille > 0)
              sprintf( dcountbuf, "%u.%u0%% done", info.c_permille/10, info.c_permille%10);
            else
              strcat( dcountbuf, " done" );

            //[....] RC5: Saved 12345678:ABCDEF00 4*2^28 (5.20% done)
            //       1.23:45:67:89 - [987,654,321 keys/s]
            //[....] OGR: Saved 25/1-6-13-8-16-18 (12.34 Mnodes done)
            //       1.23:45:67:89 - [987,654,321 nodes/s]
            //[....] RC5: Completed 68E0D85A:A0000000 4*2^28 (4.00 stats units)
            //       1.23:45:67:89 - [987,654,321 keys/s]
            //[....] OGR: Completed 22/1-3-5-7 (12.30 stats units)
            //       1.23:45:67:89 - [987,654,321 nodes/s]
            Log("%s: %s %s (%s)\n%s - [%s/s]\n", 
              CliGetContestNameFromID(cont_i), action_msg, info.sigbuf, dcountbuf,
              CliGetTimeString( &tv, 2 ), info.ratebuf );
            if (finito && info.show_exact_iterations_done)
            {
              Log("%s: %s [%s]\n", CliGetContestNameFromID(cont_i), info.sigbuf,
              ProblemComputeRate(cont_i, 0, 0, info.tcounthi, info.tcountlo, 0, 0, dcountbuf, sizeof(dcountbuf)));
            }
          } /* if (reason_msg) else */
        } /* if (action_msg) */
      } /* if (thisprob->GetProblemInfo( ... ) != -1) */
    
      /* we can purge the object now */
      /* we don't wait when aborting. thread might be hung */
      ProblemRetrieveState( thisprob, NULL, NULL, 1, abortive_action /*==dontwait*/ );

    } /* unload needed */
  } /* is initialized */

  return did_save;
}

/* ----------------------------------------------------------------------- */

//     Internal function that loads 'wrdata' with a new workrecord
//     from the next open contest with available blocks.
// Return value:
//     if (return_single_count) is non-zero, returns number of packets
//     left for the same project work was found for, otherwise it
//     returns the *total* number of packets available for *all*
//     contests for the thread in question.
//
// Note that 'return_single_count' IS ALL IT TAKES TO DISABLE ROTATION.
static long __loadapacket( Client *client, 
                           WorkRecord *wrdata /*where to load*/, 
                           int /*ign_closed*/,  
                           unsigned int prob_i /* for which 'thread' */, 
                           int return_single_count /* see above */ )
{                    
  unsigned int cont_i; 
  long bufcount, totalcount = -1;

  for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++ )
  {
    unsigned int selproject = (unsigned int)client->loadorder_map[cont_i];
    if (selproject >= CONTEST_COUNT) /* user disabled */
      continue;
    if (!IsProblemLoadPermitted( (long)prob_i, selproject ))
    {
      continue; /* problem.cpp - result depends on #defs, threadsafety etc */
    }
    bufcount = -1;
    if (!wrdata) /* already loaded a packet */
      bufcount = GetBufferCount( client, selproject, 0, NULL );
    else         /* haven't got a packet yet */
    {
      bufcount = GetBufferRecord( client, wrdata, selproject, 0 );
      if (bufcount >= 0) /* no error */
        wrdata = 0;     /* don't load again */
    }
    if (bufcount >= 0) /* no error */
    {  
      if (totalcount < 0)
        totalcount = 0;
      totalcount += bufcount;
      if (return_single_count)
        break;
    }
  }
  return totalcount;
}  

/* ---------------------------------------------------------------------- */

#define NOLOAD_NONEWBLOCKS       -3
#define NOLOAD_ALLCONTESTSCLOSED -2
#define NOLOAD_NORANDOM          -1

/* ---------------------------------------------------------------------- */

static unsigned int __IndividualProblemLoad( Problem *thisprob, 
                    unsigned int prob_i, Client *client, int *load_needed, 
                    unsigned load_problem_count, 
                    unsigned int *loaded_for_contest,
                    int *bufupd_pending )
{
  unsigned int did_load = 0;
  int retry_due_to_failed_loadstate = 0;

  do /* while (retry_due_to_failed_loadstate) */
  {
    WorkRecord wrdata;
    int update_on_current_contest_exhaust_flag = (client->connectoften & 4);
    long bufcount;

    retry_due_to_failed_loadstate = 0;
    bufcount = __loadapacket( client, &wrdata, 1, prob_i, 
                              update_on_current_contest_exhaust_flag );

    if (bufcount < 0 && client->nonewblocks == 0)
    {
      //Log("3. BufferUpdate(client,(BUFFERUPDATE_FETCH|BUFFERUPDATE_FLUSH),0)\n");
      int didupdate = 
         BufferUpdate(client,(BUFFERUPDATE_FETCH|BUFFERUPDATE_FLUSH),0);
      if (!(didupdate < 0))
      {
        if (didupdate!=0)
          *bufupd_pending&=~(didupdate&(BUFFERUPDATE_FLUSH|BUFFERUPDATE_FETCH));
        if ((didupdate & BUFFERUPDATE_FETCH) != 0) /* fetched successfully */
          bufcount = __loadapacket( client, &wrdata, 0, prob_i,
                                    update_on_current_contest_exhaust_flag );
      }
    }
  
    *load_needed = 0;
    if (bufcount >= 0) /* load from file suceeded */
      *load_needed = 0;
    else if (client->rc564closed || client->blockcount < 0)
      *load_needed = NOLOAD_NORANDOM; /* -1 */
    else if (client->nonewblocks)
      *load_needed = NOLOAD_NONEWBLOCKS;
    else  /* using randoms is permitted */
      *load_needed = 0;

    if (*load_needed == 0)
    {
      u32 timeslice = 0x10000;
      int expected_cpu = 0, expected_core = 0;
      int expected_os  = 0, expected_build = 0;
      const ContestWork *work = &wrdata.work;
        
      #if (defined(INIT_TIMESLICE) && (INIT_TIMESLICE >= 64))
      timeslice = INIT_TIMESLICE;
      #endif

      if (bufcount < 0) /* normal load from buffer failed */
      {                 /* so generate random */
        work = CONTESTWORK_MAGIC_RANDOM;       
        *loaded_for_contest = 0; /* don't care. just to initialize */
      }
      else
      {
        *loaded_for_contest = (unsigned int)(wrdata.contest);
        expected_cpu = FILEENTRY_CPU_TO_CPUNUM( wrdata.cpu );
        expected_core = FILEENTRY_CPU_TO_CORENUM( wrdata.cpu );
        expected_os  = FILEENTRY_OS_TO_OS( wrdata.os );
        expected_build = FILEENTRY_BUILD_TO_BUILD(wrdata.buildhi,wrdata.buildlo);
        work = &wrdata.work;
        
        /* if the total number of packets in buffers is less than the number 
           of crunchers running then post a fetch request. This means that the
           effective minimum threshold is always >= num crunchers
        */   
        if (bufcount <= 1 /* only one packet left */ ||
           ((unsigned long)(bufcount)) < (load_problem_count - prob_i))
        {
          *bufupd_pending |= BUFFERUPDATE_FETCH;
        } 
      }

      /* loadstate can fail if it selcore fails or the previous problem */
      /* hadn't been purged, or the contest isn't available or ... */

      if (ProblemLoadState( thisprob, work, *loaded_for_contest, timeslice, 
           expected_cpu, expected_core, expected_os, expected_build ) == -1)
      {
        /* The problem with LoadState() failing is that it implicitely
        ** causes the block to be discarded, which means, that the 
        ** keyserver network will reissue it - a senseless undertaking
        ** if the data itself is invalid.
        */
        retry_due_to_failed_loadstate = 1;
      }
      else
      {
        *load_needed = 0;
        did_load = 1; 

        ClientEventSyncPost( CLIEVENT_PROBLEM_STARTED, &prob_i, sizeof(prob_i) );
    
        if (load_problem_count <= COMBINEMSG_THRESHOLD)
        {
          char ddonebuf[15];
          ProblemInfo info;
          info.permille_only_if_exact = 1;
          if (ProblemGetInfo( thisprob, &info, P_INFO_S_PERMIL | P_INFO_SIGBUF | P_INFO_DCOUNT) != -1)
          {
            const char *extramsg = ""; 
            char perdone[32]; 
  
            if (thisprob->pub_data.was_reset)
              extramsg="\nPacket was from a different core/client cpu/os/build.";
            else if (info.s_permille > 0 && info.s_permille < 1000)
            {
              sprintf(perdone, " (%u.%u0%% done)", (info.s_permille/10), (info.s_permille%10));
              extramsg = perdone;
            }
            else if (info.dcounthi || info.dcountlo)
            {
              strcat( strcat( strcpy(perdone, " ("), U64stringify(ddonebuf, sizeof(ddonebuf), 
                                                                  info.dcounthi, info.dcountlo, 2,
                                                                  CliGetContestUnitFromID(thisprob->pub_data.contest))),
                                                     " done)"); 
              extramsg = perdone;
            }
            
            Log("%s: Loaded %s%s%s\n",
                 CliGetContestNameFromID(thisprob->pub_data.contest),
                 ((thisprob->pub_data.is_random)?("random "):("")), info.sigbuf, extramsg );
          } /* if (thisprob->GetProblemInfo(...) != -1) */
        } /* if (load_problem_count <= COMBINEMSG_THRESHOLD) */
      } /* if (LoadState(...) != -1) */
    } /* if (*load_needed == 0) */
  } while (retry_due_to_failed_loadstate);

  return did_load;
}    

// --------------------------------------------------------------------

static int __post_summary_for_contest(unsigned int contestid)
{
  u32 iterhi, iterlo; int rc = -1;
  unsigned int packets, swucount;
  struct timeval ttime;

  TRACE_OUT((+1,"__post_summary_for_contest(%u)\n",contestid));

  if (CliGetContestInfoSummaryData( contestid, &packets, &iterhi, &iterlo,
                                    &ttime, &swucount ) == 0)
  {
    if (packets)
    {
      char ratebuf[15];
      TRACE_OUT((0,"pkts=%u, iter=%u:%u, time=%u:%u, swucount=%u\n", packets, 
                    iterhi, iterlo, ttime.tv_sec, ttime.tv_usec, swucount ));
      Log("%s: Summary: %u packet%s (%u.%02u stats units)\n%s%c- [%s/s]\n",
          CliGetContestNameFromID(contestid), 
          packets, ((packets==1)?(""):("s")), 
          swucount/100, swucount%100, 
          CliGetTimeString(&ttime,2), ((packets)?(' '):(0)), 
          ProblemComputeRate( contestid, ttime.tv_sec, ttime.tv_usec, 
                            iterhi, iterlo, 0, 0, ratebuf, sizeof(ratebuf)) );
    }                            
    rc = 0;
  }

  TRACE_OUT((-1,"__post_summary_for_contest()\n"));
  return rc;
}

// --------------------------------------------------------------------

unsigned int LoadSaveProblems(Client *client,
                              unsigned int load_problem_count,int mode)
{
  /* Some platforms need to stop asynchronously, for example, Win16 which
     gets an ENDSESSION message and has to exit then and there. So also 
     win9x when running as a service where windows suspends all threads 
     except the window thread. For these (and perhaps other platforms)
     we save our last state so calling with (0,0,0) will save the
     problem states (without hanging). see 'abortive_action' below.
  */
  static Client *previous_client = 0;
  static unsigned int previous_load_problem_count = 0, reentrant_count = 0;
  static int abortive_action = 0;

  unsigned int retval = 0;
  int changed_flag, first_time;

  int allclosed, prob_step,bufupd_pending;  
  unsigned int cont_i, prob_for, prob_first, prob_last;
  unsigned int loaded_problems_count[CONTEST_COUNT];
  unsigned int saved_problems_count[CONTEST_COUNT];
  unsigned long totalBlocksDone; /* all contests */
  
  unsigned int total_problems_loaded, total_problems_saved;
  unsigned int norandom_count, getbuff_errs, empty_problems;

  allclosed = 0;
  norandom_count = getbuff_errs = empty_problems = 0;
  changed_flag = (previous_load_problem_count == 0);
  total_problems_loaded = 0;
  total_problems_saved = 0;
  bufupd_pending = 0;
  totalBlocksDone = 0;

  /* ============================================================= */

  if (abortive_action) /* already aborted once */
  {                    /* no probfill action can happen again */
    return 0;          
  }
  if (!client)             /* abnormal end */
  {
    client = previous_client;
    if (!client)
      return 0;
    abortive_action = 1;
    mode = PROBFILL_UNLOADALL;
  }
  previous_client = client;
  if ((++reentrant_count) > 1)
  {
    --reentrant_count;
    return 0;
  }

  /* ============================================================= */

  prob_first = 0;
  prob_step  = 0;
  prob_last  = 0;
  first_time = 0;

  if (load_problem_count == 0) /* only permitted if unloading all */
  {
    if (mode != PROBFILL_UNLOADALL || previous_load_problem_count == 0)
    {
      --reentrant_count;
      return 0;
    }
    load_problem_count = previous_load_problem_count;
  }
  if (previous_load_problem_count == 0) /* must be initial load */
  {            /* [0 ... (load_problem_count - 1)] */
    prob_first = 0;
    prob_last  = (load_problem_count - 1);
    prob_step  = 1; 
    first_time = 1;
  }
  else if (mode == PROBFILL_RESIZETABLE)
  {            /* [(previousload_problem_count-1) ... load_problem_count] */
    prob_first = load_problem_count;
    prob_last  = (previous_load_problem_count - 1);
    prob_step  = -1;  
  }
  else /* PROBFILL_UNLOADALL, PROBFILL_REFRESH */
  {            /* [(load_problem_count - 1) ... 0] */
    prob_first = 0;
    prob_last  = (load_problem_count - 1);
    prob_step  = -1;
  }  

  /* ============================================================= */

  for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
  {
    unsigned int blocksdone;
    if (CliGetContestInfoSummaryData( cont_i, &blocksdone, NULL, NULL, NULL, NULL )==0)
      totalBlocksDone += blocksdone;
    loaded_problems_count[cont_i] = 0;
    saved_problems_count[cont_i] = 0;
  }

  /* ============================================================= */

  ClientEventSyncPost(CLIEVENT_PROBLEM_TFILLSTARTED, &load_problem_count, 
                                                sizeof(load_problem_count));

  for (prob_for = 0; prob_for <= (prob_last - prob_first); prob_for++)
  {
    Problem *thisprob;
    int load_needed;
    unsigned int prob_i = prob_for + prob_first;
    if ( prob_step < 0 )
      prob_i = prob_last - prob_for;
      
    thisprob = GetProblemPointerFromIndex( prob_i );
    if (thisprob == 0)
    {
      if (prob_step < 0)
        continue;
      break;
    }

    // -----------------------------------

    load_needed = 0;
    if (__IndividualProblemSave( thisprob, prob_i, client, 
        &load_needed, load_problem_count, &cont_i, &bufupd_pending,
        (mode == PROBFILL_UNLOADALL || mode == PROBFILL_RESIZETABLE ),
        abortive_action ))
    {
      changed_flag = 1;
      total_problems_saved++;
      saved_problems_count[cont_i]++;
      totalBlocksDone++;
    }
    if (load_needed)
      empty_problems++;

    //---------------------------------------

    if (load_needed && mode!=PROBFILL_UNLOADALL && mode!=PROBFILL_RESIZETABLE)
    {
      if (client->blockcount>0 && 
          totalBlocksDone>=((unsigned long)(client->blockcount)))
      {
        ; //nothing
      }
      else
      {
        load_needed = 0;
        if (__IndividualProblemLoad( thisprob, prob_i, client, 
            &load_needed, load_problem_count, &cont_i, &bufupd_pending ))
        {
          empty_problems--;
          total_problems_loaded++;
          loaded_problems_count[cont_i]++;
          changed_flag = 1;
        }
        if (load_needed)
        {
          getbuff_errs++;
          if (load_needed == NOLOAD_ALLCONTESTSCLOSED)
          {
            allclosed = 1;
            break; /* the for ... prob_i ... loop */
          }
          else if (load_needed == NOLOAD_NORANDOM)
            norandom_count++;
        }
      }
    } //if (load_needed)
  } //for (prob_i = 0; prob_i < load_problem_count; prob_i++ )

  ClientEventSyncPost(CLIEVENT_PROBLEM_TFILLFINISHED,
     ((previous_load_problem_count==0)?(&total_problems_loaded):(&total_problems_saved)),
     sizeof(total_problems_loaded));

  /* ============================================================= */

  if (mode == PROBFILL_UNLOADALL)
  {
    previous_load_problem_count = 0;
    if (client->nodiskbuffers == 0)
    {
      // close checkpoint file immediately after saving the problems to disk
      CheckpointAction( client, CHECKPOINT_CLOSE, 0 );
    }
    else /* no disk buffers */
    {
      BufferUpdate(client,BUFFERUPDATE_FLUSH,0);
      /* in case the flush fails, empty the membuf table manually */
      for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
      {
        WorkRecord data;
        while (GetBufferCount(client, cont_i, 0, 0)>0)
          GetBufferRecord( client, &data, cont_i, 0 );
        while (GetBufferCount(client, cont_i, 0, 0)>0)
          GetBufferRecord( client, &data, cont_i, 1);
      }
    }
    retval = total_problems_saved;
  }
  else /* if (mode != PROBFILL_UNLOADALL) */
  {
    /* 
    =============================================================
    // save the number of active problems, so that we can bail out
    // in an "emergency". Some platforms call us asynchronously when they
    // need to abort [win16/win32 for example]
    -------------------------------------------------------------
    */

    previous_load_problem_count = load_problem_count;

    if (bufupd_pending && client->blockcount >= 0)
    {
      int req = MODEREQ_FLUSH; // always flush while fetching
      if (!CheckExitRequestTriggerNoIO()) //((bufupd_pending & BUFFERUPDATE_FETCH)!=0)
        req |= MODEREQ_FETCH;
      ModeReqSet( req|MODEREQ_FQUIET ); /* delegate to the client.run loop */
    }

    if (!allclosed && mode != PROBFILL_RESIZETABLE)
    {
      /* 
       =============================================================
       if we are running a limited number of blocks then check if we have
       exceeded that number. If we have, but one or more crunchers are
       still at work, bump the limit. 
       ------------------------------------------------------------- 
      */
      int limitsexceeded = 0;
      if (client->blockcount < 0 && norandom_count >= load_problem_count)
        limitsexceeded = 1;
      if (client->blockcount > 0 && 
         (totalBlocksDone >= (unsigned long)(client->blockcount)))
      {
        if (empty_problems >= load_problem_count)
          limitsexceeded = 1;
        else
          client->blockcount = ((u32)(totalBlocksDone))+1;
      }
      if (limitsexceeded)
      {
        Log( "Shutdown - packet limit exceeded.\n" );
        RaiseExitRequestTrigger();
      }  
    }

    if (mode == PROBFILL_RESIZETABLE)
      retval = total_problems_saved;
    else if (mode == PROBFILL_GETBUFFERRS) 
      retval = getbuff_errs;
    else if (mode == PROBFILL_ANYCHANGED)
      retval = changed_flag;
    else  
      retval = total_problems_loaded;
  }  

  /* ============================================================= */

  for ( cont_i = 0; cont_i < CONTEST_COUNT; cont_i++) //once for each contest
  {
    int show_totals = 0;
    if (loaded_problems_count[cont_i] || saved_problems_count[cont_i])
      show_totals = 1;

    if (first_time || show_totals)
    {
      unsigned int inout;
      const char *cont_name = CliGetContestNameFromID(cont_i);

      if (loaded_problems_count[cont_i] && load_problem_count > COMBINEMSG_THRESHOLD )
      {
        Log( "%s: Loaded %u packet%s from %s\n", 
              cont_name, loaded_problems_count[cont_i],
              ((loaded_problems_count[cont_i]==1)?(""):("s")),
              (client->nodiskbuffers ? "(memory-in)" : 
              BufferGetDefaultFilename( cont_i, 0, 
                                        client->in_buffer_basename )) );
      }

      if (saved_problems_count[cont_i] && load_problem_count > COMBINEMSG_THRESHOLD
       && (client->nodiskbuffers == 0 || (mode != PROBFILL_UNLOADALL)))
      {
        Log( "%s: Saved %u packet%s to %s\n", 
              cont_name, saved_problems_count[cont_i],
              ((saved_problems_count[cont_i]==1)?(""):("s")),
              (mode == PROBFILL_UNLOADALL)?
                (client->nodiskbuffers ? "(memory-in)" : 
                BufferGetDefaultFilename( cont_i, 0, 
                                          client->in_buffer_basename ) ) :
                (client->nodiskbuffers ? "(memory-out)" : 
                BufferGetDefaultFilename( cont_i, 1, 
                                          client->out_buffer_basename )) );
      }

      if (show_totals && totalBlocksDone > 0)
      {
        // To suppress "odd" problem completion count summaries (and not be
        // quite so verbose) we only display summaries if the number of
        // completed problems is even divisible by the number of processors.
        // Requires a working GetNumberOfDetectedProcessors() [cpucheck.cpp]
        #if 0
        int cpustmp; unsigned int cpus = 1;
        if ((cpustmp = GetNumberOfDetectedProcessors()) > 1)
          cpus = (unsigned int)cpustmp;
        if (load_problem_count > cpus)
          cpus = load_problem_count;
        if ((totalBlocksDone%cpus) == 0 )
        #endif
        {
          __post_summary_for_contest(cont_i);
        }
      }

      /* -------------------------------------------------------------- */

      for (inout=0;inout<=1;inout++)
      {
        unsigned long stats_count;
        long block_count = GetBufferCount( client, cont_i, inout, &stats_count );

        if (show_totals && block_count >= 0) /* no error */ 
        {
          char buffer[(3*80)+sizeof(client->in_buffer_basename)];
          int len;

          len = sprintf(buffer, "%s: %ld packet%s ", 
                cont_name, block_count, ((block_count == 1)?(""):("s")) );
          if (stats_count)
            len += sprintf( &buffer[len], "(%lu.%02lu stats units) ",
                            stats_count/100,stats_count%100);
          len += sprintf( &buffer[len], "%s in\n%s",
              ((inout!= 0 || mode == PROBFILL_UNLOADALL)?
                 ((block_count==1)?("is"):("are")):
                 ((block_count==1)?("remains"):("remain"))),
              ((inout== 0)?
                  (client->nodiskbuffers ? "(memory-in)" : 
                   BufferGetDefaultFilename( cont_i, 0, 
                   client->in_buffer_basename ) ) :
                   (client->nodiskbuffers ? "(memory-out)": 
                   BufferGetDefaultFilename( cont_i, 1, 
                   client->out_buffer_basename ) ))
             );
          if (len < 55) /* fits on a single line, so unwrap */
          {
            char *nl = strrchr( buffer, '\n' );
            if (nl) *nl = ' ';
          }               
          if (inout != 0) /* out-buffer */ 
          {
            /* adjust bufupd_pending if outthresh has been crossed */
            /* we don't check in-buffer here since we need cumulative count */
            if (__check_outbufthresh_limit( client, cont_i, block_count, 
                                            stats_count, &bufupd_pending ))
            {
              //Log("5. bufupd_pending |= BUFFERUPDATE_FLUSH;\n");
            }
          }
          else /*in*/ if (stats_count && (mode!=PROBFILL_UNLOADALL))
          {
            timeval tv;
            tv.tv_sec = __get_thresh_secs(client, cont_i, 0, stats_count, 0 );
            if (tv.tv_sec > 0)          
            {
              tv.tv_usec = 0;
              len += sprintf(&buffer[len],
                       "\nProjected ideal time to completion: %s", 
                       CliGetTimeString( &tv, 2));
            }
          }
          Log( "%s\n", buffer );
        } //if (block_count >= 0)

      } //  for (inout=0;inout<=1;inout++)
    } //if (loaded_problems_count[cont_i] || saved_problems_count[cont_i])
  } //for ( cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)

  /* ============================================================ */

  if (mode == PROBFILL_UNLOADALL)
  {
    previous_load_problem_count = 0;
    previous_client = (Client *)0;
    if (!CheckRestartRequestTrigger())
      Log("Shutdown complete.\n");
  }
  --reentrant_count;

  return retval;
}  
  
