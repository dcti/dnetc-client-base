/* Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

//#define STRESS_RANDOMGEN
//#define STRESS_RANDOMGEN_ALL_KEYSPACE

const char *probfill_cpp(void) {
return "@(#)$Id: probfill.cpp,v 1.58.2.6 1999/10/07 18:38:58 cyp Exp $"; }

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "version.h"   // CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "client.h"    // CONTEST_COUNT, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "problem.h"   // Problem class
#include "logstuff.h"  // Log()/LogScreen()
#include "clitime.h"   // CliGetTimeString()
#include "cpucheck.h"  // GetNumberOfDetectedProcessors()
#include "util.h"      // temporary home for ogr_stubstr()
#include "random.h"    // Random()
#include "clisrate.h"  // CliGetMessageFor... et al.
#include "clicdata.h"  // CliGetContestNameFromID()
#include "clirate.h"   // CliGetKeyrateForProblem()
#include "probman.h"   // GetProblemPointerFromIndex()
#include "checkpt.h"   // CHECKPOINT_CLOSE define
#include "triggers.h"  // RaiseExitRequestTrigger()
#include "buffupd.h"   // BUFFERUPDATE_FETCH/_FLUSH define
#include "modereq.h"   // ModeReqSet() and MODEREQ_[FETCH|FLUSH]
#include "probfill.h"  // ourselves.
#include "rsadata.h"   // Get cipher/etc for random blocks
#include "confrwv.h"   // Needed to trigger .ini to be updated
#include "clievent.h"  // ClientEventSyncPost( int event_id, long parm )

// =======================================================================
// each individual problem load+save generates 4 or more messages lines 
// (+>=3 lines for every load+save cycle), so we suppress/combine individual 
// load/save messages if the 'load_problem_count' exceeds COMBINEMSG_THRESHOLD
// into a single line 'Loaded|Saved n RC5|DES packets from|to filename'.
#define COMBINEMSG_THRESHOLD 4 // anything above this and we don't show 
                               // individual load/save messages
// =======================================================================   

#define __iter2norm( iterlo, iterhi ) ((iterlo>>28)+(iterhi<<4))

/* ----------------------------------------------------------------------- */

static const char *__WrapOrTruncateLogLine( char *buffer, int dowrap )
{
  char *stop, *start = buffer;
  unsigned int maxlen = 55;
  int dotruncate = (dowrap == 0);

  if (dowrap)
  {
    while ( strlen(buffer) > maxlen )
    {
      stop = buffer+maxlen;
      while (stop > buffer && *stop!='\n' && *stop!=' ' && *stop!='\t')
        stop--;
      if (stop == buffer && *stop != '\n')
      {
        dotruncate = 1;
        break;
      }
      *stop = '\n';
      buffer = ++stop;
    }
  }
  if (dotruncate)
  {
    if ( strlen(buffer) > maxlen )
    {
      buffer[maxlen-4]=0;
      strcat(buffer, " ...");
    }
  }
  return start;
}  

/* ----------------------------------------------------------------------- */

int SetProblemLoaderFlags( const char *loaderflags_map )
{
  unsigned int prob_i = 0;
  Problem *thisprob;
  while ((thisprob = GetProblemPointerFromIndex(prob_i)) != NULL)
  {
    if (thisprob->IsInitialized())
      thisprob->loaderflags |= loaderflags_map[thisprob->contest];
    prob_i++;
  }
  return ((prob_i == 0)?(-1):((int)prob_i));
}  

/* ----------------------------------------------------------------------- */

static unsigned int __IndividualProblemSave( Problem *thisprob, 
                unsigned int prob_i, Client *client, int *is_empty, 
                unsigned load_problem_count, unsigned int *contest,
                int *bufupd_pending, int unconditional_unload )
{                    
  unsigned int norm_key_count = 0;
  *contest = 0;
  *is_empty = 0;

  if ( thisprob->IsInitialized()==0 )  
  {
    *is_empty = 1; 
    prob_i = prob_i; //get rid of warning
  }
  else 
  {
    WorkRecord wrdata;
    int resultcode;
    unsigned int cont_i;
    s32 cputype = CLIENT_CPU; /* needed for FILEENTRY_CPU macro */
    memset( (void *)&wrdata, 0, sizeof(WorkRecord));
    resultcode = thisprob->RetrieveState( &wrdata.work, &cont_i, 0 );
    #if (CLIENT_OS == OS_RISCOS)
    if (prob_i == 1) cputype = CPU_X86; 
    #endif

    if (resultcode == RESULT_FOUND || resultcode == RESULT_NOTHING )
    {
      long longcount;
      *contest = cont_i;
      *is_empty = 1; /* will soon be */

      if (client->keyport == 3064)
      {
        LogScreen("Test success was %sdetected!\n",
           (wrdata.resultcode == RESULT_NOTHING ? "not" : "") );
      }

      wrdata.contest = (u8)(cont_i);
      wrdata.resultcode = resultcode;
      wrdata.os      = CLIENT_OS;
      wrdata.cpu     = (u8)cputype;
      wrdata.buildhi = CLIENT_CONTEST;
      wrdata.buildlo = CLIENT_BUILD;
      strncpy( wrdata.id, client->id , sizeof(wrdata.id));
      wrdata.id[sizeof(wrdata.id)-1]=0;

      switch (cont_i) 
      {
        case RC5:
        case DES:
        case CSC:
        {
          norm_key_count = 
             (unsigned int)__iter2norm( (wrdata.work.crypto.iterations.lo),
                                      (wrdata.work.crypto.iterations.hi) );
          if (norm_key_count == 0) /* test block */
            norm_key_count = 1;
          break;
        }
        case OGR:
        {
          norm_key_count = 1;
          break;
        }
      }
      
      // send it back... error messages is printed by PutBufferRecord
      if ( (longcount = client->PutBufferRecord( &wrdata )) >= 0)
      {
        //---------------------
        // update the totals for this contest
        //---------------------
        if ((unsigned long)(longcount) >= (unsigned long)(client->outthreshold[*contest]))
          *bufupd_pending |= BUFFERUPDATE_FLUSH;

        if (load_problem_count <= COMBINEMSG_THRESHOLD)
        {
          Log( CliGetMessageForProblemCompleted( thisprob ) );
        }
        else /* stop the log file from being cluttered with load/save msgs */
        {
          CliGetKeyrateForProblem( thisprob ); //add to totals
        }
      }
      ClientEventSyncPost( CLIEVENT_PROBLEM_FINISHED, (long)prob_i );
    }
    else if (unconditional_unload || resultcode < 0 /* core error */ ||
      (thisprob->loaderflags & (PROBLDR_DISCARD|PROBLDR_FORCEUNLOAD)) != 0) 
    {                           
      unsigned int permille = 0;
      const char *msg = NULL;

      *contest = cont_i;
      *is_empty = 1; /* will soon be */

      cputype           = client->cputype; /* uh, "coretype" */
      wrdata.contest    = (u8)cont_i;
      wrdata.resultcode = resultcode;
      wrdata.cpu        = FILEENTRY_CPU; /* combines CLIENT_CPU and coretype */
      wrdata.os         = FILEENTRY_OS;
      wrdata.buildhi    = FILEENTRY_BUILDHI; 
      wrdata.buildlo    = FILEENTRY_BUILDLO;

      if ((thisprob->loaderflags & PROBLDR_DISCARD)!=0)
      {
        msg = "Discarded (project disabled/closed)";
        norm_key_count = 0;
      }
      else if (resultcode < 0)
      {
        msg = "Discarded (core error)";
        norm_key_count = 0;
      }
      else if (client->PutBufferRecord( &wrdata ) < 0)  // send it back...
      {
        msg = "Unable to save";
        norm_key_count = 0;
      }
      else
      {
        switch (cont_i) 
        {
          case RC5:
          case DES:
          case CSC:
                  norm_key_count = (unsigned int)__iter2norm( 
                                      (wrdata.work.crypto.iterations.lo),
                                      (wrdata.work.crypto.iterations.hi) );
                  if (norm_key_count == 0) /* test block */
                    norm_key_count = 1;
                  break;
          case OGR:
                  norm_key_count = 1;
                  break;
        }
        permille = (unsigned int)thisprob->CalcPermille();
        if (client->nodiskbuffers)
          *bufupd_pending |= BUFFERUPDATE_FLUSH;
        if (load_problem_count <= COMBINEMSG_THRESHOLD)
          msg = "Saved";
      }
      if (msg)
      {
        char workunit[80];
        switch (cont_i)
        {
          case RC5:
          case DES:
          case CSC:
                 sprintf(workunit, "%08lX:%08lX", 
                       (long) ( wrdata.work.crypto.key.hi ),
                       (long) ( wrdata.work.crypto.key.lo ) );
                 break;
          case OGR:
                 strcpy(workunit, ogr_stubstr(&wrdata.work.ogr.workstub.stub));
                 break;
        }
        Log( "%s packet %s%c(%u.%u0%% complete)\n", msg, workunit,
              ((permille == 0)?('\0'):(' ')), permille/10, permille%10 );
      }
    } /* unconditional unload */
    
    if (*is_empty) /* we can purge the object now */
      thisprob->RetrieveState( NULL, NULL, 1 );
  }

  return norm_key_count;
}

/* ----------------------------------------------------------------------- */

#ifndef STRESS_RANDOMGEN
static long __loadapacket( Client *client, WorkRecord *wrdata, 
                          int /*ign_closed*/,  unsigned int prob_i )
{                    
  unsigned int cont_i = prob_i; /* consume variable */
  long bufcount = -1;

  for (cont_i = 0; (bufcount < 0 && cont_i < CONTEST_COUNT); cont_i++ )
  {
    unsigned int selproject = (unsigned int)client->loadorder_map[cont_i];
    if (selproject >= CONTEST_COUNT) /* user disabled */
      continue;
    if (!IsProblemLoadPermitted((long)prob_i, cont_i))
      continue; /* problem.cpp - result depends on #defs, threadsafety etc */
    bufcount = client->GetBufferRecord( wrdata, selproject, 0 );
//LogScreen("trying contest %d count %ld\n", selproject, bufcount );
  }
  return bufcount;
}  
#endif

/* ---------------------------------------------------------------------- */

#define NOLOAD_NONEWBLOCKS       -3
#define NOLOAD_ALLCONTESTSCLOSED -2
#define NOLOAD_NORANDOM          -1

static unsigned int __IndividualProblemLoad( Problem *thisprob, 
                    unsigned int prob_i, Client *client, int *load_needed, 
                    unsigned load_problem_count, unsigned int *contest,
                    int *bufupd_pending )
{
  int work_was_reset = 0;
  WorkRecord wrdata;
  unsigned int norm_key_count = 0;
  int didload = 0, didrandom = 0;
  long bufcount = -1;
  
#ifndef STRESS_RANDOMGEN
  bufcount = __loadapacket( client, &wrdata, 1, prob_i );
  if (bufcount < 0 && client->nonewblocks == 0)
  {
    int didupdate = 
       BufferUpdate(client,(BUFFERUPDATE_FETCH|BUFFERUPDATE_FLUSH),0);
    if (!(didupdate < 0))
    {
      if (client->randomchanged)        
        RefreshRandomPrefix( client );
      if (didupdate!=0)
        *bufupd_pending&=~(didupdate&(BUFFERUPDATE_FLUSH|BUFFERUPDATE_FETCH));
      if ((didupdate & BUFFERUPDATE_FETCH) != 0) /* fetched successfully */
        bufcount = __loadapacket( client, &wrdata, 0, prob_i );
    }
  }
#endif

  if (bufcount >= 0) /* load from file succeeded */
  {
    didload = 1;
    if (((unsigned long)(bufcount)) < (load_problem_count - prob_i))
      *bufupd_pending |= BUFFERUPDATE_FETCH;

    *load_needed = 0;
    
    switch (wrdata.contest) 
    {
      case RC5:
      case DES:
      case CSC:
        if ( ((wrdata.work.crypto.keysdone.lo)!=0) || 
             ((wrdata.work.crypto.keysdone.hi)!=0) )
        {
          s32 cputype = client->cputype; /* needed for FILEENTRY_CPU macro */

          #if (CLIENT_OS == OS_RISCOS) /* second thread is x86 */
          if (wrdata.contest == RC5 && prob_i == 1) cputype = CPU_X86;
          #endif

          // If this is a partial block, and completed by a different 
          // cpu/os/build, then reset the keysdone to 0...
          if ((wrdata.os      != FILEENTRY_OS) ||
              (wrdata.buildhi != FILEENTRY_BUILDHI) || 
              (wrdata.cpu     != FILEENTRY_CPU) || /*CLIENT_CPU+coretype */
              (wrdata.buildlo != FILEENTRY_BUILDLO))
          {
             wrdata.work.crypto.keysdone.lo = 0;
             wrdata.work.crypto.keysdone.hi = 0;
             work_was_reset = 1;
          }
          else if (((wrdata.work.crypto.iterations.lo) & 0x00000001L) == 1)
          {
            // If packet was finished with an 'odd' number of keys done, 
            // then make redo the last key
            wrdata.work.crypto.iterations.lo = wrdata.work.crypto.iterations.lo & 0xFFFFFFFEL;
            wrdata.work.crypto.key.lo = wrdata.work.crypto.key.lo & 0xFEFFFFFFL;
          }
        }
        break;
      case OGR:
        break;
    }
  } 
  else /* normal load from buffer failed */
  {
    int norandom = 1;
    if ((client->rc564closed == 0) && (client->blockcount >= 0))
    {
      unsigned int iii;
      for (iii=0;norandom && iii<CONTEST_COUNT;iii++)
      {
        if (client->loadorder_map[iii] == 0 /* rc5 is enabled in map */)
          norandom = 0;
      }
    }  
    if (norandom)
      *load_needed = NOLOAD_NORANDOM; /* -1 */
    else if (client->nonewblocks)
      *load_needed = NOLOAD_NONEWBLOCKS; /* -3 */
    else /* random blocks permitted */
    {
      *load_needed = 0;
      didload = 1;
      didrandom = 1;
      RefreshRandomPrefix(client); //get/put an up-to-date prefix 

      if (client->randomprefix == 0)
        client->randomprefix = 100;

      u32 randomprefix = ( (u32)(client->randomprefix) + 1 ) & 0xFF;
      u32 rnd = Random(NULL,0);

#if defined(STRESS_RANDOMGEN) && defined(STRESS_RANDOMGEN_ALL_KEYSPACE)
      ++client->randomprefix;
      if (client->randomprefix > 0xff)
        client->randomprefix = 100
#endif
      
      wrdata.id[0]                 = 0;
      wrdata.resultcode            = RESULT_WORKING;
      wrdata.os                    = 0;
      wrdata.cpu                   = 0;
      wrdata.buildhi               = 0;
      wrdata.buildlo               = 0;
      wrdata.contest               = RC5; // Random blocks are always RC5
      wrdata.work.crypto.key.lo    = (rnd & 0xF0000000L);
      wrdata.work.crypto.key.hi    = (rnd & 0x00FFFFFFL) + (randomprefix<<24);
      //constants are in rsadata.h
      wrdata.work.crypto.iv.lo     = ( RC564_IVLO );     //( 0xD5D5CE79L );
      wrdata.work.crypto.iv.hi     = ( RC564_IVHI );     //( 0xFCEA7550L );
      wrdata.work.crypto.cypher.lo = ( RC564_CYPHERLO ); //( 0x550155BFL );
      wrdata.work.crypto.cypher.hi = ( RC564_CYPHERHI ); //( 0x4BF226DCL );
      wrdata.work.crypto.plain.lo  = ( RC564_PLAINLO );  //( 0x20656854L );
      wrdata.work.crypto.plain.hi  = ( RC564_PLAINHI );  //( 0x6E6B6E75L );
      wrdata.work.crypto.keysdone.lo = 0;
      wrdata.work.crypto.keysdone.hi = 0;
      wrdata.work.crypto.iterations.lo = 1L<<28;
      wrdata.work.crypto.iterations.hi = 0;
    }
  }

#ifdef DEBUG
Log("Loadblock::End. %s\n", (didrandom)?("Success (random)"):((didload)?("Success"):("Failed")) );
#endif
    
  if (didload) /* success */
  {
    u32 timeslice = 0x10000;
    #if (defined(INIT_TIMESLICE) && (INIT_TIMESLICE > 64))
    timeslice = INIT_TIMESLICE;
    #endif
    *load_needed = 0;
    *contest = (unsigned int)(wrdata.contest);

    thisprob->LoadState( &wrdata.work, *contest, timeslice, client->cputype );
    thisprob->loaderflags = 0;

    switch (wrdata.contest) 
    {
      case RC5:
      case DES:
      case CSC:
      {
        norm_key_count = (unsigned int)__iter2norm((wrdata.work.crypto.iterations.lo),
                                                   (wrdata.work.crypto.iterations.hi));
        if (norm_key_count == 0) /* test block */
          norm_key_count = 1;
        #if 0
        if (norm_key_count == 0)
        {
          Log("Ouch!: normkeycount == 0 for iter %08x:%08x\n",
                      wrdata.work.crypto.iterations.hi,
                      wrdata.work.crypto.iterations.lo);
          norm_key_count++;
        }
        #endif
        break;
      }
      case OGR:
      {
        norm_key_count = 1;
        break;
      }
    }

    ClientEventSyncPost( CLIEVENT_PROBLEM_STARTED, (long)prob_i );

    if (load_problem_count <= COMBINEMSG_THRESHOLD)
    {
      char msgbuf[80]; msgbuf[0] = '\0';
      switch (*contest) 
      {
        case RC5:
        case DES:
        case CSC:
        {
          sprintf(msgbuf, "%s %u*2^28 packet %08lX:%08lX", 
                  ((didrandom)?(" random"):("")), norm_key_count,
                  (unsigned long) ( wrdata.work.crypto.key.hi ),
                  (unsigned long) ( wrdata.work.crypto.key.lo ) );
          break;
        }
        case OGR:
        {
          sprintf(msgbuf," stub %s", ogr_stubstr(&wrdata.work.ogr.workstub.stub) );
          break;
        }
      }
      if (msgbuf[0])
      {
        char perdone[48]; 
        unsigned int permille = (unsigned int)(thisprob->startpermille);
        perdone[0]='\0';
        if (permille!=0 && permille<=1000)
          sprintf(perdone, " (%u.%u0%% done)", (permille/10), (permille%10));
        Log("Loaded %s%s%s\n%s",
           CliGetContestNameFromID(*contest), msgbuf, perdone,
             (work_was_reset ? ("Packet was from a client "
             "with another cpu/os/build.\n"):("")) );
      }
    } /* if (load_problem_count <= COMBINEMSG_THRESHOLD) */
  } /* if (didload) */

  return norm_key_count;
}    

// --------------------------------------------------------------------

unsigned int LoadSaveProblems(Client *client,
                              unsigned int load_problem_count,int mode)
{
  static unsigned int previous_load_problem_count = 0, reentrant_count = 0;

  unsigned int retval = 0;
  int changed_flag;

  int allclosed, prob_step,bufupd_pending;  
  unsigned int prob_for, prob_first, prob_last;
  unsigned int norm_key_count, cont_i;
  unsigned int loaded_problems_count[CONTEST_COUNT];
  unsigned int loaded_normalized_key_count[CONTEST_COUNT];
  unsigned int saved_problems_count[CONTEST_COUNT];
  unsigned int saved_normalized_key_count[CONTEST_COUNT];
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

  if ((++reentrant_count) > 1)
  {
    --reentrant_count;
    return 0;
  }

  /* ============================================================= */

  prob_first = 0;
  prob_step  = 0;
  prob_last  = 0;

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
    if (CliGetContestInfoSummaryData( cont_i, &blocksdone, NULL, NULL )==0)
      totalBlocksDone += blocksdone;
   
    loaded_problems_count[cont_i]=loaded_normalized_key_count[cont_i]=0;
    saved_problems_count[cont_i] =saved_normalized_key_count[cont_i]=0;

    if ( ((unsigned long)(client->inthreshold[cont_i])) <
      (((unsigned long)(load_problem_count))<<1))
    {
      client->inthreshold[cont_i] = load_problem_count<<1;
    }
  }

  /* ============================================================= */

  ClientEventSyncPost(CLIEVENT_PROBLEM_TFILLSTARTED, (long)load_problem_count);

  for (prob_for = 0; prob_for <= (prob_last - prob_first); prob_for++)
  {
    Problem *thisprob;
    int load_needed;
    unsigned int prob_i = prob_for + prob_first;
    if ( prob_step < 0 )
      prob_i = prob_last - prob_for;
      
    thisprob = GetProblemPointerFromIndex( prob_i );
    if (thisprob == NULL)
    {
      if (prob_step < 0)
        continue;
      break;
    }

    // -----------------------------------

    load_needed = 0;
    norm_key_count = __IndividualProblemSave( thisprob, prob_i, client, 
          &load_needed, load_problem_count, &cont_i, &bufupd_pending,
          (mode == PROBFILL_UNLOADALL || mode == PROBFILL_RESIZETABLE ) );
    if (load_needed)
      empty_problems++;
    if (norm_key_count)
    {
      changed_flag = 1;
      total_problems_saved++;
      saved_normalized_key_count[cont_i] += norm_key_count;
      saved_problems_count[cont_i]++;
      totalBlocksDone++;
    }

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
        norm_key_count = __IndividualProblemLoad( thisprob, prob_i, client, 
                &load_needed, load_problem_count, &cont_i, &bufupd_pending );
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
        if (norm_key_count)
        {
          empty_problems--;
          total_problems_loaded++;
          loaded_normalized_key_count[cont_i] += norm_key_count;
          loaded_problems_count[cont_i]++;
          changed_flag = 1;
        }
      }
    } //if (load_needed)
  } //for (prob_i = 0; prob_i < load_problem_count; prob_i++ )

  ClientEventSyncPost(CLIEVENT_PROBLEM_TFILLFINISHED,
     (long)((previous_load_problem_count==0)?(total_problems_loaded):(total_problems_saved)));

  /* ============================================================= */

  for ( cont_i = 0; cont_i < CONTEST_COUNT; cont_i++) //once for each contest
  {
    if (loaded_problems_count[cont_i] || saved_problems_count[cont_i])
    {
      const char *cont_name = CliGetContestNameFromID(cont_i);

      if (loaded_problems_count[cont_i] && load_problem_count > COMBINEMSG_THRESHOLD )
      {
        Log( "Loaded %u %s packet%s (%u work unit%s) from %s\n", 
              loaded_problems_count[cont_i], cont_name,
              ((loaded_problems_count[cont_i]==1)?(""):("s")),
              loaded_normalized_key_count[cont_i],
              ((loaded_normalized_key_count[cont_i]==1)?(""):("s")),
              (client->nodiskbuffers ? "(memory-in)" : 
              BufferGetDefaultFilename( cont_i, 0, 
                                        client->in_buffer_basename )) );
      }

      if (saved_problems_count[cont_i] && load_problem_count > COMBINEMSG_THRESHOLD)
      {
        Log( "Saved %u %s packet%s (%u work unit%s) to %s\n", 
              saved_problems_count[cont_i], cont_name,
              ((saved_problems_count[cont_i]==1)?(""):("s")),
              saved_normalized_key_count[cont_i],
              ((saved_normalized_key_count[cont_i]==1)?(""):("s")),
              (mode == PROBFILL_UNLOADALL)?
                (client->nodiskbuffers ? "(memory-in)" : 
                BufferGetDefaultFilename( cont_i, 0, 
                                          client->in_buffer_basename ) ) :
                (client->nodiskbuffers ? "(memory-out)" : 
                BufferGetDefaultFilename( cont_i, 1, 
                                          client->out_buffer_basename )) );
      }

      if (totalBlocksDone > 0 /* && client->randomchanged == 0 */)
      {
        // To suppress "odd" problem completion count summaries (and not be
        // quite so verbose) we only display summaries if the number of
        // completed problems is even divisible by the number of processors.
        // Requires a working GetNumberOfDetectedProcessors() [cpucheck.cpp]
        // also check randomchanged in case a contest was closed/opened and
        // statistics haven't been reset
        #if 0
        int cpustmp; unsigned int cpus = 1;
        if ((cpustmp = GetNumberOfDetectedProcessors()) > 1)
          cpus = (unsigned int)cpustmp;
        if (load_problem_count > cpus)
          cpus = load_problem_count;
        if ((totalBlocksDone%cpus) == 0 )
        #endif
        {
          Log( "Summary: %s\n", CliGetSummaryStringForContest(cont_i) );
        }
      }

      /* -------------------------------------------------------------- */

      unsigned int inout;
      for (inout=0;inout<=1;inout++)
      {
        unsigned long norm_count;
        long block_count = client->GetBufferCount( cont_i, inout, &norm_count );
        if (block_count >= 0) /* no error */
        {
          char buffer[100+128 /*sizeof(client->in_buffer_basename)*/];
          if (inout != 0)                              /* out-buffer */
          {
            if (block_count > ((long)client->outthreshold[cont_i]))
              bufupd_pending |= BUFFERUPDATE_FLUSH;
          }
          else                                         /* in-buffer */
          {
            if (((unsigned long)(block_count)) < load_problem_count)
              bufupd_pending |= BUFFERUPDATE_FETCH;
          }
          sprintf(buffer, 
              "%ld %s packet%s (%lu work unit%s) %s in %s",
              block_count, 
              cont_name, 
              ((block_count == 1)?(""):("s")),  
              norm_count,
              ((norm_count == 1)?(""):("s")),
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
          Log( "%s\n", __WrapOrTruncateLogLine( buffer, 1 ));
        } //if (block_count >= 0)
      } //  for (inout=0;inout<=1;inout++)
    } //if (loaded_problems_count[cont_i] || saved_problems_count[cont_i])
  } //for ( cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)

  /* ============================================================ */

  if (mode == PROBFILL_UNLOADALL)
  {
    previous_load_problem_count = 0;
    if (client->nodiskbuffers == 0)
      client->CheckpointAction( CHECKPOINT_CLOSE, 0 );
    else
      BufferUpdate(client,BUFFERUPDATE_FLUSH,0);
    retval = total_problems_saved;
  }
  else
  {
    /* 
    =============================================================
    // save the number of active problems, so that we can bail out
    // in an "emergency". Some platforms call us asynchronously when they
    // need to abort [win16/win32 for example]
    -------------------------------------------------------------
    */

    previous_load_problem_count = load_problem_count;

    if (bufupd_pending)
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
  
  --reentrant_count;

  return retval;
}  
  
