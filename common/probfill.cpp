/* Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ****************** THIS IS WORLD-READABLE SOURCE *********************
 *
 * $Log: probfill.cpp,v $
 * Revision 1.43.2.1  1999/04/05 07:46:36  jlawson
 * added probfill_noblocks and probfill_disabled constants.
 *
 * Revision 1.43  1999/04/01 07:15:12  jlawson
 * corrected argument warning for 64-bit platforms.
 *
 * Revision 1.42  1999/04/01 03:20:41  cyp
 * Updated to reflect changed [in|out]_buffer_[file->basename] semantics.
 *
 * Revision 1.41  1999/03/20 07:41:33  cyp
 * Modified for use with new WorkRecord structure.
 *
 * Revision 1.40  1999/03/18 06:13:08  cyp
 * disable random block generation if rc5 is disabled.
 *
 * Revision 1.39  1999/03/18 03:07:54  cyp
 * This module is now (mostly) OGR ready. Some Log() msgs still need 
 * adjustment.
 *
 * Revision 1.38  1999/03/09 07:15:45  gregh
 * Various OGR changes.
 *
 * Revision 1.37  1999/03/02 04:51:35  silby
 * Fixed % complete displayed on shutdown.
 *
 * Revision 1.36  1999/03/02 03:49:49  silby
 * Large block support added, des support fixed (unrelated.)
 *
 * Revision 1.35  1999/03/01 08:19:44  gregh
 * Changed ContestWork to a union that contains crypto (RC5/DES) and OGR data.
 *
 * Revision 1.34  1999/02/21 21:44:59  cyp
 * tossed all redundant byte order changing. all host<->net order conversion
 * as well as scram/descram/checksumming is done at [get|put][net|disk] points
 * and nowhere else.
 *
 * Revision 1.33  1999/02/13 00:15:13  silby
 * Updated iter2norm to handle 64-bit blocks.
 *
 * Revision 1.32  1999/02/01 02:31:59  cyp
 * Added protection against repetitive unload_all which is possible when a
 * client shuts down asynchronously. (win16/32 clients do this when win ends)
 *
 * Revision 1.31  1999/01/14 18:05:44  cyp
 * Fixed erroneous increment of client->randomprefix.
 *
 * Revision 1.30  1999/01/01 02:45:16  cramer
 * Part 1 of 1999 Copyright updates...
 *
 * Revision 1.29  1998/12/29 19:20:36  cyp
 * Post start/finish events at beginning/end of LoadSaveProblems()
 *
 * Revision 1.28  1998/12/28 18:15:16  cyp
 * Net update (unconditionally, don't check other contests first) if no blocks
 * are available. Implemented generic event posting for started/finished probs.
 *
 * Revision 1.27  1998/12/28 04:01:07  silby
 * Win32gui icon now changed by probfill when new blocks are loaded.
 * If MacOS has an icon to change, this would be a good place to 
 * hook in as well.
 *
 * Revision 1.26  1998/12/23 03:24:56  silby
 * Client once again listens to keyserver for next contest start time,
 * tested, it correctly updates.  Restarting after des blocks have
 * been recieved has not yet been implemented, I don't have a clean
 * way to do it yet.  Writing of contest data to the .ini has been
 * moved back to confrwv with its other ini friends.
 *
 * Revision 1.25  1998/12/22 23:03:22  silby
 * Moved rc5 cipher/iv/etc back into rsadata.h - should be in there
 * because the file is shared with the proxy source.
 *
 * Revision 1.24  1998/12/22 21:14:50  cyp
 * Change hardcoded date to Jan 18 for DES-III
 *
 * Revision 1.23  1998/12/21 19:06:08  cyp
 * Removed 'unused'/'unimplemented' sil[l|b]yness added in recent version.
 * See client.h for full comment.
 *
 * Revision 1.22  1998/12/20 23:00:35  silby
 * Descontestclosed value is now stored and retrieved from the ini file,
 * additional updated of the .ini file's contest info when fetches and
 * flushes are performed are now done.  Code to throw away old des blocks
 * has not yet been implemented.
 *
 * Revision 1.21  1998/12/20 18:26:43  silby
 * RC5 iv/cipher/plain are pulled from contestdata.h now, made preferred 
 * contest setting more truthful.
 *
 * Revision 1.20  1998/12/20 17:39:50  cyp
 * Minor tweaks for contestdone 'advertising' such as immediate write of
 * contestdone flags after a successful fetch/flush. Also added hardcoded DES
 * switchover time frames and preferred_contest to 1 (if DES is supported).
 *
 * Revision 1.19  1998/12/16 05:50:54  cyp
 * Removed connectoften support. Client::Run() does a better job.
 *
 * Revision 1.18  1998/12/14 23:45:24  remi
 * Now we don't need to switch to RC5 when more than 2 threads are
 * available during DES contests.
 *
 * Revision 1.17  1998/12/12 12:22:30  cyp
 * 'exit when buffers are empty' (ie blockcount<0) now works correctly.
 *
 * Revision 1.16  1998/12/08 05:58:18  dicamillo
 * Add MacOS GUI call to delete thread display at completion.
 *
 * Revision 1.15  1998/12/06 02:55:09  cyp
 * The problem loader now looks ahead in an attempt to guess whether the next
 * load/save cycle will have enough blocks to refresh all problems without a
 * network update.
 *
 * Revision 1.14  1998/12/04 16:50:07  cyp
 * Non-interactive fetch automatically does a flush as well. Similarly,
 * a non-interactive flush will also top off the input buffer (if the client
 * is not in the process of shutting down).
 *
 * Revision 1.13  1998/12/01 19:49:14  cyp
 * Cleaned up MULT1THREAD #define. See cputypes.h log entry for details.
 *
 * Revision 1.12  1998/11/30 23:41:12  cyp
 * Probfill now handles blockcount limits; can resize the loaded problem
 * table; closes checkpoint files when the problem table is closed; can
 * be called asynchronously from Client::Run() in the event that the client
 * is shutdown by an 'external' source such as the windoze message handler.
 *
 * Revision 1.11  1998/11/29 14:38:04  cyp
 * Fixed missing outthreshold check. Added connectoften support (haven't
 * tested yet). 
 *
 * Revision 1.10  1998/11/28 19:47:13  cyp
 * __IndividualProblemLoad() retries all contests buffers after a buffer
 * (net) update.
 *
 * Revision 1.9  1998/11/26 06:56:17  cyp
 * Overhauled __IndividualProblemLoad() and added support for new buffer
 * methods.
 *
 * Revision 1.8  1998/11/25 09:23:34  chrisb
 * various changes to support x86 coprocessor under RISC OS
 *
 * Revision 1.7  1998/11/07 14:55:46  cyp
 * Now also displays the normalized keycount (in addition to the number of
 * blocks) in buffer files.
 *
 * Revision 1.6  1998/11/06 04:31:25  cyp
 * Fixed InitializeProblemManager(): was returning 1 problem more than it was
 * being asked for.
 *
 * Revision 1.5  1998/11/06 02:32:25  cyp
 * Ok, no more restrictions (at least from the client's perspective) on the
 * number of processors that the client can run on.
 *
 * Revision 1.4  1998/10/05 02:10:19  cyp
 * Removed explicit time stamps ([%s],Time())
 *
 * Revision 1.3  1998/10/04 17:15:15  silby
 * Made test block completion notice more verbose.
 *
 * Revision 1.2  1998/10/03 23:59:50  remi
 * Removed extraneous #if defined(KWAN) && defined(MEGGS). MMX_BITSLICER is
 * now defined only when the MMX DES core is compiled.
 *
 * Revision 1.1  1998/09/28 01:16:08  cyp
 * Spun off from client.cpp, complete rewrite.
 *
*/

#if (!defined(lint) && defined(__showids__))
const char *probfill_cpp(void) {
return "@(#)$Id: probfill.cpp,v 1.43.2.1 1999/04/05 07:46:36 jlawson Exp $"; }
#endif

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
      thisprob->loaderflags |= loaderflags_map[thisprob->contest] &
          (PROBLDR_DISCARD | PROBLDR_FORCEUNLOAD);
    prob_i++;
  }
  return ((prob_i == 0) ? (-1) : ((int)prob_i));
}  

/* ----------------------------------------------------------------------- */

static unsigned int __IndividualProblemSave( Problem *thisprob, 
                unsigned int prob_i, Client *client, int *is_empty, 
                unsigned load_problem_count, unsigned int *contest,
                int *bufupd_pending, int unconditional_unload )
{                    
  WorkRecord wrdata;
  int resultcode;
  unsigned int cont_i;
  unsigned int norm_key_count = 0;
  long longcount;
  s32 cputype = client->cputype; /* needed for FILEENTRY_CPU macro */
  #if (CLIENT_OS == OS_RISCOS)
  if (prob_i == 1) cputype = CPU_X86; 
  #endif
  prob_i = prob_i; //get rid of warning

  memset( (void *)&wrdata, 0, sizeof(WorkRecord));

  *contest = 0;
  *is_empty = 0;

  if ( thisprob->IsInitialized()==0 )  
  {
    *is_empty = 1; 
  }
  else if (-1 == ( resultcode = 
            thisprob->RetrieveState( &wrdata.work, &cont_i, 0 )))
  {
    *is_empty = 1;                  /* should not have happened */
  }
  else if (resultcode == RESULT_FOUND || resultcode == RESULT_NOTHING )
  {
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
      switch (cont_i) 
      {
        case 0: // RC5
        case 1: // DES
          norm_key_count = 
            (unsigned int)__iter2norm( wrdata.work.crypto.iterations.lo,
                                       wrdata.work.crypto.iterations.hi );
          if (norm_key_count == 0) /* test block */
            norm_key_count = 1;
          break;
        case 2: //OGR
          norm_key_count = 1;
          break;
      }

      ClientEventSyncPost( CLIEVENT_PROBLEM_FINISHED, (long)prob_i );
    }

    //event posted, keyrate computed, we can purge the object now
    thisprob->RetrieveState( NULL, NULL, 1 );
  } 
  else if (unconditional_unload || 
    (thisprob->loaderflags & (PROBLDR_DISCARD|PROBLDR_FORCEUNLOAD)) != 0) 
  {                                       /* must be RESULT_WORKING */
    const char *msg = NULL;
    char workunit[80];
    unsigned long percent = 0;

    *contest = cont_i;
    *is_empty = 1; /* will soon be */

    wrdata.contest    = (u8)cont_i;
    wrdata.resultcode = resultcode;
    wrdata.cpu        = FILEENTRY_CPU; /* uses cputype variable */
    wrdata.os         = FILEENTRY_OS;
    wrdata.buildhi    = FILEENTRY_BUILDHI; 
    wrdata.buildlo    = FILEENTRY_BUILDLO;

    switch (cont_i) 
    {
      case 0: // RC5
      case 1: // DES
      {
        percent = (u32) (((double)(10000.0)) *
        (((((double)(wrdata.work.crypto.keysdone.hi))*((double)(4294967296.0)))+
                                 ((double)(wrdata.work.crypto.keysdone.lo))) /
        ((((double)(wrdata.work.crypto.iterations.hi))*((double)(4294967296.0)))+
                                 ((double)(wrdata.work.crypto.iterations.lo)))) );

        norm_key_count = 
           (unsigned int)__iter2norm( (wrdata.work.crypto.iterations.lo),
                                      (wrdata.work.crypto.iterations.hi) );
        if (norm_key_count == 0) /* test block */
          norm_key_count = 1;
        sprintf(workunit, "%08lX:%08lX", 
		(long) ( wrdata.work.crypto.key.hi ),
		(long) ( wrdata.work.crypto.key.lo ) );
        break;
      }
      case 2: // OGR
      {
        norm_key_count = 1;
        strcpy(workunit, ogr_stubstr(&wrdata.work.ogr.stub));
        break;
      }
    }

    if ((thisprob->loaderflags & PROBLDR_DISCARD)!=0)
    {
      msg = "Discarded (project disabled)";
      norm_key_count = 0;
    }
    else if (client->PutBufferRecord( &wrdata ) < 0)  // send it back...
    {
      msg = "Unable to save";
      norm_key_count = 0;
    }
    else
    {
      if (client->nodiskbuffers)
        *bufupd_pending |= BUFFERUPDATE_FLUSH;
      if (load_problem_count <= COMBINEMSG_THRESHOLD)
        msg = "Saved";
    }
    if (msg)
    {
      Log( "%s packet %s (%d.%02d%% complete)\n", msg, workunit,
            (unsigned int)(percent/100), (unsigned int)(percent%100) );
    }

    //no event for this, we can purge the object now
    thisprob->RetrieveState( NULL, NULL, 1 );

  } /* unconditional unload */

  return norm_key_count;
}


/* ----------------------------------------------------------------------- */

static long __loadapacket( Client *client, WorkRecord *wrdata, 
                          int /*ign_closed*/,  unsigned int prob_i )
{                    
  unsigned int cont_i = prob_i; /* consume variable */
  long bufcount = -1;

  for (cont_i = 0; (bufcount < 0 && cont_i < CONTEST_COUNT); cont_i++ )
  {
    unsigned int selproject = (unsigned int)client->loadorder_map[cont_i];

#ifndef GREGH
if (selproject == 2)
  continue;
#endif  

    if (selproject >= CONTEST_COUNT) /* user disabled */
      continue;
      
    #if (CLIENT_OS == OS_RISCOS) /* RISC OS x86 thread only supports RC5 */
    if (prob_i == 1 && selproject != 0)
      continue;
    #endif

    bufcount = client->GetBufferRecord( wrdata, selproject, 0 );
//LogScreen("trying contest %d count %ld\n", selproject, bufcount );
  }
  return bufcount;
}  

/* ---------------------------------------------------------------------- */

#define NOLOAD_NONEWBLOCKS       -3
#define NOLOAD_ALLCONTESTSCLOSED -2
#define NOLOAD_NORANDOM          -1

static unsigned int __IndividualProblemLoad( Problem *thisprob, 
                    unsigned int prob_i, Client *client, int *load_needed, 
                    unsigned load_problem_count, unsigned int *contest,
                    int *bufupd_pending )
{
  WorkRecord wrdata;
  unsigned int norm_key_count = 0;
  int didload = 0, didrandom = 0;
  long bufcount = __loadapacket( client, &wrdata, 1, prob_i );

  if (bufcount < 0 && client->nonewblocks == 0)
  {
    int didupdate = 
       client->BufferUpdate((BUFFERUPDATE_FETCH|BUFFERUPDATE_FLUSH),0);
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

  if (bufcount >= 0) /* load from file succeeded */
  {
    didload = 1;
    if (((unsigned long)(bufcount)) < (load_problem_count - prob_i))
      *bufupd_pending |= BUFFERUPDATE_FETCH;

    *load_needed = 0;
    
    switch (wrdata.contest) 
    {
      case 0: // RC5
      case 1: // DES
        if ( ((wrdata.work.crypto.keysdone.lo)!=0) || 
             ((wrdata.work.crypto.keysdone.hi)!=0) )
        {
          s32 cputype = client->cputype; /* needed for FILEENTRY_CPU macro */

          #if (CLIENT_OS == OS_RISCOS) /* second thread is x86 */
          if (prob_i == 1) cputype = CPU_X86;
          #endif

          // If this is a partial block, and completed by a different 
          // cpu/os/build, then reset the keysdone to 0...
          if ((wrdata.os      != FILEENTRY_OS) ||
              (wrdata.buildhi != FILEENTRY_BUILDHI) || 
              (wrdata.cpu     != FILEENTRY_CPU) || /* uses 'cputype' variable */
              (wrdata.buildlo != FILEENTRY_BUILDLO))
          {
             wrdata.work.crypto.keysdone.lo = 0;
             wrdata.work.crypto.keysdone.hi = 0;
            //LogScreen("Read partial packet from another cpu/os/build.\n"
            //                     "Marking entire packet as unchecked.\n");
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
      case 2: // OGR
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
      
      wrdata.id[0]                 = 0;
      wrdata.resultcode            = RESULT_WORKING;
      wrdata.os                    = 0;
      wrdata.cpu                   = 0;
      wrdata.buildhi               = 0;
      wrdata.buildlo               = 0;
      wrdata.contest               = 0; // Random blocks are always RC5
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
      case 0: // RC5
      case 1: // DES
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
      case 2: // OGR
      {
        norm_key_count = wrdata.work.ogr.stub.marks;
        break;
      }
    }

    ClientEventSyncPost( CLIEVENT_PROBLEM_STARTED, (long)prob_i );

    if (load_problem_count <= COMBINEMSG_THRESHOLD)
    {
      const char *cont_name = CliGetContestNameFromID(*contest);
      unsigned int startpercent = (unsigned int)( thisprob->startpercent/10 );

      switch (*contest) 
      {
        case 0: // RC5
        case 1: // DES
        {
          Log("Loaded %s%s %u*2^28 packet %08lX:%08lX%c(%u.%02u%% done)",
                  cont_name, ((didrandom)?(" random"):("")), norm_key_count,
                  (unsigned long) ( wrdata.work.crypto.key.hi ),
                  (unsigned long) ( wrdata.work.crypto.key.lo ),
                  ((startpercent!=0 && startpercent<=10000)?(' '):(0)),
                  (startpercent/100), (startpercent%100) );
          break;
        }
        case 2: // OGR
        {
          Log("Loaded %s stub %s (%u.%02u%% done)",
                  cont_name,
                  ogr_stubstr(&wrdata.work.ogr.stub),
                  ((startpercent!=0 && startpercent<=10000)?(' '):(0)),
                  (startpercent/100), (startpercent%100) );
          break;
        }
      }
    } /* if (load_problem_count <= COMBINEMSG_THRESHOLD) */
  } /* if (didload) */

  return norm_key_count;
}    

// --------------------------------------------------------------------

#ifdef DEBUG
void __event_test( int event_id, long parm )
{
  if (event_id == CLIEVENT_PROBLEM_STARTED)
    LogScreen("event: CLIEVENT_PROBLEM_STARTED. parm=%lu\n", parm );
  else if (event_id == CLIEVENT_PROBLEM_FINISHED)
    LogScreen("event: CLIEVENT_PROBLEM_FINISHED. parm=%lu\n", parm );
}
#endif

unsigned int Client::LoadSaveProblems(unsigned int load_problem_count,int mode)
{
  static unsigned int previous_load_problem_count = 0, reentrant_count = 0;

  Problem *thisprob;
  int load_needed, changed_flag;

  int allclosed, i,prob_step,bufupd_pending;  
  unsigned int norm_key_count, prob_i, prob_for, cont_i;
  unsigned int loaded_problems_count[CONTEST_COUNT];
  unsigned int loaded_normalized_key_count[CONTEST_COUNT];
  unsigned int saved_problems_count[CONTEST_COUNT];
  unsigned int saved_normalized_key_count[CONTEST_COUNT];
  unsigned long totalBlocksDone; /* all contests */
  
  char buffer[100+sizeof(in_buffer_basename[0])];
  unsigned int total_problems_loaded, total_problems_saved;
  unsigned int norandom_count, getbuff_errs, empty_problems;

  allclosed = 0;
  norandom_count = getbuff_errs = empty_problems = 0;
  changed_flag = (previous_load_problem_count == 0);
  total_problems_loaded = 0;
  total_problems_saved = 0;
  bufupd_pending = 0;
  totalBlocksDone = 0;

  if (mode == PROBFILL_UNLOADALL)
    {
    if (previous_load_problem_count == 0)
      return 0;
    //load_problem_count = previous_load_problem_count;
    }
  if ((++reentrant_count) > 1)
    {
    --reentrant_count;
    return 0;
    }
  for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
   {
   unsigned int blocksdone;
   if (CliGetContestInfoSummaryData( cont_i, &blocksdone, NULL, NULL )==0)
     totalBlocksDone += blocksdone;
   
   loaded_problems_count[cont_i]=loaded_normalized_key_count[cont_i]=0;
   saved_problems_count[cont_i] =saved_normalized_key_count[cont_i]=0;
   }


  static time_t nextrandomcheck = (time_t)0;
  time_t timenow = CliTimer(NULL)->tv_sec;
  if (nextrandomcheck <= timenow)
    {
    RefreshRandomPrefix(this); //get/put an up-to-date prefix 
    nextrandomcheck = (timenow + (time_t)(10*60));
    }

  /* ============================================================= */

  if (previous_load_problem_count == 0)
    {
    prob_step = 1;

    #ifdef DEBUG
    ClientEventAddListener(-1, __event_test );
    #endif

    i = InitializeProblemManager(load_problem_count);
    if (i<=0)
      {
      --reentrant_count;
      return 0;
      }
    load_problem_count = (unsigned int)i;

    for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
      {
      if ( ((unsigned long)(inthreshold[cont_i])) <
         (((unsigned long)(load_problem_count))<<1))
        {
        inthreshold[cont_i] = load_problem_count<<1;
        }
      }
    }
  else
    {
    if (load_problem_count == 0)
      load_problem_count = previous_load_problem_count;
    prob_step = -1;
    }

  /* ============================================================= */

  ClientEventSyncPost(CLIEVENT_PROBLEM_TFILLSTARTED, (long)load_problem_count);

  prob_for = 0;

  if (mode == PROBFILL_RESIZETABLE && previous_load_problem_count)
    {
    prob_for = load_problem_count;
    load_problem_count = previous_load_problem_count;
    }
    
  /* ============================================================= */

  for (; prob_for<load_problem_count; prob_for++)
    {
    prob_i= (prob_step < 0) ? ((load_problem_count-1)-prob_for) : (prob_for);

    thisprob = GetProblemPointerFromIndex( prob_i );
    if (thisprob == NULL)
      {
      if (prob_step < 0)
        continue;
      break;
      }

    // -----------------------------------

    load_needed = 0;
    norm_key_count = __IndividualProblemSave( thisprob, prob_i, this, 
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
      if (blockcount>0 && totalBlocksDone>=((unsigned long)(blockcount)))
        {
        ; //nothing
        }
      else
        {
        load_needed = 0;
        norm_key_count = __IndividualProblemLoad( thisprob, prob_i, this, 
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

  /* ============================================================= */

  if (mode == PROBFILL_UNLOADALL && nodiskbuffers)
    bufupd_pending = BUFFERUPDATE_FLUSH;

  /* ============================================================= */

  for ( cont_i = 0; cont_i < CONTEST_COUNT; cont_i++) //once for each contest
    {
    if (loaded_problems_count[cont_i] || saved_problems_count[cont_i])
      {
      const char *cont_name = CliGetContestNameFromID(cont_i);

      if (loaded_problems_count[cont_i] && load_problem_count > COMBINEMSG_THRESHOLD )
        {
        Log( "Loaded %u %s packet%s (%u*2^28 keys) from %s\n", 
              loaded_problems_count[cont_i], cont_name,
              ((loaded_problems_count[cont_i]==1)?(""):("s")),
              loaded_normalized_key_count[cont_i],
              (nodiskbuffers ? "(memory-in)" : 
              BufferGetDefaultFilename( cont_i, 0, in_buffer_basename )) );
        }

      if (saved_problems_count[cont_i] && load_problem_count > COMBINEMSG_THRESHOLD)
        {
        Log( "Saved %u %s packet%s (%u*2^28 keys) to %s\n", 
              saved_problems_count[cont_i], cont_name,
              ((saved_problems_count[cont_i]==1)?(""):("s")),
              saved_normalized_key_count[cont_i],
              (mode == PROBFILL_UNLOADALL)?
                (nodiskbuffers ? "(memory-in)" : 
                BufferGetDefaultFilename( cont_i, 0, in_buffer_basename ) ) :
                (nodiskbuffers ? "(memory-out)" : 
                BufferGetDefaultFilename( cont_i, 1, out_buffer_basename )) );
        }

      // To suppress "odd" problem completion count summaries (and not be
      // quite so verbose) we only display summaries if the number of
      // completed problems is even divisible by the number of processors.
      // Requires a working GetNumberOfDetectedProcessors() [cpucheck.cpp]
      // also check randomchanged in case a contest was closed/opened and
      // statistics haven't been reset

      if (totalBlocksDone > 0 && randomchanged == 0)
        {
        if ((i = GetNumberOfDetectedProcessors()) < 1)
          i = 1;
        if (load_problem_count > ((unsigned int)(i)))
          i = (int)load_problem_count;
        if (((totalBlocksDone)%((unsigned int)(i))) == 0 )
          {
          Log( "Summary: %s\n", CliGetSummaryStringForContest(cont_i) );
          }
        }
      } //if (loaded_problems_count[cont_i] || saved_problems_count[cont_i])
    } //for ( cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)

  /* ============================================================ */


  if (bufupd_pending)
    {
    if ((bufupd_pending & BUFFERUPDATE_FETCH)!=0) // always flush while
      bufupd_pending |= BUFFERUPDATE_FLUSH;      // fetching
    else if (mode!=PROBFILL_UNLOADALL)           // if not ending the client
      bufupd_pending |= BUFFERUPDATE_FETCH;      // try to fetch anyway.
    if ((bufupd_pending = BufferUpdate(bufupd_pending,0)) < 0)
      bufupd_pending = 0;
    }
  if (randomchanged)
    {
    RefreshRandomPrefix(this); //get/put an up-to-date prefix 
    nextrandomcheck = (timenow + (time_t)(10*60));
    }

  /* ============================================================ */

    
  for ( cont_i = 0; cont_i < CONTEST_COUNT; cont_i++) //once for each contest
    {
    const char *cont_name = CliGetContestNameFromID(cont_i);
    if (bufupd_pending || loaded_problems_count[cont_i] || saved_problems_count[cont_i])
      {
      unsigned int inout;
      for (inout=0;inout<=1;inout++)
        {
        unsigned long norm_count;
        long block_count = GetBufferCount( cont_i, inout, &norm_count );
        if (block_count >= 0)
          {
          sprintf(buffer, 
              "%ld %s packet%s (%lu*2^28 keys) %s in %s",
              block_count, 
              cont_name, 
              ((block_count == 1)?(""):("s")),  
              norm_count,
              ((inout!= 0 || mode == PROBFILL_UNLOADALL)?
                 ((block_count==1)?("is"):("are")):
                 ((block_count==1)?("remains"):("remain"))),
              ((inout== 0)?
                (nodiskbuffers ? "(memory-in)" : BufferGetDefaultFilename( cont_i, 0, in_buffer_basename ) ) :
                (nodiskbuffers ? "(memory-out)": BufferGetDefaultFilename( cont_i, 1, out_buffer_basename ) ))
              );
          Log( "%s\n", __WrapOrTruncateLogLine( buffer, 1 ));
          } //if (block_count >= 0)
        } //  for (inout=0;inout<=1;inout++)
      } //if (loaded_problems_count[cont_i] || saved_problems_count[cont_i])
    } //for ( cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)

  /* ============================================================ */

  ClientEventSyncPost(CLIEVENT_PROBLEM_TFILLFINISHED,
     (long)((previous_load_problem_count==0)?(total_problems_loaded):(total_problems_saved)));

  --reentrant_count;

  if (mode == PROBFILL_UNLOADALL)
    {
    previous_load_problem_count = 0;
    if (nodiskbuffers == 0)
     CheckpointAction( CHECKPOINT_CLOSE, 0 );
    DeinitializeProblemManager();

    return total_problems_saved;
    }

  if (mode == PROBFILL_RESIZETABLE)
    return total_problems_saved;

  /* ============================================================
     save the number of active problems, so that we can bail out
     in an "emergency". Some platforms call us asynchronously when they
     need to abort [win16 for example]
     ------------------------------------------------------------ */

  previous_load_problem_count = load_problem_count;

  /* =============================================================
     if we are running a limited number of blocks then check if we have
     exceeded that number. If we have, but one or more crunchers are
     still at work, bump the limit. 
     ------------------------------------------------------------- */

  if (!allclosed)
    {
    int limitsexceeded = 0;
    if (blockcount < 0 && norandom_count >= load_problem_count)
      limitsexceeded = 1;
    if (blockcount > 0 && (totalBlocksDone >= (unsigned long)(blockcount)))
      {
      if (empty_problems >= load_problem_count)
        limitsexceeded = 1;
      else
        blockcount = ((u32)(totalBlocksDone))+1;
      }
    if (limitsexceeded)
      {
      Log( "Shutdown - packet limit exceeded.\n" );
      RaiseExitRequestTrigger();
      }
    }

  /* ============================================================= */

  if (mode == PROBFILL_GETBUFFERRS) 
    return getbuff_errs;
  if (mode == PROBFILL_ANYCHANGED)
    return changed_flag;

  return total_problems_loaded;
}  

// -----------------------------------------------------------------------

