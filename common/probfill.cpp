// Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
//
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: probfill.cpp,v $
// Revision 1.14  1998/12/04 16:50:07  cyp
// Non-interactive fetch automatically does a flush as well. Similarly,
// a non-interactive flush will also top off the input buffer (if the client
// is not in the process of shutting down).
//
// Revision 1.13  1998/12/01 19:49:14  cyp
// Cleaned up MULT1THREAD #define. See cputypes.h log entry for details.
//
// Revision 1.12  1998/11/30 23:41:12  cyp
// Probfill now handles blockcount limits; can resize the loaded problem
// table; closes checkpoint files when the problem table is closed; can
// be called asynchronously from Client::Run() in the event that the client
// is shutdown by an 'external' source such as the win16 message handler.
//
// Revision 1.11  1998/11/29 14:38:04  cyp
// Fixed missing outthreshold check. Added connectoften support (haven't
// tested yet). But why, why, why isn't the inthreshold check code working?
//
// Revision 1.10  1998/11/28 19:47:13  cyp
// __IndividualProblemLoad() retries all contests buffers after a buffer
// (net) update.
//
// Revision 1.9  1998/11/26 06:56:17  cyp
// Overhauled __IndividualProblemLoad() and added support for new buffer
// methods.
//
// Revision 1.8  1998/11/25 09:23:34  chrisb
// various changes to support x86 coprocessor under RISC OS
//
// Revision 1.7  1998/11/07 14:55:46  cyp
// Now also displays the normalized keycount (in addition to the number of
// blocks) in buffer files.
//
// Revision 1.6  1998/11/06 04:31:25  cyp
// Fixed InitializeProblemManager(): was returning 1 problem more than it was
// being asked for.
//
// Revision 1.5  1998/11/06 02:32:25  cyp
// Ok, no more restrictions (at least from the client's perspective) on the
// number of processors that the client can run on.
//
// Revision 1.4  1998/10/05 02:10:19  cyp
// Removed explicit time stamps ([%s],Time())
//
// Revision 1.3  1998/10/04 17:15:15  silby
// Made test block completion notice more verbose.
//
// Revision 1.2  1998/10/03 23:59:50  remi
// Removed extraneous #if defined(KWAN) && defined(MEGGS). MMX_BITSLICER is now
// defined only when the MMX DES core is compiled.
//
// Revision 1.1  1998/09/28 01:16:08  cyp
// Spun off from client.cpp
//
// 

#if (!defined(lint) && defined(__showids__))
const char *probfill_cpp(void) {
return "@(#)$Id: probfill.cpp,v 1.14 1998/12/04 16:50:07 cyp Exp $"; }
#endif

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "version.h"   // CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "client.h"    // CONTEST_COUNT, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "problem.h"   // Problem class
#include "scram.h"     // Descramble()
#include "network.h"   // sheesh... htonl()/ntohl()
#include "logstuff.h"  // Log()/LogScreen()
#include "clitime.h"   // CliGetTimeString()
#include "cpucheck.h"  // GetNumberOfDetectedProcessors()
#include "clisrate.h"  // CliGetMessageFor... et al.
#include "clicdata.h"  // CliGetContestNameFromID()
#include "clirate.h"   // CliGetKeyrateForProblem()
#include "probman.h"   // GetProblemPointerFromIndex()
#include "checkpt.h"   // CHECKPOINT_CLOSE define
#include "triggers.h"  // RaiseExitRequestTrigger()
#include "buffupd.h"   // BUFFERUPDATE_FETCH/_FLUSH define
#include "probfill.h"  // ourselves.

// =======================================================================
// each individual problem load+save generates 4 or more messages lines 
// (+>=3 lines for every load+save cycle), so we suppress/combine individual 
// load/save messages if the 'load_problem_count' exceeds COMBINEMSG_THRESHOLD
// into a single line 'Loaded|Saved n RC5|DES blocks from|to filename'.
#define COMBINEMSG_THRESHOLD 4 // anything above this and we don't show 
                               // individual load/save messages
// =======================================================================

static unsigned long __iter2norm( unsigned long iter )
{
  if (!iter)
    iter = 16;
  else
    {
    unsigned int size =  0;
    while (iter>1 && size<28)
      { size++; iter>>=1; }
    }
  return iter;
}  

// -----------------------------------------------------------------------

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

// -----------------------------------------------------------------------

#include "iniread.h"
#include "pathwork.h"

// update contestdone and randomprefix .ini entries
static void __RefreshRandomPrefix( Client *client )
{                        
  if (client->stopiniio == 0 && client->nodiskbuffers == 0)
    {
    const char *OPTION_SECTION = "parameters";
    IniSection ini;
    unsigned int cont_i;
    char buffer[32];
    s32 tmpconfig = ini.ReadIniFile( 
                    GetFullPathForFilename( client->inifilename ) );

    if (client->randomchanged)
      {
      ini.setrecord(OPTION_SECTION, "randomprefix", IniString((s32)(client->randomprefix)));
      for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
        {
        if (cont_i==0) strcpy(buffer,"contestdone");
        else sprintf(buffer,"contestdone%u", cont_i );
        ini.setrecord(OPTION_SECTION, buffer, IniString((s32)(client->contestdone[cont_i])));
        }
      ini.WriteIniFile( GetFullPathForFilename( client->inifilename ) );
      client->randomchanged = 0;
      }
    else if (!tmpconfig)
      {  
      tmpconfig=ini.getkey(OPTION_SECTION, "randomprefix", "0")[0];
      if (tmpconfig) client->randomprefix = tmpconfig;
      for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
        {
        if (cont_i==0) strcpy(buffer,"contestdone");
        else sprintf(buffer,"contestdone%u", cont_i );
        tmpconfig=ini.getkey(OPTION_SECTION, buffer, "0")[0];
        client->contestdone[cont_i] = (tmpconfig != 0);
        }
      }
    }
  return;
}

// -----------------------------------------------------------------------

static unsigned int __IndividualProblemUnload( Problem *thisprob, 
                unsigned int prob_i, Client *client, int *load_needed, 
                unsigned load_problem_count, unsigned int *contest,
                int *bufupd_pending )
{
  FileEntry fileentry;
  unsigned int cont_i;
  unsigned int norm_key_count = 0;
  unsigned long keyhi, keylo, percent;
  const char *msg;
  s32 cputype;

  *contest = 0;
  *load_needed = 0;
  prob_i = prob_i; //get rid of warning

  if (thisprob && thisprob->IsInitialized())
    {
    cont_i = (unsigned int)thisprob->RetrieveState( (ContestWork *) &fileentry , 1 );
    fileentry.contest = (u8)cont_i;
    *contest = cont_i;

    keyhi = ntohl( fileentry.key.hi );
    keylo = ntohl( fileentry.key.lo );
    percent = (unsigned long) ( (double) 10000.0 *
                           ((double) ntohl(fileentry.keysdone.lo) /
                            (double) ntohl(fileentry.iterations.lo) ) );
    norm_key_count = 
       (unsigned int)__iter2norm( ntohl(fileentry.iterations.lo) );

    cputype           = client->cputype; /* needed for FILEENTRY_CPU macro */
    fileentry.op      = htonl( OP_DATA );
    fileentry.cpu     = FILEENTRY_CPU;
#if (CLIENT_OS == OS_RISCOS)
    fileentry.cpu     = (prob_i == 0)?FILEENTRY_CPU:FILEENTRY_RISCOS_X86_CPU;
#endif
    fileentry.os      = FILEENTRY_OS;
    fileentry.buildhi = FILEENTRY_BUILDHI; 
    fileentry.buildlo = FILEENTRY_BUILDLO;

    fileentry.checksum =
          htonl( Checksum( (u32 *) &fileentry, ( sizeof(FileEntry)/4)-2));
    Scramble( ntohl( fileentry.scramble ),
                       (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );

    if (client->PutBufferRecord( &fileentry ) < 0)  // send it back...
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
      else
        msg = NULL;
      }
    if (msg)
      {
      Log( "%s block %08lX:%08lX (%d.%02d%% complete)\n", msg,
          (unsigned long) keyhi, (unsigned long) keylo,
            (unsigned int)(percent/100), (unsigned int)(percent%100) );
      }
    }
  return norm_key_count;
}

// -----------------------------------------------------------------------

static unsigned int __IndividualProblemSave( Problem *thisprob, 
                unsigned int prob_i, Client *client, int *load_needed, 
                unsigned load_problem_count, unsigned int *contest,
                int *bufupd_pending )
{                    
  FileEntry fileentry;
  RC5Result rc5result;
  unsigned int cont_i;
  unsigned int norm_key_count = 0;
  long longcount;
  prob_i = prob_i; //get rid of warning
  
  if ( thisprob->IsInitialized() && thisprob->GetResult( &rc5result ) != -1 &&
    (rc5result.result==RESULT_FOUND || rc5result.result==RESULT_NOTHING))
    {
    *load_needed = 1;        

    //----------------------------------------
    // Figure out which contest block was from
    //----------------------------------------

    //don't purge the state yet - we need it for stats later
    cont_i = thisprob->RetrieveState( (ContestWork *)&fileentry, 0 );
    fileentry.contest = (u8)(cont_i);
    *contest = cont_i;

    //---------------------
    //put the completed problem away
    //---------------------

    // make it into a reply
    if (rc5result.result == RESULT_FOUND)
      {
      client->consecutivesolutions[cont_i]++; 
      if (client->keyport == 3064)
        LogScreen("Test block success detected!\n");
      fileentry.op = htonl( OP_SUCCESS_MULTI );
      fileentry.key.lo = htonl( ntohl( fileentry.key.lo ) +
                          ntohl( fileentry.keysdone.lo ) );
      }
    else
      {
      if (client->keyport == 3064)
        LogScreen("Test block success was not detected!\n");
      fileentry.op = htonl( OP_DONE_MULTI );
      }

    fileentry.os      = CLIENT_OS;
    fileentry.cpu     = CLIENT_CPU;
#if (CLIENT_OS == OS_RISCOS)
    fileentry.cpu = (prob_i == 1)?CPU_X86:CPU_ARM;
#endif
    fileentry.buildhi = CLIENT_CONTEST;
    fileentry.buildlo = CLIENT_BUILD;
    strncpy( fileentry.id, client->id , sizeof(fileentry.id)-1); // set owner's id
    fileentry.id[sizeof(fileentry.id)-1]=0; //in case id>58 bytes, truncate.

    fileentry.checksum =
        htonl( Checksum( (u32 *) &fileentry, (sizeof(FileEntry)/4)-2) );
    Scramble( ntohl( fileentry.scramble ),
                (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );

    // send it back...
    if ( (longcount = client->PutBufferRecord( &fileentry )) < 0)
      {
      //Log( "PutBuffer Error\n" ); //error already printed?
      }
    else
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
      norm_key_count = 
          (unsigned int)__iter2norm( ntohl(rc5result.iterations.lo) );
      }

    //we can purge the object now
    thisprob->RetrieveState( (ContestWork *)&fileentry, 1 );
      
    } //RESULT_FOUND or RESULT_NOTHING

  return norm_key_count;
}

// -----------------------------------------------------------------------

static unsigned int __IndividualProblemLoad( Problem *thisprob, 
                    unsigned int prob_i, Client *client, int *load_needed, 
                    unsigned load_problem_count, unsigned int *contest,
                    int *bufupd_pending )
{
  FileEntry fileentry;
  unsigned int cont_i;
  unsigned int norm_key_count = 0;
  unsigned int contest_count, contest_selected;
  unsigned int contest_preferred, contest_alternate;
  long longcount;
  int didupdate, didload, didrandom, resetloop;
  s32 cputype;

  cputype           = client->cputype; /* needed for FILEENTRY_CPU macro */
  contest_preferred = (client->preferred_contest_id == 0)?(0):(1);
  contest_alternate = (contest_preferred == 0)?(1):(0);
  contest_count     = 2;
    
  #if ((CLIENT_CPU == CPU_X86) || (CLIENT_OS == OS_BEOS))
  /* Must do RC5.  Bryd DES x86 cores aren't thread safe. */
  if ( prob_i != 0 && prob_i != 1 ) /* Not the 1st or 2nd cracking thread... */
    {
    #if (defined(MMX_BITSLICER) && defined(KWAN) && defined(MEGGS))
    if ( des_unit_func != des_unit_func_mmx ) // if not using mmx cores
    #endif
      {
      contest_preferred = contest_alternate = 0;
      contest_count     = 1;
      }
    }
  #endif
  #if (CLIENT_OS == OS_RISCOS)
  /* RISC OS x86 thread currently only supports RC5 */
  if (prob_i == 1)
    {
    contest_preferred = contest_alternate = 0;
    contest_count     = 1;
    }
  #endif

  didrandom = didload = didupdate = 0;
  resetloop = 1;

  while (resetloop && didload == 0)
    {
    resetloop = 0;
    for (cont_i = 0;(didload == 0 && cont_i < contest_count); cont_i++)
      {
      contest_selected = ((cont_i)?(contest_alternate):(contest_preferred));
      if ( client->contestdone[contest_selected] == 0)
        {
        longcount = client->GetBufferRecord( &fileentry, contest_selected, 0 );
        if (longcount >= 0)
          {
          if (longcount == 0)
            *bufupd_pending |= BUFFERUPDATE_FETCH;
          didload = 1;
          break;
          }
        }
      }
    if (didupdate == 0 && didload == 0 && client->nonewblocks == 0)
      {
      didupdate = 
        client->BufferUpdate((BUFFERUPDATE_FETCH|BUFFERUPDATE_FLUSH),0);
      if (didupdate < 0)
        break;
      if (didupdate!=0)
        *bufupd_pending&=~(didupdate&(BUFFERUPDATE_FLUSH|BUFFERUPDATE_FETCH));
      if ((didupdate & BUFFERUPDATE_FETCH) == 0) /* didn't fetch */
        break;
      resetloop = 1; /* we refreshed buffers, so retry all */
      }
    }  
    
  if (didload) /* normal load from buffer succeeded */
    {
    *load_needed = 0;
    
    // LoadWork expects things descrambled.
    Descramble( ntohl( fileentry.scramble ),
                (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );

    if (fileentry.contest != 1)
      fileentry.contest=0;

    if ( (ntohl(fileentry.keysdone.lo)!=0) || 
         (ntohl(fileentry.keysdone.hi)!=0) )
      {
      // If this is a partial block, and completed by a different 
      // cpu/os/build, then reset the keysdone to 0...
      if ((fileentry.os      != FILEENTRY_OS) ||
          (fileentry.buildhi != FILEENTRY_BUILDHI) || 
#if (CLIENT_OS == OS_RISCOS)
          ((prob_i == 0) && (fileentry.cpu != FILEENTRY_CPU)) ||
          ((prob_i == 1) && (fileentry.cpu != FILEENTRY_RISCOS_X86_CPU)) ||
#else
          (fileentry.cpu     != FILEENTRY_CPU) ||
#endif
          (fileentry.buildlo != FILEENTRY_BUILDLO))
        {
        fileentry.keysdone.lo = fileentry.keysdone.hi = htonl(0);
        //LogScreen("Read partial block from another cpu/os/build.\n"
        // "Marking entire block as unchecked.\n");
        }
      else if ((ntohl(fileentry.iterations.lo) & 0x00000001L) == 1)
        {
        // If a block was finished with an 'odd' number of keys done, 
        // then make redo the last key
        fileentry.iterations.lo = htonl((ntohl(fileentry.iterations.lo) & 0xFFFFFFFEL) + 1);
        fileentry.key.lo = htonl(ntohl(fileentry.key.lo) & 0xFEFFFFFFL);
        }
      }
    } 
  else /* normal load from buffer failed */
    {
    if (client->contestdone[contest_preferred] && 
        client->contestdone[contest_alternate])
      *load_needed = -2;
    else if (client->nonewblocks)
      *load_needed = -3;
    else if (client->blockcount < 0) /* no random blocks permitted */
      *load_needed = -1;
    else /* random blocks permitted */
      {
      *load_needed = 0;
      didload = 1;
      didrandom = 1;
      __RefreshRandomPrefix(client); //get/put an up-to-date prefix 

      u32 randomprefix = ( ( (u32)(client->randomprefix) ) + 1 ) & 0xFF;
      fileentry.key.lo = htonl( Random( NULL, 0 ) & 0xF0000000L );
      fileentry.key.hi = htonl( (Random( NULL, 0 ) & 0x00FFFFFFL) + ( randomprefix << 24) ); // 64 bits significant
      fileentry.iv.lo = htonl( 0xD5D5CE79L );
      fileentry.iv.hi = htonl( 0xFCEA7550L );
      fileentry.cypher.lo = htonl( 0x550155BFL );
      fileentry.cypher.hi = htonl( 0x4BF226DCL );
      fileentry.plain.lo = htonl( 0x20656854L );
      fileentry.plain.hi = htonl( 0x6E6B6E75L );
      fileentry.keysdone.lo = htonl( 0 );
      fileentry.keysdone.hi = htonl( 0 );
      fileentry.iterations.lo = htonl( 0x10000000L );
      fileentry.iterations.hi = htonl( 0 );
      fileentry.id[0] = 0;
      fileentry.op = htonl( OP_DATA );
      fileentry.os = 0;
      fileentry.cpu = 0;
      fileentry.buildhi = 0;
      fileentry.buildlo = 0;
      fileentry.contest = 0; // Random blocks are always RC5, not DES.
      fileentry.checksum =
         htonl( Checksum( (u32 *)&fileentry, ( sizeof(FileEntry) / 4 ) - 2 ) );
      fileentry.scramble = htonl( Random( NULL, 0 ) );
      }
    }
    
  if (didload) /* success */
    {
    *load_needed = 0;
    *contest = (unsigned int)(fileentry.contest);
    thisprob->LoadState( (ContestWork *) &fileentry , 
          (u32) (fileentry.contest), client->timeslice, client->cputype );

    if (load_problem_count <= COMBINEMSG_THRESHOLD)
      {
      const char *cont_name = CliGetContestNameFromID(*contest);
      unsigned long iter = ntohl(fileentry.iterations.lo);
      unsigned int startpercent = (unsigned int)( thisprob->startpercent/10 );
      norm_key_count = (unsigned int)__iter2norm( iter );
      
      Log("Loaded %s%s %u*2^28 block %08lX:%08lX%c(%u.%02u%% done)",
              cont_name, ((didrandom)?(" random"):("")), norm_key_count,
              (unsigned long) ntohl( fileentry.key.hi ),
              (unsigned long) ntohl( fileentry.key.lo ),
              ((startpercent!=0 && startpercent<=10000)?(' '):(0)),
              (startpercent/100), (startpercent%100) );
      }
    } 

  return norm_key_count;
}    

// --------------------------------------------------------------------

unsigned int Client::LoadSaveProblems(unsigned int load_problem_count,int mode)
{
  static unsigned int previous_load_problem_count = 0; 

  Problem *thisprob;
  int load_needed, changed_flag;

  int i,prob_step,bufupd_pending;  
  unsigned int norm_key_count, prob_i, prob_for, cont_i;
  unsigned int loaded_problems_count[CONTEST_COUNT];
  unsigned int loaded_normalized_key_count[CONTEST_COUNT];
  unsigned int saved_problems_count[CONTEST_COUNT];
  unsigned int saved_normalized_key_count[CONTEST_COUNT];
  unsigned long totalBlocksDone; /* all contests */
  
  char buffer[100+sizeof(in_buffer_file[0])];
  unsigned int total_problems_loaded, total_problems_saved;
  unsigned int getbuff_errs;

  getbuff_errs = 0;
  changed_flag = (previous_load_problem_count == 0);
  total_problems_loaded = 0;
  total_problems_saved = 0;
  bufupd_pending = 0;
  totalBlocksDone = 0;
  
  for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
   {
   unsigned int blocksdone;
   if (CliGetContestInfoSummaryData( cont_i, &blocksdone, NULL, NULL )==0)
     totalBlocksDone += blocksdone;
   
   loaded_problems_count[cont_i]=loaded_normalized_key_count[cont_i]=0;
   saved_problems_count[cont_i] =saved_normalized_key_count[cont_i]=0;
   }

  /* ============================================================= */

  if (previous_load_problem_count == 0)
    {
    prob_step = 1;

    i = InitializeProblemManager(load_problem_count);
    if (i<=0)
      return 0;
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
    if (previous_load_problem_count == 0) 
      {
      if ( mode != PROBFILL_UNLOADALL )  
        load_needed = 1;
      }
    else
      {
      if (mode == PROBFILL_UNLOADALL || mode == PROBFILL_RESIZETABLE )
        {
        norm_key_count = __IndividualProblemUnload( thisprob, prob_i, this, 
          &load_needed, load_problem_count, &cont_i, &bufupd_pending );
        load_needed = 0;
        changed_flag = (norm_key_count!=0);
        }
      else
        {
        norm_key_count = __IndividualProblemSave( thisprob, prob_i, this, 
          &load_needed, load_problem_count, &cont_i, &bufupd_pending );
        if (load_needed)
          {
          changed_flag = 1;
          }
        }
      if (norm_key_count)
        {
        total_problems_saved++;
        saved_normalized_key_count[cont_i] += norm_key_count;
        saved_problems_count[cont_i]++;
        totalBlocksDone++;
        }
      }

    //---------------------------------------

    if (load_needed)
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
          getbuff_errs++;
        if (norm_key_count)
          {
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
    if (bufupd_pending == 0 && connectoften)
      {
      long block_count = GetBufferCount( cont_i, 0, NULL );
      if (block_count >= 0 && block_count < ((long)(inthreshold[cont_i])))
        bufupd_pending = BUFFERUPDATE_FETCH;
      else if ((GetBufferCount( cont_i, 1, NULL ) > 0))
        bufupd_pending = BUFFERUPDATE_FLUSH;
      }    
    if (loaded_problems_count[cont_i] || saved_problems_count[cont_i])
      {
      const char *cont_name = CliGetContestNameFromID(cont_i);

      if (loaded_problems_count[cont_i] && load_problem_count > COMBINEMSG_THRESHOLD )
        {
        Log( "Loaded %u %s block%s (%u*2^28 keys) from %s\n", 
              loaded_problems_count[cont_i], cont_name,
              ((loaded_problems_count[cont_i]==1)?(""):("s")),
              loaded_normalized_key_count[cont_i],
              (nodiskbuffers ? "(memory-in)" : in_buffer_file[cont_i]) );
        }

      if (saved_problems_count[cont_i] && load_problem_count > COMBINEMSG_THRESHOLD)
        {
        Log( "Saved %u %s block%s (%u*2^28 keys) to %s\n", 
              saved_problems_count[cont_i], cont_name,
              ((saved_problems_count[cont_i]==1)?(""):("s")),
              saved_normalized_key_count[cont_i],
              (mode == PROBFILL_UNLOADALL)?
                (nodiskbuffers ? "(memory-in)" : in_buffer_file[cont_i]) :
                (nodiskbuffers ? "(memory-out)" : out_buffer_file[cont_i]) );
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
    __RefreshRandomPrefix(this); //get/put an up-to-date prefix 


  /* ============================================================ */

    
  for ( cont_i = 0; cont_i < CONTEST_COUNT; cont_i++) //once for each contest
    {
    const char *cont_name = CliGetContestNameFromID(cont_i);
    if (contestdone[cont_i]==0 && 
      (bufupd_pending || loaded_problems_count[cont_i] || saved_problems_count[cont_i]))
      {
      unsigned int inout;
      for (inout=0;inout<=1;inout++)
        {
        unsigned long norm_count;
        long block_count = GetBufferCount( cont_i, inout, &norm_count );
        if (block_count >= 0)
          {
          sprintf(buffer, 
              "%ld %s block%s (%lu*2^28 keys) %s in %s",
              block_count, 
              cont_name, 
              ((block_count == 1)?(""):("s")),  
              norm_count,
              ((inout!= 0 || mode == PROBFILL_UNLOADALL)?
                 ((block_count==1)?("is"):("are")):
                 ((block_count==1)?("remains"):("remain"))),
              ((inout== 0)?
                 ((nodiskbuffers)?("(memory-in)"):(in_buffer_file[cont_i])):
                 ((nodiskbuffers)?("(memory-out)"):(out_buffer_file[cont_i])))
              );
          Log( "%s\n", __WrapOrTruncateLogLine( buffer, 1 ));
          } //if (block_count >= 0)
        } //  for (inout=0;inout<=1;inout++)
      } //if (loaded_problems_count[cont_i] || saved_problems_count[cont_i])
    } //for ( cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)

  /* ============================================================ */


  /* ============================================================ 
     close up shop if unloading all.
     ------------------------------------------------------------ */

  if (mode == PROBFILL_UNLOADALL)
   {
   previous_load_problem_count = 0;
   if (nodiskbuffers == 0)
    CheckpointAction( CHECKPOINT_CLOSE, 0 );
   DeinitializeProblemManager();
   return total_problems_saved;
   }

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

  if (blockcount > 0 && (totalBlocksDone >= (unsigned long)(blockcount)))
    {
    int limitsexceeded = 1;
    for (prob_i = 0; prob_i < load_problem_count; prob_i++ )
      {
      thisprob = GetProblemPointerFromIndex( prob_i );
      if (thisprob != NULL)
        {
        if (thisprob->IsInitialized()) /* still have probs open */
          {
          limitsexceeded = 0;
          blockcount = 1 + (u32)(totalBlocksDone);
          break;
          }
        }
      }    
    //----------------------------------------
    // Reached the -b limit?
    //----------------------------------------
    if (limitsexceeded)
      {
      Log( "Shutdown - block limit exceeded.\n" );
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

