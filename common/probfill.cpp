// Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
//
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: probfill.cpp,v $
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
return "@(#)$Id: probfill.cpp,v 1.9 1998/11/26 06:56:17 cyp Exp $"; }
#endif

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "version.h"   // CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "problem.h"   // Problem class
#include "scram.h"     // Descramble()
#include "network.h"   // sheesh... htonl()/ntohl()
#include "buffwork.h"  // 
#include "logstuff.h"  // Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "clitime.h"   // CliGetTimeString()
#include "cpucheck.h"  // GetNumberOfDetectedProcessors()
#include "clisrate.h"
#include "clicdata.h"  // CliGetContestNameFromID()
#include "clirate.h"   // CliGetKeyrateForProblem()
#include "probman.h"   // GetProblemPointerFromIndex()
#include "checkpt.h"   // 
#include "buffupd.h"   // BUFFERUPDATE_FETCH define
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

#include "iniread.h"
#include "pathwork.h"

// update contestdone and randomprefix .ini entries
static void __RefreshRandomPrefix( Client *client )
{                        
  if (client->stopiniio == 0 && client->nodiskbuffers == 0)
    {
    const char *OPTION_SECTION = "parameters";
    IniSection ini;
    s32 tmpconfig = ini.ReadIniFile( 
                    GetFullPathForFilename( client->inifilename ) );

    if (client->randomchanged)
      {
      ini.setrecord(OPTION_SECTION,"randomprefix",IniString((s32)(client->randomprefix)));
      ini.setrecord(OPTION_SECTION, "contestdone", IniString(client->contestdone[0]));
      ini.setrecord(OPTION_SECTION, "contestdone2", IniString(client->contestdone[1]));
      ini.WriteIniFile( GetFullPathForFilename( client->inifilename ) );
      client->randomchanged = 0;
      }
    else if (!tmpconfig)
      {  
      tmpconfig=ini.getkey(OPTION_SECTION, "randomprefix", "0")[0];
      if (tmpconfig) client->randomprefix = tmpconfig;
      tmpconfig=ini.getkey(OPTION_SECTION, "contestdone", "0")[0];
      client->contestdone[0] = (tmpconfig != 0);
      tmpconfig=ini.getkey(OPTION_SECTION, "contestdone2", "0")[0];
      client->contestdone[1]= (tmpconfig != 0);
      }
    }
  return;
}

// -----------------------------------------------------------------------

static unsigned int __IndividualProblemUnload( Problem *thisprob, 
                unsigned int prob_i, Client *client, int *load_needed, 
                      unsigned load_problem_count, unsigned int *contest )
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
                       unsigned load_problem_count, unsigned int *contest )
{                    
  FileEntry fileentry;
  RC5Result rc5result;
  unsigned int cont_i;
  unsigned int norm_key_count = 0;
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
    if (client->PutBufferRecord( &fileentry ) < 0)  // send it back...
      {
      //Log( "PutBuffer Error\n" ); //error already printed?
      }
    else
      {
      //---------------------
      // update the totals for this contest
      //---------------------

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
                   unsigned load_problem_count, unsigned int *contest )
{
  FileEntry fileentry;
  unsigned int cont_i;
  unsigned int norm_key_count = 0;
  unsigned int contest_count, contest_selected;
  unsigned int contest_preferred, contest_alternate;
  long longcount;
  int didupdate, didload, didrandom;
  s32 cputype;

  cputype           = client->cputype; /* needed for FILEENTRY_CPU macro */
  contest_preferred = (client->preferred_contest_id == 0)?(0):(1);
  contest_alternate = (contest_preferred == 0)?(1):(0);
  contest_count     = 2;
    
  #if ((CLIENT_CPU == CPU_X86) || (CLIENT_OS == OS_BEOS))
  if ( prob_i >= 4 )
    {
    #if (defined(MMX_BITSLICER) && defined(KWAN) && defined(MEGGS))
    if ( des_unit_func != des_unit_func_mmx ) // if not using mmx cores
    #endif
      {
      // Not the 1st or 2nd cracking thread...
      // Must do RC5.  DES x86 cores aren't multithread safe.
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

  didrandom = didload = 0;
  
  for (cont_i = 0;(didload == 0 && cont_i < contest_count); cont_i++)
    {
    contest_selected = ((cont_i)?(contest_alternate):(contest_preferred));
    didupdate = 0;

    while (client->contestdone[contest_selected] == 0)
      {
      longcount = client->GetBufferRecord( &fileentry, contest_selected, 0 );
      if (longcount >= 0)
        {
        didload = 1;
        break;
        }
      if (client->nonewblocks || client->offlinemode || didupdate != 0) 
        {                                     /* net update not permitted */
        didload = -1;
        break;
        }
      didupdate = 1;
      if ( ((unsigned long)(client->inthreshold[contest_selected])) < 
           ((unsigned long)(load_problem_count)))
        client->inthreshold[contest_selected] = load_problem_count;
      if (client->BufferUpdate(BUFFERUPDATE_FETCH,0) < 0)
        {
        didload = -1;
        break;
        }
      }
    }
    
  if (didload > 0) /* normal load from buffer succeeded */
    {
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
  else /* if (didload <= 0) *//* normal load from buffer failed */
    {
    if (client->contestdone[contest_preferred] && 
        client->contestdone[contest_alternate])
      didload = -2;
    else if (client->nonewblocks)
      didload = -3;
    else if (client->blockcount < 0) /* no random blocks permitted */
      didload = -1;
    else /* random blocks permitted */
      {
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
    
  if (didload <= 0)
    {
    *load_needed = 1;
    }
  else /*if (didload > 0)  *//* success */
    {
    if (load_problem_count <= COMBINEMSG_THRESHOLD)
      {
      const char *name;
      unsigned long iter = ntohl(fileentry.iterations.lo);
      unsigned int startpercent = (unsigned int) ( (double) 10000.0 *
           ( (double) (ntohl(fileentry.keysdone.lo)) / (double)(iter) ) );
      if (CliGetContestInfoBaseData( fileentry.contest, &name, NULL )!=0) 
        name = "???";
      norm_key_count = (unsigned int)__iter2norm( iter );
      
      Log("Loaded %s%s %u*2^28 block %08lX:%08lX%c(%u.%02u%% done)",
              name, ((didrandom)?(" random"):("")), norm_key_count,
              (unsigned long) ntohl( fileentry.key.hi ),
              (unsigned long) ntohl( fileentry.key.lo ),
              ((startpercent)?(' '):(0)),
              (unsigned)(startpercent/100), (unsigned)(startpercent%100) );
      }
    thisprob->LoadState( (ContestWork *) &fileentry , 
          (u32) (fileentry.contest), client->timeslice, client->cputype );
    *contest = (unsigned int)(fileentry.contest);
    } 

  return norm_key_count;
}    

// --------------------------------------------------------------------

unsigned int Client::LoadSaveProblems(unsigned int load_problem_count,int mode)
{
  static unsigned int done_initial_load = 0; 

  Problem *thisprob;
  int load_needed, changed_flag;

  int i,prob_step;  
  unsigned int norm_key_count, prob_i, prob_for, cont_i;
  unsigned int loaded_problems_count[2]={0,0};
  unsigned int loaded_normalized_key_count[2]={0,0};
  unsigned int saved_problems_count[2]={0,0};
  unsigned int saved_normalized_key_count[2]={0,0};
  char buffer[100+sizeof(in_buffer_file[0])];
  unsigned int total_problems_loaded, total_problems_saved;
  unsigned int getbuff_errs;

  getbuff_errs = 0;
  changed_flag = (done_initial_load == 0);
  total_problems_loaded = 0;
  total_problems_saved = 0;

  // ====================================================

  if (done_initial_load == 0)
    {
    prob_step = 1;

    i = InitializeProblemManager(load_problem_count);
    if (i<=0)
      return 0;
    load_problem_count = (unsigned int)i;
    }
  else
    {
    if (load_problem_count == 0)
      load_problem_count = done_initial_load;
    prob_step = -1;
    }


  for (prob_for=0; prob_for<load_problem_count; prob_for++)
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
    if (done_initial_load == 0) 
      {
      if ( mode != PROBFILL_UNLOADALL )  
        load_needed = 1;
      }
    else
      {
      if (mode == PROBFILL_UNLOADALL)
        {
        norm_key_count = __IndividualProblemUnload( thisprob, prob_i, this, 
          &load_needed, load_problem_count, &cont_i );
        load_needed = 0;
        changed_flag = (norm_key_count!=0);
        }
      else
        {
        norm_key_count = __IndividualProblemSave( thisprob, prob_i, this, 
          &load_needed, load_problem_count, &cont_i );
        if (load_needed)
          changed_flag = 1;
        }
      if (norm_key_count)
        {
        total_problems_saved++;
        saved_normalized_key_count[cont_i] += norm_key_count;
        saved_problems_count[cont_i]++;
        totalBlocksDone[cont_i]++; //client class
        }
      }

    //---------------------------------------

    if (load_needed)
      {
      if (blockcount > 0 && 
        ((u32)(totalBlocksDone[0]+totalBlocksDone[1]) >= (u32)(blockcount)))
        {
        ; //nothing
        }
      else
        {
        load_needed = 0;
        norm_key_count = __IndividualProblemLoad( thisprob, prob_i, this, 
                    &load_needed, load_problem_count, &cont_i );
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

  // ====================================

  for ( cont_i = 0; cont_i < 2; cont_i++) //once for each contest
    {
    if (loaded_problems_count[cont_i] || saved_problems_count[cont_i])
      {
      const char *cont_name = CliGetContestNameFromID(cont_i);

      if (loaded_problems_count[cont_i] && load_problem_count > COMBINEMSG_THRESHOLD )
        {
        sprintf(buffer, "Loaded %u %s block%s (%u*2^28 keys) from %s", 
              loaded_problems_count[cont_i], cont_name,
              ((loaded_problems_count[cont_i]==1)?(""):("s")),
              loaded_normalized_key_count[cont_i],
              (nodiskbuffers ? "(memory-in)" : in_buffer_file[cont_i]) );
        if (strlen(buffer) > 78)
          {
          buffer[75] = 0;
          strcat( buffer, "..." );
          }
        Log( "%s\n", buffer );
        }

      if (saved_problems_count[cont_i] && load_problem_count > COMBINEMSG_THRESHOLD)
        {
        sprintf(buffer, "Saved %u %s block%s (%u*2^28 keys) to %s", 
              saved_problems_count[cont_i], cont_name,
              ((saved_problems_count[cont_i]==1)?(""):("s")),
              saved_normalized_key_count[cont_i],
              (mode == PROBFILL_UNLOADALL)?
                (nodiskbuffers ? "(memory-in)" : in_buffer_file[cont_i]) :
                (nodiskbuffers ? "(memory-out)" : out_buffer_file[cont_i]) );
        if (strlen(buffer) > 78)
          {
          buffer[75] = 0;
          strcat( buffer, "..." );
          }
        Log( "%s\n", buffer );
        }

      // To suppress "odd" problem completion count summaries (and not be
      // quite so verbose) we only display summaries if the number of
      // completed problems is even divisible by the number of processors.
      // Requires a working GetNumberOfDetectedProcessors() [cpucheck.cpp]

      if ((i = GetNumberOfDetectedProcessors()) < 1)
        i = 1;
      if (load_problem_count > ((unsigned int)(i)))
        i = (int)load_problem_count;

      if ( ( totalBlocksDone[cont_i] > 0 ) && 
        (totalBlocksDone[cont_i] % ((unsigned int)(i)) ) == 0 )
        {
        Log( "Summary: %s\n", CliGetSummaryStringForContest(cont_i) );
        }

      unsigned long in, norm_in, out, norm_out;
      in = GetBufferCount( cont_i, 0, &norm_in );
      out = GetBufferCount( cont_i, 1, &norm_out );
      const char *msg = "%lu %s block%s (%lu*2^28 keys) %s in file %s\n";

      sprintf(buffer, msg, in, cont_name, in == 1 ? "" : "s",  norm_in,
            ((mode == PROBFILL_UNLOADALL)?(in==1?"is":"are"):
                                          (in==1?"remains":"remain")),
            (nodiskbuffers ? "(memory-in)" : in_buffer_file[cont_i]));
            
      if (( strlen(buffer) + sizeof("[Nov 09 20:10:10 GMT] ") ) > 77 )
        {
        char *p = strrchr( buffer, ' ' );
        if (p) *p = '\n';
        }
      Log( "%s", buffer );

      sprintf(buffer, msg, out, cont_name, out == 1 ? "" : "s", norm_out,
            out == 1 ? "is" : "are",
            (nodiskbuffers ? "(memory-out)" : out_buffer_file[cont_i]));
      
      if (( strlen(buffer) + sizeof("[Nov 09 20:10:10 GMT] ") ) > 77 )
        {
        char *p = strrchr( buffer, ' ' );
        if (p) *p = '\n';
        }
      Log( "%s", buffer );

      }
    } //for ( cont_i = 0; cont_i < 2; cont_i++)

 if (randomchanged)
   __RefreshRandomPrefix(this); //get/put an up-to-date prefix 

 if (mode == PROBFILL_UNLOADALL)
   {
   done_initial_load = 0;
   DeinitializeProblemManager();
   if (!nodiskbuffers)                        
     CheckpointAction( CHECKPOINT_CLOSE, 0 );
   else
     BufferUpdate(BUFFERUPDATE_FLUSH,0);
   return total_problems_saved;
   }

  done_initial_load = load_problem_count;

  if (blockcount > 0 && 
    ((u32)(totalBlocksDone[0]+totalBlocksDone[1]) >= (u32)(blockcount)))
    {
    for (prob_i = 0; prob_i < load_problem_count; prob_i++ )
      {
      thisprob = GetProblemPointerFromIndex( prob_i );
      if (thisprob != NULL)
        {
        if (thisprob->IsInitialized()) /* still have probs open */
          {
          blockcount = 1 + ((u32)(totalBlocksDone[0]+totalBlocksDone[1]));
          break;
          }
        }
      }    
    }

  if (mode == PROBFILL_GETBUFFERRS) 
    return getbuff_errs;
  if (mode == PROBFILL_ANYCHANGED)
    return changed_flag;

  return total_problems_loaded;
}  

// -----------------------------------------------------------------------

