/*
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
*/
const char *buffbase_cpp(void) {
return "@(#)$Id: buffbase.cpp,v 1.12.2.37 2000/09/24 12:55:47 cyp Exp $"; }

//#define PROFILE_DISK_HITS

#include "cputypes.h"
#include "cpucheck.h" //GetNumberOfDetectedProcessors()
#include "client.h"   //client struct and threshold functions
#include "baseincs.h" //basic #includes
#include "network.h"  //ntohl(), htonl()
#include "util.h"     //trace
#include "clievent.h" //event stuff
#include "clicdata.h" //GetContestNameFromID()
#include "logstuff.h" //Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "triggers.h" //[Check|Raise][Pause|Exit]RequestTrigger()
#include "pathwork.h" //GetFullPathForFilename() or dummy if DONT_USE_PATHWORK
#include "problem.h"  //Resultcode enum
#include "probfill.h"
#include "clitime.h"  // CliClock(), CliTimer()
#include "buffupd.h"  // BUFFERUPDATE_FETCH / BUFFERUPDATE_FLUSH
#include "buffbase.h" //ourselves

/* --------------------------------------------------------------------- */

const char *BufferGetDefaultFilename( unsigned int project, int is_out_type,
                                                       const char *basename )
{
  static char filename[128];
  const char *suffix = CliGetContestNameFromID( project );
  unsigned int len, n;

  filename[0] = '\0';
  if (*basename)
  {
    while (*basename && isspace(*basename))
      basename++;
    if (*basename)
    {
      strncpy( filename, basename, sizeof(filename));
      filename[sizeof(filename)-1]='\0';
      len = strlen( filename );
      while (len && isspace( filename[len-1] ) )
        filename[--len] = '\0';
    }
  }

  if (filename[0] == 0)
  {
    strcpy( filename, ((is_out_type) ?
       BUFFER_DEFAULT_OUT_BASENAME /* "buff-out" */:
       BUFFER_DEFAULT_IN_BASENAME  /* "buff-in" */  ) );
  }

  filename[sizeof(filename)-5]='\0';
  strcat( filename, EXTN_SEP );
  len = strlen( filename );
  for (n=0;suffix[n] && n<3;n++)
    filename[len++] = (char)tolower(suffix[n]);
  filename[len]='\0';
  return filename;
}


/* ===================================================================== */
/*                    START OF MEMBUFF PRIMITIVES                        */
/* ===================================================================== */

static int BufferPutMemRecord( struct membuffstruct *membuff,
    const WorkRecord* data, unsigned long *countP ) /* returns <0 on ioerr */
{
  unsigned long count = 0;
  int retcode = -1;
  WorkRecord *dest;

  if (membuff->count == 0)
  {
    unsigned int i;
    for (i = 0; i < (sizeof(membuff->buff)/sizeof(membuff->buff[0])); i++)
      membuff->buff[i]=NULL;
  }

  if (membuff->count < (sizeof(membuff->buff)/sizeof(membuff->buff[0])))
  {
    dest = (WorkRecord *)malloc(sizeof(WorkRecord));
    if (dest != NULL)
    {
      memcpy( (void *)dest, (const void *)data, sizeof( WorkRecord ));
      membuff->buff[membuff->count] = dest;
      membuff->count++;
      count = (unsigned long)membuff->count;
      retcode = 0;
    }
  }
  if (countP)
    *countP = count;
  return retcode;
}


/* --------------------------------------------------------------------- */

static int BufferGetMemRecord( struct membuffstruct *membuff,
                            WorkRecord* data, unsigned long *countP )
{
  /*  <0 on ioerr, >0 if norecs */
  unsigned long count = 0;
  int retcode = +1;
  WorkRecord *src = (WorkRecord *)0;

  if (membuff->count > 0)
  {
    retcode = -1;
    membuff->count--;
    src = membuff->buff[membuff->count];
    membuff->buff[membuff->count] = NULL;
    if (src != NULL)
    {
      retcode = 0;
      memcpy( (void *)data, (void *)src, sizeof( WorkRecord ));
      free( (void *)src );
      count = (unsigned long)membuff->count;
    }
  }
  if (countP)
    *countP = count;
  return retcode;
}


/* --------------------------------------------------------------------- */


static int BufferCountMemRecords( struct membuffstruct *membuff,
   unsigned int contest, unsigned long *packetcountP, unsigned long *normcountP )
{
  unsigned long rec, reccount = 0, packetcount = 0, normcount = 0;
  int retcode = -1;

  if (membuff != NULL)
  {
    retcode = 0;
    normcount = 0;
    reccount = membuff->count;
    if (normcountP)
    {
      packetcount = 0;
      for (rec = 0; rec < reccount; rec++)
      {
        if ( membuff->buff[rec] != NULL )
        {
          WorkRecord *workrec = membuff->buff[rec];
          if (((unsigned int)workrec->contest) == contest)
          {
            packetcount++;
            switch (contest)
            {
              case RC5:
              case DES:
              case CSC:
                normcount += (unsigned int)
                   __iter2norm( workrec->work.crypto.iterations.lo,
                                workrec->work.crypto.iterations.hi );
                break;
              case OGR:
                normcount++;
                break;
            }
          }
        }
      }
      reccount = packetcount;
    }
  }
  if (retcode == 0)
    {
    if (normcountP)
      *normcountP = normcount;
    if (packetcountP)
      *packetcountP = reccount;
    }
  return retcode;
}

/* ===================================================================== */
/*                      END OF MEMBUFF PRIMITIVES                        */
/* ===================================================================== */

static void __FixupRandomPrefix( const WorkRecord * data, Client *client )
{
  if (data->contest == RC5 && data->work.crypto.iterations.hi == 0 /* < 2^32 */
     && (data->work.crypto.iterations.lo) != 0x00100000UL /*!test*/
     && (data->work.crypto.iterations.lo) != 0x10000000UL /* !2**28 */)
  {
    u32 randomprefix = ((data->work.crypto.key.hi) & 0xFF000000L) >> 24;
    if (randomprefix != ((u32) client->randomprefix) )
    {
      // Has the high byte changed?  If so, then remember it.
      client->randomprefix = randomprefix;
      client->randomchanged=1;
    }
  }
  return;
}

/* --------------------------------------------------------------------- */

static int __CheckBuffLimits( Client *client )
{
  /* thresholds are managed in ClientGet[In|Out]Threshold() [client.cpp] */
  if (client->nodiskbuffers == 0 &&
      client->out_buffer_basename[0] && client->in_buffer_basename[0])
  {
    if (strcmp(client->in_buffer_basename, client->out_buffer_basename) == 0)
    {
      Log("ERROR!: in- and out- buffer prefixes are identical.\n");
      //RaiseExitRequestTrigger();
      return -1;
    }
  }
  return 0;
}

/* --------------------------------------------------------------------- */

long PutBufferRecord(Client *client,const WorkRecord *data)
{
  unsigned long workstate;
  unsigned int tmp_contest;
  membuffstruct *membuff;
  const char *filename;
  unsigned long count;
  int tmp_retcode, tmp_use_out_file;

  workstate = data->resultcode;
  tmp_contest = (unsigned int)data->contest;

  if (!(tmp_contest < CONTEST_COUNT))
  {
    //LogScreen("Discarded packet from unknown contest.\n");
  }
  else if (workstate != RESULT_WORKING && workstate != RESULT_FOUND &&
           workstate != RESULT_NOTHING)
  {
    LogScreen("Discarded packet with unrecognized workstate %ld.\n",workstate);
  }
  else if (__CheckBuffLimits( client ))
  {
    //nothing. message already printed
  }
  else
  {
    tmp_retcode = 0;
    tmp_use_out_file = 0;
    count = 0;
    __FixupRandomPrefix( data, client );

    if (workstate != RESULT_WORKING)
      tmp_use_out_file = 1;

    if (client->nodiskbuffers == 0)
    {
      filename = client->in_buffer_basename;
      if (tmp_use_out_file)
        filename = client->out_buffer_basename;
      filename = BufferGetDefaultFilename(tmp_contest, tmp_use_out_file,
                                                       filename );
      #ifdef PROFILE_DISK_HITS
      LogScreen("Diskhit: BufferPutFileRecord() called from PutBufferRecord()\n");
      #endif
      tmp_retcode = BufferPutFileRecord( filename, data, &count );
                    /* returns <0 on ioerr, >0 if norecs */
    }
    else
    {
      membuff = &(client->membufftable[tmp_contest].in);
      if (tmp_use_out_file)
        membuff = &(client->membufftable[tmp_contest].out);
      tmp_retcode = BufferPutMemRecord( membuff, data, &count );
               /* returns <0 on ioerr, >0 if norecs */
    }
    if (tmp_retcode == 0)
      return (long)count;
    if (tmp_retcode < 0)
    {
      Log("Buffer seek/write error. Block discarded.\n");
    }
  }
  return -1;
}


/* --------------------------------------------------------------------- */

long GetBufferRecord( Client *client, WorkRecord* data,
                      unsigned int contest, int use_out_file)
{
  unsigned long workstate;
  unsigned int tmp_contest;
  membuffstruct *membuff;
  const char *filename;
  unsigned long count;
  int retcode, tmp_retcode, tmp_use_out_file;

  if (__CheckBuffLimits( client ))
    return -1;

  do
  {
    if (!(contest < CONTEST_COUNT))
    {
      break; /* return -1 */
    }
    else if (client->nodiskbuffers == 0)
    {
      filename = client->in_buffer_basename;
      if (use_out_file)
        filename = client->out_buffer_basename;
      filename = BufferGetDefaultFilename(contest, use_out_file, filename );
                 /* returns <0 on ioerr, >0 if norecs */
      #ifdef PROFILE_DISK_HITS
      LogScreen("Diskhit: BufferGetFileRecord() <- GetBufferRecord()\n");
      #endif
      retcode = BufferGetFileRecord( filename, data, &count );
//LogScreen("b:%d\n", retcode);
    }
    else
    {
      membuff = &(client->membufftable[contest].in);
      if (use_out_file)
        membuff = &(client->membufftable[contest].out);
                 /* returns <0 on ioerr, >0 if norecs */
      retcode = BufferGetMemRecord( membuff, data, &count );
    }
    if ( retcode != 0)
    {
      if (retcode == -123 ) /* corrupted */
      {
        // corrupted - packet invalid, discard it.
        LogScreen( "Block integrity check failed. Block discarded.\n");
      }
      else if (retcode < 0) /* io error */
      {
        LogScreen("Buffer seek/read error. Partial packet discarded.\n");
        break; /* return -1; */
      }
      else if( retcode > 0 ) // no recs
        break;
    }
    else
    {
      workstate = ( data->resultcode );
      tmp_contest = (unsigned int)data->contest;
      if (!(tmp_contest < CONTEST_COUNT))
      {
        LogScreen("Discarded packet from unknown contest.\n");
      }
      else if (workstate != RESULT_WORKING && workstate != RESULT_FOUND &&
              workstate != RESULT_NOTHING)
      {
        LogScreen("Discarded packet with unrecognized workstate %ld.\n",workstate);
      }
      else if (tmp_contest != contest ||
               (use_out_file && workstate == RESULT_WORKING) ||
               (!use_out_file && workstate != RESULT_WORKING))
      {
        tmp_use_out_file = (workstate != RESULT_WORKING);
        tmp_retcode = 0;
//      LogScreen("Cross-saving packet of another type/contest.\n");
        if (client->nodiskbuffers == 0)
        {
//LogScreen("old cont:%d, type: %d, name %s\n", contest, use_out_file, filename );
          filename = client->in_buffer_basename;
          if (tmp_use_out_file)
            filename = client->out_buffer_basename;

          filename = BufferGetDefaultFilename(tmp_contest, tmp_use_out_file,
                                              filename );
//LogScreen("new cont:%d, type: %d, name %s\n", tmp_contest, tmp_use_out_file, filename );
          #ifdef PROFILE_DISK_HITS
          LogScreen("Diskhit: BufferPutFileRecord() <- GetBufferRecord()\n");
          #endif
          tmp_retcode = BufferPutFileRecord( filename, data, NULL );
        }
        else
        {
          membuff = &(client->membufftable[tmp_contest].in);
          if (tmp_use_out_file)
            membuff = &(client->membufftable[tmp_contest].out);
                 /* returns <0 on ioerr, >0 if norecs */
          tmp_retcode = BufferPutMemRecord( membuff, data, NULL );
        }
        if (tmp_retcode < 0)
        {
          Log("Buffer seek/write error. Block discarded.\n");
        }
      }
      else /* packet is ok */
      {
        __FixupRandomPrefix( data, client );
        return (long)count; // all is well, data is a valid entry.
      }
    }
  /* bad packet, but loop if we have more in buff */
  } while (count && !CheckExitRequestTriggerNoIO());

  return -1;
}

/* --------------------------------------------------------------------- */

int BufferAssertIsBufferFull( Client *client, unsigned int contest )
{
  int isfull = 0;
  if (contest < CONTEST_COUNT)
  {
    unsigned long reccount;
    if (client->nodiskbuffers == 0)
    {
      const char *filename = client->in_buffer_basename;
      filename = BufferGetDefaultFilename(contest, 0, filename );
      #ifdef PROFILE_DISK_HITS
      LogScreen("Diskhit: BufferCountFileRecords() <- BufferAssertIsBufferFull()\n");
      #endif
      if (BufferCountFileRecords( filename, contest, &reccount, NULL ) != 0)
        isfull = 1;
      else
        isfull = (reccount > 500); /* yes, hardcoded. */
      /* This function should be the only place where maxlimit is checked */
    }
    else
    {
      struct membuffstruct *membuff = &(client->membufftable[contest].in);
      if ( BufferCountMemRecords( membuff, contest, &reccount, NULL ) != 0)
        isfull = 1;
      else
        isfull=(reccount >= (sizeof(membuff->buff)/sizeof(membuff->buff[0])));
    }
  }
  return isfull;
}

/* --------------------------------------------------------------------- */

long GetBufferCount( Client *client, unsigned int contest,
                     int use_out_file, unsigned long *normcountP )
{
  membuffstruct *membuff;
  const char *filename;
  unsigned long reccount = 0;
  int retcode = -1;

  if (__CheckBuffLimits( client ) != 0)
  {
    //nothing, message already printed
  }
  else if (contest < CONTEST_COUNT)
  {
    if (client->nodiskbuffers == 0)
    {
      filename = client->in_buffer_basename;
      if (use_out_file)
        filename = client->out_buffer_basename;
      filename = BufferGetDefaultFilename(contest, use_out_file, filename );
      #ifdef PROFILE_DISK_HITS
      LogScreen("Diskhit: BufferCountFileRecords() <- GetBufferCount()\n");
      #endif
      retcode = BufferCountFileRecords( filename, contest, &reccount, normcountP );
    }
    else
    {
      membuff = &(client->membufftable[contest].in);
      if (use_out_file)
        membuff = &(client->membufftable[contest].out);
      retcode = BufferCountMemRecords( membuff, contest, &reccount, normcountP );
    }
  }
  if (retcode != 0 && normcountP)
    *normcountP = 0;
  if (retcode != 0)
    return -1;
  return (long)reccount;
}

/* --------------------------------------------------------------------- */

/* import records from source, return -1 if err, or number of recs imported. */
/* On success, source is truncated/deleted. Used by checkpt and --import */
long BufferImportFileRecords( Client *client, const char *source_file, int interactive)
{
  unsigned long remaining, lastremaining = 0;
  unsigned int recovered = 0;
  int errs = 0;
  WorkRecord data;

  #ifdef PROFILE_DISK_HITS
  LogScreen("Diskhit: access() <- GetImportFileRecords()\n");
  #endif
  if ( access( GetFullPathForFilename(source_file), 0 )!=0 )
  {
    if (interactive)
      LogScreen("Import error: '%s' not found.\n", source_file );
    return -1L;
  }

  for (;;)
  {
    #ifdef PROFILE_DISK_HITS
    LogScreen("Diskhit: BufferGetFileRecordNoOpt() <- BufferImportFileRecords()\n");
    #endif
    if (BufferGetFileRecordNoOpt( source_file, &data, &remaining ) != 0)
      break;  //returned <0 on ioerr/corruption, > 0 if norecs
    if (lastremaining != 0)
    {
      if (lastremaining <= remaining)
      {
        if (interactive)
          LogScreen("Import error: something bad happened.\n"
                    "The source file isn't getting smaller.\n");
        errs = 1;
        recovered = 0;
        break;
      }
    }
    lastremaining = remaining;
    if ( PutBufferRecord( client, &data ) > 0 )
    {
      recovered++;
    }
  }
  if (recovered > 0)
  {
    #ifdef PROFILE_DISK_HITS
    LogScreen("Diskhit: BufferZapFileRecords() <- BufferImportFileRecords()\n");
    #endif
    BufferZapFileRecords( source_file );
  }
  if (recovered > 0 && interactive)
    LogScreen("Import::%ld records successfully imported.\n", recovered);
  else if (errs == 0 && recovered == 0 && interactive)
    LogScreen("Import::No buffer records could be imported.\n");
  return (long)recovered;
}

/* --------------------------------------------------------------------- */

// determine whether fetch should proceed/continue or not.
// ******** MUST BE CALLED FOR EACH PASS THROUGH A FETCH LOOP *************

unsigned long BufferReComputeWorkUnitsToFetch(Client *client, unsigned int contest)
{
  unsigned long lefttotrans;

  lefttotrans = 0;
  if (!BufferAssertIsBufferFull( client, contest ))
  {
    int packets;
    if ((packets = GetBufferCount( client, contest, 0, &lefttotrans )) < 0)
      lefttotrans = 0;
    else
    {
      /* get threshold in *work units* */
      int threshold = ClientGetInThreshold( client, contest, 1 /* force */ );
      if (lefttotrans < ((unsigned long)threshold))
        lefttotrans = threshold - lefttotrans;
      else /* determine if we have at least one packet per cruncher */
      {    /* (remember that GetInThreshold() returns *work units*) */
        int numcrunchers = client->numcpu;
        if (numcrunchers < 0)
          numcrunchers = GetNumberOfDetectedProcessors();
        if (numcrunchers < 1) /* force non-threaded or getnumprocs() failed */
          numcrunchers = 1;
        if (packets < numcrunchers)  /* Have at least one packet per cruncher? */
          lefttotrans = numcrunchers - packets;
        else
          lefttotrans = 0;
      }
    }
  }
  return lefttotrans;
}

/* ===================================================================== */
/* Remote buffer fetch/flush                                             */

/* remote buffer loop detection: Fetch/FlushFile operation stop after
   MAX_REMOTE_BUFFER_WARNINGS (buffer size doesn't in/decrease) */

#define MAX_REMOTE_BUFFER_WARNINGS 3

/* --------------------------------------------------------------------- */

long BufferFetchFile( Client *client, const char *loaderflags_map )
{
  unsigned long donetrans_total_wu = 0, donetrans_total_pkts = 0;
  char basename[sizeof(client->remote_update_dir)  +
                sizeof(client->in_buffer_basename) + 10 ];
  unsigned int contest;
  int failed = 0;

  // quit if we aren't configured to do remote buffer updates
  if (client->noupdatefromfile || client->remote_update_dir[0] == '\0')
    return 0;

  // Generate the full path of the remote buffer.
  if (client->in_buffer_basename[0] == '\0')
  {
    strcpy( basename,
            GetFullPathForFilenameAndDir( BUFFER_DEFAULT_IN_BASENAME,
                                          client->remote_update_dir ));
  }
  else
  {
     strcpy( basename,
             GetFullPathForFilenameAndDir(
               &(client->in_buffer_basename[
                        GetFilenameBaseOffset(client->in_buffer_basename)]),
             client->remote_update_dir ) );

  }
  basename[sizeof(basename)-1] = '\0';
//printf("basename: %s\n",basename);

  for (contest = 0; !failed && contest < CONTEST_COUNT; contest++)
  {
    const char *contname;
    char remote_file[sizeof(basename)+10];
    unsigned long totrans_wu, donetrans_pkts = 0, donetrans_wu = 0;
    long inbuffer_count = 0, inbuffer_count_last = -1, inbuffer_count_warnings = 0;

    if (CheckExitRequestTriggerNoIO())
      break;

    if (loaderflags_map[contest] != 0)
      continue; /* Skip to next contest if this one is closed or disabled. */
    contname = CliGetContestNameFromID(contest);
    if (!contname)
      continue;

    strncpy( remote_file, BufferGetDefaultFilename(contest,1,basename), sizeof(remote_file));
    remote_file[sizeof(remote_file)-1] = '\0';

    totrans_wu = 1;
    while (totrans_wu > 0 )
    {
      if (CheckExitRequestTriggerNoIO())
        break;

      /* update the count to fetch - NEVER do this outside the loop */
      totrans_wu = BufferReComputeWorkUnitsToFetch( client, contest);
      if (totrans_wu > 0)
      {
        WorkRecord wrdata;
        unsigned long remaining;

        // Retrieve a packet from the remote buffer.
        #ifdef PROFILE_DISK_HITS
        LogScreen("Diskhit: BufferGetFileRecordNoOpt() <- BufferFetchFile()\n");
        #endif
        if ( BufferGetFileRecordNoOpt( remote_file, &wrdata, &remaining ) != 0 )
        {
          //totrans_wu = 0; /* move to next contest on file error */
          break; /* move to next contest - error msg has been printed */
          /* if file doesn't exist, no error will be printed. move on as well. */
        }
        else if (((unsigned int)wrdata.contest) != contest)
        {
          Log("Remote buffer %s\ncontains non-%s packets. Stopped fetch for %s.\n",
                       remote_file, contname, contname );
          #ifdef PROFILE_DISK_HITS
          LogScreen("Diskhit: BufferPutFileRecord() <- BufferFetchFile()\n");
          #endif
          BufferPutFileRecord( remote_file, &wrdata, NULL );
          //totrans_wu = 0; /* move to next contest on file error */
          break; /* move to next contest - error msg has been printed */
        }
        else if ((inbuffer_count = PutBufferRecord( client, &wrdata )) < 0) /* can't save here? */
        {                             /* then put it back there */
          #ifdef PROFILE_DISK_HITS
          LogScreen("Diskhit: BufferPutFileRecord() <- BufferFetchFile()\n");
          #endif
          BufferPutFileRecord( remote_file, &wrdata, NULL );
          failed = -1; /* stop further local buffer I/O */
          //totrans_wu = 0; /* move to next contest on file error */
          break; /* move to next contest - error msg has been printed */
        }
        else
        {
          int workunits = 0;

          if (donetrans_pkts > 0 && inbuffer_count_last >= inbuffer_count)
          {
            // check, whether in-buffer gets larger
            // multiple clients fetching blocks from the in-buffer while fetching
            // new work from the remote in-buffer aborts action, too
            if ( inbuffer_count_warnings++ < MAX_REMOTE_BUFFER_WARNINGS)
            {
              //LogScreen("\nFetchFile error: The dest file isn't getting bigger.\n");
            }
            else
            {
              LogScreen("\n");
              Log("FetchFile error: The dest file isn't getting bigger.\n"
                  "Check for a loop in your (remote) buffer settings!\n");
              break;
            }
          }
          inbuffer_count_last = inbuffer_count;

          /* normalize to workunits for display */
          switch (contest)
          {
            case RC5:
            case DES:
            case CSC:
              workunits =  __iter2norm(wrdata.work.crypto.iterations.lo,
                                     wrdata.work.crypto.iterations.hi);
              if (workunits < 1)
                workunits = 1;
              break;
            case OGR:
              workunits = 1;
              break;
          }

          donetrans_pkts++;
          donetrans_wu += workunits;

          if (donetrans_pkts == 1) /* first pass? */
            ClientEventSyncPost( CLIEVENT_BUFFER_FETCHBEGIN, 0 );

          if (remaining == 0) /* if no more available, then we've done 100% */
            totrans_wu = 0;
          if (((unsigned long)workunits) > totrans_wu) /* done >= 100%? */
            totrans_wu = workunits; /* then the # we wanted is the # we got */
          totrans_wu -= workunits; /* how many to get for the next while() */
        }  /* if BufferGetRecord/BufferPutRecord */
      }  /* if if (totrans_wu > 0) */

      if (donetrans_wu)
      {
        unsigned long totrans = (donetrans_wu + (unsigned long)(totrans_wu));
        unsigned int percent = ((donetrans_wu*10000)/totrans);

        LogScreen( "\rRetrieved %s work unit %lu of %lu (%u.%02u%% transferred) ",
                    contname, donetrans_wu, totrans, percent/100, percent%100 );
      }
    }  /* while ( totrans_wu > 0  ) */

    if (donetrans_wu)
    {
      donetrans_total_wu   += donetrans_wu;
      donetrans_total_pkts += donetrans_pkts;

      LogScreen("\n");
      LogTo(LOGTO_FILE|LOGTO_MAIL,
            "Retrieved %lu %s work units (%lu packets) from file.\n",
                         donetrans_wu, contname, donetrans_pkts );
    }
  } /* for (contest = 0; contest < CONTEST_COUNT; contest++) */

  if (failed)
    return -((long)(donetrans_total_pkts+1));
  return donetrans_total_pkts;
}

/* --------------------------------------------------------------------- */

long BufferFlushFile( Client *client, const char *loadermap_flags )
{
  long totaltrans_pkts = 0, totaltrans_wu = 0;
  char basename[sizeof(client->remote_update_dir)  +
                sizeof(client->out_buffer_basename) + 10 ];
  unsigned int contest;
  int failed = 0;

  // If we aren't configured to do remote buffer updating, then quit.
  if (client->noupdatefromfile || client->remote_update_dir[0] == '\0')
    return 0;

  // Generate the full path of the remote buffer.
  if (client->out_buffer_basename[0] == '\0')
  {
    strcpy( basename,
            GetFullPathForFilenameAndDir( BUFFER_DEFAULT_OUT_BASENAME,
                                          client->remote_update_dir ));
  }
  else
  {
    strcpy( basename,
           GetFullPathForFilenameAndDir(
             &(client->out_buffer_basename[
                        GetFilenameBaseOffset(client->out_buffer_basename)]),
             client->remote_update_dir ) );
  }
  basename[sizeof(basename)-1] = '\0';

  for (contest = 0; !failed && contest < CONTEST_COUNT; contest++)
  {
    const char *contname;
    char remote_file[sizeof(basename)+10];
    unsigned long projtrans_wu = 0, projtrans_pkts = 0;
    WorkRecord wrdata;
    long totrans_pkts, totrans_pkts_last, totrans_pkts_warnings;

    if (CheckExitRequestTriggerNoIO())
      break;
    if (loadermap_flags[contest] != 0) /* contest is closed or disabled */
      continue; /* proceed to next contest */
    contname = CliGetContestNameFromID(contest);
    if (!contname)
      continue;

    strncpy( remote_file, BufferGetDefaultFilename(contest,1,basename), sizeof(remote_file));
    remote_file[sizeof(remote_file)-1] = '\0';

    totrans_pkts = 1;
    totrans_pkts_last = -1;
    totrans_pkts_warnings = 0;

    while (totrans_pkts > 0)
    {
      long workunits;

      if (CheckExitRequestTriggerNoIO())
        break;

      totrans_pkts = GetBufferRecord( client, &wrdata, contest, 1 );
      if (totrans_pkts < 0)
        break;

      #ifdef PROFILE_DISK_HITS
      LogScreen("Diskhit: BufferPutFileRecord() <- BufferFlushFile()\n");
      #endif
      if ( BufferPutFileRecord( remote_file, &wrdata, NULL ) != 0 )
      {
        PutBufferRecord( client, &wrdata );
        failed = -1;
        break;
      }

      if (projtrans_pkts > 0 && totrans_pkts_last <= totrans_pkts)
      {
        // check, whether out-buffer gets smaller
        // multiple clients flushing blocks to the out-buffer while flushing
        // the out-buffer to the remote out-buffer aborts action, too
        if ( totrans_pkts_warnings++ < MAX_REMOTE_BUFFER_WARNINGS)
        {
          //LogScreen("\nFlushFile warning: The source file isn't getting smaller.\n");
        }
        else
        {
          LogScreen("\n");
          Log("FlushFile error: The source file isn't getting smaller.\n"
              "Check for a loop in your (remote) buffer settings!\n");
          break;
        }
      }
      totrans_pkts_last = totrans_pkts;

      workunits = 1;
      switch (contest)
      {
        case RC5:
        case DES:
        case CSC:
          workunits =  __iter2norm(wrdata.work.crypto.iterations.lo,
                                   wrdata.work.crypto.iterations.hi);
          if (workunits < 1)
            workunits = 1;
          break;
        case OGR:
          workunits = 1;
          break;
      }

      projtrans_pkts++;
      projtrans_wu += workunits;

      if (totrans_pkts == 0) /* no more to do, can show count in work units */
      {
        LogScreen( "\rSent %s work unit %lu of %lu (100.00%% transferred)   ",
            contname, projtrans_wu, projtrans_wu  );
      }
      else /* count in packets */
      {
        unsigned long totrans = (projtrans_pkts + (unsigned long)(totrans_pkts));
        unsigned int percent = ((projtrans_pkts*10000)/totrans);
        LogScreen( "\rSent %s packet %lu of %lu (%u.%02u%% transferred)     ",
          contname, projtrans_pkts, totrans,  percent/100, percent%100 );
      }
    } /* while (totrans_pkts >=0 ) */

    if (projtrans_pkts != 0) /* transferred anything? */
    {
      totaltrans_pkts += projtrans_pkts;
      totaltrans_wu += projtrans_wu;

      LogScreen("\n");
      LogTo(LOGTO_FILE|LOGTO_MAIL,
            "Transferred %lu %s work unit%s (%lu packet%s) to file.\n",
                totaltrans_wu, contname, ((totaltrans_wu==1)?(""):("s")),
                totaltrans_pkts, ((totaltrans_pkts==1)?(""):("s")) );
    }
  } /* for (contest = 0; contest < CONTEST_COUNT; contest++) */

  if (failed)
    return -((long)(totaltrans_pkts+1));
  return totaltrans_pkts;
}

/* --------------------------------------------------------------------- */

/* BufferCheckIfUpdateNeeded() is called from BufferUpdate and also 
   from various connect-often test points.
*/   
int BufferCheckIfUpdateNeeded(Client *client, int contestid, int buffupd_flags)
{
  #define PROJECTFLAGS_CLOSED_TTL 0 //(7*24*60*60) /* 7 days */
  /* set PROJECTFLAGS_CLOSED_TTL to zero if closed flags are never to expire 
     (only a buffer update will set/clear them)
  */
  int check_flush, check_fetch, need_flush, need_fetch;
  int closed_expired, pos, cont_start, cont_count;
  int ignore_closed_flags, fill_even_if_not_totally_empty, either_or;

  check_flush = check_fetch = 0;
  if ((buffupd_flags & BUFFERUPDATE_FETCH)!=0)
    check_fetch = 1;
  if ((buffupd_flags & BUFFERUPDATE_FLUSH)!=0)
    check_flush = 1;
  if (!check_fetch && !check_flush)
    return 0;    

  /* normally an fetch_needed check will be true only if the in-buff is
     completely empty. With BUFFUPDCHECK_TOPOFF, the fetch_needed check
     will be also be true if below threshold. Used with connect_often etc.
  */   
  fill_even_if_not_totally_empty = 0;
  if ((buffupd_flags & BUFFUPDCHECK_TOPOFF)!=0)
    fill_even_if_not_totally_empty = 1;
  /* if result precision is not necessary, ie the caller only wants to
     know if _either_ fetch _or_ flush is necessary, then we can do
     some optimization. The function is then faster, but the result is
     degraded to a simple true/false.
  */
  either_or = 0;
  if (check_fetch && check_flush && (buffupd_flags & BUFFUPDCHECK_EITHER)!=0)
    either_or = 1; /* either criterion filled fulfills both criteria */
                   
  ignore_closed_flags = 0;
  cont_start = 0; cont_count = CONTEST_COUNT;
  if (contestid >= 0 && contestid < CONTEST_COUNT)
  {
    /* on some systems (linux), checking for empty-contest expiry is very 
       slow, so this can be bypassed if doing a single contest check 
       (the caller already knows its open)
    */  
    ignore_closed_flags = 1;
    cont_start = contestid;
    cont_count = contestid+1;
  }    

  need_flush = need_fetch = 0; closed_expired = -1;
  for (pos = cont_start; pos < cont_count; pos++)
  {
    unsigned int cont_i = (unsigned int)(client->loadorder_map[pos]);
    if (cont_i < CONTEST_COUNT) /* not disabled */
    {
      int isclosed = 0;
      if (!ignore_closed_flags && 
          (client->project_flags[cont_i] & PROJECTFLAGS_CLOSED) != 0)
      {
        /* this next bit is not a good candidate for a code hoist */
        if (closed_expired < 0) /* undetermined */
        {
          struct timeval tv;
          closed_expired = 0;
          if (client->last_buffupd_time == 0)
          {
            closed_expired = 1;
          }  
          else if ((client->scheduledupdatetime != 0 && 
            ((unsigned long)CliTimer(0)->tv_sec) >= 
     	      ((unsigned long)client->scheduledupdatetime)))
          {  
            closed_expired = 1;
          }  
          #if defined(PROJECTFLAGS_CLOSED_TTL) && \
              (PROJECTFLAGS_CLOSED_TTL > 0) /* any expiry time at all? */
          else if (CliClock(&tv)==0)
          {
            if (((unsigned long)tv.tv_sec) > 
              (unsigned long)(client->last_buffupd_time+PROJECTFLAGS_CLOSED_TTL)) 
            {  
              closed_expired = 1;
            }  
          }      
          #endif
        } /* if (closed_expired < 0) (undetermined) */
        if (!closed_expired)
          isclosed = 1;
      }	  
      if (!isclosed)
      {
        if (check_flush && !need_flush)
        {
          if (GetBufferCount( client, cont_i, 1 /* use_out_file */, NULL ) > 0)
          {
            need_flush = 1;
            if (either_or)    /* either criterion satisfied ... */
              need_fetch = 1; /* ... fullfills both criteria */
          }    
        }      
        if (check_fetch && !need_fetch)
        {
          if (!BufferAssertIsBufferFull(client,cont_i))
          {
            unsigned long wucount;
            if (GetBufferCount( client, cont_i, 0, &wucount ) <= 0)
              need_fetch = 1;
            else if (fill_even_if_not_totally_empty &&
                 wucount < (unsigned int)ClientGetInThreshold( client, cont_i, 0 ))
            {     
              need_fetch = 1;
            }  
          }      
          if (need_fetch && either_or) /* either criterion satisfied ... */
            need_flush = 1;            /* ... fulfills both criteria */
        }
        if (need_flush && need_fetch)
          break;
      } /* if (!isclosed) */
    } /* if (i < CONTEST_COUNT) */ /* not disabled */
  } /* for (;cont_i < cont_count; cont_i++) */

  buffupd_flags = 0;
  if (need_fetch)
    buffupd_flags |= BUFFERUPDATE_FETCH;
  if (need_flush)
    buffupd_flags |= BUFFERUPDATE_FLUSH;
  
  return buffupd_flags;
}  
