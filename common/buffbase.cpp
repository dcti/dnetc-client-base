/*
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
*/
const char *buffbase_cpp(void) {
return "@(#)$Id: buffbase.cpp,v 1.12.2.46 2000/11/11 13:15:41 cyp Exp $"; }

//#define TRACE
//#define PROFILE_DISK_HITS

#include "cputypes.h"
#include "cpucheck.h" //GetNumberOfDetectedProcessors()
#include "client.h"   //client struct and threshold functions
#include "baseincs.h" //basic #includes
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

int BufferGetRecordInfo( const WorkRecord * data, 
                         unsigned int *contest,
                         unsigned int *swucount )
{
  int rc = -1;
  if (data)
  {
    unsigned int cont_i = (unsigned int)data->contest;
    rc = ProblemGetSWUCount( &(data->work), ((unsigned int)data->resultcode),
                             cont_i, swucount );
    if (rc >= 0 && contest)
      *contest = cont_i;
  }
  return rc;
}

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
            unsigned int swucount;
            if (BufferGetRecordInfo( workrec, 0, &swucount ) != -1)
            {
              packetcount++;
              normcount += swucount;
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
  int break_pending = CheckExitRequestTriggerNoIO();
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
        return (long)count; // all is well, data is a valid entry.
      }
    }
    /* bad packet, but loop if we have more in buff */
    if (!break_pending)
    {
      if (CheckExitRequestTriggerNoIO())
        break;
    }
  } while (count);

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
  LogScreen("Diskhit: access() <- BufferImportFileRecords()\n");
  #endif
  if ( access( GetFullPathForFilename(source_file), 0 )!=0 )
  {
    if (interactive)
      Log("Import error: '%s' not found.\n", source_file );
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
          Log("Import error: something bad happened.\n"
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
    Log("Import: %ld records successfully imported.\n", recovered);
  else if (errs == 0 && recovered == 0 && interactive)
    Log("Import: No buffer records could be imported.\n");
  return (long)recovered;
}

/* --------------------------------------------------------------------- */

// determine whether fetch should proceed/continue or not.
// ******** MUST BE CALLED FOR EACH PASS THROUGH A FETCH LOOP *************

unsigned long BufferReComputeUnitsToFetch(Client *client, unsigned int contest)
{
  unsigned long swucount = 0;
  if (!BufferAssertIsBufferFull( client, contest ))
  {
    int packets = GetBufferCount( client, contest, 0, &swucount );
    if (packets < 0)
    {
      swucount = 0;
    }
    else
    {
      /* get threshold in *stats units* */
      unsigned int swuthresh = ClientGetInThreshold( client, contest, 1 /* force */ );
      if (swucount == 0)
        swucount = packets*100; /* >= 1.00 stats units per packet */
      if (swucount < swuthresh)
        swucount = swuthresh - swucount;
      else 
      {
        /* ok, we've determined that the thresholds are satisfied, now */
        /* determine if we have at least one packet per cruncher */
        /* (remember that GetInThreshold() returns *stats units*) */
        int numcrunchers = ProblemCountLoaded(contest);
        if (numcrunchers < 1) /* haven't spun any up yet */
        {
          numcrunchers = client->numcpu;
          if (numcrunchers < 0)
            numcrunchers = GetNumberOfDetectedProcessors();
          if (numcrunchers < 1) /* force non-threaded or getnumprocs() failed */
            numcrunchers = 1;
        }
        swucount = 0; /* assume we don't need more */
        if (packets < numcrunchers)  /* Have at least one packet per cruncher? */
          swucount = 100 * (numcrunchers - packets); /* >= 1.00 sus/packet */
      }
    }
  }
  return swucount;
}

/* ===================================================================== */
/* Remote buffer fetch/flush                                             */

static int __get_remote_filename(Client *client, int project, int use_out,
                                 char *buffer, unsigned int buflen)
{
  char suffix[10];
  const char *localname, *defaultname;
  unsigned int len;

  if (client->noupdatefromfile || client->remote_update_dir[0] == '\0')
    return 0;

  localname = client->in_buffer_basename;
  defaultname = BUFFER_DEFAULT_IN_BASENAME;
  if (use_out)
  {
    localname = client->out_buffer_basename;
    defaultname = BUFFER_DEFAULT_OUT_BASENAME;
  }

  if (*localname)
    localname += GetFilenameBaseOffset(localname);
  else
    localname = defaultname;

  strncpy( buffer, GetFullPathForFilenameAndDir( localname,
                   client->remote_update_dir ), buflen );
  buffer[buflen-1] = '\0';

  len = strlen(buffer);
  strcat( strcpy( suffix, EXTN_SEP ), CliGetContestNameFromID( project ));
  strncpy( &buffer[len], suffix, buflen-len);
  buffer[buflen-1] = '\0';

  return strlen(buffer);
}

/* remote buffer loop detection: Fetch/FlushFile operation stop after
   MAX_REMOTE_BUFFER_WARNINGS (buffer size doesn't in/decrease) */

#define MAX_REMOTE_BUFFER_WARNINGS 3

/* --------------------------------------------------------------------- */

long BufferFetchFile( Client *client, int break_pending, 
                      const char *loaderflags_map )
{
  unsigned long totaltrans_wu = 0, totaltrans_pkts = 0;
  unsigned int contest; int failed = 0;

  for (contest = 0; !failed && contest < CONTEST_COUNT; contest++)
  {
    const char *contname;
    char remote_file[sizeof(client->remote_update_dir)  +
                     sizeof(client->in_buffer_basename) + 10 ];
    unsigned long lefttotrans_su, projtrans_pkts = 0, projtrans_su = 0;
    long inbuffer_count = 0, inbuffer_count_last = -1;
    long inbuffer_count_warnings = 0;

    if (!break_pending && CheckExitRequestTriggerNoIO())
      break;
    if (loaderflags_map[contest] != 0)
      continue; /* Skip to next contest if this one is closed or disabled. */
    contname = CliGetContestNameFromID(contest);
    if (!contname)
      continue;
    if (!__get_remote_filename(client, contest, 0, remote_file, sizeof(remote_file)))
      continue;    

    lefttotrans_su = 1;
    while (lefttotrans_su > 0 )
    {
      if (!break_pending && CheckExitRequestTriggerNoIO())
        break;

      /* update the count to fetch - NEVER do this outside the loop */
      lefttotrans_su = BufferReComputeUnitsToFetch( client, contest);

      if (lefttotrans_su > 0)
      {
        WorkRecord wrdata;
        unsigned long remaining;

        // Retrieve a packet from the remote buffer.
        #ifdef PROFILE_DISK_HITS
        LogScreen("Diskhit: BufferGetFileRecordNoOpt() <- BufferFetchFile()\n");
        #endif
        if ( BufferGetFileRecordNoOpt( remote_file, &wrdata, &remaining ) != 0 )
        {
          //lefttotrans_su = 0; /* move to next contest on file error */
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
          //lefttotrans_su = 0; /* move to next contest on file error */
          break; /* move to next contest - error msg has been printed */
        }
        else if ((inbuffer_count = PutBufferRecord( client, &wrdata )) < 0) /* can't save here? */
        {                             /* then put it back there */
          #ifdef PROFILE_DISK_HITS
          LogScreen("Diskhit: BufferPutFileRecord() <- BufferFetchFile()\n");
          #endif
          BufferPutFileRecord( remote_file, &wrdata, NULL );
          failed = -1; /* stop further local buffer I/O */
          //lefttotrans_su = 0; /* move to next contest on file error */
          break; /* move to next contest - error msg has been printed */
        }
        else
        {
          unsigned int swucount = 0;

          if (projtrans_pkts > 0 && inbuffer_count_last >= inbuffer_count)
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

          BufferGetRecordInfo( &wrdata, 0, &swucount );
          projtrans_pkts++;
          projtrans_su += swucount;

          if (projtrans_pkts == 1) /* first pass? */
            ClientEventSyncPost( CLIEVENT_BUFFER_FETCHBEGIN, 0 );

          if (swucount == 0) /* project doesn't support statsunits on unfinished work */
            swucount = 100; /* 1.00 stats units per packet */    
          if (remaining == 0) /* if no more available, then we've done 100% */
            lefttotrans_su = 0;
          else if (swucount > lefttotrans_su) /* done >= 100%? */
            lefttotrans_su = 0; /* then the # we wanted is the # we got */
          else
            lefttotrans_su -= swucount; /* how many to get for the next while() */
        }  /* if BufferGetRecord/BufferPutRecord */
      }  /* if if (lefttotrans_su > 0) */

      if (projtrans_pkts)
      {
        unsigned long percent, totrans = 0, donesofar = 0;
        const char *unittype = "";

        if (projtrans_su) /* project can do stats units for incomplete work */
        {
          donesofar = projtrans_su/100; /* discard fraction */
          totrans = (projtrans_su + lefttotrans_su)/100; /* discard fractions */
          unittype = "stats unit";
        }
        else /* project can't do stats units on incomplete work */
        {
          donesofar = projtrans_pkts;
          totrans = projtrans_pkts + (lefttotrans_su/100); /*1.00 sus per pkt*/
          unittype = "packet";
        }
        percent = ((donesofar*10000)/totrans);

        LogScreen( "\r%s: Retrieved %s %lu of %lu (%u.%02u%%) ",
            contname, unittype, donesofar, totrans, percent/100, percent%100 );
      }
    }  /* while ( lefttotrans_su > 0  ) */

    if (projtrans_pkts)
    {
      char scratch[64];
      totaltrans_wu   += projtrans_su;
      totaltrans_pkts += projtrans_pkts;

      scratch[0] = '\0';
      if (projtrans_su)
      {
        sprintf(scratch, "(%lu.%02lu stats units) ",
                         projtrans_su/100,projtrans_su%100);
      }
      LogScreen("\n");
      LogTo(LOGTO_FILE|LOGTO_MAIL,
            "Retrieved %lu %s packet%s %sfrom file.\n",
                  projtrans_pkts, contname, 
                  ((projtrans_pkts==1)?(""):("s")), scratch );
    }
  } /* for (contest = 0; contest < CONTEST_COUNT; contest++) */

  if (failed)
    return -((long)(totaltrans_pkts+1));
  return totaltrans_pkts;
}

/* --------------------------------------------------------------------- */

long BufferFlushFile( Client *client, int break_pending,
                      const char *loadermap_flags )
{
  long totaltrans_pkts = 0;
  unsigned int contest;
  int failed = 0;

  for (contest = 0; !failed && contest < CONTEST_COUNT; contest++)
  {
    const char *contname;
    char remote_file[sizeof(client->remote_update_dir)  +
                     sizeof(client->out_buffer_basename) + 10 ];
    unsigned long projtrans_su = 0, projtrans_pkts = 0;
    WorkRecord wrdata;
    long totrans_pkts, totrans_pkts_last, totrans_pkts_warnings;

    if (!break_pending && CheckExitRequestTriggerNoIO())
      break;
    if (loadermap_flags[contest] != 0) /* contest is closed or disabled */
      continue; /* proceed to next contest */
    contname = CliGetContestNameFromID(contest);
    if (!contname)
      continue;
    if (!__get_remote_filename(client, contest, 1, remote_file, sizeof(remote_file)))
      continue;    

    totrans_pkts = 1;
    totrans_pkts_last = -1;
    totrans_pkts_warnings = 0;

    while (totrans_pkts > 0)
    {
      unsigned int swucount;

      if (!break_pending && CheckExitRequestTriggerNoIO())
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

      swucount = 0;
      BufferGetRecordInfo( &wrdata, 0, &swucount );

      projtrans_pkts++;
      projtrans_su += swucount;

      if (totrans_pkts != 0) /* only print (packets) if more to do */
      {                      /* else print (stats units) outside loop */
        unsigned long totrans = (projtrans_pkts + (unsigned long)(totrans_pkts));
        unsigned int percent = ((projtrans_pkts*10000)/totrans);
        LogScreen( "\r%s: Sent packet %lu of %lu (%u.%02u%% transferred) ",
          contname, projtrans_pkts, totrans,  percent/100, percent%100 );
      }
    } /* while (totrans_pkts >=0 ) */

    if (projtrans_pkts != 0) /* transferred anything? */
    {
      totaltrans_pkts += projtrans_pkts;

      LogScreen("\r");
      Log("%s: Transferred %lu packet%s (%lu.%02lu stats units) to file.\n",
            contname, projtrans_pkts,
            ((projtrans_pkts==1)?(""):("s")),
            projtrans_su/100, projtrans_su%100 );
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

  TRACE_OUT((+1,"BufferCheckIfUpdateNeeded(client, contestid=%d, check_fetch=%d/check_flush=%d)\n", contestid, check_fetch, check_flush));

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

  TRACE_OUT((0,"ignore_closed_flags=%d\n", ignore_closed_flags));

  need_flush = need_fetch = 0; 
  closed_expired = -1;
  for (pos = cont_start; pos < cont_count; pos++)
  {
    unsigned int cont_i = (unsigned int)(client->loadorder_map[pos]);
    if (cont_i < CONTEST_COUNT && /* not user disabled */
       IsProblemLoadPermitted(-1, cont_i)) /* core not disabled */
    {
      int isclosed = 0;
      char proj_flags = client->project_flags[cont_i];
      TRACE_OUT((0,"proj_flags[cont_i=%d] = 0x%x\n", cont_i, proj_flags));
      if (!ignore_closed_flags && 
          (proj_flags & (PROJECTFLAGS_CLOSED|PROJECTFLAGS_SUSPENDED)) != 0)
      {
        /* this next bit is not a good candidate for a code hoist */
        if (closed_expired < 0) /* undetermined */
        {
          struct timeval tv;
	  tv.tv_sec = 0; // shaddup compiler
          closed_expired = 0;
          if (client->last_buffupd_time == 0)
          {
            closed_expired = 1;
          }  
          else if ((client->scheduledupdatetime != 0 && 
            ((unsigned long)CliTimer(0)->tv_sec) >= 
     	      ((unsigned long)client->scheduledupdatetime)))
          {  
            closed_expired = 2;
          }  
          #if defined(PROJECTFLAGS_CLOSED_TTL) && \
              (PROJECTFLAGS_CLOSED_TTL > 0) /* any expiry time at all? */
          else if (CliClock(&tv)==0)
          {
            if (((unsigned long)tv.tv_sec) > 
              (unsigned long)(client->last_buffupd_time+PROJECTFLAGS_CLOSED_TTL)) 
            {  
              closed_expired = 3;
            }  
          }      
          #endif
        } /* if (closed_expired < 0) (undetermined) */
        TRACE_OUT((0,"closed_expired = %d\n", closed_expired));
        if (!closed_expired)
          isclosed = 1;
      }	  
      TRACE_OUT((0,"contest %d, closed=%d\n", cont_i, isclosed));
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
          TRACE_OUT((0,"1. contest %d, need_flush = %d need_fetch = %d\n", cont_i, need_flush, need_fetch));
        }      
        if (check_fetch && !need_fetch)
        {
          if (!BufferAssertIsBufferFull(client,cont_i))
          {
            unsigned long swucount;
            long pktcount = GetBufferCount( client, cont_i, 0, &swucount );
            if (pktcount <= 0) /* buffer is empty */
            {
              need_fetch = 1;
            }
            else if (fill_even_if_not_totally_empty)
            {
              if (swucount == 0) /* count not supported */
                swucount = pktcount * 100; /* >= 1.00 SWU's per packet */
              if (swucount < ClientGetInThreshold( client, cont_i, 0 ))
              {     
                need_fetch = 1;
              }  
            }
          }      
          if (need_fetch && either_or) /* either criterion satisfied ... */
            need_flush = 1;            /* ... fulfills both criteria */
          TRACE_OUT((0,"2. contest %d, need_flush = %d need_fetch = %d\n", cont_i, need_flush, need_fetch));
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
  
  TRACE_OUT((-1,"BufferCheckIfUpdateNeeded() => 0x%x\n", buffupd_flags ));
  return buffupd_flags;
}  
