/*
 * Copyright distributed.net 1997-2001 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * Public source buffer handling stuff
*/

const char *buffpub_cpp(void) {
return "@(#)$Id: buffpub.cpp,v 1.1.2.15 2001/03/07 00:21:06 sampo Exp $"; }

#include "cputypes.h"
#include "client.h"   //client class
#include "baseincs.h" //basic #includes
#include "util.h"     //trace
#include "logstuff.h" //Log()/LogScreen()
#include "pathwork.h" //GetFullPathForFilename()
#include "problem.h"  //Resultcode enum
#include "buffupd.h"  // BUFFERUPDATE_FETCH / BUFFERUPDATE_FLUSH
#include "buffbase.h" //the functions we're intending to export.

/* --------------------------------------------------------------------- */

int BufferDeinitialize(Client *client)
{
  /* in theory this is where we should purge mem buffers and
  ** shutdown checkpointing but because clients need to be able to 
  ** abort by calling probfill.cpp's LoadSaveProblems(NULL,0,0) 
  ** [ie, save with abortive action], putting it here is not much use.
  */
  client = client;
  return 0;
}

int BufferInitialize(Client *client)
{
  /* in theory this is where we should restore from checkpoint
  ** but so that users can use the checkpoint as a swap buffer,
  ** we do it from ClientRun() instead.
  */
  client = client;
  return 0;
}

/* --------------------------------------------------------------------- */

int BufferZapFileRecords( const char *filename )
{
  FILE *file;
  /* truncate, don't erase buffers (especially checkpoints). This reduces */
  /* disk fragmentation (on some systems) and is better on systems with a */
  /* "trash can" that saves deleted files. */

  filename = GetFullPathForFilename( filename );
  if (access( filename, 0 )!=0) //file doesn't exist, which is ok
    return 0;
  file = fopen( filename, "w" ); //truncate the file to zero length
  if (!file)
  {
    remove(filename); //write failed, so delete it
    return -1;
  }
  fclose(file);
  return 0;
}


/* --------------------------------------------------------------------- */

static FILE *BufferOpenFile( const char *filename, unsigned long *countP )
{
  /* OSs that require "b" for fopen() */
  #if ((CLIENT_OS == OS_BEOS) || (CLIENT_OS == OS_NEXTSTEP) || \
       (CLIENT_OS == OS_RISCOS) || (CLIENT_OS == OS_MACOS) || \
       (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN32) || \
       (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_OS2) || \
       (CLIENT_OS == OS_WIN16))
  #define BUFFOPEN_MODE "b"
  #else
  #define BUFFOPEN_MODE ""
  #endif

  FILE *file = NULL;
  long filelen;
  int failed = 0;
  const char *qfname = GetFullPathForFilename( filename );

  if (access(qfname, 0)!=0) // file doesn't exist, so create it
  {
    file = fopen( qfname, "w+" BUFFOPEN_MODE );
    if (file == NULL)
      failed = 1;
    else // file created. may be an exclusive open, so close and reopen
      fclose( file );
  }
  if (failed == 0)
  {
    if (access(qfname, 0)!=0) // file still doesn't exist
    {
      Log("Error opening buffer file '%s'\n"
          "Access was denied.\n", filename );
      return NULL;
    }
    file = fopen( qfname, "r+" BUFFOPEN_MODE );
    if (file == NULL)
      failed = 1;
  }
  if (failed != 0)
  {
    Log("Open failed for '%s'\n"
        "Check your file privileges (or your disk).\n", filename);
    return NULL;
  }

  if (fflush( file ) != 0 || fseek( file, 0, SEEK_END )!=0)
  {
    Log("Open failed for '%s'\n"
        "Unable to obtain directory information.\n", filename);
    fclose( file );
    return NULL;
  }
  if ((filelen = ftell(file)) == -1L)
  {
    Log("Open failed for '%s'\n"
        "Unable to determine file length.\n", filename);
    fclose( file );
    return NULL;
  }
  if ((filelen % sizeof(WorkRecord)) != 0)
  {
    Log("Open failed for '%s'\n"
        "Buffer file record count is inconsistent.\n", filename);
    fclose( file );
    return NULL;
  }
  if (countP)
  {
    *countP = (filelen / sizeof(WorkRecord));
  }    
  return file;
}

/* --------------------------------------------------------------------- */

static int BufferCloseFile( FILE *file )
{
  fclose(file);
  return 0;
}

/* --------------------------------------------------------------------- */

int UnlockBuffer( const char *filename )
{
  FILE *file = BufferOpenFile( filename, NULL );
  if (file)
  {
    BufferCloseFile( file );
    LogScreen("%s has been unlocked.\n",filename);
    return 0;
  }
  return -1; /* error message will already have been printed */
}

/* --------------------------------------------------------------------- */

int BufferNetUpdate(Client *client,int updatereq_flags, int break_pending, 
                    int interactive, char *loaderflags_map)
{
  client = client; updatereq_flags = updatereq_flags; 
  break_pending = break_pending; interactive = interactive; 
  loaderflags_map = loaderflags_map;
  return 0; /* nothing done */
}		    

/* --------------------------------------------------------------------- */

static void __switch_byte_order( WorkRecord *dest, const WorkRecord *source,
                             int from_disk /* going net->host */ )
{
  if (((const WorkRecord *)dest) != source )
    memcpy( (void *)dest, (const void *)source, sizeof(WorkRecord));

  dest->id[sizeof(dest->id)-1] = '\0';
  if (from_disk)
  {
    #if (CLIENT_OS == OS_OS390)
    __atoe(dest->id);
    #endif
  }
  else
  {
    #if (CLIENT_OS == OS_OS390)
    __etoa(dest->id);
    #endif
  }

  // Note that we do not distinguish between translating to-or-from
  // network-order, since the conversion is mathematically reversible
  // with the same operation on both little-endian and big-endian
  // machines (only on PDP machines does this assumption fail).
  switch (dest->contest)
  {
    case RC5:
    case DES:
    case CSC:
    {
      u32 *w = (u32 *)(&(dest->work));
      for (unsigned i=0; i<(sizeof(dest->work)/sizeof(u32)); i++)
        w[i] = (u32)ntohl(w[i]);
      break;
    }
    #if defined(HAVE_OGR_CORES)
    case OGR:
    {
      dest->work.ogr.workstub.stub.marks  = (u16)ntohs(dest->work.ogr.workstub.stub.marks);
      dest->work.ogr.workstub.stub.length = (u16)ntohs(dest->work.ogr.workstub.stub.length);
      for (int i = 0; i < STUB_MAX; i++)
        dest->work.ogr.workstub.stub.diffs[i] = (u16)ntohs(dest->work.ogr.workstub.stub.diffs[i]);
      dest->work.ogr.workstub.worklength  = (u32)ntohl(dest->work.ogr.workstub.worklength);
      dest->work.ogr.nodes.hi             = (u32)ntohl(dest->work.ogr.nodes.hi);
      dest->work.ogr.nodes.lo             = (u32)ntohl(dest->work.ogr.nodes.lo);
      break;
    }
    #endif
    default:
    {
      dest->contest = 0xff;
      break;
    }  
  }
  return;
}

/* --------------------------------------------------------------------- */

/* on failure returns -1, else 0 */
int BufferPutFileRecord( const char *filename, const WorkRecord * data,
                         unsigned long *countP )
{
  unsigned long reccount;
  FILE *file = BufferOpenFile( filename, &reccount );
  long count = -1L;
  if ( file )
  {
    unsigned long recno = 0, writerec = reccount;
    WorkRecord blank, scratch;
    memset( (void *)&blank, 0, sizeof(blank) );
    count = 0;

    if ( fseek( file, 0, SEEK_SET ) != 0)
      count = -1L;
    while (count >= 0 && recno < reccount)
    {
      if ( fread( (void *)&scratch, sizeof(WorkRecord), 1, file ) != 1)
        count = -1L;
      else if ( 0 != memcmp( (void *)&scratch, (void *)&blank, sizeof(WorkRecord)))
        count++;
      else /* blank record. write here */	
      {
        writerec = recno; /* blank record, write here */
        if (!countP) /* don't need to count to the end */
          break;
      }	  
      recno++;
    }
    if (count >= 0)
    {
      __switch_byte_order( &scratch, data, 0 /* going host->net order */ );
      if (fseek( file, (writerec * sizeof(WorkRecord)), SEEK_SET ) != 0)
        count = -1L;
      else if (fwrite( (void *)&scratch, sizeof(WorkRecord), 1, file ) != 1)
        count = -1L;
      else	
        count++;
    }
    BufferCloseFile( file );
  }
  if (count != -1L)
  {
    if (countP) 
      *countP = count;
    return 0;
  }
  return -1;
}

/* --------------------------------------------------------------------- */

int BufferGetFileRecordNoOpt( const char *filename, WorkRecord * data,
             unsigned long *countP ) /* returns <0 on ioerr, >0 if norecs */
{
  unsigned long reccount = 0;
  FILE *file = BufferOpenFile( filename, &reccount );
  int rc = -1;
  if ( file )
  {
    unsigned long recno = 0, blanked = 0;
    WorkRecord blank, scratch;
    memset( (void *)&blank, 0, sizeof(blank) );
    rc = +1; /* assume no records */
    if ( fseek( file, 0, SEEK_SET ) != 0)
      rc = -1;
    while (rc > 0 && recno < reccount)
    {
      if ( fread( (void *)&scratch, sizeof(WorkRecord), 1, file ) != 1)
        rc = -1;
      else if ( 0 == memcmp( (void *)&scratch, (void *)&blank, sizeof(WorkRecord)) )
        blanked++; /* blank record, ignore it */
      else if ( fseek( file, (recno * sizeof(WorkRecord)), SEEK_SET ) != 0)
        rc = -1;
      else if ( fwrite( (void *)&blank, sizeof(WorkRecord), 1, file ) != 1)
        rc = -1;
      else
      {
        blanked++;
        __switch_byte_order( data, &scratch, 1 /* going net->host order */ );
        rc = 0;
        break; /* got it */
      }	
      recno++;
    }
    BufferCloseFile( file );
    if (reccount > 0 && reccount == blanked) /* all blank */
      BufferZapFileRecords( filename );
  }
  if (countP)  
  {
    *countP = 0;
    if (rc == 0)
      BufferCountFileRecords(filename, data->contest, countP, NULL );
  }      
  return rc;
}

/* --------------------------------------------------------------------- */

int BufferGetFileRecord( const char *filename, WorkRecord * data,
               unsigned long *countP ) /* returns <0 on ioerr, >0 if norecs */
{
  return BufferGetFileRecordNoOpt( filename, data, countP );
}

/* --------------------------------------------------------------------- */

int BufferCountFileRecords( const char *filename, unsigned int contest,
                       unsigned long *packetcountP, unsigned long *normcountP )
{
  unsigned long normcount = 0, reccount = 0;
  FILE *file = BufferOpenFile( filename, &reccount );
  int failed = -1;
  if ( file )
  {
    unsigned long packetcount = 0, recno = 0;
    WorkRecord blank, scratch;
    memset( (void *)&blank, 0, sizeof(blank) );
    failed = 0;
    if ( fseek( file, 0, SEEK_SET ) != 0)
      failed = -1;
    while (!failed && recno < reccount)
    {
      if ( fread( (void *)&scratch, sizeof(WorkRecord), 1, file ) != 1)
        failed = -1;
      else if ( 0 == memcmp( (void *)&scratch, (void *)&blank, sizeof(WorkRecord)) )
        ; /* blank record, ignore it */
      else 
      {
        __switch_byte_order( &scratch, &scratch, 1 /* going net->host order */ );
        if ( ((unsigned int)(scratch.contest)) == contest )
        {
          packetcount++;
          if ( normcountP )
          {
            unsigned int swucount; 
            if (BufferGetRecordInfo( &scratch, 0, &swucount) >= 0)
            {
              normcount += swucount;
            }
          }    
        } /* contest is same */
      }
      recno++;
    }
    reccount = packetcount;
    BufferCloseFile( file );
  }
  if (failed != 0)
    normcount = reccount = 0;
  if (normcountP)
    *normcountP = normcount;
  if (packetcountP)
    *packetcountP = reccount;
  return failed;
}

/* --------------------------------------------------------------------- */
