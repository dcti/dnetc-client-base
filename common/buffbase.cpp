/*
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
*/
const char *buffbase_cpp(void) {
return "@(#)$Id: buffbase.cpp,v 1.12.2.28 2000/03/09 12:27:29 jlawson Exp $"; }

#include "cputypes.h"
#include "cpucheck.h" //GetNumberOfDetectedProcessors()
#include "client.h"   //client class
#include "baseincs.h" //basic #includes
#include "network.h"  //ntohl(), htonl()
#include "util.h"     //IsFilenameValid(), DoesFileExist(), __iter2norm()
#include "clievent.h" //event stuff
#include "clicdata.h" //GetContestNameFromID()
#include "logstuff.h" //Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "triggers.h" //[Check|Raise][Pause|Exit]RequestTrigger()
#include "pathwork.h" //GetFullPathForFilename() or dummy if DONT_USE_PATHWORK
#include "problem.h"  //Resultcode enum
#include "probfill.h"
#include "buffupd.h"  // BUFFERUPDATE_FETCH / BUFFERUPDATE_FLUSH
#include "buffbase.h" //ourselves

/* --------------------------------------------------------------------- */

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

int GetFileLengthFromStream( FILE *file, u32 *length )
{
  #if (CLIENT_OS == OS_WIN32)
    u32 result = (u32) GetFileSize((HANDLE)_get_osfhandle(fileno(file)),NULL);
    if (result == 0xFFFFFFFFL) return -1;
    *length = result;
  #elif (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN16)
    u32 result = filelength( fileno(file) );
    if (result == 0xFFFFFFFFL) return -1;
    *length = result;
  #elif (CLIENT_OS == OS_RISCOS)
    if (riscos_get_filelength(fileno(file),(unsigned long *)length) != 0)
    {
      return -1;
    }
  #else
    struct stat statbuf;
    #if (CLIENT_OS == OS_NETWARE)
    unsigned long inode;
    int vno;
    if (FEMapHandleToVolumeAndDirectory( fileno(file), &vno, &inode )!=0)
      { vno = 0; inode = 0; }
    if ( vno == 0 && inode == 0 )
    {                                       /* file on DOS partition */
      u32 result = filelength( fileno(file) );  // ugh! uses seek
      if (result == 0xFFFFFFFFL) return -1;
      *length = result;
      return 0;
    }
    #endif
    if ( fstat( fileno( file ), &statbuf ) != 0) return -1;
    *length = (u32)statbuf.st_size;
  #endif
  return 0;
}

/* --------------------------------------------------------------------- */

// Supply macros to wrap around the system-dependent file open, creation,
// and share locking operations.  Notice that you should attempt to do
// native file-system share locking when possible, even though we
// additionally implement a "soft lock" within the buffer-format itself.

#if ((CLIENT_OS == OS_BEOS) || (CLIENT_OS == OS_NEXTSTEP) || \
     (CLIENT_OS == OS_MACOS) || (CLIENT_OS == OS_RISCOS))
  // These operating systems distinguish "binary" files.
  #define BUFFEROPEN( fn )  fopen( fn, "r+b" )
  #undef BUFFEROPENCREATE
  #define BUFFERCREATE( fn )   fopen( fn, "w+b" )
#elif (CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_VMS)
  // These operating systems don't have an ftruncate equivalent.
  #define BUFFEROPEN( fn )  fopen( fn, "r+" )
  #undef BUFFEROPENCREATE
  #define BUFFERCREATE( fn )   fopen( fn, "w+" )
  #define ftruncate(h,sz)
#elif (CLIENT_OS == OS_SUNOS) && (CLIENT_CPU == CPU_68K) //Sun3
  #define BUFFEROPEN( fn )  fopen( fn, "r+" )
  #undef BUFFEROPENCREATE
  #define BUFFERCREATE( fn )   fopen( fn, "w+" )
  extern "C" int ftruncate(int, off_t); // Keep g++ happy.
#elif ((CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN32) || \
       (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_OS2) || \
       (CLIENT_OS == OS_WIN16))
  static FILE *__bufferopenlock(const char *fn, int bCreate)
  {
    #if (CLIENT_OS == OS_NETWARE)
    DIR *dir = opendir( fn );
    if ( dir != NULL && ( readdir( dir ) != NULL ) )
    {
      dir->d_cdatetime=(dir->d_cdatetime >> 16 | dir->d_cdatetime << 16);
      dir->d_adatetime=(dir->d_adatetime >> 16 | dir->d_adatetime << 16);
      long mdatetime  =(dir->d_date | dir->d_time << 16);
      dir->d_bdatetime=(dir->d_bdatetime >> 16 | dir->d_bdatetime << 16);

      SetFileInfo( (char *)(fn), 0x06, (dir->d_attr|_A_SHARE|_A_IMMPURG),
                   (char *)(&dir->d_cdatetime), (char *)(&dir->d_adatetime),
                   (char *)(&mdatetime), (char *)(&dir->d_bdatetime),
                   dir->d_uid  );
      closedir( dir );
      //technically could now use normal fopen().
    }
    #endif
    int handle;
    if (bCreate)
      handle = sopen(fn, O_RDWR|O_BINARY|O_CREAT,
                     SH_DENYRW, S_IREAD|S_IWRITE);
    else
      handle = sopen(fn, O_RDWR|O_BINARY, SH_DENYWR);

    if (handle != -1)
    {
      FILE *f = fdopen(handle, "r+b");
      if (f)
        return f;
      close (handle);
    }
    return (FILE *)0;
  }
  #define ftruncate(h,sz) chsize(h,sz)
  #define BUFFEROPEN( fn )  __bufferopenlock( fn, 0 )
  #define BUFFEROPENCREATE( fn )  __bufferopenlock( fn, 1 )
  #define BUFFERCREATE( fn ) fopen( fn, "w+b" )
#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD)
  static FILE *__bufferopenlock(const char *fn, int bCreate)
  {
    int handle;
    if (bCreate)
      handle = open(fn, O_RDWR|O_NONBLOCK|O_CREAT,
                    S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP);
    else
      handle = open(fn, O_RDWR|O_NONBLOCK);

    if (handle != -1)
    {
      if (flock(handle, LOCK_EX | LOCK_NB) >= 0)
      {
        FILE *fp = fdopen(handle, "r+");
        if (fp != NULL) return fp;
      }
      close(handle);
    }
    return (FILE*) 0;
  }
  #define BUFFEROPEN( fn )   __bufferopenlock( fn, 0 )
  #define BUFFEROPENCREATE( fn )   __bufferopenlock( fn, 1 )
  #define BUFFERCREATE( fn )   fopen( fn, "w+" )
#else
  #define BUFFEROPEN( fn )  fopen( fn, "r+" )
  #undef BUFFEROPENCREATE
  #define BUFFERCREATE( fn )   fopen( fn, "w+" )
#endif

/* --------------------------------------------------------------------- */

#ifndef SUPPRESS_FILE_BUFFER_PRIMITIVES
const char *buffwork_cpp(void) { return ""; }
const char *buffupd_cpp(void) { return ""; }

int BufferZapFileRecords( const char *filename )
{
  FILE *file;
  /* truncate, don't erase buffers (especially checkpoints). This reduces */
  /* disk fragmentation (on some systems) and is better on systems with a */
  /* "trash can" that saves deleted files. */

  if (!IsFilenameValid( filename ))
    return 0;
  filename = GetFullPathForFilename( filename );
  if (!DoesFileExist( filename )) //file doesn't exist, which is ok
    return 0;
  file = BUFFERCREATE( filename ); //truncate the file to zero length
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
  FILE *file = NULL;
  int failed = 0;
  u32 filelen;

  filename = GetFullPathForFilename( filename );

  if (!DoesFileExist( filename )) // file doesn't exist, so create it
  {
    if ((file = BUFFERCREATE( filename ))==NULL)
      failed = 1;
    else // file created. may be an exclusive open, so close and reopen
      fclose( file );
  }
  if (failed == 0)
  {
    if (!DoesFileExist( filename )) // file still doesn't exist
    {
      Log("Error opening buffer file... Access was denied.\n" );
      return NULL;
    }
    if ((file = BUFFEROPEN( filename ))==NULL)
      failed = 1;
  }
  if (failed != 0)
  {
    Log("Open failed. Check your file privileges (or your disk).\n" );
    return NULL;
  }

  if (fflush( file ) != 0 || GetFileLengthFromStream( file, &filelen ) != 0)
  {
    Log("Open failed. Unable to obtain directory information.\n");
    fclose( file );
    return NULL;
  }
  if ((filelen % sizeof(WorkRecord)) != 0)
  {
    Log("Open failed. Buffer file record count is inconsistent.\n");
    fclose( file );
    return NULL;
  }
  if (countP)
    *countP = (filelen / sizeof(WorkRecord));
  return file;
}

/* --------------------------------------------------------------------- */

static int BufferCloseFile( FILE *file )
{
  fclose(file);
  return 0;
}

/* --------------------------------------------------------------------- */

static void  __switchborder( WorkRecord *dest, const WorkRecord *source )
{
  if (((const WorkRecord *)dest) != source )
    memcpy( (void *)dest, (const void *)source, sizeof(WorkRecord));
  /* we don't do PDP byte order, so this is ok */
  switch (dest->contest)
  {
    case RC5:
    case DES:
    case CSC:
    {
      u32 *w = (u32 *)(&(dest->work));
      for (unsigned i=0; i<(sizeof(dest->work)/sizeof(u32)); i++)
        w[i] = ntohl(w[i]);
      break;
    }
    case OGR:
    {
      dest->work.ogr.workstub.stub.marks  = ntohs(dest->work.ogr.workstub.stub.marks);
      dest->work.ogr.workstub.stub.length = ntohs(dest->work.ogr.workstub.stub.length);
      for (int i = 0; i < STUB_MAX; i++)
        dest->work.ogr.workstub.stub.diffs[i] = ntohs(dest->work.ogr.workstub.stub.diffs[i]);
      dest->work.ogr.workstub.worklength  = ntohl(dest->work.ogr.workstub.worklength);
      dest->work.ogr.nodes.hi             = ntohl(dest->work.ogr.nodes.hi);
      dest->work.ogr.nodes.lo             = ntohl(dest->work.ogr.nodes.lo);
      break;
    }
  }
  return;
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

int BufferPutFileRecord( const char *filename, const WorkRecord * data,
                         unsigned long *countP )
{
  unsigned long reccount;
  FILE *file = BufferOpenFile( filename, &reccount );
  int failed = -1;
  if ( file )
  {
    unsigned long recno = 0;
    WorkRecord blank, scratch;
    memset( (void *)&blank, 0, sizeof(blank) );
    failed = 0;
    if ( fseek( file, 0, SEEK_SET ) != 0)
      failed = -1;
    while (!failed && recno < reccount)
    {
      if ( fread( (void *)&scratch, sizeof(WorkRecord), 1, file ) != 1)
        failed = -1;
      else if ( 0 == memcmp( (void *)&scratch, (void *)&blank, sizeof(WorkRecord)))
        break; /* blank record, write here */
      recno++;
    }
    if (!failed)
    {
      failed = -1;
      __switchborder( &scratch, data );
      if ( fseek( file, (recno * sizeof(WorkRecord)), SEEK_SET ) == 0 &&
                fwrite( (void *)&scratch, sizeof(WorkRecord), 1, file ) == 1 )
        failed = 0;
    }
    BufferCloseFile( file );
  }
  if (!failed && countP)
    BufferCountFileRecords( filename, data->contest, countP, NULL );
  return failed;
}

/* --------------------------------------------------------------------- */

int BufferUpdate( Client *client, int updatereq_flags, int interactive )
{
  char loaderflags_map[CONTEST_COUNT];
  int failed, dofetch, doflush, didfetch, didflush, dontfetch, dontflush;
  unsigned int i, contest_i;
  const char *ffmsg="--fetch and --flush services are not available.\n";

  if (client->noupdatefromfile || client->remote_update_dir[0] == '\0')
  {
    if (interactive)
      LogScreen( "%sThis client has been configured to run without\n"
                 "updating its buffers.\n",ffmsg);
    return -1;
  }

  dontfetch = dontflush = 1;
  if ((updatereq_flags & BUFFERUPDATE_FETCH) != 0)
    dontfetch = 0;
  if ((updatereq_flags & BUFFERUPDATE_FLUSH) != 0)
    dontflush = 0;

  dofetch = doflush = 0;
  for (i = 0; i < CONTEST_COUNT; i++)
  {
    contest_i = (unsigned int)(client->loadorder_map[i]);
    loaderflags_map[i] = 0;

    if (contest_i >= CONTEST_COUNT) /* disabled */
    {
      /* retrieve the original contest number from the load order */
      contest_i &= 0x7f;
      if (contest_i < CONTEST_COUNT)
        loaderflags_map[contest_i] = PROBLDR_DISCARD;  /* thus discardable */
    }
    else if (dofetch == 0 || doflush == 0)/* (contest_i < CONTEST_COUNT ) */
    {
      if (!dofetch && !dontfetch)
      {
        long count;
        if (GetBufferCount(client, contest_i, 0, &count) >= 0) /* no error */
        {
          if (count < ClientGetInThreshold( client, contest_i ) )
          {
            if (count <= 1 || client->connectoften || interactive)
              dofetch = 1;
          }
        }
      }
      if (!doflush && !dontflush)
      {
        long count = GetBufferCount( client, contest_i, 1 /* use_out_file */, NULL );
        if (count >= 0) /* no error */
        {
          if ( count > 0 /* count >= ClientGetOutThreshold( client, contest ) ||
            ( client->connectoften && count > 0) || (interactive && count > 0) */)
          {
            doflush = 1;
          }
        }
      }
    }
  }

  failed = didfetch = didflush = 0;
  if ((doflush || dofetch) && CheckExitRequestTriggerNoIO() == 0)
  {
    if (!client->noupdatefromfile && client->remote_update_dir[0] != '\0')
    {
      if (failed == 0 && !dontfetch && CheckExitRequestTriggerNoIO() == 0)
      {
        long transferred = BufferFetchFile( client, &loaderflags_map[0] );
        if (transferred < 0)
        {
          failed = 1;
          if (transferred < -1)
            didfetch = 1;
        }
        else if (transferred > 0)
          didfetch = 1;
      }
      if (failed == 0 && !dontflush && CheckExitRequestTriggerNoIO() == 0)
      {
        long transferred = BufferFlushFile( client, &loaderflags_map[0] );
        if (transferred < 0)
        {
          failed = 1;
          if (transferred < -1)
            didflush = 1;
        }
        else if (transferred > 0)
          didflush = 1;
      }
    }
  }

  ffmsg = "%sput buffers are %s. No %s required.\n";
  updatereq_flags = 0;
  if (didfetch)
    updatereq_flags |= BUFFERUPDATE_FETCH;
  else if (interactive && failed==0 && !dontfetch && !dofetch)
    LogScreen(ffmsg, "In", "full", "fetch");
  if (didflush)
    updatereq_flags |= BUFFERUPDATE_FLUSH;
  else if (interactive && failed==0 && !dontflush && !doflush)
    LogScreen(ffmsg, "Out", "empty", "flush");
  return (updatereq_flags);
}

/* --------------------------------------------------------------------- */

int BufferGetFileRecordNoOpt( const char *filename, WorkRecord * data,
             unsigned long *countP ) /* returns <0 on ioerr, >0 if norecs */
{
  unsigned long reccount = 0;
  FILE *file = BufferOpenFile( filename, &reccount );
  int failed = -1;
  if ( file )
  {
    unsigned long recno = 0;
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
      else if ( fseek( file, (recno * sizeof(WorkRecord)), SEEK_SET ) != 0)
        failed = -1;
      else if ( fwrite( (void *)&blank, sizeof(WorkRecord), 1, file ) != 1)
        failed = -1;
      else
        break; /* got it */
      recno++;
    }
    BufferCloseFile( file );
    if ( failed == 0 )
    {
      failed = +1;
      if (recno < reccount ) /* got it */
      {
        failed = 0;
        __switchborder( data, &scratch );
        if (reccount == 1)
          BufferZapFileRecords( filename );
      }
    }
  }
  if (!failed && countP)
    BufferCountFileRecords( filename, data->contest, countP, NULL );
  return failed;
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
      else if ( ((unsigned int)(scratch.contest)) == contest )
      {
        packetcount++;
        if ( normcountP )
        {
          switch (contest)
          {
            case RC5:
            case DES:
            case CSC:
              normcount += (unsigned int)
                 __iter2norm( ntohl(scratch.work.crypto.iterations.lo),
                              ntohl(scratch.work.crypto.iterations.hi) );
              break;
            case OGR:
              normcount++;
              break;
          }
        }
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

#endif  /* SUPPRESS_FILE_BUFFER_PRIMITIVES */

/* ===================================================================== */
/*                         END OF FILE PRIMITIVES                        */
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

  if ( !DoesFileExist( source_file ) )
  {
    if (interactive)
      LogScreen("Import error: Source '%s' doesn't exist\n", source_file );
    return -1L;
  }

  while (BufferGetFileRecordNoOpt( source_file, &data, &remaining ) == 0)
                           //returns <0 on ioerr/corruption, > 0 if norecs
  {
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
    BufferZapFileRecords( source_file );
  }
  if (recovered > 0 && interactive)
    LogScreen("Import::%ld records successfully imported.\n", recovered);
  else if (errs == 0 && recovered == 0 && interactive)
    LogScreen("Import::No buffer records could be imported.\n");
  return (long)recovered;
}

/* --------------------------------------------------------------------- */

// Flush blocks to a remote buffer fie.

long BufferFlushFile( Client *client, const char *loadermap_flags )
{
  long combinedtrans = 0, combinedworkunits = 0;
  char basename[sizeof(client->remote_update_dir)  +
                sizeof(client->out_buffer_basename) + 10 ];
  unsigned int contest;
  int failed = 0;


  //
  // If we aren't configured to do remote buffer updating, then quit.
  //
  if (client->noupdatefromfile || client->remote_update_dir[0] == '\0')
    return 0;


  //
  // Generate the full path of the remote buffer.
  //
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


  //
  // Loop through each contest and perform the fetching activity.
  //
  for (contest = 0; failed == 0  && contest < CONTEST_COUNT; contest++)
  {
    WorkRecord wrdata;
    unsigned long projtrans = 0;
    long lefttotrans;
    char remote_file[128];

    if (CheckExitRequestTriggerNoIO())
      break;

    if (loadermap_flags[contest] != 0) /* contest is closed or disabled */
      continue;

    strncpy( remote_file, BufferGetDefaultFilename(contest,1,basename), sizeof(remote_file));
    remote_file[sizeof(remote_file)-1] = '\0';

    while ((lefttotrans = GetBufferRecord( client, &wrdata, contest, 1 )) >= 0)
    {
      long workunits = 0;

      if (( wrdata.resultcode != RESULT_NOTHING &&
            wrdata.resultcode != RESULT_FOUND ) ||
            ((unsigned int)wrdata.contest) != contest)
      { /* buffer code should have handled this */
        //Log( "%sError - Bad Data - packet discarded.\n",exchname );
        continue;
      }

      if ( BufferPutFileRecord( remote_file, &wrdata, NULL ) != 0 )
      {
        PutBufferRecord( client, &wrdata );
        failed = -1;
        break;
      }

      switch (contest)
      {
        case RC5:
        case DES:
        case CSC:
          workunits =  __iter2norm(wrdata.work.crypto.iterations.lo,
                                   wrdata.work.crypto.iterations.hi);
          break;
        case OGR:
          workunits = 1;
          break;
      }

      projtrans++;
      combinedtrans++;
      combinedworkunits += workunits;

      {
        struct Fetch_Flush_Info ffinfo = {contest, projtrans, combinedtrans};
        unsigned long totrans = (projtrans + (unsigned long)(lefttotrans));

        ClientEventSyncPost(CLIEVENT_BUFFER_FLUSHFLUSHED, (long)(&ffinfo));

        unsigned int percent;
        if (totrans < projtrans)
          totrans = projtrans;
        percent = ((projtrans*10000)/totrans);

        LogScreen( "\rSent %s packet %lu of %lu (%u.%02u%% transferred)     ",
            CliGetContestNameFromID(contest),
            projtrans, totrans,  percent/100, percent%100 );
      }

      if (CheckExitRequestTriggerNoIO())
        break;

    } /* while (lefttotrans >=0 ) */

  } /* for (contest = 0; contest < CONTEST_COUNT; contest++) */


  //
  // Print out a final summary of how many blocks were transferred.
  //
  if (combinedtrans > 0)
  {
    ClientEventSyncPost( CLIEVENT_BUFFER_FLUSHEND, (long)(combinedtrans) );
    Log( "Moved %lu packet%s (%lu work units) to remote file.\n",
      combinedtrans, ((combinedtrans==1)?(""):("s")), combinedworkunits );
  }


  //
  // Return the result code.
  //
  if (failed)
    return -((long)(combinedtrans+1));
  return combinedtrans;
}

/* --------------------------------------------------------------------- */

// Fetch blocks from a remote buffer file

long BufferFetchFile( Client *client, const char *loaderflags_map )
{
  unsigned long combinedtrans = 0, combinedworkunits = 0;
  char basename[sizeof(client->remote_update_dir)  +
                sizeof(client->in_buffer_basename) + 10 ];
  unsigned int contest;
  int failed = 0;


  //
  // If we aren't configured to do remote buffer updating, then quit.
  //
  if (client->noupdatefromfile || client->remote_update_dir[0] == '\0')
    return 0;


  //
  // we need the correct number of cpus _used_ for time estimates
  // but need the correct number of _crunchers_ for thresh.
  //
  int proc, numcrunchers;
  proc = GetNumberOfDetectedProcessors();
  if (proc < 1)
    proc = 1;
  numcrunchers = client->numcpu;
  if (numcrunchers < 0)
    numcrunchers = proc;
  else if (numcrunchers == 0) /* force non-threaded */
    numcrunchers = 1;


  //
  // Generate the full path of the remote buffer.
  //
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


  //
  // Loop through each contest and perform the fetching activity.
  //
  for (contest = 0; !failed && contest < CONTEST_COUNT; contest++)
  {
    unsigned long projtrans = 0, projworkunits = 0;
    unsigned long lefttotrans;
    int packets;
    char remote_file[128];

    if (CheckExitRequestTriggerNoIO())
      break;

    if (loaderflags_map[contest] != 0) /* contest is closed or disabled */
      continue;

    if ((packets = GetBufferCount( client, contest, 0, &lefttotrans)) < 0)
      lefttotrans = 0;
    else
    {
      unsigned long threshold = ClientGetInThreshold( client, contest, 1 /* force */ );
      if (lefttotrans >= threshold) /* buffers full? */
        if (packets < numcrunchers)  /* Have at least one packet per cruncher? */
          lefttotrans = numcrunchers - packets;
        else
          lefttotrans = 0;
      else
        lefttotrans = threshold - lefttotrans;
    }

    if (lefttotrans != 0)
    {
      strncpy( remote_file, BufferGetDefaultFilename(contest,1,basename), sizeof(remote_file));
      remote_file[sizeof(remote_file)-1] = '\0';
      if (!DoesFileExist(remote_file))
        lefttotrans = 0;
//printf("remotefile: %s %ld\n",remote_file,lefttotrans);
    }

    while (lefttotrans > 0 )
    {
      WorkRecord wrdata;
      unsigned long remaining;
      int workunits = 0;

      if (CheckExitRequestTriggerNoIO() != 0 )
        break;

      if ( BufferGetFileRecordNoOpt( remote_file, &wrdata, &remaining ) != 0 )
        break;
      if (remaining == 0)
         lefttotrans = 0;

      if (PutBufferRecord( client, &wrdata ) < 0)
      {
        BufferPutFileRecord( remote_file, &wrdata, NULL );
        failed = -1;
        lefttotrans = 0;
        break;
      }

      switch (contest)
      {
        case RC5:
        case DES:
        case CSC:
          workunits =  __iter2norm(wrdata.work.crypto.iterations.lo,
                                   wrdata.work.crypto.iterations.hi);
          break;
        case OGR:
          workunits = 1;
          break;
      }

      projtrans++;
      projworkunits += workunits;
      combinedtrans++;
      combinedworkunits += workunits;
      if (((unsigned long)workunits) > lefttotrans)
        lefttotrans = 0;
      else
        lefttotrans -= workunits;

      if (combinedtrans == 1)
        ClientEventSyncPost( CLIEVENT_BUFFER_FETCHBEGIN, 0 );

      {
        struct Fetch_Flush_Info ffinfo = {contest, projtrans, combinedtrans};
        unsigned long totrans = (projworkunits + (unsigned long)(lefttotrans));

        ClientEventSyncPost(CLIEVENT_BUFFER_FLUSHFLUSHED, (long)(&ffinfo));

        unsigned int percent;
        if (totrans < projworkunits)
          totrans = projworkunits;
        percent = ((projworkunits*10000)/totrans);

        LogScreen( "\rRetrieved %s work unit %lu of %lu (%u.%02u%% transferred) ",
            CliGetContestNameFromID(contest),
            projworkunits, totrans,  percent/100, percent%100 );
      }
    }  /* while ( lefttotrans > 0  ) */

  } /* for (contest = 0; contest < CONTEST_COUNT; contest++) */


  //
  // Print out a final summary of how many blocks were transferred.
  //
  if (combinedtrans > 0)
  {
    ClientEventSyncPost( CLIEVENT_BUFFER_FETCHEND, (long)(combinedtrans) );
    Log( "Retrieved %lu work unit%s (%lu packet%s) from remote file.\n",
      combinedworkunits, ((combinedworkunits==1)?(""):("s")),
      combinedtrans, ((combinedtrans==1)?(""):("s")) );
  }


  //
  // Return the result code.
  //
  if (failed)
    return -((long)(combinedtrans+1));
  return combinedtrans;
}

/* --------------------------------------------------------------------- */

