/*
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *util_cpp(void) {
return "@(#)$Id: util.cpp,v 1.11.2.25 2000/04/14 17:19:55 cyp Exp $"; }

#include "baseincs.h" /* string.h, time.h */
#include "version.h"  /* CLIENT_CONTEST */
#include "client.h"   /* CONTEST_COUNT, stub definition */
#include "logstuff.h" /* Log() */
#include "clitime.h"  /* CliTimer(), Time()/(CliGetTimeString(NULL,1)) */
#include "cliident.h" /* CliIsDevelVersion() */
#include "clicdata.h" /* CliGetContestNameFromID() */
#include "pathwork.h" /* GetFullPathForFilename() */
#include "util.h"     /* ourselves */
#define MAX_CONTEST_NAME_LEN 3

/* ------------------------------------------------------------------- */

void trace_out( int indlevel, const char *format, ... )
{
  static int indentlevel = -1; /* uninitialized */
  const char *tracefile = "trace"EXTN_SEP"out";
  FILE *file;
  va_list arglist;
  va_start (arglist, format);

  if (indentlevel == -1) /* uninitialized */
  {
    remove(tracefile);
    indentlevel = 0;
  }

  if (indlevel < 0)
    indentlevel -= 2;
  file = fopen( tracefile, "a" );
  if (file)
  {
    char buffer[64];
    time_t t = time(NULL);
    struct tm *lt = localtime( &t );
    fprintf(file, "%02d:%02d:%02d: ", lt->tm_hour, lt->tm_min, lt->tm_sec);
    if (indentlevel > 0)
    {
      size_t spcs = ((size_t)indentlevel);
      memset((void *)(&buffer[0]),' ',sizeof(buffer));
      while (sizeof(buffer) == fwrite( buffer, 1,
         ((spcs>sizeof(buffer))?(sizeof(buffer)):(spcs)), file ))
        spcs -= sizeof(buffer);
    }
    if (indlevel != 0)
      fwrite((const char *)((indlevel < 0)?("end: "):("beg: ")), 1, 5, file );
    vfprintf(file, format, arglist);
    fflush( file );
    fclose( file );
  }
  if (indlevel > 0)
    indentlevel += 2;
  return;
}

/* ------------------------------------------------------------------- */

int utilCheckIfBetaExpired(int print_msg)
{
  if (CliIsDevelVersion()) /* cliident.cpp */
  {
    timeval expirationtime;
    time_t now = (CliTimer(NULL)->tv_sec); /* net adjusted */

    #ifndef BETA_PERIOD
    #define BETA_PERIOD (7L*24L*60L*60L) /* one week from build date */
    #endif    /* where "build date" is time of newest module in ./common/ */
    expirationtime.tv_sec = CliGetNewestModuleTime() + (time_t)BETA_PERIOD;
    expirationtime.tv_usec= 0;

    if (now >= expirationtime.tv_sec)
    {
      if (print_msg)
      {
        Log("This beta release expired on %s. Please\n"
            "download a newer beta, or run a standard-release client.\n",
            CliGetTimeString(&expirationtime,1) );
      }
      return 1;
    }
    else if (print_msg)
    {
      static time_t last_seen = 0;
      time_t wtime = time(NULL);
      if (last_seen == 0)
        last_seen = 1; //let it through once (print banner)
      else if (wtime < last_seen || (wtime - last_seen) > 10*60)
      {
        expirationtime.tv_sec -= now;
        LogScreen("*** This BETA release expires in %s. ***\n",
            CliGetTimeString(&expirationtime,2) );
        last_seen = wtime;
      }
    }
  }
  return 0;
}

/* ------------------------------------------------------------------- */

u32 __iter2norm( u32 iterlo, u32 iterhi )
{
  iterlo = ((iterlo >> 28) + (iterhi << 4));
  if (!iterlo)
    iterlo++;
  return iterlo;
}

unsigned long ogr_nodecount(const struct Stub * /* stub */)
{
  return 1;
}

const char *ogr_stubstr(const struct Stub *stub)
{
  static char buf[80];
  int i, len = (int)stub->length;

  if (len > STUB_MAX) {
    sprintf(buf, "(error:%d/%d)", (int)stub->marks, len);
    return buf;
  }
  sprintf(buf, "%d/", (int)stub->marks);
  if (len == 0) {
    strcat(buf, "-");
    return buf;
  }
  for (i = 0; i < len; i++) {
    sprintf(&buf[strlen(buf)], "%d", (int)stub->diffs[i]);
    if (i+1 < len) {
      strcat(buf, "-");
    }
  }
  return buf;
}

/* ------------------------------------------------------------------- */

int utilGatherOptionArraysToList( char *buffer, unsigned int buflen,
                                  const int *table1, const int *table2 )
{
  unsigned int donelen = 0;
  unsigned int contest;
  const char *delim = "";
  if (buffer && buflen)
    buffer[0] = '\0';
  for (contest = 0; contest < CONTEST_COUNT; contest++)
  {
    //if (1)
    {
      const char *p = CliGetContestNameFromID(contest);
      if (p)
      {
        char single[(MAX_CONTEST_NAME_LEN+1+(sizeof(int)*3)+1+(sizeof(int)*3)+1)];
        unsigned int len;
        if (table2)
          len = sprintf(single,"%s%s=%d:%d",delim, p,
                        (int)table1[contest],(int)table2[contest]);
        else
          len = sprintf(single,"%s%s=%d",delim, p, (int)table1[contest] );
        if (!buffer || !buflen)
        {
          donelen += len;
          delim = ",";
        }
        else if ((donelen + len) < (buflen-1))
        {
          strcpy( &buffer[donelen], single );
          donelen += len;
          delim = ",";
        }
      }
    }
  }
  return donelen;
}

int utilScatterOptionListToArraysEx( const char *oplist,
                                   int *table1, int *table2,
                                   const int *default1, const int *default2 )
{
  unsigned int cont_i;

  for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
  {
    if (default1)
      table1[cont_i] = default1[cont_i];
    if (table2 && default2)
      table2[cont_i] = default2[cont_i];
  }

  while (*oplist)
  {
    while (*oplist && !isalpha(*oplist)) /*contestname must begin with alpha*/
      oplist++;
    if (*oplist)
    {
      char buffer[64];
      unsigned int len = 0;
      int needbreak = 0, kwpos = 0, precspace = 0;
      unsigned int contest = CONTEST_COUNT;
      int havenondig = 0, value1 = 0, value2 = 0, haveval1 = 0, haveval2 = 0;
      while (!needbreak && len<sizeof(buffer)-1)
      {
        char c = buffer[len] = (char)(*oplist++);
        buffer[len+1] = '\0';
        if (c==',' || c==';' || !c)
        {
          c=':';
          needbreak = 1;
          oplist--;
        }
        if (c==':' || c=='=')
        {
          buffer[len] = '\0';
          precspace = 0;
          if (len != 0)
          {
            kwpos++;
            if (kwpos == 1)
            {
              for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
              {
                const char *cname = CliGetContestNameFromID(cont_i);
                if (cname)
                {
                  if (strcmp(cname, buffer)==0)
                  {
                    contest = cont_i;
                    break;
                  }
                }
              }
              if (contest >= CONTEST_COUNT)
                break;
            }
            else if (kwpos == 2)
            {
              if (havenondig)
                break;
              value1 = atoi(buffer);
              haveval1 = 1;
            }
            else if (kwpos == 3 && table2)
            {
              if (havenondig)
                break;
              value2 = atoi(buffer);
              haveval2 = 1;
            }
            else
            {
              break;
            }
            len = 0;
            havenondig = 0;
          }
        }
        else if (c == ' ' || c=='\t') /* may only be followed by [=:;,\0] */
        {
          if (len != 0) /* otherwise ignore it */
            precspace = 1;
        }
        else if (isalpha(c))
        {
          if (kwpos || precspace)
            break;
          buffer[len] = (char)toupper(c);
          havenondig = 1;
          len++;
        }
        else if (isdigit(c))
        {
          if (precspace)
            break;
          len++;
        }
        else if (c == '+' || c=='-')
        {
          if (len!=0 || !isdigit(*oplist))
            break;
          len++;
        }
      }
      if (contest < CONTEST_COUNT && haveval1)
      {
        table1[contest] = value1;
        if (!haveval2)
          value2 = value1;
        if (table2)
          table2[contest] = value2;
      }
      while (*oplist && *oplist!=',' && *oplist!=';')
        oplist++;
    }
  }
  return 0;
}

int utilScatterOptionListToArrays( const char *oplist,
                                   int *table1, int *table2, int defaultval )
{
  int defarray[CONTEST_COUNT];
  unsigned int cont_i;
  for (cont_i = 0;cont_i < CONTEST_COUNT; cont_i++)
    defarray[cont_i] = defaultval;
  return utilScatterOptionListToArraysEx( oplist,
                                 table1, table2, &defarray[0], &defarray[0]);
}

const char *projectmap_expand( const char *map )
{
  static char buffer[(CONTEST_COUNT+1)*(MAX_CONTEST_NAME_LEN+3)];
  unsigned int id;

  if (!map)
    map = projectmap_build(NULL,NULL);

  buffer[0] = '\0';
  for (id=0;id<CONTEST_COUNT;id++)
  {
    if (id > 0)
      strcat( buffer, "," );
    strcat( buffer, CliGetContestNameFromID( map[id] & 0x7f ) );
    if (( map[id] & 0x80 ) != 0)
      strcat( buffer,"=0" );
  }
  return buffer;
}

// --------------------------------------------------------------------------

const char *projectmap_build( char *buf, const char *strtomap )
{
  #if (CONTEST_COUNT != 4)
    #error static table needs fixing. (CONTEST_COUNT is not 4).
  #endif
  static char default_map[CONTEST_COUNT] = { 1,3,2,0 };
  static char map[CONTEST_COUNT];
  unsigned int map_pos, i;
  int contestid;

  if (!strtomap || !*strtomap)
  {
    if (buf)
      memcpy((void *)buf, (void *)&default_map[0], CONTEST_COUNT );
    return default_map;
  }

//printf("\nreq order: %s\n", strtomap );

  map_pos = 0;
  do
  {
    int disabled = 0;
    char scratch[10];
    while (*strtomap && !isalpha(*strtomap) && !isdigit(*strtomap))
      strtomap++;
    i = 0;
    while (i<(sizeof(scratch)-2) && (isalpha(*strtomap) || isdigit(*strtomap)))
      scratch[i++]=(char)toupper(*strtomap++);
    while (*strtomap && isspace(*strtomap))
      strtomap++;
    if (i && *strtomap == ':' || *strtomap == '=')
    {
      while (*strtomap && (*strtomap==':' || *strtomap=='=' || isspace(*strtomap)))
       strtomap++;
      if (isdigit(*strtomap) /* || *strtomap == '+' || *strtomap == '-' */)
      {
        if ( *strtomap == '0' )
          disabled = 1;
        while (isdigit(*strtomap) /* || *strtomap=='+' || *strtomap=='-' */)
          strtomap++;
      }
    }
    while (*strtomap && *strtomap!= ',' && *strtomap!=';' && !isspace(*strtomap))
    {
      if (i && i<(sizeof(scratch)-1))
        scratch[i++] = 'x'; /* make incomaptible to any contest name */
      strtomap++;
    }
    scratch[i]='\0';

    contestid = -1;

    if (i > 0)
    {
      for (i=0;i<CONTEST_COUNT;i++)
      {
        if ( strcmp( scratch, CliGetContestNameFromID(i) ) == 0 )
        {
          contestid = (int)i;
          break;
        }
      }
    }

    for (i=0; contestid != -1 && i< map_pos; i++)
    {
      if (contestid == (((int)(map[i])) & 0x7f))
        contestid = -1;
    }

    if (contestid != -1)
    {
      if (disabled)
        contestid |= 0x80;
      map[map_pos++]=(char)contestid;
    }

  } while ((map_pos < CONTEST_COUNT) && *strtomap );

  for (i=0;(map_pos < CONTEST_COUNT) && (i < CONTEST_COUNT);i++)
  {
    unsigned int n;
    contestid = (int)default_map[i];
    for (n=0; n<map_pos; n++ )
    {
      if (contestid == (((int)(map[n])) & 0x7f))
      {
        contestid = -1;
        break;
      }
    }
    if (contestid != -1) /* found contest not in map. i==its default prio */
    { /* now search for a contest *in* the map that has a default prio < i */
      /* that becomes the point at which we insert the missing contest */
      int inspos = -1; /* the position we insert at */
      for ( n = 0; (inspos == -1 && n < map_pos); n++ )
      {
        unsigned int thatprio;
        contestid = (((int)map[n]) & 0x7f); /* the contest sitting at pos n */
        /* find the default priority for the contest sitting at pos n */
        for (thatprio = 0; thatprio < CONTEST_COUNT; thatprio++ )
        {
          if (contestid == (int)default_map[thatprio] && thatprio > i)
          {                                 /* found it */
            inspos = (int)n;                /* this is the pos to insert at */
            break;
          }
        }
      }
      if (inspos == -1) /* didn't find it */
        map[map_pos++] = default_map[i]; /* so tack it on at the end */
      else
      {
        for ( n = (CONTEST_COUNT-1); n>((unsigned int)inspos); n--)
          map[n] = map[n-1];
        map[inspos] = default_map[i];
        map_pos++;
      }
    }
  }

//printf("\nresult order: %s\n", projectmap_expand( &map[0] ) );

  if (buf)
    memcpy((void *)buf, (void *)&map[0], CONTEST_COUNT );
  return map;
}

/* ------------------------------------------------------------------ */

int IsFilenameValid( const char *filename )
{
  if (!filename)
    return 0;
  while (*filename && isspace(*filename))
    filename++;
  return (*filename && strcmp( filename, "none" ) != 0); /* case sensitive */
}

int DoesFileExist( const char *filename )
{
  if ( !IsFilenameValid( filename ) )
    return 0;
  return ( access( GetFullPathForFilename( filename ), 0 ) == 0 );
}

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

/* ------------------------------------------------------------------ */

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

/* --------------------------------------------------------------------- */

const char *utilSetAppName(const char *newname)
{
  /*
   What is the official distributed.net name for this client?
   Used for argv[0] stuffing, banners, etc, etc.
   Intentionally obscure to thwart attempts to patch the binary.
   May be called with an override, but that functionality is AFAIK 
   no longer (as of Nov/2000) used.
  */
  static int initialized = -1;
  static char appname[32];
  if (newname != NULL)
  {
    /* this is bogus behavior that is never really used */
    unsigned int len;
    const char *sep = EXTN_SEP;
    while (*newname == ' ' || *newname == '\t')
      newname++;
    len = 0;
    while (*newname && *newname != ' ' && *newname != '\t' &&
           *newname != *sep && (len < (sizeof(appname)-1)))
      appname[len++] = (char)tolower(*newname++);
    if (len != 0) {
      appname[len] = '\0';
      initialized = 1;
    }
  }
  if (initialized <= 0) /* obfusciation 101 for argv[0] stuffing */
  {
    #if (CLIENT_CONTEST < 80)
    appname[0] = 'r'; appname[1] = 'c'; appname[2] = '5';
    appname[3] = 'd'; appname[4] = 'e'; appname[5] = 's';
    appname[6] = '\0';
    #else
    appname[0] = 'd'; appname[1] = 'n'; appname[2] = 'e';
    appname[3] = 't'; appname[4] = 'c'; appname[5] = '\0';
    #endif
    initialized = 1;
  }
  if (initialized > 0) /* always true */
    return (const char *)&appname[0];

  /* put the asciiz name here so the user has something to patch :) */
  #if (CLIENT_CONTEST < 80)
  return "rc5des";
  #else
  return "dnetc";
  #endif
}

const char *utilGetAppName(void)
{
  return utilSetAppName((const char *)0);
}

/* --------------------------------------------------------------------- */

#if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD) || \
   (CLIENT_OS == OS_NETBSD) || (CLIENT_OS == OS_OPENBSD)
#  include <dirent.h>        /* for direct read of /proc/ */
#elif (CLIENT_OS == OS_BEOS)
#  include <kernel/OS.h>     /* get_next_team_info() */
#  include <kernel/image.h>  /* get_next_image_info() */
#elif (CLIENT_OS == OS_WIN32)
#  include <tlhelp32.h>      /* toolhelp calls */
#  include <winperf.h>       /* perf on registry */
#endif
#ifdef __unix__
#  include <fcntl.h>
#endif /* __unix__ */

#if (CLIENT_OS == OS_WIN32)
/*
    This will work on Windows NT 3.x/4.x and Windows 2000, though
    on Win2k it's easier to use the Pshelper APIs.  This function
    ignores the leading path and file extension, if they are given.
*/
static int __utilGetPidUsingPerfCaps(const char *procname, long *pidlist, int maxnumpids )
{
  // Information on the format of the PerfCaps data is at:
  //    http://msdn.microsoft.com/library/psdk/pdh/perfdata_9feb.htm
  //    http://support.microsoft.com/support/kb/articles/Q119/1/63.asp
  int num_found = -1;
  static DWORD dwIndex_PROCESS = (DWORD) -1;
  static DWORD dwIndex_IDPROCESS = (DWORD) -1;
  unsigned int procname_baselen;


  // find and remove the pathname prefix.
  unsigned int basenamepos = strlen(procname);
  while (basenamepos > 0)
  {
    basenamepos--;
    if (procname[basenamepos] == '\\' ||
        procname[basenamepos] == '/' ||
        procname[basenamepos] == ':')
    {
      basenamepos++;
      break;
    }
  }
  procname += basenamepos;

  // strip procname of extension.
  if ((procname_baselen = strlen(procname)) > 3)
  {
    if (strcmpi(&procname[procname_baselen-4], ".com") == 0 ||
        strcmpi(&procname[procname_baselen-4], ".exe") == 0)
    {
      procname_baselen -= 4;
    }
  }

  
  
  //
  // On the first time through, we need to identify the name/index of the
  // performance counters that we are interested in later querying.
  // Since this information should not change until next reboot, we can
  // keep this information in static variables and reuse them next time.
  //
  if (dwIndex_PROCESS == (DWORD) -1 || dwIndex_IDPROCESS == (DWORD) -1)
  {
    HKEY hKeyIndex; 
    if (RegOpenKeyEx( HKEY_LOCAL_MACHINE,
         "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Perflib\\009",
          0, KEY_READ, &hKeyIndex ) == ERROR_SUCCESS)
    {
      // Get the size of the counter.
      DWORD dwBytes = 0;
      if (RegQueryValueEx( hKeyIndex, "Counter",
                   NULL, NULL, NULL,  &dwBytes ) == ERROR_SUCCESS)
      {
        char *pszBuffer = NULL;
        if (dwBytes != 0)
          pszBuffer = (char *)HeapAlloc( GetProcessHeap(),
                                       HEAP_ZERO_MEMORY, dwBytes );
        if (pszBuffer != NULL)
        {
          // Get the titles and counters. (REG_MULTI_SZ).  This registry
          // key is an alternating set of names and corresponding integers
          // that represent them.  We look for the ones we want by name.
          if (RegQueryValueEx( hKeyIndex, "Counter", NULL, NULL,
                    (LPBYTE)pszBuffer, &dwBytes ) == ERROR_SUCCESS )
          {
            DWORD pos = 0;

            while( dwIndex_PROCESS == (DWORD) -1 ||
                  dwIndex_IDPROCESS == (DWORD) -1 )
            {
              // save the current position in "valpos", and skip to the
              // end of the first string (the numeric value), while also
              // counting the length of it in "vallen".
              DWORD valpos = pos, vallen = 0;
              while (pos < dwBytes && pszBuffer[pos] != '\0')
              {
                pos++;
                vallen++;
              }
              if (pos >= dwBytes) break;

              // save the position of the start of the name in "cmpppos"
              // and then skip to the end of the string.
              DWORD cmppos = (++pos); /* skip the '\0' */
              while (pos < dwBytes && pszBuffer[pos] != '\0')
                pos++;
              if (pos >= dwBytes) break;
              pos++; /* skip the '\0' */

              // See if this value is one that we are looking for.              
              if (strcmpi( &pszBuffer[cmppos], "Process") == 0 )
              {
                if (dwIndex_PROCESS == (DWORD)-1)
                  dwIndex_PROCESS = (DWORD)atoi(&pszBuffer[valpos]);
              }
              else if (strcmpi( &pszBuffer[cmppos],"ID Process") == 0)
              {
                if (dwIndex_IDPROCESS == (DWORD)-1)
                  dwIndex_IDPROCESS = (DWORD)atoi(&pszBuffer[valpos]);
              }

            }
          }
          HeapFree( GetProcessHeap(), 0, (LPVOID)pszBuffer );
        }
      }
      RegCloseKey( hKeyIndex );
    }
  }


  //
  // Now that we have identified the performance counter that we are
  // interested in, actually do the query to retrieve the data
  // associated with it so that we can get the list of processes.
  //
  if (procname_baselen != 0 &&
      dwIndex_PROCESS != (DWORD)-1 &&
      dwIndex_IDPROCESS != (DWORD)-1)
  {
    static dwStartingBytes = 8192;
    DWORD dwBytes = dwStartingBytes;
    PPERF_DATA_BLOCK pdb;
    LONG lResult = ERROR_MORE_DATA;
    char szWorkbuffer[260];

    pdb = (PPERF_DATA_BLOCK)HeapAlloc( GetProcessHeap(),
                                       HEAP_ZERO_MEMORY, dwBytes );

    // Read in all object table/counters/instrecords for "Process".
    if (pdb != NULL)
    {
      // We call RegQueryValueEx in a loop because "If hKey specifies
      // HKEY_PERFORMANCE_DATA and the lpData buffer is too small,
      // RegQueryValueEx returns ERROR_MORE_DATA but lpcbData does not
      // return the required buffer size. This is because the size of
      // the performance data can change from one call to the next. In
      // this case, you must increase the buffer size and call
      // RegQueryValueEx again passing the updated buffer size in the
      // lpcbData parameter. Repeat this until the function succeeds"
      wsprintf(szWorkbuffer, "%d", dwIndex_PROCESS);
      do
      {
        // The SDK says "lpcbData does not return the required buffer
        // size" (see above), but NT4 has a bug, and returns lpcbData
        // set to HALF of what it was before. See...
        // http://support.microsoft.com/support/kb/articles/Q226/3/71.ASP

        DWORD dwLastSize = dwBytes; /* save last size (see note above)*/
        Sleep(10); /* let system update the data */
        lResult = RegQueryValueEx(HKEY_PERFORMANCE_DATA,
                            (LPTSTR)szWorkbuffer, NULL, NULL,
                            (LPBYTE)pdb, &dwBytes);
        if (lResult == ERROR_SUCCESS)
        {
          // Remember the required size as the starting buffer size for
          // next time (minus a little bit).  This allows us to more
          // quickly identify the right buffer size, while not forever
          // assuming that we need to use a buffer that large.
          dwStartingBytes = (dwBytes > 8192 ? dwBytes - 128 : dwBytes);
          break;
        }
        else if (lResult == ERROR_MORE_DATA)
        {
          LPVOID newmem;
          dwBytes = dwLastSize + 4096;
          if (dwBytes < dwLastSize) /* overflow */
            break;
          newmem = HeapReAlloc( GetProcessHeap(), HEAP_ZERO_MEMORY,
                                       (LPVOID)pdb, dwBytes );
          if (newmem == NULL) {
            // couldn't realloc.  free and abort.
            HeapFree( GetProcessHeap(), 0, (LPVOID) pdb);
            pdb = NULL;
            break;
          }
          pdb = (PPERF_DATA_BLOCK)newmem;
        }
      } while (lResult == ERROR_MORE_DATA);
    }


    // Now walk through the retrieved buffer and look for the processes
    // we are interested in identifying.
    if (pdb != NULL)
    {
      if (lResult == ERROR_SUCCESS)
      {
        LONG i, totalcount;
        DWORD ourpid = GetCurrentProcessId();
        PPERF_OBJECT_TYPE         pot;
        PPERF_COUNTER_DEFINITION  pcd;
        PPERF_INSTANCE_DEFINITION piddef;
        DWORD dwProcessIdOffset = 0;

        /* Get the PERF_OBJECT_TYPE. */
        pot = (PPERF_OBJECT_TYPE)(((PBYTE)pdb) + pdb->HeaderLength);

        /* Get the first counter definition. */
        pcd = (PPERF_COUNTER_DEFINITION)(((PBYTE)pot) + pot->HeaderLength);

        /* walk the counters to find the offset to the ProcessID */
        totalcount = pot->NumCounters;
        for ( i=0; i < totalcount; i++ )
        {
          /* get offset of the processID in the PERF_COUNTER_BLOCKs */
          if (pcd->CounterNameTitleIndex == dwIndex_IDPROCESS)
          {
            dwProcessIdOffset = pcd->CounterOffset;
            break;
          }
          pcd = ((PPERF_COUNTER_DEFINITION)(((PBYTE)pcd) + pcd->ByteLength));
        }

        /* Get the first process instance definition */
        piddef = (PPERF_INSTANCE_DEFINITION)(((PBYTE)pot) + pot->DefinitionLength);

//LogScreen("getpidlist 3: numpids = %u\n", pot->NumInstances );

        /* now walk the process definitions */
        totalcount = pot->NumInstances;
        for ( i = 0; i < totalcount; i++ )
        {
          PPERF_COUNTER_BLOCK pcb;
          char * foundname;
          DWORD thatpid;

          pcb = (PPERF_COUNTER_BLOCK) (((PBYTE)piddef) + piddef->ByteLength);
          thatpid = *((DWORD *) (((PBYTE)pcb) + dwProcessIdOffset));
          foundname = (char *) (((PBYTE)piddef) + piddef->NameOffset);
          if ( ((DWORD *)(((PBYTE)piddef) + piddef->NameLength)) == 0)
            foundname = NULL; /* name has zero length */

          /* we have all the data we need, skip to the next pid */
          piddef = (PPERF_INSTANCE_DEFINITION) (((PBYTE)pcb) + pcb->ByteLength);

//LogScreen("getpidlist 3a: got pid=0x%x\n", thatpid );

          if (num_found < 0) /* our enumerator is working */
          {
            num_found = 0;
          }
          if (foundname && thatpid != 0 && thatpid != ourpid)
          {
            /* convert the unicode name into ansi */
            dwBytes = (DWORD)WideCharToMultiByte( CP_ACP, 0,
                             (LPCWSTR)foundname, -1, szWorkbuffer,
                             sizeof(szWorkbuffer), NULL, NULL );
            foundname = szWorkbuffer;
            if (dwBytes > 0)
            {
              dwBytes--; /* WCTMB return value includes trailing null */
            }

//LogScreen("getpidlist 3b: got name='%s'\n", foundname );

            /* foundname and procname are both in ansi and are both
               just the basename (no path, no extension)
            */
            if ( dwBytes != 0 && procname_baselen == dwBytes &&
                 memicmp( foundname, procname, procname_baselen ) == 0 )
            {
              if (pidlist)
              {
                pidlist[num_found] = (long)thatpid;
              }
              num_found++;
              if (pidlist && num_found == maxnumpids)
              {
                break; /* for ( i = 0; i < pot->NumInstances; i++ ) */
              }
            }
          } /* if (thatpid != 0 && thatpid != ourpid) */
        } /* for ( i = 0; i < pot->NumInstances; i++ ) */
      } /* if RegQueryValueEx == ERROR_SUCCESS */

      HeapFree(GetProcessHeap(), 0, (LPVOID)pdb );
    } /* if (pdb != NULL) */

  } /* if (dwIndex_PROCESS && dwIndex_IDPROCESS && procname_base) */

  return num_found;
}

static int __utilGetPidUsingPshelper(const char *procname, long *pidlist, int maxnumpids )
{
  int num_found = -1;
  static HMODULE hKernel32 = NULL;

  // find the end of the pathname prefix.
  unsigned int basenamepos = strlen(procname);
  while (basenamepos > 0)
  {
    basenamepos--;
    if (procname[basenamepos] == '\\' ||
        procname[basenamepos] == '/' ||
        procname[basenamepos] == ':')
    {
      basenamepos++;
      break;
    }
  }

  // make sure we have a module handle.  Kernel32 should always
  // already be loaded for all Windows applications, so we only
  // need to get and keep an existing handle.
  if (hKernel32 == NULL) {
    hKernel32 = GetModuleHandle( "kernel32.dll" );
  }
  if (hKernel32 != NULL)
  {
    typedef HANDLE (WINAPI *CreateToolhelp32SnapshotT)(DWORD dwFlags, DWORD th32ProcessID);
    typedef BOOL (WINAPI *Process32FirstT)(HANDLE hSnapshot, LPPROCESSENTRY32 lppe);
    typedef BOOL (WINAPI *Process32NextT)(HANDLE hSnapshot, LPPROCESSENTRY32 lppe);
    static CreateToolhelp32SnapshotT fnCreateToolhelp32Snapshot =
                (CreateToolhelp32SnapshotT)
                GetProcAddress(hKernel32, "CreateToolhelp32Snapshot");
    static Process32FirstT fnProcess32First =
         (Process32FirstT)GetProcAddress(hKernel32, "Process32First");
    static Process32NextT fnProcess32Next =
           (Process32NextT)GetProcAddress(hKernel32, "Process32Next");

    if (fnCreateToolhelp32Snapshot != NULL &&
        fnProcess32First != NULL &&
        fnProcess32Next != NULL)
    {
      HANDLE hSnapshot = (*fnCreateToolhelp32Snapshot)(TH32CS_SNAPPROCESS,0);
      if (hSnapshot)
      {
        PROCESSENTRY32 pe;
        DWORD ourpid = GetCurrentProcessId();
        const char *p = procname;
        pe.dwSize = sizeof(pe);
        if ((*fnProcess32First)(hSnapshot, &pe))
        {
          unsigned int procnamelen = strlen( procname );
          unsigned int procsufxlen = 0;
          if (procnamelen > 3)
          {
            if (strcmpi( &procname[procnamelen-4],".com" )==0 ||
                strcmpi( &procname[procnamelen-4],".exe" )==0 )
            {
              procsufxlen = 3;
              procnamelen -=4;
            }
          }

          do
          {
            if (pe.szExeFile[0])
            {
              char *foundname = pe.szExeFile;
              DWORD thatpid = pe.th32ProcessID;
              if (num_found < 0)
              {
                num_found = 0;
              }
              if (thatpid != 0 && thatpid != ourpid)
              {
                unsigned int len;
                int cmpresult;

                /* if no path was provided, then allow
                   match if the rest is equal
                */
                if (basenamepos == 0)
                {
                  len = strlen( foundname );
                  while (len > 0)
                  {
                    len--;
                    if (foundname[len]=='\\' ||
                        foundname[len]=='/' ||
                        foundname[len]==':')
                    {
                      foundname += len+1;
                      break;
                    }
                  }
                }
                cmpresult = strcmpi( procname, foundname );

                /* if no extension was provided, then allow
                   match if basenames (sans-ext) are equal
                */
                if ( cmpresult && procsufxlen == 0)
                {
                  len = strlen( foundname );
                  if (len > 3)
                  {
                    if ( strcmpi( &foundname[len-4], ".exe" ) == 0
                      || strcmpi( &foundname[len-4], ".com" ) == 0 )
                    {
                      if ((len-4) == procnamelen)
                      {
                        cmpresult = memicmp( foundname,
                                         procname, procnamelen );
                      }
                    }
                  }
                }
                if ( cmpresult == 0 )
                {
                  if (pidlist)
                  {
                    pidlist[num_found] = (long)thatpid;
                  }
                  num_found++;
                  if (pidlist && num_found == maxnumpids)
                  {
                    break; /* while (next) */
                  }
                }
              }
            }
          } while ((*fnProcess32Next)(hSnapshot, &pe));
        }
      }
      CloseHandle(hSnapshot);
    }
  }
  return num_found;
}


#endif  /* CLIENT_OS == OS_WIN32 */


/*
    get list of pid's for procname. if procname has a path, then search
    for exactly that, else compare with basename. if pidlist is NULL
    or maxnumpids is 0, then return found count, else return number of
    pids now in list. On error return < 0.
*/
int utilGetPIDList( const char *procname, long *pidlist, int maxnumpids )
{
  int num_found = -1; /* assume all failed */

  if (!pidlist || maxnumpids < 1)
  {
    maxnumpids = 0;
    pidlist = (long *)0;
  }

  if (procname)
  {
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #if (CLIENT_OS == OS_BEOS)
    {
      team_info tInfo;
      team_id ourteam;
      int32 team;
      thread_info thisThread;

      /* procname is either a basename or a full path; determine which */
      int usefullpathcmp = (strchr( procname, '/' ) != ((char *)0));

      /* get our own team id, so that we can exclude it later */
      get_thread_info(find_thread(NULL), &thisThread);
      ourteam = thisThread.team;

      team = 0; /* begin enumeration here */
      while (get_next_team_info(&team, &tInfo) == B_OK)
      {
        if (num_found < 0) /* our scanner is working */
        {
          num_found = 0;
        }
        if (ourteam != tInfo.team) /* we don't include ourselves */
        {
          image_info iInfo;
          int32 image = 0;
          char * foundname;
          /* get the app binary's full path */
          get_next_image_info(tInfo.team, &image, &iInfo);

          foundname = iInfo.name; /* if procname is a basename, use only */
          if (!usefullpathcmp)    /* the basename from the app's path */
          {
            char *p = strrchr( foundname, '/' );
            if (p)
              foundname = p+1;
          }
          /* does the team name match? */
          if (strcmp( procname, foundname ) == 0)
          {
            if (pidlist) /* save the team number (pid) only if we have */
            {            /* someplace to save it to */
              pidlist[num_found] = (long)tInfo.team;
            }
            num_found++; /* track the number of pids found */
            if (pidlist && num_found == maxnumpids) /* done all? */
            {
              break; /* while (get_next_team_info() == B_OK) */
            }
          }
        }
      }
    }
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #elif (CLIENT_OS == OS_HPUX)
    {
      struct pst_status pst[10];
      pid_t ourpid = getpid();
      int usefullpathcmp = (strchr( procname, '/' ) != ((char *)0));
      int count, idx = 0; /* index within the context */
      num_found = -1; /* assume all failed */

      /* loop until count == 0, will occur all have been returned */
      while ((count = pstat_getproc(pst, sizeof(pst[0]),
                       (sizeof(pst)/sizeof(pst[0])), idx)) > 0)
      {
        int pspos;
        if (num_found < 0)
        {
          num_found = 0;
        }
        idx = pst[count-1].pst_idx + 1; /* start of next */
        for (pspos=0; pspos < count; pspos++)
        {
          //printf("pid: %d, cmd: %s\n",pst[pspos].pst_pid,pst[pspos].pst_ucomm);
          pid_t thatpid = (pid_t)pst[pspos].pst_pid;
          if (thatpid != ourpid)
          {
            char *foundname = ((char *)pst[pspos].pst_ucomm);
            if (!usefullpathcmp)
            {
              char *p = strrchr( foundname, '/' );
              if (p)
                foundname = p+1;
            }
            if (strcmp( procname, foundname ) == 0)
            {
              if (pidlist)
              {
                pidlist[num_found] = (long)thatpid;
              }
              num_found++;
              if (pidlist && num_found == maxnumpids)
              {
                break; /* for (pospos < count) */
              }
            }
          }
        }
        if (pidlist && num_found == maxnumpids)
        {
          break; /* while pstat_getproc() > 0 */
        }
      }
    }
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
    {
      if (*procname == '*' || *procname == '#')
      {
        /* match by window title or class name */

        char which = *procname++;
        while (*procname == ' ' || *procname == '\t')
          procname++;
        if (*procname)
        {
          HWND hwnd = NULL;
          if (which == '*') /* window name */
            hwnd = FindWindow( NULL, procname );
          else if (which == '#') /* class name */
            hwnd = FindWindow( procname, NULL );
          num_found = 0;
          if (hwnd != NULL)
          {
            DWORD pid = 0;
            #if (CLIENT_OS == OS_WIN32)
            if ( winGetVersion() >= 400 ) /* not win32s please */
            {
              if (GetWindowThreadProcessId( hwnd, &pid) == 0)
                pid = 0;
            }
            else
            #endif /* we use module handles for win16 */
            {
              #ifndef GWL_HINSTANCE /* GWW_HINSTANCE on win16 */
              #define GWL_HINSTANCE (-6)
              #endif
              HINSTANCE hinst = (HINSTANCE)GetWindowLong(hwnd, GWL_HINSTANCE);
              if (hinst)
              {
                char buffer[128];
                if (GetModuleFileName( hinst, buffer, sizeof(buffer)))
                  pid = (DWORD)GetModuleHandle( buffer );
              }
            }
            if (pid != NULL)
            {
              if (pidlist)
              {
                pidlist[num_found] = (long)pid;
              }
              num_found++;
            }
          } /* if (hwnd) */
        } /* if (*procname) */
      } /* if win or class name */
      else
      {
        /* match by process executable name */

        #if (CLIENT_OS == OS_WIN32)
        if (winGetVersion() >= 2000 && winGetVersion() < 2500) /* NT3/NT4 */
        {
          num_found = __utilGetPidUsingPerfCaps(procname, pidlist, maxnumpids);
        }
        else if (winGetVersion() >= 400) /* win9x/NT5, but not win32s please */
        {
          num_found = __utilGetPidUsingPshelper(procname, pidlist, maxnumpids);
        }
        else
        #endif
        {
          /* we should use taskfirst/tasknext, but thats a *bit* ]:)
             cowplicated from within an extender
          */
          HMODULE hMod = GetModuleHandle(procname);
          num_found = 0;
          if (hMod != NULL)
          {
            if (pidlist != NULL)
              pidlist[num_found] = (long)hMod;
            num_found++;
          }
        }
      }
    }
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #elif (CLIENT_OS == OS_NETWARE)
    {
      int nlmHandle = FindNLMHandle( procname );
      num_found = 0;
      if (nlmHandle)
      {
        if (pidlist)
          pidlist[num_found] = (long)nlmHandle;
        num_found++;
      }
    }
    #elif (defined(__unix__)) && (CLIENT_OS != OS_NEXTSTEP) && !defined(__EMX__)
    {
      char *p, *foundname;
      pid_t thatpid, ourpid = getpid();
      size_t linelen; char buffer[1024];
      int usefullpathcmp = (strchr( procname, '/' ) != ((char *)0));
      #if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD) || \
          (CLIENT_OS == OS_OPENBSD) || (CLIENT_OS == OS_NETBSD)
      {
        DIR *dirp = opendir("/proc");
        if (dirp)
        {
          struct dirent *dp;
          while ((dp = readdir(dirp)) != ((struct dirent *)0))
          {
            FILE *file;
            pid_t thatpid = (pid_t)atoi(dp->d_name);
            if (num_found < 0)
              num_found = 0;
            if (thatpid == 0 /* .,..,curproc,etc */ || thatpid == ourpid)
              continue;
            sprintf( buffer, "/proc/%s/cmdline", dp->d_name );
            if (( file = fopen( buffer, "r" ) ) == ((FILE *)0))
              continue; /* already died */
            linelen = fread( buffer, 1, sizeof(buffer), file );
            fclose( file );
            if (linelen != 0)
            {
              if (linelen == sizeof(buffer))
                linelen--;
              buffer[linelen] = '\0';
              //printf("%s: %60s\n", dp->d_name, buffer );
              foundname = &buffer[0];
              if (memcmp( foundname, "Name:", 5 ) == 0 ) /* linux status*/
                foundname += 5;
              while (*foundname && isspace(*foundname))
                foundname++;
              p = foundname;
              while (*p && !isspace(*p))
                p++;
              *p = '\0';
              if (!usefullpathcmp)
              {
                p = strrchr( foundname, '/' );
                if (p)
                  foundname = p+1;
              }
              if (strcmp( procname, foundname ) == 0)
              {
                if (pidlist)
                {
                  pidlist[num_found] = (long)thatpid;
                }
                num_found++;
                if (pidlist && num_found == maxnumpids)
                {
                  break; /* while readdir() */
                }
              }
            } /* if (len != 0) */
          } /* while readdir */
          closedir(dirp);
        }
      }
      #endif
      #if (CLIENT_OS != OS_LINUX) && (CLIENT_OS != OS_HPUX)
      {
        /* this part is only needed for operating systems that do not read /proc
           OR do not have a reliable method to set the name as read from /proc
           (as opposed to reading it from ps output)
        */
        FILE *file = ((FILE *)NULL);
        const char *pscmd = ((char *)NULL);
        #if (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_OPENBSD) || \
            (CLIENT_OS == OS_NETBSD) || (CLIENT_OS == OS_LINUX) || \
            (CLIENT_OS == OS_BSDOS) || (CLIENT_OS == OS_MACOSX)
        pscmd = "ps ax|awk '{print$1\" \"$5}' 2>/dev/null"; /* bsd, no -o */
        /* fbsd: "ps ax -o pid -o command 2>/dev/null"; */ /* bsd + -o ext */
        /* lnux: "ps ax --format pid,comm 2>/dev/null"; */ /* bsd + gnu -o */
        #elif (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS) || \
              (CLIENT_OS == OS_DEC_UNIX) || (CLIENT_OS == OS_AIX)
        pscmd = "/usr/bin/ps -ef -o pid -o comm 2>/dev/null"; /*svr4/posix*/
        #elif (CLIENT_OS == OS_IRIX) || (CLIENT_OS == OS_HPUX)
        pscmd = "/usr/bin/ps -e |awk '{print$1\" \"$4\" \"$5\" \"$6\" \"$7\" \"$8\" \"$9}' 2>/dev/null";
        #elif (CLIENT_OS == OS_NTO2)
        pscmd = "ps -A -o pid,comm 2>/dev/null";
        #else
        #error fixme: select an appropriate ps syntax (or use another method to get pidlist)
        #error "this part is only needed for OSs that do not have another way"
        #error "to get the pid+procname list (see linux/hpux above for alternative)"
        #endif
        file = (pscmd ? popen( pscmd, "r" ) : ((FILE *)NULL));
        if (num_found == 0) /* /proc read also failed/wasn't done? */
          num_found = -1;   /* assume spawn failed */
        if (file != ((FILE *)NULL))
        {
          int eof_count = 0;
          linelen = 0;
          while (file) /* dummy while */
          {
            int ch;
            if (( ch = fgetc( file ) ) == EOF )
            {
              if (ferror(file))
                break;
              if (linelen == 0)
              {
                if ((++eof_count) > 2)
                  break;
              }
              usleep(250000);
            }
            else if (ch == '\n')
            {
              eof_count = 0;
              if (linelen == 0)
                continue;
              if (linelen < (sizeof(buffer)-1)) /* otherwise line is unusable */
              {
                char *p;
                buffer[linelen]='\0';
                foundname = &buffer[0];
                while (*foundname && isspace(*foundname))
                  foundname++;
                p = foundname;
                while (isdigit(*foundname))
                  foundname++;
                if (p == foundname) /* no digits found. can't be pid */
                {
                  /* both linelen and buffer are about to be reset,
                     so we can misuse them here
                  */
                }
                else /* got a pid */
                {
                  *foundname++ = '\0';
                  thatpid = (pid_t)atol(p);
                  if (num_found < 0)
                    num_found = 0;
                  #if (CLIENT_OS == OS_BEOS)
                  if (fullpath[0])
                  {
                    foundname = fullpath;
                    if ((++threadindex) != 1)
                      thatpid = 0; /* ignore all but the main thread */
                  }
                  #endif
                  if (thatpid != 0 && thatpid != ourpid)
                  {
                    while (*foundname && isspace(*foundname))
                      foundname++;
                    p = foundname;
                    while (*p && !isspace(*p))
                      p++;
                    *p = '\0';
                    if (!usefullpathcmp)
                    {
                      p = strrchr( foundname, '/' );
                      if (p)
                        foundname = p+1;
                    }
                    /* printf("pid='%d' name='%s'\n",thatpid,foundname); */
                    if ( strcmp( procname, foundname ) == 0 )
                    {
                      if (num_found < 0)
                        num_found = 0;
                      else if (num_found > 0 && pidlist)
                      {
                        int whichpid;
                        for (whichpid = 0; whichpid < num_found; whichpid++)
                        {
                          if (((pid_t)(pidlist[whichpid])) == thatpid)
                          {
                            thatpid = 0;
                            break;
                          }
                        }
                      }
                      if (thatpid != 0)
                      {
                        if (pidlist)
                        {
                          pidlist[num_found] = (long)thatpid;
                        }
                        num_found++;
                        if (pidlist && num_found == maxnumpids)
                        {
                          break; /* while (file) */
                        }
                      }
                    }
                  } /* if (thatpid != 0 && thatpid != ourpid) */
                } /* have digits */
              } /* if (linelen < sizeof(buffer)-1) */
              linelen = 0; /* prepare for next line */
            } /* if (ch == '\n') */
            else
            {
              eof_count = 0;
              if (linelen < (sizeof(buffer)-1))
                buffer[linelen++] = ch;
            }
          } /* while (file) */
          pclose(file);
        } /* if (file != ((FILE *)NULL)) */
      }
      #endif /* spawn ps */
    }
    #endif /* #if (defined(__unix__)) */
  } /* if (procname) */

  return num_found;
}

