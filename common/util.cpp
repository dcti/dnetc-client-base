/*
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/
const char *util_cpp(void) {
return "@(#)$Id: util.cpp,v 1.34 2008/02/17 16:24:29 kakace Exp $"; }

//#define TRACE

#include "baseincs.h" /* string.h, time.h */
#include "version.h"  /* CLIENT_CONTEST */
#include "projdata.h" // general project data: ids, flags, states; names, ...
#include "client.h"   /* CONTEST_COUNT, stub definition */
#include "logstuff.h" /* Log() */
#include "clitime.h"  /* CliTimer(), Time()/(CliGetTimeString(NULL,1)) */
#include "cliident.h" /* CliIsDevelVersion() */
#include "clicdata.h" /* CliGetContestNameFromID() */
#include "pathwork.h" /* GetFullPathForFilename() */
#include "sleepdef.h" /* usleep() */
#include "util.h"     /* ourselves */
#define MAX_CONTEST_NAME_LEN 6
// PROJECT_NOT_HANDLED("is MAX_CONTEST_NAME_LEN still ok?")
// OK!

/* ------------------------------------------------------------------- */

static char __trace_tracing_filename[10] = {0};
void trace_setsrc( const char *filename )
{
  unsigned int i;
  register const char *q = (const char *)0;
  register const char *p = "";
  if (filename)
  {
    p = &filename[strlen(filename)];
    while (p > filename)
    {
      --p;
      #if (CLIENT_OS == OS_RISCOS)
      if (*p == '.')
      #else
      if (*p == ':' || *p == '/' || *p == '\\')
      #endif
      {
        p++;
        break;
      }
      if (*p == '.')
        q = p;
    }
  }
  strncpy(__trace_tracing_filename,p,sizeof(__trace_tracing_filename));
  i = ((q)?(q-p):(sizeof(__trace_tracing_filename)-1));
  __trace_tracing_filename[i] = '\0';
  i = strlen(__trace_tracing_filename);
  while (i < (sizeof(__trace_tracing_filename)-1))
    __trace_tracing_filename[i++] = ' ';
  __trace_tracing_filename[sizeof(__trace_tracing_filename)-1] = '\0';
  return;
}

void trace_out( int indlevel, const char *format, ... )
{
  static int indentlevel = -1; /* uninitialized */
  const char *tracefile = "trace"EXTN_SEP"out";
  int old_errno = errno;
  FILE *file;
  va_list arglist;
  va_start (arglist, format);

  if (indentlevel == -1) /* uninitialized */
  {
    unlink(tracefile); /* } both needed for */
    remove(tracefile); /* } some odd reason */
    indentlevel = 0;
  }

  if (indlevel < 0)
    indentlevel -= 2;
  file = fopen( tracefile, "a" );
  if (file)
  {
    char buffer[64];
    struct timeval tv;
    if (CliClock(&tv)!=0)
      tv.tv_sec = tv.tv_usec = 0;
    fprintf(file, "%05d.%03d: %s ", (int)tv.tv_sec, (int)(tv.tv_usec/1000),
                  __trace_tracing_filename );
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
  errno = old_errno;
  return;
}

/* ------------------------------------------------------------------- */

int utilCheckIfBetaExpired(int print_msg)
{
  if (CliIsDevelVersion()) /* cliident.cpp */
  {
    struct timeval expirationtime;
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
      static time_t last_seen = ((time_t)-1);
      struct timeval tv;
      if (CliClock(&tv) == 0)
      {
        if (last_seen == ((time_t)-1)) //let it through once
          last_seen = 0;
        else if (last_seen == 0 || tv.tv_sec < last_seen ||
                 (tv.tv_sec - last_seen) > 10*60)
        {
          expirationtime.tv_sec -= now;
          LogScreen("*** This BETA release expires in %s. ***\n",
            CliGetTimeString(&expirationtime,2) );
          last_seen = tv.tv_sec;
        }
      }
    }
  }
  return 0;
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
  /* current contest name regex is /[a-z][a-z0-9_-]/i */

  unsigned int cont_i;

  TRACE_OUT((+1, "utilScatter...(%s, ...)\n", oplist));

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
        else if (kwpos == 0 && !precspace) /* contest name */
        {
          buffer[len] = (char)toupper(c);
          havenondig = 1; /* don't care about (non-)digits in name */
          len++;
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
      TRACE_OUT((0, "gotone?: %s %d %d %d\n", buffer, contest, haveval1, value1));
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
  TRACE_OUT((-1, "utilScatter...()\n"));
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

// project order map + project state vec ==> string
const char *projectmap_expand( const int *map, const int *state_vec )
{
  static char buffer[(PROJECT_COUNT+1)*(MAX_PROJECT_NAME_LEN+3)];
  unsigned int i;

  if (!map)
    map = projectmap_build(NULL,NULL,NULL);

  buffer[0] = '\0';
  for (i = 0; i<PROJECT_COUNT; ++i)
  {
    int projectid = map[i];
    if (ProjectGetFlags(projectid) == PROJECT_UNSUPPORTED)
      continue;
    if (buffer[0] != '\0')
      strcat( buffer, "," );
    strcat( buffer, ProjectGetName(projectid) );
    if (state_vec && (state_vec[projectid] & PROJECTSTATE_USER_DISABLED))
      strcat( buffer,"=0" );
  }
  return buffer;
}

// --------------------------------------------------------------------------

// string or default ==> project order map + project state vec
const int* projectmap_build( int* buf, int* state, const char *strtomap )
{
  #if (CONTEST_COUNT != 7)
    #error PROJECT_NOT_HANDLED("static default load order map expects CONTEST_COUNT == 7")
  #endif
  // you must add _every_ project, obsoletes may come last
  static const int default_map[PROJECT_COUNT] = { RC5, DES, OGR, CSC, OGR_NG, OGR_P2, RC5_72 };
  static int default_map_checked = -1;
  static int map[PROJECT_COUNT];
  unsigned int map_pos, i;
  int projectid;

  if (default_map_checked == -1)
  {
    int i, p;

    for (p = 0; p < PROJECT_COUNT; ++p)
    {
      for (i = 0; i < PROJECT_COUNT; ++i)
        if (p == default_map[i])
          break;
      if (i >= PROJECT_COUNT)
        Log("default_map misses project %s (%d)\n", ProjectGetName(p), p);
    }
    default_map_checked = 1;
  }

  if (!strtomap || !*strtomap)
  {
    if (buf)
      memcpy((void *)buf, (const void *)default_map, sizeof(default_map));
    return default_map;
  }

  TRACE_OUT((0, "req order: %s\n", strtomap));

  /* parse strtomap */
  map_pos = 0;
  do
  {
    int disabled = 0;
    char scratch[2*MAX_PROJECT_NAME_LEN+2];
    while (*strtomap && !isalpha(*strtomap) && !isdigit(*strtomap))
      strtomap++;
    i = 0;
    while (i<(sizeof(scratch)-2) && (isalpha(*strtomap) || isdigit(*strtomap)
                                     || *strtomap=='-' || *strtomap=='_' ))
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
        scratch[i++] = '#'; /* make incomaptible to any contest name */
      strtomap++;
    }
    scratch[i]='\0';

    projectid = -1;

    if (i > 0)
    {
      for (i = 0; i < PROJECT_COUNT; i++)
      {
        if ( strcmp( scratch, ProjectGetName(i) ) == 0 )
        {
          projectid = (int)i;
          break;
        }
      }
    }

    for (i=0; projectid != -1 && i < map_pos; i++)
    {
      if (projectid == map[i])
        projectid = -1;
    }

    if (projectid != -1)
    {
      if (state)
      {
        if (disabled)
          state[projectid] |= PROJECTSTATE_USER_DISABLED;
        else
          state[projectid] &= ~PROJECTSTATE_USER_DISABLED;
      }
      map[map_pos++] = projectid;
    }

  } while ((map_pos < PROJECT_COUNT) && *strtomap );

  /* insert omitted projects */
  for (i = 0; (map_pos < PROJECT_COUNT) && (i < PROJECT_COUNT); i++)
  {
    unsigned int n;
    projectid = (int)default_map[i];
    for (n=0; n<map_pos; n++ )
    {
      if (projectid == map[n])
      {
        projectid = -1;
        break;
      }
    }
    if (projectid != -1) /* found project not in map. i==its default prio */
    { /* now search for a project *in* the map that has a default prio < i */
      /* that becomes the point at which we insert the missing contest */
      int inspos = -1; /* the position we insert at */
      for ( n = 0; (inspos == -1 && n < map_pos); n++ )
      {
        unsigned int thatprio;
        projectid = map[n]; /* the contest sitting at pos n */
        /* find the default priority for the contest sitting at pos n */
        for (thatprio = 0; thatprio < PROJECT_COUNT; thatprio++ )
        {
          if (projectid == default_map[thatprio] && thatprio > i)
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
        for ( n = (PROJECT_COUNT-1); n>((unsigned int)inspos); n--)
          map[n] = map[n-1];
        map[inspos] = default_map[i];
        map_pos++;
      }
    }
  }

  TRACE_OUT((0, "result order: %s\n", projectmap_expand( map, state ) ));

  /* double check map */
  {
    int i, p;

    for (p = 0; p < PROJECT_COUNT; ++p)
    {
      for (i = 0; i < PROJECT_COUNT; ++i)
        if (p == map[i])
          break;
      if (i >= PROJECT_COUNT)
        Log("built map misses project %s (%d)\n", ProjectGetName(p), p);
    }
  }

  if (buf)
    memcpy((void *)buf, (void *)map, sizeof(map));
  return map;
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
  #if (CLIENT_CONTEST < 80)
  /* put the asciiz name here so the user has something to patch :) */
  static char appname[32] = {'r','c','5','d','e','s','\0'};
  #else
  /* put the asciiz name here so the user has something to patch :) */
  static char appname[32] = {'d','n','e','t','c','\0'};
  #endif

  if (newname != NULL)
  {
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
  return (const char *)&appname[0];

}

const char *utilGetAppName(void)
{
  return utilSetAppName((const char *)0);
}

/* --------------------------------------------------------------------- */

#if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD) || \
      (CLIENT_OS == OS_NETBSD) || (CLIENT_OS == OS_OPENBSD) || \
      (CLIENT_OS == OS_PS2LINUX)
  #include <dirent.h>         // for direct read of /proc/
#elif (CLIENT_OS == OS_BEOS)
  #include <kernel/OS.h>      // get_next_team_info()
  #include <kernel/image.h>   // get_next_image_info()
#elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <tlhelp32.h> /* toolhlp32 structures and function prototypes */
#elif (CLIENT_OS == OS_WIN16)
  #include <windows.h>
#endif
#ifdef __unix__
  #include <fcntl.h>
#endif /* __unix__ */

/*
    get list of pid's for procname. if pidlist is NULL or maxnumpids is 0,
    then return found count, else return number of pids now in list.
    On error return < 0.
*/
int utilGetPIDList( const char *procname, long *pidlist, int maxnumpids )
{
  int num_found = -1; /* assume all failed */

  if (!pidlist || maxnumpids < 1)
  {
    maxnumpids = 0;
    pidlist = NULL;
  }

  if (procname != NULL)
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
    #elif (CLIENT_OS == OS_OS2)
    {
      num_found = os2GetPIDList(procname, pidlist, maxnumpids);
    }
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #elif (CLIENT_OS == OS_WIN64) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
    {
      if (*procname == '*' || *procname == '#')
      {
        /* match by window title ('*') or class name ('#') */
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
          if (hwnd != NULL) /* found! */
          {
            DWORD pid = 0;
            #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
            if ( winGetVersion() >= 400 ) /* not win32s please */
            {                             /* use real pid */
              if (GetWindowThreadProcessId( hwnd, &pid ) == 0)
                pid = 0;
            }
            else
            #endif
            {
              /* Return module handles instead of pids for win16 or win32s. */
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

            if (pid != 0) /* have a match to store */
            {
              if (pidlist)
              {
                pidlist[num_found] = (long)pid;
              }
              num_found++;
            }
          }
        }
      } /* find by window or window class */
      #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
      else if (winGetVersion() >= 400) /* not win32s please */
      {
        /* calls to CreateToolhelp32Snapshot(), Process32First() and
           Process32Next() go to platforms/win32cli/w32snapp.c which
           has stubs into toolhlp32.dll and emulation for toolhelp
           when running on NT3/4.
        */
        HANDLE hSnapshot;

        hSnapshot = CreateToolhelp32Snapshot( TH32CS_SNAPPROCESS, 0 );
        if (hSnapshot)
        {
          PROCESSENTRY32 pe;
          pe.dwSize = sizeof(pe);

          if (Process32First(hSnapshot, &pe))
          {
            DWORD ourownpid = GetCurrentProcessId();
            unsigned int basenamepos, basenamelen, suffixlen;

            /* Name matching: if any component (path,name,extension) of
              the found name or the template is not available, then those
              components are treated as lexical wildcards (match anything).
            */

            suffixlen = 0;
            basenamelen = basenamepos = strlen(procname);
            while (basenamepos > 0)
            {
              basenamepos--;
              if (procname[basenamepos] == '\\' ||
                 procname[basenamepos] == '/' ||
                 procname[basenamepos] == ':')
              {
                basenamepos++;
                basenamelen-=basenamepos;
                break;
              }
            }
            if (basenamelen > 3)
            {
              if (strcmpi( &procname[(basenamepos+basenamelen)-4],".com" )==0 ||
                 strcmpi( &procname[(basenamepos+basenamelen)-4],".exe" )==0 )
              {
                suffixlen = 3;
                basenamelen -=4;
              }
            }

            do
            {
//LogScreen("ps: %p => '%s'\n", pe.th32ProcessID, pe.szExeFile);
              if (pe.szExeFile[0])
              {
                /* our enumerator is working */
                if (num_found < 0)
                {
                  num_found = 0;
                }
                if (pe.th32ProcessID != ourownpid)
                {
                  int cmpresult = -1;
                  const char *foundname = pe.szExeFile;
                  const char *templname = procname;
                  unsigned int len = strlen( foundname );
                  unsigned int fbasenamelen = len;

                  while (len > 0)
                  {
                    len--;
                    if (foundname[len]=='\\' ||
                        foundname[len]=='/' ||
                        foundname[len]==':')
                    {
                      len++;
                      fbasenamelen-=len;
                      break;
                    }
                  }

                  /*if no path is available on one side then skip
                    the path (if it exists) on the other side
                  */
                  if (basenamepos == 0) /* no path in template */
                  {
                    foundname += len; /* then skip dir in foundname */
                  }
                  else if (len == 0) /*dir in templ, but no dir in foundname */
                  {
                    templname += basenamepos; /* then skip dir in template */
                  }
                  cmpresult = strcmpi( templname, foundname );

                  if ( cmpresult )
                  {
                    /* if either template OR foundname have no suffix, (but
                       not both, which will have been checked above) then
                       allow a match if the basenames (sans-suffix) are equal.
                    */
                    unsigned int fsuffixlen = 0;
                    if (fbasenamelen > 3)
                    {
                      /* Don't be tempted to try to optimize away
                         extension checks even when the data is from
                         performance counters- although it might
                         *APPEAR* that pe.szExeFile never has an extension
                         (when using performance counters), that is not
                         always so. -cyp
                      */
                      if ( strcmpi( &foundname[fbasenamelen-4], ".exe" ) == 0
                        || strcmpi( &foundname[fbasenamelen-4], ".com" ) == 0 )
                      {
                        fsuffixlen = 3;
                        fbasenamelen -= 4;
                      }
                    }
                    if (suffixlen != fsuffixlen && basenamelen == fbasenamelen)
                    {
                      cmpresult = memicmp( foundname, templname, basenamelen );
                    }
                  }

                  if (cmpresult == 0)
                  {
                    if (pidlist)
                    {
                      pidlist[num_found] = (long)pe.th32ProcessID;
                    }
                    num_found++;
                    if (pidlist && num_found == maxnumpids) /* hit limit? */
                    {
                      break; /* do {} while Process32Next() */
                    }
                  }
                } /* if (pe.th32ProcessID != ourownpid) */
              } /* if (pe.szExeFile[0]) */
            } while (Process32Next(hSnapshot, &pe));
          } /* if (Process32First(hSnapshot, &pe)) */
          CloseHandle( hSnapshot );
        } /* if (hSnapshot) */
      } /* else if (winGetVersion() >= 400) */
      #endif /* #if (CLIENT_OS == OS_WIN32) */
      else
      {
        /* we should use taskfirst/tasknext, but thats a *bit*
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
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #elif (CLIENT_OS == OS_NETWARE)
    {
      char namebuf[64];
      unsigned int blen, bpos;
      int need_suffix;

      blen = bpos = strlen(procname);
      while (bpos > 0 && procname[bpos-1]!='/' &&
             procname[bpos-1]!='\\' && procname[bpos-1]!=':')
        bpos--;
      blen -= bpos;
      need_suffix = 1;
      if (blen > 3)
        need_suffix = (procname[(bpos+blen)-4] != '.');
      if (bpos || need_suffix)
      {
        if (!need_suffix)
          procname += bpos;
        else if ((blen+5) >= sizeof(namebuf))
          procname = NULL;
        else
          procname = strcat(strcpy(namebuf,&procname[bpos]),".nlm");
      }
      if (procname)
      {
        int nlmHandle = FindNLMHandle( (char *)procname );
        num_found = 0;
        if (nlmHandle)
        {
          if (pidlist)
            pidlist[num_found] = (long)nlmHandle;
          num_found++;
        }
      }
    }
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #elif (CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_MORPHOS)
    {
      num_found = 0;
      long taskptr;

      #if !defined(__OS3PPC__)
      /* 68K / OS4 / MorphOS */
      taskptr = (long)FindTask((STRPTR)procname);
      #elif !defined(__POWERUP__)
      /* WarpOS */
      taskptr = (long)FindTaskPPC((char *)procname);
      #else
      /* PowerUp */
      taskptr = (long)PPCFindTask((char *)procname);
      #endif

      if (taskptr)
      {
         if (pidlist)
            pidlist[num_found] = taskptr;
         num_found++;
      }
    }
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #elif defined(__unix__)
    {
      char *foundname;
      pid_t ourpid = getpid();
      size_t linelen; char buffer[1024];
      int usefullpathcmp = (strchr( procname, '/' ) != ((char *)0));

      #if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD) || \
          (CLIENT_OS == OS_OPENBSD) || (CLIENT_OS == OS_NETBSD) || \
          (CLIENT_OS == OS_PS2LINUX)
      {
        DIR *dirp = opendir("/proc");

        if (dirp) {
          pid_t thatpid;
          struct dirent *dp;

          while ((dp = readdir(dirp)) != ((struct dirent *)0)) {
            FILE *file;
            thatpid = (pid_t)atoi(dp->d_name);
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
              char *p;

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
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      #if (CLIENT_OS != OS_LINUX) && (CLIENT_OS != OS_HPUX) && (CLIENT_OS != OS_PS2LINUX)
      {
        /* this part is only needed for operating systems that do not read /proc
           OR do not have a reliable method to set the name as read from /proc
           (as opposed to reading it from ps output)
        */
        FILE *file = ((FILE *)NULL);
        const char *pscmd = ((char *)NULL);
        #if (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_OPENBSD) || \
            (CLIENT_OS == OS_NETBSD) || (CLIENT_OS == OS_LINUX) || \
            (CLIENT_OS == OS_BSDOS) || (CLIENT_OS == OS_PS2LINUX)
        pscmd = "ps axw|awk '{print$1\" \"$5}' 2>/dev/null"; /* bsd, no -o */
        /* fbsd: "ps ax -o pid -o command 2>/dev/null"; */ /* bsd + -o ext */
        /* lnux: "ps ax --format pid,comm 2>/dev/null"; */ /* bsd + gnu -o */
        #elif (CLIENT_OS == OS_MACOSX)
        /* White spaces in directory/file names make parsing very error */
        /* prone. The workaround is to ask 'ps' to output the pid and   */
        /* executable name only, so that we can deal with white spaces  */
        /* in program names (quite frequent under Mac OS X)             */
        pscmd = "ps acxw -o pid,command 2>/dev/null";
        #elif (CLIENT_OS == OS_NEXTSTEP)
        /* NeXTstep porduces spaces in process status columns like
         * 26513 p1 SW    0:01 -bash (bash)
         * 26542 p1 R N  32:52 ./dnetc */
        pscmd = "ps axw|sed \"s/ [RUSITHPD][W >][N< ]//\"|awk '{print$1\" \"$4}' 2>/dev/null";
        #elif (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS) || \
              (CLIENT_OS == OS_DEC_UNIX) || (CLIENT_OS == OS_AIX)
        pscmd = "/usr/bin/ps -ef -o pid -o comm 2>/dev/null"; /*svr4/posix*/
	#elif (CLIENT_OS == OS_DYNIX)
	pscmd = "/bin/ps -ef -o pid -o comm 2>/dev/null";     /*svr4/posix*/
        #elif (CLIENT_OS == OS_IRIX) || (CLIENT_OS == OS_HPUX)
        pscmd = "/usr/bin/ps -e |awk '{print$1\" \"$4\" \"$5\" \"$6\" \"$7\" \"$8\" \"$9}' 2>/dev/null";
        #elif (CLIENT_OS == OS_QNX)
          #if defined(__QNXNTO__) /* neutrino */
          pscmd = "ps -A -o pid,comm 2>/dev/null";
          #else
          pscmd = "ps -A -F\"%p %c\" 2>/dev/null";
          #endif
        #elif (CLIENT_OS == OS_SCO)
          pscmd = "/bin/ps -A -o pid,comm 2>/dev/null";
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
                  pid_t thatpid;

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
                    #if (CLIENT_OS == OS_MACOSX)
                    /* Skip to the end of line, and accept white spaces */
                    while (*p)
                      p++;
                    #else
                    while (*p && !isspace(*p))
                      p++;
                    #endif
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
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #endif /* #if (defined(__unix__)) */
  } /* if (procname) */

  return num_found;
}

/* ------------------------------------------------------------------------ */

static int __utilIsUserIDAValidEmailAddress(const char *userid)
{
  const char *c = userid, *domain = NULL;
  int username_length = 0;
  int domain_length = 0, domain_has_non_digits = 0, subdomains = 0;
  int last_was_dash = 0, last_was_dot = 0;
  int rfc_violation = 0;

  if (!c || !*c)
    return 0;

  // username part
  // ToDo: check for '\' escaped characters, if someone complains :)
  for (; *c; ++c)
  {
    if (*c == '@')
    {
      break;
    }
    if (*c <= 32) // control or space
      return 0;

    if (*c == '\"' || *c == ',' || *c == '\\' ||
        *c == '<' || *c == '>' || *c == ';' ||
        *c == '{' || *c == '}' ||
        !isprint((unsigned char)*c))
      {
        // although some of these characters are allowed by RFC, we
        // enforce restrictions on these to reduce problems with log
        // file delimiters and unprintable characters.
        return 0;
      }

    // for now, every other char is allowed in the username part
    ++username_length;
  }

  if (*c == '@')
  {
    domain = ++c;
    last_was_dot = 1; // the '@' is a separator, too
    subdomains = 1;
  }

  // domain part
  for (; *c; ++c)
  {
    if (*c == '.')
    {
      if (last_was_dot)
        return 0;
      if (last_was_dash)
        ++rfc_violation;
      ++subdomains;
    }
    else if (*c == '-')
    {
      if (last_was_dot)
        ++rfc_violation;
      domain_has_non_digits = 1;
    }
    else if ('0' <= *c && *c <= '9')
    {
      // digits are ok
    }
    else if (('a' <= *c && *c <= 'z') || ('A' <= *c && *c <= 'Z'))
    {
      // letters too
      domain_has_non_digits = 1;
    }
    else
    {
      return 0; // unexpected character
    }

    ++domain_length;
    last_was_dot  = (*c == '.');
    last_was_dash = (*c == '-');
  }
  if (last_was_dash || last_was_dot || rfc_violation)
    return 0;

  if (!domain_has_non_digits)
  {
    // use of IP addresses is discouraged
    if (subdomains != 4)
      return 0;
  }

  if (username_length == 0 || domain_length == 0 || subdomains < 2)
    return 0;

  return 1;
}

/* XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX where X is a hex digit,
   may be enclosed in braces */
static int __utilIsUserIDAValidGUID(const char *userid)
{
  int has_braces = 0;
  int i;
  const char *c = userid;

  if (!c || !*c)
    return 0;

  if (*c == '{')
  {
    has_braces = 1;
    ++c;
  }

  for (i = 0; i < 32 && *c; ++i, ++c)
  {
    if (i == 8 || i == 12 || i == 16 || i == 20) // there is a dash before this digit
    {
      if (*c++ != '-')
        return 0; // no dash here ?
    }
    if (!*c || !(('0' <= *c && *c <= '9') || ('a' <= *c && *c <= 'f') || ('A' <= *c && *c <= 'F')))
      return 0; // it's not a hexdigit
  }
  if (i != 32)
    return 0; // found less than 32 digits

  if (has_braces)
  {
    if (*c != '}')
      return 0; // there was an opening brace but no closing one
    ++c;
  }

  if (*c) // still more characters?
    return 0;

  return 1; // all tests passed :)
}

/* returns 1 = valid, 0 = invalid */
int utilIsUserIDValid(const char *userid)
{
  return __utilIsUserIDAValidEmailAddress(userid) ||
         __utilIsUserIDAValidGUID(userid);
}

/* ------------------------------------------------------------------------ */

char * strncpyz(char *dest, const char *src, int n)
{
  strncpy(dest, src, n);
  if (dest && n>0)
    dest[n - 1] = '\0';
  return dest;
}
