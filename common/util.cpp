/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *util_cpp(void) {
return "@(#)$Id: util.cpp,v 1.13 1999/10/11 18:47:20 cyp Exp $"; }

#include "baseincs.h" /* string.h, time.h */
#include "version.h"  /* CLIENT_CONTEST */
#include "client.h"   /* CONTEST_COUNT, stub definition */
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
      fwrite((void *)((indlevel < 0)?("end: "):("beg: ")), 1, 5, file );
    vfprintf(file, format, arglist);
    fflush( file );
    fclose( file );
  }
  if (indlevel > 0)
    indentlevel += 2;
  return;
}  

/* ------------------------------------------------------------------- */

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

const char *utilGatherOptionArraysToList( int *table1, int *table2 )
{
  static char buffer[(CONTEST_COUNT+1)*(MAX_CONTEST_NAME_LEN+10)];
  unsigned int contest, pos = 0;
  const char *delim = "";
  buffer[0] = '\0';
  for (contest = 0; contest < CONTEST_COUNT; contest++)
  {
    /* HACK: OGR doesn't a member in the preferred_blocksize/coretype arrays */
    if (table2 || (contest != OGR)) /* HACK! OGR only for threshold arrays */
    {
      const char *p = CliGetContestNameFromID(contest);
      if (p)
      {
        char single[(MAX_CONTEST_NAME_LEN+1+(sizeof(int)*3)+1+(sizeof(int)*3)+1)];
        unsigned int len = 0;
        if (table2)
          len = sprintf(single,"%s%s=%d:%d",delim, p,
                        (int)table1[contest],(int)table2[contest]);
        else
          len = sprintf(single,"%s%s=%d",delim, p, (int)table1[contest] );
        if (len <= (MAX_CONTEST_NAME_LEN+10))
        {
          strcpy( &buffer[pos], single );
          pos += len;
          delim = ",";
        }
      }
    }
  }
  return (const char *)&buffer[0];
}

int utilScatterOptionListToArrays( const char *oplist, 
                                   int *table1, int *table2, int defaultval )
{
  unsigned int cont_i;
  for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
  {
    table1[cont_i] = defaultval;
    if (table2)
      table2[cont_i] = defaultval;
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
      int havenondig = 0, haveval1 = 0, haveval2 = 0;
      int value1 = defaultval, value2 = defaultval;
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
  static int initialized = -1;
  static char appname[32];
  unsigned int len;
  if (newname)
  {
    const char *sep = EXTN_SEP;
    while (*newname == ' ' || *newname == '\t')
      newname++;
    len = 0;
    while (*newname && *newname != ' ' && *newname != '\t' && 
           *newname != *sep && (len < (sizeof(appname)-1)))
      appname[len++] = (char)tolower(*newname++);
    appname[len] = '\0';
    if (len && initialized < 0)
      initialized = 1;
  }
  #if (CLIENT_OS == OS_NETWARE)
  if (initialized < 0)
  {
    initialized = 0; /* protect against recursion */
    strncpy( appname, nwCliGetNLMBaseName(), sizeof(appname));
    appname[sizeof(appname)-1] = '\0';
    for (len = 0; appname[len] && (len < (sizeof(appname)-1));len++)
      appname[len] = (char)tolower(appname[len]);
    if (len)
      initialized = 1;
  }  
  #endif
  if (initialized > 0)
    return (const char *)&appname[0];
  /* --- */
  #if defined(__unix__) /* obfusciation 101 for argv[0] stuffing */
  if (initialized <= 0) /* dummy if to suppress compiler warning */
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
    return (const char *)&appname[0];
  }  
  #endif
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

