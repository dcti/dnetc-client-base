/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *util_cpp(void) {
return "@(#)$Id: util.cpp,v 1.5.2.2 1999/04/24 07:37:28 jlawson Exp $"; }

#include "baseincs.h" /* string.h */
#include "client.h"   /* CONTEST_COUNT, stub definition */
#include "clicdata.h" /* CliGetContestNameFromID() */
#include "pathwork.h" /* GetFullPathForFilename() */
#include "util.h"     /* ourselves */

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

#if 0
char *strfproj( char *buffer, const char *fmt, WorkRecord *data )
{
//"Completed one RC5 packet 00000000:00000000 (4*2^28 keys)\n"
//          123:45:67:89 - [987654321 keys/sec]\n"
// Completed RC5 packet 68E0D85A:A0000000 (123456789 keys)
//          123:45:67:89 - [987654321 keys/s]
// Completed OGR stub 22/1-3-5-7 (123456789 nodes)
//          123:45:67:89 - [987654321 nodes/s]
// Summary: 4 RC5 packets 12:34:56.78 - [234.56 Kkeys/s]" 

%i == identifier (key # or stubstr)
%C == contest name (upper case)
%c == contest name (lower case)
%u == number of units (keys/nodes) in workrecord *data
%U == name of unit ("keys"/"nodes")
%t == time to complete WorkRecord 

}
#endif

/* ------------------------------------------------------------------- */

#define MAX_CONTEST_NAME_LEN 3

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
      strcat( buffer,":0" );
  }
  return buffer;
}

// --------------------------------------------------------------------------

const char *projectmap_build( char *buf, const char *strtomap )
{
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
