// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: util.cpp,v $
// Revision 1.1  1999/03/18 03:51:18  cyp
// Created.
//
//

#if (!defined(lint) && defined(__showids__))
const char *util_cpp(void) {
return "@(#)$Id: util.cpp,v 1.1 1999/03/18 03:51:18 cyp Exp $"; }
#endif

#include "baseincs.h" /* string.h */
#include "client.h"   /* CONTEST_COUNT, stub definition */
#include "clicdata.h" /* CliGetContestNameFromID() */
#include "util.h"     /* ourselves */

/* ------------------------------------------------------------------- */

const char *ogr_stubstr(const struct Stub *stub)
{
  static char buf[80];
  int i, len = (int)stub->length;
  
  if (len > 5) {
    sprintf(buf, "(error:%d/%d)", (int)stub->marks, len);
    return buf;
  }
  sprintf(buf, "%d/", (int)stub->marks);
  if (len == 0) {
    strcat(buf, "-");
    return buf;
  }
  for (i = 0; i < len; i++) {
    sprintf(&buf[strlen(buf)], "%d", stub->stub[i]);
    if (i+1 < len) {
      strcat(buf, "-");
    }
  }
  return buf;
}

/* ------------------------------------------------------------------- */

#if 0
char *strfproj( char *buffer, const char *fmt, FileEntry *data )
{
// Completed RC5 packet 68E0D85A:A0000000 (123456789 keys)
//          123:45:67:89 - [987654321 keys/s]
//"Completed one RC5 packet 00000000:00000000 (4*2^28 keys)\n"
//"%s - [%skeys/sec]\n"
// Completed RC5 packet 68E0D85A:A0000000 (123456789 keys)
//          123:45:67:89 - [987654321 keys/s]
// Completed OGR stub 22/1-3-5-7 (123456789 nodes)
//          123:45:67:89 - [987654321 nodes/s]
// Summary: 4 RC5 packets 12:34:56.78 - [234.56 Kkeys/s]" 


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
  static char default_map[CONTEST_COUNT] = { 1,2,0 };
  static char map[CONTEST_COUNT];
  unsigned int map_pos, i;
  int contestid;

  if (!strtomap || !*strtomap)
  {
    if (buf)
      memcpy((void *)buf, (void *)&default_map[0], CONTEST_COUNT );
    return default_map;
  }

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
    
  } while ((map_pos < sizeof(map)) && *strtomap );
  
  for (i=0;(map_pos < sizeof(map)) && (i < sizeof(default_map));i++)
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
    if (contestid != -1)
      map[map_pos++] = (char)contestid;
  }

  if (buf)
    memcpy((void *)buf, (void *)&map[0], CONTEST_COUNT );
  return map;
}

