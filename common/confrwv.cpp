/*
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/
const char *confrwv_cpp(void) {
return "@(#)$Id: confrwv.cpp,v 1.60.2.50.4.1 2001/03/23 20:56:35 andreasb Exp $"; }

//#define TRACE

#include "cputypes.h"
#include "client.h"    // Client class
#include "baseincs.h"  // atoi() etc
#include "iniread.h"   // [Get|Write]Profile[Int|String]()
#include "pathwork.h"  // GetFullPathForFilename()
#include "util.h"      // projectmap_*() and trace
#include "base64.h"    // base64_[en|de]code()
#include "clicdata.h"  // CliGetContestNameFromID()
#include "clitime.h"   // CliClock()
#include "triggers.h"  // OverrideNextConffileChangeTrigger()
#include "confrwv.h"   // Ourselves

/* ------------------------------------------------------------------------ */

static const char *OPTION_SECTION   = "parameters";/*all obsolete except "id"*/
static const char *OPTSECT_NET      = "networking";
static const char *OPTSECT_BUFFERS  = "buffers";
static const char *OPTSECT_MISC     = "misc";
static const char *OPTSECT_LOG      = "logging";
static const char *OPTSECT_CPU      = "processor-usage";
static const char *OPTSECT_TRIGGERS = "triggers";
static const char *OPTSECT_DISPLAY  = "display";

static const char *DEFAULT_EXITFLAGFILENAME = "exitrc5"EXTN_SEP"now";
#define OLD_PREFERREDBLOCKSIZE_DEFAULT 30 /* for converting old .inis */

/* ------------------------------------------------------------------------ */

// Get .ini's section name for a given contest id. On error, returns NULL.
static const char *__getprojsectname( unsigned int ci )
{
  if (ci < CONTEST_COUNT)
  {
    #if 1
      #if (CONTEST_COUNT != 5)
        #error fixme: (CONTEST_COUNT != 5) static table
      #endif
      static const char *sectnames[CONTEST_COUNT]={"rc5","des","ogr1_old","csc","ogr"};
      return sectnames[ci];
    #else
      const char *cname = CliGetContestNameFromID(ci);
      if (cname)
      {
        static char cont_name[16];
        ci=0;
        while (*cname && ci<(sizeof(cont_name)-1))
          cont_name[ci++] = (char)tolower((char)(*cname++));
        cont_name[ci]='\0';
        return &cont_name[0];
      }
    #endif
  }
  return ((const char *)0);
}

/* ------------------------------------------------------------------------ */

// these options are special and reflect "authoritative" information from
// a proxy. They are in separate functions to "protect" them.
static void ConfigReadUniversalNews( Client *client, const char *fn )
{
  if (!fn) fn = GetFullPathForFilename( client->inifilename );
  //client->rc564closed is necessary to supress generation of randoms
  client->rc564closed = GetPrivateProfileIntB(__getprojsectname(RC5), "closed", 0, fn );
  client->scheduledupdatetime = GetPrivateProfileIntB(OPTSECT_NET,"scheduledupdatetime", 0, fn);
  return;  
}
//ConfigWriteServerNews() may *only* be called from buffer update 
//(otherwise the ini will end up with data that is not authoritative)
void ConfigWriteUniversalNews( Client *client )
{
  if (client->stopiniio == 0 && client->nodiskbuffers == 0)
  {
    const char *fn = GetFullPathForFilename( client->inifilename );
    if ( access( fn, 0 ) == 0 ) /* we also do not write these settings */
    {                           /* if the .ini doesn't already exist */
      int did_write = 0;

      /* rc5 closed? */
      {
        const char *rc5_sect = __getprojsectname(RC5);
        int rc564closed = 0;
        if (GetPrivateProfileIntB(rc5_sect, "closed", 0, fn ))
          rc564closed = 1;
        if (client->rc564closed)
          client->rc564closed = 1;
        if (rc564closed != client->rc564closed)
        {
          WritePrivateProfileStringB(rc5_sect,"closed",(client->rc564closed)?("yes"):(NULL), fn );
          did_write = 1;
        }
      }            

      /* got a new scheduled update time? */
      if (client->scheduledupdatetime != GetPrivateProfileIntB(OPTSECT_NET,"scheduledupdatetime", 0, fn))
      {
        if (client->scheduledupdatetime == 0)
          WritePrivateProfileStringB(OPTSECT_NET, "scheduledupdatetime", NULL, fn );
        else
          WritePrivateProfileIntB(OPTSECT_NET, "scheduledupdatetime", client->scheduledupdatetime, fn );
        did_write = 1;
      }

      if (did_write)
      {
        /* prevent our own writes from kicking off a restart */
        OverrideNextConffileChangeTrigger();
      }  

    } /* if ( access( fn, 0 ) == 0 ) */
  } /* if (client->stopiniio == 0 && client->nodiskbuffers == 0) */
  return;
}

/* ------------------------------------------------------------------------ */

// Reads or writes a hostname and port in "hostname:port" format.
// aswrite - is zero to read, non-zero to write
// fn - filename of the ini file.
// sect - section within the ini file that holds the hostname.
// opt - option name within the ini file section.
// hostname/hnamelen/port - the hostname and port to read or write.
//
// if port is non-zero, but hostname is blank then writes "*:nnn"
// Always returns 0.

static int _readwrite_hostname_and_port( int aswrite, const char *fn,
                                         const char *sect, const char *opt,
                                         char *hostname, unsigned int hnamelen,
                                         int *port, int multiple )
{
  char buffer[128];
  char hostbuf[sizeof(buffer)]; /* needed for multiple parse */
  unsigned int len;
  int pos = 0;

  if (!aswrite)
  {
    // when reading, hostname and port are assumed to be
    // pre-loaded with default values
    if (hostname && hnamelen &&
        GetPrivateProfileStringB( sect, opt, "", buffer, sizeof(buffer), fn ))
    {
      int hostcount = 0, foundport = 0;
      char *p = buffer;
      while (*p)
      {
        int portnum = 0;
        len = 0;
        while (*p && (isspace(*p) || *p == ';' || *p == ','))
          p++;
        while (*p && !isspace(*p) && *p!=';' && *p!=',')
          hostbuf[len++] = *p++;
        hostbuf[len] = '\0';
        if (len)
        {
          pos = 0;
          while (hostbuf[pos] && hostbuf[pos]!=':')
            pos++;
          if (hostbuf[pos] == ':')
          {
            portnum = atoi(&hostbuf[pos+1]);
            hostbuf[pos]='\0';
            if (portnum <= 0 || portnum >= 0xffff)
              portnum = 0;
            len = pos;  
          }
          if (foundport == 0)
            foundport = portnum;
        }  
        if (len && strcmp(hostbuf,"*")!=0 && strcmp(hostbuf,"auto")!=0)
        {
          if (portnum)
          {
            sprintf(&hostbuf[len],":%d", portnum);
            len = strlen(hostbuf);
          }    
          if ((len + 3) >= hnamelen)
            break; /* no more space for hostnames */
          if (hostcount)
          {
            *hostname++ = ';';
            hnamelen--;
          }
          strcpy(hostname, hostbuf);
          hostname += len;
          hnamelen -= len;
          hostcount++;
          if (!multiple)
            break;
        }
      } /* while (*p) */
      if (foundport && port)
        *port = foundport;
    }               
    pos = 0;
  }
  else /* aswrite */
  {
    int foundport = 0;
    if (port)    
    {
      foundport = *port;
      if (foundport < 0 || foundport > 0xffff)
        foundport = 0;
    }
    if (!hostname)
      hostname = "";
    if (!foundport) /* no port to tack on, just sanitize */
    {
      len = 0;
      while (*hostname && isspace(*hostname))
        hostname++;
      len = 0;
      while (*hostname && len < (sizeof(hostbuf)-1))
      {
        if (*hostname == ',' || *hostname == ';')
        {
          if (!multiple)
            break;
          hostbuf[len++] = ';';
        }  
        else if (!isspace(*hostname))
          hostbuf[len++] = *hostname;
        else if (!multiple)
          break;  
        hostname++;
      }
      hostbuf[len] = '\0';
      hostname = hostbuf;        
    }
    else /* tack port on to first hostname (if it doesn't already have one) */
    {    
      char *p = hostname;
      len = 0; 
      while (*p && (isspace(*p) || *p == ';' || *p == ','))
        p++;
      while (*p && !isspace(*p) && *p!=';' && *p!=',')
        hostbuf[len++] = *p++;
      hostbuf[len] = '\0';
      if (!len)
      {
        strcpy(hostbuf,"*");
        len = 1;
      }
      else
      {
        pos = 0;
        while (hostbuf[pos] && hostbuf[pos]!=':')
          pos++;
        if (hostbuf[pos] == ':')
          len = 0;
      }      
      if (len) /* doesn't already have a port */
      {
        sprintf(&hostbuf[len],":%d", foundport);
        if (multiple)
        {
          len = strlen(hostbuf);
          while (*p && (isspace(*p) || *p == ';' || *p == ','))
            p++;
          if (*p)
          {
            strcat(hostbuf,";");
            len++;
            while (*p && len < (sizeof(hostbuf)-1))
            {
              if (*p == ',')
                hostbuf[len++] = ';';
              else if (!isspace(*p))
                hostbuf[len++] = *p;
              p++;
            }
            hostbuf[len] = '\0';
          }    
        }  
        hostname = hostbuf;
      }  
    } /* if need to tack on port to first name */     
TRACE_OUT((0,"host:'%s'\n",hostname));
    pos = 0;
    if (*hostname || GetPrivateProfileStringB( sect, opt, "", buffer, 2, fn ))
      pos = (!WritePrivateProfileStringB( sect, opt, hostname, fn));
  }
  return 0;
}

/* ------------------------------------------------------------------------ */

// Reads the firewall host/port, type, encoding, and authentication values.
//
// aswrite - should be zero to indicate reading, or non-zero to write.
// fn - filename of the ini file.
// client - reference to the client structure being loaded/saved.
//
// Returns the new state of uuehttpmode.

static int _readwrite_fwallstuff(int aswrite, const char *fn, Client *client)
{
  char buffer[128];
  char scratch[2];
  char *that;

  if (!aswrite) /* careful. 'aswrite' may change */
  {
    TRACE_OUT((+1,"_readwrite_fwallstuff(asread)\n"));

    if (GetPrivateProfileStringB( OPTSECT_NET, "firewall-type", "", buffer, sizeof(buffer), fn )
     || GetPrivateProfileStringB( OPTSECT_NET, "firewall-host", "", scratch, sizeof(scratch), fn )
     || GetPrivateProfileStringB( OPTSECT_NET, "firewall-auth", "", scratch, sizeof(scratch), fn )
     || GetPrivateProfileStringB( OPTSECT_NET, "encoding", "", scratch, sizeof(scratch), fn ))
    {
      //
      // Read and parse the "new" style firewall settings.
      int i;

      TRACE_OUT((+1,"newform: _readwrite_fwallstuff(asread)\n"));

      client->uuehttpmode = 0;
      for (i=0; buffer[i]; i++)
        buffer[i] = (char)tolower(buffer[i]);
      if ( strcmp( buffer, "socks4" ) == 0 ||
           strcmp( buffer, "socks" ) == 0 )
        client->uuehttpmode = 4;
      else if ( strcmp( buffer, "socks5" ) == 0 )
        client->uuehttpmode = 5;
      else
      {
        if ( strcmp( buffer, "http" ) == 0)
          client->uuehttpmode = 2;
        if (GetPrivateProfileStringB( OPTSECT_NET, "encoding", "", buffer, sizeof(buffer), fn ))
        {
          for (i=0; buffer[i]; i++)
            buffer[i] = (char)tolower(buffer[i]);
          if ( strcmp( buffer, "uue" ) == 0 )
            client->uuehttpmode |= 1;
        }
      }
      client->httpproxy[0] = '\0';
      GetPrivateProfileStringB( OPTSECT_NET, "firewall-host", "", client->httpproxy, sizeof(client->httpproxy), fn );

      client->httpid[0] = '\0';
      if (GetPrivateProfileStringB( OPTSECT_NET, "firewall-auth", "", buffer, sizeof(buffer), fn ))
      {
        if (base64_decode( client->httpid, buffer,
                       sizeof(client->httpid), strlen(buffer) ) < 0)
        {
          TRACE_OUT((0,"new decode parity err=\"%s\"\n", client->httpid ));
          client->httpid[0] = '\0'; /* parity errors */
        }
        client->httpid[sizeof(client->httpid)-1] = '\0';
      }
      TRACE_OUT((0,"new uuehttpmode=%d\n",client->uuehttpmode));
      TRACE_OUT((0,"new httpproxy=\"%s\"\n",client->httpproxy));
      TRACE_OUT((0,"new httpport=%d\n",client->httpport));
      TRACE_OUT((0,"new httpid=\"%s\"\n",client->httpid));

      TRACE_OUT((-1,"newform: _readwrite_fwallstuff(asread)\n"));
    }
    else
    {
      //
      // Read and translate the "old" style firewall settings.
      //
      TRACE_OUT((+1,"oldform: _readwrite_fwallstuff(asread)\n"));

      if (GetPrivateProfileStringB( OPTION_SECTION, "uuehttpmode", "", scratch, sizeof(scratch), fn ))
      {
        client->uuehttpmode = GetPrivateProfileIntB( OPTION_SECTION, "uuehttpmode", client->uuehttpmode, fn );
        if (client->uuehttpmode != 0)
          aswrite = 1; /* key exists, need rewrite */
      }
      TRACE_OUT((0,"old uuehttpmode=%d\n",client->uuehttpmode));
      if (GetPrivateProfileStringB( OPTION_SECTION, "httpproxy", client->httpproxy, client->httpproxy, sizeof(client->httpproxy), fn ))
      {
        if (client->httpproxy[0])
        {
          aswrite = 1;  /* key exists, need rewrite */
          that = client->httpproxy;
          while (*that && *that != ':')
            that++;
          if (*that != ':') /* no embedded port # */
          {
            int i = GetPrivateProfileIntB( OPTION_SECTION, "httpport", 0, fn );
            if (i > 0 && i < 0xffff)
              sprintf(&(client->httpproxy[strlen(client->httpproxy)]),":%d",i);
          }
        }
      }
      TRACE_OUT((0,"old httpproxy=\"%s\"\n",client->httpproxy));
      if (GetPrivateProfileStringB( OPTION_SECTION, "httpid", client->httpid, client->httpid, sizeof(client->httpid), fn ))
      {
        if (client->httpid[0])
          aswrite = 1;  /* key exists, need rewrite */
      }
      TRACE_OUT((0,"old httpid=\"%s\"\n",client->httpid));

      if (client->httpid[0] &&
          (client->uuehttpmode == 2 || client->uuehttpmode == 3))
      {
        if (base64_decode( buffer, client->httpid,
                   sizeof(buffer), strlen(client->httpid) ) < 0 )
        {
          TRACE_OUT((0,"oldconv parity err=\"%s\"\n", client->httpid ));
          client->httpid[0] = '\0';
        }
        else
        {
          strncpy( client->httpid, buffer, sizeof(client->httpid) );
          client->httpid[sizeof(client->httpid)-1]='\0';
        }
      }
      TRACE_OUT((-1,"oldform: _readwrite_fwallstuff(asread). rewrite?=%d\n",aswrite));
    }
    TRACE_OUT((-1,"_readwrite_fwallstuff(asread)\n"));
  }
  if (aswrite)
  {
    TRACE_OUT((+1,"_readwrite_fwallstuff(aswrite)\n"));

    WritePrivateProfileStringB( OPTION_SECTION, "uuehttpmode", NULL, fn);
    WritePrivateProfileStringB( OPTION_SECTION, "httpproxy", NULL, fn);
    WritePrivateProfileStringB( OPTION_SECTION, "httpport", NULL, fn);
    WritePrivateProfileStringB( OPTION_SECTION, "httpid", NULL, fn);

    that = "";
    if (client->uuehttpmode == 1 || client->uuehttpmode == 3)
      that = "uue";
    if (*that || GetPrivateProfileStringB( OPTSECT_NET, "encoding", "", scratch, sizeof(scratch), fn ))
      WritePrivateProfileStringB( OPTSECT_NET, "encoding", that, fn);

    that = "";
    if (client->uuehttpmode == 2 || client->uuehttpmode == 3)
      that = "http";
    else if (client->uuehttpmode == 4)
      that = "socks4";
    else if (client->uuehttpmode == 5)
      that = "socks5";
    TRACE_OUT((0,"new set fwall-type=%s\n",that));
    if (*that || GetPrivateProfileStringB( OPTSECT_NET, "firewall-type", "", scratch, sizeof(scratch), fn ))
      WritePrivateProfileStringB( OPTSECT_NET, "firewall-type", that, fn);

    if (client->httpproxy[0] || GetPrivateProfileStringB( OPTSECT_NET, "firewall-host", "", scratch, sizeof(scratch), fn ))
      WritePrivateProfileStringB( OPTSECT_NET, "firewall-host", client->httpproxy, fn );

    buffer[0] = '\0';
    if (client->httpid[0])
    {
      TRACE_OUT((0,"pre-write httpid=\"%s\"\n", client->httpid ));
      if (base64_encode(buffer, client->httpid, sizeof(buffer),
                     strlen(client->httpid)) < 0)
      {
        TRACE_OUT((0,"new encode length err=\"%s\"\n", buffer ));
        buffer[0]='\0';
      }
      buffer[sizeof(buffer)-1] = '\0';
      TRACE_OUT((0,"post-write httpid=\"%s\"\n", buffer ));
    }
    if (buffer[0] ||
       GetPrivateProfileStringB( OPTSECT_NET, "firewall-auth", "", scratch, sizeof(scratch), fn ))
      WritePrivateProfileStringB( OPTSECT_NET, "firewall-auth", buffer, fn);

    TRACE_OUT((-1,"_readwrite_fwallstuff(aswrite)\n"));
  }
  return client->uuehttpmode;
}

/* ------------------------------------------------------------------------ */

// Convert a "time" string in the form "[hh:]mm[:ss]"
// into an equivalent number of seconds or < 0 on error.
//
// If the argument 'oldstyle_hours_compat' is non-zero, then the textual
// input may optionally be in the format of "hh.hhh", representing a decimal
// number of hours.

static int __parse_timestring(const char *source, int oldstyle_hours_compat )
{
  int hms[3];
  int colcount = 0, fieldlen = 0;

  hms[0] = hms[1] = hms[2] = 0;
  while (*source && isspace(*source))
    source++;

  while (*source && (*source == ':' || isdigit(*source)))
  {
    if (*source == ':')
    {
      fieldlen = 0;
      if ((++colcount) > 2)
        return -1;
    }
    else
    {
      fieldlen++;
      if (fieldlen > 7 || (colcount != 0 && fieldlen > 2))
        return -1;
      hms[colcount] *= 10;
      hms[colcount] += (char)(*source - '0');
    }
    source++;
  }
  if (*source == '.' && colcount == 0 && oldstyle_hours_compat)
  {
    /* "hours" was once-upon-a-time literally hours with a decimal */
    /* point; at first with six frac digits!; go! sprintf(%f)!!! */
    /* and later "fixed" to two (still duh-ble) */
    /* fwiw, we support that historical curiosity */
    colcount++; /* we have a seconds field */
    source++;
    if (isdigit(*source) && isdigit(source[1]))
    {                                /* fill in seconds */
      hms[1] = ((((*source-'0')*10)+(source[1]-'0'))*60)/100; /* frac of hours */
      source += 2;
    }
    while (*source == '0')
      source++;
  }

  while (*source && isspace(*source))
    source++;
  if (*source)
    return -1;

  if (colcount == 0) /* only minutes */
    return hms[0] * 60;

  if (hms[0] > ((0x7fffffffl)/(60*60)) || hms[1] > 59 || hms[2] > 59)
    return -1;

  return (((hms[0] * 60)+hms[1])*60)+hms[2]; /* hh:mm or hh:mm:ss */
}

/* ------------------------------------------------------------------------ */

static int __readwrite_minutes(const char *sect, const char *keyname,
                               const int *write_value, int defaultval,
                               const char *fn, int apply_elapsed_bias )
{
  char buffer[64];
  if (write_value)
  {
    int minutes = *write_value;
    if (minutes < 0)
      minutes = defaultval;
    if (minutes != defaultval || 
        GetPrivateProfileStringB(sect,keyname, "",buffer,sizeof(buffer),fn))  
    {
      sprintf(buffer,"%u:%02u",((unsigned)minutes/60),((unsigned)minutes%60));
      return WritePrivateProfileStringB( sect, keyname, buffer, fn );
    }
    return 0;
  }
  if (GetPrivateProfileStringB(sect,keyname, "",buffer,sizeof(buffer),fn))
  {
    int seconds = __parse_timestring( buffer, 0 );
    if (seconds >= 0)
    {
      if (apply_elapsed_bias && seconds > 0) /* subtract time already elapsed */
      {
        struct timeval tv;
        if (CliClock(&tv)==0) 
        {
          if (((unsigned int)tv.tv_sec) >= ((unsigned int)seconds))
            seconds = 1;
          else
            seconds -= tv.tv_sec;
        }
      }
      return ((seconds+59) / 60); /* we only want minutes */
    }  
  }
  return defaultval;
}      
    
/* ------------------------------------------------------------------------ */

//determine the default value for "autofindkeyserver" based on the keyserver
//name. Effective only if the "autofindkeyserver" entry doesn't exist.

static int __getautofinddefault(const char *hostname)
{
  char buffer[64]; char sig[] = "distributed.net";
  unsigned int ui = 0;
  while (ui<sizeof(buffer) && hostname[ui] && hostname[ui]!=':') 
  {
    buffer[ui]=(char)tolower(hostname[ui]);
    ui++;
  }
  if (ui < sizeof(buffer))
  {
    buffer[ui] = '\0';
    if (ui == 0)
      return 1;
    if (ui == 1 && buffer[0] == '*')
      return 1;
    if (strcmp(buffer,"auto")==0 || strcmp(buffer,"(auto)")==0)
      return 1;  //one config version accidentally wrote "auto" out
    if (ui >= (sizeof( sig ) - 1) &&
                 strcmp( &buffer[(ui-(sizeof( sig )-1))], sig )==0)
    {
      if (ui == (sizeof(sig)-1))
        return 1; /* plain "distributed.net" */
      else if (buffer[(ui-(sizeof( sig )))] == '.') /* [*].distributed.net */
        return 1; /* make the hostname require reentry (once) */
    }
  }
  return 0;
}

/* ------------------------------------------------------------------------ */

/* WritePrivateProfileInt() writes unsigned "numbers", so this widget is 
   to stop -1 from appearing as 18446744073709551615 (the client doesn't 
   care of course since it gets read back correctly, but it worries users. 
   See bug #1601). Only used from __remapObsoleteParameters.
*/   
static int _WritePrivateProfile_sINT( const char *sect, const char *key,
                                      int val, const char *filename )
{
  char buffer[(sizeof(int)+2)*3]; sprintf(buffer,"%d",val);
  return WritePrivateProfileStringB( sect, key, buffer, filename );
}                                     

// Convert old ini settings to new (if new doesn't already exist/is not empty).
// Returns 0 on succes, or negative if any failures occurred.
static int __remapObsoleteParameters( Client *client, const char *fn )
{
  char buffer[128];
  char *p;
  unsigned int ui, cont_i;
  int i, modfail = 0;

  TRACE_OUT((+1,"__remapObsoleteParameters()\n"));

  /* ----------------- OPTSECT_LOG ----------------- */

  if (!GetPrivateProfileStringB( OPTSECT_LOG, "mail-log-max", "", buffer, sizeof(buffer), fn ))
  {
    i = GetPrivateProfileIntB( OPTION_SECTION, "messagelen", 0, fn );
    if (i > 0)
    {
      client->messagelen = i;
      modfail += (!WritePrivateProfileIntB( OPTSECT_LOG, "mail-log-max", i, fn));
      TRACE_OUT((0,"remapped mail-log-max (%d)\n", modfail));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_LOG, "mail-log-from", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "smtpfrom", "", buffer, sizeof(buffer), fn ))
    {
      strncpy( client->smtpfrom, buffer, sizeof(client->smtpfrom) );
      client->smtpfrom[sizeof(client->smtpfrom)-1] = '\0';
      modfail += (!WritePrivateProfileStringB( OPTSECT_LOG, "mail-log-from", buffer, fn ));
      TRACE_OUT((0,"remapped mail-log-from (%d)\n", modfail));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_LOG, "mail-log-dest", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "smtpdest", "", buffer, sizeof(buffer), fn ))
    {
      strncpy( client->smtpdest, buffer, sizeof(client->smtpdest) );
      client->smtpdest[sizeof(client->smtpdest)-1] = '\0';
      modfail += (!WritePrivateProfileStringB( OPTSECT_LOG, "mail-log-dest", buffer, fn ));
      TRACE_OUT((0,"remapped mail-log-dest (%d)\n", modfail));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_LOG, "mail-log-via", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "smtpsrvr", "", buffer, sizeof(buffer)-10, fn ))
    {
      i = GetPrivateProfileIntB( OPTION_SECTION, "smtpport", 0, fn );
      if (i != 25 && i > 0 && i < 0xffff && !strchr(buffer,':'))
        sprintf(&(buffer[strlen(buffer)]),":%d",i);
      strncpy( client->smtpsrvr, buffer, sizeof(client->smtpsrvr) );
      client->smtpsrvr[sizeof(client->smtpsrvr)-1]='\0';
      modfail += (!WritePrivateProfileStringB( OPTSECT_LOG, "mail-log-via", buffer, fn ));
      TRACE_OUT((0,"remapped smtp hostname/port (%d)\n", modfail));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_LOG, "log-file", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "logname", "", buffer, sizeof(buffer), fn ))
    {
      strncpy( client->logname, buffer, sizeof(client->logname) );
      client->logname[sizeof(client->logname)-1] = '\0';
      modfail += (!WritePrivateProfileStringB( OPTSECT_LOG, "log-file", buffer, fn ));
      TRACE_OUT((0,"remapped log-file (%d)\n", modfail));
    }
  }

  /* ----------------- project options ----------------- */

  /* we no longer save the random prefix in the .ini (all done elsewhere) */
  WritePrivateProfileStringB( __getprojsectname(RC5), "randomprefix", NULL, fn );

  #if (CLIENT_CPU != CPU_ALPHA) && (CLIENT_CPU != CPU_68K) && (CLIENT_CPU != CPU_ARM)
  /* don't have RC5 cputype->coretype mapping for Alpha or m68k or arm */
  if (!GetPrivateProfileStringB( __getprojsectname(RC5), "core", "", buffer, sizeof(buffer), fn ))
  {
    if ((i = GetPrivateProfileIntB(OPTION_SECTION, "cputype", -1, fn ))!=-1)
    {
      client->coretypes[RC5] = i;
      modfail += (!_WritePrivateProfile_sINT( __getprojsectname(RC5), "core", i, fn));
      TRACE_OUT((0,"remapped rc5 core (%d)\n", modfail));
    }
  }
  #endif

  /* PREFERRED-BLOCKSIZE MUST COME BEFORE THRESHOLDS */
  if (!GetPrivateProfileStringB( __getprojsectname(RC5), "preferred-blocksize", "", buffer, sizeof(buffer), fn )
   && !GetPrivateProfileStringB( __getprojsectname(DES), "preferred-blocksize", "", buffer, sizeof(buffer), fn ))
  {
    if ((i = GetPrivateProfileIntB(OPTION_SECTION, "preferredblocksize", -1, fn ))!=-1)
    {
      if (i >= PREFERREDBLOCKSIZE_MIN &&
          i <= PREFERREDBLOCKSIZE_MAX &&
          i != OLD_PREFERREDBLOCKSIZE_DEFAULT) /* 30 */
      {
        client->preferred_blocksize[RC5] = i;
        client->preferred_blocksize[DES] = i;
        modfail += (!_WritePrivateProfile_sINT( __getprojsectname(RC5), "preferred-blocksize", i, fn));
        modfail += (!_WritePrivateProfile_sINT( __getprojsectname(DES), "preferred-blocksize", i, fn));
      }
    }
  }
  TRACE_OUT((0,"remapping 5 (%d)\n", modfail));
  {
    int thresholdsdone = 0;
    for (cont_i=0; cont_i < CONTEST_COUNT; cont_i++)
    {
      /* if we have _any_ key already in the new format, then we're done */
      if (GetPrivateProfileStringB( __getprojsectname(cont_i), "fetch-workunit-threshold", "", buffer, sizeof(buffer), fn )
       #if !defined(NO_OUTBUFFER_THRESHOLDS)          
       || GetPrivateProfileStringB( __getprojsectname(cont_i), "flush-workunit-threshold", "", buffer, sizeof(buffer), fn )
       #endif
       )
      {
        thresholdsdone = 1;
        break;
      }
    }
    /* convert post-2.8000 and pre-2.8000 format simultaneously */
    for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++ )
    {
      const char *cont_sect = __getprojsectname(cont_i);
      if (!thresholdsdone)
      {
        int oldstyle_inout[2];
        oldstyle_inout[0] = GetPrivateProfileIntB( cont_sect, "fetch-threshold", -123, fn );
        oldstyle_inout[1] = GetPrivateProfileIntB( cont_sect, "flush-threshold", -123, fn );
        if (oldstyle_inout[0] != -123 || oldstyle_inout[1] != -123)
        {                             /* post 2.8000. */
          /* delete them here */
          WritePrivateProfileStringB( cont_sect, "fetch-threshold", NULL, fn );
          WritePrivateProfileStringB( cont_sect, "flush-threshold", NULL, fn );
        }
        else if ((cont_i == RC5) && /* pre-2.8000 */
          GetPrivateProfileStringB( OPTION_SECTION, "threshold", "", buffer, sizeof(buffer), fn ))
        {
          if ((i = atoi( buffer )) > 0)
          {
            oldstyle_inout[0] = i;
            if ((p = strchr( buffer, ':' )) != NULL)
              oldstyle_inout[1] = atoi( p+1 );
          }
          /* deleted at end of function */
        }
        if (oldstyle_inout[0] > 0)
        {
          int multiplier = 1;
          switch (cont_i)
          {
             case RC5:
             case DES:
             case CSC:
               multiplier = GetPrivateProfileIntB( cont_sect,
                            "preferred-blocksize", OLD_PREFERREDBLOCKSIZE_DEFAULT, fn );
               if ( multiplier < 1)
                 multiplier = OLD_PREFERREDBLOCKSIZE_DEFAULT;
               else if ( multiplier < PREFERREDBLOCKSIZE_MIN)
                 multiplier = PREFERREDBLOCKSIZE_MIN;
               else if (multiplier > PREFERREDBLOCKSIZE_MAX)
                 multiplier = OLD_PREFERREDBLOCKSIZE_DEFAULT;
               multiplier -= (PREFERREDBLOCKSIZE_MIN-1);
               break;
             case OGR:
               multiplier = 1;
               break;
          }
          if (oldstyle_inout[0] > 500) /* our old limit */
            oldstyle_inout[0] = 500;
          if (oldstyle_inout[0] != 10) /* oldstyle default */
          {
            /* convert using preferred blocksize */
            client->inthreshold[cont_i] =
                oldstyle_inout[0] * multiplier;
            modfail += (!_WritePrivateProfile_sINT( cont_sect, "fetch-workunit-threshold", client->inthreshold[cont_i], fn));
          }
          #if !defined(NO_OUTBUFFER_THRESHOLDS)          
          if (oldstyle_inout[1] > 0)
          {
            /* outthreshold >= inthreshold implied 'inthreshold decides' */
            if (oldstyle_inout[1] < oldstyle_inout[0]) /* old meaning */
            {
              /* convert using preferred blocksize */
              client->outthreshold[cont_i] =
                oldstyle_inout[1] * multiplier;
              modfail += (!_WritePrivateProfile_sINT( cont_sect, "flush-workunit-threshold", client->outthreshold[cont_i], fn));
            }
          }
          #endif
        }
      } /* if !donethresholds */
      WritePrivateProfileStringB( cont_sect, "fetch-threshold", NULL, fn );
      WritePrivateProfileStringB( cont_sect, "flush-threshold", NULL, fn );
    } /* for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++ ) */
  }

  /* ----------------- OPTSECT_BUFFERS ----------------- */
  TRACE_OUT((0,"remapping 6 (%d)\n", modfail));

  if (!GetPrivateProfileStringB( OPTSECT_BUFFERS, "checkpoint-filename", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "checkpointfile", "", buffer, sizeof(buffer), fn ))
    {
      strncpy(client->checkpoint_file, buffer, sizeof(client->checkpoint_file));
      client->checkpoint_file[sizeof(client->checkpoint_file)-1] = '\0';
      modfail += (!WritePrivateProfileStringB( OPTSECT_BUFFERS, "checkpoint-filename", client->checkpoint_file, fn ));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_BUFFERS, "buffer-only-in-memory", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileIntB( OPTION_SECTION, "nodisk", 0, fn ))
    {
      client->nodiskbuffers = 1;
      modfail += (!WritePrivateProfileStringB( OPTSECT_BUFFERS, "buffer-only-in-memory", "yes", fn ));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_BUFFERS, "frequent-threshold-checks", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileIntB( OPTION_SECTION, "lurkonly", 0, fn ) ||
        GetPrivateProfileIntB( OPTION_SECTION, "lurk", 0, fn ))
    { //used to set frequent=1 in the ini if lurk/lurkonly
      //this is a local implication in ClientRun now
      client->connectoften = 0;
    }  
    else if (GetPrivateProfileIntB( OPTION_SECTION, "frequent", 0, fn ))
    {
      client->connectoften = 3; /* both fetch and flush */
      modfail += (!WritePrivateProfileStringB( OPTSECT_BUFFERS, "frequent-threshold-checks", "3", fn ));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_BUFFERS, "buffer-file-basename", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "in", "", buffer, sizeof(buffer), fn ))
    {
      int basenameoffset = GetFilenameBaseOffset( buffer );
      const char *suffixsep = EXTN_SEP;
      p = strrchr( &buffer[basenameoffset], *suffixsep );
      if (p)
        *p = '\0';
      if (strcmp( buffer, "buff-in" ) != 0)
      {
        strcpy( client->in_buffer_basename, buffer );
        modfail += (!WritePrivateProfileStringB( OPTSECT_BUFFERS, "buffer-file-basename", buffer, fn ));
      }
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_BUFFERS, "output-file-basename", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "out", "", buffer, sizeof(buffer), fn ))
    {
      int basenameoffset = GetFilenameBaseOffset( buffer );
      const char *suffixsep = EXTN_SEP;
      p = strrchr( &buffer[basenameoffset], *suffixsep );
      if (p)
        *p = '\0';
      if (strcmp( buffer, "buff-out" ) != 0)
      {
        strcpy( client->out_buffer_basename, buffer );
        modfail += (!WritePrivateProfileStringB( OPTSECT_BUFFERS, "output-file-basename", buffer, fn ));
      }
    }
  }
  if (GetPrivateProfileIntB( OPTION_SECTION, "descontestclosed", 0, fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "in2", "", buffer, sizeof(buffer), fn ))
      unlink( GetFullPathForFilename( buffer ) );
    if (GetPrivateProfileStringB( OPTION_SECTION, "out2", "", buffer, sizeof(buffer), fn ))
      unlink( GetFullPathForFilename( buffer ) );
  }

  /* ----------------- OPTSECT_NET ----------------- */
  TRACE_OUT((0,"remapping 7 (%d)\n", modfail));

  if (GetPrivateProfileIntB( OPTSECT_NET, "nettimeout", 0, fn ) == 0)
  {
    if ((i = GetPrivateProfileIntB( OPTION_SECTION, "nettimeout", 0, fn ))!=0)
    {
      if (i < 0)
        i = -1;
      else if (i < 5)
        i = 5;
      else if (i > 180)
        i = 180;
      client->nettimeout = i;
      modfail += (!_WritePrivateProfile_sINT( OPTSECT_NET, "nettimeout", i, fn));
      TRACE_OUT((0,"remapped nettimeout (%d)\n", modfail));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_NET, "nofallback", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileIntB( OPTION_SECTION, "nofallback", 0, fn ))
    {
      client->nofallback = 1;
      modfail += (!WritePrivateProfileStringB( OPTSECT_NET, "nofallback", "true", fn));
      TRACE_OUT((0,"remapped nofallback (%d)\n", modfail));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_NET, "disabled", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileIntB( OPTION_SECTION, "runoffline", 0, fn ))
    {
      client->offlinemode = 1;
      if (GetPrivateProfileIntB( OPTION_SECTION, "lurkonly", 0, fn ) ||
          GetPrivateProfileIntB( OPTION_SECTION, "lurk", 0, fn ) )
        client->offlinemode = 0;
      if (client->offlinemode)
        modfail += (!WritePrivateProfileStringB( OPTSECT_NET, "disabled", "yes", fn ));
      TRACE_OUT((0,"remapped runoffline (%d)\n", modfail));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_NET, "keyserver", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "keyproxy", "", buffer, sizeof(buffer), fn ))
    {
      if (__getautofinddefault(buffer))
        buffer[0] = '\0'; /* force user to update hostname setting (once) */
    }
    if ((i = GetPrivateProfileIntB( OPTION_SECTION, "keyport", 0, fn ))!=0)
    {
      if (i < 0 || i > 0xffff || i == 2064)
        i = 0;    
    }
    if (buffer[0] || i)
    {
      strncpy( client->keyproxy, buffer, sizeof(client->keyproxy) );
      client->keyproxy[sizeof(client->keyproxy)-1] = '\0';
      client->autofindkeyserver = ((buffer[0])?(0):(1));
      WritePrivateProfileStringB( OPTSECT_NET,"autofindkeyserver",((client->autofindkeyserver)?(NULL):("no")), fn);
      client->keyport = i;
      modfail += _readwrite_hostname_and_port( 1, fn,
                                OPTSECT_NET, "keyserver",
                                client->keyproxy, sizeof(client->keyproxy),
                                &(client->keyport), 0 );
    }
    TRACE_OUT((0,"remapped keyserver/port (%d)\n", modfail));
  }
  if (!GetPrivateProfileStringB( OPTSECT_NET, "enable-start-stop", "", buffer, sizeof(buffer), fn ))
  {
    if ((i = GetPrivateProfileIntB( OPTION_SECTION, "dialwhenneeded", -123, fn )) != -123)
    {
      #ifdef LURK
      client->lurk_conf.dialwhenneeded = i;
      #endif
      if (i)
        modfail += (!WritePrivateProfileStringB( OPTSECT_NET, "enable-start-stop", "yes", fn ));
    }
    TRACE_OUT((0,"remapped nettimeout (%d)\n", modfail));
  }
  if (!GetPrivateProfileStringB( OPTSECT_NET, "dialup-watcher", "", buffer, sizeof(buffer), fn ))
  {
    buffer[0] = '\0';
    if ((i = GetPrivateProfileIntB( OPTION_SECTION, "lurk", 0, fn )) != 0)
    {
      strcpy(buffer,"active");
      #ifdef LURK
      client->lurk_conf.lurkmode = CONNECT_LURK;
      #endif
    }
    else if ((i = GetPrivateProfileIntB( OPTION_SECTION, "lurkonly", 0, fn )) != 0)
    {
      WritePrivateProfileStringB( OPTION_SECTION, "frequent", NULL, fn );
      strcpy(buffer,"passive");
      #ifdef LURK
      client->lurk_conf.lurkmode = CONNECT_LURKONLY;
      #endif
    }
    if (buffer[0])
    {
      modfail += (!WritePrivateProfileStringB( OPTSECT_NET, "dialup-watcher", buffer, fn ));
      TRACE_OUT((0,"remapped lurkmode (%d)\n", modfail));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_NET, "dialup-profile", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "connectionname", "", buffer, sizeof(buffer), fn ))
    {
      #ifdef LURK
      strncpy( client->lurk_conf.connprofile, buffer, sizeof(client->lurk_conf.connprofile) );
      client->lurk_conf.connprofile[sizeof(client->lurk_conf.connprofile)-1]='\0';
      #endif
      modfail += (!WritePrivateProfileStringB( OPTSECT_NET, "dialup-profile", buffer, fn ));
      TRACE_OUT((0,"remapped connectionname (%d)\n", modfail));
    }
  }

  /* ----------------- OPTSECT_CPU ----------------- */
  TRACE_OUT((0,"remapping 8 (%d)\n", modfail));

  if ((i=GetPrivateProfileIntB( OPTSECT_CPU, "max-threads",-12345,fn))!=-12345)
  {
    if ((i = GetPrivateProfileIntB( OPTION_SECTION, "numcpu", -12345, fn ))>=0)
    {
      client->numcpu = i;
      modfail += (!_WritePrivateProfile_sINT( OPTSECT_CPU, "max-threads", client->numcpu, fn));
    }
  }
  if ((i=GetPrivateProfileIntB( OPTSECT_CPU, "priority", -12345, fn ))!=-12345)
  {
    i = GetPrivateProfileIntB( OPTION_SECTION, "niceness", -12345, fn );
    if (i>=0 && i<=2)
    {
      client->priority = i * 4; /* ((i==2)?(8):((i==1)?(4):(0))) */
      modfail += (!_WritePrivateProfile_sINT( OPTSECT_CPU, "priority", client->priority, fn));
    }
  }

  /* ----------------- OPTSECT_MISC ----------------- */
  TRACE_OUT((0,"remapping 9 (%d)\n", modfail));

  if (!GetPrivateProfileStringB(OPTSECT_MISC,"run-time-limit","",buffer,2,fn))
  {
    if (GetPrivateProfileStringB(OPTION_SECTION,"hours","",buffer,sizeof(buffer),fn))
    {
      if ((i = __parse_timestring( buffer, 1 /* compat */ )) > 0)
      {
        client->minutes = i / 60; /* we only want minutes */
        if (client->minutes)
          modfail += __readwrite_minutes(OPTSECT_MISC, "run-time-limit",
                                             &(client->minutes), 0, fn, 0 );
      }
    }
  }
  if (!GetPrivateProfileStringB(OPTSECT_MISC,"run-work-limit","",buffer,2,fn))
  {
    if (GetPrivateProfileIntB( OPTION_SECTION, "runbuffers", 0, fn ))
      i = -1;
    else
      i = GetPrivateProfileIntB( OPTION_SECTION, "count", 0, fn );
    if (i == -1 || i > 0)
    {
      client->blockcount = i;
      modfail += (!_WritePrivateProfile_sINT( OPTSECT_MISC, "run-work-limit", i, fn));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_MISC, "project-priority", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileIntB( OPTION_SECTION, "processdes", sizeof(buffer), fn ) == 0 )
    {
      int doneclient = 0, doneini = 0;
      projectmap_build( buffer, "" );
      for (ui = 0; ui < CONTEST_COUNT && (!doneclient || !doneini); ui++)
      {
        if (client->loadorder_map[ui] == 1)
        {
          doneclient = 1;
          client->loadorder_map[ui] |= 0x80;
        }
        if (buffer[ui] == DES)
        {
          doneini = 1;
          buffer[ui] |= 0x80; /* disable */
        }
      }
      modfail += (!WritePrivateProfileStringB( OPTSECT_MISC, "project-priority", projectmap_expand(buffer), fn ));
    }
  }
  /* ----------------- OPTSECT_DISPLAY ----------------- */
  TRACE_OUT((0,"remapping 10 (%d)\n", modfail));

  if (!GetPrivateProfileStringB( OPTSECT_DISPLAY, "detached", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileIntB(OPTION_SECTION, "runhidden", 0, fn ) ||
        GetPrivateProfileIntB( OPTION_SECTION, "quiet", 0, fn ) ||
        GetPrivateProfileIntB(OPTION_SECTION, "os2hidden", 0, fn ) ||
        GetPrivateProfileIntB(OPTION_SECTION, "win95hidden", 0, fn ))
    {
      client->quietmode = 1;
      modfail += (!WritePrivateProfileStringB( OPTSECT_DISPLAY, "detached", "yes", fn));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_DISPLAY, "progress-indicator", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileIntB(OPTION_SECTION, "percentoff", 0, fn ))
    {
      client->crunchmeter = 0; /* off */
      modfail += (!WritePrivateProfileStringB( OPTSECT_DISPLAY, "progress-indicator", "off", fn));
    }
  }

  /* ----------------- OPTSECT_TRIGGERS ----------------- */
  TRACE_OUT((0,"remapping 11 (%d)\n", modfail));

  /* exit-flag-filename is a bit unusual in that the default (if the key 
     doesn't exist) is not "", but thats all handled in confread().
  */
  if (GetPrivateProfileStringB( OPTSECT_TRIGGERS, "exit-flag-filename", "*", buffer, sizeof(buffer), fn )==1)
  {      /* need two tests to ensure the key doesn't exist */
    if (buffer[0] == '*')
    {
      if ((GetPrivateProfileIntB(OPTSECT_TRIGGERS, "exit-flag-file-checks", 1, fn)==0)
       || (GetPrivateProfileIntB(OPTION_SECTION, "noexitfilecheck", 0, fn )!=0))
      {
        modfail+=WritePrivateProfileStringB( OPTSECT_TRIGGERS, 
                              "exit-flag-filename", "", fn);
      }
    }
  }  
  WritePrivateProfileStringB( OPTSECT_TRIGGERS, "exit-flag-file-checks", NULL, fn);
  if (!GetPrivateProfileStringB( OPTSECT_TRIGGERS, "pause-flag-filename", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB(OPTION_SECTION, "pausefile", "", buffer, sizeof(buffer), fn ))
    {
      strncpy( client->pausefile, buffer, sizeof(client->pausefile) );
      client->pausefile[sizeof(client->pausefile)-1]='\0';
      modfail += (!WritePrivateProfileStringB( OPTSECT_TRIGGERS, "pause-flag-filename", client->pausefile, fn));
    }
  }

  /* ---------------------------- */
  /* unconditional deletion of obsolete keys */
  /* (all keys in OPTION_SECTION except "id") */
  {
    #if 0 /* faster but causes comment loss (and puts the section at the end) */
    if (GetPrivateProfileStringB( OPTION_SECTION, "id", "", buffer, sizeof(buffer), fn ))
    {
      for (i=0;buffer[i];i++)
        buffer[i]=(char)tolower(buffer[i]);
      if (strcmp( buffer, "rc5@distributed.net" ) == 0)
        buffer[0] = '\0';
    }
    WritePrivateProfileStringB( OPTION_SECTION, NULL, "", fn);
    modfail += (!WritePrivateProfileStringB( OPTION_SECTION, "id", buffer, fn));
    #else
    static const char *obskeys[]={ /* all in "parameters" section */
             "runhidden", "os2hidden", "win95hidden", "checkpoint2",
             "firemode", "checkpointfile2", "randomprefix",
             "preferredcontest", "cktime", "exitfilechecktime",
             "niceness", "processdes", "timeslice", "runbuffers",
             "contestdone" /* now in "rc564" */, "contestdone2",
             "contestdone3", "contestdoneflags", "descontestclosed",
             "scheduledupdatetime", "checkpointfile", "hours",
             "usemmx", "runoffline", "in","out","in2","out2",
             "in3","out3","nodisk", "dialwhenneeded","connectionname",
             "cputype","threshold","threshold2","preferredblocksize",
             "logname", "keyproxy", "keyport", "numcpu", "count",
             "smtpsrvr", "smtpport", "messagelen", "smtpfrom", "smtpdest",
             "lurk", "lurkonly", "nettimeout", "nofallback", "frequent",
             "pausefile","noexitfilecheck","percentoff", "quiet" };
    for (ui = 0; ui < (sizeof(obskeys) / sizeof(obskeys[0])); ui++)
      modfail += (!WritePrivateProfileStringB( OPTION_SECTION, obskeys[ui], NULL, fn ));
    #endif
  }

  TRACE_OUT((-1,"__remapObsoleteParameters() modif failed?=%d\n", modfail));

  if (modfail)
    return -1;
  return 0;
}


/* ------------------------------------------------------------------------ */

// Main configuration reading function.
//
// 1. never printf()/logscreen()/conout() from here
// 2. never force an option based on the value of some other valid option

int ConfigRead(Client *client)
{
  int foundinifile = 1;
  char buffer[64];
  const char *cont_name;
  unsigned int i, cont_i;
  const char *fn;

  /* must actually go through settings even if file doesn't exist */
  /* in order to load client with default values. */
  fn = GetFullPathForFilename( client->inifilename );
  if ( access( fn, 0 ) != 0 )
  {
    fn = GetFullPathForFilename( "rc5des" EXTN_SEP "ini" );
    if ( access( fn, 0 ) != 0 )
      foundinifile = 0;
  }

  TRACE_OUT((+1,"ReadConfig() [%s] (was: '%s')\n", fn, client->inifilename));

  if (foundinifile)
  {
    __remapObsoleteParameters( client, fn ); /* load obsolete options */
  }  

  /* intialize the "authoritative" variables set during proxy chitchat */
  /* eg, scheduledupdatetime et al */
  ConfigReadUniversalNews( client, fn );

  if (GetPrivateProfileStringB( OPTION_SECTION, "id", "", client->id, sizeof(client->id), fn ))
  {
    strncpy(buffer,client->id,sizeof(buffer));
    buffer[sizeof(buffer)-1]='\0';
    for (cont_i=0;buffer[cont_i];cont_i++)
      buffer[cont_i]=(char)tolower(buffer[cont_i]);
    if (strcmp( buffer, "rc5@distributed.net" ) == 0)
      client->id[0]='\0';
  }

  /* --------------------- */

  client->quietmode = GetPrivateProfileIntB( OPTSECT_DISPLAY, "detached", client->quietmode, fn );
  client->crunchmeter = -1; /* auto */
  if (GetPrivateProfileStringB( OPTSECT_DISPLAY, "progress-indicator", "", buffer, sizeof(buffer), fn ))
  {
    for (i=0; i < 3; i++)
      buffer[i] = (char)tolower(buffer[i]);
    if (memcmp(buffer,"aut",3)==0) /* "aut[o-sense]" */
      client->crunchmeter = -1;
    else if (isdigit(buffer[0]) && atoi(buffer) == 0) /* 0 == off, the rest must be */
      client->crunchmeter = 0;                   /* names for forward compatibility */
    else if (memcmp(buffer,"off",3)==0) /* "off" */    
      client->crunchmeter = 0;
    else if (memcmp(buffer,"abs",3)==0) /* "abs[olute]" */
      client->crunchmeter = 1;
    else if (memcmp(buffer,"rel",3)==0) /* "rel[ative]" */
      client->crunchmeter = 2;
    else if (memcmp(buffer,"rat",3)==0) /* "rat[e]" */
      client->crunchmeter = 3;
  }

  /* --------------------- */

  GetPrivateProfileStringB( OPTSECT_TRIGGERS, "exit-flag-filename", DEFAULT_EXITFLAGFILENAME, client->exitflagfile, sizeof(client->exitflagfile), fn );
  GetPrivateProfileStringB( OPTSECT_TRIGGERS, "pause-flag-filename", client->pausefile, client->pausefile, sizeof(client->pausefile), fn );
  client->restartoninichange = GetPrivateProfileIntB( OPTSECT_TRIGGERS, "restart-on-config-file-change", client->restartoninichange, fn );
  GetPrivateProfileStringB( OPTSECT_TRIGGERS, "pause-watch-plist", client->pauseplist, client->pauseplist, sizeof(client->pauseplist), fn );
  client->nopauseifnomainspower = !GetPrivateProfileIntB( OPTSECT_TRIGGERS, "pause-on-no-mains-power", !client->nopauseifnomainspower, fn );
  client->watchcputempthresh = GetPrivateProfileIntB( OPTSECT_TRIGGERS, "pause-on-high-cpu-temp", client->watchcputempthresh, fn );
  GetPrivateProfileStringB( OPTSECT_TRIGGERS, "cpu-temperature-thresholds", client->cputempthresh, client->cputempthresh, sizeof(client->cputempthresh), fn );

  /* --------------------- */

  client->offlinemode = GetPrivateProfileIntB( OPTSECT_NET, "disabled", client->offlinemode, fn );
  _readwrite_fwallstuff( 0, fn, client );
  _readwrite_hostname_and_port( 0, fn, OPTSECT_NET, "keyserver",
                                client->keyproxy, sizeof(client->keyproxy),
                                &(client->keyport), 1 );
  //NetOpen() gets (autofindkeyserver)?(""):(client->keyproxy))
  client->autofindkeyserver = GetPrivateProfileIntB( OPTSECT_NET, "autofindkeyserver", __getautofinddefault(client->keyproxy), fn );
  client->nettimeout = GetPrivateProfileIntB( OPTSECT_NET, "nettimeout", client->nettimeout, fn );
  client->nofallback = GetPrivateProfileIntB( OPTSECT_NET, "nofallback", client->nofallback, fn );

  #if defined(LURK)
  {
    const char *p = "";
    if (client->lurk_conf.lurkmode == CONNECT_LURKONLY)
      p = "passive";
    else if (client->lurk_conf.lurkmode == CONNECT_LURK)
      p = "active";
    client->lurk_conf.lurkmode = 0;
    if (GetPrivateProfileStringB( OPTSECT_NET, "dialup-watcher", p, buffer, sizeof(buffer), fn ))
    {
      if (buffer[0] == 'a' || buffer[0] == 'A')  /*active*/
        client->lurk_conf.lurkmode = CONNECT_LURK;
      else if (buffer[0] == 'p' || buffer[0] == 'P')  /*passive*/
        client->lurk_conf.lurkmode = CONNECT_LURKONLY;
    }
    GetPrivateProfileStringB( OPTSECT_NET, "interfaces-to-watch", client->lurk_conf.connifacemask, client->lurk_conf.connifacemask, sizeof(client->lurk_conf.connifacemask), fn );
    client->lurk_conf.dialwhenneeded = GetPrivateProfileIntB( OPTSECT_NET, "enable-start-stop", client->lurk_conf.dialwhenneeded, fn );
    GetPrivateProfileStringB( OPTSECT_NET, "dialup-start-cmd", client->lurk_conf.connstartcmd, client->lurk_conf.connstartcmd, sizeof(client->lurk_conf.connstartcmd), fn );
    GetPrivateProfileStringB( OPTSECT_NET, "dialup-stop-cmd", client->lurk_conf.connstopcmd, client->lurk_conf.connstopcmd, sizeof(client->lurk_conf.connstopcmd), fn );
    GetPrivateProfileStringB( OPTSECT_NET, "dialup-profile", client->lurk_conf.connprofile, client->lurk_conf.connprofile, sizeof(client->lurk_conf.connprofile), fn );
  }
  #endif /* LURK */

  /* --------------------- */

  client->priority = GetPrivateProfileIntB( OPTSECT_CPU, "priority", client->priority, fn );
  client->numcpu = GetPrivateProfileIntB( OPTSECT_CPU, "max-threads", client->numcpu, fn );

  /* --------------------- */

  TRACE_OUT((0,"ReadConfig() [2 begin]\n"));

  client->minutes = __readwrite_minutes( OPTSECT_MISC,"run-time-limit", NULL, client->minutes, fn, 1 );
  client->blockcount = GetPrivateProfileIntB( OPTSECT_MISC, "run-work-limit", client->blockcount, fn );

  TRACE_OUT((0,"ReadConfig() [2 end]\n"));

  /* --------------------- */

  GetPrivateProfileStringB( OPTSECT_LOG, "mail-log-via", client->smtpsrvr, client->smtpsrvr, sizeof(client->smtpsrvr), fn );
  client->messagelen = GetPrivateProfileIntB( OPTSECT_LOG, "mail-log-max", client->messagelen, fn );
  GetPrivateProfileStringB( OPTSECT_LOG, "mail-log-from", client->smtpfrom, client->smtpfrom, sizeof(client->smtpfrom), fn );
  GetPrivateProfileStringB( OPTSECT_LOG, "mail-log-dest", client->smtpdest, client->smtpdest, sizeof(client->smtpdest), fn );
  GetPrivateProfileStringB( OPTSECT_LOG, "log-file", client->logname, client->logname, sizeof(client->logname), fn );
  GetPrivateProfileStringB( OPTSECT_LOG, "log-file-type", client->logfiletype, client->logfiletype, sizeof(client->logfiletype), fn );
  GetPrivateProfileStringB( OPTSECT_LOG, "log-file-limit", client->logfilelimit, client->logfilelimit, sizeof(client->logfilelimit), fn );

  /* -------------------------------- */

  client->noupdatefromfile = (!GetPrivateProfileIntB( OPTSECT_BUFFERS, "allow-update-from-altbuffer", !(client->noupdatefromfile), fn ));
  client->nodiskbuffers = GetPrivateProfileIntB( OPTSECT_BUFFERS, "buffer-only-in-memory", client->nodiskbuffers , fn );
  GetPrivateProfileStringB( OPTSECT_BUFFERS, "buffer-file-basename", client->in_buffer_basename, client->in_buffer_basename, sizeof(client->in_buffer_basename), fn );
  GetPrivateProfileStringB( OPTSECT_BUFFERS, "output-file-basename", client->out_buffer_basename, client->out_buffer_basename, sizeof(client->out_buffer_basename), fn );
  GetPrivateProfileStringB( OPTSECT_BUFFERS, "alternate-buffer-directory", client->remote_update_dir, client->remote_update_dir, sizeof(client->remote_update_dir), fn );
  GetPrivateProfileStringB( OPTSECT_BUFFERS, "checkpoint-filename", client->checkpoint_file, client->checkpoint_file, sizeof(client->checkpoint_file), fn );
  client->connectoften = GetPrivateProfileIntB( OPTSECT_BUFFERS, "frequent-threshold-checks", client->connectoften , fn );
  client->max_buffupd_interval = __readwrite_minutes( OPTSECT_BUFFERS,"threshold-check-interval", NULL, client->max_buffupd_interval, fn, 0 );
  GetPrivateProfileStringB( OPTSECT_MISC, "project-priority", "", buffer, sizeof(buffer), fn );
  projectmap_build(client->loadorder_map, buffer);

  TRACE_OUT((0,"ReadConfig() [3 begin]\n"));

  for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
  {
    if ((cont_name = __getprojsectname(cont_i)) != ((const char *)0))
    {
      client->coretypes[cont_i] =
         GetPrivateProfileIntB(cont_name, "core",
                       client->coretypes[cont_i],fn);
      if (cont_i != OGR)
      {
        /* note that the default preferred_blocksize is now <=0 (auto) */
        client->preferred_blocksize[cont_i] =
           GetPrivateProfileIntB(cont_name, "preferred-blocksize",
                         client->preferred_blocksize[cont_i], fn );
      }

      client->inthreshold[cont_i] =
           GetPrivateProfileIntB(cont_name, "fetch-workunit-threshold",
                         client->inthreshold[cont_i], fn );
      #if defined(NO_OUTBUFFER_THRESHOLDS)
      if (client->connectoften == 0)
      {
        int ot=GetPrivateProfileIntB(cont_name,"flush-workunit-threshold",0,fn);
        if (ot == 1)
          client->connectoften = 1; /* frequent in-buf checks */
        else if (ot > 0 && ot < client->inthreshold[cont_i])
          client->connectoften = 4; /* sticky contests */
        if (client->connectoften != 0) /* changed */ 
          _WritePrivateProfile_sINT( OPTSECT_BUFFERS, "frequent-threshold-checks", client->connectoften, fn );
      }
      WritePrivateProfileStringB( cont_name,"flush-workunit-threshold",NULL,fn);
      #else
      client->outthreshold[cont_i] =
           GetPrivateProfileIntB(cont_name, "flush-workunit-threshold",
                         client->outthreshold[cont_i], fn );
      if (client->outthreshold[cont_i] > client->inthreshold[cont_i])
        client->outthreshold[cont_i] = client->inthreshold[cont_i];
      #endif        
      client->timethreshold[cont_i] =
           GetPrivateProfileIntB(cont_name, "fetch-time-threshold",
                         client->timethreshold[cont_i], fn);
    }
  }
  TRACE_OUT((0,"ReadConfig() [3 end]\n"));

  TRACE_OUT((-1,"ReadConfig()\n"));

  return ((foundinifile) ? (0) : (+1));
}

// --------------------------------------------------------------------------

//conditional (if exist || if !default) ini write functions

static void __XSetProfileStr( const char *sect, const char *key,
            const char *newval, const char *fn, const char *defval )
{
  int dowrite = 1;
  if (sect == NULL)
    sect = OPTION_SECTION;
  if (newval != NULL)
  {
    if (defval == NULL)
      defval = "";
    dowrite = (strcmp( newval, defval )!=0);
    if (!dowrite)
    {
      char buffer[4];
      dowrite = (GetPrivateProfileStringB( sect, key, "=", 
                      buffer, sizeof(buffer), fn ) !=1 );
      if (!dowrite) dowrite = (buffer[0] != '=' );
    }  
  }
  if (dowrite)
    WritePrivateProfileStringB( sect, key, newval, fn );
  return;
}

static void __XSetProfileInt( const char *sect, const char *key,
    long newval, const char *fn, long defval, int asonoff /*+style=y|t|o|1*/ )
{
  int dowrite;
  char buffer[(sizeof(long)+1)*3];
  if (sect == NULL)
    sect = OPTION_SECTION;
  if (asonoff)
  {
    if (defval) defval = 1;
    if (newval) newval = 1;
  }
  dowrite = (defval != newval);
  if (!dowrite)
    dowrite = (GetPrivateProfileStringB( sect, key, "", buffer, sizeof(buffer), fn ) != 0);
  if (dowrite)
  {
    if (asonoff)
    {
      const char *p = ((newval)?("1"):("0"));
      if (asonoff == 'y' || asonoff == 'n')
        p = ((newval)?("yes"):("no"));
      else if (asonoff == 't' || asonoff == 'f')
        p = ((newval)?("true"):("false"));
      else if (asonoff == 'o')
        p = ((newval)?("on"):("off"));
      WritePrivateProfileStringB( sect, key, p, fn );
    }
    else
    {
      sprintf(buffer,"%ld",(long)newval);
      WritePrivateProfileStringB( sect, key, buffer, fn );
    }
  }
  return;
}

// --------------------------------------------------------------------------

int ConfigWrite(Client *client)
{
  char buffer[64]; int rc = -1;
  unsigned int cont_i;
  const char *p;
  const char *fn = GetFullPathForFilename( client->inifilename );
  TRACE_OUT((+1,"WriteConfig() [%s] (originally '%s')\n", fn, client->inifilename ));

  /* prevent our own writes from kicking off a restart */
  OverrideNextConffileChangeTrigger();

  if (WritePrivateProfileStringB( OPTION_SECTION, "id",
    ((strcmp( client->id,"rc5@distributed.net")==0)?(""):(client->id)), fn ))
  {    
    rc = 0; /* assume success */
  
    /* --- CONF_MENU_BUFF -- */
    __XSetProfileInt( OPTSECT_BUFFERS, "buffer-only-in-memory", (client->nodiskbuffers)?(1):(0), fn, 0, 'y' );
    __XSetProfileStr( OPTSECT_BUFFERS, "buffer-file-basename", client->in_buffer_basename, fn, NULL );
    __XSetProfileStr( OPTSECT_BUFFERS, "output-file-basename", client->out_buffer_basename, fn, NULL );
    __XSetProfileStr( OPTSECT_BUFFERS, "checkpoint-filename", client->checkpoint_file, fn, NULL );
    __XSetProfileInt( OPTSECT_BUFFERS, "allow-update-from-altbuffer", !(client->noupdatefromfile), fn, 1, 'y' );
    __XSetProfileStr( OPTSECT_BUFFERS, "alternate-buffer-directory", client->remote_update_dir, fn, NULL );
    __XSetProfileInt( OPTSECT_BUFFERS, "frequent-threshold-checks", client->connectoften, fn, 0, 0 );
    __readwrite_minutes( OPTSECT_BUFFERS,"threshold-check-interval", &(client->max_buffupd_interval), 0, fn, 0 );

    /* --- CONF_MENU_MISC __ */

    strcpy(buffer,projectmap_expand(NULL));
    __XSetProfileStr( OPTSECT_MISC, "project-priority", projectmap_expand(client->loadorder_map), fn, buffer );
    __readwrite_minutes( OPTSECT_MISC,"run-time-limit", &(client->minutes), 0, fn, 0 );
    __XSetProfileInt( OPTSECT_MISC, "run-work-limit", client->blockcount, fn, 0, 0 );

    __XSetProfileInt( OPTSECT_TRIGGERS, "restart-on-config-file-change", client->restartoninichange, fn, 0, 'n' );
    __XSetProfileStr( OPTSECT_TRIGGERS, "exit-flag-filename", client->exitflagfile, fn, DEFAULT_EXITFLAGFILENAME );
    __XSetProfileStr( OPTSECT_TRIGGERS, "pause-flag-filename", client->pausefile, fn, NULL );
    __XSetProfileStr( OPTSECT_TRIGGERS, "pause-watch-plist", client->pauseplist, fn, NULL );
    __XSetProfileInt( OPTSECT_TRIGGERS, "pause-on-no-mains-power", !client->nopauseifnomainspower, fn, 1, 'y' );
    __XSetProfileInt( OPTSECT_TRIGGERS, "pause-on-high-cpu-temp", client->watchcputempthresh, fn, 0, 'n' );
    __XSetProfileStr( OPTSECT_TRIGGERS, "cpu-temperature-thresholds", client->cputempthresh, fn, NULL );
    __XSetProfileInt( OPTSECT_DISPLAY, "detached", client->quietmode, fn, 0, 'y' );
    p = NULL;
    if      (client->crunchmeter  < 0)  p = "auto-sense"; /* -1 */
    else if (client->crunchmeter == 0)  p = "off";        /* 0 */
    else if (client->crunchmeter == 1)  p = "absolute";   /* 1 */
    else if (client->crunchmeter == 2)  p = "relative";   /* 2 */
    else if (client->crunchmeter == 3)  p = "rate";       /* 3 */
    __XSetProfileStr( OPTSECT_DISPLAY, "progress-indicator", p, fn, NULL );

    /* --- CONF_MENU_PERF -- */

    __XSetProfileInt( OPTSECT_CPU, "max-threads", client->numcpu, fn, -1, 0 );
    __XSetProfileInt( OPTSECT_CPU, "priority", client->priority, fn, 0, 0);

    /* more buffer stuff */

    TRACE_OUT((+0,"cont_i loop\n"));
    for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
    {
      TRACE_OUT((+0,"cont_i=%u\n", cont_i));
      if ((p =  __getprojsectname(cont_i)) != ((const char *)0))
      {
        __XSetProfileInt( p, "fetch-workunit-threshold", client->inthreshold[cont_i], fn,  0, 0 );
        #if !defined(NO_OUTBUFFER_THRESHOLDS)          
        __XSetProfileInt( p, "flush-workunit-threshold", client->outthreshold[cont_i], fn, 0, 0 );
        #endif
        __XSetProfileInt( p, "fetch-time-threshold", client->timethreshold[cont_i], fn, 0, 0 );
        __XSetProfileInt( p, "core", client->coretypes[cont_i], fn, -1, 0 );
        if (cont_i != OGR)
        {
          if (client->preferred_blocksize[cont_i] > 0 ||
              GetPrivateProfileStringB(p,"preferred-blocksize","",buffer,2,fn))
          {
            __XSetProfileInt( p, "preferred-blocksize",
              client->preferred_blocksize[cont_i], fn, 0, 0 );
          }
        }
      }
    }

    /* --- CONF_MENU_NET -- */

    __XSetProfileInt( OPTSECT_NET, "disabled", client->offlinemode, fn, 0, 'n');
    __XSetProfileInt( OPTSECT_NET, "nettimeout", client->nettimeout, fn, 60, 0);
    __XSetProfileInt( OPTSECT_NET, "nofallback", client->nofallback, fn, 0, 't');
    _readwrite_fwallstuff( 1, fn, client );
    __XSetProfileInt( OPTSECT_NET, "autofindkeyserver", client->autofindkeyserver, fn, 1, 'y');
    _readwrite_hostname_and_port( 1, fn, OPTSECT_NET, "keyserver",
                                client->keyproxy, sizeof(client->keyproxy),
                                &(client->keyport), 1 );

    #if defined(LURK)
    {
      p = NULL;
      if (client->lurk_conf.lurkmode == CONNECT_LURKONLY)
        p = "passive";
      else if (client->lurk_conf.lurkmode == CONNECT_LURK)
        p = "active";
      else if (GetPrivateProfileStringB(OPTSECT_NET,"dialup-watcher","",buffer,2,fn))
        p = "disabled";
      if (p)
        WritePrivateProfileStringB( OPTSECT_NET, "dialup-watcher", p, fn );
      __XSetProfileStr( OPTSECT_NET, "interfaces-to-watch", client->lurk_conf.connifacemask, fn, NULL );
      __XSetProfileInt( OPTSECT_NET, "enable-start-stop", (client->lurk_conf.dialwhenneeded!=0), fn, 0, 'n' );
      __XSetProfileStr( OPTSECT_NET, "dialup-profile", client->lurk_conf.connprofile, fn, NULL );
      __XSetProfileStr( OPTSECT_NET, "dialup-start-cmd", client->lurk_conf.connstartcmd, fn, NULL );
      __XSetProfileStr( OPTSECT_NET, "dialup-stop-cmd", client->lurk_conf.connstopcmd, fn, NULL );
    }
    #endif // defined LURK

    /* --- CONF_MENU_LOG -- */

    __XSetProfileStr( OPTSECT_LOG, "mail-log-via", client->smtpsrvr, fn, NULL);
    __XSetProfileInt( OPTSECT_LOG, "mail-log-max", client->messagelen, fn, 0, 0);
    __XSetProfileStr( OPTSECT_LOG, "mail-log-from", client->smtpfrom, fn, NULL);
    __XSetProfileStr( OPTSECT_LOG, "mail-log-dest", client->smtpdest, fn, NULL);

    __XSetProfileStr( OPTSECT_LOG, "log-file-limit", client->logfilelimit, fn, NULL );
    __XSetProfileStr( OPTSECT_LOG, "log-file", client->logname, fn, NULL );
    if ((client->logfiletype[0] && strcmp(client->logfiletype,"none")!=0) ||
      GetPrivateProfileStringB(OPTSECT_LOG,"log-file-type","",buffer,2,fn))
      WritePrivateProfileStringB( OPTSECT_LOG,"log-file-type", client->logfiletype, fn );

  } /* if (Write(id)) */

  TRACE_OUT((-1,"WriteConfig() =>%d\n", rc));

  return rc;
}

