/*
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/
const char *confrwv_cpp(void) {
return "@(#)$Id: confrwv.cpp,v 1.60.2.29 2000/04/22 11:01:59 cyp Exp $"; }

//#define TRACE

#include "cputypes.h"
#include "client.h"    // Client class
#include "baseincs.h"  // atoi() etc
#include "iniread.h"   // [Get|Write]Profile[Int|String]()
#include "pathwork.h"  // GetFullPathForFilename()
#include "util.h"      // projectmap_*() and trace
#include "base64.h"    // base64_[en|de]code()
#include "clicdata.h"  // CliGetContestNameFromID()
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

/* ------------------------------------------------------------------------ */

static const char *__getprojsectname( unsigned int ci )
{
  if (ci < CONTEST_COUNT)
  {
    #if 1
      #if (CONTEST_COUNT != 4)
        #error fixme: (CONTEST_COUNT != 4) static table
      #endif
      static const char *sectnames[CONTEST_COUNT]={"rc5","des","ogr","csc"};
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

// Reads or writes a hostname and port in "hostname:port" format.

static int _readwrite_hostname_and_port( int aswrite, const char *fn,
                                         const char *sect, const char *opt,
                                         char *hostname, unsigned int hnamelen,
                                         int *port )
{
  char buffer[128];
  int pos = 0;
  if (!aswrite)
  {
    // when reading, hostname and port are assumed to be
    // pre-loaded with default values
    if (GetPrivateProfileStringB( sect, opt, "", buffer, sizeof(buffer), fn ))
    {
      char *q, *that = buffer;
      while (*that && isspace(*that))
        that++;
      q = that;
      while (*q && *q != ':' && !isspace(*q))
        q++;
      pos = (*q == ':');
      *q++ = '\0';
      if (pos && port)
      {
        while (*q && isspace(*q))
          q++;
        pos = 0;
        while (isdigit(q[pos]))
          pos++;
        if (pos)
        {
          q[pos] = '\0';
          *port = atoi(q);
        }
      }
      if (hostname && hnamelen)
      {
        if (strcmp( that, "*" ) == 0)
          *that = '\0';
        strncpy( hostname, buffer, hnamelen );
        hostname[hnamelen-1] = '\0';
      }
    }
    pos = 0;
  }
  else /* aswrite */
  {
    char scratch[2];
    buffer[0] = '\0';
    if (!hostname)
      hostname = buffer;
    pos = 0;
    if (port)
      pos = *port;
TRACE_OUT((0,"host:%s,port=%d\n",hostname,pos));
    if (pos>0 && pos<0xffff)
    {
      if (!*hostname)
        hostname = "*";
      sprintf(buffer,"%s:%d", hostname, pos );
      hostname = buffer;
    }
    pos = 0;
    if (*hostname ||
       GetPrivateProfileStringB( sect, opt, "", scratch, sizeof(scratch), fn ))
      pos = (!WritePrivateProfileStringB( sect, opt, hostname, fn));
  }
  return 0;
}

/* ------------------------------------------------------------------------ */

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
      client->httpport = 0;
      client->httpproxy[0] = '\0';
      _readwrite_hostname_and_port( 0, fn, OPTSECT_NET, "firewall-host",
                                    client->httpproxy, sizeof(client->httpproxy),
                                    &(client->httpport) );
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
          aswrite = 1;  /* key exists, need rewrite */
      }
      TRACE_OUT((0,"old httpproxy=\"%s\"\n",client->httpproxy));
      if (GetPrivateProfileStringB( OPTION_SECTION, "httpport", "", scratch, sizeof(scratch), fn ))
      {
        client->httpport = GetPrivateProfileIntB( OPTION_SECTION, "httpport", client->httpport, fn );
        if (client->httpport)
        aswrite = 1;  /* key exists, need rewrite */
      }
      TRACE_OUT((0,"old httpport=%d\n",client->httpport));
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

    _readwrite_hostname_and_port( 1, fn, OPTSECT_NET, "firewall-host",
                                client->httpproxy, sizeof(client->httpproxy),
                                &(client->httpport) );
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

static int __parse_timestring(const char *source, int oldstyle_hours_compat )
{           /* convert hh[:mm[:ss]] into secs. return 0 for "", <0 on err */
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
    /* and later "fixed" to two (still double) */
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

static int __remapObsoleteParameters( Client *client, const char *fn ) /* <0 if failed */
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
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_LOG, "mail-log-from", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "smtpfrom", "", buffer, sizeof(buffer), fn ))
    {
      strncpy( client->smtpfrom, buffer, sizeof(client->smtpfrom) );
      client->smtpfrom[sizeof(client->smtpfrom)-1] = '\0';
      modfail += (!WritePrivateProfileStringB( OPTSECT_LOG, "mail-log-from", buffer, fn ));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_LOG, "mail-log-dest", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "smtpdest", "", buffer, sizeof(buffer), fn ))
    {
      strncpy( client->smtpdest, buffer, sizeof(client->smtpdest) );
      client->smtpdest[sizeof(client->smtpdest)-1] = '\0';
      modfail += (!WritePrivateProfileStringB( OPTSECT_LOG, "mail-log-dest", buffer, fn ));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_LOG, "mail-log-via", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "smtpsrvr", "", buffer, sizeof(buffer), fn ))
    {
      strncpy( client->smtpsrvr, buffer, sizeof(client->smtpsrvr) );
      i = GetPrivateProfileIntB( OPTION_SECTION, "smtpport", client->smtpport, fn );
      if (i == 25) i = 0;
      client->smtpport = i;
      modfail += _readwrite_hostname_and_port( 1, fn,
                                OPTSECT_LOG, "mail-log-via",
                                client->smtpsrvr, sizeof(client->smtpsrvr),
                                &(client->smtpport) );
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_LOG, "log-file", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "logname", "", buffer, sizeof(buffer), fn ))
    {
      strncpy( client->logname, buffer, sizeof(client->logname) );
      client->logname[sizeof(client->logname)-1] = '\0';
      modfail += (!WritePrivateProfileStringB( OPTSECT_LOG, "log-file", buffer, fn ));
    }
  }

  /* ----------------- project options ----------------- */

  #if (CLIENT_CPU != CPU_ALPHA) && (CLIENT_CPU != CPU_68K) && (CLIENT_CPU != CPU_ARM)
  /* don't have RC5 cputype->coretype mapping for Alpha or m68k or arm */
  if (!GetPrivateProfileStringB( __getprojsectname(RC5), "core", "", buffer, sizeof(buffer), fn ))
  {
    if ((i = GetPrivateProfileIntB(OPTION_SECTION, "cputype", -1, fn ))!=-1)
    {
      client->coretypes[RC5] = i;
      modfail += (!WritePrivateProfileIntB( __getprojsectname(RC5), "core", i, fn));
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
          i != PREFERREDBLOCKSIZE_DEFAULT)
      {
        client->preferred_blocksize[RC5] = i;
        client->preferred_blocksize[DES] = i;
        modfail += (!WritePrivateProfileIntB( __getprojsectname(RC5), "preferred-blocksize", i, fn));
        modfail += (!WritePrivateProfileIntB( __getprojsectname(DES), "preferred-blocksize", i, fn));
      }
    }
  }

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
    if (!thresholdsdone)
    {
      /* convert post-2.8000 and pre-2.8000 format simultaneously */
      for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++ )
      {
        const char *cont_sect = __getprojsectname(cont_i);
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
                            "preferred-blocksize", PREFERREDBLOCKSIZE_DEFAULT, fn );
               if ( multiplier < 1)
                 multiplier = PREFERREDBLOCKSIZE_DEFAULT;
               else if ( multiplier < PREFERREDBLOCKSIZE_MIN)
                 multiplier = PREFERREDBLOCKSIZE_MIN;
               else if (multiplier > PREFERREDBLOCKSIZE_MAX)
                 multiplier = PREFERREDBLOCKSIZE_DEFAULT;
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
            modfail += (!WritePrivateProfileIntB( cont_sect, "fetch-workunit-threshold", client->inthreshold[cont_i], fn));
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
              modfail += (!WritePrivateProfileIntB( cont_sect, "flush-workunit-threshold", client->outthreshold[cont_i], fn));
            }
          }
          #endif
        }
      } /* for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++ ) */
    } /* if !donethresholds */
  }

  /* ----------------- OPTSECT_BUFFERS ----------------- */

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
      modfail += (!WritePrivateProfileIntB( OPTSECT_NET, "nettimeout", i, fn));
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_NET, "nofallback", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileIntB( OPTION_SECTION, "nofallback", 0, fn ))
    {
      client->nofallback = 1;
      modfail += (!WritePrivateProfileStringB( OPTSECT_NET, "nofallback", "true", fn));
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
    }
  }
  if (!GetPrivateProfileStringB( OPTSECT_NET, "keyserver", "", buffer, sizeof(buffer), fn ))
  {
    client->keyproxy[0] = '\0'; //default
    client->autofindkeyserver = 1;  //default
    client->keyport = 0; //default
    if (GetPrivateProfileStringB( OPTION_SECTION, "keyproxy", "", buffer, sizeof(buffer), fn ))
    {
      char sig[] = "distributed.net";
      ui = 0;
      while (buffer[ui]) {
        buffer[ui]=(char)tolower(buffer[ui]);
        ui++;
      } 
      if (ui >= 4 && (strcmp(buffer,"auto")==0 || strcmp(buffer,"(auto)")==0))
        buffer[0] = '\0'; //one config version accidentally wrote "auto" out
      else if (ui >= (sizeof( sig ) - 1) && 
               strcmp( &buffer[(ui-(sizeof( sig )-1))], sig )==0)
      {
        if (ui == (sizeof(sig)-1)) 
          buffer[0] = '\0'; /* plain "distributed.net" */
        else if (buffer[(ui-(sizeof( sig )))] == '.') /*[*].distributed.net*/
          buffer[0] = '\0'; /* make the hostname require reentry (once) */
      }    
      if (buffer[0])
      {      
        client->autofindkeyserver = 0;
        strncpy( client->keyproxy, buffer, sizeof(client->keyproxy) );
        client->keyproxy[sizeof(client->keyproxy)-1] = '\0';
      }
    }  
    if ((i = GetPrivateProfileIntB( OPTION_SECTION, "keyport", 0, fn ))!=0)
    {
      if (i < 0 || i > 0xffff || i == 2064)
        i = 0;
      else
        client->keyport = i;
    }
    modfail += WritePrivateProfileStringB( OPTSECT_NET,"autofindkeyserver",((client->autofindkeyserver)?(NULL):("no")), fn);
    if (client->keyproxy[0] || client->keyport)
    {
      modfail += _readwrite_hostname_and_port( 1, fn,
                                OPTSECT_NET, "keyserver",
                                client->keyproxy, sizeof(client->keyproxy),
                                &(client->keyport) );
    }
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
      modfail += (!WritePrivateProfileStringB( OPTSECT_NET, "dialup-watcher", buffer, fn ));
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
    }
  }

  /* ----------------- OPTSECT_CPU ----------------- */

  if ((i=GetPrivateProfileIntB( OPTSECT_CPU, "max-threads",-12345,fn))!=-12345)
  {
    if ((i = GetPrivateProfileIntB( OPTION_SECTION, "numcpu", -12345, fn ))>=0)
    {
      client->numcpu = i;
      modfail += (!WritePrivateProfileIntB( OPTSECT_CPU, "max-threads", client->numcpu, fn));
    }
  }
  if ((i=GetPrivateProfileIntB( OPTSECT_CPU, "priority", -12345, fn ))!=-12345)
  {
    i = GetPrivateProfileIntB( OPTION_SECTION, "niceness", -12345, fn );
    if (i>=0 && i<=2)
    {
      client->priority = i * 4; /* ((i==2)?(8):((i==1)?(4):(0))) */
      modfail += (!WritePrivateProfileIntB( OPTSECT_CPU, "priority", client->priority, fn));
    }
  }

  /* ----------------- OPTSECT_MISC ----------------- */

  if (!GetPrivateProfileStringB(OPTSECT_MISC,"run-time-limit","",buffer,2,fn))
  {
    if (GetPrivateProfileStringB(OPTION_SECTION,"hours","",buffer,sizeof(buffer),fn))
    {
      if ((i = __parse_timestring( buffer, 1 /* compat */ )) > 0)
      {
        client->minutes = i / 60; /* we only want minutes */
        sprintf(buffer,"%u:%02u", (unsigned)(client->minutes/60), (unsigned)(client->minutes%60));
        modfail += (!WritePrivateProfileStringB( OPTSECT_MISC, "run-time-limit", buffer, fn ));
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
      modfail += (!WritePrivateProfileIntB( OPTSECT_MISC, "run-work-limit", i, fn));
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
      client->percentprintingoff = 1;
      modfail += (!WritePrivateProfileStringB( OPTSECT_DISPLAY, "progress-indicator", "off", fn));
    }
  }

  /* ----------------- OPTSECT_TRIGGERS ----------------- */

  /* exit-flag-filename is a bit unusual in that the default (if the key 
     doesn't exist) is not "", but thats all handled in confread().
  */
  if (GetPrivateProfileStringB( OPTSECT_TRIGGERS, "exit-flag-filename", "*", buffer, sizeof(buffer), fn )==1)
  {      /* need a double test to ensure the key doesn't exist */
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
    #if 0 /* faster but causes comment loss */
    if (GetPrivateProfileStringB( OPTION_SECTION, "id", "", buffer, sizeof(buffer), fn ))
    {
      for (i=0;buffer[i];i++)
        buffer[i]=(char)tolower(buffer[i]);
      if (strcmp( buffer, "rc5@distributed.net" ) == 0)
        buffer[0] = '\0';
    }
    if (WritePrivateProfileStringB( OPTION_SECTION, NULL, "", fn))
    {
      modfail += (!WritePrivateProfileStringB( OPTION_SECTION, "id", buffer, fn));
    }
    #else
    static const char *obskeys[]={ /* all in "parameters" section */
             "runhidden", "os2hidden", "win95hidden", "checkpoint2",
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

int ReadConfig(Client *client)
{
  // 1. never printf()/logscreen()/conout() from here
  // 2. never force an option based on the value of some other valid option

  char buffer[64];
  const char *cont_name;
  unsigned int cont_i;
  const char *fn = client->inifilename;

  fn = GetFullPathForFilename( fn );

  if ( access( fn, 0 ) != 0 )
  {
    fn = GetFullPathForFilename( "rc5des" EXTN_SEP "ini" );
    if ( access( fn, 0 ) != 0 )
      return +1; /* fall into config */
  }

  client->randomchanged = 0;
  RefreshRandomPrefix( client );

  TRACE_OUT((+1,"ReadConfig()\n"));

  __remapObsoleteParameters( client, fn ); /* load obsolete options */

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
  client->percentprintingoff = !GetPrivateProfileIntB( OPTSECT_DISPLAY, "progress-indicator", !(client->percentprintingoff), fn );

  /* --------------------- */

  GetPrivateProfileStringB( OPTSECT_TRIGGERS, "exit-flag-filename", DEFAULT_EXITFLAGFILENAME, client->exitflagfile, sizeof(client->exitflagfile), fn );
  GetPrivateProfileStringB( OPTSECT_TRIGGERS, "pause-flag-filename", client->pausefile, client->pausefile, sizeof(client->pausefile), fn );
  client->restartoninichange = GetPrivateProfileIntB( OPTSECT_TRIGGERS, "restart-on-config-file-change", client->restartoninichange, fn );
  GetPrivateProfileStringB( OPTSECT_TRIGGERS, "pause-watch-plist", client->pauseplist, client->pauseplist, sizeof(client->pauseplist), fn );

  /* --------------------- */

  client->offlinemode = GetPrivateProfileIntB( OPTSECT_NET, "disabled", client->offlinemode, fn );
  _readwrite_fwallstuff( 0, fn, client );
  _readwrite_hostname_and_port( 0, fn, OPTSECT_NET, "keyserver",
                                client->keyproxy, sizeof(client->keyproxy),
                                &(client->keyport) );
  //NetOpen() gets (autofindkeyserver)?(""):(client->keyproxy))
  client->autofindkeyserver = GetPrivateProfileIntB( OPTSECT_NET, "autofindkeyserver", client->autofindkeyserver, fn );
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

  if (GetPrivateProfileStringB(OPTSECT_MISC,"run-time-limit","",buffer,sizeof(buffer),fn))
  {
    if ((client->minutes = __parse_timestring( buffer, 0 )) < 0)
      client->minutes = 0;
    client->minutes /= 60; /* we only want minutes */
  }
  client->blockcount = GetPrivateProfileIntB( OPTSECT_MISC, "run-work-limit", client->blockcount, fn );

  TRACE_OUT((0,"ReadConfig() [2 end]\n"));

  /* --------------------- */

  _readwrite_hostname_and_port( 0, fn, OPTSECT_LOG, "mail-log-via",
                                client->smtpsrvr, sizeof(client->smtpsrvr),
                                &(client->smtpport) );
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
  GetPrivateProfileStringB( OPTSECT_MISC, "project-priority", "", buffer, sizeof(buffer), fn );
  projectmap_build(client->loadorder_map, buffer);

  TRACE_OUT((0,"ReadConfig() [3 begin]\n"));

  for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
  {
    if ((cont_name = __getprojsectname(cont_i)) != ((const char *)0))
    {
      if (cont_i != OGR)
      {
        client->coretypes[cont_i] =
           GetPrivateProfileIntB(cont_name, "core",
                         client->coretypes[cont_i],fn);
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
          WritePrivateProfileIntB( OPTSECT_BUFFERS, "frequent-threshold-checks", client->connectoften, fn );
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

  return 0;
}

// --------------------------------------------------------------------------

//conditional (if exist || if !default) ini write functions

static void __XSetProfileStr( const char *sect, const char *key,
            const char *newval, const char *fn, const char *defval )
{
  if (sect == NULL)
    sect = OPTION_SECTION;
  if (defval == NULL)
    defval = "";
  int dowrite = (strcmp( newval, defval )!=0);
  if (!dowrite)
  {
    char buffer[4];
    dowrite = (GetPrivateProfileStringB( sect, key, "=", 
                    buffer, sizeof(buffer), fn ) !=1 );
    if (!dowrite) dowrite = (buffer[0] != '=' );
  }  
  if (dowrite)
    WritePrivateProfileStringB( sect, key, newval, fn );
  return;
}

static void __XSetProfileInt( const char *sect, const char *key,
          long newval, const char *fn, long defval, int asonoff )
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

//Some OS's write run-time stuff to the .ini, so we protect
//the ini by only allowing that client's internal settings to change.

int WriteConfig(Client *client, int writefull /* defaults to 0*/)
{
  char buffer[64]; int i;
  unsigned int cont_i;
  const char *p;
  const char *fn = client->inifilename;

  fn = GetFullPathForFilename( fn );
  if ( !writefull && access( fn, 0 )!=0 )
    writefull = 1;

  if (!WritePrivateProfileStringB( OPTION_SECTION, "id",
    ((strcmp( client->id,"rc5@distributed.net")==0)?(""):(client->id)), fn ))
    return -1; //failed

  if (__remapObsoleteParameters( client, fn ) < 0)
    return -1; //file is read-only

  client->randomchanged = 1;
  RefreshRandomPrefix( client );

  if (writefull != 0)
  {
    /* --- CONF_MENU_BUFF -- */
    __XSetProfileInt( OPTSECT_BUFFERS, "buffer-only-in-memory", (client->nodiskbuffers)?(1):(0), fn, 0, 'y' );
    __XSetProfileStr( OPTSECT_BUFFERS, "buffer-file-basename", client->in_buffer_basename, fn, NULL );
    __XSetProfileStr( OPTSECT_BUFFERS, "output-file-basename", client->out_buffer_basename, fn, NULL );
    __XSetProfileStr( OPTSECT_BUFFERS, "checkpoint-filename", client->checkpoint_file, fn, NULL );
    __XSetProfileInt( OPTSECT_BUFFERS, "allow-update-from-altbuffer", !(client->noupdatefromfile), fn, 1, 1 );
    __XSetProfileStr( OPTSECT_BUFFERS, "alternate-buffer-directory", client->remote_update_dir, fn, NULL );
    __XSetProfileInt( OPTSECT_BUFFERS, "frequent-threshold-checks", client->connectoften, fn, 0, 0 );

    /* --- CONF_MENU_MISC __ */

    strcpy(buffer,projectmap_expand(NULL));
    __XSetProfileStr( OPTSECT_MISC, "project-priority", projectmap_expand(client->loadorder_map), fn, buffer );
    if (client->minutes!=0 || GetPrivateProfileStringB(OPTSECT_MISC,"run-time-limit","",buffer,2,fn))
    {
      sprintf(buffer,"%u:%02u", (unsigned)(client->minutes/60), (unsigned)(client->minutes%60));
      WritePrivateProfileStringB( OPTSECT_MISC, "run-time-limit", buffer, fn );
    }
    __XSetProfileInt( OPTSECT_MISC, "run-work-limit", client->blockcount, fn, 0, 0 );

    __XSetProfileInt( OPTSECT_TRIGGERS, "restart-on-config-file-change", client->restartoninichange, fn, 0, 'n' );
    __XSetProfileStr( OPTSECT_TRIGGERS, "exit-flag-filename", client->exitflagfile, fn, DEFAULT_EXITFLAGFILENAME );
    __XSetProfileStr( OPTSECT_TRIGGERS, "pause-flag-filename", client->pausefile, fn, NULL );
    __XSetProfileStr( OPTSECT_TRIGGERS, "pause-watch-plist", client->pauseplist, fn, NULL );

    __XSetProfileInt( OPTSECT_DISPLAY, "detached", client->quietmode, fn, 0, 'y' );
    __XSetProfileInt( OPTSECT_DISPLAY, "progress-indicator", !client->percentprintingoff, fn, 1, 'o' );

    /* --- CONF_MENU_PERF -- */

    __XSetProfileInt( OPTSECT_CPU, "max-threads", client->numcpu, fn, -1, 0 );
    __XSetProfileInt( OPTSECT_CPU, "priority", client->priority, fn, 0, 0);

    /* more buffer stuff */

    for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
    {
      if ((p =  __getprojsectname(cont_i)) != ((const char *)0))
      {
        __XSetProfileInt( p, "fetch-workunit-threshold", client->inthreshold[cont_i], fn,  0, 0 );
        #if !defined(NO_OUTBUFFER_THRESHOLDS)          
        __XSetProfileInt( p, "flush-workunit-threshold", client->outthreshold[cont_i], fn, 0, 0 );
        #endif
        __XSetProfileInt( p, "fetch-time-threshold", client->timethreshold[cont_i], fn, 0, 0 );
        if (cont_i != OGR)
        {
          __XSetProfileInt( p, "core", client->coretypes[cont_i], fn, -1, 0 );
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
                                &(client->keyport) );

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

    if ((i = client->smtpport) == 25) i = 0;
    _readwrite_hostname_and_port( 1, fn, OPTSECT_LOG, "mail-log-via",
                                client->smtpsrvr, sizeof(client->smtpsrvr),
                                &(i) );
    __XSetProfileInt( OPTSECT_LOG, "mail-log-max", client->messagelen, fn, 0, 0);
    __XSetProfileStr( OPTSECT_LOG, "mail-log-from", client->smtpfrom, fn, NULL);
    __XSetProfileStr( OPTSECT_LOG, "mail-log-dest", client->smtpdest, fn, NULL);

    __XSetProfileStr( OPTSECT_LOG, "log-file-limit", client->logfilelimit, fn, NULL );
    __XSetProfileStr( OPTSECT_LOG, "log-file", client->logname, fn, NULL );
    if ((client->logfiletype[0] && strcmp(client->logfiletype,"none")!=0) ||
      GetPrivateProfileStringB(OPTSECT_LOG,"log-file-type","",buffer,2,fn))
      WritePrivateProfileStringB( OPTSECT_LOG,"log-file-type", client->logfiletype, fn );

  } /* if (writefull != 0) */

  return 0;
}

// --------------------------------------------------------------------------

// update contestdone and randomprefix .ini entries
void RefreshRandomPrefix( Client *client )
{
  // we need to use no_trigger when reading/writing full configs

  if (client->stopiniio == 0 && client->nodiskbuffers == 0)
  {
    const char *fn = client->inifilename;
    fn = GetFullPathForFilename( fn );

    if ( client->randomchanged == 0 ) /* load */
    {
      if ( access( fn, 0 )!=0 )
        return;

      client->randomprefix =  GetPrivateProfileIntB(__getprojsectname(RC5), "randomprefix",
                                                 client->randomprefix, fn);
      client->scheduledupdatetime = GetPrivateProfileIntB(OPTSECT_NET,
                      "scheduledupdatetime", client->scheduledupdatetime, fn);

      client->rc564closed = (GetPrivateProfileIntB(__getprojsectname(RC5), "closed", 0, fn )!=0);
    }

    if (client->randomchanged)
    {
      client->randomchanged = 0;
      if (client->randomprefix != 100 || GetPrivateProfileIntB(__getprojsectname(RC5), "randomprefix", 0, fn))
      {
        if (!WritePrivateProfileIntB(__getprojsectname(RC5),"randomprefix",client->randomprefix,fn))
        {
          //client->stopiniio == 1;
          return; //write fail
        }
      }
      WritePrivateProfileStringB(__getprojsectname(RC5),"closed",(client->rc564closed)?("1"):(NULL), fn );
      if (client->scheduledupdatetime == 0)
        WritePrivateProfileStringB(OPTSECT_NET, "scheduledupdatetime", NULL, fn );
      else
        WritePrivateProfileIntB(OPTSECT_NET, "scheduledupdatetime", client->scheduledupdatetime, fn );
    }
  }
  return;
}
