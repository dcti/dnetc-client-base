/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *confrwv_cpp(void) {
return "@(#)$Id: confrwv.cpp,v 1.70 1999/11/23 15:41:36 cyp Exp $"; }

//#define TRACE

#include "cputypes.h"
#include "client.h"    // Client class
#include "baseincs.h"  // atoi() etc
#include "iniread.h"   // [Get|Write]Profile[Int|String]()
#include "pathwork.h"  // GetFullPathForFilename()
#include "util.h"      // projectmap_*() and trace
#include "lurk.h"      // lurk stuff
#include "base64.h"    // base64_[en|de]code()
#include "cpucheck.h"  // GetProcessorType() for mmx stuff
#include "triggers.h"  // RaiseRestartRequestTrigger()
#include "clicdata.h"  // CliGetContestNameFromID()
#include "cmpidefs.h"  // strcmpi()
#include "confrwv.h"   // Ourselves

/* ------------------------------------------------------------------------ */

static const char *OPTION_SECTION  = "parameters";
static const char *OPTSECT_NET     = "networking";
static const char *OPTSECT_BUFFERS = "buffers";
static const char *OPTSECT_MISC    = "misc";
static const char *OPTSECT_LOG     = "logging";

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
      if (GetPrivateProfileStringB( OPTSECT_NET, "firewall-host", "", buffer, sizeof(buffer), fn ))
      {
        that = strchr( buffer, ':' );
        if (that)
        {
          *that++ = '\0';
          client->httpport = atoi(that);
        }
        strncpy( client->httpproxy, buffer, sizeof(client->httpproxy));
        client->httpproxy[sizeof(client->httpproxy)-1] = '\0';
      }
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
      TRACE_OUT((+1,"oldform: _readwrite_fwallstuff(asread)\n"));

      client->uuehttpmode = GetPrivateProfileIntB( OPTION_SECTION, "uuehttpmode", client->uuehttpmode, fn );
      TRACE_OUT((0,"old uuehttpmode=%d\n",client->uuehttpmode));
      GetPrivateProfileStringB( OPTION_SECTION, "httpproxy", client->httpproxy, client->httpproxy, sizeof(client->httpproxy), fn );
      TRACE_OUT((0,"old httpproxy=\"%s\"\n",client->httpproxy));
      client->httpport = GetPrivateProfileIntB( OPTION_SECTION, "httpport", client->httpport, fn );
      TRACE_OUT((0,"old httpport=%d\n",client->httpport));
      GetPrivateProfileStringB( OPTION_SECTION, "httpid", client->httpid, client->httpid, sizeof(client->httpid), fn );
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
      aswrite = 1; //rewrite the stuff
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

    buffer[0] = '\0';
    if (client->httpproxy[0])
    {
      strcpy(buffer,client->httpproxy);
      if (client->httpport)
        sprintf(&buffer[strlen(buffer)],":%d",client->httpport);
    }
    if (buffer[0] || 
       GetPrivateProfileStringB( OPTSECT_NET, "firewall-host", "", scratch, sizeof(scratch), fn ))
      WritePrivateProfileStringB( OPTSECT_NET, "firewall-host", buffer, fn);
       
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

static int __remapObsoleteParameters( Client *client, const char *fn ) /* <0 if failed */
{
  static const char *obskeys[]={ /* all in "parameters" section */
             "runhidden", "os2hidden", "win95hidden", "checkpoint2",
             "niceness", "processdes", "timeslice", "runbuffers",
             "contestdone" /* now in "rc564" */, "contestdone2", 
             "contestdone3", "contestdoneflags", "descontestclosed",
             "scheduledupdatetime" /* now in OPTSECT_NET */,
             "processdes", "usemmx", "runoffline", "in","out","in2","out2",
             "in3","out3","nodisk", "dialwhenneeded","connectionname",
             "cputype","threshold","threshold2","preferredblocksize",
             "logname" };
  char buffer[128];
  char *p;
  unsigned int ui;
  int i, modfail = 0;

  TRACE_OUT((+1,"__remapObsoleteParameters()\n"));

  if (!GetPrivateProfileStringB( OPTSECT_LOG, "log-file", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "logname", "", buffer, sizeof(buffer), fn ))
    {
      strncpy( client->logname, buffer, sizeof(client->logname) );
      client->logname[sizeof(client->logname)-1] = '\0';
      modfail += (!WritePrivateProfileStringB( OPTSECT_LOG, "log-file", buffer, fn ));
    }  
  }

  #if (CLIENT_CPU != CPU_ALPHA) /* no RC5 cputype->coretype mapping for Alpha */
  if (!GetPrivateProfileStringB( __getprojsectname(RC5), "core", "", buffer, sizeof(buffer), fn ))
  {
    if ((i = GetPrivateProfileIntB(OPTION_SECTION, "cputype", -1, fn ))!=-1)
    {
      client->coretypes[RC5] = i;
      modfail += (!WritePrivateProfileIntB( __getprojsectname(RC5), "core", i, fn));
    }
  }
  #endif

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

  for (ui=0; ui<2; ui++)
  {
    i = ((ui)?(RC5):(DES));
    p = (char*)((ui)?("threshold"):("threshold2"));
    if (!GetPrivateProfileStringB( __getprojsectname(i), "fetch-threshold", "", buffer, sizeof(buffer), fn )
     && !GetPrivateProfileStringB( __getprojsectname(i), "flush-threshold", "", buffer, sizeof(buffer), fn ))
    {
      if (GetPrivateProfileStringB( OPTION_SECTION, p, "", buffer, sizeof(buffer), fn ))
      {
        if ((i = atoi(buffer))>0)
        {                                    
          int oldstyle_inout[2];
          oldstyle_inout[0] = oldstyle_inout[1] = i;
          if ((p = strchr( buffer, ':' )) != ((char *)0))
          {
            if ((i = atoi( p+1 ))>0)
              oldstyle_inout[1] = i;
          }
          i = ((ui)?(RC5):(DES));
          if (oldstyle_inout[0] != BUFTHRESHOLD_DEFAULT)
          {
            client->inthreshold[i] = oldstyle_inout[0];
            modfail += (!WritePrivateProfileIntB( __getprojsectname(i), "fetch-threshold", client->inthreshold[i], fn));
          }
          if (oldstyle_inout[1] != BUFTHRESHOLD_DEFAULT)
          {
            client->outthreshold[i] = oldstyle_inout[1];
            modfail += (!WritePrivateProfileIntB( __getprojsectname(i), "flush-threshold", client->outthreshold[i], fn));
          }
        }          
      }
    }
  }

  if (!GetPrivateProfileStringB( OPTION_SECTION, "quiet", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileIntB(OPTION_SECTION, "runhidden", 0, fn ) ||
        GetPrivateProfileIntB(OPTION_SECTION, "os2hidden", 0, fn ) ||
        GetPrivateProfileIntB(OPTION_SECTION, "win95hidden", 0, fn ))
    {
      client->quietmode = 1;
      modfail += (!WritePrivateProfileIntB( OPTION_SECTION, "quiet", 1, fn));
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
  
  if (GetPrivateProfileIntB( OPTION_SECTION, "descontestclosed", 0, fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "in2", "", buffer, sizeof(buffer), fn ))
      unlink( GetFullPathForFilename( buffer ) );
    if (GetPrivateProfileStringB( OPTION_SECTION, "out2", "", buffer, sizeof(buffer), fn ))
      unlink( GetFullPathForFilename( buffer ) );
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
  if (!GetPrivateProfileStringB( OPTSECT_NET, "enable-start-stop", "", buffer, sizeof(buffer), fn ))
  {
    if ((i = GetPrivateProfileIntB( OPTION_SECTION, "dialwhenneeded", -123, fn )) != -123)
    {
      #ifdef LURK
      dialup.dialwhenneeded = i;
      #endif
      if (i)
        modfail += (!WritePrivateProfileStringB( OPTSECT_NET, "enable-start-stop", "yes", fn ));
    }
  }   
  
  if (!GetPrivateProfileStringB( OPTSECT_NET, "dialup-profile", "", buffer, sizeof(buffer), fn ))
  {
    if (GetPrivateProfileStringB( OPTSECT_MISC, "connectionname", "", buffer, sizeof(buffer), fn ))
    {
      #ifdef LURK
      strncpy( dialup.connprofile, buffer, sizeof(dialup.connprofile) );
      dialup.connprofile[sizeof(dialup.connprofile)-1]='\0';
      #endif
      modfail += (!WritePrivateProfileStringB( OPTSECT_NET, "dialup-profile", buffer, fn ));
    }
  }
  
  if (GetPrivateProfileIntB( OPTION_SECTION, "runbuffers", 0, fn ))
  {
    client->blockcount = -1;
    modfail += (!WritePrivateProfileStringB( OPTION_SECTION, "count", "-1", fn ));
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
  
  /* unconditional deletion of obsolete keys */
  for (ui = 0; ui < (sizeof(obskeys) / sizeof(obskeys[0])); ui++)
    modfail += (!WritePrivateProfileStringB( OPTION_SECTION, obskeys[ui], NULL, fn ));

  TRACE_OUT((-1,"__remapObsoleteParameters() modif failed?=%d\n", modfail));
  
  if (modfail)
    return -1;
  return 0;
}  


/* ------------------------------------------------------------------------ */

static int confopt_IsHostnameDNetHost( const char * hostname )
{
  unsigned int len;
  const char sig[]="distributed.net";

  if (!hostname || !*hostname)
    return 1;
  if (isdigit( *hostname )) //IP address
    return 0;
  len = strlen( hostname );
  return (len > (sizeof( sig )-1) &&
      strcmpi( &hostname[(len-(sizeof( sig )-1))], sig ) == 0);
}


/* ------------------------------------------------------------------------ */

int ReadConfig(Client *client) 
{
  // 1. never printf()/logscreen()/conout() from here
  // 2. never force an option based on the value of some other valid option

  char buffer[64];
  const char *sect = OPTION_SECTION;
  const char *cont_name;
  unsigned int cont_i;
  const char *fn = client->inifilename;
  char *p; int i;

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

  if (GetPrivateProfileStringB( sect, "id", "", client->id, sizeof(client->id), fn ))
  {
    if (strcmp( client->id, "rc5@distributed.net" ) == 0)
      client->id[0] = '\0';
  }
  if (GetPrivateProfileStringB( sect, "hours", "", buffer, sizeof(buffer), fn ))
  {
    client->minutes = (atoi(buffer) * 60);
    if ((p = strchr( buffer, ':' )) == NULL)
      p = strchr( buffer, '.' );
    if (client->minutes < 0)
      client->minutes = 0;
    else if (p != NULL && strlen(p) == 3 && isdigit(p[1]) && isdigit(p[2]) && atoi(p+1)<60)
      client->minutes += atoi(p+1);
    else if (p != NULL) //strlen/isdigit check failed
      client->minutes = 0;
  }
  
  _readwrite_fwallstuff( 0, fn, client );

  client->keyport = GetPrivateProfileIntB( sect, "keyport", client->keyport, fn );
  GetPrivateProfileStringB( sect, "keyproxy", client->keyproxy, client->keyproxy, sizeof(client->keyproxy), fn );
  if (strcmpi(client->keyproxy,"auto")==0 || strcmpi(client->keyproxy,"(auto)")==0)
    client->keyproxy[0]=0; //one config version accidentally wrote "auto" out
  client->nettimeout = GetPrivateProfileIntB( sect, "nettimeout", client->nettimeout, fn );
  
  client->autofindkeyserver = (client->keyproxy[0]==0 || 
    strcmpi( client->keyproxy, "rc5proxy.distributed.net" )==0 ||
    ( confopt_IsHostnameDNetHost(client->keyproxy) &&
    GetPrivateProfileIntB( OPTSECT_NET, "autofindkeyserver", 1, fn ) ));

  i = GetPrivateProfileIntB( sect, "niceness", -12345, fn );
  if (i>=0 && i<=2) client->priority = ((i==2)?(8):((i==1)?(4):(0)));
  client->priority = GetPrivateProfileIntB( "processor-usage", "priority", client->priority, fn );
  client->numcpu = GetPrivateProfileIntB( sect, "numcpu", client->numcpu, fn );

  for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
  {
    if ((cont_name = __getprojsectname(cont_i)) != ((const char *)0))
    {
      client->inthreshold[cont_i] = 
           GetPrivateProfileIntB(cont_name, "fetch-threshold",
                         client->inthreshold[cont_i], fn );
      client->outthreshold[cont_i] = 
           GetPrivateProfileIntB(cont_name, "flush-threshold",
                         client->outthreshold[cont_i], fn );
      if (cont_i != OGR)
      {                         
        client->coretypes[cont_i] = 
           GetPrivateProfileIntB(cont_name, "core",
                         client->coretypes[cont_i],fn);
        client->preferred_blocksize[cont_i] = 
           GetPrivateProfileIntB(cont_name, "preferred-blocksize",
                         client->preferred_blocksize[cont_i], fn );
      }                         
    }
  }

  TRACE_OUT((0,"ReadConfig() [2 begin]\n"));

  GetPrivateProfileStringB( OPTSECT_MISC, "project-priority", "", buffer, sizeof(buffer), fn );
  projectmap_build(client->loadorder_map, buffer);

  TRACE_OUT((0,"ReadConfig() [2 end]\n"));
    
  client->messagelen = GetPrivateProfileIntB( sect, "messagelen", client->messagelen, fn );
  client->smtpport = GetPrivateProfileIntB( sect, "smtpport", client->smtpport, fn );
  GetPrivateProfileStringB( sect, "smtpsrvr", client->smtpsrvr, client->smtpsrvr, sizeof(client->smtpsrvr), fn );
  GetPrivateProfileStringB( sect, "smtpfrom", client->smtpfrom, client->smtpfrom, sizeof(client->smtpfrom), fn );
  GetPrivateProfileStringB( sect, "smtpdest", client->smtpdest, client->smtpdest, sizeof(client->smtpdest), fn );

  client->blockcount = GetPrivateProfileIntB( sect, "count", client->blockcount, fn );
  if (GetPrivateProfileIntB( sect, "runbuffers", 0, fn ))
    client->blockcount = -1;
  
  client->offlinemode = GetPrivateProfileIntB( OPTSECT_NET, "disabled", client->offlinemode, fn );
  client->noupdatefromfile = (!GetPrivateProfileIntB( OPTSECT_BUFFERS, "allow-update-from-altbuffer", !(client->noupdatefromfile), fn ));
  GetPrivateProfileStringB( OPTSECT_BUFFERS, "alternate-buffer-directory", client->remote_update_dir, client->remote_update_dir, sizeof(client->remote_update_dir), fn );

  client->percentprintingoff = GetPrivateProfileIntB( sect, "percentoff", client->percentprintingoff, fn );
  client->connectoften = GetPrivateProfileIntB( sect, "frequent", client->connectoften , fn );
  client->quietmode = GetPrivateProfileIntB( sect, "quiet", client->quietmode, fn );
  client->nofallback = GetPrivateProfileIntB( sect, "nofallback", client->nofallback, fn );
  client->noexitfilecheck = GetPrivateProfileIntB( sect, "noexitfilecheck", client->noexitfilecheck, fn );

  GetPrivateProfileStringB( OPTSECT_LOG, "log-file", client->logname, client->logname, sizeof(client->logname), fn );
  GetPrivateProfileStringB( OPTSECT_LOG, "log-file-type", client->logfiletype, client->logfiletype, sizeof(client->logfiletype), fn );
  GetPrivateProfileStringB( OPTSECT_LOG, "log-file-limit", client->logfilelimit, client->logfilelimit, sizeof(client->logfilelimit), fn );

  GetPrivateProfileStringB( sect, "pausefile", client->pausefile, client->pausefile, sizeof(client->pausefile), fn );
  GetPrivateProfileStringB( sect, "checkpointfile", client->checkpoint_file, client->checkpoint_file, sizeof(client->checkpoint_file), fn );

  client->nodiskbuffers = GetPrivateProfileIntB( OPTSECT_BUFFERS, "buffer-only-in-memory", client->nodiskbuffers , fn );
  GetPrivateProfileStringB( OPTSECT_BUFFERS, "buffer-file-basename", client->in_buffer_basename, client->in_buffer_basename, sizeof(client->in_buffer_basename), fn );
  GetPrivateProfileStringB( OPTSECT_BUFFERS, "output-file-basename", client->out_buffer_basename, client->out_buffer_basename, sizeof(client->out_buffer_basename), fn );

  TRACE_OUT((0,"ReadConfig() [3 begin]\n"));

  #if defined(LURK)
  i = dialup.GetCapabilityFlags();
  dialup.lurkmode = 0;
  if ((i & CONNECT_LURKONLY)!=0 && GetPrivateProfileIntB( sect, "lurkonly", 0, fn ))
    { dialup.lurkmode = CONNECT_LURKONLY; }
  else if ((i & CONNECT_LURK)!=0 && GetPrivateProfileIntB( sect, "lurk", 0, fn ))
    dialup.lurkmode = CONNECT_LURK;
  if ((i & CONNECT_IFACEMASK)!=0)
    GetPrivateProfileStringB( OPTSECT_NET, "interfaces-to-watch", dialup.connifacemask,
                              dialup.connifacemask, sizeof(dialup.connifacemask), fn );
  if ((i & CONNECT_DOD)!=0)
  {
    dialup.dialwhenneeded = GetPrivateProfileIntB( OPTSECT_NET, "enable-start-stop", 0, fn );
    if ((i & CONNECT_DODBYSCRIPT)!=0)
    {
      GetPrivateProfileStringB( OPTSECT_NET, "dialup-start-cmd", dialup.connstartcmd, dialup.connstartcmd, sizeof(dialup.connstartcmd), fn );
      GetPrivateProfileStringB( OPTSECT_NET, "dialup-stop-cmd", dialup.connstopcmd, dialup.connstopcmd, sizeof(dialup.connstopcmd), fn );
    }
    if ((i & CONNECT_DODBYPROFILE)!=0)
      GetPrivateProfileStringB( sect, "connectionname", dialup.connprofile, dialup.connprofile, sizeof(dialup.connprofile), fn );
  }
  #endif /* LURK */

  TRACE_OUT((0,"ReadConfig() [3 end]\n"));
  TRACE_OUT((-1,"ReadConfig()\n"));

  return 0;
}

// --------------------------------------------------------------------------

//conditional (if exist || if !default) ini write functions

static void __XSetProfileStr( const char *sect, const char *key, 
            const char *newval, const char *fn, const char *defval )
{
  char buffer[4];
  if (sect == NULL) 
    sect = OPTION_SECTION;
  if (defval == NULL)
    defval = "";
  int dowrite = (strcmp( newval, defval )!=0);
  if (!dowrite)
    dowrite = (GetPrivateProfileStringB( sect, key, "", buffer, 2, fn )!=0);
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
  const char *cont_name;
  const char *sect = OPTION_SECTION;
  const char *fn = client->inifilename;
  
  fn = GetFullPathForFilename( fn );
  if ( !writefull && access( fn, 0 )!=0 )
    writefull = 1;
 
  if (!WritePrivateProfileStringB( sect, "id", 
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

    __XSetProfileInt( OPTSECT_BUFFERS, "allow-update-from-altbuffer", !(client->noupdatefromfile), fn, 1, 1 );
    __XSetProfileStr( OPTSECT_BUFFERS, "alternate-buffer-directory", client->remote_update_dir, fn, NULL );

    strcpy(buffer,projectmap_expand(NULL));
    __XSetProfileStr( OPTSECT_MISC, "project-priority", projectmap_expand(client->loadorder_map), fn, buffer );

    __XSetProfileInt( sect, "frequent", client->connectoften, fn, 0, 1 );
    
    __XSetProfileStr( sect, "checkpointfile", client->checkpoint_file, fn, NULL );
    
    /* --- CONF_MENU_MISC __ */

    if (client->minutes!=0 || GetPrivateProfileStringB(sect,"hours","",buffer,2,fn))
    {
      sprintf(buffer,"%u:%02u", (unsigned)(client->minutes/60), (unsigned)(client->minutes%60)); 
      WritePrivateProfileStringB( sect, "hours", buffer, fn );
    }
    __XSetProfileInt( sect, "count", client->blockcount, fn, 0, 0 );
    __XSetProfileStr( sect, "pausefile", client->pausefile, fn, NULL );
    __XSetProfileInt( sect, "quiet", client->quietmode, fn, 0, 1 );
    __XSetProfileInt( sect, "noexitfilecheck", client->noexitfilecheck, fn, 0, 1 );
    __XSetProfileInt( sect, "percentoff", client->percentprintingoff, fn, 0, 1 );
    
    /* --- CONF_MENU_PERF -- */

    __XSetProfileInt( sect, "numcpu", client->numcpu, fn, -1, 0 );
    __XSetProfileInt( "processor-usage", "priority", client->priority, fn, 0, 0);

    for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
    {
      if ((cont_name =  __getprojsectname(cont_i)) != ((const char *)0))
      {
        __XSetProfileInt( cont_name, "fetch-threshold", client->inthreshold[cont_i], fn,  BUFTHRESHOLD_DEFAULT, 0 );
        __XSetProfileInt( cont_name, "flush-threshold", client->outthreshold[cont_i], fn, BUFTHRESHOLD_DEFAULT, 0 );
        if (cont_i != OGR)
        {
          __XSetProfileInt( cont_name, "core", client->coretypes[cont_i], fn, -1, 0 );
          __XSetProfileInt( cont_name, "preferred-blocksize", 
            client->preferred_blocksize[cont_i], fn, PREFERREDBLOCKSIZE_DEFAULT, 0 );
        }
      }
    }

    /* --- CONF_MENU_NET -- */

    __XSetProfileInt( OPTSECT_NET, "disabled", client->offlinemode, fn, 0, 'n');
    __XSetProfileInt( sect, "nettimeout", client->nettimeout, fn, 60, 0);
    __XSetProfileInt( sect, "nofallback", client->nofallback, fn, 0, 1);

    char *af=NULL, *host=client->keyproxy, *port = buffer;
    i = 0; while (host[i] && isspace(host[i])) i++;
    if (host[i]=='\0' || client->autofindkeyserver)
      host = NULL; //delete keys so that old inis stay compatible and default.
    else if (confopt_IsHostnameDNetHost(host)) //make clear that name==port
        { af = "0"; if (client->keyport != 3064) port = NULL; }
    if (port!=NULL && client->keyport==0 && !GetPrivateProfileIntB(sect,"keyport",0,fn))
      port = NULL;
    if (port!=NULL) sprintf(port,"%d",((int)(client->keyport)));

    WritePrivateProfileStringB( OPTSECT_NET, "autofindkeyserver", af, fn );
    WritePrivateProfileStringB( sect, "keyport", port, fn );
    WritePrivateProfileStringB( sect, "keyproxy", host, fn );
    _readwrite_fwallstuff( 1, fn, client );

    #if defined(LURK)
    i = dialup.GetCapabilityFlags();
    WritePrivateProfileStringB( sect, "lurk", (dialup.lurkmode==CONNECT_LURK)?("1"):(NULL), fn );
    WritePrivateProfileStringB( sect, "lurkonly", (dialup.lurkmode==CONNECT_LURKONLY)?("1"):(NULL), fn );
    if ((i & CONNECT_IFACEMASK) != 0)
      __XSetProfileStr( OPTSECT_NET, "interfaces-to-watch", dialup.connifacemask, fn, NULL );
    if ((i & CONNECT_DOD) != 0)
    {
      __XSetProfileInt( OPTSECT_NET, "enable-start-stop", (dialup.dialwhenneeded!=0), fn, 0, 'n' );
      if ((i & CONNECT_DODBYPROFILE) != 0)
        __XSetProfileStr( OPTSECT_NET, "dialup-profile", dialup.connprofile, fn, NULL );
      if ((i & CONNECT_DODBYSCRIPT) != 0)
      {
        __XSetProfileStr( OPTSECT_NET, "dialup-start-cmd", dialup.connstartcmd, fn, NULL );
        __XSetProfileStr( OPTSECT_NET, "dialup-stop-cmd", dialup.connstopcmd, fn, NULL );
      }
    }
    #endif // defined LURK
    
    /* --- CONF_MENU_LOG -- */

    __XSetProfileStr( OPTSECT_LOG, "log-file", client->logname, fn, NULL );
    if ((client->logfiletype[0] && strcmpi(client->logfiletype,"none")!=0) || 
      GetPrivateProfileStringB(OPTSECT_LOG,"log-file-type","",buffer,2,fn))
      WritePrivateProfileStringB( OPTSECT_LOG,"log-file-type", client->logfiletype, fn );
    __XSetProfileStr( OPTSECT_LOG, "log-file-limit", client->logfilelimit, fn, NULL );
    __XSetProfileInt( sect, "messagelen", client->messagelen, fn, 0, 0);
    __XSetProfileStr( sect, "smtpsrvr", client->smtpsrvr, fn, NULL);
    __XSetProfileStr( sect, "smtpfrom", client->smtpfrom, fn, NULL);
    __XSetProfileStr( sect, "smtpdest", client->smtpdest, fn, NULL);
    __XSetProfileInt( sect, "smtpport", client->smtpport, fn, 25, 0);

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


