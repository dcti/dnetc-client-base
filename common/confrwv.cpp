/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *confrwv_cpp(void) {
return "@(#)$Id: confrwv.cpp,v 1.54 1999/04/16 00:03:46 cyp Exp $"; }

#include "cputypes.h"
#include "client.h"    // Client class
#include "baseincs.h"  // atoi() etc
#include "iniread.h"   // [Get|Write]Profile[Int|String]()
#include "pathwork.h"  // GetFullPathForFilename()
#include "util.h"      // projectmap_*()
#include "lurk.h"      // lurk stuff
#include "cpucheck.h"  // GetProcessorType() for mmx stuff
#include "triggers.h"  // RaiseRestartRequestTrigger()
#include "cmpidefs.h"  // strcmpi()
#include "confrwv.h"   // Ourselves

/* ------------------------------------------------------------------------ */

static const char *OPTION_SECTION  = "parameters";
static const char *OPTSECT_NET     = "networking";
static const char *OPTSECT_BUFFERS = "buffers";
static const char *OPTSECT_MISC    = "misc";
static const char *OPTSECT_RC5     = "rc5";

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
             "in3","out3","nodisk", "dialwhenneeded","connectionname" };
  char buffer[128];
  int i, modfail = 0;

  if (!GetPrivateProfileStringB( OPTION_SECTION, "quiet", "", buffer, 1, fn ))
  {
    if (GetPrivateProfileIntB(OPTION_SECTION, "runhidden", 0, fn ) ||
        GetPrivateProfileIntB(OPTION_SECTION, "os2hidden", 0, fn ) ||
        GetPrivateProfileIntB(OPTION_SECTION, "win95hidden", 0, fn ))
    {
      client->quietmode = 1;
      modfail+=(!WritePrivateProfileIntB( OPTSECT_BUFFERS, "quiet", 1, fn));
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
  
  if (!GetPrivateProfileStringB( OPTSECT_BUFFERS, "buffer-file-basename", "", buffer, 1, fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "in", "", buffer, sizeof(buffer), fn ))
    {
      int basenameoffset = GetFilenameBaseOffset( buffer );
      const char *suffixsep = EXTN_SEP;
      char *suffix = strrchr( &buffer[basenameoffset], *suffixsep );
      if (suffix)
        *suffix = '\0';
      if (strcmp( buffer, "buff-in" ) != 0)
      {
        strcpy( client->in_buffer_basename, buffer );
        modfail += (!WritePrivateProfileStringB( OPTSECT_BUFFERS, "buffer-file-basename", buffer, fn ));
      }        
    }
  }  
  
  if (!GetPrivateProfileStringB( OPTSECT_BUFFERS, "output-file-basename", "", buffer, 1, fn ))
  {
    if (GetPrivateProfileStringB( OPTION_SECTION, "out", "", buffer, sizeof(buffer), fn ))
    {
      int basenameoffset = GetFilenameBaseOffset( buffer );
      const char *suffixsep = EXTN_SEP;
      char *suffix = strrchr( &buffer[basenameoffset], *suffixsep );
      if (suffix)
        *suffix = '\0';
      if (strcmp( buffer, "buff-out" ) != 0)
      {
        strcpy( client->out_buffer_basename, buffer );
        modfail += (!WritePrivateProfileStringB( OPTSECT_BUFFERS, "output-file-basename", buffer, fn ));
      }        
    }
  }  
  
  if (!GetPrivateProfileStringB( OPTSECT_NET, "enable-start-stop", "", buffer, 1, fn ))
  {
    if ((i = GetPrivateProfileIntB( OPTION_SECTION, "dialwhenneeded", -123, fn ))!=-123)
    {
      #ifdef LURK
      dialup.dialwhenneeded = i;
      #endif
      if (i)
        modfail += (!WritePrivateProfileStringB( OPTSECT_NET, "enable-start-stop", "yes", fn ));
    }
  }   
  
  if (!GetPrivateProfileStringB( OPTSECT_NET, "dialup-profile", "", buffer, 1, fn ))
  {
    if (GetPrivateProfileStringB( OPTSECT_MISC, "connectionname", "", buffer, 1, fn ))
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

  if (!GetPrivateProfileStringB( OPTSECT_MISC, "project-priority", "", buffer, 1, fn ))
  {
    if (GetPrivateProfileIntB( OPTION_SECTION, "processdes", 1, fn ) == 0 )
    {
      int doneclient = 0, doneini = 0;
      projectmap_build( buffer, "" );
      for (i=0;i<((int)CONTEST_COUNT) && (!doneclient || !doneini);i++)
      {
        if (client->loadorder_map[i] == 1)
        {
          doneclient = 1;
          client->loadorder_map[i] |= 0x80;
        }
        if (buffer[i] == 1 /* DES' contest id */)
        {
          doneini = 1;
          buffer[i] |= 0x80; /* disable */
        }
      }
      modfail += (!WritePrivateProfileStringB( OPTSECT_MISC, "project-priority", projectmap_expand(buffer), fn ));
    }
  }

  if (!GetPrivateProfileStringB( OPTSECT_NET, "disabled", "", buffer, 1, fn ))
  {
    if (GetPrivateProfileIntB( OPTION_SECTION, "runoffline", 0, fn ))
    {
      client->offlinemode = 1;
      if (GetPrivateProfileIntB( OPTION_SECTION, "lurkonly", 0, fn ) ||
          GetPrivateProfileIntB( OPTION_SECTION, "lurk", 0, fn ) );
        client->offlinemode = 0;
      if (client->offlinemode)
        modfail += (!WritePrivateProfileStringB( OPTSECT_NET, "disabled", "yes", fn ));
    }
  }
  
  /* unconditional deletion of obsolete keys */
  for (i = 0; i < (int)(sizeof(obskeys) / sizeof(obskeys[0])); i++)
    modfail += (!WritePrivateProfileStringB( OPTION_SECTION, obskeys[i], NULL, fn ));
  
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
  char buffer[64];
  const char *sect = OPTION_SECTION;
  char *p; int i;

  client->randomchanged = 0;
  RefreshRandomPrefix( client );

  const char *fn = client->inifilename;
  fn = GetFullPathForFilename( fn );

  if ( access( fn, 0 ) != 0 ) 
    return +1; /* fall into config */
    
  __remapObsoleteParameters( client, fn ); /* load obsolete options */

  if (!GetPrivateProfileStringB( sect, "id", "", client->id, sizeof(client->id), fn ))
    strcpy( client->id, "rc5@distributed.net" );

  if (GetPrivateProfileStringB( sect, "threshold", "", buffer, sizeof(buffer), fn ))
  {
    p = strchr( buffer, ':' );
    client->inthreshold[0] = atoi(buffer);
    client->outthreshold[0] = ((p==NULL)?(client->inthreshold[0]):(atoi(p+1)));
    for (i = 1; i < CONTEST_COUNT; i++ )
    {
      client->inthreshold[i] = client->inthreshold[0];
      client->outthreshold[i] = client->outthreshold[0];
    }
  }
  if (GetPrivateProfileStringB( sect, "threshold2", "", buffer, sizeof(buffer), fn ))
  {
    p = strchr( buffer, ':' );
    client->inthreshold[1] = atoi(buffer);
    client->outthreshold[1] = ((p==NULL)?(client->inthreshold[1]):(atoi(p+1)));
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
  
  client->uuehttpmode = GetPrivateProfileIntB( sect, "uuehttpmode", client->uuehttpmode, fn );
  GetPrivateProfileStringB( sect, "httpproxy", client->httpproxy, client->httpproxy, sizeof(client->httpproxy), fn );  
  client->httpport = GetPrivateProfileIntB( sect, "httpport", client->httpport, fn );
  GetPrivateProfileStringB( sect, "httpid", client->httpid, client->httpid, sizeof(client->httpid), fn );
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
  client->cputype = GetPrivateProfileIntB( sect, "cputype", client->cputype, fn );
  client->numcpu = GetPrivateProfileIntB( sect, "numcpu", client->numcpu, fn );
  client->preferred_blocksize = GetPrivateProfileIntB( sect, "preferredblocksize", client->preferred_blocksize, fn );

  GetPrivateProfileStringB( OPTSECT_MISC, "project-priority", "", buffer, sizeof(buffer), fn );
  projectmap_build(client->loadorder_map, buffer);
    
  client->messagelen = GetPrivateProfileIntB( sect, "messagelen", client->messagelen, fn );
  client->smtpport = GetPrivateProfileIntB( sect, "smtpport", client->smtpport, fn );
  GetPrivateProfileStringB( sect, "smtpsrvr", client->smtpsrvr, client->smtpsrvr, sizeof(client->smtpsrvr), fn );
  GetPrivateProfileStringB( sect, "smtpfrom", client->smtpfrom, client->smtpfrom, sizeof(client->smtpfrom), fn );
  GetPrivateProfileStringB( sect, "smtpdest", client->smtpdest, client->smtpdest, sizeof(client->smtpdest), fn );

  client->blockcount = GetPrivateProfileIntB( sect, "count", client->blockcount, fn );
  if (GetPrivateProfileIntB( sect, "runbuffers", 0, fn ))
    client->blockcount = -1;
  
  client->offlinemode = GetPrivateProfileIntB( OPTSECT_NET, "disabled", client->offlinemode, fn );
  GetPrivateProfileStringB( OPTSECT_BUFFERS, "alternate-buffer-directory", client->remote_update_dir, client->remote_update_dir, sizeof(client->remote_update_dir), fn );

  client->percentprintingoff = GetPrivateProfileIntB( sect, "percentoff", client->percentprintingoff, fn );
  client->connectoften = GetPrivateProfileIntB( sect, "frequent", client->connectoften , fn );
  client->quietmode = GetPrivateProfileIntB( sect, "quiet", client->quietmode, fn );
  client->nofallback = GetPrivateProfileIntB( sect, "nofallback", client->nofallback, fn );
  client->noexitfilecheck = GetPrivateProfileIntB( sect, "noexitfilecheck", client->noexitfilecheck, fn );

  GetPrivateProfileStringB( sect, "logname", client->logname, client->logname, sizeof(client->logname), fn );
  GetPrivateProfileStringB( sect, "pausefile", client->pausefile, client->pausefile, sizeof(client->pausefile), fn );
  GetPrivateProfileStringB( sect, "checkpointfile", client->checkpoint_file, client->checkpoint_file, sizeof(client->checkpoint_file), fn );

  client->nodiskbuffers = GetPrivateProfileIntB( OPTSECT_BUFFERS, "buffer-only-in-memory", client->nodiskbuffers , fn );
  GetPrivateProfileStringB( OPTSECT_BUFFERS, "buffer-file-basename", client->in_buffer_basename, client->in_buffer_basename, sizeof(client->in_buffer_basename), fn );
  GetPrivateProfileStringB( OPTSECT_BUFFERS, "output-file-basename", client->in_buffer_basename, client->out_buffer_basename, sizeof(client->out_buffer_basename), fn );

  #if defined(LURK)
  i = dialup.GetCapabilityFlags();
  dialup.lurkmode = 0;
  if ((i & CONNECT_LURKONLY)!=0 && GetPrivateProfileIntB( sect, "lurkonly", 0, fn ))
    { dialup.lurkmode = CONNECT_LURKONLY; client->connectoften = 1; }
  else if ((i & CONNECT_LURK)!=0 && GetPrivateProfileIntB( sect, "lurk", 0, fn ))
    dialup.lurkmode = CONNECT_LURK;
  if (dialup.lurkmode)
    client->offlinemode = 0;
  
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
    dowrite = (GetPrivateProfileStringB( sect, key, "", buffer, 2, fn ) != 0);
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
  const char *sect = OPTION_SECTION;

  client->randomchanged = 1;
  RefreshRandomPrefix( client );

  const char *fn = client->inifilename;
  fn = GetFullPathForFilename( fn );
  if ( !writefull && access( fn, 0 )!=0 )
    writefull = 1;

  if (0 == (i = WritePrivateProfileStringB( sect, "id", 
    ((strcmp( client->id,"rc5@distributed.net")==0)?(""):(client->id)), fn )))
    return -1; //failed

  if (__remapObsoleteParameters( client, fn ) < 0) 
    return -1; //file is read-only
  
  if (writefull != 0)
  {
    /* --- CONF_MENU_BUFF -- */
    __XSetProfileInt( OPTSECT_BUFFERS, "buffer-only-in-memory", (client->nodiskbuffers)?(1):(0), fn, 0, 'y' );
    __XSetProfileStr( OPTSECT_BUFFERS, "buffer-file-basename", client->in_buffer_basename, fn, NULL );
    __XSetProfileStr( OPTSECT_BUFFERS, "output-file-basename", client->out_buffer_basename, fn, NULL );
    __XSetProfileStr( OPTSECT_BUFFERS, "alternate-buffer-directory", client->remote_update_dir, fn, NULL );

    strcpy(buffer,projectmap_expand(NULL));
    __XSetProfileStr( OPTSECT_MISC, "project-priority", projectmap_expand(client->loadorder_map), fn, buffer );

    __XSetProfileInt( sect, "frequent", client->connectoften, fn, 0, 1 );
    __XSetProfileInt( sect, "preferredblocksize", client->preferred_blocksize, fn, 31, 0 );
    
    sprintf(buffer,"%d:%d", (int)client->inthreshold[0], (int)client->outthreshold[0]);
    __XSetProfileStr( sect, "threshold", buffer, fn, "10:10" );
    if (client->inthreshold[1] == client->inthreshold[0] && client->outthreshold[1] == client->outthreshold[0])
      WritePrivateProfileStringB( sect, "threshold2", NULL, fn );
    else
    {
      sprintf(buffer,"%d:%d", (int)client->inthreshold[1], (int)client->outthreshold[1]);
      WritePrivateProfileStringB( sect, "threshold2", buffer, fn );
    }
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

    __XSetProfileInt( sect, "cputype", client->cputype, fn, -1, 0 );
    __XSetProfileInt( sect, "numcpu", client->numcpu, fn, -1, 0 );
    __XSetProfileInt( "processor-usage", "priority", client->priority, fn, 0, 0);

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
    __XSetProfileInt( sect, "uuehttpmode", client->uuehttpmode, fn, 0, 0);
    __XSetProfileInt( sect, "httpport", client->httpport, fn, 0, 0);
    __XSetProfileStr( sect, "httpproxy", client->httpproxy, fn, NULL);
    __XSetProfileStr( sect, "httpid", client->httpid, fn, NULL);

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

    __XSetProfileStr( sect, "logname", client->logname, fn, NULL );
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

      client->randomprefix =  GetPrivateProfileIntB(OPTSECT_RC5, "randomprefix", 
                                                 client->randomprefix, fn);
      client->scheduledupdatetime = GetPrivateProfileIntB(OPTSECT_NET, 
                      "scheduledupdatetime", client->scheduledupdatetime, fn);

      client->rc564closed = (GetPrivateProfileIntB(OPTSECT_RC5, "closed", 0, fn )!=0);
    }
    
    if (client->randomchanged)
    {
      client->randomchanged = 0;
      if (!WritePrivateProfileIntB(OPTSECT_RC5,"randomprefix",client->randomprefix,fn))
      {
        //client->stopiniio == 1;
        return; //write fail
      }
      WritePrivateProfileStringB(OPTSECT_RC5,"closed",(client->rc564closed)?("1"):(NULL), fn );
      if (client->scheduledupdatetime == 0)
        WritePrivateProfileStringB(OPTSECT_NET, "scheduledupdatetime", NULL, fn );
      else
        WritePrivateProfileIntB(OPTSECT_NET, "scheduledupdatetime", client->scheduledupdatetime, fn );
    }
  }
  return;
}

