// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: confrwv.cpp,v $
// Revision 1.43  1999/02/09 03:16:41  remi
// ReadConfig() now calls dialup.GetDefaultIFaceMask() to get
// the default value of connifacemask[].
// WriteConfig() will always write it in the .ini
//
// Revision 1.42  1999/02/08 23:19:39  remi
// The right default for interface-to-watch is "ppp0:sl0" not "\0"
// (at least on Linux).
// FreeBSD now supports lurk mode also.
//
// Revision 1.41  1999/02/07 16:00:08  cyp
// Lurk changes: genericified variable names, made less OS-centric.
//
// Revision 1.40  1999/02/06 10:42:55  remi
// - the default for dialup.ifacestowatch is now 'ppp0:sl0'.
// - #ifdef'ed dialup.ifacestowatch (only Linux at the moment)
// - modified a bit the help text in confopt.cpp
//
// Revision 1.39  1999/02/06 09:08:08  remi
// Enhanced the lurk fonctionnality on Linux. Now it use a list of interfaces
// to watch for online/offline status. If this list is empty (the default), any
// interface up and running (besides the lookback one) will trigger the online
// status.
// Fixed formating in lurk.cpp.
//
// Revision 1.38  1999/02/04 10:44:19  cyp
// Added support for script-driven dialup. (currently linux only)
//
// Revision 1.37  1999/01/29 19:06:37  jlawson
// fixed formatting.
//
// Revision 1.36  1999/01/27 16:40:34  cyp
// changed conditional write of 'hours'. Also 'keyproxy'=="auto" (which one
// previous config had accidentally written to the ini) wasn't being discarded.
//
// Revision 1.35  1999/01/27 00:58:33  jlawson
// changed ini functions to use B versions instead of A versions.
//
// Revision 1.34  1999/01/26 20:19:15  cyp
// adapted for new ini stuff.
//
// Revision 1.33  1999/01/21 21:49:02  cyp
// completed toss of ValidateConfig().
//
// Revision 1.32  1999/01/19 09:45:17  patrick
// LURK: changed to not copy connection name for OS2.
//
// Revision 1.31  1999/01/17 15:57:32  cyp
// priority is now properly written to file. ValidateConfig() has poofed -
// Very little was being /properly/ validated there. Individual subsystems
// are now (as always have been) responsible for their own variables.
//
// Revision 1.30  1999/01/15 05:18:15  cyp
// disable ini i/o once we know that ini writes fail.
//
// Revision 1.29  1999/01/11 07:01:24  dicamillo
// Fixed incorrect test in ValidateConfig for priority.  It can now exceed 0.
//
// Revision 1.28  1999/01/10 15:17:48  remi
// Added "network.h" to the list of includes (needed for htonl() and ntohl())
//
// Revision 1.27  1999/01/09 00:52:12  silby
// descontestclosed and scheduledupdate time back
// to network byte order.
//
// Revision 1.26  1999/01/08 20:27:44  silby
// Fixed scheduledupdatetime and descontestclosed not being
// read from the ini.
//
// Revision 1.25  1999/01/07 20:14:55  cyp
// fixed priority=. Readini quote handling _really_ needs rewriting.
//
// Revision 1.24  1999/01/06 07:28:45  dicamillo
// Add (apparently missing) code to ReadConfig to set cputype.
//
// Revision 1.23  1999/01/06 03:07:00  remi
// Last minute patch from cyp.
//
// Revision 1.22  1999/01/05 09:02:02  silby
// Fixed bug in writeconfig - processdes=0 was being set, but
// not deleted.
//
// Revision 1.21  1999/01/04 02:47:30  cyp
// Cleaned up menu options and handling.
//
// Revision 1.20  1999/01/03 02:49:53  cyp
// Removed x86-specific hack I introduced in 1.13. This is now covered in
// confmenu. Removed autofindkeyserver perversion introducted a couple of
// versions ago. Removed keyport validation (default to zero).
//
// Revision 1.19  1999/01/02 08:00:16  silby
// Default scheduledupdatetime is now jan 2nd 17:15:00.
//
// Revision 1.18  1999/01/02 01:43:26  silby
// processdes option read/written again.
//
// Revision 1.17  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.15  1998/12/28 03:32:47  silby
// WIN32GUI internalread/writeconfig procedures are back.
//
// Revision 1.14  1998/12/28 03:03:40  silby
// Fixed problem with filenames having whitespace stripped from them.
//
// Revision 1.13  1998/12/27 03:27:08  cyp
// long pending x86-specific hack against ConfigureGeneral() crashes: on
// iniread, convert cputype 6 (non-existant type "Pentium MMX") into type 0
// ("Pentium"). See comment in code for possible (discounted) solutions.
//
// Revision 1.12  1998/12/25 05:30:28  silby
// Temporary commenting out of InternalRead/Write/Validate
//
// Revision 1.11  1998/12/25 02:32:11  silby
// ini writing functions are now not part of client object.
// This allows the win32 (and other) guis to have
// configure modules that act on a dummy client object.
// (Client::Configure should be seperated as well.)
// Also fixed bug with spaces being taken out of pathnames.
//
// Revision 1.10  1998/12/23 03:24:56  silby
// Client once again listens to keyserver for next contest start time,
// tested, it correctly updates.  Restarting after des blocks have
// been recieved has not yet been implemented, I don't have a clean
// way to do it yet.  Writing of contest data to the .ini has been
// moved back to confrwv with its other ini friends.
//
// Revision 1.9  1998/12/23 00:41:45  silby
// descontestclosed and scheduledupdatetime now read from the .ini file.
//
// Revision 1.8  1998/12/21 19:06:08  cyp
// Removed 'unused'/'unimplemented' sil[l|b]yness added in recent version.
// See client.h for full comment.
//
// Revision 1.7  1998/12/21 01:21:39  remi
// Recommitted to get the right modification time.
//
// Revision 1.6  1998/12/21 14:23:57  remi
// Fixed the weirdness of proxy, keyport, uuehttpmode etc... handling :
// - if keyproxy ends in .distributed.net, keyport and uuehttpmode are
//   kept in sync in ::ValidateConfig()
// - if uuehttpmode == 2|3, keyport = 80 and can't be changed
//   (port 80 is hardwired in the http code somewhere in network.cpp)
// - do not delete uuehttpmode from the .ini file when it's > 1 (!!)
//   this bug makes client <= 422 difficult to use with http or socks...
// - write keyport in the .ini only if it differs from the default
//   (2064 | 80 | 23) for a given uuehttmode
// - fixed bugs in ::ConfigureGeneral() related to autofindkeyserver,
//   uuehttpmode, keyproxy, and keyport.
//
// Revision 1.5  1998/12/21 00:21:01  silby
// Universally scheduled update time is now retrieved from the proxy,
// and stored in the .ini file.  Not yet used, however.
//
// Revision 1.4  1998/12/20 23:00:35  silby
// Descontestclosed value is now stored and retrieved from the ini file,
// additional updated of the .ini file's contest info when fetches and
// flushes are performed are now done.  Code to throw away old des blocks
// has not yet been implemented.
//
// Revision 1.3  1998/11/26 22:24:51  cyp
// Fixed blockcount validation (<0 _is_ valid). (b) WriteConfig() sets ini
// entries only if they already exist or they are not equal to the default.
// (c) WriteFullConfig() is now WriteConfig(1) [default arg is 0] (d) Threw
// out CheckforcedKeyport()/CheckforcedKeyproxy() [were unused/did not work]
//
// Revision 1.2  1998/11/26 06:52:59  cyp
// Fixed a couple of type warnings and threw out WriteContestAndPrefixConfig()
//
// Revision 1.1  1998/11/22 15:16:15  cyp
// Split from cliconfig.cpp; Changed runoffline/runbuffers/blockcount handling
// (runbuffers is now synonymous with blockcount=-1; offlinemode is always
// 0/1); changed 'frequent' context to describe what it does better: check
// buffers frequently and not connect frequently. Removed outthreshold[0] as
// well as both DES thresholds from the menu. Removed 'processdes' from the menu.
// Fixed various bugs. Range validation is always based on the min/max values in
// the option table.
//
//

#if (!defined(lint) && defined(__showids__))
const char *confrwv_cpp(void) {
return "@(#)$Id: confrwv.cpp,v 1.43 1999/02/09 03:16:41 remi Exp $"; }
#endif

#include "cputypes.h"
#include "client.h"    // Client class
#include "baseincs.h"  // atoi() etc
#include "iniread.h"   // 
#include "scram.h"     // InitRandom2(id)
#include "pathwork.h"  // GetFullPathForFilename()
#include "lurk.h"      // lurk stuff
#include "cpucheck.h"  // GetProcessorType() for mmx stuff
#include "confopt.h"   // conf_option table
#include "triggers.h"  // RaiseRestartRequestTrigger()
#include "clicdata.h"  // CliClearContestData()
#include "cmpidefs.h"  // strcmpi()
#include "confrwv.h"   // Outselves

// --------------------------------------------------------------------------

static const char *OPTION_SECTION="parameters"; //#define OPTION_SECTION "parameters"

//----------------------------------------------------------------------------

int ReadConfig(Client *client) //DO NOT PRINT TO SCREEN (or whatever) FROM HERE
{                              //DO NOT VALIDATE FROM HERE
  char buffer[64];
  const char *sect = OPTION_SECTION;
  const char *netsect = "networking";
  char *p; int i;

  client->randomchanged = 0;
  RefreshRandomPrefix( client, 1 /* don't trigger */ );

  const char *fn = GetFullPathForFilename( client->inifilename );
  if ( access( fn, 0 )!=0 ) 
    return -1;

  if (!GetPrivateProfileStringB( sect, "id", "", client->id, sizeof(client->id), fn ))
    strcpy( client->id, "rc5@distributed.net" );

  if (GetPrivateProfileStringB( sect, "threshold", "", buffer, sizeof(buffer), fn ))
    {
    p = strchr( buffer, ':' );
    client->inthreshold[0] = atoi(buffer);
    client->outthreshold[0] = ((p==NULL)?(client->inthreshold[0]):(atoi(p+1)));
    client->inthreshold[1] = client->inthreshold[0];
    client->outthreshold[1] = client->outthreshold[0];
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
    GetPrivateProfileIntB( netsect, "autofindkeyserver", 1, fn ) ));
  if (client->autofindkeyserver && client->keyport != 3064)  
    client->keyport = 0;
  
  i = GetPrivateProfileIntB( sect, "niceness", -12345, fn );
  if (i>=0 && i<=2) client->priority = ((i==2)?(8):((i==1)?(4):(0)));
  client->priority = GetPrivateProfileIntB( "processor-usage", "priority", client->priority, fn );
  client->cputype = GetPrivateProfileIntB( sect, "cputype", client->cputype, fn );
  client->numcpu = GetPrivateProfileIntB( sect, "numcpu", client->numcpu, fn );
  client->preferred_blocksize = GetPrivateProfileIntB( sect, "preferredblocksize", client->preferred_blocksize, fn );
  client->preferred_contest_id = (GetPrivateProfileIntB( sect, "processdes", 1, fn )?(1):(0));

  client->messagelen = GetPrivateProfileIntB( sect, "messagelen", client->messagelen, fn );
  client->smtpport = GetPrivateProfileIntB( sect, "smtpport", client->smtpport, fn );
  GetPrivateProfileStringB( sect, "smtpsrvr", client->smtpsrvr, client->smtpsrvr, sizeof(client->smtpsrvr), fn );
  GetPrivateProfileStringB( sect, "smtpfrom", client->smtpfrom, client->smtpfrom, sizeof(client->smtpfrom), fn );
  GetPrivateProfileStringB( sect, "smtpdest", client->smtpdest, client->smtpdest, sizeof(client->smtpdest), fn );

  client->blockcount = GetPrivateProfileIntB( sect, "count", client->blockcount, fn );
  if (GetPrivateProfileIntB( sect, "runbuffers", 0, fn ))
    client->blockcount = -1;
  
  client->offlinemode = GetPrivateProfileIntB( sect, "runoffline", client->offlinemode, fn );
  client->percentprintingoff = GetPrivateProfileIntB( sect, "percentoff", client->percentprintingoff, fn );
  client->connectoften = GetPrivateProfileIntB( sect, "frequent", client->connectoften , fn );
  client->nodiskbuffers = GetPrivateProfileIntB( sect, "nodisk", client->nodiskbuffers , fn );
  client->quietmode = GetPrivateProfileIntB( sect, "quiet", client->quietmode, fn );
  client->quietmode |= GetPrivateProfileIntB( sect, "win95hidden", 0, fn );
  client->quietmode |= GetPrivateProfileIntB( sect, "os2hidden", 0, fn );
  client->quietmode |= GetPrivateProfileIntB( sect, "runhidden", 0, fn );
  client->nofallback = GetPrivateProfileIntB( sect, "nofallback", 0, fn );
  client->noexitfilecheck = GetPrivateProfileIntB( sect, "noexitfilecheck", client->noexitfilecheck, fn );

  #if defined(MMX_BITSLICER) || defined(MMX_RC5)
  client->usemmx = GetPrivateProfileIntB( sect, "usemmx", 1, fn );
  #endif

  GetPrivateProfileStringB( sect, "logname", client->logname, client->logname, sizeof(client->logname), fn );
  GetPrivateProfileStringB( sect, "pausefile", client->pausefile, client->pausefile, sizeof(client->pausefile), fn );
  GetPrivateProfileStringB( sect, "checkpointfile", client->checkpoint_file, client->checkpoint_file, sizeof(client->checkpoint_file), fn );
  GetPrivateProfileStringB( sect, "in", client->in_buffer_file[0], client->in_buffer_file[0], sizeof(client->in_buffer_file[0]), fn );
  GetPrivateProfileStringB( sect, "out", client->out_buffer_file[0], client->out_buffer_file[0], sizeof(client->out_buffer_file[0]), fn );
  GetPrivateProfileStringB( sect, "in2", client->in_buffer_file[1], client->in_buffer_file[1], sizeof(client->in_buffer_file[1]), fn );
  GetPrivateProfileStringB( sect, "out2", client->out_buffer_file[1], client->out_buffer_file[1], sizeof(client->out_buffer_file[1]), fn );

  #if defined(LURK)
  i = dialup.GetCapabilityFlags();
  dialup.lurkmode = 0;
  if ((i & CONNECT_LURKONLY)!=0 && GetPrivateProfileIntB( sect, "lurkonly", 0, fn ))
    { dialup.lurkmode = CONNECT_LURKONLY; client->connectoften = 1; }
  else if ((i & CONNECT_LURK)!=0 && GetPrivateProfileIntB( sect, "lurk", 0, fn ))
    dialup.lurkmode = CONNECT_LURK;
  if ((i & CONNECT_IFACEMASK)!=0)
    GetPrivateProfileStringB( netsect, "interfaces-to-watch", dialup.connifacemask,
                              dialup.connifacemask, sizeof(dialup.connifacemask), fn );
  if ((i & CONNECT_DOD)!=0)
    {
    dialup.dialwhenneeded = GetPrivateProfileIntB( netsect, "enable-start-stop", 0, fn );
    #if (CLIENT_OS == OS_WIN32) /* old format */
    dialup.dialwhenneeded |= GetPrivateProfileIntB( sect, "dialwhenneeded", 0, fn );
    #endif
    if ((i & CONNECT_DODBYSCRIPT)!=0)
      {
      GetPrivateProfileStringB( netsect, "dialup-start-cmd", dialup.connstartcmd, dialup.connstartcmd, sizeof(dialup.connstartcmd), fn );
      GetPrivateProfileStringB( netsect, "dialup-stop-cmd", dialup.connstopcmd, dialup.connstopcmd, sizeof(dialup.connstopcmd), fn );
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
  char buffer[(sizeof(long)+1)*3];
  if (sect == NULL) 
    sect = OPTION_SECTION;
  if (asonoff)
    {
    if (defval) defval = 1;
    if (newval) newval = 1;
    }
  int dowrite = (defval != newval);
  if (!dowrite)
    dowrite = (GetPrivateProfileStringB( sect, key, "", buffer, 2, fn ) != 0);
  if (dowrite)
    {
    sprintf(buffer,"%ld",(long)newval);
    WritePrivateProfileStringB( sect, key, buffer, fn );
    }
  return;
}

// --------------------------------------------------------------------------

//Some OS's write run-time stuff to the .ini, so we protect
//the ini by only allowing that client's internal settings to change.

int WriteConfig(Client *client, int writefull /* defaults to 0*/)  
{
  char buffer[64];
  const char *sect = OPTION_SECTION;
  const char *netsect = "networking";
  int i;

  client->randomchanged = 1;
  RefreshRandomPrefix( client );

  const char *fn = GetFullPathForFilename( client->inifilename );
  if ( !writefull && access( fn, 0 )!=0 )
    writefull = 1;

  if (0 == WritePrivateProfileStringB( sect, "id", 
    ((strcmp( client->id,"rc5@distributed.net")==0)?(""):(client->id)), fn ))
    return -1; //failed
  
  if (writefull != 0)
    {
    /* --- CONF_MENU_BUFF -- */

    __XSetProfileStr( sect, "in", client->in_buffer_file[0], fn, NULL );
    __XSetProfileStr( sect, "out", client->out_buffer_file[0], fn, NULL );
    __XSetProfileStr( sect, "in2", client->in_buffer_file[1], fn, NULL );
    __XSetProfileStr( sect, "out2", client->out_buffer_file[1], fn, NULL );

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

    __XSetProfileInt( sect, "nodisk", (client->nodiskbuffers)?(1):(0), fn, 0, 1 );
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

    __XSetProfileInt( sect, "runoffline", client->offlinemode, fn, 0, 1);
    __XSetProfileInt( sect, "nettimeout", client->nettimeout, fn, 60, 0);
    __XSetProfileInt( sect, "nofallback", client->nofallback, fn, 0, 1);
    
    char *af=NULL, *host=client->keyproxy, *port = buffer;
    if (confopt_isstringblank(host) || client->autofindkeyserver)
        { //delete keys so that old inis stay compatible and default.
      host = NULL; if (client->keyport != 3064) port = NULL; }
    else if (confopt_IsHostnameDNetHost(host))
        { af = "0"; if (client->keyport != 3064) port = NULL; }
    if (port!=NULL) sprintf(port,"%ld",client->keyport);
    WritePrivateProfileStringB( netsect, "autofindkeyserver", af, fn );
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
      __XSetProfileStr( netsect, "interfaces-to-watch", dialup.connifacemask, fn, NULL );
    if ((i & CONNECT_DOD) != 0)
      {
      #if (CLIENT_OS == OS_WIN32)
      __XSetProfileInt( sect, "dialwhenneeded", dialup.dialwhenneeded, fn, 0, 1 );
      __XSetProfileStr( sect, "connectionname", dialup.connprofile, fn, NULL );
      #else
      if (dialup.dialwhenneeded || GetPrivateProfileStringB( netsect, "enable-start-stop", "", buffer, 2, fn))
        WritePrivateProfileStringB( netsect, "enable-start-stop", (dialup.dialwhenneeded)?("yes"):("no"), fn );
      if ((i & CONNECT_DODBYPROFILE) != 0)
        __XSetProfileStr( netsect, "dialup-profile", dialup.connprofile, fn, NULL );
      #endif
      if ((i & CONNECT_DODBYSCRIPT) != 0)
        {
        __XSetProfileStr( netsect, "dialup-start-cmd", dialup.connstartcmd, fn, NULL );
        __XSetProfileStr( netsect, "dialup-stop-cmd", dialup.connstopcmd, fn, NULL );
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

    /* no menu option */
  
    WritePrivateProfileStringB( sect, "processdes", ((client->preferred_contest_id==1)?(NULL):("1")), fn);

    } /* if (writefull != 0) */
  
  /* unconditional deletion of obsolete keys */
  const char *obskeys[]={"runhidden","os2hidden","win95hidden","checkpoint2",
                         "niceness","timeslice","runbuffers"};
  for (i = 0; i < (int)(sizeof(obskeys) / sizeof(obskeys[0])); i++)
    WritePrivateProfileStringB( sect, obskeys[i], NULL, fn );    
  
  /* conditional deletion of obsolete keys */
  if (GetPrivateProfileStringB( sect, "usemmx", "", buffer, 2, fn))
    {
    if (((GetProcessorType(1) & 0x100) != 0) || 
       GetPrivateProfileIntB( sect, "usemmx", 0, fn ))
    WritePrivateProfileStringB( sect, "usemmx", NULL, fn );
    }
  
  return 0;
}

// --------------------------------------------------------------------------

// update contestdone and randomprefix .ini entries
void RefreshRandomPrefix( Client *client, int no_trigger )
{       
  // we need to use no_trigger when reading/writing full configs

  if (client->stopiniio == 0 && client->nodiskbuffers == 0)
    {
    const char *fn = GetFullPathForFilename( client->inifilename );
    const char *sect = OPTION_SECTION;
    unsigned int cont_i;
    char key[32];

    if ( client->randomchanged == 0 ) /* load */
      {
      if ( access( fn, 0 )!=0 )
        return;

      client->randomprefix = 
         GetPrivateProfileIntB(sect, "randomprefix", client->randomprefix, fn);
      client->descontestclosed = GetPrivateProfileIntB(sect, "descontestclosed", client->descontestclosed, fn);
      client->scheduledupdatetime = GetPrivateProfileIntB(sect, "scheduledupdatetime", client->scheduledupdatetime, fn);
      if (client->descontestclosed == 0x0DF0EFEBL) 
        {                      //fix the net'ified values used during des-III
        client->descontestclosed = 0xBEEFF00DL;
        client->scheduledupdatetime = 0;
        client->randomchanged = 1;                //force a rewrite
        }
      int statechange = 0;
      for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
        {
        if (cont_i==0) strcpy(key,"contestdone");
        else sprintf(key,"contestdone%u", cont_i+1 );
        int oldstate = ((client->contestdone[cont_i])?(1):(0));
        int newstate = GetPrivateProfileIntB(sect, key, oldstate, fn );
        newstate = ((newstate)?(1):(0));
        if (oldstate != newstate)
          {
          statechange = 1;
          client->contestdone[cont_i] = newstate;
          CliClearContestInfoSummaryData( cont_i );
          }
        }
      if (statechange && !no_trigger)
        RaiseRestartRequestTrigger();
      }
      
    if (client->randomchanged)
      {
      client->randomchanged = 0;

      if (!WritePrivateProfileIntB(sect,"randomprefix",client->randomprefix,fn))
        return; //write fail

      for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
        {
        if (cont_i==0) strcpy(key,"contestdone");
        else sprintf(key, "contestdone%u", cont_i+1 );
        if (client->contestdone[cont_i])
          WritePrivateProfileIntB( sect, key, 1, fn );
        else
          WritePrivateProfileStringB( sect, key, NULL, fn );
        }
      WritePrivateProfileStringB( sect, "contestdoneflags", NULL, fn );
      WritePrivateProfileIntB( sect, "descontestclosed", client->descontestclosed, fn );
      WritePrivateProfileIntB( sect, "scheduledupdatetime", client->scheduledupdatetime, fn );
      }
    }
  return;
}

// -----------------------------------------------------------------------

