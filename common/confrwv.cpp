// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: confrwv.cpp,v $
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
return "@(#)$Id: confrwv.cpp,v 1.34 1999/01/26 20:19:15 cyp Exp $"; }
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
#include "network.h"   // ntohl() / htonl()
#include "cmpidefs.h"  // strcmpi()
#include "confrwv.h"   // Outselves

// --------------------------------------------------------------------------

static const char *OPTION_SECTION="parameters"; //#define OPTION_SECTION "parameters"

//----------------------------------------------------------------------------

int ReadConfig(Client *client) //DO NOT PRINT TO SCREEN (or whatever) FROM HERE
{                              //DO NOT VALIDATE FROM HERE
  char buffer[64];
  const char *sect = OPTION_SECTION;
  char *p; int i;

  client->randomchanged = 0;
  RefreshRandomPrefix( client, 1 /* don't trigger */ );

  const char *fn = GetFullPathForFilename( client->inifilename );
  if ( access( fn, 0 )!=0 ) 
    return -1;

  if (!GetPrivateProfileStringA( sect, "id", "", client->id, sizeof(client->id), fn ))
    strcpy( client->id, "rc5@distributed.net" );

  if (GetPrivateProfileStringA( sect, "threshold", "", buffer, sizeof(buffer), fn ))
    {
    p = strchr( buffer, ':' );
    client->inthreshold[0]=atoi(buffer);
    client->outthreshold[0]=((p==NULL)?(client->inthreshold[0]):(atoi(p+1)));
    client->inthreshold[1]=client->inthreshold[0];
    client->outthreshold[1]=client->outthreshold[0];
    }
  if (GetPrivateProfileStringA( sect, "threshold2", "", buffer, sizeof(buffer), fn ))
    {
    p = strchr( buffer, ':' );
    client->inthreshold[1]=atoi(buffer);
    client->outthreshold[1]=((p==NULL)?(client->inthreshold[1]):(atoi(p+1)));
    }
  if (GetPrivateProfileStringA( sect, "hours", "", buffer, sizeof(buffer), fn ))
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
  
  client->uuehttpmode = GetPrivateProfileIntA( sect, "uuehttpmode", client->uuehttpmode, fn );
  GetPrivateProfileStringA( sect, "httpproxy", client->httpproxy, client->httpproxy, sizeof(client->httpproxy), fn );  
  client->httpport = GetPrivateProfileIntA( sect, "httpport", client->httpport, fn );
  GetPrivateProfileStringA( sect, "httpid", client->httpid, client->httpid, sizeof(client->httpid), fn );
  client->keyport = GetPrivateProfileIntA( sect, "keyport", client->keyport, fn );
  GetPrivateProfileStringA( sect, "keyproxy", client->keyproxy, client->keyproxy, sizeof(client->keyproxy), fn );
  client->nettimeout = GetPrivateProfileIntA( sect, "nettimeout", client->nettimeout, fn );
  
  client->autofindkeyserver = ((client->keyproxy[0]==0 || 
    strcmpi( client->keyproxy, "rc5proxy.distributed.net" )==0 ||
    strcmpi(client->keyproxy,"auto")==0 || strcmpi(client->keyproxy,"(auto)")==0) ||
    ( confopt_IsHostnameDNetHost(client->keyproxy) &&
    GetPrivateProfileIntA( "networking", "autofindkeyserver", 1, fn ) ));
  if (client->autofindkeyserver && client->keyport != 3064)  
    client->keyport = 0;
  
  i = GetPrivateProfileIntA( sect, "niceness", -12345, fn );
  if (i != -12345) client->priority = ((i==2)?(8):((i==1)?(4):(0)));
  client->priority = GetPrivateProfileIntA( "processor-usage", "priority", client->priority, fn );
  client->cputype = GetPrivateProfileIntA( sect, "cputype", client->cputype, fn );
  client->numcpu = GetPrivateProfileIntA( sect, "numcpu", client->numcpu, fn );
  client->preferred_blocksize = GetPrivateProfileIntA( sect, "preferredblocksize", client->preferred_blocksize, fn );
  client->preferred_contest_id = (GetPrivateProfileIntA( sect, "processdes", 1, fn )?(1):(0));

  client->messagelen = GetPrivateProfileIntA( sect, "messagelen", client->messagelen, fn );
  client->smtpport = GetPrivateProfileIntA( sect, "smtpport", client->smtpport, fn );
  GetPrivateProfileStringA( sect, "smtpsrvr", client->smtpsrvr, client->smtpsrvr, sizeof(client->smtpsrvr), fn );
  GetPrivateProfileStringA( sect, "smtpfrom", client->smtpfrom, client->smtpfrom, sizeof(client->smtpfrom), fn );
  GetPrivateProfileStringA( sect, "smtpdest", client->smtpdest, client->smtpdest, sizeof(client->smtpdest), fn );

  client->blockcount = GetPrivateProfileIntA( sect, "count", client->blockcount, fn );
  if (GetPrivateProfileIntA( sect, "runbuffers", 0, fn ))
    client->blockcount = -1;
  
  client->offlinemode = GetPrivateProfileIntA( sect, "runoffline", 0, fn );
  client->percentprintingoff = GetPrivateProfileIntA( sect, "percentoff", 0, fn );
  client->connectoften = GetPrivateProfileIntA( sect, "frequent", 0, fn );
  client->nodiskbuffers = GetPrivateProfileIntA( sect, "nodisk", 0, fn );
  client->quietmode = GetPrivateProfileIntA( sect, "quiet", 0, fn );
  client->quietmode |= GetPrivateProfileIntA( sect, "win95hidden", 0, fn );
  client->quietmode |= GetPrivateProfileIntA( sect, "os2hidden", 0, fn );
  client->quietmode |= GetPrivateProfileIntA( sect, "runhidden", 0, fn );
  client->nofallback = GetPrivateProfileIntA( sect, "nofallback", 0, fn );
  client->noexitfilecheck = GetPrivateProfileIntA( sect, "noexitfilecheck", 0, fn );

  #if defined(MMX_BITSLICER) || defined(MMX_RC5)
  client->usemmx = GetPrivateProfileIntA( sect, "usemmx", 1, fn );
  #endif

  GetPrivateProfileStringA( sect, "logname", client->logname, client->logname, sizeof(client->logname), fn );
  GetPrivateProfileStringA( sect, "pausefile", client->pausefile, client->pausefile, sizeof(client->pausefile), fn );
  GetPrivateProfileStringA( sect, "checkpointfile", client->checkpoint_file, client->checkpoint_file, sizeof(client->checkpoint_file), fn );
  GetPrivateProfileStringA( sect, "in", client->in_buffer_file[0], client->in_buffer_file[0], sizeof(client->in_buffer_file[0]), fn );
  GetPrivateProfileStringA( sect, "out", client->out_buffer_file[0], client->out_buffer_file[0], sizeof(client->out_buffer_file[0]), fn );
  GetPrivateProfileStringA( sect, "in2", client->in_buffer_file[1], client->in_buffer_file[1], sizeof(client->in_buffer_file[1]), fn );
  GetPrivateProfileStringA( sect, "out2", client->out_buffer_file[1], client->out_buffer_file[1], sizeof(client->out_buffer_file[1]), fn );

  #if defined(LURK)
  dialup.lurkmode = 0;
  if (GetPrivateProfileIntA( sect, "lurkonly", 0, fn )) 
    { dialup.lurkmode = 2; client->connectoften = 1; }
  else if (GetPrivateProfileIntA( sect, "lurk", 0, fn ))
    dialup.lurkmode = 1;
  #if (CLIENT_OS == OS_WIN32)
  dialup.dialwhenneeded = GetPrivateProfileIntA( sect, "dialwhenneeded", 0, fn );
  GetPrivateProfileStringA( sect, "connectionname", dialup.connectionname, dialup.connectionname, sizeof(dialup.connectionname), fn );
  #endif
  #endif /* LURK */

  return 0;
}

// --------------------------------------------------------------------------

//conditional ini write functions

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
    dowrite = (GetPrivateProfileStringA( sect, key, "", buffer, 2, fn )!=0);
  if (dowrite)
    WritePrivateProfileStringA( sect, key, newval, fn );
}

static void __XSetProfileInt( const char *sect, const char *key, 
          s32 newval, const char *fn, s32 defval, int asbool )
{ 
  char buffer[4];
  if (sect == NULL) 
    sect = OPTION_SECTION;
  if (asbool)
    {
    defval = ((defval)?(1):(0));
    newval = ((newval)?(1):(0));
    }
  int dowrite = (defval!=newval);
  if (!dowrite)
    dowrite = (GetPrivateProfileStringA( sect, key, "", buffer, 2, fn )!=0);
  if (dowrite)
    WritePrivateProfileIntA( sect, key, newval, fn );
}

// --------------------------------------------------------------------------

//Some OS's write run-time stuff to the .ini, so we protect
//the ini by only allowing that client's internal settings to change.

int WriteConfig(Client *client, int writefull /* defaults to 0*/)  
{
  char buffer[64];
  const char *sect = OPTION_SECTION;
  int i;

  client->randomchanged = 1;
  RefreshRandomPrefix( client );

  const char *fn = GetFullPathForFilename( client->inifilename );
  if ( !writefull && access( fn, 0 )!=0 )
    writefull = 1;

  if (0 == WritePrivateProfileStringA( sect, "id", 
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
      WritePrivateProfileStringA( sect, "threshold2", NULL, fn );
    else
      {
      sprintf(buffer,"%d:%d", (int)client->inthreshold[1], (int)client->outthreshold[1]);
      WritePrivateProfileStringA( sect, "threshold2", buffer, fn );
      }

    __XSetProfileInt( sect, "nodisk", (client->nodiskbuffers)?(1):(0), fn, 0, 1 );
    __XSetProfileStr( sect, "checkpointfile", client->checkpoint_file, fn, NULL );
    
    /* --- CONF_MENU_MISC __ */
    
    sprintf(buffer,"%u:%02u", (unsigned)(client->minutes/60), (unsigned)(client->minutes%60)); 
    __XSetProfileStr( sect, "hours", buffer, fn, "0:00" );
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
    __XSetProfileInt( sect, "nofallback", client->nofallback, fn, 1, 1);
    
    char *af=NULL, *host=client->keyproxy, *port = buffer;
    if (confopt_isstringblank(host) || client->autofindkeyserver)
      { //delete keys so that old inis stay compatible and default.
      host = NULL; if (client->keyport != 3064) port = NULL; }
    else if (confopt_IsHostnameDNetHost(host))
      { af = "0"; if (client->keyport != 3064) port = NULL; }
    if (port!=NULL) sprintf(port,"%ld",client->keyport);
    WritePrivateProfileStringA( "networking", "autofindkeyserver", af, fn );
    WritePrivateProfileStringA( sect, "keyport", port, fn );
    WritePrivateProfileStringA( sect, "keyproxy", host, fn );
    __XSetProfileInt( sect, "uuehttpmode", client->uuehttpmode, fn, 0, 0);
    __XSetProfileInt( sect, "httpport", client->httpport, fn, 0, 0);
    __XSetProfileStr( sect, "httpproxy", client->httpproxy, fn, NULL);
    __XSetProfileStr( sect, "httpid", client->httpid, fn, NULL);

    #if defined(LURK)
    WritePrivateProfileStringA( sect, "lurk", (dialup.lurkmode==1)?("1"):(NULL), fn );
    WritePrivateProfileStringA( sect, "lurkonly", (dialup.lurkmode==2)?("1"):(NULL), fn );
    WritePrivateProfileStringA( sect, "dialwhenneeded", (dialup.dialwhenneeded)?("1"):(NULL), fn );
    #if (CLIENT_OS==OS_WIN32)
    __XSetProfileStr( sect, "connectionname", dialup.connectionname, fn, NULL );
    #endif
    #endif // defined LURK

    /* --- CONF_MENU_LOG -- */

    __XSetProfileStr( sect, "logname", client->logname, fn, NULL );
    __XSetProfileInt( sect, "messagelen", client->messagelen, fn, 0, 0);
    __XSetProfileStr( sect, "smtpsrvr", client->smtpsrvr, fn, NULL);
    __XSetProfileStr( sect, "smtpfrom", client->smtpfrom, fn, NULL);
    __XSetProfileStr( sect, "smtpdest", client->smtpdest, fn, NULL);
    __XSetProfileInt( sect, "smtpport", client->smtpport, fn, 25, 0);

    /* no menu option */
  
    WritePrivateProfileStringA( sect, "processdes", ((client->preferred_contest_id==1)?(NULL):("1")), fn);
      
    } /* if (writefull != 0) */
  
  /* unconditional deletion of obsolete keys */
  const char *obskeys[]={"runhidden","os2hidden","win95hidden","checkpoint2",
                         "niceness","timeslice","runbuffers"};
  for (i=0;i<(int)(sizeof(obskeys)/sizeof(obskeys[0]));i++)
    WritePrivateProfileStringA( sect, obskeys[i], NULL, fn );    
  
  /* conditional deletion of obsolete keys */
  if (GetPrivateProfileStringA( sect, "usemmx", "", buffer, 2, fn))
    {
    if (((GetProcessorType(1) & 0x100) != 0) || 
       GetPrivateProfileIntA( sect, "usemmx", 0, fn ))
    WritePrivateProfileStringA( sect, "usemmx", NULL, fn );
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
         GetPrivateProfileIntA(sect, "randomprefix", client->randomprefix, fn);
      client->descontestclosed = ntohl(
         GetPrivateProfileIntA(sect, "descontestclosed", 
                                         htonl(client->descontestclosed), fn));
      client->scheduledupdatetime = ntohl(
         GetPrivateProfileIntA(sect, "scheduledupdatetime",
                                      htonl(client->scheduledupdatetime), fn));
      int statechange = 0;
      for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
        {
        if (cont_i==0) strcpy(key,"contestdone");
        else sprintf(key,"contestdone%u", cont_i+1 );
        int oldstate = ((client->contestdone[cont_i])?(1):(0));
        int newstate = GetPrivateProfileIntA(sect, key, oldstate, fn );
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

      if (!WritePrivateProfileIntA(sect,"randomprefix",client->randomprefix,fn))
        return; //write fail

      for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
        {
        if (cont_i==0) strcpy(key,"contestdone");
        else sprintf(key, "contestdone%u", cont_i+1 );
        if (client->contestdone[cont_i])
          WritePrivateProfileIntA( sect, key, 1, fn );
        else
          WritePrivateProfileStringA( sect, key, NULL, fn );
        }
      WritePrivateProfileStringA( sect, "contestdoneflags", NULL, fn );
      WritePrivateProfileIntA( sect, "descontestclosed", 
                                    htonl(client->descontestclosed), fn );
      WritePrivateProfileIntA( sect, "scheduledupdatetime", 
                                    htonl(client->scheduledupdatetime), fn );
      }
    }
  return;
}

// -----------------------------------------------------------------------
