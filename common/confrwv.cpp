// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: confrwv.cpp,v $
// Revision 1.20  1999/01/03 02:49:53  cyp
// Removed x86-specific hack I introduced in 1.13. This is now covered by
// confmenu. Removed autofindkeyserver perversion introducted a couple of
// versions ago. Removed keyport validation (default to zero).
// btw: ValidateConfig() should *never* impose a value or make assumptions
// particularly for data that will eventually be used by a non-client method.
// The subsystems (network or whatever) must be able to grok bad data - if
// they can't then they need to be considered "not-robust".
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
return "@(#)$Id: confrwv.cpp,v 1.20 1999/01/03 02:49:53 cyp Exp $"; }
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
#include "network.h"   // ntohl()
#include "triggers.h"  // RaiseRestartRequestTrigger()
#include "clicdata.h"  // CliClearContestData()
#include "confrwv.h"   // Outselves

// --------------------------------------------------------------------------

static const char *OPTION_SECTION="parameters"; //#define OPTION_SECTION "parameters"

#define INISETKEY(key, value) ini.setrecord(OPTION_SECTION, conf_options[key].name, IniString(value))
#define INIGETKEY(key) (ini.getkey(OPTION_SECTION, conf_options[key].name, conf_options[key].defaultsetting)[0])
#define INIFIND(key) ini.findfirst(OPTION_SECTION, conf_options[key].name)

//----------------------------------------------------------------------------

int ReadConfig(Client *client)  //DO NOT PRINT TO SCREEN (or whatever) FROM HERE
{
  IniSection ini;
  s32 inierror, tempconfig;
  char buffer[64];
  char *p;
  IniRecord *tempptr;

  client->randomchanged = 0; /* force a read */
  RefreshRandomPrefix( client, 1 /* don't trigger */ );

  inierror = ini.ReadIniFile( GetFullPathForFilename( client->inifilename ) );
  if (inierror) return -1;

  client->timeslice = 0x10000;

  if (INIFIND(CONF_ID) != NULL)
    INIGETKEY(CONF_ID).copyto(client->id, sizeof(client->id));

  if (INIFIND(CONF_THRESHOLDI) != NULL)
    {
    INIGETKEY(CONF_THRESHOLDI).copyto(buffer, sizeof(buffer));
    p = strchr( buffer, ':' );
    client->inthreshold[0]=((confopt_isstringblank(buffer))?(10):(atoi(buffer)));
    client->outthreshold[0]=((p==NULL)?(client->inthreshold[0]):(atoi(p+1)));
    }

  if (INIFIND(CONF_THRESHOLDI2) != NULL)
    {
    INIGETKEY(CONF_THRESHOLDI2).copyto(buffer, sizeof(buffer));
    p = strchr( buffer, ':' );
    client->inthreshold[1]=((confopt_isstringblank(buffer))?(client->inthreshold[0]):(atoi(buffer)));
    client->outthreshold[1]=((p==NULL)?(client->inthreshold[1]):(atoi(p+1)));
    }
  else
    {
    client->outthreshold[1]=client->outthreshold[0];
    client->inthreshold[1]=client->inthreshold[0];
    }

  if (INIFIND(CONF_HOURS) != NULL)
    {
    INIGETKEY(CONF_HOURS).copyto(buffer, sizeof(buffer));
    tempconfig = atoi(buffer);
    client->minutes = -1;
    if (tempconfig > 0)
      {
      if ((p = strchr( buffer, ':' )) == NULL)
        p = strchr( buffer, '.' );
      if (p != NULL && strlen( p )==3 && isdigit(p[1]) && isdigit(p[2]))
        client->minutes = (tempconfig *60) + atoi(p+1);
      else if (p != NULL && *p == '.')
        client->minutes = (s32)(atol( buffer )* 60.);
      }
    if (client->minutes < 0)
      client->minutes = 0;
    }

  #ifdef OLDNICENESS
  if (INIFIND(CONF_NICENESS) != NULL)
    client->niceness = INIGETKEY(CONF_NICENESS);
  #else  
  if ((tempptr = ini.findfirst( "processor usage", "priority"))!=NULL)
    client->priority = (ini.getkey("processor usage", "priority", "0")[0]);
  else
    {
    client->priority = (ini.getkey(OPTION_SECTION, "niceness", "0")[0]);
    client->priority = ((client->priority==2)?(8):((client->priority==1)?(4):(0)));
    }
  #endif    

  if (INIFIND(CONF_UUEHTTPMODE) != NULL)
    client->uuehttpmode = INIGETKEY(CONF_UUEHTTPMODE);
  if (INIFIND(CONF_HTTPPROXY) != NULL)
    INIGETKEY(CONF_HTTPPROXY).copyto(client->httpproxy, sizeof(client->httpproxy));
  if (INIFIND(CONF_HTTPID) != NULL)
    INIGETKEY(CONF_HTTPID).copyto(client->httpid, sizeof(client->httpid));
  if (INIFIND(CONF_HTTPPORT) != NULL)
    client->httpport = INIGETKEY(CONF_HTTPPORT);
  if (INIFIND(CONF_KEYPORT) != NULL)
    client->keyport = INIGETKEY(CONF_KEYPORT);

  if (INIFIND(CONF_KEYPROXY) == NULL)
    {
    client->autofindkeyserver = 1;
    client->keyproxy[0]=0;
    if (client->keyport!=3064) 
      client->keyport=0; //let the network layer validate
    }
  else
    {
    //do an autofind only if the host is a dnet host AND autofindkeyserver is on.
    client->autofindkeyserver = 0;
    INIGETKEY(CONF_KEYPROXY).copyto(client->keyproxy, sizeof(client->keyproxy));
    if (confopt_isstringblank(client->keyproxy) || 
        strcmpi( client->keyproxy, "(auto)")==0 || // these two accidentally
        strcmpi( client->keyproxy, "auto")==0 ||   // made it to the ini in one beta
        strcmpi( client->keyproxy, "rc5proxy.distributed.net" )==0) 
      {                                         
      client->keyproxy[0]=0;
      client->autofindkeyserver = 1; //let Network::Open get a better name.
      if (client->keyport!=3064) 
        client->keyport=0; //let the network layer validate
      }
    else if (confopt_IsHostnameDNetHost(client->keyproxy))
      {
      tempconfig=ini.getkey("networking", "autofindkeyserver", "1")[0];
      client->autofindkeyserver = (tempconfig)?(1):(0);
      if (client->autofindkeyserver)
        client->keyproxy[0]=0;
      if (client->keyport!=3064) 
        client->keyport=0; //let the network layer validate
      }
    }

  if (INIFIND(CONF_NUMCPU) != NULL)
    client->numcpu = INIGETKEY(CONF_NUMCPU);
  if (INIFIND(CONF_MESSAGELEN) != NULL)
    client->messagelen = INIGETKEY(CONF_MESSAGELEN);
  if (INIFIND(CONF_SMTPPORT) != NULL)
    client->smtpport = INIGETKEY(CONF_SMTPPORT);
  if (INIFIND(CONF_SMTPSRVR) != NULL)
    INIGETKEY(CONF_SMTPSRVR).copyto(client->smtpsrvr, sizeof(client->smtpsrvr));
  if (INIFIND(CONF_SMTPFROM) != NULL)
    INIGETKEY(CONF_SMTPFROM).copyto(client->smtpfrom, sizeof(client->smtpfrom));
  if (INIFIND(CONF_SMTPDEST) != NULL)
    INIGETKEY(CONF_SMTPDEST).copyto(client->smtpdest, sizeof(client->smtpdest));
  if (INIFIND(CONF_LOGNAME) != NULL)
    INIGETKEY(CONF_LOGNAME).copyto(client->logname, sizeof(client->logname));
  if (INIFIND(CONF_CHECKPOINT) != NULL)
    INIGETKEY(CONF_CHECKPOINT).copyto(client->checkpoint_file[0], sizeof(client->checkpoint_file[0]));
  if (INIFIND(CONF_PROCESSDES) != NULL)
    client->preferred_contest_id = INIGETKEY(CONF_PROCESSDES);
  if (INIFIND(CONF_PREFERREDBLOCKSIZE) != NULL)
    client->preferred_blocksize = INIGETKEY(CONF_PREFERREDBLOCKSIZE);
  //if (INIFIND(CONF_COUNT) != NULL)
    client->blockcount = INIGETKEY(CONF_COUNT);
  if ((tempconfig=ini.getkey(OPTION_SECTION, "runbuffers", "0")[0])!=0)
    client->blockcount = -1;
  if ((tempconfig=ini.getkey(OPTION_SECTION, "runoffline", "0")[0])!=0)
    client->offlinemode = (tempconfig != 0);
  if ((tempconfig=ini.getkey(OPTION_SECTION, "percentoff", "0")[0])!=0)
    client->percentprintingoff = (tempconfig != 0);
  if ((tempconfig=ini.getkey(OPTION_SECTION, "frequent", "0")[0])!=0)
    client->connectoften = (tempconfig != 0);
  if ((tempconfig=ini.getkey(OPTION_SECTION, "nodisk", "0")[0])!=0)
    client->nodiskbuffers = (tempconfig != 0);
  if ((tempconfig=ini.getkey(OPTION_SECTION, "quiet", "0")[0])!=0)
    client->quietmode = (tempconfig != 0);
  if ((tempconfig=ini.getkey(OPTION_SECTION, "win95hidden", "0")[0])!=0)
    client->quietmode |= ((tempconfig != 0)?(1):(0));
  if ((tempconfig=ini.getkey(OPTION_SECTION, "os2hidden", "0")[0])!=0)
    client->quietmode |= ((tempconfig != 0)?(1):(0));
  if ((tempconfig=ini.getkey(OPTION_SECTION, "runhidden", "0")[0])!=0)
    client->quietmode |= ((tempconfig != 0)?(1):(0));
  if ((tempconfig=ini.getkey(OPTION_SECTION, "nofallback", "0")[0])!=0)
    client->nofallback= (tempconfig != 0);
  if ((tempconfig=ini.getkey(OPTION_SECTION, "nettimeout", "0")[0])!=0)
    client->nettimeout=tempconfig; 
  if ((tempconfig=ini.getkey(OPTION_SECTION, "noexitfilecheck", "0")[0])!=0)
    client->noexitfilecheck = (tempconfig != 0);
  
  ini.getkey(OPTION_SECTION,"in",client->in_buffer_file[0])[0].copyto(client->in_buffer_file[0],sizeof(client->in_buffer_file[0]));
  ini.getkey(OPTION_SECTION,"out",client->out_buffer_file[0])[0].copyto(client->out_buffer_file[0],sizeof(client->out_buffer_file[0]));
  ini.getkey(OPTION_SECTION,"in2",client->in_buffer_file[1])[0].copyto(client->in_buffer_file[1],sizeof(client->in_buffer_file[1]));
  ini.getkey(OPTION_SECTION,"out2",client->out_buffer_file[1])[0].copyto(client->out_buffer_file[1],sizeof(client->out_buffer_file[1]));
  ini.getkey(OPTION_SECTION,"pausefile",client->pausefile)[0].copyto(client->pausefile,sizeof(client->pausefile));

  #if defined(LURK)
  tempconfig=ini.getkey(OPTION_SECTION, "lurk", "0")[0];
  dialup.lurkmode = (tempconfig != 0);
  tempconfig=ini.getkey(OPTION_SECTION, "lurkonly", "0")[0];
  if (tempconfig) {dialup.lurkmode=2; client->connectoften=0;}
  tempconfig=ini.getkey(OPTION_SECTION, "dialwhenneeded", "0")[0];
  if (tempconfig) dialup.dialwhenneeded=1;
  INIGETKEY(CONF_CONNECTNAME).copyto(dialup.connectionname,sizeof(dialup.connectionname));
  #endif

  #if defined(MMX_BITSLICER) || defined(MMX_RC5)
    client->usemmx=ini.getkey(OPTION_SECTION, "usemmx", "1")[0];
  #endif

  #if defined(WIN32GUI)
  InternalReadConfig(ini);
  #endif

  ValidateConfig(client);

  return( inierror ? -1 : 0 );
}

// --------------------------------------------------------------------------

void ValidateConfig(Client *client) 
{
  //---------------------------
  //ValidateConfig() should not do more than basic whitespace 
  //stripping and range checking. It MAY not make assumptions.
  //DO NOT PRINT TO SCREEN HERE!
  //---------------------------

  unsigned int cont_i;
  unsigned int maxthresh=(unsigned int)conf_options[CONF_THRESHOLDI].choicemax;
  if (maxthresh > MAXBLOCKSPERBUFFER)
    maxthresh = MAXBLOCKSPERBUFFER;
  
  for (cont_i=0;cont_i<2;cont_i++)
    {
    if ( client->inthreshold[cont_i] < conf_options[CONF_THRESHOLDI].choicemin ) 
      client->inthreshold[cont_i] = conf_options[CONF_THRESHOLDI].choicemin;
    if ( (unsigned int)client->inthreshold[cont_i] > maxthresh ) 
      client->inthreshold[cont_i] = maxthresh;
    if ( client->outthreshold[cont_i] < conf_options[CONF_THRESHOLDI].choicemin ) 
      client->outthreshold[cont_i] = conf_options[CONF_THRESHOLDI].choicemin;
    if ( (unsigned int)client->outthreshold[cont_i] > maxthresh ) 
      client->outthreshold[cont_i] = maxthresh;
    if ( client->outthreshold[cont_i] > client->inthreshold[cont_i] ) 
      client->outthreshold[cont_i]=client->inthreshold[cont_i];

    if (client->in_buffer_file[cont_i][0] == 0)
      strcpy(client->in_buffer_file[cont_i], 
        conf_options[((cont_i==0)?(CONF_RC5IN):(CONF_DESIN))].defaultsetting );

    if (confopt_isstringblank(client->out_buffer_file[cont_i]))
    strcpy(client->out_buffer_file[cont_i], 
     conf_options[((cont_i==0)?(CONF_RC5OUT):(CONF_DESOUT))].defaultsetting );
    }

  if (client->blockcount < 0)
    client->blockcount = -1;

  client->timeslice = 0x10000;
  
  #ifdef OLDNICENESS
  if ( niceness < conf_options[CONF_NICENESS].choicemin || 
       niceness > conf_options[CONF_NICENESS].choicemax )
    niceness = conf_options[CONF_NICENESS].choicemin;
  #else
  if ( client->priority < conf_options[CONF_NICENESS].choicemin || 
       client->priority > conf_options[CONF_NICENESS].choicemax )
    client->priority = conf_options[CONF_NICENESS].choicemin;
  #endif

  if ( client->uuehttpmode < conf_options[CONF_UUEHTTPMODE].choicemin || 
       client->uuehttpmode > conf_options[CONF_UUEHTTPMODE].choicemax ) 
    client->uuehttpmode = 0;
  if ( (u32)client->randomprefix < (u32)conf_options[CONF_RANDOMPREFIX].choicemin || 
       (u32)client->randomprefix > (u32)conf_options[CONF_RANDOMPREFIX].choicemax) 
    client->randomprefix=100;
  if (client->smtpport < 0 || client->smtpport > 65535L) 
    client->smtpport=25;
  if (client->messagelen !=0 && client->messagelen < conf_options[CONF_MESSAGELEN].choicemin)
    client->messagelen = conf_options[CONF_MESSAGELEN].choicemin;
  if (( client->preferred_contest_id < 0 ) || ( client->preferred_contest_id > 1 )) 
    client->preferred_contest_id = 1;
  if (client->preferred_blocksize < conf_options[CONF_PREFERREDBLOCKSIZE].choicemin) 
    client->preferred_blocksize = conf_options[CONF_PREFERREDBLOCKSIZE].choicemin;
  if (client->preferred_blocksize > conf_options[CONF_PREFERREDBLOCKSIZE].choicemax) 
    client->preferred_blocksize = conf_options[CONF_PREFERREDBLOCKSIZE].choicemax;
  if ( client->minutes < 0 ) 
    client->minutes=0;
  if (client->nettimeout < conf_options[CONF_NETTIMEOUT].choicemin) 
    client->nettimeout=conf_options[CONF_NETTIMEOUT].choicemin;
  else if (client->nettimeout > conf_options[CONF_NETTIMEOUT].choicemax) 
    client->nettimeout=conf_options[CONF_NETTIMEOUT].choicemax;

  if (client->logname[0]==0 || strcmp(client->logname,"none")==0)
    client->logname[0]=0;
  if (confopt_isstringblank(client->pausefile) || strcmp(client->pausefile,"none")==0)
    client->pausefile[0]=0;
  if (confopt_isstringblank(client->checkpoint_file[0]) || strcmp(client->checkpoint_file[0],"none")==0)
    client->checkpoint_file[0][0]=0;

  confopt_killwhitespace(client->keyproxy);
  confopt_killwhitespace(client->httpproxy);
  confopt_killwhitespace(client->smtpsrvr);
  confopt_killwhitespace(client->id);

  if (client->id[0]==0)
    strcpy(client->id,"rc5@distributed.net");

  //validate numcpu is now in SelectCore(); //1998/06/21 cyrus

  InitRandom2( client->id );
}

// --------------------------------------------------------------------------

//Some OS's write run-time stuff to the .ini, so we protect
//the ini by only allowing that client's internal settings to change.

int WriteConfig(Client *client, int writefull /* defaults to 0*/)  
{
  IniSection ini;
  IniRecord *tempptr;

  client->randomchanged = 1; /* force a write */
  RefreshRandomPrefix( client );

  if ( ini.ReadIniFile( GetFullPathForFilename( client->inifilename ) ) )
    writefull = 1;
    
  if (writefull != 0)
    {
    char buffer[64];

    INISETKEY( CONF_ID, client->id );
    INISETKEY( CONF_RC5IN, client->in_buffer_file[0]);
    INISETKEY( CONF_RC5OUT, client->out_buffer_file[0]);
    INISETKEY( CONF_DESIN, client->in_buffer_file[1]);
    INISETKEY( CONF_DESOUT, client->out_buffer_file[1]);
  
    int default_threshold = atoi(conf_options[CONF_THRESHOLDI].defaultsetting);
    if (INIFIND(CONF_THRESHOLDI)!=NULL || 
       client->inthreshold[0]!=default_threshold || client->outthreshold[0]!=default_threshold)
      {
      sprintf(buffer,"%d:%d",(int)client->inthreshold[0],(int)client->outthreshold[0]);
      INISETKEY( CONF_THRESHOLDI, buffer );
      }
    if (client->inthreshold[1] == client->inthreshold[0] && client->outthreshold[1] == client->outthreshold[0])
      {
      tempptr = INIFIND(CONF_THRESHOLDI2);
      if (tempptr) tempptr->values.Erase();
      }
    else if (INIFIND(CONF_THRESHOLDI2)!=NULL || 
      client->inthreshold[1]!=default_threshold || client->outthreshold[1]!=default_threshold )
      {
      sprintf(buffer,"%d:%d",(int)client->inthreshold[1],(int)client->outthreshold[1]);
      INISETKEY( CONF_THRESHOLDI2, buffer );
      }
    if (client->minutes!=0 || INIFIND(CONF_HOURS)!=NULL)
      {
      sprintf(buffer,"%u:%02u", (unsigned)(client->minutes/60), (unsigned)(client->minutes%60)); 
      INISETKEY( CONF_HOURS, buffer );
      }
    
    if (client->cputype!=-1 || INIFIND(CONF_CPUTYPE)!=NULL)
      INISETKEY( CONF_CPUTYPE, client->cputype );
    if (client->numcpu!=-1 || INIFIND(CONF_NUMCPU)!=NULL)
      INISETKEY( CONF_NUMCPU, client->numcpu );
    if (INIFIND(CONF_PREFERREDBLOCKSIZE)!=NULL || client->preferred_blocksize!=
      (atoi(conf_options[CONF_PREFERREDBLOCKSIZE].defaultsetting)))
     INISETKEY( CONF_PREFERREDBLOCKSIZE, client->preferred_blocksize );
    if (client->preferred_contest_id!=0 || INIFIND(CONF_PROCESSDES)!=NULL)
      INISETKEY( CONF_PROCESSDES, client->preferred_contest_id);
    if (client->noexitfilecheck!=0 || INIFIND(CONF_NOEXITFILECHECK)!=NULL)
      INISETKEY( CONF_NOEXITFILECHECK, client->noexitfilecheck );
    #ifdef OLDNICENESS
    if (niceness != 0 || ini.findfirst( OPTION_SECTION, "niceness" )!=NULL )
      INISETKEY( CONF_NICENESS, niceness );
    #else
    if (client->priority != 0 || ini.findfirst( "processor usage", "priority")!=NULL )
      ini.setrecord("processor usage", "priority", IniString(client->priority));
    else if ((tempptr = ini.findfirst( OPTION_SECTION, "niceness"))!=NULL)
      tempptr->values.Erase();
    #endif
    if (client->percentprintingoff!=0 || INIFIND(CONF_PERCENTOFF)!=NULL)
      INISETKEY( CONF_PERCENTOFF, client->percentprintingoff );
    if (client->connectoften!=0 || INIFIND(CONF_FREQUENT)!=NULL)
      INISETKEY( CONF_FREQUENT, client->connectoften );
    if (client->nodiskbuffers!=0 || INIFIND(CONF_NODISK)!=NULL)
      INISETKEY( CONF_NODISK, IniString((client->nodiskbuffers)?("1"):("0")) );
    if (client->nofallback!=0 || INIFIND(CONF_NOFALLBACK)!=NULL)
      INISETKEY( CONF_NOFALLBACK, client->nofallback );
    if (client->nettimeout!=atoi(conf_options[CONF_NETTIMEOUT].defaultsetting) || 
      INIFIND(CONF_NETTIMEOUT)!=NULL)
      INISETKEY( CONF_NETTIMEOUT, client->nettimeout );
    if (client->logname[0]!=0 || INIFIND(CONF_LOGNAME)!=NULL)
      INISETKEY( CONF_LOGNAME, client->logname );
    if (client->checkpoint_file[0][0]!=0 || INIFIND(CONF_CHECKPOINT)!=NULL)
      INISETKEY( CONF_CHECKPOINT, client->checkpoint_file[0] );
    if ((tempptr=ini.findfirst(OPTION_SECTION,"checkpoint2"))!=NULL)/*obsolete*/
      tempptr->values.Erase();
    if ((tempptr=ini.findfirst(OPTION_SECTION,"timeslice"))!=NULL)/*obsolete*/
      tempptr->values.Erase();
    if (client->pausefile[0]!=0 || INIFIND(CONF_PAUSEFILE)!=NULL)
      INISETKEY( CONF_PAUSEFILE, client->pausefile );
    if (client->quietmode!=0 || INIFIND(CONF_QUIETMODE)!=NULL)
      INISETKEY( CONF_QUIETMODE, ((client->quietmode)?("1"):("0")) );
    if (client->offlinemode != 0 && client->offlinemode != 1) /* old runbuffers */
      { client->blockcount = -1;  client->offlinemode = 0; }
    if (client->offlinemode != 0 || ini.findfirst(OPTION_SECTION, "runoffline")!=NULL)
      ini.setrecord(OPTION_SECTION, "runoffline", IniString((client->offlinemode)?("1"):("0")));
    if (client->blockcount != 0 || INIFIND(CONF_COUNT)!=NULL)
      INISETKEY( CONF_COUNT, client->blockcount );
    if (client->uuehttpmode != 0 || INIFIND(CONF_UUEHTTPMODE)!=NULL)
      INISETKEY( CONF_UUEHTTPMODE, client->uuehttpmode );
    if (client->httpproxy[0]!= 0 || INIFIND(CONF_HTTPPROXY)!=NULL)
      INISETKEY( CONF_HTTPPROXY, client->httpproxy );
    if (client->httpport!= 0 || INIFIND(CONF_HTTPPORT)!=NULL)
      INISETKEY( CONF_HTTPPORT, client->httpport );
    if (client->httpid[0]!= 0 || INIFIND(CONF_HTTPID)!=NULL)
      INISETKEY( CONF_HTTPID, client->httpid );
    if (client->keyport!=0 || INIFIND(CONF_KEYPORT)!=NULL)
      INISETKEY( CONF_KEYPORT, client->keyport );
    if (client->messagelen!=0 || INIFIND( CONF_MESSAGELEN )!=NULL)
      INISETKEY( CONF_MESSAGELEN, client->messagelen );
    if (client->smtpsrvr[0]!=0 || INIFIND( CONF_SMTPSRVR )!=NULL)
      INISETKEY( CONF_SMTPSRVR, client->smtpsrvr );
    if (client->smtpport!=25 || INIFIND( CONF_SMTPPORT )!=NULL)
      INISETKEY( CONF_SMTPPORT, client->smtpport );
    if (client->smtpfrom[0]!=0 || INIFIND( CONF_SMTPFROM )!=NULL)
      INISETKEY( CONF_SMTPFROM, client->smtpfrom );
    if (client->smtpdest[0]!=0 || INIFIND( CONF_SMTPDEST )!=NULL)
      INISETKEY( CONF_SMTPDEST, client->smtpdest );

    #if defined(LURK)
    if (dialup.lurkmode==1)
      ini.setrecord(OPTION_SECTION, "lurk",  IniString("1"));
    else if ((tempptr = ini.findfirst(OPTION_SECTION, "lurk"))!=NULL)
      tempptr->values.Erase();
    if (dialup.lurkmode==2)
      ini.setrecord(OPTION_SECTION, "lurkonly",  IniString("1"));    
    else if ((tempptr = ini.findfirst(OPTION_SECTION, "lurkonly"))!=NULL)
      tempptr->values.Erase();
    if (dialup.dialwhenneeded)
      INISETKEY( CONF_DIALWHENNEEDED, dialup.dialwhenneeded);
    else if ((tempptr = ini.findfirst(OPTION_SECTION, "dialwhenneeded"))!=NULL)
      tempptr->values.Erase();
    if (strcmp(dialup.connectionname,conf_options[CONF_CONNECTNAME].defaultsetting)!=0)
      INISETKEY( CONF_CONNECTNAME, dialup.connectionname);
    else if ((tempptr = ini.findfirst(OPTION_SECTION, "connectionname"))!=NULL)
      tempptr->values.Erase();
    #endif
  
    //autofindkeyserver is also a PRESENCE flag. 
    //ie, it also means something if its NOT there.
    if (confopt_isstringblank(client->keyproxy) || 
      (client->autofindkeyserver && 
      confopt_IsHostnameDNetHost(client->keyproxy)))
      {
      //autokeyserver is enabled (because its on AND its a dnet host), so delete 
      //the old ini keys so that old inis stay compatible. We could at this 
      //point set keyproxy=rc5proxy.distributed.net, but why clutter up the ini?
      if ((tempptr = ini.findfirst(OPTION_SECTION, "keyproxy"))!=NULL) 
        tempptr->values.Erase();
      if ((tempptr = ini.findfirst( "networking", "autofindkeyserver"))!=NULL)
        tempptr->values.Erase();
      }
    else 
      {
      if (confopt_IsHostnameDNetHost(client->keyproxy))
        ini.setrecord("networking", "autofindkeyserver", IniString("0"));
      else if ((tempptr = ini.findfirst( "networking", "autofindkeyserver"))!=NULL)
        tempptr->values.Erase();
      INISETKEY( CONF_KEYPROXY, client->keyproxy );
      }

    } /* if (writefull != 0) */
  
  #if defined(WIN32GUI)
  InternalWriteConfig(ini);
  #endif
  
  //obsolete options
  if ((tempptr = ini.findfirst(OPTION_SECTION, "runhidden"))!=NULL)
    tempptr->values.Erase();    
  if ((tempptr = ini.findfirst(OPTION_SECTION, "os2hidden"))!=NULL)
    tempptr->values.Erase();    
  if ((tempptr = ini.findfirst(OPTION_SECTION, "win95hidden"))!=NULL)
    tempptr->values.Erase();    
  if ((tempptr = ini.findfirst(OPTION_SECTION, "usemmx"))!=NULL)
    {
    s32 tmps32 = ini.getkey(OPTION_SECTION, "usemmx", "0")[0];
    if ( tmps32!= 0 || (GetProcessorType(1) & 0x100) != 0)
      tempptr->values.Erase();
    }
  if ((tempptr = ini.findfirst(OPTION_SECTION, "timeslice"))!=NULL)
    tempptr->values.Erase();
  if ((tempptr = ini.findfirst(OPTION_SECTION, "runbuffers"))!=NULL)
    tempptr->values.Erase();   /* obsolete - uses blockcount==-1 */
    

  return( ini.WriteIniFile( GetFullPathForFilename( client->inifilename ) ) ? -1 : 0 );
}

// --------------------------------------------------------------------------

void RefreshRandomPrefix( Client *client, int no_trigger )
{       
  // we need to use no_trigger when reading/writing full configs

  if (client->stopiniio == 0 && client->nodiskbuffers == 0)
    {
    const char *OPTION_SECTION = "parameters";
    IniSection ini;
    unsigned int cont_i;
    s32 randomprefix, flagbits;
    int inierror = (ini.ReadIniFile( 
                       GetFullPathForFilename( client->inifilename ) ) != 0);
    int inichanged = 0;

    if (client->randomchanged)
      {
      randomprefix = (s32)(client->randomprefix);
      ini.setrecord(OPTION_SECTION, "randomprefix", IniString(randomprefix));

      flagbits = 0;
      for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
        {
        flagbits |= ((client->contestdone[cont_i])?(1<<cont_i):(0));

        char buffer[32];
        if (cont_i==0) strcpy(buffer,"contestdone");
        else sprintf(buffer,"contestdone%u", cont_i+1 );
        if (client->contestdone[cont_i])
          {
//LogScreen("write: client->contestdone[%u] ==> %u\n", cont_i, client->contestdone[cont_i]);
          ini.setrecord(OPTION_SECTION, buffer, 
              IniString((client->contestdone[cont_i])?("1"):("0")));
//LogScreen("write: end\n");
          }
        else
          {
//LogScreen("erase: client->contestdone[%u] ==> %u\n", cont_i, client->contestdone[cont_i]);
          IniRecord *inirec;
          if ((inirec=ini.findfirst(OPTION_SECTION, buffer))!=NULL)
            inirec->values.Erase();
//LogScreen("erase: end\n");
          }
        }
      ini.setrecord(OPTION_SECTION, "contestdoneflags", IniString(flagbits));
      ini.setrecord(OPTION_SECTION, "descontestclosed",
                    IniString((s32)htonl(client->descontestclosed)));
      ini.setrecord(OPTION_SECTION, "scheduledupdatetime",
                    IniString((s32)htonl(client->scheduledupdatetime)));
      client->randomchanged = 0;
      inichanged = 1;
      }
    else if (!inierror)
      {  
      randomprefix = ini.getkey(OPTION_SECTION, "randomprefix", "0")[0];
      if (randomprefix) client->randomprefix = randomprefix;

      u32 oldflags=0, newflags=0;

      IniRecord *inirec;
      if ((inirec=ini.findfirst(OPTION_SECTION, "contestdoneflags"))!=NULL)
        newflags = ini.getkey(OPTION_SECTION, "contestdoneflags", "0")[0];
      else
        {
        for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
          {
          char buffer[32];
          if (cont_i==0) strcpy(buffer,"contestdone");
          else sprintf(buffer,"contestdone%u", cont_i+1 );
          flagbits = ini.getkey(OPTION_SECTION, buffer, "0")[0];
          newflags |= ((flagbits)?(1<<cont_i):(0)); 
          }
        }
      oldflags = 0;
      for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
        {
        oldflags |= ((client->contestdone[cont_i])?(1<<cont_i):(0)); 
        client->contestdone[cont_i]=(((newflags&(1<<cont_i))==0)?(0):(1));
//LogScreen("read: client->contestdone[%u] ==> %u\n", cont_i, client->contestdone[cont_i]);
        }
      if (newflags != oldflags)
        {
        for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
          {
          if ((newflags & (1<<cont_i)) != (oldflags & (1<<cont_i)))
            CliClearContestInfoSummaryData( cont_i );
          }
        if (!no_trigger)
          RaiseRestartRequestTrigger();
        }
      }   
    
    if (inichanged)
      ini.WriteIniFile( GetFullPathForFilename( client->inifilename ) );
    }
  return;
}

// -----------------------------------------------------------------------

