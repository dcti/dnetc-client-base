// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: confrwv.cpp,v $
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
return "@(#)$Id: confrwv.cpp,v 1.2 1998/11/26 06:52:59 cyp Exp $"; }
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

// --------------------------------------------------------------------------

static const char *OPTION_SECTION="parameters"; //#define OPTION_SECTION "parameters"

#define INISETKEY(key, value) ini.setrecord(OPTION_SECTION, conf_options[key].name, IniString(value))
#define INIGETKEY(key) (ini.getkey(OPTION_SECTION, conf_options[key].name, conf_options[key].defaultsetting)[0])
#define INIFIND(key) ini.findfirst(OPTION_SECTION, conf_options[key].name)

//----------------------------------------------------------------------------

int Client::ReadConfig(void)  //DO NOT PRINT TO SCREEN (or whatever) FROM HERE
{
  IniSection ini;
  s32 inierror, tempconfig;
  char buffer[64];
  char *p;
  IniRecord *tempptr;

  inierror = ini.ReadIniFile( GetFullPathForFilename( inifilename ) );
  if (inierror) return -1;

  if (INIFIND(CONF_ID) != NULL)
    INIGETKEY(CONF_ID).copyto(id, sizeof(id));

  if (INIFIND(CONF_THRESHOLDI) != NULL)
    {
    INIGETKEY(CONF_THRESHOLDI).copyto(buffer, sizeof(buffer));
    p = strchr( buffer, ':' );
    inthreshold[0]=((confopt_isstringblank(buffer))?(10):(atoi(buffer)));
    outthreshold[0]=((p==NULL)?(inthreshold[0]):(atoi(p+1)));
    }

  if (INIFIND(CONF_THRESHOLDI2) != NULL)
    {
    INIGETKEY(CONF_THRESHOLDI2).copyto(buffer, sizeof(buffer));
    p = strchr( buffer, ':' );
    inthreshold[1]=((confopt_isstringblank(buffer))?(inthreshold[0]):(atoi(buffer)));
    outthreshold[1]=((p==NULL)?(inthreshold[1]):(atoi(p+1)));
    }
  else
    {
    outthreshold[1]=outthreshold[0];
    inthreshold[1]=inthreshold[0];
    }

  if (INIFIND(CONF_HOURS) != NULL)
    {
    INIGETKEY(CONF_HOURS).copyto(buffer, sizeof(buffer));
    tempconfig = atoi(buffer);
    minutes = -1;
    if (tempconfig > 0)
      {
      if ((p = strchr( buffer, ':' )) == NULL)
        p = strchr( buffer, '.' );
      if (p != NULL && strlen( p )==3 && isdigit(p[1]) && isdigit(p[2]))
        minutes = (tempconfig *60) + atoi(p+1);
      else if (p != NULL && *p == '.')
        minutes = (s32)(atol( buffer )* 60.);
      }
    if (minutes < 0)
      minutes = 0;
    }

  #if 0 /* obsolete */
  if (INIFIND(CONF_TIMESLICE) != NULL)
    timeslice = INIGETKEY(CONF_TIMESLICE);
  #else
  timeslice = 0x10000;
  #endif

  #ifdef OLDNICENESS
  if (INIFIND(CONF_NICENESS) != NULL)
    niceness = INIGETKEY(CONF_NICENESS);
  #else  
  tempptr = ini.findfirst( "processor usage", "priority");
  if (tempptr) 
    priority = (ini.getkey("processor usage", "priority", "0")[0]);
  else
    {
    priority = (ini.getkey(OPTION_SECTION, "niceness", "0")[0]);
    priority = ((priority==2)?(8):((priority==1)?(4):(0)));
    }
  #endif    

  if (INIFIND(CONF_UUEHTTPMODE) != NULL)
  uuehttpmode = INIGETKEY(CONF_UUEHTTPMODE);
  if (INIFIND(CONF_HTTPPROXY) != NULL)
  INIGETKEY(CONF_HTTPPROXY).copyto(httpproxy, sizeof(httpproxy));
  if (INIFIND(CONF_HTTPID) != NULL)
  INIGETKEY(CONF_HTTPID).copyto(httpid, sizeof(httpid));
  if (INIFIND(CONF_HTTPPORT) != NULL)
  httpport = INIGETKEY(CONF_HTTPPORT);
  if (INIFIND(CONF_KEYPORT) != NULL)
  keyport = INIGETKEY(CONF_KEYPORT);

  if (INIFIND(CONF_KEYPROXY) == NULL)
    {
    autofindkeyserver = 1;
    keyproxy[0]=0;
    }
  else
    {
    //do an autofind only if the host is a dnet host AND autofindkeyserver is on.
    autofindkeyserver = 0;
    INIGETKEY(CONF_KEYPROXY).copyto(keyproxy, sizeof(keyproxy));
    if (confopt_isstringblank(keyproxy) || strcmpi( keyproxy, "(auto)")==0 ||
      strcmpi( keyproxy, "auto")==0 || strcmpi( keyproxy, "rc5proxy.distributed.net" )==0) 
      {                                         
      keyproxy[0]=0;
      autofindkeyserver = 1; //let Resolve() get a better hostname.
      }
    else if (confopt_IsHostnameDNetHost(keyproxy))
      {
      tempconfig=ini.getkey("networking", "autofindkeyserver", "1")[0];
      autofindkeyserver = (tempconfig)?(1):(0);
      }
    }

  if (INIFIND(CONF_CPUTYPE) != NULL)
  cputype = INIGETKEY(CONF_CPUTYPE);
  if (INIFIND(CONF_NUMCPU) != NULL)
  numcpu = INIGETKEY(CONF_NUMCPU);

  if (INIFIND(CONF_MESSAGELEN) != NULL)
  messagelen = INIGETKEY(CONF_MESSAGELEN);
  if (INIFIND(CONF_SMTPPORT) != NULL)
  smtpport = INIGETKEY(CONF_SMTPPORT);
  if (INIFIND(CONF_SMTPSRVR) != NULL)
  INIGETKEY(CONF_SMTPSRVR).copyto(smtpsrvr, sizeof(smtpsrvr));
  if (INIFIND(CONF_SMTPFROM) != NULL)
  INIGETKEY(CONF_SMTPFROM).copyto(smtpfrom, sizeof(smtpfrom));
  if (INIFIND(CONF_SMTPDEST) != NULL)
  INIGETKEY(CONF_SMTPDEST).copyto(smtpdest, sizeof(smtpdest));

  if (INIFIND(CONF_RANDOMPREFIX) != NULL)
  randomprefix = INIGETKEY(CONF_RANDOMPREFIX);
  if (INIFIND(CONF_PROCESSDES) != NULL)
  preferred_contest_id = INIGETKEY(CONF_PROCESSDES);
  if (INIFIND(CONF_PREFERREDBLOCKSIZE) != NULL)
  preferred_blocksize = INIGETKEY(CONF_PREFERREDBLOCKSIZE);

  blockcount = INIGETKEY(CONF_COUNT);
  tempconfig=ini.getkey(OPTION_SECTION, "runbuffers", "0")[0];
  if (tempconfig) 
    blockcount = -1;

  tempconfig=ini.getkey(OPTION_SECTION, "runoffline", "0")[0];
  offlinemode = (tempconfig != 0);
  tempconfig=ini.getkey(OPTION_SECTION, "percentoff", "0")[0];
  percentprintingoff = (tempconfig != 0);
  tempconfig=ini.getkey(OPTION_SECTION, "frequent", "0")[0];
  connectoften = (tempconfig != 0);
  tempconfig=ini.getkey(OPTION_SECTION, "nodisk", "0")[0];
  nodiskbuffers = (tempconfig != 0);
  tempconfig=ini.getkey(OPTION_SECTION, "quiet", "0")[0];
  quietmode = (tempconfig != 0);
  
  if ( ini.findfirst( OPTION_SECTION, "win95hidden") != NULL )
    {
    tempconfig=ini.getkey(OPTION_SECTION, "win95hidden", "0")[0]; //obsolete
    quietmode = (tempconfig != 0);
    }
  if ( ini.findfirst( OPTION_SECTION, "runhidden") != NULL )
    {
    tempconfig=ini.getkey(OPTION_SECTION, "runhidden", "0")[0]; //obsolete
    quietmode = (tempconfig != 0);
    }
  
  tempconfig=ini.getkey(OPTION_SECTION, "nofallback", "0")[0];
  nofallback= (tempconfig != 0);
  tempconfig=ini.getkey(OPTION_SECTION, "nettimeout", "0")[0];
  if (tempconfig) nettimeout=tempconfig; 
  tempconfig=ini.getkey(OPTION_SECTION, "noexitfilecheck", "0")[0];
  noexitfilecheck = (tempconfig != 0);

  #if defined(LURK)
  tempconfig=ini.getkey(OPTION_SECTION, "lurk", "0")[0];
  dialup.lurkmode = (tempconfig != 0);
  tempconfig=ini.getkey(OPTION_SECTION, "lurkonly", "0")[0];
  if (tempconfig) {dialup.lurkmode=2; connectoften=0;}
  tempconfig=ini.getkey(OPTION_SECTION, "dialwhenneeded", "0")[0];
  if (tempconfig) dialup.dialwhenneeded=1;
  INIGETKEY(CONF_CONNECTNAME).copyto(dialup.connectionname,sizeof(dialup.connectionname));
  #endif

  INIGETKEY(CONF_LOGNAME).copyto(logname, sizeof(logname));
  INIGETKEY(CONF_CHECKPOINT).copyto(checkpoint_file[0], sizeof(checkpoint_file[0]));

  #if 0 /*obsolete */
  INIGETKEY(CONF_CHECKPOINT2).copyto(checkpoint_file[1], sizeof(checkpoint_file[1]));
  #endif

  ini.getkey(OPTION_SECTION,"in",in_buffer_file[0])[0].copyto(in_buffer_file[0],sizeof(in_buffer_file[0]));
  ini.getkey(OPTION_SECTION,"out",out_buffer_file[0])[0].copyto(out_buffer_file[0],sizeof(out_buffer_file[0]));
  ini.getkey(OPTION_SECTION,"in2",in_buffer_file[1])[0].copyto(in_buffer_file[1],sizeof(in_buffer_file[1]));
  ini.getkey(OPTION_SECTION,"out2",out_buffer_file[1])[0].copyto(out_buffer_file[1],sizeof(out_buffer_file[1]));
  ini.getkey(OPTION_SECTION,"pausefile",pausefile)[0].copyto(pausefile,sizeof(pausefile));

  tempconfig=ini.getkey(OPTION_SECTION, "contestdone", "0")[0];
  contestdone[0] = (tempconfig != 0);
  tempconfig=ini.getkey(OPTION_SECTION, "contestdone2", "0")[0];
  contestdone[1]= (tempconfig != 0);

  #if defined(MMX_BITSLICER) || defined(MMX_RC5)
    usemmx=ini.getkey(OPTION_SECTION, "usemmx", "1")[0];
  #endif

  #if defined(NEEDVIRTUALMETHODS)
    InternalReadConfig(ini);
  #endif

  ValidateConfig();

  return( inierror ? -1 : 0 );
}

// --------------------------------------------------------------------------

void Client::ValidateConfig( void ) //DO NOT PRINT TO SCREEN HERE!
{
  unsigned int cont_i;
  
  for (cont_i=0;cont_i<2;cont_i++)
    {
    if ( inthreshold[cont_i] < conf_options[CONF_THRESHOLDI].choicemin ) 
      inthreshold[cont_i] = conf_options[CONF_THRESHOLDI].choicemin;
    if ( inthreshold[cont_i] > conf_options[CONF_THRESHOLDI].choicemax ) 
      inthreshold[cont_i] = conf_options[CONF_THRESHOLDI].choicemax;
    if ( outthreshold[cont_i] < conf_options[CONF_THRESHOLDI].choicemin ) 
      outthreshold[cont_i] = conf_options[CONF_THRESHOLDI].choicemin;
    if ( outthreshold[cont_i] > conf_options[CONF_THRESHOLDI].choicemax ) 
      outthreshold[cont_i] = conf_options[CONF_THRESHOLDI].choicemax;
    if ( outthreshold[cont_i] > inthreshold[cont_i] ) 
      outthreshold[cont_i]=inthreshold[cont_i];

    confopt_killwhitespace(in_buffer_file[cont_i]);
    if (in_buffer_file[cont_i][0] == 0)
      strcpy(in_buffer_file[cont_i], 
        conf_options[((cont_i==0)?(CONF_RC5IN):(CONF_DESIN))].defaultsetting );

    confopt_killwhitespace(out_buffer_file[cont_i]);
    if (confopt_isstringblank(out_buffer_file[cont_i]))
    strcpy(out_buffer_file[cont_i], 
     conf_options[((cont_i==0)?(CONF_RC5OUT):(CONF_DESOUT))].defaultsetting );
    }

  if (blockcount < 0)
    blockcount = -1;

  #if 0 /* obsolete */
  if ( timeslice < 1 ) 
    timeslice = atoi(conf_options[CONF_TIMESLICE].defaultsetting);
  #endif
  
  #ifdef OLDNICENESS
  if ( niceness < conf_options[CONF_NICENESS].choicemin || 
       niceness > conf_options[CONF_NICENESS].choicemax )
    niceness = conf_options[CONF_NICENESS].choicemin;
  #else
  if ( priority < conf_options[CONF_NICENESS].choicemin || 
       priority > conf_options[CONF_NICENESS].choicemax )
    priority = conf_options[CONF_NICENESS].choicemin;
  #endif

  if ( uuehttpmode < conf_options[CONF_UUEHTTPMODE].choicemin || 
       uuehttpmode > conf_options[CONF_UUEHTTPMODE].choicemax ) 
    uuehttpmode = 0;
  if ( (u32)randomprefix < (u32)conf_options[CONF_RANDOMPREFIX].choicemin || 
       (u32)randomprefix > (u32)conf_options[CONF_RANDOMPREFIX].choicemax) 
    randomprefix=100;
  if (smtpport < 0 || smtpport > 65535L) 
    smtpport=25;
  if (messagelen !=0 && messagelen < conf_options[CONF_MESSAGELEN].choicemin)
    messagelen = conf_options[CONF_MESSAGELEN].choicemin;
  if (( preferred_contest_id < 0 ) || ( preferred_contest_id > 1 )) 
    preferred_contest_id = 1;
  if (preferred_blocksize < conf_options[CONF_PREFERREDBLOCKSIZE].choicemin) 
    preferred_blocksize = conf_options[CONF_PREFERREDBLOCKSIZE].choicemin;
  if (preferred_blocksize > conf_options[CONF_PREFERREDBLOCKSIZE].choicemax) 
    preferred_blocksize = conf_options[CONF_PREFERREDBLOCKSIZE].choicemax;
  if ( minutes < 0 ) 
    minutes=0;
  if ( blockcount < 0 ) 
    blockcount=0;
  if (nettimeout < conf_options[CONF_NETTIMEOUT].choicemin) 
    nettimeout=conf_options[CONF_NETTIMEOUT].choicemin;
  else if (nettimeout > conf_options[CONF_NETTIMEOUT].choicemax) 
    nettimeout=conf_options[CONF_NETTIMEOUT].choicemax;

  confopt_killwhitespace(keyproxy);
  confopt_killwhitespace(httpproxy);
  confopt_killwhitespace(smtpsrvr);

  confopt_killwhitespace(id);
  if (id[0]==0)
    strcpy(id,"rc5@distributed.net");

  confopt_killwhitespace(logname);
  if (logname[0]==0 || strcmp(logname,"none")==0)
    logname[0]=0;

  confopt_killwhitespace(pausefile);
  if (confopt_isstringblank(pausefile) || strcmp(pausefile,"none")==0)
    pausefile[0]=0;

  confopt_killwhitespace(checkpoint_file[0]);
  if (confopt_isstringblank(checkpoint_file[0]) || strcmp(checkpoint_file[0],"none")==0)
    checkpoint_file[0][0]=0;

  CheckForcedKeyport();

  //validate numcpu is now in SelectCore(); //1998/06/21 cyrus

#if defined(NEEDVIRTUALMETHODS)
  InternalValidateConfig();
#endif

  InitRandom2( id );
}

// --------------------------------------------------------------------------

//Some OS's write run-time stuff to the .ini, so we protect
//the ini by only allowing that client's internal settings to change.

int Client::WriteConfig(void)  
{
  IniSection ini;

  if ( ini.ReadIniFile( GetFullPathForFilename( inifilename ) ) )
    return WriteFullConfig();
    
  #if defined(NEEDVIRTUALMETHODS)
    InternalWriteConfig(ini);
  #endif

  IniRecord *tempptr;
  if ((tempptr = ini.findfirst(OPTION_SECTION, "runhidden"))!=NULL)
    tempptr->values.Erase();    
  if ((tempptr = ini.findfirst(OPTION_SECTION, "os2hidden"))!=NULL)
    tempptr->values.Erase();    
  if ((tempptr = ini.findfirst(OPTION_SECTION, "win95hidden"))!=NULL)
    tempptr->values.Erase();    
  INISETKEY( CONF_QUIETMODE, ((quietmode)?("1"):("0")) );

  return( ini.WriteIniFile( GetFullPathForFilename( inifilename ) ) ? -1 : 0 );
}

// --------------------------------------------------------------------------

int Client::WriteFullConfig(void) //construct a brand-spanking-new config
{
  IniSection ini;
  char buffer[64];
  IniRecord *tempptr;

  INISETKEY( CONF_ID, id );

  sprintf(buffer,"%d:%d",(int)inthreshold[0],(int)outthreshold[0]);
  INISETKEY( CONF_THRESHOLDI, buffer );

  if (inthreshold[1] == inthreshold[0] && outthreshold[1] == outthreshold[0])
    {
    tempptr = INIFIND(CONF_THRESHOLDI2);
    if (tempptr) tempptr->values.Erase();
    }
  else
    {
    sprintf(buffer,"%d:%d",(int)inthreshold[1],(int)outthreshold[1]);
    INISETKEY( CONF_THRESHOLDI2, buffer );
    }

  sprintf(buffer,"%u:%02u", (unsigned)(minutes/60), (unsigned)(minutes%60)); 
  INISETKEY( CONF_HOURS, buffer );

  #if 0 /* timeslice is obsolete */
  INISETKEY( CONF_TIMESLICE, timeslice );
  #endif
  
  #ifdef OLDNICENESS
  if (niceness != 0 || ini.findfirst( OPTION_SECTION, "niceness" )!=NULL )
    INISETKEY( CONF_NICENESS, niceness );
  #else
  if (priority != 0 || ini.findfirst( "processor usage", "priority")!=NULL )
    ini.setrecord("processor usage", "priority", IniString(priority));
  #endif

  INISETKEY( CONF_CPUTYPE, cputype );
  INISETKEY( CONF_NUMCPU, numcpu );

  //INISETKEY( CONF_RANDOMPREFIX, randomprefix );
  
  INISETKEY( CONF_PREFERREDBLOCKSIZE, preferred_blocksize );
  
  INISETKEY( CONF_NOEXITFILECHECK, noexitfilecheck );
  INISETKEY( CONF_PERCENTOFF, percentprintingoff );
  INISETKEY( CONF_FREQUENT, connectoften );
  INISETKEY( CONF_NODISK, IniString((nodiskbuffers)?("1"):("0")) );
  INISETKEY( CONF_NOFALLBACK, nofallback );
  INISETKEY( CONF_NETTIMEOUT, nettimeout );
  INISETKEY( CONF_LOGNAME, logname );
  INISETKEY( CONF_CHECKPOINT, checkpoint_file[0] );

  tempptr = INIFIND(CONF_CHECKPOINT2); /* obsolete */
  if (tempptr) tempptr->values.Erase();

  INISETKEY( CONF_RC5IN, in_buffer_file[0]);
  INISETKEY( CONF_RC5OUT, out_buffer_file[0]);
  INISETKEY( CONF_DESIN, in_buffer_file[1]);
  INISETKEY( CONF_DESOUT, out_buffer_file[1]);
  INISETKEY( CONF_PAUSEFILE, pausefile);

  #if defined(MMX_BITSLICER) || defined(MMX_RC5)
  /* MMX is a developer option. delete it from the ini */
  tempptr = ini.findfirst(OPTION_SECTION, "usemmx");
  if (tempptr)
    {
    s32 xyz = (ini.getkey(OPTION_SECTION, "usemmx", "0")[0]);
    if ( xyz!= 0 || (GetProcessorType(1) & 0x100) != 0)
      tempptr->values.Erase();
    }
  #endif

  if ((tempptr = ini.findfirst(OPTION_SECTION, "runhidden"))!=NULL)
    tempptr->values.Erase();    
  if ((tempptr = ini.findfirst(OPTION_SECTION, "os2hidden"))!=NULL)
    tempptr->values.Erase();    
  if ((tempptr = ini.findfirst(OPTION_SECTION, "win95hidden"))!=NULL)
    tempptr->values.Erase();    
  INISETKEY( CONF_QUIETMODE, ((quietmode)?("1"):("0")) );

  if (offlinemode != 0 && offlinemode != 1) 
    {
    blockcount = -1;
    offlinemode = 0;
    }
  ini.setrecord(OPTION_SECTION, "runoffline", IniString((offlinemode)?("1"):("0")));
  INISETKEY( CONF_COUNT, blockcount );
  
  if ((tempptr = ini.findfirst(OPTION_SECTION, "runbuffers"))!=NULL)
    tempptr->values.Erase();   /* obsolete */
    
  ini.setrecord(OPTION_SECTION, "contestdone",  IniString(contestdone[0]));
  ini.setrecord(OPTION_SECTION, "contestdone2", IniString(contestdone[1]));

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


  if ((tempptr = ini.findfirst( "networking", "autofindkeyserver"))!=NULL)
    tempptr->values.Erase();
  if (uuehttpmode <= 1)
    {
    // wipe out httpproxy and httpport & httpid
    tempptr = INIFIND( CONF_UUEHTTPMODE );
    if (tempptr) tempptr->values.Erase();
    tempptr = INIFIND( CONF_HTTPPROXY );
    if (tempptr) tempptr->values.Erase();
    tempptr = INIFIND( CONF_HTTPPORT );
    if (tempptr) tempptr->values.Erase();
    tempptr = INIFIND( CONF_HTTPID );
    if (tempptr) tempptr->values.Erase();

    if (confopt_isstringblank(keyproxy) || (autofindkeyserver && confopt_IsHostnameDNetHost(keyproxy)))
      {
      //autokeyserver is enabled (because its on AND its a dnet host), so delete 
      //the old ini keys so that old inis stay compatible. We could at this 
      //point set keyproxy=rc5proxy.distributed.net, but why clutter up the ini?
      tempptr = ini.findfirst(OPTION_SECTION, "keyproxy");
      if (tempptr) tempptr->values.Erase();
      }
    else 
      {
      if (confopt_IsHostnameDNetHost(keyproxy))
        ini.setrecord("networking", "autofindkeyserver", IniString("0"));
      INISETKEY( CONF_KEYPROXY, keyproxy );
      }
    INISETKEY( CONF_KEYPORT, keyport );
    }
  else
    {
    INISETKEY( CONF_UUEHTTPMODE, uuehttpmode );
    INISETKEY( CONF_HTTPPROXY, httpproxy );
    INISETKEY( CONF_HTTPPORT, httpport );
    INISETKEY( CONF_HTTPID, httpid);

    tempptr = INIFIND( CONF_KEYPROXY );
    if (tempptr) tempptr->values.Erase();
    tempptr = INIFIND( CONF_KEYPORT );
    if (tempptr) tempptr->values.Erase();
    }

  if (messagelen == 0)
    {
    tempptr = INIFIND( CONF_MESSAGELEN );
    if (tempptr) tempptr->values.Erase();
    tempptr = INIFIND( CONF_SMTPSRVR );
    if (tempptr) tempptr->values.Erase();
    tempptr = INIFIND( CONF_SMTPPORT );
    if (tempptr) tempptr->values.Erase();
    tempptr = INIFIND( CONF_SMTPFROM );
    if (tempptr) tempptr->values.Erase();
    tempptr = INIFIND( CONF_SMTPDEST );
    if (tempptr) tempptr->values.Erase();
    }
  else
    {  
    INISETKEY( CONF_MESSAGELEN, messagelen );
    INISETKEY( CONF_SMTPSRVR, smtpsrvr );
    INISETKEY( CONF_SMTPPORT, smtpport );
    INISETKEY( CONF_SMTPFROM, smtpfrom );
    INISETKEY( CONF_SMTPDEST, smtpdest );
    }


  #if defined(NEEDVIRTUALMETHODS)
    InternalWriteConfig(ini);
  #endif

  return( ini.WriteIniFile( GetFullPathForFilename( inifilename ) ) ? -1 : 0 );
}

// --------------------------------------------------------------------------

bool Client::CheckForcedKeyport(void)
{
  bool Forced = false;
  char *dot = strchr(keyproxy, '.');
  if (dot && (strcmpi(dot, ".v27.distributed.net") == 0 ||
      strcmpi(dot, ".distributed.net") == 0) && !autofindkeyserver)
  {
    int foundport = 2064;
    for (char *p = keyproxy; p < dot; p++)
      if (isdigit(*p)) { foundport = atoi(p); break; }
    if (foundport == 2064 || foundport == 23 || foundport == 80)
    {
      if (keyport != 3064 && keyport != foundport)
      {
        keyport = foundport;
        Forced = true;
      }
    }
  }
  return Forced;
}

// --------------------------------------------------------------------------

bool Client::CheckForcedKeyproxy(void)
{
  bool Forced = false;
  char buffer[200];
  char *temp;
  char *dot = strchr(keyproxy, '.');
  if (dot && (strcmpi(dot, ".v27.distributed.net") == 0 ||
      strcmpi(dot, ".distributed.net") == 0) && !autofindkeyserver)
  {
      if (keyport != 3064)// && keyport != foundport)
      {
        if ((keyport == 80) || (keyport == 23))
          {
          buffer[0]=0;
          for (temp=&keyproxy[0];isalpha(*temp) > 0;temp++) {};
          *temp=0;
          strcpy(buffer,keyproxy);
          sprintf(keyproxy,"%s%li.v27.distributed.net",buffer,(long)keyport);
          }
        else if (keyport == 2064)
          {
          buffer[0]=0;
          for (temp=&keyproxy[0];isalpha(*temp) > 0;temp++) {};
          *temp=0;
          strcpy(buffer,keyproxy);
          sprintf(keyproxy,"%s.v27.distributed.net",buffer);
          }
        else
          {
//          keyport = foundport;
          Forced = true;
          };
//      }
    }
  }
  return Forced;
}

// --------------------------------------------------------------------------

