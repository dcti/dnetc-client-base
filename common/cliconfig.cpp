// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: cliconfig.cpp,v $
// Revision 1.182  1998/10/03 16:56:43  sampo
// Finished the ConClear() replacement of CliScreenClear.
//
// Revision 1.181  1998/08/28 22:18:21  cyp
// Commandline parse function is now in cmdline.cpp. Also moved all non-config
// functions (SetNiceness()/RunStartup()/[Un]Install()/PrintBanner()) to
// client.cpp
//
// Revision 1.180  1998/08/21 23:44:31  cyruspatel
// Spun off SelectCore() to selcore.cpp  Client::cputype _must_ be a valid
// core type (and no longer -1) after a call to SelectCore() - Problem::Run()
// and/or Problem::LoadState() depend (or will soon depend) on it.
//
// Revision 1.179  1998/08/21 06:07:57  cyruspatel
// Extended the DES mmx define wrapper in SelectCore from #if MMX_BITSLICER
// to #if (defined(MMX_BITSLICER) && defined(KWAN) && defined(MEGGS)) to
// differentiate between DES and RC5 MMX cores. (Can't use the DES MMX core
// with NetWare). Also made all the option table strings const char *[]
// instead of char [][60]. Also, most options in the config menu now revert
// to the last-known-good value rather than the default when an invalid value
// is entered.
//
// Revision 1.178  1998/08/20 20:38:27  cyruspatel
// Mail specific option validation removed. Mail.cpp is RFC822 aware and can
// recognize/deal with invalid addresses better than ValidateConfig() can.
//
// Revision 1.177  1998/08/20 19:34:21  cyruspatel
// Removed that terrible PIPELINE_COUNT hack: Timeslice and pipeline count
// are now computed in Problem::LoadState(). Client::SelectCore() now saves
// core type to Client::cputype.
//
// Revision 1.176  1998/08/20 02:11:43  silby
// Changed MMX option to specify that it was for DES MMX cores only.
//
// Revision 1.175  1998/08/20 00:10:26  silby
// Change in syntax to make GCC happy.
//
// Revision 1.174  1998/08/15 21:28:05  jlawson
// added ifdef MMX_BITSLICE around mmx core selection to allow compilation
// without mmx cores
//
// Revision 1.173  1998/08/14 00:23:19  silby
// Changed WriteContestAndPrefixConfig so that it would not attempt to do so if nodiskbuffers was specified (aiming towards complete diskless operation aside from the inital .ini read)
//
// Revision 1.172  1998/08/14 00:04:48  silby
// Changes for rc5 mmx core integration.
//
// Revision 1.171  1998/08/10 23:02:22  cyruspatel
// xxxTrigger and pausefilefound flags are now wrapped in trigger.cpp 
//
// Revision 1.170  1998/08/07 05:28:47  silby
// Changed lurk so that win32 users can now easily select the connection to use for dial on demand.
//
// Revision 1.169  1998/08/05 19:04:14  cyruspatel
// Changed some Log()/LogScreen()s to LogRaw()/LogScreenRaw()s, ensured that
// DeinitializeLogging() is called, and ensured that InitializeLogging() is
// called only once.
//
// Revision 1.168  1998/08/05 16:39:28  cberry
// Changed register len to register int len in isstringblank()
//
// Revision 1.167  1998/08/02 16:05:08  cyruspatel
// Converted all printf()s to LogScreen()s (LogScreen() and LogScreenf() are
// synonymous)
//
// Revision 1.166  1998/08/02 03:16:20  silby
// Major reorganization:  Log,LogScreen, and LogScreenf are now in logging.cpp, and are global functions - client.h #includes logging.h, which is all you need to use those functions.  Lurk handling has been added into the Lurk class, which resides in lurk.cpp, and is auto-included by client.h if lurk is defined as well. baseincs.h has had lurk-specific win32 includes moved to lurk.cpp, cliconfig.cpp has been modified to reflect the changes to log/logscreen/logscreenf, and mail.cpp uses logscreen now, instead of printf. client.cpp has had variable names changed as well, etc.
//
// Revision 1.165  1998/07/30 05:08:52  silby
// Fixed DONT_USE_PATHWORK handling, ini_etc strings were still being included, now they are not. Also, added the logic for dialwhenneeded, which is a new lurk feature.
//
// Revision 1.164  1998/07/30 02:33:20  blast
// Lowered minimum network timeout from 30 to 5 ...
//
// Revision 1.163  1998/07/29 05:14:33  silby
// Changes to win32 so that LurkInitiateConnection now works - required the addition of a new .ini key connectionname=.  Username and password are automatically retrieved based on the connectionname.
//
// Revision 1.162  1998/07/29 01:49:44  cyruspatel
// Fixed email address from not being editable (the option to return to the
// main menu is '0', and 0 is also the value of CONF_ID). 
//
// Revision 1.161  1998/07/26 21:18:48  cyruspatel
// Modified Client::ConfigureGeneral() to work with 'autofindkeyserver'.
//
// Revision 1.160  1998/07/26 12:45:42  cyruspatel
// new inifile option: 'autofindkeyserver', ie if keyproxy= points to a
// xx.v27.distributed.net then that will be interpreted by Network::Resolve()
// to mean 'find a keyserver that covers the timezone I am in'. Network
// constructor extended to take this as an argument.
//
// Revision 1.159  1998/07/25 05:29:43  silby
// Changed all lurk options to use a LURK define (automatically set in client.h) so that lurk integration of mac/amiga clients needs only touch client.h and two functions in client.cpp
//
// Revision 1.158  1998/07/22 00:02:35  silby
// Changes so that win32gui priorities will be more shutdown friendly.
//
// Revision 1.157  1998/07/18 18:06:36  cyruspatel
// Fixed a ');' I misplaced.
//
// Revision 1.156  1998/07/18 17:10:09  cyruspatel
// DOS client specific change: PrintBanner now displays PMODE copyright.
//
// Revision 1.155  1998/07/14 09:06:30  remi
// Users are now able to change the 'usemmx' setting with -config.
//
// Revision 1.154  1998/07/14 08:03:29  myshkin
// Added #include "clirate.h" for the appropriate PPC platforms, to declare
// CliGetKeyrateForProblemNoSave().
//
// Revision 1.153  1998/07/13 23:54:19  cyruspatel
// Cleaned up NO!NETWORK handling.
//
// Revision 1.152  1998/07/13 12:40:25  kbracey
// RISC OS update. Added -noquiet option.
//
// Revision 1.151  1998/07/13 03:29:45  cyruspatel
// Added 'const's or 'register's where the compiler was complaining about
// "declaration/type or an expression" ambiguities.
//
// Revision 1.150  1998/07/11 23:49:28  cyruspatel
// Added a check for OS2/Win/WinNT VMs to the DOS client - it spits out an
// advisory notice (to use a native client) if it finds itself running in one
// those environments.
//
// Revision 1.149  1998/07/11 09:47:14  cramer
// Added support for solaris numcpu auto detection.
//
// Revision 1.148  1998/07/11 02:34:44  cramer
// Added automagic number of cpu detection for linux.  If it cannot detect the
// number of processors, a warning is issued and we assume it's only got one.
// (Note to Linus: add num_cpus to struct sysinfo.)
//
// Revision 1.147  1998/07/11 02:26:21  cyruspatel
// Removed an obsolete 'Overriding Autodetect' message from x86
// section of ::SelectCore()
//
// Revision 1.146  1998/07/11 01:12:16  silby
// Added code so that the win32 cli will change its title on start.
//
// Revision 1.145  1998/07/10 06:24:30  cramer
// Augmented the domain name requirements for logging.  If no domain is
// specified, then the smtpsrvr will be appended -- for both user, and user@
// entries.  [Silby's modification could be tricked into allowing user@.]
// TODO: fix it for no smtpsrvr being specified.  (just don't display those
// options?)
//
// Revision 1.144  1998/07/10 04:04:27  silby
// Change to thread priorities for win32 gui.
//
// Revision 1.143  1998/07/09 17:08:09  silby
// Ok, DES MMX core selection override code now works. The output of
// cpucheck and usemmx are the only things it cares about; everyone who's
// mmx capable will use it now.
//
// Revision 1.142  1998/07/09 05:27:51  silby
// Changes to the MMX autodetect - still appear to be some overrides from
// user-set settings happening on pentium mmxes.
//
// Revision 1.141  1998/07/09 03:21:20  silby
// Changed autodetect so that autodetection on x86 is always done, and mmx
// cores are forced if usemmx=1 no matter which core is selected. (prevents
// costly des keyrate loss)
//
// Revision 1.140  1998/07/09 03:13:20  remi
// Fixed -nommx message for x86 clients without the MMX bitslicer.
//
// Revision 1.139  1998/07/09 01:44:16  silby
// Added ifdefs so a non-mmx x86 build was still possible.
//
// Revision 1.138  1998/07/08 23:48:33  foxyloxy
// Typo in des-slice-meggs.cpp fixed to allow non-mmx clients to
// compile (NOTSZERO changed back to NOTZERO).
//
// Revision 1.137  1998/07/08 23:31:51  foxyloxy
// Added defines to allow non-x86 platforms to compile and properly recognize
// but ignore the -nommx command line option.
//
// Revision 1.136  1998/07/08 16:29:52  remi
// Adjusted DES credits.
//
// Revision 1.135  1998/07/08 09:46:31  remi
// Added support for the MMX bitslicer.
// Added -nommx command line option and 'usemmx' ini file setting so the user
// can disable MMX usage. Wrapped $Log comments to some reasonable value.
//
// Revision 1.134  1998/07/08 05:19:18  jlawson
// updates to get Borland C++ to compile under Win32.
//
// Revision 1.133  1998/07/07 23:03:47  jlawson
// eliminated printf warning again
//
// Revision 1.132  1998/07/07 21:55:10  cyruspatel
// client.h has been split into client.h and baseincs.h 
//
// Revision 1.131  1998/07/07 07:28:36  jlawson
// eliminated printf type warning with gcc
//
// Revision 1.130  1998/07/06 01:28:37  cyruspatel
// Modified DOS signal handling to also trap ctrl-break. Timeslice is now
// fixed at 10 times value returned from GetTimesliceBaseline(). (a lot!)
//
// Revision 1.129  1998/07/05 22:03:14  silby
// Someone forgot to #if some non-pathwork code out.
//

#if (!defined(lint) && defined(__showids__))
const char *cliconfig_cpp(void) {
return "@(#)$Id: cliconfig.cpp,v 1.182 1998/10/03 16:56:43 sampo Exp $"; }
#endif

#include "console.h"
#include "cputypes.h"
#include "client.h"   // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h" // basic (even if port-specific) #includes
#include "version.h"
#include "iniread.h"
#include "network.h"  
#include "triggers.h" //[Check|Raise][Pause|Exit]RequestTrigger()/InitXHandler()
#include "scram.h"     // InitRandom2(id)
#include "pathwork.h"
#include "logstuff.h"  //Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "selcore.h"   //SelectCore() and GetCoreNameFromCoreType()

#if ((CLIENT_OS == OS_OS2) || (CLIENT_OS==OS_WIN32))
#include "lurk.h"      //lurk stuff
#endif


#if (CLIENT_OS == OS_WIN32)
#if defined(WINNTSERVICE)
  #define NTSERVICEID "rc5desnt"
#else
  #include "sleepdef.h" //used by RunStartup()
#endif  
#endif


// --------------------------------------------------------------------------

#define OPTION_COUNT    47
#define MAXMENUENTRIES  18
static const char *OPTION_SECTION="parameters"; //#define OPTION_SECTION "parameters"

// --------------------------------------------------------------------------

#if defined(NOCONFIG)
  #define CFGTXT(x) NULL
#else
  #define CFGTXT(x) x
#endif

// --------------------------------------------------------------------------

#if !defined(NOCONFIG)
static const char *menutable[6]=
  {
  "Required Options",
  "Logging Options",
  "Communication Options",
  "Performance Options",
  "Miscellaneous Options",
  "Filenames & Path Options"
  };

static const char *nicenesstable[]=
  {
  "Extremely Nice",
  "Nice",
  "Nasty"
  };

static const char *uuehttptable[]= 
  {
  "No special encoding",
  "UUE encoding (telnet proxies)",
  "HTTP encoding",
  "HTTP+UUE encoding",
  "SOCKS4 proxy",
  "SOCKS5 proxy"
  };

static const char *offlinemodetable[]=
  {
  "Normal Operation",
  "Offline Always (no communication)",
  "Finish Buffers and exit"
  };

static const char *lurkmodetable[]=
  {
  "Normal mode",
  "Dial-up detection mode",
  "Dial-up detection ONLY mode"
  };

#endif  // !NOCONFIG

// --------------------------------------------------------------------------

struct optionstruct
  {
  const char *name;//name of the option in the .ini file
  const char *description;//description of the option
  const char *defaultsetting;//default setting
  const char *comments;//additional comments
  s32 optionscreen;//screen to appear on
  s32 type;//type: 0=other, 1=string, 2=integer, 3=boolean (yes/no)
  s32 menuposition;//number on that menu to appear as
  void *thevariable;//pointer to the variable
  const char **choicelist;//pointer to the char* array of choices
                         //(used for numeric responses)
  s32 choicemin;//minimum choice number
  s32 choicemax;//maximum choice number
  };

// --------------------------------------------------------------------------

static optionstruct options[OPTION_COUNT]=
{
//0
{ "id", CFGTXT("Your E-mail address"), "rc5@distributed.net", CFGTXT("(64 characters max)"),1,1,1,NULL},
//1
{ "threshold", CFGTXT("RC5 Blocks to Buffer"), "10", CFGTXT("(max 1000)"),1,2,2,NULL},
//2
{ "threshold", CFGTXT("RC5 block flush threshold"), "10",
    CFGTXT("\nSet this equal to RC5 Blocks to Buffer except in rare cases."),5,2,3,NULL},
//3
{ "threshold2", CFGTXT("DES Blocks to Buffer"), "10", CFGTXT("(max 1000)"),1,2,3,NULL},
//4
{ "threshold2", CFGTXT("DES block flush threshold"), "10",
    CFGTXT("\nSet this equal to DES Blocks to Buffer except in rare cases."),5,2,4,NULL},
//5
{ "count", CFGTXT("Complete this many blocks, then exit"), "0", CFGTXT("(0 = no limit)"),5,2,1,NULL},
//6
{ "hours", CFGTXT("Run for this many hours, then exit"), "0.00", CFGTXT("(0 = no limit)"),5,1,2,NULL},
//7
{ "timeslice", CFGTXT("Keys per timeslice"),
#if (CLIENT_OS == OS_RISCOS)
    "2048",
#else
    "65536",
#endif
    CFGTXT("\nThe lower the value, the less impact the client will have on your system, but\n"
    "the slower it will go. Values from 256 to 65536 are good."),4,2,5,NULL},
//8
{ "niceness", CFGTXT("Level of niceness to run at"), "0",
  CFGTXT("\n\nExtremely Nice will not slow down other running programs.\n"
  "Nice may slow down other idle-mode processes.\n"
  "Nasty will cause the client to run at regular user level priority.\n\n"
  "On a completely idle system, all options will result in the same\n"
  "keyrate. For this reason, Extremely Nice is recommended.\n"),4,2,1,NULL,
  CFGTXT(&nicenesstable[0]),0,2},
//9
{ "logname", CFGTXT("File to log to"), "none", CFGTXT("(128 characters max, none = no log)\n"),2,1,1,NULL},
//10
{ "uuehttpmode", CFGTXT("Firewall Communications mode (UUE/HTTP/SOCKS)"), "0",
  CFGTXT(""),3,2,1,NULL,CFGTXT(&uuehttptable[0]),0,5},
//11
{ "keyproxy", CFGTXT("Preferred KeyServer Proxy"), "us.v27.distributed.net",
   CFGTXT("\nThis specifies the DNS or IP address of the keyserver your client will\n"
   "communicate with. Unless you have a special configuration, use the setting\n"
   "automatically set by the client."),3,1,2,NULL},
//12
{ "keyport", CFGTXT("Preferred KeyServer Port"), "2064", CFGTXT("(TCP/IP port on preferred proxy)"),3,2,3,NULL},
//13
{ "httpproxy", CFGTXT("Local HTTP/SOCKS proxy address"),
       "wwwproxy.corporate.com", CFGTXT("(DNS or IP address)\n"),3,1,4,NULL},
//14
{ "httpport", CFGTXT("Local HTTP/SOCKS proxy port"), "80", CFGTXT("(TCP/IP port on HTTP proxy)"),3,2,5,NULL},
//15
{ "httpid", CFGTXT("HTTP/SOCKS proxy userid/password"), "", CFGTXT("(Enter userid (. to reset it to empty) )"),3,1,6,NULL},
//16
{ "cputype", CFGTXT("CPU type...not applicable in this client"), "-1", CFGTXT("(default -1)"),0,2,0,
  NULL,NULL,0,0},
//17
{ "messagelen", CFGTXT("Message Mailing (bytes)"), "0", CFGTXT("(0=no messages mailed.  10000 recommended.  125000 max.)\n"),2,2,2,NULL},
//18
{ "smtpsrvr", CFGTXT("SMTP Server to use"), "your.smtp.server", CFGTXT("(128 characters max)"),2,1,3,NULL},
//19
{ "smtpport", CFGTXT("SMTP Port"), "25", CFGTXT("(SMTP port on mail server -- default 25)"),2,2,4,NULL},
//20
{ "smtpfrom", CFGTXT("E-mail address that logs will be mailed from"), "a.computer@your.site", CFGTXT("\n(Some servers require this to be a real address)\n"),2,1,5,NULL},
//21
{ "smtpdest", CFGTXT("E-mail address to send logs to"), "you@your.site", CFGTXT("\n(Full name and site eg: you@your.site.  Comma delimited list permitted)\n"),2,1,6,NULL},
//22
#if ((CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_BEOS)) || (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_SOLARIS)
  { "numcpu", CFGTXT("Number of CPUs in this machine"), "-1 (autodetect)", "\n"
#else
  { "numcpu", CFGTXT("Number of CPUs in this machine"), "1", "\n"
#endif
,4,2,3,NULL},
//23
{ "checkpointfile", CFGTXT("RC5 Checkpoint Path/Name"),"none",
  CFGTXT("\n(Non-shared file required.  ckpoint" EXTN_SEP "rc5 recommended.  'none' to disable)\n")
  ,6,1,1,NULL},
//24
{ "checkpointfile2", "DES Checkpoint Path/Name","none",
  CFGTXT("\n(Non-shared file required.  ckpoint" EXTN_SEP "des recommended.  'none' to disable)\n")
  ,6,1,2,NULL},
//25
{ "randomprefix", CFGTXT("High order byte of random blocks"),"100",CFGTXT("Do not change this"),0,2,0,NULL},
//26
{ "preferredblocksize", CFGTXT("Preferred Block Size (2^X keys/block)"),"30",
  CFGTXT("(2^28 -> 2^31)"),5,2,5,NULL},
//27
{ "processdes", CFGTXT("Compete in DES contests?"),"1",CFGTXT(""),5,3,6,NULL},
//28
{ "quiet", CFGTXT("Disable all screen output? (quiet mode)"),"0",CFGTXT(""),5,3,7,NULL},
//29
{ "noexitfilecheck", CFGTXT("Disable exit file checking?"),"0",(""),5,3,8,NULL},
//30
{ "percentoff", CFGTXT("Disable block percent completion indicators?"),"0",CFGTXT(""),5,3,9,NULL},
//31
{ "frequent", CFGTXT("Attempt keyserver connections frequently?"),"0",CFGTXT(""),3,3,6,NULL},
//32
{ "nodisk", CFGTXT("Buffer blocks in RAM only? (no disk I/O)"),"0",
    CFGTXT("\nNote: This option will cause all buffered, unflushable blocks to be lost\n"
    "during client shutdown!"),5,3,10,NULL},
//33
{ "nofallback", CFGTXT("Disable fallback to US Round-Robin?"),"0",
  CFGTXT("\nIf your specified proxy is down, the client normally falls back\n"
  "to the US Round-Robin (us.v27.distributed.net) - this option causes\n"
  "the client to NEVER attempt a fallback if the local proxy is down."),
  3,3,7,NULL},
//34
{ "cktime", CFGTXT("Interval between saving of checkpoints (minutes):"),"5",
  CFGTXT(""),5,2,11,NULL},
//35
{ "nettimeout", CFGTXT("Network Timeout (seconds)"), "60",CFGTXT(" "),3,2,8,NULL},
//36
{ "exitfilechecktime", CFGTXT("Exit file check time (seconds)"),"30",CFGTXT(""),5,2,12,NULL},
//37
{ "runbuffers", CFGTXT("Offline operation mode"),"0",
  CFGTXT("\nNormal Operation: The client will connect to a keyserver as needed,\n"
  "        and use random blocks if a keyserver connection cannot be made.\n"
  "Offline Always: The client will never connect to a keyserver, and will\n"
  "        generate random blocks if the block buffers empty.)\n"
  "Finish Buffers and exit: The client will never connect\n"
  "        to a keyserver, and when the block buffers empty, it will\n"
  "        terminate.\n"),3,2,9,NULL,CFGTXT(&offlinemodetable[0]),0,2},
//38
{ "lurk", CFGTXT("Modem detection options"),"0",
  CFGTXT("\nNormal mode: the client will send/receive blocks only when it\n"
  "        empties the in buffer, hits the flush threshold, or the user\n"
  "        specifically requests a flush/fetch.\n"
  "Dial-up detection mode: This acts like mode 0, with the addition\n"
  "        that the client will automatically send/receive blocks when a\n"
  "        dial-up networking connection is established. Modem users\n"
  "        will probably wish to use this option so that their client\n"
  "        never runs out of blocks.\n"
  "Dial-up detection ONLY mode: Like the previous mode, this will cause\n"
  "        the client to automatically send/receive blocks when\n"
  "        connected. HOWEVER, if the client runs out of blocks,\n"
  "        it will NOT trigger auto-dial, and will instead work\n"
  "        on random blocks until a connection is detected.\n"),
  3,2,10,NULL,CFGTXT(&lurkmodetable[0]),0,2},
//39
{ "in",  CFGTXT("RC5 In-Buffer Path/Name"),  "buff-in"  EXTN_SEP "rc5",CFGTXT(""),6,1,4,NULL},
//40
{ "out", CFGTXT("RC5 Out-Buffer Path/Name"), "buff-out" EXTN_SEP "rc5",CFGTXT(""),6,1,5,NULL},
//41
{ "in2", CFGTXT("DES In-Buffer Path/Name"),  "buff-in"  EXTN_SEP "des",CFGTXT(""),6,1,6,NULL},
//42
{ "out2",CFGTXT("DES Out-Buffer Path/Name"), "buff-out" EXTN_SEP "des",CFGTXT(""),6,1,7,NULL},
//43
{ "pausefile",CFGTXT("Pausefile Path/Name"),"none",CFGTXT("(blank = no pausefile)"),6,1,3,NULL},
//44
#ifdef MMX_BITSLICER
{ "usemmx",CFGTXT("Use MMX DES cores if possible?"),"1",CFGTXT(""),4,3,4,NULL},
#else
{ "usemmx", CFGTXT("Use MMX...not applicable in this client"), "-1", CFGTXT("(default -1)"),0,2,0,
  NULL,NULL,0,0},
#endif
{ "dialwhenneeded", CFGTXT("Dial the Internet when needed?"),"0",
  CFGTXT(""),3,3,11,NULL},
{ "connectionname", CFGTXT("Dial-up Connection Name"),
  "Your Internet Connection",CFGTXT(""),3,1,12,NULL}
};

// --------------------------------------------------------------------------

static int isstringblank( const char *string )
{
  register int len = ( string ? ( strlen( string )+1 ) : 0 );

  while (len)
    {
    len--;
    if ( isprint( string[len] ) && !isspace( string[len] ) )
      return 0;
    }
  return 1;
}

static void killwhitespace( char *string )
{
  char *opos, *ipos;
  ipos = opos = string;
  while ( *ipos )
    {
    if ( !isspace( *ipos ) )
      *opos++ = *ipos;
    ipos++;
    }
  *opos = 0;
  return;
}

int yesno(char *str)
// checks for user to type yes or no.
// Returns 1=yes, 0=no, -1=unknown
{
  if (strcmpi(str, "yes")==0) return 1;
  if (strcmpi(str, "no")==0) return 0;
  return -1;
}


static int _IsHostnameDNetHost( const char * hostname )
{
  if (!hostname || !*hostname) //shouldn't happen
    return 1;
  if (isdigit( *hostname )) //illegal
    return 0;
  const char *dot = (const char *)strchr( hostname, '.');
  if ( dot == NULL || dot == hostname )
    return 0;
  return (strcmpi(dot, ".v27.distributed.net") == 0 ||
      strcmpi(dot, ".distributed.net") == 0);
}           

// --------------------------------------------------------------------------

#define CONF_ID 0
#define CONF_THRESHOLDI 1
#define CONF_THRESHOLDO 2
#define CONF_THRESHOLDI2 3
#define CONF_THRESHOLDO2 4
#define CONF_COUNT 5
#define CONF_HOURS 6
#define CONF_TIMESLICE 7
#define CONF_NICENESS 8
#define CONF_LOGNAME 9
#define CONF_UUEHTTPMODE 10
#define CONF_KEYPROXY 11
#define CONF_KEYPORT 12
#define CONF_HTTPPROXY 13
#define CONF_HTTPPORT 14
#define CONF_HTTPID 15
#define CONF_CPUTYPE 16
#define CONF_MESSAGELEN 17
#define CONF_SMTPSRVR 18
#define CONF_SMTPPORT 19
#define CONF_SMTPFROM 20
#define CONF_SMTPDEST 21
#define CONF_NUMCPU 22
#define CONF_CHECKPOINT 23
#define CONF_CHECKPOINT2 24
#define CONF_RANDOMPREFIX 25
#define CONF_PREFERREDBLOCKSIZE 26
#define CONF_PROCESSDES 27
#define CONF_QUIETMODE 28
#define CONF_NOEXITFILECHECK 29
#define CONF_PERCENTOFF 30
#define CONF_FREQUENT 31
#define CONF_NODISK 32
#define CONF_NOFALLBACK 33
#define CONF_CKTIME 34
#define CONF_NETTIMEOUT 35
#define CONF_EXITFILECHECKTIME 36
#define CONF_OFFLINEMODE 37
#define CONF_LURKMODE 38
#define CONF_RC5IN 39
#define CONF_RC5OUT 40
#define CONF_DESIN 41
#define CONF_DESOUT 42
#define CONF_PAUSEFILE 43
#define CONF_MMX 44
#define CONF_DIALWHENNEEDED 45
#define CONF_CONNECTNAME 46

// --------------------------------------------------------------------------

#if !defined(NOCONFIG)
s32 Client::ConfigureGeneral( s32 currentmenu )
{
  char parm[128],parm2[128];
  s32 choice=1;
  s32 temp;
  s32 temp2;
  char str[3];
  char *p;

  do                    //note: don't return or break from inside 
    {                   //the loop. Let it fall through instead. - cyp
    setupoptions();

    #ifndef OLDRESOLVE                   //This is an ugly, ugly, hack and
    char lkg_keyproxy[sizeof(keyproxy)]; //I'd appreciate any feedback re:
    strcpy( lkg_keyproxy, keyproxy );    //ways to improve it. - cyp
    if ( autofindkeyserver && _IsHostnameDNetHost( keyproxy ) )
      strcpy( keyproxy, "(auto)" );
    options[CONF_KEYPROXY].defaultsetting = "(auto)";
    options[CONF_KEYPROXY].comments = 
    "\nThis is the name or IP address of the machine that your client will\n"
    "obtain keys from and send completed blocks to. If your client will be\n"
    "connecting to a personal proxy, then enter the name or address of that\n"
    "machine here. If your client will be connecting directly to one of the\n"
    "distributed.net main key servers, then leave this setting at \"(auto)\".\n"
    "The client will then automatically select a key server close to you.\n";
    #endif

    // display menu

    do   //while invalid CONF_xxx option selected
      {
      ConClear(); //in logstuff.cpp
      LogScreenRaw("Distributed.Net RC5/DES Client build " 
                  CLIENT_VERSIONSTRING " config menu\n" );
      LogScreenRaw("%s\n",menutable[currentmenu-1]);
      LogScreenRaw("------------------------------------------------------------\n\n");

      for ( temp2=1; temp2 < MAXMENUENTRIES; temp2++ )
        {
        choice=findmenuoption(currentmenu,temp2);
        if (choice >= 0)
          {
          LogScreenRaw("%2d) %s ==> ",
                 (int)options[choice].menuposition,
                 options[choice].description);

          if (options[choice].type==1)
            {
            if (options[choice].thevariable != NULL)
              LogScreenRaw("%s\n",(char *)options[choice].thevariable);
            }
          else if (options[choice].type==2)
            {
            if (options[choice].choicelist == NULL)
              LogScreenRaw("%li\n",(long)*(s32 *)options[choice].thevariable);
            else 
              LogScreenRaw("%s\n",options[choice].choicelist[
                ((long)*(s32 *)options[choice].thevariable)]);
            }
          else if (options[choice].type==3)
            {
            sprintf(str, "%s", *(s32 *)options[choice].thevariable?"yes":"no");
            LogScreenRaw("%s\n",str);
            }
          }
        }
      LogScreenRaw("\n 0) Return to main menu\n");

      // get choice from user
      LogScreenRaw("\nChoice --> ");
      fflush( stdout );
      fflush( stdin );
      fgets(parm, sizeof(parm), stdin);
      fflush( stdin );
      fflush( stdout );
      choice = atoi( parm );
      if ( choice == 0 || CheckExitRequestTrigger())
        choice = -2; //quit request
      else if ( choice > 0 )
        choice = findmenuoption(currentmenu,choice); // returns -1 if !found
      else
        choice = -1;
      } while ( choice == -1 ); //while findmenuoption() says this is illegal

    if ( choice >= 0 ) //if valid CONF_xxx option
      {
      // prompt for new value
      if (options[choice].type==1)
        {
        if (options[choice].choicelist == NULL)
          {
          LogScreenRaw("\n%s %s\nDefault Setting: %s\nCurrent Setting: %s\nNew Setting --> ",
                options[choice].description, options[choice].comments,
                options[choice].defaultsetting,(char *)options[choice].thevariable);
          }
        else
          {
          LogScreenRaw("\n%s %s\n",options[choice].description,options[choice].comments);
          for ( temp = options[choice].choicemin; temp < options[choice].choicemax+1; temp++)
            LogScreenRaw("  %2d) %s\n", (int) temp,options[choice].choicelist[temp]);
          LogScreen("\nDefault Setting: %s\nCurrent Setting: %s\nNew Setting --> ",
            options[choice].defaultsetting,
            (char *)options[choice].thevariable);
          }
        }
      else if (options[choice].type==2)
        {
        if (options[choice].choicelist == NULL)
          {
          LogScreenRaw("\n%s %s\nDefault Setting: %s\nCurrent Setting: %li\nNew Setting --> ",
                options[choice].description, options[choice].comments,
                options[choice].defaultsetting, (long)*(s32 *)options[choice].thevariable);
          }
        else 
          {
          LogScreenRaw("\n%s %s\n",options[choice].description,options[choice].comments);
          for ( temp = options[choice].choicemin; temp < options[choice].choicemax+1; temp++)
            LogScreenRaw("  %2d) %s\n", (int) temp,options[choice].choicelist[temp]);
          LogScreenRaw("\nDefault Setting: %s\nCurrent Setting: %s\nNew Setting --> ",
            options[choice].choicelist[atoi(options[choice].defaultsetting)],
            options[choice].choicelist[((long)*(s32 *)options[choice].thevariable)]);
          }
        }
      else if (options[choice].type==3)
        {
        sprintf(str, "%s", atoi(options[choice].defaultsetting)?"yes":"no");
        LogScreenRaw("\n%s %s\nDefault Setting: %s\nCurrent Setting: ",
               options[choice].description, options[choice].comments,
               str);
        sprintf(str, "%s", *(s32 *)options[choice].thevariable?"yes":"no");
        LogScreenRaw("%s\nNew Setting --> ",str);
        }

      fflush( stdin );
      fflush( stdout );
      fgets(parm, sizeof(parm), stdin);
      fflush( stdin );
      fflush( stdout );
      if (CheckExitRequestTrigger())
        choice = -2;
      else
        {
        for ( p = parm; *p; p++ )
          {
          if ( !isprint(*p) )
            {
            *p = 0;
            break;
            }
          }
          while (isspace(parm[strlen(parm)-1])) parm[strlen(parm)-1]=0; //strip trailing whitespace
          while (isspace(parm[0])) strcpy(parm,(parm+1));
        } // if ( !CheckExitRequestTrigger() )
      } //if ( choice >= 0 ) //if valid CONF_xxx option

    #ifndef OLDRESOLVE
    if ( strcmpi( keyproxy, "(auto)" ) == 0 )
      {
      strcpy( keyproxy, lkg_keyproxy ); //copy whatever they had back
      autofindkeyserver = 1;
      }
    if ( choice == CONF_KEYPROXY )
      {
      autofindkeyserver = 1; //ON unless user manually entered a dnet host
      if (!parm[0] || strcmpi(parm,"(auto)")==0 || strcmpi(parm,"auto")==0)
        {
        strcpy( parm, keyproxy ); //copy back what was in there before
        }
      else if (_IsHostnameDNetHost( parm ))
        autofindkeyserver = 0; //OFF because user manually entered a dnet host
      }
    #endif
    
    if ( choice >= 0 && (parm[0] || choice == CONF_LOGNAME ))
      {
      switch ( choice )
        {
        case CONF_ID:
          strncpy( id, parm, sizeof(id) - 1 );
          ValidateConfig();
          break;
        case CONF_THRESHOLDI:
          choice=atoi(parm);
          if (choice > 0) 
      {
      outthreshold[0]=inthreshold[0]=choice;
            ValidateConfig();
      }
          break;
        case CONF_THRESHOLDO:
          choice=atoi(parm);
          if (choice > 0)
      { 
      outthreshold[0]=choice;
            ValidateConfig();
      }
          break;
        case CONF_THRESHOLDI2:
          choice=atoi(parm);
          if (choice > 0)
      { 
      outthreshold[1]=inthreshold[1]=choice;
            ValidateConfig();
      }
          break;
        case CONF_THRESHOLDO2:
          choice=atoi(parm);
          if (choice > 0)
      { 
      outthreshold[1]=choice;
            ValidateConfig();
      }
          break;
        case CONF_COUNT:
          choice = atoi(parm);
          if (choice >= 0)
            blockcount = choice;
          break;
        case CONF_HOURS:
    if (atoi(parm)>=0)
      {
            minutes = (s32) (60. * atol(parm));
            if ( minutes < 0 ) minutes = 0;
            sprintf(hours,"%u.%02u", (unsigned)(minutes/60),
            (unsigned)(minutes%60)); //1.000000 hours looks silly
      }
          break;
        case CONF_TIMESLICE:        
          // *** To allows inis to be shared, don't use platform specific 
          // *** timeslice limits. Scale the generic 0-65536 one instead.
          choice = atoi(parm);   
          if (choice >= 1)
      timeslice = choice;
          break;
        case CONF_NICENESS:
          choice = atoi(parm);
          if ( choice >= 0 && choice <= 2 )
            niceness = choice;
          break;
        case CONF_LOGNAME:
          strncpy( logname, parm, sizeof(logname) - 1 );
          if (isstringblank(logname)) strcpy (logname,"none");
          break;
        case CONF_KEYPROXY:
          strncpy( keyproxy, parm, sizeof(keyproxy) - 1 );
          ValidateConfig();
          break;
        case CONF_KEYPORT:
          keyport = atoi(parm);
          CheckForcedKeyproxy();
          ValidateConfig();
          break;
        case CONF_HTTPPROXY:
          strncpy( httpproxy, parm, sizeof(httpproxy) - 1);
          ValidateConfig();
          break;
        case CONF_HTTPPORT:
          httpport = atoi(parm); break;
        case CONF_HTTPID:
          if ( strcmp(parm,".") == 0)
            {
            strcpy(httpid,"");
            }
          else if (uuehttpmode == 4)
            {  // socks4
            strcpy(httpid, parm);
            }
          else
            {             // http & socks5
            LogScreenRaw("Enter password--> ");
            fflush( stdin );
            fflush( stdout );
            fgets(parm2, sizeof(parm2), stdin);
            fflush( stdin );
            fflush( stdout );
            for ( p = parm2; *p; p++ )
              {
              if ( !isprint(*p) )
                {
                *p = 0;
                break;
                }
              }
            if (uuehttpmode == 5)   // socks5
              sprintf(httpid, "%s:%s", parm, parm2);
            else                    // http
              strcpy(httpid,Network::base64_encode(parm, parm2));
            }
          break;
        case CONF_UUEHTTPMODE:
          choice = atoi(parm);
          if ( choice < 0 || choice > 5 )
      break;
          uuehttpmode = choice;
          #ifndef OLDRESOLVE
          autofindkeyserver=1;
          #endif
          switch (uuehttpmode)
            {
            case 0:strcpy( keyproxy, "us.v27.distributed.net");//normal communications
                   keyport=2064;
                   break;
            case 1:strcpy( keyproxy, "us23.v27.distributed.net");//UUE mode (telnet)
                   keyport=23;
                   break;
            case 2:strcpy( keyproxy, "us80.v27.distributed.net");//HTTP mode
                   keyport=80;
                   break;
            case 3:strcpy( keyproxy, "us80.v27.distributed.net");//HTTP+UUE mode
                   keyport=80;
                   break;
            case 4:strcpy( keyproxy, "us.v27.distributed.net");//SOCKS4
                   keyport=2064;
                   break;
            case 5:strcpy( keyproxy, "us.v27.distributed.net");//SOCKS5
                   keyport=2064;
                   break;
            };
          if (uuehttpmode > 1)
            {
            options[CONF_HTTPPROXY].optionscreen=3;
            options[CONF_HTTPPORT].optionscreen=3;
            options[CONF_HTTPID].optionscreen=3;
            }
            else
            {
            options[CONF_HTTPPROXY].optionscreen=0;
            options[CONF_HTTPPORT].optionscreen=0;
            options[CONF_HTTPID].optionscreen=0;
            };
          break;
        case CONF_CPUTYPE:
          choice = atoi(parm);
          if (choice >= -1 && choice <= options[CONF_CPUTYPE].choicemax)
            cputype = choice;
          break;
        case CONF_MESSAGELEN:
          messagelen = atoi(parm);
          break; //mail options are validated by mail.cpp 1998/08/20 cyrus
        case CONF_SMTPPORT:
          smtpport = atoi(parm);
          break; //mail options are validated by mail.cpp 1998/08/20 cyrus
        case CONF_SMTPSRVR:
          strncpy( smtpsrvr, parm, sizeof(smtpsrvr) - 1 );
          break; //mail options are validated by mail.cpp 1998/08/20 cyrus
        case CONF_SMTPFROM:
          strncpy( smtpfrom, parm, sizeof(smtpfrom) - 1 );
          break; //mail options are validated by mail.cpp 1998/08/20 cyrus
        case CONF_SMTPDEST:
          strncpy( smtpdest, parm, sizeof(smtpdest) - 1 );
          break; //mail options are validated by mail.cpp 1998/08/20 cyrus
        case CONF_NUMCPU:
          numcpu = atoi(parm);
          break; //validation is done in SelectCore() 1998/06/21 cyrus
        case CONF_CHECKPOINT:
          #ifdef DONT_USE_PATHWORK
          strncpy( ini_checkpoint_file[0] , parm, sizeof(ini_checkpoint_file)/2 -1 );
          #else
          strncpy( checkpoint_file[0] , parm, sizeof(checkpoint_file[1]) -1 );
          #endif
          ValidateConfig();
          break;
        case CONF_CHECKPOINT2:
          #ifdef DONT_USE_PATHWORK
          strncpy( ini_checkpoint_file[1] , parm, sizeof(ini_checkpoint_file)/2 -1 );
          #else
          strncpy( checkpoint_file[1] , parm, sizeof(checkpoint_file[1]) -1 );
          #endif
          ValidateConfig();
          break;
        case CONF_PREFERREDBLOCKSIZE:
          choice = atoi(parm);
          if (choice > 0 && choice <=1000) 
      preferred_blocksize = choice;
          break;
        case CONF_PROCESSDES:
          choice = yesno(parm);
          if ((choice >= 0) && (choice <= 1))
             preferred_contest_id = choice;
          break;
        case CONF_QUIETMODE:
          choice=yesno(parm);
          if (choice >= 0) *(s32 *)options[CONF_QUIETMODE].thevariable=choice;
          break;
        case CONF_NOEXITFILECHECK:
          choice=yesno(parm);
          if (choice >= 0) *(s32 *)options[CONF_NOEXITFILECHECK].thevariable=choice;
          break;
        case CONF_PERCENTOFF:
          choice=yesno(parm);
          if (choice >= 0) *(s32 *)options[CONF_PERCENTOFF].thevariable=choice;
          break;
        case CONF_FREQUENT:
          choice=yesno(parm);
          if (choice >= 0) *(s32 *)options[CONF_FREQUENT].thevariable=choice;
          break;
        case CONF_NODISK:
          choice=yesno(parm);
          if (choice >= 0) *(s32 *)options[CONF_NODISK].thevariable=choice;
          break;
        case CONF_NOFALLBACK:
          choice=yesno(parm);
          if (choice >= 0) *(s32 *)options[CONF_NOFALLBACK].thevariable=choice;
          break;
        case CONF_CKTIME:
          choice=atoi(parm);
          if (choice > 0) *(s32 *)options[CONF_CKTIME].thevariable=choice;
          ValidateConfig();
          break;
        case CONF_NETTIMEOUT:
          choice=atoi(parm);
          if (choice > 0) *(s32 *)options[CONF_NETTIMEOUT].thevariable=choice;
          ValidateConfig();
          break;
        case CONF_EXITFILECHECKTIME:
          choice=atoi(parm);
          if (choice > 0) *(s32 *)options[CONF_EXITFILECHECKTIME].thevariable=choice;
          ValidateConfig();
          break;
        case CONF_OFFLINEMODE:
          choice=atoi(parm);
          if (choice >= 0 && choice <=2) 
      *(s32 *)options[CONF_OFFLINEMODE].thevariable=choice;
          break;
        case CONF_RC5IN:
          #ifdef DONT_USE_PATHWORK
          strncpy( ini_in_buffer_file[0] , parm, sizeof(ini_in_buffer_file)/2 -1 );
          #else
          strncpy( in_buffer_file[0] , parm, sizeof(in_buffer_file[0]) -1 );
          #endif
          ValidateConfig();
          break;
        case CONF_RC5OUT:
          #ifdef DONT_USE_PATHWORK
          strncpy( ini_out_buffer_file[0] , parm, sizeof(ini_out_buffer_file)/2 -1 );
          #else
          strncpy( out_buffer_file[0] , parm, sizeof(out_buffer_file[0]) -1 );
          #endif
          ValidateConfig();
          break;
        case CONF_DESIN:
          #ifdef DONT_USE_PATHWORK
          strncpy( ini_in_buffer_file[1] , parm, sizeof(ini_in_buffer_file)/2 -1 );
          #else
          strncpy( in_buffer_file[1] , parm, sizeof(in_buffer_file[1]) -1 );
          #endif
          ValidateConfig();
          break;
        case CONF_DESOUT:
          #ifdef DONT_USE_PATHWORK
          strncpy( ini_out_buffer_file[1] , parm, sizeof(ini_out_buffer_file)/2 -1 );
          #else
          strncpy( out_buffer_file[1] , parm, sizeof(out_buffer_file[1]) -1 );
          #endif
          ValidateConfig();
          break;
        case CONF_PAUSEFILE:
          #ifdef DONT_USE_PATHWORK
          strncpy( ini_pausefile, parm, sizeof(ini_pausefile) -1 );
          if (isstringblank(ini_pausefile)) strcpy (ini_pausefile,"none");
          #else
          strncpy( pausefile, parm, sizeof(pausefile) -1 );
          if (isstringblank(pausefile)) strcpy (pausefile,"none");
          #endif
          break;
        #ifdef MMX_BITSLICER
        case CONF_MMX:
          choice = yesno(parm);
    if (choice>=0) 
      usemmx = choice;
          break;
        #endif
        #ifdef LURK
        case CONF_LURKMODE:
          choice=atoi(parm);
          if (choice>=0 && choice<=2)
            {
            dialup.lurkmode=choice;
      if (choice!=1)
        connectoften=0;
            }
          break;
        case CONF_DIALWHENNEEDED:
          choice=yesno(parm);
          if (choice >= 0) *(s32 *)options[CONF_DIALWHENNEEDED].thevariable=choice;
          break;
        case CONF_CONNECTNAME:
          choice=atoi(parm);
          if ( ((choice > 0) || (parm[0]=='0')) &&
               (choice <= options[CONF_CONNECTNAME].choicemax) )
            {
            strcpy( (char *)options[CONF_CONNECTNAME].thevariable,
                     options[CONF_CONNECTNAME].choicelist[choice]);
            }
          else strncpy( (char *)options[CONF_CONNECTNAME].thevariable,
                        parm, sizeof(dialup.connectionname)-1);
          break;
        #endif
        default:
          break;
        }
      choice = 1; // continue with menu
      }
    } while ( choice >= 0 ); //while we have a valid CONF_xxx to work with
  return 0;
}
#endif

//----------------------------------------------------------------------------

s32 Client::Configure( void )
//A return of 1 indicates to save the changed configuration
//A return of -1 indicates to NOT save the changed configuration
{
#if defined(NOCONFIG)
  return 1;
#else
  s32 choice;
  char parm[128];
  s32 returnvalue=0;

  while (returnvalue == 0)
  {
    ConClear(); //in logstuff.cpp
    LogScreenRaw("Distributed.Net RC5/DES Client build "
              CLIENT_VERSIONSTRING " config menu\n" );
    LogScreenRaw("------------------------------------------------------------\n\n");
    LogScreenRaw(" 1) %s\n",menutable[0]);
    LogScreenRaw(" 2) %s\n",menutable[1]);
    LogScreenRaw(" 3) %s\n",menutable[2]);
    LogScreenRaw(" 4) %s\n",menutable[3]);
    LogScreenRaw(" 5) %s\n",menutable[4]);
    LogScreenRaw(" 6) %s\n\n",menutable[5]);
    LogScreenRaw(" 9) Discard settings and exit\n");
    LogScreenRaw(" 0) Save settings and exit\n\n");
    if (strcmpi(id,"rc5@distributed.net")==0)
      LogScreenRaw("*Note: You have not yet configured your e-mail address.\n"
            "       Please go to %s and configure it.\n",menutable[0]);
    LogScreenRaw("Choice --> ");

    fflush( stdin );
    fflush( stdout );
    fgets(parm, sizeof(parm), stdin);
    fflush( stdin );
    fflush( stdout );
    choice = atoi(parm);
    if (CheckExitRequestTrigger())
      choice = 9;

    switch (choice)
    {
      case 1: ConfigureGeneral(1);break;
      case 2: ConfigureGeneral(2);break;
      case 3: ConfigureGeneral(3);break;
      case 4: ConfigureGeneral(4);break;
      case 5: ConfigureGeneral(5);break;
      case 6: ConfigureGeneral(6);break;
      case 0: returnvalue=1;break; //Breaks and tells it to save
      case 9: returnvalue=-1;break; //Breaks and tells it NOT to save
    };
  if (CheckExitRequestTrigger())
    returnvalue=-1;
  }

  return returnvalue;
#endif
}

//----------------------------------------------------------------------------

#if !defined(NOCONFIG)
s32 Client::findmenuoption( s32 menu, s32 option)
    // Returns the id of the option that matches the menu and option
    // requested. Will return -1 if not found.
{
s32 returnvalue=-1;
s32 temp;

for (temp=0; temp < OPTION_COUNT; temp++)
  {
  if ((options[temp].optionscreen==menu) &&
      (options[temp].menuposition==option))

     returnvalue=temp;
  };

return returnvalue;
}
#endif

//----------------------------------------------------------------------------

#if !defined(NOCONFIG)
void Client::setupoptions( void )
// Sets all the pointers/etc for optionstruct options
{

options[CONF_ID].thevariable=(char *)(&id[0]);
options[CONF_THRESHOLDI].thevariable=&inthreshold[0];
options[CONF_THRESHOLDO].thevariable=&outthreshold[0];
options[CONF_THRESHOLDI2].thevariable=&inthreshold[1];
options[CONF_THRESHOLDO2].thevariable=&outthreshold[1];
options[CONF_COUNT].thevariable=&blockcount;
options[CONF_HOURS].thevariable=(char *)(&hours[0]);
#if !((CLIENT_OS==OS_MACOS) || (CLIENT_OS==OS_RISCOS) || (CLIENT_OS==OS_WIN16))
options[CONF_TIMESLICE].optionscreen=0;
#endif
options[CONF_TIMESLICE].thevariable=&timeslice;
options[CONF_NICENESS].thevariable=&niceness;
options[CONF_UUEHTTPMODE].thevariable=&uuehttpmode;
options[CONF_KEYPROXY].thevariable=(char *)(&keyproxy[0]);
options[CONF_KEYPORT].thevariable=&keyport;
options[CONF_HTTPPROXY].thevariable=(char *)(&httpproxy[0]);
options[CONF_HTTPPORT].thevariable=&httpport;
options[CONF_HTTPID].thevariable=(char *)(&httpid[0]);
options[CONF_MESSAGELEN].thevariable=&messagelen;
options[CONF_SMTPSRVR].thevariable=(char *)(&smtpsrvr[0]);
options[CONF_SMTPPORT].thevariable=&smtpport;
options[CONF_SMTPFROM].thevariable=(char *)(&smtpfrom[0]);
options[CONF_SMTPDEST].thevariable=(char *)(&smtpdest[0]);
options[CONF_NUMCPU].thevariable=&numcpu;

options[CONF_RANDOMPREFIX].thevariable=&randomprefix;
options[CONF_PREFERREDBLOCKSIZE].thevariable=&preferred_blocksize;
options[CONF_PROCESSDES].thevariable=&preferred_contest_id;
options[CONF_QUIETMODE].thevariable=&quietmode;
options[CONF_NOEXITFILECHECK].thevariable=&noexitfilecheck;
options[CONF_PERCENTOFF].thevariable=&percentprintingoff;
options[CONF_FREQUENT].optionscreen=0;
options[CONF_FREQUENT].thevariable=&connectoften;
options[CONF_NODISK].thevariable=&nodiskbuffers;
options[CONF_NOFALLBACK].thevariable=&nofallback;
options[CONF_CKTIME].thevariable=&checkpoint_min;
options[CONF_NETTIMEOUT].thevariable=&nettimeout;
options[CONF_EXITFILECHECKTIME].thevariable=&exitfilechecktime;
options[CONF_OFFLINEMODE].thevariable=&offlinemode;

options[CONF_LOGNAME].thevariable=(char *)(&logname[0]);
#ifdef DONT_USE_PATHWORK
options[CONF_CHECKPOINT].thevariable=&ini_checkpoint_file[0];
options[CONF_CHECKPOINT2].thevariable=&ini_checkpoint_file[1];
options[CONF_RC5IN].thevariable=&ini_in_buffer_file[0];
options[CONF_RC5OUT].thevariable=&ini_out_buffer_file[0];
options[CONF_DESIN].thevariable=&ini_in_buffer_file[1];
options[CONF_DESOUT].thevariable=&ini_out_buffer_file[1];
options[CONF_PAUSEFILE].thevariable=&ini_pausefile;
#else
options[CONF_CHECKPOINT].thevariable=(char *)(&checkpoint_file[0][0]);
options[CONF_CHECKPOINT2].thevariable=(char *)(&checkpoint_file[1][0]);
options[CONF_RC5IN].thevariable=(char *)(&in_buffer_file[0][0]);
options[CONF_RC5OUT].thevariable=(char *)(&out_buffer_file[0][0]);
options[CONF_DESIN].thevariable=(char *)(&in_buffer_file[1][0]);
options[CONF_DESOUT].thevariable=(char *)(&out_buffer_file[1][0]);
options[CONF_PAUSEFILE].thevariable=(char *)(&pausefile[0]);
#endif

#ifdef MMX_BITSLICER
options[CONF_MMX].thevariable=&usemmx;
#endif

options[CONF_CPUTYPE].optionscreen=0;
options[CONF_CPUTYPE].choicemax=0;
const char *corename = GetCoreNameFromCoreType(0);
if (corename && *corename)
  {
  static const char *cputypetable[10];
  unsigned int tablesize = 2;
  cputypetable[0]="Autodetect";
  cputypetable[1]=corename;
  do
    {
    corename = GetCoreNameFromCoreType(tablesize-1);
    if (!corename || !*corename)
      break;
    cputypetable[tablesize++]=corename;
    } while (tablesize<((sizeof(cputypetable)/sizeof(cputypetable[0]))));
  options[CONF_CPUTYPE].name="cputype";
  options[CONF_CPUTYPE].description=CFGTXT("Processor type");
  options[CONF_CPUTYPE].defaultsetting="-1";
  options[CONF_CPUTYPE].comments=CFGTXT("\n");
  options[CONF_CPUTYPE].optionscreen=4;
  options[CONF_CPUTYPE].type=2;
  options[CONF_CPUTYPE].menuposition=2;
  options[CONF_CPUTYPE].thevariable=&cputype;
  options[CONF_CPUTYPE].choicelist=&cputypetable[1];
  options[CONF_CPUTYPE].choicemin=-1;
  options[CONF_CPUTYPE].choicemax=tablesize-2;
  }


#if (!defined(LURK))
options[CONF_LURKMODE].optionscreen=0;
options[CONF_DIALWHENNEEDED].optionscreen=0;
options[CONF_CONNECTNAME].optionscreen=0;
#else
options[CONF_LURKMODE].thevariable=&dialup.lurkmode;
options[CONF_DIALWHENNEEDED].thevariable=&dialup.dialwhenneeded;
options[CONF_CONNECTNAME].thevariable=&dialup.connectionname;

char *connectnames = dialup.GetEntryList(&options[CONF_CONNECTNAME].choicemax);

if (options[CONF_CONNECTNAME].choicemax < 1)
  {
  options[CONF_CONNECTNAME].optionscreen=0;
  options[CONF_CONNECTNAME].choicelist=NULL;
  }
else
  {
  static char *connectnamelist[10];
  if (options[CONF_CONNECTNAME].choicemax>10)
    options[CONF_CONNECTNAME].choicemax=10;
  for (int i=0;i<((int)(options[CONF_CONNECTNAME].choicemax));i++)
    connectnamelist[i]=&(connectnames[i*60]);
  options[CONF_CONNECTNAME].choicelist=(const char **)(&connectnamelist[0]);
  options[CONF_CONNECTNAME].choicemin=0;
  options[CONF_CONNECTNAME].choicemax--;
  };
#endif

if (messagelen != 0)
  {
  options[CONF_SMTPSRVR].optionscreen=2;
  options[CONF_SMTPPORT].optionscreen=2;
  options[CONF_SMTPDEST].optionscreen=2;
  options[CONF_SMTPFROM].optionscreen=2;
  }
  else
  {
  options[CONF_SMTPSRVR].optionscreen=0;
  options[CONF_SMTPPORT].optionscreen=0;
  options[CONF_SMTPDEST].optionscreen=0;
  options[CONF_SMTPFROM].optionscreen=0;
  };

if (uuehttpmode > 1)
  {
  options[CONF_HTTPPROXY].optionscreen=3;
  options[CONF_HTTPPORT].optionscreen=3;
  options[CONF_HTTPID].optionscreen=3;
  }
  else
  {
  options[CONF_HTTPPROXY].optionscreen=0;
  options[CONF_HTTPPORT].optionscreen=0;
  options[CONF_HTTPID].optionscreen=0;
  };
}
#endif

//----------------------------------------------------------------------------

s32 Client::ReadConfig(void)  //DO NOT PRINT TO SCREEN (or whatever) FROM HERE
{
  IniSection ini;
  s32 inierror, tempconfig;
  char buffer[64];
  char *p;

  #ifdef DONT_USE_PATHWORK
  inierror = ini.ReadIniFile( inifilename );
  #else
  inierror = ini.ReadIniFile( GetFullPathForFilename( inifilename ) );
  #endif

#define INIGETKEY(key) (ini.getkey(OPTION_SECTION, options[key].name, options[key].defaultsetting)[0])

  INIGETKEY(CONF_ID).copyto(id, sizeof(id));
  INIGETKEY(CONF_THRESHOLDI).copyto(buffer, sizeof(buffer));
  p = strchr( buffer, ':' );
  if (p == NULL) {
    outthreshold[0]=inthreshold[0]=atoi(buffer);
  } else {
    outthreshold[0]=atoi(p+1);
    *p=0;
    inthreshold[0]=atoi(buffer);
  }
  INIGETKEY(CONF_THRESHOLDI2).copyto(buffer, sizeof(buffer));
  p = strchr( buffer, ':' );
  if (p == NULL) {
    outthreshold[1]=inthreshold[1]=atoi(buffer);
  } else {
    outthreshold[1]=atoi(p+1);
    *p=0;
    inthreshold[1]=atoi(buffer);
  }
  blockcount = INIGETKEY(CONF_COUNT);
  INIGETKEY(CONF_HOURS).copyto(hours, sizeof(hours));
  minutes = (s32) (atol(hours) * 60.);
  timeslice = INIGETKEY(CONF_TIMESLICE);
  niceness = INIGETKEY(CONF_NICENESS);

  INIGETKEY(CONF_KEYPROXY).copyto(keyproxy, sizeof(keyproxy));
  tempconfig=ini.getkey("networking", "autofindkeyserver", "1")[0];
  autofindkeyserver = (tempconfig)?(1):(0);
  #ifndef OLDRESOLVE
  if (strcmpi( keyproxy, "rc5proxy.distributed.net" )==0) //old generic name
    {                                                  //now CNAME to us.v27
    strcpy( keyproxy, "us.v27.distributed.net" ); 
    autofindkeyserver = 1; //let Resolve() get a better hostname.
    }
  #endif

  keyport = INIGETKEY(CONF_KEYPORT);
  INIGETKEY(CONF_HTTPPROXY).copyto(httpproxy, sizeof(httpproxy));
  httpport = INIGETKEY(CONF_HTTPPORT);
  uuehttpmode = INIGETKEY(CONF_UUEHTTPMODE);
  INIGETKEY(CONF_HTTPID).copyto(httpid, sizeof(httpid));
  cputype = INIGETKEY(CONF_CPUTYPE);
  messagelen = INIGETKEY(CONF_MESSAGELEN);
  smtpport = INIGETKEY(CONF_SMTPPORT);
  INIGETKEY(CONF_SMTPSRVR).copyto(smtpsrvr, sizeof(smtpsrvr));
  INIGETKEY(CONF_SMTPFROM).copyto(smtpfrom, sizeof(smtpfrom));
  INIGETKEY(CONF_SMTPDEST).copyto(smtpdest, sizeof(smtpdest));
  numcpu = INIGETKEY(CONF_NUMCPU);

  randomprefix = INIGETKEY(CONF_RANDOMPREFIX);
  preferred_contest_id = INIGETKEY(CONF_PROCESSDES);
  preferred_blocksize = INIGETKEY(CONF_PREFERREDBLOCKSIZE);

  tempconfig=ini.getkey(OPTION_SECTION, "runbuffers", "0")[0];
  if (tempconfig) {
    offlinemode=2;
  } else {
    tempconfig=ini.getkey(OPTION_SECTION, "runoffline", "0")[0];
    if (tempconfig) offlinemode=1;
  }
  tempconfig=ini.getkey(OPTION_SECTION, "percentoff", "0")[0];
  if (tempconfig) percentprintingoff=1;
  tempconfig=ini.getkey(OPTION_SECTION, "frequent", "0")[0];
  if (tempconfig) connectoften=1;
  tempconfig=ini.getkey(OPTION_SECTION, "nodisk", "0")[0];
  if (tempconfig) nodiskbuffers=1;
  tempconfig=ini.getkey(OPTION_SECTION, "quiet", "0")[0];
  if (tempconfig) quietmode=1;
  tempconfig=ini.getkey(OPTION_SECTION, "nofallback", "0")[0];
  if (tempconfig) nofallback=1;
  tempconfig=ini.getkey(OPTION_SECTION, "cktime", "0")[0];
  if (tempconfig) checkpoint_min=max(2,tempconfig);
  tempconfig=ini.getkey(OPTION_SECTION, "nettimeout", "60")[0];
  if (tempconfig) nettimeout=min(300,max(5,nettimeout));
  tempconfig=ini.getkey(OPTION_SECTION, "noexitfilecheck", "0")[0];
  if (tempconfig) noexitfilecheck=1;
  tempconfig=ini.getkey(OPTION_SECTION, "exitfilechecktime", "30")[0];
  if (tempconfig) exitfilechecktime=max(tempconfig,1);
#if ((CLIENT_OS==OS_WIN32) && (!defined(WINNTSERVICE)))
  tempconfig=ini.getkey(OPTION_SECTION, "win95hidden", "0")[0]; //obsolete
  if (tempconfig) runhidden=1;
  tempconfig=ini.getkey(OPTION_SECTION, "runhidden", "0")[0]; //now known as...
  if (tempconfig) runhidden=1;
#endif
#if defined(LURK)
  tempconfig=ini.getkey(OPTION_SECTION, "lurk", "0")[0];
  if (tempconfig) dialup.lurkmode=1;
  tempconfig=ini.getkey(OPTION_SECTION, "lurkonly", "0")[0];
  if (tempconfig) {dialup.lurkmode=2; connectoften=0;}
  tempconfig=ini.getkey(OPTION_SECTION, "dialwhenneeded", "0")[0];
  if (tempconfig) dialup.dialwhenneeded=1;
  INIGETKEY(CONF_CONNECTNAME).copyto(dialup.connectionname,sizeof(dialup.connectionname));
#endif

INIGETKEY(CONF_LOGNAME).copyto(logname, sizeof(logname));
#ifdef DONT_USE_PATHWORK
  INIGETKEY(CONF_CHECKPOINT).copyto(ini_checkpoint_file[0], sizeof(ini_checkpoint_file)/2);
  INIGETKEY(CONF_CHECKPOINT2).copyto(ini_checkpoint_file[1], sizeof(ini_checkpoint_file)/2);
  ini.getkey(OPTION_SECTION,"in",ini_in_buffer_file[0])[0].copyto(ini_in_buffer_file[0],sizeof(ini_in_buffer_file)/2);
  ini.getkey(OPTION_SECTION,"out",ini_out_buffer_file[0])[0].copyto(ini_out_buffer_file[0],sizeof(ini_out_buffer_file)/2);
  ini.getkey(OPTION_SECTION,"in2",ini_in_buffer_file[1])[0].copyto(ini_in_buffer_file[1],sizeof(ini_in_buffer_file)/2);
  ini.getkey(OPTION_SECTION,"out2",ini_out_buffer_file[1])[0].copyto(ini_out_buffer_file[1],sizeof(ini_out_buffer_file)/2);
  ini.getkey(OPTION_SECTION,"pausefile",ini_pausefile)[0].copyto(ini_pausefile,sizeof(ini_pausefile));
#else
  INIGETKEY(CONF_CHECKPOINT).copyto(checkpoint_file[0], sizeof(checkpoint_file[0]));
  INIGETKEY(CONF_CHECKPOINT2).copyto(checkpoint_file[1], sizeof(checkpoint_file[1]));
  ini.getkey(OPTION_SECTION,"in",in_buffer_file[0])[0].copyto(in_buffer_file[0],sizeof(in_buffer_file[0]));
  ini.getkey(OPTION_SECTION,"out",out_buffer_file[0])[0].copyto(out_buffer_file[0],sizeof(out_buffer_file[0]));
  ini.getkey(OPTION_SECTION,"in2",in_buffer_file[1])[0].copyto(in_buffer_file[1],sizeof(in_buffer_file[1]));
  ini.getkey(OPTION_SECTION,"out2",out_buffer_file[1])[0].copyto(out_buffer_file[1],sizeof(out_buffer_file[1]));
  ini.getkey(OPTION_SECTION,"pausefile",pausefile)[0].copyto(pausefile,sizeof(pausefile));
#endif
  tempconfig=ini.getkey(OPTION_SECTION, "contestdone", "0")[0];
  if (tempconfig) contestdone[0]=1;
  tempconfig=ini.getkey(OPTION_SECTION, "contestdone2", "0")[0];
  if (tempconfig) contestdone[1]=1;
#ifdef MMX_BITSLICER
  usemmx=INIGETKEY(CONF_MMX);
#endif

#undef INIGETKEY
#if defined(NEEDVIRTUALMETHODS)
  InternalReadConfig(ini);
#endif

  ValidateConfig();

  return( inierror ? -1 : 0 );
}

// --------------------------------------------------------------------------

void Client::ValidateConfig( void ) //DO NOT PRINT TO SCREEN HERE!
{
  killwhitespace(id);
  killwhitespace(keyproxy);
  killwhitespace(httpproxy);
  killwhitespace(smtpsrvr);

  if ( inthreshold[0] < 1   ) inthreshold[0] = 1;
  if ( inthreshold[0] > 1000 ) inthreshold[0] = 1000;
  if ( outthreshold[0] < 1   ) outthreshold[0] = 1;
  if ( outthreshold[0] > 1000 ) outthreshold[0] = 1000;
  if ( outthreshold[0] > inthreshold[0] ) outthreshold[0]=inthreshold[0];
  if ( inthreshold[1] < 1   ) inthreshold[1] = 1;
  if ( inthreshold[1] > 1000 ) inthreshold[1] = 1000;
  if ( outthreshold[1] < 1   ) outthreshold[1] = 1;
  if ( outthreshold[1] > 1000 ) outthreshold[1] = 1000;
  if ( outthreshold[1] > inthreshold[1] ) outthreshold[1]=inthreshold[1];
  if ( blockcount < 0 ) blockcount = 0;
#if (CLIENT_OS == OS_RISCOS)
  if ( timeslice < 1 ) timeslice = 2048;
#else
  if ( timeslice < 1 ) timeslice = 65536;
#endif
  if ( niceness < 0 || niceness > 2 ) niceness = 0;
  if ( uuehttpmode < 0 || uuehttpmode > 5 ) uuehttpmode = 0;
  if ( randomprefix <0  ) randomprefix=100;
  if ( randomprefix >255) randomprefix=100;
  if (smtpport < 0) smtpport=25;
  if (smtpport > 65535L) smtpport=25;

  if (( preferred_contest_id < 0 ) || ( preferred_contest_id > 1 )) preferred_contest_id = 1;
  if (preferred_blocksize < 28) preferred_blocksize = 28;
  if (preferred_blocksize > 31) preferred_blocksize = 31;
  if ( minutes < 0 ) minutes=0;
  if ( blockcount < 0 ) blockcount=0;
  if (checkpoint_min < 2) checkpoint_min=2;
    else if (checkpoint_min > 30) checkpoint_min=30;
  if (exitfilechecktime < 5) exitfilechecktime=5;
    else if (exitfilechecktime > 600) exitfilechecktime=600;
  nettimeout=min(300,max(5,nettimeout));

if (isstringblank(logname))
  strcpy (logname,"none");

#ifndef DONT_USE_PATHWORK  //ie use it

  if (isstringblank(in_buffer_file[0]))
    strcpy(in_buffer_file[0],"buff-in" EXTN_SEP "rc5");
  if (isstringblank(out_buffer_file[0]))
    strcpy(out_buffer_file[0],"buff-out" EXTN_SEP "rc5");
  if (isstringblank(in_buffer_file[1]))
    strcpy(in_buffer_file[1],"buff-in" EXTN_SEP "des");
  if (isstringblank(out_buffer_file[1]))
    strcpy(out_buffer_file[1],"buff-out" EXTN_SEP "des");
  if (isstringblank(pausefile))
    strcpy(pausefile,"none");
  if (isstringblank(checkpoint_file[0]))
    strcpy(checkpoint_file[0],"none");
  if (isstringblank(checkpoint_file[1]))
    strcpy(checkpoint_file[1],"none");

#else //ie don't use pathwork

  if (isstringblank(ini_in_buffer_file[0]))
    strcpy(ini_in_buffer_file[0],"buff-in" EXTN_SEP "rc5");
  if (isstringblank(ini_out_buffer_file[0]))
    strcpy(ini_out_buffer_file[0],"buff-out" EXTN_SEP "rc5");
  if (isstringblank(ini_in_buffer_file[1]))
    strcpy(ini_in_buffer_file[1],"buff-in" EXTN_SEP "des");
  if (isstringblank(ini_out_buffer_file[1]))
    strcpy(ini_out_buffer_file[1],"buff-out" EXTN_SEP "des");
  if (isstringblank(ini_pausefile)) strcpy(ini_pausefile,"none");
  if (isstringblank(ini_checkpoint_file[0])) strcpy(ini_checkpoint_file[0],"none");
  if (isstringblank(ini_checkpoint_file[1])) strcpy(ini_checkpoint_file[1],"none");

  // now, add path of exe to filenames if path isn't specified

  strcpy(exit_flag_file,InternalGetLocalFilename(ini_exit_flag_file));
  strcpy(in_buffer_file[0],InternalGetLocalFilename(ini_in_buffer_file[0]));
  strcpy(out_buffer_file[0],InternalGetLocalFilename(ini_out_buffer_file[0]));
  strcpy(in_buffer_file[1],InternalGetLocalFilename(ini_in_buffer_file[1]));
  strcpy(out_buffer_file[1],InternalGetLocalFilename(ini_out_buffer_file[1]));
  strcpy(pausefile,InternalGetLocalFilename(ini_pausefile));
  strcpy(checkpoint_file[0],InternalGetLocalFilename(ini_checkpoint_file[0]));
  strcpy(checkpoint_file[1],InternalGetLocalFilename(ini_checkpoint_file[1]));

#endif //DONT_USE_PATHWORK

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

s32 Client::WriteConfig(void)  
{
  IniSection ini;
  #ifdef DONT_USE_PATHWORK
  if ( ini.ReadIniFile( inifilename ) )
    return WriteFullConfig();
  #else
  if ( ini.ReadIniFile( GetFullPathForFilename( inifilename ) ) )
    return WriteFullConfig();
  #endif
    
  #if defined(NEEDVIRTUALMETHODS)
    InternalWriteConfig(ini);
  #endif

  IniRecord *tempptr;
  while ((tempptr = ini.findfirst(OPTION_SECTION, "win95hidden"))!=NULL)
    tempptr->key = "runhidden";
  while ((tempptr = ini.findfirst(OPTION_SECTION, "os2hidden"))!=NULL)
    tempptr->key = "runhidden";

  #ifdef DONT_USE_PATHWORK
  return( ini.WriteIniFile(inifilename) ? -1 : 0 );
  #else
  return( ini.WriteIniFile( GetFullPathForFilename( inifilename ) ) ? -1 : 0 );
  #endif
}

// --------------------------------------------------------------------------

s32 Client::WriteFullConfig(void) //construct a brand-spanking-new config
{
  IniSection ini;
  char buffer[64];

#define INISETKEY(key, value) ini.setrecord(OPTION_SECTION, options[key].name, IniString(value))

  INISETKEY( CONF_ID, id );
  sprintf(buffer,"%d:%d",(int)inthreshold[0],(int)outthreshold[0]);
  INISETKEY( CONF_THRESHOLDI, buffer );
  sprintf(buffer,"%d:%d",(int)inthreshold[1],(int)outthreshold[1]);
  INISETKEY( CONF_THRESHOLDI2, buffer );
  INISETKEY( CONF_COUNT, blockcount );
  sprintf(hours,"%u.%02u", (unsigned)(minutes/60), (unsigned)(minutes%60)); 
  INISETKEY( CONF_HOURS, hours );
  INISETKEY( CONF_TIMESLICE, timeslice );
  INISETKEY( CONF_NICENESS, niceness );

  #ifndef OLDRESOLVE
  if (autofindkeyserver && _IsHostnameDNetHost( keyproxy ))
    {
    //INISETKEY( CONF_KEYPROXY, "rc5proxy.distributed.net" );
    IniRecord *tempptr = ini.findfirst(OPTION_SECTION, "keyproxy");
    if (tempptr) tempptr->values.Erase();
    tempptr = ini.findfirst( "networking", "autofindkeyserver");
    if (tempptr) tempptr->values.Erase();
    }
  else
  #endif
    {
    INISETKEY( CONF_KEYPROXY, keyproxy );
    ini.setrecord("networking", "autofindkeyserver", IniString(autofindkeyserver?"1":"0"));
    }

  INISETKEY( CONF_KEYPORT, keyport );
  INISETKEY( CONF_HTTPPROXY, httpproxy );
  INISETKEY( CONF_HTTPPORT, httpport );
  INISETKEY( CONF_UUEHTTPMODE, uuehttpmode );
  INISETKEY( CONF_HTTPID, httpid);
  INISETKEY( CONF_CPUTYPE, cputype );
  INISETKEY( CONF_MESSAGELEN, messagelen );
  INISETKEY( CONF_SMTPSRVR, smtpsrvr );
  INISETKEY( CONF_SMTPPORT, smtpport );
  INISETKEY( CONF_SMTPFROM, smtpfrom );
  INISETKEY( CONF_SMTPDEST, smtpdest );
  INISETKEY( CONF_NUMCPU, numcpu );
  INISETKEY( CONF_RANDOMPREFIX, randomprefix );
  INISETKEY( CONF_PREFERREDBLOCKSIZE, preferred_blocksize );
  INISETKEY( CONF_PROCESSDES, (s32)(preferred_contest_id) );
  INISETKEY( CONF_QUIETMODE, (quietmode?"1":"0") );
  INISETKEY( CONF_NOEXITFILECHECK, noexitfilecheck );
  INISETKEY( CONF_PERCENTOFF, percentprintingoff );
  INISETKEY( CONF_FREQUENT, connectoften );
  INISETKEY( CONF_NODISK, nodiskbuffers );
  INISETKEY( CONF_NOFALLBACK, nofallback );
  INISETKEY( CONF_CKTIME, checkpoint_min );
  INISETKEY( CONF_NETTIMEOUT, nettimeout );
  INISETKEY( CONF_EXITFILECHECKTIME, exitfilechecktime );

#ifdef LURK
  INISETKEY( CONF_DIALWHENNEEDED, dialup.dialwhenneeded);
  INISETKEY( CONF_CONNECTNAME, dialup.connectionname);
#endif

INISETKEY( CONF_LOGNAME, logname );
#ifdef DONT_USE_PATHWORK
  INISETKEY( CONF_CHECKPOINT, ini_checkpoint_file[0] );
  INISETKEY( CONF_CHECKPOINT2, ini_checkpoint_file[1] );
  INISETKEY( CONF_RC5IN, ini_in_buffer_file[0]);
  INISETKEY( CONF_RC5OUT, ini_out_buffer_file[0]);
  INISETKEY( CONF_DESIN, ini_in_buffer_file[1]);
  INISETKEY( CONF_DESOUT, ini_out_buffer_file[1]);
  INISETKEY( CONF_PAUSEFILE, ini_pausefile);
#else
  INISETKEY( CONF_CHECKPOINT, checkpoint_file[0] );
  INISETKEY( CONF_CHECKPOINT2, checkpoint_file[1] );
  INISETKEY( CONF_RC5IN, in_buffer_file[0]);
  INISETKEY( CONF_RC5OUT, out_buffer_file[0]);
  INISETKEY( CONF_DESIN, in_buffer_file[1]);
  INISETKEY( CONF_DESOUT, out_buffer_file[1]);
  INISETKEY( CONF_PAUSEFILE, pausefile);
#endif

#ifdef MMX_BITSLICER
  INISETKEY( CONF_MMX, (s32)(usemmx) );
#endif

  if (offlinemode == 0)
    {
    IniRecord *tempptr;
    tempptr = ini.findfirst(OPTION_SECTION, "runbuffers");
    if (tempptr) tempptr->values.Erase();
    tempptr=NULL;
    tempptr = ini.findfirst(OPTION_SECTION, "runoffline");
    if (tempptr) tempptr->values.Erase();
    }
  else if (offlinemode == 1)
    {
    IniRecord *tempptr;
    s32 tempvalue;
    tempptr = ini.findfirst(OPTION_SECTION, "runbuffers");
    if (tempptr) tempptr->values.Erase();
    tempvalue=1;
    ini.setrecord(OPTION_SECTION, "runoffline", IniString(tempvalue));
    }
  else if (offlinemode == 2)
    {
    IniRecord *tempptr;
    s32 tempvalue;
    tempptr = ini.findfirst(OPTION_SECTION, "runoffline");
    if (tempptr) tempptr->values.Erase();
    tempvalue=1;
    ini.setrecord(OPTION_SECTION, "runbuffers", IniString(tempvalue));
    };

#undef INISETKEY

  ini.setrecord(OPTION_SECTION, "contestdone",  IniString(contestdone[0]));
  ini.setrecord(OPTION_SECTION, "contestdone2", IniString(contestdone[1]));

#if defined(LURK)

  if (dialup.lurkmode==0)
    {
    IniRecord *tempptr;
    tempptr = ini.findfirst(OPTION_SECTION, "lurk");
    if (tempptr) tempptr->values.Erase();
    tempptr = ini.findfirst(OPTION_SECTION, "lurkonly");
    if (tempptr) tempptr->values.Erase();
    }
  else if (dialup.lurkmode==1)
    {
    IniRecord *tempptr;
    s32 tempvalue=1;
    tempptr = ini.findfirst(OPTION_SECTION, "lurkonly");
    if (tempptr) tempptr->values.Erase();
    ini.setrecord(OPTION_SECTION, "lurk",  IniString(tempvalue));
    }
  else if (dialup.lurkmode==2)
    {
    IniRecord *tempptr;
    s32 tempvalue=1;
    tempptr = ini.findfirst(OPTION_SECTION, "lurk");
    if (tempptr) tempptr->values.Erase();
    ini.setrecord(OPTION_SECTION, "lurkonly",  IniString(tempvalue));
    };
#endif

#define INIFIND(key) ini.findfirst(OPTION_SECTION, options[key].name)
  if (uuehttpmode <= 1)
  {
    // wipe out httpproxy and httpport & httpid
    IniRecord *ptr;
    ptr = INIFIND( CONF_HTTPPROXY );
    if (ptr) ptr->values.Erase();
    ptr = INIFIND( CONF_HTTPPORT );
    if (ptr) ptr->values.Erase();
    ptr = INIFIND( CONF_HTTPID );
    if (ptr) ptr->values.Erase();
  }
#undef INIFIND

#if defined(NEEDVIRTUALMETHODS)
  InternalWriteConfig(ini);
#endif

  #ifdef DONT_USE_PATHWORK
  return( ini.WriteIniFile(inifilename) ? -1 : 0 );
  #else
  return( ini.WriteIniFile( GetFullPathForFilename( inifilename ) ) ? -1 : 0 );
  #endif
}

// --------------------------------------------------------------------------

s32 Client::WriteContestandPrefixConfig(void)
    // returns -1 on error, 0 otherwise
    // only writes contestdone and randomprefix .ini entries
{
  IniSection ini;

  if (nodiskbuffers)
    return 0;

  ini.ReadIniFile( GetFullPathForFilename( inifilename ) );

  #define INISETKEY(key, value) ini.setrecord(OPTION_SECTION, options[key].name, IniString(value))
  INISETKEY( CONF_RANDOMPREFIX, randomprefix );
  #undef INISETKEY

  ini.setrecord(OPTION_SECTION, "contestdone",  IniString(contestdone[0]));
  ini.setrecord(OPTION_SECTION, "contestdone2", IniString(contestdone[1]));

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
      strcmpi(dot, ".distributed.net") == 0))
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
      strcmpi(dot, ".distributed.net") == 0))
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

#ifdef DONT_USE_PATHWORK
const char *Client::InternalGetLocalFilename(const char *filename)
//If there is no path given, add on the path of the client's executAble
{
  if (strcmpi(filename,"none") != 0)
    {
    #if (CLIENT_OS == OS_NETWARE)           //thanks, but no thanks.
    #elif (CLIENT_OS == OS_DOS)   //nothin' - this code doesn't work for DOS
    #elif (CLIENT_OS == OS_OS2) //doesn't work for OS/2 either
    #elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN32S) || (CLIENT_OS == OS_WIN16)
      {
      char drive[_MAX_DRIVE];
      char fname[_MAX_FNAME];
      char dir[_MAX_DIR];
      char ext[_MAX_EXT];
      _splitpath( filename, drive, dir, fname, ext );
      if ((strlen(drive)==0) && (strlen(dir)==0))
        {
        static char buffer[200];
        ::GetModuleFileName(NULL, buffer, sizeof(buffer));
        char *slash = strrchr(buffer, '\\');
        if (slash == NULL) strcpy(buffer, filename);
        else strcpy(slash + 1, filename);
        return buffer;
        }
      }
    #elif (CLIENT_OS == OS_RISCOS)
      return riscos_localise_filename(filename);
    #elif (defined( DONT_USE_PATHWORK ))
      {
      if ( (strrchr(filename,PATH_SEP_C) == NULL) && (strrchr(inifilename,PATH_SEP_C) != NULL) )
        //check that we need to add a path, and that we have a path to add
        {
        char buffer[200];
        strcpy( buffer,inifilename );
        char *slash = strrchr(buffer, PATH_SEP_C);
        *(slash+1) = 0; // we have to add path info in!

        // no path already here, add it
        strcat(buffer,filename);
        return buffer;
        }
      }
    #endif
    } //if (strcmpi(filename,"none") != 0)
  return filename;
}
#endif

// --------------------------------------------------------------------------
