// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: cliconfig.cpp,v $
// Revision 1.193  1998/11/09 14:32:16  chrisb
// Set the RISC OS CPU priority message, and fixed a typo of CLIEN_OS == WIN32S.
//
// Revision 1.192  1998/11/06 02:21:38  cyp
// Fixed a missing return type for findmenuoption().
//
// Revision 1.191  1998/11/04 21:27:52  cyp
// Removed redundant ::hidden option. ::quiet was always equal to ::hidden.
//
// Revision 1.190  1998/10/30 04:38:33  foxyloxy
// If we don't have an .ini file, we don't need to read it.
//
// Revision 1.189  1998/10/26 02:48:40  cyp
// Many fixes
//
// Revision 1.188  1998/10/20 14:01:24  remi
// "rc5@distributed.net" is for 'id' not 'logname' in ValidateConfig().
// Some indentation.
//
// Revision 1.187  1998/10/19 13:10:19  cyp
// Changed the description for 'id' to better reflect what purpose the
// distributed.net ID serves.
//
// Revision 1.186  1998/10/19 12:29:45  cyp
// completed implementation of 'priority'.
//
// Revision 1.185  1998/10/09 22:23:07  remi
// Fixed nettimeout .ini setting not honored. Some indentation.
//
// Revision 1.184  1998/10/04 11:43:24  remi
// Print an error message if the luser do "./rc5des -config | somepager".
// Wrapped Log comments.
//
// Revision 1.183  1998/10/03 23:46:52  remi
// Use 'usemmx' .ini setting if any MMX core is compiled in.
//
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
// Changed WriteContestAndPrefixConfig so that it would not attempt to
// do so if nodiskbuffers was specified (aiming towards complete
// diskless operation aside from the inital .ini read)
//
// Revision 1.172  1998/08/14 00:04:48  silby
// Changes for rc5 mmx core integration.
//
// Revision 1.171  1998/08/10 23:02:22  cyruspatel
// xxxTrigger and pausefilefound flags are now wrapped in trigger.cpp 
//
// Revision 1.170  1998/08/07 05:28:47  silby
// Changed lurk so that win32 users can now easily select the
// connection to use for dial on demand.
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
// Major reorganization: Log,LogScreen, and LogScreenf are now in
// logging.cpp, and are global functions - client.h #includes
// logging.h, which is all you need to use those functions.  Lurk
// handling has been added into the Lurk class, which resides in
// lurk.cpp, and is auto-included by client.h if lurk is defined as
// well. baseincs.h has had lurk-specific win32 includes moved to
// lurk.cpp, cliconfig.cpp has been modified to reflect the changes to
// log/logscreen/logscreenf, and mail.cpp uses logscreen now, instead
// of printf. client.cpp has had variable names changed as well, etc.
//
// Revision 1.165  1998/07/30 05:08:52  silby
// Fixed DONT_USE_PATHWORK handling, ini_etc strings were still being
// included, now they are not. Also, added the logic for
// dialwhenneeded, which is a new lurk feature.
//
// Revision 1.164  1998/07/30 02:33:20  blast
// Lowered minimum network timeout from 30 to 5 ...
//
// Revision 1.163  1998/07/29 05:14:33  silby
// Changes to win32 so that LurkInitiateConnection now works -
// required the addition of a new .ini key connectionname=.  Username
// and password are automatically retrieved based on the
// connectionname.
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
// Changed all lurk options to use a LURK define (automatically set in
// client.h) so that lurk integration of mac/amiga clients needs only
// touch client.h and two functions in client.cpp
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
return "@(#)$Id: cliconfig.cpp,v 1.193 1998/11/09 14:32:16 chrisb Exp $"; }
#endif

#include "cputypes.h"
#include "console.h"
#include "client.h"   // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h" // basic (even if port-specific) #includes
#include "iniread.h"
#include "network.h"  
#include "triggers.h" //[Check|Raise][Pause|Exit]RequestTrigger()/InitXHandler()
#include "scram.h"     // InitRandom2(id)
#include "pathwork.h"
#include "logstuff.h"  //Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "selcore.h"   //GetCoreNameFromCoreType()
#include "lurk.h"      //lurk stuff
#include "cpucheck.h"  //GetProcessorType() for mmx stuff

// --------------------------------------------------------------------------

#define CONF_ID                    0
#define CONF_THRESHOLDI            1
#define CONF_THRESHOLDO            2
#define CONF_THRESHOLDI2           3
#define CONF_THRESHOLDO2           4
#define CONF_COUNT                 5
#define CONF_HOURS                 6
#define CONF_TIMESLICE             7
#define CONF_NICENESS              8
#define CONF_LOGNAME               9
#define CONF_UUEHTTPMODE          10
#define CONF_KEYPROXY             11
#define CONF_KEYPORT              12
#define CONF_HTTPPROXY            13
#define CONF_HTTPPORT             14
#define CONF_HTTPID               15
#define CONF_CPUTYPE              16
#define CONF_MESSAGELEN           17
#define CONF_SMTPSRVR             18
#define CONF_SMTPPORT             19
#define CONF_SMTPFROM             20
#define CONF_SMTPDEST             21
#define CONF_NUMCPU               22
#define CONF_CHECKPOINT           23
#define CONF_CHECKPOINT2          24
#define CONF_RANDOMPREFIX         25
#define CONF_PREFERREDBLOCKSIZE   26
#define CONF_PROCESSDES           27
#define CONF_QUIETMODE            28
#define CONF_NOEXITFILECHECK      29
#define CONF_PERCENTOFF           30
#define CONF_FREQUENT             31
#define CONF_NODISK               32
#define CONF_NOFALLBACK           33
#define CONF_CKTIME               34
#define CONF_NETTIMEOUT           35
#define CONF_EXITFILECHECKTIME    36
#define CONF_OFFLINEMODE          37
#define CONF_LURKMODE             38
#define CONF_RC5IN                39
#define CONF_RC5OUT               40
#define CONF_DESIN                41
#define CONF_DESOUT               42
#define CONF_PAUSEFILE            43
#define CONF_DIALWHENNEEDED       44
#define CONF_CONNECTNAME          45
#define OPTION_COUNT              46

#define MAXMENUENTRIES 18

static const char *OPTION_SECTION="parameters"; //#define OPTION_SECTION "parameters"
static const char *CONFMENU_CAPTION="RC5DES Client Configuration: %s\n"
"----------------------------------------------------------------------------\n";

#define INISETKEY(key, value) ini.setrecord(OPTION_SECTION, options[key].name, IniString(value))
#define INIGETKEY(key) (ini.getkey(OPTION_SECTION, options[key].name, options[key].defaultsetting)[0])
#define INIFIND(key) ini.findfirst(OPTION_SECTION, options[key].name)


#if defined(NOCONFIG)
  #define CFGTXT(x) NULL
#else
  #define CFGTXT(x) x
#endif

// --------------------------------------------------------------------------

#if !defined(NOCONFIG)
static const char *menutable[6]=
  {
  "Block and Buffer Options",
  "Logging Options",
  "Network and Communication Options",
  "Performance and Processor Options",
  "Miscellaneous Options"
  };

#ifdef OLDNICENESS
static const char *nicenesstable[]=
  {
  "Extremely Nice",
  "Nice",
  "Nasty"
  };
#endif

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
{ "id", CFGTXT("Your distributed.net ID"), "", 
  CFGTXT(
  "Completed blocks sent back to distributed.net are tagged with the email\n"
  "address of the person whose machine completed those blocks. That address\n"
  "is used as a unique 'account' identifier in three ways: (a) this is how\n"
  "distributed.net will contact the owner of the machine that submits the\n"
  "winning key; (b) The owner of that address receives credit for completed\n"
  "blocks which may then be transferred to a team account; (c) The number of\n"
  "blocks completed may be used as votes in the selection of a recipient of\n"
  "the prize-money reserved for a non-profit organization.\n"
  ),1,1,1,NULL},
//1
{ "threshold", CFGTXT("RC5 block fetch threshold"), "10", 
  CFGTXT(
  "This option specifies how many blocks your client will buffer between\n"
  "communications with a keyserver. The client operates directly on blocks\n"
  "stored in the input buffer, and puts finished blocks into the output buffer.\n"
  "When the number of blocks in the input buffer reaches 0, the client will\n"
  "attempt to connect to a keyserver, fill the input buffer to the threshold,\n"
  "and send in all completed blocks. Keep the number of blocks to buffer low\n"
  "(10 or less) if you have a fixed (static) connection to the internet. If\n"
  "you use a dial-up connection buffer as many blocks as you would complete\n"
  "in a day (running the client with -benchmark will give you a hint as what\n"
  "might be accomplished by this machine in one day). You may also force a\n"
  "buffer exchange by starting the client with the -update option.  Do not\n"
  "buffer more than what might be accomplished in one week; you might not\n"
  "receive credit for them. The maximum number of blocks that can be buffered\n"
  "is 1000. The number of blocks defined for the flush threshold should\n"
  "generally be the same as what is defined for the fetch threshold.\n"
  ),1,2,2,NULL},
//2
{ "threshold", CFGTXT("RC5 block flush threshold"), "10",
  "" /*options[CONF_THRESHOLDI].comments*/,1,2,3,NULL},
//3
{ "threshold2", CFGTXT("DES block fetch threshold"), "10", 
  "" /*options[CONF_THRESHOLDI].comments*/,1,2,4,NULL},
//4
{ "threshold2", CFGTXT("DES block flush threshold"), "10",
  "" /*options[CONF_THRESHOLDI].comments*/,1,2,5,NULL},
//5
{ "count", CFGTXT("Complete this many blocks, then exit"), "0", 
  CFGTXT(
  "This option specifies that you wish to have the client exit after it has\n"
  "crunched a predefined number of blocks. Use 0 (zero) to apply 'no limit'.\n"
  ),5,2,1,NULL},
//6
{ "hours", CFGTXT("Run for this many hours, then exit"), "0.00", 
  CFGTXT(
  "This option specifies that you wish to have the client exit after it has\n"
  "crunched a predefined number of hours. Use 0 (zero) to apply 'no limit'.\n"
  ),5,1,2,NULL},
//7
{ "timeslice", CFGTXT("Keys per timeslice"),
#if (CLIENT_OS == OS_RISCOS)
    "2048",
#else
    "65536",
#endif
    CFGTXT("The lower the value, the less impact the client will have on your system, but\n"
    "the slower it will go. Values from 256 to 65536 are good."),
    0 /*timeslice is obsolete. was menu 4 */,2,5,NULL},
//8
#ifdef OLDNICENESS
{ "niceness", CFGTXT("Level of niceness to run at"), "0",
  CFGTXT("Extremely Nice will not slow down other running programs.\n"
  "Nice may slow down other idle-mode processes.\n"
  "Nasty will cause the client to run at regular user level priority.\n\n"
  "On a completely idle system, all options will result in the same\n"
  "keyrate. For this reason, Extremely Nice is recommended.\n"),4,2,1,NULL,
  CFGTXT(&nicenesstable[0]),0,2},
#else
{ "priority", CFGTXT("Priority level to run at"), "0",
#if (CLIENT_OS == OS_NETWARE) 
  CFGTXT(
  "The priority option is ignored on this machine. The distributed.net client\n"
  "for NetWare dynamically adjusts its process priority.\n"
  ),
#elif (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN32S)
  CFGTXT(
  "The priority option is ignored on this machine. distributed.net clients\n"
  "for Windows always run at lowest ('idle') priority.\n"
  ),
#elif (CLIENT_OS == OS_RISCOS)
  CFGTXT(
  "The priority option is ignored on this machine. The distributed.net client\n"
  "for RISC OS dynamically adjusts its process priority.\n"
  ),
#elif (CLIENT_OS==OS_MACOS)
  CFGTXT(
  "DESCRIPTION IS MISSING"
  ),
#else
  CFGTXT(
  "The higher the client's priority, the greater will be its demand for\n"
  "processor time. The operating system will fulfill this demand only after\n"
  "the demands of other processes with a higher or equal priority are fulfilled\n"
  "first." /*" That is, the higher the priority, the more often a\n"
  "a process will get a chance to run. Whether that process then actually\n"
  "does something with the chance or simply allows the operating system to pass\n"
  "it to another process (or the client) is another matter." */ " At priority zero,\n"
  "the client will get processing time only when all other processes are idle\n"
  "(give up their chance to run). At priority nine, the client will always get\n"
  "CPU time unless there is a time-critical process waiting to be run - this is\n"
  "obviously not a good idea unless the machine is running no other programs.\n"
  ),
#endif  
  4 /*optionscreen*/, 2 /*integer*/, 1 /*menupos*/, NULL, NULL, 0, 9 },
#endif
//9
{ "logname", CFGTXT("File to log to"), "", 
  CFGTXT(
  "To enable logging to file you must specify the name of a logfile. The filename\n"
  "is limited a length of 128 characters and may not contain spaces. The file\n"
  "will be created to be in the client's directory unless a path is specified.\n"
  ),2,1,1,NULL},
//10
{ "uuehttpmode", CFGTXT("Firewall Communications mode (UUE/HTTP/SOCKS)"), "0",
  CFGTXT(
  "This option determines what protocol to use when communicating via a SOCKS\n"
  "or HTTP proxy, or optionally when communicating directly with a keyserver\n"
  "that is listening a telnet port. Specify 0 (zero) if you have a direct\n"
  "connection to either a personal proxy or to a distributed.net keyserver\n"
  "on the internet.\n"
  ),3,2,1,NULL,CFGTXT(&uuehttptable[0]),0,5},
//11
{ "keyproxy", CFGTXT("Preferred Keyserver"), "(auto)",
  CFGTXT(
  "This is the name or IP address of the machine that your client will\n"
  "obtain keys from and send completed blocks to. If your client will be\n"
  "connecting to a personal proxy, then enter the name or address of that\n"
  "machine here. If your client will be connecting directly to one of the\n"
  "distributed.net main key servers, then clear this setting, or leave it\n"
  "at \"(auto)\". The client will then automatically select a key server\n"
  "close to you.\n"
  ),3,1,2,NULL},
//12
{ "keyport", CFGTXT("Keyserver port"), "2064", 
  CFGTXT(
  "This field determines which keyserver port the client should connect to.\n"
  "Leave this option at 2064 unless you are using a specially configured\n"
  "personal proxy or you are using the client from behind a firewall.\n"
  ),3,2,3,NULL,NULL,1,0xFFFF},
//13
{ "httpproxy", CFGTXT("Local HTTP/SOCKS proxy address"), "proxy.mydomain.com",
  CFGTXT(
  "This field determines the hostname or IP address of the firewall proxy\n"
  "through which the client should communicate.\n"
  ),3,1,4,NULL},
//14
{ "httpport", CFGTXT("Local HTTP/SOCKS proxy port"), "80", 
  CFGTXT(
  "This field determines the port number on the firewall proxy to which the\n"
  "the client should connect.\n"
  ),3,2,5,NULL,NULL,1,0xFFFF},
//15
{ "httpid", CFGTXT("HTTP/SOCKS proxy userid/password"), "", 
  CFGTXT(
  "Specify a username or password in this field if your SOCKS host requires\n"
  "authentication before permitting communication through it.\n"
  ),3,1,6,NULL},
//16
{ "cputype", CFGTXT("Processor type"), "-1 (autodetect)", 
  CFGTXT(
  "This option determines which processor the client will optimize operations\n"
  "for.  While auto-detection is preferrable for most processor families, you may\n"
  "wish to set the processor type manually if detection fails or your machine's\n"
  "processor is detected incorrectly.\n"
  ),0,2,0,
  NULL,NULL,0,0},
//17
{ "messagelen", CFGTXT("Log by mail spool size (bytes)"), "0", 
  CFGTXT(
  "The client is capable of sending you a log of the client's progress by mail.\n"
  "To activate this capability, specify how much you want the client to buffer\n"
  "before sending. The minimum is 2048 bytes, the maximum is approximately 130000\n"
  "bytes. Specify 0 (zero) to disable logging by mail.\n"
  ),2,2,2,NULL,NULL,2048,125000},
//18
{ "smtpsrvr", CFGTXT("SMTP Server to use"), "", 
  CFGTXT(
  "Specify the name or DNS address of the SMTP host via which the client should\n"
  "relay mail logs. The default is the hostname component of the email address from\n"
  "which logs will be mailed.\n"
  ),2,1,3,NULL},
//19
{ "smtpport", CFGTXT("SMTP Port"), "25", 
  CFGTXT(
  "Specify the port on the SMTP host to which the client's mail subsystem should\n"
  "connect when sending mail logs. The default is port 25.\n"
  ),2,2,4,NULL},
//20
{ "smtpfrom", CFGTXT("E-mail address that logs will be mailed from"), 
  "" /* *((const char *)(options[CONF_ID].thevariable)) */, 
  CFGTXT(
  "(Some servers require this to be a real address)\n"
  ),2,1,5,NULL},
//21
{ "smtpdest", CFGTXT("E-mail address to send logs to"), 
  "" /* *((const char *)(options[CONF_ID].thevariable)) */, 
  CFGTXT(
  "Full name and site eg: you@your.site.  Comma delimited list permitted.\n"
  ),2,1,6,NULL},
//22
{ "numcpu", CFGTXT("Number of processors available"), "-1 (autodetect)", 
  CFGTXT(
  "This option specifies the number of threads you want the client to work on.\n"
  "On multi-processor machines this should be set to the number of processors\n"
  "available or to -1 to have the client attempt to auto-detect the number of\n"
  "processors. Multi-threaded clients can be forced to run single-threaded by\n" 
  "setting this option to zero.\n"
  ),4,2,3,NULL},
//23
{ "checkpointfile", CFGTXT("RC5 Checkpoint Path/Name"),"",
  CFGTXT(
  "This option sets the location of the RC5 checkpoint file. Checkpoints are\n"
  "where the client writes its progress to disk so that it can recover partially\n"
  "completed work if there is a crash or power outage in the middle of a block.\n"
  "DO NOT SHARE CHECKPOINTS BETWEEN CLIENTS OR BETWEEN CONTESTS. Avoid the use of\n"
  "checkpoints unless your machine suffers from frequent crashes or you live in\n"
  "an area with an unstable power supply.\n"
  ),1,1,13,NULL},
//24
{ "checkpointfile2", "DES Checkpoint Path/Name","",
  "" /* option[CONF_CHECKPOINT].comments */,1,1,14,NULL},
//25
{ "randomprefix", CFGTXT(""),"100",
  CFGTXT(""),/*not user changeable */0,2,0,NULL},
//26
{ "preferredblocksize", CFGTXT("Preferred Block Size (2^X keys/block)"),"31",
  CFGTXT(
  "When fetching blocks from a keyserver, the client will request blocks with\n"
  "the size you specify in this option. Running the client with the -benchmark\n"
  "switch will give you a hint as to what the preferred block size for this\n"
  "machine might be. Block sizes are specified as powers of 2. The minimum and\n"
  "maximum block sizes are 28 and 31 respectively.\n"
  ),1,2,6,NULL,NULL,28,31},
//27
{ "processdes", CFGTXT("Compete in DES contests?"),"1",
   CFGTXT(
   "Under certain circumstances, it may become necessary to prevent the client\n"
   "from competing in DES contests.\n"
   ),5,3,4,NULL},
//28
{ "quiet", CFGTXT("Disable all screen output? (quiet mode)"),"0",
  CFGTXT(
  "When enabled, this option will cause the client to suppress all screen output.\n"
  "Distributed.net strongly encourages the use of logging to file if you choose to\n"
  "run the client with disabled screen output. This option is synonymous with the\n"
  "-runhidden and -quiet command line switches and can be overridden with the\n"
  "-noquiet switch.\n"
  ),5,3,5,NULL},
//29
{ "noexitfilecheck", CFGTXT("Disable exit file checking?"),"0",
  CFGTXT(
  "When disabled, this option will cause the client to watch for a file named\n"
  "\"exitrc5.now\", the presence of which being a request to the client to\n"
  "shut itself down. (The name of the exit flag file may be set in the ini.)\n"
  ),5,3,7,NULL},
//30
{ "percentoff", CFGTXT("Disable the block completion indicator?"),"0",
  CFGTXT(
  ""
  ),5,3,6,NULL},
//31
{ "frequent", CFGTXT("Attempt keyserver connections frequently?"),"0",
  CFGTXT(
  "Enabling this option will cause the client to flush/fetch every few\n"
  "minutes or so. You might want to use this if you have a single computer\n"
  "with a network connecting \"feeding\" other clients via a buff-in.* file\n"
  "so that the buffer never reaches empty. If you're behind a firewall and\n"
  "experience frequent connection failures, this may be useful as well.\n"
  ),3,3,6,NULL},
//32
{ "nodisk", CFGTXT("Buffer blocks in RAM only? (no disk I/O)"),"0",
   CFGTXT(
   "This option is for machines with permanent connections to a keyserver\n"
   "but without local disks. Note: This option will cause all buffered,\n"
   "unflushable blocks to be lost by a client shutdown.\n"
   ),1,3,7,NULL},
//33
{ "nofallback", CFGTXT("Disable fallback to a distributed.net keyserver?"),"0",
  CFGTXT(
  "If the host you specify in the 'preferred keyserver' option is down, the\n"
  "client normally falls back to a distributed.net keyserver.\n"
  ),
  3,3,7,NULL},
//34
{ "cktime", CFGTXT("Interval between saving of checkpoints (minutes):"),"5",
  CFGTXT(
  "This option determines the frequency (in minutes) between checkpoint writes.\n"
  )
  ,1,2,8,NULL},
//35
{ "nettimeout", CFGTXT("Network Timeout (seconds)"), "60",
  CFGTXT(
  "This option determines the amount of time the client will wait for a network\n"
  "read or write acknowledgement before it assumes that the connection has been\n"
  "broken.\n"
  ),3,2,8,NULL},
//36
{ "exitfilechecktime", "","30","", /* obsolete */ 0,2,0,NULL},
//37
{ "runbuffers", CFGTXT("Offline operation mode"),"0",
  CFGTXT(
  "Normal Operation: The client will connect to a keyserver as needed,\n"
  "        and use random blocks if a keyserver connection cannot be made.\n"
  "Offline Always: The client will never connect to a keyserver, and will\n"
  "        generate random blocks if the block buffers empty.\n"
  "Finish Buffers and exit: The client will never connect to a keyserver,\n"
  "        and when the block buffers empty, it will terminate.\n"
  ),3,2,9,NULL,CFGTXT(&offlinemodetable[0]),0,2},
//38
{ "lurk", CFGTXT("Modem detection options"),"0",
  CFGTXT(
  "Normal mode: the client will send/receive blocks only when it\n"
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
  "        on random blocks until a connection is detected.\n"
  ),
  3,2,10,NULL,CFGTXT(&lurkmodetable[0]),0,2},
//39
{ "in",  CFGTXT("RC5 In-Buffer Path/Name"),  "buff-in"  EXTN_SEP "rc5",CFGTXT(""),1,1,9,NULL},
//40
{ "out", CFGTXT("RC5 Out-Buffer Path/Name"), "buff-out" EXTN_SEP "rc5",CFGTXT(""),1,1,10,NULL},
//41
{ "in2", CFGTXT("DES In-Buffer Path/Name"),  "buff-in"  EXTN_SEP "des",CFGTXT(""),1,1,11,NULL},
//42
{ "out2",CFGTXT("DES Out-Buffer Path/Name"), "buff-out" EXTN_SEP "des",CFGTXT(""),1,1,12,NULL},
//43
{ "pausefile",CFGTXT("Pausefile Path/Name"),"",CFGTXT(""),5,1,3,NULL},
//44
{ "dialwhenneeded", CFGTXT("Dial the Internet when needed?"),"0",
  CFGTXT(""),3,3,11,NULL},
//45  
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

// checks for user to type yes or no. Returns 1=yes, 0=no, -1=unknown
static int yesno(char *str)
{
  if (strcmpi(str, "yes")==0 || strcmpi(str, "y")==0 || 
      strcmpi(str, "true")==0 || strcmpi(str, "t")==0 || 
      strcmpi(str, "1")==0) 
    return 1;
  if (strcmpi(str, "no")==0 || strcmpi(str, "n")==0 || 
      strcmpi(str, "false")==0 || strcmpi(str, "f")==0 || 
      strcmpi(str, "0")==0) 
    return 0;
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

static s32 findmenuoption( s32 menu, s32 option)
    // Returns the id of the option that matches the menu and option
    // requested. Will return -1 if not found.
{
  int tpos;

  for (tpos=0; tpos < OPTION_COUNT; tpos++)
    {
    if ((options[tpos].optionscreen==menu) &&
        (options[tpos].menuposition==option))
      return (s32)tpos;
    }
  return -1;
}

// --------------------------------------------------------------------------

#if !defined(NOCONFIG)
s32 Client::ConfigureGeneral( s32 currentmenu )
{
  char parm[128],parm2[128];
  s32 choice=1;
  s32 temp;
  s32 temp2;
  char *p;

  do                    //note: don't return or break from inside 
    {                   //the loop. Let it fall through instead. - cyp

    /* ------ Sets all the pointers/etc for optionstruct options ------- */

    if (strcmpi(id,"rc5@distributed.net") == 0)
      id[0]=0; /*is converted back to 'rc5@distributed.net' in ValidateConfig()*/
    
    options[CONF_ID].thevariable=(char *)(&id[0]);
    options[CONF_THRESHOLDI].thevariable=&inthreshold[0];
    options[CONF_THRESHOLDO].thevariable=&outthreshold[0];
    options[CONF_THRESHOLDI2].thevariable=&inthreshold[1];
    options[CONF_THRESHOLDO2].thevariable=&outthreshold[1];
    options[CONF_THRESHOLDO].comments=options[CONF_THRESHOLDI2].comments=
    options[CONF_THRESHOLDO2].comments=options[CONF_THRESHOLDI].comments;
    options[CONF_COUNT].thevariable=&blockcount;
    options[CONF_HOURS].thevariable=(char *)(&hours[0]);
    options[CONF_TIMESLICE].thevariable=&timeslice;
    
    #if !((CLIENT_OS==OS_MACOS) || (CLIENT_OS==OS_RISCOS) || (CLIENT_OS==OS_WIN16))
    options[CONF_TIMESLICE].optionscreen=0;
    #endif
    #ifdef OLDNICENESS
    options[CONF_NICENESS].thevariable=&niceness;
    #else
    options[CONF_NICENESS].thevariable=&priority;
    #endif
    
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
    options[CONF_SMTPFROM].defaultsetting=
    options[CONF_SMTPDEST].defaultsetting=(char *)options[CONF_ID].thevariable;
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
    options[CONF_CHECKPOINT].thevariable=(char *)(&checkpoint_file[0][0]);
    options[CONF_CHECKPOINT2].thevariable=(char *)(&checkpoint_file[1][0]);
    options[CONF_RC5IN].thevariable=(char *)(&in_buffer_file[0][0]);
    options[CONF_RC5OUT].thevariable=(char *)(&out_buffer_file[0][0]);
    options[CONF_DESIN].thevariable=(char *)(&in_buffer_file[1][0]);
    options[CONF_DESOUT].thevariable=(char *)(&out_buffer_file[1][0]);
    options[CONF_PAUSEFILE].thevariable=(char *)(&pausefile[0]);
    
    options[CONF_CPUTYPE].optionscreen=0;
    options[CONF_CPUTYPE].choicemax=0;
    options[CONF_CPUTYPE].choicemin=0;
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
      options[CONF_CPUTYPE].defaultsetting="-1";
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
    
    if (uuehttpmode > 1) /* not telnet and not normal */
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
      
    /* -------------------- end setup options --------------------- */
    
    int lkg_autofind = (autofindkeyserver != 0);
    char lkg_keyproxy[sizeof(keyproxy)];
    strcpy( lkg_keyproxy, keyproxy );
    if ( isstringblank( keyproxy) || 
      ( autofindkeyserver && _IsHostnameDNetHost( keyproxy ) ))
      {
      autofindkeyserver = 1;
      strcpy( keyproxy, "(auto)" );
      }

    // display menu

    do   //while invalid CONF_xxx option selected
      {
      ConClear(); //in logstuff.cpp
      LogScreenRaw(CONFMENU_CAPTION, menutable[currentmenu-1]);

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
            if (options[choice].choicelist != NULL)
              strcpy(parm,options[choice].choicelist[
                ((long)*(s32 *)options[choice].thevariable)]);
            else if ((long)*(s32 *)options[choice].thevariable == 
                (long)(atoi(options[choice].defaultsetting)))
              strcpy(parm,options[choice].defaultsetting);
            else
              sprintf(parm,"%li",(long)*(s32 *)options[choice].thevariable);
            LogScreenRaw("%s\n",parm);
            }
          else if (options[choice].type==3)
            {
            sprintf(parm, "%s", *(s32 *)options[choice].thevariable?"yes":"no");
            LogScreenRaw("%s\n",parm);
            }
          }
        }
      LogScreenRaw("\n 0) Return to main menu\n");

      // get choice from user
      LogScreenRaw("\nChoice --> ");
      ConInStr( parm, 4, 0 );
      choice = atoi( parm );

      if ( choice == 0 || CheckExitRequestTriggerNoIO())
        choice = -2; //quit request
      else if ( choice > 0 )
        choice = findmenuoption(currentmenu,choice); // returns -1 if !found
      else
        choice = -1;
      } while ( choice == -1 ); //while findmenuoption() says this is illegal

    if ( choice >= 0 ) //if valid CONF_xxx option
      {
      ConClear(); //in logstuff.cpp
      LogScreenRaw(CONFMENU_CAPTION, menutable[currentmenu-1]);
      LogScreenRaw("\n%s:\n\n", options[choice].description );
      p = (char *)options[choice].comments;
      while (strlen(p) > (sizeof(parm)-1))
        {
        strncpy(parm,p,(sizeof(parm)-1));
        parm[(sizeof(parm)-1)]=0;
        LogScreenRaw("%s",parm);
        p+=(sizeof(parm)-1);
        }
      LogScreenRaw("%s\n",p);

      if ( options[choice].type==1 || options[choice].type==2 )
        {
        if (options[choice].choicelist !=NULL)
          {
          for ( temp = options[choice].choicemin; temp < options[choice].choicemax+1; temp++)
            LogScreenRaw("  %2d) %s\n", (int) temp,options[choice].choicelist[temp]);
          }
        if (options[choice].type==1)
          strcpy(parm,(char *)options[choice].thevariable);
        else 
          sprintf(parm,"%li",(long)*(s32 *)options[choice].thevariable);
        LogScreenRaw("Default Setting: %s\nCurrent Setting: %s\nNew Setting --> ",
                      options[choice].defaultsetting, parm );
        ConInStr( parm, sizeof(parm), CONINSTR_BYEXAMPLE );

        for ( p = parm; *p; p++ )
          {
          if ( !isprint(*p) )
            {
            *p = 0;
            break;
            }
          }
        if (strlen(parm) != 0)
          {
          p = &parm[strlen(parm)-1];
          while (p >= &parm[0] && isspace(*p))
            *p-- = 0;
          p = parm;
          while (*p && isspace(*p))
            p++;
          if (p > &parm[0])
            strcpy( parm, p );
          }
        }
      else if (options[choice].type==3)
        {
        sprintf(parm, "%s", *(s32 *)options[choice].thevariable?"yes":"no");
        LogScreenRaw("Default Setting: %s\nCurrent Setting: %s\nNew Setting --> ",
               *(options[choice].defaultsetting)=='0'?"no":"yes", parm );
        parm[1]=0;
        ConInStr( parm, 2, CONINSTR_BYEXAMPLE );
        if (parm[0] == 'y' || parm[0] == 'Y')
          strcpy(parm,"yes");
        else if (parm[0] == 'n' || parm[0] == 'N')
          strcpy(parm,"no");
        else parm[0]=0;
        }
      else
        choice = -1;

      if (CheckExitRequestTriggerNoIO())
        choice = -2;
      } //if ( choice >= 0 ) //if valid CONF_xxx option

    if ( choice != CONF_KEYPROXY )
      {
      strcpy( keyproxy, lkg_keyproxy ); //copy whatever they had back
      autofindkeyserver = (lkg_autofind != 0);
      }
    else
      {
      autofindkeyserver = 0; //OFF unless the user left it at auto
      if (!parm[0] || strcmpi(parm,"(auto)")==0 || strcmpi(parm,"auto")==0)
        {
        autofindkeyserver = 1;
        strcpy( parm, lkg_keyproxy ); //copy back what was in there before
        if (isstringblank( parm )) //dummy value so that we don't fall into
          strcpy( parm, "rc5proxy.distributed.net" ); //the null string trap.
        }
      }
    
    if ( choice >= 0 ) // && (parm[0] || choice == CONF_LOGNAME ))
      {
      if ( parm[0] == 0)
        strcpy( parm, options[choice].defaultsetting );

      switch ( choice )
        {
        case CONF_ID:
          strncpy( id, parm, sizeof(id) - 1 );
          //ValidateConfig();
          break;
        case CONF_THRESHOLDI:
          choice=atoi(parm);
          if (choice > 0) 
            {
            inthreshold[0]=choice;
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
            inthreshold[1]=choice;
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
          if ( choice >= options[CONF_NICENESS].choicemin && 
               choice <= options[CONF_NICENESS].choicemax )
            #ifdef OLDNICENESS
            niceness = choice;
            #else
            priority = choice;
            #endif
          break;
        case CONF_LOGNAME:
          strncpy( logname, parm, sizeof(logname) - 1 );
          if (isstringblank(logname)) 
            logname[0]=0;
          break;
        case CONF_KEYPROXY:
          strncpy( keyproxy, parm, sizeof(keyproxy) - 1 );
          break;
        case CONF_KEYPORT:
          choice = atoi(parm);
          if (choice > 0 && choice <= 65535)
            keyport = choice;
          break;
        case CONF_HTTPPROXY:
          strncpy( httpproxy, parm, sizeof(httpproxy) - 1);
          ValidateConfig();
          break;
        case CONF_HTTPPORT:
          httpport = atoi(parm); break;
        case CONF_HTTPID:
          if ( strcmp(parm,".") == 0)
            httpid[0]=0;
          else if (uuehttpmode == 4) // socks4
            strcpy(httpid, parm);
          else
            {             // http & socks5
            LogScreenRaw("Enter password--> ");

            ConInStr( parm2, sizeof(parm2), 0 /* CONINSTR_ASPASSWORD */ );
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
          autofindkeyserver=1; //we are using a default, so turn it back on
          switch (uuehttpmode)
            {
            case 1: /* UUE mode (telnet) */ p = "23"; keyport = 23; break;   
            case 2: /* HTTP mode */
            case 3: /* HTTP+UUE mode */     p = "80"; keyport = 80; break;
            case 4: /* SOCKS4 */
            case 5: /* SOCKS5 */
            default:/* normal */            p = ""; keyport = 2064; break;
            }
          sprintf(keyproxy,"us%s.v27.distributed.net", p );
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
          strncpy( checkpoint_file[0] , parm, sizeof(checkpoint_file[1]) -1 );
          ValidateConfig();
          break;
        case CONF_CHECKPOINT2:
          strncpy( checkpoint_file[1] , parm, sizeof(checkpoint_file[1]) -1 );
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
          strncpy( in_buffer_file[0] , parm, sizeof(in_buffer_file[0]) -1 );
          ValidateConfig();
          break;
        case CONF_RC5OUT:
          strncpy( out_buffer_file[0] , parm, sizeof(out_buffer_file[0]) -1 );
          ValidateConfig();
          break;
        case CONF_DESIN:
          strncpy( in_buffer_file[1] , parm, sizeof(in_buffer_file[1]) -1 );
          ValidateConfig();
          break;
        case CONF_DESOUT:
          strncpy( out_buffer_file[1] , parm, sizeof(out_buffer_file[1]) -1 );
          ValidateConfig();
          break;
        case CONF_PAUSEFILE:
          strncpy( pausefile, parm, sizeof(pausefile) -1 );
          if (isstringblank(pausefile)) 
            pausefile[0]=0;
          break;
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
  int returnvalue = 1;
#if !defined(NOCONFIG)
  unsigned int choice;
  char parm[128];
  returnvalue=0;

  if (!ConIsScreen())
    {
    ConOutErr("Can't configure when stdin or stdout is redirected.\n");
    returnvalue = -1;
    }

  
  while (returnvalue == 0)
    {
    ConClear(); //in logstuff.cpp
    LogScreenRaw(CONFMENU_CAPTION, "");
    for (choice=1;choice<(sizeof(menutable)/sizeof(menutable[0]));choice++)
      LogScreenRaw(" %u) %s\n",choice, menutable[choice-1]);
    LogScreenRaw("\n 9) Discard settings and exit"
                 "\n 0) Save settings and exit\n\n");

    if (isstringblank(id) || strcmpi(id,"rc5@distributed.net")==0)
      LogScreenRaw("Note: You have not yet provided a distributed.net ID.\n"
              "       Please go to the '%s' and set it.\n\n",menutable[0]);
    LogScreenRaw("Choice --> ");

    ConInStr(parm, 2, 0);

    choice = 9999;
    if (strlen(parm)==1 && isdigit(parm[0]))
      choice = atoi(parm);
    if (CheckExitRequestTriggerNoIO() || choice==9)
      returnvalue = -1; //Breaks and tells it NOT to save
    else if (choice == 0)
      returnvalue=1; //Breaks and tells it to save
    else if (choice >0 && choice<(sizeof(menutable)/sizeof(menutable[0])))
      ConfigureGeneral(choice);
    }
#endif
  return returnvalue;
}


//----------------------------------------------------------------------------

int Client::ReadConfig(void)  //DO NOT PRINT TO SCREEN (or whatever) FROM HERE
{
  IniSection ini;
  s32 inierror, tempconfig;
  char buffer[64];
  char *p;

  inierror = ini.ReadIniFile( GetFullPathForFilename( inifilename ) );
  if (inierror) return -1;

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

  #if 0
  timeslice = INIGETKEY(CONF_TIMESLICE);
  #else
  timeslice = 65536;
  #endif

#ifdef OLDNICENESS
  niceness = INIGETKEY(CONF_NICENESS);
#else  
  IniRecord *tempptr = ini.findfirst( "processor usage", "priority");
  if (tempptr) 
    priority = (ini.getkey("processor usage", "priority", "0")[0]);
  else
    {
    priority = (ini.getkey(OPTION_SECTION, "niceness", "0")[0]);
    priority = ((priority==2)?(8):((priority==1)?(4):(0)));
    }
#endif    

  //do an autofind only if the host is a dnet host AND autofindkeyserver is on.
  autofindkeyserver = 0;
  INIGETKEY(CONF_KEYPROXY).copyto(keyproxy, sizeof(keyproxy));
  if (isstringblank(keyproxy) || strcmpi( keyproxy, "(auto)")==0 ||
    strcmpi( keyproxy, "auto")==0 || strcmpi( keyproxy, "rc5proxy.distributed.net" )==0) 
    {                                         
    strcpy( keyproxy, "us.v27.distributed.net" ); 
    autofindkeyserver = 1; //let Resolve() get a better hostname.
    }
  else if (_IsHostnameDNetHost(keyproxy))
    {
    tempconfig=ini.getkey("networking", "autofindkeyserver", "1")[0];
    autofindkeyserver = (tempconfig)?(1):(0);
    }

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
  tempconfig=ini.getkey(OPTION_SECTION, "win95hidden", "0")[0]; //obsolete
  if (tempconfig) quietmode=1;
  tempconfig=ini.getkey(OPTION_SECTION, "runhidden", "0")[0]; //obsolete
  if (tempconfig) quietmode=1;
  tempconfig=ini.getkey(OPTION_SECTION, "nofallback", "0")[0];
  if (tempconfig) nofallback=1;
  tempconfig=ini.getkey(OPTION_SECTION, "cktime", "0")[0];
  if (tempconfig) checkpoint_min=max(2,tempconfig);
  tempconfig=ini.getkey(OPTION_SECTION, "nettimeout", "60")[0];
  if (tempconfig) nettimeout=min(300,max(5,tempconfig));
  tempconfig=ini.getkey(OPTION_SECTION, "noexitfilecheck", "0")[0];
  if (tempconfig) noexitfilecheck=1;
  tempconfig=ini.getkey(OPTION_SECTION, "exitfilechecktime", "30")[0];
  if (tempconfig) exitfilechecktime=max(tempconfig,1);
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
INIGETKEY(CONF_CHECKPOINT).copyto(checkpoint_file[0], sizeof(checkpoint_file[0]));
INIGETKEY(CONF_CHECKPOINT2).copyto(checkpoint_file[1], sizeof(checkpoint_file[1]));

ini.getkey(OPTION_SECTION,"in",in_buffer_file[0])[0].copyto(in_buffer_file[0],sizeof(in_buffer_file[0]));
ini.getkey(OPTION_SECTION,"out",out_buffer_file[0])[0].copyto(out_buffer_file[0],sizeof(out_buffer_file[0]));
ini.getkey(OPTION_SECTION,"in2",in_buffer_file[1])[0].copyto(in_buffer_file[1],sizeof(in_buffer_file[1]));
ini.getkey(OPTION_SECTION,"out2",out_buffer_file[1])[0].copyto(out_buffer_file[1],sizeof(out_buffer_file[1]));
ini.getkey(OPTION_SECTION,"pausefile",pausefile)[0].copyto(pausefile,sizeof(pausefile));

tempconfig=ini.getkey(OPTION_SECTION, "contestdone", "0")[0];
if (tempconfig) contestdone[0]=1;
tempconfig=ini.getkey(OPTION_SECTION, "contestdone2", "0")[0];
if (tempconfig) contestdone[1]=1;

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
  if ( inthreshold[0] < 1   ) 
    inthreshold[0] = 1;
  if ( inthreshold[0] > 1000 ) 
    inthreshold[0] = 1000;
  if ( outthreshold[0] < 1   ) 
    outthreshold[0] = 1;
  if ( outthreshold[0] > 1000 ) 
    outthreshold[0] = 1000;
  if ( outthreshold[0] > inthreshold[0] ) 
    outthreshold[0]=inthreshold[0];
  if ( inthreshold[1] < 1   ) 
    inthreshold[1] = 1;
  if ( inthreshold[1] > 1000 ) 
    inthreshold[1] = 1000;
  if ( outthreshold[1] < 1   ) 
    outthreshold[1] = 1;
  if ( outthreshold[1] > 1000 ) 
    outthreshold[1] = 1000;
  if ( outthreshold[1] > inthreshold[1] ) 
    outthreshold[1]=inthreshold[1];
  if ( blockcount < 0 ) 
    blockcount = 0;
  if ( timeslice < 1 ) 
    timeslice = atoi(options[CONF_TIMESLICE].defaultsetting);
  #ifdef OLDNICENESS
  if ( niceness < options[CONF_NICENESS].choicemin || 
       niceness > options[CONF_NICENESS].choicemax )
    niceness = options[CONF_NICENESS].choicemin;
  #else
  if ( priority < options[CONF_NICENESS].choicemin || 
       priority > options[CONF_NICENESS].choicemax )
    priority = options[CONF_NICENESS].choicemin;
  #endif

  if ( uuehttpmode < 0 || uuehttpmode > 5 ) 
    uuehttpmode = 0;
  if ( randomprefix <0 || randomprefix >255) 
    randomprefix=100;
  if (smtpport < 0 || smtpport > 65535L) 
    smtpport=25;
  if (messagelen !=0 && messagelen < 2048)
    messagelen=2048;
  if (( preferred_contest_id < 0 ) || ( preferred_contest_id > 1 )) 
    preferred_contest_id = 1;
  if (preferred_blocksize < 28) 
    preferred_blocksize = 28;
  if (preferred_blocksize > 31) 
    preferred_blocksize = 31;
  if ( minutes < 0 ) 
    minutes=0;
  if ( blockcount < 0 ) 
    blockcount=0;
  if (checkpoint_min < 2) 
    checkpoint_min=2;
  else if (checkpoint_min > 30) 
    checkpoint_min=30;
  if (exitfilechecktime < 5) 
    exitfilechecktime=5;
  else if (exitfilechecktime > 600) 
    exitfilechecktime=600;
  if (nettimeout < 5) 
    nettimeout=5;
  else if (nettimeout > 300) 
    nettimeout=300;

  killwhitespace(id);
  killwhitespace(keyproxy);
  killwhitespace(httpproxy);
  killwhitespace(smtpsrvr);

  if (isstringblank(id))
    strcpy(id,"rc5@distributed.net");
  if (isstringblank(logname) || strcmp(logname,"none")==0)
    logname[0]=0;
  if (isstringblank(in_buffer_file[0]))
    strcpy(in_buffer_file[0], options[CONF_RC5IN].defaultsetting );
  if (isstringblank(out_buffer_file[0]))
    strcpy(out_buffer_file[0], options[CONF_RC5OUT].defaultsetting );
  if (isstringblank(in_buffer_file[1]))
    strcpy(in_buffer_file[1], options[CONF_DESIN].defaultsetting );
  if (isstringblank(out_buffer_file[1]))
    strcpy(out_buffer_file[1], options[CONF_DESOUT].defaultsetting );
  if (isstringblank(pausefile) || strcmp(pausefile,"none")==0)
    pausefile[0]=0;
  if (isstringblank(checkpoint_file[0]) || strcmp(checkpoint_file[0],"none")==0)
    checkpoint_file[0][0]=0;
  if (isstringblank(checkpoint_file[1]) || strcmp(checkpoint_file[1],"none")==0)
    checkpoint_file[1][0]=0;

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
  sprintf(buffer,"%d:%d",(int)inthreshold[1],(int)outthreshold[1]);
  INISETKEY( CONF_THRESHOLDI2, buffer );
  INISETKEY( CONF_COUNT, blockcount );
  sprintf(hours,"%u.%02u", (unsigned)(minutes/60), (unsigned)(minutes%60)); 
  INISETKEY( CONF_HOURS, hours );

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

  if ((tempptr = ini.findfirst( "networking", "autofindkeyserver"))!=NULL)
    tempptr->values.Erase();
  if (isstringblank(keyproxy) || (autofindkeyserver && _IsHostnameDNetHost(keyproxy)))
    {
    //autokeyserver is enabled (because its on AND its a dnet host), so delete 
    //the old ini keys so that old inis stay compatible. We could at this 
    //point set keyproxy=rc5proxy.distributed.net, but why clutter up the ini?
    tempptr = ini.findfirst(OPTION_SECTION, "keyproxy");
    if (tempptr) tempptr->values.Erase();
    }
  else 
    {
    if (_IsHostnameDNetHost(keyproxy))
      ini.setrecord("networking", "autofindkeyserver", IniString("0"));
    INISETKEY( CONF_KEYPROXY, keyproxy );
    }

  INISETKEY( CONF_KEYPORT, keyport );
  INISETKEY( CONF_CPUTYPE, cputype );
  INISETKEY( CONF_NUMCPU, numcpu );

  //INISETKEY( CONF_RANDOMPREFIX, randomprefix );
  
  INISETKEY( CONF_PREFERREDBLOCKSIZE, preferred_blocksize );
  INISETKEY( CONF_PROCESSDES, (s32)(preferred_contest_id) );
  
  INISETKEY( CONF_NOEXITFILECHECK, noexitfilecheck );
  INISETKEY( CONF_PERCENTOFF, percentprintingoff );
  INISETKEY( CONF_FREQUENT, connectoften );
  INISETKEY( CONF_NODISK, nodiskbuffers );
  INISETKEY( CONF_NOFALLBACK, nofallback );
  INISETKEY( CONF_CKTIME, checkpoint_min );
  INISETKEY( CONF_NETTIMEOUT, nettimeout );
  INISETKEY( CONF_EXITFILECHECKTIME, exitfilechecktime );
  INISETKEY( CONF_LOGNAME, logname );
  INISETKEY( CONF_CHECKPOINT, checkpoint_file[0] );
  INISETKEY( CONF_CHECKPOINT2, checkpoint_file[1] );
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

  if ((tempptr = ini.findfirst(OPTION_SECTION, "runhidden"))!=NULL)
    tempptr->values.Erase();    
  if ((tempptr = ini.findfirst(OPTION_SECTION, "os2hidden"))!=NULL)
    tempptr->values.Erase();    
  if ((tempptr = ini.findfirst(OPTION_SECTION, "win95hidden"))!=NULL)
    tempptr->values.Erase();    
  INISETKEY( CONF_QUIETMODE, ((quietmode)?("1"):("0")) );

  if (offlinemode == 1)
    ini.setrecord(OPTION_SECTION, "runoffline", IniString("1"));
  else if ((tempptr = ini.findfirst(OPTION_SECTION, "runoffline"))!=NULL)
    tempptr->values.Erase();
  if (offlinemode == 2)
    ini.setrecord(OPTION_SECTION, "runbuffers", IniString("1"));
  else if ((tempptr = ini.findfirst(OPTION_SECTION, "runbuffers"))!=NULL)
    tempptr->values.Erase();

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
  if (strcmp(dialup.connectionname,options[CONF_CONNECTNAME].defaultsetting)!=0)
    INISETKEY( CONF_CONNECTNAME, dialup.connectionname);
  else if ((tempptr = ini.findfirst(OPTION_SECTION, "connectionname"))!=NULL)
    tempptr->values.Erase();
  #endif

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
    }
  else
    {
    INISETKEY( CONF_UUEHTTPMODE, uuehttpmode );
    INISETKEY( CONF_HTTPPROXY, httpproxy );
    INISETKEY( CONF_HTTPPORT, httpport );
    INISETKEY( CONF_HTTPID, httpid);
    }

#if defined(NEEDVIRTUALMETHODS)
  InternalWriteConfig(ini);
#endif

  return( ini.WriteIniFile( GetFullPathForFilename( inifilename ) ) ? -1 : 0 );
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

  INISETKEY( CONF_RANDOMPREFIX, randomprefix );

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

