// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: confopt.cpp,v $
// Revision 1.8  1998/12/21 18:48:54  cyp
// Removed 'unused'/'unimplemented' sil[l|b]yness added in recent version.
// See client.h for full comment.
//
// Revision 1.7  1998/12/21 00:21:01  silby
// Universally scheduled update time is now retrieved from the proxy,
// and stored in the .ini file.  Not yet used, however.
//
// Revision 1.6  1998/12/20 23:00:35  silby
// Descontestclosed value is now stored and retrieved from the ini file,
// additional updated of the .ini file's contest info when fetches and
// flushes are performed are now done.  Code to throw away old des blocks
// has not yet been implemented.
//
// Revision 1.5  1998/12/01 11:24:11  chrisb
// more riscos x86 changes
//
// Revision 1.4  1998/11/26 22:27:24  cyp
// Fixed _IsHostnameDNetHost() to work with any/all distributed.net hostnames.
//
// Revision 1.3  1998/11/26 06:51:31  cyp
// Added missing log entry.
//
//

#if (!defined(lint) && defined(__showids__))
const char *confopt_cpp(void) {
return "@(#)$Id: confopt.cpp,v 1.8 1998/12/21 18:48:54 cyp Exp $"; }
#endif

#include "cputypes.h" // CLIENT_OS, s32
#include "baseincs.h" // strcmp() etc as used by isstringblank() et al.
#include "cmpidefs.h" // strcmpi() 
#include "confopt.h"  // ourselves
#include "pathwork.h" // EXTN_SEP

// --------------------------------------------------------------------------

#if defined(NOCONFIG)
  #define CFGTXT(x) NULL
#else
  #define CFGTXT(x) x
#endif

// --------------------------------------------------------------------------

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

static const char *lurkmodetable[]=
  {
  "Normal mode",
  "Dial-up detection mode",
  "Dial-up detection ONLY mode"
  };


// --------------------------------------------------------------------------

struct optionstruct conf_options[OPTION_COUNT]=
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
{ "threshold", CFGTXT("Block fetch/flush threshold"), "10", 
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
  "is 500.\n" /* The number of blocks defined for the flush threshold should\n"
  "generally be the same as what is defined for the fetch threshold.\n" */
  ),1,2,2,NULL,NULL,1,500},
//2
{ "threshold", "" /*CFGTXT("RC5 block flush threshold")*/, "10",
  "" /*options[CONF_THRESHOLDI].comments*/,/* obsolete 1*/0,2,3,NULL},
//3
{ "threshold2", "" /*CFGTXT("DES block fetch threshold")*/, "10", 
  "" /*options[CONF_THRESHOLDI].comments*/,/* obsolete 1*/0,2,4,NULL},
//4
{ "threshold2", ""/*CFGTXT("DES block flush threshold")*/, "10",
  "" /*options[CONF_THRESHOLDI].comments*/,/* obsolete 1*/0,2,5,NULL},
//5
{ "count", CFGTXT("Complete this many blocks, then exit"), "0", 
  CFGTXT(
  "This option specifies that you wish to have the client exit after it has\n"
  "crunched a predefined number of blocks. Use 0 (zero) to apply 'no limit',\n"
  "or -1 to have the client exit when the input buffer is empty (this is the\n"
  "equivalent to the -runbuffers command line option.)\n"
  ),5,2,1,NULL},
//6
{ "hours", CFGTXT("Run for this many hours, then exit"), "0:00", 
  CFGTXT(
  "This option specifies that you wish to have the client exit after it has\n"
  "crunched a predefined number of hours. Use 0 (zero) to apply 'no limit'.\n"
  ),5,1,2,NULL},
//7
{ "timeslice", CFGTXT("Keys per timeslice"),
    "65536",
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
  ),4,2,2,
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
  ),2,2,4,NULL,NULL,1,0xFFFF},
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
#if (CLIENT_OS == OS_RISCOS)
  CFGTXT(
  "If you have a RiscPC-style x86 processor card, you can make the client\n"
  "crack keys on it by setting this option to 2. If you don't have such a\n"
  "card, you should set it to 1.\n")
#else
  CFGTXT(
  "This option specifies the number of threads you want the client to work on.\n"
  "On multi-processor machines this should be set to the number of processors\n"
  "available or to -1 to have the client attempt to auto-detect the number of\n"
  "processors. Multi-threaded clients can be forced to run single-threaded by\n" 
  "setting this option to zero.\n")
#endif
  ,4,2,3,NULL,NULL,-1,128},
//23
{ "checkpointfile", CFGTXT("Checkpoint Filename"),"",
  CFGTXT(
  "This option sets the location of the checkpoint file. The checkpoint is\n"
  "where the client writes its progress to disk so that it can recover partially\n"
  "completed work if the client had previously failed to shutdown normally.\n"
  "DO NOT SHARE CHECKPOINTS BETWEEN CLIENTS. Avoid the use of checkpoints unless\n"
  "your client is running in an environment where it might not be able to shutdown\n"
  "properly.\n"
  ),1,1,13,NULL},
//24
{ "checkpointfile2", "" /* "DES Checkpoint Path/Name" */,"",
  "" /* option[CONF_CHECKPOINT].comments */,0 /*obsolete */,1,14,NULL},
//25
{ "randomprefix", CFGTXT(""),"100",
  CFGTXT(""),/*not user changeable */0,2,0,NULL,NULL,0,255},
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
{ "processdes", "" /* CFGTXT("Compete in DES contests?") */,"1",
   CFGTXT(
   "Under certain circumstances, it may become necessary to prevent the client\n"
   "from competing in DES contests.\n"
   ),/* obsolete was 5 */0,3,4,NULL},
//28
{ "quiet", CFGTXT("Disable all screen output? (quiet mode)"),"0",
  CFGTXT(
  "When enabled, this option will cause the client to suppress all screen output\n"
  "and detach itself (run in the background). Because the client is essentially\n"
  "invisible, distributed.net strongly encourages the use of logging to file if\n"
  "you choose to run the client with disabled screen output. This option is\n"
  "synonymous with the -runhidden and -quiet command line switches and can be\n"
  "overridden with the -noquiet switch.\n"
  ),5,3,5,NULL,NULL,0,1},
//29
{ "noexitfilecheck", CFGTXT("Disable exit file checking?"),"0",
  CFGTXT(
  "When disabled, this option will cause the client to watch for a file named\n"
  "\"exitrc5.now\", the presence of which being a request to the client to\n"
  "shut itself down. (The name of the exit flag file may be set in the ini.)\n"
  ),5,3,7,NULL,NULL,0,1},
//30
{ "percentoff", CFGTXT("Disable the block completion indicator?"),"0",
  CFGTXT(
  ""
  ),5,3,6,NULL,NULL,0,1},
//31
{ "frequent", CFGTXT("Frequently check for empty buffers?"),"0",
  CFGTXT(
  "Enabling this option will cause the client to check the input buffers\n"
  "every few minutes or so. You might want to use this if you have a\n"
  "single computer with a network connecting \"feeding\" other clients via\n"
  "a common buff-in.* file so that the buffer never reaches empty.\n"
  ),1,3,15,NULL,NULL,0,1},
//32
{ "nodisk", CFGTXT("Buffer blocks in RAM only? (no disk I/O)"),"0",
   CFGTXT(
   "This option is for machines with permanent connections to a keyserver\n"
   "but without local disks. Note: This option will cause all buffered,\n"
   "unflushable blocks to be lost by a client shutdown.\n"
   ),1,3,7,NULL,NULL,0,1},
//33
{ "nofallback", CFGTXT("Disable fallback to a distributed.net keyserver?"),"0",
  CFGTXT(
  "If the host you specify in the 'preferred keyserver' option is not\n"
  "reachable, the client normally falls back to a distributed.net keyserver.\n"
  ),
  3,3,7,NULL,NULL,0,1},
//34
{ "" /* "cktime" */, "","5", "", /* obsolete */ 0,2,0,NULL},
//35
{ "nettimeout", CFGTXT("Network Timeout (seconds)"), "60",
  CFGTXT(
  "This option determines the amount of time the client will wait for a network\n"
  "read or write acknowledgement before it assumes that the connection has been\n"
  "broken.\n"
  ),3,2,8,NULL,NULL,5,300},
//36
{ "" /* "exitfilechecktime" */, "","30","", /* obsolete */ 0,2,0,NULL},
//37
{ "runoffline", CFGTXT("Offline operation mode"),"0",
  CFGTXT(
  "Yes: The client will never connect to a keyserver.\n"
  " No: The client will connect to a keyserver as needed.\n"
  ),3,3,9,NULL,NULL,0,1},
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
  CFGTXT(""),3,3,11,NULL,NULL,0,1},
//45  
{ "connectionname", CFGTXT("Dial-up Connection Name"),
  "Your Internet Connection",CFGTXT(""),3,1,12,NULL}
};

// --------------------------------------------------------------------------

int confopt_IsHostnameDNetHost( const char * hostname )
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

// --------------------------------------------------------------------------

int confopt_isstringblank( const char *string )
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

// --------------------------------------------------------------------------

void confopt_killwhitespace( char *string )
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
  if ( strcmpi(string,"none") == 0 )
    string[0]=0;
  return;
}

// --------------------------------------------------------------------------

