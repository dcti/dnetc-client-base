/* 
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *confopt_cpp(void) {
return "@(#)$Id: confopt.cpp,v 1.28.2.1 1999/04/13 19:45:19 jlawson Exp $"; }

/* ----------------------------------------------------------------------- */

#include "cputypes.h" // CLIENT_OS, s32
#include "baseincs.h" // strcmp() etc as used by isstringblank() et al.
#include "cmpidefs.h" // strcmpi()
#include "client.h"   // only the MAXBLOCKSPERBUFFER define
#include "confopt.h"  // ourselves
#include "pathwork.h" // EXTN_SEP

/* ----------------------------------------------------------------------- */

static const char *uuehttptable[] =
{
  "No special encoding",
  "UUE encoding (telnet proxies)",
  "HTTP encoding",
  "HTTP+UUE encoding",
  "SOCKS4 proxy",
  "SOCKS5 proxy"
};

static const char *lurkmodetable[] =
{
  "Normal mode",
  "Dial-up detection mode",
  "Dial-up detection ONLY mode"
};

// --------------------------------------------------------------------------

#define CFGTXT(x) x

struct optionstruct conf_options[CONF_OPTION_COUNT]=
{
//0
{ CFGTXT("Your email address (distributed.net ID)"), "",
  CFGTXT(
  "Completed packets sent back to distributed.net are tagged with the email\n"
  "address of the person whose machine completed those packets. That address\n"
  "is used as a unique 'account' identifier in three ways: (a) this is how\n"
  "distributed.net will contact the owner of the machine that submits the\n"
  "winning key; (b) The owner of that address receives credit for completed\n"
  "packets which may then be transferred to a team account; (c) The number of\n"
  "packets completed may be used as votes in the selection of a recipient of\n"
  "the prize-money reserved for a non-profit organization.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//1
{  CFGTXT("Buffer packets in RAM only? (no disk I/O)"),"0",
   CFGTXT(
   "This option is for machines with permanent connections to a keyserver\n"
   "but without local disks. Note: This option will cause all buffered,\n"
   "unflushable packets to be lost by a client shutdown.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL},
//2
{ CFGTXT("Frequently check for empty buffers?"),"0",
  CFGTXT(
  "Enabling this option will cause the client to check the input buffers\n"
  "every few minutes or so. You might want to use this if you have a\n"
  "single computer with a network connecting \"feeding\" other clients via\n"
  "a common buff-in.* file so that the buffer never reaches empty.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL},
//3
{ CFGTXT("Preferred RC5/DES packet size (2^X keys/packet)"),"31 (default)",
  CFGTXT(
  "When fetching RC5 or DES packets from a keyserver, the client will request\n"
  "packets with the size you specify in this option. Running the client with\n"
  "the -benchmark switch will give you a hint as to what the best packet size\n"
  "for this machine might be. Packet sizes are specified as powers of 2.\n"
  "The minimum and maximum packet sizes are 28 and 33 respectively.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_INT,NULL,NULL,28,33,NULL},
//4
{ CFGTXT("Packet fetch/flush threshold"), "10 (default)",
  CFGTXT(
  "This option specifies how many packets your client will buffer between\n"
  "communications with a keyserver. The client operates directly on packets\n"
  "stored in the input buffer, and puts finished packets into the output buffer.\n"
  "When the number of packets in the input buffer reaches 0, the client will\n"
  "attempt to connect to a keyserver, fill the input buffer to the threshold,\n"
  "and send in all completed packets. Keep the number of packets to buffer low\n"
  "if you have a fixed (static) connection to the internet, or the cost of your\n"
  "dialup connection is negligible.\n"
  "In general, you should not buffer more than your client(s) can complete in\n"
  "one day (running the client with -benchmark will give you a hint as what\n"
  "might be accomplished by this machine in one day), and you should think twice\n"
  "about buffering more than can be accomplished in 3 days.\n"
  "You may also force a buffer exchange by starting the client with the -update\n"
  "option.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_INT,NULL,NULL,1,MAXBLOCKSPERBUFFER,NULL},
//5
{ CFGTXT("In-Buffer Filename Prefix"), "buff-in",
  CFGTXT(
  "Enter the prefix (the base name, ie a filename without an 'extension') of the\n"
  "buffer files where unfinished work will be stored. The default is \"buff-in\".\n"
  "The name of the project will concatenated internally to this base name name\n"
  "to construct the full name of the buffer file. Thus, \"buff-in\" will become\n"
  "\"buff-in"EXTN_SEP"rc5\" for the RC5 input buffer, \"buff-in"EXTN_SEP"des\"\n"
  "for the DES input buffer and \"buff-in"EXTN_SEP"ogr\" for the OGR input\n"
  "buffer. Note: if a path is not specified, the files will be created in the\n"
  "same directory as the .ini file, which is - by default - created in the same\n"
  "directory as rc5des itself.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//6
{ CFGTXT("Out-Buffer Filename Prefix"), "buff-out",
  CFGTXT(
  "Enter the prefix (the base name, ie a filename without an 'extension') of the\n"
  "buffer files where finished work will be stored. The default is \"buff-out\".\n"
  "The name of the project will concatenated internally to this base name name\n"
  "to construct the full name of the buffer file. Thus, \"buff-out\" will become\n"
  "\"buff-out"EXTN_SEP"rc5\" for the RC5 output buffer, \"buff-out"EXTN_SEP"des\"\n"
  "for the DES output buffer and \"buff-out"EXTN_SEP"ogr\" for the OGR output\n"
  "buffer. Note: if a path is not specified, the files will be created in the\n"
  "same directory as the .ini file, which is - by default - created in the same\n"
  "directory as rc5des itself.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//7
{ CFGTXT("Checkpoint Filename"),"",
  CFGTXT(
  "This option sets the location of the checkpoint file. The checkpoint is\n"
  "where the client writes its progress to disk so that it can recover partially\n"
  "completed work if the client had previously failed to shutdown normally.\n"
  "DO NOT SHARE CHECKPOINTS BETWEEN CLIENTS. Avoid the use of checkpoints unless\n"
  "your client is running in an environment where it might not be able to shutdown\n"
  "properly.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//8
{ CFGTXT("Alternate buffer directory"),"",
  CFGTXT(
  "When a client runs out of work to do and cannot fetch more work from a\n"
  "keyserver, it will fetch/flush from/to files in this directory.\n"
  "\n"
  "This option specifies a *directory*, and not a filename. The full paths\n"
  "effectively used are constructed from the name of the project and the\n"
  "filename component in the \"[Out|In]-Buffer Filename Prefix\" options.\n"
  "For example, if the \"In-Buffer Filename Prefix\" is \"~/buff-in\", and\n"
  "the alternate buffer directory is \"/there/\" then the alternate in-buffer-file\n"
  "for RC5 becomes \"/there/buff-in.rc5\"\n"
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},

/* ------------------------------------------------------------ */

//8
{ CFGTXT("Complete this many packets, then exit"), "0 (no limit)",
  CFGTXT(
  "This option specifies that you wish to have the client exit after it has\n"
  "crunched a predefined number of packets. Use 0 (zero) to apply 'no limit',\n"
  "or -1 to have the client exit when the input buffer is empty (this is the\n"
  "equivalent to the -runbuffers command line option.)\n"
  ),CONF_MENU_MISC,CONF_TYPE_INT,NULL,NULL,0,0,NULL}, /* no min max here */
//9
{ CFGTXT("Run for this long, then exit"), "0:00 (no limit)",
  CFGTXT(
  "This option specifies that you wish to have the client exit after it has\n"
  "crunched a predefined number of hours. Use 0:00 (or clear the field) to\n"
  "specify 'no limit'.\n"
  ),CONF_MENU_MISC,CONF_TYPE_TIMESTR,NULL,NULL,0,0,NULL},
//10
{ CFGTXT("Pausefile Path/Name"),"",
  CFGTXT(
  "While running, the client will occasionally look for the the presence of this\n"
  "file. If it exists, the client will immediately suspend itself and will continue\n"
  "to remain suspended as long as the file is present.\n"
  ),CONF_MENU_MISC,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//11
{ CFGTXT("Disable all screen output? (quiet mode)"),"0",
  CFGTXT(
  "When enabled, this option will cause the client to suppress all screen output\n"
  "and detach itself (run in the background). Because the client is essentially\n"
  "invisible, distributed.net strongly encourages the use of logging to file if\n"
  "you choose to run the client with disabled screen output. This option is\n"
  "synonymous with the -runhidden and -quiet command line switches and can be\n"
  "overridden with the -noquiet switch.\n"
  ),CONF_MENU_MISC,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL},
//12
{ CFGTXT("Disable exit file checking?"),"0",
  CFGTXT(
  "When disabled, this option will cause the client to watch for a file named\n"
  "\"exitrc5.now\", the presence of which being a request to the client to\n"
  "shut itself down. (The name of the exit flag file may be set in the ini.)\n"
  ),CONF_MENU_MISC,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL},
//13
{ CFGTXT("Disable the packet completion indicator?"),"0",
  CFGTXT(""
  ),CONF_MENU_MISC,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL},
//14
{ CFGTXT("Project Priority"), "DES,CSC,OGR,RC5",
  CFGTXT(
  "Enter the order in which the client will search for active projects, for\n"
  "instance \"DES,RC5\" specifies that DES packets (if available) will be\n"
  "crunched before RC5 packets. Leaving out a project name does *not* disable\n"
  "it. To disable a project, append \"=0\" to project's name. For example,\n"
  "\"DES,OGR=0,RC5\" will disable the client's OGR support. Project names not\n"
  "found on the list when the client starts will be inserted automatically\n"
  "according to their default priority. Thus, specifying \"DES,OGR\" is\n"
  "equivalent to specifying \"DES,CSC,OGR,RC5\", and \"OGR,DES\" is equivalent\n"
  "to \"CSC,OGR,DES,RC5\".\n"
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},

/* ------------------------------------------------------------ */

//15
{ CFGTXT("Processor type"), "-1 (autodetect)",
  CFGTXT(
  "This option determines which processor the client will optimize operations\n"
  "for.  While auto-detection is preferrable for most processor families, you may\n"
  "wish to set the processor type manually if detection fails or your machine's\n"
  "processor is detected incorrectly.\n"
  ),CONF_MENU_PERF,CONF_TYPE_INT,NULL,NULL,0,0,NULL},
//16
{ CFGTXT("Number of processors available"), "-1 (autodetect)",
#if (CLIENT_OS == OS_RISCOS)
  "This option specifies the number of threads you want the client to work on.\n"
  "On multi-processor machines this should be set to the number of processors\n"
  "available or to -1 to have the client attempt to auto-detect the number of\n"
  "processors. Multi-threaded clients can be forced to run single-threaded by\n"
  "setting this option to zero.\n"
  "Under RISC OS, processor 1 is the ARM, and processor 2 is an x86 processor\n"
  "card, if fitted.\n"
#else
  CFGTXT(
  "This option specifies the number of threads you want the client to work on.\n"
  "On multi-processor machines this should be set to the number of processors\n"
  "available or to -1 to have the client attempt to auto-detect the number of\n"
  "processors. Multi-threaded clients can be forced to run single-threaded by\n"
  "setting this option to zero.\n"
  )
#endif
  ,CONF_MENU_PERF,CONF_TYPE_INT,NULL,NULL,-1,128,NULL},
//17
{ CFGTXT("Priority level to run at"), "0 (lowest/idle)",
#if (CLIENT_OS == OS_NETWARE)
  CFGTXT(
  "The priority option is ignored on this machine. The distributed.net client\n"
  "for NetWare dynamically adjusts its process priority.\n"
  )
#elif (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN32S)
  CFGTXT(
  "The priority option is ignored on this machine. distributed.net clients\n"
  "for Windows always run at lowest ('idle') priority.\n"
  "This does not mean that the client will run slower than at a higher\n"
  "priority. It simply means that all other processes have a better chance\n"
  "to get processor time than client. If none of them want/need processor\n"
  "time, the client will get it.\n"
  )
#elif (CLIENT_OS == OS_RISCOS) || (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_MACOS)
  CFGTXT(
  "The priority option is ignored on this machine. The distributed.net client\n"
  "for "CLIENT_OS_NAME" dynamically adjusts its process priority.\n"
  )
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
  )
#endif
  ,CONF_MENU_PERF, CONF_TYPE_INT, NULL, NULL, 0, 9, NULL },

/* ------------------------------------------------------------ */

//18
{ CFGTXT("File to log to"), "",
  CFGTXT(
  "To enable logging to file you must specify the name of a logfile. The filename\n"
  "is limited a length of 128 characters and may not contain spaces. The file\n"
  "will be created to be in the client's directory unless a path is specified.\n"
  ),CONF_MENU_LOG,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//19
{ CFGTXT("Log by mail spool size (bytes)"), "0 (mail disabled)",
  CFGTXT(
  "The client is capable of sending you a log of the client's progress by mail.\n"
  "To activate this capability, specify how much you want the client to buffer\n"
  "before sending. The minimum is 2048 bytes, the maximum is approximately 130000\n"
  "bytes. Specify 0 (zero) to disable logging by mail.\n"
  ),CONF_MENU_LOG,CONF_TYPE_INT,NULL,NULL,0,125000,NULL},
//20
{ CFGTXT("SMTP Server to use"), "",
  CFGTXT(
  "Specify the name or DNS address of the SMTP host via which the client should\n"
  "relay mail logs. The default is the hostname component of the email address from\n"
  "which logs will be mailed.\n"
  ),CONF_MENU_LOG,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//21
{ CFGTXT("SMTP Port"), "25 (default)",
  CFGTXT(
  "Specify the port on the SMTP host to which the client's mail subsystem should\n"
  "connect when sending mail logs. The default is port 25.\n"
  ),CONF_MENU_LOG,CONF_TYPE_INT,NULL,NULL,1,0xFFFF,NULL},
//22
{ CFGTXT("E-mail address that logs will be mailed from"),
  "" /* *((const char *)(options[CONF_ID].thevariable)) */,
  CFGTXT(
  "(Some servers require this to be a real address)\n"
  ),CONF_MENU_LOG,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//23
{ CFGTXT("E-mail address to send logs to"),
  "" /* *((const char *)(options[CONF_ID].thevariable)) */,
  CFGTXT(
  "Full name and site eg: you@your.site.  Comma delimited list permitted.\n"
  ),CONF_MENU_LOG,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},

/* ------------------------------------------------------------ */

//24
{ CFGTXT("Offline operation mode"),"0",
  CFGTXT(
  "Yes: The client will never connect to a keyserver.\n"
  " No: The client will connect to a keyserver as needed.\n"
  ),CONF_MENU_NET,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL},
//25
{ CFGTXT("Network Timeout (seconds)"), "60 (default)",
  CFGTXT(
  "This option determines the amount of time the client will wait for a network\n"
  "read or write acknowledgement before it assumes that the connection has been\n"
  "broken. Any value between 5 and 300 seconds is valid and setting the timeout\n"
  "to -1 forces a blocking connection.\n"
  ),CONF_MENU_NET,CONF_TYPE_INT,NULL,NULL,-1,300,NULL},
//26
{ CFGTXT("Firewall Protocol/Communications mode"), "0 (direct connection)",
  CFGTXT(
  "This option determines what protocol to use when communicating via a SOCKS\n"
  "or HTTP proxy, or optionally when communicating directly with a keyserver\n"
  "that is listening a telnet port. Specify 0 (zero) if you have a direct\n"
  "connection to either a personal proxy or to a distributed.net keyserver\n"
  "on the internet.\n"
  ),CONF_MENU_NET,CONF_TYPE_INT,NULL,CFGTXT(&uuehttptable[0]),0,5,NULL},
//27
{ CFGTXT("Automatically select a distributed.net keyserver?"), "1",
  CFGTXT(
  "Set this option to 'Yes' UNLESS your client will not be communicating\n"
  "with a personal proxy (instead of one of the main distributed.net\n"
  "keyservers) OR your client will be connecting through an HTTP proxy\n"
  "(firewall) and you have been explicitely advised by distributed.net\n"
  "staff to use a specific IP address.\n"
  ),CONF_MENU_NET,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL},
//28
{ CFGTXT("Keyserver hostname"), "",
  CFGTXT(
  "This is the name or IP address of the machine that your client will\n"
  "obtain keys from and send completed packets to. Avoid IP addresses\n"
  "if possible unless your client will be communicating through a HTTP\n"
  "proxy (firewall) and you have trouble fetching or flushing packets.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//29
{ CFGTXT("Keyserver port"), "0 (auto select)",
  CFGTXT(
  "This field determines which keyserver port the client should connect to.\n"
  "You should leave this at zero unless:\n"
  "a) You are connecting to a personal proxy is that is *not* listening on\n"
  "   port 2064.\n"
  "b) You are connecting to a keyserver (regardless of type: personal proxy\n"
  "   or distributed.net host) through a firewall, and the firewall does\n"
  "   *not* permit connections to port 2064.\n"
  "\n"
  "All keyservers (personal proxy as well as distributed.net hosts) accept\n"
  "all encoding methods (UUE, HTTP, raw) on any/all ports the listen on.\n"
  ),CONF_MENU_NET,CONF_TYPE_INT,NULL,NULL,0,0xFFFF,NULL},
//30
{ CFGTXT("Keyserver is a personal proxy on a protected LAN?"),"0",
  CFGTXT(
  "If the keyserver that your client will be connecting to is a personal\n"
  "proxy inside a protected LAN (inside a firewall), set this option to 'yes'.\n"
  "Otherwise leave it at 'No'.\n"
  ),CONF_MENU_NET,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL},
//31
{ CFGTXT("Firewall hostname"), "",
  CFGTXT(
  "This field determines the hostname or IP address of the firewall proxy\n"
  "through which the client should communicate. The proxy is expected to be\n"
  "on a local network.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//32
{ CFGTXT("Firewall port"), "" /* note: atol("")==0 */,
  CFGTXT(
  "This field determines the port number on the firewall proxy to which the\n"
  "the client should connect. The port number must be valid.\n"
  ),CONF_MENU_NET,CONF_TYPE_INT,NULL,NULL,1,0xFFFF,NULL},
//33
{ CFGTXT("Firewall username"), "",
  CFGTXT(
  "Specify a username in this field if your SOCKS host requires\n"
  "authentication before permitting communication through it.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//34
{ CFGTXT("Firewall password"), "",
  CFGTXT(
  "Specify the password in this field if your SOCKS host requires\n"
  "authentication before permitting communication through it.\n"
  ),CONF_MENU_NET,CONF_TYPE_PASSWORD,NULL,NULL,0,0,NULL},
//35
{ CFGTXT("Modem detection options"),"0",
  CFGTXT(
  "Normal mode: the client will send/receive packets only when it\n"
  "        empties the in buffer, hits the flush threshold, or the user\n"
  "        specifically requests a flush/fetch.\n"
  "Dial-up detection mode: This acts like mode 0, with the addition\n"
  "        that the client will automatically send/receive packets when a\n"
  "        dial-up networking connection is established. Modem users\n"
  "        will probably wish to use this option so that their client\n"
  "        never runs out of packets.\n"
  "Dial-up detection ONLY mode: Like the previous mode, this will cause\n"
  "        the client to automatically send/receive packets when\n"
  "        connected. HOWEVER, if the client runs out of packets,\n"
  "        it will NOT trigger auto-dial, and will instead work\n"
  "        on random packets until a connection is detected.\n"
  ),CONF_MENU_NET,CONF_TYPE_INT,NULL,CFGTXT(&lurkmodetable[0]),0,2,NULL},
//36
{ CFGTXT("Interfaces to watch"), "",
  CFGTXT(
  "Colon-separated list of interface names to monitor for a connection,\n"
  "For example: \"ppp0:ppp1:eth1\". Wildcards are permitted, ie \"ppp*\".\n"
  "1) An empty list implies all interfaces that are identifiable as dialup,\n"
  "   ie \"ppp*:sl*:...\" (dialup interface names vary from platform to\n"
  "   platform. FreeBSD for example, also includes 'dun*' interfaces).\n"
  "2) if you have an intermittent ethernet connection through which you can\n"
  "   access the Internet, put the corresponding interface name in this list,\n"
  "   typically 'eth0'\n"
  "3) To include all devices, as might be preferrable for portable computers\n"
  "   which access the Internet via a LAN in one location but via a modem\n"
  "   in another, set this option to '*'.\n"
  "The command line equivalent of this option is --interfaces-to-watch\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//37
{ /*dialwhenneeded*/ 
   #if (CLIENT_OS == OS_WIN32)
   CFGTXT("Use a specific DUN profile for connecting to the net?"),
   #elif ((CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32))
   CFGTXT("Load/unload Winsock to initiate/hangup net connections?"),
   #else
   CFGTXT("Use scripts to initiate/hangup dialup connections?"),
   #endif
   "0",CFGTXT(
   "Select 'yes' to have the client control how network connections\n"
   "are initiatiated if none is active.\n"
   ),CONF_MENU_NET,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL},
//38
{ CFGTXT("Dial-up Connection Profile"),"",
  #if (CLIENT_OS == OS_WIN32)
  CFGTXT("Select the DUN profile to use when dialing-as-needed.\n")
  #else
  CFGTXT("")
  #endif
  ,CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//39
{ CFGTXT("Command/script to start dialup"),"",
  CFGTXT(
  "Enter any valid shell command or script name to use to initiate a\n"
  "network connection. \"Dial the Internet as needed?\" must be enabled for\n"
  "this to be of any use.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL},
//40
{ CFGTXT("Command/script to stop dialup"),"",
  CFGTXT(
  "Enter any valid shell command or script name to use to shutdown a\n"
  "network connection previously initiated with the script/command specified\n"
  "in the \"Command/script to start dialup\" option.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL}
};

#if 0
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
  if ( strcmpi(string, "none") == 0 )
    string[0]=0;
  return;
}

// --------------------------------------------------------------------------
#endif
