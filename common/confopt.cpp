// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: confopt.cpp,v $
// Revision 1.27  1999/03/18 05:40:43  cyp
// oops. "Project priority" was in the wrong menu.
//
// Revision 1.26  1999/03/18 03:59:09  cyp
// new "Project priority" option.
//
// Revision 1.25  1999/03/02 04:26:39  foxyloxy
// Upped the maximum for preferred block size to 33.
//
// Revision 1.24  1999/02/20 03:07:17  gregh
// Add OGR options to configuration data.
//
// Revision 1.23  1999/02/09 23:41:39  cyp
// Lurk iface mask changes: a) default iface mask no longer needs to be known
// outside lurk; b) iface mask now supports wildcards; c) redid help text.
//
// Revision 1.22  1999/02/09 03:24:34  remi
// Reverted the previous patch. connifacemask default is now set in confmenu.cpp.
//
// Revision 1.21  1999/02/08 23:19:39  remi
// The right default for interface-to-watch is "ppp0:sl0" not "\0"
// (at least on Linux).
// FreeBSD now supports lurk mode also.
//
// Revision 1.20  1999/02/07 16:00:08  cyp
// Lurk changes: genericified variable names, made less OS-centric.
//
// Revision 1.19  1999/02/06 10:42:55  remi
// - the default for dialup.ifacestowatch is now 'ppp0:sl0'.
// - #ifdef'ed dialup.ifacestowatch (only Linux at the moment)
// - modified a bit the help text in confopt.cpp
//
// Revision 1.18  1999/02/06 09:08:08  remi
// Enhanced the lurk fonctionnality on Linux. Now it use a list of interfaces
// to watch for online/offline status. If this list is empty (the default), any
// interface up and running (besides the lookback one) will trigger the online
// status.
//
// Revision 1.17  1999/02/04 10:44:19  cyp
// Added support for script-driven dialup. (currently linux only)
//
// Revision 1.16  1999/01/29 18:59:52  jlawson
// fixed formatting.
//
// Revision 1.15  1999/01/29 01:25:59  cyp
// permitting nettimeout=-1 got lost in one of the last two revs.
//
// Revision 1.14  1999/01/15 00:32:44  cyp
// changed phrasing of 'distributed.net ID' at Nugget's request.
//
// Revision 1.13  1999/01/13 15:17:02  kbracey
// Fixes to RISC OS processor detection and scheduling
//
// Revision 1.12  1999/01/12 14:57:35  cyp
// -1 is a legal nettimeout value (force blocking net i/o).
//
// Revision 1.11  1999/01/04 02:47:30  cyp
// Cleaned up menu options and handling.
//
// Revision 1.9  1998/12/23 00:41:45  silby
// descontestclosed and scheduledupdatetime now read from the .ini file.
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
return "@(#)$Id: confopt.cpp,v 1.27 1999/03/18 05:40:43 cyp Exp $"; }
#endif

#include "cputypes.h" // CLIENT_OS, s32
#include "baseincs.h" // strcmp() etc as used by isstringblank() et al.
#include "cmpidefs.h" // strcmpi()
#include "client.h"   // only the MAXBLOCKSPERBUFFER define
#include "confopt.h"  // ourselves
#include "pathwork.h" // EXTN_SEP

// --------------------------------------------------------------------------

#if defined(NOCONFIG)
  #define CFGTXT(x) NULL
#else
  #define CFGTXT(x) x
#endif

// --------------------------------------------------------------------------

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
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,1,NULL,NULL,0,0},
//1
{  CFGTXT("Buffer packets in RAM only? (no disk I/O)"),"0",
   CFGTXT(
   "This option is for machines with permanent connections to a keyserver\n"
   "but without local disks. Note: This option will cause all buffered,\n"
   "unflushable packets to be lost by a client shutdown.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_BOOL,2,NULL,NULL,0,1},
//2
{ CFGTXT("Frequently check for empty buffers?"),"0",
  CFGTXT(
  "Enabling this option will cause the client to check the input buffers\n"
  "every few minutes or so. You might want to use this if you have a\n"
  "single computer with a network connecting \"feeding\" other clients via\n"
  "a common buff-in.* file so that the buffer never reaches empty.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_BOOL,3,NULL,NULL,0,1},
//3
{ CFGTXT("Preferred RC5/DES packet size (2^X keys/packet)"),"31 (default)",
  CFGTXT(
  "When fetching RC5 or DES packets from a keyserver, the client will request\n"
  "packets with the size you specify in this option. Running the client with\n"
  "the -benchmark switch will give you a hint as to what the best packet size\n"
  "for this machine might be. Packet sizes are specified as powers of 2.\n"
  "The minimum and maximum packet sizes are 28 and 33 respectively.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_INT,4,NULL,NULL,28,33},
//4
{ CFGTXT("Packet fetch/flush threshold"), "10 (default)",
  CFGTXT(
  "This option specifies how many packets your client will buffer between\n"
  "communications with a keyserver. The client operates directly on packets\n"
  "stored in the input buffer, and puts finished packets into the output buffer.\n"
  "When the number of packets in the input buffer reaches 0, the client will\n"
  "attempt to connect to a keyserver, fill the input buffer to the threshold,\n"
  "and send in all completed packets. Keep the number of packets to buffer low\n"
  "(10 or less) if you have a fixed (static) connection to the internet. If\n"
  "you use a dial-up connection buffer as many packets as you would complete\n"
  "in a day (running the client with -benchmark will give you a hint as what\n"
  "might be accomplished by this machine in one day). You may also force a\n"
  "buffer exchange by starting the client with the -update option.  Do not\n"
  "buffer more than what might be accomplished in one week; you might not\n"
  "receive credit for them.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_INT,5,NULL,NULL,1,MAXBLOCKSPERBUFFER},
//5
{ CFGTXT("RC5 In-Buffer Path/Name"),  "buff-in"  EXTN_SEP "rc5",
  CFGTXT(""
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,6,NULL,NULL,0,0},
//6
{ CFGTXT("RC5 Out-Buffer Path/Name"), "buff-out" EXTN_SEP "rc5",
  CFGTXT(""
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,7,NULL,NULL,0,0},
//7
{ CFGTXT("DES In-Buffer Path/Name"),  "buff-in"  EXTN_SEP "des",
  CFGTXT(""
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,9,NULL,NULL,0,0},
//8
{ CFGTXT("DES Out-Buffer Path/Name"), "buff-out" EXTN_SEP "des",
  CFGTXT(""
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,10,NULL,NULL,0,0},
//9
{ CFGTXT("OGR In-Buffer Path/Name"),  "buff-in"  EXTN_SEP "ogr",
  CFGTXT(""
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,11,NULL,NULL,0,0},
//10
{ CFGTXT("OGR Out-Buffer Path/Name"), "buff-out" EXTN_SEP "ogr",
  CFGTXT(""
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,12,NULL,NULL,0,0},
//11
{ CFGTXT("Checkpoint Filename"),"",
  CFGTXT(
  "This option sets the location of the checkpoint file. The checkpoint is\n"
  "where the client writes its progress to disk so that it can recover partially\n"
  "completed work if the client had previously failed to shutdown normally.\n"
  "DO NOT SHARE CHECKPOINTS BETWEEN CLIENTS. Avoid the use of checkpoints unless\n"
  "your client is running in an environment where it might not be able to shutdown\n"
  "properly.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,13,NULL,NULL,0,0},

/* ------------------------------------------------------------ */

//12
{ CFGTXT("Complete this many packets, then exit"), "0 (no limit)",
  CFGTXT(
  "This option specifies that you wish to have the client exit after it has\n"
  "crunched a predefined number of packets. Use 0 (zero) to apply 'no limit',\n"
  "or -1 to have the client exit when the input buffer is empty (this is the\n"
  "equivalent to the -runbuffers command line option.)\n"
  ),CONF_MENU_MISC,CONF_TYPE_INT,1,NULL,NULL,0,0}, /* no min max here */
//13
{ CFGTXT("Run for this long, then exit"), "0:00 (no limit)",
  CFGTXT(
  "This option specifies that you wish to have the client exit after it has\n"
  "crunched a predefined number of hours. Use 0:00 (or clear the field) to\n"
  "specify 'no limit'.\n"
  ),CONF_MENU_MISC,CONF_TYPE_TIMESTR,2,NULL,NULL,0,0},
//14
{ CFGTXT("Pausefile Path/Name"),"",
  CFGTXT(""
  ),CONF_MENU_MISC,CONF_TYPE_ASCIIZ,3,NULL,NULL,0,0},
//15
{ CFGTXT("Disable all screen output? (quiet mode)"),"0",
  CFGTXT(
  "When enabled, this option will cause the client to suppress all screen output\n"
  "and detach itself (run in the background). Because the client is essentially\n"
  "invisible, distributed.net strongly encourages the use of logging to file if\n"
  "you choose to run the client with disabled screen output. This option is\n"
  "synonymous with the -runhidden and -quiet command line switches and can be\n"
  "overridden with the -noquiet switch.\n"
  ),CONF_MENU_MISC,CONF_TYPE_BOOL,4,NULL,NULL,0,1},
//16
{ CFGTXT("Disable exit file checking?"),"0",
  CFGTXT(
  "When disabled, this option will cause the client to watch for a file named\n"
  "\"exitrc5.now\", the presence of which being a request to the client to\n"
  "shut itself down. (The name of the exit flag file may be set in the ini.)\n"
  ),CONF_MENU_MISC,CONF_TYPE_BOOL,5,NULL,NULL,0,1},
//17
{ CFGTXT("Disable the packet completion indicator?"),"0",
  CFGTXT(""
  ),CONF_MENU_MISC,CONF_TYPE_BOOL,6,NULL,NULL,0,1},
//18
{ CFGTXT("Project Priority"), "DES,OGR,RC5",
  CFGTXT(
  "Enter the order in which the client will search for active projects,\n"
  "for instance \"DES,OGR,RC5\" specifies that DES packets (if available) will\n"
  "be crunched before OGR or RC5 packets.\n"
  ),CONF_MENU_MISC,CONF_TYPE_ASCIIZ,7,NULL,NULL,0,0},

/* ------------------------------------------------------------ */

//19
{ CFGTXT("Processor type"), "-1 (autodetect)",
  CFGTXT(
  "This option determines which processor the client will optimize operations\n"
  "for.  While auto-detection is preferrable for most processor families, you may\n"
  "wish to set the processor type manually if detection fails or your machine's\n"
  "processor is detected incorrectly.\n"
  ),CONF_MENU_PERF,CONF_TYPE_INT,1,NULL,NULL,0,0},
//20
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
  ,CONF_MENU_PERF,CONF_TYPE_INT,2,NULL,NULL,-1,128},
//21
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
  ,CONF_MENU_PERF, CONF_TYPE_INT, 3 /*menupos*/, NULL, NULL, 0, 9 },

/* ------------------------------------------------------------ */

//22
{ CFGTXT("File to log to"), "",
  CFGTXT(
  "To enable logging to file you must specify the name of a logfile. The filename\n"
  "is limited a length of 128 characters and may not contain spaces. The file\n"
  "will be created to be in the client's directory unless a path is specified.\n"
  ),CONF_MENU_LOG,CONF_TYPE_ASCIIZ,1,NULL,NULL,0,0},
//23
{ CFGTXT("Log by mail spool size (bytes)"), "0 (mail disabled)",
  CFGTXT(
  "The client is capable of sending you a log of the client's progress by mail.\n"
  "To activate this capability, specify how much you want the client to buffer\n"
  "before sending. The minimum is 2048 bytes, the maximum is approximately 130000\n"
  "bytes. Specify 0 (zero) to disable logging by mail.\n"
  ),CONF_MENU_LOG,CONF_TYPE_INT,2,NULL,NULL,0,125000},
//24
{ CFGTXT("SMTP Server to use"), "",
  CFGTXT(
  "Specify the name or DNS address of the SMTP host via which the client should\n"
  "relay mail logs. The default is the hostname component of the email address from\n"
  "which logs will be mailed.\n"
  ),CONF_MENU_LOG,CONF_TYPE_ASCIIZ,3,NULL,NULL,0,0},
//25
{ CFGTXT("SMTP Port"), "25 (default)",
  CFGTXT(
  "Specify the port on the SMTP host to which the client's mail subsystem should\n"
  "connect when sending mail logs. The default is port 25.\n"
  ),CONF_MENU_LOG,CONF_TYPE_INT,4,NULL,NULL,1,0xFFFF},
//26
{ CFGTXT("E-mail address that logs will be mailed from"),
  "" /* *((const char *)(options[CONF_ID].thevariable)) */,
  CFGTXT(
  "(Some servers require this to be a real address)\n"
  ),CONF_MENU_LOG,CONF_TYPE_ASCIIZ,5,NULL,NULL,0,0},
//27
{ CFGTXT("E-mail address to send logs to"),
  "" /* *((const char *)(options[CONF_ID].thevariable)) */,
  CFGTXT(
  "Full name and site eg: you@your.site.  Comma delimited list permitted.\n"
  ),CONF_MENU_LOG,CONF_TYPE_ASCIIZ,6,NULL,NULL,0,0},

/* ------------------------------------------------------------ */

//28
{ CFGTXT("Offline operation mode"),"0",
  CFGTXT(
  "Yes: The client will never connect to a keyserver.\n"
  " No: The client will connect to a keyserver as needed.\n"
  ),CONF_MENU_NET,CONF_TYPE_BOOL,1,NULL,NULL,0,1},
//29
{ CFGTXT("Network Timeout (seconds)"), "60 (default)",
  CFGTXT(
  "This option determines the amount of time the client will wait for a network\n"
  "read or write acknowledgement before it assumes that the connection has been\n"
  "broken. Any value between 5 and 300 seconds is valid and setting the timeout\n"
  "to -1 forces a blocking connection.\n"
  ),CONF_MENU_NET,CONF_TYPE_INT,2,NULL,NULL,-1,300},
//30
{ CFGTXT("Firewall Protocol/Communications mode"), "0 (direct connection)",
  CFGTXT(
  "This option determines what protocol to use when communicating via a SOCKS\n"
  "or HTTP proxy, or optionally when communicating directly with a keyserver\n"
  "that is listening a telnet port. Specify 0 (zero) if you have a direct\n"
  "connection to either a personal proxy or to a distributed.net keyserver\n"
  "on the internet.\n"
  ),CONF_MENU_NET,CONF_TYPE_INT,3,NULL,CFGTXT(&uuehttptable[0]),0,5},
//31
{ CFGTXT("Automatically select a distributed.net keyserver?"), "1",
  CFGTXT(
  "Set this option to 'Yes' UNLESS your client will not be communicating\n"
  "with a personal proxy (instead of one of the main distributed.net\n"
  "keyservers) OR your client will be connecting through an HTTP proxy\n"
  "(firewall) and you have been explicitely advised by distributed.net\n"
  "staff to use a specific IP address.\n"
  ),CONF_MENU_NET,CONF_TYPE_BOOL,4,NULL,NULL,0,1},
//32
{ CFGTXT("Keyserver to communicate with"), "",
  CFGTXT(
  "This is the name or IP address of the machine that your client will\n"
  "obtain keys from and send completed packets to. Avoid IP addresses\n"
  "if possible unless your client will be communicating through a HTTP\n"
  "proxy (firewall) and you have trouble fetching or flushing packets.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,5,NULL,NULL,0,0},
//33
{ CFGTXT("Keyserver port to connect to"), "0 (auto select)",
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
  ),CONF_MENU_NET,CONF_TYPE_INT,6,NULL,NULL,0,0xFFFF},
//34
{ CFGTXT("Keyserver is a personal proxy on a protected LAN?"),"0",
  CFGTXT(
  "If the keyserver that your client will be connecting to is a personal\n"
  "proxy inside a protected LAN (inside a firewall), set this option to 'yes'.\n"
  "Otherwise leave it at 'No'.\n"
  ),CONF_MENU_NET,CONF_TYPE_BOOL,7,NULL,NULL,0,1},
//35
{ CFGTXT("HTTP/SOCKS proxy hostname"), "proxy.mydomain.com",
  CFGTXT(
  "This field determines the hostname or IP address of the firewall proxy\n"
  "through which the client should communicate. The proxy is expected to be\n"
  "on a local network.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,8,NULL,NULL,0,0},
//36
{ CFGTXT("HTTP/SOCKS proxy port"), "" /* note: atol("")==0 */,
  CFGTXT(
  "This field determines the port number on the firewall proxy to which the\n"
  "the client should connect. The port number must be valid.\n"
  ),CONF_MENU_NET,CONF_TYPE_INT,9,NULL,NULL,1,0xFFFF},
//37
{ CFGTXT("HTTP/SOCKS proxy username"), "",
  CFGTXT(
  "Specify a username in this field if your SOCKS host requires\n"
  "authentication before permitting communication through it.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,10,NULL,NULL,0,0},
//38
{ CFGTXT("HTTP/SOCKS5 proxy password"), "",
  CFGTXT(
  "Specify the password in this field if your SOCKS host requires\n"
  "authentication before permitting communication through it.\n"
  ),CONF_MENU_NET,CONF_TYPE_PASSWORD,11,NULL,NULL,0,0},
//39
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
  ),CONF_MENU_NET,CONF_TYPE_INT,12,NULL,CFGTXT(&lurkmodetable[0]),0,2},
//40
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
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,13,NULL,NULL,0,0},
//41
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
   ),CONF_MENU_NET,CONF_TYPE_BOOL,14,NULL,NULL,0,1},
//42
{ CFGTXT("Dial-up Connection Profile"),"",
  #if (CLIENT_OS == OS_WIN32)
  CFGTXT("Select the DUN profile to use when dialing-as-needed.\n")
  #else
  CFGTXT("")
  #endif
  ,CONF_MENU_NET,CONF_TYPE_ASCIIZ,15,NULL,NULL,0,0},
//43
{ CFGTXT("Command/script to start dialup"),"",
  CFGTXT(
  "Enter any valid shell command or script name to use to initiate a\n"
  "network connection. \"Dial the Internet as needed?\" must be enabled for\n"
  "this to be of any use.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,16,NULL,NULL,0,0},
//44
{ CFGTXT("Command/script to stop dialup"),"",
  CFGTXT(
  "Enter any valid shell command or script name to use to shutdown a\n"
  "network connection previously initiated with the script/command specified\n"
  "in the \"Command/script to start dialup\" option.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,17,NULL,NULL,0,0}
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
  if ( strcmpi(string, "none") == 0 )
    string[0]=0;
  return;
}

// --------------------------------------------------------------------------

