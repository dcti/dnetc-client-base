/* 
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *confopt_cpp(void) {
return "@(#)$Id: confopt.cpp,v 1.34.2.18 2000/01/28 07:41:37 mfeiri Exp $"; }

/* ----------------------------------------------------------------------- */

#include "cputypes.h" // CLIENT_OS
#include "pathwork.h" // EXTN_SEP
#include "baseincs.h" // NULL
#include "client.h"   // BUFTHRESHOLD_MAX etc
#include "confopt.h"  // ourselves

/* ----------------------------------------------------------------------- */

static const char *lurkmodetable[] =
{
  "Normal mode",
  "Dial-up detection mode",
  "Dial-up detection ONLY mode"
};

// --------------------------------------------------------------------------

#define CFGTXT(x) (x)

struct optionstruct conf_options[] = //CONF_OPTION_COUNT=
{
//0
{ CFGTXT("General Client Options"),"",
  CFGTXT(""),CONF_MENU_MAIN,CONF_TYPE_MENU,NULL,NULL,CONF_MENU_MISC,0,NULL,NULL},
//1
{ CFGTXT("Your email address (distributed.net ID)"), "",
  CFGTXT(
  "Completed work sent back to distributed.net are tagged with the email\n"
  "address of the person whose machine completed that work. That address\n"
  "is used as a unique 'account' identifier in three ways: (a) this is how\n"
  "distributed.net will contact the owner of the machine that submits the\n"
  "winning key; (b) The owner of that address receives credit for completed\n"
  "work which may then be transferred to a team account; (c) The number of\n"
  "work-units completed may be used as votes in the selection of a recipient of\n"
  "the prize-money reserved for a non-profit organization.\n"
  ),CONF_MENU_MISC,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//2
{ CFGTXT("Complete this many packets, then exit"), "0 (no limit)",
  CFGTXT(
  "This option specifies that you wish to have the client exit after it has\n"
  "crunched a predefined number of packets. Use 0 (zero) to apply 'no limit',\n"
  "or -1 to have the client exit when the input buffer is empty (this is the\n"
  "equivalent to the -runbuffers command line option.)\n"
  ),CONF_MENU_MISC,CONF_TYPE_INT,NULL,NULL,0,0,NULL,NULL}, /* no min max here */
//3
{ CFGTXT("Run for this long, then exit"), "0:00 (no limit)",
  CFGTXT(
  "This option specifies that you wish to have the client exit after it has\n"
  "crunched a predefined number of hours. Use 0:00 (or clear the field) to\n"
  "specify 'no limit'.\n"
  ),CONF_MENU_MISC,CONF_TYPE_TIMESTR,NULL,NULL,0,0,NULL,NULL},
//4
{ CFGTXT("Pausefile Path/Name"),"",
  CFGTXT(
  "While running, the client will occasionally look for the the presence of\n"
  "this file. If it exists, the client will immediately suspend itself and\n"
  "will continue to remain suspended as long as the file is present.\n"
  ),CONF_MENU_MISC,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//5
{ CFGTXT("Disable all screen output? (quiet mode)"),"0",
  CFGTXT(
  "When enabled, this option will cause the client to suppress all screen\n"
  "output and detach itself (run in the background). Because the client is\n"
  "essentially invisible, distributed.net strongly encourages the use of\n"
  "logging to file if you choose to run the client with disabled screen\n"
  "output. This option is synonymous with the -hide and -quiet command line\n"
  "switches and can be overridden with the -noquiet switch.\n"
  ),CONF_MENU_MISC,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL,NULL},
//6
{ CFGTXT("Disable exit file checking?"),"0",
  CFGTXT(
  "When disabled, this option will cause the client to watch for a file named\n"
  "\"exitrc5.now\" the presence of which being a request to the client to\n"
  "shut itself down.\n"
  ),CONF_MENU_MISC,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL,NULL},
//7
{ CFGTXT("Disable the packet completion indicator?"),"0",
  CFGTXT(""
  ),CONF_MENU_MISC,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL,NULL},
//8
{ CFGTXT("Project order"/*"Load-work precedence"*/), "DES,CSC,OGR,RC5", 
  /* CFGTXT( */
  "The client looks for work in the order specified here. For example, \"OGR,\n"
  "RC5\" instructs the client to work on OGR until those buffers are exhausted;\n"
  "afterwards, it works on RC5. If *all* 'flush thresholds' are at -1 (default)\n"
  "then the client will obtain new work from the network only when all buffers\n"
  "are empty; ie it will rotate through the list.\n"
  "\n"
  "You can turn off a project by setting \":0\" or \"=0\" after the project's\n"
  "name - for instance, \"OGR:0\" tells your client not to work on, or request\n"
  "for, the OGR project.\n"
  "\n"
  "Projects not found in the list you enter here will be inserted in their\n"
  "default position.\n"
  "\n"
#if 0  
  "Please note: when DES is active & enabled, the client will clear input\n"
  "buffers for all other projects, thus ensuring that clients sharing those\n"
  "buffer files do not inadvertantly work on the \"wrong\" project for the\n"
  "few hours DES is active.\n"
#endif  
  /*) */,CONF_MENU_MISC,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},

/* ------------------------------------------------------------ */

//9
{ CFGTXT("Buffer and Buffer Update Options"),"",
  CFGTXT(""),CONF_MENU_MAIN,CONF_TYPE_MENU,NULL,NULL,CONF_MENU_BUFF,0,NULL},
//10
{  CFGTXT("Buffer in RAM only? (no disk I/O)"),"0",
   CFGTXT(
   "This option is for machines with permanent connections to a keyserver\n"
   "but without local disks. Note: This option will cause all buffered,\n"
   "unflushable work to be lost by a client shutdown.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL,NULL},
//11
{ CFGTXT("In-Buffer Filename Prefix"), BUFFER_DEFAULT_IN_BASENAME,
  CFGTXT(
  "Enter the prefix (the base name, ie a filename without an 'extension') of\n"
  "the buffer files where unfinished work will be stored. The default is\n"
  "\""BUFFER_DEFAULT_IN_BASENAME"\". The name of the project will be concatenated\n"
  "internally to this base name to construct the full name of the buffer\n"
  "file. For example, \""BUFFER_DEFAULT_IN_BASENAME"\" becomes \""BUFFER_DEFAULT_IN_BASENAME""EXTN_SEP"rc5\"\n"
  "for the RC5 input buffer\n"
  "Note: if a path is not specified, the files will be created in the same\n"
  "directory as the .ini file, which, by default, is created in the same\n"
  "directory as the client itself.\n"
  "(A new buffer file format is forthcoming. The new format will have native\n"
  "support for First-In-First-Out packets (this functionality is currently\n"
  "available but is not efficient when used with large buffers); improved\n"
  "locking semantics; all buffers for all projects will be contained in a\n"
  "single file).\n"
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//12
{ CFGTXT("Out-Buffer Filename Prefix"), BUFFER_DEFAULT_OUT_BASENAME,
  CFGTXT(
  "Enter the prefix (the base name, ie a filename without an 'extension') of\n"
  "the buffer files where finished work will be stored. The default is\n"
  "\""BUFFER_DEFAULT_OUT_BASENAME"\". The name of the project will be concatenated\n"
  "internally to this base name to construct the full name of the buffer\n"
  "file. For example, \""BUFFER_DEFAULT_OUT_BASENAME"\" becomes \""BUFFER_DEFAULT_OUT_BASENAME""EXTN_SEP"rc5\"\n"
  "for the RC5 output buffer\n"
  "Note: if a path is not specified, the files will be created in the same\n"
  "directory as the .ini file, which, by default, is created in the same\n"
  "directory as the client itself.\n"
  "(This option will eventually disappear. Refer to the \"In-Buffer\n"
  "Filename Prefix\" option for details).\n"
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//13
{ CFGTXT("Checkpoint Filename"),"",
  CFGTXT(
  "This option sets the location of the checkpoint file. The checkpoint is\n"
  "where the client writes its progress to disk so that it can recover\n"
  "partially completed work if the client had previously failed to shutdown\n"
  "normally.\n"
  "DO NOT SHARE CHECKPOINTS BETWEEN CLIENTS. Avoid the use of checkpoints\n"
  "unless your client is running in an environment where it might not be able\n"
  "to shutdown properly.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//14
{ CFGTXT("Disable buffer updates from/to a keyserver"),"0",
  CFGTXT(
  "Yes: The client will never connect to a keyserver.\n"
  " No: The client will connect to a keyserver as needed.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL,NULL},
//15
{ CFGTXT("Keyserver<->client connectivity options"),"",
  CFGTXT(""),CONF_MENU_BUFF,CONF_TYPE_MENU,NULL,NULL,CONF_MENU_NET,0,NULL,NULL},
//16
{ CFGTXT("Disable buffer updates from/to remote buffers"),"0",
  CFGTXT(
  "Yes: The client will not use remote files.\n"
  " No: The client will use remote files if updating from a keyserver is\n"
  "     disabled or it fails or if insufficient packets were sent/received.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL,NULL},
//17
{ CFGTXT("Remote buffer directory"),"",
  CFGTXT(
  "When a client runs out of work to do and cannot fetch more work from a\n"
  "keyserver, it will fetch/flush from/to files in this directory.\n"
  "\n"
  "This option specifies a *directory*, and not a filename. The full paths\n"
  "effectively used are constructed from the name of the project and the\n"
  "filename component in the \"[Out|In]-Buffer Filename Prefix\" options.\n"
  "For example, if the \"In-Buffer Filename Prefix\" is \"~/"BUFFER_DEFAULT_IN_BASENAME"\", and\n"
  "the alternate buffer directory is \"/there/\" then the alternate in-buffer-file\n"
  "for RC5 becomes \"/there/"BUFFER_DEFAULT_IN_BASENAME".rc5\"\n"
  ),CONF_MENU_BUFF,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//18
{ CFGTXT("Frequently update empty buffers?"),"0",
  CFGTXT(
  "Enabling this option will cause the client to check the input buffers\n"
  "every few minutes or so. You might want to use this if you have a\n"
  "single computer with a network connection \"feeding\" other clients via\n"
  "a common input file.\n"
  "Note: enabling (modem-) connection detection implies that buffers will\n"
  "be updated frequently while a connection is detected.\n" 
  ),CONF_MENU_BUFF,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL,NULL},
//19
{ CFGTXT("Preferred packet size (2^X keys/packet)"), "-1 (auto)",
  /* CFGTXT( */
  "When fetching key-based packets from a server, the client will request\n"
  "packets with the size you specify in this option. Packet sizes are\n"
  "specified as powers of 2.\n"
  "The minimum and maximum packet sizes are " _TEXTIFY(PREFERREDBLOCKSIZE_MIN) " and " _TEXTIFY(PREFERREDBLOCKSIZE_MAX) " respectively,\n"
  "and specifying '-1' permits the client to use internal defaults.\n"
  "Note: the number you specify is the *preferred* size. Although the\n"
  "keyserver will do its best to serve that size, there is no guarantee that\n"
  "it will always do so.\n"
  #if (PREFERREDBLOCKSIZE_MAX > 31)
  "*Warning*: clients older than v2.7106 do not know how to deal with packets\n"
  "larger than 2^31 keys. Do not share buffers with such a client if you set\n"
  "the preferred packet size to a value greater than 31.\n"
  #endif
  /*)*/,CONF_MENU_BUFF,CONF_TYPE_IARRAY,NULL,NULL,PREFERREDBLOCKSIZE_MIN,PREFERREDBLOCKSIZE_MAX,NULL,NULL},
//20
{ CFGTXT("Fetch:flush work threshold"), "-1 (default size or determine from time threshold)",
  CFGTXT(
  "This option specifies how many work units your client will buffer between\n"
  "communications with a keyserver. When the number of work units in the\n"
  "input buffer reaches 0, the client will attempt to connect to a keyserver,\n"
  "fill the input buffer to the threshold, and send in all completed work\n"
  "units. Keep the number of workunits to buffer low if you have a fixed\n"
  "connection to the internet, or the cost of your dialup connection is\n"
  "negligible.\n"
  "\n"
  "Thresholds as displayed here are in the form \"fetch:flush\":\n"
  "A value of -1 for the 'fetch setting' indicates that a time threshold should\n"
  "be used instead. If that too is unspecified, then the client will use defaults.\n"
  "A value of -1 for the 'flush setting' indicates that the client is not to\n"
  "monitor the level of the output buffer. If all flush settings are -1, the\n"
  "client will update buffers only when all input buffers are empty.\n"
  ),CONF_MENU_BUFF,CONF_TYPE_IARRAY,NULL,NULL,1,BUFTHRESHOLD_MAX,NULL,NULL},
//21
{ CFGTXT("Fetch:flush time threshold (in hours)"), "0 (disabled)",
  "This option specifies that instead of fetching a specific number of\n"
  "work units from the keyservers, enough work units should be downloaded\n"
  "to keep your client busy for a specified number of hours.  This causes\n"
  "the work unit threshold option to be constantly recalculated based on the\n"
  "current speed of your client on your machine.\n\n"
  "For fixed (static) connections, you should set this to a low value, like\n"
  "three to six hours.  For dialup connections, set this based on how often\n"
  "you connect to the network.\n"
#ifdef HAVE_OGR_CORES
  "\nCurrently not implemented for OGR because the amount of work in an\n"
  "unprocessed packet cannot be predicted.\n\n"
#endif
  ,CONF_MENU_BUFF,CONF_TYPE_IARRAY,NULL,NULL,0,336,NULL,NULL},
/* ------------------------------------------------------------ */

//22
{ CFGTXT("Performance related options"),"",
  CFGTXT(""),CONF_MENU_MAIN,CONF_TYPE_MENU,NULL,NULL,CONF_MENU_PERF,0,NULL},
//23
{ CFGTXT("Core selection"), "-1 (autodetect)",
  CFGTXT(
  "This option determines core selection. Auto-select is usually best since\n"
  "it allows the client to pick other cores as they become available. Please\n"
  "let distributed.net know if you find the client auto-selecting a core that\n"
  "manual benchmarking shows to be less than optimal.\n"
  ),CONF_MENU_PERF,CONF_TYPE_IARRAY,NULL,NULL,0,0,NULL,NULL},
//24
{ CFGTXT("Number of crunchers to run simultaneously"), "-1 (autodetect)",
  /* CFGTXT( */
  "This option specifies the number of threads you want the client to work on.\n"
  "On multi-processor machines this should be set to the number of processors\n"
  "available or to -1 to have the client attempt to auto-detect the number of\n"
  "processors. Multi-threaded clients can be forced to run single-threaded by\n"
  "setting this option to zero.\n"
#if (CLIENT_OS == OS_RISCOS)
  "Under RISC OS, processor 1 is the ARM, and processor 2 is an x86 processor\n"
  "card, if fitted.\n"
#endif
  /*) */,CONF_MENU_PERF,CONF_TYPE_INT,NULL,NULL,-1,128,NULL,NULL},
//25
{ CFGTXT("Priority level to run at"), "0 (lowest/at-idle)",
#if (CLIENT_OS == OS_RISCOS)
  CFGTXT(
  "The priority option is ignored on this machine. The distributed.net client\n"
  "for "CLIENT_OS_NAME" dynamically adjusts its process priority.\n"
  )
#elif (CLIENT_OS==OS_WIN16) //|| (CLIENT_OS==OS_WIN32)
  CFGTXT(
  "The priority option is ignored on this machine. distributed.net clients\n"
  "for Windows always run at lowest ('idle') priority.\n"
  )
#elif (CLIENT_OS==OS_NETWARE)
  CFGTXT(
  "The priority option for the distributed.net client for NetWare is directly\n"
  "proportionate to the rate at which the client yields. While the formula\n"
  "itself is simple, (priority+1)*500 microseconds (ie priority 0 == 0.5\n"
  "milliseconds), which translates to a nominal rate of 1000 yields per\n"
  "millisecond, the actual yield rate is far higher and varies from environment\n"
  "to environment. Thus, it is a really a matter of experimentation to find the\n"
  "best \"priority\" for your machine, and while \"priority\" zero will probably\n"
  "give you a less-than-ideal crunch rate, it will never be \"wrong\".\n" 
  )
#else
  /* CFGTXT( */
  "The higher the client's priority, the greater will be its demand for\n"
  "processor time. The operating system will fulfill this demand only after\n"
  "the demands of other processes with a higher or equal priority are fulfilled\n"
  "first. At priority zero, the client will get processing time only when all\n"
  "other processes are idle (give up their chance to run). At priority nine, the\n"
  "client will always get CPU time unless there is a time-critical process\n"
  "waiting to be run - this is obviously not a good idea unless the client is\n"
  "running on a machine that does nothing else.\n"
  "On *nix'ish OSs, the higher the priority, the less nice(1) the process.\n"
  #if (CLIENT_OS == OS_WIN32)
  "*Warning*: Running the Win32 client at any priority level other than zero is\n"
  "destructive to Operating System safety. Win32's thread scheduler is not nice.\n"
  "Besides, a zero priority does not mean that the client will run slower than at\n"
  "a higher priority. It simply means that all other processes have a better\n"
  "*chance* to get processor time than client. If none of them want/need processor\n"
  "time, the client will get it. Do *not* change the value of this option unless\n"
  "you are intimately familiar with the way Win32 thread scheduling works.\n"
  #endif
  /* ) */
#endif
  ,CONF_MENU_PERF, CONF_TYPE_INT, NULL, NULL, 0, 9, NULL,NULL },

/* ------------------------------------------------------------ */

//26
{ CFGTXT("Logging Options"),"",
  CFGTXT(""),CONF_MENU_MAIN,CONF_TYPE_MENU,NULL,NULL,CONF_MENU_LOG,0,NULL},
//27
{ CFGTXT("Log file type"), "0",
  CFGTXT(
  "This option determines what kind of file-based logging is preferred:\n"
  "\n"
  "0) none      altogether disables logging to file.\n"
  "1) no limit  the size of the file is not limited. This is the default if\n"
  "             a limit is not specified in the \"Log file limit\" option.\n"
  "2) restart   the log will be deleted/recreated when the file size specified\n"
  "             in the \"Log file limit\" option is reached.\n"
  "3) fifo      the oldest lines in the file will be discarded when the size\n"
  "             of the file exceeds the limit in the \"Log file limit\" option.\n"
  "4) rotate    a new file will be created when the rotation interval specified\n"
  "             in the \"Log file limit\" option is exceeded.\n"
  ),CONF_MENU_LOG,CONF_TYPE_INT,NULL,NULL /*logtypes[]*/,0,0,NULL,NULL},
//28
{ CFGTXT("File to log to"), "",
  CFGTXT(
  "The log file name is required for all log types except \"rotate\", for which it\n"
  "is optional. The effective file name used for the \"rotate\" log file type is\n"
  "constructed from a unique identifier for the period (time limit) concatenated\n"
  "to whatever you specify here. Thus, if the interval is weekly, the name of the\n"
  "log file used will be [file_to_log_to]yearweek"EXTN_SEP"log.\n"
  ),CONF_MENU_LOG,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//29
{ CFGTXT("Log file limit/interval"), "",
  CFGTXT(
  "For the \"rotate\" log type, this option determines the interval with which\n"
  "a new file will be opened. The interval may be specified as a number of days,\n"
  "or as \"daily\",\"weekly\",\"monthly\" etc.\n"
  "For other log types, this option determines the maximum file size in kilobytes.\n"
  "The \"fifo\" log type will enforce a minimum of 100KB to avoid excessive\n"
  "file I/O.\n"
  ),CONF_MENU_LOG,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//30
{ CFGTXT("Log by mail spool size (bytes)"), "0 (mail disabled)",
  CFGTXT(
  "The client is capable of sending you a log of the client's progress by mail.\n"
  "To activate this capability, specify how much you want the client to buffer\n"
  "before sending. The minimum is 2048 bytes, the maximum is approximately 130000\n"
  "bytes. Specify 0 (zero) to disable logging by mail.\n"
  ),CONF_MENU_LOG,CONF_TYPE_INT,NULL,NULL,0,125000,NULL,NULL},
//31
{ CFGTXT("SMTP Server to use"), "",
  CFGTXT(
  "Specify the name or DNS address of the SMTP host via which the client should\n"
  "relay mail logs. The default is the hostname component of the email address from\n"
  "which logs will be mailed.\n"
  ),CONF_MENU_LOG,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//32
{ CFGTXT("SMTP Port"), "25 (default)",
  CFGTXT(
  "Specify the port on the SMTP host to which the client's mail subsystem should\n"
  "connect when sending mail logs. The default is port 25.\n"
  ),CONF_MENU_LOG,CONF_TYPE_INT,NULL,NULL,0,0xFFFF,NULL,NULL},
//33
{ CFGTXT("E-mail address that logs will be mailed from"),
  "" /* *((const char *)(options[CONF_ID].thevariable)) */,
  CFGTXT(
  "(Some servers require this to be a real address)\n"
  ),CONF_MENU_LOG,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//34
{ CFGTXT("E-mail address to send logs to"),
  "" /* *((const char *)(options[CONF_ID].thevariable)) */,
  CFGTXT(
  "Full name and site eg: you@your.site.  Comma delimited list permitted.\n"
  ),CONF_MENU_LOG,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},

/* ------------------------------------------------------------ */

//35
{ CFGTXT("Network Timeout (seconds)"), "60 (default)",
  CFGTXT(
  "This option determines the amount of time the client will wait for a network\n"
  "read or write acknowledgement before it assumes that the connection has been\n"
  "broken. Any value between 5 and 300 seconds is valid and setting the timeout\n"
  "to -1 forces a blocking connection.\n"
  ),CONF_MENU_NET,CONF_TYPE_INT,NULL,NULL,-1,300,NULL,NULL},
//36
{ CFGTXT("Automatically select a distributed.net keyserver?"), "1",
  CFGTXT(
  "Set this option to 'Yes' UNLESS your client will be communicating\n"
  "with a personal proxy (instead of one of the main distributed.net\n"
  "keyservers) OR your client will be connecting through an HTTP proxy\n"
  "(firewall) and you have been explicitly advised by distributed.net\n"
  "staff to use a specific IP address.\n"
  ),CONF_MENU_NET,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL,NULL},
//37
{ CFGTXT("Keyserver hostname"), "",
  CFGTXT(
  "This is the name or IP address of the machine that your client will\n"
  "obtain keys from and send completed packets to. Avoid IP addresses\n"
  "if possible unless your client will be communicating through a HTTP\n"
  "proxy (firewall) and you have trouble fetching or flushing packets.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//38
{ CFGTXT("Keyserver port"), "", /* atoi("") is zero too. */
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
  ),CONF_MENU_NET,CONF_TYPE_INT,NULL,NULL,0,0xFFFF,NULL,NULL},
//39
{ CFGTXT("Keyserver is a personal proxy on a protected LAN?"),"0",
  CFGTXT(
  "If the keyserver that your client will be connecting to is a personal\n"
  "proxy inside a protected LAN (inside a firewall), set this option to 'yes'.\n"
  "Otherwise leave it at 'No'.\n"
  ),CONF_MENU_NET,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL,NULL},
//40
{ CFGTXT("Firewall/proxy protocol"), "none/transparent/mapped" /* note: atol("")==0 */,
  CFGTXT(
  "This field determines what protocol to use when communicating via a\n"
  "SOCKS or HTTP proxy.\n"
  ),CONF_MENU_NET,CONF_TYPE_INT,NULL,NULL,0,20,NULL,NULL},
//41
{ CFGTXT("Firewall hostname"), "",
  CFGTXT(
  "This field determines the hostname or IP address of the firewall proxy\n"
  "through which the client should communicate. The proxy is expected to be\n"
  "on a local network.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//42
{ CFGTXT("Firewall port"), "" /* note: atol("")==0 */,
  CFGTXT(
  "This field determines the port number on the firewall proxy to which the\n"
  "the client should connect. The port number must be valid.\n"
  ),CONF_MENU_NET,CONF_TYPE_INT,NULL,NULL,0,0xFFFF,NULL,NULL},
//43
{ CFGTXT("Firewall username"), "",
  CFGTXT(
  "Specify a username in this field if your SOCKS host requires\n"
  "authentication before permitting communication through it.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//44
{ CFGTXT("Firewall password"), "",
  CFGTXT(
  "Specify the password in this field if your SOCKS host requires\n"
  "authentication before permitting communication through it.\n"
  ),CONF_MENU_NET,CONF_TYPE_PASSWORD,NULL,NULL,0,0,NULL,NULL},
//45
{ CFGTXT("Use HTTP encapsulation even if not using an HTTP proxy?"),"0",
  CFGTXT(
  "Enable this option if you have an HTTP port-mapped proxy or other\n"
  "configuration that allows HTTP packets but not unencoded packets.\n"
  ),CONF_MENU_NET,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL,NULL},
//46
{ CFGTXT("Always use UUEncoding?"),"0",
  CFGTXT(
  "Enable this option if your network environment only supports 7bit traffic.\n"
  ),CONF_MENU_NET,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL,NULL},
//47
{ CFGTXT("Modem detection options"),"0",
  CFGTXT(
  "0) Normal mode: the client will send/receive packets only when it\n"
  "         empties the in buffer, hits the flush threshold, or\n"
  "         the user specifically requests a flush/fetch.\n"
  "1) Dial-up detection mode: This acts like mode 0, with the addition\n"
  "         that the client will automatically send/receive packets when a\n"
  "         dial-up networking connection is established. Modem users\n"
  "         will probably wish to use this option so that their client\n"
  "         never runs out of packets.\n"
  "2) Dial-up detection ONLY mode: Like the previous mode, this will cause\n"
  "         the client to automatically send/receive packets when\n"
  "         connected. HOWEVER, if the client runs out of packets,\n"
  "         it will NOT trigger auto-dial, and will instead work\n"
  "         on random packets until a connection is detected.\n"
  ),CONF_MENU_NET,CONF_TYPE_INT,NULL,&lurkmodetable[0],0,2,NULL,NULL},
//48
{ CFGTXT("Interfaces to watch"), "",
  /* CFGTXT( */
  "Colon-separated list of interface names to monitor for a connection,\n"
  "For example: \"ppp0:ppp1:eth1\". Wildcards are permitted, ie \"ppp*\".\n"
  "a) An empty list implies all interfaces that are identifiable as dialup,\n"
  "   ie \"ppp*:sl*:...\" (dialup interface names vary from platform to\n"
  "   platform. FreeBSD for example, also includes 'dun*' interfaces. Win32\n"
  "   emulates SLIP as a subset of PPP: sl* interfaces are seen as ppp*).\n"
  "b) if you have an intermittent ethernet connection through which you can\n"
  "   access the Internet, put the corresponding interface name in this list,\n"
  "   typically 'eth0'\n"
  "c) To include all interfaces, set this option to '*'.\n"
  #if (CLIENT_OS == OS_WIN32)
  "** All Win32 network adapters (regardless of medium; dialup/non-dialup)\n"
  "   also have a unique unchanging alias: 'lan0', 'lan1' and so on. Both\n"
  "   naming conventions can be used simultaneously.\n"
  "** a list of interfaces can be obtained from ipconfig.exe or winipcfg.exe\n"
  "   The first dialup interface is ppp0, the second is ppp1 and so on, while\n"
  "   the first non-dialup interface is eth0, the second is eth1 and so on.\n"
  #else
  "** The command line equivalent of this option is --interfaces-to-watch\n"
  #endif
  /* ) */,CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//49
{ /*dialwhenneeded*/ 
   #if (CLIENT_OS == OS_WIN32)
   CFGTXT("Use a specific DUN profile to connect with?"),
   #elif ((CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32))
   CFGTXT("Load/unload Winsock to initiate/hangup net connections?"),
   #else
   CFGTXT("Use scripts to initiate/hangup dialup connections?"),
   #endif
   "0",CFGTXT(
   "Select 'yes' to have the client control how network connections\n"
   "are initiatiated if none is active.\n"
   ),CONF_MENU_NET,CONF_TYPE_BOOL,NULL,NULL,0,1,NULL,NULL},
//50
{ CFGTXT("Dial-up Connection Profile"),"",
  #if (CLIENT_OS == OS_WIN32)
  CFGTXT("Select the DUN profile to use when dialing-as-needed.\n")
  #else
  CFGTXT("")
  #endif
  ,CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//51
{ CFGTXT("Command/script to start dialup"),"",
  CFGTXT(
  "Enter any valid shell command or script name to use to initiate a\n"
  "network connection. \"Dial the Internet as needed?\" must be enabled for\n"
  "this option to be of any use.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL},
//52
{ CFGTXT("Command/script to stop dialup"),"",
  CFGTXT(
  "Enter any valid shell command or script name to use to shutdown a\n"
  "network connection previously initiated with the script/command specified\n"
  "in the \"Command/script to start dialup\" option.\n"
  ),CONF_MENU_NET,CONF_TYPE_ASCIIZ,NULL,NULL,0,0,NULL,NULL}
};

