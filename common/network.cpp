// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: network.cpp,v $
// Revision 1.71  1999/01/19 05:39:08  cyp
// added connect() error message support for win16/32
//
// Revision 1.70  1999/01/19 04:47:35  cyp
// Fixed win16/32 connect bug. EINVAL is (of course!) not equal to WSAEINVAL.
//
// Revision 1.69  1999/01/13 19:49:09  cyp
// Tried to fix http and connect. Lets see...
//
// Revision 1.68  1999/01/11 23:39:34  michmarc
// Fix printing of debug packets when default char type is signed.
//
// Revision 1.67  1999/01/08 03:34:25  dicamillo
// Define ERRNO_IS_UNUSABLE for Mac OS.
//
// Revision 1.66  1999/01/07 22:01:41  cyp
// fixed a bad #if in errno checking.
//
// Revision 1.65  1999/01/05 22:56:29  cyp
// Added support for SOCKS5 address type 0x03 (hostname) so a failed lookup
// on the target hostname is no longer fatal. Nevertheless, SOCKS5 still
// says "address type unsupported", and since I don't have IPv6, I assume my
// SOCKS5 server is faulty.
//
// Revision 1.64  1999/01/04 21:47:58  cyp
// SOCKS4 works fine. SOCKS5 says "address type not supported".
//
// Revision 1.63  1999/01/04 16:05:06  silby
// Fixed byte ordering problem with socks code; Still does not work, however.
//
// Revision 1.62  1999/01/04 04:47:55  cyp
// Minor fixes for platforms without network support.
//
// Revision 1.61  1999/01/04 03:56:39  cyp
// ack! ::nofallback was not being assigned.
//
// Revision 1.60  1999/01/03 06:19:35  cyp
// Cleared an unused variable notice.
//
// Revision 1.59  1999/01/03 02:36:58  cyp
// A strlwr() equivalent is not really needed here... It was a remnant of a
// debug session.
//
// Revision 1.58  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.57  1999/01/01 01:17:41  silby
// Added dctistrg module so that a portable string
// lowercasing function can be added.
//
// Revision 1.56  1998/12/31 17:55:50  cyp
// changes to Network::Open(): (a) retry loop is inside ::Open() (was from
// the external NetOpen()) (b) cleaned up the various hostname/addr/port
// variables to make sense and be uniform throughout. (c) nofallback handling
// is performed by ::Open() and not by the external NetOpen().
//
// Revision 1.55  1998/12/24 05:22:03  dicamillo
// Mac OS updates: use ioctl for blocking; support connection timeout value.
//
// Revision 1.54  1998/12/22 02:38:39  snake
// EPROTO doesn't exist on some systems
//
// Revision 1.53  1998/12/21 17:54:23  cyp
// (a) Network connect is now non-blocking. (b) timeout param moved from
// network::Get() to object scope.
//
// Revision 1.52  1998/12/08 05:56:18  dicamillo
// Add MacOS code for LowLevelConditionSocket.
//
// Revision 1.51  1998/11/16 19:07:01  cyp
// Fixed integer truncation warnings.
//
// Revision 1.50  1998/11/12 07:33:36  remi
// Solved the round-robin bug. Network::Close() sets retries=0, and it was
// called by Network::Open(), so multiple Network::Open() won't see
// anything but retries==0. Network::LowLevelCreateSocket() also called
// Network::Close(), instead of Network::LowLevelCloseSocket() I think.
//
// Revision 1.49  1998/10/26 02:55:04  cyp
// win16 changes
//
// Revision 1.48  1998/10/16 16:14:58  remi
// Corrected my $Log comments.
//
// Revision 1.47  1998/10/16 16:09:13  remi
// Fixed SOCKS4 / SOCKS5 support. In InitializeConnection() 'lastport' is the
// SOCKS proxy port, not the keyproxy port. SOCKS4 doesn't work on my 
// machines, but that's perhaps a bug in my SOCKS server. SOCKS5 seems Ok.
//
// Revision 1.46  1998/09/30 22:27:55  remi
// http connections should always be sent to port 80 (?).
//
// Revision 1.45  1998/09/28 03:44:16  cyp
// Added no-support wrapper around shutdown()
//
// Revision 1.44  1998/09/25 04:32:09  pct
// DEC Ultrix port changes
//
// Revision 1.43  1998/09/20 15:24:26  blast
// AmigaOS changes
//
// Revision 1.42  1998/09/06 02:30:48  cyp
// Added check 'if (t_errno < t_nerr)' before getting t_errlist[t_errno].
//
// Revision 1.41  1998/09/06 01:05:10  cyp
// Fixed a missing underscore in an #ifdef _TIUSER[_]
//
// Revision 1.40  1998/09/04 10:35:09  chrisb
// RISCOSism for the #include <errno.h>
//
// Revision 1.39  1998/09/03 16:01:32  cyp
// Added TLI support. Any other SYSV (-type) takers?
//
// Revision 1.38  1998/08/28 22:04:49  cyp
// Cleaned up/extended the division between "high level" and "low level"
// methods: inet/socket functions/structures are encapsulated in small
// "low level" methods and "high level" methods are completely clean of
// any inet/socket library/include dependancies. Also fixed assumptions
// that zero is an invalid socket/fd.
//
// Revision 1.37  1998/08/25 00:06:53  cyp
// Merged (a) the Network destructor and DeinitializeNetwork() into NetClose()
// (b) the Network constructor and InitializeNetwork() into NetOpen().
// These two new functions (in netinit.cpp) are essentially what the static
// FetchFlushNetwork[Open|Close]() functions in buffupd.cpp used to be.
//
// Revision 1.36  1998/08/20 19:27:10  cyruspatel
// Made the purpose of NetworkInitialize/Deinitialize a little more
// transparent.
//
// Revision 1.35  1998/08/10 21:53:51  cyruspatel
// Two changes to work around a lack of a method to detect if the network
// availability state had changed (or existed to begin with) and also protect
// against any re-definition of client.offlinemode. (a) The NO!NETWORK define 
// is now obsolete. Whether a platform has networking capabilities or not is 
// now a purely network.cpp thing. (b) NetworkInitialize()/NetworkDeinitialize
// are no longer one-shot-and-be-done-with-it affairs. ** Documentation ** is
// in netinit.cpp.
//
// Revision 1.34  1998/08/07 20:27:10  cyruspatel
// Added timestamps to network messages.
//
// Revision 1.33  1998/08/02 16:18:12  cyruspatel
// Completed support for logging.
//
// Revision 1.32  1998/07/26 12:46:12  cyruspatel
// new inifile option: 'autofindkeyserver', ie if keyproxy= points to a
// xx.v27.distributed.net then that will be interpreted by Network::Resolve()
// to mean 'find a keyserver that covers the timezone I am in'. Network
// constructor extended to take this as an argument.
//
// Revision 1.31  1998/07/18 05:53:10  ziggyb
// Removed a unneeded return 0, which was causing a 'unreachable code' 
// warning in Watcom
//
// Revision 1.30  1998/07/13 23:54:25  cyruspatel
// Cleaned up NO!NETWORK handling.
//
// Revision 1.29  1998/07/13 03:30:07  cyruspatel
// Added 'const's or 'register's where the compiler was complaining about
// "declaration/type or an expression" ambiguities.
//
// Revision 1.28  1998/07/08 09:24:54  jlawson
// eliminated integer size warnings on win16
//
// Revision 1.27  1998/07/07 21:55:46  cyruspatel
// client.h has been split into client.h and baseincs.h 
//
// Revision 1.26  1998/06/29 06:58:06  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.25  1998/06/29 04:22:26  jlawson
// Updates for 16-bit Win16 support
//
// Revision 1.24  1998/06/26 06:56:38  daa
// convert all the strncmp's in the http code with strncmpi to 
// deal with proxy's case shifting HTML
//
// Revision 1.23  1998/06/22 01:04:58  cyruspatel
// DOS changes. Fixes various compile-time errors: removed extraneous ')' in
// sleepdef.h, resolved htonl()/ntohl() conflict with same def in client.h
// (is now inline asm), added NO!NETWORK wrapper around Network::Resolve()
//
// Revision 1.22  1998/06/15 12:04:03  kbracey
// Lots of consts.
//
// Revision 1.21  1998/06/14 13:08:09  ziggyb
// Took out all OS/2 DOD stuff, being moved to platforms\os2cli\dod.cpp
//
// Revision 1.20  1998/06/14 10:13:29  ziggyb
// There was a stray '\' on line 567 that's not supposed to be there
//
// Revision 1.19  1998/06/14 08:26:52  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.18  1998/06/14 08:13:00  friedbait
// 'Log' keywords added to maintain automatic change history
//
// Revision 1.18  1998/06/13 23:33:18  cyruspatel
// Fixed NetWare stuff and added #include "sleepdef.h" (which should now 
// warn if macros are not the same)
//

#if (!defined(lint) && defined(__showids__))
const char *network_cpp(void) {
return "@(#)$Id: network.cpp,v 1.71 1999/01/19 05:39:08 cyp Exp $"; }
#endif

//----------------------------------------------------------------------

#include "cputypes.h"
#include "baseincs.h"  // standard stuff
#include "sleepdef.h"  // Fix sleep()/usleep() macros there! <--
#include "autobuff.h"  // Autobuffer class
#include "cmpidefs.h"  // strncmpi(), strcmpi()
#include "logstuff.h"  // LogScreen()
#include "clitime.h"   // CliGetTimeString(NULL,1);
#include "triggers.h"  // CheckExitRequestTrigger()
#include "network.h"   // thats us

#if (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_MACOS)
#define ERRNO_IS_UNUSABLE_FOR_CONN_ERRMSG 
#endif

extern int NetCheckIsOK(void); // used before doing i/o

//----------------------------------------------------------------------

#ifdef NONETWORK
#error NONETWORK define is obsolete. Networking capability is now a purely network.cpp thing.
#endif

//----------------------------------------------------------------------

#define VERBOSE_OPEN //print cause of ::Open() errors
#define UU_DEC(Ch) (char) (((Ch) - ' ') & 077)
#define UU_ENC(Ch) (char) (((Ch) & 077) != 0 ? ((Ch) & 077) + 0x20 : '`')

//----------------------------------------------------------------------

#pragma pack(1)               // no padding allowed

typedef struct _socks4 {
  unsigned char VN;           // version == 4
  unsigned char CD;           // command code, CONNECT == 1
  u16 DSTPORT;                // destination port, network order
  u32 DSTIP;                  // destination IP, network order
  char USERID[1];             // variable size, null terminated
} SOCKS4;

//----------------------------------------------------------------------

typedef struct _socks5methodreq {
  unsigned char ver;          // version == 5
  unsigned char nMethods;     // number of allowable methods following
  unsigned char Methods[2];   // this program will request at most two
} SOCKS5METHODREQ;

//----------------------------------------------------------------------

typedef struct _socks5methodreply {
  unsigned char ver;          // version == 1
  unsigned char Method;       // server chose method ...
  char end;
} SOCKS5METHODREPLY;

//----------------------------------------------------------------------

typedef struct _socks5userpwreply {
  unsigned char ver;          // version == 1
  unsigned char status;       // 0 == success
  char end;
} SOCKS5USERPWREPLY;

//----------------------------------------------------------------------

typedef struct _socks5 {
  unsigned char ver;          // version == 5
  unsigned char cmdORrep;     // cmd: 1 == connect, rep: 0 == success
  unsigned char rsv;          // must be 0
  unsigned char atyp;         // address type we assume IPv4 == 1
  u32 addr;                   // network order
  u16 port;                   // network order
  char end;
} SOCKS5;

#pragma pack()

//----------------------------------------------------------------------

const char *Socks5ErrorText[9] =
{
 /* 0 */ "" /* success */,
  "general SOCKS server failure",
  "connection not allowed by ruleset",
  "Network unreachable",
  "Host unreachable",
  "Connection refused",
  "TTL expired",
  "Command not supported",
  "Address type not supported"
};

//======================================================================

//short circuit the u32 -> in_addr.s_addr ->inet_ntoa method.
//besides, it works around context issues.
static const char *__inet_ntoa__(u32 addr)
{
  static char buff[18];
  char *p = (char *)(&addr);
  sprintf( buff, "%d.%d.%d.%d", (p[0]&255),(p[1]&255),(p[2]&255),(p[3]&255) );
  return buff;
}  

static void __print_packet( const char *label, const char *apacket, int alen )
{
  int i,n;
  for (i=0;i<alen;i+=16)
    {
    char buffer[128];
    char *p=&buffer[0];
    p+=sprintf(buffer,"%s %04x: ", label, i );
    for (n=0;(n<16 && ((n+i)<alen));n++)
      p+=sprintf(p,"%02x ", (unsigned char)apacket[n+i]);
    for (;n<16;n++)
      {*p++=' ';*p++=' ';*p++=' ';}
    for (n=0;(n<16 && ((n+i)<alen));n++)        
      *p++=((apacket[n+1]!='\r' && apacket[n+i]!='\n' && apacket[n+i]!='\t'
           && isprint(apacket[n+i]))?(apacket[n+i]):('.'));
    *p++=0;
    LogRaw("%s\n",buffer);
    }
  LogRaw("%s total len: %d\n",label, alen);
  return;
}

static void __hostnamecpy( char *dest, const char *source,unsigned int maxlen)
{
  unsigned int len = 0;
  while (*source && isspace(*source))
    source++;
  while (((++len)<maxlen) && *source && !isspace(*source))
    *dest++=(char)tolower(*source++);
  *dest=0;
  return;
}  

//======================================================================

Network::Network( const char * servname, s16 servport, int _nofallback,
                  int AutoFindKeyServer, int _iotimeout )
{
  // intialize communication parameters
  server_name[0] = 0;
  if (servname)
    __hostnamecpy( server_name, servname, sizeof(server_name));
  server_port = servport;
  autofindkeyserver = AutoFindKeyServer;

  mode = startmode = 0;
  sock = INVALID_SOCKET;
  gotuubegin = gothttpend = 0;
  httplength = 0;
  reconnected = 0;
  nofallback = _nofallback;
  
  fwall_hostaddr = svc_hostaddr = conn_hostaddr = 0;
  fwall_userpass[0] = 0;

  isnonblocking = 0;      /* whether the socket could be set non-blocking */
  iotimeout = _iotimeout; /* if iotimeout is <0, use blocking calls */
  if (iotimeout < 0)
    iotimeout = -1;
  else if (iotimeout < 5)
    iotimeout = 5;
  else if (iotimeout > 120)
    iotimeout = 120;

  #ifdef DEBUG
  verbose_level = 2;
  #elif defined(VERBOSE_OPEN)
  verbose_level = 1;
  #else
  verbose_level = 0; //quiet
  #endif
  
  // check that the packet structures have been correctly packed
  size_t dummy;
  if (((dummy = offsetof(SOCKS4, USERID[0])) != 8) ||
     ((dummy = offsetof(SOCKS5METHODREQ, Methods[0])) != 2) ||
     ((dummy = offsetof(SOCKS5METHODREPLY, end)) != 2) ||
     ((dummy = offsetof(SOCKS5USERPWREPLY, end)) != 2) ||
     ((dummy = offsetof(SOCKS5, end)) != 10))
    LogScreen("Network::Socks Incorrectly packed structures.\n");
}


//----------------------------------------------------------------------

Network::~Network(void)
{
  Close();
}

//----------------------------------------------------------------------

void Network::SetModeUUE( int is_enabled )
{
  if (is_enabled)
    {
    startmode &= ~(MODE_SOCKS4 | MODE_SOCKS5);
    startmode |= MODE_UUE;
    }
  else 
    {
    startmode &= ~MODE_UUE;
    }
  return;
}

//----------------------------------------------------------------------

void Network::SetModeHTTP( const char *httphost, s16 httpport, const char *httpusername)
{
  conn_hostaddr = svc_hostaddr = fwall_hostaddr = 0;

  if (httphost && httphost[0])
    {
    startmode &= ~(MODE_SOCKS4 | MODE_SOCKS5);
    startmode |= MODE_HTTP;
    fwall_hostport = httpport;
    strncpy( fwall_userpass, httpusername, sizeof(fwall_userpass));
    __hostnamecpy( fwall_hostname, httphost, sizeof(fwall_hostname));
    }
  else 
    {
    startmode &= ~MODE_HTTP;
    }
  return;
}

//----------------------------------------------------------------------

void Network::SetModeSOCKS4(const char *sockshost, s16 socksport,
      const char * socksusername )
{
  conn_hostaddr = svc_hostaddr = fwall_hostaddr = 0;

  if (sockshost && sockshost[0])
    {
    startmode &= ~(MODE_HTTP | MODE_SOCKS5 | MODE_UUE);
    startmode |= MODE_SOCKS4;
    fwall_hostport = socksport;
    __hostnamecpy(fwall_hostname, sockshost, sizeof(fwall_hostname));
    fwall_userpass[0] = 0;
    if (socksusername && *socksusername)
      strncpy(fwall_userpass, socksusername, sizeof(fwall_userpass));
    if (fwall_hostport == 0)
      fwall_hostport = 1080;
    }
  else
    {
    startmode &= ~MODE_SOCKS4;
    fwall_hostname[0] = 0;
    }
  return;
}

//----------------------------------------------------------------------

void Network::SetModeSOCKS5(const char *sockshost, s16 socksport,
      const char * socksusernamepw )
{
  conn_hostaddr = svc_hostaddr = fwall_hostaddr = 0;

  if (sockshost && sockshost[0])
    {
    startmode &= ~(MODE_HTTP | MODE_SOCKS4 | MODE_UUE);
    startmode |= MODE_SOCKS5;
    fwall_hostport = socksport;
    __hostnamecpy(fwall_hostname, sockshost, sizeof(fwall_hostname));
    fwall_userpass[0] = 0;
    if (socksusernamepw && *socksusernamepw)
      strncpy(fwall_userpass, socksusernamepw, sizeof(fwall_userpass));
    if (fwall_hostport == 0)
      fwall_hostport = 1080;
    }
  else
    {
    startmode &= ~MODE_SOCKS5;
    fwall_hostname[0] = 0;
    }
  return;
}

/* ----------------------------------------------------------------------- */

// returns -1 on error, 0 on success
int Network::Open( SOCKET insock)
{
  sock = insock;

  // set communications settings
  mode = startmode;
  gotuubegin = gothttpend = puthttpdone = gethttpdone = 0;
  httplength = 0;
  netbuffer.Clear();
  uubuffer.Clear();

  // make socket non-blocking
  isnonblocking = (iotimeout < 0)?(0):(MakeNonBlocking() == 0);

  return 0;
}  

/* ----------------------------------------------------------------------- */

int Network::Open( void )               // returns -1 on error, 0 on success
{
  gethttpdone = puthttpdone = 0;
  netbuffer.Clear();
  uubuffer.Clear();
  gotuubegin = gothttpend = 0;
  httplength = 0;
  mode = startmode;

  unsigned int retries = 0;
  unsigned int maxtries = 5; /* 3 for preferred server, 2 for fallback */
  unsigned int preftries = 3;
  if (nofallback) maxtries = preftries;
  
  while (retries < maxtries) 
    {
    int success = 0;
    const char *netcheckfailed = "Network::Open Error - TCP/IP Connection Lost.\n";

    if (CheckExitRequestTrigger())
      return -1;

    if ((startmode & MODE_PROXIED)!=0 && (fwall_hostname[0]==0 || fwall_hostport==0))
      {
      Log("Network::Invalid %s proxy hostname or port.\n"
         "Connect cancelled.\n",((startmode & MODE_HTTP)?("HTTP"):("SOCKS")));
      return -1;
      }

    if (!NetCheckIsOK()) /* connection broken */
      {
      LogScreen(netcheckfailed);
      return -1;
      }

    /* ---------- create a new socket --------------- */

    success = (LowLevelCreateSocket() == 0);      
    isnonblocking = 0;

    if (!success)
      {
      if (verbose_level > 0) 
        LogScreen("Network::failed to create network socket.\n");
      }

    /* --- resolve the addresses(es) --- */

    if (success)  
      {
      svc_hostname = server_name;
      svc_hostport = server_port;

      if (svc_hostname[0]==0 || retries > (preftries-1) /* fallback*/)
        {
        svc_hostaddr = 0; 
        svc_hostname = "rc5proxy.distributed.net";
        if (svc_hostport != 80 && svc_hostport != 23 && 
          svc_hostport != 2064 && svc_hostport != 3064 /* && 
          svc_hostport != 21 && svc_hostport != 25 && svc_hostport!=110 */)
          {
          svc_hostport = 0;
          }
        autofindkeyserver = 1;
        }
      if (svc_hostport == 0)
        {
        if ((startmode & MODE_HTTP)!=0)
          svc_hostport = 80;
        else if ((startmode & MODE_UUE)!=0)
          svc_hostport = 23;
        else
          svc_hostport = DEFAULT_PORT;
        }

      if ((startmode & MODE_HTTP) == 0) /* we always re-resolve unless http */
        {                               // socks5 needs a 'good' hostname
        svc_hostaddr = 0;               // (obtained from resolve_hostname)
        }                               // if name resolution fails - cyp
      
      if (!NetCheckIsOK())
        {
        success = 0;
        retries = maxtries;
        LogScreen(netcheckfailed);
        }
      else if ((startmode & MODE_PROXIED) != 0)
        {
        if (Resolve( fwall_hostname, &fwall_hostaddr, fwall_hostport ) < 0)
          {
          success = 0;
          fwall_hostaddr = 0;
          if (verbose_level > 0)
            LogScreen("Network::failed to resolve name \"%s\"\n", 
                                                     resolve_hostname );
          retries = maxtries; //unrecoverable error. retry won't help
          }
        else if (svc_hostaddr == 0) /* always true unless http */
          {
          if ( Resolve(svc_hostname, &svc_hostaddr, svc_hostport ) < 0 )
            {
            svc_hostaddr = 0;
            if ((startmode & (MODE_SOCKS4 /*| MODE_SOCKS5*/))!=0)
              {
              success = 0; // socks needs the address to resolve now.
              if (verbose_level > 0)
                LogScreen("Network::failed to resolve hostname \"%s\"\n", 
                                                     resolve_hostname);
              }
            }
          svc_hostname = resolve_hostname; //socks5 will use this
          }
        conn_hostaddr = fwall_hostaddr;
        conn_hostname = fwall_hostname;
        conn_hostport = fwall_hostport;
        }
      else /* resolve for non-proxied connect */
        {
        if (Resolve( svc_hostname, &svc_hostaddr, svc_hostport ) < 0) 
          {
          success = 0;
          svc_hostaddr = 0;
          if (verbose_level > 0)
            LogScreen("Network::failed to resolve name \"%s\"\n", 
                                                    resolve_hostname );
          }
        svc_hostname = resolve_hostname;
        conn_hostaddr = svc_hostaddr;
        conn_hostname = svc_hostname;
        conn_hostport = svc_hostport;
        }
      }

    /* ------ connect ------- */

    if (success)
      {  
      if (verbose_level > 0 && reconnected == 0)
        {
        const char *targethost = svc_hostname;
        unsigned int len = strlen( targethost );
        const char sig[]=".distributed.net";
        char scratch[sizeof(sig)+20];
        if ((len > (sizeof( sig )-1) && autofindkeyserver &&
           strcmpi( &targethost[(len-(sizeof( sig )-1))], sig ) == 0))
          {
          targethost = sig+1;
          if (svc_hostaddr)
            {
            sprintf(scratch, "%s %s", sig+1, __inet_ntoa__(svc_hostaddr));
            targethost = scratch;
            }
          }
        
        if ((startmode & MODE_PROXIED)==0)
          {  
          #ifdef DEBUG
          Log
          #else
          LogScreen
          #endif
          ("Connecting to %s:%u...\n", targethost,
                                     ((unsigned int)(svc_hostport)) );
          }
        else
          {
          #ifdef DEBUG
          Log
          #else
          LogScreen
          #endif
          ("Connecting to %s:%u\nvia %s proxy %s:%u\n", targethost,
                                     ((unsigned int)(svc_hostport)),
                      ((startmode & MODE_SOCKS5)?("SOCKS5"):
                        ((startmode & MODE_SOCKS4)?("SOCKS4"):
                         ((startmode & MODE_HTTP)?("HTTP"):("???")))),
                      fwall_hostname, (unsigned int)fwall_hostport );
          }
        }

      #ifndef ENSURE_CONNECT_WITH_BLOCKING_SOCKET
      if (iotimeout > 0)
        {
        isnonblocking = ( MakeNonBlocking() == 0 );
        if (verbose_level > 1) //debug
          {
          LogScreen("Debug::Connecting with %sblocking socket.\n", 
              ((isnonblocking)?("non-"):("")) );
          }
        }
      #endif
      
      success = ( LowLevelConnectSocket( conn_hostaddr, conn_hostport ) == 0 );
      
      #ifdef ENSURE_CONNECT_WITH_BLOCKING_SOCKET
      if (success && iotimeout > 0)
        {
        isnonblocking = ( MakeNonBlocking() == 0 );
        if (verbose_level > 1) //debug
          {
          LogScreen("Debug::Connected (%sblocking).\n", 
              ((isnonblocking)?("non-"):("")) );
          }
        }
      #endif
      
      if (success)
        reconnected = 1;
      else //if (!success)   /* connect failed */
        {
        if (verbose_level > 0)
          {
          LogScreen( "Connect to host %s:%u failed.\n",
             __inet_ntoa__(conn_hostaddr), (unsigned int)(conn_hostport));
          #if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32)
          int err = errno; /* connect copies it from WSAGetLastError(); */
          const char *msg = "unrecognized error";
          if      (err == WSAEADDRINUSE)  msg="The specified address is already in use";
          else if (err == WSAEADDRNOTAVAIL) msg="The specified address is not available from the local machine";
          else if (err == WSAECONNREFUSED) msg="The attempt to connect was rejected. Destination may be busy.";
          else if (err == WSAEISCONN)      msg="The socket is already connected"; //udp only?
          else if (err == WSAENETUNREACH)  msg="The network cannot be reached from this host at this time";
          else if (err == WSAENOBUFS)      msg="No buffer space is currently available";
          else if (err == WSAETIMEDOUT)    msg="Attempt to connect timed out";
          LogScreen( " %s  Error %d (%s)\n",CliGetTimeString(NULL,0), 
                                      err, msg );
          #elif (!defined(ERRNO_IS_UNUSABLE_FOR_CONN_ERRMSG))
          if (NetCheckIsOK()) //use errno only if netcheck says ok.
            {
            #if defined(_TIUSER_)
              LogScreen( " %s  Error %d (%s)\n", 
                  CliGetTimeString(NULL,0),  t_errno, 
                ((t_errno>=t_nerr)?("undefined error"):(t_errlist[t_errno])) );
            #else
              LogScreen( " %s  Error %d (%s)\n",CliGetTimeString(NULL,0), 
                                      errno, strerror(errno) );
            #endif                            
            }
          #endif
          }
        } //connect failed
      } // resolve succeeded
      
    /* ---- initialize the connection ---- */
       
    if (success)
      {
      int rc = InitializeConnection();
      if (rc != 0)
        success = 0;
      if (rc < 0)           /* unrecoverable error (negotiation failure) */
        retries = maxtries; /* so don't retry */
      }

    /* ---- clean up ---- */
      
    if (success)
      return 0;
      
    Close();
    retries++;

    if (retries < maxtries)
      {
      LogScreen( "Network::Open Error - sleeping for 3 seconds\n" );
      sleep( 3 );
      }
    }
    
  return -1;
}  

// -----------------------------------------------------------------------
    
int Network::InitializeConnection(void)
{
  // set communications settings
  gotuubegin = gothttpend = puthttpdone = gethttpdone = 0;
  httplength = 0;
  netbuffer.Clear();
  uubuffer.Clear();

  if (!NetCheckIsOK())
    return -1;
  if (sock == INVALID_SOCKET)
    return -1;

  if (startmode & MODE_SOCKS5)
    {
    int success = 0; //assume failed
    int recoverable = 0; //assume non-recoverable error (negotiation failure)
    
    char socksreq[600];  // room for large username/pw (255 max each)
    SOCKS5METHODREQ *psocks5mreq = (SOCKS5METHODREQ *)socksreq;
    SOCKS5METHODREPLY *psocks5mreply = (SOCKS5METHODREPLY *)socksreq;
    SOCKS5USERPWREPLY *psocks5userpwreply = (SOCKS5USERPWREPLY *)socksreq;
    SOCKS5 *psocks5 = (SOCKS5 *)socksreq;
    u32 len;

    int tmp_isnonblocking = (isnonblocking != 0);
    if (isnonblocking)
      {
      isnonblocking = ((MakeBlocking() == 0)?(0):(1));
      if (isnonblocking)
        {
        if (verbose_level > 0)
          LogScreen("SOCKS5: unable to temporarily revert to blocking mode\n");
        iotimeout = -1; //force blocking mode in next try
        return +1; //recoverable
        }
      }

    // transact a request to the SOCKS5 proxy requesting
    // authentication methods.  If the username/password
    // is provided we ask for no authentication or user/pw.
    // Otherwise we ask for no authentication only.

    psocks5mreq->ver = 5;
    psocks5mreq->nMethods = (unsigned char) (fwall_userpass[0] ? 2 : 1);
    psocks5mreq->Methods[0] = 0;  // no authentication
    psocks5mreq->Methods[1] = 2;  // username/password

    int authaccepted = 0;

    len = 2 + psocks5mreq->nMethods;
    if (LowLevelPut(len, socksreq) < 0)
      {
      if (verbose_level > 0)
        LogScreen("SOCKS5: error sending negotiation request\n");
      recoverable = 1;
      }
    else if ((u32)LowLevelGet(2, socksreq) != 2)
      {
      if (verbose_level > 0)
        LogScreen("SOCKS5: failed to get negotiation request ack.\n");
      recoverable = 1;
      }
    else if (psocks5mreply->ver != 5)
      {
      if (verbose_level > 0)
        LogScreen("SOCKS5: authentication has wrong version, %d should be 5\n", 
                            psocks5mreply->ver);
      }
    else if (psocks5mreply->Method == 2)  // username and pw
      {
      char username[255];
      char password[255];
      char *pchSrc, *pchDest;
      int userlen, pwlen;

      pchSrc = fwall_userpass;
      pchDest = username;
      while (*pchSrc && *pchSrc != ':')
        *pchDest++ = *pchSrc++;
      *pchDest = 0;
      userlen = pchDest - username;
      if (*pchSrc == ':')
        pchSrc++;
      strcpy(password, pchSrc);
      pwlen = strlen(password);

      //   username/password request looks like
      // +----+------+----------+------+----------+
      // |VER | ULEN |  UNAME   | PLEN |  PASSWD  |
      // +----+------+----------+------+----------+
      // | 1  |  1   | 1 to 255 |  1   | 1 to 255 |
      // +----+------+----------+------+----------+

      len = 0;
      socksreq[len++] = 1;    // username/pw subnegotiation version
      socksreq[len++] = (char) userlen;
      memcpy(socksreq + len, username, (int) userlen);
      len += userlen;
      socksreq[len++] = (char) pwlen;
      memcpy(socksreq + len, password, (int) pwlen);
      len += pwlen;

      if (LowLevelPut(len, socksreq) < 0)
        {
        if (verbose_level > 0)
          LogScreen("SOCKS5: failed to send sub-negotiation request.\n");
        recoverable = 1;
        }
      else if ((u32)LowLevelGet(2, socksreq) != 2)
        {
        if (verbose_level > 0)
          LogScreen("SOCKS5: failed to get sub-negotiation response.\n");
        recoverable = 1;
        }
      else if (psocks5userpwreply->ver != 1 ||
           psocks5userpwreply->status != 0)
        {
        if (verbose_level > 0)
          LogScreen("SOCKS5: user %s rejected by server.\n", username);
        }       
      else
        {
        authaccepted = 1;
        }
      } //username and pw
    else if (psocks5mreply->Method == 1)  // GSSAPI
      {
      if (verbose_level > 0)
        LogScreen("SOCKS5: GSSAPI per-message authentication is\n"
                  "not supported. Please use SOCKS4 or HTTP.\n");
      }
    else if (psocks5mreply->Method == 0)       // no authentication required
      {
      // nothing to do for no authentication method
      authaccepted = 1;
      }
    else //if (psocks5mreply->Method > 2)
      {
      if (verbose_level > 0)
        LogScreen("SOCKS5 authentication method rejected.\n");
      }
      
    if (authaccepted)
      {
      // after subnegotiation, send connect request
      psocks5->ver = 5;
      psocks5->cmdORrep = 1;   // connnect
      psocks5->rsv = 0;   // must be zero
      psocks5->atyp = 1;  // IPv4 = 1
      psocks5->addr = svc_hostaddr;
      psocks5->port = (u16)htons(svc_hostport); //(u16)(htons((server_name[0]!=0)?((u16)port):((u16)(DEFAULT_PORT))));
      int packetsize = 10;

      if (svc_hostaddr == 0)           
        {                              
        psocks5->atyp = 0x03; //fully qualified domainname
        char *p = (char *)(&psocks5->addr);
        strcpy( p+1, svc_hostname );     //at this point svc_hostname is a
        *p = (char)(len = strlen( p+1 )); //ptr to a resolve_hostname
        p += (++len);
        *((u16 *)(p)) = (u16)htons(svc_hostport);
        packetsize = (10-sizeof(u32))+len;
        }
        
      if (LowLevelPut(packetsize, socksreq) < 0)
        {
        if (verbose_level > 0)
          LogScreen("SOCKS5: failed to send connect request.\n");
        recoverable = 1;
        }
      else if ((u32)LowLevelGet(packetsize, socksreq) < 10 /*ok for both atyps*/)
        {
        if (verbose_level > 0)
          LogScreen("SOCKS5: failed to get connect request ack.\n");
        recoverable = 1;
        }
      else if (psocks5->ver != 5)
        {
        if (verbose_level > 0)
           LogScreen("SOCKS5: reply has wrong version, %d should be 5\n", 
                       psocks5->ver);
        }
      else if (psocks5->cmdORrep == 0)  // 0 is successful connect
        {
        success = 1; 
        if (psocks5->atyp == 1)  // IPv4
          svc_hostaddr = psocks5->addr;
        }
      else if (verbose_level > 0)
        {
        const char *p = ((psocks5->cmdORrep >=
                         (sizeof Socks5ErrorText / sizeof Socks5ErrorText[0]))
                         ? ("") : (Socks5ErrorText[ psocks5->cmdORrep ]));
        LogScreen("SOCKS5: server error 0x%02x%s%s%s\nconnecting to %s:%u", 
                 ((int)(psocks5->cmdORrep)),
                 ((*p) ? (" (") : ("")), p, ((*p) ? (")") : ("")),
                 svc_hostname, (unsigned int)svc_hostport );
        }
      } //if (authaccepted)
      
    if (tmp_isnonblocking)
      isnonblocking = (MakeNonBlocking()==0);
      
    return ((success)?(0):((recoverable)?(+1):(-1)));
    } //if (startmode & MODE_SOCKS5)

  if (startmode & MODE_SOCKS4)
    {
    int success = 0; //assume failed
    int recoverable = 0; //assume non-recoverable error (negotiation failure)

    char socksreq[128];  // min sizeof(fwall_userpass) + sizeof(SOCKS4)
    SOCKS4 *psocks4 = (SOCKS4 *)socksreq;
    u32 len;

    int tmp_isnonblocking = (isnonblocking != 0);
    if (isnonblocking)
      {
      isnonblocking = ((MakeBlocking() == 0)?(0):(1));
      if (isnonblocking)
        {
        if (verbose_level > 0)
          LogScreen("SOCKS4: unable to temporarily revert to blocking mode\n");
        iotimeout = -1; //force blocking mode in next try
        return +1; //recoverable
        }
      }

    // transact a request to the SOCKS4 proxy giving the
    // destination ip/port and username and process its reply.

    psocks4->VN = 4;
    psocks4->CD = 1;  // CONNECT
    psocks4->DSTPORT = (u16)htons(svc_hostport); //(u16)htons((server_name[0]!=0)?((u16)port):((u16)DEFAULT_PORT));
    psocks4->DSTIP = svc_hostaddr;   //lasthttpaddress;
    strncpy(psocks4->USERID, fwall_userpass, sizeof(fwall_userpass));

    len = sizeof(*psocks4) - 1 + strlen(fwall_userpass) + 1;
    if (LowLevelPut(len, socksreq) < 0)
      {
      if (verbose_level > 0)
        LogScreen("SOCKS4: Error sending connect request\n");
      recoverable = 1;
      }
    else
      {
      len = sizeof(*psocks4) - 1;  // - 1 for the USERID[1]
      s32 gotlen = LowLevelGet(len, socksreq);
      if (((u32)(gotlen)) != len )
        {
        if (verbose_level > 0)
          LogScreen("SOCKS4:%s response from server.\n",
                                     ((gotlen<=0)?("No"):("Invalid")));
        recoverable = 1;
        }
      else //if ( (u32)(gotlen)) == len)
        {
        if (psocks4->VN == 0 && psocks4->CD == 90) // 90 is successful return
          {
          success = 1;
          }
        else if (verbose_level > 0)
          {
          LogScreen("SOCKS4: request rejected%s.\n", 
            (psocks4->CD == 91)
             ? ""
             :
             (psocks4->CD == 92)
             ? ", no identd response"
             :
             (psocks4->CD == 93)
             ? ", invalid identd response"
             :
             ", unexpected response");
          }
        }
      }

    if (tmp_isnonblocking)
      isnonblocking = (MakeNonBlocking()==0);

    return ((success)?(0):((recoverable)?(+1):(-1)));
    }
    
  return 0;
}

// -----------------------------------------------------------------------

int Network::Close(void)
{
  LowLevelCloseSocket();
  gethttpdone = puthttpdone = 0;
  netbuffer.Clear();
  uubuffer.Clear();

  gotuubegin = gothttpend = 0;
  httplength = 0;

  return 0;
}  
    
// -----------------------------------------------------------------------

// Returns length of read buffer.
s32 Network::Get( u32 length, char * data )
{
  int need_close = 0;

  time_t timestop = 0, timenow = 0;

  while ((netbuffer.GetLength() < length) && (timestop >= timenow))
  {
    timenow = time(NULL);
    if (isnonblocking && timestop == 0)
      {
      timestop = timenow + ((time_t)(iotimeout));
      if (verbose_level > 1)
         Log("startget: time()==%u, timeout at %u\n", timenow, timestop );
      }
  
    int nothing_done = 1;

    if ((mode & MODE_HTTP) && !gothttpend)
    {
      // []---------------------------------[]
      // |  Process HTTP headers on packets  |
      // []---------------------------------[]
      uubuffer.Reserve(500);
      s32 numRead = LowLevelGet(uubuffer.GetSlack(), uubuffer.GetTail());
      if (numRead > 0) uubuffer.MarkUsed(numRead);
      else if (numRead == 0) need_close = 1;       // connection closed

      AutoBuffer line;
      while (uubuffer.RemoveLine(line))
      {
        nothing_done = 0;
        if (strncmpi(line, "Content-Length: ", 16) == 0)
        {
          httplength = atoi((const char*)line + 16);
        }
        else if ((svc_hostaddr == 0) &&
          (strncmpi(line, "X-KeyServer: ", 13) == 0))
        {
          if (Resolve( line + 13, &svc_hostaddr, svc_hostport ) < 0)
            svc_hostaddr = 0;
        }
        else if (line.GetLength() < 1)
        {
          // got blank line separating header from body
          gothttpend = 1;
          if (uubuffer.GetLength() >= 6 &&
            strncmpi(uubuffer.GetHead(), "begin ", 6) == 0)
          {
            mode |= MODE_UUE;
            gotuubegin = 0;
          }
          if (!(mode & MODE_UUE))
          {
            if (httplength > uubuffer.GetLength())
            {
              netbuffer += uubuffer;
              httplength -= uubuffer.GetLength();
              uubuffer.Clear();
            } else {
              netbuffer += AutoBuffer(uubuffer, httplength);
              uubuffer.RemoveHead(httplength);
              gothttpend = 0;
              httplength = 0;

              // our http only allows one packet per socket
              nothing_done = gethttpdone = 1;
            }
          }
          break;
        }
      } // while
    }
    else if (mode & MODE_UUE)
    {
      // []----------------------------[]
      // |  Process UU Encoded packets  |
      // []----------------------------[]
      uubuffer.Reserve(500);
      s32 numRead = LowLevelGet(uubuffer.GetSlack(), uubuffer.GetTail());
      if (numRead > 0) uubuffer.MarkUsed(numRead);
      else if (numRead == 0) need_close = 1;       // connection closed

      AutoBuffer line;
      while (uubuffer.RemoveLine(line))
      {
        nothing_done = 0;

        if (strncmpi(line, "begin ", 6) == 0) gotuubegin = 1;
        else if (strncmpi(line, "POST ", 5) == 0) mode |= MODE_HTTP;
        else if (strncmpi(line, "end", 3) == 0)
        {
          gotuubegin = gothttpend = 0;
          httplength = 0;
          break;
        }
        else if (gotuubegin && line.GetLength() > 0)
        {
          // start decoding this single line
          char *p = line.GetHead();
          int n = UU_DEC(*p);
          for (++p; n > 0; p += 4, n -= 3)
          {
            char ch;
            if (n >= 3)
            {
              netbuffer.Reserve(3);
              ch = char((UU_DEC(p[0]) << 2) | (UU_DEC(p[1]) >> 4));
              netbuffer.GetTail()[0] = ch;
              ch = char((UU_DEC(p[1]) << 4) | (UU_DEC(p[2]) >> 2));
              netbuffer.GetTail()[1] = ch;
              ch = char((UU_DEC(p[2]) << 6) | UU_DEC(p[3]));
              netbuffer.GetTail()[2] = ch;
              netbuffer.MarkUsed(3);
            } else {
              netbuffer.Reserve(2);
              if (n >= 1) {
                ch = char((UU_DEC(p[0]) << 2) | (UU_DEC(p[1]) >> 4));
                netbuffer.GetTail()[0] = ch;
                netbuffer.MarkUsed(1);
              }
              if (n >= 2) {
                ch = char((UU_DEC(p[1]) << 4) | (UU_DEC(p[2]) >> 2));
                netbuffer.GetTail()[0] = ch;
                netbuffer.MarkUsed(1);
              }
            }
          }
        }
      } // while
    }
    else
    {
      // []--------------------------------------[]
      // |  Processing normal, unencoded packets  |
      // []--------------------------------------[]
      AutoBuffer tempbuffer;
      s32 wantedSize = ((mode & MODE_HTTP) && httplength) ? httplength : 500;
      tempbuffer.Reserve(wantedSize);

      s32 numRead = LowLevelGet(wantedSize, tempbuffer.GetTail());
      if (numRead > 0)
      {
        nothing_done = 0;
        tempbuffer.MarkUsed(numRead);

        // decrement from total if processing from a http body
        if ((mode & MODE_HTTP) && httplength)
        {
          httplength -= numRead;

          // our http only allows one packet per socket
          if (httplength == 0) nothing_done = gethttpdone = 1;
        }

        // see if we should upgrade to different mode
        if (tempbuffer.GetLength() >= 6 &&
            strncmpi(tempbuffer.GetHead(), "begin ", 6) == 0)
        {
          mode |= MODE_UUE;
          uubuffer = tempbuffer;
          gotuubegin = 0;
        }
        else if (tempbuffer.GetLength() >= 5 &&
            strncmpi(tempbuffer.GetHead(), "POST ", 5) == 0)
        {
          mode |= MODE_HTTP;
          uubuffer = tempbuffer;
          gothttpend = 0;
          httplength = 0;
        }
        else netbuffer += tempbuffer;
      }
      else if (numRead == 0) need_close = 1;
    }

    if (nothing_done)
    {
      if (need_close || gethttpdone) 
        break;
      #if (CLIENT_OS == OS_VMS) || (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_ULTRIX)
        sleep(1); // full 1 second due to so many reported network problems.
      #else
        usleep( 100000 );  // Prevent racing on error (1/10 second)
      #endif
    }
  } // while (netbuffer.GetLength() < blah)


  // transfer back what was read in
  u32 bytesfilled = (netbuffer.GetLength() < length ?
      netbuffer.GetLength() : length);
  memmove(data, netbuffer.GetHead(), (int) bytesfilled);
  netbuffer.RemoveHead(bytesfilled);

  if (need_close) 
    Close();

  if (verbose_level > 1) //DEBUG
    __print_packet("Get", data, bytesfilled );

  return bytesfilled;
}


//--------------------------------------------------------------------------

// returns -1 on error, or 0 on success
s32 Network::Put( u32 length, const char * data )
{
  AutoBuffer outbuf;

  // if the connection is closed, try to reopen it once.
  if ((sock == INVALID_SOCKET) || puthttpdone) 
    {
    if (Open()) 
      return -1;
    }

  if (mode & MODE_UUE)
    {
    /**************************/
    /***  Need to uuencode  ***/
    /**************************/
    outbuf += AutoBuffer("begin 644 query.txt\r\n");

    while (length > 0)
      {
      char line[80];
      char *b = line;

      int n = (int) (length > 45 ? 45 : length);
      length -= n;
      *b++ = UU_ENC(n);

      for (; n > 2; n -= 3, data += 3)
        {
        *b++ = UU_ENC((char)(data[0] >> 2));
        *b++ = UU_ENC((char)(((data[0] << 4) & 060) | ((data[1] >> 4) & 017)));
        *b++ = UU_ENC((char)(((data[1] << 2) & 074) | ((data[2] >> 6) & 03)));
        *b++ = UU_ENC((char)(data[2] & 077));
        }

      if (n != 0)
        {
        char c2 = (char)(n == 1 ? 0 : data[1]);
        char ch = (char)(data[0] >> 2);
        *b++ = UU_ENC(ch);
        *b++ = UU_ENC((char)(((data[0] << 4) & 060) | ((c2 >> 4) & 017)));
        *b++ = UU_ENC((char)((c2 << 2) & 074));
        *b++ = UU_ENC(0);
        }

      *b++ = '\r';
      *b++ = '\n';
      outbuf += AutoBuffer(line, b - line);
      }
    outbuf += AutoBuffer("end\r\n");
    }
  else
    {
    outbuf = AutoBuffer(data, length);
    }

  if ((mode & MODE_HTTP)!=0)
    {
    char header[500];
    sprintf(header, "POST http://%s:%u/cgi-bin/rc5.cgi HTTP/1.0\r\n"
                    "%s%s%s"
                    "Content-Type: application/octet-stream\r\n"
                    "Content-Length: %lu\r\n\r\n",
                    ((svc_hostaddr)?(__inet_ntoa__(svc_hostaddr)):(svc_hostname)),
                    ((unsigned int)(svc_hostport)),
                    ((fwall_userpass[0])?("Proxy-authorization: Basic "):("")),
                    ((fwall_userpass[0])?(fwall_userpass):("")),
                    ((fwall_userpass[0])?("\r\nProxy-Connection: Keep-Alive\r\n"):("")),
                    (unsigned long) outbuf.GetLength());
    #if (CLIENT_OS == OS_OS390)
      __etoa(header);
    #endif
    outbuf = AutoBuffer(header) + outbuf;
    puthttpdone = 1;
    }

  if (verbose_level > 1) //DEBUG
    __print_packet("Put", outbuf, outbuf.GetLength() );

  return (LowLevelPut(outbuf.GetLength(), outbuf) != -1 ? 0 : -1);
}

//=====================================================================
// From here on down we have "bottom half" functions that (a) use socket
// functions or (b) are TCP/IP specific or (c) actually do i/o.
//---------------------------------------------------------------------

int Network::GetHostName( char *buffer, unsigned int len )
{  
  if (!buffer)
    return -1;
  buffer[0]=0;
  if (!len)
    return -1;
  if (len >= sizeof( "127.0.0.1" ))
    strcpy( buffer, "127.0.0.1" );
  #if ( defined(_TIUSER_) || (defined(AF_INET) && defined(SOCK_STREAM)) )
  if (NetCheckIsOK())
    return gethostname(buffer, len);
  #endif
  return -1;
}  

/* ----------------------------------------------------------------------- */

int Network::LowLevelCreateSocket(void)
{
  LowLevelCloseSocket(); //make sure the socket is closed already

  if (!NetCheckIsOK())
    return -1;

#if defined(_TIUSER_)                                              //TLI
  sock = t_open("/dev/tcp", O_RDWR, NULL);
  if ( sock != -1 ) 
    return 0;
  sock = INVALID_SOCKET;
  return -1;
#else
  #if (defined(AF_INET) && defined(SOCK_STREAM)) //BSD socks
  sock = socket(AF_INET, SOCK_STREAM, 0);
  if ( !(( (int)(sock) ) < 0 ) )
    {
    #if (CLIENT_OS == OS_RISCOS)
      int on = 1;
      // allow blocking socket calls to preemptively multitask
      ioctl(sock, FIOSLEEPTW, &on);
    #endif
    return 0; //success
    }
  #endif
  sock = INVALID_SOCKET;
  return -1;
#endif
}

//------------------------------------------------------------------------

int Network::LowLevelCloseSocket(void)
{
#if defined( _TIUSER_ )                                //TLI
   if ( sock != INVALID_SOCKET )
     {
     t_blocking( sock ); /* turn blocking back on */
     if ( t_getstate( sock )!= T_UNBND )
       {
       t_sndrel( sock );   /* initiate close */
       t_rcvrel( sock );   /* wait for conn release by peer */
       t_unbind( sock );   /* close our own socket */
       }
     int rc = t_close( sock );
     sock = INVALID_SOCKET;
     return rc;
     }
#else                                                  //BSD socks
   if ( sock != INVALID_SOCKET )
     {
     LowLevelConditionSocket( CONDSOCK_BLOCKING_ON );
     #if (defined(AF_INET) && defined(SOCK_STREAM))  
     shutdown( sock, 2 );
     #endif
     int retcode = (int)close( sock );
     sock = INVALID_SOCKET;
     return (retcode);
     }
#endif
   return 0;
}   

// -----------------------------------------------------------------------   

int Network::LowLevelConnectSocket( u32 that_address, u16 that_port )
{
  if ( sock == INVALID_SOCKET )
    return -1;
  if (!NetCheckIsOK())
    return -1;

#if defined(_TIUSER_)                                            //TLI
  int rc = -1;
  if ( t_bind( sock, NULL, NULL ) != -1 )
    {
    struct t_call *sndcall = (struct t_call *)t_alloc(sock, T_CALL, T_ADDR);
    if ( sndcall != NULL )
      {
      sndcall->addr.len  = sizeof(struct sockaddr_in);
      sndcall->opt.len   = 0;
      sndcall->udata.len = 0;
      struct sockaddr_in *sin = (struct sockaddr_in *) sndcall->addr.buf;
      sin->sin_addr.s_addr = that_address;
      sin->sin_family = AF_INET;
      sin->sin_port = htons(that_port);
      rc = t_connect( sock, sndcall, NULL);
      if (isnonblocking && rc == -1)
        {
        time_t stoptime = time(NULL) + 
                          (time_t)(1+((iotimeout<=0)?(0):(iotimeout)));
        while (rc == -1 && t_error == TNODATA && time(NULL) < stoptime)
          {
          if (t_rcvconnect(sock, NULL) != -1) 
            rc = 0;
          }
        }
      t_free((char *)sndcall, T_CALL);
      }
    }
  return rc;
#elif (CLIENT_OS == OS_MACOS)
    // The Mac OS client simulates just the most essential socket calls, as a
    // convenience in interfacing to a non-socket network library. "Select" is
    // not available, but the timeout for a connection can be specified.

    // set up the address structure
    struct sockaddr_in sin;
    memset((void *) &sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_port = htons( that_port ); 
    sin.sin_addr.s_addr = that_address;

  // set timeout for connect
  if (iotimeout > 0)
  {
    // timeout for this call must be >0 to not have default used
    socket_set_conn_timeout(sock, iotimeout);
  }

    return(connect(sock, (struct sockaddr *)&sin, sizeof(sin)));
    
#elif defined(AF_INET) //BSD sox

  // set up the address structure
  struct sockaddr_in sin;
  memset((void *) &sin, 0, sizeof(sin));
  sin.sin_family = AF_INET;
  sin.sin_port = htons( that_port ); 
  sin.sin_addr.s_addr = that_address;

  int rc = -1;
  time_t starttime = time(NULL);

  do{
    if ( connect(sock, (struct sockaddr *)&sin, sizeof(sin)) >= 0 )
      {
      rc = 0;
      break;
      }

    #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
    errno = WSAGetLastError();
    #undef  EISCONN
    #define EISCONN WSAEISCONN
    #undef  EINPROGRESS
    #define EINPROGRESS WSAEINPROGRESS
    #undef  EALREADY
    #define EALREADY WSAEALREADY
    #undef  EWOULDBLOCK
    #define EWOULDBLOCK WSAEWOULDBLOCK
    #undef  ETIMEDOUT
    #define ETIMEDOUT WSAETIMEDOUT
    if (errno == WSAEINVAL) /* ws1.1 returns WSAEINVAL instead of WSAEALREADY */
      errno = EALREADY;  
    #elif (CLIENT_OS == OS_OS2)
    errno = sock_errno();
    #endif

    if (isnonblocking == 0)
      {
      rc = -1;
      break;
      }
    if (errno == EISCONN)
      {
      rc = 0;
      break;
      }
    if (errno != EINPROGRESS && errno != EALREADY && errno != EWOULDBLOCK )
      {
      rc = -1;
      break;
      }
      
    if ( (time(NULL)-starttime) > iotimeout )
      {
      rc = -1;
      #ifndef ERRNO_IS_UNUSABLE_FOR_CONN_ERRMSG
      errno = ETIMEDOUT;
      #endif
      break;
      }
    sleep(1);
    rc = -1;
    } while (isnonblocking); /* always true */
  
  return rc;  

#else //no socket support

  int rc = -1;
  if (!that_address && !that_port) /* use up variables */
    rc = -1;  
  return rc;

#endif
}  

//------------------------------------------------------------------------

#if (!defined(AF_INET) || !defined(SOCK_STREAM))  
int Network::Resolve( const char *, u32 *, int )
{ return -1; }
#else
#include "netres.cpp"
#endif

// -----------------------------------------------------------------------   

// Returns length of sent data or -1 if error
s32 Network::LowLevelPut(u32 length, const char *data)  
{                                                       
  if ( sock == INVALID_SOCKET )
    return -1;
  if (!NetCheckIsOK())                                  
    return -1;
#if defined(_TIUSER_)                                          //TLI
  int rc = 0;
  unsigned int totalsent = 0;

  if (length > 0)
    {
    unsigned sendquota = 512;
    struct t_info info;
    if ( t_getinfo( sock, &info ) != -1)
      {
      if (info.tdsu > 0)
        sendquota = info.tdsu;
      else if (info.tdsu == -1) /* no limit */
        sendquota = (unsigned int)length;
      else if (info.tdsu == 0) /* no boundaries */
        sendquota = 1;
      else //if (info.tdsu == -2) /* normal send not supp'd (ever happens?)*/
        return -1;
      }
    while (length != 0)
      {
      unsigned int sendlen = (unsigned int)length;
      if (sendlen > sendquota)
        sendlen = sendquota;
      rc = t_snd(sock, (char *)data, (unsigned int)sendlen, 
                  (((sendlen < (unsigned int)length)?(T_MORE):(0))) );
      if (rc == -1)
        {
        if (t_errno == TFLOW ) /* sending too fast */
          {
          usleep(500000); // 0.5 secs
          continue;
          }
        break;
        }
      totalsent += rc;
      data += rc;
      length -= rc;
      }
    }
  return (s32)((rc == -1 && totalsent == 0)?(-1):(totalsent));
#else                                                         //BSD sox
  u32 writelen = write(sock, (char*)data, length);
  return (s32) (writelen != length ? -1 : (s32)writelen);
#endif  
}

// ----------------------------------------------------------------------

// Returns length of read buffer. 0 if conn closed or -1 if no data waiting
s32 Network::LowLevelGet(u32 length, char *data)
{
  if ( sock == INVALID_SOCKET )
    return -1;
  if (!NetCheckIsOK())
    return -1;

#if defined(_TIUSER_)                                     //TLI
  int flags, rc;
  if (( rc = t_rcv( sock, data, length, &flags ) ) == -1 )
    {
    if ( t_errno == TNODATA ) 
      return (s32)(-1); 
    return 0; /* TLOOK (async event) or TSYSERR */
    }
  return (s32)(rc);
#else                                                     //BSD 4.3 sockets
  #if defined(SELECT_FIRST)
    fd_set rs;
    timeval tv = {0,0};
    FD_ZERO(&rs);
    FD_SET(sock, &rs);
    select(sock + 1, &rs, NULL, NULL, &tv);
    if (!FD_ISSET(sock, &rs)) return -1;
  #endif

  s32 numRead = read(sock, data, length);
  if (numRead < 0) numRead = -1;

  #if (CLIENT_OS == OS_HPUX)
    // HPUX incorrectly returns 0 on a non-blocking socket with
    // data waiting to be read instead of -1.
    if (numRead == 0) numRead = -1;
  #endif

  return numRead;
#endif //TLI or BSD sockets
}

// ----------------------------------------------------------------------

int Network::LowLevelConditionSocket( unsigned long cond_type )
{  
  if ( sock == INVALID_SOCKET )
    return -1;

  if ( cond_type==CONDSOCK_BLOCKING_ON || cond_type==CONDSOCK_BLOCKING_OFF )
    {
    #if defined(_TIUSER_)                                    //TLI
      if ( cond_type == CONDSOCK_BLOCKING_ON )
        return ( t_blocking( sock ) );
      else         /* same as an ioctl call with I_SETDELAY and 1 as args. */
        return ( t_nonblocking( sock ) );
    #elif (!defined(FIONBIO) && !(defined(F_SETFL) && (defined(FNDELAY) || defined(O_NONBLOCK))))
      return -1;
    #elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
      unsigned long flagon = ((cond_type == CONDSOCK_BLOCKING_OFF)?(1):(0));
      return ioctlsocket(sock, FIONBIO, &flagon);
    #elif ((CLIENT_OS == OS_VMS) && defined(__VMS_UCX__))
      // nonblocking sockets not directly supported by UCX
      // - DIGITAL's work around requires system privileges to use
      return -1;
    #elif ((CLIENT_OS == OS_VMS) && defined(MULTINET))
      unsigned long flagon = ((cond_type == CONDSOCK_BLOCKING_OFF)?(1):(0));
      return socket_ioctl(sock, FIONBIO, &flagon);
    #elif (CLIENT_OS == OS_RISCOS)
      int flagon = ((cond_type == CONDSOCK_BLOCKING_OFF) ? (1): (0));
      return ioctl(sock, FIONBIO, &flagon);
    #elif (CLIENT_OS == OS_OS2)
      int flagon = ((cond_type == CONDSOCK_BLOCKING_OFF) ? (1): (0));
      return ioctl(sock, FIONBIO, (char *) &flagon, sizeof(flagon));
    #elif (CLIENT_OS == OS_AMIGAOS)
      char flagon = ((cond_type == CONDSOCK_BLOCKING_OFF) ? (1): (0));
      return IoctlSocket(sock, FIONBIO, &flagon);
    #elif (CLIENT_OS == OS_MACOS)
      char flagon = ((cond_type == CONDSOCK_BLOCKING_OFF) ? (1): (0));
      return ioctl(sock, FIONBIO, &flagon);    
    #elif (defined(F_SETFL) && (defined(FNDELAY) || defined(O_NONBLOCK)))
      {
      int flag, res, arg;
      #if (defined(FNDELAY))
        flag = FNDELAY;
      #else
        flag = O_NONBLOCK;
      #endif
      arg = ((cond_type == CONDSOCK_BLOCKING_OFF) ? (flag): (0) );

      if (( res = fcntl(sock, F_GETFL, flag ) ) == -1)
        return -1;
      if ((arg && res) || (!arg && !res))
        return 0;
      if ((res = fcntl(sock, F_SETFL, arg )) == -1)
        return -1;
      if (( res = fcntl(sock, F_GETFL, flag ) ) == -1)
        return -1;
      if ((arg && res) || (!arg && !res))
        return 0;
      }
    #else
      return -1;
    #endif
    }
  return -1;
}

// ----------------------------------------------------------------------
