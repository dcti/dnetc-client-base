// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: network.cpp,v $
// Revision 1.78  1999/02/01 18:02:44  cyp
// undid last SillyB change. (so, whats new?)
//
// Revision 1.77  1999/02/01 08:19:59  silby
// Network class once again allows autofindkeyserver to be disabled.
//
// Revision 1.76  1999/01/31 20:19:09  cyp
// Discarded all 'bool' type wierdness. See cputypes.h for explanation.
//
// Revision 1.75  1999/01/29 18:56:32  jlawson
// fixed formatting.  changed some int vars to bool.
//
// Revision 1.74  1999/01/29 04:14:11  cyp
// some more time spent trying to get http to work... :(
//
// Revision 1.73  1999/01/23 21:34:07  patrick
// OS2-EMX coredumps in __print_packet (undefined for the moment)
// EMX uses errno instead of sock_errno
//
// Revision 1.72  1999/01/21 22:01:04  cyp
// fixed LowLevelSend() which didn't know /anything/ about non-blocking sox.
//
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


#if (!defined(lint) && defined(__showids__))
const char *network_cpp(void) {
return "@(#)$Id: network.cpp,v 1.78 1999/02/01 18:02:44 cyp Exp $"; }
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

//#defined PERMIT_NODEZERO_ACCESS //allow access to fullserver on nodezero 
                                  //FOR DCTI USE ONLY!!!

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

// short circuit the u32 -> in_addr.s_addr ->inet_ntoa method.
// besides, it works around context issues.
static const char *__inet_ntoa__(u32 addr)
{
  static char buff[sizeof("255.255.255.255  ")];
  char *p = (char *)(&addr);
  sprintf( buff, "%d.%d.%d.%d",
      (p[0]&255), (p[1]&255), (p[2]&255), (p[3]&255) );
  return buff;
}  

static void __print_packet( const char *label, const char *apacket, int alen )
{
  int i,n;
  for (i = 0; i < alen; i += 16)
    {
    char buffer[128];
    char *p = &buffer[0];
    p += sprintf(buffer,"%s %04x: ", label, i );
    for (n = 0; (n < 16 && ((n + i) < alen)); n++)
      p += sprintf(p,"%02x ", (unsigned char)apacket[n+i]);
    for (; n < 16; n++)
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

static void __hostnamecpy( char *dest,
    const char *source, unsigned int maxlen)
{
  unsigned int len = 0;
  while (*source && isspace(*source))
    source++;
  while (((++len) < maxlen) && *source && !isspace(*source))
    *dest++ = (char)tolower(*source++);
  *dest = 0;
  return;
}  

static int __fixup_dnethostname( char *host, int *portP )
{
  int autofind = 0;
  if (host[0]==0 || strcmpi(host,"auto")==0 || strcmpi(host,"(auto)")==0)
    autofind = 1;
  else if (!autofind && strchr( host, '.' )!=NULL)
    {
    char *p = host;
    int len = 0;
    while (*p && isspace(*p))
      p++;
    while (*p && !isspace(*p))
      host[len++]=(char)tolower(*p++);
    host[len]=0;
    if ( len>15 && strcmp( &host[len-15], "distributed.net" )==0 &&
         (( len == 15 ) || host[len-16]=='.'))
      {
      if (len == 15)
        autofind = 1;
      #ifdef PERMIT_NODEZERO_ACCESS
      else if ( strcmp( host, "nodezero.distributed.net")==0 )
        autofind = (access("dcti_test.ok",0)!=0); // FOR DCTI USE ONLY!!!
      #endif
      else if ( strcmp( strchr( host, '.' ), ".v27.distributed.net" )!=0 )
        autofind = 1;
      else
        {
        int i, isvalid=0;
        const char *dzones[]={"us","euro","asia","aussie","jp"};
        for (i=0;(!isvalid && i<((int)(sizeof(dzones)/sizeof(dzones[0]))));i++)
          {
          int len2 = strlen(dzones[i]);
          if ( memcmp( dzones[i], host, len2 )==0)
            {
            int foundport = ((host[len2]=='.')?(2064):(atoi(&host[len2])));
            if (foundport != 80 && foundport != 23 && foundport != 2064)
              break;
            else if (*portP == 0) //note: the hostname determines port
              *portP = foundport; //not viceversa. 2064 also accepts http/uue
            else if (*portP != 3064 && (*portP != foundport))
              break;
            isvalid = 1;
            }
          }
        if (!isvalid)
          autofind = 1;
        }
      }
    }
  if (autofind)
    host[0]=0;
  return autofind;
}    

//======================================================================

Network::Network( const char * servname, int servport, int _nofallback,
                  int _iotimeout, int _enctype, const char *_fwallhost, 
                  int _fwallport, const char *_fwalluid)
{
  // check that the packet structures have been correctly packed
  size_t dummy;
  if (((dummy = offsetof(SOCKS4, USERID[0])) != 8) ||
     ((dummy = offsetof(SOCKS5METHODREQ, Methods[0])) != 2) ||
     ((dummy = offsetof(SOCKS5METHODREPLY, end)) != 2) ||
     ((dummy = offsetof(SOCKS5USERPWREPLY, end)) != 2) ||
     ((dummy = offsetof(SOCKS5, end)) != 10))
    LogScreen("Network::Socks Incorrectly packed structures.\n");

  // intialize communication parameters
  server_name[0] = 0;
  if (servname)
     __hostnamecpy( server_name, servname, sizeof(server_name));
  server_port = servport;
  autofindkeyserver = __fixup_dnethostname(server_name,&server_port);

  reconnected = 0;
  nofallback = _nofallback;
  sock = INVALID_SOCKET;

  gotuubegin = gothttpend = 0;
  httplength = 0;
  
  fwall_hostaddr = svc_hostaddr = conn_hostaddr = 0;
  fwall_hostname[0] = fwall_userpass[0] = 0;

  mode = startmode = 0;
  if (_enctype == 1 /*uue*/ || _enctype == 3 /*http+uue*/)
    {
    startmode |= MODE_UUE;
    }
  if (_enctype == 2 /*http*/ || _enctype == 3 /*http+uue*/)
    {
    startmode |= MODE_HTTP;
    if (_fwallhost && _fwallhost[0])
      {
      fwall_hostport = _fwallport;
      if (_fwalluid)
        strncpy( fwall_userpass, _fwalluid, sizeof(fwall_userpass));
      __hostnamecpy( fwall_hostname, _fwallhost, sizeof(fwall_hostname));
      }
    }
  else if (_enctype == 4 /*socks4*/ || _enctype == 5 /*socks5*/)
    {
    if (_fwallhost && _fwallhost[0])
      {
      startmode |= ((_enctype == 4)?(MODE_SOCKS4):(MODE_SOCKS5));
      fwall_hostport = _fwallport;
      __hostnamecpy(fwall_hostname, _fwallhost, sizeof(fwall_hostname));
      if (_fwalluid)
        strncpy(fwall_userpass, _fwalluid, sizeof(fwall_userpass));
      if (fwall_hostport == 0)
        fwall_hostport = 1080;
      }
    }
  mode = startmode;

  isnonblocking = 0;      /* whether the socket could be set non-blocking */
  iotimeout = _iotimeout; /* if iotimeout is <0, use blocking calls */
  if (iotimeout < 0)
    iotimeout = -1;
  else if (iotimeout < 5)
    iotimeout = 5;
  else if (iotimeout > 300)
    iotimeout = 300;

  #ifdef NETDEBUG
  verbose_level = 2;
  #elif defined(VERBOSE_OPEN)
  verbose_level = 1;
  #else
  verbose_level = 0; //quiet
  #endif
  
  return;
}

//----------------------------------------------------------------------

Network::~Network(void)
{
  Close();
}

//----------------------------------------------------------------------

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
  isnonblocking = (iotimeout >= 0 && MakeNonBlocking() == 0);

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
    const char *netcheckfailed =
        "Network::Open Error - TCP/IP Connection Lost.\n";

    if (CheckExitRequestTrigger())
      return -1;

    if ((startmode & (MODE_SOCKS4 | MODE_SOCKS5)) != 0 &&
        (fwall_hostname[0] == 0 || fwall_hostport == 0))
      {
      Log("Network::Invalid %s proxy hostname or port.\n"
          "Connect cancelled.\n",
          ((startmode & MODE_HTTP) ? ("HTTP") : ("SOCKS")));
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

      if (svc_hostname[0] == 0 || retries > (preftries - 1)) /* fallback*/
        {
        svc_hostaddr = 0; 
        svc_hostname = "rc5proxy.distributed.net";
        if (svc_hostport != 80 && svc_hostport != 23 && 
          svc_hostport != 2064 && svc_hostport != 3064 /* && 
          svc_hostport != 21 && svc_hostport != 25 && svc_hostport != 110 */)
          {
          svc_hostport = 0;
          }
        autofindkeyserver = 1;
        }
      if (svc_hostport == 0)
          {
        #if 0                             //2064 also accepts http/uue - cyp
        if ((startmode & MODE_HTTP) != 0)
          svc_hostport = 80;
        else if ((startmode & MODE_UUE) != 0)
          svc_hostport = 23;
        else
        #endif
          svc_hostport = DEFAULT_PORT;
        }

      if (!NetCheckIsOK())
        {
        success = 0;
        retries = maxtries;
        LogScreen(netcheckfailed);
        }
      else if ((startmode & MODE_PROXIED) != 0 && fwall_hostname[0] != 0)
        {
        if ((startmode & MODE_HTTP) == 0)
          {
          // we always re-resolve unless http.
          // socks5 needs a 'good' hostname
          // (obtained from resolve_hostname)
          // if name resolution fails - cyp
          svc_hostaddr = 0;
          }
        if (Resolve( fwall_hostname, &fwall_hostaddr, fwall_hostport ) < 0)
          {
          success = 0;
          fwall_hostaddr = 0;
          if (verbose_level > 0)
            LogScreen("Network::failed to resolve name \"%s\"\n", 
                       resolve_hostname );

          // unrecoverable error. retry won't help
          retries = maxtries;
          }
        else if (svc_hostaddr == 0) /* always 1 unless http */
          {
          if ( Resolve(svc_hostname, &svc_hostaddr, svc_hostport ) < 0 )
            {
            svc_hostaddr = 0;
            if ((startmode & (MODE_SOCKS4 /*| MODE_SOCKS5*/)) != 0)
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
      if (verbose_level > 0 && !reconnected)
        {
        const char *targethost = svc_hostname;
        unsigned int len = strlen( targethost );
        const char sig[]=".distributed.net";
        char scratch[sizeof(sig) + 20];
        if ((len > (sizeof( sig ) - 1) && autofindkeyserver &&
           strcmpi( &targethost[(len-(sizeof( sig )-1))], sig ) == 0))
          {
          targethost = sig + 1;
          if (svc_hostaddr)
            {
            sprintf(scratch, "%s %s", sig+1, __inet_ntoa__(svc_hostaddr));
            targethost = scratch;
            }
          }
        
        if ((startmode & MODE_PROXIED) == 0 || fwall_hostname[0] == 0 /*http*/)
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
          ( "Connecting to %s:%u\n"
            "via %s proxy %s:%u\n",
            targethost,
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
              ((isnonblocking) ? ("non-") : ("")) );
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
              ((isnonblocking) ? ("non-") : ("")) );
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
          if      (err == WSAEADDRINUSE)
            msg = "The specified address is already in use";
          else if (err == WSAEADDRNOTAVAIL)
            msg = "The specified address is not available from the local machine";
          else if (err == WSAECONNREFUSED)
            msg = "The attempt to connect was rejected. Destination may be busy.";
          else if (err == WSAEISCONN)
            msg = "The socket is already connected"; //udp only?
          else if (err == WSAENETUNREACH)
            msg = "The network cannot be reached from this host at this time";
          else if (err == WSAENOBUFS)
            msg = "No buffer space is currently available";
          else if (err == WSAETIMEDOUT)
            msg = "Attempt to connect timed out";
          LogScreen( " %s  Error %d (%s)\n",
                     CliGetTimeString(NULL,0), err, msg );
        #elif (!defined(ERRNO_IS_UNUSABLE_FOR_CONN_ERRMSG))
          if (NetCheckIsOK()) //use errno only if netcheck says ok.
            {
            #if defined(_TIUSER_)
              LogScreen( " %s  Error %d (%s)\n", 
                  CliGetTimeString(NULL, 0), t_errno, 
                ((t_errno >= t_nerr) ? ("undefined error") : (t_errlist[t_errno])) );
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

    #if 0
    int tmp_isnonblocking = (isnonblocking != 0);
    if (isnonblocking)
      {
      isnonblocking = (MakeBlocking() != 0);
      if (isnonblocking)
        {
        if (verbose_level > 0)
          LogScreen("SOCKS5: unable to temporarily revert to blocking mode\n");
        iotimeout = -1; //force blocking mode in next try
        return +1; //recoverable
        }
      }
    #endif

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
    if (LowLevelPut(socksreq, (int) len) != (int) len)
      {
      if (verbose_level > 0)
        LogScreen("SOCKS5: error sending negotiation request\n");
      recoverable = 1;
      }
    else if (LowLevelGet(socksreq, 2) != 2)
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

      if (LowLevelPut(socksreq, len) != (int) len)
        {
        if (verbose_level > 0)
          LogScreen("SOCKS5: failed to send sub-negotiation request.\n");
        recoverable = 1;
        }
      else if (LowLevelGet(socksreq, 2) != 2)
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
        // at this point svc_hostname is a ptr to a resolve_hostname.
        strcpy( p+1, svc_hostname );
        *p = (char)(len = strlen( p+1 )); 
        p += (++len);
        *((u16 *)(p)) = (u16)htons(svc_hostport);
        packetsize = (10-sizeof(u32))+len;
        }
        
      if (LowLevelPut(socksreq, packetsize) != packetsize)
        {
        if (verbose_level > 0)
          LogScreen("SOCKS5: failed to send connect request.\n");
        recoverable = 1;
        }
      else if (LowLevelGet( socksreq, packetsize) < 10 /*ok for both atyps*/)
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
      
    #if 0
    if (tmp_isnonblocking)
      isnonblocking = (MakeNonBlocking() == 0);
    #endif
      
    return ((success) ? (0) : ((recoverable) ? (+1) : (-1)));
    } //if (startmode & MODE_SOCKS5)

  if (startmode & MODE_SOCKS4)
    {
    int success = 0; //assume failed
    int recoverable = 0; //assume non-recoverable error (negotiation failure)

    char socksreq[128];  // min sizeof(fwall_userpass) + sizeof(SOCKS4)
    SOCKS4 *psocks4 = (SOCKS4 *)socksreq;
    u32 len;

    #if 0
    int tmp_isnonblocking = (isnonblocking != 0);
    if (isnonblocking)
      {
      isnonblocking = (MakeBlocking() != 0);
      if (isnonblocking)
        {
        if (verbose_level > 0)
          LogScreen("SOCKS4: unable to temporarily revert to blocking mode\n");
        iotimeout = -1; //force blocking mode in next try
        return +1; //recoverable
        }
      }
    #endif

    // transact a request to the SOCKS4 proxy giving the
    // destination ip/port and username and process its reply.

    psocks4->VN = 4;
    psocks4->CD = 1;  // CONNECT
    psocks4->DSTPORT = (u16)htons(svc_hostport); //(u16)htons((server_name[0]!=0)?((u16)port):((u16)DEFAULT_PORT));
    psocks4->DSTIP = svc_hostaddr;   //lasthttpaddress;
    strncpy(psocks4->USERID, fwall_userpass, sizeof(fwall_userpass));

    len = sizeof(*psocks4) - 1 + strlen(fwall_userpass) + 1;
    if (LowLevelPut(socksreq, len) != (int) len)
      {
      if (verbose_level > 0)
        LogScreen("SOCKS4: Error sending connect request\n");
      recoverable = 1;
      }
    else
      {
      len = sizeof(*psocks4) - 1;  // - 1 for the USERID[1]
      int gotlen = LowLevelGet(socksreq,len);
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

    #if 0
    if (tmp_isnonblocking)
      isnonblocking = (MakeNonBlocking() == 0);
    #endif

    return ((success) ? (0) : ((recoverable) ? (+1) : (-1)));
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
int Network::Get( char * data, int length )
{
  time_t starttime = 0;
  int need_close = 0;

  int tmp_isnonblocking = (isnonblocking != 0); //we handle timeout ourselves
  isnonblocking = 0;                 //so stop LowLevelGet() from doing it.

  while (netbuffer.GetLength() < (u32)length)
  {
    int nothing_done = 1;

    if (starttime == 0) /* first pass through */
      time(&starttime);
    else if (!tmp_isnonblocking) /* we are blocking, so no more chances */
      break;
    else if ((time(NULL) - starttime) > iotimeout)
      break;

    if ((mode & MODE_HTTP) && !gothttpend)
    {
      // []---------------------------------[]
      // |  Process HTTP headers on packets  |
      // []---------------------------------[]
      uubuffer.Reserve(500);
      int numRead = LowLevelGet(uubuffer.GetTail(), (int)uubuffer.GetSlack());
      if (numRead > 0) uubuffer.MarkUsed((u32)numRead);
      else if (numRead == 0) need_close = 1;       // connection closed

      AutoBuffer line;
      while (uubuffer.RemoveLine(line))
      {
        nothing_done = 0;
        if (strncmpi(line, "Content-Length: ", 16) == 0)
        {
          httplength = atoi((const char*)line + 16);
        }
        else if ( /* (svc_hostaddr == 0) && */
          (strncmpi(line, "X-KeyServer: ", 13) == 0))
        {
          u32 newaddr = 0;
          if (!(Resolve( line + 13, &newaddr, svc_hostport ) < 0))
            {
            svc_hostaddr = newaddr;
            if (verbose_level > 1)
              Log("X-Keyserver: %s\n", __inet_ntoa__(svc_hostaddr));
            }
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
      int numRead = LowLevelGet(uubuffer.GetTail(), (int)uubuffer.GetSlack());
      if (numRead > 0) uubuffer.MarkUsed((u32)numRead);
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
      int wantedSize = ((mode & MODE_HTTP) && httplength) ? (int)httplength : 500;
      tempbuffer.Reserve((u32)wantedSize);

      int numRead = LowLevelGet(tempbuffer.GetTail(),wantedSize);
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

  isnonblocking = (tmp_isnonblocking!=0); //restore the old state

  if (need_close) 
    Close();

  // transfer back what was read in
  int bytesfilled = length;
  if (((u32)(netbuffer.GetLength())) < ((u32)(length)))
    bytesfilled = netbuffer.GetLength();
  if (bytesfilled != 0)
    {
    memmove(data, netbuffer.GetHead(), bytesfilled);
    netbuffer.RemoveHead((u32)bytesfilled);
    #if !defined (__EMX__) // coredumps on OS2
    if (verbose_level > 1) //DEBUG
      __print_packet("Get", data, bytesfilled );
    #endif
    }

  if (verbose_level > 1) //DEBUG
    Log("Get: toread:%d read:%d\n", length, bytesfilled );

  return bytesfilled;
}

//--------------------------------------------------------------------------

// returns bytes sent, -1 on error
int Network::Put( const char * data, int length )
{
  AutoBuffer outbuf;
  int requested_length = length;

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

  if (mode & MODE_HTTP)
  {
    char header[500];
    sprintf(header,
            "POST http://%s:%u/cgi-bin/rc5.cgi HTTP/1.0\r\n"
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

  #if !defined (__EMX__) // coredumps on OS2
  if (verbose_level > 1) //DEBUG
    __print_packet("Put", outbuf, outbuf.GetLength() );
  #endif

  int towrite = (int)outbuf.GetLength();
  int written = LowLevelPut(outbuf,towrite);

  if (verbose_level > 1) //DEBUG
    Log("Put: towrite:%d written:%d success:%d\n", towrite, written, (towrite==written) );
  
  return ((towrite == written)?(requested_length):(-1));
}

//=====================================================================
// From here on down we have "bottom half" functions that (a) use socket
// functions or (b) are TCP/IP specific or (c) actually do i/o.
//---------------------------------------------------------------------

int Network::GetHostName( char *buffer, unsigned int len )
{  
  if (!buffer || !len)
    return -1;
  buffer[0] = 0;
  if (len < 2)
    return -1;
  buffer[len-1] = 0;
  strncpy( buffer, "1.0.0.127.in-addr.arpa", len );
  if (buffer[len-1] != 0)
    buffer[0] = 0;
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
      // allow blocking socket calls to preemptively multitask
      int on = 1;
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
    if ( t_getstate( sock ) != T_UNBND )
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
  if (!that_address || !that_port)
    return -1;  

#if defined(_TIUSER_)                                         //OSI/XTI/TLI
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
      if (isnonblocking && rc == -1 && t_error == TNODATA) 
        {
        time_t stoptime = time(NULL) + (time_t)iotimeout;
        while (rc == -1)
          {
          if (t_rcvconnect(sock, NULL) != -1) 
            rc = 0;
          else if (t_error == TNODATA)
            break;
          else if (time(NULL) > stoptime)
            break;
          else
            usleep(250000);
            }
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

  do
    {
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
    #elif (CLIENT_OS == OS_OS2) && !defined(__EMX__)
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
    } while (isnonblocking); /* always 1 */
  
  return rc;  

#else //no socket support
  return -1;
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
int Network::LowLevelPut(const char *data,int length)
{                                                       
  if ( sock == INVALID_SOCKET )
    return 0; /* sock closed */
  if (!NetCheckIsOK())                                  
    return -1;
  if (length == 0)
    return -1;
    
  u32 totaltowrite = length;
  u32 totalwritten = 0;
  u32 sendquota = 1500; /* how much to send per send() call */
  time_t timenow = 0, stoptime = 0;
  int sleptcount = 0; /* ... in a row */
  int sleepms = 250; /* sleep time in millisecs. adjust here if needed */

  if (isnonblocking)
    stoptime = (time(NULL))+(time_t)iotimeout;

  #if defined(_TIUSER_)
  sendquota = 512;
  struct t_info info;
  if ( t_getinfo( sock, &info ) != -1)
    {
    if (info.tdsu > 0)
      sendquota = info.tdsu;
    else if (info.tdsu == -1) /* no limit */
      sendquota = ((length > INT_MAX)?(INT_MAX):(length));
    else if (info.tdsu == 0) /* no boundaries */
      sendquota = 1;
    else //if (info.tdsu == -2) /* normal send not supp'd (ever happens?)*/
      return -1;
    }
  #endif
  
  #if (CLIENT_OS == OS_WIN16 || CLIENT_OS == OS_WIN32S)
  if (sendquota > 0x7FFF)  
    sendquota = 0x7FFF;
  #else
  if (sendquota > INT_MAX)  
    sendquota = INT_MAX;
  #endif

  do
    {
    int towrite = (int)((((u32)length)>((u32)sendquota))?(sendquota):(length));

    #if defined(_TIUSER_)                              //TLI/XTI
    int written = -2;
    while (written == -2)
        {
      int flag = (((length - towrite)==0) ? (0) : (T_MORE));
      written = t_snd(sock, (char *)data, (unsigned int)towrite, flag );
      if (written == -1 && t_errno == TFLOW ) /* sending too fast */
          {
        usleep(500000); // 0.5 secs
        written = -2;
          }
        }
    #else                                              //BSD 4.3 sockets
    int written = send(sock, (char*)data, towrite, 0 );
    /*
      When used on a blocking SOCK_STREAM socket, send() requests block 
      until all of the client's data can be sent or buffered by the socket. 
      When used on a nonblocking socket, send() requests send or buffer the 
      maximum amount of data that can be handled without blocking and 
      return the amount that was taken. If no data is taken, they return 
      a value of -1, indicating an EWOULDBLOCK error.
    */
    #endif

    if (written > 0)
      {
      totalwritten += written;    
      length -= written;
      data += written;
      sleptcount = 0;
      if (length == 0) /* sent all */
        break;
      }
    if (isnonblocking == 0)
      {
      if (written <= 0)
        break;
      }
    else //if (isnonblocking) 
      {
      if (time(&timenow) > stoptime)
        {
        if (written <= 0 && sleptcount > 10)
          break;
        else if (written > 0)
          stoptime = timenow+1;
        }
      if (written <= 0) /* nothing sent but haven't timed out yet */
        {
        unsigned long sleepdur = ((unsigned long)(++sleptcount)) * sleepms;
        if (sleepdur > 1000000UL)
          sleep( sleepdur / 1000000UL );
        if ((sleepdur % 1000000UL) != 0)
          usleep( sleepdur % 1000000UL );
        }
      }
    } while (length);

  if (verbose_level > 1)
    Log("LLPut: towrite=%d, written=%d\n", totaltowrite, totalwritten );

  return ((totalwritten != 0) ? ((int)totalwritten) : (-1));
}

// ----------------------------------------------------------------------

// Returns length of read buffer or 0 if conn closed or -1 if no data waiting+timeout
int Network::LowLevelGet(char *data,int length)
{
  if ( sock == INVALID_SOCKET )
    return 0; //conn closed
  if (!NetCheckIsOK())
    return -1;
  if (!length)
    return -1;

  u32 totalread = 0;
  u32 writequota = 1500;
  time_t timenow = 0, stoptime = 0;
  int sleptcount = 0; /* ... in a row */
  int sleepms = 250; /* sleep time in millisecs. adjust here if needed */
  int sockclosed = 0;

  if (isnonblocking)
    stoptime = (time(NULL))+(time_t)iotimeout;

  #if ((CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32))
  if (writequota > 0x7FFF)
    writequota = 0x7FFF;
  #else
  if (writequota > INT_MAX)
    writequota = INT_MAX;
  #endif

  do
    {
    int toread = (int)((((u32)length)>((u32)writequota))?(writequota):(length));
    int bytesread = 0;

    #if defined(_TIUSER_)                               //TLI/XTI
      int flags;
      bytesread = t_rcv( sock, data, toread, &flags );
      if (bytesread == -1)
          {
        if ( t_errno != TNODATA ) /* TLOOK (async event) or TSYSERR */
          bytesread = 0; /* set as socket closed */
          }
      else if (bytesread == 0) /* should never happen? */
        bytesread = -1; /* set as none waiting */
    #else                                               //BSD 4.3
      #if defined(SELECT_FIRST) || (CLIENT_OS == OS_HPUX)
      fd_set rs;
      timeval tv = {0,0};
      FD_ZERO(&rs);
      FD_SET(sock, &rs);
      bytesread = select(sock + 1, &rs, NULL, NULL, &tv);
      if (bytesread < 0)   /* error */
        bytesread = 0; /* == sock closed */
      else if (bytesread != 1) /* not ready */
        bytesread = -1;
      else
      #endif
      bytesread = read(sock, data, toread );
      
      #if (CLIENT_OS == OS_HPUX)
      // HPUX incorrectly returns 0 on a non-blocking socket with
      // data waiting to be read instead of -1.
      if (bytesread == 0 && isnonblocking) 
        bytesread = -1;
      #endif
    #endif /* TLI/XTI or BSD */
    
    if (verbose_level > 1) //DEBUG
      Log("LLGet: read(%d)-> %d\n", toread, bytesread );

    if (bytesread == 0) /* sock closed */
      {
      sockclosed = 1;
      break;
      }
    if (bytesread > 0) /* have data */
      {
      totalread += bytesread;
      data += bytesread;
      length -= bytesread;
      if (length == 0) /* done all */
        break;
      }
    if (!isnonblocking)
      break;
    if (bytesread < 0)
      {
      if (totalread != 0)
        break;
      if (time(&timenow) > stoptime)
        break;
      ++sleptcount;
      }
    unsigned long sleepdur = ((unsigned long)(sleptcount+1)) * sleepms;
    if (sleepdur > 1000000UL)
      sleep( sleepdur / 1000000UL );
    if ((sleepdur % 1000000UL) != 0)
      usleep( sleepdur % 1000000UL );
      } while (length);

  if (verbose_level > 1) //DEBUG
    Log("LLGet: got %u (requested %u) sockclosed:%s\n", 
              totalread, totalread+length, ((sockclosed)?("yes"):("no")));
    
  if (totalread!=0)
    return (int)totalread;
  if (sockclosed)
    return 0;
  return -1;
}

// ----------------------------------------------------------------------

int Network::LowLevelConditionSocket( unsigned long cond_type )
{  
  if ( sock == INVALID_SOCKET )
    return -1;

  if ( cond_type == CONDSOCK_BLOCKING_ON ||
       cond_type == CONDSOCK_BLOCKING_OFF )
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
      int ok = ioctl(sock, FIONBIO, &flagon);
      if (ok && !flagon)
          {
        int on = 1;
        // allow blocking socket calls to preemptively multitask
        ioctl(sock, FIOSLEEPTW, &on);
          }
    #elif (CLIENT_OS == OS_OS2)
      int flagon = ((cond_type == CONDSOCK_BLOCKING_OFF) ? (1): (0));
      return ioctl(sock, FIONBIO, (char *) &flagon, sizeof(flagon));
    #elif (CLIENT_OS == OS_AMIGAOS)
      char flagon = ((cond_type == CONDSOCK_BLOCKING_OFF) ? (1): (0));
      return IoctlSocket(sock, FIONBIO, &flagon);
    #elif (CLIENT_OS == OS_DOS)
      return ((cond_type==CONDSOCK_BLOCKING_OFF)?(0):(-1)); //always non-blocking
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

