/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/
const char *network_cpp(void) {
return "@(#)$Id: network.cpp,v 1.113 2000/01/02 04:07:20 cyp Exp $"; }

//----------------------------------------------------------------------

//#define TRACE
//#define DEBUGTHIS

#include "cputypes.h"
#include "baseincs.h"  // standard stuff
#include "sleepdef.h"  // Fix sleep()/usleep() macros there! <--
#include "autobuff.h"  // Autobuffer class
#include "cmpidefs.h"  // strncmpi(), strcmpi()
#include "logstuff.h"  // LogScreen()
#include "base64.h"    // base64_encode()
#include "clitime.h"   // CliGetTimeString(NULL,1);
#include "triggers.h"  // CheckExitRequestTrigger()
#include "util.h"      // trace
#include "netres.h"    // NetResolve()
#include "network.h"   // thats us

#if (CLIENT_OS == OS_DOS)
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

#ifndef MIPSpro
# pragma pack(1)               // no padding allowed
#endif /* ! MIPSpro */

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

#ifndef MIPSpro
# pragma pack()
#endif /* ! MIPSpro */

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

const char *__print_packet( const char *label, const char *apacket, unsigned int alen )
{
  unsigned int i;
  for (i = 0; i < alen; i += 16)
  {
    char buffer[128];
    char *p, *q; unsigned int n;
    sprintf(buffer,"%s %04x: ", label, i );
    q = 48 + (p = &buffer[strlen(buffer)]);
    for (n = 0; n < 16; n++)
    {
      unsigned int c = ' ';
      p[0] = p[1] = ' ';
      if (( n + i ) < alen )
      {
        static const char *tox="0123456789abcdef";
        c = (((unsigned int)apacket[n+i]) & 0xff);
        p[0] = (char)tox[c>>4];
        p[1] = (char)tox[c&0x0f];
        if (!isprint(c) || /*isctrl(c)*/ c=='\r' || c=='\n' || c=='\t')
          c = '.';
      }
      p+=2;
      *p++ = ' ';
      *q++ = (char)c;
    }
    *q = '\0';
    LogRaw("\n%s\n",buffer);
  }
  LogRaw("%s total len: %d\n",label, alen);
  return "";
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

static int __fixup_dnethostname( char *host, int *portP, int mode, 
                                  int *nofallback )
{
  static char *dnet = ".distributed.net";
  int faultport = 0, autofind = 0;

  char *p = host;
  int len = 0;

  while (*p && isspace(*p))
    p++;
  while (*p && !isspace(*p))
    host[len++] = (char)tolower(*p++);
  host[len]=0;

  if (len==0 || strcmp(host,"auto")==0 || strcmp(host,"(auto)")==0 ||
                strcmp( host, dnet ) == 0 || strcmp( host, dnet+1 ) == 0 )
    faultport = autofind = 1;
  else if ( len < 16 )
    ; //not distributed.net, do nothing
  else if ( strcmp( &host[len-16], dnet ) != 0 )
    ; //not distributed.net, do nothing
  else if ( len == 21 && memcmp( host, "n0cgi.", 6 ) == 0 )
    *nofallback |= (mode & MODE_HTTP); //for (win32gui) newer-client check
  else if ( len > 22 && memcmp( &host[len-22], ".proxy.", 7 ) == 0 )
    *nofallback |= (mode & MODE_HTTP); // direct connect or redir service
  else if ( len < 20 || memcmp( &host[len-20], ".v27.", 5 ) != 0 )
    faultport = autofind = 1; // rc5proxy.distributed.net et al
  else
  {
    int i; const char *dzones[]={"us","euro","asia","aussie","jp"};
    autofind = 1; /* assume this until we know better */
    for (i=0;(autofind && i<((int)(sizeof(dzones)/sizeof(dzones[0]))));i++)
    {
      int len2 = strlen(dzones[i]);
      if ( memcmp( dzones[i], host, len2 )==0)
      {
        int foundport = ((host[len2]=='.')?(2064):(atoi(&host[len2])));
        if (foundport != 80 && foundport != 23 && foundport != 2064)
          break;
        else if (*portP == 0) //note: the hostname determines port
          *portP = foundport; //not viceversa.
        else if (*portP != 3064 && (*portP != foundport))
          break;
        autofind = 0; /* everything checks out */
      }
    }
  }
  if (autofind)
  {
    host[0]=0;
    if (faultport && *portP == 0 && (mode & MODE_HTTP) != 0 )
      *portP = 80;
  }
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

  reconnected = 0;
  shown_connection = 0;
  nofallback = _nofallback;
  sock = INVALID_SOCKET;
  iotimeout = _iotimeout; /* if iotimeout is <0, use blocking calls */
  isnonblocking = 0;      /* used later */

  gotuubegin = gothttpend = 0;
  httplength = 0;

  fwall_hostaddr = svc_hostaddr = conn_hostaddr = 0;
  fwall_hostname[0] = fwall_userpass[0] = 0;
  resolve_hostname[0] = '\0';
  resolve_addrcount = -1; /* uninitialized */

  #ifndef HTTPUUE_WORKS /* a lie */
  if (_enctype == 3 ) /*http+uue*/
  _enctype = 2; /* http only */
  #endif

  mode = startmode = 0;
  if (_enctype == 1 /*uue*/ || _enctype == 3 /*http+uue*/)
  {
    iotimeout = -1;
    startmode |= MODE_UUE;
  }
  if (_enctype == 2 /*http*/ || _enctype == 3 /*http+uue*/)
  {
    iotimeout = -1;
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

  autofindkeyserver = __fixup_dnethostname(server_name, &server_port,
                                           startmode, &nofallback);

#if (CLIENT_OS == OS_RISCOS)
  iotimeout = -1; // always blocking please
#endif    
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
int Network::Reset(u32 thataddress)
{
  reconnected = 1;
  svc_hostaddr = thataddress;
//LogScreen("netreset: %s\n",__inet_ntoa__(svc_hostaddr));
  return Open();
}

/* ---------------------------------------------------------------------- */

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

  return 0;
}

/* ----------------------------------------------------------------------- */

void Network::ShowConnection(void)
{
  if (verbose_level > 0 && !shown_connection)
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
      LogScreen("Connected to %s:%u...\n", targethost,
               ((unsigned int)(svc_hostport)) );
    }
    else
    {
      LogScreen( "Connected to %s:%u\nvia %s proxy %s:%u\n",
                 ( autofindkeyserver ? "distributed.net" : server_name ),
                 ((unsigned int)(svc_hostport)),
                 ((startmode & MODE_SOCKS5)?("SOCKS5"):
                 ((startmode & MODE_SOCKS4)?("SOCKS4"):("HTTP"))),
            fwall_hostname, (unsigned int)fwall_hostport );
    }
  }
  shown_connection = 1;
  return;
}

/* ----------------------------------------------------------------------- */

int Network::Open( void )               // returns -1 on error, 0 on success
{
  int didfallback, whichtry, maxtries;

  gethttpdone = puthttpdone = 0;
  netbuffer.Clear();
  uubuffer.Clear();
  gotuubegin = gothttpend = 0;
  httplength = 0;
  mode = startmode;

  didfallback = 0; /* have we fallen back already? */
  maxtries = 3; /* two for preferred, one for fallback (if permitted) */

  if ((startmode & (MODE_SOCKS4 | MODE_SOCKS5))!=0)
  {
    if (fwall_hostname[0] == 0 || fwall_hostport == 0)
    {
      Log("Network::Invalid %s proxy hostname or port.\n"
          "Connect cancelled.\n",
          ((startmode & MODE_HTTP) ? ("HTTP") : ("SOCKS")));
      return -1;
    }
  }
  
  for (whichtry = 0; whichtry < maxtries; whichtry++) /* forever true */
  {
    int success = 0;
    const char *netcheckfailed =
        "Network::Open Error - TCP/IP Connection Lost.\n";

    if (whichtry > 0) /* failed before */
    {
      if (CheckExitRequestTriggerNoIO())
        break;
      LogScreen( "Network::Open Error - sleeping for 3 seconds\n" );
      sleep(1);
      if (CheckExitRequestTriggerNoIO())
        break;
      sleep(1);
      if (CheckExitRequestTriggerNoIO())
        break;
      sleep(1);
      if (CheckExitRequestTriggerNoIO())
        break;
    }

    if (!NetCheckIsOK()) /* connection broken */
    {
      LogScreen(netcheckfailed);
      break; /* return -1; */
    }

    /* ---------- get the show on the road ---------- */
    
    success = 1; /* assumed */

    /* --- resolve the addresses(es) --- */

    if (success)
    {
      if (!reconnected || svc_hostaddr == 0)
      {
        if (!didfallback && !nofallback && whichtry == (maxtries-1) &&
           ((startmode & MODE_PROXIED) == 0 || fwall_hostname[0] == 0))
        {                                              /* last chance */
          if (autofindkeyserver || server_name[0]=='\0' ||
              strstr(server_name,".distributed.net"))
            ; /* can't fallback further than a fullserver */
          else
          {
            resolve_addrcount = -1;
            didfallback = 1;
            server_name[0] = '\0';
            maxtries = 1;
            whichtry = 0;
          }
        }
        svc_hostname = server_name;
        svc_hostport = server_port;

        if (svc_hostname[0] == 0)
        {
          svc_hostaddr = 0;
          svc_hostname = "rc5proxy.distributed.net"; /* special name for resolve */
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
          svc_hostport = DEFAULT_PORT;
          if ((startmode & MODE_PROXIED) != 0 && fwall_hostname[0] != 0)
          {
            if ((startmode & MODE_HTTP) != 0)
              svc_hostport = 80;
            else if ((startmode & MODE_UUE) != 0)
              svc_hostport = 23;
          }
        }
      }


      if (!NetCheckIsOK())
      {
        success = 0;
        maxtries = 0;
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
        if (NetResolve( fwall_hostname, fwall_hostport, 0,
                        &fwall_hostaddr, 1, ((char *)0), 0 ) < 0)
        {
          success = 0;
          fwall_hostaddr = 0;
          if (verbose_level > 0)
            LogScreen("Network::failed to resolve name \"%s\"\n",
                       fwall_hostname );

          // unrecoverable error. retry won't help
          maxtries = 0;
        }
        else if (svc_hostaddr == 0) /* always true unless http */
        {
          if (resolve_addrcount < 1)
          {
            resolve_addrcount =
               NetResolve( svc_hostname, svc_hostport, autofindkeyserver,
               &resolve_addrlist[0], 
               sizeof(resolve_addrlist)/sizeof(resolve_addrlist[0]),
               resolve_hostname, sizeof(resolve_hostname) );
            if (resolve_addrcount > maxtries)
              maxtries = resolve_addrcount;
          }
          if (resolve_addrcount > 0) /* found */
          {
            svc_hostaddr = resolve_addrlist[whichtry % resolve_addrcount];
          }
          else /* not found */
          {
            svc_hostaddr = 0;
            if ((startmode & (MODE_SOCKS4 /*| MODE_SOCKS5*/)) != 0)
            {
              success = 0; // socks needs the address to resolve now.
              if (verbose_level > 0)
                LogScreen("Network::failed to resolve hostname \"%s\"\n",
                           svc_hostname);
            }
          }
          svc_hostname = resolve_hostname; //socks5 will use this
        }
        conn_hostaddr = fwall_hostaddr;
        conn_hostname = fwall_hostname;
        conn_hostport = fwall_hostport;
      }
      else /* not proxied */
      {
        if (resolve_addrcount < 1)
        {
          resolve_addrcount =
             NetResolve( svc_hostname, svc_hostport, autofindkeyserver,
             &resolve_addrlist[0], 
             sizeof(resolve_addrlist)/sizeof(resolve_addrlist[0]),
             resolve_hostname, sizeof(resolve_hostname) );
          if (resolve_addrcount > maxtries) /* found! */
            maxtries = resolve_addrcount;
//LogScreen("resolved: \"%s\" => %d addresses\n", resolve_hostname, resolve_addrcount);
        }
        if (resolve_addrcount > 0) /* found */
        {
          if (!reconnected || !svc_hostaddr)
          {
            svc_hostaddr = resolve_addrlist[whichtry % resolve_addrcount];
//LogScreen("Try %d: picked %s\n", whichtry+1, __inet_ntoa__(svc_hostaddr));
          }
        }
        else
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

    /* ---------- create a new socket --------------- */

    if (success)
    {
      if (LowLevelCreateSocket() != 0)
      {
        success = 0;
        if (verbose_level > 0)
        {
          #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
          LogScreen("Network::failed to create network socket. (err=%d)\n",
                     WSAGetLastError());
          #else
          LogScreen("Network::failed to create network socket.\n");
          #endif
        }
        /* no point retrying is we can't create a socket */
        break; /* return -1; */
      }
    }

    /* ------ connect ------- */

    isnonblocking = 0;
    if (success)
    {
      #ifndef ENSURE_CONNECT_WITH_BLOCKING_SOCKET
      if (iotimeout > 0)
      {
        isnonblocking = ( LowLevelSetSocketOption( CONDSOCK_BLOCKMODE, 0 ) == 0 );
        #ifdef DEBUGTHIS
        Log("Debug::Connecting with %sblocking socket.\n",
              ((isnonblocking) ? ("non-") : ("")) );
        #endif
      }
      #endif

      success = ( LowLevelConnectSocket( conn_hostaddr, conn_hostport ) == 0 );

      #ifdef ENSURE_CONNECT_WITH_BLOCKING_SOCKET
      if (success && iotimeout > 0)
      {
        isnonblocking = ( LowLevelSetSocketOption( CONDSOCK_BLOCKMODE, 0 ) == 0 );
        #ifdef DEBUGTHIS
        Log("Debug::Connected (%sblocking).\n",
            ((isnonblocking) ? ("non-") : ("")) );
        #endif
      }
      #endif

      if (CheckExitRequestTriggerNoIO())
      {
        success = 0;
        maxtries = 0;
      }
      else if (!success)   /* connect failed */
      {
        if (verbose_level > 0)
        {
          char errreasonbuf[100];
          const char *msg = (const char *)0;
          int err = 0;
          #if (defined(ERRNO_IS_UNUSABLE_FOR_CONN_ERRMSG))
          /* nothing */
          #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32)
          err = errno; /* connect copies it from WSAGetLastError(); */
          msg = "unrecognized error";
          if      (err == WSAEADDRINUSE)
            msg = "The specified address is already in use";
          else if (err == WSAEADDRNOTAVAIL)
            msg = "The specified address is not available from the local machine";
          else if (err == WSAECONNREFUSED)
            msg = "Connect was rejected. Destination may be busy.";
          else if (err == WSAEISCONN)
            msg = "The socket is already connected"; //udp only?
          else if (err == WSAENETUNREACH)
            msg = "The network cannot be reached from this host at this time";
          else if (err == WSAENOBUFS)
            msg = "No buffer space is currently available";
          else if (err == WSAETIMEDOUT)
            msg = "Attempt to connect timed out";
          #elif defined(_TIUSER_)
          char tlookerr[100];
          err = t_errno;
          msg = "undefined error";
          if (err == TLOOK)
          {
            static int looknums[] = {
            T_LISTEN,T_CONNECT,T_DATA,T_EXDATA,T_DISCONNECT,T_ORDREL,
            T_UDERR,T_GODATA,T_GOEXDATA };
            static const char *looknames[] = {
            "t_listen","t_connect","t_data","t_exdata","t_disconnect",
            "t_ordrel","t_uderr","t_godata","t_goexdata"};
            int look = t_look(sock);
            msg = "";
            for (err=0;err<(int)(sizeof(looknums)/sizeof(looknums[0]));err++)
            {
              if (looknums[err] == look)
              {
                msg = looknames[err];
                break;
              }
            }
            err = TLOOK;
            sprintf(tlookerr, "asynchronous event %d %s%s%soccurred", 
              look, ((*msg)?("("):("")), msg, ((*msg)?(") "):("")) );
            msg = (const char *)&tlookerr[0];
          }
          else if (err == TSYSERR)
          {
            err = errno;
            msg = strerror(errno);
          }
          else if (err > 0 && err < t_nerr)
            msg = t_errlist[t_errno];
          #else
          err = errno;
          msg = strerror(errno);
          #endif
          errreasonbuf[0]='\0';
          if (msg)
          {
            int ml = sprintf(errreasonbuf,"Error %d (",err);
            strncpy(&errreasonbuf[ml], msg, (sizeof(errreasonbuf)-1)-ml);
            errreasonbuf[sizeof(errreasonbuf)-1] = '\0';
            strcat(errreasonbuf,")\n");
          }
          LogScreen( "Connect to host %s:%u failed.\n%s",
             __inet_ntoa__(conn_hostaddr), (unsigned int)(conn_hostport),
             errreasonbuf );
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
        maxtries = 0; /* so don't retry */
    }

    /* ---- clean up ---- */

    if (success)
    {
      reconnected = 1;
      return 0;
    }

    Close();
    svc_hostaddr = 0;

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

  if (startmode & MODE_HTTP)
  {
    LowLevelSetSocketOption( CONDSOCK_KEEPALIVE, 1 );
  }
  else if (startmode & MODE_SOCKS5)
  {
    int success = 0; //assume failed
    int recoverable = 0; //assume non-recoverable error (negotiation failure)

    char socksreq[600];  // room for large username/pw (255 max each)
    SOCKS5METHODREQ *psocks5mreq = (SOCKS5METHODREQ *)socksreq;
    SOCKS5METHODREPLY *psocks5mreply = (SOCKS5METHODREPLY *)socksreq;
    SOCKS5USERPWREPLY *psocks5userpwreply = (SOCKS5USERPWREPLY *)socksreq;
    SOCKS5 *psocks5 = (SOCKS5 *)socksreq;
    u32 len;

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
      psocks5->port = (u16)htons((u16)svc_hostport); //(u16)(htons((server_name[0]!=0)?((u16)port):((u16)(DEFAULT_PORT))));
      int packetsize = 10;

      if (svc_hostaddr == 0)
      {
        psocks5->atyp = 0x03; //fully qualified domainname
        char *p = (char *)(&psocks5->addr);
        // at this point svc_hostname is a ptr to a resolve_hostname.
        strcpy( p+1, svc_hostname );
        *p = (char)(len = strlen( p+1 ));
        p += (++len);
        *((u16 *)(p)) = (u16)htons((u16)svc_hostport);
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
        LogScreen("SOCKS5: server error 0x%02x%s%s%s\nconnecting to %s:%u\n",
                 ((int)(psocks5->cmdORrep)),
                 ((*p) ? (" (") : ("")), p, ((*p) ? (")") : ("")),
                 svc_hostname, (unsigned int)svc_hostport );
      }
    } //if (authaccepted)

    return ((success) ? (0) : ((recoverable) ? (+1) : (-1)));
  } //if (startmode & MODE_SOCKS5)

  if (startmode & MODE_SOCKS4)
  {
    int success = 0; //assume failed
    int recoverable = 0; //assume non-recoverable error (negotiation failure)

    char socksreq[128];  // min sizeof(fwall_userpass) + sizeof(SOCKS4)
    SOCKS4 *psocks4 = (SOCKS4 *)socksreq;
    u32 len;

    // transact a request to the SOCKS4 proxy giving the
    // destination ip/port and username and process its reply.

    psocks4->VN = 4;
    psocks4->CD = 1;  // CONNECT
    psocks4->DSTPORT = (u16)htons((u16)svc_hostport); //(u16)htons((server_name[0]!=0)?((u16)port):((u16)DEFAULT_PORT));
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
  int need_close = 0, timed_out = 0; //timed_out is only used with blocking sox

  int tmp_isnonblocking = (isnonblocking != 0); //we handle timeout ourselves
  isnonblocking = 0;                 //so stop LowLevelGet() from doing it.

  while (netbuffer.GetLength() < (u32)length && !CheckExitRequestTrigger())
  {
    int nothing_done = 1;

    if (starttime == 0) /* first pass through */
      time(&starttime);
    else if (!tmp_isnonblocking) /* we are blocking, so no more chances */
      ; //keep going till socket close or timeout or we have data
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
      else if (numRead < 0 && !tmp_isnonblocking) timed_out = 1;

      AutoBuffer line;
      while (uubuffer.RemoveLine(&line))
      {
        nothing_done = 0;
        if (strncmpi(line, "Content-Length: ", 16) == 0)
        {
          httplength = atoi((const char*)line + 16);
        }
        else if ( (svc_hostaddr == 0) &&
          (strncmpi(line, "X-KeyServer: ", 13) == 0))
        {
          u32 newaddr = 0;
          char newhostname[64];
          if (NetResolve( line + 13, svc_hostport, 0, &newaddr, 1, 
                          newhostname, sizeof(newhostname) ) > 0)
          {
            resolve_addrlist[0] = svc_hostaddr = newaddr;
            resolve_addrcount = 1;
            strncpy( resolve_hostname, newhostname, sizeof(resolve_hostname));
            resolve_hostname[sizeof(resolve_hostname)-1]='\0';
            #ifdef DEBUGTHIS
            Log("Debug:X-Keyserver: %s\n", __inet_ntoa__(svc_hostaddr));
            #endif
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
      else if (numRead < 0 && !tmp_isnonblocking) timed_out = 1;

      AutoBuffer line;
      while (uubuffer.RemoveLine(&line))
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
      if (numRead < 0 && !tmp_isnonblocking)
        timed_out = 1; // timed out
      else if (numRead == 0)
        need_close = 1;
      else if (numRead > 0)
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
    }

    if (nothing_done)
    {
      if (need_close || gethttpdone || (timed_out && tmp_isnonblocking))
        break;
      #if (CLIENT_OS == OS_VMS) || (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_ULTRIX)
        sleep(1); // full 1 second due to so many reported network problems.
      #else
        usleep( 100000 );  // Prevent racing on error (1/10 second)
      #endif
      if (CheckExitRequestTriggerNoIO())
        break;
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
    #ifdef DEBUGTHIS
    Log(__print_packet("Get", data, bytesfilled ));
    #endif
  }

  #ifdef DEBUGTHIS
  Log("Get: toread:%d read:%d\n", length, bytesfilled );
  #endif

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
    if (svc_hostaddr == 0)
      return -1;
    if (Reset(svc_hostaddr) != 0) //Open(sock)
      return -1;
  }

  #if (CLIENT_OS != OS_390) /* can't do UUE with EBCDIC */ 
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
  #endif
  {
    outbuf = AutoBuffer(data, length);
  }

  if (mode & MODE_HTTP)
  {
    AutoBuffer header;
    char userpass[(((sizeof(fwall_userpass)+1)*4)/3)];

    userpass[0] = '\0';
    if (fwall_userpass[0])
    {
      if ( base64_encode( userpass, fwall_userpass, 
                  sizeof(userpass), strlen(fwall_userpass)) < 0 )
        userpass[0] = '\0';
      userpass[sizeof(userpass)-1]='\0';
    }
    
    header.Reserve(1024);
    header.MarkUsed(
            sprintf(header.GetHead(),
            "POST http://%s:%u/cgi-bin/rc5.cgi HTTP/1.0\r\n"
            "Proxy-Connection: Keep-Alive\r\n"
            "%s%s%s"
            "Content-Type: application/octet-stream\r\n"
            "Content-Length: %lu\r\n\r\n",
            ((!nofallback && svc_hostaddr)?(__inet_ntoa__(svc_hostaddr)):
                                           (svc_hostname)),
            ((unsigned int)(svc_hostport)),
            ((userpass[0])?("Proxy-authorization: Basic "):("")),
            ((userpass[0])?(userpass):("")),
            ((userpass[0])?("\r\n"):("")),
            (unsigned long) outbuf.GetLength()));
    #if (CLIENT_OS == OS_OS390)
      __etoa(header.GetHead());
    #endif
    //outbuf = header + outbuf;
    header += outbuf;
    outbuf = header;
             
    puthttpdone = 1;
  }

  #ifdef DEBUGTHIS
  Log(__print_packet("Put", outbuf, outbuf.GetLength() ));
  #endif

  int towrite = (int)outbuf.GetLength();
  int written = LowLevelPut(outbuf,towrite);

  #ifdef DEBUGTHIS
  Log("Put: towrite:%d written:%d success:%d\n", towrite, written, (towrite==written) );
  #endif

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
    LowLevelSetSocketOption( CONDSOCK_SETMINBUFSIZE, 2048/* at least this */);
    LowLevelSetSocketOption( CONDSOCK_BLOCKMODE, 1 ); /* really only needed for RISCOS */
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
      t_rcvrel( sock );   /* wait for conn release by peer */
      t_sndrel( sock );   /* initiate close */
      t_unbind( sock );   /* close our own socket */
    }
    int rc = t_close( sock );
    sock = INVALID_SOCKET;
    return rc;
  }
#else                                                  //BSD socks
   if ( sock != INVALID_SOCKET )
   {
     LowLevelSetSocketOption( CONDSOCK_BLOCKMODE, 1 );
     #if (defined(AF_INET) && defined(SOCK_STREAM))
     shutdown( sock, 2 );
     #endif
     #if (CLIENT_OS == OS_OS2)
     int retcode = (int)soclose( sock );
     #elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
     int retcode = (int)closesocket( sock );
     #elif (CLIENT_OS == OS_AMIGAOS)
     int retcode = (int)CloseSocket( sock );
     #elif (CLIENT_OS == OS_BEOS)
     int retcode = (int)closesocket( sock );
     #elif (CLIENT_OS == OS_VMS) && defined(MULTINET)
     int retcode = (int)socket_close( sock );
     #else
     int retcode = (int)close( sock );
     #endif
     sock = INVALID_SOCKET;
     return (retcode);
   }
#endif
   return 0;
}

// -----------------------------------------------------------------------

#ifdef _TIUSER_
#ifndef DEBUGTHIS
#define debugtli(_x,_y) /* nothing */
#else
void debugtli(const char *label, int fd)
{
  int err = t_errno;
  if (err == TLOOK)
  {
    static int looknums[] = {
    T_LISTEN,T_CONNECT,T_DATA,T_EXDATA,T_DISCONNECT,T_ORDREL,
    T_ERROR,T_UDERR,T_GODATA,T_GOEXDATA,T_EVENTS };
    static const char *looknames[] = {
    "T_LISTEN","T_CONNECT","T_DATA","T_EXDATA","T_DISCONNECT","T_ORDREL",
    "T_ERROR","T_UDERR","T_GODATA","T_GOEXDATA","T_EVENTS"};
    const char *msg = "";
    int i, look = t_look(fd);
    for (i=0;i<(int)(sizeof(looknums)/sizeof(looknums[0]));i++)
    {
      if (looknums[i] == look)
      {
        msg = looknames[i];
        break;
      }
    }
    Log("%s: TLOOK: \"%s\" (%d)\n", label, msg, look );
  }
  else if (err == TOUTSTATE)
  {
    static int nums[] = {T_UNBND,T_IDLE,T_OUTCON,T_INCON,T_DATAXFER,T_OUTREL};
    static const char *names[] = {"T_UNBND","T_IDLE","T_OUTCON","T_INCON","T_DATAXFER","T_OUTREL"};
    const char *msg = "unknown";
    int i, state = t_getstate(fd);
    t_errno = err;
    if (state != -1)
    {
      for (i=0;i<(int)(sizeof(nums)/sizeof(nums[0]));i++)
      {
        if (nums[i] == state)
        {
          msg = names[i];
          break;
        }
      }
    }
    Log("%s: TOUTSTATE: \"primitive issued in wrong sequence\" (%d)."
           "currstate: %s (%d)\n", label, err, msg, state);
  }
  else if (err == TSYSERR)
    Log("%s: TSYSERR: \"%s\" (%d)\n", label, strerror(errno), errno);
  else if (err == 0)
    ;
  else if (err > 0 && err < t_nerr)
    Log("%s: t_error: \"%s\" (%d)\n", label, t_errlist[err], err);
  else 
    Log("%s: unknown error %d\n", label, err );
  return;
}  
#endif
#endif    

// -----------------------------------------------------------------------

int Network::LowLevelConnectSocket( u32 that_address, int that_port )
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
    time_t stoptime = time(NULL) + 2;
    #if 0
    while (sndcall != ((struct t_call *)0) && t_getstate(sock) == T_UNBND)  
    {
      if (time(NULL) > stoptime)
      {
        t_free((char *)sndcall, T_CALL);
        sndcall = (struct t_call *)0;
      }
    }  
    #endif
    if ( sndcall != ((struct t_call *)0) )
    {
      struct sockaddr_in *sin = (struct sockaddr_in *) sndcall->addr.buf;
      sndcall->addr.len  = sizeof(struct sockaddr_in);
      sndcall->opt.len   = 0;
      sndcall->udata.len = 0;
      sin->sin_addr.s_addr = that_address;
      sin->sin_family = AF_INET;
      sin->sin_port = htons(((u16)that_port));
      
      rc = t_connect( sock, sndcall, NULL);
      t_free((char *)sndcall, T_CALL);
      if (rc < 0)
      {
        int err = t_errno;
        debugtli("t_connect1", sock);
        if (isnonblocking && err == TNODATA)
        {
          stoptime = time(NULL) + (time_t)iotimeout;
          while (rc < 0 && err == TNODATA && time(NULL) <= stoptime)
          {
            if (CheckExitRequestTriggerNoIO())
              break;
            usleep(250000);
            rc = t_rcvconnect(sock, NULL);
            if (rc < 0) 
            {
              err = t_errno;
              debugtli("t_connect2", sock);
              if (err == TLOOK)
              {
                if (t_look(sock) == T_CONNECT)
                  rc = 0;
              }
            }
          }
        }
        if (rc < 0 && err == TLOOK)
        {
          int look = t_look(sock);
          if (look == T_CONNECT)
            rc = 0;
          else if (look == T_DISCONNECT)
          {
            struct t_discon tdiscon;
            tdiscon.udata.buf = (char *)0;
            tdiscon.udata.maxlen = 0;
            tdiscon.udata.len = 0;
            if (t_rcvdis(sock, &tdiscon) < 0)
              tdiscon.reason = ECONNREFUSED;
            t_errno = TSYSERR;
            errno = tdiscon.reason;
          }
        }
      }
      #if 0
      if (rc == 0)
      {
        int state = t_getstate(sock);
        if ( state != T_DATAXFER && state != T_OUTCON )
          rc = -1;
      }
      #endif
    }
  }
  return rc;
#elif defined(AF_INET) //BSD sox

  // set up the address structure
  struct sockaddr_in sin;
  memset((void *) &sin, 0, sizeof(sin));
  sin.sin_family = AF_INET;
  sin.sin_port = htons(((u16)that_port));
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
    if (CheckExitRequestTriggerNoIO())
    {
      rc = -1;
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
    if ( CheckExitRequestTriggerNoIO() || ((time(NULL)-starttime)>iotimeout) )
    {
      rc = -1;
      #ifndef ERRNO_IS_UNUSABLE_FOR_CONN_ERRMSG
      errno = ETIMEDOUT;
      #endif
      break;
    }
    sleep(1);
    rc = -1;
    if (CheckExitRequestTriggerNoIO())
      break;
  } while (isnonblocking); /* always 1 */

  return rc;
#else //no socket support
  return -1;
#endif
}

//------------------------------------------------------------------------

// Returns length of sent data or 0 if the socket is closed, or -1 if timeout+nodata
int Network::LowLevelPut(const char *ccdata,int length)
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
  int firsttime = 1;
  time_t timenow = 0, stoptime = 0;
  int sleptcount = 0; /* ... in a row */
  int sleepms = 250; /* sleep time in millisecs. adjust here if needed */
  char *data;
  *((const char **)&data) = ccdata; /* get around const being used for send */

  if (isnonblocking)
    stoptime = (time(NULL))+(time_t)iotimeout;

  #if defined(_TIUSER_)
  sendquota = 512;
  struct t_info info;
  if ( t_getinfo( sock, &info ) != -1)
  {
    if (info.tsdu > 0)
      sendquota = info.tsdu;
    else if (info.tsdu == -1) /* no limit */
      sendquota = length;
    else if (info.tsdu == 0) /* no boundaries */
      sendquota = 1500;
    else //if (info.tsdu == -2) /* normal send not supp'd (ever happens?)*/
      return -1;
  }
  #elif (CLIENT_OS == OS_WIN16 || CLIENT_OS == OS_WIN32S)
  if (sendquota > 0x7FFF)  /* 16 bit OS but int is 32 bits */
    sendquota = 0x7FFF;
  #else
  if (sendquota > INT_MAX)
    sendquota = INT_MAX;
  #endif

  #ifdef DEBUGTHIS
  Log("LLPut: total to send=%d, quota:%d\n", length, sendquota );
  #endif

  do
  {
    int towrite = (int)((((u32)length)>((u32)sendquota))?(sendquota):(length));
    int written;

    #if defined(_TIUSER_)                              //TLI/XTI
    int noiocount = 0;
    written = -2;
    while (written == -2)
    {
      //int flag = (((length - towrite)==0) ? (0) : (T_MORE));
      written = t_snd(sock, (char *)data, (unsigned int)towrite, 0 /* flag */ );
      if (written == 0)       /* transport provider accepted nothing */
      {                   /* should never happen unless 'towrite' was 0*/
        if ((++noiocount) < 3)
        {
          written = -2;   /* retry */
          usleep(500000); // 0.5 secs
        }
      }
      else if (written < 0)
      {
        written = -1;
        debugtli("t_snd", sock);
        if ( t_errno == TFLOW ) /* sending too fast */
        {
          usleep(500000); // 0.5 secs
          written = -2;
        }
        else if (t_errno == TLOOK)
        {
          int look = t_look(sock);
          if ( look == T_ORDREL)
          {
             //t_sndrel( sock );
             //t_rcvrel( sock );
             written = 0;
          }
          if (look == T_DISCONNECT || look == T_UDERR|| look == T_ERROR)
            return 0;
        }
      }
    }
    #elif defined(AF_INET) && defined(SOCK_STREAM)      //BSD 4.3 sockets
    if (firsttime)
    {
      int ready;
      fd_set rs;
      timeval tv = {0,0};
      FD_ZERO(&rs);
      FD_SET(sock, &rs);
      ready = select(sock + 1, &rs, NULL, NULL, &tv);
      if (ready < 0)   /* ENETDOWN, EINVAL, EINTR */
        return 0; /* assume sock closed */
      else if (ready == 1)
      {
        /*
        For connection-oriented sockets, readability can also indicate
        that a close request has been received from the peer. If the virtual
        circuit was closed gracefully, then a recv will return immediately
        with zero bytes read. If the virtual circuit was reset, then a
        recv will complete immediately with an error code.

        Platforms that support OSI/XTI/TLI (MacOS, all SysV, etc) should
        use that instead.
        */
        char scratch[2];
        if ( recv(sock, &scratch[0], sizeof(scratch), MSG_PEEK ) <= 0)
          return 0;
      }
      firsttime = 0;
    }
    written = send(sock, (char*)data, towrite, 0 );

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
      firsttime = 0;
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
  } while (length && !CheckExitRequestTriggerNoIO());

  #ifdef DEBUGTHIS
  Log("LLPut: towrite=%d, written=%d\n", totaltowrite, totalwritten );
  #endif
  totaltowrite = totaltowrite; //squash compiler warning
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
  u32 rcvquota = 1500;
  time_t timenow = 0, stoptime = 0;
  int sleptcount = 0; /* ... in a row */
  int sleepms = 250; /* sleep time in millisecs. adjust here if needed */
  int sockclosed = 0;

  #if defined(_TIUSER_)
  rcvquota = 512;
  struct t_info info;
  if ( t_getinfo( sock, &info ) < 0)
    info.tsdu = 0; /* assume tdsu not suppported */
  else if (info.tsdu > 0)
    rcvquota = info.tsdu;
  else if (info.tsdu == -1) /* no limit */
    rcvquota = length;
  else if (info.tsdu == 0) /* no boundaries */
    rcvquota = 1500;
  else //if (info.tsdu == -2) /* normal send not supp'd (ever happens?)*/
    return -1;
  #elif ((CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32))
  if (rcvquota > 0x7FFF)
    rcvquota = 0x7FFF;
  #else
  if (rcvquota > INT_MAX)
    rcvquota = INT_MAX;
  #endif

  if (isnonblocking)
    stoptime = (time(NULL))+(time_t)iotimeout;

  #ifdef DEBUGTHIS
  Log("LLGet: total to recv=%d, quota:%d\n", length, rcvquota );
  #endif

  do
  {
    int toread = (int)((((u32)length)>((u32)rcvquota))?(rcvquota):(length));
    int bytesread = 0;

    #if defined(_TIUSER_)                               //OSI/TLI/XTI
    int flags = 0; /* T_MORE, T_EXPEDITED etc */
    bytesread = t_rcv( sock, data, toread, &flags );
    if (bytesread == 0) /* peer sent a zero byte message */
      bytesread = -1; /* treat as none waiting */
    else if (bytesread < 0) 
    {
      int look, err = t_errno;
      bytesread = -1;
      debugtli("t_rcv", sock);
      if (err == TNODATA )
        bytesread = -1; /* fall through */
      else if (err != TLOOK) /* TSYSERR et al */
        bytesread = 0; /* set as socket closed */
      else if ((look = t_look(sock)) == T_ORDREL)
      {                /* connection closing... */
        t_rcvrel( sock ); 
        bytesread = 0; /* treat as closed */
      }
      else if (look == T_DISCONNECT || look == T_ERROR )
        bytesread = 0; /* treat as closed */
      else /* else T_DATA (Normal data received), and T_GODATA and family */
        bytesread = -1;
    }
    #elif (defined(AF_INET) && defined(SOCK_STREAM))        //BSD 4.3
    {
      fd_set rs;
      timeval tv = {0,0};
      FD_ZERO(&rs);
      FD_SET(sock, &rs);
      bytesread = select(sock + 1, &rs, NULL, NULL, &tv);
      if (bytesread < 0)   /* ENETDOWN, EINVAL, EINTR */
        bytesread = 0; /* == sock closed */
      else if (bytesread != 1) /* not ready */
        bytesread = -1;
      else /* socket says ready, but that could also mean closed :) */
      {
        bytesread = recv(sock, data, toread, 0 );
        if (bytesread <= 0)
          bytesread = 0;
      }
    }
    #endif /* TLI/XTI or BSD */

    #ifdef DEBUGTHIS
    Log("LLGet: read(%d)-> %d\n", toread, bytesread );
    #endif

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
      if (CheckExitRequestTrigger())
        break;
      ++sleptcount;
    }
    unsigned long sleepdur = ((unsigned long)(sleptcount+1)) * sleepms;
    if (sleepdur > 1000000UL)
      sleep( sleepdur / 1000000UL );
    if ((sleepdur % 1000000UL) != 0)
      usleep( sleepdur % 1000000UL );
  } while (length && !CheckExitRequestTriggerNoIO());

  #ifdef DEBUGTHIS
  Log("LLGet: got %u (requested %u) sockclosed:%s\n",
              totalread, totalread+length, ((sockclosed)?("yes"):("no")));
  #endif

  if (totalread!=0)
    return (int)totalread;
  if (sockclosed)
    return 0;
  return -1;
}

// ----------------------------------------------------------------------

int Network::LowLevelSetSocketOption( int cond_type, int parm )
{
  if ( sock == INVALID_SOCKET )
    return -1;

  if ( cond_type == CONDSOCK_KEEPALIVE )
  {
    #if defined(SOL_SOCKET) && defined(SO_KEEPALIVE)
    int on = ((parm == 0/* off */)?(0):(1));
    if (!setsockopt(sock,SOL_SOCKET,SO_KEEPALIVE,(char *)&on, sizeof(on)))
      return 0;
    #endif
    return -1;
  }
  else if ( cond_type == CONDSOCK_SETMINBUFSIZE )
  {
    #if (defined(SOL_SOCKET) && defined(SO_RCVBUF) && defined(SO_SNDBUF))
    int which;
    for (which = 0; which < 2; which++ )
    {
      #if (defined(__GLIBC__) && (__GLIBC__ >= 2)) || (CLIENT_OS == OS_MACOS)
      #define SOCKLEN_T socklen_t
      #elif ((CLIENT_OS == OS_BSDOS) && (_BSDI_VERSION > 199701))
      #define SOCKLEN_T size_t
      #else
      #define SOCKLEN_T int
      #endif
      int type = ((which == 0)?(SO_RCVBUF):(SO_SNDBUF));
      int sz = 0;
      SOCKLEN_T szint = (SOCKLEN_T)sizeof(int);
      if (getsockopt(sock, SOL_SOCKET, type, (char *)&sz, &szint)<0)
        ;
      else if (sz < parm)
      {
        sz = parm;
        setsockopt(sock, SOL_SOCKET, type, (char *)&sz, szint);
      }
    }
    return 0;
    #endif
  }
  else if ( cond_type == CONDSOCK_BLOCKMODE )
  {
    #if defined(_TIUSER_)                                    //TLI
      if ( parm != 0 ) /* blocking on */
        return ( t_blocking( sock ) );
      else
        return ( t_nonblocking( sock ) );
    #elif (!defined(FIONBIO) && !(defined(F_SETFL) && (defined(FNDELAY) || defined(O_NONBLOCK))))
      return -1;
    #elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
      unsigned long flagon = ((parm == 0/* off */)?(1):(0));
      return ioctlsocket(sock, FIONBIO, &flagon);
    #elif ((CLIENT_OS == OS_VMS) && defined(__VMS_UCX__))
      // nonblocking sockets not directly supported by UCX
      // - DIGITAL's work around requires system privileges to use
      return -1;
    #elif ((CLIENT_OS == OS_VMS) && defined(MULTINET))
      unsigned long flagon = ((parm == 0 /* off */)?(1):(0));
      return socket_ioctl(sock, FIONBIO, &flagon);
    #elif (CLIENT_OS == OS_RISCOS)
      int flagon = ((parm == 0 /* off */) ? (1): (0));
      if (ioctl(sock, FIONBIO, &flagon) && !flagon) // allow blocking socket calls
      { flagon = 1; ioctl(sock, FIOSLEEPTW, &flagon); } //to preemptively multitask
    #elif (CLIENT_OS == OS_OS2)
      int flagon = ((parm == 0 /* off */) ? (1): (0));
      return ioctl(sock, FIONBIO, (char *) &flagon, sizeof(flagon));
    #elif (CLIENT_OS == OS_AMIGAOS)
      char flagon = ((parm == 0 /* off */) ? (1): (0));
      return IoctlSocket(sock, FIONBIO, &flagon);
    #elif (CLIENT_OS == OS_DOS)
      return ((parm == 0 /* off */)?(0):(-1)); //always non-blocking
    #elif (defined(F_SETFL) && (defined(FNDELAY) || defined(O_NONBLOCK)))
    {
      int flag, res, arg;
      #if (defined(FNDELAY))
        flag = FNDELAY;
      #else
        flag = O_NONBLOCK;
      #endif
      arg = ((parm == 0 /* off */) ? (flag): (0) );

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
    #endif
  }
  parm = parm; /* shaddup compiler */
  return -1;
}

// ----------------------------------------------------------------------

