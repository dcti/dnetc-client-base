// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

#include "network.h"
#define VERBOSE_OPEN //print cause of ::Open() errors

#include <stddef.h> // for offsetof
#include <assert.h> // asserts present should be optimised out, so no need for NDEBUG
#include <errno.h>

#define UU_DEC(Ch) (char) (((Ch) - ' ') & 077)
#define UU_ENC(Ch) (char) (((Ch) & 077) != 0 ? ((Ch) & 077) + 0x20 : '`')

#if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_DOS)
  #include <varargs.h>
#elif (CLIENT_OS == OS_AMIGAOS)
  static struct Library *SocketBase;
#elif ((CLIENT_OS == OS_SUNOS) && (CLIENT_CPU==CPU_68K))
  extern "C" void usleep(unsigned int);
  #if defined(_SUNOS3_)
  extern "C" int fcntl(int, int, int);
  #endif
  extern "C" {
  int socket(int, int, int);
  int setsockopt(int, int, int, char *, int);
  int connect(int, struct sockaddr *, int);
  }
#elif (CLIENT_OS == OS_NETWARE)
  #ifdef gethostbyname     //this is in "hbyname.cpp"
  #undef gethostbyname
  #endif
  #ifdef inet_ntoa         //this is also in "hbyname.cpp"
  #undef inet_ntoa
  #endif
  #include "netware/hbyname.cpp"  // this is a domain name resolver
#endif

#pragma pack(1)               // no padding allowed

typedef struct _socks4 {
  unsigned char VN;           // version == 4
  unsigned char CD;           // command code, CONNECT == 1
  u16 DSTPORT;                // destination port, network order
  u32 DSTIP;                  // destination IP, network order
  char USERID[1];             // variable size, null terminated
} SOCKS4;

typedef struct _socks5methodreq {
  unsigned char ver;          // version == 5
  unsigned char nMethods;     // number of allowable methods following
  unsigned char Methods[2];   // this program will request at most two
} SOCKS5METHODREQ;

typedef struct _socks5methodreply {
  unsigned char ver;          // version == 1
  unsigned char Method;       // server chose method ...
  char end;
} SOCKS5METHODREPLY;

typedef struct _socks5userpwreply {
  unsigned char ver;          // version == 1
  unsigned char status;       // 0 == success
  char end;
} SOCKS5USERPWREPLY;

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

const char *Socks5ErrorText[9] =
{
 /* 0 */ NULL,                              // success
  "general SOCKS server failure",
  "connection not allowed by ruleset",
  "Network unreachable",
  "Host unreachable",
  "Connection refused",
  "TTL expired",
  "Command not supported",
  "Address type not supported"
};

//////////////////////////////////////////////////////////////////////////////

class NetTimer
{
  time_t start;
public:
  NetTimer(void) : start(time(NULL)) {};
  operator u32 (void) {return time(NULL) - start;}
};

//////////////////////////////////////////////////////////////////////////////

void NetworkInitialize(void)
{
#if !defined(NONETWORK)
  // perform platform specific socket initialization
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
    WSADATA wsaData;
    WSAStartup(0x0101, &wsaData);
  #elif defined(DJGPP) || (CLIENT_OS == OS_OS2)
    sock_init();
  #elif (CLIENT_OS == OS_AMIGAOS)
    #define SOCK_LIB_NAME "bsdsocket.library"
    SocketBase = OpenLibrary((unsigned char *)SOCK_LIB_NAME, 4UL);
    __assert((int)SocketBase, "Failed to open " SOCK_LIB_NAME "\n",
        __FILE__, __LINE__);
  #endif
  #ifdef SOCKS
    LIBPREFIX(init)("rc5-client");
  #endif

  // check that the packet structures have been correctly packed
  // do it here to make sure the asserts go off
  // if all is correct, the asserts should get totally optimised out :)
  assert(offsetof(SOCKS4, USERID) == 8);
  assert(offsetof(SOCKS5METHODREQ, Methods) == 2);
  assert(offsetof(SOCKS5METHODREPLY, end) == 2);
  assert(offsetof(SOCKS5USERPWREPLY, end) == 2);
  assert(offsetof(SOCKS5, end) == 10);
#endif
}

//////////////////////////////////////////////////////////////////////////////

void NetworkDeinitialize(void)
{
#if !defined(NONETWORK)
  // perform platform specific socket deinitialization
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
    WSACleanup();
  #elif (CLIENT_OS == OS_AMIGAOS)
  if (SocketBase) {
    CloseLibrary(SocketBase);
    SocketBase = NULL;
  }
  #endif
#endif
}

//////////////////////////////////////////////////////////////////////////////

Network::Network( const char * Preferred, const char * Roundrobin, s16 Port)
{
#if !defined(NONETWORK)
  // intialize communication parameters
  strncpy(server_name, (Preferred ? Preferred : ""), sizeof(server_name));
  strncpy(rrdns_name, (Roundrobin ? Roundrobin : ""), sizeof(rrdns_name));
  port = (s16) (Port ? Port : DEFAULT_PORT);
  mode = startmode = 0;
  retries = 0;
  sock = 0;
  gotuubegin = gothttpend = false;
  httplength = 0;
  lasthttpaddress = lastaddress = 0;
  httpid[0] = 0;

#if (CLIENT_OS == OS_OS2)
  ensureonline();
#endif

#endif
}

//////////////////////////////////////////////////////////////////////////////

Network::~Network(void)
{
#if !defined(NONETWORK)
  Close();

#if (CLIENT_OS == OS_OS2)
  checkoffline();
#endif

#endif
}

//////////////////////////////////////////////////////////////////////////////

void Network::SetModeUUE( bool enabled )
{
  if (enabled)
  {
    startmode &= ~(MODE_SOCKS4 | MODE_SOCKS5);
    startmode |= MODE_UUE;
  }
  else startmode &= ~MODE_UUE;
}

//////////////////////////////////////////////////////////////////////////////

void Network::SetModeHTTP( const char *httpproxyin, s16 httpportin,
    const char *httpidin)
{
  if (httpproxyin && httpproxyin[0])
  {
    startmode &= ~(MODE_SOCKS4 | MODE_SOCKS5);
    startmode |= MODE_HTTP;
    httpport = httpportin;
    strncpy(httpid, httpidin, 128);
    strncpy(httpproxy, httpproxyin, 64);
  }
  else startmode &= ~MODE_HTTP;
  lastaddress = lasthttpaddress = 0;
}

//////////////////////////////////////////////////////////////////////////////

void Network::SetModeSOCKS4(const char *sockshost, s16 socksport,
      const char * socksusername )
{
  if (sockshost && sockshost[0])
  {
    startmode &= ~(MODE_HTTP | MODE_SOCKS5 | MODE_UUE);
    startmode |= MODE_SOCKS4;
    httpport = socksport;
    strncpy(httpproxy, sockshost, sizeof(httpproxy));
    if (socksusername) {
      strncpy(httpid, socksusername, sizeof(httpid));
    } else {
      httpid[0] = 0;
    }
  }
  else
  {
    startmode &= ~MODE_SOCKS4;
    httpproxy[0] = 0;
  }
  lastaddress = lasthttpaddress = 0;
}

//////////////////////////////////////////////////////////////////////////////

void Network::SetModeSOCKS5(const char *sockshost, s16 socksport,
      const char * socksusernamepw )
{
  if (sockshost && sockshost[0])
  {
    startmode &= ~(MODE_HTTP | MODE_SOCKS4 | MODE_UUE);
    startmode |= MODE_SOCKS5;
    httpport = socksport;
    strncpy(httpproxy, sockshost, sizeof(httpproxy));
    if (socksusernamepw) {
      strncpy(httpid, socksusernamepw, sizeof(httpid));
    } else {
      httpid[0] = 0;
    }
  }
  else
  {
    startmode &= ~MODE_SOCKS5;
    httpproxy[0] = 0;
  }
  lastaddress = lasthttpaddress = 0;
}

//////////////////////////////////////////////////////////////////////////////

// returns -1 on error, 0 on success
s32 Network::Resolve(const char *host, u32 &hostaddress)
{
#if !defined(NONETWORK)

#if (CLIENT_OS == OS_AMIGAOS) && (CLIENT_CPU == CPU_POWERPC)
  if ((hostaddress = inet_addr((unsigned char*)host)) == 0xFFFFFFFFL)
#else
  if ((hostaddress = inet_addr((char*)host)) == 0xFFFFFFFFL)
#endif
  {
    struct hostent *hp;
#if (CLIENT_OS == OS_AMIGAOS) && (CLIENT_CPU == CPU_POWERPC)
    if ((hp = gethostbyname((unsigned char*)host)) == NULL) return -1;
#else
    if ((hp = gethostbyname((char*)host)) == NULL) return -1;
#endif

#if defined(_SUNOS3_)
    // gethostbyname() returns only one address.
    memcpy((void*) &hostaddress, hp->h_addr, sizeof(u32));
#else
    int addrcount;
  #if ((CLIENT_OS == OS_SUNOS) && (CLIENT_CPU == CPU_68K))
    // struct hostent is seriously broken when returned from gethostbyname()
    // on sun3/SunOS 4.1.1. The h_addr_list member does NOT point to an array
    // of pointers to IP addresses, it points to an array of the IP addresses
    // themselves, and *then* pointers to them! Work around this problem by
    // searching through the array for an entry that points to the start of
    // the array, and fixing up h_addr_list.
    for (addrcount = 0;
         hp->h_addr_list[addrcount] &&
                        (hp->h_addr_list[addrcount] != (char *)hp->h_addr_list);
         addrcount++) ;
    hp->h_addr_list += addrcount;
  #endif

    // randomly select one
    for (addrcount = 0; hp->h_addr_list[addrcount]; addrcount++);
    int index = rand() % addrcount;
    memcpy((void*) &hostaddress, hp->h_addr_list[index], sizeof(u32));
#endif
  }
#endif

  return 0;
}

//////////////////////////////////////////////////////////////////////////////

void Network::LogScreen ( const char * text)
{
#if (CLIENT_OS == OS_NETWARE)
  if (!quietmode) printf("%s", text );
#else
  if (!quietmode) fwrite(text, 1, strlen(text), stdout);
#endif
}

//////////////////////////////////////////////////////////////////////////////

// returns -1 on error, 0 on success
s32 Network::Open( SOCKET insock)
{
#if !defined(NONETWORK)
  // set communications settings
  mode = startmode;
  gotuubegin = gothttpend = puthttpdone = gethttpdone = false;
  httplength = 0;
  netbuffer.Clear();
  uubuffer.Clear();

  // make socket non-blocking
  sock = insock;
  MakeNonBlocking(sock, true);
#endif
  return 0;
}

//////////////////////////////////////////////////////////////////////////////

// returns -1 on error, 0 on success
s32 Network::Open( void )
{
#if !defined(NONETWORK)
  Close();
  mode = startmode;


  // create a new socket
  if ((int)(sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
  {
#ifdef VERBOSE_OPEN
    LogScreen("Network::failed to create network socket. \n" );
#endif
    return(-1);
  }

  // allow the socket to bind to "in use" ports
#if (CLIENT_OS != OS_NETWARE)    // don't need this. No bind() in sight
  int on = 1;
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char *)&on, sizeof(on));
#endif

#if (CLIENT_OS == OS_RISCOS)
  // allow blocking socket calls to preemptively multitask
  ioctl(sock, FIOSLEEPTW, &on);
#endif

  // resolve the address of keyserver, or if performing a proxied
  // connection, the address of the proxy gateway server.
  if (lastaddress == 0)
  {
    const char * hostname;
    if (mode & MODE_PROXIED) hostname = httpproxy;
    else
    {
      hostname = (retries < 4 ? server_name : rrdns_name);
      if (!hostname[0]) {
        hostname = DEFAULT_RRDNS;
        lastport = DEFAULT_PORT;
      } else lastport = port;
    }

#ifdef VERBOSE_OPEN
    sprintf(logstr,"Connecting to %s...\n",hostname);
    LogScreen(logstr);
#endif
    if (Resolve(hostname, lastaddress) < 0)
    {
      lastaddress = 0;
      close(sock);
      sock = 0;
      retries++;
#ifdef VERBOSE_OPEN
      sprintf(logstr,"Network::failed to resolve hostname \"%s\"\n", hostname);
      LogScreen(logstr);
#endif
      return (-1);
    }
  }


  // if we are doing a proxied connection, resolve the actual destination
  if ((mode & MODE_PROXIED) && (lasthttpaddress == 0))
  {
    const char * hostname;
    hostname = (retries < 4 ? server_name : rrdns_name);
    if (!hostname[0]) {
      hostname = DEFAULT_RRDNS;
      lastport = DEFAULT_PORT;
    } else lastport = port;

#ifdef VERBOSE_OPEN
    sprintf(logstr,"Connecting to %s...\n",hostname);
    LogScreen(logstr);
#endif
    if (Resolve(hostname, lasthttpaddress) < 0)
    {
      lasthttpaddress = 0;

      if (mode & (MODE_SOCKS4 | MODE_SOCKS5))
      {
        // socks needs the address to resolve now.
        lastaddress = 0;
        close(sock);
        sock = 0;
        retries++;
#ifdef VERBOSE_OPEN
        sprintf(logstr,"Network::failed to resolve hostname \"%s\"\n", hostname);
        LogScreen(logstr);
#endif
        return (-1);
      }
    }
  }


  // set up the address structure
  struct sockaddr_in sin;
  memset((void *) &sin, 0, sizeof(sin));
  sin.sin_family = AF_INET;
  sin.sin_port = htons((mode & MODE_PROXIED) ? httpport : lastport);
  sin.sin_addr.s_addr = lastaddress;


  // try connecting
  if ((int) connect(sock, (struct sockaddr *) &sin, sizeof(sin)) < 0)
  {
#if (CLIENT_OS == OS_NETWARE)
    char *s;
    int i=errno;          //later make i=0 if lastaddress needs clearing
    if (i==ECONNREFUSED)       {i=1;s="%d:ECONNREFUSED (connection refused)"; }
    else if (i==ETIMEDOUT)     {i=1;s="%d:ETIMEDOUT (connection timed out)";}
    else if (i==ETOOMANYREFS)  {i=0;s="%d:ETOOMANYREFS (can't splice)";}
    else if (i==ESHUTDOWN)     {i=0;s="%d:ESHUTDOWN (can't send after socket shutdown)";}
    else if (i==ENOTCONN)      {i=0;s="%d:ENOTCONN (\"socket not connected\") (of course!)";}
    else if (i==EISCONN)       {i=0;s="%d:EISCONN (already connected) (huh?)";}
    else if (i==ENOBUFS)       {i=0;s="%d:ENOBUFS (no buffer space available)";}
    else if (i==ECONNRESET)    {i=1;s="%d:ECONNRESET (connection reset by peer)";}
    else if (i==ECONNABORTED)  {i=1;s="%d:ECONNABORTED (connection aborted by request)";}
    else if (i==ENETRESET)     {i=0;s="%d:ENETRESET (network dropped connection on reset)";}
    else if (i==ENETUNREACH)   {i=0;s="%d:ENETUNREACH (network is unreachable)";}
    else if (i==ENETDOWN)      {i=0;s="%d:ENETDOWN (network is down)";}
    else                       {i=0;s="for unknown reasons. errno=%d";}

    if (i==0) lastaddress = 0;
#ifdef VERBOSE_OPEN
    printf("Network::connect() to proxy %s failed\n         ",
        inet_ntoa( *(struct in_addr *)(&sin.sin_addr.s_addr) ));
    printf(s, errno );
    printf("\n");
#endif

#else
    lastaddress = 0;
#ifdef VERBOSE_OPEN
    printf("Network::connect() to proxy %s failed\n         ",
#if (CLIENT_OS == OS_AMIGAOS)
        inet_ntoa( sin.sin_addr ));
#else
        inet_ntoa( *(struct in_addr *)(&sin.sin_addr.s_addr) ));
#endif
    printf("\n");
#endif
#endif
    close(sock);
    sock = 0;
    retries++;
    return(-1);
  }


  // set communications settings
  gotuubegin = gothttpend = puthttpdone = gethttpdone = false;
  httplength = 0;
  netbuffer.Clear();
  uubuffer.Clear();



  if (startmode & MODE_SOCKS4)
  {
    char socksreq[128];  // min sizeof(httpid) + sizeof(SOCKS4)
    SOCKS4 *psocks4 = (SOCKS4 *)socksreq;
    u32 len;

    sin.sin_addr.s_addr = lasthttpaddress;

    // transact a request to the SOCKS4 proxy giving the
    // destination ip/port and username and process its reply.

    psocks4->VN = 4;
    psocks4->CD = 1;  // CONNECT
    psocks4->DSTPORT = htons(lastport);
    psocks4->DSTIP = lasthttpaddress;
    strncpy(psocks4->USERID, httpid, sizeof(httpid));

    len = sizeof(*psocks4) - 1 + strlen(httpid) + 1;
    if (LowLevelPut(len, socksreq) < 0)
      goto Socks4Failure;

    len = sizeof(*psocks4) - 1;  // - 1 for the USERID[1]
    if ((u32)LowLevelGet(len, socksreq) != len)
      goto Socks4Failure;

    if (psocks4->VN != 0 ||
        psocks4->CD != 90) // 90 is successful return
    {
      fprintf(stderr, "SOCKS4 request rejected%s\n",
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
      fflush(stderr);

Socks4Failure:
      close(sock);
      sock = 0;
      retries++;
      return(-1);
    }
  }
  else if (startmode & MODE_SOCKS5)
  {
    char socksreq[600];  // room for large username/pw (255 max each)
    SOCKS5METHODREQ *psocks5mreq = (SOCKS5METHODREQ *)socksreq;
    SOCKS5METHODREPLY *psocks5mreply = (SOCKS5METHODREPLY *)socksreq;
    SOCKS5USERPWREPLY *psocks5userpwreply = (SOCKS5USERPWREPLY *)socksreq;
    SOCKS5 *psocks5 = (SOCKS5 *)socksreq;
    u32 len;

    sin.sin_addr.s_addr = lasthttpaddress;

    // transact a request to the SOCKS5 proxy requesting
    // authentication methods.  If the username/password
    // is provided we ask for no authentication or user/pw.
    // Otherwise we ask for no authentication only.

    psocks5mreq->ver = 5;
    psocks5mreq->nMethods = (unsigned char) (httpid[0] ? 2 : 1);
    psocks5mreq->Methods[0] = 0;  // no authentication
    psocks5mreq->Methods[1] = 2;  // username/password

    len = 2 + psocks5mreq->nMethods;
    if (LowLevelPut(len, socksreq) < 0)
      goto Socks5Failure;

    if ((u32)LowLevelGet(2, socksreq) != 2)
      goto Socks5Failure;

    if (psocks5mreply->ver != 5)
    {
      fprintf(stderr, "SOCKS5 authentication method reply has wrong "\
                      "version, %d should be 5.\n", psocks5mreply->ver);
      fflush(stderr);

      goto Socks5Failure;
    }

    if (psocks5mreply->Method == 0)       // no authentication
    {
      // nothing to do for no authentication method
    }
    else if (psocks5mreply->Method == 2)  // username and pw
    {
      char username[255];
      char password[255];
      char *pchSrc, *pchDest;
      int userlen, pwlen;

      pchSrc = httpid;
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
      memcpy(socksreq + len, username, userlen);
      len += userlen;
      socksreq[len++] = (char) pwlen;
      memcpy(socksreq + len, password, pwlen);
      len += pwlen;

      if (LowLevelPut(len, socksreq) < 0)
        goto Socks5Failure;

      if ((u32)LowLevelGet(2, socksreq) != 2)
        goto Socks5Failure;

      if (psocks5userpwreply->ver != 1 ||
          psocks5userpwreply->status != 0)
      {
        fprintf(stderr, "SOCKS5 user %s rejected by server.\n", username);
        fflush(stderr);

        goto Socks5Failure;
      }
    }
    else
    {
      fprintf(stderr, "SOCKS5 authentication method rejected.\n");
      fflush(stderr);

      goto Socks5Failure;
    }

    // after subnegotiation, send connect request
    psocks5->ver = 5;
    psocks5->cmdORrep = 1;   // connnect
    psocks5->rsv = 0;   // must be zero
    psocks5->atyp = 1;  // IPv4
    psocks5->addr = lasthttpaddress;
    psocks5->port = htons(lastport);

    if (LowLevelPut(10, socksreq) < 0)
      goto Socks5Failure;

    if ((u32)LowLevelGet(10, socksreq) != 10)
      goto Socks5Failure;

    if (psocks5->ver != 5)
    {
      fprintf(stderr, "SOCKS5 reply has wrong version, %d should be 5\n", psocks5->ver);
      fflush(stderr);
      goto Socks5Failure;
    }

    if (psocks5->cmdORrep != 0)          // 0 is successful connect
    {
      fprintf(stderr, "SOCKS5 server error connecting to keyproxy: %s\n",
              (psocks5->cmdORrep >=
                (sizeof Socks5ErrorText / sizeof Socks5ErrorText[0]))
                ? "unrecognized SOCKS5 error"
                : Socks5ErrorText[ psocks5->cmdORrep ] );

Socks5Failure:
      close(sock);
      sock = 0;
      retries++;
      return(-1);
    }
  }

  // change socket to non-blocking
  MakeNonBlocking(sock, true);


#endif
  return 0;
}

//////////////////////////////////////////////////////////////////////////////

s32 Network::Close(void)
{
#if !defined(NONETWORK)
  if ( sock )
  {
    close( sock );
    sock = 0;
    retries = 0;
    gethttpdone = puthttpdone = false;
    netbuffer.Clear();
    uubuffer.Clear();
  }
#endif
  return 0;
}

//////////////////////////////////////////////////////////////////////////////

// Returns length of read buffer.
s32 Network::Get( u32 length, char * data, u32 timeout )
{
#if !defined(NONETWORK)
  NetTimer timer;
  bool need_close = false;

  while (netbuffer.GetLength() < length && timer <= timeout)
  {
    bool nothing_done = true;

    if ((mode & MODE_HTTP) && !gothttpend)
    {
      // []---------------------------------[]
      // |  Process HTTP headers on packets  |
      // []---------------------------------[]
      uubuffer.Reserve(500);
      s32 numRead = LowLevelGet(uubuffer.GetSlack(), uubuffer.GetTail());
      if (numRead > 0) uubuffer.MarkUsed(numRead);
      else if (numRead == 0) need_close = true;       // connection closed

      AutoBuffer line;
      while (uubuffer.RemoveLine(line))
      {
        nothing_done = false;
        if (strncmp(line, "Content-Length: ", 16) == 0)
        {
          httplength = atoi((const char*)line + 16);
        }
        else if ((lasthttpaddress == 0) &&
          (strncmp(line, "X-KeyServer: ", 13) == 0))
        {
#if defined(_OLD_NEXT_)
          if (Resolve((char *)line + 13, lasthttpaddress) < 0)
#else
          if (Resolve(line + 13, lasthttpaddress) < 0)
#endif
            lasthttpaddress = 0;
        }
        else if (line.GetLength() < 1)
        {
          // got blank line separating header from body
          gothttpend = true;
          if (uubuffer.GetLength() >= 6 &&
            strncmp(uubuffer.GetHead(), "begin ", 6) == 0)
          {
            mode |= MODE_UUE;
            gotuubegin = false;
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
              gothttpend = false;
              httplength = 0;

              // our http only allows one packet per socket
              nothing_done = gethttpdone = true;
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
      else if (numRead == 0) need_close = true;       // connection closed

      AutoBuffer line;
      while (uubuffer.RemoveLine(line))
      {
        nothing_done = false;

        if (strncmp(line, "begin ", 6) == 0) gotuubegin = true;
        else if (strncmp(line, "POST ", 5) == 0) mode |= MODE_HTTP;
        else if (strncmp(line, "end", 3) == 0)
        {
          gotuubegin = gothttpend = false;
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
        nothing_done = false;
        tempbuffer.MarkUsed(numRead);

        // decrement from total if processing from a http body
        if ((mode & MODE_HTTP) && httplength)
        {
          httplength -= numRead;

          // our http only allows one packet per socket
          if (httplength == 0) nothing_done = gethttpdone = true;
        }

        // see if we should upgrade to different mode
        if (tempbuffer.GetLength() >= 6 &&
            strncmp(tempbuffer.GetHead(), "begin ", 6) == 0)
        {
          mode |= MODE_UUE;
          uubuffer = tempbuffer;
          gotuubegin = false;
        }
        else if (tempbuffer.GetLength() >= 5 &&
            strncmp(tempbuffer.GetHead(), "POST ", 5) == 0)
        {
          mode |= MODE_HTTP;
          uubuffer = tempbuffer;
          gothttpend = false;
          httplength = 0;
        }
        else netbuffer += tempbuffer;
      }
      else if (numRead == 0) need_close = true;
    }

    if (nothing_done)
    {
      if (need_close || gethttpdone) break;
#if (CLIENT_OS == OS_WIN32)
      Sleep( 100 );   // Prevent racing on error (1/10 second)
#elif (CLIENT_OS == OS_WIN16)
      // nothing
#elif (CLIENT_OS == OS_OS2)
      DosSleep( 100000 );   // Prevent racing on error (1/10 second)
#elif (CLIENT_OS == OS_NETWARE)
      delay( 1000 );  // full 1 second on Netware due to so many reported network problems.
#elif defined(__WATCOM__)
      delay( 100 );  // Prevent racing on error (1/10 second)
#elif (CLIENT_OS == OS_BEOS)
      snooze( 100000 );  // Prevent racing on error (1/10 second)
#elif (CLIENT_OS == OS_VMS)
      sleep( 1 );  // Prevent racing on error (1 second)
#elif (CLIENT_OS == OS_SOLARIS)
      sleep(1); // full 1 second on Solaris due to so many reported network problems.
#elif (CLIENT_OS == OS_HPUX)
  #ifdef _STRUCT_TIMESPEC
      // HP-UX 10.x has nanosleep() rather than usleep()
      struct timespec interval, remainder;
      interval.tv_sec = 0;
      interval.tv_nsec = 100000000;  // Prevent racing on error (1/10 second)
      nanosleep(&interval, &remainder);
  #else
      // HP-UX 9.x doesn't have nanosleep() or usleep()
      sleep(1);  // Prevent racing on error (1 second)
  #endif
#else
      usleep( 100000 );  // Prevent racing on error (1/10 second)
#endif

    }
  } // while (netbuffer.GetLength() < blah)


  // transfer back what was read in
  u32 bytesfilled = (netbuffer.GetLength() < length ?
      netbuffer.GetLength() : length);
  memmove(data, netbuffer.GetHead(), bytesfilled);
  netbuffer.RemoveHead(bytesfilled);

  if (need_close) Close();
  return bytesfilled;
#else
  return 0;
#endif
}

//////////////////////////////////////////////////////////////////////////////

// returns -1 on error, or 0 on success
s32 Network::Put( u32 length, const char * data )
{
#if !defined(NONETWORK)
  AutoBuffer outbuf;

  // if the connection is closed, try to reopen it once.
  if (sock == 0 || puthttpdone) if (Open()) return -1;


  if (mode & MODE_UUE)
  {
    /**************************/
    /***  Need to uuencode  ***/
    /**************************/
    outbuf += AutoBuffer("begin 644 query.txt\r\n");

    while (length > 0)
    {
      char line[80], *b = line;

      int n = (length > 45 ? 45 : length);
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
    char header[500], ipbuff[64];
    if (lasthttpaddress) {
      in_addr addr;
      addr.s_addr = lasthttpaddress;
      strncpy(ipbuff, inet_ntoa(addr), sizeof(ipbuff));
    } else {
      strncpy(ipbuff, server_name, sizeof(ipbuff));
    }

    if ( httpid[0] ) {
      sprintf(header, "POST http://%s:%li/cgi-bin/rc5.cgi HTTP/1.0\r\n"
         "Proxy-authorization: Basic %s\r\n"
         "Proxy-Connection: Keep-Alive\r\n"
         "Content-Type: application/octet-stream\r\n"
         "Content-Length: %lu\r\n\r\n",
         ipbuff, (long) lastport,
         httpid,
         (unsigned long) outbuf.GetLength());
    } else {
      sprintf(header, "POST http://%s:%li/cgi-bin/rc5.cgi HTTP/1.0\r\n"
         "Content-Type: application/octet-stream\r\n"
         "Content-Length: %lu\r\n\r\n",
         ipbuff, (long) lastport,
         (unsigned long) outbuf.GetLength());
    }
#if (CLIENT_OS == OS_OS390)
    __etoa(header);
#endif
    outbuf = AutoBuffer(header) + outbuf;
    puthttpdone = true;
  }

  return (LowLevelPut(outbuf.GetLength(), outbuf) != -1 ? 0 : -1);
#else
  return 0;
#endif
}

//////////////////////////////////////////////////////////////////////////////

// Returns length of read buffer
//    or 0 if connection closed
//    or -1 if no data waiting
#if !defined(NONETWORK)
s32 Network::LowLevelGet(u32 length, char *data)
{
  #if defined(SELECT_FIRST)
    fd_set rs;
    timeval tv = {0,0};
    FD_ZERO(&rs);
    FD_SET(sock, &rs);
    select(sock + 1, &rs, NULL, NULL, &tv);
    if (!FD_ISSET(sock, &rs)) return -1;
  #endif
  #if (CLIENT_OS == OS_BEOS)
  s32 numRead = recv(sock, data, length, 0);
  #else
  s32 numRead = read(sock, data, length);
  #endif

  #if (CLIENT_OS == OS_HPUX)
    // HPUX incorrectly returns 0 on a non-blocking socket with
    // data waiting to be read instead of -1.
    if (numRead == 0) numRead = -1;
  #endif

  return numRead;
}
#endif

//////////////////////////////////////////////////////////////////////////////

// returns length of sent data
//    or -1 on error
#if !defined(NONETWORK)
s32 Network::LowLevelPut(u32 length, const char *data)
{
  u32 writelen = write(sock, (char*)data, length);
  return (s32) (writelen != length ? -1 : (s32)writelen);
}
#endif

//////////////////////////////////////////////////////////////////////////////

void Network::MakeBlocking()
{
  MakeNonBlocking(sock, false);
}

//////////////////////////////////////////////////////////////////////////////

void MakeNonBlocking(SOCKET socket, bool nonblocking)
{
  // change socket to non-blocking
#if defined(NONETWORK)
  // nothing
#elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
  unsigned long flagon = nonblocking;
  ioctlsocket(socket, FIONBIO, &flagon);
#elif (CLIENT_OS == OS_VMS)
  #ifdef __VMS_UCX__
    // nonblocking sockets not directly supported by UCX
    // - DIGITAL's work around requires system privileges to use
  #else
    unsigned long flagon = nonblocking;
    socket_ioctl(socket, FIONBIO, &flagon);
  #endif
#elif (CLIENT_OS == OS_RISCOS)
  int flagon = nonblocking;
  ioctl(socket, FIONBIO, &flagon);
#elif (CLIENT_OS == OS_OS2)
  ioctl(socket, FIONBIO, (char *) &nonblocking, sizeof(nonblocking));
#elif (CLIENT_OS == OS_AMIGAOS)
  char flagon = nonblocking;
  IoctlSocket(socket, FIONBIO, &flagon);
#elif defined(FNDELAY)
  fcntl(socket, F_SETFL, nonblocking ? FNDELAY : 0);
#else
  fcntl(socket, F_SETFL, nonblocking ? O_NONBLOCK : 0);
#endif
}

//////////////////////////////////////////////////////////////////////////////

#if (CLIENT_OS == OS_OS2)
void Network::ensureonline()
{
  // DOD stuff
  DOD_On = 0;
/* Future stuff
  DOD_Retries = 0;
*/
  // If no DOD files, don't bother
  if(!(access(StartDOD, R_OK) || access(StopDOD, R_OK)))
  {
    DOD_Sleeptime = 60;

    if(!access(DODCfg, R_OK))      // read DOD config file if there is one
    {
      printf("Reading %s for DOD parameters...\n", DODCfg);

      FILE *cfg;
      cfg = fopen(DODCfg, "r+t");

      fgets(netstring, 25, cfg);     // Use netstring as temp buffer
      DOD_Sleeptime = atoi(netstring);   // Get sleep time

      fgets(netstring, 25, cfg);     // Get the real netstring

      nl = strlen(netstring) - 1;    // make sure there's no \n
      if(netstring[nl] == '\n')
        netstring[nl] = '\0';
/*
      fgets(netstring, 25, cfg);     // Get max retries
      if(atoi(netstring) > 0)
        DOD_Retries = atoi(netstring);
*/
      fclose(cfg);
    }
    else
      strcpy(netstring, "default");  // default for english OS/2

    if(!rweonline())
    {
      printf("Not Connected, Starting StartDOD.cmd...\n"
          "Waiting for %d seconds to be connected.\n", DOD_Sleeptime);
      do
      {
        spawnl( P_NOWAIT, StartDOD, (char *)NULL );
        sleep( DOD_Sleeptime );
        if(!rweonline())   // Still not connected.
        {
          printf("Could not connect, hanging up and trying again...\n");
          spawnl( P_NOWAIT, StopDOD, (char *)NULL );
          sleep( 10 );     // Give it time to run whatever program that hangs up
        }
        else
          DOD_On = 1;
      } while(!rweonline());
    }
  }
}
#endif

//////////////////////////////////////////////////////////////////////////////

#if (CLIENT_OS == OS_OS2)
void Network::checkoffline()
{
  if(DOD_On)
  {
    printf("Dialed in with DOD, disconnecting...\n");
    flushall();
    spawnl( P_NOWAIT, StopDOD, (char *)NULL );
  }
}
#endif

//////////////////////////////////////////////////////////////////////////////

#if (CLIENT_OS == OS_OS2)
int Network::rweonline(void)
{
  FILE *fp;
  char line[80];
  int online = 0, len;

  system("netstat -r > netstat.dat");

  fp = fopen("netstat.dat", "r+t");

  while(!feof(fp))
  {
    fgets( line, sizeof(line), fp );
    if(strstr( line, "default" ) != 0 )
      // found default route!
      online = 1;
  }

  // Clean up
  fclose(fp);
  unlink("netstat.dat");
  return online;
}
#endif

//////////////////////////////////////////////////////////////////////////////

#define B64_ENC(Ch) (char) (base64table[(char)(Ch) & 63])
char * Network::base64_encode(char *username, char *password)
{
  static char in[80], out[80];
  static const char base64table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
                                    "ghijklmnopqrstuvwxyz0123456789+/";

  u32 length = strlen(username) + strlen(password) + 1;
  if ((length+1) >= sizeof(in) || (length + 2) * 4 / 3 >= sizeof(out)) return NULL;
  sprintf(in, "%s:%s\n", username, password);

  char *b = out, *data = in;
  for (; length > 2; length -= 3, data += 3)
  {
    *b++ = B64_ENC(data[0] >> 2);
    *b++ = B64_ENC(((data[0] << 4) & 060) | ((data[1] >> 4) & 017));
    *b++ = B64_ENC(((data[1] << 2) & 074) | ((data[2] >> 6) & 03));
    *b++ = B64_ENC(data[2] & 077);
  }

  if (length == 1)
  {
    *b++ = B64_ENC(data[0] >> 2);
    *b++ = B64_ENC((data[0] << 4) & 060);
    *b++ = '=';
    *b++ = '=';
  }
  else if (length == 2)
  {
    *b++ = B64_ENC(data[0] >> 2);
    *b++ = B64_ENC(((data[0] << 4) & 060) | ((data[1] >> 4) & 017));
    *b++ = B64_ENC((data[1] << 2) & 074);
    *b++ = '=';
  }
  *b = 0;

  return out;
}

//////////////////////////////////////////////////////////////////////////////

