// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

const char *netio_cpp(void) {
return "@(#)$Id: netio.cpp,v 1.2 2000/06/03 03:24:39 jlawson Exp $"; }

#define __NETIO_CPP__ /* suppress redefinitions in netio.h */
#include "netio.h"


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <errno.h>

#if (defined(__GLIBC__) && (__GLIBC__ >= 2))
#define SOCKLEN_T socklen_t
#elif (CLIENT_OS == OS_MACOS) || (CLIENT_OS == OS_NETBSD)
#define SOCKLEN_T socklen_t
#elif ((CLIENT_OS == OS_BSDOS) && (_BSDI_VERSION > 199701))
#define SOCKLEN_T size_t
#elif ((CLIENT_OS == OS_NTO2) || (CLIENT_OS == OS_QNX))
#define SOCKLEN_T size_t
#else
#define SOCKLEN_T int
#endif


#if (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_MACOS)
#define ERRNO_IS_UNUSABLE_FOR_CONN_ERRMSG
#endif


// ----------------------------------------------------------------------

const char *netio_describe_error(void)
{
#if (CLIENT_OS == OS_WIN32)
  switch (WSAGetLastError())
  {
  case WSAEINTR: return "(WSAEINTR): Interrupted system call";
  case WSAEBADF: return "(WSAEBADF): Bad file number";
  case WSAEACCES: return "(WSAEACCES): Permission denied";
  case WSAEFAULT: return "(WSAEFAULT): Bad address passed";
  case WSAEINVAL: return "(WSAEINVAL): Invalid parameter passed";
  case WSAEMFILE: return "(WSAEMFILE): Too many open files";
  case WSAEWOULDBLOCK: return "(WSAEWOULDBLOCK): Operation would block";
  case WSAEINPROGRESS: return "(WSAEINPROGRESS): Operation is now in progress";
  case WSAEALREADY: return "(WSAEALREADY): Operation is already in progress";
  case WSAENOTSOCK: return "(WSAENOTSOCK): Socket operation on non-socket";
  case WSAEDESTADDRREQ: return "(WSAEDESTADDRREQ): Destination address required";
  case WSAEMSGSIZE: return "(WSAEMSGSIZE): Message is too long";
  case WSAEPROTOTYPE: return "(WSAEPROTOTYPE): The protocol is of the wrong type for the socket";
  case WSAENOPROTOOPT: return "(WSAENOPROTOOPT): The requested protocol is not available";
  case WSAEPROTONOSUPPORT: return "(WSAEPROTONOSUPPORT): The requested protocol is not supported";
  case WSAESOCKTNOSUPPORT: return "(WSAESOCKTNOSUPPORT): The specified socket type is not supported";
  case WSAEOPNOTSUPP: return "(WSAEOPNOTSUPP): The specified operation is not supported";
  case WSAEPFNOSUPPORT: return "(WSAEPFNOSUPPORT): The specified protocol family is not supported";
  case WSAEAFNOSUPPORT: return "(WSAEAFNOSUPPORT): The specified address family is not supported";
  case WSAEADDRINUSE: return "(WSAEADDRINUSE): The specified address is already in use";
  case WSAEADDRNOTAVAIL: return "(WSAEADDRNOTAVAIL): The requested address is unassignable";
  case WSAENETDOWN: return "(WSAENETDOWN): The network appears to be down";
  case WSAENETUNREACH: return "(WSAENETUNREACH): The network is unreachable";
  case WSAENETRESET: return "(WSAENETRESET): The network dropped the connection on reset";
  case WSAECONNABORTED: return "(WSAECONNABORTED): Software caused a connection abort";
  case WSAECONNRESET: return "(WSAECONNRESET): Connection was reset by peer";
  case WSAENOBUFS: return "(WSAENOBUFS): Out of buffer space";
  case WSAEISCONN: return "(WSAEISCONN): Socket is already connected";
  case WSAENOTCONN: return "(WSAENOTCONN): Socket is not presently connected";
  case WSAESHUTDOWN: return "(WSAESHUTDOWN): Can't send data because socket is shut down";
  case WSAETOOMANYREFS: return "(WSAETOOMANYREFS): Too many references, unable to splice";
  case WSAETIMEDOUT: return "(WSAETIMEDOUT): The connection timed out";
  case WSAECONNREFUSED: return "(WSAECONNREFUSED): The connection was refused";
  case WSAELOOP: return "(WSAELOOP): Too many symbolic link levels";
  case WSAENAMETOOLONG: return "(WSAENAMETOOLONG): File name is too long";
  case WSAEHOSTDOWN: return "(WSAEHOSTDOWN): The host appears to be down";
  case WSAEHOSTUNREACH: return "(WSAEHOSTUNREACH): The host is unreachable";
  case WSAENOTEMPTY: return "(WSAENOTEMPTY): The directory is not empty";
  case WSAEPROCLIM: return "(WSAEPROCLIM): There are too many processes";
  case WSAEUSERS: return "(WSAEUSERS): There are too many users";
  case WSAEDQUOT: return "(WSAEDQUOT): The disk quota is exceeded";
  case WSAESTALE: return "(WSAESTALE): Bad NFS file handle";
  case WSAEREMOTE: return "(WSAEREMOTE): There are too many levels of remote in the path";
  case WSASYSNOTREADY: return "(WSASYSNOTREADY): Network sub-system is not ready or unusable";
  case WSAVERNOTSUPPORTED: return "(WSAVERNOTSUPPORTED): The requested version is not supported";
  case WSANOTINITIALISED: return "(WSANOTINITIALISED): Socket system is not initialized";
  default: break;
  }
  return "Unknown error";
#elif (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_OS2)
  // publicly accessible error descriptions aren't available?
  return strerror(errno);
#else
  if (errno < sys_nerr)
    return sys_errlist[errno];
  return "Unknown error";
#endif
}

// ----------------------------------------------------------------------

static void netio_logerr(const char *fmt, ... )
{
  va_list ap;
  char scratch[128];
  va_start( ap, fmt );
  vsprintf( scratch, fmt, ap );
  va_end( ap );
#ifdef PROXYTYPE
  globalLogger->LogInfo(LOG_GENERAL | LOG_ERRSEVERE,scratch);
#else
  //Log("%s\n", scratch );
  printf("%s\n", scratch);
#endif
  return;
}

// ----------------------------------------------------------------------

int netio_geterrno(void)
{
  #if (CLIENT_OS == OS_WIN32)
  int err = WSAGetLastError();
  if (err == WSAEINTR)
    err = EINTR;
  return err;
  #elif (CLIENT_OS == OS_OS2)
  return sock_errno();
  #else
  return errno;
  #endif
}

// ----------------------------------------------------------------------

int netio_seterrno(int err)
{
  int last = netio_geterrno();
  #if (CLIENT_OS == OS_WIN32)
  WSASetLastError(err);
  #else
  errno = err;
  #endif
  return last;
}

// ----------------------------------------------------------------------

// Returns a pointer to a static buffer that receives a
// dotted decimal ASCII conversion of the specified IP address.
// short circuit the u32 -> in_addr.s_addr ->inet_ntoa method.
// besides, it works around context issues.
const char *netio_ntoa(u32 addr)
{
  static char buff[sizeof("255.255.255.255  ")];
  char *p = (char *)(&addr);
  sprintf( buff, "%d.%d.%d.%d",
      (p[0]&255), (p[1]&255), (p[2]&255), (p[3]&255) );
  return buff;
}

// ----------------------------------------------------------------------

static unsigned int _opensocketcount = 0;
unsigned int netio_getopensocketcount(void)
{
  return _opensocketcount;
}

// ----------------------------------------------------------------------

// Creates a new network socket handle.
// Returns -1 on error, 0 on success.

int netio_createsocket(SOCKET &sock, bool nonblocking)
{
  sock = INVALID_SOCKET;
#if defined(_TIUSER_)                        //TLI
  sock = t_open("/dev/tcp", O_RDWR, NULL);
  if ( sock == -1 )
    sock = INVALID_SOCKET;
#elif (defined(AF_INET) && defined(SOCK_STREAM)) //BSD socks
  if (( (int)(sock = socket(AF_INET, SOCK_STREAM, 0)) ) < 0 )
    sock = INVALID_SOCKET;
#endif
  if (sock != INVALID_SOCKET)
  {
    // Set blocking or non-blocking mode.  Although the created
    // socket begins as blocking, some (like RISCOS) need to prep the
    // socket, (in RISCOS's case, allow the OS to preemptively multitask
    // on blocking socks)
    if (netio_setsockopt( sock, CONDSOCK_BLOCKMODE, nonblocking ? 0 : 1 )!=0)
    {
      if (nonblocking) //ignore the error if blocking
      {
        netio_close(sock);
        sock = INVALID_SOCKET;
      }
    }
  }
  if (sock != INVALID_SOCKET)
  {
    // permit tcp packets of at least this size.
    #if defined(PROXYTYPE)
    netio_setsockopt( sock, CONDSOCK_SETMINBUFSIZE, 1024*16);
    #else
    netio_setsockopt( sock, CONDSOCK_SETMINBUFSIZE, 1024*2);
    #endif
    _opensocketcount++;
    return 0;
  }
  return -1;
}

// ----------------------------------------------------------------------

// Closes an opened socket handle.
// Note that sock is modified (by reference) to INVALID_SOCKET
// as a side-effect.
// Returns -1 on error, 0 on success.

int netio_close(SOCKET &sock)
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
    _opensocketcount--;
    sock = INVALID_SOCKET;
    return rc;
  }
#else                                                  //BSD socks
  if ( sock != INVALID_SOCKET )
  {
    #if defined(USES_SIGIO)
    netio_setsockopt( sock, CONDSOCK_ASYNCSIG, 0 );
    #endif
    netio_setsockopt( sock, CONDSOCK_BLOCKMODE, 1 );
    #if (defined(AF_INET) && defined(SOCK_STREAM))
    shutdown( sock, 2 );
    #endif
    #if (CLIENT_OS == OS_WIN32)
    int retcode = closesocket(sock);
    #elif (CLIENT_OS == OS_OS2)
    int retcode = soclose(sock);
    #else
    int retcode = close(sock);
    #endif
    _opensocketcount--;
    sock = INVALID_SOCKET;
    return (retcode);
  }
#endif
  sock = INVALID_SOCKET;
  return 0;
}

// ----------------------------------------------------------------------

// Performs a name-server lookup of the specified name and randomly
// returns one of the corresponding IP Addresses (if there is more
// than one matching address).
// Returns -1 on error, 0 on success.

int netio_resolve(const char *hosttgt, u32 &hostaddress)
{
  char host[128];

  hostaddress = 0;
  if ( hosttgt)
  {
    unsigned int len = 0;
    const char *p = hosttgt;
    while (*p && isspace(*p))
      p++;
    while (*p && !isspace(*p) && len < sizeof(host))
      host[len++] = (char)*p++;
    host[len] = 0;
    if (len != 0)
      hostaddress = (u32) inet_addr( host );
    hosttgt = (const char *)&host[0];
  }

  if (hostaddress == ((u32)-1L))
  {
    struct hostent *hp;
    #if (CLIENT_OS == OS_NETWARE)
    struct nwsockent s_nwsockent;
    hp = NetDBgethostbyname( &s_nwsockent, host );
    #else
    hp = gethostbyname( host );
    #endif
    if (hp != NULL)
    {
      int index = 0;
      #if 0 /* let nameserver optimize for us */
      // randomly select one
      while (hp->h_addr_list[index])
        index++;
      if (index)
        index = rand() % index;
      #endif
      if ((hostaddress = *((u32 *)(hp->h_addr_list[index]))) == ((u32)-1L))
        hostaddress = 0;
    }
    else
      hostaddress = 0;
  }
  if (hostaddress == 0)
    return -1;
  return(0);
}

// ----------------------------------------------------------------------

// Retrieves the hostname of the local machine.
// Returns -1 on error, 0 on success.

int netio_gethostname( char *buffer, unsigned int buflen )
{
  char scratch[256];
  if (!buffer || !buflen)
    return -1;
  if (gethostname( scratch, sizeof(scratch) ) != 0)
    return -1;
  if (buflen > sizeof(scratch))
    buflen = sizeof(scratch);
  strncpy( buffer, scratch, buflen );
  buffer[buflen-1] = '\0';
  return 0;
}

// ----------------------------------------------------------------------

// Retrieves the primary IP Address of the local machine.
// Returns -1 on error, 0 on success.

int netio_gethostaddr( u32 * addr )
{
  char buffer[512];
  u32 tmpaddr = 0;
  if ( netio_gethostname( buffer, sizeof(buffer) ) != 0 )
    return -1;
  if ( netio_resolve( buffer, tmpaddr ) != 0)
    return -1;
  if ( addr )
    *addr = tmpaddr;
  return 0;
}

// ----------------------------------------------------------------------

// bind to a specific address. this is *NOT OPTIONAL*.
// Every socket, regardless whether as a client or a server must be bound.
// Returns -1 on error, 0 on success.

int netio_bind( SOCKET sock, u32 addr, u16 port )
{
  int rc = -1;
  if (sock == INVALID_SOCKET)
    return -1;

  #if defined(_TIUSER_)
  if (!addr && !port)
    rc = t_bind( sock, NULL, NULL ); /* bind the fd to default proto */
  else
  {

  }
  #elif defined(AF_INET) && defined(SOCK_STREAM)
  if (!addr && !port)
    rc = 0; /* not needed since connect() will bind for us */
  else
  {
    /* bind the socket to address and port given */
    struct sockaddr_in sin;
    memset(&sin, 0, sizeof(sin));
    sin.sin_addr.s_addr = addr;
    sin.sin_port = htons(port);
    sin.sin_family = PF_INET;
    while (bind(sock, (struct sockaddr *)&sin, sizeof(sin)) < 0)
    {
      if (netio_geterrno() == EINTR) /* caught a signal */
        continue;                    /* retry */
      return -1;
    }
  }
  rc = 0;
  #endif
  addr = addr;
  port = port;
  return rc;
}

// ----------------------------------------------------------------------

// Set up to receive connections on a socket
// Returns -1 on error, 0 on success.

int netio_listen( SOCKET sock, int backlog )
{
  if (sock == INVALID_SOCKET || backlog < 1)
  {                  /* BSD permits negative backlog to mean max. We don't. */
    netio_seterrno(0 /*EBADF*/);
    return -1;
  }
  #if defined(_TIUSER_)
  return 0; /* not needed */
  #elif defined(AF_INET) && defined(SOCK_STREAM)
  while (listen(sock, backlog) < 0)
  {
    if (netio_geterrno() == EINTR) /* caught a signal */
      continue;                    /* retry */
    return -1;
  }
  return 0;
  #else
  return -1;
  #endif
}

// ----------------------------------------------------------------------

// Creates a socket and initializes it as a listener to accept incoming
// TCP/IP connections on the specified interface and port number.
// Returns -1 on error, 0 on success.

int netio_openlisten(SOCKET &sock, u32 addr, u16 port, bool nonblocking)
{
  /* allocate the new socket */
  if (netio_createsocket(sock, nonblocking) < 0)
  {
    netio_logerr("socket(): %s", netio_describe_error() );
    sock = INVALID_SOCKET;
    return -1;
  }

  /* try to reuse sockets. Although technically not necessary for */
  /* the client or proxy, some OSs (which?) will not allow a listener */
  /* to bind() for some time if the socket was not explicitely/properly */
  /* closed by the previous owner */
  netio_setsockopt(sock, CONDSOCK_REUSEADDR, 1);

  /* bind the socket to address and port given */
  if (netio_bind( sock, addr, port ) < 0)
  {
    netio_logerr( "bind(%s:%u): %s ", netio_ntoa( addr ),
                  (unsigned int)port, netio_describe_error() );
    netio_close(sock);
    sock = INVALID_SOCKET;
    return -1;
  }

  /* set up to receive connections on this socket */
  if (netio_listen( sock, 5 /*backlog*/ ) < 0)
  {
    netio_logerr( "listen(%s:%u): %s ", netio_ntoa( addr ),
                  (unsigned int)port, netio_describe_error() );
    netio_close(sock);
    sock = INVALID_SOCKET;
    return -1;
  }

  #if defined(USES_SIGIO)
  if (nonblocking) /* created in nonblocking mode */
  {
    /* so turn it back off */
    if (netio_setsockopt( sock, CONDSOCK_BLOCKMODE, 1 ) < 0)
    {
      netio_logerr( "ioctl(): %s ", netio_describe_error() );
      netio_close(sock);
      sock = INVALID_SOCKET;
      return -1;
    }
  }
  /* enable interrupt by signal on this socket */
  if (netio_setsockopt( sock, CONDSOCK_ASYNCSIG, 1 )!=0)
  {
    netio_logerr( "ioctl(): %s ", netio_describe_error() );
    netio_close(sock);
    sock = INVALID_SOCKET;
    return -1;
  }
  #endif

  return 0;
}

// ----------------------------------------------------------------------

// connect to a desination host. socket *MUST* have been bound.
// Returns -1 on error, 0 on success.

int netio_lconnect( SOCKET sock, u32 addr, u16 port )
{
#if defined(_TIUSER_)                                         //OSI/XTI/TLI
  int rc = -1;
  //if ( t_bind( sock, NULL, NULL ) != -1 ) //already bound
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
      sin->sin_port = htons( that_port );
      rc = t_connect( sock, sndcall, NULL);
      #error UNFINISHED. needs nonblocking handling
      t_free((char *)sndcall, T_CALL);
    }
  }
  return rc;
#elif defined(AF_INET) && defined(SOCK_STREAM)
  int repeatme = 0;
  do
  {
    struct sockaddr_in sin;
    memset(&sin, 0, sizeof(sin));
    sin.sin_addr.s_addr = addr;
    sin.sin_port = htons(port);
    sin.sin_family = PF_INET;

    if (connect(sock, (struct sockaddr *)&sin, sizeof(sin)) < 0)
    {
      int myerrno = netio_geterrno();
      if (myerrno == EINPROGRESS)
        break;
      if (myerrno != EINTR) /* not sigpipe and family */
        return -1;          /* don't restart the connect() */
      repeatme = 1;
    }
  } while (repeatme);
  return 0;
#else
  return -1;
#endif
}

// ----------------------------------------------------------------------

// Creates a new socket and attempts to connect it to the specified
// target host address and port.  If a listenaddr is specified, then
// the outgoing connection will be created such that it originates
// from the specified local interface.
// Returns -1 on error, 0 on success.

int netio_connect(SOCKET &sock, const char *host, u16 port, u32 &addr,
    u32 localaddr, bool nonblocking)
{
  if (host) /* we already have an 'addr' if host is null */
  {
    /* resolve the address we will connect to. */
    if (netio_resolve(host, addr) < 0)
      addr = 0;
  }
  if (!addr)
  {
    netio_logerr("Unable to resolve hostname: \"%s\"",(host?host:"(null)"));
    sock = INVALID_SOCKET;
    return -1;
  }

  /* allocate the new socket */
  if (netio_createsocket(sock, nonblocking) < 0)
  {
    netio_logerr("socket(): %s", netio_describe_error() );
    sock = INVALID_SOCKET;
    return -1;
  }

  /* bind the socket to address and port given. NOT OPTIONAL. See function. */
  if (netio_bind( sock, localaddr, 0 ) < 0)
  {
    netio_logerr( "bind(%s:0): %s ", netio_ntoa( addr ),
                  netio_describe_error() );
    netio_close(sock);
    sock = INVALID_SOCKET;
    return -1;
  }

  /* connect to the destination host */
  if (netio_lconnect( sock, addr, port ) < 0)
  {
    netio_logerr( "connect(%s:%u): %s ", netio_ntoa( addr ),
                  (unsigned int)port, netio_describe_error() );
    netio_close(sock);
    sock = INVALID_SOCKET;
    return -1;
  }

  #if defined(USES_SIGIO)
  if (nonblocking) /* created in nonblocking mode */
  {
    /* so turn it back off */
    if (netio_setsockopt( sock, CONDSOCK_BLOCKMODE, 1 ) < 0)
    {
      netio_logerr( "ioctl(): %s ", netio_describe_error() );
      netio_close(sock);
      sock = INVALID_SOCKET;
      return -1;
    }
  }
  /* enable interrupt by signal on this socket */
  if (netio_setsockopt( sock, CONDSOCK_ASYNCSIG, 1 )!=0)
  {
    netio_logerr( "ioctl(): %s ", netio_describe_error() );
    netio_close(sock);
    sock = INVALID_SOCKET;
    return -1;
  }
  #endif

  return 0;
}

// ----------------------------------------------------------------------

int netio_laccept(SOCKET sock, SOCKET *thatsock, u32 *thataddr, int *thatport)
{
  u32 taddr = 0;
  int tport = 0;
  SOCKET snew = INVALID_SOCKET;

  #if defined(_TIUSER_)
  {
    struct sockaddr_in Sin;


    tport = ((int)(ntohs(Sin.sin_port))) & 0xffff;
    taddr = Sin.sin_addr.s_addr;
  }
  #elif defined(AF_INET) && defined(SOCK_STREAM)
  {
    struct sockaddr_in Sin;
    SOCKLEN_T sinlen = (SOCKLEN_T) sizeof(Sin);
    while (((int)(snew = accept(sock, (struct sockaddr *)&Sin, &sinlen))) < 0)
    {
      if (netio_geterrno() == EINTR)
        continue;
      snew = INVALID_SOCKET;
      break;
    }
    tport = ((int)(ntohs(Sin.sin_port))) & 0xffff;
    taddr = Sin.sin_addr.s_addr;
  }
  #endif

  if (snew != INVALID_SOCKET)
  {
    if (thatsock)
      *thatsock = snew;
    if (thataddr)
      *thataddr = taddr;
    if (thatport)
      *thatport = tport;
    _opensocketcount++;
    return 0;
  }
  return -1;
}

// ----------------------------------------------------------------------

// Waits for a new incoming connection to be detected on the specified
// listener socket and returns a new socket handle that corresponds
// with the newly connected client.  The IP Address of the newly
// connected client is also made available.  The nonblocking option
// controls whether the newly accepted socket is to be made nonblocking,
// not whether the acceptance option occurs in a nonblocking fashion.
// Returns -1 on error, 0 on success.

int netio_accept(SOCKET sock, SOCKET *newsock, u32 *thataddr, int *thatport,
                 bool nonblocking)
{
  u32 taddr;
  int tport;
  SOCKET snew;

  if (netio_laccept(sock, &snew, &taddr, &tport) < 0)
    return -1;

  /* Reject connections originating from a secure port. */
  if (tport < 1024)
  {
    netio_logerr( "Rejected connection from %s:%u", netio_ntoa(taddr), tport);
    netio_close(snew);
    return -1;
  }

  #if defined(USES_SIGIO)
  nonblocking = nonblocking; //unused
  if (netio_setsockopt( snew, CONDSOCK_ASYNCSIG, 1 ) != 0)
  {
    netio_close(snew);
    return -1;
  }
  #else
  if (netio_setsockopt( snew, CONDSOCK_BLOCKMODE, nonblocking ? 0 : 1 )!=0)
  {
    if (nonblocking)
    {
      netio_close(snew);
      return -1;
    }
  }
  #endif

  // permit tcp packets of at least this size.
  #if defined(PROXYTYPE)
  netio_setsockopt( sock, CONDSOCK_SETMINBUFSIZE, 1024*16);
  #else
  netio_setsockopt( sock, CONDSOCK_SETMINBUFSIZE, 1024*2);
  #endif

  if (newsock) *newsock = snew;
  if (thataddr) *thataddr = taddr;
  if (thatport) *thatport = tport;
  return 0;
}

// ----------------------------------------------------------------------

// Translates a numeric IP Address into a printable ASCII text-string.
// shortcut to avoid use of socket.h structures outside netio.cpp
// as well as 'inet_ntoa(*(in_addr*)&hostaddr)' nastiness.
// so... yes, "it is apparently better to reinvent the wheel".
// Returns -1 on error, 0 on success.

const char *netio_ntoa(u32 hostaddr)
{
  static char buff[18];
  char *p = (char *)(&hostaddr);
  sprintf( buff, "%d.%d.%d.%d", (p[0]&255),(p[1]&255),(p[2]&255),(p[3]&255));
  return buff;
}

// ----------------------------------------------------------------------

int netio_recv(SOCKET s, void *data, int len)
{
  if (!length)
    return -1;

  u32 totalread = 0;
  u32 rcvquota = 1500;
  time_t timenow = 0, starttime = 0, stoptime = 0;
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
  #elif (CLIENT_OS == OS_MACOS)
  if (rcvquota > 0xFFFF)  // Mac network library uses "unsigned short"
    rcvquota = 0xFFFF;
  #else
  if (rcvquota > INT_MAX)
    rcvquota = INT_MAX;
  #endif

  if (isnonblocking)
    stoptime = time(&starttime) + (time_t)iotimeout;

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
    #elif (CLIENT_OS == OS_MACOS)
    // Note: MacOS client does not use XTI, and the socket emulation
    // code doesn't support select.
    {
      bytesread = read( sock, data, toread);
      if (bytesread == -1)
      {
        if ( !valid_socket(sock) )
          bytesread = 0; // set as socket closed
      }
      else if (bytesread == 0) // should never happen?
        bytesread = -1; // set as none waiting
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
      if (time(&timenow) > stoptime || timenow < starttime)
        break;
      if (!isnonblocking && CheckExitRequestTrigger())
        break;
      ++sleptcount;
    }
    unsigned long sleepdur = ((unsigned long)(sleptcount+1)) * sleepms;
    if (sleepdur > 1000000UL)
      sleep( sleepdur / 1000000UL );
    if ((sleepdur % 1000000UL) != 0)
      usleep( sleepdur % 1000000UL );
    if (!isnonblocking && CheckExitRequestTrigger())
      break;
  } while (length);

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

// Returns length of sent data or 0 if the socket is
// closed, or -1 if timeout/nodata

int netio_send(SOCKET s, const void *ccdata, int length)
{
  if (length == 0)
    return -1;

  u32 totaltowrite = length;
  u32 totalwritten = 0;
  u32 sendquota = 1500; /* how much to send per send() call */
  int firsttime = 1;
  time_t timenow = 0, starttime = 0, stoptime = 0;
  int sleptcount = 0; /* ... in a row */
  int sleepms = 250; /* sleep time in millisecs. adjust here if needed */
  char *data;
  *((const char **)&data) = ccdata; /* get around const being used for send */

  if (isnonblocking)
    stoptime = time(&starttime) + (time_t)iotimeout;

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
  #elif (CLIENT_OS == OS_WIN16)
  if (sendquota > 0x7FFF)  /* 16 bit OS but int is 32 bits */
    sendquota = 0x7FFF;
  #elif (CLIENT_OS == OS_MACOS)
  if (sendquota > 0xFFFF)  // Mac network library uses "unsigned short"
    sendquota = 0xFFFF;
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
   #elif (CLIENT_OS == OS_MACOS)
    // Note: MacOS client does not use XTI, and the socket emulation
    // code doesn't support select.
    int noiocount = 0;
    written = -2;
    while (written == -2)
    {
      written = write(sock, data, (unsigned long)towrite);
      if (written == 0)       // transport provider accepted nothing
      {                   // should never happen unless 'towrite' was 0
        if ((++noiocount) < 3)
        {
          written = -2;   // retry
          usleep(500000); // 0.5 secs
        }
      }
      else if (written == -1)
      {
        if (!valid_socket(sock)) return(0);
      }
    }
    #elif defined(AF_INET) && defined(SOCK_STREAM)      //BSD 4.3 sockets
    #if (CLIENT_OS != OS_BEOS)
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
    #endif
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
      if (time(&timenow) < starttime)
        break;
      else if (timenow > stoptime)
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
    if (!isnonblocking && CheckExitRequestTrigger())
      break;
  } while (length);

  #ifdef DEBUGTHIS
  Log("LLPut: towrite=%d, written=%d\n", totaltowrite, totalwritten );
  #endif
  totaltowrite = totaltowrite; //squash compiler warning
  return ((totalwritten != 0) ? ((int)totalwritten) : (-1));
}

// ----------------------------------------------------------------------

#if defined(XFD_SETSIZE)
#pragma message "xfd_select services will be used"
#include "sleepdef.h"

void XFD_CLR(SOCKET fd, xfd_set *set)
{
  register u_int i, count = set->fd_count;
  for (i=0; i< count; i++)
  {
    if (set->fd_array[i] == fd)
    {
      set->fd_count = --count;
      while (i < count)
        set->fd_array[i] = set->fd_array[i+1];
      break;
    }
  }
  return;
}

void XFD_ZERO(xfd_set *set)
{
  set->fd_count = 0;
}

int XFD_ISSET(SOCKET fd, xfd_set *set)
{
  register u_int i, count = set->fd_count;
  for (i=0; i < count; i++)
  {
    if (set->fd_array[i] == fd)
      return 1;
  }
  return 0;
}

void XFD_SET(SOCKET fd, xfd_set *set)
{
  if (set->fd_count < XFD_SETSIZE)
    set->fd_array[set->fd_count++] = fd;
  return;
}

int xfd_select( int width, xfd_set *or, xfd_set *ow, xfd_set *ox, struct timeval *tvp )
{
  struct {u_int ocount; fd_set fdset; fd_set *fdsetp; xfd_set *src, *osrc;} table[3];
  u_int biggest_xfdset;
  unsigned long idle_count, time_stop;
  char *buf, *mbuf;
  int rc; unsigned int pos, bufsize;

  // compute the time at which we should stop waiting.
  idle_count = 0;
  time_stop = ((unsigned long)(-1L));
  if (tvp)
  {
    time_stop = 0;
    if (tvp->tv_usec != 0)
    {
      unsigned long cps = CLOCKS_PER_SEC; // \ squelch compiler warning
      if (cps <= 1000000ul)               // / about always being true
      {
        time_stop = (unsigned long)
         ( tvp->tv_usec / (((1000000ul+(CLOCKS_PER_SEC-1))/CLOCKS_PER_SEC)) );
      }
      else
      {
        time_stop = (unsigned long)( (CLOCKS_PER_SEC / 256) *
             (((1000000ul + ((tvp->tv_usec)-1)) << 8 ) / tvp->tv_usec) );
      }
      if (!time_stop)
        time_stop++;
    }
    time_stop += (unsigned long)(CLOCKS_PER_SEC * tvp->tv_sec);
    if (time_stop != 0)
      time_stop += (unsigned long)clock();
  }

  // short sleep to prevent spinning.
  usleep(100);

  // put our input sets into an array.
  table[0].osrc = or;
  table[1].osrc = ow;
  table[2].osrc = ox;

  // compute how much memory we will need to allocate.
  bufsize = 0;
  for (pos = 0; pos < 3; pos++)
  {
    if (table[pos].osrc)
    {
      if ((table[pos].osrc)->fd_count == 0)
        table[pos].osrc = (xfd_set *)0;  /* nothing to do here */
      else
      {
      bufsize += sizeof(u_int);
      bufsize += ( sizeof((table[pos].osrc)->fd_array[0]) *
                 ((table[pos].osrc)->fd_count + 1) );
      }
    }
  }

  // if no sockets specified, then return.
  if (bufsize == 0)
  {
    if (!tvp)  /* should block forever */
    {
      errno = EINTR;
      return -1;
    }
    tvp->tv_sec = 0;
    tvp->tv_usec = 0;
    if (time_stop != 0)
    {
      while (time_stop < (unsigned long)clock())
        usleep(1000);
    }
    return 0;
  }

  // allocate the buffer.
  buf = (char *)malloc( bufsize );
  if (!buf)
  {
    #if (CLIENT_OS == OS_WIN32)  /* others will have ENOMEM set */
    WSASetLastError(WSAENOBUFS);
    #endif
    return -1;
  }

  // copy input sets into our buffer and count maximum set size.
  biggest_xfdset = 0;
  mbuf = buf;
  for (pos = 0; pos < 3; pos++)
  {
    table[pos].ocount = 0;
    table[pos].src = (xfd_set *)0;
    if (table[pos].osrc)
    {
      unsigned int bsize;
      table[pos].ocount = (table[pos].osrc)->fd_count;
      table[pos].src = (xfd_set *)mbuf;
      bsize = sizeof(u_int);
      bsize += ( sizeof((table[pos].osrc)->fd_array[0]) *
                 ((table[pos].osrc)->fd_count /* + 1 */) );
      memcpy((void *)(table[pos].src),(void *)(table[pos].osrc),bsize);
      table[pos].src->fd_count = 0;

      if (table[pos].ocount > biggest_xfdset)
        biggest_xfdset = table[pos].ocount;
      mbuf += bsize;
    }
  }

  // main select loop occurs here.
  rc = 0;
  while (rc == 0)
  {
    u_int i;

    for (i = 0; i < biggest_xfdset; i += FD_SETSIZE)
    {
      width = 0;

      // build the three real fdsets that we will use.
      for (pos = 0; pos < 3; pos++ )
      {
        u_int ocount = table[pos].ocount; /* original fd_count for this xfd_set */
        table[pos].fdsetp = (fd_set *)0;
        if (i < ocount )
        {
          u_int n; int swidth = 0;
          FD_ZERO( &table[pos].fdset );
          for (n = 0; n < FD_SETSIZE && (n+i) < ocount; n++)
          {
            FD_SET( table[pos].src->fd_array[n+i], &table[pos].fdset );
            swidth++;
          }
          if ( swidth )
          {
            table[pos].fdsetp = &table[pos].fdset;
            if (swidth > width)
              width = swidth;
          }
        }
      }

      // perform the actual select.
      if ( width )
      {
        if ( biggest_xfdset <= FD_SETSIZE )
        {
          // if the select fits in one operation, then do it all at once.
          width = select( width, table[0].fdsetp,
              table[1].fdsetp, table[2].fdsetp, tvp );
          time_stop = 0;      // forces us to drop out after one loop.
        }
        else
        {
          struct timeval tv = { 0, 100 };
          width = select( width, table[0].fdsetp,
              table[1].fdsetp, table[2].fdsetp, &tv );
        }
      }

      // if the select failed...
      if ( width < 0 )
      {
        if (netio_geterrno() == EINTR)
          width = 0; /* ignore it */
        else
        {
          rc = -1;
          break;
        }
      }

      // if the select succeeded...
      if ( width > 0 )
      {
        for (pos = 0; pos < 3; pos++ )
        {
          if ( table[pos].fdsetp )
          {
            u_int n, ocount = table[pos].ocount;
            for (n = 0; n < FD_SETSIZE && (n+i) < ocount; n++)
            {
              SOCKET fd = table[pos].src->fd_array[n+i];
              if (FD_ISSET(fd, table[pos].fdsetp))
              {
                table[pos].src->fd_array[table[pos].src->fd_count++] = fd;
                rc++;
              }
            }
          }
        }
      }

    } /* for (i = 0; i < XFDSETSIZE; i += FD_SETSIZE) */

    if (rc == 0 && tvp )
    {
      if (time_stop == 0) /* tv_sec and tv_usec are both zero */
        break;
      if (((unsigned long)clock()) > time_stop)
        break;
      idle_count++;
      usleep(1000);
    }
  }

  // if we didn't loop, then throw in another sleep to prevent spinning.
  if (idle_count == 0)
    usleep(100);

  // do final result handling.
  if (rc > 0)
  {
    for (pos = 0; pos < 3; pos++)
    {
      if (table[pos].osrc)
      {
        memcpy((void *)table[pos].osrc, (void *)table[pos].src, (sizeof(u_int)+
        (sizeof((table[pos].osrc)->fd_array[0])*((table[pos].osrc)->fd_count)) ));
      }
    }
  }
  else if (rc < 0)
  {
    if (tvp) /* fudge. should return time remaining */
    {
      tvp->tv_sec = 0;
      tvp->tv_usec = 0;
    }
  }
  else /* if (rc == 0) */
  {
    for (pos = 0; pos < 3; pos++)
    {
      if (table[pos].osrc)
        (table[pos].osrc)->fd_count = 0;
    }
    if (tvp) /* timeout */
    {
      tvp->tv_sec = 0;
      tvp->tv_usec = 0;
    }
  }

  // free buffer and return.
  free(buf);
  return rc;
}

#undef  select
#define select(i,r,w,x,t) xfd_select(i,r,w,x,t)
#undef  fd_set
#define fd_set xfd_set
#endif

// ----------------------------------------------------------------------

int netio_select( int width, fd_set *r, fd_set *w, fd_set *x, struct timeval *tvp )
{
  // there are two possible points of failure for portable select():
  // 1. not all implementations return tvp with the time *left*
  // 2. some implementations trash the fd_sets on error

  return select( width, r, w, x, tvp );
}

// ----------------------------------------------------------------------

int netio_setsockopt( SOCKET sock, int cond_type, int parm )
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
      int type = ((which == 0)?(SO_RCVBUF):(SO_SNDBUF));
      int sz = 0;
      SOCKLEN_T szint = (SOCKLEN_T) sizeof(int);
      if (!(getsockopt(sock, SOL_SOCKET, type, (char *)&sz, &szint)<0))
      {
        if (sz < parm)
        {
          sz = parm;
          setsockopt(sock, SOL_SOCKET, type, (char *)&sz, szint);
        }
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
      if (ioctl(sock, FIONBIO, &flagon) == 0)
      // allow calls on blocking socket to preemptively multitask
      { if (!flagon) {flagon=1; ioctl(sock, FIOSLEEPTW, &flagon);} return 0;}
    #elif (CLIENT_OS == OS_OS2)
      int flagon = ((parm == 0 /* off */) ? (1): (0));
      return ioctl(sock, FIONBIO, (char *) &flagon, sizeof(flagon));
    #elif (CLIENT_OS == OS_AMIGAOS)
      char flagon = ((parm == 0 /* off */) ? (1): (0));
      return IoctlSocket(sock, FIONBIO, &flagon);
    #elif (CLIENT_OS == OS_DOS)
      return ((parm == 0 /* off */)?(0):(-1)); //always non-blocking
    #elif (CLIENT_OS == OS_MACOS)
      char flagon = ((parm == 0 /* off */) ? (1): (0));
      return ioctl(sock, FIONBIO, &flagon);
    #elif (defined(F_SETFL) && (defined(FNDELAY) || defined(O_NONBLOCK)))
    {
      int flag, res, arg;
      #if (defined(FNDELAY))
        flag = FNDELAY;
      #else
        flag = O_NONBLOCK;
      #endif
      arg = ((parm == 0 /* off */) ? (flag): (0) );

      if (( res = fcntl(sock, F_GETFL, flag /* usually unused */) ) == -1)
        return -1;
      if ((arg && (res&flag)) || (!arg && !(res&flag)))
        return 0;
      if ((res = fcntl(sock, F_SETFL, arg|(res&~flag) )) == -1)
        return -1;
      if (( res = fcntl(sock, F_GETFL, flag /* usually unused */) ) == -1)
        return -1;
      if ((arg && (res&flag)) || (!arg && !(res&flag)))
        return 0;
    }
    #endif
  }
  else if ( cond_type == CONDSOCK_ASYNCSIG )
  {
    #if !defined(USES_SIGIO)
    return -1;
    #elif defined(F_SETFL) && defined(O_ASYNC)
    int flag = O_ASYNC, res, arg = ((parm) ? (O_ASYNC): (0));
    if (( res = fcntl(sock, F_GETFL, flag /* usually unused */) ) == -1)
      return -1;
    if ((arg && (res&flag)) || (!arg && !(res&flag)))
      return 0;
    if ((res = fcntl(sock, F_SETFL, arg|(res&~flag) )) == -1)
      return -1;
    if (( res = fcntl(sock, F_GETFL, flag /* usually unused */) ) == -1)
      return -1;
    if ((arg && (res&flag)) || (!arg && !(res&flag)))
    {
      if (fcntl(sock, F_SETOWN, getpid() ) == 0)
        return 0;
      fcntl(sock, F_SETFL, (res^flag));
    }
    #elif defined(FIOASYNC)
    int onoff = ((parm)?(1):(0));
    return ioctl(sock, FIOASYNC, &onoff);
    #elif defined(S_INPUT) && defined(S_OUTPUT) && defined(S_MSG)
    int arg = S_INPUT|S_OUTPUT|S_MSG; //prep fd to post sigpoll or sigio
    if (!parm) arg = 0;                   //on any of these events
    return (ioctl(sock,I_SETSIG,(void *)arg)==-1 ? -1 : 0);
    #endif
  }
  else if ( cond_type == CONDSOCK_REUSEADDR )
  {
    int mytrue = (parm != 0 ? 1 : 0);
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char*)&mytrue, sizeof(int));
    return 0;
  }

  parm = parm; /* shaddup compiler */
  return -1;
}

// ----------------------------------------------------------------------

int netio_getsockopt( SOCKET sock, int cond_type, int *parm )
{
  if (cond_type == CONDSOCK_GETERROR)
  {
    SOCKLEN_T parmlen = (SOCKLEN_T) sizeof(int);
    if (getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*) parm, &parmlen) != 0)
      return -1;
    netio_seterrno(*parm);
    return 0;
  }
  else if ( cond_type == CONDSOCK_BLOCKMODE )
  {
    #if defined(F_GETFL) && (defined(FNDELAY) || defined(O_NONBLOCK))
    int flag, res;
    #if (defined(FNDELAY))
      flag = FNDELAY;
    #else
      flag = O_NONBLOCK;
    #endif
    if (( res = fcntl(sock, F_GETFL, flag ) ) != -1)
    {
      *parm = ((res & flag) == 0);
      return 0;
    }
    #endif
  }
  else if ( cond_type == CONDSOCK_ASYNCSIG )
  {
    #if !defined(USES_SIGIO)
    return -1;
    #elif defined(F_GETFL) && (defined(O_ASYNC))
    int flag, res;
    flag = O_ASYNC;
    if (( res = fcntl(sock, F_GETFL, flag /* usually unused */) ) != -1)
    {
      *parm = (res & flag);
      return 0;
    }
    #endif
  }

  parm = parm; /* shaddup compiler */
  return -1;
}

// ----------------------------------------------------------------------

