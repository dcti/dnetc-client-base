// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

const char *netio_cpp(void) {
return "@(#)$Id: netio.cpp,v 1.1.2.3 1999/04/18 00:38:30 jlawson Exp $"; }

#define __NETIO_CPP__ /* suppress redefinitions in netio.h */
#include "netio.h"

#ifdef PROXYTYPE
  #include "globals.h"
#else
  #include "logstuff.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <errno.h>

// ----------------------------------------------------------------------

static const char *netio_describe_error()
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
//  case WSAEDISCO: return "(WSAEDISCO): Disconnect";
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

// Creates a new network socket handle.
// Returns -1 on error, 0 on success.

int netio_createsocket(SOCKET &sock)
{
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
    //LowLevelSetSocketOption( CONDSOCK_SETMINBUFSIZE, 2048/* at least this */);
    //LowLevelSetSocketOption( CONDSOCK_BLOCKMODE, 1 ); /* really only needed for RISCOS */
    return 0; //success
  }
  #endif
  sock = INVALID_SOCKET;
  return -1;
#endif
}

// ----------------------------------------------------------------------

// Closes an opened socket handle.
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
    sock = INVALID_SOCKET;
    return rc;
  }
#else                                                  //BSD socks
  if ( sock != INVALID_SOCKET )
  {
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
    sock = INVALID_SOCKET;
    return (retcode);
  }
#endif
  return -1;
}

// ----------------------------------------------------------------------

// Performs a name-server lookup of the specified name and randomly
// returns one of the corresponding IP Addresses (if there is more
// than one matching address).
// Returns -1 on error, 0 on success.

int netio_resolve(const char *hosttgt, u32 &hostaddress)
{
  char host[128];
  unsigned int len = 0;
  const char *p = hosttgt;

  while (*p && isspace(*p))
    p++;
  while (*p && !isspace(*p) && len < sizeof(host))
    host[len++] = (char)*p++;
  host[len] = 0;

  if (len == 0)
  {
    hostaddress = 0;
    return -1;
  }

  hostaddress = (u32) inet_addr( host );
  if (hostaddress == 0) return -1;
  if (hostaddress == (u32) -1L)
  {
    struct hostent *hp;
    #if (CLIENT_OS == OS_NETWARE)
    struct nwsockent s_nwsockent;
    hp = NetDBgethostbyname( &s_nwsockent, host );
    #else
    hp = gethostbyname( host );
    #endif
    if (hp == NULL) { hostaddress = 0; return -1; }

    // randomly select one
    int addrcount;
    for (addrcount = 0; hp->h_addr_list[addrcount]; addrcount++) {};
    if (addrcount < 1) return -1;
    int index = rand() % addrcount;
    memcpy((void*) &hostaddress, hp->h_addr_list[index], sizeof(u32));
    if (hostaddress == 0) return -1;
    if (hostaddress == (u32) -1L) { hostaddress = 0; return -1; }
  }
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

// Creates a socket and initializes it as a listener to accept incoming
// TCP/IP connections on the specified interface and port number.
// Returns -1 on error, 0 on success.

#ifdef PROXYTYPE
int netio_openlisten(SOCKET &sock, u32 addr, u16 port)
{
  /* allocate the new socket */
  if (netio_createsocket(sock) < 0)
  {
#ifdef PROXYTYPE
    globalLogger->LogInfo(LOG_GENERAL | LOG_ERRSEVERE,
#else
    Log(
#endif
       "socket(): %s", netio_describe_error());
    return -1;
  }


  /* try to reuse sockets */
  netio_setsockopt(sock, CONDSOCK_REUSEADDR, 1);


  /* bind the socket to address and port given */
  struct sockaddr_in sin;
  memset(&sin, 0, sizeof(sin));
  sin.sin_addr.s_addr = addr;
  sin.sin_port = htons(port);
  sin.sin_family = PF_INET;

  if (bind(sock, (struct sockaddr *)&sin, sizeof(sin)) < 0)
  {
#ifdef PROXYTYPE
    globalLogger->LogInfo(LOG_GENERAL | LOG_ERRSEVERE,
#else
    Log(
#endif
                   "bind(%s:%u): %s ", netio_ntoa( addr ),
                   (unsigned int)port, netio_describe_error());
    netio_close(sock);
    return -1;
  }

  /* set up to receive connections on this socket */
  if (listen(sock, 5) < 0)
  {
#ifdef PROXYTYPE
    globalLogger->LogInfo(LOG_GENERAL | LOG_ERRSEVERE,
#else
    Log(
#endif
        "listen(): %s", netio_describe_error());
    netio_close(sock);
    return -1;
  }

  return 0;
}
#endif

// ----------------------------------------------------------------------

// Creates a new socket and attempts to connect it to the specified
// target host address and port.  If a listenaddr is specified, then
// the outgoing connection will be created such that it originates
// from the specified local interface.
// Returns -1 on error, 0 on success.

int netio_connect(SOCKET &sock, char *host, u16 port, u32 &addr, u32 listenaddr)
{
  // resolve the address we will connect to.
  if ((host && netio_resolve(host, addr) < 0) || !addr)
  {
#ifdef PROXYTYPE
    globalLogger->LogInfo(LOG_GENERAL | LOG_ERRSEVERE,
#else
    Log(
#endif
        "Error resolving host %s", (host ? host : "") );
    sock = INVALID_SOCKET;
    return -1;
  }

  // allocate a new socket.
  if (netio_createsocket(sock) < 0)
  {
#ifdef PROXYTYPE
    globalLogger->LogInfo(LOG_GENERAL | LOG_ERRSEVERE,
#else
    Log(
#endif
        "socket(): %s", netio_describe_error());
    return -1;
  }


  // bind the socket to address and port given.
  if (listenaddr)
  {
    struct sockaddr_in sin;
    memset(&sin, 0, sizeof(sin));
    sin.sin_addr.s_addr = listenaddr;
    sin.sin_port = htons(0);
    sin.sin_family = PF_INET;

    if (bind(sock, (struct sockaddr *)&sin, sizeof(sin)) < 0)
    {
#ifdef PROXYTYPE
      globalLogger->LogInfo(LOG_GENERAL | LOG_ERRSEVERE,
#else
      Log(
#endif
         "bind(%s:%u): %s", netio_ntoa( listenaddr ),
         (unsigned int)port, netio_describe_error());
      netio_close(sock);
      return -1;
    }
  }


  // connect to the desination host.
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
      sin->sin_port = htons( that_port );
      rc = t_connect( sock, sndcall, NULL);
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
  // timeout for this call must be >0 to not have default used
  //socket_set_conn_timeout(sock, iotimeout);

  return(connect(sock, (struct sockaddr *)&sin, sizeof(sin)));
    
#elif defined(AF_INET) //BSD sox

  struct sockaddr_in sin;
  memset(&sin, 0, sizeof(sin));
  sin.sin_addr.s_addr = addr;
  sin.sin_port = htons(port);
  sin.sin_family = PF_INET;

  if (connect(sock, (struct sockaddr *)&sin, sizeof(sin)) < 0)
  {
#if (CLIENT_OS == OS_WIN32)
    int myerrno = WSAGetLastError();
#elif (CLIENT_OS == OS_OS2)
    int myerrno = sock_errno();
#else
    int myerrno = errno;
#endif    
    if (myerrno != EINPROGRESS)
    {
#ifdef PROXYTYPE
      globalLogger->LogInfo(LOG_GENERAL | LOG_ERRSEVERE,
#else
      Log(
#endif
          "connect(): %s", netio_describe_error());
      netio_close(sock);
      return -1;
    }
  }
  return 0;
#else //no socket support
  return -1;
#endif
}

// ----------------------------------------------------------------------

// Waits for a new incoming connection to be detected on the specified
// listener socket and returns a new socket handle that corresponds
// with the newly connected client.  The IP Address of the newly
// connected client is also made available.
// Returns -1 on error, 0 on success.

#ifdef PROXYTYPE
int netio_accept(SOCKET s, SOCKET &snew, u32 &hostaddress)
{
  struct sockaddr_in Sin;

  //both bsd <sys/socket.h> and the posix version say addrlen is an int *.
#if (CLIENT_OS == OS_LINUX)
  #ifndef _SOCKETBITS_H
  int sinlen;
  #else
  socklen_t sinlen;
  #endif
#elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_OS2)
  int sinlen;
#else
  unsigned sinlen;
#endif
  sinlen = sizeof(Sin);
  snew = accept(s, (struct sockaddr *)&Sin, &sinlen);
  if ((int) snew < 0)
  {
    return -1;
  }

  if (ntohs(Sin.sin_port) < 1024)
  {
    globalLogger->LogInfo(LOG_GENERAL | LOG_ERRSEVERE,
        "Rejected connection from %s:%i",
        netio_ntoa(Sin.sin_addr.s_addr), ntohs(Sin.sin_port));
    netio_close(snew);
    return -1;
  }      
  hostaddress = Sin.sin_addr.s_addr;
  return 0;
}
#endif

// ----------------------------------------------------------------------

// Translates a numeric IP Address into a printable ASCII text-string.
// Returns -1 on error, 0 on success.

const char *netio_ntoa(u32 hostaddr)
{
#if 1
  // it's apparently better to reinvent the wheel.
  static char buff[18];
  char *p = (char *)(&hostaddr);
  sprintf( buff, "%d.%d.%d.%d", (p[0]&255),(p[1]&255),(p[2]&255),(p[3]&255) );
  return buff;
#else
  return inet_ntoa(*(in_addr*)&hostaddr);  
#endif
}

// ----------------------------------------------------------------------

int netio_recv(SOCKET s, void *data, int len)
{
#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_NETWARE)
  return recv(s, (char*)data, len, 0);
#else
  return read(s, (char*)data, len);
#endif
}

// ----------------------------------------------------------------------

int netio_send(SOCKET s, void *data, int len)
{
#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_NETWARE)
  return send(s, (char*)data, len, 0);
#else
  return write(s, (char*)data, len);
#endif
}

// ----------------------------------------------------------------------

#if defined(XFD_SETSIZE)
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

  usleep(100);

  table[0].osrc = or;
  table[1].osrc = ow;
  table[2].osrc = ox;

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


  buf = (char *)malloc( bufsize );
  if (!buf)
  {
    #if (CLIENT_OS == OS_WIN32)  /* others will have ENOMEM set */
    WSASetLastError(WSAENOBUFS);
    #endif
    return -1; 
  }
  
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

  rc = 0;
  while (rc == 0)
  {
    u_int i;
    
    for (i = 0; i < biggest_xfdset; i += FD_SETSIZE)
    {
      width = 0;

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
    
      if ( width )
      {
        struct timeval tv = { 0, 0 };
        width = select( width, table[0].fdsetp, table[1].fdsetp, table[2].fdsetp, &tv );
      }

      if ( width < 0 )
      {
        #if (CLIENT_OS == OS_WIN32)
        if (WSAGetLastError() == WSAEINTR)
        #else
        if (errno == EINTR) 
        #endif
          width = 0; /* ignore it */
        else
        {
          rc = -1;
          break;
        }
      }

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

  if (idle_count == 0)
    usleep(100);

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
      int sz = 0, szint = (int)sizeof(int);
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
  else if ( cond_type == CONDSOCK_REUSEADDR )
  {
    int mytrue = (parm != 0 ? 1 : 0);
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char*)&mytrue, sizeof(int));
  }

  parm = parm; /* shaddup compiler */
  return -1;
}

// ----------------------------------------------------------------------


