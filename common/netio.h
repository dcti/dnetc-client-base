// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

#ifndef __NETIO_H__
#define __NETIO_H__ "@(#)$Id: netio.h,v 1.5 2000/07/05 21:10:19 mfeiri Exp $"

#include "cputypes.h"

// ------------------

#ifdef UNSAFEHEADERS
  // Some environments include old system headers that are not safe for
  // direct inclusion within C++ programs and need to be explicitly
  // wrapped with extern.  However, this should not be unconditionally
  // done for all platforms, since some platform headers intentionally
  // try to prototype C++ versions of functions
  extern "C" {
#endif

// ------------------

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
  #define WIN32_LEAN_AND_MEAN
  #ifndef STRICT
    #define STRICT
  #endif
  #include <windows.h>
  #if (CLIENT_OS == OS_WIN32)
    #include <winsock.h>
  #else
    #include "w32sock.h" //winsock win16 wrappers
  #endif
  #include <io.h>
  #include <errno.h>
  #define EINPROGRESS WSAEWOULDBLOCK
  #define ECONNRESET  WSAECONNRESET
#elif (CLIENT_OS == OS_OS2)
  #define BSD_SELECT
  #include <fcntl.h>
  #include <netdb.h>
  #include <netinet/in.h>
  #include <errno.h>
  #if defined(__EMX__)
    #include <sys/types.h>
    #include <io.h>
  #else
    #include <types.h>
  #endif
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <sys/select.h>
  #include <sys/ioctl.h>
  typedef int SOCKET;
#elif (CLIENT_OS == OS_RISCOS)
  #include <socklib.h>
  #include <inetlib.h>
  #include <unixlib.h>
  #include <sys/ioctl.h>
  #include <unistd.h>
  #include <netdb.h>
  typedef int SOCKET;
#elif (CLIENT_OS == OS_DOS)
  //ntohl()/htonl() defines are in...
  #include "platforms/dos/clidos.h"
#elif (CLIENT_OS == OS_VMS)
  #include <signal.h>
  #ifdef __VMS_UCX__
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <sys/time.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <netdb.h>
    #include <unixio.h>
  #elif defined(MULTINET)
    #include "multinet_root:[multinet.include.sys]types.h"
    #include "multinet_root:[multinet.include.sys]ioctl.h"
    #include "multinet_root:[multinet.include.sys]param.h"
    #include "multinet_root:[multinet.include.sys]time.h"
    #include "multinet_root:[multinet.include.sys]socket.h"
    #include "multinet_root:[multinet.include]netdb.h"
    #include "multinet_root:[multinet.include.netinet]in.h"
    #include "multinet_root:[multinet.include.netinet]in_systm.h"
    #ifndef multinet_inet_addr
      extern "C" unsigned long int inet_addr(const char *cp);
    #endif
    #ifndef multinet_inet_ntoa
      extern "C" char *inet_ntoa(struct in_addr in);
    #endif
  #endif
  typedef int SOCKET;
#elif (CLIENT_OS == OS_AMIGAOS)
  #include "platforms/amiga/amiga.h"
  #include <assert.h>
  #include <clib/socket_protos.h>
  #include <pragmas/socket_pragmas.h>
  #include <sys/ioctl.h>
  #include <sys/time.h>
  #include <netdb.h>
  extern struct Library *SocketBase;
  #define inet_ntoa(addr) Inet_NtoA(addr.s_addr)
  #ifndef __PPC__
     #define inet_addr(host) inet_addr((unsigned char *)host)
     #define gethostbyname(host) gethostbyname((unsigned char *)host)
  #endif
  typedef int SOCKET;
#elif (CLIENT_OS == OS_BEOS)
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <sys/time.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <netdb.h>
  #include <ctype.h>
  typedef int SOCKET;
#else
  #include <errno.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <netdb.h>
  #include <netinet/in.h>
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <arpa/inet.h>
  typedef int SOCKET;
#endif

// ------------------

#ifdef UNSAFEHEADERS
} // End the extern needed to handle unsafe system headers.
#endif

////////////////////////////////////////////////////////////////////////////

// Use sigio handling on systems that provide it, since it offers the
// potential for lower-overhead when polling a large number of very
// active sockets.  However on the keymaster, sigio handling is
// usually a disadvantage because it receives a smaller number of
// larger sized data packets from its incoming fullserver connections.
// This method should be possible on FreeBSD, NetWare, Solaris and
// maybe others.  However, FreeBSD's poll() is apparently buggy in
// many situations, leading to unexpected stalling.

#if defined(ENABLE_SIGIO)
  #define USES_SIGIO
  #include <poll.h>
  #ifndef POLLRDNORM         // usually POLLIN is a union or RDNORM and
  #define POLLRDNORM POLLIN  // RDBAND (rdband being active only if fcntl
  #define POLLRDBAND POLLIN  // has set it so)
  #endif                     // some implementations do not differentiate
  #ifndef POLLWRNORM         // so catch that here. We don't care about
  #define POLLWRNORM POLLOUT // out-of-band data anyway.
  #define POLLWRBAND POLLOUT
  #endif
  #include <sys/ioctl.h>
  #include <signal.h>
  #undef FD_SETSIZE            /* we don't care because we don't use it */
  #define FD_SETSIZE INT_MAX
#else
  #undef USES_SIGIO
#endif

////////////////////////////////////////////////////////////////////////////

/* Provide extended select() operations for platforms with
 * significantly impaired fd_set counts (ex: NetWare 3.1) */

#define PROXY_MIN_FD_SETSIZE 256

#if defined(PROXYTYPE) && \
      (FD_SETSIZE < PROXY_MIN_FD_SETSIZE) && (CLIENT_OS == OS_NETWARE)
  #define XFD_SETSIZE PROXY_MIN_FD_SETSIZE

  typedef struct xfd_set { u_int fd_count;
                           SOCKET fd_array[XFD_SETSIZE]; } xfd_set;
  extern void XFD_CLR(SOCKET fd, xfd_set *set);
  extern void XFD_ZERO(xfd_set *set);
  extern int  XFD_ISSET(SOCKET fd, xfd_set *set);
  extern void XFD_SET(SOCKET fd, xfd_set *set);
  extern int xfd_select( int w, xfd_set *, xfd_set *, xfd_set *, struct timeval * );

  #ifndef __NETIO_CPP__
    #undef  FD_SETSIZE
    #define FD_SETSIZE        XFD_SETSIZE
    #undef  FD_SET
    #define FD_SET(fd,fds)    XFD_SET(fd,fds)
    #undef  FD_ISSET
    #define FD_ISSET(fd,fds)  XFD_ISSET(fd,fds)
    #undef  FD_CLR
    #define FD_CLR(fd,fds)    XFD_CLEAR(fd,fds)
    #undef  FD_ZERO
    #define FD_ZERO(fds)      XFD_ZERO(fds)
    #undef  fd_set
    #define fd_set xfd_set
    #undef  select
    #define select(i,r,w,x,t) xfd_select(i,r,w,x,t)
  #endif
#else
  #undef XFD_SETSIZE
#endif

////////////////////////////////////////////////////////////////////////////

/* Socket modes that can be adjusted with the netio_setsockopt
 * function, or retrieved with the netio_getsockopt function. */

#define CONDSOCK_BLOCKMODE        1
#define CONDSOCK_KEEPALIVE        2
#define CONDSOCK_SETMINBUFSIZE    3
#define CONDSOCK_REUSEADDR        4
#define CONDSOCK_GETERROR         5
#define CONDSOCK_ASYNCSIG         6

////////////////////////////////////////////////////////////////////////////

/* Symbolic constant to represent uninitialized/closed sockets */

#ifndef INVALID_SOCKET
#define INVALID_SOCKET  ((SOCKET)(-1))
#endif

////////////////////////////////////////////////////////////////////////////

/* Functions exported by netio.cpp */

extern const char *netio_describe_error(void);
extern int netio_geterrno(void);
extern int netio_seterrno(int err);
extern const char *netio_ntoa(u32 addr);
extern unsigned int netio_getopensocketcount(void);
extern int netio_createsocket(SOCKET &sock, bool nonblocking = false);
extern int netio_close(SOCKET &sock);
extern int netio_resolve(const char *hosttgt, u32 &hostaddress);
extern int netio_gethostname( char *buffer, unsigned int buflen );
extern int netio_gethostaddr( u32 * addr );
extern int netio_bind( SOCKET sock, u32 addr, u16 port );
extern int netio_listen( SOCKET sock, int backlog );
extern int netio_openlisten(SOCKET &sock, u32 addr, u16 port,
    bool nonblocking = false);

extern int netio_lconnect( SOCKET sock, u32 addr, u16 port );
extern int netio_connect(SOCKET &sock, const char *host,
    u16 port, u32 &addr, u32 listenaddr, bool nonblocking = false);

extern int netio_laccept(SOCKET sock, SOCKET *thatsock,
    u32 *thataddr, int *thatport);
extern int netio_accept(SOCKET sock, SOCKET *snew,
    u32 *hostaddress, int *hostport, bool nonblocking = false);

extern int netio_lrecv(SOCKET sock, void *data, int toread,
                      bool doprecheck = true);
extern int netio_recv(SOCKET s, void *data, int length,
               int iotimeout, int (*exitcheckfn)(void) = NULL );

extern int netio_lsend(SOCKET s, const void *ccdata, int towrite,
                       bool doprecheck = true);
extern int netio_send(SOCKET s, const void *ccdata, int length,
               int iotimeout, int (*exitcheckfn)(void) = NULL );

extern int netio_select(int width, fd_set *, fd_set *, fd_set *,
    struct timeval *);

extern int netio_setsockopt( SOCKET sock, int cond_type, int parm );
extern int netio_getsockopt( SOCKET sock, int cond_type, int *parm );

////////////////////////////////////////////////////////////////////////////

#endif /* NETIO_H */

