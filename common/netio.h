// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

#ifndef __NETIO_H__
#define __NETIO_H__ "@(#)$Id: netio.h,v 1.3 2000/06/20 08:15:49 jlawson Exp $"

#include "cputypes.h"

#if (CLIENT_OS == OS_WIN32)
  #include <winsock2.h>
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
  #else
    #include <types.h>
  #endif
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <sys/select.h>
  #include <sys/ioctl.h>
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


/* Socket modes that can be adjusted with the netio_setsockopt
 * function, or retrieved with the netio_getsockopt function. */

#define CONDSOCK_BLOCKMODE        1
#define CONDSOCK_KEEPALIVE        2
#define CONDSOCK_SETMINBUFSIZE    3
#define CONDSOCK_REUSEADDR        4
#define CONDSOCK_GETERROR         5
#define CONDSOCK_ASYNCSIG         6


/* Symbolic constant to represent uninitialized/closed sockets */

#ifndef INVALID_SOCKET
#define INVALID_SOCKET  ((SOCKET)(-1))
#endif


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
extern int netio_lrecv(SOCKET sock, void *data, int toread);
extern int netio_recv(SOCKET s, void *data, int length,
               int iotimeout, int (*exitcheckfn)(void) = NULL );
extern int netio_lsend(SOCKET s, const void *ccdata, int towrite);
extern int netio_send(SOCKET s, const void *ccdata, int length,
               int iotimeout, int (*exitcheckfn)(void) = NULL );
extern int netio_select(int width, fd_set *, fd_set *, fd_set *,
    struct timeval *);
extern int netio_setsockopt( SOCKET sock, int cond_type, int parm );
extern int netio_getsockopt( SOCKET sock, int cond_type, int *parm );

#endif /* NETIO_H */

