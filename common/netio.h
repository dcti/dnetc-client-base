// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

#ifndef __NETIO_H__
#define __NETIO_H__ "@(#)$Id: netio.h,v 1.1.2.3 1999/04/22 09:15:36 jlawson Exp $"

#include "cputypes.h"

#if (CLIENT_OS == OS_WIN32)
  #include <winsock.h>
  #define EINPROGRESS WSAEWOULDBLOCK
  #define ECONNRESET  WSAECONNRESET
#elif (CLIENT_OS == OS_OS2)
  #define BSD_SELECT
  #include <fcntl.h>
  #include <netdb.h>
  #include <netinet/in.h>
  #if defined(__EMX__)
    #include <sys/types.h>
  #else
    #include <types.h>
  #endif
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <sys/select.h>
  #include <sys/ioctl.h>
  #define closesocket(sock) soclose(sock)
  typedef int SOCKET;
#else
  #include <unistd.h>
  #include <fcntl.h>
  #include <netdb.h>
  #include <netinet/in.h>
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <arpa/inet.h>
  #define closesocket(sock) close(sock)
  typedef int SOCKET;
#endif

#if (FD_SETSIZE < 512) && defined(PROXYTYPE)

  #define XFD_SETSIZE 512 /* **** (512*4)+4 per fd_set == 2052 bytes **** */

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
#endif /* FD_SETSIZE < 512 */


#define CONDSOCK_BLOCKMODE        1
#define CONDSOCK_KEEPALIVE        2   
#define CONDSOCK_SETMINBUFSIZE    3
#define CONDSOCK_REUSEADDR        4


#ifndef INVALID_SOCKET
#define INVALID_SOCKET  ((SOCKET)(-1))
#endif


/* netio.cpp */

extern int netio_createsocket(SOCKET &sock);
extern int netio_close(SOCKET &sock);
extern int netio_resolve(const char *hosttgt, u32 &hostaddress);
extern int netio_openlisten(SOCKET &sock, u32 addr, u16 port);
extern int netio_connect(SOCKET &sock, const char *host, u16 port, u32 &addr, u32 listenaddr);
extern int netio_accept(SOCKET sock, SOCKET &snew, u32 &hostaddress);
extern const char *netio_ntoa(u32 hostaddr);
extern int netio_recv(SOCKET sock, void *data, int len);
extern int netio_send(SOCKET sock, const void *data, int len);
extern int netio_select(int width, fd_set *, fd_set *, fd_set *, struct timeval *);
extern int netio_gethostname( char *buffer, unsigned int len );
extern int netio_gethostaddr( u32 *addr );
extern int netio_setsockopt( SOCKET sock, int cond_type, int parm );

#endif /* NETIO_H */

