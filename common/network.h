// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// $Log: network.h,v $
// Revision 1.40.2.4  1999/01/04 02:16:26  remi
// Synced with :
//
//  Revision 1.47  1999/01/02 07:18:23  dicamillo
//  Add ctype.h for BeOS.
//
//  Revision 1.46  1999/01/01 02:45:16  cramer
//  Part 1 of 1999 Copyright updates...
//
// Revision 1.40.2.3  1998/12/28 15:49:10  remi
// Synced with :
//  Revision 1.44  1998/12/24 05:19:55  dicamillo
//  Add socket_ioctl to Mac OS definitions.
//
//  Revision 1.43  1998/12/22 15:58:24  jcmichot
//  QNX port.
//
//  Revision 1.42  1998/12/21 17:54:23  cyp
//  (a) Network connect is now non-blocking. (b) timeout param moved from
//  network::Get() to object scope.
//
//  Revision 1.41  1998/12/08 05:57:03  dicamillo
//  Add defines for MacOS.
//
// Revision 1.40.2.2  1998/11/08 11:51:36  remi
// Lots of $Log tags.
//
// Sychronized with official 1.40

#ifndef NETWORK_H
#define NETWORK_H

#include "cputypes.h"
//#include "autobuff.h"


#if ((CLIENT_OS == OS_AMIGAOS)|| (CLIENT_OS == OS_RISCOS))
extern "C" {
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>    // for offsetof
#include <time.h>
#include <errno.h>     // for errno and EINTR

#if ((CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_RISCOS))
}
#endif

#if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
  #define WIN32_LEAN_AND_MEAN
  #define STRICT
  #include <windows.h>
  #if (CLIENT_OS == OS_WIN32)
   #include <winsock.h>
  #else
  #include "w32sock.h" //winsock wrappers
  #endif
  #include <io.h>
  #define write(sock, buff, len) send(sock, (char*)buff, (int)len, 0)
  #define read(sock, buff, len) recv(sock, (char*)buff, (int)len, 0)
  #define close(sock) closesocket(sock)
#elif (CLIENT_OS == OS_RISCOS)
  extern "C" {
  #include <socklib.h>
  #include <inetlib.h>
  #include <unixlib.h>
  #include <sys/ioctl.h>
  #include <unistd.h>
  #include <netdb.h>
  #define SOCKET int
  }
#elif (CLIENT_OS == OS_DOS) 
  //generally NO!NETWORK, but to be safe we...
  #include "platform/dos/clidos.h" 
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
    #define write(sock, buff, len) socket_write(sock, buff, len)
    #define read(sock, buff, len) socket_read(sock, buff, len)
    #define close(sock) socket_close(sock)
  #endif
  typedef int SOCKET;
#elif (CLIENT_OS == OS_MACOS)
  #include "socket_glue.h"
  #define write(sock, buff, len) socket_write(sock, buff, len)
  #define read(sock, buff, len) socket_read(sock, buff, len)
  #define close(sock) socket_close(sock)
  #define ioctl(sock, request, arg) socket_ioctl(sock, request, arg)
  extern Boolean myNetInit(void);
#elif (CLIENT_OS == OS_OS2)
  #include <process.h>
  #include <io.h>

  #if defined(__WATCOMC__)
    #include <i86.h>
  #endif
  // All the OS/2 specific headers are here
  // This is nessessary since the order of the OS/2 defines are important
  #include "platforms/os2cli/os2defs.h"
  typedef int SOCKET;
  #define close(s) soclose(s)
  #define read(sock, buff, len) recv(sock, (char*)buff, len, 0)
  #define write(sock, buff, len) send(sock, (char*)buff, len, 0)
#elif (CLIENT_OS == OS_AMIGAOS)
  extern "C" {
  #include "platforms/amiga/amiga.h"
  #include <assert.h>
  #include <clib/socket_protos.h>
  #include <pragmas/socket_pragmas.h>
  #include <sys/ioctl.h>
  #include <sys/time.h>
  #include <netdb.h>
  extern struct Library *SocketBase;
  #define write(sock, buff, len) send(sock, (unsigned char*)buff, len, 0)
  #define read(sock, buff, len) recv(sock, (unsigned char*)buff, len, 0)
  #define close(sock) CloseSocket(sock)
  #define inet_ntoa(addr) Inet_NtoA(addr.s_addr)
  #ifndef __PPC__
     #define inet_addr(host) inet_addr((unsigned char *)host)
     #define gethostbyname(host) gethostbyname((unsigned char *)host)
  #endif
  typedef int SOCKET;
  }
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
  #define write(sock, buff, len) send(sock, (unsigned char*)buff, len, 0)
  #define read(sock, buff, len) recv(sock, (unsigned char*)buff, len, 0)
  #define close(sock) closesocket(sock)
#else

#if (CLIENT_OS == OS_QNX)
  #include <sys/select.h>
#endif
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <sys/time.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <netdb.h>
  typedef int SOCKET;
  #if (CLIENT_OS == OS_LINUX) && (CLIENT_CPU == CPU_ALPHA)
    #include <asm/byteorder.h>
  #elif (CLIENT_OS == OS_DYNIX) && defined(NTOHL)
    #define ntohl(x)  NTOHL(x)
    #define htonl(x)  HTONL(x)
    #define ntohs(x)  NTOHS(x)
    #define htons(x)  HTONS(x)
  #endif
  #if (CLIENT_OS == OS_AIX) || (CLIENT_OS == OS_DYNIX)
    #include <errno.h>
  #endif
  #if ((CLIENT_OS == OS_SUNOS) && (CLIENT_CPU==CPU_68K))
    #if defined(_SUNOS3_)
      #define _SOCKET_H_ALREADY_
      extern "C" int fcntl(int, int, int);
    #endif
    extern "C" {
    int socket(int, int, int);
    int setsockopt(int, int, int, char *, int);
    int connect(int, struct sockaddr *, int);
    }
  #endif
  #if (CLIENT_OS == OS_ULTRIX)
    extern "C" {
      int socket(int, int, int);
      int setsockopt(int, int, int, char *, int);
      int connect(int, struct sockaddr *, int);
    }
  #endif
  #if (CLIENT_OS == OS_NETWARE)
    #include "platforms/netware/netware.h" //symbol redefinitions
    extern "C" {
    #pragma pack(1)
    #include <tiuser.h> //using TLI
    #pragma pack()
    }
  #endif  
#endif

#endif //NETWORK_H
