// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: network.h,v $
// Revision 1.25  1998/07/07 21:55:48  cyruspatel
// Serious house cleaning - client.h has been split into client.h (Client
// class, FileEntry struct etc - but nothing that depends on anything) and
// baseincs.h (inclusion of generic, also platform-specific, header files).
// The catchall '#include "client.h"' has been removed where appropriate and
// replaced with correct dependancies. cvs Ids have been encapsulated in
// functions which are later called from cliident.cpp. Corrected other
// compile-time warnings where I caught them. Removed obsolete timer and
// display code previously def'd out with #if NEW_STATS_AND_LOGMSG_STUFF.
// Made MailMessage in the client class a static object (in client.cpp) in
// anticipation of global log functions.
//
// Revision 1.24  1998/06/29 08:01:13  ziggyb
// DOD defines
//
// Revision 1.23  1998/06/29 06:58:10  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.22  1998/06/26 09:19:40  jlawson
// removed inclusion of dos.h for win32
//
// Revision 1.21  1998/06/26 07:13:56  daa
// move strcmpi and strncmpi defs to cmpidefs.h
//
// Revision 1.20  1998/06/26 06:48:51  daa
// add macro defination for strncmpi
//
// Revision 1.19  1998/06/22 01:05:01  cyruspatel
// DOS changes. Fixes various compile-time errors: removed extraneous ')' in
// sleepdef.h, resolved htonl()/ntohl() conflict with same def in client.h
// (is now inline asm), added NONETWORK wrapper around Network::Resolve()
//
// Revision 1.18  1998/06/15 09:12:54  jlawson
// moved more sleep defines into sleepdef.h
//
// Revision 1.17  1998/06/15 08:28:39  jlawson
// moved win32 sleep macros out of network.h
//
// Revision 1.16  1998/06/14 13:07:23  ziggyb
// Took out all OS/2 DOD stuff, being moved to platforms\os2cli\dod.h
//
// Revision 1.15  1998/06/14 11:24:14  ziggyb
// Added os2defs.h and adjusted for the sleep defines. Now compile without
// errors. Woohoo!
//
// Revision 1.14  1998/06/14 10:14:36  ziggyb
// There are ^M's everywhere, got rid of them and some OS/2 header changes
//
// Revision 1.13  1998/06/14 08:13:02  friedbait
// 'Log' keywords added to maintain automatic change history
//
// Revision 1.12 1998/06/13 23:33:20  cyruspatel 
// Fixed NetWare stuff and added #include "sleepdef.h" (which should now 
// warn if macros are not the same)
//

//#define NONETWORK         // define to eliminate network functionality
//#define SELECT_FIRST      // define to perform select() before reading
//#define __VMS_UCX__       // define for UCX instead of Multinet on VMS

///////////////////////////////////////////////////////////////////////////

#ifndef NETWORK_H
#define NETWORK_H

#define NETTIMEOUT (60)


#include "cputypes.h"
#include "autobuff.h"

#if ((CLIENT_OS == OS_AMIGAOS)|| (CLIENT_OS == OS_RISCOS))
extern "C" {
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if ((CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_RISCOS))
}
#endif

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
  #include <winsock.h>
  #include <io.h>
  #define write(sock, buff, len) send(sock, (char*)buff, len, 0)
  #define read(sock, buff, len) recv(sock, (char*)buff, len, 0)
  #define close(sock) closesocket(sock);
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
  //generally NONETWORK, but to be safe we...
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
  #include <gusi.h>
  typedef int SOCKET;
#elif (CLIENT_OS == OS_OS2)
  #include <process.h>
  #include <io.h>

  #if defined(__WATCOMC__)
    #include <i86.h>
  #endif
  // All the OS/2 specific headers are here
  // This is nessessary since the order of the OS/2 defines are important
  #include "platforms\os2cli\os2defs.h"
  typedef int SOCKET;
  #define close(s) soclose(s)
  #define read(sock, buff, len) recv(sock, (char*)buff, len, 0)
  #define write(sock, buff, len) send(sock, (char*)buff, len, 0)
#elif (CLIENT_OS == OS_AMIGAOS)
  extern "C" {
  #include "platforms/amiga.h"
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
  typedef int SOCKET;
  #define write(sock, buff, len) send(sock, (unsigned char*)buff, len, 0)
  #define read(sock, buff, len) recv(sock, (unsigned char*)buff, len, 0)
  #define close(sock) closesocket(sock)
#else
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
  #endif
  #if (CLIENT_OS == OS_AIX)
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
  #if (CLIENT_OS == OS_NETWARE)
    #define read(sock, buff, len) recv(sock, (char*)buff, len, 0)
    #define write(sock, buff, len) send(sock, (char*)buff, len, 0)
    #ifdef gethostbyname     //this is in "hbyname.cpp"
    #undef gethostbyname
    #endif
    extern struct hostent *gethostbyname( char *hostname );
    #ifdef inet_ntoa         //this is also in "hbyname.cpp"
    #undef inet_ntoa
    #endif
    extern char *inet_ntoa( struct in_addr addr );
    #ifdef inet_addr         //this is also in "hbyname.cpp"
    #undef inet_addr         //also bad proto in netware sdk13
    #endif                   
    extern long my_inet_addr( char *aaddr );
    #define inet_addr(x) my_inet_addr(x)
  #endif
#endif

///////////////////////////////////////////////////////////////////////////

#define MODE_UUE        1
#define MODE_HTTP       2
#define MODE_SOCKS4     4
#define MODE_PROXIED    (MODE_HTTP | MODE_SOCKS4 | MODE_SOCKS5)
#define MODE_SOCKS5     8
#define DEFAULT_RRDNS   "us.v27.distributed.net"
#define DEFAULT_PORT    2064

extern void NetworkInitialize(void);
extern void NetworkDeinitialize(void);

///////////////////////////////////////////////////////////////////////////

class Network
{
public:
  s32 quietmode;
protected:
  char server_name[64];
  char rrdns_name[64];
  s16  port;
  s32  mode, startmode;
  SOCKET sock;          // socket file handle
  s32  retries;         // 3 retries, then do RRDNS lookup
  u32  lastaddress;
  s16  lastport;

  // http related
  char httpproxy[64];   // also used for socks4
  s16 httpport;         // also used for socks4
  char httpid[128];     // also used for socks4
  u32 lasthttpaddress;

  // communications and decoding buffers
  AutoBuffer netbuffer, uubuffer;
  bool gotuubegin, gothttpend, puthttpdone, gethttpdone;
  u32 httplength;
  char logstr[1024];

  friend void MakeNonBlocking(SOCKET sock, bool nonblocking = true);

  s32 LowLevelGet(u32 length, char *data);
    // Returns length of read buffer
    //    or 0 if connection closed
    //    or -1 if no data waiting

  s32 LowLevelPut(u32 length, const char *data);
    // Returns length of sent buffer
    //    or -1 on error

public:
  Network( const char * preferred, const char * roundrobin, s16 port);
    // constructor: If preferred or roundrobin are NULL, use DEFAULT_RRDNS.
    // They can be IP's and not necessarily names, though a name is better
    // suited for roundrobin use.
    // if port is 0, use DEFAULT_PORT

  ~Network( void );
    // guess

  void SetModeUUE( bool enabled );
    // enables or disable uuencoding of the data stream

  void SetModeHTTP( const char *httpproxyin = NULL, s16 httpportin = 80,
      const char * httpidin = NULL );
    // enables http-proxy tunnelling transmission.
    // Specifying NULL for httpproxy will disable.

  void SetModeSOCKS4(const char *sockshost = NULL, s16 socksport = 1080,
      const char * socksusername = NULL );
    // enables socks4-proxy tunnelling transmission.
    // Specifying NULL for sockshost will disable.

  static char * base64_encode(char *username, char *password);
    // encodes the username & password pair

  void SetModeSOCKS5(const char *sockshost = NULL, s16 socksport = 1080,
      const char * socksusernamepw = NULL );
    // enables socks5-proxy tunnelling transmission.
    // Specifying NULL for sockshost will disable.
    // usernamepw is username:pw
    // An empty or NULL usernamepw means use only no authentication method

  s32  Open( SOCKET insock );
    // takes over a preconnected socket to a client.
    // returns -1 on error, 0 on success

  s32  Open( void );
    // [re]open the connection using the current settings.
    // returns -1 on error, 0 on success

  s32  Close( void );
    // close the connection

  static s32 Resolve( const char *host, u32 &hostaddress );
    // perform a DNS lookup, handling random selection of DNS lists
    // returns -1 on error, 0 on success

  s32  Get( u32 length, char * data, u32 timeout = 10 );
    // recv data over the open connection, uue/http based on ( mode & MODE_UUE ) etc.
    // Returns length of read buffer.

  s32  Put( u32 length, const char * data );
    // send data over the open connection, uue/http based on ( mode & MODE_UUE ) etc.
    // returns -1 on error, or 0 on success

  void MakeBlocking();
    // makes the socket operate in blocking mode.

  void LogScreen( const char * txt) const;
};

///////////////////////////////////////////////////////////////////////////

#endif //NETWORK_H
