// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: network.h,v $
// Revision 1.50  1999/01/08 02:57:37  michmarc
// Wrapper around #define STRICT to avoid a _HUGE_ pile of warnings
// under VC6/AlphaNT
//
// Revision 1.49  1999/01/05 22:44:34  cyp
// Resolve() copies the hostname being resolved (first if from a list) to a
// buffer in the network object. This is later used by SOCKS5 if lookup fails.
//
// Revision 1.48  1999/01/04 04:47:55  cyp
// Minor fixes for platforms without network support.
//
// Revision 1.47  1999/01/02 07:18:23  dicamillo
// Add ctype.h for BeOS.
//
// Revision 1.46  1999/01/01 02:45:16  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.45  1998/12/31 17:55:50  cyp
// changes to Network::Open(): (a) retry loop is inside ::Open() (was from
// the external NetOpen()) (b) cleaned up the various hostname/addr/port
// variables to make sense and be uniform throughout. (c) nofallback handling
// is performed by ::Open() and not by the external NetOpen().
//
// Revision 1.44  1998/12/24 05:19:55  dicamillo
// Add socket_ioctl to Mac OS definitions.
//
// Revision 1.43  1998/12/22 15:58:24  jcmichot
// *** empty log message ***
//
// Revision 1.42  1998/12/21 17:54:23  cyp
// (a) Network connect is now non-blocking. (b) timeout param moved from
// network::Get() to object scope.
//
// Revision 1.41  1998/12/08 05:57:03  dicamillo
// Add defines for MacOS.
//
// Revision 1.40  1998/10/26 03:21:53  cyp
// More tags fun.
//
// Revision 1.39  1998/10/19 12:42:06  cyp
// win16 changes
//
// Revision 1.38  1998/09/25 11:31:15  chrisb
// Added stuff to support 3 cores in the ARM clients.
//
// Revision 1.37  1998/09/25 04:32:12  pct
// DEC Ultrix port changes
//
// Revision 1.36  1998/09/03 16:01:35  cyp
// Added TLI support. Any other SYSV (-type) takers?
//
// Revision 1.34  1998/08/28 22:05:49  cyp
// Added prototypes for new/extended "low level" methods.
//
// Revision 1.33  1998/08/25 00:06:57  cyp
// Merged (a) the Network destructor and DeinitializeNetwork() into NetClose()
// (b) the Network constructor and InitializeNetwork() into NetOpen().
// These two new functions (in netinit.cpp) are essentially what the static
// FetchFlushNetwork[Open|Close]() functions in buffupd.cpp used to be.
//
// Revision 1.32  1998/08/10 21:53:59  cyruspatel
// Changes: (a) now have a method to determine if net availability state has 
// changed (or existed to begin with) and (b) also protect against any 
// re-definition of client.offlinemode (c) The NO!NETWORK define is
// now obsolete. Whether a platform has networking capabilities or not is now
// a purely network.cpp thing. ** Documentation ** is in netinit.cpp
//
// Revision 1.31  1998/08/02 16:18:22  cyruspatel
// Completed support for logging.
//
// Revision 1.30  1998/07/29 21:34:33  blast
// AmigaOS specific change due to change from platforms/ to platforms/amiga
// for AmigaOS specific files ...
//
// Revision 1.29  1998/07/26 12:46:16  cyruspatel
// new inifile option: 'autofindkeyserver', ie if keyproxy= points to a
// xx.v27.distributed.net then that will be interpreted by Network::Resolve()
// to mean 'find a keyserver that covers the timezone I am in'. Network
// constructor extended to take this as an argument.
//
// Revision 1.28  1998/07/16 21:23:04  nordquist
// More DYNIX port changes.
//
// Revision 1.27  1998/07/08 09:24:58  jlawson
// eliminated integer size warnings on win16
//
// Revision 1.26  1998/07/08 05:19:34  jlawson
// updates to get Borland C++ to compile under Win32.
//
// Revision 1.25  1998/07/07 21:55:48  cyruspatel
// client.h has been split into client.h and baseincs.h 
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
// (is now inline asm), added NO!NETWORK wrapper around Network::Resolve()
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
#include <stddef.h>    // for offsetof
#include <time.h>
#include <errno.h>     // for errno and EINTR

#if ((CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_RISCOS))
}
#endif

#if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
  #define WIN32_LEAN_AND_MEAN
  #ifndef STRICT
    #define STRICT
  #endif
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

////////////////////////////////////////////////////////////////////////////

#define MODE_UUE        1
#define MODE_HTTP       2
#define MODE_SOCKS4     4
#define MODE_PROXIED    (MODE_HTTP | MODE_SOCKS4 | MODE_SOCKS5)
#define MODE_SOCKS5     8
#define DEFAULT_RRDNS   "us.v27.distributed.net"
#define DEFAULT_PORT    2064
#define CONDSOCK_BLOCKING_ON     0x00000011L
#define CONDSOCK_BLOCKING_OFF    0x00000010L
#define CONDSOCK_NONBLOCKING_ON  CONDSOCK_BLOCKING_OFF
#define CONDSOCK_NONBLOCKING_OFF CONDSOCK_BLOCKING_ON
#ifndef INVALID_SOCKET
#define INVALID_SOCKET  ((SOCKET)(-1))
#endif

//-------------------------------------------------------------------------

class Network; //for forward resolution

// two functions that combine the functionality of NetworkInitialize()+
// Network::Network() and NetworkDeinitialize()+Network::~Network() 
// must be called instead of the Network constructor or destructor.
extern Network *NetOpen(const char *keyserver, s32 keyserverport, 
                int nofallback = 1, int autofindks = 0, 
                int iotimeout = 10, s32 proxytype = 0, 
                const char *proxyhost = NULL, s32 proxyport = 0, 
                const char *proxyuid = NULL);

extern int NetClose( Network *net );

///////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------
// Only the "LowLevel..()" methods (and Resolve()) use BSD socket functions.
// Protected functions do not (should not) display error messages.
//----------------------------------------------------------------

class Network
{
protected:
  char server_name[64];   // used only by ::Open
  s16  server_port;       // used only by ::Open
  int  nofallback;        // used only by ::Open

  int  startmode;
  int  autofindkeyserver; // implies 'only if hostname is a dnet keyserver'
  int  verbose_level;     // 0 == no messages, 1 == user, 2 = diagnostic/debug
  int  iotimeout;         // use blocking calls if iotimeout is <0

  int  mode;              // startmode as modified at runtime
  SOCKET sock;            // socket file handle
  int  isnonblocking;     // whether the socket could be set non-blocking
  int  reconnected;       // set to 1 once a connect succeeds 

  char fwall_hostname[64]; //intermediate
  s16  fwall_hostport;
  u32  fwall_hostaddr;
  char fwall_userpass[128]; //username+password
  char resolve_hostname[64]; //last hostname Resolve() did a lookup on.
                          //used by socks5 if the svc_hostname doesn't resolve

  char *svc_hostname;  //name of the final dest (server_name or rrdns_name)
  s16  svc_hostport;   //the port of the final destination
  u32  svc_hostaddr;   //resolved if direct connection or socks.

  char *conn_hostname; //hostname we connect()ing to (fwall or server)
  s16  conn_hostport;  //port we are connect()ing to
  u32  conn_hostaddr;  //the address we are connect()ing to

  //u32  lastaddress;
  //s16  lastport;

  // http related
  //char httpproxy[64];     // also used for socks4
  //s16 httpport;           // also used for socks4
  //u32 lasthttpaddress;

  // communications and decoding buffers
  AutoBuffer netbuffer, uubuffer;
  int gotuubegin, gothttpend, puthttpdone, gethttpdone;
  u32 httplength;

  int InitializeConnection(void); //high level method. Used internally by Open
    //currently only negotiates/authenticates the SOCKSx session. 
    // returns < 0 on error, 0 on success

  int Resolve( const char *host, u32 *hostaddress, int hostport ); //LowLevel.. 
    // perform a DNS lookup, handling random selection of DNS lists
    // returns < 0 on error, 0 on success

  int LowLevelCreateSocket(void);
    // Returns < 0 on error, else assigns fd to this->sock and returns 0.
    
  int LowLevelCloseSocket(void);
    // destroys this->sock if (this->sock != INVALID_SOCKET) and returns 0.

  int LowLevelConnectSocket( u32 that_address, u16 that_port ); 
    // connect to address:port.  Return < 0 if error

  int LowLevelConditionSocket( unsigned long cond_type );
    // Returns < 0 if error - see CONDSOCK... defines above

  s32 LowLevelGet(u32 length, char *data);
    // returns 0 if success/-1 if error. 
    // length of read buffer stored back into 'length'

  s32 LowLevelPut(u32 length, const char *data);
    // returns 0 if success/-1 if error. 
    // length of sent buffer stored back into 'length'

  Network( const char *preferred, s16 port, int nofallback,
           int AutoFindKeyServer, int _iotimeout );
    // constructor: If preferred is NULL, use DEFAULT_RRDNS.
    // Can be IP and not necessarily a name, though a name is better
    // suited for roundrobin use.  if port is 0, use DEFAULT_PORT
    // protected!: used by friend NetOpen() below.

  ~Network( void );
    // guess. 

  int  Open( SOCKET insock );
    // takes over a preconnected socket to a client.
    // returns -1 on error, 0 on success

  int  Open( void );
    // [re]open the connection using the current settings.
    // returns -1 on error, 0 on success

  friend Network *NetOpen(const char *keyserver, s32 keyserverport, 
                          int nofallback, int autofindks, int iotimeout, 
                          s32 proxytype, const char *proxyhost, s32 proxyport, 
                          const char *proxyuid );

  friend int NetClose( Network *net );

public:

  void SetModeUUE( int is_enabled );
    // enables or disable uuencoding of the data stream

  void SetModeHTTP( const char *httpproxyin = NULL, s16 httpportin = 80,
      const char * httpidin = NULL );
    // enables http-proxy tunnelling transmission.
    // Specifying NULL for httpproxy will disable.

  void SetModeSOCKS4(const char *sockshost = NULL, s16 socksport = 1080,
      const char * socksusername = NULL );
    // enables socks4-proxy tunnelling transmission.
    // Specifying NULL for sockshost will disable.

  void SetModeSOCKS5(const char *sockshost = NULL, s16 socksport = 1080,
      const char * socksusernamepw = NULL );
    // enables socks5-proxy tunnelling transmission.
    // Specifying NULL for sockshost will disable.
    // usernamepw is username:pw
    // An empty or NULL usernamepw means use only no authentication method

  int  Close( void );
    // close the connection

  s32  Get( u32 length, char * data );
    // recv data over the open connection, uue/http based on ( mode & MODE_UUE ) etc.
    // Returns length of read buffer.

  s32  Put( u32 length, const char * data );
    // send data over the open connection, uue/http based on ( mode & MODE_UUE ) etc.
    // returns -1 on error, or 0 on success

  int GetHostName( char *buffer, unsigned int len ); //used by mail.
    // like gethostname() 
    
  int MakeBlocking(void) // make the socket operate in blocking mode.
      { return LowLevelConditionSocket( CONDSOCK_BLOCKING_ON ); }

  int MakeNonBlocking(void) //the other shortcut
      { return LowLevelConditionSocket( CONDSOCK_BLOCKING_OFF ); };
};

///////////////////////////////////////////////////////////////////////////

#endif //NETWORK_H
