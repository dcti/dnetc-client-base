/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __NETWORK_H__
#define __NETWORK_H__ "@(#)$Id: network.h,v 1.68.2.1 1999/11/08 00:01:20 cyp Exp $"

#include "cputypes.h"
#include "autobuff.h"


//#define SELECT_FIRST      // define to perform select() before reading
//#define __VMS_UCX__       // define for UCX instead of Multinet on VMS

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

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
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
  //ntohl()/htonl() defines are in...
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
  #define BSD_SELECT
  #include <sys/types.h>
  #include <fcntl.h>
  #include <netdb.h>
  #include <netinet/in.h>
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <sys/select.h>
  #include <sys/ioctl.h>
  #if defined(__EMX__)
    // this has to stay as long as the define below is needed
    #include <io.h>
    #define soclose(sock) close(sock)
  #endif
  typedef int SOCKET;
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
  #elif (CLIENT_OS == OS_QNX)
    #include <sys/select.h>
  #elif (CLIENT_OS == OS_DYNIX) && defined(NTOHL)
    #define ntohl(x)  NTOHL(x)
    #define htonl(x)  HTONL(x)
    #define ntohs(x)  NTOHS(x)
    #define htons(x)  HTONS(x)
  #elif ((CLIENT_OS == OS_SUNOS) && (CLIENT_CPU==CPU_68K))
    #if defined(_SUNOS3_)
      #define _SOCKET_H_ALREADY_
      extern "C" int fcntl(int, int, int);
    #endif
    extern "C" {
    int socket(int, int, int);
    int setsockopt(int, int, int, char *, int);
    int connect(int, struct sockaddr *, int);
    }
  #elif (CLIENT_OS == OS_ULTRIX)
    extern "C" {
      int socket(int, int, int);
      int setsockopt(int, int, int, char *, int);
      int connect(int, struct sockaddr *, int);
    }
  #elif (CLIENT_OS == OS_NETWARE)
    #include "platforms/netware/netware.h" //symbol redefinitions
    extern "C" {
    #pragma pack(1)
    #include <tiuser.h> //using TLI
    #pragma pack()
    }
  #endif  
#endif

////////////////////////////////////////////////////////////////////////////

#define MODE_UUE              0x01
#define MODE_HTTP             0x02
#define MODE_SOCKS4           0x04
#define MODE_PROXIED          (MODE_HTTP | MODE_SOCKS4 | MODE_SOCKS5)
#define MODE_SOCKS5           0x08
#define DEFAULT_RRDNS           ""
#define DEFAULT_PORT          2064
#define CONDSOCK_BLOCKMODE       1
#define CONDSOCK_KEEPALIVE       2   
#define CONDSOCK_SETMINBUFSIZE   3
#ifndef INVALID_SOCKET
#define INVALID_SOCKET  ((SOCKET)(-1))
#endif

////////////////////////////////////////////////////////////////////////////

int InitializeConnectivity(void);   //per instance initialization
int DeinitializeConnectivity(void);

class Network
{
protected:
  char server_name[64];   // used only by ::Open
  int server_port;       // used only by ::Open
  int nofallback;        // used only by ::Open

  int  startmode;
  int autofindkeyserver; // implies 'only if hostname is a dnet keyserver'
  int  verbose_level;     // 0 == no messages, 1 == user, 2 = diagnostic/debug
  int  iotimeout;         // use blocking calls if iotimeout is <0

  int  mode;              // startmode as modified at runtime
  SOCKET sock;            // socket file handle
  int isnonblocking;     // whether the socket could be set non-blocking
  int reconnected;       // set to 1 once a connect succeeds 
  int shown_connection;

  char fwall_hostname[64]; //intermediate
  int  fwall_hostport;
  u32  fwall_hostaddr;
  char fwall_userpass[128]; //username+password

  char resolve_hostname[64]; //last hostname Resolve() did a lookup on.
                             //used by socks5 if the svc_hostname doesn't resolve
  u32  resolve_addrlist[32]; //list of resolved (proxy) addresses
  int  resolve_addrcount;    //number of addresses in there. <0 if uninitialized

  char *svc_hostname;  //name of the final dest (server_name or rrdns_name)
  int  svc_hostport;   //the port of the final destination
  u32  svc_hostaddr;   //resolved if direct connection or socks.

  char *conn_hostname; //hostname we connect()ing to (fwall or server)
  int  conn_hostport;  //port we are connect()ing to
  u32  conn_hostaddr;  //the address we are connect()ing to

  // communications and decoding buffers
  AutoBuffer netbuffer, uubuffer;
  int gotuubegin, gothttpend, puthttpdone, gethttpdone;
  u32 httplength;

  int LowLevelCreateSocket(void);
    // Returns < 0 on error, else assigns fd to this->sock and returns 0.
    
  int LowLevelCloseSocket(void);
    // destroys this->sock if (this->sock != INVALID_SOCKET) and returns 0.

  int LowLevelConnectSocket( u32 that_address, int that_port ); 
    // connect to address:port.  Return < 0 if error

  int LowLevelSetSocketOption( int cond_type, int parm );
    // Returns < 0 if error - see CONDSOCK... defines above

  int LowLevelGet(char *data, int length);
    // returns 0 if sock closed, -1 if timeout, else length of rcvd data

  int LowLevelPut(const char *data, int length);
    // returns 0 if sock closed, -1 if timeout, else length of sent data

  int InitializeConnection(void); //high level method. Used internally by Open
    //currently only negotiates/authenticates the SOCKSx session. 
    // returns < 0 on error, 0 on success

  int  Close( void );
    // close the connection

  int  Open( SOCKET insock);
    // reset http/uue settings and reconnect the socket

  int  Open( void );
    // [re]open the connection using the current settings.
    // returns -1 on error, 0 on success

  ~Network( void );
    // guess. 
 
  Network( const char *servname, int servport, 
           int _nofallback = 1, int _iotimeout = -1, int _enctype = 0, 
           const char *_fwallhost = NULL, int _fwallport = 0, 
           const char *_fwalluid = NULL );
    // protected!: used by friend NetOpen() below.

public:

  friend Network *NetOpen( const char *servname, int servport, 
           int _nofallback = 1, int _iotimeout = -1, int _enctype = 0, 
           const char *_fwallhost = NULL, int _fwallport = 0, 
           const char *_fwalluid = NULL );

  friend int NetClose( Network *net );

  int Get( char * data, int length );
    // recv data over the open connection, handle uue/http translation,
    // Returns length of read buffer.

  int Put( const char * data, int length );
    // send data over the open connection, handle uue/http translation,
    // Returns length of sent buffer

  int GetHostName( char *buffer, unsigned int len );
    // used by mail.
    // like gethostname(). returns !0 on error

  int SetPeerAddress( u32 addr ) 
    { if (svc_hostaddr == 0) svc_hostaddr = addr; return 0; }
    // used by buffupd when proxies return an address in a packet
    
  int Reset( u32 thataddress );
    // reset the connection (if thataddress==0, then hard).
    // the old connection is invalid on return (even if reset fails). 
    
  u32 GetPeerAddress(void)  { return svc_hostaddr; }
    //for debugging

  void ShowConnection(void);
    //show who we are connected to. (::Open() no longer does this)
};

#endif /* __NETWORK_H__ */

