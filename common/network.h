/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __NETWORK_H__
#define __NETWORK_H__ "@(#)$Id: network.h,v 1.78 2000/07/03 07:13:15 jlawson Exp $"

#include "cputypes.h"
#include "autobuff.h"
#include "baseincs.h"
#include "netio.h"

//#define SELECT_FIRST      // define to perform select() before reading
//#define __VMS_UCX__       // define for UCX instead of Multinet on VMS


////////////////////////////////////////////////////////////////////////////

#define MODE_UUE              0x01
#define MODE_HTTP             0x02
#define MODE_SOCKS4           0x04
#define MODE_PROXIED          (MODE_HTTP | MODE_SOCKS4 | MODE_SOCKS5)
#define MODE_SOCKS5           0x08
#define DEFAULT_RRDNS           ""
#define DEFAULT_PORT          2064

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

protected:

  int InitializeConnection(void);
    // high level method. Used internally by Open
    // currently only negotiates/authenticates the SOCKSx session.
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
           bool _nofallback = true, int _iotimeout = -1, int _enctype = 0,
           const char *_fwallhost = NULL, int _fwallport = 0,
           const char *_fwalluid = NULL );
    // protected!: used by friend NetOpen() below.

public:

  friend Network *NetOpen( const char *servname, int servport,
           bool _nofallback = true, int _iotimeout = -1, int _enctype = 0,
           const char *_fwallhost = NULL, int _fwallport = 0,
           const char *_fwalluid = NULL );

  friend int NetClose( Network *net );

  int Get( char * data, int length );
    // recv data over the open connection, handle uue/http translation,
    // Returns length of read buffer.

  int Put( const char * data, int length );
    // send data over the open connection, handle uue/http translation,
    // Returns length of sent buffer

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

////////////////////////////////////////////////////////////////////////////

#endif /* __NETWORK_H__ */

