// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: network.cpp,v $
// Revision 1.42  1998/09/06 02:30:48  cyp
// Added check 'if (t_errno < t_nerr)' before getting t_errlist[t_errno].
//
// Revision 1.41  1998/09/06 01:05:10  cyp
// Fixed a missing underscore in an #ifdef _TIUSER[_]
//
// Revision 1.40  1998/09/04 10:35:09  chrisb
// RISCOSism for the #include <errno.h>
//
// Revision 1.39  1998/09/03 16:01:32  cyp
// Added TLI support. Any other SYSV (-type) takers?
//
// Revision 1.38  1998/08/28 22:04:49  cyp
// Cleaned up/extended the division between "high level" and "low level"
// methods: inet/socket functions/structures are encapsulated in small
// "low level" methods and "high level" methods are completely clean of
// any inet/socket library/include dependancies. Also fixed assumptions
// that zero is an invalid socket/fd.
//
// Revision 1.37  1998/08/25 00:06:53  cyp
// Merged (a) the Network destructor and DeinitializeNetwork() into NetClose()
// (b) the Network constructor and InitializeNetwork() into NetOpen().
// These two new functions (in netinit.cpp) are essentially what the static
// FetchFlushNetwork[Open|Close]() functions in buffupd.cpp used to be.
//
// Revision 1.36  1998/08/20 19:27:10  cyruspatel
// Made the purpose of NetworkInitialize/Deinitialize a little more
// transparent.
//
// Revision 1.35  1998/08/10 21:53:51  cyruspatel
// Two changes to work around a lack of a method to detect if the network
// availability state had changed (or existed to begin with) and also protect
// against any re-definition of client.offlinemode. (a) The NO!NETWORK define 
// is now obsolete. Whether a platform has networking capabilities or not is 
// now a purely network.cpp thing. (b) NetworkInitialize()/NetworkDeinitialize
// are no longer one-shot-and-be-done-with-it affairs. ** Documentation ** is
// in netinit.cpp.
//
// Revision 1.34  1998/08/07 20:27:10  cyruspatel
// Added timestamps to network messages.
//
// Revision 1.33  1998/08/02 16:18:12  cyruspatel
// Completed support for logging.
//
// Revision 1.32  1998/07/26 12:46:12  cyruspatel
// new inifile option: 'autofindkeyserver', ie if keyproxy= points to a
// xx.v27.distributed.net then that will be interpreted by Network::Resolve()
// to mean 'find a keyserver that covers the timezone I am in'. Network
// constructor extended to take this as an argument.
//
// Revision 1.31  1998/07/18 05:53:10  ziggyb
// Removed a unneeded return 0, which was causing a 'unreachable code' 
// warning in Watcom
//
// Revision 1.30  1998/07/13 23:54:25  cyruspatel
// Cleaned up NO!NETWORK handling.
//
// Revision 1.29  1998/07/13 03:30:07  cyruspatel
// Added 'const's or 'register's where the compiler was complaining about
// "declaration/type or an expression" ambiguities.
//
// Revision 1.28  1998/07/08 09:24:54  jlawson
// eliminated integer size warnings on win16
//
// Revision 1.27  1998/07/07 21:55:46  cyruspatel
// client.h has been split into client.h and baseincs.h 
//
// Revision 1.26  1998/06/29 06:58:06  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.25  1998/06/29 04:22:26  jlawson
// Updates for 16-bit Win16 support
//
// Revision 1.24  1998/06/26 06:56:38  daa
// convert all the strncmp's in the http code with strncmpi to 
// deal with proxy's case shifting HTML
//
// Revision 1.23  1998/06/22 01:04:58  cyruspatel
// DOS changes. Fixes various compile-time errors: removed extraneous ')' in
// sleepdef.h, resolved htonl()/ntohl() conflict with same def in client.h
// (is now inline asm), added NO!NETWORK wrapper around Network::Resolve()
//
// Revision 1.22  1998/06/15 12:04:03  kbracey
// Lots of consts.
//
// Revision 1.21  1998/06/14 13:08:09  ziggyb
// Took out all OS/2 DOD stuff, being moved to platforms\os2cli\dod.cpp
//
// Revision 1.20  1998/06/14 10:13:29  ziggyb
// There was a stray '\' on line 567 that's not supposed to be there
//
// Revision 1.19  1998/06/14 08:26:52  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.18  1998/06/14 08:13:00  friedbait
// 'Log' keywords added to maintain automatic change history
//
// Revision 1.18  1998/06/13 23:33:18  cyruspatel
// Fixed NetWare stuff and added #include "sleepdef.h" (which should now 
// warn if macros are not the same)
//

#if (!defined(lint) && defined(__showids__))
const char *network_cpp(void) {
return "@(#)$Id: network.cpp,v 1.42 1998/09/06 02:30:48 cyp Exp $"; }
#endif

//----------------------------------------------------------------------

#include "cputypes.h"
#include "sleepdef.h"  // Fix sleep()/usleep() macros there! <--
#include "autobuff.h"  // Autobuffer class
#include "cmpidefs.h"  // strncmpi(), strcmpi()
#include "logstuff.h"  // LogScreen()
#include "clitime.h"   // CliGetTimeString(NULL,1);
#include <stddef.h>    // for offsetof
#if (CLIENT_OS == OS_RISCOS)
extern "C"
{
#include <errno.h>     // for errno and EINTR
};
#else
#include <errno.h>     // for errno and EINTR
#endif


#include "network.h"   // thats us
extern int NetCheckIsOK(void); // used before doing i/o
#define Time() (CliGetTimeString(NULL,1))

//----------------------------------------------------------------------

#ifdef NONETWORK
#error NONETWORK define is obsolete. Networking capability is now a purely network.cpp thing.
#endif

//----------------------------------------------------------------------

#define VERBOSE_OPEN //print cause of ::Open() errors
#define UU_DEC(Ch) (char) (((Ch) - ' ') & 077)
#define UU_ENC(Ch) (char) (((Ch) & 077) != 0 ? ((Ch) & 077) + 0x20 : '`')

//----------------------------------------------------------------------

#pragma pack(1)               // no padding allowed

typedef struct _socks4 {
  unsigned char VN;           // version == 4
  unsigned char CD;           // command code, CONNECT == 1
  u16 DSTPORT;                // destination port, network order
  u32 DSTIP;                  // destination IP, network order
  char USERID[1];             // variable size, null terminated
} SOCKS4;

//----------------------------------------------------------------------

typedef struct _socks5methodreq {
  unsigned char ver;          // version == 5
  unsigned char nMethods;     // number of allowable methods following
  unsigned char Methods[2];   // this program will request at most two
} SOCKS5METHODREQ;

//----------------------------------------------------------------------

typedef struct _socks5methodreply {
  unsigned char ver;          // version == 1
  unsigned char Method;       // server chose method ...
  char end;
} SOCKS5METHODREPLY;

//----------------------------------------------------------------------

typedef struct _socks5userpwreply {
  unsigned char ver;          // version == 1
  unsigned char status;       // 0 == success
  char end;
} SOCKS5USERPWREPLY;

//----------------------------------------------------------------------

typedef struct _socks5 {
  unsigned char ver;          // version == 5
  unsigned char cmdORrep;     // cmd: 1 == connect, rep: 0 == success
  unsigned char rsv;          // must be 0
  unsigned char atyp;         // address type we assume IPv4 == 1
  u32 addr;                   // network order
  u16 port;                   // network order
  char end;
} SOCKS5;

#pragma pack()

//----------------------------------------------------------------------

const char *Socks5ErrorText[9] =
{
 /* 0 */ NULL,                              // success
  "general SOCKS server failure",
  "connection not allowed by ruleset",
  "Network unreachable",
  "Host unreachable",
  "Connection refused",
  "TTL expired",
  "Command not supported",
  "Address type not supported"
};

//======================================================================

class NetTimer
{
  time_t start;
public:
  NetTimer(void) : start(time(NULL)) {};
  operator u32 (void) {return time(NULL) - start;}
};

//======================================================================

//short circuit the u32 -> in_addr.s_addr ->inet_ntoa method.
//besides, it works around context issues.
static const char *__inet_ntoa__(u32 addr)
{
  static char buff[18];
  char *p = (char *)(&addr);
  sprintf( buff, "%d.%d.%d.%d", (p[0]&255),(p[1]&255),(p[2]&255),(p[3]&255) );
  return buff;
}  

//======================================================================

Network::Network( const char * Preferred, const char * Roundrobin, 
                  s16 Port, int AutoFindKeyServer )
{
  // intialize communication parameters
  strncpy(server_name, (Preferred ? Preferred : ""), sizeof(server_name));
  strncpy(rrdns_name, (Roundrobin ? Roundrobin : ""), sizeof(rrdns_name));
  port = (s16) (Port ? Port : DEFAULT_PORT);
  autofindkeyserver = AutoFindKeyServer;
  isnonblocking = 0; // whether the socket could be set non-blocking
  mode = startmode = 0;
  retries = 0;
  sock = INVALID_SOCKET;
  gotuubegin = gothttpend = 0;
  httplength = 0;
  lasthttpaddress = lastaddress = 0;
  httpid[0] = 0;

  #ifdef DEBUG
  verbose_level = 2;
  #elif defined(VERBOSE_OPEN)
  verbose_level = 1;
  #else
  verbose_level = 0; //quiet
  #endif
  
  // check that the packet structures have been correctly packed
  size_t dummy;
  if (((dummy = offsetof(SOCKS4, USERID[0])) != 8) ||
     ((dummy = offsetof(SOCKS5METHODREQ, Methods[0])) != 2) ||
     ((dummy = offsetof(SOCKS5METHODREPLY, end)) != 2) ||
     ((dummy = offsetof(SOCKS5USERPWREPLY, end)) != 2) ||
     ((dummy = offsetof(SOCKS5, end)) != 10))
    LogScreenf("[%s] Network::Socks Incorrectly packed structures.\n",Time());
}


//----------------------------------------------------------------------

Network::~Network(void)
{
  Close();
}

//----------------------------------------------------------------------

void Network::SetModeUUE( int is_enabled )
{
  if (is_enabled)
    {
    startmode &= ~(MODE_SOCKS4 | MODE_SOCKS5);
    startmode |= MODE_UUE;
    }
  else startmode &= ~MODE_UUE;
}

//----------------------------------------------------------------------

void Network::SetModeHTTP( const char *httpproxyin, s16 httpportin,
    const char *httpidin)
{
  if (httpproxyin && httpproxyin[0])
  {
    startmode &= ~(MODE_SOCKS4 | MODE_SOCKS5);
    startmode |= MODE_HTTP;
    httpport = httpportin;
    strncpy(httpid, httpidin, 128);
    strncpy(httpproxy, httpproxyin, 64);
  }
  else startmode &= ~MODE_HTTP;
  lastaddress = lasthttpaddress = 0;
}

//----------------------------------------------------------------------

void Network::SetModeSOCKS4(const char *sockshost, s16 socksport,
      const char * socksusername )
{
  if (sockshost && sockshost[0])
  {
    startmode &= ~(MODE_HTTP | MODE_SOCKS5 | MODE_UUE);
    startmode |= MODE_SOCKS4;
    httpport = socksport;
    strncpy(httpproxy, sockshost, sizeof(httpproxy));
    if (socksusername) {
      strncpy(httpid, socksusername, sizeof(httpid));
    } else {
      httpid[0] = 0;
    }
  }
  else
  {
    startmode &= ~MODE_SOCKS4;
    httpproxy[0] = 0;
  }
  lastaddress = lasthttpaddress = 0;
}

//----------------------------------------------------------------------

void Network::SetModeSOCKS5(const char *sockshost, s16 socksport,
      const char * socksusernamepw )
{
  if (sockshost && sockshost[0])
  {
    startmode &= ~(MODE_HTTP | MODE_SOCKS4 | MODE_UUE);
    startmode |= MODE_SOCKS5;
    httpport = socksport;
    strncpy(httpproxy, sockshost, sizeof(httpproxy));
    if (socksusernamepw) {
      strncpy(httpid, socksusernamepw, sizeof(httpid));
    } else {
      httpid[0] = 0;
    }
  }
  else
  {
    startmode &= ~MODE_SOCKS5;
    httpproxy[0] = 0;
  }
  lastaddress = lasthttpaddress = 0;
}

/* ----------------------------------------------------------------------- */

// returns -1 on error, 0 on success
int Network::Open( SOCKET insock)
{
  sock = insock;

  // set communications settings
  mode = startmode;
  gotuubegin = gothttpend = puthttpdone = gethttpdone = 0;
  httplength = 0;
  netbuffer.Clear();
  uubuffer.Clear();

  // make socket non-blocking
  isnonblocking = ( MakeNonBlocking() == 0);
  return 0;
}  

/* ----------------------------------------------------------------------- */

int Network::Open( void )               // returns -1 on error, 0 on success
{
  int success;
  mode = startmode;
  const char *conntohost = "";

  Close();

  if (!NetCheckIsOK())
    return -1;
  
  success = (LowLevelCreateSocket() == 0);      // create a new socket

  if (!success)
    {
    if (verbose_level > 0) 
      LogScreen("[%s] Network::failed to create network socket.\n", Time() );
    }
    
  if (success)  
    {
    // resolve the address of keyserver, or if performing a proxied
    // connection, the address of the proxy gateway server.
    if (lastaddress == 0)
      {
      if (mode & MODE_PROXIED) 
        {
        conntohost = httpproxy;
        lastport = httpport;
        }
      else
        {
        lastport = port;
        if ( retries < 4 )
          conntohost = server_name;
        else
          {
          conntohost = rrdns_name;
          autofindkeyserver = 1;
          }
        if (!conntohost[0]) 
          {
          autofindkeyserver = 1;
          conntohost = DEFAULT_RRDNS;
          lastport = DEFAULT_PORT;
          } 
        }

      if (!NetCheckIsOK())
        {
        success = 0;
        lastaddress = 0;
        retries = 4;
        }
      else
        {
        if (Resolve(conntohost, lastaddress ) < 0)
          {
          lastaddress = 0;
          success = 0;
          retries++;

          if (verbose_level > 0)
            LogScreen("[%s] Network::failed to resolve hostname \"%s\"\n", 
                         Time(), conntohost);
          }
        }  // NetCheckIsOK())
      } // if (lastaddress == 0)
    } // if (success)  


  if ( success )
    {
    // if we are doing a proxied connection, resolve the actual destination
    if ((mode & MODE_PROXIED) && (lasthttpaddress == 0))
      {
      const char * hostname;
      lastport = httpport;
      if ( retries < 4 )
        hostname = server_name;
      else
        {
        hostname = rrdns_name;
        autofindkeyserver = 1;
        }
      if (!hostname[0]) 
        {
        autofindkeyserver = 1;
        hostname = DEFAULT_RRDNS;
        lastport = DEFAULT_PORT;
        } 

      if (!NetCheckIsOK())
        {
        lasthttpaddress = 0;
        success = 0;
        retries = 4;
        }
      else
        {
        if (Resolve(hostname, lasthttpaddress ) < 0)
          {
          lasthttpaddress = 0;
          success = 0;
          if (mode & (MODE_SOCKS4 | MODE_SOCKS5))
            {
            // socks needs the address to resolve now.
            lastaddress = 0;
            
            if (verbose_level > 0)
              LogScreen("[%s] Network::failed to resolve hostname \"%s\"\n", 
                          Time(), hostname);
            }
          } // if (Resolve(hostname, lasthttpaddress ) < 0)
        } // if (NetCheckIsOK())
      } // if ((mode & MODE_PROXIED) && (lasthttpaddress == 0))
    } // if (success)  


  if (success)
    {
    if (verbose_level > 0)
      {
      const char *p = strchr( conntohost, '.' );
      if (!*conntohost)
        conntohost = __inet_ntoa__( lastaddress );
      else if (p && (strcmpi(p,".distributed.net")==0 || 
           strcmpi(p,".v27.distributed.net")==0) && autofindkeyserver)
        conntohost = "distributed.net";
      LogScreen("[%s] Connecting to %s:%u...\n", Time(), conntohost,
                                              ((unsigned int)(lastport)) );
      }
    success = ( LowLevelConnectSocket( lastaddress, lastport ) == 0 );
    if (!success)
      {
      if (verbose_level > 0)
        {
        LogScreen( "[%s] Connect to host %s:%u failed.\n",
           Time(),  __inet_ntoa__(lastaddress), (unsigned int)(lastport));
        }
      #if (CLIENT_OS == OS_WIN32) //blagh! some don't have errno
        {
        lastaddress = 0; //so reset
        }
      #else
        {
        if (!NetCheckIsOK()) //use errno only if netcheck says ok.
          lastaddress = 0; //reset
        else
          {
          #if defined(_TIUSER_)
            char *errmsg = "undefined error";
            if (t_errno < t_nerr)
              errmsg = t_errlist[t_errno];
            LogScreen( " %s  Error %d (%s)\n",CliGetTimeString(NULL,0), 
                                    t_errno, errmsg );
          #else
          int my_errno = errno;
          if (verbose_level > 0 )
            LogScreen( " %s  Error %d (%s)\n",CliGetTimeString(NULL,0), 
                                    my_errno, strerror(my_errno) );
          //#ifdef EINTR
          //if (my_errno != EINTR) //should retry
          //#endif
          #endif
          lastaddress = 0; //or reset
          }
        }
      #endif                            
      } // if (!success)
    } // if (success)

  if (success)
    {
    success = ( InitializeConnection() == 0 );
    if (!success && verbose_level > 0)
      {
      LogScreen( "[%s] Network::Failed to initialize %sconnection.\n",
       Time(), ((!startmode)?(""):((startmode & MODE_SOCKS5)?("SOCKS5 "):
       ((startmode & MODE_SOCKS4)?("SOCKS4 "):
       ((startmode & MODE_HTTP)?("HTTP "):("??? "))))));
      }
    }
    
  if (success)
    {
    isnonblocking = ( MakeNonBlocking() == 0 );
    if (verbose_level > 1) //debug
      LogScreen("[%s] Network::Connected (%sblocking).\n", Time(),
                                 ((isnonblocking)?("non-"):("")) );
    }

  if (!success)
    {
    LowLevelCloseSocket();
    retries++;
    return -1;
    }
  return 0;
}  

// -----------------------------------------------------------------------
    
int Network::InitializeConnection(void)
{
  // set communications settings
  gotuubegin = gothttpend = puthttpdone = gethttpdone = 0;
  httplength = 0;
  netbuffer.Clear();
  uubuffer.Clear();

  if (!NetCheckIsOK())
    return -1;
  if (sock == INVALID_SOCKET)
    return -1;

  if (startmode & MODE_SOCKS5)
    {
    int success = 0; //assume failed
    char socksreq[600];  // room for large username/pw (255 max each)
    SOCKS5METHODREQ *psocks5mreq = (SOCKS5METHODREQ *)socksreq;
    SOCKS5METHODREPLY *psocks5mreply = (SOCKS5METHODREPLY *)socksreq;
    SOCKS5USERPWREPLY *psocks5userpwreply = (SOCKS5USERPWREPLY *)socksreq;
    SOCKS5 *psocks5 = (SOCKS5 *)socksreq;
    u32 len;

    // transact a request to the SOCKS5 proxy requesting
    // authentication methods.  If the username/password
    // is provided we ask for no authentication or user/pw.
    // Otherwise we ask for no authentication only.

    psocks5mreq->ver = 5;
    psocks5mreq->nMethods = (unsigned char) (httpid[0] ? 2 : 1);
    psocks5mreq->Methods[0] = 0;  // no authentication
    psocks5mreq->Methods[1] = 2;  // username/password

    len = 2 + psocks5mreq->nMethods;
    if (LowLevelPut(len, socksreq) < 0)
      goto Socks5InitEnd;

    if ((u32)LowLevelGet(2, socksreq) != 2)
      goto Socks5InitEnd;

    if (psocks5mreply->ver != 5)
      {
      if (verbose_level > 0)
      LogScreen("[%s] SOCKS5 authentication has wrong version, %d should be 5\n", 
                           Time(), psocks5mreply->ver);
      goto Socks5InitEnd;
      }

    if (psocks5mreply->Method == 0)       // no authentication
      {
      // nothing to do for no authentication method
      }
    else if (psocks5mreply->Method == 2)  // username and pw
      {
      char username[255];
      char password[255];
      char *pchSrc, *pchDest;
      int userlen, pwlen;

      pchSrc = httpid;
      pchDest = username;
      while (*pchSrc && *pchSrc != ':')
        *pchDest++ = *pchSrc++;
      *pchDest = 0;
      userlen = pchDest - username;
      if (*pchSrc == ':')
        pchSrc++;
      strcpy(password, pchSrc);
      pwlen = strlen(password);

      //   username/password request looks like
      // +----+------+----------+------+----------+
      // |VER | ULEN |  UNAME   | PLEN |  PASSWD  |
      // +----+------+----------+------+----------+
      // | 1  |  1   | 1 to 255 |  1   | 1 to 255 |
      // +----+------+----------+------+----------+

      len = 0;
      socksreq[len++] = 1;    // username/pw subnegotiation version
      socksreq[len++] = (char) userlen;
      memcpy(socksreq + len, username, (int) userlen);
      len += userlen;
      socksreq[len++] = (char) pwlen;
      memcpy(socksreq + len, password, (int) pwlen);
      len += pwlen;

      if (LowLevelPut(len, socksreq) < 0)
        goto Socks5InitEnd;

      if ((u32)LowLevelGet(2, socksreq) != 2)
        goto Socks5InitEnd;

      if (psocks5userpwreply->ver != 1 ||
          psocks5userpwreply->status != 0)
        {
        if (verbose_level > 0)
          {
          LogScreen("[%s] SOCKS5 user %s rejected by server.\n", 
                     Time(), username);
          }       
        goto Socks5InitEnd;
        }
      }
    else 
      {
      if (verbose_level > 0)
        LogScreen("[%s] SOCKS5 authentication method rejected.\n", Time());
      goto Socks5InitEnd;
      }

    // after subnegotiation, send connect request
    psocks5->ver = 5;
    psocks5->cmdORrep = 1;   // connnect
    psocks5->rsv = 0;   // must be zero
    psocks5->atyp = 1;  // IPv4
    psocks5->addr = lasthttpaddress;
    psocks5->port = htons(lastport);

    if (LowLevelPut(10, socksreq) < 0)
      goto Socks5InitEnd;

    if ((u32)LowLevelGet(10, socksreq) != 10)
      goto Socks5InitEnd;

    if (psocks5->ver != 5)
      {
      if (verbose_level > 0)
        LogScreen("[%s] SOCKS5 reply has wrong version, %d should be 5\n", 
                           Time(), psocks5->ver);
      goto Socks5InitEnd;
      }

    if (psocks5->cmdORrep == 0)          // 0 is successful connect
      {
      success = 1; 
      }
    else if (verbose_level > 0)
      {
      LogScreen("[%s] SOCKS5 server error connecting to keyproxy: \n"
                " %s  %s",
               Time(), CliGetTimeString(NULL,0),
                (psocks5->cmdORrep >=
                (sizeof Socks5ErrorText / sizeof Socks5ErrorText[0]))
                ? "unrecognized SOCKS5 error"
                : Socks5ErrorText[ psocks5->cmdORrep ] );
      }
      
Socks5InitEnd: 
    return ((success)?(0):(-1));
    }    

  if (startmode & MODE_SOCKS4)
    {
    int success = 0; //assume failed
    char socksreq[128];  // min sizeof(httpid) + sizeof(SOCKS4)
    SOCKS4 *psocks4 = (SOCKS4 *)socksreq;
    u32 len;

    // transact a request to the SOCKS4 proxy giving the
    // destination ip/port and username and process its reply.

    psocks4->VN = 4;
    psocks4->CD = 1;  // CONNECT
    psocks4->DSTPORT = htons(lastport);
    psocks4->DSTIP = lasthttpaddress;
    strncpy(psocks4->USERID, httpid, sizeof(httpid));

    len = sizeof(*psocks4) - 1 + strlen(httpid) + 1;
    if (!(LowLevelPut(len, socksreq) < 0))
      {
      len = sizeof(*psocks4) - 1;  // - 1 for the USERID[1]
      if ((u32)LowLevelGet(len, socksreq) == len)
        {
        if (psocks4->VN == 0 && psocks4->CD == 90) // 90 is successful return
          {
          success = 1;
          }
        else if (verbose_level > 0)
          {
          LogScreen("[%s] SOCKS4 request rejected%s\n", Time(), 
            (psocks4->CD == 91)
             ? ""
             :
             (psocks4->CD == 92)
             ? ", no identd response"
             :
             (psocks4->CD == 93)
             ? ", invalid identd response"
             :
             ", unexpected response");
          }
        }
      }
    return ((success)?(0):(-1));
    }
    
  return 0;
}

// -----------------------------------------------------------------------

int Network::Close(void)
{
  LowLevelCloseSocket();
  retries = 0;
  gethttpdone = puthttpdone = 0;
  netbuffer.Clear();
  uubuffer.Clear();

  gotuubegin = gothttpend = 0;
  httplength = 0;

  return 0;
}  
    
// -----------------------------------------------------------------------

// Returns length of read buffer.
s32 Network::Get( u32 length, char * data, u32 timeout )
{
  NetTimer timer;
  int need_close = 0;

  while (netbuffer.GetLength() < length && timer <= timeout)
  {
    int nothing_done = 1;

    if ((mode & MODE_HTTP) && !gothttpend)
    {
      // []---------------------------------[]
      // |  Process HTTP headers on packets  |
      // []---------------------------------[]
      uubuffer.Reserve(500);
      s32 numRead = LowLevelGet(uubuffer.GetSlack(), uubuffer.GetTail());
      if (numRead > 0) uubuffer.MarkUsed(numRead);
      else if (numRead == 0) need_close = 1;       // connection closed

      AutoBuffer line;
      while (uubuffer.RemoveLine(line))
      {
        nothing_done = 0;
        if (strncmpi(line, "Content-Length: ", 16) == 0)
        {
          httplength = atoi((const char*)line + 16);
        }
        else if ((lasthttpaddress == 0) &&
          (strncmpi(line, "X-KeyServer: ", 13) == 0))
        {
          if (Resolve( line + 13, lasthttpaddress) < 0)
            lasthttpaddress = 0;
        }
        else if (line.GetLength() < 1)
        {
          // got blank line separating header from body
          gothttpend = 1;
          if (uubuffer.GetLength() >= 6 &&
            strncmpi(uubuffer.GetHead(), "begin ", 6) == 0)
          {
            mode |= MODE_UUE;
            gotuubegin = 0;
          }
          if (!(mode & MODE_UUE))
          {
            if (httplength > uubuffer.GetLength())
            {
              netbuffer += uubuffer;
              httplength -= uubuffer.GetLength();
              uubuffer.Clear();
            } else {
              netbuffer += AutoBuffer(uubuffer, httplength);
              uubuffer.RemoveHead(httplength);
              gothttpend = 0;
              httplength = 0;

              // our http only allows one packet per socket
              nothing_done = gethttpdone = 1;
            }
          }
          break;
        }
      } // while
    }
    else if (mode & MODE_UUE)
    {
      // []----------------------------[]
      // |  Process UU Encoded packets  |
      // []----------------------------[]
      uubuffer.Reserve(500);
      s32 numRead = LowLevelGet(uubuffer.GetSlack(), uubuffer.GetTail());
      if (numRead > 0) uubuffer.MarkUsed(numRead);
      else if (numRead == 0) need_close = 1;       // connection closed

      AutoBuffer line;
      while (uubuffer.RemoveLine(line))
      {
        nothing_done = 0;

        if (strncmpi(line, "begin ", 6) == 0) gotuubegin = 1;
        else if (strncmpi(line, "POST ", 5) == 0) mode |= MODE_HTTP;
        else if (strncmpi(line, "end", 3) == 0)
        {
          gotuubegin = gothttpend = 0;
          httplength = 0;
          break;
        }
        else if (gotuubegin && line.GetLength() > 0)
        {
          // start decoding this single line
          char *p = line.GetHead();
          int n = UU_DEC(*p);
          for (++p; n > 0; p += 4, n -= 3)
          {
            char ch;
            if (n >= 3)
            {
              netbuffer.Reserve(3);
              ch = char((UU_DEC(p[0]) << 2) | (UU_DEC(p[1]) >> 4));
              netbuffer.GetTail()[0] = ch;
              ch = char((UU_DEC(p[1]) << 4) | (UU_DEC(p[2]) >> 2));
              netbuffer.GetTail()[1] = ch;
              ch = char((UU_DEC(p[2]) << 6) | UU_DEC(p[3]));
              netbuffer.GetTail()[2] = ch;
              netbuffer.MarkUsed(3);
            } else {
              netbuffer.Reserve(2);
              if (n >= 1) {
                ch = char((UU_DEC(p[0]) << 2) | (UU_DEC(p[1]) >> 4));
                netbuffer.GetTail()[0] = ch;
                netbuffer.MarkUsed(1);
              }
              if (n >= 2) {
                ch = char((UU_DEC(p[1]) << 4) | (UU_DEC(p[2]) >> 2));
                netbuffer.GetTail()[0] = ch;
                netbuffer.MarkUsed(1);
              }
            }
          }
        }
      } // while
    }
    else
    {
      // []--------------------------------------[]
      // |  Processing normal, unencoded packets  |
      // []--------------------------------------[]
      AutoBuffer tempbuffer;
      s32 wantedSize = ((mode & MODE_HTTP) && httplength) ? httplength : 500;
      tempbuffer.Reserve(wantedSize);

      s32 numRead = LowLevelGet(wantedSize, tempbuffer.GetTail());
      if (numRead > 0)
      {
        nothing_done = 0;
        tempbuffer.MarkUsed(numRead);

        // decrement from total if processing from a http body
        if ((mode & MODE_HTTP) && httplength)
        {
          httplength -= numRead;

          // our http only allows one packet per socket
          if (httplength == 0) nothing_done = gethttpdone = 1;
        }

        // see if we should upgrade to different mode
        if (tempbuffer.GetLength() >= 6 &&
            strncmpi(tempbuffer.GetHead(), "begin ", 6) == 0)
        {
          mode |= MODE_UUE;
          uubuffer = tempbuffer;
          gotuubegin = 0;
        }
        else if (tempbuffer.GetLength() >= 5 &&
            strncmpi(tempbuffer.GetHead(), "POST ", 5) == 0)
        {
          mode |= MODE_HTTP;
          uubuffer = tempbuffer;
          gothttpend = 0;
          httplength = 0;
        }
        else netbuffer += tempbuffer;
      }
      else if (numRead == 0) need_close = 1;
    }

    if (nothing_done)
    {
      if (need_close || gethttpdone) 
        break;
      #if (CLIENT_OS == OS_VMS) || (CLIENT_OS == OS_SOLARIS)
        sleep(1); // full 1 second due to so many reported network problems.
      #else
        usleep( 100000 );  // Prevent racing on error (1/10 second)
      #endif
    }
  } // while (netbuffer.GetLength() < blah)


  // transfer back what was read in
  u32 bytesfilled = (netbuffer.GetLength() < length ?
      netbuffer.GetLength() : length);
  memmove(data, netbuffer.GetHead(), (int) bytesfilled);
  netbuffer.RemoveHead(bytesfilled);

  if (need_close) Close();
  return bytesfilled;
}


//--------------------------------------------------------------------------

// returns -1 on error, or 0 on success
s32 Network::Put( u32 length, const char * data )
{
  AutoBuffer outbuf;

  // if the connection is closed, try to reopen it once.
  if ((sock == INVALID_SOCKET) || puthttpdone) 
    {
    if (Open()) 
      return -1;
    }

  if (mode & MODE_UUE)
    {
    /**************************/
    /***  Need to uuencode  ***/
    /**************************/
    outbuf += AutoBuffer("begin 644 query.txt\r\n");

    while (length > 0)
      {
      char line[80];
      char *b = line;

      int n = (int) (length > 45 ? 45 : length);
      length -= n;
      *b++ = UU_ENC(n);

      for (; n > 2; n -= 3, data += 3)
        {
        *b++ = UU_ENC((char)(data[0] >> 2));
        *b++ = UU_ENC((char)(((data[0] << 4) & 060) | ((data[1] >> 4) & 017)));
        *b++ = UU_ENC((char)(((data[1] << 2) & 074) | ((data[2] >> 6) & 03)));
        *b++ = UU_ENC((char)(data[2] & 077));
        }

      if (n != 0)
        {
        char c2 = (char)(n == 1 ? 0 : data[1]);
        char ch = (char)(data[0] >> 2);
        *b++ = UU_ENC(ch);
        *b++ = UU_ENC((char)(((data[0] << 4) & 060) | ((c2 >> 4) & 017)));
        *b++ = UU_ENC((char)((c2 << 2) & 074));
        *b++ = UU_ENC(0);
        }

      *b++ = '\r';
      *b++ = '\n';
      outbuf += AutoBuffer(line, b - line);
      }
    outbuf += AutoBuffer("end\r\n");
    }
  else
    {
    outbuf = AutoBuffer(data, length);
    }

  if (mode & MODE_HTTP)
    {
    char header[500];
    char ipbuff[64];
    if (lasthttpaddress) 
      {
      strncpy( ipbuff, __inet_ntoa__(lasthttpaddress), sizeof(ipbuff));
      #if 0      
        in_addr addr;
        addr.s_addr = lasthttpaddress;
        #if (CLIENT_OS == OS_WIN16)
          _fstrncpy(ipbuff, inet_ntoa(addr), sizeof(ipbuff));
        #else
          strncpy(ipbuff, inet_ntoa(addr), sizeof(ipbuff));
        #endif
      #endif
      } 
    else
     {
     strncpy(ipbuff, server_name, sizeof(ipbuff));
     }

    if ( httpid[0] ) 
      {
      sprintf(header, "POST http://%s:%li/cgi-bin/rc5.cgi HTTP/1.0\r\n"
         "Proxy-authorization: Basic %s\r\n"
         "Proxy-Connection: Keep-Alive\r\n"
         "Content-Type: application/octet-stream\r\n"
         "Content-Length: %lu\r\n\r\n",
         ipbuff, (long) lastport,
         httpid,
         (unsigned long) outbuf.GetLength());
      } 
    else 
      {
      sprintf(header, "POST http://%s:%li/cgi-bin/rc5.cgi HTTP/1.0\r\n"
         "Content-Type: application/octet-stream\r\n"
         "Content-Length: %lu\r\n\r\n",
         ipbuff, (long) lastport,
         (unsigned long) outbuf.GetLength());
      }
    #if (CLIENT_OS == OS_OS390)
      __etoa(header);
    #endif
    outbuf = AutoBuffer(header) + outbuf;
    puthttpdone = 1;
    }

  return (LowLevelPut(outbuf.GetLength(), outbuf) != -1 ? 0 : -1);
}

//------------------------------------------------------------------------

#define B64_ENC(Ch) (char) (base64table[(char)(Ch) & 63])
char * Network::base64_encode(char *username, char *password)
{
  static char in[80], out[80];
  static const char base64table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
                                    "ghijklmnopqrstuvwxyz0123456789+/";

  u32 length = strlen(username) + strlen(password) + 1;
  if ((length+1) >= sizeof(in) || (length + 2) * 4 / 3 >= sizeof(out)) 
    return NULL;
  sprintf(in, "%s:%s\n", username, password);

  char *b = out, *data = in;
  for (; length > 2; length -= 3, data += 3)
    {
    *b++ = B64_ENC(data[0] >> 2);
    *b++ = B64_ENC(((data[0] << 4) & 060) | ((data[1] >> 4) & 017));
    *b++ = B64_ENC(((data[1] << 2) & 074) | ((data[2] >> 6) & 03));
    *b++ = B64_ENC(data[2] & 077);
    }

  if (length == 1)
    {
    *b++ = B64_ENC(data[0] >> 2);
    *b++ = B64_ENC((data[0] << 4) & 060);
    *b++ = '=';
    *b++ = '=';
    }
  else if (length == 2)
    {
    *b++ = B64_ENC(data[0] >> 2);
    *b++ = B64_ENC(((data[0] << 4) & 060) | ((data[1] >> 4) & 017));
    *b++ = B64_ENC((data[1] << 2) & 074);
    *b++ = '=';
    }
  *b = 0;

  return out;
}

//=====================================================================
// From here on down we have "bottom half" functions that (a) use socket
// functions or (b) are TCP/IP specific or (c) actually do i/o.
//---------------------------------------------------------------------

int Network::GetHostName( char *buffer, unsigned int len )
{  
  if (!buffer)
    return -1;
  buffer[0]=0;
  if (!len)
    return -1;
  if (len >= sizeof( "127.0.0.1" ))
    strcpy( buffer, "127.0.0.1" );
  #if ( defined(_TIUSER_) || (defined(AF_INET) && defined(SOCK_STREAM)) )
  if (NetCheckIsOK())
    return gethostname(buffer, len);
  #endif
  return -1;
}  

/* ----------------------------------------------------------------------- */

int Network::LowLevelCreateSocket(void)
{
  Close(); //make sure the socket is closed already

  if (!NetCheckIsOK())
    return -1;

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
    #if (CLIENT_OS == OS_RISCOS)
      int on = 1;
      // allow blocking socket calls to preemptively multitask
      ioctl(sock, FIOSLEEPTW, &on);
    #endif
    return 0; //success
    }
  #endif
  sock = INVALID_SOCKET;
  return -1;
#endif
}

//------------------------------------------------------------------------

int Network::LowLevelCloseSocket(void)
{
#if defined( _TIUSER_ ) //TLI
   if ( sock != INVALID_SOCKET )
     {
     t_blocking( sock ); /* turn blocking back on */
     if ( t_getstate( sock )!= T_UNBND )
       {
       t_sndrel( sock );   /* initiate close */
       t_rcvrel( sock );   /* wait for conn release by peer */
       t_unbind( sock );   /* close our own socket */
       }
     int rc = t_close( sock );
     sock = INVALID_SOCKET;
     return rc;
     }
#else //BSD socks
   if ( sock != INVALID_SOCKET )
     {
     LowLevelConditionSocket( CONDSOCK_BLOCKING_ON );
     shutdown( sock, 2 );
     int retcode = (int)close( sock );
     sock = INVALID_SOCKET;
     return (retcode);
     }
#endif
   return 0;
}   

// -----------------------------------------------------------------------   

int Network::LowLevelConnectSocket( u32 that_address, u16 that_port )
{
  if ( sock == INVALID_SOCKET )
    return -1;
  if (!NetCheckIsOK())
    return -1;

#if defined(_TIUSER_)                                            //TLI
  if ( t_bind( sock, NULL, NULL ) == -1 )
    return -1;
  struct t_call *sndcall = (struct t_call *)t_alloc(sock, T_CALL, T_ADDR);
  if ( sndcall == NULL )
    return -1;
  sndcall->addr.len  = sizeof(struct sockaddr_in);
  sndcall->opt.len   = 0;
  sndcall->udata.len = 0;
  struct sockaddr_in *sin = (struct sockaddr_in *) sndcall->addr.buf;
  sin->sin_addr.s_addr = that_address;
  sin->sin_family = AF_INET;
  sin->sin_port = htons(that_port);
  int rc = t_connect( sock, sndcall, NULL);
  t_free((char *)sndcall, T_CALL);
  return rc;
#else
  #if (defined( SOL_SOCKET ) && defined( SO_REUSEADDR ) && defined(AF_INET))
    {
    #if 0                      // don't need this. No bind() in sight 
    // allow the socket to bind to "in use" ports
    int on = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char *)&on, sizeof(on));
    #endif

    // set up the address structure
    struct sockaddr_in sin;
    memset((void *) &sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_port = htons( that_port ); 
    sin.sin_addr.s_addr = that_address;

    if (connect(sock, (struct sockaddr *) &sin, sizeof(sin)) < 0)
      return -1;
    return 0;
    }
  #else //no socket support
  if (!that_address && !that_port)
    return -1;  
  #endif
#endif
  return -1;
}  

//------------------------------------------------------------------------

#if (!defined(AF_INET) || !defined(SOCK_STREAM))  
s32 Network::Resolve( const char *, u32 & )
{ return -1; }
#else
#include "netres.cpp"
#endif

// -----------------------------------------------------------------------   

// Returns length of sent data or -1 if error
s32 Network::LowLevelPut(u32 length, const char *data)  
{                                                       
  if ( sock == INVALID_SOCKET )
    return -1;
  if (!NetCheckIsOK())                                  
    return -1;
#if defined(_TIUSER_)                                          //TLI
  if (!length) return 0;
  int rc;
  do{
    rc = t_snd(sock, (char *)data, (unsigned int)length, 0);
    if ( rc == -1 && t_errno == TFLOW ) /* sending too fast */
      usleep(500000); // 0.5 secs
    } while ( rc == -1 && t_errno == TFLOW );
  return (s32)(rc);
#else                                                         //BSD sox
  u32 writelen = write(sock, (char*)data, length);
  return (s32) (writelen != length ? -1 : (s32)writelen);
#endif  
}

// ----------------------------------------------------------------------

// Returns length of read buffer. 0 if conn closed or -1 if no data waiting
s32 Network::LowLevelGet(u32 length, char *data)
{
  if ( sock == INVALID_SOCKET )
    return -1;
  if (!NetCheckIsOK())
    return -1;

#if defined(_TIUSER_)                                     //TLI
  int flags, rc;
  if (( rc = t_rcv( sock, data, length, &flags ) ) == -1 )
    {
    if ( t_errno == TNODATA ) 
      return (s32)(-1); 
    return 0; /* TLOOK (async event) or TSYSERR */
    }
  return (s32)(rc);
#else                                                     //BSD 4.3 sockets
  #if defined(SELECT_FIRST)
    fd_set rs;
    timeval tv = {0,0};
    FD_ZERO(&rs);
    FD_SET(sock, &rs);
    select(sock + 1, &rs, NULL, NULL, &tv);
    if (!FD_ISSET(sock, &rs)) return -1;
  #endif

  s32 numRead = read(sock, data, length);
  if (numRead < 0) numRead = -1;

  #if (CLIENT_OS == OS_HPUX)
    // HPUX incorrectly returns 0 on a non-blocking socket with
    // data waiting to be read instead of -1.
    if (numRead == 0) numRead = -1;
  #endif

  return numRead;
#endif //TLI or BSD sockets
}

// ----------------------------------------------------------------------

int Network::LowLevelConditionSocket( unsigned long cond_type )
{  
  if ( sock == INVALID_SOCKET )
    return -1;

  if ( cond_type==CONDSOCK_BLOCKING_ON || cond_type==CONDSOCK_BLOCKING_OFF )
    {
    #if defined(_TIUSER_)                                    //TLI
      if ( cond_type == CONDSOCK_BLOCKING_ON )
        return ( t_blocking( sock ) );
      else         /* same as an ioctl call with I_SETDELAY and 1 as args. */
        return ( t_nonblocking( sock ) );
    #elif (!defined(FIONBIO) && !(defined(F_SETFL) && (defined(FNDELAY) || defined(O_NONBLOCK))))
      return -1;
    #elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
      unsigned long flagon = ((cond_type == CONDSOCK_BLOCKING_OFF)?(1):(0));
      return ioctlsocket(sock, FIONBIO, &flagon);
    #elif ((CLIENT_OS == OS_VMS) && defined(__VMS_UCX__))
      // nonblocking sockets not directly supported by UCX
      // - DIGITAL's work around requires system privileges to use
      return -1;
    #elif ((CLIENT_OS == OS_VMS) && defined(MULTINET))
      unsigned long flagon = ((cond_type == CONDSOCK_BLOCKING_OFF)?(1):(0));
      return socket_ioctl(sock, FIONBIO, &flagon);
    #elif (CLIENT_OS == OS_RISCOS)
      int flagon = ((cond_type == CONDSOCK_BLOCKING_OFF) ? (1): (0));
      return ioctl(sock, FIONBIO, &flagon);
    #elif (CLIENT_OS == OS_OS2)
      int flagon = ((cond_type == CONDSOCK_BLOCKING_OFF) ? (1): (0));
      return ioctl(sock, FIONBIO, (char *) &flagon, sizeof(flagon));
    #elif (CLIENT_OS == OS_AMIGAOS)
      char flagon = ((cond_type == CONDSOCK_BLOCKING_OFF) ? (1): (0));
      return IoctlSocket(sock, FIONBIO, &flagon);
    #elif (defined(F_SETFL) && (defined(FNDELAY) || defined(O_NONBLOCK)))
      {
      int flag, res, arg;
      #if (defined(FNDELAY))
        flag = FNDELAY;
      #else
        flag = O_NONBLOCK;
      #endif
      arg = ((cond_type == CONDSOCK_BLOCKING_OFF) ? (flag): (0) );

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
    #else
      return -1;
    #endif
    }
  return -1;
}

// ----------------------------------------------------------------------
