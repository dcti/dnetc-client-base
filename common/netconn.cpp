/*
 * This module contains netconn_xxx() and support routines which are
 * high level connection open/close/read/write/reset with optional
 * on-the-fly http/uue tunneling and http/socks proxy support.
 *
 * for use primarily for connection to keyservers, but are flexible 
 * enough for other uses as well, and may be preferred since they 
 * are verbose (print error messages), take care of ^C checking and
 * are (from the caller's perspective) fully blocking.
 *
 * Written October 2000 by Cyrus Patel <cyp@fb14.uni-mainz.de>
 * based on lessons learned from the old Network class methods
 *
*/
const char *netconn_cpp(void) {
return "@(#)$Id: netconn.cpp,v 1.1.2.9 2001/03/28 08:58:18 cyp Exp $"; }

//#define TRACE
//#define DUMP_PACKET

#include "cputypes.h"
#include "baseincs.h"  // standard stuff
#include "logstuff.h"  // LogScreen()
#include "base64.h"    // base64_encode()
#include "clitime.h"   // CliTimeGetMinutesWest()
#include "triggers.h"  // CheckExitRequestTrigger()
#include "util.h"      // trace
#include "netbase.h"   // net_*() primitives
#include "netconn.h"   // thats us

#if (CLIENT_OS == OS_QNX) || (CLIENT_OS == OS_RISCOS)
  #undef offsetof
  #define offsetof(__typ,__id) ((size_t)&(((__typ*)0)->__id))
#else
  #include <stddef.h> /* offsetof */
#endif

/* modebits for _enctype parameter to netconn_open */
#define MODE_UUE              0x01 /* encodingtype */
#define MODE_HTTP             0x02 /* encodingtype and proxytype */
#define MODE_SOCKS4           0x04 /* proxytype */
#define MODE_SOCKS5           0x08 /* proxytype */

#define OCTET(__ch) ((char)(__ch))
/* unlike most UU macros, the following are net portable - they work on net */
/* data (which is always ascii) even when the host's charset is ebcdic, and */
/* work even if 'chars' are always promoted to 'int'.  The basic ascii host */
/* forms are ENC(c) => (((c) & 077) + ' '), DEC(c) => (((c)-' ')&077) */
#define UU_DEC(__ch) OCTET((OCTET(__ch) - 040) & 077)                         
#define UU_ENC(__ch) OCTET(((OCTET(__ch) & 077) != 0)?((OCTET(__ch) & 077)+040):(0140))

#if (CLIENT_OS == OS_OS390)
  #define __hton_str(__ebcdic_string) __etoa(__ebcdic_string)
  #define __ntoh_str(__ascii_string)  __atoe(__ascii_string)
#else
  #define __hton_str(__native_string) /* nothing */
  #define __ntoh_str(__ascii_string)  /* nothing */
#endif

/* ==================================================================== */

typedef struct
{
  char *mem;
  unsigned int size;
  unsigned int used;
} NETSPOOL;

typedef struct 
{
  long magic;             // signature. destroyed when state is unusable.
  int  verbose_level;     // currently, only 0 == no messages
  int  iotimeout;         // use blocking calls if iotimeout is <0
  int  startmode;         // initial (user preference) encoding/proxy flags
  int  break_pending;     // if ^Cs was seen at start, don't do ^C checks

  int  mode;              // startmode as modified at runtime
  SOCKET sock;            // socket file handle
  int  connection_closed; // connection is closed (not the same as closed sock)
  int  reconnected;       // set to 1 once a connect succeeds 
  int  shown_connection;

  char fwall_hostname[64]; //intermediate
  int  fwall_hostport;
  u32  fwall_hostaddr;
  char fwall_userpass[128]; //username+password

  #define MAX_SVC_HOSTNAMES 8
  char servername_buffer[256];
  char *servername_ptrs[MAX_SVC_HOSTNAMES]; 
  int servername_ports[MAX_SVC_HOSTNAMES];
  unsigned int servername_count;
  unsigned servername_selector;
  #undef MAX_SVC_HOSTNAMES

  u32  resolve_addrlist[32]; //list of resolved (proxy) addresses
  int  resolve_addrcount;    //number of addresses in there. <0 if uninitialized

  const char *svc_hostname; // name of the final dest (server_name or rrdns_name)
  int  svc_hostport;   // the port of the final destination
  u32  svc_hostaddr;   // resolved if direct connection or socks.

  const char *conn_hostname; // hostname we connect()ing to (fwall or server)
  int  conn_hostport;  // port we are connect()ing to
  u32  conn_hostaddr;  // the address we are connect()ing to
  int  local_hostport; // the port we are connect()ing from
  u32  local_hostaddr; // the address we are connect()ing from

  NETSPOOL netbuffer;  // "persistant" storage decoding HTTP/UUE stream
  int puthttpdone;     // reset the connection before every http ::Put

} NETSTATE;

#define NETSTATE_MAGIC ((((((('n')<<8)|('e'))<<8)|('s'))<<8)|('t'))

static NETSTATE *__cookie2netstate(void *cookie)
{
  NETSTATE *netstate = ((NETSTATE *)cookie);
  if (netstate)
  {
    if (netstate->magic != NETSTATE_MAGIC)
      netstate = (NETSTATE *)0;
  }
  return netstate;
}

static int __break_check(NETSTATE *netstate) 
{
  /* if break was already seen at netconn_open() time, 
     then do any more break checks. otherwise final 
     mail and nodiskbuffer flush won't work.
  */
  if (!netstate->break_pending)
    return CheckExitRequestTrigger();
  return 0;
}

/* ==================================================================== */

#ifndef DUMP_PACKET
#define DUMP_PACKET( __ctx, __pkt, __len ) /* nothing */
#else
#undef DUMP_PACKET
// Logs a hexadecimal dump of the specified raw buffer of data.
static void DUMP_PACKET( const char *label, const char *apacket, 
                                            unsigned int alen )
{
  unsigned int i;
  for (i = 0; i < alen; i += 16)
  {
    char buffer[128];
    char *p, *q; unsigned int n;
    sprintf(buffer,"%s %04x: ", label, i );
    q = 48 + (p = &buffer[strlen(buffer)]);
    for (n = 0; n < 16; n++)
    {
      unsigned int c = ' ';
      p[0] = p[1] = ' ';
      if (( n + i ) < alen )
      {
        static const char *tox="0123456789abcdef";
        c = (((unsigned int)apacket[n+i]) & 0xff);
        p[0] = (char)tox[c>>4];
        p[1] = (char)tox[c&0x0f];
        if (!isprint(c) || /*isctrl(c)*/ c=='\r' || c=='\n' || c=='\t')
          c = '.';
      }
      p+=2;
      *p++ = ' ';
      *q++ = (char)c;
    }
    *q = '\0';
    #if defined(TRACE)
    TRACE_OUT((0,"\n%s\n",buffer));
    #else
    LogRaw("\n%s\n",buffer);
    #endif
  }
  #if defined(TRACE)
  TRACE_OUT((0,"%s total len: %d\n",label, alen));
  #else
  LogRaw("%s total len: %d\n",label, alen);
  #endif
}
#endif

/* ====================================================================== */

// clears all flags and communication buffers and mark connection
// as closed. Does not actually close the socket (which is done by 
// __open_connection() by way of reset(), or by the final net_close())
static int __close_connection(void *cookie)
{
  NETSTATE *netstate = __cookie2netstate(cookie);
  if (!netstate)
    return -1;

  TRACE_OUT((+1,"__close_connection()\n"));

  netstate->puthttpdone = 0;
  netstate->netbuffer.used = 0;

  netstate->connection_closed = 1;

  TRACE_OUT((-1,"__close_connection() => 0\n"));
  return 0;
}

/* ====================================================================== */

#ifndef MIPSpro
# pragma pack(1)               // no padding allowed
#endif /* ! MIPSpro */
// SOCKS4 protocol spec:  http://www.socks.nec.com/protocol/socks4.protocol
typedef struct _socks4 {
    unsigned char VN;           // version == 4
    unsigned char CD;           // command code, CONNECT == 1
    u16 DSTPORT;                // destination port, network order
    u32 DSTIP;                  // destination IP, network order
    char USERID[1];             // variable size, null terminated
} SOCKS4;
// SOCKS5 protocol RFC:  http://www.socks.nec.com/rfc/rfc1928.txt
// SOCKS5 authentication RFC:  http://www.socks.nec.com/rfc/rfc1929.txt
typedef struct _socks5methodreq {
    unsigned char ver;          // version == 5
    unsigned char nMethods;     // number of allowable methods following
    unsigned char Methods[2];   // this program will request at most two
} SOCKS5METHODREQ;
typedef struct _socks5methodreply {
    unsigned char ver;          // version == 1
    unsigned char Method;       // server chose method ...
    char end;
} SOCKS5METHODREPLY;
typedef struct _socks5userpwreply {
    unsigned char ver;          // version == 1
    unsigned char status;       // 0 == success
    char end;
} SOCKS5USERPWREPLY;
typedef struct _socks5 {
    unsigned char ver;          // version == 5
    unsigned char cmdORrep;     // cmd: 1 == connect, rep: 0 == success
    unsigned char rsv;          // reserved, must be 0
    unsigned char atyp;         // address type (IPv4 == 1, fqdn == 3)
    u32 addr;                   // network order
    u16 port;                   // network order
    char end;
} SOCKS5;
#ifndef MIPSpro
# pragma pack()
#endif /* ! MIPSpro */
const char *Socks5ErrorText[9] =
{
   /* 0 */ "" /* success */,
    "general SOCKS server failure",
    "connection not allowed by ruleset",
    "Network unreachable",
    "Host unreachable",
    "Connection refused",
    "TTL expired",
    "Command not supported",
    "Address type not supported"
};

/* ====================================================================== */

//returns zero on success, <0 on fatal err, >0 on recoverable error (try again)
//only called from __open_connection()
static int __init_connection(NETSTATE *netstate)
{
  const char *proto_init_error_msg = "protocol initialization error";
  int rc = 0;
  char socksreq[600];  // SOCKS5: sizeof(SOCKS5)+large username/pw (255 max each)
                       // SOCKS4: sizeof(SOCKS4)+passwd+hostname

  TRACE_OUT((+1,"__init_connection()\n"));

  // set communications settings
  netstate->puthttpdone = 0;
  netstate->netbuffer.used = 0;

  if (netstate->startmode & MODE_SOCKS5)
  {
    int success, recoverable;
    unsigned int len, len2;
    SOCKS5METHODREQ *psocks5mreq = (SOCKS5METHODREQ *)socksreq;
    SOCKS5METHODREPLY *psocks5mreply = (SOCKS5METHODREPLY *)socksreq;
    SOCKS5USERPWREPLY *psocks5userpwreply = (SOCKS5USERPWREPLY *)socksreq;
    SOCKS5 *psocks5 = (SOCKS5 *)socksreq;
    
    success = 0; //assume failed
    recoverable = 0; //assume non-recoverable error (negotiation failure)

    // transact a request to the SOCKS5 proxy requesting
    // authentication methods.  If the username/password
    // is provided we ask for no authentication or user/pw.
    // Otherwise we ask for no authentication only.

    psocks5mreq->ver = 5;
    psocks5mreq->nMethods = (unsigned char) (netstate->fwall_userpass[0] ? 2 : 1);
    psocks5mreq->Methods[0] = 0;  // no authentication
    psocks5mreq->Methods[1] = 2;  // username/password

    int authaccepted = 0;

    len = len2 = 2 + psocks5mreq->nMethods;
    rc = net_write(netstate->sock,socksreq,&len,netstate->conn_hostaddr,netstate->conn_hostport,netstate->iotimeout);
    if (rc != 0)
    {
      if (netstate->verbose_level > 0)
        LogScreen("SOCKS5: %s\n%s\n", proto_init_error_msg, net_strerror(rc,netstate->sock));
      recoverable = 0;
    }
    else if (len != len2)
    {
      if (netstate->verbose_level > 0)
        LogScreen("SOCKS5: error sending negotiation request\n");
      recoverable = 1;
    }
    else 
    {
      len = len2 = 2;
      rc = net_read(netstate->sock,socksreq,&len,netstate->conn_hostaddr,netstate->conn_hostport,netstate->iotimeout);
      if (rc != 0)
      {
        if (netstate->verbose_level > 0)
          LogScreen("SOCKS5: %s\n%s\n", proto_init_error_msg, net_strerror(rc,netstate->sock));
        recoverable = 0;
      }
      else if (len != len2)
      {
        if (netstate->verbose_level > 0)
          LogScreen("SOCKS5: failed to get negotiation request ack.\n");
        recoverable = 1;
      }
      else if (psocks5mreply->ver != 5)
      {
        if (netstate->verbose_level > 0)
          LogScreen("SOCKS5: authentication has wrong version, %d should be 5\n",
                              psocks5mreply->ver);
      }
      else if (psocks5mreply->Method == 0)       // no authentication required
      {
        // nothing to do for no authentication method
        authaccepted = 1;
      }
      else if (psocks5mreply->Method == 1)  // GSSAPI
      {
        if (netstate->verbose_level > 0)
          LogScreen("SOCKS5: GSSAPI per-message authentication is\n"
                    "not supported. Please use SOCKS4 or HTTP.\n");
      }
      else if (psocks5mreply->Method == 2)  // username and pw
      {
        char username[255];
        char password[255];
        char *pchSrc, *pchDest;
        int userlen, pwlen;

        pchSrc = netstate->fwall_userpass;
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

        len2 = len;
        rc = net_write(netstate->sock,socksreq,&len,netstate->conn_hostaddr,netstate->conn_hostport,netstate->iotimeout);
        if (rc != 0)
        {
          if (netstate->verbose_level > 0)
            LogScreen("SOCKS5: %s\n%s\n", proto_init_error_msg, net_strerror(rc,netstate->sock));
          recoverable = 0;
        }
        else if (len != len2)
        {
          if (netstate->verbose_level > 0)
            LogScreen("SOCKS5: failed to send sub-negotiation request.\n");
          recoverable = 1;
        }
        else
        {
          len = len2 = 2;
          rc = net_read(netstate->sock,socksreq,&len,netstate->conn_hostaddr,netstate->conn_hostport,netstate->iotimeout);
          if (rc != 0)
          {
            if (netstate->verbose_level > 0)
              LogScreen("SOCKS5: %s\n%s\n", proto_init_error_msg, net_strerror(rc,netstate->sock));
            recoverable = 0;
          }
          else if (len != len2)
          {
            if (netstate->verbose_level > 0)
              LogScreen("SOCKS5: failed to get sub-negotiation response.\n");
            recoverable = 1;
          }
          else if (psocks5userpwreply->ver != 1 ||
              psocks5userpwreply->status != 0)
          {
            if (netstate->verbose_level > 0)
              LogScreen("SOCKS5: user %s rejected by server.\n", username);
            recoverable = 0;
          }
          else
          {
            authaccepted = 1;
          }
        } /* else if (psocks5mreply->Method == 2)  // username and pw */
      } 
      else //if (psocks5mreply->Method > 2)
      {
        if (netstate->verbose_level > 0)
          LogScreen("SOCKS5 authentication method rejected.\n");
      }
    }

    if (authaccepted)
    {
      // after subnegotiation, send connect request
      psocks5->ver = 5;
      psocks5->cmdORrep = 1;   // connnect
      psocks5->rsv = 0;   // must be zero
      psocks5->atyp = 1;  // IPv4 = 1
      psocks5->addr = netstate->svc_hostaddr;
      psocks5->port = (u16)htons((u16)netstate->svc_hostport); //(u16)(htons((server_name[0]!=0)?((u16)port):((u16)(DEFAULT_PORT))));
      unsigned int packetsize = 10;

      if (netstate->svc_hostaddr == 0)
      {
        psocks5->atyp = 3; //fully qualified domainname
        char *p = (char *)(&psocks5->addr);
        // at this point netstate->svc_hostname is a ptr to a resolve_hostname.
        strcpy( p+1, netstate->svc_hostname );
        *p = (char)(len = strlen( p+1 ));
        p += (++len);
        u16 xx = (u16)htons((u16)netstate->svc_hostport);
        *(p+0) = *(((char*)(&xx)) + 0);
        *(p+1) = *(((char*)(&xx)) + 1);
        packetsize = (10-sizeof(u32))+len;
      }

      len = packetsize;
      rc = net_write(netstate->sock,socksreq,&len,netstate->conn_hostaddr,netstate->conn_hostport,netstate->iotimeout);
      if (rc != 0)
      {
        if (netstate->verbose_level > 0)
          LogScreen("SOCKS5: %s\n%s\n", proto_init_error_msg, net_strerror(rc,netstate->sock));
        recoverable = 0;
      }
      else if (len != packetsize)
      {
        if (netstate->verbose_level > 0)
          LogScreen("SOCKS5: failed to send connect request.\n");
        recoverable = 1;
      }
      else 
      {
        len = packetsize;
        rc = net_read(netstate->sock,socksreq,&len,netstate->conn_hostaddr,netstate->conn_hostport,netstate->iotimeout);
        if (rc != 0)        
        {
          if (netstate->verbose_level > 0)
            LogScreen("SOCKS5: %s\n%s\n", proto_init_error_msg, net_strerror(rc,netstate->sock));
          recoverable = 0;
        }
        else if (len < 10) /* too small for either atyp */
        {
          if (netstate->verbose_level > 0)
            LogScreen("SOCKS5: failed to get connect request ack.\n");
          recoverable = 1;
        }
        else if (psocks5->ver != 5)
        {
          if (netstate->verbose_level > 0)
             LogScreen("SOCKS5: reply has wrong version, %d should be 5\n",
                       psocks5->ver);
        }
        else if (psocks5->cmdORrep == 0)  // 0 is successful connect
        {
          success = 1;
          if (psocks5->atyp == 1)  // IPv4
            netstate->svc_hostaddr = psocks5->addr;
        }
        else if (netstate->svc_hostaddr != 0 /* we used an IP address */
             && (psocks5->cmdORrep == 3 || /*"Network unreachable"*/
                 psocks5->cmdORrep == 4 || /*"Host unreachable"   */
                 psocks5->cmdORrep == 5))  /*"Connection refused"*/
        {
          recoverable = 1; /* retry using a different IP address */
          if (netstate->verbose_level > 0)
          {
            const char *failcause = "refused.";
            if (psocks5->cmdORrep == 3)
              failcause = "failed. (network unreachable)";
            else if (psocks5->cmdORrep == 4)
              failcause = "failed. (host unreachable)";
            LogScreen("SOCKS5: connect to %s:%u %s\n",
                        net_ntoa(netstate->svc_hostaddr),
                        (unsigned int)netstate->svc_hostport, 
                        failcause );
          }
        }
        else if (netstate->verbose_level > 0)
        {
          const char *p = ((psocks5->cmdORrep >=
                           (sizeof Socks5ErrorText / sizeof Socks5ErrorText[0]))
                           ? ("") : (Socks5ErrorText[ psocks5->cmdORrep ]));
          LogScreen("SOCKS5: server error 0x%02x%s%s%s\n"
                    "connecting to %s:%u\n",
                   ((int)(psocks5->cmdORrep)),
                   ((*p) ? (" (") : ("")), p, ((*p) ? (")") : ("")),
                   netstate->svc_hostname, 
                   (unsigned int)netstate->svc_hostport );
        }
      }
    } //if (authaccepted)

    rc = ((success) ? (0) : ((recoverable) ? (+1) : (-1)));
  } //if (netstate->startmode & MODE_SOCKS5)

  else if (netstate->startmode & MODE_SOCKS4)
  {
    int success, recoverable;
    unsigned int len, len2;
    SOCKS4 *psocks4 = (SOCKS4 *)socksreq;

    success = 0; //assume failed
    recoverable = 0; //assume non-recoverable error (negotiation failure)

    // transact a request to the SOCKS4 proxy giving the
    // destination ip/port and username and process its reply.

    //+----+----+----+----+----+----+----+----+----+----+....+----+
    //| VN | CD | DSTPORT |      DSTIP        | USERID       |NULL|
    //+----+----+----+----+----+----+----+----+----+----+....+----+
    //  1    1      2              4           variable       1

    psocks4->VN = 4;
    psocks4->CD = 1;  // CONNECT
    psocks4->DSTPORT = (u16)htons((u16)netstate->svc_hostport); //(u16)htons((server_name[0]!=0)?((u16)port):((u16)DEFAULT_PORT));
    psocks4->DSTIP = netstate->svc_hostaddr;
    strcpy( psocks4->USERID, netstate->fwall_userpass );
    len = (sizeof(*psocks4) - 1) + strlen(psocks4->USERID) + 1;

    if (psocks4->DSTIP == 0) /* no IP address - so use hostname (socks4a) */
    {
      //+----+----+----+----+----+----+----+----+----+----+....+----+----+----+....+----+
      //| VN | CD | DSTPORT |  0    0    0   x  | USERID       |NULL| HOSTNAME     |NULL|
      //+----+----+----+----+----+----+----+----+----+----+....+----+----+----+....+----+
      //  1    1      2              4           variable       1    variable        1

      ((char *)&(psocks4->DSTIP))[3] = 1; /* address == 0.0.0.x */
      strcpy( &(psocks4->USERID[strlen(psocks4->USERID)+1]),
              netstate->svc_hostname );
      len += strlen( netstate->svc_hostname ) + 1;
    }      

    len2 = len;
    rc = net_write(netstate->sock,socksreq,&len,netstate->conn_hostaddr,netstate->conn_hostport,netstate->iotimeout);
    if (rc != 0)
    {
      if (netstate->verbose_level > 0)
        LogScreen("SOCKS4: %s\n%s\n", proto_init_error_msg, net_strerror(rc,netstate->sock));
      recoverable = 0;
    }
    else if (len2 != len)
    {
      if (netstate->verbose_level > 0)
        LogScreen("SOCKS4: Error sending connect request\n");
      recoverable = 1;
    }
    else
    {
      len = sizeof(*psocks4) - 1;  // - 1 for the USERID[1]
      len2 = len;

      rc = net_read(netstate->sock,socksreq,&len,netstate->conn_hostaddr,netstate->conn_hostport,netstate->iotimeout);
      if (rc != 0)
      {
        if (netstate->verbose_level > 0)
          LogScreen("SOCKS4: %s\n%s\n", proto_init_error_msg, net_strerror(rc,netstate->sock));
        recoverable = 0;
      }
      else if (len2 != len )
      {
        if (netstate->verbose_level > 0)
          LogScreen("SOCKS4:%s response from server.\n",
                                     ((len==0)?("No"):("Invalid")));
        recoverable = 1;
      }
      else
      {
        if (psocks4->VN == 0 && psocks4->CD == 90) // 90 is successful return
        {
          success = 1;
        }
        else if (netstate->verbose_level > 0)
        {
          LogScreen("SOCKS4: request rejected%s.\n",
            (psocks4->CD == 91)
             ? " or failed"
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

    rc = ((success) ? (0) : ((recoverable) ? (+1) : (-1)));
  }

  TRACE_OUT((-1,"__init_connection() => %d\n", rc));
  return rc;
}

/* ====================================================================== */

// Initiates a connection opening sequence, and performs the initial
// negotiation for SOCKS4/SOCKS5 connections.
// returns -1 on error, 0 on success

static int __open_connection(void *cookie)
{
  NETSTATE *netstate = __cookie2netstate(cookie);
  int whichtry, maxtries;
  int return_code = -1;

  if (!netstate)
    return -1;

  netstate->puthttpdone = 0;
  netstate->netbuffer.used = 0;
  netstate->mode = netstate->startmode;

  TRACE_OUT((+1,"__open_connection()\n"));

  whichtry = 0, maxtries = 0; /* initially */
  /* The total possible number of attempts will be H * A, (max is 8 * 32),
     where 'H' is the number of hostnames in servername_buffer including
     (possibly multiple) auto-select-keyservers, and/or fallback hostname,
     and 'A' is the number of addresses that were resolved for each 
     hostname in turn, (or if resolution failed and using http or socks5 
     proxy, then the hostname itself)
  */
  for (;;)
  {
    int rc, success = 1;
    int is_fwalled = ((netstate->fwall_hostname[0] != 0) &&
                     ((netstate->startmode & (MODE_HTTP|MODE_SOCKS4|MODE_SOCKS5))!=0) );

    if (__break_check(netstate)) 
    {
      success = 0;
      maxtries = -1; /* we'll want to break out of the loop below */
    }

    TRACE_OUT((0,"whichtry=%d, maxtries=%d, netstate->reconnected=%d, svc='%s'\n", whichtry, maxtries, netstate->reconnected,netstate->svc_hostname));

    /* ---------- create a new socket --------------- */

    if (success)
    {
      SOCKET newsock;
      rc = net_open( &newsock, 0, 0 );
      if (rc != 0)
      {
        newsock = INVALID_SOCKET;
        if (netstate->verbose_level > 0)
        {
          LogScreen("Unable to create network endpoint\n%s\n",
                    net_strerror( rc, newsock) );
        }
        success = 0;
        maxtries = -1; /* no point retrying if we can't create a socket */
      }
      else /* success */
      {
        if (netstate->sock != INVALID_SOCKET)
          net_close(netstate->sock);
        netstate->sock = newsock;
      }
    }

    /* --- resolve the firewall hostname --- */

    if (success && is_fwalled)
    {
      TRACE_OUT((+1,"resolve netstate->fwall_hostname\n"));
      if (netstate->fwall_hostaddr == 0) /* no address for firewall yet */
      {
        unsigned int count;
        if (!netstate->reconnected && netstate->verbose_level > 0)
          LogScreen( "\rAttempting to resolve %s...", netstate->fwall_hostname );

        count = 1;
        rc = net_resolve( netstate->fwall_hostname, &netstate->fwall_hostaddr, &count);
        if (rc != 0)
        {
          success = 0;
          netstate->fwall_hostaddr = 0;
          if (netstate->verbose_level > 0)
            LogScreen("\rNet::failed to resolve name \"%s\"\n%s\n",
                      netstate->fwall_hostname, net_strerror( rc, netstate->sock ) );
           maxtries = -1; // unrecoverable error. retry won't help
        }
      }
      TRACE_OUT((-1,"resolve netstate->fwall_hostname = %s\n",net_ntoa(netstate->fwall_hostaddr)));
      netstate->conn_hostaddr = netstate->fwall_hostaddr;
      netstate->conn_hostname = netstate->fwall_hostname;
      netstate->conn_hostport = netstate->fwall_hostport;
    }

    /* --- resolve the service hostname --- */

    if (success)
    {
      TRACE_OUT((+1,"resolve netstate->svc_hostname\n"));
      if ((netstate->startmode & MODE_HTTP) == 0)
      {
        netstate->svc_hostaddr = 0; /* always pick another hostaddress unless http. */
      }    
      if (!netstate->reconnected || netstate->svc_hostaddr == 0)
      {
        /* target address resolution is optional when using socks4/5/http
        ++ (although such support is not defined in the original socks4 spec,
        ++ it is supported in the socks4a (SOCKS 4.3) protocol, which is what
        ++ we implement here (with backwards compatibility)
        ++ socks 4: http://www.socks.nec.com/protocol/socks4.protocol
        ++ socks 4a: http://www.socks.nec.com/protocol/socks4a.protocol
        */
        if (netstate->resolve_addrcount < 1)
        {
          unsigned int count;
          if (!netstate->reconnected && netstate->verbose_level > 0)
            LogScreen( "\rAttempting to resolve '%s'...",netstate->svc_hostname );

          count = sizeof(netstate->resolve_addrlist)/sizeof(netstate->resolve_addrlist[0]);
          rc = net_resolve( netstate->svc_hostname, &netstate->resolve_addrlist[0], &count);
          if (rc == 0)
          {
            maxtries = netstate->resolve_addrcount = count;
          }
          else if (!is_fwalled) //resolution _must_ complete only if not fwalled
          {                      
            success = 0; 
            netstate->svc_hostaddr = 0;
            if (!netstate->reconnected && netstate->verbose_level > 0)
              LogScreen("\rNet::failed to resolve name \"%s\"\n%s\n",
                         netstate->svc_hostname, net_strerror( rc, netstate->sock ) );
          }
        }
        if (netstate->resolve_addrcount > 0)
        {
          netstate->svc_hostaddr = netstate->resolve_addrlist[whichtry % netstate->resolve_addrcount];
        }
      }
      TRACE_OUT((-1,"resolve netstate->svc_hostname = %s\n",net_ntoa(netstate->svc_hostaddr)));
      if (!is_fwalled) /* otherwise conn_* settings are already set */
      {
        netstate->conn_hostaddr = netstate->svc_hostaddr;
        netstate->conn_hostname = netstate->svc_hostname;
        netstate->conn_hostport = netstate->svc_hostport;
      }
    }

    /* ------ connect ------- */

    if (success)
    {
      if (!netstate->reconnected && netstate->verbose_level > 0)
      {
        LogScreen( "\rConnecting to %s:%u...",
               ((netstate->conn_hostaddr)?(net_ntoa(netstate->conn_hostaddr)):(netstate->conn_hostname)),
             (unsigned int)(netstate->conn_hostport) );
      }
      netstate->local_hostaddr = 0;
      netstate->local_hostport = 0;
      rc = net_connect(netstate->sock, 
                       &(netstate->conn_hostaddr), 
                       &(netstate->conn_hostport), 
                       &(netstate->local_hostaddr),
                       &(netstate->local_hostport),
                       netstate->iotimeout );
      if (rc != 0)
      {
        if (netstate->verbose_level > 0 && !__break_check(netstate))
        {
          LogScreen( "%sonnect to host %s:%u failed.\n%s\n",
                     ((netstate->reconnected)?("Rec"):("\rC")),
                     net_ntoa(netstate->conn_hostaddr),
                     (unsigned int)(netstate->conn_hostport), 
                     net_strerror(rc, netstate->sock) );
        }
        success = 0; 
      }
    }

    /* ---- initialize the connection ---- */

    if (success)   /* connect succeeded */
    {
      TRACE_OUT((+1,"initialize_connection\n"));
      rc = __init_connection( netstate );
      TRACE_OUT((-1,"initialize_connection =>%d\n",rc));
      if (rc == 0)
      {
        if (netstate->verbose_level > 0 && !netstate->shown_connection)
        {
          if (!is_fwalled)
          {
            LogScreen("\rConnected to %s:%u...\n", netstate->svc_hostname,
                  ((unsigned int)(netstate->svc_hostport)) );
          }
          else
          {
            LogScreen( "\rConnected to %s:%u\nvia %s proxy %s:%u\n",
                       netstate->svc_hostname, ((unsigned int)(netstate->svc_hostport)),
                       ((netstate->startmode & MODE_SOCKS5)?("SOCKS5"):
                       ((netstate->startmode & MODE_SOCKS4)?("SOCKS4"):("HTTP"))),
                       netstate->fwall_hostname, (unsigned int)netstate->fwall_hostport );
          }
          netstate->shown_connection = 1;
        }
        netstate->connection_closed = 0;
        netstate->reconnected = 1;
        return_code = 0;
        break;
      }
      success = 0;
      if (rc < 0)           /* unrecoverable error (negotiation failure) */
        maxtries = -1; /* so don't retry */
    }

    /* ----------------------------------- */

    if (maxtries < 0) /* don't retry */
    {
      break;
    }
    if ((++whichtry) < maxtries)
    {
      ; /* nothing - proceed normally */
    }
    else if (netstate->reconnected)
    {
      break; /* no sense skipping to next */
    }
    else if ((++netstate->servername_selector) >= netstate->servername_count)
    {
      netstate->servername_selector = 0;
      TRACE_OUT((0,"no more hostnames to try\n"));
      break;
    }
    else
    {
      netstate->resolve_addrcount = -1; /* need to re-resolve */
      whichtry = maxtries = 0;
      netstate->svc_hostname = netstate->servername_ptrs[netstate->servername_selector];
      netstate->svc_hostport = netstate->servername_ports[netstate->servername_selector];
      TRACE_OUT((0,"changed hostname to '%s:%d'\n", netstate->svc_hostname,netstate->svc_hostport));
    }

  } /* for (;;) */

  TRACE_OUT((-1,"__open_connection() => %d\n", return_code));
  return return_code;
}

/* ====================================================================== */

/* returns number of chars to skip to nextline, or zero if the current line */
/* wasn't eol terminated, or -1 if error (data is binary) */
/* if the netline was eol terminated, then copies first 'n' octets to linebuf*/
/* The returned 'linebuf' is always lower case, always native charset */
/* (ascii/ebcdic) and always '\0' terminated. */
static int __peek_netline(const char *bufp, unsigned int buflen, 
                          unsigned int *asc_linelen, 
                          char *linebuf, unsigned int linebufsz )
{
  unsigned int linelen = 0, eolcount = 0;
  while (linelen < buflen)
  {
    if (bufp[linelen] == 0x0d) /* ascii '\r' */
    {
      if (linelen == (buflen-1)) /* not enough space for '\n' */
        break;
      eolcount++;
      if (bufp[linelen+1] == 0x0a) /* ascii '\n' */
        eolcount++;
      break;
    }
    if (bufp[linelen] == 0x0a) /* ascii '\n' */
    {
      eolcount++;
      break;      
    }
    if (bufp[linelen] < 0x20)  /* ascii ctrl chars */
    {                          /* should we check for 8th bit set too? */
      return -1; /* have binary data */
    }
    linelen++;
  }
  if (asc_linelen)
  {
    *asc_linelen = linelen;
  }
  if (eolcount && linebuf && linebufsz)
  {
    if (!linelen)
      *linebuf = '\0';
    else
    {
      unsigned int pos;
      linebufsz--;
      if (linebufsz > linelen)
        linebufsz = linelen;
      memcpy(linebuf, bufp, linebufsz);
      linebuf[linebufsz] = 0;  
      __ntoh_str( linebuf );
      for (pos = 0; linebuf[pos]; pos++)
        linebuf[pos] = (char)tolower(linebuf[pos]);
    }
  }
  if (!eolcount)
    return 0;
  return linelen+eolcount;
}

/* detect either HTTP *or* UUE on the head of a raw network stream, */
/* or return -1 if there isn't sufficient data to make that determination */
static int auto_sense_http_uue(const char *netdata, unsigned int netdatalen)
{ 
  int read_mode = -1; /* if failed */
  if (netdatalen >= 6) /* enough data to make make a determination? */
  {
    unsigned int pos;
    char scratch[10];
    memcpy( scratch, netdata, 6 );
    scratch[6] = 0; /* binary */
    __ntoh_str( scratch );

    read_mode = 0;
    if ( memcmp( scratch, "HTTP/1", 6 ) == 0 )
      read_mode = MODE_HTTP; /* note: '=' not '|=' */
    else if ( memcmp( scratch, "begin ", 6 ) == 0 )
      read_mode = MODE_UUE; /* note: '=' not '|=' */
    else
    {
      for (pos = 0; pos < 6; pos++)
        scratch[pos] = (char)tolower(scratch[pos]);
      if ( memcmp( scratch, "http/1", 6 ) == 0 )
        read_mode = MODE_HTTP;
      else if ( memcmp( scratch, "begin ", 6 ) == 0 )
        read_mode = MODE_UUE;     
    }
  }
  return read_mode;
}

/* --------------------------------------------------------------------- */

static unsigned int netspool_reserve(NETSPOOL *spool, unsigned int len)
{
  if (spool)
  {
    if ((spool->size - spool->used) >= len)
      return len;
    len += len % 1024;
    if (spool->mem)
    {
      unsigned int newsize = spool->size + len;
      if (newsize > spool->size) /* no wrap */
      {
        char *mem = (char *)realloc(spool->mem, newsize);
        if (mem)
        {
          spool->mem = mem;
          spool->size = newsize;
          return spool->size - spool->used;
        }
      }
    }
    else
    {
      spool->mem = (char *)malloc(len);
      spool->size = spool->used = 0;
      if (spool->mem)
      {
        spool->size = len;
        return len;
      }
    }
    LogScreen("Net::read: ENOMEM: out of memory\n");
  }
  return 0;
}

static void netspool_pophead(NETSPOOL *spool, unsigned int len)
{
  if (spool)
  {
    if (len >= spool->used)
      spool->used = 0;
    else
    {
      spool->used -= len;
      memmove( spool->mem, &(spool->mem[len]), spool->used );
    }
  }
}

/* --------------------------------------------------------------------- */

/* netconn_read(): receives data from the connection with any
 * necessary decoding. Returns number of bytes copied to 'data'
*/
int netconn_read( void *cookie, char * data, int numRequested )
{
  NETSTATE *netstate = __cookie2netstate(cookie);
  unsigned int numRead, totalRead = 0;
  /* lcbuf must be big enough to hold a full 'Content-Length'/'X-Keyserver' line */
  char lcbuf[64]; char *bufp; 
  int rc, read_mode = 0;  

  if (!netstate)
    return -1;

  TRACE_OUT((+1,"netconn_read(requested=%d)\n", numRequested ));

  if (numRequested > 0)
  {
    totalRead = netstate->netbuffer.used;
    if (totalRead >= ((unsigned int)numRequested))
    {
      totalRead = numRequested;
      memcpy(data, netstate->netbuffer.mem, totalRead);
      netspool_pophead( &(netstate->netbuffer), totalRead );
      DUMP_PACKET("Get", data, totalRead );    
    }
    else
    {
      bufp = data;
      totalRead = 0;
      while (totalRead < ((unsigned int)numRequested))
      {
        numRead = ((unsigned int)numRequested) - totalRead;
        rc = net_read(netstate->sock, bufp, &numRead, 
                      netstate->conn_hostaddr, 
                      netstate->conn_hostport, 
                      netstate->iotimeout);
        if (rc != 0)
        {
          if (netstate->verbose_level > 0 && !__break_check(netstate))
            LogScreen("Net::read: %s\n", net_strerror(rc, netstate->sock ));
          break;
        }
        if (numRead == 0)
        {
          if (netstate->verbose_level > 0)
            LogScreen("Network read error: ETIMEDOUT: operation timed out\n");
          break;
        }
        DUMP_PACKET("read", bufp, numRead );    
        totalRead += numRead;
        bufp += numRead;
      }  
      if (totalRead == ((unsigned int)numRequested)) /* no errors */
      {
        /* detect either HTTP *or* UUE on an input stream */
        read_mode = auto_sense_http_uue( data, totalRead );
        if (read_mode == -1) /* couldn't be determined */
        {                    /* so, inherit from ::Put() */
          read_mode = 0;
           /* must be either/or, not both, and HTTP first */
          if ((netstate->mode & MODE_HTTP) != 0)
            read_mode = MODE_HTTP;        
          else if ((netstate->mode & MODE_UUE) != 0)
            read_mode = MODE_UUE;
        }
      }
    }
  }
  TRACE_OUT((0,"read_mode = %s\n", ((read_mode & MODE_HTTP)?("HTTP"):
                     ((read_mode & MODE_UUE)?("UUE"):("0"))) ));
  if ((read_mode & (MODE_UUE|MODE_HTTP)) != 0)
  {
    NETSPOOL decodebuffer;
    unsigned int content_length = 0;
    int modedet_pending = 0, http_close = 0;
    int need_close = 0, gotfullpacket = 0;
    memset( &decodebuffer, 0, sizeof(decodebuffer));

    need_close = 1; /* assume failed */
    if (netspool_reserve( &decodebuffer, totalRead ) >= totalRead)
    {
      need_close = 0; /* didn't fail */
      memcpy(decodebuffer.mem, data, totalRead);
      decodebuffer.used += totalRead;
    }

    while (!need_close && !gotfullpacket)
    {
      /* ++++++++++++++++ */

      /* modedet_pending == we _were_ in http but don't know what format */
      /* the rest is in (we are waiting for the first data after the header)*/
      if (modedet_pending)
      {
        if (content_length < 6) /* no way this can be UUE */
        {
          read_mode = 0; /* must be binary data */
          modedet_pending = 0; /* mode detection no longer pending */
        }
        else
        {
          /* detect either HTTP *or* UUE on the input stream head */
          /* (can't be HTTP again though, since we've already done it) */
          rc = auto_sense_http_uue( decodebuffer.mem, decodebuffer.used);
          if (rc != -1) /* mode was determined ok */
          {
            read_mode = rc; /* will be either UUE or nothing */
            modedet_pending = 0; /* mode detection no longer pending */
          }
        }
        TRACE_OUT((0,"modedet_pending read_mode = %s\n", 
                     ((read_mode & MODE_HTTP)?("HTTP"):
                     ((read_mode & MODE_UUE)?("UUE"):("0"))) ));
      } /* modedet_pending */

      /* ++++++++++++++++ */

      if ((read_mode & (MODE_HTTP))!=0) /* in the HTTP header */
      {
        netstate->mode |= MODE_HTTP; /* add this on for Put() */
        for (;;)
        {
          unsigned int linelen;

          int skipcount = __peek_netline( decodebuffer.mem, decodebuffer.used,
                                          &linelen, lcbuf, sizeof(lcbuf) );
          if (skipcount == 0) /* no end-of-line yet */
            break;            /* go get more data */

          if (skipcount < 0)
          {
            if (netstate->verbose_level > 0)
              LogScreen("Net::read: unexpected binary data in HTTP header\n" );
            need_close = 1;
            break;
          }
          TRACE_OUT((0,"line: '%s'\n", lcbuf ));
 
          if (linelen < 1) /* blank line separating header from body */
          {
            netspool_pophead( &decodebuffer, skipcount );
            read_mode &= ~MODE_HTTP; /* no longer do HTTP */
            modedet_pending = 1; /* we don't know what follows yet */
            if (content_length == 0) /* we didn't get a content-length pragma */
            {            
              if (netstate->verbose_level > 0)
                LogScreen("HTTP error: 500 missing 'Content-Length' pragma\n");
              need_close = 1;
            }
            break;
          }

          /* decodebuffer.RemoveHead(skipcount) is at end of this 'if/else' */
          if (memcmp(lcbuf, "http/1.", 7 ) == 0) 
          {                     /* 'HTTP/1.0 500 Cache Detected Error' */
            bufp = decodebuffer.mem;
            bufp[linelen] = 0;
            bufp += 7;
            __ntoh_str(bufp);
            while (*bufp && *bufp != ' ')
              bufp++;
            while (*bufp == ' ')
              bufp++;
            rc = atoi(bufp);
            if (rc < 200 || rc >= 300) /* not "200 ok" */
            {
              LogScreen("HTTP error: '%s'\n", bufp );
              need_close = 1;
              netstate->svc_hostaddr = 0;
              break; /* while remove line */
            }
          }
          else if (memcmp(lcbuf, "content-length: ", 16) == 0) //"Content-Length: "
          {
            bufp = &lcbuf[16];
            content_length = atoi( bufp );
          }
          else if (memcmp(lcbuf, "x-keyserver: ", 13) == 0) //"X-KeyServer: "
          {
            if (netstate->svc_hostaddr == 0)
            {
              u32 newaddr = 0; unsigned int count = 1;
              bufp = &lcbuf[13];
              TRACE_OUT((+1,"X-Keyserver: '%s'\n", bufp ));
              if (net_resolve( bufp, &newaddr, &count )==0)
                netstate->svc_hostaddr = newaddr;
              TRACE_OUT((-1,"X-Keyserver: => %s\n", net_ntoa(newaddr)));
            }
          }
          else if (memcmp(lcbuf, "connection: close", 17 ) == 0)
          { 
            http_close = 1;
          }
          netspool_pophead( &decodebuffer, skipcount );

        } /* for (;;) */
      } /* while (!need_close && (read_mode & (MODE_HTTP))!=0) */
      else if ((read_mode & (MODE_UUE))!=0)
      {
        netstate->mode |= MODE_UUE; /* add this on for Put() */
        for (;;)
        {
          unsigned int linelen;
          int skipcount = __peek_netline( decodebuffer.mem, decodebuffer.used,
                                          &linelen, lcbuf, 10 /*sizeof(lcbuf)*/ );
          if (skipcount == 0) /* no end-of-line yet */
            break;            /* go get more data */

          if (skipcount < 0)
          {
            if (netstate->verbose_level > 0)
              LogScreen("Net::read: unexpected binary data in UUE stream.\n");
            need_close = 1;
            break;
          }
          TRACE_OUT((0,"line: '%s'\n", lcbuf ));

          /* decodebuffer.RemoveHead(skipcount) is at end of this 'if/else' */
          if (memcmp( lcbuf, "end", 3 ) == 0)
          {
            gotfullpacket = 1;
            break; /* finis */
          }
          else if (linelen > 1 && memcmp( lcbuf, "begin ", 6 ) != 0) /* data */
          {
            /* decode each line in place */
            const char *p = decodebuffer.mem; /* source (const char *) */
            int n = UU_DEC(*p);               /* decoded length */
            unsigned int uulen = (((n/3)*4)+((n%3)+1)); /* source len */
            bufp = decodebuffer.mem;          /* destination (char *) */

            TRACE_OUT((0,"uue: c='%c', linelen=%u, n=%d, uulen=%d\n",
                       *p, linelen, n, uulen ));

            if (linelen < uulen ) /* physical line is shorter than the UUE */
            {                     /* length byte says it should be */
              if (netstate->verbose_level > 1)
                LogScreen("Net::read UUE decode error (%d.%d:%d.%d)\n",
                           *p, n, linelen, uulen );
              need_close = 1;
              break;
            }

            p++; /* skip the length marker */
            while (n > 0)
            {
              if (n >= 1)
                *bufp++=(char)(((UU_DEC(p[0])<<2)| (UU_DEC(p[1])>>4)));
              if (n >= 2)
                *bufp++=(char)(((UU_DEC(p[1])<<4)| (UU_DEC(p[2])>>2)));
              if (n >= 3)
                *bufp++=(char)(((UU_DEC(p[2])<<6)| (UU_DEC(p[3])) ));
              n -= 3;
              p += 4;
            }
            linelen = bufp - decodebuffer.mem;
            if (netspool_reserve( &(netstate->netbuffer), linelen) < linelen)
            {
              need_close = 1;
              break;
            }
            memcpy( &(netstate->netbuffer.mem[netstate->netbuffer.used]),
                    decodebuffer.mem, linelen);
            netstate->netbuffer.used += linelen;
          } /* its data */
          netspool_pophead( &decodebuffer, skipcount );

        } /* for (;;) */
      } /* UUE */
      else /* binary data, or haven't determined binary/UUE mode yet */
      {
        /* don't touch the data if mode determination is still pending */
        /* if it isn't pending when we get here, its guaranteed binary data */
        if (!modedet_pending)
        {
          unsigned int buflen = decodebuffer.used;
          TRACE_OUT((0,"binary data. read length = %u\n", buflen ));
          if (netspool_reserve( &(netstate->netbuffer), buflen) < buflen)
          {
            need_close = 1;
            break;
          }
          memcpy( &(netstate->netbuffer.mem[netstate->netbuffer.used]),
                    decodebuffer.mem, buflen );
          netstate->netbuffer.used += buflen;
          decodebuffer.used = 0;

          buflen = netstate->netbuffer.used; /* new size */
          TRACE_OUT((0,"binary data. total length = %u\n", buflen ));
          if (buflen >= content_length)
          {
            gotfullpacket = 1;
          }
        }
      } /* binary_data */

      /* ++++++++++++++++++++ */

      if (!gotfullpacket && !need_close)
      {
        numRead = 500;
        if (content_length)
        {
          numRead = content_length - decodebuffer.used;
          if (content_length < decodebuffer.used)
            numRead = 0;
        }  
        if (numRead < 1)
          ; /* nothing */
        else if (netspool_reserve( &decodebuffer, numRead ) < numRead)
          need_close = 1;
        else
        {  
          bufp = &decodebuffer.mem[decodebuffer.used];
          rc = net_read(netstate->sock, bufp, &numRead, 
                        netstate->conn_hostaddr, netstate->conn_hostport, 
                        netstate->iotimeout);
          if (rc != 0)
          {
            if (netstate->verbose_level > 0 && !__break_check(netstate))
              LogScreen("Net::read: %s\n", net_strerror(rc, netstate->sock ));
            need_close = 1;
          }
          else if (numRead == 0)
          {
            if (netstate->verbose_level > 0)
              LogScreen("Network read error: ETIMEDOUT: operation timed out\n");
            need_close = 1;
          }
          else /* no error, not timeout */
          {
            DUMP_PACKET("read", bufp, numRead );    
            decodebuffer.used += numRead;
          } 
        }
      } /* if (!first_time) */

    } /* while (!need_close && !gotfullpacket) */

    if (decodebuffer.mem)
    {
      decodebuffer.used = decodebuffer.size = 0;
      free((void *)decodebuffer.mem);
      decodebuffer.mem = 0;      
    }

    totalRead = 0;
    if (netstate->netbuffer.mem)
    {
      totalRead = netstate->netbuffer.used;
      TRACE_OUT((0,"netbuffer length = %u\n", totalRead ));
      if (totalRead > ((unsigned int)numRequested))
        totalRead = numRequested;
      TRACE_OUT((0,"totalread = %u\n", totalRead ));
      memcpy(data, netstate->netbuffer.mem, totalRead);
      netstate->netbuffer.used -= totalRead;
      DUMP_PACKET("Get", data, totalRead );    
    }

    if (need_close || http_close)
    { 
      /* CloseConnection() must come after the copy */
      __close_connection(netstate);
    }
  }

  TRACE_OUT((-1,"netconn_read() => %u\n", totalRead ));
  return totalRead;
}

/* =================================================================== */

/* netconn_reset(): reset the connection. Fails (by design) if
 * thataddress is zero.
*/
int netconn_reset(void *cookie, u32 thataddress)
{
  NETSTATE *netstate = __cookie2netstate(cookie);
  int rc = -1;
  TRACE_OUT((+1,"netconn_reset(%s)\n", net_ntoa(thataddress) ));
  if (netstate)
  {
    if (thataddress == 0)
    {
      TRACE_OUT((0,"cannot reset to a zero address\n"));
    }
    else
    {
      netstate->reconnected = 1;
      netstate->svc_hostaddr = thataddress;
      rc = __open_connection(netstate);
    }
  }
  TRACE_OUT((-1,"netconn_reset(...) => %d\n", rc ));
  return rc;  
}

/* ====================================================================== */

/* netconn_write(): sends data over the connection with any
 * necessary encoding. Returns 'length' on success, or -1 on error.
*/
int netconn_write( void *cookie, const char * data, int length )
{
  NETSTATE *netstate = __cookie2netstate(cookie);
  unsigned int towrite; char *allocbuf;
  int rc = 0;

  if (!netstate)
    return -1;

  TRACE_OUT((+1,"netconn_write(%p, %d)\n", data, length));

  if (netstate->connection_closed || netstate->puthttpdone)
  {
    TRACE_OUT((0, "netstate->connection_closed\n" ));
    if (netconn_reset(netstate, netstate->svc_hostaddr) != 0)
    {
      if (netstate->verbose_level > 0)
        LogScreen("Net::write error: could not reestablish connection.\n");
      rc = -1;
    }
  }

  /* ++++++++++ */

  allocbuf = (char *)0;
  if (rc == 0 /* reset not needed, or reset succeeded */
     &&  (netstate->mode & (MODE_UUE|MODE_HTTP))!=0)
  {
    unsigned int alloclen = 0;
    if ((netstate->mode & MODE_UUE)!=0)
      alloclen += (length * 2) + 200;
    if ((netstate->mode & MODE_HTTP)!=0)
      alloclen += 1024;
    if (alloclen > ((unsigned int)length)) /* didn't wrap */
      allocbuf = (char *)malloc(alloclen);

    if (!allocbuf)
    {
      rc = -1;
      if (netstate->verbose_level > 0)
        LogScreen("Net::send error. Out of memory\n");
    }
  }

  /* ++++++++++ */

  towrite = length;
  if (rc == 0 && (netstate->mode & MODE_UUE)!=0)
  {
    int copylen;
    char *bufp, *bufpstart;

    bufpstart = allocbuf;
    if ((netstate->mode & MODE_HTTP)!=0)
      bufpstart += 1024;        
    bufp = bufpstart;

    copylen = strlen(strcpy( bufp, "begin 644 query.txt\r\n"));
    __hton_str( bufp );
    bufp += copylen;

    copylen = length;
    while (copylen > 0)      
    {
      int linelen = copylen;
      if (linelen > 45)
        linelen = 45;
      copylen -= linelen;
      *bufp++ = UU_ENC(linelen);
  
      while (linelen > 2)
      {
        *bufp++ = UU_ENC((char)(data[0] >> 2));
        *bufp++ = UU_ENC((char)(((data[0] << 4) & 060) | ((data[1] >> 4) & 017)));
        *bufp++ = UU_ENC((char)(((data[1] << 2) & 074) | ((data[2] >> 6) & 03)));
        *bufp++ = UU_ENC((char)(data[2] & 077));
        data += 3;
        linelen -= 3;
      }
      if (linelen != 0)
      {
        char c = (char)(linelen == 1 ? 0 : data[1]);
        *bufp++ = UU_ENC((char)(data[0] >> 2));
        *bufp++ = UU_ENC((char)(((data[0] << 4) & 060) | ((c >> 4) & 017)));
        *bufp++ = UU_ENC((char)((c << 2) & 074));
        *bufp++ = UU_ENC(0);
        data += linelen;
      }

      *bufp++ = 0x0d; /* binary '\r' */
      *bufp++ = 0x0a; /* binary '\n' */
    }
    copylen = strlen(strcpy( bufp, "end\r\n" ));
    __hton_str( bufp );
    bufp += copylen;

    towrite = bufp - bufpstart;
    data = bufpstart;
  } /* rc == 0 && UUE */

  /* ++++++++++ */

  if (rc == 0 && (netstate->mode & MODE_HTTP)!=0)
  {
    unsigned long hdrlen;
    char userpass[(((sizeof(netstate->fwall_userpass)+1)*4)/3)];

    userpass[0] = '\0';
    if (netstate->fwall_userpass[0])
    {
      if ( base64_encode( userpass, netstate->fwall_userpass,
           sizeof(userpass), strlen(netstate->fwall_userpass)) < 0 )
      {
        userpass[0] = '\0';
      }
      userpass[sizeof(userpass)-1]='\0';
    }
    hdrlen = sprintf( allocbuf,
                      "POST http://%s:%u/cgi-bin/rc5.cgi HTTP/1.0\r\n"
                      //"Connection: Keep-Alive\r\n" /* HTTP/1.1 */
                      "Proxy-Connection: Keep-Alive\r\n" /* HTTP/1.0 */
                      "%s%s%s"
                      "Content-Type: application/octet-stream\r\n"
                      "Content-Length: %lu\r\n\r\n",
                      ((netstate->svc_hostaddr)?(net_ntoa(netstate->svc_hostaddr)):(netstate->svc_hostname)),
                      ((unsigned int)(netstate->svc_hostport)),
                      ((userpass[0])?("Proxy-authorization: Basic "):("")),
                      ((userpass[0])?(userpass):("")),
                      ((userpass[0])?("\r\n"):("")),
                      (unsigned long) towrite );
    __hton_str( allocbuf );
    memcpy( &allocbuf[hdrlen], data, towrite );
    data = allocbuf;
    towrite += hdrlen;
  } /* rc == 0 && HTTP */

  /* ++++++++++ */

  if (rc == 0)
  {  
    unsigned int written = towrite;

    DUMP_PACKET("Put", data, towrite );
    rc = net_write(netstate->sock, data, &written, 
                   netstate->conn_hostaddr, netstate->conn_hostport, 
                   netstate->iotimeout);
    if (rc == 0) /* success! sent all. */
    {
      rc = length; /* we return the requested length */
      if ((netstate->mode & MODE_HTTP)!=0)
        netstate->puthttpdone = 1;
    }
    else
    {
      if (netstate->verbose_level > 0 && !__break_check(netstate))
        LogScreen("Net::write: %s\n", net_strerror(rc,netstate->sock) );
      rc = -1;
    }
  }

  if (allocbuf)
  {
    free((void *)allocbuf);
  }

  TRACE_OUT((-1,"Put() => %d\n", rc ));
  return rc;
}

/* ====================================================================== */

/* netconn_getname(): name of host as determined at open time.
 * Returns zero on success or -1 on error.
*/
int netconn_getname(void *cookie, char *buffer, unsigned int len )
{
  NETSTATE *netstate = __cookie2netstate(cookie);
  if (!netstate)
    return -1;
  if (net_gethostname(buffer, len) != 0)
    return -1;
  return 0;
}

/* ====================================================================== */

/* netconn_getpeer(): get address of host connected to, or zero
 * on error. Probably only useful for debugging.
*/
u32 netconn_getpeer(void *cookie)
{
  NETSTATE *netstate = __cookie2netstate(cookie);
  if (!netstate)
    return 0;
  return netstate->svc_hostaddr;
}

/* ====================================================================== */

/* netconn_setpeer(): set address of host to connect to in the event
 * of a disconnect. (in the event of an HTTP/1.0 close)
*/
int netconn_setpeer(void *cookie, u32 address)
{
  NETSTATE *netstate = __cookie2netstate(cookie);
  if (!netstate)
    return 0;
  if (!netstate->svc_hostaddr)
    netstate->svc_hostaddr = address;
  return netstate->svc_hostaddr;
}

/* ====================================================================== */

/* netconn_getaddr(): get address connected from, or zero
 * on error (or not connected).
*/
u32 netconn_getaddr(void *cookie)
{
  NETSTATE *netstate = __cookie2netstate(cookie);
  if (!netstate)
    return 0;
  return netstate->local_hostaddr;
}

/* ====================================================================== */

/* netconn_close(): close the connection. Cookie is then no longer
 * usable.
*/
int netconn_close(void *cookie)
{
  NETSTATE *netstate = __cookie2netstate(cookie);
  TRACE_OUT((+1,"netconn_close(%p)\n",cookie));
  if (netstate)
  {
    __close_connection(netstate);
    TRACE_OUT((0,"netconn_close() 1\n"));
    if (netstate->verbose_level > 0 && netstate->shown_connection)
      LogScreen("Connection closed.\n");
    TRACE_OUT((0,"netconn_close() 2 [netstate->netbuffer=%p]\n",netstate->netbuffer.mem));
    if (netstate->netbuffer.mem)
    {
      TRACE_OUT((+1,"delete netstate->netbuffer\n"));
      free((void *)netstate->netbuffer.mem);
      memset(&(netstate->netbuffer),0,sizeof(netstate->netbuffer));
      TRACE_OUT((-1,"delete netstate->netbuffer\n"));
    }
    TRACE_OUT((0,"netconn_close() 3\n"));
    if (netstate->sock != INVALID_SOCKET)
      net_close(netstate->sock);
    netstate->sock = INVALID_SOCKET;
    TRACE_OUT((0,"netconn_close() 4\n"));
    netstate->magic = 0;
    TRACE_OUT((+1,"free(netstate)\n"));
    free(netstate);
    TRACE_OUT((-1,"free(netstate)\n"));
    netstate = (NETSTATE *)0;
  }
  TRACE_OUT((-1,"netconn_close()=>0\n"));
  return 0;
}

/* ====================================================================== */

/* netconn_open(): create a new connection. Returns a 'handle' for
 * subsequent netconn_xxx() operations or NULL on error.
*/
void *netconn_open( const char * _servname, int _servport, 
                    int _nofallback, int _iotimeout, int _enctype, 
                    const char *_fwallhost, int _fwallport, 
                    const char *_fwalluid )
{ 
  NETSTATE *netstate = ((NETSTATE *)0);
  int rc = 0;

  TRACE_OUT((+1,"netconn_open()\n"));

  if (rc == 0)
  {
    size_t dummy; /* shaddup compiler about 'if' always being false */
    if (((dummy = offsetof(SOCKS4, USERID[0])) != 8) ||
        ((dummy = offsetof(SOCKS5METHODREQ, Methods[0])) != 2) ||
        ((dummy = offsetof(SOCKS5METHODREPLY, end)) != 2) ||
        ((dummy = offsetof(SOCKS5USERPWREPLY, end)) != 2) ||
        ((dummy = offsetof(SOCKS5, end)) != 10))
    {
      LogScreen("Net::Socks Incorrectly packed structures.\n");
      rc = -1;
    }
  }

  /* -------------------------------------------------- */

  if (rc == 0)
  {
    netstate = (NETSTATE *)malloc(sizeof(NETSTATE));
    if (!netstate)
    {
      LogScreen("Net::open error: insufficient memory\n");
      rc = -1;
    }
    else
    {
      memset((void *)netstate, 0, sizeof(NETSTATE)); 
      rc = 0;
    }
  }

  /* -------------------------------------------------- */

  if (rc == 0)
  {
    netstate->magic = NETSTATE_MAGIC;
    netstate->reconnected = 0;
    netstate->shown_connection = 0;
    netstate->sock = INVALID_SOCKET;
    netstate->connection_closed = 1;
    netstate->mode = 0;
    netstate->startmode = 0;
    netstate->puthttpdone = 0;
    netstate->break_pending = CheckExitRequestTrigger();

    netstate->svc_hostport = netstate->conn_hostport = 0;
    netstate->svc_hostaddr = netstate->conn_hostaddr = 0;
    netstate->svc_hostname = netstate->conn_hostname = (const char *)0;
    netstate->resolve_addrcount = -1; /* uninitialized */

    netstate->verbose_level = 1; /* currently one one verbose level */
    /* later use 'reconnected' and 'shown_connection' for less verbosity */
  }

  /* -------------------------------------------------- */

  if (rc == 0)
  {
    /* Set and validate the connection timeout value. */
    /* argument is in seconds, netstate->iotimeout is in millisecs */
    netstate->iotimeout = -1; /* assume blocking mode */
    if (_iotimeout >= 0)      /* not blocking mode */
    {
      if (_iotimeout < 5)
        _iotimeout = 5;       /* 5 seconds minimum. */
      else if (_iotimeout > 300)
        _iotimeout = 300;     /* 5 minutes maximum. */
      netstate->iotimeout = _iotimeout * 1000; /* secs->millisecs */
    }
  }

  /* -------------------------------------------------- */

  /* take care of netstate->startmode (encoding) and firewall host:port settings */
  /* this must be done before the service host:port is handled (below) */
  if (rc == 0)
  {
    int need_fwallname = 0; /* zap the name later if need_fwallname is zero */
    int have_fwallname = 0; /* do we have one? */

    netstate->fwall_hostport = 0;
    netstate->fwall_hostaddr = 0;
    netstate->fwall_hostname[0] = '\0';
    netstate->fwall_userpass[0] = '\0';

    if (_fwallhost)
    {
      unsigned int pos = 0;
      while (*_fwallhost == ';' || *_fwallhost == ',' || isspace(*_fwallhost))
        _fwallhost++;
      while (*_fwallhost && pos < (sizeof(netstate->fwall_hostname)-1))
      {
        if (*_fwallhost == ':')
          break;
        if (*_fwallhost == ';' || *_fwallhost == ',' || isspace(*_fwallhost))
          break;
        netstate->fwall_hostname[pos++] = (char)(*_fwallhost++);
      }
      netstate->fwall_hostname[pos] = '\0';
      if (*_fwallhost == ':') /* embedded port number */
      {
        int foundport = 0;
        _fwallhost++;
        while (isdigit(*_fwallhost))
        {
          foundport = (foundport * 10)+(*_fwallhost - '0');
          if (foundport > 0xffff)
            break;
          _fwallhost++;
        }
        if (!*_fwallhost || *_fwallhost == ';' || *_fwallhost == ',' ||
           isspace(*_fwallhost))
        {
          _fwallport = foundport;
        }
      }
      have_fwallname = (netstate->fwall_hostname[0] != '\0');
    }
    if (_enctype == 1 /*uue*/ || _enctype == 3 /*http+uue*/)
    {
      netstate->startmode |= MODE_UUE;
      netstate->mode = netstate->startmode;
    }
    if (_enctype == 2 /*http*/ || _enctype == 3 /*http+uue*/)
    {
      netstate->startmode |= MODE_HTTP;
      netstate->mode = netstate->startmode;
      if (have_fwallname)
      {
        netstate->fwall_hostport = _fwallport;
        if (netstate->fwall_hostport == 0)
          netstate->fwall_hostport = 8080;
        need_fwallname = 1;
      }
    }
    else if (_enctype == 4 /*socks4*/ || _enctype == 5 /*socks5*/)
    {
      if (have_fwallname)
      {
        netstate->startmode |= ((_enctype == 4)?(MODE_SOCKS4):(MODE_SOCKS5));
        netstate->mode = netstate->startmode;
        netstate->fwall_hostport = _fwallport;
        if (netstate->fwall_hostport == 0)
          netstate->fwall_hostport = 1080;
        need_fwallname = 1;
      }
      else
      {
        LogScreen("Net::error: proxy hostname required for SOCKS%d support.\n"
                  "Connect cancelled.\n", _enctype );
        rc = -1;
      }
    }
    if (!need_fwallname) /* don't need proxification for this mode */
    {
      netstate->fwall_hostname[0] = '\0'; /*so clear it to not confuse open()*/
    }
    else if (_fwalluid) /* yes we need the name, and have a user:pass too */
    {
      strncpy( netstate->fwall_userpass, _fwalluid, 
               sizeof(netstate->fwall_userpass));
      netstate->fwall_userpass[sizeof(netstate->fwall_userpass)-1] = '\0';
    }
  } /* rc == 0 */

  /* -------------------------------------------------- */

  /* deal with _servname, _servport and _fallback hostname and port */
  /* this must be the last thing done, since port numbers depends on netstate->startmode */
  if (rc == 0)
  {
    static const struct  // this structure defines which proxies are
    {                    // 'responsible' for which time zone. The timezones
      const char *name;  // overlap, and users in an overlapped area will
      int minzone;       // have multiple proxies at their disposal.
      int maxzone;       // 
    } proxyzoi[] = {
               { "euro",   -2, +4  }, //euro crosses 0 degrees longitude
               { "asia",   +3, +10 },
               { "aussie", +9, -9  }, //jp and aussie cross the dateline
               { "jp",    +10, -10 },
               { "us",    +12, -12 }  }; //default (must be last)
    unsigned int buf_used = 0;
    int have_a_dnet_proxy = 0;
    int fallback_port;
    netstate->servername_buffer[0] = '\0';
    netstate->servername_count = 0;
    netstate->servername_selector = 0;
    memset(&netstate->servername_ptrs[0], 0, sizeof(netstate->servername_ptrs));
    memset(&netstate->servername_ports[0], 0, sizeof(netstate->servername_ports));

    #define IS_DNET_PROXY_VALID_PORT(__pp) \
           (__pp == 2064 || __pp == 80 || __pp == 23 || __pp == 3064)

    /* determine port # used for invalid _servport and fallback host */
    fallback_port = _servport;
    if (!IS_DNET_PROXY_VALID_PORT(fallback_port))
    {
      fallback_port = 2064;
      if ((netstate->startmode & MODE_HTTP) != 0)
        fallback_port = 80;
      else if ((netstate->startmode & MODE_UUE) != 0)
        fallback_port = 23;
    }
    if (_servport <= 0 || _servport >= 0xffff) /* _servport is invalid */
      _servport = fallback_port;               /* so use the default port */

    TRACE_OUT((0,"server:%s:%d\n", (_servname)?(_servname):(""), _servport));
    if (_servname)
    {
      const char *p = _servname;
      while (*p)
      {
        int selport = -1, badname = 0; 
        unsigned int namelen = 0;

        while (*p && (*p == ',' || *p == ';' || isspace(*p)))
          p++;
        while (*p && !(*p == ',' || *p == ';' || isspace(*p)))
        {
          if (!badname) /* otherwise just keep skipping to next name */
          {
            if (*p == ':') /* embedded port # */
            {
              if (selport >= 0) /* already have a port number */
                badname = 1;
              else if (namelen == 0) /* port without name */
                badname = 1;
              else if (!isdigit(*(p+1))) /* is not a number */
                badname = 1;    
              else          /* skip to the end of the number */
              {             /* try to construct a port number while skipping */
                selport = 0;
                while (isdigit(*++p))
                {
                  if (selport >= 0)
                  {
                    selport = ((selport * 10)+ (*p - '0'));
                    if (selport > 0xffff)
                      selport = -1;
                  }
                }
                if (selport <= 0) /* invalid port number follows ':' */
                {
                  selport = -1;
                  badname = 1;
                }
                continue; /* don't do another p++ */
              }
            } /* (*p == ':') */
            else if (selport >= 0) /* ack! part of name follows port # */
            {
              badname = 1;
            }  
            else if ((buf_used+namelen) == (sizeof(netstate->servername_buffer)-1) )
            {
              badname = 1;
            }
            else if (badname == 0)
            {
              netstate->servername_buffer[buf_used+namelen] = (char)tolower(*p);
              namelen++;
            }
            p++;
          } /* if (!badname) */
        } /* while (*p && !(*p == ',' || *p == ';' || isspace(*p))) */
        netstate->servername_buffer[buf_used+namelen] = '\0';
        if (!badname && namelen)
        {
          const char *hostname = &netstate->servername_buffer[buf_used];
          if (strcmp(hostname,"*") == 0 ||
              strcmp(hostname,"auto")==0 || strcmp(hostname,"(auto)") == 0)
          {
            badname = 1;
            if (IS_DNET_PROXY_VALID_PORT(selport))
              fallback_port = selport;
          }            
          else if (namelen < 15)
          {
            ; /* nothing - less than sizeof("distributed.net") */
          }
          else if (strcmp("distributed.net",&hostname[namelen-15]) != 0)
          {
            ; /* nothing - does not end in "distributed.net" */
          }
          else if (namelen == 15)
          {
            badname = 1; /* name is exactly "distributed.net" */
          }
          else if (hostname[namelen-16] != '.')
          {
            ; /* nothing - does not end in ".distributed.net" */
          }
          else if (namelen == 21 && memcmp(hostname, "n0cgi.",6)==0)
          {              
            ; /* nothing - its a special (non-proxy) d.net host */
          }
          else if (namelen < 20) /* all d.net proxy hostnames are >= 20 */
          {
            badname = 1; /* its a d.net hostname but not a proxy and not special */
          }
          else if (namelen > 22 && memcmp(&hostname[namelen-22], ".proxy.", 7)==0)
          {              
            have_a_dnet_proxy = 1; /* we have a dnet proxy */
          }
          else if (memcmp(".v27.", &hostname[namelen-20],5) !=0)
          {          
            badname = 1; /* its a d.net hostname, but not a proxy */
          }
          else /* we have a .v27.d.net proxy, validate it */
          { 
            unsigned int pos;
            badname = 1; /* assume this, clear it later */
            for (pos = 0;pos < (sizeof(proxyzoi)/sizeof(proxyzoi[0])); pos++ )
            {
              unsigned int len = strlen(proxyzoi[pos].name);
              if ( (len+20) == namelen &&
                memcmp( proxyzoi[pos].name, hostname, len )==0)
              {
                int namedport = 2064;
                if (hostname[len] != '.')
                  namedport = atoi( &hostname[len] );
                if (!IS_DNET_PROXY_VALID_PORT(namedport))
                  break; /* invalid port number in name */
                badname = 0; /* ok, hostname is valid */
                have_a_dnet_proxy = 1; /* yes, we have a dnet proxy */
                if (selport != 3064)
                  selport = namedport; /* the hostname has determined port */
                break;
              }
            }
          }
        }
        if (!badname && namelen)
        {
          if (selport < 0)       /* no port number specified */
            selport = _servport; /* inherit port number */
          netstate->servername_ports[netstate->servername_count] = selport;
          netstate->servername_ptrs[netstate->servername_count] = 
                                     &netstate->servername_buffer[buf_used];
          buf_used += namelen+1; 
          netstate->servername_count++;
          if (netstate->servername_count == 
                  (sizeof(netstate->servername_ptrs)/
                  sizeof(netstate->servername_ptrs[0])))
          {
            break;
          }
        }
      } /* while (*p) */
    } /* if (_servname) */

    /* +++++++++++++ */
    /*
     * if zero servernames, then we will use auto-selection
     * if > 1  servernames, then nofallback is implicitely off
     * if exactly one servername, 
     *    and nofallback is not explicitely on, 
     *    and the servername is _not_ already a distributed.net proxy
     *    then add an auto-selection name for fallback
    */
    if ((netstate->servername_count == 0) ||
      (netstate->servername_count == 1 && !_nofallback && !have_a_dnet_proxy))
    {
      int tzmineast = -CliTimeGetMinutesWest();  /* clitime.cpp */
      unsigned int pos;

      for (pos = 0;pos < (sizeof(proxyzoi)/sizeof(proxyzoi[0])); pos++ )
      {
        int inrange = 1; /* default is always in range */
        if (pos < ((sizeof(proxyzoi)/sizeof(proxyzoi[0]))-1)) /* not default */
        {
          int tz_min = proxyzoi[pos].minzone * 60;
          int tz_max = proxyzoi[pos].maxzone * 60;
          if (tz_min > 0 && tz_max < 0) /* straddles the date line */
            inrange = (tzmineast >= tz_min && tzmineast <= +(12*60)) ||
                      (tzmineast <= tz_max && tzmineast >= -(12*60));
          else
            inrange = ( tzmineast >= tz_min && tzmineast <= tz_max );
        }
        if ( inrange ) 
        {
          char proxyname[15+sizeof(".v27.distributed.net")];
          strcpy( proxyname, proxyzoi[pos].name );
          if (fallback_port != 2064 && fallback_port != 3064)
            sprintf( &proxyname[strlen(proxyname)], "%d", fallback_port );
          strcat( proxyname, ".v27.distributed.net" );
          if ((strlen(proxyname)+buf_used+2) >= sizeof(netstate->servername_buffer))
            break;
          strcpy( &netstate->servername_buffer[buf_used], proxyname );
       
          netstate->servername_ports[netstate->servername_count] = fallback_port;
          netstate->servername_ptrs[netstate->servername_count] = 
                                   &netstate->servername_buffer[buf_used];
          buf_used += strlen( proxyname )+1;
          netstate->servername_count++;
          if (netstate->servername_count == 
                (sizeof(netstate->servername_ptrs)/
                sizeof(netstate->servername_ptrs[0])))
          {
            break;
          }
        } /* if inrange */
      } /* for (pos = 0; ... */
    } /* if ((servername_count == ... */
    netstate->servername_selector = 0;
    netstate->svc_hostname = netstate->servername_ptrs[0];
    netstate->svc_hostport = netstate->servername_ports[0];
  } /* rc == 0 */

  /* -------------------------------------------------- */

  if (rc == 0)
  {
    rc = __open_connection(netstate);
  }

  /* -------------------------------------------------- */

  if (rc != 0 && netstate)
  {
    netconn_close(netstate); /* frees (netstate) too */
    netstate = (NETSTATE *)0;
  }

  TRACE_OUT((-1,"netconn_open() =>%s\n",((netstate)?("ok"):("failed")) ));
  return (void *)netstate;
}


