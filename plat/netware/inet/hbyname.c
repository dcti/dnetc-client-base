/*
 * Classification: network address resolution
 * Service: Internet Network Library
 * Author: Cyrus Patel <cyp@fb14.uni-mainz.de>
 * Copyright: none (nameser defines from @(#)nameser.h 8.1 (Berkeley) 6/2/93)
 *
 * contains stubs/replacements for...
 * NETDB.NLM's gethostbyname() and gethostbyaddr() 
 * They're quick and dirty, but better than nothing (NetWare 3.x)
 * They do not support recursion or TCP virtual loops.
 *
 * $Log: hbyname.c,v $
 * Revision 1.1.2.1  2001/01/21 15:10:30  cyp
 * restructure and discard of obsolete elements
 *
 * Revision 1.1.2.4  2000/06/03 12:19:43  cyp
 * fixed newlines
 *
 * Revision 1.1.2.3  2000/06/02 17:55:18  cyp
 * re-sync
 *
 * Revision 1.1.2.2  2000/06/02 17:51:44  cyp
 * netware changes
 *
 *
*/
#if defined(__showids__)
const char *hbyname_c(void) {
return "@(#)$Id: hbyname.c,v 1.1.2.1 2001/01/21 15:10:30 cyp Exp $"; }
#endif

//#define DEBUG   /* print raw dns I/O */
//#define STANDALONE /* include main() */

/* -------------------------------------------------------------------- */

#define BYTE_ORDER 1234  /* x86!!! least sig byte first */
typedef unsigned short u_int16_t;   /* define these!! */
typedef unsigned long  u_int32_t;
typedef unsigned char  u_char;


#define LITTLE_ENDIAN 1234  /* least-significant byte first (vax, pc) */
#define BIG_ENDIAN    4321  /* most-significant byte first (IBM, net) */
#define PDP_ENDIAN    3412  /* LSB first in word, MSW first in long (pdp)*/

/*
 * Structure for query header.  The order of the fields is machine- and
 * compiler-dependent, depending on the byte/bit order and the layout
 * of bit fields.  We use bit fields only in int variables, as this
 * is all ANSI requires.  This requires a somewhat confusing rearrangement.
 */

typedef struct {
  unsigned  id :16;         /* query identification number */
#if BYTE_ORDER == BIG_ENDIAN
  /* fields in third byte */
  unsigned  qr: 1;         /* response flag */
  unsigned  opcode: 4;     /* purpose of message */
  unsigned  aa: 1;         /* authoritive answer */
  unsigned  tc: 1;         /* truncated message */
  unsigned  rd: 1;         /* recursion desired */
  /* fields in fourth byte */
  unsigned  ra: 1;         /* recursion available */
  unsigned  unused :1;     /* unused bits (MBZ as of 4.9.3a3) */
  unsigned  ad: 1;         /* authentic data from named */
  unsigned  cd: 1;         /* checking disabled by resolver */
  unsigned  rcode :4;      /* response code */
#endif
#if BYTE_ORDER == LITTLE_ENDIAN || BYTE_ORDER == PDP_ENDIAN
  /* fields in third byte */
  unsigned  rd :1;         /* recursion desired */
  unsigned  tc :1;         /* truncated message */
  unsigned  aa :1;         /* authoritive answer */
  unsigned  opcode :4;     /* purpose of message */
  unsigned  qr :1;         /* response flag */
  /* fields in fourth byte */
  unsigned  rcode :4;      /* response code */
  unsigned  cd: 1;         /* checking disabled by resolver */
  unsigned  ad: 1;         /* authentic data from named */
  unsigned  unused :1;     /* unused bits (MBZ as of 4.9.3a3) */
  unsigned  ra :1;         /* recursion available */
#endif
  /* remaining bytes */
  unsigned  qdcount :16;   /* number of question entries */
  unsigned  ancount :16;   /* number of answer entries */
  unsigned  nscount :16;   /* number of authority entries */
  unsigned  arcount :16;   /* number of resource entries */
} HEADER;


/*
 * Define constants based on rfc883
 */
#define PACKETSZ  512      /* maximum packet size */
#define MAXDNAME  1025     /* maximum presentation domain name */
#define MAXCDNAME 255      /* maximum compressed domain name */
#define MAXLABEL  63       /* maximum length of domain label */
#define HFIXEDSZ  12       /* #/bytes of fixed data in header */
#define QFIXEDSZ  4        /* #/bytes of fixed data in query */
#define RRFIXEDSZ 10       /* #/bytes of fixed data in r record */
#define INT32SZ   4        /* for systems without 32-bit ints */
#define INT16SZ   2        /* for systems without 16-bit ints */
#define INADDRSZ  4        /* IPv4 T_A */
#define IN6ADDRSZ 16       /* IPv6 T_AAAA */

/*
 * Internet nameserver port number
 */
#define NAMESERVER_PORT 53

/*
 * Currently defined opcodes
*/ 
#define QUERY     0x0       /* standard query */
#define IQUERY    0x1       /* inverse query */
#define STATUS    0x2       /* nameserver status query */
/*#define xxx   0x3*/       /* 0x3 reserved */
#define NS_NOTIFY_OP  0x4   /* notify secondary of SOA change */

/*
 * Currently defined response codes
 */
#define NOERROR   0   /* no error */
#define FORMERR   1   /* format error */
#define SERVFAIL  2   /* server failure */
#define NXDOMAIN  3   /* non existent domain */
#define NOTIMP    4   /* not implemented */
#define REFUSED   5   /* query refused */

/*
 * Type values for resources and queries
 */
#define T_A         1   /* host address */
#define T_NS        2   /* authoritative server */
#define T_MD        3   /* mail destination */
#define T_MF        4   /* mail forwarder */
#define T_CNAME     5   /* canonical name */
#define T_SOA       6   /* start of authority zone */
#define T_MB        7   /* mailbox domain name */
#define T_MG        8   /* mail group member */
#define T_MR        9   /* mail rename name */
#define T_NULL     10   /* null resource record */
#define T_WKS      11   /* well known service */
#define T_PTR      12   /* domain name pointer */
#define T_HINFO    13   /* host information */
#define T_MINFO    14   /* mailbox information */
#define T_MX       15   /* mail routing information */
#define T_TXT      16   /* text strings */
#define T_RP       17   /* responsible person */
#define T_AFSDB    18   /* AFS cell database */
#define T_X25      19   /* X_25 calling address */
#define T_ISDN     20   /* ISDN calling address */
#define T_RT       21   /* router */
#define T_NSAP     22   /* NSAP address */
#define T_NSAP_PTR 23   /* reverse NSAP lookup (deprecated) */
#define T_SIG      24    /* security signature */
#define T_KEY      25    /* security key */
#define T_PX       26    /* X.400 mail mapping */
#define T_GPOS     27    /* geographical position (withdrawn) */
#define T_AAAA     28    /* IP6 Address */
#define T_LOC      29    /* Location Information */
#define T_NXT      30    /* Next Valid Name in Zone */
#define T_EID      31    /* Endpoint identifier */
#define T_NIMLOC   32    /* Nimrod locator */
#define T_SRV      33    /* Server selection */
#define T_ATMA     34    /* ATM Address */
#define T_NAPTR    35    /* Naming Authority PoinTeR */
  /* non standard */
#define T_UINFO   100    /* user (finger) information */
#define T_UID     101    /* user ID */
#define T_GID     102    /* group ID */
#define T_UNSPEC  103    /* Unspecified format (binary data) */
  /* Query type values which do not appear in resource records */
#define T_IXFR    251    /* incremental zone transfer */
#define T_AXFR    252    /* transfer zone of authority */
#define T_MAILB   253    /* transfer mailbox records */
#define T_MAILA   254    /* transfer mail agent records */
#define T_ANY     255    /* wildcard match */

/*
 * Values for class field
 */

#define C_IN      1   /* the arpa internet */
#define C_CHAOS   3   /* for chaos net (MIT) */
#define C_HS      4   /* for Hesiod name server (MIT) (XXX) */
  /* Query class values which do not appear in resource records */
#define C_ANY   255   /* wildcard match */


/*
 * Inline versions of get/put short/long.  Pointer is advanced.
 *
 * These macros demonstrate the property of C whereby it can be
 * portable or it can be elegant but rarely both.
 */
#define GETSHORT(s, cp) { \
  register u_char *t_cp = (u_char *)(cp); \
  (s) = ((u_int16_t)t_cp[0] << 8) \
      | ((u_int16_t)t_cp[1]) \
      ; \
  (cp) += INT16SZ; \
}

#define GETLONG(l, cp) { \
  register u_char *t_cp = (u_char *)(cp); \
  (l) = ((u_int32_t)t_cp[0] << 24) \
      | ((u_int32_t)t_cp[1] << 16) \
      | ((u_int32_t)t_cp[2] << 8) \
      | ((u_int32_t)t_cp[3]) \
      ; \
  (cp) += INT32SZ; \
}

#define PUTSHORT(s, cp) { \
  register u_int16_t t_s = (u_int16_t)(s); \
  register u_char *t_cp = (u_char *)(cp); \
  *t_cp++ = (u_char)(t_s >> 8); \
  *t_cp   = (u_char)(t_s); \
  (cp) += INT16SZ; \
}

#define PUTLONG(l, cp) { \
  register u_int32_t t_l = (u_int32_t)(l); \
  register u_char *t_cp = (u_char *)(cp); \
  *t_cp++ = (u_char)(t_l >> 24); \
  *t_cp++ = (u_char)(t_l >> 16); \
  *t_cp++ = (u_char)(t_l >> 8); \
  *t_cp   = (u_char)(t_l); \
  (cp) += INT32SZ; \
}


/* ------------------------- end of nameser.h extract ------------------- */

#ifdef __cplusplus
extern "C" 
{  
#endif

#ifdef DEBUG
extern void ConsolePrintf(const char *fmt,...);
#define DEBUGPRINTF(x) ConsolePrintf x
#else
#define DEBUGPRINTF(x)
#endif

/* kernel or A3112.  pasSymName is a pascal format "string" */
extern void *ImportPublicSymbol( int nlmHandle, const char *pasSymName );
extern int UnImportPublicSymbol( int nlmHandle, const char *pasSymName );

#include <stdio.h>         /* fopen etc */
#include <errno.h>         /* errno */
#include <stdlib.h>        /* malloc()/free() */
#include <io.h>            /* close(socket) */
#include <process.h>       /* GetNLMHandle(), spawnlp() */
#include <string.h>

#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>          /* we really only care about hostent */
#include <netinet/in.h>
#include <arpa/inet.h>

#include <sys/time.h>  /* this is for watcom h only (struct timeval) */
#ifdef _SYS_TIMEVAL_H_ //using nwsdk, watcom sdk calls it _SYS_TIME_H_INCLUDED
#include <sys/bsdskt.h>
#endif
#ifndef EINTR             /* and now its even worse */
#define EINTR 63          
#endif

#ifdef __cplusplus
}  
#endif

/* =================================================================== */

#define RES_TIMEOUT  10  /* seconds to timeout for dns requests */
#define MAXNAMESERVERS 5

static const char *resolvconftable[3] = { "sys:/etc/resolv.conf",
                                          "sys:/etc/resolv.cfg", 
                                          "/etc/resolv.conf" };
                                          
static char domainname[63+1]; /* MAXDNAME is defined in nameser.h at 1025 */
static unsigned int nameservercount=0;
static char searchorder[10];
static struct in_addr nameservers[MAXNAMESERVERS+1];

#if 0
struct  hostent {
  char  *h_name;  /* official name of host */
  char  **h_aliases;  /* alias list */
  int h_addrtype; /* host address type */
  int h_length; /* length of address */
  char  **h_addr_list;  /* list of addresses from name server */
  #define h_addr  h_addr_list[0]  /* address, for backward compatiblity */
};
#endif

#if 0
struct in_addr {
  union {
    struct { unsigned char s_b1,s_b2,s_b3,s_b4; } S_un_b;
    struct { unsigned short s_w1,s_w2; } S_un_w;
    unsigned long S_addr;
  } S_un;
};
#endif

/* =================================================================== */

#ifndef DEBUG
#define packetprint(T,P,L) /* NOTHING */
#else
static char *__inet_ntoa( struct in_addr inaddr )
{
  static char addr[ sizeof "255.255.255.255" ];

  sprintf(addr,"%d.%d.%d.%d", 
    inaddr.S_un.S_un_b.s_b1, inaddr.S_un.S_un_b.s_b2,
    inaddr.S_un.S_un_b.s_b3, inaddr.S_un.S_un_b.s_b4 );
  return addr;
}
static void packetprint( const char *label, const void *pkt, unsigned int alen )
{
  const char *apacket = (const char *)pkt;
  unsigned int i;
  for (i = 0; i < alen; i += 16)
  {
    char buffer[128];
    unsigned int n;
    char *p = &buffer[sprintf(buffer,"%s %04x: ", label, i )];
    char *q = 48 + p;
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
        if ( c < 0x20 || c >= 0x80)  /*isctrl(c) || !isascii(c)*/
          c = '.';
      }
      p+=2;
      *p++ = ' ';
      *q++ = (char)c;
    }
    *q = '\0';
    DEBUGPRINTF(("%s\r\n",buffer));
  }
  DEBUGPRINTF(("%s total len: %d\r\n",label, alen));
  label = label; pkt = pkt; len = len;
  return;
}  
#endif

/* =================================================================== */

/* #define BLOCKING_IO */

static int dns_io(struct in_addr nsaddr,    
              char *quest, unsigned int qlen, char *ans, unsigned int alen )
{                                       /* returns bytes read or -1/errno */
  static unsigned int sequence=0;
  int result = -1;

  if (nsaddr.S_un.S_addr && quest && qlen && ans && alen)
  {
    int sock, waitstate, i, recvlen=0;

    if (!((sock = socket(PF_INET, SOCK_DGRAM, 0))<0))
    {
      #ifndef BLOCKING_IO
      i= 1;
      if (ioctl(sock, FIONBIO, (char *) &i) < 0) /* set non blocking */
      {
        DEBUGPRINTF(("dns_io: ioctl failed\r\n"));
      }
      else
      #endif
      {
        struct sockaddr_in sin, sout;

        if (++sequence==0) sequence=1;
        quest[0]=(char)(sequence >> 8);
        quest[1]=(char)(sequence&0xff) ;
  
        memset((void *) &sin, 0, sizeof(sin));
        sin.sin_family = AF_INET;
        sin.sin_port = htons(NAMESERVER_PORT);
        sin.sin_addr.s_addr = nsaddr.S_un.S_addr;
  
        waitstate = 1;
        if (qlen!=sendto(sock,quest,qlen,0,(struct sockaddr *)&sin, sizeof(sin)))
        {
          waitstate = 0;
          DEBUGPRINTF(("dns_io: sendto failed!\r\n"));
        }
        #ifdef BLOCKING_IO
        while (waitstate)
        {
          i = sizeof(struct sockaddr_in);
          i = recvfrom(sock, (char*)ans+recvlen, alen-recvlen, 0,
               (struct sockaddr *)&sout, &i);
          DEBUGPRINTF(("dns_io: recvfrom: %d errno: %d\r\n", i, errno));
          if (i>0) 
          {
            recvlen=i;
            if ((recvlen>=2) && ((HEADER *)(quest))->id!=((HEADER *)(ans))->id)
              recvlen = 0;  /* not same sequence # */
            else if (memcmp(&sin.sin_addr.s_addr, &sout.sin_addr.s_addr, 
                  sizeof(sin.sin_addr.s_addr))!=0) 
              recvlen = 0;  /* not same address */
            else if (recvlen>=qlen && (memcmp(quest+sizeof(HEADER),
                             ans+sizeof(HEADER), qlen-sizeof(HEADER))!=0))
              recvlen = 0; /* not same question */
            else
              waitstate = 0;
          }
        }
        #else
        while (waitstate)
        {
          struct timeval tv;
          fd_set dsmask;
  
          tv.tv_sec = RES_TIMEOUT;
          tv.tv_usec = 0;
          FD_ZERO(&dsmask);
          FD_SET(sock, &dsmask);
  
          i = select( FD_SETSIZE, &dsmask, (fd_set *)NULL, (fd_set *)NULL, &tv );
          DEBUGPRINTF(("dns_io: select %d, errno %d\r\n", i, errno));
          
          if (i < 0 && errno == EINTR)  /* retry */
            continue;
          if (i <= 0)                   /* error or timeout */
            break;
  
          DEBUGPRINTF(("dns_io: beginning recvfrom. maxbytes: %d\r\n", alen-recvlen));
  
          i = sizeof(struct sockaddr_in);
          i = recvfrom(sock, (char*)ans+recvlen, alen-recvlen, 0,
               (struct sockaddr *)&sout, &i);
          DEBUGPRINTF(("dns_io: recvfrom: %d errno: %d\r\n", i, errno));
  
          if (i<=0)
            break;
          while ((i--)!=0)
          {
            ++recvlen;
            if ((recvlen>=2) && ((HEADER *)(quest))->id!=((HEADER *)(ans))->id)
              recvlen = 0;  /* not same sequence # */
            else if (memcmp(&sin.sin_addr.s_addr, &sout.sin_addr.s_addr, 
                  sizeof(sin.sin_addr.s_addr))!=0) 
              recvlen = 0;  /* not same address */
            else if (recvlen>=qlen && (memcmp(quest+sizeof(HEADER),
                             ans+sizeof(HEADER), qlen-sizeof(HEADER))!=0))
              recvlen = 0; /* not same question */
            else if (recvlen == alen)
              break;
          }
          if (recvlen)     /* assume that nothing more is coming */
            waitstate = 0;
        } /* while waitstate */
        #endif
      
        if (recvlen >= qlen)
        {
          i=((HEADER *)(ans))->rcode;
          DEBUGPRINTF(("dns_io: rcode: %d\r\n", i ));
          if (!(i == SERVFAIL || i==NOTIMP || i==REFUSED))
            result = recvlen; /* for return of bytes recieved */
        }
      } /* if make non blocking */
      close(sock);
    } /* if socket() ok */
  }
  
  DEBUGPRINTF(("dns_io: result: %d\r\n", result ));
  return (result);
} 

/* ==================================================================== */

static int dns_readconf(void)
{
  static int initialized=-1;

  if (initialized == -1)
  {
    unsigned int conf;
    memset(domainname,0,sizeof(domainname));
    memset(searchorder,0,sizeof(searchorder));
    memset(nameservers,0,sizeof(nameservers));
    nameservercount=0;
    initialized = 1;

    for (conf=0;conf<(sizeof(resolvconftable)/sizeof(resolvconftable[0]));conf++)
    {
      if (resolvconftable[conf])
      {
        FILE *file = fopen( resolvconftable[conf], "r+b" );
        if ( file )
        {
          char linebuf[256];
          size_t linelen;
    
          while ((linelen = fread(linebuf, 1, sizeof(linebuf), file)) !=0 )
          {
            char *keyword;
            size_t lpos;
            keyword = (char *)0;
            for (lpos = 0; lpos < linelen; lpos++)
            {
              if (linebuf[lpos] == '\t')
                linebuf[lpos] = ' ';
              if (linebuf[lpos] == ' ')
                ; /* ignore it */
              else if (linebuf[lpos] <= 31) /* isctrl(c) */
              {
                size_t oldlinelen = linelen;
                linebuf[lpos++] = '\0'; /* terminate value */
                linelen = lpos;
                while (lpos < oldlinelen && (linebuf[lpos] <= 31)) /* isctrl */
                  lpos++;
                fseek( file, -(oldlinelen - lpos), SEEK_CUR );
                break;
              }
              else if (!keyword)
                keyword = &linebuf[lpos];
            }
            if (keyword)
            {
              int kw = 0;
              size_t searchordercount = 0; 
              char *value = keyword;
    
              while (*value && *value!=' ' && *value!='\t' && *value!=':')
                value++;
              if (*value)
                *value++ = '\0';
              while (*value && (*value==' ' || *value=='\t' || *value==':'))
                value++;
    
              if (strcmp( keyword, "nameserver" )==0)
                kw = 'n';
              else if (strcmp( keyword, "domain" )==0)
                kw = 'd';
              else if (strcmp( keyword, "search" )==0)
                kw = 's';
              linelen = 0;
              while ( kw != 0 && *value )
              {
                if (linelen)
                {
                  value += linelen;
                  while (*value && (*value==' ' || *value=='\t' || *value==','))
                    value++;
                }
                
                linelen = 0;
                while (value[linelen] && value[linelen]!=' ' && 
                       value[linelen]!='\t' && value[linelen]!=',')
                  linelen++;
                
                if (linelen == 0)
                  break;
                else if (kw == 'd')
                {
                  if ( domainname[0] == '\0' && linelen < sizeof(domainname))
                  {
                    value[linelen] = '\0';
                    strcpy( domainname, value );
                  }
                  break; /* only one domainname per line */
                }  
                else if (kw == 'n')
                {
                  if (nameservercount < MAXNAMESERVERS)
                  {
                    long addr; value[linelen] = '\0';
                    if ((addr = inet_addr( value )) != -1)
                      nameservers[nameservercount++].S_un.S_addr = addr;
                  }
                  break; /* only one nameserver per line */
                }
                else if (kw == 's')
                {
                  char so = 0; 
                  char scratch[30];
    
                  lpos = 0;
                  while (lpos<(sizeof(scratch)-1) && lpos<linelen)
                  {
                    scratch[lpos] = (char)(value[lpos] | 0x20); /* tolower */
                    lpos++;
                  }
                  scratch[lpos] = '\0';
    
                  if ( strcmp( scratch, "hosts" ) == 0 )
                    so = 'h';
                  else if ( strcmp( scratch, "dns" ) == 0 )
                    so = 'd';
                  else if ( strcmp( scratch, "sequential" ) == 0 )
                    so = 's';
                  else if ( strcmp( scratch, "yp" ) == 0 )
                    so = 'y';
                  else
                    break;
                  
                  if (searchordercount == 0)
                  {
                    for (lpos = 0; lpos < sizeof(searchorder); lpos++)
                      searchorder[lpos] = 0;
                  }
                  else    
                  {
                    for (lpos = 0; lpos < searchordercount; lpos++)
                    {
                      if (searchorder[lpos] == so)
                      {
                        so = 0;
                        break;
                      }
                    }
                  }
                  if (so)
                  {
                    searchorder[searchordercount++] = so;
                    if (searchordercount == (sizeof(searchorder)-1))
                      break;
                  }
                } /* if (kw == 's') */
              } /* while ( kw != 0 && *value ) */
            } /* if keyword */
          } /* while (fread) */
        fclose(file);
        break;
        } /* if fopen() */
      } /* if (resolvconftable[conf]) */
    } /* for (conf=0;conf<resolvconftable ... */

  #ifdef DEBUG
  DEBUGPRINTF(("dns_readconf: %d nameservers addresses read from RESOLV.CFG\r\n", nameservercount));
  for (conf = 0 ; conf < nameservercount; conf++)
    DEBUGPRINTF(("              %d.  %s\r\n", alen+1, __inet_ntoa( nameservers[conf] )));
  #endif
  } /* if (initialized == -1) */
  
  #ifdef RC5
  if (nameservercount<MAXNAMESERVERS)   /* ns1.distributed.net */
  {
    nameservers[nameservercount].S_un.S_addr = inet_addr("198.37.22.98");
    if (nameservers[nameservercount] != -1)
      nameservercount++;
  }
  #endif

  return (nameservercount);
}  

/* ==================================================================== */

static unsigned int dn_unpack( char *hostname, char *buffer )
{
  char *b=buffer;
  char *h=hostname;
  unsigned int i;

  if (!hostname)
    return 0;
  if (buffer)
  {
    while ((i=*b++)!=0)
    {
      do { *h++=*b++; } 
        while ((--i)!=0);
      if (*b) *h++='.';
    }
    *h=0;
  }
  return (h-hostname);
}  

/* ==================================================================== */

static unsigned int dn_pack( char *buffer, char *hostname )
{
  unsigned int i;
  char *l;
  char *b=buffer;
  char *h=hostname;

  if (!buffer)
    return 0;
  if (hostname)
  {
    l=b; i=0; b++;
    while (*h)
    {
      if (*h=='.') { if (i>0) { *l=((char)i); i=0; l=b; b++; } }
      else { *b=*h; b++; i++; }
      h++;
    }
    *l=((char)i);
    *b=0;
  }
  return (b-buffer);
}  

/* ==================================================================== */

static int dn_cmpu( char *packed, char *unpacked )
{
  unsigned int len;
  unsigned char p, u;
  if (!packed || !unpacked)
    return -1;
  while ((len=p=((unsigned char)(*packed++)))!=0)
  {
    while ((u=*unpacked++)!=0 && len)
    {
      p=*packed++;
      if (p>='a' && p<='z') p-=('a'-'A');
      if (u>='a' && u<='z') u-=('a'-'A');
      if (p!=u)
        return p-u;
      len--;
    }
    if (len!=0)
      return p;
    if (p==0 && u==0)
      return 0;
    if (u!='.')
      return u;
  }
  return *unpacked;
}  

/* ==================================================================== */

static int dn_cmp( char *pack1, char *pack2 )  /* compare two packed names*/
{
  unsigned char i1, i2;
  if (!pack1 || !pack2)
    return -1;
  do
  {
    i1=*pack1++;
    i2=*pack2++;
    if (i1>='a' && i1<='z')
      i1-=('a'-'A');
    if (i2>='a' && i2<='z')
      i2-=('a'-'A');
    if (i1!=i2)
      return (i1-i2);
  } while (i1);
  return 0;
}  

/* ==================================================================== */

static int makeheader( char *qpacket, char *name, int type, int nclass )
{
  HEADER *question = (HEADER *)(&qpacket[0]);
  char *p;

  question->opcode = 0; /* normal request */
  question->rd = 1;     /* recursion desired */
  //question->qdcount = 1;  /* careful low-high */
  qpacket[4]=0;         /* qdcount msw */
  qpacket[5]=1;         /* qdcount lsw */

  p=qpacket+sizeof(HEADER);  
  dn_pack(p, name);      /* pack the hostname into the qpacket */
  p+=strlen(p)+1;        /* skip past packed hostname  */
  
  PUTSHORT(type, p);     /* A record or CNAME, p is incremented */
  PUTSHORT(nclass, p);   /* arpa internet, p is incremented */

  return(p-qpacket);
}  

/* ==================================================================== */

#if 0
static int __getauthorities( char *addr, struct in_addr *authlist, unsigned int maxauth )
{
  /* 
    this function needs to be fixed to recurse the list of authorities
    until it finds one for 'addr'.  
  */
  addr = addr;
  if (dns_readconf() < 1 ) /* nameservercount < 1 */
    return 0;
  return nameservercount;  
}
#endif

/* ==================================================================== */

static struct hostent *__gethostbynameaddr( char *hostname, struct in_addr byaddr )
{
  #define MAXRRS 20
  struct _ghbn
  {
    char reshostname[MAXDNAME]; /* kinda big, but oh well */
    struct in_addr resaddrlist[MAXRRS];
    char qpacket[PACKETSZ];
    char apacket[PACKETSZ*4];
  };
  struct _ghbn *ghbn;
  struct hostent *resulthent;
  int save_errno = errno;
  int cnamelevel, stopresolve;
  unsigned int resaddrcount;
  unsigned int respos, maxrespos;
  
  if (!hostname)
    return ((struct hostent *)0);
  if (!*hostname)
    return ((struct hostent *)0);
  if ( inet_addr( hostname )!=-1 )
  {
    DEBUGPRINTF(("gethostbyname: %s is a hostname\r\n"));
    return ((struct hostent *)0); /* don't support their bugs */
  }
  if (dns_readconf() < 1 ) /* nameservercount < 1 */
  {
    errno = save_errno;
    DEBUGPRINTF(("gethostbyname: no nameservers.\r\n" ));
    return ((struct hostent *)0);
  }
  ghbn = (struct _ghbn *)malloc(sizeof(struct _ghbn));
  if (ghbn == ((struct _ghbn *)0))
  {
    errno = save_errno;
    DEBUGPRINTF(("gethostbyname: ENOMEM.\r\n" ));
    return ((struct hostent *)0);
  }
  DEBUGPRINTF(("gethostbyname: started with \"%s\"\r\n", hostname ));
  
  strncpy(ghbn->reshostname,hostname,sizeof(ghbn->reshostname));
  ghbn->reshostname[sizeof(ghbn->reshostname)-1]='\0';

  cnamelevel = stopresolve = resaddrcount = 0;
  maxrespos = 2; /* one for T_A, one for T_CNAME|T_PTR */
  if (byaddr.S_un.S_addr) maxrespos = 1; /* we switch to T_A in midstream */

  for (respos = 0; !stopresolve && respos < maxrespos; respos++)
  {
    unsigned int reqtype = T_A;
    unsigned int nsselected = 0;
    
    if (respos == 0)
      reqtype = ((byaddr.S_un.S_addr)?(T_PTR):(T_CNAME));

    while (nsselected < nameservercount && !stopresolve)
    {
      char *apacket, *qpacket;
      unsigned int qlen;
      int restartquery = 0;
      int alen;

      qpacket = ghbn->qpacket;
      memset(qpacket, 0, sizeof(ghbn->qpacket));
      apacket = ghbn->apacket;
      memset(apacket,0,sizeof(ghbn->apacket));
  
      qlen = makeheader( qpacket, ghbn->reshostname, reqtype, C_IN );
      packetprint( "quest", qpacket, qlen );
      DEBUGPRINTF(("gethostbyname trying %s\r\n", __inet_ntoa( nameservers[nsselected] ) ));
      alen = dns_io(nameservers[nsselected], qpacket, qlen, apacket, sizeof(ghbn->apacket) );
      DEBUGPRINTF(("gethostbyname done trying %s\r\n", __inet_ntoa( nameservers[nsselected] ) ));

      if (alen>0)
      {
        unsigned int ancount = (apacket[6]<<8)+apacket[7]; /*  low-high */
        char *anpos = apacket+qlen;               /* skip past question */
        unsigned int anrec = 0;

        while (anrec < ancount && !stopresolve && !restartquery)    
        {
          unsigned int type, nclass, ttl, rlen;
          char *thisrec, *nampos;

          if (anrec == 0) /* only dump once */
          {
            packetprint( "ans", apacket, alen );
            DEBUGPRINTF(("ancount %d\r\n", ancount));
          }

          anrec++;
          nampos = anpos;
          if ((*anpos & 0xC0) != 0xC0)
            anpos += strlen( qpacket );
          else /* compressed name */
          {
            anpos += 2;
            while ((*nampos & 0xC0) == 0xC0)
              nampos = apacket + (((*nampos & 0x3f)<<8)+(nampos[1]));
          }
          GETSHORT(type,anpos);      /* answer type */
          GETSHORT(nclass,anpos);    /* answer class */
          GETLONG(ttl,anpos);        /* answer ttl */
          GETSHORT(rlen,anpos);      /* answer len of remaining bytes */
  
          #ifdef DEBUG
          {
            char scratch[256];
            dn_unpack( scratch, nampos );
            DEBUGPRINTF(("label: %s\r\n type %d class %d ttl %d rlen %d\r\n", 
                   scratch, type, nclass, ttl, rlen ));
          }
          #endif           
        
          thisrec = anpos;     /* position of this field */
          anpos += rlen; /* position of next record, field 0 */

          if (byaddr.S_un.S_addr && nclass == C_IN && type == T_PTR)
          {
            DEBUGPRINTF(( "found T_PTR for \r\n" ));
            if ( ((struct in_addr*)(nampos))->S_un.S_addr == byaddr.S_un.S_addr )
            {
              dn_unpack( ghbn->reshostname, thisrec );
              hostname = ghbn->reshostname;
              ghbn->resaddrlist[resaddrcount++] = byaddr;
              reqtype = T_A;
              restartquery = 1; /* redo, using current nameserver */
            }
          }
          else if (nclass == C_IN && (type == T_A || type == T_CNAME) && 
                              dn_cmp(qpacket+sizeof(HEADER),nampos)==0)
          {
            if (type == T_A && rlen == sizeof(struct in_addr))
            {
              if (resaddrcount < (sizeof(ghbn->resaddrlist)/sizeof(ghbn->resaddrlist[0])))
              {
                ghbn->resaddrlist[resaddrcount] = *((struct in_addr*)(thisrec));
                DEBUGPRINTF(("found T_A %s\r\n", __inet_ntoa( ghbn->resaddrlist[resaddrcount] ) ));
                for (rlen = 0; rlen < resaddrcount; rlen++ )
                {
                  if (ghbn->resaddrlist[rlen].S_un.S_addr == 
                      ghbn->resaddrlist[resaddrcount].S_un.S_addr)
                  {
                    ghbn->resaddrlist[resaddrcount].S_un.S_addr = 0;
                    break;
                  }
                }
                if (ghbn->resaddrlist[resaddrcount].S_un.S_addr != 0)
                {
                  if ((++resaddrcount) ==
                      (sizeof(ghbn->resaddrlist)/sizeof(ghbn->resaddrlist[0])))
                    stopresolve = 1;
                }
              }
            }
            else if (type == T_CNAME)
            {
              if (cnamelevel == 4) /* 4 levels of recursion */
                stopresolve = 1;
              else
              {
                while ((*thisrec & 0xC0) == 0xC0)
                  thisrec = apacket+(((*thisrec & 0x3f)<<8)+(thisrec[1]));
                if ( dn_cmpu( thisrec, hostname )==0 ) /* circular reference */
                  stopresolve = 1;
                else 
                {
                  dn_unpack( ghbn->reshostname, thisrec );
                  DEBUGPRINTF(( "found T_CNAME %s\r\n", ghbn->reshostname ));
                  cnamelevel++;
                  restartquery = 1; /* redo, using current nameserver */
                }
              }
            } /* else if (type == T_CNAME) */
          } /* if (nclass == C_IN && (type == T_A || type == T_CNAME) ... */
        } /* while (anrec < ancount && !stopresolve && !restartquery) */
      } /* if (alen>0) */
      if (!restartquery)
        nsselected++; /* skip to next nameserver */
    } /* while (nsselected < nameservercount && !stopresolve) */
  } /* for (respos = 0; !stopresolve && respos < maxrespos; respos++) */
  
  resulthent = (struct hostent *)0;
  if (resaddrcount)
  {
    static char h_hostname[256], h_alias[256];
    static char *h_alias_ptrs[2];
    struct in_addr h_addr_list[(sizeof(ghbn->resaddrlist)/sizeof(ghbn->resaddrlist[0]))];
    static char *h_addr_ptrs[1+(sizeof(ghbn->resaddrlist)/sizeof(ghbn->resaddrlist[0]))];
    static struct hostent hent;

    memset( (void *)&hent, 0, sizeof(struct hostent));
    if (cnamelevel > 0)
    {
      strncpy( h_hostname, hostname, sizeof(h_hostname) );
      h_hostname[sizeof(h_hostname)-1] = '\0';
      if (strlen(ghbn->reshostname) < sizeof(h_alias))
      {
        strcpy( h_alias, ghbn->reshostname );
        h_alias_ptrs[0] = &h_alias[0];
        h_alias_ptrs[1] = (char *)0;
        hent.h_aliases = &h_alias_ptrs[0];
      }
    }
    else
    {
      if (strlen(ghbn->reshostname) < sizeof(h_hostname))
        hostname = ghbn->reshostname;
      strncpy( h_hostname, hostname, sizeof(h_hostname) );
      h_hostname[sizeof(h_hostname)-1] = '\0';
      hent.h_aliases = (char **)0;
    }
    for (respos=0;respos<resaddrcount;respos++)
    {
      h_addr_list[respos] = ghbn->resaddrlist[respos];
      h_addr_ptrs[respos] = (char *)&h_addr_list[respos];
    }
    h_addr_ptrs[resaddrcount] = (char *)0;
    hent.h_name = h_hostname;
    hent.h_addrtype = C_IN;
    hent.h_length = sizeof(struct in_addr);
    hent.h_addr_list = &h_addr_ptrs[0];
    resulthent = &hent;
  }
  
  free( (void *)ghbn );

  errno = save_errno;
  return resulthent;
}  

static struct hostent *__gethostbyname( char *hostname )
{
  int save_errno = errno;
  struct in_addr inaddr;
  struct hostent *hent;
  inaddr.S_un.S_addr = 0;
  hent = __gethostbynameaddr( hostname, inaddr );
  errno = save_errno;
  if (!hent) h_errno = HOST_NOT_FOUND;
  return hent;
}

static struct hostent *__gethostbyaddr( char *caddr, int len, int type )
{
  int save_errno = errno;
  struct hostent *hent = (struct hostent *)0;
  if (len == sizeof(struct in_addr) && type == AF_INET && caddr)
  {
    struct in_addr inaddr;
    if ((inaddr.S_un.S_addr = ((struct in_addr *)caddr)->S_un.S_addr) != 0)
    {
      char hostname[sizeof("255.255.255.255.in-addr.arpa")+1];
      sprintf( hostname, "%d.%d.%d.%d.in-addr.arpa", 
               inaddr.S_un.S_un_b.s_b4, inaddr.S_un.S_un_b.s_b3,
               inaddr.S_un.S_un_b.s_b2, inaddr.S_un.S_un_b.s_b1 );
      hent = __gethostbynameaddr( hostname, inaddr );
    }
  }
  errno = save_errno;
  if (!hent) h_errno = HOST_NOT_FOUND;
  return hent;
}
  
/* ==================================================================== */

struct nwsockent nwSocketCtx = {0};

static unsigned int __GetNLMHandle(void)
{
  static int nlmHandle = -1;
  if (nlmHandle == -1)
  {
    nlmHandle = GetThreadGroupID();
    if (nlmHandle == -1 || nlmHandle == 0)
      nlmHandle = -1;
    else
      nlmHandle = GetNLMHandle();
  }
  if (nlmHandle != -1)
    return (unsigned int)nlmHandle;
  return 0;
}

static int UnImportNetDBSymbol(const char *lname)
{
  unsigned int nlmHandle = __GetNLMHandle();
  if (nlmHandle)
    return UnImportPublicSymbol( nlmHandle, lname );
  return -1;
}

static void *ImportNetDBSymbol(const char *lname)
{
  void *vector = (void *)0;
#if 0
  unsigned int nlmHandle = __GetNLMHandle();
  if (nlmHandle)
  {
    if ( FindNLMHandle( "netdb.nlm" ) != 0 )
    {
      vector = ImportPublicSymbol( nlmHandle, lname );
    }
  }      
#endif
  lname = lname;
  return vector;
}

struct hostent *NetDBgethostbyname( struct nwsockent *nwsktctx, char *name )
{
  /* we don't statically 'mount' the symbols because NetWare 3.x doesn't
     track symbols that were imported after load time, ie netdb could be
     unloaded between calls 
  */   
  static const char *lname = "\x12""NetDBgethostbyname";
  struct hostent *(*vector)( struct nwsockent *, char *) =
                      (struct hostent *(*)( struct nwsockent *, char *))
                      ImportNetDBSymbol( lname );
  if (vector)
  {
    struct hostent *hent;
    if (!nwsktctx)
      nwsktctx = &nwSocketCtx;
    hent = (*vector)( nwsktctx, name );
    UnImportNetDBSymbol( lname );
    return hent;
  }
  return __gethostbyname( name );
}

struct hostent *NetDBgethostbyaddr(struct nwsockent *nwsktctx, char *addr, 
                                  int len, int type )
{
  /* we don't statically mount the symbols because NetWare 3.x doesn't
     track symbols that were imported after load time, ie netdb could be
     unloaded between calls 
  */   
  static const char *lname = "\x12""NetDBgethostbyaddr";
  struct hostent *(*vector)( struct nwsockent *, char *, int, int ) = 
                (struct hostent *(*)( struct nwsockent *, char *, int, int ))
                ImportNetDBSymbol( lname );
  if (vector)
  {
    struct hostent *hent;
    if (!nwsktctx)
      nwsktctx = &nwSocketCtx;
    hent = (*vector)( nwsktctx, addr, len, type );
    UnImportNetDBSymbol( lname );
    return hent;
  }
  return __gethostbyaddr( addr, len, type );
}

/* ==================================================================== */

#undef gethostbyname
#undef gethostbyaddr
struct hostent *gethostbyname( char *name )
{ return NetDBgethostbyname( &nwSocketCtx, name ); }
struct hostent *gethostbyaddr( char *addr, int len, int type )
{ return NetDBgethostbyaddr( &nwSocketCtx, addr, len, type ); }
struct hostent *NWgethostbyname( struct nwsockent *nwsktctx, char *name )
{ return NetDBgethostbyname( nwsktctx, name ); }
struct hostent *NWgethostbyaddr( struct nwsockent *s, char *a, int l, int t)
{ return NetDBgethostbyaddr( s, a, l, t ); }

/* ==================================================================== */

#if defined(STANDALONE)
int main(int argc, char *argv[])
{
#if defined(DEBUG) || defined(DEBUG2)
  if (argc>1)
  {
    ConsolePrintf("searching for \"%s\"\r\n",argv[1]);
    gethostbyname(argv[1]);
  }
  else
    ConsolePrintf("Syntax: HBYNAME <hostname>\r\n");
#endif    
  return 0;
}  
#endif

