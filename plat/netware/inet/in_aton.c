/*
 * Classification: arpa
 * Service: Internet Network Library
 * Author: Cyrus Patel <cyp@fb14.uni-mainz.de>
 * Copyright: none
 *
 * $Log: in_aton.c,v $
 * Revision 1.1.2.1  2001/01/21 15:10:30  cyp
 * restructure and discard of obsolete elements
 *
 * Revision 1.1.2.1  1999/11/14 20:44:22  cyp
 * all new
 *
 *
*/

#if defined(__showids__)
const char *in_aton_c(void) { 
return "$Id: in_aton.c,v 1.1.2.1 2001/01/21 15:10:30 cyp Exp $"; } 
#endif


#include <netinet/in.h> /* struct in_addr */
#include <arpa/inet.h>  /* inet_aton, inet_addr, inet_network, inet_netof */


static int _inet_atox( const char *cp, struct in_addr *inp, int minparts, int addrtype )
{
  int err = 1;

  if (cp)
  {
    int parts = 0, bracket = 0;
    unsigned long maxval = 0xfffffffful;
    unsigned long buf[4];

    err = 0;

    if ( *cp == '[' && addrtype != 'n')
    {
      bracket = 1;
      cp++;
    }

    while (*cp && !err)
    {
      register unsigned long val;
      unsigned int radix, len;
    
      radix = 10;
      len = 0;
      val = 0;
      
      if ( addrtype == 'n' ) /* for inet_network() each component can */
        maxval = 0xff;       /* only be 8 bits */

      if ( *cp == '0' )
      {
        radix = 8;
        cp++;
        len = 1;
      }
      if ( *cp == 'X' || *cp == 'x')
      {
        radix = 16;
        cp++;
        len = 0;
      }
        
      do
      {
        register int c = (int)*cp;
        if (c >= '0' && c <= '9') /* safe */
          c -= '0';
        else if (radix == 16 && c >= 'A' && c <= 'F') /* safe */
          c = 10 + (c - 'A');
        else if (radix == 16 && c >= 'a' && c <= 'f') /* safe */
          c = 10 + (c - 'a');
        else
          break;
        if ( c < 0 || c >= radix )
          err = 1;
        else if (val > ((maxval - c) / radix))
          err = 1;
        else if (( val *= radix ) > maxval )
          err = 1;
        else
        {
          val += c;
          cp++;
          len++;
        }
      } while ( !err && val <= maxval );
  
      if ( err || val > maxval || len == 0 )
        err = 1;
      else if (!( parts < 4 )) /* already did 4 */
        err = 1;
      else if (parts && buf[parts-1]>0xff) /* completed parts must be <=0xff */
        err = 1;
      else
      {
        buf[parts++] = val;
        maxval >>= 8; /* for next round */
        if ( *cp != '.' )
          break;
        cp++;
        if (!*cp)
          err = 1;
      }
    } /* while (*cp && !err) */

    if (err || parts == 0 || parts < minparts)
      err = 1;
    else if ( bracket && *cp !=']' ) /* unmatched bracket */
      err = 1;
    else if (*cp && *cp!='\n' && *cp!='\r' && *cp!='\t' && *cp!=' ')
      err = 1;
    else if (inp)
    {
      register int n;
      if (addrtype == 'n')                   /* inet_network() */
      {
        inp->s_addr = 0;
        for ( n = 0; n < parts; n++ )
        {
          inp->s_addr <<= 8;
          inp->s_addr |= (buf[n] & 0xff);
        }
      }
      else                                   /* inet_aton() or inet_addr() */
      {  
        char *p = (char *)(&(inp->s_addr));
        if (parts == 1)
          maxval = buf[0];
        else if (parts == 2)
          maxval = (buf[0]<<24) | buf[1];
        else if (parts == 3)
          maxval = (buf[0]<<24) | (buf[1]<<16) | buf[2];
        else if (parts == 4)
          maxval = (buf[0]<<24) | (buf[1]<<16) | (buf[2]<<8) | buf[3];
        inp->s_addr = 0;
        for ( n= 0; n < 4; n++ )
          *p++ = (char)((maxval >> ((3-n)<<3)) & 0xff);
      }
    }
  } /* if (cp) */
  
  return !err; /* return 0 if valid, !0 if not */
}  


/*
  inet_aton(const char *cp, struct in_addr *inp) converts the internet
  host address <cp> from the standard numbers-and-dots notation into binary
  data in network byte order and stores it in the structure that <inp> 
  points to. inet_aton returns non-zero if <cp> is a valid host address, 
  zero if not. It does not update <errno>.
*/

int inet_aton( const char *cp, struct in_addr *inp )
{
  return (_inet_atox( cp, inp, 1, 'h' ));
}


/*
  inet_addr(const char *cp) converts the character string <cp> from
  the standard numbers-and-dots notation into a long value in network
  byte order that can be used as an internet host address. Upon success, 
  it returns the internet address value and -1 upon detection of malformed 
  requests. It does not update errno.  

  Keep in mind that -1 is a valid internet address (255.255.255.255) and 
  inet_aton provides a cleaner way to indicate error return. 
  
  a        When only one part of the address is given, the value is stored 
           directly in the network address without any byte rearrangement.
           (for example "127" will be returned as 127)
  a.b      When a two-part address is supplied, the last part is interpreted 
           as a 24-bit quantity and placed in the rightmost 3 bytes of the 
           network address. This makes the two-part address format convenient 
           for specifying Class A network addresses in net.host format 
           (for example, "89.1" where the network address is "89" and the 
           host address is "1", will be returned as 89<<24 + 1).
  a.b.c    When a three-part address is specified, the last part is 
           interpreted as a 16-bit quantity and placed in the rightmost 
           2 bytes of the network address. This makes the three-part address 
           format convenient for specifying Class B network addresses in 
           net.host format (for example, "128.1.3" where the network address 
           is "128.1" and the host address is "3" will be returned as 
           128<<24 + 1 << 16 + 3 << 8).
  a.b.c.d  When four parts are specified, each part is interpreted as a 
           byte of data. "134.93.246.124" will be returned as 
           134<<24 + 93 << 16 + 246 << 8 + 124
*/  

unsigned long inet_addr(const char *cp)
{
  struct in_addr inaddr;                
  if (_inet_atox( cp, &inaddr, 1, 'h' ) == 0) 
    return ((unsigned long)(-1L));
  return (unsigned long)inaddr.s_addr;
}



/*
  inet_network(const char *cp) extracts the network number in network
  byte order from the address cp in numbers-and-dots notation. inet_network
  returns -1 upon detection of malformed requests. It does not update the 
  errno. Note that inet_network, unlike inet_addr(), does not rearrange the 
  address by a[.b[.c[.d]]] components: "89.1" will be returned as 
  89<<24 + 1<<16 + 0<<8 + 0.
*/  

unsigned long inet_network(const char *cp)
{
  struct in_addr inaddr;
  if (_inet_atox( cp, &inaddr, 1, 'n' ) == 0)
    return ((unsigned long)(-1L));
  return (unsigned long)inaddr.s_addr;
}
