/*
 * Classification: arpa
 * Service: Internet Network Library
 * Author: Cyrus Patel <cyp@fb14.uni-mainz.de>
 * Copyright: none
 *
 * $Log: in_makea.c,v $
 * Revision 1.1.2.1  2001/01/21 15:10:30  cyp
 * restructure and discard of obsolete elements
 *
 * Revision 1.1.2.1  1999/11/14 20:44:22  cyp
 * all new
 *
 *
*/

#if defined(__showids__)
const char *in_makea(void) { 
return "$Id: in_makea.c,v 1.1.2.1 2001/01/21 15:10:30 cyp Exp $"; }
#endif


#include <netinet/in.h> /* struct in_addr */
#include <arpa/inet.h>  /* inet_makeaddr() */

/*
  inet_makeaddr() makes an internet host address in network byte order
  by combining the network number net with the local address host
  in network net, both in local byte order.
*/  

struct in_addr inet_makeaddr( u_long net, u_long host )
{
  struct in_addr inaddr;

  if ( net < 0x80ul )          /* IN_CLASSA_MAX */
  {
    net <<= 24;                /* IN_CLASSA_NSHIFT */
    host &= 0x00fffffful;      /* IN_CLASSA_HOST */
  }
  else if ( net < 0x10000ul )  /* IN_CLASSB_MAX */
  {
    net <<= 16;                /* IN_CLASSB_NSHIFT */
    host &= 0x0000fffful;      /* IN_CLASSB_HOST */
  }
  else if ( net < 0x1000000ul ) /* IN_CLASSC_NET */
  {
    net <<= 8;                 /* IN_CLASSC_NSHIFT */
    host &= 0x000000fful;      /* IN_CLASSC_HOST */
  }
  
  host |= net;
  inaddr.S_un.S_addr = 0;
  inaddr.S_un.S_un_b.s_b4 = host & 0Xff;
  inaddr.S_un.S_un_b.s_b3 = (host >> 8) & 0Xff;
  inaddr.S_un.S_un_b.s_b2 = (host >> 16) & 0Xff;
  inaddr.S_un.S_un_b.s_b1 = (host >> 24) & 0Xff;
  return inaddr;
}

