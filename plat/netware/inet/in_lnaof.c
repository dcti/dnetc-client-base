/*
 * Classification: arpa
 * Service: Internet Network Library
 * Copyright: none
 * Author: Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * $Log: in_lnaof.c,v $
 * Revision 1.1.2.1  2001/01/21 15:10:30  cyp
 * restructure and discard of obsolete elements
 *
 * Revision 1.1.2.1  1999/11/14 20:44:22  cyp
 * all new
 *
 *
*/

#if defined(__showids__)
const char *in_lnaof_c(void) { 
return "$Id: in_lnaof.c,v 1.1.2.1 2001/01/21 15:10:30 cyp Exp $"; }
#endif

#include <netinet/in.h> /* struct in_addr */
#include <arpa/inet.h>  /* inet_lnaof() */

/*
 * inet_lnaof( struct in_addr ) return the local network address part 
 * of an internet address. 
*/


unsigned long inet_lnaof( struct in_addr inaddr )
{
  unsigned long i = (((unsigned long)(inaddr.S_un.S_un_b.s_b4))      ) | 
                    (((unsigned long)(inaddr.S_un.S_un_b.s_b3)) <<  8) |
                    (((unsigned long)(inaddr.S_un.S_un_b.s_b2)) << 16) |
                    (((unsigned long)(inaddr.S_un.S_un_b.s_b1)) << 24);
  if (((long)(i) & 0x80000000) == 0)
    i &= 0x00ffffffUL;
  else if (((long)(i) & 0xc0000000) == 0x80000000)
    i &= 0x0000ffffUL;
  else
    i &= 0x000000ffUL;
  return i;
}                      

