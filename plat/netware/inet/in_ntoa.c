/*
 * Classification: arpa
 * Service: Internet Network Library
 * Author: Cyrus Patel <cyp@fb14.uni-mainz.de>
 * Copyright: none
 *
 * $Log: in_ntoa.c,v $
 * Revision 1.1.2.1  2001/01/21 15:10:30  cyp
 * restructure and discard of obsolete elements
 *
 * Revision 1.1.2.1  1999/11/14 20:44:22  cyp
 * all new
 *
 *
*/

#if defined(__showids__)
const char *in_ntoa_c(void) { 
return "$Id: in_ntoa.c,v 1.1.2.1 2001/01/21 15:10:30 cyp Exp $"; } 
#endif

#define NOT_NETWARE_386 /* suppress symbol redefinition macros */
#include <netinet/in.h> /* struct in_addr */
#include <arpa/inet.h>  /* inet_ntoa() */

/*
  inet_ntoa_r(char *buff, struct in_addr inaddr) converts the internet 
  host address from the binary data in network byte order in inaddr and 
  stores the result in standard numbers-and-dots notation in the buffer 
  pointed to by buff. The caller is responsible for ensuring that the 
  buffer is large enough to accomodate the formatted string (>=17 octets). 
  The function returns the value of 'buff'.
*/

char *inet_ntoa_r(char *buff, struct in_addr inaddr)
{
  if ( buff )
  {
    unsigned int i;
    char *a = (char *)(&inaddr), *b = buff;

    for ( i=0; i<4; i++ )
    {
      register unsigned int c = (((unsigned int)(*a++)) & 255 );
      register unsigned int r = ((c/100) | (((c%100)/10)<<8));
      if ( i ) 
        *b++ = '.';
      if (( r & 0xff ) != 0) 
        *b++ = (char)((  r & 0x00ff ) + '0');
      if ( r != 0 )    
        *b++ = (char)((( r & 0xff00 ) >> 8) + '0');
      *b++ = (char)((c%10) + '0');
    }  
    *b++=0;
  }
  return buff;
}



/*
  inet_ntoa(struct in_addr inaddr) converts the internet host address
  from the binary data in network byte order in inaddr and returns a 
  string in standard numbers-and-dots notation. The string is formatted
  in a statically allocated buffer, which subsequent calls will overwrite.
*/

char *inet_ntoa(struct in_addr inaddr)
{ 
  static char buff[sizeof "255.255.255.255  "];
  return inet_ntoa_r( buff, inaddr);
}

char *NWinet_ntoa( char *cp, struct in_addr in)
{
  return inet_ntoa_r( cp, in );
}
