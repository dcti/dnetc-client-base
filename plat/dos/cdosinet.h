/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * 
 * -------------------------------------------
 * just stubs...
 * -------------------------------------------
 *
*/
#ifndef __CLIDOS_INET_H__
#define __CLIDOS_INET_H__ "@(#)$Id: cdosinet.h,v 1.1.2.1 2001/01/21 15:10:22 cyp Exp $"

  typedef int SOCKET;

  #if defined(DJGPP)
    #include <netinet/in.h> //ntohl() and htonl()
  #elif defined(__WATCOMC__) //we can do better
    unsigned short ntohs( unsigned short );
    unsigned short htons( unsigned short );
    unsigned long ntohl( unsigned long );
    unsigned long htonl( unsigned long );
    #pragma aux htons = 0x86 0xc4 /* xchg al,ah */ \
                        parm [ax] value[ax] modify exact[ax]
    #pragma aux (htons) ntohs;  /* ntohs is identical to htons */
    #ifdef __386__
    #pragma aux htonl = 0x86 0xc4 0xc1 0xc0 0x10 0x86 0xc4 \
                        parm [eax] value[eax] modify exact[eax]
                     /* xchg al,ah  rol eax,16   xchg al,ah */
    #pragma aux (htonl) ntohl;  /* ntohl is identical to htonl */
    #else
    #define ntohl(x) ((((x)&0xff)<<24) | (((x)>>24)&0xff) | \
                      (((x)&0xff00)<<8) | (((x)>>8)&0xff00))
    #define htonl(x) ntohl(x)                      
    #endif
  #else
    #define ntohs(x) ((((x)&0xff)<<8) | (((x)>>8)&0xff))
    #define htons(x) ntohs(x)
    #define ntohl(x) ((((x)&0xff)<<24) | (((x)>>24)&0xff) | \
                      (((x)&0xff00)<<8) | (((x)>>8)&0xff00))
    #define htonl(x) ntohl(x)                      
  #endif

#endif //__CLIDOS_INET_H__
