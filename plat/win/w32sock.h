/* Hey, Emacs, this a -*-C++-*- file !
 *
 * ----------------------------------------------------
 * this header file is essential for win386 (win16 extender), non-essential
 * for everything else, but doesn't hurt to include.
 * ----------------------------------------------------
 *
 * Shim layer between Winsock DLL and application.
 * While this shim is essential for Watcom 386 flat memory model extensions
 * for win16, is also used by the win32cli client to allow the client to
 * run on systems without winsock.dll/wsock32.dll/ws2_32.dll
 *
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * This module is known to work with 16bit Novell, Microsoft and Trumpet
 * stacks, as well as the 32bit Alpha and x86 Microsoft stacks.
 *
 * Little endian byte format is assumed throughout.
 *
 * The only functions in the Winsock 1.x spec that are not in this file are
 * the Winsock async extensions.
 * 
 * ---------------------------------------------------------------------
 * When used with Watcom 386 flat model extensions, the winsock.h header
 * must be #included with the following wrapper:
 *   #include <windows.h> // use default shimming/redefs
 *   #ifdef __WINDOWS_386__
 *     #undef FAR
 *     #define FAR
 *   #endif
 *   #include <winsock.h> // use the 16 bit version
 *   #ifdef __WINDOWS_386__
 *     #undef FAR
 *     #define FAR far
 *     #undef FD_ISSET // macro applies a FAR cast that results in truncation
 *     #define FD_ISSET(fd, set) __WSAFDIsSet(fd, set) //...so discard the cast
 *   #endif
*/
#ifndef __W32SOCK_H__
#define __W32SOCK_H__ "@(#)$Id: w32sock.h,v 1.1.2.1 2001/01/21 15:10:26 cyp Exp $"

#if !defined(_INC_WINDOWS)
  /* windows.h must be included first! (don't default LEAN_AND_MEAN, 
  ** STRICT etc). But do basic (non-network specific) stuff anyway.
  */
  #define ntohs(x) ((unsigned short)(((((unsigned short)(x))&0xff)<<8) | \
                                       ((((unsigned short)(x))>>8)&0xff) ))
  #define htons(x) ntohs(x)
  #define ntohl(x) ((unsigned long)( ((((unsigned long)(x))&0xff)<<24) | \
                                     ((((unsigned long)(x))>>24)&0xff) | \
                                       ((((unsigned long)(x))&0xff00)<<8) | \
                                       ((((unsigned long)(x))>>8)&0xff00) ))
  #define htonl(x) ntohl(x)
  struct timeval {
      long    tv_sec;         /* seconds */
      long    tv_usec;        /* and microseconds */
  };
#else

#ifndef _INC_WINDOWS /* older winsock.h does not include windows.h */
  #ifndef WIN32_LEAN_AND_MEAN /* don't want winsock.h included here */
    #define WIN32_LEAN_AND_MEAN
    #define __WE_DEFINED_LEAN_AND_MEAN
  #endif
  #ifdef STRICT
    #error Turn off STRICT or you will get overload errors!
    /* certainly true for __WINDOWS_386__ */
  #endif
  #include <windows.h>
  #ifdef __WE_DEFINED_LEAN_AND_MEAN
    #undef WIN32_LEAN_AND_MEAN
    #undef __WE_DEFINED_LEAN_AND_MEAN
  #endif
#endif

#ifdef __WINDOWS_386__ /* the matching undo is at the end of this file */
  #undef FAR
  #define FAR
#endif

#include <winsock.h>

#ifdef __WINDOWS_386__
  /*FD_ISSET applies a FAR * cast before calling __WSAFDIsSet, so remove it.*/
  #undef FD_ISSET /* __WSAFDIsSet((SOCKET)(fd), (fd_set FAR *)(set)) */
  #define FD_ISSET(fd, set) __WSAFDIsSet(fd, set)
#endif

#if defined(__WATCOMC__) 
  /* Watcom C with full warnings doesn't like 'do {} while(0)' constructs, 
     (it complains about 'while (0)' always being false), so we change 
     'while(0)' into something that implies the same thing.
  */
  #undef FD_CLR
  #define FD_CLR(fd, set) /* do */ { \
    u_int __i; \
    for (__i = 0; __i < ((fd_set FAR *)(set))->fd_count ; __i++) { \
        if (((fd_set FAR *)(set))->fd_array[__i] == fd) { \
            while (__i < ((fd_set FAR *)(set))->fd_count-1) { \
                ((fd_set FAR *)(set))->fd_array[__i] = \
                    ((fd_set FAR *)(set))->fd_array[__i+1]; \
                __i++; \
            } \
            ((fd_set FAR *)(set))->fd_count--; \
            break; \
        } \
    } \
  } /* while(0) */

  #undef FD_SET
  #define FD_SET(fd, set) /* do */ { \
    if (((fd_set FAR *)(set))->fd_count < FD_SETSIZE) \
        ((fd_set FAR *)(set))->fd_array[((fd_set FAR *)(set))->fd_count++]=(fd);\
  } /* while(0) */
#endif /* __WATCOMC__ */

#ifdef __WINDOWS_386__
  #undef FAR
  #define FAR far
#endif

#endif /* _INCLUDE_BYTEORDER_ONLY */
#endif /* __W32SOCK_H__ */
