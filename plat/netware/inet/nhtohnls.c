/*
 * Classification: netinet
 * Service: Internet Network Library
 * Author: Cyrus Patel <cyp@fb14.uni-mainz.de>
 * Copyright: none
 *
 * $Log: nhtohnls.c,v $
 * Revision 1.1.2.1  2001/01/21 15:10:30  cyp
 * restructure and discard of obsolete elements
 *
 * Revision 1.1.2.1  1999/11/14 20:44:22  cyp
 * all new
 *
 *
*/

#if defined(__showids__)
const char *nhtohnls_c(void) { 
return "$Id: nhtohnls.c,v 1.1.2.1 2001/01/21 15:10:30 cyp Exp $"; } 
#endif

/* #include <netinet/in.h> */
#include <limits.h>

#if !defined(USHRT_MAX) && defined(USHORT_MAX)
#define USHRT_MAX USHORT_MAX
#endif
#if !defined(UINT_MAX) || !defined(ULONG_MAX) || !defined(USHRT_MAX)
#error limits.h is broken: no UINT_MAX and/or ULONG_MAX and/or USH[O]RT_MAX
#endif
#if (ULONG_MAX < UINT_MAX) || (USHRT_MAX > UINT_MAX)
#error limits.h is broken: ULONG_MAX < UINT_MAX and/or USH[O]RT_MAX > UINT_MAX
#endif

#if defined(__U32)
#undef __U32
#endif
#if defined(__U16)
#undef __U16
#endif

#if (UINT_MAX != 0xff)
#  if (UINT_MAX == 0xffff)
#    if (USHRT_MAX == UINT_MAX)
#      define __U16 unsigned short
#    else
#      define __U16 unsigned int
#    endif
#    if (ULONG_MAX != UINT_MAX)
#      if (ULONG_MAX == 0xfffffffful)
#        define __U32 unsigned long
#      endif
#    endif
#  elif (UINT_MAX == 0xffffffff)
#    if (ULONG_MAX == UINT_MAX)
#      define __U32 unsigned long
#    else
#      define __U32 unsigned int
#    endif
#    if (USHRT_MAX == 0xffff)
#      define __U16 unsigned short
#    endif
#  elif (UINT_MAX == 0xffffffffffffffff)
#    if (USHRT_MAX == 0xffffffff)
#      define __U32 unsigned short
#      ifdef UCHAR_MAX
#        if (UCHAR_MAX == 0xffff)
#          define __U16 unsigned char
#        endif
#      endif
#    endif  
#  endif
#endif

#if !defined(__U16) || !defined(__U32)
#error Unable to determine 16bit/32bit typedefs
#else
typedef __U16 __u16;
typedef __U32 __u32;
#endif

static __u32 __bigendian_l( register __u32 l )
{ register __u32 s1=((l>>16) & 0xffff), s2 = (l & 0xffff);
  return ((((__u32)((s2>>8) | (s2<<8)))<<16)|(s1>>8 | s1<<8));
}

static __u32 __littleendian_l( register __u32 l )
{ return l; }

static __u32 __pdpendian_l( register __u32 l )
{ return (((l<<16)&0xffff0000ul) | ((l>>16)&0x0000fffful)); }

static __u16 __bigendian_s( register __u16 s )
{ return (((s<<8)&0xff00) | ((s>>8)&0x00ff)); }

static __u16 __littleendian_s( register __u16 s )
{ return s; }

static __u16 check_16( register __u16 );
static __u32 check_32( register __u32 );

static volatile __u16 (*__tos)( register __u16 ) = check_16;
static volatile __u32 (*__tol)( register __u32 ) = check_32;

unsigned int getbyteorder( void )
{
  static __u32 byte_order = 0x01020304ul;

  if (byte_order == 0x01020304ul)
  {
    register char *p = (char *)&byte_order;
  
    if (p[sizeof(byte_order)-1] == 4)
    {
      byte_order = 1234;
      __tol = __littleendian_l;
      __tos = __littleendian_s;
    }
    else if (p[sizeof(byte_order)-1] == 1)
    {
      byte_order = 4321;
      __tol = __bigendian_l;
      __tos = __bigendian_s;
    }
    else
    {
      byte_order = 3412;
      __tol = __pdpendian_l;
      __tos = __littleendian_s;
    }
  } 
  return (unsigned int)byte_order;
}

__u16 check_16( register __u16 s )
{ getbyteorder(); return (*__tos)(s); }

__u32 check_32( register __u32 l )
{ getbyteorder(); return (*__tol)(l); }

__u32 ntohl(__u32 l)
{ return (*__tol)(l); }  

__u32 htonl(__u32 l)
{ return (*__tol)(l); }  

__u16 ntohs(__u16 s)
{ return (*__tos)(s); }  

__u16 htons(__u16 s)
{ return (*__tos)(s); }  

