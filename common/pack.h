/* Hey, Emacs, this a -*-C-*- file !
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __PACK_H__
#define __PACK_H__  "@(#)$Id: pack.h,v 1.1.2.2 2003/08/25 16:41:24 mweiser Exp $"

#if (!defined(__GNUC__) || (__GNUC__ < 2) || \
     ((__GNUC__ == 2) && (__GNUC_MINOR__ < 91)))
# if defined(MIPSpro)
   /* don't use anything on MIPSpro */
#  pragma warning "no packed structures!"
# else
   /* use pack() on anything we don't know and old gcc's */
#  define DNETC_USE_PACK 1
# endif

# define DNETC_PACKED1
# define DNETC_ALIGNED2
# define DNETC_ALIGNED4
# define DNETC_ALIGNED8
# define DNETC_ALIGNED16
# define DNETC_ALIGNED32
#else
# define DNETC_PACKED1   __attribute__((packed))
# define DNETC_ALIGNED2  __attribute__((aligned(2)))
# define DNETC_ALIGNED4  __attribute__((aligned(4)))
# define DNETC_ALIGNED8  __attribute__((aligned(8)))
# define DNETC_ALIGNED16 __attribute__((aligned(16)))
# define DNETC_ALIGNED32 __attribute__((aligned(32)))
#endif

#endif /* __PACK_H__ */
