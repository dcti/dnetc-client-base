/* Hey, Emacs, this a -*-C-*- file !
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __PACK_H__
#define __PACK_H__  "@(#)$Id: pack.h,v 1.2 2003/09/12 23:19:10 mweiser Exp $"

/* DNETC_USE_PACK will make pack[0-8].h use #pragma pack
**
** definition of DNETC_USE_PACK_POP or DNETC_USE_PACK0 will make
** pack0.h use #pragma pack(pop) or pack(0) respectively when
** switching off packing/alignment. If only DNETC_USE_PACK is defined,
** pack0.h will use #pragma pack() */

#if defined(__GNUC__) && ((__GNUC__ > 2) || \
    ((__GNUC__ == 2) && (__GNUC_MINOR__ >= 91)))
  /* newer versions of gcc use a vendor-specific attribute. */
  #define DNETC_PACKED1   __attribute__((packed))
  #define DNETC_ALIGNED2  __attribute__((aligned(2)))
  #define DNETC_ALIGNED4  __attribute__((aligned(4)))
  #define DNETC_ALIGNED8  __attribute__((aligned(8)))
  #define DNETC_ALIGNED16 __attribute__((aligned(16)))
  #define DNETC_ALIGNED32 __attribute__((aligned(32)))

#elif defined(__GNUC__)
  /* use pack() on old gcc's. */
  #define DNETC_USE_PACK 1

#elif defined(MIPSpro)
  /* don't use anything on MIPSpro. */
  #pragma warning "no packed structures!"

#elif defined(_MSC_VER) && (_MSC_VER >= 800)
  /* Visual C++ prints an infomational warning with pack changes. */
  #pragma warning(disable:4103)
  #define DNETC_USE_PACK 1

#elif defined(__xlC__)
  /* IBM xlC has pack but doesn't support pack() - instead we have to
  ** use pack(pop) or pack(unpack) */
  #define DNETC_USE_PACK 1
  #define DNETC_USE_PACK_POP 1

#else
  /* use pack() on anything we don't know. */
  #define DNETC_USE_PACK 1
#endif


/* If these weren't defined above, then just define them to
** nothing. */
#ifndef DNETC_PACKED1
# define DNETC_PACKED1
# define DNETC_ALIGNED2
# define DNETC_ALIGNED4
# define DNETC_ALIGNED8
# define DNETC_ALIGNED16
# define DNETC_ALIGNED32
#endif

#endif /* __PACK_H__ */
