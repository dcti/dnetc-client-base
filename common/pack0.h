/* Hey, Emacs, this a -*-C-*- file !
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __PACK0_H__
# define __PACK0_H__ "@(#)$Id: pack0.h,v 1.1.2.3 2003/09/12 13:20:34 mweiser Exp $"
#endif

#if !defined(__PACK_H__)
# error "you must include pack{,1,4,8}.h first!"
#endif

#if defined(DNETC_USE_PACK_POP)
# pragma pack(pop)
#elif defined(DNETC_USE_PACK0)
# pragma pack(0)
#elif defined(DNETC_USE_PACK)
# pragma pack()
#endif

/* clean up behind us */
#undef DNETC_PACKED

/* end of file */

