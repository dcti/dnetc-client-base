/* Hey, Emacs, this a -*-C-*- file !
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __PACK4_H__
# define __PACK4_H__ "@(#)$Id: pack4.h,v 1.2 2003/09/12 23:19:10 mweiser Exp $"

# include "pack.h"
#endif

/* make sure we don't get confused by predefined macros and initialise
** DNETC_PACKED to do 4-byte alignment */
#undef DNETC_PACKED
#define DNETC_PACKED DNETC_ALIGNED4

#ifdef DNETC_USE_PACK
# pragma pack(4)
#endif

/* end of file */
