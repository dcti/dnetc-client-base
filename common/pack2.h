/* Hey, Emacs, this a -*-C-*- file !
 *
 * Copyright distributed.net 1997-2011 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __PACK2_H__
# define __PACK2_H__ "@(#)$Id: pack2.h,v 1.5 2011/03/31 05:07:29 jlawson Exp $"

# include "pack.h"
#endif

/* make sure we don't get confused by predefined macros and initialise
** DNETC_PACKED to do 2-byte alignment */
#undef DNETC_PACKED
#define DNETC_PACKED DNETC_ALIGNED2

#ifdef DNETC_USE_PACK
# pragma pack(2)
#endif

/* end of file */
