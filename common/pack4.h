/* Hey, Emacs, this a -*-C-*- file !
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

/* "@(#)$Id: pack4.h,v 1.1.2.1 2003/08/25 08:37:59 mweiser Exp $" */

#include "pack.h"

/* make sure we don't get confused by predefined macros and initialise
** DNETC_PACKED to do nothing */
#undef DNETC_PACKED
#define DNETC_PACKED DNETC_ALIGNED4

#ifdef DNETC_USE_PACK
# pragma pack(4)
#endif

/* end of file */
