/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/
#ifndef __CONVCSC_H__
#define __CONVCSC_H__ "@(#)$Id: convcsc.h,v 1.2 1999/07/29 23:16:22 trevorh Exp $"

#ifdef __WATCOMC__
extern "C" {
#endif
extern const int csc_bit_order[64];

// convert to/from two different key formats
extern void convert_key_from_csc_to_inc (u32 *deshi, u32 *deslo);
extern void convert_key_from_inc_to_csc (u32 *deshi, u32 *deslo);
#ifdef __WATCOMC__
}
#endif
#endif /* __CONVCSC_H__ */
