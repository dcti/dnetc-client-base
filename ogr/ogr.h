// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: ogr.h,v $
// Revision 1.2  1999/04/04 15:01:27  cyp
// Included "client2.h" from here. Um, can it get another name, please?
//
// Revision 1.1  1999/03/20 21:58:05  gregh
// Rename from stub.h.
//
// Revision 1.1  1999/03/20 17:54:35  gregh
// Rename from stub.h.
//
//

#ifndef __OGR_H__
#define __OGR_H__

#ifndef u16
#include "cputypes.h"
#endif
#include "client2.h"

#define STUB_MAX 10 /* change ogr_packet_t in packets.h when changing this */

struct Stub { /* size is 24 */
  u16 marks;
  u16 length;
  u16 stub[STUB_MAX];
};

#endif
