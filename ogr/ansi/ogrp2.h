/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/
#ifndef __OGRP2_H__
#define __OGRP2_H__ "@(#)$Id: ogrp2.h,v 1.4 2008/03/08 20:18:31 kakace Exp $"

#include "ansi/ogr-interface.h"

/* ===================================================================== */

#define OGR_BITMAPS_LENGTH  160


/* specifies the number of ruler diffs can be represented.
** Warning: increasing this will cause all structures based
** on workunit_t in packets.h to change, possibly breaking
** network and buffer structure operations.
*/
#define STUB_MAX 10
#define MAXDEPTH 30

#ifndef __SUNPRO_CC
  #include "pack1.h"
#else
  #undef DNETC_PACKED
  #define DNETC_PACKED
#endif

struct Stub {           /* size is 24 */
  u16 marks;            /* N-mark ruler to which this stub applies */
  u16 length;           /* number of valid elements in the stub[] array */
  u16 diffs[STUB_MAX];  /* first <length> differences in ruler */
} DNETC_PACKED;

struct WorkStub {       /* size is 28 */
  struct Stub stub;     /* stub we're working on */
  u32 worklength;       /* depth of current state */
} DNETC_PACKED;

#ifndef __SUNPRO_CC
  #include "pack0.h"
#else
  #undef DNETC_PACKED
#endif


#endif /* __OGRP2_H__ */

