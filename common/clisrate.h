/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * This file contains functions for formatting keyrate/time/summary data
 * statistics obtained from clirate.cpp into strings suitable for display.
 * ----------------------------------------------------------------------
*/ 
#ifndef __CLISRATE_H__
#define __CLISRATE_H__ "@(#)$Id: clisrate.h,v 1.20.2.6 2000/10/27 17:58:42 cyp Exp $"

/* Log("Summary: 4 RC5 Blocks 12:34:56.78 - [123456789 keys/s]");  */
int CliPostSummaryStringForContest( int contestid );

/* returns keyrate as string (maxlen=26) "nnnn.nn ['k'|'M'|'G'|'T']" */
/* return value is a pointer to buffer.                           */
const char *CliGetKeyrateAsString( char *buffer, double rate );

#endif /* __CLISRATE_H__ */
