// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// ----------------------------------------------------------------------
// This file contains functions for formatting keyrate/time/summary data
// statistics obtained from clirate.cpp into strings suitable for display.
// ----------------------------------------------------------------------
// 
// $Log: clisrate.h,v $
// Revision 1.17  1999/02/21 21:44:59  cyp
// tossed all redundant byte order changing. all host<->net order conversion
// as well as scram/descram/checksumming is done at [get|put][net|disk] points
// and nowhere else.
//
// Revision 1.16  1999/01/29 19:04:48  jlawson
// fixed formatting.
//
// Revision 1.15  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.14  1998/07/07 21:55:28  cyruspatel
// Serious house cleaning - client.h has been split into client.h (Client
// class, FileEntry struct etc - but nothing that depends on anything) and
// baseincs.h (inclusion of generic, also platform-specific, header files).
// The catchall '#include "client.h"' has been removed where appropriate and
// replaced with correct dependancies. cvs Ids have been encapsulated in
// functions which are later called from cliident.cpp. Corrected other
// compile-time warnings where I caught them. Removed obsolete timer and
// display code previously def'd out with #if NEW_STATS_AND_LOGMSG_STUFF.
// Made MailMessage in the client class a static object (in client.cpp) in
// anticipation of global log functions.
//
// Revision 1.13  1998/06/29 06:57:53  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.12  1998/06/24 21:53:47  cyruspatel
// Created CliGetMessageForProblemCompletedNoSave() in clisrate.cpp. It
// is similar to its non-nosave pendant but doesn't affect cumulative
// statistics.  Modified Client::Benchmark() to use the function.
//
// Revision 1.11  1998/06/14 08:12:43  friedbait
// 'Log' keywords added to maintain automatic change history
// 
// Revision 1.10  1998/06/09 10:39:44  daa
// fix casts
// 
// Revision 1.9  1998/06/09 10:30:14  daa
// fix casts
// 
// Revision 1.8  1998/06/09 08:54:29  jlawson
// Committed Cyrus' changes: phrase "keys/s" is changed back to "k/s" in
// CliGetSummaryStringForContest. That line has exactly 3 characters to 
// spare, and >9 blocks (or >9 days, or >999Kk/s) will cause line to wrap.
//
// Revision 1.7  1998/06/08 18:11:32  remi
// Added a short line to tell [X]Emacs to use C++ mode instead of C mode
// for these files.
//
// Revision 1.6  1998/06/08 15:47:08  kbracey
// Added lots of "const"s and "static"s to reduce compiler warnings, and
// hopefully improve output code, too.
//
// Revision 1.5  1998/06/08 14:11:47  kbracey
// Changed "Kkeys" to "kkeys".
//
// Revision 1.4  1998/05/29 08:01:08  bovine
// copyright update, indents
//
// Revision 1.3  1998/05/27 18:21:29  bovine
// SGI Irix warnings and configure fixes
//
// Revision 1.2  1998/05/25 02:54:19  bovine
// fixed indents
//
// Revision 1.1  1998/05/24 14:25:49  daa
// Import 5/23/98 client tree
//
// Revision 0.0  1998/05/01 05:01:01  cyruspatel
// Created
//
// =============================================================================
// 

#ifndef _CLICSTAT_H_
#define _CLICSTAT_H_

//#include "cputypes.h" // u64
//#include "problem.h"  // Problem class
//#include "client.h"   // Fileentry struct

// ---------------------------------------------------------------------------

#ifndef _U32LimitDouble_
  #define _U32LimitDouble_  ((double)(0xFFFFFFFFul))
  #define U64TODOUBLE( hi, lo ) ((double)((((double)(hi))* \
           (((double)(_U32LimitDouble_))+((double)(1))))+((double)(lo))))
#endif

// returns keyrate as string (maxlen=26) "nnnn.nn ['k'|'M'|'G'|'T']"
// return value is a pointer to buffer.
char *CliGetKeyrateAsString( char *buffer, double rate );

// return iter/keysdone/whatever as string. 
// set contestID = -1 to have the ID ignored
const char *CliGetU64AsString( u64 *u, int /*inNetOrder*/, int contestID );

// combines CliGetKeyrateForProblem() and CliGetKeyrateAsString()
const char *CliGetKeyrateStringForProblem( Problem *prob );

// combines CliGetKeyrateForContest() and CliGetKeyrateAsString()
const char *CliGetKeyrateStringForContest( int contestid );

// "4 RC5 Blocks 12:34:56.78 - [123456789 keys/s]"
const char *CliGetSummaryStringForContest( int contestid );

// "Loaded RC5 1*2^30 block 68E0D85A:A0000000 (10.25% done)"
const char *CliGetMessageForFileentryLoaded( FileEntry *fileentry );

// Completed RC5 block 68E0D85A:A0000000 (123456789 keys)
//           123:45:67:89 - [987654321 keys/s]
const char *CliGetMessageForProblemCompleted( Problem *problem );

// same as above, but does not affect cumulative stats
const char *CliGetMessageForProblemCompletedNoSave( Problem *problem );

#endif // ifdef _CLICSTAT_H_

