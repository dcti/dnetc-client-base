// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Log: convdes.h,v $
// Revision 1.7  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.6  1998/07/07 21:55:35  cyruspatel
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
// Revision 1.5  1998/06/14 08:12:49  friedbait
// 'Log' keywords added to maintain automatic change history
//
// 

#ifndef CONVDES_H
#define CONVDES_H

// odd_parity[n] = (n & 0xFE) | b
// b set so that odd_parity[n] has an odd number of bits
extern const u8 odd_parity[256];

// convert to/from two different key formats
extern void convert_key_from_des_to_inc (u32 *deshi, u32 *deslo);
extern void convert_key_from_inc_to_des (u32 *deshi, u32 *deslo);

#endif

