// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: cliident.h,v $
// Revision 1.3  1999/01/29 19:17:51  jlawson
// fixed formatting.
//
// Revision 1.2  1999/01/01 02:45:14  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.1  1998/07/07 21:55:21  cyruspatel
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
//

#ifndef __CLIIDENT_H__
#define __CLIIDENT_H__

void CliIdentifyModules(void);

#endif //__CLIIDENT_H__
