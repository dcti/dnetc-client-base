// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Log: version.h,v $
// Revision 1.11  1998/08/20 02:40:41  silby
// Kicked version to 2.7100.418-BETA1, ensured that clients report the string ver (which has beta1 in it) in the startup.
//
// Revision 1.10  1998/07/22 04:28:49  jlawson
// updated version to 417
//
// Revision 1.9  1998/07/12 09:09:24  silby
// updates to 416
//
// Revision 1.8  1998/07/12 08:05:12  silby
// Updated to 416, updated changelog
//
// Revision 1.7  1998/07/07 21:55:55  cyruspatel
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
// Revision 1.6  1998/07/07 03:10:22  silby
// Updated to build 414
//
// Revision 1.5  1998/06/29 17:05:43  daa
// bump to 413.
//
// Revision 1.4  1998/06/28 20:52:11  jlawson
// added version string without leading "v" character
//
// Revision 1.3  1998/06/26 15:48:08  daa
// Its Here....V2.7100.412
//
// Revision 1.2  1998/06/25 14:07:18  daa
// add DCTI copyright notice and cvs log header
// bump version to 7026.411
//
//

#ifndef _VERSION_H
#define _VERSION_H

#define CLIENT_CONTEST      71
#define CLIENT_BUILD        00
#define CLIENT_BUILD_FRAC   418

#define CLIENT_VERSIONSTRING    "v2.7100.418-BETA1"
#define CLIENT_VERSIONSTRING2   "2.7100.418-BETA1"        // no leading "v"

#endif // _VERSION_H

