// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
/*
   Version bump guide: 
   "2.7106.435"
    | | |   |
    | | |   `--- Significant changes, eg serious bugs squashed, in 
    | | |        ./common/ code, or simply a significant number (your call)
    | | |        of bug fixes, gets a "build fraction" change.
    | | `------- New cores, for any platform, *requires* a "build version"
    | |          change, ie 2.7103.x to 2.7104.x. This is needed to be able
    | |          to isolate bad blocks in the master keyspace bitmap.
    | `--------- A "client contest" change follows a code freeze at which
    |            point the client is assumed to be stable. Code that
    |            would make clients incompatible with previous clients or
    |            proxies must be a accompanied by 'client contest' change.
    `----------- Denotes a client rewrite.
*/
// $Log: version-conflict.h,v $
// Revision 1.43  1999/02/15 06:27:19  silby
// Changed version to 2.7107 due to rewrite of Problem::Run.
//
// Revision 1.42  1999/02/14 04:37:14  silby
// 2.7106.436
//
// Revision 1.41  1999/02/03 17:56:16  cyp
// 2.7106.435 for new alpha core; discarded/merged VERSIONSTRING (was with "v",
// now without) and VERSIONSTRING2 (was without "v", now gone); added doc.
//
// Revision 1.40  1999/01/26 17:29:04  michmarc
// .434
//
// Revision 1.39  1999/01/21 22:33:49  cyp
// .433
//
// Revision 1.38  1999/01/17 12:57:57  remi
// .432
//
// Revision 1.37  1999/01/15 05:15:54  cyp
// .431
//
// Revision 1.36  1999/01/13 19:48:18  cyp
// blah. 2.7105.430
//
// Revision 1.35  1999/01/12 09:13:40  silby
// Moved to 2.7104.428 in honor of new des mmx core.
//
// Revision 1.34  1999/01/09 01:40:52  silby
// Welcome to .427
//
// Revision 1.33  1999/01/06 01:20:01  cyp
// .426  I herewith remind everyone that a/b/c style revs are on a
// platform basis.
//
// Revision 1.32  1999/01/02 06:12:13  silby
// Er, now we're at 2.7103.425
//
// Revision 1.31  1999/01/02 06:11:01  silby
// 2.7103.425, happy new year.
//
// Revision 1.30  1999/01/01 02:45:16  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.29  1998/12/28 23:12:43  cyp
// .424 and counting...
//
// Revision 1.28  1998/12/25 09:25:22  cyp
// voila! .423
//
// Revision 1.27  1998/12/20 17:53:02  cyp
// .422: other things that go bump in the night. :)
//
// Revision 1.26  1998/12/01 00:34:17  cyp
// Bump to 421.
//
// Revision 1.25  1998/11/13 15:52:13  silby
// Bumping to 420.
//
// Revision 1.24  1998/11/08 21:11:17  cyp
// 2.7102.419 actually...
//
// Revision 1.23  1998/11/08 19:18:23  cyp
// Let'errip. We are now at .419.
//
// Revision 1.22  1998/10/29 15:00:47  chrisb
// bumped expiry date by another couple of weeks
//
// Revision 1.21  1998/10/26 03:39:29  cyp
// Enough changes to warrant a bump to beta 3. No date change.
//
// Revision 1.20  1998/10/19 07:43:51  chrisb
// Bumped BETA_EXPIRATION_TIME by another 2 weeks. Now buys the farm 
// at ~10:00 GMT on November 2nd.
//
// Revision 1.19  1998/10/10 18:26:19  silby
// Updated version information in preparation of beta2 launch, added 
// comments about versioning.
//
// Revision 1.18  1998/10/05 07:22:15  chrisb
// Added 2 weeks to "BETA_EXPIRATION_TIME" since it expired this weekend.
//
// Revision 1.17  1998/10/04 03:23:39  silby
// Bumped the beta timeout ahead a few days.
//
// Revision 1.16  1998/09/28 12:46:26  cyp
// removed checkifbetaexpired prototype
//
// Revision 1.15  1998/09/23 22:25:43  silby
// There, now it's int. All better. :)
//
// Revision 1.14  1998/09/23 22:20:19  blast
// Blargh Silby.
// Changed my #if 0 to //
//
// Revision 1.13  1998/09/23 22:17:33  blast
// Added #if 0 around the whole BETA client thing
//
// Revision 1.12  1998/09/19 08:50:22  silby
// Added in beta test client timeouts.  Enabled/controlled from version.h 
// by defining BETA, and setting the expiration time.
//
// Revision 1.11  1998/08/20 02:40:41  silby
// Kicked version to 2.7100.418-BETA1, ensured that clients report the 
// string ver (which has beta1 in it) in the startup.
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
// client.h has been split into client.h and baseincs.h 
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

#define CLIENT_MAJOR_VER     2
#define CLIENT_CONTEST      71
#define CLIENT_BUILD        07
#define CLIENT_BUILD_FRAC   437

#define CLIENT_VERSIONSTRING    "2.7107.437"

// When releasing a beta client, please set the expiration time to
// about two weeks into the future; that should be an adequate beta
// time period.

#if 0
#define BETA
#define BETA_EXPIRATION_TIME    911433600  /* Nov 19 00:00:00 GMT */
#endif


#endif // _VERSION_H

