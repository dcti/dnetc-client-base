/* -*-C-*-
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * DO NOT USE C++ style comments ('//') in this file! 
 *
 * ---------------------------------------------------------------------
 * Version bump guide: 
 * "2.7106.435"
 *  | | |   |
 *  | | |   `--- Significant changes, eg serious bugs squashed, in 
 *  | | |        ./common/ code, or simply a significant number (your call)
 *  | | |        of bug fixes, gets a "build fraction" change.
 *  | | `------- New cores, for any platform, *requires* a "build version"
 *  | |          change, ie 2.7103.x to 2.7104.x. This is needed to be able
 *  | |          to isolate bad blocks in the master keyspace bitmap.
 *  | |          New significant feature(s) also increment build version.
 *  | `--------- A "client contest" change follows a code freeze at which
 *  |            point the client is assumed to be stable. Code that
 *  |            would make clients incompatible with previous clients or
 *  |            proxies must be a accompanied by 'client contest' change.
 *  `----------- Denotes a client rewrite.
 * ---------------------------------------------------------------------
*/
#ifndef __VERSION_H__
#define __VERSION_H__ "@(#)$Id: version.h,v 1.76.2.14 2004/01/15 18:13:48 piru Exp $"

/* BETA etc is handled internally/at-runtime by cliident.cpp. */
/* Do not adjust for BETA here, particularly CLIENT_VERSIONSTRING. */

/* DO NOT USE C++ style comments ('//') in this file! */

#define CLIENT_MAJOR_VER       2
#define CLIENT_MAJOR_VER_HEX   0x02   /* needed for macos version resource */
#define CLIENT_CONTEST         90
#define CLIENT_CONTEST_HEX     0x5A   /* needed for macos version resource */
#define CLIENT_BUILD           7
#define CLIENT_BUILD_HEX       0x07   /* needed for macos version resource */
#define CLIENT_BUILD_FRAC      488
#define CLIENT_BUILD_FRAC_HEX  0x01E8 /* needed for macos version resource */
#define CLIENT_VERSIONSTRING   "2.9007-488"

/* combined version used in packets etc. ... */
#define CLIENT_VERSION         ( (((u32)(CLIENT_CONTEST))    * 1000000UL) +  \
                                 (((u32)(CLIENT_BUILD))      *   10000UL) +  \
                                 (((u32)(CLIENT_BUILD_FRAC)) *       1UL) )

/* sanity check */
#if (CLIENT_MAJOR_VER  != CLIENT_MAJOR_VER_HEX) || \
    (CLIENT_CONTEST    != CLIENT_CONTEST_HEX) || \
    (CLIENT_BUILD      != CLIENT_BUILD_HEX) || \
    (CLIENT_BUILD_FRAC != CLIENT_BUILD_FRAC_HEX)
#error inconsistency between dec and hex version number parts
#endif

#endif /* __VERSION_H__ */

