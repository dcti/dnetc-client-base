// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: guistuff.h,v $
// Revision 1.3  1998/12/28 04:01:07  silby
// Win32gui icon now changed by probfill when new blocks are loaded.
// If MacOS has an icon to change, this would be a good place to hook in as well.
//
// Revision 1.2  1998/10/04 11:37:47  remi
// Added Log and Id tags.
//
//

#if ((CLIENT_OS == OS_WIN32) && defined(WIN32GUI))

#if !defined(GUICLIENT)
#define GUICLIENT

extern void UpdatePercentBar(void);
extern void UpdateBufferBars(void);
extern void SetIcon(s32 currentcontest);
#endif

#endif

