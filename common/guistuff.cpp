// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// This file contains functions that act as hooks for the various GUIs
// that have been implemented.  Ideally, a gui can be implemented
// without touching the common tree, as all hooks will already be
// common, and all that will be needed are #ifdefs in here.
//
// $Log: guistuff.cpp,v $
// Revision 1.7  1998/12/28 22:39:58  silby
// SetIcon just moved over to platforms/win32gui/guicons.cpp
//
// Revision 1.6  1998/12/28 04:09:04  silby
// Urk.  Fixed some foolishness with out of order defines.
//
// Revision 1.5  1998/12/28 04:04:00  silby
// Functions not defined if GUICLIENT is !defined.
//
// Revision 1.4  1998/12/28 04:01:07  silby
// Win32gui icon now changed by probfill when new blocks are loaded.
// If MacOS has an icon to change, this would be a good place to hook in as well.
//
// Revision 1.3  1998/10/04 11:37:45  remi
// Added Log and Id tags.
//

// -----------------------------------------------------------------------

#if (!defined(lint) && defined(__showids__))
const char *guistuff_cpp(void) {
return "@(#)$Id: guistuff.cpp,v 1.7 1998/12/28 22:39:58 silby Exp $"; }
#endif

#include "guistuff.h"

#if defined (GUICLIENT)

#if (CLIENT_OS == OS_WIN32) && defined(WIN32GUI)
#include "vwindow.hpp"
#include "..\platforms\win32gui\guiwin.h"
#endif


void UpdatePercentBar(void)
{
#if (CLIENT_OS == OS_WIN32) && defined(WIN32GUI)
// Hook to cause the win32gui progress bar to update.
// Actually updates the full progress window, more specific hook may be
// added in the future
  MainWindow->progresswin.RefreshPercentBar();
#endif
}

void UpdateBufferBars(void)
{
#if (CLIENT_OS == OS_WIN32) && defined(WIN32GUI)
// Hook to cause the win32gui progress bar to update.
// Actually updates the full progress window, more specific hook may be
// added in the future
  MainWindow->progresswin.RefreshBufferBars();
#endif
}


#endif
