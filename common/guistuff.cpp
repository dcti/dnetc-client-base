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
// Revision 1.3  1998/10/04 11:37:45  remi
// Added Log and Id tags.
//

// -----------------------------------------------------------------------

#if (!defined(lint) && defined(__showids__))
const char *guistuff_cpp(void) {
return "@(#)$Id: guistuff.cpp,v 1.3 1998/10/04 11:37:45 remi Exp $"; }
#endif

#include "guistuff.h"
#if (CLIENT_OS == OS_WIN32) && defined(NEEDVIRTUALMETHODS)
#include "vwindow.hpp"
#include "..\platforms\win32gui\guiwin.h"
#endif


void UpdatePercentBar(void)
{
#if (CLIENT_OS == OS_WIN32) && defined(NEEDVIRTUALMETHODS)
// Hook to cause the win32gui progress bar to update.
// Actually updates the full progress window, more specific hook may be
// added in the future
  MainWindow->progresswin.RefreshPercentBar();
#endif
}

void UpdateBufferBars(void)
{
#if (CLIENT_OS == OS_WIN32) && defined(NEEDVIRTUALMETHODS)
// Hook to cause the win32gui progress bar to update.
// Actually updates the full progress window, more specific hook may be
// added in the future
  MainWindow->progresswin.RefreshBufferBars();
#endif
}

