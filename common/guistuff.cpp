// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// This file contains functions that act as hooks for the various GUIs
// that have been implemented.  Ideally, a gui can be implemented
// without touching the common tree, as all hooks will already be
// common, and all that will be needed are #ifdefs in here.

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
if (MainWindow->IsWindowVisible())
   ::InvalidateRect(MainWindow->GetSafeWindow(), NULL, TRUE);
#endif
}

void UpdateBufferBars(void)
{
#if (CLIENT_OS == OS_WIN32) && defined(NEEDVIRTUALMETHODS)
// Hook to cause the win32gui progress bar to update.
// Actually updates the full progress window, more specific hook may be
// added in the future
if (MainWindow->IsWindowVisible())
   ::InvalidateRect(MainWindow->GetSafeWindow(), NULL, TRUE);
#endif
}

