/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: dnetcgui_protos.h,v 1.1.2.1 2002/04/11 11:46:08 oliver Exp $
 *
 * Created by Oliver Roberts <oliver@futaura.co.uk>
 *
 * ----------------------------------------------------------------------
 * ReAction GUI module for AmigaOS clients - Library function prototypes
 * ----------------------------------------------------------------------
*/

#ifndef CLIB_DNETCGUI_PROTOS_H
#define CLIB_DNETCGUI_PROTOS_H

#include "dnetcgui.h"

ULONG dnetcguiOpen(ULONG cpu, UBYTE *programname, struct WBArg *iconname, const char *vstring);
BOOL  dnetcguiClose(struct ClientGUIParams *params);
ULONG dnetcguiHandleMsgs(ULONG signals);
VOID  dnetcguiConsoleOut(ULONG cpu, UBYTE *output, BOOL overwrite);

#endif
