/* Automatically generated header! Do not Edit! */

struct ClientGUIParams;
struct WBArg;



#ifndef CLIB_DNETCGUI_PROTOS_H
#define CLIB_DNETCGUI_PROTOS_H

#include "dnetcgui.h"

ULONG dnetcguiOpen(ULONG cpu, UBYTE *programname, struct WBArg *iconname, const char *vstring);
BOOL  dnetcguiClose(struct ClientGUIParams *params);
ULONG dnetcguiHandleMsgs(ULONG signals);
VOID  dnetcguiConsoleOut(ULONG cpu, UBYTE *output, BOOL overwrite);

#endif
