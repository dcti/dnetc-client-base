/* Automatically generated header! Do not edit! */

#ifndef _INLINE_DNETCGUI_H
#define _INLINE_DNETCGUI_H

#ifndef __INLINE_MACROS_H
#include <inline/macros.h>
#endif /* !__INLINE_MACROS_H */

#ifndef DNETCGUI_BASE_NAME
#define DNETCGUI_BASE_NAME DnetcBase
#endif /* !DNETCGUI_BASE_NAME */

#define dnetcguiClose(params) \
	LP1(0x24, BOOL, dnetcguiClose, struct ClientGUIParams *, params, a0, \
	, DNETCGUI_BASE_NAME)

#define dnetcguiConsoleOut(cpu, output, overwrite) \
	LP3NR(0x30, dnetcguiConsoleOut, ULONG, cpu, d0, UBYTE *, output, a0, BOOL, overwrite, d1, \
	, DNETCGUI_BASE_NAME)

#define dnetcguiHandleMsgs(signals) \
	LP1(0x2a, ULONG, dnetcguiHandleMsgs, ULONG, signals, d0, \
	, DNETCGUI_BASE_NAME)

#define dnetcguiOpen(cpu, programname, iconify, vstring) \
	LP4(0x1e, ULONG, dnetcguiOpen, ULONG, cpu, d0, UBYTE *, programname, a0, struct WBArg *, iconify, a1, const char *, vstring, a2, \
	, DNETCGUI_BASE_NAME)

#endif /* !_INLINE_DNETCGUI_H */
