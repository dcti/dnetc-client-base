/* Automatically generated header! Do not edit! */

#ifndef _PPCINLINE_DNETCGUI_H
#define _PPCINLINE_DNETCGUI_H

#ifndef __PPCINLINE_MACROS_H
#include <powerup/ppcinline/macros.h>
#endif /* !__PPCINLINE_MACROS_H */

#ifndef DNETCGUI_BASE_NAME
#define DNETCGUI_BASE_NAME DnetcBase
#endif /* !DNETCGUI_BASE_NAME */

#define dnetcguiClose(params) \
	LP1(0x24, BOOL, dnetcguiClose, struct ClientGUIParams *, params, a0, \
	, DNETCGUI_BASE_NAME, IF_CACHEFLUSHALL, NULL, 0, IF_CACHEFLUSHALL, NULL, 0)

#define dnetcguiConsoleOut(cpu, output, overwrite) \
	LP3NR(0x30, dnetcguiConsoleOut, ULONG, cpu, d0, UBYTE *, output, a0, BOOL, overwrite, d1, \
	, DNETCGUI_BASE_NAME, IF_CACHEFLUSHALL, NULL, 0, IF_CACHEFLUSHALL, NULL, 0)

#define dnetcguiHandleMsgs(signals) \
	LP1(0x2a, ULONG, dnetcguiHandleMsgs, ULONG, signals, d0, \
	, DNETCGUI_BASE_NAME, IF_CACHEFLUSHALL, NULL, 0, IF_CACHEFLUSHALL, NULL, 0)

#define dnetcguiOpen(cpu, programname, iconify, vstring) \
	LP4(0x1e, ULONG, dnetcguiOpen, ULONG, cpu, d0, UBYTE *, programname, a0, struct WBArg *, iconify, a1, const char *, vstring, a2, \
	, DNETCGUI_BASE_NAME, IF_CACHEFLUSHALL, NULL, 0, IF_CACHEFLUSHALL, NULL, 0)

#endif /* !_PPCINLINE_DNETCGUI_H */
