/* Automatically generated header! Do not edit! */

#ifndef _PPCINLINE_DNETCGUI_H
#define _PPCINLINE_DNETCGUI_H

#ifndef __PPCINLINE_MACROS_H
#include <ppcinline/macros.h>
#endif /* !__PPCINLINE_MACROS_H */

#ifndef DNETCGUI_BASE_NAME
#define DNETCGUI_BASE_NAME DnetcBase
#endif /* !DNETCGUI_BASE_NAME */

#define dnetcguiClose(__p0) \
	LP1(36, BOOL , dnetcguiClose, \
		struct ClientGUIParams *, __p0, a0, \
		, DNETCGUI_BASE_NAME, 0, 0, 0, 0, 0, 0)

#define dnetcguiHandleMsgs(__p0) \
	LP1(42, ULONG , dnetcguiHandleMsgs, \
		ULONG , __p0, d0, \
		, DNETCGUI_BASE_NAME, 0, 0, 0, 0, 0, 0)

#define dnetcguiOpen(__p0, __p1, __p2, __p3) \
	LP4(30, ULONG , dnetcguiOpen, \
		ULONG , __p0, d0, \
		UBYTE *, __p1, a0, \
		struct WBArg *, __p2, a1, \
		const char *, __p3, a2, \
		, DNETCGUI_BASE_NAME, 0, 0, 0, 0, 0, 0)

#define dnetcguiConsoleOut(__p0, __p1, __p2) \
	LP3NR(48, dnetcguiConsoleOut, \
		ULONG , __p0, d0, \
		UBYTE *, __p1, a0, \
		BOOL , __p2, d1, \
		, DNETCGUI_BASE_NAME, 0, 0, 0, 0, 0, 0)

#endif /* !_PPCINLINE_DNETCGUI_H */
