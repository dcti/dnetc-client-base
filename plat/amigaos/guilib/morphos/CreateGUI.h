/*
 * Copyright distributed.net 2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: CreateGUI.h,v 1.1.2.1 2004/01/09 22:43:27 piru Exp $
 *
 * Created by Ilkka Lehtoranta <ilkleht@isoveli.org>
 *
 * ----------------------------------------------------------------------
 * MUI GUI module for MorphOS client - MUI application class code
 * ----------------------------------------------------------------------
*/

#ifndef	DNETC_CREATEGUI_H
#define	DNETC_CREATEGUI_H

#include "../dnetcgui.h"

#ifndef	DNETC_LIBHEADER_H
#include	"LibHeader.h"
#endif

#ifndef MAKE_ID
#define MAKE_ID(a,b,c,d) ((ULONG) (a)<<24 | (ULONG) (b)<<16 | (ULONG) (c)<<8 | (ULONG) (d))
#endif

#define	MENUBASE	0x8000

enum { MENU_IGNORE_ID = MENUBASE, MENU_MUISETTINGS_ID, MENU_ABOUT_ID, MENU_QUIT_ID,
       MENU_PPCPAUSE_ID, MENU_PPCRESTART_ID, MENU_PPCBENCHMARK_ID, MENU_PPCBENCHMARKALL_ID,
       MENU_PPCTEST_ID, MENU_PPCCONFIG_ID, MENU_PPCFETCH_ID, MENU_PPCFLUSH_ID, MENU_PPCUPDATE_ID, MENU_PPCSHUTDOWN_ID };

#define DNETC_MSG_RESTART			0x001
#define DNETC_MSG_SHUTDOWN			0x002
#define DNETC_MSG_PAUSE				0x004
#define DNETC_MSG_UNPAUSE			0x008

struct ObjStore
{
	Object	*wnd, *req, *lst, *req_ok;
};

Object	*CreateGUI	(struct IClass *cl, Object *obj, struct ObjStore *os, struct DnetcLibrary *LibBase);
VOID		HodgePodge	(struct Library *MyMUIMasterBase);

#endif	/* DNETC_CREATEGUI_H */
