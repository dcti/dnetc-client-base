/*
 * Copyright distributed.net 2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: AppClass.h,v 1.1.2.2 2004/01/14 01:21:19 piru Exp $
 *
 * Created by Ilkka Lehtoranta <ilkleht@isoveli.org>
 *
 * ----------------------------------------------------------------------
 * MUI GUI module for MorphOS client - MUI application class code
 * ----------------------------------------------------------------------
*/

#ifndef	DNETC_APPCLASS_H
#define	DNETC_APPCLASS_H

#ifndef DNETC_LIBHEADER_H
#include	"LibHeader.h"
#endif

struct Application_Data
{
	Object	*mainwnd, *list;
	Object	*req, *req_ok;
	ULONG	Paused;
};

#define	TAG_BASE	0xfece0000

#define	MUIM_MyApplication_NativeCall			(TAG_BASE + 0xf001)
#define	MUIM_MyApplication_OpenMainWindow	(TAG_BASE + 0xf002)
#define	MUIM_MyApplication_InsertNode			(TAG_BASE + 0xf003)
#define	MUIM_MyApplication_GetMenuItem		(TAG_BASE + 0xf004)
#define	MUIM_MyApplication_CloseReq			(TAG_BASE + 0xf005)
#define MUIM_MyApplication_UnIconified		(TAG_BASE + 0xf006)

struct MUIP_MyApplication_NativeCall	{ ULONG MethodID; ULONG (*func)(ULONG *); ULONG param1; };
struct MUIP_MyApplication_InsertNode	{ ULONG MethodID; CONST_STRPTR output; ULONG overwrite; };
struct MUIP_MyApplication_GetMenuItem	{ ULONG MethodID; ULONG menu; };

DISPATCHERPROTO(MyApp_Dispatcher);

#endif	/* DNETC_APPCLASS_H */
