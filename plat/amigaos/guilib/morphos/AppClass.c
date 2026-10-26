/*
 * Copyright distributed.net 2004-2005 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: AppClass.c,v 1.4 2013/07/26 00:27:42 piru Exp $
 *
 * Created by Ilkka Lehtoranta <ilkleht@isoveli.org>
 *
 * ----------------------------------------------------------------------
 * MUI GUI module for MorphOS client - MUI application class code
 * ----------------------------------------------------------------------
*/

#include	<clib/alib_protos.h>
#include	<proto/exec.h>
#include	<proto/intuition.h>
#include	<proto/muimaster.h>
#include	<proto/dos.h>
#include	<proto/icon.h>
#include	"AppClass.h"
#include	"CreateGUI.h"
#include	"LibHeader.h"
#include	"ListClass.h"
#include	"Support.h"

/**********************************************************************
	mNew
**********************************************************************/

static ULONG mNew(struct IClass *cl, Object *obj, struct DnetcLibrary *LibBase)
{
	struct ObjStore	os;

	if (obj = CreateGUI(cl, obj, &os, LibBase))
	{
		struct Application_Data	*data	= (struct Application_Data *)INST_DATA(cl, obj);

		data->mainwnd	= os.wnd;
		data->list		= os.lst;
		data->req		= os.req;

		DoMethod(data->mainwnd, MUIM_Notify, MUIA_Window_CloseRequest, TRUE, (ULONG)obj, 2, MUIM_Application_ReturnID, MUIV_Application_ReturnID_Quit);
		DoMethod(data->mainwnd, MUIM_Notify, MUIA_Window_MenuAction, MUIV_EveryTime, (ULONG)obj, 2, MUIM_MyApplication_GetMenuItem, MUIV_TriggerValue);
		DoMethod(obj, MUIM_Notify, MUIA_Application_Iconified, FALSE, (ULONG)obj, 1, MUIM_MyApplication_UnIconified);
	}

	return (ULONG)obj;
}

/**********************************************************************
	mOpenMainWindow
**********************************************************************/

static ULONG mOpenMainWindow(struct Application_Data *data, struct DnetcLibrary *LibBase, Object *obj)
{
	if (LibBase->dobj)
	{
		if (FindToolType(((struct DiskObject *)LibBase->dobj)->do_ToolTypes, "HIDE"))
		{
			set(obj, MUIA_Application_Iconified, TRUE);
		}
	}

	return set(data->mainwnd, MUIA_Window_Open, TRUE);
}

/**********************************************************************
	mCloseReq
**********************************************************************/

static ULONG mCloseReq(struct Application_Data *data, struct DnetcLibrary *LibBase)
{
	return set(data->req, MUIA_Window_Open, FALSE);
}

/**********************************************************************
	mUnIconified
**********************************************************************/

static ULONG mUnIconified(struct Application_Data *data)
{
	return DoMethod(data->list, MUIM_List_Jump, MUIV_List_Jump_Bottom);
}

/**********************************************************************
	mClerConsole
**********************************************************************/

static ULONG mClearConsole(struct Application_Data *data)
{
	return DoMethod(data->list, MUIM_List_Clear);
}

/**********************************************************************
	mGetMenuItem
**********************************************************************/

static ULONG mGetMenuItem(struct Application_Data *data, struct DnetcLibrary *LibBase, struct MUIP_MyApplication_GetMenuItem *msg, Object *obj)
{
	ULONG	cmd	= 0;

	switch (msg->menu)
	{
		case MENU_MUISETTINGS_ID	: DoMethod(obj, MUIM_Application_OpenConfigWindow, 0); break;
		case MENU_CLEAR_ID			: DoMethod(obj, MUIM_MyApplication_ClearConsole); break;
		case MENU_ABOUT_ID	: set(data->req, MUIA_Window_Open, TRUE); break;

		case MENU_PPCPAUSE_ID	:
		{
			if (data->Paused)
			{
				cmd	= DNETC_MSG_UNPAUSE;
				data->Paused	= 0;
			}
			else
			{
				cmd	= DNETC_MSG_PAUSE;
				data->Paused	= 1;
			}
		}
		break;

		case MENU_QUIT_ID					: cmd	= DNETC_MSG_SHUTDOWN; break;
		case MENU_PPCRESTART_ID			: cmd	= DNETC_MSG_RESTART; break;
		case MENU_PPCBENCHMARK_ID		: cmd	= DNETC_MSG_BENCHMARK; break;
		case MENU_PPCBENCHMARKALL_ID	: cmd	= DNETC_MSG_BENCHMARK_ALL; break;
		case MENU_PPCTEST_ID				: cmd	= DNETC_MSG_TEST; break;
		case MENU_PPCCONFIG_ID			: cmd	= DNETC_MSG_CONFIG; break;
		case MENU_PPCFETCH_ID			: cmd	= DNETC_MSG_FETCH; break;
		case MENU_PPCFLUSH_ID			: cmd	= DNETC_MSG_FLUSH; break;
		case MENU_PPCUPDATE_ID			: cmd	= DNETC_MSG_FETCH | DNETC_MSG_FLUSH; break;
	}

	LibBase->Commands	|= cmd;

	return 0;
}

/**********************************************************************
	mInsertNode
**********************************************************************/

static ULONG mInsertNode(struct Application_Data *data, struct DnetcLibrary *LibBase, struct MUIP_MyApplication_InsertNode *msg)
{
	LONG first_index, visible_lines, total_lines;

	GetAttr(MUIA_List_First, data->list, &first_index);
	GetAttr(MUIA_List_Visible, data->list, &visible_lines);
	GetAttr(MUIA_List_Entries, data->list, &total_lines);

	if (msg->overwrite)
	{
		ULONG entries;
		struct LogListEntry *e;
		STRPTR line_copy;

		GetAttr(MUIA_List_Entries, data->list, &entries);
		DoMethod(data->list, MUIM_List_GetEntry, entries - 1, (IPTR)&e);

		if (e && (line_copy = AllocVecTaskPooled(strlen(msg->output) + 1)))
		{
			strcpy(line_copy, msg->output);
			if (e->LogListLine) FreeVecTaskPooled(e->LogListLine);
			e->LogListLine = line_copy;
			DoMethod(data->list, MUIM_List_Redraw, entries - 1, (IPTR)e);
		}
	}
	else
	{
		if (total_lines >= 1000) DoMethod(data->list, MUIM_List_Remove, MUIV_List_Remove_First);
		DoMethod(data->list, MUIM_List_InsertSingle, (ULONG)msg->output, MUIV_List_Insert_Bottom);
		DoMethod(data->list, MUIM_List_Jump, MUIV_List_Jump_Bottom);
	}

	return TRUE;
}

DISPATCHERPROTO(MyApp_Dispatcher)
{
	DISPATCHERARG

	struct Application_Data	*data	= (struct Application_Data *)INST_DATA(cl, obj);
	struct DnetcLibrary	*LibBase	= (struct DnetcLibrary *)cl->cl_UserData;

	switch (msg->MethodID)
	{
		case OM_NEW										: return mNew				(cl, obj, LibBase);
		case MUIM_MyApplication_OpenMainWindow	: return mOpenMainWindow(data, LibBase, obj);
		case MUIM_MyApplication_InsertNode		: return mInsertNode		(data, LibBase, (APTR)msg);
		case MUIM_MyApplication_GetMenuItem		: return mGetMenuItem	(data, LibBase, (APTR)msg, obj);
		case MUIM_MyApplication_CloseReq			: return mCloseReq		(data, LibBase);
		case MUIM_MyApplication_UnIconified		: return mUnIconified		(data);
		case MUIM_MyApplication_ClearConsole		: return mClearConsole		(data);
	}
	return DoSuperMethodA(cl, obj, msg);
}
