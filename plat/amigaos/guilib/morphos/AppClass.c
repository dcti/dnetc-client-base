/*
 * Copyright distributed.net 2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: AppClass.c,v 1.1.2.1 2004/01/09 22:43:27 piru Exp $
 *
 * Created by Ilkka Lehtoranta <ilkleht@isoveli.org>
 *
 * ----------------------------------------------------------------------
 * MUI GUI module for MorphOS client - MUI application class code
 * ----------------------------------------------------------------------
*/

#include	<mui/NList_mcc.h>

#include	<clib/alib_protos.h>
#include	<proto/intuition.h>
#include	<proto/muimaster.h>
#include	<proto/dos.h>
#include	"AppClass.h"
#include	"CreateGUI.h"
#include	"LibHeader.h"

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
		data->req_ok	= os.req_ok;

		DoMethod(data->req, MUIM_Notify, MUIA_Window_CloseRequest, TRUE, (ULONG)obj, 1, MUIM_MyApplication_CloseReq);
		DoMethod(data->mainwnd, MUIM_Notify, MUIA_Window_CloseRequest, TRUE, (ULONG)obj, 2, MUIM_Application_ReturnID, MUIV_Application_ReturnID_Quit);
		DoMethod(data->mainwnd, MUIM_Notify, MUIA_Window_MenuAction, MUIV_EveryTime, (ULONG)obj, 2, MUIM_MyApplication_GetMenuItem, MUIV_TriggerValue);
		DoMethod(data->req_ok, MUIM_Notify, MUIA_Pressed, FALSE, (ULONG)obj, 1, MUIM_MyApplication_CloseReq);
	}

	return (ULONG)obj;
}

/**********************************************************************
	mOpenMainWindow
**********************************************************************/

static ULONG mOpenMainWindow(struct Application_Data *data, struct DnetcLibrary *LibBase)
{
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
	mNativeCall
**********************************************************************/

static ULONG mNativeCall(struct MUIP_MyApplication_NativeCall *msg)
{
	return msg->func(&msg->param1);
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
	set(data->list, MUIA_NList_Quiet, MUIV_NList_Quiet_Full);

	if (msg->overwrite)
		DoMethod(data->list, MUIM_NList_Remove, MUIV_NList_Remove_Last);

	DoMethod(data->list, MUIM_NList_InsertSingle, (ULONG)msg->output, MUIV_NList_Insert_Bottom);
	DoMethod(data->list, MUIM_NList_Jump, MUIV_NList_Jump_Bottom);

	return set(data->list, MUIA_NList_Quiet, MUIV_NList_Quiet_None);
}

DISPATCHERPROTO(MyApp_Dispatcher)
{
	DISPATCHERARG

	struct Application_Data	*data	= (struct Application_Data *)INST_DATA(cl, obj);
	struct DnetcLibrary	*LibBase	= (struct DnetcLibrary *)cl->cl_UserData;

	switch (msg->MethodID)
	{
		case OM_NEW										: return mNew				(cl, obj, LibBase);
		case MUIM_MyApplication_NativeCall		: return mNativeCall		((APTR)msg);
		case MUIM_MyApplication_OpenMainWindow	: return mOpenMainWindow(data, LibBase);
		case MUIM_MyApplication_InsertNode		: return mInsertNode		(data, LibBase, (APTR)msg);
		case MUIM_MyApplication_GetMenuItem		: return mGetMenuItem	(data, LibBase, (APTR)msg, obj);
		case MUIM_MyApplication_CloseReq			: return mCloseReq		(data, LibBase);
	}
	return DoSuperMethodA(cl, obj, msg);
}
