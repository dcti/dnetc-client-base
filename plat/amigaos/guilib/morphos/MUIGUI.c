/*
 * Copyright distributed.net 2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: MUIGUI.c,v 1.1.2.2 2004/01/09 23:36:28 piru Exp $
 *
 * Created by Ilkka Lehtoranta <ilkleht@isoveli.org>
 *
 * ----------------------------------------------------------------------
 * MUI GUI module for MorphOS client - MUI application class code
 * ----------------------------------------------------------------------
*/

#include	<libraries/mui.h>
#include	<workbench/startup.h>

#include	<clib/alib_protos.h>
#include	<proto/exec.h>
#include	<proto/dos.h>
#include	<proto/icon.h>
#include	<proto/intuition.h>
#include	<proto/muimaster.h>

#include	"AppClass.h"
#include	"CreateGUI.h"
#include	"LibHeader.h"

#include	"Support.h"

#if 0
# define kprintf(fmt, tags...)	({ULONG _tags[] = { 0 , ## tags }; RawDoFmt(fmt, (APTR)&_tags[1], (void (*)(void)) __dputch, (APTR) *((APTR *)4));})
static const UWORD __dputch[5] = {0xCD4B, 0x4EAE, 0xFDFC, 0xCD4B, 0x4E75};
#else
# define kprintf(fmt, tags...)
#endif

/**********************************************************************
	GUI_Open
**********************************************************************/

ULONG NATDECLFUNC_5(GUI_Open, d0, ULONG, cpu, a0, UBYTE *, ProgramName, a1, struct WBArg *, IconName, a2, CONST_STRPTR, vstring, a6, struct DnetcLibrary *, LibBase)
{
	DECLARG_5(d0, ULONG, cpu, a0, UBYTE *, ProgramName, a1, struct WBArg *, IconName, a2, CONST_STRPTR, vstring, a6, struct DnetcLibrary *, LibBase)

	ULONG	sigmask;

	(void)ProgramName;
	(void)cpu;

	if (!LibBase->dobj)
	{
		if (IconName)
		{
			BPTR olddir;

			olddir = CurrentDir(IconName->wa_Lock);
			LibBase->dobj	= GetDiskObject(IconName->wa_Name);
			CurrentDir(olddir);
		}
	}

	sigmask	= 0;

	ObtainSemaphore(&LibBase->SemaphoreGUI);

	if (!LibBase->App)
	{
		Object	*app	= NewObjectA(LibBase->AppMCC->mcc_Class, NULL, NULL);

		if (app)
		{
			LibBase->App			= app;
			LibBase->OwnerTask	= FindTask(NULL);

			strcpy(LibBase->Version, vstring);

			DoMethod(app, MUIM_MyApplication_OpenMainWindow);

			sigmask	= 0xffff0000;
		}
	}

	ReleaseSemaphore(&LibBase->SemaphoreGUI);

	return sigmask;
}

/**********************************************************************
	GUI_Close
**********************************************************************/

BOOL NATDECLFUNC_2(GUI_Close, a0, struct ClientGUIParams *, params, a6, struct DnetcLibrary *, LibBase)
{
	DECLARG_2(a0, struct ClientGUIParams *, params, a6, struct DnetcLibrary *, LibBase)

	(void)params;

	ObtainSemaphore(&LibBase->SemaphoreGUI);

	if (FindTask(NULL) == LibBase->OwnerTask)
	{
		MUI_DisposeObject(LibBase->App);

		if (LibBase->dobj)
			FreeDiskObject(LibBase->dobj);

		LibBase->OwnerTask	= NULL;
		LibBase->App			= NULL;
		LibBase->dobj			= NULL;
	}

	ReleaseSemaphore(&LibBase->SemaphoreGUI);

	return TRUE;
}

/**********************************************************************
	GUI_HandleMsgs
**********************************************************************/

ULONG NATDECLFUNC_2(GUI_HandleMsgs, d0, ULONG, signals, a6, struct DnetcLibrary *, LibBase)
{
	DECLARG_2(d0, ULONG, signals, a6, struct DnetcLibrary *, LibBase)
	ULONG	cmd;

	(void)signals;

	cmd	= 0;

	ObtainSemaphore(&LibBase->SemaphoreGUI);

	if (FindTask(NULL) == LibBase->OwnerTask)
	{
		ULONG	sigmask	= 0xffff0000;
		int done = FALSE;

		LibBase->Commands	= 0;

		/* Yes, this is very ugly. */

		while (!done)
		{
			ULONG	ret	= DoMethod(LibBase->App, MUIM_Application_NewInput, (ULONG)&sigmask);

			if (ret == MUIV_Application_ReturnID_Quit)
			{
				ret	= 0;
				cmd	|= DNETC_MSG_SHUTDOWN;
				done	= TRUE;
			}

			if (ret && ret < MENUBASE)
			{
				cmd	|= ret;		/* Rexx command! (see CreateGUI.c) */
				done	= TRUE;
			}

			if (sigmask)
			{
				cmd	|= LibBase->Commands;
				done = TRUE;
			}
		}
	}

	ReleaseSemaphore(&LibBase->SemaphoreGUI);

	return cmd;
}

/**********************************************************************
	GUI_ConsoleOut

	Console to listview
**********************************************************************/

VOID NATDECLFUNC_4(GUI_ConsoleOut, d0, ULONG, cpu, a0, STRPTR, output, d1, BOOL, overwrite, a6, struct DnetcLibrary *, LibBase)
{
	DECLARG_4(d0, ULONG, cpu, a0, STRPTR, output, d1, BOOL, overwrite, a6, struct DnetcLibrary *, LibBase)

	(void)cpu;

	DoMethod(LibBase->App, MUIM_MyApplication_InsertNode, (ULONG)output, overwrite);
}
