/*
 * Copyright distributed.net 2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: LibInit.c,v 1.1.2.1 2004/01/09 22:43:27 piru Exp $
 *
 * Created by Ilkka Lehtoranta <ilkleht@isoveli.org>
 *
 * ----------------------------------------------------------------------
 * MUI GUI module for MorphOS client - MUI application class code
 * ----------------------------------------------------------------------
*/

#include	<libraries/mui.h>

#include	<clib/alib_protos.h>
#include	<proto/dos.h>
#include	<proto/exec.h>
#include	<proto/muimaster.h>

#include	"AppClass.h"
#include	"CreateGUI.h"
#include	"LibHeader.h"

#ifdef	__MORPHOS__
#define	AppDispatcher	LibBase->TrapAppMCC
#else
#define	AppDispatcher	MyApp_Dispatcher
#endif

/*********************************************************************
 * @LibReserved																		*
 ********************************************************************/

ULONG LibReserved(void)
{
	return 0;
}

/**********************************************************************
	LibInit
**********************************************************************/

MASM struct Library	*LibInit(MREG(d0, struct DnetcLibrary *LibBase), MREG(a0, BPTR SegList), MREG(a6, struct ExecBase *MySysBase))
{
	LibBase->SegList	= SegList;
	LibBase->MySysBase	= MySysBase;

	InitSemaphore(&LibBase->SemaphoreGUI);
	InitSemaphore(&LibBase->Semaphore);

#ifdef	__MORPHOS__
	LibBase->TrapAppMCC.Trap	= TRAP_LIB;
	LibBase->TrapAppMCC.Func	= (APTR)&MyApp_DispatcherPPC;
#endif

	return (struct Library *)LibBase;
}

/*********************************************************************
 * @DeleteLib																			*
 ********************************************************************/

static BPTR DeleteLib(struct DnetcLibrary *LibBase)
{
	BPTR	SegList	= 0;

	if (LibBase->Library.lib_OpenCnt == 0)
	{
		SegList	= LibBase->SegList;

		Remove(&LibBase->Library.lib_Node);
		FreeMem((APTR)((ULONG)(LibBase) - (ULONG)(LibBase->Library.lib_NegSize)), LibBase->Library.lib_NegSize + LibBase->Library.lib_PosSize);
	}

	return SegList;
}

/*********************************************************************
 * @UserLibClose																		*
 ********************************************************************/

static void UserLibClose(struct DnetcLibrary *LibBase)
{
	if (MUIMasterBase)
	{
		if (LibBase->AppMCC)
		{
			MUI_DeleteCustomClass(LibBase->AppMCC);
		}

		CloseLibrary(MUIMasterBase);

		LibBase->MyMUIMasterBase	= NULL;
		LibBase->AppMCC			= NULL;
	}

	CloseLibrary((struct Library *)IntuitionBase);
	CloseLibrary(UtilityBase);
	CloseLibrary((struct Library *)DOSBase);
	CloseLibrary(IconBase);

	IntuitionBase		= NULL;
	UtilityBase		= NULL;
	DOSBase			= NULL;
	IconBase		= NULL;
}

/*********************************************************************
 * @LibExpunge																			*
 ********************************************************************/

BPTR NATDECLFUNC_1(LibExpunge, a6, struct DnetcLibrary *, LibBase)
{
	DECLARG_1(a6, struct DnetcLibrary *, LibBase)

	LibBase->Library.lib_Flags	|= LIBF_DELEXP;

	return DeleteLib(LibBase);
}

/*********************************************************************
 * @LibClose																			*
 ********************************************************************/

BPTR NATDECLFUNC_1(LibClose, a6, struct DnetcLibrary *, LibBase)
{
	DECLARG_1(a6, struct DnetcLibrary *, LibBase)

	BPTR	SegList	= 0;

	ObtainSemaphore(&LibBase->Semaphore);

	LibBase->Library.lib_OpenCnt--;

	if (LibBase->Library.lib_OpenCnt == 0)
	{
		LibBase->Alloc	= 0;
		UserLibClose(LibBase);
	}

	ReleaseSemaphore(&LibBase->Semaphore);

	if (LibBase->Library.lib_Flags & LIBF_DELEXP)
		SegList	= DeleteLib(LibBase);

	return SegList;
}

/**********************************************************************
	LibOpen
**********************************************************************/

struct Library	*NATDECLFUNC_1(LibOpen, a6, struct DnetcLibrary *, LibBase)
{
	DECLARG_1(a6, struct DnetcLibrary *, LibBase)
	struct Library	*base;

	base	= &LibBase->Library;

	LibBase->Library.lib_Flags &= ~LIBF_DELEXP;
	LibBase->Library.lib_OpenCnt++;

	ObtainSemaphore(&LibBase->Semaphore);

	if (LibBase->Alloc == 0)
	{
		if ((MUIMasterBase	= (APTR)OpenLibrary("muimaster.library", 11)) != NULL)
		if ((IntuitionBase	= (APTR)OpenLibrary("intuition.library", 36)) != NULL)
		if ((UtilityBase	= (APTR)OpenLibrary("utility.library"  , 36)) != NULL)
		if ((DOSBase		= (APTR)OpenLibrary("dos.library"      , 36)) != NULL)
		if ((IconBase		= (APTR)OpenLibrary("icon.library"     , 36)) != NULL)
		if ((LibBase->AppMCC	= MUI_CreateCustomClass(NULL, MUIC_Application, NULL, sizeof(struct Application_Data), (APTR)&AppDispatcher)) != NULL)
		{
			LibBase->AppMCC->mcc_Class->cl_UserData	= (ULONG)LibBase;

			HodgePodge(MUIMasterBase);

			LibBase->Alloc	= 1;
				goto done;
		}

		UserLibClose(LibBase);
		LibBase->Library.lib_OpenCnt--;
		base	= NULL;
	}

done:
	ReleaseSemaphore(&LibBase->Semaphore);

	return base;
}
