/*
 * Copyright distributed.net 2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: CreateGUI.c,v 1.1.2.4 2004/01/10 15:00:36 piru Exp $
 *
 * Created by Ilkka Lehtoranta <ilkleht@isoveli.org>
 *
 * ----------------------------------------------------------------------
 * MUI GUI module for MorphOS client - MUI application class code
 * ----------------------------------------------------------------------
*/

#undef	USE_INLINE_STDARG

#include	<libraries/gadtools.h>
#include	<libraries/mui.h>
#include	<mui/NFloattext_mcc.h>

#include	<clib/alib_protos.h>
#include	<proto/icon.h>
#include	<proto/muimaster.h>

#include	"CreateGUI.h"
#include	"guilib_version.h"

struct Library		*MUIMasterBase	= NULL;

#define	MUIA_Application_UsedClasses	0x8042e9a7	/* V20 STRPTR *	i..	*/

static const char VerString[]	= "\0$VER: dnetcgui.library " PROGRAM_VER " (9.1.2004)";

static CONST_STRPTR ClassList[]	=
{
	"NListviews.mcc",
	NULL
};

static struct NewMenu Menus[]	=
{
   { NM_TITLE, "Project",	NULL, 0, 0, NULL },
	{	NM_ITEM,	"MUI settings...", "M", 0, 0, (APTR)MENU_MUISETTINGS_ID },
   {  NM_ITEM, NM_BARLABEL,	NULL, 0, 0, NULL},
   {  NM_ITEM, "About...",		"A",  0, 0, (APTR)MENU_ABOUT_ID },
   {  NM_ITEM, NM_BARLABEL,	NULL, 0, 0, NULL},
   {  NM_ITEM, "Quit",		"Q",  0, 0, (APTR)MENU_QUIT_ID  },
   { NM_TITLE, "Client",    NULL, 0, 0, NULL },
   {  NM_ITEM, "Pause",		NULL, CHECKIT | MENUTOGGLE, 0, (APTR)MENU_PPCPAUSE_ID },
   {  NM_ITEM, NM_BARLABEL,	NULL, 0, 0, NULL},
   {  NM_ITEM, "Restart",  	NULL, 0, 0, (APTR)MENU_PPCRESTART_ID },
   {  NM_ITEM, "Benchmark",  	NULL, 0, 0, (APTR)MENU_PPCBENCHMARK_ID },
   {  NM_ITEM, "Benchmark All",	NULL, 0, 0, (APTR)MENU_PPCBENCHMARKALL_ID },
   {  NM_ITEM, "Test",		NULL, 0, 0, (APTR)MENU_PPCTEST_ID },
   {  NM_ITEM, "Configure",	NULL, 0, 0, (APTR)MENU_PPCCONFIG_ID },
   {  NM_ITEM, NM_BARLABEL,	NULL, 0, 0, NULL},
   {  NM_ITEM, "Fetch",		NULL, 0, 0, (APTR)MENU_PPCFETCH_ID },
   {  NM_ITEM, "Flush",		NULL, 0, 0, (APTR)MENU_PPCFLUSH_ID  },
   {  NM_ITEM, "Update",	NULL, 0, 0, (APTR)MENU_PPCUPDATE_ID },
   { NM_END }
};

static struct MUI_Command commands[]	=
{
	{ "PAUSE"	, MC_TEMPLATE_ID, DNETC_MSG_PAUSE, NULL },
	{ "UNPAUSE"	, MC_TEMPLATE_ID, DNETC_MSG_UNPAUSE, NULL },
	{ "UPDATE"	, MC_TEMPLATE_ID, DNETC_MSG_FETCH | DNETC_MSG_FLUSH, NULL },
	{ "FETCH"	, MC_TEMPLATE_ID, DNETC_MSG_FETCH, NULL },
	{ "FLUSH"	, MC_TEMPLATE_ID, DNETC_MSG_FLUSH, NULL },
	{ "RESTART"	, MC_TEMPLATE_ID, DNETC_MSG_RESTART, NULL }
};

static const UBYTE about[]	= "\33cdistributed.net client - a product of distributed.net\n\nCopyright 1997-2004 distributed.net\n\n\nMorphOS client maintained by\nHarry Sintonen\n<sintonen@iki.fi>\n\n\nMUI GUI module (v1.0) maintained by\nIlkka Lehtoranta\n<ilkleht@isoveli.org>";

/**********************************************************************
	CreateGUI
**********************************************************************/

Object *CreateGUI(struct IClass *cl, Object *obj, struct ObjStore *os, struct DnetcLibrary *LibBase)
{
	return DoSuperNew(cl, obj,
		MUIA_Application_DiskObject	, LibBase->dobj,
		MUIA_Application_Commands		, commands,
		MUIA_Application_Version		, &VerString[1],
		MUIA_Application_Copyright		, "distributed.net",
		MUIA_Application_Author			, "Ilkka Lehtoranta",
		MUIA_Application_Base			, "DNETC",
		MUIA_Application_UsedClasses	, ClassList,
		MUIA_Application_Title			, "dnetc",
		MUIA_Application_Description	, "GUI for distributed.net client",

		SubWindow, os->wnd	= WindowObject,
			MUIA_Window_Title		, "distributed.net client",
			MUIA_Window_ID			, MAKE_ID('M','A','I','N'),
			MUIA_Window_Menustrip, MUI_MakeObject(MUIO_MenustripNM, &Menus, 0),

			WindowContents, VGroup,
				Child, NListviewObject,
					//MUIA_Background, MUII_ReadListBack,
					//MUIA_Frame, MUIV_Frame_ReadList,
					MUIA_Listview_List, os->lst	= NFloattextObject,
						MUIA_CycleChain, TRUE,
						MUIA_ContextMenu, MUIV_NList_ContextMenu_Never,
						MUIA_NList_ConstructHook, MUIV_NList_ConstructHook_String,
						MUIA_NList_DestructHook, MUIV_NList_DestructHook_String,
						End,
					End,
				End,
			End,
		SubWindow, os->req	= WindowObject,
			MUIA_Window_Title	, "About...",
			WindowContents, VGroup,
				MUIA_Background, MUII_RequesterBack,
				Child, TextObject, TextFrame, MUIA_Background, MUII_TextBack, MUIA_Text_Contents, about, End,
				Child, HGroup,
					Child, RectangleObject, End,
					Child, os->req_ok	= MUI_MakeObject(MUIO_Button, "_Moo!"),
					Child, RectangleObject, End,
					End,
				End,
			End,
		End;
}

/**********************************************************************
	HodgePodge
**********************************************************************/

VOID HodgePodge(struct Library *MyMUIMasterBase)
{
	MUIMasterBase	= MyMUIMasterBase;
}
