/*
 * Copyright distributed.net 2004-2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: CreateGUI.c,v 1.7 2013/07/26 00:27:42 piru Exp $
 *
 * Created by Ilkka Lehtoranta <ilkleht@isoveli.org>
 *
 * ----------------------------------------------------------------------
 * MUI GUI module for MorphOS client - MUI application class code
 * ----------------------------------------------------------------------
*/

#include	<libraries/gadtools.h>
#include	<libraries/mui.h>

#include	<clib/alib_protos.h>
#include	<proto/exec.h>
#include	<proto/intuition.h>
#include	<proto/icon.h>
#include	<proto/muimaster.h>

#include    <mui/Aboutbox_mcc.h>

#include	"CreateGUI.h"
#include	"guilib_version.h"
#include	"AppClass.h"

//struct Library		*MUIMasterBase	= NULL;

#define	MUIA_Application_UsedClasses	0x8042e9a7	/* V20 STRPTR *	i..	*/

static const char VerString[]	= "\0$VER: " PROGRAM_NAME " " PROGRAM_VER " " PROGRAM_DATE;

static CONST_STRPTR ClassList[]	=
{
	NULL
};

static struct NewMenu Menus[]	=
{
   { NM_TITLE, "Project",	NULL, 0, 0, NULL },
   {  NM_ITEM, "MUI settings...",	"M", 0, 0, (APTR)MENU_MUISETTINGS_ID },
   {  NM_ITEM, NM_BARLABEL,	NULL, 0, 0, NULL},
   {  NM_ITEM, "Clear Window",	"Z",  0, 0, (APTR)MENU_CLEAR_ID },
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

static
ULONG arexx_clear(struct Hook *MyHook,
                  Object *app,
                  LONG *params)
{
	DoMethod(app, MUIM_MyApplication_ClearConsole, 0);

	return 0;
}

static const struct Hook arexx_clear_hook =
{
	{0, 0},
	(ULONG (*)(void)) HookEntry,
	(ULONG (*)(void)) arexx_clear,
	0
};

static struct MUI_Command commands[]	=
{
	{ "PAUSE"	, MC_TEMPLATE_ID, DNETC_MSG_PAUSE, NULL },
	{ "UNPAUSE"	, MC_TEMPLATE_ID, DNETC_MSG_UNPAUSE, NULL },
	{ "UPDATE"	, MC_TEMPLATE_ID, DNETC_MSG_FETCH | DNETC_MSG_FLUSH, NULL },
	{ "FETCH"	, MC_TEMPLATE_ID, DNETC_MSG_FETCH, NULL },
	{ "FLUSH"	, MC_TEMPLATE_ID, DNETC_MSG_FLUSH, NULL },
	{ "RESTART"	, MC_TEMPLATE_ID, DNETC_MSG_RESTART, NULL },
	{ "CLEAR"	, NULL, 0, &arexx_clear_hook },
	{ NULL, NULL, 0, NULL}
};


static inline void* memcpy(void *dest, const void *src, int n)
{
	char *d2 = (char*)dest;
	const char *s2 = (const char*)src;

	while (n--) *d2++ = *s2++;
	return dest;
}


/**********************************************************************
	CreateGUI
**********************************************************************/

Object *CreateGUI(struct IClass *cl, Object *obj, struct ObjStore *os, struct DnetcLibrary *LibBase)
{
	UBYTE about[512];
	ULONG array[] =
	{
		(ULONG) LibBase->Version,
		LibBase->Library.lib_Version,
		LibBase->Library.lib_Revision
	};

//#define SysBase LibBase->MySysBase
	RawDoFmt(
	 "\33cdistributed.net client - a product of distributed.net\n"
	 "%s\n"
	 "Copyright 1997-2011 distributed.net\n"
	 "\n"
	 "\n"
	 "MorphOS client maintained by\n"
	 "Harry Sintonen\n"
	 "<sintonen@iki.fi>\n"
	 "\n"
	 "\n"
	 "MUI GUI module (v%ld.%ld) maintained by\n"
	 "Ilkka Lehtoranta\n"
	 "<ilkleht@isoveli.org>",
	array, NULL, about);
//#undef SysBase

	return (Object*)DoSuperNew(cl, obj,
		MUIA_Application_DiskObject, (IPTR)LibBase->dobj,
		MUIA_Application_Commands, (IPTR)commands,
		MUIA_Application_Version, (IPTR)&VerString[1],
		MUIA_Application_Copyright, (IPTR)"distributed.net",
		MUIA_Application_Author, (IPTR)"Ilkka Lehtoranta",
		MUIA_Application_Base, (IPTR)"DNETC",
		MUIA_Application_UsedClasses, (IPTR)ClassList,
		MUIA_Application_Title, (IPTR)"dnetc",
		MUIA_Application_Description, (IPTR)"GUI for distributed.net client",
		MUIA_Application_Window, (IPTR)(os->wnd = MUI_NewObject(MUIC_Window,
			MUIA_Window_Title, (IPTR)"distributed.net client",
			MUIA_Window_ID, MAKE_ID('M','A','I','N'),
			MUIA_Window_Width, MUIV_Window_Width_Visible(55),
			MUIA_Window_Height, MUIV_Window_Height_Visible(45),
			MUIA_Window_Menustrip, (IPTR)MUI_MakeObject(MUIO_MenustripNM, (IPTR)&Menus, 0),
			MUIA_Window_RootObject, (IPTR)MUI_NewObject(MUIC_Group,
				MUIA_Group_Child, (IPTR)(os->lst = NewObject(LibBase->ListMCC->mcc_Class, NULL,
					MUIA_Background, MUII_ReadListBack,
					MUIA_Frame, MUIV_Frame_ReadList,
					MUIA_CycleChain, TRUE,
				TAG_END)),
			TAG_END),
		TAG_END)),
		MUIA_Application_Window, (IPTR)(os->req = MUI_NewObject(MUIC_Aboutbox,
			MUIA_Aboutbox_Credits, (IPTR)about,
		TAG_END)),
	TAG_END);
}

/**********************************************************************
	HodgePodge
**********************************************************************/

/*
VOID HodgePodge(struct Library *MyMUIMasterBase)
{
	MUIMasterBase	= MyMUIMasterBase;
}
*/
