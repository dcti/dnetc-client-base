/*
 * Copyright distributed.net 2004-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: LibHeader.c,v 1.1.2.4 2008/10/27 16:13:32 piru Exp $
 *
 * Created by Ilkka Lehtoranta <ilkleht@isoveli.org>
 *
 * ----------------------------------------------------------------------
 * MUI GUI module for MorphOS client - MUI application class code
 * ----------------------------------------------------------------------
*/

#include	<exec/initializers.h>
#include	<exec/nodes.h>
#include	<exec/resident.h>

#include	<proto/exec.h>

#include	"LibHeader.h"
#include	"guilib_version.h"

static const struct MyInitData	InitData;
static const ULONG					InitTable[];
static const char 					LibId[];
static const char 					LibName[];

LONG ReturnError(VOID)
{
	return -1;
}

const struct Resident RomTag	=
{
	RTC_MATCHWORD,
	(struct Resident *)&RomTag,
	(struct Resident *)&RomTag+1,
	RTF_AUTOINIT | RTF_PPC | RTF_EXTENDED,
	COMPILE_VERSION,
	NT_LIBRARY,
	0,
	(char *)&LibName[0],
	(char *)&LibId[0],
	(APTR)&InitTable[0]

#ifdef	__MORPHOS__
	, COMPILE_REVISION, NULL
#endif
};

static const APTR FuncTable[] =
{
#ifdef __MORPHOS__
	(APTR)	FUNCARRAY_32BIT_NATIVE, 
#endif

	(APTR)	LibOpen,
	(APTR)	LibClose,
	(APTR)	LibExpunge,
	(APTR)	LibReserved,

	(APTR)	GUI_Open,
	(APTR)	GUI_Close,
	(APTR)	GUI_HandleMsgs,
	(APTR)	GUI_ConsoleOut,

	(APTR)	-1
};

static const ULONG InitTable[] =
{
	sizeof(struct DnetcLibrary),
	(ULONG)	FuncTable,
	(ULONG)	&InitData,
	(ULONG)	LibInit
};

static const struct MyInitData InitData	=
{
	0xa0,8,		NT_LIBRARY,0,
	0xa0,9,		0xfb,0,					/* 0xfb -> priority -5 */
	0x80,10,	(ULONG)&LibName[0],
	0xa0,14,	LIBF_SUMUSED|LIBF_CHANGED,0,
	0x90,20,	COMPILE_VERSION,
	0x90,22,	COMPILE_REVISION,
	0x80,24,	(ULONG)&LibId[0],
	0
};

#define LIBNAME PROGRAM_NAME

static const char LibId[]	= LIBNAME " " PROGRAM_VER " " PROGRAM_DATE
                                  " Copyright © 2004-2008 distributed.net. All rights reserved. Written by Ilkka Lehtoranta.";
static const char LibName[]	= LIBNAME;

/**********************************************************************
	Globals
**********************************************************************/

#ifdef	__MORPHOS__
const ULONG __abox__	= 1;
#endif
