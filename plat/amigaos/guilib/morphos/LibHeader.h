/*
 * Copyright distributed.net 2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: LibHeader.h,v 1.1.2.1 2004/01/09 22:43:27 piru Exp $
 *
 * Created by Ilkka Lehtoranta <ilkleht@isoveli.org>
 *
 * ----------------------------------------------------------------------
 * MUI GUI module for MorphOS client - MUI application class code
 * ----------------------------------------------------------------------
*/

#ifndef	DNETC_LIBHEADER_H
#define	DNETC_LIBHEADER_H

#ifndef	DOS_DOS_H
#include <dos/dos.h>
#endif

#ifndef	EXEC_LIBRARIES_H
#include <exec/libraries.h>
#endif

#ifndef	EXEC_SEMAPHORES_H
#include <exec/semaphores.h>
#endif

#ifndef	LIBRARIES_MUI_H
#include <libraries/mui.h>
#endif

#ifndef	__DECLGATE_H__
#include "declgate.h"
#endif

#ifdef __MORPHOS__
# define MREG(reg,arg) arg
#else
# define	RTF_PPC		0
# define	RTF_EXTENDED	0
#endif

#ifndef	__GNUC__
# if !defined(__attribute__)
#  define __attribute__(x)
# endif
#endif

#ifdef __PPC__
# define	MASM
#endif

/**********************************************************************
	Structures
**********************************************************************/

#ifdef __GNUC__
# pragma pack(2)
#endif

struct MyInitData
{
	UBYTE ln_Type_Init[4];
	UBYTE ln_Pri_Init[4];
	UBYTE ln_Name_Init[2];
	ULONG ln_Name_Content;
	UBYTE lib_Flags_Init[4];
	UBYTE lib_Version_Init[2]; UWORD lib_Version_Content;
	UBYTE lib_Revision_Init[2]; UWORD lib_Revision_Content;
	UBYTE lib_IdString_Init[2];
	ULONG lib_IdString_Content;
	UWORD EndMark;
};

#ifdef __GNUC__
# pragma pack()
#endif

struct DnetcLibrary
{
	struct Library		Library;
	UBYTE			Alloc;
	UBYTE			Pad;
	BPTR			SegList;

	struct ExecBase		*MySysBase;
	struct Library		*MyMUIMasterBase;
	struct IntuitionBase	*MyIntuitionBase;
	struct Library		*MyUtilityBase;
	struct DosLibrary	*MyDOSBase;
	struct Library		*MyIconBase;

	struct SignalSemaphore	SemaphoreGUI;
	struct SignalSemaphore	Semaphore;
	struct MUI_CustomClass	*AppMCC;

	/* GUI data */

	struct Task		*OwnerTask;
	Object			*App;
	ULONG			Commands;
	UBYTE			Version[60];
	APTR			dobj;

#ifdef	__MORPHOS__
	struct EmulLibEntry	TrapAppMCC;
#endif
};


#ifdef	USE_INLINE_STDARG
# define SysBase		LibBase->MySysBase
# define MUIMasterBase		LibBase->MyMUIMasterBase
#endif

#define IntuitionBase		LibBase->MyIntuitionBase
#define UtilityBase		LibBase->MyUtilityBase
#define DOSBase			LibBase->MyDOSBase
#define IconBase		LibBase->MyIconBase

#ifdef __MORPHOS__
# define PROTOTYPE(name) void name ## _DispatcherPPC(void); static struct EmulLibEntry name ## _Dispatcher = {TRAP_LIB, 0, (void (*)(void)) name ## _DispatcherPPC };
# define MAKEHOOK(name, func, trap)	const struct EmulHook name = { NULL, NULL, (HOOKFUNC)& ## (name) ## .emul, NULL, NULL, TRAP_ ## trap , 0, (void *)(func) };
# define HOOKPROTO(name, ret, obj, param) static ret name (void)
# define HOOKPROTONH(name, ret, obj, param) static ret name (void)
# define HOOKPROTONHNO(name, ret, param) static ret name (void)
# define INSTALLHOOK(hook, func) hook ## .realhook.h_Entry = (HOOKFUNC)& ## hook ## .emul; hook ## .emul.Trap = TRAP_LIB; hook ## .emul.Extension = 0; hook ## .emul.Func = (void *)(func);
# define INSTALLHOOKNR(hook, func) hook ## .realhook.h_Entry = (HOOKFUNC)& ## hook ## .emul; hook ## .emul.Trap = TRAP_LIBNR; hook ## .emul.Extension = 0; hook ## .emul.Func = (void *)(func);
# define DISPATCHERPROTO(name) ULONG name ## PPC(void)
# define DISPATCHERARG DECLARG_3(a0, struct IClass *, cl, a2, Object *, obj, a1, Msg, msg)
# define HOOKPROTO_ARG(a,b,c,d) DECLARG_3(a0, struct Hook *, hook, a2, a, b, a1, c, d)
# define HOOKPROTONH_ARG(a,b,c,d) DECLARG_2(a2, a, b, a1, c, d)
# define HOOKPROTONHNO_ARG(a,b) DECLARG_1(a1, a, b)
#else
# define PROTOTYPE(name) void name ## _Dispatcher(void);
# define HOOKPROTO_ARG(a,b,c,d)
# define HOOPROTONH_ARG(a,b,c,d)
# define HOOKPROTONHNO_ARG(a,b)
# define HOOKPROTO(name, ret, obj, param) static SAVEDS ASM(ret) name(REG(a0, struct Hook *hook), REG(a2, obj), REG(a1, param))
# define HOOKPROTONH(name, ret, obj, param) static SAVEDS ASM(ret) name(REG(a2, obj), REG(a1, param))
# define HOOKPROTONHNO(name, ret, param) static SAVEDS ASM(ret) name(REG(a1, param))
# define DISPATCHERPROTO(name) static ASM(ULONG) SAVEDS name(REG(a0, struct IClass * cl), REG(a2, Object * obj), REG(a1, Msg msg))
# define DISPATCHERARG
#endif

/**********************************************************************
	Prototypes
**********************************************************************/

ULONG			LibReserved(void);
MASM struct Library	*LibInit(MREG(d0, struct DnetcLibrary *LibBase), MREG(a0, BPTR SegList), MREG(a6, struct ExecBase *MySysBase));
BPTR			NATDECLFUNC_1(LibExpunge, a6, struct DnetcLibrary *, LibBase);
BPTR			NATDECLFUNC_1(LibClose, a6, struct DnetcLibrary *, LibBase);
struct Library	*	NATDECLFUNC_1(LibOpen, a6, struct DnetcLibrary *, LibBase);

ULONG	NATDECLFUNC_4(GUI_Open, d0, ULONG, cpu, a0, UBYTE *, ProgramName, a1, struct WBArg *, IconName, a2, CONST_STRPTR, vstring);
BOOL	NATDECLFUNC_2(GUI_Close, a0, struct ClientGUIParams *, params, a6, struct DnetcLibrary *, LibBase);
ULONG	NATDECLFUNC_1(GUI_HandleMsgs, d0, ULONG, signals);
VOID	NATDECLFUNC_3(GUI_ConsoleOut, d0, ULONG, cpu, a0, STRPTR, output, d1, BOOL, Overwrite);

#endif	/* DNETC_LIBHEADER_H */
