/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: libbase.c,v 1.1.2.1 2002/04/11 11:44:56 oliver Exp $
 *
 * Created by Oliver Roberts <oliver@futaura.co.uk>
 *
 * ----------------------------------------------------------------------
 * ReAction GUI module for AmigaOS clients - Library base / startup code
 * ----------------------------------------------------------------------
*/

#include "common.h"
#include "dnetcgui_rev.h"

LONG ReturnError(VOID)
{
   return(-1);
}

__aligned const char ExLibName[] = "dnetcgui.library";
#define ExLibID (VSTRING "\0Copyright © 2001-2002 Oliver Roberts. All rights reserved.")

struct ExecBase *SysBase       = NULL;
struct DosLibrary  *DOSBase       = NULL;
struct IntuitionBase *IntuitionBase  = NULL;
ALIAS(__UtilityBase,UtilityBase);
struct Library *UtilityBase    = NULL;
struct GfxBase *GfxBase        = NULL;

BPTR CloseLib2(struct LibBase *lb, BOOL dosem);

SAVEDS ASM struct LibBase *InitLib(REG(a6,struct Library *sysbase), REG(a0,APTR seglist), REG(d0,struct LibBase *lb))
{
   lb->lb_LibNode.lib_Revision = REVISION;
   lb->lb_SegList = (BPTR)seglist;
   lb->lb_SysBase = sysbase;
   SysBase = (struct ExecBase *)sysbase;

   InitSemaphore(&lb->lb_BaseLock);
   InitSemaphore(&lb->lb_GUILock);

   return(lb);
}

SAVEDS ASM struct LibBase *OpenLib(REG(a6,struct LibBase *lb))
{
   BOOL libfail = FALSE;

   ObtainSemaphore(&lb->lb_BaseLock);

   lb->lb_LibNode.lib_OpenCnt++;
   lb->lb_LibNode.lib_Flags &= ~LIBF_DELEXP;

   if (lb->lb_LibNode.lib_OpenCnt == 1) {
      libfail = TRUE;

      lb->lb_DOSBase = OpenLibrary((UBYTE *)"dos.library",36);
      lb->lb_GfxBase = OpenLibrary((UBYTE *)"graphics.library",36);
      lb->lb_IntuitionBase = OpenLibrary((UBYTE *)"intuition.library",36);
      lb->lb_UtilityBase = OpenLibrary((UBYTE *)"utility.library",36);

      if (lb->lb_DOSBase && lb->lb_GfxBase && lb->lb_IntuitionBase &&
          lb->lb_UtilityBase)
      {
         DOSBase = (struct DosLibrary *)lb->lb_DOSBase;
         GfxBase = (struct GfxBase *)lb->lb_GfxBase;
         IntuitionBase = (struct IntuitionBase *)lb->lb_IntuitionBase;
         UtilityBase = lb->lb_UtilityBase;
         libfail = FALSE;
      }

      if (libfail) CloseLib2(lb,FALSE);
   }

   ReleaseSemaphore(&lb->lb_BaseLock);

   if (libfail) lb = NULL;

   return(lb);
}

SAVEDS ASM BPTR ExpungeLib(REG(a6,struct LibBase *lb))
{
   BPTR seglist = (BPTR)NULL;

   if (lb->lb_LibNode.lib_OpenCnt == 0) {
      seglist = (BPTR)lb->lb_SegList;
      Remove((struct Node *)lb);
      FreeMem((BYTE *)lb - lb->lb_LibNode.lib_NegSize,
              lb->lb_LibNode.lib_NegSize + lb->lb_LibNode.lib_PosSize);
   }
   else {
      lb->lb_LibNode.lib_Flags |= LIBF_DELEXP;
   }

   return(seglist);
}

SAVEDS ASM BPTR CloseLib(REG(a6,struct LibBase *lb))
{
   return(CloseLib2(lb,TRUE));
}

BPTR CloseLib2(struct LibBase *lb, BOOL dosem)
{
   BPTR seglist = (BPTR)NULL;

   if (dosem) ObtainSemaphore(&lb->lb_BaseLock);

   if (--lb->lb_LibNode.lib_OpenCnt == 0) {
      CloseLibrary(lb->lb_UtilityBase);
      CloseLibrary(lb->lb_IntuitionBase);
      CloseLibrary(lb->lb_GfxBase);
      CloseLibrary(lb->lb_DOSBase);
      lb->lb_UtilityBase = NULL;
      lb->lb_IntuitionBase = NULL;
      lb->lb_GfxBase = NULL;
      lb->lb_DOSBase = NULL;
      UtilityBase = NULL;
      IntuitionBase = NULL;
      GfxBase = NULL;
      DOSBase = NULL;
   }

   if (dosem) ReleaseSemaphore(&lb->lb_BaseLock);

   if (lb->lb_LibNode.lib_Flags & LIBF_DELEXP) {
      seglist = ExpungeLib(lb);
   }

   return(seglist);
}

ULONG ExtFuncLib(VOID)
{
   return(0);
}

static const APTR FuncTab[] = {
   (APTR)OpenLib,
   (APTR)CloseLib,
   (APTR)ExpungeLib,
   (APTR)ExtFuncLib,

   (APTR)dnetcguiOpen,
   (APTR)dnetcguiClose,
   (APTR)dnetcguiHandleMsgs,
   (APTR)dnetcguiConsoleOut,

   (APTR)-1
};

struct InitTable {
   ULONG              LibBaseSize;
   APTR              *FunctionTable;
   struct MyDataInit *DataTable;
   APTR               InitLibTable;
};

static const struct InitTable InitTab = {
   sizeof(struct LibBase),
   (APTR)FuncTab,
   NULL,
   (APTR)InitLib
};

extern VOID ENDCODE(VOID);

static const struct Resident RomTag = {
   RTC_MATCHWORD,
   (struct Resident *)&RomTag,
   (APTR)&ENDCODE,
   RTF_AUTOINIT,
   VERSION,
   NT_LIBRARY,
   0,
   (char *)ExLibName,
   (char *)ExLibID,
   &InitTab
};
