/*
 * Copyright distributed.net 1997-2016 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: libbase.c,v 1.3 2007/10/22 16:48:30 jlawson Exp $
 *
 * Created by Oliver Roberts <oliver@futaura.co.uk>
 *
 * ----------------------------------------------------------------------
 * ReAction GUI module for AmigaOS clients - Library base / startup code
 * ----------------------------------------------------------------------
*/

#include "common.h"
#include "dnetcgui_rev.h"

LONG _start(VOID)
{
   return(-1);
}

const char ExLibName[] = "dnetcgui.library";
#define ExLibID (VSTRING "\0Copyright © 2001-2016 Oliver Roberts. All rights reserved.")

struct ExecBase *SysBase       = NULL;
struct DosLibrary  *DOSBase       = NULL;
struct IntuitionBase *IntuitionBase  = NULL;
ALIAS(__UtilityBase,UtilityBase);
#ifdef __amigaos4__
struct UtilityBase *UtilityBase    = NULL;
#else
struct Library *UtilityBase    = NULL;
#endif
struct GfxBase *GfxBase        = NULL;

#ifdef __amigaos4__
struct ExecIFace *IExec = NULL;
struct DOSIFace *IDOS = NULL;
struct IntuitionIFace *IIntuition = NULL;
struct UtilityIFace *IUtility = NULL;
struct GraphicsIFace *IGraphics = NULL;

struct Library *OpenLibraryIFace( STRPTR name, ULONG version, APTR *iface )
{
   struct Library *lib;

   if( (lib = OpenLibrary( name, version ) ) ) {
      if( ! ( *iface = (APTR) GetInterface( lib, "main", 1L, NULL ) ) ) {
         CloseLibrary( lib );
         lib = NULL;
      }
   }

   return lib;
}

VOID CloseLibraryIFace( struct Library *lib, APTR iface )
{
   if( lib ) {
      if( iface ) {
         DropInterface( (struct Interface *) iface );
      }
      CloseLibrary( lib );
   }
}

STATIC BPTR LibClose2(struct Interface *self, BOOL dosem);
#else
STATIC BPTR LibClose2(struct LibBase *lb, BOOL dosem);
#endif

STATIC SAVEDS ASM struct LibBase *LibInit(
#ifdef __amigaos4__
	struct LibBase *lb, APTR seglist, struct ExecIFace *pIExec)
{
   IExec = pIExec;
   pIExec->Obtain();
#else
        REG(a6,struct Library *sysbase), REG(a0,APTR seglist), REG(d0,struct LibBase *lb))
{
   lb->lb_SysBase = sysbase;
   SysBase = (struct ExecBase *)sysbase;
#endif
   lb->lb_LibNode.lib_Revision = REVISION;
   lb->lb_SegList = (BPTR)seglist;

   InitSemaphore(&lb->lb_BaseLock);
   InitSemaphore(&lb->lb_GUILock);

   return(lb);
}

STATIC SAVEDS ASM struct LibBase *LibOpen(
#ifdef __amigaos4__
   struct Interface *self, ULONG version)
{
   struct LibBase *lb = (struct LibBase *)self->Data.LibBase;
#else
   REG(a6,struct LibBase *lb))
{
#endif
   BOOL libfail = FALSE;

   ObtainSemaphore(&lb->lb_BaseLock);

   lb->lb_LibNode.lib_OpenCnt++;
   lb->lb_LibNode.lib_Flags &= ~LIBF_DELEXP;

   if (lb->lb_LibNode.lib_OpenCnt == 1) {
      libfail = TRUE;

      lb->lb_DOSBase = OpenLibraryIFace((UBYTE *)"dos.library",36,(APTR *)&IDOS);
      lb->lb_GfxBase = OpenLibraryIFace((UBYTE *)"graphics.library",36,(APTR *)&IGraphics);
      lb->lb_IntuitionBase = OpenLibraryIFace((UBYTE *)"intuition.library",36,(APTR *)&IIntuition);
      lb->lb_UtilityBase = OpenLibraryIFace((UBYTE *)"utility.library",36,(APTR *)&IUtility);

      if (lb->lb_DOSBase && lb->lb_GfxBase && lb->lb_IntuitionBase &&
          lb->lb_UtilityBase)
      {
         DOSBase = (struct DosLibrary *)lb->lb_DOSBase;
         GfxBase = (struct GfxBase *)lb->lb_GfxBase;
         IntuitionBase = (struct IntuitionBase *)lb->lb_IntuitionBase;
         UtilityBase = (struct UtilityBase *)lb->lb_UtilityBase;
         libfail = FALSE;
      }

      #ifdef __amigaos4__
      if (libfail) LibClose2(self,FALSE);
      #else
      if (libfail) LibClose2(lb,FALSE);
      #endif
   }

   ReleaseSemaphore(&lb->lb_BaseLock);

   if (libfail) lb = NULL;

   return(lb);
}

STATIC SAVEDS ASM BPTR LibExpunge(
#ifdef __amigaos4__
   struct Interface *self)
{
   struct LibBase *lb = (struct LibBase *)self->Data.LibBase;
#else
   REG(a6,struct LibBase *lb))
{
#endif
   BPTR seglist = (BPTR)NULL;

   if (lb->lb_LibNode.lib_OpenCnt == 0) {
      seglist = (BPTR)lb->lb_SegList;
      Remove((struct Node *)lb);
      #ifdef __amigaos4__
      DeleteLibrary((struct Library *)lb);
      IExec->Release();
      #else
      FreeMem((BYTE *)lb - lb->lb_LibNode.lib_NegSize,
              lb->lb_LibNode.lib_NegSize + lb->lb_LibNode.lib_PosSize);
      #endif
   }
   else {
      lb->lb_LibNode.lib_Flags |= LIBF_DELEXP;
   }

   return(seglist);
}

STATIC SAVEDS ASM BPTR LibClose(
#ifdef __amigaos4__
   struct Interface *self)
{
   return(LibClose2(self,TRUE));
}
#else
   REG(a6,struct LibBase *lb))
{
   return(LibClose2(lb,TRUE));
}
#endif

#ifdef __amigaos4__
STATIC BPTR LibClose2(struct Interface *self, BOOL dosem)
{
   struct LibBase *lb = (struct LibBase *)self->Data.LibBase;
#else
STATIC BPTR LibClose2(struct LibBase *lb, BOOL dosem)
{
#endif
   BPTR seglist = (BPTR)NULL;

   if (dosem) ObtainSemaphore(&lb->lb_BaseLock);

   if (--lb->lb_LibNode.lib_OpenCnt == 0) {
      CloseLibraryIFace(lb->lb_UtilityBase,IUtility);
      CloseLibraryIFace(lb->lb_IntuitionBase,IIntuition);
      CloseLibraryIFace(lb->lb_GfxBase,IGraphics);
      CloseLibraryIFace(lb->lb_DOSBase,IDOS);
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
#ifdef __amigaos4__
      seglist = LibExpunge(self);
#else
      seglist = LibExpunge(lb);
#endif
   }

   return(seglist);
}

/****************************************************************************/

#ifdef __amigaos4__

STATIC ULONG LibObtain(struct Interface *self)
{
   return(self->Data.RefCount++);
}

STATIC ULONG LibRelease(struct Interface *self)
{
   return(self->Data.RefCount--);
}

STATIC CONST APTR LibManagerVectors[] =
{
   (APTR)LibObtain,
   (APTR)LibRelease,
   NULL,
   NULL,
   (APTR)LibOpen,
   (APTR)LibClose,
   (APTR)LibExpunge,
   NULL,
   (APTR)-1
};

STATIC CONST struct TagItem LibManagerTags[] =
{
   {MIT_Name,             (ULONG)"__library"},
   {MIT_VectorTable,      (ULONG)LibManagerVectors},
   {MIT_Version,          1},
   {TAG_DONE,             0}
};

STATIC CONST APTR MainVectors[] =
{
   (APTR)LibObtain,
   (APTR)LibRelease,
   NULL,
   NULL,
   (APTR)dnetcguiOpen,
   (APTR)dnetcguiClose,
   (APTR)dnetcguiHandleMsgs,
   (APTR)dnetcguiConsoleOut,
   (APTR)-1
};

STATIC CONST struct TagItem MainTags[] =
{
   {MIT_Name,              (ULONG)"main"},
   {MIT_VectorTable,       (ULONG)MainVectors},
   {MIT_Version,           1},
   {TAG_DONE,              0}
};

STATIC CONST ULONG LibInterfaces[] =
{
   (ULONG)LibManagerTags,
   (ULONG)MainTags,
   0
};

extern APTR VecTable68K[]; /* dnetcgui_68k.s */

STATIC CONST struct TagItem LibCreateTags[] =
{
   {CLT_DataSize,   (ULONG)(sizeof(struct LibBase))},
   {CLT_Interfaces, (ULONG)LibInterfaces},
   {CLT_Vector68K,  (ULONG)&VecTable68K},
   {CLT_InitFunc,   (ULONG)LibInit},
   {TAG_DONE,       0}
};

STATIC CONST struct Resident RomTag __attribute__((used)) = {
   RTC_MATCHWORD,
   (struct Resident *)&RomTag,
   (struct Resident *)&RomTag+1,
   RTF_AUTOINIT | RTF_NATIVE,
   VERSION,
   NT_LIBRARY,
   0,
   (char *)ExLibName,
   (char *)ExLibID,
   (APTR)LibCreateTags
};

#else

ULONG LibExtFunc(VOID)
{
   return(0);
}

static const APTR FuncTab[] = {
   (APTR)LibOpen,
   (APTR)LibClose,
   (APTR)LibExpunge,
   (APTR)LibExtFunc,

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
   (APTR)LibInit
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

#endif
