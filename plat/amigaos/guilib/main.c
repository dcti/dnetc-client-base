/*
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: main.c,v 1.4 2003/11/01 14:20:15 mweiser Exp $
 *
 * Created by Oliver Roberts <oliver@futaura.co.uk>
 *
 * ----------------------------------------------------------------------
 * ReAction GUI module for AmigaOS clients - main API and GUI code
 * ----------------------------------------------------------------------
*/

#include "common.h"

#include <classes/window.h>
#include <classes/arexx.h>
#include <classes/requester.h>
#include <gadgets/layout.h>
#include <gadgets/listbrowser.h>
#include <reaction/reaction_macros.h>

#include <proto/icon.h>
#include <proto/gadtools.h>
#include <proto/layout.h>
#include <proto/listbrowser.h>
#include <proto/requester.h>
#include <proto/arexx.h>
#include <proto/window.h>

#include "dnetcgui.h"
#include "prefs.h"
#include "main.h"

#define DNETC_MSG_RESTART	0x1
#define DNETC_MSG_SHUTDOWN	0x2
#define DNETC_MSG_PAUSE		0x4
#define DNETC_MSG_UNPAUSE	0x8

enum { REXX_QUIT=0, REXX_HIDE, REXX_SHOW, REXX_PAUSE, REXX_UNPAUSE,
       REXX_UPDATE, REXX_FETCH, REXX_FLUSH, REXX_RESTART, NUM_REXX };

struct ClientGUIParams GUIParams;

struct TaskSpecificData {
   struct Task *task;
   ULONG sigmask;
   BYTE idcmpsigbit;
   BYTE appsigbit;
   BYTE arexxsigbit;
};

struct {
   ULONG open;
   struct TaskSpecificData *master;
   struct TaskSpecificData *slave;
   struct TaskSpecificData client68k;
   struct TaskSpecificData clientppc;
   UBYTE version[60];
   BOOL iconify;
} GUIInfo;

struct Library *IconBase;
struct Library *GadToolsBase;
struct Library *RequesterBase;
struct Library *LayoutBase;
struct Library *ListBrowserBase;
struct Library *GetFontBase;
struct Library *ButtonBase;
struct Library *LabelBase;
struct Library *IntegerBase;
struct Library *CheckBoxBase;
struct Library *ChooserBase;
struct Library *WorkbenchBase;
struct Library *ARexxBase;
struct Library *WindowBase;

struct MsgPort *IDCMPPort, *AppPort, *ArexxPort;

struct Gadget *GlbGadgetsP[NUM_GADS];

Object *GlbWindowP, *ArexxObj;
struct Window *GlbIWindowP;
struct Menu *GlbMenu;
APTR GlbVisualInfo;
struct AppMenuItem *AppMenu;
struct Hook AppMsgHook;

ULONG WindowSigMask, ArexxSigMask;
ULONG ArexxHookCmds;

struct ConsoleLines ConsoleLines68K, ConsoleLinesPPC;

enum { MENU_IGNORE_ID, MENU_PAUSE_ID, MENU_HIDE_ID, MENU_ABOUT_ID, MENU_QUIT_ID, MENU_68KPAUSE_ID,
       MENU_68KRESTART_ID, MENU_68KBENCHMARK_ID, MENU_68KBENCHMARKALL_ID,
       MENU_68KTEST_ID, MENU_68KCONFIG_ID, MENU_68KFETCH_ID, MENU_68KFLUSH_ID, MENU_68KUPDATE_ID, MENU_68KSHUTDOWN_ID,
       MENU_PPCPAUSE_ID, MENU_PPCRESTART_ID, MENU_PPCBENCHMARK_ID, MENU_PPCBENCHMARKALL_ID,
       MENU_PPCTEST_ID, MENU_PPCCONFIG_ID, MENU_PPCFETCH_ID, MENU_PPCFLUSH_ID, MENU_PPCUPDATE_ID, MENU_PPCSHUTDOWN_ID,
       MENU_PREFS_ID, MENU_SNAPSHOT_ID };

struct NewMenu ClientMenus[] = {
   { NM_TITLE, "Project",	NULL, 0, 0, NULL },
   {  NM_ITEM, "About...",		"A",  0, 0, (APTR)MENU_ABOUT_ID },
   {  NM_ITEM, NM_BARLABEL,	NULL, 0, 0, NULL},
   {  NM_ITEM, "Pause All",	"P",  CHECKIT | MENUTOGGLE, 0, (APTR)MENU_PAUSE_ID },
   {  NM_ITEM, NM_BARLABEL,	NULL, 0, 0, NULL},
   {  NM_ITEM, "Hide",		"H",  0, 0, (APTR)MENU_HIDE_ID },
   {  NM_ITEM, NM_BARLABEL,	NULL, 0, 0, NULL},
   {  NM_ITEM, "Quit",		"Q",  0, 0, (APTR)MENU_QUIT_ID  },
   { NM_TITLE, "68K Client",    NULL, NM_MENUDISABLED, 0, NULL },
   {  NM_ITEM, "Pause",		NULL, CHECKIT | MENUTOGGLE, 0, (APTR)MENU_68KPAUSE_ID },
   {  NM_ITEM, NM_BARLABEL,	NULL, 0, 0, NULL},
   {  NM_ITEM, "Restart",  	NULL, 0, 0, (APTR)MENU_68KRESTART_ID },
   {  NM_ITEM, "Benchmark",  	NULL, 0, 0, (APTR)MENU_68KBENCHMARK_ID },
   {  NM_ITEM, "Benchmark All",	NULL, 0, 0, (APTR)MENU_68KBENCHMARKALL_ID },
   {  NM_ITEM, "Test",		NULL, 0, 0, (APTR)MENU_68KTEST_ID },
   {  NM_ITEM, "Configure",	NULL, 0, 0, (APTR)MENU_68KCONFIG_ID },
   {  NM_ITEM, NM_BARLABEL,	NULL, 0, 0, NULL},
   {  NM_ITEM, "Fetch",		NULL, 0, 0, (APTR)MENU_68KFETCH_ID },
   {  NM_ITEM, "Flush",		NULL, 0, 0, (APTR)MENU_68KFLUSH_ID  },
   {  NM_ITEM, "Update",	NULL, 0, 0, (APTR)MENU_68KUPDATE_ID },
   { NM_TITLE, "PPC Client",    NULL, NM_MENUDISABLED, 0, NULL },
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
   { NM_TITLE, "Settings",      NULL, 0, 0, NULL },
   {  NM_ITEM, "Preferences...","S", 0, 0, (APTR)MENU_PREFS_ID },
   { NM_END }
};

const char AboutText[] =
   "distributed.net client - a product of distributed.net\n"
   "%s\nCopyright 1997-2003 distributed.net\n\n"
   "\33cAmigaOS clients maintained by\n"
   "Oliver Roberts <oliver@futaura.co.uk>\n\n"
   "ClassAct/ReAction GUI module (v%ld.%ld) maintained by\n"
   "Oliver Roberts <oliver@futaura.co.uk>";

APTR NewObject( struct IClass *classPtr, CONST_STRPTR classID, ULONG tag1, ... )
{
   return NewObjectA(classPtr,(STRPTR)classID,(struct TagItem *)&tag1);
}

static void __stuffChar(void)
{
   __asm__ __volatile__ ("move.b d0,(a3)+" : : );
}

#if 0
VOID SPrintf(UBYTE *ostring, UBYTE *format, ...)
{
   va_list va;
   va_start(va,format);
   RawDoFmt(format,va,__stuffChar,ostring);
   va_end(va);
}
#endif

static struct DiskObject *ReadTooltypeOptions(struct WBArg *iconarg)
{
   struct DiskObject *icon = NULL;
   char *str;

   if (iconarg) {
      BPTR dir = CurrentDir(iconarg->wa_Lock);
      if ((icon = GetDiskObject(iconarg->wa_Name))) {
         icon->do_CurrentX = NO_ICON_POSITION;
         icon->do_CurrentY = NO_ICON_POSITION;
         GUIInfo.iconify = (FindToolType(icon->do_ToolTypes,"HIDE") != NULL);
      }
      CurrentDir(dir);
   }

   return icon;
}

VOID DisplayError(const char *error, ...)
{
   va_list va;
   Object *ReqObject;
   char *buffer;

   va_start(va,error);

   if (buffer = AllocVec(2048,MEMF_PUBLIC)) {
      RawDoFmt((STRPTR)error,va,__stuffChar,buffer);

      ReqObject = NewObject( REQUESTER_GetClass(), NULL,
         REQ_TitleText, (ULONG)"distributed.net client",
         REQ_Type, REQTYPE_INFO,
         REQ_GadgetText, (ULONG)"_Moo!",
         REQ_BodyText, (ULONG)buffer,
         TAG_END);

      if (ReqObject) {
         SetAttrs(GlbWindowP, WA_BusyPointer, TRUE, TAG_DONE);
         OpenRequester(ReqObject,GlbIWindowP);
         DisposeObject(ReqObject);
         SetAttrs(GlbWindowP, WA_BusyPointer, FALSE, TAG_DONE);
      }

      FreeVec(buffer);
   }

   va_end(va);
}

VOID UpdateGadget(struct Window *win, struct Gadget *gad, ...)
{
   va_list va;

   if (gad) {
      va_start(va,gad);

      if (SetGadgetAttrsA(gad, win, NULL, (struct TagItem *)va))
         if (win) RethinkLayout(gad, win, NULL, TRUE);

      va_end(va);
   }
}

VOID UpdateMenus(VOID)
{
   if (GlbIWindowP) {
      if (GUIInfo.client68k.task) {
         OnMenu(GlbIWindowP,FULLMENUNUM(1,NOITEM,NOSUB));
      }
      else {
         OffMenu(GlbIWindowP,FULLMENUNUM(1,NOITEM,NOSUB));
      }
      if (GUIInfo.clientppc.task) {
         OnMenu(GlbIWindowP,FULLMENUNUM(2,NOITEM,NOSUB));
      }
      else {
         OffMenu(GlbIWindowP,FULLMENUNUM(2,NOITEM,NOSUB));
      }
   }
}

VOID Iconify(VOID)
{
   ClosePrefsWindow();
   UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_CON68K],LISTBROWSER_Labels,~0,TAG_END);
   UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_CONPPC],LISTBROWSER_Labels,~0,TAG_END);
   DoMethod(GlbWindowP, ISFLAGSET(Prefs.flags,PREFSFLAG_SHOWICON) ? WM_ICONIFY : WM_CLOSE);
   if (ISFLAGSET(Prefs.flags,PREFSFLAG_SHOWMENU)) AppMenu = (struct AppMenuItem *)AddAppMenuItemA((ULONG)GlbWindowP,0,"distributed.net client",AppPort,NULL);
   FreeVisualInfo(GlbVisualInfo);
   GlbVisualInfo = NULL;
   GlbIWindowP = NULL;
}

VOID UnIconify(VOID)
{
   struct Screen *scr;
   if (scr = LockPubScreen(NULL)) {
      if (GlbVisualInfo = GetVisualInfoA(scr,NULL)) {
         if ((GlbIWindowP = (struct Window *)DoMethod(GlbWindowP, WM_OPEN)))
         {
            if (AppMenu) {
               RemoveAppMenuItem(AppMenu);
               AppMenu = NULL;
	    }
            LayoutMenus(GlbMenu, GlbVisualInfo,
                        GTMN_NewLookMenus, TRUE,
                        TAG_DONE);
            UpdateMenus();
            UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_CON68K],LISTBROWSER_Labels,(ULONG)&ConsoleLines68K,LISTBROWSER_Top,ConsoleLines68K.numlines,TAG_END);
            UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_CON68K],LISTBROWSER_Top,ConsoleLines68K.numlines,TAG_END);
            UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_CONPPC],LISTBROWSER_Labels,(ULONG)&ConsoleLinesPPC,LISTBROWSER_Top,ConsoleLinesPPC.numlines,TAG_END);
            UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_CONPPC],LISTBROWSER_Top,ConsoleLinesPPC.numlines,TAG_END);
	 }
         else {
            FreeVisualInfo(GlbVisualInfo);
            GlbVisualInfo = NULL;
	 }
      }
      UnlockPubScreen(NULL,scr);
   }
}

SAVEDS VOID HandleAppMenu(REG(a0,struct Hook *hook), REG(a2,Object *winobj), REG(a1,struct AppMessage *msg))
{
   if (msg->am_Type == AMTYPE_APPMENUITEM) {
      UnIconify();
   }
}

SAVEDS VOID HandleRexx(REG(a0,struct ARexxCmd *cmd), REG(a1,struct RexxMsg *rm))
{
   static const ULONG id2cmd[NUM_REXX] = {
      DNETC_MSG_SHUTDOWN,
      0,
      0,
      DNETC_MSG_PAUSE,
      DNETC_MSG_UNPAUSE,
      DNETC_MSG_FETCH | DNETC_MSG_FLUSH,
      DNETC_MSG_FETCH,
      DNETC_MSG_FLUSH,
      DNETC_MSG_RESTART
   };

   ULONG id = cmd->ac_ID; 

   switch (id) {

      case REXX_HIDE:
         if (GlbWindowP) Iconify();
         break;

      case REXX_SHOW:
         if (!GlbWindowP) UnIconify();
         break;

      default:
         ArexxHookCmds = id2cmd[id];
         break;

   }
}

LIBFUNC ULONG dnetcguiHandleMsgs(REG(d0,ULONG signals),REG(a6,struct LibBase *lb))
{
   WORD code;
   ULONG result, cmds = 0, cmds68k = 0, cmdsppc = 0, slavecmds;
   static ULONG pendingcmds = 0;  // shared between tasks, so use semaphores!

   ObtainSemaphore(&lb->lb_GUILock);

   if (FindTask(NULL) == GUIInfo.master->task) {

      if (signals & ArexxSigMask) {
         ArexxHookCmds = 0;
         RA_HandleRexx(ArexxObj);
         cmds |= ArexxHookCmds;
      }

      if (signals & WindowSigMask) {
         HandlePrefsWindow();

         while ((result = RA_HandleInput(GlbWindowP, &code)) != WMHI_LASTMSG) {
            switch (result & WMHI_CLASSMASK) {
               case WMHI_CLOSEWINDOW:
                  cmds |= DNETC_MSG_SHUTDOWN;
                  break;

               //case WMHI_GADGETUP:
               //   break;

               case WMHI_RAWKEY:
                  switch (result & WMHI_KEYMASK) {
                     case 0x45: // Esc
                        cmds |= DNETC_MSG_SHUTDOWN;
                        break;
		  }
                  break;

               case WMHI_MENUPICK:
                  while ( (result & WMHI_MENUMASK) != MENUNULL ) {
                     struct MenuItem *msel = ItemAddress(GlbMenu, result & WMHI_MENUMASK);
                        if (msel) {
                           switch((ULONG)GTMENUITEM_USERDATA(msel)) {
                              case MENU_ABOUT_ID:
                                 DisplayError(AboutText,GUIInfo.version,lb->lb_LibNode.lib_Version,lb->lb_LibNode.lib_Revision);
                                 break;

                              case MENU_PREFS_ID:
                                 OpenPrefsWindow(GUIInfo.version);
                                 break;

                              case MENU_HIDE_ID:
                                 Iconify();
                                 break;

                              case MENU_PAUSE_ID:
                                 {
                                    struct MenuItem *pause68k, *pauseppc;
                                    pause68k = ItemAddress(GlbMenu,FULLMENUNUM(1,0,NOSUB));
                                    pauseppc = ItemAddress(GlbMenu,FULLMENUNUM(2,0,NOSUB));
                                    if (msel->Flags & CHECKED) {
                                       pause68k->Flags |= CHECKED;
                                       pauseppc->Flags |= CHECKED;
                                       cmds |= DNETC_MSG_PAUSE;
				    }
                                    else {
                                       pause68k->Flags &= ~CHECKED;
                                       pauseppc->Flags &= ~CHECKED;
                                       cmds |= DNETC_MSG_UNPAUSE;
				    }
				 }
                                 break;

                              case MENU_QUIT_ID:
                                 cmds |= DNETC_MSG_SHUTDOWN;
                                 break;

                              case MENU_68KSHUTDOWN_ID:
                                 cmds68k |= DNETC_MSG_SHUTDOWN;
                                 break;

                              case MENU_68KFETCH_ID:
                                 cmds68k |= DNETC_MSG_FETCH;
                                 break;

                              case MENU_68KFLUSH_ID:
                                 cmds68k |= DNETC_MSG_FLUSH;
                                 break;

                              case MENU_68KUPDATE_ID:
                                 cmds68k |= DNETC_MSG_FETCH | DNETC_MSG_FLUSH;
                                 break;

                              case MENU_68KBENCHMARK_ID:
                                 cmds68k |= DNETC_MSG_BENCHMARK;
                                 break;

                              case MENU_68KBENCHMARKALL_ID:
                                 cmds68k |= DNETC_MSG_BENCHMARK_ALL;
                                 break;

                              case MENU_68KPAUSE_ID:
                                 cmds68k |= (msel->Flags & CHECKED) ? DNETC_MSG_PAUSE : DNETC_MSG_UNPAUSE;
                                 break;

                              case MENU_68KRESTART_ID:
                                 cmds68k |= DNETC_MSG_RESTART;
                                 break;

                              case MENU_68KTEST_ID:
                                 cmds68k |= DNETC_MSG_TEST;
                                 break;

                              case MENU_68KCONFIG_ID:
                                 cmds68k |= DNETC_MSG_CONFIG;
                                 break;

                              case MENU_PPCSHUTDOWN_ID:
                                 cmdsppc |= DNETC_MSG_SHUTDOWN;
                                 break;

                              case MENU_PPCFETCH_ID:
                                 cmdsppc |= DNETC_MSG_FETCH;
                                 break;

                              case MENU_PPCFLUSH_ID:
                                 cmdsppc |= DNETC_MSG_FLUSH;
                                 break;

                              case MENU_PPCUPDATE_ID:
                                 cmdsppc |= DNETC_MSG_FETCH | DNETC_MSG_FLUSH;
                                 break;

                              case MENU_PPCBENCHMARK_ID:
                                 cmdsppc |= DNETC_MSG_BENCHMARK;
                                 break;

                              case MENU_PPCBENCHMARKALL_ID:
                                 cmdsppc |= DNETC_MSG_BENCHMARK_ALL;
                                 break;

                              case MENU_PPCPAUSE_ID:
                                 cmdsppc |= (msel->Flags & CHECKED) ? DNETC_MSG_PAUSE : DNETC_MSG_UNPAUSE;
                                 break;

                              case MENU_PPCRESTART_ID:
                                 cmdsppc |= DNETC_MSG_RESTART;
                                 break;

                              case MENU_PPCTEST_ID:
                                 cmdsppc |= DNETC_MSG_TEST;
                                 break;

                              case MENU_PPCCONFIG_ID:
                                 cmdsppc |= DNETC_MSG_CONFIG;
                                 break;
			   }
                           result = msel->NextSelect;
			}
                        else {
                           break;
			}
		  }
                  break;

               case WMHI_ICONIFY:
                  Iconify();
                  break;

               case WMHI_UNICONIFY:
                  UnIconify();
                  break;
	    }
	 }
      }

      if (GUIInfo.master->task == GUIInfo.client68k.task) {
         slavecmds = cmds | cmdsppc;
         cmds |= cmds68k;
      }
      else {
         slavecmds = cmds | cmds68k;
         cmds |= cmdsppc;
      }
      if (slavecmds && GUIInfo.slave) {
         if (!pendingcmds) Signal(GUIInfo.slave->task,GUIInfo.slave->sigmask);
         pendingcmds |= slavecmds;
      }
   }
   else { /* slave */
      cmds = pendingcmds;
      pendingcmds = 0;
   }

   ReleaseSemaphore(&lb->lb_GUILock);

   return(cmds);
}

static BOOL AllocSlaveResources(struct TaskSpecificData *clientdata)
{
   BOOL success = FALSE;

   clientdata->idcmpsigbit = AllocSignal(-1);
   clientdata->appsigbit = AllocSignal(-1);
   clientdata->arexxsigbit = AllocSignal(-1);

   if (clientdata->appsigbit != -1 && clientdata->idcmpsigbit != -1 &&
       clientdata->arexxsigbit)
   {
      clientdata->sigmask = 1L << clientdata->idcmpsigbit |
                            1L << clientdata->appsigbit |
                            1L << clientdata->arexxsigbit;
      success = TRUE;
   }

   return(success);
}

static void FreeSlaveResources(struct TaskSpecificData *clientdata)
{
   clientdata->sigmask = 0;
   if (clientdata->arexxsigbit != -1) {
      FreeSignal(clientdata->arexxsigbit);
   }
   if (clientdata->appsigbit != -1) {
      FreeSignal(clientdata->appsigbit);
   }
   if (clientdata->idcmpsigbit != -1) {
      FreeSignal(clientdata->idcmpsigbit);
   }
}

static void ChangeSlaveToMaster(struct TaskSpecificData *clientdata)
{
   Forbid();
   IDCMPPort->mp_SigBit = clientdata->idcmpsigbit;
   AppPort->mp_SigBit = clientdata->appsigbit;
   ArexxPort->mp_SigBit = clientdata->arexxsigbit;
   IDCMPPort->mp_SigTask = AppPort->mp_SigTask = ArexxPort->mp_SigTask = clientdata->task;
   Permit();

   WindowSigMask = 1L << clientdata->idcmpsigbit | 1L << clientdata->appsigbit;
   ArexxSigMask = 1L << clientdata->arexxsigbit;
}

struct ARexxCmd RexxCmds[] = {
   { "QUIT", REXX_QUIT, HandleRexx, NULL, 0, NULL, 0, 0, NULL },
   { "HIDE", REXX_HIDE, HandleRexx, NULL, 0, NULL, 0, 0, NULL },
   { "SHOW", REXX_SHOW, HandleRexx, NULL, 0, NULL, 0, 0, NULL },
   { "PAUSE", REXX_PAUSE, HandleRexx, NULL, 0, NULL, 0, 0, NULL },
   { "UNPAUSE", REXX_UNPAUSE, HandleRexx, NULL, 0, NULL, 0, 0, NULL },
   { "UPDATE", REXX_UPDATE, HandleRexx, NULL, 0, NULL, 0, 0, NULL },
   { "FETCH", REXX_FETCH, HandleRexx, NULL, 0, NULL, 0, 0, NULL },
   { "FLUSH", REXX_FLUSH, HandleRexx, NULL, 0, NULL, 0, 0, NULL },
   { "RESTART", REXX_RESTART, HandleRexx, NULL, 0, NULL, 0, 0, NULL },
   { 0 }
};

const struct ColumnInfo CInfo68K[] = {
   { 100, "M68K", 0 },
   { -1, NULL, 0 }
};

const struct ColumnInfo CInfoPPC[] = {
   { 100, "PowerPC", 0 },
   { -1, NULL, 0 }
};

static struct Gadget *NewConsoleObject(const struct ColumnInfo *ci)
{
   return ListBrowserObject,
             GA_ReadOnly, TRUE,
             LISTBROWSER_ColumnTitles, FLAGTOBOOL(Prefs.flags,PREFSFLAG_SHOWCONTITLES),
             LISTBROWSER_ColumnInfo, ci,
          ListBrowserEnd;
}

static struct ClientGUIParams *InitGUI(struct TaskSpecificData *clientdata, UBYTE *programname, struct WBArg *iconname)
{
   struct ClientGUIParams *params = NULL;
   struct Screen *screen;
   struct DiskObject *icon;

   NewList(&ConsoleLines68K.list);
   NewList(&ConsoleLinesPPC.list);
   ConsoleLines68K.numlines = 0;
   ConsoleLinesPPC.numlines = 0;

   if (screen = LockPubScreen(NULL)) {
      if (GlbVisualInfo = GetVisualInfoA(screen,NULL)) {
         if (IDCMPPort = CreateMsgPort()) {
            clientdata->idcmpsigbit = IDCMPPort->mp_SigBit;
            if (AppPort = CreateMsgPort()) {
               clientdata->appsigbit = AppPort->mp_SigBit;
               if (GlbMenu = CreateMenusA(ClientMenus, NULL)) {
                  LayoutMenus(GlbMenu, GlbVisualInfo,
                              GTMN_NewLookMenus, TRUE,
                              TAG_DONE);
	       }
               if (ArexxObj = (Object *)NewObject( AREXX_GetClass(), NULL,
                                 AREXX_HostName, (ULONG)"DNETC",
                                 AREXX_Commands, (ULONG)RexxCmds,
                                 TAG_END))
               {
                  Forbid();
                  ArexxPort = FindPort("DNETC.1");
                  Permit();
                  clientdata->arexxsigbit = ArexxPort->mp_SigBit;
                  AppMsgHook.h_Entry = (HOOKFUNC)HandleAppMenu;
                  AppMsgHook.h_SubEntry = (HOOKFUNC)NULL;

                  PrefsDefaults(screen);
                  LoadPrefs();

                  icon = ReadTooltypeOptions(iconname);

                  if (GlbWindowP = WindowObject,
                             WA_Left,              Prefs.winx,
                             WA_Top,               Prefs.winy,
                             WA_InnerWidth,        Prefs.winwidth,
                             WA_InnerHeight,       Prefs.winheight,
                             WA_SizeBBottom,       TRUE,
                             WA_DragBar,           TRUE,
                             WA_CloseGadget,       TRUE,
                             WA_DepthGadget,       TRUE,
                             WA_SizeGadget,        TRUE,
                             WA_Activate,          TRUE,
                             WA_NoCareRefresh,     TRUE,
                             WA_Title,             "distributed.net client",
                             WA_ScreenTitle,       GUIInfo.version,
                             WINDOW_SharedPort,    IDCMPPort,
                             WINDOW_AppPort,       AppPort,
                             WINDOW_AppMsgHook,    &AppMsgHook,
                             WINDOW_MenuStrip,     GlbMenu,
                             WINDOW_Icon,          icon,
                             WINDOW_IconTitle,     "dnetc",
                             WINDOW_IconifyGadget, TRUE,
                             WINDOW_Position,      Prefs.winposmode,
                             WINDOW_Layout,
                                GlbGadgetsP[GAD_MAINLAYOUT] = VLayoutObject,
                                   LAYOUT_SpaceInner, TRUE,
                                   LAYOUT_SpaceOuter, TRUE,
                                   LAYOUT_DeferLayout, TRUE,
                                   (Prefs.font.ta_Name ? GA_TextAttr : TAG_IGNORE), (ULONG)&Prefs.font,

                                   LAYOUT_AddChild, GlbGadgetsP[(clientdata == &GUIInfo.client68k) ? GAD_CON68K : GAD_CONPPC] = NewConsoleObject((clientdata == &GUIInfo.client68k) ? CInfo68K : CInfoPPC),
                                LayoutEnd,
                             WindowEnd)
                  {
                     GlbGadgetsP[(clientdata == &GUIInfo.clientppc) ? GAD_CON68K : GAD_CONPPC] = NULL;
                     WindowSigMask = 1L << clientdata->idcmpsigbit |
                                     1L << clientdata->appsigbit;
                     ArexxSigMask = 1L << clientdata->arexxsigbit;
                     clientdata->sigmask = WindowSigMask | ArexxSigMask;
                     if (ISFLAGSET(Prefs.flags,PREFSFLAG_STARTICONIFIED) || GUIInfo.iconify) {
                        Iconify();
                        params = &GUIParams;
	             }
                     else {
                        if (GlbIWindowP = (struct Window *)DoMethod(GlbWindowP,WM_OPEN)) {
                           params = &GUIParams;
			}
		     }
		  }
	       }
	    }
	 }
      }
      UnlockPubScreen(NULL,screen);
   }

   return (params);
}

static void FreeGUI(void)
{
   ClosePrefsWindow();

   if (AppMenu) RemoveAppMenuItem(AppMenu);

   if (GlbIWindowP) {
      DoMethod(GlbWindowP,WM_CLOSE);
      GlbIWindowP = NULL;
   }
   if (GlbWindowP) {
      DisposeObject(GlbWindowP);
      GlbWindowP = NULL;
   }
   if (ArexxObj) {
      DisposeObject(ArexxObj);
      ArexxObj = NULL;
   }

   FreeMenus(GlbMenu);
   DeleteMsgPort(AppPort);
   DeleteMsgPort(IDCMPPort);
   FreeVisualInfo(GlbVisualInfo);

   GlbMenu = NULL;
   AppPort = NULL;
   IDCMPPort = NULL;
   GlbVisualInfo = NULL;

   FreeListBrowserList(&ConsoleLines68K.list);
   FreeListBrowserList(&ConsoleLinesPPC.list);
}

static VOID CloseLibs(void )
{
   CloseLibrary(LabelBase);
   CloseLibrary(ChooserBase);
   CloseLibrary(IntegerBase);
   CloseLibrary(CheckBoxBase);
   CloseLibrary(ButtonBase);
   CloseLibrary(GetFontBase);
   CloseLibrary(ListBrowserBase);
   CloseLibrary(LayoutBase);
   CloseLibrary(RequesterBase);
   CloseLibrary(WindowBase);
   CloseLibrary(ARexxBase);
   CloseLibrary(GadToolsBase);
   CloseLibrary(WorkbenchBase);
   CloseLibrary(IconBase);
}

static BOOL OpenLibs(void)
{
   if(!(IconBase = OpenLibrary("icon.library",36)))    return FALSE;
   if(!(WorkbenchBase = OpenLibrary("workbench.library",36)))    return FALSE;
   if(!(GadToolsBase = OpenLibrary("gadtools.library",36)))    return FALSE;
   if(!(ARexxBase = OpenLibrary("arexx.class",0))) return FALSE;
   if(!(WindowBase = OpenLibrary("window.class",0))) return FALSE;
   if(!(RequesterBase = OpenLibrary("requester.class",0))) return FALSE;
   if(!(LayoutBase = OpenLibrary("gadgets/layout.gadget",0))) return FALSE;
   if(!(ListBrowserBase = OpenLibrary("gadgets/listbrowser.gadget",0))) return FALSE;
   if(!(GetFontBase = OpenLibrary("gadgets/getfont.gadget",0))) return FALSE;
   if(!(ButtonBase = OpenLibrary("gadgets/button.gadget",0))) return FALSE;
   if(!(CheckBoxBase = OpenLibrary("gadgets/checkbox.gadget",0))) return FALSE;
   if(!(IntegerBase = OpenLibrary("gadgets/integer.gadget",0))) return FALSE;
   if(!(ChooserBase = OpenLibrary("gadgets/chooser.gadget",0))) return FALSE;
   if(!(LabelBase = OpenLibrary("images/label.image",0))) return FALSE;
   return TRUE;
}

LIBFUNC ULONG dnetcguiOpen(REG(d0,ULONG cpu),REG(a0,UBYTE *programname),REG(a1,struct WBArg *iconname),REG(a2,const char *vstring),REG(a6,struct LibBase *lb))
{
   ULONG sigmask = 0;
   struct TaskSpecificData *clientdata = (cpu == DNETCGUI_68K) ? &GUIInfo.client68k : &GUIInfo.clientppc;
   struct Task *task = FindTask(NULL);

   ObtainSemaphore(&lb->lb_GUILock);

   if (!GUIInfo.master) {
      /* Start the GUI */
      if (OpenLibs()) {
         strcpy(GUIInfo.version,vstring);
         if (InitGUI(clientdata,programname,iconname)) {
            clientdata->task = task;
            GUIInfo.open |= cpu;
            GUIInfo.master = clientdata;
            GUIInfo.slave = NULL;
            sigmask = clientdata->sigmask;
            UpdateMenus();
         }
         else {
            FreeGUI();
            CloseLibs();
	 }
      }
   }
   else {
      /* GUI already open - store task of other client, or return failure */
      if ((cpu == DNETCGUI_PPC && !(GUIInfo.open & DNETCGUI_PPC)) ||
          (cpu == DNETCGUI_68K && !(GUIInfo.open & DNETCGUI_68K)))
      {
         if (AllocSlaveResources(clientdata)) {      
            clientdata->task = task;
            GUIInfo.slave = clientdata;
            GUIInfo.open |= cpu;
            sigmask = clientdata->sigmask;
            if (cpu == DNETCGUI_PPC) {
               GlbGadgetsP[GAD_CONPPC] = NewConsoleObject(CInfoPPC);
               UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_MAINLAYOUT],
                              LAYOUT_AddChild,GlbGadgetsP[GAD_CONPPC],
                              TAG_END);
	    }
            else {
               GlbGadgetsP[GAD_CON68K] = NewConsoleObject(CInfo68K);
               UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_MAINLAYOUT],
                              LAYOUT_Inverted,TRUE,
                              LAYOUT_AddChild,GlbGadgetsP[GAD_CON68K],
                              TAG_END);
	    }
            UpdateMenus();
	 }
         else {
            FreeSlaveResources(clientdata);
	 }
      }
   }

   ReleaseSemaphore(&lb->lb_GUILock);

   return (sigmask);
}

LIBFUNC BOOL dnetcguiClose(REG(a0,struct ClientGUIParams *params),REG(a6,struct LibBase *lb))
{
   BOOL success = FALSE;
   struct Task *task = FindTask(NULL);

   ObtainSemaphore(&lb->lb_GUILock);

   if (GUIInfo.open && (GUIInfo.client68k.task == task || GUIInfo.clientppc.task == task)) {
      if (task == GUIInfo.master->task) {
         if (GUIInfo.master && GUIInfo.slave) {
            if (task == GUIInfo.client68k.task) {
               GUIInfo.open &= ~DNETCGUI_68K;
               UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_MAINLAYOUT],
                              LAYOUT_RemoveChild,GlbGadgetsP[GAD_CON68K],
                              TAG_END);
               GlbGadgetsP[GAD_CON68K] = NULL;
	    }
            else { /* task == GUIInfo.clientppc */
               GUIInfo.open &= ~DNETCGUI_PPC;
               UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_MAINLAYOUT],
                              LAYOUT_RemoveChild,GlbGadgetsP[GAD_CONPPC],
                              TAG_END);
               GlbGadgetsP[GAD_CONPPC] = NULL;
	    }
            ChangeSlaveToMaster(GUIInfo.slave);
            FreeSlaveResources(GUIInfo.master);
            GUIInfo.master->task = NULL;
            GUIInfo.master = GUIInfo.slave;
            GUIInfo.slave = NULL;
            success = TRUE;
	 }
         else { /* master */
            FreeGUI();
            CloseLibs();
            GUIInfo.open = 0;
            GUIInfo.master = NULL;
            GUIInfo.client68k.task = NULL;
            GUIInfo.clientppc.task = NULL;
	 }
      }
      else {
         FreeSlaveResources(GUIInfo.slave);
         if (task == GUIInfo.client68k.task) {
            GUIInfo.open &= ~DNETCGUI_68K;
            UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_MAINLAYOUT],
                           LAYOUT_RemoveChild,GlbGadgetsP[GAD_CON68K],
                           TAG_END);
            GlbGadgetsP[GAD_CON68K] = NULL;
	 }
         else { /* task == GUIInfo.clientppc.task */
            GUIInfo.open &= ~DNETCGUI_PPC;
            UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_MAINLAYOUT],
                           LAYOUT_RemoveChild,GlbGadgetsP[GAD_CONPPC],
                           TAG_END);
            GlbGadgetsP[GAD_CONPPC] = NULL;
	 }
         GUIInfo.slave->task = NULL;
         GUIInfo.slave = NULL;
         success = TRUE;
         UpdateMenus();
      }
   }

   ReleaseSemaphore(&lb->lb_GUILock);

   return (success);
}

LIBFUNC VOID dnetcguiConsoleOut(REG(d0,ULONG cpu),REG(a0,UBYTE *output),REG(d1,BOOL overwrite))
{
   struct Gadget *gadget;
   struct Node *node;
   struct ConsoleLines *consolelist;

   if (cpu == DNETCGUI_PPC) {
      gadget = GlbGadgetsP[GAD_CONPPC];
      consolelist = &ConsoleLinesPPC;
   }
   else {
      gadget = GlbGadgetsP[GAD_CON68K];
      consolelist = &ConsoleLines68K;
   }

   if ((node = AllocListBrowserNode(1,LBNCA_CopyText, TRUE,
                                      LBNCA_Text,(ULONG)output,
                                      TAG_END)))
   {
      if (GlbIWindowP) {
         UpdateGadget(GlbIWindowP,gadget,LISTBROWSER_Labels,~0,TAG_END);
      }
      if ((overwrite && !IsListEmpty(&consolelist->list)) ||
          consolelist->numlines >= Prefs.maxlines)
      {
         FreeListBrowserNode(overwrite ? RemTail(&consolelist->list) : RemHead(&consolelist->list));
      }
      else {
         consolelist->numlines++;
      }
      AddTail(&consolelist->list,node);
      if (GlbIWindowP) {
         ULONG top, total, visible;
         GetAttr(LISTBROWSER_VPropTop,gadget,&top);
         GetAttr(LISTBROWSER_VPropTotal,gadget,&total);
         GetAttr(LISTBROWSER_VPropVisible,gadget,&visible);
         UpdateGadget(GlbIWindowP,gadget,LISTBROWSER_Labels,(ULONG)consolelist,(top+visible >= total) ? LISTBROWSER_Top : TAG_IGNORE,consolelist->numlines,TAG_END);
      }
   }
}
