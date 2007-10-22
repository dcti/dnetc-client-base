/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: prefs.c,v 1.3 2007/10/22 16:48:30 jlawson Exp $
 *
 * Created by Oliver Roberts <oliver@futaura.co.uk>
 *
 * ----------------------------------------------------------------------
 * ReAction GUI module for AmigaOS clients - GUI Preferences
 * ----------------------------------------------------------------------
*/

#include "common.h"

#include <classes/window.h>
#include <gadgets/chooser.h>
#include <gadgets/integer.h>
#include <gadgets/checkbox.h>
#include <gadgets/layout.h>
#include <gadgets/listbrowser.h>
#include <gadgets/getfont.h>
#include <gadgets/button.h>
#include <images/label.h>
#include <reaction/reaction_macros.h>
#ifndef __amigaos4__
#include <clib/reaction_lib_protos.h>
#endif

#include <proto/layout.h>
#include <proto/listbrowser.h>
#include <proto/window.h>
#include <proto/getfont.h>
#include <proto/button.h>
#include <proto/label.h>
#include <proto/integer.h>
#include <proto/checkbox.h>
#include <proto/chooser.h>

#include "prefs.h"
#include "main.h"

#define PREFS_VER 1

enum { GAD_FONT=1, GAD_MAXLINES, GAD_TITLES, GAD_SHOWICON, GAD_SHOWMENU, GAD_STARTHIDDEN,
       GAD_WINPOS, GAD_SNAPSHOT, GAD_WINLEFT, GAD_WINTOP, GAD_WINWIDTH, GAD_WINHEIGHT,
       GAD_SAVE, GAD_USE, GAD_CANCEL, NUM_PREFS_GADS };

struct GUIPrefs Prefs;
Object *PrefsWinObj;
struct Window *PrefsWin;
struct Gadget *PrefsGads[NUM_PREFS_GADS];
UBYTE FontName[256];

#ifndef __amigaos4__
struct List *WinPosLabels;
#endif

VOID PrefsDefaults(struct Screen *scr)
{
   Prefs.flags = PREFSFLAG_SHOWCONTITLES | PREFSFLAG_SHOWICON | PREFSFLAG_SHOWMENU | PREFSFLAG_SNAPSHOT;
   Prefs.flagsfreemask = PREFSFLAGMASK;
   Prefs.maxlines = 200;
   Prefs.winwidth = 535;
   Prefs.winheight = 160;
   Prefs.winx = 0;
   Prefs.winy = 0;
   Prefs.winposmode = WPOS_CENTERMOUSE;
   CopyMem(scr->Font,&Prefs.font,sizeof(struct TextAttr));
   strcpy(FontName,Prefs.font.ta_Name);
   Prefs.font.ta_Name = FontName;
}

VOID LoadPrefs(VOID)
{
   BPTR file;
   LONG len;
   char *str;
   ULONG flags;

   if (file = Open("ENV:dnetcgui_ra.prefs",MODE_OLDFILE)) {
      flags = Prefs.flags;

      FRead(file,&Prefs.structsize,2,1);
      FRead(file,&Prefs.prefsver,Prefs.structsize-2,1);
      len = FGetC(file);
      FRead(file,FontName,1,len);
      Close(file);

      Prefs.flags |=  (flags & Prefs.flagsfreemask);
      Prefs.font.ta_Name = FontName;
   }
}

VOID StorePrefs(VOID)
{
   ULONG val;
   struct TextAttr *ta;
   BOOL updatecon = FALSE;

   GetAttr(GETFONT_TextAttr,PrefsGads[GAD_FONT],&val);
   ta = (struct TextAttr *)val;
   if (Stricmp(Prefs.font.ta_Name,ta->ta_Name) != 0 ||
       Prefs.font.ta_YSize != ta->ta_YSize)
   {
      strcpy(Prefs.font.ta_Name,ta->ta_Name);
      Prefs.font.ta_YSize = ta->ta_YSize;
      UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_CON68K],GA_TextAttr,&Prefs.font,TAG_END);
      UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_CONPPC],GA_TextAttr,&Prefs.font,TAG_END);
   }

   GetAttr(GA_Selected,PrefsGads[GAD_TITLES],&val);
   if (val != FLAGTOBOOL(Prefs.flags,PREFSFLAG_SHOWCONTITLES)) {
      Prefs.flags ^= PREFSFLAG_SHOWCONTITLES;
      UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_CON68K],LISTBROWSER_ColumnTitles,val,TAG_END);
      UpdateGadget(GlbIWindowP,GlbGadgetsP[GAD_CONPPC],LISTBROWSER_ColumnTitles,val,TAG_END);
      RethinkLayout(GlbGadgetsP[GAD_MAINLAYOUT],GlbIWindowP,NULL,TRUE);
   }

   GetAttr(INTEGER_Number,PrefsGads[GAD_MAXLINES],&val);
   Prefs.maxlines = val;

   GetAttr(GA_Selected,PrefsGads[GAD_SHOWICON],&val);
   if (val != FLAGTOBOOL(Prefs.flags,PREFSFLAG_SHOWICON)) {
      Prefs.flags ^= PREFSFLAG_SHOWICON;
   }

   GetAttr(GA_Selected,PrefsGads[GAD_SHOWMENU],&val);
   if (val != FLAGTOBOOL(Prefs.flags,PREFSFLAG_SHOWMENU)) {
      Prefs.flags ^= PREFSFLAG_SHOWMENU;
   }

   GetAttr(GA_Selected,PrefsGads[GAD_STARTHIDDEN],&val);
   if (val != FLAGTOBOOL(Prefs.flags,PREFSFLAG_STARTICONIFIED)) {
      Prefs.flags ^= PREFSFLAG_STARTICONIFIED;
   }

   GetAttr(CHOOSER_Selected,PrefsGads[GAD_WINPOS],&val);
   Prefs.winposmode = val;

   GetAttr(GA_Selected,PrefsGads[GAD_SNAPSHOT],&val);
   if (val != FLAGTOBOOL(Prefs.flags,PREFSFLAG_SNAPSHOT)) {
      Prefs.flags ^= PREFSFLAG_SNAPSHOT;
   }
   if (ISFLAGSET(Prefs.flags,PREFSFLAG_SNAPSHOT)) {
      Prefs.winx = GlbIWindowP->LeftEdge;
      Prefs.winy = GlbIWindowP->TopEdge;
      Prefs.winwidth = GlbIWindowP->GZZWidth;
      Prefs.winheight = GlbIWindowP->GZZHeight;
   }

/*
   GetAttr(INTEGER_Number,PrefsGads[GAD_WINLEFT],&val);
   Prefs.winx = val;
   GetAttr(INTEGER_Number,PrefsGads[GAD_WINTOP],&val);
   Prefs.winy = val;
   GetAttr(INTEGER_Number,PrefsGads[GAD_WINWIDTH],&val);
   Prefs.winwidth = val;
   GetAttr(INTEGER_Number,PrefsGads[GAD_WINHEIGHT],&val);
   Prefs.winheight = val;

   SetAttrs(GlbWindowP,WA_Left,(Prefs.winposmode ? GlbIWindowP->LeftEdge : Prefs.winx),
                       WA_Top,(Prefs.winposmode ? GlbIWindowP->TopEdge : Prefs.winy),
                       WA_InnerWidth, Prefs.winwidth, WA_InnerHeight, Prefs.winheight, TAG_END);

   ChangeWindowBox(GlbIWindowP,(Prefs.winposmode ? GlbIWindowP->LeftEdge : Prefs.winx),
                               (Prefs.winposmode ? GlbIWindowP->TopEdge : Prefs.winy),
                               Prefs.winwidth,Prefs.winheight);
   GetAttr(WA_Height,GlbWindowP,&val);
   DisplayError("%ld",val);
   SetAttrs(GlbWindowP,WA_Width,Prefs.winwidth,TAG_END);
*/
}

VOID SavePrefs(char *filename)
{
   BPTR file;
   ULONG len;

   if (file = Open(filename,MODE_NEWFILE)) {
      Prefs.structsize = sizeof(struct GUIPrefs);
      Prefs.prefsver = PREFS_VER;
      Prefs.flagsfreemask = PREFSFLAGMASK;
      FWrite(file,&Prefs,sizeof(struct GUIPrefs),1);
      len = strlen(Prefs.font.ta_Name) + 1;
      FPutC(file,len);
      FWrite(file,Prefs.font.ta_Name,1,len);
      Close(file);
   }
}

VOID OpenPrefsWindow(UBYTE *screentitle)
{
   if (!PrefsWinObj) {
#ifdef __amigaos4__
      static STRPTR WinPosLabels[] = {"Custom (Snapshot)","Centre of screen","Mouse relative",NULL};
#else
      WinPosLabels = ChooserLabels("Custom (Snapshot)","Centre of screen","Mouse relative",NULL);
#endif

      if (PrefsWinObj = WindowObject,
         WA_SizeBBottom, TRUE,
         WA_DragBar, TRUE,
         WA_CloseGadget, TRUE,
         WA_DepthGadget, TRUE,
         WA_SizeGadget, TRUE,
         WA_Activate, TRUE,
         WA_NoCareRefresh, TRUE,
         WA_Title, "GUI Preferences",
         WA_ScreenTitle, screentitle,
         WINDOW_SharedPort, IDCMPPort,
         WINDOW_Position,   WPOS_CENTERSCREEN,
         WINDOW_RefWindow, GlbIWindowP,
         WINDOW_Layout,
            VLayoutObject,
               LAYOUT_SpaceInner, TRUE,
               LAYOUT_SpaceOuter, TRUE,
               LAYOUT_DeferLayout, TRUE,

               LAYOUT_AddChild, HLayoutObject,

                  LAYOUT_AddChild, VLayoutObject,
                     LAYOUT_SpaceInner, TRUE,
                     LAYOUT_SpaceOuter, TRUE,
                     LAYOUT_BevelStyle, BVS_GROUP,
                     LAYOUT_Label,      "Console",

                     LAYOUT_AddChild, PrefsGads[GAD_FONT] = GetFontObject,
                        GA_ID, GAD_FONT,
                        GA_RelVerify, TRUE,
                        GETFONT_TextAttr, &Prefs.font,
                     End,
                     CHILD_Label, LabelObject, LABEL_Text, "_Font", LabelEnd,

                     LAYOUT_AddChild, PrefsGads[GAD_MAXLINES] = IntegerObject,
                        INTEGER_MaxChars, 5,
                        INTEGER_Minimum, 0,
                        INTEGER_Maximum, 9999,
                        INTEGER_Number, Prefs.maxlines,
                     IntegerEnd,
                     CHILD_Label, LabelObject, LABEL_Text, "_Max. Lines", LabelEnd,

                     LAYOUT_AddChild, PrefsGads[GAD_TITLES] = CheckBoxObject,
                        GA_Text, "Show _CPU Tit_les",
                        GA_Selected, FLAGTOBOOL(Prefs.flags,PREFSFLAG_SHOWCONTITLES),
                     CheckBoxEnd,
                  LayoutEnd,

                  LAYOUT_AddChild, VLayoutObject,
                     LAYOUT_SpaceInner, TRUE,
                     LAYOUT_SpaceOuter, TRUE,
                     LAYOUT_BevelStyle, BVS_GROUP,
                     LAYOUT_Label,      "Hide",

                     LAYOUT_AddChild, PrefsGads[GAD_SHOWICON] = CheckBoxObject,
                        GA_ID, GAD_SHOWICON,
                        GA_RelVerify, TRUE,
                        GA_Text, "Show _Icon",
                        GA_Selected, FLAGTOBOOL(Prefs.flags,PREFSFLAG_SHOWICON),
                     CheckBoxEnd,

                     LAYOUT_AddChild, PrefsGads[GAD_SHOWMENU] = CheckBoxObject,
                        GA_ID, GAD_SHOWMENU,
                        GA_RelVerify, TRUE,
                        GA_Text, "_Tools Menu",
                        GA_Selected, FLAGTOBOOL(Prefs.flags,PREFSFLAG_SHOWMENU),
                     CheckBoxEnd,

                     LAYOUT_AddChild, PrefsGads[GAD_STARTHIDDEN] = CheckBoxObject,
                        GA_Text, "_Hidden on Startup",
                        GA_Selected, FLAGTOBOOL(Prefs.flags,PREFSFLAG_STARTICONIFIED),
                     CheckBoxEnd,
                  LayoutEnd,
                  CHILD_WeightedWidth, 70,
               LayoutEnd,

               LAYOUT_AddChild, HLayoutObject,
                  LAYOUT_BevelStyle, BVS_GROUP,
                  LAYOUT_Label,      "Window",
                  LAYOUT_SpaceInner, TRUE,
                  LAYOUT_SpaceOuter, TRUE,

                  LAYOUT_AddChild, PrefsGads[GAD_WINPOS] = ChooserObject,
#ifdef __amigaos4__
                     CHOOSER_LabelArray, WinPosLabels,
#else
                     CHOOSER_Labels, WinPosLabels,
#endif
                     CHOOSER_Selected, Prefs.winposmode,
                     CHOOSER_AutoFit, TRUE,
                     CHOOSER_Justification, CHJ_CENTER,
                  ChooserEnd,
                  CHILD_Label, LabelObject, LABEL_Text, "Initial _Position", LabelEnd,

                  LAYOUT_AddChild, PrefsGads[GAD_SNAPSHOT] = CheckBoxObject,
                     GA_Text, "S_napshot",
                     GA_Selected, FLAGTOBOOL(Prefs.flags,PREFSFLAG_SNAPSHOT),
                  CheckBoxEnd,
               LayoutEnd,
/*
                  LAYOUT_AddChild, HLayoutObject,
                     LAYOUT_SpaceInner, TRUE,
                     LAYOUT_SpaceOuter, TRUE,

                     LAYOUT_AddChild, PrefsGads[GAD_WINLEFT] = IntegerObject,
                        GA_Disabled, Prefs.winposmode,
                        INTEGER_MaxChars, 5,
                        INTEGER_Minimum, 0,
                        INTEGER_Maximum, 16384,
                        INTEGER_Number, Prefs.winx,
                     IntegerEnd,
                     CHILD_Label, LabelObject, LABEL_Text, "Left", LabelEnd,

                     LAYOUT_AddChild, PrefsGads[GAD_WINTOP] = IntegerObject,
                        GA_Disabled, Prefs.winposmode,
                        INTEGER_MaxChars, 5,
                        INTEGER_Minimum, 0,
                        INTEGER_Maximum, 16384,
                        INTEGER_Number, Prefs.winy,
                     IntegerEnd,
                     CHILD_Label, LabelObject, LABEL_Text, "Top", LabelEnd,

                     LAYOUT_AddChild, PrefsGads[GAD_WINWIDTH] = IntegerObject,
                        INTEGER_MaxChars, 5,
                        INTEGER_Minimum, 0,
                        INTEGER_Maximum, 16384,
                        INTEGER_Number, Prefs.winwidth,
                     IntegerEnd,
                     CHILD_Label, LabelObject, LABEL_Text, "Width", LabelEnd,

                     LAYOUT_AddChild, PrefsGads[GAD_WINHEIGHT] = IntegerObject,
                        INTEGER_MaxChars, 5,
                        INTEGER_Minimum, 0,
                        INTEGER_Maximum, 16384,
                        INTEGER_Number, Prefs.winheight,
                     IntegerEnd,
                     CHILD_Label, LabelObject, LABEL_Text, "Height", LabelEnd,
                  LayoutEnd,
               LayoutEnd,
*/

               LAYOUT_AddChild, HLayoutObject,
                  LAYOUT_BevelStyle, BVS_SBAR_VERT,
                  LAYOUT_TopSpacing, 6,
                  LAYOUT_EvenSize, TRUE,

                  LAYOUT_AddChild, PrefsGads[GAD_SAVE] = ButtonObject,
                     GA_RelVerify, TRUE,
                     GA_ID, GAD_SAVE,
                     GA_Text, "_Save",
                  ButtonEnd,
                  CHILD_WeightedWidth, 0,

                  LAYOUT_AddChild, PrefsGads[GAD_USE] = ButtonObject,
                     GA_RelVerify, TRUE,
                     GA_ID, GAD_USE,
                     GA_Text, "_Use",
                  ButtonEnd,
                  CHILD_WeightedWidth, 0,

                  LAYOUT_AddChild, PrefsGads[GAD_CANCEL] = ButtonObject,
                     GA_RelVerify, TRUE,
                     GA_ID, GAD_CANCEL,
                     GA_Text, "_Cancel",
                  ButtonEnd,
                  CHILD_WeightedWidth, 0,
               LayoutEnd,
               CHILD_WeightedHeight, 0,

            LayoutEnd,
         WindowEnd)
      {
         PrefsWin = (struct Window *)DoMethod(PrefsWinObj,WM_OPEN);
      }
   }
   else {
      SetAttrs(PrefsWinObj,WINDOW_FrontBack,WT_FRONT,TAG_END);
      ActivateWindow(PrefsWin);
   }
}

VOID ClosePrefsWindow(VOID)
{
   if (PrefsWin) {
      DoMethod(PrefsWinObj,WM_CLOSE);
      PrefsWin = NULL;
   }
   if (PrefsWinObj) {
      DisposeObject(PrefsWinObj);
      PrefsWinObj = NULL;
   }
#ifndef __amigaos4__
   if (WinPosLabels) {
      FreeChooserLabels(WinPosLabels);
      WinPosLabels = NULL;
   }
#endif
}

VOID HandlePrefsWindow(VOID)
{
   if (PrefsWinObj) {
      WORD code;
      ULONG result,val;
      BOOL close = FALSE;

      while ((result = RA_HandleInput(PrefsWinObj, &code)) != WMHI_LASTMSG) {
         switch (result & WMHI_CLASSMASK) {
            case WMHI_CLOSEWINDOW:
               close = TRUE;
               break;

            case WMHI_GADGETUP:
               switch (result & WMHI_GADGETMASK) {
                  case GAD_FONT:
                     gfRequestFont((Object *)PrefsGads[GAD_FONT], PrefsWin);
                     break;

                  case GAD_SHOWICON:
                     if (code == 0) {
                        GetAttr(GA_Selected,PrefsGads[GAD_SHOWMENU],&val);
                        if (val == 0) UpdateGadget(PrefsWin,PrefsGads[GAD_SHOWMENU],GA_Selected,TRUE,TAG_END);
		     }
                     break;

                  case GAD_SHOWMENU:
                     if (code == 0) {
                        GetAttr(GA_Selected,PrefsGads[GAD_SHOWICON],&val);
                        if (val == 0) UpdateGadget(PrefsWin,PrefsGads[GAD_SHOWICON],GA_Selected,TRUE,TAG_END);
		     }
                     break;

                  /*
                  case GAD_WINPOS:
                     UpdateGadget(PrefsWin,PrefsGads[GAD_WINLEFT],GA_Disabled,code,TAG_END);
                     UpdateGadget(PrefsWin,PrefsGads[GAD_WINTOP],GA_Disabled,code,TAG_END);
                     break;

                  case GAD_SNAPSHOT:
                     UpdateGadget(PrefsWin,PrefsGads[GAD_WINLEFT],INTEGER_Number,GlbIWindowP->LeftEdge,TAG_END);
                     UpdateGadget(PrefsWin,PrefsGads[GAD_WINTOP],INTEGER_Number,GlbIWindowP->TopEdge,TAG_END);
                     UpdateGadget(PrefsWin,PrefsGads[GAD_WINWIDTH],INTEGER_Number,GlbIWindowP->GZZWidth,TAG_END);
                     UpdateGadget(PrefsWin,PrefsGads[GAD_WINHEIGHT],INTEGER_Number,GlbIWindowP->GZZHeight,TAG_END);
                     break;
                  */

                  case GAD_SAVE:
                     StorePrefs();
                     SavePrefs("ENVARC:dnetcgui_ra.prefs");
                  case GAD_USE:
                     if (result & WMHI_GADGETMASK) StorePrefs();
                     SavePrefs("ENV:dnetcgui_ra.prefs");
                  case GAD_CANCEL:
                     close = TRUE;
                     break;
	       }
               break;
	 }
      }

      if (close) ClosePrefsWindow();
   }
}
