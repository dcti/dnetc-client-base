/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: prefs.h,v 1.2.4.1 2004/05/08 10:29:31 oliver Exp $
 *
 * Created by Oliver Roberts <oliver@futaura.co.uk>
 *
 * ----------------------------------------------------------------------
 * ReAction GUI module for AmigaOS clients
 * ----------------------------------------------------------------------
*/

#define PREFSFLAG_STARTICONIFIED	0x01
#define PREFSFLAG_SHOWCONTITLES		0x02
#define PREFSFLAG_SHOWICON		0x04
#define PREFSFLAG_SHOWMENU		0x08
#define PREFSFLAG_SNAPSHOT		0x10

#define PREFSFLAGMASK			~0x1f

#define FLAGTOBOOL(flags,flag) ((flags & flag) == flag)
#define ISFLAGSET(flags,flag) (flags & flag)

#ifdef __amigaos4__
#pragma pack(2)
#endif

struct GUIPrefs {
   UWORD structsize;
   UWORD prefsver;

   struct TextAttr font;
   UWORD winx;
   UWORD winy;
   UWORD winwidth;
   UWORD winheight;
   UWORD winposmode;
   UWORD maxlines;
   ULONG flags;
   ULONG flagsfreemask;
};

#ifdef __amigaos4__
#pragma pack()
#endif

extern struct GUIPrefs Prefs;
extern struct Window *PrefsWin;
extern Object *PrefsWinObj;

VOID OpenPrefsWindow(UBYTE *screentitle);
VOID ClosePrefsWindow(VOID);
VOID HandlePrefsWindow(VOID);
VOID PrefsDefaults(struct Screen *scr);
VOID LoadPrefs(VOID);
