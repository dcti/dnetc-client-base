/*
 * This is the console driver backend, and supports the GUI window,
 * the pipe interface and a native console interface through the
 * common w32ConXXX() layer. Interface selection is automatic, and the 
 * caller does not need to know what kind of interface is in use.
 *
 * Created 03.Oct.98 by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/
const char *w32cons_cpp(void) {
return "@(#)$Id: w32cons.cpp,v 1.1.2.1 2001/01/21 15:10:24 cyp Exp $"; }

#define TRACE
//define any/all/some of the following to TRACE_OUT(x) for sectional tracing
#define TRACE_PAINT(x)    //TRACE_OUT(x)
#define TRACE_FLOW(x)     //TRACE_OUT(x)
#define TRACE_CONFIO(x)   //TRACE_OUT(x)
#define TRACE_FONT(x)     //TRACE_OUT(x)
#define TRACE_TRAY(x)     //TRACE_OUT(x)
#define TRACE_ADJRECT(x)  //TRACE_OUT(x)
#define TRACE_INITEXIT(x) //TRACE_OUT(x)
#define TRACE_PIPE(x)     //TRACE_OUT(x)
#define TRACE_MENU(x)     //TRACE_OUT(x)
#define TRACE_DLG(x)      //TRACE_OUT(x)

#define WIN32_LEAN_AND_MEAN /* for win32 */
#define INCLUDE_SHELLAPI_H  /* for win386 (include <shellapi.h> directly!) */
#include <windows.h>  // include shellapi for win16 here */
#include <shellapi.h> // now include shellapi.h for win32 (tray/registry)

#include "cputypes.h"
#include "clitime.h"  // CliGetTimeString(&tv,4),GetBuildDate()
#include "console.h"  // for CLICONS_[SHORT|LONG]NAME, ConOut, ConInKey
#include "cliident.h" // CliGetFullVersionDescriptor()
#include "clievent.h" // event handler
#include "triggers.h" // for clisetupsignals
#include "logstuff.h" // Log()
#include "setprio.h"  // SetGlobalPriority();
#include "util.h"     // utilGetAppName()/TRACE
#include "cpucheck.h" // GetNumberOfProcessors()
#include "client.h"   // modereq needs client
#include "probfill.h" //LoadSaveProblems(), PROBFILL_UNLOADALL
#define __w16ClientHardStop() LoadSaveProblems(NULL,0,PROBFILL_UNLOADALL)
#include "modereq.h"  // "modes": options
#include "client.h"   // "modes": CONTEST_COUNT for bench
#include "problem.h"  // "modes": IsProblemLoadPermitted before bench
#include "clicdata.h" // "modes": CliGetContestNameFromID() names for bench
#include "w32svc.h"   // win9x win32CliInitializeService()
#include "w32util.h"  // winGetVersion()
#include "w32exe.h"   // winIsGUIExecutable()
#include "w32pre.h"   // winGetInstanceHandle()/winGetShowCommand()
#include "w32ini.h"   // [Write|Get]DCTIProfile[String|Int]()
#include "w32cons.h"  // ourselves

#if (CLIENT_OS == OS_WIN32) /* for _open_osfhandle as used with pipes */
  #include <io.h>
  #include <fcntl.h>
#else
  #define SetForegroundWindow BringWindowToTop
  #define SetBrushOrgEx( __0, __1, __2, __3 ) SetBrushOrg( __0, __1, __2 )
#endif

/* ---------------------------------------------------- */

/* --------- message extensions (also see .h for public ones) --------- */
// caution: WM_USER is also used for control messages (grr!).
#define WM_USER_SHELLNOTIFYICON      (WM_USER+1) /* for tray icon */
#define WM_USER_W16CONS              (WM_USER+0)
// WM_USER_W16CONS wParams command message constants 
   #define W16CONS_CMD_CLEARSCREEN   0x01
   #define W16CONS_CMD_PRINTSTR      0x02
   #define W16CONS_CMD_ISKBHIT       0x03
   #define W16CONS_CMD_GETCH         0x04
   #define W16CONS_CMD_SETPOS        0x05
   #define W16CONS_CMD_GETPOS        0x06
   #define W16CONS_CMD_GETSIZE       0x07
   #define W16CONS_CMD_INDIRDESTROY  0x08 /* indirect WM_DESTROY */
   #define W16CONS_CMD_ECHOLPARAM    0x09 /* just return LPARAM (to ident)*/
// WM_COMMAND wParams,
// public ones are 0.... (DNETC_WCMD_INTERNAL_FIRST-1)
// See w32cons.h for list (they're also duplicated here)
   #define WMCMD_SHUTDOWN             DNETC_WCMD_INTERNAL_FIRST /* 512 */
   #define WMCMD_RESTART              (1+WMCMD_SHUTDOWN)        /* 513 */ 
   #define WMCMD_PAUSE                (1+WMCMD_RESTART)         /* 514 */ 
   #define WMCMD_UNPAUSE              (1+WMCMD_PAUSE)           /* 515 */ 
   #define WMCMD_UPDATE               (1+WMCMD_UNPAUSE)         /* 516 */ 
   #define WMCMD_FETCH                (1+WMCMD_UPDATE)          /* 517 */ 
   #define WMCMD_FLUSH                (1+WMCMD_FETCH)           /* 518 */ 
   #define WMCMD_SVCINSTALL           (1+WMCMD_FLUSH)           /* 519 */ 
   #define WMCMD_SWITCHVIEW           (1+WMCMD_SVCINSTALL)      /* 520 */ 
   #define WMCMD_CLOSEVIEW            (1+WMCMD_SWITCHVIEW)      /* 521 */ 
   #define WMCMD_REFRESHVIEW          (1+WMCMD_CLOSEVIEW)       /* 522 */ 
   #define WMCMD_ABOUT                (1+WMCMD_REFRESHVIEW)     /* 523 */ 
   #define WMCMD_HELP_DOC             (1+WMCMD_ABOUT)           /* 524 */ 
   #define WMCMD_HELP_FAQ             (1+WMCMD_HELP_DOC)        /* 525 */ 
   #define WMCMD_HELP_BUG             (1+WMCMD_HELP_FAQ)        /* 526 */ 
   #define WMCMD_HELP_MAILTO          (1+WMCMD_HELP_BUG)        /* 527 */ 
   #define WMCMD_BENCHMARK            (1+WMCMD_HELP_MAILTO)     /* 528 */ 
   #define WMCMD_EVENT                (1+WMCMD_BENCHMARK)       /* 529 */
   #define WMCMD_CONFIG               (1+WMCMD_EVENT+1+(CONTEST_COUNT*2))
   #define WMCMD_PASTE   WM_PASTE     /* 0x0302 */
   #define WMCMD_COPY    WM_COPY
   #define WMCMD_RESTORE SC_RESTORE   /* 0xF120 */

struct WMCMD_EVENT_DATA
{
  int id;
  const void *parm;
  int isize;
};

struct WM_CREATE_DATA
{
  UINT nCmdShow;
  int create_pending;
  int create_errorcode;
  int *client_run_startstop_level_ptr;
};

/* ------------------------------------------------------------------ */

//#define FONT_RESPECTS_ASPECT
//#define USE_NATIVE_CONSOLEIO //define this to force win32 native console i/o
// define to the dimensions of the virtual screen
#define W16CONS_WIDTH       80
#define W16CONS_HEIGHT      25
// undefine the following to allow the window to be resized
//#define W16CONS_FIXEDSIZE
// define to maximum keyboard buffer size
#define W16CONS_KEYBUFF     128
// smooth font scaling or not (also depends on system capabilities of course)
//#define W16CONS_SMOOTHSIZING

/* --------------------------------------------------------------------- */

// data structure associated with the window
typedef struct W16ConsoleStruc
{
  HWND hwnd;
  char buff[W16CONS_HEIGHT][W16CONS_WIDTH];
  char literal_buff[(W16CONS_HEIGHT+1)*(W16CONS_WIDTH+2)];
  int *client_run_startstop_level_ptr;
  int literal_buff_is_valid;
  int have_marked; /* anything marked? */
  int mark_down;   /* is the lbutton currently to be treated as 'down' */
  POINT mark_mlastpos; /* last _tracked_ mouse position */
  int mark_lastrow; /* most recent row that changed to 'marked' */
  int mark_ignorenextlbdown; /* last left-click was a change-focus click */
  char marked_buff[W16CONS_HEIGHT];
  int currow, curcol;
  struct { int shift:1, alt:1, ctrl:1; } kbflags;
  int keybuff[W16CONS_KEYBUFF];
  int keycount;
  HFONT hfont;
  int fontchecked;
  int fontisstock;
  int fontisvarpitch;
  int fontx, fonty;
  int indentx, indenty;
  int dispcols, disprows;
  int caretpos;
  int smoothsizing;
  int havewinextn; /* need check for SPI_GETDRAGFULLWINDOWS */
  int needsnaphandler; /* we currently don't have SPI_GETDRAGFULLWINDOWS */
  UINT ssmessage;
  //struct { int intray, verasst; char tip[64]; } traydata;
  struct { int top, left, fx, fy, state; } lastpos;
  UINT dnetc_cmdmsg;
  RECT lastpaintrect;
  HICON hSmallIcon;
  int no_handle_wm_poschanged; /* menu generation or WM_DESTROY in progress */
  int nCmdShow; /* initial state of window (any/all views) */
  struct
  {
    FARPROC func; /* MakeProcInstance() handle */
    HWND hwnd; /* dialog window */
  } rate_view;
} * W16CONP;

//typedef struct W16ConsoleStruc * W16CONP;

#define W16CONS_ERR_CREATETHREAD 1
#define W16CONS_ERR_NOFONT       2
#define W16CONS_ERR_NOMEM        3
#define W16CONS_ERR_CREATEWIN    4
#define W16CONS_ERR_REGCLASS     5
#define W16CONS_ERR_GETINST      6
#define W16CONS_ERR_NOSLOT       7
#define W16CONS_ERR_NCCREATE     8

/* ------------------------------------------------ */

/* ------------------------------------------------ */

void __w16writelog( const char *format, ... )
{
  static char *openmode = "w";
  va_list argptr;
  FILE *f;
  va_start(argptr, format);
  if (strlen(format)>=6 && memcmp(format,"callee",6)==0)
  {
    f = fopen("debug.log","a+");
    if (f)
    {
      openmode = "a+";
      fclose(f);
    }
  }
  f = fopen("debug.log",openmode);
  if (f)
  {
    openmode = "a+";
    vfprintf( f, format, argptr );
    if (*format)
      format += (strlen(format)-1);
    if (*format != '\n')
      fwrite("\n",1,1,f);
    fflush(f);
    fclose(f);
  }
  va_end(argptr);
  return;
}

#if (CLIENT_OS == OS_WIN32)
void __w16showerror(const char *caption)
{
  LPVOID lpMsgBuf;
  FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER |
     FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
     GetLastError(),  0 /*MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT)*/,
     (LPTSTR) &lpMsgBuf, 0, NULL );// Process any inserts in lpMsgBuf.
  __w16writelog( "%s: %s\n", caption, lpMsgBuf );
  // w32ConOutModal((const char *)lpMsgBuf );
  LocalFree( lpMsgBuf );
  return;
}
#endif

/* =============================================================== */

static W16CONP __win16GetHwndConsole( HWND hwnd )
{
  #if (CLIENT_OS == OS_WIN32)
  return (W16CONP)GetWindowLong(hwnd, GWL_USERDATA);
  #else
  return (W16CONP)GetWindowLong(hwnd, 0);
  #endif
}

static W16CONP __win16SetHwndConsole( HWND hwnd, W16CONP console )
{
  #if (CLIENT_OS == OS_WIN32)
  return (W16CONP)SetWindowLong(hwnd, GWL_USERDATA, (LONG)console);
  #else
  return (W16CONP)SetWindowLong(hwnd, 0, (LONG)console);
  #endif
}

/* ---------------------------------------------------- */

void __conssize_saveupdateload(W16CONP console, HWND hwnd,
                               int saveupdateload /*-1,0,+1*/)
{
  if (console)
  {
    RECT rect;
    const char *sect = "client";
    if (saveupdateload <= 0) /* refresh or save */
    {
      if (!IsWindow(hwnd))
        ;
      else if (IsIconic(hwnd))
        console->lastpos.state = WS_MINIMIZE;
      else if (IsZoomed(hwnd))
        console->lastpos.state = WS_MAXIMIZE;
      else
      {
        console->lastpos.fy = console->fonty;
        console->lastpos.fx = console->fontx;
        GetWindowRect(hwnd,&rect);
        console->lastpos.top = rect.top;
        console->lastpos.left = rect.left;
        console->lastpos.state = 0;

        TRACE_CONFIO((0,"conf: update pos: %d,%d\n",console->lastpos.left,console->lastpos.top));
        TRACE_CONFIO((0,"conf: update fxy: %d,%d\n",console->lastpos.fx,console->lastpos.fy));
      }
      if (saveupdateload < 0) /* save */
      {
        extern int winSetInstanceShowCmd(int nshow);
        if (console->lastpos.state == WS_MINIMIZE)
          winSetInstanceShowCmd(SW_MINIMIZE);
        else if (console->lastpos.state == WS_MAXIMIZE)
          winSetInstanceShowCmd(SW_MAXIMIZE);
        else
          winSetInstanceShowCmd(SW_SHOWNORMAL);
        TRACE_CONFIO((0,"conf: save pos: %d,%d\n",console->lastpos.left,console->lastpos.top));
        TRACE_CONFIO((0,"conf: save fxy: %d,%d\n",console->lastpos.fx,console->lastpos.fy));
        WriteDCTIProfileInt(sect, "fx", console->lastpos.fx );
        WriteDCTIProfileInt(sect, "fy", console->lastpos.fy );
        WriteDCTIProfileInt(sect, "wtop", console->lastpos.top );
        WriteDCTIProfileInt(sect, "wleft", console->lastpos.left );
        WriteDCTIProfileInt(sect, "state", console->lastpos.state );
      }
    }
    else /* (saveupdateload > 0) */  /* restore */
    {
      int maxwidth  = (int)GetSystemMetrics(SM_CXSCREEN); //width
      int maxheight = (int)GetSystemMetrics(SM_CYSCREEN); //height

      console->lastpos.state = GetDCTIProfileInt(sect, "state", 0 );
      if (console->lastpos.state != WS_MINIMIZE &&
          console->lastpos.state != WS_MAXIMIZE)
        console->lastpos.state = 0;

      if (maxwidth < 700) /* 640x480 */
      {
        console->fontx = 7; //6;
        console->fonty = 12; //10;
      }
      else if (maxwidth < 900) /* 800x600 */
      {
        console->fontx =  8;
        console->fonty = 14;
      }
      else
      {
        console->fontx = 10; // 9;
        console->fonty = 16; //15;
      }
      rect.top = GetDCTIProfileInt( sect, "fy", console->fonty );
      rect.left = GetDCTIProfileInt( sect, "fx", console->fontx );

      if (rect.top > 1 && rect.top <= (maxheight/W16CONS_HEIGHT) &&
         rect.left > 1 && rect.left <= (maxwidth/W16CONS_WIDTH))
      {
        console->fonty = rect.top;
        console->fontx = rect.left;
      }

      console->lastpos.fx = console->fontx;
      console->lastpos.fy = console->fonty;

      rect.top = GetDCTIProfileInt(sect, "wtop", maxheight+1 );
      rect.left = GetDCTIProfileInt(sect, "wleft", maxwidth+1 );

      TRACE_CONFIO((0,"conf: load pos: %d,%d\n",rect.left,rect.top));

      /* just some basic validation for top and left */
      if (rect.top < 0 || rect.top > maxheight )
        rect.top = (rand() % (maxheight>>1)); /* somewhere in the top half */
      if (rect.left < 0 || rect.left > maxwidth )
        rect.left = (rand() % (maxwidth>>1)); /* somewhere in the left half */

      console->lastpos.top = rect.top;
      console->lastpos.left = rect.left;

      TRACE_CONFIO((0,"conf: load pos: %d,%d\n",console->lastpos.left,console->lastpos.top));
      TRACE_CONFIO((0,"conf: load fxy: %d,%d\n",console->lastpos.fx,console->lastpos.fy));
    }
  }
  return;
}

/* ---------------------------------------------------------------------- */

static HINSTANCE my_ShellExecute( HWND hParent, const char *lpOper,
                                  const char *lpFile, const char *lpParams,
                                  const char *lpDir, UINT nShowCmd)
{
  HINSTANCE hInst = NULL;
  #if (CLIENT_OS == OS_WIN32) /* avoid loading shell32 if not needed (4MB+) */
  int weloaded = 0;           /* shell32 postloads SHLWAPI.DLL COMCTL32.DL */
  HMODULE hShell32 = GetModuleHandle("shell32.dll");
  if (!hShell32)
  { 
    UINT olderrmode = SetErrorMode(SEM_NOOPENFILEERRORBOX);
    hShell32 = LoadLibrary( "shell32.dll" );
    SetErrorMode(olderrmode);
    weloaded = 1;
  }
  if (hShell32)
  {
    typedef HINSTANCE (WINAPI *ShellExecuteA_t)(HWND,LPCTSTR,LPCTSTR,LPCTSTR,LPCTSTR,INT);
    ShellExecuteA_t _ShellExecute = (ShellExecuteA_t) GetProcAddress(hShell32, "ShellExecuteA");
    if (_ShellExecute) 
      hInst = (*_ShellExecute)(hParent,lpOper,lpFile,lpParams,lpDir,nShowCmd);
    if (weloaded)
      FreeLibrary(hShell32);
  }
  #else
  hInst = ShellExecute(hParent,lpOper,lpFile,lpParams,lpDir,nShowCmd);
  #endif
  return hInst;
}

#if (CLIENT_OS == OS_WIN32)
/* our tray handler does some optimization if it knows it has shell 4.71 */
/* or greater (ie which supprts the TaskbarCreated message) */
static int GetShell32Version(void)
{
  static int ver = -1;
  if (ver < 0)
  {
    HINSTANCE hShell32 = LoadLibrary("shell32.dll");
    if (hShell32)
    {
      #pragma pack(1)
      typedef struct
      {
        DWORD cbSize;
        DWORD dwMajorVersion;
        DWORD dwMinorVersion;
        DWORD dwBuildNumber;
        DWORD dwPlatformID;
      } my_DllVersionInfo;
      #pragma pack(0)
      typedef DWORD (CALLBACK *_DllGetVersion)(my_DllVersionInfo *);
      _DllGetVersion proc = (_DllGetVersion)
                                GetProcAddress(hShell32, "DllGetVersion");
      if (proc)
      {
        my_DllVersionInfo dvi; 
        memset(&dvi,0,sizeof(dvi));
        dvi.cbSize = sizeof(dvi);
        if (((*proc)(&dvi)) == NOERROR)
          ver = (dvi.dwMajorVersion*100)+(dvi.dwMinorVersion%100);
        else
          ver = 471; /* 4.71 and above support DllGetVersion */
      }
      FreeLibrary(hShell32);
    }
    if (ver < 0)
      ver = (int)(winGetVersion()%2000);
  }
  return ver;
}
#endif

#if (CLIENT_OS == OS_WIN32)

#if 0 //!defined(NOTIFYICONDATA_V1_SIZE)
#pragma pack(1)
typedef struct _NOTIFYICONDATAA_V2 { 
   DWORD cbSize;
   HWND hWnd; 
   UINT uID; 
   UINT uFlags; 
   UINT uCallbackMessage; 
   HICON hIcon; 
   CHAR  szTip[128]; //64 in <5.0
   DWORD dwState; //Version 5.0
   DWORD dwStateMask; //Version 5.0
   CHAR szInfo[256]; //Version 5.0
   UINT  uTimeout; //Version 5.0
   //union with uTimeout: UINT uVersion; //Version 5.0
   CHAR szInfoTitle[64]; //Version 5.0
   DWORD dwInfoFlags; //Version 5.0
} NOTIFYICONDATAA_V2, *PNOTIFYICONDATAA_V2; 
#pragma pack()
#define NOTIFYICONDATA_V1_SIZE \
        (4+sizeof(HWND)+(sizeof(UINT)*3)+sizeof(HICON)+(sizeof(CHAR)*64))
#undef NOTIFYICONDATA
#define NOTIFYICONDATA NOTIFYICONDATAA_V2
#undef PNOTIFYICONDATA
#define PNOTIFYICONDATA PNOTIFYICONDATAA_V2
#define NIM_SETFOCUS    0x00000003
#define NIM_SETVERSION  0x00000004
#define NIF_STATE       0x00000008 
#define NIF_INFO        0x00000010
#define NIIF_INFO       0x00000001
#define NIIF_WARNING    0x00000002
#define NIIF_ERROR      0x00000003
#endif


#undef NOTIFYICONDATA_V1_SIZE /* don't want balloon support after all */
                              /* since it keeps the ballon visible forever */

static int my_Shell_NotifyIcon( DWORD dwMessage, PNOTIFYICONDATA pnid )
{
  typedef BOOL (WINAPI *Shell_NotifyIconAT)(DWORD, PNOTIFYICONDATA);
  static HMODULE hShell32;
  static Shell_NotifyIconAT _Shell_NotifyIcon = NULL;
  static int havenotifyiconproc = -1;
  BOOL success = FALSE;

  TRACE_TRAY((+1,"my_Shell_NotifyIcon(NIM_%s, pnid)\n",
      ((dwMessage==NIM_ADD)?("ADD"):((dwMessage==NIM_DELETE)?("DEL"):("MOD"))) ));

  if (havenotifyiconproc < 0)
  {
    TRACE_TRAY((0,"(havenotifyiconproc < 0)\n"));
    if (dwMessage == NIM_DELETE)
    {
      TRACE_TRAY((0,"No action necessary because NIM_DELETE.\n"));
      success = TRUE;
    }
    else
    {
      havenotifyiconproc = 0;
      if ((winGetVersion() % 2000) >= 400)        // Win95+, NT4+
      {
        UINT olderrmode = SetErrorMode(SEM_NOOPENFILEERRORBOX);
        hShell32 = LoadLibrary( "shell32.dll" );
        SetErrorMode(olderrmode);
        TRACE_TRAY((0,"LoadLibrary( \"shell32.dll\" ) => %x\n", hShell32));
        if (hShell32 != NULL)
        {
          _Shell_NotifyIcon = (Shell_NotifyIconAT) GetProcAddress(hShell32, "Shell_NotifyIconA");
          TRACE_TRAY((0,"GetProcAddress( hmod, \"Shell_NotifyIconA\" ) => %p\n", _Shell_NotifyIcon));
          if (_Shell_NotifyIcon != NULL)
            havenotifyiconproc = +1;
          else
            FreeLibrary( hShell32 );
        }
      }
    }
  }

  if (havenotifyiconproc > 0)
  {
    TRACE_TRAY((+1,"((*_Shell_NotifyIcon)(NIM_%s, pnid ))\n",
       ((dwMessage==NIM_ADD)?("ADD"):((dwMessage==NIM_DELETE)?("DEL"):("MOD"))) ));
    success = (*_Shell_NotifyIcon)(dwMessage, pnid );
    TRACE_TRAY((-1,"((*_Shell_NotifyIcon)()) => %s\n",
                    ((success)?("TRUE (success)"):("FALSE (failed)")) ));

    if (success && dwMessage == NIM_DELETE)
    {
      TRACE_TRAY((0,"FreeLibrary(hShell32)\n"));
      havenotifyiconproc = -1;
      FreeLibrary( hShell32 );
    }
  }
  TRACE_TRAY((-1,"my_Shell_NotifyIcon()=>%s\n",((success)?("TRUE (success)"):("FALSE (failed)")) ));
  return success;
}
#endif

/* action :  <0 (remove from tray), >=0 (add/update to tray). */
/* calledfrom :  debugging trace string. */

static int __DoTrayStuff( HWND hwnd, int action, const char *tip,
                          const char *calledfrom )
{
  int retcode = -1; /* assume failed */
  hwnd = hwnd; action = action; tip = tip; calledfrom = calledfrom; /* shaddup compiler */

  #if (CLIENT_OS == OS_WIN32)
  if ((winGetVersion() % 2000) >= 400)          // Win95+, NT4+
  {
    static int recursive = 0;
    if ((++recursive)==1)
    {
      if (IsWindow(hwnd))
      {
        static int intray = 0;
        DWORD realaction = 0;

        TRACE_TRAY(( +1, "__DoTrayStuff: action=%d, intray=%d, calledfrom=%s\n",
                              action, intray, ((calledfrom)?(calledfrom):("???")) ));

        /* tray stuff is computationally expensive, so
           take some simple steps to avoid unecessary changes
        */
        if (action < 0) /* remove from tray */
        {
          action = 0; /* assume nothing to do */
          if (intray) /* currently in tray? */
          {
            action = 1; /* then something to do */
            realaction = NIM_DELETE;
          }
        }
        else /* if (action >= 0): refresh or create trayicon */
        {
          action = 0;  /* assume nothing to do */
          if (IsIconic(hwnd)) /* now iconic? */
          {
            action = 1;  /* then something to do */
            realaction = NIM_ADD;
          }
          else if (intray) /* not iconic, but currently in the tray? */
          {
            action = 1;  /* then something to do */
            realaction = NIM_DELETE;
          }
        }

        if (action)
        {
          HICON hIcon;
          NOTIFYICONDATA tnd;

          /* WM_GETICON is only 4.x and greater */
          hIcon = (HICON)SendMessage( hwnd, WM_GETICON, 0 /* small */, 0 );
          if (!hIcon)
            hIcon = (HICON)SendMessage( hwnd, WM_GETICON, 1 /* large */, 0 );
          if (!hIcon)
            hIcon = (HICON)GetClassLong(hwnd,GCL_HICON);
          TRACE_TRAY(( 0, "got icon? =%x\n", hIcon ));

          /* construct a default structure */
          memset( (void *)&tnd, 0, sizeof(tnd));
          tnd.cbSize    = sizeof(NOTIFYICONDATA);
          #if (defined(NOTIFYICONDATA_V1_SIZE))
          tnd.cbSize    = NOTIFYICONDATA_V1_SIZE;
          #endif
          tnd.uCallbackMessage= WM_USER_SHELLNOTIFYICON;
          tnd.hWnd      = hwnd;
          tnd.uID       = 1; /* App-defined id of the taskbar icon */
          tnd.hIcon     = hIcon;
          tnd.uFlags    = NIF_MESSAGE|((hIcon)?(NIF_ICON):(0));
          tnd.szTip[0]  = '\0';

          if (realaction != NIM_DELETE) /* NIM_ADD or NIM_MODIFY */
          {
            char szTitle[64]; /* size of V1 tip */
            if (!GetWindowText( hwnd, szTitle, sizeof(szTitle)))
              szTitle[0] = '\0';
            else if (szTitle[0])
            {
              strncpy(tnd.szTip, szTitle, sizeof(tnd.szTip));
              tnd.szTip[63] = '\0';
              tnd.uFlags |= NIF_TIP;
            } 
            if (tip)
            {
              if (*tip)
              {
                strncpy(tnd.szTip, tip, sizeof(tnd.szTip));
                tnd.szTip[63] = '\0';
                tnd.uFlags |= NIF_TIP;
                if (GetShell32Version()>=500)
                {
                  #if (defined(NOTIFYICONDATA_V1_SIZE))
                  strcpy(tnd.szInfoTitle, szTitle);
                  tnd.szInfoTitle[sizeof(tnd.szInfoTitle)-1] = '\0';
                  tnd.dwInfoFlags = NIIF_INFO;
                  tnd.uFlags = NIF_INFO; /* note: equals */
                  tnd.uTimeout = 10000; /* thats minimum */
                  strncpy(tnd.szInfo, tip, sizeof(tnd.szInfo));
                  tnd.szInfo[sizeof(tnd.szInfo)-1] = '\0';
                  tnd.cbSize = sizeof(NOTIFYICONDATA);
                  #endif
                }
              }
            }

            if (intray)
            {
              TRACE_TRAY((+1,"if (intray) {\n")); 
              tnd.uFlags &= ~NIF_ICON;
              /* NIM_MODIFY fails if explorer just got restarted */
              if (my_Shell_NotifyIcon(NIM_MODIFY, &tnd))
                retcode = 0;
              else /* explorer got restarted, so delete and re-add */
              {
                tnd.uFlags|=((tnd.hIcon)?(NIF_ICON):(0));
                my_Shell_NotifyIcon(NIM_DELETE, &tnd);
              }
              TRACE_TRAY((-1,"} => retcode=%d\n",retcode)); 
            }
            if (retcode != 0)  
            {
              if (my_Shell_NotifyIcon(NIM_ADD, &tnd))
              {
                intray = 1;
                if (IsWindowVisible(hwnd))
                {
                  TRACE_TRAY((+1,"ShowWindow(SW_HIDE)\n")); 
                  ShowWindow(hwnd, SW_HIDE);
                  TRACE_TRAY((-1,"ShowWindow(SW_HIDE)\n")); 
                }
                retcode = 0;
              }
            }
          }
          else if (realaction == NIM_DELETE)
          {
            /* NIM_DELETE fails if explorer just got restarted */
            my_Shell_NotifyIcon(NIM_DELETE, &tnd);
            intray = 0;
            retcode = 0;
          }
        }

        TRACE_TRAY(( -1, "__DoTrayStuff: retcode=%d intray=%d\n", retcode, intray ));
      }
    }
    --recursive;
  }
  #endif
  return retcode;
}

/* ---------------------------------------------------------------------- */

static int __stacksaving_IsOEMCharsetOk(void)
{
  char oemv[128], ansiv[128];
  unsigned int c, len = 0;
  for (c = 0x20; c < 0x7F; c++)
  {
    ansiv[c-0x20] = oemv[c-0x20] = (char)c;
    len++;
  }
  OemToAnsiBuff( oemv, ansiv, (short)len ); /* short is needed for win16 */
  if ( memcmp( oemv, ansiv, len ) != 0)
    return 0;
  return 1;
}

static BOOL __w16IsOEMCharsetOk(void) /* are our [' '-'~'] chars the same? */
{
  static int isok = -1;
  if (isok < 0)
    isok = __stacksaving_IsOEMCharsetOk();
  return isok;
}
#if 0 /* call commented out */
static BOOL __w16IsFontMonospaced(HDC hdc, UINT uFirstChar,
    UINT uLastChar, UINT uAvgWidth)
{
  BOOL isok = FALSE;
  LPABC lpabc = (LPABC) malloc(sizeof(ABC) * (uLastChar - uFirstChar + 1));
  if (lpabc != NULL)
  {
    if (GetCharABCWidths(hdc, uFirstChar, uLastChar, lpabc) != 0)
    {
      int myi;
      isok = TRUE;
      for (myi = uLastChar - uFirstChar; isok && myi >= 0; myi--)
        isok = (uAvgWidth == (lpabc[myi].abcA + lpabc[myi].abcB +
          lpabc[myi].abcC));
    }
    free((void *)lpabc);
  }
  return isok;
}
#endif

static inline DWORD my_InitMapperFlags(HDC hdc) /* used by fontomatic and wmpaint */
{
  DWORD mapperFlags = SetMapperFlags( hdc, 0 );
  #if 0 //#ifdef FONT_RESPECTS_ASPECT
  SetMapperFlags( hdc, mapperFlags|1 );
  #else
  SetMapperFlags( hdc, mapperFlags & ~1 );
  #endif
  return mapperFlags;
}


/* find a monospaced font whose extents are */
/* *nearest* to the requested height/width. */
static BOOL __w16Fontomatic( HWND hwnd, HDC hdc, SIZE *newfontsize, 
                             int textarea_width, int textarea_height,
                             int numcols, int numrows, 
                             HFONT *hfontP, int *isstockfont,
                             LOGFONT *logfontP, int *varpitchfont )
                             /* returns TRUE if font changed */
{
  BOOL fontchanged = FALSE;              
  int oldMapMode, newfontx, newfonty, turn, varpitch, owndc;
  DWORD mapperFlags;

  owndc = 0;
  if (!hdc)
  {
    owndc = 1;
    hdc = GetDC( hwnd );
    if (!hdc)
      return FALSE;
  }

  oldMapMode = SetMapMode( hdc, MM_TEXT);
  mapperFlags = my_InitMapperFlags( hdc );
  newfonty  = ((textarea_height + (numrows>>1))/numrows);
  newfontx  = ((textarea_width + (numcols>>1))/numcols);
  varpitch  = ((varpitchfont)?(*varpitchfont):(0));

  //if (console->smoothsizing)
  //{
  //  if ((newfonty * W16CONS_HEIGHT) != textarea_height)
  //    newfonty  = ((height+W16CONS_HEIGHT)/W16CONS_HEIGHT);
  //  if ((newfontx * W16CONS_WIDTH) != textarea_width)
  //    newfontx  = ((width+W16CONS_WIDTH)/W16CONS_WIDTH);
  //}

  TRACE_FONT((+1, "font change: want %d:%d\n", newfontx, newfonty ));

  for (turn=0;;turn++)
  {
    HFONT hfont = NULL;
    LOGFONT lfont;
    int isstock = 0, lastchance = 0;

    //how the font mapper works is well (and understandably) documented
    //in http://msdn.microsoft.com/library/techart/msdn_fontmap.htm

    if (turn == 0)
    {
      //try to avoid a change of typeface if we can help it
      if (*hfontP && !*isstockfont && !varpitch)
      {
        GetObject( *hfontP, sizeof(LOGFONT), (LPSTR) &lfont);
        lfont.lfHeight = newfonty;
        lfont.lfWidth = newfontx;
        hfont = CreateFontIndirect(&lfont);
        isstock = 0;
      }
    }
    #if 0 //old and probably effective only when hfontP == NULL
    else if (turn <= 5)
    {
      isstock = 0;
      hfont = CreateFont( newfonty, newfontx, 0,0, 0, 0,0,0, 0,
          ((turn > 3)?(0):(OUT_TT_PRECIS)),
          ((turn > 2)?(0):(CLIP_CHARACTER_PRECIS)),
          ((turn > 1)?(0):(PROOF_QUALITY)),
          FIXED_PITCH|FF_MODERN,
          ((turn > 4)?(""):("Courier")) );
    }
    #else
    else if (turn <= 9)
    {
      isstock = 0;
      if (turn == 9)
        hfont = CreateFont( newfonty, newfontx,  0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, FIXED_PITCH|FF_MODERN, NULL );
      else
      {
        memset(&lfont,0,sizeof(lfont));
        lfont.lfHeight = newfonty;
        lfont.lfWidth = newfontx;
        lfont.lfWeight = FW_NORMAL;
        lfont.lfCharSet = ANSI_CHARSET; /* critical */
        lfont.lfOutPrecision = OUT_TT_PRECIS;
        lfont.lfClipPrecision = CLIP_DEFAULT_PRECIS;
        lfont.lfQuality = DEFAULT_QUALITY;
        lfont.lfPitchAndFamily = (FIXED_PITCH | FF_MODERN); /* monospaced */

        if (turn == 1)
          strcpy(lfont.lfFaceName,"Lucida Console");
        else if (turn == 2)
          strcpy( lfont.lfFaceName, "Courier New" );

        /* increasingly less nice (3==same as 1/2, but no name) */
        if (turn >= 4) lfont.lfOutPrecision = OUT_DEVICE_PRECIS;
        if (turn >= 5) lfont.lfWeight = FW_DONTCARE;
        if (turn >= 6) lfont.lfCharSet = OEM_CHARSET;
        if (turn >= 7) lfont.lfOutPrecision = OUT_DEFAULT_PRECIS;
        if (turn >= 8) lfont.lfCharSet = DEFAULT_CHARSET;
        hfont = CreateFontIndirect(&lfont);
      }
    }
    #endif
    else
    {
      if (*hfontP == NULL)
      {
        // define to the font type (must be fixed width)
        hfont = (HFONT)GetStockObject(SYSTEM_FIXED_FONT); //ANSI_FIXED_FONT
        isstock = 1;
      }
      lastchance = 1;
    }
    if (hfont)
    {
      TEXTMETRIC tm;
      HFONT hOldfont = (HFONT)SelectObject(hdc, hfont);
      if (GetTextMetrics( hdc, &tm ))
      {
        #ifdef TRACE
        GetObject( hfont, sizeof(LOGFONT), (LPSTR) &lfont);
        TRACE_FONT((0, "turn %d name='%s' pf=%x/%x cs=%d/%d h=%d/%d/%d w=%d/%d,%d/%d\n",
                          turn, lfont.lfFaceName,
                          lfont.lfPitchAndFamily, tm.tmPitchAndFamily,
                          lfont.lfCharSet, tm.tmCharSet,
                          lfont.lfHeight, tm.tmHeight, newfonty,
                          lfont.lfWidth, tm.tmAveCharWidth, tm.tmMaxCharWidth,
                          newfontx ));
        #endif
        if ((tm.tmPitchAndFamily & 0xf0) != FF_DECORATIVE &&
            (tm.tmPitchAndFamily & 0xf0) != FF_SCRIPT)
        {
          /* SDK note on TEXTMETRIC:
             TMPF_FIXED_PITCH    If this bit is set the font is a 
                                 variable pitch font. If this bit is clear 
                                 the font is a fixed pitch font. 
                                 Note very carefully that those meanings 
                                 are the opposite of what the constant 
                                 name implies.
          */
          int fixedpitch = ((tm.tmPitchAndFamily & TMPF_FIXED_PITCH) == 0);
          if (fixedpitch && tm.tmMaxCharWidth != tm.tmAveCharWidth)
            fixedpitch = 0;
          if (*hfontP == NULL || tm.tmHeight == newfonty)
          {
            if (*hfontP == NULL || (fixedpitch && tm.tmAveCharWidth==newfontx)
                             || (!fixedpitch && tm.tmMaxCharWidth <= newfontx))
            {
              if ((tm.tmCharSet == ANSI_CHARSET) ||
                  (tm.tmCharSet == OEM_CHARSET && __w16IsOEMCharsetOk()))
              {
                if (!fixedpitch)
                {
                  fontchanged = TRUE;
                  varpitch = TRUE;
                }
                else /* if (((tm.tmPitchAndFamily & TMPF_FIXED_PITCH) == 0) ||
                  (tm.tmAveCharWidth == tm.tmMaxCharWidth) ||
                  __w16IsFontMonospaced(hdc, 0x20, 0x7E, tm.tmAveCharWidth)) */
                {
                  fontchanged = TRUE;
                  varpitch = FALSE;
                }
                if (fontchanged == TRUE && logfontP)
                  GetObject( hfont, sizeof(LOGFONT), (LPSTR) &lfont);
              }
            }
          }
        }
      }
      SelectObject(hdc, hOldfont);
      if (lastchance && *hfontP == NULL)
      {
        fontchanged = TRUE; /* we _have_ to take it */
        varpitch = FALSE;
        if ((tm.tmPitchAndFamily & TMPF_FIXED_PITCH) == 0)
          varpitch = TRUE; 
      }
      if (fontchanged)
      {
        TRACE_FONT((0, "font change: turn %d accepted\n", turn ));
        if (*hfontP != NULL && *isstockfont == 0)
          DeleteObject(*hfontP);
        if (logfontP)
          memcpy( logfontP, &lfont, sizeof(LOGFONT));
        *isstockfont = isstock;
        if (varpitchfont) *varpitchfont = varpitch;
        *hfontP = hfont;
        newfontsize->cx = newfontx = tm.tmAveCharWidth;
        newfontsize->cy = newfonty = tm.tmHeight;
        break;
      }
      TRACE_FONT((0, "font change: turn %d rejected\n", turn ));
      if (!isstock)
        DeleteObject(hfont);
      hfont = NULL;
    }
    if (lastchance)
      break;
  } //for (turn=0;;turn++)

  SetMapperFlags( hdc, mapperFlags );
  SetMapMode( hdc, oldMapMode );
  if (!owndc)
    ReleaseDC( hwnd, hdc );

  TRACE_FONT((-1, "font %schanged\n", (fontchanged)?(""):("NOT ") ));
  return fontchanged;
}

/* ---------------------------------------------------- */

#define WM_SIZING_3X WM_USER /* anything *BUT* WM_SIZING, WM_NULL or */
                             /* anything else handled below */

#if defined(TRACE)
const char *__trace_map_msg_toname(UINT message)
{
  if (message == WM_CREATE)    return "WM_CREATE";
  if (message == WM_PAINT)     return "WM_PAINT";
  #if defined(WM_SIZING)
  if (message == WM_SIZING)   return "WM_SIZING";
  #endif
  if (message == WM_SIZING_3X) return "WM_SIZING_3X";
  if (message == WM_SIZE)      return "WM_SIZE";
  return "???";
}
#endif

static LRESULT __w16AdjustRect( W16CONP console, HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
  if (console)
  {
    console->disprows = W16CONS_HEIGHT;
    console->dispcols = W16CONS_WIDTH;

    #if defined(WM_SIZING)
    if (message == WM_SIZING)
    {
      if (wParam == WMSZ_BOTTOM)           wParam = HTBOTTOM;
      else if (wParam == WMSZ_BOTTOMLEFT)  wParam = HTBOTTOMLEFT;
      else if (wParam == WMSZ_BOTTOMRIGHT) wParam = HTBOTTOMRIGHT;
      else if (wParam == WMSZ_TOP)         wParam = HTTOP;
      else if (wParam == WMSZ_TOPLEFT)     wParam = HTTOPLEFT;
      else if (wParam == WMSZ_TOPRIGHT)    wParam = HTTOPRIGHT;
      else if (wParam == WMSZ_LEFT)        wParam = HTLEFT;
      else if (wParam == WMSZ_RIGHT)       wParam = HTRIGHT;
      else return 0;
      message = WM_SIZING_3X;
    }
    #endif

    if ( message == WM_CREATE ||
         message == WM_PAINT  ||
         message == WM_SIZING_3X ||
         message == WM_SIZE   ||
         message == 0 )
    {
      int top, left, width, height, oldwidth, oldheight;
      int nctwidth, nctheight; /* non-client width height */
      RECT rect;

      GetWindowRect(hwnd, &rect);
      top = rect.top;
      left = rect.left;
      nctwidth = rect.right - rect.left;
      nctheight = rect.bottom - rect.top;
      TRACE_ADJRECT((0,"adjrect: curr wrect=top=%d, left=%d, bottom=%d, right=%d\n", rect.top,rect.left,rect.bottom,rect.right));
      GetClientRect(hwnd, &rect);
      width = (rect.right - rect.left);
      height = (rect.bottom - rect.top);
      nctwidth -= width;
      nctheight -= height;
      oldwidth  = (console->dispcols * console->fontx)+(console->indentx<<1);
      oldheight = (console->disprows * console->fonty)+(console->indenty<<1);
     
      /* ------------------------------------------- */
      /* oldwidth/oldheight == old client dimentions */
      /* width/height == requested client dimensions */
      /* nctwidth/nctheight == non-client totals     */
      /* ------------------------------------------- */

      if (message == WM_SIZE)
      {
        TRACE_ADJRECT((0, "adjustrect: WM_SIZE\n"));
        width = LOWORD(lParam);  // width of client area
        height = HIWORD(lParam); // height of client area
      }
      else if (message == 0)
      {
        TRACE_ADJRECT((0, "adjustrect: 0\n"));
        oldwidth = oldheight = 0;
      }
      else if (message == WM_PAINT) //dummy meaning wParam and lParam are invalid
      {
        TRACE_ADJRECT((0, "adjustrect: WM_PAINT\n"));
        if (console->hfont && rect.bottom < rect.top) /* rollup support */
          return 0;
      }
      else if (message == WM_SIZING_3X)
      {
        TRACE_ADJRECT((0, "adjustrect: WM_SIZING\n"));
        RECT *wmsizing_rect = (RECT *) lParam;
        height = (wmsizing_rect->bottom - wmsizing_rect->top) - nctheight;
        width  = (wmsizing_rect->right - wmsizing_rect->left) - nctwidth;
        //wParam specifies the edge of window being sized
      }
      else if (message == WM_CREATE)
      {
        TRACE_ADJRECT((0, "adjustrect: WM_CREATE\n"));
        __conssize_saveupdateload(console, hwnd, +1/*saveupdateload=-1,0,+1*/);
        top = console->lastpos.top;
        left = console->lastpos.left;
        width = (console->fontx * console->dispcols)+(console->indentx << 1);
        height = (console->fonty * console->disprows)+(console->indenty << 1);
        oldwidth = oldheight = 0;
      }

      TRACE_ADJRECT((0,"non-client width=%d, height=%d\n", nctwidth, nctheight));
      TRACE_ADJRECT((0,"requested width=%d height=%d\n", width, height));
      TRACE_ADJRECT((0,"current width=%d height=%d\n", oldwidth, oldheight));
      TRACE_ADJRECT((0,"current font=%d x %d (text_area=>%d x %d)\n", 
                           console->fontx, console->fonty,
                           console->fontx * console->dispcols,
                           console->fonty * console->disprows ));

      if (console->fontisstock || console->hfont == NULL ||
         /* no point doing this if the font isn't going to change */
         ((abs(oldwidth - width)-(console->indentx<<1)) >= console->dispcols) ||
         ((abs(oldheight - height)-(console->indenty<<1)) >= console->disprows) )
      {
        SIZE newfontsize; /* in logical units */
        int oldfx, oldfy;
        oldfx = console->fontx;
        oldfy = console->fonty;

        if (__w16Fontomatic( hwnd, NULL, &newfontsize, 
                             (width - (console->indentx << 1)), //textarea.cx
                             (height - (console->indenty << 1)), //textarea.cy
                             console->dispcols, console->disprows,
                             &(console->hfont), &(console->fontisstock), 
                             NULL, &console->fontisvarpitch ))
        {
          /* font has changed */

          //save new font size
          console->fontx = newfontsize.cx;
          console->fonty = newfontsize.cy;
          console->fontchecked = 0;

          TRACE_ADJRECT((0,"new font=%d x %d (text_area=>%d x %d)\n", 
                            console->fontx, console->fonty,
                            console->fontx * console->dispcols,
                            console->fonty * console->disprows ));
        }
      }
      if (console->hfont == NULL)
        return FALSE;

      //this is the width/height that was expected of us
      oldwidth = width;
      oldheight = height;

      //make width and height the new client dimensions
      width  = (console->dispcols * console->fontx) + (console->indentx << 1);
      height = (console->disprows * console->fonty) + (console->indenty << 1);

      TRACE_ADJRECT((0,"requested width=%d height=%d\n", oldwidth, oldheight));
      TRACE_ADJRECT((0,"accepted width=%d height=%d\n", width, height));

//char tbuf[128];
//sprintf(tbuf,"%d,%d %d,%d (%d,%d)", oldwidth,oldheight,width,height,(console->dispcols * console->fontx), (console->disprows * console->fonty) );
//SetWindowText(hwnd,tbuf);

      //if this a WM_SIZING message, then adjust the new size and return
      if (message == WM_SIZING_3X)
      {
        RECT *wmsizing_rect = (RECT *) lParam;
        //make width and height the new window dimensions
        if (wParam == HTTOP || wParam == HTTOPRIGHT || wParam == HTTOPLEFT)
          wmsizing_rect->top = wmsizing_rect->bottom - (height + nctheight);
        if (wParam == HTBOTTOM || wParam == HTBOTTOMRIGHT || wParam == HTBOTTOMLEFT)
          wmsizing_rect->bottom = wmsizing_rect->top + (height + nctheight);
        if (wParam == HTTOPRIGHT || wParam == HTBOTTOMRIGHT || wParam == HTRIGHT)
          wmsizing_rect->right = wmsizing_rect->left + (width + nctwidth);
        if (wParam == HTTOPLEFT || wParam == HTBOTTOMLEFT || wParam == HTLEFT)
          wmsizing_rect->left = wmsizing_rect->right - (width + nctwidth);
        return TRUE;
      }
      else if (message == WM_PAINT)
      {
        TRACE_ADJRECT((+1, "WM_PAINT setwindowpos(%d,%d,%d,%d)\n",left,top,(width+nctwidth),(height+nctheight) ));
        //SetWindowPos(hwnd, 0, left, top,
        //             (width + nctwidth), (height + nctheight),
        //             SWP_NOREDRAW|SWP_NOMOVE|SWP_NOZORDER|SWP_NOACTIVATE);
        TRACE_ADJRECT((-1, "WM_PAINT setwindowpos\n"));
        return TRUE;
      }
      else if (message == WM_CREATE)
      {
        TRACE_ADJRECT((+1, "WM_CREATE setwindowpos(%d,%d,%d,%d)\n",left,top,(width+nctwidth),(height+nctheight) ));
        SetWindowPos(hwnd, NULL, left, top,
                     (width + nctwidth), (height + nctheight),
                     SWP_NOREDRAW|SWP_NOZORDER|SWP_NOACTIVATE);
        GetWindowRect(hwnd, &rect);
        TRACE_ADJRECT((-1, "WM_CREATE setwindowpos=%d,%d,%d,%d\n", rect.left, rect.top, rect.right-rect.left, rect.bottom-rect.top ));
        return TRUE;
      }
      else if (message == WM_SIZE && (oldwidth != width || oldheight != height))
      {
        TRACE_ADJRECT((+1, "WM_SIZE invalidaterect/updatewindow\n"));
        //Invalidate the rectange (although HREDRAW|VREDRAW should do this)
        InvalidateRect(hwnd,NULL,FALSE);
        UpdateWindow( hwnd );
        TRACE_ADJRECT((-1, "WM_SIZE invalidaterect/updatewindow\n"));
        return TRUE;
      }
      else if (oldwidth != width || oldheight != height)
      {
        #if 0 //(CLIENT_OS == OS_WIN16) /* this will crash on win32 */
        {
          TRACE_ADJRECT((0, "step 9.3 MoveWindow(%d,%d,width=%d,height=%d)\n",left,top,width,height));
          MoveWindow(hwnd, left, top, (width + nctwidth), (height + nctheight), TRUE);
        }
        #endif
        return TRUE;
      }

    } /* if (message == ... ) */
  } /* if console */
  return FALSE;
}

/* ---------------------------------------------------- */

/* used by console WM_PAINT and all owner-draw button paints */
static void __w16DrawRecessedFrame( HDC hDC, const RECT *rect, HBRUSH hBGBrush)
{
  HPEN hPen; HBRUSH hBrush;
  POINT polys[10];

  if (!hBGBrush)
    hBGBrush = (HBRUSH)GetStockObject(NULL_BRUSH);

  hBrush = (HBRUSH)SelectObject(hDC, hBGBrush);
  hPen = CreatePen(PS_SOLID, 2, GetSysColor(COLOR_BTNSHADOW)); //dark grey
  hPen = (HPEN)SelectObject(hDC, hPen);
  Rectangle(hDC,rect->left,rect->top,rect->right,rect->bottom);
  hPen = (HPEN)SelectObject(hDC, hPen);
  hBrush = (HBRUSH)SelectObject(hDC, hBrush);
  DeleteObject(hPen);
  //DeleteObject(hBrush);

  /* *xx* legend: Polyline() draws upto, but not including, so */
  /* go one pixel further */

  if (hBGBrush != (HBRUSH)GetStockObject(BLACK_BRUSH))
  {
    hPen = (HPEN)SelectObject(hDC,GetStockObject(BLACK_PEN));
    polys[0].x = rect->right-2; polys[0].y = rect->top+1;
    polys[1].x = rect->left+1;  polys[1].y = rect->top+1;
    polys[2].x = rect->left+1;  polys[2].y = rect->bottom-2; /* *-3* */
    Polyline(hDC,&polys[0],3);
    SelectObject(hDC, hPen);
  }

  hPen = CreatePen(PS_SOLID, 1, GetSysColor(COLOR_BTNHIGHLIGHT)); //white
  hPen = (HPEN)SelectObject(hDC, hPen);
  polys[0].x = rect->left;      polys[0].y = rect->bottom-1;
  polys[1].x = rect->right-1;   polys[1].y = rect->bottom-1;
  polys[2].x = rect->right-1;   polys[2].y = rect->top-1; /* *-0* */
  Polyline(hDC,&polys[0],3);
  DeleteObject(SelectObject(hDC, hPen));

  hPen = CreatePen(PS_SOLID, 1, GetSysColor(COLOR_BTNFACE)); //light grey
  hPen = (HPEN)SelectObject(hDC, hPen);
  polys[0].x++; /* left+1  */ polys[0].y--; /* bottom-2 */
  polys[1].x--; /* right-2 */ polys[1].y--; /* bottom-2 */
  polys[2].x--; /* right-2 */ polys[2].y++; /* top+1 */
  Polyline(hDC,&polys[0],3);
  DeleteObject(SelectObject(hDC, hPen));

  return;
}

/* ---------------------------------------------------- */

/* these next two functions are primarily for win16 that doesn't get
** WM_SIZING messages, and also for win32 if SPI_GETDRAGFULLWINDOWS
** is disabled or running on win95 with WindowsPlus! or running on NT4.
** In essence __w16Handle_NCLBUTTONDOWN() makes the window modal, and
** snapping the resize frame to the next best size when the user 
** moves the mouse. This is approximately what happens when dragfullwindows 
** is supported but disabled.
*/
/* our emulation is so good, we emulate dragfullwindows unless debugging */
/* or when running on a platform where drag full windows is unknown */
#define SHOWDRAGGING ((winGetVersion()%2000) >= 400)

static void __DrawResizeRect(HWND hwnd, const RECT *rect)
{
  HDC hdc = NULL;
  if (!SHOWDRAGGING)
    hdc = GetDC(NULL); /* get DC for entire screen */
  if (hdc)
  {
    int cxFrame, cyFrame;
    int width = (rect->right - rect->left);
    int height = (rect->bottom - rect->top);
    HBRUSH hb, oldbrush;
    BITMAP bm;
    HBITMAP hbm;
    hwnd = hwnd; /* shaddup compiler */

    // See the KB Article Q68569 for information about how to draw the 
    // resizing rectangle.  That's where this pattern comes from.
    WORD aZigzag[] = { 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA };

    // Fill out the bitmap structure for the PatBlt calls later
    bm.bmType = 0;
    bm.bmWidth = 8;
    bm.bmHeight = 8;
    bm.bmWidthBytes = 2;
    bm.bmPlanes = 1;
    bm.bmBitsPixel = 1;
    bm.bmBits = aZigzag;

    hbm = CreateBitmapIndirect(&bm);
    hb = CreatePatternBrush(hbm);   
  
    cxFrame = GetSystemMetrics(SM_CXFRAME);
    cyFrame = GetSystemMetrics(SM_CYFRAME);

    oldbrush = (HBRUSH)SelectObject(hdc, hb);

    PatBlt(hdc, rect->left, rect->top, width, cyFrame, PATINVERT);
    PatBlt(hdc, rect->left, (rect->top + cyFrame), cxFrame, height - cyFrame, 
                PATINVERT);
    PatBlt(hdc, rect->left + cxFrame, rect->bottom - cyFrame, width - cxFrame,
			cyFrame, PATINVERT);
    PatBlt(hdc, rect->right - cxFrame, rect->top + cyFrame, cxFrame,
  		    height - cyFrame - cyFrame, PATINVERT);

    SelectObject(hdc,oldbrush);
    ReleaseDC(NULL, hdc);
    DeleteObject(hb);
    DeleteObject(hbm);
  }
  return;
}

static BOOL __w16FixupRect( W16CONP console, HWND hwnd, 
                            UINT message, UINT hittest,
                            const RECT *oldrect, RECT *rect )
{
  if (console)
  {
    RECT adjtype, newrect;
    BOOL grow = (rect->top < oldrect->top || rect->bottom > oldrect->bottom ||
                 rect->left < oldrect->left || rect->right > oldrect->right);

    message = message;     
    #if defined(WM_SIZING)
    if (message == WM_SIZING)
    {
      if (hittest == WMSZ_BOTTOM)           hittest = HTBOTTOM;
      else if (hittest == WMSZ_BOTTOMLEFT)  hittest = HTBOTTOMLEFT;
      else if (hittest == WMSZ_BOTTOMRIGHT) hittest = HTBOTTOMRIGHT;
      else if (hittest == WMSZ_TOP)         hittest = HTTOP;
      else if (hittest == WMSZ_TOPLEFT)     hittest = HTTOPLEFT;
      else if (hittest == WMSZ_TOPRIGHT)    hittest = HTTOPRIGHT;
      else if (hittest == WMSZ_LEFT)        hittest = HTLEFT;
      else if (hittest == WMSZ_RIGHT)       hittest = HTRIGHT;
      else return 0;
    }
    #endif

    adjtype.left =   (hittest == HTLEFT   || hittest == HTTOPLEFT    || hittest == HTBOTTOMLEFT);
    adjtype.right =  (hittest == HTRIGHT  || hittest == HTTOPRIGHT   || hittest == HTBOTTOMRIGHT);
    adjtype.top =    (hittest == HTTOP    || hittest == HTTOPLEFT    || hittest == HTTOPRIGHT);
    adjtype.bottom = (hittest == HTBOTTOM || hittest == HTBOTTOMLEFT || hittest == HTBOTTOMRIGHT);

    if (adjtype.top)    adjtype.top = ((grow)?(-1):(+1));
    if (adjtype.bottom) adjtype.bottom = ((grow)?(+1):(-1));
    if (adjtype.left)   adjtype.left = ((grow)?(-1):(+1));
    if (adjtype.right)  adjtype.right = ((grow)?(+1):(-1));
     
    if (winGetVersion() <= 400) /* assume slow machine */
    {
      /* primitive optimization by assuming that the window has to */
      /* grow/shrink at least one pixel per character cell */
      adjtype.top *= console->disprows;
      adjtype.bottom *= console->disprows;
      adjtype.left *= console->dispcols;
      adjtype.right *= console->dispcols;
    }
        
    memcpy( &newrect, rect, sizeof(RECT));
    for (;;)
    {
      RECT tmprect; 
      memcpy( &tmprect, &newrect, sizeof(RECT));
      __w16AdjustRect( console, hwnd, WM_SIZING_3X, hittest, (LPARAM)&tmprect);
      if ( memcmp( &tmprect, rect, sizeof(RECT) ) != 0)
      { 
        memcpy( rect, &tmprect, sizeof(RECT));      
        return TRUE;
      }
      if ((adjtype.top || adjtype.bottom) && console->fonty <= 4)
        break;  
      if ((adjtype.left || adjtype.right) && console->fontx <= 2)
        break;  
      newrect.top += adjtype.top;
      newrect.bottom += adjtype.bottom;
      newrect.left += adjtype.left;
      newrect.right += adjtype.right;
    }
  }
  return FALSE;
}

static LRESULT __w16Handle_NCLBUTTONDOWN(W16CONP console, HWND hwnd, 
                           UINT message, WPARAM wParam, LPARAM lParam)
                                         
{                                         
  if (console) 
  {
    if (wParam >= HTLEFT && wParam <= HTBOTTOMRIGHT)
    { 
      int minheight, minwidth, ncheight, ncwidth, maxright, maxbottom;
      RECT rect, cliprect, adjtype; POINT lastpos;

      GetWindowRect(GetDesktopWindow(),&rect);
      maxright = rect.right;
      maxbottom = rect.bottom;

      GetWindowRect(hwnd, &rect);
      GetClientRect(hwnd, &cliprect);
      ncheight = (rect.bottom - rect.top)-(cliprect.bottom - cliprect.top);
      ncwidth  = (rect.right - rect.left)-(cliprect.right - cliprect.left);
      minwidth = (console->dispcols * 2)+(console->indentx *2)+ncwidth;
      minheight = (console->disprows * 4)+(console->indenty *2)+ncheight;

      adjtype.left =   (wParam == HTLEFT   || wParam == HTTOPLEFT    || wParam == HTBOTTOMLEFT);
      adjtype.right =  (wParam == HTRIGHT  || wParam == HTTOPRIGHT   || wParam == HTBOTTOMRIGHT);
      adjtype.top =    (wParam == HTTOP    || wParam == HTTOPLEFT    || wParam == HTTOPRIGHT);
      adjtype.bottom = (wParam == HTBOTTOM || wParam == HTBOTTOMLEFT || wParam == HTBOTTOMRIGHT);

      SetWindowPos(hwnd,HWND_TOPMOST,0,0,0,0,SWP_NOMOVE|SWP_NOSIZE);
      SetCapture(hwnd);
      GetClipCursor(&cliprect);
      ClipCursor(NULL);
      __DrawResizeRect(hwnd, &rect);
      GetCursorPos(&lastpos); /* don't use lParam */
      for (;;)
      {
        MSG msg;
        while (!PeekMessage(&msg,hwnd,WM_MOUSEFIRST, WM_MOUSELAST, PM_REMOVE))
          WaitMessage();
        #if (CLIENT_OS == OS_WIN32)
        if (GetForegroundWindow() != hwnd)
          break;
        #endif
        if (msg.message == WM_LBUTTONUP)
          break;
        else if (msg.message == WM_MOUSEMOVE)
        {
          RECT framerect; POINT pos;
          GetCursorPos(&pos); /* don't use lParam */
          memcpy( &framerect, &rect, sizeof(framerect)); 

          if (adjtype.left)
            framerect.left = pos.x;
          else if (adjtype.right)
            framerect.right = pos.x;
          if (adjtype.top)
            framerect.top = pos.y;
          else if (adjtype.bottom)
            framerect.bottom = pos.y;

          lastpos.x = pos.x;
          lastpos.y = pos.y;
          
          if (__w16FixupRect(console, hwnd, message, wParam, &rect, &framerect))
          { 
            if (memcmp(&rect, &framerect, sizeof(RECT))!=0)
            {
              __DrawResizeRect(hwnd, &rect);
              if (SHOWDRAGGING) /* don't shock the user */
              {
                MoveWindow(hwnd, framerect.left, framerect.top, 
                                 framerect.right-framerect.left+1, 
                                 framerect.bottom-framerect.top+1, TRUE);  
              }
              memcpy(&rect,&framerect,sizeof(rect));
              __DrawResizeRect(hwnd, &rect); 
            }
          }
        } /* if (msg.message == WM_MOUSEMOVE) */
      } /* for (;;) */
      __DrawResizeRect(hwnd, &rect);
      ClipCursor(&cliprect);
      ReleaseCapture();
      SetWindowPos(hwnd,HWND_NOTOPMOST,0,0,0,0,SWP_NOMOVE|SWP_NOSIZE);
      MoveWindow(hwnd, rect.left, rect.top, rect.right-rect.left+1, 
                 rect.bottom-rect.top+1, TRUE);  
      return 0;
    }
  } 
  return DefWindowProc(hwnd, message, wParam, lParam);
}

/* ---------------------------------------------------- */

static int __win16AdjustCaret(HWND hwnd, W16CONP console, int destroy_first )
{
  if (console)
  {
    if (destroy_first)
      DestroyCaret();
  
    if (console->rate_view.hwnd)
    {
      ; //nothing
    }
    else if (console->smoothsizing == 0)
    {
      int col = console->caretpos;
      if (col == 0)
        col = console->curcol * console->fontx;
      CreateCaret( hwnd, NULL, console->fontx, console->fonty / 6);
      SetCaretPos( console->indentx + col,
                   console->indenty + 
                  (console->currow + 1) * console->fonty -(console->fonty / 6) );
      ShowCaret( hwnd );
    }
    else
    {
      RECT clirect;
      int row, col, cwidth, cheight;
      GetClientRect( hwnd, &clirect );
    
      clirect.top += console->indenty;
      clirect.left += console->indentx;
      clirect.bottom -= console->indenty;
      clirect.right -= console->indentx;

      col = ( ((unsigned long)(clirect.right-clirect.left)) *
            ((unsigned long)(console->curcol)) ) / console->dispcols;
      row = ( ((unsigned long)(clirect.bottom-clirect.top)) *
            ((unsigned long)(console->currow)) ) / console->disprows;
      cwidth = (clirect.right-clirect.left) / console->dispcols;
      cheight = (clirect.bottom-clirect.top) / console->disprows;

      if (console->caretpos > 0)
        col = console->caretpos;

      CreateCaret( hwnd, NULL, cwidth, cheight / 6);
      SetCaretPos( console->indentx+col, 
                   console->indenty+row+(cheight-cheight/6) );
      ShowCaret( hwnd );
    }
  }
  return 0;
}

/* ---------------------------------------------------- */

static int __have_uri_support(int uri_type) /* 'h'==http, 'm'==mailto */
{
  static int http = -1, mailto = -1;
  if (http == -1)
  {
    HKEY hKey; LONG res;
    const char *lookfor;

    lookfor = "\\.html\\ShellEx";
    #if (CLIENT_OS == OS_WIN32)
    res = RegOpenKeyEx(HKEY_CLASSES_ROOT, lookfor, 0, KEY_EXECUTE, &hKey);
    #else
    res = RegOpenKey(HKEY_CLASSES_ROOT, lookfor, &hKey);
    #endif
    if (res == ERROR_SUCCESS)
      RegCloseKey(hKey);
    http = ((res == ERROR_SUCCESS || res == ERROR_ACCESS_DENIED)?(1):(0));

    lookfor = "\\mailto\\shell\\open\\command";
    #if (CLIENT_OS == OS_WIN32)
    res = RegOpenKeyEx(HKEY_CLASSES_ROOT, lookfor, 0, KEY_EXECUTE, &hKey);
    #else
    res = RegOpenKey(HKEY_CLASSES_ROOT, lookfor, &hKey);
    #endif
    if (res == ERROR_SUCCESS)
      RegCloseKey(hKey);
    mailto = ((res == ERROR_SUCCESS || res == ERROR_ACCESS_DENIED)?(1):(0));
  }
  return ((uri_type == 'm')?(mailto):(http));
}

/* recursively delete a menu, its submenus, the submenus of those submenus,...*/
static void __w16WindowDestroyMenu(HMENU hMenu)
{
  if (hMenu)
  {
    TRACE_MENU((+1,"__w16WindowDestroyMenu(hMenu=%p)\n",hMenu));
    int n, itemcount = GetMenuItemCount(hMenu);
    for (n=itemcount-1; n>=0; n--)
    {
      HMENU hSubMenu = GetSubMenu(hMenu, n);
      if (hSubMenu)
        __w16WindowDestroyMenu(hSubMenu);
    }
    DestroyMenu(hMenu);
    TRACE_MENU((-1,"__w16WindowDestroyMenu(hMenu=%p)\n",hMenu));
  }
  return;
}

static HMENU __w16WindowConstructMenu(W16CONP console, HWND hwnd, 
                                      int message, int intray)
{
  HMENU hretmenu = NULL;
  int exiting = CheckExitRequestTrigger();
  int aspopup = (message == 0); /* else WM_CREATE/WM_INITMENU etc */
  hwnd = hwnd; /* shaddup compiler */

  TRACE_MENU((+1,"__w16WindowConstructMenu(aspopup=%d, exiting=%d)\n",aspopup,exiting));
  if (!(aspopup && exiting)) /* popup+exiting => no popup menu */ 
  {
    int israteview = 0, isconsview = 0; /* yes, we need both */
    int havemarked = 0;
    int modebits = ModeReqIsSet(-1); /* get all */
    hretmenu = ((aspopup)?(CreatePopupMenu()):(CreateMenu()));

    TRACE_MENU((0,"1. CreatePopupMenu()=%p, modebits=0x%x\n",hretmenu,modebits));
    if (hretmenu)
    {
      int addrestore = 0;
      if (console)
      {
        if (console->rate_view.func) /* is switchable */
        {
          israteview = (console->rate_view.hwnd != NULL);
          isconsview = (console->rate_view.hwnd == NULL);
        }
        havemarked = console->have_marked;
      }
      if (aspopup && (modebits & MODEREQ_CONFIG)!=0)
      {
        if (intray)
          addrestore = 1;
        else
        {
          AppendMenu(hretmenu, (IsClipboardFormatAvailable(CF_TEXT)?MF_ENABLED:MF_GRAYED),
                             WMCMD_PASTE, "Paste");
          AppendMenu(hretmenu, ((havemarked)?(MF_ENABLED):(MF_GRAYED)),
                             WMCMD_COPY, "Copy" );
        }
      }
      else
      {
        int ispaused;
        char oplabel[128];
        const char *oplabelp;
        HMENU hbench = NULL;

        if (aspopup)
        {
          AppendMenu(hretmenu, ((havemarked)?(MF_ENABLED):(MF_GRAYED)),
                              WMCMD_COPY, "Copy" );
          AppendMenu(hretmenu, MF_SEPARATOR, 0, "" );
        }
        AppendMenu(hretmenu, ((exiting || modebits & (MODEREQ_FLUSH|MODEREQ_CONFIG))?(MF_GRAYED):(MF_ENABLED)),
                            WMCMD_FLUSH, "F&lush Work" );
        AppendMenu(hretmenu, ((exiting || modebits & (MODEREQ_FETCH|MODEREQ_CONFIG))?(MF_GRAYED):(MF_ENABLED)),
                              WMCMD_FETCH, "F&etch Work" );
        AppendMenu(hretmenu, ((exiting || modebits & (MODEREQ_FETCH|MODEREQ_FLUSH|MODEREQ_CONFIG))?(MF_GRAYED):(MF_ENABLED)),
                              WMCMD_UPDATE, "&Update Buffers (Fetch and Flush)" );
        AppendMenu(hretmenu, MF_SEPARATOR, 0, "" );
        AppendMenu(hretmenu, ((exiting || israteview || (modebits & (MODEREQ_CONFIG)))?(MF_GRAYED):(MF_ENABLED)),
                              WMCMD_CONFIG, "Con&figure" );


        #if 0 //(CLIENT_OS == OS_WIN32)
        if (win32CliIsServiceInstalled()==0) /* <0=err,0==no,>0=yes */
        {
          char buff[64];
          long ver = winGetVersion();
          sprintf(buff, "&Install as %s service", ((ver < 2000) ? "Win9x" : "WinNT"));
          AppendMenu(hretmenu, MF_SEPARATOR, 0, "" );
          AppendMenu(hretmenu, ((modebits)?(MF_GRAYED):(MF_ENABLED)),
                           WMCMD_SVCINSTALL, buff );
          AppendMenu(hretmenu, MF_SEPARATOR, 0, "" );
        }
        #endif

        hbench = NULL;  
        if (!exiting && !intray && isconsview &&
           (modebits & (MODEREQ_CONFIG|MODEREQ_BENCHMARK)) == 0)
        {
          hbench = CreatePopupMenu();
        }
        if (hbench)
        {
          unsigned int contest;
          int mpos = WMCMD_BENCHMARK;
          AppendMenu(hbench, MF_ENABLED, mpos++, "All long" );
          AppendMenu(hbench, MF_ENABLED, mpos++, "All short" );
          for (contest = 0;contest < CONTEST_COUNT; contest++)
          {
            int ok2bench = (IsProblemLoadPermitted(-1,contest)?(MF_ENABLED):(MF_GRAYED));
            oplabelp = CliGetContestNameFromID(contest);
            sprintf(oplabel,"%s long", oplabelp );
            AppendMenu(hbench, ok2bench, mpos++, oplabel );
            sprintf(oplabel,"%s short", oplabelp );
            AppendMenu(hbench, ok2bench, mpos++, oplabel );
          }
        }
        AppendMenu(hretmenu, ((hbench)?(MF_ENABLED|MF_POPUP):(MF_GRAYED)),
                                  (UINT)hbench, "&Benchmark" );
        AppendMenu(hretmenu, MF_SEPARATOR, 0, "" );

        ispaused = CheckPauseRequestTriggerNoIO() & TRIGSETBY_SIGNAL;
        // only dnetc -pause and menu pause can be cleared by menu
        oplabelp = ((ispaused)?("Res&ume"):("&Pause"));
        if (!exiting && modebits)
        {
          sprintf(oplabel,"%s (may be delayed)", oplabelp);
          oplabelp = (const char *)&oplabel[0];
        }
        AppendMenu(hretmenu, ((exiting)?(MF_GRAYED):(MF_ENABLED)),
           ((ispaused)?(WMCMD_UNPAUSE):(WMCMD_PAUSE)), oplabelp );
        AppendMenu(hretmenu, MF_ENABLED, WMCMD_RESTART, "&Restart");

        AppendMenu(hretmenu, ((exiting)?(MF_GRAYED):(MF_ENABLED)),
                         WMCMD_SHUTDOWN,
                         ((exiting)?("Shutdown (is pending)"):("Shutdown")));
        if (intray && !exiting)
        {
          AppendMenu(hretmenu, MF_SEPARATOR, 0, "" );
          addrestore = 1;
        }
      }
      if (addrestore && !exiting)
        AppendMenu(hretmenu, MF_ENABLED, WMCMD_RESTORE, "Restore");
    } /* if hretmenu */

    if (hretmenu && !aspopup)
    {
      HMENU hpopup = CreateMenu();
      BOOL appended;
      TRACE_MENU((0,"2. CreateMenu()=%p\n",hpopup));
      if (hpopup)
      {  
        HMENU hswap = hretmenu;
        hretmenu = hpopup;
        hpopup = hswap;
        appended = AppendMenu(hretmenu, MF_ENABLED|MF_POPUP, (UINT)hpopup, "&Client" );
        TRACE_MENU((0,"AppendMenu(Client)=>%s\n",((appended)?("ok"):("failed")) ));
        hpopup = CreateMenu();
      }
      if (hpopup) 
      {
        if (!isconsview)
        {
          AppendMenu(hretmenu, MF_GRAYED, 0, "&Edit" );
        }
        else
        {
          AppendMenu(hpopup, ((havemarked)?(MF_ENABLED):(MF_GRAYED)),
                            WMCMD_COPY, "Copy" );
          AppendMenu(hpopup, (((modebits & MODEREQ_CONFIG)!=0 && 
                         IsClipboardFormatAvailable(CF_TEXT))?MF_ENABLED:MF_GRAYED),
                         WMCMD_PASTE, "Paste");
          appended = AppendMenu(hretmenu, MF_ENABLED|MF_POPUP, (UINT)hpopup, "&Edit" );
          TRACE_MENU((0,"AppendMenu(Edit)=>%s\n",((appended)?("ok"):("failed")) ));
          hpopup = CreateMenu();
        }
      }
      if (hpopup)
      {
        AppendMenu(hpopup, ((isconsview)?(MF_GRAYED):(MF_ENABLED)), WMCMD_SWITCHVIEW, "&Console" );
        AppendMenu(hpopup, ((israteview || (modebits & MODEREQ_CONFIG)!=0)?(MF_GRAYED):(MF_ENABLED)), WMCMD_SWITCHVIEW, "Core &Throughput" );
        appended = AppendMenu(hretmenu, ((isconsview || israteview)?(MF_ENABLED):(MF_GRAYED))|MF_POPUP, (UINT)hpopup, "&View" );
        TRACE_MENU((0,"AppendMenu(View)=>%s\n",((appended)?("ok"):("failed")) ));
        hpopup = CreateMenu();
      }
      if (hpopup)
      {
        int mf = ((__have_uri_support('h'))?(MF_ENABLED):(MF_GRAYED));
        AppendMenu(hpopup, mf, WMCMD_HELP_DOC, "Online Documentation" );
        AppendMenu(hpopup, mf, WMCMD_HELP_FAQ, "Online FAQ" );
        AppendMenu(hpopup, mf, WMCMD_HELP_BUG, "Bug Reports" );
        AppendMenu(hpopup, MF_SEPARATOR, 0, "" );
        AppendMenu(hpopup, MF_ENABLED, WMCMD_ABOUT, "&About" );
        appended = AppendMenu(hretmenu,  MF_POPUP, (UINT)hpopup, "&Help" );
        TRACE_MENU((0,"AppendMenu(About)=>%s\n",((appended)?("ok"):("failed")) ));
      }
      if (!hpopup)
      {
        __w16WindowDestroyMenu(hretmenu);
        hretmenu = NULL;
      }
    } /* if (hretmenu && !aspopup) */
  }
  TRACE_MENU((-1,"__w16WindowConstructMenu(...)=>%p\n",hretmenu));
  return hretmenu;
}




/* ---------------------------------------------------- */

static HFONT __w16FixupDlgFont(HWND hdlg, HFONT hFixedUpFont)
{
  /* dialog fonts are bold on NT3.x/Win3x - fix that */
  HFONT hfontDlg = NULL;
  if ((winGetVersion()%2000) < 400) /* NT3.x/Win3x */
  {
    hfontDlg = (HFONT)SendMessage(hdlg, WM_GETFONT, 0, 0L);
    if (!hfontDlg && hFixedUpFont) /* ie unset */
    {
      hfontDlg = (HFONT)GetStockObject(ANSI_VAR_FONT);
    }
    else if (hfontDlg && !hFixedUpFont) /* ie set */
    {
      LOGFONT lFont;
      if (!GetObject(hfontDlg, sizeof(LOGFONT), (LPSTR) &lFont)) 
        hfontDlg = (HFONT)NULL;
      else 
      {
        lFont.lfWeight = FW_NORMAL;
        hfontDlg = CreateFontIndirect(&lFont);
      }
    }
    if (hfontDlg) 
    {
      HWND hCtrl = GetWindow(hdlg, GW_CHILD);
      while (hCtrl && IsChild(hdlg, hCtrl))
      {
        SendMessage(hCtrl, WM_SETFONT, (WPARAM) hfontDlg, 0);
        hCtrl = GetWindow(hCtrl, GW_HWNDNEXT);
      }
    }
    if (hFixedUpFont)
    {
      DeleteObject(hFixedUpFont);
      return NULL;
    }
  }
  return hfontDlg;
}

static void __w16Set_BS_OWNERDRAW(HWND button_hwnd)
{  /* Convert custom buttons to BS_OWNERDRAW at runtime */
   /* the resource has non-BS_OWNERDRAW buttons for easy visualization */
   /* and so that the dialog does measureitem for us. */
  if (button_hwnd)
  {
    LONG x_styles = BS_3STATE|BS_AUTO3STATE|BS_AUTOCHECKBOX|BS_USERBUTTON|
                    BS_CHECKBOX|BS_DEFPUSHBUTTON|BS_GROUPBOX|BS_LEFTTEXT|
                    BS_PUSHBUTTON|BS_RADIOBUTTON|BS_AUTORADIOBUTTON;
    SetWindowLong( button_hwnd, GWL_STYLE, 
        ((GetWindowLong( button_hwnd, GWL_STYLE ) & ~x_styles)|BS_OWNERDRAW) );
  }
  return;
}


DWORD CALLBACK __w16AboutBox( HWND dialog, UINT msg, WORD wParam, LONG lParam )
{
  struct dlgdata
  {
    HWND hOwner;
    COLORREF clrBG, clrFG;
    HBRUSH hBGBrush;
    int bugs_uri_visited;
    int http_uri_able;
    int mail_uri_able;
    HFONT hFont;
  } *dd;
  char buffer[64];
  HWND hwnd; 
  HDC hDC;

  switch (msg)
  {
    case WM_INITDIALOG:
    {
      dd = (struct dlgdata *)malloc(sizeof(struct dlgdata));
      if (dd)
      {
        SetWindowLong(dialog, DWL_USER, (LONG)dd);
        if (dd != ((struct dlgdata *)GetWindowLong(dialog, DWL_USER)))
        {
          free((void *)dd);
          dd = (struct dlgdata *)0;
        }
      }
      if (!dd)
      {
        SetWindowLong(dialog,DWL_USER,0);
        EndDialog( dialog, TRUE );
        return( TRUE );
      }
      dd->clrBG = GetSysColor(COLOR_BTNFACE);
      dd->clrFG = GetSysColor(COLOR_BTNTEXT);
      dd->hBGBrush = CreateSolidBrush(dd->clrBG); /* don't use GetSysColorBrush()!! */
      dd->hOwner = (HWND)lParam;
      dd->bugs_uri_visited = 0;
      dd->http_uri_able = __have_uri_support('h' /*http*/);
      dd->mail_uri_able = __have_uri_support('m' /*mail*/);
      dd->hFont = __w16FixupDlgFont(dialog, NULL);
     
      if (dd->hOwner)
      {
        hwnd = dd->hOwner; 
        #if defined(WM_GETICON)
        /* although dialog boxes don't have an icon, do this so that */
        /* alt-tab will show something other than the microsoft flag */
        SendMessage(dialog,WM_SETICON,0,SendMessage(hwnd,WM_GETICON,0,0));
        SendMessage(dialog,WM_SETICON,1,SendMessage(hwnd,WM_GETICON,1,0));
        #endif
        GetWindowText(hwnd,buffer,sizeof(buffer));
        strcat(buffer," "); /* hide from other clients */
        SetWindowText(dialog,buffer); 
      }

      if ((hwnd = GetDlgItem( dialog, 201 )) != NULL)
      {
        SetWindowText( hwnd, CliGetFullVersionDescriptor() );
      }
      if ((hwnd = GetDlgItem( dialog, 202 )) != NULL)
      {
        SetWindowText( hwnd, "This client is maintained by\n"
        #if defined(_M_ALPHA)
        "Mike Marcelais <michmarc@microsoft.com>"
        #else
        "Cyrus Patel <cyp@distributed.net>, <cyp@fb14.uni-mainz.de>"
        #endif
        );
      }
      if ((hwnd = GetDlgItem( dialog, 203 )) != NULL)
      {
        SetWindowText( hwnd, "http://www.distributed.net/bugs/" );
        __w16Set_BS_OWNERDRAW(hwnd);
      }
      if ((hwnd = GetDlgItem( dialog, 204 )) != NULL)
      { 
        SetWindowText( hwnd, "help@distributed.net" );
        __w16Set_BS_OWNERDRAW(hwnd);
      }
      return( TRUE );
    }
    case WM_DRAWITEM:
    {
      dd = (struct dlgdata *)GetWindowLong(dialog, DWL_USER);
      if (dd)
      {
        int id = wParam; //lpdis.CtlID;
        DWORD len = GetDlgItemText(dialog,id,buffer,sizeof(buffer));
        if (len)
        {
          int need_refocus = 1;
          COLORREF fg = dd->clrFG;
          HFONT hFont = NULL;
          #if defined(__WINDOWS_386__) /* need 16:32 pointer */
          DRAWITEMSTRUCT far *lpdis = (DRAWITEMSTRUCT far *)MK_FP32((void *)lParam);
          #else
          LPDRAWITEMSTRUCT lpdis = (LPDRAWITEMSTRUCT)lParam;
          #endif
          hDC = lpdis->hDC;
          if (id == 205)
          {
          
          }
          else if ((id == 203 && dd->http_uri_able) ||
                   (id == 204 && dd->mail_uri_able) )
          {
            LOGFONT lf; 
            hFont = (HFONT)SelectObject(hDC,GetStockObject(SYSTEM_FONT));
            SelectObject(hDC, hFont);
            GetObject(hFont,sizeof(lf), &lf);
            lf.lfUnderline = TRUE;
            hFont = CreateFontIndirect(&lf);
            need_refocus = 0; 
            if (hFont)
            {
              int mute_color = 0;
              hFont = (HFONT)SelectObject(hDC, hFont);
#if 0
              if (lpdis->itemAction == ODA_SELECT &&
                  lpdis->itemState == ODS_FOCUS)
              {
                if (id == 203)
                {
                }
                else
                {
                }
                need_refocus = 1;
              }
#endif
              if (id == 203 && dd->bugs_uri_visited)
                mute_color = 1;
              fg = RGB( 0, 0, ((mute_color)?(128):(255)) );
            }
          }
          SetTextColor( hDC, fg );
          SetBkColor( hDC, dd->clrBG );
          SetBkMode( hDC, TRANSPARENT);
          SetBrushOrgEx( hDC, 0, 0, NULL );
          SelectObject( hDC, dd->hBGBrush );
          TextOut( hDC, 0, 0, buffer, (short)len ); /* short for win16 */
          if (hFont)
            DeleteObject(SelectObject(hDC,hFont)); 
          if (need_refocus)
            SetFocus(GetDlgItem(dialog,IDOK));
        }
      } /* if (dd) */
      SetWindowLong(dialog, DWL_MSGRESULT, TRUE);
      return TRUE;
    }
    #if defined(WM_CTLCOLOREDIT) /* win32 and win32s */
    case WM_CTLCOLOREDIT:
    {
      if (GetDlgCtrlID((HWND)lParam) != 205) 
        break;
      /* fallthrough */
    }
    #endif
    #if !defined(WM_CTLCOLOR) /* only defined in windowsx.h */
    #define WM_CTLCOLOR         0x0019
    #endif
    case WM_CTLCOLOR:
    {
      dd = (struct dlgdata *)GetWindowLong(dialog, DWL_USER);
      if (dd)
      {
        hDC = (HDC) wParam;
        SetTextColor(hDC, dd->clrFG);
        SetBkColor(hDC, dd->clrBG);
        SetBkMode( hDC, TRANSPARENT);
        SetBrushOrgEx( hDC, 0, 0, NULL );
        if (msg != WM_CTLCOLOR) 
        {
          SelectObject( hDC, dd->hBGBrush );
          SetWindowLong(dialog, DWL_MSGRESULT, (LONG)dd->hBGBrush );
        }
        return (LRESULT)dd->hBGBrush;
      }
      return FALSE;
    }
    case WM_COMMAND:
    {
      if (LOWORD(wParam) == IDOK )
        PostMessage(dialog, WM_CLOSE, 0, 0 );
      else 
      {
        dd = (struct dlgdata *)GetWindowLong(dialog, DWL_USER);
        if (dd)
        {
          if (!dd->hOwner)
            dd = (struct dlgdata *)0;
        }
        if (dd && LOWORD(wParam) == 204)
        {
          SetFocus(GetDlgItem(dialog,IDOK));
          PostMessage( dd->hOwner, WM_COMMAND, WMCMD_HELP_MAILTO, 0);
        }
        else if (dd && LOWORD(wParam) == 203)
        {
          SetFocus(GetDlgItem(dialog,IDOK));
          dd->bugs_uri_visited = 1;
          PostMessage( dd->hOwner, WM_COMMAND, WMCMD_HELP_BUG, 0);
        }  
      }
      return FALSE;
    }
    case WM_CLOSE:
    {
      dd = (struct dlgdata *)GetWindowLong(dialog, DWL_USER);
      if (dd)
      {
        if (dd->hBGBrush)
        {
          DeleteObject( dd->hBGBrush );
          dd->hBGBrush = NULL;
        }
        if (dd->hFont)
        {
          __w16FixupDlgFont(dialog, dd->hFont);
          dd->hFont = NULL;
        }
        free((void *)dd);
        SetWindowLong(dialog,DWL_USER,0);
      }
      EndDialog( dialog, TRUE );
      return( TRUE );
    }
    default:
    {
      break;
    }
  } /* switch (msg) */
  return( FALSE );
}

static void __launch_about_box(HWND hParent)
{
  #ifndef GWL_HINSTANCE
  #define GWL_HINSTANCE (-6)
  #endif
  HINSTANCE hInst = (HINSTANCE)GetWindowLong(hParent, GWL_HINSTANCE);
  if ( hInst )
  {
    if (FindResource( hInst, MAKEINTRESOURCE(1), RT_DIALOG ))
    {
      FARPROC func = MakeProcInstance( (FARPROC)__w16AboutBox, hInst);
      DialogBoxParam( hInst, MAKEINTRESOURCE(1), hParent, 
                      (DLGPROC)func, (LPARAM)hParent );
      (void)FreeProcInstance( func );
    }
  }
  return;
}

static int __IsViewable(HWND hwnd) /* whether part of a window is visible */
{                                  /* to the user */
  /* is there a faster way to do this? */
  int isvis = (IsWindowVisible(hwnd) != 0);
  if (isvis)
  {
    HDC hDC = GetDC(hwnd); //GetWindowDC(hwnd);
    if (hDC)
    {
      RECT rect;
      GetClientRect(hwnd,&rect);
      isvis = (RectVisible(hDC, &rect) != 0);
      ReleaseDC(hwnd, hDC);
    }
  }
  return isvis;
}

static void __w16DrawList( HWND hwnd, W16CONP console, 
                           int font_height, int __tab_width,
                           DRAWITEMSTRUCT far *lpdis) /*far needed for w16*/
{
  HDC hDC = lpdis->hDC;
  UINT mapmode = SetMapMode(hDC, MM_TEXT);
  HBRUSH hBrush; //HPEN hPen; 
  RECT rect;
  rect.top = lpdis->rcItem.top; rect.left = lpdis->rcItem.left;
  rect.right = lpdis->rcItem.right; rect.bottom = lpdis->rcItem.bottom;

  TRACE_DLG((+1,"__w16DrawList(%p,%p,%p)\n",hwnd,console,lpdis));

  hBrush = (HBRUSH)SelectObject(hDC, GetStockObject(NULL_BRUSH));
  FillRect(hDC, &rect, hBrush );
  SelectObject(hDC, hBrush);

  if (lpdis->itemID != ((UINT)-1) /* && (lpdis->itemAction == ODA_SELECT ||
                             lpdis->itemAction == ODA_DRAWENTIRE)*/ )
  {
    const char *linep;
    char buffer[W16CONS_WIDTH+1];
    unsigned int linelen = 0;

    TRACE_DLG((0,"lpdis->itemData = %p, &buff[lpdis->itemID][0] = %p\n",
             lpdis->itemData, &(console->buff[lpdis->itemID][0]) ));
    linep = (const char *)(lpdis->itemData);
    if (console && lpdis->itemID < W16CONS_HEIGHT)
    {
      linep = &(console->buff[lpdis->itemID][0]);
      if ((unsigned int)console->currow == lpdis->itemID && console->curcol != 0)
        linep = (const char *)0;
    }
    linelen = 0;
    if (linep)
    {
      memcpy(buffer, linep, sizeof(buffer));
      buffer[sizeof(buffer)-1] = '\0';
      linep = &buffer[0];
      linelen = 0;
      while (buffer[linelen] == ' ')
        linelen++;
      if (!buffer[linelen])
        linelen = 0;
      else
      {        
        if (buffer[0] == ' ')
        {
          buffer[--linelen] = '\t'; 
          linep = &buffer[linelen];
        }
        else if (buffer[0] == '[')
        {
          char *q = strchr(buffer,']');
          if (q)
          {
            if (*(++q))
              *q = '\t';
          }
        }
        linelen = strlen(linep); 
        while (linelen > 0 && linep[linelen-1]==' ') 
          linelen--;
        TRACE_DLG((0,"draw list: line='%s', linelen=%d\n", linep, linelen));
      }
    }
    if (linelen > 0)
    {
      #if defined(__WINDOWS_386__)
      short tab_width = __tab_width;
      #else
      int tab_width = __tab_width;
      #endif
      SIZE fsize; 
      fsize.cy = (rect.bottom + rect.top - font_height) / 2;
      fsize.cx = 0;
      TabbedTextOut(hDC,fsize.cx,fsize.cy,linep,(UINT)linelen,1,&tab_width,0);
    }  /* linelen > 0 */
    if (SendMessage(hwnd,LB_GETSEL,lpdis->itemID,0))
      InvertRect(hDC, &rect); //DrawFocusRect(hDC, &rect);
  } /* if (lpdis->itemID >= 0) */

  TRACE_DLG((-1,"__w16DrawList(...)\n"));
  SetMapMode(hDC, mapmode);
  return;
}


static int __w16GetSliderDims(const RECT *slider_area, RECT *slider_rect)
{
  int range; RECT btn;
  btn.top    = slider_area->top;
  btn.left   = slider_area->left;
  btn.bottom = slider_area->bottom - 2;
  btn.right  = btn.left + ((btn.bottom - btn.top)+1)/2;
  range = ((slider_area->right - slider_area->left)-2)-((btn.right-btn.left)-1); 
  if (slider_rect)
    memcpy(slider_rect, &btn, sizeof(RECT));
  return range;
}

static void __ShadeRect(HDC hDC, const RECT *lpRect, BOOL dark)
{
  //http://support.microsoft.com/support/kb/articles/Q128/7/86.asp
  HDC hMemDC = CreateCompatibleDC( hDC );
  if (hMemDC)
  {
    WORD aZigZag[8] = { 0x0055, 0x00aa, 0x0055, 0x00aa,
                        0x0055, 0x00aa, 0x0055, 0x00aa };
    HBITMAP hBrushBitmap = CreateBitmap( 8, 8, 1, 1, &aZigZag[0] );
    if (hBrushBitmap)
    {
      HBRUSH hBrush = CreatePatternBrush( hBrushBitmap );
      if (hBrush)
      { 
        UINT nWidth = lpRect->right - lpRect->left + 1;
        UINT nHeight = lpRect->bottom - lpRect->top + 1;
        HBITMAP hBitmap = CreateCompatibleBitmap( hDC, nWidth, nHeight );
        if (hBitmap)
        {
          RECT rc; COLORREF oldfgcolor, oldbgcolor; UINT oldbkmode;           
          rc.top = rc.left = 0; rc.right = nWidth; rc.bottom = nHeight;
    
          hBitmap = (HBITMAP)SelectObject( hMemDC, hBitmap );
    
          //fill the memory object with the pattern 
          FillRect( hMemDC, &rc, hBrush );
          //BitBlt the source image over the pattern using SRCAND so that 
          //only the "on" destination pixels are transferred. 
          BitBlt( hMemDC, rc.left, rc.top, rc.right, rc.bottom, hDC,
                  lpRect->left, lpRect->top, SRCAND );
   
          oldfgcolor = SetTextColor( hDC, 
                GetSysColor( (dark)?(COLOR_BTNSHADOW):(COLOR_HIGHLIGHT) ));
          oldbgcolor = SetBkColor( hDC, RGB(0,0,0) ); 
          oldbkmode  = SetBkMode( hDC, OPAQUE );
        
          //hBrush = SelectObject( hDC, hBrush );
          FillRect( hDC, lpRect, hBrush );
          BitBlt( hDC, lpRect->left, lpRect->top, nWidth, nHeight,
                  hMemDC, 0, 0, SRCPAINT );
          //hBrush = SelectObject( hDC, hBrush ); 

          SetBkMode( hDC, oldbkmode );
          SetBkColor( hDC, oldbgcolor );
          SetTextColor( hDC, oldfgcolor );

          hBitmap = (HBITMAP)SelectObject( hMemDC, hBitmap );
          DeleteObject( hBitmap );
        } /* if (hBitmap) */
        DeleteObject(hBrush);  
      } /* if (hBrush) */
      DeleteObject(hBrushBitmap);
    } /* if (hBrushBitmap) */
    DeleteDC( hMemDC );
  } /* if (hMemDC) */
  return;
}

#define GRAPH_DIALOG	    2
#define IDC_PROJLIST	  101
#define IDC_CURRATE	  102
#define IDC_GRAPH	  103
#define IDC_CRUNCHCOUNT	  104
#define IDC_AMP_FRAME     105
#define IDC_AMP           106
#define IDC_FREQ_FRAME    107
#define IDC_FREQ          108
#define IDC_LOG	          109
#define IDC_BUFIN_PKTS    110
#define IDC_BUFIN_SWU     111
#define IDC_BUFIN_TIME    112
#define IDC_BUFOUT_SWU	  113
#define IDC_BUFOUT_PKTS	  114
#define IDC_SUM_TIME      115
#define IDC_SUM_RATE      116
#define IDC_SUM_SWU       117
#define IDC_SUM_PKTS      118
#define IDC_SHOWAVG       119
#define IDC_SHOWNOISE     120

extern int LogGetContestLiveRate(unsigned int contest_i,
                                 u32 *ratehiP, u32 *rateloP,
                                 u32 *walltime_hiP, u32 *walltime_loP,
                                 u32 *coretime_hiP, u32 *coretime_loP);

DWORD CALLBACK __w16GraphView( HWND dialog, UINT msg, WORD wParam, LONG lParam )
{
  struct dlgdata
  {
    HWND hOwner;
    COLORREF clrBG, clrFG;
    HBRUSH hBGBrush;
    W16CONP console;
    int show_noise;
    int show_avg;
    int mode_pending;
    unsigned long effmax;
    int numcrunch_total; 
    struct {
      unsigned long rate[120]; /* 120 seconds */
      int numcrunchers;
      struct 
      {
        long threshold;
        int thresh_in_swu;
        long blk_count;
        long swu_count;
        time_t till_completion;
      } buffers[2];
      u32 last_ratelo;
      /* sizeof avgrate array => tray refresh interval in secs */
      struct {u32 hi,lo;} avgrate[5]; 
      unsigned int avgrate_count;
    } cdata[CONTEST_COUNT];
    unsigned int cont_sel;
    int cont_sel_uncertain;
    int cont_sel_explicit;
    DWORD cont_sel_exptime;
    DWORD last_rate_disp;
    int last_in_tray;
    int timer_cont_sel;
    UINT timer;
    struct      /* for positional accuracy, 'scale' members are always */
    {           /* in the range 0..100 inclusive. */
      int amp;  /* 'amp' eval range: 0.00...1.00 */
      int freq; /* 'freq' eval range: 1-5 secs */
      int mouse_x; /* mouse x coord relative to the left edge of the thumb */
    } scale;
    HFONT hFont;
    struct
    {
      int rowcount; /* shortcut SendMessage(LB_GETCOUNT) */
      int font_height;
      int tab_width;
    } logwindow;
    HDC hMemDC;
    HBITMAP hMemBitmap;
  } *dd;
  char buffer[128]; /* NO LESS THAN 128!! */
  unsigned int cont_i;
  HDC hDC; RECT rect; 
  HMENU hmenu;
  HWND hwnd;

  if (msg == WM_COMMAND && wParam == WMCMD_EVENT)
  {
    struct WMCMD_EVENT_DATA *evdata = (struct WMCMD_EVENT_DATA *)lParam;
    if (!lParam)
      return 0;
    dd = (struct dlgdata *)GetWindowLong(dialog, DWL_USER);
    if (!dd)
      return 0;
    if (evdata->id == CLIEVENT_BUFFER_UPDATEBEGIN ||
        evdata->id == CLIEVENT_BUFFER_UPDATEEND)
    {
      /* clients running on Win16 won't be getting WM_TIMER while */
      /* networking is in progress, so gray the graph in that period. */
      #if (CLIENT_OS == OS_WIN32)
      if (winGetVersion() < 400)
      #endif   
      {  
        dd->mode_pending = (evdata->id == CLIEVENT_BUFFER_UPDATEBEGIN);
        InvalidateRect(GetDlgItem(dialog,IDC_GRAPH),0,0);
      }
      return 0;
    }
    if (evdata->id != CLIEVENT_CLIENT_RUNIDLE)
      return 0;
    if (dd->timer)
      return 0;
    msg = WM_TIMER;
  }  

  switch (msg)
  {
    case WM_INITDIALOG:
    {
      dd = (struct dlgdata *)malloc(sizeof(struct dlgdata));
      if (dd)
      {
        SetWindowLong(dialog, DWL_USER, (LONG)dd);
        if (dd != ((struct dlgdata *)GetWindowLong(dialog, DWL_USER)))
        {
          free((void *)dd);
          dd = (struct dlgdata *)0;
        }
      }
      if (!dd)
      {
        SetWindowLong(dialog,DWL_USER,0);
        EndDialog( dialog, TRUE );
        return( TRUE );
      }
      TRACE_DLG((+1,"WM_INITDIALOG, dd=%p\n",dd));
      memset(dd, 0, sizeof(struct dlgdata));

      dd->cont_sel = 0;
      dd->hOwner = hwnd = (HWND)lParam;
      dd->console = __win16GetHwndConsole( hwnd );
      dd->cont_sel_uncertain = 1;
      dd->cont_sel_explicit = -1;
      dd->timer_cont_sel = -1; 
      dd->cont_sel = 0; 
      dd->effmax = 1000;
      dd->hFont = __w16FixupDlgFont(dialog, NULL);
      if (GetNumberOfDetectedProcessors() > 1)
        dd->effmax *= GetNumberOfDetectedProcessors();
      if (dd->console) /* we need to do this so that menu reflects the view */
        dd->console->rate_view.hwnd = dialog; /* the caller won't have set it yet */
      dd->clrBG = GetSysColor(COLOR_BTNFACE);
      dd->clrFG = GetSysColor(COLOR_BTNTEXT);
      dd->hBGBrush = CreateSolidBrush(dd->clrBG); /* don't use GetSysColorBrush()!! */

      dd->numcrunch_total = ProblemCountLoaded(-1);
      dd->scale.amp = GetDCTIProfileInt( "scale", "ry", 0 );
      if (dd->scale.amp < 0 || dd->scale.amp > 100) 
        dd->scale.amp = 0;
      dd->scale.freq = GetDCTIProfileInt( "scale", "rx", 0 );
      if (dd->scale.freq < 1 || dd->scale.freq > 100) 
        dd->scale.freq = 1;
      dd->show_noise = !GetDCTIProfileInt( "scale", "squelch", 1 );
      dd->show_avg = GetDCTIProfileInt( "scale", "savg", 0 );
 
      if (!GetParent(dialog)) /* not a child of console */
      {
        hmenu = GetMenu(dialog);
        SetMenu(dialog,__w16WindowConstructMenu(dd->console,dialog,WM_INITMENU,0));
        __w16WindowDestroyMenu(hmenu);
        if (dd->hOwner) /* this is the console */
        {
          hwnd = dd->hOwner; 
          #if defined(WM_GETICON)
          /* although dialog boxes don't have an icon, do this so that */
          /* alt-tab will show something other than the microsoft flag */
          SendMessage(dialog,WM_SETICON,0,SendMessage(hwnd,WM_GETICON,0,0));
          SendMessage(dialog,WM_SETICON,1,SendMessage(hwnd,WM_GETICON,1,0));
          #endif
          GetWindowText(hwnd,buffer,sizeof(buffer));
          strcat(buffer," "); /* hide from other clients */
          SetWindowText(dialog,buffer); 
        }
      }

      __w16Set_BS_OWNERDRAW(GetDlgItem(dialog,IDC_GRAPH));
      __w16Set_BS_OWNERDRAW(GetDlgItem(dialog,IDC_CRUNCHCOUNT));
      __w16Set_BS_OWNERDRAW(GetDlgItem(dialog,IDC_AMP));
      __w16Set_BS_OWNERDRAW(GetDlgItem(dialog,IDC_FREQ));

      hwnd = GetDlgItem( dialog, IDC_LOG );
      if (hwnd)
      {
        SetWindowLong(hwnd,GWL_STYLE, LBS_OWNERDRAWFIXED|LBS_EXTENDEDSEL|
           (GetWindowLong(hwnd,GWL_STYLE)&~(LBS_SORT|LBS_HASSTRINGS)) );
        ShowScrollBar( hwnd, SB_VERT, TRUE );
        PostMessage( dialog, WM_COMMAND, WMCMD_REFRESHVIEW, 0 );
        //WMCMD_REFRESHVIEW will invalidate the window
      }

      hwnd = GetDlgItem( dialog, IDC_PROJLIST );
      if (hwnd)
      {
        //SetWindowLong(hwnd,GWL_STYLE,GetWindowLong(hwnd,GWL_STYLE)|LBS_NOTIFY);
        for (cont_i=0; cont_i<CONTEST_COUNT;cont_i++)
        {
          SendMessage(hwnd,LB_INSERTSTRING, (WORD)cont_i,
                      (LPARAM)CliGetContestNameFromID(cont_i));
          if (dd->cont_sel_uncertain && ProblemCountLoaded(cont_i) > 0)
          {
            dd->cont_sel = (int)cont_i;
            dd->cont_sel_uncertain = 0;
          } 
        } 
        SendMessage(hwnd,LB_SETCURSEL,
               (WPARAM)((dd->cont_sel_uncertain)?(-1):(dd->cont_sel)), 0);
        //ShowScrollBar(hwnd, SB_VERT, TRUE);
        SetFocus(hwnd);  
      }
      hwnd = GetDlgItem(dialog, IDC_SHOWNOISE);
      if (hwnd)
      {     
        SetWindowText(hwnd,"Plot Impurity");
        SendMessage(hwnd,BM_SETCHECK,dd->show_noise,0);
        //EnableWindow(hwnd,  (dd->scale.amp >= 5));
      }
      hwnd = GetDlgItem(dialog, IDC_SHOWAVG);
      if (hwnd)
      {     
        SetWindowText(hwnd,"Plot Average");
        SendMessage(hwnd,BM_SETCHECK,dd->show_avg,0);
      }
      /* NYI */
      hwnd = GetDlgItem(dialog, IDC_FREQ);
      if (hwnd)
      {
        ShowWindow(hwnd, SW_HIDE);
      }
      hwnd = GetDlgItem(dialog, IDC_FREQ_FRAME);
      if (hwnd)
      {
        ShowWindow(hwnd, SW_HIDE);
      }
      if (winGetVersion() >= 400) /* not win3x or win32s */
      {                           /* where timers are scarce */
        dd->timer = SetTimer(dialog,1,1000,NULL);
      }
      /* there is a placement bug for the AMP button on win3.1 */
      /* when ctl3d.dll is globally loaded */
      hwnd = GetDlgItem(dialog,IDC_AMP_FRAME);
      if (hwnd)
      {
        GetClientRect(hwnd,&rect);
        MapWindowPoints(hwnd,dialog,(POINT *)&rect,2);
        hwnd = GetDlgItem(dialog,IDC_AMP);
        if (hwnd)
        {
          rect.left += 4;
          rect.top += 8+4+4, /* 8 for title */
          rect.bottom -= 4;
          rect.right -= 4;
          MoveWindow(hwnd,rect.left,rect.top,(rect.right-rect.left),
                          (rect.bottom-rect.top),FALSE);   
        } 
      }
      TRACE_DLG((-1,"WM_INITDIALOG\n"));
      return FALSE; /* we changed focus */
    }
    case WM_KILLFOCUS: 
    {
      if (IsChild(dialog, ((HWND) wParam))) /* a control is receiving focus? */
      {
        hwnd = GetDlgItem(dialog, IDC_PROJLIST);
        if (hwnd && hwnd != ((HWND) wParam))
        {
          if (((HWND) wParam) != GetDlgItem(dialog, IDC_LOG))
            SetFocus(hwnd);
        }
      }
      return FALSE;
    }
    #if !defined(WM_CTLCOLOR) /* only defined in windowsx.h */
    #define WM_CTLCOLOR         0x0019
    #endif
    case WM_CTLCOLOR:
    {
      if (HIWORD(lParam) != CTLCOLOR_BTN) /* includes BS_GROUPBOX */
      {	
        dd = (struct dlgdata *)GetWindowLong(dialog, DWL_USER);
        if (dd)
        {
          hDC = (HDC) wParam;
          SetTextColor(hDC, dd->clrFG);
          SetBkColor(hDC, dd->clrBG);
          SetBkMode( hDC, TRANSPARENT);
          SetBrushOrgEx( hDC, 0, 0, NULL );
          SelectObject( hDC, dd->hBGBrush );
          return (LRESULT)dd->hBGBrush;
        }
      }
      return FALSE;
    }
    case WM_TIMER:
    {
      dd = (struct dlgdata *)GetWindowLong(dialog, DWL_USER);
      TRACE_DLG((+1,"WM_TIMER dd=%p\n",dd));
      if (dd)
      {
        int crunch_count_change = 0, buffers_changed = 0;
        int rate_cont_i = -1; u32 rate_ratehi = 0, rate_ratelo = 0;
        DWORD tick_count = GetTickCount();

        if (dd->cont_sel_uncertain) /* no contest selected yet */
        { 
          for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
          {
            if (ProblemCountLoaded(cont_i) > 0)
            { 
              dd->cont_sel = cont_i;
              dd->cont_sel_uncertain = 0;
              SendDlgItemMessage(dialog, IDC_PROJLIST,
                                 LB_SETCURSEL,(WPARAM)cont_i,0);
              break;
            }
          }
        }
        for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
        {
          u32 ratehi,ratelo,wtimehi,wtimelo,ctimehi,ctimelo; 
          unsigned long efficiency = 0;
          int numcrunchers;
          int curpos;

          numcrunchers = LogGetContestLiveRate(cont_i, &ratehi, &ratelo,
                   &wtimehi, &wtimelo, &ctimehi, &ctimelo);
          if (numcrunchers < 1)
          {
            ratehi = ratelo = 0;
            numcrunchers = 0;
          }
          else
          {
            wtimelo = (wtimelo / 1000)+(wtimehi * 1000);
            ctimelo = ((ctimelo+499) / 1000)+(ctimehi * 1000);
            if (wtimelo)
            {
              efficiency = (((unsigned long)ctimelo) * 1000ul)/wtimelo;
              if (efficiency > dd->effmax)
                efficiency = dd->effmax;
            } 
          }  
          if (efficiency == 0)
            efficiency = 1;

          /* +++++++++++++++++++++++ */  

          if (cont_i == dd->cont_sel)
          {
            int out_buffer_changed = 0;
            rate_cont_i = cont_i;
            rate_ratehi = ratehi;
            rate_ratelo = ratelo;

            if (numcrunchers == 0 && (dd->cont_sel_explicit != ((int)cont_i) 
              || (dd->cont_sel_exptime/10000) != (tick_count/10000)))
            {
              /* contest switched. Switch to another in the display */
              dd->cont_sel_uncertain = 1; /* switch to another */
            }
            dd->numcrunch_total = ProblemCountLoaded(-1);
            if (numcrunchers != dd->cdata[cont_i].numcrunchers)
              crunch_count_change = 1;
            if (cont_i != ((unsigned int)dd->timer_cont_sel))
              buffers_changed  = 1;

            if (!IsProblemLoadPermitted(-1, cont_i))
            {
              ; /* nothing */
            }
            else if (buffers_changed || crunch_count_change || 
                     (((tick_count+499)/1000)%4) == 0) 
            {
              int sel_buf;
              for (sel_buf = 0; sel_buf < 2; sel_buf++)
              {
                long threshold, blk_count, swu_count; 
                unsigned int till_completion;
                int thresh_in_swu; 
                if (ProbfillGetBufferCounts( cont_i, sel_buf,
                             &threshold, &thresh_in_swu,
                             &blk_count, &swu_count, &till_completion )>=0)
                {                               
                  if (buffers_changed 
                   || dd->cdata[cont_i].buffers[sel_buf].threshold != threshold
                   || dd->cdata[cont_i].buffers[sel_buf].thresh_in_swu != thresh_in_swu
                   || dd->cdata[cont_i].buffers[sel_buf].blk_count != blk_count
                   || dd->cdata[cont_i].buffers[sel_buf].swu_count != swu_count
                   || dd->cdata[cont_i].buffers[sel_buf].till_completion != (int)till_completion)
                  {
                     dd->cdata[cont_i].buffers[sel_buf].threshold = threshold;
                     dd->cdata[cont_i].buffers[sel_buf].thresh_in_swu = thresh_in_swu;
                     dd->cdata[cont_i].buffers[sel_buf].blk_count = blk_count;
                     dd->cdata[cont_i].buffers[sel_buf].swu_count = swu_count;
                     dd->cdata[cont_i].buffers[sel_buf].till_completion = till_completion;
                     buffers_changed = 1;
                     if (sel_buf == 1)
                       out_buffer_changed = 1;
                  }
                }
              } /* for sel_buf ... */
            } /* need buffer level check */ 

            if (buffers_changed || ratelo > dd->cdata[cont_i].last_ratelo)
            {
              long ll;
              
              if (buffers_changed)
              {
                ll = dd->cdata[cont_i].buffers[1].swu_count;
                sprintf(buffer,"%d.%02d", ll/100, ll%100);
                SetDlgItemText(dialog,IDC_BUFOUT_SWU, buffer);
                SetDlgItemInt(dialog,IDC_BUFOUT_PKTS,
                     (UINT)(dd->cdata[cont_i].buffers[1].blk_count), FALSE);
  
                SetDlgItemInt(dialog,IDC_BUFIN_PKTS,
                     (UINT)(dd->cdata[cont_i].buffers[0].blk_count), FALSE);
        
                buffer[0] = '\0';
                if (dd->cdata[cont_i].buffers[0].swu_count == 0 &&
                    dd->cdata[cont_i].buffers[0].blk_count != 0)
                {
                  SetDlgItemText(dialog,IDC_BUFIN_SWU, "-.--");
                }
                else
                {
                  ll = dd->cdata[cont_i].buffers[0].swu_count;
                  sprintf( buffer, "%d.%02d", ll/100, ll%100);
                  SetDlgItemText(dialog, IDC_BUFIN_SWU, buffer);
                }
              }

              buffer[0] = '\0';
              if (dd->cdata[cont_i].buffers[0].swu_count == 0)
              {
                if (dd->cdata[cont_i].buffers[0].blk_count == 0)
                  strcpy(buffer, "0.00:00:00");
                /* otherwise fallthrough to "-.--:--:--" */
              }
              else
              {
                ll = dd->cdata[cont_i].buffers[0].till_completion;
                if (!ll && ratehi == 0 && ratelo != 0 &&
                   ((dd->cdata[cont_i].buffers[0].swu_count)%100) == 0)
                {
                  ll = ((1+(1ul << 28))/ratelo); /* secs per *work* unit */
                  ll *= ((dd->cdata[cont_i].buffers[0].swu_count)/100);
                }
                if (ll) /* otherwise, not available (yet) */
                {
                  int days = (ll / 86400UL);
                  if (days >= 0 && days <= 365)
                  {
                    sprintf( buffer,  "%d.%02d:%02d:%02d", days,
                              (int) ((ll % 86400L) / 3600UL), 
                              (int) ((ll % 3600UL)/60),
                              (int) (ll % 60) );
                  }
                }
              }
              if (!buffer[0])
                strcpy(buffer, "-.--:--:--" );
              SetDlgItemText(dialog,IDC_BUFIN_TIME, buffer);
              dd->cdata[cont_i].last_ratelo  = ratelo;

              if (out_buffer_changed)
              {
                u32 iterhi, iterlo;
                unsigned int packets, swucount;
                struct timeval ttime;

                if (CliGetContestInfoSummaryData( cont_i, 
                     &packets, &iterhi, &iterlo,
                     &ttime, &swucount ) == 0)
                {
                  ProblemComputeRate( cont_i, ttime.tv_sec, ttime.tv_usec, 
                            iterhi, iterlo, 0, 0, buffer, sizeof(buffer) );
                  char *p = strchr(buffer,' ');
                  if (p) *p = '\0';
                  SetDlgItemText(dialog,IDC_SUM_RATE,buffer);
                  SetDlgItemInt(dialog,IDC_SUM_PKTS,(UINT)packets,FALSE);
                  sprintf(buffer, "%u.%02u", swucount/100, swucount%100);
                  SetDlgItemText(dialog,IDC_SUM_SWU,buffer);
                  ll = ttime.tv_sec;
                  sprintf( buffer,  "%d.%02d:%02d:%02d", (ll / 86400UL),
                           (int) ((ll % 86400L) / 3600UL), 
                           (int) ((ll % 3600UL)/60),
                           (int) (ll % 60) );
                  SetDlgItemText(dialog,IDC_SUM_TIME,buffer);
                }
              } /* if (out_buffer_changed) */
            } /* if (buffers_changed) */
          } /* if (cont_i == dd->cont_sel) */
          /* +++++++++++++++++++++++ */

          /* must come after (cont_i == dd->cont_sel) section */
          dd->cdata[cont_i].avgrate_count++;
          if (dd->cdata[cont_i].avgrate_count > 
                   (sizeof(dd->cdata[cont_i].avgrate)/
                   sizeof(dd->cdata[cont_i].avgrate[0])))
          {
            dd->cdata[cont_i].avgrate_count = 
                   (sizeof(dd->cdata[cont_i].avgrate)/
                   sizeof(dd->cdata[cont_i].avgrate[0]));
            memmove( &(dd->cdata[cont_i].avgrate[0]),
                     &(dd->cdata[cont_i].avgrate[1]),
                     sizeof(dd->cdata[cont_i].avgrate)-
                     sizeof(dd->cdata[cont_i].avgrate[0]));
          }
          curpos = dd->cdata[cont_i].avgrate_count-1;
          dd->cdata[cont_i].avgrate[curpos].hi = ratehi;
          dd->cdata[cont_i].avgrate[curpos].lo = ratelo;

          memmove( &(dd->cdata[cont_i].rate[0]),
                   &(dd->cdata[cont_i].rate[1]),
                   sizeof(dd->cdata[cont_i].rate)-
                   sizeof(dd->cdata[cont_i].rate[0]));
          curpos = (sizeof(dd->cdata[cont_i].rate)/
                   sizeof(dd->cdata[cont_i].rate[0]))-1;
          dd->cdata[cont_i].rate[curpos] = efficiency;
          dd->cdata[cont_i].numcrunchers = numcrunchers;

        } /* for (cont_i = 0; ... ) */

        dd->timer_cont_sel = dd->cont_sel; 

        /* +++++++++++ */
        {
          int avg_interval = (sizeof(dd->cdata[cont_i].avgrate)/
                              sizeof(dd->cdata[cont_i].avgrate[0]));
          int in_tray = 0, rate_disp_due = 0;

          if (dd->hOwner && (winGetVersion()%2000)>=400 && 
               !IsWindowVisible(dd->hOwner) && IsIconic(dd->hOwner))
          {
            in_tray = 1;
          }
          if (buffers_changed || crunch_count_change 
             #if 1 /* update every second if not in tray */
             || !in_tray 
             #endif
             || dd->last_in_tray != in_tray 
             || dd->last_rate_disp != (tick_count/(avg_interval*1000)) )
          {
            rate_disp_due = 1;
            dd->last_in_tray = in_tray;
            dd->last_rate_disp = (tick_count/(avg_interval*1000));
          }
 
          buffer[0] = '\0';
          if (in_tray && rate_disp_due && CheckPauseRequestTrigger())
          {
            if (!GetWindowText(dd->hOwner,buffer,sizeof(buffer)-30))
              buffer[0] = '\0';
            strcat(buffer," (paused)");  
          }
          else if (rate_disp_due && 
              (rate_cont_i >= 0 && rate_cont_i < CONTEST_COUNT))
          {
            unsigned int len;
            char avg_ratebuf[sizeof("n,nnn,nnn,nnn xxxxx\0")];
            {
              u32 avg_ratehi = 0, avg_ratelo = 0;
              unsigned __int64 tot;  
              for (len = 0; len < dd->cdata[rate_cont_i].avgrate_count;
                   len++)
              {
                avg_ratelo += dd->cdata[rate_cont_i].avgrate[len].lo;
                avg_ratehi += dd->cdata[rate_cont_i].avgrate[len].hi;
                if (avg_ratelo < dd->cdata[rate_cont_i].avgrate[len].lo)
                  avg_ratehi++;
              }
              tot = (((unsigned __int64)avg_ratehi)<<32)+
                     ((unsigned __int64)avg_ratelo);
              tot /= dd->cdata[rate_cont_i].avgrate_count;
              avg_ratehi = (u32)(tot>>32);
              avg_ratelo = (u32)(tot&0xffffffff);
              ProblemComputeRate( rate_cont_i, 0, 0, avg_ratehi,avg_ratelo, 0, 0,
                                    avg_ratebuf, sizeof(avg_ratebuf) );
            }
 
            if (in_tray)
            {
              len = sprintf(buffer,"%s: %s/sec\n-in: %lu packets",
                            CliGetContestNameFromID(rate_cont_i),
                            avg_ratebuf,
                            dd->cdata[rate_cont_i].buffers[0].blk_count);
              if (dd->cdata[rate_cont_i].buffers[0].swu_count == 0 &&
                  dd->cdata[rate_cont_i].buffers[0].blk_count != 0)
              {
                //len += sprintf(&buffer[len], " (-.-- units)");
              }
              else
              {
                len += sprintf(&buffer[len], " (%lu.%02lu units)",
                        (dd->cdata[rate_cont_i].buffers[0].swu_count / 100),
                        (dd->cdata[rate_cont_i].buffers[0].swu_count % 100));
              }
              buffer[len] = '\0';
            }  
            else
            {
              #if 0
              len = strlen(strcpy(buffer, "Core Throughput: "));
              ProblemComputeRate( rate_cont_i, 0, 0, rate_ratehi, rate_ratelo, 0, 0,
                                    &buffer[len], sizeof(avg_ratebuf) );
              strcat(buffer,"/sec");
              #else
              len = sprintf(buffer, "Core Throughput: %s/sec", avg_ratebuf);
              if (avg_interval == dd->cdata[rate_cont_i].avgrate_count)
                sprintf(&buffer[len]," (%d sec sliding average)", avg_interval );
              #endif
            }
          }

          if (in_tray)
          { 
            if (rate_disp_due)
            {
              /* buffer[0] may be '\0' here */
              __DoTrayStuff(dd->hOwner, +1, buffer, "ScopeView" );
            }
          }
          else
          {
            if (rate_disp_due)
            {
              if (buffer[0] == '\0')
                strcpy(buffer, "Core Throughput:" );
              SetDlgItemText(dialog, IDC_CURRATE, buffer);
            }
            if (buffers_changed)
            {
              InvalidateRect(dialog,0,0);
            } 
            else /* invalidate individually */
            {
              hwnd = GetDlgItem(dialog,IDC_GRAPH);
              if (hwnd) 
              {
                if (__IsViewable(hwnd))
                {
                  InvalidateRect(hwnd,0,0);
                }
              }
              if (crunch_count_change)
              {
                hwnd = GetDlgItem(dialog,IDC_CRUNCHCOUNT);
                if (hwnd && __IsViewable(hwnd))
                  InvalidateRect(hwnd,0,0);
              }
            } 
          } /* in tray or not */
        } /* +++++++++++ */
      } /* if (dd) */
      TRACE_DLG((-1,"WM_TIMER\n"));
      return FALSE;
    }
    #if 0 /* done in drawlist now */
    case WM_MEASUREITEM: /* sent *before* WM_INITDIALOG! */
    {
      if (wParam == IDC_LOG)
      {
        #if defined(__WINDOWS_386__) /* need 16:32 pointer */
        MEASUREITEMSTRUCT far* lpmi = (MEASUREITEMSTRUCT far *)MK_FP32((void *)lParam);
        #else
        LPMEASUREITEMSTRUCT lpmi = (LPMEASUREITEMSTRUCT)lParam;
        #endif
        if (lpmi)
        {
          GetClientRect(GetDlgItem(dialog,IDC_LOG),&rect);
          lpmi->itemHeight = 8+5;
          lpmi->itemWidth = rect.right-rect.left;
        }
      }
      return FALSE;
    }
    #endif
    case WM_SETCURSOR: /* we don't get WM_LBUTTONDOWN and WM_PARENTNOTIFY */
                       /* doesn't seem very reliable. */
    case WM_MOUSEMOVE: /* use MOUSEMOVE instead of SETCURSOR+MOUSEMOVE because */
                       /* MOUSEMOVE is cumulative (not called so often) */
    {
      dd = (struct dlgdata *)GetWindowLong(dialog, DWL_USER);
      TRACE_DLG((+1,"WM_MOUSEMOVE+MK_LBUTTON, dd=%p\n",dd));
      if (dd)
      {
        if ((msg == WM_MOUSEMOVE && (wParam & MK_LBUTTON)!=0) ||
            (msg == WM_SETCURSOR && HIWORD(lParam) == WM_LBUTTONDOWN))
        { 
          POINT pt;
          int id = IDC_AMP;
          GetCursorPos(&pt);
          
          while (id != -1)
          {
            hwnd = GetDlgItem(dialog, id);
            if (hwnd)
            {
              if (!IsWindowVisible(hwnd) ||
                  IsWindowEnabled(hwnd)) /* note: this is reversed */
               hwnd = NULL;         /* (don't get messages when _enabled_) */
            }
            if (hwnd)
            {
              GetWindowRect(hwnd,&rect);
              rect.left+=2; rect.right-=2;
              rect.top+=2; rect.bottom-=2;
              if (pt.x >= rect.left && pt.x <= (rect.right-2) &&
                  pt.y >= rect.top && pt.y <= rect.bottom)
              {
                RECT thumb;
                int slide_range = __w16GetSliderDims(&rect, &thumb);
                if (slide_range > 0)
                {
                  int old_pos, new_pos;
                  if (id == IDC_AMP)
                    old_pos = dd->scale.amp;
                  else
                    old_pos = dd->scale.freq;
                  new_pos = ((pt.x - rect.left)*100)/slide_range;
                  if (new_pos > 100)
                    new_pos = 100;
                  if (old_pos != new_pos)
                  {
                    int toff = (old_pos * slide_range)/100;
                    thumb.left += toff;
                    thumb.right += toff;
                    if (msg == WM_MOUSEMOVE) /* user moved the thumb */
                    {
                      toff = pt.x - dd->scale.mouse_x;
                      if (toff < rect.left)
                        new_pos = 0;   
                      else
                        new_pos = ((toff - rect.left)*100)/slide_range;
                    }
                    else if (pt.x >= thumb.left && pt.x <= thumb.right)
                    {                  /* left-clicked _on_ the old thumb */
                      dd->scale.mouse_x = pt.x - thumb.left;
                      new_pos = old_pos; /* don't move the thumb */
                    }
                    else
                    {
                      thumb.left -= toff;
                      thumb.right -= toff;
                      toff = pt.x - (thumb.right-thumb.left)/2;
                      if (toff < rect.left)
                        new_pos = 0;   
                      else
                        new_pos = ((toff - rect.left)*100)/slide_range;
                      toff = (new_pos * slide_range)/100;
                      thumb.left += toff;
                      dd->scale.mouse_x = pt.x - thumb.left;
                    }  
                  }
                  if (new_pos > 100)
                    new_pos = 100;
                  if (old_pos != new_pos)
                  {  
                    HWND hOther; 
                    if (id == IDC_AMP)
                    {
                      dd->scale.amp = new_pos; 
                      #if 0
                      if ((new_pos >= 5 && old_pos < 5) || 
                          (old_pos >= 5 && new_pos < 5))  
                      {
                        hOther = GetDlgItem(dialog, IDC_SHOWNOISE);
                        if (hOther) 
                          EnableWindow(hOther, (new_pos >= 5));
                      }
                      #endif
                    }
                    else
                    {
                      dd->scale.freq = new_pos;
                    }
                    InvalidateRect(hwnd,NULL,FALSE);
                    hOther = GetDlgItem(dialog,IDC_GRAPH);
                    if (hOther)
                      InvalidateRect(hOther,NULL,FALSE);
                  } /* if position changed */
                } /* if (slide_range > 0) */
                break;  
              } /* if (PointInRect()) */
              hwnd = NULL;
            }
            if (id == IDC_AMP)
              id = IDC_FREQ;
            else /* if (id == IDC_FREQ) */
              id = -1;
          }
        } /* if ((wParam & MK_LBUTTON)!=0) */
      } /* if (dd) */
      TRACE_DLG((-1,"WM_MOUSEMOVE+MK_LBUTTON\n"));
      return FALSE;
    }
    case WM_DRAWITEM:
    {
      dd = (struct dlgdata *)GetWindowLong(dialog, DWL_USER);
      TRACE_DLG((+1,"WM_DRAWITEM, dd=%p\n",dd));
      if (dd)
      {
        #if defined(__WINDOWS_386__) /* need 16:32 pointer */
        DRAWITEMSTRUCT far *lpdis = (DRAWITEMSTRUCT far *)MK_FP32((void *)lParam);
        #else
        LPDRAWITEMSTRUCT lpdis = (LPDRAWITEMSTRUCT)lParam;
        #endif
        int id = wParam;
        hwnd = NULL;
        cont_i = 0;

        if (lpdis)
        {
          hwnd = GetDlgItem(dialog,id);
        }
        if (id == IDC_GRAPH || id == IDC_CRUNCHCOUNT)
        {
          if (hwnd)
          {
            cont_i = dd->cont_sel;
            if (cont_i >= CONTEST_COUNT)
              hwnd = NULL;
          }
        }
        else if (id == IDC_LOG && hwnd)
        {
          if (dd->logwindow.font_height == 0)
            PostMessage(dialog,WM_COMMAND,WMCMD_REFRESHVIEW,0); //POST!!
          else
            __w16DrawList( hwnd, dd->console, dd->logwindow.font_height, 
                           dd->logwindow.tab_width, lpdis); 
          hwnd = NULL; /* don't fall through */
        }
        else if (id == IDC_FREQ || id == IDC_AMP)
        {
          ;
        }
        if (hwnd)
        {
          int bmp_height, bmp_width;
          POINT polys[(sizeof(dd->cdata[0].rate)/
                      sizeof(dd->cdata[0].rate[0]))];
          HPEN hPen; HBRUSH hBrush;

          GetClientRect(hwnd,&rect);
          bmp_height = (rect.bottom - rect.top)+1;
          bmp_width  = (rect.right - rect.left)+1;
       
          if (!dd->hMemDC && id == IDC_GRAPH) /* need the bigger bitmap */
          { 
            dd->hMemDC = CreateCompatibleDC(lpdis->hDC);
            if (dd->hMemDC)
            {
              dd->hMemBitmap = CreateCompatibleBitmap(lpdis->hDC, bmp_width, bmp_height);
              if (dd->hMemBitmap)
                dd->hMemBitmap = (HBITMAP)SelectObject(dd->hMemDC, dd->hMemBitmap);
              else 
              {
                DeleteDC(dd->hMemDC);
                dd->hMemDC = NULL;
              }
            }
          }
          hDC = dd->hMemDC;
          if (!hDC)
          {
            hDC = lpdis->hDC;
          }

          __w16DrawRecessedFrame( hDC, &rect, (HBRUSH)GetStockObject(BLACK_BRUSH) );

          if (id == IDC_AMP || id == IDC_FREQ)
          {
            int pos, xrange;
            rect.left+=2; rect.top += 2;
            rect.bottom-=2; rect.right-=2;

            if (id == IDC_AMP)
            {
              pos = dd->scale.amp;
              sprintf(buffer,"Amp: %d.%02d x", pos/100, pos%100);
              SetDlgItemText(dialog, IDC_AMP_FRAME, buffer);
            }
            else
            {
              pos = dd->scale.freq;
              sprintf(buffer,"Freq: %d sec%s", pos, ((pos==1)?(""):("s")));
              SetDlgItemText(dialog, IDC_FREQ_FRAME, buffer);
            }
      
            xrange = __w16GetSliderDims(&rect /*IN:area*/, &rect/*OUT:knob*/);
            pos = (pos * xrange)/100;

            hBrush = CreateSolidBrush(GetSysColor(COLOR_BTNFACE));
            hBrush = (HBRUSH)SelectObject(hDC, hBrush);
            hPen = CreatePen(PS_SOLID,1,GetSysColor(COLOR_BTNHIGHLIGHT));
            hPen = (HPEN)SelectObject(hDC, hPen);
            Rectangle(hDC,rect.left+pos,rect.top,rect.right+pos+1,rect.bottom+1);
            DeleteObject(SelectObject(hDC,hBrush));
            DeleteObject(SelectObject(hDC,hPen));
            polys[0].x = rect.right+pos;    polys[0].y = rect.top;
            polys[1].x = rect.right+pos;    polys[1].y = rect.bottom+1;
            polys[2].x = (rect.left-1)+pos; polys[2].y = rect.bottom+1;
            hPen = (HPEN)SelectObject(hDC,GetStockObject(BLACK_PEN));
            Polyline(hDC,&polys[0],3);
            SelectObject(hDC, hPen);
            polys[0].x--; polys[0].y++;
            polys[1].x--; polys[1].y--;
            polys[2].x++; polys[2].y--;
            hPen = CreatePen(PS_SOLID,1,GetSysColor(COLOR_BTNSHADOW));
            hPen = (HPEN)SelectObject(hDC, hPen);
            Polyline(hDC,&polys[0],3);
            DeleteObject(SelectObject(hDC,hPen));

            if (IsWindowEnabled(hwnd))   /* note: this is reversed */
            {                  /* (don't get mouse messages when _enabled_) */
              __ShadeRect(hDC, &rect, TRUE);
            }
          }
          else if (id == IDC_CRUNCHCOUNT)
          {
            LOGFONT lf; 
            HFONT hFont = (HFONT)GetStockObject(ANSI_VAR_FONT);

            rect.top += 2;
            rect.bottom -= 3;
            rect.left += 2;
            rect.right -= 2;

            if (GetObject(hFont,sizeof(lf), &lf))
            {
              lf.lfHeight = (rect.bottom-rect.top)+1;
              lf.lfWidth = 0;
              lf.lfFaceName[0] = '\0';
              hFont = CreateFontIndirect(&lf);

              if (hFont)
              {
                SIZE ex; int len; const char *fmt;
                COLORREF fg = SetTextColor(hDC, RGB(0,255,0));
                COLORREF bg = SetBkColor(hDC, RGB(0,0,0));
                fmt = "%d of %d";
                if (dd->cdata[cont_i].numcrunchers > 9 ||
                    dd->numcrunch_total > 9)
                {
                  fmt = "%d/%d";
                } 
                len = sprintf(buffer, fmt, dd->cdata[cont_i].numcrunchers,
                                           dd->numcrunch_total );
                hFont = (HFONT)SelectObject(hDC,hFont);
                GetTextExtentPoint(hDC,buffer,len,&ex);
                ex.cy = (ex.cy >= (rect.bottom-rect.top)) ? 0 :
                        (((rect.bottom-rect.top)-ex.cy)>>1);
                ex.cx = (ex.cx >= (rect.right-rect.left)) ? 0 :
                        (((rect.right-rect.left)-ex.cx)>>1);
                ExtTextOut(hDC,rect.left+ex.cx,(rect.top+ex.cy)-1,
                           ETO_CLIPPED,&rect,buffer,len,NULL);
                DeleteObject(SelectObject(hDC,hFont));
                SetBkColor(hDC,bg);
                SetTextColor(hDC,fg);
              }
            }
          }
          else if (id == IDC_GRAPH)
          {
            int xy, czone, height, width;
            int amp = dd->scale.amp;
            unsigned long x, y, last_y, sigma = 0;

            rect.bottom -= 2+1; 
            rect.top += 2;
            rect.right -= 2;
            rect.left += 2;

            height = (rect.bottom - rect.top); /* intentionally one short */
            width  = (rect.right - rect.left); /* intentionally one short */

            hPen = CreatePen(PS_SOLID, 1, RGB(0,128,0));
            hPen = (HPEN)SelectObject(hDC, hPen);

            xy = 9-((GetTickCount()/1000)%10);
            for (;xy < width; xy += 10)
            {
              MoveToEx(hDC, rect.left+xy, rect.top, NULL);
              LineTo(hDC, rect.left+xy, rect.bottom);
            }
            for (xy = 0; xy < 100; xy+=10)
            {
              y = (xy * xy * height)/(100 /* max_xy */ * 100); /* min pos */
              x = (xy * 1  * height)/(100 /* max_xy */ * 1);   /* max pos */
              y += ((x - y)*(100 /* max_amp */ - amp))/(100 /* max_amp */);
              MoveToEx(hDC, rect.left, rect.top+height-((UINT)y), NULL);
              LineTo(hDC, rect.right, rect.top+height-((UINT)y));
            }  
            DeleteObject(SelectObject(hDC, hPen));

            last_y = (unsigned long)-1;
            czone = ((amp > 0 && dd->show_noise)?(0):(1));
            for (; czone <= 1; czone++)
            {
              int donecount = 0;
              for (xy = (sizeof(dd->cdata[0].rate)/
                         sizeof(dd->cdata[0].rate[0]))-1; xy >= 0; xy--)
              {
                /* valid y is 1 ... dd->effmax */
                y = dd->cdata[cont_i].rate[xy];
                if (y < 1)
                  break;
                if (y == 1) /* magic - means set to 0.0 */
                  y = 0;
          
                if (czone == 0)
                {
                  y = (y * height)/dd->effmax;
                  #if 0 /* uncomment to see unmodified point */
                  if (amp == 100)
                    y = (y * y)/height;
                  else if (amp == 0)
                    ; /* nothing */
                  else
                  #endif 
                  if (y > 0)
                  {
                    y -= ((height - y) * amp)/100;
                    if (((long)y) < 0) 
                      y = 0;
                  }
                }
                else
                {
                  unsigned long z = y;   
                  y = (z * z * height)/(dd->effmax * dd->effmax); /* min pos */
                  x = (z * 1 * height)/(dd->effmax * 1);          /* max pos */
                  y += ((x - y)*(100 /* max_amp */ - amp))/(100 /* max_amp */);
                  sigma += y;
                }
                x = (xy * width)/ (sizeof(dd->cdata[0].rate)/
                                   sizeof(dd->cdata[0].rate[0]));
                if (donecount >= 2 && y == last_y)
                {
                  polys[donecount-1].x = rect.left + ((UINT)x);
                }
                else
                {
                  last_y = -1; //y;
                  polys[donecount].x = rect.left + ((UINT)x);
                  polys[donecount].y = rect.top + height - ((UINT)y);
                  donecount++;
                }
              }
              if (donecount < 2)
                break;
              hPen = CreatePen(PS_SOLID, 1, 
                              ((czone == 0)?(RGB(255,0,0)):(RGB(0,255,0))));
              hPen = (HPEN)SelectObject(hDC, hPen);
              Polyline(hDC,&polys[0],donecount);
              DeleteObject(SelectObject(hDC, hPen));
              if (czone == 1 && dd->show_avg)
              {
                y = (sigma + (donecount>>1))/donecount;
                if (y > 0)
                {
                  UINT oldmode = SetBkMode(hDC, TRANSPARENT);
                  polys[0].y = rect.top + height - ((UINT)y);
                              
                  hPen = CreatePen(PS_DOT, 1, RGB(255,255,0));
                  hPen = (HPEN)SelectObject(hDC, hPen);
                  MoveToEx(hDC, polys[0].x, polys[0].y, NULL);
                  LineTo(hDC, polys[donecount-1].x, polys[0].y );
                  DeleteObject(SelectObject(hDC, hPen));
                  SetBkMode(hDC, oldmode);
                }
              }
            } /* for (czone) */  
  
            if (dd->mode_pending)
            {
              COLORREF fg; UINT bkmode;
              __ShadeRect(hDC, &rect, TRUE);

              fg = SetTextColor(hDC, RGB(0,255,0));
              bkmode = SetBkMode(hDC, TRANSPARENT);
              DrawText(hDC, "Please Wait...", -1, &rect, 
                       DT_CENTER|DT_NOPREFIX|DT_SINGLELINE|DT_VCENTER);
              SetBkMode(hDC,bkmode);
              SetTextColor(hDC,fg);
            }
          } /* IDC_GRAPH */

          if (hDC != lpdis->hDC) /* used a mem context */
          {
            BitBlt(lpdis->hDC, 0, 0, bmp_width, bmp_height, hDC, 0, 0, SRCCOPY);
          }
        } /* if (hwnd) */
        if (lpdis && id != IDC_PROJLIST && id != IDC_LOG)
        {
          if (lpdis->itemState == ODS_FOCUS)
            SetFocus(GetDlgItem(dialog, IDC_PROJLIST));
        }
      } /* if (dd) */
      SetWindowLong(dialog, DWL_MSGRESULT, TRUE);
      TRACE_DLG((-1,"WM_DRAWITEM\n"));
      return TRUE;
    }
    case WM_CLOSE:
    {
      dd = (struct dlgdata *)GetWindowLong(dialog, DWL_USER);
      TRACE_DLG((+1,"WM_CLOSE dd=%p\n",dd));
      if (dd)
      {
        PostMessage(dd->hOwner,WM_CLOSE,0,0);
      }
      TRACE_DLG((-1,"WM_CLOSE\n"));
      return FALSE;
    }
    case WM_COMMAND:
    {
      dd = (struct dlgdata *)GetWindowLong(dialog, DWL_USER);
      TRACE_DLG((+1,"WM_COMMAND dd=%p\n",dd));
      if (dd)
      {
        WORD id = (WORD)LOWORD(wParam);   /* control or menu item identifier */
        #if (CLIENT_OS == OS_WIN32)       /*,-------------------------------,*/
        HWND hwnd = (HWND)lParam;         /*|         | cmd          | hwnd |*/
        WORD cmd  = (WORD)HIWORD(wParam); /*|---------+--------------+------|*/
        #else                             /*|menu opt | 0            | NULL |*/
        HWND hwnd = (HWND)LOWORD(lParam); /*|accel    | 1            | NULL |*/  
        WORD cmd  = (WORD)HIWORD(lParam); /*|control  | notification | hwnd |*/
        #endif                            /*'---------'--------------'------'*/

        if (id == WMCMD_CLOSEVIEW) /* internal between views */
        {
          TRACE_DLG((0,"WMCMD_CLOSEVIEW\n"));
          WriteDCTIProfileInt("scale", "ry", dd->scale.amp );
          WriteDCTIProfileInt("scale", "rx", dd->scale.freq );
          WriteDCTIProfileInt("scale", "squelch", !(dd->show_noise) );
          WriteDCTIProfileInt("scale", "savg", (dd->show_avg) );

          if (dd->timer)
          {
            KillTimer(dialog,dd->timer);
            dd->timer = 0;
          }
          if (dd->hOwner && (winGetVersion()%2000)>=400 && 
                   !IsWindowVisible(dd->hOwner) && IsIconic(dd->hOwner))
          {
            __DoTrayStuff(dd->hOwner, +1, NULL, "ScopeView" );
          }
          dd->console = NULL;
          dd->hOwner = 0;
          if (dd->hBGBrush)
          {
            DeleteObject( dd->hBGBrush );
            dd->hBGBrush = NULL;
          }
          if (dd->hFont)
          {
            __w16FixupDlgFont(dialog, dd->hFont);
            dd->hFont = NULL; 
          }
          if (dd->hMemDC)
          {
            DeleteObject(SelectObject( dd->hMemDC, dd->hMemBitmap ));
            DeleteDC(dd->hMemDC);
            dd->hMemBitmap = NULL;
            dd->hMemDC = NULL;
          }
          free((void *)dd);
          SetWindowLong(dialog, DWL_USER, 0);
          if (!GetParent(dialog))
          {
            hmenu = GetMenu(dialog);
            SetMenu(dialog,NULL);
            __w16WindowDestroyMenu(hmenu); 
            EndDialog(dialog,0);
          }
        }
        else if (id == WMCMD_REFRESHVIEW)
        {
          TRACE_DLG((0,"WMCMD_REFRESHVIEW\n"));
          hwnd = GetDlgItem(dialog,IDC_LOG);
          if (hwnd)
          { 
            int newcount = dd->console->currow;
            int curcount = dd->logwindow.rowcount;
            if (newcount > curcount)
            {
              while (curcount < newcount)
              {
                SendMessage(hwnd, LB_INSERTSTRING, (WORD)curcount, 
                            (LPARAM)(&(dd->console->buff[curcount][0])) );
                curcount++;
              }  
              dd->logwindow.rowcount = newcount;
            }
            if (dd->logwindow.font_height == 0)
            {
              hDC = GetDC(hwnd);
              if (hDC)
              {
                /* don't use a char with decenders in 'linep' */
                const char *linep = "[XXX 99 99:99:99 UTC] ";
                HFONT hFont = (HFONT)SelectObject(hDC, GetStockObject(ANSI_VAR_FONT));
                SIZE fsize; UINT mmode = SetMapMode(hDC, MM_TEXT);
                GetTextExtentPoint(hDC, linep, (UINT)strlen(linep), &fsize);
                SelectObject(hDC, hFont);
                SetMapMode(hDC, mmode);
                dd->logwindow.font_height = fsize.cy;
                dd->logwindow.tab_width = fsize.cx;
                #ifndef GCL_STYLE
                #define GCL_STYLE GCW_STYLE /* -26 */
                #endif
                if ((GetClassLong(hwnd,GCL_STYLE) & CS_CLASSDC)==0)
                  ReleaseDC(hwnd, hDC);
              }
              if (dd->logwindow.font_height)
              {
                SetScrollRange(hwnd, SB_VERT, 0, W16CONS_HEIGHT-1, FALSE);
                SetScrollPos(hwnd, SB_VERT, W16CONS_HEIGHT-1, TRUE);
                ShowScrollBar(hwnd, SB_VERT, TRUE);
                SendMessage( hwnd, LB_SETITEMHEIGHT, 0, 
                                   (WORD)(dd->logwindow.font_height) );
              }   
            }
            InvalidateRect(hwnd,NULL,FALSE);
          }
        }
        else if (hwnd && id == IDC_SHOWNOISE)
        {
          dd->show_noise = (SendMessage(hwnd,BM_GETCHECK,0,0) == 1);
        }
        else if (hwnd && id == IDC_SHOWAVG)
        {
          dd->show_avg = (SendMessage(hwnd,BM_GETCHECK,0,0) == 1);
        }
        else if (hwnd && id == IDC_PROJLIST &&
           (cmd == LBN_SELCHANGE || cmd == 0))
        { 
          TRACE_DLG((0,"IDC_PROJLIST LBN_SELCHANGE\n"));
          short pos = (short)SendMessage(hwnd,LB_GETCURSEL,0,0);
          if (pos >= 0 && pos < CONTEST_COUNT)
          {
            dd->cont_sel = (unsigned int)pos;
            dd->timer_cont_sel = -1; /* contest at last WM_TIMER */
            dd->cont_sel_explicit = (unsigned int)pos;
            dd->cont_sel_exptime = GetTickCount(); /* switch time */
            PostMessage(dialog,WM_TIMER,1,0); /* load changed buffer info */
            InvalidateRect(dialog,0,0); /* everything needs repainting */
          }
        }
        else if (id >= 1 && id <= 7)  /* IDCANCEL etc */
        { 
          TRACE_DLG((0,"unhandled WM_COMMAND %d (id=%d, hwnd=%p)\n", wParam, id, hwnd));
          ; /* ignore it */
        }
        else if (!hwnd) /* not a control message */
        {
          TRACE_DLG((0,"misc WM_COMMAND %d\n", wParam));
          SendMessage(dd->hOwner,WM_COMMAND,id,0);
          if (!GetParent(dialog) && !CheckExitRequestTriggerNoIO())
          {
            hmenu = GetMenu(dialog);
            SetMenu(dialog,__w16WindowConstructMenu(dd->console,dialog,WM_INITMENU,0));
            __w16WindowDestroyMenu(hmenu);
            DrawMenuBar(dialog);
          }        
        }
      }
      TRACE_DLG((-1,"WM_COMMAND\n"));
      return FALSE; 
    }

  } /* case */

  TRACE_DLG((0,"unhandled DLG message 0x%x wParam=0x%x, lParam=0x%x\n", msg, wParam, lParam));
  return FALSE;
}


#if 0
#define XTRACE_MSG(__msg)  case ##__msg: \
           trace_out(0, #__msg " wParam=%x, lParam=%x\n",wParam,lParam);\
           break
    XTRACE_MSG( WM_SIZE );
    XTRACE_MSG( WM_MOVE );
#endif

/* ---------------------------------------------------- */

static void __clear_marked(W16CONP console)
{
  if (console != NULL)
  {
    memset(&(console->marked_buff[0]),0,sizeof(console->marked_buff));
    console->mark_lastrow = console->mark_down = console->have_marked = 0;
  }
  return;
}

/* ---------------------------------------------------- */

static int __w16WindowHandle_DNETC_WCMD(HWND hwnd, UINT message,
                        WPARAM wParam, LPARAM lParam, LRESULT *lResultP )
{
  int handled = 0;
  hwnd = hwnd; lParam = lParam; // shaddup compiler

  /* the reason we have two forms is 
  ** a) the DNETC_WCMD_* forms are constant between all versions of the 
  **    client, the WMCMD_* forms are volatile. 
  ** b) the DNETC_WCMD_* forms have the same identifiers as standard 
  **    dialog IDOK etc identifiers.
  */
  if (message == WM_COMMAND)
  {
    if (wParam == DNETC_WCMD_SHUTDOWN /* 1 */
      || wParam == WMCMD_SHUTDOWN)
    {
      RaiseExitRequestTrigger();
      CheckExitRequestTrigger();
      handled = 1;
    }
    else if (wParam == DNETC_WCMD_RESTART /* 2 */
      || wParam == WMCMD_RESTART)
    {
      RaiseRestartRequestTrigger();
      CheckExitRequestTrigger();
      handled = 1;
    }
    else if (wParam == DNETC_WCMD_PAUSE /* 3 */
      || wParam == WMCMD_PAUSE)
    {
      RaisePauseRequestTrigger();
      CheckPauseRequestTrigger();
      handled = 1;
    }
    else if (wParam == DNETC_WCMD_UNPAUSE /* 13 */
      || wParam == WMCMD_UNPAUSE)
    {
      ClearPauseRequestTrigger();
      CheckPauseRequestTrigger();
      handled = 1;
    }
  }
  if (handled && lResultP)
    *lResultP = DNETC_WCMD_ACKMAGIC;
  return handled;
}

/* ---------------------------------------------------- */

//#define RESIZE_AFTER_PAINT
static void __w16PaintProc(HWND hwnd, W16CONP console)
{
  RECT clirect;
  if (GetUpdateRect(hwnd, &clirect, 1)!= 0) /* something needs updating */
  {
    int done_paint = 0;
    if (console)
    {
      GetClientRect(hwnd,&clirect);
      if (clirect.bottom > clirect.top) /* .bottom is -5 for rollups */
      {
        if (!console->hfont ||
             memcmp(&(console->lastpaintrect),&clirect,sizeof(RECT))!=0 )
        {
          __w16AdjustRect( console, hwnd, WM_PAINT, 0, 0);
          GetClientRect(hwnd,&clirect);
          memcpy(&(console->lastpaintrect),&clirect,sizeof(RECT));
        }
        if (console->hfont)
        {
          {
            PAINTSTRUCT ps;
            HDC paintdc = BeginPaint(hwnd, &ps);
            #ifdef RESIZE_AFTER_PAINT
            int longestline = 0;
            #endif
            RECT orgrect;
            orgrect.top = clirect.top;
            orgrect.left = clirect.left;
            orgrect.bottom = clirect.bottom;
            orgrect.right = clirect.right;
            if (paintdc)
            {
              HFONT oldFont; HBRUSH hBrush[2]; HBRUSH oldBrush;
              int oldMapMode; DWORD mapperFlags;
              //int offsetrow = 0; //GetScrollPos(hwnd, SB_VERT);
              //int offsetcol = 0; //GetScrollPos(hwnd, SB_HORZ);
              COLORREF oldTClr, oldBClr;
              HBITMAP oldbitmap = NULL; HDC workdc;

              workdc = CreateCompatibleDC(paintdc);
              if (workdc)
              {
                oldbitmap = CreateCompatibleBitmap( paintdc,
                    #ifdef RESIZE_AFTER_PAINT
                               (clirect.right*2)+1, 
                    #else
                               (clirect.right)+1, 
                    #endif
                               clirect.bottom+1 );
                if (oldbitmap)
                {
                  oldbitmap = (HBITMAP)SelectObject( workdc, oldbitmap );
                }
                else
                {
                  DeleteDC(workdc);
                  workdc = NULL;
                }
              }
              if (!workdc)
                workdc = paintdc;

              TRACE_PAINT((0,"WM_PAINT 1\n" ));
              oldMapMode = SetMapMode( workdc, MM_TEXT);
              mapperFlags = my_InitMapperFlags( workdc );
              oldTClr = GetSysColor(COLOR_WINDOWTEXT); /* RGB( 0,0,0 ) */
              oldBClr = GetSysColor(COLOR_WINDOW); /* RGB( 224,224,224 ) */
              if (oldBClr == RGB(255,255,255) && oldTClr == RGB(0,0,0))
                oldBClr = GetNearestColor(workdc, RGB(0xee,0xee,0xee)); //RGB(0xf2,0xf1,0xf4));

              oldTClr = SetTextColor(workdc, oldTClr );
              oldBClr = SetBkColor(workdc, oldBClr );
              oldFont = (HFONT)SelectObject(workdc, console->hfont);
              hBrush[0] = CreateSolidBrush( GetBkColor( workdc ) );
              hBrush[1] = CreateSolidBrush( GetTextColor( workdc ) );
              oldBrush = (HBRUSH)SelectObject( workdc, hBrush[0] );
              TRACE_PAINT((0,"WM_PAINT 2\n" ));

              //FillRect(workdc, &clirect, hBrush[0]);
              __w16DrawRecessedFrame( workdc, &clirect, hBrush[0] );
   
              clirect.top += console->indenty;
              clirect.left += console->indentx;
              clirect.bottom -= console->indenty;
              clirect.right -= console->indentx;

              console->caretpos = 0;
#if 0
              if (console->fontisvarpitch)
              {
                int n, row, rectlen = -1, blen = -1;
                int longestline = 0;
                char blankline[W16CONS_WIDTH];
                memset(blankline,' ',sizeof(blankline));
                for (row = 0; row <= console->currow; row++)
                {
                  RECT dtr; int dispcols = console->dispcols, skipped = 0;
                  const char *line = &console->buff[row][0];
                  if (memcmp(blankline,line,console->dispcols)==0)
                  {
                    continue;
                  }
                  if (*line == '[' && blen == -1)
                  {
                    n = 0;
                    while (n < dispcols && line[n] != ']')
                      n++;
                    if (n != dispcols)
                    {
                      dtr.top = dtr.left = dtr.bottom = dtr.right = 0;
                      if (DrawText(workdc, line, n, &dtr,DT_CALCRECT|
                        DT_NOCLIP|DT_SINGLELINE|DT_LEFT|DT_NOPREFIX|DT_TOP)!=0)
                      {
                        rectlen = dtr.right - dtr.left;
                        blen = n;
                      }
                    }
                  }
                  dtr.top    = console->indenty + row * console->fonty;
                  dtr.left   = console->indentx;
                  dtr.bottom = dtr.top + console->fonty;
                  dtr.right  = dtr.left + console->fontx * dispcols;
                  if (blen > 0 && memcmp(blankline,line,blen)==0)
                  {
                    line += blen;
                    dispcols -= blen;
                    dtr.left += rectlen;
                    skipped = 1;
                  }  
                  n = dispcols;
                  while (n > 0 && line[n-1]==' ')
                    n--;
                  if (n > 0)
                  {
                    DrawText(workdc, line, n, &dtr, 
                          DT_NOCLIP|DT_SINGLELINE|DT_LEFT|DT_NOPREFIX|DT_TOP);
                    #ifdef RESIZE_AFTER_PAINT
                    if (DrawText( workdc,line, n, &dtr, DT_CALCRECT|
                          DT_NOCLIP|DT_SINGLELINE|DT_LEFT|DT_NOPREFIX|DT_TOP))
                    {
                      n = dtr.right - dtr.left;
                      if (skipped)
                        n += rectlen;
                      if (n > longestline) 
                        longestline = n; 
                    }  
                    #endif
                    if (!skipped && row == console->currow && console->curcol>0)
                    {
                      if (DrawText(workdc, line, console->curcol, &dtr, DT_CALCRECT|
                          DT_NOCLIP|DT_SINGLELINE|DT_LEFT|DT_NOPREFIX|DT_TOP))
                      {
                        console->caretpos = dtr.right-dtr.left;
                      }                       
                    }
                  }
                }
              } /* font is varpitch */
              else
#endif
              {
                //if (winGetVersion() < 400) 
                {
                  int row;
                  UINT oldalignment = SetTextAlign(workdc, TA_LEFT|TA_TOP ); 
                  for (row = 0; row < console->disprows; row++)
                  {  
                    TextOut(workdc, clirect.left, 
                                    clirect.top + (row * console->fonty),
                                    &(console->buff[row][0]), 
                                    console->dispcols );
                  }
                  SetTextAlign(workdc, oldalignment ); 
                }
                #if 0
                else
                {
                  if (!console->literal_buff_is_valid)
                  {              
                    int row; char *dest = console->literal_buff;
                    char blankline[W16CONS_WIDTH];
                    memset( blankline, ' ', sizeof(blankline) );
                    for (row = 0; row < console->disprows; row++)
                    {
                      if (memcmp(blankline, &console->buff[row][0], console->dispcols))
                      {
                        memcpy(dest, &console->buff[row][0], console->dispcols);
                        dest += console->dispcols;
                      }
                      *dest++ = '\r';
                      *dest++ = '\n';
                    }
                    *dest = '\0';
                    console->literal_buff_is_valid = 1;
                  }
                  DrawText(workdc, console->literal_buff, -1,
                           &clirect, DT_NOCLIP|DT_LEFT|DT_NOPREFIX|DT_TOP);
                }
                #endif
              }
              /* +++++++++ */
              {
                int row, rowstart = -1, rowcount = 0;
                for (row = 0; row < console->disprows; row++)
                {
                  if (console->marked_buff[row]) 
                  {
                    if (rowstart < 0)
                      rowstart = row;
                    rowcount++;
                    if (row < (console->disprows-1))
                      continue;
                  }
                  if (rowstart >= 0)
                  {
                    clirect.top    = console->indenty + rowstart * console->fonty;
                    clirect.bottom = clirect.top + rowcount * console->fonty;
                    InvertRect( workdc, &clirect );
                    rowstart = -1;
                    rowcount = 0;  
                  }
                }
                done_paint = 1;
              }

              if (workdc != paintdc)
              {
                BitBlt( paintdc, 0, 0, orgrect.right, orgrect.bottom,
                        workdc, 0, 0, SRCCOPY);
              }

              SelectObject( workdc, oldBrush );
              DeleteObject( hBrush[0] );
              DeleteObject( hBrush[1] );
              SetBkColor( workdc, oldBClr );
              SetTextColor( workdc, oldTClr );
              SelectObject(workdc, oldFont );
              SetMapperFlags( workdc, mapperFlags );
              SetMapMode( workdc, oldMapMode );     

              if (workdc != paintdc)
              {
                DeleteObject(SelectObject( workdc, oldbitmap ));
                DeleteDC( workdc );
              }
              EndPaint(hwnd, &ps);
            } /* if (paintdc) */
            #ifdef RESIZE_AFTER_PAINT
            if (done_paint && longestline > (orgrect.right - orgrect.left))
            {
              GetWindowRect(hwnd,&clirect);
              clirect.right += longestline - (orgrect.right - orgrect.left);
              MoveWindow(hwnd,clirect.left,clirect.top,
                              (clirect.right-clirect.left),
                              (clirect.bottom-clirect.top),FALSE);
            }
            #endif
          } /* if (literal_buff) */
        } /* if (hfont) */
      } /* if (clirect.bottom > clirect.top) */ /* .bottom is -5 for rollups */
    } /* if (console) */
    TRACE_PAINT((0,"WM_PAINT 14\n" ));
    if (hwnd == GetFocus())
      __win16AdjustCaret(hwnd, console, 1);
    TRACE_PAINT((0,"WM_PAINT 15\n" ));
    if (!done_paint)
      ValidateRect(hwnd, NULL); /* don't send more paint msgs if paint fails */
    TRACE_PAINT((0,"WM_PAINT 16 (end)\n" ));
  } /* if something needs updating */
  return;
}

static void __w16UpdateWinMenu(W16CONP console, HWND hwnd)
{
  HMENU hmenu = __w16WindowConstructMenu(console, hwnd, WM_INITMENU, 0);
  if (hmenu)
  {
    HMENU hOldMenu = GetMenu(hwnd);
    if (SetMenu(hwnd, hmenu))
    {
      if (hOldMenu)
        __w16WindowDestroyMenu(hOldMenu);
    }
    DrawMenuBar(hwnd);
  }
  return;
}

static void __w16UpdateSysMenu(W16CONP console, HWND hwnd)
{
  if (console)
  {
    HMENU hmenu; DWORD oldstyle, newstyle;

    oldstyle = GetWindowLong(hwnd, GWL_STYLE);
    newstyle = oldstyle | (WS_MAXIMIZEBOX);
    if (console->rate_view.hwnd)
      newstyle &= ~(WS_MAXIMIZEBOX);
    if (oldstyle != newstyle)
      SetWindowLong(hwnd, GWL_STYLE, newstyle);

    hmenu = GetSystemMenu(hwnd, FALSE);
    if (hmenu)
    {
      UINT newbits, oldbits; 
      char text[64]; 
          
      oldbits = GetMenuState(hmenu, SC_SIZE, MF_BYCOMMAND);
      if (oldbits != ((UINT)-1))
      {
        newbits = oldbits & ~MF_GRAYED;
        if (console->needsnaphandler || console->rate_view.hwnd)
          newbits |= MF_GRAYED;
        if (newbits != oldbits)
        {
          GetMenuString(hmenu, SC_SIZE, text, sizeof(text),MF_BYCOMMAND);
          ModifyMenu(hmenu, SC_SIZE, MF_BYCOMMAND|newbits, SC_SIZE, text);
        }
      }
      oldbits = GetMenuState(hmenu, SC_MAXIMIZE, MF_BYCOMMAND);
      if (oldbits != ((UINT)-1))
      {
        newbits = oldbits & ~MF_GRAYED;
        if (console->rate_view.hwnd)
          newbits |= MF_GRAYED;
        if (newbits != oldbits)
        {
          GetMenuString(hmenu, SC_MAXIMIZE, text, sizeof(text),MF_BYCOMMAND);
          ModifyMenu(hmenu, SC_MAXIMIZE, MF_BYCOMMAND|newbits, SC_MAXIMIZE, text);
        }
      }
    } /* if (hMenu) */
  } /* if (console) */
  return;
}    


static LRESULT __w16WindowHandleMouse(W16CONP console, HWND hwnd, UINT message, 
                                      WPARAM wParam, LPARAM lParam )
{
  switch (message)
  {
    case WM_LBUTTONUP:
    case WM_LBUTTONDOWN:
    case WM_MOUSEACTIVATE:
    case WM_MOUSEMOVE:
    {
      // The mouse selection is designed such that:
      //    * Dragging within the window allows you to select multiple
      //      rows at a time.      
      //    * You must drag at least GetSystemMetrics(SM_CXDRAG|SM_CYDRAG)
      //      before the "dragthreshold" is satisfied and rows become 
      //      automatically selected.
      //      This has the purpose of eliminating selections caused by
      //      accidental single-clicks within the window (that is
      //      "accidental" clicks other than click-to-activate, which is
      //      automatically discarded by trapping the WM_MOUSEACTIVATE case)
      //    * Ctrl-clicking allows you to explicitly select or unselect
      //      single rows at a time, even after multiple rows have been
      //      automatically selected by dragging.
      //    * Shift-clicking allows you to select a range of rows from
      //      the last selected row to the currently clicked row.
      //    * Releasing the mouse button does not alter the selection.
      //      To actually copy the contents of the window, you must
      //      activate the "Copy" option in the context-menu or press ^C.
      //    * Deselecting the rows without copying is achieval by
      //      clicking and releasing within the window, and without
      //      triggering the "dragtreshold" amount.
      POINT prevmpos, currmpos;
      prevmpos.x = currmpos.x = 0;
      prevmpos.y = currmpos.y = 0;
      if (console != NULL)
      {
        if (message == WM_MOUSEACTIVATE)
        {
          if (LOWORD(lParam) == HTCLIENT && HIWORD(lParam) == WM_LBUTTONDOWN)
            console->mark_ignorenextlbdown = 1;
          console = NULL; /* nothing more to do with this message */
        }
        else 
        {
          prevmpos.x = currmpos.x = LOWORD(lParam);
          prevmpos.y = currmpos.y = HIWORD(lParam);
          prevmpos.x = console->mark_mlastpos.x;
          prevmpos.y = console->mark_mlastpos.y;
          console->mark_mlastpos.x = currmpos.x;
          console->mark_mlastpos.y = currmpos.y;
          if (message == WM_LBUTTONDOWN)
          {
            if (console->mark_ignorenextlbdown)
            {
              console->mark_ignorenextlbdown = 0;
              console = NULL; /* ignore this lbdown */
            }
          }
          else if (message == WM_MOUSEMOVE)
          {
            //note that we need *both* these tests, otherwise an 
            //an intervening window size/move will mess up the logic
            if ((wParam & MK_LBUTTON) == 0 || !console->mark_down)
              console = NULL; /* don't do anything with the message */
          }
          else if (message == WM_LBUTTONUP)
          {
            console->mark_down = 0;
            console = NULL; /* don't do anything with the message */
          }
        }
      }
      if (console != NULL) /* WM_LBUTTONDOWN or WM_MOUSEMOVE+MK_LBUTTON */
      {
        int i, row = 0;
        /* find the window row the mouse is on */
        while (row <= console->disprows)
        {
          if (currmpos.y >= (row * console->fonty) &&
              currmpos.y <  ((row+1) * console->fonty))
            break;
          row++;
        }
        if (row >= 0 && row < W16CONS_HEIGHT)
        {
          int need_refresh = 0;   
          /*
           * The wParam bits need to be tested _first_. Click+shift or 
           * click+control would be converted by windows into a move+xxx
           * if there was even a one pixel difference in position.
          */
          if ((wParam & MK_SHIFT) != 0) /* WM_LBUTTONDOWN is implicit */
          {
            if (console->have_marked && 
                console->mark_lastrow >= 0 && /* should be true */
                console->mark_lastrow < W16CONS_HEIGHT)                 
            { 
              /* mark range between last marked row and current. */
              /* note that unlike MK_CONTROL, the anchor (mark_lastrow) 
                 doesn't change: Imagine 3 shift-clicks - the first sets an
                 anchor. The second marks an area on one side of the anchor.
                 Now, if the third is further away from the anchor AND on the
                 same side (up/down) as the second click, then the marked
                 area is extended as expected. Inversely, if the third is 
                 closer to the anchor than the second (and on the same side) 
                 the area between second and third needs to be unmarked; or
                 looking at it from another angle, the entire area is unmarked
                 and then remarked upto the third shift-click. The same
                 thing then applies when the anchor is on the _other_ side of
                 the anchor: unmark all, then mark from anchor to third click.
                 This conforms to the way shift-click is handled for both 
                 listbox and edit control.
              */
              need_refresh = 1;
              for (i = 0; i < W16CONS_HEIGHT; i++)
                console->marked_buff[i] = 0;
              console->marked_buff[console->mark_lastrow] = 1;
              if (row != console->mark_lastrow)
              {
                i = row;
                while (i < console->mark_lastrow)
                  console->marked_buff[i++] = 1;
                while (i >= console->mark_lastrow)
                  console->marked_buff[i--] = 1;
              }  
            } 
            else
            {
              for (i = 0; i < W16CONS_HEIGHT; i++)
                console->marked_buff[i] = 0;
              console->mark_lastrow = row; /* new anchor */
              need_refresh = 1;
            }
            if (need_refresh)
            {
              console->marked_buff[row] = 1;
              console->have_marked = 1;
              console->mark_down = 1;
            }
          }
          else if ((wParam & MK_CONTROL) != 0) /* WM_LBUTTONDOWN is implicit */
          {
            /* additive select/deselect of individual lines */
            /* This conforms to the way ctrl-click is handled for listbox */
            /* edit control deals with ctrl-click the same as standard click */
            if (console->marked_buff[row])
            {
              console->marked_buff[row] = 0;
              console->have_marked = 0;   /* assumed */
              console->mark_lastrow = -1; /* invalid */
              for (i = 0; i < W16CONS_HEIGHT; i++)
              {
                if (console->marked_buff[i])
                {
                  console->have_marked = 1;
                  break;
                }
              }
              if (!console->have_marked)
              {
                console->mark_lastrow = row;
                console->mark_down = 1;
                console->have_marked = 1;
              }
            }
            else
            {
              console->marked_buff[row] = 1;
              console->mark_lastrow = row;
              console->mark_down = 1;
              console->have_marked = 1;
            }
            need_refresh = 1; 
          }
          else if (message == WM_MOUSEMOVE) /* (wParam & MK_LBUTTON) is implicit */
          { 
            /* extend marked area for as many lines as the mouse was moved. */
            /* This conforms to the way shift-click is handled for both */
            /* listbox and edit control. */
            if (console->have_marked) /* already marking? */
            {                       /* extend/shrink marked scope then */
              if (console->marked_buff[row]) /* marked? then undo */
              {           
                /* note that we undo in the direction opposite to which
                   the mouse is moving, and don't include the 'current'
                   row in the undo (otherwise we'll end up without a
                   visible anchor). This is similar to what shift+click
                   does.
                */ 
                if (row > console->mark_lastrow) /* moving down */
                {                         
                  i = row-1;
                  need_refresh = 1;
                  while (i >= console->mark_lastrow)
                    console->marked_buff[i--] = 0;  
                }
                else if (row < console->mark_lastrow) /* moving up */
                {
                  i = row+1;
                  need_refresh = 1;
                  while (i <= console->mark_lastrow)
                   console->marked_buff[i++] = 0;
                }
                if (need_refresh) /* moving vertically */
                {
                  console->mark_lastrow = row;
                }
              } /* if marked */
              else                           /* row isn't marked. mark it */
              {
                /* note that its possible that the number of lines affected
                   is greater than one. This is because the mouse could be
                   moved faster than windows sends WM_MOUSEMOVE messages
                */                   
                if (console->mark_lastrow < 0 || /* shouldn't happen */
                    console->mark_lastrow >= W16CONS_HEIGHT)
                  console->mark_lastrow = row;
                int i = row;
                while (i > console->mark_lastrow) /* moving down? */
                  console->marked_buff[i--] = 1;                  
                while (i < console->mark_lastrow) /* moving up? */
                  console->marked_buff[i++] = 1;                
                console->have_marked = 1;
                console->marked_buff[row] = 1;
                console->mark_lastrow = row;
                need_refresh = 1; 
              }
            } /* if (console->mark_down) */
            else                                /* were not marking */
            {
              /* if we hadn't been marking before, then figure out if a
                 drag threshold has been crossed, and start marking if it
                 has (marking everything between)
              */
              int absdiffx, absdiffy;
              int dragthreshx = (console->fontx+1)>>1; /* half cell in */
              int dragthreshy = (console->fonty+1)>>1; /* any direction */
              #if defined(SM_CXDRAG) && defined(SM_CYDRAG)
              i = GetSystemMetrics(SM_CXDRAG); /* generally 4x4 */
              if (i > dragthreshx)
                dragthreshx = i;
              i = GetSystemMetrics(SM_CYDRAG);
              if (i > dragthreshy)
                dragthreshy = i;
              #endif
              if (dragthreshx < 4)
                dragthreshx = 4;
              if (dragthreshy < 4)
                dragthreshy = 4;
              absdiffx = currmpos.x - prevmpos.x;
              if (currmpos.x < prevmpos.x)
                absdiffx = prevmpos.x - currmpos.x;
              absdiffy = currmpos.y - prevmpos.y;
              if (currmpos.y < prevmpos.y)
                absdiffy = prevmpos.y - currmpos.y;
              if (absdiffx > dragthreshx || absdiffy > dragthreshy)
              {
                need_refresh = 1; 
                for (i = 0; i < W16CONS_HEIGHT; i++)
                  console->marked_buff[i] = 0;
                console->mark_down = 1;
                console->marked_buff[row] = 1;
                console->mark_lastrow = row;
                console->have_marked = 1;
                /* its probably max one row up/down difference, but we */
                /* don't know what the system metrics are, so play safe */
                for (i=0; i <= console->disprows; i++)
                {
                  if (prevmpos.y >= (i * console->fonty) &&
                      prevmpos.y <  ((i+1) * console->fonty))
                  {
                    if (i != row && i < W16CONS_HEIGHT)
                    {
                      while (i < row)
                        console->marked_buff[i++] = 1; 
                      while (i > row)
                        console->marked_buff[i--] = 1;
                    }
                    break;
                  }
                }
              }
              else /* dragthresh not met yet. "go back" to start */
              {
                console->mark_mlastpos.x = prevmpos.x;
                console->mark_mlastpos.y = prevmpos.y;
              }
            } /* if !mark_down */
          }
          else                                  /* plain left-click */
          {
            /* clear all marked rows.
               This conforms to how an edit control deals with it.
               (A list box also clears but then sets a new anchor)
            */
            for (i = 0; i < W16CONS_HEIGHT; i++)
              console->marked_buff[i] = 0;
            console->have_marked = 0;
            console->mark_lastrow = row;
            /* mark_down must be set for move tracking */ 
            console->mark_down = 1;
            need_refresh = 1; 
          }
          if (need_refresh)
          {
            __w16UpdateWinMenu(console, hwnd);
            InvalidateRect(hwnd, NULL, FALSE);
          }
        } /* if (row >= 0 && row < W16CONS_HEIGHT) */
      } /* if (console) */
      return DefWindowProc(hwnd, message, wParam, lParam);
    }
  } /* switch */
  return DefWindowProc(hwnd, message, wParam, lParam); 
}



static void __GetViewWindowRect(HWND hwnd, W16CONP console, RECT *rect)
{
  /* note that is function will not work with windows in iconic state.
  ** Windows 95 (all) has a broken GetWindowSize() and 
  ** GetWindowPlacement() where the size of the "restored" window
  ** is always the size of the icon.
  */
  RECT wrect, crect;
  int ncheight, ncwidth;
  GetWindowRect(hwnd, &wrect);
#if 0 /* this following code fragment is useless on win95 (all OSRs)*/
  if (IsIconic(hwnd))
  {
    WINDOWPLACEMENT wp;
    memset(&wp,0,sizeof(wp));
    wp.length = sizeof(wp);
    if (GetWindowPlacement(hwnd, &wp))
      memcpy(&wrect,&(wp.rcNormalPosition),sizeof(wrect));
  }
#endif
  TRACE_ADJRECT((0,"H=%p/%p, wrect=%d,%d,%d,%d\n",
     hwnd, console->rate_view.hwnd, wrect.left, wrect.top, 
     wrect.right-wrect.left, wrect.bottom-wrect.top));
  GetClientRect(hwnd, &crect);
  ncheight = (wrect.bottom - wrect.top)-(crect.bottom - crect.top);
  ncwidth  = (wrect.right - wrect.left)-(crect.right - crect.left);
  TRACE_ADJRECT((0,"H=%p ncwidth=%d ncheight=%d\n",
                         console->rate_view.hwnd, ncwidth, ncheight));
  if (console->rate_view.hwnd)
  {
    GetWindowRect(console->rate_view.hwnd,&crect);
  }
  else
  {
    crect.top = crect.left = 0;
    crect.bottom = (console->fonty * console->disprows)+(console->indenty<<1);
    crect.right = (console->fontx * console->dispcols)+(console->indentx<<1);
  }
  rect->top = wrect.top;
  rect->left = wrect.left;
  rect->bottom = wrect.top + ncheight + ((crect.bottom - crect.top)-1);
  rect->right = wrect.left + ncwidth + ((crect.right - crect.left)-1);
  TRACE_ADJRECT((0,"H=%p/%p top=%d, left=%d, bottom=%d, right=%d\n",
     hwnd, console->rate_view.hwnd, rect->top, rect->left, rect->bottom, rect->right));
  return;
}

static void __w16_WM_WININICHANGE(W16CONP console, HWND hwnd)
{
  hwnd = hwnd;
  if (console)
  {
    #ifdef W16CONS_SMOOTHSIZING
    if (console->havewinextn)
    {
      #if !defined(SPI_GETFONTSMOOTHING) 
      #define SPI_GETFONTSMOOTHING 74
      #endif
      UINT oldval = 0;
      if (!SystemParametersInfo(SPI_GETFONTSMOOTHING,0,&oldval,0))
        oldval = 0;
      console->smoothsizing = ((oldval)?(1):(0));
    }
    #endif
    #if defined(WM_SIZING) /* otherwise always need snaphandler */
    if (console->havewinextn)
    {
      #if !defined(SPI_GETDRAGFULLWINDOWS)
      #define SPI_GETDRAGFULLWINDOWS 38
      #endif
      UINT oldval = 0;
      if (!SystemParametersInfo(SPI_GETDRAGFULLWINDOWS,0,&oldval,0))
        oldval = 0;
      console->needsnaphandler = ((oldval)?(0):(1));
    }
    #endif
  }
  return;
}


static LRESULT __w16WindowFuncInternal(int nestinglevel, HWND hwnd,
                                UINT message, WPARAM wParam, LPARAM lParam)
{
  W16CONP console = __win16GetHwndConsole( hwnd );
  nestinglevel = nestinglevel; /* shaddup compiler */

  if (console != NULL)
  {
    if (console->ssmessage && console->ssmessage == message)
    {
      message = WM_COMMAND;
      if (wParam == DNETC_WCMD_SHUTDOWN)
        console->ssmessage = 0;
    }
    else if (console->dnetc_cmdmsg && console->dnetc_cmdmsg == message)
      message = WM_COMMAND;
  }

  #if defined(TRACE) && (CLIENT_CPU == CPU_X86)
  {
    static char *applsp = NULL;
    if (applsp == NULL)
    {
      applsp = (char *)&hwnd;
      _asm sub applsp,esp
      TRACE_FLOW((0,"__w16WindowFunc local var size = %ld (0x%lx)\n",applsp,applsp));
    }
  }
  #endif

  switch (message)
  {
    case WM_LBUTTONUP:
    case WM_LBUTTONDOWN:
    case WM_MOUSEACTIVATE:
    case WM_MOUSEMOVE:
    {
      if (console)
      {
        if (console->rate_view.hwnd)
           return DefWindowProc(hwnd, message, wParam, lParam);
      }
      return __w16WindowHandleMouse(console, hwnd, message, wParam, lParam );
    }
    case WM_NCHITTEST:
    {
      LRESULT lResult = DefWindowProc(hwnd, message, wParam, lParam);
      if (console && (lResult == HTTOPLEFT || lResult == HTTOPRIGHT || 
                      lResult == HTBOTTOMLEFT || lResult == HTBOTTOMRIGHT || 
                      lResult == HTTOP || lResult == HTBOTTOM || 
                      lResult == HTLEFT || lResult == HTRIGHT))
      {
        if (console->rate_view.hwnd)
          lResult = HTNOWHERE;   
      } 
      return lResult;
    }
    case WM_CREATE:
    {
      #ifndef GWL_HINSTANCE
      #define GWL_HINSTANCE (-6)
      #endif
      HINSTANCE hinst = (HINSTANCE)GetWindowLong(hwnd,GWL_HINSTANCE);
      struct WM_CREATE_DATA *conscreate_data = NULL;

      if (lParam)
      {
        #if defined(__WINDOWS_386__)
        CREATESTRUCT far *lpCreateStruct = (CREATESTRUCT far *)MK_FP32((void *)lParam);
        #else
        LPCREATESTRUCT lpCreateStruct = (LPCREATESTRUCT)lParam;
        #endif
        if (lpCreateStruct)
          conscreate_data = (struct WM_CREATE_DATA *)(lpCreateStruct->lpCreateParams);
      }
      if (!conscreate_data)
      {
        TRACE_INITEXIT((0,"WM_CREATE,conscreate_data == NULL\n"));
        return -1;
      }
      
      TRACE_INITEXIT((0,"WM_CREATE,hInstance=%ld (0x%x)\n", hinst,hinst));

      console = (W16ConsoleStruc *)malloc(sizeof(W16ConsoleStruc));
      if (!console)
      {
        conscreate_data->create_pending = 0;
        conscreate_data->create_errorcode = W16CONS_ERR_NOMEM;
        // fail to create
        return -1;
      }

      // initialize our private structure
      memset((void *)console, 0, sizeof(W16ConsoleStruc));
      memset((void *)(&(console->buff[0][0])),' ',sizeof(console->buff));

      console->client_run_startstop_level_ptr = conscreate_data->client_run_startstop_level_ptr;

      console->hwnd = hwnd;
      console->disprows = W16CONS_HEIGHT;
      console->dispcols = W16CONS_WIDTH;
      console->smoothsizing = 0;
      console->ssmessage = GlobalFindAtom( W32CLI_SSATOM_NAME );
      console->dnetc_cmdmsg = RegisterWindowMessage(W32CLI_CONSOLE_NAME);
      console->indentx = console->indenty = 4;

      console->nCmdShow = conscreate_data->nCmdShow;
      #if (CLIENT_OS == OS_WIN32)
      if (console->nCmdShow == SW_SHOWDEFAULT)
      {
        STARTUPINFO si;
        si.cb = sizeof(si);
        GetStartupInfo(&si);
        if (si.wShowWindow)
          console->nCmdShow = si.wShowWindow;
        if (console->nCmdShow == SW_SHOWDEFAULT)
          console->nCmdShow = SW_SHOWNORMAL;
      }
      #endif

      if (console->nCmdShow != SW_HIDE && ModeReqIsSet(-1) == 0)
      {
        if (FindResource( hinst, MAKEINTRESOURCE(GRAPH_DIALOG), RT_DIALOG ))
        {
          /* this affects menu */
          console->rate_view.func = MakeProcInstance( (FARPROC)__w16GraphView, hinst);
        }
      }

      /* ------------ things that affect font selection ------- */
      console->fontx = (int)GetSystemMetrics(SM_CXSCREEN);
      if (console->fontx < 700) /* 640x480 */
      { console->fontx =  7; console->fonty = 12; }
      if (console->fontx < 900) /* 800x600 */
      { console->fontx =  8; console->fonty = 14; }
      else
      { console->fontx =  10; console->fonty = 16; }

      console->havewinextn = 0; /* need check for SPI_GETDRAGFULLWINDOWS */
      console->smoothsizing = 0;
      console->needsnaphandler = 1;
      if ((winGetVersion()%2000) >= 400)
      {
        if (winGetVersion() > 400) /* NT4+ or win98+ */
          console->havewinextn = 1;
        else
        { 
          #if !defined(SPI_GETWINDOWSEXTENSION) 
          #define SPI_GETWINDOWSEXTENSION    92
          #endif
          if (SystemParametersInfo(SPI_GETWINDOWSEXTENSION,1,0,0))
            console->havewinextn = 1;
        }
      }
      __w16_WM_WININICHANGE(console, hwnd);

      /* ---- all things that affect menu options come before here */

      GetSystemMenu( hwnd, 0); /* make our sys menu a copy of the std one */
      /* need to do win menu _before_ adjust rect otherwise the size will be off */
      __w16UpdateWinMenu(console, hwnd);
      __w16UpdateSysMenu(console, hwnd);
      EnableScrollBar(hwnd, SB_BOTH, ESB_DISABLE_BOTH);

      /* ---- all things that affect window size come before here */

      __w16AdjustRect( console, hwnd, WM_CREATE, 0, 0 );
      TRACE_INITEXIT((0,"WM_CREATE visible=%d,zoomed=%d,iconic=%d\n",IsWindowVisible(hwnd),IsZoomed(hwnd),IsIconic(hwnd)));
      if (console->hfont == NULL)
      {
        if (console->rate_view.func)
        {
          (void)FreeProcInstance(console->rate_view.func);
        }  
        free((void *)console);
        conscreate_data->create_pending = 0; /* failed */
        conscreate_data->create_errorcode = W16CONS_ERR_NOFONT;
        return -1;
      }

      /* ---- unimportant doo-wah from here on --------- */
 
      #if defined(WM_GETICON) && defined(WM_SETICON)
      if ((winGetVersion() % 2000) >= 400)        // Win95+, NT4+
      {
        HICON hIcon;
        UINT olderrmode = SetErrorMode(SEM_NOOPENFILEERRORBOX);
        HMODULE hInst = LoadLibrary( "user32.dll" );
        SetErrorMode(olderrmode);
        if (hInst != NULL)
        {
          typedef HANDLE (WINAPI *LoadImageAT)(HINSTANCE,LPCTSTR,UINT,int,int,UINT);
          LoadImageAT _LoadImage = (LoadImageAT) GetProcAddress(hInst, "LoadImageA");
          if (_LoadImage != NULL)
          {
            hIcon = (HICON)(*_LoadImage)(
                       (HINSTANCE)GetWindowLong(hwnd,GWL_HINSTANCE),
                         MAKEINTRESOURCE(1), IMAGE_ICON,
                           GetSystemMetrics(SM_CXSMICON),
                           GetSystemMetrics(SM_CYSMICON), 0 );
            if (hIcon != NULL)
            {
              console->hSmallIcon = hIcon;
              SendMessage(hwnd, WM_SETICON, 0 /*ICON_SMALL*/, (LPARAM)hIcon);
            }
          }
          FreeLibrary(hInst);
        }
        hIcon = (HICON)GetClassLong(hwnd, GCL_HICON);
        if (hIcon)
          SendMessage( hwnd, WM_SETICON, 1 /*ICON_LARGE*/, (LPARAM)hIcon );
      }
      #endif

      /* -------- end of unimportant doo-wah ----- */

      __win16SetHwndConsole( hwnd, console );

      if (console->rate_view.func && 
            GetDCTIProfileInt( "client", "view", 'c' ) == 'r')
      {
        PostMessage(hwnd,WM_COMMAND,WMCMD_SWITCHVIEW,WM_CREATE);
      }
      else /* can't showwindow() directly from here */
      {
        PostMessage(hwnd,WM_COMMAND,WMCMD_REFRESHVIEW,WM_CREATE);
      }
      conscreate_data->create_errorcode = 0;
      conscreate_data->create_pending = 0;
      break; /* return 0 */
    }
    case WM_WININICHANGE: /* aka WM_SETTINGCHANGE for WINVER >= 0x0400 */
    {
      __w16_WM_WININICHANGE(console,hwnd);
      return DefWindowProc(hwnd,message,wParam,lParam); 
    }
    case WM_CLOSE:
    {
      __conssize_saveupdateload(console, hwnd, -1 /*saveupdateload=-1,0,+1*/);
      RaiseExitRequestTrigger();
      return DNETC_WCMD_ACKMAGIC; /* all win cares about is non-zero */
    }
    case WM_QUERYENDSESSION:
    {
      return TRUE; /* yes, we can terminate */
    }
    case WM_ENDSESSION:
    {
      if (!wParam) /* not terminating */
        break;
      __conssize_saveupdateload(console, hwnd,-1/*saveupdateload=-1,0,+1*/);
      /* for the record:
         a) win95 (only) has a borked service shutdown
         sequence that permits other threads belonging to the process
         to continue running even when the window thread has died. This
         often causes a segfault because thread data has disappeared too.
         b) All win32: When *not* running as a service all threads *except*
         the window thread are suspended once shutdown is in progress.
         Waiting on a child will thus result in a 'hang' (win will kill
         the process after 10 seconds or so).
      */
      #if (CLIENT_OS == OS_WIN32)
      if (win32CliServiceRunning())
      {
        if (((lParam & ENDSESSION_LOGOFF)!=0))
          return 1;  /* if we are running as a service. don't exit */
        if (console)
        {
          int *client_run_startstop_level_ptr = 
                   console->client_run_startstop_level_ptr;
          if (*client_run_startstop_level_ptr > 0)
          {
            RaiseExitRequestTrigger();
            while (*client_run_startstop_level_ptr > 0)
              Sleep(100);
          }
        }
      }
      else
      #endif
      {
        RaiseExitRequestTrigger();
        CheckExitRequestTrigger(); /* print "*break* message to log */
        __w16ClientHardStop();
      }  
      break;
    }
    case WM_DESTROY:
    {
      if (console)
      {
        console->no_handle_wm_poschanged = 1;
        if (console->rate_view.hwnd)
        {
          SendMessage(hwnd,WM_COMMAND,WMCMD_SWITCHVIEW,WM_DESTROY);
        }
        if (console->rate_view.func)
        {
          (void)FreeProcInstance(console->rate_view.func);
          console->rate_view.func = NULL;
        }
      }
      __DoTrayStuff( hwnd, -1, NULL /* tip */, "DESTROY" );
      __conssize_saveupdateload(console, hwnd, -1   /* saveupdateload=-1/0/+1 */  );
      if (console)
      {
        #if defined(WM_GETICON) && defined(WM_SETICON)
        if (console->hSmallIcon)
        {
          SendMessage(hwnd, WM_SETICON, 0 /*ICON_SMALL*/, NULL); /* remove it */
          DestroyIcon(console->hSmallIcon);
        }
        #endif
        if (console->hfont && !console->fontisstock)
        {
          DeleteObject(console->hfont);
          console->hfont = NULL;
        }
        __win16SetHwndConsole( hwnd, NULL );
        free((void *)console);
      }
      {
        HMENU hMenu = GetMenu(hwnd);
        if (hMenu)
        {
          SetMenu(hwnd,NULL);
          __w16WindowDestroyMenu(hMenu);
        }
      }
      PostQuitMessage(0);
      return DefWindowProc(hwnd, message, wParam, lParam);
    }
    case WM_ERASEBKGND:
    {
      if (console)
      {
        if (console->rate_view.hwnd)
        {
          HDC hDC = (HDC)wParam;
          if (hDC)
          {
            HBRUSH hBrush = CreateSolidBrush(GetSysColor(COLOR_BTNHIGHLIGHT));
            if (hBrush)
            {
              RECT rect;
              GetClientRect(hwnd, &rect);
              hBrush = (HBRUSH)SelectObject(hDC, hBrush);
              FillRect(hDC, &rect, hBrush); 
              DeleteObject(SelectObject(hDC, hBrush)); 
            }
          }
        }
      }
      return TRUE; /* FALSE sets ps.fErase for WM_PAINT */
    }
    case WM_PAINT:
    {
      int nopaint = (IsIconic(hwnd) || !IsWindowVisible(hwnd));
      if (!nopaint && console)
      {
        if (console->rate_view.hwnd)
          nopaint = 1;
      }
      if (!nopaint)
        __w16PaintProc(hwnd, console);
      else
      {
        PAINTSTRUCT ps;
        HDC hDC = BeginPaint(hwnd, &ps);
        if (hDC)
          EndPaint(hwnd, &ps);
        break;
      }
      break;
    }
    case WM_GETMINMAXINFO:
    {
      MINMAXINFO FAR * lpmmi = (MINMAXINFO FAR *) lParam;
      #if defined(__WINDOWS_386__) /* convert 16:16 pointer to 16:32 */
         lpmmi = (MINMAXINFO FAR *)MK_FP32( (void *)lpmmi );
      #endif
      /* approximation of NC space without menu or scrollbars */
      UINT cxframe = 2 * GetSystemMetrics(SM_CXFRAME);
      UINT cyframe = 2 * GetSystemMetrics(SM_CYFRAME) + 
                         GetSystemMetrics(SM_CYCAPTION);

      lpmmi->ptMinTrackSize.x = cxframe + (W16CONS_WIDTH * 2);
      lpmmi->ptMinTrackSize.y = cyframe + (W16CONS_HEIGHT * 4);
      lpmmi->ptMaxTrackSize.x = cxframe + (W16CONS_WIDTH * 12);
      lpmmi->ptMaxTrackSize.y = cyframe + (W16CONS_HEIGHT * 29);

      if (lpmmi->ptMaxTrackSize.x > lpmmi->ptMaxSize.x)
        lpmmi->ptMaxTrackSize.x = lpmmi->ptMaxSize.x;
      else
        lpmmi->ptMaxSize.x = lpmmi->ptMaxTrackSize.x;

      if (lpmmi->ptMaxTrackSize.y > lpmmi->ptMaxSize.y)
        lpmmi->ptMaxTrackSize.y = lpmmi->ptMaxSize.y;
      else
        lpmmi->ptMaxSize.y = lpmmi->ptMaxTrackSize.y;

      return DefWindowProc(hwnd, message, wParam, lParam);
    }
    case WM_WINDOWPOSCHANGING:
    {
      /* This isn't strictly necessary, and is only here to
         prevent the user from using some external utility to
         show the window when the client isn't expecting it.
         Trying to adjust the rect from here will NOT work:
         Neither IsIconic() nor IsZoomed() are meaningful
         at this point, and although it might be possible to
         track the syscommand that (may have) brought us here,
         its really not worth it.
      */
      if (console)
      {
        WINDOWPOS FAR *pwp = (WINDOWPOS FAR *)lParam;
        #if defined(__WINDOWS_386__) /* convert 16:16 pointer to 16:32 */
          pwp = (WINDOWPOS FAR *)(MK_FP32((void *)pwp));
        #endif
        if ((pwp->flags & SWP_SHOWWINDOW)==0 && console->nCmdShow == SW_HIDE)
        {
          pwp->flags ^= SWP_SHOWWINDOW;
          pwp->flags |= SWP_HIDEWINDOW;
        }
      }
      return DefWindowProc(hwnd, message, wParam, lParam);
    }
    case WM_WINDOWPOSCHANGED:
    {
      /* at this point we have an updated IsIconized(), and
         consequently can fix the tray stuff. This is a good place
         since no painting has been done yet
      */
      if (console)
      {
        if (!console->no_handle_wm_poschanged)
        { 
          __conssize_saveupdateload(console, hwnd, 0 /*saveupdateload=-1,0,+1*/);
          /* this may cause taskbar/tray to get rearranged, so don't do it
             if a popup menu is in progress or we are WM_DESTROYing
          */
          __DoTrayStuff( hwnd, +1, NULL /* tip */, "WM_WINDOWPOSCHANGED" );
        }
      }
      return DefWindowProc(hwnd, message, wParam, lParam);
    }
    #if defined(WM_SIZING)
    case WM_SIZING: /* WM_SIZING is sent by WM_NCLBUTTONDOWN */
    {
      if (console && !console->rate_view.hwnd && !console->needsnaphandler)
        __w16AdjustRect( console, hwnd, message, wParam, lParam);
      return DefWindowProc(hwnd, message, wParam, lParam);
    }
    #endif
    case WM_NCLBUTTONDOWN:
    {
      if (console && !console->rate_view.hwnd && console->needsnaphandler)
        return __w16Handle_NCLBUTTONDOWN(console, hwnd, message, wParam, lParam);
      return DefWindowProc(hwnd, message, wParam, lParam);
    }
    case WM_SIZE:
    {
      if (console)
      { 
        /* we only need to catch SIZE_MAXIMIZED because the others
        ** will not influence (or not have influenced) the size of the
        ** client rect since we handle WM_SIZING (win32) or track/adjust
        ** on WM_LCBUTTONDOWN (win16 or win32 without drag-full-windows)
        */
        if (wParam == SIZE_MAXIMIZED /*|| wParam == SIZE_RESTORED*/
           || (console->needsnaphandler && !console->rate_view.hwnd) )
        {
          __w16AdjustRect( console, hwnd, message, wParam, lParam);
          if (wParam == SIZE_RESTORED)
            __conssize_saveupdateload(console, hwnd,0/*saveupdateload=-1,0,+1*/);
          return 0;
        }
      }
      return DefWindowProc(hwnd, message, wParam, lParam);
    }
#if 0
    case WM_MOVE:
    {
      __conssize_saveupdateload(console, hwnd, 0 /*saveupdateload=-1,0,+1*/);
      return DefWindowProc(hwnd, message, wParam, lParam);
    }
#endif
    case WM_RBUTTONUP:
    {
      if (console && !CheckExitRequestTriggerNoIO())
      {
        int intray = (lParam == (LPARAM)0XfedccdefL);
        HMENU hmenu = __w16WindowConstructMenu(console, hwnd, 0, intray );

        if (hmenu)
        {
          POINT ptPos; ptPos.x = 0;
          if (intray || (lParam==((LPARAM)0xffffffffL)))
            GetCursorPos(&ptPos);
          else
          {
            ptPos.x = LOWORD(lParam); 
            ptPos.y = HIWORD(lParam);
            ClientToScreen(hwnd, &ptPos);
          }
          if (console)                 /* recursion guard against msgs */
            console->no_handle_wm_poschanged++; /* sent by SetForegroundWindow */

          SetForegroundWindow(hwnd);  //needed for proper focus on tray popup
          TrackPopupMenu(hmenu, TPM_LEFTALIGN, ptPos.x, ptPos.y, 0, hwnd, NULL);
          PostMessage(hwnd, WM_NULL, 0, 0);

          if (console)
            console->no_handle_wm_poschanged--;

          __w16WindowDestroyMenu(hmenu);
        }
      }
      break;
    }
    /* case WM_INITMENU: */ /* 0x116 */
    case WM_INITMENUPOPUP: /* 0x117 */
    {
      if (message == WM_INITMENUPOPUP && HIWORD(lParam)) /* system menu */
        __w16UpdateSysMenu(console, hwnd);
      return DefWindowProc(hwnd, message, wParam, lParam); 
    }
    case WM_SYSCOMMAND:
    {
      if ((wParam & 0xfff0) == SC_MOUSEMENU ||
          (wParam & 0xfff0) == SC_KEYMENU)
      {
        /* menu needs to be updated here, rather than via 
        ** WM_INITMENU/WM_INITPOPUP due to differences in
        ** WM_*MENU handling by various versions of windows.
        ** For W9x, changing it at WM_INITMENU time is useless,
        ** and WM_INITPOPUPMENU is done too often to be efficient.
        */
        __w16UpdateWinMenu(console, hwnd);
      }
      else if ((wParam & 0xfff0) == SC_MAXIMIZE ||
               (wParam & 0xfff0) == SC_SIZE)
      {
        if (console)
        { 
          if (console->rate_view.hwnd)
          {
            break;
          }  
        }
      }
      return DefWindowProc(hwnd, message, wParam, lParam);
    }
    case WM_USER_SHELLNOTIFYICON:
    {
      if (lParam == WM_RBUTTONUP)
      {
        SendMessage(hwnd,WM_RBUTTONUP,0,((LPARAM)0xfedccdefL));
      }
      else if (lParam == WM_LBUTTONDBLCLK)
      { 
        TRACE_TRAY((+1,"WM_USER_SHELLNOTIFYICON: WM_LBUTTONDBLCLK\n"));
        #if (CLIENT_OS == OS_WIN32)
        UINT t = GetDoubleClickTime()>>1; //was 300
        TRACE_TRAY((+1,"Sleep(%d)\n", t));
        Sleep(t); //was 300 //absorb "extra-" click
        TRACE_TRAY((-1,"Sleep(%d)\n", t));
        #endif
        PostMessage(hwnd,WM_COMMAND,WMCMD_RESTORE,0); 
        TRACE_TRAY((-1,"WM_USER_SHELLNOTIFYICON: WM_LBUTTONDBLCLK\n"));
      }
      else if (lParam == WM_USER_SHELLNOTIFYICON)
      {
        __DoTrayStuff(hwnd, +1, NULL /* tip */, "SHELLNOTIFYICON" );
      }
      break;
    }
    case WM_COMMAND:
    {
      LRESULT lResult = 0;
      char* URL = (char *)0;
      int postmenuchange = 0;

      if (wParam == WMCMD_EVENT)
      {
        if ((console)?(console->rate_view.hwnd):(NULL))
        {  
          SendMessage(console->rate_view.hwnd, message, wParam, lParam ); 
        }
        else if (((struct WMCMD_EVENT_DATA *)lParam)->id == CLIEVENT_CLIENT_CRUNCHOMETER)
        {
          if (!IsWindowVisible(hwnd) && IsIconic(hwnd)) /* in tray */
          {
            const char *p = (const char *)(((struct WMCMD_EVENT_DATA *)lParam)->parm);
            while (*p == '\r' || *p == '\n')
              p++;  
            __DoTrayStuff(hwnd, 0, p, "WMCMD_CLIENTEVENT_CRUNCHOMETER" );
          }
        }
      }
      else if (wParam == WMCMD_RESTORE)
      {
        if (IsIconic(hwnd) && !IsWindowVisible(hwnd)) /* in-tray */
        {
          SetForegroundWindow(hwnd);
          __DoTrayStuff(hwnd, -1,  NULL /* tip */,"WMCMD_RESTORE" );
          PostMessage(hwnd, WM_NULL, 0, 0);
          ShowWindow(hwnd,SW_RESTORE); /* needed by __GetViewWindowRect */
          SetWindowPos( hwnd, HWND_TOP, 0, 0, 0, 0,
                        SWP_SHOWWINDOW|SWP_NOSIZE|SWP_NOMOVE);
          return 0;  
        }
        ShowWindow(hwnd, ((IsZoomed(hwnd))?(SW_SHOW):(SW_SHOWNORMAL)) );
      }
      else if (wParam == WMCMD_PASTE)
      {
        SendMessage( hwnd, WM_CHAR, 22 /* ^V */, 0);
      }
      else if (wParam == WMCMD_COPY)
      {
        SendMessage( hwnd, WM_CHAR, 24 /* ^X */, 0);
        postmenuchange = 1;
      }
      else if (wParam == WMCMD_CONFIG)
      {
        __clear_marked(console);
        ModeReqSet(MODEREQ_CONFIG | MODEREQ_CONFRESTART);
        if (IsIconic(hwnd)) /* by popular request */
          PostMessage(hwnd, WM_COMMAND, WMCMD_RESTORE, 0 );
      }
      else if (wParam == WMCMD_SVCINSTALL)
      {
        __clear_marked(console);
        #if (CLIENT_OS == OS_WIN32)
        win32CliInstallService(0); // == 0) /* no err */
        {
        /*
          if (IDYES == MessageBox( NULL,
              "The client is now installed as a service.\n",
               "Would you now like to restart the client to let it run as a service?",
               "distributed.net RC5DES client", MB_YESNO|MB_TASKMODAL ) )
          {
            //what we need to do is start a new process AFTER shutting down
            //the main thread but BEFORE we (the winproc) die. very hairy.
          }
        */
        }
        #endif
      }
      else if (wParam == WMCMD_FETCH)
      {
        __clear_marked(console);
        ModeReqSet(MODEREQ_FETCH);
        postmenuchange = 1;
      }
      else if (wParam == WMCMD_FLUSH)
      {
        __clear_marked(console);
        ModeReqSet(MODEREQ_FLUSH);
        postmenuchange = 1;
      }
      else if (wParam == WMCMD_UPDATE)
      {
        __clear_marked(console);
        ModeReqSet(MODEREQ_FETCH|MODEREQ_FLUSH);
        postmenuchange = 1;
      }
      else if (wParam >= WMCMD_BENCHMARK &&
               wParam <  (WMCMD_BENCHMARK+2+(CONTEST_COUNT*2)))
      {
        int do_mode = MODEREQ_BENCHMARK;
        if (((wParam - WMCMD_BENCHMARK) & 1) != 0)
          do_mode = MODEREQ_BENCHMARK_QUICK;
        __clear_marked(console);
        ModeReqSet(do_mode);
        if (wParam >= (WMCMD_BENCHMARK+2))
          ModeReqLimitProject(do_mode, (wParam-(WMCMD_BENCHMARK+2))>>1);
        postmenuchange = 1;
      }
      else if (wParam == WMCMD_REFRESHVIEW)
      {
        if (console && lParam == WM_CREATE)
        {
          ShowWindow( hwnd, console->nCmdShow ); /* last state */
        }
      }
      else if (wParam == WMCMD_SWITCHVIEW)
      {
        if (console)
        {
          int nCmdShow;  
          RECT rect;
          if (console->rate_view.hwnd)
          {
            /* order is critical here. Don't change it without checking
            ** that it works on win16 or systems without drag full windows
            */
            HWND hDlg = console->rate_view.hwnd;
            ShowScrollBar(hwnd, SB_VERT, TRUE);
            SendMessage(hDlg,WM_COMMAND,WMCMD_CLOSEVIEW,lParam);
            DestroyWindow(hDlg);
            console->rate_view.hwnd = NULL;

            if (lParam != WM_DESTROY)
            {
              postmenuchange = 1;
              WriteDCTIProfileInt("client", "view", 'c' );
              nCmdShow = SW_SHOWNORMAL;
              if (IsZoomed(hwnd))
                nCmdShow = SW_SHOWMAXIMIZED;  
              /* we can only switch back to a normal or maximized state */
              /* iconic is off limits because it causes */
              /* __GetViewWindowRect() to fail on win95 (only) */
              ShowWindow(hwnd,nCmdShow);
              ShowWindow(hwnd,SW_RESTORE); /* needed by __GetViewWindowRect */
              SetWindowPos( hwnd, HWND_TOP, 0, 0, 0, 0,
                            SWP_SHOWWINDOW|SWP_NOSIZE|SWP_NOMOVE);
              __GetViewWindowRect(hwnd,console,&rect);
              SetWindowPos( hwnd, 0,rect.left,rect.top,
                            rect.right-rect.left+1, rect.bottom-rect.top+1,
                            /*SWP_NOREDRAW|*/SWP_NOZORDER); 
              SetFocus(hwnd); /* needed for caret update and WM_CHAR handling */ 
              SetForegroundWindow(hwnd);
              __w16UpdateSysMenu(console, hwnd);
            }
          }
          else if (console->rate_view.func) /* switch possible */
          {  
            nCmdShow = SW_SHOW;
            if (lParam == WM_CREATE)
              nCmdShow = console->nCmdShow;
            else if (IsIconic(hwnd))
              nCmdShow = SW_SHOWMINIMIZED;
            else if (IsZoomed(hwnd))
              nCmdShow = SW_SHOWMAXIMIZED;
            else if (IsWindowVisible(hwnd))
              nCmdShow = SW_SHOW;
            if (nCmdShow != SW_HIDE)
            {
              console->rate_view.hwnd = CreateDialogParam(
                            (HINSTANCE)GetWindowLong(hwnd, GWL_HINSTANCE), 
                            MAKEINTRESOURCE(GRAPH_DIALOG), 
                            hwnd, (DLGPROC)console->rate_view.func, 
                            (LPARAM)hwnd );
              if (console->rate_view.hwnd)
              {
                WriteDCTIProfileInt("client", "view", 'r' );
                ShowScrollBar(hwnd, SB_VERT, FALSE);
                __GetViewWindowRect(hwnd,console,&rect);
                SetWindowPos( hwnd, 0, rect.left, rect.top,
                              rect.right-rect.left+1,
                              rect.bottom-rect.top+1,
                              /*SWP_NOREDRAW|*/SWP_NOZORDER);
                ShowWindow( hwnd, nCmdShow );
                ShowWindow(console->rate_view.hwnd,SW_SHOW);
                SetForegroundWindow(hwnd);
                postmenuchange = 1;
                __w16UpdateSysMenu(console, hwnd);
              }
            }
          } 
        }
      }
      else if (wParam == WMCMD_ABOUT)
      {
        __launch_about_box(hwnd);
      }
      else if (wParam == WMCMD_HELP_DOC)
      {
        URL = "http://www.distributed.net/docs/";
      }
      else if (wParam == WMCMD_HELP_FAQ)
      {
        URL = "http://www.distributed.net/faq/cache/1.html";
      }
      else if (wParam == WMCMD_HELP_BUG)
      {
        URL = "http://www.distributed.net/bugs/";
        if (winGetVersion()>=400) /* exceeds cmdline len otherwise */
        {
          URL = "http://www.distributed.net/bugs/buglist.cgi?product=Client&bug_status=UNCONFIRMED&bug_status=NEW&bug_status=ASSIGNED&bug_status=REOPENED&order=Bug+Number";
          //URL = "http://www.distributed.net/bugs/buglist.cgi?product=Client&bug_status=UNCONFIRMED&bug_status=NEW&bug_status=ASSIGNED&bug_status=REOPENED&order=Bug+Number&component=Beta-Test&component=config&component=&Configuration&component=Core-Selection&component=Core-Speed&component=Crashes%2FHangs&component=display+and+UI&component=Network%2FCommunications";
        }
      }
      else if (wParam == WMCMD_HELP_MAILTO)
      {
        URL = "mailto:help@distributed.net";
      }
      else
      {
        if (__w16WindowHandle_DNETC_WCMD(hwnd,message,wParam,lParam,&lResult))
          postmenuchange = 1; /* and lResult == DNETC_WCMD_ACKMAGIC */
        else
          lResult = 0;
      }
      if (URL)
      {
        HCURSOR hCursor;
        SetFocus(NULL);  
        hCursor = LoadCursor(NULL,MAKEINTRESOURCE(IDC_WAIT));
        if (hCursor)
          hCursor = SetCursor(hCursor);
        my_ShellExecute( GetDesktopWindow(), "open", URL, NULL, "\\", SW_RESTORE/*SW_SHOWNORMAL*/);
      }
      if (postmenuchange)
      {
        __w16UpdateWinMenu(console, hwnd);
      }
      return lResult;
    }
    case WM_KEYUP:
    {
      if ( console && ((int)(wParam)) == VK_INSERT &&
         ( GetKeyState(VK_SHIFT) & 0x8000 ) != 0 )  //shift-insert
      {
        wParam = 22; //^V
        message = WM_CHAR;
        //fall through!
      }
      else if ( console && ((int)(wParam)) == VK_F1 && ModeReqIsSet(-1) == 0 )
      {
        POINT ptPos; RECT rc;
        GetCursorPos(&ptPos);
        GetWindowRect(hwnd, &rc);
        if ((ptPos.x < rc.left) || (ptPos.x > rc.right) ||
            (ptPos.y < rc.top) || (ptPos.y > rc.bottom)) /*mouse is outside*/
        {
          GetClientRect(hwnd, &rc);
          ptPos.y = rc.top + (rand()%(rc.bottom-rc.top));
          ptPos.x = rc.left + (rand()%(rc.right-rc.left));
        }
        else
          ScreenToClient(hwnd, &ptPos);
        PostMessage(hwnd, WM_RBUTTONUP, 0, (((LPARAM)(ptPos.y))<<16)+ptPos.x);
        break;
      }
      else
      {
        return DefWindowProc(hwnd, message, wParam, lParam);
      }
    }
    case WM_CHAR:
    {
      if (console)
      {
        int kbuffsize = (int)(sizeof(console->keybuff)/sizeof(console->keybuff[0]));

        if (((int)(wParam)) == 0x03 /* ^C */ && console->have_marked)
          wParam = 24; /* convert to ^X since copy accel is usually ^C */
        if (((wParam>='A' && wParam<='Z') || (wParam>='a' && wParam<='z')) 
          && (GetAsyncKeyState(VK_CONTROL) & 0x8000)!=0)
        {
          wParam -= 'A';
        }
        if (((int)(wParam)) == VK_PAUSE && ((lParam & KF_EXTENDED) != 0))
                                                // pause/resume
        {
          if (CheckPauseRequestTriggerNoIO())
            ClearPauseRequestTrigger();
          else
            RaisePauseRequestTrigger();
        }
        else if (((int)(wParam)) == ('S'-'A'))  //^Q = resume
        {
          ClearPauseRequestTrigger();
        }
        else if (((int)(wParam)) == ('S'-'A')) //^S = pause
        {
          RaisePauseRequestTrigger();
        }
        else if (((int)(wParam)) == 0x03) //^C = break
        {
          RaiseExitRequestTrigger();
        }
        else if ( ((int)(wParam)) == 22 ) //^V = paste
        {
          if (ModeReqIsSet(MODEREQ_CONFIG))
          {
            if (IsClipboardFormatAvailable(CF_TEXT))
            {
              if (OpenClipboard(hwnd))
              {
                HGLOBAL hglb = GetClipboardData(CF_TEXT);
                if (hglb != NULL)
                {
                  #if defined(__WINDOWS_386__)
                  char far *lptstr = NULL;
                  void *_lptstr = (void *)GlobalLock(hglb);
                  if (_lptstr) lptstr = (char far *)(MK_FP32(_lptstr));
                  #else
                  LPTSTR lptstr = (LPTSTR)GlobalLock(hglb);
                  #endif
                  if (lptstr)
                  {
                    int len = 0, pos = 0;
                    while (pos < kbuffsize)
                    {
                      char c = lptstr[pos++];
                      if (!c || c == '\r' || c == '\n')
                        break;
                      console->keybuff[len++] = ((int)c) & 0xff;
                    }
                    if (len)
                    {
                      __clear_marked(console);
                      console->keycount = len;
                    }
                    GlobalUnlock(hglb);
                  }
                }
                CloseClipboard();
              }
            }
          }
        }
        else if (((int)(wParam)) == 24)   //^X = copy
        {
          if (console->have_marked)
          {
            const char *failmsg = NULL;
            char *copybuff = (char *)malloc(W16CONS_HEIGHT*(W16CONS_WIDTH+2));
            if (!copybuff)
              failmsg = "Insufficent memory for copy operation";
            else
            {
              int pos, copylen = 0;
              for (pos = 0; pos < console->disprows; pos++)
              {
                if (console->marked_buff[pos]) /* row is marked */
                {
                  int colpos, spcpos, linelen;
                  if (copylen) /* previous line needs '\n' */
                  {
                    // append linebreak to existing buffer contents.
                    copybuff[copylen++] = '\r';
                    copybuff[copylen++] = '\n';
                  }
                  // find end of actual non-blank characters.
                  spcpos = linelen = 0;
                  for (colpos = 0; colpos < console->dispcols; colpos++)
                  {
                    char c = console->buff[pos][colpos];
                    if (c == '\0')
                      break;
                    if (c == ' ' )
                      spcpos++;
                    else
                    {
                      linelen += spcpos+1;
                      spcpos = 0;
                    }
                  }
                  // copy the line to the buffer.
                  for (colpos = 0; colpos < linelen; colpos++)
                    copybuff[copylen++] = console->buff[pos][colpos];
                }
              }
              if (copylen)
              {
                if (!OpenClipboard(hwnd))
                  failmsg = "Unable to open clipboard for copy operation";
                else
                {
                  HGLOBAL hglb = GlobalAlloc(GMEM_MOVEABLE|GMEM_DDESHARE,copylen+1);
                  if (!hglb)
                    failmsg = "Unable to allocate global memory for copy operation";
                  else
                  {
                    #if defined(__WINDOWS_386__)
                    char far *lptstr = NULL;
                    void *_lptstr = (void *)GlobalLock(hglb);
                    if (_lptstr) lptstr = (char far *)(MK_FP32(_lptstr));
                    #else
                    char * lptstr = (char *)GlobalLock(hglb);
                    #endif
                    if (!lptstr)
                      failmsg = "Unable to lock global memory for copy operation";
                    else
                    {
                      for (pos=0;pos<copylen;pos++)
                        *lptstr++ = copybuff[pos];
                      *lptstr = 0;
                      GlobalUnlock(hglb);
                      EmptyClipboard();
                      if (!SetClipboardData(CF_TEXT,hglb))
                      {
                        failmsg = "Unable to set clipboard data";
                        GlobalFree(hglb);
                      }
                    }
                  }
                  CloseClipboard();
                }
              } /* if copylen */
              free((void *)copybuff);
            } /* if (buf) */
            if (failmsg)
            {
              MessageBox( NULL, failmsg, W32CLI_CONSOLE_NAME" ", MB_ICONHAND);
            }
            else
            {
              __clear_marked(console);
              if (!(console && console->rate_view.hwnd))
                InvalidateRect(hwnd,NULL,FALSE);
            }
          } /* if (console->have_marked) */
        }
        else if (console->keycount < kbuffsize)
        {
          // anything else.
          console->keybuff[console->keycount++] = (int) wParam;
          if (console->have_marked)
            __clear_marked(console);
        }
        else
        {
          // beep, keyboard buffer full
          MessageBeep(MB_OK);
        }
      }
      break;
    }
    case WM_SETFOCUS:
    {
      LRESULT lResult = DefWindowProc(hwnd, message, wParam, lParam);
      if (console)
        __win16AdjustCaret(console->hwnd, console, 0 );
      return lResult;
    }
    case WM_KILLFOCUS:
    {
      if (console)
        DestroyCaret();
      return DefWindowProc(hwnd, message, wParam, lParam);
    }
    case WM_USER_W16CONS:
    {
      if (wParam == W16CONS_CMD_ECHOLPARAM) /* identify me */
        return lParam;
      if (!console)
      {
        //__w16writelog("(WM_USER+0) 0x%04x called when no console.", (int)wParam);
        return -1;
      }
      switch (wParam)
      {
        case W16CONS_CMD_INDIRDESTROY: // we need this because we can't directly
        {                          // send a WM_DESTROY from another thread
          //SendMessage(hwnd, WM_DESTROY, 0, 0 );
          DestroyWindow(hwnd);
          return 0;
        }
        case W16CONS_CMD_CLEARSCREEN:
        {
          memset((void *)(&(console->buff[0][0])),' ',sizeof(console->buff));
          __clear_marked(console);
          console->currow = 0;
          console->curcol = 0;

          if (console->rate_view.hwnd)
            PostMessage(console->rate_view.hwnd,WM_COMMAND,WMCMD_REFRESHVIEW,0);
          else
          {
            InvalidateRect(hwnd, NULL, FALSE);
            UpdateWindow(hwnd);
          }
          break;
        }
        case W16CONS_CMD_PRINTSTR:
        {
          char ch;
          const char *text = (const char *)lParam;
          memcpy(&console->literal_buff[0],&console->buff[0][0],sizeof(console->buff));
          console->literal_buff_is_valid = 0;

          do{
            // check the validity of the current
            // cursor position, scrolling the window if necessary.

            if (console->currow < 0)
              console->currow = console->curcol = 0;
            else if (console->curcol < 0)
              console->curcol = 0;
            if (console->curcol >= W16CONS_WIDTH)
            {
              console->curcol = 0;
              console->currow++;
            }
            if (console->currow >= W16CONS_HEIGHT)
            {
              int row;
              memmove( &(console->buff[0][0]),
                      (&(console->buff[0][0])) + W16CONS_WIDTH,
                        sizeof(console->buff) - W16CONS_WIDTH );
              memset( &(console->buff[W16CONS_HEIGHT-1][0]),' ',W16CONS_WIDTH);
              if (console->have_marked)
              {
                if (console->mark_lastrow > 0)
                  --console->mark_lastrow;
                console->have_marked = 0;
                for (row = 0; row < (W16CONS_HEIGHT - 1); row++)
                {
                  console->marked_buff[row] = console->marked_buff[row+1];
                  if (console->marked_buff[row])
                    console->have_marked = 1;
                }
                console->marked_buff[W16CONS_HEIGHT-1]=0;
              }
              console->currow = W16CONS_HEIGHT - 1;
              console->curcol = 0;
            }

            //update the screen buffer

            ch = *text++;
            if (!ch)
              break;
            if (ch == 0x0A)         /* \n  new-line */
            {
              console->currow++; console->curcol = 0;
            }
            else if (ch == 0x0D)   /* \r    carriage return */
            {
              console->curcol = 0;
            }
            else if (ch == '\t')   /* \t    horizontal tab  */
            {
              console->curcol += (console->curcol%8);
            }  
            else if (ch == '\a')   /* \a  bell (alert)      */
              MessageBeep(MB_OK);
            else if (ch == '\f')   /* \f    form-feed       */
            {
              memset((void *)(&(console->buff[0][0])),' ',sizeof(console->buff));
              console->currow = 0;
              console->curcol = 0;
            }
            else if (ch == '\b')   /* \b  backspace         */
            { 
              if (console->curcol > 0)
              {
                console->curcol--; 
              }
            }
            else if (ch == '\v')   /* \v  vertical tab      */
            {
              console->currow++;
            }
            else if (ch <= 26) /* don't print other ctrl-chars */
              ;
            else
            {
              console->buff[console->currow][console->curcol++] = ch;
            }
          } while (ch); /* always true */

          if (memcmp( &console->literal_buff[0],  &console->buff[0][0],
                      sizeof(console->buff))!=0)
          {
            if (console->rate_view.hwnd)
              PostMessage(console->rate_view.hwnd,WM_COMMAND,WMCMD_REFRESHVIEW,0);
            else
            {
              // force repaint
              InvalidateRect(hwnd, NULL, FALSE);
              //UpdateWindow(hwnd);
            }
          }
          break;
        }
        case W16CONS_CMD_ISKBHIT:
        {
          return ((console->keycount > 0)?(1):(0));
        }
        case W16CONS_CMD_GETSIZE:
        {
          return MAKELRESULT(W16CONS_HEIGHT,W16CONS_WIDTH);
        }
        case W16CONS_CMD_GETPOS:
        {
          return MAKELRESULT(console->currow,console->curcol);
        }
        case W16CONS_CMD_SETPOS:
        {
          int row = LOWORD(lParam);
          int col = HIWORD(lParam);

          if (row < 0)
            row = 0;
          else if (row >= W16CONS_HEIGHT)
            row = W16CONS_HEIGHT-1;
          if (col < 0)
            col = 0;
          else if (col >= W16CONS_WIDTH)
            col = W16CONS_WIDTH-1;

          console->currow = row;
          console->curcol = col;

          if (hwnd == GetFocus())
            __win16AdjustCaret(hwnd, console, 1 );
          break;
        }
        case W16CONS_CMD_GETCH:
        {
          LRESULT keyval = -1L;
          if (console->keycount > 0)
          {
            int c = console->keybuff[0];
            keyval = (LRESULT)(c & 0xff);
            if (!keyval)
              console->keybuff[0] = (((int)(c >> 8)) & 0xff);
            else
            {
              for (c = 1; c < console->keycount; c++)
                console->keybuff[c-1] = console->keybuff[c];
              --console->keycount;
            }
          }
          return keyval;
        }
      } /* switch (wParam) */
    }
    default:
    {
      #if (CLIENT_OS == OS_WIN32)
      static UINT taskbarCreatedMsg = WM_USER;
      if (taskbarCreatedMsg == WM_USER)
      {
        taskbarCreatedMsg = 0;
        if ((winGetVersion() % 2000) >= 400)
        {
          taskbarCreatedMsg = RegisterWindowMessage("TaskbarCreated");
        }
      }
      if (taskbarCreatedMsg && message == taskbarCreatedMsg)
      {
        __DoTrayStuff(hwnd, +1,  NULL /* tip */,"TaskbarCreated" );
        return 0;
      }
      #endif

      TRACE_FLOW((0, "defwndproc: msg: 0x%04x, wParam: 0x%04x, lParam: 0x%08x\n", message, wParam, lParam ));
      return DefWindowProc(hwnd, message, wParam, lParam);
    }
  }
  return 0;
}

LRESULT CALLBACK __w16WindowFunc(HWND hwnd, UINT message,
                                 WPARAM wParam, LPARAM lParam)
{
  static nestinglevel = 0;
  LRESULT lResult;

  ++nestinglevel;
  TRACE_FLOW((+1, "__w16WindowFunc: nesttinglevel=%d, msg: 0x%04x, wParam: 0x%04x, lParam: 0x%08x\n", nestinglevel, message, wParam, lParam ));
  lResult = __w16WindowFuncInternal(nestinglevel, hwnd, message, wParam, lParam );
  TRACE_FLOW((-1, "__w16WindowFunc: lResult=%ld (0x%lx)\n", lResult, lResult ));
  --nestinglevel;

  return lResult;
}

/* ------------------------------------------------ */

static struct
{
  HWND    hwndList[1];
  char    szClassName[32];
  HANDLE  hmutex;
  int     hidden;
  FILE    *fstdout;
  FILE    *fstdin;
  int     iconisstock;
  HICON   hIcon;
  int     errorcode;
  int     asthread;
  int     debugon;
  int     nativecons;
  int     client_run_startstop_level; //inc on ClientRun start, dec on stop
  void   *devpipe;   /* named pipe (r/w) or output end of anon pipe */
  void   *devpipein; /* NULL if named pipe or input end of anon pipe */
  HWND    shimwatcher;
  HINSTANCE ctl3d;
} constatics =
{
  {NULL},   //hwndList[1];
  {0},      //szClassName[32];
  NULL,     //hmutex;
  0,        //hidden;
  NULL,     //FILE *fstdout;
  NULL,     //FILE *fstdin;
  0,        //iconisstock;
  NULL,     //hIcon;
  0,        //errorcode;
  0,        //asthread;
  0,        //debugon;
  0,        //nativecons;
  0,        //client_run_startstop_level,
  NULL,     //devpipe;
  NULL,     //devpipein;
  NULL,     //shimwatcher;
  NULL      //ctl3d
};

static BOOL my_IsDialogMessage(HWND hwnd, MSG *msg)
{
  if (hwnd && hwnd != constatics.hwndList[0])
  {
    #if defined(GCW_STYLE) && !defined(GCL_STYLE)
    #define GCL_STYLE GCW_STYLE
    #endif
    BOOL needcheck = ((GetClassLong(hwnd, GCL_STYLE) & CS_GLOBALCLASS)!=0);
    //if (!needcheck)
    //  needcheck = (GetClassWord(hwnd, GCW_ATOM) == 32770);
    //if (!needcheck && (GetWindowLong(hwnd, GWL_STYLE) & WS_CHILD)!=0)
    //{
    //  needcheck = (GetClassWord(GetParent(hwnd), GCW_ATOM) == 32770);
    //  if (!needcheck)
    //    
    //}
    if (needcheck)
      return IsDialogMessage(hwnd,msg);
  }
  return 0;
}

static void w16Yield(void)
{
  MSG msg;
  #if (CLIENT_OS == OS_WIN32)
  if (constatics.asthread || constatics.devpipe || constatics.nativecons)
  {
    Sleep(1);
    return;
  }
  #endif
  while (PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE ))
  {
    if (!GetMessage(&msg, NULL, 0, 0))
      break;
    if (!my_IsDialogMessage(msg.hwnd,&msg))
    {
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
  }
  return;
}

static void w16Sleep(unsigned int millisecs)
{
  DWORD last = 0; MSG msg;
  #if (CLIENT_OS == OS_WIN32)
  if (constatics.asthread || constatics.devpipe || constatics.nativecons)
  {
    Sleep(millisecs);
    return;
  }
  #endif
  if (millisecs > 50)
  {
    UINT hTimer = (UINT)SetTimer(NULL, 0, (UINT)millisecs, NULL);
    if (hTimer)
    {
      for (;;)
      {
        if (!GetMessage(&msg, NULL, 0, 0))
          break;
        if (msg.message == WM_TIMER && (!msg.wParam || msg.wParam == hTimer)) /* THIS timer */
          break;
        if (!my_IsDialogMessage(msg.hwnd,&msg))
        {
          TranslateMessage(&msg);
          DispatchMessage(&msg);
        }
      }
      KillTimer(NULL,hTimer);
      return;
    }
  }
  do
  {
    DWORD now;
    int gotmsg;
    if (millisecs)
    {
      now = GetTickCount();
      if (last) 
      {
        DWORD elapsed = now - last;
        if (now < last)
          elapsed = (now + ( 0xfffffffful - last ) + 1);
        if (elapsed > millisecs)
          elapsed = millisecs;
        millisecs -= elapsed;
      }
      last = now;
    }
    gotmsg = 0; 
    while (PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE ))
    {
      if (!GetMessage(&msg, NULL, 0, 0))
      {
        gotmsg = -1;
        break;
      }
      gotmsg = +1;
      if (!my_IsDialogMessage(msg.hwnd,&msg))
      {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
      }
    }
    if (gotmsg < 0)
      break;
    if (gotmsg == 0)
    {
      #if (CLIENT_OS == OS_WIN32)
      if (winGetVersion() >= 400) /* not win32s */
      {
        Sleep(50);
      }
      else
      #endif
      {
        for (;;)
        { 
          if (GetQueueStatus(QS_KEY|QS_MOUSE|QS_PAINT|QS_POSTMESSAGE|
                QS_SENDMESSAGE|QS_TIMER) != 0)
            break;
          now = GetTickCount();
          if (now < last || now > (last+50))
            break;
          #if (CLIENT_OS == OS_WIN32)
          Sleep(0); /* win32s */
          #else
          Yield();Yield();Yield();Yield();Yield();
          #endif
        }
      }
    }
  } while (millisecs);
  return;
}

/* ------------------------------------------------ */

struct helperArg
{
  int nCmdShow;
  int nSuccess;
  int nErrorCode;
  int asthread;
  HWND hwnd;
  HINSTANCE hInstance;
};

/* ------------------------------------------------ */

static void __win16WinCreateHelper( void *xarg )
{
  struct helperArg *arg = (struct helperArg *)(xarg);
  struct WM_CREATE_DATA createdata;
  int asthread = arg->asthread;
  HINSTANCE hInstance = arg->hInstance;
  HWND hwnd;

  if (asthread)
  {
    SetGlobalPriority(9); /* classprio=idle, threadprio=highest */
  }

  createdata.nCmdShow = arg->nCmdShow;
  createdata.create_pending = 1;
  createdata.create_errorcode = W16CONS_ERR_NCCREATE;
  createdata.client_run_startstop_level_ptr = 
                          &constatics.client_run_startstop_level;

  /* don't use WS_EX_CLIENTEDGE - the client will do this itself */
  /* don't use any WS_EX_* that Win(NT)3.x doesn't explicitely support */
  TRACE_INITEXIT((+0,"begin createwindow\n" ));
  hwnd = CreateWindow( constatics.szClassName, 
                       (W32CLI_CONSOLE_NAME" "),
                       (WS_OVERLAPPEDWINDOW|WS_VSCROLL),
                       CW_USEDEFAULT, CW_USEDEFAULT, 0, 0,
                       NULL, NULL, hInstance, &createdata );
  TRACE_INITEXIT((+0,"end createwindow\n" ));

  if (hwnd == NULL)
  {
    if (createdata.create_errorcode == 0)
      createdata.create_errorcode = W16CONS_ERR_CREATEWIN;
    arg->nErrorCode = createdata.create_errorcode;
    arg->nSuccess = 0;
    return;
  }

  while (createdata.create_pending)
  {
    #if (CLIENT_OS == OS_WIN32)
    Sleep(1);
    #else
    Yield();
    #endif
  }

  if (createdata.create_errorcode)
  {
    DestroyWindow(hwnd);
    arg->nErrorCode = createdata.create_errorcode;
    arg->nSuccess = 0;
    return;
  }

  arg->hwnd = hwnd;
  arg->nErrorCode = 0;
  arg->nSuccess = 1;

  if (asthread)
  {
    MSG msg;
    while (GetMessage(&msg, (HWND) NULL, 0, 0))
    {
      if (!my_IsDialogMessage(msg.hwnd,&msg))
      {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
      }
    }
  }

  return;
}

/* ------------------------------------------------ */

static HWND w16ConsoleCreate(int nCmdShow)
{
  struct helperArg arg;
  int newclass = 0;
  WNDCLASS wcl;
  memset((void *)&wcl,0,sizeof(wcl));

  constatics.errorcode = 0;
  arg.hInstance = winGetInstanceHandle(); /* w32pre.cpp */

  if (!arg.hInstance)
  {
    constatics.errorcode = W16CONS_ERR_GETINST;
    return 0;
  }
  if (constatics.hwndList[0])
  {
    constatics.errorcode = W16CONS_ERR_NOSLOT;
    return 0;
  }

  if ((winGetVersion()%2000)<400)
  {
    UINT olderrmode = SetErrorMode(SEM_NOOPENFILEERRORBOX);
    #if (CLIENT_OS == OS_WIN32)
    constatics.ctl3d = LoadLibrary("ctl3d32.dll");
    #elif 0
    constatics.ctl3d = LoadLibrary("ctl3dv2.dll");
    if (constatics.ctl3d < ((HINSTANCE)32))
      constatics.ctl3d = LoadLibrary("ctl3d.dll");
    #endif
    SetErrorMode(olderrmode);
    if (constatics.ctl3d < ((HINSTANCE)32))
      constatics.ctl3d = NULL;
    if (constatics.ctl3d)
    {
      FARPROC proc = GetProcAddress(constatics.ctl3d,"Ctl3dRegister");
      if (proc)
      {
        #if defined(__WINDOWS_386__)
        _Call16(proc,"w",arg.hInstance);
        #else
        (*((BOOL (WINAPI *)(HINSTANCE))proc))(arg.hInstance);
        #endif
        proc = GetProcAddress(constatics.ctl3d,"Ctl3dAutoSubclass");
        if (proc)
        {
          #if defined(__WINDOWS_386__)
          _Call16(proc,"w",arg.hInstance);
          #else
          (*((BOOL (WINAPI *)(HINSTANCE))proc))(arg.hInstance);
          #endif
        }
      }
    }
  }

  TRACE_INITEXIT((0,"begin register class\n"));

  if (constatics.szClassName[0]==0)
  {
    strcpy(constatics.szClassName,"DCTICLI");
    #ifdef __WINDOWS_386__ //win16 pmode - cannot share memory
    sprintf( constatics.szClassName, "DCTI%u", arg.hInstance );
    #endif

    constatics.hIcon = LoadIcon(arg.hInstance, MAKEINTRESOURCE(1));
    constatics.iconisstock = 0;
    if (!constatics.hIcon)
    {
      LoadIcon(NULL, MAKEINTRESOURCE(IDI_APPLICATION));
      constatics.iconisstock = 1;
    }

    /* define a window class */
    wcl.hInstance = arg.hInstance;
    wcl.lpszClassName = constatics.szClassName;
    wcl.lpfnWndProc = (WNDPROC)__w16WindowFunc;
    wcl.style = CS_HREDRAW | CS_VREDRAW; /* caution! don't BYTEALIGNCLIENT */
                 /* (CS_HREDRAW | CS_VREDRAW | CS_SAVEBITS |
                 CS_BYTEALIGNCLIENT | CS_BYTEALIGNWINDOW); */
    wcl.hIcon = constatics.hIcon;
    wcl.hCursor = LoadCursor(NULL, IDC_ARROW);
    wcl.lpszMenuName = NULL;
    wcl.cbClsExtra = 0;
    wcl.cbWndExtra = sizeof(void *); /* we need space for a pointer */
    wcl.hbrBackground = NULL; //(HBRUSH)GetStockObject(NULL_BRUSH);

    /* register the window class */
    if (!RegisterClass(&wcl))
    {
      constatics.szClassName[0]=0;
      if (!constatics.iconisstock)
        DestroyIcon( constatics.hIcon );
      constatics.hIcon = NULL;
      constatics.errorcode = W16CONS_ERR_REGCLASS;
      return 0;
    }
    newclass = 1;
  }

  TRACE_INITEXIT((0,"end register class\n"));

  arg.nSuccess = -1;
  arg.nCmdShow = nCmdShow;
  arg.asthread = 0;

  #if (CLIENT_OS == OS_WIN32)
  constatics.asthread = arg.asthread = 1;
  if (_beginthread( __win16WinCreateHelper, 0/*(1024*32)*/, (void *)&arg  ) == NULL)
  {
    constatics.asthread = arg.asthread = 0;
  }
  #endif
  if (arg.asthread == 0)
  {
    __win16WinCreateHelper( (void *)&arg );
  }
  while (arg.nSuccess == -1)
    w16Sleep(100);

  if (arg.nSuccess == 0)
  {
    constatics.errorcode = arg.nErrorCode;
    if (newclass && constatics.szClassName[0])
    {
      if (UnregisterClass( constatics.szClassName, arg.hInstance ) == 0)
      {
        constatics.szClassName[0] = 0;
        if (constatics.hIcon && !constatics.iconisstock)
          DestroyIcon( constatics.hIcon );
        constatics.hIcon = NULL;
      }
    }
    constatics.asthread = 0;
    return 0;
  }

  constatics.hwndList[0] = arg.hwnd;
  return arg.hwnd;
}

/* ------------------------------------------------ */

static int w16ConsoleDestroy(void)
{
  int rc = 0;
  HWND hwnd = constatics.hwndList[0];
  if (hwnd)
  {
    constatics.hwndList[0] = NULL;
    //this way because we can't directly send a WM_DESTROY to another thread
    SendMessage( hwnd, WM_USER_W16CONS, W16CONS_CMD_INDIRDESTROY, 0 );

    #if 0 /* doesn't work. why? */
    if (constatics.szClassName[0])
    {
      if (UnregisterClass( constatics.szClassName, winGetInstanceHandle())==0)
      {
        constatics.szClassName[0] = 0;
        if (constatics.hIcon && !constatics.iconisstock)
          DestroyIcon( constatics.hIcon );
        constatics.hIcon = NULL;
      }
    }
    #endif
    rc = 1;
  }
  if (constatics.ctl3d)
  {
    FARPROC proc = GetProcAddress(constatics.ctl3d,"Ctl3dUnregister");
    if (proc)
    {
      #if defined(__WINDOWS_386__)
      _Call16(proc,"w", winGetInstanceHandle());
      #else
      (*((BOOL (WINAPI *)(HINSTANCE))proc))(winGetInstanceHandle());
      #endif
    }
    FreeLibrary(constatics.ctl3d);
    constatics.ctl3d = NULL;
  }
  return rc;
}

/* ------------------------------------------------ */

static void w16SetConsoleTitle(const char *name)
{
  w16Yield();
  if (constatics.hwndList[0] == NULL)
    return;
  SetWindowText( constatics.hwndList[0], (LPSTR)name );
}

/* ------------------------------------------------ */

static int w16HaveWindow(void)
{
  return (constatics.hwndList[0] != NULL);
}

/* ------------------------------------------------ */

static int w16ConsoleClear(void)
{
  w16Yield();
  if (constatics.hwndList[0] == NULL)
    return -1;
  SendMessage( constatics.hwndList[0], WM_USER_W16CONS, W16CONS_CMD_CLEARSCREEN, 0 );
  return 0;
}

/* ------------------------------------------------ */

static int w16ConsolePrint(const char *text)
{
  unsigned int len;
  w16Yield();
  if (constatics.hwndList[0] == NULL)
    return -1;
  if (!text)
    return -1;
  if ((len = strlen(text)) == 0)
    return 0;
  SendMessage( constatics.hwndList[0], WM_USER_W16CONS, W16CONS_CMD_PRINTSTR, (LPARAM)(text) );
  return len;
}

/* ------------------------------------------------ */

int w16ConSetPos(int col, int row)
{
  w16Yield();
  if (constatics.hwndList[0] == NULL)
    return -1;
  SendMessage( constatics.hwndList[0], WM_USER_W16CONS, W16CONS_CMD_SETPOS, MAKELONG(row, col));
  return 0;
}

/* ------------------------------------------------ */

static int w16ConGetPos(int *col, int *row)
{
  w16Yield();
  if (constatics.hwndList[0] == NULL)
    return -1;
  LRESULT cr = SendMessage( constatics.hwndList[0], WM_USER_W16CONS, W16CONS_CMD_GETPOS, 0);
  if (col) *col = HIWORD(cr);
  if (row) *row = LOWORD(cr);
  return 0;
}

/* ------------------------------------------------ */

static int w16ConGetSize( int *width, int *height)
{
  if (constatics.hwndList[0] == NULL)
    return -1;
  LRESULT wh = SendMessage( constatics.hwndList[0], WM_USER_W16CONS, W16CONS_CMD_GETSIZE, 0);
  if (width)  *width  = HIWORD(wh);
  if (height) *height = LOWORD(wh);
  return 0;
}

/* ------------------------------------------------ */

static int w16ConsoleKbhit(void)
{
  w16Yield();
  if (constatics.hwndList[0] == NULL)
    return 0;
  return SendMessage( constatics.hwndList[0], WM_USER_W16CONS, W16CONS_CMD_ISKBHIT, 0 );
}

/* ------------------------------------------------ */

static int w16ConsoleGetch(void)
{
  LRESULT keyval = 0;
  while (constatics.hwndList[0])
  {
    if (SendMessage( constatics.hwndList[0], WM_USER_W16CONS, W16CONS_CMD_ISKBHIT, 0 ))
    {
      keyval = SendMessage( constatics.hwndList[0], WM_USER_W16CONS, W16CONS_CMD_GETCH, 0 );
      if (keyval != -1L) /* huh? no key! */
        break;
      keyval = 0;
    }
    w16Yield();
  }
  return (int)keyval;
}

/* ===================================================== */
/* ************* END OF GUI PRIMITIVES ***************** */
/* ===================================================== */

#if (CLIENT_OS == OS_WIN32)
static HANDLE __pipe_gethandle(int which)
{
  if (which == STD_INPUT_HANDLE)
  {
    if (constatics.devpipein) /* input end of anon pipe */
      return (HANDLE)constatics.devpipein;
  }
  return (HANDLE)constatics.devpipe;  /* pipe is r/w named pipe */
}
static int __pipe_clear(void)
{
  char buffer[64];
  HANDLE hIn = __pipe_gethandle(STD_INPUT_HANDLE);
  DWORD bytesRead, totalBytesAvail;
  TRACE_PIPE((+1,"__pipe_clear()\n"));
  if (!PeekNamedPipe( hIn, (LPVOID)&buffer[0], sizeof(buffer),
                      &bytesRead, &totalBytesAvail, 0 ))
  {
    if (GetLastError() == ERROR_BROKEN_PIPE)
    {
      RaiseExitRequestTrigger(); /* sigpipe */
      return -1;
    }
    totalBytesAvail = 0;
  }
  TRACE_PIPE((0,"__pipe_clear(). totalavail=%u\n",totalBytesAvail));
  while (totalBytesAvail)
  {
    bytesRead = sizeof(buffer);
    if (bytesRead > totalBytesAvail)
      bytesRead = totalBytesAvail;
    TRACE_PIPE((0,"__pipe_clear(). toread=%u\n",bytesRead));
    if (!ReadFile(hIn, buffer, sizeof(buffer), &bytesRead, 0 ))
    {
      RaiseExitRequestTrigger(); /* sigpipe */
      return -1;
    }
    if (!bytesRead)
      break;
    totalBytesAvail -= bytesRead;
  }
  TRACE_PIPE((-1,"__pipe_clear()\n"));
  return 0;
}
static int __pipe_kbhit(void)
{
  char buffer[8];
  DWORD bytesRead, totalBytesAvail, bytesLeftThisMessage;
  if (!PeekNamedPipe(__pipe_gethandle(STD_INPUT_HANDLE),
                       (LPVOID)&buffer[0],
                       sizeof(buffer),
                       &bytesRead,
                       &totalBytesAvail,
                       &bytesLeftThisMessage ))
  {
    if (GetLastError() == ERROR_BROKEN_PIPE)
    {
      RaiseExitRequestTrigger(); /* sigpipe */
      return -1;
    }
    TRACE_PIPE((0,"__pipe_kbhit: peekpipe failed\n"));
  }
  else if (totalBytesAvail)
    return 1;
  return 0;
}
static int __pipe_getchar(int *chP)
{
  int ch, rc;
TRACE_PIPE((+1,"__pipe_getchar()\n"));
  rc = __pipe_kbhit();
  if (rc > 0)
  {
    char buffer[8];
    DWORD bytesRead = 0;
    TRACE_PIPE((+1,"ReadFile(1)\n"));
    BOOL ok = ReadFile(__pipe_gethandle(STD_INPUT_HANDLE),
                  (LPVOID)&buffer[0], 1, &bytesRead, NULL);
    TRACE_PIPE((-1,"ReadFile()=>%s\n",(ok)?("ok"):("fail") ));
    if (!ok)
    {
      if (GetLastError() == ERROR_BROKEN_PIPE)
      {
        TRACE_PIPE((-1,"BROKEN_PIPE\n" ));
        RaiseExitRequestTrigger(); /* sigpipe */
        return -1;
      }
      rc = 0;
    }
    else 
    {    
      ch = (((int)buffer[0]) & 0xff);
      TRACE_PIPE((0, "__pipe_getchar: %c (%d,0x%02x)\n", ch, ch, ch ));
      if (chP)
        *chP = ch;
      rc = +1;
    }
  } 
TRACE_PIPE((-1,"__pipe_getchar()=>%d\n",rc));
  return rc;
}

static int __pipe_puts_noclear(const char *msg,unsigned int len)
{
  int totalBytesWritten = 0;
  HANDLE hPipe = __pipe_gethandle(STD_OUTPUT_HANDLE);
TRACE_PIPE((+1,"__pipe_puts_noclear('%s',%u)\n",msg, len));
  do
  {
    DWORD numberOfBytesWritten = 0;
TRACE_PIPE((+1,"1. WriteFile(...) towrite = %u\n", len));
    BOOL ok = WriteFile(hPipe, (LPCVOID)msg, len, 
         &numberOfBytesWritten, NULL);
TRACE_PIPE((-1,"1. WriteFile(...)=>%s, numwritten=%u\n", (ok)?("ok"):("fail"), numberOfBytesWritten ));
    if (!ok)
    {
      // win9x has a broken pipe implementation which causes WriteFile to 
      // _sometimes_ fail with an invalid file handle.
      if (GetLastError() == ERROR_BROKEN_PIPE)
      {
TRACE_PIPE((0,"ERROR_BROKEN_PIPE\n"));
        RaiseExitRequestTrigger(); /* sigpipe */
        return -1;
      }
TRACE_PIPE((+1,"FlushFileBuffers()\n"));
      FlushFileBuffers(hPipe); //block
TRACE_PIPE((-1,"FlushFileBuffers()\n"));
TRACE_PIPE((+1,"2. WriteFile(...) towrite = %u\n", len));
      ok = WriteFile(hPipe, (LPCVOID)msg, len, &numberOfBytesWritten, NULL);
TRACE_PIPE((-1,"2. WriteFile(...)=>%s, numwritten=%u\n", (ok)?("ok"):("fail"), numberOfBytesWritten ));
      if (!ok)
        break;
    }
    totalBytesWritten += numberOfBytesWritten;
    msg += numberOfBytesWritten;
    len -= numberOfBytesWritten;
  } while (len);
TRACE_PIPE((-1,"__pipe_puts_noclear()=>%u\n", totalBytesWritten));
  return totalBytesWritten;
}
static int __pipe_puts(const char *msg,unsigned int len)
{
  int rc = 0;
TRACE_PIPE((+1,"__pipe_puts()\n"));
  if (len) /* avoid overhead */
    rc = __pipe_puts_noclear(msg,len);
  if (rc >= 0)
    __pipe_clear(); /* flush input pipe */
TRACE_PIPE((-1,"__pipe_puts()=>%u\n", rc));
  return rc;
}
static int __pipe_putchar(int ch)
{
  char msg;
  msg = (char)ch;
  if (__pipe_puts(&msg,1) <= 0)
    return -1;
  return 1;
}
/*
// position and size is always one based,
// unless no tty, in which case the pipe server ansi returns 0,0
*/
static int __pipe_getsizeorposisatty(int assize, int *width, int *height, 
                                     int *istty)
{
  static int __isatty_asserted = -1; /* not yet asserted */
  int rc = -1;
TRACE_PIPE((+1,"__pipe_getsizeorposisatty(%d, %p, %p, %p)\n",assize,width,height,istty));
  if (__isatty_asserted == 0)
  {
    if (width) *width = 0;
    if (height) *height = 0;
    if (istty) *istty = 0;
    rc = 0;
  }
  else
  {
    int broken_pipe = 0, savex = 0, xy = ((width)?(0):(1));
    for (;rc != 0 && !broken_pipe && xy < 2;xy++)
    {
      const char *msg;
TRACE_PIPE((+1,"flush input pipe with __pipe_getchar(NULL)\n"));
      __pipe_clear(); /* flush input pipe */
TRACE_PIPE((-1,"flush input pipe with __pipe_getchar(NULL)\n"));
      if (assize) /* getsize */
        msg = ((xy == 0)?("\x1B""[X"):("\x1B""[Y"));
      else /* getpos */
        msg = ((xy == 0)?("\x1B""[x"):("\x1B""[y"));
TRACE_PIPE((0,"__pipe_getsizeorposisatty: __pipe_puts_noclear()\n"));
      if (__pipe_puts_noclear(msg,3) != 3)
      {
        broken_pipe = 1;
        break; /* broken pipe */
      } 
      while (!CheckExitRequestTriggerNoIO())
      {
        int ch, kbstate;
TRACE_PIPE((0,"__pipe_getsizeorposisatty: __pipe_getchar()\n"));
        if ((kbstate = __pipe_getchar(&ch)) < 0)
        {
          broken_pipe = 1;
          break; /* return -1; broken pipe */
        }
        if (kbstate != 0) /* got a char */
        {
TRACE_PIPE((0,"got a char ch = %d\n", ch));
          if (__isatty_asserted < 0) /* haven't determined this yet */
          {
            __isatty_asserted = (ch ? 1 : 0);
TRACE_PIPE((0,"__isatty_asserted = %d\n", __isatty_asserted));
            if (__isatty_asserted == 0)
            {
              if (height) *height = 0;
              if (width) *width = 0;
              if (istty) *istty = 0;
              rc = 0;
              break;
            }
          }
          if (xy == 0) /* width loop */
          {
            if (ch == 0 || !height)
            {
              if (height) *height = 0;
              if (width)  *width = ch;
              if (istty)  *istty = 1;
              rc = 0; /* we're done */
            }
            savex = ch;
          }
          else           /* height loop */
          {
            if (height) *height = ch;
            if (width) *width = savex;
            if (istty) *istty = 1;
            rc = 0; /* we're done */
          }
TRACE_PIPE((0,"got an answer\n"));
          break; /* got an answer, don't wait */
        }
TRACE_PIPE((0,"begin sleep\n"));
        Sleep(10);
TRACE_PIPE((0,"end sleep\n"));
      } /* while (!CheckExitRequestTriggerNoIO()) */
    }
  }
TRACE_PIPE((-1,"__pipe_getsizeorposisatty()=>%d\n",rc));
  return rc;
}
static int __pipe_isatty(void) /* < 0 = err, 0=no, >0=yes */
{
  int istty = 0;
TRACE_PIPE((+1,"__pipe_isatty()\n"));
  if (__pipe_getsizeorposisatty(0, 0, 0,  &istty) < 0)
    istty = -1;
  else if (istty)
    istty = +1;
TRACE_PIPE((-1,"__pipe_isatty() =>%d\n",istty));
  return istty;
}
static int __pipe_getxy(int *col, int *row)
{
  int istty = 0, r, c;
  if (__pipe_getsizeorposisatty(0,((col)?(&c):(0)),((row)?(&r):(0)),&istty)<0)
    return -1;
  if (!istty)  /* pos will be zero if no tty */
    return -1;
  if (col) *col = c-1; /* zero base results */
  if (row) *row = r-1;
  return 0;
}
static int __pipe_getwinsz(int *wP, int *hP)
{
  int istty = 0, w, h;
  if (__pipe_getsizeorposisatty(1,((wP)?(&w):(0)),((hP)?(&h):(0)),&istty)<0)
    return -1;
  if (!istty)  /* size will be zero if no tty */
    return -1;
  if (wP) *wP = w;
  if (hP) *hP = h;
  return 0;
}
/* pipe server supports an internal command set similar to ansi:
 * but instead of "esc[num;num'cmd" it uses "esc]cmd'opt...\0"
 * making it possible to send virtually anything
*/
static int __pipe_set_title(const char *title)
{
  char cmdbuffer[128];
  cmdbuffer[0]=((char)0x1B);
  cmdbuffer[1]=']'; /* note reversed bracket */
  cmdbuffer[2]='1'; /* cmd: setconsoletitle */
  strncpy(&cmdbuffer[3],title,sizeof(cmdbuffer)-3);
  cmdbuffer[sizeof(cmdbuffer)-1] = '\0'; /* also, our command char is '\0' */
  return __pipe_puts(cmdbuffer,2+strlen(&cmdbuffer[1]));
}
static int __pipe_advertise_hwnd(HANDLE hwnd)
{
  char cmdbuffer[128];
  cmdbuffer[0]=((char)0x1B);
  cmdbuffer[1]=']'; /* note reversed bracket */
  cmdbuffer[2]='2'; /* cmd: advertise handle */
  sprintf(&cmdbuffer[3],"%lu",(unsigned long)hwnd);
  return __pipe_puts(cmdbuffer,2+strlen(&cmdbuffer[1]));
}
static int __pipe_detach(void)
{
  return __pipe_putchar(0x03);
}
static void __pipe_sleep(unsigned int millisecs)
{
  Sleep(millisecs); /* nothing yet */
}
#endif

/* ===================================================== */
/* ************* END OF PRIMITIVES ********************* */
/* ===================================================== */

static void __ClientEventCallback(int event_id, const void *parm, int isize)
{
  if (event_id == CLIEVENT_CLIENT_RUNIDLE ||
      event_id == CLIEVENT_BUFFER_UPDATEBEGIN ||
      event_id == CLIEVENT_BUFFER_UPDATEEND ||  
      event_id == CLIEVENT_CLIENT_CRUNCHOMETER)
  {  
    if (constatics.hwndList[0])
    {
      struct WMCMD_EVENT_DATA evdata;
      evdata.id = event_id;
      evdata.parm = parm;
      evdata.isize = isize;
      /* this has to be SendMessage so that this (the main) thread */
      /* blocks while the graph data is updated */
      /* UPDATEBEGIN/END are used by the win16 client that needs to */
      /* disable the graph while an update is in progress */
      SendMessage( constatics.hwndList[0], WM_COMMAND, WMCMD_EVENT, (LPARAM)&evdata );
    }
  }
  else if ( event_id == CLIEVENT_CLIENT_RUNSTARTED)
    constatics.client_run_startstop_level++;
  else if (event_id == CLIEVENT_CLIENT_RUNFINISHED)
    constatics.client_run_startstop_level--;
  return;
}

/* =============================================================== */

int w32DeinitializeConsole(int pauseonclose)
{
  pauseonclose = pauseonclose;
  #if (CLIENT_OS == OS_WIN32)
  if (constatics.hmutex != NULL)
  {
    ReleaseMutex( constatics.hmutex );
    CloseHandle( constatics.hmutex );
    constatics.hmutex = NULL;
  }
  if (constatics.shimwatcher != NULL)
  {                                      // self destroying
    int sleeploops = 0;
    SendMessage( constatics.shimwatcher, WM_USER_W16CONS, W16CONS_CMD_INDIRDESTROY, 0 );
    while ((++sleeploops) < 20 && constatics.shimwatcher)
      Sleep(100);
  }
  if (constatics.devpipe)
  {
    __pipe_set_title(utilGetAppName());
    if (!constatics.devpipein) /* pipe is named pipe */
      CloseHandle(constatics.devpipe);
    else /* pipe is pipe pair of anonpipe */
      constatics.devpipein = NULL;
    constatics.devpipe = NULL;
  }
  else if (constatics.nativecons)
  {
    SetConsoleTitle(utilGetAppName());
    if (constatics.fstdout || constatics.fstdin)
      FreeConsole();
    if (constatics.fstdout)
      fclose(constatics.fstdout);
    constatics.fstdout = NULL;
    if (constatics.fstdin)
      fclose(constatics.fstdin);
    constatics.fstdin = NULL;
  }
  else
  #endif
  {
    if (w16HaveWindow())
    {
      if (pauseonclose)
      {
        int init = 0;
        time_t nowtime = 0, endtime = 0;
        int row = -1, height = 0;
        w16ConGetPos(NULL, &row);
        w16ConGetSize(NULL, &height);
        if (height > 2 && row != -1)
          w16ConSetPos(0, height-((row<(height-2))?(3):(1)));
        do
        {
          int sleeploops;
          nowtime = time(NULL);
          if (endtime == 0)
            endtime = nowtime + 15;
          for (sleeploops = 0;sleeploops < ((!init)?(1):(4));sleeploops++)
          {
            if (sleeploops)
              w16Sleep(250);
            if (w16ConsoleKbhit() || CheckExitRequestTriggerNoIO())
            {
              nowtime = endtime;
              break;
            }
          }
          if (nowtime < endtime)
          {
            char buffer[80];
            sprintf( buffer, "%sPress any key to continue... %d  ",
                     ((!init)?("\n\n"):("\r")), (int)(endtime-nowtime) );
            init = 1;
            w16ConsolePrint( buffer );
          }
        } while (nowtime < endtime);
      }
      w16ConsoleDestroy();
    }
  }
  ClientEventRemoveListener(-1,__ClientEventCallback);

  return 0;
}


/* ---------------------------------------------------- */

#if (CLIENT_OS == OS_WIN32)
#if defined(__WATCOMC__)
static void __w32SigTriggerControl(int sig)
{
  signal(sig,__w32SigTriggerControl);
  RaiseExitRequestTrigger();
  return;
}
#endif
static BOOL WINAPI __w32NativeTriggerControl(DWORD dwCtrlType)
{
  if (dwCtrlType == CTRL_LOGOFF_EVENT && win32CliServiceRunning())
  {
    //should never happen since we don't open a console if running as a service
    return TRUE;
  }
  else if (dwCtrlType == CTRL_BREAK_EVENT)
  {
    RaiseRestartRequestTrigger();
    return TRUE;
  }
  else if ( dwCtrlType == CTRL_C_EVENT )
  {
    RaiseExitRequestTrigger();
    return TRUE;
  }
  else if ( dwCtrlType == CTRL_CLOSE_EVENT || /* totally fscked on Win9x */
            dwCtrlType == CTRL_SHUTDOWN_EVENT || /* none of these are called */
            dwCtrlType == CTRL_LOGOFF_EVENT )
  {
    /* http://support.microsoft.com/support/kb/articles/q130/7/17.asp */
    //no use calling RaiseExitRequestTrigger() from here. We have to terminate
    //here or win will throw up a message box and then call ExitProcess().
    //we have 5 seconds for CLOSE, and 20 secs for SHUTDOWN/LOGOFF
    RaiseExitRequestTrigger();
    while (constatics.client_run_startstop_level > 0)
      Sleep(500);
    //__w16ClientHardStop();
    ExitProcess(0);
    //return TRUE;
  }
  return FALSE; //DBG_CONTROL_C
}
LRESULT CALLBACK __w32ShimWatcherWProc(HWND hwnd, UINT message, WPARAM wParam,
                                     LPARAM lParam)
{
  static UINT dnetc_cmdmsg = 0;
  LRESULT lResult;
  if (dnetc_cmdmsg && message == dnetc_cmdmsg)
    message = WM_COMMAND;

  if (message == WM_CREATE)
    dnetc_cmdmsg = RegisterWindowMessage(W32CLI_CONSOLE_NAME);
  else if (message == WM_CLOSE)
  {
    RaiseExitRequestTrigger();
    return DNETC_WCMD_ACKMAGIC;
  }
  else if (__w16WindowHandle_DNETC_WCMD(hwnd,message,wParam,lParam,&lResult))
    return lResult;
  else if (message == WM_USER_W16CONS && wParam == W16CONS_CMD_INDIRDESTROY)
  {
    //SendMessage(hwnd, WM_DESTROY, 0, 0 );
    DestroyWindow(hwnd);
    PostQuitMessage(0);
  }
  return DefWindowProc(hwnd,message,wParam,lParam);
}
static void __win32ShimWatcher(void *) /* lives as long as the client */
{
  int normalstop = 0;
  HINSTANCE hInstance = winGetInstanceHandle(); /* w32pre.cpp */

  if (hInstance)
  {
    static int classisreg = 0;
    WNDCLASS wcl;
    /* define a window class */
    wcl.hInstance = hInstance;
    wcl.lpszClassName = "DCTICLISTUB";
    wcl.lpfnWndProc = (WNDPROC)__w32ShimWatcherWProc;
    wcl.style = 0; //CS_HREDRAW | CS_VREDRAW;
    wcl.hIcon = NULL;
    wcl.hCursor = NULL;
    wcl.lpszMenuName = NULL;
    wcl.cbClsExtra = 0;
    wcl.cbWndExtra = 0;
    wcl.hbrBackground = NULL;

    /* register the window class */
    if (RegisterClass(&wcl))
      classisreg = 1;
    if (classisreg)
    {
      HWND hwnd = CreateWindow( wcl.lpszClassName, W32CLI_CONSOLE_NAME,
                            0/*WS_POPUP|WS_CLIPSIBLINGS|WS_OVERLAPPED*/,
                            0, 0, 0, 0, NULL, NULL, wcl.hInstance, NULL );
      if (hwnd)
      {
        MSG msg;

        //ShowWindow(hwnd, SW_HIDE);

        if (constatics.devpipe)
          __pipe_advertise_hwnd(hwnd);

        constatics.shimwatcher = hwnd;
        while (GetMessage(&msg, hwnd, 0, 0))
        {
          TranslateMessage(&msg);
          DispatchMessage(&msg);
        }
        constatics.shimwatcher = NULL;

        if (constatics.devpipe)
          __pipe_advertise_hwnd(NULL);

        if (IsWindow(hwnd))
          DestroyWindow(hwnd);
      }
      if (UnregisterClass( wcl.lpszClassName, wcl.hInstance ))
        classisreg = 0;
    }
  }
  return;
}
char *my_getenvvar(const char *envvar,char *buffer,unsigned int buflen)
{                               
  char fn[MAX_PATH+1];
  int len = GetModuleFileName(NULL,fn,sizeof(fn)-1);
  char *q;

  if (len != 0)
  {
    while (len > 0 && fn[len-1]!='\\' && fn[len-1]!='/' && fn[len-1]!=':')
    {
      if (fn[len] == '.') 
        fn[len] = '\0';
      len--;
    }
    if (len > 0)
    {
      q = &fn[len];
      len = 0;
      while (*q) 
        fn[len++] = *q++;
      strcpy(&fn[len],envvar);
    }
  }
  if (len < 1)
  {
    strcat( strcpy( fn, "dnetc" ), envvar );
  }
  envvar = fn;
  /* there is something buggy in windows 95's GetEnvironmentVariable(), */
  /* so we use getenv() instead (which uses GetEnvironmentStrings()) */
  q = getenv(envvar); 
  if (q)
  {
    strncpy(buffer,q,buflen);
    buffer[buflen-1] = '\0';
  }
  else
  {
    DWORD len = GetEnvironmentVariable(envvar,buffer,buflen);
    if (len == 0 || len > buflen)
      return (char *)0;
  }
  return buffer;
}
BOOL my_SetStdHandle(DWORD nStdHandle, HANDLE hHandle)
{
  if (hHandle != INVALID_HANDLE_VALUE)
  {
    int tgt_fd = ((nStdHandle == STD_INPUT_HANDLE)?(0):
                 ((nStdHandle == STD_OUTPUT_HANDLE)?(1):(2)));
    SetStdHandle(nStdHandle, hHandle);
    if (((HANDLE)_get_osfhandle(tgt_fd)) != hHandle)
    {
      int mode = (nStdHandle == STD_INPUT_HANDLE)?(O_RDONLY):(O_WRONLY);
      int fd = _open_osfhandle((long)hHandle, mode|O_TEXT);
      if (fd != -1)
      {
        dup2(fd, tgt_fd );
        if (nStdHandle == STD_OUTPUT_HANDLE && 
          GetStdHandle(STD_ERROR_HANDLE)==INVALID_HANDLE_VALUE)
        {
          SetStdHandle(STD_ERROR_HANDLE,hHandle);
          dup2(fd, 2);
        }  
        else if (nStdHandle == STD_ERROR_HANDLE && 
          GetStdHandle(STD_OUTPUT_HANDLE)==INVALID_HANDLE_VALUE)
        {
          SetStdHandle(STD_OUTPUT_HANDLE,hHandle);
          dup2(fd, 1);
        }
      }
    }
    return TRUE;
  }
  return FALSE;
}
int __pipe_init_pair(HANDLE pstdin, HANDLE pstdout)
{
  int isanonpipe = 1; /* assume a bidirectional anon pipe */
  
  /* at this point pstdout should never be INVALID_HANDLE_VALUE */
  if (pstdout == INVALID_HANDLE_VALUE)
    return -1;
  /* pstdin will be INVALID_HANDLE_VALUE if pstdout is a named pipe */
  if (pstdin == INVALID_HANDLE_VALUE)
    isanonpipe = 0;

  if (constatics.hidden || constatics.nativecons)
  {
    DWORD numberOfBytesWritten;
    char *p = "\x03";
    __pipe_detach(); /* close the pipe == fork() :) */
    WriteFile(pstdout, (LPCVOID)p, 1, &numberOfBytesWritten, NULL);
    if (!isanonpipe)
      CloseHandle(pstdout);
  }
  else
  {
    __pipe_set_title(W32CLI_CONSOLE_NAME);
    constatics.devpipe = (void *)pstdout;
    if (isanonpipe)
      constatics.devpipein = (void *)pstdin;
    else
      pstdin = pstdout;
    //my_SetStdHandle(STD_INPUT_HANDLE, pstdin);
    my_SetStdHandle(STD_OUTPUT_HANDLE, pstdout);
    my_SetStdHandle(STD_ERROR_HANDLE, pstdout);
  }
  return 0;
}
#endif

/* ---------------------------------------------------- */

int w32InitializeConsole(int runhidden, int runmodes)
{
  const char *wintitle = W32CLI_CONSOLE_NAME;
  int isservicified = 0, retcode = 0;
  char *p; char scratch[256]; /* not smaller! */

  constatics.hidden = (!runmodes && runhidden);
  constatics.nativecons = 0;
  constatics.devpipe = 0;

  TRACE_INITEXIT((+1,"w32InitializeConsole(hidden=%d, runmodes=%d)\n",runhidden,runmodes));

  //quickly change to a normal cursor
  SetCursor(LoadCursor(NULL, MAKEINTRESOURCE(IDC_ARROW)));

  #if (CLIENT_OS == OS_WIN32)
  if (!constatics.hidden && retcode == 0)
  {
    if (win32CliServiceRunning())
    {
      isservicified = 1;
      constatics.hidden = 1;
      TRACE_INITEXIT((+0,"service active. forcing hidden"));
    }
  }
  #endif

  // ------------------------------------
  // single instance check
  // ------------------------------------

  if (!runmodes && retcode==0 && getenv("dnetc_multiok")==NULL)
  {
    retcode = w32PostRemoteWCMD( DNETC_WCMD_EXISTCHECK );
    if ((retcode & 0x04)!=0) /* svc flag found */
    {
      if (win32CliServiceRunning()) /* we ourselves are the service */
        retcode ^= 0x04;
    }
    TRACE_INITEXIT((+0,"other instance running?=0x%x\n", retcode ));
    if (retcode != 0)
      retcode = -1;
  }

  #if (CLIENT_OS == OS_WIN32)
  if (!runmodes && retcode == 0)
  {
    SECURITY_ATTRIBUTES sa;
    TRACE_INITEXIT((+0,"begin create mutex\n" ));
    memset(&sa,0,sizeof(sa));
    sa.nLength = sizeof(sa);
    constatics.hmutex = CreateMutex(&sa, FALSE, W32CLI_MUTEX_NAME);
    if (!constatics.hmutex)
      retcode = -1;
    TRACE_INITEXIT((+0,"end create mutex\n" ));
  }
  #endif

  // ----------------------------
  // console as CUI or pipe?
  // ----------------------------

  #if (CLIENT_OS == OS_WIN32)
  if (retcode == 0)
  {
    #if defined(USE_NATIVE_CONSOLEIO)
    constatics.nativecons = 1;
    #endif
    if (retcode == 0 && !constatics.nativecons)
    {
      STARTUPINFO si;
      GetStartupInfo( &si );

      if ((si.cbReserved2 != 0) &&  (si.lpReserved2 != NULL))
      {
        int num,numhandles; HANDLE *posfhnd; 
        char *posfile = (char *)(si.lpReserved2);
        numhandles = *((int *)posfile); posfile += sizeof(int);
        posfhnd = (HANDLE *)(posfile + numhandles);

        TRACE_INITEXIT((+0,"begin try console 1\n" ));

        if ((si.dwFlags & STARTF_USESTDHANDLES)==0)
          si.hStdInput = si.hStdOutput = si.hStdError = INVALID_HANDLE_VALUE;

        if (numhandles > 3) numhandles = 3; /* we only care about std* */
        for (num = 0; num < numhandles; num++)
        {
          #define m_FOPEN          0x01    /* file handle open */
          #define m_FPIPE          0x08    /* file handle refers to a pipe */
          #define m_FAPPEND        0x20    /* file handle opened O_APPEND */
          #define m_FDEV           0x40    /* file handle refers to device */
          #define m_FTEXT          0x80    /* file handle is in text mode */

          /* GetFileType() will block if its a pipe with input pending */
          if ((posfile[num] & m_FPIPE)==0)
          { 
            DWORD ftype = GetFileType( posfhnd[num] );
            if (ftype == FILE_TYPE_UNKNOWN)
              posfile[num] = 0;
            else if ((ftype & 0xff) == FILE_TYPE_PIPE)
              posfile[num] |= m_FPIPE;
          }
          if (posfhnd[num] == INVALID_HANDLE_VALUE ||
               (posfile[num] & m_FOPEN)==0 )
          {
            posfhnd[num] = INVALID_HANDLE_VALUE;
            posfile[num] = 0;
          }
          else if ( (posfile[num] & m_FPIPE)==0 )
          {
            if (num == 0 && si.hStdInput == INVALID_HANDLE_VALUE)
              si.hStdInput = posfhnd[num];
            else if (num == 1 && si.hStdOutput == INVALID_HANDLE_VALUE)
              si.hStdOutput = posfhnd[num];
            else if (num == 3 && si.hStdError == INVALID_HANDLE_VALUE)
              si.hStdError = posfhnd[num];
            si.dwFlags |= STARTF_USESTDHANDLES;
          }
          else if (num == 1 && (posfile[1] & m_FPIPE)!=0 && 
                               (posfile[0] & m_FPIPE)!=0 )
          {
            __pipe_init_pair(posfhnd[0], posfhnd[1]);
            break;
          }
        } 
        TRACE_INITEXIT((+0,"end try console 1\n" ));
      }
      if ((si.dwFlags & STARTF_USESTDHANDLES)!=0)
      {
        constatics.nativecons = 1;
        my_SetStdHandle(STD_OUTPUT_HANDLE,si.hStdOutput);
        my_SetStdHandle(STD_INPUT_HANDLE,si.hStdInput);
        my_SetStdHandle(STD_ERROR_HANDLE,si.hStdError);
        SetConsoleTitle(wintitle);
      }
    }
    if (retcode == 0 && !constatics.nativecons && !constatics.devpipe)
    {
      DWORD lpMode;
      if (GetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE),&lpMode) ||
          GetConsoleMode(GetStdHandle(STD_INPUT_HANDLE),&lpMode))
      {
        SetConsoleTitle(wintitle);
        constatics.nativecons = 1;
        //SetStdHandle(STD_OUTPUT_HANDLE, (HANDLE)_get_osfhandle(fileno(stdout)));
        //SetStdHandle(STD_INPUT_HANDLE, (HANDLE)_get_osfhandle(fileno(stdin)));
        //SetStdHandle(STD_ERROR_HANDLE, (HANDLE)_get_osfhandle(fileno(stdout)));
      }
    }

    if (retcode == 0 && !constatics.nativecons && !constatics.devpipe)
    {
      if (my_getenvvar(".ttyhandles",scratch,sizeof(scratch)))
      {
        unsigned int which;
        HANDLE inouterr[3];
        p = scratch;
        for (which = 0; which < 3; which++)
        {
          HANDLE h = (HANDLE)atol(p);
          if (*p == '-')
            p++;
          if (!h && *p != '0')
            break;
          inouterr[which]=h;
          if (which == 0 || which == 1)
          {
            while (isdigit(*p))
              p++;
            while (*p && !isdigit(*p))
              p++;
          }
          else
          {
            if (inouterr[0] != INVALID_HANDLE_VALUE)
              my_SetStdHandle(STD_INPUT_HANDLE, inouterr[0]);
            if (inouterr[1] != INVALID_HANDLE_VALUE)
              my_SetStdHandle(STD_OUTPUT_HANDLE, inouterr[1]);
            if (inouterr[2] != INVALID_HANDLE_VALUE)
              my_SetStdHandle(STD_ERROR_HANDLE, inouterr[2]);
            constatics.nativecons = 1;
            SetConsoleTitle(wintitle);
          }
        }
      }
    }

    if (retcode == 0 && !constatics.devpipe)
    {
      HANDLE pipe = INVALID_HANDLE_VALUE;
      HANDLE inpipe = INVALID_HANDLE_VALUE;
      int anonpipe = 0, gotpipeprompt = 0;

      if (winGetVersion() >= 2000 &&
        my_getenvvar(".namedpipe",scratch,sizeof(scratch)))
      {
        gotpipeprompt = 1;
        pipe = CreateFile(scratch, GENERIC_READ|GENERIC_WRITE, 0,
                                   NULL, OPEN_EXISTING, 0, 0 );
      }
      if (!gotpipeprompt &&  
         my_getenvvar(".apipe.out",scratch,sizeof(scratch)))
      {
        gotpipeprompt = anonpipe = 1;
        if (isdigit(scratch[0]) || (scratch[0] == '-' && isdigit(scratch[1])))
        {
          pipe = (HANDLE)atol(scratch);
          if (pipe != INVALID_HANDLE_VALUE && 
            my_getenvvar(".apipe.in",scratch,sizeof(scratch)))
          {
            if (isdigit(scratch[0]) || (scratch[0]=='-' && isdigit(scratch[1])))
              inpipe = (HANDLE)atol(scratch);
          }
        }
        if (inpipe == INVALID_HANDLE_VALUE)
          pipe = INVALID_HANDLE_VALUE;
      }
      if (gotpipeprompt)
      {
        if (pipe != INVALID_HANDLE_VALUE)
        {
          retcode = __pipe_init_pair(inpipe,pipe);
        }
        else if (!constatics.nativecons) /* no error if we already have a con */
        {
          if (!constatics.hidden)
            w32ConOutErr( "Unable to open client end of console pipe" );
          retcode = -1;
        }
      }
    }

    if (retcode == 0 && !constatics.nativecons &&
        !constatics.hidden && !constatics.devpipe )
    {                     //use guicode (don't allocconsole) if we're hidden
      if (winGetMyModuleFilename(scratch,sizeof(scratch)) > 4) /* win32pre.cpp */
      {
        if (strcmpi( &scratch[strlen(scratch)-4], ".com" ) == 0)
        {
          p = "Unable to create console window.";
          if (!AllocConsole())
            retcode = -1;
          else
          {
            FILE *hfin, *hfout;
            HWND hwnd;
            p = "Unable to open console for write.";
            retcode = 0;
            sprintf(scratch,"%s - %08x%08x",
                  wintitle, GetTickCount(), GetModuleHandle(NULL));
            SetConsoleTitle(scratch);
            while ((hwnd = FindWindow( NULL, scratch )) == NULL)
            {
              if ((++retcode) > 25)
                break;
              Sleep(40); //Delay needed for title to update
            }
            SetConsoleTitle(wintitle);
            if ( hwnd == NULL )
              retcode = -1;
            else if ((hfout = fopen("CONOUT$", "w+t")) == NULL)
              retcode = -1;
            else if ((hfin = fopen("CONIN$", "rt")) == NULL)
            {
              p = "Unable to open console for read.";
              fclose( hfout );
              retcode = -1;
            }
            else
            {
              retcode = 0;
              constatics.fstdout = hfout;
              constatics.fstdin = hfin;
              *stdout = *hfout;
              setvbuf(stdout, NULL, _IONBF, 0);
              *stderr = *hfout;
              setvbuf(stderr, NULL, _IONBF, 0);
              *stdin = *hfin;
              setvbuf(stdin, NULL, _IONBF, 0);
              constatics.nativecons = 1;
            }
            if (retcode != 0)
              FreeConsole();
          }
          if (p && retcode != 0 && !constatics.hidden)
            w32ConOutErr( p );
        }
      }
    }

    if (retcode == 0 && constatics.nativecons)
    {
      DWORD dwConMode;
      #if defined(__WATCOMC__)
      int sig;
      for (sig = 0; sig < 255; sig++)
        signal(sig, __w32SigTriggerControl );
      #endif
      dwConMode = ~ENABLE_WINDOW_INPUT;
      //if (GetConsoleMode( GetStdHandle(STD_INPUT_HANDLE),&dwConMode))
      {
        //if ((dwConMode & ENABLE_PROCESSED_INPUT)==0)
        {
          dwConMode |= ENABLE_PROCESSED_INPUT;
          SetConsoleMode( GetStdHandle(STD_INPUT_HANDLE), dwConMode );
        }
      }
      SetProcessShutdownParameters(0x100,SHUTDOWN_NORETRY);
      SetConsoleCtrlHandler((PHANDLER_ROUTINE)__w32NativeTriggerControl,TRUE);

      if (constatics.hidden)
        FreeConsole(); //fork() :)
    }
  }
  #endif /* (CLIENT_OS == OS_WIN32) */

  // ---------------------------
  // console as GUI?
  // ---------------------------

  if (retcode == 0 && !constatics.devpipe && !constatics.nativecons)
  {
    int nCmdShow = winGetInstanceShowCmd();
    if (constatics.hidden)
      nCmdShow = SW_HIDE;

    if (w16ConsoleCreate(nCmdShow))
      w16SetConsoleTitle(wintitle);
    else
    {
      if (!constatics.hidden)
      {
        p = NULL;
        if (constatics.errorcode == W16CONS_ERR_CREATETHREAD)
          p = "create console thread";
        else if (constatics.errorcode == W16CONS_ERR_NOFONT)
         p = "assign font";
        else if (constatics.errorcode == W16CONS_ERR_NOMEM)
          p = "create window data area";
        else if (constatics.errorcode == W16CONS_ERR_CREATEWIN)
          p = "create window client area";
        else if (constatics.errorcode == W16CONS_ERR_NCCREATE)
          p = "create window non-client area";
        else if (constatics.errorcode == W16CONS_ERR_REGCLASS)
          p = "register class";
        else if (constatics.errorcode == W16CONS_ERR_GETINST)
          p = "get instance handle";
        else if (constatics.errorcode == W16CONS_ERR_NOSLOT)
          p = "find a window slot";
        sprintf(scratch,"Unable to create console window.%s%s)",
              ((!p)?(" (Unknown error"):("\n(Failed to ")),
              ((!p)?(""):(p)) );
        w32ConOutErr(scratch);
      }
      retcode = -1;
    }
  }

  if (retcode == 0)
  {
    //quickly change to a normal cursor
    SetCursor(LoadCursor(NULL, MAKEINTRESOURCE(IDC_ARROW)));

    //watch for ClientRun() start/stop/sleep(1) events
    ClientEventAddListener(-1,__ClientEventCallback);

    #if (CLIENT_OS == OS_WIN32)
    if (constatics.devpipe || constatics.nativecons)
    {
      int havethread = 0;
      if (constatics.shimwatcher)
        havethread = 1;
      else if (_beginthread( __win32ShimWatcher, 512, NULL ))
      {
        while ((++havethread) < 20 && !constatics.shimwatcher)
          Sleep(100);
        if (!constatics.shimwatcher)
          havethread = 0;
      }
      if (havethread) //we don't want the window to be findable
        wintitle = W32CLI_CONSOLE_NAME" ";
      if (constatics.devpipe)
        __pipe_set_title(wintitle);
      else if (constatics.nativecons)
        SetConsoleTitle(wintitle);
    }
    #endif
  }
  #if (CLIENT_OS == OS_WIN32)
  else if (constatics.hmutex)
  {
    ReleaseMutex( constatics.hmutex );
    CloseHandle( constatics.hmutex );
    constatics.hmutex = NULL;
  }
  #endif

  TRACE_INITEXIT((-1,"w32InitializeConsole() => retcode=%d\n",retcode));

  return (retcode);
}

/* ================================================================== */
/* ================================================================== */

int w32ConKbhit(void)
{
  #if (CLIENT_OS == OS_WIN32)
  if (constatics.devpipe)
  {
    if (__pipe_kbhit() <= 0)
      return 0;
    return 1;
  }
  else if (constatics.nativecons)
    return (kbhit());
  #endif
  return w16ConsoleKbhit();
}

/* ---------------------------------------------------- */

int w32ConGetch(void)
{
  #if (CLIENT_OS == OS_WIN32)
  if (constatics.devpipe)
  {
    static int hibyte = 0;
    int ch;
    if (hibyte != 0)
    {
      ch = hibyte;
      hibyte = 0;
      return ch;
    }
    while (!CheckExitRequestTriggerNoIO())
    {
      int kbstate = __pipe_getchar(&ch);
      if (kbstate < 0) /* broken pipe */
        break;
      if (kbstate != 0)
      {
        if (ch == 0)
        {
          kbstate = __pipe_getchar(&ch);
          if (kbstate < 0) /* broken pipe */
            break;
          hibyte = ch;
        }
        return ch;
      }
      Sleep(100);
    }
    return 0;
  }
  else if (constatics.nativecons)
  {
    return getch();
  }
  #endif
  return w16ConsoleGetch();
}

/* ---------------------------------------------------- */

static int __w32ConOutX(const char *text, int iserr)
{
  int handled = 0;
  if (win32CliServiceRunning())
  {
    handled = 1;
    if (iserr) /* we don't print anything if !err */
    {
      /* do log stuff for NT here */
      return 0;
    }
    return -1;
  }
  #if (CLIENT_OS == OS_WIN32)
  if (!handled)
  {
    if (constatics.nativecons)
      handled = 1;
    else
    {
      char filename[MAX_PATH+1];
      if (winGetMyModuleFilename(filename, sizeof(filename)) != 0) //w32pre.cpp
      {
        if (winIsGUIExecutable( filename )==0) /*w32util <0=err,0=cui,>0=gui*/
          handled = 1;
      }
    }
    if (handled)
    {
      FILE *file = stdout;
      if (!iserr)
        fprintf( file, "%s\n", text );
      else
      {
        file = stderr;
        fprintf(file,"%s: %s\n", utilGetAppName(), text);
      }
      fflush(file);
    }
  }
  if (!handled)
  {
    int needclose = 0;
    if (!constatics.devpipe)
    {
      STARTUPINFO si;
      GetStartupInfo( &si );

      if ((si.cbReserved2 != 0) &&  (si.lpReserved2 != NULL))
      {
        HANDLE *handles;
        char *p = (char *)si.lpReserved2; 
        int numhandles = *((int *)p);
        char *modes = (p += sizeof(int));
        p += numhandles;
        handles = (HANDLE *)(p);
        if (iserr && numhandles >= 3 &&
           (modes[2] & 0x09)==0x09 && /* 0x09 is m_FOPEN+m_FPIPE */
            handles[2] != INVALID_HANDLE_VALUE)
          constatics.devpipe = handles[2];
        else if (numhandles >= 2 && 
           (modes[1] & 0x09)==0x09 && /* 0x09 is m_FOPEN+m_FPIPE */
            handles[1] != INVALID_HANDLE_VALUE)
          constatics.devpipe = handles[1];
      }
    }
    if (!constatics.devpipe)
    {
      char buf[64];
      char *p = my_getenvvar(".apipe.out",buf,sizeof(buf));
      if (p)
      {
        HANDLE h = (HANDLE)atol(p);
        if (h && h!=INVALID_HANDLE_VALUE)
        {
          constatics.devpipe = (void *)h;
          needclose = 1;
        }
      }
    }
    if (constatics.devpipe)
    {
      if (iserr)
      {
        __pipe_puts(utilGetAppName(),strlen(utilGetAppName()));
        __pipe_puts(": ",2);
      }
      __pipe_puts(text,strlen(text));
      __pipe_puts("\n",1);
      if (needclose)
        constatics.devpipe = NULL;
      handled = 1;
    }
  }
  #endif
  if (!handled)
  {
    /* note the spaces around the caption! Don't let this window be "findable" */
    MessageBox(NULL,text, " "W32CLI_CONSOLE_NAME" ",MB_OK|MB_TASKMODAL
                            |(iserr?MB_ICONHAND:MB_ICONINFORMATION));
  }
  return 0;
}

int w32ConOutErr(const char *text)
{ return __w32ConOutX(text, 1); }
int w32ConOutModal(const char *text)
{ return __w32ConOutX(text, 0); }

/* ---------------------------------------------------- */

/* triggers.cpp calls w32ConOut("") to keep ^C checking alive */
int w32ConOut(const char *text)
{
  int len = (int)strlen(text);
  #if (CLIENT_OS == OS_WIN32)
  if (constatics.devpipe)
  {
    //char buf[16]; sprintf(buf,"%08x:",GetTickCount());
    //__pipe_puts(buf,strlen(buf));
    if (len)
      len = __pipe_puts(text,len);
    /* we don't need to flush since pipes are unbuffered */
    return len;     
  }
  else if (constatics.nativecons)
  {
    if (len)
      len = fwrite( text, sizeof(char), len, stdout);
    fflush(stdout);
    return len;
  }
  #endif
  if (len)
    len = w16ConsolePrint(text);
  return len;
}

/* ---------------------------------------------------- */

int w32ConIsScreen(void)
{
  if (!constatics.hidden)
  {
    #if (CLIENT_OS == OS_WIN32)
    if (constatics.devpipe)
    {
TRACE_PIPE((+1,"w32ConIsScreen()\n"));
      int istty = 1;
      if (__pipe_isatty() <= 0) /* <0=err, 0=no, >0=yes */
        istty = 0;
TRACE_PIPE((-1,"w32ConIsScreen() =>%d\n",istty));
      return istty;
    }
    if (constatics.nativecons)
      return isatty(fileno(stdout));
    #endif
    return w16HaveWindow();
  }
  return 0;
}

/* ---------------------------------------------------- */

void w32Sleep(unsigned int millisecs)
{
  #if (CLIENT_OS == OS_WIN32)
  if (constatics.devpipe)
  {
    __pipe_sleep(millisecs);
    return;
  }
  if (constatics.nativecons)
  {
    Sleep(millisecs);
    return;
  }
  #endif
  w16Sleep(millisecs);
  return;
}

/* ---------------------------------------------------- */

void w32Yield(void)
{
  #if (CLIENT_OS == OS_WIN32)
  if (constatics.nativecons || constatics.devpipe)
  {
    w32Sleep(1); /* millisecs */
    return;
  }
  #endif
  w16Yield();
  return;
}

/* ---------------------------------------------------- */

int w32ConGetSize( int *width, int *height) /* one based */
{
  if (!constatics.hidden)
  {
    #if (CLIENT_OS == OS_WIN32)
    if (constatics.devpipe)
    {
      static int _cache_height = -1, _cache_width = -1;
      if (_cache_height == -1)
      {
        if (__pipe_getwinsz(&_cache_width, &_cache_height)<0) /* not a tty */
          _cache_height = _cache_width = 0;
      }
      if (_cache_height == 0 && _cache_width == 0)
        return -1;
      if (width)  *width  = _cache_width;
      if (height) *height = _cache_height;
      return 0;
    }
    else if (constatics.nativecons)
    {
      HANDLE hStdout;
      CONSOLE_SCREEN_BUFFER_INFO csbiInfo;

      hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
      if (hStdout == INVALID_HANDLE_VALUE)
        return -1;
      if (! GetConsoleScreenBufferInfo(hStdout, &csbiInfo))
        return -1;
      if (height) *height=csbiInfo.srWindow.Bottom - csbiInfo.srWindow.Top + 1;
      if (width) *width = csbiInfo.srWindow.Right - csbiInfo.srWindow.Left + 1;
      return 0;
    }
    #endif
    return w16ConGetSize( width, height );
  }
  return -1;
}

/* ---------------------------------------------------- */

int w32ConClear(void)
{
  if (!constatics.hidden)
  {
    #if (CLIENT_OS == OS_WIN32)
    if (constatics.devpipe)
    {
      if (__pipe_puts(("\x1B""[2J""\x1B""[1;1H"), 10 ) < 0)
        return -1;
      return 0;
    }
    else if (constatics.nativecons)
    {
      CONSOLE_SCREEN_BUFFER_INFO csbi;
      DWORD nLength, nWritten;
      COORD topleft = {0,0};
      HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
      if (hStdout == INVALID_HANDLE_VALUE)
        return -1;
      if (! GetConsoleScreenBufferInfo(hStdout, &csbi))
        return -1;
      nLength = csbi.dwSize.X * csbi.dwSize.Y;
      FillConsoleOutputCharacter(hStdout, (TCHAR) ' ', nLength, topleft, &nWritten);
      FillConsoleOutputAttribute(hStdout, csbi.wAttributes, nLength, topleft, &nWritten);
      SetConsoleCursorPosition(hStdout, topleft);
      return 0;
    }
    #endif
    return w16ConsoleClear();
  }
  return -1;
}

/* ---------------------------------------------------- */

int w32ConSetPos(int col, int row) /* zero based */
{
  if (!constatics.hidden)
  {
    #if (CLIENT_OS == OS_WIN32)
    if (constatics.devpipe)
    {
      char buffer[64];
      if (__pipe_puts(buffer,
          sprintf(buffer,"\x1B""[%d;%dH", row+1, col+1 )) < 0)
        return -1;
      return 0;
    }
    else if (constatics.nativecons)
    {
      HANDLE hStdout;
      COORD pos = {(SHORT)col,(SHORT)row};
      hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
      if (hStdout == INVALID_HANDLE_VALUE)
        return -1;
      SetConsoleCursorPosition(hStdout, pos);
      return 0;
    }
    #endif
    return w16ConSetPos(col, row);
  }
  return -1;
}

/* ------------------------------------------------ */

int w32ConGetPos(int *col, int *row) /* zero based */
{
  if (!constatics.hidden)
  {
    #if (CLIENT_OS == OS_WIN32)
    if (constatics.devpipe)
    {
      return __pipe_getxy(col, row);
    }
    else if (constatics.nativecons)
    {
      HANDLE hStdout;
      CONSOLE_SCREEN_BUFFER_INFO csbiInfo;

      hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
      if (hStdout == INVALID_HANDLE_VALUE)
        return -1;
      if (! GetConsoleScreenBufferInfo(hStdout, &csbiInfo))
        return -1;
      if (col) *col=(int)csbiInfo.dwCursorPosition.X;
      if (row) *row=(int)csbiInfo.dwCursorPosition.Y;
      return 0;
    }
    #endif
    return w16ConGetPos(col,row);
  }
  return -1;
}

/* ------------------------------------------------ */

int w32ConGetType(void)
{
  if (constatics.hidden)
    return 0;
  #if (CLIENT_OS == OS_WIN32)
  if (constatics.nativecons)
    return 'C';
  if (constatics.devpipe)
    return 'c';
  #endif
  /* gui must be tested _after_ CUI */
  if (constatics.hwndList[0])
  {
    if (IsWindow(constatics.hwndList[0]))
    {
      int isvis = IsWindowVisible(constatics.hwndList[0]);
      int isico = IsIconic(constatics.hwndList[0]);
      int rc = MAKEWORD('g',0); /* assume normal */  
      if (!isvis && isico) /* in tray */
        rc = MAKEWORD('g','t');
      else if (!isvis && !isico) /* hidden */
        rc = MAKEWORD('g','h');
      else if (isvis && isico) /* minimized */
        rc = MAKEWORD('g','m');
      return rc;
    }
  }      
  //full GUI should return 'G';
  return 0;
}  

/* ------------------------------------------------ */

int printf(const char *format,...)
{
  va_list arglist;
  int retlen; char *buf;
  va_start (arglist, format);
  buf = (char *)malloc(8192);
  retlen = -1;
  if (buf)
  {
    retlen = vsprintf(buf, format, arglist );
    if (retlen > 0)
    {
      if (constatics.hwndList[0] || constatics.devpipe || constatics.nativecons)
        w32ConOut(buf);
      else
        w32ConOutModal(buf);
    }
    free((void *)buf);
  }
  return retlen;
}

/* ------------------------------------------------ */

struct __cbsendcmd
{
  UINT msg;
  WPARAM wParam;
  LPARAM lParam;
  int usepost;
  int foundcount;
  int ackcount;
  UINT dnetc_cmdmsg;
};

static BOOL CALLBACK __SendCmdEnumFunc(HWND hwnd,LPARAM lParam)
{
  char wintitle[128];
  if (GetWindowText( hwnd, wintitle, sizeof(wintitle) ))
  {
    int isours = 0, knowsdnet_cmdmsg = 0;
    if ( strcmp( wintitle, W32CLI_CONSOLE_NAME ) == 0 )
      isours = knowsdnet_cmdmsg = 1;
    else if ( strcmp( wintitle, W32CLI_OLD_CONSOLE_NAME ) == 0 )
      isours = 1;
    if (isours)
    {
      int iscui = 0;
      long ver = winGetVersion();
      struct __cbsendcmd *cbsc = (struct __cbsendcmd *)lParam;
      cbsc->foundcount++;

      if (ver >= 400 && GetClassName(hwnd,wintitle,sizeof(wintitle)))
        iscui = (!strcmp(wintitle,(ver>=2000)?("ConsoleWindowClass"):("tty")));

      if (!iscui)
      {
        int ishandled = 0;
        if (cbsc->msg == WM_COMMAND)
        {
          if (knowsdnet_cmdmsg && cbsc->dnetc_cmdmsg)
          {
            if (DNETC_WCMD_ACKMAGIC ==
              SendMessage(hwnd,cbsc->dnetc_cmdmsg,cbsc->wParam,cbsc->lParam))
            {
              cbsc->ackcount++;
              ishandled = 1;
            }
          }
          else if (cbsc->wParam == DNETC_WCMD_SHUTDOWN)
          {
            cbsc->usepost = 0;
            if (DNETC_WCMD_ACKMAGIC == SendMessage( hwnd, WM_CLOSE, 0, 0 ))
            {
              cbsc->ackcount++;
              ishandled = 1;
            }
          }
        }
        if (!ishandled)
        {
          if (cbsc->usepost)
            PostMessage( hwnd, cbsc->msg, cbsc->wParam, cbsc->lParam );
          else if (DNETC_WCMD_ACKMAGIC == SendMessage( hwnd, cbsc->msg,
                                    cbsc->wParam, cbsc->lParam ))
            cbsc->ackcount++;
        }
      }
      #if (CLIENT_OS == OS_WIN32)
      else if (cbsc->msg == WM_COMMAND &&
                (cbsc->wParam == DNETC_WCMD_RESTART ||
                cbsc->wParam == DNETC_WCMD_SHUTDOWN))
      {
        DWORD pid;
        if (GetWindowThreadProcessId(hwnd,&pid) != 0)
        {
          DWORD event = CTRL_BREAK_EVENT; /* restart */
          if (cbsc->msg == WM_CLOSE)
            event = CTRL_C_EVENT;
          if (GenerateConsoleCtrlEvent(event,pid))
            cbsc->ackcount++; /* assume it so */
        }
      }
      #endif
    }
  }
  return TRUE;
}

static int __findOtherClient(void) /* 0=none,0x1=bywindow,2=bymux,4=ntsvc*/
{
  int rc = 0;

  if (FindWindow( NULL, W32CLI_CONSOLE_NAME ))
    rc |= 0x01;
  else if (FindWindow( NULL, W32CLI_OLD_CONSOLE_NAME ))
    rc |= 0x01;

  #if (CLIENT_OS == OS_WIN32)
  {
    HANDLE hmutex;
    SECURITY_ATTRIBUTES sa;
    memset(&sa,0,sizeof(sa));
    sa.nLength = sizeof(sa);
    SetLastError(0);
    hmutex = CreateMutex(&sa, FALSE, W32CLI_MUTEX_NAME);
    if (hmutex)
    {
      if (GetLastError())  /* ie, it exists */
        rc |= 0x2;
      ReleaseMutex( hmutex );
      CloseHandle( hmutex );
    }
    /* this next part is a workaround: for some reason the mutex
    check above doesn't work to detect NT service when wanting
    to -shutdown etc, but does work for single-instance protection
    check when just starting normally.
    Detection is backwards compatible for NT service, but not for
    for w9x. However, win9x service will have been found by window
    name, (and the problem is NT only anyway) so its not an issue.
    */
    if (rc == 0)
    {
      extern int win32CliDetectRunningService(void);
      if (win32CliDetectRunningService() > 0) /* <0=err, 0=no, >0=yes */
        rc |= 0x04;
    }
  }
  #endif
  return rc;
}


int w32PostRemoteWCMD( int cmd ) /* returns <0 if not found, or */
{                                /* >0 = found+msgfailed, 0=found+msgok */
  int rc = -1;
  int foundflags = __findOtherClient();

  if (cmd == DNETC_WCMD_EXISTCHECK)
    return foundflags;

  if (foundflags) /* client is running */
  {
    rc = +1; /* assume msgfailed */

    #if (CLIENT_OS == OS_WIN32)
    /* take away focus from all windows - this is particularly critical
       for win9x console sessions since they hog cputime when in foreground */
    //SetForegroundWindow(GetDesktopWindow());
    //SetActiveWindow(GetDesktopWindow());
    #endif

    #if (CLIENT_OS == OS_WIN32)
    if (winGetVersion()>=2000)  /* NT Only */
    {
      int svccmd = -1;
      if (cmd == DNETC_WCMD_SHUTDOWN)
        svccmd = SERVICE_CONTROL_STOP;
      else if (cmd == DNETC_WCMD_PAUSE)
        svccmd = SERVICE_CONTROL_PAUSE;
      else if (cmd == DNETC_WCMD_UNPAUSE)
        svccmd = SERVICE_CONTROL_CONTINUE;
      else if (cmd == DNETC_WCMD_RESTART)
        svccmd = CLIENT_SERVICE_CONTROL_RESTART; //128 //service control #
      if (svccmd != -1)
      {
        /* <0=err, 0=none running, >0=msg sent */
        if (win32CliSendServiceControlMessage(svccmd) > 0) /* msg sent */
          rc = 0; /* message went */
      }
    }
    #endif

    if ((foundflags & 0x01) != 0) /* do by window */
    {
      struct __cbsendcmd cbsc;
      cbsc.msg = WM_COMMAND;
      cbsc.wParam = (WPARAM)cmd;
      cbsc.lParam = 0;
      cbsc.usepost = 1;
      cbsc.foundcount = 0;
      cbsc.ackcount = 0;
      cbsc.dnetc_cmdmsg = RegisterWindowMessage(W32CLI_CONSOLE_NAME);

      cbsc.foundcount = 0;
      if ( EnumWindows( (WNDENUMPROC)__SendCmdEnumFunc, (LPARAM)&cbsc ) )
      {
        if ( cmd != DNETC_WCMD_SHUTDOWN || cbsc.foundcount == 0 ||
               cbsc.foundcount == cbsc.ackcount )
          rc = 0; /* assume success for all but DNETC_WCMD_SHUTDOWN */
        else
        {
          DWORD elapsedticks = 0, lasttick = 0;
          while (rc && elapsedticks < 5000)
          {
            DWORD nowticks = GetTickCount();
            if (nowticks == 0)
              nowticks++;
            if (lasttick == 0)
              ;
            else if (nowticks >= lasttick)
              elapsedticks += (nowticks - lasttick);
            else
              elapsedticks += (nowticks + (1+(0xfffffffful - lasttick)));
            lasttick = nowticks;

            if (__findOtherClient() == 0)
              rc = 0; /* success! */
            else
            {
              #if (CLIENT_OS == OS_WIN32)
              Sleep(500);
              #else
              Yield();
              #endif
            }
          }
        }
      }
    }
  }
  return rc;
}

