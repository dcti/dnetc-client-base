/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Win32/Win16 Screen Saver stuff.
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/

const char *w32ss_cpp(void) {
return "@(#)$Id: w32ss.cpp,v 1.1.2.5 2001/05/14 15:24:38 cyp Exp $"; }

#include "cputypes.h"
#define INCLUDE_COMMDLG_H
#define INCLUDE_SHELLAPI_H
#define INCLUDE_VER_H
#include <windows.h>
#include <commdlg.h>
#include "w32util.h" /* int (PASCAL *__SSMAIN)(HINSTANCE,HINSTANCE,LPSTR,int);*/
#include "w32ini.h"  /* [Get|Write]DCTIProfile[String|Int]() */
#include "w32cons.h" /* W32CLI_CONSOLE_NAME, W32CLI_MUTEX_NAME */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#if !defined(MAX_PATH)           //win16
#define MAX_PATH 256
#endif
#ifndef GWL_HINSTANCE            //win16
#define GWL_HINSTANCE (-6)
#endif

#ifndef SM_CMONITORS
#define SM_XVIRTUALSCREEN       76
#define SM_YVIRTUALSCREEN       77
#define SM_CXVIRTUALSCREEN      78
#define SM_CYVIRTUALSCREEN      79
#define SM_CMONITORS            80
#define SM_SAMEDISPLAYFORMAT    81
#endif

/* ---------------------------------------------------- */
    
static const char *szSSIniSect = "ScreenSaver";
static const char *szAppName = "distributed.net client launcher";
static char szAlternateAppName[64] = {0};

#define SSWriteProfileString(e,v) WriteDCTIProfileString(szSSIniSect,e,v)
#define SSGetProfileString(e,d,b,l) GetDCTIProfileString(szSSIniSect,e,d,b,l)
#define SSWriteProfileInt(e,v)    WriteDCTIProfileInt(szSSIniSect,e,v)
#define SSGetProfileInt(e,d)        GetDCTIProfileInt(szSSIniSect,e,d)

/* ---------------------------------------------------- */

static int SSIsTransparencyAvailable(void)
{
  static int iswin95 = -1;
  if (iswin95 == -1)
    iswin95 = (winGetVersion()>=400 && winGetVersion()<2000)?(+1):(0);
  return iswin95;
}  

/* ---------------------------------------------------- */

static int SSChangePassword(HINSTANCE hInst, HWND hwnd)
{
  hInst = hInst;
  #if (CLIENT_OS != OS_WIN32)
  hwnd = hwnd;
  #else
  HINSTANCE hmpr = LoadLibrary("MPR.DLL");
  if (hmpr != NULL) 
  {
    typedef VOID (WINAPI *PWDCHANGEPASSWORD)
          (LPCSTR lpcRegkeyname,HWND hwnd,UINT uiReserved1,UINT uiReserved2);
    PWDCHANGEPASSWORD PwdChangePassword=
      (PWDCHANGEPASSWORD)GetProcAddress(hmpr, "PwdChangePasswordA" );
    if (PwdChangePassword != NULL)
      PwdChangePassword( "SCRSAVE", hwnd, 0,0 );
    FreeLibrary(hmpr);
  }
  #endif
  return 0;
}                                

/* ---------------------------------------------------- */

static LRESULT CALLBACK SaverWindowProc(HWND hwnd,UINT msg,WPARAM wParam,LPARAM lParam)
{ 
  static struct 
  {
    HWND hParentWnd;
    POINT initCursorPos;
    DWORD initTime;
    int isDialogActive;
    int reallyClose;
    int sstype; /* 0 == blank, -1 == none */
    UINT top, left, height, width;
    int mmthreshX, mmthreshY;
  } ss = {0,{0,0},0,0,0,0,0,0,0,0};
  
  switch (msg)
  { 
    case WM_CREATE:
    { 
      RECT rc;
      GetCursorPos(&(ss.initCursorPos)); 
      ss.initTime = GetTickCount();
      GetWindowRect(hwnd, &rc);
      ss.top = rc.top;
      ss.left = rc.left;
      ss.height = (UINT)(rc.bottom - rc.top);
      ss.width = (UINT)(rc.right - rc.left);
      ss.hParentWnd = GetParent(hwnd);
      if (ss.hParentWnd && GetDesktopWindow() == ss.hParentWnd)
        ss.hParentWnd = NULL;
      ss.mmthreshX = ss.mmthreshY = 4;
      #if defined(SM_CXDRAG) && defined(SM_CYDRAG)
      ss.mmthreshX = GetSystemMetrics(SM_CXDRAG); /* generally 2x2 */
      ss.mmthreshY = GetSystemMetrics(SM_CYDRAG);
      #endif
      if (!SSIsTransparencyAvailable())
        ss.sstype = 0;
      else if ((ss.sstype = SSGetProfileInt( "type", 0 )) != -1)
        ss.sstype = 0;
      else if (!ss.hParentWnd && (ss.width > (UINT)GetSystemMetrics(SM_CXSCREEN) ||
                                 ss.height > (UINT)GetSystemMetrics(SM_CYSCREEN)) )
      { /* compensate for multi-monitor */
        GetWindowRect(GetDesktopWindow(), &rc);
        ss.top = rc.top;
        ss.left = rc.left;
        ss.height = (UINT)(rc.bottom - rc.top);
        ss.width = (UINT)(rc.right - rc.left);
        MoveWindow( hwnd, ss.left,ss.top, ss.width, ss.height, 0 );
      }
      break;
    } 
    case WM_ERASEBKGND:
    {
      if (ss.sstype == -1)
        return 1;
      break;
    }
    case WM_PAINT:
    {
      if (ss.sstype == 0)
      {
        PAINTSTRUCT ps;
        BeginPaint(hwnd, &ps);
        EndPaint(hwnd,&ps);
        return 0;
      }
      if (ss.sstype == -1) /* transparent */
      {
        PAINTSTRUCT ps;
        HDC hDC = BeginPaint(hwnd, &ps);
        if (hDC)
        {
          HWND hDesktop = GetDesktopWindow();
          HDC hDesktopDC = GetDC(hDesktop);
          SetBkMode(hDC, TRANSPARENT);
          if (hDesktopDC)
          {
            if (ss.hParentWnd) /* preview mode + transparent */
            {
              UINT oldbltmode = STRETCH_ORSCANS;
              #if defined(STRETCH_HALFTONE)
              oldbltmode = STRETCH_HALFTONE;
              #endif
              oldbltmode = SetStretchBltMode( hDC, oldbltmode );
              #if defined(STRETCH_HALFTONE)
              SetBrushOrgEx(hDC, 0, 0, NULL);
              #endif
              StretchBlt(hDC, 0, 0, ss.width, ss.height, 
                         hDesktopDC, 0, 0, 
                         GetSystemMetrics(SM_CXSCREEN),
                         GetSystemMetrics(SM_CYSCREEN),
                         SRCCOPY);
              SetStretchBltMode( hDC, oldbltmode );
            }
            else /* fullscreen mode + transparent */
            {
              BitBlt(hDC, 0, 0, ss.width, ss.height, 
                hDesktopDC, 0, 0, SRCCOPY);
            }
            ReleaseDC( hDesktop, hDesktopDC );
          }
        }
        EndPaint(hwnd,&ps);
      }
      break;
    }
    #if 0 /* can't use this otherwise broadcast messages will destroy us */
    case WM_ACTIVATE: 
    case WM_ACTIVATEAPP: 
    case WM_NCACTIVATE:
    { 
      if (LOWORD(wParam)==WA_INACTIVE && !ss.hParentWnd && !ss.isDialogActive)
      {
        ss.reallyClose = 1;
        PostMessage(hwnd,WM_CLOSE,0,0);
      }
      break;
    }
    #endif
    case WM_SETCURSOR:
    { 
      if (ss.hParentWnd==NULL && !ss.isDialogActive && ss.sstype != -1)
        SetCursor(NULL);
      else
        SetCursor(LoadCursor(NULL,IDC_ARROW));
      break;
    }
    case WM_LBUTTONDOWN: 
    case WM_MBUTTONDOWN: 
    case WM_RBUTTONDOWN: 
    case WM_KEYDOWN:
    { 
      if (ss.hParentWnd==NULL && !ss.isDialogActive)
      {
        ss.reallyClose = 1;
        PostMessage(hwnd,WM_CLOSE,0,0);
      }  
      break;
    }
    case WM_MOUSEMOVE:
    { 
      if (ss.hParentWnd==NULL && !ss.isDialogActive)
      {
        POINT pt; int dx, dy;
        GetCursorPos(&pt); 
        dx= pt.x - ss.initCursorPos.x; if (dx<0) dx=-dx; 
        dy= pt.y - ss.initCursorPos.y; if (dy<0) dy=-dy;
        if (dy > ss.mmthreshY || dx > ss.mmthreshX)
        {
          ss.reallyClose = 1;
          PostMessage(hwnd,WM_CLOSE,0,0);
        }  
        else /* adjust for a mouse on an uneven surface */
        {
          ss.initCursorPos.x = pt.x;
          ss.initCursorPos.y = pt.y;
        }
      }
      break;
    }
    case WM_SYSCOMMAND:
    { 
      if (ss.hParentWnd==NULL && (wParam==SC_SCREENSAVE || wParam==SC_CLOSE))
        return 0;
      break;
    }
    case (WM_CLOSE):
    { 
      if (ss.hParentWnd==NULL) 
      {
        if (ss.reallyClose && !ss.isDialogActive)
        {
          BOOL canClose = TRUE;
          #if (CLIENT_OS == OS_WIN32)
          /*transparent has no password, and NT manages passwords by itself*/
          if (ss.sstype!=-1 && winGetVersion()>=400 && winGetVersion()<2000)
          {
            /* we skip the password dialog if we have barely started */
            canClose = ((GetTickCount() - ss.initTime) < (1000*5));
          }
          if (!canClose)
          {
            HINSTANCE hpwdcpl = LoadLibrary("PASSWORD.CPL");
            canClose = TRUE;
            if (hpwdcpl != NULL) 
            {
              typedef BOOL (WINAPI *VERIFYSCREENSAVEPWD)(HWND hwnd);
              VERIFYSCREENSAVEPWD VerifyScreenSavePwd;
              VerifyScreenSavePwd = (VERIFYSCREENSAVEPWD)
                       GetProcAddress(hpwdcpl,"VerifyScreenSavePwd");
              if (VerifyScreenSavePwd)
              {
                ss.isDialogActive = 1; 
                SendMessage( hwnd, WM_SETCURSOR, 0, 0 );
                canClose = VerifyScreenSavePwd(hwnd);
                ss.isDialogActive = 0; 
                SendMessage( hwnd, WM_SETCURSOR, 0, 0 );
              }
              FreeLibrary( hpwdcpl );
            }
          }
          #endif
          if (canClose)
            DestroyWindow(hwnd);
        }
        return 0;
      }  
      break;
    }
    case WM_DESTROY:
    {
      SetCursor(LoadCursor(NULL,IDC_ARROW));
      PostQuitMessage(0);
      break;
    }
  }
  return DefWindowProc(hwnd,msg,wParam,lParam);
}

/* ---------------------------------------------------- */

#pragma pack(1)
struct _ifh /* size 20 */
 {
   WORD    Machine;                /* 0 */
   WORD    NumberOfSections;       /* 2 */
   DWORD   TimeDateStamp;          /* 4 */
   DWORD   PointerToSymbolTable;   /* 8 */
   DWORD   NumberOfSymbols;        /* 12 */
   WORD    SizeOfOptionalHeader;   /* 16 */
   WORD    Characteristics;        /* 18 */
};
struct _ofh /* size 224 */
{
   WORD    Magic;                  /*  0 */
   BYTE    MajorLinkerVersion;     /*  2 */
   BYTE    MinorLinkerVersion;     /*  3 */
   DWORD   SizeOfCode;             /*  4 */
   DWORD   SizeOfInitializedData;  /*  8 */
   DWORD   SizeOfUninitializedData;/* 12 */
   DWORD   AddressOfEntryPoint;    /* 16 */
   DWORD   BaseOfCode;             /* 20 */
   DWORD   BaseOfData;             /* 24 */
   DWORD   ImageBase;              /* 28 */
   /* --- end of std header, begin NT specific part */
   DWORD   SectionAlignment;       /* 32 */
   DWORD   FileAlignment;          /* 36 */
   WORD    MajorOSVersion;         /* 40 */
   WORD    MinorOSVersion;         /* 42 */
   WORD    MajorImageVersion;      /* 44 */
   WORD    MinorImageVersion;      /* 46 */
   WORD    MajorSubsystemVersion;  /* 48 */
   WORD    MinorSubsystemVersion;  /* 50 */
   DWORD   Win32VersionValue;      /* 52 always 0? */
   DWORD   SizeOfImage;            /* 56 */
   DWORD   SizeOfHeaders;          /* 60 */
   DWORD   CheckSum;               /* 64 */
   WORD    Subsystem;              /* 68 */
   WORD    DllCharacteristics;     /* 70 */
   DWORD   SizeOfStackReserve;     /* 72 */
   DWORD   SizeOfStackCommit;      /* 76 */
   DWORD   SizeOfHeapReserve;      /* 80 */
   DWORD   SizeOfHeapCommit;       /* 84 */
   DWORD   LoaderFlags;            /* 88 */
   DWORD   NumberOfRvaAndSizes;    /* 92 */
   struct { DWORD VirtualAddress, Size; } DataDirectory[16]; /* 96-224 */
   //IMAGE_DATA_DIRECTORY DataDirectory[IMAGE_NUMBEROF_DIRECTORY_ENTRIES];
};
/* first section begins here */
struct _ish /* size 40 */
{
  BYTE    Name[8];
  union 
  { DWORD   PhysicalAddress;
    DWORD   VirtualSize;
  } Misc;
  DWORD   VirtualAddress;
  DWORD   SizeOfRawData;
  DWORD   PointerToRawData;
  DWORD   PointerToRelocations;
  DWORD   PointerToLinenumbers;
  WORD    NumberOfRelocations;
  WORD    NumberOfLinenumbers;
  DWORD   Characteristics;
};
/* end of fixed headers */
struct _irde 
{
  union __dummy1 {
    struct __dummy2 
    { DWORD NameOffset:31;
      DWORD NameIsString:1;
    };
    DWORD  Name;
    WORD  Id;
  };
  union __dummy3 {
    DWORD   OffsetToData;
    struct __dummy4 {
      DWORD   OffsetToDirectory:31;
      DWORD   DataIsDirectory:1;
    };
  };
};
struct _ird
{
  DWORD   Characteristics;
  DWORD   TimeDateStamp;
  WORD    MajorVersion;
  WORD    MinorVersion;
  WORD    NumberOfNamedEntries;
  WORD    NumberOfIdEntries;
  // followed by struct _irde 
  // ImageResourceDirectoryEntries[NumberOfNamedEntries+NumberOfIdEntries]
};
#pragma pack()

static int SSGetFileData(const char *filename, int *verp, int *isguip,
                          char *ssnbuf, unsigned int ssnbuflen)
{
  int ver = -1;
  int isgui = 0;
  HFILE handle = _lopen( filename, OF_READ|OF_SHARE_DENY_NONE);
  
  if (ssnbuf && ssnbuflen)
    ssnbuf[0] = 0;
  
  if (handle != HFILE_ERROR)
  {
    char rbuf[0x100]; char *p;
    if (_lread( handle, rbuf, sizeof(rbuf) ) == sizeof(rbuf))
    {
      if ((rbuf[0]=='M' && rbuf[1]=='Z') || (rbuf[1]=='M' && rbuf[2]=='Z'))
      {
        DWORD offset;
        p = &rbuf[0x3C]; /*off from start of file to NE/LE image */
        if ((offset = *((DWORD *)p)) != 0)
        {
          if (_llseek( handle, (LONG)offset, 0 ) == ((LONG)offset))
          {
            if (_lread( handle, rbuf, sizeof(rbuf) ) != sizeof(rbuf))
            { /* overwrite in case read was partial */
              rbuf[0] = 'M'; rbuf[1] = 'Z'; 
            }
          }
        }
      }
      if ((rbuf[0]=='N' && rbuf[1]=='E' && 
           rbuf[0x36] == 0x02 /* win */ || rbuf[0x36]==0x04 /* win386 */))
      {
        ver = (rbuf[0x3F]*100)+rbuf[0x3E];
        isgui = 1;
      
        if (ssnbuf && ssnbuflen>1)
        {
          p = &rbuf[0x2C]; /*off from start of *file* to nonres names */
          if (_llseek(handle,((LONG)*((DWORD *)p)),0)!=((LONG)*((DWORD *)p)))
            ;
          else if (_lread( handle, rbuf, sizeof(rbuf)) < 1)
            ;
          else if (!rbuf[0])
            ;
          else
          {
            if (((unsigned int)rbuf[0]) > (sizeof(rbuf)-2))
              rbuf[0] = ((char)(sizeof(rbuf)-2));
            rbuf[1+rbuf[0]]='\0';
            rbuf[0] = rbuf[9]; rbuf[9] = '\0';
            p = &rbuf[((strcmpi(&rbuf[1],"SCRNSAVE")==0)?(9):(1))];
            rbuf[9] = rbuf[0];
            while (*p == ' ' || *p == ':')
              p++;
            if (*p)
            {
              strncpy(ssnbuf,p,ssnbuflen);
              ssnbuf[ssnbuflen-1] = '\0';
            }
          }
        }
      }
      else if (rbuf[0]=='P' && rbuf[1]=='E' && !rbuf[2] && !rbuf[3])
      {
        struct _ifh *ifh;
        struct _ofh *ofh;
        p = &rbuf[4];
        ifh = (struct _ifh *)p;
        p += sizeof(struct _ifh);
        ofh = (struct _ofh *)p;
        if (ifh->SizeOfOptionalHeader >= sizeof(struct _ofh) &&
            ofh->Magic == 0x010B /* file == 0x10B, rom == 0x107 */ )
        {
          ver = (ofh->MajorSubsystemVersion * 100)+
                (ofh->MinorSubsystemVersion);
          ver+= 2000; /* 32bit */
          isgui = (ofh->Subsystem == 2); /* IMAGE_SUBSYSTEM_WINDOWS_GUI */
          
          #if (CLIENT_OS == OS_WIN32)
          if (ssnbuf && ssnbuflen>1)
          {
            HINSTANCE hInst = LoadLibraryEx(filename, NULL,
                LOAD_LIBRARY_AS_DATAFILE | DONT_RESOLVE_DLL_REFERENCES);
            if (hInst) 
            {
              LoadString( hInst, (UINT)1, ssnbuf, (UINT)ssnbuflen);
              FreeLibrary( hInst );
            }
          }
          #endif
        }
      }
    }
    _lclose(handle);
  }
  if (isguip)
    *isguip = isgui;
  if (verp)
    *verp = ((ver==-1)?(0):(ver));
  return ver;
}

/* ---------------------------------------------------- */

static UINT SSRunMessageLoop(void)
{
  UINT rc = 0; MSG msg;
  while (PeekMessage(&msg, NULL, 0, 0, (/*PM_NOYIELD|*/PM_NOREMOVE)))
  {
    GetMessage(&msg,NULL,0,0);
    rc = msg.message; 
    if (msg.message == WM_QUIT)
      break;
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
  return rc;
}

/* ---------------------------------------------------- */

struct w16ProcessTrack { HINSTANCE hInst; HMODULE hModule; };

#if (CLIENT_OS == OS_WIN32)
static BOOL CALLBACK __SSFreeProcessEnumWinProcNT(HWND hwnd,LPARAM lParam)
{
   DWORD thatpid = 0, searchpid = (DWORD)lParam;
   GetWindowThreadProcessId( hwnd, &thatpid );
   if (thatpid == searchpid)
     PostMessage(hwnd, WM_CLOSE, 0, 0);
   return TRUE;
}
#endif
static BOOL CALLBACK __SSFreeProcessEnumWinProc(HWND hwnd,LPARAM lParam)
{   
   HINSTANCE thatpid, searchpid = (HINSTANCE)lParam;
   thatpid = (HINSTANCE)GetWindowLong(hwnd,GWL_HINSTANCE);
   if (thatpid == searchpid)
     PostMessage(hwnd, WM_CLOSE, 0, 0);
   return TRUE;
}

/* only valid if we didn't wait for exit */
int SSFreeProcess( void *handle, int withShutdown ) 
{
  withShutdown = withShutdown; /* shaddup compiler */
  if (handle && handle != ((void *)ULONG_MAX))
  {
    #if (CLIENT_OS == OS_WIN32)
    if (winGetVersion() >= 400) /* win32s used winexec */
    {
      PROCESS_INFORMATION *pidp = (PROCESS_INFORMATION *)handle;
      if (withShutdown) /* shut it down before free() */
      {
        int i;
        for (i=0;i<3;i++)
        {
          if (WaitForSingleObject(pidp->hProcess,0) != WAIT_TIMEOUT)
            break;
          if (i > 0)
          {
            Sleep(500);
            if (WaitForSingleObject(pidp->hProcess,0) != WAIT_TIMEOUT)
              break;
          }
          EnumWindows((WNDENUMPROC)__SSFreeProcessEnumWinProcNT, 
                      ((LPARAM)(pidp->dwProcessId)) );
        }
      }
      CloseHandle(pidp->hProcess);
      CloseHandle(pidp->hThread);
    }
    else
    #endif
    {
      if (withShutdown) /* shut it down before free() */
      {
        int i;
        struct w16ProcessTrack *pidtrack = (struct w16ProcessTrack *)handle;
        UINT hTimer = SetTimer(NULL,0, 500, NULL);
        for (i=0;i<3;i++)
        {
          MSG msg; char buffer[MAX_PATH+1];
          if (!GetModuleFileName(pidtrack->hInst,buffer,sizeof(buffer)))
            break;
          if (GetModuleHandle(buffer) != pidtrack->hModule)
            break;
          while (PeekMessage(&msg,NULL,0,0,PM_REMOVE))
          {
            if (msg.message == WM_QUIT)
            {
              i = -1;
              break;
            }
            if (msg.hwnd == NULL && msg.message == WM_TIMER)
              break;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
          }
          if (!GetModuleFileName(pidtrack->hInst,buffer,sizeof(buffer)))
            break;
          if (GetModuleHandle(buffer) != pidtrack->hModule)
            break;
          if (i < 0)
            break;
          EnumWindows((WNDENUMPROC)__SSFreeProcessEnumWinProc, 
                      ((LPARAM)(pidtrack->hInst)) );
        }   
        if (hTimer)
          KillTimer(NULL,hTimer);
      }
    }
    free(handle);
    return 0;      
  }
  return -1;
}

/* ---------------------------------------------------- */

void *SSLaunchProcess( const char *filename, const char *args, 
                       int waitMode, int prio )
{
  MSG msg;
  char cmdline[MAX_PATH+80+4];
  WORD nCmdShow = 0;

  //if (args && strstr( args, "-hide" ))
  //  nCmdShow = SW_HIDE;

  #if (CLIENT_OS == OS_WIN32)
  if (winGetVersion() >= 400) /* win32s wait needs winexec */
  {
    void *pidp = malloc(sizeof(PROCESS_INFORMATION));
    if (pidp)
    {
      int issuccess = 0;
      UINT hTimer = 0;
      HANDLE hProcess = GetCurrentProcess();
      DWORD classprio = GetPriorityClass(hProcess);
      STARTUPINFO su;
      PROCESS_INFORMATION *pi = (PROCESS_INFORMATION *)pidp;
      memset(pi,0,sizeof(PROCESS_INFORMATION));
      memset((void *)&su,0,sizeof(su));
      su.cb = sizeof(su);

      if (strchr(filename,' '))
        strcat(strcat(strcpy(cmdline, "\""),filename),"\"");
      else
        strcpy(cmdline,filename);
      if (args)
        strncpy(&cmdline[strlen(strcat(cmdline," "))], args, 80);
      cmdline[sizeof(cmdline)-1] = '\0';
   
      if (nCmdShow != 0)
      {
        su.dwFlags |= STARTF_USESHOWWINDOW;
        su.wShowWindow |= nCmdShow;
      }
      if (prio < 0)
        prio = IDLE_PRIORITY_CLASS;
      else if (prio > 0)
        prio = HIGH_PRIORITY_CLASS;
      else
        prio = NORMAL_PRIORITY_CLASS;
  
      if (waitMode < 0)
      {
        waitMode = +1;
        #if 0
        long filever = -waitMode;
        waitMode = -1;
        if (filever < 100)
          filever = SSGetFileData(filename, 0, 0, 0, 0);
        if (filever < 2000) 
        {
          waitMode = +1;
          #if 0
          if ((hTimer = SetTimer(NULL,0, 100, NULL)) == 0)
          {
            MessageBox(NULL,"Insufficient resources to launch 16 bit process",
                             filename,MB_OK|MB_ICONHAND);
            return 0;
          }
          #endif
        }
        #endif
      }
      SetPriorityClass(hProcess,NORMAL_PRIORITY_CLASS);
      if (CreateProcess(NULL,cmdline,NULL,NULL,TRUE,
                        prio, NULL, NULL, &su, (PROCESS_INFORMATION *)pi))
      {
        issuccess = 1;
        if (waitMode < 0) /* hard block - we don't have a window */
        {
          WaitForSingleObject(pi->hProcess,INFINITE);
        }
        else if (waitMode > 0) /* soft block - we have a window (or timer) */
        {
          if (WaitForInputIdle(pi->hProcess,3000) != ((DWORD)-1))
          {
            while (waitMode && WaitForSingleObject(pi->hProcess,300)==WAIT_TIMEOUT)
            {
              while (PeekMessage(&msg,NULL,0,0,PM_REMOVE))
              {
                if (msg.message == WM_QUIT)
                {
                  TerminateProcess(pi->hProcess,0);
                  waitMode = 0;
                  break;
                }
                if (msg.hwnd != NULL || msg.message != WM_TIMER)
                {
                  TranslateMessage(&msg);
                  DispatchMessage(&msg);
                }
              }
            }
          }
        }
        issuccess = 1;
      }
      if (hTimer)
        KillTimer(NULL,hTimer);
      SetPriorityClass(hProcess,classprio);

      if (issuccess)
      {
        if (waitMode)
        {
          CloseHandle(pi->hProcess);
          CloseHandle(pi->hThread);
          free(pidp);
          return ((void *)ULONG_MAX);
        }
        return (pidp);
      }
      free(pidp);
    }
  }
  else
  #endif
  {
    void *pidp = malloc(sizeof(struct w16ProcessTrack));
    if (pidp)
    {
      HINSTANCE that;
      if (nCmdShow == 0)
        nCmdShow = SW_SHOW;
      if (strchr(filename,' '))
        strcat(strcat(strcpy(cmdline, "\""),filename),"\"");
      else
        strcpy(cmdline, filename );
      if (args) 
        strncpy(&cmdline[strlen(strcat(cmdline," "))], args, 80);
      cmdline[sizeof(cmdline)-1] = '\0';

      ((struct w16ProcessTrack *)pidp)->hInst = 0;
      if ((that = ((HINSTANCE)WinExec( cmdline, nCmdShow ))) >= ((HINSTANCE)32))
      {
        HMODULE hModule; 
        GetModuleFileName(that, cmdline, sizeof(cmdline));
        hModule = GetModuleHandle(cmdline);
        ((struct w16ProcessTrack *)pidp)->hInst = that;
        ((struct w16ProcessTrack *)pidp)->hModule = hModule;

        if (waitMode)
        {
          UINT hTimer = SetTimer(NULL,0, 250, NULL);
          #if (CLIENT_OS != OS_WIN32)
          int orefcount = -1;
          int os32file32 = 0;
          //if (winGetVersion()>=400 && SSGetFileData(filename,0,0,0,0)>=2000)
          //  os32file32 = 1;
          #endif
          if (waitMode < 0 && hTimer)
            waitMode = +1;
          while (waitMode) /* dummy while */
          {
            #if (CLIENT_OS != OS_WIN32)
            if (!os32file32)
            {
              int refcount = prio; /* use up prio variable */
              if ((refcount = GetModuleUsage( ((HINSTANCE)that) )) == 0)
                break;
              if (orefcount == -1)
                orefcount = refcount;
              else if (refcount != orefcount)
                break;
            }
            else /* file == win32 *AND* OS == win32 */
            #endif
            {
              if (!GetModuleFileName(that, cmdline, sizeof(cmdline)))
                break;
              if (GetModuleHandle(cmdline) != hModule)
                break;
            }
            if (waitMode > 0) /* we have a window (and hopefully a timer) */
            {
              if (!GetMessage(&msg,NULL,0,0))
                break;
              if (msg.message != WM_TIMER)
              {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
              }
            }
            else /* hope this never gets executed */
            {
              DWORD nowms = 0, endms = 0; 
              do
              {
                Yield();                           
                nowms = GetTickCount(); /* has real msec resolution */
                if (endms == 0)         /* see w32in16.cpp */  
                  endms = nowms + 500;
              } while (nowms <= endms);
            }
          }
          if (hTimer)
            KillTimer(NULL,hTimer);
        }
      } /* WinExec()'d ok */
      
      if (((struct w16ProcessTrack *)pidp)->hInst)
      {
        if (waitMode)
        {
          free(pidp);
          return ((void *)ULONG_MAX);
        }
        return (pidp);
      }
      free(pidp);
    }
    
  }
  return 0;      
}


/* ---------------------------------------------------- */

static HWND SSGetPreviewWindow(void)
{
  HWND hPreview = (HWND)0;
  #if (CLIENT_OS == OS_WIN32)
  if ((winGetVersion()%2000) >= 400)
  {
    HINSTANCE hInst = LoadLibrary("USER32");
    if (hInst != NULL) 
    {
      typedef HWND (WINAPI *FINDPARENTEX)(HWND,HWND,LPSTR,LPSTR);
      FINDPARENTEX FindWndEx = (FINDPARENTEX)
                          GetProcAddress(hInst,"FindWindowExA");
      if (FindWndEx)
      {
        char *sn = "SSDemoParent";
        HWND hParent = GetForegroundWindow();
        if ((hPreview = FindWndEx( hParent, NULL, sn, NULL )) == NULL)
        {
          HWND hCh1 = NULL;
          while ((hCh1 = FindWndEx( hParent, hCh1, NULL, NULL )) != NULL)
          {
            if ((hPreview = FindWndEx( hCh1, NULL, sn, NULL )) == NULL)
            {
              HWND hCh2 = NULL;
              while ((hCh2 = FindWndEx( hCh1, hCh2, NULL, NULL)) != NULL)
              {
                if ((hPreview = FindWndEx( hCh2, NULL, sn, NULL )) == NULL)
                {
                  HWND hCh3 = NULL;
                  while ((hCh3 = FindWndEx( hCh2, hCh3, NULL, NULL )) != NULL)
                  {
                    hPreview = FindWndEx( hCh3, NULL, sn, NULL );
                    if (hPreview)
                      break;
                  }
                }
                if (hPreview)
                  break;
              }
              if (hPreview)
                break;
            }
            if (hPreview)
              break;
          }
        }
      }
      FreeLibrary( hInst );
    }
  }
  #endif
  return hPreview;
}      

/* ---------------------------------------------------- */

static int IsDCTIClientRunning(void)
{
  int found = (FindWindow( NULL, W32CLI_CONSOLE_NAME ) != NULL);
  if (!found)
    found = (FindWindow( NULL, W32CLI_OLD_CONSOLE_NAME ) != NULL);
  #if (CLIENT_OS == OS_WIN32)
  if (!found)
  {
    HANDLE hmutex;
    SECURITY_ATTRIBUTES sa;
    memset(&sa,0,sizeof(sa));
    sa.nLength = sizeof(sa);
    SetLastError(0);
    hmutex = CreateMutex( &sa, TRUE, W32CLI_MUTEX_NAME);
    if (GetLastError() != 0)  /* ie, it exists */
      found = 1;
    ReleaseMutex( hmutex );
    CloseHandle( hmutex );
  }
  #endif
  return found;
}  

/* ---------------------------------------------------- */

static int SSVerifyFileExists( const char *filename )
{
  OFSTRUCT ofstruct;
  ofstruct.cBytes = sizeof(ofstruct);
  #ifndef OF_SEARCH
  #define OF_SEARCH 0x0400
  #endif
  return (OpenFile( filename, &ofstruct, OF_EXIST|OF_SEARCH) != HFILE_ERROR );
}  

/* ---------------------------------------------------- */

#if 1
static int SSSetiControl(const char *) { return 0 };
/* thankfully no longer necessary (sez bug 2195) */
#else
static int SSSetiControl(const char *ssname) 
{ 
  /* I wish, oh, I wish this wasn't needed.
     a) if we're about to launch a screensaver...
        1) if the ss _is_ seti, then make sure the seticlient is already 
           running otherwise the screensaver will exit right away. Duh!
        2) if the ss _is_not_ seti, then ensure the seti client is _not_
           running otherwise the seti client will intercept the ss
           WM_SYSCOMMANDs and everything goes all to hell. Double-Duh!
     b) the screensaver has returned control to us...
        ensure the seti client is no longer running otherwise the seticlient 
        client will intercept the next ss WM_SYSCOMMANDs and nothing works.
     Whoever wrote that piece of shit software needs to learn that although
     it aint rocket science, one still needs to play by the rules.
  */        
  //static char seti[]={'S'|0x80,'E'|0x80,'T'|0x80,'I'|0x80,'@'|0x80,'H'|0x80,'o'|0x80,'m'|0x80,'e'|0x80,' '|0x80,
  //                    'C'|0x80,'l'|0x80,'i'|0x80,'e'|0x80,'n'|0x80,'t'|0x80,'\0'};
  #ifdef _MSC_VER
  #pragma warning(disable:4305) /* warning C4305: 'initializing' : truncation from 'const int' to 'char' */
  #pragma warning(disable:4310) /* warning C4310: cast truncates constant value */
  #endif
  static char seticlass[]={(char)('S'|0x80),(char)('e'|0x80),(char)('t'|0x80),(char)('i'|0x80),
                           (char)('C'|0x80),(char)('l'|0x80),(char)('i'|0x80),(char)('e'|0x80),(char)('n'|0x80),(char)('t'|0x80),
                           (char)('P'|0x80),(char)('a'|0x80),(char)('r'|0x80),(char)('e'|0x80),(char)('n'|0x80),(char)('t'|0x80),'\0'};
  if (seticlass[0]!='S')
  {
    int i;
    for (i=0;seticlass[i];i++)
      seticlass[i]^=0x80;
  }  
  if (ssname) /* start */
  {
    char path[MAX_PATH+2];
    char *p = (char*) strrchr(ssname, '\\');
    char *q = (char*) strrchr(ssname, '/');
    if (q > p) p = q;
    q = (char*) strrchr(ssname, ':');
    if (q > p) p = q;
    strcpy( path, ((p)?(p+1):(ssname)) );
    p = strchr( path, '.' );
    if (p && strcmpi(p, ".scr") == 0)
    {
      char ssss[10];
      *p = '\0';
      memcpy((void *)&ssss[0],(void *)&seticlass[0],4);
      ssss[4]='h';ssss[5]='o';ssss[6]='m';ssss[7]='e';ssss[8]='\0';
      if ( strcmpi( path, ssss ) == 0 )
      {
        int started = 1;
        if ( FindWindow( seticlass, NULL ) == NULL)
        {
          started = 0;
          strcat( path, ".ini" );
          if (GetPrivateProfileString("boot", "ClientPath", 
                   "", &path[1], sizeof(path)-1, path) != 0)
          {
            p = &path[1];
            if (strchr( p,' ' ))
            {
              *--p = '\"'; 
              strcat( p, "\"" );
            }
            strcat( p, " -min" );
            if (WinExec( p, SW_MINIMIZE ) >= 32)
            {
              DWORD tick = 0, tock = 0;
              while (tick <= tock)
              {
                started = (FindWindow( seticlass, NULL ) != NULL);
                if (started)
                  break;
                if (SSRunMessageLoop() == WM_QUIT)
                  break;
                tick = GetTickCount();
                if (tock == 0)
                  tock = tick + 1000;
                #if (CLIENT_OS == OS_WIN32)
                Sleep(50);
                #else
                Yield();
                #endif
              }  
            }
          }
        } 
        if (started)
          return +1; /* Seti client started */
        return -1; /* Seti client failed to start */
      }
    }
    /* not the seti screensaver. stop seti client */
    ssname = NULL;
  }
  if (!ssname)
  {
    /* grr. SETIclient does not respond to WM_CLOSE. Only WM_DESTROY works */
    HWND hwnd; int which, found = 0;
    char childclass[20]; /* "SetiClientParent" and "SetiClientChild" */
    strcpy(childclass,seticlass); 
    strcpy(&childclass[10],"Child");
    for (which = 0;found < 100 && which < 2; which++)
    {
      char *classname = ((which==0)?(seticlass):(childclass));
      found = 0;
      while ((++found)<100 && (hwnd = FindWindow( classname, NULL ))!=NULL) 
      {
        SendMessage(hwnd,WM_DESTROY,0,0);
        if (SSRunMessageLoop() == WM_QUIT)
          found = 100;
        else
        {
          #if (CLIENT_OS == OS_WIN32)
          Sleep(50);
          #else
          Yield();
          #endif
        }
      }
    }
  }
  return 0;
}    
#endif

/* ---------------------------------------------------- */

#if 0
static LRESULT CALLBACK ExternalPreviewProc(HWND hwnd,UINT msg,
                                            WPARAM wParam,LPARAM lParam)
{
  static HWND hwndChild = NULL;
  switch(msg)
  {
    case WM_PARENTNOTIFY:
    {
      switch (LOWORD(wParam))
      {
        case WM_CREATE:
        {
          hwndChild = (HWND)lParam;
        }
        case WM_DESTROY:
        {
          if (hwndChild == ((HWND)lParam))
          {
            hwndChild = 0;
            PostMessage(hwnd,WM_DESTROY,0,0);
          }
        }  
      }
      return 0;
    }    
    default:
    {
      if (hwndChild)
        return SendMessage(hwndChild,msg,wParam,lParam);
      if (msg == WM_DESTROY)
      {
        hwndChild = NULL;
        PostQuitMessage(0);
      }
      break;
    }
  }
  return DefWindowProc(hwnd,msg,wParam,lParam);
}
#endif

/* ---------------------------------------------------- */

static int SSDoSaver(HINSTANCE hInstance, HWND hParentWnd )
{ 
  char path[MAX_PATH+2];
  int sstype = SSGetProfileInt("type",0);
  int inPreview = 0;
  void *clientHandle = 0;
  ATOM syncAtom = 0;
  
  if (hParentWnd)
  {
    inPreview = 1;
    if (hParentWnd == GetDesktopWindow()) //internal preview
      hParentWnd = NULL;
  }
  
  if (!inPreview && !IsDCTIClientRunning())
  {
    if (SSGetProfileString("launch","",path,sizeof(path))!=0)
    { 
      if (GlobalFindAtom( W32CLI_SSATOM_NAME ) == 0)
        syncAtom = GlobalAddAtom( W32CLI_SSATOM_NAME );
      if (syncAtom)
        clientHandle = SSLaunchProcess( path, "-hide -priority 0", 0, 0);
    }
  }

  if (sstype > 0)
  {
    if (SSGetProfileString("file","",path,sizeof(path)) == 0)
      sstype = 0;
    else if (!SSVerifyFileExists(path))
      sstype = 0;
    else if (SSSetiControl(path) >= 0) /* is not seti, or was started ok */
    {
      int filever = SSGetFileData(path,0,0,0,0);
      if (filever < 0)
        sstype = 0;
      else if (!hParentWnd)
        SSLaunchProcess( path, "/s", -filever, -1 ); /* fullblock wait */
      else if ((filever%2000)<400) /* < 32bit 4.0 */
        sstype = 0;
      else
      {
        char cmdline[32];
        void *handle;
        sprintf( cmdline, "/p %lu", (long)hParentWnd );
        handle = SSLaunchProcess( path, cmdline, 0 /*-filever*/, 0 );
        if (handle) 
          SSFreeProcess( handle, 0 ); /* free, but don't kill it */
      }
    }
  } 

  SSSetiControl(NULL); /* stop seti (if running) */
  
  if (sstype <= 0)
  {
    char classname[64];
    WNDCLASS wc;
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = (WNDPROC)SaverWindowProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;
    wc.hIcon = NULL;
    wc.hCursor = NULL;
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszMenuName = NULL;
    #if (CLIENT_OS == OS_WIN32)
    strcpy( classname, "ScrClass" );
    #else
    sprintf( classname, "ScrClass%d", hInstance);
    #endif
    wc.lpszClassName = classname;

    if (sstype < 0)
    {
      if (!SSIsTransparencyAvailable())
        sstype = 0;
      else     
        wc.hbrBackground = NULL;   /* transparent */
    }
              
    if ( RegisterClass(&wc) )
    {
      //in preview mode, hwnd is the parent window, in runmode, hwnd is null
      HWND hFocus = NULL;
      HWND hScrWindow;
      DWORD exbits = 0, wsbits = 0;
      int cx = 0, cy = 0, ox = 0, oy = 0;

      if (hParentWnd)
      { 
        RECT rc; GetWindowRect(hParentWnd, &rc);
        cx = rc.right-rc.left; 
        cy = rc.bottom-rc.top;  
        wsbits = WS_CHILD|WS_VISIBLE;
      }
      else
      {
        cx=GetSystemMetrics(SM_CXSCREEN); 
        cy=GetSystemMetrics(SM_CYSCREEN);
#ifdef WITH_MULTIMON_SUPPORT
        if (((winGetVersion()>=410 && winGetVersion()<2000) ||
            winGetVersion()>=2500) && GetSystemMetrics(SM_CMONITORS)>1)
        {
          cx=GetSystemMetrics(SM_CXVIRTUALSCREEN); 
          cy=GetSystemMetrics(SM_CYVIRTUALSCREEN);
          ox=GetSystemMetrics(SM_XVIRTUALSCREEN); 
          oy=GetSystemMetrics(SM_XVIRTUALSCREEN); 
        }
#endif
        exbits = WS_EX_TOPMOST;
        wsbits = WS_POPUP|WS_VISIBLE;
        if (wc.hbrBackground == NULL) /* transparent */
        {
          exbits |= WS_EX_TRANSPARENT;
          #if defined(WS_EX_TOOLWINDOW)
          exbits |= WS_EX_TOOLWINDOW;
          #endif
          hFocus = GetFocus();
        }
      }
      hScrWindow = CreateWindowEx(exbits, wc.lpszClassName, "",
                   wsbits, (UINT)ox, (UINT)oy, (UINT)cx, (UINT)cy, 
                   hParentWnd, NULL, hInstance, NULL );
      if (hScrWindow)
      {
        MSG msg;
        if (hFocus)
          SetFocus(hFocus);
        #if (defined(SPI_SCREENSAVERRUNNING))
        if (hParentWnd == NULL)
        {
          UINT oldval = 0;
          SystemParametersInfo(SPI_SCREENSAVERRUNNING,1,&oldval,0);
        }
        #endif
        while ( GetMessage( &msg, NULL, 0, 0 ) )
        { 
          TranslateMessage( &msg );
          DispatchMessage( &msg );
        }
        #if (defined(SPI_SCREENSAVERRUNNING))
        if (hParentWnd == NULL)
        {
          UINT oldval = 0;
          SystemParametersInfo(SPI_SCREENSAVERRUNNING,0,&oldval,0);
        }
        #endif
      }
      UnregisterClass( wc.lpszClassName, hInstance );
    }
  }

  if (clientHandle)
  {
    //HWND otherguy = FindWindow( NULL, W32CLI_CONSOLE_NAME );
    //if (otherguy) SendMessage( hwnd, WM_CLOSE, 0, 0 );
    SendMessage( HWND_BROADCAST, syncAtom, DNETC_WCMD_SHUTDOWN, 0 );
    SSFreeProcess( clientHandle, 1 ); /* shut it down first before free */
  }
  if (syncAtom)
  {
    GlobalDeleteAtom( syncAtom );
    syncAtom = 0;
  }
  return 0;
}  

/* ---------------------------------------------------- */

#if defined(__WINDOWS_386__)
BOOL PASCAL VerQueryValue(const void * lpvBlock, LPCSTR lpszSubBlock, 
                          VOID ** lplpBuf, UINT * lpcb)
{
  int rc = 0;
  if (lpcb) 
    *lpcb = 0;
  if (lpszSubBlock && lpvBlock)
  {
    UINT olderrmode = SetErrorMode(SEM_NOOPENFILEERRORBOX);
    HINSTANCE hLib = LoadLibrary("VER.DLL");
    SetErrorMode(olderrmode);
    if (hLib > HINSTANCE_ERROR)
    {
      FARPROC lpfn = GetProcAddress( hLib, "VerQueryValue" );
      if (lpfn)
      {
        UINT len = 0;
        DWORD tgt = 0, alias = (DWORD)AllocAlias16( (void *)lpvBlock );
        rc = 0xffff & _Call16(lpfn,"dppp",alias,lpszSubBlock,&tgt,&len);
        if (rc)
        { 
          if (lplpBuf)
            *lplpBuf = MapAliasToFlat( tgt );
          if (lpcb)
            *lpcb = len;
        }
        FreeAlias16(alias);
      }        
      FreeLibrary(hLib);
    }
  }
  return (BOOL)(rc != 0);
}
#endif

/* ---------------------------------------------------- */

static int SSGetFileVersionInfo(const char *filename, 
                     char *filedescr, unsigned int filedescrlen,
                     char *companyname, unsigned int companynamelen,
                     char *versionstr, unsigned int versionstrlen )
{
  int retval = 0;
  DWORD dwHandle, dwLen;

  #if (CLIENT_OS == OS_WIN32)
  DWORD (APIENTRY *__GetFileVersionInfoSize)(LPSTR, LPDWORD);
  BOOL (APIENTRY *__GetFileVersionInfo)(LPSTR, DWORD, DWORD, LPVOID );
  BOOL (APIENTRY *__VerQueryValue)(const LPVOID, LPSTR, LPVOID *, PUINT);
  HINSTANCE hVerLib = LoadLibrary("VERSION.DLL");
  if (!hVerLib)
    return 0;
  __GetFileVersionInfoSize = (DWORD (APIENTRY *)(LPSTR, LPDWORD))
                            GetProcAddress(hVerLib, "GetFileVersionInfoSizeA");
  __GetFileVersionInfo = (BOOL (APIENTRY *)(LPSTR, DWORD, DWORD, LPVOID ))
                            GetProcAddress(hVerLib, "GetFileVersionInfoA");
  __VerQueryValue = (BOOL (APIENTRY *)(const LPVOID, LPSTR, LPVOID *, PUINT))
                            GetProcAddress(hVerLib, "VerQueryValueA");
  if (!__GetFileVersionInfoSize || !__GetFileVersionInfo || !__VerQueryValue)
  {
    FreeLibrary(hVerLib);
    return 0;
  }
  #else
  #define __GetFileVersionInfoSize GetFileVersionInfoSize
  #define __VerQueryValue          VerQueryValue
  #define __GetFileVersionInfo     GetFileVersionInfo
  #endif

  dwHandle = 0;
  dwLen = __GetFileVersionInfoSize( ((char *)filename), &dwHandle );
  if ( dwLen )
  {
    BYTE abDataBuf[2048];
    BYTE *abData = &abDataBuf[0];
    void *mallocbuf = NULL;
    if (dwLen > sizeof(abDataBuf))
    {
      mallocbuf = malloc(dwLen);
      abData = (BYTE *)mallocbuf;
    }
    if (abData)
    {
      if (__GetFileVersionInfo( ((char *)filename), dwHandle, dwLen, abData))
      {
        void *cname; UINT dwSize;
        unsigned int tgtpos;
        char szName[128];
        int namelen = strlen(strcpy(szName,"\\StringFileInfo\\040904E4\\"));
        struct {const char *label; char *buf; unsigned int buflen;} tgtarray[3];

        tgtarray[0].label = "FileDescription";
        tgtarray[0].buf = filedescr;
        tgtarray[0].buflen = filedescrlen;
        tgtarray[1].label = "CompanyName";
        tgtarray[1].buf = companyname; 
        tgtarray[1].buflen = companynamelen;
        tgtarray[2].label = "FileVersion";
        tgtarray[2].buf = versionstr;
        tgtarray[2].buflen = versionstrlen;
        
        for (tgtpos = 0;tgtpos < 3; tgtpos++ )
        {
          if (tgtarray[tgtpos].buf && tgtarray[tgtpos].buflen)
          {
            char *buf = tgtarray[tgtpos].buf;
            unsigned int maxlen = tgtarray[tgtpos].buflen;
            *buf = 0;
            if (maxlen > 1)
            {
              retval++;
              strcpy( &szName[namelen], tgtarray[tgtpos].label );
              if (__VerQueryValue(abData, szName, &cname, &dwSize))
              {
                if (dwSize)
                {
                  unsigned int cpypos = 0;
                  char *cpyptr = (char *)cname;
                  maxlen--;
                  while (cpypos < maxlen && cpypos < dwSize)
                    buf[cpypos++] = *cpyptr++;
                  buf[cpypos] = '\0';
                }
              }    
            }  
          }
        }
      }
    }
    if (mallocbuf)
      free(mallocbuf);
  }
  #if (CLIENT_OS == OS_WIN32)
  FreeLibrary(hVerLib);
  #endif
  return retval;
}

/* ---------------------------------------------------- */

static const char *SSGetDCTIFileVersion(const char *filename, 
                                        int dctionly, int withtype)
{
  char companyBuf[25]; char versionBuf[40]; char filedescrBuf[128];

  if (SSGetFileVersionInfo(filename, filedescrBuf, sizeof(filedescrBuf),
                           companyBuf, sizeof(companyBuf),
                           versionBuf, sizeof(versionBuf) ) )
  {
    if (versionBuf[0])
    {
      companyBuf[21]='\0';
      if (strcmp( companyBuf, "Distributed Computing") != 0)
      {
        if (!dctionly)
          withtype = 0;
        else
          filedescrBuf[0] = '\0';
      }
      if (filedescrBuf[0])
      {
        static char verstring[sizeof(filedescrBuf)+30];
        unsigned int rpos = 0;
        unsigned int wpos = strlen( strcpy( verstring, filedescrBuf ) );
        if ((wpos+2) < (sizeof(verstring)-1))
        {
          verstring[wpos++]=' ';
          verstring[wpos++]='v';
        }  
        while (wpos<(sizeof(verstring)-1) && versionBuf[rpos])
          verstring[wpos++] = versionBuf[rpos++];
        verstring[wpos] = '\0';
        if (withtype && (wpos < (sizeof(verstring)-9)))
        {
          int filever = SSGetFileData(filename,0,0,0,0);
          if (filever>=2000)
            strcat(verstring," (32bit)");
          else if (filever>=100)
            strcat(verstring," (16bit)");
        }
        return (const char *)&verstring[0];
      }
    }
  }
  return NULL;
}      

/* ---------------------------------------------------- */

static int SSExtractScreennameFromFile( const char *filename, 
                      char *screenname, unsigned int buflen)
{
  /* what this function _should_ do is extract the app description
     for win16 apps and extract #1 from the string table for 32 bit apps.
     and return the filebasename only when those fail. 
     We're ignorant and just do it the win95 way (use filebasename).
  */ 
  unsigned int len = 0; int ver;

  if (SSGetFileData(filename, &ver, 0, screenname, buflen) >= 0)
    len = strlen(screenname);  

  if (len == 0 && SSVerifyFileExists( filename ))
  {
    unsigned int upcnt = 0, locnt = 0;
    char ch, prevchar; const char *ptr;

    if ((ptr = (const char *)strrchr( filename, '\\' )) != NULL)
      filename = ptr+1;
    ptr = (const char *)strrchr( filename, '.' );
    prevchar = ' ';
    while (len < (buflen-1) && *filename && filename != ptr)
    {
      ch = (char)(*filename++);
      if (ch == '\t')
        ch = ' ';
      if (ch == ' ' && prevchar == ' ')
        continue;
      upcnt++;
      locnt++;
      if (ch >= 'a' && ch <= 'z')
        upcnt--;
      else if (ch >= 'A' && ch <= 'Z')
        locnt--;
      screenname[len++] = prevchar = ch;
    } 
    screenname[len]='\0';
    if (len && ((upcnt == len) || (locnt == len)))
    {
      len = 0; prevchar = ' ';
      while (screenname[len])
      {
        ch = screenname[len];
        if (prevchar != ' ')
        {
          if (ch >= 'A' && ch <= 'Z')
            screenname[len] |= ' ';
        }
        else if (ch >= 'a' && ch <= 'z')
          screenname[len] &= ~(' ');
        len++;
        prevchar = ch;
      }  
    }    
  }
  memset((char *)&screenname[len],' ',buflen-len);
  screenname[buflen-1] = '\0';
  return len;
}  

/* ---------------------------------------------------- */

#if (CLIENT_OS != OS_WIN32)
#include <direct.h>
#endif

static const char *SSFindNext(void *dirbase, char *buf, unsigned int bufsize)
{
  if (dirbase)
  {
    #if (CLIENT_OS == OS_WIN32)
    WIN32_FIND_DATA fdata;
    if (!FindNextFile( (HANDLE)dirbase, &fdata))
      return (const char *)0;  
    strncpy( buf, fdata.cFileName, bufsize );
    #else
    DIR *dir;
    if ((dir = readdir( (DIR *)dirbase )) == ((DIR *)0))
      return (const char *)0;
    strncpy(buf,dir->d_name,bufsize);
    #endif
    buf[bufsize-1] = '\0';
    return buf;
  }
  return (const char *)0;  
}

/* ---------------------------------------------------- */

static int SSFindClose(void *dirbase)
{
  if (dirbase)
  {
    #if (CLIENT_OS == OS_WIN32)
    FindClose((HANDLE)dirbase);
    #else
    closedir((DIR *)dirbase);
    #endif
    return 0;
  }
  return -1;
}

/* ---------------------------------------------------- */

static void *SSFindFirst(const char *path, char *buf, unsigned int bufsize)
{
  #if (CLIENT_OS == OS_WIN32)
  WIN32_FIND_DATA fdata;
  HANDLE dirbase = FindFirstFile( path, &fdata );
  if (dirbase == INVALID_HANDLE_VALUE)
    return (void *)0;
  strncpy( buf, fdata.cFileName, bufsize );
  buf[bufsize-1] = '\0';
  #else
  DIR *dirbase = opendir(path);
  if ((dirbase != ((DIR *)0)) && !SSFindNext((void *)dirbase, buf, bufsize))
    dirbase = (SSFindClose(dirbase)?((DIR *)0):((DIR *)0));
  #endif
  return (void *)dirbase;
}

/* ---------------------------------------------------- */

#if defined(__WINDOWS_386__)
static int SSCB_FINDSTRINGEXACT(HWND hwnd,const char *screenname)
{
  LONG rc;
  DWORD alias = AllocAlias16((void *)screenname);
  if (!alias) return CB_ERR;
  rc = _16SendMessage( hwnd, CB_FINDSTRINGEXACT, -1, (LPARAM)alias);
  FreeAlias16(alias);
  return rc;
}  
#else
#define SSCB_FINDSTRINGEXACT(_hwnd,_sname) \
   ((int)SendMessage(_hwnd,CB_FINDSTRINGEXACT,((WPARAM)-1),(LPARAM)(_sname)))
#endif     

/* ---------------------------------------------------- */

static struct sslstruct
{
  struct sslstruct *next;
  char screenname[32];
  char filename[MAX_PATH+2];
  int type;
} *ssl = NULL; 

static struct sslstruct ssl_blank={NULL,"Blank Screen",{0},0};
static struct sslstruct ssl_trans={NULL,"(None::transparent)",{0},-1};

static int _ConstructSSList(HWND hwnd, int *oseltype, const char *selname)
{
  struct sslstruct *tail = ssl = NULL;
  int pos, addcount = 0, seltype = *oseltype;
  const char *selstring = NULL;
  
  pos = strlen(ssl_blank.screenname);
  memset((char *)&ssl_blank.screenname[pos],' ',sizeof(ssl_blank.screenname)-pos);
  ssl_blank.screenname[sizeof(ssl_blank.screenname)-1]='\0';
  pos = strlen(ssl_trans.screenname);
  memset((void *)&ssl_trans.screenname[pos],' ',sizeof(ssl_trans.screenname)-pos);
  ssl_trans.screenname[sizeof(ssl_trans.screenname)-1]='\0';

  pos = (int)SendMessage( hwnd, CB_ADDSTRING, 0, 
                          (LPARAM)(&ssl_blank.screenname[0]) );

  if (pos != CB_ERR && pos != CB_ERRSPACE)
  {
    tail = ssl = &ssl_blank; 
    tail->type = 0; //blank screen
    tail->next = NULL;
    addcount++;
    if (seltype == 0)
      selstring = tail->screenname;

    if (SSIsTransparencyAvailable())
    {
      pos = (int)SendMessage( hwnd, CB_ADDSTRING, 0, 
                              (LPARAM)(&ssl_trans.screenname[0]) );
      if (pos != CB_ERR && pos != CB_ERRSPACE)
      {
        tail->next = &ssl_trans;
        tail = tail->next;
        tail->type = -1; //transparent
        tail->next = NULL;
        addcount++;
        if (seltype < 0)
          selstring = tail->screenname;
      }
    }
  }
        
  if (pos != CB_ERR && pos != CB_ERRSPACE)
  {
    int curtype; char *basename;
    char ourfile[MAX_PATH+2];
    
    if (GetModuleFileName((HINSTANCE)GetWindowLong(hwnd,GWL_HINSTANCE),
                          ourfile,sizeof(ourfile)) == 0)
      ourfile[0] = '\0';
    else if ((basename = strrchr(ourfile, '\\')) != NULL)
      strcpy( ourfile, basename+1 );
    if ((basename = strrchr( ourfile, '.' )) != NULL)
      *basename = '\0';
    strcat( ourfile, ".scr" );

    for (curtype = 1; pos != CB_ERRSPACE && curtype <= 2; curtype++ )
    {
      char searchpath[MAX_PATH+2];
      unsigned int len;
      void *dirhandle = (void *)0; 
      char dirbuf[MAX_PATH+2];

      if (curtype == 1)
        len = GetWindowsDirectory(searchpath,sizeof(searchpath));
      else
        len = GetSystemDirectory(searchpath,sizeof(searchpath));

      if (len != 0)
      {
        if (searchpath[len-1] == '\\')
          searchpath[--len] = '\0';
        strcat(searchpath,"\\*.scr");
        dirhandle = SSFindFirst(searchpath, dirbuf, sizeof(dirbuf));
        searchpath[len++] = '\\';
        searchpath[len] = '\0';
      }
    
      if (dirhandle)
      {
        do
        {
          struct sslstruct *next = 
             (struct sslstruct *)malloc(sizeof(struct sslstruct));
          if (!next)
            break;
          if ((basename = strrchr( dirbuf, '\\' )) == NULL)
            basename = strrchr( dirbuf, '/' );
          if (basename == NULL)
          {
            basename = dirbuf;
            len = strlen(strcpy(next->filename,searchpath));
            strncpy( &next->filename[len], dirbuf,
                     sizeof(next->filename)-(len+1));
          }
          else
          {
            strncpy( next->filename, dirbuf, sizeof(next->filename));
            basename++;
            len = 0;
          }
          next->filename[sizeof(next->filename)-1] = '\0';

          if (strcmpi( basename, ourfile ) == 0)
            pos = CB_ERR;
          else if (SSGetDCTIFileVersion(next->filename,1,0)) //in case there
            pos = CB_ERR;                                    //is more than one
          else if (SSExtractScreennameFromFile(next->filename,
               next->screenname, sizeof(next->screenname)) == 0)
            pos = CB_ERR;
          else if (SSCB_FINDSTRINGEXACT( hwnd, next->screenname ) != CB_ERR)
            pos = CB_ERR;
          else 
            pos = (int)SendMessage( hwnd, CB_ADDSTRING, 0,
                      (LPARAM)(next->screenname) );

          if (pos == CB_ERR || pos == CB_ERRSPACE)
          {
            free((void *)next);
            if (pos == CB_ERRSPACE)
              break;
            pos = 0;
          }
          else
          {
            next->type = curtype;
            next->next = NULL;
            tail->next = next;
            tail = next;
            addcount++;
            if (seltype > 0 && selstring == NULL && 
                strcmp( selname, next->filename ) == 0 )
              selstring = tail->screenname;
          }
        } while (SSFindNext( dirhandle, dirbuf, sizeof(dirbuf)));
        SSFindClose(dirhandle);
      }
    } /* for (curtype = 1; curtype <= 2; curtype++ ) */
  } /* if (pos != CB_ERR && pos != CB_ERRSPACE) */

  if (!selstring)
  {
    seltype = 0;
    selstring = ssl_blank.screenname;
  }
  
  if (addcount != 0)
    SendMessage( hwnd, CB_SELECTSTRING, ((WPARAM)-1), (LPARAM)((LPCSTR)selstring));
  
  *oseltype = seltype;
  EnableWindow( hwnd, (addcount>1) );
  return addcount;
}  

/* ---------------------------------------------------- */

static int _MapSSList(HWND hwnd, int *seltype, char *selname)
{
  if (hwnd && IsWindowEnabled(hwnd))
  {
    char buf[sizeof(ssl->screenname)+1];
    int sel = (int)SendMessage( hwnd, CB_GETCURSEL, 0, 0 );  
    if (sel != CB_ERR)
    {
      sel = (int)SendMessage( hwnd, CB_GETLBTEXT, (UINT)sel, (LPARAM)&buf[0] );
      if (sel != CB_ERR)
        buf[sel] = '\0';
    }
    if (sel != CB_ERR)
    {
      struct sslstruct *tail = ssl;
      while (tail)
      {
        if (strcmp( buf, tail->screenname )==0)
        {
          *seltype = tail->type;
          strcpy( selname, tail->filename );
          return sel;
        }
        tail = tail->next;
      }  
    }
  }
  *seltype = 0;
  *selname = '\0';
  return CB_ERR;
}  

/* ---------------------------------------------------- */

static int _DestroySSList(HWND hwnd, int *seltype, char *selname)
{
  int sel = CB_ERR;
  if (seltype && selname)
    sel = _MapSSList( hwnd, seltype, selname );
  while (ssl)
  {
    struct sslstruct *head = ssl;
    ssl = ssl->next;
    if (head->filename[0])
      free((void *)head);
  }
  return sel;
}

/* ---------------------------------------------------- */

#if 0 //(CLIENT_OS == OS_WIN32)
static LRESULT CALLBACK SSGenChildWndProc(HWND hwnd,UINT msg,
                                          WPARAM wParam,LPARAM lParam)
{
  //if (msg == WM_CREATE)
  //  ShowWindow(hwnd,SW_HIDE); 
  return DefWindowProc(hwnd,msg,wParam,lParam);
}

static HWND SSGenChild(HWND hParentWnd)
{
  if (hParentWnd)
  {
    WNDCLASS wc;
    wc.style = CS_SAVEBITS|CS_BYTEALIGNWINDOW|CS_DBLCLKS;
    wc.lpfnWndProc = (WNDPROC)SSGenChildWndProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = 0; //(HINSTANCE)GetWindowLong( hParentWnd,GWL_HINSTANCE);
    wc.hIcon = NULL;
    wc.hCursor = NULL;
    wc.hbrBackground = NULL; //(HBRUSH)GetStockObject(BLACK_BRUSH); //NULL
    wc.lpszMenuName = NULL;
    wc.lpszClassName = "ssGenChild1"; //dialog
    if ( RegisterClass(&wc) )
    {
      HWND hScrWindow;
      RECT rc; GetClientRect(hParentWnd, &rc);
      hScrWindow = CreateWindow(wc.lpszClassName, "blah", WS_CHILD,
                   0,0, rc.right-rc.left, rc.bottom-rc.top, hParentWnd, 
                   NULL, wc.hInstance, NULL );
      if (hScrWindow)
        return hScrWindow;
//MessageBox(NULL,"childwin not created","blah",MB_OK);        
      UnregisterClass( wc.lpszClassName, wc.hInstance );
    }
  }
  return NULL;
}

static void SSUnGenChild(HWND hChildWnd)
{
  if (hChildWnd)
  {
    HINSTANCE hInst = (HINSTANCE)GetWindowLong( hChildWnd,GWL_HINSTANCE);
    if (IsWindow(hChildWnd))
      DestroyWindow(hChildWnd);
    UnregisterClass("ssGenChild1",hInst);
  }
  return;
}
#endif

/* ---------------------------------------------------- */

#if (CLIENT_OS == OS_WIN32)
//<BovineMoo> comdlg32.dll has a very huge memory footprint
//<BovineMoo> pulls in like 20 dlls or so
//<BovineMoo> well, comdlg32 pulls in winspool, which pulls in activeds, 
//<BovineMoo> which pulls in tons of network and crypto things
//<BovineMoo> (on win2k)
//<BovineMoo> actuall winspool is delay loaded
//<BovineMoo> but it's still pretty bad
BOOL __GetOpenFileName(LPOPENFILENAME lpofn)
{
  BOOL rc = 0;
  HINSTANCE hinst = LoadLibrary("COMDLG32.DLL");
  if (hinst)
  {
    FARPROC GetOFN = GetProcAddress(hinst, "GetOpenFileNameA" );
    if (GetOFN)
      rc = (*((BOOL (WINAPI *)(LPOPENFILENAME))GetOFN))( lpofn );
    FreeLibrary( hinst );
  }
  return rc;
}
#else   
#define __GetOpenFileName(__ofnp) GetOpenFileName(__ofnp) 
#endif


static BOOL CALLBACK SSConfigDialogProc( HWND dialog,
                    UINT msg, UINT wparam, LONG lparam )
{
  static char dlghelptext[80];
  static HWND dlghlpWnd = NULL;
  static HWND sslistWnd = NULL;
  static int running_child = 0;

  if (msg == WM_INITDIALOG)
  {
    HWND hwnd;  int sslisttype = 0;

    hwnd = GetDlgItem( dialog, 403 /* SSCONFIG_FNMRU */ );
    if (szAlternateAppName[0])
    {
      SetWindowText( dialog, szAlternateAppName );
      if (hwnd)
        ShowWindow( hwnd, SW_HIDE );
      hwnd = GetDlgItem( dialog, 402 /* SSCONFIG_FNCAPTION */ ); 
      if (hwnd)
        SetWindowText(hwnd, "Application to launch in the background:");
      hwnd = GetDlgItem( dialog, 405 /* SSCONFIG_FNCONFIG */ ); 
      if (hwnd)
        ShowWindow( hwnd, SW_HIDE );
    }
    else if (hwnd)
    {
      SendMessage( hwnd, BM_SETCHECK, 1 /* BST_CHECKED */, 0 );
      EnableWindow( hwnd, 0 );
    }

    dlghlpWnd = GetDlgItem( dialog, 450 /* dialog help text */);
    dlghelptext[0]='\0';
    
    sslistWnd = GetDlgItem( dialog, 408 /* SSCONFIG_SSLIST */ );
    if (sslistWnd)
    {
      char sslistfname[MAX_PATH+2];
      sslisttype = SSGetProfileInt("type",0);
      if (sslisttype < 0 && !SSIsTransparencyAvailable())
        sslisttype = 0;
      else if (sslisttype > 0)
      {
        sslisttype = 1;
        if (SSGetProfileString("file","",sslistfname,sizeof(sslistfname))==0)
          sslisttype = 0;
      }
      if (_ConstructSSList(sslistWnd, &sslisttype, sslistfname ) !=0 )
        SendMessage( sslistWnd, CB_SETEXTENDEDUI, 1, 0L );
      else
      {
        EnableWindow( sslistWnd, 0 );
        sslistWnd = NULL;
      }
    }
    hwnd = GetDlgItem( dialog, 409 /*SSCONFIG_LISTCONFIG */ );
    if (hwnd)
      EnableWindow( hwnd, (sslisttype > 0));
    lparam = 0; wparam = 406; /*SSCONFIG_FN */
    #if (CLIENT_OS == OS_WIN32)
    wparam |= (EN_CHANGE << 16);
    #else
    lparam |= (EN_CHANGE << 16);
    #endif
    SSConfigDialogProc( dialog, WM_COMMAND, wparam, lparam );
    return TRUE;
  }
  else if (msg == WM_COMMAND)
  {
    WORD id = (WORD)LOWORD(wparam);
    #if (CLIENT_OS == OS_WIN32)
    HWND hwnd = (HWND)lparam;
    WORD cmd  = (WORD)HIWORD(wparam);
    #else
    HWND hwnd = (HWND)LOWORD(lparam);
    WORD cmd  = (WORD)HIWORD(lparam);
    #endif
    char fullpath[MAX_PATH + 20];
    switch( id )
    {
      case 1: //IDOK:
      {
        if (!running_child)
        {
          hwnd = GetDlgItem( dialog, 406 /*SSCONFIG_FN */ );
          if (hwnd)
          {
            if (GetWindowText( hwnd, fullpath, sizeof(fullpath) ) != 0)
              SSWriteProfileString("launch", fullpath);
          }
          if (sslistWnd)
          {
            int sslisttype;
            _DestroySSList(sslistWnd, &sslisttype, fullpath);
            SSWriteProfileInt( "type", sslisttype );
            SSWriteProfileString( "file", fullpath );
          }
          EndDialog( dialog, TRUE );
        }
        return TRUE;
      }
      case 2: //IDCANCEL:
      {
        if (!running_child)
        {
          if (sslistWnd)
            _DestroySSList(sslistWnd, NULL, NULL);
          EndDialog( dialog, FALSE );
        }
        return TRUE;
      }
      case 403: /* SSCONFIG_FNMRU */
      {
        break;
      }
      case 405: /* SSCONFIG_FNCONFIG */
      {
        if (running_child)
          return TRUE;
        hwnd = GetDlgItem( dialog, 406 /*SSCONFIG_FN */ );
        if (hwnd)
        {
          if (GetWindowText( hwnd, fullpath, sizeof(fullpath) ) == 0)
            fullpath[0] = '\0';
          if (fullpath[0])
          {
            char origpath[MAX_PATH+2];
            hwnd = GetActiveWindow();
            SSGetProfileString("launch","",origpath,sizeof(origpath));
            EnableWindow( dialog, 0 );
            SSLaunchProcess( fullpath, "-config", +1, 0 ); /* soft block */
            EnableWindow( dialog, 1 ); 
            SSWriteProfileString("launch",origpath);
            SetWindowPos( dialog, HWND_TOP, 0,0,0,0, SWP_SHOWWINDOW|SWP_NOMOVE|SWP_NOSIZE);
            SetActiveWindow(dialog);
            SetFocus( dialog );
            SetActiveWindow(hwnd);
            BringWindowToTop(dialog);
          }
        }
        break;
      }
      case 404: /* SSCONFIG_FNBROWSE */
      {
        if (running_child)
          return TRUE;
        cmd = 0;
        hwnd = GetDlgItem( dialog, 406 /*SSCONFIG_FN */ );
        if (hwnd)
        {
          char dirbuf[sizeof(fullpath)];
          char *p1, *p2;
          OPENFILENAME ofn;
          memset((void *)&ofn,0,sizeof(ofn));
          ofn.lStructSize = sizeof(ofn);
          ofn.hwndOwner = dialog;
          //ofn.hInstance = (HINSTANCE)GetWindowLong(dialog,GWL_HINSTANCE);
          ofn.lpstrFilter = 
                        "distributed.net clients\0dnetc.exe;rc5des*.exe;\0"
                        "All Executable Files\0*.EXE;*.COM\0\0";
          if (szAlternateAppName[0])
            ofn.lpstrFilter = "All Executable Files\0*.EXE;*.COM\0\0";
          if (GetWindowText( hwnd, fullpath, sizeof(fullpath) ) == 0)
            fullpath[0] = '\0';
          ofn.lpstrFile = fullpath;
          ofn.nMaxFile = sizeof(fullpath);
          strcpy( dirbuf, fullpath );
          p1 = strrchr(dirbuf,'\\');
          p2 = strrchr(dirbuf,'/');
          if (p2 > p1) p1 = p2;
          p2 = strrchr(dirbuf,':');
          if (p2 > p1) p1 = p2;
          if (!p1)
            p1 = dirbuf;
          else
          {
            strcpy(fullpath, p1+1 );
            if (*p1 == ':')
              p1++;
          }
          *p1 = '\0';
          ofn.lpstrInitialDir = dirbuf;
          ofn.lpstrTitle = "Browse"; //"Find the distributed.net client";
          ofn.Flags = /*OFN_FILEMUSTEXIST|OFN_PATHMUSTEXIST|*/
                      OFN_NOTESTFILECREATE|OFN_SHAREAWARE|OFN_HIDEREADONLY;
          if (__GetOpenFileName(&ofn))
          {
            SetWindowText(hwnd, ofn.lpstrFile );
            cmd = EN_CHANGE;
          } 
        }
        //fallthrough
      }
      case 406: /*SSCONFIG_FN */
      {
        if (running_child)
          return TRUE;
        if (cmd == EN_CHANGE)
        {
          static char lastchecked[sizeof(fullpath)] = {0};
          fullpath[0] = '\0';
          if (hwnd != NULL)
          {
            if (GetWindowText( hwnd, fullpath, sizeof(fullpath) ) == 0)
              fullpath[0] = '\0';
          }
          else
          {
            hwnd = GetDlgItem( dialog, 406 /*SSCONFIG_FN */ );
            if (hwnd)
            {
              if (SSGetProfileString("launch","",fullpath,sizeof(fullpath))==0)
                fullpath[0] = '\0';
              SetWindowText( hwnd, fullpath );
            }
          }
          if (!lastchecked[0] || strcmp(lastchecked,fullpath)!=0)
          {
            static int lastfound = -1;
            int found = 0;
            strcpy(lastchecked,fullpath);
            strcpy(dlghelptext, ((szAlternateAppName[0])
              ?("Please specify an application to execute in the background.")
              :("Please provide a path to a distributed.net client")) );
            if (fullpath[0])
            {
              const char *p = SSGetDCTIFileVersion(fullpath,(szAlternateAppName[0]=='\0'),1);
              if (!p)
              {
                strcpy(dlghelptext,
                ((szAlternateAppName[0])?("No version information available")
                :("Path does not point to a distributed.net client for Windows")));
              }
              else 
              {
                SSGetFileData(fullpath, 0, &found /* isgui */, 0, 0);
                if (!found)
                {
                  strcpy(dlghelptext,"Path does not point to a "
                                     "distributed.net GUI client");
                }
                else
                {
                  found = 1;
                  strncpy( dlghelptext, p, sizeof(dlghelptext));
                  dlghelptext[sizeof(dlghelptext)-1]='\0';
                }
              }
            }
            if (dlghlpWnd)
              SetWindowText( dlghlpWnd, dlghelptext );
            if (lastfound != found)
            {
              hwnd = GetDlgItem( dialog, 405 /* SSCONFIG_FNCONFIG */ );
              if (hwnd)
                EnableWindow( hwnd, (BOOL)found );
              hwnd = GetDlgItem( dialog, 1 /* IDOK */ );
              if (hwnd)
                EnableWindow( hwnd, (BOOL)found );
              lastfound = found;
            }
          }
        }
        break;
      }
      /* *************************************************************** */
      case 407: /* SSCONFIG_SSPREVIEW */
      {
        if (sslistWnd)
        {
          char lastname[sizeof(fullpath)];
          int newtype, lasttype; 

          EnableWindow( dialog, 0 ); //ShowWindow(dialog,SW_HIDE);

          lasttype = SSGetProfileInt("type",0);
          SSGetProfileString("file","",lastname,sizeof(lastname));
          _MapSSList(sslistWnd, &newtype, fullpath );
           SSWriteProfileInt( "type", newtype );
          SSWriteProfileString( "file", fullpath );
          SSDoSaver( (HINSTANCE)GetWindowLong(dialog,GWL_HINSTANCE), 
                          GetDesktopWindow() );
          SSWriteProfileInt( "type", lasttype );
          SSWriteProfileString( "file", lastname );

          EnableWindow( dialog, 1 ); //ShowWindow(dialog,SW_SHOW);
          SetWindowPos( dialog, HWND_TOP, 0,0,0,0, SWP_SHOWWINDOW|SWP_NOMOVE|SWP_NOSIZE);
          SetActiveWindow(dialog);
          SetFocus( dialog );
          SetActiveWindow(hwnd);
          SetFocus(sslistWnd);
        }
        break;
      }
      case 408: /* SSCONFIG_SSLIST */
      {
        if (cmd == CBN_SELCHANGE && sslistWnd)
        {
          hwnd = GetDlgItem( dialog, 409 /*SSCONFIG_LISTCONFIG */ );
          if (hwnd)
          {
            int sslisttype;
            _MapSSList(sslistWnd, &sslisttype, fullpath );
            EnableWindow( hwnd, (sslisttype > 0));
          }
        }
        break;
      }
      case 409: /* SSCONFIG_LISTCONFIG */
      {
        if (sslistWnd)
        {
          int sslisttype;
          _MapSSList(sslistWnd, &sslisttype, fullpath );
          if (sslisttype <= 0) /* ack! */
            EnableWindow( hwnd, 0 );
          else
          {
            int filever = SSGetFileData(fullpath,0,0,0,0);
            if (filever>=2400) /* win32 and 4.x exe */
            {
              char cmd[32];
              sprintf(cmd,"/c:%ld",dialog);
              SSLaunchProcess( fullpath, cmd, +1, +1 ); /* soft block + higher prio */
            }
            else if (filever >= 0)
            {
              EnableWindow( dialog, 0 ); 
              SSLaunchProcess( fullpath, "/c", +1, +1 ); /* soft block + higher prio */
              EnableWindow( dialog, 1 );
            }
            SetFocus( dialog );
          }
        }
      }
      default:
         break;
    } /* switch( id ) */
  } /* if (msg == WM_COMMAND) */
  return FALSE;
}

/* ---------------------------------------------------- */

static int SSConfigure(HINSTANCE hInstance, HWND hwnd)
{
  FARPROC func = MakeProcInstance( (FARPROC)SSConfigDialogProc, hInstance );
  if (func)
  {
    if (!hwnd) 
    {
      #if (CLIENT_OS == OS_WIN32)
      hwnd = GetForegroundWindow(); 
      #else 
      if ((winGetVersion()%2000)>=400)
        hwnd = GetFocus(); //GetDesktopWindow()
      #endif
    }
    DialogBox( hInstance, MAKEINTRESOURCE(2003 /* DLG_SCRNSAVECONFIGURE*/), 
               hwnd, (DLGPROC)func );
    (void)FreeProcInstance( func );
    if (hwnd)
      EnableWindow(hwnd,1);
  }
  return 0;
}  

/* ---------------------------------------------------- */

static void SSAssertConfiguration(HINSTANCE hInst)
{
  char pathbuffer[MAX_PATH+1];
  szAlternateAppName[0] = '\0';

  if (SSGetProfileString("launch","",pathbuffer,sizeof(pathbuffer))==0)
  {
    if (GetDCTIProfileString(0, "MRUClient", "", pathbuffer, 
        sizeof(pathbuffer))!=0)
    {
      SSWriteProfileString("launch",pathbuffer);
    }
  }

  if (GetModuleFileName( hInst, pathbuffer, sizeof(pathbuffer)))
  {
    if (SSGetFileVersionInfo( pathbuffer, 
                              szAlternateAppName, sizeof(szAlternateAppName),
                              pathbuffer, sizeof(pathbuffer),
                              NULL, 0 ))
    {
      if (strstr(szAlternateAppName,"RC5DES") || 
          strstr(szAlternateAppName,"distributed.net"))
        szAlternateAppName[0] = '\0';
      else
      {
        szAppName = (const char *)&szAlternateAppName[0];
        szSSIniSect = (const char *)&szAlternateAppName[0];
        SetDCTIProfileContext(pathbuffer);
      }
    }
  }
  return;
}

/* ---------------------------------------------------- */

int PASCAL SSMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR lpszCmdLine, int nCmdShow)
{
  szAlternateAppName[0] = '\0';
  SSAssertConfiguration( hInst );

  nCmdShow = nCmdShow; /* shaddup compiler */
  if (!hPrevInst && winGetVersion()<400)
  {
    #if (CLIENT_OS == OS_WIN32)
    {
      MessageBox(NULL, "This screen saver cannot be used with win32s\n"
                       "Use the native win16 version instead.",
        szAppName, MB_OK|MB_ICONSTOP);
      return 0;
    }
    #else
    if (((GetVersion()>>24)%10) == 0) /* OS/2 */
    {
      MessageBox(NULL, "This screen saver cannot be used on OS/2",
        szAppName, MB_OK|MB_ICONSTOP);
      return 0;
    }
    #endif
  }

  {
    {
      HWND hwndParent = (HWND)0;
      char mode = 0;

      if (lpszCmdLine)
      {
        while (*lpszCmdLine == ' ' || *lpszCmdLine == '\t' || 
               *lpszCmdLine == '\"' || *lpszCmdLine == '\'' || 
               *lpszCmdLine == '/' || *lpszCmdLine == '-')
          lpszCmdLine++;
        if (*lpszCmdLine =='s' || *lpszCmdLine == 'S') //don't need hwnd
        {
          mode = 's';
          if (SSGetPreviewWindow() != NULL)
            hwndParent = GetDesktopWindow();
        }
        else //need hwnd
        {
          mode = *lpszCmdLine++;
          if (mode=='p' || mode=='P' || mode=='l' || mode=='L')  
            mode = 'p';
          else if (mode=='c' || mode=='C')
            mode = 'c';
          else if (mode=='a' || mode=='A')
            mode = 'a';
          else 
            mode = 0;
          if (mode)
          {
            while (*lpszCmdLine==' ' || *lpszCmdLine=='\t' || *lpszCmdLine==':')
              lpszCmdLine++; 
            hwndParent = (HWND)atoi(lpszCmdLine);
          }
        }
      }      
      if (mode == 0 || mode == 'c') //configure
      {
        if (!hPrevInst)
        {
          #if (CLIENT_OS == OS_WIN32)
          HANDLE hmutex;
          SECURITY_ATTRIBUTES sa;
          memset(&sa,0,sizeof(sa));
          sa.nLength = sizeof(sa);
          SetLastError(0);
          hmutex = CreateMutex( &sa, TRUE, "distributed.net ScreenSaver");
          if (GetLastError() == 0)  /* ie, does not exist */
          #endif
          SSConfigure(hInst, hwndParent);
          #if 0 //(CLIENT_OS == OS_WIN32)
          ReleaseMutex( hmutex );
          CloseHandle( hmutex );
          #endif
        }
      }
      else if (mode == 'p' || mode == 's') // preview or save
        SSDoSaver(hInst, hwndParent );
      else if (mode == 'a') //change auth
        SSChangePassword(hInst, hwndParent);
    }  
  }
  return 0;
}  
    
#if defined(SSSTANDALONE)
  int PASCAL WinMain(HINSTANCE hInst, HINSTANCE hPrev, LPSTR lpszCmdLine, int nCmdShow)
  { return SSMain(hInst, hPrev, lpszCmdLine, nCmdShow); }
#elif defined(__WATCOMC__)
  static void _installhandler(void) { __SSMAIN = SSMain; }
  #pragma pack(1)
  struct ib_data { char resfield; char level; void (*proc)(void); };
  #pragma pack()
  #pragma data_seg ( "XIB" );
  #pragma data_seg ( "XI" );
  struct ib_data _ssboot_ = { 0, 255, _installhandler };
  #pragma data_seg ( "XIE" );
  #pragma data_seg ( "_DATA" );
#elif defined(__cplusplus)
  static class ssboot {public: ssboot(){__SSMAIN = SSMain;} } _ssboot_;
#else
  #error foo  
#endif

