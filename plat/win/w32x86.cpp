/*  
 *  ----------------------------------------------------------------
 *  This module is for x86 executables only.
 *
 *  GetTickCount() (the only function currently in here) is for
 *  executables _tagged_with_a_version_number_less_than_3.5_. It provides
 *  real millisec resolution regardless of which client on which Windows.
 *
 *  Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *  ----------------------------------------------------------------
*/
const char *w32x86_cpp(void) {
return "@(#)$Id: w32x86.cpp,v 1.1.2.1 2001/01/21 15:10:26 cyp Exp $"; }

#define WIN32_LEAN_AND_MEAN /* for win32 */
#define INCLUDE_TOOLHELP_H /* win16 and __WINDOWS386__ */
#include <windows.h>
#include "w32util.h"   /* winGetVersion() */

/* -------------------------------------------------------------------- */

#if defined(__WINDOWS_386__)
  #undef WINAPI
  #define WINAPI PASCAL
  static HINDIR __IndirTimerCount = (HINDIR)0;
  static DWORD __timerinfo_alias16 = 0;
  static HINSTANCE __toolhelp = 0;
  static void _deinstallhandler(void)
  {
    if (__IndirTimerCount)
      FreeIndirectFunctionHandle( __IndirTimerCount );
    __IndirTimerCount = 0;
    if (__timerinfo_alias16)
      FreeAlias16(__timerinfo_alias16);
    __timerinfo_alias16 = 0;
    if (__toolhelp)
      FreeLibrary( __toolhelp );
    __toolhelp = 0;
  }  
  #pragma pack(1)
  struct ib_data { char resfield; char level; void (*proc)(void); };
  #pragma pack()
  #pragma data_seg ( "YIB" );
  #pragma data_seg ( "YI" );
  struct ib_data __toolhelp_fini = { 0, 0, _deinstallhandler };
  #pragma data_seg ( "YIE" );
  #pragma data_seg ( "_DATA" );
#endif

/* -------------------------------------------------------------------- */

/* we need this for scheduling when running on win16/win32s */
DWORD WINAPI GetTickCount (VOID) /* gives us real millisec resolution */
{                                /* regardless of which client on which os */
  #if defined(_WINNT_)
  typedef DWORD (WINAPI *_GETCURRENTTIME)(void);
  static _GETCURRENTTIME __GetCurrentTime = (_GETCURRENTTIME)0;
  #undef GetCurrentTime
  #define GetCurrentTime __GetCurrentTime
  if (!__GetCurrentTime)
  {
    HINSTANCE hInst = GetModuleHandle("kernel32");
    if (hInst)
      __GetCurrentTime = (_GETCURRENTTIME)GetProcAddress(hInst,"GetTickCount");
    if (!__GetCurrentTime)
    {
      static int checked = -1;
      if (checked == -1)
      {
        checked = 0;
        MessageBox(NULL,"Unable to get high resolution time",
                        "Fatal error",MB_OK|MB_ICONHAND);
        ExitProcess(1);
      }                        
      return 0;
    }
  }
  #endif

  if (winGetVersion()<400)
  {
    #if defined(__WINDOWS_386__)
    static TIMERINFO timerinfo;
    static int initialized = -1;
    if (initialized == -1)
    {
      UINT olderrmode = SetErrorMode(SEM_NOOPENFILEERRORBOX);
      initialized = 0;
      __IndirTimerCount = 0;
      __timerinfo_alias16 = 0;
      __toolhelp = LoadLibrary( "toolhelp.dll" );
      SetErrorMode(olderrmode);
      if (__toolhelp <= ((HINSTANCE)(32)))
        __toolhelp = 0;
      else
      { 
        FARPROC proc = GetProcAddress( __toolhelp, "TimerCount" ); 
        if (proc)
        {
          __IndirTimerCount = GetIndirectFunctionHandle(proc,INDIR_DWORD,INDIR_ENDLIST);
          if (__IndirTimerCount)
          {
            __timerinfo_alias16 = AllocAlias16( (void *)&timerinfo );
            if (!__timerinfo_alias16)
            {
              FreeIndirectFunctionHandle( __IndirTimerCount );
              __IndirTimerCount = 0;
            }
          }
        }
        if (!__IndirTimerCount)
        {
          FreeLibrary( __toolhelp );
          __toolhelp = 0;
        }
      }  
    }
    if (__timerinfo_alias16)
    {
      timerinfo.dwSize = sizeof(TIMERINFO);
      if ((InvokeIndirectFunction(__IndirTimerCount,__timerinfo_alias16) & 0xffff)!=0)
        return timerinfo.dwmsThisVM;
    }
    #elif !defined(_WINNT_) /* normal win16 */
    if ((GetWinFlags() & WF_STANDARD)==0)
    {
      DWORD __VTDVector = 0;
      WORD vseg,voff;
      _asm push es
      _asm push di
      _asm push bx
      _asm mov ax, 1684h /* get device api entry point */
      _asm mov bx, 5     /* vtd.386 */
      _asm int 2Fh
      _asm mov vseg,es
      _asm mov voff,di
      _asm pop bx
      _asm pop di
      _asm pop es
      __VTDVector = (((DWORD)vseg)<<16)+voff;
      if (__VTDVector) /* have VTD.386 */
      {
        DWORD vmtime;
        char err = 0;
        _asm mov ax,0x102
        _asm call dword ptr [__VTDVector]
        _asm sbb cl,cl
        _asm mov err, cl
        _asm mov vmtime, eax
        if (!err)
          return vmtime;
      }
    }
    #endif /* defined(__WINDOWS_386__) or normal win16 */

    /* probably win32s, but could be win386 without toolhelp.dll */
    {
      static DWORD lastvmtime = 0;
      DWORD nowvmtime = 0;
      DWORD nexttick = 0;
      WORD pit = 0;
      do
      {
        nowvmtime = nexttick;
        _asm xor   al, al
        _asm out   43h, al
        _asm in    al, 40h
        _asm mov   ah, al
        _asm in    al, 40h
        _asm xchg  ah, al
        _asm not   ax
        _asm mov   pit,ax
        nexttick = GetCurrentTime();
        if (nowvmtime == 0)
          nowvmtime = nexttick - 1;
      } while (nexttick != nowvmtime);
      nowvmtime += (((DWORD)pit)/1193);
      if (nowvmtime > lastvmtime)
        lastvmtime = nowvmtime;
      return lastvmtime;
    }

  } /* if (winGetVersion()<400) */
  return GetCurrentTime();
}      

/* -------------------------------------------------------------------- */

