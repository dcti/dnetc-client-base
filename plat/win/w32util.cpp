/* 
 * This module contains general utility stuff including some functions
 * that are called from the installer.
 *
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 * 
*/
const char *w32util_cpp(void) {
return "@(#)$Id: w32util.cpp,v 1.1.2.1 2001/01/21 15:10:26 cyp Exp $"; }

#define WIN32_LEAN_AND_MEAN /* for win32 */
#define INCLUDE_SHELLAPI_H /* __WINDOWS386__ */
#include <windows.h>
#include <shellapi.h>
#if !defined(_WINNT_)
#include <string.h>
#include <stdlib.h>
#endif

/* ScreenSaver boot vector (initialized from w32ss.cpp if linked) */
int (PASCAL *__SSMAIN)(HINSTANCE,HINSTANCE,LPSTR,int) = NULL;

/* ---------------------------------------------------- */

/* get DOS style version: ((IsNT)?(2000):(0)) + (major*100) + minor */

long winGetVersion(void)
{
  static long ver = 0;

  if (ver == 0)
  {
    unsigned int versionmajor = 0, versionminor = 0;
    #if defined(_WINNT_)
    {
      OSVERSIONINFO osver;

      osver.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
      GetVersionEx(&osver);
      versionmajor = (int)osver.dwMajorVersion;
      versionminor = (int)osver.dwMinorVersion;
      if (VER_PLATFORM_WIN32_NT == osver.dwPlatformId)
        versionmajor += 20;
      else if (VER_PLATFORM_WIN32s == osver.dwPlatformId)
      {   /* version number is unusable (1.xx) */
        DWORD dwVersion = GetVersion();
        versionmajor = LOBYTE(LOWORD(dwVersion));
        versionminor = HIBYTE(LOWORD(dwVersion));  
      }
    }
    #else
    {
      DWORD dwVersion = GetVersion();
      versionmajor = LOBYTE(LOWORD(dwVersion));
      versionminor = HIBYTE(LOWORD(dwVersion));  
      //build number in (HIWORD(dwVersion) & 0x7FFF) in all but plain win3x

      /* heck, even the DOS client has better version detection than this */
 
      if ((dwVersion & 0x80000000)==0) //winNT or win3x without win32s
      {
        const char *p = (const char *)getenv("OS"); 
        // NT 3.x returns 3.x, NT 4.x returns 4.x, but win2k returns 3.95, 
        // which not only sounds wierd, but is wrong according to 
        // http://msdn.microsoft.com/library/psdk/sysmgmt/sysinfo_41bi.htm
        if (versionmajor == 3 && versionminor == 95) /* win2k */
        {
          versionmajor = 5; 
          versionminor = 0;
        }
        if (p && strcmp(p,"Windows_NT")==0)
          versionmajor += 20;
        //if ( GetModuleHandle("KERNEL32") != NULL )
        //  versionmajor += 20;
      }
      else if (versionmajor == 3 && versionminor >= 95) /* win9x */
      {
        versionmajor = 4;
        versionminor = 1;
      }
    }
    #endif

    ver = (((long)(versionmajor))*100)+versionminor;
  }
  return ver;
}

/* ---------------------------------------------------- */


