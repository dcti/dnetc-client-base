/* 
 * portable (between win16/32) interface to profile/configuration as
 * used for windows-client specific things, such as location of 
 * executable, window position etc. 
 * Used by both client and screen-saver shim.
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * The four funcs follow the same format at [Get|Write]ProfileString().
 * For win32 (registry) HKLM\Software\D C T I\sect.key = value
 * For win16 (win.ini)               [D C T I]sect.key = value
 * ('D C T I' expands to "Distributed Computing Technologies, Inc.")
 * 'sect' is optional. If NULL or "", the format becomes key=value.
 *
 * $Id: w32ini.cpp,v 1.1.2.1 2001/01/21 15:10:25 cyp Exp $
*/

#define WIN32_LEAN_AND_MEAN /* for win32 */
#define INCLUDE_SHELLAPI_H  /* for win386 (include <shellapi.h> directly!) */
#include <windows.h>  // include shellapi for win16 here */
#include <shellapi.h> // now include shellapi.h for win32 
#include <string.h>   /* needed for win386 */
#include "w32util.h"  /* winGetVersion() */
#include "w32ini.h"   /* ourselves */

#ifndef MAX_PATH
#define MAX_PATH 256
#endif

static char DCTI_KEYCTX[80] = "Distributed Computing Technologies, Inc.\0";

/* ---------------------------------------------------- */

const char *SetDCTIProfileContext(const char *ctx)
{
  unsigned int i=0;
  if (!ctx || !*ctx)
    ctx = "Distributed Computing Technologies, Inc.";
  for (;*ctx && i<(sizeof(DCTI_KEYCTX)-1);i++)
    DCTI_KEYCTX[i] = *ctx++;
  DCTI_KEYCTX[i] = '\0';
  return DCTI_KEYCTX;  
}    

/* ---------------------------------------------------- */

#if defined(__WINDOWS_386__)
/* Note that although not all functions are referenced to from this
   module, they are referenced elsewhere.
*/
static FARPROC __get_reg_func(const char *symname)
{
  HMODULE hMod = GetModuleHandle("shell.dll");
  if (hMod) return GetProcAddress(hMod, symname);
  return (FARPROC)0;
}
LONG PASCAL /*!FAR*/ RegOpenKey(HKEY hKey, LPCSTR lpzName, HKEY FAR* hRes)
{
  LONG res = ERROR_CANTOPEN;
  FARPROC p = __get_reg_func("RegOpenKey");
  if (p) res = _Call16(p,"dpp", hKey, lpzName, hRes );
  return res;
}
LONG PASCAL /*!FAR*/ RegCreateKey(HKEY hKey, LPCSTR lpzName, HKEY FAR* hRes)
{
  LONG res = ERROR_CANTOPEN;
  FARPROC p = __get_reg_func("RegCreateKey");
  if (p) res = _Call16(p,"dpp", hKey, lpzName, hRes );
  return res;
}
LONG PASCAL /*!FAR*/ RegQueryValue( HKEY hKey, LPCSTR lpSubKey, 
                           LPSTR lpszValue, LONG FAR *lpcb)
{
  LONG res = ERROR_CANTWRITE;
  FARPROC p = __get_reg_func("RegQueryValue");
  if (p) res = _Call16(p,"dppp", hKey, lpSubKey, lpszValue, lpcb );
  return res;
}
LONG PASCAL /*!FAR*/ RegSetValue( HKEY hKey, LPCSTR lpSubKey, DWORD dwType,
                         LPCSTR lpData, DWORD cbData )
{
  LONG res = ERROR_CANTWRITE;
  FARPROC p = __get_reg_func("RegSetValue");
  if (p) res = _Call16(p,"dpdpd", hKey, lpSubKey, dwType, lpData, cbData );
  return res;
}
LONG PASCAL /*!FAR*/ RegCloseKey( HKEY hKey )
{
  LONG res = ERROR_CANTWRITE;
  FARPROC p = __get_reg_func("RegCloseKey");
  if (p) res = _Call16(p,"d", hKey );
  return res;
}
HINSTANCE PASCAL ShellExecute(HWND hwnd, LPCSTR lpszOp, LPCSTR lpFile,
                              LPCSTR lpParams, LPCSTR lpDir, int nShowCmd )
{
  HINSTANCE hInst = (HINSTANCE)0;
  FARPROC p = __get_reg_func("ShellExecute");  
  if (p) hInst = (HINSTANCE)_Call16(p, "wppppw", hwnd, lpszOp, lpFile, 
                                    lpParams, lpDir, nShowCmd);
  return hInst;
}
#endif

static void __do_winnt_thing(void) /* let win16 client use the registry too */
{                                /* doesn't work - always ERROR_ACCESSDENIED */
  #if !defined(_WINNT_)
  static int dunit = 0;
  if (!dunit)
  {
    dunit = -1;
    if (winGetVersion() >= 2000) /* win16 client running on NT */
    {
      char scratch[MAX_PATH + 2];
      HKEY hInstKey, hBaseKey = (( HKEY ) 0x80000002 );/*HKEY_LOCAL_MACHINE;*/
      lstrcat( lstrcpy( scratch, "\\Software\\Microsoft\\Windows NT\\"
               "CurrentVersion\\IniFileMapping\\win.ini" ), DCTI_KEYCTX );
      if (ERROR_SUCCESS == RegCreateKey( hBaseKey, scratch, &hInstKey ))
      {
        lstrcat( lstrcpy( scratch, "#SYS:" ), DCTI_KEYCTX );
        if (ERROR_SUCCESS == RegSetValue( hInstKey, NULL, 
                             REG_SZ, scratch, lstrlen(scratch)))
          dunit = 1;
        RegCloseKey( hInstKey );
      }
    }
  }
  #endif
  return;
}

/* ---------------------------------------------------- */

int WriteDCTIProfileString(const char *sect, const char *entry, const char *val)
{
  int rc = 0;  
  char fullentryname[MAX_PATH + 2];
  if (entry && sect && *sect) /* !delete section */
    entry = lstrcat( lstrcat( lstrcpy( fullentryname, sect ), "." ), entry );
  sect = DCTI_KEYCTX;

  #if defined(_WINNT_)
  if (winGetVersion() >= 400) /* not win32s */
  {
    LONG ec;
    HKEY hInstKey, hBaseKey = (( HKEY ) 0x80000002 );/*HKEY_LOCAL_MACHINE;*/
    char scratch[MAX_PATH + 2];
  
    lstrcat(lstrcpy(scratch,"Software\\"),sect);
    if (entry != NULL && val != NULL) /* create/update key=value */
    {
      rc = 0;
      if (RegCreateKey(hBaseKey, scratch, &hInstKey) == ERROR_SUCCESS)
      {
        unsigned int len = lstrlen(val);
        if (RegSetValueEx(hInstKey,entry,0,REG_SZ,(BYTE *)val,len)==ERROR_SUCCESS)
          rc = 1;
        RegCloseKey(hInstKey);
      }
    }
    else if (entry != NULL) /* val == NUL so delete key=value */
    {
      rc = 0;
      if (RegOpenKey(hBaseKey, scratch, &hInstKey) == ERROR_SUCCESS)
      {
        if (RegDeleteKey( hInstKey, entry ) == ERROR_SUCCESS)
          rc = 1;
        RegCloseKey( hInstKey );
      }
    }
    else /* both key and entry are null so delete entire section */
    {
      while (RegOpenKey(hBaseKey, scratch, &hInstKey) == ERROR_SUCCESS)
      {
        FILETIME ft;
        DWORD gsz = sizeof(fullentryname);
        ec = RegEnumKeyEx( hInstKey, 0, fullentryname, &gsz, NULL,
                        NULL, NULL, &ft );
        if (ec == ERROR_SUCCESS)
          ec = RegDeleteKey(hInstKey, fullentryname );
        RegCloseKey( hInstKey );
        if (ec != ERROR_SUCCESS)
          break;
      }
      rc = 0;
      if (RegOpenKey(hBaseKey, "Software", &hInstKey) == ERROR_SUCCESS)
      {
        if (RegDeleteKey( hInstKey, sect ) == ERROR_SUCCESS)
          rc = 1;
        RegCloseKey( hInstKey );
      }
    }
  }
  else
  #endif
  {
    __do_winnt_thing();  /* let win16 client use the registry too */
    if ((rc = WriteProfileString(sect,entry,val)) == 0)
      rc = WriteProfileString(sect,entry,val);
  }
  return rc;
}  

/* ---------------------------------------------------- */

unsigned int GetDCTIProfileString(const char *sect, const char *entry, 
                        const char *def, char *buf, unsigned int bufsize )
{
  char fullentryname[MAX_PATH + 2];
  if (!buf || !bufsize || !entry)
    return 0;
  buf[0] = 0;
  if (bufsize < 2)
    return 0;
  if (sect && *sect)
    entry = lstrcat( lstrcat( lstrcpy( fullentryname, sect ), "." ), entry );
  sect = DCTI_KEYCTX;

  #if defined(_WINNT_)
  if (winGetVersion() >= 400) /* not win32s */
  {
    int usedef = 1;
    HKEY hInstKey, hBaseKey = (( HKEY ) 0x80000002 );/* HKEY_LOCAL_MACHINE; */
    char scratch[MAX_PATH + 2];
    if (RegOpenKey( hBaseKey, lstrcat(lstrcpy(scratch,"Software\\"),sect), &hInstKey) == ERROR_SUCCESS)
    {
      DWORD typ = REG_SZ, size = bufsize;
      if (RegQueryValueEx(hInstKey,entry,NULL,&typ,(BYTE *)buf,&size)
            == ERROR_SUCCESS) 
      {
        bufsize = (unsigned int)size;
        usedef = 0;
      }
      RegCloseKey(hInstKey);
    }
    if (usedef)
    {
      if (!def || !*def)
        bufsize = 0;
      else
      {
        lstrcpyn( buf, def, bufsize );
        buf[bufsize-1] = '\0';
        bufsize = lstrlen( buf );
      }
    }
  }
  else
  #endif
  {
    __do_winnt_thing();  /* let win16 client use the registry too */
    bufsize = GetProfileString(sect, entry, def, buf, (short)bufsize) & 0xffff;
  }
  return bufsize;
}

/* ---------------------------------------------------- */

int GetDCTIProfileInt(const char *sect, const char *entry, int defval )
{
  char s[sizeof(long)*3]; int x;
  if ((x = (int)GetDCTIProfileString(sect, entry, "", s, sizeof(s))) == 0)
    return defval;
  else if (x==2 && (s[0]|' ')=='o' && (s[1]|' ')=='n')
    return 1;
  else if (x==3 && (s[0]|' ')=='y' && (s[1]|' ')=='e' && (s[2]|' ')=='s')
    return 1;
  else if (x==4 && (s[0]|' ')=='t' && (s[1]|' ')=='r' && (s[2]|' ')=='u' && (s[3]|' ')=='e')
    return 1;
  x = defval = 0; if (s[0] == '-' || s[0]=='+') x++;
  while(s[x]>='0' && s[x]<='9') defval = (defval*10)+(s[x++]-'0');
  return ((s[0]=='-') ? -defval : defval);
}

/* ---------------------------------------------------- */

int WriteDCTIProfileInt(const char *sect, const char *entry, int val )
{
  char str[sizeof(long)*3];
  unsigned int x = (unsigned int) (val<0)?(-val):(val);
  int pos = sizeof(str);
  str[--pos]='\0';
  do {
    str[--pos] = (char) ('0' + (char)(x % 10));
    x /= 10;
  } while (x);
  if (val < 0) str[--pos] = '-';
  return WriteDCTIProfileString(sect, entry, &str[pos] );
}

/* ---------------------------------------------------- */

