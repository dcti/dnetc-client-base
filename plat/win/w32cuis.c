/*
 * Console <-> GUI std handle redirector for use with distributed.net projects.
 * Comes in two flavors: 
 *   1) HAVE_EXEC_AS_TEMPFILE (davehart's solution)
 *      - dups .exe to %temp%, patches header, exec's, waits for exit, del temp
 *   2) HAVE_EXEC_WITH_PIPES  (cyp's solution)
 *      - uses anonymous pipes
 * 
 * The pipe-driven solution supports common ansi sequences, and also provides
 * a rudimentary means for supporting windows-specific stuff like setting the
 * console title via 'reverse ansi' sequences, ie, esc]... instead of esc[...
 * The pipe shim itself is generic and will work without changes with any 
 * backend that SetStdHandle()s the 'advertised' pipe ends.
 *
 * $Id: w32cuis.c,v 1.1.2.2 2001/04/10 00:51:09 cyp Exp $
*/

/* #define HAVE_EXEC_AS_TEMPFILE */  /* davehart's solution */
#define HAVE_EXEC_WITH_PIPES  /* cyp's solution */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

extern int main(void);

/* this next section sets all the options necessary to stub out all 
   the clib/crt0 stuff, none of which is needed since this is a pure 
   winapi executable. Its a size saving of about 20K.
*/
#if defined(__WATCOMC__)

  #pragma intrinsic(memset)             /* ZeroMemory is redef'd to memset */
  #pragma off (check_stack)             /* doesn't work. need wcl /s ... */
  #pragma comment(lib,"kernel32")       /* need kernel32.lib */
  #pragma comment(lib,"user32")         /* need user32.lib as well */
  #pragma code_seg ( "BEGTEXT" );       /* entry point is main */
  int __syscall mainCRTStartup(void) { return main(); }
  #pragma code_seg ( "_TEXT" );        
  int _argc;                            /* compiler always references this */
  int __syscall _cstart_(void) { return main(); } /* ... and this */

#elif defined(_MSC_VER)

  #pragma intrinsic(memset)             /* ZeroMemory is redef'd to memset */
  #pragma check_stack(off)              /* turn off stack checking */
  #pragma comment(lib,"kernel32")       /* need kernel32.lib */
  #pragma comment(lib,"user32")         /* need user32.lib as well */
  int mainCRTStartup(void) {return main();} /* entry point is main */  
  #pragma comment(linker, "/subsystem:console")
  #pragma comment(linker, "/entry:main") /* entry point is main */

#elif defined(__BORLANDC__)

  /* mainCRTStartup() [or whatever] must be the first function in the file */
  int mainCRTStartup(void) {return main();} /* entry point is main */  

  void *memset(void *s,int ch,unsigned int len) 
  {                             /* grrr! ZeroMemory() is redef'd! */
    char *p = (char *)s;
    char *e = p+len;
    while (p<e)
      *p++ = (char)ch;
    return s;
  } 

#else

  int mainCRTStartup(void) {return main();} /* entry point is main */  

#endif

/* -------------------------------------------------------------------- */

static void ErrOutLen(const char *msg, DWORD len)
{
  WriteConsole(GetStdHandle(STD_ERROR_HANDLE),(CONST VOID *)msg,len,&len,NULL);
}
static void ErrOut(const char *msg) { ErrOutLen(msg, lstrlen(msg)); }
static void errMessage(const char *msg, DWORD lastError /* GetLastError() */)
{
  /* eg: errMessage("This application requires WindowsNT", 0 );
         => "BLAH.COM: This application requires WindowsNT\n"
     or  errMessage("Spawn failed", GetLastError() );
         => "BLAH.COM: Spawn failed: The system cannot open the file.\n"
  */
  int len; char buf[MAX_PATH+1];
  if ((len = ((int)GetModuleFileName(NULL,buf,sizeof(buf))))!=0)
  {
    while (len > 0)
    {
      len--;
      if (buf[len]=='/' || buf[len]=='\\' || buf[len]==':')
      {
        len++;
        break;
      }
    }
    ErrOutLen( &buf[len], CharLowerBuff( &buf[len], lstrlen( &buf[len] )) );
    ErrOutLen( ": ", 2 );
  }
  if (msg)
  {
    if ((len = ((int)lstrlen( msg ))) != 0)
    {
      len--;
      while (len>=0 && (msg[len]==' ' || msg[len]=='\t' || 
                        msg[len]==':' || msg[len]=='\n'))
        len--;
      if (len > 0)
      {
        ErrOutLen( msg, len+1 );
        if (lastError)
          ErrOutLen( ": ", 2 );
      }
    }
  }  
  if (lastError)
  {
    LPVOID lpMsgBuf;
    if (lastError == ((DWORD)-1))
      lastError = GetLastError();
    FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | 
       FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
       lastError,  0 /*MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT)*/, 
     (LPTSTR) &lpMsgBuf, 0, NULL ); /* Process any inserts in lpMsgBuf. */
    ErrOut( (const char *)lpMsgBuf );
    LocalFree( lpMsgBuf );
  }
  ErrOutLen( "\n", 1 );
  return;
}  

/* -------------------------------------------------------------------- */

static char *__constructCmdLine(const char *filename,
                                char *buffer, unsigned int bufsize)
{
  char fnbuffer[MAX_PATH+1];
  if (!filename)
  {
    if (!GetModuleFileName(NULL,fnbuffer,sizeof(fnbuffer)))
      fnbuffer[0] = '\0';
    filename = (const char *)&fnbuffer[0];
  }  
  buffer[0] = '\0';
  if (*filename) /* if not, return "" */
  {
    unsigned int len;
    const char *cmdline = (const char *)&filename[0];
    while (*cmdline && *cmdline!=' ' && *cmdline !='\t')
      cmdline++;
    if (!*cmdline) /* no spaces in filename */
      len = lstrlen( lstrcpy( buffer, filename ) );
    else /* put the filename in quotes */
    {
      buffer[0] = '\"';
      len = 1 + lstrlen( lstrcpy( &buffer[1], filename ) );
      buffer[len++] = '\"';
      buffer[len] = '\0';
    }
    if (len) /* if not, we have a bad filename, return "" */
    {
      cmdline = (const char *)GetCommandLine();
      if (cmdline) /* should always be so */
      {
        if (*cmdline) /* skip the appname */
        {
          char ch1 = ' ', ch2 = '\t';
          if (*cmdline == '\"' || *cmdline == '\'')
            ch1 = ch2 = (char)(*cmdline++);
          while (*cmdline && ((char)*cmdline) != ch1 && ((char)*cmdline) != ch2)
            cmdline++;
          if (ch1 != ' ' && ((char)*cmdline) == ch1)
            cmdline++;
          while (*cmdline == ' ' && *cmdline == '\t')
            cmdline++;
        }
        if (*cmdline) /* tack it on after the filename */
        {
          buffer[len++] = ' ';
          lstrcpyn( &buffer[len], cmdline, (bufsize-len)-1 );
          buffer[bufsize-1] = '\0';
        }
      }    
    }
  }
  return buffer;
}  

/* ---------------------------------------------------------------- */

static HWND __advertised_cmd_hwnd = 0;
static DWORD __dwChildPid = 0;
static BOOL WINAPI TriggerControl(DWORD dwCtrlType)
{
  if (dwCtrlType == CTRL_BREAK_EVENT || /* \_ only break and ^C are  */
      dwCtrlType == CTRL_C_EVENT ||     /* /  callable on win9x      */
      dwCtrlType == CTRL_CLOSE_EVENT || 
      dwCtrlType == CTRL_SHUTDOWN_EVENT || 
      dwCtrlType == CTRL_LOGOFF_EVENT ) 
  {
    if (__dwChildPid)
      GenerateConsoleCtrlEvent(dwCtrlType,__dwChildPid);
    if (__advertised_cmd_hwnd)
    {
      if (!IsWindow(__advertised_cmd_hwnd))
        __advertised_cmd_hwnd = 0;
      else
        SendMessage(__advertised_cmd_hwnd,WM_CLOSE,0,0);
    }
    if (__dwChildPid || __advertised_cmd_hwnd)
    {
      /* max 20secs for CTRL_LOGOFF_EVENT and CTRL_SHUTDOWN_EVENT,
       * max  5secs for CTRL_CLOSE_EVENT, no limit for the others
       * http://support.microsoft.com/support/kb/articles/q130/7/17.asp   
      */
      int sleepTime = 18000; 
      if (dwCtrlType == CTRL_CLOSE_EVENT)
        sleepTime = 4000;
      while (sleepTime > 0 && (__advertised_cmd_hwnd || __dwChildPid)) 
      {
        Sleep(250);
        sleepTime -= 250;
      }
    }
    return TRUE;
  }
  return FALSE; /* DBG_CONTROL_C */
}

/* ---------------------------------------------------------------- */

#if defined(HAVE_EXEC_AS_TEMPFILE)
static BOOL __cleanup_temp_files(const char *pszTempDir,const char *appname)
{
  HANDLE hSearch;
  WIN32_FIND_DATA findData;
  char szfnBuffer[MAX_PATH+1];
  unsigned int pathLen, pos;

  pathLen = 0;
  szfnBuffer[0] = '\0';
  if (*pszTempDir)
  {
    pos = lstrlen(lstrcpy(szfnBuffer,pszTempDir));
    if (szfnBuffer[pos-1]!=':' && szfnBuffer[pos-1]!='\\' && szfnBuffer[pos-1]!='/')
    {
      szfnBuffer[pos++] = '\\';
      szfnBuffer[pos] = '\0';
    }
    pathLen = pos;
  }
  pos = lstrlen(appname);
  while (pos>0 && appname[pos-1]!='/' && 
         appname[pos-1]!='\\' && appname[pos-1]!=':')
    pos--;
  appname += pos;
  pos = pathLen;
  while (*appname && *appname!='.')
    szfnBuffer[pos++] = *appname++;
  lstrcpy( &szfnBuffer[pos], "-*.exe" );
  
  /* handle new style "%temp%\rc5des-xxxxx.exe", "%temp%\dnetc-xxxxx.exe" */
  if ((hSearch = FindFirstFile( szfnBuffer, &findData ))!=INVALID_HANDLE_VALUE)
  {
    do
    {
      lstrcpy( &szfnBuffer[pathLen], findData.cFileName );
       DeleteFile( szfnBuffer );
    } while ( FindNextFile( hSearch, &findData ));
    FindClose( hSearch );
  }
  
  /* handle old style "%temp%\rc5xxxxxx.tmp" too */
  lstrcpy( &szfnBuffer[pathLen], "rc5*.tmp" );
  if ((hSearch = FindFirstFile( szfnBuffer, &findData ))!=INVALID_HANDLE_VALUE)
  {
    do
    {
      lstrcpy( &szfnBuffer[pathLen], findData.cFileName );
       DeleteFile( szfnBuffer );
    } while ( FindNextFile( hSearch, &findData ));
    FindClose( hSearch );
  }
  
  
  return TRUE;
}
#endif /* HAVE_EXEC_AS_TEMPFILE */

/* ---------------------------------------------------------------- */

#if defined(HAVE_EXEC_AS_TEMPFILE)
static DWORD __exec_as_tempfile(const char *filename) /* davehart's solution */
{                                                 /* we are .exe, exec .com */
  /* Implementation is to make a temporary copy of rc5des.exe and
   * modify its PE header (win32 EXE format) to change it from
   * Win32 GUI to Win32 CUI (console).  rc5des.exe itself takes
   * note of the full path to the (real) EXE in its command line
   * so it sees its own EXE name as the real path not the temp
   * copy.
   * Addendum from cyp: I've modified it to [a] use a more descriptive 
   * temp file: uses a combination of the basename of file to launch and
   * the tempfile created and adds an .exe extension. So, instead of 
   * rc5xxxxxxx.tmp, it uses yyyy-xxxxxx.exe (yyyy being 'rc5des' or 'dnetc'
   * or whatever). [b] use winapi functions instead of clib [c] use stack
   * instead of statics [d] restructured and threw out the gotos.
  */
  DWORD dwExitCode = 2; /* what to return on fail */
  char szTempDir[MAX_PATH];

  if (!GetTempPath(sizeof(szTempDir) / sizeof(szTempDir[0]), szTempDir))
    errMessage("Unable to get temp path", ((DWORD)-1));
  else if (__cleanup_temp_files( szTempDir, filename )) /* always true */
  {
    char szTempChildPath[MAX_PATH];
    if (!GetTempFileName(szTempDir, "rc5", 0, szTempChildPath))
      errMessage("Unable to create temp file", ((DWORD)-1));
    else
    {
      if (!CopyFile( filename, szTempChildPath, FALSE)) 
        errMessage("Unable to duplicate executable", ((DWORD)-1));
      else
      {
        HANDLE hfileExe;
        char buffer[1024];

        /* ---------------------
         * niceify the temp filename
         * ---------------------
        */

        {
          int pos, len = lstrlen(lstrcpy( buffer, szTempChildPath ));
          while (len>0 && buffer[len-1]!='\\' && 
               buffer[len-1]!='/' && buffer[len-1]!=':')
            len--;
          pos = lstrlen(filename);
          while (pos>0 && filename[pos-1]!='\\' && 
               filename[pos-1]!='/' && filename[pos-1]!=':')
            pos--;
          lstrcpy( &buffer[len], &filename[pos] );
          pos = len;
          while (buffer[pos] && buffer[pos] != '.')
            pos++;
          len += 3;
          buffer[pos++] = '-';
          while ( szTempChildPath[len] && szTempChildPath[len] != '.' )
            buffer[pos++] = szTempChildPath[len++];
          lstrcpy( &buffer[pos], ".exe" /* .tmp */ );
          CharLowerBuff(buffer, lstrlen(buffer));
          if (MoveFile(szTempChildPath, buffer))
            lstrcpy(szTempChildPath, buffer);
        }

        /* ---------------------
         * patch the header
         * ---------------------
        */
        
        hfileExe = CreateFile(
                       szTempChildPath,
                       GENERIC_READ | GENERIC_WRITE,
                       0,     /* no sharing allowed */
                       NULL,  /* security attributes */
                       OPEN_EXISTING,
                       FILE_ATTRIBUTE_TEMPORARY,
                       NULL   /* hfileTemplate */
                       );
    
        if (INVALID_HANDLE_VALUE == hfileExe) 
          errMessage("Unable to open temporary copy", ((DWORD)-1));
        else
        {
          BOOL bPatchedOk = FALSE;
          HANDLE hmapExe = CreateFileMapping(hfileExe, NULL, PAGE_READWRITE,
                                             0,  0, NULL );
          if (!hmapExe)
            errMessage("Unable to create file mapping", ((DWORD)-1));
          else 
          {
            BYTE *pExe = (BYTE *) MapViewOfFile( hmapExe,
                                  FILE_MAP_READ | FILE_MAP_WRITE, 0,0,0 );
            if (!pExe)
              errMessage("Unable to map view of file", ((DWORD)-1));
            else
            {
              BYTE *pNEPE = pExe;
              if (((PIMAGE_DOS_HEADER)pNEPE)->e_magic == IMAGE_DOS_SIGNATURE)
                pNEPE += ((PIMAGE_DOS_HEADER)pNEPE)->e_lfanew;
              if (((PIMAGE_NT_HEADERS)pNEPE)->Signature != IMAGE_NT_SIGNATURE ||
                  ((PIMAGE_NT_HEADERS)pNEPE)->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR_MAGIC)
                errMessage("hmm. Doesn't look like an executable image.",0);
              else
              {
                ((PIMAGE_NT_HEADERS)pNEPE)->OptionalHeader.Subsystem =
                   IMAGE_SUBSYSTEM_WINDOWS_CUI;
                bPatchedOk = TRUE;
              }
              UnmapViewOfFile(pExe);
            }
            CloseHandle(hmapExe);
          }
          CloseHandle(hfileExe);
          if (!bPatchedOk)
            hfileExe = INVALID_HANDLE_VALUE;
        }
        
        /* -------------------
         * spawn the child
         * -------------------
        */
        
        if (hfileExe != INVALID_HANDLE_VALUE)
        {
          PROCESS_INFORMATION pi;
          STARTUPINFO si;

          ZeroMemory(&si, sizeof(si));
          si.cb = sizeof(si);

          __constructCmdLine(filename, buffer, sizeof(buffer));

          if (! CreateProcess( szTempChildPath, /* ptr to filename to exec */
                               buffer, /* ptr to cmd line string */
                               NULL,   /* process security attributes */
                               NULL,   /* thread security attributes */
                               TRUE,   /* bInherhitHandles */
                               CREATE_DEFAULT_ERROR_MODE, /* creation flags */
                               NULL,  /* pointer to new environment block */
                               NULL, /* pointer to current directory name */
                               &si, /* pointer to STARTUPINFO */
                               &pi )) /* pointer to PROCESS_INFORMATION */
          {
            errMessage("Spawn failed", ((DWORD)-1));
          }
          else
          {
            CloseHandle(pi.hThread);
            __dwChildPid = pi.dwProcessId;
            #if 0 /* succeeds! but does not work */
            if ((hfileExe = CreateFile( szTempChildPath, GENERIC_READ,
                          FILE_SHARE_READ, NULL, OPEN_EXISTING,
                          FILE_ATTRIBUTE_TEMPORARY|FILE_FLAG_DELETE_ON_CLOSE,
                          NULL )) != INVALID_HANDLE_VALUE)
            {                          
              szTempChildPath[0] = '\0';
              CloseHandle( hfileExe );
            }
            else
            #endif
            {
              WaitForSingleObject(pi.hProcess, INFINITE);
              DeleteFile(szTempChildPath);
            }
            __dwChildPid = 0;
            GetExitCodeProcess(pi.hProcess, &dwExitCode);
            CloseHandle(pi.hProcess);
          } /* spawn ok */
        } /* mapping ok */
      } /* copy ok */
      if (szTempChildPath[0])
        DeleteFile(szTempChildPath);
    } /* get temp file name ok */
  } /* get temp path ok */
  return dwExitCode;
}
#endif /* HAVE_EXEC_AS_TEMPFILE */

/* ---------------------------------------------------------------- */

#if defined(HAVE_EXEC_WITH_PIPES)
static int __pushOutput( HANDLE hFile, DWORD *dwFileTypeP, 
                  const char *buffer, unsigned int buflen, BOOL bFlush )
{
  int written = 0;
  if (buflen)
  {
    DWORD bytesSent = 0;
    DWORD dwFileType = ((dwFileTypeP)?(*dwFileTypeP):(GetFileType( hFile )));

    if (dwFileType == FILE_TYPE_CHAR)
    {
      if (!WriteConsole(hFile, (CONST VOID *)buffer, (DWORD)buflen,
                        &bytesSent, NULL))
      {
        dwFileType = FILE_TYPE_UNKNOWN;
        written = -1;
      }
    }
    if (dwFileType != FILE_TYPE_CHAR)
    {
      if (!WriteFile( hFile, (LPCVOID)buffer, (DWORD)buflen, 
                               &bytesSent, NULL ))
        written = -1;
    }  
    if (written != -1)
    {
      written = (int)bytesSent;
      if ( bFlush )
        FlushFileBuffers( hFile );
    }
    if (dwFileTypeP)
      *dwFileTypeP = dwFileType;
  }
  return written;
}                            
#endif /* HAVE_EXEC_WITH_PIPES */

/* ---------------------------------------------------------------- */

#if defined(HAVE_EXEC_WITH_PIPES)
static int __ansigetopts( const char *cmdbuf, int cmdbuflen,
                          int *opt1, int *opt2 )
{
  if (cmdbuflen < 3)
    return -1;
  if (cmdbuf[0] != 0x1B || cmdbuf[1] != '[')
    return -1;
  if (!((cmdbuf[cmdbuflen-1]>='A' && cmdbuf[cmdbuflen-1]<='Z') ||
        (cmdbuf[cmdbuflen-1]>='a' && cmdbuf[cmdbuflen-1]<='z')))
    return -1;
  if (cmdbuflen > 3)
  {
    int pos, val1, val2, have1, have2, opcount;
    val1 = val2 = have1 = have2 = opcount = 0;
    for (pos = 2; pos < cmdbuflen; pos++)
    {
      char c = cmdbuf[pos];
      if (c >= '0' && c <= '9')
      {
        c = (char)(c - '0');
        if (opcount == 0)
        {
          have1++;
          val1 *= 10;
          val1 += c;
        }
        else 
        {
          have2++;
          val2 *= 10;
          val2 += c;
        }
      }  
      else if (c == ';' || pos == (cmdbuflen-1))
      {
        if ((++opcount) > 1) /* opcount == 2 */
        {
          break;
        }
      }
      else
      {
        break;
      }
    }

    if (opt1 && have1)
      *opt1 = val1;
    if (opt2 && have2)
      *opt2 = val2;
  }
  return 0;
}
#endif /* HAVE_EXEC_WITH_PIPES */

/* ---------------------------------------------------------------- */

static HWND __get_console_hwnd(void)
{
  HWND hwnd = NULL;
  char szWinTitle[MAX_PATH]; 
  DWORD dwLen = GetConsoleTitle( szWinTitle, sizeof( szWinTitle ) );
  if (dwLen > 0 && dwLen < (sizeof(szWinTitle)-20))
  {
    wsprintf( &szWinTitle[dwLen],"-%8x%8x",GetTickCount(),GetModuleHandle(NULL));
    if ( SetConsoleTitle( szWinTitle ))
    {
      int tries = 0;
      while ((hwnd = FindWindow( NULL, szWinTitle )) == NULL)
      {
        if ((++tries) > 25)
          break;
        Sleep(40); /* Delay needed for title to update */
      }
      szWinTitle[dwLen] = '\0';
      SetConsoleTitle(szWinTitle);
    }
  }
  return hwnd;
}

/* ---------------------------------------------------------------- */

static void __getset_icons(HWND hwnd, const char *filename, HICON oldicons[2])
{
  if (hwnd && IsWindow(hwnd))
  {
    int i;
    if (!filename) /* restore */
    {
      if (oldicons[0] || oldicons[1])
      {
        for (i=0; i<2; i++)
          SendMessage( hwnd, WM_SETICON, i, (LPARAM)(oldicons[i]) );
      }
    }
    else /* save */
    {
      for (i=0; i<2; i++)
      {
        oldicons[i] = (HICON)SendMessage(hwnd,WM_GETICON,i,0);
      }
      if (oldicons[0] || oldicons[1])
      { 
        int madechange = 0;
        HMODULE hInst = GetModuleHandle( "user32.dll" );
        if (hInst)
        {
          typedef HANDLE (WINAPI *LoadImageAT)(HINSTANCE,LPCTSTR,UINT,int,int,UINT);
          LoadImageAT _LoadImage = (LoadImageAT) GetProcAddress(hInst, "LoadImageA");
          if (_LoadImage)
          {
            hInst = LoadLibraryEx(filename, 0, LOAD_LIBRARY_AS_DATAFILE);
            if (hInst)
            {
              HICON newicons[2];
              newicons[0] = (HICON)(*_LoadImage)(hInst, 
                                             MAKEINTRESOURCE(1), IMAGE_ICON,
                                             GetSystemMetrics(SM_CXSMICON),
                                             GetSystemMetrics(SM_CYSMICON), 0);
              newicons[1] = (HICON)(*_LoadImage)(hInst, 
                                             MAKEINTRESOURCE(1), IMAGE_ICON,
                                             GetSystemMetrics(SM_CXICON),
                                             GetSystemMetrics(SM_CYICON), 0 );
              if (newicons[0] || newicons[1])
              {
                for (i=0;i<2;i++)
                  SendMessage( hwnd, WM_SETICON, i, (LPARAM)(newicons[i]) );
                madechange = 1;
              }
              FreeLibrary( hInst );
            }
          }
        }       
        if (!madechange)
          oldicons[0] = oldicons[1] = NULL;
      }
    }
  }
  return;
}

/* ---------------------------------------------------------------- */

#if defined(HAVE_EXEC_WITH_PIPES)
static DWORD __exec_with_pipes(const char *filename) /* cyp's solution */
{                                                 /* we are .com, exec .exe */
  /*
   * pipe server for use with the distributed.net win client. 
   * Both client and server were written in a 10 hour blaze of creative 
   * genius :) on 25.Oct.99 by Cyrus Patel <cyp@fb14.uni-mainz.de> :)
  */
  DWORD dwExitCode = 2; /* what to return on fail */
  HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
  HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
  HWND hConsoleWnd = __get_console_hwnd();

  if (hStdout == INVALID_HANDLE_VALUE || hStdin == INVALID_HANDLE_VALUE)
  {
    errMessage("Unable to get standard handle(s)", 0 );
  }
  else
  {
    HANDLE binHandle;
    if (INVALID_HANDLE_VALUE == 
         (binHandle = CreateFile( filename, /* pointer to name of the file */
                                  GENERIC_READ, /* access (read-write) mode */
                                  FILE_SHARE_READ, /* share mode */
                                  NULL, /* pointer to security attributes */
                                  OPEN_EXISTING, /* how to create */
                                  0,             /* file attributes */
                                  NULL ))) /* handle to template */
    {
      errMessage(filename, ((DWORD)-1));
    }
    else
    {
      HANDLE servIn, clientOut;
      SECURITY_ATTRIBUTES sa;
      sa.nLength = sizeof(SECURITY_ATTRIBUTES);
      sa.lpSecurityDescriptor = NULL;
      sa.bInheritHandle = TRUE;

      if (!CreatePipe(&servIn, &clientOut, &sa, 512))
      {
        errMessage("Unable to create output pipe", ((DWORD)-1));
      }
      else
      {
        HANDLE clientIn, servOut;
        if (!CreatePipe(&clientIn, &servOut, &sa, 512))
        {
          errMessage("Unable to create input pipe", ((DWORD)-1));
        }
        else
        {
          char buffer[1024];
          int wpos = 0, ok = lstrlen(filename);
          while (ok > 0 && filename[ok-1]!='\\' && 
                           filename[ok-1]!='/' &&
                           filename[ok-1]!=':')
            ok--;
          while (filename[ok] && filename[ok]!='.')
            buffer[wpos++] = filename[ok++];
          lstrcpy( &buffer[wpos], ".apipe.in" );
          wsprintf(&buffer[sizeof(buffer)-64],"%ld",(long)clientIn);
          ok = 0;
          if (SetEnvironmentVariable(buffer,&buffer[sizeof(buffer)-64]))
          {
            wsprintf(&buffer[sizeof(buffer)-64],"%ld",(long)clientOut);
            lstrcpy( &buffer[wpos], ".apipe.out" );
            if (SetEnvironmentVariable(buffer,&buffer[sizeof(buffer)-64]))
              ok = 1;
          }
          if (!ok)
          {
            errMessage("Unable to setup environment", ((DWORD)-1));
          }
          else
          {
            PROCESS_INFORMATION pi;
            STARTUPINFO si; 
            char cHandleInfo[sizeof(int)+(1*3)+(sizeof(HANDLE)*3)];

            si.cb = sizeof(STARTUPINFO);
            GetStartupInfo(&si);
            si.dwFlags &= ~STARTF_FORCEONFEEDBACK;
            si.dwFlags |= STARTF_FORCEOFFFEEDBACK;

            {
              char *p;
              if (si.cbReserved2 < (sizeof(int)+((1+sizeof(HANDLE))*3)) ||
                  si.lpReserved2 == NULL)
              {
                si.cbReserved2 = (sizeof(int)+((1+sizeof(HANDLE))*3));
                si.lpReserved2 = (void *)(&cHandleInfo[0]);
                *((int *)(si.lpReserved2)) = 3;
              }
              /* this is how the msvc runtime does it */
              #define m_FOPEN          0x01    /* file handle open */
              #define m_FPIPE          0x08    /* file handle refers to a pipe */
              #define m_FTEXT          0x80    /* file handle is in text mode */
              p = (char *)si.lpReserved2;               p+= sizeof(int);
              *p = (char)(m_FOPEN|m_FPIPE|m_FTEXT);     p++;
              *p = (char)(m_FOPEN|m_FPIPE|m_FTEXT);     p++;
              *p = (char)(m_FOPEN|m_FPIPE|m_FTEXT);     p++;
              *((HANDLE *)p) = clientIn;                p+= sizeof(HANDLE);
              *((HANDLE *)p) = clientOut;               p+= sizeof(HANDLE);
              *((HANDLE *)p) = clientOut;               p+= sizeof(HANDLE);
            }

            __constructCmdLine(filename, buffer, sizeof(buffer));

            if (!CreateProcess(NULL, /* ptr to name of filename to exec */
                               buffer, /* ptr to cmd line string */
                               &sa, /* ptr to process security attributes */
                               &sa, /* ptr to thread security attributes */
                               TRUE, /* handle inheritance flag */
                               NORMAL_PRIORITY_CLASS, /* creation flags  */
                               NULL, /* pointer to new environment block */
                               NULL, /* pointer to current directory name  */
                               &si, /* pointer to STARTUPINFO  */
                               &pi )) /* pointer to PROCESS_INFORMATION */
            {
              errMessage("Spawn failed", ((DWORD)-1));
            }
            else
            {
              HICON hIcons[2] = {0,0};
              DWORD stdoutType, stdinType, dwTemp, dwPollInterval;
              BOOL bClientClosed;
              char ansicmdbuf[128]; int ansicmdbuflen = 0;
              char intcmdbuf[128]; int intcmdbuflen = 0;
              char ansicmdquote=0, lastByte = 0;

              __getset_icons(hConsoleWnd, filename, hIcons);

              __dwChildPid = pi.dwProcessId;
              stdoutType = GetFileType(hStdout);
              stdinType = GetFileType(hStdin);

              if (stdinType == FILE_TYPE_CHAR)
              {
                if (!GetConsoleMode( hStdin, &dwTemp ))
                  stdinType = FILE_TYPE_UNKNOWN;
                else 
                {
                  dwTemp &= ~(ENABLE_WINDOW_INPUT|ENABLE_MOUSE_INPUT);
                  SetConsoleMode( hStdin, dwTemp);
                }
              }
              if (stdoutType == FILE_TYPE_CHAR)
              {
                if (!GetConsoleMode( hStdout, &dwTemp ))
                  stdoutType = FILE_TYPE_UNKNOWN;
                else
                {
                  dwTemp|= (ENABLE_PROCESSED_OUTPUT|
                            ENABLE_WRAP_AT_EOL_OUTPUT);
                  SetConsoleMode(hStdout,dwTemp);
                }
              }

              dwPollInterval = 500;
              bClientClosed = FALSE;
              while (!bClientClosed)
              {
                BOOL bPipeClosed = FALSE;
                char respBuffer[64];
                unsigned int respBuflen = 0;
                DWORD bytesRead, totalAvail, thisRemaining;

                if (stdinType == FILE_TYPE_CHAR)
                {
                  HANDLE aArray[2];
                  aArray[0] = pi.hProcess;
                  aArray[1] = hStdin;
                  if (WaitForMultipleObjects(2,&aArray[0],0,
                                     dwPollInterval) == WAIT_TIMEOUT)
                    dwPollInterval = INFINITE;
                  else
                    dwPollInterval = 0;
                }
                if (dwPollInterval != INFINITE)
                {
                  if (WAIT_TIMEOUT != WaitForSingleObject( 
                        pi.hProcess, dwPollInterval ))
                  {                        
                    GetExitCodeProcess(pi.hProcess, &dwExitCode);
                    bClientClosed = TRUE;
                    /* we still need to check for trailing client output */
                  }
                }

                bytesRead = 1; 
                while (bytesRead && !bPipeClosed)
                {
                  /*
                   * Check and parse input
                  */
                  if (!PeekNamedPipe( servIn,
                                      (LPVOID)&buffer[0],
                                      sizeof(buffer),
                                      &bytesRead,
                                      &totalAvail,
                                      &thisRemaining ))
                  {
                    bPipeClosed = TRUE;
                    errMessage("Broken pipe (1)", ((DWORD)-1));
                    break;
                  }
                  if (!totalAvail)
                    bytesRead = 0;
                  else if (!ReadFile( servIn,
                      (LPVOID)&buffer[0], sizeof(buffer), &bytesRead, NULL))
                  {
                    bPipeClosed = TRUE;
                    errMessage("Broken pipe (2)", ((DWORD)-1));
                    break;
                  }
                  if (!bytesRead)
                  {
                    dwPollInterval += 50;
                    if (dwPollInterval > 250)
                      dwPollInterval = 250;
                  }
                  else
                  {
                    DWORD bytePos;
                    char outputBuffer[128]; /* unflushed buffer */
                    int  outputBufferLen = 0;
                    dwPollInterval = 0;

                    for (bytePos = 0; bytePos < bytesRead; bytePos++)
                    {
                      char c = buffer[bytePos];
                      if (c == 0x03)
                      {
                        bPipeClosed = TRUE;
                        break; /* client closed pipe */
                      }
                      else if (c == 0x1B)
                      {
                        lastByte = c;
                      }
                      else if (c == ']' && lastByte == 0x1B)
                      {             
                        intcmdbuflen = 0; /* clear pending command */
                        intcmdbuf[intcmdbuflen++] = (char)0x1B;
                        intcmdbuf[intcmdbuflen++] = lastByte = c;
                        /* the rest will be done in the next if */
                      }
                      else if (intcmdbuflen)
                      {
                        if (intcmdbuflen < (sizeof(intcmdbuf)-2))
                          intcmdbuf[intcmdbuflen++] = c;
                        if (c == 0)
                        {
                          if (intcmdbuf[intcmdbuflen-1] == c)
                          {
                            if (intcmdbuflen > 2)
                            {
                              c = intcmdbuf[2];
                              if (c == '1') /* set console title */
                              {
                                SetConsoleTitle(&intcmdbuf[3]);
                              }
                              else if (c == '2') /* advertise cmd window */
                              {
                                unsigned long pos, l = 0;
                                for (pos = 3; intcmdbuf[pos]; pos++)
                                {
                                  l *= 10;
                                  l += (intcmdbuf[pos]-'0');
                                }
                                if (IsWindow((HWND)l))
                                  __advertised_cmd_hwnd = ((HWND)l);
                              }
                            }
                          }
                          intcmdbuflen = 0;
                        }
                      }
                      else if (c == '[' && lastByte == 0x1B)
                      {
                        ansicmdbuflen = 0; /* clear pending command */
                        ansicmdbuf[ansicmdbuflen++] = (char)0x1B;
                        ansicmdbuf[ansicmdbuflen++] = lastByte = c;
                        ansicmdquote = 0;
                        /* the rest will be done in the next if */
                      }
                      else if (ansicmdbuflen)
                      {
                        if (ansicmdbuflen < (sizeof(ansicmdbuf)-2))
                          ansicmdbuf[ansicmdbuflen++] = c;
                        if (c == '\'' || c == '\"')
                        {
                          if (ansicmdquote == 0)
                            ansicmdquote = c;
                          else if (c == ansicmdquote)
                            ansicmdquote = 0;
                        }
                        if (!ansicmdquote &&
                           (c>='A' && c<='Z') || (c>='a' && c<='z'))
                        {
                          if (ansicmdbuflen < (sizeof(ansicmdbuf)-1))
                          {           /* command fit completely in buffer */   
                            /* evaluate ansi commands here */
                            CONSOLE_SCREEN_BUFFER_INFO csbi;
                            COORD coord;
                            ansicmdbuf[ansicmdbuflen] = '\0';
                            
                            if (c == 'X' || c == 'Y') /* get size */
                            {
                              static int height = -1, width = 0;
                              if (height == -1)
                              {
                                height = width = 0;
                                if (stdoutType == FILE_TYPE_CHAR &&
                                 GetConsoleScreenBufferInfo(hStdout,&csbi)) 
                                {
                                  height = (csbi.srWindow.Bottom - 
                                            csbi.srWindow.Top) + 1;
                                  width =  (csbi.srWindow.Right - 
                                            csbi.srWindow.Left) + 1;
                                }
                              }
                              if (c == 'X')
                                respBuffer[0] = (char)width;
                              else
                                respBuffer[0] = (char)height;
                              respBuflen = 1;
                            }
                            else if (c == 'x' || c == 'y') /* get pos */
                            {
                              respBuffer[0] = 0; /* error condition */
                              if (stdoutType == FILE_TYPE_CHAR &&
                                GetConsoleScreenBufferInfo(hStdout, &csbi)) 
                              {
                                respBuffer[0] = (c == 'x') ?
                                 ((char)(((int)csbi.dwCursorPosition.X)+1))
                                :((char)(((int)csbi.dwCursorPosition.Y)+1));
                              }     
                              respBuflen = 1;
                            }
                            #if 0 
                            else if (....)
                            /* 
                             * other cmds that don't need a real tty
                            */
                            #endif
                            else if (stdoutType != FILE_TYPE_CHAR)
                              ;
                            else if (!GetConsoleScreenBufferInfo( hStdout, 
                                                                  &csbi))
                              ;
                            else if (c == 'J')
                            {
                              if (ansicmdbuf[2] == '2')
                              {
                                coord.X = coord.Y = 0;
                                FillConsoleOutputCharacter(hStdout, 
                                  (TCHAR) ' ', 
                                  (csbi.dwSize.X * csbi.dwSize.Y), 
                                  coord, &dwTemp);
                                FillConsoleOutputAttribute(hStdout, 
                                  csbi.wAttributes, 
                                  (csbi.dwSize.X * csbi.dwSize.Y), 
                                  coord, &dwTemp);
                              }
                            }
                            else if (c == 'K')
                            {
                              if (csbi.dwCursorPosition.X < csbi.dwSize.X)
                              {
                                coord.X = ((SHORT)csbi.dwCursorPosition.X);
                                coord.Y = ((SHORT)csbi.dwCursorPosition.Y);
                                FillConsoleOutputCharacter(hStdout, 
                                  (TCHAR) ' ', 
                                  (csbi.dwSize.X - csbi.dwCursorPosition.X), 
                                  coord, &dwTemp);
                                FillConsoleOutputAttribute(hStdout, 
                                  csbi.wAttributes, 
                                  (csbi.dwSize.X - csbi.dwCursorPosition.X),
                                  coord, &dwTemp);
                              }
                            }
                            else if (c == 'H' || c == 'f')
                            {
                              int row = 1+((int)csbi.dwCursorPosition.Y);
                              int col = 1+((int)csbi.dwCursorPosition.X);
                              if (__ansigetopts( ansicmdbuf, ansicmdbuflen,
                                                 &row, &col )==0)
                              {
                                dwTemp = 0; /* gotchange = FALSE; */
                                coord.X = ((SHORT)csbi.dwCursorPosition.X);
                                coord.Y = ((SHORT)csbi.dwCursorPosition.Y);
                                if (row >= 1 && row <= 
                                    (csbi.srWindow.Bottom - 
                                          csbi.srWindow.Top))
                                {
                                  row--;
                                  if (((int)(coord.Y)) != row)
                                  {
                                    dwTemp = 1;
                                    coord.Y = (SHORT)row;
                                  }
                                }
                                if (col >= 1 && col <= 
                                    (csbi.srWindow.Right - 
                                          csbi.srWindow.Left))
                                {
                                  col--;
                                  if (((int)(coord.X)) != col)
                                  {
                                    dwTemp = 1;
                                    coord.X = ((SHORT)col);
                                  }
                                }
                                if (dwTemp)
                                  SetConsoleCursorPosition(hStdout, coord);
                              }
                            }
                            else if (c >= 'A' && c <= 'D')
                            {
                              int diff = 0;
                              if (__ansigetopts( ansicmdbuf, ansicmdbuflen,
                                                 &diff, NULL )==0)
                              {
                                if (diff > 0)
                                {
                                  int col = ((int)csbi.dwCursorPosition.X);
                                  int row = ((int)csbi.dwCursorPosition.Y);
                                  if (c == 'A') row-=diff; /* cursor up */
                                  else if (c == 'B') row+=diff; /* down */
                                  else if (c == 'C') col-=diff; /* left */
                                  else if (c == 'D') col+=diff; /* right */
                                  if (row >= 0 
                                    && row < (csbi.srWindow.Bottom - 
                                              csbi.srWindow.Top)
                                    && col >= 0
                                    && col < (csbi.srWindow.Right - 
                                              csbi.srWindow.Left))
                                  {
                                    coord.X = (SHORT)col;
                                    coord.Y = (SHORT)row; 
                                    SetConsoleCursorPosition(hStdout,coord);
                                  }
                                }
                              }
                            }
                            else if (c == 's' || c == 'u')
                            {
                              static COORD ansi_su = {0,0};
                              static char ansi_su_lastcmd = 'u';
                              if (c == 's')
                              {
                                ansi_su.X =((SHORT)csbi.dwCursorPosition.X);
                                ansi_su.Y =((SHORT)csbi.dwCursorPosition.Y);
                                ansi_su_lastcmd = 's';
                              }
                              else if (ansi_su_lastcmd == 's')
                              {
                                SetConsoleCursorPosition(hStdout, ansi_su);
                                /* ansi_su_lastcmd == 'u'; */
                              }
                            }
                            /* end of ansi */
                          }
                          ansicmdbuflen = 0;
                        }
                      } /* else if (ansicmdbuflen) */
                      else /* normal unescaped char */
                      {
                        if (lastByte == 0x1B) /* whoops, literal esc */
                          outputBuffer[outputBufferLen++] = (char)0x1B;
                        outputBuffer[outputBufferLen++] = c;
                        lastByte = c;
                        if ( c == '\n' || outputBufferLen > 80)
                        {
                          __pushOutput( hStdout, &stdoutType,
                                      outputBuffer, outputBufferLen, FALSE);
                          outputBufferLen = 0;
                        }
                      }
                    } /* for (bytePos = 0; bytePos<bytesRead; bytePos++) */
                    
                    if (outputBufferLen) /* have unflushed data */
                    {
                      __pushOutput( hStdout, &stdoutType,
                                    outputBuffer, outputBufferLen, FALSE);
                    }
                  } /* if (bytesRead) */
                } /* while (bytesRead && !bPipeClosed) */

                if (bPipeClosed)
                  break;
                
                if (!bClientClosed)
                {
                  /* 
                   * Send output
                  */
 
                  if (respBuflen == 0) /* no data pending output */
                  {
                    if (stdinType == FILE_TYPE_CHAR)
                    {
                      if (!GetNumberOfConsoleInputEvents(hStdin,&totalAvail))
                      {
                        stdinType = FILE_TYPE_UNKNOWN;
                      }
                      else
                      {
                        while (totalAvail && 
                               respBuflen < (sizeof(respBuffer)-1))
                        {
                          INPUT_RECORD ir;
                          if (!PeekConsoleInput(hStdin, &ir,1,&thisRemaining))
                            break;
                          if (!thisRemaining)
                            break;
                          if (!ReadConsoleInput(hStdin, &ir,1,&thisRemaining))
                            break;
                          if (!thisRemaining)
                            break;
                          if (ir.EventType == KEY_EVENT)
                          {
                            if (ir.Event.KeyEvent.bKeyDown)
                            {
                              char c = (char)ir.Event.KeyEvent.uChar.AsciiChar;
                              if (c != 0)
                              {
                                respBuffer[respBuflen++] = c;
#if 0
                                {
                                  char scratch[128];
                                  wsprintf(scratch,"got char %d, '%c'", c, c );
                                  errMessage(scratch,0);
                                }    
#endif                             
                              }
                            }
                          }
                          if (!GetNumberOfConsoleInputEvents(hStdin,
                                        &totalAvail))
                            break;
                        }
                        FlushConsoleInputBuffer(hStdin);
                      }
                    } /* if (stdinType == FILE_TYPE_CHAR) */
                    if (stdinType != FILE_TYPE_CHAR) /* may have changed */
                    {
                      if (ReadFile( hStdin,
                         (LPVOID)&respBuffer[0], sizeof(respBuffer), 
                         &bytesRead, NULL))
                      {
                        respBuflen = (unsigned int)bytesRead;
                      }
                    }
                  }
                      
                  if (respBuflen != 0) /* have data pending output */
                  {  
                    if (!WriteFile( servOut, (LPCVOID)&respBuffer[0], 
                               respBuflen, &bytesRead, NULL))
                    {
                      errMessage("Broken pipe (3)", ((DWORD)-1));
                      break;
                    }
                    dwPollInterval = 0;
                    FlushFileBuffers(servOut);
                  }
                } /* if (!bClientClosed) */

              } /* while (WaitForSingleObject(hProcess,....) */

              __getset_icons(hConsoleWnd, NULL, hIcons);           
              __advertised_cmd_hwnd = NULL;
              __dwChildPid = 0;
              CloseHandle((HANDLE)pi.hProcess);
              CloseHandle((HANDLE)pi.hThread);
            }
          } /* env got set */
          CloseHandle(clientIn);
          CloseHandle(servOut);
        } /* created second pipe pair */
        CloseHandle(servIn);
        CloseHandle(clientOut);
      } /* created first pipe pair */
      CloseHandle(binHandle);
    } /* file opened ok */
  } /* got valid handles */
  return dwExitCode;
}
#endif /* HAVE_EXEC_WITH_PIPES */

/* ---------------------------------------------------------------- */

int main(void)
{
  char filename[MAX_PATH+1];
  DWORD filenamelen, dwExitCode = 2;

  SetProcessShutdownParameters(0x100,SHUTDOWN_NORETRY);
  SetConsoleCtrlHandler((PHANDLER_ROUTINE)TriggerControl,TRUE);

  if (4>(filenamelen=GetModuleFileName(NULL,filename,sizeof(filename))))
  {
    errMessage("Unable to get my application name", 0);
  }
  else 
  {
    lstrcpy( &filename[filenamelen-3],
       (lstrcmpi( &filename[filenamelen-3], "com" )==0)?("exe"):("com"));
    #if defined(HAVE_EXEC_AS_TEMPFILE) && defined(HAVE_EXEC_WITH_PIPES)
    if ( filename[filenamelen-3] == 'e' ) /* we're a .com, execute .exe */ 
      dwExitCode = __exec_with_pipes(filename);  /* cyp's */
    else                                  /* we're an .exe, execute .com */
      dwExitCode = __exec_as_tempfile(filename); /* davehart's */
    #elif defined(HAVE_EXEC_AS_TEMPFILE)
    dwExitCode = __exec_as_tempfile(filename); /* davehart's */
    #elif defined(HAVE_EXEC_WITH_PIPES)
    dwExitCode = __exec_with_pipes(filename);  /* cyp's */
    #else
    #error What's up doc?
    #endif
  }
  return (int)dwExitCode;
}  
