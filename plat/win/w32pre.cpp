/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 *  This is the "prelude" module for the win32/win16 console client.
 *  win32GUI shims are elsewhere.
 *
 *  "prelude" serves as a shim between WinMain() [in client.cpp] and
 *  realmain() [also in client.cpp]
*/

const char *w32pre_cpp(void) {
return "@(#)$Id: w32pre.cpp,v 1.1.2.1 2001/01/21 15:10:25 cyp Exp $"; }

#include "cputypes.h"  //CLIENT_OS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifndef MAX_PATH     //for win16
#define MAX_PATH 256 //don't make it less than 256
#endif
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "w32svc.h"  //win32Cli[InitializeService|IsServiceInstalled]()
#include "w32ini.h"  //WriteDCTIProfileString
#include "w32util.h" //int (PASCAL *__SSMAIN)(HINSTANCE,HINSTANCE,LPSTR,int);
#include "w32exe.h"  //install_cmd_exever(), install_cmd_copyfile()
#include "w32cons.h" //w32PostRemoteWCMD( DNETC_WCMD_SHUTDOWN );
#include "w32pre.h"  //ourselves

static struct 
{
  HINSTANCE hInstance;
  int nCmdShow;
  int argc;
  char **argv;
} prestatics = {NULL,0,0,NULL};

HINSTANCE winGetInstanceHandle(void) { return prestatics.hInstance; }
int winSetInstanceShowCmd(int nshow) { return (prestatics.nCmdShow = nshow); }
int winGetInstanceShowCmd(void)      { return prestatics.nCmdShow; }
int winGetInstanceArgc(void)         { return prestatics.argc; }
char **winGetInstanceArgv(void)      { return prestatics.argv; }

/* ---------------------------------------------------- */

int winGetMyModuleFilename(char *buffer, unsigned int len)
{
  if (buffer && len)
  {  
    if (prestatics.argv && len > 1)
    {
      strncpy(buffer,prestatics.argv[0],len);
      buffer[len-1] = '\0';
      return strlen(buffer);
    }
    buffer[0] = '\0';
  }
  return 0;
}

/* ---------------------------------------------------- */

static int parseCmdLine(const char *firstarg, const char *cmdline,
                        char *argbuff, unsigned int argbuffsize,
                        char *argv[], unsigned int maxargvelems )
{
  unsigned int argc=0;

  if (!argv || !maxargvelems)
    return 0;
  if (argv && maxargvelems >=2 && argbuff && argbuffsize >= 2)
  {
    char *aptr = &argbuff[0];
    char *eptr = &argbuff[argbuffsize-1];
    if (firstarg)
    {
      argv[argc++] = aptr;
      while (*firstarg && (aptr < eptr))
        *aptr++ = (char)(*firstarg++);
      *aptr++ = '\0';
    }
    if (cmdline)
    {
      maxargvelems--; /* last argv is a null */
      while (*cmdline && (aptr < eptr) && (argc < maxargvelems))
      {
        char bka = ' ', bkb = '\t';
        while (*cmdline==' ' || *cmdline=='\t')
          cmdline++;
        if (*cmdline == '\"' || *cmdline == '\'')
          bka = bkb = ((char)(*cmdline++));
        firstarg = cmdline;
        while (*cmdline && ((char)(*cmdline))!=bka && ((char)(*cmdline))!=bkb)
          cmdline++;
        if ((cmdline-firstarg) > ((eptr-aptr)+1))
          break;
        argv[argc++] = aptr;
        while (firstarg < cmdline)
          *aptr++ = (char)(*firstarg++);
        *aptr++ = '\0';
        if (*cmdline == bka)
          cmdline++;
      }
    }
  }
  argv[argc] = (char *)0;
  return (int)argc;
}  
  
/* ---------------------------------------------------- */

#if (CLIENT_OS == OS_WIN32)
static int __setgetappnameevar(char *inoutbuf, unsigned int inoutbufsize)
{
  char appName[20+MAX_PATH+1];
  char *cmdline = (char *)GetCommandLine();
  int nameLen, evarLen;
  nameLen = evarLen = strlen(strcpy( appName, "dnetc.exe=" ));

  if (cmdline)
  {
    if (*cmdline == '\"' || *cmdline == '\'')
    {
      char c = ((char)(*cmdline++));
      while (*cmdline && c != ((char)*cmdline))
        appName[nameLen++] = ((char)(*cmdline++));
    }
    else 
    {
      while (*cmdline && *cmdline != ' ' && *cmdline != '\t')
        appName[nameLen++] = ((char)(*cmdline++));
    }
    appName[nameLen] = '\0';
    if (nameLen != evarLen)   /* have a filename? */
    {                        
      if ((appName[evarLen+1] != ':') /* have drive spec */
          && (appName[evarLen]!='\\' || appName[evarLen+1]!='\\')) /* UNC */
        nameLen = evarLen;  /* not canonical, so fail */
      else if (GetFileAttributes( &appName[evarLen] ) == 0xFFFFFFFF)
        nameLen = evarLen;  /* file doesn't exist, so fail */
    }
  }
  if (nameLen != evarLen) /* have an apppath */
  {
    strncpy( inoutbuf, &appName[evarLen], inoutbufsize );
    inoutbuf[inoutbufsize-1] = '\0';
  }
  else /* no or incomplete apppath, use the default */
  {
    strncpy( &appName[evarLen], inoutbuf, (sizeof(appName)-evarLen));
    appName[sizeof(appName)-1] = '\0';
  }
  if (putenv( appName )!=0) /* broadcast it to the rest of the application */
  {
    MessageBox(NULL,"Not enough memory to initialize environment", 
                    appName, MB_ICONHAND|MB_OK);
    return -1;
  }
#if 0
  else
  {
    extern int __GetMyModuleFilename(char *buffer, unsigned int len);
    inoutbuf = getenv("dnetc.exe");
    if (!inoutbuf) inoutbuf = "getenv failed";
    sprintf(appName,"%d",GetCurrentThreadId());
    MessageBox(NULL,inoutbuf,appName,MB_OK);
    winGetMyModuleFilename(appName, sizeof(appName));
    __GetMyModuleFilename(appName, sizeof(appName));
  }
#endif
  return 0;
}
#endif

/* ---------------------------------------------------- */

#if (CLIENT_OS == OS_WIN32)
static void __auto_fixup_dnetc_scr_ver(const char *modfn)
{
  char ssfullpath[MAX_PATH];
  unsigned int which;
  char *argstack[4];

  argstack[0] = (char *)modfn;
  argstack[1] = ssfullpath;
  argstack[2] = (((winGetVersion()%2000)<400)?("3.10"):("4.0"));
  argstack[3] = NULL;

  for (which = 0; which <= 2; which++)
  {
    unsigned int pathlen;
    if (which == 0)
    {
      pathlen = strlen(modfn);
      while (pathlen>0 && modfn[pathlen-1]!='/' && 
             modfn[pathlen-1]!='\\' && modfn[pathlen-1]!=':')
        pathlen--;
      strcpy( ssfullpath, modfn );
      ssfullpath[pathlen] = '\0';
      modfn += pathlen;
    }
    else if (which == 1)
      pathlen = GetWindowsDirectory(ssfullpath,sizeof(ssfullpath));
    else
      pathlen = GetSystemDirectory(ssfullpath,sizeof(ssfullpath));

    if (pathlen)
    {
      char *p = (char *)modfn;
      unsigned int len = 0;
      if (ssfullpath[pathlen-1]!='\\')
      {
        ssfullpath[pathlen++]='\\';
        ssfullpath[pathlen]='\0';
      }
      while (pathlen < (sizeof(ssfullpath)-1))
      {
        if (!p[len] || p[len]=='.')
        {
          if (p[0] == '.')
            break;
          p = ".scr";
          len = 0;
        }
        ssfullpath[pathlen++] = (char)p[len++];
      }
      ssfullpath[pathlen] = '\0';
      install_cmd_exever( 3, &argstack[0] );
    }
  }
  return;
}    
#endif

/* ---------------------------------------------------- */
    
int winClientPrelude(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    LPSTR lpszCmdLine, int nCmdShow, int (*realmain)(int, char **))
{
  int argc;
  char *argv[64];
  char argbuff[MAX_PATH*2];

  GetModuleFileName(hInstance, argbuff, sizeof(argbuff));

  #if (CLIENT_OS == OS_WIN32)
  __auto_fixup_dnetc_scr_ver(argbuff);
  if (__setgetappnameevar( argbuff, sizeof(argbuff) )!=0)
    return -1;
  #endif

  argv[0] = argbuff; 
  argv[1] = NULL;

  prestatics.argc = argc = 1;
  prestatics.argv = &argv[0];
  prestatics.hInstance = hInstance;
  prestatics.nCmdShow = nCmdShow;

  prestatics.argc = argc = parseCmdLine( argbuff, lpszCmdLine, 
                               argbuff, sizeof(argbuff), 
                               &argv[0], (sizeof(argv)/sizeof(argv[0])) );

  /* the following .options are for internal use by the installer */
  if (__SSMAIN && argc > 1 && strcmp(argv[1], ".scr")==0)
  {
    WriteDCTIProfileString( "ScreenSaver", "launch", argv[0] );
    lpszCmdLine = strstr(lpszCmdLine, argv[1]) + 4;
    return (*__SSMAIN)(hInstance, hPrevInstance, lpszCmdLine, nCmdShow);
  }
  else if (argc > 3 && strcmp(argv[1], ".exever")==0)
  {
    argv[1] = argv[0];
    return install_cmd_exever( argc-1, &argv[1] );
  }
  else if (argc > 3 && strcmp(argv[1], ".copyfile")==0)
  {
    argv[1] = argv[0];
    return install_cmd_copyfile( argc-1, &argv[1] );
  }
  else if (argc > 1 && strcmp(argv[1], ".svcstart")==0)
  {                                           /* *installed* client */
    argv[1] = argv[0];
    return win32CliStartService( argc-1, &argv[1] ); /* <0=err, 0=ok */
  }                                                   
  else if (argc > 1 && strcmp(argv[1], ".shutdown")==0)
  {
    return w32PostRemoteWCMD( DNETC_WCMD_SHUTDOWN ); // <0=err (none there),
  }                                                  // 0=ok, >0=failed
  else if (argc > 1 && strcmp(argv[1], ".svcinstall")==0)
  {
    return win32CliInstallService(1); /* <0=err, 0=ok */
  }
  else if (argc > 1 && strcmp(argv[1], ".svcuninstall")==0)
  {
    return win32CliUninstallService(1); /* <0=err, 0=ok(or not installed) */
  }

  WriteDCTIProfileString( 0, "MRUClient", argv[0] );
  
  if ( win32CliInitializeService(argc,argv,realmain) == 0) /* started ok */
    return 0;  /* service started so quit here */

  return (*realmain)( argc, argv );
}

