/*
 * distributed.net client-for-NetWare console management functions.
 * written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * Functions in this module:
 *   int nwCliInitializeConsole(int hidden, int doingmodes);
 *   int nwCliDeinitializeConsole(int dopauseonclose);
 *   int nwCliCheckForUserBreak(void); <= called from triggers.cpp 
 *   int nwCliKbHit(void);
 *   int nwCliGetCh(void);
 *
 * $Id: nwccons.c,v 1.1.2.1 2001/01/21 15:10:29 cyp Exp $
*/

#include <stdio.h>    /* printf(), EOF */
#include <time.h>     /* time_t, time() */
#include <string.h>   /* memset */
#include <errno.h>    /* errno, strerror() */
#include <conio.h>    /* kbhit(), getch(), netware console functions */
#include <process.h>  /* GetThreadGroupID() */
#include <nwadv.h>    /* [Un]RegisterConsoleCommand(), AllocResourceTag() */

#include "triggers.h" /* [Check|Raise][Exit|Pause|Restart]RequestTrigger()*/
#include "nwcmisc.h"  /* nwCliGetBasename() [the name of out console cmd] */
#include "nwcconf.h"  /* nwCliLoadSettings(NULL) */
#include "nwccons.h"  /* ourselves */

#define K_CTRL_SIGINT   0x03 /* ^C */
#define K_CTRL_SIGTSTP  19   /* ^S */
#define K_CTRL_SIGCONT  17   /* ^Q */
#define K_CTRL_SIGHUP   18   /* ^R */

static struct
{
  int runhidden;
  int conhandle;
  int doingmodes;
  int kbbuffer[16];
  int kbbuflen;
  int wepaused;
  int cmdlinesig;
} nwconstatics = {0,0,0,{0},0,0,0};  

/* ===================================================================== */

static int __haveCLIBContext(void)
{
  int havectx = GetThreadGroupID();
  return (havectx != 0 && havectx != -1);
}

/* ===================================================================== */

static int __getcmdtouseforconcmd(char *buffer, unsigned int maxlen)
{
  #if 0
  char scratch[128];
  if ( -1 == GetNLMNameFromNLMID( GetNLMID(),
             concmdhandler.cmdname, scratch ))
    return -1;
  #else
  const char *p = nwCliGetNLMBaseName();
  if (!p || !*p)
    return -1;
  if (strlen(p) >= (maxlen-1))
    return -1;
  strcpy( buffer, p );
  return 0;
  #endif
}

static int __conCmdHandlerInitDeinit( int, LONG screenID, char *cmdline );
static LONG __conCmdHandlerCallback( LONG screenID, BYTE *cmdline ) 
{ return __conCmdHandlerInitDeinit( 0, screenID, (char *)cmdline ); }
static int __conCmdHandlerInitDeinit( int doWhat,LONG screenID, char *cmdline)
{
  static int concmdhandler_initialized = -1;
  static struct 
  { int cmdnamelen;
    char cmdname[64];
    struct commandParserStructure cmdparserstruct;
    void (*_RegisterConsoleCommand)( struct commandParserStructure * );
    LONG (*_UnRegisterConsoleCommand)( struct commandParserStructure * );
  } concmdhandler;

  if (doWhat > 0) /* initialize */
  {
    if (concmdhandler_initialized < 0)
    {
      unsigned int nlmHandle;      
      if ( !__haveCLIBContext() ) /* bah! the only thing we need ctx for */
        return -1;                /* is GetNLMHandle(). */
      nlmHandle = GetNLMHandle();
      memset( &concmdhandler, 0, sizeof(concmdhandler) );

      if (__getcmdtouseforconcmd( concmdhandler.cmdname, 
                           sizeof(concmdhandler.cmdname) )!=0)
        return -1;

      concmdhandler._RegisterConsoleCommand = 
          (void (*)(struct commandParserStructure *))
          ImportSymbol(nlmHandle, "RegisterConsoleCommand");
      if (!concmdhandler._RegisterConsoleCommand)
          return -1;
      concmdhandler._UnRegisterConsoleCommand = 
          (LONG (*)(struct commandParserStructure *))
          ImportSymbol(nlmHandle, "UnRegisterConsoleCommand");
      if (!concmdhandler._UnRegisterConsoleCommand)
      { 
        UnimportSymbol(nlmHandle, "RegisterConsoleCommand");
        return -1;
      }
        
      while (concmdhandler.cmdname[concmdhandler.cmdnamelen] &&
             concmdhandler.cmdname[concmdhandler.cmdnamelen] != '.')
        concmdhandler.cmdnamelen++;
      concmdhandler.cmdname[concmdhandler.cmdnamelen] = '\0';
      concmdhandler.cmdparserstruct.parseRoutine = __conCmdHandlerCallback;
      cmdline = "Console Command Processor"; /* unsigned char * */

      cmdline = "-restart, -[un]pause, -shutdown Command Parser";

      concmdhandler.cmdparserstruct.RTag = 
         AllocateResourceTag( nlmHandle, (BYTE *)cmdline, 
                              0x4D4F4343 /* ConsoleCommandSignature */);
      if (!concmdhandler.cmdparserstruct.RTag)
        return -1;
      concmdhandler_initialized = 0;
    }
    if (concmdhandler_initialized == 0)
    {
      (*concmdhandler._RegisterConsoleCommand)
         ( &(concmdhandler.cmdparserstruct) );
      concmdhandler_initialized = 1;
      return 0;
    }
  }
  else if (doWhat < 0) /* uninitialize */
  {
    if (concmdhandler_initialized > 0)
    {
      (*concmdhandler._UnRegisterConsoleCommand)
         ( &(concmdhandler.cmdparserstruct) );
      concmdhandler_initialized = 0;
      return 0;
    }
  }
  else /* handler */
  {
    unsigned int len, cmdlen;
    int for_us;

    while (*cmdline == ' ' || *cmdline == '\t')
      cmdline++;
    if (strlen(cmdline) > 4 && (cmdline[4] == ' ' || cmdline[4] == '\t') &&
        memicmp( cmdline, "load", 4 ) == 0) /* shouldn't happen... */
    {                                               
      cmdline+=4;
      while (*cmdline == ' ' || *cmdline == '\t')
        cmdline++;
    }

    for_us = 0;
    len = concmdhandler.cmdnamelen;
    cmdlen = strlen(cmdline);
    if ( cmdlen > len ) /* we don't do exactly "cmd" */
    {  
      if (cmdline[len] == ' ' || cmdline[len]=='\t')
        for_us = ( memicmp( cmdline, concmdhandler.cmdname, len ) == 0 );
      else if (cmdlen > (len+3) && (cmdline[len+3]==' ' || cmdline[len+3]=='\t')
           && memicmp( cmdline, concmdhandler.cmdname, len ) == 0)
      {
        for_us = ( memicmp( cmdline+len, "cmd", 3 ) == 0 );
        if (for_us) len+=3;
      }
    }
    if (for_us)
    {
      int quietly = 0;

      cmdline += len;
      while (*cmdline == ' ' || *cmdline == '\t')
        cmdline++;
      
      while (*cmdline)
      {
        char keyword[32];
        int hyphens[3];

        len = 0;
        while (*cmdline == '-' && len<(sizeof(hyphens)-1))
          hyphens[len++]=*cmdline++;
        hyphens[len] = '\0';
        
        len = 0;
        while (*cmdline && len<(sizeof(keyword)-1))
        {
          if (*cmdline == ' ' || *cmdline=='\t')
            break; 
          keyword[len++] = *cmdline++;
        }
        keyword[len] = '\0';
        
        while (*cmdline == ' ' || *cmdline == '\t')
           cmdline++;

        if ( strcmp( keyword, "quiet" )==0 || strcmp( keyword, "hide" )==0)
          quietly = 1;
        else
        {
          static struct   { const char *signame, *sigalias; int key; } 
          signame2key[]={ { "hup",  "restart",  K_CTRL_SIGHUP  },
                          { "tstp", "pause",    K_CTRL_SIGTSTP },
                          { "cont", "unpause",  K_CTRL_SIGCONT },
                          { "kill", "shutdown", K_CTRL_SIGINT  }  };
          for (len=0;len<(sizeof(signame2key)/sizeof(signame2key[0]));len++)
          {
            if ( strcmp( keyword, signame2key[len].signame ) == 0 ||
                 strcmp( keyword, signame2key[len].sigalias ) == 0 )
            {
              nwconstatics.cmdlinesig = signame2key[len].key;
              if (!quietly)
              {
                OutputToScreen( screenID, "%s: signal acknowledged. "
                "Process will %s shortly...\r\n", concmdhandler.cmdname,
                signame2key[len].sigalias );
              }
              return 0;
            }
          }
          if (!quietly && GetFileServerMajorVersionNumber() < 5)
          {
            OutputToScreen( screenID, "%s: Unrecognized keyword '%s%s'.\r\n",
                concmdhandler.cmdname, hyphens, keyword );
            return 0;
          }
          break; /* return error */
        } /* is not "-quiet" option */
      } /* while (*cmdline) */
    } /* is "appname ... " command */
  }
  return -1;
}

/* ===================================================================== */

static int __conInitDeinit(int which, int param1, int param2 )
{
  static int initialized = -1;
  int haveCLIBctx = (__haveCLIBContext());

  if (initialized == -1)
  {
    memset((void *)&nwconstatics,0,sizeof(nwconstatics));
    initialized = 0;
  }
  if (which > 0 && initialized == 0) /* init */
  {
    int hidden = param1;
    int doingmodes = param2;
    int handle;

    if (hidden && doingmodes)
      hidden = 0;

    handle = 0;
    if (!hidden)
    {
      if (!haveCLIBctx)
        errno = 22; /* EBADHNDL - What CLIB _would_ return for bad ctx */
      else
        handle = CreateScreen( "distributed.net client for NetWare",
                      AUTO_DESTROY_SCREEN|DONT_CHECK_CTRL_CHARS );
      if (handle == 0 || handle == -1)
      {
        ConsolePrintf("%s: Unable to create client screen (%s)\r\n",
                 nwCliGetNLMBaseName(), strerror(errno));
        return -1;
      }
    }
    memset((void *)&nwconstatics,0,sizeof(nwconstatics));
    nwconstatics.conhandle = handle;
    nwconstatics.runhidden = hidden;
    nwconstatics.doingmodes = doingmodes;
    if (!doingmodes)
      __conCmdHandlerInitDeinit( +1, 0, "");
    initialized = +1;
  }
  else if (which < 0 && initialized > 0) /* deinitialize */
  {
    int dopauseonclose = param1;
    int handle = nwconstatics.conhandle;
    int doingmodes = nwconstatics.doingmodes;

    memset((void *)&nwconstatics,0,sizeof(nwconstatics));
    if (!doingmodes)
      __conCmdHandlerInitDeinit( -1, 0, "");

    if (haveCLIBctx && handle && dopauseonclose)
    {
       time_t nowtime = 0, endtime = 0;
       unsigned short row, init;
       if (GetPositionOfOutputCursor( &row, &init /* dummy */ ) == 0)
       {
         unsigned short height;
         if (GetSizeOfScreen( &height, &init /* dummy */ ) == 0)
           gotoxy((short)0, (short)(height-((row<(height-2))?(3):(1))) );
        }
        init = 0;
        do
        {
          nowtime = time(NULL);
          if (endtime == 0)
            endtime = nowtime + 15;
          if (kbhit() || CheckExitRequestTriggerNoIO())
            break;
          if (nowtime < endtime)
          {
            if (!init)
            {
              if (DisplayScreen( handle )!=0)
                break;
            }
            printf( "%sPress any key to continue... %d  ",
                    ((!init)?("\n\n"):("\r")), (int)(endtime-nowtime) );
            delay(220);
          }
          init = 1;
        } while (nowtime < endtime);
      }
      
    if (haveCLIBctx && handle)
    {
      DestroyScreen(handle);
      /* 
      the next bit is a workaround for rconsole
      which doesn't update its own list of screens
      when a screen dies, but goes on showing the 
      dead screen.
      */
      handle = CreateScreen("System Console",0);
      if (handle)
      {
        DisplayScreen(handle);
        DestroyScreen(handle);
      }
    } 
    initialized = 0;
  }
  return 0;
}

int nwCliInitializeConsole(int hidden, int doingmodes)
{ 
  nwCliLoadSettings(NULL); /* use last inifilename */
  return __conInitDeinit(+1, hidden, doingmodes);
}

int nwCliDeinitializeConsole(int dopauseonclose)
{ return __conInitDeinit(-1, dopauseonclose, 0); }

/* ===================================================================== */

static int __keybpoll(void) /* returns !0 if raised sig */
{
  static int lastconhandle = -1;
  int sigset = 0;
  __conInitDeinit(0, 0, 0 );
  if (__haveCLIBContext())
  {
    int ch = 0, oflow = 0;
    if (lastconhandle != nwconstatics.conhandle)
    {
      nwconstatics.kbbuflen = 0;
      lastconhandle = nwconstatics.conhandle;
    }
    for (;;)
    {
      if (nwconstatics.cmdlinesig)
      {
        ch = nwconstatics.cmdlinesig;
        nwconstatics.cmdlinesig = 0;
      }
      /* if (CheckIfScreenDisplayed(nwconstatics.conhandle,0)) */
      else if (nwconstatics.conhandle && kbhit())
      {
        errno = 0;
        ch = getch();
        if (ch == EOF && errno)
        {
          for (ch=0; ch<50 && kbhit(); ch++)
          {
            if (!getch())
              getch();
          }
          ch = 0;
          break;
        }
        else if (ch == 0)
          ch = ((getch() << 8) & 0xff);
      }
      else
      {
        break;
      }
      if (ch == K_CTRL_SIGINT)
      {
        RaiseExitRequestTrigger();
        sigset = ch;
        break;
      }
      else if (!nwconstatics.doingmodes && ch == K_CTRL_SIGTSTP) /* ^S */
      {
        nwconstatics.wepaused = 1;
        RaisePauseRequestTrigger();
        sigset = ch;
        break;
      }
      else if (!nwconstatics.doingmodes && ch == K_CTRL_SIGCONT) /* ^Q */
      {
        if (nwconstatics.wepaused)
        {
          nwconstatics.wepaused = 0;
          ClearPauseRequestTrigger();
        }
      }
      else if (!nwconstatics.doingmodes && ch == K_CTRL_SIGHUP) /* ^R */
      {
        RaiseRestartRequestTrigger();
      }
      else if (ch != 0)
      {
        if (((unsigned int)nwconstatics.kbbuflen) < 
          (sizeof(nwconstatics.kbbuffer)/sizeof(nwconstatics.kbbuffer[0])))
          nwconstatics.kbbuffer[nwconstatics.kbbuflen++]=ch;
        else
          oflow = 1;
      }
    }
    if (sigset)
    {
      nwconstatics.kbbuflen = 0;
      for (ch=0; ch<50 && kbhit(); ch++)
      {
        if (!getch())
          getch();
      }
    }
    if (oflow)
      printf("\a"); /* RingTheBell() */
  }  
  return sigset;
}  
  
int nwCliCheckForUserBreak(void)
{
  return __keybpoll();
}

int nwCliKbHit(void)
{
  __conInitDeinit(0, 0, 0 );
  if (__haveCLIBContext())
  {
    if (DisplayInputCursor() == 0) /* does a threadswitch */
    {
      if (__keybpoll())
        return 0;
      return (nwconstatics.kbbuflen != 0);
    }
  }
  return 0;
}

int nwCliGetCh(void)
{
  int ch = 0;
  __conInitDeinit(0, 0, 0 );
  if (__haveCLIBContext())
  {
    while (nwconstatics.conhandle)
    {
      if ((ch = __keybpoll()) != 0)
        break;
      if (nwconstatics.kbbuflen)
      {
        ch = nwconstatics.kbbuffer[0];
        if (ch && ((ch & 0xff) == 0))
          nwconstatics.kbbuffer[0] = (ch >> 8);
        else 
        {
          int i; nwconstatics.kbbuflen--;
          for (i=1; i<=nwconstatics.kbbuflen; i++)
            nwconstatics.kbbuffer[i-1] = nwconstatics.kbbuffer[i];
        }
        if (ch != 0)
        {
          ch &= 0xff; /* may be zero */
          break;
        }
      }
      if (DisplayInputCursor() != 0) /* does a threadswitch */
        break;
      if (!nwconstatics.conhandle)
        break;
      delay(110);
    }
  }
  return ch;
}  
  
