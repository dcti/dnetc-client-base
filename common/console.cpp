// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// ----------------------------------------------------------------------
// This module contains the screen i/o primitives/wrappers for all
// those GUIs and less-than-G UIs we have been threatened with :)
// and we pretty much have scattered about anyway.
//
// Implementation guidelines: none. see what the neighbour did.
// Keep the functions small (total less than 25 lines) or make calls
// to functions in modules in your own platform area.   - cyp
// ----------------------------------------------------------------------
//
// $Log: console.cpp,v $
// Revision 1.42  1999/02/19 03:32:56  silby
// Uses termios for hpux now.
//
// Revision 1.41  1999/02/04 22:49:06  trevorh
// Corrected another problem with Vio calls being incorrect
//
// Revision 1.40  1999/02/04 14:50:34  patrick
//
// added TERMIOS support for AIX, now the menues work again. There seams to be
// a problem with non-termios unix clients and the cursor postioning though.
// The initial position is at pos 1 (for integers between 1 and 9 at least).
// This pos is counted as zero though. Thus the old value can not be erased.
//
// Revision 1.39  1999/01/31 20:19:08  cyp
// Discarded all 'bool' type wierdness. See cputypes.h for explanation.
//
// Revision 1.38  1999/01/29 18:50:33  jlawson
// fixed formatting.  changed some int vars to bool.
//
// Revision 1.37  1999/01/29 04:15:35  pct
// Updates for the initial attempt at a multithreaded/multicored Digital
// Unix Alpha client.  Sorry if these changes cause anyone any grief.
//
// Revision 1.36  1999/01/28 00:20:18  trevorh
// Corrected VioCalls for OS/2 and fixed getch() in Watcom!
//
// Revision 1.35  1999/01/25 23:49:07  trevorh
// #ifdef around the WinMessageBox() calls to make them used only when the
// OS/2 GUI is being compiled. WinMessageBox() is invalid in the CLI.
// VioGetCurPos() will not compile under Watcom 11.0
//
// Revision 1.34  1999/01/24 23:27:43  silby
// Change so conisatty even when win32gui is run in hidden mode.
//
// Revision 1.33  1999/01/19 15:37:25  patrick
//
// had to add sys/select.h for AIX 4.1 compiles. AIX 4.2 and later has the
// stuff in sys/time.h, which is included as default.
//
// Revision 1.32  1999/01/19 09:46:48  patrick
//
// changed behaviour for OS2-EMX to be more *ix like, added some casts.
//
// Revision 1.31  1999/01/13 08:49:27  cramer
// Removed the stdin isatty() check -- who gives a flip if stdin is a tty;
// stdout is what matters. (Note: if rc5des is started from a script...)
//
// Revision 1.30  1999/01/12 15:01:41  cyp
// Created an itty-bitty ConBeep(). (used by Client::Configure())
//
// Revision 1.29  1999/01/09 05:46:08  cyp
// x86dos clear screen fix.
//
// Revision 1.28  1999/01/07 02:15:57  cyp
// ConInStr() now has a special 'boolean' mode. woohoo!
//
// Revision 1.27  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.26  1998/12/30 08:56:31  silby
// Conclose_delay is now not active if WIN32GUI defined.
//
// Revision 1.25  1998/12/29 09:28:35  dicamillo
// For MacOS, ConOut now call macConOut.
//
// Revision 1.24  1998/12/08 05:40:19  dicamillo
// MacOS: define conistty; call yield routine after output; remove
// GetKeys- Mac client never reads console input.
//
// Revision 1.23  1998/11/16 19:07:06  cyp
// Fixed integer truncation warnings.
//
// Revision 1.22  1998/11/16 17:04:38  cyp
// Cleared up a couple of unused parameter warnings
//
// Revision 1.21  1998/11/11 05:09:31  cyp
// Passed on 'doingmodes' flag from InitializeConsole() to w32InitializeCon -
// used on win16 to ensure only a single instance of the client can run.
//
// Revision 1.20  1998/11/10 21:36:02  cyp
// Changed InitializeConsole() so that terms know in advance whether the
// client will be running "modes" or not.
//
// Revision 1.19  1998/11/10 09:49:28  silby
// added termios for freebsd, console input works much nicer now.
//
// Revision 1.18  1998/11/09 17:24:35  chrisb
// Added riscos_backspace() to work round some lame implementation of consoles.
//
// Revision 1.17  1998/11/09 16:54:04  cyp
// Added bksp handling for any ANSI compliant term.
//
// Revision 1.16  1998/11/08 22:24:15  foxyloxy
// Really did the below comment this time.
//
// Revision 1.15  1998/11/08 22:16:19  foxyloxy
// Made sure that termios.h is included for Irix/Solaris builds.
// This fixes the "I can't backspace" problem.
//
// Revision 1.14  1998/11/08 19:05:02  cyp
// Created new function ConGetSize(int *width, int *height) from stuff in
// DisplayHelp().
//
// Revision 1.13  1998/10/31 03:31:39  sampo
// removed MacOS specific #include, checked for EOF input
//
// Revision 1.12  1998/10/29 03:15:26  sampo
// Finally got a MacOS keyboard input thingie.  Not final, but close.
//
// Revision 1.11  1998/10/26 02:53:55  cyp
// Added "Press any key..." functionality to DeinitializeConsole()
//
// Revision 1.10  1998/10/19 12:42:15  cyp
// win16 changes
//
// Revision 1.9  1998/10/11 05:24:29  cyp
// Implemented ConIsScreen(): a real (not a macro) isatty wrapper.
//
// Revision 1.8  1998/10/11 00:53:10  cyp
// new win32 callouts: w32ConOut(), w32ConGetCh(), w32ConKbhit()
//
// Revision 1.7  1998/10/07 20:43:37  silby
// Various quick hacks to make the win32gui operational again (will
// be cleaned up).
//
// Revision 1.6  1998/10/07 18:36:18  silby
// Changed logic in ConInKey once more so it's not reading uninit
// variables.  Should be solid now. :)
//
// Revision 1.5  1998/10/07 12:56:46  silby
// Reordered Deinitconsole so console functions would still be
// available during w32deinitconsole.
//
// Revision 1.4  1998/10/07 12:25:04  silby
// Figured out that MSVC doesn't understand continue as it was used;
// changed ConInKey's loop so that it doesn't rely on continue.
// (Functionality unchanged.)
//
// Revision 1.3  1998/10/07 04:04:20  silby
// Fixed ConInKey - the logic was reversed when checking for timeout
//
// Revision 1.2  1998/10/04 18:55:58  remi
// We want to output something, even stdout is redirected, grr...
//
// Revision 1.1  1998/10/03 05:34:45  cyp
// Created.
//
//
#if (!defined(lint) && defined(__showids__))
const char *console_cpp(void) {
return "@(#)$Id: console.cpp,v 1.42 1999/02/19 03:32:56 silby Exp $"; }
#endif

#define CONCLOSE_DELAY 15 /* secs to wait for keypress when not auto-close */

#include "cputypes.h"
#include "baseincs.h"
#include "network.h"
#include "clitime.h"
#include "triggers.h"
#include "console.h" //also has CLICONS_SHORTNAME, CLICONS_LONGNAME
#include "sleepdef.h" //usleep
#if (CLIENT_OS==OS_AIX)
#include <sys/select.h>		// only needed if compiled on AIX 4.1
#endif

#if !defined(NOTERMIOS) && ((CLIENT_OS==OS_SOLARIS) || (CLIENT_OS==OS_IRIX) || \
    (CLIENT_OS==OS_LINUX) || (CLIENT_OS==OS_NETBSD) || (CLIENT_OS==OS_BEOS) \
    || (CLIENT_OS==OS_FREEBSD) || defined(__EMX__) || (CLIENT_OS==OS_AIX) \
    || (CLIENT_OS==OS_DEC_UNIX) || (CLIENT_OS==OS_HPUX) ) 
#include <termios.h>
#define TERMIOS_IS_AVAILABLE
#endif

#if (CLIENT_OS == OS_DEC_UNIX)    || (CLIENT_OS == OS_HPUX)    || \
    (CLIENT_OS == OS_QNX)         || (CLIENT_OS == OS_OSF1)    || \
    (CLIENT_OS == OS_BSDI)        || (CLIENT_OS == OS_SOLARIS) || \
    (CLIENT_OS == OS_IRIX)        || (CLIENT_OS == OS_SCO)     || \
    (CLIENT_OS == OS_LINUX)       || (CLIENT_OS == OS_NETBSD)  || \
    (CLIENT_OS == OS_UNIXWARE)    || (CLIENT_OS == OS_DYNIX)   || \
    (CLIENT_OS == OS_MINIX)       || (CLIENT_OS == OS_MACH10)  || \
    (CLIENT_OS == OS_AIX)         || (CLIENT_OS == OS_AUX)     || \
    (CLIENT_OS == OS_OPENBSD)     || (CLIENT_OS == OS_SUNOS)   || \
    (CLIENT_OS == OS_ULTRIX)      || (CLIENT_OS == OS_DGUX)    || \
    (CLIENT_OS == OS_VMS)         || (CLIENT_OS == OS_OS390)   || \
    (CLIENT_OS == OS_OS9)         || (CLIENT_OS == OS_BEOS)    || \
    (CLIENT_OS == OS_MVS)         || (CLIENT_OS == OS_MACH10)
#define TERM_IS_ANSI_COMPLIANT
#endif

#if (CLIENT_OS == OS_RISCOS)
extern "C" void riscos_backspace();
#endif

/* ---------------------------------------------------- */

static struct
{
  int initlevel;
  int runhidden;
  int conisatty;
  int doingmodes;
} constatics = {0,0,0,0};

/* ---------------------------------------------------- */

int DeinitializeConsole(void)
{
  if (constatics.initlevel == 1)
    {
    if (constatics.doingmodes && !constatics.runhidden)
      {
      #if (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S) || \
          ((CLIENT_OS==OS_WIN32) && (!defined(WIN32GUI))) || \
          (CLIENT_OS==OS_NETWARE) || \
          ( (CLIENT_OS==OS_OS2) && !defined (__EMX__) )
        {
        int init = 0;
        time_t endtime = (CliTimer(NULL)->tv_sec) + CONCLOSE_DELAY;
        int row = -1, height = 0;
        ConGetPos(NULL, &row);
        ConGetSize(NULL, &height);
        if (height > 2 && row != -1)
          ConSetPos(0, height-((row<(height-2))?(3):(1)));
        do
          {
          char buffer[80];
          int remaining = (int)(endtime - (CliTimer(NULL)->tv_sec));
          if (remaining <= 0)
            break;
          sprintf( buffer, "%sPress any key to continue... %d  ",
                   ((!init)?("\n\n"):("\r")), remaining );
          init = 1;
          ConOut( buffer );
          } while (ConInKey(1000) == 0);
        }
      #endif
      }
    #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
      w32DeinitializeConsole();
    #endif
      }

  constatics.initlevel--;

  return 0;
}

/* ---------------------------------------------------- */

int InitializeConsole(int runhidden,int doingmodes)
{
  int retcode = 0;

  if ((++constatics.initlevel) == 1)
    {
    memset( (void *)&constatics, 0, sizeof(constatics) );
    constatics.initlevel = 1;
    constatics.runhidden = runhidden;
    constatics.doingmodes = doingmodes;

    #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
    retcode = w32InitializeConsole(runhidden,doingmodes);
    #ifdef WIN32GUI
    runhidden=0;
    #endif
    #endif

    if (retcode != 0)
      --constatics.initlevel;
    else if (!runhidden)
      {
      #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
        constatics.conisatty = w32ConIsScreen();
      #elif (CLIENT_OS == OS_RISCOS)
        constatics.conisatty = 1;
      #elif (CLIENT_OS == OS_MACOS)
        constatics.conisatty = 1;
      #else
        constatics.conisatty = (isatty(fileno(stdout)));
      #endif
      }
    } /* constatics.initlevel == 1 */

  return retcode;
}

/* ---------------------------------------------------- */

int ConIsScreen(void)
{
  return (constatics.initlevel > 0 && constatics.conisatty );
}

/* ---------------------------------------------------- */

/*
** ConBeep() does uh..., well..., like... causes the console speaker to beep
*/
int ConBeep(void)
{
  if (constatics.initlevel > 0 && constatics.conisatty) /*can't beep to file*/
    {
    ConOut("\a");
    return 0;
    }
  return -1;
}

/* ---------------------------------------------------- */

/*
** ConOut() does what printf("%s",str) would do
*/
int ConOut(const char *msg)
{
  if (constatics.initlevel > 0 /*&& constatics.conisatty*/ )
    {
    #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
      w32ConOut(msg);
    #elif (CLIENT_OS == OS_MACOS)
      macConOut(msg);
    #else
      fwrite( msg, sizeof(char), strlen(msg), stdout);
      fflush(stdout);
    #endif
    return 0;
    }
  return -1;
}

/* ---------------------------------------------------- */

/*
** ConOutModal() should only be used when the console is known to be
** uninitialized and should be avoided. Not affected by -hidden/-quiet mode
*/

int ConOutModal(const char *msg)
{
  #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
     MessageBox( NULL, msg, CLICONS_LONGNAME,
           MB_OK | MB_ICONINFORMATION );
  #elif (CLIENT_OS == OS_OS2) && defined(OS2_PM)
    WinMessageBox( HWND_DESKTOP, HWND_DESKTOP, msg,
       CLICONS_LONGNAME, (PSZ)NULL, MB_OK | MB_INFORMATION | MB_MOVEABLE );
  #else
    fprintf( stderr, "%s\n", msg );
    fflush( stderr );
  #endif
  return 0;
}

/* ---------------------------------------------------- */

/*
** ConOutErr() does what fprintf(stderr "\nRC5DES: %s\n",msg) would do.
** Can be blocking. Note the leading and trailing newlines.
*/

int ConOutErr(const char *msg)
{
  #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
    MessageBox( NULL, msg, CLICONS_LONGNAME,
                 MB_OK | MB_TASKMODAL | MB_ICONSTOP /*MB_ICONERROR*/ );
  #elif (CLIENT_OS == OS_OS2) && defined(OS2_PM)
     WinMessageBox( HWND_DESKTOP, HWND_DESKTOP, (PSZ)msg,
           CLICONS_LONGNAME,  (PSZ)NULL, MB_OK | MB_APPLMODAL | MB_ERROR | MB_MOVEABLE );
  #elif (CLIENT_OS == OS_NETWARE)
    ConsolePrintf( "%s: %s\r\n", CLICONS_SHORTNAME, msg );
  #else
    fprintf( stderr, "%s: %s\n", CLICONS_SHORTNAME, msg );
    fflush( stderr );
  #endif
  return 0;
}

/* ---------------------------------------------------- */

/*
** ConInKey() does what a (non-blocking and polling) DOS-ish getch() would do
** key is not echoed. timeout ==> 0 == don't wait, -1 == wait forever.
*/

int ConInKey(int timeout_millisecs) /* Returns -1 if err. 0 if timed out. */
{
  timeval timenow, timestop;
  int ch = -1;

  if (constatics.initlevel > 0 && constatics.conisatty)
    {
    if (timeout_millisecs > 0)
      {
      CliTimer(&timestop);
      timestop.tv_sec += timeout_millisecs/1000;
      timestop.tv_usec += ( timeout_millisecs % 1000 )*1000;
      timestop.tv_sec += ( timestop.tv_usec / 1000000 );
      timestop.tv_usec %= 1000000;
      }
    ch = 0;

    do
      {
      #if (CLIENT_OS == OS_RISCOS)
        {
        ch = _swi(OS_ReadC, _RETURN(0));
        }
      #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S) || \
         (CLIENT_OS == OS_WIN32)
        {
        if (w32ConKbhit())
          {
          ch = w32ConGetch();
          if (!ch)
            ch = (w32ConGetch() << 8);
          }
        }
      #elif (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_NETWARE) || \
         ( (CLIENT_OS == OS_OS2) && !defined(__EMX__)  )
        {
        fflush(stdout);
        if (kbhit())
          {
          ch = getch();
          if (!ch)
            ch = (getch() << 8);
          }
        }
      #elif (defined(TERMIOS_IS_AVAILABLE))
        {
        struct termios stored;
        struct termios newios;

        fflush(stdout);
        tcgetattr(0,&stored); /* Get the original termios configuration */
        memcpy(&newios,&stored,sizeof(struct termios));
        newios.c_lflag &= ~(ICANON|ECHO|ECHONL); /* disable canonical mode */
        newios.c_cc[VTIME] = 0;                  /* ... and echo */
        newios.c_cc[VMIN] = 1;         /* set buffer size to 1 byte */
        tcsetattr(0,TCSANOW,&newios);  /* Activate the new settings */
        ch = getchar();                /* Read the single character */
        tcsetattr(0,TCSANOW,&stored);  /* Restore the original settings */
        if (ch == EOF) ch = 0;
        }
      #elif (CLIENT_OS == OS_MACOS)
      // Mac code never does console input
      #else
        {
        setvbuf(stdin, (char *)NULL, _IONBF, 0);

        int fd = fileno(stdin);
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET( fd, &rfds );

        fflush(stdout);
        fflush(stdin);
        errno = 0;
        if ( select( fd+1, &rfds, NULL, NULL, NULL) && errno != EINTR )
          {
          ch = fgetc( stdin );
          if (ch == EOF)
            ch = 0;
          }
        }
      #endif
      if (ch || timeout_millisecs == 0 || CheckExitRequestTriggerNoIO())
        break;
      usleep(50*1000);

      CliTimer(&timenow);

      } while ( (timeout_millisecs == -1) ||
              (( timenow.tv_sec < timestop.tv_sec ) ||
               (( timenow.tv_sec == timestop.tv_sec ) &&
               ( timenow.tv_usec < timestop.tv_usec ))));
    }

  return ch;
}

/* ---------------------------------------------------- */

/* ConInStr() is similar to readline(); buffer is always '\0' terminated.
** Returns -1 if err (stdin is not a tty)
*/

int ConInStr(char *buffer, unsigned int buflen, int flags )
{
  int ch;
  int exitreq;
  unsigned int pos;
  int asbool, boolistf, boolval, redraw;

  if (constatics.initlevel < 1 || !constatics.conisatty)
    return -1;

  if (!buffer || !buflen)
    return 0;

  //if ((flags & CONINSTR_ASPASSWORD) != 0)
  //  flags = CONINSTR_ASPASSWORD;

  redraw = asbool = boolval = boolistf = 0;
  if ((flags & CONINSTR_ASBOOLEAN) != 0)
    {
    asbool = 1;
    if ((flags & CONINSTR_BYEXAMPLE) != 0)
      {
      if (buffer[0] =='t' || buffer[0]=='T')
        boolistf = boolval = 1;
      else if (buffer[0] == 'f' || buffer[0] == 'F')
        boolistf = 1;
      else if (buffer[0] == 'y' || buffer[0] == 'Y')
        boolval = 1;
      else if (buffer[0] == 'n' || buffer[0] == 'N') /* default */
        boolval = 0;
      }
    flags = CONINSTR_ASBOOLEAN; /* strip by_example */
    if (buflen > 1) /* we only return a single char */
      buffer[1] = 0;
    redraw = 1;
    }


  if ((flags & CONINSTR_BYEXAMPLE) != 0)
    {
    pos = strlen( buffer );
    if ((flags & CONINSTR_ASPASSWORD)!=0)
      {
      char scratch[2];
      scratch[1] = 0; scratch[0] = '*';
      for (ch = 0; ch < ((int)(pos)); ch++)
        ConOut(scratch);
      }
    else
      {
      ConOut(buffer);
      }
    }
  else
    {
    pos = 0;
    buffer[pos] = 0;
    }

  do
    {
    if (asbool && redraw)
      {
      char scratch[8];
      strcpy(scratch, ((boolval)?((boolistf)?("1 "):("yes  ")):
                                ((boolistf)?("0"):("no   "))) );
      #if (CLIENT_OS == OS_RISCOS)
      if (redraw) /* not the first round */
        riscos_backspace();
      #endif

      ConOut(scratch);

      for (ch = 0; scratch[ch] != 0; ch++)
        {
        #ifdef TERM_IS_ANSI_COMPLIANT
        ConOut("\x1B" "[1D" );
        #elif (CLIENT_OS == OS_RISCOS)
        if (scratch[ch+1]!=0) /* not the first char */
         riscos_backspace();
        #else
        ConOut("\b");
        #endif
        }
      buffer[0] = ((boolval)?('y'):('n'));
      pos = 1;
      redraw = 0;
      }

    ch = ConInKey(-1);
    exitreq = (CheckExitRequestTriggerNoIO() != 0 ? 1 : 0);

    if (!exitreq)
      {
      if (ch == '\n' || ch == '\r')
        {
        ConOut("\n");
        exitreq = 1;
        }
      else if (asbool)
        {
        if (ch == 'y' || ch == 'Y')
          {
          redraw = (boolistf || !boolval);
          boolistf = 0; boolval = 1;
          }
        else if (ch == 't' || ch == 'T')
          {
          redraw = (!boolistf || !boolval);
          boolistf = boolval = 1;
          }
        else if (ch == 'n' || ch == 'N')
          {
          redraw = (boolistf || boolval);
          boolistf = boolval = 0;
          }
        else if (ch == 'f' || ch == 'F')
          {
          redraw = (!boolistf || boolval);
          boolistf = 1; boolval = 0;
          }
        else
          {
          ConBeep();
          }
        }
      else if (ch == 0x08 || ch == '\177') /* backspace */
        {
        if (pos > 0)
          {
          #ifdef TERM_IS_ANSI_COMPLIANT
          ConOut("\x1B" "[1D" " " "\x1B" "[1D");
          #elif (CLIENT_OS == OS_RISCOS)
          riscos_backspace();
          #else
          ConOut("\b \b");
          #endif
          pos--;
          }
        }
      else if (pos < (buflen-1))
        {
        buffer[pos++] = (char)ch;
        if ((flags & CONINSTR_ASPASSWORD) != 0)
          ch = '*';
        if (isalpha(ch) || isspace(ch) || isdigit(ch) || ispunct(ch))
          {
          /* if (!isctrl(ch)) */
          char x[2];
          x[0]=(char)ch;
          x[1]=0;
          ConOut(x);
          }
        }
      }
    } while (!exitreq);

  ConOut(""); /* flush */
  buffer[pos] = 0;
  return pos; /* length */
}


/* ---------------------------------------------------- */

int ConGetPos( int *col, int *row )  /* zero-based */
{
  if (constatics.initlevel > 0 && constatics.conisatty)
    {
    #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
    return w32ConGetPos(col,row);
    #elif (CLIENT_OS == OS_NETWARE)
    short x, y;
    GetOutputCursorPosition( &x, &y );
    if (row) *row = (int)y; if (col) *col = (int)x;
    return 0;
    #elif (CLIENT_OS == OS_DOS)
    return dosCliConGetPos( col, row );
    #elif (CLIENT_OS == OS_OS2)
    return ((VioGetCurPos( (USHORT*)&row, (USHORT*)&col,
                 0 /*handle*/) != 0)?(-1):(0));
    #else
    return ((row == NULL && col == NULL) ? (0) : (-1));
    #endif
    }
  return -1;
}

/* ---------------------------------------------------- */

int ConSetPos( int col, int row )  /* zero-based */
{
  if (constatics.initlevel > 0 && constatics.conisatty)
    {
    #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
    return w32ConSetPos(col,row);
    #elif (CLIENT_OS == OS_NETWARE)
    short c = col, r = row;
    gotoxy( c, r );
    return 0;
    #elif (CLIENT_OS == OS_DOS)
    return dosCliConSetPos( col, row );
    #elif (CLIENT_OS == OS_OS2)
    return ((VioSetCurPos(row, col, 0 /*handle*/) != 0)?(-1):(0));
    return 0;
    #else
    return -1;
    #endif
    }
  return -1;
}

/* ---------------------------------------------------- */

int ConGetSize(int *widthP, int *heightP) /* one-based */
{
  int width = 0, height = 0;

  if (constatics.initlevel <= 0 || !constatics.conisatty)
    return -1;

  #if (CLIENT_OS == OS_RISCOS)
    {
    static const int var[3] = { 133, 135, -1 };
    int value[3];
    if (!riscos_in_taskwindow)
      {
      if (_swix(OS_ReadVduVariables, _INR(0,1), var, value) == 0)
        {
        // nlines = TWBRow - TWTRow + 1
        height = value[0] - value[1] + 1;
        }
      }
    }
  #elif (CLIENT_OS == OS_DOS)
    if (dosCliConGetSize( &width, &height ) < 0)
      height = width = 0;
  #elif (CLIENT_OS == OS_WIN32)
    if ( w32ConGetSize(&width,&height) < 0 )
      height = width = 0;
  #elif (CLIENT_OS == OS_NETWARE)
    WORD ht, wdth;
    GetSizeOfScreen( &ht, &wt );
    height = ht; width = wt;
  #else
    {
    #if 0 //-- no longer needed since paging is disabled on unix targets
    // grrr... terminfo database location is installation dependent
    // search some standard (?) locations
    static int slines=0, scolumns=0;

    if (slines == 0 && scolumns == 0)
      {
      unsigned int loc = 0;
      int success = 0;

      while (!success)
        {
        int termerr;
        if (setupterm( NULL, 1, &termerr ) != ERR)
          {
          if (termerr == 1)
            {
            success = 1;
            break;
            }
          }
        char *terminfo_locations[] =
            {
          "/usr/share/terminfo",       // ncurses 1.9.9g defaults
          "/usr/local/share/terminfo", //
          "/usr/lib/terminfo",         // Debian 1.3x use this one
	  "/usr/share/lib/terminfo",   // ex. AIX (has a link to /usr/lib)
          "/usr/local/lib/terminfo",   // variation
          "/etc/terminfo",             // found something here on my machine, doesn't hurt
          "~/.terminfo",               // last resort
          NULL                         // stop tag
            };
        if (terminfo_locations[loc] == NULL)
          break;
        setenv( "TERMINFO", terminfo_locations[loc], 1);
        loc++;
        }
      if (success)
        {
        slines = tigetnum( "lines" );
        scolums = tigetnum( "columns" );
        }
      }
    height = slines; width = scolumns;
    #endif
    }
  #endif

  if (height == 0)
    {
    char *p = getenv( "LINES" );
    if (p) height = atoi( p );
    }
  if (width == 0)
    {
    char *p = getenv( "COLUMNS" );
    if (p) width = atoi( p );
    }
  if (height <= 0 || height >= 300)
    height = 25;
  if (width <=0 || width >=300)
    width = 80;

  if (heightP) *heightP = height;
  if (widthP)  *widthP = width;
  return 0;
}

/* ---------------------------------------------------- */

/*
** ConClear() clears the screen.
** Returns -1 if err
*/

int ConClear(void)
{
  if (constatics.initlevel > 0 && constatics.conisatty)
    {
    #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
      return w32ConClear();
    #elif (CLIENT_OS == OS_OS2)
      USHORT attrib = 7;
      VioScrollUp(0, 0, (USHORT)-1, (USHORT)-1, (USHORT)-1, (PCH)&attrib, 0);
      VioSetCurPos(0, 0, 0);      /* move cursor to upper left */
      return 0;
    #elif (CLIENT_OS == OS_DOS)
      return dosCliConClear();
    #elif (CLIENT_OS == OS_NETWARE)
      clrscr();
      return 0;
    #elif (CLIENT_OS == OS_RISCOS)
      riscos_clear_screen();
      return 0;
    #elif defined(TERM_IS_ANSI_COMPLIANT)
      printf("\x1B" "[2J" "\x1B" "[H" "\r       \r" );
      /* ANSI cls  '\r space \r' is in case ansi is not supported */
      return 0;
    #endif
    }
  return -1;
}

/* ---------------------------------------------------- */

