/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * This module contains the screen i/o primitives/wrappers for all
 * those GUIs and less-than-G UIs we have been threatened with :)
 * and we pretty much have scattered about anyway.
 *
 * Implementation guidelines: none. see what the neighbour did.
 * Keep the functions small (total less than 25 lines) or make calls
 * to functions in modules in your own platform area.   - cyp
 * ----------------------------------------------------------------------
*/
const char *console_cpp(void) {
return "@(#)$Id: console.cpp,v 1.45.2.3 1999/04/24 07:34:59 jlawson Exp $"; }

/* -------------------------------------------------------------------- */


#include "cputypes.h"
#include "baseincs.h"
#include "clitime.h"
#include "triggers.h"
#include "console.h" //also has CLICONS_SHORTNAME, CLICONS_LONGNAME
#include "modereq.h"
#include "sleepdef.h" //usleep
#if (CLIENT_OS==OS_AIX)
#include <sys/select.h>   // only needed if compiled on AIX 4.1
#endif

#define CONCLOSE_DELAY 15 /* secs to wait for keypress when not auto-close */
#if !defined(NOTERMIOS) && ((CLIENT_OS==OS_SOLARIS) || (CLIENT_OS==OS_IRIX) || \
    (CLIENT_OS==OS_LINUX) || (CLIENT_OS==OS_NETBSD) || (CLIENT_OS==OS_BEOS) \
    || (CLIENT_OS==OS_FREEBSD) || defined(__EMX__) || (CLIENT_OS==OS_AIX) \
    || (CLIENT_OS==OS_DEC_UNIX) || (CLIENT_OS==BSDI) \
  || (CLIENT_OS==OS_OPENBSD) || (CLIENT_OS==OS_HPUX) )
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
  int pauseonclose;
} constatics = {0,0,0,0};

/* ---------------------------------------------------- */

int DeinitializeConsole(void)
{
  if (constatics.initlevel == 1)
  {
    if (constatics.pauseonclose && !constatics.runhidden)
    {
      #if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S) || \
          ((CLIENT_OS == OS_WIN32) && (!defined(WIN32GUI))) || \
          ((CLIENT_OS == OS_OS2) && !defined (__EMX__)) || \
          (CLIENT_OS == OS_NETWARE)
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
    constatics.pauseonclose = (doingmodes && ModeReqIsSet(MODEREQ_CONFIG)==0);

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
      #elif defined(OS2_PM)
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
    #elif (CLIENT_OS == OS_OS2 && defined(OS2_PM))
      os2conout(msg);
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
       CLICONS_LONGNAME, NULL, MB_OK | MB_INFORMATION | MB_MOVEABLE );
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
           CLICONS_LONGNAME,  NULL, MB_OK | MB_APPLMODAL | MB_ERROR | MB_MOVEABLE );
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
        char x[2];
        buffer[pos++] = (char)ch;
        if (!(isalpha(ch) || isspace(ch) || isdigit(ch) || ispunct(ch)))
          ch = '?';
        else if ((flags & CONINSTR_ASPASSWORD) != 0)
          ch = '*';
        x[0]=(char)ch;
        x[1]=0;
        ConOut(x);
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
    char *envp = getenv( "LINES" );
    if (envp != NULL)
    {
      height = atoi(envp);
      if (height <= 0 || height >= 300)
        height = 0;
    }
    envp = getenv( "COLUMNS" );
    if (envp != NULL)
    {
      width = atoi(envp);
      if (width <= 0 || width >= 300)
        width = 0;
    }
    #if 0 //-- no longer needed since paging is disabled on unix targets
    if (height == 0 && width == 0)
    {
      unsigned int loc = 0;
      char *terminfo_locations[] =
      {
        // grrr... terminfo database location is installation dependent
        // search some standard (!?) locations
        "/usr/share/terminfo",       // ncurses 1.9.9g defaults
        "/usr/local/share/terminfo", //
        "/usr/lib/terminfo",         // Debian 1.3x use this one
        "/usr/share/lib/terminfo",   // ex. AIX (has a link to /usr/lib)
        "/usr/local/lib/terminfo",   // linux variation
        "/etc/terminfo",             // another linux variation
        "~/.terminfo",               // last resort
        NULL                         // stop tag
      };
      envp = NULL;
      do
      {
        int termerr;
        if ( envp )
          setenv( "TERMINFO", envp, 1);
        if (setupterm( NULL, 1, &termerr ) != ERR)
        {
          if (termerr == 1)
          {
            char buf[sizeof(int)*3];
            height = tigetnum( "lines" );
            if (height <= 0 || height >= 300)
              height = 0;
            else
            {
              sprintf( buf, "%d", height );
              setenv( "LINES", buf , 1);
            }
            width = tigetnum( "columns" );
            if (width <= 0 || width >= 300)
              width = 0;
            else
            {
              sprintf( buf, "%d", width );
              setenv( "COLUMNS", buf , 1);
            }
            if (width != 0 && height != 0)
              break;
          }
        }
        envp = terminfo_locations[loc++];
      } while (envp != NULL);
    }
    #endif
  }
  #endif

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

