// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
/*
   This module contains the screen i/o primitives/wrappers for all
   those GUIs and less-than-G UIs we have been threatened with :)
   and we pretty much have scattered about anyway.
   
   Implementation guidelines: none. see what the neighbour did.  
   Keep the functions small (total less than 25 lines) or make calls
   to functions in modules in your own platform area. 
*/
// $Log: console.cpp,v $
// Revision 1.18  1998/11/09 17:24:35  chrisb
// Added riscos_backspace() to work round some lame implementation of consoles.
//
// Revision 1.17  1998/11/09 16:54:04  cyp
// Added special bksp handling for ANSI compliant terms.
//
// Revision 1.16  1998/11/08 22:24:15  foxyloxy
// Really did the below comment this time.
//
// Revision 1.15  1998/11/08 22:16:19  foxyloxy
//
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
// Various quick hacks to make the win32gui operational again (will be cleaned up).
//
// Revision 1.6  1998/10/07 18:36:18  silby
// Changed logic in ConInKey once more so it's not reading uninit variables.  Should be solid now. :)
//
// Revision 1.5  1998/10/07 12:56:46  silby
// Reordered Deinitconsole so console functions would still be available during w32deinitconsole.
//
// Revision 1.4  1998/10/07 12:25:04  silby
// Figured out that MSVC doesn't understand continue as it was used; changed ConInKey's loop so that it doesn't rely on continue.  (Functionality unchanged.)
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
return "@(#)$Id: console.cpp,v 1.18 1998/11/09 17:24:35 chrisb Exp $"; }
#endif

#define CONCLOSE_DELAY 15 /* secs to wait for keypress when not auto-close */

#include "cputypes.h"
#include "baseincs.h"
#include "network.h"
#include "clitime.h"
#include "triggers.h"
#include "console.h" //also has CLICONS_SHORTNAME, CLICONS_LONGNAME
#include "sleepdef.h" //usleep

#if !defined(NOTERMIOS) && ((CLIENT_OS==OS_SOLARIS) || (CLIENT_OS==OS_IRIX) || \
    (CLIENT_OS==OS_LINUX) || (CLIENT_OS==OS_NETBSD) || (CLIENT_OS==OS_BEOS))
#include <termios.h>
#define TERMIOS_IS_AVAILABLE
#endif

#if (CLIENT_OS==OS_VMS) || (CLIENT_OS==OS_DEC_UNIX) || (CLIENT_OS==OS_OS9) || \
    (CLIENT_OS==OS_HPUX) || (CLIENT_OS==OS_QNX) || (CLIENT_OS==OS_OSF1) || \
    (CLIENT_OS==OS_BSDI) || (CLIENT_OS==OS_SOLARIS) || (CLIENT_OS==OS_IRIX) || \
    (CLIENT_OS==OS_LINUX) || (CLIENT_OS==OS_NETBSD) || (CLIENT_OS==OS_BEOS) || \
    (CLIENT_OS==OS_SCO) || (CLIENT_OS==OS_UNIXWARE) || (CLIENT_OS==OS_MVS) || \
    (CLIENT_OS==OS_MINIX) || (CLIENT_OS==OS_MACH10) || (CLIENT_OS==OS_AIX) || \
    (CLIENT_OS==OS_AUX) || (CLIENT_OS==OS_OPENBSD) || (CLIENT_OS==OS_SUNOS) || \
    (CLIENT_OS==OS_ULTRIX) || (CLIENT_OS==OS_DGUX) || (CLIENT_OS==OS_DYNIX) || \
    (CLIENT_OS==OS_OS390)
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
} constatics = {0,0,0};

/* ---------------------------------------------------- */

int DeinitializeConsole(int autoclose)
{
  if (constatics.initlevel == 1)
    {
    if (!autoclose && !constatics.runhidden)
      {
      #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || \
          (CLIENT_OS==OS_WIN32S)
        {
        int init = 0;
        time_t endtime = (CliTimer(NULL)->tv_sec) + CONCLOSE_DELAY;
        do{
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
    #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
      w32DeinitializeConsole();
    #endif
    } 

  constatics.initlevel--;

  return 0;
}  

/* ---------------------------------------------------- */

int InitializeConsole(int runhidden)
{
  int retcode = 0;

  if ((++constatics.initlevel) == 1) 
    {
    memset( (void *)&constatics, 0, sizeof(constatics) );
    constatics.initlevel = 1;
    constatics.runhidden = runhidden;

    #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
    retcode = w32InitializeConsole(runhidden);
    #endif

    if (retcode != 0)
      --constatics.initlevel;
    else if (!runhidden)
      {
      #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
        constatics.conisatty = w32ConIsScreen();
      #elif (CLIENT_OS == OS_RISCOS)
        constatics.conisatty = 1;
      #else
        constatics.conisatty = ((isatty(fileno(stdout))) && 
          (isatty(fileno(stdin))));
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
** ConOut() does what printf("%s",str) would do 
*/ 
int ConOut(const char *msg)
{
  if (constatics.initlevel > 0 /*&& constatics.conisatty*/ )
    {
    #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
      w32ConOut(msg);
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
  #elif (CLIENT_OS == OS_OS2)
    WinMessageBox( HWND_DESKTOP, HWND_DESKTOP, msg, (PSZ)NULL,
       CLICONS_LONGNAME, MB_OK | MB_INFORMATION | MB_MOVEABLE );
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
  #elif (CLIENT_OS == OS_OS2)
     WinMessageBox( HWND_DESKTOP, HWND_DESKTOP, msg, (PSZ)NULL,
           CLICONS_LONGNAME, MB_OK | MB_APPLMODAL | MB_ERROR | MB_MOVEABLE );
  #elif (CLIENT_OS == OS_NETWARE)
    ConsolePrintf( "%s: %s\r\n", CLICONS_SHORTNAME, msg );
  #else
    fprintf( stderr, "%s: %s\n", CLICONS_SHORTNAME, msg );
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
    
    do{
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
         (CLIENT_OS == OS_OS2)
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
        {
    unsigned long keys[4];
    GetKeys(keys);
    if (keys[0] != 0 || keys[1] != 0 || keys[2] != 0 || keys[3] != 0)
      {
        ch = getchar();
        if (ch == EOF)
          ch = 0;
      }
        }
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
      usleep(50*1000); /* with a 50ms delay, no visible processor activity */
                       /* with NT4/P200 and still responsive to user requests */

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
  int ch, exitreq;
  unsigned int pos;

  if (constatics.initlevel < 1 || !constatics.conisatty)
    return -1;

  if (!buffer || !buflen)
    return 0;

  if ((flags & CONINSTR_ASPASSWORD)!=0)
    flags = CONINSTR_ASPASSWORD;

  if ((flags & CONINSTR_BYEXAMPLE) != 0)
    {
    ConOut(buffer);
    pos = strlen( buffer );
    }
  else
    {
    pos = 0;
    buffer[pos] = 0;
    }

  do {
     ch = ConInKey(-1);
     exitreq = CheckExitRequestTriggerNoIO();

     if (!exitreq)
       {
//printf("***%d***\n",ch);
       if (ch == 0x08 || ch == '\177') /* backspace */
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
       else if (ch == '\n' || ch == '\r') 
         {
         ConOut("\n");
         exitreq = 1;
         }
       else if (pos < (buflen-1)) 
         {
         buffer[pos++] = ch;
         if ((flags & CONINSTR_ASPASSWORD) != 0)
           ch = '*';
         if (isalpha(ch) || isspace(ch) || isdigit(ch) || ispunct(ch))
           {
           /* if (!isctrl(ch)) */
           char x[2];
           x[0]=ch; x[1]=0;
           ConOut(x);
           }
         }
       }
     } while (!exitreq);

   ConOut(""); /* flush */
   buffer[pos] = 0;

   return strlen(buffer);
}

#if 0
int ConInStr(char *buffer, unsigned int buflen )
{
  char buff[256];
  unsigned int len;

  if (constatics.initlevel < 1 || !constatics.conisatty)
    return -1;

  if (!buffer || !buflen)
    return 0;

  len = 0;
  *buffer = 0;

  if ( fgets( buff, sizeof(buff), stdin ) != NULL )
    {
    for (len = 0; buff[len]!=0; len++ )
      {
      if ( iscntrl( buff[len] ) )
        {
        buff[len]=0;
        break;
        }
      }
    len = strlen( buff );
    if (len > buflen)
      {
      len = buflen;
      buff[len-1] = 0;
      }
    if (len > 0)
      strcpy( buffer, buff );
    }
  return len;
}  
#endif

/* ---------------------------------------------------- */

#if 0 //unimplemented
int ConGetPos( int *col, int *row )  /* zero-based */
{
  if (constatics.initlevel > 0 && constatics.conisatty)
    {
    #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
    w32ConGetPos(col,row);
    #elif (CLIENT_OS == OS_NETWARE)
    short x, y;
    GetOutputCursorPosition( &x, &y );
    row = (int)y; col = (int)x;
    #endif
    return 0;
    }
  return -1;
}
#endif

/* ---------------------------------------------------- */

#if 0 //unimplemented
int ConSetPos( int col, int row )  /* zero-based */
{
  if (constatics.initlevel > 0 && constatics.conisatty)
    {
    #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
    w32ConSetPos(col,row);
    #elif (CLIENT_OS == OS_NETWARE)
    WORD c = col, r = row;
    gotoxy( c, r );
    #endif
    return 0;
    }
  return -1;
}
#endif

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
      
      while (success == 0)
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
      if (w32ClearScreen() != 0)
        return -1;
    #elif (CLIENT_OS == OS_OS2)
      BYTE space[] = " ";
      VioScrollUp(0, 0, -1, -1, -1, space, 0);
      VioSetCurPos(0, 0, 0);      /* move cursor to upper left */
    #elif (CLIENT_OS == OS_DOS)
      dosCliClearScreen(); /* in platform/dos/clidos.cpp */
    #elif (CLIENT_OS == OS_NETWARE)
      clrscr();
    #elif (CLIENT_OS == OS_RISCOS)
      riscos_clear_screen();
    #else
      printf("\x1B" "[2J" "\x1B" "[H" "\r       \r" );
      /* ANSI cls  '\r space \r' is in case ansi is not supported */
    #endif
    return 0;
    }
  return -1;
}

/* ---------------------------------------------------- */

