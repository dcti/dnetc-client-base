// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
/*
   This module contains the screen i/o primitives/wrappers for all
   those GUIs and less-than-G UIs we have been threatened with :)
   and we pretty much have scattered about anyway.
   
   Implementation guidelines: none. see what the neighbour did.  
   Keep the functions small (total less than 25 lines) or make calls
   to functions in modules in your own platform/ area. 
*/
// $Log: console.cpp,v $
// Revision 1.1  1998/10/03 05:34:45  cyp
// Created.
//
//
#if (!defined(lint) && defined(__showids__))
const char *console_cpp(void) {
return "@(#)$Id: console.cpp,v 1.1 1998/10/03 05:34:45 cyp Exp $"; }
#endif

#include "cputypes.h"
#include "baseincs.h"
#include "network.h"
#include "clitime.h"
#include "triggers.h"
#include "console.h" //also has CLICONS_SHORTNAME, CLICONS_LONGNAME
#include "sleepdef.h" //usleep
#if (CLIENT_OS==OS_LINUX) || (CLIENT_OS==OS_NETBSD) || (CLIENT_OS==OS_BEOS)
#include <termios.h>
#define USE_TERMIOS_FOR_INKEY
#endif

/* ---------------------------------------------------- */

static struct 
{
  int initlevel;
  int runhidden;
  int conisatty;
} constatics = {0,0,0};

/* ---------------------------------------------------- */

int DeinitializeConsole(void)
{
  constatics.initlevel--;

  if (constatics.initlevel == 0)
    {
    #if (CLIENT_OS == OS_WIN32)
      w32DeinitializeConsole();
    #endif
    } /* constatics.initlevel == 0 */

  return 0;
}  

/* ---------------------------------------------------- */

int InitializeConsole(int runhidden)
{
  int retcode = 0;

  constatics.initlevel++;
  if (constatics.initlevel == 1) 
    {
    memset( (void *)&constatics, 0, sizeof(constatics) );
    constatics.initlevel = 1;
    constatics.runhidden = runhidden;

    #if (CLIENT_OS == OS_WIN32)
    retcode = w32InitializeConsole(runhidden);
    #endif

    if (retcode != 0)
      DeinitializeConsole(); /* decrement constatics.initlevel */
    else if (!runhidden)
      constatics.conisatty = (IS_STDOUT_A_TTY() && IS_STDIN_A_TTY());
    } /* constatics.initlevel == 1 */
    
  return retcode;
}
  
/* ---------------------------------------------------- */

/* 
** ConOut() does what printf("%s",str) would do 
** writes only if stdout is a tty. (or equivalent)
*/ 

int ConOut(const char *msg)
{
  if (constatics.initlevel > 0 && constatics.conisatty )
    {
    fwrite( msg, sizeof(char), strlen(msg), stdout);
    fflush(stdout);
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
  #if (CLIENT_OS == OS_WIN32)
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
  #if (CLIENT_OS == OS_WIN32)
    MessageBox( NULL, msg, CLICONS_LONGNAME, 
                 MB_OK | MB_TASKMODAL | MB_ICONERROR );
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
      #elif (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_NETWARE) || \
         (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) || \
        (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
        {
        if (kbhit())
          {
          ch = getch();
          if (!ch)
            ch = (getch() << 8);
          }
        }
      #elif (defined(USE_TERMIOS_FOR_INKEY))
        {
        struct termios stored;
        struct termios newios;

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
      if (timeout_millisecs < 0)
        continue;
      
      CliTimer(&timenow);
      } while (( timenow.tv_sec > timestop.tv_sec ) ||
                 (( timenow.tv_sec == timestop.tv_sec ) &&
                  ( timenow.tv_usec > timestop.tv_usec )));
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

  if ((flags & (CONINSTR_BYEXAMPLE|CONINSTR_ASPASSWORD)) == CONINSTR_BYEXAMPLE)
    {
    printf("%s",buffer);
    pos = strlen( buffer );
    }
  else
    {
    pos = 0;
    buffer[pos] = 0;
    }

  do {
     fflush(stdout);
     ch = ConInKey(-1);
     exitreq = CheckExitRequestTriggerNoIO();

     if (!exitreq)
       {
       if (ch == 0x08 || ch == '\177') /* backspace */
         {
         if (pos > 0)  
           {
           putchar('\b');
           putchar(' ');
           putchar('\b');
           pos--;
           }
         }
       else if (ch == '\n' || ch == '\r') 
         {
         putchar('\n');
         exitreq = 1;
         }
       else if (pos < (buflen-1)) 
         {
         buffer[pos++] = ch;
         if ((flags & CONINSTR_ASPASSWORD) != 0)
           ch = '*';
         if (isalpha(ch) || isspace(ch) || isdigit(ch) || ispunct(ch))
         /* if (!isctrl(ch)) */
           putchar(ch);
         }
       }
     } while (!exitreq);

   fflush(stdout);
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

#if 0  /* unimplemented */
int ConGetPos( int *row, int *col )  /* zero-based */
{
  if (constatics.initlevel > 0 && constatics.conisatty)
    {
    #if (CLIENT_OS == OS_NETWARE)
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

#if 0 /* unimplemented */
int ConSetPos( int row, int col )  /* zero-based */
{
  if (constatics.initlevel > 0 && constatics.conisatty)
    {
    #if (CLIENT_OS == OS_WIN32)
    COORD newpos = {col,row};
    SetConsoleCursorPosition( GetStdHandle(STD_OUTPUT_HANDLE), newpos );
    #elif (CLIENT_OS == OS_NETWARE)
    gotoxy( (short)col, (short)row );
    #endif
    return 0;
    }
  return -1;
}
#endif

/* ---------------------------------------------------- */

/* 
** ConClear() clears the screen. 
** Returns -1 if err
*/

int ConClear(void)
{
  if (constatics.initlevel > 0 && constatics.conisatty)
    {
    #if (CLIENT_OS == OS_WIN32)
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

