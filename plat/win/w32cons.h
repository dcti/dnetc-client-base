/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/ 

#ifndef __W32CONS_H__
#define __W32CONS_H__ "@(#)$Id: w32cons.h,v 1.1.2.1 2001/01/21 15:10:25 cyp Exp $"

/* ********* DO NOT EVER CHANGE THE W32CLI_*_NAME defines ********** */
#define W32CLI_MUTEX_NAME       "Bovine RC5/DES Win32 Client"
#define W32CLI_CONSOLE_NAME     "distributed.net client"
#define W32CLI_OLD_CONSOLE_NAME "distributed.net RC5DES client"
#define W32CLI_SSATOM_NAME      "distributed.net ScreenSaver"

/* 
   DNETC_WCMD_* are public identifiers used by other clients 
   to control us. Do not change number or meaning.
*/   
#define DNETC_WCMD_ACKMAGIC         ((LRESULT)-12345)
#define DNETC_WCMD_EXISTCHECK       0 /* are any running?, used internally */
#define DNETC_WCMD_SHUTDOWN         1
#define DNETC_WCMD_RESTART          2
#define DNETC_WCMD_PAUSE            3
#define DNETC_WCMD_UNPAUSE         13
#define DNETC_WCMD_INTERNAL_FIRST 512 /* no more public cmds from here on! */

/* send one of the WCMD commands above. Returns < 0 if failed. */
extern int w32PostRemoteWCMD( int cmd );

#ifndef SSSTANDALONE

int w32InitializeConsole(int runhidden, int runmodes);
int w32DeinitializeConsole(int pauseonclose);

/* clear window - returns !0 if error */
int w32ConClear(void);

/* activate client window if not -hidden */
int w32ConShowWindow(void); 

/* getch() */
int w32ConGetch(void);

/* kbhit() */
int w32ConKbhit(void);

/* print a string to the console */
int w32ConOut(const char *text);

/* print a string in "APPNAME: xxxx\n" format if tty, else MessageBox */
int w32ConOutErr(const char *text);

/* print a string in "xxxx\n" format if tty, else MessageBox */
int w32ConOutModal(const char *text);

/* pump waiting messages */
void w32Yield(void);

/* pump waiting messages for x millsecs */
void w32Sleep(unsigned int millisecs);

/* does console refer to the screen */
int w32ConIsScreen(void);

/* get size of "console" window (one-based) */
int w32ConGetSize( int *width, int *height);

/* setpos (zerobased) */
int w32ConSetPos( int col, int row);

/* getpos (zerobased) */
int w32ConGetPos( int *col, int *row);

/* LOBYTE => 'C'=native console, 'c'=pipe console, 'g'=lite GUI, 'G'=fat GUI */
/* HIBYTE => 't' = in tray, 'm' = minimized, 'h' = hidden, '\0' = normal */
int w32ConGetType(void);

#endif /* SSSTANDALONE */

#endif /* __W32CONS_H__ */
