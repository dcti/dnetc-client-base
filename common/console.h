// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: console.h,v $
// Revision 1.25  1999/01/07 02:15:57  cyp
// ConInStr() now has a special 'boolean' mode. woohoo!
//
// Revision 1.24  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.23  1998/11/10 21:36:01  cyp
// Changed InitializeConsole() so that terms know in advance whether the
// client will be running "modes" or not. This is needed for platforms where
// the client uses a different screen for "modes" or for others that wait
// with a "Press any key..." message before destroying the screen/window.
//
// Revision 1.22  1998/11/08 19:05:03  cyp
// Created new function ConGetSize(int *width, int *height) from stuff in
// DisplayHelp().
//
// Revision 1.21  1998/10/26 02:52:46  cyp
// Remved IS_A_TTY() macros.
//
// Revision 1.2  1998/10/11 05:24:31  cyp
// Implemented ConIsScreen(): a real (not a macro) isatty wrapper.
//
// Revision 1.1  1998/10/03 05:34:47  cyp
// Created.

#ifndef __CONSOLE_H__
#define __CONSOLE_H__

#include "version.h"
#define CLICONS_SHORTNAME  "RC5DES"
#define CLICONS_LONGNAME "Distributed.Net RC5/DES Client " CLIENT_VERSIONSTRING ""

// ConIsScreen() returns true (!0) if console (both stdin and stdout) 
// represents the screen. also returns 0 if the console is not initialized.
int ConIsScreen(void);

// ConOut() does what printf("%s",str) would do 
// writes only if stdout is a tty. (or equivalent)
int ConOut(const char *str);

// ConOutErr() does what fprintf(stderr "\nRC5DES: %s\n",msg) would do.
// Can be blocking. Note the leading and trailing newlines. 
int ConOutErr(const char *msg); //Can be used at any time. Always succeeds.

// ConOutModal() should only be used when the console is known to be 
// uninitialized. Can be blocking. Not affected by -hidden/-quiet mode
int ConOutModal(const char *str); //currently no use for it.

// ConInKey() does what a (non-blocking and polling) DOS-ish getch() would 
// do key is not echoed. timeout ==> 0 == don't wait, -1 == wait forever. 
int ConInKey(int timeout_millisecs); // Returns -1 if err. 0 if timed out.

// ConInStr() does what gets() would do (without the trailing '\n') and the
// buffer is always '\0' terminated. Returns -1 if console is not a tty
int ConInStr(char *buffer, unsigned int len, int flags );
#define CONINSTR_BYEXAMPLE  0x01  /* the buffer contains a 'live' example */
#define CONINSTR_ASPASSWORD 0x02  /* print '*' for each character typed */
#define CONINSTR_ASBOOLEAN  0x04  /* get 'y' or 'n' */

// ConClear() clears the screen. 
// returns -1 if console is not a tty;
int ConClear(void);

// ConGetPos gets the cursor position (zero-based)
// returns -1 if console is not a tty;
int ConGetPos( int *row, int *col );

// Set the cursor position (zero-based)
// returns -1 if console is not a tty
int ConSetPos( int row, int col );  

// Get screen size (one-based)
// returns -1 if console is not a tty 
int ConGetSize( int *width, int *height );

// Deinitialize console functionality. 
int DeinitializeConsole(void);

// Initialize console functionality. Returns !0 on failure.
// doingmodes is used on some platforms to use a separate screen and by
// others to wait with "Press any key..." before destroying the screen
int InitializeConsole(int runhidden, int doingmodes);

#endif
