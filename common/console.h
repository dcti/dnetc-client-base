// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: console.h,v $
// Revision 1.2  1998/10/11 05:24:31  cyp
// Implemented ConIsScreen(): a real (not a macro) isatty wrapper.
//
// Revision 1.1  1998/10/03 05:34:47  cyp
// Created.
//
//

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

// ConInStr() does what gets() would do (without the trailing '\n') and
// buffer is always '\0' terminated. Returns -1 if err (stdin is not a tty)
int ConInStr(char *buffer, unsigned int len, int flags );
#define CONINSTR_BYEXAMPLE  1
#define CONINSTR_ASPASSWORD 2

// ConClear() clears the screen. 
// returns -1 if console is not a tty;
int ConClear(void);

// ConGetPos gets the cursor position (zero-based)
// returns -1 if console is not a tty;
int ConGetPos( int *row, int *col );

//set the cursor position (zero-based)
// returns -1 if console is not a tty
int ConSetPos( int row, int col );  

// Deinitialize and Initialize console functionality.
// returns -1 if console could not be initialized
int DeinitializeConsole(void);
int InitializeConsole(int runhidden);

#define IS_STDOUT_A_TTY() (ConIsScreen())
#define IS_STDIN_A_TTY() (ConIsScreen())

#endif
