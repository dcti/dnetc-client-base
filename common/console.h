/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __CONSOLE_H__
#define __CONSOLE_H__ "@(#)$Id: console.h,v 1.28.2.3 1999/11/02 16:03:42 cyp Exp $"

// ConIsScreen() returns non-zero if console represents the screen
// also returns 0 if the console is not initialized.
int ConIsScreen(void);

// Are we running under a non-command line oriented UI? 
// (for cases where GUIishness will have been determined at runtime)
int ConIsGUI(void);

// ConOut() does what printf("%s",str) would do
// writes only if stdout is a tty. (or equivalent)
int ConOut(const char *str);

// ConOutErr() does what fprintf(stderr "\nAPPNAME: %s\n",msg) would do.
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

//ConBeep() does uh..., well..., like... causes the console speaker to beep
// returns -1 if console is not a tty;
int ConBeep(void);

// ConGetPos gets the cursor position (zero-based)
// returns -1 if console is not a tty;
int ConGetPos( int *row, int *col );

// Set the cursor position (zero-based)
// returns -1 if console is not a tty
int ConSetPos( int row, int col );

// Get screen size (one-based)
// returns -1 if console is not a tty
int ConGetSize( int *width, int *height );

// Deinitialize console functionality. 'waitforuser' is set if the client
// ran modes, so that the user can see the output before the screen disappears
int DeinitializeConsole(int waitforuser);

// Initialize console functionality. Returns !0 on failure.
// doingmodes is used on some platforms to use a separate screen.
int InitializeConsole(int runhidden, int doingmodes);

#endif
