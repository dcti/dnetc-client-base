/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/ 

#ifndef __W32PRE_H__
#define __W32PRE_H__ "@(#)$Id: w32pre.h,v 1.1.2.1 2001/01/21 15:10:25 cyp Exp $"

int winGetMyModuleFilename(char *buffer,unsigned int len);

int winGetInstanceShowCmd(void); /* get the nCmdShow */
int winGetInstanceArgc(void);    /* argc */
char **winGetInstanceArgv(void); /* argv */

#ifdef _INC_WINDOWS
HINSTANCE winGetInstanceHandle(void); /* get the client's instance handle */
HWND winGetParentWindow(void);
void winSetSSMainVector(int (PASCAL *ssmain)(HINSTANCE,HINSTANCE,LPSTR,int));

/* this is the abstraction layer between WinMain() and realmain() */
int winClientPrelude(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    LPSTR lpszCmdLine, int nCmdShow, int (*realmain)(int argc, char **argv));
#endif

#endif /* __W32PRE_H__ */
