/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/ 

#ifndef __W32UTIL_H__
#define __W32UTIL_H__ "@(#)$Id: w32util.h,v 1.1.2.1 2001/01/21 15:10:26 cyp Exp $"

#ifdef _INC_WINDOWS
/* ScreenSaver boot vector (initialized from w32ss.cpp if linked) */
extern int (PASCAL *__SSMAIN)(HINSTANCE,HINSTANCE,LPSTR,int);
#endif

/* get DOS style version: (major*100)+minor. major is >=20 if NT */
extern long winGetVersion(void); 

#endif /* __W32UTIL_H__ */
