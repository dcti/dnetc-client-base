/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/ 

#ifndef __W32UTIL_H__
#define __W32UTIL_H__ "@(#)$Id: w32util.h,v 1.1.2.3 2002/03/28 01:07:48 andreasb Exp $"

#ifdef _INC_WINDOWS
/* ScreenSaver boot vector (initialized from w32ss.cpp if linked) */
extern int (PASCAL *__SSMAIN)(HINSTANCE,HINSTANCE,LPSTR,int);
#endif

/* get DOS style version: (major*100)+minor. major is >=20 if NT */
extern long winGetVersion(void);

/* Table of Windows OSes and their version numbers returned by 
   GetVersionEx() and winGetVersion():

OS              dwMajorVersion.dwMinorVersion   winGetVersion()
Windows 3.x                  ?.?                 3?? ???
Windows 95                   4.0                 400
Windows 98                   4.10                410
Windows Me                   4.90                490
Windows NT 3.51              3.51               2351
Windows NT 4.0               4.0                2400
Windows 2000                 5.0                2500
Windows XP                   5.1                2510
Windows .NET Server          5.1                2510 ???

*/

#endif /* __W32UTIL_H__ */
