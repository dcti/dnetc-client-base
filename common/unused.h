/*
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __UNUSED_H__
#define __UNUSED_H__  "@(#)$Id: unused.h,v 1.1.2.2 2003/09/02 00:48:54 mweiser Exp $"

/* This file is a bit empty at the moment because it's meant to be
** extended for all the compilers encountered by this code. If it
** doesn't work for you, put an
**
** #ifdef (compiler)
**
** but please not based on OS/platform around this and add #elif's as
** necessary.
**
** #else */

/* gcc's __attribute__((unused)) only works for C and not C++. So we
** have to use a useless/non-effective reference to make the compiler
** believe we know what we're doing. There are several ways to do so:
** A bad way is
**
** #define DNETC_UNUSED_PARAM(x) (x = x)       - bad!
**
** because it will break on C++ references and call the = operator
** on classes which might have undesired side-effects. Better is
**
** #define DNETC_UNUSED_PARAM(x) {if (&x) {}}
**
** but might not get optimised away by the compiler. Third there's
** an unassigned rvalue like
**
** #define DNETC_UNUSED_PARAM ((void)x)
**
** which might still cause senselessness warnings on some compilers
** though. We choose the last option for now. */

#define DNETC_UNUSED_PARAM(x)   ((void)x)

/* all occurences of DNETC_UNUSED should actually be fixed to not have
** those variables in the first place */
#define DNETC_UNUSED(x)         DNETC_UNUSED_PARAM(x)

/* #endif */

#endif /* __UNUSED_H__ */
