/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __NEXT_SUP_H__
#define __NEXT_SUP_H__

#include <sys/types.h> /* size_t, off_t */
#include <sys/wait.h>  /* union wait, WIFEXITED */

char *strdup(const char *);
int setenv (const char *,const char *, int);

#define WIFTERMSIG(x)  ((x).w_termsig)
#define WEXITSTATUS(x) ((x).w_retcode)
#define WSTOPSIG(x)    ((x).w_stopsig)

int waitpid(int, int *, int);

#endif /* __NEXT_SUP_H__ */
