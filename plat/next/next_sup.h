/* $Id: next_sup.h,v 1.1.2.2 2003/08/09 12:21:41 mweiser Exp $ */

#ifndef __NEXTSTEP_SUP_H__
#define __NEXTSTEP_SUP_H__

#include <sys/types.h> /* size_t, off_t */
#include <sys/wait.h>  /* union wait, WIFEXITED */

char *strdup(const char *);
int setenv (const char *,const char *, int);

#define WIFTERMSIG(x)  ((x).w_termsig)
#define WEXITSTATUS(x) ((x).w_retcode)
#define WSTOPSIG(x)    ((x).w_stopsig)

int waitpid(int, int *, int);

#endif
