/* $Id: next_sup.cpp,v 1.1.2.2 2003/08/09 12:21:41 mweiser Exp $ */

#include "next_sup.h"
#include <stdlib.h>   /* malloc */
#include <string.h>   /* strcpy */
#include <libc.h>     /* wait4 */
#include <errno.h>    /* errno, ENOMEM */

char *strdup(const char *src)
{
  char *dst = (char *)malloc(strlen(src + 1));

  if (dst != NULL)
    strcpy(dst, src);

  return dst;
}

/* from openbsd cvs 20030722:
 *
 * Copyright (c) 1987 Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE. */

/*
 * OpenBSD: getenv.c,v 1.6 2003/06/02 20:18:37 millert Exp 
 *
 * __findenv --
 *      Returns pointer to value associated with name, if any, else NULL.
 *      Sets offset to be the offset of the name/value combination in the
 *      environmental array, for use by setenv(3) and unsetenv(3).
 *      Explicitly removes '=' in argument name.
 *
 *      This routine *should* be a static; don't use it.
 */
char *__findenv(const char *name, unsigned int *offset)
{
  extern char **environ;
  register unsigned int len, i;
  register const char *np;
  register char **p, *cp;

  if (name == NULL || environ == NULL)
    return (NULL);
  for (np = name; *np && *np != '='; ++np)
    ;
  len = np - name;
  for (p = environ; (cp = *p) != NULL; ++p) {
    for (np = name, i = len; i && *cp; i--)
      if (*cp++ != *np++)
        break;
    if (i == 0 && *cp++ == '=') {
      *offset = p - environ;
      return (cp);
    }
  }
  return (NULL);
}

/*
 * OpenBSD: setenv.c,v 1.6 2003/06/02 20:18:38 millert Exp 
 *
 * setenv --
 *      Set the value of the environmental variable "name" to be
 *      "value".  If rewrite is set, replace any current value. */
int setenv(register const char *name,
           register const char *value,
           int rewrite)
{
  extern char **environ;
  static int alloced;                   /* if allocated space before */
  register char *C;
  register const char *C2;
  unsigned int l_value, offset;

  if (*value == '=')                    /* no `=' in value */
    ++value;
  l_value = strlen(value);
  if ((C = __findenv(name, &offset))) { /* find if already exists */
    if (!rewrite)
      return (0);
    if (strlen(C) >= l_value) {         /* old larger; copy over */
      while ((*C++ = *value++))
        ;
      return (0);
    }
  } else {                              /* create new slot */
    register unsigned int   cnt;
    register char         **P;

    for (P = environ, cnt = 0; *P; ++P, ++cnt);
    if (alloced) {                      /* just increase size */
      P = (char **)realloc((void *)environ,
                           (size_t)(sizeof(char *) * (cnt + 2)));
      if (!P)
        return (-1);

      environ = P;
    }
    else {                              /* get new space */
      alloced = 1;                      /* copy old entries into it */
      P = (char **)malloc((size_t)(sizeof(char *) *
                                   (cnt + 2)));
      if (!P)
        return (-1);
      bcopy(environ, P, cnt * sizeof(char *));
      environ = P;
    }
    environ[cnt + 1] = NULL;
    offset = cnt;
  }
  for (C2 = name; *C2 && *C2 != '='; ++C2);     /* no `=' in name */
  if (!(environ[offset] = (char *)              /* name + `=' + value */
        malloc((size_t)((int)(C2 - name) + l_value + 2))))
    return (-1);
  for (C = environ[offset]; (*C = *name++) && *C != '='; ++C)
    ;
  for (*C++ = '='; (*C++ = *value++); )
    ;
  return (0);
}

/* from openssh-3.6.1p2:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */
int waitpid(int pid, int *stat_loc, int options)
{
  union wait statusp;
  int wait_pid;

  if (pid <= 0) {
    if (pid != -1) {
      errno = EINVAL;
      return -1;
    }

    pid = 0;   /* wait4() wants pid=0 for indiscriminate wait. */
  }

  wait_pid = wait4(pid, &statusp, options, NULL);

  if (stat_loc)
    *stat_loc = (int) statusp.w_status;

  return wait_pid;
}

/* end of file */
