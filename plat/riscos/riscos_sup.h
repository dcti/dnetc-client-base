/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * By Kevin Bracey <kbracey@acorn.com> and Chris Berry <cberry@acorn.com>
 * $Id: riscos_sup.h,v 1.1.2.2 2001/05/06 10:43:53 teichp Exp $
*/
#ifndef __RISCOS_SUP_H__
#define __RISCOS_SUP_H__

#include "riscos_x86.h"
#include "riscos_asm.h"

#ifdef __cplusplus
extern "C" {
#endif

int riscos_check_taskwindow(void);
static const char *riscos_x86_ident(void);
static unsigned int riscos_hsleep(unsigned long hsecs);
const char *riscos_x86_determine_name(void);
int riscos_count_cpus(void);
void riscos_clear_screen(void);
void riscos_backspace(void);
static const char *riscos_get_local_directory(const char *appname);
const char *riscos_localise_filename(const char *filename);
int riscos_find_local_directory(const char *progname);

unsigned int sleep(unsigned int s); /* unistd replacement */
void usleep(unsigned int us);  /* unistd replacement */
void sched_yield(void); /* if (riscos_check_taskwindow()) riscos_upcall_6(); */

#ifdef __cplusplus
}
#endif

#endif /* __RISCOS_SUP_H__ */
