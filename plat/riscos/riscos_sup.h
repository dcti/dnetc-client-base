/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * By Kevin Bracey <kbracey@acorn.com> and Chris Berry <cberry@acorn.com>
 * $Id: riscos_sup.h,v 1.1.2.1 2001/01/21 15:10:28 cyp Exp $
*/
#ifndef __RISCOS_SUP_H__
#define __RISCOS_SUP_H__

#include "riscos_x86.h"
#include "riscos_asm.h"

#ifdef __cplusplus
extern "C" {
#endif

//#include <sys/stat.h>

int riscos_check_taskwindow(void);
//static const char *riscos_x86_ident(void);
//static unsigned int riscos_hsleep(unsigned long hsecs);
int riscos_get_filesize(const char *filename, unsigned long *fsizeP);
int riscos_get_filelength(int fd, unsigned long *fsizeP);
int riscos_get_file_modified(const char *filename, unsigned long *timestampP);

int riscos_chsize(int fd, unsigned long newsize);
const char *riscos_x86_determine_name(void);
int riscos_count_cpus(void);
void riscos_clear_screen(void);
void riscos_backspace(void);
//static const char *riscos_get_local_directory(const char *appname);
const char *riscos_localise_filename(const char *filename);
int riscos_find_local_directory(const char *progname);

unsigned int sleep(unsigned int s); /* unistd replacement */
void usleep(unsigned int us);  /* unistd replacement */
//typedef unsigned int off_t;
int ftruncate(int fd, off_t size);  /* unistd replacement */
void sched_yield(void); /* if (riscos_check_taskwindow()) riscos_upcall_6(); */

char *strdup(const char *s);

/* RiscOS time() returns local time, and gmtime()/localtime() are made to
   operate on local time. gettimeofday() from Socketlib returns UTC. */
time_t riscos_utcbase_time(time_t *t);
static time_t __utc_to_libc(time_t utc);
struct tm *riscos_utcbase_gmtime(const time_t *utc);
struct tm *riscos_utcbase_localtime(const time_t *utc);

#ifdef __cplusplus
}
#endif

#endif /* __RISCOS_SUP_H__ */
