/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * By Kevin Bracey <kbracey@acorn.com> and Chris Berry <cberry@acorn.com>
 * $Id: riscos_sup.cpp,v 1.1.2.1 2001/01/21 15:10:28 cyp Exp $
*/

#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h> /* printf() for debugging */
#include <swis.h>  /* swix() */
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unixlib.h>
#include <sys/types.h>
#include <sys/time.h>
#ifdef __cplusplus
}
#endif
//#include "client.h"
#include "riscos_x86.h"  /* const char *riscos_x86_ident(void) */
#include "riscos_asm.h"  /* riscos_upcall_6() */
#include "riscos_sup.h"  /* ourselves */

int __root_stack_size = 16384;

int riscos_in_taskwindow=0;
int guiriscos=0;
int guirestart=0;

int riscos_check_taskwindow(void)
{
  static int in_taskwindow = -1;
  if (in_taskwindow < 0)
  {
    in_taskwindow = 1;
    if (_swix(TaskWindow_TaskInfo, _IN(0)|_OUT(0), 0, &in_taskwindow))
      in_taskwindow = 0;
    else if (in_taskwindow != 0)
      in_taskwindow = 1;
  }
  return in_taskwindow;
}

static unsigned int riscos_hsleep(unsigned long hsecs)
{
  unsigned int lasttime = 0;
  do
  {
    // This doesn't work - it won't return in a taskwindow when
    // the screen blanks if DPMS is in use.
    //_swix(OS_Byte,_INR(0,2),129,(s*100)&0xff,((s*100)>>8)&0x7f);
    if (riscos_check_taskwindow())
      riscos_upcall_6();
    else
    {
      printf("");
    }
    if (hsecs)
    {
      unsigned int timenow = read_monotonic_time();
      if (lasttime == 0)
        lasttime = timenow;
      else if (lasttime > timenow)
        hsecs--; /* oops */
      else if (lasttime < timenow)
      {
        if ((timenow - lasttime) > hsecs)
          lasttime = timenow - hsecs;
        hsecs -= (timenow - lasttime);
      }
      lasttime = timenow;
    }
  } while (hsecs);

  return 0;
}

int riscos_get_file_modified(const char *filename, unsigned long *timestampP)
{
    unsigned int obtype;
    unsigned long timestamp=0;
    unsigned long load,exec;

  if (_swix(OS_File, _INR(0,1)|_OUT(0)|_OUT(2)|_OUT(3),
            17, riscos_localise_filename(filename), &obtype, &load,&exec))
    return -1;
  if (obtype != 1)
    return -1;

  if (timestampP)
  {
      if ((load&0xfff00000) == 0xfff00000)
      {
	  timestamp = (exec >> 8)|(load<<24);
      }
      *timestampP=timestamp;
  }
  return 0;

}

int riscos_get_filesize(const char *filename, unsigned long *fsizeP)
{
  unsigned int obtype;  unsigned long fsize;
  //    printf("stat: filename = %s\n",filename);
  if (_swix(OS_File, _INR(0,1)|_OUT(0)|_OUT(4),
            17, riscos_localise_filename(filename), &obtype, &fsize))
    return -1;
  if (obtype != 1)
    return -1;
  if (fsizeP)
    *fsizeP = fsize;
  return 0;
}

int riscos_get_filelength(int fd, unsigned long *fsizeP)
{
  unsigned long fsize;
  if (_swix(OS_Args, _INR(0,1)|_OUT(2), 2, fd, &fsize))
    return -1;
  if (fsizeP)
    *fsizeP = fsize;
  return 0;
}

int riscos_chsize(int fd, unsigned long newsize)
{
  if (_swix(OS_Args, _INR(0,2), 3, fd, newsize))
    return -1;
  return 0;
}

#ifdef HAVE_X86_CARD_SUPPORT
const char *riscos_x86_determine_name(void)
{
  const char *name = riscos_x86_ident();
  if (!name)
    name = "";
  return name;
}
#endif

int riscos_count_cpus(void)
{
#ifdef HAVE_X86_CARD_SUPPORT

  if (riscos_x86_ident())
    return 2;
#endif
  return 1;
}

void riscos_clear_screen(void)
{
  _swix(OS_WriteI + 12, 0);
}

void riscos_backspace(void)
{
 _swix(OS_WriteI + 8, 0);
 _swix(OS_WriteI + 32, 0);
 _swix(OS_WriteI + 8, 0);
}


unsigned int sleep(unsigned int s)
{
  return riscos_hsleep(s * 100);
}

void usleep(unsigned int us)
{
  riscos_hsleep((us + 5000) / 10000);
}

void sched_yield(void)
{
  riscos_hsleep(0);
}

//typedef unsigned long off_t;

int ftruncate(int fd, off_t size)
{
  return riscos_chsize(fd, size);
}

static const char *riscos_get_local_directory(const char *appname)
{
  static char local_directory[1024];
  static int have_ld = -1;

  if (have_ld < 0)
  {
    _kernel_oserror *e;
    if (appname)
      e = _swix(OS_FSControl, _INR(0,5), 37, appname, local_directory,
                              "Run$Path", 0, sizeof(local_directory)-1 );
    else
      e = _swix(OS_FSControl, _INR(0,5), 37, "@", local_directory,
                                0, 0, sizeof(local_directory)-1 );
    have_ld = 0;
    local_directory[sizeof(local_directory)-1]='\0';
    if (!e)
    {
      unsigned int pos = 0, fnpos = 0;
      while (local_directory[pos])
      {
        if (local_directory[pos] == '.')
          fnpos = pos+1;
        pos++;
      }
      if (fnpos)
      {
        local_directory[fnpos] = '\0';
        have_ld = 1;
      }
    }
  }
  if (have_ld > 0)
    return (const char *)&local_directory[0];
  return (const char *)0;
}

const char *riscos_localise_filename(const char *filename)
{
  const char *local_directory = riscos_get_local_directory((const char *)0);
  if (local_directory)
  {
    static char buffer[1024];
    _kernel_oserror *e;
    buffer[0] = '\0';
    e = _swix(OS_FSControl, _INR(0,5), 37, filename, buffer,
                                0, local_directory, sizeof(buffer));
    if (!e)
    {
      buffer[sizeof(buffer)-1] = '\0';
      //printf("Localised to \"%s\"\n", buffer);
      return buffer;
    }
  }
  return filename;
}

int riscos_find_local_directory(const char *progname)
{
  if (riscos_get_local_directory(progname))
    return 0;
  return -1;
}

/* RiscOS time() returns local time, and gmtime()/localtime() are made to
   operate on local time. gettimeofday() from Socketlib returns UTC. */

time_t riscos_utcbase_time(time_t *t)
{
  struct timeval tv;

  gettimeofday(&tv, 0);
  if (t)
    *t = tv.tv_sec;

  return tv.tv_sec;
}

static time_t __utc_to_libc(time_t utc)
{
  struct timeval tv;

  gettimeofday(&tv, 0);

  return utc - tv.tv_sec + time(0);
  /*     utc - utc        + localtime */
}

struct tm *riscos_utcbase_gmtime(const time_t *utc)
{
  time_t loc;

  loc = __utc_to_libc(*utc);

  return gmtime(&loc);
}

struct tm *riscos_utcbase_localtime(const time_t *utc)
{
  time_t loc;

  loc = __utc_to_libc(*utc);

  return localtime(&loc);
}

char *strdup(const char *s)
{
  char *r;

  r=malloc(strlen(s)+1);
  strcpy(r, s);

  return r;
}


