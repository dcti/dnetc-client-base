/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * By Kevin Bracey <kbracey@acorn.com> and Chris Berry <cberry@acorn.com>
 * $Id: riscos_sup.cpp,v 1.1.2.2 2001/05/06 10:43:53 teichp Exp $
*/

#include <stdio.h> /* printf() for debugging */
#include <kernel.h>
#include <sys/swis.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

#include "riscos_x86.h"  /* const char *riscos_x86_ident(void) */
#include "riscos_asm.h"  /* riscos_upcall_6() */
#include "riscos_sup.h"  /* ourselves */

int riscos_in_taskwindow=0;
int guiriscos=0;
int guirestart=0;

int riscos_check_taskwindow(void)
{
  static int in_taskwindow = -1;
  _kernel_swi_regs regs;
  _kernel_oserror *error;

  if (in_taskwindow < 0)
  {
    in_taskwindow = 1;
    regs.r[0]=0;
    error=_kernel_swi(XOS_Bit|TaskWindow_TaskInfo, &regs, &regs);
    in_taskwindow=regs.r[0];
    if (error)
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
  _kernel_swi_regs regs;

  _kernel_swi(XOS_Bit|(OS_WriteI + 12), &regs, &regs);
}

void riscos_backspace(void)
{
  _kernel_swi_regs regs;

  _kernel_swi(XOS_Bit|(OS_WriteI + 8), &regs, &regs);
  _kernel_swi(XOS_Bit|(OS_WriteI + 32), &regs, &regs);
  _kernel_swi(XOS_Bit|(OS_WriteI + 8), &regs, &regs);
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


static const char *riscos_get_local_directory(const char *appname)
{
  static char local_directory[1024];
  static int have_ld = -1;
  _kernel_swi_regs regs;

  if (have_ld < 0)
  {
    _kernel_oserror *e;
    if (appname) {
      regs.r[0]=37;
      regs.r[1]=(int)appname;
      regs.r[2]=(int)local_directory;
      regs.r[3]=(int)"Run$Path";
      regs.r[4]=0;
      regs.r[5]=sizeof(local_directory)-1;
      e = _kernel_swi(XOS_Bit|OS_FSControl, &regs, &regs);
    } else {
      regs.r[0]=37;
      regs.r[1]=(int)"@";
      regs.r[2]=(int)local_directory;
      regs.r[3]=0;
      regs.r[4]=0;
      regs.r[5]=sizeof(local_directory)-1;
      e = _kernel_swi(XOS_Bit|OS_FSControl, &regs, &regs);
    }
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
  _kernel_swi_regs regs;

  if (local_directory)
  {
    static char buffer[1024];
    _kernel_oserror *e;
    
    buffer[0] = '\0';
    regs.r[0]=37;
    regs.r[1]=(int)filename;
    regs.r[2]=(int)buffer;
    regs.r[3]=0;
    regs.r[4]=(int)local_directory;
    regs.r[5]=sizeof(buffer);
    e = _kernel_swi(XOS_Bit|OS_FSControl, &regs, &regs);
    if (!e)
    {
      buffer[sizeof(buffer)-1] = '\0';
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
