/*
 * read in the netware specific settings and functions for accessing them.
 * Assumes cwd is already set up. Uses my readini stuff in common.
 *
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 * $Id: nwcconf.c,v 1.1.2.1 2001/01/21 15:10:29 cyp Exp $
 *
*/

#include <stdio.h>
#include <string.h>
#include "iniread.h" 

#include "nwlemu.h"  /* kernel stuff */
#include "nwcconf.h" /* ourselves */

static struct
{
  int fullypreemptive;
  int restartablethreads;
  int utilsupression;
  int pollallowed;
} nwcstatics = {-1,-1,-1,-1};

/* -------------------------------------------------------------------- */

#if defined(HAVE_POLLPROC_SUPPORT)
int nwCliGetPollingAllowedFlag(void)
{
  return (nwcstatics.pollallowed > 0); /* default is no */
}
#endif

int nwCliGetUtilizationSupressionFlag(void)
{
  return (nwcstatics.utilsupression > 0); /* default is no */
}

int nwCliAreCrunchersRestartable(void)
{
  return (nwcstatics.restartablethreads != 0); /* default is yes */
}

int nwCliIsPreemptiveEnv(void)
{
  return (nwcstatics.fullypreemptive > 0); /* default is no */
}

void nwCliLoadSettings(const char *inifile)
{
  static char last_inifile[64] = {0};
  if (!inifile)
  {
    if (!last_inifile[0])
      return;
    inifile = last_inifile;
  }
  else if (strlen(inifile) < (sizeof(last_inifile)-1))
    strcpy(last_inifile, inifile);

  if (access(inifile, 0)==0)
  {
    int i;
    const char *sect = "netware";

    i = 0;  
    if (GetFileServerMajorVersionNumber() >= 5)
      i = GetPrivateProfileIntB( sect,"fully-preemptive",0xDEF,inifile);
    nwcstatics.fullypreemptive = ((i==0)?(0):((i==0xDEF)?(-1):(1)));
    //ConsolePrintf("%s:fully-preemptive = %d\r\n", inifile, nwCliIsPreemptiveEnv());

    /* ++++++++++++++++++++++++++++++++++ */
    i=GetPrivateProfileIntB( sect,"restartable-threads",0xDEF,inifile);
    nwcstatics.restartablethreads = ((i==0)?(0):((i==0xDEF)?(-1):(1)));
    //ConsolePrintf("%s:restartable-threads = %d\r\n", inifile, nwCliAreCrunchersRestartable() );

    /* ++++++++++++++++++++++++++++++++++ */
    #if defined(HAVE_POLLPROC_SUPPORT) 
    if (GetNumberOfRegisteredProcessors()>1 
       /* || GetFileServerMajorVersionNumber()>=4 */ )
      nwcstatics.pollallowed = 0;
    else if (nwcstatics.pollallowed == -1)
    {
      if ((i=GetPrivateProfileIntB( sect,"use-polling-loop",0xDEF,inifile))!=0xDEF)
        nwcstatics.pollallowed = ((i == 0) ? (0) : (1));
      else if ((i=GetPrivateProfileIntB( sect,"pollallowed",0xDEF,inifile))!=0xDEF)
        nwcstatics.pollallowed = ((i == 0) ? (0) : (1));
      //ConsolePrintf("%s:use-polling-loop = %d\r\n", inifile, nwCliGetPollingAllowedFlag() );
    }
    #endif

    /* ++++++++++++++++++++++++++++++++++ */
    i = 0;
    if (GetFileServerMajorVersionNumber() < 5)
      i = GetPrivateProfileIntB( sect,"squelch-util-indicators",0xDEF,inifile);
    nwcstatics.utilsupression = ((i==0)?(0):((i==0xDEF)?(-1):(1)));
    //ConsolePrintf("%s:squelch-util-indicators = %d\r\n", inifile, nwCliGetUtilizationSupressionFlag() );
  }
  return;
}  

