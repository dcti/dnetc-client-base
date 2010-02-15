/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __CLIENT_H__
#define __CLIENT_H__ "@(#)$Id: client.h,v 1.159 2010/02/15 19:44:26 stream Exp $"

#include "projdata.h" /* PROJECT_COUNT */
#include "problem.h"  /* WorkRecord, CONTEST_COUNT */
#include "lurk.h"     /* lurk_conf structure */

#define __TEXTIFY(x) #x
#define _TEXTIFY(x) __TEXTIFY(x)

#if (CLIENT_CPU == CPU_CELLBE) || (CLIENT_CPU == CPU_CUDA) || (CLIENT_CPU == CPU_ATI_STREAM)
#define PREFERREDBLOCKSIZE_DEFAULT       64
#else
#define PREFERREDBLOCKSIZE_DEFAULT       1
#endif
#define PREFERREDBLOCKSIZE_MIN           1
#define PREFERREDBLOCKSIZE_MAX           1024
#define BUFFER_DEFAULT_IN_BASENAME  "buff-in"
#define BUFFER_DEFAULT_OUT_BASENAME "buff-out"
#define MINCLIENTOPTSTRLEN   64 /* no asciiz var is smaller than this */
#define NO_OUTBUFFER_THRESHOLDS /* no longer have outthresholds */
#define DEFAULT_EXITFLAGFILENAME "exitdnet"EXTN_SEP"now"

// ------------------

typedef struct Client_struct
{
  /* non-user-configurable */
  int  nonewblocks;
  int  stopiniio;
  u32  scheduledupdatetime;
  char inifilename[MINCLIENTOPTSTRLEN*2];
  u32  last_buffupd_time; /* monotonic. goes with max_buffupd_[retry_]interval */
  int  last_buffupd_failed_time;
  int  buffupd_retry_delay;
  int  net_update_status;
  int  remote_update_status;
  int project_state[PROJECT_COUNT]; /* do NOT save states received from proxy to disk! */

  /* -- general -- */
  char id[MINCLIENTOPTSTRLEN];
  int  quietmode;
  int  blockcount;
  int  minutes;
  int  crunchmeter;
  int  corenumtotestbench;

  /* -- buffers -- */
  int  nodiskbuffers;
  char in_buffer_basename[MINCLIENTOPTSTRLEN*2];
  char out_buffer_basename[MINCLIENTOPTSTRLEN*2];
  char checkpoint_file[MINCLIENTOPTSTRLEN*2];
  int  offlinemode;
    int  nettimeout;
    int  nofallback;
    int  autofindkeyserver;
    char keyproxy[MINCLIENTOPTSTRLEN];
    int  keyport;
    char httpproxy[MINCLIENTOPTSTRLEN];
    int  uuehttpmode;
    char httpid[MINCLIENTOPTSTRLEN*2];
  int  noupdatefromfile;
    char remote_update_dir[MINCLIENTOPTSTRLEN*2];
  #ifdef LURK 
  struct dialup_conf lurk_conf;
  #endif
  int connectoften; /* 0=no,1=check both flush/fetch thresh, 2=only=flush*/
  // Don't use inthreshold directly, Use ClientGetInThreshold(client, contest)
  int inthreshold[CONTEST_COUNT]; 
  int timethreshold[CONTEST_COUNT];  /* in hours */
  int max_buffupd_interval; /* the better 'outthreshold'. in minutes */
  int max_buffupd_retry_interval;
  #if (!defined(NO_OUTBUFFER_THRESHOLDS))
  int outthreshold[CONTEST_COUNT];
  #endif
  int preferred_blocksize[CONTEST_COUNT];
  int project_order_map[PROJECT_COUNT];

  /* -- perf -- */
  int  numcpu;
  int  priority;
  int  coretypes[CONTEST_COUNT];

  /* -- triggers -- */
  int  restartoninichange;
  char pauseplist[MINCLIENTOPTSTRLEN]; /* processname list */
  char pausefile[MINCLIENTOPTSTRLEN*2];
  char exitflagfile[MINCLIENTOPTSTRLEN*2];
  int  nopauseifnomainspower;
  int  watchcputempthresh;
  char cputempthresh[MINCLIENTOPTSTRLEN]; /* [lowwatermark:]highwatermark */

  /* -- log -- */
  char logname[MINCLIENTOPTSTRLEN*2];
  char logfiletype[MINCLIENTOPTSTRLEN]; /* "none", "no limit", "rotate", "restart", "fifo" */
  char logfilelimit[MINCLIENTOPTSTRLEN]; /* "nnn K|M|days" etc */
  int  messagelen;
  int  logrotateUTC;                    /* true = UTC, false = local time */
  char smtpsrvr[MINCLIENTOPTSTRLEN];
  char smtpfrom[MINCLIENTOPTSTRLEN];
  char smtpdest[MINCLIENTOPTSTRLEN];

} Client;

// ------------------

void ResetClientData(Client *client); /* reset everything */
int ClientRun(Client *client);  /* run the loop, do the work */

// ------------------

#endif /* __CLIENT_H__ */
