/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __CLIENT_H__
#define __CLIENT_H__ "@(#)$Id: client.h,v 1.133.2.24 2002/03/25 01:45:46 andreasb Exp $"

#include "problem.h" /* WorkRecord, CONTEST_COUNT */
#include "lurk.h"    /* lurk_conf structure */

#define __TEXTIFY(x) #x
#define _TEXTIFY(x) __TEXTIFY(x)

#define PREFERREDBLOCKSIZE_DEFAULT       31  /* was 30, now 31 */
#define PREFERREDBLOCKSIZE_MIN           28
#define PREFERREDBLOCKSIZE_MAX           33
#define BUFFER_DEFAULT_IN_BASENAME  "buff-in"
#define BUFFER_DEFAULT_OUT_BASENAME "buff-out"
#define MINCLIENTOPTSTRLEN   64 /* no asciiz var is smaller than this */
#define NO_OUTBUFFER_THRESHOLDS /* no longer have outthresholds */

struct membuffstruct 
{ 
  unsigned long count; 
  WorkRecord *buff[500];
};

// ------------------

/* project flags are updated only from net. never saved to disk
 * user-disabled projects don't have a flag. They are 'invisible' to all
 * but the configuration functions ('invalid' slot in the loadorder_map)
*/
#define PROJECTFLAGS_CLOSED     0x01
#define PROJECTFLAGS_SUSPENDED  0x02 /* no data available */

// ------------------

typedef struct
{
  /* non-user-configurable */
  int  nonewblocks;
  int  rc564closed;
  int  stopiniio;
  u32  scheduledupdatetime;
  char inifilename[MINCLIENTOPTSTRLEN*2];
  u32  last_buffupd_time; /* monotonic. goes with max_buffupd_[retry_]interval */
  int  last_buffupd_failed_time;
  char project_flags[CONTEST_COUNT]; /* do NOT save to disk! */

  /* -- general -- */
  char id[MINCLIENTOPTSTRLEN];
  int  quietmode;
  int  blockcount;
  int  minutes;
  int  crunchmeter;
  int  corenumtotestbench;

  /* -- buffers -- */
  int  nodiskbuffers;
  struct { struct membuffstruct in, out; } membufftable[CONTEST_COUNT];
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
  char loadorder_map[CONTEST_COUNT];

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
  char smtpsrvr[MINCLIENTOPTSTRLEN];
  char smtpfrom[MINCLIENTOPTSTRLEN];
  char smtpdest[MINCLIENTOPTSTRLEN];

} Client;

// ------------------

void ResetClientData(Client *client); /* reset everything */
int ClientRun(Client *client);  /* run the loop, do the work */

// ------------------

#endif /* __CLIENT_H__ */
