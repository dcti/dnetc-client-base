/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2001 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __BUFFUPD_H__
#define __BUFFUPD_H__ "@(#)$Id: buffupd.h,v 1.5.2.5 2001/02/03 18:20:36 cyp Exp $"

// pass flags ORd with BUFFERUPDATE_FETCH/*_FLUSH. 
// if interactive, prints "Input buffer full. No fetch required" etc.
// returns updated flags or < 0 if failed. (offlinemode!=0/NetOpen() etc)

// BUFFUPDCHECK_* bits are for use with BufferCheckIfUpdateNeeded().
// For a longer description of how they are used refer to buffbase.cpp

#define BUFFERUPDATE_FETCH       0x01 /* do or check_need_for fetch */
#define BUFFERUPDATE_FLUSH       0x02 /* do or check_need_for flush */
#define BUFFUPDCHECK_TOPOFF      0x20 /* fill_even_if_not_completely_empty */
#define BUFFUPDCHECK_EITHER      0x40 /* either true sets both true */
#define BUFFERUPDATE_LASTBIT     0x40 /* first free is _LASTBIT<<1 */

int BufferUpdate( Client *client, int updatereq_flags, int interactive );

int BufferCheckIfUpdateNeeded(Client *client, int contestid, int upd_flags);

/* these next 3 (and the following BufferNetUpdate() proto)
** are internal to BufferUpdate()<->Buffer[Net|File]Update()
*/
#define BUFFERUPDATE_STATE_NEWS      (BUFFERUPDATE_LASTBIT<<1)
#define BUFFERUPDATE_STATE_TRANSERR  (BUFFERUPDATE_LASTBIT<<2)
#define BUFFERUPDATE_STATE_MSGPOSTED (BUFFERUPDATE_LASTBIT<<3)
int BufferNetUpdate(Client *client,int updatereq_flags, int break_pending, 
                    int interactive, char *loaderflags_map);

#endif /* __BUFFUPD_H__ */
