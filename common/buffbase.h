/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ------------------------------------------------------------------
 * Buffer *file* primitives used by Client Buffer[Get|Put] methods,
 * checkpt etc. [membuff primitives are static and are in buffbase.cpp]
 * ------------------------------------------------------------------
*/
#ifndef __BUFFBASE_H__
#define __BUFFBASE_H__ "@(#)$Id: buffbase.h,v 1.2 1999/11/09 16:30:52 cyp Exp $"

/* ..Put() returns <0 on ioerr, else 0 on success */
int BufferPutFileRecord( const char *filename, const WorkRecord * data, 
               unsigned long *countP ); 
               
/* ..Get() returns <0 on err (ioerr ==-1, corrupt==-123), else 0 if ok */
int BufferGetFileRecord( const char *filename, WorkRecord * data, 
               unsigned long *countP ); //returns <0 on err, ==0 if norecs
               
/* ..GetNoOpt() is like Get() but is non-optimized/faster/for chkpt etc */
int BufferGetFileRecordNoOpt( const char *filename, WorkRecord * data, 
               unsigned long *countP );

/* ..Count() returns <0 on ioerr, else 0 on success */
int BufferCountFileRecords( const char *filename, unsigned int contest, 
               unsigned long *countP, unsigned long *normcountP );

/* "erase" a buffer to zero blocks */
int BufferZapFileRecords( const char *filename );

/* used by -forceunlock */
int UnlockBuffer( const char *filename ); 

/* import records from source, return -1 if err, or number of recs imported. */
/* On success, source is truncated/deleted. Used by checkpt and --import */
long BufferImportFileRecords( Client *client, const char *source_file, int interactive);

/* fetch/flush from remote file */
long BufferFlushFile( Client *client, const char *loadermap_flags );
long BufferFetchFile( Client *client, const char *loadermap_flags );

long PutBufferRecord(Client *client, const WorkRecord * data);
long GetBufferRecord(Client *client, WorkRecord * data, unsigned int contest, int use_out_file);
long GetBufferCount(Client *client, unsigned int contest, int use_out_file, unsigned long *normcountP );

#endif /* __BUFFBASE_H__ */
