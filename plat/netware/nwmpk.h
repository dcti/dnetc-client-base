#ifndef __nwmpk_h__
#define __nwmpk_h__
/*============================================================================
=  NetWare C NLM Runtime Library source code
=
=  Unpublished Copyright (C) 1997 by Novell, Inc. All rights reserved.
=
=  No part of this file may be duplicated, revised, translated, localized or
=  modified in any manner or compiled, linked or uploaded or downloaded to or
=  from any computer system without the prior written consent of Novell, Inc.
=
=  nwmpk.h
==============================================================================
*/
#include <nwtypes.h>
#include <nwthread.h>
//#include <nwintxx.h>

/* ---------------------------------------------------------------------------
** Note: The use of many of these interfaces is not preferred. Use instead
** the thread management and synchronization services from nwthread.h and
** nwsemaph.h for portability across NetWare platforms or the interfaces from
** npl/thread.h and npl/synch.h for portability across NetWare and other plat-
** forms like NT and UNIX.
** ---------------------------------------------------------------------------
*/

/* thread creation attributes */
#define MPK_THREAD_ATTRIB_DETACHED     0
#define MPK_THREAD_ATTRIB_JOINABLE     1

/* for mutex and semaphore names */
#define MAX_MPK_NAME_LEN               63

/* read-write lock status */
#define MPK_RWLOCK_STATUS_FREE         0x1111
#define MPK_RWLOCK_STATUS_READ_LOCK    0x2222
#define MPK_RWLOCK_STATUS_WRITE_LOCK   0x4444

/* maximum initial value of a semaphore */
#define MPK_SEMAPHORE_VALUE_MAX			0x7fffffff

/* type definitions for NetWare Multiprocessing Kernel (MPK) APIs... */
typedef unsigned long      MPKError, MPKProc;
typedef void               *MPKAppl, *MPKBarrier, *MPKMutex, *MPKQue,
									*MPKQLite, *MPKRwLock, *MPKSema, *MPKCond,
                           *MPKThread;

#ifdef INT64
typedef unsigned INT64   DLONG;
#endif

typedef struct mpk_qlink
{
   struct mpk_qlink  *link;
   int               value;
   void              *head;
   void              *ptr;
} MPKQLink;

typedef struct mpk_qlink_lite
{
   struct mpk_qlink_lite   *link;
} MPKQLinkLite;

/* thread management prototypes... */
MPKThread   MPKCurrentThread( void );
MPKThread   MPKStartThread( const char *name, void (*start)(), void *stackAddr,
               size_t stackSize, void *arg );
MPKThread   MPKCreateThread( const char *name, void (*start)(), void *stackAddr,
               size_t stackSize, void *arg );
MPKError MPKScheduleThread( MPKThread thread );
void     MPKYieldThread( void );
void     MPKSuspendThread( MPKThread thread );
MPKError MPKResumeThread( MPKThread thread );
MPKError MPKGetThreadName( MPKThread thread, void *name, size_t maxLen );
MPKError MPKSetThreadName( MPKThread thread, const void *name );
MPKError MPKDestroyThread( MPKThread thread );
MPKError MPKExitThread( void *status );
MPKError MPKGetThreadExitCode( MPKThread thread, void **status );
MPKError MPKThreadCheckFromSuspendKill( void );
MPKError MPKGetThreadAttributes( MPKThread thread, LONG *attributes );
MPKError MPKSetThreadAttributes( MPKThread thread, LONG attributes );
LONG     MPKGetThreadPriority( MPKThread thread );
MPKError MPKSetThreadPriority( MPKThread thread, LONG priority );
MPKError MPKGetThreadList( MPKAppl app, MPKThread *buffer, LONG slots,
            LONG *slotsUsed );
MPKError MPKSetThreadUserData( MPKThread thread, void *data );
void     *MPKGetThreadUserData( MPKThread thread );

/* prototypes for entering and exiting the classic NetWare binding... */
MPKError	MPKExitClassicNetWare( void );
void     MPKEnterNetWare( void );
void     MPKExitNetWare( void );

/* read-write lock prototypes... */
MPKRwLock   MPKRWLockAlloc( const char *name );
MPKError MPKRWLockFree( MPKRwLock rwlock );
MPKError MPKRWReadLock( MPKRwLock rwlock );
MPKError MPKRWReadUnlock( MPKRwLock rwlock );
MPKError MPKRWReadTryLock( MPKRwLock rwlock );
MPKError MPKRWWriteLock( MPKRwLock rwlock );
MPKError MPKRWWriteUnlock( MPKRwLock rwlock );
MPKError MPKRWWriteTryLock( MPKRwLock rwlock );
MPKError MPKRWLockInfo(MPKRwLock rwlock, LONG *status, MPKThread *owner );

/* mutex prototypes... */
MPKMutex MPKMutexAlloc( const char *name );
MPKError MPKMutexFree( MPKMutex mutex );
MPKError MPKMutexLock( MPKMutex mutex );
MPKError MPKMutexUnlock( MPKMutex mutex );
MPKError MPKMutexTryLock( MPKMutex mutex );
LONG     MPKMutexRecursiveCount( MPKMutex mutex );
LONG     MPKMutexWaitCount( MPKMutex mutex );

/* semaphore prototypes... */
MPKSema  MPKSemaphoreAlloc( const char *name, long cnt );
MPKError MPKSemaphoreFree( MPKSema sema );
MPKError MPKSemaphoreWait( MPKSema sema );
MPKError MPKSemaphoreTimedWait( MPKSema sema, LONG milliseconds );
MPKError MPKSemaphoreSignal( MPKSema sema );
MPKError MPKSemaphoreTry( MPKSema sema );
long     MPKSemaphoreExamineCount( MPKSema sema );
long     MPKSemaphoreWaitCount( MPKSema sema );

/* barrier prototypes... */
MPKBarrier  MPKBarrierAlloc( const char *name, LONG threadCnt );
MPKError MPKBarrierFree( MPKBarrier barrier );
MPKError MPKBarrierWait( MPKBarrier barrier );
MPKError MPKBarrierIncrement( MPKBarrier barrier );
MPKError MPKBarrierDecrement( MPKBarrier barrier );
MPKError MPKBarrierThreadCount( MPKBarrier barrier, LONG *cnt );
MPKError MPKBarrierWaitCount( MPKBarrier barrier, LONG *cnt );

/* condition variable prototypes... */
MPKError MPKConditionAlloc( const char *name, MPKCond *cv );
MPKError MPKConditionDestroy( MPKCond cv );
MPKError MPKConditionWait( MPKCond cv, MPKMutex mutex );
MPKError MPKConditionTimedWait( MPKCond cv, MPKMutex mutex, LONG milliseconds );
MPKError MPKConditionSignal( MPKCond cv );
MPKError MPKConditionBroadcast( MPKCond cv );

/* atomic function prototypes... */
void     atomic_inc( LONG *addr );
void     atomic_dec( LONG *addr );
void     atomic_add( LONG *addr, LONG value );
void     atomic_sub( LONG *addr, LONG value );
LONG     atomic_bts( LONG *base, LONG offset );
LONG     atomic_btr( LONG *base, LONG offset );
LONG     atomic_xchg( LONG *addr, LONG value );

/* data queue management prototypes... */
MPKQue   MPKAllocQue( void );
MPKQue   MPKAllocQueNoSleep( void );
void     MPKFreeQue( MPKQue queue );
LONG     MPKQueCount( MPKQue queue );
MPKError MPKEnQue( MPKQue queue, MPKQLink *item );
MPKError MPKEnQueNoLock( MPKQue queue, MPKQLink *item );
#ifdef INT64
MPKError MPKEnQueOrdered( MPKQue queue, MPKQLink *item, INT64 value );
MPKError MPKEnQueOrderedNoLock( MPKQue queue, MPKQLink *item, INT64 value );
#endif
MPKError MPKPushQue( MPKQue queue, MPKQLink *item );
MPKError MPKPushQueNoLock( MPKQue queue, MPKQLink *item );
#ifdef INT64
MPKError MPKPushQueOrdered(MPKQue queue, MPKQLink *item, INT64 value );
MPKError MPKPushQueOrderedNoLock(MPKQue queue, MPKQLink *item, INT64 value );
#endif
MPKQLink *MPKDeQue( MPKQue queue );
MPKQLink *MPKDeQueNoLock( MPKQue queue );
MPKQLink *MPKDeQueByQLink( MPKQue queue, MPKQLink *item );
MPKQLink *MPKDeQueByQLinkNoLock( MPKQue queue, MPKQLink *item );
MPKQLink *MPKDeQueWait( MPKQue queue );
MPKQLink *MPKDeQueWaitNoLock( MPKQue queue );
MPKQLink *MPKDeQueAll( MPKQue queue );
MPKQLink *MPKDeQueAllNoLock( MPKQue queue );
MPKQLink *MPKFirstQLinkNoLock( MPKQue queue );

/* lighter-weight data queue management prototypes... */
MPKQLite       MPKAllocQueLite(void);
MPKQLite       MPKAllocQueLiteNoSleep(void);
void           MPKFreeQueLite(MPKQLite queue);
LONG           MPKQueLiteCount(MPKQLite queue);
MPKError       MPKEnQueLite(MPKQLite queue, MPKQLinkLite *item);
MPKError       MPKEnQueLiteNoLock( MPKQLite queue, MPKQLinkLite *item );
MPKError       MPKPushQueLite(MPKQLite queue, MPKQLinkLite *item);
MPKError       MPKPushQueLiteNoLock(MPKQLite queue, MPKQLinkLite *item);
MPKQLinkLite   *MPKDeQueLite(MPKQLite queue);
MPKQLinkLite   *MPKDeQueLiteNoLock(MPKQLite queue);
MPKQLinkLite   *MPKDeQueLiteByQueLink(MPKQLite queue, MPKQLinkLite *item);
MPKQLinkLite   *MPKDeQueLiteByQueLinkNoLock(MPKQLite queue,MPKQLinkLite *item);
MPKQLinkLite   *MPKDeQueLiteWait(MPKQLite queue);
MPKQLinkLite   *MPKDeQueLiteWaitNoLock(MPKQLite queue);
MPKQLinkLite   *MPKDeQueLiteAll(MPKQLite queue);
MPKQLinkLite   *MPKDeQueLiteAllNoLock(MPKQLite queue);
MPKQLinkLite   *MPKFirstQueLinkLiteNoLock(MPKQLite queue);

#endif
