/*
 * Emulation functions/stubs for portability across NetWare versions.
 * All functions here are CLIB safe (don't require context)
 *
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * $Id: nwlemu.h,v 1.1.2.1 2001/01/21 15:10:30 cyp Exp $
*/

#ifndef __CNW_EMU_H__
#define __CNW_EMU_H__ 

#ifdef __cplusplus
extern "C" {
#endif

/* CLIB (none require context) */
extern void  *ImportSymbol(int NLMHandle, char *cname); /* 3.12/4.1 */
extern int    UnimportSymbol(int NLMHandle, char *cname); /* 3.12/4.1 */
extern void   ThreadSwitchWithDelay(void);               /* 3.12/4.1 */
extern void   ThreadSwitchLowPriority(void);             /* 4.11 */
extern int    NWSMPIsLoaded(void);                        /* 4.11 */
extern int    NWSMPIsAvailable(void);                     /* 4.11 */
extern void   NWThreadToMP(void);                         /* 4.11 */
extern void   NWThreadToNetWare(void);                    /* 4.11 */
extern void   NWSMPThreadToMP(void);                      /* 5.x */
extern void   NWSMPThreadToNetWare(void);                 /* 5.x */
extern int    GetServerConfigurationInfo(int *servType,int *loaderType); /*>=3.12/4.02*/
extern unsigned long NWGetSuperHighResolutionTimer(void); /* 4.11 1us always */

/* kernel */
extern unsigned long GetCurrentTime(void);              /* in *ticks* */
extern unsigned long GetSuperHighResolutionTimer(void); /* 838ns (NW3) or 1us (NW411) */
extern unsigned long GetHighResolutionTimer(void);      /* 100us timer */
extern void GetClockStatus(unsigned long _dataPtr[3]); /* 4.x */
extern unsigned long GetSystemConsoleScreen(void);      /* == systemConsoleScreen */
extern void RingTheBell(void);                          /* StartBell()/StopBell() */
extern unsigned long GetSetableParameterValue( unsigned long connum, 
                                    unsigned char *setParamName, void *val );
extern unsigned long SetSetableParameterValue( unsigned long connum, 
                                    unsigned char *setParamName, void *val );
extern unsigned int GetFileServerMajorVersionNumber(void);
extern unsigned int GetFileServerMinorVersionNumber(void);
extern unsigned int GetFileServerRevisionNumber(void);
extern unsigned int GetMaximumNumberOfPollingLoops(void);
extern unsigned int GetNumberOfPollingLoops(void);
extern void CRescheduleLastWithDelay(void), CYieldWithDelay(void), CRescheduleWithDelay(void);
extern void CRescheduleLastLowPriority(void), CYieldUntilIdle(void);
extern void CRescheduleLast(void), CRescheduleMyself(void), CYieldIfNeeded(void);
extern unsigned int GetProcessorUtilization(void); /* SMP.NLM or kernel */
extern unsigned int GetNumberOfRegisteredProcessors(void);
extern int ReturnFileServerName(char *buffer); /* to get len call with buffer==NULL */
extern int GetNestedInterruptLevel(void);
extern int GetDiskIOsPending(void); /* NWGet...() in 4.x */
extern int AddPollingProcedureRTag( void (*)(void), unsigned long rTag );
extern void RemovePollingProcedure( void (*)(void) );

#ifdef __cplusplus
}
#endif

#endif /* __CNW_EMU_H__ */
