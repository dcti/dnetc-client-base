/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/ 

#ifndef __W32SVC_H__
#define __W32SVC_H__ "@(#)$Id: w32svc.h,v 1.1.2.1 2001/01/21 15:10:26 cyp Exp $"

int win32CliUninstallService(int quiet); 
/*returns 0 on success. quiet is used internally by the service itself */
                        
int win32CliInstallService(int quiet);                        
/*returns 0 on success. quiet is used internally by the service itself */

int win32CliStartService(int argc, char **argv);
/* equivalent of "net start ... ", but works for 9x as well. */

int win32CliInitializeService(int argc, char **argv, int (*)(int, char **));
/* new style InitializeService. Returns 0 if service started ok */
                        
int win32CliServiceRunning(void);
/* returns !0 if *running* as a service */

int win32CliIsServiceInstalled(void);
/* returns <0=err, 0=no, >0=yes */

int win32CliSendServiceControlMessage(int msg); /* NT only! */
/* msg == SERVICE_CONTROL_STOP|PAUSE|CONTINUE|(re)START */
#define CLIENT_SERVICE_CONTROL_RESTART  128 //service control #
/* <0=err, 0=all stopped or not installed */

/* compatibility with old version (in)dependant stuff */
#define winInstallClient(_quiet) win32CliInstallService(_quiet)
#define winUninstallClient(_quiet) win32CliUninstallService(_quiet)

#endif /* __W32SVC_H__ */
