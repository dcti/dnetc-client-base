/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __CLIIDENT_H__
#define __CLIIDENT_H__ "@(#)$Id: cliident.h,v 1.10.4.1 2003/01/19 22:49:49 snake Exp $"

extern void CliIdentifyModules(void);
extern time_t CliGetNewestModuleTime(void);
extern int CliIsDevelVersion(void);
extern const char *CliGetFullVersionDescriptor(void);

#endif /* __CLIIDENT_H__ */
