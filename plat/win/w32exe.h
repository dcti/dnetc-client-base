/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/ 

#ifndef __W32EXE_H__
#define __W32EXE_H__ "@(#)$Id: w32exe.h,v 1.2 2002/09/02 00:35:53 andreasb Exp $"

/* is executable gui subsys? <0=err, 0=no, >0=yes */
extern int winIsGUIExecutable( const char *filename );

/* installer support */
extern int install_cmd_exever(int argc, char *argv[]);
extern int install_cmd_copyfile(int argc, char *argv[]);

#endif /* __W32EXE_H__ */
