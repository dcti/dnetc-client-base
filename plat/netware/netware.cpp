/* 
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Catch-all source file for the distributed.net client for NetWare 
 * written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 * 
 * (no, its not nice to do things this way, but its the only way
 * when one doesn't want to repeatedly muck with a shared makefile)
 *
 * $Id: netware.cpp,v 1.2 2002/09/02 00:35:51 andreasb Exp $
*/

#include "nwlemu.c"     /* kernel/clib portability stubs */
#include "nwlcomp.c"    /* ANSI/POSIX compatibility functions */
#include "nwmpk.c"      /* MP Kernel stubs/emulation */

#include "nwccons.c"    /* client console management */
#include "nwcmisc.c"    /* init/exit and other misc functions */
#include "nwcconf.c"    /* client-for-netware specific settings */

