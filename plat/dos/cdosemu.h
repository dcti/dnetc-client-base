/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * 
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * ----------------------------------------------------------------
 * get informational message used by the client (from PrintBanner)
 * used nowhere else, and is now quite useless :) 
 *
 * [it used to be the function that detected which yield method to use, 
 * but that is now in cdosidle]
 * ----------------------------------------------------------------
*/
#ifndef __CLIDOS_EMU_H__ 
#define __CLIDOS_EMU_H__ "@(#)$Id: cdosemu.h,v 1.1.2.2 2002/04/12 23:56:47 andreasb Exp $"

  const char *dosCliGetEmulationDescription(void);

#endif //__CLIDOS_EMU_H__
