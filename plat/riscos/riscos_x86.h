/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef _RISCOS_X86_H_
#define _RISCOS_X86_H_ "@(#)$Id: riscos_x86.h,v 1.1.2.1 2001/01/21 15:10:28 cyp Exp $";

#ifdef __cplusplus
extern "C" {
#endif

const char *riscos_x86_ident(void);
/* returns NULL if no x86 present otherwise name or "" if no name */

//s32 rc5_unit_func_x86( RC5UnitWork *, u32 * );
//internal

#ifdef __cplusplus
}
#endif

#endif /* ifndef _RISCOS_X86_H_ */
