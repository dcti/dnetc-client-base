/* Hey, Emacs, this a -*-C-*- file !
 *
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * RISC OS assembler support functions
*/
#ifndef __RISCOS_ASM_H__
#define __RISCOS_ASM_H__ "@(#)$Id: riscos_asm.h,v 1.2.4.1 2003/10/28 17:11:58 teichp Exp $"

#ifdef __cplusplus
extern "C" {
#endif

extern unsigned int read_monotonic_time(void);
extern unsigned long ARMident(void);
extern void riscos_upcall_6(void);

#if 0
 allocate_nonzero
 deallocate_nonzero
 adcon_nonzero
#endif

#ifdef __cplusplus
}
#endif

#endif /* __RISCOS_ASM_H__ */

