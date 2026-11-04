/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Reset entries of the Tenstorrent RC5-72 firmware, one per RISC-V core of
 * a Tensix tile, R72TT_ENTRY_STRIDE (16) bytes apart from L1 address 0:
 * BRISC, NCRISC, TRISC0, TRISC1, TRISC2. BRISC always starts at 0; the host
 * points the other cores' reset PCs at their entries. Each entry passes its
 * core index to r72tt_main(). Every core's stack is at the top of the first
 * 4 KiB of its private local data RAM.
*/

.section .text.start
.global _start
.extern r72tt_main

#define ENTRY(core) \
    .balign 16;           \
    li sp, 0xFFB01000;    \
    li a0, core;          \
    j r72tt_main

_start:
    ENTRY(0)
    ENTRY(1)
    ENTRY(2)
    ENTRY(3)
    ENTRY(4)
