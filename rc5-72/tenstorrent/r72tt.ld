/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Firmware image for a Tensix BRISC: linked at L1 address 0, below the
 * mailbox (R72TT_MAILBOX_ADDR). The image must not need .bss zeroing.
*/

ENTRY(_start)

SECTIONS
{
    . = 0;
    .text : {
        *(.text.start)
        *(.text*)
    }
    .rodata : {
        *(.rodata*)
        *(.srodata*)
    }
    .data : {
        *(.data*)
        *(.sdata*)
    }
    .bss : {
        *(.bss*)
        *(.sbss*)
        *(COMMON)
    }
    ASSERT(. <= 0x8000, "firmware overlaps the mailbox")
    ASSERT(SIZEOF(.bss) == 0, "firmware must not use .bss")
}
