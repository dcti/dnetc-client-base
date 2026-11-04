/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Host <-> firmware interface of the Tenstorrent RC5-72 core. Shared by the
 * host core and the device firmware, so it uses plain C types only.
 *
 * Each of the five RISC-V cores of every Tensix tile runs the same
 * persistent firmware, polling its own mailbox in the tile's L1. The host
 * fills in a work description and then bumps 'seq'; the core checks 'count'
 * keys starting at the given key (in the client's key order), fills in the
 * results and finally sets 'done_seq' = 'seq'.
 *
 * $Id:
*/

#ifndef R72TT_FW_H
#define R72TT_FW_H

#include <stdint.h>

/* Firmware text/data occupy L1 from address 0 up to the mailboxes */
#define R72TT_MAILBOX_ADDR    0x8000u
#define R72TT_MAILBOX_STRIDE  128u
#define R72TT_FW_MAX_BYTES    R72TT_MAILBOX_ADDR

/* Cores per tile, in mailbox order. Core c starts at L1 address
   c * R72TT_ENTRY_STRIDE; BRISC's reset address is fixed at 0. */
#define R72TT_CORE_BRISC      0u
#define R72TT_CORE_NCRISC     1u
#define R72TT_CORE_TRISC0     2u
#define R72TT_CORE_TRISC1     3u
#define R72TT_CORE_TRISC2     4u
#define R72TT_CORES           5u
#define R72TT_ENTRY_STRIDE    16u

#define R72TT_MAGIC_READY     0x52373254u   /* "R72T", written by firmware at boot */
#define R72TT_ABI_VERSION     3u

/* Mailbox flags. TRISC1 drives the Vector Unit (SFPU) when FLAG_SFPU is
   set; FLAG_VERIFY then additionally checks every key's SFPU result
   against the scalar core, counting differences in verify_errors. */
#define R72TT_FLAG_SFPU       1u
#define R72TT_FLAG_VERIFY     2u

typedef struct
{
  /* status */
  uint32_t magic;          /* fw: R72TT_MAGIC_READY once booted */
  uint32_t abi_version;    /* fw: R72TT_ABI_VERSION */
  uint32_t seq;            /* host: bumped to start a run */
  uint32_t done_seq;       /* fw: set to seq once results are valid */

  /* input */
  uint32_t key_hi, key_mid, key_lo;   /* first key, client format (RC5_72UnitWork.L0) */
  uint32_t count;                     /* number of keys to check */
  uint32_t plain_lo, plain_hi;
  uint32_t cypher_lo, cypher_hi;
  uint32_t flags;                     /* R72TT_FLAG_* */

  /* output; indices are relative to the first key */
  uint32_t keys_done;      /* == count, or index of the full match */
  uint32_t cmc_count;      /* partial matches in [0, keys_done] */
  uint32_t cmc_index;      /* index of the last partial match */
  uint32_t found;          /* 1 if keys_done is a full match */
  uint32_t cycles_lo, cycles_hi;      /* device clocks spent on the run */
  uint32_t verify_errors;             /* see R72TT_FLAG_VERIFY */
} r72tt_mailbox_t;

#define R72TT_MAILBOX(core)   (R72TT_MAILBOX_ADDR + (core) * R72TT_MAILBOX_STRIDE)

#endif /* R72TT_FW_H */
