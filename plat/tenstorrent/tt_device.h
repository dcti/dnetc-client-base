/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Minimal access layer for a Tenstorrent Blackhole device. The host reaches
 * each Tensix tile's L1 and registers through 2 MiB PCIe BAR0 TLB windows.
 * Two backends: a real device through the tenstorrent kernel driver, or the
 * ttsim simulator (libttsim.so), which presents a virtual PCIe device.
 *
 * $Id:
*/

#ifndef TT_DEVICE_H
#define TT_DEVICE_H

#include <stddef.h>
#include "cputypes.h"

#define TT_MAX_TILES 140

/* Blackhole Tensix clock, for converting between device clocks and time */
#define TT_AICLK_MHZ 1350

/* Tensix tile local addresses (as seen by a tile's own RISC-V cores) */
#define TT_L1_SIZE            (1536u * 1024u)
#define TT_REG_SOFT_RESET     0xFFB121B0u
#define TT_REG_WALL_CLOCK_LO  0xFFB121F0u
#define TT_REG_WALL_CLOCK_HI  0xFFB121F8u
#define TT_NOC0_REGS          0xFFB20000u

#define TT_REG_TRISC0_RESET_PC          0xFFB12228u  /* TRISC1, TRISC2 follow */
#define TT_REG_TRISC_RESET_PC_OVERRIDE  0xFFB12234u  /* bit n: TRISCn uses its RESET_PC */
#define TT_REG_NCRISC_RESET_PC          0xFFB12238u
#define TT_REG_NCRISC_RESET_PC_OVERRIDE 0xFFB1223Cu

/* soft reset bits: BRISC, TRISC0..2, NCRISC */
#define TT_SOFT_RESET_BRISC   0x00800u
#define TT_SOFT_RESET_TRISC0  0x01000u   /* TRISC1, TRISC2 follow */
#define TT_SOFT_RESET_NCRISC  0x40000u
#define TT_SOFT_RESET_ALL     0x47800u

/* A 2 MiB host window onto one tile's address space */
typedef struct
{
  volatile u32 *map;      /* device: mmap()ed window */
  ui64 paddr;             /* simulator: physical address of the window */
  u32 id;                 /* device: driver TLB id; simulator: BAR0 TLB index */
  u32 coord;              /* NOC coordinate mapped, or 0xFFFFFFFF */
  u32 base;               /* tile address mapped at the window start */
} tt_window_t;

typedef struct
{
  int is_sim;

  /* simulator */
  void *dl;
  void (*sim_init)(void);
  void (*sim_exit)(void);
  u32  (*cfg_rd32)(u32 bdf, u32 offset);
  void (*set_dma_callbacks)(void (*rd)(ui64, void *, u32),
                            void (*wr)(ui64, const void *, u32));
  void (*mem_rd)(ui64 paddr, void *dst, u32 size);
  void (*mem_wr)(ui64 paddr, const void *src, u32 size);
  void (*clock)(u32 n_clocks);
  ui64 bar0;

  /* device */
  int fd;

  tt_window_t reg_window;              /* moved between tiles' registers */
  tt_window_t l1_window[TT_MAX_TILES];

  unsigned tile_count;
  u32 tile_coord[TT_MAX_TILES];   /* NOC coordinate, x | (y << 6) */
  char name[64];
} tt_device_t;

/* Return 0 on success, < 0 on failure (with a message in err). */
/* Both use at most max_tiles Tensix tiles. */
int  tt_device_open_sim(tt_device_t *dev, const char *libttsim_path, unsigned max_tiles,
                        char *err, size_t errlen);
int  tt_device_open(tt_device_t *dev, const char *path, unsigned max_tiles,
                    char *err, size_t errlen);
/* Holds all RISC-V cores of all tiles in reset, then releases the device */
void tt_device_close(tt_device_t *dev);

/* Accesses must be 4-byte aligned. */
void tt_l1_write(tt_device_t *dev, unsigned tile, u32 addr, const void *src, u32 size);
void tt_l1_read(tt_device_t *dev, unsigned tile, u32 addr, void *dst, u32 size);
void tt_l1_write32(tt_device_t *dev, unsigned tile, u32 addr, u32 value);
u32  tt_l1_read32(tt_device_t *dev, unsigned tile, u32 addr);
void tt_reg_write32(tt_device_t *dev, unsigned tile, u32 addr, u32 value);
u32  tt_reg_read32(tt_device_t *dev, unsigned tile, u32 addr);

/* Lets the device make progress for about n_clocks device clocks. The
   simulator only advances inside this call; a real device runs on its own,
   so this just waits. */
void tt_advance(tt_device_t *dev, u32 n_clocks);

#endif /* TT_DEVICE_H */
