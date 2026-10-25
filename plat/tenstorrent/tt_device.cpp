/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id:
*/

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "tt_device.h"
#include "tt_kmd_ioctl.h"

#define BH_VENDOR_ID           0x1E52u
#define BH_DEVICE_ID           0xB140u

#define TLB_WINDOW_BYTES       (2u * 1024u * 1024u)
#define REG_WINDOW_BASE        0xFFA00000u   /* covers the tile registers we use */

/* Simulator only: BAR0 TLB configuration registers and window indices. The
   kernel driver reserves the last 2 MiB window (201) on real devices; the
   simulator is kept to the same layout. */
#define SIM_TLB_CFG_BASE       0x1FC00000u
#define SIM_REG_WINDOW         200u

/* Blackhole Tensix columns (NOC0 x coordinates) and rows */
static const u32 bh_tensix_columns[] = { 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16 };
#define BH_TENSIX_COLUMNS ((unsigned)(sizeof(bh_tensix_columns) / sizeof(bh_tensix_columns[0])))
#define BH_TENSIX_FIRST_ROW 2u
#define BH_TENSIX_LAST_ROW  11u

/* NOC0 registers, relative to TT_NOC0_REGS */
#define NOC_NIU_CFG_0          0x100u  /* bit 14: coordinate translation enabled */
#define NOC_ROUTER_CFG_1       0x108u  /* physical columns that are not Tensix or are harvested */
#define NOC_ID_LOGICAL         0x148u

/* --- windows -------------------------------------------------------------- */

static int window_configure(tt_device_t *dev, tt_window_t *w, u32 coord, u32 base)
{
  if (w->coord == coord && w->base == base)
    return 0;
  if (dev->is_sim)
  {
    ui64 cfg_addr = dev->bar0 + SIM_TLB_CFG_BASE + (ui64)w->id * 12;
    u32 cfg[3];
    int i;
    cfg[0] = base >> 21;
    cfg[1] = coord << 11;
    cfg[2] = 0;
    /* TLB configuration registers only accept 32-bit writes */
    for (i = 0; i < 3; i++)
      dev->mem_wr(cfg_addr + i * 4, &cfg[i], sizeof(cfg[i]));
  }
  else
  {
    struct tenstorrent_configure_tlb c;
    memset(&c, 0, sizeof(c));
    c.in.id = w->id;
    c.in.config.addr = base;
    c.in.config.x_end = coord & 63;
    c.in.config.y_end = coord >> 6;
    c.in.config.ordering = 1;   /* strict */
    if (ioctl(dev->fd, TENSTORRENT_IOCTL_CONFIGURE_TLB, &c) != 0)
      return -1;
  }
  w->coord = coord;
  w->base = base;
  return 0;
}

static void window_read(tt_device_t *dev, tt_window_t *w, u32 offset, void *dst, u32 size)
{
  if (dev->is_sim)
    dev->mem_rd(w->paddr + offset, dst, size);
  else
  {
    /* uncached BAR space: whole, aligned 32-bit accesses only */
    const volatile u32 *src = w->map + offset / 4;
    u8 *d = (u8 *)dst;
    for (; size >= 4; size -= 4, d += 4)
    {
      u32 v = *src++;
      memcpy(d, &v, 4);
    }
    if (size)
    {
      u32 v = *src;
      memcpy(d, &v, size);
    }
  }
}

static void window_write(tt_device_t *dev, tt_window_t *w, u32 offset, const void *src, u32 size)
{
  if (dev->is_sim)
    dev->mem_wr(w->paddr + offset, src, size);
  else
  {
    volatile u32 *dst = w->map + offset / 4;
    const u8 *s = (const u8 *)src;
    for (; size >= 4; size -= 4, s += 4)
    {
      u32 v;
      memcpy(&v, s, 4);
      *dst++ = v;
    }
    if (size)
    {
      u32 v = *dst;
      memcpy(&v, s, size);
      *dst = v;
    }
  }
}

/* Device only: gets a 2 MiB window from the driver */
static int window_allocate(tt_device_t *dev, tt_window_t *w)
{
  struct tenstorrent_allocate_tlb a;
  void *map;

  memset(&a, 0, sizeof(a));
  a.in.size = TLB_WINDOW_BYTES;
  if (ioctl(dev->fd, TENSTORRENT_IOCTL_ALLOCATE_TLB, &a) != 0)
    return -1;
  map = mmap(NULL, TLB_WINDOW_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd,
             (off_t)a.out.mmap_offset_uc);
  if (map == MAP_FAILED)
  {
    struct tenstorrent_free_tlb f;
    memset(&f, 0, sizeof(f));
    f.in.id = a.out.id;
    ioctl(dev->fd, TENSTORRENT_IOCTL_FREE_TLB, &f);
    return -1;
  }
  w->map = (volatile u32 *)map;
  w->id = a.out.id;
  w->coord = 0xFFFFFFFFu;
  return 0;
}

static void window_free(tt_device_t *dev, tt_window_t *w)
{
  if (!dev->is_sim && w->map)
  {
    struct tenstorrent_free_tlb f;
    munmap((void *)w->map, TLB_WINDOW_BYTES);
    memset(&f, 0, sizeof(f));
    f.in.id = w->id;
    ioctl(dev->fd, TENSTORRENT_IOCTL_FREE_TLB, &f);
  }
  w->map = NULL;
}

/* --- tile access -------------------------------------------------------- */

static u32 reg_read32_at(tt_device_t *dev, u32 coord, u32 addr)
{
  u32 value = 0;
  window_configure(dev, &dev->reg_window, coord, REG_WINDOW_BASE);
  window_read(dev, &dev->reg_window, addr - REG_WINDOW_BASE, &value, sizeof(value));
  return value;
}

void tt_l1_write(tt_device_t *dev, unsigned tile, u32 addr, const void *src, u32 size)
{
  window_write(dev, &dev->l1_window[tile], addr, src, size);
}

void tt_l1_read(tt_device_t *dev, unsigned tile, u32 addr, void *dst, u32 size)
{
  window_read(dev, &dev->l1_window[tile], addr, dst, size);
}

void tt_l1_write32(tt_device_t *dev, unsigned tile, u32 addr, u32 value)
{
  tt_l1_write(dev, tile, addr, &value, sizeof(value));
}

u32 tt_l1_read32(tt_device_t *dev, unsigned tile, u32 addr)
{
  u32 value = 0;
  tt_l1_read(dev, tile, addr, &value, sizeof(value));
  return value;
}

void tt_reg_write32(tt_device_t *dev, unsigned tile, u32 addr, u32 value)
{
  window_configure(dev, &dev->reg_window, dev->tile_coord[tile], REG_WINDOW_BASE);
  window_write(dev, &dev->reg_window, addr - REG_WINDOW_BASE, &value, sizeof(value));
}

u32 tt_reg_read32(tt_device_t *dev, unsigned tile, u32 addr)
{
  return reg_read32_at(dev, dev->tile_coord[tile], addr);
}

void tt_advance(tt_device_t *dev, u32 n_clocks)
{
  if (dev->is_sim)
    dev->clock(n_clocks);
  else
  {
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = (long)n_clocks * 1000 / TT_AICLK_MHZ;
    nanosleep(&ts, NULL);
  }
}

/* Find the usable Tensix tiles. With NOC coordinate translation enabled
   (as set up by board firmware), translated Tensix columns list the active
   physical columns first and the harvested ones last, so the first
   'active' translated columns are usable. Without translation, use the
   non-harvested physical columns directly. */
static int enumerate_tiles(tt_device_t *dev, unsigned max_tiles, char *err, size_t errlen)
{
  const u32 probe = bh_tensix_columns[0] | (BH_TENSIX_FIRST_ROW << 6);
  u32 niu_cfg = reg_read32_at(dev, probe, TT_NOC0_REGS + NOC_NIU_CFG_0);
  u32 harvested = reg_read32_at(dev, probe, TT_NOC0_REGS + NOC_ROUTER_CFG_1);
  int translated = (niu_cfg >> 14) & 1;
  unsigned active = 0, c, row;
  u32 columns[BH_TENSIX_COLUMNS];

  for (c = 0; c < BH_TENSIX_COLUMNS; c++)
  {
    if (!(harvested & (1u << bh_tensix_columns[c])))
      columns[active++] = bh_tensix_columns[c];
  }
  if (translated)
  {
    for (c = 0; c < active; c++)
      columns[c] = bh_tensix_columns[c];
  }

  dev->tile_count = 0;
  for (c = 0; c < active; c++)
  {
    for (row = BH_TENSIX_FIRST_ROW; row <= BH_TENSIX_LAST_ROW; row++)
    {
      u32 coord = columns[c] | (row << 6);
      u32 id = reg_read32_at(dev, coord, TT_NOC0_REGS + NOC_ID_LOGICAL) & 0xFFFu;
      if (translated && id != coord)
      {
        snprintf(err, errlen, "tile %u-%u reports logical id %u-%u",
                 coord & 63, coord >> 6, id & 63, id >> 6);
        return -1;
      }
      if (dev->tile_count < max_tiles)
        dev->tile_coord[dev->tile_count++] = coord;
    }
  }
  if (!dev->tile_count)
  {
    snprintf(err, errlen, "no usable Tensix tiles");
    return -1;
  }
  return 0;
}

/* Points one window at the L1 of each tile */
static int map_tiles(tt_device_t *dev, char *err, size_t errlen)
{
  unsigned t;

  for (t = 0; t < dev->tile_count; t++)
  {
    if (window_configure(dev, &dev->l1_window[t], dev->tile_coord[t], 0) != 0)
    {
      snprintf(err, errlen, "cannot configure TLB window: %s", strerror(errno));
      return -1;
    }
  }
  return 0;
}

/* --- simulator ------------------------------------------------------------ */

static void sim_dma_rd(ui64 paddr, void *dst, u32 size)
{
  /* The RC5-72 firmware never touches host memory */
  fprintf(stderr, "[tenstorrent] unexpected device DMA read of %u bytes at 0x%llx\n",
          size, (unsigned long long)paddr);
  memset(dst, 0, size);
}

static void sim_dma_wr(ui64 paddr, const void *src, u32 size)
{
  (void)src;
  fprintf(stderr, "[tenstorrent] unexpected device DMA write of %u bytes at 0x%llx\n",
          size, (unsigned long long)paddr);
}

#define LOAD_SYM(field, name) \
  do { \
    *(void **)(&dev->field) = dlsym(dev->dl, name); \
    if (!dev->field) { \
      snprintf(err, errlen, "%s: missing symbol %s", path, name); \
      dlclose(dev->dl); \
      dev->dl = NULL; \
      return -1; \
    } \
  } while (0)

int tt_device_open_sim(tt_device_t *dev, const char *path, unsigned max_tiles,
                       char *err, size_t errlen)
{
  u32 bdf;
  unsigned t;

  memset(dev, 0, sizeof(*dev));
  dev->is_sim = 1;
  dev->fd = -1;
  dev->dl = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!dev->dl)
  {
    snprintf(err, errlen, "%s", dlerror());
    return -1;
  }
  LOAD_SYM(sim_init, "libttsim_init");
  LOAD_SYM(sim_exit, "libttsim_exit");
  LOAD_SYM(cfg_rd32, "libttsim_pci_config_rd32");
  LOAD_SYM(set_dma_callbacks, "libttsim_set_pci_dma_mem_callbacks");
  LOAD_SYM(mem_rd, "libttsim_pci_mem_rd_bytes");
  LOAD_SYM(mem_wr, "libttsim_pci_mem_wr_bytes");
  LOAD_SYM(clock, "libttsim_clock");

  dev->set_dma_callbacks(sim_dma_rd, sim_dma_wr);
  dev->sim_init();

  /* Use the first Blackhole endpoint; endpoints need not be contiguous */
  for (bdf = 0; bdf < 0x10000u; bdf += 8)
  {
    if (dev->cfg_rd32(bdf, 0) == ((BH_DEVICE_ID << 16) | BH_VENDOR_ID))
    {
      dev->bar0 = ((ui64)dev->cfg_rd32(bdf, 0x14) << 32) | (dev->cfg_rd32(bdf, 0x10) & ~0xFu);
      break;
    }
  }
  if (bdf >= 0x10000u)
  {
    snprintf(err, errlen, "simulator exposes no Blackhole device");
    tt_device_close(dev);
    return -1;
  }

  dev->reg_window.id = SIM_REG_WINDOW;
  dev->reg_window.paddr = dev->bar0 + (ui64)SIM_REG_WINDOW * TLB_WINDOW_BYTES;
  dev->reg_window.coord = 0xFFFFFFFFu;
  for (t = 0; t < TT_MAX_TILES; t++)
  {
    dev->l1_window[t].id = t;
    dev->l1_window[t].paddr = dev->bar0 + (ui64)t * TLB_WINDOW_BYTES;
    dev->l1_window[t].coord = 0xFFFFFFFFu;
  }
  if (enumerate_tiles(dev, max_tiles, err, errlen) < 0 || map_tiles(dev, err, errlen) < 0)
  {
    tt_device_close(dev);
    return -1;
  }
  snprintf(dev->name, sizeof(dev->name), "Blackhole (ttsim), %u Tensix tiles", dev->tile_count);
  return 0;
}

/* --- device --------------------------------------------------------------- */

int tt_device_open(tt_device_t *dev, const char *path, unsigned max_tiles,
                   char *err, size_t errlen)
{
  struct tenstorrent_get_device_info info;
  struct tenstorrent_power_state power;
  unsigned t;
  int probe;

  memset(dev, 0, sizeof(*dev));
  dev->reg_window.coord = 0xFFFFFFFFu;

  /* The driver grants exclusive ownership to O_EXCL opens; verify that it
     does, as older drivers silently ignore O_EXCL. */
  dev->fd = open(path, O_RDWR | O_CLOEXEC | O_APPEND | O_EXCL | O_NONBLOCK);
  if (dev->fd < 0)
  {
    if (errno == EAGAIN || errno == EBUSY)
      snprintf(err, errlen, "%s is in use by another program", path);
    else
      snprintf(err, errlen, "cannot open %s: %s", path, strerror(errno));
    return -1;
  }
  probe = open(path, O_RDWR | O_CLOEXEC | O_APPEND | O_NONBLOCK);
  if (probe >= 0 || (errno != EAGAIN && errno != EBUSY))
  {
    if (probe >= 0)
      close(probe);
    snprintf(err, errlen, "the tenstorrent driver does not support exclusive device "
             "ownership; please update it");
    tt_device_close(dev);
    return -1;
  }

  memset(&info, 0, sizeof(info));
  info.in.output_size_bytes = sizeof(info.out);
  if (ioctl(dev->fd, TENSTORRENT_IOCTL_GET_DEVICE_INFO, &info) != 0 ||
      info.out.vendor_id != BH_VENDOR_ID || info.out.device_id != BH_DEVICE_ID)
  {
    snprintf(err, errlen, "%s is not a Blackhole device", path);
    tt_device_close(dev);
    return -1;
  }

  /* Ungate the Tensix tiles and raise their clock while we own the device;
     GDDR (MRISC PHYs) and the L2CPU are not used, so are not requested.
     The driver drops this request when the file is closed. */
  memset(&power, 0, sizeof(power));
  power.argsz = sizeof(power);
  power.validity = TT_POWER_VALIDITY(4, 0);
  power.power_flags = TT_POWER_FLAG_MAX_AI_CLK | TT_POWER_FLAG_TENSIX_ENABLE;
  if (ioctl(dev->fd, TENSTORRENT_IOCTL_SET_POWER_STATE, &power) != 0)
  {
    snprintf(err, errlen, "cannot power up %s: %s", path, strerror(errno));
    tt_device_close(dev);
    return -1;
  }

  if (window_allocate(dev, &dev->reg_window) != 0)
  {
    snprintf(err, errlen, "cannot get a TLB window on %s: %s", path, strerror(errno));
    tt_device_close(dev);
    return -1;
  }
  if (enumerate_tiles(dev, max_tiles, err, errlen) < 0)
  {
    tt_device_close(dev);
    return -1;
  }
  /* One window per tile, as far as the driver has windows to spare */
  for (t = 0; t < dev->tile_count; t++)
  {
    if (window_allocate(dev, &dev->l1_window[t]) != 0)
      break;
  }
  if (!t)
  {
    snprintf(err, errlen, "cannot get TLB windows on %s: %s", path, strerror(errno));
    tt_device_close(dev);
    return -1;
  }
  dev->tile_count = t;
  if (map_tiles(dev, err, errlen) < 0)
  {
    tt_device_close(dev);
    return -1;
  }
  snprintf(dev->name, sizeof(dev->name), "Blackhole, %u Tensix tiles", dev->tile_count);
  return 0;
}

void tt_device_close(tt_device_t *dev)
{
  unsigned t;

  /* Leave no firmware running behind us */
  if (dev->is_sim ? dev->dl != NULL : dev->reg_window.map != NULL)
  {
    for (t = 0; t < dev->tile_count; t++)
      tt_reg_write32(dev, t, TT_REG_SOFT_RESET, TT_SOFT_RESET_ALL);
  }
  if (dev->is_sim)
  {
    if (dev->dl)
    {
      dev->sim_exit();
      dlclose(dev->dl);
    }
  }
  else
  {
    for (t = 0; t < TT_MAX_TILES; t++)
      window_free(dev, &dev->l1_window[t]);
    window_free(dev, &dev->reg_window);
    if (dev->fd >= 0)
      close(dev->fd);
  }
  memset(dev, 0, sizeof(*dev));
  dev->fd = -1;
}
