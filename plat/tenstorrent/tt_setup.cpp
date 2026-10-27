/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id:
*/

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "cputypes.h"
#include "logstuff.h"
#include "tt_setup.h"

static int num_devices = -12345;
static tt_device_t devices[TT_MAX_DEVICES];
static pthread_mutex_t device_locks[TT_MAX_DEVICES];
static char device_paths[TT_MAX_DEVICES][64];
static const char *sim_path;
static unsigned max_tiles = TT_MAX_TILES;

static void close_devices(void)
{
  int i;

  for (i = 0; i < num_devices; i++)
  {
    pthread_mutex_lock(&device_locks[i]);
    tt_device_close(&devices[i]);
    pthread_mutex_unlock(&device_locks[i]);
  }
  num_devices = 0;
}

static int compare_ints(const void *a, const void *b)
{
  return *(const int *)a - *(const int *)b;
}

/* Opens the Blackhole devices in /dev/tenstorrent, in numeric order */
static void open_devices(void)
{
  int ids[TT_MAX_DEVICES * 4], n = 0, i;
  DIR *dir = opendir("/dev/tenstorrent");
  struct dirent *entry;
  char err[256];

  if (!dir)
    return;
  while ((entry = readdir(dir)) != NULL && n < (int)(sizeof(ids) / sizeof(ids[0])))
  {
    char *end;
    long id = strtol(entry->d_name, &end, 10);
    if (end != entry->d_name && !*end && id >= 0)
      ids[n++] = (int)id;
  }
  closedir(dir);
  qsort(ids, n, sizeof(ids[0]), compare_ints);

  for (i = 0; i < n && num_devices < TT_MAX_DEVICES; i++)
  {
    char *path = device_paths[num_devices];
    snprintf(path, sizeof(device_paths[0]), "/dev/tenstorrent/%d", ids[i]);
    if (tt_device_open(&devices[num_devices], path, max_tiles, err, sizeof(err)) < 0)
      Log("Tenstorrent: skipping %s: %s\n", path, err);
    else
      num_devices++;
  }
}

int InitializeTenstorrent(void)
{
  char err[256];
  int i;

  if (num_devices != -12345)
    return num_devices;
  num_devices = 0;
  for (i = 0; i < TT_MAX_DEVICES; i++)
    pthread_mutex_init(&device_locks[i], NULL);
  if (getenv(TT_MAX_TILES_ENV))
  {
    long n = atol(getenv(TT_MAX_TILES_ENV));
    if (n > 0 && n < TT_MAX_TILES)
      max_tiles = (unsigned)n;
  }

  sim_path = getenv(TT_SIM_ENV);
  if (sim_path && *sim_path)
  {
    /* The simulator is a process-wide singleton presenting one device */
    if (tt_device_open_sim(&devices[0], sim_path, max_tiles, err, sizeof(err)) < 0)
      Log("Tenstorrent: unable to open simulator %s: %s\n", sim_path, err);
    else
      num_devices = 1;
  }
  else
    open_devices();

  if (num_devices > 0)
    atexit(close_devices);
  else if (!sim_path || !*sim_path)
    Log("Tenstorrent: to use the simulator instead, set " TT_SIM_ENV " to the path of libttsim.so\n");
  return num_devices;
}

int getTenstorrentDeviceCount(void)
{
  return num_devices;
}

u32 getTenstorrentDeviceFreq(int device)
{
  /* Simulated time has no fixed relation to wall clock time */
  if (device < 0 || device >= num_devices || devices[device].is_sim)
    return 0;
  return TT_AICLK_MHZ;
}

long getTenstorrentRawProcessorID(int device, const char **cpuname)
{
  if (device < 0 || device >= num_devices)
  {
    *cpuname = NULL;
    return -1;
  }
  *cpuname = devices[device].name;
  return 0xB140;
}

void TenstorrentPrintExtendedInfo(int device)
{
  if (device < 0 || device >= num_devices)
    return;
  LogRaw("Tensix tiles: %u\n", devices[device].tile_count);
  if (devices[device].is_sim)
    LogRaw("Simulator: %s\n", sim_path);
  else
    LogRaw("Device: %s\n", device_paths[device]);
}

tt_device_t *tt_lock_device(int device)
{
  if (device < 0 || device >= num_devices)
  {
    Log("INTERNAL ERROR: bad Tenstorrent device index %d (detected %d)!\n", device, num_devices);
    return NULL;
  }
  pthread_mutex_lock(&device_locks[device]);
  return &devices[device];
}

void tt_unlock_device(int device)
{
  pthread_mutex_unlock(&device_locks[device]);
}
