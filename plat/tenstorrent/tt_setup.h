/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id:
*/

#ifndef TT_SETUP_H
#define TT_SETUP_H

#include "cputypes.h"
#include "tt_device.h"

/* Environment variable naming the libttsim.so to run on instead of the
   devices in /dev/tenstorrent */
#define TT_SIM_ENV "DNETC_TTSIM"

/* Optional environment variable limiting the Tensix tiles used per device,
   for bring-up and debugging */
#define TT_MAX_TILES_ENV "DNETC_TT_MAX_TILES"

#define TT_MAX_DEVICES 32

/*
 * InitializeTenstorrent() returns < 0 on a fatal error, >= 0 otherwise
 * (even if no devices were found).
 * getTenstorrentDeviceCount() returns < 0 if initialization failed or was
 * not done, otherwise the number of devices.
 */
int  InitializeTenstorrent(void);
int  getTenstorrentDeviceCount(void);
u32  getTenstorrentDeviceFreq(int device);
long getTenstorrentRawProcessorID(int device, const char **cpuname);
void TenstorrentPrintExtendedInfo(int device);

/* Device access for cores. A device must only be used between
   tt_lock_device() and tt_unlock_device(). Returns NULL for a bad index. */
tt_device_t *tt_lock_device(int device);
void tt_unlock_device(int device);

#endif /* TT_SETUP_H */
