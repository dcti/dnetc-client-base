/*
* Copyright distributed.net 2009-2014 - All Rights Reserved
* For use in distributed.net projects only.
* Any other distribution or use of this source violates copyright.
*
* $Id: ocl_info.h 2014/08/19 22:18:25 gkhanna Exp $
*/

#ifndef OCL_COMMON_H
#define OCL_COMMON_H

#include "cputypes.h"
#if (CLIENT_OS == OS_WIN64) || (CLIENT_OS == OS_WIN32) || \
    (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_LINUX) || \
    (CLIENT_OS == OS_ANDROID)
#include <CL/cl.h>
#elif (CLIENT_OS == OS_MACOSX)
#include <OpenCL/opencl.h>
#endif
#include "ccoreio.h"
#include "logstuff.h"  // LogScreen()
#include "triggers.h"
#include "ocl_context.h"

#define P 0xB7E15163
#define Q 0x9E3779B9

#define SHL(x, s) ((u32) ((x) << ((s) & 31)))
#define SHR(x, s) ((u32) ((x) >> (32 - ((s) & 31))))
#define ROTL(x, s) ((u32) (SHL((x), (s)) | SHR((x), (s))))
#define ROTL3(x) ROTL(x, 3)
#define SWAP32(x) (((((x) >> 24) | ((x) << 24)) | (((x) & 0x00ff0000) >> 8)) | (((x) & 0x0000ff00 ) << 8))

void key_incr(u32 *hi, u32 *mid, u32 *lo, u32 incr);
u32 sub72(u32 m1, u32 h1, u32 m2, u32 h2);
//inline u32 swap32(u32 a);
void OCLReinitializeDevice(ocl_context_t *cont);

s32 rc5_72_unit_func_ansi_ref (RC5_72UnitWork *rc5_72unitwork);
cl_int ocl_diagnose(cl_int result, const char *where, ocl_context_t *cont);
const char* clStrError(cl_int status);
bool BuildCLProgram(ocl_context_t *cont, const char* programText, const char *kernelName);

#endif //OCL_COMMON_H
