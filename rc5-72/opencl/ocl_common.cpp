/*
* Copyright distributed.net 2009-2026 - All Rights Reserved
* For use in distributed.net projects only.
* Any other distribution or use of this source violates copyright.
*
* $Id: ocl_common.cpp 2014/08/11 22:18:25 ertyu Exp $
*/
#include "cputypes.h"
#include "ocl_common.h"
#include "base64.h"
#include <stdlib.h>
#include <string.h>
#include "stdio.h"

//rc5-72 test
#define P 0xB7E15163
#define Q 0x9E3779B9

#define SHL(x, s) ((u32) ((x) << ((s) & 31)))
#define SHR(x, s) ((u32) ((x) >> (32 - ((s) & 31))))
#define ROTL(x, s) ((u32) (SHL((x), (s)) | SHR((x), (s))))
#define ROTL3(x) ROTL(x, 3)

#if 0
inline u32 swap32(u32 a)
{
  u32 t=(a>>24)|(a<<24);
  t|=(a&0x00ff0000)>>8;
  t|=(a&0x0000ff00)<<8;
  return t;
}
#endif

s32 rc5_72_unit_func_ansi_ref (RC5_72UnitWork *rc5_72unitwork)
{
  u32 i, j, k;
  u32 A, B;
  u32 S[26];
  u32 L[3];
  u32 kiter = 1;
  while (kiter--)
  {
    L[2] = rc5_72unitwork->L0.hi;
    L[1] = rc5_72unitwork->L0.mid;
    L[0] = rc5_72unitwork->L0.lo;
    for (S[0] = P, i = 1; i < 26; i++)
      S[i] = S[i-1] + Q;
      
    for (A = B = i = j = k = 0;
         k < 3*26; k++, i = (i + 1) % 26, j = (j + 1) % 3)
    {
      A = S[i] = ROTL3(S[i]+(A+B));
      B = L[j] = ROTL(L[j]+(A+B),(A+B));
    }
    A = rc5_72unitwork->plain.lo + S[0];
    B = rc5_72unitwork->plain.hi + S[1];
    for (i=1; i<=12; i++)
    {
      A = ROTL(A^B,B)+S[2*i];
      B = ROTL(B^A,A)+S[2*i+1];
    }
    if (A == rc5_72unitwork->cypher.lo)
    {
        return RESULT_FOUND;
    }
  }
  return RESULT_NOTHING;
}


//Key increment

void key_incr(u32 *hi, u32 *mid, u32 *lo, u32 incr)
{
  *hi+=incr;
  u32 ad=*hi>>8;
  if(*hi<incr)
    ad+=0x01000000;
  //u32 t_m=swap32(*mid)+ad;
  u32 t_m=SWAP32(*mid)+ad;
  //u32 t_l=swap32(*lo);
  u32 t_l=SWAP32(*lo);
  if(t_m<ad)
    t_l++;

  *hi=*hi&0xff;
  //*mid=swap32(t_m);
  *mid=SWAP32(t_m);
  //*lo=swap32(t_l);
  *lo=SWAP32(t_l);
}

//Subtract two 72-bit numbers res=N1-N2
//Assumptions:
//N1>=N2, res<2^32
u32 sub72(u32 m1, u32 h1, u32 m2, u32 h2)
{
  //m1=swap32(m1); 
  m1=SWAP32(m1); 
  //m2=swap32(m2);
  m2=SWAP32(m2);

  u32 h3=h1-h2;
  u32 borrow=(h3>h1)?1:0;
  u32 m3=m1-m2-borrow;

  return (m3<<8)|(h3&0xff);
}

cl_int ocl_diagnose(cl_int result, const char *where, ocl_context_t *cont)
{
  if (result!=CL_SUCCESS)
  {
    if (where && cont)
      Log("Error %s on device %u\n", where, cont->clientDeviceNo);
    Log("Error code %d, message: %s\n", result, clStrError(result));
  }

  return result;
}

const char* clStrError(cl_int status)
{
  switch (status) {
    case CL_SUCCESS:                            return "Success!";
    case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
    case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:                   return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
    case CL_MAP_FAILURE:                        return "Map failure";
    case CL_INVALID_VALUE:                      return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
    case CL_INVALID_PLATFORM:                   return "Invalid platform";
    case CL_INVALID_DEVICE:                     return "Invalid device";
    case CL_INVALID_CONTEXT:                    return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
    case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
    case CL_INVALID_SAMPLER:                    return "Invalid sampler";
    case CL_INVALID_BINARY:                     return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
    case CL_INVALID_PROGRAM:                    return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
    case CL_INVALID_KERNEL:                     return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
    case CL_INVALID_EVENT:                      return "Invalid event";
    case CL_INVALID_OPERATION:                  return "Invalid operation";
    case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
    default: return "Unknown";
  }
}

#define BUFFER_INCREMENT 4096

static unsigned char* Decompress(const unsigned char *inbuf, unsigned length)
{
  unsigned char *outbuf=NULL;
  unsigned buflen=BUFFER_INCREMENT;
  unsigned used=0;
  unsigned todo=length;
  outbuf=(unsigned char*)malloc(BUFFER_INCREMENT);
  if(outbuf==NULL)
    return NULL;

  while(todo) {
    if(*inbuf & 0x80) {    //compressed
      unsigned len=*inbuf&0x7f;

      if(buflen <= (used+len)) {
        buflen += BUFFER_INCREMENT;
        outbuf = (unsigned char*)realloc(outbuf,buflen);
        if(outbuf==NULL)
          break;
      }

      unsigned off=*(inbuf+1)+*(inbuf+2)*256;
      inbuf += 3;
      todo -= 3;
      for( ; len>0; len--) {
        outbuf[used] = outbuf[used-off];
        used++;
      }
    } else {  //plain
      unsigned len=*inbuf&0x7f;
      if(buflen <= (used+len)) {
        buflen += BUFFER_INCREMENT;
        outbuf = (unsigned char*)realloc(outbuf,buflen);
        if(outbuf==NULL)
          break;
      }
      todo--;
      inbuf++;
      for( ; len>0; len--) {
        outbuf[used++] = *inbuf;
        inbuf++; todo--;
      }
    }
  }
  outbuf[used]=0;
  return outbuf;
}

bool GetNVComputeCapability(cl_device_id device, int &sm_version)
{
  cl_uint vendor;

  sm_version = 0;
  if (clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, NULL) != CL_SUCCESS)
    return false;
  if (vendor != 0x10DE)
    return false; // Not NVIDIA

  cl_uint sm_major = 0, sm_minor = 0;
  cl_int maj_status = clGetDeviceInfo(device, 0x4000, sizeof(sm_major), &sm_major, NULL);
  cl_int min_status = clGetDeviceInfo(device, 0x4001, sizeof(sm_minor), &sm_minor, NULL);
  
  if (maj_status == CL_SUCCESS && min_status == CL_SUCCESS)
  {
    sm_version = (int)sm_major * 10 + (int)sm_minor;

    //LogTo(LOGTO_FILE, "Queried NVIDIA SM_%d from CL_DEVICE_COMPUTE_CAPABILITY\n", sm_version);

    return true;
  }
 
  //LogTo(LOGTO_FILE, "Failed to query NVIDIA SM version\n");

  return false;
}

bool GetNVRegisterHint(int sm_version, int &regs_2pipe, int &regs_4pipe)
{
  regs_2pipe = 0;
  regs_4pipe = 0;

  if (sm_version >= 50 && sm_version <= 72) // Maxwell, Pascal, Volta
  {
    regs_2pipe = 80;
    regs_4pipe = 80;
  }
  else if (sm_version >= 75 && sm_version <= 89) // Turing, Ampere, Ada Lovelace
  {
    regs_2pipe = 64;
    regs_4pipe = 128;
  }
  else if (sm_version >= 90 && sm_version <= 121) // Hopper, Blackwell
  {
    regs_2pipe = 80;
    regs_4pipe = 80;
  }

  if (regs_2pipe > 0 && regs_4pipe > 0)
    return true;   
    
  //LogTo(LOGTO_FILE, "Failed to find NVIDIA SM_%d in register hint lookup table\n", sm_version);

  return false;
}

bool GetAMDComputeCapability(cl_device_id device, int &gfx_hex)
{
  gfx_hex = 0x0000;

  cl_uint vendor;
  if (clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, NULL) != CL_SUCCESS)
    return false;
    
  if (vendor != 0x1002) // AMD PCI Vendor ID
    return false;

  bool found_gfx_ver = false;

  // 1. Try to parse the official OpenCL Device Name String
  char nameBuffer[256] = {0};
  if (clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(nameBuffer), nameBuffer, NULL) == CL_SUCCESS)
  {
    char *gfx_ptr = strstr(nameBuffer, "gfx");
    if (gfx_ptr == NULL)
      gfx_ptr = strstr(nameBuffer, "GFX");

    if (gfx_ptr != NULL)
    {
      char *end_ptr = NULL;
      long parsed_val = strtol(gfx_ptr + 3, &end_ptr, 16);
      
      // Ensure strtol actually advanced (meaning it consumed valid hex digits)
      if (parsed_val > 0 && end_ptr != (gfx_ptr + 3)) 
      {
        gfx_hex = (int)parsed_val;
        found_gfx_ver = true;

        //LogTo(LOGTO_FILE, "Parsed an AMD gfx%x from CL_DEVICE_NAME\n", gfx_hex);
      }
    }
  }

  // 2. Fallback: Query AMD's Proprietary Integer Extensions (CL_DEVICE_GFXIP_MAJOR/MINOR_AMD)
  if (!found_gfx_ver)
  {
    cl_uint gfxip_major = 0;
    cl_uint gfxip_minor = 0;
        
    cl_int maj_status = clGetDeviceInfo(device, 0x404A, sizeof(gfxip_major), &gfxip_major, NULL); // CL_DEVICE_GFXIP_MAJOR_AMD
    cl_int min_status = clGetDeviceInfo(device, 0x404B, sizeof(gfxip_minor), &gfxip_minor, NULL); // CL_DEVICE_GFXIP_MINOR_AMD
        
    if (maj_status == CL_SUCCESS && min_status == CL_SUCCESS)
    {
      // Convert decimal Major/Minor pair to standard hex format
      gfx_hex = (((gfxip_major / 10) << 12) | ((gfxip_major % 10) << 8) | (gfxip_minor << 4));
      found_gfx_ver = true;

      //LogTo(LOGTO_FILE, "Queried an AMD gfx%x from CL_DEVICE_GFXIP fallback\n", gfx_hex);
    }
  }

  if (found_gfx_ver)
    return true;
  
  //LogTo(LOGTO_FILE, "Failed to parse or query an AMD gfx compute id\n");
  return false;
}

bool GetAMDRegisterHint(int gfx_hex, int &regs_2pipe, int &regs_4pipe) // VGPRs
{
  regs_2pipe = 0;
  regs_4pipe = 0;

  if (gfx_hex >= 0x600 && gfx_hex < 0x900)
  {
    // GCN 1.0 - 4.0 (gfx600 to gfx8xx)
    // 256 VGPRs per SIMD - 4 VGPR Granularity - 10 waves max
    regs_2pipe = 64;    // 4 waves - 40% occupancy
    regs_4pipe = 128;   // 2 waves - 20% occupancy
  }
  else if (gfx_hex >= 0x900 && gfx_hex <= 0x90c && gfx_hex != 0x908 && gfx_hex != 0x90a)
  {
    // Vega / GCN 5.0 (gfx900 to gfx90c)
    // 256 VGPRs per SIMD - 4 VGPR Granularity - 10 waves max
    regs_2pipe = 64;    // 4 waves - 40% occupancy
    regs_4pipe = 128;   // 2 waves - 20% occupancy
  }
  else if (gfx_hex >= 0x1000 && gfx_hex < 0x1030)
  {
    // RDNA 1 (gfx1010 to gfx1013)
    // 1024 VGPRs per SIMD - 8 VGPR Granularity - 20 waves max
    regs_2pipe = 64;    // 16 waves - 80% occupancy
    regs_4pipe = 128;   //  8 waves - 40% occupancy
  }
  else if (gfx_hex >= 0x1030 && gfx_hex < 0x1100)
  {
    // RDNA 2 (gfx1030 to gfx1036)
    // 1024 VGPRs per SIMD - 16 VGPR Granularity - 16 waves max
    regs_2pipe = 64;    // 16 waves - 100% occupancy
    regs_4pipe = 128;   //  8 waves - 50% occupancy
  }
  else if (gfx_hex >= 0x1100 && gfx_hex < 0x1200)
  {
    // RDNA 3 (gfx11xx)
    // 1536 VGPRs per SIMD - 24 VGPR Granularity - 16 waves max
    regs_2pipe = 96;    // 16 waves - 100% occupancy
    regs_4pipe = 144;   // 10 waves - 63% occupancy
  }
  else if (gfx_hex >= 0x1200 && gfx_hex < 0x1300)
  {
    // RDNA 4 (gfx12xx)
    // 1536 VGPRs per SIMD - 24 VGPR Granularity - 16 waves max
    // (Dynamic VGPR mode is disabled)
    regs_2pipe = 96;    // 16 waves - 100% occupancy
    regs_4pipe = 144;   // 10 waves - 63% occupancy
  }

  if (regs_2pipe > 0 && regs_4pipe > 0)
  {
    return true;   
  }

  //LogTo(LOGTO_FILE, "Failed to find AMD gfx%x in VGPR register hint lookup table\n", gfx_hex);

  return false;
}

bool BuildCLProgram(ocl_context_t *cont, const char* programText, const char *kernelName)
{
  char *decoded_src = (char*)malloc(strlen(programText)+1);
  if (!decoded_src)
    return false;

  u32 decoded_len = base64_decode(decoded_src, programText, strlen(programText), strlen(programText));
  unsigned char *decompressed_src = Decompress((unsigned char*)decoded_src,decoded_len);
  free(decoded_src);
  if (decompressed_src == NULL)
    return false;

  cl_int status;
  cont->program = clCreateProgramWithSource(cont->clcontext, 1, (const char**)&decompressed_src, NULL, &status);
  free(decompressed_src);
  if (status == CL_SUCCESS)
  {
    const char *clOption = "-cl-std=CL1.1";  // support older macOS
    char buildOptions[80];
    int nv_sm = 0;
    int amd_gfx = 0;

    if (GetNVComputeCapability(cont->deviceID, nv_sm))  // NVIDIA
    {
      char nvOption1[16] = "";

      snprintf(nvOption1, sizeof(nvOption1), "-D NV_SM=%d", nv_sm); // SM version

      char nvOption2[32] = "";
      int nv_maxreg2, nv_maxreg4;

      if (GetNVRegisterHint(nv_sm, nv_maxreg2, nv_maxreg4)) // maximize ILP and occupancy
      {
        if (strstr(kernelName, "2pipe"))
          snprintf(nvOption2, sizeof(nvOption2), "-cl-nv-maxrregcount=%d", nv_maxreg2);
        else if (strstr(kernelName, "4pipe"))
          snprintf(nvOption2, sizeof(nvOption2), "-cl-nv-maxrregcount=%d", nv_maxreg4);
      }
   
      snprintf(buildOptions, sizeof(buildOptions), "%s %s %s", clOption, nvOption1, nvOption2); // NVIDIA build options
    }
    else if (GetAMDComputeCapability(cont->deviceID, amd_gfx)) // AMD
    {
      char amdOption1[24] = "";

      snprintf(amdOption1, sizeof(amdOption1), "-D AMD_GFX=0x%x", amd_gfx); // SM version

      char amdOption2[16] = "";
      int amd_maxreg2, amd_maxreg4;

      if (GetAMDRegisterHint(amd_gfx, amd_maxreg2, amd_maxreg4)) // maximize ILP and occupancy
      {
        if (strstr(kernelName, "2pipe"))
          snprintf(amdOption2, sizeof(amdOption2), "-D AMD_VGPR=%d", amd_maxreg2);
        else if (strstr(kernelName, "4pipe"))
          snprintf(amdOption2, sizeof(amdOption2), "-D AMD_VGPR=%d", amd_maxreg4);
      }

      snprintf(buildOptions, sizeof(buildOptions), "%s %s", clOption, amdOption1, amdOption2); // AMD build options
    }
    else // GENERIC
      snprintf(buildOptions, sizeof(buildOptions), "%s", clOption);  // Generic manufacturer build options

    status = clBuildProgram(cont->program, 1, &cont->deviceID, buildOptions, NULL, NULL);
    
    /* 
    if (status == CL_SUCCESS)
      LogTo(LOGTO_FILE, "clBuildProgram() successful for kernel: %s %s\n", kernelName, buildOptions);
    else if (status != CL_SUCCESS)
      LogTo(LOGTO_FILE, "clBuildProgram() failed for kernel: %s %s\n", kernelName, buildOptions);
    */

    if (status != CL_SUCCESS)  // fallback
    {
      status = clBuildProgram(cont->program, 1, &cont->deviceID, NULL, NULL, NULL); // fallback build options
      
      /*
      if (status == CL_SUCCESS)
        LogTo(LOGTO_FILE, "clBuildProgram() successful for kernel: %s with fallback build options\n", kernelName);
      else if (status != CL_SUCCESS)
        LogTo(LOGTO_FILE, "clBuildProgram() failed for kernel: %s with fallback build options\n", kernelName);
      */
    }
  }

  if (ocl_diagnose(status, "building cl program", cont) != CL_SUCCESS)
  {
    //static char buf[0x10001]={0};
    size_t log_size;

    clGetProgramBuildInfo( cont->program,
                           cont->deviceID,
                           CL_PROGRAM_BUILD_LOG,
                           0,
                           NULL,
                           &log_size );

    char *buf = (char *) malloc(log_size);
    clGetProgramBuildInfo( cont->program,
                           cont->deviceID,
                           CL_PROGRAM_BUILD_LOG,
                           log_size,
                           buf,
                           NULL );
    
    buf[log_size - 1] = '\0';
    Log("Build log returned %ld bytes\n", (long)log_size);
    LogRaw("Build Log:\n");
    LogRaw("%s\n", buf);
   
    free(buf);

    return false;
  }

  cont->kernel = clCreateKernel(cont->program, kernelName, &status);
  if (ocl_diagnose(status, "building kernel", cont) != CL_SUCCESS)
    return false;

  return true;
}
