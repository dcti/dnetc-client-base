/*
* Copyright distributed.net 2009 - All Rights Reserved
* For use in distributed.net projects only.
* Any other distribution or use of this source violates copyright.
*
* $Id: 
*/
#include "cputypes.h"
#include "ocl_common.h"
#include "base64.h"
#include <stdlib.h>
#include <string.h>

//rc5-72 test
#define P 0xB7E15163
#define Q 0x9E3779B9

#define SHL(x, s) ((u32) ((x) << ((s) & 31)))
#define SHR(x, s) ((u32) ((x) >> (32 - ((s) & 31))))
#define ROTL(x, s) ((u32) (SHL((x), (s)) | SHR((x), (s))))
#define ROTL3(x) ROTL(x, 3)

inline u32 swap32(u32 a)
{
  u32 t=(a>>24)|(a<<24);
  t|=(a&0x00ff0000)>>8;
  t|=(a&0x0000ff00)<<8;
  return t;
}

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
  u32 t_m=swap32(*mid)+ad;
  u32 t_l=swap32(*lo);
  if(t_m<ad)
    t_l++;

  *hi=*hi&0xff;
  *mid=swap32(t_m);
  *lo=swap32(t_l);
}

//Subtract two 72-bit numbers res=N1-N2
//Assumptions:
//N1>=N2, res<2^32
u32 sub72(u32 m1, u32 h1, u32 m2, u32 h2)
{
  m1=swap32(m1); 
  m2=swap32(m2);

  u32 h3=h1-h2;
  u32 borrow=(h3>h1)?1:0;
  u32 m3=m1-m2-borrow;

  return (m3<<8)|(h3&0xff);
}

cl_int ocl_diagnose(cl_int result, const char *where, u32 DeviceIndex)
{
  if(result!=CL_SUCCESS)
  {
    Log("Error %s on device %u\n", where, DeviceIndex);
    Log("Error code %d, message: %s\n", result, clStrError(result));
  }

  return result;
}

char* clStrError(cl_int status)
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


bool BuildCLProgram(unsigned deviceID, const char* programText, const char *kernelName)
{
  char *decoded_src=(char*)malloc(strlen(programText)+1);
  if(!decoded_src)
	  return false;

  u32 decoded_len=base64_decode(decoded_src, programText, strlen(programText), strlen(programText));
  unsigned char *decompressed_src=Decompress((unsigned char*)decoded_src,decoded_len);
  free(decoded_src);
  if(decompressed_src==NULL)
    return false;

  cl_int status;
  ocl_context[deviceID].program = clCreateProgramWithSource(ocl_context[deviceID].clcontext, 1, (const char**)&decompressed_src, NULL, &status);
  free(decompressed_src);
  status |= clBuildProgram(ocl_context[deviceID].program, 1, &ocl_context[deviceID].deviceID, NULL, NULL, NULL);
  if(ocl_diagnose(status, "building cl program", deviceID) !=CL_SUCCESS)
  {
    static char buf[0x10000]={0};

    clGetProgramBuildInfo( ocl_context[deviceID].program,
                           ocl_context[deviceID].deviceID,
                           CL_PROGRAM_BUILD_LOG,
                           0x10000,
                           buf,
                           NULL );
    LogRaw(buf);
    
    return false;
  }

  ocl_context[deviceID].kernel = clCreateKernel(ocl_context[deviceID].program, kernelName, &status);
  if(ocl_diagnose(status, "building kernel", deviceID) !=CL_SUCCESS)
    return false;

  return true;
}