/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: r72stream-common.cpp,v 1.16 2010/09/13 16:23:07 sla Exp $
*/

#include "r72stream-common.h"
#include "base64.h"

static inline u32 swap32(u32 a)
{
  u32 t=(a>>24)|(a<<24);
  t|=(a&0x00ff0000)>>8;
  t|=(a&0x0000ff00)<<8;
  return t;
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


fastlock_t ATIstream_cMutex;
fastlock_t ATIstream_RDPMutex;

int ati_RC_error;
void AMDInitMutex()
{
  fastlock_init(&ATIstream_cMutex);
  fastlock_init(&ATIstream_RDPMutex);
  ati_RC_error=0;
}

u32 setRemoteConnectionFlag()
{
  if(isRemoteSession())
  {
    fastlock_lock(&ATIstream_RDPMutex);
    if(!ati_RC_error)
    {
      LogScreen("Remote connection is active. Paused\n");
      ati_RC_error=1;
    }
    fastlock_unlock(&ATIstream_RDPMutex);
    return 1;
  }
  return 0;
}

u32 checkRemoteConnectionFlag()
{
  if(ati_RC_error)
  {
    if(isRemoteSession())
      return 1;

    fastlock_lock(&ATIstream_RDPMutex);
    if(ati_RC_error)
    {
      LogScreen("Remote connection is no longer active. Resuming\n");

      for(int i=0; i<AMD_STREAM_MAX_GPUS; i++)
        CContext[i].coreID=CORE_NONE;
      ati_RC_error=0;
    }
    fastlock_unlock(&ATIstream_RDPMutex);
    return 0;
  }
  return 0;
}

#define BUFFER_INCREMENT 32768

unsigned char* Decompress(unsigned char *inbuf, unsigned length)
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

CALresult runCompiler(CALcontext *ctx, CALimage *image, CALmodule *module, CALchar *src, CALtarget target, bool verbose)
{
  CALobject s_obj=0;
  CALresult result;
  int Device = 999;  // todo

  fastlock_lock(&ATIstream_cMutex);
  result = calclCompile(&s_obj, CAL_LANGUAGE_IL, src, target);
  // RC5-72 code blindly tries to compile few cores until it finds one suitable
  // for current GPU architecture, so print error only when requested (by OGR code).
  if (verbose)
    ati_verbose_cl(result, "compiling program", Device);
  if (result == CAL_RESULT_OK)
  {
    result = calclLink(image, &s_obj, 1);
    if (ati_verbose_cl(result, "linking program", Device) == CAL_RESULT_OK)
      result = ati_verbose( calModuleLoad(module, *ctx, *image), "loading module", Device );
    ati_verbose_cl( calclFreeObject(s_obj), "freeing compiled object", Device);
  }

  fastlock_unlock(&ATIstream_cMutex);
  return result;
}

CALresult compileProgram(CALcontext *ctx, CALimage *image, CALmodule *module, CALchar *src, CALtarget target, bool globalFlag)
{
  CALresult result;

  char *decoded_src=(char*)malloc(strlen(src)+1);
  if(decoded_src==NULL)
    return CAL_RESULT_ERROR;
  u32 decoded_len=base64_decode(decoded_src,src,strlen(src),strlen(src));
  unsigned char *decompressed_src=Decompress((unsigned char*)decoded_src,decoded_len);
  free(decoded_src);
  if(decompressed_src==NULL)
    return CAL_RESULT_ERROR;

  //replace '#' with appropriate symbol
  char *p,*tempB;
  tempB=(char*)malloc(strlen((const char*)decompressed_src)+1);
  if(!tempB)
    return CAL_RESULT_ERROR;
  strcpy(tempB,(const char*)decompressed_src);
  do {
    p = strchr(tempB,'#');
    if(p) {
      if(globalFlag) {
        *p = ' ';
      } else {
        *p = ';';
        *(p+1) = ';';     //TODO:HACK!!
      }
    }
  } while(p);

  result = runCompiler(ctx, image, module, (CALchar*)tempB, target);

  free(decompressed_src);
  free(tempB);
  return result;
}
