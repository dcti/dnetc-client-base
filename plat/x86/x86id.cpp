/*
 * Copyright distributed.net 1997-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: x86id.cpp,v 1.10 2008/10/17 00:25:53 snikkel Exp $
 *
 * Gold mine of technical details:
 *    http://datasheets.chipdb.org/
 *    http://sandpile.org/   
 */

#include <string.h>
#include <stdio.h>
#include "ccoreio.h"
#include "cputypes.h"
#include "cpucheck.h"
#include "logstuff.h"
#include "x86id.h"


/*----------------------------------------------------------------------------*/

/*
** A container to obtain results from 'cpuid'
*/
union PageInfos {
  char string[12];
  struct {
    u32 ebx;
    u32 edx;
    u32 ecx;
    u32 eax;
  } regs;
};


/* 0x00000001:EDX */
#define X86_HAS_MMX       (1 << 23)
#define X86_HAS_SSE       (1 << 25)
#define X86_HAS_SSE2      (1 << 26)
#define X86_HAS_HTT       (1 << 28)

/* 0x00000001:ECX */
#define X86_HAS_SSE3      (1 <<  0)
#define X86_HAS_SSSE3     (1 <<  9)
#define X86_HAS_SSE4_1    (1 << 19)
#define X86_HAS_SSE4_2    (1 << 20)

/* 0x80000001:EDX (Intel) */
#define X86_HAS_EM64T     (1 << 29)

/* 0x80000001:EDX (AMD) */
#define AMD_HAS_MMX_EXT   (1 << 22)
#define AMD_HAS_3DNOW     (1 << 31)
#define AMD_HAS_3DNOW_EXT (1 << 30)
#define AMD_HAS_LM        (1 << 29)     /* 64-bit mode */

/* 0x80000001:EDX (Cyrix) */
#define CYRIX_HAS_MMX_EXT (1 << 24)

#if defined(__amd64__) || defined(__x86_64__)
  #if defined(__GNUC__)
  static inline s32 x86getid(void)
  {
    return -1;
  }

  static u32 x86cpuid(u32 page, union PageInfos* infos)
  {
    u32 _ax, _bx, _cx, _dx;

    asm volatile ("xorl %1,%1\n\t"
                  "xorl %2,%2\n\t"
                  "xorl %3,%3\n\t"
                  "cpuid"
                  : "=a"(_ax), "=b"(_bx), "=c"(_cx), "=d"(_dx)
                  : "0"(page)
                  : /* nothing */
                 );
    infos->regs.eax = _ax;
    infos->regs.ebx = _bx;
    infos->regs.ecx = _cx;
    infos->regs.edx = _dx;
    return _ax;
  }
  #endif
#elif (CLIENT_CPU == CPU_X86)
  #if (CLIENT_OS == OS_LINUX) && !defined(__ELF__)
    extern "C" s32 x86getid(void) asm ("x86getid");
    extern "C" u32 x86cpuid(u32 page, union PageInfos* infos) asm ("x86cpuid");
  #else
    // x86getid()/x86cpuid() can destroy all registers except ebx/esi/edi/ebp
    // => must be declared as "cdecl" to allow compiler save necessary
    //    registers.
    extern "C" s32 CDECL x86getid(void);
    extern "C" u32 CDECL x86cpuid(u32 page, union PageInfos* infos);
  #endif

#endif

#if (CLIENT_OS == OS_LINUX) && !defined(__ELF__)
  extern "C" ui64 x86rdtsc( void ) asm ("x86rdtsc");
#else
  extern "C" ui64 CDECL x86rdtsc( void );
#endif
ui64 x86ReadTSC(void)
{
  return x86rdtsc();
}

/*----------------------------------------------------------------------------*/

static s32 x86id_fixup(s32 x86id_result)
{
#if (CLIENT_OS == OS_LINUX)
  if (x86id_result == MAKE_CPUID(VENDOR_CYRIX, 0, 4, 0, 0)) /* Cyrix indeterminate */
  {
    FILE *file = fopen("/proc/cpuinfo", "r");
    if (file)
    {
      int vendor_id = 0, family = 0, model = 0;
      char buf[128]; 
      while (fgets(buf, sizeof(buf)-1, file))
      {
        char *p; int c;
        buf[sizeof(buf)-1] = '\0';
        p = strchr(buf, '\n');
        if (p) 
          *p = '\0';
        else
        {
          c = 1;
          while (c != EOF && c != '\n')
            c = fgetc(file);
          p = &buf[sizeof(buf-1)]; /* "" */
        }      
        c = 0;
        while (buf[c] && buf[c] != ':')
          c++;
        if (buf[c] == ':') 
          p = &buf[c+1];
        while (c > 0 && (buf[c-1]==' ' || buf[c-1] == '\t'))
          c--;
        buf[c] = '\0';
        while (*p == ' ' || *p == '\t')
          p++;
        c = 0;
        /* printf("key='%s', val='%s'\n", buf, p); */
        while (p[c] && p[c] != ' ' && p[c] != '\t')
          c++;
        p[c] = '\0';  
        
        if (strcmp(buf,"vendor_id") == 0)
        {
          if (strcmp(p,"CyrixInstead") == 0) /* we only care about this one */
            vendor_id = VENDOR_CYRIX;
          else
            break;
        }
        else if (strcmp(buf, "model name")==0)
        {
          if (strncmp(p, "5x86", 4)!=0)
            break;
          family = 4; model = 9; 
          /* linux simulates 5x86 as fam=4,mod=1,step=5, x86ident() as 4,9,x */
        }  
      }
      fclose(file);
      if (vendor_id == VENDOR_CYRIX && family == 4 && model == 9)
        return MAKE_CPUID(vendor_id, 0, family, model, 0);
    } /* if (file) */
  } /* if (cyrix indeterminate) */
#endif /* (CLIENT_OS == OS_LINUX) */
  return x86id_result;
}  


/*
** Search the specified cache descriptor (one byte) in the descriptors list
** (4-byte). Returns 0 if not found, otherwise -1.
*/
static int x86FindCacheDescrInReg(u32 reg, int descr)
{
  int i;
  for (i = 0; i < 4; i++) {
    if (descr == (int) (reg & 0xFF))
      return -1;
    reg >>= 4;
  }
  return 0;
}


/*
** Search the specified cache descriptor (one byte) in all descriptors lists
** (four registers). Returns 0 if not found, otherwise -1.
*/
static int x86FindCacheDescriptor(union PageInfos *infos, int descriptor)
{
  int result = -1;

  if ((infos->regs.eax & 0x80000000U) == 0) {
    result &= x86FindCacheDescrInReg(infos->regs.eax & 0xFFFFFF00U, descriptor);
  }
  if ((infos->regs.ebx & 0x80000000U) == 0) {
    result &= x86FindCacheDescrInReg(infos->regs.ebx, descriptor);
  }
  if ((infos->regs.ecx & 0x80000000U) == 0) {
    result &= x86FindCacheDescrInReg(infos->regs.ecx, descriptor);
  }
  if ((infos->regs.edx & 0x80000000U) == 0) {
    result &= x86FindCacheDescrInReg(infos->regs.edx, descriptor);
  }
  
  return result;
}


/*
** Obtain a generic ID (Unknown vendor).
*/
static u32 x86GetDefaultId(u32 maxfunc)
{
  union PageInfos infos;
  u32 cpuid = 0;

  if (maxfunc >= 1) {
    int family, model, step;
    
    u32 signature = x86cpuid(0x00000001, &infos);
    family  = FIELD_FAMILY(signature);
    model   = FIELD_MODEL(signature);
    step    = FIELD_STEPPING(signature);
    
    cpuid = MAKE_CPUID(VENDOR_UNKNOWN, 0, family, model, step);
  }
  
  return cpuid;
}


static u32 x86GetIntelId(u32 maxfunc)
{
  union PageInfos infos;
  u32 cpuid = 0;

  if (maxfunc >= 1) {
    int extfam, extmod, family, model, step, brandid;

    u32 signature = x86cpuid(0x00000001, &infos);
    extfam  = FIELD_EXT_FAMILY(signature);
    extmod  = FIELD_EXT_MODEL(signature);
    family  = FIELD_FAMILY(signature);
    model   = FIELD_MODEL(signature);
    step    = FIELD_STEPPING(signature);
    brandid = FIELD_BRAND_ID(infos.regs.ebx);

    /* That's how Intel do it */
    if (family == 15) {
      family += extfam;
    }
    if (family == 6 || family == 15) {
      model |= (extmod << 4);
    }

    /*
    ** Make a composite identifier to differenciate between P-II/Xeon/Celeron
    ** model 5. We use the "brandid" field to store this information.
    ** ID = 0x1000605x : Pentium II PE
    ** ID = 0x1010605x : Celeron (Covington)
    ** ID = 0x1020605x : Celeron-A (Mendocino)
    ** ID = 0x1030605x : Pentium II/Xeon
    ** ID = 0x1040605x : Pentium II Xeon
    ** BUG: On MP systems we should make sure that successive calls to the cpuid
    **      instruction run on a single CPU.
    */
    if (family == 6 && model == 5 && maxfunc >= 2) {
      int count = x86cpuid(0x00000002, &infos) & 0xFF;

      do {
        if (x86FindCacheDescriptor(&infos, 0x40)) {   /* No L2 cache */
          brandid = 1;    /* Celeron */
          break;
        }
        if (x86FindCacheDescriptor(&infos, 0x41)) {   /* 128-KB L2 cache */
          brandid = 2;    /* Celeron-A */
          break;
        }
        if (x86FindCacheDescriptor(&infos, 0x43)) {   /* 512-KB L2 cache */
          brandid = 3;    /* Pentium II/Xeon */
          break;
        }
        if (x86FindCacheDescriptor(&infos, 0x44)) {   /* 1-MB L2 cache */
          brandid = 4;    /* Pentium II Xeon */
          break;
        }
        if (x86FindCacheDescriptor(&infos, 0x45)) {   /* 2-MB L2 cache */
          brandid = 4;    /* Pentium II Xeon */
          break;
        }
        x86cpuid(0x00000002, &infos);
      } while (--count > 0);
    }

    /*
     ** Make a composite identifier to differenciate between P-II/Xeon/Celeron
     ** model 6. We use the "brandid" field to store this information.
     ** ID = 0x1000606x : Pentium II (Mendocino)
     ** ID = 0x1010606x : Celeron-A (Mendocino)
     ** BUG: On MP systems we should make sure that successive calls to the cpuid
     **      instruction run on a single CPU.
     */
    if (family == 6 && model == 6 && maxfunc >= 2) {
      int count = x86cpuid(0x00000002, &infos) & 0xFF;
      
      do {
        if (x86FindCacheDescriptor(&infos, 0x41)) {   /* 128-KB L2 cache */
          brandid = 1;    /* Celeron-A */
          break;
        }
        x86cpuid(0x00000002, &infos);
      } while (--count > 0);
    }
    
    /*
     ** Make a composite identifier to differenciate between P-II/Xeon
     ** model 7. We use the "brandid" field to store this information.
     ** ID = 0x1000607x : Pentium II (Katmai)
     ** ID = 0x1010607x : Pentium II Xeon (Katmai)
     ** BUG: On MP systems we should make sure that successive calls to the cpuid
     **      instruction run on a single CPU.
     */
    if (family == 6 && model == 7 && maxfunc >= 2) {
      int count = x86cpuid(0x00000002, &infos) & 0xFF;
      
      do {
        if (x86FindCacheDescriptor(&infos, 0x44)) {   /* 1-MB L2 cache */
          brandid = 1;    /* Pentium II Xeon */
          break;
        }
        if (x86FindCacheDescriptor(&infos, 0x45)) {   /* 2-MB L2 cache */
          brandid = 1;    /* Pentium II Xeon */
          break;
        }
        x86cpuid(0x00000002, &infos);
      } while (--count > 0);
    }
    
    cpuid = MAKE_CPUID(VENDOR_INTEL, brandid, family, model, step);
  }

  return cpuid;
}


/* NOTE : Early models are reported using their regular signature. Starting
**        from the K8 family, AMD uses the model field to encode model-specific
**        features that are not relevant. Because all K8 models are grouped
**        under the same 'detectedtype' identifier, our brand ID field is
**        customized to represent the brand model (Athlon, Opteron, etc)
*/
static u32 x86GetAmdId(u32 maxfunc)
{
  union PageInfos infos;
  u32 cpuid = 0;

  if (maxfunc >= 1) {
    int family, model, step, brandid = 0;
    
    u32 signature = x86cpuid(0x00000001, &infos);
    family  = FIELD_FAMILY(signature);
    model   = FIELD_MODEL(signature);
    step    = FIELD_STEPPING(signature);

    if (family == 15) {     /* AMD K8 variants */
      int extbrandid = 0;
      brandid = FIELD_BRAND_ID(infos.regs.ebx);
      family += FIELD_EXT_FAMILY(signature);
      model  |= FIELD_EXT_MODEL(signature) << 4;

      maxfunc = x86cpuid(0x80000000U, &infos);
      if (maxfunc >= 0x80000001U) {
        x86cpuid(0x80000001U, &infos);
        extbrandid = infos.regs.ebx & 0xFFFF;
      }

      if (family == 15) {
        if (brandid) {
          brandid = (brandid >> 5) & 0x1C;
        }
        else if (extbrandid) {
          brandid = (extbrandid >> 6) & 0xFF;
          /* Should be 0x3FF, but our field is only 8-bit wide */
        }

        if (brandid >= 0x29 && brandid <= 0x3A)
          brandid = AMDM15_DC_OPTERON;
        else if (brandid >= 0x0C && brandid <= 0x17)
          brandid = AMDM15_OPTERON;
        else switch (brandid) {
           case 0x04: brandid = AMDM15_ATHLON_64; break;
           case 0x05: brandid = AMDM15_ATHLON_64_X2_DC; break;
           case 0x08:
           case 0x09: brandid = AMDM15_MOBILE_ATHLON_64; break;
           case 0x0A:
           case 0x0B: brandid = AMDM15_TURION_64; break;
           case 0x18: brandid = AMDM15_ATHLON_64; break;
           case 0x1D:
           case 0x1E: brandid = AMDM15_MOBILE_ATHLON_XP; break;
           case 0x20: brandid = AMDM15_ATHLON_XP; break;
           case 0x21: brandid = AMDM15_MOBILE_SEMPRON; break;
           case 0x22: brandid = AMDM15_SEMPRON; break;
           case 0x23: brandid = AMDM15_MOBILE_SEMPRON; break;
           case 0x24: brandid = AMDM15_ATHLON_64_FX; break;
           case 0x26: brandid = AMDM15_SEMPRON; break;
           default:   brandid = AMDM15_UNKNOWN; break;
        }
        model = 0;    /* Scrub the model number (irrelevant) */
      }
      else if (family == 16) {
        int code = 0;
        int pkg  = 0;

        if (maxfunc >= 0x80000001U) {
          code = (infos.regs.ebx >> 11) & 0x0F;    /* AMD:string1 */
          pkg  = (infos.regs.ebx >> 28) & 0x0F;
        }

        if (maxfunc >= 0x80000008U) {
          x86cpuid(0x80000008U, &infos);
          code |= (infos.regs.ecx & 0xFF) << 4;    /* Create a composite code with AMD:NC */
        }

        if(pkg == 0) {      /* Socket Fr2 */
          switch (code) {   /* Code := Nibble1=NC, Nibble2=string1 */
             case 0x10:
             case 0x11: brandid = AMDM16_DC_OPTERON; break;
             case 0x30:
             case 0x31: brandid = AMDM16_QC_OPTERON; break;
             case 0x32:
             case 0x33:
             case 0x34: brandid = AMDM16_EMBEDDED_OPTERON; break;
             default:   brandid = AMDM16_UNKNOWN; break;
          }
        }
        else if (pkg == 1) {  /* Socket AM2r2 */
          /* As of Feb. 16, 2008, there's only a single entry */
          if (code == 0x32)
            brandid = AMDM16_PHENOM;
          else
            brandid = AMDM16_UNKNOWN;
        }
        model = 0;    /* Scrub the model number (irrelevant) */
      }
      /* Otherwise we don't know much yet, so we'd better don't touch */
    }
    cpuid = MAKE_CPUID(VENDOR_AMD, brandid, family, model, step);
  }

  return cpuid;
}


static u32 x86GetTransmetaId(u32 maxfunc)
{
  /* - Basic identification -
  ** Crusoe TM3x00   : ID = 0x5042
  ** Crusoe TM5x00   : ID = 0x5043
  ** Efficeon TM8x00 : ID = 0xF02x / 0xF03x
  */
  return x86GetDefaultId(maxfunc) | (VENDOR_TRANSMETA << 28);
}


/*
** Note : The CPUID instruction is disabled by default on the oldest chips.
*/
static u32 x86GetCyrixId(u32 maxfunc)
{
  /* - Basic identification -
  ** ID = 0x402x : 5x86                     (source Cyrix)
  ** ID = 0x404x : MediaGX
  ** ID = 0x409x : 5x86                     (misc sources - May apply to older models...)
  ** ID = 0x502x : 6x86
  ** ID = 0x503x : 6x86-M1
  ** ID = 0x504x : MediaGX MMX / GXm
  ** ID = 0x504x : Geode GXm / Geode GXLV   (VendorID == "CyrixInstead" !!)
  ** ID = 0x600x : 6x86MX / M II
  ** ID = 0x605x : Cyrix III Joshua
  */
  return x86GetDefaultId(maxfunc) | (VENDOR_CYRIX << 28);
}


static u32 x86GetUmcId(u32 maxfunc)
{
  /* - Basic identification -
  ** ID = 0x401x : U5D
  ** ID = 0x402x : U5S
  */
  return x86GetDefaultId(maxfunc) | (VENDOR_UMC << 28);
}


static u32 x86GetNexGenId(u32 maxfunc)
{
  /* - Basic identification -
  ** ID = 0x500x : Nx586
  */
  return x86GetDefaultId(maxfunc) | (VENDOR_NEXGEN << 28);
}


/*
** Centaur/IDT/VIA (VendorID == "CentaurHauls")
*/
static u32 x86GetCentaurId(u32 maxfunc)
{
  /* - Basic identification -
  ** ID = 0x504x : IDT WinChip C6 / Centaur C2
  ** ID = 0x508x : IDT WinChip 2 / Centaur C3
  ** ID = 0x509x : IDT WinChip 3 / Centaur C6
  ** ID = 0x606x : VIA C3 Samuel
  ** ID = 0x607x : VIA C3 Samuel 2 / VIA Eden ESP (Ezra)
  ** ID = 0x608x : VIA C3 Ezra-T
  ** ID = 0x609x : VIA C3 Nehemia / VIA C3-M Nehemia
  **             / VIA Eden ESP (Ezra/Nehemia) / VIA Antaur
  ** ID = 0x60Ax : VIA C7 Esther
  */
  return x86GetDefaultId(maxfunc) | (VENDOR_CENTAUR << 28);
}


static u32 x86GetRiseId(u32 maxfunc)
{
  /* - Basic identification -
  ** ID = 0x500x / 0x502x : mP6 iDragon
  ** ID = 0x508x / 0x509x : mP6 iDragon II
  */
  return x86GetDefaultId(maxfunc) | (VENDOR_RISE << 28);
}


/*
** VendorID = "Geode by NSC"
*/
static u32 x86GetNscId(u32 maxfunc)
{
  /* - Basic identification -
  ** ID = 0x504x : Geode GX1
  ** ID = 0x505x : Geode GX2
  ** ID = 0x50Ax : Geode LX
  */
  return x86GetDefaultId(maxfunc) | (VENDOR_NSC << 28);
}


static u32 x86GetSisId(u32 maxfunc)
{
  /* - Basic identification -
  ** ID = 0x500x : 55x
  */
  return x86GetDefaultId(maxfunc) | (VENDOR_SIS << 28);
}


/*
** Display eax/ebx/ecx/edx content (in that order) for the specified function.
*/
static void x86DumpFunctions(u32 function)
{
  union PageInfos infos;
  u32 maxfunc;

  maxfunc = x86cpuid(function, &infos);
  if ((maxfunc & 0xFFFFFFF0U) == 0x500) {    /* Fix-up P5 Step-A */
    maxfunc = 1;
  }

  if (maxfunc >= function) {
    do {
      LogRaw("F_%08X : %08X %08X %08X %08X\n", function, infos.regs.eax, infos.regs.ebx,
              infos.regs.ecx, infos.regs.edx);
      x86cpuid(++function, &infos);
    } while (function <= maxfunc);
  }
}


/*----------------------------------------------------------------------------*/

const char* x86GetVendorName(u32 detectedtype)
{
  switch (ID_VENDOR_CODE(detectedtype)) {
     case VENDOR_INTEL:     return "Intel";
     case VENDOR_TRANSMETA: return "Transmeta";
     case VENDOR_NSC:       return "National Semiconductor";
     case VENDOR_AMD:       return "AMD";
     case VENDOR_CYRIX:     return "Cyrix";
     case VENDOR_NEXGEN:    return "NexGen";
     case VENDOR_CENTAUR:   return (ID_FAMILY(detectedtype) == 5) ? "IDT" : "VIA";
     case VENDOR_UMC:       return "UMC";
     case VENDOR_RISE:      return "Rise";
     case VENDOR_SIS:       return "SiS";
     default: break;
  }
  return "";
}


u32 x86GetDetectedType(void)
{
  static u32 detectedtype = 0;

  if (0 == detectedtype) {
    s32 id = x86id_fixup(x86getid());

    if (id != -1) {
      detectedtype = id;      /* Simple ID, mostly hand-made */
    }
    else {                    /* Support cpuid */
      union PageInfos infos;
      u32 maxfunc = x86cpuid(0, &infos);

      if ((maxfunc & 0xFFFFFFF0U) == 0x500) {    /* Fix-up P5 Step-A */
        maxfunc = 1;
        strncpy(infos.string, "GenuineIntel", 12);
      }

      if (strncmp(infos.string, "GenuineIntel", 12) == 0)
        detectedtype = x86GetIntelId(maxfunc);
      else if (strncmp(infos.string, "GenuineTMx86", 12) == 0)
        detectedtype = x86GetTransmetaId(maxfunc);
      else if (strncmp(infos.string, "AuthenticAMD", 12) == 0)
        detectedtype = x86GetAmdId(maxfunc);
      else if (strncmp(infos.string, "CyrixInstead", 12) == 0)
        detectedtype = x86GetCyrixId(maxfunc);
      else if (strncmp(infos.string, "UMC UMC UMC ", 12) == 0)
        detectedtype = x86GetUmcId(maxfunc);
      else if (strncmp(infos.string, "NexGenDriven", 12) == 0)
        detectedtype = x86GetNexGenId(maxfunc);
      else if (strncmp(infos.string, "CentaurHauls", 12) == 0)
        detectedtype = x86GetCentaurId(maxfunc);
      else if (strncmp(infos.string, "RiseRiseRise", 12) == 0)
        detectedtype = x86GetRiseId(maxfunc);
      else if (strncmp(infos.string, "Geode by NSC", 12) == 0)
        detectedtype = x86GetNscId(maxfunc);
      else if (strncmp(infos.string, "SiS SiS SiS ", 12) == 0)
        detectedtype = x86GetSisId(maxfunc);
      else
        detectedtype = x86GetDefaultId(maxfunc);
    }
  }

  return detectedtype;
}


u32 x86GetFeatures(void)
{
  u32 features = 0;

  if (x86getid() == -1) {
    union PageInfos infos;
    u32 cpuid = x86GetDetectedType();
    u32 maxfunc = x86cpuid(0, &infos);

    if (maxfunc >= 0x00000001) {
      u32 fecx, fedx;

      x86cpuid(0x00000001, &infos);
      fedx = infos.regs.edx;
      fecx = infos.regs.ecx;

      /* Ignore bogus Intel Pentium MMX P55C model 4 stepping 5 */
      if (cpuid != 0x10005045 && (fedx & X86_HAS_MMX) != 0) {
        features |= CPU_F_MMX;
      }
      if ((fedx & X86_HAS_SSE) != 0) {
        features |= CPU_F_SSE;
      }
      if ((fedx & X86_HAS_SSE2) != 0) {
        features |= CPU_F_SSE2;
      }
      if ((fecx & X86_HAS_SSE3) != 0) {
        features |= CPU_F_SSE3;
      }
      if (ID_VENDOR_CODE(cpuid) == VENDOR_INTEL) {
        if ((infos.regs.ebx & 0xFF0000) > 1 && (fedx & X86_HAS_HTT) != 0) {
          features |= CPU_F_HYPERTHREAD;      /* Hyperthreading enabled */
        }
        if ((fecx & X86_HAS_SSSE3) != 0) {
          features |= CPU_F_SSSE3;
        }
        if ((fecx & X86_HAS_SSE4_1) != 0) {
          features |= CPU_F_SSE4_1;
        }
        if ((fecx & X86_HAS_SSE4_2) != 0) {
          features |= CPU_F_SSE4_2;
        }
      }
    }

    maxfunc = x86cpuid(0x80000000U, &infos);
    if (maxfunc >= 0x80000001U) {
      u32 flags;

      x86cpuid(0x80000001U, &infos);
      flags = infos.regs.edx;

      if (ID_VENDOR_CODE(cpuid) == VENDOR_INTEL) {
        if ((flags & X86_HAS_EM64T) != 0) {
          features |= CPU_F_EM64T;
        }
      }
      if (ID_VENDOR_CODE(cpuid) == VENDOR_AMD || ID_VENDOR_CODE(cpuid) == VENDOR_CENTAUR) {
        if ((flags & X86_HAS_MMX) != 0) {
          features |= CPU_F_MMX;
        }
        if ((flags & AMD_HAS_3DNOW) != 0) {
          features |= CPU_F_3DNOW;
        }
      }
      if (ID_VENDOR_CODE(cpuid) == VENDOR_AMD) {
        if ((flags & AMD_HAS_MMX_EXT) != 0) {
          features |= CPU_F_AMD_MMX_PLUS;
        }
        if ((flags & AMD_HAS_3DNOW_EXT) != 0) {
          features |= CPU_F_3DNOW_PLUS;
        }
        if ((flags & AMD_HAS_LM) != 0) {
          features |= CPU_F_AMD64;
        }
      }
      if (ID_VENDOR_CODE(cpuid) == VENDOR_CYRIX || ID_VENDOR_CODE(cpuid) == VENDOR_NSC) {
        if ((flags & X86_HAS_MMX) != 0) {
          features |= CPU_F_MMX;
        }
        if ((flags & CYRIX_HAS_MMX_EXT) != 0) {
          features |= CPU_F_CYRIX_MMX_PLUS;
        }
      }
    }
  }

  return features;
}


void x86ShowInfos(void)
{
  if (x86getid() == -1) {         /* cpuid is available */
    union PageInfos vendor, brand;
    union {
      struct {u32 eax; u32 ebx; u32 ecx; u32 edx;} regs[3];
      char brandname[48];
    } brandbuffer;
    u32 maxfunc;

    LogRaw("\nRaw processor informations :\n");
    maxfunc = x86cpuid(0, &vendor);
    if ((maxfunc & 0xFFFFFFF0U) == 0x500) {    /* Fix-up P5 Step-A */
      strncpy(vendor.string, "GenuineIntel", 12);
    }

    if (maxfunc > 0) {
      char vendorbuff[13];
      strncpy(vendorbuff, vendor.string, 12);
      vendorbuff[12] = '\0';
      LogRaw(" Vendor ID : \"%s\"\n", vendorbuff);
    }

    maxfunc = x86cpuid(0x80000000U, &brand);
    if (maxfunc >= 0x80000004U) {
      int i;
      int p = 0;
      char last = ' ';

      for (i = 0; i <= 2; i++) {
        x86cpuid(0x80000002U + i, &brand);
        brandbuffer.regs[i].eax = brand.regs.eax;
        brandbuffer.regs[i].ebx = brand.regs.ebx;
        brandbuffer.regs[i].ecx = brand.regs.ecx;
        brandbuffer.regs[i].edx = brand.regs.edx;
      }
      brandbuffer.brandname[47] = '\0';   /* Better be safe than sorry... */

      /* Remove extra white spaces (Intel appears to like them a lot...) */
      for (i = 0; i < 48; i++) {
        char c = brandbuffer.brandname[i];
        if (c != ' ' || last != ' ') {
          brandbuffer.brandname[p++] = last = c;
        }
      }
      LogRaw("  Brand ID : \"%s\"\n", brandbuffer.brandname);
    }

    x86DumpFunctions(0);
    if (strncmp(vendor.string, "CentaurHauls", 12) == 0) {
      /* Show Centaur/IDT/VIA specific functions */
      x86DumpFunctions(0xC0000000U);
    }
    if (strncmp(vendor.string, "GenuineTMx86", 12) == 0) {
      /* Show Transmeta-specific functions */
      x86DumpFunctions(0x80860000U);
    }

    x86DumpFunctions(0x80000000U);
  }
}
