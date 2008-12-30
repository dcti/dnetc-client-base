/*
 * Copyright distributed.net 1997-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: x86id.h,v 1.4 2008/12/30 20:58:45 andreasb Exp $
 */

#ifndef client_x86id_H
#define client_x86id_H


/* Extended CPU ID format :
 *     bits [31, 28] : Brand code
 *     bits [27, 20] : Brand ID field
 *     bits [19, 12] : Family
 *     bits [11,  4] : Model
 *     bits [ 3,  0] : Stepping
 */


/*
 ** Vendor codes
 */
#define VENDOR_UNKNOWN    0
#define VENDOR_INTEL      1
#define VENDOR_TRANSMETA  2
#define VENDOR_NSC        3
#define VENDOR_AMD        4
#define VENDOR_CYRIX      5
#define VENDOR_NEXGEN     6
#define VENDOR_CENTAUR    7     /* Centaur/IDT/VIA */
#define VENDOR_UMC        8
#define VENDOR_RISE       9
#define VENDOR_SIS       10
#define VENDOR_MAX_      15


#define MAKE_CPUID(bcode,bid,fam,mod,step) (((bcode) << 28) | ((bid) << 20) \
| ((fam) << 12) | ((mod) << 4) | (step))

#define ID_VENDOR_CODE(id)  (((id) >> 28) & 0x0F)
#define ID_BRAND_ID(id)     (((id) >> 20) & 0xFF)
#define ID_FAMILY(id)       (((id) >> 12) & 0xFF)
#define ID_MODEL(id)        (((id) >>  4) & 0xFF)
#define ID_STEPPING(id)     ((id)         & 0x0F)

#define FIELD_EXT_FAMILY(sign) (((sign) >> 20) & 0xFF)
#define FIELD_EXT_MODEL(sign)  (((sign) >> 16) & 0x0F)
#define FIELD_FAMILY(sign)     (((sign) >>  8) & 0x0F)
#define FIELD_MODEL(sign)      (((sign) >>  4) & 0x0F)
#define FIELD_STEPPING(sign)   ((sign)         & 0x0F)

#define FIELD_BRAND_ID(sign)   ((sign) & 0xFF)


/*
 ** Pseudo BrandID for AMD CPUs
 */
enum AmdModel15 {
   AMDM15_UNKNOWN,
   AMDM15_ATHLON_64,               /* AMD Athlon(tm) 64 */
   AMDM15_ATHLON_64_X2_DC,         /* AMD Athlon(tm) 64 X2 Dual Core */
   AMDM15_MOBILE_ATHLON_64,        /* Mobile AMD Athlon(tm) 64 */
   AMDM15_TURION_64,               /* AMD Turion(tm) 64 Mobile Technology */
   AMDM15_OPTERON,                 /* AMD Opteron(tm) */
   AMDM15_MOBILE_ATHLON_XP,        /* Mobile AMD Athlon(tm) XP-M */
   AMDM15_ATHLON_XP,               /* AMD Athlon(tm) XP */
   AMDM15_MOBILE_SEMPRON,          /* Mobile AMD Sempron(tm) */
   AMDM15_SEMPRON,                 /* AMD Sempron(tm) */
   AMDM15_ATHLON_64_FX,            /* AMD Athlon(tm) 64 FX */
   AMDM15_DC_OPTERON,              /* Dual Core AMD Opteron(tm) */
   AMDM15_TURION_64_X2_DC,         /* AMD Turion(tm) 64 X2 Mobile Technology */
   AMDM15_LAST_MODEL = AMDM15_TURION_64_X2_DC
};

enum AmdModel16 {
   AMDM16_UNKNOWN = AMDM15_LAST_MODEL + 1,
   AMDM16_DC_OPTERON,              /* Dual-core AMD Opteron(tm) */
   AMDM16_QC_OPTERON,              /* Quad-core AMD Opteron(tm) */
   AMDM16_EMBEDDED_OPTERON,        /* Embedded AMD Opteron(tm) */
   AMDM16_PHENOM,                  /* AMD Phenom(tm) */
   AMDM16_LAST_MODEL = AMDM16_PHENOM
};


const char* x86GetVendorName(u32);
u32         x86GetDetectedType(void);
u32         x86GetFeatures(void);
ui64        x86ReadTSC(void);
void        x86ShowInfos(void);

#if (CLIENT_CPU == CPU_X86)
    extern "C" u32 x86ident_haveioperm; /* default is zero, referenced in cpucheck.cpp */
#endif

#endif	/* client_x86id_H */
