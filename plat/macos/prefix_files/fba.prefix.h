/* $Id: fba.prefix.h,v 1.1.2.1 2001/01/21 15:14:28 cyp Exp $ */

/* ========================== build specific stuff ===================== */
#define MAC_FBA /* if building FBA */
#define LURK
#define LURK_LISTENER
//#define BETA
//#define RESDEBUG
//#define DEBUG_PATHWORK
//#define TRACE
#if (__MRC__)
#define __inline inline
#endif

/* ======================= project specific enables ===================== */
#if __powerc //TARGET_CPU_PPC
  #define HAVE_OGR_CORES
#endif
#if 0 //CSC
  #define HAVE_CSC_CORES
#endif
#if 0 //DES
  #define MEGGS
  #define KWAN
  #define HAVE_DES_CORES
#endif

