/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: cli.prefix.h,v 1.2 2002/09/02 00:35:51 andreasb Exp $
*/

/* ========================== build specific stuff ===================== */
//#define MAC_FBA /* if building FBA */
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

