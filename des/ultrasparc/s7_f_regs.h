/* s7_f_regs.h v3.0 */
/* upon entry to this, g_k1, g_k6, g_a5, gt_k3, g_a4, g_a2 are live */
/* This allocates g_k1 to GT5, g_k6 to  */
/* also registers FT1, FT2, FT3, FT4 are live from s6 */

/* $Log: s7_f_regs.h,v $
/* Revision 1.1.1.1  1998/06/14 14:23:50  remi
/* Initial integration.
/* */


#ifdef MANUAL_REGISTER_ALLOCATION

#ifdef USE_SAME_FLOAT_REGISTERS
/* avoid using FT1, FT2, FT3, FT4, FT5, FT6 for above live variables */
#define GT1 FT1
#define GT2 FT2
#define GT3 FT3
#define GT4 FT4
#define GT5 FT5
#define GT6 FT6
#define GT7 FT7
#define GT8 FT8
#define GT9 FT9
#define GT10 FT10
#define GT11 FT11
#define GT12 FT12
#define GT13 FT13
#define GT14 FT14
#define GT15 FT15
#define GT16 FT16

#else /* USE_SAME_FLOAT_REGISTERS */

    register INNER_LOOP_FSLICE GT1, GT2, GT3, GT4, GT5, GT6, GT7, GT8;
    register INNER_LOOP_FSLICE GT9, GT10, GT11, GT12, GT13, GT14, GT15;
    register INNER_LOOP_FSLICE GT16;

#endif /* USE_SAME_FLOAT_REGISTERS */

    INNER_LOOP_SLICE *gkey1, *gkey6;
    INNER_LOOP_SLICE *gt_key1, *gt_key2, *gt_key3, *gt_key4, *gt_key5, *gt_key6;

#define gt_a2	GT16
#define g_a2	GT16

#define gt_k2	GT15
/* FREE */
#define gx4	GT15
#define gx24	GT15

#define gt_k3	GT14
/* FREE */
#define gx8	GT14
#define gx9	GT14
#define gx23	GT14

#define gt_k4	GT13
/* FREE */
#define gx3	GT13
#define gx25	GT13
#define gx26	GT13
/* FREE */
#define gx31	GT13
#define gx32	GT13

#define g_k6	GT12
/* FREE */
#define gx7	GT12
#define gx33	GT12

#define gt_k5	GT11
/* FREE */
#define gx16	GT11
#define gx17	GT11
#define gx18	GT11
#define gx19	GT11
/* FREE */
#define gx27	GT11
#define gx28	GT11
#define gx29	GT11
#define gx30	GT11

/* FREE */
#define gx12	GT10
#define gx13	GT10
#define gx14	GT10
/* FREE */
#define gx34	GT10
#define gx35	GT10
#define gx36	GT10
#define gx37	GT10

/* FREE */
#define g_a3	GT9
#define gx38	GT9
#define gx39	GT9
#define gx40	GT9
#define gx41	GT9

#define g_a5	GT8
#define gx42	GT8
#define gx43	GT8

/* FREE */
#define gx5	GT7
#define gx6	GT7
/* FREE */
#define Preloadg_1 GT7

#define gt_a4	GT6
#define g_a4	GT6
#define gx20	GT6
#define gx21	GT6
#define gx22	GT6
#define gx48	GT6
#define gx49	GT6
#define gx50	GT6
#define gx51	GT6

#define g_k1	GT5
/* FREE */
#define Preloadg_2 GT5

/* FREE */
#define gx15	GT4
#define gx44	GT4
#define gx45	GT4
#define gx46	GT4
#define gx47	GT4

/* FREE */
#define gx1	GT3
#define gx2	GT3
/* FREE */
#define Preloadg_3 GT3

/* FREE */
#define gx10	GT2
#define gx11	GT2
/* FREE */
#define g_a1	GT2

/* FREE */
#define g_a6	GT1
/* FREE */
#define Preloadg_4 GT1

#else /* MANUAL_REGISTER_ALLOCATION */

    INNER_LOOP_FSLICE gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8;
    INNER_LOOP_FSLICE gx9, gx10, gx11, gx12, gx13, gx14, gx15, gx16;
    INNER_LOOP_FSLICE gx17, gx18, gx19, gx20, gx21, gx22, gx23, gx24;
    INNER_LOOP_FSLICE gx25, gx26, gx27, gx28, gx29, gx30, gx31, gx32;
    INNER_LOOP_FSLICE gx33, gx34, gx35, gx36, gx37, gx38, gx39, gx40;
    INNER_LOOP_FSLICE gx41, gx42, gx43, gx44, gx45, gx46, gx47, gx48;
    INNER_LOOP_FSLICE gx49, gx50, gx51, gx52, gx53;
    INNER_LOOP_FSLICE Preloadg_1, Preloadg_2, Preloadg_3, Preloadg_4;

    INNER_LOOP_FSLICE g_k1, g_k2, g_k3, g_k6;
    INNER_LOOP_FSLICE *gkey1, *gkey2, *gkey3, *gkey6;

    INNER_LOOP_FSLICE g_a1, g_a2, g_a3, g_a4, g_a5, g_a6;

    INNER_LOOP_FSLICE gt_a1, gt_a2, gt_a3, gt_a4, gt_a5;
    INNER_LOOP_FSLICE gt_k1, gt_k2, gt_k3, gt_k4, gt_k5, gt_k6;
    INNER_LOOP_FSLICE *gt_key1, *gt_key2, *gt_key3, *gt_key4, *gt_key5, *gt_key6;

#endif /* MANUAL_REGISTER_ALLOCATION */

/* end of s7_f.h */
