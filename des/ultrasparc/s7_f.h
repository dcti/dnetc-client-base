/* s7_f.h v3.0 */

/*
 * $Log: s7_f.h,v $
 * Revision 1.2  1998/06/14 15:19:23  remi
 * Avoid tons of warnings due to a brain-dead CVS.
 *
 * Revision 1.1.1.1  1998/06/14 14:23:50  remi
 * Initial integration.
 *
 */


/* This is a floating point version of the s7 sbox */

    ASM_F_D_LOAD (g_a3, in[S73]);		/* PIPELINE */

    ASM_F_D_LOAD (g_a6, in[S76]);
    ASM_F_XOR (g_a3, g_a3, gt_k3);		/* PIPELINE */

	ASM_F_AND (gx1, g_a2, g_a4);		/*             M  */

	ASM_F_XOR (gx2, gx1, g_a5);		/*             M  */

	ASM_F_AND (gx3, g_a4, gx2);		/*             M  */

	ASM_F_XOR_NOT (gx4, gx3, g_a2);		/*             M  */

	ASM_F_OR (gx10, g_a2, g_a4);		/*               O */

	ASM_F_OR (gx11, gx10, g_a5);		/*               O */

	ASM_F_AND_NOT (gx12, g_a5, g_a2);	/*              N */

	ASM_F_OR (gx13, g_a3, gx12);		/*              N */

	ASM_F_XOR (gx14, gx11, gx13);		/*              N */

	ASM_F_XOR (g_a6, g_a6, g_k6);

	ASM_F_AND (gx5, g_a3, gx4);		/*             M  */

	ASM_F_XOR_NOT (gx24, g_a4, gx4);	/*           K    */

	ASM_F_AND_NOT (gx20, g_a4, g_a3);	/*          J     */

	ASM_F_XOR_NOT (gx7, g_a3, gx5);		/*             M  */

	ASM_F_XOR (gx6, gx2, gx5);		/*               O */

	ASM_F_XOR (gx15, gx3, gx6);		/*            L   */

    ASM_F_D_LOAD (g_a1, in[S71]);
	ASM_F_OR (gx16, g_a6, gx15);		/*            L   */

	ASM_F_AND (gx8, g_a6, gx7);		/*             M  */

	ASM_F_XOR (gx9, gx6, gx8);		/*             M  */

	ASM_F_OR (gx42, g_a5, gx20);		/*    D           */

	ASM_F_AND_NOT (gx21, g_a2, gx20);	/*          J     */

	ASM_F_AND (gx22, g_a6, gx21);		/*          J     */

	ASM_F_XOR (gx17, gx14, gx16);		/*            L N */

	ASM_F_XOR (gx44, g_a2, gx15);		/* A              */

    ASM_F_D_LOAD (Preloadg_1, merge[D71]);	/*                */
	ASM_F_XOR (g_a1, g_a1, g_k1);

    ASM_F_D_LOAD (Preloadg_2, merge[D72]);	/*                */
	ASM_F_AND (gx18, g_a1, gx17);		/*            L   */

    ASM_F_D_LOAD (Preloadg_3, merge[D73]);	/*                */
	ASM_F_XOR (gx19, gx9, gx18);		/*            LM  */

	ASM_F_XOR_NOT (gx23, gx9, gx22);	/*          J     */

	ASM_F_AND (gx48, g_a3, gx22);		/*   C       J    */

	ASM_F_XOR (Preloadg_1, Preloadg_1, gx19); /*            L   */

	ASM_F_OR (gx34, g_a2, gx24);		/*      F         */

	ASM_F_XOR_NOT (gx35, gx34, gx19);	/*      F         */

	ASM_F_OR (gx36, g_a6, gx35);		/*      F         */

	ASM_F_XOR (gx27, g_a3, gx3);		/*         I      */

	ASM_F_AND (gx28, gx27, g_a2);		/*         I      */

	ASM_F_AND_NOT (gx29, g_a6, gx28);	/*         I      */

	ASM_F_OR (gx25, g_a3, gx3);		/*           K    */

	ASM_F_XOR (gx26, gx24, gx25);		/*           K    */

	ASM_F_AND_NOT (gx38, gx26, g_a3);	/*     E G        */

	ASM_F_AND_NOT (gx45, gx24, gx44);	/* A              */

	ASM_F_AND (gx46, g_a6, gx45);		/* A              */

    ASM_F_D_LOAD (Preloadg_4, merge[D74]);	/*                */
	ASM_F_XOR (gx30, gx26, gx29);		/*         I K    */

	ASM_F_XOR_NOT (gx33, gx7, gx30);	/*        H       */

	ASM_F_OR (gx31, g_a1, gx30);		/*         I      */

	ASM_F_XOR (gx32, gx23, gx31);		/*         IJ     */

	ASM_F_XOR (Preloadg_2, Preloadg_2, gx32); /*         I      */

	ASM_F_XOR (gx43, gx42, gx33);		/*    D           */

	ASM_F_XOR (gx37, gx33, gx36);		/*      F H       */

	ASM_F_XOR (gx49, gx48, gx46);		/* A              */

	ASM_F_OR (gx39, gx38, gx30);		/*     E          */

	ASM_F_OR_NOT (gx40, g_a1, gx39);	/*     E          */

	ASM_F_XOR (gx41, gx37, gx40);		/*     EF         */

	ASM_F_XOR (Preloadg_3, Preloadg_3, gx41); /*     E          */

	ASM_F_XOR (gx47, gx43, gx46);		/*  B D           */

    ASM_F_D_STORE (out[D71], Preloadg_1);	/*            L   */
	ASM_F_OR (gx50, g_a1, gx49);		/* A C            */

    ASM_F_D_STORE (out[D72], Preloadg_2);	/*         I      */
	ASM_F_XOR (gx51, gx47, gx50);		/* AB             */

    ASM_F_D_STORE (out[D73], Preloadg_3);	/*     E          */
	ASM_F_XOR (Preloadg_4, Preloadg_4, gx51); /* A              */

    ASM_F_D_STORE (out[D74], Preloadg_4);	/* A              */

    ASM_COMMENT_END_INCLUDE (end_of_s7_f);

#include "s0.h"
/* end of s7_f.h */
