/* s6_f.h v3.0 */

/*
 * $Log: s6_f.h,v $
 * Revision 1.2  1998/06/14 15:19:16  remi
 * Avoid tons of warnings due to a brain-dead CVS.
 *
 * Revision 1.1.1.1  1998/06/14 14:23:50  remi
 * Initial integration.
 *
 */


/* This is a floating point version of the s6 sbox */

	ASM_F_XOR (f_a4, f_a4, ft_k4);

	ASM_F_XOR (fx1, f_a5, f_a1);		/*              N */

	ASM_F_AND (fx3, f_a1, f_a6);		/*          J     */

	ASM_F_AND_NOT (fx4, fx3, f_a5);		/*              N */

	ASM_F_XOR (fx7, f_a6, fx3);		/*          J     */

	ASM_F_XOR (fx2, fx1, f_a6);		/*              N */

	ASM_F_OR (fx8, fx4, fx7);		/*          J     */

	ASM_F_AND_NOT (fx5, f_a4, fx4);		/*              N */

    ASM_F_D_LOAD (f_k2, ((INNER_LOOP_FSLICE *)fkey2)[0]); /* PIPELINE into s3_4 */
	ASM_F_AND_NOT (fx9, fx8, f_a4);		/*          J     */

    ASM_F_D_LOAD (f_k3, ((INNER_LOOP_FSLICE *)fkey3)[0]); /* PIPELINE into s3_4 */
	ASM_F_XOR (fx6, fx2, fx5);		/*              N */

	ASM_F_XOR (fx10, fx7, fx9);		/*          J     */

    ASM_F_D_LOAD (f_a2, in[S62]);		/* PIPELINE into s3_4 */
	ASM_F_OR (fx13, f_a6, fx6);		/*             M  */

	ASM_F_OR (fx15, fx4, fx10);		/*          J     */

	ASM_F_XOR (f_a2, f_a2, f_k2);

	ASM_F_XOR (f_a3, f_a3, f_k3);

	ASM_F_AND_NOT (fx14, fx13, f_a5);	/*             M  */

	ASM_F_AND_NOT (fx16, f_a2, fx15);	/*          J     */

    ASM_F_D_LOAD (Preloadf_1, merge[D61]);	/*                */
	ASM_F_AND (fx11, f_a2, fx10);		/*              N */

	ASM_F_XOR (fx12, fx6, fx11);		/*             M  */

	ASM_F_XOR (fx17, fx14, fx16);		/*          J  M  */

	ASM_F_AND_NOT (fx18, fx17, f_a3);	/*          J     */

	ASM_F_XOR_NOT (fx19, fx12, fx18);	/*          J   N */

	ASM_F_AND_NOT (fx20, fx19, fx1);	/*          J     */

	ASM_F_XOR (fx21, fx20, fx15);		/*          J     */

	ASM_F_XOR (Preloadf_1, Preloadf_1, fx19); /*          J     */

	ASM_F_XOR (fx32, fx3, fx6);		/*     E          */

	ASM_F_AND_NOT (fx33, fx32, fx10);	/*     E          */

	ASM_F_AND_NOT (fx22, f_a6, fx21);	/*          J     */

	ASM_F_XOR (fx23, fx22, fx6);		/*          J     */

	ASM_F_AND (fx42, f_a5, fx7);		/*  B             */

    ASM_F_D_LOAD (Preloadf_4, merge[D64]);	/*                */
	ASM_F_AND_NOT (fx43, f_a4, fx42);	/*  B             */

	ASM_F_OR (fx26, f_a5, f_a6);		/*    D           */

	ASM_F_AND_NOT (fx27, fx26, fx1);	/*    D           */

	ASM_F_AND (fx48, fx26, fx33);		/* A  DE          */

	ASM_F_AND_NOT (fx24, f_a2, fx23);	/*          J     */

	ASM_F_AND_NOT (fx28, f_a2, fx24);	/*          J L   */

	ASM_F_XOR (fx29, fx27, fx28);		/*    D       L   */

	ASM_F_XOR (fx25, fx21, fx24);		/*          J     */

	ASM_F_AND_NOT (fx38, fx21, f_a5);	/*        H J     */

	ASM_F_AND_NOT (fx30, f_a3, fx29);	/*            L   */

	ASM_F_XOR_NOT (fx31, fx25, fx30);	/*          J L   */

	ASM_F_XOR (Preloadf_4, Preloadf_4, fx31); /*            L   */

	ASM_F_XOR (fx34, f_a6, fx25);		/*         I      */

	ASM_F_AND_NOT (fx35, f_a5, fx34);	/*         I      */

	ASM_F_OR (fx46, fx23, fx35);		/*   C   G  J     */

	ASM_F_OR (fx41, fx35, fx2);		/*       G        */

    ASM_F_D_LOAD (Preloadf_3, merge[D63]);	/*                */
	ASM_F_AND_NOT (fx36, f_a2, fx35);	/*         I      */

	ASM_F_XOR (fx37, fx33, fx36);		/*     E   I      */

	ASM_F_OR (fx39, f_a3, fx38);		/*        H       */

	ASM_F_XOR_NOT (fx40, fx37, fx39);	/*        HI      */

	ASM_F_XOR (Preloadf_3, Preloadf_3, fx40); /*        H       */

	ASM_F_OR (fx44, f_a2, fx43);		/*  B             */

	ASM_F_XOR (fx45, fx41, fx44);		/*  B    G        */

	ASM_F_XOR (fx47, fx46, fx5);		/*   C            */

	ASM_F_XOR (fx49, fx48, fx2);		/* A              */

	ASM_F_AND (fx50, f_a2, fx49);		/* A              */

	ASM_F_XOR (fx51, fx47, fx50);		/* A C            */

    ASM_F_D_LOAD (Preloadf_2, merge[D62]);	/*                */
	ASM_F_AND_NOT (fx52, f_a3, fx51);	/* A              */

	ASM_F_XOR_NOT (fx53, fx45, fx52);	/* AB             */

	ASM_F_XOR (Preloadf_2, Preloadf_2, fx53); /* A              */

    ASM_F_D_STORE (out[D61], Preloadf_1);	/*          J     */

    ASM_F_D_STORE (out[D64], Preloadf_4);	/*            L   */

    ASM_F_D_STORE (out[D63], Preloadf_3);	/*        H       */

    ASM_F_D_STORE (out[D62], Preloadf_2);	/* A              */

#include "s0.h"
/* end of s6_f.h */
