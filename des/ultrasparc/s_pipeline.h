/* s_pipeline.h v4.0 */

/*
 * $Log: s_pipeline.h,v $
 * Revision 1.4  1998/06/20 14:50:13  remi
 * Fixed a comment.
 *
 * Revision 1.3  1998/06/16 06:27:43  remi
 * - Integrated some patches in the UltraSparc DES code.
 * - Cleaned-up C++ style comments in the UltraSparc DES code.
 * - Replaced "rm `find ..`" by "find . -name ..." in superclean.
 *
 * Revision 1.2  1998/06/14 15:19:32  remi
 * Avoid tons of warnings due to a brain-dead CVS.
 *
 * Revision 1.1.1.1  1998/06/14 14:23:51  remi
 * Initial integration.
 *
 */


#define ASM_PIPELINE_S1(in, offsets)				\
    ASM_A_LOAD (t_key3, offsets->Key_Ptrs[OFFSET1 + 3]); /* PIPELINE */ \
    ASM_D_LOAD (t_a3, in[S13]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k3, ((INNER_LOOP_SLICE *)t_key3)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key5, offsets->Key_Ptrs[OFFSET1 + 5]); /* PIPELINE */ \
    ASM_D_LOAD (t_a5, in[S15]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k5, ((INNER_LOOP_SLICE *)t_key5)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key4, offsets->Key_Ptrs[OFFSET1 + 4]); /* PIPELINE */ \
    ASM_D_LOAD (a4, in[S14]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k4, ((INNER_LOOP_SLICE *)t_key4)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key6, offsets->Key_Ptrs[OFFSET1 + 6]); /* PIPELINE */ \
    ASM_D_LOAD (a6, in[S16]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k6, ((INNER_LOOP_SLICE *)t_key6)[0]);	/* PIPELINE */ \
								\
    ASM_XOR (a3, t_a3, t_k3);				/* PIPELINE */ \
    ASM_XOR (a5, t_a5, t_k5);				/* PIPELINE */ \
								\
    ASM_XOR (a4, a4, t_k4);				/* PIPELINE */ \
    ASM_XOR (a6, a6, t_k6);				/* PIPELINE */

#define ASM_PIPELINE_S2(in, offsets)				\
    ASM_A_LOAD (t_key1, offsets->Key_Ptrs[OFFSET2 + 1]); /* PIPELINE */ \
    ASM_D_LOAD (t_a1, in[S21]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k1, ((INNER_LOOP_SLICE *)t_key1)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key6, offsets->Key_Ptrs[OFFSET2 + 6]); /* PIPELINE */ \
    ASM_D_LOAD (t_a6, in[S26]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k6, ((INNER_LOOP_SLICE *)t_key6)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key5, offsets->Key_Ptrs[OFFSET2 + 5]); /* PIPELINE */ \
    ASM_D_LOAD (a5, in[S25]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k5, ((INNER_LOOP_SLICE *)t_key5)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key2, offsets->Key_Ptrs[OFFSET2 + 2]); /* PIPELINE */ \
    ASM_D_LOAD (a2, in[S22]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k2, ((INNER_LOOP_SLICE *)t_key2)[0]); /* PIPELINE */ \
								\
    ASM_XOR (a1, t_a1, t_k1);				/* PIPELINE */ \
    ASM_XOR (a6, t_a6, t_k6);				/* PIPELINE */ \
								\
    ASM_XOR (a5, a5, t_k5);				/* PIPELINE */ \
    ASM_XOR (a2, a2, t_k2);				/* PIPELINE */

#define ASM_PIPELINE_S3(in, offsets)				\
    ASM_A_LOAD (t_key2, offsets->Key_Ptrs[OFFSET3 + 2]); /* PIPELINE */ \
    ASM_D_LOAD (t_a2, in[S32]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k2, ((INNER_LOOP_SLICE *)t_key2)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key3, offsets->Key_Ptrs[OFFSET3 + 3]); /* PIPELINE */ \
    ASM_D_LOAD (t_a3, in[S33]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k3, ((INNER_LOOP_SLICE *)t_key3)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key6, offsets->Key_Ptrs[OFFSET3 + 6]); /* PIPELINE */ \
    ASM_D_LOAD (a6, in[S36]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k6, ((INNER_LOOP_SLICE *)t_key6)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key5, offsets->Key_Ptrs[OFFSET3 + 5]); /* PIPELINE */ \
    ASM_D_LOAD (a5, in[S35]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k5, ((INNER_LOOP_SLICE *)t_key5)[0]); /* PIPELINE */ \
								\
    ASM_XOR (a2, t_a2, t_k2);				/* PIPELINE */ \
    ASM_XOR (a3, t_a3, t_k3);				/* PIPELINE */ \
								\
    ASM_XOR (a6, a6, t_k6);				/* PIPELINE */ \
    ASM_XOR (a5, a5, t_k5);				/* PIPELINE */

#define ASM_PIPELINE_S4(in, offsets)				\
    ASM_A_LOAD (t_key1, offsets->Key_Ptrs[OFFSET4 + 1]); /* PIPELINE */ \
    ASM_D_LOAD (t_a1, in[S41]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k1, ((INNER_LOOP_SLICE *)t_key1)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key3, offsets->Key_Ptrs[OFFSET4 + 3]); /* PIPELINE */ \
    ASM_D_LOAD (t_a3, in[S43]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k3, ((INNER_LOOP_SLICE *)t_key3)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key5, offsets->Key_Ptrs[OFFSET4 + 5]); /* PIPELINE */ \
    ASM_D_LOAD (a5, in[S45]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k5, ((INNER_LOOP_SLICE *)t_key5)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key2, offsets->Key_Ptrs[OFFSET4 + 2]); /* PIPELINE */ \
    ASM_D_LOAD (a2, in[S42]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k2, ((INNER_LOOP_SLICE *)t_key2)[0]); /* PIPELINE */ \
								\
    ASM_XOR (a1, t_a1, t_k1);				/* PIPELINE */ \
    ASM_XOR (a3, t_a3, t_k3);				/* PIPELINE */ \
								\
    ASM_XOR (a5, a5, t_k5);				/* PIPELINE */ \
    ASM_XOR (a2, a2, t_k2);				/* PIPELINE */

#define ASM_PIPELINE_S5(in, offsets)				\
    ASM_A_LOAD (t_key3, offsets->Key_Ptrs[OFFSET5 + 3]); /* PIPELINE */ \
    ASM_D_LOAD (t_a3, in[S53]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k3, ((INNER_LOOP_SLICE *)t_key3)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key4, offsets->Key_Ptrs[OFFSET5 + 4]); /* PIPELINE */ \
    ASM_D_LOAD (t_a4, in[S54]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k4, ((INNER_LOOP_SLICE *)t_key4)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key1, offsets->Key_Ptrs[OFFSET5 + 1]); /* PIPELINE */ \
    ASM_D_LOAD (a1, in[S51]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k1, ((INNER_LOOP_SLICE *)t_key1)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key6, offsets->Key_Ptrs[OFFSET5 + 6]); /* PIPELINE */ \
    ASM_D_LOAD (a6, in[S56]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k6, ((INNER_LOOP_SLICE *)t_key6)[0]); /* PIPELINE */ \
								\
    ASM_XOR (a3, t_a3, t_k3);				/* PIPELINE */ \
    ASM_XOR (a4, t_a4, t_k4);				/* PIPELINE */ \
								\
    ASM_XOR (a1, a1, t_k1);				/* PIPELINE */ \
    ASM_XOR (a6, a6, t_k6);				/* PIPELINE */

#define ASM_PIPELINE_S6(in, offsets)				\
    ASM_A_LOAD (t_key5, offsets->Key_Ptrs[OFFSET6 + 5]); /* PIPELINE */ \
    ASM_D_LOAD (t_a5, in[S65]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k5, ((INNER_LOOP_SLICE *)t_key5)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key1, offsets->Key_Ptrs[OFFSET6 + 1]); /* PIPELINE */ \
    ASM_D_LOAD (t_a1, in[S61]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k1, ((INNER_LOOP_SLICE *)t_key1)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key6, offsets->Key_Ptrs[OFFSET6 + 6]); /* PIPELINE */ \
    ASM_D_LOAD (a6, in[S66]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k6, ((INNER_LOOP_SLICE *)t_key6)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key4, offsets->Key_Ptrs[OFFSET6 + 4]); /* PIPELINE */ \
    ASM_D_LOAD (a4, in[S64]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k4, ((INNER_LOOP_SLICE *)t_key4)[0]); /* PIPELINE */ \
								\
    ASM_XOR (a5, t_a5, t_k5);				/* PIPELINE */ \
    ASM_XOR (a1, t_a1, t_k1);				/* PIPELINE */ \
								\
    ASM_XOR (a6, a6, t_k6);				/* PIPELINE */ \
    ASM_XOR (a4, a4, t_k4);				/* PIPELINE */

#define ASM_PIPELINE_S7(in, offsets)				\
    ASM_A_LOAD (t_key2, offsets->Key_Ptrs[OFFSET7 + 2]); /* PIPELINE */ \
    ASM_D_LOAD (t_a2, in[S72]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k2, ((INNER_LOOP_SLICE *)t_key2)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key4, offsets->Key_Ptrs[OFFSET7 + 4]); /* PIPELINE */ \
    ASM_D_LOAD (t_a4, in[S74]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k4, ((INNER_LOOP_SLICE *)t_key4)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key5, offsets->Key_Ptrs[OFFSET7 + 5]); /* PIPELINE */ \
    ASM_D_LOAD (a5, in[S75]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k5, ((INNER_LOOP_SLICE *)t_key5)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key3, offsets->Key_Ptrs[OFFSET7 + 3]); /* PIPELINE */ \
    ASM_D_LOAD (a3, in[S73]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k3, ((INNER_LOOP_SLICE *)t_key3)[0]); /* PIPELINE */ \
								\
    ASM_XOR (a2, t_a2, t_k2);				/* PIPELINE */ \
    ASM_XOR (a4, t_a4, t_k4);				/* PIPELINE */ \
								\
    ASM_XOR (a5, a5, t_k5);				/* PIPELINE */ \
    ASM_XOR (a3, a3, t_k3);				/* PIPELINE */

#define ASM_PIPELINE_S8(in, offsets)				\
    ASM_A_LOAD (t_key3, offsets->Key_Ptrs[OFFSET8 + 3]); /* PIPELINE */ \
    ASM_D_LOAD (t_a3, in[S83]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k3, ((INNER_LOOP_SLICE *)t_key3)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key1, offsets->Key_Ptrs[OFFSET8 + 1]); /* PIPELINE */ \
    ASM_D_LOAD (t_a1, in[S81]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k1, ((INNER_LOOP_SLICE *)t_key1)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key4, offsets->Key_Ptrs[OFFSET8 + 4]); /* PIPELINE */ \
    ASM_D_LOAD (a4, in[S84]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k4, ((INNER_LOOP_SLICE *)t_key4)[0]);	/* PIPELINE */ \
								\
    ASM_A_LOAD (t_key5, offsets->Key_Ptrs[OFFSET8 + 5]); /* PIPELINE */ \
    ASM_D_LOAD (a5, in[S85]);				/* PIPELINE */ \
    ASM_D_LOAD (t_k5, ((INNER_LOOP_SLICE *)t_key5)[0]); /* PIPELINE */ \
								\
    ASM_XOR (a3, t_a3, t_k3);				/* PIPELINE */ \
    ASM_XOR (a1, t_a1, t_k1);				/* PIPELINE */ \
								\
    ASM_XOR (a4, a4, t_k4);				/* PIPELINE */ \
    ASM_XOR (a5, a5, t_k5);				/* PIPELINE */

/* floating point pipe manually placed into other functions
 * already placed into s2_3
 *  ASM_F_D_LOAD (ft_a5, in[S65]);		/ PIPELINE into s2_3 /
 *  ASM_F_D_LOAD (ft_a1, in[S61]);		/ PIPELINE into s2_3 /
 *
 *  ASM_F_D_LOAD (f_a6, in[S66]);		/ PIPELINE into s2_3 /
 *  ASM_F_D_LOAD (f_a4, in[S64]);		/ PIPELINE into s2_3 /
 */

#define ASM_PIPELINE_F_S6(in, offsets)				\
    ASM_F_A_LOAD (ft_key5, offsets->Key_Ptrs[OFFSET6 + 5]);	/* PIPELINE */ \
    ASM_F_A_LOAD (ft_key1, offsets->Key_Ptrs[OFFSET6 + 1]);	/* PIPELINE */ \
								\
    ASM_F_D_LOAD (ft_k5, ((INNER_LOOP_FSLICE *)ft_key5)[0]); /* PIPELINE */ \
    ASM_F_D_LOAD (ft_k1, ((INNER_LOOP_FSLICE *)ft_key1)[0]); /* PIPELINE */ \
								\
    ASM_F_A_LOAD (ft_key6, offsets->Key_Ptrs[OFFSET6 + 6]);	/* PIPELINE */ \
    ASM_F_XOR (f_a5, ft_a5, ft_k5);		/* PIPELINE */ \
    ASM_F_A_LOAD (ft_key4, offsets->Key_Ptrs[OFFSET6 + 4]);	/* PIPELINE */ \
    ASM_F_XOR (f_a1, ft_a1, ft_k1);		/* PIPELINE */ \
								\
    ASM_F_D_LOAD (ft_k6, ((INNER_LOOP_FSLICE *)ft_key6)[0]); /* PIPELINE */ \
    ASM_F_D_LOAD (ft_k4, ((INNER_LOOP_FSLICE *)ft_key4)[0]); /* PIPELINE */ \
								\
    ASM_F_D_LOAD (f_a3, in[S63]);		/* PIPELINE */ \
    ASM_F_XOR (f_a6, f_a6, ft_k6);		/* PIPELINE */

/* already placed into s3_4
 *  ASM_F_A_LOAD (fkey2, offsets->Key_Ptrs[OFFSET6 + 2]); / * PIPELINE into s3_4 * /
 *  ASM_F_A_LOAD (fkey3, offsets->Key_Ptrs[OFFSET6 + 3]); / * PIPELINE into s3_4 * /
 *
 *  ASM_F_D_LOAD (f_k2, ((INNER_LOOP_FSLICE *)fkey2)[0]); / * PIPELINE into s3_4 * /
 *  ASM_F_D_LOAD (f_k3, ((INNER_LOOP_FSLICE *)fkey3)[0]); / * PIPELINE into s3_4 * /
 *
 *  ASM_F_D_LOAD (f_a2, in[S62]);		/ * PIPELINE into s3_4 * /
 */

#define ASM_PIPELINE_F_S6_all(in, offsets)				\
    ASM_F_D_LOAD (ft_a5, in[S65]);		/* PIPELINE into s2_3 */ \
    ASM_F_D_LOAD (ft_a1, in[S61]);		/* PIPELINE into s2_3 */ \
								\
    ASM_F_D_LOAD (f_a6, in[S66]);		/* PIPELINE into s2_3 */ \
    ASM_F_D_LOAD (f_a4, in[S64]);		/* PIPELINE into s2_3 */ \
								\
    ASM_F_A_LOAD (ft_key5, offsets->Key_Ptrs[OFFSET6 + 5]);	/* PIPELINE */ \
    ASM_F_A_LOAD (ft_key1, offsets->Key_Ptrs[OFFSET6 + 1]);	/* PIPELINE */ \
								\
    ASM_F_D_LOAD (ft_k5, ((INNER_LOOP_FSLICE *)ft_key5)[0]); /* PIPELINE */ \
    ASM_F_D_LOAD (ft_k1, ((INNER_LOOP_FSLICE *)ft_key1)[0]); /* PIPELINE */ \
								\
    ASM_F_A_LOAD (ft_key6, offsets->Key_Ptrs[OFFSET6 + 6]);	/* PIPELINE */ \
    ASM_F_XOR (f_a5, ft_a5, ft_k5);		/* PIPELINE */ \
    ASM_F_A_LOAD (ft_key4, offsets->Key_Ptrs[OFFSET6 + 4]);	/* PIPELINE */ \
    ASM_F_XOR (f_a1, ft_a1, ft_k1);		/* PIPELINE */ \
								\
    ASM_F_D_LOAD (ft_k6, ((INNER_LOOP_FSLICE *)ft_key6)[0]); /* PIPELINE */ \
    ASM_F_D_LOAD (ft_k4, ((INNER_LOOP_FSLICE *)ft_key4)[0]); /* PIPELINE */ \
								\
    ASM_F_D_LOAD (f_a3, in[S63]);		/* PIPELINE */ \
    ASM_F_XOR (f_a6, f_a6, ft_k6);		/* PIPELINE */ \
								\
    ASM_F_A_LOAD (fkey2, offsets->Key_Ptrs[OFFSET6 + 2]); /* PIPELINE into s3_4 */ \
    ASM_F_A_LOAD (fkey3, offsets->Key_Ptrs[OFFSET6 + 3]); /* PIPELINE into s3_4 */ \
								\
    ASM_F_D_LOAD (f_k2, ((INNER_LOOP_FSLICE *)fkey2)[0]); /* PIPELINE into s3_4 */ \
    ASM_F_D_LOAD (f_k3, ((INNER_LOOP_FSLICE *)fkey3)[0]); /* PIPELINE into s3_4 */ \
								\
    ASM_F_D_LOAD (f_a2, in[S62]);		/* PIPELINE into s3_4 */

/* upon exit from this, g_k1, g_k6, g_a5, gt_k3, g_a4, g_a2 are live */
#define ASM_PIPELINE_F_S7(in, offsets)				\
    ASM_F_D_LOAD (gt_a2, in[S72]);		/* PIPELINE */ \
    ASM_F_D_LOAD (gt_a4, in[S74]);		/* PIPELINE */ \
								\
    ASM_F_D_LOAD (g_a5, in[S75]);		/* PIPELINE */ \
								\
    ASM_F_A_LOAD (gt_key2, offsets->Key_Ptrs[OFFSET7 + 2]); /* PIPELINE */ \
    ASM_F_A_LOAD (gt_key4, offsets->Key_Ptrs[OFFSET7 + 4]); /* PIPELINE */ \
								\
    ASM_F_D_LOAD (gt_k2, ((INNER_LOOP_FSLICE *)gt_key2)[0]);	/* PIPELINE */ \
    ASM_F_D_LOAD (gt_k4, ((INNER_LOOP_FSLICE *)gt_key4)[0]);	/* PIPELINE */ \
								\
    ASM_F_A_LOAD (gt_key5, offsets->Key_Ptrs[OFFSET7 + 5]); /* PIPELINE */ \
    ASM_F_XOR (g_a2, gt_a2, gt_k2);		/* PIPELINE */ \
    ASM_F_A_LOAD (gt_key3, offsets->Key_Ptrs[OFFSET7 + 3]); /* PIPELINE */ \
    ASM_F_XOR (g_a4, gt_a4, gt_k4);		/* PIPELINE */ \
								\
    ASM_F_D_LOAD (gt_k5, ((INNER_LOOP_FSLICE *)gt_key5)[0]);	/* PIPELINE */ \
    ASM_F_D_LOAD (gt_k3, ((INNER_LOOP_FSLICE *)gt_key3)[0]); /* PIPELINE */ \
								\
    ASM_F_XOR (g_a5, g_a5, gt_k5);		/* PIPELINE */ \
								\
    ASM_F_A_LOAD (gkey6, offsets->Key_Ptrs[OFFSET7 + 6]);	\
    ASM_F_A_LOAD (gkey1, offsets->Key_Ptrs[OFFSET7 + 1]);	\
    ASM_F_D_LOAD (g_k6, ((INNER_LOOP_FSLICE *)gkey6)[0]);	\
    ASM_F_D_LOAD (g_k1, ((INNER_LOOP_FSLICE *)gkey1)[0]);	

/* already placed into s5_8_w6
 *    ASM_F_D_LOAD (g_a3, in[S73]);		/ * PIPELINE * /
 *    ASM_F_XOR (g_a3, g_a3, gt_k3);		/ * PIPELINE * /
 *    ASM_F_D_LOAD (g_a6, in[S76]);
 *    ASM_F_D_LOAD (g_a1, in[S71]);
 */

#define ASM_PIPELINE_F_S7_all(in, offsets)				\
    ASM_F_D_LOAD (gt_a2, in[S72]);		/* PIPELINE */ \
    ASM_F_D_LOAD (gt_a4, in[S74]);		/* PIPELINE */ \
								\
    ASM_F_A_LOAD (gt_key2, offsets->Key_Ptrs[OFFSET7 + 2]); /* PIPELINE */ \
    ASM_F_A_LOAD (gt_key4, offsets->Key_Ptrs[OFFSET7 + 4]); /* PIPELINE */ \
								\
    ASM_F_D_LOAD (gt_k2, ((INNER_LOOP_FSLICE *)gt_key2)[0]);	/* PIPELINE */ \
    ASM_F_D_LOAD (gt_k4, ((INNER_LOOP_FSLICE *)gt_key4)[0]);	/* PIPELINE */ \
								\
    ASM_F_D_LOAD (g_a5, in[S75]);		/* PIPELINE */ \
								\
    ASM_F_A_LOAD (gt_key5, offsets->Key_Ptrs[OFFSET7 + 5]); /* PIPELINE */ \
    ASM_F_XOR (g_a2, gt_a2, gt_k2);		/* PIPELINE */ \
    ASM_F_A_LOAD (gt_key3, offsets->Key_Ptrs[OFFSET7 + 3]); /* PIPELINE */ \
    ASM_F_XOR (g_a4, gt_a4, gt_k4);		/* PIPELINE */ \
								\
    ASM_F_D_LOAD (gt_k5, ((INNER_LOOP_FSLICE *)gt_key5)[0]);	/* PIPELINE */ \
    ASM_F_D_LOAD (gt_k3, ((INNER_LOOP_FSLICE *)gt_key3)[0]); /* PIPELINE */ \
								\
    ASM_F_A_LOAD (gkey6, offsets->Key_Ptrs[OFFSET7 + 6]);	\
    ASM_F_XOR (g_a5, g_a5, gt_k5);		/* PIPELINE */ \
    ASM_F_A_LOAD (gkey1, offsets->Key_Ptrs[OFFSET7 + 1]);	\
    ASM_F_D_LOAD (g_k6, ((INNER_LOOP_FSLICE *)gkey6)[0]);	\
    ASM_F_D_LOAD (g_k1, ((INNER_LOOP_FSLICE *)gkey1)[0]);	

/* end of s_pipeline.h */
