/* s6_f_regs.h v3.0 */

/* $Log: s6_f_regs.h,v $
/* Revision 1.1  1998/06/14 14:23:50  remi
/* Initial revision
/* */

#ifdef MANUAL_REGISTER_ALLOCATION

    register INNER_LOOP_FSLICE FT1, FT2, FT3, FT4, FT5, FT6, FT7, FT8;
    register INNER_LOOP_FSLICE FT9, FT10, FT11, FT12, FT13, FT14, FT15;
    register INNER_LOOP_FSLICE FT16;

    INNER_LOOP_SLICE *fkey2, *fkey3;
    INNER_LOOP_SLICE *ft_key1, *ft_key2, *ft_key3, *ft_key4, *ft_key5, *ft_key6;

/* FREE */
#define fx1	FT16
#define fx27	FT16

/* FREE */
#define f_k2	FT15
/* FREE */
#define fx16	FT15
#define fx17	FT15
#define fx18	FT15
#define fx19	FT15
/* FREE */
#define fx28	FT15
#define fx29	FT15
#define fx30	FT15
#define fx31	FT15

#define ft_a1	FT14
#define f_a1	FT14
#define fx3	FT14
#define fx32	FT14
#define fx33	FT14

/* FREE */
#define fx8	FT13
#define fx9	FT13
#define fx10	FT13
/* FREE */
#define fx24	FT13
#define fx25	FT13
#define fx34	FT13
#define fx35	FT13
#define fx36	FT13
#define fx37	FT13

#define ft_k4	FT12
/* FREE */
#define f_k3	FT12
/* FREE */
#define fx11	FT12
#define fx12	FT12
/* FREE */
#define fx20	FT12
#define fx21	FT12
#define fx38	FT12
#define fx39	FT12
#define fx40	FT12

#define ft_a5	FT11
#define f_a5	FT11
/* FREE */
#define fx41	FT11

/* FREE */
#define fx5	FT10

/* FREE */
#define fx2	FT9

/* FREE */
#define f_a2	FT8

#define f_a3	FT7

#define ft_k6	FT6
/* FREE */
#define fx7	FT6
#define fx42	FT6
#define fx43	FT6
#define fx44	FT6
#define fx45	FT6

#define f_a4	FT5
/* FREE */
#define fx26	FT5
#define fx48	FT5
#define fx49	FT5
#define fx50	FT5
#define fx51	FT5
#define fx52	FT5
#define fx53	FT5

#define ft_k1	FT4
/* FREE */
#define fx13	FT4
#define fx14	FT4
/* FREE */
#define Preloadf_1 FT4

#define ft_k5	FT3
/* FREE */
#define fx6	FT3
/* FREE */
#define Preloadf_4 FT3

#define f_a6	FT2
/* FREE */
#define Preloadf_3 FT2

#define fx4	FT1
#define fx15	FT1
/* FREE */
#define fx22	FT1
#define fx23	FT1
#define fx46	FT1
#define fx47	FT1
/* FREE */
#define Preloadf_2 FT1

#else /* MANUAL_REGISTER_ALLOCATION */

    INNER_LOOP_FSLICE fx1, fx2, fx3, fx4, fx5, fx6, fx7, fx8;
    INNER_LOOP_FSLICE fx9, fx10, fx11, fx12, fx13, fx14, fx15, fx16;
    INNER_LOOP_FSLICE fx17, fx18, fx19, fx20, fx21, fx22, fx23, fx24;
    INNER_LOOP_FSLICE fx25, fx26, fx27, fx28, fx29, fx30, fx31, fx32;
    INNER_LOOP_FSLICE fx33, fx34, fx35, fx36, fx37, fx38, fx39, fx40;
    INNER_LOOP_FSLICE fx41, fx42, fx43, fx44, fx45, fx46, fx47, fx48;
    INNER_LOOP_FSLICE fx49, fx50, fx51, fx52, fx53;
    INNER_LOOP_FSLICE Preloadf_1, Preloadf_2, Preloadf_3, Preloadf_4;

    INNER_LOOP_FSLICE f_k1, f_k2, f_k3, f_k6;
    INNER_LOOP_FSLICE *fkey1, *fkey2, *fkey3, *fkey6;

    INNER_LOOP_FSLICE f_a1, f_a2, f_a3, f_a4, f_a5, f_a6;

    INNER_LOOP_FSLICE ft_a1, ft_a2, ft_a3, ft_a4, ft_a5;
    INNER_LOOP_FSLICE ft_k1, ft_k2, ft_k3, ft_k4, ft_k5, ft_k6;
    INNER_LOOP_FSLICE *ft_key1, *ft_key2, *ft_key3, *ft_key4, *ft_key5, *ft_key6;

#endif /* MANUAL_REGISTER_ALLOCATION */

/* end of s6_f.h */
