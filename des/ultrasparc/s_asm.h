/* s_asm.h v3.0 */

/* $Log: s_asm.h,v $
/* Revision 1.1.1.1  1998/06/14 14:23:51  remi
/* Initial integration.
/* */


#ifdef ASM

#undef SET_64_BIT_SENTINEL
#undef RETURN_64_BIT_SENTINEL

#ifdef USE_64_BIT_SENTINEL

#ifdef STACK_POINTER_64_BIT
#define SET_64_BIT_SENTINEL			\
    {	register unsigned long temp = MAGIC_SENTINEL_VAL;			\
	__asm__ volatile ("sllx %%i7, 32, %%i7" : : );				\
	__asm__ volatile ("srlx %%i7, 32, %%i7" : : );				\
	__asm__ volatile ("sllx %1, 32, %0" : "=r" (temp) : "r" (temp));	\
	__asm__ volatile ("or %%i7, %0, %%i7	!##" : : "r" (temp));		\
    }
#else /* STACK_POINTER_64_BIT */
#define SET_64_BIT_SENTINEL			\
    {	register unsigned long temp = MAGIC_SENTINEL_VAL;			\
	__asm__ volatile ("sllx %1, 32, %0" : "=r" (temp) : "r" (temp));	\
	__asm__ volatile ("or %%i7, %0, %%i7	!##" : : "r" (temp));		\
    }
#endif /* STACK_POINTER_64_BIT */

#define RETURN_64_BIT_SENTINEL			\
    {	register unsigned long temp;						\
	__asm__ volatile ("srlx %%i7, 32, %0	!##" : "=r" (temp) : );		\
	return (temp);								\
    }
#else /* USE_64_BIT_SENTINEL */
#define SET_64_BIT_SENTINEL
#define RETURN_64_BIT_SENTINEL			return (MAGIC_SENTINEL_VAL);
#endif /* USE_64_BIT_SENTINEL */

#define ASM_COMMENT_END_INCLUDE(arg)		\
__asm__ volatile ("			!# end of included file" : : );

#define ASM_AND_NOT(result, op1, op2)		\
__asm__ volatile ("andn %1, %2, %0" : "=r" (result): "r" (op1), "r" (op2));

#define ASM_OR_NOT(result, op1, op2)		\
__asm__ volatile ("orn %1, %2, %0" : "=r" (result): "r" (op1), "r" (op2));

#define ASM_XOR_NOT(result, op1, op2)		\
__asm__ volatile ("xnor %1, %2, %0" : "=r" (result): "r" (op1), "r" (op2));

#define ASM_AND(result, op1, op2)		\
__asm__ volatile ("and %1, %2, %0" : "=r" (result): "r" (op1), "r" (op2));

#define ASM_OR(result, op1, op2)		\
__asm__ volatile ("or %1, %2, %0" : "=r" (result): "r" (op1), "r" (op2));

#define ASM_XOR(result, op1, op2)		\
__asm__ volatile ("xor %1, %2, %0" : "=r" (result): "r" (op1), "r" (op2));

/* the rumor is that LDUW loads a word without sign extention faster than LD */
#define ASM_A_LOAD(result, addr)		\
__asm__ volatile ("lduw %1, %0	!+" : "=r" (result) : "m" (addr));

#ifdef LOAD_STORE_64_BIT_INTS

#define ASM_R_LOAD(result, addr, offset)	\
__asm__ volatile ("ldx [%1+%2], %0	!#" : "=r" (result) : "r" (addr), "r" (offset));

#define ASM_D_LOAD(result, addr)		\
__asm__ volatile ("ldx %1, %0	!#" : "=r" (result) : "m" (addr));

#define ASM_D_STORE(addr, result)		\
__asm__ volatile ("stx %1, %0	!#" : "=m" (addr): "r" (result));

#else /* LOAD_STORE_64_BIT_INTS */

#define ASM_R_LOAD(result, addr, offset)	\
__asm__ volatile ("ld [%1+%2], %0	!#" : "=r" (result) : "r" (addr), "r" (offset));

#define ASM_D_LOAD(result, addr)		\
__asm__ volatile ("ld %1, %0	!#" : "=r" (result) : "m" (addr));

#define ASM_D_STORE(addr, result)		\
__asm__ volatile ("st %1, %0	!#" : "=m" (addr): "r" (result));

#endif /* LOAD_STORE_64_BIT_INTS */

#define ASM_MOVE(result, src)			\
__asm__ volatile ("mov %1, %0" : "=r" (result) : "r" (src));

#define ASM_NOOP(reg)				\
__asm__ volatile ("nop	!#" : :);


#ifdef DO_FLOAT_PIPE

#define ASM_F_AND_NOT(result, op1, op2)		\
__asm__ volatile ("fandnot2 %1, %2, %0" : "=f" (result): "f" (op1), "f" (op2));

#define ASM_F_OR_NOT(result, op1, op2)		\
__asm__ volatile ("fornot2 %1, %2, %0" : "=f" (result): "f" (op1), "f" (op2));

#define ASM_F_XOR_NOT(result, op1, op2)		\
__asm__ volatile ("fxnor %1, %2, %0" : "=f" (result): "f" (op1), "f" (op2));

#define ASM_F_AND(result, op1, op2)		\
__asm__ volatile ("fand %1, %2, %0" : "=f" (result): "f" (op1), "f" (op2));

#define ASM_F_OR(result, op1, op2)		\
__asm__ volatile ("for %1, %2, %0" : "=f" (result): "f" (op1), "f" (op2));

#define ASM_F_XOR(result, op1, op2)		\
__asm__ volatile ("fxor %1, %2, %0" : "=f" (result): "f" (op1), "f" (op2));

#define ASM_F_A_LOAD(result, addr)			\
__asm__ volatile ("lduw %1, %0	!#" : "=r" (result) : "m" (addr));

#define ASM_F_LOAD(result, addr)		\
__asm__ volatile ("ldd %1, %0	!#" : "=f" (result) : "m" (addr));
/* __asm__ volatile ("lddf %1, %0	!#" : "=f" (result) : "m" (addr)); */

#define ASM_F_R_LOAD(result, addr, offset)			\
__asm__ volatile ("ldd [%1+%2], %0	!#" : "=f" (result) : "r" (addr), "r" (offset));

#define ASM_F_D_LOAD(result, addr)		\
__asm__ volatile ("ldd %1, %0	!#" : "=f" (result) : "m" (addr));

#define ASM_F_D_STORE(addr, result)		\
__asm__ volatile ("std %1, %0	!#" : "=m" (addr): "f" (result));
/* __asm__ volatile ("std %1, %0	!#" : "=m" (addr): "f" (result)); */

#define ASM_F_NOOP(reg)				\
__asm__ volatile ("fmovd %0, %0	!#" : "=f" (reg) :"f" (reg));
/* __asm__ volatile ("std %1, %0	!#" : "=m" (addr): "f" (result)); */

#else /* DO_FLOAT_PIPE */

#define ASM_F_AND_NOT(result, op1, op2)
#define ASM_F_OR_NOT(result, op1, op2)
#define ASM_F_XOR_NOT(result, op1, op2)
#define ASM_F_AND(result, op1, op2)
#define ASM_F_OR(result, op1, op2)
#define ASM_F_XOR(result, op1, op2)
#define ASM_F_A_LOAD(result, addr)
#define ASM_F_LOAD(result, addr)
#define ASM_F_R_LOAD(result, addr, offset)
#define ASM_F_D_LOAD(result, addr)
#define ASM_F_D_STORE(addr, val)
#define ASM_F_NOOP(reg)

#endif /* DO_FLOAT_PIPE */

#else /* not ASM */

#define SET_64_BIT_SENTINEL
#define RETURN_64_BIT_SENTINEL			return (MAGIC_SENTINEL_VAL);
#define ASM_COMMENT_END_INCLUDE(arg)

#define ASM_AND_NOT(result, op1, op2)		result = op1 & ~op2;
#define ASM_OR_NOT(result, op1, op2)		result = op1 | ~op2;
#define ASM_XOR_NOT(result, op1, op2)		result = op1 ^ ~op2;
#define ASM_AND(result, op1, op2)		result = op1 &  op2;
#define ASM_OR(result, op1, op2)		result = op1 |  op2;
#define ASM_XOR(result, op1, op2)		result = op1 ^  op2;
#define ASM_A_LOAD(result, addr)		(INNER_LOOP_SLICE *)result = addr;
#define ASM_R_LOAD(result, addr, offset)	\
		result = *(INNER_LOOP_SLICE *)((long)addr + (long)offset);
#define ASM_D_LOAD(result, addr)		result = addr;
#define ASM_D_STORE(addr, val)			addr = val;
#define ASM_MOVE(result, src)			result = src;
#define ASM_NOOP(reg)				reg = reg

#ifdef DO_FLOAT_PIPE

#define ASM_F_AND_NOT(result, op1, op2)		result = op1 & ~op2;
#define ASM_F_OR_NOT(result, op1, op2)		result = op1 | ~op2;
#define ASM_F_XOR_NOT(result, op1, op2)		result = op1 ^ ~op2;
#define ASM_F_AND(result, op1, op2)		result = op1 &  op2;
#define ASM_F_OR(result, op1, op2)		result = op1 |  op2;
#define ASM_F_XOR(result, op1, op2)		result = op1 ^  op2;
#define ASM_F_A_LOAD(result, addr)		(INNER_LOOP_FSLICE *)result = addr;
#define ASM_F_LOAD(result, addr)		(INNER_LOOP_FSLICE *)result = addr;
#define ASM_F_R_LOAD(result, addr, offset)	\
			result = *(DES_FSLICE *)((long)addr + (long)offset);
#define ASM_F_D_LOAD(result, addr)		result = addr;
#define ASM_F_D_STORE(addr, val)		addr = val;
#define ASM_F_NOOP(reg)				reg = reg;

#else /* DO_FLOAT_PIPE */

#define ASM_F_AND_NOT(result, op1, op2)
#define ASM_F_OR_NOT(result, op1, op2)
#define ASM_F_XOR_NOT(result, op1, op2)
#define ASM_F_AND(result, op1, op2)
#define ASM_F_OR(result, op1, op2)
#define ASM_F_XOR(result, op1, op2)
#define ASM_F_A_LOAD(result, addr)
#define ASM_F_LOAD(result, addr)
#define ASM_F_R_LOAD(result, addr, offset)
#define ASM_F_D_LOAD(result, addr)
#define ASM_F_D_STORE(addr, val)
#define ASM_F_NOOP(reg)

#endif /* DO_FLOAT_PIPE */

#endif /* ASM */

/* end of s_asm.h */
