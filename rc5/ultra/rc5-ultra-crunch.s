! $Id: rc5-ultra-crunch.s,v 1.1.2.1 2001/05/20 00:30:02 andreasb Exp $
! Generated from Id: rc5-ultra-crunch.cpp,v 1.9.2.4 2000/02/03 22:50:16 ivo Exp
! using command:
! gcc -S -W ... -fcaller-saves -fomit-frame-pointer -fno-inline-functions \
!    -fstrict-aliasing -O2 -mcpu=cypress -DASM_SPARC -funroll-loops  \
!    -Dsolaris -I./common rc5/ultra/rc5-ultra-crunch.cpp \
!    -o rc5/ultra/rc5-ultra-crunch.s

	.file	"rc5-ultra-crunch.cpp"
gcc2_compiled.:
.section	".rodata"
	.align 8
.LLC0:
	.asciz	"Lawrence Butcher 16/9/97 lbutcher@eng.sun.com  LB Version 1.3"
.section	".text"
	.align 4
	.global LB_VERSION__Fv
	.type	 LB_VERSION__Fv,#function
	.proc	0102
LB_VERSION__Fv:
.LLFB1:
	!#PROLOGUE# 0
	!#PROLOGUE# 1
	sethi	%hi(.LLC0), %o0
	retl
	or	%o0, %lo(.LLC0), %o0
.LLFE1:
.LLfe1:
	.size	 LB_VERSION__Fv,.LLfe1-LB_VERSION__Fv
.section	".data"
	.align 4
	.type	 S0_INIT,#object
	.size	 S0_INIT,240
S0_INIT:
	.uaword	-1089828067
	.uaword	1444465436
	.uaword	-196066091
	.uaword	-1836597618
	.uaword	817838151
	.uaword	-822693376
	.uaword	1831742393
	.uaword	191210866
	.uaword	-1449320661
	.uaword	1205115108
	.uaword	-435416419
	.uaword	-2075947946
	.uaword	578487823
	.uaword	-1062043704
	.uaword	1592392065
	.uaword	-48139462
	.uaword	-1688670989
	.uaword	965764780
	.uaword	-674766747
	.uaword	1979669022
	.uaword	339137495
	.uaword	-1301394032
	.uaword	1353041737
	.uaword	-287489790
	.uaword	-1928021317
	.uaword	726414452
	.uaword	-1089828067
	.uaword	0
	.uaword	0
	.uaword	0
	.uaword	-1089828067
	.uaword	1444465436
	.uaword	-196066091
	.uaword	-1836597618
	.uaword	817838151
	.uaword	-822693376
	.uaword	1831742393
	.uaword	191210866
	.uaword	-1449320661
	.uaword	1205115108
	.uaword	-435416419
	.uaword	-2075947946
	.uaword	578487823
	.uaword	-1062043704
	.uaword	1592392065
	.uaword	-48139462
	.uaword	-1688670989
	.uaword	965764780
	.uaword	-674766747
	.uaword	1979669022
	.uaword	339137495
	.uaword	-1301394032
	.uaword	1353041737
	.uaword	-287489790
	.uaword	-1928021317
	.uaword	726414452
	.uaword	-1089828067
	.uaword	0
	.uaword	0
	.uaword	0
.section	".text"
	.align 4
	.type	 crunch__FP11RC5UnitWorkUi,#function
	.proc	016
crunch__FP11RC5UnitWorkUi:
.LLFB2:
	!#PROLOGUE# 0
	save	%sp, -608, %sp
.LLCFI0:
	!#PROLOGUE# 1
	st	%i0, [%fp+68]
	ldub	[%i0+16], %g2
	sethi	%hi(16777216), %i5
	ldub	[%i0+17], %i2
	sll	%g2, 24, %g2
	ldub	[%i0+18], %i4
	sll	%i2, 16, %i2
	ldub	[%i0+20], %i3
	or	%i2, %g2, %i2
	ldub	[%i0+21], %g3
	sll	%i4, 8, %i4
	ld	[%fp+68], %o7
	or	%i4, %i2, %i4
	ldub	[%i0+22], %i0
	sll	%i3, 24, %i3
	ldub	[%o7+19], %g2
	sll	%g3, 16, %g3
	ldub	[%o7+23], %i2
	or	%g3, %i3, %g3
	sll	%i0, 8, %i0
	or	%g2, %i4, %g2
	or	%i0, %g3, %i0
	st	%i1, [%fp+72]
	andn	%g2, %i5, %o1
	or	%i2, %i0, %o0
	mov	32, %l5
	st	%i1, [%fp-508]
	st	%o1, [%fp-504]
	cmp	%i1, 0
	be	.LL6
	st	%o0, [%fp-500]
	add	%fp, -256, %l6
.LL7:
	ld	[%fp+68], %i0
	sethi	%hi(S0_INIT), %i5
	ldub	[%i0+4], %i3
	or	%i5, %lo(S0_INIT), %l7
	ldub	[%i0+5], %i2
	sll	%i3, 24, %i3
	sll	%i2, 16, %i2
	or	%i2, %i3, %i2
	ldub	[%i0+6], %i1
	ld	[%fp+68], %i3
	sll	%i1, 8, %i1
	ldub	[%i3+7], %i4
	or	%i1, %i2, %i1
	ldub	[%i0], %g2
	or	%i4, %i1, %l3
	ldub	[%i0+1], %g3
	sll	%g2, 24, %g2
	ldub	[%i0+2], %i0
	sll	%g3, 16, %g3
	ld	[%fp-508], %i4
	or	%g3, %g2, %g3
	ldub	[%i3+3], %i3
	sll	%i0, 8, %i0
	ld	[%l7+4], %l1
	or	%i0, %g3, %i0
	sethi	%hi(-1089828864), %g2
	add	%i4, -1, %i4
	or	%i3, %i0, %l4
	or	%g2, 797, %o2
	st	%i4, [%fp-508]
	add %o0, %o2, %i1
	add %o0, %o2, %g3
	sll %i1, %o2, %i2
	sub %l5, %o2, %i3
	srl %i1, %i3, %i1
	sub %l5, %o2, %g1
	sll %g3, %o2, %g4
	or %i2, %i1, %o3
	srl %g3, %g1, %g3
	add %l1, %o2, %i0
	ld	[%l7+8], %l2
	add %o3, %i0, %i1
	or %g4, %g3, %o5
	sll %i1, 3, %i2
	add %l1, %o2, %g2
	mov	%l2, %l1
	srl %i1, 29, %i1
	add %o5, %g2, %g3
	sll %g3, 3, %g4
	or %i2, %i1, %i4
	srl %g3, 29, %g3
	add %o3, %i4, %i0
	st	%i4, [%l6+4]
	add %o1, %i0, %i1
	sethi	%hi(16777216), %g2
	add	%o1, %g2, %o1
	or %g4, %g3, %i5
	sub %l5, %i0, %i3
	add %o5, %i5, %g2
	st	%i5, [%l6+124]
	sll %i1, %i0, %i2
	add %o1, %g2, %g3
	srl %i1, %i3, %i1
	sub %l5, %g2, %g1
	sll %g3, %g2, %g4
	or %i2, %i1, %o4
	srl %g3, %g1, %g3
	add %l1, %i4, %i0
	ld	[%l7+12], %l2
	or %g4, %g3, %l0
	add %o4, %i0, %i1
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+132], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l6+8]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l6+128]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+16], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+136], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l6+12]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l6+132]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+20], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+140], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l6+16]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l6+136]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+24], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+144], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l6+20]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l6+140]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+28], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+148], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l6+24]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l6+144]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+32], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+152], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l6+28]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l6+148]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+36], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+156], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l6+32]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l6+152]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+40], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+160], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l6+36]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l6+156]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+44], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+164], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l6+40]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l6+160]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+48], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+168], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l6+44]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l6+164]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+52], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+172], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l6+48]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l6+168]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+56], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+176], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l6+52]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l6+172]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+60], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+180], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l6+56]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l6+176]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+64], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+184], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l6+60]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l6+180]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+68], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+188], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l6+64]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l6+184]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+72], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+192], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l6+68]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l6+188]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+76], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+196], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l6+72]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l6+192]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+80], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+200], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l6+76]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l6+196]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+84], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+204], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l6+80]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l6+200]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+88], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+208], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l6+84]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l6+204]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+92], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+212], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l6+88]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l6+208]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+96], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+216], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l6+92]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l6+212]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+100], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+220], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l6+96]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l6+216]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+104], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+224], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l6+100]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l6+220]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	ld	[%l6+4], %l2
	add	%fp, -496, %l7
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l6+124], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%fp-496]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l7+120]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l6+8], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l6+128], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l7+4]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l7+124]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l6+12], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l6+132], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l7+8]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l7+128]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l6+16], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l6+136], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l7+12]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l7+132]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l6+20], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l6+140], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l7+16]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l7+136]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l6+24], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l6+144], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l7+20]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l7+140]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l6+28], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l6+148], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l7+24]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l7+144]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l6+32], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l6+152], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l7+28]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l7+148]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l6+36], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l6+156], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l7+32]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l7+152]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l6+40], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l6+160], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l7+36]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l7+156]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l6+44], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l6+164], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l7+40]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l7+160]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l6+48], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l6+168], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l7+44]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l7+164]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l6+52], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l6+172], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l7+48]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l7+168]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l6+56], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l6+176], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l7+52]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l7+172]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l6+60], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l6+180], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l7+56]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l7+176]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l6+64], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l6+184], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l7+60]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l7+180]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l6+68], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l6+188], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l7+64]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l7+184]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l6+72], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l6+192], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l7+68]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l7+188]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l6+76], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l6+196], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l7+72]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l7+192]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l6+80], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l6+200], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l7+76]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l7+196]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l6+84], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l6+204], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l7+80]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l7+200]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l6+88], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l6+208], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l7+84]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l7+204]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l6+92], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l6+212], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l7+88]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l7+208]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l6+96], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l6+216], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l7+92]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l7+212]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l6+100], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l6+220], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	st	%o7, [%l7+96]
	add %o3, %i0, %i1
	or %o2, %g4, %o2
	sub %l5, %i0, %i3
	add %l0, %o2, %g2
	st	%o2, [%l7+216]
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l6+104], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	st	%i4, [%l7+100]
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	add %o5, %i5, %g2
	sub %l5, %i0, %i3
	st	%i5, [%l7+220]
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	ld	[%l7], %i0
	add	%o4, %i4, %g2
	ld	[%l7+120], %l1
	add	%g2, %i0, %i1
	ld	[%l7+4], %l2
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+124], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	add %o4, %o7, %i0
	or %o2, %g4, %o2
	add %o3, %i0, %i1
	add %l0, %o2, %g2
	sub %l5, %i0, %i3
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+8], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+128], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	add %o3, %i4, %i0
	or %i5, %g4, %i5
	add %o4, %i0, %i1
	sub %l5, %i0, %i3
	add %o5, %i5, %g2
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+12], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	add	%l3, %o2, %o0
	add	%l4, %i5, %o1
	add	%l3, %o7, %l3
	add	%l4, %i4, %l4
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+132], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	xor %l3, %l4, %l3
	or %o2, %g4, %o2
	sub %l5, %l4, %i3
	sll %l3, %l4, %i2
	xor %o0, %o1, %o0
	srl %l3, %i3, %l3
	sub %l5, %o1, %g1
	sll %o0, %o1, %g4
	or %i2, %l3, %l3
	srl %o0, %g1, %o0
	add %l3, %o7, %l3
	add %o4, %o7, %i0
	or %g4, %o0, %o0
	add %o3, %i0, %i1
	add %o0, %o2, %o0
	add %l0, %o2, %g2
	sub %l5, %i0, %i3
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+16], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+136], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	xor %l4, %l3, %l4
	or %i5, %g4, %i5
	sub %l5, %l3, %i3
	sll %l4, %l3, %i2
	xor %o1, %o0, %o1
	srl %l4, %i3, %l4
	sub %l5, %o0, %g1
	sll %o1, %o0, %g4
	or %i2, %l4, %l4
	srl %o1, %g1, %o1
	add %l4, %i4, %l4
	or %g4, %o1, %o1
	add %o3, %i4, %i0
	add %o1, %i5, %o1
	add %o4, %i0, %i1
	sub %l5, %i0, %i3
	add %o5, %i5, %g2
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+20], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+140], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	xor %l3, %l4, %l3
	or %o2, %g4, %o2
	sub %l5, %l4, %i3
	sll %l3, %l4, %i2
	xor %o0, %o1, %o0
	srl %l3, %i3, %l3
	sub %l5, %o1, %g1
	sll %o0, %o1, %g4
	or %i2, %l3, %l3
	srl %o0, %g1, %o0
	add %l3, %o7, %l3
	add %o4, %o7, %i0
	or %g4, %o0, %o0
	add %o3, %i0, %i1
	add %o0, %o2, %o0
	add %l0, %o2, %g2
	sub %l5, %i0, %i3
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+24], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+144], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	xor %l4, %l3, %l4
	or %i5, %g4, %i5
	sub %l5, %l3, %i3
	sll %l4, %l3, %i2
	xor %o1, %o0, %o1
	srl %l4, %i3, %l4
	sub %l5, %o0, %g1
	sll %o1, %o0, %g4
	or %i2, %l4, %l4
	srl %o1, %g1, %o1
	add %l4, %i4, %l4
	or %g4, %o1, %o1
	add %o3, %i4, %i0
	add %o1, %i5, %o1
	add %o4, %i0, %i1
	sub %l5, %i0, %i3
	add %o5, %i5, %g2
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+28], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+148], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	xor %l3, %l4, %l3
	or %o2, %g4, %o2
	sub %l5, %l4, %i3
	sll %l3, %l4, %i2
	xor %o0, %o1, %o0
	srl %l3, %i3, %l3
	sub %l5, %o1, %g1
	sll %o0, %o1, %g4
	or %i2, %l3, %l3
	srl %o0, %g1, %o0
	add %l3, %o7, %l3
	add %o4, %o7, %i0
	or %g4, %o0, %o0
	add %o3, %i0, %i1
	add %o0, %o2, %o0
	add %l0, %o2, %g2
	sub %l5, %i0, %i3
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+32], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+152], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	xor %l4, %l3, %l4
	or %i5, %g4, %i5
	sub %l5, %l3, %i3
	sll %l4, %l3, %i2
	xor %o1, %o0, %o1
	srl %l4, %i3, %l4
	sub %l5, %o0, %g1
	sll %o1, %o0, %g4
	or %i2, %l4, %l4
	srl %o1, %g1, %o1
	add %l4, %i4, %l4
	or %g4, %o1, %o1
	add %o3, %i4, %i0
	add %o1, %i5, %o1
	add %o4, %i0, %i1
	sub %l5, %i0, %i3
	add %o5, %i5, %g2
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+36], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+156], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	xor %l3, %l4, %l3
	or %o2, %g4, %o2
	sub %l5, %l4, %i3
	sll %l3, %l4, %i2
	xor %o0, %o1, %o0
	srl %l3, %i3, %l3
	sub %l5, %o1, %g1
	sll %o0, %o1, %g4
	or %i2, %l3, %l3
	srl %o0, %g1, %o0
	add %l3, %o7, %l3
	add %o4, %o7, %i0
	or %g4, %o0, %o0
	add %o3, %i0, %i1
	add %o0, %o2, %o0
	add %l0, %o2, %g2
	sub %l5, %i0, %i3
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+40], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+160], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	xor %l4, %l3, %l4
	or %i5, %g4, %i5
	sub %l5, %l3, %i3
	sll %l4, %l3, %i2
	xor %o1, %o0, %o1
	srl %l4, %i3, %l4
	sub %l5, %o0, %g1
	sll %o1, %o0, %g4
	or %i2, %l4, %l4
	srl %o1, %g1, %o1
	add %l4, %i4, %l4
	or %g4, %o1, %o1
	add %o3, %i4, %i0
	add %o1, %i5, %o1
	add %o4, %i0, %i1
	sub %l5, %i0, %i3
	add %o5, %i5, %g2
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+44], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+164], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	xor %l3, %l4, %l3
	or %o2, %g4, %o2
	sub %l5, %l4, %i3
	sll %l3, %l4, %i2
	xor %o0, %o1, %o0
	srl %l3, %i3, %l3
	sub %l5, %o1, %g1
	sll %o0, %o1, %g4
	or %i2, %l3, %l3
	srl %o0, %g1, %o0
	add %l3, %o7, %l3
	add %o4, %o7, %i0
	or %g4, %o0, %o0
	add %o3, %i0, %i1
	add %o0, %o2, %o0
	add %l0, %o2, %g2
	sub %l5, %i0, %i3
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+48], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+168], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	xor %l4, %l3, %l4
	or %i5, %g4, %i5
	sub %l5, %l3, %i3
	sll %l4, %l3, %i2
	xor %o1, %o0, %o1
	srl %l4, %i3, %l4
	sub %l5, %o0, %g1
	sll %o1, %o0, %g4
	or %i2, %l4, %l4
	srl %o1, %g1, %o1
	add %l4, %i4, %l4
	or %g4, %o1, %o1
	add %o3, %i4, %i0
	add %o1, %i5, %o1
	add %o4, %i0, %i1
	sub %l5, %i0, %i3
	add %o5, %i5, %g2
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+52], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+172], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	xor %l3, %l4, %l3
	or %o2, %g4, %o2
	sub %l5, %l4, %i3
	sll %l3, %l4, %i2
	xor %o0, %o1, %o0
	srl %l3, %i3, %l3
	sub %l5, %o1, %g1
	sll %o0, %o1, %g4
	or %i2, %l3, %l3
	srl %o0, %g1, %o0
	add %l3, %o7, %l3
	add %o4, %o7, %i0
	or %g4, %o0, %o0
	add %o3, %i0, %i1
	add %o0, %o2, %o0
	add %l0, %o2, %g2
	sub %l5, %i0, %i3
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+56], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+176], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	xor %l4, %l3, %l4
	or %i5, %g4, %i5
	sub %l5, %l3, %i3
	sll %l4, %l3, %i2
	xor %o1, %o0, %o1
	srl %l4, %i3, %l4
	sub %l5, %o0, %g1
	sll %o1, %o0, %g4
	or %i2, %l4, %l4
	srl %o1, %g1, %o1
	add %l4, %i4, %l4
	or %g4, %o1, %o1
	add %o3, %i4, %i0
	add %o1, %i5, %o1
	add %o4, %i0, %i1
	sub %l5, %i0, %i3
	add %o5, %i5, %g2
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+60], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+180], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	xor %l3, %l4, %l3
	or %o2, %g4, %o2
	sub %l5, %l4, %i3
	sll %l3, %l4, %i2
	xor %o0, %o1, %o0
	srl %l3, %i3, %l3
	sub %l5, %o1, %g1
	sll %o0, %o1, %g4
	or %i2, %l3, %l3
	srl %o0, %g1, %o0
	add %l3, %o7, %l3
	add %o4, %o7, %i0
	or %g4, %o0, %o0
	add %o3, %i0, %i1
	add %o0, %o2, %o0
	add %l0, %o2, %g2
	sub %l5, %i0, %i3
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+64], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+184], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	xor %l4, %l3, %l4
	or %i5, %g4, %i5
	sub %l5, %l3, %i3
	sll %l4, %l3, %i2
	xor %o1, %o0, %o1
	srl %l4, %i3, %l4
	sub %l5, %o0, %g1
	sll %o1, %o0, %g4
	or %i2, %l4, %l4
	srl %o1, %g1, %o1
	add %l4, %i4, %l4
	or %g4, %o1, %o1
	add %o3, %i4, %i0
	add %o1, %i5, %o1
	add %o4, %i0, %i1
	sub %l5, %i0, %i3
	add %o5, %i5, %g2
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+68], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+188], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	xor %l3, %l4, %l3
	or %o2, %g4, %o2
	sub %l5, %l4, %i3
	sll %l3, %l4, %i2
	xor %o0, %o1, %o0
	srl %l3, %i3, %l3
	sub %l5, %o1, %g1
	sll %o0, %o1, %g4
	or %i2, %l3, %l3
	srl %o0, %g1, %o0
	add %l3, %o7, %l3
	add %o4, %o7, %i0
	or %g4, %o0, %o0
	add %o3, %i0, %i1
	add %o0, %o2, %o0
	add %l0, %o2, %g2
	sub %l5, %i0, %i3
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+72], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+192], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	xor %l4, %l3, %l4
	or %i5, %g4, %i5
	sub %l5, %l3, %i3
	sll %l4, %l3, %i2
	xor %o1, %o0, %o1
	srl %l4, %i3, %l4
	sub %l5, %o0, %g1
	sll %o1, %o0, %g4
	or %i2, %l4, %l4
	srl %o1, %g1, %o1
	add %l4, %i4, %l4
	or %g4, %o1, %o1
	add %o3, %i4, %i0
	add %o1, %i5, %o1
	add %o4, %i0, %i1
	sub %l5, %i0, %i3
	add %o5, %i5, %g2
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+76], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+196], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	xor %l3, %l4, %l3
	or %o2, %g4, %o2
	sub %l5, %l4, %i3
	sll %l3, %l4, %i2
	xor %o0, %o1, %o0
	srl %l3, %i3, %l3
	sub %l5, %o1, %g1
	sll %o0, %o1, %g4
	or %i2, %l3, %l3
	srl %o0, %g1, %o0
	add %l3, %o7, %l3
	add %o4, %o7, %i0
	or %g4, %o0, %o0
	add %o3, %i0, %i1
	add %o0, %o2, %o0
	add %l0, %o2, %g2
	sub %l5, %i0, %i3
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+80], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+200], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	xor %l4, %l3, %l4
	or %i5, %g4, %i5
	sub %l5, %l3, %i3
	sll %l4, %l3, %i2
	xor %o1, %o0, %o1
	srl %l4, %i3, %l4
	sub %l5, %o0, %g1
	sll %o1, %o0, %g4
	or %i2, %l4, %l4
	srl %o1, %g1, %o1
	add %l4, %i4, %l4
	or %g4, %o1, %o1
	add %o3, %i4, %i0
	add %o1, %i5, %o1
	add %o4, %i0, %i1
	sub %l5, %i0, %i3
	add %o5, %i5, %g2
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+84], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+204], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	xor %l3, %l4, %l3
	or %o2, %g4, %o2
	sub %l5, %l4, %i3
	sll %l3, %l4, %i2
	xor %o0, %o1, %o0
	srl %l3, %i3, %l3
	sub %l5, %o1, %g1
	sll %o0, %o1, %g4
	or %i2, %l3, %l3
	srl %o0, %g1, %o0
	add %l3, %o7, %l3
	add %o4, %o7, %i0
	or %g4, %o0, %o0
	add %o3, %i0, %i1
	add %o0, %o2, %o0
	add %l0, %o2, %g2
	sub %l5, %i0, %i3
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+88], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+208], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	xor %l4, %l3, %l4
	or %i5, %g4, %i5
	sub %l5, %l3, %i3
	sll %l4, %l3, %i2
	xor %o1, %o0, %o1
	srl %l4, %i3, %l4
	sub %l5, %o0, %g1
	sll %o1, %o0, %g4
	or %i2, %l4, %l4
	srl %o1, %g1, %o1
	add %l4, %i4, %l4
	or %g4, %o1, %o1
	add %o3, %i4, %i0
	add %o1, %i5, %o1
	add %o4, %i0, %i1
	sub %l5, %i0, %i3
	add %o5, %i5, %g2
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+92], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+212], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	xor %l3, %l4, %l3
	or %o2, %g4, %o2
	sub %l5, %l4, %i3
	sll %l3, %l4, %i2
	xor %o0, %o1, %o0
	srl %l3, %i3, %l3
	sub %l5, %o1, %g1
	sll %o0, %o1, %g4
	or %i2, %l3, %l3
	srl %o0, %g1, %o0
	add %l3, %o7, %l3
	add %o4, %o7, %i0
	or %g4, %o0, %o0
	add %o3, %i0, %i1
	add %o0, %o2, %o0
	add %l0, %o2, %g2
	sub %l5, %i0, %i3
	sll %i1, %i0, %o3
	add %o5, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %o5
	or %o3, %i2, %o3
	srl %g3, %g1, %g4
	add %l2, %o7, %i0
	ld	[%l7+96], %l2
	or %o5, %g4, %o5
	add %o3, %i0, %i1
	sll %i1, 3, %i4
	add %l1, %o2, %g2
	ld	[%l7+216], %l1
	srl %i1, 29, %i2
	add %o5, %g2, %g3
	sll %g3, 3, %i5
	or %i4, %i2, %i4
	srl %g3, 29, %g4
	xor %l4, %l3, %l4
	or %i5, %g4, %i5
	sub %l5, %l3, %i3
	sll %l4, %l3, %i2
	xor %o1, %o0, %o1
	srl %l4, %i3, %l4
	sub %l5, %o0, %g1
	sll %o1, %o0, %g4
	or %i2, %l4, %l4
	srl %o1, %g1, %o1
	add %l4, %i4, %l4
	or %g4, %o1, %o1
	add %o3, %i4, %i0
	add %o1, %i5, %o1
	add %o4, %i0, %i1
	sub %l5, %i0, %i3
	add %o5, %i5, %g2
	sll %i1, %i0, %o4
	add %l0, %g2, %g3
	srl %i1, %i3, %i2
	sub %l5, %g2, %g1
	sll %g3, %g2, %l0
	or %o4, %i2, %o4
	srl %g3, %g1, %g4
	add %l2, %i4, %i0
	ld	[%l7+100], %l2
	add %o4, %i0, %i1
	or %l0, %g4, %l0
	sll %i1, 3, %o7
	add %l1, %i5, %g2
	ld	[%l7+220], %l1
	srl %i1, 29, %i2
	add %l0, %g2, %g3
	sll %g3, 3, %o2
	or %o7, %i2, %o7
	srl %g3, 29, %g4
	xor %l3, %l4, %l3
	or %o2, %g4, %o2
	sub %l5, %l4, %i3
	sll %l3, %l4, %i2
	xor %o0, %o1, %o0
	srl %l3, %i3, %l3
	sub %l5, %o1, %g1
	sll %o0, %o1, %g4
	or %i2, %l3, %l3
	srl %o0, %g1, %o0
	add %l3, %o7, %l3
	ld	[%fp+68], %i5
	or	%o0, %g4, %o0
	ldub	[%i5+12], %i0
	ldub	[%i5+13], %g3
	sll	%i0, 24, %i0
	ldub	[%i5+14], %g2
	sll	%g3, 16, %g3
	or	%g3, %i0, %g3
	sll	%g2, 8, %g2
	ldub	[%i5+15], %i0
	or	%g2, %g3, %g2
	or	%i0, %g2, %l7
	cmp	%l7, %l3
	be	.LL9
	add	%o0, %o2, %o0
	cmp	%l7, %o0
	bne	.LL22
	ld	[%fp-504], %i4
.LL9:
	add	%o4, %o7, %i0
	add	%l0, %o2, %g2
	add	%o3, %i0, %i1
	add	%o5, %g2, %g3
	sub	%l5, %i0, %i3
	sub	%l5, %g2, %g1
	srl	%i1, %i3, %i2
	srl	%g3, %g1, %g4
	sll	%i1, %i0, %o3
	sll	%g3, %g2, %o5
	add	%l2, %o7, %i0
	add	%l1, %o2, %g2
	or	%o3, %i2, %o3
	or	%o5, %g4, %o5
	add	%o3, %i0, %i1
	add	%o5, %g2, %g3
	srl	%i1, 29, %i2
	srl	%g3, 29, %g4
	sll	%i1, 3, %i4
	sll	%g3, 3, %i5
	ld	[%fp-500], %g3
	or	%i4, %i2, %i4
	or	%i5, %g4, %i5
	xor	%l4, %l3, %l4
	xor	%o1, %o0, %o1
	ld	[%fp-500], %o7
	sll	%l4, %l3, %i2
	sll	%o1, %o0, %g4
	sub	%l5, %l3, %i3
	sub	%l5, %o0, %g1
	ld	[%fp+68], %i1
	srl	%g3, 24, %g2
	srl	%l4, %i3, %l4
	srl	%o1, %g1, %o1
	or	%l4, %i2, %l4
	or	%o1, %g4, %o1
	srl	%g3, 16, %g3
	srl	%o7, 8, %i0
	cmp	%l7, %l3
	stb	%g2, [%i1+20]
	stb	%g3, [%i1+21]
	stb	%i0, [%i1+22]
	add	%l4, %i4, %l4
	add	%o1, %i5, %o1
	bne	.LL10
	stb	%o7, [%i1+23]
	ldub	[%i1+8], %g2
	ldub	[%i1+9], %i0
	sll	%g2, 24, %g2
	ldub	[%i1+10], %g3
	sll	%i0, 16, %i0
	or	%i0, %g2, %i0
	sll	%g3, 8, %g3
	ldub	[%i1+11], %g2
	or	%g3, %i0, %g3
	or	%g2, %g3, %g2
	cmp	%g2, %l4
	bne	.LL23
	ld	[%fp+68], %g3
	ld	[%fp+72], %i2
	ld	[%fp-508], %i3
	ld	[%fp-504], %i4
	sub	%i2, %i3, %i0
	ld	[%fp+68], %i5
	add	%i0, -1, %i0
	srl	%i4, 24, %g2
	srl	%i4, 16, %g3
	srl	%i4, 8, %i1
	stb	%i4, [%i5+19]
	stb	%g2, [%i5+16]
	stb	%g3, [%i5+17]
	stb	%i1, [%i5+18]
	b	.LL21
	sll	%i0, 1, %i0
.LL10:
	ld	[%fp+68], %g3
.LL23:
	ldub	[%g3+12], %g2
	ldub	[%g3+13], %i0
	sll	%g2, 24, %g2
	ldub	[%g3+14], %g3
	sll	%i0, 16, %i0
	ld	[%fp+68], %o7
	or	%i0, %g2, %i0
	sll	%g3, 8, %g3
	ldub	[%o7+15], %g2
	or	%g3, %i0, %g3
	or	%g2, %g3, %g2
	cmp	%g2, %o0
	bne	.LL22
	ld	[%fp-504], %i4
	ldub	[%o7+8], %g2
	ldub	[%o7+9], %i0
	sll	%g2, 24, %g2
	ldub	[%o7+10], %g3
	sll	%i0, 16, %i0
	or	%i0, %g2, %i0
	sll	%g3, 8, %g3
	ldub	[%o7+11], %g2
	or	%g3, %i0, %g3
	or	%g2, %g3, %g2
	cmp	%g2, %o1
	bne	.LL24
	sethi	%hi(33554432), %g2
	ld	[%fp+72], %i1
	sethi	%hi(16777216), %g2
	ld	[%fp-508], %i2
	ld	[%fp-504], %i3
	sub	%i1, %i2, %i0
	add	%i3, %g2, %g2
	add	%i0, -1, %i0
	sll	%i0, 1, %i0
	srl	%g2, 24, %g3
	srl	%g2, 16, %i1
	srl	%g2, 8, %i2
	stb	%g2, [%o7+19]
	stb	%g3, [%o7+16]
	stb	%i1, [%o7+17]
	stb	%i2, [%o7+18]
	b	.LL21
	or	%i0, 1, %i0
.LL22:
	sethi	%hi(33554432), %g2
.LL24:
	add	%i4, %g2, %o1
	sethi	%hi(-16777216), %i4
	andcc	%o1, %i4, %g0
	bne	.LL12
	ld	[%fp-500], %o0
	sethi	%hi(65536), %i2
	sethi	%hi(16776192), %g2
	or	%g2, 1023, %i3
	add	%o1, %i2, %g3
	and	%g3, %i3, %o1
	sethi	%hi(16711680), %i0
	andcc	%o1, %i0, %g0
	bne,a	.LL25
	ld	[%fp-508], %i5
	sethi	%hi(64512), %g3
	or	%g3, 768, %i1
	or	%g3, 1023, %g3
	add	%o1, 256, %g2
	and	%g2, %g3, %o1
	andcc	%o1, %i1, %g0
	bne,a	.LL25
	ld	[%fp-508], %i5
	add	%o1, 1, %g2
	andcc	%g2, 255, %o1
	bne,a	.LL12
	st	%o0, [%fp-500]
	sethi	%hi(16777216), %g2
	add	%o0, %g2, %o0
	andcc	%o0, %i4, %g0
	bne,a	.LL12
	st	%o0, [%fp-500]
	add	%o0, %i2, %g2
	and	%g2, %i3, %o0
	andcc	%o0, %i0, %g0
	bne,a	.LL12
	st	%o0, [%fp-500]
	add	%o0, 256, %g2
	and	%g2, %g3, %o0
	andcc	%o0, %i1, %g0
	bne,a	.LL12
	st	%o0, [%fp-500]
	add	%o0, 1, %g2
	and	%g2, 255, %o0
	st	%o0, [%fp-500]
.LL12:
	ld	[%fp-508], %i5
.LL25:
	cmp	%i5, 0
	bne	.LL7
	st	%o1, [%fp-504]
.LL6:
	ld	[%fp-500], %g3
	srl	%o1, 24, %i1
	ld	[%fp-500], %o7
	srl	%g3, 24, %g2
	ld	[%fp+68], %i4
	srl	%o7, 8, %i0
	srl	%g3, 16, %g3
	srl	%o1, 16, %i2
	srl	%o1, 8, %i3
	stb	%i0, [%i4+22]
	stb	%o1, [%i4+19]
	stb	%g2, [%i4+20]
	stb	%g3, [%i4+21]
	stb	%o7, [%i4+23]
	stb	%i1, [%i4+16]
	stb	%i2, [%i4+17]
	stb	%i3, [%i4+18]
	ld	[%fp+72], %i5
	sll	%i5, 1, %i0
.LL21:
	ret
	restore
.LLFE2:
.LLfe2:
	.size	 crunch__FP11RC5UnitWorkUi,.LLfe2-crunch__FP11RC5UnitWorkUi
	.align 4
	.global rc5_unit_func_ultrasparc_crunch
	.type	 rc5_unit_func_ultrasparc_crunch,#function
	.proc	016
rc5_unit_func_ultrasparc_crunch:
.LLFB3:
	!#PROLOGUE# 0
	save	%sp, -112, %sp
.LLCFI1:
	!#PROLOGUE# 1
	mov	%i0, %o0
	call	crunch__FP11RC5UnitWorkUi, 0
	mov	%i1, %o1
	ret
	restore %g0, %o0, %o0
.LLFE3:
.LLfe3:
	.size	 rc5_unit_func_ultrasparc_crunch,.LLfe3-rc5_unit_func_ultrasparc_crunch

.section	".eh_frame",#alloc,#write
__FRAME_BEGIN__:
	.uaword	.LLECIE1-.LLSCIE1
.LLSCIE1:
	.uaword	0x0
	.byte	0x1
	.byte	0x0
	.byte	0x1
	.byte	0x7c
	.byte	0x65
	.byte	0xc
	.byte	0xe
	.byte	0x0
	.byte	0x9
	.byte	0x65
	.byte	0xf
	.align 4
.LLECIE1:
	.uaword	.LLEFDE1-.LLSFDE1
.LLSFDE1:
	.uaword	.LLSFDE1-__FRAME_BEGIN__
	.uaword	.LLFB1
	.uaword	.LLFE1-.LLFB1
	.align 4
.LLEFDE1:
	.uaword	.LLEFDE3-.LLSFDE3
.LLSFDE3:
	.uaword	.LLSFDE3-__FRAME_BEGIN__
	.uaword	.LLFB2
	.uaword	.LLFE2-.LLFB2
	.byte	0x4
	.uaword	.LLCFI0-.LLFB2
	.byte	0xd
	.byte	0x1e
	.byte	0x2d
	.byte	0x9
	.byte	0x65
	.byte	0x1f
	.align 4
.LLEFDE3:
	.uaword	.LLEFDE5-.LLSFDE5
.LLSFDE5:
	.uaword	.LLSFDE5-__FRAME_BEGIN__
	.uaword	.LLFB3
	.uaword	.LLFE3-.LLFB3
	.byte	0x4
	.uaword	.LLCFI1-.LLFB3
	.byte	0xd
	.byte	0x1e
	.byte	0x2d
	.byte	0x9
	.byte	0x65
	.byte	0x1f
	.align 4
.LLEFDE5:
	.ident	"GCC: (GNU) 2.95.2 19991024 (release)"
