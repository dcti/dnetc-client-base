# $Log: simple.handopt.s,v $
# Revision 1.2  1998/06/15 02:44:30  djones
# First build of UltraSPARC 64-bit/VIS DES client:
# - many configure file tweaks: split up C++, ASM and C files; make "gcc" the
#   compiler.
# - "tr" on SunOS 4.1.4 goes into endless loop when faced with "..-"; change
#   to "..\-".
# - Enable generation of whack16()
#
# Revision 1.1.1.1  1998/06/14 14:23:52  remi
# Initial integration.
#

	.file	"simple.c"
gcc2_compiled.:
	.data
id:	.ascii "@(#)$Id: simple.handopt.s,v 1.2 1998/06/15 02:44:30 djones Exp $"
	.byte 0

	.text
	.align 4
	.global asm_do_s1
	.type	 asm_do_s1,#function
	.proc	016
asm_do_s1:
	!#PROLOGUE# 0
	!# vars= 0, regs= 18/0, args= 0, extra= 84
	add %sp,-184,%sp
	st %i7,[%sp+96]
	sub %sp,-184,%i7	!# set up frame pointer
	st %o7,[%sp+100]
	std %l0,[%sp+104]
	std %l2,[%sp+112]
	std %l4,[%sp+120]
	st %l6,[%sp+128]
	std %i0,[%sp+136]
	std %i2,[%sp+144]
	std %i4,[%sp+152]
	!#PROLOGUE# 1
	ld [%o0+200],%l3
	sethi %hi(1513943058),%g2
	ld [%o0+204],%l5
	or %g2,%lo(1513943058),%g2
	ld [%o0+208],%l6
	sllx %g2, 32, %g2
	or %i7, %g2, %i7	!##
	lduw [%o0+16], %g2	!+
	ldx [%l3+8], %i3	!#
	ldx [%g2], %i2	!#
	lduw [%o0+24], %g2	!+
	ldx [%l3+24], %i1	!#
	ldx [%g2], %i0	!#
	lduw [%o0+20], %g2	!+
	ldx [%l3+16], %o7	!#
	ldx [%g2], %g3	!#
	lduw [%o0+28], %g2	!+
	ldx [%l3+32], %l2	!#
	ldx [%g2], %g2	!#
	xor %i3, %i2, %i3
	xor %i1, %i0, %l4
	xor %o7, %g3, %o7
	xor %l2, %g2, %l2
.LL10:
	lduw [%o0+12], %o5	!+
	andn %i3, %l4, %g2
	andn %i3, %o7, %i5
	ldx [%l3], %i0	!#
	or %i5, %l4, %i2
	xor %i3, %o7, %i4
	ldx [%o5], %o5	!#
	xor %g2, %o7, %g1
	and %l2, %i2, %l1
	lduw [%o0+8], %l0	!+
	xor %g1, %l1, %g4
	xor %l4, %l1, %o1
	ldx [%l3+248], %i1	!#
	and %o1, %i4, %o1
	andn %l4, %o7, %o3
	ldx [%l0], %l0	!#
	andn %o7, %l4, %o4
	xor %i5, %o1, %o2
	xor %i0, %o5, %i0
	or %g4, %o4, %o5
	xor %i1, %l0, %i1
	or %l2, %o2, %o2
	andn %l2, %i4, %l0
	xor %o3, %o2, %o2
	xor %o4, %l0, %o4
	or %i0, %o2, %o3
	lduw [%o0+216], %l3	!+
	or %i0, %o4, %o4
	xor %o1, %o3, %o3
	lduw [%o0+212], %o0	!+
	xor %g4, %o4, %o4
	and %i1, %o3, %o3
	ldx [%l5+128], %g4	!#
	xnor %o4, %o3, %o3
	or %g2, %l1, %g2
	lduw [%o0+16], %o4	!+
	xor %g4, %o3, %g4
	stx %g4, [%l6+128]	!#
	andn %o2, %g1, %g1
	xor %g2, %i4, %g2
	andn %i0, %g1, %i4
	ldx [%o4], %o4	!#
	xor %g2, %i4, %i4
	xor %l0, %g2, %g3
	lduw [%o0+24], %o3	!+
	andn %o2, %g3, %g3
	xor %o5, %g1, %g1
	lduw [%o0+20], %g4	!+
	and %i0, %g3, %o1
	and %i3, %o5, %o5
	ldx [%o3], %o3	!#
	xor %g1, %o1, %o1
	or %i3, %g3, %g3
	ldx [%g4], %g4	!#
	and %i1, %o1, %o7
	andn %o2, %o5, %g1
	ldx [%l5+240], %i3	!#
	xor %i4, %o7, %o7
	andn %g2, %g1, %g2
	lduw [%o0+28], %i4	!+
	xor %i3, %o7, %i3
	stx %i3, [%l6+240]	!#
	or %g2, %i5, %g2
	andn %g2, %i0, %i3
	or %i0, %i5, %i5
	ldx [%i4], %i4	!#
	xor %g3, %i3, %i3
	xor %g1, %i5, %i5
	ldx [%l3+8], %g1	!#
	or %g2, %o2, %g2
	andn %i1, %i3, %i3
	ldx [%l5+64], %o2	!#
	xnor %i5, %i3, %i3
	xor %g2, %l4, %g2
	ldx [%l3+32], %l2	!#
	xor %o2, %i3, %o2
	stx %o2, [%l6+64]	!#
	xor %i2, %o5, %o5
	andn %o5, %l1, %o5
	andn %i0, %g2, %g2
	ldx [%l5+176], %l1	!#
	andn %o1, %l0, %o1
	xor %o5, %g2, %g2
	ldx [%l3+16], %o7	!#
	xor %o1, %i5, %o1
	or %i1, %g2, %g2
	ldx [%l3+24], %i5	!#
	xor %g1, %o4, %i3
	xnor %o1, %g2, %g2
	lduw [%o0+204], %l5	!+
	xor %l1, %g2, %l1
	stx %l1, [%l6+176]	!#
	xor %i5, %o3, %l4
	xor %o7, %g4, %o7
	xor %l2, %i4, %l2
				!# end of included file
	cmp %l5,0
	bne .LL10
	ld [%o0+208],%l6
	srlx %i7, 32, %o0	!##
	!#EPILOGUE#
	ld [%sp+96],%i7
	ld [%sp+100],%o7
	ldd [%sp+104],%l0
	ldd [%sp+112],%l2
	ldd [%sp+120],%l4
	ld [%sp+128],%l6
	ldd [%sp+136],%i0
	ldd [%sp+144],%i2
	ldd [%sp+152],%i4
	retl
	add %sp,184,%sp
.LLfe1:
	.size	 asm_do_s1,.LLfe1-asm_do_s1
	.align 4
	.global asm_do_s1_s3
	.type	 asm_do_s1_s3,#function
	.proc	016
asm_do_s1_s3:
	!#PROLOGUE# 0
	!# vars= 0, regs= 18/0, args= 0, extra= 84
	add %sp,-184,%sp
	st %i7,[%sp+96]
	sub %sp,-184,%i7	!# set up frame pointer
	st %o7,[%sp+100]
	std %l0,[%sp+104]
	std %l2,[%sp+112]
	std %l4,[%sp+120]
	std %l6,[%sp+128]
	std %i0,[%sp+136]
	std %i2,[%sp+144]
	std %i4,[%sp+152]
	!#PROLOGUE# 1
	mov %o0,%l0
	ld [%l0+200],%l5
	sethi %hi(1513943058),%g2
	ld [%l0+204],%l6
	or %g2,%lo(1513943058),%g2
	ld [%l0+208],%l7
	sllx %g2, 32, %g2
	or %i7, %g2, %i7	!##
	lduw [%l0+16], %g2	!+
	ldx [%l5+8], %i3	!#
	ldx [%g2], %i2	!#
	lduw [%l0+24], %g2	!+
	ldx [%l5+24], %i1	!#
	ldx [%g2], %i0	!#
	lduw [%l0+20], %g2	!+
	ldx [%l5+16], %o1	!#
	ldx [%g2], %g3	!#
	lduw [%l0+28], %g2	!+
	ldx [%l5+32], %o0	!#
	ldx [%g2], %g2	!#
	xor %i3, %i2, %l3
	xor %i1, %i0, %o2
	xor %o1, %g3, %o1
	xor %o0, %g2, %o0
.LL15:
	lduw [%l0+12], %o4	!+
	andn %l3, %o2, %g2
	andn %l3, %o1, %i5
	ldx [%l5], %l4	!#
	or %i5, %o2, %i0
	xor %l3, %o1, %i4
	ldx [%o4], %o4	!#
	xor %g2, %o1, %o5
	and %o0, %i0, %l2
	lduw [%l0+8], %l1	!+
	xor %o5, %l2, %g1
	xor %o2, %l2, %i3
	ldx [%l5+248], %g4	!#
	and %i3, %i4, %i3
	andn %o2, %o1, %o7
	ldx [%l1], %l1	!#
	andn %o1, %o2, %o3
	xor %i5, %i3, %i2
	xor %l4, %o4, %l4
	or %g1, %o3, %o4
	xor %g4, %l1, %g4
	or %o0, %i2, %i2
	andn %o0, %i4, %l1
	xor %o7, %i2, %i2
	xor %o3, %l1, %o3
	or %l4, %i2, %o7
	or %l4, %o3, %o3
	xor %i3, %o7, %o7
	xor %g1, %o3, %o3
	and %g4, %o7, %o7
	ldx [%l6+128], %g1	!#
	xnor %o3, %o7, %o7
	or %g2, %l2, %g2
	lduw [%l0+60], %o3	!+
	xor %g1, %o7, %g1
	stx %g1, [%l7+128]	!#
	andn %i2, %o5, %o5
	xor %g2, %i4, %g2
	andn %l4, %o5, %i4
	ldx [%o3], %o3	!#
	xor %g2, %i4, %i4
	xor %l1, %g2, %g3
	lduw [%l0+64], %o7	!+
	andn %i2, %g3, %g3
	xor %o4, %o5, %o5
	lduw [%l0+76], %g1	!+
	and %l4, %g3, %i3
	and %l3, %o4, %o4
	ldx [%o7], %o7	!#
	xor %o5, %i3, %i3
	or %l3, %g3, %g3
	ldx [%g1], %g1	!#
	and %g4, %i3, %o1
	andn %i2, %o4, %o5
	ldx [%l6+240], %i1	!#
	xor %i4, %o1, %o1
	andn %g2, %o5, %g2
	lduw [%l0+72], %i4	!+
	xor %i1, %o1, %i1
	stx %i1, [%l7+240]	!#
	or %g2, %i5, %g2
	andn %g2, %l4, %i1
	or %l4, %i5, %i5
	ldx [%i4], %i4	!#
	xor %g3, %i1, %i1
	xor %o5, %i5, %i5
	ldx [%l5+64], %o5	!#
	or %g2, %i2, %g2
	andn %g4, %i1, %i1
	ldx [%l6+64], %i2	!#
	xnor %i5, %i1, %i1
	xor %g2, %o2, %g2
	ldx [%l5+96], %o0	!#
	xor %i2, %i1, %i2
	stx %i2, [%l7+64]	!#
	xor %i0, %o4, %o4
	andn %o4, %l2, %o4
	andn %l4, %g2, %g2
	ldx [%l6+176], %l2	!#
	andn %i3, %l1, %i3
	xor %o4, %g2, %g2
	ldx [%l5+88], %o2	!#
	xor %i3, %i5, %i3
	or %g4, %g2, %g2
	ldx [%l5+72], %i5	!#
	xor %o5, %o3, %l4
	xnor %i3, %g2, %g2
	xor %l2, %g2, %l2
	stx %l2, [%l7+176]	!#
	xor %i5, %o7, %l3
	xor %o0, %g1, %o0
	xor %o2, %i4, %o2
				!# end of included file
	lduw [%l0+56], %i4	!+
	xor %l4, %l3, %g3
	and %l3, %o0, %g2
	ldx [%l5+56], %g4	!#
	xor %g3, %o0, %g3
	and %l3, %o2, %i5
	ldx [%i4], %i4	!#
	and %l4, %g3, %g1
	xor %i5, %g3, %i5
	lduw [%l0+68], %i3	!+
	or %o2, %g1, %o3
	andn %o0, %g1, %i2
	ldx [%l5+80], %o1	!#
	xor %g3, %o3, %l2
	xor %i2, %o2, %o5
	ldx [%i3], %i3	!#
	xor %g4, %i4, %g4
	xor %l3, %g1, %i4
	and %g4, %o5, %o7
	xor %o1, %i3, %o1
	andn %i4, %o2, %o4
	xor %o2, %o7, %o7
	lduw [%l0+216], %l5	!+
	or %g4, %o4, %i3
	or %o1, %o7, %o7
	lduw [%l0+212], %l0	!+
	andn %g3, %o4, %g3
	xor %l2, %i3, %i3
	ldx [%l6+40], %l1	!#
	xor %i3, %o7, %o7
	xor %g3, %g2, %i3
	or %o5, %g3, %o5
	and %o0, %o7, %i0
	lduw [%l0+16], %i1	!+
	xor %l1, %o7, %l1
	stx %l1, [%l7+40]	!#
	xor %i0, %i4, %i0
	or %g2, %g1, %g2
	or %g4, %i3, %i3
	ldx [%i1], %i1	!#
	xor %o5, %g2, %o5
	xor %g2, %o2, %l1
	lduw [%l0+24], %i4	!+
	or %g4, %o5, %o2
	or %l4, %o4, %o7
	lduw [%l0+20], %g1	!+
	xor %l1, %i3, %i3
	xor %o7, %o3, %o3
	ldx [%i4], %i4	!#
	xor %o3, %o2, %o2
	ldx [%g1], %g1	!#
	andn %o1, %o2, %o2
	andn %o4, %l3, %o4
	ldx [%l6+232], %o3	!#
	xnor %i3, %o2, %o2
	xor %o0, %g2, %g2
	lduw [%l0+28], %i3	!+
	xor %o3, %o2, %o3
	stx %o3, [%l7+232]	!#
	andn %g2, %l2, %g2
	or %i2, %o5, %o5
	and %g4, %g2, %g2
	ldx [%i3], %i3	!#
	or %g4, %o4, %o4
	xor %o5, %g2, %g2
	ldx [%l5+8], %o5	!#
	xor %i5, %o4, %o4
	and %o1, %g2, %g2
	ldx [%l6+120], %i5	!#
	or %l4, %g3, %g3
	xor %o4, %g2, %g2
	ldx [%l5+24], %o4	!#
	xor %i5, %g2, %i5
	stx %i5, [%l7+120]	!#
	andn %g2, %o7, %g2
	andn %i0, %g4, %i0
	or %g4, %g2, %g2
	ldx [%l6+184], %l2	!#
	xor %g3, %l1, %g3
	xor %i0, %g2, %g2
	ldx [%l5+32], %o0	!#
	xor %g3, %i0, %i0
	and %o1, %g2, %g2
	ldx [%l5+16], %o1	!#
	xor %o5, %i1, %l3
	xnor %i0, %g2, %g2
	lduw [%l0+204], %l6	!+
	xor %l2, %g2, %l2
	stx %l2, [%l7+184]	!#
	xor %o4, %i4, %o2
	xor %o1, %g1, %o1
	xor %o0, %i3, %o0
				!# end of included file
	cmp %l6,0
	bne .LL15
	ld [%l0+208],%l7
	srlx %i7, 32, %o0	!##
	!#EPILOGUE#
	ld [%sp+96],%i7
	ld [%sp+100],%o7
	ldd [%sp+104],%l0
	ldd [%sp+112],%l2
	ldd [%sp+120],%l4
	ldd [%sp+128],%l6
	ldd [%sp+136],%i0
	ldd [%sp+144],%i2
	ldd [%sp+152],%i4
	retl
	add %sp,184,%sp
.LLfe2:
	.size	 asm_do_s1_s3,.LLfe2-asm_do_s1_s3
	.align 4
	.global asm_do_s2
	.type	 asm_do_s2,#function
	.proc	016
asm_do_s2:
	!#PROLOGUE# 0
	!# vars= 0, regs= 16/0, args= 0, extra= 84
	add %sp,-176,%sp
	st %i7,[%sp+96]
	sub %sp,-176,%i7	!# set up frame pointer
	st %o7,[%sp+100]
	std %l0,[%sp+104]
	std %l2,[%sp+112]
	std %l4,[%sp+120]
	std %i0,[%sp+128]
	std %i2,[%sp+136]
	std %i4,[%sp+144]
	!#PROLOGUE# 1
	ld [%o0+200],%l0
	sethi %hi(1513943058),%g2
	ld [%o0+204],%l4
	or %g2,%lo(1513943058),%g2
	ld [%o0+208],%l5
	sllx %g2, 32, %g2
	or %i7, %g2, %i7	!##
	lduw [%o0+32], %g2	!+
	ldx [%l0+24], %i3	!#
	ldx [%g2], %i2	!#
	lduw [%o0+52], %g2	!+
	ldx [%l0+64], %i1	!#
	ldx [%g2], %i0	!#
	lduw [%o0+48], %g2	!+
	ldx [%l0+56], %i5	!#
	ldx [%g2], %g3	!#
	lduw [%o0+36], %g2	!+
	ldx [%l0+32], %g4	!#
	ldx [%g2], %g2	!#
	xor %i3, %i2, %l2
	xor %i1, %i0, %i2
	xor %i5, %g3, %i5
	xor %g4, %g2, %g4
.LL20:
	lduw [%o0+40], %o5	!+
	xor %l2, %i2, %l1
	and %i2, %i5, %i1
	ldx [%l0+40], %i3	!#
	xor %l1, %i5, %g1
	andn %l2, %i1, %o4
	ldx [%o5], %o5	!#
	andn %g4, %o4, %o2
	andn %i5, %o4, %i0
	lduw [%o0+44], %o3	!+
	xor %g1, %o2, %g3
	or %i1, %o2, %o2
	ldx [%l0+48], %i4	!#
	andn %o2, %l1, %o2
	or %i0, %g4, %l3
	ldx [%o3], %o3	!#
	xor %i3, %o5, %i3
	andn %i2, %o4, %o5
	lduw [%o0+216], %l0	!+
	or %i3, %o2, %o1
	xor %i4, %o3, %i4
	lduw [%o0+212], %o0	!+
	xor %g3, %o1, %o1
	and %i4, %l3, %o7
	ldx [%l4+96], %o3	!#
	xnor %o1, %o7, %o7
	xor %g3, %i0, %i0
	lduw [%o0+32], %l1	!+
	xor %o3, %o7, %o3
	stx %o3, [%l5+96]	!#
	xor %o4, %o7, %o7
	andn %o7, %g4, %o7
	and %g4, %i0, %i0
	ldx [%l1], %l1	!#
	xor %g1, %o7, %o7
	xor %o5, %i0, %o4
	and %i3, %o4, %o4
	or %g3, %l2, %g3
	lduw [%o0+52], %o3	!+
	xor %o7, %o4, %l2
	xor %i5, %g4, %o7
	lduw [%o0+48], %o4	!+
	andn %o7, %o2, %o2
	xor %g3, %g4, %g3
	ldx [%o3], %o3	!#
	andn %i3, %g3, %i2
	or %o5, %o2, %o5
	ldx [%o4], %o4	!#
	xor %o2, %i2, %i2
	or %g3, %i0, %g3
	lduw [%o0+36], %o2	!+
	or %i4, %i2, %g2
	xor %o5, %o1, %o5
	ldx [%l4+8], %o1	!#
	xor %l2, %g2, %g2
	and %o7, %g3, %o7
	ldx [%o2], %o2	!#
	xor %o1, %g2, %o1
	stx %o1, [%l5+8]	!#
	andn %l3, %o7, %o7
	and %i3, %g3, %g3
	or %i4, %o7, %o7
	ldx [%l0+24], %l3	!#
	xor %o5, %g3, %g3
	xor %g4, %g1, %g2
	ldx [%l4+136], %o1	!#
	xnor %g3, %o7, %o7
	andn %g2, %o5, %g1
	ldx [%l0+64], %o5	!#
	xor %o1, %o7, %o1
	stx %o1, [%l5+136]	!#
	xor %g1, %i2, %g1
	or %i1, %i0, %i0
	and %i3, %i1, %i1
	ldx [%l0+56], %i5	!#
	andn %i3, %g1, %g1
	xor %i0, %i1, %i1
	ldx [%l0+32], %g4	!#
	xor %g2, %g1, %g1
	andn %i4, %i1, %i1
	ldx [%l4+216], %g2	!#
	xor %l3, %l1, %l2
	xnor %g1, %i1, %i1
	lduw [%o0+204], %l4	!+
	xor %g2, %i1, %g2
	stx %g2, [%l5+216]	!#
	xor %o5, %o3, %i2
	xor %i5, %o4, %i5
	xor %g4, %o2, %g4
				!# end of included file
	cmp %l4,0
	bne .LL20
	ld [%o0+208],%l5
	srlx %i7, 32, %o0	!##
	!#EPILOGUE#
	ld [%sp+96],%i7
	ld [%sp+100],%o7
	ldd [%sp+104],%l0
	ldd [%sp+112],%l2
	ldd [%sp+120],%l4
	ldd [%sp+128],%i0
	ldd [%sp+136],%i2
	ldd [%sp+144],%i4
	retl
	add %sp,176,%sp
.LLfe3:
	.size	 asm_do_s2,.LLfe3-asm_do_s2
	.align 4
	.global asm_do_s3
	.type	 asm_do_s3,#function
	.proc	016
asm_do_s3:
	!#PROLOGUE# 0
	!# vars= 0, regs= 18/0, args= 0, extra= 84
	add %sp,-184,%sp
	st %i7,[%sp+96]
	sub %sp,-184,%i7	!# set up frame pointer
	st %o7,[%sp+100]
	std %l0,[%sp+104]
	std %l2,[%sp+112]
	std %l4,[%sp+120]
	std %l6,[%sp+128]
	std %i0,[%sp+136]
	std %i2,[%sp+144]
	std %i4,[%sp+152]
	!#PROLOGUE# 1
	ld [%o0+200],%l3
	sethi %hi(1513943058),%g2
	ld [%o0+204],%l6
	or %g2,%lo(1513943058),%g2
	ld [%o0+208],%l7
	sllx %g2, 32, %g2
	or %i7, %g2, %i7	!##
	lduw [%o0+60], %g2	!+
	ldx [%l3+64], %i3	!#
	ldx [%g2], %i2	!#
	lduw [%o0+64], %g2	!+
	ldx [%l3+72], %i1	!#
	ldx [%g2], %i0	!#
	lduw [%o0+76], %g2	!+
	ldx [%l3+96], %l1	!#
	ldx [%g2], %g3	!#
	lduw [%o0+72], %g2	!+
	ldx [%l3+88], %o7	!#
	ldx [%g2], %g2	!#
	xor %i3, %i2, %l5
	xor %i1, %i0, %l4
	xor %l1, %g3, %l1
	xor %o7, %g2, %o7
.LL25:
	lduw [%o0+56], %g1	!+
	xor %l5, %l4, %g3
	and %l4, %l1, %g2
	ldx [%l3+56], %i1	!#
	xor %g3, %l1, %g3
	and %l4, %o7, %g4
	ldx [%g1], %g1	!#
	and %l5, %g3, %i5
	xor %g4, %g3, %g4
	lduw [%o0+68], %o1	!+
	or %o7, %i5, %o3
	andn %l1, %i5, %i4
	ldx [%l3+80], %i2	!#
	xor %g3, %o3, %l2
	xor %i4, %o7, %o5
	ldx [%o1], %o1	!#
	xor %i1, %g1, %i1
	xor %l4, %i5, %g1
	and %i1, %o5, %o2
	xor %i2, %o1, %i2
	andn %g1, %o7, %o4
	xor %o7, %o2, %o2
	lduw [%o0+216], %l3	!+
	or %i1, %o4, %o1
	or %i2, %o2, %o2
	lduw [%o0+212], %o0	!+
	andn %g3, %o4, %g3
	xor %l2, %o1, %o1
	ldx [%l6+40], %l0	!#
	xor %o1, %o2, %o2
	xor %g3, %g2, %o1
	or %o5, %g3, %o5
	and %l1, %o2, %i0
	lduw [%o0+60], %i3	!+
	xor %l0, %o2, %l0
	stx %l0, [%l7+40]	!#
	xor %i0, %g1, %i0
	or %g2, %i5, %g2
	or %i1, %o1, %o1
	ldx [%i3], %i3	!#
	xor %o5, %g2, %o5
	xor %g2, %o7, %l0
	lduw [%o0+64], %g1	!+
	or %i1, %o5, %o7
	or %l5, %o4, %o2
	lduw [%o0+76], %i5	!+
	xor %l0, %o1, %o1
	xor %o2, %o3, %o3
	ldx [%g1], %g1	!#
	xor %o3, %o7, %o7
	ldx [%i5], %i5	!#
	andn %i2, %o7, %o7
	andn %o4, %l4, %o4
	ldx [%l6+232], %o3	!#
	xnor %o1, %o7, %o7
	xor %l1, %g2, %g2
	lduw [%o0+72], %o1	!+
	xor %o3, %o7, %o3
	stx %o3, [%l7+232]	!#
	andn %g2, %l2, %g2
	or %i4, %o5, %o5
	and %i1, %g2, %g2
	ldx [%o1], %o1	!#
	or %i1, %o4, %o4
	xor %o5, %g2, %g2
	ldx [%l3+64], %o5	!#
	xor %g4, %o4, %o4
	and %i2, %g2, %g2
	ldx [%l6+120], %g4	!#
	or %l5, %g3, %g3
	xor %o4, %g2, %g2
	ldx [%l3+72], %o4	!#
	xor %g4, %g2, %g4
	stx %g4, [%l7+120]	!#
	andn %g2, %o2, %g2
	andn %i0, %i1, %i0
	or %i1, %g2, %g2
	ldx [%l3+96], %l1	!#
	xor %g3, %l0, %g3
	xor %i0, %g2, %g2
	ldx [%l3+88], %o7	!#
	xor %g3, %i0, %i0
	and %i2, %g2, %g2
	ldx [%l6+184], %l2	!#
	xor %o5, %i3, %l5
	xnor %i0, %g2, %g2
	lduw [%o0+204], %l6	!+
	xor %l2, %g2, %l2
	stx %l2, [%l7+184]	!#
	xor %o4, %g1, %l4
	xor %l1, %i5, %l1
	xor %o7, %o1, %o7
				!# end of included file
	cmp %l6,0
	bne .LL25
	ld [%o0+208],%l7
	srlx %i7, 32, %o0	!##
	!#EPILOGUE#
	ld [%sp+96],%i7
	ld [%sp+100],%o7
	ldd [%sp+104],%l0
	ldd [%sp+112],%l2
	ldd [%sp+120],%l4
	ldd [%sp+128],%l6
	ldd [%sp+136],%i0
	ldd [%sp+144],%i2
	ldd [%sp+152],%i4
	retl
	add %sp,184,%sp
.LLfe4:
	.size	 asm_do_s3,.LLfe4-asm_do_s3
	.align 4
	.global asm_do_s4
	.type	 asm_do_s4,#function
	.proc	016
asm_do_s4:
	!#PROLOGUE# 0
	!# vars= 0, regs= 16/0, args= 0, extra= 84
	add %sp,-176,%sp
	st %i7,[%sp+96]
	sub %sp,-176,%i7	!# set up frame pointer
	st %o7,[%sp+100]
	std %l0,[%sp+104]
	std %l2,[%sp+112]
	st %l4,[%sp+120]
	std %i0,[%sp+128]
	std %i2,[%sp+136]
	std %i4,[%sp+144]
	!#PROLOGUE# 1
	ld [%o0+200],%l2
	sethi %hi(1513943058),%g2
	ld [%o0+204],%l3
	or %g2,%lo(1513943058),%g2
	ld [%o0+208],%l4
	sllx %g2, 32, %g2
	or %i7, %g2, %i7	!##
	lduw [%o0+80], %g2	!+
	ldx [%l2+88], %i3	!#
	ldx [%g2], %i2	!#
	lduw [%o0+88], %g2	!+
	ldx [%l2+104], %i1	!#
	ldx [%g2], %i0	!#
	lduw [%o0+96], %g2	!+
	ldx [%l2+120], %l0	!#
	ldx [%g2], %g3	!#
	lduw [%o0+84], %g2	!+
	ldx [%l2+96], %g1	!#
	ldx [%g2], %g2	!#
	xor %i3, %i2, %o5
	xor %i1, %i0, %i0
	xor %l0, %g3, %l0
	xor %g1, %g2, %g1
.LL30:
	lduw [%o0+92], %o2	!+
	or %o5, %i0, %o7
	or %g1, %i0, %l1
	ldx [%l2+112], %i1	!#
	and %l0, %o7, %o7
	andn %i0, %o5, %o4
	ldx [%o2], %o2	!#
	xor %o5, %o7, %o1
	xor %i0, %o7, %o7
	lduw [%o0+100], %i5	!+
	xor %o1, %l1, %l1
	or %o4, %o1, %o4
	ldx [%l2+128], %i2	!#
	andn %g1, %o7, %o3
	xor %i0, %l0, %g4
	ldx [%i5], %i5	!#
	lduw [%o0+216], %l2	!+
	lduw [%o0+212], %o0	!+
	xor %o4, %o3, %o3
	or %o7, %o1, %o7
	ldx [%l2+88], %i4	!#
	xor %i1, %o2, %i1
	xor %i2, %i5, %i2
	lduw [%o0+80], %i3	!+
	and %g1, %o4, %o4
	andn %g4, %g1, %g4
	lduw [%o0+88], %i5	!+
	xor %i0, %o4, %i0
	xor %l0, %o4, %o4
	ldx [%i3], %i3	!#
	xor %o7, %g4, %o2
	and %i1, %o4, %g3
	ldx [%i5], %i5	!#
	or %i1, %o2, %o2
	xor %l1, %g3, %g3
	lduw [%o0+96], %l1	!+
	xor %o3, %o2, %o2
	xor %i0, %g4, %i0
	lduw [%o0+84], %g4	!+
	or %i2, %o2, %o5
	and %g1, %o4, %o4
	ldx [%l1], %l1	!#
	xor %o4, %o7, %o4
	andn %i1, %i0, %i0
	ldx [%g4], %g4	!#
	xor %g3, %o5, %o5
	xor %o4, %i0, %i0
	ldx [%l2+104], %o4	!#
	and %i2, %o2, %o2
	xor %g3, %i0, %o7
	ldx [%l3+200], %g2	!#
	xnor %o2, %g3, %g3
	andn %g1, %o7, %o3
	ldx [%l3+152], %o1	!#
	xor %g2, %o5, %g2
	stx %g2, [%l4+200]	!#
	xor %o5, %o3, %o3
	xor %o1, %g3, %o1
	stx %o1, [%l4+152]	!#
	andn %o7, %i1, %o7
	xor %o3, %o7, %o7
	ldx [%l3+72], %o1	!#
	or %i2, %o7, %o3
	xor %o2, %o7, %o7
	ldx [%l3], %o2	!#
	xor %i4, %i3, %o5
	xnor %i0, %o3, %o3
	lduw [%o0+204], %l3	!+
	xor %o1, %o3, %o1
	stx %o1, [%l4+72]	!#
	xor %o7, %o3, %o7
	ldx [%l2+120], %l0	!#
	ldx [%l2+96], %g1	!#
	xor %o2, %o7, %o2
	stx %o2, [%l4]	!#
	xor %o4, %i5, %i0
	xor %l0, %l1, %l0
	xor %g1, %g4, %g1
				!# end of included file
	cmp %l3,0
	bne .LL30
	ld [%o0+208],%l4
	srlx %i7, 32, %o0	!##
	!#EPILOGUE#
	ld [%sp+96],%i7
	ld [%sp+100],%o7
	ldd [%sp+104],%l0
	ldd [%sp+112],%l2
	ld [%sp+120],%l4
	ldd [%sp+128],%i0
	ldd [%sp+136],%i2
	ldd [%sp+144],%i4
	retl
	add %sp,176,%sp
.LLfe5:
	.size	 asm_do_s4,.LLfe5-asm_do_s4
	.align 4
	.global asm_do_s5
	.type	 asm_do_s5,#function
	.proc	016
asm_do_s5:
	!#PROLOGUE# 0
	!# vars= 0, regs= 18/0, args= 0, extra= 84
	add %sp,-184,%sp
	st %i7,[%sp+96]
	sub %sp,-184,%i7	!# set up frame pointer
	st %o7,[%sp+100]
	std %l0,[%sp+104]
	std %l2,[%sp+112]
	std %l4,[%sp+120]
	std %l6,[%sp+128]
	std %i0,[%sp+136]
	std %i2,[%sp+144]
	std %i4,[%sp+152]
	!#PROLOGUE# 1
	mov %o0,%l0
	ld [%l0+200],%l3
	sethi %hi(1513943058),%g2
	ld [%l0+204],%l6
	or %g2,%lo(1513943058),%g2
	ld [%l0+208],%l7
	sllx %g2, 32, %g2
	or %i7, %g2, %i7	!##
	lduw [%l0+112], %g2	!+
	ldx [%l3+136], %i3	!#
	ldx [%g2], %i2	!#
	lduw [%l0+116], %g2	!+
	ldx [%l3+144], %i1	!#
	ldx [%g2], %i0	!#
	lduw [%l0+104], %g2	!+
	ldx [%l3+120], %o3	!#
	ldx [%g2], %g3	!#
	lduw [%l0+124], %g2	!+
	ldx [%l3+160], %o4	!#
	ldx [%g2], %g2	!#
	xor %i3, %i2, %l4
	xor %i1, %i0, %o5
	xor %o3, %g3, %o3
	xor %o4, %g2, %o4
.LL35:
	lduw [%l0+120], %g1	!+
	andn %l4, %o5, %o1
	andn %o3, %l4, %g3
	ldx [%l3+152], %i2	!#
	xor %o1, %o3, %i0
	xor %o5, %o3, %l5
	ldx [%g1], %g1	!#
	or %o4, %g3, %o0
	or %l5, %o1, %o1
	lduw [%l0+108], %g4	!+
	xor %i0, %o0, %l2
	and %l4, %o1, %l1
	ldx [%l3+128], %i5	!#
	xor %l1, %o5, %l1
	xor %o5, %g3, %i3
	ldx [%g4], %g4	!#
	andn %l1, %g3, %g3
	or %o4, %i3, %i3
	xor %i2, %g1, %i2
	xor %i5, %g4, %i5
	andn %o1, %o4, %i1
	xor %g3, %i3, %g3
	xor %l4, %i1, %g4
	or %i2, %g3, %i4
	lduw [%l0+216], %l3	!+
	or %i2, %g4, %g1
	xor %l1, %i4, %i4
	lduw [%l0+212], %l0	!+
	xor %l2, %g1, %g1
	andn %i4, %i5, %g2
	ldx [%l6+16], %o4	!#
	xor %g1, %g2, %g2
	and %o5, %o0, %o0
	lduw [%l0+112], %g1	!+
	xor %o4, %g2, %o4
	stx %o4, [%l7+16]	!#
	xor %o3, %g4, %o2
	xor %o0, %g3, %o0
	and %i0, %o2, %i0
	andn %i2, %i0, %i0
	or %o5, %o2, %o4
	ldx [%g1], %g1	!#
	xor %o0, %i0, %i0
	andn %o4, %i5, %o3
	ldx [%l6+104], %o7	!#
	xor %i0, %o3, %o3
	andn %i1, %o5, %i1
	lduw [%l0+116], %o0	!+
	xor %o7, %o3, %o7
	stx %o7, [%l7+104]	!#
	xor %i1, %l4, %i1
	and %g3, %l2, %g3
	and %i2, %i1, %l2
	ldx [%o0], %o0	!#
	andn %o1, %g3, %o1
	xor %g4, %g3, %g3
	lduw [%l0+104], %g4	!+
	xor %o1, %l2, %l2
	or %l1, %i3, %i3
	lduw [%l0+124], %l1	!+
	or %i2, %g3, %o7
	andn %i4, %o1, %o1
	ldx [%g4], %g4	!#
	xor %i3, %o7, %o7
	or %i0, %o1, %i0
	ldx [%l1], %l1	!#
	or %i5, %o7, %o7
	xor %o1, %o2, %o1
	ldx [%l6+192], %o2	!#
	xnor %l2, %o7, %o7
	xor %i0, %l5, %i0
	ldx [%l3+136], %l2	!#
	xor %o2, %o7, %o2
	stx %o2, [%l7+192]	!#
	xor %g2, %g3, %g2
	and %l5, %g3, %g3
	andn %o4, %g2, %g2
	ldx [%l3+144], %l5	!#
	xor %g3, %i1, %g3
	and %i2, %g2, %g2
	ldx [%l3+120], %o3	!#
	andn %i2, %i0, %i0
	xor %g3, %g2, %g2
	ldx [%l3+160], %o4	!#
	xor %o1, %i0, %i0
	or %i5, %g2, %g2
	ldx [%l6+56], %o1	!#
	xor %l2, %g1, %l4
	xor %i0, %g2, %g2
	lduw [%l0+204], %l6	!+
	xor %o1, %g2, %o1
	stx %o1, [%l7+56]	!#
	xor %l5, %o0, %o5
	xor %o3, %g4, %o3
	xor %o4, %l1, %o4
				!# end of included file
	cmp %l6,0
	bne .LL35
	ld [%l0+208],%l7
	srlx %i7, 32, %o0	!##
	!#EPILOGUE#
	ld [%sp+96],%i7
	ld [%sp+100],%o7
	ldd [%sp+104],%l0
	ldd [%sp+112],%l2
	ldd [%sp+120],%l4
	ldd [%sp+128],%l6
	ldd [%sp+136],%i0
	ldd [%sp+144],%i2
	ldd [%sp+152],%i4
	retl
	add %sp,184,%sp
.LLfe6:
	.size	 asm_do_s5,.LLfe6-asm_do_s5
	.align 4
	.global asm_do_s6
	.type	 asm_do_s6,#function
	.proc	016
asm_do_s6:
	!#PROLOGUE# 0
	!# vars= 0, regs= 18/0, args= 0, extra= 84
	add %sp,-184,%sp
	st %i7,[%sp+96]
	sub %sp,-184,%i7	!# set up frame pointer
	st %o7,[%sp+100]
	std %l0,[%sp+104]
	std %l2,[%sp+112]
	std %l4,[%sp+120]
	st %l6,[%sp+128]
	std %i0,[%sp+136]
	std %i2,[%sp+144]
	std %i4,[%sp+152]
	!#PROLOGUE# 1
	ld [%o0+200],%l2
	sethi %hi(1513943058),%g2
	ld [%o0+204],%l5
	or %g2,%lo(1513943058),%g2
	ld [%o0+208],%l6
	sllx %g2, 32, %g2
	or %i7, %g2, %i7	!##
	lduw [%o0+144], %g2	!+
	ldx [%l2+184], %i3	!#
	ldx [%g2], %i2	!#
	lduw [%o0+128], %g2	!+
	ldx [%l2+152], %i1	!#
	ldx [%g2], %i0	!#
	lduw [%o0+148], %g2	!+
	ldx [%l2+192], %g4	!#
	ldx [%g2], %g3	!#
	lduw [%o0+140], %g2	!+
	ldx [%l2+176], %l3	!#
	ldx [%g2], %g2	!#
	xor %i3, %i2, %l1
	xor %i1, %i0, %o4
	xor %g4, %g3, %g4
	xor %l3, %g2, %l3
.LL40:
	lduw [%o0+132], %g1	!+
	xor %l1, %o4, %o1
	and %o4, %g4, %g3
	ldx [%l2+160], %i1	!#
	andn %g3, %l1, %i4
	xor %g4, %g3, %i0
	ldx [%g1], %g1	!#
	xor %o1, %g4, %l4
	or %i4, %i0, %o3
	lduw [%o0+136], %i5	!+
	andn %l3, %i4, %i3
	andn %o3, %l3, %o3
	ldx [%l2+168], %i2	!#
	xor %l4, %i3, %l0
	xor %i0, %o3, %o3
	ldx [%i5], %i5	!#
	or %g4, %l0, %o5
	or %i4, %o3, %i4
	xor %i1, %g1, %i1
	xor %i2, %i5, %i2
	andn %o5, %l1, %o5
	andn %i1, %i4, %o7
	lduw [%o0+216], %l2	!+
	and %i1, %o3, %g1
	xor %o5, %o7, %o7
	lduw [%o0+212], %o0	!+
	xor %l0, %g1, %g1
	andn %o7, %i2, %o7
	ldx [%l5+24], %o5	!#
	xnor %g1, %o7, %o7
	or %l1, %g4, %g1
	lduw [%o0+144], %i5	!+
	xor %o5, %o7, %o5
	stx %o5, [%l6+24]	!#
	andn %o7, %o1, %o7
	xor %o7, %i4, %o7
	andn %g1, %o1, %o1
	andn %g4, %o7, %o5
	xor %g3, %l0, %g3
	ldx [%i5], %i5	!#
	xor %o5, %l0, %o5
	andn %g3, %o3, %g3
	lduw [%o0+128], %i4	!+
	andn %i1, %o5, %o3
	andn %o7, %l1, %o2
	lduw [%o0+148], %l0	!+
	andn %i1, %o3, %g2
	xor %o7, %o3, %o3
	ldx [%i4], %i4	!#
	xor %o1, %g2, %g2
	xor %g4, %o3, %o1
	ldx [%l0], %l0	!#
	andn %l1, %o1, %o1
	andn %i2, %g2, %g2
	ldx [%l5+144], %o7	!#
	xnor %o3, %g2, %g2
	andn %i1, %o1, %o4
	lduw [%o0+140], %o3	!+
	xor %o7, %g2, %o7
	stx %o7, [%l6+144]	!#
	xor %g3, %o4, %o4
	or %i2, %o2, %o2
	and %l1, %i0, %i0
	ldx [%l5+80], %o7	!#
	xnor %o4, %o2, %o2
	andn %l3, %i0, %i0
	ldx [%o3], %o3	!#
	xor %o7, %o2, %o7
	stx %o7, [%l6+80]	!#
	or %o1, %l4, %o2
	or %i1, %i0, %i0
	and %g1, %g3, %g3
	ldx [%l2+184], %g1	!#
	xor %o2, %i0, %i0
	xor %g3, %l4, %g3
	ldx [%l2+152], %l4	!#
	or %o5, %o1, %o1
	and %i1, %g3, %g3
	ldx [%l2+192], %g4	!#
	xor %o1, %i3, %o1
	xor %o1, %g3, %g3
	ldx [%l2+176], %l3	!#
	andn %i2, %g3, %g3
	ldx [%l5+224], %o1	!#
	xor %g1, %i5, %l1
	xnor %i0, %g3, %g3
	lduw [%o0+204], %l5	!+
	xor %o1, %g3, %o1
	stx %o1, [%l6+224]	!#
	xor %l4, %i4, %o4
	xor %g4, %l0, %g4
	xor %l3, %o3, %l3
				!# end of included file
	cmp %l5,0
	bne .LL40
	ld [%o0+208],%l6
	srlx %i7, 32, %o0	!##
	!#EPILOGUE#
	ld [%sp+96],%i7
	ld [%sp+100],%o7
	ldd [%sp+104],%l0
	ldd [%sp+112],%l2
	ldd [%sp+120],%l4
	ld [%sp+128],%l6
	ldd [%sp+136],%i0
	ldd [%sp+144],%i2
	ldd [%sp+152],%i4
	retl
	add %sp,184,%sp
.LLfe7:
	.size	 asm_do_s6,.LLfe7-asm_do_s6
	.align 4
	.global asm_do_s6f
	.type	 asm_do_s6f,#function
	.proc	016
asm_do_s6f:
	!#PROLOGUE# 0
	!# vars= 0, regs= 6/0, args= 0, extra= 84
	add %sp,-136,%sp
	st %i7,[%sp+96]
	sub %sp,-136,%i7	!# set up frame pointer
	st %o7,[%sp+100]
	std %i0,[%sp+104]
	st %i2,[%sp+112]
	!#PROLOGUE# 1
	ld [%o0+200],%i0
	sethi %hi(1513943058),%g2
	ld [%o0+204],%i1
	or %g2,%lo(1513943058),%g2
	ld [%o0+208],%i2
	sllx %g2, 32, %g2
	or %i7, %g2, %i7	!##
	ldd [%i0+184], %f0	!#
	ldd [%i0+152], %f2	!#
	ldd [%i0+192], %f30	!#
	ldd [%i0+176], %f20	!#
	lduw [%o0+144], %g2	!#
	lduw [%o0+128], %g3	!#
	ldd [%g2], %f26	!#
	ldd [%g3], %f28	!#
	lduw [%o0+148], %g3	!#
	fxor %f0, %f26, %f0
	lduw [%o0+140], %g2	!#
	fxor %f2, %f28, %f2
	ldd [%g3], %f24	!#
	ldd [%g2], %f16	!#
	ldd [%i0+168], %f6	!#
	fxor %f30, %f24, %f30
	lduw [%o0+132], %g2	!#
	lduw [%o0+136], %g3	!#
	ldd [%g2], %f12	!#
	ldd [%g3], %f16	!#
	ldd [%i0+160], %f22	!#
	fxor %f20, %f16, %f20
	fxor %f0, %f2, %f4
	fand %f2, %f30, %f2
	fandnot2 %f2, %f0, %f18
	fxor %f30, %f2, %f24
	fxor %f4, %f30, %f8
	for %f18, %f24, %f14
	fandnot2 %f20, %f18, %f10
	ldd [%g2], %f12	!#
	fandnot2 %f14, %f20, %f14
	ldd [%g3], %f16	!#
	fxor %f8, %f10, %f26
	fxor %f24, %f14, %f14
	ldd [%i0+160], %f22	!#
	for %f30, %f26, %f28
	for %f18, %f14, %f18
	fxor %f22, %f12, %f22
	fxor %f6, %f16, %f6
	fandnot2 %f28, %f0, %f28
	fandnot2 %f22, %f18, %f12
	ldd [%i1+24], %f28	!#
	fand %f22, %f14, %f16
	fxor %f26, %f16, %f16
	fxor %f28, %f12, %f12
	fandnot2 %f12, %f6, %f12
	fxnor %f16, %f12, %f12
	fandnot2 %f12, %f4, %f16
	fxor %f16, %f18, %f16
	fxor %f28, %f12, %f28
	fxor %f2, %f26, %f2
	fandnot2 %f2, %f14, %f2
	fandnot2 %f30, %f16, %f18
	fxor %f18, %f26, %f18
	fand %f0, %f24, %f24
	ldd [%i1+144], %f26	!#
	fandnot2 %f20, %f24, %f24
	for %f0, %f30, %f20
	fandnot2 %f20, %f4, %f4
	fand %f20, %f2, %f20
	fandnot2 %f22, %f18, %f14
	fandnot2 %f22, %f14, %f12
	fxor %f4, %f12, %f12
	fxor %f16, %f14, %f14
	fandnot2 %f16, %f0, %f16
	fandnot2 %f6, %f12, %f12
	fxnor %f14, %f12, %f12
	fxor %f26, %f12, %f26
	fxor %f30, %f14, %f14
	fandnot2 %f0, %f14, %f14
	for %f18, %f14, %f18
	for %f14, %f8, %f0
	ldd [%i1+80], %f30	!#
	fandnot2 %f22, %f14, %f14
	fxor %f2, %f14, %f14
	for %f6, %f16, %f16
	fxnor %f14, %f16, %f16
	fxor %f30, %f16, %f30
	for %f22, %f24, %f24
	fxor %f0, %f24, %f24
	fxor %f18, %f10, %f18
	fxor %f20, %f8, %f20
	fand %f22, %f20, %f20
	fxor %f18, %f20, %f20
	ldd [%i1+224], %f18	!#
	fandnot2 %f6, %f20, %f20
	fxnor %f24, %f20, %f20
	fxor %f18, %f20, %f18
	std %f28, [%i2+24]	!#
	std %f26, [%i2+144]	!#
	std %f30, [%i2+80]	!#
	std %f18, [%i2+224]	!#
	srlx %i7, 32, %o0	!##
	!#EPILOGUE#
	ld [%sp+96],%i7
	ld [%sp+100],%o7
	ldd [%sp+104],%i0
	ld [%sp+112],%i2
	retl
	add %sp,136,%sp
.LLfe8:
	.size	 asm_do_s6f,.LLfe8-asm_do_s6f
	.align 4
	.global asm_do_s7f
	.type	 asm_do_s7f,#function
	.proc	016
asm_do_s7f:
	!#PROLOGUE# 0
	!# vars= 0, regs= 6/0, args= 0, extra= 84
	add %sp,-136,%sp
	st %i7,[%sp+96]
	sub %sp,-136,%i7	!# set up frame pointer
	st %o7,[%sp+100]
	std %i0,[%sp+104]
	st %i2,[%sp+112]
	!#PROLOGUE# 1
	ld [%o0+200],%i0
	sethi %hi(1513943058),%g2
	ld [%o0+204],%i1
	or %g2,%lo(1513943058),%g2
	ld [%o0+208],%i2
	sllx %g2, 32, %g2
	or %i7, %g2, %i7	!##
	ldd [%i0+192], %f8	!#
	ldd [%i0+208], %f4	!#
	lduw [%o0+156], %g2	!#
	lduw [%o0+164], %g3	!#
	ldd [%g2], %f24	!#
	ldd [%g3], %f14	!#
	ldd [%i0+216], %f10	!#
	lduw [%o0+168], %g3	!#
	fxor %f8, %f24, %f8
	lduw [%o0+160], %g2	!#
	fxor %f4, %f14, %f4
	ldd [%g3], %f12	!#
	ldd [%g2], %f22	!#
	lduw [%o0+172], %g3	!#
	fxor %f10, %f12, %f10
	lduw [%o0+152], %g2	!#
	ldd [%g3], %f30	!#
	ldd [%g2], %f0	!#
	ldd [%i0+200], %f2	!#
	ldd [%i0+224], %f26	!#
	fxor %f2, %f22, %f2
	fand %f8, %f4, %f28
	fxor %f28, %f10, %f28
	fand %f4, %f28, %f14
	fxnor %f14, %f8, %f24
	for %f8, %f4, %f20
	for %f20, %f10, %f20
	fandnot2 %f10, %f8, %f16
	for %f2, %f16, %f16
	fxor %f20, %f16, %f16
	fxor %f26, %f30, %f26
	fand %f2, %f24, %f18
	fxnor %f4, %f24, %f24
	fandnot2 %f4, %f2, %f4
	fxnor %f2, %f18, %f30
	fxor %f28, %f18, %f18
	fxor %f14, %f18, %f6
	ldd [%i0+184], %f20	!#
	for %f26, %f6, %f12
	fand %f26, %f30, %f22
	fxor %f18, %f22, %f22
	for %f10, %f4, %f10
	fandnot2 %f8, %f4, %f4
	fand %f26, %f4, %f4
	fxor %f16, %f12, %f12
	fxor %f8, %f6, %f6
	ldd [%i1+248], %f18	!#
	fxor %f20, %f0, %f20
	ldd [%i1+88], %f0	!#
	fand %f20, %f12, %f12
	ldd [%i1+168], %f28	!#
	fxor %f22, %f12, %f12
	fxnor %f22, %f4, %f22
	fand %f2, %f4, %f4
	fxor %f18, %f12, %f18
	for %f8, %f24, %f16
	fxnor %f16, %f12, %f16
	for %f26, %f16, %f16
	fxor %f2, %f14, %f12
	fand %f12, %f8, %f12
	fandnot2 %f26, %f12, %f12
	for %f2, %f14, %f14
	fxor %f24, %f14, %f14
	fandnot2 %f14, %f2, %f2
	fandnot2 %f24, %f6, %f6
	fand %f26, %f6, %f6
	ldd [%i1+48], %f26	!#
	fxor %f14, %f12, %f12
	fxnor %f30, %f12, %f30
	for %f20, %f12, %f14
	fxor %f22, %f14, %f14
	fxor %f0, %f14, %f0
	fxor %f10, %f30, %f10
	fxor %f30, %f16, %f16
	fxor %f4, %f6, %f4
	for %f2, %f12, %f2
	fornot2 %f20, %f2, %f2
	fxor %f16, %f2, %f2
	fxor %f28, %f2, %f28
	fxor %f10, %f6, %f6
	std %f18, [%i2+248]	!#
	for %f20, %f4, %f4
	std %f0, [%i2+88]	!#
	fxor %f6, %f4, %f4
	std %f28, [%i2+168]	!#
	fxor %f26, %f4, %f26
	std %f26, [%i2+48]	!#
				!# end of included file
	srlx %i7, 32, %o0	!##
	!#EPILOGUE#
	ld [%sp+96],%i7
	ld [%sp+100],%o7
	ldd [%sp+104],%i0
	ld [%sp+112],%i2
	retl
	add %sp,136,%sp
.LLfe9:
	.size	 asm_do_s7f,.LLfe9-asm_do_s7f
	.align 4
	.global asm_do_s7
	.type	 asm_do_s7,#function
	.proc	016
asm_do_s7:
	!#PROLOGUE# 0
	!# vars= 0, regs= 18/0, args= 0, extra= 84
	add %sp,-184,%sp
	st %i7,[%sp+96]
	sub %sp,-184,%i7	!# set up frame pointer
	st %o7,[%sp+100]
	std %l0,[%sp+104]
	std %l2,[%sp+112]
	std %l4,[%sp+120]
	st %l6,[%sp+128]
	std %i0,[%sp+136]
	std %i2,[%sp+144]
	std %i4,[%sp+152]
	!#PROLOGUE# 1
	mov %o0,%l1
	ld [%l1+200],%l4
	sethi %hi(1513943058),%g2
	ld [%l1+204],%l5
	or %g2,%lo(1513943058),%g2
	ld [%l1+208],%l6
	sllx %g2, 32, %g2
	or %i7, %g2, %i7	!##
	lduw [%l1+156], %g2	!+
	ldx [%l4+192], %i3	!#
	ldx [%g2], %i2	!#
	lduw [%l1+164], %g2	!+
	ldx [%l4+208], %i1	!#
	ldx [%g2], %i0	!#
	lduw [%l1+168], %g2	!+
	ldx [%l4+216], %l3	!#
	ldx [%g2], %g3	!#
	lduw [%l1+160], %g2	!+
	ldx [%l4+200], %l0	!#
	ldx [%g2], %g2	!#
	xor %i3, %i2, %l2
	xor %i1, %i0, %o0
	xor %l3, %g3, %l3
	xor %l0, %g2, %l0
.LL47:
	lduw [%l1+172], %o1	!+
	and %l2, %o0, %g1
	or %l2, %o0, %o3
	ldx [%l4+224], %i0	!#
	xor %g1, %l3, %g1
	or %o3, %l3, %o3
	ldx [%o1], %o1	!#
	and %o0, %g1, %o2
	andn %l3, %l2, %g4
	lduw [%l1+152], %o4	!+
	xnor %o2, %l2, %i1
	or %l0, %g4, %g4
	ldx [%l4+184], %i2	!#
	and %l0, %i1, %o7
	xor %o3, %g4, %g4
	ldx [%o4], %o4	!#
	xnor %l0, %o7, %i5
	xor %g1, %o7, %o7
	xor %i0, %o1, %i0
	xor %i2, %o4, %i2
	lduw [%l1+216], %l4	!+
	and %i0, %i5, %g2
	xor %o2, %o7, %g1
	lduw [%l1+212], %l1	!+
	xor %o7, %g2, %g2
	or %i0, %g1, %o4
	xor %g4, %o4, %o4
	andn %o0, %l0, %g3
	lduw [%l1+156], %i4	!+
	and %i2, %o4, %o4
	andn %l2, %g3, %o3
	ldx [%l5+248], %g4	!#
	xor %g2, %o4, %o4
	and %i0, %o3, %o3
	ldx [%i4], %i4	!#
	xor %g4, %o4, %g4
	stx %g4, [%l6+248]	!#
	xnor %o0, %i1, %i1
	or %l0, %o2, %o7
	xor %l0, %o2, %o2
	lduw [%l1+164], %i3	!+
	xor %i1, %o7, %o7
	and %o2, %l2, %o2
	lduw [%l1+168], %g4	!+
	xnor %g2, %o3, %g2
	andn %i0, %o2, %o2
	ldx [%i3], %i3	!#
	xor %o7, %o2, %o0
	or %l2, %i1, %o5
	ldx [%g4], %g4	!#
	or %i2, %o0, %o2
	xnor %i5, %o0, %i5
	ldx [%l5+88], %o1	!#
	xor %g2, %o2, %o2
	xnor %o5, %o4, %o5
	lduw [%l1+160], %o4	!+
	xor %o1, %o2, %o1
	stx %o1, [%l6+88]	!#
	or %i0, %o5, %o5
	xor %i5, %o5, %o5
	andn %o7, %l0, %o7
	ldx [%o4], %o4	!#
	or %o7, %o0, %o7
	or %l3, %g3, %g3
	ldx [%l4+192], %o0	!#
	orn %i2, %o7, %o7
	xor %g3, %i5, %g3
	ldx [%l5+168], %i5	!#
	xor %o5, %o7, %o7
	xor %l2, %g1, %g1
	ldx [%l4+208], %o5	!#
	xor %i5, %o7, %i5
	stx %i5, [%l6+168]	!#
	andn %i1, %g1, %g1
	and %i0, %g1, %g1
	and %l0, %o3, %o3
	ldx [%l4+216], %l3	!#
	xor %o0, %i4, %l2
	xor %o3, %g1, %o3
	ldx [%l4+200], %l0	!#
	xor %g3, %g1, %g3
	or %i2, %o3, %o3
	ldx [%l5+48], %o1	!#
	xor %g3, %o3, %o3
	lduw [%l1+204], %l5	!+
	xor %o1, %o3, %o1
	stx %o1, [%l6+48]	!#
	xor %o5, %i3, %o0
	xor %l3, %g4, %l3
	xor %l0, %o4, %l0
				!# end of included file
	cmp %l5,0
	bne .LL47
	ld [%l1+208],%l6
	srlx %i7, 32, %o0	!##
	!#EPILOGUE#
	ld [%sp+96],%i7
	ld [%sp+100],%o7
	ldd [%sp+104],%l0
	ldd [%sp+112],%l2
	ldd [%sp+120],%l4
	ld [%sp+128],%l6
	ldd [%sp+136],%i0
	ldd [%sp+144],%i2
	ldd [%sp+152],%i4
	retl
	add %sp,184,%sp
.LLfe10:
	.size	 asm_do_s7,.LLfe10-asm_do_s7
	.align 4
	.global asm_do_s8
	.type	 asm_do_s8,#function
	.proc	016
asm_do_s8:
	!#PROLOGUE# 0
	!# vars= 0, regs= 18/0, args= 0, extra= 84
	add %sp,-184,%sp
	st %i7,[%sp+96]
	sub %sp,-184,%i7	!# set up frame pointer
	st %o7,[%sp+100]
	std %l0,[%sp+104]
	std %l2,[%sp+112]
	std %l4,[%sp+120]
	std %l6,[%sp+128]
	std %i0,[%sp+136]
	std %i2,[%sp+144]
	std %i4,[%sp+152]
	!#PROLOGUE# 1
	ld [%o0+200],%l3
	sethi %hi(1513943058),%g2
	ld [%o0+204],%l6
	or %g2,%lo(1513943058),%g2
	ld [%o0+208],%l7
	sllx %g2, 32, %g2
	or %i7, %g2, %i7	!##
	lduw [%o0+184], %g2	!+
	ldx [%l3+232], %i3	!#
	ldx [%g2], %i2	!#
	lduw [%o0+176], %g2	!+
	ldx [%l3+216], %i1	!#
	ldx [%g2], %i0	!#
	lduw [%o0+188], %g2	!+
	ldx [%l3+240], %l1	!#
	ldx [%g2], %g3	!#
	lduw [%o0+192], %g2	!+
	ldx [%l3+248], %g4	!#
	ldx [%g2], %g2	!#
	xor %i3, %i2, %l5
	xor %i1, %i0, %o4
	xor %l1, %g3, %l1
	xor %g4, %g2, %g4
.LL52:
	lduw [%o0+180], %o5	!+
	xor %l5, %o4, %l0
	andn %o4, %l5, %o2
	ldx [%l3+224], %g3	!#
	xor %o2, %l1, %i4
	and %g4, %o2, %g2
	ldx [%o5], %o5	!#
	or %g4, %i4, %o1
	andn %l0, %g4, %i5
	lduw [%o0+196], %g1	!+
	xor %l0, %o1, %o1
	xor %g3, %o5, %g3
	ldx [%l3], %i2	!#
	andn %o1, %o4, %i0
	or %o1, %g4, %o5
	ldx [%g1], %g1	!#
	or %i0, %l1, %o3
	xor %i0, %l5, %l4
	lduw [%o0+216], %l3	!+
	or %g3, %i0, %i0
	xor %i2, %g1, %i2
	lduw [%o0+212], %o0	!+
	xor %o3, %l0, %o3
	andn %l4, %g4, %l2
	xor %l1, %l2, %l2
	xor %o3, %g4, %o3
	lduw [%o0+184], %i3	!+
	andn %g3, %l2, %g1
	andn %i4, %o3, %o7
	lduw [%o0+176], %l0	!+
	xor %o1, %g1, %g1
	xor %o7, %l4, %i1
	ldx [%i3], %i3	!#
	andn %g3, %i1, %o7
	xor %o5, %i4, %i4
	ldx [%l0], %l0	!#
	xor %o3, %o7, %o7
	and %o4, %o5, %o5
	lduw [%o0+188], %l4	!+
	or %i2, %o7, %o7
	andn %g1, %l1, %o4
	ldx [%l6+32], %o1	!#
	xnor %g1, %o7, %o7
	xor %g2, %o4, %g2
	ldx [%l4], %l4	!#
	xor %o1, %o7, %o1
	stx %o1, [%l7+32]	!#
	and %g3, %g2, %g2
	andn %g3, %o4, %o4
	xor %o5, %g2, %g2
	lduw [%o0+192], %o5	!+
	xor %i4, %o4, %o4
	andn %g2, %i2, %g2
	ldx [%l6+112], %o1	!#
	xor %o4, %g2, %g2
	andn %l5, %i1, %i1
	ldx [%o5], %o5	!#
	xor %o1, %g2, %o1
	stx %o1, [%l7+112]	!#
	or %l2, %i1, %i1
	andn %o2, %o3, %o3
	or %i4, %g2, %g2
	ldx [%l3+232], %l2	!#
	xor %l5, %g4, %o2
	andn %g3, %g2, %g2
	ldx [%l3+216], %i4	!#
	xor %o3, %g2, %g2
	xor %o2, %o3, %o2
	ldx [%l3+248], %g4	!#
	or %i2, %g2, %g2
	andn %o2, %g3, %o2
	ldx [%l6+208], %o3	!#
	xor %i1, %i0, %i0
	or %i5, %l1, %i5
	ldx [%l3+240], %l1	!#
	xnor %i0, %g2, %g2
	xor %i5, %o2, %o2
	ldx [%l6+160], %i5	!#
	xor %o3, %g2, %o3
	stx %o3, [%l7+208]	!#
	and %i2, %o2, %o2
	xor %l2, %i3, %l5
	xnor %g1, %o2, %o2
	lduw [%o0+204], %l6	!+
	xor %i5, %o2, %i5
	stx %i5, [%l7+160]	!#
	xor %i4, %l0, %o4
	xor %l1, %l4, %l1
	xor %g4, %o5, %g4
				!# end of included file
	cmp %l6,0
	bne .LL52
	ld [%o0+208],%l7
	srlx %i7, 32, %o0	!##
	!#EPILOGUE#
	ld [%sp+96],%i7
	ld [%sp+100],%o7
	ldd [%sp+104],%l0
	ldd [%sp+112],%l2
	ldd [%sp+120],%l4
	ldd [%sp+128],%l6
	ldd [%sp+136],%i0
	ldd [%sp+144],%i2
	ldd [%sp+152],%i4
	retl
	add %sp,184,%sp
.LLfe11:
	.size	 asm_do_s8,.LLfe11-asm_do_s8
	.align 4
	.global asm_do_all
	.type	 asm_do_all,#function
	.proc	016
asm_do_all:
	!#PROLOGUE# 0
	!# vars= 0, regs= 18/0, args= 0, extra= 84
	add %sp,-184,%sp
	st %i7,[%sp+96]
	sub %sp,-184,%i7	!# set up frame pointer
	st %o7,[%sp+100]
	std %l0,[%sp+104]
	std %l2,[%sp+112]
	std %l4,[%sp+120]
	std %l6,[%sp+128]
	std %i0,[%sp+136]
	std %i2,[%sp+144]
	std %i4,[%sp+152]
	!#PROLOGUE# 1
	mov %o0,%l2
	ld [%l2+200],%l3
	sethi %hi(1513943058),%g2
	ld [%l2+204],%l6
	or %g2,%lo(1513943058),%g2
	ld [%l2+208],%l7
	sllx %g2, 32, %g2
	or %i7, %g2, %i7	!##
	lduw [%l2+32], %g2	!+
	ldx [%l3+24], %i3	!#
	ldx [%g2], %i2	!#
	lduw [%l2+52], %g2	!+
	ldx [%l3+64], %i1	!#
	ldx [%g2], %i0	!#
	lduw [%l2+48], %g2	!+
	ldx [%l3+56], %o5	!#
	ldx [%g2], %g3	!#
	lduw [%l2+36], %g2	!+
	ldx [%l3+32], %o0	!#
	ldx [%g2], %g2	!#
	xor %i3, %i2, %o4
	xor %i1, %i0, %i4
	xor %o5, %g3, %o5
	xor %o0, %g2, %o0
.LL57:
	lduw [%l2+40], %g1	!+
	xor %o4, %i4, %l1
	and %i4, %o5, %i1
	ldx [%l3+40], %l0	!#
	xor %l1, %o5, %g4
	andn %o4, %i1, %o3
	ldx [%g1], %g1	!#
	andn %o0, %o3, %o1
	andn %o5, %o3, %i0
	lduw [%l2+44], %o2	!+
	xor %g4, %o1, %g3
	or %i1, %o1, %o1
	ldx [%l3+48], %i5	!#
	andn %o1, %l1, %o1
	or %i0, %o0, %l4
	ldx [%o2], %o2	!#
	xor %l0, %g1, %l0
	andn %i4, %o3, %g1
	ldd [%l3+184], %f6	!#
	or %l0, %o1, %o7
	xor %i5, %o2, %i5
	ldd [%l3+152], %f18	!#
	xor %g3, %o7, %o7
	and %i5, %l4, %i3
	ldx [%l6+96], %o2	!#
	xnor %o7, %i3, %i3
	xor %g3, %i0, %i0
	lduw [%l2+60], %l1	!+
	xor %o2, %i3, %o2
	stx %o2, [%l7+96]	!#
	xor %o3, %i3, %i3
	andn %i3, %o0, %i3
	and %o0, %i0, %i0
	ldx [%l1], %l1	!#
	xor %g4, %i3, %i3
	xor %g1, %i0, %o3
	ldd [%l3+192], %f24	!#
	and %l0, %o3, %o3
	or %g3, %o4, %g3
	lduw [%l2+64], %o2	!+
	xor %i3, %o3, %o4
	xor %o5, %o0, %i3
	lduw [%l2+76], %o3	!+
	andn %i3, %o1, %o1
	xor %g3, %o0, %g3
	ldx [%o2], %o2	!#
	andn %l0, %g3, %i2
	or %g1, %o1, %g1
	ldx [%o3], %o3	!#
	xor %o1, %i2, %i2
	or %g3, %i0, %g3
	lduw [%l2+72], %o1	!+
	or %i5, %i2, %g2
	xor %g1, %o7, %g1
	ldx [%l6+8], %o7	!#
	xor %o4, %g2, %g2
	and %i3, %g3, %i3
	ldx [%o1], %o1	!#
	xor %o7, %g2, %o7
	stx %o7, [%l7+8]	!#
	andn %l4, %i3, %i3
	and %l0, %g3, %g3
	or %i5, %i3, %i3
	ldx [%l3+64], %l4	!#
	xor %g1, %g3, %g3
	xor %o0, %g4, %g2
	ldx [%l6+136], %o7	!#
	xnor %g3, %i3, %i3
	andn %g2, %g1, %g4
	ldx [%l3+72], %g1	!#
	xor %o7, %i3, %o7
	stx %o7, [%l7+136]	!#
	xor %g4, %i2, %g4
	or %i1, %i0, %i0
	and %l0, %i1, %i1
	ldx [%l3+96], %i4	!#
	andn %l0, %g4, %g4
	xor %i0, %i1, %i1
	ldx [%l3+88], %o5	!#
	xor %g2, %g4, %g4
	andn %i5, %i1, %i1
	ldx [%l6+216], %g2	!#
	xor %l4, %l1, %o0
	xnor %g4, %i1, %i1
	ldd [%l3+176], %f20	!#
	xor %g2, %i1, %g2
	stx %g2, [%l7+216]	!#
	xor %g1, %o2, %l0
	xor %i4, %o3, %i4
	xor %o5, %o1, %o5
				!# end of included file
	lduw [%l2+144], %g2	!#
	lduw [%l2+128], %g3	!#
	ldd [%g2], %f28	!#
	ldd [%g3], %f14	!#
	lduw [%l2+148], %g3	!#
	fxor %f6, %f28, %f6
	lduw [%l2+140], %g2	!#
	fxor %f18, %f14, %f18
	ldd [%g3], %f8	!#
	ldd [%g2], %f12	!#
	ldd [%l3+168], %f0	!#
	fxor %f24, %f8, %f24
	lduw [%l2+56], %o3	!+
	fxor %f20, %f12, %f20
	xor %o0, %l0, %g3
	and %l0, %i4, %g2
	ldx [%l3+56], %o4	!#
	fxor %f6, %f18, %f26
	xor %g3, %i4, %g3
	and %l0, %o5, %l1
	ldx [%o3], %o3	!#
	fand %f18, %f24, %f18
	and %o0, %g3, %l5
	xor %l1, %g3, %l1
	lduw [%l2+68], %i1	!+
	fandnot2 %f18, %f6, %f10
	or %o5, %l5, %o7
	andn %i4, %l5, %i0
	ldx [%l3+80], %i5	!#
	fxor %f24, %f18, %f8
	xor %g3, %o7, %g1
	xor %i0, %o5, %o2
	ldx [%i1], %i1	!#
	fxor %f26, %f24, %f16
	xor %o4, %o3, %o4
	xor %l0, %l5, %o3
	lduw [%l2+132], %o1	!#
	for %f10, %f8, %f2
	and %o4, %o2, %i2
	xor %i5, %i1, %i5
	lduw [%l2+136], %g4	!#
	fandnot2 %f20, %f10, %f22
	andn %o3, %o5, %i3
	xor %o5, %i2, %i2
	ldd [%o1], %f4	!#
	fandnot2 %f2, %f20, %f2
	or %o4, %i3, %i1
	or %i5, %i2, %i2
	ldd [%g4], %f12	!#
	fxor %f16, %f22, %f28
	andn %g3, %i3, %g3
	xor %g1, %i1, %i1
	ldx [%l6+40], %l4	!#
	fxor %f8, %f2, %f2
	xor %i1, %i2, %i2
	xor %g3, %g2, %i1
	ldd [%l3+160], %f30	!#
	for %f24, %f28, %f14
	or %o2, %g3, %o2
	and %i4, %i2, %o1
	lduw [%l2+80], %g4	!+
	xor %l4, %i2, %l4
	stx %l4, [%l7+40]	!#
	xor %o1, %o3, %o1
	for %f10, %f2, %f10
	or %g2, %l5, %g2
	or %o4, %i1, %i1
	ldx [%g4], %g4	!#
	fxor %f30, %f4, %f30
	xor %o2, %g2, %o2
	xor %g2, %o5, %l4
	lduw [%l2+96], %l5	!+
	fxor %f0, %f12, %f0
	or %o4, %o2, %o5
	or %o0, %i3, %i2
	lduw [%l2+88], %o3	!+
	fandnot2 %f14, %f6, %f14
	xor %l4, %i1, %i1
	xor %i2, %o7, %o7
	ldx [%l5], %l5	!#
	fandnot2 %f30, %f10, %f4
	xor %o7, %o5, %o5
	ldx [%o3], %o3	!#
	fand %f30, %f2, %f12
	andn %i5, %o5, %o5
	andn %i3, %l0, %i3
	ldx [%l6+232], %o7	!#
	fxor %f28, %f12, %f12
	xnor %i1, %o5, %o5
	xor %i4, %g2, %g2
	lduw [%l2+84], %i1	!+
	fxor %f14, %f4, %f4
	ldd [%l6+24], %f14	!#
	fandnot2 %f4, %f0, %f4
	xor %o7, %o5, %o7
	stx %o7, [%l7+232]	!#
	andn %g2, %g1, %g2
	fxnor %f12, %f4, %f4
	or %i0, %o2, %o2
	and %o4, %g2, %g2
	ldx [%i1], %i1	!#
	fandnot2 %f4, %f26, %f12
	or %o4, %i3, %i3
	xor %o2, %g2, %g2
	ldx [%l3+88], %o2	!#
	fxor %f12, %f10, %f12
	xor %l1, %i3, %i3
	and %i5, %g2, %g2
	ldx [%l6+120], %l1	!#
	fxor %f14, %f4, %f14
	or %o0, %g3, %g3
	xor %i3, %g2, %g2
	ldx [%l3+104], %i3	!#
	fxor %f18, %f28, %f18
	xor %l1, %g2, %l1
	stx %l1, [%l7+120]	!#
	andn %g2, %i2, %g2
	fandnot2 %f18, %f2, %f18
	andn %o1, %o4, %o1
	or %o4, %g2, %g2
	ldx [%l3+120], %o5	!#
	fandnot2 %f24, %f12, %f10
	xor %g3, %l4, %g3
	xor %o1, %g2, %g2
	ldx [%l3+96], %o0	!#
	fxor %f10, %f28, %f10
	xor %g3, %o1, %o1
	and %i5, %g2, %g2
	ldx [%l6+184], %g1	!#
	fand %f6, %f8, %f8
	xor %o2, %g4, %o4
	xnor %o1, %g2, %g2
	ldd [%l6+144], %f28	!#
	fandnot2 %f20, %f8, %f8
	xor %g1, %g2, %g1
	stx %g1, [%l7+184]	!#
	xor %i3, %o3, %l0
	for %f6, %f24, %f20
	xor %o5, %l5, %o5
	xor %o0, %i1, %o0
				!# end of included file
	lduw [%l2+92], %o1	!+
	fandnot2 %f20, %f26, %f26
	or %o4, %l0, %o7
	or %o0, %l0, %l5
	ldx [%l3+112], %i5	!#
	fand %f20, %f18, %f20
	and %o5, %o7, %o7
	andn %l0, %o4, %o3
	ldx [%o1], %o1	!#
	fandnot2 %f30, %f10, %f2
	xor %o4, %o7, %i3
	xor %l0, %o7, %o7
	lduw [%l2+100], %l4	!+
	fandnot2 %f30, %f2, %f4
	xor %i3, %l5, %l5
	or %o3, %i3, %o3
	ldx [%l3+128], %i4	!#
	fxor %f26, %f4, %f4
	andn %o0, %o7, %o2
	xor %l0, %o5, %l1
	ldx [%l4], %l4	!#
	fxor %f12, %f2, %f2
	xor %o3, %o2, %o2
	or %o7, %i3, %o7
	ldx [%l3+136], %i2	!#
	fandnot2 %f12, %f6, %f12
	xor %i5, %o1, %i5
	xor %i4, %l4, %i4
	lduw [%l2+116], %l4	!+
	fandnot2 %f0, %f4, %f4
	and %o0, %o3, %o3
	andn %l1, %o0, %l1
	lduw [%l2+112], %i1	!+
	fxnor %f2, %f4, %f4
	xor %l0, %o3, %i0
	xor %o5, %o3, %o3
	ldx [%l4], %l4	!#
	fxor %f28, %f4, %f28
	xor %o7, %l1, %o1
	and %i5, %o3, %g3
	ldx [%i1], %i1	!#
	fxor %f24, %f2, %f2
	or %i5, %o1, %o1
	xor %l5, %g3, %g3
	lduw [%l2+104], %l5	!+
	fandnot2 %f6, %f2, %f2
	xor %o2, %o1, %o1
	xor %i0, %l1, %i0
	lduw [%l2+124], %l1	!+
	for %f10, %f2, %f10
	or %i4, %o1, %o4
	and %o0, %o3, %o3
	ldx [%l5], %l5	!#
	for %f2, %f16, %f6
	ldd [%l6+80], %f24	!#
	fandnot2 %f30, %f2, %f2
	xor %o3, %o7, %o3
	andn %i5, %i0, %i0
	ldx [%l1], %l1	!#
	fxor %f18, %f2, %f2
	xor %g3, %o4, %o4
	xor %o3, %i0, %i0
	ldx [%l3+144], %o3	!#
	for %f0, %f12, %f12
	and %i4, %o1, %o1
	xor %g3, %i0, %o7
	ldx [%l6+200], %g2	!#
	fxnor %f2, %f12, %f12
	xnor %o1, %g3, %g3
	andn %o0, %o7, %o2
	ldx [%l6+152], %i3	!#
	fxor %f24, %f12, %f24
	xor %g2, %o4, %g2
	stx %g2, [%l7+200]	!#
	xor %o4, %o2, %o2
	for %f30, %f8, %f8
	xor %i3, %g3, %i3
	stx %i3, [%l7+152]	!#
	andn %o7, %i5, %o7
	fxor %f6, %f8, %f8
	xor %o2, %o7, %o7
	ldx [%l6+72], %i3	!#
	fxor %f10, %f22, %f10
	or %i4, %o7, %o2
	xor %o1, %o7, %o7
	ldx [%l6], %o1	!#
	fxor %f20, %f16, %f20
	xor %i2, %i1, %l0
	xnor %i0, %o2, %o2
	ldx [%l3+120], %o4	!#
	fand %f30, %f20, %f20
	xor %i3, %o2, %i3
	stx %i3, [%l7+72]	!#
	xor %o7, %o2, %o7
	fxor %f10, %f20, %f20
	ldd [%l6+224], %f10	!#
	fandnot2 %f0, %f20, %f20
	ldx [%l3+160], %i4	!#
	fxnor %f8, %f20, %f20
	xor %o1, %o7, %o1
	stx %o1, [%l7]	!#
	xor %o3, %l4, %i5
	fxor %f10, %f20, %f10
	xor %o4, %l5, %o4
	xor %i4, %l1, %i4
				!# end of included file
	ldd [%l3+192], %f26	!#
	ldd [%l3+208], %f8	!#
	ldd [%l3+216], %f30	!#
	lduw [%l2+156], %g2	!#
	lduw [%l2+164], %g3	!#
	ldd [%g2], %f4	!#
	ldd [%g3], %f2	!#
	lduw [%l2+168], %g3	!#
	fxor %f26, %f4, %f26
	lduw [%l2+160], %g2	!#
	fxor %f8, %f2, %f8
	ldd [%g3], %f6	!#
	ldd [%g2], %f18	!#
	fxor %f30, %f6, %f30
	lduw [%l2+172], %g2	!#
	lduw [%l2+152], %g3	!#
	ldd [%g2], %f12	!#
	ldd [%g3], %f20	!#
	lduw [%l2+120], %o3	!+
	andn %l0, %i5, %o1
	andn %o4, %l0, %g3
	ldx [%l3+152], %o5	!#
	xor %o1, %o4, %i0
	xor %i5, %o4, %g4
	ldx [%o3], %o3	!#
	or %i4, %g3, %g1
	or %g4, %o1, %o1
	lduw [%l2+108], %l1	!+
	xor %i0, %g1, %l5
	and %l0, %o1, %l4
	ldx [%l3+128], %o0	!#
	xor %l4, %i5, %l4
	xor %i5, %g3, %i2
	ldx [%l1], %l1	!#
	andn %l4, %g3, %g3
	or %i4, %i2, %i2
	std %f14, [%l7+24]	!#
	xor %o5, %o3, %o5
	xor %o0, %l1, %o0
	std %f28, [%l7+144]	!#
	andn %o1, %i4, %i1
	xor %g3, %i2, %g3
	std %f24, [%l7+80]	!#
	xor %l0, %i1, %l1
	or %o5, %g3, %i3
	std %f10, [%l7+224]	!#
	or %o5, %l1, %o3
	xor %l4, %i3, %i3
	ldd [%l3+200], %f16	!#
	xor %l5, %o3, %o3
	andn %i3, %o0, %g2
	ldx [%l6+16], %i4	!#
	xor %o3, %g2, %g2
	and %i5, %g1, %g1
	lduw [%l2+184], %o3	!+
	xor %i4, %g2, %i4
	stx %i4, [%l7+16]	!#
	xor %o4, %l1, %o2
	xor %g1, %g3, %g1
	and %i0, %o2, %i0
	ldd [%l3+224], %f10	!#
	fxor %f16, %f18, %f16
	andn %o5, %i0, %i0
	or %i5, %o2, %i4
	ldx [%o3], %o3	!#
	fand %f26, %f8, %f28
	xor %g1, %i0, %i0
	andn %i4, %o0, %o4
	ldx [%l6+104], %o7	!#
	fxor %f28, %f30, %f28
	xor %i0, %o4, %o4
	andn %i1, %i5, %i1
	lduw [%l2+176], %g1	!+
	fand %f8, %f28, %f2
	xor %o7, %o4, %o7
	stx %o7, [%l7+104]	!#
	xor %i1, %l0, %i1
	fxnor %f2, %f26, %f4
	and %g3, %l5, %g3
	and %o5, %i1, %l5
	ldx [%g1], %g1	!#
	for %f26, %f8, %f24
	andn %o1, %g3, %o1
	xor %l1, %g3, %g3
	lduw [%l2+188], %l1	!+
	for %f24, %f30, %f24
	xor %o1, %l5, %l5
	or %l4, %i2, %i2
	lduw [%l2+192], %l4	!+
	fandnot2 %f30, %f26, %f22
	or %o5, %g3, %o7
	andn %i3, %o1, %o1
	ldx [%l1], %l1	!#
	for %f16, %f22, %f22
	xor %i2, %o7, %o7
	or %i0, %o1, %i0
	ldx [%l4], %l4	!#
	fxor %f24, %f22, %f22
	or %o0, %o7, %o7
	xor %o1, %o2, %o1
	ldx [%l6+192], %o2	!#
	fxor %f10, %f12, %f10
	xnor %l5, %o7, %o7
	xor %i0, %g4, %i0
	ldx [%l3+232], %l5	!#
	fand %f16, %f4, %f0
	xor %o2, %o7, %o2
	stx %o2, [%l7+192]	!#
	xor %g2, %g3, %g2
	fxnor %f8, %f4, %f4
	and %g4, %g3, %g3
	andn %i4, %g2, %g2
	ldx [%l3+216], %g4	!#
	fandnot2 %f8, %f16, %f8
	xor %g3, %i1, %g3
	and %o5, %g2, %g2
	ldx [%l3+240], %i5	!#
	fxnor %f16, %f0, %f12
	andn %o5, %i0, %i0
	xor %g3, %g2, %g2
	ldx [%l3+248], %o5	!#
	fxor %f28, %f0, %f0
	xor %o1, %i0, %i0
	or %o0, %g2, %g2
	ldx [%l6+56], %o1	!#
	fxor %f2, %f0, %f14
	xor %l5, %o3, %l0
	xor %i0, %g2, %g2
	ldd [%l3+184], %f24	!#
	for %f10, %f14, %f6
	xor %o1, %g2, %o1
	stx %o1, [%l7+56]	!#
	xor %g4, %g1, %o4
	fand %f10, %f12, %f18
	xor %i5, %l1, %i5
	xor %o5, %l4, %o5
				!# end of included file
	lduw [%l2+180], %o2	!+
	fxor %f0, %f18, %f18
	xor %l0, %o4, %l5
	andn %o4, %l0, %o7
	ldx [%l3+224], %o0	!#
	for %f30, %f8, %f30
	xor %o7, %i5, %l1
	and %o5, %o7, %g2
	ldx [%o2], %o2	!#
	fandnot2 %f26, %f8, %f8
	or %o5, %l1, %i3
	andn %l5, %o5, %l4
	lduw [%l2+196], %o3	!+
	fand %f10, %f8, %f8
	xor %l5, %i3, %i3
	xor %o0, %o2, %o0
	ldx [%l3], %i4	!#
	fxor %f22, %f6, %f6
	andn %i3, %o4, %i0
	or %i3, %o5, %o2
	ldx [%o3], %o3	!#
	fxor %f26, %f14, %f14
	or %i0, %i5, %o1
	xor %i0, %l0, %g4
	ldd [%l6+248], %f0	!#
	fxor %f24, %f20, %f24
	or %o0, %i0, %i0
	xor %i4, %o3, %i4
	ldd [%l6+88], %f20	!#
	fand %f24, %f6, %f6
	xor %o1, %l5, %o1
	andn %g4, %o5, %g1
	ldd [%l6+168], %f28	!#
	fxor %f18, %f6, %f6
	xor %i5, %g1, %g1
	xor %o1, %o5, %o1
	lduw [%l2+16], %i1	!+
	fxnor %f18, %f8, %f18
	andn %o0, %g1, %o3
	andn %l1, %o1, %i2
	lduw [%l2+24], %l5	!+
	fand %f16, %f8, %f8
	xor %i3, %o3, %o3
	xor %i2, %g4, %g3
	ldx [%i1], %i1	!#
	fxor %f0, %f6, %f0
	andn %o0, %g3, %i2
	xor %o2, %l1, %l1
	ldx [%l5], %l5	!#
	for %f26, %f4, %f22
	xor %o1, %i2, %i2
	and %o4, %o2, %o2
	lduw [%l2+20], %g4	!+
	fxnor %f22, %f6, %f22
	or %i4, %i2, %i2
	andn %o3, %i5, %o4
	ldx [%l6+32], %i3	!#
	for %f10, %f22, %f22
	xnor %o3, %i2, %i2
	xor %g2, %o4, %g2
	ldx [%g4], %g4	!#
	fxor %f16, %f2, %f6
	xor %i3, %i2, %i3
	stx %i3, [%l7+32]	!#
	and %o0, %g2, %g2
	fand %f6, %f26, %f6
	andn %o0, %o4, %o4
	xor %o2, %g2, %g2
	lduw [%l2+28], %o2	!+
	fandnot2 %f10, %f6, %f6
	xor %l1, %o4, %o4
	andn %g2, %i4, %g2
	ldx [%l6+112], %i3	!#
	for %f16, %f2, %f2
	xor %o4, %g2, %g2
	andn %l0, %g3, %g3
	ldx [%o2], %o2	!#
	fxor %f4, %f2, %f2
	xor %i3, %g2, %i3
	stx %i3, [%l7+112]	!#
	or %g1, %g3, %g3
	fandnot2 %f2, %f16, %f16
	andn %o7, %o1, %o1
	or %l1, %g2, %g2
	ldx [%l3+8], %g1	!#
	fandnot2 %f4, %f14, %f14
	xor %l0, %o5, %o7
	andn %o0, %g2, %g2
	ldx [%l3+24], %l1	!#
	fand %f10, %f14, %f14
	xor %o1, %g2, %g2
	xor %o7, %o1, %o7
	ldd [%l6+48], %f10	!#
	fxor %f2, %f6, %f6
	or %i4, %g2, %g2
	andn %o7, %o0, %o7
	ldx [%l6+208], %o1	!#
	fxnor %f12, %f6, %f12
	xor %g3, %i0, %i0
	or %l4, %i5, %l4
	ldx [%l3+16], %i5	!#
	for %f24, %f6, %f2
	xnor %i0, %g2, %g2
	xor %l4, %o7, %o7
	ldx [%l6+160], %l4	!#
	fxor %f18, %f2, %f2
	xor %o1, %g2, %o1
	stx %o1, [%l7+208]	!#
	and %i4, %o7, %o7
	fxor %f20, %f2, %f20
	xor %g1, %i1, %l0
	xnor %o3, %o7, %o7
	ldx [%l3+32], %i4	!#
	fxor %f30, %f12, %f30
	xor %l4, %o7, %l4
	stx %l4, [%l7+160]	!#
	xor %l1, %l5, %o5
	fxor %f12, %f22, %f22
	xor %i5, %g4, %i5
	xor %i4, %o2, %i4
				!# end of included file
	lduw [%l2+12], %o3	!+
	fxor %f8, %f14, %f8
	andn %l0, %o5, %g2
	andn %l0, %i5, %l1
	ldx [%l3], %o0	!#
	for %f16, %f6, %f16
	or %l1, %o5, %i0
	xor %l0, %i5, %g4
	ldx [%o3], %o3	!#
	fornot2 %f24, %f16, %f16
	xor %g2, %i5, %o2
	and %i4, %i0, %l5
	lduw [%l2+8], %l4	!+
	fxor %f22, %f16, %f16
	xor %o2, %l5, %g1
	xor %o5, %l5, %i3
	ldx [%l3+248], %o4	!#
	fxor %f28, %f16, %f28
	and %i3, %g4, %i3
	andn %o5, %i5, %o7
	ldx [%l4], %l4	!#
	fxor %f30, %f14, %f14
	andn %i5, %o5, %o1
	xor %l1, %i3, %i2
	std %f0, [%l7+248]	!#
	for %f24, %f8, %f8
	xor %o0, %o3, %o0
	or %g1, %o1, %o3
	std %f20, [%l7+88]	!#
	fxor %f14, %f8, %f8
	xor %o4, %l4, %o4
	or %i4, %i2, %i2
	std %f28, [%l7+168]	!#
	fxor %f10, %f8, %f10
	andn %i4, %g4, %l4
	xor %o7, %i2, %i2
	std %f10, [%l7+48]	!#
	xor %o1, %l4, %o1
	or %o0, %i2, %o7
	lduw [%l2+216], %l3	!+
	or %o0, %o1, %o1
	xor %i3, %o7, %o7
	lduw [%l2+212], %l2	!+
	xor %g1, %o1, %o1
	and %o4, %o7, %o7
	ldx [%l6+128], %g1	!#
	xnor %o1, %o7, %o7
	or %g2, %l5, %g2
	lduw [%l2+32], %o1	!+
	xor %g1, %o7, %g1
	stx %g1, [%l7+128]	!#
	andn %i2, %o2, %o2
	xor %g2, %g4, %g2
	andn %o0, %o2, %g4
	ldx [%o1], %o1	!#
	xor %g2, %g4, %g4
	xor %l4, %g2, %g3
	lduw [%l2+52], %o7	!+
	andn %i2, %g3, %g3
	xor %o3, %o2, %o2
	lduw [%l2+48], %g1	!+
	and %o0, %g3, %i3
	and %l0, %o3, %o3
	ldx [%o7], %o7	!#
	xor %o2, %i3, %i3
	or %l0, %g3, %g3
	ldx [%g1], %g1	!#
	and %o4, %i3, %i5
	andn %i2, %o3, %o2
	ldx [%l6+240], %i1	!#
	xor %g4, %i5, %i5
	andn %g2, %o2, %g2
	lduw [%l2+36], %g4	!+
	xor %i1, %i5, %i1
	stx %i1, [%l7+240]	!#
	or %g2, %l1, %g2
	andn %g2, %o0, %i1
	or %o0, %l1, %l1
	ldx [%g4], %g4	!#
	xor %g3, %i1, %i1
	xor %o2, %l1, %l1
	ldx [%l3+24], %o2	!#
	or %g2, %i2, %g2
	andn %o4, %i1, %i1
	ldx [%l6+64], %i2	!#
	xnor %l1, %i1, %i1
	xor %g2, %o5, %g2
	ldx [%l3+56], %o5	!#
	xor %i2, %i1, %i2
	stx %i2, [%l7+64]	!#
	xor %i0, %o3, %o3
	andn %o3, %l5, %o3
	andn %o0, %g2, %g2
	ldx [%l6+176], %l5	!#
	andn %i3, %l4, %i3
	xor %o3, %g2, %g2
	lduw [%l2+204], %l6	!+
	xor %i3, %l1, %i3
	or %o4, %g2, %g2
	ldx [%l3+64], %l1	!#
	xor %o2, %o1, %o4
	xnor %i3, %g2, %g2
	ldx [%l3+32], %o0	!#
	xor %l5, %g2, %l5
	stx %l5, [%l7+176]	!#
	xor %l1, %o7, %i4
	xor %o5, %g1, %o5
	xor %o0, %g4, %o0
				!# end of included file
	cmp %l6,0
	bne .LL57
	ld [%l2+208],%l7
	srlx %i7, 32, %o0	!##
	!#EPILOGUE#
	ld [%sp+96],%i7
	ld [%sp+100],%o7
	ldd [%sp+104],%l0
	ldd [%sp+112],%l2
	ldd [%sp+120],%l4
	ldd [%sp+128],%l6
	ldd [%sp+136],%i0
	ldd [%sp+144],%i2
	ldd [%sp+152],%i4
	retl
	add %sp,184,%sp
.LLfe12:
	.size	 asm_do_all,.LLfe12-asm_do_all
	.align 4
	.global asm_do_all_fancy
	.type	 asm_do_all_fancy,#function
	.proc	016
asm_do_all_fancy:
	!#PROLOGUE# 0
	!# vars= 0, regs= 18/0, args= 0, extra= 84
	add %sp,-184,%sp
	st %i7,[%sp+96]
	sub %sp,-184,%i7	!# set up frame pointer
	st %o7,[%sp+100]
	std %l0,[%sp+104]
	std %l2,[%sp+112]
	std %l4,[%sp+120]
	std %l6,[%sp+128]
	std %i0,[%sp+136]
	std %i2,[%sp+144]
	std %i4,[%sp+152]
	!#PROLOGUE# 1
	mov %o0,%l2
	ld [%l2+200],%l5
	sethi %hi(1513943058),%g2
	ld [%l2+204],%l6
	or %g2,%lo(1513943058),%g2
	ld [%l2+208],%l7
	sllx %g2, 32, %g2
	or %i7, %g2, %i7	!##
	cmp %o1,0
	ble .LL62
	cmp %o1,1
	bne .LL63
	nop
	lduw [%l2+16], %g2	!+
	ldx [%l5+8], %i3	!#
	ldx [%g2], %i2	!#
	lduw [%l2+24], %g2	!+
	ldx [%l5+24], %i1	!#
	ldx [%g2], %i0	!#
	lduw [%l2+20], %g2	!+
	ldx [%l5+16], %g4	!#
	ldx [%g2], %g3	!#
	lduw [%l2+28], %g2	!+
	ldx [%l5+32], %i5	!#
	ldx [%g2], %g2	!#
	xor %i3, %i2, %l1
	xor %i1, %i0, %i4
	xor %g4, %g3, %g4
	xor %i5, %g2, %i5
	lduw [%l2+12], %o2	!+
	andn %l1, %i4, %g2
	andn %l1, %g4, %l3
	ldx [%l5], %o0	!#
	or %l3, %i4, %i0
	xor %l1, %g4, %l0
	ldx [%o2], %o2	!#
	xor %g2, %g4, %o3
	and %i5, %i0, %g1
	lduw [%l2+8], %l4	!+
	xor %o3, %g1, %o4
	xor %i4, %g1, %i2
	ldx [%l5+248], %o5	!#
	and %i2, %l0, %i2
	andn %i4, %g4, %o7
	ldx [%l4], %l4	!#
	andn %g4, %i4, %o1
	xor %l3, %i2, %i3
	xor %o0, %o2, %o0
	or %o4, %o1, %o2
	xor %o5, %l4, %o5
	or %i5, %i3, %i3
	andn %i5, %l0, %l4
	xor %o7, %i3, %i3
	xor %o1, %l4, %o1
	or %o0, %i3, %o7
	lduw [%l2+216], %l5	!+
	or %o0, %o1, %o1
	xor %i2, %o7, %o7
	lduw [%l2+212], %l2	!+
	xor %o4, %o1, %o1
	and %o5, %o7, %o7
	ldx [%l6+128], %o4	!#
	xnor %o1, %o7, %o7
	or %g2, %g1, %g2
	lduw [%l2+32], %o1	!+
	xor %o4, %o7, %o4
	stx %o4, [%l7+128]	!#
	andn %i3, %o3, %o3
	xor %g2, %l0, %g2
	andn %o0, %o3, %l0
	ldx [%o1], %o1	!#
	xor %g2, %l0, %l0
	xor %l4, %g2, %g3
	lduw [%l2+52], %o7	!+
	andn %i3, %g3, %g3
	xor %o2, %o3, %o3
	lduw [%l2+48], %o4	!+
	and %o0, %g3, %i2
	and %l1, %o2, %o2
	ldx [%o7], %o7	!#
	xor %o3, %i2, %i2
	or %l1, %g3, %g3
	ldx [%o4], %o4	!#
	and %o5, %i2, %g4
	andn %i3, %o2, %o3
	ldx [%l6+240], %i1	!#
	xor %l0, %g4, %g4
	andn %g2, %o3, %g2
	lduw [%l2+36], %l0	!+
	xor %i1, %g4, %i1
	stx %i1, [%l7+240]	!#
	or %g2, %l3, %g2
	andn %g2, %o0, %i1
	or %o0, %l3, %l3
	ldx [%l0], %l0	!#
	xor %g3, %i1, %i1
	xor %o3, %l3, %l3
	ldx [%l5+24], %o3	!#
	or %g2, %i3, %g2
	andn %o5, %i1, %i1
	ldx [%l6+64], %i3	!#
	xnor %l3, %i1, %i1
	xor %g2, %i4, %g2
	ldx [%l5+56], %i4	!#
	xor %i3, %i1, %i3
	stx %i3, [%l7+64]	!#
	xor %i0, %o2, %o2
	andn %o2, %g1, %o2
	andn %o0, %g2, %g2
	ldx [%l6+176], %g1	!#
	andn %i2, %l4, %i2
	xor %o2, %g2, %g2
	lduw [%l2+204], %l6	!+
	xor %i2, %l3, %i2
	or %o5, %g2, %g2
	ldx [%l5+64], %l3	!#
	xor %o3, %o1, %o5
	xnor %i2, %g2, %g2
	ldx [%l5+32], %o0	!#
	xor %g1, %g2, %g1
	stx %g1, [%l7+176]	!#
	xor %l3, %o7, %i5
	xor %i4, %o4, %i4
	xor %o0, %l0, %o0
				!# end of included file
	b .LL64
	ld [%l2+208],%l7
.LL63:
	lduw [%l2+156], %g2	!+
	ldx [%l5+192], %i3	!#
	ldx [%g2], %i2	!#
	lduw [%l2+164], %g2	!+
	ldx [%l5+208], %i1	!#
	ldx [%g2], %i0	!#
	lduw [%l2+168], %g2	!+
	ldx [%l5+216], %i4	!#
	ldx [%g2], %g3	!#
	lduw [%l2+160], %g2	!+
	ldx [%l5+200], %l1	!#
	ldx [%g2], %g2	!#
	xor %i3, %i2, %o0
	xor %i1, %i0, %g4
	xor %i4, %g3, %i4
	xor %l1, %g2, %l1
	lduw [%l2+172], %o7	!+
	and %o0, %g4, %l0
	or %o0, %g4, %o1
	ldx [%l5+224], %i5	!#
	xor %l0, %i4, %l0
	or %o1, %i4, %o1
	ldx [%o7], %o7	!#
	and %g4, %l0, %o2
	andn %i4, %o0, %l3
	lduw [%l2+152], %o3	!+
	xnor %o2, %o0, %i0
	or %l1, %l3, %l3
	ldx [%l5+184], %o5	!#
	and %l1, %i0, %i3
	xor %o1, %l3, %l3
	ldx [%o3], %o3	!#
	xnor %l1, %i3, %l4
	xor %l0, %i3, %i3
	xor %i5, %o7, %i5
	xor %o5, %o3, %o5
	lduw [%l2+216], %l5	!+
	and %i5, %l4, %g3
	xor %o2, %i3, %l0
	lduw [%l2+212], %l2	!+
	xor %i3, %g3, %g3
	or %i5, %l0, %o3
	xor %l3, %o3, %o3
	andn %g4, %l1, %g2
	lduw [%l2+16], %i2	!+
	and %o5, %o3, %o3
	andn %o0, %g2, %o1
	ldx [%l6+248], %l3	!#
	xor %g3, %o3, %o3
	and %i5, %o1, %o1
	ldx [%i2], %i2	!#
	xor %l3, %o3, %l3
	stx %l3, [%l7+248]	!#
	xnor %g4, %i0, %i0
	or %l1, %o2, %i3
	xor %l1, %o2, %o2
	lduw [%l2+24], %i1	!+
	xor %i0, %i3, %i3
	and %o2, %o0, %o2
	lduw [%l2+20], %l3	!+
	xnor %g3, %o1, %g3
	andn %i5, %o2, %o2
	ldx [%i1], %i1	!#
	xor %i3, %o2, %g4
	or %o0, %i0, %o4
	ldx [%l3], %l3	!#
	or %o5, %g4, %o2
	xnor %l4, %g4, %l4
	ldx [%l6+88], %o7	!#
	xor %g3, %o2, %o2
	xnor %o4, %o3, %o4
	lduw [%l2+28], %o3	!+
	xor %o7, %o2, %o7
	stx %o7, [%l7+88]	!#
	or %i5, %o4, %o4
	xor %l4, %o4, %o4
	andn %i3, %l1, %i3
	ldx [%o3], %o3	!#
	or %i3, %g4, %i3
	or %i4, %g2, %g2
	ldx [%l5+8], %g4	!#
	orn %o5, %i3, %i3
	xor %g2, %l4, %g2
	ldx [%l6+168], %l4	!#
	xor %o4, %i3, %i3
	xor %o0, %l0, %l0
	ldx [%l5+24], %o4	!#
	xor %l4, %i3, %l4
	stx %l4, [%l7+168]	!#
	andn %i0, %l0, %l0
	and %i5, %l0, %l0
	and %l1, %o1, %o1
	ldx [%l5+32], %i5	!#
	xor %g4, %i2, %l1
	xor %o1, %l0, %o1
	ldx [%l5+16], %g4	!#
	xor %g2, %l0, %g2
	or %o5, %o1, %o1
	ldx [%l6+48], %o7	!#
	xor %g2, %o1, %o1
	lduw [%l2+204], %l6	!+
	xor %o7, %o1, %o7
	stx %o7, [%l7+48]	!#
	xor %o4, %i1, %i4
	xor %g4, %l3, %g4
	xor %i5, %o3, %i5
	ld [%l2+208],%l7
	lduw [%l2+12], %o2	!+
	andn %l1, %i4, %g2
	andn %l1, %g4, %l3
	ldx [%l5], %o0	!#
	or %l3, %i4, %i0
	xor %l1, %g4, %l0
	ldx [%o2], %o2	!#
	xor %g2, %g4, %o3
	and %i5, %i0, %g1
	lduw [%l2+8], %l4	!+
	xor %o3, %g1, %o4
	xor %i4, %g1, %i3
	ldx [%l5+248], %o5	!#
	and %i3, %l0, %i3
	andn %i4, %g4, %o7
	ldx [%l4], %l4	!#
	andn %g4, %i4, %o1
	xor %l3, %i3, %i2
	xor %o0, %o2, %o0
	or %o4, %o1, %o2
	xor %o5, %l4, %o5
	or %i5, %i2, %i2
	andn %i5, %l0, %l4
	xor %o7, %i2, %i2
	xor %o1, %l4, %o1
	or %o0, %i2, %o7
	or %o0, %o1, %o1
	xor %i3, %o7, %o7
	xor %o4, %o1, %o1
	and %o5, %o7, %o7
	ldx [%l6+128], %o4	!#
	xnor %o1, %o7, %o7
	or %g2, %g1, %g2
	lduw [%l2+32], %o1	!+
	xor %o4, %o7, %o4
	stx %o4, [%l7+128]	!#
	andn %i2, %o3, %o3
	xor %g2, %l0, %g2
	andn %o0, %o3, %l0
	ldx [%o1], %o1	!#
	xor %g2, %l0, %l0
	xor %l4, %g2, %g3
	lduw [%l2+52], %o7	!+
	andn %i2, %g3, %g3
	xor %o2, %o3, %o3
	lduw [%l2+48], %o4	!+
	and %o0, %g3, %i3
	and %l1, %o2, %o2
	ldx [%o7], %o7	!#
	xor %o3, %i3, %i3
	or %l1, %g3, %g3
	ldx [%o4], %o4	!#
	and %o5, %i3, %g4
	andn %i2, %o2, %o3
	ldx [%l6+240], %i1	!#
	xor %l0, %g4, %g4
	andn %g2, %o3, %g2
	lduw [%l2+36], %l0	!+
	xor %i1, %g4, %i1
	stx %i1, [%l7+240]	!#
	or %g2, %l3, %g2
	andn %g2, %o0, %i1
	or %o0, %l3, %l3
	ldx [%l0], %l0	!#
	xor %g3, %i1, %i1
	xor %o3, %l3, %l3
	ldx [%l5+24], %o3	!#
	or %g2, %i2, %g2
	andn %o5, %i1, %i1
	ldx [%l6+64], %i2	!#
	xnor %l3, %i1, %i1
	xor %g2, %i4, %g2
	ldx [%l5+56], %i4	!#
	xor %i2, %i1, %i2
	stx %i2, [%l7+64]	!#
	xor %i0, %o2, %o2
	andn %o2, %g1, %o2
	andn %o0, %g2, %g2
	ldx [%l6+176], %g1	!#
	andn %i3, %l4, %i3
	xor %o2, %g2, %g2
	ldx [%l5+32], %o0	!#
	xor %i3, %l3, %i3
	or %o5, %g2, %g2
	ldx [%l5+64], %l3	!#
	xor %o3, %o1, %o5
	xnor %i3, %g2, %g2
	xor %g1, %g2, %g1
	stx %g1, [%l7+176]	!#
	xor %l3, %o7, %i5
	xor %i4, %o4, %i4
	xor %o0, %l0, %o0
				!# end of included file
.LL64:
	lduw [%l2+40], %o3	!+
	xor %o5, %i5, %l0
	and %i5, %i4, %i1
	ldx [%l5+40], %l1	!#
	xor %l0, %i4, %o4
	andn %o5, %i1, %o2
	ldx [%o3], %o3	!#
	andn %o0, %o2, %o7
	andn %i4, %o2, %i0
	lduw [%l2+44], %o1	!+
	xor %o4, %o7, %g3
	or %i1, %o7, %o7
	ldx [%l5+48], %g4	!#
	andn %o7, %l0, %o7
	or %i0, %o0, %l3
	ldx [%o1], %o1	!#
	xor %l1, %o3, %l1
	andn %i5, %o2, %o3
	ldd [%l5+184], %f12	!#
	or %l1, %o7, %i5
	xor %g4, %o1, %g4
	ldd [%l5+152], %f26	!#
	xor %g3, %i5, %i5
	and %g4, %l3, %i3
	ldx [%l6+96], %o1	!#
	xnor %i5, %i3, %i3
	xor %g3, %i0, %i0
	lduw [%l2+60], %l0	!+
	xor %o1, %i3, %o1
	stx %o1, [%l7+96]	!#
	xor %o2, %i3, %i3
	andn %i3, %o0, %i3
	and %o0, %i0, %i0
	ldx [%l0], %l0	!#
	xor %o4, %i3, %i3
	xor %o3, %i0, %o2
	ldd [%l5+192], %f16	!#
	and %l1, %o2, %o2
	or %g3, %o5, %g3
	lduw [%l2+64], %o1	!+
	xor %i3, %o2, %o5
	xor %i4, %o0, %i3
	lduw [%l2+76], %o2	!+
	andn %i3, %o7, %o7
	xor %g3, %o0, %g3
	ldx [%o1], %o1	!#
	andn %l1, %g3, %i2
	or %o3, %o7, %o3
	ldx [%o2], %o2	!#
	xor %o7, %i2, %i2
	or %g3, %i0, %g3
	lduw [%l2+72], %o7	!+
	or %g4, %i2, %g2
	xor %o3, %i5, %o3
	ldx [%l6+8], %i5	!#
	xor %o5, %g2, %g2
	and %i3, %g3, %i3
	ldx [%o7], %o7	!#
	xor %i5, %g2, %i5
	stx %i5, [%l7+8]	!#
	andn %l3, %i3, %i3
	and %l1, %g3, %g3
	or %g4, %i3, %i3
	ldx [%l5+64], %l3	!#
	xor %o3, %g3, %g3
	xor %o0, %o4, %g2
	ldx [%l6+136], %i5	!#
	xnor %g3, %i3, %i3
	andn %g2, %o3, %o4
	ldx [%l5+72], %o3	!#
	xor %i5, %i3, %i5
	stx %i5, [%l7+136]	!#
	xor %o4, %i2, %o4
	or %i1, %i0, %i0
	and %l1, %i1, %i1
	ldx [%l5+96], %i5	!#
	andn %l1, %o4, %o4
	xor %i0, %i1, %i1
	ldx [%l5+88], %i4	!#
	xor %g2, %o4, %o4
	andn %g4, %i1, %i1
	ldx [%l6+216], %g2	!#
	xor %l3, %l0, %o0
	xnor %o4, %i1, %i1
	ldd [%l5+176], %f14	!#
	xor %g2, %i1, %g2
	stx %g2, [%l7+216]	!#
	xor %o3, %o1, %l1
	xor %i5, %o2, %i5
	xor %i4, %o7, %i4
				!# end of included file
	lduw [%l2+144], %g3	!#
	lduw [%l2+128], %g2	!#
	ldd [%g3], %f22	!#
	ldd [%g2], %f18	!#
	lduw [%l2+148], %g3	!#
	fxor %f12, %f22, %f12
	lduw [%l2+140], %g2	!#
	fxor %f26, %f18, %f26
	ldd [%g3], %f8	!#
	ldd [%g2], %f6	!#
	ldd [%l5+168], %f30	!#
	fxor %f16, %f8, %f16
	lduw [%l2+56], %o3	!+
	fxor %f14, %f6, %f14
	xor %o0, %l1, %g3
	and %l1, %i5, %g2
	ldx [%l5+56], %o5	!#
	fxor %f12, %f26, %f0
	xor %g3, %i5, %g3
	and %l1, %i4, %o4
	ldx [%o3], %o3	!#
	fand %f26, %f16, %f26
	and %o0, %g3, %l0
	xor %o4, %g3, %o4
	lduw [%l2+68], %i1	!+
	fandnot2 %f26, %f12, %f10
	or %i4, %l0, %o7
	andn %i5, %l0, %i0
	ldx [%l5+80], %g4	!#
	fxor %f16, %f26, %f8
	xor %g3, %o7, %l4
	xor %i0, %i4, %o2
	ldx [%i1], %i1	!#
	fxor %f0, %f16, %f20
	xor %o5, %o3, %o5
	xor %l1, %l0, %o3
	lduw [%l2+132], %o1	!#
	for %f10, %f8, %f4
	and %o5, %o2, %i2
	xor %g4, %i1, %g4
	lduw [%l2+136], %g1	!#
	fandnot2 %f14, %f10, %f24
	andn %o3, %i4, %i3
	xor %i4, %i2, %i2
	ldd [%o1], %f2	!#
	fandnot2 %f4, %f14, %f4
	or %o5, %i3, %i1
	or %g4, %i2, %i2
	ldd [%g1], %f6	!#
	fxor %f20, %f24, %f22
	andn %g3, %i3, %g3
	xor %l4, %i1, %i1
	ldx [%l6+40], %l3	!#
	fxor %f8, %f4, %f4
	xor %i1, %i2, %i2
	xor %g3, %g2, %i1
	ldd [%l5+160], %f28	!#
	for %f16, %f22, %f18
	or %o2, %g3, %o2
	and %i5, %i2, %o1
	lduw [%l2+80], %g1	!+
	xor %l3, %i2, %l3
	stx %l3, [%l7+40]	!#
	xor %o1, %o3, %o1
	for %f10, %f4, %f10
	or %g2, %l0, %g2
	or %o5, %i1, %i1
	ldx [%g1], %g1	!#
	fxor %f28, %f2, %f28
	xor %o2, %g2, %o2
	xor %g2, %i4, %l3
	lduw [%l2+96], %l0	!+
	fxor %f30, %f6, %f30
	or %o5, %o2, %i4
	or %o0, %i3, %i2
	lduw [%l2+88], %o3	!+
	fandnot2 %f18, %f12, %f18
	xor %l3, %i1, %i1
	xor %i2, %o7, %o7
	ldx [%l0], %l0	!#
	fandnot2 %f28, %f10, %f2
	xor %o7, %i4, %i4
	ldx [%o3], %o3	!#
	fand %f28, %f4, %f6
	andn %g4, %i4, %i4
	andn %i3, %l1, %i3
	ldx [%l6+232], %o7	!#
	fxor %f22, %f6, %f6
	xnor %i1, %i4, %i4
	xor %i5, %g2, %g2
	lduw [%l2+84], %i1	!+
	fxor %f18, %f2, %f2
	ldd [%l6+24], %f18	!#
	fandnot2 %f2, %f30, %f2
	xor %o7, %i4, %o7
	stx %o7, [%l7+232]	!#
	andn %g2, %l4, %g2
	fxnor %f6, %f2, %f2
	or %i0, %o2, %o2
	and %o5, %g2, %g2
	ldx [%i1], %i1	!#
	fandnot2 %f2, %f0, %f6
	or %o5, %i3, %i3
	xor %o2, %g2, %g2
	ldx [%l5+88], %o2	!#
	fxor %f6, %f10, %f6
	xor %o4, %i3, %i3
	and %g4, %g2, %g2
	ldx [%l6+120], %o4	!#
	fxor %f18, %f2, %f18
	or %o0, %g3, %g3
	xor %i3, %g2, %g2
	ldx [%l5+104], %i3	!#
	fxor %f26, %f22, %f26
	xor %o4, %g2, %o4
	stx %o4, [%l7+120]	!#
	andn %g2, %i2, %g2
	fandnot2 %f26, %f4, %f26
	andn %o1, %o5, %o1
	or %o5, %g2, %g2
	ldx [%l5+120], %i4	!#
	fandnot2 %f16, %f6, %f10
	xor %g3, %l3, %g3
	xor %o1, %g2, %g2
	ldx [%l5+96], %o0	!#
	fxor %f10, %f22, %f10
	xor %g3, %o1, %o1
	and %g4, %g2, %g2
	ldx [%l6+184], %l4	!#
	fand %f12, %f8, %f8
	xor %o2, %g1, %o5
	xnor %o1, %g2, %g2
	ldd [%l6+144], %f22	!#
	fandnot2 %f14, %f8, %f8
	xor %l4, %g2, %l4
	stx %l4, [%l7+184]	!#
	xor %i3, %o3, %l1
	for %f12, %f16, %f14
	xor %i4, %l0, %i4
	xor %o0, %i1, %o0
				!# end of included file
	lduw [%l2+92], %o1	!+
	fandnot2 %f14, %f0, %f0
	or %o5, %l1, %o7
	or %o0, %l1, %l3
	ldx [%l5+112], %g4	!#
	fand %f14, %f26, %f14
	and %i4, %o7, %o7
	andn %l1, %o5, %o3
	ldx [%o1], %o1	!#
	fandnot2 %f28, %f10, %f4
	xor %o5, %o7, %i3
	xor %l1, %o7, %o7
	lduw [%l2+100], %l0	!+
	fandnot2 %f28, %f4, %f2
	xor %i3, %l3, %l3
	or %o3, %i3, %o3
	ldx [%l5+128], %i5	!#
	fxor %f0, %f2, %f2
	andn %o0, %o7, %o2
	xor %l1, %i4, %o4
	ldx [%l0], %l0	!#
	fxor %f6, %f4, %f4
	xor %o3, %o2, %o2
	or %o7, %i3, %o7
	ldx [%l5+136], %i2	!#
	fandnot2 %f6, %f12, %f6
	xor %g4, %o1, %g4
	xor %i5, %l0, %i5
	lduw [%l2+116], %l0	!+
	fandnot2 %f30, %f2, %f2
	and %o0, %o3, %o3
	andn %o4, %o0, %o4
	lduw [%l2+112], %i1	!+
	fxnor %f4, %f2, %f2
	xor %l1, %o3, %i0
	xor %i4, %o3, %o3
	ldx [%l0], %l0	!#
	fxor %f22, %f2, %f22
	xor %o7, %o4, %o1
	and %g4, %o3, %g3
	ldx [%i1], %i1	!#
	fxor %f16, %f4, %f4
	or %g4, %o1, %o1
	xor %l3, %g3, %g3
	lduw [%l2+104], %l3	!+
	fandnot2 %f12, %f4, %f4
	xor %o2, %o1, %o1
	xor %i0, %o4, %i0
	lduw [%l2+124], %o4	!+
	for %f10, %f4, %f10
	or %i5, %o1, %o5
	and %o0, %o3, %o3
	ldx [%l3], %l3	!#
	for %f4, %f20, %f12
	ldd [%l6+80], %f16	!#
	fandnot2 %f28, %f4, %f4
	xor %o3, %o7, %o3
	andn %g4, %i0, %i0
	ldx [%o4], %o4	!#
	fxor %f26, %f4, %f4
	xor %g3, %o5, %o5
	xor %o3, %i0, %i0
	ldx [%l5+144], %o3	!#
	for %f30, %f6, %f6
	and %i5, %o1, %o1
	xor %g3, %i0, %o7
	ldx [%l6+200], %g2	!#
	fxnor %f4, %f6, %f6
	xnor %o1, %g3, %g3
	andn %o0, %o7, %o2
	ldx [%l6+152], %i3	!#
	fxor %f16, %f6, %f16
	xor %g2, %o5, %g2
	stx %g2, [%l7+200]	!#
	xor %o5, %o2, %o2
	for %f28, %f8, %f8
	xor %i3, %g3, %i3
	stx %i3, [%l7+152]	!#
	andn %o7, %g4, %o7
	fxor %f12, %f8, %f8
	xor %o2, %o7, %o7
	ldx [%l6+72], %i3	!#
	fxor %f10, %f24, %f10
	or %i5, %o7, %o2
	xor %o1, %o7, %o7
	ldx [%l6], %o1	!#
	fxor %f14, %f20, %f14
	xor %i2, %i1, %l1
	xnor %i0, %o2, %o2
	ldx [%l5+120], %o5	!#
	fand %f28, %f14, %f14
	xor %i3, %o2, %i3
	stx %i3, [%l7+72]	!#
	xor %o7, %o2, %o7
	fxor %f10, %f14, %f14
	ldd [%l6+224], %f10	!#
	fandnot2 %f30, %f14, %f14
	ldx [%l5+160], %i5	!#
	fxnor %f8, %f14, %f14
	xor %o1, %o7, %o1
	stx %o1, [%l7]	!#
	xor %o3, %l0, %g4
	fxor %f10, %f14, %f10
	xor %o5, %l3, %o5
	xor %i5, %o4, %i5
				!# end of included file
	lduw [%l2+120], %o3	!+
	andn %l1, %g4, %o1
	andn %o5, %l1, %g3
	ldx [%l5+152], %i4	!#
	xor %o1, %o5, %i0
	xor %g4, %o5, %g1
	ldx [%o3], %o3	!#
	or %i5, %g3, %l0
	or %g1, %o1, %o1
	lduw [%l2+108], %o4	!+
	xor %i0, %l0, %l4
	and %l1, %o1, %l3
	ldx [%l5+128], %o0	!#
	xor %l3, %g4, %l3
	xor %g4, %g3, %i2
	ldx [%o4], %o4	!#
	andn %l3, %g3, %g3
	or %i5, %i2, %i2
	std %f18, [%l7+24]	!#
	xor %i4, %o3, %i4
	xor %o0, %o4, %o0
	std %f22, [%l7+144]	!#
	andn %o1, %i5, %i1
	xor %g3, %i2, %g3
	std %f16, [%l7+80]	!#
	xor %l1, %i1, %o4
	or %i4, %g3, %i3
	std %f10, [%l7+224]	!#
	or %i4, %o4, %o3
	xor %l3, %i3, %i3
	ldd [%l5+200], %f20	!#
	xor %l4, %o3, %o3
	andn %i3, %o0, %g2
	ldx [%l6+16], %i5	!#
	xor %o3, %g2, %g2
	and %g4, %l0, %l0
	lduw [%l2+184], %o3	!+
	xor %i5, %g2, %i5
	stx %i5, [%l7+16]	!#
	xor %o5, %o4, %o2
	xor %l0, %g3, %l0
	and %i0, %o2, %i0
	ldd [%l5+224], %f10	!#
	fxor %f20, %f26, %f20
	andn %i4, %i0, %i0
	or %g4, %o2, %i5
	ldx [%o3], %o3	!#
	fand %f0, %f8, %f22
	xor %l0, %i0, %i0
	andn %i5, %o0, %o5
	ldx [%l6+104], %o7	!#
	fxor %f22, %f28, %f22
	xor %i0, %o5, %o5
	andn %i1, %g4, %i1
	lduw [%l2+176], %l0	!+
	fand %f8, %f22, %f4
	xor %o7, %o5, %o7
	stx %o7, [%l7+104]	!#
	xor %i1, %l1, %i1
	fxnor %f4, %f0, %f2
	and %g3, %l4, %g3
	and %i4, %i1, %l4
	ldx [%l0], %l0	!#
	for %f0, %f8, %f16
	andn %o1, %g3, %o1
	xor %o4, %g3, %g3
	lduw [%l2+188], %o4	!+
	for %f16, %f28, %f16
	xor %o1, %l4, %l4
	or %l3, %i2, %i2
	lduw [%l2+192], %l3	!+
	fandnot2 %f28, %f0, %f24
	or %i4, %g3, %o7
	andn %i3, %o1, %o1
	ldx [%o4], %o4	!#
	for %f20, %f24, %f24
	xor %i2, %o7, %o7
	or %i0, %o1, %i0
	ldx [%l3], %l3	!#
	fxor %f16, %f24, %f24
	or %o0, %o7, %o7
	xor %o1, %o2, %o1
	ldx [%l6+192], %o2	!#
	fxor %f10, %f6, %f10
	xnor %l4, %o7, %o7
	xor %i0, %g1, %i0
	ldx [%l5+232], %l4	!#
	fand %f20, %f2, %f30
	xor %o2, %o7, %o2
	stx %o2, [%l7+192]	!#
	xor %g2, %g3, %g2
	fxnor %f8, %f2, %f2
	and %g1, %g3, %g3
	andn %i5, %g2, %g2
	ldx [%l5+216], %g1	!#
	fandnot2 %f8, %f20, %f8
	xor %g3, %i1, %g3
	and %i4, %g2, %g2
	ldx [%l5+240], %g4	!#
	fxnor %f20, %f30, %f6
	andn %i4, %i0, %i0
	xor %g3, %g2, %g2
	ldx [%l5+248], %i4	!#
	fxor %f22, %f30, %f30
	xor %o1, %i0, %i0
	or %o0, %g2, %g2
	ldx [%l6+56], %o1	!#
	fxor %f4, %f30, %f18
	xor %l4, %o3, %l1
	xor %i0, %g2, %g2
	ldd [%l5+184], %f16	!#
	for %f10, %f18, %f12
	xor %o1, %g2, %o1
	stx %o1, [%l7+56]	!#
	xor %g1, %l0, %o5
	fand %f10, %f6, %f26
	xor %g4, %o4, %g4
	xor %i4, %l3, %i4
				!# end of included file
	lduw [%l2+180], %o2	!+
	xor %l1, %o5, %l3
	andn %o5, %l1, %o7
	ldx [%l5+224], %o0	!#
	xor %o7, %g4, %o4
	and %i4, %o7, %g2
	ldx [%o2], %o2	!#
	or %i4, %o4, %i3
	andn %l3, %i4, %l0
	lduw [%l2+196], %o3	!+
	xor %l3, %i3, %i3
	xor %o0, %o2, %o0
	ldx [%l5], %i5	!#
	andn %i3, %o5, %g3
	or %i3, %i4, %o2
	ldx [%o3], %o3	!#
	or %g3, %g4, %o1
	xor %g3, %l1, %g1
	lduw [%l2+216], %l5	!+
	or %o0, %g3, %g3
	xor %i5, %o3, %i5
	lduw [%l2+212], %l2	!+
	xor %o1, %l3, %o1
	andn %g1, %i4, %l4
	xor %g4, %l4, %l4
	xor %o1, %i4, %o1
	lduw [%l2+32], %i1	!+
	andn %o0, %l4, %o3
	andn %o4, %o1, %i2
	lduw [%l2+52], %l3	!+
	xor %i3, %o3, %o3
	xor %i2, %g1, %i0
	ldx [%i1], %i1	!#
	andn %o0, %i0, %i2
	xor %o2, %o4, %o4
	ldx [%l3], %l3	!#
	xor %o1, %i2, %i2
	and %o5, %o2, %o2
	lduw [%l2+48], %g1	!+
	or %i5, %i2, %i2
	andn %o3, %g4, %o5
	ldx [%l6+32], %i3	!#
	xnor %o3, %i2, %i2
	xor %g2, %o5, %g2
	ldx [%g1], %g1	!#
	xor %i3, %i2, %i3
	stx %i3, [%l7+32]	!#
	and %o0, %g2, %g2
	andn %o0, %o5, %o5
	xor %o2, %g2, %g2
	lduw [%l2+36], %o2	!+
	xor %o4, %o5, %o5
	andn %g2, %i5, %g2
	ldx [%l6+112], %i3	!#
	xor %o5, %g2, %g2
	andn %l1, %i0, %i0
	ldx [%o2], %o2	!#
	xor %i3, %g2, %i3
	stx %i3, [%l7+112]	!#
	or %l4, %i0, %i0
	andn %o7, %o1, %o1
	or %o4, %g2, %g2
	ldx [%l5+24], %l4	!#
	xor %l1, %i4, %o7
	andn %o0, %g2, %g2
	ldx [%l5+64], %o4	!#
	xor %o1, %g2, %g2
	xor %o7, %o1, %o7
	ldx [%l5+56], %i4	!#
	or %i5, %g2, %g2
	andn %o7, %o0, %o7
	ldx [%l5+32], %o0	!#
	xor %i0, %g3, %g3
	or %l0, %g4, %l0
	ldx [%l6+208], %o1	!#
	xnor %g3, %g2, %g2
	xor %l0, %o7, %o7
	ldx [%l6+160], %l0	!#
	xor %o1, %g2, %o1
	stx %o1, [%l7+208]	!#
	and %i5, %o7, %o7
	xor %l4, %i1, %o5
	xnor %o3, %o7, %o7
	lduw [%l2+204], %l6	!+
	xor %l0, %o7, %l0
	stx %l0, [%l7+160]	!#
	xor %o4, %l3, %i5
	xor %i4, %g1, %i4
	xor %o0, %o2, %o0
				!# end of included file
	b .LL66
	ld [%l2+208],%l7
.LL62:
	lduw [%l2+32], %g2	!+
	ldx [%l5+24], %i3	!#
	ldx [%g2], %i2	!#
	lduw [%l2+52], %g2	!+
	ldx [%l5+64], %i1	!#
	ldx [%g2], %i0	!#
	lduw [%l2+48], %g2	!+
	ldx [%l5+56], %i4	!#
	ldx [%g2], %g3	!#
	lduw [%l2+36], %g2	!+
	ldx [%l5+32], %o0	!#
	ldx [%g2], %g2	!#
	xor %i3, %i2, %o5
	xor %i1, %i0, %i5
	xor %i4, %g3, %i4
	xor %o0, %g2, %o0
.LL66:
	lduw [%l2+40], %o4	!+
	xor %o5, %i5, %l0
	and %i5, %i4, %i1
	ldx [%l5+40], %l1	!#
	xor %l0, %i4, %g1
	andn %o5, %i1, %o3
	ldx [%o4], %o4	!#
	andn %o0, %o3, %o1
	andn %i4, %o3, %i0
	lduw [%l2+44], %o2	!+
	xor %g1, %o1, %g3
	or %i1, %o1, %o1
	ldx [%l5+48], %g4	!#
	andn %o1, %l0, %o1
	or %i0, %o0, %l3
	ldx [%o2], %o2	!#
	xor %l1, %o4, %l1
	andn %i5, %o3, %o4
	ldd [%l5+184], %f12	!#
	or %l1, %o1, %o7
	xor %g4, %o2, %g4
	ldd [%l5+152], %f26	!#
	xor %g3, %o7, %o7
	and %g4, %l3, %i3
	ldx [%l6+96], %o2	!#
	xnor %o7, %i3, %i3
	xor %g3, %i0, %i0
	lduw [%l2+60], %l0	!+
	xor %o2, %i3, %o2
	stx %o2, [%l7+96]	!#
	xor %o3, %i3, %i3
	andn %i3, %o0, %i3
	and %o0, %i0, %i0
	ldx [%l0], %l0	!#
	xor %g1, %i3, %i3
	xor %o4, %i0, %o3
	ldd [%l5+192], %f16	!#
	and %l1, %o3, %o3
	or %g3, %o5, %g3
	lduw [%l2+64], %o2	!+
	xor %i3, %o3, %o5
	xor %i4, %o0, %i3
	lduw [%l2+76], %o3	!+
	andn %i3, %o1, %o1
	xor %g3, %o0, %g3
	ldx [%o2], %o2	!#
	andn %l1, %g3, %i2
	or %o4, %o1, %o4
	ldx [%o3], %o3	!#
	xor %o1, %i2, %i2
	or %g3, %i0, %g3
	lduw [%l2+72], %o1	!+
	or %g4, %i2, %g2
	xor %o4, %o7, %o4
	ldx [%l6+8], %o7	!#
	xor %o5, %g2, %g2
	and %i3, %g3, %i3
	ldx [%o1], %o1	!#
	xor %o7, %g2, %o7
	stx %o7, [%l7+8]	!#
	andn %l3, %i3, %i3
	and %l1, %g3, %g3
	or %g4, %i3, %i3
	ldx [%l5+64], %l3	!#
	xor %o4, %g3, %g3
	xor %o0, %g1, %g2
	ldx [%l6+136], %o7	!#
	xnor %g3, %i3, %i3
	andn %g2, %o4, %g1
	ldx [%l5+72], %o4	!#
	xor %o7, %i3, %o7
	stx %o7, [%l7+136]	!#
	xor %g1, %i2, %g1
	or %i1, %i0, %i0
	and %l1, %i1, %i1
	ldx [%l5+96], %i5	!#
	andn %l1, %g1, %g1
	xor %i0, %i1, %i1
	ldx [%l5+88], %i4	!#
	xor %g2, %g1, %g1
	andn %g4, %i1, %i1
	ldx [%l6+216], %g2	!#
	xor %l3, %l0, %o0
	xnor %g1, %i1, %i1
	ldd [%l5+176], %f14	!#
	xor %g2, %i1, %g2
	stx %g2, [%l7+216]	!#
	xor %o4, %o2, %l1
	xor %i5, %o3, %i5
	xor %i4, %o1, %i4
				!# end of included file
	lduw [%l2+144], %g3	!#
	lduw [%l2+128], %g2	!#
	ldd [%g3], %f22	!#
	ldd [%g2], %f18	!#
	lduw [%l2+148], %g3	!#
	fxor %f12, %f22, %f12
	lduw [%l2+140], %g2	!#
	fxor %f26, %f18, %f26
	ldd [%g3], %f8	!#
	ldd [%g2], %f6	!#
	ldd [%l5+168], %f30	!#
	fxor %f16, %f8, %f16
	lduw [%l2+56], %o3	!+
	fxor %f14, %f6, %f14
	xor %o0, %l1, %g3
	and %l1, %i5, %g2
	ldx [%l5+56], %o5	!#
	fxor %f12, %f26, %f0
	xor %g3, %i5, %g3
	and %l1, %i4, %o4
	ldx [%o3], %o3	!#
	fand %f26, %f16, %f26
	and %o0, %g3, %l3
	xor %o4, %g3, %o4
	lduw [%l2+68], %i1	!+
	fandnot2 %f26, %f12, %f10
	or %i4, %l3, %o7
	andn %i5, %l3, %i0
	ldx [%l5+80], %g4	!#
	fxor %f16, %f26, %f8
	xor %g3, %o7, %l4
	xor %i0, %i4, %o2
	ldx [%i1], %i1	!#
	fxor %f0, %f16, %f20
	xor %o5, %o3, %o5
	xor %l1, %l3, %o3
	lduw [%l2+132], %o1	!#
	for %f10, %f8, %f4
	and %o5, %o2, %i2
	xor %g4, %i1, %g4
	lduw [%l2+136], %g1	!#
	fandnot2 %f14, %f10, %f24
	andn %o3, %i4, %i3
	xor %i4, %i2, %i2
	ldd [%o1], %f2	!#
	fandnot2 %f4, %f14, %f4
	or %o5, %i3, %i1
	or %g4, %i2, %i2
	ldd [%g1], %f6	!#
	fxor %f20, %f24, %f22
	andn %g3, %i3, %g3
	xor %l4, %i1, %i1
	ldx [%l6+40], %l0	!#
	fxor %f8, %f4, %f4
	xor %i1, %i2, %i2
	xor %g3, %g2, %i1
	ldd [%l5+160], %f28	!#
	for %f16, %f22, %f18
	or %o2, %g3, %o2
	and %i5, %i2, %o1
	lduw [%l2+80], %g1	!+
	xor %l0, %i2, %l0
	stx %l0, [%l7+40]	!#
	xor %o1, %o3, %o1
	for %f10, %f4, %f10
	or %g2, %l3, %g2
	or %o5, %i1, %i1
	ldx [%g1], %g1	!#
	fxor %f28, %f2, %f28
	xor %o2, %g2, %o2
	xor %g2, %i4, %l0
	lduw [%l2+96], %l3	!+
	fxor %f30, %f6, %f30
	or %o5, %o2, %i4
	or %o0, %i3, %i2
	lduw [%l2+88], %o3	!+
	fandnot2 %f18, %f12, %f18
	xor %l0, %i1, %i1
	xor %i2, %o7, %o7
	ldx [%l3], %l3	!#
	fandnot2 %f28, %f10, %f2
	xor %o7, %i4, %i4
	ldx [%o3], %o3	!#
	fand %f28, %f4, %f6
	andn %g4, %i4, %i4
	andn %i3, %l1, %i3
	ldx [%l6+232], %o7	!#
	fxor %f22, %f6, %f6
	xnor %i1, %i4, %i4
	xor %i5, %g2, %g2
	lduw [%l2+84], %i1	!+
	fxor %f18, %f2, %f2
	ldd [%l6+24], %f18	!#
	fandnot2 %f2, %f30, %f2
	xor %o7, %i4, %o7
	stx %o7, [%l7+232]	!#
	andn %g2, %l4, %g2
	fxnor %f6, %f2, %f2
	or %i0, %o2, %o2
	and %o5, %g2, %g2
	ldx [%i1], %i1	!#
	fandnot2 %f2, %f0, %f6
	or %o5, %i3, %i3
	xor %o2, %g2, %g2
	ldx [%l5+88], %o2	!#
	fxor %f6, %f10, %f6
	xor %o4, %i3, %i3
	and %g4, %g2, %g2
	ldx [%l6+120], %o4	!#
	fxor %f18, %f2, %f18
	or %o0, %g3, %g3
	xor %i3, %g2, %g2
	ldx [%l5+104], %i3	!#
	fxor %f26, %f22, %f26
	xor %o4, %g2, %o4
	stx %o4, [%l7+120]	!#
	andn %g2, %i2, %g2
	fandnot2 %f26, %f4, %f26
	andn %o1, %o5, %o1
	or %o5, %g2, %g2
	ldx [%l5+120], %i4	!#
	fandnot2 %f16, %f6, %f10
	xor %g3, %l0, %g3
	xor %o1, %g2, %g2
	ldx [%l5+96], %o0	!#
	fxor %f10, %f22, %f10
	xor %g3, %o1, %o1
	and %g4, %g2, %g2
	ldx [%l6+184], %l4	!#
	fand %f12, %f8, %f8
	xor %o2, %g1, %o5
	xnor %o1, %g2, %g2
	ldd [%l6+144], %f22	!#
	fandnot2 %f14, %f8, %f8
	xor %l4, %g2, %l4
	stx %l4, [%l7+184]	!#
	xor %i3, %o3, %l1
	for %f12, %f16, %f14
	xor %i4, %l3, %i4
	xor %o0, %i1, %o0
				!# end of included file
	lduw [%l2+92], %o1	!+
	fandnot2 %f14, %f0, %f0
	or %o5, %l1, %o7
	or %o0, %l1, %l3
	ldx [%l5+112], %g4	!#
	fand %f14, %f26, %f14
	and %i4, %o7, %o7
	andn %l1, %o5, %o3
	ldx [%o1], %o1	!#
	fandnot2 %f28, %f10, %f4
	xor %o5, %o7, %i3
	xor %l1, %o7, %o7
	lduw [%l2+100], %l0	!+
	fandnot2 %f28, %f4, %f2
	xor %i3, %l3, %l3
	or %o3, %i3, %o3
	ldx [%l5+128], %i5	!#
	fxor %f0, %f2, %f2
	andn %o0, %o7, %o2
	xor %l1, %i4, %o4
	ldx [%l0], %l0	!#
	fxor %f6, %f4, %f4
	xor %o3, %o2, %o2
	or %o7, %i3, %o7
	ldx [%l5+136], %i2	!#
	fandnot2 %f6, %f12, %f6
	xor %g4, %o1, %g4
	xor %i5, %l0, %i5
	lduw [%l2+116], %l0	!+
	fandnot2 %f30, %f2, %f2
	and %o0, %o3, %o3
	andn %o4, %o0, %o4
	lduw [%l2+112], %i1	!+
	fxnor %f4, %f2, %f2
	xor %l1, %o3, %i0
	xor %i4, %o3, %o3
	ldx [%l0], %l0	!#
	fxor %f22, %f2, %f22
	xor %o7, %o4, %o1
	and %g4, %o3, %g3
	ldx [%i1], %i1	!#
	fxor %f16, %f4, %f4
	or %g4, %o1, %o1
	xor %l3, %g3, %g3
	lduw [%l2+104], %l3	!+
	fandnot2 %f12, %f4, %f4
	xor %o2, %o1, %o1
	xor %i0, %o4, %i0
	lduw [%l2+124], %o4	!+
	for %f10, %f4, %f10
	or %i5, %o1, %o5
	and %o0, %o3, %o3
	ldx [%l3], %l3	!#
	for %f4, %f20, %f12
	ldd [%l6+80], %f16	!#
	fandnot2 %f28, %f4, %f4
	xor %o3, %o7, %o3
	andn %g4, %i0, %i0
	ldx [%o4], %o4	!#
	fxor %f26, %f4, %f4
	xor %g3, %o5, %o5
	xor %o3, %i0, %i0
	ldx [%l5+144], %o3	!#
	for %f30, %f6, %f6
	and %i5, %o1, %o1
	xor %g3, %i0, %o7
	ldx [%l6+200], %g2	!#
	fxnor %f4, %f6, %f6
	xnor %o1, %g3, %g3
	andn %o0, %o7, %o2
	ldx [%l6+152], %i3	!#
	fxor %f16, %f6, %f16
	xor %g2, %o5, %g2
	stx %g2, [%l7+200]	!#
	xor %o5, %o2, %o2
	for %f28, %f8, %f8
	xor %i3, %g3, %i3
	stx %i3, [%l7+152]	!#
	andn %o7, %g4, %o7
	fxor %f12, %f8, %f8
	xor %o2, %o7, %o7
	ldx [%l6+72], %i3	!#
	fxor %f10, %f24, %f10
	or %i5, %o7, %o2
	xor %o1, %o7, %o7
	ldx [%l6], %o1	!#
	fxor %f14, %f20, %f14
	xor %i2, %i1, %l1
	xnor %i0, %o2, %o2
	ldx [%l5+120], %o5	!#
	fand %f28, %f14, %f14
	xor %i3, %o2, %i3
	stx %i3, [%l7+72]	!#
	xor %o7, %o2, %o7
	fxor %f10, %f14, %f14
	ldd [%l6+224], %f10	!#
	fandnot2 %f30, %f14, %f14
	ldx [%l5+160], %i5	!#
	fxnor %f8, %f14, %f14
	xor %o1, %o7, %o1
	stx %o1, [%l7]	!#
	xor %o3, %l0, %g4
	fxor %f10, %f14, %f10
	xor %o5, %l3, %o5
	xor %i5, %o4, %i5
				!# end of included file
	ldd [%l5+192], %f0	!#
	ldd [%l5+208], %f8	!#
	ldd [%l5+216], %f28	!#
	lduw [%l2+156], %g2	!#
	lduw [%l2+164], %g3	!#
	ldd [%g2], %f2	!#
	ldd [%g3], %f4	!#
	lduw [%l2+168], %g3	!#
	fxor %f0, %f2, %f0
	lduw [%l2+160], %g2	!#
	fxor %f8, %f4, %f8
	ldd [%g3], %f12	!#
	ldd [%g2], %f26	!#
	fxor %f28, %f12, %f28
	lduw [%l2+172], %g2	!#
	lduw [%l2+152], %g3	!#
	ldd [%g2], %f6	!#
	ldd [%g3], %f14	!#
	lduw [%l2+120], %o3	!+
	andn %l1, %g4, %o1
	andn %o5, %l1, %g3
	ldx [%l5+152], %i4	!#
	xor %o1, %o5, %i0
	xor %g4, %o5, %g1
	ldx [%o3], %o3	!#
	or %i5, %g3, %l0
	or %g1, %o1, %o1
	lduw [%l2+108], %o4	!+
	xor %i0, %l0, %l4
	and %l1, %o1, %l3
	ldx [%l5+128], %o0	!#
	xor %l3, %g4, %l3
	xor %g4, %g3, %i2
	ldx [%o4], %o4	!#
	andn %l3, %g3, %g3
	or %i5, %i2, %i2
	std %f18, [%l7+24]	!#
	xor %i4, %o3, %i4
	xor %o0, %o4, %o0
	std %f22, [%l7+144]	!#
	andn %o1, %i5, %i1
	xor %g3, %i2, %g3
	std %f16, [%l7+80]	!#
	xor %l1, %i1, %o4
	or %i4, %g3, %i3
	std %f10, [%l7+224]	!#
	or %i4, %o4, %o3
	xor %l3, %i3, %i3
	ldd [%l5+200], %f20	!#
	xor %l4, %o3, %o3
	andn %i3, %o0, %g2
	ldx [%l6+16], %i5	!#
	xor %o3, %g2, %g2
	and %g4, %l0, %l0
	lduw [%l2+184], %o3	!+
	xor %i5, %g2, %i5
	stx %i5, [%l7+16]	!#
	xor %o5, %o4, %o2
	xor %l0, %g3, %l0
	and %i0, %o2, %i0
	ldd [%l5+224], %f10	!#
	fxor %f20, %f26, %f20
	andn %i4, %i0, %i0
	or %g4, %o2, %i5
	ldx [%o3], %o3	!#
	fand %f0, %f8, %f22
	xor %l0, %i0, %i0
	andn %i5, %o0, %o5
	ldx [%l6+104], %o7	!#
	fxor %f22, %f28, %f22
	xor %i0, %o5, %o5
	andn %i1, %g4, %i1
	lduw [%l2+176], %l0	!+
	fand %f8, %f22, %f4
	xor %o7, %o5, %o7
	stx %o7, [%l7+104]	!#
	xor %i1, %l1, %i1
	fxnor %f4, %f0, %f2
	and %g3, %l4, %g3
	and %i4, %i1, %l4
	ldx [%l0], %l0	!#
	for %f0, %f8, %f16
	andn %o1, %g3, %o1
	xor %o4, %g3, %g3
	lduw [%l2+188], %o4	!+
	for %f16, %f28, %f16
	xor %o1, %l4, %l4
	or %l3, %i2, %i2
	lduw [%l2+192], %l3	!+
	fandnot2 %f28, %f0, %f24
	or %i4, %g3, %o7
	andn %i3, %o1, %o1
	ldx [%o4], %o4	!#
	for %f20, %f24, %f24
	xor %i2, %o7, %o7
	or %i0, %o1, %i0
	ldx [%l3], %l3	!#
	fxor %f16, %f24, %f24
	or %o0, %o7, %o7
	xor %o1, %o2, %o1
	ldx [%l6+192], %o2	!#
	fxor %f10, %f6, %f10
	xnor %l4, %o7, %o7
	xor %i0, %g1, %i0
	ldx [%l5+232], %l4	!#
	fand %f20, %f2, %f30
	xor %o2, %o7, %o2
	stx %o2, [%l7+192]	!#
	xor %g2, %g3, %g2
	fxnor %f8, %f2, %f2
	and %g1, %g3, %g3
	andn %i5, %g2, %g2
	ldx [%l5+216], %g1	!#
	fandnot2 %f8, %f20, %f8
	xor %g3, %i1, %g3
	and %i4, %g2, %g2
	ldx [%l5+240], %g4	!#
	fxnor %f20, %f30, %f6
	andn %i4, %i0, %i0
	xor %g3, %g2, %g2
	ldx [%l5+248], %i4	!#
	fxor %f22, %f30, %f30
	xor %o1, %i0, %i0
	or %o0, %g2, %g2
	ldx [%l6+56], %o1	!#
	fxor %f4, %f30, %f18
	xor %l4, %o3, %l1
	xor %i0, %g2, %g2
	ldd [%l5+184], %f16	!#
	for %f10, %f18, %f12
	xor %o1, %g2, %o1
	stx %o1, [%l7+56]	!#
	xor %g1, %l0, %o5
	fand %f10, %f6, %f26
	xor %g4, %o4, %g4
	xor %i4, %l3, %i4
				!# end of included file
	lduw [%l2+180], %o2	!+
	fxor %f30, %f26, %f26
	xor %l1, %o5, %l3
	andn %o5, %l1, %o7
	ldx [%l5+224], %o0	!#
	for %f28, %f8, %f28
	xor %o7, %g4, %o4
	and %i4, %o7, %g2
	ldx [%o2], %o2	!#
	fandnot2 %f0, %f8, %f8
	or %i4, %o4, %i3
	andn %l3, %i4, %l0
	lduw [%l2+196], %o3	!+
	fand %f10, %f8, %f8
	xor %l3, %i3, %i3
	xor %o0, %o2, %o0
	ldx [%l5], %i5	!#
	fxor %f24, %f12, %f12
	andn %i3, %o5, %i0
	or %i3, %i4, %o2
	ldx [%o3], %o3	!#
	fxor %f0, %f18, %f18
	or %i0, %g4, %o1
	xor %i0, %l1, %g1
	ldd [%l6+248], %f30	!#
	fxor %f16, %f14, %f16
	or %o0, %i0, %i0
	xor %i5, %o3, %i5
	ldd [%l6+88], %f14	!#
	fand %f16, %f12, %f12
	xor %o1, %l3, %o1
	andn %g1, %i4, %l4
	ldd [%l6+168], %f22	!#
	fxor %f26, %f12, %f12
	xor %g4, %l4, %l4
	xor %o1, %i4, %o1
	lduw [%l2+16], %i1	!+
	fxnor %f26, %f8, %f26
	andn %o0, %l4, %o3
	andn %o4, %o1, %i2
	lduw [%l2+24], %l3	!+
	fand %f20, %f8, %f8
	xor %i3, %o3, %o3
	xor %i2, %g1, %g3
	ldx [%i1], %i1	!#
	fxor %f30, %f12, %f30
	andn %o0, %g3, %i2
	xor %o2, %o4, %o4
	ldx [%l3], %l3	!#
	for %f0, %f2, %f24
	xor %o1, %i2, %i2
	and %o5, %o2, %o2
	lduw [%l2+20], %g1	!+
	fxnor %f24, %f12, %f24
	or %i5, %i2, %i2
	andn %o3, %g4, %o5
	ldx [%l6+32], %i3	!#
	for %f10, %f24, %f24
	xnor %o3, %i2, %i2
	xor %g2, %o5, %g2
	ldx [%g1], %g1	!#
	fxor %f20, %f4, %f12
	xor %i3, %i2, %i3
	stx %i3, [%l7+32]	!#
	and %o0, %g2, %g2
	fand %f12, %f0, %f12
	andn %o0, %o5, %o5
	xor %o2, %g2, %g2
	lduw [%l2+28], %o2	!+
	fandnot2 %f10, %f12, %f12
	xor %o4, %o5, %o5
	andn %g2, %i5, %g2
	ldx [%l6+112], %i3	!#
	for %f20, %f4, %f4
	xor %o5, %g2, %g2
	andn %l1, %g3, %g3
	ldx [%o2], %o2	!#
	fxor %f2, %f4, %f4
	xor %i3, %g2, %i3
	stx %i3, [%l7+112]	!#
	or %l4, %g3, %g3
	fandnot2 %f4, %f20, %f20
	andn %o7, %o1, %o1
	or %o4, %g2, %g2
	ldx [%l5+8], %l4	!#
	fandnot2 %f2, %f18, %f18
	xor %l1, %i4, %o7
	andn %o0, %g2, %g2
	ldx [%l5+24], %o4	!#
	fand %f10, %f18, %f18
	xor %o1, %g2, %g2
	xor %o7, %o1, %o7
	ldd [%l6+48], %f10	!#
	fxor %f4, %f12, %f12
	or %i5, %g2, %g2
	andn %o7, %o0, %o7
	ldx [%l6+208], %o1	!#
	fxnor %f6, %f12, %f6
	xor %g3, %i0, %i0
	or %l0, %g4, %l0
	ldx [%l5+16], %g4	!#
	for %f16, %f12, %f4
	xnor %i0, %g2, %g2
	xor %l0, %o7, %o7
	ldx [%l6+160], %l0	!#
	fxor %f26, %f4, %f4
	xor %o1, %g2, %o1
	stx %o1, [%l7+208]	!#
	and %i5, %o7, %o7
	fxor %f14, %f4, %f14
	xor %l4, %i1, %l1
	xnor %o3, %o7, %o7
	ldx [%l5+32], %i5	!#
	fxor %f28, %f6, %f28
	xor %l0, %o7, %l0
	stx %l0, [%l7+160]	!#
	xor %o4, %l3, %i4
	fxor %f6, %f24, %f24
	xor %g4, %g1, %g4
	xor %i5, %o2, %i5
				!# end of included file
	lduw [%l2+12], %o3	!+
	fxor %f8, %f18, %f8
	andn %l1, %i4, %g2
	andn %l1, %g4, %l0
	ldx [%l5], %o0	!#
	for %f20, %f12, %f20
	or %l0, %i4, %i0
	xor %l1, %g4, %g1
	ldx [%o3], %o3	!#
	fornot2 %f16, %f20, %f20
	xor %g2, %g4, %o2
	and %i5, %i0, %l4
	lduw [%l2+8], %l3	!+
	fxor %f24, %f20, %f20
	xor %o2, %l4, %o4
	xor %i4, %l4, %i3
	ldx [%l5+248], %o5	!#
	fxor %f22, %f20, %f22
	and %i3, %g1, %i3
	andn %i4, %g4, %o7
	ldx [%l3], %l3	!#
	fxor %f28, %f18, %f18
	andn %g4, %i4, %o1
	xor %l0, %i3, %i2
	std %f30, [%l7+248]	!#
	for %f16, %f8, %f8
	xor %o0, %o3, %o0
	or %o4, %o1, %o3
	std %f14, [%l7+88]	!#
	fxor %f18, %f8, %f8
	xor %o5, %l3, %o5
	or %i5, %i2, %i2
	std %f22, [%l7+168]	!#
	fxor %f10, %f8, %f10
	andn %i5, %g1, %l3
	xor %o7, %i2, %i2
	std %f10, [%l7+48]	!#
	xor %o1, %l3, %o1
	or %o0, %i2, %o7
	lduw [%l2+216], %l5	!+
	or %o0, %o1, %o1
	xor %i3, %o7, %o7
	lduw [%l2+212], %l2	!+
	xor %o4, %o1, %o1
	and %o5, %o7, %o7
	ldx [%l6+128], %o4	!#
	xnor %o1, %o7, %o7
	or %g2, %l4, %g2
	lduw [%l2+32], %o1	!+
	xor %o4, %o7, %o4
	stx %o4, [%l7+128]	!#
	andn %i2, %o2, %o2
	xor %g2, %g1, %g2
	andn %o0, %o2, %g1
	ldx [%o1], %o1	!#
	xor %g2, %g1, %g1
	xor %l3, %g2, %g3
	lduw [%l2+52], %o7	!+
	andn %i2, %g3, %g3
	xor %o3, %o2, %o2
	lduw [%l2+48], %o4	!+
	and %o0, %g3, %i3
	and %l1, %o3, %o3
	ldx [%o7], %o7	!#
	xor %o2, %i3, %i3
	or %l1, %g3, %g3
	ldx [%o4], %o4	!#
	and %o5, %i3, %g4
	andn %i2, %o3, %o2
	ldx [%l6+240], %i1	!#
	xor %g1, %g4, %g4
	andn %g2, %o2, %g2
	lduw [%l2+36], %g1	!+
	xor %i1, %g4, %i1
	stx %i1, [%l7+240]	!#
	or %g2, %l0, %g2
	andn %g2, %o0, %i1
	or %o0, %l0, %l0
	ldx [%g1], %g1	!#
	xor %g3, %i1, %i1
	xor %o2, %l0, %l0
	ldx [%l5+24], %o2	!#
	or %g2, %i2, %g2
	andn %o5, %i1, %i1
	ldx [%l6+64], %i2	!#
	xnor %l0, %i1, %i1
	xor %g2, %i4, %g2
	ldx [%l5+56], %i4	!#
	xor %i2, %i1, %i2
	stx %i2, [%l7+64]	!#
	xor %i0, %o3, %o3
	andn %o3, %l4, %o3
	andn %o0, %g2, %g2
	ldx [%l6+176], %l4	!#
	andn %i3, %l3, %i3
	xor %o3, %g2, %g2
	lduw [%l2+204], %l6	!+
	xor %i3, %l0, %i3
	or %o5, %g2, %g2
	ldx [%l5+64], %l0	!#
	xor %o2, %o1, %o5
	xnor %i3, %g2, %g2
	ldx [%l5+32], %o0	!#
	xor %l4, %g2, %l4
	stx %l4, [%l7+176]	!#
	xor %l0, %o7, %i5
	xor %i4, %o4, %i4
	xor %o0, %g1, %o0
				!# end of included file
	cmp %l6,0
	bne .LL66
	ld [%l2+208],%l7
	srlx %i7, 32, %o0	!##
	!#EPILOGUE#
	ld [%sp+96],%i7
	ld [%sp+100],%o7
	ldd [%sp+104],%l0
	ldd [%sp+112],%l2
	ldd [%sp+120],%l4
	ldd [%sp+128],%l6
	ldd [%sp+136],%i0
	ldd [%sp+144],%i2
	ldd [%sp+152],%i4
	retl
	add %sp,184,%sp
.LLfe13:
	.size	 asm_do_all_fancy,.LLfe13-asm_do_all_fancy
	.ident	"GCC: (GNU) 2.7.2.2"
