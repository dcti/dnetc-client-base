	.file	"r72-kks2.cpp"
gcc2_compiled.:
	.global .umul
.section	".text"
	.align 4
	.global rc5_72_unit_func_KKS_2
	.type	 rc5_72_unit_func_KKS_2,#function
	.proc	04
rc5_72_unit_func_KKS_2:
.LLFB1:
	!#PROLOGUE# 0
	save	%sp, -544, %sp
.LLCFI0:
	!#PROLOGUE# 1
	sethi	%hi(-1640531968), %g2
	sethi	%hi(-1209970688), %g3
	or	%g2, 441, %g2
	or	%g3, 355, %g3
	mov	%g2, %i2
	add	%g3, %g2, %g3
	mov	1, %g2
	mov	%i0, %l5
	st	%i1, [%fp+72]
	st	%g2, [%fp-436]
	mov	32, %l4
	add	%fp, -224, %i1
	add	%fp, -432, %i0
	mov	4, %g2
.LL6:
	st	%g3, [%g2+%i0]
	st	%g3, [%g2+%i1]
	ld	[%fp-436], %i3
	add	%g3, %i2, %g3
	add	%i3, 1, %i3
	st	%i3, [%fp-436]
	cmp	%i3, 25
	bleu	.LL6
	add	%g2, 4, %g2
	sethi	%hi(-1089828864), %i1
	ld	[%l5+24],%i0
	mov	%i1, %g2
	or	%g2, 797, %g2
	mov	%i1, %g3
	add	%i0, %g2, %i0
	or	%g3, 797, %g3
	mov	32, %i2
	sub	%i2, %g3, %g2
	srl	%i0, %g2, %g2
	sll	%i0, %g3, %i0
	ld	[%l5+20], %g3
	or	%i0, %g2, %i0
	st	%i0, [%fp-444]
	sethi	%hi(-1089828864), %i3
	ld	[%fp-220], %i1
	or	%i3, 797, %i3
	add	%i1, %i3, %i1
	ld	[%fp-444], %i3
	add	%i1, %i3, %i1
	srl	%i1, 29, %i3
	sll	%i1, 3, %l6
	ld	[%fp-444], %g2
	or	%l6, %i3, %l6
	add	%l6, %g2, %i1
	add	%g3, %i1, %g3
	ld	[%fp+72], %i0
	sub	%i2, %i1, %i2
	ld	[%fp-216], %g2
	srl	%g3, %i2, %i2
	sll	%g3, %i1, %l7
	ld	[%i0], %g3
	or	%l7, %i2, %l7
	add	%g2, %l6, %g2
	add	%g2, %l7, %g2
	srl	%g2, 29, %i0
	sll	%g2, 3, %g2
	srl	%g3, 1, %g3
	or	%g2, %i0, %g2
	std	%g2, [%fp-440]
.LL25:
	ld	[%l5+16], %i0
	sethi	%hi(-1089828864), %i1
	add	%i0, 1, %o3
	or	%i1, 797, %i1
	ld	[%fp-440], %i3
	st	%i1, [%fp-328]
	st	%i1, [%fp-120]
	add %i3, %l7, %i1
	add %i0, %i1, %i5
	st  %i3, [%fp-112]
	add %i3, %l7, %g2
	sub %l4, %i1, %g3
	ld	[%fp-212], %o0
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-440], %g3
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	st	%i3, [%fp-320]
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	st	%l6, [%fp-324]
	srl %o7, %i4, %o7
	add %g3, %o0, %i2
	st	%l6, [%fp-116]
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	ld	[%fp-444], %i0
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %i0, %i1, %i5
	st	%i2, [%fp-108]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-316]
	sll %i5, %i1, %l0
	add %i0, %g2, %o7
	ld	[%fp-208], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %l7, %i1, %i5
	st	%i2, [%fp-104]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-312]
	sll %i5, %i1, %o5
	add %l7, %g2, %o7
	ld	[%fp-204], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-100]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-308]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-200], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-96]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-304]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-196], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-92]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-300]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-192], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-88]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-296]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-188], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-84]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-292]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-184], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-80]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-288]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-180], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-76]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-284]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-176], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-72]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-280]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-172], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-68]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-276]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-168], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-64]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-272]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-164], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-60]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-268]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-160], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-56]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-264]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-156], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-52]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-260]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-152], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-48]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-256]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-148], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-44]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-252]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-144], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-40]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-248]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-140], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-36]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-244]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-136], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-32]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-240]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-132], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-28]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-236]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-128], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-24]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-232]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-124], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o0, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-20]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-228]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	sethi	%hi(-1089828864), %i1
	srl %o7, %i4, %o7
	or	%i1, 797, %i1
	or %o5, %i5, %o5
	add %i2, %i1, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %i1, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-120]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-328]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %l6, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %l6, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-116]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-324]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-440], %o2
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	ld	[%fp-320], %g2
	srl %o7, %i4, %o7
	add %i2, %o2, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %g2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-112]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-320]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-108], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-316], %o2
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-108]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-316]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-104], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-312], %o2
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-104]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-312]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-100], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-308], %o2
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-100]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-308]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-96], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-304], %o2
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-96]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-304]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-92], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-300], %o2
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-92]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-300]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-88], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-296], %o2
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-88]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-296]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-84], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-292], %o2
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-84]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-292]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-80], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-288], %o2
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-80]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-288]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-76], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-284], %o2
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-76]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-284]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-72], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-280], %o2
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-72]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-280]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-68], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-276], %o2
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-68]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-276]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-64], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-272], %o2
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-64]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-272]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-60], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-268], %o2
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-60]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-268]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-56], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-264], %o2
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-56]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-264]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-52], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-260], %o2
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-52]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-260]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-48], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-256], %o2
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-48]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-256]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-44], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-252], %o2
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-44]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-252]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-40], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-248], %o2
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-40]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-248]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-36], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-244], %o2
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-36]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-244]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-32], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-240], %o2
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-32]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-240]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-28], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-236], %o2
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	st	%i2, [%fp-28]
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-236]
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-24], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-232], %o2
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o5, %i1
	or %i3, %o7, %i3
	add %o4, %i1, %i5
	st	%i2, [%fp-24]
	add %i3, %l3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-232]
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	ld	[%fp-20], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-228], %o2
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %o4, %i1
	or %i3, %o7, %i3
	add %l0, %i1, %i5
	st	%i2, [%fp-20]
	add %i3, %o3, %g2
	sub %l4, %i1, %g3
	st	%i3, [%fp-228]
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	ld	[%fp-120], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-328], %o2
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add %i2, %l0, %i1
	or %i3, %o7, %i3
	add %o5, %i1, %i5
	add %i3, %l1, %g2
	sub %l4, %i1, %g3
	ld	[%l5+4], %g4
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	ld	[%fp-116], %o0
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	ld	[%fp-324], %o2
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	add	%g4, %i2, %o1
	add	%g4, %i3, %g4
	ld	[%l5], %l2
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	or %i2, %i5, %i2
	srl %i3, 29, %i3
	add	%l2, %i2, %g1
	or %i3, %o7, %i3
	add	%l2, %i3, %l2
	add %i2, %o5, %i1
	add %i3, %l3, %g2
	ld	[%fp-112], %o0
	add %o4, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-320], %o2
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	xor %o1, %g1, %o1
	srl %i3, 29, %i3
	xor %g4, %l2, %g4
	sll %o1, %g1, %i1
	sub %l4, %g1, %g3
	sll %g4, %l2, %g2
	sub %l4, %l2, %i4
	srl %o1, %g3, %o1
	or %i2, %i5, %i2
	srl %g4, %i4, %g4
	or %i3, %o7, %i3
	or %o1, %i1, %o1
	or %g4, %g2, %g4
	add %o1, %i2, %o1
	add %g4, %i3, %g4
	add %i2, %o4, %i1
	add %i3, %o3, %g2
	ld	[%fp-108], %o0
	add %l0, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-316], %o2
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	xor %g1, %o1, %g1
	srl %i3, 29, %i3
	xor %l2, %g4, %l2
	sll %g1, %o1, %i1
	sub %l4, %o1, %g3
	sll %l2, %g4, %g2
	sub %l4, %g4, %i4
	srl %g1, %g3, %g1
	or %i2, %i5, %i2
	srl %l2, %i4, %l2
	or %i3, %o7, %i3
	or %g1, %i1, %g1
	or %l2, %g2, %l2
	add %g1, %i2, %g1
	add %l2, %i3, %l2
	add %i2, %l0, %i1
	add %i3, %l1, %g2
	ld	[%fp-104], %o0
	add %o5, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-312], %o2
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	xor %o1, %g1, %o1
	srl %i3, 29, %i3
	xor %g4, %l2, %g4
	sll %o1, %g1, %i1
	sub %l4, %g1, %g3
	sll %g4, %l2, %g2
	sub %l4, %l2, %i4
	srl %o1, %g3, %o1
	or %i2, %i5, %i2
	srl %g4, %i4, %g4
	or %i3, %o7, %i3
	or %o1, %i1, %o1
	or %g4, %g2, %g4
	add %o1, %i2, %o1
	add %g4, %i3, %g4
	add %i2, %o5, %i1
	add %i3, %l3, %g2
	ld	[%fp-100], %o0
	add %o4, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-308], %o2
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	xor %g1, %o1, %g1
	srl %i3, 29, %i3
	xor %l2, %g4, %l2
	sll %g1, %o1, %i1
	sub %l4, %o1, %g3
	sll %l2, %g4, %g2
	sub %l4, %g4, %i4
	srl %g1, %g3, %g1
	or %i2, %i5, %i2
	srl %l2, %i4, %l2
	or %i3, %o7, %i3
	or %g1, %i1, %g1
	or %l2, %g2, %l2
	add %g1, %i2, %g1
	add %l2, %i3, %l2
	add %i2, %o4, %i1
	add %i3, %o3, %g2
	ld	[%fp-96], %o0
	add %l0, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-304], %o2
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	xor %o1, %g1, %o1
	srl %i3, 29, %i3
	xor %g4, %l2, %g4
	sll %o1, %g1, %i1
	sub %l4, %g1, %g3
	sll %g4, %l2, %g2
	sub %l4, %l2, %i4
	srl %o1, %g3, %o1
	or %i2, %i5, %i2
	srl %g4, %i4, %g4
	or %i3, %o7, %i3
	or %o1, %i1, %o1
	or %g4, %g2, %g4
	add %o1, %i2, %o1
	add %g4, %i3, %g4
	add %i2, %l0, %i1
	add %i3, %l1, %g2
	ld	[%fp-92], %o0
	add %o5, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-300], %o2
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	xor %g1, %o1, %g1
	srl %i3, 29, %i3
	xor %l2, %g4, %l2
	sll %g1, %o1, %i1
	sub %l4, %o1, %g3
	sll %l2, %g4, %g2
	sub %l4, %g4, %i4
	srl %g1, %g3, %g1
	or %i2, %i5, %i2
	srl %l2, %i4, %l2
	or %i3, %o7, %i3
	or %g1, %i1, %g1
	or %l2, %g2, %l2
	add %g1, %i2, %g1
	add %l2, %i3, %l2
	add %i2, %o5, %i1
	add %i3, %l3, %g2
	ld	[%fp-88], %o0
	add %o4, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-296], %o2
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	xor %o1, %g1, %o1
	srl %i3, 29, %i3
	xor %g4, %l2, %g4
	sll %o1, %g1, %i1
	sub %l4, %g1, %g3
	sll %g4, %l2, %g2
	sub %l4, %l2, %i4
	srl %o1, %g3, %o1
	or %i2, %i5, %i2
	srl %g4, %i4, %g4
	or %i3, %o7, %i3
	or %o1, %i1, %o1
	or %g4, %g2, %g4
	add %o1, %i2, %o1
	add %g4, %i3, %g4
	add %i2, %o4, %i1
	add %i3, %o3, %g2
	ld	[%fp-84], %o0
	add %l0, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-292], %o2
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	xor %g1, %o1, %g1
	srl %i3, 29, %i3
	xor %l2, %g4, %l2
	sll %g1, %o1, %i1
	sub %l4, %o1, %g3
	sll %l2, %g4, %g2
	sub %l4, %g4, %i4
	srl %g1, %g3, %g1
	or %i2, %i5, %i2
	srl %l2, %i4, %l2
	or %i3, %o7, %i3
	or %g1, %i1, %g1
	or %l2, %g2, %l2
	add %g1, %i2, %g1
	add %l2, %i3, %l2
	add %i2, %l0, %i1
	add %i3, %l1, %g2
	ld	[%fp-80], %o0
	add %o5, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-288], %o2
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	xor %o1, %g1, %o1
	srl %i3, 29, %i3
	xor %g4, %l2, %g4
	sll %o1, %g1, %i1
	sub %l4, %g1, %g3
	sll %g4, %l2, %g2
	sub %l4, %l2, %i4
	srl %o1, %g3, %o1
	or %i2, %i5, %i2
	srl %g4, %i4, %g4
	or %i3, %o7, %i3
	or %o1, %i1, %o1
	or %g4, %g2, %g4
	add %o1, %i2, %o1
	add %g4, %i3, %g4
	add %i2, %o5, %i1
	add %i3, %l3, %g2
	ld	[%fp-76], %o0
	add %o4, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-284], %o2
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	xor %g1, %o1, %g1
	srl %i3, 29, %i3
	xor %l2, %g4, %l2
	sll %g1, %o1, %i1
	sub %l4, %o1, %g3
	sll %l2, %g4, %g2
	sub %l4, %g4, %i4
	srl %g1, %g3, %g1
	or %i2, %i5, %i2
	srl %l2, %i4, %l2
	or %i3, %o7, %i3
	or %g1, %i1, %g1
	or %l2, %g2, %l2
	add %g1, %i2, %g1
	add %l2, %i3, %l2
	add %i2, %o4, %i1
	add %i3, %o3, %g2
	ld	[%fp-72], %o0
	add %l0, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-280], %o2
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	xor %o1, %g1, %o1
	srl %i3, 29, %i3
	xor %g4, %l2, %g4
	sll %o1, %g1, %i1
	sub %l4, %g1, %g3
	sll %g4, %l2, %g2
	sub %l4, %l2, %i4
	srl %o1, %g3, %o1
	or %i2, %i5, %i2
	srl %g4, %i4, %g4
	or %i3, %o7, %i3
	or %o1, %i1, %o1
	or %g4, %g2, %g4
	add %o1, %i2, %o1
	add %g4, %i3, %g4
	add %i2, %l0, %i1
	add %i3, %l1, %g2
	ld	[%fp-68], %o0
	add %o5, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-276], %o2
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	xor %g1, %o1, %g1
	srl %i3, 29, %i3
	xor %l2, %g4, %l2
	sll %g1, %o1, %i1
	sub %l4, %o1, %g3
	sll %l2, %g4, %g2
	sub %l4, %g4, %i4
	srl %g1, %g3, %g1
	or %i2, %i5, %i2
	srl %l2, %i4, %l2
	or %i3, %o7, %i3
	or %g1, %i1, %g1
	or %l2, %g2, %l2
	add %g1, %i2, %g1
	add %l2, %i3, %l2
	add %i2, %o5, %i1
	add %i3, %l3, %g2
	ld	[%fp-64], %o0
	add %o4, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-272], %o2
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	xor %o1, %g1, %o1
	srl %i3, 29, %i3
	xor %g4, %l2, %g4
	sll %o1, %g1, %i1
	sub %l4, %g1, %g3
	sll %g4, %l2, %g2
	sub %l4, %l2, %i4
	srl %o1, %g3, %o1
	or %i2, %i5, %i2
	srl %g4, %i4, %g4
	or %i3, %o7, %i3
	or %o1, %i1, %o1
	or %g4, %g2, %g4
	add %o1, %i2, %o1
	add %g4, %i3, %g4
	add %i2, %o4, %i1
	add %i3, %o3, %g2
	ld	[%fp-60], %o0
	add %l0, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-268], %o2
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	xor %g1, %o1, %g1
	srl %i3, 29, %i3
	xor %l2, %g4, %l2
	sll %g1, %o1, %i1
	sub %l4, %o1, %g3
	sll %l2, %g4, %g2
	sub %l4, %g4, %i4
	srl %g1, %g3, %g1
	or %i2, %i5, %i2
	srl %l2, %i4, %l2
	or %i3, %o7, %i3
	or %g1, %i1, %g1
	or %l2, %g2, %l2
	add %g1, %i2, %g1
	add %l2, %i3, %l2
	add %i2, %l0, %i1
	add %i3, %l1, %g2
	ld	[%fp-56], %o0
	add %o5, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-264], %o2
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	xor %o1, %g1, %o1
	srl %i3, 29, %i3
	xor %g4, %l2, %g4
	sll %o1, %g1, %i1
	sub %l4, %g1, %g3
	sll %g4, %l2, %g2
	sub %l4, %l2, %i4
	srl %o1, %g3, %o1
	or %i2, %i5, %i2
	srl %g4, %i4, %g4
	or %i3, %o7, %i3
	or %o1, %i1, %o1
	or %g4, %g2, %g4
	add %o1, %i2, %o1
	add %g4, %i3, %g4
	add %i2, %o5, %i1
	add %i3, %l3, %g2
	ld	[%fp-52], %o0
	add %o4, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-260], %o2
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	xor %g1, %o1, %g1
	srl %i3, 29, %i3
	xor %l2, %g4, %l2
	sll %g1, %o1, %i1
	sub %l4, %o1, %g3
	sll %l2, %g4, %g2
	sub %l4, %g4, %i4
	srl %g1, %g3, %g1
	or %i2, %i5, %i2
	srl %l2, %i4, %l2
	or %i3, %o7, %i3
	or %g1, %i1, %g1
	or %l2, %g2, %l2
	add %g1, %i2, %g1
	add %l2, %i3, %l2
	add %i2, %o4, %i1
	add %i3, %o3, %g2
	ld	[%fp-48], %o0
	add %l0, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-256], %o2
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	xor %o1, %g1, %o1
	srl %i3, 29, %i3
	xor %g4, %l2, %g4
	sll %o1, %g1, %i1
	sub %l4, %g1, %g3
	sll %g4, %l2, %g2
	sub %l4, %l2, %i4
	srl %o1, %g3, %o1
	or %i2, %i5, %i2
	srl %g4, %i4, %g4
	or %i3, %o7, %i3
	or %o1, %i1, %o1
	or %g4, %g2, %g4
	add %o1, %i2, %o1
	add %g4, %i3, %g4
	add %i2, %l0, %i1
	add %i3, %l1, %g2
	ld	[%fp-44], %o0
	add %o5, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-252], %o2
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	xor %g1, %o1, %g1
	srl %i3, 29, %i3
	xor %l2, %g4, %l2
	sll %g1, %o1, %i1
	sub %l4, %o1, %g3
	sll %l2, %g4, %g2
	sub %l4, %g4, %i4
	srl %g1, %g3, %g1
	or %i2, %i5, %i2
	srl %l2, %i4, %l2
	or %i3, %o7, %i3
	or %g1, %i1, %g1
	or %l2, %g2, %l2
	add %g1, %i2, %g1
	add %l2, %i3, %l2
	add %i2, %o5, %i1
	add %i3, %l3, %g2
	ld	[%fp-40], %o0
	add %o4, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-248], %o2
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	xor %o1, %g1, %o1
	srl %i3, 29, %i3
	xor %g4, %l2, %g4
	sll %o1, %g1, %i1
	sub %l4, %g1, %g3
	sll %g4, %l2, %g2
	sub %l4, %l2, %i4
	srl %o1, %g3, %o1
	or %i2, %i5, %i2
	srl %g4, %i4, %g4
	or %i3, %o7, %i3
	or %o1, %i1, %o1
	or %g4, %g2, %g4
	add %o1, %i2, %o1
	add %g4, %i3, %g4
	add %i2, %o4, %i1
	add %i3, %o3, %g2
	ld	[%fp-36], %o0
	add %l0, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-244], %o2
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	xor %g1, %o1, %g1
	srl %i3, 29, %i3
	xor %l2, %g4, %l2
	sll %g1, %o1, %i1
	sub %l4, %o1, %g3
	sll %l2, %g4, %g2
	sub %l4, %g4, %i4
	srl %g1, %g3, %g1
	or %i2, %i5, %i2
	srl %l2, %i4, %l2
	or %i3, %o7, %i3
	or %g1, %i1, %g1
	or %l2, %g2, %l2
	add %g1, %i2, %g1
	add %l2, %i3, %l2
	add %i2, %l0, %i1
	add %i3, %l1, %g2
	ld	[%fp-32], %o0
	add %o5, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-240], %o2
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	xor %o1, %g1, %o1
	srl %i3, 29, %i3
	xor %g4, %l2, %g4
	sll %o1, %g1, %i1
	sub %l4, %g1, %g3
	sll %g4, %l2, %g2
	sub %l4, %l2, %i4
	srl %o1, %g3, %o1
	or %i2, %i5, %i2
	srl %g4, %i4, %g4
	or %i3, %o7, %i3
	or %o1, %i1, %o1
	or %g4, %g2, %g4
	add %o1, %i2, %o1
	add %g4, %i3, %g4
	add %i2, %o5, %i1
	add %i3, %l3, %g2
	ld	[%fp-28], %o0
	add %o4, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-236], %o2
	sll %i5, %i1, %o4
	add %o3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %o3
	or %o4, %i5, %o4
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %o3, %o7, %o3
	add %i2, %o4, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %o3, %i3
	sll %i3, 3, %o7
	xor %g1, %o1, %g1
	srl %i3, 29, %i3
	xor %l2, %g4, %l2
	sll %g1, %o1, %i1
	sub %l4, %o1, %g3
	sll %l2, %g4, %g2
	sub %l4, %g4, %i4
	srl %g1, %g3, %g1
	or %i2, %i5, %i2
	srl %l2, %i4, %l2
	or %i3, %o7, %i3
	or %g1, %i1, %g1
	or %l2, %g2, %l2
	add %g1, %i2, %g1
	add %l2, %i3, %l2
	add %i2, %o4, %i1
	add %i3, %o3, %g2
	ld	[%fp-24], %o0
	add %l0, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-232], %o2
	sll %i5, %i1, %l0
	add %l1, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l1
	or %l0, %i5, %l0
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l1, %o7, %l1
	add %i2, %l0, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l1, %i3
	sll %i3, 3, %o7
	xor %o1, %g1, %o1
	srl %i3, 29, %i3
	xor %g4, %l2, %g4
	sll %o1, %g1, %i1
	sub %l4, %g1, %g3
	sll %g4, %l2, %g2
	sub %l4, %l2, %i4
	srl %o1, %g3, %o1
	or %i2, %i5, %i2
	srl %g4, %i4, %g4
	or %i3, %o7, %i3
	or %o1, %i1, %o1
	or %g4, %g2, %g4
	ld	[%l5+12], %i1
	add %o1, %i2, %o1
	add %g4, %i3, %g4
	cmp	%o1, %i1
	be	.LL12
	cmp	%g4, %i1
	bne,a	.LL28
	ld	[%l5+12], %g2
.LL12:
	add %i2, %l0, %i1
	add %i3, %l1, %g2
	ld	[%fp-20], %o0
	add %o5, %i1, %i5
	sub %l4, %i1, %g3
	ld	[%fp-228], %o2
	sll %i5, %i1, %o5
	add %l3, %g2, %o7
	srl %i5, %g3, %i5
	sub %l4, %g2, %i4
	sll %o7, %g2, %l3
	or %o5, %i5, %o5
	srl %o7, %i4, %o7
	add %i2, %o0, %i2
	or %l3, %o7, %l3
	add %i2, %o5, %i2
	sll %i2, 3, %i5
	add %i3, %o2, %i3
	srl %i2, 29, %i2
	add %i3, %l3, %i3
	sll %i3, 3, %o7
	xor %g1, %o1, %g1
	srl %i3, 29, %i3
	xor %l2, %g4, %l2
	sll %g1, %o1, %i1
	sub %l4, %o1, %g3
	sll %l2, %g4, %g2
	sub %l4, %g4, %i4
	srl %g1, %g3, %g1
	or %i2, %i5, %i2
	srl %l2, %i4, %l2
	or %i3, %o7, %i3
	or %g1, %i1, %g1
	or %l2, %g2, %l2
	add %g1, %i2, %g1
	add %l2, %i3, %l2
	ld	[%l5+12], %g2
.LL28:
	cmp	%o1, %g2
	bne,a	.LL29
	ld	[%l5+12], %g2
	ld	[%l5+28], %g2
	ld	[%l5+16], %i2
	ld	[%l5+20], %o0
	ld	[%l5+24], %o4
	ld	[%l5+8], %i0
	add	%g2, 1, %g2
	st	%g2, [%l5+28]
	st	%i2, [%l5+32]
	st	%o0, [%l5+36]
	cmp	%g1, %i0
	bne	.LL13
	st	%o4, [%l5+40]
	ld	[%fp+72], %g3
	ld	[%fp-436], %i0
	ld	[%g3], %g2
	ld	[%fp+72], %i1
	sll	%i0, 1, %g3
	sub	%g2, %g3, %g2
	st	%g2, [%i1]
	b	.LL27
	mov	2, %i0
.LL13:
	ld	[%l5+12], %g2
.LL29:
	cmp	%g4, %g2
	bne	.LL30
	ld	[%l5+16], %g2
	ld	[%l5+28], %g3
	ld	[%l5+20], %i4
	ld	[%l5+24], %o1
	ld	[%l5+8], %i1
	add	%g3, 1, %g3
	add	%g2, 1, %g2
	st	%g3, [%l5+28]
	st	%g2, [%l5+32]
	st	%i4, [%l5+36]
	cmp	%l2, %i1
	bne	.LL15
	st	%o1, [%l5+40]
	ld	[%fp+72], %i3
	ld	[%fp-436], %i0
	ld	[%i3], %g2
	sll	%i0, 1, %g3
	add	%g2, 1, %g2
	sub	%g2, %g3, %g2
	st	%g2, [%i3]
	b	.LL27
	mov	2, %i0
.LL15:
	ld	[%l5+16], %g2
.LL30:
	add	%g2, 2, %g2
	and	%g2, 255, %g2
	cmp	%g2, 0
	bne	.LL10
	st	%g2, [%l5+16]
	ld	[%l5+20], %g3
	sethi	%hi(16777216), %o4
	sethi	%hi(-16777216), %o3
	add	%g3, %o4, %g3
	andcc	%g3, %o3, %g0
	bne	.LL18
	st	%g3, [%l5+20]
	sethi	%hi(16776192), %o1
	sethi	%hi(65536), %o2
	or	%o1, 1023, %o1
	add	%g3, %o2, %g3
	sethi	%hi(16711680), %o0
	and	%g3, %o1, %g3
	andcc	%g3, %o0, %g0
	bne	.LL18
	st	%g3, [%l5+20]
	sethi	%hi(64512), %i1
	add	%g3, 256, %g3
	or	%i1, 1023, %o7
	and	%g3, %o7, %g3
	or	%i1, 768, %i5
	andcc	%g3, %i5, %g0
	bne	.LL18
	st	%g3, [%l5+20]
	add	%g3, 1, %g3
	and	%g3, 255, %g3
	cmp	%g3, 0
	bne	.LL18
	st	%g3, [%l5+20]
	ld	[%l5+24], %g2
	andcc	%g2, %o3, %g0
	bne	.LL22
	st	%g2, [%l5+24]
	add	%g2, %o2, %i2
	and	%i2, %o1, %g2
	andcc	%g2, %o0, %g0
	bne	.LL22
	st	%g2, [%l5+24]
	add	%g2, 256, %i0
	and	%i0, %o7, %g2
	andcc	%g2, %i5, %g0
	bne	.LL22
	st	%g2, [%l5+24]
	add	%g2, 1, %g2
	st	%g2, [%l5+24]
.LL22:
	ld	[%l5+24], %g3
	sethi	%hi(-1089828864), %i3
	or	%i3, 797, %i3
	mov	32, %i1
	add	%g3, %i3, %g3
	sub	%i1, %i3, %i1
	srl	%g3, %i1, %i1
	ld	[%fp-220], %g2
	sll	%g3, %i3, %g3
	or	%g3, %i1, %g3
	add	%g2, %i3, %g2
	add	%g2, %g3, %g2
	st	%g3, [%fp-444]
	srl	%g2, 29, %g3
	sll	%g2, 3, %l6
	or	%l6, %g3, %l6
.LL18:
	ld	[%fp-444], %g2
	ld	[%l5+20], %i0
	add	%l6, %g2, %i1
	mov	32, %g3
	add	%i0, %i1, %i0
	sub	%g3, %i1, %g3
	ld	[%fp-216], %g2
	srl	%i0, %g3, %g3
	sll	%i0, %i1, %l7
	or	%l7, %g3, %l7
	add	%g2, %l6, %g2
	add	%g2, %l7, %g2
	srl	%g2, 29, %g3
	sll	%g2, 3, %g2
	or	%g2, %g3, %g2
	st	%g2, [%fp-440]
.LL10:
	ld	[%fp-436], %g3
	addcc	%g3, -1, %g3
	bne	.LL25
	st	%g3, [%fp-436]
	mov	1, %i0
.LL27:
	ret
	restore
.LLFE1:
.LLfe1:
	.size	 rc5_72_unit_func_KKS_2,.LLfe1-rc5_72_unit_func_KKS_2

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
	.byte	0x4
	.uaword	.LLCFI0-.LLFB1
	.byte	0xd
	.byte	0x1e
	.byte	0x2d
	.byte	0x9
	.byte	0x65
	.byte	0x1f
	.align 4
.LLEFDE1:
	.ident	"GCC: (GNU) 2.95.3 20010315 (release)"
