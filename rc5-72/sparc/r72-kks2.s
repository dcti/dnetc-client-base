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
	save	%sp, -616, %sp
.LLCFI0:
	!#PROLOGUE# 1
	st	%i0, [%fp+68]
	ldub	[%i0+25], %g2
	sll	%g2, 16, %o5
	ldub	[%i0+21], %g2
	ldub	[%i0+24], %o7
	sll	%g2, 16, %g3
	ldub	[%i0+20], %i5
	sll	%o7, 24, %o7
	ldub	[%i0+16], %i3
	or	%o5, %o7, %o5
	st	%g3, [%fp-464]
	ld	[%fp+68], %g4
	sll	%i5, 24, %i5
	ldub	[%i0+17], %g1
	sll	%i3, 24, %i3
	ldub	[%i0+4], %i2
	sll	%g1, 16, %g1
	ldub	[%i0+5], %l7
	sll	%i2, 24, %i2
	ldub	[%i0], %o2
	sll	%l7, 16, %l7
	ldub	[%i0+1], %l6
	sll	%o2, 24, %o2
	ldub	[%i0+12], %o1
	sll	%l6, 16, %l6
	ldub	[%i0+13], %l5
	sll	%o1, 24, %o1
	ldub	[%i0+8], %o0
	sll	%l5, 16, %l5
	ldub	[%i0+9], %l4
	sll	%o0, 24, %o0
	ldub	[%i0+28], %i4
	sll	%l4, 16, %l4
	ldub	[%i0+29], %l3
	sll	%i4, 24, %i4
	ldub	[%g4+41], %l0
	sll	%l3, 16, %l3
	ldub	[%g4+36], %g3
	sll	%l0, 16, %l0
	ldub	[%g4+37], %o4
	or	%g1, %i3, %g1
	ldub	[%g4+32], %g2
	or	%l7, %i2, %l7
	ldub	[%g4+33], %o3
	or	%l6, %o2, %l6
	ldub	[%g4+26], %l2
	or	%l5, %o1, %l5
	ldub	[%g4+22], %l1
	or	%l4, %o0, %l4
	ldub	[%i0+40], %i0
	or	%l3, %i4, %l3
	st	%o5, [%fp-452]
	ld	[%fp-464], %o7
	sll	%i0, 24, %i0
	ldub	[%g4+18], %o5
	or	%o7, %i5, %i5
	st	%i5, [%fp-468]
	or	%l0, %i0, %l0
	ldub	[%g4+27], %i0
	sll	%g3, 24, %g3
	ldub	[%g4+6], %o7
	sll	%o4, 16, %o4
	ldub	[%g4+2], %i5
	or	%o4, %g3, %o4
	ldub	[%g4+14], %i3
	sll	%l2, 8, %l2
	ldub	[%g4+10], %i2
	sll	%l1, 8, %l1
	ldub	[%g4+30], %o2
	sll	%i5, 8, %i5
	ldub	[%g4+42], %o1
	or	%i5, %l6, %i5
	ldub	[%g4+38], %o0
	sll	%o7, 8, %o7
	ldub	[%g4+34], %i4
	or	%o7, %l7, %o7
	st	%i0, [%fp-456]
	ldub	[%g4+23], %g3
	sll	%i3, 8, %i3
	ld	[%fp+68], %i0
	st	%g3, [%fp-472]
	ldub	[%i0+19], %i0
	sll	%i2, 8, %i2
	st	%i0, [%fp-480]
	ld	[%fp-452], %g4
	sll	%o2, 8, %o2
	ld	[%fp-468], %g3
	or	%l2, %g4, %l2
	ld	[%fp+68], %i0
	or	%l1, %g3, %l1
	ldub	[%i0+3], %g3
	or	%i2, %l4, %i2
	ldub	[%i0+7], %g4
	or	%i3, %l5, %i3
	ld	[%fp+68], %l6
	st	%g3, [%fp-492]
	ldub	[%l6+11], %g3
	or	%o2, %l3, %o2
	ld	[%fp+68], %l7
	or	%g4, %o7, %g4
	ldub	[%l7+43], %l5
	sll	%o1, 8, %o1
	ldub	[%l7+39], %l4
	or	%o1, %l0, %o1
	ldub	[%l7+35], %l3
	sll	%g2, 24, %g2
	ldub	[%i0+15], %i0
	sll	%o3, 16, %o3
	ldub	[%l6+31], %l6
	or	%o3, %g2, %o3
	st	%g4, [%fp-484]
	ld	[%fp-492], %g4
	sll	%i4, 8, %i4
	ld	[%fp-472], %l0
	or	%g4, %i5, %i5
	or	%g3, %i2, %g3
	or	%i4, %o3, %i4
	sll	%o5, 8, %o5
	ld	[%fp-456], %o7
	or	%l0, %l1, %g4
	or	%o5, %g1, %o5
	sll	%o0, 8, %o0
	sethi	%hi(-1209970688), %g2
	st	%g3, [%fp-500]
	sethi	%hi(-1640531968), %o3
	ld	[%fp-480], %l1
	or	%o3, 441, %g3
	or	%i0, %i3, %i0
	or	%o0, %o4, %o0
	or	%g2, 355, %g2
	or	%o7, %l2, %g1
	st	%i0, [%fp-496]
	mov	%g3, %i2
	or	%l1, %o5, %o5
	or	%l5, %o1, %l5
	or	%l4, %o0, %l4
	or	%l3, %i4, %l3
	st	%i1, [%fp+72]
	add	%g2, %g3, %g2
	mov	1, %l2
	st	%i5, [%fp-488]
	st	%o5, [%fp-476]
	or	%l6, %o2, %l6
	st	%l5, [%fp-504]
	st	%l4, [%fp-508]
	st	%l3, [%fp-512]
	mov	32, %l7
	st	%l2, [%fp-436]
	add	%fp, -224, %i1
	add	%fp, -432, %i0
	mov	4, %g3
.LL6:
	st	%g2, [%g3+%i0]
	st	%g2, [%g3+%i1]
	ld	[%fp-436], %l3
	add	%g2, %i2, %g2
	add	%l3, 1, %l3
	st	%l3, [%fp-436]
	cmp	%l3, 25
	bleu	.LL6
	add	%g3, 4, %g3
	sethi	%hi(-1089828864), %l4
	or	%l4, 797, %l4
	mov	32, %i0
	add	%g1, %l4, %i1
	sub	%i0, %l4, %g2
	srl	%i1, %g2, %g2
	ld	[%fp-220], %g3
	sll	%i1, %l4, %i1
	or	%i1, %g2, %i1
	add	%g3, %l4, %g3
	st	%i1, [%fp-444]
	add	%g3, %i1, %g3
	srl	%g3, 29, %g2
	sll	%g3, 3, %i1
	ld	[%fp-444], %l5
	or	%i1, %g2, %i1
	add	%i1, %l5, %i2
	add	%g4, %i2, %g3
	sub	%i0, %i2, %i0
	srl	%g3, %i0, %i0
	ld	[%fp-216], %g2
	sll	%g3, %i2, %g3
	or	%g3, %i0, %g3
	ld	[%fp+72], %i0
	add	%g2, %i1, %g2
	add	%g2, %g3, %g2
	st	%g3, [%fp-448]
	ld	[%i0], %g3
	srl	%g2, 29, %i0
	sll	%g2, 3, %g2
	srl	%g3, 1, %g3
	or	%g2, %i0, %g2
	std	%g2, [%fp-440]

!INNER LOOP
.LL25:
	ld	[%fp-440], %g3
	ld	[%fp-448], %i5
	ld	[%fp-476], %g2

	add %g3, %i5, %i2
	add	%g2, 1, %l0
	
	add %g2, %i2, %i3
	ld	[%fp-212], %o0

	add %g3, %i5, %i4
	sub %l7, %i2, %i5
	ld	[%fp-440], %o1

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	st	%i1, [%fp-324]

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	st	%i1, [%fp-116]

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %o1, %o0, %g2
	st	%g3, [%fp-112]

	or %l0, %o7, %l0
	add %g2, %l1, %g2
	st	%g3, [%fp-320]

	sll %g2, 3, %i3
	add %g3, %o0, %g3
	ld	[%fp-444], %o2

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %o2, %i2, %i3
	st	%g2, [%fp-108]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-316]

	sll %i3, %i2, %l2
	add %o2, %i4, %o7
	ld	[%fp-208], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3
	ld	[%fp-448], %o3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %o3, %i2, %i3
	st	%g2, [%fp-104]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-312]

	sll %i3, %i2, %l3
	add %o3, %i4, %o7
	ld	[%fp-204], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-100]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-308]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-200], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-96]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-304]
	
	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-196], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-92]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-300]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-192], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-88]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-296]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-188], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-84]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-292]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-184], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-80]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-288]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-180], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-76]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-284]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-176], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-72]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-280]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-172], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-68]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-276]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-168], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-64]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-272]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-164], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-60]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-268]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-160], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-56]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-264]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-156], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-52]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-260]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-152], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-48]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-256]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-148], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-44]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-252]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-144], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-40]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-248]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-140], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-36]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-244]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-136], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-32]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-240]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-132], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-28]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-236]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-128], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-24]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-232]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-124], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o0, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-20]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-228]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	sethi	%hi(-1089828864), %i2

	srl %o7, %i0, %o7
	or	%i2, 797, %i2

	or %l3, %i3, %l3
	add %g2, %i2, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %i2, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-120]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-328]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %i1, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %i1, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-116]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-324]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-320], %o5

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o1, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-112]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-320]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-108], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-316], %o5

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-108]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-316]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-104], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-312], %o5

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-104]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-312]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-100], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-308], %o5

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-100]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-308]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-96], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-304], %o5

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-96]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-304]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-92], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-300], %o5

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-92]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-300]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-88], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-296], %o5

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-88]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-296]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-84], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-292], %o5

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-84]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-292]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-80], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-288], %o5

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-80]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-288]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-76], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-284], %o5

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-76]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-284]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-72], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-280], %o5

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-72]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-280]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-68], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-276], %o5

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-68]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-276]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-64], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-272], %o5

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-64]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-272]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-60], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-268], %o5

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-60]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-268]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-56], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-264], %o5

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-56]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-264]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-52], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-260], %o5

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-52]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-260]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-48], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-256], %o5

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-48]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-256]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-44], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-252], %o5

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-44]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-252]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-40], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-248], %o5

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-40]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-248]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-36], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-244], %o5

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-36]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-244]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-32], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-240], %o5

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-32]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-240]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-28], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-236], %o5

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3
	st	%g2, [%fp-28]

	add %g3, %l4, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-236]

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-24], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-232], %o5

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l3, %i2

	or %g3, %o7, %g3
	add %l1, %i2, %i3
	st	%g2, [%fp-24]

	add %g3, %l5, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-232]

	sll %i3, %i2, %l1
	add %l0, %i4, %o7
	ld	[%fp-20], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-228], %o5

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add %g2, %l1, %i2

	or %g3, %o7, %g3
	add %l2, %i2, %i3
	st	%g2, [%fp-20]

	add %g3, %l0, %i4
	sub %l7, %i2, %i5
	st	%g3, [%fp-228]

	sll %i3, %i2, %l2
	add %l4, %i4, %o7
	ld	[%fp-120], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-328], %o5

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2
	ld	[%fp-484], %o4

	srl %g3, 29, %g3
	add %g2, %l2, %i2

	or %g3, %o7, %g3
	add %l3, %i2, %i3

	add %g3, %l4, %i4
	sub %l7, %i2, %i5

	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	ld	[%fp-116], %o0

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	ld	[%fp-324], %o5

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	add	%o4, %g2, %o1
	add	%o4, %g3, %o4
	ld	[%fp-488], %o3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	or %g2, %i3, %g2

	srl %g3, 29, %g3
	add	%o3, %g2, %o2

	or %g3, %o7, %g3
	add	%o3, %g3, %o3

	add %g2, %l3, %i2
	add %g3, %l5, %i4
	ld	[%fp-112], %o0

	add %l1, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-320], %o5

	sll %i3, %i2, %l1
	add %l0, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	xor %o1, %o2, %o1

	srl %g3, 29, %g3
	xor %o4, %o3, %o4

	sll %o1, %o2, %i2
	sub %l7, %o2, %i5

	sll %o4, %o3, %i4
	sub %l7, %o3, %i0

	srl %o1, %i5, %o1
	or %g2, %i3, %g2

	srl %o4, %i0, %o4
	or %g3, %o7, %g3

	or %o1, %i2, %o1
	or %o4, %i4, %o4

	add %o1, %g2, %o1
	add %o4, %g3, %o4

	add %g2, %l1, %i2
	add %g3, %l0, %i4
	ld	[%fp-108], %o0

	add %l2, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-316], %o5

	sll %i3, %i2, %l2
	add %l4, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	xor %o2, %o1, %o2

	srl %g3, 29, %g3
	xor %o3, %o4, %o3

	sll %o2, %o1, %i2
	sub %l7, %o1, %i5

	sll %o3, %o4, %i4
	sub %l7, %o4, %i0

	srl %o2, %i5, %o2
	or %g2, %i3, %g2

	srl %o3, %i0, %o3
	or %g3, %o7, %g3

	or %o2, %i2, %o2
	or %o3, %i4, %o3

	add %o2, %g2, %o2
	add %o3, %g3, %o3

	add %g2, %l2, %i2
	add %g3, %l4, %i4
	ld	[%fp-104], %o0

	add %l3, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-312], %o5

	sll %i3, %i2, %l3
	add %l5, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	xor %o1, %o2, %o1

	srl %g3, 29, %g3
	xor %o4, %o3, %o4

	sll %o1, %o2, %i2
	sub %l7, %o2, %i5

	sll %o4, %o3, %i4
	sub %l7, %o3, %i0

	srl %o1, %i5, %o1
	or %g2, %i3, %g2

	srl %o4, %i0, %o4
	or %g3, %o7, %g3

	or %o1, %i2, %o1
	or %o4, %i4, %o4

	add %o1, %g2, %o1
	add %o4, %g3, %o4

	add %g2, %l3, %i2
	add %g3, %l5, %i4
	ld	[%fp-100], %o0

	add %l1, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-308], %o5

	sll %i3, %i2, %l1
	add %l0, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	xor %o2, %o1, %o2

	srl %g3, 29, %g3
	xor %o3, %o4, %o3

	sll %o2, %o1, %i2
	sub %l7, %o1, %i5

	sll %o3, %o4, %i4
	sub %l7, %o4, %i0

	srl %o2, %i5, %o2
	or %g2, %i3, %g2

	srl %o3, %i0, %o3
	or %g3, %o7, %g3

	or %o2, %i2, %o2
	or %o3, %i4, %o3

	add %o2, %g2, %o2
	add %o3, %g3, %o3

	add %g2, %l1, %i2
	add %g3, %l0, %i4
	ld	[%fp-96], %o0

	add %l2, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-304], %o5

	sll %i3, %i2, %l2
	add %l4, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	xor %o1, %o2, %o1

	srl %g3, 29, %g3
	xor %o4, %o3, %o4

	sll %o1, %o2, %i2
	sub %l7, %o2, %i5

	sll %o4, %o3, %i4
	sub %l7, %o3, %i0

	srl %o1, %i5, %o1
	or %g2, %i3, %g2

	srl %o4, %i0, %o4
	or %g3, %o7, %g3

	or %o1, %i2, %o1
	or %o4, %i4, %o4

	add %o1, %g2, %o1
	add %o4, %g3, %o4

	add %g2, %l2, %i2
	add %g3, %l4, %i4
	ld	[%fp-92], %o0

	add %l3, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-300], %o5

	sll %i3, %i2, %l3
	add %l5, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	xor %o2, %o1, %o2

	srl %g3, 29, %g3
	xor %o3, %o4, %o3

	sll %o2, %o1, %i2
	sub %l7, %o1, %i5

	sll %o3, %o4, %i4
	sub %l7, %o4, %i0

	srl %o2, %i5, %o2
	or %g2, %i3, %g2

	srl %o3, %i0, %o3
	or %g3, %o7, %g3

	or %o2, %i2, %o2
	or %o3, %i4, %o3

	add %o2, %g2, %o2
	add %o3, %g3, %o3

	add %g2, %l3, %i2
	add %g3, %l5, %i4
	ld	[%fp-88], %o0

	add %l1, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-296], %o5

	sll %i3, %i2, %l1
	add %l0, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	xor %o1, %o2, %o1

	srl %g3, 29, %g3
	xor %o4, %o3, %o4

	sll %o1, %o2, %i2
	sub %l7, %o2, %i5

	sll %o4, %o3, %i4
	sub %l7, %o3, %i0

	srl %o1, %i5, %o1
	or %g2, %i3, %g2

	srl %o4, %i0, %o4
	or %g3, %o7, %g3

	or %o1, %i2, %o1
	or %o4, %i4, %o4

	add %o1, %g2, %o1
	add %o4, %g3, %o4

	add %g2, %l1, %i2
	add %g3, %l0, %i4
	ld	[%fp-84], %o0

	add %l2, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-292], %o5

	sll %i3, %i2, %l2
	add %l4, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	xor %o2, %o1, %o2

	srl %g3, 29, %g3
	xor %o3, %o4, %o3

	sll %o2, %o1, %i2
	sub %l7, %o1, %i5

	sll %o3, %o4, %i4
	sub %l7, %o4, %i0

	srl %o2, %i5, %o2
	or %g2, %i3, %g2

	srl %o3, %i0, %o3
	or %g3, %o7, %g3

	or %o2, %i2, %o2
	or %o3, %i4, %o3

	add %o2, %g2, %o2
	add %o3, %g3, %o3

	add %g2, %l2, %i2
	add %g3, %l4, %i4
	ld	[%fp-80], %o0

	add %l3, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-288], %o5

	sll %i3, %i2, %l3
	add %l5, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	xor %o1, %o2, %o1

	srl %g3, 29, %g3
	xor %o4, %o3, %o4

	sll %o1, %o2, %i2
	sub %l7, %o2, %i5

	sll %o4, %o3, %i4
	sub %l7, %o3, %i0

	srl %o1, %i5, %o1
	or %g2, %i3, %g2

	srl %o4, %i0, %o4
	or %g3, %o7, %g3

	or %o1, %i2, %o1
	or %o4, %i4, %o4

	add %o1, %g2, %o1
	add %o4, %g3, %o4

	add %g2, %l3, %i2
	add %g3, %l5, %i4
	ld	[%fp-76], %o0

	add %l1, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-284], %o5

	sll %i3, %i2, %l1
	add %l0, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	xor %o2, %o1, %o2

	srl %g3, 29, %g3
	xor %o3, %o4, %o3

	sll %o2, %o1, %i2
	sub %l7, %o1, %i5

	sll %o3, %o4, %i4
	sub %l7, %o4, %i0

	srl %o2, %i5, %o2
	or %g2, %i3, %g2

	srl %o3, %i0, %o3
	or %g3, %o7, %g3

	or %o2, %i2, %o2
	or %o3, %i4, %o3

	add %o2, %g2, %o2
	add %o3, %g3, %o3

	add %g2, %l1, %i2
	add %g3, %l0, %i4
	ld	[%fp-72], %o0

	add %l2, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-280], %o5

	sll %i3, %i2, %l2
	add %l4, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	xor %o1, %o2, %o1

	srl %g3, 29, %g3
	xor %o4, %o3, %o4

	sll %o1, %o2, %i2
	sub %l7, %o2, %i5

	sll %o4, %o3, %i4
	sub %l7, %o3, %i0

	srl %o1, %i5, %o1
	or %g2, %i3, %g2

	srl %o4, %i0, %o4
	or %g3, %o7, %g3

	or %o1, %i2, %o1
	or %o4, %i4, %o4

	add %o1, %g2, %o1
	add %o4, %g3, %o4

	add %g2, %l2, %i2
	add %g3, %l4, %i4
	ld	[%fp-68], %o0

	add %l3, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-276], %o5

	sll %i3, %i2, %l3
	add %l5, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	xor %o2, %o1, %o2

	srl %g3, 29, %g3
	xor %o3, %o4, %o3

	sll %o2, %o1, %i2
	sub %l7, %o1, %i5

	sll %o3, %o4, %i4
	sub %l7, %o4, %i0

	srl %o2, %i5, %o2
	or %g2, %i3, %g2

	srl %o3, %i0, %o3
	or %g3, %o7, %g3

	or %o2, %i2, %o2
	or %o3, %i4, %o3

	add %o2, %g2, %o2
	add %o3, %g3, %o3

	add %g2, %l3, %i2
	add %g3, %l5, %i4
	ld	[%fp-64], %o0

	add %l1, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-272], %o5

	sll %i3, %i2, %l1
	add %l0, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	xor %o1, %o2, %o1

	srl %g3, 29, %g3
	xor %o4, %o3, %o4

	sll %o1, %o2, %i2
	sub %l7, %o2, %i5

	sll %o4, %o3, %i4
	sub %l7, %o3, %i0

	srl %o1, %i5, %o1
	or %g2, %i3, %g2

	srl %o4, %i0, %o4
	or %g3, %o7, %g3

	or %o1, %i2, %o1
	or %o4, %i4, %o4

	add %o1, %g2, %o1
	add %o4, %g3, %o4

	add %g2, %l1, %i2
	add %g3, %l0, %i4
	ld	[%fp-60], %o0

	add %l2, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-268], %o5

	sll %i3, %i2, %l2
	add %l4, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	xor %o2, %o1, %o2

	srl %g3, 29, %g3
	xor %o3, %o4, %o3

	sll %o2, %o1, %i2
	sub %l7, %o1, %i5

	sll %o3, %o4, %i4
	sub %l7, %o4, %i0

	srl %o2, %i5, %o2
	or %g2, %i3, %g2

	srl %o3, %i0, %o3
	or %g3, %o7, %g3

	or %o2, %i2, %o2
	or %o3, %i4, %o3

	add %o2, %g2, %o2
	add %o3, %g3, %o3

	add %g2, %l2, %i2
	add %g3, %l4, %i4
	ld	[%fp-56], %o0

	add %l3, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-264], %o5

	sll %i3, %i2, %l3
	add %l5, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	xor %o1, %o2, %o1

	srl %g3, 29, %g3
	xor %o4, %o3, %o4

	sll %o1, %o2, %i2
	sub %l7, %o2, %i5

	sll %o4, %o3, %i4
	sub %l7, %o3, %i0

	srl %o1, %i5, %o1
	or %g2, %i3, %g2

	srl %o4, %i0, %o4
	or %g3, %o7, %g3

	or %o1, %i2, %o1
	or %o4, %i4, %o4

	add %o1, %g2, %o1
	add %o4, %g3, %o4

	add %g2, %l3, %i2
	add %g3, %l5, %i4
	ld	[%fp-52], %o0

	add %l1, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-260], %o5

	sll %i3, %i2, %l1
	add %l0, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	xor %o2, %o1, %o2

	srl %g3, 29, %g3
	xor %o3, %o4, %o3

	sll %o2, %o1, %i2
	sub %l7, %o1, %i5

	sll %o3, %o4, %i4
	sub %l7, %o4, %i0

	srl %o2, %i5, %o2
	or %g2, %i3, %g2

	srl %o3, %i0, %o3
	or %g3, %o7, %g3

	or %o2, %i2, %o2
	or %o3, %i4, %o3

	add %o2, %g2, %o2
	add %o3, %g3, %o3

	add %g2, %l1, %i2
	add %g3, %l0, %i4
	ld	[%fp-48], %o0

	add %l2, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-256], %o5

	sll %i3, %i2, %l2
	add %l4, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	xor %o1, %o2, %o1

	srl %g3, 29, %g3
	xor %o4, %o3, %o4

	sll %o1, %o2, %i2
	sub %l7, %o2, %i5

	sll %o4, %o3, %i4
	sub %l7, %o3, %i0

	srl %o1, %i5, %o1
	or %g2, %i3, %g2

	srl %o4, %i0, %o4
	or %g3, %o7, %g3

	or %o1, %i2, %o1
	or %o4, %i4, %o4

	add %o1, %g2, %o1
	add %o4, %g3, %o4

	add %g2, %l2, %i2
	add %g3, %l4, %i4
	ld	[%fp-44], %o0

	add %l3, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-252], %o5

	sll %i3, %i2, %l3
	add %l5, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	xor %o2, %o1, %o2

	srl %g3, 29, %g3
	xor %o3, %o4, %o3

	sll %o2, %o1, %i2
	sub %l7, %o1, %i5

	sll %o3, %o4, %i4
	sub %l7, %o4, %i0

	srl %o2, %i5, %o2
	or %g2, %i3, %g2

	srl %o3, %i0, %o3
	or %g3, %o7, %g3

	or %o2, %i2, %o2
	or %o3, %i4, %o3

	add %o2, %g2, %o2
	add %o3, %g3, %o3

	add %g2, %l3, %i2
	add %g3, %l5, %i4
	ld	[%fp-40], %o0

	add %l1, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-248], %o5

	sll %i3, %i2, %l1
	add %l0, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	xor %o1, %o2, %o1

	srl %g3, 29, %g3
	xor %o4, %o3, %o4

	sll %o1, %o2, %i2
	sub %l7, %o2, %i5

	sll %o4, %o3, %i4
	sub %l7, %o3, %i0

	srl %o1, %i5, %o1
	or %g2, %i3, %g2

	srl %o4, %i0, %o4
	or %g3, %o7, %g3

	or %o1, %i2, %o1
	or %o4, %i4, %o4

	add %o1, %g2, %o1
	add %o4, %g3, %o4

	add %g2, %l1, %i2
	add %g3, %l0, %i4
	ld	[%fp-36], %o0

	add %l2, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-244], %o5

	sll %i3, %i2, %l2
	add %l4, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	xor %o2, %o1, %o2

	srl %g3, 29, %g3
	xor %o3, %o4, %o3

	sll %o2, %o1, %i2
	sub %l7, %o1, %i5

	sll %o3, %o4, %i4
	sub %l7, %o4, %i0

	srl %o2, %i5, %o2
	or %g2, %i3, %g2

	srl %o3, %i0, %o3
	or %g3, %o7, %g3

	or %o2, %i2, %o2
	or %o3, %i4, %o3

	add %o2, %g2, %o2
	add %o3, %g3, %o3

	add %g2, %l2, %i2
	add %g3, %l4, %i4
	ld	[%fp-32], %o0

	add %l3, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-240], %o5

	sll %i3, %i2, %l3
	add %l5, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l5
	or %l3, %i3, %l3

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l5, %o7, %l5
	add %g2, %l3, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l5, %g3

	sll %g3, 3, %o7
	xor %o1, %o2, %o1

	srl %g3, 29, %g3
	xor %o4, %o3, %o4

	sll %o1, %o2, %i2
	sub %l7, %o2, %i5

	sll %o4, %o3, %i4
	sub %l7, %o3, %i0

	srl %o1, %i5, %o1
	or %g2, %i3, %g2

	srl %o4, %i0, %o4
	or %g3, %o7, %g3

	or %o1, %i2, %o1
	or %o4, %i4, %o4

	add %o1, %g2, %o1
	add %o4, %g3, %o4

	add %g2, %l3, %i2
	add %g3, %l5, %i4
	ld	[%fp-28], %o0

	add %l1, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-236], %o5

	sll %i3, %i2, %l1
	add %l0, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l0
	or %l1, %i3, %l1

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l0, %o7, %l0
	add %g2, %l1, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l0, %g3

	sll %g3, 3, %o7
	xor %o2, %o1, %o2

	srl %g3, 29, %g3
	xor %o3, %o4, %o3

	sll %o2, %o1, %i2
	sub %l7, %o1, %i5

	sll %o3, %o4, %i4
	sub %l7, %o4, %i0

	srl %o2, %i5, %o2
	or %g2, %i3, %g2

	srl %o3, %i0, %o3
	or %g3, %o7, %g3

	or %o2, %i2, %o2
	or %o3, %i4, %o3

	add %o2, %g2, %o2
	add %o3, %g3, %o3

	add %g2, %l1, %i2
	add %g3, %l0, %i4
	ld	[%fp-24], %o0

	add %l2, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-232], %o5

	sll %i3, %i2, %l2
	add %l4, %i4, %o7

	srl %i3, %i5, %i3
	sub %l7, %i4, %i0

	sll %o7, %i4, %l4
	or %l2, %i3, %l2

	srl %o7, %i0, %o7
	add %g2, %o0, %g2

	or %l4, %o7, %l4
	add %g2, %l2, %g2

	sll %g2, 3, %i3
	add %g3, %o5, %g3

	srl %g2, 29, %g2
	add %g3, %l4, %g3

	sll %g3, 3, %o7
	xor %o1, %o2, %o1

	srl %g3, 29, %g3
	xor %o4, %o3, %o4

	sll %o1, %o2, %i2
	sub %l7, %o2, %i5

	sll %o4, %o3, %i4
	sub %l7, %o3, %i0

	srl %o1, %i5, %o1
	or %g2, %i3, %g2
	ld	[%fp-496], %i3

	srl %o4, %i0, %o4
	or %g3, %o7, %g3

	or %o1, %i2, %o1
	or %o4, %i4, %o4

	add %o1, %g2, %o1
	add %o4, %g3, %o4
	ld	[%fp-476], %i2

	cmp	%o1, %i3
	be	.LL12
	add	%i2, 1, %l0

	cmp	%o4, %i3
	bne	.LL28
	ld	[%fp-496], %i4

.LL12:
	add %g2, %l2, %i2
	add %g3, %l4, %i4
	ld	[%fp-20], %o0
	add %l3, %i2, %i3
	sub %l7, %i2, %i5
	ld	[%fp-228], %o5
	sll %i3, %i2, %l3
	add %l5, %i4, %o7
	srl %i3, %i5, %i3
	sub %l7, %i4, %i0
	sll %o7, %i4, %l5
	or %l3, %i3, %l3
	srl %o7, %i0, %o7
	add %g2, %o0, %g2
	or %l5, %o7, %l5
	add %g2, %l3, %g2
	sll %g2, 3, %i3
	add %g3, %o5, %g3
	srl %g2, 29, %g2
	add %g3, %l5, %g3
	sll %g3, 3, %o7
	xor %o2, %o1, %o2
	srl %g3, 29, %g3
	xor %o3, %o4, %o3
	sll %o2, %o1, %i2
	sub %l7, %o1, %i5
	sll %o3, %o4, %i4
	sub %l7, %o4, %i0
	srl %o2, %i5, %o2
	or %g2, %i3, %g2
	srl %o3, %i0, %o3
	or %g3, %o7, %g3
	or %o2, %i2, %o2
	or %o3, %i4, %o3
	add %o2, %g2, %o2
	add %o3, %g3, %o3
	ld	[%fp-496], %i4

.LL28:
	cmp	%o1, %i4
	bne	.LL29
	ld	[%fp-496], %l4

	ld	[%fp-500], %i5
	add	%l6, 1, %l6
	ld	[%fp-476], %g2
	cmp	%o2, %i5
	st	%g2, [%fp-512]
	st	%g4, [%fp-508]
	bne	.LL13
	st	%g1, [%fp-504]
	ld	[%fp+72], %g4
	srl	%g1, 16, %i1
	ld	[%fp-436], %o0
	srl	%g1, 8, %i2
	ld	[%g4], %g3
	sll	%o0, 1, %g2
	ld	[%fp-508], %o1
	sub	%g3, %g2, %g3
	ld	[%fp-512], %o2
	and	%o1, 255, %o4
	st	%g3, [%g4]
	ld	[%fp-504], %l0
	and	%o2, 255, %o5
	ld	[%fp-508], %l1
	srl	%o1, 16, %i3
	ld	[%fp-512], %l2
	srl	%o1, 8, %i4
	ld	[%fp+68], %l3
	srl	%o2, 16, %i5
	srl	%o2, 8, %o7
	and	%i1, 255, %i1
	and	%i2, 255, %i2
	and	%l0, 255, %o3
	and	%i3, 255, %i3
	and	%i4, 255, %i4
	and	%i5, 255, %i5
	and	%o7, 255, %o7
	srl	%l0, 24, %o0
	srl	%l1, 24, %o1
	srl	%l2, 24, %o2
	srl	%l6, 8, %i0
	srl	%l6, 24, %g2
	srl	%l6, 16, %g3
	stb	%i0, [%l3+30]
	stb	%o5, [%l3+35]
	stb	%g2, [%l3+28]
	stb	%g3, [%l3+29]
	stb	%l6, [%l3+31]
	stb	%o0, [%l3+40]
	stb	%i1, [%l3+41]
	stb	%i2, [%l3+42]
	stb	%o3, [%l3+43]
	stb	%o1, [%l3+36]
	stb	%i3, [%l3+37]
	stb	%i4, [%l3+38]
	stb	%o4, [%l3+39]
	stb	%o2, [%l3+32]
	stb	%i5, [%l3+33]
	stb	%o7, [%l3+34]
	stb	%o0, [%l3+24]
	stb	%i1, [%l3+25]
	stb	%i2, [%l3+26]
	stb	%o3, [%l3+27]
	stb	%o1, [%l3+20]
	stb	%i3, [%l3+21]
	stb	%i4, [%l3+22]
	stb	%o4, [%l3+23]
	stb	%o2, [%l3+16]
	stb	%i5, [%l3+17]
	stb	%o7, [%l3+18]
	stb	%o5, [%l3+19]
	b	.LL27
	mov	2, %i0
.LL13:
	ld	[%fp-496], %l4
.LL29:
	cmp	%o4, %l4
	bne,a	.LL30
	ld	[%fp-476], %l4
	ld	[%fp-500], %l5
	st	%l0, [%fp-512]
	cmp	%o3, %l5
	add	%l6, 1, %l6
	st	%g4, [%fp-508]
	bne	.LL15
	st	%g1, [%fp-504]
	ld	[%fp+72], %l7
	srl	%g4, 16, %o7
	ld	[%l7], %g2
	srl	%g4, 8, %o0
	ld	[%fp-436], %i0
	add	%g2, 1, %g2
	sll	%i0, 1, %g3
	sub	%g2, %g3, %g2
	st	%g2, [%l7]
	ld	[%fp-476], %i3
	srl	%g1, 16, %i4
	ld	[%fp+68], %l2
	srl	%i3, 8, %i0
	ld	[%fp-504], %i1
	srl	%i3, 24, %g2
	ld	[%fp-508], %i2
	srl	%i3, 16, %g3
	ld	[%fp-512], %g4
	and	%i1, 255, %l0
	stb	%i0, [%l2+18]
	stb	%g4, [%l2+35]
	stb	%g2, [%l2+16]
	stb	%g3, [%l2+17]
	ld	[%fp-476], %l3
	and	%i2, 255, %l1
	srl	%i1, 24, %o4
	srl	%i2, 24, %o5
	srl	%g1, 8, %i5
	and	%i4, 255, %i4
	and	%i5, 255, %i5
	and	%o7, 255, %o7
	and	%o0, 255, %o0
	srl	%l6, 24, %i1
	srl	%l6, 16, %i2
	srl	%l6, 8, %i3
	srl	%g4, 24, %o1
	srl	%g4, 16, %o2
	srl	%g4, 8, %o3
	stb	%l3, [%l2+19]
	stb	%i1, [%l2+28]
	stb	%i2, [%l2+29]
	stb	%i3, [%l2+30]
	stb	%l6, [%l2+31]
	stb	%o4, [%l2+40]
	stb	%i4, [%l2+41]
	stb	%i5, [%l2+42]
	stb	%l0, [%l2+43]
	stb	%o5, [%l2+36]
	stb	%o7, [%l2+37]
	stb	%o0, [%l2+38]
	stb	%l1, [%l2+39]
	stb	%o1, [%l2+32]
	stb	%o2, [%l2+33]
	stb	%o3, [%l2+34]
	stb	%o4, [%l2+24]
	stb	%i4, [%l2+25]
	stb	%i5, [%l2+26]
	stb	%l0, [%l2+27]
	stb	%o5, [%l2+20]
	stb	%o7, [%l2+21]
	stb	%o0, [%l2+22]
	stb	%l1, [%l2+23]
	b	.LL27
	mov	2, %i0
.LL15:
	ld	[%fp-476], %l4

! KEY INCREMENTATION
.LL30:
	add	%l4, 2, %g2				! key_hi += 2
	andcc	%g2, 255, %g2
	bne	.LL10
	st	%g2, [%fp-476]

	sethi	%hi(16777216), %o7
	add	%g4, %o7, %g4
	sethi	%hi(-16777216), %i5
	andcc	%g4, %i5, %g0
	bne	.LL31
	ld	[%fp-444], %i0

	sethi	%hi(65536), %i0
	sethi	%hi(16776192), %g2
	or	%g2, 1023, %i4
	add	%g4, %i0, %g3
	and	%g3, %i4, %g4
	sethi	%hi(16711680), %i2
	andcc	%g4, %i2, %g0
	bne,a	.LL31
	ld	[%fp-444], %i0

	sethi	%hi(64512), %g3
	or	%g3, 768, %i3
	or	%g3, 1023, %g3
	add	%g4, 256, %g2
	and	%g2, %g3, %g4
	andcc	%g4, %i3, %g0
	bne,a	.LL31
	ld	[%fp-444], %i0

	add	%g4, 1, %g2
	andcc	%g2, 255, %g4
	bne,a	.LL31
	ld	[%fp-444], %i0

	add	%g1, %o7, %g1
	andcc	%g1, %i5, %g0
	bne	.LL32
	sethi	%hi(-1089828864), %l5

	add	%g1, %i0, %g2
	and	%g2, %i4, %g1
	andcc	%g1, %i2, %g0
	bne	.LL33
	or	%l5, 797, %l5

	add	%g1, 256, %g2
	and	%g2, %g3, %g1
	andcc	%g1, %i3, %g0
	bne	.LL32
	sethi	%hi(-1089828864), %l5

	add	%g1, 1, %g2
	and	%g2, 255, %g1
.LL32:
	or	%l5, 797, %l5
.LL33:
	mov	32, %g2
	add	%g1, %l5, %i0
	sub	%g2, %l5, %g2
	srl	%i0, %g2, %g2
	ld	[%fp-220], %g3
	sll	%i0, %l5, %i0
	or	%i0, %g2, %i0
	add	%g3, %l5, %g3
	add	%g3, %i0, %g3
	srl	%g3, 29, %g2
	sll	%g3, 3, %i1
	st	%i0, [%fp-444]
	or	%i1, %g2, %i1
	ld	[%fp-444], %i0
.LL31:
	mov	32, %g2
	add	%i1, %i0, %i2
	add	%g4, %i2, %i0
	sub	%g2, %i2, %g2
	srl	%i0, %g2, %g2
	ld	[%fp-216], %g3
	sll	%i0, %i2, %i0
	or	%i0, %g2, %i0
	add	%g3, %i1, %g3
	add	%g3, %i0, %g3
	srl	%g3, 29, %g2
	sll	%g3, 3, %g3
	or	%g3, %g2, %g3
	st	%i0, [%fp-448]
	st	%g3, [%fp-440]

! LOOP END
.LL10:
	ld	[%fp-436], %i2
	addcc	%i2, -1, %i2
	bne	.LL25
	st	%i2, [%fp-436]

	ld	[%fp-476], %i5
	srl	%g1, 24, %g2
	srl	%i5, 16, %o0
	srl	%i5, 24, %i4
	srl	%i5, 8, %o7
	ld	[%fp+68], %i5
	st	%o0, [%fp-516]
	ld	[%fp-504], %l0
	srl	%g1, 16, %g3
	ld	[%fp-508], %l1
	srl	%g1, 8, %i0
	ld	[%fp-508], %l3
	srl	%g4, 24, %i1
	ld	[%fp-512], %l4
	srl	%g4, 16, %i2
	ld	[%fp-512], %l7
	srl	%g4, 8, %i3
	stb	%i0, [%i5+26]
	stb	%l7, [%i5+35]
	stb	%g2, [%i5+24]
	stb	%g3, [%i5+25]
	stb	%g1, [%i5+27]
	stb	%i1, [%i5+20]
	stb	%i2, [%i5+21]
	stb	%i3, [%i5+22]
	stb	%g4, [%i5+23]
	stb	%i4, [%i5+16]
	ld	[%fp-516], %g2
	srl	%l6, 24, %o0
	stb	%g2, [%i5+17]
	stb	%o7, [%i5+18]
	ld	[%fp-476], %g3
	srl	%l0, 24, %o3
	srl	%l0, 16, %o4
	srl	%l0, 8, %o5
	srl	%l6, 16, %o1
	srl	%l6, 8, %o2
	stb	%o0, [%i5+28]
	stb	%g3, [%i5+19]
	stb	%o1, [%i5+29]
	stb	%o2, [%i5+30]
	stb	%l6, [%i5+31]
	stb	%o3, [%i5+40]
	stb	%o4, [%i5+41]
	stb	%o5, [%i5+42]
	ld	[%fp-504], %g4
	srl	%l1, 24, %l0
	srl	%l1, 16, %l1
	stb	%g4, [%i5+43]
	stb	%l0, [%i5+36]
	stb	%l1, [%i5+37]
	srl	%l3, 8, %l2
	stb	%l2, [%i5+38]
	ld	[%fp-508], %o0
	srl	%l4, 24, %l3
	srl	%l4, 16, %l4
	srl	%l7, 8, %l5
	stb	%o0, [%i5+39]
	stb	%l3, [%i5+32]
	stb	%l4, [%i5+33]
	stb	%l5, [%i5+34]
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
