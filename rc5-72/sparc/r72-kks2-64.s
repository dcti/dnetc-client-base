	.file	"r72-kks2.cpp"
	.section	".text"
	.align 4
	.global rc5_72_unit_func_KKS_2
	.type	rc5_72_unit_func_KKS_2, #function
	.proc	04
rc5_72_unit_func_KKS_2:
.LLFB3:
	!#PROLOGUE# 0
	save	%sp, -1040, %sp
.LLCFI0:
	!#PROLOGUE# 1
	ldub	[%i0+20], %g5
	ldub	[%i0+21], %i5
	sllx	%g5, 24, %g5
	stx	%i0, [%fp+2175]
	stx	%g5, [%fp+1511]
	ldub	[%i0+16], %i4
	sllx	%i5, 16, %g5
	ldx	[%fp+2175], %o0
	ldub	[%i0+17], %i3
	ldub	[%i0+4], %i2
	ldub	[%i0+24], %g1
	ldub	[%i0+25], %g4
	ldub	[%i0+5], %l7
	sllx	%i4, 24, %i5
	stx	%g5, [%fp+1503]
	ldub	[%o0+12], %o7
	ldx	[%fp+2175], %o1
	ldub	[%i0], %i0
	sllx	%i3, 16, %g5
	stx	%i5, [%fp+1463]
	ldx	[%fp+2175], %o4
	ldub	[%o0+8], %o5
	ldub	[%o0+1], %l5
	ldub	[%o0+13], %l4
	ldub	[%o0+9], %l2
	ldub	[%o0+28], %l6
	ldub	[%o0+29], %l0
	ldub	[%o0+40], %l3
	ldub	[%o1+36], %l1
	ldub	[%o1+37], %o2
	sllx	%i2, 24, %i5
	sllx	%i0, 24, %i4
	sllx	%o7, 24, %i3
	ldub	[%o4+33], %o3
	sllx	%o5, 24, %i2
	ldub	[%o0+41], %o0
	ldub	[%o1+32], %o1
	sllx	%g4, 16, %g4
	stx	%g5, [%fp+1455]
	stx	%i5, [%fp+1431]
	stx	%i4, [%fp+1415]
	stx	%i3, [%fp+1399]
	sllx	%g1, 24, %g1
	stx	%i2, [%fp+1383]
	or	%g4, %g1, %g1
	sllx	%l5, 16, %l5
	stx	%g1, [%fp+1543]
	ldx	[%fp+1503], %g4
	ldx	[%fp+1511], %g1
	sllx	%l7, 16, %l7
	or	%g4, %g1, %g4
	ldub	[%o4+26], %i0
	stx	%g4, [%fp+1495]
	ldx	[%fp+1463], %g1
	ldx	[%fp+1455], %g4
	ldub	[%o4+22], %g5
	or	%g4, %g1, %g4
	sllx	%l2, 16, %l2
	stx	%g4, [%fp+1447]
	ldx	[%fp+1431], %g1
	ldx	[%fp+1415], %g4
	or	%l7, %g1, %l7
	or	%l5, %g4, %l5
	stx	%i0, [%fp+1191]
	ldx	[%fp+1383], %g4
	stx	%g5, [%fp+1183]
	ldub	[%o4+6], %i5
	ldx	[%fp+1399], %g1
	or	%l2, %g4, %l2
	sllx	%o1, 24, %o1
	sllx	%l4, 16, %l4
	sllx	%o3, 16, %o3
	ldub	[%o4+18], %g5
	ldub	[%o4+2], %i4
	ldub	[%o4+14], %i3
	ldub	[%o4+10], %i2
	ldub	[%o4+30], %i0
	ldub	[%o4+42], %o7
	ldub	[%o4+38], %o5
	or	%l4, %g1, %l4
	or	%o3, %o1, %o3
	ldx	[%fp+1191], %g4
	sllx	%l6, 24, %l6
	sllx	%l3, 24, %l3
	sllx	%l1, 24, %l1
	sllx	%l0, 16, %l0
	sllx	%o0, 16, %o0
	sllx	%o2, 16, %o2
	sllx	%i5, 8, %i5
	or	%l0, %l6, %l0
	or	%o0, %l3, %o0
	or	%o2, %l1, %o2
	ldx	[%fp+1183], %o1
	or	%i5, %l7, %i5
	ldub	[%o4+34], %o4
	sllx	%g4, 8, %g1
	sllx	%o5, 8, %o5
	ldx	[%fp+2175], %l7
	ldx	[%fp+1543], %l1
	ldx	[%fp+1495], %l3
	ldx	[%fp+1447], %l6
	sllx	%o1, 8, %g4
	or	%o5, %o2, %o5
	sllx	%g5, 8, %g5
	sllx	%i4, 8, %i4
	sllx	%i3, 8, %i3
	sllx	%i2, 8, %i2
	sllx	%i0, 8, %i0
	sllx	%o7, 8, %o7
	sllx	%o4, 8, %o4
	or	%g1, %l1, %g1
	or	%g4, %l3, %g4
	or	%g5, %l6, %g5
	or	%i4, %l5, %i4
	or	%i3, %l4, %i3
	or	%i2, %l2, %i2
	or	%i0, %l0, %i0
	or	%o7, %o0, %o7
	or	%o4, %o3, %o4
	ldub	[%l7+27], %o2
	ldub	[%l7+15], %l1
	ldub	[%l7+23], %o3
	ldub	[%l7+19], %o1
	ldub	[%l7+7], %o0
	ldub	[%l7+3], %l0
	ldub	[%l7+11], %l2
	ldub	[%l7+31], %l3
	ldub	[%l7+43], %l4
	ldub	[%l7+39], %l5
	ldub	[%l7+35], %l6
	or	%o2, %g1, %o2
	st	%o2, [%fp+1579]
	or	%o3, %g4, %g1
	st	%g1, [%fp+1531]
	or	%o0, %i5, %g4
	or	%o1, %g5, %g1
	or	%l1, %i3, %i5
	or	%l6, %o4, %g5
	st	%g1, [%fp+1483]
	or	%l0, %i4, %g1
	st	%g4, [%fp+1443]
	st	%g5, [%fp+1323]
	or	%l2, %i2, %i4
	or	%l3, %i0, %i3
	sethi	%hi(1640530944), %g4
	mov	1, %g5
	st	%g1, [%fp+1427]
	st	%i5, [%fp+1411]
	or	%l4, %o7, %i2
	or	%l5, %o5, %i0
	xor	%g4, -583, %i5
	sethi	%hi(-1209970688), %g1
	st	%g5, [%fp+1611]
	st	%i4, [%fp+1395]
	st	%i3, [%fp+1371]
	st	%i2, [%fp+1355]
	st	%i0, [%fp+1339]
	stx	%i1, [%fp+2183]
	or	%g1, 355, %g4
	mov	32, %l1
	add	%fp, 2035, %g5
.LL6:
	ld	[%fp+1611], %i2
	mulx	%i2, %i5, %i3
	add	%i2, 1, %i1
	add	%i3, %g4, %i0
	st	%i1, [%fp+1611]
	st	%i0, [%g5-208]
	st	%i0, [%g5-416]
	cmp	%i1, 25
	bleu,pt	%icc, .LL6
	add	%g5, 4, %g5
	sethi	%hi(-1089828864), %i1
	mov	32, %i5
	or	%i1, 797, %g5
	ld	[%fp+1579], %g1
	add	%g1, %g5, %i2
	sub	%i5, %g5, %g4
	srl	%i2, %g4, %i0
	sll	%i2, %g5, %i4
	ld	[%fp+1827], %i3
	or	%i4, %i0, %i1
	add	%i3, %g5, %g1
	st	%i1, [%fp+1595]
	st	%i3, [%fp+1303]
	ldx	[%fp+2183], %i1
	ld	[%fp+1595], %i3
	ld	[%fp+1531], %g4
	add	%g1, %i3, %i2
	ld	[%fp+1831], %g5
	srl	%i2, 29, %i0
	sll	%i2, 3, %i4
	or	%i4, %i0, %l6
	st	%g5, [%fp+1299]
	add	%l6, %i3, %i0
	ld	[%i1], %i4
	add	%g4, %i0, %i2
	sub	%i5, %i0, %g1
	srl	%i2, %g1, %i1
	sll	%i2, %i0, %i5
	add	%g5, %l6, %i3
	or	%i5, %i1, %l2
	st	%i0, [%fp+1291]
	srl	%i4, 1, %g4
	add	%i3, %l2, %i0
	ld	[%fp+1835], %i2
	srl	%i0, 29, %g5
	sll	%i0, 3, %i1
	ld	[%fp+1839], %i5
	st	%i4, [%fp+1307]
	st	%g4, [%fp+1611]
	st	%i2, [%fp+1287]
	or	%i1, %g5, %l5
	ld	[%fp+1843], %i4
	ld	[%fp+1847], %g1
	ld	[%fp+1851], %g4
	ld	[%fp+1855], %i3
	ld	[%fp+1859], %i2
	ld	[%fp+1863], %i0
	ld	[%fp+1867], %g5
	ld	[%fp+1871], %i1
	st	%i5, [%fp+1283]
	st	%i4, [%fp+1279]
	st	%g1, [%fp+1275]
	st	%g4, [%fp+1271]
	st	%i3, [%fp+1267]
	st	%i2, [%fp+1263]
	st	%i0, [%fp+1259]
	st	%g5, [%fp+1255]
	st	%i1, [%fp+1251]
	ld	[%fp+1875], %i5
	st	%i5, [%fp+1247]
	ld	[%fp+1879], %i4
	ld	[%fp+1883], %g1
	ld	[%fp+1887], %g4
	st	%i4, [%fp+1243]
	st	%g1, [%fp+1239]
	st	%g4, [%fp+1235]
	ld	[%fp+1891], %i3
	ld	[%fp+1895], %i2
	ld	[%fp+1899], %i0
	ld	[%fp+1903], %g5
	ld	[%fp+1907], %i1
	ld	[%fp+1911], %i5
	ld	[%fp+1915], %i4
	ld	[%fp+1919], %g1
	ld	[%fp+1923], %g4
	st	%i3, [%fp+1231]
	st	%i2, [%fp+1227]
	st	%i0, [%fp+1223]
	st	%g5, [%fp+1219]
	st	%i1, [%fp+1215]
	st	%i5, [%fp+1211]
	st	%i4, [%fp+1207]
	st	%g1, [%fp+1203]
	st	%g4, [%fp+1199]
.LL7:
	sethi	%hi(-1089828864), %i1
	ld	[%fp+1483], %i4
	or	%i1, 797, %g1
	add	%i4, 1, %o5
	st	%g1, [%fp+1719]
	st	%g1, [%fp+1927]
	st	%l6, [%fp+1723]
	st	%l6, [%fp+1931]
	st	%o5, [%fp+1295]
	add %l5, %l2, %i2
	mov	0, %g4
	or %l5, %g4, %i4
	ld	[%fp+1483], %o7
	add %o7, %i2, %i5
	st	%l5, [%fp+1935]
	add %i4, %l2, %g4
	sub %l1, %i2, %i3
	st	%i4, [%fp+1727]
	sll %i5, %i2, %g5
	add %o5, %g4, %g1
	srl %i5, %i3, %i0
	sub %l1, %g4, %i1
	sll %g1, %g4, %i5
	or %g5, %i0, %o1
	srl %g1, %i1, %i2
	ld	[%fp+1287], %l0
	add %l5, %l0, %g4
	or %i5, %i2, %l3
	add %g4, %o1, %i3
	sll %i3, 3, %i1
	add %i4, %l0, %i0
	srl %i3, 29, %g5
	add %i0, %l3, %g1
	sll %g1, 3, %i2
	or %g5, %i1, %i0
	srl %g1, 29, %g4
	add %i0, %o1, %i3
	or %g4, %i2, %i4
	ld	[%fp+1595], %l4
	add %l4, %i3, %i1
	st	%i0, [%fp+1939]
	add %i4, %l3, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1731]
	sll %i1, %i3, %g5
	ld	[%fp+1595], %l7
	add %l7, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %l4
	srl %g1, %i1, %i3
	ld	[%fp+1283], %g1
	add %i0, %g1, %g4
	or %i2, %i3, %l7
	add %g4, %l4, %i0
	sll %i0, 3, %g5
	add %i4, %g1, %i3
	srl %i0, 29, %i2
	add %i3, %l7, %i1
	sll %i1, 3, %g1
	or %i2, %g5, %i5
	srl %i1, 29, %g4
	add %i5, %l4, %i0
	or %g4, %g1, %i4
	add %l2, %i0, %i3
	st	%i5, [%fp+1943]
	add %i4, %l7, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1735]
	sll %i3, %i0, %g5
	add %l2, %g4, %i1
	srl %i3, %i2, %i0
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i0, %o0
	srl %i1, %i2, %i0
	ld	[%fp+1279], %g4
	add %i5, %g4, %i2
	or %g1, %i0, %l0
	add %i2, %o0, %i3
	sll %i3, 3, %i1
	add %i4, %g4, %i0
	srl %i3, 29, %g5
	add %i0, %l0, %g1
	sll %g1, 3, %g4
	or %g5, %i1, %i0
	srl %g1, 29, %i2
	add %i0, %o0, %i3
	or %i2, %g4, %i4
	add %o1, %i3, %i1
	st	%i0, [%fp+1947]
	add %i4, %l0, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1739]
	sll %i1, %i3, %g5
	add %l3, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %o1
	srl %g1, %i1, %g4
	ld	[%fp+1275], %g5
	add %i0, %g5, %g1
	or %i2, %g4, %l3
	add %g1, %o1, %i3
	sll %i3, 3, %g4
	add %i4, %g5, %i0
	srl %i3, 29, %i2
	add %i0, %l3, %i1
	sll %i1, 3, %g1
	or %i2, %g4, %i0
	srl %i1, 29, %g4
	add %i0, %o1, %i3
	or %g4, %g1, %i4
	add %l4, %i3, %g5
	st	%i0, [%fp+1951]
	add %i4, %l3, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1743]
	sll %g5, %i3, %l4
	add %l7, %g4, %i1
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	sll %i1, %g4, %g1
	or %l4, %i3, %l4
	srl %i1, %g5, %i2
	ld	[%fp+1271], %o2
	add %i0, %o2, %g4
	or %g1, %i2, %l7
	add %g4, %l4, %i3
	sll %i3, 3, %g5
	add %i4, %o2, %i0
	srl %i3, 29, %i2
	add %i0, %l7, %i1
	sll %i1, 3, %g1
	or %i2, %g5, %i0
	srl %i1, 29, %g4
	add %i0, %l4, %i3
	or %g4, %g1, %i4
	add %o0, %i3, %g5
	st	%i0, [%fp+1955]
	add %i4, %l7, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1747]
	sll %g5, %i3, %o0
	add %l0, %g4, %i1
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	sll %i1, %g4, %g1
	or %o0, %i3, %o0
	srl %i1, %g5, %i2
	ld	[%fp+1267], %o3
	add %i0, %o3, %g4
	or %g1, %i2, %l0
	add %g4, %o0, %i3
	sll %i3, 3, %i2
	add %i4, %o3, %i0
	srl %i3, 29, %g5
	add %i0, %l0, %i1
	sll %i1, 3, %g1
	or %g5, %i2, %i0
	srl %i1, 29, %g4
	add %i0, %o0, %i2
	or %g4, %g1, %i4
	add %o1, %i2, %g5
	st	%i0, [%fp+1959]
	add %i4, %l0, %g4
	sub %l1, %i2, %i3
	st	%i4, [%fp+1751]
	sll %g5, %i2, %o1
	add %l3, %g4, %i1
	srl %g5, %i3, %g1
	sub %l1, %g4, %g5
	sll %i1, %g4, %i2
	or %o1, %g1, %o1
	srl %i1, %g5, %g1
	ld	[%fp+1263], %o4
	add %i0, %o4, %g4
	or %i2, %g1, %l3
	add %g4, %o1, %i0
	sll %i0, 3, %g5
	add %i4, %o4, %i3
	srl %i0, 29, %g1
	add %i3, %l3, %i1
	sll %i1, 3, %i2
	or %g1, %g5, %i0
	srl %i1, 29, %g4
	add %i0, %o1, %g1
	or %g4, %i2, %i4
	add %l4, %g1, %i3
	st	%i0, [%fp+1963]
	add %i4, %l3, %g4
	sub %l1, %g1, %i2
	st	%i4, [%fp+1755]
	sll %i3, %g1, %g5
	add %l7, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %l4
	srl %i1, %i2, %i3
	ld	[%fp+1259], %o5
	add %i0, %o5, %i2
	or %g1, %i3, %i0
	add %i2, %l4, %g4
	sll %g4, 3, %g5
	add %i4, %o5, %i3
	srl %g4, 29, %g1
	add %i3, %i0, %i1
	sll %i1, 3, %i2
	or %g1, %g5, %i5
	srl %i1, 29, %g4
	add %i5, %l4, %g1
	or %g4, %i2, %i4
	add %o0, %g1, %i3
	st	%i5, [%fp+1967]
	add %i4, %i0, %g4
	sub %l1, %g1, %i2
	st	%i4, [%fp+1759]
	sll %i3, %g1, %g5
	add %l0, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %o0
	srl %i1, %i2, %i3
	ld	[%fp+1255], %o7
	add %i5, %o7, %i2
	or %g1, %i3, %l0
	add %i2, %o0, %g4
	sll %g4, 3, %i1
	add %i4, %o7, %i3
	srl %g4, 29, %g5
	add %i3, %l0, %g1
	sll %g1, 3, %i2
	or %g5, %i1, %i5
	srl %g1, 29, %g4
	add %i5, %o0, %i3
	or %g4, %i2, %i4
	add %o1, %i3, %i1
	st	%i5, [%fp+1971]
	add %i4, %l0, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1763]
	sll %i1, %i3, %g5
	add %l3, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %l3
	or %g5, %i3, %o1
	srl %g1, %i1, %i3
	ld	[%fp+1251], %g1
	add %i5, %g1, %i2
	or %l3, %i3, %l3
	add %i2, %o1, %g4
	sll %g4, 3, %g5
	add %i4, %g1, %i3
	srl %g4, 29, %g1
	add %i3, %l3, %i1
	sll %i1, 3, %i2
	or %g1, %g5, %i5
	srl %i1, 29, %g4
	add %i5, %o1, %g1
	or %g4, %i2, %i4
	add %l4, %g1, %i3
	st	%i5, [%fp+1975]
	add %i4, %l3, %g4
	sub %l1, %g1, %i2
	st	%i4, [%fp+1767]
	sll %i3, %g1, %g5
	add %i0, %g4, %i1
	srl %i3, %i2, %i0
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i0, %l4
	srl %i1, %i2, %i0
	ld	[%fp+1247], %g4
	add %i5, %g4, %i2
	or %g1, %i0, %l7
	add %i2, %l4, %i3
	sll %i3, 3, %i1
	add %i4, %g4, %i0
	srl %i3, 29, %g5
	add %i0, %l7, %g1
	sll %g1, 3, %g4
	or %g5, %i1, %i0
	srl %g1, 29, %i2
	add %i0, %l4, %i3
	or %i2, %g4, %i4
	add %o0, %i3, %i1
	st	%i0, [%fp+1979]
	add %i4, %l7, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1771]
	sll %i1, %i3, %g5
	add %l0, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %o0
	srl %g1, %i1, %g4
	ld	[%fp+1243], %g5
	add %i0, %g5, %g1
	or %i2, %g4, %l0
	add %g1, %o0, %i3
	sll %i3, 3, %g4
	add %i4, %g5, %i0
	srl %i3, 29, %i2
	add %i0, %l0, %i1
	sll %i1, 3, %g1
	or %i2, %g4, %i0
	srl %i1, 29, %g4
	add %i0, %o0, %i3
	or %g4, %g1, %i4
	add %o1, %i3, %g5
	st	%i0, [%fp+1983]
	add %i4, %l0, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1775]
	sll %g5, %i3, %o1
	add %l3, %g4, %i1
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	sll %i1, %g4, %g1
	or %o1, %i3, %o1
	srl %i1, %g5, %i2
	ld	[%fp+1239], %o2
	add %i0, %o2, %g4
	or %g1, %i2, %l3
	add %g4, %o1, %i3
	sll %i3, 3, %g5
	add %i4, %o2, %i0
	srl %i3, 29, %i2
	add %i0, %l3, %i1
	sll %i1, 3, %g1
	or %i2, %g5, %i0
	srl %i1, 29, %g4
	add %i0, %o1, %i3
	or %g4, %g1, %i4
	add %l4, %i3, %g5
	st	%i0, [%fp+1987]
	add %i4, %l3, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1779]
	sll %g5, %i3, %l4
	add %l7, %g4, %i1
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	sll %i1, %g4, %g1
	or %l4, %i3, %l4
	srl %i1, %g5, %i2
	ld	[%fp+1235], %o3
	add %i0, %o3, %g4
	or %g1, %i2, %l7
	add %g4, %l4, %i3
	sll %i3, 3, %i2
	add %i4, %o3, %i0
	srl %i3, 29, %g5
	add %i0, %l7, %i1
	sll %i1, 3, %g1
	or %g5, %i2, %i0
	srl %i1, 29, %g4
	add %i0, %l4, %i2
	or %g4, %g1, %i4
	add %o0, %i2, %g5
	st	%i0, [%fp+1991]
	add %i4, %l7, %g4
	sub %l1, %i2, %i3
	st	%i4, [%fp+1783]
	sll %g5, %i2, %o0
	add %l0, %g4, %i1
	srl %g5, %i3, %g1
	sub %l1, %g4, %g5
	sll %i1, %g4, %i2
	or %o0, %g1, %o0
	srl %i1, %g5, %g1
	ld	[%fp+1231], %o4
	add %i0, %o4, %g4
	or %i2, %g1, %l0
	add %g4, %o0, %i0
	sll %i0, 3, %g5
	add %i4, %o4, %i3
	srl %i0, 29, %g1
	add %i3, %l0, %i1
	sll %i1, 3, %i2
	or %g1, %g5, %i0
	srl %i1, 29, %g4
	add %i0, %o0, %g1
	or %g4, %i2, %i4
	add %o1, %g1, %i3
	st	%i0, [%fp+1995]
	add %i4, %l0, %g4
	sub %l1, %g1, %i2
	st	%i4, [%fp+1787]
	sll %i3, %g1, %g5
	add %l3, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %o1
	srl %i1, %i2, %i3
	ld	[%fp+1227], %o5
	add %i0, %o5, %i2
	or %g1, %i3, %i0
	add %i2, %o1, %g4
	sll %g4, 3, %g5
	add %i4, %o5, %i3
	srl %g4, 29, %g1
	add %i3, %i0, %i1
	sll %i1, 3, %i2
	or %g1, %g5, %i5
	srl %i1, 29, %g4
	add %i5, %o1, %g1
	or %g4, %i2, %i4
	add %l4, %g1, %i3
	st	%i5, [%fp+1999]
	add %i4, %i0, %g4
	sub %l1, %g1, %i2
	st	%i4, [%fp+1791]
	sll %i3, %g1, %g5
	add %l7, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %l4
	srl %i1, %i2, %i3
	ld	[%fp+1223], %o7
	add %i5, %o7, %i2
	or %g1, %i3, %l7
	add %i2, %l4, %g4
	sll %g4, 3, %i1
	add %i4, %o7, %i3
	srl %g4, 29, %g5
	add %i3, %l7, %g1
	sll %g1, 3, %i2
	or %g5, %i1, %i5
	srl %g1, 29, %g4
	add %i5, %l4, %i3
	or %g4, %i2, %i4
	add %o0, %i3, %i1
	st	%i5, [%fp+2003]
	add %i4, %l7, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1795]
	sll %i1, %i3, %g5
	add %l0, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %l0
	or %g5, %i3, %o0
	srl %g1, %i1, %i3
	ld	[%fp+1219], %g1
	add %i5, %g1, %i2
	or %l0, %i3, %l0
	add %i2, %o0, %g4
	sll %g4, 3, %g5
	add %i4, %g1, %i3
	srl %g4, 29, %g1
	add %i3, %l0, %i1
	sll %i1, 3, %i2
	or %g1, %g5, %i5
	srl %i1, 29, %g4
	add %i5, %o0, %g1
	or %g4, %i2, %i4
	add %o1, %g1, %i3
	st	%i5, [%fp+2007]
	add %i4, %l0, %g4
	sub %l1, %g1, %i2
	st	%i4, [%fp+1799]
	sll %i3, %g1, %g5
	add %i0, %g4, %i1
	srl %i3, %i2, %i0
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i0, %o1
	srl %i1, %i2, %i0
	ld	[%fp+1215], %g4
	add %i5, %g4, %i2
	or %g1, %i0, %l3
	add %i2, %o1, %i3
	sll %i3, 3, %i1
	add %i4, %g4, %i0
	srl %i3, 29, %g5
	add %i0, %l3, %g1
	sll %g1, 3, %g4
	or %g5, %i1, %i0
	srl %g1, 29, %i2
	add %i0, %o1, %i3
	or %i2, %g4, %i4
	add %l4, %i3, %i1
	st	%i0, [%fp+2011]
	add %i4, %l3, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1803]
	sll %i1, %i3, %g5
	add %l7, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %l4
	srl %g1, %i1, %g4
	ld	[%fp+1211], %g5
	add %i0, %g5, %g1
	or %i2, %g4, %l7
	add %g1, %l4, %i3
	sll %i3, 3, %g4
	add %i4, %g5, %i0
	srl %i3, 29, %i2
	add %i0, %l7, %i1
	sll %i1, 3, %g1
	or %i2, %g4, %i0
	srl %i1, 29, %g4
	add %i0, %l4, %i3
	or %g4, %g1, %i4
	add %o0, %i3, %g5
	st	%i0, [%fp+2015]
	add %i4, %l7, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1807]
	sll %g5, %i3, %o0
	add %l0, %g4, %i1
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	sll %i1, %g4, %g1
	or %o0, %i3, %o0
	srl %i1, %g5, %i2
	ld	[%fp+1207], %o2
	add %i0, %o2, %g4
	or %g1, %i2, %l0
	add %g4, %o0, %i3
	sll %i3, 3, %g5
	add %i4, %o2, %i0
	srl %i3, 29, %i2
	add %i0, %l0, %i1
	sll %i1, 3, %g1
	or %i2, %g5, %i0
	srl %i1, 29, %g4
	add %i0, %o0, %i3
	or %g4, %g1, %i4
	add %o1, %i3, %g5
	st	%i0, [%fp+2019]
	add %i4, %l0, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1811]
	sll %g5, %i3, %o1
	add %l3, %g4, %i1
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	sll %i1, %g4, %g1
	or %o1, %i3, %o1
	srl %i1, %g5, %i2
	ld	[%fp+1203], %o3
	add %i0, %o3, %g4
	or %g1, %i2, %l3
	add %g4, %o1, %i3
	sll %i3, 3, %i2
	add %i4, %o3, %i0
	srl %i3, 29, %g5
	add %i0, %l3, %i1
	sll %i1, 3, %g1
	or %g5, %i2, %i0
	srl %i1, 29, %g4
	add %i0, %o1, %i2
	or %g4, %g1, %i4
	add %l4, %i2, %g5
	st	%i0, [%fp+2023]
	add %i4, %l3, %g4
	sub %l1, %i2, %i3
	st	%i4, [%fp+1815]
	sll %g5, %i2, %l4
	add %l7, %g4, %i1
	srl %g5, %i3, %g1
	sub %l1, %g4, %g5
	sll %i1, %g4, %i2
	or %l4, %g1, %l4
	srl %i1, %g5, %g1
	ld	[%fp+1199], %o4
	add %i0, %o4, %g4
	or %i2, %g1, %l7
	add %g4, %l4, %i0
	sll %i0, 3, %g5
	add %i4, %o4, %i3
	srl %i0, 29, %g1
	add %i3, %l7, %i1
	sll %i1, 3, %i2
	or %g1, %g5, %i0
	srl %i1, 29, %g4
	add %i0, %l4, %g1
	or %g4, %i2, %i4
	add %o0, %g1, %i3
	st	%i0, [%fp+2027]
	add %i4, %l7, %g4
	sub %l1, %g1, %i2
	st	%i4, [%fp+1819]
	sll %i3, %g1, %g5
	add %l0, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %o0
	srl %i1, %i2, %i3
	sethi	%hi(-1089828864), %g5
	or	%g5, 797, %o5
	add %i0, %o5, %i2
	or %g1, %i3, %i0
	add %i2, %o0, %g4
	sll %g4, 3, %i1
	add %i4, %o5, %i3
	srl %g4, 29, %g5
	add %i3, %i0, %g1
	sll %g1, 3, %i2
	or %g5, %i1, %i5
	srl %g1, 29, %g4
	add %i5, %o0, %i3
	or %g4, %i2, %i4
	add %o1, %i3, %i1
	st	%i5, [%fp+1927]
	add %i4, %i0, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1719]
	sll %i1, %i3, %g5
	add %l3, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %o1
	srl %g1, %i1, %i3
	add %i5, %l6, %g4
	or %i2, %i3, %l3
	add %g4, %o1, %g1
	sll %g1, 3, %g5
	add %i4, %l6, %i3
	srl %g1, 29, %i2
	add %i3, %l3, %i1
	sll %i1, 3, %g4
	or %i2, %g5, %i5
	srl %i1, 29, %g1
	add %i5, %o1, %i3
	or %g1, %g4, %i4
	add %l4, %i3, %g5
	st	%i5, [%fp+1931]
	add %i4, %l3, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1723]
	sll %g5, %i3, %l4
	add %l7, %g4, %i1
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1727], %g1
	sll %i1, %g4, %i2
	or %l4, %i3, %l4
	srl %i1, %g5, %i3
	add %i5, %l5, %g4
	or %i2, %i3, %l7
	add %g4, %l4, %i2
	sll %i2, 3, %g5
	add %i4, %g1, %i3
	srl %i2, 29, %g1
	add %i3, %l7, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %l4, %i3
	or %i2, %g4, %i4
	add %o0, %i3, %g5
	st	%i5, [%fp+1935]
	add %i4, %l7, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1727]
	sll %g5, %i3, %o0
	add %i0, %g4, %i1
	ld	[%fp+1939], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1731], %i2
	sll %i1, %g4, %g1
	or %o0, %i3, %o0
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l0
	add %g4, %o0, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l0, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o0, %i0
	or %i2, %g4, %i4
	add %o1, %i0, %g5
	st	%i5, [%fp+1939]
	add %i4, %l0, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1731]
	sll %g5, %i0, %o1
	add %l3, %g4, %i1
	ld	[%fp+1943], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1735], %i2
	sll %i1, %g4, %g1
	or %o1, %i3, %o1
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l3
	add %g4, %o1, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l3, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o1, %i0
	or %i2, %g4, %i4
	add %l4, %i0, %g5
	st	%i5, [%fp+1943]
	add %i4, %l3, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1735]
	sll %g5, %i0, %l4
	add %l7, %g4, %i1
	ld	[%fp+1947], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1739], %i2
	sll %i1, %g4, %g1
	or %l4, %i3, %l4
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l7
	add %g4, %l4, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l7, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %l4, %i0
	or %i2, %g4, %i4
	add %o0, %i0, %g5
	st	%i5, [%fp+1947]
	add %i4, %l7, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1739]
	sll %g5, %i0, %o0
	add %l0, %g4, %i1
	ld	[%fp+1951], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1743], %i2
	sll %i1, %g4, %g1
	or %o0, %i3, %o0
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l0
	add %g4, %o0, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l0, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o0, %i0
	or %i2, %g4, %i4
	add %o1, %i0, %g5
	st	%i5, [%fp+1951]
	add %i4, %l0, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1743]
	sll %g5, %i0, %o1
	add %l3, %g4, %i1
	ld	[%fp+1955], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1747], %i2
	sll %i1, %g4, %g1
	or %o1, %i3, %o1
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l3
	add %g4, %o1, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l3, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o1, %i0
	or %i2, %g4, %i4
	add %l4, %i0, %g5
	st	%i5, [%fp+1955]
	add %i4, %l3, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1747]
	sll %g5, %i0, %l4
	add %l7, %g4, %i1
	ld	[%fp+1959], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1751], %i2
	sll %i1, %g4, %g1
	or %l4, %i3, %l4
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l7
	add %g4, %l4, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l7, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %l4, %i0
	or %i2, %g4, %i4
	add %o0, %i0, %g5
	st	%i5, [%fp+1959]
	add %i4, %l7, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1751]
	sll %g5, %i0, %o0
	add %l0, %g4, %i1
	ld	[%fp+1963], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1755], %i2
	sll %i1, %g4, %g1
	or %o0, %i3, %o0
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l0
	add %g4, %o0, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l0, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o0, %i0
	or %i2, %g4, %i4
	add %o1, %i0, %g5
	st	%i5, [%fp+1963]
	add %i4, %l0, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1755]
	sll %g5, %i0, %o1
	add %l3, %g4, %i1
	ld	[%fp+1967], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1759], %i2
	sll %i1, %g4, %g1
	or %o1, %i3, %o1
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l3
	add %g4, %o1, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l3, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o1, %i0
	or %i2, %g4, %i4
	add %l4, %i0, %g5
	st	%i5, [%fp+1967]
	add %i4, %l3, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1759]
	sll %g5, %i0, %l4
	add %l7, %g4, %i1
	ld	[%fp+1971], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1763], %i2
	sll %i1, %g4, %g1
	or %l4, %i3, %l4
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l7
	add %g4, %l4, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l7, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %l4, %i0
	or %i2, %g4, %i4
	add %o0, %i0, %g5
	st	%i5, [%fp+1971]
	add %i4, %l7, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1763]
	sll %g5, %i0, %o0
	add %l0, %g4, %i1
	ld	[%fp+1975], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1767], %i2
	sll %i1, %g4, %g1
	or %o0, %i3, %o0
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l0
	add %g4, %o0, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l0, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o0, %i0
	or %i2, %g4, %i4
	add %o1, %i0, %g5
	st	%i5, [%fp+1975]
	add %i4, %l0, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1767]
	sll %g5, %i0, %o1
	add %l3, %g4, %i1
	ld	[%fp+1979], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1771], %i2
	sll %i1, %g4, %g1
	or %o1, %i3, %o1
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l3
	add %g4, %o1, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l3, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o1, %i0
	or %i2, %g4, %i4
	add %l4, %i0, %g5
	st	%i5, [%fp+1979]
	add %i4, %l3, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1771]
	sll %g5, %i0, %l4
	add %l7, %g4, %i1
	ld	[%fp+1983], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1775], %i2
	sll %i1, %g4, %g1
	or %l4, %i3, %l4
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l7
	add %g4, %l4, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l7, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %l4, %i0
	or %i2, %g4, %i4
	add %o0, %i0, %g5
	st	%i5, [%fp+1983]
	add %i4, %l7, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1775]
	sll %g5, %i0, %o0
	add %l0, %g4, %i1
	ld	[%fp+1987], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1779], %i2
	sll %i1, %g4, %g1
	or %o0, %i3, %o0
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l0
	add %g4, %o0, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l0, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o0, %i0
	or %i2, %g4, %i4
	add %o1, %i0, %g5
	st	%i5, [%fp+1987]
	add %i4, %l0, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1779]
	sll %g5, %i0, %o1
	add %l3, %g4, %i1
	ld	[%fp+1991], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1783], %i2
	sll %i1, %g4, %g1
	or %o1, %i3, %o1
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l3
	add %g4, %o1, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l3, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o1, %i0
	or %i2, %g4, %i4
	add %l4, %i0, %g5
	st	%i5, [%fp+1991]
	add %i4, %l3, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1783]
	sll %g5, %i0, %l4
	add %l7, %g4, %i1
	ld	[%fp+1995], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1787], %i2
	sll %i1, %g4, %g1
	or %l4, %i3, %l4
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l7
	add %g4, %l4, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l7, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %l4, %i0
	or %i2, %g4, %i4
	add %o0, %i0, %g5
	st	%i5, [%fp+1995]
	add %i4, %l7, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1787]
	sll %g5, %i0, %o0
	add %l0, %g4, %i1
	ld	[%fp+1999], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1791], %i2
	sll %i1, %g4, %g1
	or %o0, %i3, %o0
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l0
	add %g4, %o0, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l0, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o0, %i0
	or %i2, %g4, %i4
	add %o1, %i0, %g5
	st	%i5, [%fp+1999]
	add %i4, %l0, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1791]
	sll %g5, %i0, %o1
	add %l3, %g4, %i1
	ld	[%fp+2003], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1795], %i2
	sll %i1, %g4, %g1
	or %o1, %i3, %o1
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l3
	add %g4, %o1, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l3, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o1, %i0
	or %i2, %g4, %i4
	add %l4, %i0, %g5
	st	%i5, [%fp+2003]
	add %i4, %l3, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1795]
	sll %g5, %i0, %l4
	add %l7, %g4, %i1
	ld	[%fp+2007], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1799], %i2
	sll %i1, %g4, %g1
	or %l4, %i3, %l4
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l7
	add %g4, %l4, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l7, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %l4, %i0
	or %i2, %g4, %i4
	add %o0, %i0, %g5
	st	%i5, [%fp+2007]
	add %i4, %l7, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1799]
	sll %g5, %i0, %o0
	add %l0, %g4, %i1
	ld	[%fp+2011], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1803], %i2
	sll %i1, %g4, %g1
	or %o0, %i3, %o0
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l0
	add %g4, %o0, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l0, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o0, %i0
	or %i2, %g4, %i4
	add %o1, %i0, %g5
	st	%i5, [%fp+2011]
	add %i4, %l0, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1803]
	sll %g5, %i0, %o1
	add %l3, %g4, %i1
	ld	[%fp+2015], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1807], %i2
	sll %i1, %g4, %g1
	or %o1, %i3, %o1
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l3
	add %g4, %o1, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g1
	add %i3, %l3, %i1
	sll %i1, 3, %g4
	or %g1, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o1, %i0
	or %i2, %g4, %i4
	add %l4, %i0, %g5
	st	%i5, [%fp+2015]
	add %i4, %l3, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1807]
	sll %g5, %i0, %l4
	add %l7, %g4, %i1
	ld	[%fp+2019], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1811], %i2
	sll %i1, %g4, %g1
	or %l4, %i3, %l4
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %g1, %i3, %l7
	add %g4, %l4, %i3
	sll %i3, 3, %g5
	add %i4, %i2, %i1
	srl %i3, 29, %g1
	add %i1, %l7, %g4
	sll %g4, 3, %i2
	or %g1, %g5, %i5
	srl %g4, 29, %i0
	add %i5, %l4, %i3
	or %i0, %i2, %i4
	add %o0, %i3, %i1
	st	%i5, [%fp+2019]
	add %i4, %l7, %g4
	sub %l1, %i3, %i2
	st	%i4, [%fp+1811]
	sll %i1, %i3, %g5
	add %l0, %g4, %g1
	ld	[%fp+2023], %i0
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	ld	[%fp+1815], %i2
	sll %g1, %g4, %l0
	or %g5, %i3, %o0
	srl %g1, %i1, %g4
	add %i5, %i0, %g1
	or %l0, %g4, %l0
	add %g1, %o0, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i3
	srl %i0, 29, %g4
	add %i3, %l0, %i1
	sll %i1, 3, %g1
	or %g4, %g5, %i5
	srl %i1, 29, %i2
	add %i5, %o0, %i0
	or %i2, %g1, %i4
	add %o1, %i0, %g5
	st	%i5, [%fp+2023]
	add %i4, %l0, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1815]
	sll %g5, %i0, %o1
	add %l3, %g4, %i1
	ld	[%fp+2027], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1819], %i2
	sll %i1, %g4, %g1
	or %o1, %i3, %o1
	srl %i1, %g5, %g4
	add %i5, %i0, %i3
	or %g1, %g4, %l3
	add %i3, %o1, %i0
	sll %i0, 3, %g5
	add %i4, %i2, %i1
	srl %i0, 29, %g1
	add %i1, %l3, %g4
	sll %g4, 3, %i2
	or %g1, %g5, %i5
	srl %g4, 29, %i3
	add %i5, %o1, %i0
	or %i3, %i2, %i4
	add %l4, %i0, %i1
	st	%i5, [%fp+2027]
	add %i4, %l3, %g4
	sub %l1, %i0, %i2
	st	%i4, [%fp+1819]
	sll %i1, %i0, %g5
	add %l7, %g4, %g1
	ld	[%fp+1927], %i0
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	ld	[%fp+1719], %i2
	sll %g1, %g4, %l7
	or %g5, %i3, %l4
	srl %g1, %i1, %g4
	add %i5, %i0, %g1
	or %l7, %g4, %l7
	add %g1, %l4, %i1
	sll %i1, 3, %i0
	add %i4, %i2, %i3
	srl %i1, 29, %g5
	add %i3, %l7, %g4
	sll %g4, 3, %g1
	or %g5, %i0, %i5
	srl %g4, 29, %i2
	add %i5, %l4, %i0
	or %i2, %g1, %i4
	add %o0, %i0, %g5
	ld	[%fp+1443], %i1
	add %i4, %l7, %g4
	sub %l1, %i0, %i2
	mov	%i1, %o4
	sll %g5, %i0, %o0
	add %l0, %g4, %i1
	ld	[%fp+1931], %i0
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	ld	[%fp+1723], %g1
	sll %i1, %g4, %l0
	or %o0, %i3, %o0
	ld	[%fp+1427], %i2
	add	%o4, %i5, %o5
	add	%o4, %i4, %o4
	srl %i1, %g5, %i3
	add %i5, %i0, %g4
	or %l0, %i3, %l0
	add %g4, %o0, %i0
	sll %i0, 3, %i1
	add %i4, %g1, %i3
	srl %i0, 29, %g5
	add %i3, %l0, %g1
	sll %g1, 3, %g4
	or %g5, %i1, %i5
	srl %g1, 29, %i0
	add	%i2, %i5, %o3
	or %i0, %g4, %i4
	add	%i2, %i4, %o2
	add %i5, %o0, %i3
	add %i4, %l0, %g4
	ld	[%fp+1935], %i0
	add %o1, %i3, %i1
	sub %l1, %i3, %i2
	ld	[%fp+1727], %o7
	sll %i1, %i3, %g5
	add %l3, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %o1
	srl %g1, %i1, %g4
	add %i5, %i0, %i3
	or %i2, %g4, %l3
	add %i3, %o1, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l3, %g4
	sll %g4, 3, %i1
	xor %o5, %o3, %o5
	srl %g4, 29, %i4
	xor %o4, %o2, %o4
	sll %o5, %o3, %g1
	sub %l1, %o3, %i2
	sll %o4, %o2, %g4
	sub %l1, %o2, %g5
	srl %o5, %i2, %i0
	or %i5, %i3, %i5
	srl %o4, %g5, %i2
	or %i4, %i1, %i4
	or %i0, %g1, %i3
	or %i2, %g4, %i0
	add %i3, %i5, %o5
	add %i0, %i4, %o4
	add %i5, %o1, %i1
	add %i4, %l3, %g4
	ld	[%fp+1939], %i0
	add %l4, %i1, %i3
	sub %l1, %i1, %i2
	ld	[%fp+1731], %o7
	sll %i3, %i1, %g5
	add %l7, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %l4
	srl %i1, %i2, %i3
	add %i5, %i0, %i2
	or %g1, %i3, %l7
	add %i2, %l4, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l7, %g4
	sll %g4, 3, %i1
	xor %o3, %o5, %o3
	srl %g4, 29, %i4
	xor %o2, %o4, %o2
	sll %o3, %o5, %g1
	sub %l1, %o5, %i2
	sll %o2, %o4, %g4
	sub %l1, %o4, %g5
	srl %o3, %i2, %i0
	or %i5, %i3, %i5
	srl %o2, %g5, %i3
	or %i4, %i1, %i4
	or %i0, %g1, %i2
	or %i3, %g4, %i0
	add %i2, %i5, %o3
	add %i0, %i4, %o2
	add %i5, %l4, %i3
	add %i4, %l7, %g4
	ld	[%fp+1943], %i0
	add %o0, %i3, %i1
	sub %l1, %i3, %i2
	ld	[%fp+1735], %o7
	sll %i1, %i3, %g5
	add %l0, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %o0
	srl %g1, %i1, %g4
	add %i5, %i0, %i3
	or %i2, %g4, %l0
	add %i3, %o0, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l0, %g4
	sll %g4, 3, %i1
	xor %o5, %o3, %o5
	srl %g4, 29, %i4
	xor %o4, %o2, %o4
	sll %o5, %o3, %g1
	sub %l1, %o3, %i2
	sll %o4, %o2, %g4
	sub %l1, %o2, %g5
	srl %o5, %i2, %i0
	or %i5, %i3, %i5
	srl %o4, %g5, %i2
	or %i4, %i1, %i4
	or %i0, %g1, %i3
	or %i2, %g4, %i0
	add %i3, %i5, %o5
	add %i0, %i4, %o4
	add %i5, %o0, %i1
	add %i4, %l0, %g4
	ld	[%fp+1947], %i0
	add %o1, %i1, %i3
	sub %l1, %i1, %i2
	ld	[%fp+1739], %o7
	sll %i3, %i1, %g5
	add %l3, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %o1
	srl %i1, %i2, %i3
	add %i5, %i0, %i2
	or %g1, %i3, %l3
	add %i2, %o1, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l3, %g4
	sll %g4, 3, %i1
	xor %o3, %o5, %o3
	srl %g4, 29, %i4
	xor %o2, %o4, %o2
	sll %o3, %o5, %g1
	sub %l1, %o5, %i2
	sll %o2, %o4, %g4
	sub %l1, %o4, %g5
	srl %o3, %i2, %i0
	or %i5, %i3, %i5
	srl %o2, %g5, %i3
	or %i4, %i1, %i4
	or %i0, %g1, %i2
	or %i3, %g4, %i0
	add %i2, %i5, %o3
	add %i0, %i4, %o2
	add %i5, %o1, %i3
	add %i4, %l3, %g4
	ld	[%fp+1951], %i0
	add %l4, %i3, %i1
	sub %l1, %i3, %i2
	ld	[%fp+1743], %o7
	sll %i1, %i3, %g5
	add %l7, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %l4
	srl %g1, %i1, %g4
	add %i5, %i0, %i3
	or %i2, %g4, %l7
	add %i3, %l4, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l7, %g4
	sll %g4, 3, %i1
	xor %o5, %o3, %o5
	srl %g4, 29, %i4
	xor %o4, %o2, %o4
	sll %o5, %o3, %g1
	sub %l1, %o3, %i2
	sll %o4, %o2, %g4
	sub %l1, %o2, %g5
	srl %o5, %i2, %i0
	or %i5, %i3, %i5
	srl %o4, %g5, %i2
	or %i4, %i1, %i4
	or %i0, %g1, %i3
	or %i2, %g4, %i0
	add %i3, %i5, %o5
	add %i0, %i4, %o4
	add %i5, %l4, %i1
	add %i4, %l7, %g4
	ld	[%fp+1955], %i0
	add %o0, %i1, %i3
	sub %l1, %i1, %i2
	ld	[%fp+1747], %o7
	sll %i3, %i1, %g5
	add %l0, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %o0
	srl %i1, %i2, %i3
	add %i5, %i0, %i2
	or %g1, %i3, %l0
	add %i2, %o0, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l0, %g4
	sll %g4, 3, %i1
	xor %o3, %o5, %o3
	srl %g4, 29, %i4
	xor %o2, %o4, %o2
	sll %o3, %o5, %g1
	sub %l1, %o5, %i2
	sll %o2, %o4, %g4
	sub %l1, %o4, %g5
	srl %o3, %i2, %i0
	or %i5, %i3, %i5
	srl %o2, %g5, %i3
	or %i4, %i1, %i4
	or %i0, %g1, %i2
	or %i3, %g4, %i0
	add %i2, %i5, %o3
	add %i0, %i4, %o2
	add %i5, %o0, %i3
	add %i4, %l0, %g4
	ld	[%fp+1959], %i0
	add %o1, %i3, %i1
	sub %l1, %i3, %i2
	ld	[%fp+1751], %o7
	sll %i1, %i3, %g5
	add %l3, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %o1
	srl %g1, %i1, %g4
	add %i5, %i0, %i3
	or %i2, %g4, %l3
	add %i3, %o1, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l3, %g4
	sll %g4, 3, %i1
	xor %o5, %o3, %o5
	srl %g4, 29, %i4
	xor %o4, %o2, %o4
	sll %o5, %o3, %g1
	sub %l1, %o3, %i2
	sll %o4, %o2, %g4
	sub %l1, %o2, %g5
	srl %o5, %i2, %i0
	or %i5, %i3, %i5
	srl %o4, %g5, %i2
	or %i4, %i1, %i4
	or %i0, %g1, %i3
	or %i2, %g4, %i0
	add %i3, %i5, %o5
	add %i0, %i4, %o4
	add %i5, %o1, %i1
	add %i4, %l3, %g4
	ld	[%fp+1963], %i0
	add %l4, %i1, %i3
	sub %l1, %i1, %i2
	ld	[%fp+1755], %o7
	sll %i3, %i1, %g5
	add %l7, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %l4
	srl %i1, %i2, %i3
	add %i5, %i0, %i2
	or %g1, %i3, %l7
	add %i2, %l4, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l7, %g4
	sll %g4, 3, %i1
	xor %o3, %o5, %o3
	srl %g4, 29, %i4
	xor %o2, %o4, %o2
	sll %o3, %o5, %g1
	sub %l1, %o5, %i2
	sll %o2, %o4, %g4
	sub %l1, %o4, %g5
	srl %o3, %i2, %i0
	or %i5, %i3, %i5
	srl %o2, %g5, %i3
	or %i4, %i1, %i4
	or %i0, %g1, %i2
	or %i3, %g4, %i0
	add %i2, %i5, %o3
	add %i0, %i4, %o2
	add %i5, %l4, %i3
	add %i4, %l7, %g4
	ld	[%fp+1967], %i0
	add %o0, %i3, %i1
	sub %l1, %i3, %i2
	ld	[%fp+1759], %o7
	sll %i1, %i3, %g5
	add %l0, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %o0
	srl %g1, %i1, %g4
	add %i5, %i0, %i3
	or %i2, %g4, %l0
	add %i3, %o0, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l0, %g4
	sll %g4, 3, %i1
	xor %o5, %o3, %o5
	srl %g4, 29, %i4
	xor %o4, %o2, %o4
	sll %o5, %o3, %g1
	sub %l1, %o3, %i2
	sll %o4, %o2, %g4
	sub %l1, %o2, %g5
	srl %o5, %i2, %i0
	or %i5, %i3, %i5
	srl %o4, %g5, %i2
	or %i4, %i1, %i4
	or %i0, %g1, %i3
	or %i2, %g4, %i0
	add %i3, %i5, %o5
	add %i0, %i4, %o4
	add %i5, %o0, %i1
	add %i4, %l0, %g4
	ld	[%fp+1971], %i0
	add %o1, %i1, %i3
	sub %l1, %i1, %i2
	ld	[%fp+1763], %o7
	sll %i3, %i1, %g5
	add %l3, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %o1
	srl %i1, %i2, %i3
	add %i5, %i0, %i2
	or %g1, %i3, %l3
	add %i2, %o1, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l3, %g4
	sll %g4, 3, %i1
	xor %o3, %o5, %o3
	srl %g4, 29, %i4
	xor %o2, %o4, %o2
	sll %o3, %o5, %g1
	sub %l1, %o5, %i2
	sll %o2, %o4, %g4
	sub %l1, %o4, %g5
	srl %o3, %i2, %i0
	or %i5, %i3, %i5
	srl %o2, %g5, %i3
	or %i4, %i1, %i4
	or %i0, %g1, %i2
	or %i3, %g4, %i0
	add %i2, %i5, %o3
	add %i0, %i4, %o2
	add %i5, %o1, %i3
	add %i4, %l3, %g4
	ld	[%fp+1975], %i0
	add %l4, %i3, %i1
	sub %l1, %i3, %i2
	ld	[%fp+1767], %o7
	sll %i1, %i3, %g5
	add %l7, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %l4
	srl %g1, %i1, %g4
	add %i5, %i0, %i3
	or %i2, %g4, %l7
	add %i3, %l4, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l7, %g4
	sll %g4, 3, %i1
	xor %o5, %o3, %o5
	srl %g4, 29, %i4
	xor %o4, %o2, %o4
	sll %o5, %o3, %g1
	sub %l1, %o3, %i2
	sll %o4, %o2, %g4
	sub %l1, %o2, %g5
	srl %o5, %i2, %i0
	or %i5, %i3, %i5
	srl %o4, %g5, %i2
	or %i4, %i1, %i4
	or %i0, %g1, %i3
	or %i2, %g4, %i0
	add %i3, %i5, %o5
	add %i0, %i4, %o4
	add %i5, %l4, %i1
	add %i4, %l7, %g4
	ld	[%fp+1979], %i0
	add %o0, %i1, %i3
	sub %l1, %i1, %i2
	ld	[%fp+1771], %o7
	sll %i3, %i1, %g5
	add %l0, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %o0
	srl %i1, %i2, %i3
	add %i5, %i0, %i2
	or %g1, %i3, %l0
	add %i2, %o0, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l0, %g4
	sll %g4, 3, %i1
	xor %o3, %o5, %o3
	srl %g4, 29, %i4
	xor %o2, %o4, %o2
	sll %o3, %o5, %g1
	sub %l1, %o5, %i2
	sll %o2, %o4, %g4
	sub %l1, %o4, %g5
	srl %o3, %i2, %i0
	or %i5, %i3, %i5
	srl %o2, %g5, %i3
	or %i4, %i1, %i4
	or %i0, %g1, %i2
	or %i3, %g4, %i0
	add %i2, %i5, %o3
	add %i0, %i4, %o2
	add %i5, %o0, %i3
	add %i4, %l0, %g4
	ld	[%fp+1983], %i0
	add %o1, %i3, %i1
	sub %l1, %i3, %i2
	ld	[%fp+1775], %o7
	sll %i1, %i3, %g5
	add %l3, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %o1
	srl %g1, %i1, %g4
	add %i5, %i0, %i3
	or %i2, %g4, %l3
	add %i3, %o1, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l3, %g4
	sll %g4, 3, %i1
	xor %o5, %o3, %o5
	srl %g4, 29, %i4
	xor %o4, %o2, %o4
	sll %o5, %o3, %g1
	sub %l1, %o3, %i2
	sll %o4, %o2, %g4
	sub %l1, %o2, %g5
	srl %o5, %i2, %i0
	or %i5, %i3, %i5
	srl %o4, %g5, %i2
	or %i4, %i1, %i4
	or %i0, %g1, %i3
	or %i2, %g4, %i0
	add %i3, %i5, %o5
	add %i0, %i4, %o4
	add %i5, %o1, %i1
	add %i4, %l3, %g4
	ld	[%fp+1987], %i0
	add %l4, %i1, %i3
	sub %l1, %i1, %i2
	ld	[%fp+1779], %o7
	sll %i3, %i1, %g5
	add %l7, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %l4
	srl %i1, %i2, %i3
	add %i5, %i0, %i2
	or %g1, %i3, %l7
	add %i2, %l4, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l7, %g4
	sll %g4, 3, %i1
	xor %o3, %o5, %o3
	srl %g4, 29, %i4
	xor %o2, %o4, %o2
	sll %o3, %o5, %g1
	sub %l1, %o5, %i2
	sll %o2, %o4, %g4
	sub %l1, %o4, %g5
	srl %o3, %i2, %i0
	or %i5, %i3, %i5
	srl %o2, %g5, %i3
	or %i4, %i1, %i4
	or %i0, %g1, %i2
	or %i3, %g4, %i0
	add %i2, %i5, %o3
	add %i0, %i4, %o2
	add %i5, %l4, %i3
	add %i4, %l7, %g4
	ld	[%fp+1991], %i0
	add %o0, %i3, %i1
	sub %l1, %i3, %i2
	ld	[%fp+1783], %o7
	sll %i1, %i3, %g5
	add %l0, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %o0
	srl %g1, %i1, %g4
	add %i5, %i0, %i3
	or %i2, %g4, %l0
	add %i3, %o0, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l0, %g4
	sll %g4, 3, %i1
	xor %o5, %o3, %o5
	srl %g4, 29, %i4
	xor %o4, %o2, %o4
	sll %o5, %o3, %g1
	sub %l1, %o3, %i2
	sll %o4, %o2, %g4
	sub %l1, %o2, %g5
	srl %o5, %i2, %i0
	or %i5, %i3, %i5
	srl %o4, %g5, %i2
	or %i4, %i1, %i4
	or %i0, %g1, %i3
	or %i2, %g4, %i0
	add %i3, %i5, %o5
	add %i0, %i4, %o4
	add %i5, %o0, %i1
	add %i4, %l0, %g4
	ld	[%fp+1995], %i0
	add %o1, %i1, %i3
	sub %l1, %i1, %i2
	ld	[%fp+1787], %o7
	sll %i3, %i1, %g5
	add %l3, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %o1
	srl %i1, %i2, %i3
	add %i5, %i0, %i2
	or %g1, %i3, %l3
	add %i2, %o1, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l3, %g4
	sll %g4, 3, %i1
	xor %o3, %o5, %o3
	srl %g4, 29, %i4
	xor %o2, %o4, %o2
	sll %o3, %o5, %g1
	sub %l1, %o5, %i2
	sll %o2, %o4, %g4
	sub %l1, %o4, %g5
	srl %o3, %i2, %i0
	or %i5, %i3, %i5
	srl %o2, %g5, %i3
	or %i4, %i1, %i4
	or %i0, %g1, %i2
	or %i3, %g4, %i0
	add %i2, %i5, %o3
	add %i0, %i4, %o2
	add %i5, %o1, %i3
	add %i4, %l3, %g4
	ld	[%fp+1999], %i0
	add %l4, %i3, %i1
	sub %l1, %i3, %i2
	ld	[%fp+1791], %o7
	sll %i1, %i3, %g5
	add %l7, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %l4
	srl %g1, %i1, %g4
	add %i5, %i0, %i3
	or %i2, %g4, %l7
	add %i3, %l4, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l7, %g4
	sll %g4, 3, %i1
	xor %o5, %o3, %o5
	srl %g4, 29, %i4
	xor %o4, %o2, %o4
	sll %o5, %o3, %g1
	sub %l1, %o3, %i2
	sll %o4, %o2, %g4
	sub %l1, %o2, %g5
	srl %o5, %i2, %i0
	or %i5, %i3, %i5
	srl %o4, %g5, %i2
	or %i4, %i1, %i4
	or %i0, %g1, %i3
	or %i2, %g4, %i0
	add %i3, %i5, %o5
	add %i0, %i4, %o4
	add %i5, %l4, %i1
	add %i4, %l7, %g4
	ld	[%fp+2003], %i0
	add %o0, %i1, %i3
	sub %l1, %i1, %i2
	ld	[%fp+1795], %o7
	sll %i3, %i1, %g5
	add %l0, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %o0
	srl %i1, %i2, %i3
	add %i5, %i0, %i2
	or %g1, %i3, %l0
	add %i2, %o0, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l0, %g4
	sll %g4, 3, %i1
	xor %o3, %o5, %o3
	srl %g4, 29, %i4
	xor %o2, %o4, %o2
	sll %o3, %o5, %g1
	sub %l1, %o5, %i2
	sll %o2, %o4, %g4
	sub %l1, %o4, %g5
	srl %o3, %i2, %i0
	or %i5, %i3, %i5
	srl %o2, %g5, %i3
	or %i4, %i1, %i4
	or %i0, %g1, %i2
	or %i3, %g4, %i0
	add %i2, %i5, %o3
	add %i0, %i4, %o2
	add %i5, %o0, %i3
	add %i4, %l0, %g4
	ld	[%fp+2007], %i0
	add %o1, %i3, %i1
	sub %l1, %i3, %i2
	ld	[%fp+1799], %o7
	sll %i1, %i3, %g5
	add %l3, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %o1
	srl %g1, %i1, %g4
	add %i5, %i0, %i3
	or %i2, %g4, %l3
	add %i3, %o1, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l3, %g4
	sll %g4, 3, %i1
	xor %o5, %o3, %o5
	srl %g4, 29, %i4
	xor %o4, %o2, %o4
	sll %o5, %o3, %g1
	sub %l1, %o3, %i2
	sll %o4, %o2, %g4
	sub %l1, %o2, %g5
	srl %o5, %i2, %i0
	or %i5, %i3, %i5
	srl %o4, %g5, %i2
	or %i4, %i1, %i4
	or %i0, %g1, %i3
	or %i2, %g4, %i0
	add %i3, %i5, %o5
	add %i0, %i4, %o4
	add %i5, %o1, %i1
	add %i4, %l3, %g4
	ld	[%fp+2011], %i0
	add %l4, %i1, %i3
	sub %l1, %i1, %i2
	ld	[%fp+1803], %o7
	sll %i3, %i1, %g5
	add %l7, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %l4
	srl %i1, %i2, %i3
	add %i5, %i0, %i2
	or %g1, %i3, %l7
	add %i2, %l4, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l7, %g4
	sll %g4, 3, %i1
	xor %o3, %o5, %o3
	srl %g4, 29, %i4
	xor %o2, %o4, %o2
	sll %o3, %o5, %g1
	sub %l1, %o5, %i2
	sll %o2, %o4, %g4
	sub %l1, %o4, %g5
	srl %o3, %i2, %i0
	or %i5, %i3, %i5
	srl %o2, %g5, %i3
	or %i4, %i1, %i4
	or %i0, %g1, %i2
	or %i3, %g4, %i0
	add %i2, %i5, %o3
	add %i0, %i4, %o2
	add %i5, %l4, %i3
	add %i4, %l7, %g4
	ld	[%fp+2015], %i0
	add %o0, %i3, %i1
	sub %l1, %i3, %i2
	ld	[%fp+1807], %o7
	sll %i1, %i3, %g5
	add %l0, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %o0
	srl %g1, %i1, %g4
	add %i5, %i0, %i3
	or %i2, %g4, %l0
	add %i3, %o0, %g5
	sll %g5, 3, %i3
	add %i4, %o7, %g1
	srl %g5, 29, %i5
	add %g1, %l0, %g4
	sll %g4, 3, %i1
	xor %o5, %o3, %o5
	srl %g4, 29, %i4
	xor %o4, %o2, %o4
	sll %o5, %o3, %g1
	sub %l1, %o3, %i2
	sll %o4, %o2, %g4
	sub %l1, %o2, %g5
	srl %o5, %i2, %i0
	or %i5, %i3, %i5
	srl %o4, %g5, %i2
	or %i4, %i1, %i4
	or %i0, %g1, %i3
	or %i2, %g4, %i0
	add %i3, %i5, %o5
	add %i0, %i4, %o4
	add %i5, %o0, %i1
	add %i4, %l0, %g4
	ld	[%fp+2019], %i0
	add %o1, %i1, %i3
	sub %l1, %i1, %i2
	ld	[%fp+1811], %o7
	sll %i3, %i1, %g5
	add %l3, %g4, %i1
	srl %i3, %i2, %i3
	sub %l1, %g4, %i2
	sll %i1, %g4, %g1
	or %g5, %i3, %o1
	srl %i1, %i2, %i3
	add %i5, %i0, %i2
	or %g1, %i3, %i0
	add %i2, %o1, %i1
	sll %i1, 3, %i3
	add %i4, %o7, %g1
	srl %i1, 29, %i5
	add %g1, %i0, %g4
	sll %g4, 3, %i1
	xor %o3, %o5, %o3
	srl %g4, 29, %i4
	xor %o2, %o4, %o2
	sll %o3, %o5, %g1
	sub %l1, %o5, %i2
	sll %o2, %o4, %g4
	sub %l1, %o4, %g5
	srl %o3, %i2, %o3
	or %i5, %i3, %i5
	srl %o2, %g5, %i3
	or %i4, %i1, %i4
	or %o3, %g1, %g5
	or %i3, %g4, %i2
	add %g5, %i5, %o3
	add %i2, %i4, %o2
	add %i5, %o1, %i3
	add %i4, %i0, %g4
	ld	[%fp+2023], %i0
	add %l4, %i3, %i1
	sub %l1, %i3, %i2
	ld	[%fp+1815], %o7
	sll %i1, %i3, %g5
	add %l7, %g4, %g1
	srl %i1, %i2, %i3
	sub %l1, %g4, %i1
	sll %g1, %g4, %i2
	or %g5, %i3, %l4
	srl %g1, %i1, %g4
	add %i5, %i0, %i3
	or %i2, %g4, %l7
	add %i3, %l4, %g1
	sll %g1, 3, %i3
	add %i4, %o7, %i2
	srl %g1, 29, %i5
	add %i2, %l7, %g5
	sll %g5, 3, %i1
	xor %o5, %o3, %o5
	srl %g5, 29, %i4
	xor %o4, %o2, %o4
	sll %o5, %o3, %g1
	sub %l1, %o3, %i2
	sll %o4, %o2, %i0
	sub %l1, %o2, %g5
	srl %o5, %i2, %g4
	or %i5, %i3, %i5
	srl %o4, %g5, %i3
	or %i4, %i1, %i4
	or %g4, %g1, %i1
	or %i3, %i0, %g1
	add %i1, %i5, %o5
	add %g1, %i4, %o4
	ld	[%fp+1411], %g4
	xor	%o5, %g4, %i2
	xor	%o4, %g4, %i0
	subcc	%g0, %i2, %g0
	subx	%g0, -1, %i3
	subcc	%g0, %i0, %g0
	subx	%g0, -1, %g5
	orcc	%i3, %g5, %g0
	be,a,pt	%icc, .LL29
	ld	[%fp+1411], %i5
	add %i5, %l4, %i1
	add %i4, %l7, %g4
	ld	[%fp+2027], %i0
	add %o0, %i1, %g5
	sub %l1, %i1, %i2
	ld	[%fp+1819], %o7
	sll %g5, %i1, %o0
	add %l0, %g4, %i1
	srl %g5, %i2, %i3
	sub %l1, %g4, %g5
	sll %i1, %g4, %g1
	or %o0, %i3, %i2
	srl %i1, %g5, %g4
	add %i5, %i0, %i3
	or %g1, %g4, %g5
	add %i3, %i2, %i1
	sll %i1, 3, %i3
	add %i4, %o7, %g1
	srl %i1, 29, %i5
	add %g1, %g5, %g4
	sll %g4, 3, %i1
	xor %o3, %o5, %o3
	srl %g4, 29, %i4
	xor %o2, %o4, %i0
	sll %o3, %o5, %g1
	sub %l1, %o5, %i2
	sll %i0, %o4, %g4
	sub %l1, %o4, %g5
	srl %o3, %i2, %o3
	or %i5, %i3, %i2
	srl %i0, %g5, %o2
	or %i4, %i1, %i0
	or %o3, %g1, %i3
	or %o2, %g4, %g5
	add %i3, %i2, %o3
	add %g5, %i0, %o2
	ld	[%fp+1411], %i5
.LL29:
	cmp	%o5, %i5
	be,a,pn	%icc, .LL27
	ld	[%fp+1371], %i0
.LL11:
	ld	[%fp+1411], %i3
	cmp	%o4, %i3
	be,pn	%icc, .LL28
	ld	[%fp+1371], %g5
.LL13:
	ld	[%fp+1483], %g4
	add	%g4, 2, %i1
	andcc	%i1, 255, %g5
	bne,pt	%icc, .LL9
	st	%g5, [%fp+1483]
	ld	[%fp+1531], %i0
	sethi	%hi(16777216), %i5
	add	%i0, %i5, %g1
	sethi	%hi(-16777216), %i4
	andcc	%g1, %i4, %g0
	bne,pt	%icc, .LL16
	st	%g1, [%fp+1531]
	sethi	%hi(65536), %g5
	sethi	%hi(16776192), %i1
	add	%g1, %g5, %i3
	or	%i1, 1023, %i2
	and	%i3, %i2, %g4
	sethi	%hi(16711680), %i3
	andcc	%g4, %i3, %g0
	bne,pt	%icc, .LL16
	st	%g4, [%fp+1531]
	sethi	%hi(64512), %g1
	add	%g4, 256, %g4
	or	%g1, 1023, %i1
	or	%g1, 768, %i0
	and	%g4, %i1, %g4
	andcc	%g4, %i0, %g0
	bne,pt	%icc, .LL16
	st	%g4, [%fp+1531]
	add	%g4, 1, %g1
	andcc	%g1, 255, %g4
	bne,pt	%icc, .LL16
	st	%g4, [%fp+1531]
	ld	[%fp+1579], %g1
	add	%g1, %i5, %g4
	andcc	%g4, %i4, %g0
	bne,pt	%icc, .LL20
	st	%g4, [%fp+1579]
	add	%g4, %g5, %i4
	and	%i4, %i2, %g1
	andcc	%g1, %i3, %g0
	bne,pt	%icc, .LL20
	st	%g1, [%fp+1579]
	add	%g1, 256, %i2
	and	%i2, %i1, %g1
	andcc	%g1, %i0, %g0
	bne,pt	%icc, .LL20
	st	%g1, [%fp+1579]
	add	%g1, 1, %g5
	and	%g5, 255, %i5
	st	%i5, [%fp+1579]
.LL20:
	sethi	%hi(-1089828864), %i2
	ld	[%fp+1579], %g4
	or	%i2, 797, %i5
	ld	[%fp+1303], %g1
	add	%g4, %i5, %i1
	sub	%g0, %i5, %i4
	srl	%i1, %i4, %i3
	sll	%i1, %i5, %i0
	or	%i0, %i3, %g5
	add	%g1, %i5, %i2
	st	%g5, [%fp+1595]
	ld	[%fp+1595], %i3
	add	%i2, %i3, %g4
	srl	%g4, 29, %i4
	sll	%g4, 3, %i1
	or	%i1, %i4, %l6
	add	%l6, %i3, %i0
	st	%i0, [%fp+1291]
.LL16:
	ld	[%fp+1291], %i2
	ld	[%fp+1531], %g4
	add	%g4, %i2, %i5
	sub	%g0, %i2, %g5
	srl	%i5, %g5, %i4
	sll	%i5, %i2, %i1
	ld	[%fp+1299], %g1
	or	%i1, %i4, %l2
	add	%g1, %l6, %i3
	add	%i3, %l2, %i0
	srl	%i0, 29, %g5
	sll	%i0, 3, %i5
	or	%i5, %g5, %l5
.LL9:
	ld	[%fp+1611], %i3
	addcc	%i3, -1, %i0
	bne,pt	%icc, .LL7
	st	%i0, [%fp+1611]
	ld	[%fp+1355], %l5
	ld	[%fp+1579], %i3
	srl	%l5, 24, %o3
	srl	%l5, 16, %o2
	srl	%l5, 8, %o1
	ld	[%fp+1531], %i0
	ldx	[%fp+2175], %l5
	ld	[%fp+1483], %l3
	ld	[%fp+1371], %l4
	ld	[%fp+1339], %l6
	ld	[%fp+1323], %l7
	srl	%i3, 24, %g1
	srl	%i3, 8, %g5
	srl	%i3, 16, %g4
	stb	%g1, [%l5+24]
	stb	%l7, [%l5+35]
	stb	%g4, [%l5+25]
	stb	%g5, [%l5+26]
	ld	[%fp+1579], %g5
	srl	%i0, 24, %i5
	srl	%i0, 16, %i4
	srl	%i0, 8, %i3
	stb	%g5, [%l5+27]
	stb	%i5, [%l5+20]
	stb	%i4, [%l5+21]
	stb	%i3, [%l5+22]
	ld	[%fp+1531], %i5
	srl	%l3, 24, %i2
	srl	%l3, 16, %i1
	srl	%l3, 8, %i0
	stb	%i0, [%l5+18]
	stb	%i5, [%l5+23]
	stb	%i2, [%l5+16]
	stb	%i1, [%l5+17]
	ld	[%fp+1483], %i2
	srl	%l4, 24, %o7
	srl	%l4, 16, %o5
	srl	%l4, 8, %o4
	stb	%i2, [%l5+19]
	stb	%o7, [%l5+28]
	stb	%o5, [%l5+29]
	stb	%o4, [%l5+30]
	ld	[%fp+1371], %i4
	srl	%l6, 24, %o0
	stb	%i4, [%l5+31]
	stb	%o3, [%l5+40]
	stb	%o2, [%l5+41]
	stb	%o1, [%l5+42]
	ld	[%fp+1355], %i1
	srl	%l6, 16, %l0
	stb	%i1, [%l5+43]
	srl	%l6, 8, %l1
	stb	%o0, [%l5+36]
	stb	%l0, [%l5+37]
	stb	%l1, [%l5+38]
	srl	%l7, 24, %l2
	ld	[%fp+1339], %g1
	srl	%l7, 16, %l3
	srl	%l7, 8, %l4
	stb	%g1, [%l5+39]
	stb	%l2, [%l5+32]
	stb	%l3, [%l5+33]
	stb	%l4, [%l5+34]
	ba,pt	%xcc, .LL1
	mov	1, %i0
.LL28:
	ld	[%fp+1395], %i4
	add	%g5, 1, %g4
	ld	[%fp+1295], %l7
	ld	[%fp+1531], %g1
	ld	[%fp+1579], %g5
	cmp	%o2, %i4
	st	%l7, [%fp+1323]
	st	%g4, [%fp+1371]
	st	%g1, [%fp+1339]
	bne,pt	%icc, .LL13
	st	%g5, [%fp+1355]
	ld	[%fp+1611], %i5
	ld	[%fp+1307], %g1
	add	%i5, %i5, %i4
	ld	[%fp+1339], %i3
	sub	%g1, %i4, %i2
	ldx	[%fp+2183], %g4
	srl	%g5, 16, %i1
	add	%i2, 1, %i0
	srl	%g5, 8, %i5
	st	%i0, [%g4]
	srl	%i3, 16, %i4
	srl	%i3, 8, %i2
	and	%i5, 255, %g5
	ld	[%fp+1355], %i3
	and	%i1, 255, %g4
	and	%i4, 255, %i5
	ld	[%fp+1483], %i1
	and	%i3, 0xff, %o2
	and	%i2, 255, %i4
	srl	%i3, 24, %o0
	ldx	[%fp+2175], %l3
	ld	[%fp+1339], %i0
	ld	[%fp+1371], %l1
	srl	%i1, 8, %i2
	srl	%i1, 24, %g1
	srl	%i1, 16, %i3
	and	%i0, 0xff, %o1
	srl	%i0, 24, %l0
	stb	%l7, [%l3+35]
	stb	%g1, [%l3+16]
	stb	%i3, [%l3+17]
	stb	%i2, [%l3+18]
	ld	[%fp+1483], %i2
	srl	%l1, 24, %i1
	srl	%l1, 16, %i0
	srl	%l1, 8, %o7
	srl	%l7, 24, %o5
	srl	%l7, 16, %o4
	srl	%l7, 8, %o3
	stb	%i2, [%l3+19]
	stb	%i1, [%l3+28]
	stb	%i0, [%l3+29]
	stb	%o7, [%l3+30]
	stb	%l1, [%l3+31]
	stb	%o0, [%l3+40]
	stb	%g4, [%l3+41]
	stb	%g5, [%l3+42]
	stb	%o2, [%l3+43]
	stb	%l0, [%l3+36]
	stb	%i5, [%l3+37]
	stb	%i4, [%l3+38]
	stb	%o1, [%l3+39]
	stb	%o5, [%l3+32]
	stb	%o4, [%l3+33]
	stb	%o3, [%l3+34]
	stb	%o0, [%l3+24]
	stb	%g4, [%l3+25]
	stb	%g5, [%l3+26]
	stb	%o2, [%l3+27]
	stb	%l0, [%l3+20]
	stb	%i5, [%l3+21]
	stb	%i4, [%l3+22]
	stb	%o1, [%l3+23]
.LL26:
	ba,pt	%xcc, .LL1
	mov	2, %i0
.LL27:
	ld	[%fp+1395], %i3
	add	%i0, 1, %l4
	ld	[%fp+1483], %l7
	ld	[%fp+1531], %i4
	ld	[%fp+1579], %g4
	cmp	%o3, %i3
	st	%l4, [%fp+1371]
	st	%l7, [%fp+1323]
	st	%i4, [%fp+1339]
	bne,pt	%icc, .LL11
	st	%g4, [%fp+1355]
	ld	[%fp+1611], %i0
	ld	[%fp+1307], %i1
	add	%i0, %i0, %i2
	ldx	[%fp+2183], %i5
	sub	%i1, %i2, %g1
	srl	%g4, 16, %i4
	st	%g1, [%i5]
	srl	%g4, 8, %g5
	and	%g4, 0xff, %o3
	srl	%g4, 24, %o7
	ld	[%fp+1339], %o1
	ld	[%fp+1339], %g4
	and	%i4, 255, %i5
	and	%o1, 0xff, %o2
	and	%g5, 255, %i4
	srl	%g4, 24, %o5
	srl	%o1, 16, %i3
	srl	%o1, 8, %i2
	srl	%l7, 16, %i1
	srl	%l7, 8, %i0
	ldx	[%fp+2175], %l2
	and	%i3, 255, %i3
	and	%i2, 255, %i2
	and	%i1, 255, %i1
	and	%i0, 255, %i0
	and	%l7, 0xff, %o1
	srl	%l7, 24, %o4
	srl	%l4, 24, %g1
	srl	%l4, 16, %g4
	srl	%l4, 8, %g5
	stb	%o1, [%l2+35]
	stb	%g1, [%l2+28]
	stb	%g4, [%l2+29]
	stb	%g5, [%l2+30]
	stb	%l4, [%l2+31]
	stb	%o7, [%l2+40]
	stb	%i5, [%l2+41]
	stb	%i4, [%l2+42]
	stb	%o3, [%l2+43]
	stb	%o5, [%l2+36]
	stb	%i3, [%l2+37]
	stb	%i2, [%l2+38]
	stb	%o2, [%l2+39]
	stb	%o4, [%l2+32]
	stb	%i1, [%l2+33]
	stb	%i0, [%l2+34]
	stb	%o7, [%l2+24]
	stb	%i5, [%l2+25]
	stb	%i4, [%l2+26]
	stb	%o3, [%l2+27]
	stb	%o5, [%l2+20]
	stb	%i3, [%l2+21]
	stb	%i2, [%l2+22]
	stb	%o2, [%l2+23]
	stb	%o4, [%l2+16]
	stb	%i1, [%l2+17]
	stb	%i0, [%l2+18]
	ba,pt	%xcc, .LL26
	stb	%o1, [%l2+19]
.LL1:
	return	%i7+8
	nop
.LLFE3:
	.size	rc5_72_unit_func_KKS_2, .-rc5_72_unit_func_KKS_2
	.ident	"GCC: (GNU) 3.3.2"
