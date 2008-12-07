	.text
	.align	3
	.global	ogr_cycle_256_test

	# Parameters

	.set	oState,			3
	.set	pnodes,			4
	.set	upchoose,		5

	# Constants

	.set	const_0,		6
	.set	const_2,		7
	.set	const_31,		8
	.set	const_64,		9
	.set	const_ffff,		10
	.set	consth_0x0001,		11	# halfwords
	.set	V_ONES,			12	# ones in all bits
	.set	const_0x0FF0,		13

	.set	p_gkeyvectors,		15
	.set	p_grvalues,		16
	.set	p_grinsertpos,		17

	# "Level" data

	.set	compV0,			20
	.set	compV1,			21
	.set	listV0,			22
	.set	listV1,			23
	.set	distV0,			24
	.set	distV1,			25
	.set	newbit,			26
	

	# cached oState

	.set	maxlen_m1,		27
	.set	maxdepthm1,		28
	.set	half_depth,		29
	.set	half_depth2,		30
	.set	stopdepth,		31
	.set	depth,			32
	.set	lev_half_depth,		33

	# main variables

	.set	lev,			34
	.set	limit,			35
	.set	mark,			36
	.set	nodes,			37
	.set	s,			38

	# temp variables, short-lived

	.set	temp,			40
	.set	inv_s,			41
	.set	temp2,			42
	.set	tempshuf, 		42	# reused
	.set	ones_shl_inv,		43
	.set	ones_shl_s,		44
	.set	new_compV1,		45
	.set	new_distV0,		46

	# temp variables for pchoose fetch

	.set	hash_mul16,		51
	.set	grpid_vector,		52
	.set	keyvector,		53
	.set	element,		54
	.set	pvalues,		55
	.set	tempvector,		56
	.set	mfc_src,		57

	# condition codes, short-lived but may intersect

	.set	cond_limit_temp,	61
	.set	cond_depth_stopdepth,	62
	.set	cond_mark_limit,	63
	.set	cond_comp0_minus1,	64
	.set	cond_depth_maxm1,	65
	.set	cond_depth_hd,          66
	.set	cond_depth_hd2,		67

	.set	addr_forloop,		70
	.set	addr_no_forloop,	71
	.set	addr_is_forloop,	72

ogr_cycle_256_test:

	# preload useful constants

	fsmbi		$V_ONES,	65535
	il		$const_64,	64
	il		$const_31,	31
	il		$const_0,	0
	il		$const_2,	2
	ilh		$consth_0x0001, 0x0001
	ila		$const_ffff,	65535
	ila		$const_0x0FF0,  0x0FF0
	ila		$p_grinsertpos, group_insertpos
	ila		$p_grvalues,	group_values
	ila		$p_gkeyvectors,	group_keysvectors


	ila		$addr_forloop,    for_loop
	ila		$addr_no_forloop, no_for_loop

	# cache data from oState

	lqd		$temp, 0($oState)
	ai		$maxlen_m1,  $temp, -1	# maxlen_m1 = oState->max - 1
	rotqbyi		$maxdepthm1, $temp, 8
	rotqbyi		$half_depth, $temp, 12

	lqd		$temp, 16($oState)
	ori		$half_depth2, $temp, 0
	rotqbyi		$stopdepth,   $temp, 8
	rotqbyi		$depth,       $temp, 12

	lqd		$nodes, 0($pnodes)
	rotqby		$nodes, $nodes, $pnodes	# nodes = *pnodes

	# if (depth < oState->maxdepthm1) { newbit = vec_splat_u32(1); } else = 0;
	#
	# note that only 32 bits of newbit are valid (we'll never shift in more)

	il		$newbit, 1
	cgt		$temp, $maxdepthm1, $depth
	and		$newbit, $newbit, $temp		# 1 if (maxdepthm1 > depth), 0 otherwise

	# struct OgrLevel *lev = &oState->Levels[oState->depth];

	shli		$lev, $depth, 7		# sizeof(Level) = 128
	ai		$lev, $lev, 32		# &->Levels[depth]
	a		$lev, $lev, $oState	# &oState->Levels

	# SETUP_TOP_STATE(lev)

	lqd		$listV0, 0($lev)
	lqd		$listV1, 16($lev)
	lqd		$distV0, 32($lev)
	lqd		$distV1, 48($lev)
	lqd		$compV0, 64($lev)
	lqd		$compV1, 80($lev)

	# Levels[oState->half_depth] used below
	# note: offsetof(Levels) not added here
	shli		$lev_half_depth, $half_depth, 7
	a		$lev_half_depth, $lev_half_depth, $oState

	# setup of outer_loop (loop header is split)
	lqd		$limit, 112($lev)	# int limit = lev->limit;
	lqd		$mark,  96($lev)	# int mark  = lev->mark;

	nop

outer_loop:

for_loop:
	# shift argument of cntlz right to produce exactly what we've needed -
	# a version with return values 1..32 (32 for both -2 and -1)

	nor		$s, $compV0, $compV0	# ~comp0
	ceqi		$cond_comp0_minus1, $compV0, -1
	rotmi		$s, $s, -1		# magic shift
	selb		$addr_is_forloop, $addr_no_forloop, $addr_forloop, $cond_comp0_minus1

outer_loop_comp0_ready:
for_loop_comp0_ready:

	clz		$s, $s			# get it
	hbr		.L_fl, $addr_is_forloop

	#  if ((mark += s) > limit) { break; }

	a		$mark, $mark, $s
	shlqbybi	$ones_shl_s,   $V_ONES,       $s	# selMask_s

	# Use free clocks in dependency chain to precalculate

	sfi		$inv_s, $s, 128				# inv_s = 128 -s
	shlqbybi	$new_compV1, $compV1, $s

	cgt		$cond_mark_limit, $mark, $limit

	shlqbybi	$ones_shl_inv, $V_ONES,       $inv_s	# selMask_inv (use free clock again)

	shlqbi		$ones_shl_s,   $ones_shl_s,   $s
	brnz		$cond_mark_limit, break_for

	# COMP_LEFT_LIST_RIGHT(lev, s);

	shlqbi		$new_compV1, $new_compV1, $s
	shlqbi		$ones_shl_inv, $ones_shl_inv, $inv_s
	selb		$listV1, $listV0, $listV1, $ones_shl_s
	selb		$listV0, $newbit, $listV0, $ones_shl_s
	selb		$compV0, $compV0, $compV1, $ones_shl_inv
	rotqbybi	$listV0, $listV0, $inv_s
	nop
	rotqbybi	$listV1, $listV1, $inv_s
	ori		$compV1, $new_compV1, 0
	rotqbybi	$compV0, $compV0, $s
	rotqbi		$listV0, $listV0, $inv_s
	rotqbi		$listV1, $listV1, $inv_s
	il		$newbit, 0				# newbit = 0
	rotqbi		$compV0, $compV0, $s

	# if (spu_extract(save_compV0, 0) == (SCALAR)~0) { continue; }

	ceq		$cond_depth_maxm1, $depth, $maxdepthm1	# if (depth == oState->maxdepthm1) ...
.L_fl:
	bi		$addr_is_forloop
	
	.align		3
no_for_loop:

	# lev->mark = mark;
	# if (depth == oState->maxdepthm1) {  goto exit; }

	or		$new_distV0, $distV0, $listV0	# distV0 = vec_or(distV0, listV0); (precalc)
	stqd		$mark, 96($lev)
	nop
	brnz		$cond_depth_maxm1, do_exit

	#  PUSH_LEVEL_UPDATE_STATE(lev);
	#  lev->comp = comp;
	#  lev->list = list;
	#  dist |= list;
	#  comp |= dist;
	# hash = (group_of32 ^ (group_of32 >> 8)) & (GROUPS_COUNT-1);

	rotmi		$hash_mul16,  $new_distV0, -20	# 12345678 => 00000123
	shufb		$grpid_vector, $new_distV0, $new_distV0, $consth_0x0001 # copy high 16 bits of dist0 to all slots for vector compare

	rotmi		$temp,  $new_distV0, -12	# 12345678 => 00012345
	lnop
	
	ori		$distV0, $new_distV0, 0		# copy preloaded distV0
	stqd		$compV0, 64($lev)		# lev->comp[0].v = compV0;

	or		$distV1, $distV1, $listV1	# distV1 = vec_or(distV1, listV1);
	stqd		$compV1, 80($lev)		# lev->comp[1].v = compV1;

	or		$compV0, $compV0, $distV0	# compV0 = vec_or(compV0, distV0);
	stqd		$listV0, 0($lev)		# lev->list[0].v = listV0;

	xor		$hash_mul16, $temp, $hash_mul16	# 123 ^ 345
	lnop
	
	or		$compV1, $compV1, $distV1	# compV1 = vec_or(compV1, distV1);
	stqd		$listV1, 16($lev)		# lev->list[1].v = listV1;

	# ++lev;
	# ++depth;
	#
	# define choose(dist, seg) direct_dma_fetch(upchoose, ((dist) >> (SCALAR_BITS-CHOOSE_DIST_BITS)), (seg))
	#      limit = choose(dist0, depth);
	# i.e. group = high 16 bits of dist0

	and		$hash_mul16, $hash_mul16, $const_0x0FF0	# ... (123 ^ 345) & ((GROUPS_COUNT-1) * 16)
							# hash*16 to access array[hash],
	ai		$depth, $depth, 1	# ++depth;

	shli		$pvalues, $hash_mul16, 5	# 1D: hash * 8 * 32 * 2  [ (hash * 16) * 32 ]
	lqx		$keyvector, $hash_mul16, $p_gkeyvectors	# keyvector = group_keysvectors[hash]

	a		$temp2, $depth, $depth	# depth * 2

	il		$newbit, 1		# newbit = 1 from PUSH_LEVEL

	ai		$lev, $lev, 128		# ++lev;
	lnop

	a		$pvalues, $pvalues, $temp2	# 1D + 3D (hash * 8 * 32 * 2 + depth * 2)

	# eqbits    = spu_extract(spu_cntlz(spu_gather(spu_cmpeq(keyvector, (u16)group_of32))), 0);
	# element   = eqbits - 24;
	# also start making new $s using free clocks

		nor	$s, $compV0, $compV0	# ~comp0
	ceqh		$element, $keyvector, $grpid_vector	# spu_cmpeq(keyvector, (u16)group_of32)
		ceqi	$cond_comp0_minus1, $compV0, -1
		rotmi	$s, $s, -1		# magic shift
	gbh		$temp, $element		# spu_gather(...)
		selb	$addr_is_forloop, $addr_no_forloop, $addr_forloop, $cond_comp0_minus1
	clz		$element, $temp		# spu_cntlz(...)

	# if (element == 8) /* 32 zeros => Out of bounds => no match */

	ai		$element, $element, -24
	brz		$temp, fetch_new_pchoose	# if all bits were zero, i.e. clz==32 or element==8

	#            u16              8          32
	# return group_values[hash][element][index_in32];

	shli		$element, $element, 6		# 2D: element * 32 * 2
	hbrr		.L_1, for_loop_comp0_ready
	a		$pvalues, $pvalues, $element	# final byte offset
	lnop
	a		$temp, $pvalues, $p_grvalues	# full address (for shift)
	lqx		$tempvector, $pvalues, $p_grvalues	# get data at full address
fetch_pvalues:

	# if (depth > half_depth && depth <= half_depth2) { ...

	# Compiler generated nice compilcated straight-forward code, but chance for
	# 'if' body to execute is about 3%. Dumb jump is better in our case.

	# finish load of 'limit'

	cgt		$cond_depth_hd,  $depth, $half_depth
	cgt		$cond_depth_hd2, $depth, $half_depth2
	ai		$temp, $temp, 14			# prepare u16 shift to PS
	andc		$cond_depth_hd,  $cond_depth_hd, $cond_depth_hd2
	ai		$nodes, $nodes, -1
	rotqby		$tempvector, $tempvector, $temp		# shift result to PS
	and		$limit, $const_ffff, $tempvector	# u16 -> u32

	brnz		$cond_depth_hd, update_limit

	stqd		$limit, 112($lev)
.L_1:
	brnz		$nodes, for_loop_comp0_ready
	br		save_mark_and_exit

	#
	# Visited in less then 3% cases and not important
	#
	.align		3	
update_limit:
	nop
	lqd		$temp, 128($lev_half_depth)	# offsetof(Levels) added here

	# while $temp is loading (6 clocks), calc other things

	nor		$temp2, $distV0, $distV0
	hbrr		.L_2, for_loop_comp0_ready
	cgt		$cond_depth_hd2, $half_depth2, $depth
	clz		$temp2, $temp2
	ai		$temp2, $temp2, 1
	
	sf		$temp, $temp, $maxlen_m1
	cgt		$cond_limit_temp, $limit, $temp

	selb		$limit, $limit, $temp, $cond_limit_temp
	sf		$temp2, $temp2, $limit
	selb		$limit, $limit, $temp2, $cond_depth_hd2
	
	stqd		$limit, 112($lev)
.L_2:
	brnz		$nodes, for_loop_comp0_ready
	
save_mark_and_exit:
	stqd		$mark, 96($lev)

do_exit:
	# SAVE_FINAL_STATE(lev);
	stqd		$listV0, 0($lev)
	stqd		$listV1, 16($lev)
	stqd		$distV0, 32($lev)
	stqd		$distV1, 48($lev)
	stqd		$compV0, 64($lev)
	stqd		$compV1, 80($lev)

	# *pnodes -= nodes;

	lqd		$tempvector, 0($pnodes)
	cwd		$tempshuf, 0($pnodes)
	rotqby		$temp, $tempvector, $pnodes
	sf		$temp, $nodes, $temp
	shufb		$tempvector, $temp, $tempvector, $tempshuf
	stqd		$tempvector, 0($pnodes)

	ori		$3, $depth, 0		# <result>, depth
	bi		$lr

	.align		3

break_for:
	# --lev;
	# --depth;
	ai		$lev, $lev, -128
	hbrr		.L33, outer_loop_comp0_ready
	ai		$depth, $depth, -1
	fsmbi		$newbit, 0		# newbit = 0

	# POP_LEVEL(lev);

	lqd		$compV0, 64($lev)
	lqd		$compV1, 80($lev)
	cgt		$cond_depth_stopdepth, $depth, $stopdepth
	lqd		$listV0, 0($lev)
	lqd		$listV1, 16($lev)

	# copy loop header from outer_loop (placed here to break dependency)
	# and finish POP_LEVEL
	
	lqd		$limit, 112($lev)	# int limit = lev->limit;
		nor	$s, $compV0, $compV0	# ~comp0
		ceqi	$cond_comp0_minus1, $compV0, -1
	andc		$distV0, $distV0, $listV0
		rotqmbii $s, $s, -1		# magic shift (use slot 1)
		selb	$addr_is_forloop, $addr_no_forloop, $addr_forloop, $cond_comp0_minus1
	lqd		$mark,  96($lev)	# int mark  = lev->mark;
	andc		$distV1, $distV1, $listV1

	# while (depth > oState->stopdepth);
.L33:
	brnz		$cond_depth_stopdepth, outer_loop_comp0_ready
	br		do_exit

fetch_new_pchoose:

	# v32_t tempvect = group_insertpos[hash];
	# element        = spu_extract(tempvect, 0) & (GROUPS_LENGTH - 1);

	lqx		$element, $hash_mul16, $p_grinsertpos
	andi		$element, $element, 7

	#               u16               8     [32]
	# pvalues = group_values[hash][element];

	rotmi		$pvalues, $hash_mul16, -1	# hash * 8 [ hash * 16 / 2 ]
	a		$pvalues, $pvalues, $element	# hash * 8 + element
	shli		$pvalues, $pvalues, 6		# (...) * 32 * 2 (last dimension and u16 to bytes)
	a		$pvalues, $pvalues, $p_grvalues	# full byte address

	# mfc_get(pvalues, upchoose + group_of32 * GROUP_ELEMENTS * 2, GROUP_ELEMENTS * 2, DMA_ID, 0, 0);

	and		$mfc_src, $grpid_vector, $const_ffff
	shli		$mfc_src, $mfc_src, 6		# * 32 * 2
	a		$mfc_src, $mfc_src, $upchoose

	wrch		$ch16, $pvalues
	wrch		$ch17, $const_0
	wrch		$ch18, $mfc_src
	wrch		$ch19, $const_64
	wrch		$ch20, $const_31
	wrch		$ch21, $const_64

	# group_insertpos[hash]   = spu_add(tempvect, 1);

	shli		$temp, $element, 1	# current pos to bytes

	ai		$element, $element, 1
	stqx		$element, $hash_mul16, $p_grinsertpos	# group_insertpos[hash]

	# group_keysvectors[hash] = spu_insert((u16)group_of32, keyvector, element);

	chx		$temp, $sp, $temp	# mask to insert vector element 'newpos'
	shufb		$temp, $grpid_vector, $keyvector, $temp
	stqx		$temp, $hash_mul16, $p_gkeyvectors

	# prepare element full address for fetch below

	a		$temp, $depth, $depth	# convert last index to bytes
	a		$temp, $temp, $pvalues	# full byte address

	# mfc_read_tag_status_all();

	wrch		$ch23, $const_2
	rdch		$temp2, $ch24

	lqd		$tempvector, 0($temp)

	br		fetch_pvalues	# byte address in temp, values in tempvector
