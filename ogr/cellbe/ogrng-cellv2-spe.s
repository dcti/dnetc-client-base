	;#
	;# Assembly OGR-NG core for Cell SPU
	;#
	;# Created by Roman Trunov <stream@distributed.net>
	;#
	;# Additional References:
	;#    1) C OGR-NG core for Cell SPU (ogrng-cell-spe-wrapper.c) -
	;# describes algorithm used in this code, especially vector tricks and
	;# caching method.
	;#    2) Assembly SPU core for OGR-P2 by Decio (ogr-cellv1-spe.s) - 
	;# describes lot of useful SPU programming and optimization tricks.
	;#
	;# This assembly code, in general, uses same algorithm as C version.
	;# (ogrng-cell-spe-wrapper.c). Instructions were manually choosen
	;# and reordered for best pipeline and dual-execution usage. Also,
	;# speculative calculations 'in advance' are widely used. The code now
	;# looks like complete mess. So be it.
	;#

	;# History:
	;#
	;# 2009-04-05:
	;#    Store of new 'limit' is delayed. It can be stored before
	;# PUSH_LEVEL and when core exits. Same for 'mark'.
	;#
	;# 2009-04-10:
	;#    Reordered shifts in main loop to avoid few dependency stalls.
	;#    Rewriten POP_LEVEL to avoid dependency on --lev.
	;#    Two lnops in for_loop increased speed a bit, but I don't know why.
	;# (If you'll add another two lnops to following two commands, speed
	;# will decrease back. Weird.)
	;#    Rewriten extraction of u16 limit from vectorized cache to break
	; dependencies.
	;#
	;# 2009-10-20:
	;#    Get rid of "outer_loop" and "for_loop_clz_ready" labels and their
	;# jumps. This code is copied now. It also allowed additional scheduling
	;# optimizations.

	;#
	;# Known non-optimal things:
	;#
	;#    1) Calculation of 's' has long dependency chain. Most of codepathes
	;# will preload 's' "in background", at least partially, but it's not
	;# possible in inner loop which is too short and 'compV0' is ready
	;# only in the end of it.
	;#
	;#    2) The cache is a pain in the back. Since SPU has only 256 KBytes
	;# of memory, the 'choose' array is dynamically cached, using circular
	;# algorithm to purge out old entries (check C source to understand it).
	;# For some types of stubs, cache overflows often, slowing real
	;# (not benchmark!) performance greatly. Even in shortest codepath,
	;# cache lookup procedure is a long sequence of slow and dependent
	;# instructions - lot of clocks are lost here and, I'm afraid, nothing
	;# can be done.
	;#

	.data
	.align	4

data_vec_to_u16_shuf:
	.int		0x1E1F1011, 0x12131415, 0x16171819, 0x1A1B1C1D

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
	.set	const_vec_to_u16_shuf,	14

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
	lqa		$const_vec_to_u16_shuf, data_vec_to_u16_shuf
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
	# note that only 32 bits of newbit are valid (we will never shift in more)

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

;#	This label must be aligned - insert nops if necessary

for_loop:

	;#
	;# Four commands below are in dependency chain. So, if possible,
	;# (~comp0 >> 1) and $addr_is_forloop must be precalculated 
	;# 'in background' using free slots of other commands.
	;#

	# shift argument of cntlz right to produce exactly what we are needed -
	# a version with return values 1..32 (32 for both -2 and -1)

	nor		$s, $compV0, $compV0	# ~comp0
	lnop					# with these 2 lnops a bit faster??? caching issues???
	ceqi		$cond_comp0_minus1, $compV0, -1
	lnop
	rotmi		$s, $s, -1		# magic shift
	selb		$addr_is_forloop, $addr_no_forloop, $addr_forloop, $cond_comp0_minus1

	clz		$s, $s			# get it
	lnop

	;# Sequence above is not optimal. For codepaths other then inner
	;# loop, it's recommended to avoid jump to 'for_loop'. Distribute these
	;# instrutions between other code using free clocks/slots and finally
	;# use for_loop_clz_ready_*. macro. Even better, if these macros are
	;# distributed too.

.macro	for_loop_clz_ready_part1
	sfi		$inv_s, $s, 128			# inv_s = 128 -s
	shlqbybi	$ones_shl_s, $V_ONES, $s	# selMask_s
.endm

.macro	for_loop_clz_ready_part2  lab

	#  if ((mark += s) > limit) { break; }
	a		$mark, $mark, $s
	hbr		\lab, $addr_is_forloop

	# Use free clocks in dependency chain to precalculate

	nop
	shlqbybi	$ones_shl_inv, $V_ONES, $inv_s	# selMask_inv (use free clock again)

	cgt		$cond_mark_limit, $mark, $limit
	shlqbybi	$new_compV1, $compV1, $s

	shlqbi		$ones_shl_s, $ones_shl_s, $s

	brnz		$cond_mark_limit, break_for

	shlqbi		$ones_shl_inv, $ones_shl_inv, $inv_s

	# COMP_LEFT_LIST_RIGHT(lev, s);

	shlqbi		$new_compV1, $new_compV1, $s
	selb		$listV1, $listV0, $listV1, $ones_shl_s
	selb		$listV0, $newbit, $listV0, $ones_shl_s
	selb		$compV0, $compV0, $compV1, $ones_shl_inv
	rotqbybi	$listV1, $listV1, $inv_s
	rotqbybi	$listV0, $listV0, $inv_s
	rotqbybi	$compV0, $compV0, $s
	ori		$compV1, $new_compV1, 0
	rotqbi		$listV1, $listV1, $inv_s
	nop
	rotqbi		$listV0, $listV0, $inv_s
	il		$newbit, 0				# newbit = 0
	rotqbi		$compV0, $compV0, $s

	# if (spu_extract(save_compV0, 0) == (SCALAR)~0) { continue; }

	ceq		$cond_depth_maxm1, $depth, $maxdepthm1	# if (depth == oState->maxdepthm1) ...
\lab:
	bi		$addr_is_forloop
.endm

	for_loop_clz_ready_part1
	for_loop_clz_ready_part2 .L_fl
	
	.align		3
no_for_loop:

	# lev->mark = mark;
	# if (depth == oState->maxdepthm1) {  goto exit; }
	or		$new_distV0, $distV0, $listV0	# distV0 = vec_or(distV0, listV0); (precalc)
	brnz		$cond_depth_maxm1, save_mark_and_exit

	#  PUSH_LEVEL_UPDATE_STATE(lev);
	#  lev->comp = comp;
	#  lev->list = list;
	#  dist |= list;
	#  comp |= dist;
	# hash = (group_of32 ^ (group_of32 >> 8)) & (GROUPS_COUNT-1);

	rotmi		$hash_mul16,  $new_distV0, -20	# 12345678 => 00000123
	shufb		$grpid_vector, $new_distV0, $new_distV0, $consth_0x0001 # copy high 16 bits of dist0 to all slots for vector compare

	rotmi		$temp,  $new_distV0, -12	# 12345678 => 00012345
	stqd		$limit, 112($lev)		# delayed lev->limit = limit
	
	ori		$distV0, $new_distV0, 0		# copy preloaded distV0
	stqd		$compV0, 64($lev)		# lev->comp[0].v = compV0;

	or		$distV1, $distV1, $listV1	# distV1 = vec_or(distV1, listV1);
	stqd		$compV1, 80($lev)		# lev->comp[1].v = compV1;

	or		$compV0, $compV0, $distV0	# compV0 = vec_or(compV0, distV0);
	stqd		$mark, 96($lev)			# lev->mark = mark (delayed)

	xor		$hash_mul16, $temp, $hash_mul16	# 123 ^ 345
	stqd		$listV0, 0($lev)		# lev->list[0].v = listV0;
	
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
		clz	$s, $s					# precount s (clz)!
		lnop

	# if (element == 8) /* 32 zeros => Out of bounds => no match */

	ai		$element, $element, -24
	brz		$temp, fetch_new_pchoose	# if all bits were zero, i.e. clz==32 or element==8

	#            u16              8          32
	# return group_values[hash][element][index_in32];

	shli		$element, $element, 6		# 2D: element * 32 * 2
	lnop
	;# 3 clocks lost here
	a		$pvalues, $pvalues, $element	# final byte offset
	;# 2 clocks lost here
	lqx		$tempvector, $pvalues, $p_grvalues	# get data at full address
fetch_pvalues:

	# if (depth > half_depth && depth <= half_depth2) { ...

	# It is possible to generate compilcated straight-forward code, but chance for
	# 'if' body to execute is about 3%. Dumb jump is better in our case.

	# finish load of 'limit'

	cgt		$cond_depth_hd,  $depth, $half_depth
	rotqby		$pvalues, $const_vec_to_u16_shuf, $pvalues # create shuffle mask from byte address

	cgt		$cond_depth_hd2, $depth, $half_depth2
	lnop

	for_loop_clz_ready_part1

	andc		$cond_depth_hd,  $cond_depth_hd, $cond_depth_hd2
	hbr		.L_fl2, $addr_is_forloop

	and		$pvalues, $const_ffff, $pvalues		# set upper word of shuffle mask to zero
	shlqbybi	$ones_shl_inv, $V_ONES, $inv_s	# selMask_inv (use free clock again)

	;# Note: limit is not ready yet, operation delayed in update_limit
	ai		$nodes, $nodes, -1
	brnz		$cond_depth_hd, update_limit
	shufb		$limit, $const_ffff, $tempvector, $pvalues # select required u16 and clear high part (use zeros from const)

	# store of limit delayed

	brz		$nodes, save_mark_and_exit

	;# modifed version of for_loop_clz_ready_part2 code
	;# hbr and one shift pushed up so it was possible to reorder instructions better.

	#  if ((mark += s) > limit) { break; }
	a		$mark, $mark, $s
	shlqbi		$ones_shl_s, $ones_shl_s, $s

	nop
	shlqbybi	$new_compV1, $compV1, $s

	cgt		$cond_mark_limit, $mark, $limit
	shlqbi		$ones_shl_inv, $ones_shl_inv, $inv_s

	# COMP_LEFT_LIST_RIGHT(lev, s);

	;# it's ok to corrupt listV1 before jump - it's reloaded in break_for (POP_LEVEL)
	selb		$listV1, $listV0, $listV1, $ones_shl_s
	brnz		$cond_mark_limit, break_for

	selb		$listV0, $newbit, $listV0, $ones_shl_s
	shlqbi		$new_compV1, $new_compV1, $s
	selb		$compV0, $compV0, $compV1, $ones_shl_inv
	rotqbybi	$listV1, $listV1, $inv_s
	rotqbybi	$listV0, $listV0, $inv_s
	rotqbybi	$compV0, $compV0, $s
	ori		$compV1, $new_compV1, 0
	rotqbi		$listV1, $listV1, $inv_s
	nop
	rotqbi		$listV0, $listV0, $inv_s
	il		$newbit, 0				# newbit = 0
	rotqbi		$compV0, $compV0, $s

	# if (spu_extract(save_compV0, 0) == (SCALAR)~0) { continue; }

	ceq		$cond_depth_maxm1, $depth, $maxdepthm1	# if (depth == oState->maxdepthm1) ...
.L_fl2:
	bi		$addr_is_forloop


	.align		3	
update_limit:
	;#
	;# Visited in less then 3% cases and not so important
	;#
	;# Caller already prepared $s and executed for_loop_clz_ready_part1

	# while $temp is loading (6 clocks), calc other things

	nor		$temp2, $distV0, $distV0
	lqd		$temp, 128($lev_half_depth)	# offsetof(Levels) added here

	cgt		$cond_depth_hd2, $half_depth2, $depth	;# if (depth < oState->half_depth2)
	;# Complete delayed fetch of limit
	shufb		$limit, $const_ffff, $tempvector, $pvalues # select required u16 and clear high part (use zeros from const)
	clz		$temp2, $temp2
	ai		$temp2, $temp2, 1			;# temp2 = LOOKUP_FIRSTBLANK
	
	sf		$temp, $temp, $maxlen_m1		;# temp  = ... (ready)
	sf		$temp2, $temp2, $temp			;# temp2 = temp - LFB() (probably will use)
	selb		$temp, $temp, $temp2, $cond_depth_hd2	;# use temp or "temp -= ..."

	cgt		$cond_limit_temp, $limit, $temp		;# if (limit > temp)
	selb		$limit, $limit, $temp, $cond_limit_temp ;#   limit = temp;
	
	# Store of limit delayed

	brz		$nodes, save_mark_and_exit

	for_loop_clz_ready_part2  .L_fl3

	
save_mark_and_exit:
	stqd		$mark, 96($lev)
	stqd		$limit, 112($lev)
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
	ai		$depth, $depth, -1

	# POP_LEVEL(lev);
	# copy loop header from for_loop to break dependencies
	# and finish POP_LEVEL
	# decrement of lev delayed to avoid dependency, "-128" added directly to offset

	lqd		$compV0, 64-128($lev)
	nop
	lqd		$listV0, 0-128($lev)
	lqd		$listV1, 16-128($lev)
	lqd		$compV1, 80-128($lev)
	cgt		$cond_depth_stopdepth, $depth, $stopdepth
	lqd		$limit, 112-128($lev)	# int limit = lev->limit;
	lqd		$mark,  96-128($lev)	# int mark  = lev->mark;
		nor	$s, $compV0, $compV0	# ~comp0
	andc		$distV0, $distV0, $listV0
		lnop
	andc		$distV1, $distV1, $listV1
		rotqmbii $s, $s, -1		# magic shift (use slot 1)

		ceqi	$cond_comp0_minus1, $compV0, -1
	fsmbi		$newbit, 0		# newbit = 0

	ai		$lev, $lev, -128	# finally, --lev
	
		selb	$addr_is_forloop, $addr_no_forloop, $addr_forloop, $cond_comp0_minus1

		clz		$s, $s			# get it

	# while (depth > oState->stopdepth);

	brz		$cond_depth_stopdepth, save_mark_and_exit

	for_loop_clz_ready_part1
	for_loop_clz_ready_part2  .L_fl1


fetch_new_pchoose:
	;#
	;# Called when new portion of pchoose entries must be copied from main
	;# memory using DMA. Since this is very long process by itself, which MUST
	;# not be executed often, optimizations are useless here.
	;#

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
	a		$pvalues, $temp, $pvalues	# full byte address

	# mfc_read_tag_status_all();

	wrch		$ch23, $const_2
	rdch		$temp2, $ch24

	lqd		$tempvector, 0($pvalues)

	# byte address in pvalues, values in tempvector
	br		fetch_pvalues
