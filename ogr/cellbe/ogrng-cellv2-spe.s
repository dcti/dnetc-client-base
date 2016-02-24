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
	;# dependencies.
	;#
	;# 2009-10-20:
	;#    Get rid of "outer_loop" and "for_loop_clz_ready" labels and their
	;# jumps. This code is copied now. It also allowed additional scheduling
	;# optimizations.
	;#
	;# 2010-03-17:
	;#    Keep cached comp0 from previous level in register (save it during
	;# PUSH_LEVEL). Thus we can immediately start calculating CLZ etc. on it
	;# combining with rest of POP_LEVEL code in free slots.
	;#
	;# 2010-06-30:
	;#    Since we have lot of free/lost clocks during handling of pchoose cache,
	;# implement 64-bit version of CLZ in these gaps. Also remove limitation of
	;# 32 bits width of newbit to make big shifts work right.
	;#    It was so many free clocks there that is was even possible to speculative
	;# calculate whole next shift round, giving a nice boost (almost 10% total).
	;#
	;# 2013-10-20:
	;#    Better alignment of jump targets gained one clock (~400K) without
	;# changes in the code.
	;#    Gained ~200K by removing one-clock dependency for first "or" in
	;# "no_for_loop"->"no_for_loop" path.

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
	;# can be done. (In current version timing of cache access codepath is
	;# precisiously synchronized with speculative calculation of next ogr
	;# iteration, so losses are almost eliminated).
	;#

	;#
	;# Current speed (PS3): 36,736,769 nodes/sec
	;#

	.data
	.align	4

data_vec_to_u16_shuf:
	.int		0x1E1F1011, 0x12131415, 0x16171819, 0x1A1B1C1D
clz_addons_init:
	.int		0, 32, 64, 96

	.text

	;# Alignment could cause crazy and unpredictable effects when it comes
	;# to instructions starvation (empty prefetch buffers) because datastore
	;# has higher priority then IFETCH. Cell Handbook says that jumps aligned
	;# to 64 bytes could improve performance in such cases. Let's start code
	;# aligned to 128 bytes (full IFETCH line) to be able to align on any
	;# value below.

	.align	7
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

	.set	mask_2words,		18	# zero in words 0-1, ones in 2-3 (to mask words 0-1)
	.set	clz_addons,		19	# values to add to clz for each word (0, 32, ...)

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
	.set	comp0_prevlev,		39

	# temp variables, short-lived

	.set	temp,			40
	.set	inv_s,			41
	.set	temp2,			42
	.set	tempshuf, 		42	# reused
	.set	ones_shl_inv,		43
	.set	ones_shl_s,		44
	.set	new_compV1,		45
	.set	new_distV0,		46

	.set	const_singlebit,	47
	.set	new_listV1,		48

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
	.set	addr_no_forloop_no_or,	73

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
	fsmbi		$mask_2words,   0x00FF
	lqa		$clz_addons,    clz_addons_init

	ila		$addr_forloop,    for_loop
	ila		$addr_no_forloop, no_for_loop
	ila		$addr_no_forloop_no_or, no_for_loop_no_or

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

	il		$const_singlebit, 1
	rotqmbyi	$const_singlebit, $const_singlebit, -12	# keep 1 only in rightmost bit of reg
	cgt		$temp, $maxdepthm1, $depth
	and		$newbit, $const_singlebit, $temp # 1 if (maxdepthm1 > depth), 0 otherwise

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
	lqd		$comp0_prevlev, 64-128($lev)

	# Levels[oState->half_depth] used below
	# note: offsetof(Levels) not added here
	shli		$lev_half_depth, $half_depth, 7
	a		$lev_half_depth, $lev_half_depth, $oState

	# setup of outer_loop (loop header is split)
	lqd		$limit, 112($lev)	# int limit = lev->limit;
	lqd		$mark,  96($lev)	# int mark  = lev->mark;

	;# Alignment craziness starts right here :)
	;#
	;# This label must be aligned to exactly 64 bytes. Align it to 128 bytes -
	;# and you'll lose a clock (~440KNodes). Look like cause of the problem is
	;# a code at "break_for". Jump to this code is executed once per node and
	;# is always unpredicted (hard to predict, no room for hint). Current best
	;# performace value (36,556 KNodes) happens when break_for is EXACTLY WHERE
	;# IT IS - at address offset 0x0398. Move it by just a few instructions -
	;# speed will change. There is a "stable" speed of 36,531K in most cases,
	;# but at some offsets (e.g. if break_for aligned to 64, code offset 0x03C0)
	;# you'll lose a full clock (speed will drop to 36,097K). Be aware! Move this
	;# damn code to the stable point before you'll attempt to add or remove
	;# intructions above it! (Align to 128 / offset 0x0400 looks safe).

	;# So, as said above, "for_loop" must be aligned to 64 (file offset 0x00C0).
	;# Be avare when changing number of instructions in initialization code above.
	;# Moving this label could have side effects of speed penalties. Test it at
	;# new offsets first.

	.align	6

for_loop:

	;#
	;# Four commands below are in dependency chain. So, if possible,
	;# (~comp0 >> 1) and $addr_is_forloop must be precalculated 
	;# 'in background' using free slots of other commands.
	;#

	# shift argument of cntlz right to produce exactly what we are needed -
	# a version with return values 1..32 (32 for both -2 and -1)

	nor		$s, $compV0, $compV0	# ~comp0
	ceqi		$cond_comp0_minus1, $compV0, -1
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

.macro	for_loop_clz_ready_part2_head  lab

	#  if ((mark += s) > limit) { break; }
	a		$mark, $mark, $s
	hbr		\lab, $addr_is_forloop

	# Use free clocks in dependency chain to precalculate

	nop
	shlqbybi	$ones_shl_inv, $V_ONES, $inv_s	# selMask_inv (use free clock again)
.endm

.macro	for_loop_clz_ready_part2_body  lab
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
	il		$newbit, 0				# newbit = 0
	rotqbi		$listV0, $listV0, $inv_s
	ceq		$cond_depth_maxm1, $depth, $maxdepthm1	# if (depth == oState->maxdepthm1) ...
	rotqbi		$compV0, $compV0, $s

	# if (spu_extract(save_compV0, 0) == (SCALAR)~0) { continue; }

#	nop
\lab:
	bi		$addr_is_forloop
.endm

	for_loop_clz_ready_part1
	for_loop_clz_ready_part2_head .L_fl
	for_loop_clz_ready_part2_body .L_fl
	
	.align		3
no_for_loop:

	# lev->mark = mark;
	# if (depth == oState->maxdepthm1) {  goto exit; }
	or		$new_distV0, $distV0, $listV0	# distV0 = vec_or(distV0, listV0); (precalc)
no_for_loop_no_or:
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
	rotqbyi		$comp0_prevlev, $compV0, 0	# copy via odd pipeline, store delayed

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
	stqd		$comp0_prevlev, 64($lev)	# lev->comp[0].v = compV0; (delayed)

	# we will have lot of free clocks while making pchoose index and limit
	# use them to make 64-bit version of clz

	ai		$depth, $depth, 1	# ++depth;
	lnop

	shli		$pvalues, $hash_mul16, 5	# 1D: hash * 8 * 32 * 2  [ (hash * 16) * 32 ]
	lqx		$keyvector, $hash_mul16, $p_gkeyvectors	# keyvector = group_keysvectors[hash]

		# stuck by shli, could push calc of temp2 down
		nor	$cond_comp0_minus1, $compV0, $mask_2words # ~comp0 (free clock). words 2-3 forced to zero.

	a		$temp2, $depth, $depth	# depth * 2

	ori		$newbit, $const_singlebit, 0	# newbit = 1 from PUSH_LEVEL (f.c.)
		rotqmbii $s, $cond_comp0_minus1, -1	# magic shift ~comp0 for whole 128 bits

	a		$pvalues, $pvalues, $temp2	# 1D + 3D (hash * 8 * 32 * 2 + depth * 2)
		orx	$cond_comp0_minus1, $cond_comp0_minus1  # word0==0 if all required parts of comp0 were -1

	ai		$lev, $lev, 128		# (f.c.) ++lev

	# eqbits    = spu_extract(spu_cntlz(spu_gather(spu_cmpeq(keyvector, (u16)group_of32))), 0);
	# element   = eqbits - 24;

	# this one stuck by lqx
	ceqh		$element, $keyvector, $grpid_vector	# spu_cmpeq(keyvector, (u16)group_of32)

		clz	$s, $s					# precount s (clz)
		lnop

		ceqi	$cond_comp0_minus1, $cond_comp0_minus1, 0 # (f.c.)
	gbh		$temp, $element		# spu_gather(...)

		andi	$temp2, $s, 32		# if 32 is in word0, then use clz result from word1

		a	$s, $s, $clz_addons	# clz(word1) += 32

		# In this path we even have time to precalculate top "or" command
		selb	$addr_is_forloop, $addr_no_forloop_no_or, $addr_forloop, $cond_comp0_minus1
		lnop

	clz		$element, $temp		# spu_cntlz(...)
		rotqbybi $s, $s, $temp2		# use clz(word1) if necessary

		cgt	$cond_depth_hd,  $depth, $half_depth	# (f.c.) depth > oState->half_depth
	brz		$temp, fetch_new_pchoose	# if all bits were zero, i.e. clz==32 or element==8

	# if (element == 8) /* 32 zeros => Out of bounds => no match */

	ai		$element, $element, -24
	lnop

fetch_pvalues:
		cgt		$cond_depth_hd2, $depth, $half_depth2	# (f.c.) depth <= half_depth2
		lnop
#		hbr		.L_fl2, $addr_is_forloop

	#            u16              8          32
	# return group_values[hash][element][index_in32];
	#
	# ($s is also ready in this clock)

#		for_loop_clz_ready_part1 # (f.c.)

	shli		$element, $element, 6		# 2D: element * 32 * 2
		shlqbybi	$ones_shl_s, $V_ONES, $s	# selMask_s

		sfi		$inv_s, $s, 128			# inv_s = 128 -s
		shlqbybi	$new_compV1, $compV1, $s

		andc		$cond_depth_hd,  $cond_depth_hd, $cond_depth_hd2  # (f.c.)
#		lnop
		hbr		.L_fl2, $addr_is_forloop

		ai		$nodes, $nodes, -1		# --nodes (f.c.)
		shlqbybi	$ones_shl_inv, $V_ONES, $inv_s	# selMask_inv (f.c.)

	# if (depth > half_depth && depth <= half_depth2) { ...
	#
	# It is possible to generate compilcated straight-forward code, but chance for
	# 'if' body to execute is about 3%. Dumb jump is better in our case.
	#
	# Be careful to match calculation of ones_shl_* above and in update_limit

	a		$pvalues, $pvalues, $element	# final byte offset
		brnz		$cond_depth_hd, update_limit

		shlqbi		$ones_shl_s, $ones_shl_s, $s

	lqx		$tempvector, $pvalues, $p_grvalues	# get data at full address

	rotqby		$pvalues, $const_vec_to_u16_shuf, $pvalues # create shuffle mask from byte address

		shlqbi		$ones_shl_inv, $ones_shl_inv, $inv_s

		selb		$new_listV1, $listV0, $listV1, $ones_shl_s
		brz		$nodes, finish_limit_and_save_exit

	;# Now is's safe to corrupt list and comp - they'll be reloaded in POP_LEVEL
	;# if we'll go there.

		selb		$listV0, $newbit, $listV0, $ones_shl_s
		shlqbi		$new_compV1, $new_compV1, $s

	and		$pvalues, $const_ffff, $pvalues		# set upper word of shuffle mask to zero
		rotqbybi	$listV1, $new_listV1, $inv_s

		selb		$compV0, $compV0, $compV1, $ones_shl_inv
		rotqbybi	$listV0, $listV0, $inv_s

		a		$mark, $mark, $s		# mark += s
	shufb		$limit, $const_ffff, $tempvector, $pvalues # select required u16 and clear high part (use zeros from const)

		ori		$compV1, $new_compV1, 0
		rotqbybi	$compV0, $compV0, $s

		il		$newbit, 0				# newbit = 0
		lnop

		ceq		$cond_depth_maxm1, $depth, $maxdepthm1	# if (depth == oState->maxdepthm1) ...
		rotqbi		$listV0, $listV0, $inv_s

	cgt		$cond_mark_limit, $mark, $limit
		rotqbi		$listV1, $listV1, $inv_s

		rotqbi		$compV0, $compV0, $s

	brnz		$cond_mark_limit, break_for

	# if (spu_extract(save_compV0, 0) == (SCALAR)~0) { continue; }

	or		$new_distV0, $distV0, $listV0	# distV0 = vec_or(distV0, listV0); (precalc)
.L_fl2:
	bi		$addr_is_forloop


	.align		3	
update_limit:
	;#
	;# Visited in less then 3% cases and not so important
	;# (gain/loss is just about 1,5 Knodes per clock in benchmark)
	;#
	;# Caller already prepared $s and executed for_loop_clz_ready_part1

	# complete calculation of limit which was not executed by caller 

	nor			$temp2, $distV0, $distV0
	lqd			$temp, 128($lev_half_depth)	# offsetof(Levels) added here

	cgt			$cond_depth_hd2, $half_depth2, $depth	;# if (depth < oState->half_depth2)
		lqx		$tempvector, $pvalues, $p_grvalues	# get data at full address

	clz			$temp2, $temp2
		rotqby		$pvalues, $const_vec_to_u16_shuf, $pvalues # create shuffle mask from byte address

			nop
			shlqbi	$ones_shl_s, $ones_shl_s, $s

	ai			$temp2, $temp2, 1			;# temp2 = LOOKUP_FIRSTBLANK
	brz			$nodes, finish_update_limit_and_exit

	;# ok to update mark after jump above

			a	$mark, $mark, $s	#  if ((mark += s) > limit) { break; }
			shlqbi	$ones_shl_inv, $ones_shl_inv, $inv_s

	sf			$temp, $temp, $maxlen_m1		;# temp  = ... (ready)
			shlqbi	$new_compV1, $new_compV1, $s

			selb	$listV1, $listV0, $listV1, $ones_shl_s
			hbr	.L_fl3, $addr_is_forloop


	sf			$temp2, $temp2, $temp			;# temp2 = temp - LFB() (probably will use)
	lnop

		and		$pvalues, $const_ffff, $pvalues		# set upper word of shuffle mask to zero

	selb			$temp, $temp, $temp2, $cond_depth_hd2	;# use temp or "temp -= ..."

			selb	$listV0, $newbit, $listV0, $ones_shl_s
		shufb		$limit, $const_ffff, $tempvector, $pvalues # select required u16 and clear high part (use zeros from const)

			selb	 $compV0, $compV0, $compV1, $ones_shl_inv
			rotqbybi $listV1, $listV1, $inv_s

			ori	 $compV1, $new_compV1, 0
			rotqbybi $listV0, $listV0, $inv_s

			ceq	$cond_depth_maxm1, $depth, $maxdepthm1	# if (depth == oState->maxdepthm1) ...
			rotqbybi $compV0, $compV0, $s

	cgt			$cond_limit_temp, $limit, $temp		;# if (limit > temp)
			fsmbi	$newbit, 0			;# newbit = 0

			nop
			rotqbi	$listV1, $listV1, $inv_s

	selb		$limit, $limit, $temp, $cond_limit_temp ;#   limit = temp;
			rotqbi	$listV0, $listV0, $inv_s
	
	cgt		$cond_mark_limit, $mark, $limit
	rotqbi		$compV0, $compV0, $s

	# both delayed by 1 clock
	or		$new_distV0, $distV0, $listV0	# distV0 = vec_or(distV0, listV0); (precalc)
	brnz		$cond_mark_limit, break_for

.L_fl3:
	bi		$addr_is_forloop

finish_update_limit_and_exit:
#	nor		$temp2, $distV0, $distV0
#	lqd		$temp, 128($lev_half_depth)	# offsetof(Levels) added here

#	cgt		$cond_depth_hd2, $half_depth2, $depth	;# if (depth < oState->half_depth2)
#		lqx		$tempvector, $pvalues, $p_grvalues	# get data at full address

#	clz		$temp2, $temp2
#		rotqby		$pvalues, $const_vec_to_u16_shuf, $pvalues # create shuffle mask from byte address

#	ai		$temp2, $temp2, 1			;# temp2 = LOOKUP_FIRSTBLANK

	sf		$temp, $temp, $maxlen_m1		;# temp  = ... (ready)

		and		$pvalues, $const_ffff, $pvalues		# set upper word of shuffle mask to zero

	sf		$temp2, $temp2, $temp			;# temp2 = temp - LFB() (probably will use)

		shufb		$limit, $const_ffff, $tempvector, $pvalues # select required u16 and clear high part (use zeros from const)
	
	selb		$temp, $temp, $temp2, $cond_depth_hd2	;# use temp or "temp -= ..."

	cgt		$cond_limit_temp, $limit, $temp		;# if (limit > temp)
	selb		$limit, $limit, $temp, $cond_limit_temp ;#   limit = temp;
	br		save_mark_and_exit


finish_limit_and_save_exit:
	and		$pvalues, $const_ffff, $pvalues		# set upper word of shuffle mask to zero
	shufb		$limit, $const_ffff, $tempvector, $pvalues # select required u16 and clear high part (use zeros from const)
	br		save_mark_and_exit

save_mark_and_exit_dec_lev:
	ai		$lev, $lev, -128	# finally, --lev
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

	;# Weird magic again. See comment before for_loop label.
	;#
	;# This code works a bit better only at given - unaligned - location!
	;# (file offset must be 0x0398). Although other locations are acceptable,
	;# (penalty is just about 24K) there exist really bad ones where speed
	;# drops by 400+K (lose of full clock). E.g. align 64 / offset 0x03C0. Be aware!

	;# Use nops to push the code to best known location

#	nop
#	lnop

break_for:

	# --lev;
	# --depth;

	# POP_LEVEL(lev);
	# copy loop header from for_loop to break dependencies
	# and finish POP_LEVEL
	# decrement of lev delayed to avoid dependency, "-128" added directly to offset

		nor	$s, $comp0_prevlev, $comp0_prevlev	# ~ (comp0 from previous level already here!)
	lqd		$listV0, 0-128($lev)

		ceqi	$cond_comp0_minus1, $comp0_prevlev, -1
	lqd		$listV1, 16-128($lev)

	ai		$depth, $depth, -1
		rotqmbii $s, $s, -1		# magic shift (use slot 1)

		selb	$addr_is_forloop, $addr_no_forloop, $addr_forloop, $cond_comp0_minus1
	lqd		$mark,  96-128($lev)	# int mark  = lev->mark;

	cgt		$cond_depth_stopdepth, $depth, $stopdepth
	lqd		$limit, 112-128($lev)	# int limit = lev->limit;

	ori		$compV0, $comp0_prevlev, 0	# copy comp0 to true place
	lqd		$compV1, 80-128($lev)

		clz		$s, $s			# get it
	lqd		$comp0_prevlev, 64-128-128($lev)	# prefetch comp0 from one more level below

	andc		$distV0, $distV0, $listV0
	fsmbi		$newbit, 0		# newbit = 0

		for_loop_clz_ready_part1

	# while (depth > oState->stopdepth);

	andc		$distV1, $distV1, $listV1
	brz		$cond_depth_stopdepth, save_mark_and_exit_dec_lev	# --lev delayed!

	# together with reordered part2-head
	
	#  if ((mark += s) > limit) { break; }
	a		$mark, $mark, $s
	hbr		.L_fl1, $addr_is_forloop

	# Use free clocks in dependency chain to precalculate

	ai		$lev, $lev, -128	# finally, --lev
	shlqbybi	$ones_shl_inv, $V_ONES, $inv_s	# selMask_inv (use free clock again)

	for_loop_clz_ready_part2_body  .L_fl1


	.align	3
fetch_new_pchoose:
	;#
	;# Called when new portion of pchoose entries must be copied from main
	;# memory using DMA. Since this is very long process by itself, which MUST
	;# not be executed often, optimizations are useless here.
	;#

	# v32_t tempvect = group_insertpos[hash];
	# element        = spu_extract(tempvect, 0) & (GROUPS_LENGTH - 1);
	#               u16               8     [32]
	# pvalues = group_values[hash][element];
	#
	# mfc_get(pvalues, upchoose + group_of32 * GROUP_ELEMENTS * 2, GROUP_ELEMENTS * 2, DMA_ID, 0, 0);

	rotmi		$pvalues, $hash_mul16, -1	# hash * 8 [ hash * 16 / 2 ]
	lqx		$element, $hash_mul16, $p_grinsertpos
		and		$mfc_src, $grpid_vector, $const_ffff
		shli		$mfc_src, $mfc_src, 6		# * 32 * 2
	andi		$element, $element, 7
		a		$mfc_src, $mfc_src, $upchoose
	a		$pvalues, $pvalues, $element	# hash * 8 + element
	shli		$pvalues, $pvalues, 6		# (...) * 32 * 2 (last dimension and u16 to bytes)
	a		$temp, $pvalues, $p_grvalues	# full byte address

	wrch		$ch16, $temp
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
	il		$element, 0

	# mfc_read_tag_status_all();

	wrch		$ch23, $const_2
	rdch		$temp2, $ch24

#	lqd		$tempvector, 0($pvalues)

	br		fetch_pvalues
