# Copyright distributed.net 1997-2003 - All Rights Reserved
# For use in distributed.net projects only.
# Any other distribution or use of this source violates copyright.
#
# Author: Decio Luiz Gazzoni Filho <decio@distributed.net>
# $Id: ogr-cellv1-spe.s,v 1.4 2008/02/19 09:54:45 stream Exp $

	#################################################################
	# Hackers: see the comments right above the .text section for a #
	# general overview of the core architecture and suggestions for #
	# avenues of improvement.                                       #
	#################################################################


	# Offsets for accessing relevant structs


	#  struct Level {
	#    VECTOR listV, compV, distV;
	#    int limit;
	#    U comp0, dist0, list0;
	#    int mark;
	#  };

	# Original struct
	.set	LEVEL_LISTV,		  0	# listV vector
	.set	LEVEL_COMPV,		 16	# compV vector
	.set	LEVEL_DISTV,		 32	# distV vector
	.set	LEVEL_LIMIT,		 48	# limit
	.set	LEVEL_COMP0,		 52	# comp0 scalar
	.set	LEVEL_DIST0,		 56	# dist0 scalar
	.set	LEVEL_LIST0,		 60	# list0 scalar
	.set	LEVEL_MARK,		 64	# mark

	# Modified version of the struct where all scalars are converted to
	# vectors
	.set	LEVEL_QUAD_LISTV,	  0	# listV vector
	.set	LEVEL_QUAD_COMPV,	 16	# compV vector
	.set	LEVEL_QUAD_DISTV,	 32	# distV vector
	.set	LEVEL_QUAD_LIMIT,	 48	# limit
	.set	LEVEL_QUAD_COMP0,	 64	# comp0 scalar
	.set	LEVEL_QUAD_DIST0,	 80	# dist0 scalar
	.set	LEVEL_QUAD_LIST0,	 96	# list0 scalar
	.set	LEVEL_QUAD_MARK,	112	# mark


	# struct State {
	#   int max;
	#   int maxdepth;
	#   int maxdepthm1;
	#   int half_length;
	#   int half_depth;
	#   int half_depth2;
	#   int startdepth;
	#   int depth;
	#   struct Level Levels[OGR_MAXDEPTH];
	#   int node_offset;
	# };

	.set	STATE_MAX,		  0
	.set	STATE_MAXDEPTH,		  4
	.set	STATE_MAXDEPTHM1,	  8
	.set	STATE_HALFLENGTH,	 12
	.set	STATE_HALFDEPTH,	 16
	.set	STATE_HALFDEPTH2,	 20
	.set	STATE_STARTDEPTH,	 24
	.set	STATE_DEPTH,		 28
	.set	STATE_LEVELS,		 32


	# Constants

	.set	SIZEOF_LEVEL,		 80
	.set	SIZEOF_LEVEL_QUAD,	128

	.set	CORE_S_SUCCESS,		  2
	.set	CORE_S_CONTINUE,	  1
	.set	CORE_S_OK,		  0

	.set	BITMAP_LENGTH,		160	# OGR_BITMAPS_LENGTH
	.set	CHOOSE_DIST_BITS,	 12
	.set	ttmDISTBITS,		 32-CHOOSE_DIST_BITS



	# Registers

	# Function arguments
	.set	state,		  3
	.set	pnodes,		  4
	.set	choose,		  5
	.set	OGR,		  6	# We actually use our own version of
					# the OGR array in vector form,
					# declared below. Remember to update
					# it if better OGRs are found.

	# struct Level data
	.set	comp0,		  7
	.set	compV,		  8
	.set	list0,		  9
	.set	listV,		 10
	.set	dist0,		 11
	.set	distV,		 12
	.set	mark,		 13
	.set	limit,		 14

	.set	lev,		 15


	# struct State data

	# Some of the variables here are only used outside of the main loop,
	# e.g. in some of the rarely taken branches, so they're not
	# performance-critical. In the event that a modification to the core
	# requires lots of registers, these can be repurposed, only loading
	# them to temporaries and storing them as soon as you're done.
	.set	depth,		 16
	.set	nodes,		 17
	.set	maxlength,	 18
	.set	remdepth,	 19
	.set	halfdepth,	 20
	.set	halfdepth2,	 21
	.set	half_length,	 22

	.set	levHalfDepth,	 23


	# Miscellaneous variables
	.set	s,		 24	# for counting leading zeros in comp0
	.set	retval,		 25	# return value of the core
	.set	_splatchoose,	 26	# shuffle mask for splatting contents
					# of choose array
	.set	pnodes_data,	 27	# value read from pnodes pointer
	.set	newbit,		 28	# `newbit' shifted in after certain
					# operations
	.set	pslotmask,	 29	# AND mask to clear values not in
					# the preferred slot
	.set	_splatRSM,	 30	# shuffle mask for splatting data
					# read from memory to all reg slots
	.set	ttmDISTBITS_msk, 31	# (unsigned)0xFFFFFFFF >> ttmDISTBITS


	# Temporaries

	# Despite not really running short of registers, an effort was made
	# to minimize the amount of temporaries used, out of respect for
	# some future version of the core where free registers aren't such a
	# commodity, and this is how the current mess was born. Temporaries
	# are mapped to variables and reused in a mostly heuristic way, so
	# there's no point in associating each temporary to a function, they
	# were just numbered.
	#
	# If a good soul would like to inject some sense into this mess,
	# without increasing (at least not too much) the number of used
	# registers, it would be much appreciated.

	.set	temp1,		 32
	.set	temp2,		 33
	.set	temp3,		 34
	.set	temp4,		 35
	.set	temp5,		 36
	.set	temp6,		 37
	.set	temp7,		 38
	.set	temp8,		 39
	.set	temp9,		 40
	.set	temp_RSM,	 temp3
	.set	temp_WSM,	 temp4


	# Variables for the found_one() function

	# This function is very rarely called, so the same advice as above
	# is valid: if pressed for registers, just repurpose these and
	# replace them by loads to temporaries followed by stores before
	# exiting the function.

	.set	i,		 41
	.set	j,		 42
	.set	max,		 43
	.set	maxdepth,	 44
	.set	levels,		 45
	.set	diffs,		 46
	.set	marks_i,	 47
	.set	diff,		 48
	.set	mask,		 49


	# Branch addresses

	# These are used for dynamic branch hints using the hbr instruction.

	.set	stay_addr,	 50
	.set	end_stay_addr,	 51
	.set	up_addr,	 52
	.set	cont_stay_addr,	 53
	.set	cllr_addr,	 54



	.section bss
	.align	4

	# This is a copy of the struct Level array found in struct State,
	# adjusted to a new vector-friendly format which saves a lot of
	# shuffling.
	.lcomm	Levels,		 30*SIZEOF_LEVEL_QUAD

	# Temporary array for the found_one() function (16-bytes elements used)
	.lcomm	diffs_data,	 (((1024 - BITMAP_LENGTH) + 31) / 32) * 16



	.data

	# Vector-friendly version of the OGR array. Remember to update it in
	# case better OGRs are found.
OGR_table:
	.int		  0,   0,   0,   0
	.int		  1,   0,   0,   0
	.int		  3,   0,   0,   0
	.int		  6,   0,   0,   0
	.int		 11,   0,   0,   0
	.int		 17,   0,   0,   0
	.int		 25,   0,   0,   0
	.int		 34,   0,   0,   0
	.int		 44,   0,   0,   0
	.int		 55,   0,   0,   0
	.int		 72,   0,   0,   0
	.int		 85,   0,   0,   0
	.int		106,   0,   0,   0
	.int		127,   0,   0,   0
	.int		151,   0,   0,   0
	.int		177,   0,   0,   0
	.int		199,   0,   0,   0
	.int		216,   0,   0,   0
	.int		246,   0,   0,   0
	.int		283,   0,   0,   0
	.int		333,   0,   0,   0
	.int		356,   0,   0,   0
	.int		372,   0,   0,   0
	.int		425,   0,   0,   0
	.int		480,   0,   0,   0
	.int		492,   0,   0,   0
	.int		553,   0,   0,   0
	.int		585,   0,   0,   0
	.int		623,   0,   0,   0

	# AND mask for isolating values in the preferred slot only
pslotmask_data:
	.int	0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000

	# Shuffle mask for splatting data read from memory to the preferred
	# slot, while zeroing out the other slots
_splatRSM_data:
	.int	0x00010203, 0x80808080, 0x80808080, 0x80808080


	# Now for a note on the general architecture of the core, so anybody
	# wishing to hack it can make some sense of it, at least on a high
	# level.
	#
	# The core started out as a hand-compilation of the C code for
	# ogr_cycle() in ogr/ansi/ogr.cpp, obviously converting scalar code
	# to vector code where possible, so please refer to that code as
	# required. At this stage, the main optimization was careful choice
	# of the order of branch targets, so as to ensure falling through is
	# the most common case, given that 1. it avoids a branch instruction
	# and 2. all branches, including unconditional branches, are
	# predicted not taken in Cell, so a branch hint is mandatory (and
	# would be easy to add, but it might conflict with other branch
	# hints, so it's better to avoid them if at all possible).
	# Eventually the current layout was settled on, which is the up
	# label followed by the stay label followed by the for (;;) loop
	# body. The advantages of this layout, as well as other choices made
	# throughout the core, are supported by the following branch
	# statistics, obtained by instrumenting the C code and running some
	# stubs with it (the branch labels correspond to those found in the
	# C code):
	#
	# if (depth <= halfdepth2)
	# 	taken = 269368662 = 0.150828%
	# 	not taken = 178324477945 = 99.849172%
	# if (depth <= halfdepth)
	# 	taken = 19703508 = 7.314699%
	# 	not taken = 249665154 = 92.685301%
	# if (nodes >= *pnodes)
	# 	taken = 130751 = 0.663592%
	# 	not taken = 19572757 = 99.336408%
	# if (limit > oState->half_length)
	# 	taken = 19572757 = 100.000000%
	# 	not taken = 0 = 0.000000%
	# if (limit >= maxlength - levHalfDepth->mark)
	# 	taken = 249568687 = 99.961361%
	# 	not taken = 96467 = 0.038639%
	# if (comp0 < 0xfffffffe)
	# 	taken = 261747580375 = 69.934750%
	# 	not taken = 112526410949 = 30.065250%
	# if ((mark += s) > limit)
	# 	taken = 83926618910 = 32.063952%
	# 	not taken = 177820961465 = 67.936048%
	# if ((mark += 32) > limit)
	# 	taken = 94667096942 = 84.128780%
	# 	not taken = 17859314007 = 15.871220%
	# if (comp0 == ~0u)
	# 	taken = 17086559646 = 95.673102%
	# 	not taken = 772754361 = 4.326898%
	# if (remdepth == 0)
	# 	taken = 0 = 0.000000%
	# 	not taken = 178593715826 = 100.000000%
	#
	# Also worthy of mention is that at this point, all scalar values
	# were converted to vectors, and even data in memory (such as the
	# struct Level array) were converted to a vector-friendly format to
	# avoid unnecessary shuffling.
	#
	# Next up was instruction scheduling and inclusion of static branch
	# hints where appropriate (for unconditional branches, and for
	# conditional branches which are usually taken according to the
	# statistics above, but which couldn't be made to fall through in
	# the code). Scheduling is noted throughout the code in a simple
	# notation, indicating the number of cycles since the start, and
	# whether the instruction was scheduled in the odd or even pipeline.
	# This is standard practice and doesn't need any more explanation,
	# though it's worth a mention.
	#
	# At this point it was necessary to start changing the structure of
	# the code. The main change was merging the two codepaths in the
	# stay label (comp0 < 0xFFFFFFFE and otherwise). As expected, the
	# main source of slowdowns in the core are (unpredictable) branches,
	# so eliminating this branch provided some significant gains. This
	# also causes another two branches (one inside each codepath) to be
	# merged into a single one. The theory behind this merge is that
	# both cases of the branch perform the same computation, except for
	# the `goto stay' in one of the cases, but this difference is easily
	# dealt with. Also, the macro COMP_LEFT_LIST_RIGHT_32 requires less
	# work than the general case COMP_LEFT_LIST_RIGHT, but according to
	# the branch stats above, the former is executed less frequently so
	# it does not influence performance so much.
	#
	# The other modification was to push back the computation of
	# s = LOOKUP_FIRSTBLANK(comp0), which means considering every point
	# in the code that branches to the stay label and doing the
	# computation there, before branching to stay (or falling through).
	# There are some advantages to doing this, such as:
	#
	# -branch hints have more breathing room to take effect (they have
	#  to be placed 11 cycles plus 4 instruction pairs away from the
	#  branch itself, or the branch will stall until that condition is
	#  satisfied);
	# -filling instruction scheduling gaps in the blocks that branch to
	#  the stay label.
	#
	# The only drawback is code duplication, and it's more of a
	# maintenance problem than anything else since it's a very small
	# amount of code.
	#
	# A similar modification was made with the code that accesses the
	# choose array, but in this case the only other point it needs to be
	# duplicated is the initialization section outside the main loop
	# (SETUP_TOP_STATE).
	#
	# The final source of speedups was adding dynamic branch prediction
	# to the code. This is where some computation is done to decide the
	# outcome of the branch, producing the predicted branch target
	# (depending on whether the branch is taken or not taken), which is
	# then used with the hbr instruction. These will be identified
	# throughout the code.
	#
	# IDEAS FOR SPEEDING UP THE CORE
	#
	# If you want to speed up the core, although you're free to
	# experiment as you wish, here are some ideas to get you started:
	#
	# 1. Work further on branch prediction. If you believe the Cell
	#    simulator, 40%+ of the time is being spent on stalls due to
	#    mispredicted branches (actually I think the figure is lower
	#    since it's simulating incorrectly the behavior of some branch
	#    hints, but it's still a very significant value). This can go
	#    from moving around the branch hint instructions trying to find
	#    a position which yields a better node rate, to devising new
	#    strategies for predicting some of the branches. IIRC, the
	#    branch at `mark_gt_limit' is the most problematic right now, so
	#    I suggest focusing on it.
	#
	# 2. The positioning of branch hints (i.e. how far away they're
	#    placed from the branches and other branch hints), as well as the
	#    memory alignment of branch targets, affect performance in ways
	#    that are rather difficult to predict. To get acquainted with the
	#    issues involved, have a look at
	#
	#    http://www.ibm.com/developerworks/power/library/pa-cellspu/index.html
	#
	#    which has lots of materials on Cell's instruction prefetch
	#    buffers. Also, read up on pipelined hint mode in the Cell BE
	#    Handbook (section 24.3.3.4 on the current version of the docs).
	#    One could in theory trace through the core in the Cell
	#    simulator, but I have strong suspicions that its models are
	#    inaccurate, and in that case a better suggestion is to randomly
	#    move branch hints around and/or randomly insert nops in the
	#    code to change the relative alignment of branch targets, and
	#    who knows, maybe you'll get lucky. Yes, I'm bitter about it,
	#    thanks for asking.
	#
	# 3. Access to the choose array is somewhat slow due to its
	#    byte-oriented layout, which requires a lot of shuffling. One
	#    could try to change the layout to one which is easier to
	#    access, but obviously keeping in mind the size limitations of
	#    the local store array, which has to include code and libraries
	#    apart from data. For instance, a full conversion from a
	#    byte-oriented to a vector-oriented array would overflow the
	#    local store, but I believe going to a word-oriented array would
	#    be feasible, although whether that would speed up array
	#    accesses remains to be seen.
	#
	# 4. Pushing even more code behind branches, as done with the
	#    computation of s = LOOKUP_FIRSTBLANK(comp0), explained above.
	#
	# 5. Replacing instances of unconditional branches with copies of
	#    the code, so the branch and its corresponding hint can be
	#    removed. Note that some instruction encodings, notably branch
	#    hints, require instructions to be close together, so a branch
	#    hint for a branch in a far away point of the code may not
	#    compile due to instruction encoding limitations, so don't go
	#    too wild with this idea unless it brings perceptible
	#    performance gains.
	#
	# 6. The macros PUSH_LEVEL_UPDATE_STATE and POP_LEVEL perform many
	#    loads and stores which could be avoided if the struct Level
	#    array were stored in registers. Now, there aren't enough
	#    registers to store all array elements, but some profiling
	#    revealed that a few levels are visited much more often than
	#    others (IIRC the core stays at levels 5-15 something like 99%
	#    of the time). Hence, by keeping as many levels as possible in
	#    registers, and choosing these levels to be those most often
	#    visited, one can reap virtually all the benefits of this
	#    optimization.
	#
	# 7. An idea that can be combined with item (6) above, or done on
	#    its own, is to make as many copies of the code as there are
	#    levels (IIRC there are 30), so that the level is implied by the
	#    codepath that is currently running. This would bloat the code,
	#    particularly if combined with e.g. item (5), and may not be
	#    feasible in combination with item (3), although I expect to
	#    obtain more gains from this idea than from item (3). A concrete
	#    benefit of implementing this idea is that some computations
	#    which rely on the level (such as incrementing/decrementing
	#    the variables depth, remdepth, lev, ...) could simply be
	#    eliminated since the level is implied. Another possibility,
	#    which is just speculation on my part, is that the branches for
	#    each level may have different behaviors depending on the level,
	#    so that e.g. perhaps a branch that is usually taken in level 10
	#    is usually not taken in level 15. So having different branch
	#    prediction strategies for each level *may* bring about some
	#    performance improvements, but obviously this needs to be
	#    validated by instrumenting the code and collecting branch
	#    statistics.


	.text

	.global ogr_cycle_cellv1_spe_core
	.global _ogr_cycle_cellv1_spe_core

ogr_cycle_cellv1_spe_core:
_ogr_cycle_cellv1_spe_core:

	# Allocate stack space
	# Uncomment this code and adjust it accordingly, in conjunction with
	# the code below, if you need to use registers above and including
	# $80.
	# il		  $2, -16*(127-80+1)
	# a		 $SP,  $SP,   $2

	# Save non-volatile registers
	# Uncomment (parts of) this code if you use registers above and
	# including $80.
	# stqd		 $80,   0($SP)
	# stqd		 $81,  16($SP)
	# stqd		 $82,  32($SP)
	# stqd		 $83,  48($SP)
	# stqd		 $84,  64($SP)
	# stqd		 $85,  80($SP)
	# stqd		 $86,  96($SP)
	# stqd		 $87, 112($SP)
	# stqd		 $88, 128($SP)
	# stqd		 $89, 144($SP)
	# stqd		 $90, 160($SP)
	# stqd		 $91, 176($SP)
	# stqd		 $92, 192($SP)
	# stqd		 $93, 208($SP)
	# stqd		 $94, 224($SP)
	# stqd		 $95, 240($SP)
	# stqd		 $96, 256($SP)
	# stqd		 $97, 272($SP)
	# stqd		 $98, 288($SP)
	# stqd		 $99, 304($SP)
	# stqd		$100, 320($SP)
	# stqd		$101, 336($SP)
	# stqd		$102, 352($SP)
	# stqd		$103, 368($SP)
	# stqd		$104, 384($SP)
	# stqd		$105, 400($SP)
	# stqd		$106, 416($SP)
	# stqd		$107, 432($SP)
	# stqd		$108, 448($SP)
	# stqd		$109, 464($SP)
	# stqd		$110, 480($SP)
	# stqd		$111, 496($SP)
	# stqd		$112, 512($SP)
	# stqd		$113, 528($SP)
	# stqd		$114, 544($SP)
	# stqd		$115, 560($SP)
	# stqd		$116, 576($SP)
	# stqd		$117, 592($SP)
	# stqd		$118, 608($SP)
	# stqd		$119, 624($SP)
	# stqd		$120, 640($SP)
	# stqd		$121, 656($SP)
	# stqd		$122, 672($SP)
	# stqd		$123, 688($SP)
	# stqd		$124, 704($SP)
	# stqd		$125, 720($SP)
	# stqd		$126, 736($SP)
	# stqd		$127, 752($SP)

	# Load OGR table
	ila		$OGR, OGR_table

	# Constant for splats that will be needed in the following loads
	lqa		$_splatRSM, _splatRSM_data

	# This macro reads a struct member with base pointer stored at
	# `base_reg' and offset stored `offset', and saves it to `reg'.
.macro	RdStructMember	reg, base_reg, offset
	# Read value from memory
	lqd		\reg, \offset(\base_reg)

	# Compute effective address offset(base_reg)
	ai		$temp_RSM, \base_reg, \offset

	# Move the desired member of the struct to the preferred slot
	rotqby		\reg, \reg, $temp_RSM

	# Splat the desired member to remaining words of the register
	shufb		\reg, \reg, \reg, $_splatRSM
.endm

	# depth = state->depth + 1;
	RdStructMember	$depth, $state, STATE_DEPTH
	ai		$depth, $depth, 1

	RdStructMember	$maxlength, $state, STATE_MAX

	# remdepth = state->maxdepthm1;
	RdStructMember	$remdepth, $state, STATE_MAXDEPTHM1

	# halfdepth = state->half_depth;
	RdStructMember	$halfdepth, $state, STATE_HALFDEPTH

	# halfdepth2 = state->half_depth2;
	RdStructMember	$halfdepth2, $state, STATE_HALFDEPTH2

	# half_length = state->half_length;
	RdStructMember	$half_length, $state, STATE_HALFLENGTH

	RdStructMember	$pnodes_data, $pnodes, 0

	# Read the struct Level array (30 elements) while storing it in a
	# vector-friendly format.
	ai		$temp1, $state, STATE_LEVELS
	ila		  $lev, Levels
	ila		$temp2, Levels + 30*SIZEOF_LEVEL_QUAD

loop_loadLevels:
	RdStructMember	$comp0, $temp1, LEVEL_COMP0
	stqd		$comp0, LEVEL_QUAD_COMP0($lev)

	RdStructMember	$dist0, $temp1, LEVEL_DIST0
	stqd		$dist0, LEVEL_QUAD_DIST0($lev)

	RdStructMember	$list0, $temp1, LEVEL_LIST0
	stqd		$list0, LEVEL_QUAD_LIST0($lev)

	lqd		$compV, LEVEL_COMPV($temp1)
	stqd		$compV, LEVEL_QUAD_COMPV($lev)

	lqd		$distV, LEVEL_DISTV($temp1)
	stqd		$distV, LEVEL_QUAD_DISTV($lev)

	lqd		$listV, LEVEL_LISTV($temp1)
	stqd		$listV, LEVEL_QUAD_LISTV($lev)

	RdStructMember	 $mark, $temp1, LEVEL_MARK
	stqd		 $mark, LEVEL_QUAD_MARK($lev)

	RdStructMember	$limit, $temp1, LEVEL_LIMIT
	stqd		$limit, LEVEL_QUAD_LIMIT($lev)

	ai		$temp1, $temp1, SIZEOF_LEVEL
	ai		  $lev,   $lev, SIZEOF_LEVEL_QUAD
	ceq		$temp3,   $lev, $temp2
	brz		$temp3, loop_loadLevels

	# Start at level 0 (i.e. Levels + 0*SIZEOF_LEVEL)
	ila		$lev, Levels

	# levHalfDepth = Levels[halfdepth]
	shli		$temp1, $halfdepth, 7
	a		$levHalfDepth, $lev, $temp1

	# lev = Levels[depth]
	shli		$temp1, $depth, 7
	a		  $lev,   $lev, $temp1

	# mark = lev->mark
	lqd		 $mark, LEVEL_QUAD_MARK($lev)

	# nodes = 0
	xor		$nodes, $nodes, $nodes

	# temp1 = state->startdepth;
	RdStructMember	$temp1, $state, STATE_STARTDEPTH

	# remdepth -= depth;
	sf		$remdepth, $depth, $remdepth

	# halfdepth -= startdepth;
	sf		$halfdepth, $temp1, $halfdepth

	# halfdepth2 -= startdepth;
	sf		$halfdepth2, $temp1, $halfdepth2

	# depth -= startdepth;
	sf		$depth, $temp1, $depth

	# Load the shuffle mask for splatting data read from the choose
	# array
	ilhu		$_splatchoose, 0x8080
	iohl		$_splatchoose, 0x8000

	# Load the AND mask required for loading data from the choose array
	ilhu		$ttmDISTBITS_msk, 0xFFF0
	iohl		$ttmDISTBITS_msk, 0x0000

	# Load the AND mask for zeroing data outside the preferred slot
	lqa		$pslotmask, pslotmask_data

	# Load branch address for branch hint instructions
	ila		$stay_addr, stay
	ila		$end_stay_addr, end_stay
	ila		$up_addr, up
	ila		$cont_stay_addr, cont_stay
	ila		$cllr_addr, cllr


	#    #define SETUP_TOP_STATE(lev)  \
	#      U comp0 = lev->comp[0],     \
	#        comp1 = lev->comp[1],     \
	#        comp2 = lev->comp[2],     \
	#        comp3 = lev->comp[3],     \
	#        comp4 = lev->comp[4];     \
	#      U list0 = lev->list[0],     \
	#        list1 = lev->list[1],     \
	#        list2 = lev->list[2],     \
	#        list3 = lev->list[3],     \
	#        list4 = lev->list[4];     \
	#      U dist0 = lev->dist[0],     \
	#        dist1 = lev->dist[1],     \
	#        dist2 = lev->dist[2],     \
	#        dist3 = lev->dist[3],     \
	#        dist4 = lev->dist[4];     \
	#      int newbit = 1;

	# This is interleaved with the first access to the choose array
	# and the computation of s = LOOKUP_FIRSTBLANK(comp0). The method
	# for accessing the choose array may not be immediately clear, but
	# to avoid a duplicate explanation, please refer to the other point
	# in the code where this is done, after the `end_stay' label.

SETUP_TOP_STATE:
	lqd		$comp0, LEVEL_QUAD_COMP0($lev)
	lqd		$compV, LEVEL_QUAD_COMPV($lev)

	lqd		$list0, LEVEL_QUAD_LIST0($lev)
	lqd		$listV, LEVEL_QUAD_LISTV($lev)

	lqd		$dist0, LEVEL_QUAD_DIST0($lev)
	lqd		$distV, LEVEL_QUAD_DISTV($lev)

	il		$newbit, 1
	and		$newbit, $newbit, $pslotmask

	and		$temp1,  $dist0, $ttmDISTBITS_msk
	ai		$temp3, $remdepth, 3

	rotmi		$temp1,  $temp1, -(ttmDISTBITS-2)
	a		$temp3,  $temp3, $temp1
	a		$temp1,  $temp1, $temp1
	a		$temp1,  $temp3, $temp1

	a		$temp2, $choose, $temp1
	lqx		$temp1, $choose, $temp1

	andi		$temp4,  $temp2,    0xF
	a		$temp4,  $temp4, $_splatchoose

	# s = LOOKUP_FIRSTBLANK(comp0);
	nand		$temp5, $comp0, $comp0
	rotmi		$temp5, $temp5, -1

	# if (depth <= halfdepth2)
	ceqi		$temp6, $comp0, -1

	clz		    $s, $temp5

	br		for_loop


	# The main loop starts here. Every instruction inside the main loop
	# is documented with scheduling information. There are two columns,
	# the first being scheduling relative to the beginning of the
	# current label, and the second being relative to the beginning of
	# the loop.
	#
	# A note on listX: only [list|dist|comp]0 is treated as a scalar,
	# whereas [list|dist|comp]1..4 are grouped as a vector in
	# [list|dist|comp]V. This is similar to the organization of the
	# PowerPC vector core.

	# A note on this weird alignment: I found out the hard way that this
	# code only achieves peak performance if the code is aligned on an
	# address == 120 mod 128. The reason for that is outlined above,
	# amidst the suggestions for performance improvement (it's the
	# arrangement that works best with Cell's instruction prefetch
	# buffers). If you shuffle things around such that branch targets
	# land on different addresses, and you don't get an improvement out
	# of your code (or get less than expected), try playing with the
	# alignment before giving up.
	.align	7
	. = . + 120
up:
	#    #define POP_LEVEL(lev)  \
	#      list0 = lev->list[0]; \
	#      list1 = lev->list[1]; \
	#      list2 = lev->list[2]; \
	#      list3 = lev->list[3]; \
	#      list4 = lev->list[4]; \
	#      dist0 &= ~list0;      \
	#      comp0 = lev->comp[0]; \
	#      dist1 &= ~list1;      \
	#      comp1 = lev->comp[1]; \
	#      dist2 &= ~list2;      \
	#      comp2 = lev->comp[2]; \
	#      dist3 &= ~list3;      \
	#      comp3 = lev->comp[3]; \
	#      dist4 &= ~list4;      \
	#      comp4 = lev->comp[4]; \
	#      newbit = 0;

	# depth--;
	ai		$depth, $depth, -1				 #  1e	 1e
	# Load comp0
	lqd		$comp0, LEVEL_QUAD_COMP0-SIZEOF_LEVEL_QUAD($lev) #  1o	 1o

	# remdepth++;
	ai		$remdepth, $remdepth, 1				 #  2e	 2e
	# Load list0
	lqd		$list0, LEVEL_QUAD_LIST0-SIZEOF_LEVEL_QUAD($lev) #  2o	 2o

	xor		$newbit, $newbit, $newbit			 #  3e	 3e
	# Load listV
	lqd		$listV, LEVEL_QUAD_LISTV-SIZEOF_LEVEL_QUAD($lev) #  3o	 3o

	# if (depth <= 0)
	cgti		$temp2, $depth, 0				 #  4e	 4e
	# Load compV
	lqd		$compV, LEVEL_QUAD_COMPV-SIZEOF_LEVEL_QUAD($lev) #  4o	 4o

	# s = LOOKUP_FIRSTBLANK(comp0);
	#
	# A note on LOOKUP_FIRSTBLANK(): since we only have a `count leading
	# zeros' instruction, first we have to negate the argument, so
	# temp1 = ~comp0.
	nand		$temp1, $comp0, $comp0				 #  7e	 7e
	# limit = lev->limit;
	lqd		$limit, LEVEL_QUAD_LIMIT-SIZEOF_LEVEL_QUAD($lev) #  7o	 7o

	# dist0 &= ~list0;
	#
	# Note that on input, values outside the preferred slot are 0, and
	# are assumed to remain so. This is the case due to ANDing with
	# dist0.
	andc		$dist0, $dist0, $list0				 #  8e	 8e
	# mark = lev->mark;
	lqd		 $mark, LEVEL_QUAD_MARK-SIZEOF_LEVEL_QUAD($lev)  #  8o	 8o

	# Also, LOOKUP_FIRSTBLANK() expects the result to be off by one, so
	# we shift the input right by one place, so temp1 = (~comp0) >> 1.
	rotmi		$temp1, $temp1, -1				 #  9e	 9e

	# distV &= ~listV;
	andc		$distV, $distV, $listV				 # 10e	10e

	# Test whether comp0 == 0xFFFFFFFF, used in a branch in the stay
	# label below.
	ceqi		$temp6, $comp0, -1				 # 11e	11e
	# branch if depth <= 0
	brz		$temp2, depth_le_0				 # 11o	11o

	# Finally, count leading zeros on temp1 = (~comp0) >> 1.
	clz		    $s, $temp1					 # 12e	12e

	# Compute branch target for the end_stay_branch below. This branch
	# is always predicted correctly, since the hint is based on the same
	# value used in the branch itself.
	selb		$temp7, $end_stay_addr, $cont_stay_addr, $temp6	 # 13e	13e

	# lev--;
	ai		  $lev,   $lev, -SIZEOF_LEVEL_QUAD		 # 14e	14e

	# goto stay; -- just fall through

	.align		3
stay:
	# mark += s;
	a		 $mark,  $mark,     $s				 #  1e	15e
	# Hint for branch below.
	hbr		end_stay_branch, $temp7				 #  1o	15o

	# Compute 32 - s, required for some shifts below
	sfi		$temp1,     $s,     32				 #  2e	16e
	# Set temp8 = list0 using an odd-pipeline instruction
	rotqbii		$temp8, $list0,      0				 #  2o	16o

	# Test whether (mark += s) > limit
	cgt		$temp7,  $mark, $limit				 #  3e  17e

	# Compute -s and -(32 - s), required for some shifts below
	sfi		$temp2,     $s,      0				 #  4e  18e

	sfi		$temp3, $temp1,      0				 #  5e  19e

	# if ((mark += s) > limit) goto up;
	# According to our branch statistics above, this is taken with
	# probability 32.1%
mark_gt_limit:
	brnz		$temp7, up					 #  5o	19o


	#    #define COMP_LEFT_LIST_RIGHT(lev, s)    \
	#      U temp1, temp2;                       \
	#      int ss = 32 - (s);                    \
	#      comp0 <<= s;                          \
	#      temp1 = newbit << ss;                 \
	#      temp2 = list0 << ss;                  \
	#      list0 = (list0 >> (s)) | temp1;       \
	#      temp1 = list1 << ss;                  \
	#      list1 = (list1 >> (s)) | temp2;       \
	#      temp2 = list2 << ss;                  \
	#      list2 = (list2 >> (s)) | temp1;       \
	#      temp1 = list3 << ss;                  \
	#      list3 = (list3 >> (s)) | temp2;       \
	#      temp2 = comp1 >> ss;                  \
	#      list4 = (list4 >> (s)) | temp1;       \
	#      temp1 = comp2 >> ss;                  \
	#      comp0 |= temp2;                       \
	#      temp2 = comp3 >> ss;                  \
	#      comp1 = (comp1 << (s)) | temp1;       \
	#      temp1 = comp4 >> ss;                  \
	#      comp2 = (comp2 << (s)) | temp2;       \
	#      comp4 = comp4 << (s);                 \
	#      comp3 = (comp3 << (s)) | temp1;       \
	#      newbit = 0;                           

	# Note that the conversion of this code to vector form implies that
	# it doesn't look much like the macro above. It's easier to think in
	# terms of what the code is supposed to accomplish, rather than
	# paying attention to low level details. Suppose we have 160-bit
	# values comp and list, an integer n and a bit newbit. We want to
	# shift comp left by n bits and shift list right by n bits, while
	# shifting in the value newbit.

	# A note about 128-bit shifting in Cell: given a shift amount s, the
	# shifts are broken up in two stages, one that shifts by a number of
	# bytes (8*floor(s/8)) and one that shifts by the number of bits
	# remaining (s mod 8). Yeah, it's kind of a mess.

	.align		3
cllr:
	# comp0 <<= s;
	shl		$comp0, $comp0,     $s				#  1e	20e
	# listV >>= s; (shift bits)
	rotqmbi		$listV, $listV, $temp2				#  1o	20o

	# temp4 = compV >> ss; (note that we only want the value in the
	# preferred slot in this case, so it's a vector 32-bit shift, not a
	# 128-bit shift).
	rotm		$temp4, $compV, $temp3				#  2e	21e
	# compV <<= s; (shift bits)
	shlqbi		$compV, $compV,     $s				#  2o	21o

	# This is required to work around the brokenness of rotqmbybi. To
	# see what I mean, please refer to
	# http://www.insomniacgames.com/tech/articles/0807/why_rotqmbybi_is_broken.php
	sfi		$temp5,     $s,      7				#  3e	22e
	# temp1 = newbit << ss; (note that temp1 here is in the notation
	# of the comment above and is completely unrelated to the register
	# $temp1 -- in fact the result is saved to newbit itself).
	shl		$newbit, $newbit, $temp1			#  4e	23e

	# list0 >>= s;
	rotm		$list0, $list0, $temp2				#  5e	24e
	# listV >>= s; (shift bytes)
	rotqmbybi	$listV, $listV, $temp5				#  5o	24o

	# Shift in the value that was shifted out of compV
	or		$comp0, $comp0, $temp4				#  6e	25e
	# compV <<= s; (shift bytes)
	shlqbybi	$compV, $compV,     $s				#  6o	25o

	# temp4 = list0 >> s; (recall that temp8 is a copy of the original
	# value of list0)
	shl		$temp4, $temp8, $temp1				#  7e	26e
	# Branch hint for the unconditional branch below. This has to be
	# carefully positioned to ensure pipelined hint mode is activated,
	# or it will cancel the effect of the branch hint for the branch at
	# `end_stay_branch'.
	hbra		stay_branch, stay				#  7o	26o

	# Make a copy of newbit
	lr		$temp5, $newbit					#  8e	27e
	lnop								#  8o	27o

	# Zero values outside of the preferred slot
	and		$comp0, $comp0, $pslotmask			#  9e	28e
	# Zero newbit using an odd-pipeline instruction
	rotqmbyi	$newbit, $newbit, -16				#  9o	28o

	# Shift in newbit
	or		$list0, $list0, $temp5				# 10e	29e
	lnop								# 10o	29o

	# Shift in the value that was shifted out of list0
	or		$listV, $listV, $temp4				# 11e	30e

	# if (comp0 == ~0u) { ...; goto stay; } (i.e. fall through here)
end_stay_branch:
	brz		$temp6, end_stay				# 11o	30o

cont_stay:
	# s = LOOKUP_FIRSTBLANK(comp0);
	# Refer to the comments for the same code snippet at the `up' target
	# if necessary.
	nand		$temp1, $comp0, $comp0				# 12e	31e
	ceqi		$temp6, $comp0, -1				# 13e	32e

	rotmi		$temp1, $temp1, -1				# 14e	33e
	selb		$temp7, $end_stay_addr, $cont_stay_addr, $temp6	# 15e	34e

	clz		    $s, $temp1					# 18e	37e
	# goto stay;
stay_branch:
	br		stay						# 18o	37o

	.align		3
end_stay:
	nop								#  1e	38e
	# if (remdepth == 0)
	brz		$remdepth, remdepth_eq_0			#  1o	38o

	#    #define PUSH_LEVEL_UPDATE_STATE(lev)    \
	#      lev->list[0] = list0; dist0 |= list0; \
	#      lev->list[1] = list1; dist1 |= list1; \
	#      lev->list[2] = list2; dist2 |= list2; \
	#      lev->list[3] = list3; dist3 |= list3; \
	#      lev->list[4] = list4; dist4 |= list4; \
	#      lev->comp[0] = comp0; comp0 |= dist0; \
	#      lev->comp[1] = comp1; comp1 |= dist1; \
	#      lev->comp[2] = comp2; comp2 |= dist2; \
	#      lev->comp[3] = comp3; comp3 |= dist3; \
	#      lev->comp[4] = comp4; comp4 |= dist4; \
	#      newbit = 1;

	# dist0 |= list0;
	or		$dist0, $dist0, $list0				#  2e	39e
	# Store comp0
	stqd		$comp0, LEVEL_QUAD_COMP0($lev)			#  2o	39o

	# distV |= listV;
	or		$distV, $distV, $listV				#  3e	40e
	# Store compV
	stqd		$compV, LEVEL_QUAD_COMPV($lev)			#  3o	40o

	# limit = maxlength - choose(dist0 >> ttmDISTBITS, remdepth);
	# define choose(x,y) (ogr_choose_dat[((x)<<3)+((x)<<2)+(y)+3])

	# Instead of computing 12*x, where x = dist0 >> ttmDISTBITS, we
	# compute 3*x', where x' = (dist0 & 0xFFF00000) >> (ttmDISTBITS-2).
	#
	# temp1 = dist0 & 0xFFF00000;
	and		$temp1,  $dist0, $ttmDISTBITS_msk		#  4e	41e
	# Branch hint for the unconditional branch below.
	hbra		branch_end_for_loop, stay			#  4o	41o

	# remdepth--;
	ai		$remdepth, $remdepth, -1			#  5e	42e

	# temp1 = (dist0 & 0xFFF00000) >> (ttmDISTBITS-2);
	rotmi		$temp1,  $temp1, -(ttmDISTBITS-2)		#  6e	43e

	# temp3 = remdepth + 3;
	ai		$temp3, $remdepth, 3				#  7e	44e
	# lev->mark = mark;
	stqd		$mark, LEVEL_QUAD_MARK($lev)			#  7o	44o

	# comp0 |= dist0;
	or		$comp0, $comp0, $dist0				#  8e	45e
	# Store list0
	stqd		$list0, LEVEL_QUAD_LIST0($lev)			#  8o	45o

	# compV |= distV;
	or		$compV, $compV, $distV				#  9e	46e
	# Store listV
	stqd		$listV, LEVEL_QUAD_LISTV($lev)			#  9o	46o

	# temp3 = ((dist0 & 0xFFF00000) >> (ttmDISTBITS-2)) + remdepth + 3;
	a		$temp3,  $temp3, $temp1				# 10e	47e
	# s = LOOKUP_FIRSTBLANK(comp0);
	# Again, see comments at the `up' label for explanations regarding
	# these operations.
	nand		$temp5, $comp0, $comp0				# 11e	48e

	# temp1 = 2*((dist0 & 0xFFF00000) >> (ttmDISTBITS-2));
	a		$temp1,  $temp1, $temp1				# 12e	49e
	rotmi		$temp5, $temp5, -1				# 13e	50e

	# temp1 = 3*((dist0 & 0xFFF00000) >> (ttmDISTBITS-2)) + remdepth + 3;
	# This is the index to the choose array.
	a		$temp1,  $temp3, $temp1				# 14e	51e
	# Test whether comp0 == 0xFFFFFFFF
	ceqi		$temp6, $comp0, -1				# 15e	52e

	# Compute effective address of the choose array element in memory,
	# to shuffle it into place in the preferred slot later. This is due
	# to Cell's alignment restrictions.
	a		$temp2, $choose, $temp1				# 16e	53e
	# Load the element from the choose array
	lqx		$temp1, $choose, $temp1				# 16o	53o

	# Compute branch target (explained below)
	selb		$temp9, $cllr_addr, $up_addr, $temp6		# 17e	54e
	# newbit = 1;
	il		$newbit, 1					# 18e	55e

	# Recall temp2 is the effective address of the target choose array
	# element in memory, and since Cell performs aligned loads (i.e.
	# reading from the address temp2 & 0xFFFFFFF0), it follows that the
	# address' least significant nibble indicates which of the byte
	# slots in a register the target element will occupy. Here we
	# isolate it, then insert it in a shuffle mask to shuffle it into
	# position.
	andi		$temp4,  $temp2,    0xF				# 19e	56e
	# This is an heuristic branch hint. Refer to the branch statistics
	# above to see that, when comp0 == 0xFFFFFFFF, the branch is usually
	# taken. Mispredicts may happen, but still there's a net gain.
	hbr		mark_gt_limit, $temp9				# 19o	56o

	# lev++;
	ai		$lev, $lev, SIZEOF_LEVEL_QUAD			# 20e	57e
	lnop								# 20o	57o

	# temp4 contains the byte slot occupied by the target choose array
	# element. Insert it into a shuffle mask as explained above.
	a		$temp4,  $temp4, $_splatchoose			# 21e	58e
	# depth++;
	ai		$depth, $depth, 1				# 22e	59e

	# Zero out values of newbit outside the preferred slot
	and		$newbit, $newbit, $pslotmask			# 23e	60e
	# Shuffle the target choose array element into position.
	shufb		$temp1,  $temp1, $temp1, $temp4			# 23o	60o

	clz		    $s, $temp5					# 22e	61e

	# continue; -- just fall through

	# This is one of the cases where Cell is just fragile: the code
	# below is scheduled incorrectly and 3 cycles could be gained by
	# removing the unnecessary lnop below. However, do that and the node
	# rate drops by a huge amount. This is most likely related to the
	# relative positioning of the two branch hints above and the branch
	# below; if they do not obey certain rules (which are not clear to
	# me) then pipelined hint mode is not enabled and the second hint
	# just replaces the first. Feel free to experiment with this by
	# removing the lnop and trying to find an appropriate placement of
	# the branch hints above such that they work as expected.

	.align		3
for_loop:
	# Test whether depth > halfdepth2
	cgt		$temp2, $depth, $halfdepth2			#  1e	62e

	# Compute branch target for the end_stay_branch below. This branch
	# is always predicted correctly, since the hint is based on the same
	# value used in the branch itself.
	selb		$temp7, $end_stay_addr, $cont_stay_addr, $temp6	#  2e	63e
	lnop								#  3o	64o

	# Finish the computation of limit
	sf		$limit,  $temp1, $maxlength			#  4e	65e
	# Taken with probability 0.15% according to our branch statistics,
	# so assume it falls through
	brz		$temp2, depth_le_halfdepth2			#  5o	66o

cont_depth_le_halfdepth2:
	# nodes++;
	ai		$nodes, $nodes, 1				#  6e   67e
	# lev->limit = limit;
	stqd		$limit, LEVEL_QUAD_LIMIT($lev)			#  7o	68o

branch_end_for_loop:
	# goto stay;
	br		stay						#  8o	69o


	# This is the end of the main loop. Anything outside the main loop
	# is rarely executed and is not worth the optimization effort,
	# except for freeing up some registers if required. Hence from now
	# on the comments won't be as detailed as before.

	.align		3
depth_le_0:
	# retval = CORE_S_OK;
	il		$retval, CORE_S_OK

	# break;
	br		epilogue


	.align		3
depth_le_halfdepth2:
	# if (depth <= halfdepth)
	cgt		$temp1, $depth, $halfdepth
	brz		$temp1, depth_le_halfdepth

	# else if (limit >= maxlength - levHalfDepth->mark)
	lqd		$temp1, LEVEL_QUAD_MARK($levHalfDepth)
	ai		$temp2, $maxlength, -1
	sf		$temp1, $temp1, $temp2

	cgt		$temp2, $limit, $temp1

	#   limit = maxlength - levHalfDepth->mark - 1;
	selb		$limit, $limit, $temp1, $temp2

cont_depth_le_halfdepth:
	br		cont_depth_le_halfdepth2


	.align		3
depth_le_halfdepth:
	# if (nodes >= *pnodes)
	ai		$temp1, $nodes, 1
	clgt		$temp1, $nodes, $pnodes_data

	# Taken with probability 0.66%
	brnz		$temp1, nodes_ge_pnodes

	# limit = maxlength - OGR[remdepth];
	shli		$temp1, $remdepth, 4
	lqx		$temp1, $OGR, $temp1

	sf		$limit, $temp1, $maxlength

	# if (limit > half_length)
	cgt		$temp1, $limit, $half_length

	#   limit = half_length;
	selb		$limit, $limit, $half_length, $temp1

	br		cont_depth_le_halfdepth


	.align		3
nodes_ge_pnodes:
	# retval = CORE_S_CONTINUE;
	il		$retval, CORE_S_CONTINUE

	# break;
	br		epilogue


	.align		3
remdepth_eq_0:
	# retval = found_one();
	br		found_one

	.align		3
cont_found_one:
	# if (retval != CORE_S_CONTINUE)
	ceqi		$temp1, $retval, CORE_S_CONTINUE

	# break;
	brz		$temp1, epilogue

	# s = LOOKUP_FIRSTBLANK(comp0);
	nand		$temp1, $comp0, $comp0
	rotmi		$temp1, $temp1, -1

	selb		$temp7, $end_stay_addr, $stay_addr, $temp6
	lnop

	clz		    $s, $temp1

	# goto stay
	br		stay



	.align		3
epilogue:
	#    #define SAVE_FINAL_STATE(lev) \
	#      lev->list[0] = list0;       \
	#      lev->list[1] = list1;       \
	#      lev->list[2] = list2;       \
	#      lev->list[3] = list3;       \
	#      lev->list[4] = list4;       \
	#      lev->dist[0] = dist0;       \
	#      lev->dist[1] = dist1;       \
	#      lev->dist[2] = dist2;       \
	#      lev->dist[3] = dist3;       \
	#      lev->dist[4] = dist4;       \
	#      lev->comp[0] = comp0;       \
	#      lev->comp[1] = comp1;       \
	#      lev->comp[2] = comp2;       \
	#      lev->comp[3] = comp3;       \
	#      lev->comp[4] = comp4;
	stqd		$list0, LEVEL_QUAD_LIST0($lev)
	stqd		$listV, LEVEL_QUAD_LISTV($lev)

	stqd		$dist0, LEVEL_QUAD_DIST0($lev)
	stqd		$distV, LEVEL_QUAD_DISTV($lev)

	stqd		$comp0, LEVEL_QUAD_COMP0($lev)
	stqd		$compV, LEVEL_QUAD_COMPV($lev)

.macro	WrStructMember	reg, base_reg, offset
	# Read original values
	lqd		$temp_RSM, \offset(\base_reg)

	# Generate insertion mask for the struct member
	cwd		$temp_WSM, \offset(\base_reg)

	# Insert the struct member
	shufb		$temp_RSM, \reg, $temp_RSM, $temp_WSM

	# Write new values
	stqd		$temp_RSM, \offset(\base_reg)
.endm

	# lev->mark = mark;
	stqd		$mark, LEVEL_QUAD_MARK($lev)

	# depth = depth - 1 + startdepth;
	RdStructMember	$temp1, $state, STATE_STARTDEPTH
	ai		$depth, $depth, -1
	a		$depth, $depth, $temp1

	# state->depth = depth;
	WrStructMember	$depth, $state, STATE_DEPTH

	# *pnodes = nodes;
	WrStructMember	$nodes, $pnodes, 0

	ai		$temp1, $state, STATE_LEVELS
	ila		  $lev, Levels
	ila		$temp2, Levels + 30*SIZEOF_LEVEL_QUAD

loop_storeLevels:
	lqd		$comp0, LEVEL_QUAD_COMP0($lev)
	WrStructMember	$comp0, $temp1, LEVEL_COMP0

	lqd		$dist0, LEVEL_QUAD_DIST0($lev)
	WrStructMember	$dist0, $temp1, LEVEL_DIST0

	lqd		$list0, LEVEL_QUAD_LIST0($lev)
	WrStructMember	$list0, $temp1, LEVEL_LIST0

	lqd		$compV, LEVEL_QUAD_COMPV($lev)
	stqd		$compV, LEVEL_COMPV($temp1)

	lqd		$distV, LEVEL_QUAD_DISTV($lev)
	stqd		$distV, LEVEL_DISTV($temp1)

	lqd		$listV, LEVEL_QUAD_LISTV($lev)
	stqd		$listV, LEVEL_LISTV($temp1)

	lqd		 $mark, LEVEL_QUAD_MARK($lev)
	WrStructMember	 $mark, $temp1, LEVEL_MARK

	lqd		$limit, LEVEL_QUAD_LIMIT($lev)
	WrStructMember	$limit, $temp1, LEVEL_LIMIT

	ai		$temp1, $temp1, SIZEOF_LEVEL
	ai		  $lev,   $lev, SIZEOF_LEVEL_QUAD
	ceq		$temp3,   $lev, $temp2
	brz		$temp3, loop_storeLevels

	# Return value goes on $3
	lr		  $3, $retval

	# Restore registers
	# Uncomment (parts of) this code if you use registers above and
	# including $80.
	# lqd		 $80,   0($SP)
	# lqd		 $81,  16($SP)
	# lqd		 $82,  32($SP)
	# lqd		 $83,  48($SP)
	# lqd		 $84,  64($SP)
	# lqd		 $85,  80($SP)
	# lqd		 $86,  96($SP)
	# lqd		 $87, 112($SP)
	# lqd		 $88, 128($SP)
	# lqd		 $89, 144($SP)
	# lqd		 $90, 160($SP)
	# lqd		 $91, 176($SP)
	# lqd		 $92, 192($SP)
	# lqd		 $93, 208($SP)
	# lqd		 $94, 224($SP)
	# lqd		 $95, 240($SP)
	# lqd		 $96, 256($SP)
	# lqd		 $97, 272($SP)
	# lqd		 $98, 288($SP)
	# lqd		 $99, 304($SP)
	# lqd		$100, 320($SP)
	# lqd		$101, 336($SP)
	# lqd		$102, 352($SP)
	# lqd		$103, 368($SP)
	# lqd		$104, 384($SP)
	# lqd		$105, 400($SP)
	# lqd		$106, 416($SP)
	# lqd		$107, 432($SP)
	# lqd		$108, 448($SP)
	# lqd		$109, 464($SP)
	# lqd		$110, 480($SP)
	# lqd		$111, 496($SP)
	# lqd		$112, 512($SP)
	# lqd		$113, 528($SP)
	# lqd		$114, 544($SP)
	# lqd		$115, 560($SP)
	# lqd		$116, 576($SP)
	# lqd		$117, 592($SP)
	# lqd		$118, 608($SP)
	# lqd		$119, 624($SP)
	# lqd		$120, 640($SP)
	# lqd		$121, 656($SP)
	# lqd		$122, 672($SP)
	# lqd		$123, 688($SP)
	# lqd		$124, 704($SP)
	# lqd		$125, 720($SP)
	# lqd		$126, 736($SP)
	# lqd		$127, 752($SP)

	# Restore stack pointer
	# Uncomment this code and adjust it accordingly, in conjunction with
	# the code above, if you need to use registers above and including
	# $80.
	# il		  $2, 16*(127-80+1)
	# a		 $SP,  $SP,   $2

	# Return to caller
	bi		$LR


	.align		3
found_one:
	# max = state->max;
	RdStructMember	$max, $state, STATE_MAX

	# maxdepth = state->maxdepth;
	RdStructMember	$maxdepth, $state, STATE_MAXDEPTH
	shli		$maxdepth, $maxdepth, 7

	# levels = state->Levels[0]
	ila		$levels, Levels + LEVEL_QUAD_MARK

	# Load diffs
	ila		$diffs, diffs_data

	# Assume retval = CORE_S_CONTINUE
	il		$retval, CORE_S_CONTINUE

	# zero diffs[max >> (5+1) .. 0]
	rotmi		$i, $max, -(5+1)
	shli		$i, $i, 4
	xor		$temp1, $temp1, $temp1
	ai		$i, $i, 16

loop_zerodiffs:
	ai		$i, $i, -16
	stqx		$temp1, $diffs, $i
	brnz		$i, loop_zerodiffs

	# for (i = 1; i < maxdepth; i++)
	il		$i, 128

loop_maxdepth:
	# marks_i = levels[i].mark;
	lqx		$marks_i, $levels, $i

	# for (j = 0; j < i; j++)
	xor		$j, $j, $j

loop_0_i:
	# diff = marks_i - levels[j].mark
	lqx		$diff, $levels, $j
	sf		$diff, $diff, $marks_i

	# if (diff <= OGR_BITMAPS_LENGTH)
	cgti		$temp1, $diff, BITMAP_LENGTH
	#   break;
	brz		$temp1, inc_i

	# if (diff+diff <= max)
	a		$temp1, $diff, $diff
	cgt		$temp1, $temp1, $max
	brnz		$temp1, inc_j

	# mask = 1 << (diff & 31);
	il		$temp1, 1
	andi		$temp2, $diff, 31
	shl		$mask, $temp1, $temp2

	# diff = (diff >> 5) - (OGR_BITMAPS_LENGTH / 32);
	rotmi		$diff, $diff, -5
	ai		$diff, $diff, -(BITMAP_LENGTH/32)

	# if ((diffs[diff] & mask) != 0)
	shli		$temp1, $diff, 4
	lqx		$temp1, $diffs, $temp1
	and		$temp1, $temp1, $mask

	#   return CORE_S_CONTINUE;
	brnz		$temp1, cont_found_one

inc_j:
	ai		$j, $j, 128
	cgt		$temp3, $i, $j
	brnz		$temp3, loop_0_i

inc_i:
	ai		$i, $i, 128
	cgt		$temp1, $maxdepth, $i
	brnz		$temp1, loop_maxdepth

	il		$retval, CORE_S_SUCCESS

	br		cont_found_one
