# Copyright distributed.net 1997-2003 - All Rights Reserved
# For use in distributed.net projects only.
# Any other distribution or use of this source violates copyright.
#
# Author: Decio Luiz Gazzoni Filho <decio@distributed.net>
# $Id: ogr-cellv1-spe.s,v 1.1.2.1 2007/08/20 15:39:00 decio Exp $

	#  struct Level {
	#    VECTOR listV, compV, distV;
	#    int limit;
	#    U comp0, dist0, list0;
	#    int mark;
	#  };

	.set	LEVEL_LISTV,		  0	# listV vector
	.set	LEVEL_COMPV,		 16	# compV vector
	.set	LEVEL_DISTV,		 32	# distV vector
	.set	LEVEL_COMP0,		 52	# comp0 scalar
	.set	LEVEL_DIST0,		 56	# dist0 scalar
	.set	LEVEL_LIST0,		 60	# list0 scalar

	.set	LEVEL_QUAD_LISTV,	  0	# listV vector
	.set	LEVEL_QUAD_COMPV,	 16	# compV vector
	.set	LEVEL_QUAD_DISTV,	 32	# distV vector
	.set	LEVEL_QUAD_LIMIT,	 48	# limit
	.set	LEVEL_QUAD_COMP0,	 64	# comp0 scalar
	.set	LEVEL_QUAD_DIST0,	 80	# dist0 scalar
	.set	LEVEL_QUAD_LIST0,	 96	# list0 scalar
	.set	LEVEL_QUAD_MARK,	112

	# struct State {
	#   int max;
	#   int maxdepth;
	#   int maxdepthm1;
	#   int half_length;
	#   int half_depth;
	#   int half_depth2;
	#   int startdepth;
	#   int depth;
	#   struct Level Levels[MAXDEPTH];
	#   int node_offset;
	# };

	# Structure members dependencies (offsets)
	.set	STATE_DEPTH,		 28
	.set	STATE_LEVELS,		 32
	.set	SIZEOF_LEVEL,		 80
	.set	SIZEOF_LEVEL_QUAD,	128
	.set	STATE_MAX,		  0	# OGR[stub.marks-1] = max length
	.set	STATE_MAXDEPTH,		  4
	.set	STATE_MAXDEPTHM1,	  8	# stub.marks - 1
	.set	STATE_STARTDEPTH,	 24	# workstub->stub.length
	.set	STATE_HALFDEPTH,	 16	# first mark of the middle segment
	.set	STATE_HALFDEPTH2,	 20	# last mark of the middle segment
	.set	STATE_HALFLENGTH,	 12	# maximum position of the middle segment
	.set	LEVEL_MARK,		 64
	.set	LEVEL_LIMIT,		 48
	.set	SIZEOF_OGR,		  4
	.set	OGR_SHIFT,		  2

	# Constants
	.set	CORE_S_SUCCESS,		  2
	.set	CORE_S_CONTINUE,	  1
	.set	CORE_S_OK,		  0
	.set	BITMAP_LENGTH,		160	# bitmap : 5 * 32-bits
#	.set	DIST_SHIFT,		 20
#	.set	DIST_BITS,		 32-DIST_SHIFT
	.set	CHOOSE_DIST_BITS,	 12
	.set	ttmDISTBITS,		 32-CHOOSE_DIST_BITS


	# Parameters for rlwinm (choose addressing)
	.set	DIST_SH, DIST_BITS
	.set	DIST_MB, DIST_SHIFT
	.set	DIST_ME, 31

	# Registers

	# Core arguments
	.set	state,			  3
	.set	pnodes,			  4
	.set	choose,			  5
	.set	OGR,			  6

	# struct Level data
	.set	comp0,			  7
	.set	compV,			  8
	.set	list0,			  9
	.set	listV,			 10
	.set	dist0,			 11
	.set	distV,			 12
	.set	mark,			 13
	.set	limit,			 14

	.set	lev,			 15

	# struct State {
	#   int max;
	#   int maxdepth;
	#   int maxdepthm1;
	#   int half_length;
	#   int half_depth;
	#   int half_depth2;
	#   int startdepth;
	#   int depth;
	#   struct Level Levels[MAXDEPTH];
	#   int node_offset;
	# };

	# struct State data
	.set	depth,			 16
	.set	nodes,			 17
	.set	maxlength,		 18
	.set	remdepth,		 19
	.set	halfdepth,		 20
	.set	halfdepth2,		 21
	.set	half_length,		 22

	.set	levHalfDepth,		 23

	# misc
	.set	s,			 24
	.set	retval,			 25
	.set	_splatchoose,		 26
	.set	pnodes_data,		 27
	.set	newbit,			 28
	.set	shr32,			 29
	.set	shl32v,			 30
	.set	shl32s,			 31
	.set	pslotmask,		 32
	.set	_splatRSM,		 33
	.set	ttmDISTBITS_mask,	 55

	# temporaries
	.set	temp1,			 34
	.set	temp2,			 35
	.set	temp3,			 36
	.set	temp4,			 37
	.set	temp5,			 38
	.set	temp6,			 39
	.set	temp7,			 40
	.set	temp8,			 41
	.set	temp_RSM,		 temp3
	.set	temp_WSM,		 temp4

	# found_one()
	.set	i,			 42
	.set	j,			 43
	.set	max,			 44
	.set	maxdepth,		 45
	.set	levels,			 46
	.set	diffs,			 47
	.set	marks_i,		 48
	.set	diff,			 49
	.set	mask,			 50

	# branches
	.set	stay_addr,		 51
	.set	end_stay_addr,		 52
	.set	up_addr,		 53
	.set	cont_stay_addr,		 54


	.section bss
	.align	4

	.lcomm	Levels,			 30*SIZEOF_LEVEL_QUAD
	.lcomm	diffs_data,		 ((1024 - BITMAP_LENGTH) + 31) / 32



	.data

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

pslotmask_data:
	.int	0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000
_splatRSM_data:
	.int	0x00010203, 0x80808080, 0x80808080, 0x80808080
shr32_data:
	.int	0x00010203, 0x10111213, 0x14151617, 0x18191A1B
shl32v_data:
	.int	0x14151617, 0x18191A1B, 0x1C1D1E1F, 0x80808080
shl32s_data:
	.int	0x10111213, 0x80808080, 0x80808080, 0x80808080



	.text

	.global ogr_cycle_cellv1_spe_core
	.global _ogr_cycle_cellv1_spe_core

ogr_cycle_cellv1_spe_core:
_ogr_cycle_cellv1_spe_core:

	# Allocate stack space
	il		  $2, -16*(127-80+1)
	a		 $SP,  $SP,   $2

	# Save non-volatile registers
	stqd		 $80,   0($SP)
	stqd		 $81,  16($SP)
	stqd		 $82,  32($SP)
	stqd		 $83,  48($SP)
	stqd		 $84,  64($SP)
	stqd		 $85,  80($SP)
	stqd		 $86,  96($SP)
	stqd		 $87, 112($SP)
	stqd		 $88, 128($SP)
	stqd		 $89, 144($SP)
	stqd		 $90, 160($SP)
	stqd		 $91, 176($SP)
	stqd		 $92, 192($SP)
	stqd		 $93, 208($SP)
	stqd		 $94, 224($SP)
	stqd		 $95, 240($SP)
	stqd		 $96, 256($SP)
	stqd		 $97, 272($SP)
	stqd		 $98, 288($SP)
	stqd		 $99, 304($SP)
	stqd		$100, 320($SP)
	stqd		$101, 336($SP)
	stqd		$102, 352($SP)
	stqd		$103, 368($SP)
	stqd		$104, 384($SP)
	stqd		$105, 400($SP)
	stqd		$106, 416($SP)
	stqd		$107, 432($SP)
	stqd		$108, 448($SP)
	stqd		$109, 464($SP)
	stqd		$110, 480($SP)
	stqd		$111, 496($SP)
	stqd		$112, 512($SP)
	stqd		$113, 528($SP)
	stqd		$114, 544($SP)
	stqd		$115, 560($SP)
	stqd		$116, 576($SP)
	stqd		$117, 592($SP)
	stqd		$118, 608($SP)
	stqd		$119, 624($SP)
	stqd		$120, 640($SP)
	stqd		$121, 656($SP)
	stqd		$122, 672($SP)
	stqd		$123, 688($SP)
	stqd		$124, 704($SP)
	stqd		$125, 720($SP)
	stqd		$126, 736($SP)
	stqd		$127, 752($SP)

	# Load OGR table
	ila		$OGR, OGR_table

	# Constant for splats that will be needed in the following loads
	lqa		$_splatRSM, _splatRSM_data

.macro	RdStructMember	reg, base_reg, offset
	# Read value from memory
	lqd		\reg, \offset(\base_reg)

	# Compute effective address offset($3)
	ai		$temp_RSM, \base_reg, \offset

	# Move the desired member of the struct to the preferred slot
	rotqby		\reg, \reg, $temp_RSM

	# Splat the desired member to remaining words of the register
	shufb		\reg, \reg, \reg, $_splatRSM
.endm

	# depth = state->depth + 1
	RdStructMember	$depth, $state, STATE_DEPTH
	ai		$depth, $depth, 1

	RdStructMember	$maxlength, $state, STATE_MAX

	# remdepth = state->maxdepthm1
	RdStructMember	$remdepth, $state, STATE_MAXDEPTHM1

	# halfdepth = state->half_depth
	RdStructMember	$halfdepth, $state, STATE_HALFDEPTH

	# halfdepth2 = state->half_depth2
	RdStructMember	$halfdepth2, $state, STATE_HALFDEPTH2

	RdStructMember	$half_length, $state, STATE_HALFLENGTH

	RdStructMember	$pnodes_data, $pnodes, 0

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


	ilhu		$_splatchoose, 0x8080
	iohl		$_splatchoose, 0x8000

	ilhu		$ttmDISTBITS_mask, 0xFFF0
	iohl		$ttmDISTBITS_mask, 0x0000

	lqa		$shr32, shr32_data
	lqa		$shl32v, shl32v_data
	lqa		$shl32s, shl32s_data
	lqa		$pslotmask, pslotmask_data

	ila		$stay_addr, stay
	ila		$end_stay_addr, end_stay
	ila		$up_addr, up
	ila		$cont_stay_addr, cont_stay


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
SETUP_TOP_STATE:
	lqd		$comp0, LEVEL_QUAD_COMP0($lev)
	lqd		$compV, LEVEL_QUAD_COMPV($lev)

	lqd		$list0, LEVEL_QUAD_LIST0($lev)
	lqd		$listV, LEVEL_QUAD_LISTV($lev)

	lqd		$dist0, LEVEL_QUAD_DIST0($lev)
	lqd		$distV, LEVEL_QUAD_DISTV($lev)

	il		$newbit, 1
	and		$newbit, $newbit, $pslotmask

	and		$temp1,  $dist0, $ttmDISTBITS_mask
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
	nand		$temp5, $comp0, $comp0				# 10e
	rotmi		$temp5, $temp5, -1				# 12e

	# if (depth <= halfdepth2)
	ceqi		$temp6, $comp0, -1				# 14e

	clz		    $s, $temp5					# 22e

	br		for_loop


	.align		3
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
	ai		$depth, $depth, -1				 #  1e
	lqd		$comp0, LEVEL_QUAD_COMP0-SIZEOF_LEVEL_QUAD($lev) #  1o

	# remdepth++;
	ai		$remdepth, $remdepth, 1				 #  2e
	lqd		$list0, LEVEL_QUAD_LIST0-SIZEOF_LEVEL_QUAD($lev) #  2o

	xor		$newbit, $newbit, $newbit			 #  3e
	lqd		$listV, LEVEL_QUAD_LISTV-SIZEOF_LEVEL_QUAD($lev) #  3o

	# if (depth <= 0)
	cgti		$temp2, $depth, 0				 #  4e
	lqd		$compV, LEVEL_QUAD_COMPV-SIZEOF_LEVEL_QUAD($lev) #  4o

	# limit = lev->limit;
	lqd		$limit, LEVEL_QUAD_LIMIT-SIZEOF_LEVEL_QUAD($lev) #  5o

	# mark = lev->mark;
	lqd		 $mark, LEVEL_QUAD_MARK-SIZEOF_LEVEL_QUAD($lev)  #  6o

	# s = LOOKUP_FIRSTBLANK(comp0);
	nand		$temp1, $comp0, $comp0				 #  7e

	# Assuming values outside the preferred slot are 0, then after
	# applying andc they'll remain 0
	andc		$dist0, $dist0, $list0				 #  8e

	rotmi		$temp1, $temp1, -1				 #  9e

	andc		$distV, $distV, $listV				 # 10e

	ceqi		$temp6, $comp0, -1				 # 11e
	lnop

	# lev--;
	ai		  $lev,   $lev, -SIZEOF_LEVEL_QUAD		 # 12e
	brz		$temp2, depth_le_0				 # 12o

	# goto stay; -- just fall through

	clz		    $s, $temp1					 # 13e
	rotqbii		$temp8, $list0,      0				 # 13o

	selb		$temp7, $end_stay_addr, $cont_stay_addr, $temp6	 # 14e


	.align		3
stay:
	a		 $mark,  $mark,     $s
	hbr		end_stay_branch, $temp7

	sfi		$temp1,     $s,     32
	lnop

	# if ((mark += s) > limit) goto up;
	cgt		$temp7,  $mark, $limit
	sfi		$temp2,     $s,      0

	# Taken with probability 32.1%
	sfi		$temp3, $temp1,      0
	brnz		$temp7, up


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

	.align		3

	shl		$comp0, $comp0,     $s				#  1e
	rotqmbi		$listV, $listV, $temp2				#  1o

	rotm		$temp4, $compV, $temp3				#  2e
	shlqbi		$compV, $compV,     $s				#  2o

	sfi		$temp5,     $s,      7				#  3e
	shl		$newbit, $newbit, $temp1			#  4e

	rotm		$list0, $list0, $temp2				#  5e
	rotqmbybi	$listV, $listV, $temp5				#  5o

	or		$comp0, $comp0, $temp4				#  6e
	shlqbybi	$compV, $compV,     $s				#  6o

	shl		$temp4, $temp8, $temp1				#  7e
	hbra		stay_branch, stay				#  7o

	lr		$temp5, $newbit					#  8e
	lnop								#  8o

	and		$comp0, $comp0, $pslotmask			#  9e
	rotqmbyi	$newbit, $newbit, -16				#  9o

	or		$list0, $list0, $temp5				# 10e
	lnop								# 11o

	or		$listV, $listV, $temp4				# 12e

end_stay_branch:
	brz		$temp6, end_stay				# 12o

cont_stay:
	# s = LOOKUP_FIRSTBLANK(comp0);
	nand		$temp1, $comp0, $comp0				# 13e
	ceqi		$temp6, $comp0, -1				# 14e

	rotmi		$temp1, $temp1, -1				# 15e
	selb		$temp7, $end_stay_addr, $cont_stay_addr, $temp6	# 16e

	lr		$temp8, $list0					# 17e
	nop								# 18e

	clz		    $s, $temp1					# 19e
stay_branch:
	br		stay						# 19o


	.align		3
end_stay:
	nop								#  1e
	# if (remdepth == 0)
	brz		$remdepth, remdepth_eq_0			#  1o

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
	or		$dist0, $dist0, $list0				#  2e
	stqd		$comp0, LEVEL_QUAD_COMP0($lev)			#  2o

	or		$distV, $distV, $listV				#  3e
	stqd		$compV, LEVEL_QUAD_COMPV($lev)			#  3o

	# limit = maxlength - choose(dist0 >> ttmDISTBITS, remdepth);
	# define choose(x,y) (ogr_choose_dat[((x)<<3)+((x)<<2)+(y)+3])

	# Instead of computing 12*x, where x = dist0 >> ttmDISTBITS, we
	# compute 3*x', where x' = (dist0 & 0xFFF00000) >> (ttmDISTBITS-2).
	and		$temp1,  $dist0, $ttmDISTBITS_mask		#  4e
	hbra		branch_end_for_loop, stay			#  4o

	# remdepth--;
	ai		$remdepth, $remdepth, -1			#  5e
	rotmi		$temp1,  $temp1, -(ttmDISTBITS-2)		#  5o

	ai		$temp3, $remdepth, 3				#  6e
	lnop								#  6o

	or		$comp0, $comp0, $dist0				#  7e
	stqd		$list0, LEVEL_QUAD_LIST0($lev)			#  7o

	or		$compV, $compV, $distV				#  8e
	stqd		$listV, LEVEL_QUAD_LISTV($lev)			#  8o

	a		$temp3,  $temp3, $temp1				#  9e
	# s = LOOKUP_FIRSTBLANK(comp0);
	nand		$temp5, $comp0, $comp0				# 10e

	a		$temp1,  $temp1, $temp1				# 11e
	rotmi		$temp5, $temp5, -1				# 12e

	a		$temp1,  $temp3, $temp1				# 13e
	# if (depth <= halfdepth2)
	ceqi		$temp6, $comp0, -1				# 14e

	a		$temp2, $choose, $temp1				# 15e
	lqx		$temp1, $choose, $temp1				# 15o


	il		$newbit, 1					# 16e
	# lev->mark = mark;
	stqd		$mark, LEVEL_QUAD_MARK($lev)			# 16o

	andi		$temp4,  $temp2,    0xF				# 17e
	# lev++;
	ai		$lev, $lev, SIZEOF_LEVEL_QUAD			# 18e

	a		$temp4,  $temp4, $_splatchoose			# 19e
	# depth++;
	ai		$depth, $depth, 1				# 20e

	and		$newbit, $newbit, $pslotmask			# 21e
	shufb		$temp1,  $temp1, $temp1, $temp4			# 21o

	clz		    $s, $temp5					# 22e
	lnop								# 22o

	# continue; -- just fall through

	.align		3
for_loop:
	cgt		$temp2, $depth, $halfdepth2			# 23e

	selb		$temp7, $end_stay_addr, $cont_stay_addr, $temp6	# 24e
	rotqbii		$temp8, $list0,      0				# 24o

	sf		$limit,  $temp1, $maxlength			# 25e
	# Taken with probability 0.15%
	brz		$temp2, depth_le_halfdepth2			# 25o

cont_depth_le_halfdepth2:
	# nodes++;
	ai		$nodes, $nodes, 1				# 25e
	# lev->limit = limit;
	stqd		$limit, LEVEL_QUAD_LIMIT($lev)			# 25o

branch_end_for_loop:
	br		stay						# 9o


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
	rotqbii		$temp8, $list0,      0

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
	lqd		 $80,   0($SP)
	lqd		 $81,  16($SP)
	lqd		 $82,  32($SP)
	lqd		 $83,  48($SP)
	lqd		 $84,  64($SP)
	lqd		 $85,  80($SP)
	lqd		 $86,  96($SP)
	lqd		 $87, 112($SP)
	lqd		 $88, 128($SP)
	lqd		 $89, 144($SP)
	lqd		 $90, 160($SP)
	lqd		 $91, 176($SP)
	lqd		 $92, 192($SP)
	lqd		 $93, 208($SP)
	lqd		 $94, 224($SP)
	lqd		 $95, 240($SP)
	lqd		 $96, 256($SP)
	lqd		 $97, 272($SP)
	lqd		 $98, 288($SP)
	lqd		 $99, 304($SP)
	lqd		$100, 320($SP)
	lqd		$101, 336($SP)
	lqd		$102, 352($SP)
	lqd		$103, 368($SP)
	lqd		$104, 384($SP)
	lqd		$105, 400($SP)
	lqd		$106, 416($SP)
	lqd		$107, 432($SP)
	lqd		$108, 448($SP)
	lqd		$109, 464($SP)
	lqd		$110, 480($SP)
	lqd		$111, 496($SP)
	lqd		$112, 512($SP)
	lqd		$113, 528($SP)
	lqd		$114, 544($SP)
	lqd		$115, 560($SP)
	lqd		$116, 576($SP)
	lqd		$117, 592($SP)
	lqd		$118, 608($SP)
	lqd		$119, 624($SP)
	lqd		$120, 640($SP)
	lqd		$121, 656($SP)
	lqd		$122, 672($SP)
	lqd		$123, 688($SP)
	lqd		$124, 704($SP)
	lqd		$125, 720($SP)
	lqd		$126, 736($SP)
	lqd		$127, 752($SP)

	# Restore stack pointer
	il		  $2, 16*(127-80+1)
	a		 $SP,  $SP,   $2

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

	# if (diff <= BITMAPS_LENGTH)
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

	# diff = (diff >> 5) - (BITMAPS_LENGTH / 32);
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
