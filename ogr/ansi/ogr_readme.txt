$Id: ogr_readme.txt,v 1.2 2009/01/02 16:59:42 kakace Exp $


                        OGR - Optimum Golomb Rulers


1. History

   Many algorithms have been devised in the last decades to find Golomb rulers.
   A clever method, named Shift Algorithm, was invented by David Mc Cracken for
   his thesis at Duke University. His work derived from the Scientific American
   Algorithm published in the december 1985 issue of Scientific American.

   In 1996, Mark Garry and David Vanderschel brought improvements to the Shift
   Algorithm. Their method was named GVANT for Garry/Vanderschel ANTenna
   placement program. A major breakthrough by Mark Garry and Roland Adorni led
   to the GARSP algorithm (Garry's Adaptation of Rado's Searching Principles).

   The latest improvements were made by Michael Feiri and Didier Levet to reduce
   the search space as much as possible. That's when GARSP became FLEGE
   (Feiri/Levet Enhanced Garsp Engine).


2. Overview

   Golomb rulers are defined by a set of distinct positive integers that
   represent the position of the marks on a ruler, so that no two pairs of
   marks measure the same distance. If M is the number of marks on a Golomb
   ruler, then that ruler measures exactly M(M-1)/2 distances, each of which
   being unique. Said otherwise, if a Golomb ruler can measure the distance D,
   then there is one and only one pair of marks that measures D.

   The length of a ruler is the distance between its first and last marks. By
   convention, the first mark is placed at position 0. Then, the length of a
   ruler is also the position of the last mark on that ruler.

   A Golomb ruler is said optimum if it is the shortest ruler that can be built
   with the specified number of marks. The only way to prove optimality is to
   do an exhaustive search to determine whether any known ruler is the shortest.
   By exhaustive search, I mean we have to try every possible sets of marks to
   determine whether the resulting ruler is Golomb and, if it is, whether it is
   shorter than the shortest known ruler.


3. GARSP/FLEGE algorithms

   The GARSP algorithm is both quite simple and very clever. Most of the work is
   done by performing basic operations on three bitmaps (multiprecision shifts
   and bitwise OR).

   These bitmaps shall be large enough to automatically assert the 'Golombness'
   of the resulting ruler. Ideally, the minimum width is half the length of the
   final ruler. The bitmaps can be shorter, but then the Golomb characteristic
   must be asserted by a dedicated function. The benefits of this tradeoff in
   term of processing time is not obvious, as it depends on how often the Golomb
   check has to be done.

   By convention, the leftmost bit of a bitmap is bit #1, and the rightmost bit
   of a bitmap is bit #N for a N-bit bitmap.


   3.1 The LIST bitmap

   This bitmap can be seen as a graphical representation of a part of the ruler
   being constructed. This representation involves an implied bit (bit #0),
   which is not part of the bitmap. This extra bit represents the last mark
   placed on the ruler. With this in mind, all bits set in the LIST bitmap
   represent the marks already placed on the left of this last mark. That is,
   if bit #10 is set, then there is a mark 10 positions away on the left of
   this last mark.

   3.2 The DIST bitmap

   This bitmap collects all the distances measured by the ruler as we place new
   marks. Bit #1 represents distance 1, bit #2 represents distance 2, and so on.

   3.3 The COMP bitmap

   This bitmap indicates where the next mark can be placed (or where the current
   mark can be moved to) relatively to the current position. All bits set show
   positions that are not allowed. Therefore, if bit #13 is the leftmost bit
   cleared, then a new mark can be placed (or the current mark can be moved) 13
   positions away on the right of the current position.

   3.4 Clever !

   Proper operations on the LIST, DIST and COMP bitmaps ensure that the
   resulting ruler will be Golomb. It turns out these operations are
   surprisingly very simple. The bulk of the GARSP algorithm looks as follow :

   ----8<-----------------------------------------------------------------------
   int distance = lookup_firstblank(COMP);
   if (mark + distance > limit)
      break;                        // No room.

   COMP <<= distance;
   LIST = newbit::LIST >> distance;

   DIST |= LIST;
   COMP |= DIST;
   limit = compute_limit();
   ----8<-----------------------------------------------------------------------

   By nature, a straightforward implementation would be recursive. For dnetc,
   the algorithm has been modified to be linear (hence the many book-keeping
   operations).

   "newbit" acts as a flag to indicate whether we shall move the current mark
   (newbit == 0), or append a new mark (newbit == 1). There are two equivalent
   ways to introduce the "newbit" information in the LIST bitmap :
   - By shifting newbit in the LIST bitmap (multi-precision shift), as if
     "newbit" was bit #0.
   - By shifting the LIST bitmap first, then setting bit #(distance) in the LIST
     bitmap if "newbit" is set.
   The "newbit" flag shall be set/unset at the right places, otherwise the
   content of the bitmaps quickly becomes erratic. Ideally, this flag should
   be cleared after it has been merged in the LIST bitmap, and it shall be set
   when moving to the next mark.

   The lookup_firstblank() function returns the index of the leftmost bit
   cleared in its argument.

   The compute_limit() function returns the maximum position the next mark can
   occupy. This limit is obtained by various means.


4. OGR-NG implementation

   OGR-NG is based on 256-bit bitmaps versus 160 bits for OGR/OGR-P2. This
   change is aimed toward the search of rulers made of 26 marks or more. As a
   result, existing cores are not compatible with OGR-NG.

   The workhorse lies in ogr_cycle_256(). This function requires several macros
   that provide critical parts of the algorithm.

   4.1 SETUP_TOP_STATE(level_pointer)

   This macro shall declare at least three local variables of the proper type :
   - SCALAR comp0 = lev->comp[0];
   - SCALAR dist0;
   - <type> newbit = (depth < oState->maxdepthm1) ? 1 : 0;

   It may declare and initialize core-specific variables as well. "dist0" is not
   used until later in the code, so its initialization can be postponed. The
   type of "newbit" depends on the implementation of the core itself (and more
   precisely, on how it is merged in the LIST bitmap).

   4.2 COMP_LEFT_LIST_RIGHT(level_pointer, distance)

   Shift the COMP bitmap to the left by 'distance', and the LIST bitmap to the
   right by 'distance'. The value of "newbit" is merged appropriately in the
   LIST bitmap.

   4.3 COMP_LEFT_LIST_RIGHT_WORD(level_pointer)

   Similar to COMP_LEFT_LIST_RIGHT(), except that the bitmaps are shifted by
   SCALAR_BITS (usually, one bitmap word). "newbit" MUST be cleared after being
   merged into the LIST bitmap, because the code may jump back to a call to
   COMP_LEFT_LIST_RIGHT(). For the very same reason, "comp0" MUST be loaded
   with the new leftmost word of the COMP bitmap.

   4.4 PUSH_LEVEL_UPDATE_STATE(level_pointer)

   Update the DIST and COMP bitmaps, then store the content of the three bitmaps
   to initialize the next level. Basically, the update is done as follow :
   DIST |= LIST
   COMP |= DIST
   In addition, "newbit" shall be set to 1 because the next level will obviously
   place a new mark. "dist0" shall be loaded with the leftmost word of the
   updated DIST bitmap.

   4.5 POP_LEVEL(level_pointer)

   Reload the value of "comp0" and "dist0" for the specified level. "newbit"
   shall be cleared because we'll try to move an existing mark. Other than that,
   the state of the preceeding level is restored.
   For cores that keep the bitmaps in registers, the DIST bitmap can be
   recovered with :
   DIST &= ~LIST;
   
   4.6 SAVE_FINAL_STATE(level_pointer)

   This macro allows core that keep datas in registers to write these datas
   back into the Level structure.

   4.7 LOOKUP_FIRSTBLANK(bitmap)

   Returns the index of the leftmost bit cleared in the specified bitmap. When
   used to lookup "comp0", implementors can choose to return either SCALAR_BITS
   or SCALAR_BITS+1 when all bits are set in "comp0". Both will work the same
   due to how the bulk code select one of the COMP_LEFT_LIST_RIGHT macros.

   4.8 choose(dist_bitmap, depth)

   One major improvement in FLEGE is that the limits are more effective than
   in OGR/OGR-P2. This is due to a significantly larger "choose" array. In
   addition, most limits are now cached in a large lookup table. In most cases,
   no computations are necessary. The remaining cases apply to the middle
   segment which is made of 2 or 3 marks.
   The lookup table is organized as in : u16 choose[1 << CHOOSE_DIST_BITS][32];
   Note that the value of "dist0" shall be scaled appropriately to peek a data
   in the choose array. That is, the leftmost CHOOSE_DIST_BITS are significant
   and shall be shifted to the right.

   4.9 Datatypes and constants.

   The generic core requires a few datatypes and related constants to work
   properly :
   - SCALAR : An unsigned integer type. The core shall behave as if this type
     was the underlying datatype used to represent the bitmaps. It is also the
     type of the "comp0" and "dist0" variables.
   - SCALAR_BITS : The width (number of bits) of the SCALAR type.
   - BMAP : Some datatype used to represent the bitmaps. It may or may not be
     identical to SCALAR. For instance, one may define SCALAR to be equivalent
     to "u32", and assign BMAP to some vector datatype.
   - OGRNG_BITMAPS_WORDS : How many BMAP datas are required to represent a
     single bitmap.


5. Adding new cores

   Compared to the legacy OGR implementation, the core has been simplified and
   the number of settings has been reduced.

   5.1 Settings

   OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM : 0 / 1 / 2
   If 0, no assembly support is provided. If set to 1, partial assembly support
   does exist but the lookup table is still required. If set to 2, the assembly
   code is standalone and the lookup table is not required.
   Assembly code shall be provided by either the __CNTLZ() macro (full support)
   or __CNTLZ_ARRAY_BASED() macro (partial support).
   This setting and the associated macro are used in first_blank.h to implement
   the LOOKUP_FIRSTBLANK() macro.
   CAUTION : Assembly code shall be prepared to properly deal with the SCALAR
             datatype. Implementations usually take each arguments in a single
             register : that doesn't work with compound types such as ui64 on
             32-bit CPUs.

   OGROPT_ALTERNATE_CYCLE : 0 / 1
   If set to 1, then the ogr_cycle_256() is replaced by a core-specific
   implementation. Note that macros described in chapter 4 still have to be
   defined for use in ogr_create().

   OGROPT_SPECIFIC_LEVEL_STRUCT
   If defined, then the core provides a specific implementation of the OgrLevel
   structure.

   OGROPT_SPECIFIC_STATE_STRUCT
   If defined, then the core provides a specific implementation of the OgrState
   structure.

   OGR_NG_GET_DISPATCH_TABLE_FXN
   Specifies the name of the dispatch table.

   5.2 Strategy

   From a client standpoint, each code shall have its own dispatch table.
   Therefore, it is not possible to create multiple cores from the same set of
   code/header files. To add a new core, implementors shall create a new .cpp
   file to host the settings, macros and other definitions.

   Let call that file newcore.cpp for clarity. This file may include ogrng-32.h
   or ogrng-64.h to import the SCALAR and BMAP datatypes as well as the bitmaps
   manipulation macros. Another approach would be to include ogrng-32.cpp or
   ogrng-64.cpp to import existing settings and specialized macros, but that
   route is more tedious because some things would then have to be undone (such
   as the need to redefine the dispatch table name). Implementors may also
   choose to duplicate an existing core to build upon.

   The next step is to define the settings (OGROPT_*), and the name of the
   dispatch table. If new/modified bitmaps manipulation macros are required,
   redefinitions of the relevant macros will follow (assuming they have been
   imported by including ogrng-xx.h). If offsets in the OgrLevel or OgrState
   structures need to be asserted, then include ogrng_corestate.h to import
   them.

   Then, ogrng_codebase.cpp shall be included verbatim. That file is NOT meant
   to be compiled alone. Finally, a core-specific implementation of
   ogr_create_256() may follow (if required).

   newcore.cpp shall then be added to the makefile, and the new core shall be
   accounted for in common/core_ogr_ng.cpp.


6. Technical considerations

   6.1 How to remove branches ?

   Some branches can be removed if the core is split in two parts.
   The first part shall run until depth > oState->half_depth2. In this part,
   the test "depth == oState->maxdepthm1" becomes unecessary.

   The core shall then switch to the second part, in which we always have
   depth > oState->half_depth2. Then, the calculation of the limit collapsed
   into : lev->limit = choose(dist0, depth);
   
   Both parts shall exit when the node count decrements to 0. The second part
   shall also switch back to the first part as soon as we have depth <=
   oState->half_depth2. In addition, execution shall jump to the right part
   upon entry.

   This kind of implementation is not straightforward, but it could well be
   worth the efforts because the core spends much of the processing time
   passed the middle segment.

   6.2 Lookup tables (pre-computed limits).

   These tables are a key part of the search space reduction. Although they
   introduce latencies due to memory accesses, the trade-off is worth it
   because the computations to obtain the position limits are greatly reduced.
   In addition, the OGR-NG cores have to check much less nodes than GARSP cores
   when processing the very same stubs (10 times less on average !).
   
   The side effect is that the client now requires much more memory than before.
   Selftests may run out of memory on old platforms because of that.

   6.3 Why is "LOOKUP_FIRSTBLANK(0xffffffff) == 32" safe ?

   When used to determine the shift count, the code actually checks the argument
   and switch to an alternate method that shift the bitmaps by 32 whenever
   necessary.
   When used to compute the limit of the middle mark in odd rulers, the reason
   is far less obvious. If we have "dist0 == 0xffffffff", then the leftmost part
   of the ruler being constructed measures all the distances from 1 to 32
   (inclusive). The remaining part of this ruler (yet unknown) cannot measure
   these distances so as to not break the Golomb criteria. The "choose" array
   gives us the minimum length of this remaining part, which leads to a maximum
   position for the middle mark. Another bound is given by the position of the
   last mark placed so far because we want to filter out mirrored image (middle
   segment reduction method). This calculation involves LOOKUP_FIRSTBLANK.
   Finally, the code picks the lowest bound.

   Let G be the length of the ruler we're searching, C the value obtained from
   the "choose" array, and P the position of the last mark placed when we need
   to compute the limit of the middle mark. We need to ensure that :
                           G - C  <=  G - 1 - P - 32
   otherwise implementations of LOOKUP_FIRSTBLANK that return 33 instead of 32
   will get a better limit. This equation simplifies into "P <= C - 33", so
   the error first occurs when P == C-32. But because the remaining part of the
   ruler is of length C (at least), and because the middle mark must be placed
   at a position x >= P+33, the entire ruler would be of length :
                          G' = (C-32) + 33 + C = 2*C + 1
   It turns out that for any odd ruler of up to 29 marks, G' > G. Therefore,
   implementations of LOOKUP_FIRSTBLANK can safely return 32 or 33 when the
   argument is 0xFFFFFFFF because this bit pattern cannot occur. In addition,
   no special care is required for 64-bit implementations because we always have
   LOOKUP_FIRSTBLANK(dist0) < 33.
