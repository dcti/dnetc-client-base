;
; $Log: keycheck_opt.s,v $
; Revision 1.3  1998/07/28 11:45:00  blast
; Amiga specific changes
;
; Revision 1.2  1998/06/14 10:30:40  friedbait
; 'Log' keyword added.
;
;
; The code is a bit smaller now. This should causes less cache misses for
; 68020/68030 based machines. But 68020 doesn't have a data cache. So I
; don't know if this code will be faster or not with a 68020.
; It should be faster on 68030 based machines (but I can't verify this
; myself), as well as on 68040 based machines.
;
; It IS FASTER when running on a 68060 : I get 116000 keys/s now using
; this code (it passes the 32 tests. Should be okay to use).
;
; I think that further optimizations can't be done here now. But the main
; loop (which calls this part) may be optimized. 
; Inlining the code in the loop no longer needs jsr/rts. This will give
; a noticeable speedup (approx 12 cycles less per key, 68060). But you'll
; have to rewrite the main loop in asm to do so...


;       MACHINE MC68020
        ; at least :)

        XDEF @rc5_unit_func

    SECTION rc564,CODE

@rc5_unit_func
	  movem.l d2-d5,-(sp)

		; I have reordered some instructions to optimize pipeline uses.

            move.l  $0014(a3),d0	; Looking at the C main loop shows that a0 is loaded with
            move.l  #$15235639,d2	; a3. Using a3 here avoid us to save a spare register on the
            lea     _datas(pc),a1	; stack (2 cycles less) and doesn't require an additional
            addi.l  #$BF0A8B1D,d0	; move (1 cycle less). This is a 'tiny' optimization, and this
            move.l  $0010(a3),d1	; looks not so good... But who will care about that ? :))
            ror.l   #3,d0
            lea     4-$0068(sp),a0	; There is no need for "subq.l #$0068,sp" :)))

; First stage. Can't be optimized further (instruction order point of view)
		
            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+		; Post Incremented mode should be faster on 68020/68030 than
            add.l   d3,d1		; (d8,An). This makes the code a little smaller too.
            add.l   (a1)+,d2		; This appears to be a bit faster than "addi.l #$xxx,d2". This
            rol.l   d3,d1		; makes the code a bit smaller too (68030)

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a1)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a1)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a1)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a1)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a1)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a1)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a1)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a1)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a1)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a1)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a1)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a1)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a1)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a1)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a1)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a1)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a1)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a1)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a1)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a1)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a1)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a1)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a1)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)
            add.l   d3,d1
            add.l   (a1),d2
            rol.l   d3,d1

;2nd stage. Can't be optimized further (instruction order point of view).

            add.l   d1,d2

	lea -$0068(sp),a0		; Will be dispatched with the previous instruction.
					; So this occurs no time penalty for the 68060.

            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a0),d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a0),d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a0),d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a0),d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a0),d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a0),d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a0),d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a0),d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a0),d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a0),d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a0),d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a0),d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a0),d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a0),d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a0),d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a0),d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a0),d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a0),d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a0),d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a0),d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a0),d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a0),d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a0),d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   (a0),d2
            rol.l   d3,d1

            add.l   d1,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d1,d3
            move.l  d2,(a0)+
            add.l   d3,d0
            add.l   (a0),d2
            rol.l   d3,d0

            add.l   d0,d2
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            move.l  d2,(a0)+
            add.l   d3,d1
            add.l   -$0068(sp),d2
            rol.l   d3,d1

; 3rd stage. 1 cycle less (68060) by changing instructions order)

            add.l   d1,d2		; 1
            move.l  4(a3),d4		; 0
            rol.l   #3,d2		; 1

	lea 4-$0068(sp),a0		; 0  (No time penalty)

            move.l  d2,d3		; 1
            add.l   d1,d3		; 0
            add.l   d2,d4		; 1
            add.l   d3,d0		; 0
            add.l   (a0)+,d2		; 1
            rol.l   d3,d0		; 0

            add.l   d0,d2
            move.l  (a3),d5
            rol.l   #3,d2
            move.l  d2,d3
            add.l   d0,d3
            add.l   d2,d5
            add.l   d3,d1
            add.l   (a0)+,d2
            rol.l   d3,d1

            add.l   d1,d2		; Each of these block needs 1 cycle less now :)
            eor.l   d5,d4		; Not so bad isn't it ? :)
            rol.l   #3,d2
            rol.l   d5,d4
            move.l  d2,d3
            add.l   d1,d3
            add.l   d2,d4
            add.l   d3,d0
            add.l   (a0)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            eor.l   d4,d5
            rol.l   #3,d2
            rol.l   d4,d5
            move.l  d2,d3
            add.l   d0,d3
            add.l   d2,d5
            add.l   d3,d1
            add.l   (a0)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            eor.l   d5,d4
            rol.l   #3,d2
            rol.l   d5,d4
            move.l  d2,d3
            add.l   d1,d3
            add.l   d2,d4
            add.l   d3,d0
            add.l   (a0)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            eor.l   d4,d5
            rol.l   #3,d2
            rol.l   d4,d5
            move.l  d2,d3
            add.l   d0,d3
            add.l   d2,d5
            add.l   d3,d1
            add.l   (a0)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            eor.l   d5,d4
            rol.l   #3,d2
            rol.l   d5,d4
            move.l  d2,d3
            add.l   d1,d3
            add.l   d2,d4
            add.l   d3,d0
            add.l   (a0)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            eor.l   d4,d5
            rol.l   #3,d2
            rol.l   d4,d5
            move.l  d2,d3
            add.l   d0,d3
            add.l   d2,d5
            add.l   d3,d1
            add.l   (a0)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            eor.l   d5,d4
            rol.l   #3,d2
            rol.l   d5,d4
            move.l  d2,d3
            add.l   d1,d3
            add.l   d2,d4
            add.l   d3,d0
            add.l   (a0)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            eor.l   d4,d5
            rol.l   #3,d2
            rol.l   d4,d5
            move.l  d2,d3
            add.l   d0,d3
            add.l   d2,d5
            add.l   d3,d1
            add.l   (a0)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            eor.l   d5,d4
            rol.l   #3,d2
            rol.l   d5,d4
            move.l  d2,d3
            add.l   d1,d3
            add.l   d2,d4
            add.l   d3,d0
            add.l   (a0)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            eor.l   d4,d5
            rol.l   #3,d2
            rol.l   d4,d5
            move.l  d2,d3
            add.l   d0,d3
            add.l   d2,d5
            add.l   d3,d1
            add.l   (a0)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            eor.l   d5,d4
            rol.l   #3,d2
            rol.l   d5,d4
            move.l  d2,d3
            add.l   d1,d3
            add.l   d2,d4
            add.l   d3,d0
            add.l   (a0)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            eor.l   d4,d5
            rol.l   #3,d2
            rol.l   d4,d5
            move.l  d2,d3
            add.l   d0,d3
            add.l   d2,d5
            add.l   d3,d1
            add.l   (a0)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            eor.l   d5,d4
            rol.l   #3,d2
            rol.l   d5,d4
            move.l  d2,d3
            add.l   d1,d3
            add.l   d2,d4
            add.l   d3,d0
            add.l   (a0)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            eor.l   d4,d5
            rol.l   #3,d2
            rol.l   d4,d5
            move.l  d2,d3
            add.l   d0,d3
            add.l   d2,d5
            add.l   d3,d1
            add.l   (a0)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            eor.l   d5,d4
            rol.l   #3,d2
            rol.l   d5,d4
            move.l  d2,d3
            add.l   d1,d3
            add.l   d2,d4
            add.l   d3,d0
            add.l   (a0)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            eor.l   d4,d5
            rol.l   #3,d2
            rol.l   d4,d5
            move.l  d2,d3
            add.l   d0,d3
            add.l   d2,d5
            add.l   d3,d1
            add.l   (a0)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            eor.l   d5,d4
            rol.l   #3,d2
            rol.l   d5,d4
            move.l  d2,d3
            add.l   d1,d3
            add.l   d2,d4
            add.l   d3,d0
            add.l   (a0)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            eor.l   d4,d5
            rol.l   #3,d2
            rol.l   d4,d5
            move.l  d2,d3
            add.l   d0,d3
            add.l   d2,d5
            add.l   d3,d1
            add.l   (a0)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            eor.l   d5,d4
            rol.l   #3,d2
            rol.l   d5,d4
            move.l  d2,d3
            add.l   d1,d3
            add.l   d2,d4
            add.l   d3,d0
            add.l   (a0)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            eor.l   d4,d5
            rol.l   #3,d2
            rol.l   d4,d5
            move.l  d2,d3
            add.l   d0,d3
            add.l   d2,d5
            add.l   d3,d1
            add.l   (a0)+,d2
            rol.l   d3,d1

            add.l   d1,d2
            eor.l   d5,d4
            rol.l   #3,d2
            rol.l   d5,d4
            move.l  d2,d3
            add.l   d1,d3
            add.l   d2,d4
            add.l   d3,d0
            add.l   (a0)+,d2
            rol.l   d3,d0

            add.l   d0,d2
            eor.l   d4,d5
            rol.l   #3,d2
            rol.l   d4,d5
            move.l  d2,d3
            add.l   d0,d3
            add.l   d2,d5
            add.l   d3,d1
            add.l   (a0)+,d2
            rol.l   d3,d1

            add.l   d1,d2	; 1
            eor.l   d5,d4	; 0
            rol.l   #3,d2	; 1
            rol.l   d5,d4	; 0
            add.l   d2,d4	; 1
            cmp.l   12(a3),d4	; 1
            bne.s   .lb0119B8

            move.l  d2,d3
            add.l   d1,d3
            add.l   d3,d0
            add.l   (a0)+,d2
            rol.l   d3,d0

            add.l   d0,d2       ; 1
            eor.l   d4,d5       ; 0
            rol.l   #3,d2       ; 1
            rol.l   d4,d5       ; 0
            move.l  d2,d3       ; 1
            add.l   d0,d3       ; 0
            add.l   d2,d5       ; 1
            add.l   d3,d1       ; 0
            moveq   #1,d0       ; 1		; Moved one line up.
            rol.l   d3,d1       ; 0
            cmp.l   8(a3),d5    ; 1
            beq.s   .lb0119BA

.lb0119B8:  moveq   #0,d0       ; 0		(this is local labels (Devpac syntax))
.lb0119BA:  movem.l (sp)+,d2-d5 ; 4
            rts

	CNOP 0,8		; Align to an even multiple of 8.

_datas:     dc.l    $F45044D5,$9287BE8E,$30BF3847,$CEF6B200
            dc.l    $6D2E2BB9,$0B65A572,$A99D1F2B,$47D498E4
            dc.l    $E60C129D,$84438C56,$227B060F,$C0B27FC8
            dc.l    $5EE9F981,$FD21733A,$9B58ECF3,$399066AC
            dc.l    $D7C7E065,$75FF5A1E,$1436D3D7,$B26E4D90
            dc.l    $50A5C749,$EEDD4102,$8D14BABB,$2B4C3474
            dc.l    $BF0A8B1D


