; Cyrix 6x86 optimized version (nasm version of .cpp core)
;
; $Id: rg-6x86.asm,v 1.1.2.1 2001/01/21 17:44:41 cyp Exp $
;
; 980226 :
;	- Corrected bug in the key incrementation algorithm that caused the
;	  client to core-dump at the end of some blocks.
;	  As a side-effect, this fix re-enable support for blocks of up to 2^64 keys
;	- Added alignement pseudo-instructions.
;	- Converted :
;		subl $0x01000000, %%reg   to   addl $0xFF000100, %%reg
;		addl $0x00000100, %%reg
;	  and :
;		subl $0x00010000, %%reg   to   addl $0xFFFF0001, %%reg
;		addl $0x00000001, %%reg
;
; 980118 :
;	- Sean McPherson <sean@ntr.net> has allowed me to use his Linux box to
;	  test & debug this new core. Thanks a lot, Sean !
;	  This one is 17% faster than the previous ! On Sean PC
;	  (6x86 PR200) it ran at ~354 kkeys/s where the previous core got
;	  only 304 kkeys/s
;
; 980104 :
;	- precalculate some things for ROUND1 & ROUND2
;
; 971226 :
;	* Edited ROUND1 & ROUND2 to avoid potential stall on such code :
;		roll	$3,    %%eax
;		movl	%%eax, %%ecx
;	* ROUND3 cycle counts was bad, it was really 14 cycles in 971220,
;	  not 13 cycles. Modified a bit and back to 13 real cycles.
;
; 971220 :
;	* It seems that this processor can't decode two instructions
;	  per clock when one of the pair is >= 7 bytes.
;	  Unfortunatly, there's a lot in this code, because "s2" is located
;	  more than 127 bytes from %esp. So each access to S2 will be coded
;	  as a long (4 bytes) displacement.
;	  S1 access will be coded with a short displacement (8 bits signed).
;	* Modified ROUND3 to access s1 & s2 with ebp as the base, and so with
;	  a short displacement.
;	* Modified also ROUND1 with ROUND2 as a template, since ROUND2 seems
;	  to suffer less from this limitation.
;
; 971214 :
;	* modified ROUND1 & ROUND2 to avoid :
;		leal	(%%eax,%%edx), %%ecx
;		movl	%%eax, "S1(N)"
;		addl	%%ecx, %%ebx	# doesn't seems to pair with 2nd clock of "leal"
;	  -> 1 clock less in each macro
;
;	* modified ROUND3 to avoid a stall on the Y pipe
;	  -> 1 clock less
;
; 971209 :
;	* First public release
;
;
; PRating versus clock speed:
;
;    6x86		 6x86MX (aka M2)
;
; PR200 = 150		PR266 = 225 | 233
; PR166 = 133		PR233 = 188 | 200
; PR150 = 120		PR200 = 166
; PR133 = 110		PR166 = 150

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
[SECTION _TEXT USE32 ALIGN=16]
%else
[SECTION .text]
%endif

[GLOBAL _rc5_unit_func_6x86]
[GLOBAL rc5_unit_func_6x86]

%define work_size       276

%define RC5UnitWork     esp+work_size+4
%define timeslice       esp+work_size+8


; The S0 values for key expansion round 1 are constants.

%define P         0xB7E15163
%define Q         0x9E3779B9
%define S_not(N)  (P+Q*(N))

;#define S0_ROTL3  _(((P<<3) | (P>>29)))
%define S0_ROTL3 0xbf0a8b1d
;#define FIRST_ROTL _((S0_ROTL3 & 0x1f))
%define FIRST_ROTL 0x1d
;#define S1_S0_ROTL3 _((S_not(1) + S0_ROTL3))
%define S1_S0_ROTL3 0x15235639


;  Offsets to access work_struct fields.

%define save_ebp   esp+0
%define save_edi   esp+4
%define save_esi   esp+8
%define save_ebx   esp+12
%define work_s1         esp+16
%define work_s2         esp+104+16
%define work_add_iter   esp+208+16
%define work_P_0        esp+212+16
%define work_P_1        esp+216+16
%define work_C_0        esp+220+16
%define work_C_1        esp+224+16
%define work_key2_edi   esp+228+16
%define work_key2_esi   esp+232+16
%define work_key_hi     esp+236+16
%define work_key_lo     esp+240+16
%define work_iterations esp+244+16
%define work_pre1_r1    esp+248+16
%define work_pre2_r1    esp+252+16
%define work_pre3_r1    esp+256+16

; Offsets to access RC5UnitWork fields

%define RC5UnitWork_plainhi   eax+0
%define RC5UnitWork_plainlo   eax+4
%define RC5UnitWork_cipherhi  eax+8
%define RC5UnitWork_cipherlo  eax+12
%define RC5UnitWork_L0hi      eax+16
%define RC5UnitWork_L0lo      eax+20

  ; A1   = %eax  A2   = %ebp
  ; Llo1 = %ebx  Llo2 = %esi
  ; Lhi1 = %edx  Lhi2 = %edi


;3 cycles
;	roll	%%cl,  %%esi		#
;	addl	$"S_not(N)", %%eax	#
;	addl	$"S_not(N)", %%ebp	#
;
;it's not the length of instruction, it's that it can't decode when instruction is executed
;3 cycles :
;	roll	%%cl,  %%edx	#
;	addl	0(%%ebp),%%eax	#
;	xorl	%%edi, %%esi	#
;
;
;leal is subject of AGI (2 cycles) if an operand is modified 1 cycle before
;leal is subject of AGI (1 cycles) if an operand is modified 2 cycles before
;
;leal (%%edx,%%eax), %%eax take one cycle and is pairable
;leal 12345678(%%edx,%%eax), %%eax takes two cycles and isn't pairable

%define S1(N) [work_s1+((N)*4)]
%define S2(N) [work_s2+((N)*4)]

; ------------------------------------------------------------------
; S1(N) = A1 = ROTL3 (A1 + Lhi1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Lhi2 + S_not(N));
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
%macro ROUND_1_EVEN 1
        add     eax, S_not(%1)          ; . pairs with roll in previous iteration
        add     ebp, edi                ; 1
        add     eax, edx                ;
        rol     ebp, 3                  ; 1
        rol     eax, 3                  ;
        mov     ecx, eax                ; 1
        add     ecx, edx                ;   yes, it works
        mov     S1(%1), eax             ; 1
        add     ebx, ecx                ;
        add     eax, S_not((%1)+1)      ; 1
        mov     S2(%1), ebp             ;
        rol     ebx, cl                 ; 2
        lea     ecx, [ebp+edi]          ;
        add     esi, ecx                ; 1
        add     ebp, S_not((%1)+1)      ;
        rol     esi, cl                 ; 2
%endmacro

; S1(N) = A1 = ROTL3 (A1 + Llo1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Llo2 + S_not(N));
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
%macro ROUND_1_ODD 1(N) \
        add     eax, ebx                ; . pairs with roll in previous iteration
        add     ebp, esi                ; 1
        rol     eax, 3                  ;
        rol     ebp, 3                  ; 1
        mov     S1(%1), eax             ;
        mov     ecx, eax                ; 1
        add     ecx, ebx                ;
        add     edx, ecx                ; 1
        mov     S2(%1), ebp             ;
        rol     edx, cl                 ; 2
        lea     ecx, [ebp+esi]          ;
        add     edi, ecx                ; 1
        add     ebp, S_not((%1)+1)      ;
        rol     edi, cl                 ; 2	sum = 19 (r1 & r2) \n"
%endmacro

%macro ROUND_1_LAST 1
        add     eax, ebx                ; . pairs with roll in previous iteration
        add     ebp, esi                ; 1
        rol     eax, 3                  ;
        rol     ebp, 3                  ; 1
        mov     S1(%1), eax             ;   yes, it works !
        mov     ecx, eax                ; 1
        add     ecx, ebx                ;
        add     edx, ecx                ; 1
        mov     S2(%1), ebp             ;
        rol     edx, cl                 ; 2
        lea     ecx, [ebp+esi]          ;
        add     edi, ecx                ; 1
        add     eax, S0_ROTL3           ;
        rol     edi, cl                 ; 2	sum = 19 (r1 & r2)
%endmacro

%macro ROUND_1_EVEN_AND_ODD 1
       ROUND_1_EVEN %1
       ROUND_1_ODD %1+1
%endmacro

; ------------------------------------------------------------------
; S1N = A1 = ROTL3 (A1 + Lhi1 + S1N);
; S2N = A2 = ROTL3 (A2 + Lhi2 + S2N);
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
%macro ROUND_2_EVEN 1
        add     eax, S1(%1)             ; . pairs with roll in previous iteration
        add     ebp, edi                ; 1
        add     eax, edx                ;
        rol     ebp, 3                  ; 1
        rol     eax, 3                  ;
        mov     ecx, eax                ; 1
        add     ecx, edx                ;   yes, it works
        mov     S1(%1), eax             ; 1
        add     ebx, ecx                ;
        add     eax, S1(%1+1)           ; 1
        mov     S2(%1), ebp             ;
        rol     ebx, cl                 ; 2
        lea     ecx, [ebp+edi]          ;
        add     esi, ecx                ; 1
        add     ebp, S2(%1+1)           ;
        rol     esi, cl                 ; 2
%endmacro

; S1N = A1 = ROTL3 (A1 + Llo1 + S1N);
; S2N = A2 = ROTL3 (A2 + Llo2 + S2N);
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
%macro ROUND_2_ODD 1
        add     eax, ebx                ; . pairs with roll in previous iteration
        add     ebp, esi                ; 1
        rol     eax, 3                  ;
        rol     ebp, 3                  ; 1
        mov     S1(%1), eax             ;   yes, it works !
        mov     ecx, eax                ; 1
        add     ecx, ebx                ;
        add     edx, ecx                ; 1
        mov     S2(%1), ebp             ;
        rol     edx, cl                 ; 2
        lea     ecx, [ebp+esi]          ;
        add     edi, ecx                ; 1
        add     ebp, S2(%1+1)           ;
        rol     edi, cl                 ; 2	sum = 19 (r1 & r2)
%endmacro

%macro ROUND_2_LAST 1
        add     eax, ebx                ; . pairs with roll in previous iteration
        add     ebp, esi                ; 1
        rol     eax, 3                  ;
        rol     ebp, 3                  ; 1
        mov     S1(%1), eax             ;   yes, it works !
        mov     ecx, eax                ; 1
        add     ecx, ebx                ;
        add     edx, ecx                ; 1
        mov     S2(%1), ebp             ;
        rol     edx, cl                 ; 2
        lea     ecx, [ebp+esi]          ;
        add     edi, ecx                ; 1
        mov     [work_key2_esi], esi    ;
        rol     edi, cl                 ; 2	sum = 19 (r1 & r2)
%endmacro

%macro ROUND_2_EVEN_AND_ODD 1
	ROUND_2_EVEN %1
	ROUND_2_ODD %1+1
%endmacro

; ------------------------------------------------------------------
; eA1 = ROTL (eA1 ^ eB1, eB1) + (A1 = ROTL3 (A1 + Lhi1 + S1(N)));
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; eB1 = ROTL (eA1 ^ eB1, eA1) + (A1 = ROTL3 (A1 + Llo1 + S1(N)));
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);

; A  = %eax  eA = %esi
; L0 = %ebx  eB = %edi
; L1 = %edx  .. = %ebp
; %%ebp is either &S1 or &S2
%define S3(N) [(N)*4+ebp]

%macro ROUND_3_EVEN_AND_ODD 1
        add     eax, S3(%1)     ;
        xor     esi, edi        ;
        mov     ecx, edi        ; 1
        add     eax, edx        ;
        rol     esi, cl         ; 2
        rol     eax, 3          ;
        mov     ecx, eax        ;
        add     ecx, edx        ; 1
        add     esi, eax        ;
        add     ebx, ecx        ; 1
        add     eax, S3(%1+1)   ;
        rol     ebx, cl         ; 2

        xor     edi, esi        ;
        mov     ecx, esi        ;
        rol     edi, cl         ; 2
        add     eax, ebx        ;
        rol     eax, 3          ;
        mov     ecx, eax        ; 1
        add     ecx, ebx        ;
        add     edx, ecx        ; 1
        add     edi, eax        ;
        rol     edx, cl         ; 2
%endmacro

; ------------------------------------------------------------------
; rc5_unit will get passed an RC5WorkUnit to complete
;
; Returns number of keys checked before a possible good key is found, or
; timeslice*PIPELINE_COUNT if no keys are 'good' keys.
; (ie:      if (result == timeslice*PIPELINE_COUNT) NOTHING_FOUND
;      else if (result < timeslice*PIPELINE_COUNT) SOMETHING_FOUND at result+1
;      else SOMETHING_WENT_WRONG... )

align 4
_rc5_unit_func_6x86:
rc5_unit_func_6x86:
;u32 rc5_unit_func_6x86( RC5UnitWork * rc5unitwork, u32 timeslice )

     sub esp, work_size ; set up stack

     mov [save_ebp], ebp ; save registers
     mov [save_edi], edi
     mov [save_esi], esi
     mov [save_ebx], ebx

     mov ebp, [timeslice]

     mov dword [work_add_iter], 0x00000000
;    work.add_iter = 0;

     mov [work_iterations], ebp

     mov eax, [RC5UnitWork] ; load pointer to rc5unitwork into eax
;    work.iterations = timeslice;

	; load parameters
        mov     ebx, [RC5UnitWork_L0lo]                 ; ebx = l0 = Llo1
        mov     edx, [RC5UnitWork_L0hi]                 ; edx = l1 = Lhi1
        mov     esi, ebx                                ; esi = l2 = Llo2
        lea     edi, [0x01000000+edx]                   ; edi = l3 = lhi2
        mov     [work_key_lo], ebx
        mov     [work_key_hi], edx

	; Save other parameters
	; (it's faster to do so, since we will only load 1 value
	; each time in RC5_ROUND_3xy, instead of two if we save
	; only the pointer to the RC5 struct)
        mov     ebp, [RC5UnitWork_plainlo]
        mov     [work_P_0], ebp
        mov     ebp, [RC5UnitWork_plainhi]
        mov     [work_P_1], ebp
        mov     ebp, [RC5UnitWork_cipherlo]
        mov     [work_C_0], ebp
        mov     ebp, [RC5UnitWork_cipherhi]
        mov     [work_C_1], ebp

	; Pre-calculate things. Assume work.key_lo won't change it this loop
	; (it's pretty safe to assume that, because we're working on 28 bits
	; blocks)
	; It means also that %%ebx == %%esi (Llo1 == Llo2)

align 4
_bigger_loop_6x86:
        add     ebx, S0_ROTL3
        rol     ebx, FIRST_ROTL
        mov     [work_pre1_r1], ebx

        lea     eax, [S1_S0_ROTL3+ebx]
        rol     eax, 3
        mov     [work_pre2_r1], eax

        lea     ecx, [eax+ebx]
        mov     [work_pre3_r1], ecx

align 4
_loaded_6x86:
    ; ------------------------------
    ; Begin round 1 of key expansion
    ; ------------------------------

        mov     ebx, [work_pre1_r1]     ; 1
        mov     eax, [work_pre2_r1]     ;
        mov     esi, ebx                ; 1
        mov     ebp, eax                ;

        mov     ecx, [work_pre3_r1]     ; 1
        add     edx, ecx                ;
        rol     edx, cl                 ; 2
        add     edi, ecx                ;
        mov     S1(1), eax              ; 1
        add     ebp, S_not(2)           ;
        rol     edi, cl                 ; 2     sum = 8

	ROUND_1_EVEN_AND_ODD  2
	ROUND_1_EVEN_AND_ODD  4
	ROUND_1_EVEN_AND_ODD  6
	ROUND_1_EVEN_AND_ODD  8
	ROUND_1_EVEN_AND_ODD 10
	ROUND_1_EVEN_AND_ODD 12
	ROUND_1_EVEN_AND_ODD 14
	ROUND_1_EVEN_AND_ODD 16
	ROUND_1_EVEN_AND_ODD 18
	ROUND_1_EVEN_AND_ODD 20
	ROUND_1_EVEN_AND_ODD 22
	ROUND_1_EVEN         24
	ROUND_1_LAST         25
align 4
_end_round1_6x86:
    ; ------------------------------
    ; Begin round 2 of key expansion
    ; ------------------------------

        add     ebp, S0_ROTL3           ;
        add     eax, edx                ; 1
        add     ebp, edi                ;
        rol     eax, 3                  ; 1
        rol     ebp, 3                  ;
        mov     ecx, eax                ; 1
        add      ecx, edx               ;
        mov     S1(0), eax              ; 1
        add     ebx, ecx                ;
        add     eax, S1(1)              ; 1
        mov     S2(0), ebp              ;
        rol     ebx, cl                 ; 2
        lea     ecx, [ebp+edi]          ;
        add     esi, ecx                ; 1
        add     ebp, S1(1)              ;
        rol     esi, cl                 ; 2

	ROUND_2_ODD           1
        ROUND_2_EVEN_AND_ODD  2
        ROUND_2_EVEN_AND_ODD  4
        ROUND_2_EVEN_AND_ODD  6
        ROUND_2_EVEN_AND_ODD  8
        ROUND_2_EVEN_AND_ODD 10
        ROUND_2_EVEN_AND_ODD 12
        ROUND_2_EVEN_AND_ODD 14
        ROUND_2_EVEN_AND_ODD 16
        ROUND_2_EVEN_AND_ODD 18
        ROUND_2_EVEN_AND_ODD 20
        ROUND_2_EVEN_AND_ODD 22
	ROUND_2_EVEN         24
	ROUND_2_LAST         25

    ; Save 2nd key parameters
align 4
_end_round2_6x86:
        mov     [work_key2_edi], edi

    ; ----------------------------------------------------
    ; Begin round 3 of key expansion mixed with encryption
    ; ----------------------------------------------------
    ; (first key)					

	; A  = %eax  eA = %esi
	; L0 = %ebx  eB = %edi
	; L1 = %edx  .. = %ebp

        lea     ebp, S1(0)

	; A = ROTL3(S00 + A + L1);
	; eA = P_0 + A;
	; L0 = ROTL(L0 + A + L1, A + L1);
        add     eax, S3(0)      ;       (pairs with lea)
        add     eax, edx        ; 1
        mov     esi, [work_P_0] ;
        rol     eax, 3          ; 1
        mov     ecx, edx        ;
        add     esi, eax        ; 1
        add     ecx, eax        ;
        add     ebx, ecx        ; 1
        add     eax, S3(1)      ;
        rol     ebx, cl         ; 2
	; A = ROTL3(S01 + A + L0);
	; eB = P_1 + A;
	; L1 = ROTL(L1 + A + L0, A + L0);
        add     eax, ebx        ;
        mov     ecx, ebx        ; 1
        rol     eax, 3          ;
        mov     edi, [work_P_1] ; 1
        add     edi, eax        ;
        add     ecx, eax        ; 1
        add     edx, ecx        ; 1
        rol     edx, cl         ; 2
	ROUND_3_EVEN_AND_ODD  2
	ROUND_3_EVEN_AND_ODD  4
	ROUND_3_EVEN_AND_ODD  6
	ROUND_3_EVEN_AND_ODD  8
	ROUND_3_EVEN_AND_ODD 10
	ROUND_3_EVEN_AND_ODD 12
	ROUND_3_EVEN_AND_ODD 14
	ROUND_3_EVEN_AND_ODD 16
	ROUND_3_EVEN_AND_ODD 18
	ROUND_3_EVEN_AND_ODD 20
	ROUND_3_EVEN_AND_ODD 22

	; early exit
align 4
_end_round3_1_6x86:
        add     eax, S3(24)     ;       A = ROTL3(S24 + A + L1);
        mov     ecx, edi        ;       eA = ROTL(eA ^ eB, eB) + A;
        add     eax, edx        ;
        xor     esi, edi        ; 1
        rol     eax, 3          ;
        rol     esi, cl         ; 2
        add     esi, eax        ; 1
					
        cmp     esi, [work_C_0]
        jne     __exit_1_6x86
					
        mov     ecx, eax        ; 1     L0 = ROTL(L0 + A + L1, A + L1);
        add     ecx, edx        ;       A = ROTL3(S25 + A + L0);
        xor     edi, esi        ;       eB = ROTL(eB ^ eA, eA) + A;
        add     ebx, ecx        ;
        rol     ebx, cl         ; 2
        add     eax, S3(25)     ;
        mov     ecx, esi        ; 1
        add     eax, ebx        ;
        rol     edi, cl         ; 2
        rol     eax, 3          ;
        add     edi, eax        ; 1

        cmp     edi, [work_C_1]
        je near _full_exit_6x86

align 4
__exit_1_6x86:
    ; Restore 2nd key parameters
        mov     edx, [work_key2_edi]
        mov     ebx, [work_key2_esi]
        mov     eax, S2(25)

    ; ---------------------------------------------------- */
    ; Begin round 3 of key expansion mixed with encryption */
    ; ---------------------------------------------------- */
    ; (second key)					    */

	; A  = %eax  eA = %esi
	; L0 = %ebx  eB = %edi
	; L1 = %edx  .. = %ebp

        lea     ebp, S2(0)

	; A = ROTL3(S00 + A + L1);
	; eA = P_0 + A;
	; L0 = ROTL(L0 + A + L1, A + L1);
        add     eax, edx        ; 1
        mov     ecx, edx        ;
        add     eax, S3(0)      ; 1
        rol     eax, 3          ; 1
        mov     esi, [work_P_0] ;
        add     esi, eax        ; 1
        add     ecx, eax        ;
        add     ebx, ecx        ; 1
        add     eax, S3(1)      ;
        rol     ebx, cl         ; 2
	; A = ROTL3(S01 + A + L0);
	; eB = P_1 + A;
	; L1 = ROTL(L1 + A + L0, A + L0);
        add     eax, ebx        ;
        mov     ecx, ebx        ; 1
        rol     eax, 3          ;
        mov     edi, [work_P_1] ; 1
        add     edi, eax        ;
        add     ecx, eax        ; 1
        add     edx, ecx        ; 1
        rol     edx, cl         ; 2
	ROUND_3_EVEN_AND_ODD  2
	ROUND_3_EVEN_AND_ODD  4
	ROUND_3_EVEN_AND_ODD  6
	ROUND_3_EVEN_AND_ODD  8
	ROUND_3_EVEN_AND_ODD 10
	ROUND_3_EVEN_AND_ODD 12
	ROUND_3_EVEN_AND_ODD 14
	ROUND_3_EVEN_AND_ODD 16
	ROUND_3_EVEN_AND_ODD 18
	ROUND_3_EVEN_AND_ODD 20
	ROUND_3_EVEN_AND_ODD 22
	; early exit
align 4
_end_round3_2_6x86:
        add     eax, S3(24)     ;       A = ROTL3(S24 + A + L1);
        mov     ecx, edi        ; 1     eA = ROTL(eA ^ eB, eB) + A;
        add     eax, edx        ;
        xor     esi, edi        ; 1
        rol     eax, 3          ;
        rol     esi, cl         ; 2
        add     esi, eax        ; 1
					
        cmp     esi, [work_C_0]
        jne     __exit_2_6x86
	
        mov     ecx, eax        ; 1     L0 = ROTL(L0 + A + L1, A + L1);
        add     ecx, edx        ;       A = ROTL3(S25 + A + L0);
        xor     edi, esi        ; 1     eB = ROTL(eB ^ eA, eA) + A;
        add     ebx, ecx        ;
        rol     ebx, cl         ; 2
        add     eax, S3(25)     ;
        mov     ecx, esi        ; 1
        add     eax, ebx        ;
        rol     edi, cl         ; 2
        rol     eax, 3          ;
        add     edi, eax        ; 1

        cmp     edi, [work_C_1]
        jne     __exit_2_6x86
        mov     dword [work_add_iter], 1
        jmp     _full_exit_6x86

align 4
__exit_2_6x86:
        mov     edx, [work_key_hi]

; Jumps not taken are faster
        add     edx, 0x02000000
        jc near _next_inc_6x86

align 4
_next_iter_6x86:
        mov     [work_key_hi], edx
        lea     edi, [0x01000000+edx]
        dec     dword [work_iterations]
        jg near _loaded_6x86
        mov     eax, [RC5UnitWork]                      ; pointer to rc5unitwork
        mov     ebx, [work_key_lo]
        mov     [RC5UnitWork_L0lo], ebx                 ; Update real data
        mov     [RC5UnitWork_L0hi], edx                 ; (used by caller)
        jmp     _full_exit_6x86

align 4
_next_iter2_6x86:
        mov     [work_key_lo], ebx
        mov     [work_key_hi], edx
        lea     edi, [0x01000000+edx]
        mov     esi, ebx
        dec     dword [work_iterations]
        jg near _bigger_loop_6x86
        mov     eax, [RC5UnitWork]                      ; pointer to rc5unitwork
        mov     [RC5UnitWork_L0lo], ebx                 ; Update real data
        mov     [RC5UnitWork_L0hi], edx                 ; (used by caller)
        jmp     _full_exit_6x86

align 4
_next_inc_6x86:
        add     edx, 0x00010000
        test    edx, 0x00FF0000
        jnz near _next_iter_6x86

        add     edx, 0xFF000100
        test    edx, 0x0000FF00
        jnz near _next_iter_6x86

        add     edx, 0xFFFF0001
        test    edx, 0x000000FF
        jnz near _next_iter_6x86


        mov     ebx, [work_key_lo]

        sub     edx, 0x00000100
        add     ebx, 0x01000000
        jnc near _next_iter2_6x86

        add     ebx, 0x00010000
        test    ebx, 0x00FF0000
        jnz near _next_iter2_6x86

        add     ebx, 0xFF000100
        test    ebx, 0x0000FF00
        jnz near _next_iter2_6x86

        add     ebx, 0xFFFF0001
        test    ebx, 0x000000FF
        jnz near _next_iter2_6x86

	; Moo !
	; We have just finished checking the last key
	; of the rc5-64 keyspace...
	; Not much to do here, since we have finished the block ...


align 4
_full_exit_6x86:
mov ebp, [timeslice]
sub ebp, [work_iterations]
mov eax, [work_add_iter]
lea edx, [eax+ebp*2]
mov eax, edx

;    return (timeslice - work.iterations) * 2 + work.add_iter;


      mov ebx, [save_ebx]
      mov esi, [save_esi]
      mov edi, [save_edi]
      mov ebp, [save_ebp]

     add esp, work_size ; restore stack pointer

     ret


