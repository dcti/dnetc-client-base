; 386/486 optimized version
; Rémi Guyomarch - rguyom@mail.dotcom.fr
;
; $Id: rg-486.asm,v 1.1.2.1 2001/01/21 17:44:41 cyp Exp $
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
; 980104 :
;	- precalculate some things for ROUND1 & ROUND2
;
; Checking two keys at once is still a (slight) win for a 486
; probably because less load/store operations

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
[SECTION _TEXT USE32 ALIGN=16]
%else
[SECTION .text]
%endif

[GLOBAL _rc5_unit_func_486]
[GLOBAL rc5_unit_func_486]

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
%define work_add_iter   esp+16
%define work_s1         esp+4+16
%define work_s2         esp+108+16
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

;  Macros to access the S arrays.

%define S1(N)    [((N)*4)+work_s1]
%define S2(N)    [((N)*4)+work_s2]


  ; A1   = %eax  A2   = %ebp
  ; Llo1 = %ebx  Llo2 = %esi
  ; Lhi1 = %edx  Lhi2 = %edi

; ------------------------------------------------------------------
; S1(N) = A1 = ROTL3 (A1 + Lhi1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Lhi2 + S_not(N));
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
%macro ROUND_1_EVEN 1
        lea     eax, [S_not(%1)+eax+edx]        ; 2
        lea     ebp, [S_not(%1)+ebp+edi]        ; 2
        rol     eax, 3                          ; 2
        mov     S1(%1), eax                     ; 1
        rol     ebp, 3                          ; 2
        mov     S2(%1), ebp                     ; 1
        lea     ecx, [eax+edx]                  ; 2
        add     ebx, ecx                        ; 1
        rol     ebx, cl                         ; 3
        lea     ecx, [ebp+edi]                  ; 2
        add     esi, ecx                        ; 1
        rol     esi, cl                         ; 3     sum = 22
%endmacro
; S1(N) = A1 = ROTL3 (A1 + Llo1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Llo2 + S_not(N));
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
%macro ROUND_1_ODD 1
        lea     eax, [S_not(%1)+eax+ebx]        ; 2
        lea     ebp, [S_not(%1)+ebp+esi]        ; 2
        rol     eax, 3                          ; 2
        mov     S1(%1), eax                     ; 1
        rol     ebp, 3                          ; 2
        mov     S2(%1), ebp                     ; 1
        lea     ecx, [eax+ebx]                  ; 2
        add     edx, ecx                        ; 1
        rol     edx, cl                         ; 3
        lea     ecx, [ebp+esi]                  ; 2
        add     edi, ecx                        ; 1
        rol     edi, cl                         ; 3     sum = 22
%endmacro

%macro  ROUND_1_ODD_AND_EVEN 2
	ROUND_1_ODD  %1
	ROUND_1_EVEN %2
%endmacro

; ------------------------------------------------------------------
; S1N = A1 = ROTL3 (A1 + Lhi1 + S1N);
; S2N = A2 = ROTL3 (A2 + Lhi2 + S2N);
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
%macro  ROUND_2_EVEN 1
        add     eax, edx                ; 1
        add     eax, S1(%1)             ; 2
        add     ebp, edi                ; 1
        add     ebp, S2(%1)             ; 2
        rol     eax, 3                  ; 2
        mov     S1(%1), eax             ; 1
        rol     ebp, 3                  ; 2
        mov     S2(%1), ebp             ; 1
        lea     ecx, [eax+edx]          ; 2
        add     ebx, ecx                ; 1
        rol     ebx, cl                 ; 3
        lea     ecx, [ebp+edi]          ; 2
        add     esi, ecx                ; 1
        rol     esi, cl                 ; 3     sum = 24
%endmacro
	
; S1N = A1 = ROTL3 (A1 + Llo1 + S1N);
; S2N = A2 = ROTL3 (A2 + Llo2 + S2N);
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
%macro ROUND_2_ODD 1
        add     eax, ebx                ; 1
        add     eax, S1(%1)             ; 2
        add     ebp, esi                ; 1
        add     ebp, S2(%1)             ; 2
        rol     eax, 3                  ; 2
        mov     S1(%1), eax             ; 1
        rol     ebp, 3                  ; 2
        mov     S2(%1), ebp             ; 1
        lea     ecx, [eax+ebx]          ; 2
        add     edx, ecx                ; 1
        rol     edx, cl                 ; 3
        lea     ecx, [ebp+esi]          ; 2
        add     edi, ecx                ; 1
        rol     edi, cl                 ; 3     sum = 24
%endmacro

%macro  ROUND_2_ODD_AND_EVEN 2
	ROUND_2_ODD  %1
	ROUND_2_EVEN %2
%endmacro

; ------------------------------------------------------------------
; It's faster to do 1 key at a time with round3 and encryption mixed
; than to do 2 keys at once but round3 and encryption separated
; Too bad x86 hasn't more registers ...
	
; eA1 = ROTL (eA1 ^ eB1, eB1) + (A1 = ROTL3 (A1 + Lhi1 + S1(N)));
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; eB1 = ROTL (eA1 ^ eB1, eA1) + (A1 = ROTL3 (A1 + Llo1 + S1(N)));
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);

; A  = %eax  eA = %esi
; L0 = %ebx  eB = %edi
; L1 = %edx  .. = %ebp

%define Sx(N,M) [work_s1+((N)*4)+(M-1)*104]

%macro ROUND_3_EVEN_AND_ODD 2
        add     eax, edx                ; 1
        add     eax, Sx(%1,%2)          ; 2
        rol     eax, 3                  ; 2
        mov     ecx, edi                ; 1
        xor     esi, edi                ; 1
        rol     esi, cl                 ; 3
        add     esi, eax                ; 1
        lea     ecx, [eax+edx]          ; 2
        add     ebx, ecx                ; 1
        rol     ebx, cl                 ; 3
	
        add     eax, ebx                ; 1
        add     eax, Sx(%1+1,%2)        ; 2
        rol     eax, 3                  ; 2
        mov     ecx, esi                ; 1
        xor     edi, esi                ; 1
        rol     edi, cl                 ; 3
        add     edi, eax                ; 1
        lea     ecx, [eax+ebx]          ; 2
        add     edx, ecx                ; 1
        rol     edx, cl                 ; 3     sum = 34
%endmacro

; ------------------------------------------------------------------
; rc5_unit will get passed an RC5WorkUnit to complete
; this is where all the actually work occurs, this is where you optimize.
; assembly gurus encouraged.
; Returns number of keys checked before a possible good key is found, or
; timeslice*PIPELINE_COUNT if no keys are 'good' keys.
; (ie:      if (result == timeslice*PIPELINE_COUNT) NOTHING_FOUND
;      else if (result < timeslice*PIPELINE_COUNT) SOMETHING_FOUND at result+1
;      else SOMETHING_GET_WRONG... )

align 4
_rc5_unit_func_486:
rc5_unit_func_486:
;u32 rc5_unit_func_486( RC5UnitWork * rc5unitwork, u32 timeslice )

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
_bigger_loop_486:
        add     ebx, S0_ROTL3                   ; 1
        rol     ebx, FIRST_ROTL                 ; 3
        mov     [work_pre1_r1], ebx             ; 1

        lea     eax, [S1_S0_ROTL3+ebx]          ; 1
        rol     eax, 3                          ; 2
        mov     [work_pre2_r1], eax             ; 1

        lea     ecx, [eax+ebx]                  ; 2
        mov     [work_pre3_r1], ecx             ; 1

align 4
_loaded_486:
    ; ------------------------------
    ; Begin round 1 of key expansion
    ; ------------------------------

        mov     ebx, [work_pre1_r1]             ; 1
        mov     esi, ebx                        ; 1
        mov     eax, [work_pre2_r1]             ; 1
        mov     ebp, eax                        ; 1

        mov     ecx, [work_pre3_r1]
        add     edx, ecx                        ; 1
        rol     edx, cl                         ; 3
        add     edi, ecx                        ; 1
        rol     edi, cl                         ; 3     sum = 12

	ROUND_1_EVEN             2
	ROUND_1_ODD_AND_EVEN  3, 4
	ROUND_1_ODD_AND_EVEN  5, 6
	ROUND_1_ODD_AND_EVEN  7, 8
	ROUND_1_ODD_AND_EVEN  9,10
	ROUND_1_ODD_AND_EVEN 11,12
	ROUND_1_ODD_AND_EVEN 13,14
	ROUND_1_ODD_AND_EVEN 15,16
	ROUND_1_ODD_AND_EVEN 17,18
	ROUND_1_ODD_AND_EVEN 19,20
	ROUND_1_ODD_AND_EVEN 21,22
	ROUND_1_ODD_AND_EVEN 23,24
	ROUND_1_ODD          25


    ; ------------------------------
    ; Begin round 2 of key expansion
    ; ------------------------------

align 4
_end_round1_486:
        lea     eax, [S0_ROTL3+eax+edx]         ; 2
        lea     ebp, [S0_ROTL3+ebp+edi]         ; 2
        rol     eax, 3                  ; 2
        mov     S1(0), eax              ; 1
        rol     ebp, 3                  ; 2
        mov     S2(0), ebp              ; 1
	
        lea     ecx, [eax+edx]          ; 2
        add     ebx, ecx                ; 1
        rol     ebx, cl                 ; 3
        lea     ecx, [ebp+edi]          ; 2
        add     esi, ecx                ; 1
        rol     esi, cl                 ; 3

        mov     ecx, [work_pre2_r1]     ; 1
        add     eax, ebx                ; 1
        add     eax, ecx                ; 1
        add     ebp, esi                ; 1
        add     ebp, ecx                ; 1
        rol     eax, 3                  ; 2
        mov     S1(1), eax              ; 1
        rol     ebp, 3                  ; 2
        mov     S2(1), ebp              ; 1
        lea     ecx, [eax+ebx]          ; 2
        add     edx, ecx                ; 1
        rol     edx, cl                 ; 3
        lea     ecx, [ebp+esi]          ; 2
        add     edi, ecx                ; 1
        rol     edi, cl                 ; 3     sum = 23

        ROUND_2_EVEN             2
        ROUND_2_ODD_AND_EVEN  3, 4
        ROUND_2_ODD_AND_EVEN  5, 6
        ROUND_2_ODD_AND_EVEN  7, 8
        ROUND_2_ODD_AND_EVEN  9,10
        ROUND_2_ODD_AND_EVEN 11,12
        ROUND_2_ODD_AND_EVEN 13,14
        ROUND_2_ODD_AND_EVEN 15,16
        ROUND_2_ODD_AND_EVEN 17,18
        ROUND_2_ODD_AND_EVEN 19,20
        ROUND_2_ODD_AND_EVEN 21,22
        ROUND_2_ODD_AND_EVEN 23,24
	ROUND_2_ODD 25

    ; Save 2nd key parameters and initialize result variable

align 4
_end_round2_486:
        mov     [work_key2_esi], esi
        mov     [work_key2_edi], edi

    ; ----------------------------------------------------
    ; Begin round 3 of key expansion mixed with encryption
    ; ----------------------------------------------------
    ; (first key)					

	; A  = %eax  eA = %esi
	; L0 = %ebx  eB = %edi
	; L1 = %edx  .. = %ebp

        add     eax, edx                ; 1     A = ROTL3(S00 + A + L1);
        add     eax, S1(0)              ; 2
        rol     eax, 3                  ; 2
        mov     esi, [work_P_0]         ; 1     eA = P_0 + A;
        add     esi, eax                ; 1
        lea     ecx, [eax+edx]          ; 2     L0 = ROTL(L0 + A + L1, A + L1);
        add     ebx, ecx                ; 1
        rol     ebx, cl                 ; 3
	       			
        add     eax, ebx                ; 1     A = ROTL3(S01 + A + L0);
        add     eax, S1(1)              ; 2
        rol     eax, 3                  ; 2
        mov     edi, [work_P_1]         ; 1     eB = P_1 + A;
        add     edi, eax                ; 1
        lea     ecx, [eax+ebx]          ; 2     L1 = ROTL(L1 + A + L0, A + L0);
        add     edx, ecx                ; 1
        rol     edx, cl                 ; 3     sum = 26

	ROUND_3_EVEN_AND_ODD  2,1
	ROUND_3_EVEN_AND_ODD  4,1
	ROUND_3_EVEN_AND_ODD  6,1
	ROUND_3_EVEN_AND_ODD  8,1
	ROUND_3_EVEN_AND_ODD 10,1
	ROUND_3_EVEN_AND_ODD 12,1
	ROUND_3_EVEN_AND_ODD 14,1
	ROUND_3_EVEN_AND_ODD 16,1
	ROUND_3_EVEN_AND_ODD 18,1
	ROUND_3_EVEN_AND_ODD 20,1
	ROUND_3_EVEN_AND_ODD 22,1

	; early exit
align 4
_end_round3_1_486:
        add     eax, edx                ; 1     A = ROTL3(S24 + A + L1);
        add     eax, S1(24)             ; 2
        rol     eax, 3                  ; 2
        mov     ecx, edi                ; 1     eA = ROTL(eA ^ eB, eB) + A
        xor     esi, edi                ; 1
        rol     esi, cl                 ; 3
        add     esi, eax                ; 1
					
        cmp     esi, [work_C_0]
        jne     __exit_1_486
					
        lea     ecx, [eax+edx]          ; 2     L0 = ROTL(L0 + A + L1, A + L1);
        add     ebx, ecx                ; 1
        rol     ebx, cl                 ; 3
        add     eax, ebx                ; 1     A = ROTL3(S25 + A + L0);
        add     eax, S1(25)             ; 2
        rol     eax, 3                  ; 2
        mov     ecx, esi                ; 1     eB = ROTL(eB ^ eA, eA) + A

        xor     edi, esi                ; 1
        rol     edi, cl                 ; 3
        add     edi, eax                ; 1

        cmp     edi, [work_C_1]
        je near _full_exit_486

align 4
__exit_1_486:
    ; Restore 2nd key parameters */
        mov     edx, [work_key2_edi]
        mov     ebx, [work_key2_esi]
	mov	eax, S2(25)

    ; ---------------------------------------------------- */
    ; Begin round 3 of key expansion mixed with encryption */
    ; ---------------------------------------------------- */
    ; (second key)					    */

	; A  = %eax  eA = %esi
	; L0 = %ebx  eB = %edi
	; L1 = %edx  .. = %ebp

        add     eax, edx                ; 1     A = ROTL3(S00 + A + L1);
        add     eax, S2(0)              ; 2
        rol     eax, 3                  ; 2
        mov     esi, [work_P_0]         ; 1     eA = P_0 + A;
        add     esi, eax                ; 1
        lea     ecx, [eax+edx]          ; 2     L0 = ROTL(L0 + A + L1, A + L1);
        add     ebx, ecx                ; 1
        rol     ebx, cl                 ; 3
	       			
        add     eax, ebx                ; 1     A = ROTL3(S01 + A + L0);
        add     eax, S2(1)              ; 2
        rol     eax, 3                  ; 2
        mov     edi, [work_P_1]         ; 1     eB = P_1 + A;
        add     edi, eax                ; 1
        lea     ecx, [eax+ebx]          ; 2     L1 = ROTL(L1 + A + L0, A + L0);
        add     edx, ecx                ; 1
        rol     edx, cl                 ; 3     sum = 26
	ROUND_3_EVEN_AND_ODD  2,2
	ROUND_3_EVEN_AND_ODD  4,2
	ROUND_3_EVEN_AND_ODD  6,2
	ROUND_3_EVEN_AND_ODD  8,2
	ROUND_3_EVEN_AND_ODD 10,2
	ROUND_3_EVEN_AND_ODD 12,2
	ROUND_3_EVEN_AND_ODD 14,2
	ROUND_3_EVEN_AND_ODD 16,2
	ROUND_3_EVEN_AND_ODD 18,2
	ROUND_3_EVEN_AND_ODD 20,2
	ROUND_3_EVEN_AND_ODD 22,2

	; early exit
align 4
_end_round3_2_486:
        add     eax, edx                ; 1     A = ROTL3(S24 + A + L1);
        add     eax, S2(24)             ; 2
        rol     eax, 3                  ; 2
        mov     ecx, edi                ; 1     eA = ROTL(eA ^ eB, eB) + A
        xor     esi, edi                ; 1
        rol     esi, cl                 ; 3
        add     esi, eax                ; 1
					
        cmp     esi, [work_C_0]
        jne     __exit_2_486
					
        lea     ecx, [eax+edx]          ; 2     L0 = ROTL(L0 + A + L1, A + L1);
        add     ebx, ecx                ; 1
        rol     ebx, cl                 ; 3
        add     eax, ebx                ; 1     A = ROTL3(S25 + A + L0);
        add     eax, S2(25)             ; 2
        rol     eax, 3                  ; 2
        mov     ecx, esi                ; 1     eB = ROTL(eB ^ eA, eA) + A
        xor     edi, esi                ; 1
        rol     edi, cl                 ; 3
        add     edi, eax                ; 1

        cmp     edi, [work_C_1]
        jne     __exit_2_486
        mov     dword [work_add_iter], 1
        jmp     _full_exit_486

align 4
__exit_2_486:
        mov     edx, [work_key_hi]

; Jumps not taken are faster
        add     edx, 0x02000000
        jc near _next_inc_486

align 4
_next_iter_486:
        mov     [work_key_hi], edx
        lea     edi, [0x01000000+edx]
        dec     dword [work_iterations]
        jg near _loaded_486
        mov     eax, [RC5UnitWork]                ; pointer to rc5unitwork
        mov     ebx, [work_key_lo]
        mov     [RC5UnitWork_L0lo], ebx           ; update real data
        mov     [RC5UnitWork_L0hi], edx           ; (used by caller)
        jmp     _full_exit_486

align 4
_next_iter2_486:
        mov     [work_key_lo], ebx
        mov     [work_key_hi], edx
        lea     edi, [0x01000000+edx]
        mov     esi, ebx
        dec     dword [work_iterations]
        jg near _bigger_loop_486
        mov     eax, [RC5UnitWork]               ; pointer to rc5unitwork
        mov     [RC5UnitWork_L0lo], ebx          ; update real data
        mov     [RC5UnitWork_L0hi], edx          ; (used by caller)
        jmp     _full_exit_486

align 4
_next_inc_486:
        add     edx, 0x00010000
        test    edx, 0x00FF0000
        jnz near _next_iter_486

        add     edx, 0xFF000100
        test    edx, 0x0000FF00
        jnz near _next_iter_486

        add     edx, 0xFFFF0001
        test    edx, 0x000000FF
        jnz near _next_iter_486


        mov     ebx, [work_key_lo]

        sub     edx, 0x00000100
        add     ebx, 0x01000000
        jnc near _next_iter2_486

        add     ebx, 0x00010000
        test    ebx, 0x00FF0000
        jnz near _next_iter2_486

        add     ebx, 0xFF000100
        test    ebx, 0x0000FF00
        jnz near _next_iter2_486

        add     ebx, 0xFFFF0001
        test    ebx, 0x000000FF
        jnz near _next_iter2_486

	; Moo !
	; We have just finished checking the last key
	; of the rc5-64 keyspace...
	; Not much to do here, since we have finished the block ...


align 4
_full_exit_486:
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


