; Pentium optimized version
; Rémi Guyomarch - rguyom@mail.dotcom.fr - 97/07/13
;
; $Id: brf-p5.asm,v 1.1.2.1 2001/01/21 17:44:40 cyp Exp $
;
; Minor improvements:
; Bruce Ford - b.ford@qut.edu.au - 97/12/21
;
; roll %cl, ... can't pair
; roll $3,  ... can't pair either :-(
; (despite what intel say)
; (their manual is really screwed up :-( )
;
; it seems that only roll $1, ... can pair :-(
;
; read after write, do not pair
; write after write, do not pair
;
; write after read, pair OK
; read after read, pair OK
; read and write after read, pair OK
;
; For a really *good* pentium optimization manual :
;	http://announce.com/agner/assem

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
[SECTION _TEXT USE32 ALIGN=16]
%else
[SECTION .text]
%endif


[GLOBAL _rc5_unit_func_p5]
[GLOBAL rc5_unit_func_p5]

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
%define work_add_iter   esp+16     ; +  0
%define work_key_hi     esp+4+16   ; +  4
%define work_key_hi1    esp+5+16   ; +  5
%define work_key_hi2    esp+6+16   ; +  6
%define work_key_hi3    esp+7+16   ; +  7
%define work_key_lo     esp+8+16   ; +  8
%define work_key_lo1    esp+9+16   ; +  9
%define work_key_lo2    esp+10+16  ; +  10
%define work_key_lo3    esp+11+16  ; +  11
%define work_L0_ecx     esp+12+16  ; +  12 Used to store results from
%define work_L0_ebx     esp+16+16  ; +  16 L0 calculations made outside
%define work_L0_esi     esp+20+16  ; +  20 the main key loop.  BRF
%define work_P_0        esp+24+16  ; +  24 Order changed to help with cache access.  BRF
%define work_P_1        esp+28+16  ; +  28
%define work_s1         esp+32+16  ; +  32
%define work_s2         esp+36+16  ; +  136
%define work_C_0        esp+240+16 ; +  240
%define work_C_1        esp+244+16 ; +  244
 ; key2_ebp removed as it is identical to s2[25]. BRF
%define work_key2_edi   esp+248+16 ; +  248
%define work_key2_esi   esp+252+16 ; +  252
%define work_iterations esp+256+16 ; +  256

; Offsets to access RC5UnitWork fields

%define RC5UnitWork_plainhi   eax+0
%define RC5UnitWork_plainlo   eax+4
%define RC5UnitWork_cipherhi  eax+8
%define RC5UnitWork_cipherlo  eax+12
%define RC5UnitWork_L0hi      eax+16
%define RC5UnitWork_L0lo      eax+20

;  Macros to access the S arrays.

%define S1(N)    [((N)*8)+work_s1]
%define S2(N)    [((N)*8)+work_s2]

  ; A1   = %eax  A2   = %ebp
  ; Llo1 = %ebx  Llo2 = %esi
  ; Lhi1 = %edx  Lhi2 = %edi

; ------------------------------------------------------------------
; Merge end of previous iteration with next iteration
; to avoid AGI stall on %edi / %esi
; ROUND_1_LAST will merge with ROUND_2_EVEN

; S1(N) = A1 = ROTL3 (A1 + Lhi1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Lhi2 + S_not(N));
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
%macro ROUND_1_EVEN 1
        rol     esi, cl                         ; 4
        lea     ecx, [ebx+edx]                  ; 1
        mov     S2(%1), ebp                     ;
        add     eax, ecx                        ; 1
        lea     ebp, [S_not(%1+1)+ebp+esi]      ;
        rol     ebp, 3                          ; 1
        rol     eax, cl                         ; 4
        lea     ecx, [ebp+esi]                  ; 1
        mov     S1(%1), ebx                     ;
        add     edi, ecx                        ; 1
        lea     ebx, [S_not(%1+1)+ebx+eax]      ;
        rol     ebx, 3                          ; 1  sum = 14
%endmacro

; S1(N) = A1 = ROTL3 (A1 + Llo1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Llo2 + S_not(N));
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
%macro ROUND_1_ODD 1
        rol     edi, cl                         ; 4
        lea     ecx, [eax+ebx]                  ; 1
        mov     S2(%1), ebp                     ;
        add     edx, ecx                        ; 1
        lea     ebp, [S_not(%1+1)+ebp+edi]      ;
        rol     ebp, 3                          ; 1
        rol     edx, cl                         ; 4
        lea     ecx, [ebp+edi]                  ; 1
        mov     S1(%1), ebx                     ;
        add     esi, ecx                        ; 1
        lea     ebx, [S_not(%1+1)+ebx+edx]      ;
        rol     ebx, 3                          ; 1  sum = 14
%endmacro

; Same as above, but wrap to first part of round 2
%macro ROUND_1_LAST 1
        rol     edi, cl                         ; 4
        lea     ecx, [eax+ebx]                  ; 1
        mov     S2(25), ebp                     ;
        add     edx, ecx                        ; 1
        lea     ebp, [S0_ROTL3+ebp+edi]         ;
        rol     ebp, 3                          ; 1
        rol     edx, cl                         ; 4
        lea     ecx, [ebp+edi]                  ; 1
        mov     S1(25), ebx                     ;
        lea     ebx, [S0_ROTL3+ebx+edx]         ; 1
        add     esi, ecx                        ;
        rol     ebx, 3                          ; 1
        rol     esi, cl                         ; 4
        mov     S2(0), ebp                      ; 1
        mov     ecx, [work_L0_ebx]              ;
        add     ebp, ecx                        ; 1
        lea     ecx, [ebx+edx]                  ;
        add     ebp, esi                        ; 1
        mov     S1(0), ebx                      ;
        add     eax, ecx                        ; 1   Spare slot
                                                ;     sum = 22
%endmacro

%macro  ROUND_1_ODD_AND_EVEN 2
	ROUND_1_ODD  %1
	ROUND_1_EVEN %2
%endmacro
; ------------------------------------------------------------------
; Merge 'even' with 'odd', it reduce this macros by 1 cycle

; S1N = A1 = ROTL3 (A1 + Lhi1 + S1N);
; S2N = A2 = ROTL3 (A2 + Lhi2 + S2N);
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
%macro ROUND_2_EVEN 1
        rol     ebp, 3                          ; 1
        rol     edx, cl                         ; 4
        lea     ecx, [ebp+edi]                  ; 1
        mov     S2(%1), ebp                     ;
        add     ebx, edx                        ; 1
        add     esi, ecx                        ;
        add     ebx, S1(%1)                     ; 2
        add     ebp, S2(%1+1)                   ;
        rol     ebx, 3                          ; 1
        rol     esi, cl                         ; 4
        lea     ecx, [ebx+edx]                  ; 1
        add     ebp, esi                        ;
        mov     S1(%1), ebx                     ; 1
        add     eax, ecx                        ;   sum = 16
%endmacro
	
; S1N = A1 = ROTL3 (A1 + Llo1 + S1N);
; S2N = A2 = ROTL3 (A2 + Llo2 + S2N);
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
%macro ROUND_2_ODD 1
        rol     ebp, 3                          ; 1
        rol     eax, cl                         ; 4
        lea     ecx, [ebp+esi]                  ; 1
        mov     S2(%1), ebp                     ;
        add     ebx, eax                        ; 1
        add     edi, ecx                        ;
        add     ebx, S1(%1)                     ; 2
        add     ebp, S2(%1+1)                   ;
        rol     ebx, 3                          ; 1
        rol     edi, cl                         ; 4
        lea     ecx, [ebx+eax]                  ; 1
        add     ebp, edi                        ;
        mov     S1(%1), ebx                     ; 1
        add     edx, ecx                        ;   sum = 16
%endmacro

%macro  ROUND_2_ODD_AND_EVEN 2
	ROUND_2_ODD  %1
	ROUND_2_EVEN %2
%endmacro

; ------------------------------------------------------------------
; It's faster to do 1 key at a time with round3 and encryption mixed
; than to do 2 keys at once but round3 and encryption separated
; Too bad x86 hasn't more registers ...
	
; Assume the following code has already been executed :
;	movl	S1(N),  %ebp
; It reduce this macro by 2 cycles.
; note: the last iteration will be test for short exit, the
; last iteration of this macros won't be the last iteration for
; the third round.
; well, if it's not very clear, look at RC5_ROUND_3...

; eA1 = ROTL (eA1 ^ eB1, eB1) + (A1 = ROTL3 (A1 + Lhi1 + S1(N)));
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; eB1 = ROTL (eA1 ^ eB1, eA1) + (A1 = ROTL3 (A1 + Llo1 + S1(N)));
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);

; A  = %eax  eA = %esi
; L0 = %ebx  eB = %edi
; L1 = %edx  .. = %ebp

%define Sx(N,M) [work_s1+((N)*8)+(M-1)*4]

%macro ROUND_3_EVEN_AND_ODD 2
        add     ebx, edx                ; 1
        mov     ecx, edi                ;
        add     ebx, ebp                ; 1
        xor     esi, edi                ;
        rol     ebx, 3                  ; 1
        rol     esi, cl                 ; 4
        lea     ecx, [ebx+edx]          ; 1
        add     esi, ebx                ;
        add     eax, ecx                ; 1
        mov     ebp, Sx(%1+1,%2)        ;
        rol     eax, cl                 ; 4
					
        add     ebx, eax                ; 1
        mov     ecx, esi                ;
        add     ebx, ebp                ; 1
        xor     edi, esi                ;
        rol     ebx, 3                  ; 1
        rol     edi, cl                 ; 4
        lea     ecx, [eax+ebx]          ; 1
        add     edi, ebx                ;
        add     edx, ecx                ; 1
        mov     ebp, Sx(%1+2,%2)        ;
        rol     edx, cl                 ; 4   sum = 26
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
_rc5_unit_func_p5:
rc5_unit_func_p5:
;u32 rc5_unit_func_p5( RC5UnitWork * rc5unitwork, u32 timeslice )

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

align 4
_loaded_p5:
    ; -----------------------------------------------------------
    ; Pre-calculate first rotate of L0 as it rarely changes.  BRF
    ; -----------------------------------------------------------

        mov     esi, [work_key_lo]      ; 1
        mov     ebx, S1_S0_ROTL3        ;
        add     esi, S0_ROTL3           ; 1   Spare slot (not that it matters here)  BRF
        rol     esi, FIRST_ROTL         ; 1
        add     ebx, esi                ; 1
        mov     ecx, esi                ;
        rol     ebx, 3                  ; 1
        add     ecx, ebx                ; 1
        mov     [work_L0_ebx], ebx      ;
        mov     [work_L0_esi], esi      ; 1
        mov     [work_L0_ecx], ecx      ;  sum = 7 every 2147483648 loops or on subroutine
					;        entry.  The latter happens more often.  BRF

align 4
_next_key:
    ; ------------------------------
    ; Begin round 1 of key expansion
    ; ------------------------------

    mov   edi, [work_key_hi]            ; 1
    mov   ebx, [work_L0_ebx]            ;
    add   edi, ecx                      ;
    mov   edx, edi                      ; 1
    add   edi, 0x01000000               ;
    rol   edi, cl                       ; 4
    rol   edx, cl                       ; 4
    lea   ebp, [S_not(2)+ebx+edi]       ; 1
    mov   ecx, edi                      ;
    rol   ebp, 3                        ; 1
    lea   ebx, [S_not(2)+ebx+edx]       ; 1
    add   ecx, ebp                      ;
    rol   ebx, 3                        ; 1
    add   esi, ecx                      ; 1
    mov   eax, [work_L0_esi]            ;  sum = 16
					
  	ROUND_1_EVEN          2
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
  	ROUND_1_LAST         25


    ; ------------------------------
    ; Begin round 2 of key expansion
    ; ------------------------------

	; see end of ROUND_1_LAST

align 4
_end_round1_p5:
        rol     ebp, 3                  ; 1
        rol     eax, cl                 ; 4
        lea     ecx, [ebp+esi]          ; 1
        mov     S2(1), ebp              ;
        add     ebx, eax                ; 1
        add     edi, ecx                ;
        add     ebx, [work_L0_ebx]      ; 2
        add     ebp, S2(2)              ;
        rol     ebx, 3                  ; 1
        rol     edi, cl                 ; 4
        lea     ecx, [ebx+eax]          ; 1
        add     ebp, edi                ;
        mov     S1(1), ebx              ; 1
        add     edx, ecx                ;   sum = 16

     ROUND_2_EVEN          2
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

        rol     ebp, 3                  ; 1
        rol     eax, cl                 ; 4
        lea     ecx, [ebp+esi]          ; 1
        mov     S2(25), ebp             ;
        add     ebx, eax                ; 1
        mov     ebp, S1(25)             ;
        add     ebx, ebp                ; 1
        add     edi, ecx                ;
        rol     ebx, 3                  ; 1
        rol     edi, cl                 ; 4
        lea     ecx, [ebx+eax]          ; 1
        mov     [work_key2_edi], edi    ;
        mov     S1(25), ebx             ; 1
        add     edx, ecx                ;
        rol     edx, cl                 ; 4
        mov     [work_key2_esi], esi    ; 1
        add     ebx, edx                ;   sum = 20

align 4
_end_round2_p5:
    ; Save 2nd key parameters and initialize result variable


    ; ----------------------------------------------------
    ; Begin round 3 of key expansion mixed with encryption
    ; ----------------------------------------------------
    ; (first key)					

	; A  = %eax  eA = %esi
	; L0 = %ebx  eB = %edi
	; L1 = %edx  .. = %ebp

        mov     ebp, S1(0)              ; 1
        mov     esi, [work_P_0]         ;       eA = P_0 + A;
        add     ebp, ebx                ; 1
        mov     ebx, S1(1)              ;
        rol     ebp, 3                  ; 1
        add     esi, ebp                ; 1
        add     ebx, ebp                ;
        lea     ecx, [ebp+edx]          ; 1     L0 = ROTL(L0 + A + L1, A + L1);
        mov     edi, [work_P_1]         ;       eB = P_1 + A;
        add     eax, ecx                ; 1
                                        ;  Spare slot
        rol     eax, cl                 ; 4
					
        add     ebx, eax                ; 1     A = ROTL3(S00 + A + L1);
        mov     ecx, eax                ;       A = ROTL3(S03 + A + L0);
        rol     ebx, 3                  ; 1
        add     edi, ebx                ; 1
        add     ecx, ebx                ;
        add     edx, ecx                ; 1
        mov     ebp, S1(2)              ;
        rol     edx, cl                 ; 4     sum = 18

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
_end_round3_1_p5:
        add     ebx, edx                ; 1     A = ROTL3(S24 + A + L1);
        mov     ecx, edi                ;          eA = ROTL(eA ^ eB, eB) + A
        add     ebx, ebp                ; 1
        xor     esi, edi                ;
        rol     ebx, 3                  ; 1
        rol     esi, cl                 ; 4
        add     esi, ebx                ; 1
        mov     ebp, [work_C_0]         ;     Places je in V pipe for pairing.  BRF
					
        cmp     esi, ebp                ; 1
        je near _testC1_1_p5            ;  sum = 9
					
align 4
_second_key:
    ; Restore 2nd key parameters
        mov     edx, [work_key2_edi]    ; 1
        mov     ebx, S2(25)             ;
        mov     eax, [work_key2_esi]    ; 1
        add     ebx, edx                ;

    ; ----------------------------------------------------
    ; Begin round 3 of key expansion mixed with encryption
    ; ----------------------------------------------------
    ; (second key)					

	; A  = %eax  eA = %esi
	; L0 = %ebx  eB = %edi
	; L1 = %edx  .. = %ebp

        mov     ebp, S2(0)              ; 1
        mov     esi, [work_P_0]         ;       eA = P_0 + A;
        add     ebp, ebx                ; 1
        mov     ebx, S2(1)              ;
        rol     ebp, 3                  ; 1
        add     esi, ebp                ; 1
        add     ebx, ebp                ;
        lea     ecx, [ebp+edx]          ; 1     L0 = ROTL(L0 + A + L1, A + L1);
        mov     edi, [work_P_1]         ;       eB = P_1 + A;
        add     eax, ecx                ; 1
                                        ;  Spare slot
        rol     eax, cl                 ; 4

        add     ebx, eax                ; 1     A = ROTL3(S00 + A + L1);
        mov     ecx, eax                ;       A = ROTL3(S03 + A + L0);
        rol     ebx, 3                  ; 1
        add     edi, ebx                ; 1
        add     ecx, ebx                ;
        add     edx, ecx                ; 1
        mov     ebp, S2(2)              ;
        rol     edx, cl                 ; 4     sum = 20

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
_end_round3_2_p5:
        add     ebx, edx                ; 1     A = ROTL3(S24 + A + L1);
        mov     ecx, edi                ;       eA = ROTL(eA ^ eB, eB) + A
        add     ebx, ebp                ; 1
        xor     esi, edi                ;
        rol     ebx, 3                  ; 1
        rol     esi, cl                 ; 4
        add     esi, ebx                ; 1
        mov     ebp, [work_C_0]         ;    Places je in V pipe for pairing.  BRF
					
        cmp     esi, ebp                ; 1
        je near _testC1_2_p5            ;  sum = 9

align 4
_incr_key:
    dec dword [work_iterations]         ; 3
    jz near _full_exit_p5

    mov   dl, [work_key_hi3]            ; 1  All this is to try and save one clock
					;    at the jnc below
    mov   ecx, [work_L0_ecx]            ;    Costs nothing (in clocks) to try.  BRF
    add   dl, 2                         ; 1
    mov   esi, [work_L0_esi]            ;
    mov   [work_key_hi3], dl            ; 1
    jnc near _next_key                  ;

    inc   byte [work_key_hi2]
    jnz near _next_key
    inc   byte [work_key_hi1]
    jnz near _next_key
    inc   byte [work_key_hi]
    jnz near _next_key
    inc   byte [work_key_lo3]
    jnz near _loaded_p5
    inc   byte [work_key_lo2]

    jnz near _loaded_p5
    inc   byte [work_key_lo1]
    jnz near _loaded_p5
    inc   byte [work_key_lo]
    jmp   _loaded_p5                     ; Wrap the keyspace

align 4
_testC1_1_p5:
        lea     ecx, [ebx+edx]           ; 1    L0 = ROTL(L0 + A + L1, A + L1);
        mov     ebp, S1(25)              ;
        add     eax, ecx                 ; 1
        xor     edi, esi                 ;
        rol     eax, cl                  ; 4
        add     ebx, eax                 ; 1    A = ROTL3(S25 + A + L0);
        mov     ecx, esi                 ;      eB = ROTL(eB ^ eA, eA) + A
        add     ebx, ebp                 ; 1
                                         ;  Spare slot (not that it matters)  BRF
        rol     ebx, 3                   ; 1
        rol     edi, cl                  ; 4
        add     edi, ebx                 ; 1
        mov     ebp, [work_C_1]          ; Places jne in V pipe for pairing.  BRF

        cmp     edi, ebp                 ; 1
        jne near _second_key
    jmp   _done

align 4
_testC1_2_p5:
        lea     ecx, [ebx+edx]           ; 1     L0 = ROTL(L0 + A + L1, A + L1);
        mov     ebp, S2(25)              ;
        add     eax, ecx                 ; 1
        xor     edi, esi                 ;
        rol     eax, cl                  ; 4
        add     ebx, eax                 ; 1     A = ROTL3(S25 + A + L0);
        mov     ecx, esi                 ;       eB = ROTL(eB ^ eA, eA) + A
        add     ebx, ebp                 ; 1
                                         ;  Spare slot (not that it matters)  BRF
        rol     ebx, 3                   ; 1
        rol     edi, cl                  ; 4
        add     edi, ebx                 ; 1
        mov     ebp, [work_C_1]          ;       Places jne in V pipe for pairing.  BRF

        cmp     edi, ebp                 ; 1
        jne near _incr_key
        mov     dword [work_add_iter], 1
    jmp   _done

align 4
_full_exit_p5:
    add   byte [work_key_hi3], 2
    jnc   _key_updated
    inc   byte [work_key_hi2]
    jnz   _key_updated
    inc   byte [work_key_hi1]
    jnz   _key_updated
    inc   byte [work_key_hi]
    jnz   _key_updated
    inc   byte [work_key_lo3]
    jnz   _key_updated
    inc   byte [work_key_lo2]
    jnz   _key_updated
    inc   byte [work_key_lo1]
    jnz   _key_updated
    inc   byte [work_key_lo]

align 4
_key_updated:
        mov     eax, [RC5UnitWork]               ; pointer to rc5unitwork
        mov     ebx, [work_key_lo]
        mov     edx, [work_key_hi]
        mov     [RC5UnitWork_L0lo], ebx          ; update real data
        mov     [RC5UnitWork_L0hi], edx          ; (used by caller)

align 4
_done:
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


