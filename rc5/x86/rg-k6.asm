; Faster K6 core :
; 	query : K6/200 - Linux 2.1.129 - client running in the background
; 				       - idle otherwise
; 	(reported speed and %CPU usage for -benchmarkrc5)
; 	previous :	this one :
; 	350.7 (91.5%)	365.5 (91.2%)
; 	348.0 (91.0%)	366.8 (91.5%)
; 	351.5 (91.7%)	366.7 (91.5%)
;
; 	nodezero : K6-2/300 - FreeBSD 2.2.7-STABLE - client running
; 			    - apache - dcti full proxy - mysql etc ...
; 	(user time for -benchmarkrc5)
; 	previous : 17.18, 17.25, 17.32
; 	this one : 16.17, 16.12, 16.18
;
; K6 optimized version
; R�mi Guyomarch - rguyom@mail.dotcom.fr
;
; $Id: rg-k6.asm,v 1.3 2002/10/09 22:22:16 andreasb Exp $
;
; 980307 :
;	- Finally a K6 optimization that works ...
;	- Based on the 486/980226 version

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
[SECTION _TEXT FLAT USE32 ALIGN=16 CLASS=CODE]
%else
[SECTION .text]
%endif

[GLOBAL rc5_unit_func_k6_]
[GLOBAL _rc5_unit_func_k6]
[GLOBAL rc5_unit_func_k6]

%define work_size       276

%define RC5UnitWork     esp+work_size+4
%define timeslice       esp+work_size+8

%define P 0xB7E15163
%define Q 0x9E3779B9
%define S_not(N) ((P+Q*(N)))

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
;%define work_key2_ebp   esp+228+16
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
        lea     eax, [S_not(%1)+eax+edx]        ; 1
        lea     ebp, [S_not(%1)+ebp+edi]        ;
        lea     ecx, [eax*8]                    ; 1 st
        shr     eax, 29                         ;   aluX
        or      eax, ecx                        ; 1 aluX
        lea     ecx, [ebp*8]                    ;   st
        shr     ebp, 29                         ; 1 aluX
        mov     S1(%1),eax                      ; 1 st
        or      ebp, ecx                        ;        aluY
        lea     ecx, [eax+edx]                  ;   aluX
        add     ebx, ecx                        ; 1 aluX
        mov     S2(%1),ebp                      ;        st
        rol     ebx, cl                         ; 2
        lea     ecx, [ebp+edi]                  ; 1
        add     esi, ecx                        ;
        rol     esi, cl                         ; 2	sum = 12
%endmacro

; S1(N) = A1 = ROTL3 (A1 + Llo1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Llo2 + S_not(N));
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
%macro ROUND_1_ODD 1
        lea     eax, [S_not(%1)+eax+ebx]        ; 1
        lea     ebp, [S_not(%1)+ebp+esi]        ;
        lea     ecx, [eax*8]                    ; 1 st
        shr     eax, 29                         ;   aluX
        or      eax, ecx                        ; 1 aluX
        lea     ecx, [ebp*8]                    ;   st
        shr     ebp, 29                         ; x aluX
        mov     S1(%1), eax                     ; 1
        or      ebp, ecx                        ;        aluY
        lea     ecx, [eax+ebx]                  ;
        add     edx, ecx                        ; 1
        mov     S2(%1), ebp                     ;
        rol     edx, cl                         ; 2
        lea     ecx, [ebp+esi]                  ; 1
        add     edi, ecx                        ;
        rol     edi, cl                         ; 2     sum = 12
%endmacro

%macro ROUND_1_LAST 1
        lea     eax, [S_not(%1)+eax+ebx]        ; 1
        lea     ebp, [S_not(%1)+ebp+esi]        ;
        lea     ecx, [eax*8]                    ; 1 st
        shr     eax, 29                         ;   aluX
        or      eax, ecx                        ; 1 aluX
        lea     ecx, [ebp*8]                    ;   st
        shr     ebp, 29                         ; x aluX
        mov     S1(%1), eax                     ; 1
        or      ebp, ecx                        ;        aluY
        lea     ecx, [eax+ebx]                  ;
        add     eax, S0_ROTL3                   ;   alu
        add     edx, ecx                        ; 1
        mov     S2(%1), ebp                     ;
        rol     edx, cl                         ; 2
        lea     ecx, [ebp+esi]                  ; 1
        add     ebp, S0_ROTL3                   ;   alu
        add     edi, ecx                        ;
        rol     edi, cl                         ; 2     sum = 12
%endmacro

%macro ROUND_1_ODD_AND_EVEN 2
	ROUND_1_ODD %1
	ROUND_1_EVEN %2
%endmacro


; ------------------------------------------------------------------
; S1N = A1 = ROTL3 (A1 + Lhi1 + S1N);
; S2N = A2 = ROTL3 (A2 + Lhi2 + S2N);
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
%macro ROUND_2_EVEN 1
        add     eax, edx                ; 1 alu
        add     ebp, edi                ;   alu
        lea     ecx, [eax*8]            ; 1 st
        shr     eax, 29                 ;   aluX
        or      eax, ecx                ; 1 aluX
        lea     ecx, [ebp*8]            ;   st
        shr     ebp, 29                 ; 1 aluX
        mov     S1(%1), eax             ;   st
        or      ebp, ecx                ; 1 alu
        lea     ecx, [eax+edx]          ;   st
        add     ebx, ecx                ; 1 alu
        mov     S2(%1), ebp             ;   st
        rol     ebx, cl                 ; 2 ?
        lea     ecx, [ebp+edi]          ;   st
        add     eax, S1(%1+1)           ;   ld  alu
        add     esi, ecx                ;   alu
        add     ebp, S2(%1+1)           ;   ld  alu
        rol     esi, cl                 ; 2 ?
%endmacro

; S1N = A1 = ROTL3 (A1 + Llo1 + S1N);
; S2N = A2 = ROTL3 (A2 + Llo2 + S2N);
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
%macro ROUND_2_ODD 1
        add     eax, ebx                ;
        add     ebp, esi                ;
        lea     ecx, [eax*8]            ;
        shr     eax, 29                 ;
        or      eax, ecx                ;
        lea     ecx, [ebp*8]            ;
        shr     ebp, 29                 ;
        mov     S1(%1), eax             ;
        or      ebp, ecx                ;
        lea     ecx, [eax+ebx]          ;
        add     edx, ecx                ;
        mov     S2(%1), ebp             ;
        rol     edx, cl                 ;
        lea     ecx, [ebp+esi]          ;
        add     eax, S1(%1+1)           ;
        add     edi, ecx                ;
        add     ebp, S2(%1+1)           ;
        rol     edi, cl                 ;
%endmacro

%macro ROUND_2_LAST 1
        add     eax, ebx                ;
        add     ebp, esi                ;
        lea     ecx, [eax*8]            ;
        shr     eax, 29                 ;
        or      eax, ecx                ;
        lea     ecx, [ebp*8]            ;
        shr     ebp, 29                 ;
        or      ebp, ecx                ;
        mov     S1(%1), eax             ;
        lea     ecx, [eax+ebx]          ;
        add     edx, ecx                ;
        mov     S2(%1), ebp             ;
        rol     edx, cl                 ;
        lea     ecx, [ebp+esi]          ;
;        mov     [work_key2_ebp], ebp    ;
        add     edi, ecx                ;
        mov     [work_key2_esi], esi    ;
        rol     edi, cl                 ;
%endmacro

%macro  ROUND_2_ODD_AND_EVEN 2
	ROUND_2_ODD %1
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
        add     eax, edx
        add     eax, Sx(%1,%2)
        rol     eax, 3
        mov     ecx, edi
        xor     esi, edi
        rol     esi, cl
        add     esi, eax
        lea     ecx, [eax+edx]
        add     ebx, ecx
        rol     ebx, cl

        add     eax, ebx
        add     eax, Sx(%1+1,%2)
        rol     eax, 3
        mov     ecx, esi
        xor     edi, esi
        rol     edi, cl
        add     edi, eax
        lea     ecx, [eax+ebx]
        add     edx, ecx
        rol     edx, cl
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

; I use a structure to make this function multi-thread safe.
; (can't use static variables, and can't use push/pop in this
;  function because &work_struct is relative to %esp)

rc5_unit_func_k6_:
_rc5_unit_func_k6:
rc5_unit_func_k6:
;u32 rc5_unit_func_k6( RC5UnitWork * rc5unitwork, u32 timeslice )

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

;	/* load parameters */
        mov     ebx, [RC5UnitWork_L0lo]                 ; ebx = l0 = Llo1
        mov     edx, [RC5UnitWork_L0hi]                 ; edx = l1 = Lhi1
        mov     esi, ebx				; esi = l2 = Llo2

        lea     edi, [0x01000000+edx]                   ; edi = l3 = lhi2
        mov     [work_key_lo], ebx
        mov     [work_key_hi], edx

;	/* Save other parameters */
;	/* (it's faster to do so, since we will only load 1 value */
;	/* each time in RC5_ROUND_3xy, instead of two if we save  */
;	/* only the pointer to the RC5 struct)                    */
        mov     ebp, [RC5UnitWork_plainlo]
        mov     [work_P_0], ebp
        mov     ebp, [RC5UnitWork_plainhi]
        mov     [work_P_1], ebp
        mov     ebp, [RC5UnitWork_cipherlo]
        mov     [work_C_0], ebp
        mov     ebp, [RC5UnitWork_cipherhi]
        mov     [work_C_1], ebp

;	/* Pre-calculate things. Assume work.key_lo won't change it this loop */
;	/* (it's pretty safe to assume that, because we're working on 28 bits */
;	/* blocks) */
;	/* It means also that ebx == esi (Llo1 == Llo2) */

align 4
_bigger_loop_k6:

        add     ebx, S0_ROTL3                   ; 1
        rol     ebx, FIRST_ROTL                 ; 3
        mov     [work_pre1_r1], ebx             ; 1

        lea     eax, [S1_S0_ROTL3+ebx]          ; 1
        rol     eax, 3                          ; 2
        mov     [work_pre2_r1], eax             ; 1

        lea     ecx, [eax+ebx]                  ; 2
        mov     [work_pre3_r1], ecx             ; 1

align 8
_loaded_k6:
;     ------------------------------
;     Begin round 1 of key expansion
;     ------------------------------

        mov     ebx, [work_pre1_r1]
        mov     esi, ebx
        mov     eax, [work_pre2_r1]
        mov     ebp, eax

        mov     ecx, [work_pre3_r1]
        add     edx, ecx
        add     edi, ecx
        rol     edx, cl
        rol     edi, cl



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
	ROUND_1_LAST         25


;     ------------------------------
;     Begin round 2 of key expansion
;     ------------------------------

_end_round1_k6:
        add     eax, edx
        add     ebp, edi
        rol     eax, 3
        rol     ebp, 3
        mov     S1(0), eax
        mov     S2(0), ebp
	
        lea     ecx, [eax+edx]
        add     ebx, ecx
        rol     ebx, cl
        lea     ecx, [ebp+edi]
        add     esi, ecx
        rol     esi, cl

        mov     ecx, [work_pre2_r1]
        add     eax, ebx
        add     eax, ecx
        add     ebp, esi
        add     ebp, ecx
        rol     eax, 3
        rol     ebp, 3
        mov     S1(1), eax
        mov     S2(1), ebp
        lea     ecx, [eax+ebx]
        add     edx, ecx
        rol     edx, cl
        lea     ecx, [ebp+esi]
        add     eax, S1(2)
        add     edi, ecx
        add     ebp, S2(2)
        rol     edi, cl

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
	ROUND_2_LAST         25

;     Save 2nd key parameters and initialize result variable

;     I'm using the stack instead of a memory location, because
;     gcc don't allow me to put more than 10 constraints in an
;     asm() statement.
;

_end_round2_k6:
        mov     [work_key2_edi], edi

;     ----------------------------------------------------
;     Begin round 3 of key expansion mixed with encryption
;     ----------------------------------------------------
;     (first key)					

	; A  = %eax  eA = %esi
	; L0 = %ebx  eB = %edi
	; L1 = %edx  .. = %ebp

        add     eax, edx		; 1	A = ROTL3(S00 + A + L1);
        add     eax, S1(0)		; 2
        rol     eax, 3		        ; 2
        mov     esi, [work_P_0]         ; 1     eA = P_0 + A;
        add     esi, eax		; 1
        lea     ecx, [eax+edx]          ; 2	L0 = ROTL(L0 + A + L1, A + L1);
        add     ebx, ecx                ; 1
        rol     ebx, cl                 ; 3
	       			
        add     eax, ebx		; 1	A = ROTL3(S01 + A + L0);
        add     eax, S1(1)              ; 2
        rol     eax, 3                  ; 2
        mov     edi, [work_P_1]         ; 1     eB = P_1 + A;
        add     edi, eax                ; 1
        lea     ecx, [eax+ebx]          ; 2	L1 = ROTL(L1 + A + L0, A + L0);
        add     edx, ecx                ; 1
        rol     edx, cl                 ; 3     sum = 26

	ROUND_3_EVEN_AND_ODD  2,1 ; 1 == S1
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

;       early exit

_end_round3_1_k6:
        add     eax, edx                ; 1	A = ROTL3(S24 + A + L1);
        add     eax, S1(24)             ; 2
        rol     eax, 3                  ; 2
        mov     ecx, edi                ; 1	eA = ROTL(eA ^ eB, eB) + A
        xor     esi, edi                ; 1
        rol     esi, cl                 ; 3
        add     esi, eax                ; 1
					
        cmp     esi, [work_C_0]
        jne     __exit_1_k6
					
        lea     ecx, [eax+edx]          ; 2	L0 = ROTL(L0 + A + L1, A + L1);
        add     ebx, ecx                ; 1
        rol     ebx, cl                 ; 3
        add     eax, ebx                ; 1	A = ROTL3(S25 + A + L0);
        add     eax, S1(25)             ; 2
        rol     eax, 3                  ; 2
        mov     ecx, esi                ; 1	eB = ROTL(eB ^ eA, eA) + A
        xor     edi, esi                ; 1
        rol     edi, cl                 ; 3
        add     edi, eax                ; 1

        cmp     edi, [work_C_1]
        je near _full_exit_k6

__exit_1_k6:

;       Restore 2nd key parameters
        mov     edx, [work_key2_edi]
        mov     ebx, [work_key2_esi]
;        mov     eax, [work_key2_ebp]
        mov     eax, S2(25)

;     ----------------------------------------------------
;     Begin round 3 of key expansion mixed with encryption
;     ----------------------------------------------------
;     (second key)					

	; A  = %eax  eA = %esi
	; L0 = %ebx  eB = %edi
	; L1 = %edx  .. = %ebp

        add     eax, edx                ; 1     A = ROTL3(S00 + A + L1);
        add     eax, S2(0)              ; 2
        rol     eax, 3                  ; 2
        mov     esi, [work_P_0]         ; 1	eA = P_0 + A;
        add     esi, eax                ; 1
        lea     ecx, [eax+edx]          ; 2	L0 = ROTL(L0 + A + L1, A + L1);
        add     ebx, ecx                ; 1
        rol     ebx, cl                 ; 3
	       			
        add     eax, ebx                ; 1	A = ROTL3(S01 + A + L0);
        add     eax, S2(1)              ; 2
        rol     eax, 3                  ; 2
        mov     edi, [work_P_1]         ; 1	eB = P_1 + A;
        add     edi, eax                ; 1
        lea     ecx, [eax+ebx]          ; 2	L1 = ROTL(L1 + A + L0, A + L0);
        add     edx, ecx                ; 1
        rol     edx, cl                 ; 3	sum = 26 \n"
	ROUND_3_EVEN_AND_ODD  2,2 ; 2 == S2
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

;	 early exit

_end_round3_2_k6:
        add     eax, edx                ; 1     A = ROTL3(S24 + A + L1);
        add     eax, S2(24)             ; 2
        rol     eax, 3                  ; 2
        mov     ecx, edi                ; 1	eA = ROTL(eA ^ eB, eB) + A
        xor     esi, edi                ; 1
        rol     esi, cl                 ; 3
        add     esi, eax                ; 1
					
        cmp     esi, [work_C_0]
        jne     __exit_2_k6
					
        lea     ecx, [eax+edx]          ; 2	L0 = ROTL(L0 + A + L1, A + L1);
        add     ebx, ecx                ; 1
        rol     ebx, cl                 ; 3
        add     eax, ebx                ; 1	A = ROTL3(S25 + A + L0);
        add     eax, S2(25)             ; 2
        rol     eax, 3                  ; 2
        mov     ecx, esi                ; 1	eB = ROTL(eB ^ eA, eA) + A
        xor     edi, esi                ; 1
        rol     edi, cl                 ; 3
        add     edi, eax                ; 1

        cmp     edi, [work_C_1]
        jne     __exit_2_k6
        mov dword [work_add_iter], 1
        jmp     _full_exit_k6

__exit_2_k6:
        mov     edx, [work_key_hi]

; Jumps not taken are faster
        add     edx, 0x02000000
        jc near _next_inc_k6

align 4
_next_iter_k6:
        mov     [work_key_hi], edx
        lea     edi, [edx+0x01000000]
        dec dword [work_iterations]
        jg near _loaded_k6
        mov     eax, [RC5UnitWork]                      ; pointer to rc5unitwork
        mov     ebx, [work_key_lo]
        mov     [RC5UnitWork_L0lo], ebx                 ; Update real data
        mov     [RC5UnitWork_L0hi], edx                 ; (used by caller)
        jmp     _full_exit_k6

align 4
_next_iter2_k6:
        mov     [work_key_lo], ebx
        mov     [work_key_hi], edx
        lea     edi, [0x01000000+edx]
        mov     esi, ebx
        dec dword [work_iterations]
        jg near _bigger_loop_k6
        mov     eax, [RC5UnitWork]                      ; pointer to rc5unitwork
        mov     [RC5UnitWork_L0lo], ebx	                ; Update real data
        mov     [RC5UnitWork_L0hi], edx	                ; (used by caller)
        jmp     _full_exit_k6

_next_inc_k6:
        add     edx, 0x00010000
        test    edx, 0x00FF0000
        jnz near _next_iter_k6

        add     edx, 0xFF000100
        test    edx, 0x0000FF00
        jnz near _next_iter_k6

        add     edx, 0xFFFF0001
        test    edx, 0x000000FF
        jnz near _next_iter_k6


        mov     ebx, [work_key_lo]

        sub     edx, 0x00000100
        add     ebx, 0x01000000
        jnc near _next_iter2_k6

        add     ebx, 0x00010000
        test    ebx, 0x00FF0000
        jnz near _next_iter2_k6

        add     ebx, 0xFF000100
        test    ebx, 0x0000FF00
        jnz near _next_iter2_k6

        add     ebx, 0xFFFF0001
        test    ebx, 0x000000FF
        jnz near _next_iter2_k6

	; Moo !
	; We have just finished checking the last key
	; of the rc5-64 keyspace...
	; Not much to do here, since we have finished the block ...


_full_exit_k6:
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
