; PentiumIV optimized version
;

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
[SECTION _TEXT USE32 ALIGN=16]
%else
[SECTION .text]
%endif


[GLOBAL _rc5_unit_func_p7]
[GLOBAL rc5_unit_func_p7]

%define work_size       532

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

%define save_ebp        esp+0
%define save_edi        esp+4
%define save_esi        esp+8
%define save_ebx        esp+12
%define work_add_iter   esp+16
%define work_s1         esp+4+16
%define work_s2         esp+108+16
%define work_s3         esp+212+16
%define work_P_0        esp+316+16
%define work_P_1        esp+320+16
%define work_C_0        esp+324+16
%define work_C_1        esp+328+16
%define work_key_hi     esp+332+16
%define work_key_lo     esp+336+16
%define work_iterations esp+340+16
%define work_pre1_r1    esp+344+16
%define work_pre2_r1    esp+348+16
%define work_pre3_r1    esp+352+16
%define work_Lhi1       esp+356+16
%define work_Lhi2       esp+360+16
%define work_Lhi3       esp+364+16
%define work_Llo1       esp+368+16
%define work_Llo2       esp+372+16
%define work_Llo3       esp+376+16

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
%define S3(N)    [((N)*4)+work_s3]

; eax = A1     ; esi = hi1
; ebx = A2     ; ebp = hi2
; edx = A3     ; edi = hi3
        
; S1(N) = A1 = ROTL3 (A1 + Lhi1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Lhi2 + S_not(N));
; S3(N) = A3 = ROTL3 (A3 + Lhi3 + S_not(N));
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
; Llo3 = ROTL (Llo3 + A3 + Lhi3, A3 + Lhi1);
%macro ROUND_1_EVEN 1
        lea     eax, [S_not(%1)+eax+esi]
        lea     ebx, [S_not(%1)+ebx+ebp]
        lea     edx, [S_not(%1)+edx+edi]
        rol     eax, 3
        rol     ebx, 3
        rol     edx, 3
        mov     [work_Lhi1], esi
        mov     [work_Lhi2], ebp
        mov     [work_Lhi3], edi
        mov     S1(%1), eax
        add     esi, eax
        mov     S2(%1), ebx
        add     ebp, ebx
        mov     S3(%1), edx
        add     edi, edx
        mov     ecx, esi
        add     esi, [work_Llo1]
        rol     esi, cl
        mov     ecx, ebp
        add     ebp, [work_Llo2]
        rol     ebp, cl
        mov     ecx, edi
        add     edi, [work_Llo3]
        rol     edi, cl
%endmacro

%macro ROUND_1_ODD 1
        lea     eax, [S_not(%1)+eax+esi]
        lea     ebx, [S_not(%1)+ebx+ebp]
        lea     edx, [S_not(%1)+edx+edi]
        rol     eax, 3
        rol     ebx, 3
        rol     edx, 3
        mov     [work_Llo1], esi
        mov     [work_Llo2], ebp
        mov     [work_Llo3], edi
        mov     S1(%1), eax
        add     esi, eax
        mov     S2(%1), ebx
        add     ebp, ebx
        mov     S3(%1), edx
        add     edi, edx
        mov     ecx, esi
        add     esi, [work_Lhi1]
        rol     esi, cl
        mov     ecx, ebp
        add     ebp, [work_Lhi2]
        rol     ebp, cl
        mov     ecx, edi
        add     edi, [work_Lhi3]
        rol     edi, cl
%endmacro


%macro ROUND_1_ODD_AND_EVEN 2
  ROUND_1_ODD %1
  ROUND_1_EVEN %2
%endmacro

; ------------------------------------------------------------------
; S1N = A1 = ROTL3 (A1 + Lhi1 + S1N);
; S2N = A2 = ROTL3 (A2 + Lhi2 + S2N);
; S3N = A3 = ROTL3 (A3 + Lhi3 + S3N);
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
; Llo3 = ROTL (Llo3 + A3 + Lhi3, A3 + Lhi3);

%macro ROUND_2_EVEN 1
        add     eax, esi
        add     ebx, ebp
        add     edx, edi
        rol     eax, 3
        rol     ebx, 3
        rol     edx, 3
        mov     [work_Lhi1], esi
        mov     [work_Lhi2], ebp
        mov     [work_Lhi3], edi
        mov     S1(%1), eax
        add     esi, eax
        mov     S2(%1), ebx
        add     ebp, ebx
        mov     S3(%1), edx
        add     edi, edx
        mov     ecx, esi
        add     esi, [work_Llo1]
        rol     esi, cl
        add     eax, S1(%1+1)
        mov     ecx, ebp
        add     ebp, [work_Llo2]
        rol     ebp, cl
        add     ebx, S2(%1+1)
        mov     ecx, edi
        add     edi, [work_Llo3]
        rol     edi, cl
        add     edx, S3(%1+1)
%endmacro

%macro ROUND_2_ODD 1
        add     eax, esi
        add     ebx, ebp
        add     edx, edi
        rol     eax, 3
        rol     ebx, 3
        rol     edx, 3
        mov     [work_Llo1], esi
        mov     [work_Llo2], ebp
        mov     [work_Llo3], edi
        mov     S1(%1), eax
        add     esi, eax
        mov     S2(%1), ebx
        add     ebp, ebx
        mov     S3(%1), edx
        add     edi, edx
        mov     ecx, esi
        add     esi, [work_Lhi1]
        rol     esi, cl
        add     eax, S1(%1+1)
        mov     ecx, ebp
        add     ebp, [work_Lhi2]
        rol     ebp, cl
        add     ebx, S2(%1+1)
        mov     ecx, edi
        add     edi, [work_Lhi3]
        rol     edi, cl
        add     edx, S3(%1+1)
%endmacro

%macro ROUND_2_LAST 1
        add     eax, esi
        add     ebx, ebp
        add     edx, edi
        rol     eax, 3
        rol     ebx, 3
        rol     edx, 3
        mov     [work_Llo1], esi
        mov     [work_Llo2], ebp
        mov     [work_Llo3], edi
        mov     S1(%1), eax
        add     esi, eax
        mov     S2(%1), ebx
        add     ebp, ebx
        mov     S3(%1), edx
        add     edi, edx
        mov     ecx, esi
        add     esi, [work_Lhi1]
        rol     esi, cl
        add     eax, S1(0)
        mov     ecx, ebp
        add     ebp, [work_Lhi2]
        rol     ebp, cl
        add     ebx, S2(0)
        mov     ecx, edi
        add     edi, [work_Lhi3]
        rol     edi, cl
        add     edx, S3(0)
%endmacro

%macro ROUND_2_ODD_AND_EVEN 2
  ROUND_2_ODD %1
  ROUND_2_EVEN %2
%endmacro
; ------------------------------------------------------------------
; A = ROTL3 (A + Lhi + S(N));
; Llo = ROTL (Llo + A + Lhi, A + Lhi);
; A = ROTL3 (A + Llo + S(N));
; Lhi = ROTL (Lhi + A + Llo, A + Llo);

; eA = ROTL (eA ^ eB, eB) + A;
; eB = ROTL (eA ^ eB, eA) + A;

; eax =  a1  esi = hi1
; ebx =  a2  ebp = hi2
; edx =  a3  edi = hi3

; eax = eA1  esi = eB1
; ebx = eA2  ebp = eB2
; edx = eA3  edi = eB3

%macro ROUND_3A_EVEN 1
        add     eax, esi
        add     ebx, ebp
        add     edx, edi
        rol     eax, 3
        rol     ebx, 3
        rol     edx, 3
        mov     [work_Lhi1], esi
        mov     [work_Lhi2], ebp
        mov     [work_Lhi3], edi
        mov     S1(%1), eax
        add     esi, eax
        mov     S2(%1), ebx
        add     ebp, ebx
        mov     S3(%1), edx
        add     edi, edx
        mov     ecx, esi
        add     esi, [work_Llo1]
        rol     esi, cl
        add     eax, S1(%1+1)
        mov     ecx, ebp
        add     ebp, [work_Llo2]
        rol     ebp, cl
        add     ebx, S2(%1+1)
        mov     ecx, edi
        add     edi, [work_Llo3]
        rol     edi, cl
        add     edx, S3(%1+1)
%endmacro

%macro ROUND_3A_ODD 1
        add     eax, esi
        add     ebx, ebp
        add     edx, edi
        rol     eax, 3
        rol     ebx, 3
        rol     edx, 3
        mov     [work_Llo1], esi
        mov     [work_Llo2], ebp
        mov     [work_Llo3], edi
        mov     S1(%1), eax
        add     esi, eax
        mov     S2(%1), ebx
        add     ebp, ebx
        mov     S3(%1), edx
        add     edi, edx
        mov     ecx, esi
        add     esi, [work_Lhi1]
        rol     esi, cl
        add     eax, S1(%1+1)
        mov     ecx, ebp
        add     ebp, [work_Lhi2]
        rol     ebp, cl
        add     ebx, S2(%1+1)
        mov     ecx, edi
        add     edi, [work_Lhi3]
        rol     edi, cl
        add     edx, S3(%1+1)
%endmacro

%macro ROUND_3B_EVEN 1
        xor     eax, esi
        xor     ebx, ebp
        xor     edx, edi
        mov     ecx, esi
        rol     eax, cl
        mov     ecx, ebp
        rol     ebx, cl
        mov     ecx, edi
        rol     edx, cl
        add     eax, S1(%1)
        add     ebx, S2(%1)
        add     edx, S3(%1)
%endmacro
        
%macro ROUND_3B_ODD 1
        xor     esi, eax
        xor     ebp, ebx
        xor     edi, edx
        mov     ecx, eax
        rol     esi, cl
        mov     ecx, ebx
        rol     ebp, cl
        mov     ecx, edx
        rol     edi, cl
        add     esi, S1(%1)
        add     ebp, S2(%1)
        add     edi, S3(%1)
%endmacro

%macro ROUND_3A_EVEN_AND_ODD 1
    ROUND_3A_EVEN %1
    ROUND_3A_ODD  %1+1
%endmacro

%macro ROUND_3B_EVEN_AND_ODD 1
    ROUND_3B_EVEN %1
    ROUND_3B_ODD  %1+1
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
_rc5_unit_func_p7:
rc5_unit_func_p7:
;u32 rc5_unit_func_p7( RC5UnitWork * rc5unitwork, u32 timeslice )

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

;   load parameters
        mov     esi, [RC5UnitWork_L0lo]           ; esi = l0 = Llo1
        mov     edx, [RC5UnitWork_L0hi]           
        add     edx, 0x02000000
        mov     [work_Lhi3], edx
        sub     edx, 0x01000000
        mov     [work_Lhi2], edx
        sub     edx, 0x01000000
        mov     [work_Lhi1], edx
               
        mov     [work_key_lo], esi
        mov     [work_key_hi], edx

  ; Save other parameters
  ; (it's faster to do so, since we will only load 1 value
  ; each time in RC5_ROUND_3xy, instead of two if we save
  ; only the pointer to the RC5 struct)

        mov     ebx, [RC5UnitWork_plainlo]
        mov     [work_P_0], ebx
        mov     ebx, [RC5UnitWork_plainhi]
        mov     [work_P_1], ebx
        mov     ebx, [RC5UnitWork_cipherlo]
        mov     [work_C_0], ebx
        mov     ebx, [RC5UnitWork_cipherhi]
        mov     [work_C_1], ebx
    
    ; status check:
    ; eax, ebx, and ecx are currently free.

	; Pre-calculate things. Assume work.key_lo won't change it this loop */
	; (it's pretty safe to assume that, because we're working on 28 bits */
	; blocks) */
	; It means also that %%esi == %%edi (Llo1 == Llo2) */

align 4
bigger_loop_p7:
        add     esi, S0_ROTL3
        rol     esi, FIRST_ROTL
        mov     [work_pre1_r1], esi     ; Llo1 = ROTL(Llo1 + A1, A1)

        lea     eax, [S1_S0_ROTL3+esi]
        rol     eax, 3                  ; A1 = ROTL3(A1)
        mov     [work_pre2_r1], eax

        lea     ecx, [eax+esi]          ; tmp1 = A1 + Llo1
        mov     [work_pre3_r1], ecx

align 4
_loaded_p7:
    ; ------------------------------
    ; Begin round 1 of key expansion
    ; ------------------------------

        mov	  	edi, [work_pre1_r1]
        mov		  [work_Llo1], edi
        mov     [work_Llo2], edi
        mov		  [work_Llo3], edi

        mov     esi, [work_Lhi1]
        mov     ebp, [work_Lhi2]
        mov     edi, [work_Lhi3]
        
        mov     ecx, [work_pre3_r1]
        add     esi, ecx
        rol     esi, cl
        add     ebp, ecx
        rol     ebp, cl
        add     edi, ecx
        rol     edi, cl
        
        mov     eax, [work_pre2_r1]
        mov     ebx, eax
        mov     edx, eax

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
	ROUND_1_ODD             25


    ; ------------------------------
    ; Begin round 2 of key expansion
    ; ------------------------------

align 4
_end_round1_p7:

        lea     eax, [S0_ROTL3+eax+esi]
        lea     ebx, [S0_ROTL3+ebx+ebp]
        lea     edx, [S0_ROTL3+edx+edi]
        rol     eax, 3
        rol     ebx, 3
        rol     edx, 3
        mov     [work_Lhi1], esi
        mov     [work_Lhi2], ebp
        mov     [work_Lhi3], edi
        mov     S1(0), eax
        add     esi, eax
        mov     S2(0), ebx
        add     ebp, ebx
        mov     S3(0), edx
        add     edi, edx
        mov     ecx, esi
        add     esi, [work_Llo1]
        rol     esi, cl
        add     eax, [work_pre2_r1]
        mov     ecx, ebp
        add     ebp, [work_Llo2]
        rol     ebp, cl
        add     ebx, [work_pre2_r1]
        mov     ecx, edi
        add     edi, [work_Llo3]
        rol     edi, cl
        add     edx, [work_pre2_r1]

	ROUND_2_ODD_AND_EVEN  1, 2
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
	ROUND_2_LAST 25

;       Save 2nd key parameters and initialize result variable


align 4
_end_round2_p7:

; eax = A1  esi = hi1
; ebx = A2  ebp = hi2
; edx = A3  edi = hi3

    ; ----------------------------------------------------
    ; Begin round 3 of key expansion mixed with encryption
    ; ----------------------------------------------------

	ROUND_3A_EVEN_AND_ODD  0
	ROUND_3A_EVEN_AND_ODD  2
	ROUND_3A_EVEN_AND_ODD  4
	ROUND_3A_EVEN_AND_ODD  6
	ROUND_3A_EVEN_AND_ODD  8
	ROUND_3A_EVEN_AND_ODD 10
	ROUND_3A_EVEN_AND_ODD 12
	ROUND_3A_EVEN_AND_ODD 14
	ROUND_3A_EVEN_AND_ODD 16
	ROUND_3A_EVEN_AND_ODD 18
	ROUND_3A_EVEN_AND_ODD 20
	ROUND_3A_EVEN_AND_ODD 22
	ROUND_3A_EVEN         24

        mov     [work_Llo1], esi
        mov     [work_Llo2], ebp
        mov     [work_Llo3], edi
        
        mov     eax, [work_P_0]
        mov     esi, [work_P_1]
        mov     ebx, eax
        mov     edx, eax
        mov     ebp, esi
        mov     edi, esi
        add     eax, S1(0)
        add     ebx, S2(0)
        add     edx, S3(0)    
        add     esi, S1(1)
        add     ebp, S2(1)
        add     edi, S3(1)

	ROUND_3B_EVEN_AND_ODD  2
	ROUND_3B_EVEN_AND_ODD  4
	ROUND_3B_EVEN_AND_ODD  6
	ROUND_3B_EVEN_AND_ODD  8
	ROUND_3B_EVEN_AND_ODD 10
	ROUND_3B_EVEN_AND_ODD 12
	ROUND_3B_EVEN_AND_ODD 14
	ROUND_3B_EVEN_AND_ODD 16
	ROUND_3B_EVEN_AND_ODD 18
	ROUND_3B_EVEN_AND_ODD 20
	ROUND_3B_EVEN_AND_ODD 22
  ROUND_3B_EVEN         24

  ; early exit
align 4
_end_round3_p7:

; eax = eA1  esi = eB1
; ebx = eA2  ebp = eB2
; edx = eA3  edi = eB3

        cmp     eax, [work_C_0]
        jne     __exit_1_p7
        					
        mov     ecx, eax
        mov     eax, S1(24)
        add     eax, S1(25)
        add     eax, [work_Llo1]
        rol     eax, 3
        xor     esi, ecx
        rol     esi, cl
        add     esi, eax
        
        cmp     esi, [work_C_1]
        je near _full_exit_p7

 align 4
 __exit_1_p7:
 
        cmp     ebx, [work_C_0]
        jne     __exit_2_p7
        
        mov     ecx, ebx
        mov     ebx, S2(24)
        add     ebx, S2(25)
        add     ebx, [work_Llo2]
        rol     ebx, 3
        xor     ebp, ecx
        rol     ebp, cl
        add     ebp, ebx
        
        cmp     ebp, [work_C_1]
        jne     __exit_2_p7
        mov     dword [work_add_iter], 1
        jmp     _full_exit_p7

 align 4
 __exit_2_p7:
 
        cmp     edx, [work_C_0]
        jne     __exit_3_p7
        
        mov     ecx, edx
        mov     edx, S3(24)
        add     edx, S3(25)
        add     edx, [work_Llo3]
        rol     edx, 3
        xor     edi, ecx
        rol     edi, cl
        add     edi, edx
        
        cmp     edi, [work_C_1]
        jne     __exit_3_p7
        mov     dword [work_add_iter], 2
        jmp     _full_exit_p7

align 4
__exit_3_p7:
        mov     edx, [work_key_hi]

; Jumps not taken are faster
        add     edx, 0x03000000
        jc near _next_inc_p7

align 4
_next_iter_p7:
        mov     [work_key_hi], edx
        add     edx, 0x02000000
        mov     [work_Lhi3], edx
        sub     edx, 0x01000000
        mov     [work_Lhi2], edx
        sub     edx, 0x01000000
        mov     [work_Lhi1], edx
        dec     dword [work_iterations]
        jg near _loaded_p7
        mov     eax, [RC5UnitWork]                      ; pointer to rc5unitwork
        mov     ebx, [work_key_lo]
        mov     [RC5UnitWork_L0lo], ebx                 ; Update real data
        mov     [RC5UnitWork_L0hi], edx                 ; (used by caller)
        jmp     _full_exit_p7

align 4
_next_iter2_p7:
        mov     [work_key_lo], ebx
        mov     [work_key_hi], edx
        add     edx, 0x02000000
        mov     [work_Lhi3], edx
        sub     edx, 0x01000000
        mov     [work_Lhi2], edx
        sub     edx, 0x01000000
        mov     [work_Lhi1], edx
        mov     esi, ebx
        dec     dword [work_iterations]
        jg near bigger_loop_p7
        mov     eax, [RC5UnitWork]                      ; pointer to rc5unitwork
        mov     [RC5UnitWork_L0lo], ebx                 ; Update real data
        mov     [RC5UnitWork_L0hi], edx                 ; (used by caller)
        jmp     _full_exit_p7

align 4
_next_inc_p7:
        add     edx, 0x00010000
        test    edx, 0x00FF0000
        jnz near _next_iter_p7

        add     edx, 0xFF000100
        test    edx, 0x0000FF00
        jnz near _next_iter_p7

        add     edx, 0xFFFF0001
        test    edx, 0x000000FF
        jnz near _next_iter_p7


        mov     ebx, [work_key_lo]

        sub     edx, 0x00000100
        add     ebx, 0x01000000
        jnc near _next_iter2_p7

        add     ebx, 0x00010000
        test    ebx, 0x00FF0000
        jnz near _next_iter2_p7

        add     ebx, 0xFF000100
        test    ebx, 0x0000FF00
        jnz near _next_iter2_p7

        add     ebx, 0xFFFF0001
        test    ebx, 0x000000FF
        jnz near _next_iter2_p7

	; Moo !
	; We have just finished checking the last key
	; of the rc5-64 keyspace...
	; Not much to do here, since we have finished the block ...


align 4
_full_exit_p7:
mov ebp, [timeslice]
sub ebp, [work_iterations]
mov eax, [work_add_iter]
lea edx, [eax+ebp*2]
add edx, ebp
mov eax, edx

;    return (timeslice - work.iterations) * 4 + work.add_iter;


      mov ebx, [save_ebx]
      mov esi, [save_esi]
      mov edi, [save_edi]
      mov ebp, [save_ebp]

     add esp, work_size ; restore stack pointer

     ret


