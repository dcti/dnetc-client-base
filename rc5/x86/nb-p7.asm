; PentiumIV optimized version
;

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
[SECTION _TEXT USE32 ALIGN=16]
%else
[SECTION .text]
%endif


[GLOBAL _rc5_unit_func_p7]
[GLOBAL rc5_unit_func_p7]

%define work_size       524

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
%define work_s4         esp+316+16
%define work_P_0        esp+420+16
%define work_P_1        esp+424+16
%define work_C_0        esp+428+16
%define work_C_1        esp+432+16
%define work_key_hi     esp+436+16
%define work_key_lo     esp+440+16
%define work_iterations esp+444+16
%define work_pre1_r1    esp+448+16
%define work_pre2_r1    esp+452+16
%define work_pre3_r1    esp+456+16
%define work_Lhi1       esp+460+16
%define work_Lhi2       esp+464+16
%define work_Lhi3       esp+468+16
%define work_Lhi4       esp+472+16
%define work_Llo1       esp+476+16
%define work_Llo2       esp+480+16
%define work_Llo3       esp+484+16
%define work_Llo4       esp+488+16
%define work_A1         esp+492+16
%define work_A2         esp+496+16
%define work_A3         esp+500+16
%define work_A4         esp+504+16


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
%define S4(N)    [((N)*4)+work_s4]

;        use1  use2        use1  use2
;        -----|-----|      -----|-----|
;   eax  A1           ebx  A2    tmp
;   ecx  A3    Lhi1   edx  A4    Lhi2  
;   esi  Llo1  Lhi3   edi  Llo3  Lhi4
;   ebp  Llo4  Llo2      

; S1(N) = A1 = ROTL3 (A1 + Lhi1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Lhi2 + S_not(N));
; S3(N) = A3 = ROTL3 (A3 + Lhi3 + S_not(N));
; S4(N) = A4 = ROTL3 (A4 + Lhi4 + S_not(N));
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
; Llo3 = ROTL (Llo3 + A3 + Lhi3, A3 + Lhi1);
; Llo4 = ROTL (Llo4 + A4 + Lhi4, A4 + Lhi2);
%macro ROUND_1_EVEN 1
        mov     ecx, S3(%1-1)
        lea     ebx, [S_not(%1)+ebx+edx]    ; A2 = (A2 + Lhi2 + S_not(N));
        mov     edx, S4(%1-1)
        lea     ecx, [S_not(%1)+ecx+esi]    ; A3 = (A3 + Lhi3 + S_not(N));
        lea     edx, [S_not(%1)+edx+edi]    ; A4 = (A4 + Lhi4 + S_not(N));
        rol     edx, 3
        rol     ecx, 3
        rol     ebx, 3
        rol     eax, 3
        mov     ebp, [work_Llo4]
        mov     S4(%1), edx
        lea     ecx, [edx+edi]
        add     ebp, ecx
        rol     ebp, cl                     ; Llo4 = ROTL (Llo4 + A4 + Lhi4, A4 + Lhi4);
        mov     [work_Lhi4], edi
        mov     edi, [work_Llo3]
        mov     S3(%1), ecx
        add     ecx, esi
        add     edi, ecx
        rol     edi, cl                     ; Llo3 = ROTL (Llo3 + A3 + Lhi3, A3 + Lhi3);
        mov     [work_Lhi3], esi
        mov     ecx, [work_Lhi1]
        mov     esi, [work_Llo1]
        mov     S1(%1), eax
        add     ecx, eax
        add     esi, ecx
        rol     esi, cl                     ; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
        mov     edx, [work_Lhi2]
        mov     [work_Llo4],ebp
        mov     ebp, [work_Llo2]
        mov     S2(%1), ebx
        lea     ecx, [ebx+edx]
        add     ebp, ecx
        rol     ebp, cl                     ; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);                                
        mov     ecx, S3(%1)
        lea     ecx, [S_not(%1+1)+ecx+edi]
        mov     eax, S1(%1)
        mov     ebx, S2(%1)
%endmacro

; S1(N) = A1 = ROTL3 (A1 + Llo1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Llo2 + S_not(N));
; S3(N) = A3 = ROTL3 (A3 + Llo3 + S_not(N));
; S4(N) = A4 = ROTL3 (A4 + Llo4 + S_not(N));
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
; Lhi3 = ROTL (Lhi3 + A3 + Llo3, A3 + Llo3);
; Lhi4 = ROTL (Lhi4 + A4 + Llo4, A4 + Llo4);

;        use1  use2        use1  use2
;        -----|-----|      -----|-----|
;   eax  A1           ebx  A2    tmp
;   ecx  A3    Lhi1   edx  A4    Lhi2  
;   esi  Llo1  Lhi3   edi  Llo3  Lhi4
;   ebp  Llo4  Llo2      

%macro ROUND_1_ODD 1
        rol     ecx, 3                      ; A3 = ROTL3 (A3 + Llo3 + S_not(N));
        lea     eax, [S_not(%1)+eax+esi]
        mov     edx, S4(%1-1)
        rol     eax, 3                      ; A1 = ROTL3 (A1 + Llo1 + S_not(N));
        lea     ebx, [S_not(%1)+ebx+ebp]
        mov     [work_Llo2], ebp
        mov     ebp, [work_Llo4]
        rol     ebx, 3                      ; A2 = ROTL3 (A2 + Llo2 + S_not(N));
        lea     edx, [S_not(%1)+edx+ebp]
        rol     edx, 3                      ; A4 = ROTL3 (A4 + Llo4 + S_not(N));
        mov     S1(%1), eax
        mov     S2(%1), ebx
        mov     S3(%1), ecx
        mov     S4(%1), edx
        lea     ecx, [eax+esi]
        mov     eax, [work_Lhi1]
        add     eax, ecx
        rol     eax, cl                     ; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
        lea     ecx, [edx+ebp]
        mov     edx, [work_Lhi4]
        add     edx, ecx
        rol     edx, cl                     ; Lhi4 = ROTL (Lhi4 + A4 + Llo4, A4 + Llo4);
        mov     ebp, [work_Llo2]
        lea     ecx, [ebx+ebp]
        mov     ebx, [work_Lhi2]
        add     ebx, ecx
        rol     ebx, cl                     ; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
        mov     [work_Llo1], esi
        mov     esi, S3(%1)
        lea     ecx, [esi+edi]
        mov     esi, [work_Lhi3]
        add     esi, ecx
        rol     esi, cl                     ; Lhi3 = ROTL (Lhi3 + A3 + Llo3, A3 + Llo3);
        mov     ecx, S1(%1)
        mov     [work_Lhi1], eax
        lea     eax, [S_not(%1+1)+ecx+eax]
        mov     [work_Llo3], edi
        mov     edi, edx
        mov     edx, ebx
        mov     [work_Lhi2], ebx
        mov     ebx, S2(%1)
%endmacro

%macro ROUND_1_LAST 1
        rol     ecx, 3                      ; A3 = ROTL3 (A3 + Llo3 + S_not(N));
        lea     eax, [S_not(%1)+eax+esi]
        mov     edx, S4(%1-1)
        rol     eax, 3                      ; A1 = ROTL3 (A1 + Llo1 + S_not(N));
        lea     ebx, [S_not(%1)+ebx+ebp]
        mov     [work_Llo2], ebp
        mov     ebp, [work_Llo4]
        rol     ebx, 3                      ; A2 = ROTL3 (A2 + Llo2 + S_not(N));
        lea     edx, [S_not(%1)+edx+ebp]
        rol     edx, 3                      ; A4 = ROTL3 (A4 + Llo4 + S_not(N));
        mov     S1(%1), eax
        mov     S2(%1), ebx
        mov     S3(%1), ecx
        mov     S4(%1), edx
        lea     ecx, [eax+esi]
        mov     eax, [work_Lhi1]
        add     eax, ecx
        rol     eax, cl                     ; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
        lea     ecx, [edx+ebp]
        mov     edx, [work_Lhi4]
        add     edx, ecx
        rol     edx, cl                     ; Lhi4 = ROTL (Lhi4 + A4 + Llo4, A4 + Llo4);
        mov     ebp, [work_Llo2]
        lea     ecx, [ebx+ebp]
        mov     ebx, [work_Lhi2]
        add     ebx, ecx
        rol     ebx, cl                     ; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
        mov     [work_Llo1], esi
        mov     esi, S3(%1)
        lea     ecx, [esi+edi]
        mov     esi, [work_Lhi3]
        add     esi, ecx
        rol     esi, cl                     ; Lhi3 = ROTL (Lhi3 + A3 + Llo3, A3 + Llo3);
        mov     ecx, S1(%1)
        mov     [work_Lhi1], eax
        lea     eax, [S0_ROTL3+ecx+eax]
        mov     [work_Llo3], edi
        mov     edi, edx
        mov     edx, ebx
        mov     [work_Lhi2], ebx
        mov     ebx, S2(%1)
%endmacro

%macro ROUND_1_ODD_AND_EVEN 2
  ROUND_1_ODD %1
  ROUND_1_EVEN %2
%endmacro

; ------------------------------------------------------------------
; S1N = A1 = ROTL3 (A1 + Lhi1 + S1N);
; S2N = A2 = ROTL3 (A2 + Lhi2 + S2N);
; S3N = A3 = ROTL3 (A3 + Lhi3 + S3N);
; S4N = A4 = ROTL3 (A4 + Lhi4 + S4N);
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
; Llo3 = ROTL (Llo3 + A3 + Lhi3, A3 + Lhi3);
; Llo4 = ROTL (Llo4 + A4 + Lhi4, A4 + Lhi4);

%macro ROUND_2_EVEN 1
        add     eax, [work_Lhi1]
        rol     eax, 3
        add     ebx, [work_Lhi2]
        rol     ebx, 3
        add     ecx, esi
        rol     ecx, 3
        add     edx, edi
        rol     edx, 3
        mov     S1(%1), eax
        mov     S2(%1), ebx
        mov     S3(%1), ecx
        mov     S4(%1), edx
        mov     ebx, ecx
        lea     ecx, [edx+edi]
        add     edx, S4(%1+1)
        add     ebp, ecx
        rol     ebp, cl                     ; Llo4 = ROTL (Llo4 + A4 + Lhi4, A4 + Lhi4);
        mov     [work_Lhi4], edi
        mov     edi, [work_Llo3]        
        lea     ecx, [ebx+esi]
        add     ebx, S3(%1+1)
        add     edi, ecx
        rol     edi, cl                     ; Llo3 = ROTL (Llo3 + A3 + Lhi3, A3 + Lhi3);  
        mov     [work_A3], ebx
        mov     ecx, S2(%1)
        mov     ebx, ecx
        add     ecx, [work_Lhi2]
        mov     [work_Llo4], ebp
        mov     ebp, [work_Llo2]
        add     ebx, S2(%1+1)
        add     ebp, ecx
        rol     ebp, cl                     ; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
        mov     [work_Lhi3], esi
        mov     esi, [work_Llo1]
        mov     [work_A4], edx
        mov     edx, [work_Lhi1]
        lea     ecx, [eax+edx] 
        add     eax, S1(%1+1)
        add     esi, ecx
        rol     esi, cl                     ; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
%endmacro

; S1N = A1 = ROTL3 (A1 + Llo1 + S1N);
; S2N = A2 = ROTL3 (A2 + Llo2 + S2N);
; S3N = A3 = ROTL3 (A3 + Llo3 + S3N);
; S4N = A4 = ROTL3 (A4 + Llo4 + S4N);
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
; Lhi3 = ROTL (Lhi3 + A3 + Llo3, A3 + Llo3);
; Lhi4 = ROTL (Lhi4 + A4 + Llo4, A4 + Llo4);
;        use1  use2        use1  use2
;        -----|-----|      -----|-----|
;   eax  A1           ebx  A2    tmp
;   ecx  A3    Lhi1   edx  A4    Lhi2  
;   esi  Llo1  Lhi3   edi  Llo3  Lhi4
;   ebp  Llo4  Llo2      
%macro ROUND_2_ODD 1
        mov     ecx, [work_A3]
        mov     edx, [work_A4]
        add     ebx, ebp
        rol     ebx, 3
        add     ecx, edi
        rol     ecx, 3
        add     edx, [work_Llo4]
        rol     edx, 3
        add     eax, esi
        rol     eax, 3
        mov     S2(%1), ebx
        mov     S3(%1), ecx
        mov     S4(%1), edx
        mov     S1(%1), eax
        lea     ecx, [eax+esi]
        add     eax, S1(%1+1)
        mov     [work_A1], eax
        add     eax, [work_Lhi1]
        rol     eax, cl                 ; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
        mov     edx, [work_Lhi2]
        lea     ecx, [ebx+ebp]
        add     ebx, S2(%1+1)
        add     edx, ecx
        rol     edx, cl                 ; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
        mov     [work_Llo1], esi
        mov     [work_Lhi1], eax
        mov     esi, [work_Lhi3]
        mov     eax, S3(%1)
        lea     ecx, [eax+edi]
        add     eax, S3(%1+1)
        add     esi, ecx
        rol     esi, cl                 ; Lhi3 = ROTL (Lhi3 + A3 + Llo3, A3 + Llo3);
        mov     [work_Llo2], ebp
        mov     ebp, [work_Llo4]
        mov     [work_Llo3], edi
        mov     edi, [work_Lhi4]
        mov     [work_Lhi2], edx
        mov     edx, S4(%1)
        lea     ecx, [edx+ebp]
        add     edx, S4(%1+1)
        add     edi, ecx
        rol     edi, cl                 ; Lhi4 = ROTL (Lhi4 + A4 + Llo4, A4 + Llo4);
        mov     ecx, eax
        mov     eax, [work_A1]                  
%endmacro

%macro ROUND_2_LAST 1
        mov     ecx, [work_A3]
        add     ebx, ebp
        rol     ebx, 3
        add     ecx, edi
        rol     ecx, 3
        add     edx, [work_Llo4]
        rol     edx, 3
        add     eax, esi
        rol     eax, 3
        mov     S2(%1), ebx
        mov     S3(%1), ecx
        mov     S4(%1), edx
        mov     S1(%1), eax
        lea     ecx, [eax+esi]
        add     eax, S1(0)
        mov     [work_A1], eax
        add     eax, [work_Lhi1]
        rol     eax, cl                 ; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
        mov     edx, [work_Lhi2]
        lea     ecx, [ebx+ebp]
        add     edx, ecx
        rol     edx, cl                 ; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
        mov     [work_Llo1], esi
        mov     [work_Lhi1], eax
        mov     esi, [work_Lhi3]
        mov     eax, S3(%1)
        lea     ecx, [eax+edi]
        add     esi, ecx
        rol     esi, cl                 ; Lhi3 = ROTL (Lhi3 + A3 + Llo3, A3 + Llo3);
        mov     [work_Llo2], ebp
        mov     ebp, [work_Llo4]
        mov     [work_Llo3], edi
        mov     edi, [work_Lhi4]
        mov     [work_Lhi2], edx
        mov     edx, S4(%1)
        lea     ecx, [edx+ebp]
        add     edi, ecx
        rol     edi, cl                 ; Lhi4 = ROTL (Lhi4 + A4 + Llo4, A4 + Llo4);
        mov     ecx, eax
        mov     eax, [work_A1]                  
%endmacro

; S1N = A1 = ROTL3 (A1 + Llo1 + S1N);
; S2N = A2 = ROTL3 (A2 + Llo2 + S2N);
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);

%macro ROUND_2_ODD_AND_EVEN 2
  ROUND_2_ODD %1
  ROUND_2_EVEN %2
%endmacro
; ------------------------------------------------------------------
; A = ROTL3 (A + Lhi + S(N));
; Llo = ROTL (Llo + A + Lhi, A + Lhi);
; eA = ROTL (eA ^ eB, eB) + A;

; A = ROTL3 (A + Llo + S(N));
; Lhi = ROTL (Lhi + A + Llo, A + Llo);
; eB = ROTL (eA ^ eB, eA) + A;

; A  = %eax  eA = %esi
; L0 = %ebx  eB = %edi
; L1 = %edx  .. = %ebp

%define Sx(N,M) [work_s1+((N)*4)+(M-1)*104]

%macro ROUND_3_EVEN_AND_ODD 2
        add     eax, Sx(%1,%2)
        add     eax, edx
        mov     ecx, edi
        rol     eax, 3
        xor     esi, edi
        rol     esi, cl
        lea     ecx, [eax+edx]
        add     esi, eax
        add     ebx, ecx
        rol     ebx, cl

        add     eax, Sx(%1+1,%2)
        add     eax, ebx
        rol     eax, 3
        mov     ecx, esi
        xor     edi, esi
        rol     edi, cl
        lea     ecx, [eax+ebx]
        add     edi, eax
        add     edx, ecx
        rol     edx, cl
%endmacro

;
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

;   let's do some register allocation
;
;        use1  use2        use1  use2
;        -----|-----|      -----|-----|
;   eax  A1    Llo4   ebx  A2    
;   ecx  A3           edx  A4    
;   esi  Llo1         edi  Llo2
;   ebp  Llo3         

;   load parameters
        mov     esi, [RC5UnitWork_L0lo]           ; esi = l0 = Llo1
        mov     edx, [RC5UnitWork_L0hi]           
        add     edx, 0x03000000
        mov     [work_Lhi4], edx
        sub     edx, 0x01000000
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

        mov     ecx, [work_Lhi1]
        mov     edx, [work_Lhi2]
        mov     esi, [work_Lhi3]
        mov     edi, [work_Lhi4]

        mov     ecx, [work_pre3_r1]

        add     ecx, ebx
        rol     ecx, cl
        add     edx, ebx
        rol     edx, cl
        add     esi, ebx
        rol     esi, cl
        add     edi, ebx
        rol     edi, cl
        
        mov		eax, [work_pre1_r1]
        mov		[work_Llo1], eax
        mov		[work_Llo2], eax
        mov		[work_Llo3], eax
        mov		[work_Llo4], eax
        mov     eax, [work_pre2_r1]
        mov     ebx, eax
        mov     [work_A3], eax
        mov     [work_A4], eax
        lea     eax, [S_not(2)+eax+ecx]


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


    ; ------------------------------
    ; Begin round 2 of key expansion
    ; ------------------------------

align 4
_end_round1_p7:

;        use1  use2        use1  use2
;        -----|-----|      -----|-----|
;   eax  A1           ebx  A2    tmp
;   ecx  A3    Lhi1   edx  A4    Lhi2  
;   esi  Llo1  Lhi3   edi  Llo3  Lhi4
;   ebp  Llo4  Llo2      

        mov     ecx, S3(25)
        lea     ebx, [S0_ROTL3+ebx+edx]    ; A2 = (A2 + Lhi2 + S_not(N));
        mov     edx, S4(25)
        lea     ecx, [S0_ROTL3+ecx+esi]    ; A3 = (A3 + Lhi3 + S_not(N));
        lea     edx, [S0_ROTL3+edx+edi]    ; A4 = (A4 + Lhi4 + S_not(N));
        rol     edx, 3
        rol     ecx, 3
        rol     ebx, 3
        rol     eax, 3
        mov     S1(0), eax
        mov     S2(0), ebx
        mov     S3(0), ecx
        mov     S4(0), edx
        mov     ebx, ecx
        lea     ecx, [edx+edi]
        add     edx, [work_pre2_r1]
        add     ebp, ecx
        rol     ebp, cl                     ; Llo4 = ROTL (Llo4 + A4 + Lhi4, A4 + Lhi4);
        mov     [work_Lhi4], edi
        mov     edi, [work_Llo3]        
        lea     ecx, [ebx+esi]
        add     ebx, [work_pre2_r1]
        add     edi, ecx
        rol     edi, cl                     ; Llo3 = ROTL (Llo3 + A3 + Lhi3, A3 + Lhi3);  
        mov     [work_A3], ebx
        mov     ecx, S2(0)
        mov     ebx, ecx
        add     ecx, [work_Lhi2]
        mov     [work_Llo4], ebp
        mov     ebp, [work_Llo2]
        add     ebx, [work_pre2_r1]
        add     ebp, ecx
        rol     ebp, cl                     ; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
        mov     [work_Lhi3], esi
        mov     esi, [work_Llo1]
        mov     [work_A4], edx
        mov     edx, [work_Lhi1]
        lea     ecx, [eax+edx] 
        add     eax, [work_pre2_r1]
        add     esi, ecx
        rol     esi, cl                     ; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);

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

        mov     [work_Lhi3], esi
        mov     [work_Lhi4], edi
        
        mov     ebx, [work_Llo1]
        mov     edx, [work_Lhi1]

    ; ----------------------------------------------------
    ; Begin round 3 of key expansion mixed with encryption
    ; ----------------------------------------------------
    ; (first key)

	; A  = %eax  eA = %esi
	; L0 = %ebx  eB = %edi
	; L1 = %edx  .. = %ebp

        add     eax, edx
        rol     eax, 3
        mov     esi, [work_P_0]
        lea     ecx, [eax+edx]
        add     esi, eax
        add     ebx, ecx
        rol     ebx, cl

        add     eax, S1(1)
        add     eax, ebx
        rol     eax, 3
        mov     edi, [work_P_1]
        lea     ecx, [eax+ebx]
        add     edi, eax
        add     edx, ecx
        rol     edx, cl
	
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

  ; early exit
align 4
_end_round3_1_p7:
        add     eax, S1(24)
        add     eax, edx
        rol     eax, 3
        mov     ecx, edi
        xor     esi, edi
        rol     esi, cl
        add     esi, eax
					
        cmp     esi, [work_C_0]
        jne     __exit_1_p7
					
        lea     ecx, [eax+edx]
        add     ebx, ecx
        rol     ebx, cl
        add     eax, S1(25)
        add     eax, ebx
        rol     eax, 3
        mov     ecx, esi
        xor     edi, esi
        rol     edi, cl
        add     edi, eax

        cmp     edi, [work_C_1]
        je near _full_exit_p7

align 4
__exit_1_p7:

    ; Restore 2nd key parameters
        mov     edx, [work_Lhi2]
        mov     ebx, [work_Llo2]
        mov     eax, S2(25)

    ; ----------------------------------------------------
    ; Begin round 3 of key expansion mixed with encryption
    ; ----------------------------------------------------
    ; (second key)

  ; A  = %eax  eA = %esi
  ; L0 = %ebx  eB = %edi
  ; L1 = %edx  .. = %ebp

        add     eax, S2(0)
        add     eax, edx
        rol     eax, 3
        mov     esi, [work_P_0]
        lea     ecx, [eax+edx]
        add     esi, eax
        add     ebx, ecx
        rol     ebx, cl

        add     eax, S2(1)
        add     eax, ebx
        rol     eax, 3
        mov     edi, [work_P_1]
        lea     ecx, [eax+ebx]
        add     edi, eax
        add     edx, ecx
        rol     edx, cl
	
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
_end_round3_2_p7:
        add     eax, S2(24)
        add     eax, edx
        rol     eax, 3
        mov     ecx, edi
        xor     esi, edi
        rol     esi, cl
        add     esi, eax
					
        cmp     esi, [work_C_0]
        jne     __exit_2_p7
					
        lea     ecx, [eax+edx]
        add     ebx, ecx
        rol     ebx, cl
        add     eax, S2(25)
        add     eax, ebx
        rol     eax, 3
        mov     ecx, esi
        xor     edi, esi
        rol     edi, cl
        add     edi, eax

        cmp     edi, [work_C_1]
        jne      __exit_2_p7
        mov     dword [work_add_iter], 1
        jmp     _full_exit_p7

align 4
__exit_2_p7:

    ; Restore 3rd key parameters
        mov     edx, [work_Lhi3]
        mov     ebx, [work_Llo3]
        mov     eax, S3(25)

    ; ----------------------------------------------------
    ; Begin round 3 of key expansion mixed with encryption
    ; ----------------------------------------------------
    ; (second key)

  ; A  = %eax  eA = %esi
  ; L0 = %ebx  eB = %edi
  ; L1 = %edx  .. = %ebp

        add     eax, S3(0)
        add     eax, edx
        rol     eax, 3
        mov     esi, [work_P_0]
        lea     ecx, [eax+edx]
        add     esi, eax
        add     ebx, ecx
        rol     ebx, cl

        add     eax, S3(1)
        add     eax, ebx
        rol     eax, 3
        mov     edi, [work_P_1]
        lea     ecx, [eax+ebx]
        add     edi, eax
        add     edx, ecx
        rol     edx, cl
	
	ROUND_3_EVEN_AND_ODD  2,3
	ROUND_3_EVEN_AND_ODD  4,3
	ROUND_3_EVEN_AND_ODD  6,3
	ROUND_3_EVEN_AND_ODD  8,3
	ROUND_3_EVEN_AND_ODD 10,3
	ROUND_3_EVEN_AND_ODD 12,3
	ROUND_3_EVEN_AND_ODD 14,3
	ROUND_3_EVEN_AND_ODD 16,3
	ROUND_3_EVEN_AND_ODD 18,3
	ROUND_3_EVEN_AND_ODD 20,3
	ROUND_3_EVEN_AND_ODD 22,3

  ; early exit
align 4
_end_round3_3_p7:
        add     eax, S3(24)
        add     eax, edx
        rol     eax, 3
        mov     ecx, edi
        xor     esi, edi
        rol     esi, cl
        add     esi, eax
					
        cmp     esi, [work_C_0]
        jne     __exit_3_p7
					
        lea     ecx, [eax+edx]
        add     ebx, ecx
        rol     ebx, cl
        add     eax, S3(25)
        add     eax, ebx
        rol     eax, 3
        mov     ecx, esi
        xor     edi, esi
        rol     edi, cl
        add     edi, eax

        cmp     edi, [work_C_1]
        jne      __exit_3_p7
        mov     dword [work_add_iter], 2
        jmp     _full_exit_p7

align 4
__exit_3_p7:

    ; Restore 4th key parameters
        mov     edx, [work_Lhi4]
        mov     ebx, [work_Llo4]
        mov     eax, S4(25)

    ; ----------------------------------------------------
    ; Begin round 3 of key expansion mixed with encryption
    ; ----------------------------------------------------
    ; (second key)

  ; A  = %eax  eA = %esi
  ; L0 = %ebx  eB = %edi
  ; L1 = %edx  .. = %ebp

        add     eax, S4(0)
        add     eax, edx
        rol     eax, 3
        mov     esi, [work_P_0]
        lea     ecx, [eax+edx]
        add     esi, eax
        add     ebx, ecx
        rol     ebx, cl

        add     eax, S4(1)
        add     eax, ebx
        rol     eax, 3
        mov     edi, [work_P_1]
        lea     ecx, [eax+ebx]
        add     edi, eax
        add     edx, ecx
        rol     edx, cl
	
	ROUND_3_EVEN_AND_ODD  2,4
	ROUND_3_EVEN_AND_ODD  4,4
	ROUND_3_EVEN_AND_ODD  6,4
	ROUND_3_EVEN_AND_ODD  8,4
	ROUND_3_EVEN_AND_ODD 10,4
	ROUND_3_EVEN_AND_ODD 12,4
	ROUND_3_EVEN_AND_ODD 14,4
	ROUND_3_EVEN_AND_ODD 16,4
	ROUND_3_EVEN_AND_ODD 18,4
	ROUND_3_EVEN_AND_ODD 20,4
	ROUND_3_EVEN_AND_ODD 22,4

  ; early exit
align 4
_end_round3_4_p7:
        add     eax, S4(24)
        add     eax, edx
        rol     eax, 3
        mov     ecx, edi
        xor     esi, edi
        rol     esi, cl
        add     esi, eax
					
        cmp     esi, [work_C_0]
        jne     __exit_4_p7
					
        lea     ecx, [eax+edx]
        add     ebx, ecx
        rol     ebx, cl
        add     eax, S4(25)
        add     eax, ebx
        rol     eax, 3
        mov     ecx, esi
        xor     edi, esi
        rol     edi, cl
        add     edi, eax

        cmp     edi, [work_C_1]
        jne      __exit_4_p7
        mov     dword [work_add_iter], 3
        jmp     _full_exit_p7

align 4
__exit_4_p7:
        mov     edx, [work_key_hi]

; Jumps not taken are faster
        add     edx, 0x04000000
        jc near _next_inc_p7

align 4
_next_iter_p7:
        mov     [work_key_hi], edx
        add     edx, 0x03000000
        mov     [work_Lhi4], edx
        sub     edx, 0x01000000
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
        add     edx, 0x03000000
        mov     [work_Lhi4], edx
        sub     edx, 0x01000000
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
lea edx, [eax+ebp*4]
mov eax, edx

;    return (timeslice - work.iterations) * 4 + work.add_iter;


      mov ebx, [save_ebx]
      mov esi, [save_esi]
      mov edi, [save_edi]
      mov ebp, [save_ebp]

     add esp, work_size ; restore stack pointer

     ret


