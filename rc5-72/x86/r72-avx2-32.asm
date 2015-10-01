; RC5-72 AVX-2 core
; Based on 64-bit core created by Yasuhiro Katsui (bakabaka9xx@gmail.com, bakabaka9xx@github)
;
; 8-pipe
; Need popcnt and AVX2 instruction
; Optimized for uOP cache
;
; Created by Roman Trunov (stream@distributed.net)
; This is a pretty straightforward cut of Yasuhiro Katsu' core, with only
; one AVX pipeline left (32-bit mode has not enough CPU registers).
; It's not optimized up to the last clock and was written only for peoples
; which have modern hardware but still run 32-bit OS due to unknown reason.
; 64-bit core is significantly faster. And GPU is much, much better then 64-bit core! :)
;
; Some hints for possible optimization:
; - some cmov's in original code were replaced by jumps (not enough registers to cache
;   their source data). Really, it's not clear which approach is better. According to
;   Agner Fog's optimization books, predicted jump in some cases could be better then cmov.
;   In my setup, any performance change is below benchmarking error.
; - edi could be freed by copying whole RC5_72UnitWork structure in work area (and copying it
;   back on exit) or by reloading after another usage.
;

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
[SECTION _TEXT USE32 ALIGN=16 CLASS=CODE]
%else
[SECTION .text]
%endif

[GLOBAL _rc5_72_unit_func_avx2]
[GLOBAL rc5_72_unit_func_avx2]

%define _mm256_set1_d(N) N,N,N,N,N,N,N,N
%define _mm512_set1_d(N) N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N

%define RESULT_NOTHING 1
%define RESULT_FOUND   2

%assign work_size 0

%macro defidef 2
    %define %1 ebp-%2
%endmacro

; This area grows down!
%macro defwork 1-2 1
    %assign work_size work_size+4*(%2)
    defidef %1,work_size
%endmacro

%assign work32_size 0

%macro defidef32 2
    %define %1 ebp+%2
%endmacro

; This area moves up
%macro defwork32 1-2 1
    defidef32 %1,work32_size
    %assign work32_size work32_size+32*(%2)
%endmacro

; local storage variables - referenced via [EBP-xxx]

defwork L0lo           ; very rare access - ok to use memory
defwork L0mid          ; accessed every 256/8 iterations - let it be a memory...
defwork work_mid_ecx
defwork work_pre_B_0
defwork work_pre_B_1
defwork work_pre_B_2
defwork work_pre_S1
defwork work_pre_S2

; Second pool of variables - aligned by 32, referenced via [EBP+xxx]
; Add padding area for alignment
%assign work_size work_size + 32
; Start of secondary area (non-aligned yet)
%assign work_offset work_size

; Original core used ebp-trick in setup_key. Instead of calculating S(-1),
; make S exactly second element of the area (will have offset of 32),
; then ebp will point exactly to S(-1). We can "cmov ..., ebp" when need S(-1).
; But I cannot find any gain using trick on my CPU, if one exist, it's close
; to benchmark errors. So I left it in straightforward way. For those who wish to try,
; layout of this area must be:
;    defwork32 work_C_0
;    defwork32 work_S, 26
;    defwork32 work_C_1

defwork32 work_C_0
defwork32 work_C_1
defwork32 work_S, 26

; Concatenate two work areas together
%assign work_size work_size + work32_size

;; offsets within the parameter structure
%define RC5_72UnitWork_plainhi  edi+0
%define RC5_72UnitWork_plainlo  edi+4
%define RC5_72UnitWork_cipherhi edi+8
%define RC5_72UnitWork_cipherlo edi+12
%define RC5_72UnitWork_L0hi     edi+16
%define RC5_72UnitWork_L0mid    edi+20
%define RC5_72UnitWork_L0lo     edi+24
%define RC5_72UnitWork_CMCcount edi+28
%define RC5_72UnitWork_CMChi    edi+32
%define RC5_72UnitWork_CMCmid   edi+36
%define RC5_72UnitWork_CMClo    edi+40

%define S1(N)   [work_S+((N)*32)]

%define L0hi      ebx
%define L0hi_byte  bl
;%define L0mid   r14d
;%define L0lo    r15d

%define T1      ymm0
%define T2      ymm1

%define A1      ymm2     ; must coexist with C1/D1
%define B1_0    C1       ; live only during key_setup
%define B1_1    ymm3     ; must coexist with C1/D1
%define B1_2    D1       ; live only during key_setup

%define C1      ymm4
%define D1      ymm5

%define R31     ymm6
%define R32     ymm7

align 16
startseg:
_rc5_72_unit_func_avx2:
rc5_72_unit_func_avx2:

    push    ebx
    push    esi
    push    edi
    push    ebp                ; pushed 10h bytes
%define param_unitwork     esp+work_size+10h+4
%define param_piterations  esp+work_size+10h+8

    sub     esp, work_size
; Move EBP to start of secondary area, after alignment space
; Align secondary area by 32 bytes.
    lea     ebp, [esp+work_offset]
    and     ebp, -32

    mov     edi, [param_unitwork]
    mov     esi, [param_piterations]
    mov     esi, [esi]
    shr     esi, 3

    vpbroadcastd    T1, [RC5_72UnitWork_cipherlo]
    vpbroadcastd    T2, [RC5_72UnitWork_cipherhi]
    vmovdqa [work_C_0], T1
    vmovdqa [work_C_1], T2

    mov     L0hi, [RC5_72UnitWork_L0hi]
    mov     eax, [RC5_72UnitWork_L0mid]
    mov     [L0mid], eax
    mov     eax, [RC5_72UnitWork_L0lo]
    mov     [L0lo], eax

    vmovdqa R31, [rel .I31]
    vmovdqa R32, [rel .I32]

.key_setup_1_bigger_loop:

    mov     ecx, [L0lo]
    add     ecx, 0xBF0A8B1D
    rol     ecx, 29
    mov     [work_pre_B_0], ecx

    mov     eax, ecx
    add     eax, 0x15235639
    rol     eax, 3
    mov     [work_pre_S1], eax
    add     ecx, eax
    mov     [work_mid_ecx], ecx
.key_setup_1_mid_loop:
    mov     edx, [L0mid]
    add     edx, ecx
    rol     edx, cl
    mov     [work_pre_B_1], edx

    add     eax, edx
    add     eax, 0xF45044D5
    rol     eax, 3
    mov     [work_pre_S2], eax
    add     edx, eax
    mov     [work_pre_B_2], edx

.key_setup_1_inner_loop:

    vpbroadcastd    A1, [work_pre_S2]
    vpbroadcastd    B1_0, [work_pre_B_0]
    vpbroadcastd    B1_1, [work_pre_B_1]

    vpbroadcastd    T1, [work_pre_S1]
    vmovdqa S1(1), T1
    vmovdqa S1(2), A1

    vpbroadcastd    B1_2, [RC5_72UnitWork_L0hi]
    vpbroadcastd    T1, [work_pre_B_2]
    vpaddd  B1_2, [rel .I1]
    vpaddd  B1_2, T1
    vpand   T1, R31, T1
    vpsubd  T2, R32, T1
    vpsllvd T1, B1_2, T1

    lea     eax, [rel .IS3]
    lea     edx, S1(3)
    xor     ecx, ecx
    jmp     .round1

.round:
    vpaddd  A1, [eax-32]
    cmp     ecx, 7
    jne     .f1
    lea     edx, S1(1)
.f1:

    vpaddd  A1, B1_1, A1
    vpslld  T1, A1, 3
    add     ecx, 1

    vpsrld  A1, A1, 29

    vpor    A1, T1

    vmovdqa [edx-32], A1
    vpaddd  T1, B1_1, A1

    vpaddd  B1_2, T1
    vpand   T1, R31, T1

    vpsubd  T2, R32, T1

    vpsllvd T1, B1_2, T1
.round1:
    vpsrlvd B1_2, B1_2, T2

    cmp     ecx, 8
    cmove   eax, edx

    vpor    B1_2, T1
    vpaddd  A1, [eax]

    vpaddd  A1, B1_2, A1
    vpslld  T1, A1, 3

    vpsrld  A1, A1, 29

    vpor    A1, T1
    vmovdqa [edx], A1

    vpaddd  T1, B1_2, A1
    vpaddd  B1_0, T1

    vpand   T1, R31, T1

    vpsubd  T2, R32, T1

    vpsllvd T1, B1_0, T1

    vpsrlvd B1_0, B1_0, T2

    vpor    B1_0, T1

    cmp     ecx, 16
;   cmove   edx, ebp         ; requires special ebp-frame, see above
;   cmove   eax, ebp
    jne     .f2
    lea     edx, S1(-1)
    lea     eax, S1(-1)
.f2:

    vpaddd  A1, [eax+32]
    vpaddd  A1, B1_0, A1

    vpslld  T1, A1, 3
        add     eax, 96

    vpsrld  A1, A1, 29

    vpor    A1, T1

    vmovdqa [edx+32], A1
    vpaddd  T1, B1_0, A1

    vpaddd  B1_1, T1
    vpand   T1, R31, T1

    vpsubd  T2, R32, T1

    vpsllvd T1, B1_1, T1

    vpsrlvd B1_1, B1_1, T2

    vpor    B1_1, T1

        add     edx, 96
    cmp     ecx, 24
    jne     .round
.roundEnd:

    vpbroadcastd  T1, [RC5_72UnitWork_plainlo]
    vpbroadcastd  T2, [RC5_72UnitWork_plainhi]
    vpaddd  C1, T1, S1(0)
    vpaddd  D1, T2, S1(1)

    lea     eax, S1(2)
    xor     edx,edx

.encrypt:
    inc     edx
    vpxor   C1, D1
    vpand   T1, R31, D1

    vpsubd  T2, R32, T1
    vpsllvd T1, C1, T1

    vpsrlvd C1, C1, T2

    vpor    C1, T1

    vpaddd  C1, [eax]

    vpxor   D1, C1
    vpand   T1, R31, C1

    vpsubd  T2, R32, T1
    vpsllvd T1, D1, T1

    vpsrlvd D1, D1, T2

    vpor    D1, T1

    vpaddd  D1, [eax+32]
    add     eax, 64


    cmp     edx, 11
    jne     .encrypt
.encryptEnd:

    vpxor   C1, D1
    vpand   T1, R31, D1
    vpsubd  T2, R32, T1
    vpsllvd T1, C1, T1

    vpsrlvd C1, C1, T2

    vpor    C1, T1

    vpaddd  C1, S1(24)
    vpcmpeqd    T1, C1, [work_C_0]
    vmovmskps   eax, T1

    cmp     eax, 0
    jne     .checkKey1Hi
.nextKey:
    dec     esi
    jz      .finished_Found_nothing

    add     L0hi_byte, 8
    movzx   L0hi, L0hi_byte
    mov     [RC5_72UnitWork_L0hi], L0hi
    jnc     .key_setup_1_inner_loop

    mov     eax, [L0mid]
    bswap   eax
    adc     eax, 0
    bswap   eax
    mov     [L0mid], eax
    mov     eax, [work_pre_S1]
    mov     ecx, [work_mid_ecx]
    jnc     .key_setup_1_mid_loop

    mov     eax, [L0lo]
    bswap   eax
    adc     eax, 0
    bswap   eax
    mov     [L0lo], eax
    jmp     .key_setup_1_bigger_loop

.checkKey1Hi:
    popcnt  edx, eax
    add     dword [RC5_72UnitWork_CMCcount], edx
    bsr     eax, eax
    add     eax, L0hi
    mov     [RC5_72UnitWork_CMChi], eax
    mov     eax, [L0mid]
    mov     [RC5_72UnitWork_CMCmid], eax
    mov     eax, [L0lo]
    mov     [RC5_72UnitWork_CMClo], eax

    vpaddd  A1, S1(25)
    vpaddd  A1, B1_1, A1
    vpslld  T1, A1, 3
    vpsrld  A1, A1, 29
    vpor    A1, T1
    vpxor   D1, C1
    vpand   T1, R31, C1
    vpsubd  T2, R32, T1
    vpsllvd T1, D1, T1
    vpsrlvd D1, D1, T2
    vpor    D1, T1
    vpaddd  D1, A1

    vpcmpeqd    D1, [work_C_1]
    vmovmskps   eax, D1
    cmp     eax, 0
    je      .nextKey

; It seems that this code cancels cmc's found above, and adding only one
; for found key. (at this point only one bit is usually set in eax,
; "winning" key is only one, and false double matches, which decrypts
; only two first words but not the whole message, are very, very rare).
; This is questionable - it could be a partial match before full one in the
; pipeline, but I think it's acceptable for now because exact cmc_count
; does not really matter when RESULT_FOUND is reported; such results are
; always handled manually - either we win, either it's false double match
; (and block must be reprocessed completely manually), either core has
; reported wrong result and block will be discarded and recycled.
    popcnt  ecx, eax
    sub     edx, ecx
    sub     dword [RC5_72UnitWork_CMCcount], edx
; update number of iterations done and CMChi using exact key number
    bsr     eax, eax
    shl     esi, 3
    sub     esi, eax
    mov     ecx, [param_piterations]
    sub     [ecx], esi
    add     eax, L0hi
    mov     [RC5_72UnitWork_CMChi], eax

; Save rest of current key before exit. L0hi was saved during key update.
; Note that saved key point to start of pipeline-block where full match was found.
    mov     eax, [L0mid]
    mov     [RC5_72UnitWork_L0mid], eax
    mov     eax, [L0lo]
    mov     [RC5_72UnitWork_L0lo], eax
    mov     eax, RESULT_FOUND
    jmp     .finished

.finished_Found_nothing:
    add     L0hi_byte, 8
    mov     [RC5_72UnitWork_L0hi], L0hi

    mov     ecx, [L0mid]
    mov     eax, [L0lo]
    bswap   ecx
    bswap   eax
    adc     ecx, 0
    adc     eax, 0
    bswap   ecx
    bswap   eax
    mov     [RC5_72UnitWork_L0mid], ecx
    mov     [RC5_72UnitWork_L0lo], eax
    mov     eax, RESULT_NOTHING

.finished:
    add     esp, work_size
    pop     ebp
    pop     edi
    pop     esi
    pop     ebx
    ret

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
[SECTION _DATA USE32 ALIGN=16 CLASS=DATA]
%else
[SECTION .data]
%endif
align 32
.I1:    dd  0, 1,  2,  3,  4,  5,  6,  7
.I31:   dd _mm256_set1_d(0x0000001F)
.I32:   dd _mm256_set1_d(0x00000020)
.IS3:   dd _mm256_set1_d(0x9287BE8E)
.IS4:   dd _mm256_set1_d(0x30BF3847)
.IS5:   dd _mm256_set1_d(0xCEF6B200)
.IS6:   dd _mm256_set1_d(0x6D2E2BB9)
.IS7:   dd _mm256_set1_d(0x0B65A572)
.IS8:   dd _mm256_set1_d(0xA99D1F2B)
.IS9:   dd _mm256_set1_d(0x47D498E4)
.IS10:  dd _mm256_set1_d(0xE60C129D)
.IS11:  dd _mm256_set1_d(0x84438C56)
.IS12:  dd _mm256_set1_d(0x227B060F)
.IS13:  dd _mm256_set1_d(0xC0B27FC8)
.IS14:  dd _mm256_set1_d(0x5EE9F981)
.IS15:  dd _mm256_set1_d(0xFD21733A)
.IS16:  dd _mm256_set1_d(0x9B58ECF3)
.IS17:  dd _mm256_set1_d(0x399066AC)
.IS18:  dd _mm256_set1_d(0xD7C7E065)
.IS19:  dd _mm256_set1_d(0x75FF5A1E)
.IS20:  dd _mm256_set1_d(0x1436D3D7)
.IS21:  dd _mm256_set1_d(0xB26E4D90)
.IS22:  dd _mm256_set1_d(0x50A5C749)
.IS23:  dd _mm256_set1_d(0xEEDD4102)
.IS24:  dd _mm256_set1_d(0x8D14BABB)
.IS25:  dd _mm256_set1_d(0x2B4C3474)
.IS0:   dd _mm256_set1_d(0xBF0A8B1D)
