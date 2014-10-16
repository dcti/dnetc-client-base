; 16-pipe
; Need popcnt and AVX2 instruction
; Optimized for uOP cache

[SECTION .data]
[SECTION .text]
BITS 64

[GLOBAL _rc5_72_unit_func_avx2]
[GLOBAL rc5_72_unit_func_avx2]

%define _mm256_set1_d(N) N,N,N,N,N,N,N,N
%define _mm512_set1_d(N) N,N,N,N,N,N,N,N,N,N,N,N,N,N,N,N

%define RESULT_NOTHING 1
%define RESULT_FOUND   2

%assign work_size 0

%macro defidef 2
    %define %1 rsp+%2
%endmacro

%macro defwork 1-2 1
    defidef %1,work_size
    %assign work_size work_size+4*(%2)
%endmacro

%assign work32_size 0

%macro defidef32 2
    %define %1 rbp+%2
%endmacro

%macro defwork32 1-2 1
    defidef32 %1,work32_size
    %assign work32_size work32_size+32*(%2)
%endmacro

; local storage variables

%ifdef _WINDOWS
defwork save_xmm6,4
defwork save_xmm7,4
defwork save_xmm8,4
defwork save_xmm9,4
defwork save_xmm10,4
defwork save_xmm11,4
defwork save_xmm12,4
defwork save_xmm13,4
defwork save_xmm14,4
defwork save_xmm15,4
defwork save_rsi,2
defwork save_rdi,2
%endif
defwork save_rbx,2
defwork save_rbp,2
defwork save_r12,2
defwork save_r13,2
defwork save_r14,2
defwork save_r15,2

defwork work_pre_A
defwork work_pre_B_0
defwork work_pre_B_1
defwork work_pre_B_2
defwork work_pre_S1
defwork work_pre_S2

%assign work_offset work_size

defwork32 work_C_0
defwork32 work_C_1
defwork32 work_S, (26*2)

%assign work_size work_offset + work32_size + 32
%assign work_size work_size + (8 - work_size % 16)

;; offsets within the parameter structure
%define RC5_72UnitWork_plainhi  rdi+0
%define RC5_72UnitWork_plainlo  rdi+4
%define RC5_72UnitWork_cipherhi rdi+8
%define RC5_72UnitWork_cipherlo rdi+12
%define RC5_72UnitWork_L0hi     rdi+16
%define RC5_72UnitWork_L0mid    rdi+20
%define RC5_72UnitWork_L0lo     rdi+24
%define RC5_72UnitWork_CMCcount rdi+28
%define RC5_72UnitWork_CMChi    rdi+32
%define RC5_72UnitWork_CMCmid   rdi+36
%define RC5_72UnitWork_CMClo    rdi+40

%undef  work_pre_A
%define work_pre_A              rbp-24
%undef  work_pre_B_0
%define work_pre_B_0            rbp-20
%undef  work_pre_B_1
%define work_pre_B_1            rbp-16
%undef  work_pre_B_2
%define work_pre_B_2            rbp-12
%undef  work_pre_S1
%define work_pre_S1             rbp-8
%undef  work_pre_S2
%define work_pre_S2             rbp-4

%define S1(N)   [work_S+((N)*64)]
%define S2(N)   [work_S+((N)*64)+32]

%define L0hi    r13d
%define L0mid   r14d
%define L0lo    r15d

%define T1      ymm0
%define T2      ymm1
%define T3      ymm2
%define T4      ymm3

%define A1      ymm6
%define A2      ymm7
%define B1_0    ymm8
%define B1_1    ymm9
%define B1_2    ymm10
%define B2_0    ymm11
%define B2_1    ymm12
%define B2_2    ymm13

%define C1      ymm8
%define C2      ymm10
%define D1      ymm11
%define D2      ymm13

%define R31     ymm14
%define R32     ymm15

align 16
startseg:
_rc5_72_unit_func_avx2:
rc5_72_unit_func_avx2:

    sub     rsp, work_size

%ifdef _WINDOWS
    vmovdqa [save_xmm6], xmm6
    vmovdqa [save_xmm7], xmm7
    vmovdqa [save_xmm8], xmm8
    vmovdqa [save_xmm9], xmm9
    vmovdqa [save_xmm10], xmm10
    vmovdqa [save_xmm11], xmm11
    vmovdqa [save_xmm12], xmm12
    vmovdqa [save_xmm13], xmm13
    vmovdqa [save_xmm14], xmm14
    vmovdqa [save_xmm15], xmm15
    mov     [save_rsi], rsi
    mov     [save_rdi], rdi
    mov     rdi, rcx
    mov     rsi, rdx
%else

%endif
    mov     [save_rbx], rbx
    mov     [save_rbp], rbp
    mov     [save_r12], r12
    mov     [save_r13], r13
    mov     [save_r14], r14
    mov     [save_r15], r15

    lea     rbp, [rsp+work_offset+32]
    and     bpl, 0xE0

    mov     r10d, [rsi]
    shr     r10d, 4

    vpbroadcastd    T1, [RC5_72UnitWork_cipherlo]
    vpbroadcastd    T2, [RC5_72UnitWork_cipherhi]
    vmovdqa [work_C_0], T1
    vmovdqa [work_C_1], T2

    mov     L0hi, [RC5_72UnitWork_L0hi]
    mov     L0mid, [RC5_72UnitWork_L0mid]
    mov     L0lo, [RC5_72UnitWork_L0lo]

    vmovdqa R31, [rel .I31]
    vmovdqa R32, [rel .I32]

    lea     r12, S1(1)

.key_setup_1_bigger_loop:

    mov     ecx, L0lo
    add     ecx, 0xBF0A8B1D
    rol     ecx, 29
    mov     [work_pre_B_0], ecx

    mov     eax, ecx
    add     eax, 0x15235639
    rol     eax, 3
    mov     [work_pre_S1], eax
    add     ecx, eax
.key_setup_1_mid_loop
    lea     edx, [L0mid+ecx]
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
        vmovdqa     A2, A1
        vmovdqa     B2_0, B1_0
        vmovdqa     B2_1, B1_1

    vpbroadcastd    T1, [work_pre_S1]
    vmovdqa S1(1), T1
    vmovdqa S2(1), T1
    vmovdqa S1(2), A1
    vmovdqa S2(2), A1

    vpbroadcastd    B1_2, [RC5_72UnitWork_L0hi]
    vpbroadcastd    T1, [work_pre_B_2]
        vmovdqa     B2_2, B1_2
        vmovdqa     T3, T1
    vpaddd  B1_2, [rel .I1]
    vpaddd  B1_2, T1
    vpand   T1, R31, T1
    vpsubd  T2, R32, T1
    vpsllvd T1, B1_2, T1
        vpaddd  B2_2, [rel .I2]

    lea     rax, [rel .IS3]
    lea     rdx, S1(3)
    xor     ebx, ebx
    jmp     .round1

.round:
        vpsrlvd B2_1, B2_1, T4

    vpaddd  A1, [rax-64]
    cmp     ebx, 7
    cmove   rdx, r12

    vpaddd  A1, B1_1, A1
        vpor    B2_1, T3
    vpslld  T1, A1, 3
    add     ebx, 1

        vpaddd  A2, [rax-32]
    vpsrld  A1, A1, 29
        vpaddd  A2, B2_1, A2

    vpor    A1, T1
        vpslld  T3, A2, 3

    vmovdqa [rdx-64], A1
    vpaddd  T1, B1_1, A1
        vpsrld  A2, A2, 29

    vpaddd  B1_2, T1
    vpand   T1, R31, T1
        vpor    A2, T3

        vmovdqa [rdx-32], A2
    vpsubd  T2, R32, T1
        vpaddd  T3, B2_1, A2

    vpsllvd T1, B1_2, T1
.round1:
    vpsrlvd B1_2, B1_2, T2

        vpaddd  B2_2, T3
        vpand   T3, R31, T3
    cmp     ebx, 8
    cmove   rax, rdx

    vpor    B1_2, T1
        vpsubd  T4, R32, T3
    vpaddd  A1, [rax]

        vpsllvd T3, B2_2, T3

        vpsrlvd B2_2, B2_2, T4

    vpaddd  A1, B1_2, A1
        vpor    B2_2, T3
    vpslld  T1, A1, 3

        vpaddd  A2, [rax+32]
    vpsrld  A1, A1, 29
        vpaddd  A2, B2_2, A2

    vpor    A1, T1
        vpslld  T3, A2, 3
    vmovdqa [rdx], A1

    vpaddd  T1, B1_2, A1
        vpsrld  A2, A2, 29
    vpaddd  B1_0, T1

        vpor    A2, T3
    vpand   T1, R31, T1
        vmovdqa [rdx+32], A2

    vpsubd  T2, R32, T1
        vpaddd  T3, B2_2, A2

    vpsllvd T1, B1_0, T1

    vpsrlvd B1_0, B1_0, T2

        vpaddd  B2_0, T3
        vpand   T3, R31, T3
    vpor    B1_0, T1
        vpsubd  T4, R32, T3

    cmp     ebx, 16
    cmove   rdx, rbp
    cmove   rax, rbp

        vpsllvd T3, B2_0, T3

        vpsrlvd B2_0, B2_0, T4

    vpaddd  A1, [rax+64]
        vpor    B2_0, T3
    vpaddd  A1, B1_0, A1

    vpslld  T1, A1, 3
        vpaddd  A2, [rax+96]
        add     rax, 192

    vpsrld  A1, A1, 29
        vpaddd  A2, B2_0, A2

    vpor    A1, T1
        vpslld  T3, A2, 3

    vmovdqa [rdx+64], A1
    vpaddd  T1, B1_0, A1
        vpsrld  A2, A2, 29

    vpaddd  B1_1, T1
    vpand   T1, R31, T1
        vpor    A2, T3

        vmovdqa [rdx+96], A2
    vpsubd  T2, R32, T1
        vpaddd  T3, B2_0, A2

    vpsllvd T1, B1_1, T1

    vpsrlvd B1_1, B1_1, T2

        vpaddd  B2_1, T3
        vpand   T3, R31, T3
    vpor    B1_1, T1
        vpsubd  T4, R32, T3

        vpsllvd T3, B2_1, T3

        add     rdx, 192
    cmp     ebx, 24
    jne     .round
.roundEnd:
        vpsrlvd B2_1, B2_1, T4
        vpor    B2_1, T3

    vpbroadcastd  T1, [RC5_72UnitWork_plainlo]
    vpbroadcastd  T3, [RC5_72UnitWork_plainhi]
    vpaddd  C1, T1, S1(0)
    vpaddd  D1, T3, S1(1)
        vpaddd  C2, T1, S2(0)
        vmovdqa  D2, T3

    lea     rax, S2(1)
    xor     ebx,ebx

.encrypt:
    inc     ebx
    vpxor   C1, D1
    vpand   T1, R31, D1
        vpor    D2, T3

    vpsubd  T2, R32, T1
    vpsllvd T1, C1, T1
        vpaddd  D2, [rax]

    vpsrlvd C1, C1, T2
        vpxor   C2, D2
        vpand   T3, R31, D2

    vpor    C1, T1
        vpsubd  T4, R32, T3
        vpsllvd T3, C2, T3

    vpaddd  C1, [rax+32]
        vpsrlvd C2, C2, T4

    vpxor   D1, C1
    vpand   T1, R31, C1
        vpor    C2, T3

    vpsubd  T2, R32, T1
    vpsllvd T1, D1, T1
        vpaddd  C2, [rax+64]

    vpsrlvd D1, D1, T2
        vpxor   D2, C2
        vpand   T3, R31, C2

    vpor    D1, T1
        vpsubd  T4, R32, T3
        vpsllvd T3, D2, T3

    vpaddd  D1, [rax+96]
    add     rax, 128
        vpsrlvd D2, D2, T4


    cmp     ebx, 11
    jne     .encrypt
.encryptEnd:

    vpxor   C1, D1
    vpand   T1, R31, D1
        vpor    D2, T3
    vpsubd  T2, R32, T1
    vpsllvd T1, C1, T1
        vpaddd  D2, [rax]

    vpsrlvd C1, C1, T2
        vpxor   C2, D2
        vpand   T3, R31, D2

    vpor    C1, T1
        vpsubd  T4, R32, T3
        vpsllvd T3, C2, T3

    vpaddd  C1, S1(24)
        vpsrlvd C2, C2, T4
    vpcmpeqd    T1, C1, [work_C_0]
        vpor    C2, T3
    vmovmskps   eax, T1
        vpaddd  C2, S2(24)

    cmp     eax, 0
    jne     .checkKey1Hi
.checkKey2:
    vpcmpeqd    T2, C2, [work_C_0]
    vmovmskps   eax, T2

    cmp     eax, 0
    jne     .checkKey2Hi
.nextKey:
    dec     r10d
    jz      .finished_Found_nothing

    add     r13b, 16
    movzx   L0hi, r13b
    mov     [RC5_72UnitWork_L0hi], L0hi
    jnc     .key_setup_1_inner_loop

    bswap   L0mid
    adc     L0mid, 0
    bswap   L0mid
    mov     eax, [work_pre_S1]
    jnc     .key_setup_1_mid_loop

    bswap   L0lo
    adc     L0lo, 0
    bswap   L0lo
    jmp     .key_setup_1_bigger_loop

.checkKey1Hi:
    popcnt  edx, eax
    add     dword [RC5_72UnitWork_CMCcount], edx
    bsr     ebx, eax
    add     ebx, L0hi
    mov     [RC5_72UnitWork_CMChi], ebx
    mov     [RC5_72UnitWork_CMCmid], L0mid
    mov     [RC5_72UnitWork_CMClo], L0lo

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
    je      .checkKey2

    popcnt  ecx, eax
    sub     edx, ecx
    sub     dword [RC5_72UnitWork_CMCcount], edx
    bsr     eax, eax
    shl     r10d, 4
    sub     r10d, eax
    sub     [rsi], r10d
    add     eax, L0hi
    mov     [RC5_72UnitWork_CMChi], eax

    mov     [RC5_72UnitWork_L0mid], L0mid
    mov     [RC5_72UnitWork_L0lo], L0lo
    mov     eax, RESULT_FOUND
    jmp     .finished

.checkKey2Hi:
    popcnt  edx, eax
    add     dword [RC5_72UnitWork_CMCcount], edx
    bsr     ebx, eax
    add     ebx, 8
    add     ebx, L0hi
    mov     [RC5_72UnitWork_CMChi], ebx
    mov     [RC5_72UnitWork_CMCmid], L0mid
    mov     [RC5_72UnitWork_CMClo], L0lo

        vpaddd  A2, S2(25)
        vpaddd  A2, B2_1, A2
        vpslld  T1, A2, 3
        vpsrld  A2, A2, 29
        vpor    A2, T1
        vpxor   D2, C2
        vpand   T1, R31, C2
        vpsubd  T2, R32, T1
        vpsllvd T1, D2, T1
        vpsrlvd D2, D2, T2
        vpor    D2, T1
        vpaddd  D2, A2

    vpcmpeqd    D2, [work_C_1]
    vmovmskps   eax, D2
    cmp     eax, 0
    je      .nextKey

    popcnt  ecx, eax
    sub     edx, ecx
    sub     dword [RC5_72UnitWork_CMCcount], edx
    bsr     eax, eax
    add     eax, 8
    shl     r10d, 4
    sub     r10d, eax
    sub     [rsi], r10d
    add     eax, L0hi
    mov     [RC5_72UnitWork_CMChi], eax

    mov     [RC5_72UnitWork_L0mid], L0mid
    mov     [RC5_72UnitWork_L0lo], L0lo
    mov     eax, RESULT_FOUND
    jmp     .finished

.finished_Found_nothing:
    add     r13b, 16
    mov     [RC5_72UnitWork_L0hi], L0hi

    bswap   L0mid
    bswap   L0lo
    adc     L0mid, 0
    adc     L0lo, 0
    bswap   L0mid
    bswap   L0lo
    mov     [RC5_72UnitWork_L0mid], L0mid
    mov     [RC5_72UnitWork_L0lo], L0lo
    mov     eax, RESULT_NOTHING

.finished:

%ifdef _WINDOWS
    vmovdqa xmm6, [save_xmm6]
    vmovdqa xmm7, [save_xmm7]
    vmovdqa xmm8, [save_xmm8]
    vmovdqa xmm9, [save_xmm9]
    vmovdqa xmm10, [save_xmm10]
    vmovdqa xmm11, [save_xmm11]
    vmovdqa xmm12, [save_xmm12]
    vmovdqa xmm13, [save_xmm13]
    vmovdqa xmm14, [save_xmm14]
    vmovdqa xmm15, [save_xmm15]
    mov     rsi, [save_rsi]
    mov     rdi, [save_rdi]
%endif
    mov     rbx, [save_rbx]
    mov     rbp, [save_rbp]
    mov     r12, [save_r12]
    mov     r13, [save_r13]
    mov     r14, [save_r14]
    mov     r15, [save_r15]

    add     rsp, work_size

    ret

[SECTION .rodata]
align 32
.I1:    dd  0, 1,  2,  3,  4,  5,  6,  7
.I2:    dd  8, 9, 10, 11, 12, 13, 14, 15
.I31:   dd _mm256_set1_d(0x0000001F)
.I32:   dd _mm256_set1_d(0x00000020)
.IS3:   dd _mm512_set1_d(0x9287BE8E)
.IS4:   dd _mm512_set1_d(0x30BF3847)
.IS5:   dd _mm512_set1_d(0xCEF6B200)
.IS6:   dd _mm512_set1_d(0x6D2E2BB9)
.IS7:   dd _mm512_set1_d(0x0B65A572)
.IS8:   dd _mm512_set1_d(0xA99D1F2B)
.IS9:   dd _mm512_set1_d(0x47D498E4)
.IS10:  dd _mm512_set1_d(0xE60C129D)
.IS11:  dd _mm512_set1_d(0x84438C56)
.IS12:  dd _mm512_set1_d(0x227B060F)
.IS13:  dd _mm512_set1_d(0xC0B27FC8)
.IS14:  dd _mm512_set1_d(0x5EE9F981)
.IS15:  dd _mm512_set1_d(0xFD21733A)
.IS16:  dd _mm512_set1_d(0x9B58ECF3)
.IS17:  dd _mm512_set1_d(0x399066AC)
.IS18:  dd _mm512_set1_d(0xD7C7E065)
.IS19:  dd _mm512_set1_d(0x75FF5A1E)
.IS20:  dd _mm512_set1_d(0x1436D3D7)
.IS21:  dd _mm512_set1_d(0xB26E4D90)
.IS22:  dd _mm512_set1_d(0x50A5C749)
.IS23:  dd _mm512_set1_d(0xEEDD4102)
.IS24:  dd _mm512_set1_d(0x8D14BABB)
.IS25:  dd _mm512_set1_d(0x2B4C3474)
.IS0:   dd _mm512_set1_d(0xBF0A8B1D)
