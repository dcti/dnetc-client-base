//	RC5-64 core decrypt for Alpha-AXP CPU's
//	Copyright (c) Mike Marcelais, 1998
//	All Commercial Rights Reserved.
//	Uses NT Calling conventions

// define NT register names
#define v0    $0
#define t0    $1
#define t1    $2
#define t2    $3
#define t3    $4
#define t4    $5
#define t5    $6
#define t6    $7
#define t7    $8
#define s0    $9
#define s1   $10
#define s2   $11
#define s3   $12
#define s4   $13
#define s5   $14
#define s6   $15
#define a0   $16
#define a1   $17
#define a2   $18
#define a3   $19
#define a4   $20
#define a5   $21
#define t8   $22
#define t9   $23
#define t10  $24
#define t11  $25
#define ra   $26
#define t12  $27
#define at   $28
#define gp   $29
#define sp   $30
#define zero $31



     .text
     .globl rc5_unit_func
     .ent   rc5_unit_func
     .frame sp,272,ra
     .set   noat

rc5_unit_func:
//   On entry:
//      a0 --> RC5UnitWork
//             0(a0)  = plaintext
//             8(a0)  = cyphertext
//             16(a0) = L0
//      a1 --> Timeslice
//   On exit:
//      v0 = 0 if no match found
//         = "initial key + timeslice*2 - v0" is the found key
//
//   Also note, under Windows NT, the Alpha processor
//   runs in "little-endian" mode (ie, the same byte
//   order as x86 processors), not "big-endian" mode
//   (which is what many unix OS's for AlphaAXP seem
//   to use).  This core will not work on a "big-endian"
//   Alpha machine.

//   Be sure to use the /O1 switch when compiling with
//   asaxp, so that the assembler will do instruction
//   reordering after the macros are expanded.

     lda    sp,-288(sp)             // Save registers
     stq    ra,0(sp)
     stq    s0,8(sp)
     stq    s1,16(sp)
     stq    s2,24(sp)
     stq    s3,32(sp)
     stq    s4,40(sp)
     stq    s5,48(sp)
     stq    s6,56(sp)
     .prologue
     sll    a1,0x1,v0              // Convert timeslice to keycount
     stq    v0,272(sp)             // Save away for final answer

     // We now have all of the registers to work with

// Two pipe approach, using memory to slowly cache the S array
// into and out of memory

// Register allocation
//   v0 = key counter
//   a0 = pRC5Work
//   t0 = S[0]
//   t1-t12 = S work registers
//   s0-s2,s4-s6 = scratch registers
//   a1-a4 = L work registers
//   s3,a5,ra,at = Decode work registers
//   sp = Stack pointer
//   gp = Global pointer (can't change)
//   zero = Zero register (can't change)

     ldl    a3,16(a0)
     ldl    a4,20(a0)

NextKey:
     ldah   a1,0x0100(a3)          // Second Pipe = +1 of first pipe

     ldah   t0,-0x40F5(zero)
     lda    t0,-0x74E3(t0)         // S[0] = P rotl 3 = 0xBF0A8B1D

     addq   a4,t0,a4               // New L[0]
     zap    a4,0xF0,t11            // Clear garbage
     sll    a4,0x1D,at             // Shift left
     srl    t11,0x3,t12            // Shift right
     bis    t12,at,a4              // And combine
     bis    t12,at,a2              // Pipe 2's value same as pipe 1

// Testing showed that using precalculated P+nQ values and adding
// them using ldah/lda proved faster than calculating that array
// at runtime by using addq instructions.  Plus it saves two registers
// that I may find a use for later
/////////////////////////////////////////////////////////////////////
#define ROUND1(Aold, Bold, Anew, Bnew, Scr, PnQh, PnQl)             \
     addq   Aold,Bold,Anew;                                         \
     ldah   Anew,PnQh(Anew);                                        \
     lda    Anew,PnQl(Anew);        /* Anew = S[i]+A+B         */   \
     zap    Anew,0xF0,Anew;         /* Remove garbage bits     */   \
     srl    Anew,29,Scr;            /* Rotl3                   */   \
     s8addl Anew,Scr,Anew;                                          \
     addq   Bold,Anew,Scr;          /* Get shift count         */   \
     addq   Bnew,Scr,Bnew;                                          \
     and    Scr,31,Scr;             /* Mask for shift left     */   \
     zap    Bnew,0xF0,Bnew;         /* Remove garbage bits     */   \
     sll    Bnew,Scr,Scr;           /* Shift                   */   \
     srl    Scr,32,Bnew;                                            \
     bis    Scr,Bnew,Bnew;                                          \
/////////////////////////////////////////////////////////////////////

// Testing shows that rewriting this to do a single stq is
// noticably slower
/////////////////////////////////////////////////////////////////////
#define SAVEVALUES(A, B, loc)                                       \
     stl    A,loc(sp);                                              \
     stl    B,loc+4(sp);                                            \
/////////////////////////////////////////////////////////////////////

// Testing shows that this is faster than ldq A,loc(sp); sll A,32,B
/////////////////////////////////////////////////////////////////////
#define LOADVALUES(A, B, loc)                                       \
     ldl    A,loc(sp);                                              \
     ldl    B,loc+4(sp);                                            \
/////////////////////////////////////////////////////////////////////

     ROUND1(t0 ,a4,t1 ,a3, s0,  0x5619, -0x34E4)     ROUND1(t0 ,a2,t2 ,a1, s4,  0x5619, -0x34E4)
     SAVEVALUES(t1, t2,  72)

     ROUND1(t1 ,a3,t3 ,a4, s1, -0x0BB0,  0x44D5)     ROUND1(t2 ,a1,t4 ,a2, s5, -0x0BB0,  0x44D5)
     SAVEVALUES(t3, t4,  80)

     ROUND1(t3 ,a4,t5 ,a3, s2, -0x6D78, -0x4172)     ROUND1(t4 ,a2,t6 ,a1, s6, -0x6D78, -0x4172)
     SAVEVALUES(t5, t6,  88)

     ROUND1(t5 ,a3,t7 ,a4, s0,  0x30BF,  0x3847)     ROUND1(t6 ,a1,t8 ,a2, s4,  0x30BF,  0x3847)
     SAVEVALUES(t7, t8,  96)

     ROUND1(t7 ,a4,t9 ,a3, s1, -0x3109, -0x4E00)     ROUND1(t8 ,a2,t10,a1, s5, -0x3109, -0x4E00)
     SAVEVALUES(t9, t10,104)

     ROUND1(t9 ,a3,t11,a4, s2,  0x6D2E,  0x2BB9)     ROUND1(t10,a1,t12,a2, s6,  0x6D2E,  0x2BB9)
     SAVEVALUES(t11,t12,112)

     ROUND1(t11,a4,t1 ,a3, s0,  0x0B66, -0x5A8E)     ROUND1(t12,a2,t2 ,a1, s4,  0x0B66, -0x5A8E)
     SAVEVALUES(t1, t2, 120)

     ROUND1(t1 ,a3,t3 ,a4, s1, -0x5663,  0x1F2B)     ROUND1(t2 ,a1,t4 ,a2, s5, -0x5663,  0x1F2B)
     SAVEVALUES(t3, t4, 128)

     ROUND1(t3 ,a4,t5 ,a3, s2,  0x47D5, -0x671C)     ROUND1(t4 ,a2,t6 ,a1, s6,  0x47D5, -0x671C)
     SAVEVALUES(t5, t6, 136)

     ROUND1(t5 ,a3,t7 ,a4, s0, -0x19F4,  0x129D)     ROUND1(t6 ,a1,t8 ,a2, s4, -0x19F4,  0x129D)
     SAVEVALUES(t7, t8, 144)

     ROUND1(t7 ,a4,t9 ,a3, s1, -0x7BBC, -0x73AA)     ROUND1(t8 ,a2,t10,a1, s5, -0x7BBC, -0x73AA)
     SAVEVALUES(t9, t10,152)

     ROUND1(t9 ,a3,t11,a4, s2,  0x227B,  0x060F)     ROUND1(t10,a1,t12,a2, s6,  0x227B,  0x060F)
     SAVEVALUES(t11,t12,160)

     ROUND1(t11,a4,t1 ,a3, s0, -0x3F4E,  0x7FC8)     ROUND1(t12,a2,t2 ,a1, s4, -0x3F4E,  0x7FC8)
     SAVEVALUES(t1, t2, 168)

     ROUND1(t1 ,a3,t3 ,a4, s1,  0x5EEA, -0x067F)     ROUND1(t2 ,a1,t4 ,a2, s5,  0x5EEA, -0x067F)
     SAVEVALUES(t3, t4, 176)

     ROUND1(t3 ,a4,t5 ,a3, s2, -0x02DF,  0x733A)     ROUND1(t4 ,a2,t6 ,a1, s6, -0x02DF,  0x733A)
     SAVEVALUES(t5, t6, 184)

     ROUND1(t5 ,a3,t7 ,a4, s0, -0x64A7, -0x130D)     ROUND1(t6 ,a1,t8 ,a2, s4, -0x64A7, -0x130D)
     SAVEVALUES(t7, t8, 192)

     ROUND1(t7 ,a4,t9 ,a3, s1,  0x3990,  0x66AC)     ROUND1(t8 ,a2,t10,a1, s5,  0x3990,  0x66AC)
     SAVEVALUES(t9, t10,200)

     ROUND1(t9 ,a3,t11,a4, s2, -0x2838, -0x1F9B)     ROUND1(t10,a1,t12,a2, s6, -0x2838, -0x1F9B)
     SAVEVALUES(t11,t12,208)

     ROUND1(t11,a4,t1 ,a3, s0,  0x75FF,  0x5A1E)     ROUND1(t12,a2,t2 ,a1, s4,  0x75FF,  0x5A1E)
     SAVEVALUES(t1, t2, 216)

     ROUND1(t1 ,a3,t3 ,a4, s1,  0x1437, -0x2C29)     ROUND1(t2 ,a1,t4 ,a2, s5,  0x1437, -0x2C29)
     SAVEVALUES(t3, t4, 224)

     ROUND1(t3 ,a4,t5 ,a3, s2, -0x4D92,  0x4D90)     ROUND1(t4 ,a2,t6 ,a1, s6, -0x4D92,  0x4D90)
     SAVEVALUES(t5, t6, 232)

     ROUND1(t5 ,a3,t7 ,a4, s0,  0x50A6, -0x38B7)     ROUND1(t6 ,a1,t8 ,a2, s4,  0x50A6, -0x38B7)
     SAVEVALUES(t7, t8, 240)

     ROUND1(t7 ,a4,t9 ,a3, s1, -0x1123,  0x4102)     ROUND1(t8 ,a2,t10,a1, s5, -0x1123,  0x4102)
     SAVEVALUES(t9, t10,248)

     ROUND1(t9 ,a3,t11,a4, s2, -0x72EB, -0x4545)     ROUND1(t10,a1,t12,a2, s6, -0x72EB, -0x4545)
     SAVEVALUES(t11,t12,256)

     ROUND1(t11,a4,t1 ,a3, s0,  0x2B4C,  0x3474)     ROUND1(t12,a2,t2 ,a1, s4,  0x2B4C,  0x3474)
     SAVEVALUES(t1, t2, 264)

// Second round -- same as the first, except that the entries already
// in the S table should be used, rather than the P+nQ formula

/////////////////////////////////////////////////////////////////////
#define ROUND2(Aold, Bold, Anew, Bnew, Scr)                         \
     addq   Aold,Anew,Anew;         /* Anew += A+B */               \
     addq   Bold,Anew,Anew;                                         \
     zap    Anew,0xF0,Anew;         /* Remove garbage bits     */   \
     srl    Anew,29,Scr;            /* Rotl3                   */   \
     s8addl Anew,Scr,Anew;                                          \
     addq   Bold,Anew,Scr;          /* Get shift count         */   \
     addq   Bnew,Scr,Bnew;                                          \
     and    Scr,31,Scr;             /* Mask for shift left     */   \
     zap    Bnew,0xF0,Bnew;         /* Remove garbage bits     */   \
     sll    Bnew,Scr,Scr;           /* Shift                   */   \
     srl    Scr,32,Bnew;                                            \
     bis    Scr,Bnew,Bnew;                                          \
/////////////////////////////////////////////////////////////////////

     bis    t0,zero,s3;
     ROUND2(t1, a3,t0,  a4, s1)     ROUND2(t2, a1,s3, a2, s5)
     SAVEVALUES(t0, s3,  64)

     LOADVALUES(t3, t4,  72)
     ROUND2(t0, a4,t3,  a3, s2)     ROUND2(s3, a2,t4, a1, s6)
     SAVEVALUES(t3, t4,  72)

     LOADVALUES(t5, t6,  80)
     ROUND2(t3, a3,t5,  a4, s0)     ROUND2(t4, a1,t6, a2, s4)
     SAVEVALUES(t5, t6,  80)

     LOADVALUES(t7, t8,  88)
     ROUND2(t5, a4,t7,  a3, s1)     ROUND2(t6, a2,t8, a1, s5)
     SAVEVALUES(t7, t8,  88)

     LOADVALUES(t9, t10, 96)
     ROUND2(t7, a3,t9,  a4, s2)     ROUND2(t8, a1,t10,a2, s6)
     SAVEVALUES(t9, t10, 96)

     LOADVALUES(t11,t12,104)
     ROUND2(t9, a4,t11, a3, s0)     ROUND2(t10,a2,t12,a1, s4)
     SAVEVALUES(t11,t12,104)

     LOADVALUES(t1, t2, 112)
     ROUND2(t11,a3,t1,  a4, s1)     ROUND2(t12,a1,t2, a2, s5)
     SAVEVALUES(t1, t2, 112)

     LOADVALUES(t3, t4, 120)
     ROUND2(t1, a4,t3,  a3, s2)     ROUND2(t2, a2,t4, a1, s6)
     SAVEVALUES(t3, t4, 120)

     LOADVALUES(t5, t6, 128)
     ROUND2(t3, a3,t5,  a4, s0)     ROUND2(t4, a1,t6, a2, s4)
     SAVEVALUES(t5, t6, 128)

     LOADVALUES(t7, t8, 136)
     ROUND2(t5, a4,t7,  a3, s1)     ROUND2(t6, a2,t8, a1, s5)
     SAVEVALUES(t7, t8, 136)

     LOADVALUES(t9, t10,144)
     ROUND2(t7, a3,t9,  a4, s2)     ROUND2(t8, a1,t10,a2, s6)
     SAVEVALUES(t9, t10,144)

     LOADVALUES(t11,t12,152)
     ROUND2(t9, a4,t11, a3, s0)     ROUND2(t10,a2,t12,a1, s4)
     SAVEVALUES(t11,t12,152)

     LOADVALUES(t1, t2, 160)
     ROUND2(t11,a3,t1,  a4, s1)     ROUND2(t12,a1,t2, a2, s5)
     SAVEVALUES(t1, t2, 160)

     LOADVALUES(t3, t4, 168)
     ROUND2(t1, a4,t3,  a3, s2)     ROUND2(t2, a2,t4, a1, s6)
     SAVEVALUES(t3, t4, 168)

     LOADVALUES(t5, t6, 176)
     ROUND2(t3, a3,t5,  a4, s0)     ROUND2(t4, a1,t6, a2, s4)
     SAVEVALUES(t5, t6, 176)

     LOADVALUES(t7, t8, 184)
     ROUND2(t5, a4,t7,  a3, s1)     ROUND2(t6, a2,t8, a1, s5)
     SAVEVALUES(t7, t8, 184)

     LOADVALUES(t9, t10,192)
     ROUND2(t7, a3,t9,  a4, s2)     ROUND2(t8, a1,t10,a2, s6)
     SAVEVALUES(t9, t10,192)

     LOADVALUES(t11,t12,200)
     ROUND2(t9, a4,t11, a3, s0)     ROUND2(t10,a2,t12,a1, s4)
     SAVEVALUES(t11,t12,200)

     LOADVALUES(t1, t2, 208)
     ROUND2(t11,a3,t1,  a4, s1)     ROUND2(t12,a1,t2, a2, s5)
     SAVEVALUES(t1, t2, 208)

     LOADVALUES(t3, t4, 216)
     ROUND2(t1, a4,t3,  a3, s2)     ROUND2(t2, a2,t4, a1, s6)
     SAVEVALUES(t3, t4, 216)

     LOADVALUES(t5, t6, 224)
     ROUND2(t3, a3,t5,  a4, s0)     ROUND2(t4, a1,t6, a2, s4)
     SAVEVALUES(t5, t6, 224)

     LOADVALUES(t7, t8, 232)
     ROUND2(t5, a4,t7,  a3, s1)     ROUND2(t6, a2,t8, a1, s5)
     SAVEVALUES(t7, t8, 232)

     LOADVALUES(t9, t10,240)
     ROUND2(t7, a3,t9,  a4, s2)     ROUND2(t8, a1,t10,a2, s6)
     SAVEVALUES(t9, t10,240)

     LOADVALUES(t11,t12,248)
     ROUND2(t9, a4,t11, a3, s0)     ROUND2(t10,a2,t12,a1, s4)
     SAVEVALUES(t11,t12,248)

     LOADVALUES(t1, t2, 256)
     ROUND2(t11,a3,t1,  a4, s1)     ROUND2(t12,a1,t2, a2, s5)
     SAVEVALUES(t1, t2, 256)

     LOADVALUES(t3, t4, 264)
     ROUND2(t1, a4,t3,  a3, s2)     ROUND2(t2, a2,t4, a1, s6)
     SAVEVALUES(t3, t4, 264)

// Round 3, basically the same as round 2, but we are also
// doing decyrption at the same time.

/////////////////////////////////////////////////////////////////////
#define ROUND3(Sa, Sb, A, B, Scr)                                   \
     xor    A,B,A;                  /* A = A ^ B               */   \
     and    B,31,Scr;               /* Shift A left B bits     */   \
     zap    A,0xF0,A;               /* Discard useless bits    */   \
     sll    A,Scr,Scr;                                              \
     srl    Scr,32,A;                                               \
     bis    Scr,A,A;                                                \
     addq   A,Sa,A;                 /* Add S[2*i]              */   \
     xor    A,B,B;                  /* B = A ^ B               */   \
     and    A,31,Scr;               /* Shift B left A bits     */   \
     zap    B,0xF0,B;               /* Discard useless bits    */   \
     sll    B,Scr,Scr;                                              \
     srl    Scr,32,B;                                               \
     bis    B,Scr,B;                                                \
     addq   B,Sb,B;                 /* Add S[2*i+1]            */   \
/////////////////////////////////////////////////////////////////////

     LOADVALUES(t7, t8,  64)
     ROUND2(t3 ,a3,t7 ,a4, s1)     ROUND2(t4 ,a1,t8 ,a2, s5)

     LOADVALUES(t9, t10, 72)
     ROUND2(t7 ,a4,t9 ,a3, s2)     ROUND2(t8 ,a2,t10,a1, s6)

// Pipe 1 uses s3/at; Pipe 2 uses a5/ra as the A/B decoding values

     ldl    s0,0(a0)            // Fetch pRC->pt
     ldl    s4,4(a0)

     addq   t7,s4,s3            // A (decode) = PT(high)+S[0]
     addq   t8,s4,a5

     addq   t9,s0,at            // B (decode) = PT(low)+S[1]
     addq   t10,s0,ra

     LOADVALUES(t11,t12, 80)
     ROUND2(t9 ,a3,t11,a4, s1)     ROUND2(t10,a1,t12,a2, s5)

     LOADVALUES(t1 ,t2,  88)
     ROUND2(t11,a4,t1 ,a3, s2)     ROUND2(t12,a2,t2 ,a1, s6)

     ROUND3(t11,t1 ,s3, at, s0)    ROUND3(t12,t2 ,a5, ra, s4)

     LOADVALUES(t3, t4,  96)
     ROUND2(t1 ,a3,t3, a4, s1)     ROUND2(t2 ,a1,t4, a2, s5)

     LOADVALUES(t5, t6, 104)
     ROUND2(t3 ,a4,t5, a3, s2)     ROUND2(t4 ,a2,t6, a1, s6)

     ROUND3(t3, t5 ,s3, at, s0)    ROUND3(t4, t6 ,a5, ra, s4)

     LOADVALUES(t7, t8, 112)
     ROUND2(t5 ,a3,t7, a4, s1)     ROUND2(t6 ,a1,t8, a2, s5)

     LOADVALUES(t9, t10,120)
     ROUND2(t7 ,a4,t9 ,a3, s2)     ROUND2(t8 ,a2,t10,a1, s6)

     ROUND3(t7, t9 ,s3, at, s0)    ROUND3(t8, t10,a5, ra, s4)

     LOADVALUES(t11,t12,128)
     ROUND2(t9 ,a3,t11,a4, s1)     ROUND2(t10,a1,t12,a2, s5)

     LOADVALUES(t1 ,t2, 136)
     ROUND2(t11,a4,t1 ,a3, s2)     ROUND2(t12,a2,t2 ,a1, s6)

     ROUND3(t11,t1 ,s3, at, s0)    ROUND3(t12,t2 ,a5, ra, s4)

     LOADVALUES(t3, t4, 144)
     ROUND2(t1 ,a3,t3, a4, s1)     ROUND2(t2 ,a1,t4, a2, s5)

     LOADVALUES(t5, t6, 152)
     ROUND2(t3 ,a4,t5, a3, s2)     ROUND2(t4 ,a2,t6, a1, s6)

     ROUND3(t3, t5 ,s3, at, s0)    ROUND3(t4, t6 ,a5, ra, s4)

     LOADVALUES(t7, t8, 160)
     ROUND2(t5 ,a3,t7, a4, s1)     ROUND2(t6 ,a1,t8, a2, s5)

     LOADVALUES(t9, t10,168)
     ROUND2(t7 ,a4,t9 ,a3, s2)     ROUND2(t8 ,a2,t10,a1, s6)

     ROUND3(t7, t9 ,s3, at, s0)    ROUND3(t8, t10,a5, ra, s4)

     LOADVALUES(t11,t12,176)
     ROUND2(t9 ,a3,t11,a4, s1)     ROUND2(t10,a1,t12,a2, s5)

     LOADVALUES(t1 ,t2, 184)
     ROUND2(t11,a4,t1 ,a3, s2)     ROUND2(t12,a2,t2 ,a1, s6)

     ROUND3(t11,t1 ,s3, at, s0)    ROUND3(t12,t2 ,a5, ra, s4)

     LOADVALUES(t3, t4, 192)
     ROUND2(t1 ,a3,t3, a4, s1)     ROUND2(t2 ,a1,t4, a2, s5)

     LOADVALUES(t5, t6, 200)
     ROUND2(t3 ,a4,t5, a3, s2)     ROUND2(t4 ,a2,t6, a1, s6)

     ROUND3(t3, t5 ,s3, at, s0)    ROUND3(t4, t6 ,a5, ra, s4)

     LOADVALUES(t7, t8, 208)
     ROUND2(t5 ,a3,t7, a4, s1)     ROUND2(t6 ,a1,t8, a2, s5)

     LOADVALUES(t9, t10,216)
     ROUND2(t7 ,a4,t9 ,a3, s2)     ROUND2(t8 ,a2,t10,a1, s6)

     ROUND3(t7, t9 ,s3, at, s0)    ROUND3(t8, t10,a5, ra, s4)

     LOADVALUES(t11,t12,224)
     ROUND2(t9 ,a3,t11,a4, s1)     ROUND2(t10,a1,t12,a2, s5)

     LOADVALUES(t1 ,t2, 232)
     ROUND2(t11,a4,t1 ,a3, s2)     ROUND2(t12,a2,t2 ,a1, s6)

     ROUND3(t11,t1 ,s3, at, s0)    ROUND3(t12,t2 ,a5, ra, s4)

     LOADVALUES(t3, t4, 240)
     ROUND2(t1 ,a3,t3, a4, s1)     ROUND2(t2 ,a1,t4, a2, s5)

     LOADVALUES(t5, t6, 248)
     ROUND2(t3 ,a4,t5, a3, s2)     ROUND2(t4 ,a2,t6, a1, s6)

     ROUND3(t3, t5 ,s3, at, s0)    ROUND3(t4, t6 ,a5, ra, s4)

     LOADVALUES(t7, t8, 256)
     ROUND2(t5 ,a3,t7, a4, s1)     ROUND2(t6 ,a1,t8, a2, s5)

     ldl    s2,8(a0)            // Fetch pRC->ct
     ldl    s6,12(a0)           // S2 = expB, S6 = expA

/* First half of ROUND3, pipe 1   (t7, t9, s3, at, s0) */
     xor    s3,at,s3
     and    at,31,s0
     zap    s3,0xF0,s3
     sll    s3,s0,s0
     srl    s0,32,s3
     bis    s0,s3,s3
     addq   s3,t7,s3

/* First half of ROUND3, pipe 2  (t8, t10, a5, ra, s4) */
     xor    a5,ra,a5
     and    ra,31,s4
     zap    a5,0xF0,a5
     sll    a5,s4,s4
     srl    s4,32,a5
     bis    s4,a5,a5
     addq   a5,t8,a5

     subl   s3,s6,s3
     subl   a5,s6,a5
     beq    s3,Possible1
     beq    a5,Possible2

Failure:
     ldl    t5,16(a0)           // Fetch next key
     ldl    a4,20(a0)
     lda    v0,-2(v0)

     ldah   a3,0x0200(t5)
     zapnot a3,0x08,t0
     beq    t0,Carry

NoCarry:
     stl    a3,16(a0)              // Update memory copy     
     beq    v0,EndSearch
     br     NextKey

EndSearch:
     ldq    s6,56(sp)
     ldq    s5,48(sp)
     ldq    s4,40(sp)
     ldq    s3,32(sp)
     ldq    s2,24(sp)
     ldq    s1,16(sp)
     ldq    s0,8(sp)
     ldq    ra,0(sp)

// Adjust return value to be consistant with other cores
// return values
     ldq    at,272(sp)
     subl   at,v0,v0

     lda    sp,288(sp)
     ret    ra

Carry:
     ldah   a3,0x0001(a3)
     zapnot a3,0x04,t0
     zap    a3,0xF8,a3
     bne    t0,NoCarry

     lda    a3,0x0100(a3)
     zapnot a3,0x02,t0
     zap    a3,0xFC,a3
     bne    t0,NoCarry

     lda    a3,0x0001(a3)
     zapnot a3,0x01,t0
     zap    a3,0xFE,a3
     bne    t0,NoCarry

// Major carry -- very rare
     ldah   a4,0x0100(a4)
     zapnot a4,0x08,t0
     bne    t0,DoneMajorCarry

     ldah   a4,0x0001(a4)
     zapnot a4,0x04,t0
     zap    a4,0xF8,a4
     bne    t0,DoneMajorCarry

     lda    a4,0x0100(a4)
     zapnot a4,0x02,t0
     zap    a4,0xFC,a4
     bne    t0,DoneMajorCarry

     lda    a4,0x0001(a4)
     zap    a4,0xFE,a4

DoneMajorCarry:
     stl    a4,20(a0)
     br     NoCarry

Possible1:
/* Possible Solution -- Pipe 1 */
     LOADVALUES(t9, t10,264)

/* First half of ROUND2, pipe 1 (t7, a4, t9, a3, s0) */
     addq   t7,t9,t9
     addq   a4,t9,t9
     zap    t9,0xF0,t9
     srl    t9,29,s0
     s8addl t9,s0,t9

/* Second half of ROUND3, pipe 1 (t7, t9, s6, at, s1) */
     xor    s6,at,at
     and    s6,31,s1
     zap    at,0xF0,at
     sll    at,s1,s1
     srl    s1,32,at
     bis    at,s1,at
     addq   at,t9,at

     subl   at,s2,at
     beq    at,EndSearch
     bne    a5,Failure    // Was pipe 2 also a possible solution?

Possible2:
/* Possible Solution -- Pipe 2 */
     LOADVALUES(t9, t10,264)

/* First half of ROUND2, pipe 2 (t8, a2, t10,a1, s4) */
     addq   t8,t10,t10
     addq   a2,t10,t10
     zap    t10,0xF0,t10
     srl    t10,29,s4
     s8addl t10,s4,t10

/* Second half of ROUND3, pipe 1 (t8, t10, s6, ra, s5) */
     xor    s6,ra,ra
     and    s6,31,s5
     zap    ra,0xF0,ra
     sll    ra,s5,s5
     srl    s5,32,ra
     bis    ra,s5,ra
     addq   ra,t10,ra

     subl   ra,s2,ra
     bne    ra,Failure
     lda    v0,-1(v0)      // Offset keyvalue by one, b/c found on second pipe
     br     EndSearch

     .end   rc5_unit_func
