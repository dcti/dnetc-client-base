#include "sboxes.h"

#ifndef KWAN
#error "You must compile with -DKWAN.  Set this and then recompile cliconfig.cpp"
#endif

/*
 * Generated S-box files.
 *
 * Produced by Matthew Kwan - March 1998
 */

/*
sboxes-kwan4.cpp

Non-standard gates

It turns out that many CPUs these days have instructions for non-standard
logical operations, such as NAND, NOR, NXOR, AND-NOT,
and OR-NOT. It was a fairly minor modification to my algorithm to generate
S-boxes using these gates (although it required a lot more CPU
time). All NAND and NOR gates were then removed so that the code could
run efficiently on SPARC and Alpha architectures. Just for the
hell of it, most of the NXOR and all of the OR-NOT gates we also
removed. The resulting S-boxes had an average of 55.4 gates. This was
later improved to 54.5
*/

void
s1 (
        unsigned long   a1,
        unsigned long   a2,
        unsigned long   a3,
        unsigned long   a4,
        unsigned long   a5,
        unsigned long   a6,
        unsigned long   &out1,
        unsigned long   &out2,
        unsigned long   &out3,
        unsigned long   &out4
) {
        unsigned long   x1, x2, x3, x4, x5, x6, x7, x8;
        unsigned long   x9, x10, x11, x12, x13, x14, x15, x16;
        unsigned long   x17, x18, x19, x20, x21, x22, x23, x24;
        unsigned long   x25, x26, x27, x28, x29, x30, x31, x32;
        unsigned long   x33, x34, x35, x36, x37, x38, x39, x40;
        unsigned long   x41, x42, x43, x44, x45, x46, x47, x48;
        unsigned long   x49, x50, x51, x52, x53, x54, x55, x56;
        unsigned long   x57, x58, x59, x60, x61, x62, x63;

        x1 = a1 ^ ~a6;
        x2 = x1 ^ a4;
        x3 = a6 | x2;
        x4 = a5 & x3;
        x5 = x2 ^ x4;
        x6 = a4 & x1;
        x7 = a6 ^ x6;
        x8 = a1 & x5;
        x9 = a5 & ~x8;
        x10 = x7 ^ x9;
        x11 = a3 & x10;
        x12 = x5 ^ x11;
        x13 = x2 ^ ~x10;
        x14 = a5 & x13;
        x15 = a5 ^ x14;
        x16 = a1 & ~x7;
        x17 = a5 | ~x16;
        x18 = x10 ^ x17;
        x19 = a3 | ~x18;
        x20 = x15 ^ x19;
        x21 = a2 & x20;
        x22 = x12 ^ x21;
        out3 ^= x22;
        x23 = x1 | x15;
        x24 = x23 ^ a5;
        x25 = x1 & x7;
        x26 = a4 & x2;
        x27 = x3 ^ x26;
        x28 = a5 | x27;
        x29 = x25 ^ ~x28;
        x30 = a3 & x29;
        x31 = x24 ^ x30;
        x32 = x29 & ~x13;
        x33 = x27 & ~x8;
        x34 = a3 & x33;
        x35 = x32 ^ x34;
        x36 = a2 | x35;
        x37 = x31 ^ x36;
        out1 ^= x37;
        x38 = a1 | x18;
        x39 = a5 & ~x2;
        x40 = x38 ^ x39;
        x41 = a6 | x14;
        x42 = a3 & x41;
        x43 = x40 ^ ~x42;
        x44 = x1 ^ x29;
        x45 = x44 ^ a5;
        x46 = a1 & ~x12;
        x47 = a5 | x46;
        x48 = x41 ^ x47;
        x49 = a3 & x48;
        x50 = x45 ^ ~x49;
        x51 = a2 | x50;
        x52 = x43 ^ x51;
        out4 ^= x52;
        x53 = x45 & ~x48;
        x54 = a5 & x5;
        x55 = x13 ^ x54;
        x56 = a3 | x55;
        x57 = x53 ^ x56;
        x58 = x22 | x35;
        x59 = x8 | x2;
        x60 = a3 | x59;
        x61 = x58 ^ ~x60;
        x62 = a2 & x61;
        x63 = x57 ^ x62;
        out2 ^= x63;
}


void
s2 (
        unsigned long   a1,
        unsigned long   a2,
        unsigned long   a3,
        unsigned long   a4,
        unsigned long   a5,
        unsigned long   a6,
        unsigned long   &out1,
        unsigned long   &out2,
        unsigned long   &out3,
        unsigned long   &out4
) {
        unsigned long   x1, x2, x3, x4, x5, x6, x7, x8;
        unsigned long   x9, x10, x11, x12, x13, x14, x15, x16;
        unsigned long   x17, x18, x19, x20, x21, x22, x23, x24;
        unsigned long   x25, x26, x27, x28, x29, x30, x31, x32;
        unsigned long   x33, x34, x35, x36, x37, x38, x39, x40;
        unsigned long   x41, x42, x43, x44, x45, x46, x47, x48;
        unsigned long   x49, x50, x51, x52, x53, x54, x55, x56;

        x1 = a1 ^ a5;
        x2 = a3 | ~a6;
        x3 = x1 ^ x2;
        x4 = a1 & ~a6;
        x5 = a5 & x4;
        x6 = a3 ^ x5;
        x7 = a3 & x6;
        x8 = a2 ^ x7;
        x9 = a2 & x8;
        x10 = x3 ^ x9;
        x11 = a5 & ~x2;
        x12 = a6 | x5;
        x13 = x12 | ~a2;
        x14 = x11 ^ x13;
        x15 = a4 & x14;
        x16 = x10 ^ x15;
        out2 ^= x16;
        x17 = a1 ^ x12;
        x18 = a1 & x11;
        x19 = a3 | x18;
        x20 = x17 ^ x19;
        x21 = x10 & ~a1;
        x22 = a2 & ~x21;
        x23 = x20 ^ x22;
        x24 = a5 ^ ~x5;
        x25 = a2 & x24;
        x26 = x24 ^ x25;
        x27 = a4 | x26;
        x28 = x23 ^ x27;
        out1 ^= x28;
        x29 = a1 | a5;
        x30 = a3 | ~x29;
        x31 = x5 ^ x30;
        x32 = x17 & ~x23;
        x33 = a2 & ~x32;
        x34 = x31 ^ x33;
        x35 = x3 ^ ~x34;
        x36 = x4 ^ ~x17;
        x37 = a3 & x36;
        x38 = x35 ^ x37;
        x39 = x12 & ~a3;
        x40 = a2 | x39;
        x41 = x38 ^ x40;
        x42 = a4 & x41;
        x43 = x34 ^ x42;
        out3 ^= x43;
        x44 = a5 ^ x21;
        x45 = a6 ^ x34;
        x46 = a3 & x45;
        x47 = x44 ^ x46;
        x48 = x35 & ~x39;
        x49 = a2 | ~x48;
        x50 = x47 ^ x49;
        x51 = a5 & x17;
        x52 = x29 & ~x17;
        x53 = a2 & x52;
        x54 = x51 ^ ~x53;
        x55 = a4 & x54;
        x56 = x50 ^ x55;
        out4 ^= x56;
}


void
s3 (
        unsigned long   a1,
        unsigned long   a2,
        unsigned long   a3,
        unsigned long   a4,
        unsigned long   a5,
        unsigned long   a6,
        unsigned long   &out1,
        unsigned long   &out2,
        unsigned long   &out3,
        unsigned long   &out4
) {
        unsigned long   x1, x2, x3, x4, x5, x6, x7, x8;
        unsigned long   x9, x10, x11, x12, x13, x14, x15, x16;
        unsigned long   x17, x18, x19, x20, x21, x22, x23, x24;
        unsigned long   x25, x26, x27, x28, x29, x30, x31, x32;
        unsigned long   x33, x34, x35, x36, x37, x38, x39, x40;
        unsigned long   x41, x42, x43, x44, x45, x46, x47, x48;
        unsigned long   x49, x50, x51, x52, x53, x54, x55;

        x1 = a6 ^ a2;
        x2 = a5 & ~a3;
        x3 = x1 ^ x2;
        x4 = a4 | a5;
        x5 = x3 ^ x4;
        x6 = a3 ^ ~a6;
        x7 = a2 & x6;
        x8 = a3 ^ x7;
        x9 = a5 & x8;
        x10 = x8 ^ x9;
        x11 = a6 & ~x7;
        x12 = x11 ^ a5;
        x13 = a4 | ~x12;
        x14 = x10 ^ x13;
        x15 = a1 & x14;
        x16 = x5 ^ x15;
        out4 ^= x16;
        x17 = a3 ^ x5;
        x18 = a6 & ~x3;
        x19 = a5 | ~a6;
        x20 = x18 ^ x19;
        x21 = a4 & x20;
        x22 = x17 ^ x21;
        x23 = x10 & ~a3;
        x24 = x7 ^ x18;
        x25 = a4 & x24;
        x26 = x23 ^ x25;
        x27 = a1 | x26;
        x28 = x22 ^ x27;
        out2 ^= x28;
        x29 = x1 ^ x24;
        x30 = x6 | x29;
        x31 = a5 & x30;
        x32 = x29 ^ x31;
        x33 = x32 ^ a4;
        x34 = x8 ^ x20;
        x35 = a6 | a2;
        x36 = a5 & ~x35;
        x37 = x34 ^ x36;
        x38 = a4 & x37;
        x39 = x34 ^ x38;
        x40 = a1 | x39;
        x41 = x33 ^ x40;
        out1 ^= x41;
        x42 = x6 & x35;
        x43 = x42 ^ a5;
        x44 = a5 | x22;
        x45 = a2 ^ x44;
        x46 = a4 & x45;
        x47 = x43 ^ ~x46;
        x48 = a6 | x37;
        x49 = a5 | ~x46;
        x50 = x48 ^ x49;
        x51 = x3 ^ ~x30;
        x52 = a4 | x51;
        x53 = x50 ^ x52;
        x54 = a1 | x53;
        x55 = x47 ^ x54;
        out3 ^= x55;
}


void
s4 (
        unsigned long   a1,
        unsigned long   a2,
        unsigned long   a3,
        unsigned long   a4,
        unsigned long   a5,
        unsigned long   a6,
        unsigned long   &out1,
        unsigned long   &out2,
        unsigned long   &out3,
        unsigned long   &out4
) {
        unsigned long   x1, x2, x3, x4, x5, x6, x7, x8;
        unsigned long   x9, x10, x11, x12, x13, x14, x15, x16;
        unsigned long   x17, x18, x19, x20, x21, x22, x23, x24;
        unsigned long   x25, x26, x27, x28, x29, x30, x31, x32;
        unsigned long   x33, x34, x35, x36, x37, x38, x39, x40;
        unsigned long   x41;

        x1 = a1 | a3;
        x2 = a5 & x1;
        x3 = a1 ^ x2;
        x4 = a2 | a3;
        x5 = x3 ^ x4;
        x6 = a5 & a1;
        x7 = x1 ^ x6;
        x8 = a2 & x7;
        x9 = a5 ^ x8;
        x10 = a4 & x9;
        x11 = x5 ^ x10;
        x12 = a3 ^ ~x2;
        x13 = a2 & x12;
        x14 = x7 ^ x13;
        x15 = x12 & ~x3;
        x16 = a3 ^ ~a5;
        x17 = a2 | x16;
        x18 = x15 ^ x17;
        x19 = a4 | x18;
        x20 = x14 ^ x19;
        x21 = a6 | x20;
        x22 = x11 ^ x21;
        out1 ^= x22;
        x23 = a6 & x20;
        x24 = x11 ^ ~x23;
        out2 ^= x24;
        x25 = a2 & x9;
        x26 = x15 ^ x25;
        x27 = a3 ^ x9;
        x28 = a2 & x27;
        x29 = a5 ^ ~x28;
        x30 = a4 & x29;
        x31 = x26 ^ x30;
        x32 = x11 ^ x31;
        x33 = a2 & x32;
        x34 = x22 ^ ~x33;
        x35 = a4 | x32;
        x36 = x34 ^ x35;
        x37 = a6 | x36;
        x38 = x31 ^ x37;
        out3 ^= x38;
        x39 = x20 ^ ~x36;
        x40 = a6 & x39;
        x41 = x31 ^ x40;
        out4 ^= x41;
}


void
s5 (
        unsigned long   a1,
        unsigned long   a2,
        unsigned long   a3,
        unsigned long   a4,
        unsigned long   a5,
        unsigned long   a6,
        unsigned long   &out1,
        unsigned long   &out2,
        unsigned long   &out3,
        unsigned long   &out4
) {
        unsigned long   x1, x2, x3, x4, x5, x6, x7, x8;
        unsigned long   x9, x10, x11, x12, x13, x14, x15, x16;
        unsigned long   x17, x18, x19, x20, x21, x22, x23, x24;
        unsigned long   x25, x26, x27, x28, x29, x30, x31, x32;
        unsigned long   x33, x34, x35, x36, x37, x38, x39, x40;
        unsigned long   x41, x42, x43, x44, x45, x46, x47, x48;
        unsigned long   x49, x50, x51, x52, x53, x54, x55, x56;
        unsigned long   x57, x58, x59, x60, x61, x62;

        x1 = a3 & ~a5;
        x2 = a2 ^ x1;
        x3 = a5 & ~a3;
        x4 = a1 | x3;
        x5 = x2 ^ x4;
        x6 = x5 ^ a6;
        x7 = a3 & x5;
        x8 = a5 ^ x7;
        x9 = a1 & x8;
        x10 = a2 ^ x9;
        x11 = x1 | x10;
        x12 = a1 & x11;
        x13 = x7 ^ x12;
        x14 = a6 & x13;
        x15 = x10 ^ x14;
        x16 = a4 | x15;
        x17 = x6 ^ x16;
        out2 ^= x17;
        x18 = x3 | x15;
        x19 = a2 | x7;
        x20 = a1 & ~x19;
        x21 = x18 ^ x20;
        x22 = a2 & ~a5;
        x23 = x2 ^ x18;
        x24 = a1 | x23;
        x25 = x22 ^ ~x24;
        x26 = a6 | x25;
        x27 = x21 ^ x26;
        x28 = x8 & ~x2;
        x29 = x28 ^ a1;
        x30 = a1 & ~x22;
        x31 = x2 ^ x30;
        x32 = a6 | x31;
        x33 = x29 ^ ~x32;
        x34 = a4 | x33;
        x35 = x27 ^ x34;
        out1 ^= x35;
        x36 = a3 | x27;
        x37 = x8 ^ x36;
        x38 = a1 & x21;
        x39 = x37 ^ x38;
        x40 = a1 | x17;
        x41 = x8 ^ x40;
        x42 = a6 | x41;
        x43 = x39 ^ ~x42;
        x44 = a1 & x28;
        x45 = x3 ^ x44;
        x46 = a1 | x2;
        x47 = x22 ^ ~x46;
        x48 = a6 & x47;
        x49 = x45 ^ x48;
        x50 = a4 | x49;
        x51 = x43 ^ x50;
        out4 ^= x51;
        x52 = a1 & x31;
        x53 = x18 ^ ~x52;
        x54 = x23 ^ x38;
        x55 = a6 & x54;
        x56 = x53 ^ x55;
        x57 = x24 | x41;
        x58 = x25 & ~x3;
        x59 = a6 | x58;
        x60 = x57 ^ ~x59;
        x61 = a4 & x60;
        x62 = x56 ^ x61;
        out3 ^= x62;
}


void
s6 (
        unsigned long   a1,
        unsigned long   a2,
        unsigned long   a3,
        unsigned long   a4,
        unsigned long   a5,
        unsigned long   a6,
        unsigned long   &out1,
        unsigned long   &out2,
        unsigned long   &out3,
        unsigned long   &out4
) {
        unsigned long   x1, x2, x3, x4, x5, x6, x7, x8;
        unsigned long   x9, x10, x11, x12, x13, x14, x15, x16;
        unsigned long   x17, x18, x19, x20, x21, x22, x23, x24;
        unsigned long   x25, x26, x27, x28, x29, x30, x31, x32;
        unsigned long   x33, x34, x35, x36, x37, x38, x39, x40;
        unsigned long   x41, x42, x43, x44, x45, x46, x47, x48;
        unsigned long   x49, x50, x51, x52, x53, x54, x55, x56;
        unsigned long   x57;

        x1 = a1 ^ ~a4;
        x2 = a1 | a3;
        x3 = a2 & x2;
        x4 = x1 ^ x3;
        x5 = a1 & ~a3;
        x6 = a6 | x5;
        x7 = x4 ^ x6;
        x8 = a1 | a4;
        x9 = x8 ^ ~a3;
        x10 = a2 & x4;
        x11 = x9 ^ x10;
        x12 = a1 & ~x7;
        x13 = a2 | x12;
        x14 = a4 ^ ~x13;
        x15 = a6 | x14;
        x16 = x11 ^ x15;
        x17 = a5 & x16;
        x18 = x7 ^ ~x17;
        out3 ^= x18;
        x19 = a3 | ~x8;
        x20 = a1 | x1;
        x21 = a2 & x20;
        x22 = x19 ^ x21;
        x23 = a4 & ~x5;
        x24 = a2 & ~x23;
        x25 = x13 ^ ~x24;
        x26 = a6 & x25;
        x27 = x22 ^ x26;
        x28 = x9 ^ ~x22;
        x29 = x28 ^ a2;
        x30 = x21 | ~x13;
        x31 = a6 | x30;
        x32 = x29 ^ x31;
        x33 = a5 | x32;
        x34 = x27 ^ x33;
        out2 ^= x34;
        x35 = x2 ^ x4;
        x36 = a3 | x1;
        x37 = a2 & ~x36;
        x38 = x8 ^ ~x37;
        x39 = a6 | x38;
        x40 = x35 ^ x39;
        x41 = x1 ^ x19;
        x42 = x27 & ~x20;
        x43 = a6 & x42;
        x44 = x41 ^ x43;
        x45 = a5 | x44;
        x46 = x40 ^ x45;
        out4 ^= x46;
        x47 = x11 | x37;
        x48 = x1 & x9;
        x49 = a2 & x41;
        x50 = x48 ^ x49;
        x51 = a6 | x50;
        x52 = x47 ^ ~x51;
        x53 = x20 & x50;
        x54 = a6 & x53;
        x55 = x36 ^ x54;
        x56 = a5 & x55;
        x57 = x52 ^ x56;
        out1 ^= x57;
}


void
s7 (
        unsigned long   a1,
        unsigned long   a2,
        unsigned long   a3,
        unsigned long   a4,
        unsigned long   a5,
        unsigned long   a6,
        unsigned long   &out1,
        unsigned long   &out2,
        unsigned long   &out3,
        unsigned long   &out4
) {
        unsigned long   x1, x2, x3, x4, x5, x6, x7, x8;
        unsigned long   x9, x10, x11, x12, x13, x14, x15, x16;
        unsigned long   x17, x18, x19, x20, x21, x22, x23, x24;
        unsigned long   x25, x26, x27, x28, x29, x30, x31, x32;
        unsigned long   x33, x34, x35, x36, x37, x38, x39, x40;
        unsigned long   x41, x42, x43, x44, x45, x46, x47, x48;
        unsigned long   x49, x50, x51, x52, x53, x54, x55, x56;
        unsigned long   x57;

        x1 = a2 ^ ~a4;
        x2 = a2 ^ a5;
        x3 = a4 & x2;
        x4 = a4 ^ x3;
        x5 = a3 & x4;
        x6 = x1 ^ x5;
        x7 = a2 & x3;
        x8 = a3 & ~x4;
        x9 = x7 ^ ~x8;
        x10 = a1 & x9;
        x11 = x6 ^ x10;
        x12 = a5 & ~x3;
        x13 = x12 ^ x8;
        x14 = x11 & x13;
        x15 = x14 | ~a3;
        x16 = a5 ^ x15;
        x17 = a1 & x16;
        x18 = x13 ^ ~x17;
        x19 = a6 | x18;
        x20 = x11 ^ x19;
        out3 ^= x20;
        x21 = a4 & a2;
        x22 = a5 ^ x21;
        x23 = a3 & x6;
        x24 = x22 ^ x23;
        x25 = a5 ^ x11;
        x26 = a4 ^ x14;
        x27 = a3 & x26;
        x28 = x25 ^ x27;
        x29 = a1 & x28;
        x30 = x24 ^ x29;
        x31 = a3 & ~x6;
        x32 = x3 ^ ~x24;
        x33 = a1 & x32;
        x34 = x31 ^ ~x33;
        x35 = a6 & x34;
        x36 = x30 ^ x35;
        out1 ^= x36;
        x37 = x16 ^ x28;
        x38 = a1 | x37;
        x39 = x24 ^ ~x38;
        x40 = x2 ^ x32;
        x41 = a3 | x40;
        x42 = x3 ^ ~x41;
        x43 = a3 & a2;
        x44 = x7 ^ x43;
        x45 = a1 | x44;
        x46 = x42 ^ x45;
        x47 = a6 & x46;
        x48 = x39 ^ x47;
        out2 ^= x48;
        x49 = a3 | ~x42;
        x50 = x13 ^ x49;
        x51 = x48 & ~x37;
        x52 = a1 & ~x51;
        x53 = x50 ^ x52;
        x54 = a1 & x51;
        x55 = x7 ^ x54;
        x56 = a6 | x55;
        x57 = x53 ^ ~x56;
        out4 ^= x57;
}


void
s8 (
        unsigned long   a1,
        unsigned long   a2,
        unsigned long   a3,
        unsigned long   a4,
        unsigned long   a5,
        unsigned long   a6,
        unsigned long   &out1,
        unsigned long   &out2,
        unsigned long   &out3,
        unsigned long   &out4
) {
        unsigned long   x1, x2, x3, x4, x5, x6, x7, x8;
        unsigned long   x9, x10, x11, x12, x13, x14, x15, x16;
        unsigned long   x17, x18, x19, x20, x21, x22, x23, x24;
        unsigned long   x25, x26, x27, x28, x29, x30, x31, x32;
        unsigned long   x33, x34, x35, x36, x37, x38, x39, x40;
        unsigned long   x41, x42, x43, x44, x45, x46, x47, x48;
        unsigned long   x49, x50, x51, x52;

        x1 = a5 | a4;
        x2 = a3 ^ ~x1;
        x3 = a4 & a5;
        x4 = a2 & ~x3;
        x5 = x2 ^ x4;
        x6 = a5 | x2;
        x7 = a2 | x6;
        x8 = x1 ^ ~x7;
        x9 = a1 & x8;
        x10 = x5 ^ x9;
        x11 = x6 & ~a4;
        x12 = a3 ^ ~a5;
        x13 = a2 | x12;
        x14 = x11 ^ x13;
        x15 = a4 | x2;
        x16 = a2 | x15;
        x17 = x8 ^ x16;
        x18 = a1 & x17;
        x19 = x14 ^ x18;
        x20 = a6 & x19;
        x21 = x10 ^ x20;
        out4 ^= x21;
        x22 = x3 ^ ~x6;
        x23 = a2 | ~x1;
        x24 = x22 ^ x23;
        x25 = x12 ^ ~x22;
        x26 = a2 & x25;
        x27 = x17 ^ ~x26;
        x28 = a1 | x27;
        x29 = x24 ^ x28;
        x30 = x10 ^ ~x29;
        x31 = a6 & x30;
        x32 = x29 ^ x31;
        out1 ^= x32;
        x33 = x1 ^ x22;
        x34 = x33 ^ a2;
        x35 = a5 ^ x10;
        x36 = a2 & ~x5;
        x37 = x35 ^ x36;
        x38 = a1 & x37;
        x39 = x34 ^ x38;
        x40 = x36 & ~a4;
        x41 = x23 ^ ~x26;
        x42 = a1 & x41;
        x43 = x40 ^ x42;
        x44 = a6 & x43;
        x45 = x39 ^ x44;
        out3 ^= x45;
        x46 = x34 ^ x37;
        x47 = a5 & x13;
        x48 = a1 & x47;
        x49 = x46 ^ x48;
        x50 = x42 & ~x27;
        x51 = a6 | x50;
        x52 = x49 ^ x51;
        out2 ^= x52;
}
