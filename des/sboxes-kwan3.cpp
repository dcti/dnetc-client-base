//
// $Log: sboxes-kwan3.cpp,v $
// Revision 1.4  1998/07/08 23:42:25  remi
// Added support for CliIdentifyModules().
//
// Revision 1.3  1998/06/14 08:27:11  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.2  1998/06/14 08:13:27  friedbait
// 'Log' keywords added to maintain automatic change history
//
//

#if (!defined(lint) && defined(__showids__))
const char *sboxes_kwan3_cpp(void) {
return "@(#)$Id: sboxes-kwan3.cpp,v 1.4 1998/07/08 23:42:25 remi Exp $"; }
#endif

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
sboxes-kwan3.cpp - updated sbox code, using standard gates
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
        unsigned long   x57, x58, x59, x60, x61, x62, x63, x64;
        unsigned long   x65, x66, x67, x68, x69, x70, x71;

        x1 = ~a1;
        x2 = ~a6;
        x3 = a1 ^ x2;
        x4 = x3 ^ a3;
        x5 = a6 | x1;
        x6 = a3 | x5;
        x7 = a6 ^ x6;
        x8 = a4 | x7;
        x9 = x4 ^ x8;
        x10 = x3 | x7;
        x11 = a4 & x10;
        x12 = x7 ^ x11;
        x13 = a2 | x12;
        x14 = x9 ^ x13;
        x15 = x2 & x14;
        x16 = a4 & x5;
        x17 = x15 ^ x16;
        x18 = a3 & x4;
        x19 = x1 ^ x7;
        x20 = a4 & x19;
        x21 = x18 ^ x20;
        x22 = a2 | x21;
        x23 = x17 ^ x22;
        x24 = a5 | x23;
        x25 = x14 ^ x24;
        out4 ^= x25;
        x26 = a3 | x7;
        x27 = x2 ^ x26;
        x28 = x27 ^ x11;
        x29 = x2 | x4;
        x30 = x29 ^ x20;
        x31 = a2 & x30;
        x32 = x28 ^ x31;
        x33 = a3 ^ x7;
        x34 = x6 ^ x18;
        x35 = a4 | x34;
        x36 = x33 ^ x35;
        x37 = a1 & a6;
        x38 = a4 | x37;
        x39 = x6 ^ x38;
        x40 = a2 | x39;
        x41 = x36 ^ x40;
        x42 = a5 | x41;
        x43 = x32 ^ x42;
        out2 ^= x43;
        x44 = a1 & a3;
        x45 = a4 & x44;
        x46 = x9 ^ x45;
        x47 = x15 & x33;
        x48 = a2 | x47;
        x49 = x46 ^ x48;
        x50 = a1 & x28;
        x51 = a4 & x50;
        x52 = x39 ^ x51;
        x53 = x9 | x45;
        x54 = a2 | x53;
        x55 = x52 ^ x54;
        x56 = a5 & x55;
        x57 = x49 ^ x56;
        out3 ^= x57;
        x58 = a3 ^ x27;
        x59 = a6 ^ x34;
        x60 = a4 | x59;
        x61 = x58 ^ x60;
        x62 = a4 | x14;
        x63 = a2 & x62;
        x64 = x61 ^ x63;
        x65 = a4 | x61;
        x66 = x4 ^ x65;
        x67 = a1 & x53;
        x68 = a2 & x67;
        x69 = x66 ^ x68;
        x70 = a5 | x69;
        x71 = x64 ^ x70;
        out1 ^= x71;
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
        unsigned long   x57, x58, x59, x60, x61, x62, x63, x64;

        x1 = ~a5;
        x2 = ~a3;
        x3 = a5 ^ x2;
        x4 = a2 & a4;
        x5 = x3 ^ x4;
        x6 = a5 & a4;
        x7 = a3 ^ x6;
        x8 = a2 & x7;
        x9 = x6 ^ x8;
        x10 = a1 | x9;
        x11 = x5 ^ x10;
        x12 = a2 & x2;
        x13 = a6 ^ x12;
        x14 = a3 & x1;
        x15 = a5 ^ x7;
        x16 = a2 | x15;
        x17 = x14 ^ x16;
        x18 = a1 & x17;
        x19 = x13 ^ x18;
        x20 = a6 & x19;
        x21 = x11 ^ x20;
        out1 ^= x21;
        x22 = a3 | x1;
        x23 = a4 & x22;
        x24 = x3 ^ x23;
        x25 = x24 ^ x12;
        x26 = x25 ^ a1;
        x27 = a3 | x6;
        x28 = x27 ^ x4;
        x29 = a5 & x8;
        x30 = a1 & x29;
        x31 = x28 ^ x30;
        x32 = a6 | x31;
        x33 = x26 ^ x32;
        out2 ^= x33;
        x34 = x7 | x24;
        x35 = x34 ^ a2;
        x36 = a3 | x34;
        x37 = a2 | x36;
        x38 = x26 ^ x37;
        x39 = a1 & x38;
        x40 = x35 ^ x39;
        x41 = x1 & x13;
        x42 = a4 & a3;
        x43 = x1 ^ x42;
        x44 = ~x29;
        x45 = x44 ^ a4;
        x46 = a2 & x45;
        x47 = x43 ^ x46;
        x48 = a1 | x47;
        x49 = x41 ^ x48;
        x50 = a6 & x49;
        x51 = x40 ^ x50;
        out3 ^= x51;
        x52 = a2 & x1;
        x53 = x45 ^ x52;
        x54 = x14 ^ x52;
        x55 = a1 | x54;
        x56 = x53 ^ x55;
        x57 = x1 & x11;
        x58 = a2 | x11;
        x59 = x57 ^ x58;
        x60 = x1 & x21;
        x61 = a1 | x60;
        x62 = x59 ^ x61;
        x63 = a6 & x62;
        x64 = x56 ^ x63;
        out4 ^= x64;
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
        unsigned long   x49, x50, x51, x52, x53, x54, x55, x56;
        unsigned long   x57, x58, x59, x60, x61, x62, x63;

        x1 = ~a3;
        x2 = ~a6;
        x3 = a6 ^ a2;
        x4 = a5 & x1;
        x5 = x3 ^ x4;
        x6 = a4 | a5;
        x7 = x5 ^ x6;
        x8 = a3 ^ x2;
        x9 = a2 & x8;
        x10 = a3 ^ x9;
        x11 = a5 & x10;
        x12 = x10 ^ x11;
        x13 = x2 | x9;
        x14 = x13 ^ a5;
        x15 = a4 | x14;
        x16 = x12 ^ x15;
        x17 = a1 & x16;
        x18 = x7 ^ x17;
        out4 ^= x18;
        x19 = a3 ^ x7;
        x20 = a6 & x7;
        x21 = a5 | x2;
        x22 = x20 ^ x21;
        x23 = a4 & x22;
        x24 = x19 ^ x23;
        x25 = x1 & x12;
        x26 = x9 ^ x20;
        x27 = a4 & x26;
        x28 = x25 ^ x27;
        x29 = a1 | x28;
        x30 = x24 ^ x29;
        out2 ^= x30;
        x31 = x3 ^ x26;
        x32 = a3 | x9;
        x33 = a5 & x32;
        x34 = x31 ^ x33;
        x35 = x3 | x8;
        x36 = x32 & x35;
        x37 = a5 | x36;
        x38 = x35 ^ x37;
        x39 = a4 | x38;
        x40 = x34 ^ x39;
        x41 = a3 ^ x26;
        x42 = a4 & x38;
        x43 = x41 ^ x42;
        x44 = a1 & x43;
        x45 = x40 ^ x44;
        out1 ^= x45;
        x46 = a3 | a6;
        x47 = a2 | x46;
        x48 = x8 ^ x47;
        x49 = x48 ^ a5;
        x50 = a5 | x24;
        x51 = a2 ^ x50;
        x52 = a4 & x51;
        x53 = x49 ^ x52;
        x54 = x34 | x49;
        x55 = a5 & x54;
        x56 = x20 ^ x55;
        x57 = a6 ^ x22;
        x58 = a5 & x57;
        x59 = x48 ^ x58;
        x60 = a4 & x59;
        x61 = x56 ^ x60;
        x62 = a1 | x61;
        x63 = x53 ^ x62;
        out3 ^= x63;
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
        unsigned long   x41, x42, x43, x44, x45;

        x1 = ~a5;
        x2 = ~a4;
        x3 = a5 ^ a4;
        x4 = a4 | x1;
        x5 = x4 ^ a3;
        x6 = a2 | x5;
        x7 = x3 ^ x6;
        x8 = a5 | x5;
        x9 = x2 & x5;
        x10 = a2 & x9;
        x11 = x8 ^ x10;
        x12 = a1 & x11;
        x13 = x7 ^ x12;
        x14 = x1 ^ x5;
        x15 = a5 ^ x13;
        x16 = a2 & x15;
        x17 = x14 ^ x16;
        x18 = x2 | x8;
        x19 = a2 | x9;
        x20 = x18 ^ x19;
        x21 = a1 | x20;
        x22 = x17 ^ x21;
        x23 = a6 | x22;
        x24 = x13 ^ x23;
        out3 ^= x24;
        x25 = a5 | x20;
        x26 = a2 | x25;
        x27 = x17 ^ x26;
        x28 = a4 ^ x18;
        x29 = a1 & x28;
        x30 = x27 ^ x29;
        x31 = x3 | x5;
        x32 = a2 & x31;
        x33 = x15 ^ x32;
        x34 = a4 | x15;
        x35 = x3 | x22;
        x36 = a2 | x35;
        x37 = x34 ^ x36;
        x38 = a1 & x37;
        x39 = x33 ^ x38;
        x40 = a6 & x39;
        x41 = x30 ^ x40;
        out1 ^= x41;
        x42 = a6 | x39;
        x43 = x30 ^ x42;
        out2 ^= x43;
        x44 = a6 & x22;
        x45 = x13 ^ x44;
        out4 ^= x45;
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
        unsigned long   x57, x58, x59, x60, x61, x62, x63, x64;
        unsigned long   x65, x66, x67, x68, x69, x70;

        x1 = ~a1;
        x2 = ~a3;
        x3 = x1 & x2;
        x4 = a4 & a1;
        x5 = x3 ^ x4;
        x6 = x1 | x2;
        x7 = a4 & x6;
        x8 = x6 ^ x7;
        x9 = a2 | x8;
        x10 = x5 ^ x9;
        x11 = a3 & x8;
        x12 = a4 | x3;
        x13 = a2 | x12;
        x14 = x11 ^ x13;
        x15 = a6 & x14;
        x16 = x10 ^ x15;
        x17 = a1 & x7;
        x18 = x8 & x10;
        x19 = a6 & x18;
        x20 = x17 ^ x19;
        x21 = a5 | x20;
        x22 = x16 ^ x21;
        out2 ^= x22;
        x23 = a1 ^ x6;
        x24 = x23 ^ a4;
        x25 = x24 ^ a2;
        x26 = x10 & x14;
        x27 = x26 ^ a2;
        x28 = a6 | x27;
        x29 = x25 ^ x28;
        x30 = x2 ^ x12;
        x31 = a2 | x22;
        x32 = x30 ^ x31;
        x33 = x3 | x8;
        x34 = a2 | x33;
        x35 = x12 ^ x34;
        x36 = a6 & x35;
        x37 = x32 ^ x36;
        x38 = a5 | x37;
        x39 = x29 ^ x38;
        out3 ^= x39;
        x40 = x2 | x26;
        x41 = a2 | x7;
        x42 = x40 ^ x41;
        x43 = a6 & x23;
        x44 = x42 ^ x43;
        x45 = a4 ^ x39;
        x46 = a2 | x45;
        x47 = a3 ^ x46;
        x48 = x10 | x39;
        x49 = a2 | x24;
        x50 = x48 ^ x49;
        x51 = a6 | x50;
        x52 = x47 ^ x51;
        x53 = a5 | x52;
        x54 = x44 ^ x53;
        out4 ^= x54;
        x55 = x6 | x22;
        x56 = a2 | x55;
        x57 = x25 ^ x56;
        x58 = a2 & a1;
        x59 = x10 ^ x58;
        x60 = a6 & x59;
        x61 = x57 ^ x60;
        x62 = a2 & x39;
        x63 = x7 ^ x62;
        x64 = x1 ^ x40;
        x65 = a2 & x10;
        x66 = x64 ^ x65;
        x67 = a6 | x66;
        x68 = x63 ^ x67;
        x69 = a5 | x68;
        x70 = x61 ^ x69;
        out1 ^= x70;
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
        unsigned long   x57, x58, x59, x60, x61, x62, x63;

        x1 = ~a1;
        x2 = ~a4;
        x3 = a1 ^ x2;
        x4 = x3 ^ a6;
        x5 = x4 ^ a2;
        x6 = a6 | x1;
        x7 = a2 & x1;
        x8 = x6 ^ x7;
        x9 = a3 | x8;
        x10 = x5 ^ x9;
        x11 = a4 & x1;
        x12 = a6 & x11;
        x13 = a1 ^ x12;
        x14 = x6 & x10;
        x15 = a2 & x14;
        x16 = x13 ^ x15;
        x17 = a2 | x6;
        x18 = a3 & x17;
        x19 = x16 ^ x18;
        x20 = a5 & x19;
        x21 = x10 ^ x20;
        out3 ^= x21;
        x22 = x4 & x13;
        x23 = x22 ^ a2;
        x24 = a6 & x4;
        x25 = x2 ^ x24;
        x26 = a2 & x25;
        x27 = a1 ^ x26;
        x28 = a3 | x27;
        x29 = x23 ^ x28;
        x30 = a2 & x12;
        x31 = x3 ^ x30;
        x32 = x1 & x25;
        x33 = a3 | x32;
        x34 = x31 ^ x33;
        x35 = a5 | x34;
        x36 = x29 ^ x35;
        out4 ^= x36;
        x37 = a2 ^ x30;
        x38 = a3 & x37;
        x39 = x5 ^ x38;
        x40 = x13 ^ x24;
        x41 = a2 | x40;
        x42 = x8 ^ x41;
        x43 = x10 ^ x24;
        x44 = x43 ^ x17;
        x45 = a3 | x44;
        x46 = x42 ^ x45;
        x47 = a5 | x46;
        x48 = x39 ^ x47;
        out1 ^= x48;
        x49 = a2 & x2;
        x50 = x4 ^ x49;
        x51 = a2 & x40;
        x52 = x6 ^ x51;
        x53 = a3 & x52;
        x54 = x50 ^ x53;
        x55 = x2 | x14;
        x56 = a2 | x55;
        x57 = x12 ^ x56;
        x58 = a1 | a4;
        x59 = x58 ^ x17;
        x60 = a3 & x59;
        x61 = x57 ^ x60;
        x62 = a5 & x61;
        x63 = x54 ^ x62;
        out2 ^= x63;
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
        unsigned long   x57, x58, x59, x60, x61, x62;

        x1 = ~a4;
        x2 = ~a5;
        x3 = a4 ^ a5;
        x4 = a4 & x2;
        x5 = x4 ^ a2;
        x6 = a3 | x5;
        x7 = x3 ^ x6;
        x8 = a2 & x4;
        x9 = x8 ^ a6;
        x10 = a6 & x9;
        x11 = x7 ^ x10;
        x12 = x3 & x5;
        x13 = a3 & x5;
        x14 = x12 ^ x13;
        x15 = a6 & x14;
        x16 = a1 ^ x15;
        x17 = a1 & x16;
        x18 = x11 ^ x17;
        out4 ^= x18;
        x19 = x3 ^ x5;
        x20 = a2 | a4;
        x21 = x4 ^ x20;
        x22 = a3 | x21;
        x23 = x19 ^ x22;
        x24 = a3 & x21;
        x25 = a6 ^ x24;
        x26 = a6 & x25;
        x27 = x23 ^ x26;
        x28 = a2 | x4;
        x29 = a2 | x2;
        x30 = a3 & x29;
        x31 = x28 ^ x30;
        x32 = x3 ^ x20;
        x33 = x32 ^ x22;
        x34 = a6 | x33;
        x35 = x31 ^ x34;
        x36 = a1 & x35;
        x37 = x27 ^ x36;
        out1 ^= x37;
        x38 = x1 ^ x32;
        x39 = a3 & a2;
        x40 = x38 ^ x39;
        x41 = x8 ^ x25;
        x42 = a4 | x35;
        x43 = a3 | x42;
        x44 = x41 ^ x43;
        x45 = a6 & x44;
        x46 = x40 ^ x45;
        x47 = x23 ^ x40;
        x48 = x9 ^ x39;
        x49 = a6 & x48;
        x50 = x47 ^ x49;
        x51 = a1 & x50;
        x52 = x46 ^ x51;
        out2 ^= x52;
        x53 = x25 ^ x50;
        x54 = x28 ^ x37;
        x55 = a6 | x54;
        x56 = x53 ^ x55;
        x57 = a3 & x1;
        x58 = x53 ^ x57;
        x59 = a6 & x58;
        x60 = x47 ^ x59;
        x61 = a1 | x60;
        x62 = x56 ^ x61;
        out3 ^= x62;
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
        unsigned long   x49, x50, x51, x52, x53, x54, x55, x56;
        unsigned long   x57, x58;

        x1 = ~a4;
        x2 = ~a5;
        x3 = x1 & x2;
        x4 = x3 ^ a3;
        x5 = x1 | x2;
        x6 = a2 & x5;
        x7 = x4 ^ x6;
        x8 = a5 | x4;
        x9 = a2 | x8;
        x10 = x3 ^ x9;
        x11 = a1 & x10;
        x12 = x7 ^ x11;
        x13 = x1 & x8;
        x14 = a3 ^ x2;
        x15 = a2 | x14;
        x16 = x13 ^ x15;
        x17 = a4 | x4;
        x18 = a2 | x17;
        x19 = x10 ^ x18;
        x20 = a1 & x19;
        x21 = x16 ^ x20;
        x22 = a6 & x21;
        x23 = x12 ^ x22;
        out4 ^= x23;
        x24 = x5 ^ x8;
        x25 = x24 ^ a2;
        x26 = ~x4;
        x27 = a2 & x26;
        x28 = x14 ^ x27;
        x29 = a1 | x28;
        x30 = x25 ^ x29;
        x31 = x14 ^ x25;
        x32 = a4 & a3;
        x33 = a2 | x32;
        x34 = x31 ^ x33;
        x35 = a1 & x34;
        x36 = a1 ^ x35;
        x37 = a6 | x36;
        x38 = x30 ^ x37;
        out2 ^= x38;
        x39 = x12 ^ x24;
        x40 = a2 | x3;
        x41 = x39 ^ x40;
        x42 = x3 ^ x34;
        x43 = a1 | x42;
        x44 = x41 ^ x43;
        x45 = a6 | x44;
        x46 = x12 ^ x45;
        out1 ^= x46;
        x47 = x10 ^ x32;
        x48 = a2 | x19;
        x49 = x47 ^ x48;
        x50 = a5 & x7;
        x51 = a1 | x50;
        x52 = x49 ^ x51;
        x53 = x16 | x39;
        x54 = a2 ^ x41;
        x55 = a1 & x54;
        x56 = x53 ^ x55;
        x57 = a6 | x56;
        x58 = x52 ^ x57;
        out3 ^= x58;
}
