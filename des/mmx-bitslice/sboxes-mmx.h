
//
// $Log: sboxes-mmx.h,v $
// Revision 1.4  1998/07/12 05:29:18  fordbr
// Replaced sboxes 1, 2 and 7 with Kwan versions
// Now 1876 kkeys/s on a P5-200MMX
//
// Revision 1.3  1998/07/08 18:51:37  remi
// There is 14 locals in older sboxes, not 15.
//
// Revision 1.2  1998/07/08 15:47:15  remi
// Added $Log
//
//


typedef unsigned long long slice;

// My own way of crippling the code (grin)
// This is to have 64 bits aligned parameters and local variables
typedef struct {
	slice mmNOT;
	slice a1,a2,a3,a4,a5,a6;
	slice i0,i1,i2,i3;
	slice *o0,*o1,*o2,*o3;
	slice locals[14];
} stOldMmxParams;

typedef struct {
	slice mmNOT;
	slice i0,i1,i2,i3;
	slice locals[15];
} stNewMmxParams;

typedef union {
	stOldMmxParams older;
	stNewMmxParams newer;
} stMmxParams;


// how to pass 1 parameter in %eax with GCC
#define REGPARAM __attribute__ ((__regparm__(1)));

extern "C" {

    slice whack16(slice *P, slice *C, slice *K);

    void mmxs1_kwan (stMmxParams *params) REGPARAM;
    void mmxs2_kwan (stMmxParams *params) REGPARAM;
    void mmxs3_kwan (stMmxParams *params) REGPARAM;
    void mmxs4_kwan (stMmxParams *params) REGPARAM;
    void mmxs5_kwan (stMmxParams *params) REGPARAM;
    void mmxs6_kwan (stMmxParams *params) REGPARAM;
    void mmxs7_kwan (stMmxParams *params) REGPARAM;
    void mmxs8_kwan (stMmxParams *params) REGPARAM;
}


