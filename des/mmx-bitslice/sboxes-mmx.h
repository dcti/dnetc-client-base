
//
// $Log: sboxes-mmx.h,v $
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
	slice locals[15];
} stOldMmxParams;

typedef struct {
	slice mmNOT;
	slice i0,i1,i2,i3;
	slice locals[14];
} stNewMmxParams;

typedef union {
	stOldMmxParams older;
	stNewMmxParams newer;
} stMmxParams;


// how to pass 1 parameter in %eax with GCC
#define REGPARAM __attribute__ ((__regparm__(1)));

extern "C" {

    slice whack16(slice *P, slice *C, slice *K);

    void mmxs1      (stMmxParams *params) REGPARAM;
    void mmxs2      (stMmxParams *params) REGPARAM;
    void mmxs3_kwan (stMmxParams *params) REGPARAM;
    void mmxs4_kwan (stMmxParams *params) REGPARAM;
    void mmxs5_kwan (stMmxParams *params) REGPARAM;
    void mmxs6_kwan (stMmxParams *params) REGPARAM;
    void mmxs7      (stMmxParams *params) REGPARAM;
    void mmxs8_kwan (stMmxParams *params) REGPARAM;
}


