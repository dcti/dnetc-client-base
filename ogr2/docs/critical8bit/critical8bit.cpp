// $Id: critical8bit.cpp,v 1.1.2.1 2001/04/01 16:38:27 andreasb Exp $

#include <stdio.h>

static const int OGR_length[] = { /* use: OGR_length[depth] */
/* marks */
  /*  1 */    0,   1,   3,   6,  11,  17,  25,  34,  44,  55,
  /* 11 */   72,  85, 106, 127, 151, 177, 199, 216, 246, 283,
  /* 21 */  333, 356, 372, 425, 480, 492, 553, 585, 623,
  /* http://www.research.ibm.com/people/s/shearer/grtab.html */
  /* 30 */ 680, 747, 784, 859, 938, 987, 1005, 1099, 1146, 1252, 1282
};


void test8bitcritical(int marks) {
  int depth = marks-1; 
  int ld, md = 1, rd; // ld+md+rd = depth;
  int ll, ml, rl;     // ll+ml+rl = ogr_length[depth]
  int mindiffsum_depthm1 = (depth * (depth - 1))/2; // sum (i = 0; i <= depth-1) i
  
  if (OGR_length[depth] - mindiffsum_depthm1 > 255)
    printf("!!! OGR-%02d %3d(%2d) Critical length: %d   mindiffsum(depth-1) = %d\n", 
           marks, OGR_length[depth], depth,
           OGR_length[depth] - mindiffsum_depthm1, mindiffsum_depthm1);
  else
    printf("!!! OGR-%02d %3d(%2d) not critical   mindiffsum(depth-1) = %d   rest = %d\n",
           marks, OGR_length[depth], depth, 
           mindiffsum_depthm1, OGR_length[depth] - mindiffsum_depthm1);
  
  for (ld = 0; ld + md <= depth; ++ld) {
    rd = depth - ld - md;
    if (rd < ld)
      break;
    ll = OGR_length[ld];
    rl = OGR_length[rd];
    if (ld == rd)
      ++rl;
    ml = OGR_length[depth] - ll - rl;
    
    printf("OGR-%02d %3d(%2d) = %3d(%2d) - %3d(%d) - %3d(%2d) %s\n",
           marks, OGR_length[depth], depth, ll, ld, ml, md, rl, rd, 
           ( ml > 255 ? "!!!" : ""));
  }
  
}

int main() 
{
  for (int m = 20; m <= 40; ++m)
    test8bitcritical(m);
}
