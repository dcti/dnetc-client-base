#include "rc5.h"

#define P     0xB7E15163
#define Q     0x9E3779B9

#define S_not(n)      P+Q*n

class SnotN {
	static SnotN SnotN_;
	SnotN();
};

SnotN	SnotN::SnotN_;

unsigned long		S_not_n[26];

#if signal_debug_980508!=hamajima
#include <stdio.h>
#include <signal.h>

void catchSignal(int sig){
	fprintf(stderr,"catch signal : %d\n", sig);
}
#endif

SnotN::SnotN(){
#if signal_debug_980508!=hamajima
	signal(SIGINT,catchSignal);
	signal(SIGTERM,catchSignal);
	signal(SIGTSTP,catchSignal);
#endif
	S_not_n[0]=0xBF0A8B1D; // =ROTL3(S_not(0))
	S_not_n[1]=0x15235639; // =S_not(1)+S1[0]
	for( int i=2; i<26; i++ )
		S_not_n[i]=S_not(i);
}

extern u32	rc5_unit_func( RC5UnitWork* );

