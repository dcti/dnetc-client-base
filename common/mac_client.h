// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: mac_client.h,v $
// Revision 1.2  1998/12/09 07:19:20  dicamillo
// Created
//

#if (CLIENT_OS == OS_MACOS)

class Mac_Client : public Client
{
unsigned long file_dates[4];

public:
	int InitializeClient(void);
	void ResetClient(void);
	void DeInitializeClient(void);
	double TimeCore(u32 numk, short core);
	s32 Startup( int argc, char *argv[] );
	void UpdateFileDates(void);
	unsigned long GetFileDate(char *filename);
	Boolean ChangedFileDates(void);
	#if defined(MAC_GUI)
	  void RefreshBufferCounts(void);
	  void UpdateProblemStatus(unsigned int problem_count);
	#endif
};

#endif