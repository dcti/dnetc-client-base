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