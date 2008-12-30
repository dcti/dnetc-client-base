/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev
 * PanAm
 * Alexei Chupyatov
 *
 * $Id: amdstream_setup.cpp,v 1.1 2008/12/30 16:54:56 andreasb Exp $
*/

#include "amdstream_setup.h"

bool cInit=false;

stream_context_t CContext[16];	//MAXCPUS?
static CALuint numDevices = 0;

u32 getAMDStreamDeviceCount()	
{
	if(!cInit){
		numDevices=0;
		calInit();
		cInit=true;
	    // Finding number of devices
		if(calDeviceGetCount(&numDevices)!=CAL_RESULT_OK)
			return 0;
		if(numDevices==0) {
			LogScreen("No supported devices found!");
			exit(-1);		//TODO:
			return 0;
		}
		numDevices=1;		//TODO: add multigpu support
		if(numDevices>16)
			numDevices=16;
	}else
		return numDevices;
	for(u32 i=0;i<16;i++){
		CContext[i].active=false;
		CContext[i].coreID=CORE_NONE;
		CContext[i].constMem=0; CContext[i].constName=0; CContext[i].constRes=0; 
		CContext[i].ctx=0; CContext[i].device=0; CContext[i].module=0;
		CContext[i].outName0=0; CContext[i].outputMem0=0; CContext[i].outputRes0=0;
		CContext[i].obj=NULL; CContext[i].image=NULL;

		if(i<numDevices){
			// Opening device
			calDeviceOpen(&CContext[i].device, i);

			// Querying device attribs
			CContext[i].attribs.struct_size = sizeof(CALdeviceattribs);
			if(calDeviceGetAttribs(&CContext[i].attribs, i)!=CAL_RESULT_OK)
				continue;
			CContext[i].active=true;

			CContext[i].domainSizeX=32;
			CContext[i].domainSizeY=32;
			CContext[i].maxIters=512;
		}
	}
	return numDevices;
}

unsigned getAMDStreamDeviceFreq()
{
	return CContext[0].attribs.engineClock;		//TODO:device id
}

long __GetRawProcessorID(const char **cpuname)
{
  static char buf[30];
  buf[0]='\0';
  if(!cInit)
	  getAMDStreamDeviceCount();
  switch(CContext[0].attribs.target) {			//TODO:device id
	case CAL_TARGET_600:
		strcpy(buf,"R600");
		break;
	case CAL_TARGET_610:
		strcpy(buf,"RV610");
		break;
	case CAL_TARGET_630:
		strcpy(buf,"RV630");
		break;
	case CAL_TARGET_670:
		strcpy(buf,"RV670");
		break;
	case CAL_TARGET_7XX:
		strcpy(buf,"R700 class");
		break;
	case CAL_TARGET_770:
		strcpy(buf,"RV770");
		break;
	case CAL_TARGET_710:
		strcpy(buf,"RV710");
		break;
	case CAL_TARGET_730:
		strcpy(buf,"RV730");
		break;
	default:
		break;
	}
    *cpuname=buf;
	return CContext[0].attribs.target;			//TODO:device id
}

void Deinitialize_rc5_72_il4(u32 Device)
{
	// Unloading the module
	if(CContext[Device].module)
	{
		calModuleUnload(CContext[Device].ctx, CContext[Device].module);
		CContext[Device].module=0;
	}

	// Freeing compiled program binary
	if(CContext[Device].image) {
		calclFreeImage(CContext[Device].image);
		CContext[Device].image=NULL;
	}
	if(CContext[Device].obj) {
		calclFreeObject(CContext[Device].obj);
		CContext[Device].obj=NULL;
	}

	// Releasing resource from context
	if(CContext[Device].constMem) {
		calCtxReleaseMem(CContext[Device].ctx, CContext[Device].constMem);
		CContext[Device].constMem=0;
	}
	if(CContext[Device].outputMem0) {
		calCtxReleaseMem(CContext[Device].ctx, CContext[Device].outputMem0);
		CContext[Device].outputMem0=0;
	}

	// Deallocating resources
	if(CContext[Device].constRes) {
		calResFree(CContext[Device].constRes);
		CContext[Device].constRes=0;
	}

	// Deallocating resources
	if(CContext[Device].outputRes0) {
		calResFree(CContext[Device].outputRes0);
		CContext[Device].outputRes0=0;
	}

	// Destroying context
	calCtxDestroy(CContext[Device].ctx);
	CContext[Device].coreID=CORE_NONE;
}
