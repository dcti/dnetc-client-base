
const char *ReadTemp_C(void) {
return "@(#)$Id: ReadTemp.cpp,v 1.1.2.1 2001/01/21 15:17:32 cyp Exp $"; }

#include <Memory.h>
#include <Devices.h>
#include <Errors.h>

#include "ReadTemp.h"

/* We really don't want anybody messing around with anything in this file other than */
/* the GetProcessorTemperature() entry point, so let's make everything else static.  */
static int  ReadTempUp(void);
static int  ReadTempDown(void);
static void EnterPPCSupervisorMode(void);
static void LeavePPCSupervisorMode(void);
static void InstallHandler(void);
static void RemoveHandler(void);
static void HandlerCode(void);
static void HandlerCodeEnd(void);

typedef struct MyHandlerData
{
	unsigned long	OriginalHandler;
	unsigned long	r4;
	unsigned long	cr;
	unsigned long	Code[];
} MyHandlerData;

static MyHandlerData*	GlobalHandlerData;

/* register numbers */
#define SPRG1	273
#define SPRG2	274

#define LOCATION(x) ((unsigned long*)x)[0]

int GetProcessorTemperature(void)
  
  {
    int Lo,Hi;
    
    EnterPPCSupervisorMode();
    Lo = ReadTempUp();
    Hi = ReadTempDown();
    LeavePPCSupervisorMode();
    return (Lo + ((Hi - Lo) / 2)); // split the difference in case they don't match (very common)
  }
    
    
static asm int ReadTempUp(void)
  {
// overall:
// start at zero and increase until we get a "temp is lower than r5" crossing, then 
// return r5

// Assumes CPU is in Supervisor mode (If it isn't, this routine will bomb out with a priv violation
// exception at the first mtspr instruction)
      li        r5,0         // start thresh at zero
@retry
      addi      r5,r5,1          
      li        r3,0
      mtspr     1020,r3      // Stop any temp check in progress
      mtspr     1021,r3
      mtspr     1022,r3
      isync                  // Let everything settle
      clrlslwi. r3,r5,25,23  // if r5 ROL 23 & 0x3f80 == 0, exit. Happens when r5 becomes 0. 
                             // as a side effect, shifts r5 (thresh) into thresh bits for
                             // stuffing into THRM1. Neat instruction!
      beq       @done
      ori       r3,r3,0x0004 // otherwise, set direction bit
      ori       r3,r3,0x0001 // and valid bit
      mtspr     1020,r3      // then stuff into THRM1
      sync                   // let everything settle
      li        r3,16382     // just a big number for the SITV value
      ori       r3,r3,0x0001 // Set the enable bit
      mtspr     1022,r3      // stuff THRM3 to start the check
      sync                   // Let everything settle
@reread
      mfspr     r3,1020      // grab the value in THRM1
      rlwinm.   r4,r3,0,1,1  // extract bit number 1 (TIV)
      beq       @reread      // TIV == false. Loop back and try again.
      mfspr     r3,1020      // re-get it since we wasted it with the rlwinm instruction
      clrrwi.   r4,r3,31     // extract bit number 0 (TIN)
      beq       @retry       // Didn't cross. Bump thresh and try again.
@done                        // We're all done here. Clean up and go home.
      li        r3,0
      mtspr     1020,r3      // Make sure temp compares are stopped
      mtspr     1021,r3
      mtspr     1022,r3
      mr        r3,r5        // set up to return r5 as temp read
      blr                    // and it's game over.
  }
  
static asm int ReadTempDown(void)
  {    
// overall:
// start at 128, and decrease until we get a "temp is higher than r5" crossing, then
// return r5
// For the "play by play", see ReadTempUp()'s code. This is the same, except going down
// from 128 instead of up from zero.

      li r5,128
@retry
      subi r5,r5,1
      li r3,0
      mtspr 1020,r3
      mtspr 1021,r3
      mtspr 1022,r3
      isync
      clrlslwi. r3,r5,25,23
      beq @done
      ori r3,r3,0x0000
      ori r3,r3,0x0001
      mtspr 1020,r3
      sync
      li r3,16382
      ori r3,r3,0x0001
      mtspr 1022,r3
      sync
@reread      
      mfspr r3,1020
      rlwinm. r4,r3,0,1,1
      beq @reread
      mfspr r3,1020
      clrrwi. r4,r3,31
      beq @retry
@done 
      li r3,0
      mtspr 1020,r3
      mtspr 1021,r3
      mtspr 1022,r3
      mr r3,r5
      blr
  }

static void EnterPPCSupervisorMode(void)

    {        
      GlobalHandlerData = NULL;
    	InstallHandler(); /* Set up the exception handler we're about to invoke */
    	asm {
  	        /* Set up caller signature */
            lis	    r3,'RC';
            ori	    r3,r3,'5C';
            /* invoke the exception handler */
          	sc;
          }
      /* Now get rid of the handler */
    	RemoveHandler();
    	/* Theoretically, we could install the handler at init time, then uninstall it at shutdown */
    	/* to save time. However, I feel that this way is a bit safer.                             */
    }
    
static asm void LeavePPCSupervisorMode(void)

   {
      mfmsr r4;           /* Get current setting */
      ori   r4,r4,0xc000; /* Set PR bit and EE bit (User mode, and re-enable interrupts) */
      mtmsr r4;           /* Stuff modified back into MSR */
      /* We're now safely back in user mode */
   }
   
static void InstallHandler(void)

  {
  	void				           *HandlerPhys;
  	unsigned long		       CodeLength, Count;
  	MyHandlerData          *Handler;
  	LogicalToPhysicalTable L2PTable;
  	OSErr				   Err;
  	
  	CodeLength =  LOCATION(HandlerCodeEnd) - LOCATION(HandlerCode); /* How big? */
  	Handler =  (MyHandlerData*)NewPtrSys(sizeof(MyHandlerData) + CodeLength); /* Make a hole */
  	Handler->OriginalHandler = *(unsigned long*)0x5FFFE450; /* save sc vector */
  	GlobalHandlerData = Handler; /* copy out to global so we can access it later */
  		
  	BlockMove((void *)LOCATION(HandlerCode),Handler->Code,CodeLength); /* Move it into place */
  	LockMemoryContiguous(Handler,sizeof(MyHandlerData) + CodeLength); /* Freeze, sucka! */
  	MakeDataExecutable(Handler->Code,CodeLength); /* Sync caches (and other needed stuff) */

    /* Figure out where the handler lives */
  	L2PTable.logical.address = &Handler->Code;
  	L2PTable.logical.count	 = 1024;
  	Count = sizeof(L2PTable) / sizeof(MemoryBlock) - 1;
  	Err = GetPhysical(&L2PTable, &Count);
  	if (Err != noErr)
  	    HandlerPhys = &Handler->Code;
  	else
  	    HandlerPhys = (Ptr)L2PTable.physical[0].address;
      /* Now that we know the location, stuff it into the vector */
    *(unsigned long*)0x5FFFE450 = (unsigned long)HandlerPhys; /* Chain in */
  }


static void RemoveHandler(void)

  {
    unsigned long CodeLength;
      
    CodeLength = LOCATION(HandlerCodeEnd) - LOCATION(HandlerCode);      /* Figure out code length */
  	*(unsigned long*)0x5FFFE450 = GlobalHandlerData->OriginalHandler; /* Put the old handler back */
    UnlockMemory(GlobalHandlerData, sizeof(MyHandlerData) + CodeLength); /* Let things move again */
  	DisposePtr((char *)GlobalHandlerData);                                /* Prevent memory leaks */
  	GlobalHandlerData = NULL;                    /* Make sure it errors out if re-used after this */	
  }


static asm void HandlerCode(void)

  {
    /* When execution gets here, we're operating in Supervisor mode. As soon as the rfi happens,  */
    /* we'll lose that luxury unless we take some steps to hold onto it (kinda like freedom...)   */
  	/* Make room for a local copy of MyHandlerData */
  	subi	sp,sp,sizeof(MyHandlerData);
  	
  	/* Save r4 and cr in case the call isn't for us */ 
  	stw	    r4,MyHandlerData.r4(sp);
  	mfcr	  r4;
  	stw	    r4,MyHandlerData.cr(sp);	

  	/* Test the signature. Is this call really meant for us? */
  	lis	    r4,'RC';
  	ori	    r4,r4,'5C';
  	cmpw	  r3,r4;
  	bne 	  @callOriginalHandler; /* Not ours. Pass on */

  	/* It's for us. Put cr back, and set up for the rfi */
  	lwz	    r4,MyHandlerData.cr(sp);
  	mtcrf   0xFF,r4;
  	mfspr   r4,SPRG2;
  	mtlr	  r4;
  	mfspr 	sp,SPRG1;
  	/* We need to clear the PR bit of the MSR, but it's already clear, since we're executing as  */
  	/* an exception handler. (This implicitly means that the PR bit is clear) But it will be put */ 
  	/* back to whatever it was before we got called as part of the normal operation of the rfi   */
  	/* instruction. So what do we do???? The answer:                                             */
  	/* Mess with the saved (by the sc instruction that got us here) copy of the MSR that's       */
  	/* stored in the srr1 register! This assures that our change will "stick" after the rfi      */
  	/* instruction is executed.                                                                  */                             	
  	/* We're also going to shut down all maskable interrupts for the duration of our time in     */
  	/* supervisor mode.                                                                          */
  	mfsrr1	r4;
  	li      r3,0x3fff;
  	oris    r3,r3,0xffff;
  	and     r4,r4,r3;
  	mtsrr1  r4;
  	rfi;

  @callOriginalHandler:
  	/* Nope, not for us. Put r4/cr back like we found 'em, and pass through to whatever */
  	/* handler we may have cut ahead of. */
  	lwz	    r4,MyHandlerData.cr(sp);
  	mtcrf	  0xFF,r4;
  	lwz	    r4,MyHandlerData.r4(sp);
  	lwz	    sp,MyHandlerData.OriginalHandler(sp);
  	mtlr	  sp;
  	blrl;
  entry HandlerCodeEnd
  HandlerCodeEnd:	
  }
