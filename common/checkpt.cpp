/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * -----------------------------------------------------------------
 * Rewritten from scratch by Cyrus Patel <cyp@fb14.uni-mainz.de>
 * Client::CheckpointAction( int action, unsigned int load_problem_count )
 * is called with action == CHECKPOINT_[OPEN|REFRESH|CLOSE]. It uses the
 * Problem manager table 'load_problem_count' times to walk the problem
 * list when refreshing the checkpoint file. CheckpointAction() returns 
 * non-zero if the client should not use checkpointing. The checkpoint 
 * interval is controlled from Client::Run()
 * -----------------------------------------------------------------
 *
*/
const char *checkpt_cpp(void) {
return "@(#)$Id: checkpt.cpp,v 1.12 1999/06/09 15:06:16 cyp Exp $"; }

#include "client.h"   // FileHeader, Client class
#include "baseincs.h" // memset(), strlen()
#include "util.h"     // IsFilenameValid(), DoesFileExist()
#include "buffwork.h" // Buffer[Zap|Put|Import]FileRecord[s]()
#include "problem.h"  // Problem class
#include "probman.h"  // GetProblemPointerFromIndex()
#include "version.h"  // CLIENT_* defines used by FILEENTRY_* macros
#include "probfill.h" // FILEENTRY_xxx macros
#include "logstuff.h" // LogScreen()
#include "checkpt.h"  // ourselves

/* ----------------------------------------------------------------- */

int Client::CheckpointAction( int action, unsigned int load_problem_count )
{
  int do_checkpoint = (nodiskbuffers==0);

  if (do_checkpoint)
  {
    unsigned int len = 0;
    while (checkpoint_file[len]!=0 && isspace(checkpoint_file[len]))
      len++;
    if (len)
      strcpy( checkpoint_file, &checkpoint_file[len] );
    len = strlen(checkpoint_file);
    while (len>0 && isspace(checkpoint_file[len-1]))
      checkpoint_file[--len]=0;
    do_checkpoint = (nodiskbuffers==0 && IsFilenameValid( checkpoint_file ));
  }

  if ( action == CHECKPOINT_OPEN )
  {
    if (do_checkpoint)
    {
      if ( DoesFileExist( checkpoint_file ))
      {
        long recovered = BufferImportFileRecords( this, checkpoint_file, 0, -1 );
        if (recovered > 0)  
        {
          LogScreen("Recovered %d checkpoint block%s\n", recovered, 
            ((recovered == 1)?(""):("s")) );
        }
      }
    action = CHECKPOINT_CLOSE;
    }
  }  

  /* --------------------------------- */

  if ( action == CHECKPOINT_CLOSE || action == CHECKPOINT_REFRESH )
  {
    if (do_checkpoint)
      BufferZapFileRecords( checkpoint_file ); 
  }

  /* --------------------------------- */
    
  if ( action == CHECKPOINT_REFRESH )
  {
    if ( do_checkpoint )
    {
      unsigned int prob_i;
      for ( prob_i = 0 ; prob_i < load_problem_count ; prob_i++)
      {
        Problem *thisprob = GetProblemPointerFromIndex(prob_i);
        if ( thisprob )
        {
          if (thisprob->IsInitialized())
          {
            WorkRecord work;
            unsigned int cont_i;
            memset((void *)&work, 0, sizeof(WorkRecord));
            thisprob->RetrieveState((ContestWork *)&work, &cont_i, 0);
            if (cont_i < CONTEST_COUNT /* 0,1,2...*/ )
            {
              work.resultcode = RESULT_WORKING;
              work.contest = (u8)cont_i;
              work.cpu     = FILEENTRY_CPU;
              work.os      = FILEENTRY_OS;
              work.buildhi = FILEENTRY_BUILDHI; 
              work.buildlo = FILEENTRY_BUILDLO;

              if (BufferPutFileRecord( checkpoint_file, &work, NULL ) < 0) 
              {                        /* returns <0 on ioerr */
                //Log( "Checkpoint %u, Buffer Error \"%s\"\n", 
                //                     prob_i+1, checkpoint_file );
                //break;
              }
            }
          } 
        } 
      }  // for ( prob_i = 0 ; prob_i < load_problem_count ; prob_i++)
    } // if ( !nodiskbuffers )
  }
  
  return (do_checkpoint == 0); /* return !0 if don't do checkpoints */
}  
