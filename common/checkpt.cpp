// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: checkpt.cpp,v $
// Revision 1.2  1999/01/01 02:45:14  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.1  1998/11/26 07:09:44  cyp
// Merged DoCheckpoint(), UndoCheckpoint() and checkpoint deletion code into
// one function and spun it off into checkpt.cpp
//
//
#if (!defined(lint) && defined(__showids__))
const char *checkpt_cpp(void) {
return "@(#)$Id: checkpt.cpp,v 1.2 1999/01/01 02:45:14 cramer Exp $"; }
#endif

#include "client.h"   // FileHeader, Client class
#include "baseincs.h" // unlink()
#include "scram.h"    // Descramble()/Scramble()
#include "buffwork.h" // IsFilenameValid(), DoesFileExist(), EraseCheckpoint()
#include "problem.h"  // Problem class
#include "probman.h"  // GetProblemPointerFromIndex()
#include "probfill.h" // FILEENTRY_xxx macros
#include "version.h"  // CLIENT_* defines used by FILEENTRY_* macros
#include "logstuff.h" // LogScreen()
#include "network.h"  // gawd, htonl(), ntohl()
#include "checkpt.h"  // ourselves

/* ----------------------------------------------------------------- */

int Client::CheckpointAction( int action, unsigned int load_problem_count )
{
  FileEntry fileentry;
  unsigned int prob_i, cont_i, cont_count, recovered;
  int do_checkpoint, retval = 0;
  unsigned long remaining, lastremaining;
  char *ckfile;

  cont_count = 2;
  #ifdef COMBINE_ALL_CHECKPOINTS_IN_ONE_FILE
  cont_count = 1;
  #endif

  do_checkpoint = 0;
  if (nodiskbuffers == 0)
    {
    for (cont_i = 0; cont_i < cont_count; cont_i++)
      {
      if ( IsFilenameValid( checkpoint_file[cont_i] ) )
        do_checkpoint = 1;
      }
    }

  if ( action == CHECKPOINT_OPEN )
    {
    if (do_checkpoint)
      {
      recovered = 0;
  
      for ( cont_i = 0; cont_i < cont_count; cont_i++ )
        {
        ckfile = &(checkpoint_file[cont_i][0]);
        if ( DoesFileExist( ckfile ))
          {
          lastremaining = 0;
          while (BufferGetFileRecord( ckfile, &fileentry, &remaining ) == 0) 
            //returns <0 on ioerr, >0 if norecs               
            {
            if (lastremaining != 0)
              {
              if (lastremaining <= remaining)
                {
                recovered = 0;
                break;
                }
              }
            lastremaining = remaining;
            if ( PutBufferRecord( &fileentry ) > 0 )
              {
              recovered++;              
              }
            }
          }
        }
      if (recovered)  
        {
        LogScreen("Recovered %u checkpoint block%s\n", recovered, 
            ((recovered == 1)?(""):("s")) );
        }
      action = CHECKPOINT_CLOSE;
      }
    retval = (do_checkpoint == 0); /* return !0 if don't do checkpoints */
    }  

  /* --------------------------------- */

  if ( action == CHECKPOINT_CLOSE || action == CHECKPOINT_REFRESH )
    {
    for (cont_i = 0; cont_i < cont_count; cont_i++)
      {
      ckfile = &(checkpoint_file[cont_i][0]);
      if ( IsFilenameValid( ckfile ) )
        {
        EraseCheckpointFile( ckfile ); 
        }
      }
    #ifdef COMBINE_ALL_CHECKPOINTS_IN_ONE_FILE
    if ( IsFilenameValid( checkpoint_file[1] ) )
      unlink( checkpoint_file[1] );
    #endif
    }

  /* --------------------------------- */
    
  if ( action == CHECKPOINT_REFRESH )
    {
    if ( do_checkpoint )
      {
      for ( prob_i = 0 ; prob_i < load_problem_count ; prob_i++)
        {
        Problem *thisprob = GetProblemPointerFromIndex(prob_i);
        if ( thisprob )
          {
          cont_i = (unsigned int)thisprob->RetrieveState(
                                             (ContestWork *) &fileentry, 0);
          if (cont_i == 0 || cont_i == 1)
            {
            #ifdef COMBINE_ALL_CHECKPOINTS_IN_ONE_FILE
            ckfile = &(checkpoint_file[0][0]);
            #else
            ckfile = &(checkpoint_file[cont_i][0]);
            #endif
  
            if ( IsFilenameValid( ckfile ) )
              {
              fileentry.contest = (u8)cont_i;
              fileentry.op      = htonl( OP_DATA );
              fileentry.cpu     = FILEENTRY_CPU;
              fileentry.os      = FILEENTRY_OS;
              fileentry.buildhi = FILEENTRY_BUILDHI; 
              fileentry.buildlo = FILEENTRY_BUILDLO;
              fileentry.checksum=
                 htonl( Checksum( (u32 *) &fileentry, (sizeof(FileEntry)/4)-2));
              Scramble( ntohl( fileentry.scramble ),
                           (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );
                           
              if (BufferPutFileRecord( ckfile, &fileentry, NULL ) < 0) 
                {                        /* returns <0 on ioerr */
                //Log( "Checkpoint %u, Buffer Error \"%s\"\n", 
                //                     prob_i+1, ckfile );
                //break;
                }
              }
            } 
          } 
        }  // for ( prob_i = 0 ; prob_i < load_problem_count ; prob_i++)
      } // if ( !nodiskbuffers )
  
    retval = (do_checkpoint == 0); /* return !0 if don't do checkpoints */
    }
  
  return retval;
}  
