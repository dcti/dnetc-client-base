// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: checkpt.cpp,v $
// Revision 1.5  1999/01/29 19:02:48  jlawson
// fixed formatting.  changed some int vars to bool.
//
// Revision 1.4  1999/01/17 14:26:38  cyp
// added leading/trailing whitespace stripping for checkpoint_file.
//
// Revision 1.3  1999/01/04 02:49:10  cyp
// Enforced single checkpoint file for all contests.
//
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
return "@(#)$Id: checkpt.cpp,v 1.5 1999/01/29 19:02:48 jlawson Exp $"; }
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
  unsigned int prob_i, cont_i, recovered;
  unsigned long remaining, lastremaining;
  bool do_checkpoint = (nodiskbuffers==0);

  if (do_checkpoint)
  {
    while (checkpoint_file[0] != 0 && isspace(checkpoint_file[0]))
      strcpy( checkpoint_file, &checkpoint_file[1] );
    recovered = strlen(checkpoint_file);
    while (recovered > 0 && isspace(checkpoint_file[recovered-1]))
      checkpoint_file[--recovered] = 0;
    do_checkpoint = (nodiskbuffers == 0 && IsFilenameValid( checkpoint_file ));
  }

  if ( action == CHECKPOINT_OPEN )
  {
    if (do_checkpoint)
    {
      recovered = false;
      if ( DoesFileExist( checkpoint_file ))
      {
        lastremaining = 0;
        while (BufferGetFileRecord( checkpoint_file, &fileentry, &remaining ) == 0) 
                               //returns <0 on ioerr, >0 if norecs
        {
          if (lastremaining != 0)
          {
            if (lastremaining <= remaining)
            {
              recovered = false;
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
      if (recovered)  
      {
        LogScreen("Recovered %u checkpoint block%s\n", recovered, 
            ((recovered == 1)?(""):("s")) );
      }
      action = CHECKPOINT_CLOSE;
    }
  }

  /* --------------------------------- */

  if ( action == CHECKPOINT_CLOSE || action == CHECKPOINT_REFRESH )
  {
    if (do_checkpoint)
    {
      EraseCheckpointFile( checkpoint_file ); 
    }
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
          if (thisprob->IsInitialized())
          {
            cont_i = (unsigned int)thisprob->RetrieveState(
                                    (ContestWork *) &fileentry, 0);
            if (cont_i == 0 || cont_i == 1 )
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
                           
              if (BufferPutFileRecord( checkpoint_file, &fileentry, NULL ) < 0) 
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


