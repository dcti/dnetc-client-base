// Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
/* 
 * This is an amazingly trivial :p event handling mechanism for client 
 * platforms that do something when a special condition arises. 
 *
 * Event posting to listeners is currently synchronous (blocking) only,
 * although asynchronous posting could be implemented with a minimum of
 * effort (should the need ever arise).
 *
 * Since the listener queue is a static resource, and it is in everyone's 
 * best interest to keep the queue small, platforms that implement 
 * an event listener that 'lives' for the clients entire lifetime should do 
 * so in a cascaded fashion, ie with one wrapper function that branches to 
 * subfunctions. Register that wrapper function with event_id == -1 to
 * listen for all events and branch internally as appropriate.
 *
 * If you need to add a new event id that requires more than a long as a
 * parameter FIRST make sure that that information cannot be obtained any 
 * other way and THEN declare a struct in clievent.h, fill it in the code
 * and pass the pointer to the struct as the parm. Keep the structures 
 * generic so that they can be used by other events as well. For example:
 * struct struct_EVENT_ll { long l1, l2; };
 *
 */ 
// $Log: clievent.cpp,v $
// Revision 1.3  1999/01/24 00:27:48  silby
// I want more listeners! :)
//
// Revision 1.2  1999/01/01 02:45:14  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.1  1998/12/28 18:16:19  cyp
// Created.
//
//
//

#if (!defined(lint) && defined(__showids__))
const char *clievent_cpp(void) {
return "@(#)$Id: clievent.cpp,v 1.3 1999/01/24 00:27:48 silby Exp $"; }
#endif

#include "baseincs.h"   /* NULL, memset */
#include "clievent.h"   /* keep prototypes in sync */


#ifndef MAX_EVENT_LISTENERS
#define MAX_EVENT_LISTENERS 8
#endif

/* ------------------------------------------------------------ */


struct event_listener
{
  int event_id; 
  void (*proc)(int event_id, long parm);
};
struct event_listener listeners[MAX_EVENT_LISTENERS];
unsigned int listener_count = 0;

/* ------------------------------------------------------------ */

int ClientEventAddListener(int event_id, void (*proc)(int event_id, long parm))
{
  unsigned int i;

  if (listener_count == 0)
    memset( (void *)(&listeners[0]), 0, sizeof(listeners ));
    
  if (event_id == 0 || proc == NULL)
    return -1;

  for (i=0;i<(sizeof(listeners)/sizeof(listeners[0]));i++)
    {
    if (listeners[i].event_id == 0)
      {
      listeners[i].event_id = event_id;
      listeners[i].proc = proc;
      listener_count++;
      return (i+1);
      }
    }
  return -1;
}  
  
 
/* ------------------------------------------------------------ */

int ClientEventRemoveListener(int event_id, void (*proc)(int event_id, long parm))
{
  unsigned int i;

  if (event_id == 0 || proc == NULL)
    return -1;

  if (listener_count == 0)
    return 0;

  for (i=0;i<(sizeof(listeners)/sizeof(listeners[0]));i++)
    {
    if (listeners[i].event_id == event_id && listeners[i].proc == proc)
      {
      listeners[i].event_id = 0;
      listeners[i].proc = NULL;
      listener_count--;
      return 0;
      }
    }
  return -1;
}  
    
/* ------------------------------------------------------------ */

int ClientEventSyncPost( int event_id, long parm )
{
  unsigned int i;
  int posted_count = 0;

  if (event_id == 0)
    return -1;

  if (listener_count == 0)
    return 0;

  for (i=0;i<(sizeof(listeners)/sizeof(listeners[0]));i++)
    {
    if (listeners[i].proc != NULL && 
        (listeners[i].event_id == event_id || listeners[i].event_id == -1))
      {
      (*(listeners[i].proc))( event_id, parm );
      posted_count++;
      }
    }
  return posted_count;
}   
      
/* ------------------------------------------------------------ */
      
