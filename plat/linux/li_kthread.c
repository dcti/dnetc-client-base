/*
 * Linux kernel thread create/join.
 *
 * Created Nov 2000, by Cyrus Patel <cyp@fb14.uni-mainz.de>
 * clone() cloned from wine source (wine/misc/port.c)
 * $Id: li_kthread.c,v 1.1.2.2 2001/02/18 23:58:40 cyp Exp $
*/
#define __NO_STRING_INLINES /* work around bugs in glibc bits/strings2.h */

#include <sched.h>       /* sched_yield() */
#include <stdlib.h>      /* malloc()/free() */
#include <unistd.h>      /* sched_yield() */
#include <signal.h>      /* SIGCHLD */
#include <string.h>      /* memset() */
#include <errno.h>       /* errno */
#include <sys/types.h>   /* pid_t */
#include <sys/wait.h>    /* waitpid() */
#include <sys/syscall.h> /* SYS_clone */

#ifndef CLONE_VM
#  define CLONE_VM      0x00000100
#  define CLONE_FS      0x00000200
#  define CLONE_FILES   0x00000400
#  define CLONE_SIGHAND 0x00000800
#  define CLONE_PID     0x00001000
#endif

#ifdef __cplusplus
extern "C" {
#endif
int kthread_join( long __ctx );
long kthread_create( void (*thr_fxn)(void *), int stack_size, void *thr_arg );
int kthread_yield( void );
#ifdef __cplusplus
}
#endif

struct thread_ctx
{
  unsigned long magic; /* THREADCTX_MAGIC */
  int pid;
  size_t stack_size;
  char *stack_buf;
  unsigned long stack_guard;
};  

#define THREADCTX_MAGIC ((('t')<<24)+(('h')<<16)+(('r')<<8)+('d'))

int kthread_yield(void)
{
  usleep(0);
  return 0;
}  
/* int sched_yield(void) { return kthread_yield(); } */

static int my_clone( int (*fn)(void *), void *stack, int flags, void *arg )
{
#ifdef __i386__
  int ret;
  void **stack_ptr = (void **)stack;
  *--stack_ptr = (void *)arg;  /* Push argument on stack */
  *--stack_ptr = (void *)fn;   /* Push function pointer (popped into ebx) */
  __asm__ __volatile__( "pushl %%ebx\n\t"
                        "movl %2,%%ebx\n\t" /* load flags */
                        "int $0x80\n\t" /* eax=SYS_CLONE,ebx=flags,ecx=stacktop */
                        "popl %%ebx\n\t" /* Contains fn in the child */
                        "testl %%eax,%%eax\n\t"
                        "jnz 0f\n\t"
                        "call *%%ebx\n\t" /* call the function */
                        "xorl %%eax,%%eax\n\t" /* zero if the child returns */
                        "0:"
                        : "=a" (ret) 
                        : "0" (SYS_clone), "r" (flags), "c" (stack_ptr) );
  if (ret == 0) /* If ret is 0, we returned from the child function */
    _exit(0);
  else if (ret > 0)
    return ret; /* pid of child */
  errno = -ret;
#else
  errno = EINVAL;
#endif /* __i386__ */ 
  return -1;
}  



int kthread_join( long __ctx )
{
  struct thread_ctx *ctx = (struct thread_ctx *)__ctx;
  if (ctx)
  {
    if (ctx->magic == THREADCTX_MAGIC)
    {
      if (ctx->stack_guard == THREADCTX_MAGIC)  
      {
        waitpid( ctx->pid, 0, 0 );
        ctx->magic = 0;
        free((void *)ctx->stack_buf);
        return 0;
      }
    }
  }
  return -1;
}          

long kthread_create( void (*thr_fxn)(void *), int stack_size, void *thr_arg ) 
{
  int flags = CLONE_VM|CLONE_FS/*|CLONE_FILES*/ /*|CLONE_SIGHAND*/|SIGCHLD;
  char *mem, *stack_top;
  struct thread_ctx *ctx;

  if (!thr_fxn || stack_size < 0)
  {
    errno = EINVAL;
    return -1;  
  }  
  if (stack_size < 8192)
  {
    stack_size = 8192;
  } 
  stack_size += (sizeof(struct thread_ctx));
  stack_size += (stack_size & 4096);
  mem = (char *)malloc((size_t)stack_size);                              
  if (!mem)
  {
    /* errno = ENOMEM; */
    return -1;
  }  
  memset(mem, 0, (size_t)stack_size);

  ctx = (struct thread_ctx *)mem;
  stack_top = mem+(stack_size-(sizeof(void *)));
  
  ctx->magic = THREADCTX_MAGIC;
  ctx->stack_size = stack_size;
  ctx->stack_buf = mem;
  ctx->stack_guard = THREADCTX_MAGIC;
  ctx->pid = my_clone( (int (*)(void *))thr_fxn, 
                       stack_top, flags, thr_arg );
  if (ctx->pid < 0)
  {
    ctx->magic = 0;
    free((void *)mem);
    return -1;
  }  
  return (long)ctx;
}    
