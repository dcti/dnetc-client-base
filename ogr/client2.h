/*
 * Generic interface to distributed type cores.
 * Greg Hewgill 1998-11-28
 */

/*
 * Constants for return values from all the below functions.
 * Those starting with CORE_S are success codes, and those starting
 * with CORE_E are error codes.
 */
#define CORE_S_OK       0
#define CORE_S_CONTINUE 1
#define CORE_E_MEMORY   (-1)
#define CORE_E_IO       (-2)
#define CORE_E_FORMAT   (-3)
#define CORE_E_STOPPED  (-4)

/*
 * Dispatch table structure. A pointer to one of these should be returned
 * from an exported function in the module.
 *
 * The result callback is weird, it implies that only one client in the
 * entire process will use this core (but that client can have multiple
 * threads). Oh well.
 */
typedef struct {
  /*
   * Initialize the core, called once for all threads.
   */
  int (*init)();

  /*
   * Create a new work unit, called once for each thread.
   * The format of input is defined by the core.
   */
  int (*create)(void *input, int inputlen, void **state);

  /*
   * Continue working, return CORE_S_OK if no more work to do, or
   * CORE_S_CONTINUE if things need to keep going.
   * On input, nodes should contain the number of algorithm iterations
   * to do. On output, nodes will contain the actual number of iterations
   * done.
   */
  int (*cycle)(void *state, int *nodes);

  /*
   * Clean up state structure.
   */
  int (*destroy)(void *state);

  /*
   * Return the number of bytes needed to serialize this state.
   */
  int (*count)(void *state);

  /*
   * Serialize the state into a flat data structure suitable for
   * persistent storage.
   * buflen must be at least as large as the number of bytes returned
   * by count().
   * Does not destroy the state structure.
   */
  int (*save)(void *state, void *buffer, int buflen);

  /*
   * Load the state from persistent storage buffer.
   */
  int (*load)(void *buffer, int buflen, void **state);

  /*
   * Clean up anything allocated in init().
   */
  int (*cleanup)();

  /*
   * Result callback, filled in by client.
   * If anything other than CORE_S_OK is returned, the result was
   * NOT saved.
   * On failure, the core should not proceed further and should return
   * CORE_E_STOPPED on further calls to cycle().
   * The format of result is defined by the core.
   */
  int (*result)(void *state, void *result, int resultlen);
} CoreDispatchTable;
