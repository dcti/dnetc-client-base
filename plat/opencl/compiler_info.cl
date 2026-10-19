//Get compiler definitions for debugging
//CORENAME=compiler_info_src
__kernel void compiler_info(__global uint *debug_out)
{
    // Index 0: __NVPTX__ 
    #if defined(__NVPTX__)
        debug_out[0] = 1;
    #else
        debug_out[0] = 0;
    #endif

    // Index 1: cl_amd_media_ops
    #if defined(cl_amd_media_ops)
        debug_out[1] = 1;
    #else
        debug_out[1] = 0;
    #endif

    // Index 2: __clang__
    #if defined(__clang__)
        debug_out[2] = 1;
    #else
        debug_out[2] = 0;
    #endif

   // Index 3: spare
   debug_out[3] = 0;
}
