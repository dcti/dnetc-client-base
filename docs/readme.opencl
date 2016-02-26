
 distributed.net client for OpenCL-compatible GPUs
 document revision $Id: readme.opencl,v 1.1 2015/11/28 22:56:01 stream Exp $

 Welcome to the distributed.net client.

 This document covers information specific to the client for the
 OpenCL-capable video card GPU. Refer to other enclosed documentation or
 browse the online FAQ at <http://faq.distributed.net/> for
 non-platform-specific documentation.

    1.0  Getting started
    2.0  OpenCL specific notes
    3.0  Troubleshooting and known issues
         3.1 Multiple GPUs
         3.2 GPU Client priority
         3.3 RC5-72 block sizes

 1.0  Getting started ------------------------------------------------
 
    Just unpack/unzip/untar the client in a directory of your choice and 
    fire it up.

    If you have never run the client before, it will initiate the
    menu-driven configuration. Save and quit when done.

    Then, simply restart the client. From that point on it will use the 
    saved configuration.
    
    The configuration options are fairly self-explanatory and can be run
    at any time by starting the client with the '-config' option.
    A list of command line options is available by starting the client 
    with '-help' or '--help'.

    A complete step-by-step guide to running your first client is 
    available at <http://www.distributed.net/docs/tutor_clients.php>


 2.0  OpenCL specific notes -----------------------------------------

    The client requires OpenCL version 1.1. It will not run on OpenCL 1.0
    devices. An error will be reported during initialization of computing
    core.


 3.0  Troubleshooting and known issues ------------------------------

    NVIDIA devices may use up to full CPU core when running OpenCL code.
    This is a known "feature" of NVIDIA drivers. In some cases you may also
    need to increase priority of the client (see section 3.2).

  3.1   Multiple GPUs -----------------------------------------------

    The current client code can only select one core for crunching a project.
    In the modern world, this does not work well when multiple GPUs are
    installed in one system. Each GPU can be a different age, model or brand
    with different features. As a result each may require different cores for
    optimal performance, or, even worse, some crunching cores could be
    unsupported due to the differences.

    To list all of the GPUs detected by the client, run "dnetc -gpuinfo".

    By default the client will try to use all of the available GPUs. The first
    detected GPU (GPU0) is used as a reference for auto-selection of the
    crunching cores. All benchmarks and tests will run only on GPU0 by
    default. In many cases, all of the GPUs in a system can run the crunching
    core which is automatically selected for GPU0, and their performance on
    this core is also optimal or close to it. In this case, there is no need
    to change anything. In other cases, it may be beneficial to change the
    default behaviour, which can be done with command-line options or
    configuration parameters.

    To run benchmarks or tests on GPUs other then GPU0, use the device number
    option. On the command line: "-devicenum N", or in the configuration:
    "Performance related options" -> "Run on the selected device number only".
    For example, to benchmark GPU1 (the second GPU detected), run
    "dnetc -devicenum 1 -bench".

    If you need or want to use different parameters for each GPU, you must use
    the device number option. With this option specified, the client will run
    only ONE process and it will run only on the specified device, i.e. you
    must run as many copies of the client as the number of GPUs you want to
    use, each with a unique device number setting. Identification and
    automatic core selection will be done using the GPU specified when using
    the device number option.

    It is possible to run multiple copies of the client from the same
    directory using different command-line options. You can also run each copy
    of the client in its own folder or you can copy the client executable
    under different names in the same folder so it will use different
    configuration files. For example, if you name them as "dnetc_gpu0.exe" and
    "dnetc_gpu1.exe", their configuration files will be automatically named
    "dnetc_gpu0.ini" and "dnetc_gpu1.ini" correspondingly). Windows users may
    need to add "-multiok=1" to command-line options, otherwise multiple
    copies of the client will not be allowed to run. If you are using
    checkpoint files, make sure that each copy of the client is using its own
    private file.

  3.2   GPU Client priority -----------------------------------------

    Some fast GPUs may require a significant amount of CPU power to keep them
    busy. If you are running both GPU and CPU clients, and your real crunch
    rates are significantly less than the benchmarked values, you may feel
    that you need to give more CPU time to the GPU client. There are two ways
    that you can do this. One way is to decrease the number of CPU cores used
    by the CPU client ("-numcpu" option) so that a CPU core is available to be
    used by the GPU client. This is the safest solution. Obviously, the output
    of your CPU client will be decreased. The other way (possibly a better
    solution) is to increase the priority of the GPU client using the priority
    option at runtime ("-priority N" or in the configuration: "Performance
    related options" -> "Priority level to run at"). The default priority
    level is 0 (idle). You can increase it from 0 to 9 until the GPU client
    gets enough CPU time to keep the GPU busy. If you set the priority level
    too high, it may decrease responsiveness of other programs running on your
    computer, so take care when using this option.

  3.3   RC5-72 block sizes ------------------------------------------

    With the speed of modern GPUs, the overhead caused by accessing the buffer
    files and sending updates to the network may become noticeable and reduce
    the overall client performance. By default, GPU clients request larger
    packets than CPU clients, with a size of 64 RC5-72 units. For GPUs which
    are quite fast (1 Gkey/sec or faster), even this packet will only take a
    few minutes to complete. It is possible to increase the packet size in the
    configuration: "Buffer and Buffer Update Options"/"Preferred packet size".
    Similarly, if your GPU is slower and the default packet size takes too
    much time to complete, the packet size can be decreased. We currently
    support packet sizes in the range of 1 - 1024 units. Note that this is the
    preferred value the client software sends to the key server network in a
    request. The server may return a smaller packet if no larger ones are
    available. It is also recommended to minimize the network update frequency;
    the default configuration is usually sufficient. To decrease delays caused
    by network updates, we recommend that you use the distributed.net personal
    proxy software. ( http://www.distributed.net/Download_proxies ) Note that
    if you are using the personal proxy software, you may need to alter the
    proxy configuration to request larger packet sizes as well.
