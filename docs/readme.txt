Distributed.Net RC5-64/DES Client version 2.7100 updates.

Thank you for downloading the Distributed.Net version 2.7100 rc5/des
client. By installing this client, you will ensure that you obtain the
most optimal performance you can obtain for the DES-II-2 contest on July
13th, 1998, as well as the most optimal performance for the ongoing
RC5-64 contest. Please read this file in its entirety, as many
noteworthy changes have been made, which may or may not affect you.
Also, please check for any platform specific readme files that may have
accompanied this client.

There are many changes in version 2.7100 over 2.70x that you should be
aware of. They are as follows:

- Buffer file formats have NOT changed. You may use the exact same
  buffer files you have been using with your 2.70x clients. However,
  buffer files from 2.64x clients should be deleted before installing
  the 2.7100 client, as the format is different.

- New contest switching logic has been implemented. 2.7100 clients will
  query the status of the ongoing contests during each connect to a
  keyserver, ensuring that your clients will switch to DES the first
  time they connect to a keyserver after DES has begun. Also, 2.7100
  clients will recognize which blocks were from DES-II-1 (some may still
  be sitting around in your buffer files), and skip them so that you do
  not waste time. With a 2.7100 client installed, all you need to do for
  DES-II-2 is to cause your client to do an update, flush, or fetch
  on July 13th, 1998, and it will start working on DES-II-2.

- New DES cores have been implemented on all platforms. You will see
  speedups ranging from 50 kkeys/sec (on ppros) to 8 megakeys/second
  (on ultrasparcs). As a result, it is in the best interest of your
  keyrate for you to install new clients whereever possible!

- A completely new config menu has been implemented. It now includes
  many more options that previously required .ini file editing to
  activate. The menu has been arranged so that in most situations, you
  will never have to configure any options besides the ones on the
  Required Options menu.

- Two .ini / command line options have changed

  - Firemode. This option has been removed. All the functionality of
    firemode is in uuehttpmode. If you were using a configuration with
    firemode 0 or 5, you need to make no changes to you configuration.
    However, if you were using firemode 1 through 4, you should go into
    the configuration menu Communications Options and ensure that the
    firewall mode is set as you need it.
  - Preferredcontest. This option has been changed. The new option
    replacing it is processdes. Processdes is a 1/0 option. When it is
    set to 1, the client will participate in DES contests. When it is
    set to 0, the client will NOT participate in DES contests. The
    primary reason for changing this option was so that if you're
    upgrading over an old client (which may have changed
    preferredcontest back to rc5), you will not have to worry about it
    accidently not working on DES. Please make sure you change any
    references to preferredcontest to use processdes instead (described
    in rc5des.txt.) 
  - You may also notice two new .ini settings, contestdone and
    contestdone2. These options are set automatically by the client to
    tell it which contests are currently ongoing. Please do not change
    these options.

To maximize your DES performance, you should do two things:

1. Set your preferred block size to 2^31, instead of the default 2^30.
Using 2^31 blocks will ensure that our master keyserver does not become
overloaded, and will help make the contest smoother for all
participants.

2. Make sure you keep a small amount of blocks in your buffers. We
expect DES-II-2 to be a short contest, hopefully lasting less than two
weeks. To ensure that your work is counted, and we make constant
progress, please buffer only ONE DAY'S worth of blocks at any time. If
your clients are on constant Internet connections, you should set them
to buffer as few blocks as possible (1 sounds like a good number) so
that there is as little latency as possible.

Upgrade procedure(s):
---------------------

If you are currently running a 2.70x client:
1. Shut down the currently running 2.70x client.
2. Move the new executible from the 2.7100 client into the place of the
   2.70x client.
3. If you are using a firewall, please read the note about the removal
   of firemode= and ensure that you are not adversely affected by it.
4. Start the new client up, you're ready to roll!

If you are currently running a 2.64x client:
1. Make sure to flush any finished blocks out of your current client
   (rc564 -flush)
2. Shut down the 2.64x client.
3. Delete the 2.64 client, and its buffer files (buff-in.rc5, buff-
   out.rc5)
4. Move the new 2.7100 client into the directory the old client was in.
5. Start rc5des -config, and ensure that the required options are set
   properly. If you are using firewall support, see the section on the
   removal of firemode= and its consequences.
6. Start the new client up, you're ready to roll!
