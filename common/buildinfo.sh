#!/bin/sh
#
# Copyright distributed.net 2026 - All Rights Reserved
# For use in distributed.net projects only.
# Any other distribution or use of this source violates copyright.
#
# Prints the compiler flags that tell common/cliident.cpp which source
# revisions the client is built from and when it was built. The makefile
# passes them when compiling cliident.cpp only:
#
#   -DBUILDINFO_TIME=<epoch>           time of this build, UTC seconds
#   -DBUILDINFO_PUB_HASH=0x<hex>UL     public repo HEAD, first 8 hex digits
#   -DBUILDINFO_PUB_DIRTY=0|1          tracked files differ from HEAD
#   -DBUILDINFO_PUB_TIME=<epoch>       commit time of HEAD
#   -DBUILDINFO_PRIV_HASH etc.         the same for the private repo
#
# Only numbers are passed, so no quoting is needed in any make dialect.
# A repo's flags are left out when it is not a git checkout of its own
# (an exported copy, say). The private repo is the directory above the
# public one, and is looked at only when its sources are present
# (../private/buffpriv.cpp, as tested by makefile.vc).
# Setting SOURCE_DATE_EPOCH overrides the build time.
#
# plat/win/buildinfo.cmd is the equivalent for makefile.vc.

dir=`dirname "$0"`
cd "$dir/.." || exit 0

repo_flags () {   # $1 = directory, $2 = PUB or PRIV
  (
    cd "$1" 2>/dev/null || exit 0
    # Use git only if this directory is the top of its own work tree,
    # so that an enclosing repo (the private one) can't answer for it.
    git rev-parse --git-dir >/dev/null 2>&1 || exit 0
    [ -z "`git rev-parse --show-cdup 2>/dev/null`" ] || exit 0
    hash=`git rev-parse --verify -q HEAD 2>/dev/null | cut -c1-8`
    [ -n "$hash" ] || exit 0
    # Untracked files don't count, as with "git describe --dirty".
    git update-index -q --refresh >/dev/null 2>&1
    if git diff-index --quiet --ignore-submodules HEAD -- 2>/dev/null; then
      dirty=0
    else
      dirty=1
    fi
    flags="-DBUILDINFO_$2_HASH=0x${hash}UL -DBUILDINFO_$2_DIRTY=$dirty"
    when=`git log -1 --format=%ct HEAD 2>/dev/null`
    case "$when" in ''|*[!0-9]*) ;; *) flags="$flags -DBUILDINFO_$2_TIME=$when" ;; esac
    echo "$flags"
  )
}

flags=`repo_flags . PUB`
if [ -f ../private/buffpriv.cpp ]; then
  flags="$flags `repo_flags .. PRIV`"
fi

now="$SOURCE_DATE_EPOCH"
case "$now" in ''|*[!0-9]*) now=`date -u +%s 2>/dev/null` ;; esac
case "$now" in ''|*[!0-9]*) now=`awk 'BEGIN { srand(); print srand() }' 2>/dev/null` ;; esac
case "$now" in ''|*[!0-9]*) ;; *) flags="$flags -DBUILDINFO_TIME=$now" ;; esac

echo $flags
exit 0
