#! /usr/bin/perl -w

sub usage
{
  print "Usage: $0 infile outfile arrayname\n";
  exit(1);
}

$inf = $ARGV[0] or usage;
$outf = $ARGV[1] or usage;
$name = $ARGV[2] or usage;
open(INF, $inf) or die "Could not open $inf: $!";
binmode INF;
open(OUTF, ">$outf") or die "Could not create $outf: $!";
print OUTF "unsigned char ${name}[] = {\n";
while ($n = read INF, $buf, 16) {
  for $i (0..$n-1) {
    print OUTF sprintf("0x%02X,", unpack("c", substr($buf, $i, 1)) & 0xff);
  }
  print OUTF "\n";
}
print OUTF "};\n";
close(INF);
close(OUTF);
