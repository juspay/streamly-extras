{ mkDerivation, atomic-primops, base, containers, deepseq
, exceptions, fetchgit, gauge, ghc-prim, heaps, hspec
, lockfree-queue, monad-control, mtl, network, QuickCheck, random
, stdenv, transformers, transformers-base, typed-process
}:
mkDerivation {
  pname = "streamly";
  version = "0.6.1";
  src = fetchgit {
    url = "https://github.com/composewell/streamly.git";
    rev = "fbec11f24deda94a4e55dc4bec5d4c16d3db3d0c";
    sha256 = "0a63dldjqgcpddjbqp43bcw1zsi87fbwlfqb1cg46b6nfirj2adv";
    fetchSubmodules = true;
  };
  isLibrary = true;
  isExecutable = true;
  libraryHaskellDepends = [
    atomic-primops base containers deepseq exceptions ghc-prim heaps
    lockfree-queue monad-control mtl network transformers
    transformers-base
  ];
  testHaskellDepends = [
    base containers exceptions hspec mtl QuickCheck random transformers
  ];
  benchmarkHaskellDepends = [
    base deepseq gauge random typed-process
  ];
  homepage = "https://github.com/composewell/streamly";
  description = "Beautiful Streaming, Concurrent and Reactive Composition";
  license = stdenv.lib.licenses.bsd3;
  doCheck = false;
  doHaddock = true;
}
