{ mkDerivation, atomic-primops, base, containers, deepseq
, exceptions, fetchFromGitHub, gauge, ghc-prim, heaps, hspec
, lockfree-queue, monad-control, mtl, network, QuickCheck, random
, stdenv, transformers, transformers-base, typed-process
}:
mkDerivation {
  pname = "streamly";
  version = "0.7.0";
  src = fetchFromGitHub {
    owner = "composewell";
    repo = "streamly";
    rev = "v0.7.0";
    sha256 = "10qm72l7r4drqsajqrg3i1pqdi1bscz8p3k23vpi2ahrscd9kfdz";
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
