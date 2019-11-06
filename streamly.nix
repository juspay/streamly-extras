{ mkDerivation, atomic-primops, base, containers, deepseq
, exceptions, fetchFromGitHub, gauge, ghc-prim, heaps, hspec
, lockfree-queue, monad-control, mtl, network, QuickCheck, random
, stdenv, transformers, transformers-base, typed-process
}:
mkDerivation {
  pname = "streamly";
  version = "0.6.1";
  src = fetchFromGitHub {
    owner = "composewell";
    repo = "streamly";
    rev = "be920a2bfa906d85ad7c41b6e6b9a7ce731ac774";
    sha256 = "1di6d2b9rxcgf6jrg57r0hhkasbr7b181v96a177spjw23j5sxv9";
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
