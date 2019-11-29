{ mkDerivation, base, containers, monad-control, mtl, stdenv, stm
, streamly, network, bytestring, exceptions, streaming-commons
}:
mkDerivation {
  pname = "streamly-extras";
  version = "0.0.1";
  src = ./.;
  isLibrary = true;
  isExecutable = true;
  libraryHaskellDepends = [
    base containers monad-control mtl stm streamly network bytestring exceptions streaming-commons
  ];
  homepage = "https://github.com/juspay/streamly-extras";
  description = "To provide extra utility functions on top of Streamly";
  license = stdenv.lib.licenses.bsd3;
}
