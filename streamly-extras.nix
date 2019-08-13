{ mkDerivation, base, containers, stdenv, stm, streamly }:
mkDerivation {
  pname = "streamly-extras";
  version = "0.0.1";
  src = ./.;
  libraryHaskellDepends = [ base containers stm streamly ];
  homepage = "https://github.com/juspay/streamly-extras";
  description = "To provide extra utility functions on top of Streamly";
  license = stdenv.lib.licenses.bsd3;
}
