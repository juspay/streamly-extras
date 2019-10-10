{ mkDerivation, base, containers, mtl, stdenv, stm, streamly }:
mkDerivation {
  pname = "streamly-extras";
  version = "0.0.1";
  src = ./.;
  libraryHaskellDepends = [ base containers mtl stm streamly ];
  homepage = "https://github.com/juspay/streamly-extras";
  description = "To provide extra utility functions on top of Streamly";
  license = stdenv.lib.licenses.bsd3;
}
