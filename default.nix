{
  withHoogle ? true
}:
let
  inherit (import <nixpkgs> {}) fetchFromGitHub;
  nixpkgs = builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/20.03.tar.gz";
    sha256 = "0182ys095dfx02vl2a20j1hz92dx3mfgz2a6fhn31bqlp1wa8hlq";
  };

  config = {
    packageOverrides = pkgs: rec {
      haskellPackages = pkgs.haskellPackages.override {
        overrides = self: super: rec {
          ghc =
            super.ghc // { withPackages = if withHoogle then super.ghc.withHoogle else super.ghc ; };
          ghcWithPackages =
            self.ghc.withPackages;
          streamly-extras =
            self.callPackage ./streamly-extras.nix { };
          fusion-plugin-types =
            pkgs.haskell.lib.dontCheck
              (self.callCabal2nix "fusion-plugin-types" (pkgs.fetchgit {
                url = "https://github.com/composewell/fusion-plugin-types.git";
                rev = "1a7e1c39b4496543b2dc95d59aafbf44041554f1";
                sha256 = "1mmph6gawi4cbsqmswawi1c951f6pq41qfqjbvnygm36d2qfv64i";
                fetchSubmodules = true;
              }) { });
          streamly =
            pkgs.haskell.lib.dontCheck
              (self.callCabal2nix "streamly" (pkgs.fetchgit {
                url = "https://github.com/composewell/streamly.git";
                rev = "500d187b8fb5b6b1bc424725dd179650fb41c49c";
                sha256 = "175cqn0sxg8r32f3dd0alkivkvjrwhmwm7xkmf4ki2j68smmjc3y";
                fetchSubmodules = true;
              }) { });

        };
      };
    };
  };
  pkgs = import nixpkgs { inherit config; };
  drv = pkgs.haskellPackages.streamly-extras;
in
  if pkgs.lib.inNixShell
    then
      drv.env.overrideAttrs(attrs:
        { buildInputs =
          with pkgs.haskellPackages;
          [
            cabal-install
            cabal2nix
            ghcid
            hindent
            hlint
          ] ++ [ zlib ] ++ attrs.buildInputs;
        })
        else drv
