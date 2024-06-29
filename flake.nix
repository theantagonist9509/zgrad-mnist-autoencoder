{
  outputs = { nixpkgs, ... }:
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs;[ libGL xorg.libX11 ];
      shellHook = ''
        export PS1="\n\e[0;36m[raylib:\w]\$\e[0m "
        export LD_LIBRARY_PATH=${pkgs.libGL}/lib:${pkgs.xorg.libX11}/lib:$LD_LIBRARY_PATH
      '';
    };
  };
}
