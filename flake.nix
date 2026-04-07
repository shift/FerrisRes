{
  description = "FerrisRes: Rust/Vulkan Block AttnRes Engine Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      system = "x86_64-linux";
      overlays = [ (import rust-overlay) ];
      pkgs = import nixpkgs { inherit system overlays; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          # Rust Toolchain
          (rust-bin.stable.latest.default.override {
            extensions = [ "rust-src" "rust-analyzer" ];
          })

          # Vulkan dependencies for compilation and runtime
          vulkan-headers
          vulkan-loader
          vulkan-tools
          vulkan-validation-layers

          # GPU drivers (Mesa for Intel/AMD)
          mesa
          libdrm

          # Shader compilation
          shaderc

          # Utilities
          pkg-config
          cmake
          fontconfig
          lsof
        ];

        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
          vulkan-loader
          mesa
          libdrm
          wayland
          libxkbcommon
        ]);

        VK_DRIVER_FILES = "${pkgs.mesa}/share/vulkan/icd.d:${pkgs.vulkan-loader}/share/vulkan/icd.d";

        shellHook = ''
          echo "FerrisRes development environment activated."
          echo "Vulkan SDK tools available (try running vulkaninfo)."
          echo "Rust version:" $(rustc --version)
        '';
      };
    };
}
