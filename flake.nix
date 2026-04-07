{
  description = "FerrisRes: Rust/Vulkan Block AttnRes Engine Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      # You can override this to aarch64-darwin for Apple Silicon Macs
      # or aarch64-linux for ARM-based NPUs
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
          
          # Shader compilation
          shaderc # Essential for compiling GLSL/HLSL to SPIR-V for compute shaders
          
          # Utilities
          pkg-config
          cmake
          fontconfig
        ];

        # Required for the Rust application to find the Vulkan loader at runtime
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
          vulkan-loader
          wayland
          libxkbcommon
        ]);

        shellHook = ''
          echo "🦀 FerrisRes development environment activated."
          echo "🌋 Vulkan SDK tools available (try running vulkaninfo)."
          echo "🛠️  Rust version:" $(rustc --version)
        '';
      };
    };
}
