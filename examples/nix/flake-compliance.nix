{
  description = "FerrisRes Chief Architect Enterprise Compliance Dev Environment";
  # Enterprise dev shell for Chief Architects implementing compliance in FerrisRes

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
          # Rust Toolchain - FerrisRes core
          (rust-bin.stable.latest.default.override {
            extensions = [ "rust-src" "rust-analyzer" "fmt" "clippy" ];
          })
          cargo
          rustfmt

          # Vulkan (FerrisRes GPU acceleration)
          vulkan-headers
          vulkan-loader
          vulkan-tools

          # K8s SME tools (audit streams, webhook compliance)
          kubectl
          helm
          k9s
          cilium-cli

          # OPA - Rego policy testing (compliance oracle)
          opa

          # Python - stream simulation (HL7, Kafka, WebSockets)
          python311
          python311Packages.pip
          python311Packages.websockets
          python311Packages.kafka-python-ng

          # Event streaming
          confluent-kafka
          kafka

          # Utilities
          pkg-config
          cmake
          git
          curl
          wget
        ];

        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
          vulkan-loader
          confluent-kafka
        ]);

        # Active modules summary
        shellHook = ''
          echo "╔═══════════════════════════════════════════════════════════╗"
          echo "║   FerrisRes Enterprise Compliance Environment       ║"
          echo "╠═══════════════════════════════════════════════════════════╣"
          echo "║ ACTIVE MODULES:                                ║"
          echo "║  • FerrisRes Core: Rust + Vulkan             ║"
          echo "║  • K8s Audit: kubectl, helm, k9s         ║"
          echo "║  • Compliance: opa (Rego)                ║"
          echo "║  • Streaming: Kafka + HL7 + WebSocket     ║"
          echo "║  • Policy: BaFin, GDPR, NIS2, HIPAA      ║"
          echo "╚═══════════════════════════════════════════════════╝"
          echo ""
          echo "Ready for compliance implementation."
        '';
      };
    };
}