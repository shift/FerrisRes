#!/bin/bash
set -e
# Build the ferrisres .deb package
# Usage: ./build-deb.sh [output-dir]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${2:-${ROOT_DIR}/target/deb}"

# Version: explicit argument, else the crate version. The binary and the
# package must agree; CI passes the value read from Cargo.toml.
if [ -n "${1:-}" ]; then
  VERSION="$1"
else
  VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "${ROOT_DIR}/Cargo.toml" | head -1)"
fi
[ -n "$VERSION" ] || { echo "ERROR: cannot determine version" >&2; exit 1; }
ARCH="amd64"
PKG_NAME="ferrisres_${VERSION}_${ARCH}"

echo "Building ferrisres ${VERSION} .deb package..."

# Create package structure
mkdir -p "${OUTPUT_DIR}/${PKG_NAME}/DEBIAN"
mkdir -p "${OUTPUT_DIR}/${PKG_NAME}/usr/bin"
mkdir -p "${OUTPUT_DIR}/${PKG_NAME}/usr/share/vulkan/icd.d"
mkdir -p "${OUTPUT_DIR}/${PKG_NAME}/usr/share/doc/ferrisres"

# Copy binary (must be built first: cargo build --release --features vulkan)
if [ ! -f "${ROOT_DIR}/target/release/ferrisres" ]; then
    echo "ERROR: target/release/ferrisres not found. Run: cargo build --release --no-default-features --features vulkan"
    exit 1
fi

cp "${ROOT_DIR}/target/release/ferrisres" "${OUTPUT_DIR}/${PKG_NAME}/usr/bin/ferrisres"
strip "${OUTPUT_DIR}/${PKG_NAME}/usr/bin/ferrisres" 2>/dev/null || true
chmod 755 "${OUTPUT_DIR}/${PKG_NAME}/usr/bin/ferrisres"

# Portability guard. Colab and Kaggle run Ubuntu 22.04 (GLIBC 2.35). A binary
# linked against a newer glibc builds here and fails there at the loader.
# Build the package on ubuntu-22.04 (CI); a NixOS build is not shippable.
GLIBC_MAX="$(objdump -T "${OUTPUT_DIR}/${PKG_NAME}/usr/bin/ferrisres" 2>/dev/null \
  | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sed 's/GLIBC_//' | sort -V | tail -1)"
if [ -n "$GLIBC_MAX" ] && [ "$(printf '%s\n2.35\n' "$GLIBC_MAX" | sort -V | tail -1)" != "2.35" ]; then
  echo "ERROR: the binary requires GLIBC_${GLIBC_MAX}; Colab/Kaggle need <= 2.35." >&2
  echo "Build the .deb on ubuntu-22.04 (CI), not on this host." >&2
  if [ "${FERRISRES_DEB_ALLOW_GLIBC:-0}" != "1" ]; then
    exit 1
  fi
  echo "WARNING: continuing because FERRISRES_DEB_ALLOW_GLIBC=1; this package will not run on Kaggle." >&2
fi

# NVIDIA Vulkan ICD — makes GPU visible on Colab/AWS/GCP cloud instances
# where driver libs exist but ICD JSON is missing
cp "${SCRIPT_DIR}/nvidia_icd.x86_64.json" \
   "${OUTPUT_DIR}/${PKG_NAME}/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json"
chmod 644 "${OUTPUT_DIR}/${PKG_NAME}/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json"

# Compute installed size
INSTALLED_SIZE=$(du -sk "${OUTPUT_DIR}/${PKG_NAME}" | cut -f1)

# Write control file
cat > "${OUTPUT_DIR}/${PKG_NAME}/DEBIAN/control" << EOF
Package: ferrisres
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Depends: libvulkan1 (>= 1.3)
Recommends: vulkan-tools
Suggests: nvidia-driver-535 | nvidia-driver-525 | nvidia-driver-470
Installed-Size: ${INSTALLED_SIZE}
Maintainer: shift <shift@users.noreply.github.com>
Homepage: https://github.com/shift/FerrisRes
Description: Block AttnRes inference and training engine
 FerrisRes is a Rust-native AI inference and training engine built around
 Block AttnRes (linear-time transformer). Supports Gemma 4, LLaMA, Mistral,
 Phi-3, and Qwen models via safetensors or GGUF format. Profile-driven GPU
 dispatch with automatic tiling for constrained VRAM.
 .
 This package ships the NVIDIA Vulkan ICD configuration file for
 cloud GPU instances (Google Colab T4/A100, AWS, GCP) where the driver
 libraries are present but the ICD JSON is not installed by default.
EOF

# Write postinst — verify GPU after install
cat > "${OUTPUT_DIR}/${PKG_NAME}/DEBIAN/postinst" << 'EOF'
#!/bin/bash
set -e
echo "FerrisRes installed. Verifying GPU..."
if command -v vulkaninfo &>/dev/null; then
    GPU=$(vulkaninfo --summary 2>/dev/null | grep "deviceName" | head -1 | sed 's/.*= *//')
    if [ -n "$GPU" ]; then
        echo "  GPU detected: $GPU"
    else
        echo "  WARNING: No Vulkan GPU detected. Install NVIDIA driver or run on a GPU instance."
    fi
else
    echo "  (vulkaninfo not found, install vulkan-tools to verify GPU)"
fi
echo "Run: ferrisres info"
EOF
chmod 755 "${OUTPUT_DIR}/${PKG_NAME}/DEBIAN/postinst"

# Build .deb
dpkg-deb --build "${OUTPUT_DIR}/${PKG_NAME}"

# Clean up build tree
rm -rf "${OUTPUT_DIR}/${PKG_NAME}"

DEB_FILE="${OUTPUT_DIR}/${PKG_NAME}.deb"
echo ""
echo "Built: ${DEB_FILE}"
echo "Size:  $(du -h "${DEB_FILE}" | cut -f1)"
echo ""
echo "Install:  sudo dpkg -i ${DEB_FILE}"
echo "Colab:    dpkg -i ${DEB_FILE}  (no sudo needed)"
