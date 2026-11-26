# Packaging

The `.deb` is the Colab/Kaggle install path. It ships the binary, the NVIDIA
Vulkan ICD (`nvidia_icd.x86_64.json`) and a `libvulkan1` dependency, because
those images have the driver but not the ICD JSON.

## Where it is built

**On `ubuntu-22.04`, in CI. Never on NixOS.** The Colab and Kaggle images are
22.04 with glibc 2.35; a binary linked against a newer glibc builds locally and
fails at the loader on the target. `build-deb.sh` refuses a package whose
`GLIBC_*` requirement is above 2.35 unless
`FERRISRES_DEB_ALLOW_GLIBC=1` is set; that escape is for a local experiment and
the package will not run on Kaggle.

## Version

The version is the crate version from `Cargo.toml`, not a number typed into the
workflow. `build-deb.sh` reads it when no argument is given; CI reads it into a
step output; the release workflow refuses a tag `vX.Y.Z` that does not match
`Cargo.toml`'s `X.Y.Z`. A package whose version does not match its binary is a
support case waiting to happen.

## How a release is produced

1. Merge the change to `main`.
2. Bump `Cargo.toml` if needed, then tag `vX.Y.Z` matching it.
3. The release workflow builds on `ubuntu-22.04`, runs `build-deb.sh`, and
   attaches `ferrisres_X.Y.Z_amd64.deb` to the GitHub release.
4. The CI `deb` job also builds on every push to `main` as an artifact, but that
   artifact expires in 30 days; the release asset is the stable URL.

For a pre-release from a feature branch (for example the Qwen/teacher-benchmark
build), tag that branch with a matching version and mark the release
pre-release. The notebook then points at the asset, not at a CI artifact.

## Install and verify

```bash
sudo dpkg -i ferrisres_X.Y.Z_amd64.deb
ferrisres info
ferrisres devices        # lists adapters; needs the gpu-selection build
```

On Kaggle, use the `.deb` from a private dataset at
`/kaggle/input/ferrisres-deb/`; the workbooks in `notebooks/kaggle/` expect
that path.

## What the package does not do

It does not install a driver, a tokenizer, a model or a licence. The tokenizer
and GGUF are inputs; the licence is `license-verify` for a shippable build (the
test bot uses `unlicensed-release`, which must never be published).
