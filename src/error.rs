use thiserror::Error;

#[derive(Error, Debug)]
pub enum FerrisResError {
    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("Tensor shape error: {0}")]
    Shape(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error("Shader compilation error: {0}")]
    Shader(String),

    #[error("Out of memory: {0}")]
    OOM(String),

    #[error("Not supported: {0}")]
    Unsupported(String),
}

impl From<std::io::Error> for FerrisResError {
    fn from(e: std::io::Error) -> Self {
        FerrisResError::Device(format!("IO error: {}", e))
    }
}

impl From<String> for FerrisResError {
    fn from(e: String) -> Self {
        FerrisResError::Shape(e)
    }
}

impl From<crate::multi_gpu::MultiGpuError> for FerrisResError {
    fn from(e: crate::multi_gpu::MultiGpuError) -> Self {
        // The multi-GPU layer's messages already name the rank, step and field;
        // keep them verbatim so a refusal survives the conversion.
        FerrisResError::Unsupported(format!("multi-gpu: {e}"))
    }
}

pub type Result<T> = std::result::Result<T, FerrisResError>;
