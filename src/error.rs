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

pub type Result<T> = std::result::Result<T, FerrisResError>;
