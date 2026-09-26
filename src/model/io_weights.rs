//! Owning embedding/head checkpoint I/O (implementation follows contract tests).
use std::sync::Arc;
use wgpu::{Device,Queue};
use crate::error::{FerrisResError,Result};
use super::{embedding::TokenEmbedding,lm_head::LMHead};

pub enum HeadWeights<'a> { Untied(&'a [f32]), TiedToEmbedding }
pub struct ModelIo { embedding: TokenEmbedding, head: LMHead }
impl ModelIo {
    pub fn from_weights(_device:Arc<Device>,_queue:Arc<Queue>,_vocab:usize,_hidden:usize,_embedding:&[f32],_head:HeadWeights<'_>)->Result<Self> {
        Err(FerrisResError::Unsupported("model I/O ownership not implemented".into()))
    }
    pub fn embedding(&self)->&TokenEmbedding { &self.embedding }
    pub fn lm_head(&self)->&LMHead { &self.head }
}
