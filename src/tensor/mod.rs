use std::fmt::Debug;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor<T: Clone + Debug> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T: Clone + Debug + Default> Tensor<T> {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![T::default(); size],
        }
    }
}

impl<T: Clone + Debug> Tensor<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Self {
        Self { shape, data }
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        assert_eq!(
            self.size(),
            shape.iter().product(),
            "Cannot reshape from {:?} to {:?}",
            self.shape,
            shape
        );
        Self {
            shape,
            data: self.data.clone(),
        }
    }
}
