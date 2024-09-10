use burn::prelude::*;
use rayon::prelude::*;
use rand::thread_rng;
use rand_distr::{num_traits::ToPrimitive, Distribution, WeightedIndex};

pub fn multinominal_sample<B: Backend>(weights: Tensor<B, 1, Float>, num_samples: usize) -> Tensor<B, 1> {
    let device = weights.device();
    let weights_vec = tensor_to_vec(weights);
    let dist = WeightedIndex::new(&weights_vec).unwrap();
    let mut rng = thread_rng();
    let weighted_vec: Vec<f32> = (0..num_samples)
        .map(|_| dist.sample(&mut rng))
        .map(|x| x as f32).collect();
    Tensor::<B, 1>::from_floats(weighted_vec.as_slice(), &device)
}

pub fn tensor_to_vec<B: Backend>(tensor: Tensor<B, 1>) -> Vec<f32> {
    let tensor_vec_data = tensor.to_data().value;
    let vec: Vec<f32> = tensor_vec_data.par_iter()
        .map(|x| x.to_f32().unwrap()).collect();
    vec
}