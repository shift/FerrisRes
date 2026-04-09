pub fn sample_argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .reduce(|(i, a), (j, b)| if b > a { (j, b) } else { (i, a) })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub fn sample_temperature(logits: &mut [f32], temperature: f32) -> usize {
    if temperature > 0.0 {
        for v in logits.iter_mut() {
            *v /= temperature;
        }
    }
    sample_argmax(logits)
}

pub fn sample_top_k(logits: &mut [f32], k: usize) -> usize {
    let k = k.min(logits.len());
    if k >= logits.len() {
        softmax_inplace(logits);
        return sample_argmax(logits);
    }

    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    softmax_inplace_values(&mut indexed);

    let total: f32 = indexed.iter().map(|(_, p)| p).sum();
    let mut rng = rand::thread_rng();
    let mut r: f32 = rand::Rng::gen_range(&mut rng, 0.0..total);
    for (idx, p) in indexed.iter() {
        r -= p;
        if r <= 0.0 {
            return *idx;
        }
    }
    indexed.last().map(|(i, _)| *i).unwrap_or(0)
}

pub fn sample_top_p(logits: &mut [f32], p: f32) -> usize {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0f32;
    let mut cutoff = indexed.len();
    for (i, &(_, v)) in indexed.iter().enumerate() {
        cumulative += v;
        if cumulative >= p {
            cutoff = i + 1;
            break;
        }
    }
    indexed.truncate(cutoff);

    let max_val = indexed.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for (_, v) in indexed.iter_mut() {
        *v = (*v - max_val).exp();
        sum_exp += *v;
    }
    for (_, v) in indexed.iter_mut() {
        *v /= sum_exp;
    }

    let mut rng = rand::thread_rng();
    let mut r: f32 = rand::Rng::gen_range(&mut rng, 0.0..1.0);
    for (idx, prob) in indexed.iter() {
        r -= prob;
        if r <= 0.0 {
            return *idx;
        }
    }
    indexed.last().map(|(i, _)| *i).unwrap_or(0)
}

fn softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    for v in logits.iter_mut() {
        *v /= sum;
    }
}

fn softmax_inplace_values(indexed: &mut [(usize, f32)]) {
    if indexed.is_empty() {
        return;
    }
    let max_val = indexed.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, v) in indexed.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    for (_, v) in indexed.iter_mut() {
        *v /= sum;
    }
}
