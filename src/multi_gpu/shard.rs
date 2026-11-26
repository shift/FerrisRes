//! Deterministic batch ownership by `(rank, global step)`.
//!
//! At every global step the `world_size` ranks take one contiguous group of
//! `world_size` batch indices, one each. Groups do not overlap and, over the
//! run, cover every batch exactly once. The seed rotates which rank receives
//! which batch inside each group; that decorrelates rank placement between
//! runs without changing coverage, and a resumed plan at step `k` reproduces
//! the same `(rank, batch)` pairs because the mapping is a pure function.

use crate::multi_gpu::error::MultiGpuError;

/// Batch ownership for one rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardPlan {
    world_size: usize,
    rank: usize,
    seed: u64,
}

/// splitmix64's finalizer.
///
/// A plain integer mix, so the rotation is identical in every process and is
/// not affected by `HashMap` seeding or thread scheduling.
fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

impl ShardPlan {
    pub fn new(world_size: usize, rank: usize, seed: u64) -> Result<Self, MultiGpuError> {
        if world_size == 0 {
            return Err(MultiGpuError::ZeroWorldSize);
        }
        if rank >= world_size {
            return Err(MultiGpuError::RankOutOfRange { rank, world_size });
        }
        Ok(Self {
            world_size,
            rank,
            seed,
        })
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The batch this rank processes at `global_step`.
    ///
    /// Batch id `step * world_size + slot` keeps each step's group contiguous
    /// and the groups disjoint. The seed only chooses the slot.
    pub fn batch_index(&self, global_step: u64) -> u64 {
        let step = global_step.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let rotation = (mix(self.seed ^ step) % self.world_size as u64) as usize;
        let slot = (self.rank + rotation) % self.world_size;
        global_step * self.world_size as u64 + slot as u64
    }

    /// The `(global step, batch index)` sequence this rank owns, starting at
    /// `start_step`.
    ///
    /// Restarting at `start_step` after a pre-emption yields exactly the tail of
    /// the uninterrupted sequence, because nothing is stored between calls.
    pub fn batches_from(&self, start_step: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        (start_step..).map(move |step| (step, self.batch_index(step)))
    }
}

#[cfg(test)]
mod tests {
    use super::ShardPlan;
    use crate::multi_gpu::error::MultiGpuError;
    use std::collections::BTreeSet;

    fn plans(world_size: usize, seed: u64) -> Vec<ShardPlan> {
        (0..world_size)
            .map(|rank| ShardPlan::new(world_size, rank, seed).unwrap())
            .collect()
    }

    #[test]
    fn union_over_ranks_covers_every_batch_once() {
        let world_size = 3;
        let steps = 40u64;
        let plans = plans(world_size, 7);

        let mut covered = BTreeSet::new();
        for step in 0..steps {
            for plan in &plans {
                covered.insert(plan.batch_index(step));
            }
        }
        let expected: BTreeSet<u64> = (0..steps * world_size as u64).collect();
        assert_eq!(covered, expected, "coverage must be exactly 0..steps*world");
    }

    #[test]
    fn ranks_are_disjoint_at_every_step() {
        let world_size = 4;
        let plans = plans(world_size, 11);
        for step in 0..100u64 {
            let mut owned = BTreeSet::new();
            for plan in &plans {
                assert!(
                    owned.insert(plan.batch_index(step)),
                    "step {step} assigned one batch to two ranks"
                );
            }
            assert_eq!(owned.len(), world_size);
        }
    }

    #[test]
    fn same_seed_reproduces_the_same_sequence() {
        let a = ShardPlan::new(4, 2, 1_234_567).unwrap();
        let b = ShardPlan::new(4, 2, 1_234_567).unwrap();
        let first: Vec<_> = a.batches_from(0).take(50).collect();
        let second: Vec<_> = b.batches_from(0).take(50).collect();
        assert_eq!(first, second);
    }

    #[test]
    fn different_seed_changes_at_least_one_assignment() {
        let a = ShardPlan::new(4, 0, 1).unwrap();
        let b = ShardPlan::new(4, 0, 2).unwrap();
        assert!(
            (0..64).any(|step| a.batch_index(step) != b.batch_index(step)),
            "a different seed must move at least one batch"
        );
    }

    #[test]
    fn resume_at_step_k_continues_the_same_sequence() {
        let full: Vec<_> = ShardPlan::new(2, 1, 99)
            .unwrap()
            .batches_from(0)
            .take(60)
            .collect();

        // A resumed process constructs a fresh plan and skips to step 37.
        let resumed: Vec<_> = ShardPlan::new(2, 1, 99)
            .unwrap()
            .batches_from(37)
            .take(23)
            .collect();

        assert_eq!(resumed, full[37..], "resume must be the tail of the run");
    }

    #[test]
    fn iterator_agrees_with_batch_index() {
        let plan = ShardPlan::new(3, 2, 5).unwrap();
        for (step, batch) in plan.batches_from(10).take(20) {
            assert_eq!(batch, plan.batch_index(step));
        }
    }

    #[test]
    fn zero_world_size_is_refused() {
        assert!(matches!(
            ShardPlan::new(0, 0, 1),
            Err(MultiGpuError::ZeroWorldSize)
        ));
    }

    #[test]
    fn out_of_range_rank_is_refused() {
        assert!(matches!(
            ShardPlan::new(2, 2, 1),
            Err(MultiGpuError::RankOutOfRange {
                rank: 2,
                world_size: 2
            })
        ));
    }
}
