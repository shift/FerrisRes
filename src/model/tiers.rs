//! Model tier presets.
//!
//! A tier is a *shape*, not a checkpoint: the same engine runs every tier, and
//! the differences are declared per checkpoint. These presets are the four
//! supported shapes. They are used by `ferrisres capabilities --tier <name>`
//! and by tests; no other CLI wiring exists in this milestone.
//!
//! The loop is described as `prelude + core * passes + coda` layer *visits*,
//! while `layers` is the number of unique layers. A looped tier therefore has
//! more visits than unique layers.

/// Supported model shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Nano,
    Small,
    Base,
    Large,
}

/// The concrete dimensions of a tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierSpec {
    pub hidden: usize,
    pub layers: usize,
    pub prelude: usize,
    pub core: usize,
    pub passes: usize,
    pub coda: usize,
    pub experts: usize,
    pub vocab: usize,
    pub intermediate: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub window: usize,
    pub globals: usize,
    pub context_max: usize,
}

impl Tier {
    /// Parse a tier name, case-insensitively.
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "nano" => Some(Tier::Nano),
            "small" => Some(Tier::Small),
            "base" => Some(Tier::Base),
            "large" => Some(Tier::Large),
            _ => None,
        }
    }

    /// Stable lower-case name, as used in checkpoint feature strings.
    pub fn name(self) -> &'static str {
        match self {
            Tier::Nano => "nano",
            Tier::Small => "small",
            Tier::Base => "base",
            Tier::Large => "large",
        }
    }

    /// The dimensions of this tier.
    pub fn spec(self) -> TierSpec {
        match self {
            Tier::Nano => TierSpec {
                hidden: 512,
                layers: 6,
                prelude: 3,
                core: 1,
                passes: 2,
                coda: 2,
                experts: 2,
                vocab: 32768,
                intermediate: 1408,
                kv_heads: 1,
                head_dim: 32,
                window: 512,
                globals: 1,
                context_max: 32768,
            },
            Tier::Small => TierSpec {
                hidden: 768,
                layers: 8,
                prelude: 4,
                core: 2,
                passes: 2,
                coda: 2,
                experts: 2,
                vocab: 32768,
                intermediate: 2048,
                kv_heads: 1,
                head_dim: 48,
                window: 1024,
                globals: 2,
                context_max: 32768,
            },
            Tier::Base => TierSpec {
                hidden: 1024,
                layers: 12,
                prelude: 6,
                core: 2,
                passes: 3,
                coda: 4,
                experts: 4,
                vocab: 151936,
                intermediate: 2816,
                kv_heads: 1,
                head_dim: 64,
                window: 1024,
                globals: 3,
                context_max: 32768,
            },
            Tier::Large => TierSpec {
                hidden: 1024,
                layers: 12,
                prelude: 6,
                core: 2,
                passes: 3,
                coda: 4,
                experts: 8,
                vocab: 151936,
                intermediate: 2816,
                kv_heads: 1,
                head_dim: 64,
                window: 1024,
                globals: 3,
                context_max: 32768,
            },
        }
    }

    /// Layer visits: `prelude + core * passes + coda`.
    pub fn visits(self) -> usize {
        let s = self.spec();
        s.prelude + s.core * s.passes + s.coda
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_names_roundtrip() {
        for tier in [Tier::Nano, Tier::Small, Tier::Base, Tier::Large] {
            assert_eq!(Tier::parse(tier.name()), Some(tier));
            assert_eq!(Tier::parse(&tier.name().to_ascii_uppercase()), Some(tier));
        }
        assert_eq!(Tier::parse("huge"), None);
    }

    #[test]
    fn tier_visits_match_the_table() {
        assert_eq!(Tier::Nano.visits(), 7);
        assert_eq!(Tier::Small.visits(), 10);
        assert_eq!(Tier::Base.visits(), 16);
        assert_eq!(Tier::Large.visits(), 16);
    }

    #[test]
    fn tier_specs_have_consistent_layer_counts() {
        for tier in [Tier::Nano, Tier::Small, Tier::Base, Tier::Large] {
            let s = tier.spec();
            assert_eq!(s.layers, s.prelude + s.core + s.coda, "{:?}", tier);
            assert!(s.passes >= 2, "{:?} is not looped", tier);
            assert!(tier.visits() > s.layers, "{:?} visits", tier);
        }
    }

    #[test]
    fn large_differs_from_base_only_in_experts() {
        let base = Tier::Base.spec();
        let large = Tier::Large.spec();
        assert_eq!(base.hidden, large.hidden);
        assert_eq!(base.layers, large.layers);
        assert_eq!(base.experts * 2, large.experts);
    }
}
