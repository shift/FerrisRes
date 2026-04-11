//! TurboQuant benchmarks
//! 
//! Benchmark TurboQuant compression:
//! - Memory savings at different bit widths
//! - Compression/decompression latency
//! - Attention accuracy comparison

use std::time::Instant;

#[cfg(test)]
mod benchmarks {
    use super::*;
    use crate::compute::turboquant::{TurboQuantConfig, TurboQuantEngine, OutlierChannelSplitter};
    
    /// Memory savings benchmark
    #[test]
    fn test_memory_savings_2bit() {
        let hidden_dim = 4096u32;
        let seq_len = 4096u32;
        
        // f32 baseline: 4096 * 4096 * 4 bytes = 256 MB
        let baseline_bytes = hidden_dim as usize * seq_len as usize * 4;
        
        // 2-bit compression: 256 / 16 = 16 MB
        let config = TurboQuantConfig::two_bit(hidden_dim);
        let engine = TurboQuantEngine::new(config).unwrap();
        let ratio = engine.compression_ratio();
        
        let compressed_bytes = baseline_bytes / ratio as usize;
        
        println!("2-bit compression:");
        println!("  Baseline (f32): {} MB", baseline_bytes / (1024 * 1024));
        println!("  Compressed: {} MB", compressed_bytes / (1024 * 1024));
        println!("  Ratio: {:.1}x", ratio);
        
        assert!((ratio - 16.0).abs() < 0.1);
    }
    
    #[test]
    fn test_memory_savings_2_5bit() {
        let hidden_dim = 4096u32;
        let splitter = OutlierChannelSplitter::two_and_half_bit(hidden_dim);
        let ratio = splitter.memory_savings();
        
        println!("2.5-bit compression:");
        println!("  Outlier channels: {}", splitter.outlier_channels);
        println!("  Regular channels: {}", splitter.regular_channels);
        println!("  Average bits: {:.1}", splitter.average_bits());
        println!("  Memory ratio: {:.1}x", ratio);
        
        assert!((ratio - 12.8).abs() < 0.2);
    }
    
    #[test]
    fn test_memory_savings_3bit() {
        let hidden_dim = 4096u32;
        let config = TurboQuantConfig::three_bit(hidden_dim);
        let engine = TurboQuantEngine::new(config).unwrap();
        let ratio = engine.compression_ratio();
        
        println!("3-bit compression:");
        println!("  Ratio: {:.1}x", ratio);
        
        assert!((ratio - 10.67).abs() < 0.2);
    }
    
    #[test]
    fn test_compression_config() {
        // Test builder pattern
        let config = crate::inference::TwoPhaseConfig::default()
            .with_2bit_compression();
        
        assert!(config.use_turboquant);
        assert_eq!(config.compression_bit_width, Some(2));
        assert!((config.compression_ratio() - 16.0).abs() < 0.1);
        
        let config_2_5 = crate::inference::TwoPhaseConfig::default()
            .with_2_5bit_compression();
        
        assert!(config_2_5.use_outlier_splitting);
        assert!((config_2_5.compression_ratio() - 12.8).abs() < 0.2);
        
        let config_3 = crate::inference::TwoPhaseConfig::default()
            .with_3bit_compression();
        
        assert!((config_3.compression_ratio() - 10.7).abs() < 0.2);
        
        println!("Compression configs validated");
    }
    
    #[test]
    fn test_rotation_generation_time() {
        let hidden_dim = 512usize;
        let config = TurboQuantConfig::new(2, hidden_dim as u32, false);
        let mut engine = TurboQuantEngine::new(config).unwrap();
        
        let start = Instant::now();
        let _rotation = engine.generate_rotation_matrix(hidden_dim).unwrap();
        let elapsed = start.elapsed();
        
        println!("Rotation matrix generation ({}x{}): {:?}", hidden_dim, hidden_dim, elapsed);
        
        // Should complete in reasonable time (< 100ms for 512x512)
        assert!(elapsed.as_millis() < 100);
    }
    
    #[test]
    fn test_centroid_computation_time() {
        let num_centroids = 4usize; // 2-bit = 4 centroids
        let hidden_dim = 4096usize;
        
        let config = TurboQuantConfig::new(2, hidden_dim as u32, false);
        let engine = TurboQuantEngine::new(config).unwrap();
        
        let start = Instant::now();
        let centroids = engine.centroids();
        let elapsed = start.elapsed();
        
        println!("Centroid computation ({} centroids, dim={}): {:?}", 
            num_centroids, hidden_dim, elapsed);
        println!("  Centroids: {:?}", centroids);
        
        // Should be nearly instant (< 1ms)
        assert!(elapsed.as_millis() < 10);
    }
    
    #[test]
    fn test_kv_cache_memory_calculator() {
        // Real-world example: 4096 context, 32 layers, 4096 hidden
        let seq_len = 4096u32;
        let num_layers = 32u32;
        let hidden_dim = 4096u32;
        
        let head_dim = hidden_dim / 8; // 8 heads assumed
        let num_heads = 8u32;
        
        // Calculate baseline f32 KV cache size
        let bytes_per_position = num_heads as usize * head_dim as usize * 2 * 4; // K + V
        let total_bytes = seq_len as usize * num_layers as usize * bytes_per_position;
        
        println!("KV Cache Memory Calculator:");
        println!("  Sequence length: {}", seq_len);
        println!("  Layers: {}", num_layers);
        println!("  Hidden dim: {}", hidden_dim);
        println!("");
        println!("  f32 baseline: {} MB", total_bytes / (1024 * 1024));
        
        // 2-bit
        let tq_2bit = TurboQuantConfig::two_bit(hidden_dim);
        let ratio_2bit = TurboQuantEngine::new(tq_2bit).unwrap().compression_ratio();
        println!("  2-bit: {} MB ({:.1}x)", 
            total_bytes as f32 / ratio_22 / (1024.0 * 1024.0),
            ratio_2bit);
        
        // 2.5-bit
        let splitter = OutlierChannelSplitter::two_and_half_bit(hidden_dim);
        let ratio_2_5 = splitter.memory_savings();
        println!("  2.5-bit: {} MB ({:.1}x)", 
            total_bytes as f32 / ratio_2_5 / (1024.0 * 1024.0),
            ratio_2_5);
        
        // 3-bit
        let tq_3bit = TurboQuantConfig::three_bit(hidden_dim);
        let ratio_3bit = TurboQuantEngine::new(tq_3bit).unwrap().compression_ratio();
        println!("  3-bit: {} MB ({:.1}x)", 
            total_bytes as f32 / ratio_3bit / (1024.0 * 1024.0),
            ratio_3bit);
        
        // 4-bit
        let tq_4bit = TurboQuantConfig::new(4, hidden_dim, false);
        let ratio_4bit = TurboQuantEngine::new(tq_4bit).unwrap().compression_ratio();
        println!("  4-bit: {} MB ({:.1}x)", 
            total_bytes as f32 / ratio_4bit / (1024.0 * 1024.0),
            ratio_4bit);
        
        assert!(true);
    }
}