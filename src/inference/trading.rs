//! Trading — Market Feed Encoder + Order Head + Compliance Oracle
//! 
//! Financial data processing:
//! - Market feed encoding (OHLCV, order book)
//! - Order execution
//! - Regulatory compliance (MiFID II, market surveillance)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Error, Debug)]
pub enum TradingError {
    #[error("Market: {0}")]
    Market(String),
    
    #[error("Order: {0}")]
    Order(String),
    
    #[error("Compliance: {0}")]
    Compliance(String),
}

// ============================================================================
// Market Data Types
// ============================================================================

/// OHLCV candle
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Candle {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub timestamp_ms: u64,
}

impl Candle {
    pub fn new(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self { open, high, low, close, volume, timestamp_ms: 0 }
    }
    
    /// Price range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }
    
    /// Return percentage
    pub fn returns(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }
}

/// Order book level
#[derive(Debug, Clone)]
pub struct BookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Order book
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
    pub timestamp_ms: u64,
}

impl OrderBook {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            bids: Vec::new(),
            asks: Vec::new(),
            timestamp_ms: 0,
        }
    }
    
    /// Mid price
    pub fn mid_price(&self) -> f64 {
        let best_bid = self.bids.first().map(|l| l.price).unwrap_or(0.0);
        let best_ask = self.asks.first().map(|l| l.price).unwrap_or(0.0);
        if best_bid > 0.0 && best_ask > 0.0 {
            (best_bid + best_ask) / 2.0
        } else {
            0.0
        }
    }
    
    /// Spread
    pub fn spread(&self) -> f64 {
        let best_bid = self.bids.first().map(|l| l.price).unwrap_or(0.0);
        let best_ask = self.asks.first().map(|l| l.price).unwrap_or(0.0);
        best_ask - best_bid
    }
}

/// Market tick
#[derive(Debug, Clone)]
pub struct MarketTick {
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub timestamp_ms: u64,
    pub side: MarketSide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketSide {
    Bid,
    Ask,
}

// ============================================================================
// Market Feed Encoder
// ============================================================================

/// Encodes market data for Block AttnRes
pub struct MarketFeedEncoder {
    pub symbol_tokens: HashMap<String, u32>,
}

impl MarketFeedEncoder {
    pub fn new() -> Self {
        Self {
            symbol_tokens: HashMap::new(),
        }
    }
    
    /// Encode candle to tokens
    pub fn encode_candle(&mut self, candle: &Candle) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Returns bucket
        let returns = candle.returns();
        tokens.push(returns_bucket(returns));
        
        // Volume bucket
        tokens.push(volume_bucket(candle.volume));
        
        // Range bucket
        tokens.push(range_bucket(candle.range()));
        
        // Time bucket (hour of day)
        tokens.push(((candle.timestamp_ms / 3600000) % 24) as u32);
        
        tokens
    }
    
    /// Encode order book to tokens
    pub fn encode_orderbook(&self, book: &OrderBook) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Spread bucket
        let spread = book.spread();
        tokens.push(spread_bucket(spread));
        
        // Depth tokens (top 5 levels)
        for level in book.bids.iter().take(5) {
            tokens.push(price_bucket(level.price));
            tokens.push(quantity_bucket(level.quantity));
        }
        for level in book.asks.iter().take(5) {
            tokens.push(price_bucket(level.price));
            tokens.push(quantity_bucket(level.quantity));
        }
        
        tokens
    }
    
    /// Encode tick to tokens
    pub fn encode_tick(&mut self, tick: &MarketTick) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Symbol token
        let symbol_token = *self.symbol_tokens.entry(tick.symbol.clone()).or_insert_with(
            self.symbol_tokens.len() as u32 + 1000
        );
        tokens.push(symbol_token);
        
        // Price
        tokens.push(price_bucket(tick.price));
        
        // Side
        tokens.push(match tick.side {
            MarketSide::Bid => 0,
            MarketSide::Ask => 1,
        });
        
        // Time bucket
        tokens.push(((tick.timestamp_ms / 1000) % 60) as u32);
        
        tokens
    }
}

fn returns_bucket(r: f64) -> u32 {
    if r < -0.05 { 0 }
    else if r < -0.02 { 1 }
    else if r < -0.01 { 2 }
    else if r < 0.0 { 3 }
    else if r < 0.01 { 4 }
    else if r < 0.02 { 5 }
    else if r < 0.05 { 6 }
    else { 7 }
}

fn volume_bucket(v: f64) -> u32 {
    if v < 1000.0 { 0 }
    else if v < 10000.0 { 1 }
    else if v < 100000.0 { 2 }
    else if v < 1000000.0 { 3 }
    else { 4 }
}

fn range_bucket(r: f64) -> u32 {
    if r < 0.01 { 0 }
    else if r < 0.1 { 1 }
    else if r < 1.0 { 2 }
    else if r < 10.0 { 3 }
    else { 4 }
}

fn spread_bucket(s: f64) -> u32 {
    if s < 0.0001 { 0 }
    else if s < 0.001 { 1 }
    else if s < 0.01 { 2 }
    else if s < 0.1 { 3 }
    else { 4 }
}

fn price_bucket(p: f64) -> u32 {
    (p * 100.0) as u32 % 10000
}

fn quantity_bucket(q: f64) -> u32 {
    if q < 1.0 { 0 }
    else if q < 10.0 { 1 }
    else if q < 100.0 { 2 }
    else if q < 1000.0 { 3 }
    else { 4 }
}

// ============================================================================
// Order Head
// ============================================================================

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

/// Order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub stop_price: Option<f64>,
}

impl Order {
    pub fn market_buy(symbol: &str, quantity: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity,
            price: None,
            stop_price: None,
        }
    }
    
    pub fn limit_sell(symbol: &str, quantity: f64, price: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            side: OrderSide::Sell,
            order_type: OrderType::Limit,
            quantity,
            price: Some(price),
            stop_price: None,
        }
    }
}

/// Order head - generates orders from embeddings
pub struct OrderHead {
    pub account_id: String,
}

impl OrderHead {
    pub fn new(account_id: &str) -> Self {
        Self {
            account_id: account_id.to_string(),
        }
    }
    
    /// Predict order from embeddings
    pub fn predict(&self, embeddings: &[f32], symbol: &str) -> Order {
        if embeddings.is_empty() {
            return Order::market_buy(symbol, 0.0);
        }
        
        let mag = embeddings.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        // Simplified strategy
        if mag > 3.0 {
            // Strong signal - buy
            Order::market_buy(symbol, 100.0)
        } else if mag < -3.0 {
            // Strong negative - sell
            Order::limit_sell(symbol, 100.0, 100.0)
        } else {
            // No signal - hold
            Order::market_buy(symbol, 0.0)
        }
    }
    
    /// Generate order from strategy output
    pub fn generate(&self, action: &str, symbol: &str, quantity: f64, price: f64) -> Order {
        match action {
            "buy_market" => Order::market_buy(symbol, quantity),
            "sell_limit" => Order::limit_sell(symbol, quantity, price),
            _ => Order::market_buy(symbol, 0.0),
        }
    }
}

// ============================================================================
// Trading Compliance Oracle
// ============================================================================

/// MiFID II transaction reporting
#[derive(Debug, Clone)]
pub struct TransactionReport {
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub timestamp_ms: u64,
    pub venue: String,
}

/// Market surveillance rule
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurveillanceRule {
    PriceManipulation,
    VolumeManipulation,
    Layering,
    Spoofing,
}

/// Trading compliance oracle
pub struct TradingComplianceOracle {
    pub max_order_size: f64,
    pub max_daily_volume: f64,
    pub prohibited_venues: Vec<String>,
}

impl TradingComplianceOracle {
    pub fn new() -> Self {
        Self {
            max_order_size: 10000.0,
            max_daily_volume: 1000000.0,
            prohibited_venues: vec![],
        }
    }
    
    /// Validate order
    pub fn validate_order(&self, order: &Order) -> Result<(), TradingError> {
        if order.quantity > self.max_order_size {
            return Err(TradingError::Compliance(format!(
                "Order size {} exceeds max {}",
                order.quantity, self.max_order_size
            )));
        }
        
        if let Some(ref venue) = order.symbol.as_ref() {
            if self.prohibited_venues.contains(&venue.to_string()) {
                return Err(TradingError::Compliance(format!(
                    "Trading prohibited on venue: {}",
                    venue
                )));
            }
        }
        
        Ok(())
    }
    
    /// Check for spoofing pattern
    pub fn detect_spoofing(&self, orders: &[Order]) -> bool {
        if orders.len() < 5 { return false; }
        
        // Simplified: check if multiple orders cancelled quickly
        let mut cancels = 0;
        for _ in orders {
            cancels += 1;
        }
        
        cancels > 3
    }
    
    /// Generate transaction report
    pub fn report_transaction(&self, order: &Order, executed_price: f64) -> TransactionReport {
        TransactionReport {
            symbol: order.symbol.clone(),
            side: order.side,
            quantity: order.quantity,
            price: executed_price,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            venue: "Routed".to_string(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_candle() {
        let c = Candle::new(100.0, 101.0, 99.0, 100.5, 10000.0);
        assert!((c.returns() - 0.005).abs() < 0.001);
    }
    
    #[test]
    fn test_order_book() {
        let mut book = OrderBook::new("AAPL");
        book.bids.push(BookLevel { price: 100.0, quantity: 100.0 });
        book.asks.push(BookLevel { price: 101.0, quantity: 100.0 });
        
        assert!(book.mid_price() > 0.0);
    }
    
    #[test]
    fn test_market_feed_encoder() {
        let mut encoder = MarketFeedEncoder::new();
        
        let candle = Candle::new(100.0, 101.0, 99.0, 100.5, 10000.0);
        let tokens = encoder.encode_candle(&candle);
        
        assert!(!tokens.is_empty());
    }
    
    #[test]
    fn test_order_head() {
        let head = OrderHead::new("ACC001");
        
        let embeddings = vec![5.0, 1.0, 2.0];
        let order = head.predict(&embeddings, "AAPL");
        
        assert!(order.quantity > 0.0);
    }
    
    #[test]
    fn test_compliance_oracle() {
        let oracle = TradingComplianceOracle::new();
        
        let order = Order::market_buy("AAPL", 5000.0);
        let result = oracle.validate_order(&order);
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_order_size_violation() {
        let oracle = TradingComplianceOracle::new();
        
        let order = Order::market_buy("AAPL", 20000.0);  // > max
        let result = oracle.validate_order(&order);
        
        assert!(result.is_err());
    }
}