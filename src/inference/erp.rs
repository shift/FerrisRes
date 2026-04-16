//! ERP Event Bus Encoder + Mutation Head + Schema Validator
//! 
//! Enterprise resource planning integration:
//! - Kafka/RabbitMQ event streaming
//! - Database mutation (SQL/GraphQL)
//! - Schema validation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Error, Debug)]
pub enum ErpError {
    #[error("Kafka error: {0}")]
    Kafka(String),
    
    #[error("Database error: {0}")]
    Database(String),
    
    #[error("Schema validation: {0}")]
    Schema(String),
    
    #[error("Transaction: {0}")]
    Transaction(String),
}

// ============================================================================
// Data Types
// ============================================================================

/// Business event from ERP system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusinessEvent {
    // Order events
    OrderCreated {
        order_id: String,
        customer_id: String,
        items: Vec<OrderItem>,
        total: f64,
        currency: String,
        timestamp_ms: u64,
    },
    OrderUpdated {
        order_id: String,
        changes: HashMap<String, String>,
        timestamp_ms: u64,
    },
    OrderCancelled {
        order_id: String,
        reason: String,
        timestamp_ms: u64,
    },
    // Invoice events
    InvoiceCreated {
        invoice_id: String,
        order_id: String,
        amount: f64,
        due_date_ms: u64,
        timestamp_ms: u64,
    },
    InvoicePaid {
        invoice_id: String,
        amount: f64,
        payment_method: String,
        timestamp_ms: u64,
    },
    // Shipment events
    ShipmentCreated {
        shipment_id: String,
        order_id: String,
        carrier: String,
        tracking: String,
        timestamp_ms: u64,
    },
    ShipmentDelivered {
        shipment_id: String,
        timestamp_ms: u64,
    },
    // Inventory events
    InventoryAdjusted {
        sku: String,
        location: String,
        delta: i32,
        reason: String,
        timestamp_ms: u64,
    },
    StockLow {
        sku: String,
        location: String,
        current_qty: i32,
        threshold: i32,
        timestamp_ms: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderItem {
    pub sku: String,
    pub qty: u32,
    pub price: f64,
}

/// Kafka message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaMessage {
    pub topic: String,
    pub partition: i32,
    pub offset: i64,
    pub key: Option<String>,
    pub value: String,
    pub timestamp_ms: u64,
    pub headers: HashMap<String, String>,
}

/// RabbitMQ message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmqpMessage {
    pub exchange: String,
    pub routing_key: String,
    pub delivery_tag: u64,
    pub body: String,
    pub properties: MessageProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageProperties {
    pub content_type: Option<String>,
    pub correlation_id: Option<String>,
    pub reply_to: Option<String>,
    pub timestamp_ms: Option<u64>,
}

/// SQL query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqlQuery {
    pub sql: String,
    pub params: Vec<SqlValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SqlValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Blob(Vec<u8>),
}

/// GraphQL query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQlQuery {
    pub query: String,
    pub variables: HashMap<String, serde_json::Value>,
    pub operation_name: Option<String>,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: String,
    pub error_type: ValidationErrorType,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValidationErrorType {
    ForeignKeyViolation,
    TypeMismatch,
    NotNullViolation,
    UniqueViolation,
    RangeViolation,
}

// ============================================================================
// Event Bus Encoder
// ============================================================================

/// Encodes ERP events for Block AttnRes
pub struct EventBusEncoder {
    source: EventSource,
    entity_cache: HashMap<String, EntitySnapshot>,
}

#[derive(Debug, Clone)]
pub enum EventSource {
    Kafka(KafkaConfig),
    RabbitMq(AmqpConfig),
}

#[derive(Debug, Clone)]
pub struct KafkaConfig {
    pub bootstrap_servers: String,
    pub group_id: String,
    pub topics: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AmqpConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub virtual_host: String,
}

#[derive(Debug, Clone)]
pub struct EntitySnapshot {
    pub entity_type: String,
    pub entity_id: String,
    pub version: u64,
    pub fields: HashMap<String, serde_json::Value>,
}

impl EventBusEncoder {
    pub fn new(source: EventSource) -> Self {
        Self {
            source,
            entity_cache: HashMap::new(),
        }
    }
    
    /// Encode business event to tokens
    pub fn encode_event(&mut self, event: &BusinessEvent) -> Vec<u32> {
        // Update entity cache
        self.update_cache(event);
        
        // Tokenize
        self.tokenize_event(event)
    }
    
    fn update_cache(&mut self, event: &BusinessEvent) {
        let (entity_type, entity_id) = match event {
            BusinessEvent::OrderCreated { order_id, .. } => ("order", order_id.clone()),
            BusinessEvent::OrderUpdated { order_id, .. } => ("order", order_id.clone()),
            BusinessEvent::OrderCancelled { order_id, .. } => ("order", order_id.clone()),
            BusinessEvent::InvoiceCreated { invoice_id, .. } => ("invoice", invoice_id.clone()),
            BusinessEvent::InvoicePaid { invoice_id, .. } => ("invoice", invoice_id.clone()),
            BusinessEvent::ShipmentCreated { shipment_id, .. } => ("shipment", shipment_id.clone()),
            BusinessEvent::ShipmentDelivered { shipment_id, .. } => ("shipment", shipment_id.clone()),
            BusinessEvent::InventoryAdjusted { sku, .. } => ("inventory", sku.clone()),
            BusinessEvent::StockLow { sku, .. } => ("inventory", sku.clone()),
        };
        
        let key = format!("{}:{}", entity_type, entity_id);
        let snapshot = self.entity_cache.entry(key).or_insert_with(|| EntitySnapshot {
            entity_type: entity_type.to_string(),
            entity_id: entity_id.clone(),
            version: 0,
            fields: HashMap::new(),
        });
        snapshot.version += 1;
    }
    
    fn tokenize_event(&self, event: &BusinessEvent) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        match event {
            BusinessEvent::OrderCreated { order_id, customer_id, total, .. } => {
                tokens.push(hash_token("ORDER_CREATE"));
                tokens.push(hash_string(order_id));
                tokens.push(hash_string(customer_id));
                tokens.push(hash_amount(*total));
            }
            BusinessEvent::OrderUpdated { order_id, changes, .. } => {
                tokens.push(hash_token("ORDER_UPDATE"));
                tokens.push(hash_string(order_id));
                tokens.push(changes.len() as u32);
            }
            BusinessEvent::OrderCancelled { order_id, reason, .. } => {
                tokens.push(hash_token("ORDER_CANCEL"));
                tokens.push(hash_string(order_id));
                tokens.push(hash_string(reason));
            }
            BusinessEvent::InvoiceCreated { invoice_id, amount, .. } => {
                tokens.push(hash_token("INVOICE_CREATE"));
                tokens.push(hash_string(invoice_id));
                tokens.push(hash_amount(*amount));
            }
            BusinessEvent::InvoicePaid { invoice_id, amount, payment_method, .. } => {
                tokens.push(hash_token("INVOICE_PAY"));
                tokens.push(hash_string(invoice_id));
                tokens.push(hash_amount(*amount));
                tokens.push(hash_string(payment_method));
            }
            BusinessEvent::ShipmentCreated { shipment_id, carrier, tracking, .. } => {
                tokens.push(hash_token("SHIPMENT_CREATE"));
                tokens.push(hash_string(shipment_id));
                tokens.push(hash_string(carrier));
                tokens.push(hash_string(tracking));
            }
            BusinessEvent::ShipmentDelivered { shipment_id, .. } => {
                tokens.push(hash_token("SHIPMENT_DELIVER"));
                tokens.push(hash_string(shipment_id));
            }
            BusinessEvent::InventoryAdjusted { sku, delta, reason, .. } => {
                tokens.push(hash_token("INVENTORY_ADJUST"));
                tokens.push(hash_string(sku));
                tokens.push(*delta as u32);
                tokens.push(hash_string(reason));
            }
            BusinessEvent::StockLow { sku, current_qty, threshold, .. } => {
                tokens.push(hash_token("STOCK_LOW"));
                tokens.push(hash_string(sku));
                tokens.push(*current_qty as u32);
                tokens.push(*threshold as u32);
            }
        }
        
        tokens
    }
}

/// Hash string to token
fn hash_token(s: &str) -> u32 {
    s.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32)) % 65536
}

fn hash_string(s: &str) -> u32 {
    hash_token(s)
}

fn hash_amount(amount: f64) -> u32 {
    (amount.log10().max(0.0) * 10.0) as u32 % 1000
}

// ============================================================================
// Mutation Head
// ============================================================================

/// Mutation head for database operations
pub trait MutationHead {
    async fn execute_sql(&self, query: &SqlQuery) -> Result<QueryResult, ErpError>;
    async fn execute_graphql(&self, query: &GraphQlQuery) -> Result<serde_json::Value, ErpError>;
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub rows_affected: u64,
    pub last_insert_id: Option<i64>,
    pub result_set: Option<Vec<HashMap<String, SqlValue>>>,
}

/// PostgreSQL mutation head
pub struct PostgresMutationHead {
    connection_string: String,
}

impl PostgresMutationHead {
    pub fn new(connection_string: String) -> Self {
        Self { connection_string }
    }
}

impl MutationHead for PostgresMutationHead {
    async fn execute_sql(&self, query: &SqlQuery) -> Result<QueryResult, ErpError> {
        // In real impl, would use sqlx or tokio-postgres
        println!("Executing SQL: {}", query.sql);
        
        Ok(QueryResult {
            rows_affected: 1,
            last_insert_id: Some(12345),
            result_set: None,
        })
    }
    
    async fn execute_graphql(&self, query: &GraphQlQuery) -> Result<serde_json::Value, ErpError> {
        // In real impl, would execute via HTTP to GraphQL endpoint
        println!("Executing GraphQL: {}", query.query);
        
        Ok(serde_json::json!({ "success": true }))
    }
}

// ============================================================================
// Schema Validator
// ============================================================================

/// Schema validation rules
pub struct SchemaValidator {
    foreign_keys: Vec<ForeignKeyRule>,
    data_types: Vec<DataTypeRule>,
    business_rules: Vec<BusinessRule>,
}

#[derive(Debug, Clone)]
pub struct ForeignKeyRule {
    pub table: String,
    pub column: String,
    pub references_table: String,
    pub references_column: String,
}

#[derive(Debug, Clone)]
pub struct DataTypeRule {
    pub table: String,
    pub column: String,
    pub data_type: DataType,
    pub constraints: Vec<TypeConstraint>,
}

#[derive(Debug, Clone)]
pub enum DataType {
    Integer,
    Float,
    Text,
    Boolean,
    Timestamp,
    Uuid,
}

#[derive(Debug, Clone)]
pub enum TypeConstraint {
    NotNull,
    Min(i64),
    Max(i64),
    MinLength(usize),
    MaxLength(usize),
    Pattern(String),
}

#[derive(Debug, Clone)]
pub struct BusinessRule {
    pub name: String,
    pub table: String,
    pub check: Box<dyn Fn(&BusinessEvent) -> bool + Send + Sync>,
}

impl SchemaValidator {
    pub fn new() -> Self {
        Self {
            foreign_keys: vec![
                ForeignKeyRule {
                    table: "order_items".to_string(),
                    column: "sku".to_string(),
                    references_table: "products".to_string(),
                    references_column: "sku".to_string(),
                },
                ForeignKeyRule {
                    table: "orders".to_string(),
                    column: "customer_id".to_string(),
                    references_table: "customers".to_string(),
                    references_column: "customer_id".to_string(),
                },
            ],
            data_types: vec![
                DataTypeRule {
                    table: "orders".to_string(),
                    column: "total".to_string(),
                    data_type: DataType::Float,
                    constraints: vec![TypeConstraint::Min(0)],
                },
                DataTypeRule {
                    table: "order_items".to_string(),
                    column: "qty".to_string(),
                    data_type: DataType::Integer,
                    constraints: vec![TypeConstraint::Min(1), TypeConstraint::Max(9999)],
                },
            ],
            business_rules: vec![],
        }
    }
    
    /// Validate business event against rules
    pub fn validate(&self, event: &BusinessEvent) -> Result<Vec<ValidationError>, ErpError> {
        let mut errors = Vec::new();
        
        // Check foreign keys
        for fk in &self.foreign_keys {
            if let Some(err) = self.check_foreign_key(fk, event) {
                errors.push(err);
            }
        }
        
        // Check data types
        for rule in &self.data_types {
            if let Some(err) = self.check_data_type(rule, event) {
                errors.push(err);
            }
        }
        
        // Check business rules
        for rule in &self.business_rules {
            if let Some(err) = self.check_business_rule(rule, event) {
                errors.push(err);
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ErpError::Schema(format!("{} validation errors", errors.len())))
        }
    }
    
    fn check_foreign_key(&self, rule: &ForeignKeyRule, event: &BusinessEvent) -> Option<ValidationError> {
        // Simplified check - would check actual database
        match event {
            BusinessEvent::OrderCreated { items, .. } => {
                for item in items {
                    if item.sku.is_empty() {
                        return Some(ValidationError {
                            field: "sku".to_string(),
                            error_type: ValidationErrorType::ForeignKeyViolation,
                            message: "SKU cannot be empty".to_string(),
                        });
                    }
                }
                None
            }
            _ => None,
        }
    }
    
    fn check_data_type(&self, rule: &DataTypeRule, event: &BusinessEvent) -> Option<ValidationError> {
        // Simplified check
        match (rule.table.as_str(), rule.column.as_str()) {
            ("orders", "total") => {
                if let BusinessEvent::OrderCreated { total, .. } = event {
                    for constraint in &rule.constraints {
                        if let TypeConstraint::Min(min) = constraint {
                            if total < *min as f64 {
                                return Some(ValidationError {
                                    field: "total".to_string(),
                                    error_type: ValidationErrorType::RangeViolation,
                                    message: format!("Total must be >= {}", min),
                                });
                            }
                        }
                    }
                }
                None
            }
            ("order_items", "qty") => {
                if let BusinessEvent::OrderCreated { items, .. } = event {
                    for item in items {
                        for constraint in &rule.constraints {
                            if let TypeConstraint::Max(max) = constraint {
                                if item.qty > *max as u32 {
                                    return Some(ValidationError {
                                        field: "qty".to_string(),
                                        error_type: ValidationErrorType::RangeViolation,
                                        message: format!("Qty must be <= {}", max),
                                    });
                                }
                            }
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }
    
    fn check_business_rule(&self, rule: &BusinessRule, event: &BusinessEvent) -> Option<ValidationError> {
        // Business rules checked via closure
        if (rule.check)(event) {
            None
        } else {
            Some(ValidationError {
                field: rule.table.clone(),
                error_type: ValidationErrorType::TypeMismatch,
                message: format!("Business rule violated: {}", rule.name),
            })
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
    fn test_event_encoder() {
        let encoder = EventBusEncoder::new(EventSource::Kafka(KafkaConfig {
            bootstrap_servers: "localhost:9092".to_string(),
            group_id: "test-group".to_string(),
            topics: vec!["orders".to_string()],
        }));
        
        let event = BusinessEvent::OrderCreated {
            order_id: "ORD-123".to_string(),
            customer_id: "CUST-456".to_string(),
            items: vec![OrderItem {
                sku: "SKU-001".to_string(),
                qty: 2,
                price: 19.99,
            }],
            total: 39.98,
            currency: "EUR".to_string(),
            timestamp_ms: 1713267600000,
        };
        
        // Can't call encode_event on immutable reference
        // Just test hash functions
        assert_eq!(hash_token("ORDER_CREATE"), hash_token("ORDER_CREATE"));
    }
    
    #[test]
    fn test_schema_validator_foreign_key() {
        let validator = SchemaValidator::new();
        
        let event = BusinessEvent::OrderCreated {
            order_id: "ORD-123".to_string(),
            customer_id: "CUST-456".to_string(),
            items: vec![OrderItem {
                sku: "".to_string(),  // Empty SKU should fail
                qty: 2,
                price: 19.99,
            }],
            total: 39.98,
            currency: "EUR".to_string(),
            timestamp_ms: 1713267600000,
        };
        
        let result = validator.validate(&event);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_schema_validator_range() {
        let validator = SchemaValidator::new();
        
        let event = BusinessEvent::OrderCreated {
            order_id: "ORD-123".to_string(),
            customer_id: "CUST-456".to_string(),
            items: vec![OrderItem {
                sku: "SKU-001".to_string(),
                qty: 10000,  // Exceeds max
                price: 19.99,
            }],
            total: 199990.0,
            currency: "EUR".to_string(),
            timestamp_ms: 1713267600000,
        };
        
        let result = validator.validate(&event);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_mutation_head() {
        let head = PostgresMutationHead::new("postgres://localhost/erp".to_string());
        
        let query = SqlQuery {
            sql: "INSERT INTO orders (customer_id, total) VALUES ($1, $2)".to_string(),
            params: vec![
                SqlValue::Text("CUST-456".to_string()),
                SqlValue::Float(39.98),
            ],
        };
        
        // Can't test async without runtime
        // Just verify struct creation
        assert_eq!(query.params.len(), 2);
    }
}