# ERP Event Bus Research

## Overview
Enterprise resource planning integration via:
- Apache Kafka / RabbitMQ event streaming
- MutationHead for database transactions
- Schema validation for data integrity

## Kafka Consumer Protocol

### Consumer Groups
```rust
pub struct KafkaConsumer {
    group_id: String,
    topics: Vec<String>,
    offset_strategy: OffsetStrategy,
}

pub enum OffsetStrategy {
    Earliest,  // Start from beginning
    Latest,    // Only new messages
    Timestamp(i64),  // Resume from time
}
```

### Message Structure
```json
{
  "topic": "orders.created",
  "partition": 0,
  "offset": 12345,
  "timestamp": 1713267600000,
  "key": "order-12345",
  "value": {
    "order_id": "12345",
    "customer_id": "cust-789",
    "items": [{"sku": "PROD-001", "qty": 2}],
    "total": 99.99,
    "currency": "EUR"
  }
}
```

### Rebalancing
```rust
pub trait ConsumerRebalanceHandler {
    fn on_assign(&mut self, topic: &str, partitions: &[i32]);
    fn on_revoke(&mut self, topic: &str, partitions: &[i32]);
    fn on_lost(&mut self, topic: &str, partitions: &[i32]);
}
```

## RabbitMQ AMQP

### Connection Model
```rust
pub struct AmqpConnection {
    host: String,
    port: u16,
    virtual_host: String,
    username: String,
}

pub struct AmqpConsumer {
    queue: String,
    exchange: String,
    routing_key: String,
}
```

### Message Flow
```
Publisher → Exchange → Routing Key → Queue → Consumer
              ↓
         [fanout|topic|direct]
```

## EventBusEncoder Design

### Tokenization Strategy
```
[entity_type] [event_type] [entity_id] [field_changes...] [timestamp]
```

### Example Events
| Event | Token Sequence |
|-------|----------------|
| Order created | ORDER CREATE ord-123 cust-789 2026-04-16T10:00:00Z |
| Invoice paid | INVOICE PAY inv-456 99.99 EUR |
| Shipment shipped | SHIPMENT SEND ship-789 FedEx TRACK123 |

```rust
pub struct EventBusEncoder {
    source: EventSource,  // Kafka or RabbitMQ
    entity_extractor: EntityExtractor,
}

impl StreamEncoder for EventBusEncoder {
    fn encode(&self, event: &BusinessEvent) -> Vec<u32> {
        let mut tokens = vec![];
        
        // Entity type + event
        tokens.push(self.tokenize(event.entity_type()));
        tokens.push(self.tokenize(event.event_type()));
        
        // Entity ID
        tokens.push(self.tokenize(event.entity_id()));
        
        // Field changes
        for (field, value) in event.changes() {
            tokens.push(self.tokenize(field));
            tokens.push(self.tokenize(value));
        }
        
        // Timestamp
        tokens.push(self.tokenize(event.timestamp()));
        
        tokens
    }
}
```

## MutationHead

### Transaction Types
```rust
pub trait MutationHead {
    async fn execute_sql(&self, query: &SqlQuery) -> Result<QueryResult>;
    async fn execute_abap(&self, module: &AbapModule) -> Result<AbapResult>;
    async fn execute_graphql(&self, query: &GraphQlQuery) -> Result<JsonResult>;
}
```

### SQL Transaction
```rust
pub async fn create_order(conn: &Pool<Postgres>, order: &Order) -> Result<OrderId> {
    let tx = conn.begin().await?;
    
    let order_id: OrderId = tx
        .query_one(
            "INSERT INTO orders (customer_id, total) VALUES ($1, $2) RETURNING id",
            [&order.customer_id, &order.total],
        )
        .await?;
    
    for item in &order.items {
        tx.execute(
            "INSERT INTO order_items (order_id, sku, qty, price) VALUES ($1, $2, $3, $4)",
            [&order_id, &item.sku, &(item.qty as i32), &item.price],
        ).await?;
    }
    
    tx.commit().await?;
    Ok(order_id)
}
```

### SAP RFC/BAPI Integration
```rust
pub struct SapConnector {
    host: String,
    client: u8,
    system_id: String,
}

impl MutationHead for SapConnector {
    async fn execute_abap(&self, module: &AbapModule) -> Result<AbapResult> {
        // RFC_CALL_FUNCTION module="BAPI_SALESORDER_CREATEFROMDAT2"
        // Convert JSON → ABAP structures → BAPI call → Result
        Ok(AbapResult::Success)
    }
}
```

## Schema Validator

### Validation Rules
```rust
pub struct SchemaValidator {
    foreign_keys: HashMap<String, ForeignKeyRule>,
    data_types: HashMap<String, DataTypeRule>,
    business_rules: Vec<BusinessRule>,
}

impl SchemaValidator {
    pub fn validate(&self, event: &BusinessEvent) -> ValidationResult {
        let mut errors = Vec::new();
        
        // Foreign key check
        for fk in &self.foreign_keys {
            if !self.check_foreign_key(fk, event) {
                errors.push(ValidationError::ForeignKeyViolation(fk.clone()));
            }
        }
        
        // Data type check
        for (field, rule) in &self.data_types {
            if let Some(value) = event.get(field) {
                if !rule.validate(value) {
                    errors.push(ValidationError::TypeMismatch(field.clone()));
                }
            }
        }
        
        // Business rules
        for rule in &self.business_rules {
            if let Err(e) = rule.check(event) {
                errors.push(e);
            }
        }
        
        ValidationResult { errors }
    }
}
```

### Example Rules
```yaml
# SKU must exist in products table
foreign_key:
  table: products
  column: sku
  references: order_items.sku

# Quantity must be positive integer
data_type:
  field: quantity
  type: integer
  min: 1
  max: 9999

# Order total must match sum of items
business_rule:
  name: total_matches_items
  expression: order.total == sum(order_items.price * order_items.qty)
```

## Implementation Notes

### Kafka Best Practices
- Use semantic partitioning (by entity_id for ordering)
- Enable exactly-once semantics (idempotent producer)
- Monitor consumer lag (should be <1 minute)

### RabbitMQ Best Practices
- Dead letter queues for failed messages
- Publisher confirms for reliability
- Lazy queues for large messages

## References
- Kafka: https://kafka.apache.org/documentation/
- RabbitMQ: https://www.rabbitmq.com/documentation.html
- SAP BAPI: https://help.sap.com/viewer/fin_globs/2.0/en-US/