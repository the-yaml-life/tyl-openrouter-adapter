# CLAUDE.md - tyl-{module-name}

## 📋 **Module Context**

**tyl-{module-name}** is the {replace with module function} module for the TYL framework.

## 🏗️ **Architecture**

### **Port (Interface)**
```rust
trait {MainTrait} {
    fn operation(&self, input: &str) -> {Module}Result<String>;
}
```

### **Adapters (Implementations)**
- `{BasicAdapter}` - Simple implementation for basic use cases
- Add more adapters as needed

### **Core Types**
- `{MainType}` - Main configuration/data type
- `{Module}Error` - Error types with thiserror
- `{Module}Result<T>` - Result type alias

## 🧪 **Testing**

```bash
cargo test -p tyl-{module-name}
cargo test --doc -p tyl-{module-name}
cargo run --example basic_usage -p tyl-{module-name}
```

## 📂 **File Structure**

```
tyl-{module-name}/
├── src/lib.rs                 # Core implementation
├── examples/
│   └── basic_usage.rs         # Basic usage example
├── tests/
│   └── integration_tests.rs   # Integration tests
├── README.md                  # Main documentation
├── CLAUDE.md                  # This file
└── Cargo.toml                 # Package metadata
```

## 🔧 **How to Use**

### **Basic Usage**
```rust
use tyl_{module_name}::{MainTrait, BasicAdapter, MainType};

let config = MainType::new("my-config");
let adapter = BasicAdapter::new(config);
let result = adapter.operation("input").unwrap();
```

### **Custom Implementation**
```rust
struct MyCustomAdapter {
    // Custom fields
}

impl {MainTrait} for MyCustomAdapter {
    fn operation(&self, input: &str) -> {Module}Result<String> {
        // Custom implementation
        Ok(format!("Custom: {}", input))
    }
}
```

## 🛠️ **Useful Commands**

```bash
cargo clippy -p tyl-{module-name}
cargo fmt -p tyl-{module-name}  
cargo doc --no-deps -p tyl-{module-name} --open
cargo test -p tyl-{module-name} --verbose
```

## 📦 **Dependencies**

### **Runtime**
- `serde` - Serialization support
- `serde_json` - JSON handling
- `thiserror` - Error handling
- `uuid` - Unique identifier generation

### **Development**
- Standard Rust testing framework

## 🎯 **Design Principles**

1. **Hexagonal Architecture** - Clean separation of concerns
2. **Trait-based Extensibility** - Easy to add new implementations
3. **Error Handling** - Comprehensive error types with context
4. **Serialization** - First-class serde support
5. **Testing** - Comprehensive test coverage

## ⚠️ **Known Limitations**

- {Add any current limitations}
- {Add any planned improvements}

## 📝 **Notes for Contributors**

- Follow TDD approach
- Maintain hexagonal architecture
- Document all public APIs with examples
- Add integration tests for new features
- Keep dependencies minimal

## 🔗 **Related TYL Modules**

- [`tyl-errors`](https://github.com/the-yaml-life/tyl-errors) - Error handling
- [`tyl-logging`](https://github.com/the-yaml-life/tyl-logging) - Structured logging