# TYL Module Template

🏗️ **Template repository for creating new TYL framework modules**

This template provides a complete foundation for building hexagonal architecture modules in the TYL framework ecosystem.

## 🚀 Quick Start

### 1. Use This Template
Click "Use this template" button on GitHub or:
```bash
gh repo create the-yaml-life/tyl-your-module --template the-yaml-life/tyl-module-template --public
```

### 2. Replace Placeholders
Search and replace the following placeholders throughout the codebase:

- `{module-name}` → your module name (e.g., `cache`, `config`)
- `{module_name}` → snake_case version (e.g., `cache`, `config`) 
- `{MainTrait}` → your main trait name (e.g., `CacheManager`, `ConfigLoader`)
- `{MainType}` → your main type name (e.g., `CacheConfig`, `ConfigSettings`)
- `{BasicAdapter}` → your adapter name (e.g., `MemoryCache`, `EnvConfig`)
- `{Module}` → PascalCase module name (e.g., `Cache`, `Config`)
- `{Module Name}` → Human readable name (e.g., `Cache Management`, `Configuration`)

### 3. Update Package Metadata
Edit `Cargo.toml`:
- Update `name`, `description`, `keywords`, `categories`
- Set correct repository URLs
- Adjust dependencies for your module

### 4. Implement Your Module
- Define your port (trait) in `src/lib.rs`
- Implement adapters
- Add comprehensive tests
- Update documentation

## 📁 What's Included

### ✅ **Complete Structure**
```
tyl-module-template/
├── src/lib.rs                 # Core implementation with hexagonal architecture
├── examples/basic_usage.rs    # Usage examples
├── tests/integration_tests.rs # Integration tests
├── .github/workflows/         # CI/CD pipelines
│   ├── ci.yml                # Tests, clippy, fmt, security audit
│   └── release.yml           # Automated releases
├── Cargo.toml                # Package configuration
├── README.md                 # Documentation template
├── CLAUDE.md                 # Claude Code context
├── CHANGELOG.md              # Version history
├── LICENSE                   # AGPL-3.0 license
├── .gitignore               # Git ignore rules
├── rustfmt.toml             # Code formatting config
└── clippy.toml              # Linting configuration
```

### ✅ **Built-in Features**
- 🏛️ **Hexagonal Architecture** - Clean ports and adapters pattern
- 🧪 **TDD Ready** - Test structure following TDD principles
- 📖 **Documentation** - Complete docs with examples
- ⚙️ **CI/CD** - GitHub Actions for testing and releases
- 🔒 **Quality Gates** - Clippy, rustfmt, security audit
- 📦 **Serialization** - Serde support built-in
- 🎯 **Error Handling** - Comprehensive error types with thiserror
- 🔧 **Branch Protection** - Configured for team development

## 🏗️ Architecture

Follows hexagonal architecture principles:

```rust
// Port (Interface)
trait YourTrait {
    fn operation(&self, input: &str) -> YourResult<String>;
}

// Adapter (Implementation)  
struct YourAdapter {
    config: YourConfig,
}

impl YourTrait for YourAdapter {
    fn operation(&self, input: &str) -> YourResult<String> {
        // Implementation
    }
}
```

## 🧪 Testing Strategy

### **TDD Approach**
1. Write failing tests first
2. Implement minimal code to pass
3. Refactor and improve

### **Test Coverage**
- Unit tests in `src/lib.rs`
- Integration tests in `tests/`
- Doc tests in documentation
- Example tests via CI

## 📝 Checklist After Using Template

- [ ] Replace all placeholder text
- [ ] Update `Cargo.toml` metadata
- [ ] Implement your trait and types
- [ ] Add comprehensive tests
- [ ] Update documentation
- [ ] Configure repository settings
- [ ] Set up branch protection
- [ ] Add GitHub topics

## 🎯 TYL Framework Standards

This template ensures your module follows TYL framework standards:

- ✅ Hexagonal architecture
- ✅ Extensible design without core modifications  
- ✅ Comprehensive error handling
- ✅ Full test coverage
- ✅ Complete documentation
- ✅ CI/CD automation
- ✅ Security best practices

## 🔗 Related

- [TYL Framework Documentation](https://github.com/the-yaml-life)
- [`tyl-errors`](https://github.com/the-yaml-life/tyl-errors) - Error handling
- [`tyl-logging`](https://github.com/the-yaml-life/tyl-logging) - Structured logging

## 📄 License

AGPL-3.0 - See [LICENSE](LICENSE) for details.