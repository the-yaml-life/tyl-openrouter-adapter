//! # TYL {Module Name}
//!
//! {Module description - replace with actual description}
//!
//! ## Features
//!
//! - {Feature 1}
//! - {Feature 2}
//! - Hexagonal architecture with ports and adapters
//! - Comprehensive error handling
//! - Full test coverage
//!
//! ## Quick Start
//!
//! ```rust
//! use tyl_{module_name}::{MainTrait, MainType};
//!
//! // Basic usage example
//! let instance = MainType::new();
//! // Add usage example here
//! ```
//!
//! ## Architecture
//!
//! This module follows hexagonal architecture:
//!
//! - **Port (Interface)**: `{MainTrait}` - defines the contract
//! - **Adapters**: Various implementations of the port
//! - **Domain Logic**: Core business logic independent of infrastructure
//!
//! ## Examples
//!
//! See the `examples/` directory for complete usage examples.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Result type for {module} operations
pub type {Module}Result<T> = Result<T, {Module}Error>;

/// Errors that can occur during {module} operations
#[derive(Debug, Error)]
pub enum {Module}Error {
    #[error("Configuration error: {message}")]
    Configuration { message: String },
    
    #[error("Operation failed: {message}")]
    Operation { message: String },
    
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
}

/// Port (Interface) - Replace with your actual port trait
pub trait {MainTrait} {
    /// Main operation - replace with actual methods
    fn operation(&self, input: &str) -> {Module}Result<String>;
}

/// Main type for {module} operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct {MainType} {
    // Add fields as needed
    pub name: String,
}

impl {MainType} {
    /// Create a new instance
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
        }
    }
}

impl Default for {MainType} {
    fn default() -> Self {
        Self::new("default")
    }
}

/// Adapter - Basic implementation
pub struct {BasicAdapter} {
    inner: {MainType},
}

impl {BasicAdapter} {
    pub fn new(config: {MainType}) -> Self {
        Self { inner: config }
    }
}

impl Default for {BasicAdapter} {
    fn default() -> Self {
        Self::new({MainType}::default())
    }
}

impl {MainTrait} for {BasicAdapter} {
    fn operation(&self, input: &str) -> {Module}Result<String> {
        // Replace with actual implementation
        Ok(format!("Processed: {}", input))
    }
}

// Utility functions
pub fn generate_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_{module_name}_basic_functionality() {
        // TDD: Start with failing tests, then implement
        let adapter = {BasicAdapter}::default();
        let result = adapter.operation("test input");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Processed: test input");
    }

    #[test]
    fn test_{main_type}_creation() {
        let instance = {MainType}::new("test");
        assert_eq!(instance.name, "test");
    }

    #[test]
    fn test_default_implementation() {
        let instance = {MainType}::default();
        assert_eq!(instance.name, "default");
    }

    #[test]
    fn test_trait_implementation() {
        let adapter = {BasicAdapter}::default();
        
        // Test the trait contract
        let result = adapter.operation("hello");
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_handling() {
        // Test error cases - add as needed
        let error = {Module}Error::InvalidInput {
            message: "test error".to_string(),
        };
        
        assert!(error.to_string().contains("Invalid input"));
    }

    #[test]
    fn test_serialization() {
        let instance = {MainType}::new("test");
        let json = serde_json::to_string(&instance).unwrap();
        let deserialized: {MainType} = serde_json::from_str(&json).unwrap();
        assert_eq!(instance.name, deserialized.name);
    }
}