use tyl_{module_name}::{BasicAdapter, MainTrait, MainType, {Module}Error};

#[test]
fn test_end_to_end_operation() {
    let config = MainType::new("integration-test");
    let adapter = BasicAdapter::new(config);
    
    let result = adapter.operation("test input");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Processed: test input");
}

#[test]
fn test_configuration_integration() {
    let config = MainType::new("test-service");
    let adapter = BasicAdapter::new(config);
    
    // Test that configuration affects behavior
    let result = adapter.operation("config test");
    assert!(result.is_ok());
}

#[test]
fn test_error_handling_integration() {
    let adapter = BasicAdapter::default();
    
    // Test successful operation
    let result = adapter.operation("valid");
    assert!(result.is_ok());
    
    // Add tests for error cases when you implement them
    // Example:
    // let error_result = adapter.operation("invalid");
    // assert!(error_result.is_err());
}

#[test]
fn test_serialization_integration() {
    let config = MainType::new("serialization-test");
    
    // Test JSON serialization
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: MainType = serde_json::from_str(&json).unwrap();
    
    assert_eq!(config.name, deserialized.name);
}

#[test]
fn test_trait_object_usage() {
    let adapter = BasicAdapter::default();
    
    // Test using trait object
    let trait_obj: Box<dyn MainTrait> = Box::new(adapter);
    let result = trait_obj.operation("trait object test");
    
    assert!(result.is_ok());
}