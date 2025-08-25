use tyl_{module_name}::{BasicAdapter, MainTrait, MainType};

fn main() {
    println!("=== TYL {Module Name} Basic Usage ===\n");

    // Basic usage example
    basic_usage_example();
    
    // Custom configuration example
    custom_config_example();
    
    // Error handling example
    error_handling_example();
}

fn basic_usage_example() {
    println!("--- Basic Usage ---");
    
    let config = MainType::new("my-service");
    let adapter = BasicAdapter::new(config);
    
    match adapter.operation("test input") {
        Ok(result) => println!("✅ Success: {}", result),
        Err(e) => println!("❌ Error: {}", e),
    }
    
    println!();
}

fn custom_config_example() {
    println!("--- Custom Configuration ---");
    
    let config = MainType::new("custom-service");
    let adapter = BasicAdapter::new(config);
    
    // Example with different inputs
    let inputs = vec!["hello", "world", "rust"];
    
    for input in inputs {
        match adapter.operation(input) {
            Ok(result) => println!("  {} -> {}", input, result),
            Err(e) => println!("  {} -> Error: {}", input, e),
        }
    }
    
    println!();
}

fn error_handling_example() {
    println!("--- Error Handling ---");
    
    let adapter = BasicAdapter::default();
    
    // This should work
    match adapter.operation("valid input") {
        Ok(result) => println!("✅ Valid input processed: {}", result),
        Err(e) => println!("❌ Unexpected error: {}", e),
    }
    
    // Example of error handling (implement error cases in your module)
    println!("💡 Add error cases to demonstrate error handling in your module");
    
    println!();
}