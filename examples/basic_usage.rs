//! Basic usage example for TYL OpenRouter Adapter
//!
//! This example demonstrates how to use the OpenRouter adapter to access multiple LLM providers
//! through a unified interface with the TYL framework integration.

use std::collections::HashMap;
use std::env;
use tyl_openrouter_adapter::{OpenRouterAdapter, OpenRouterConfig};
use tyl_llm_inference_port::{InferenceService, InferenceRequest, ModelType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 TYL OpenRouter Adapter - Basic Usage Example");
    println!("===============================================\n");

    // Initialize environment logger for TYL logging
    env_logger::init();

    // Get API key from environment variable
    let api_key = match env::var("OPENROUTER_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            println!("⚠️  OPENROUTER_API_KEY environment variable not set.");
            println!("   Set it with: export OPENROUTER_API_KEY='your-api-key'");
            println!("   Get your key from: https://openrouter.ai/keys");
            println!("\n📋 This example will demonstrate the adapter structure without making real API calls.\n");
            
            // Use a dummy key for demonstration
            "demo-key-for-structure-example".to_string()
        }
    };

    // Example 1: Basic adapter creation and configuration
    println!("🔧 Example 1: Adapter Configuration");
    println!("----------------------------------");

    let adapter = OpenRouterAdapter::new(&api_key)
        .with_app_name("tyl-openrouter-example")
        .with_timeout_seconds(60)
        .with_base_url("https://openrouter.ai/api/v1");

    println!("✅ OpenRouter adapter created successfully");
    println!("   App name: tyl-openrouter-example");
    println!("   Timeout: 60 seconds");
    println!("   Base URL: https://openrouter.ai/api/v1\n");

    // Example 2: Show supported models
    println!("📋 Example 2: Supported Models");
    println!("-----------------------------");

    let models = adapter.supported_models();
    println!("Available models ({}):", models.len());
    
    // Group models by provider
    let mut openai_models = Vec::new();
    let mut anthropic_models = Vec::new();
    let mut google_models = Vec::new();
    let mut meta_models = Vec::new();
    let mut mistral_models = Vec::new();
    
    for model in &models {
        if model.starts_with("openai/") {
            openai_models.push(model);
        } else if model.starts_with("anthropic/") {
            anthropic_models.push(model);
        } else if model.starts_with("google/") {
            google_models.push(model);
        } else if model.starts_with("meta-llama/") {
            meta_models.push(model);
        } else if model.starts_with("mistralai/") {
            mistral_models.push(model);
        }
    }

    if !openai_models.is_empty() {
        println!("  🤖 OpenAI: {}", openai_models.join(", "));
    }
    if !anthropic_models.is_empty() {
        println!("  🧠 Anthropic: {}", anthropic_models.join(", "));
    }
    if !google_models.is_empty() {
        println!("  🔍 Google: {}", google_models.join(", "));
    }
    if !meta_models.is_empty() {
        println!("  🦙 Meta: {}", meta_models.join(", "));
    }
    if !mistral_models.is_empty() {
        println!("  🌟 Mistral: {}", mistral_models.join(", "));
    }
    
    println!();

    // Example 3: Model type optimization
    println!("🎯 Example 3: Model Type Optimization");
    println!("------------------------------------");

    let model_types = [
        (ModelType::Coding, "Code generation and programming"),
        (ModelType::Reasoning, "Complex analysis and reasoning"),
        (ModelType::General, "General conversation and tasks"),
        (ModelType::Fast, "Quick responses and simple tasks"),
        (ModelType::Creative, "Creative writing and content"),
    ];

    for (model_type, description) in &model_types {
        println!("  {:?}: {} tokens max", model_type, model_type.typical_max_tokens());
        println!("    Description: {}", description);
        println!("    OpenAI optimal: {}", model_type.optimal_openai_model());
        println!("    Anthropic optimal: {}", model_type.optimal_anthropic_model());
    }
    
    println!();

    // Example 4: Template-based requests (structure demo)
    println!("📝 Example 4: Template-Based Requests");
    println!("------------------------------------");

    let examples = vec![
        (
            ModelType::Coding,
            "Write a {{language}} {{component}} for {{purpose}}",
            vec![
                ("language", "Rust"),
                ("component", "HTTP client"),
                ("purpose", "making API requests"),
            ],
        ),
        (
            ModelType::Reasoning,
            "Analyze the pros and cons of {{topic}} in {{context}}",
            vec![
                ("topic", "microservices architecture"),
                ("context", "a startup environment"),
            ],
        ),
        (
            ModelType::Creative,
            "Write a {{genre}} story about {{character}} who {{action}}",
            vec![
                ("genre", "sci-fi"),
                ("character", "a robot chef"),
                ("action", "discovers emotions through cooking"),
            ],
        ),
        (
            ModelType::Fast,
            "Quick summary: {{content}}",
            vec![("content", "the benefits of the TYL framework")],
        ),
    ];

    for (model_type, template, params_vec) in examples {
        println!("  🎯 Model Type: {:?}", model_type);
        println!("  📄 Template: \"{}\"", template);
        
        let mut params = HashMap::new();
        for (key, value) in params_vec {
            params.insert(key.to_string(), value.to_string());
        }
        
        let request = InferenceRequest::new(template, params, model_type);
        let rendered = request.render_template();
        
        println!("  ✨ Rendered: \"{}\"", rendered);
        println!("  🔧 Would use model: {}", model_type.optimal_anthropic_model());
        println!();
    }

    // Example 5: Configuration patterns
    println!("⚙️ Example 5: Configuration Patterns");
    println!("-----------------------------------");

    // Configuration from builder pattern
    let config1 = OpenRouterConfig::new("api-key")
        .with_app_name("my-app")
        .with_timeout_seconds(45)
        .with_max_retries(5)
        .with_logging_enabled(true)
        .with_tracing_enabled(true);

    println!("  ✅ Builder pattern configuration:");
    println!("    Timeout: {} seconds", config1.timeout_seconds);
    println!("    Max retries: {}", config1.max_retries);
    println!("    Logging enabled: {}", config1.enable_logging);
    println!("    Tracing enabled: {}", config1.enable_tracing);

    // Configuration serialization (useful for config files)
    let config_json = serde_json::to_string_pretty(&config1)?;
    println!("\n  📋 JSON serialization:");
    println!("{}", config_json);

    // Example 6: Error handling patterns
    println!("\n🚨 Example 6: Error Handling Patterns");
    println!("------------------------------------");

    // Demonstrate error types
    use tyl_openrouter_adapter::openrouter_errors;
    
    let error_examples = vec![
        openrouter_errors::api_error("Connection timeout"),
        openrouter_errors::model_not_found("non-existent/model"),
        openrouter_errors::quota_exceeded(),
        openrouter_errors::invalid_response_format("Malformed JSON"),
    ];

    for (i, error) in error_examples.iter().enumerate() {
        println!("  {}. {}", i + 1, error);
    }

    println!("\n✅ All examples completed successfully!");
    
    if api_key == "demo-key-for-structure-example" {
        println!("\n💡 To test with real API calls:");
        println!("   1. Get an API key from https://openrouter.ai/keys");
        println!("   2. Set OPENROUTER_API_KEY environment variable");
        println!("   3. Run this example again");
        println!("\n🔗 OpenRouter provides access to 200+ models from multiple providers!");
    } else {
        println!("\n🚀 Ready to make real API calls!");
        println!("💡 Try running with different model types and templates!");
    }

    Ok(())
}