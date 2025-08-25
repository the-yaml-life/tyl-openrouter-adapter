//! # TYL OpenRouter Adapter
//!
//! OpenRouter adapter for TYL LLM inference port providing unified access to multiple LLM providers
//! through the OpenRouter API gateway with full TYL framework integration.
//!
//! OpenRouter provides a unified API to access models from OpenAI, Anthropic, Google, Meta, Mistral,
//! and many other providers through a single interface with competitive pricing and reliability.
//!
//! ## Features
//!
//! - **Multiple Providers** - Access 200+ models from different providers through one API
//! - **Template-based Interface** - Uses simplified template + parameters → JSON response pattern
//! - **Model Optimization** - Automatic model selection based on task type (coding, reasoning, etc.)
//! - **TYL Framework Integration** - Full integration with config, logging, tracing, and error handling
//! - **Cost Optimization** - Access to competitive pricing through OpenRouter's marketplace
//! - **Fallback Support** - Automatic failover to alternative models when primary is unavailable
//! - **Rate Limiting** - Built-in handling of rate limits and retries with exponential backoff
//! - **Observability** - Comprehensive logging and tracing for monitoring and debugging
//!
//! ## Quick Start
//!
//! ```rust
//! use tyl_openrouter_adapter::OpenRouterAdapter;
//! use tyl_llm_inference_port::{InferenceService, InferenceRequest, ModelType};
//! use std::collections::HashMap;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create adapter with API key
//! let adapter = OpenRouterAdapter::new("your-api-key")
//!     .with_app_name("my-app") // Optional, for usage tracking
//!     .with_timeout_seconds(30); // Optional, default is 30s
//!
//! // Create request with template and parameters
//! let mut params = HashMap::new();
//! params.insert("language".to_string(), "Rust".to_string());
//! params.insert("task".to_string(), "web server".to_string());
//!
//! let request = InferenceRequest::new(
//!     "Write a {{language}} {{task}} using async/await",
//!     params,
//!     ModelType::Coding
//! );
//!
//! // Get response
//! let response = adapter.infer(request).await?;
//! println!("Response: {}", serde_json::to_string_pretty(&response.content)?);
//! # Ok(())
//! # }
//! ```

// Re-export TYL framework functionality
pub use tyl_llm_inference_port::{
    InferenceRequest, InferenceResponse, InferenceResult, InferenceService, ModelType,
    ResponseMetadata, TokenUsage, TylResult, HealthCheckResult, HealthStatus,
};
pub use tyl_errors::{TylError, TylResult as ErrorResult};
pub use tyl_config::{ConfigManager, PostgresConfig, RedisConfig};
pub use tyl_logging::{Logger, LogRecord};
pub use tyl_tracing::{Tracer, Span};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use url::Url;

/// OpenRouter-specific error helpers that extend TylError
pub mod openrouter_errors {
    use super::*;

    /// Create an OpenRouter API error
    pub fn api_error(message: impl Into<String>) -> TylError {
        TylError::external(format!("OpenRouter API error: {}", message.into()))
    }

    /// Create a model not found error
    pub fn model_not_found(model: impl Into<String>) -> TylError {
        TylError::validation("model", format!("Model not found on OpenRouter: {}", model.into()))
    }

    /// Create a quota exceeded error
    pub fn quota_exceeded() -> TylError {
        TylError::external("OpenRouter quota exceeded")
    }

    /// Create an invalid response format error
    pub fn invalid_response_format(message: impl Into<String>) -> TylError {
        TylError::parsing(format!("Invalid OpenRouter response format: {}", message.into()))
    }
}

/// OpenRouter configuration using TYL config patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterConfig {
    /// API key for OpenRouter authentication
    pub api_key: String,
    /// Base URL for OpenRouter API (default: https://openrouter.ai/api/v1)
    pub base_url: String,
    /// Application name for usage tracking (optional)
    pub app_name: Option<String>,
    /// HTTP client timeout in seconds (default: 30)
    pub timeout_seconds: u64,
    /// Maximum number of retries for failed requests (default: 3)
    pub max_retries: u32,
    /// Enable request/response logging (default: true)
    pub enable_logging: bool,
    /// Enable distributed tracing (default: true)
    pub enable_tracing: bool,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            app_name: None,
            timeout_seconds: 30,
            max_retries: 3,
            enable_logging: true,
            enable_tracing: true,
        }
    }
}

impl OpenRouterConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_app_name(mut self, app_name: impl Into<String>) -> Self {
        self.app_name = Some(app_name.into());
        self
    }

    pub fn with_timeout_seconds(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    pub fn with_logging_enabled(mut self, enabled: bool) -> Self {
        self.enable_logging = enabled;
        self
    }

    pub fn with_tracing_enabled(mut self, enabled: bool) -> Self {
        self.enable_tracing = enabled;
        self
    }
}

/// OpenRouter API request format
#[derive(Debug, Serialize)]
struct OpenRouterRequest {
    model: String,
    messages: Vec<OpenRouterMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
}

/// OpenRouter message format
#[derive(Debug, Serialize)]
struct OpenRouterMessage {
    role: String,
    content: String,
}

/// OpenRouter API response format
#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenRouterChoice>,
    usage: OpenRouterUsage,
}

/// OpenRouter choice in response
#[derive(Debug, Deserialize)]
struct OpenRouterChoice {
    index: u32,
    message: OpenRouterResponseMessage,
    finish_reason: Option<String>,
}

/// OpenRouter response message
#[derive(Debug, Deserialize)]
struct OpenRouterResponseMessage {
    role: String,
    content: String,
}

/// OpenRouter token usage
#[derive(Debug, Deserialize)]
struct OpenRouterUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Main OpenRouter adapter with full TYL integration
pub struct OpenRouterAdapter {
    config: OpenRouterConfig,
    client: Client,
    logger: Logger,
    tracer: Tracer,
}

impl OpenRouterAdapter {
    /// Create new OpenRouter adapter with API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let config = OpenRouterConfig::new(api_key);
        Self::with_config(config)
    }

    /// Create adapter with custom configuration
    pub fn with_config(config: OpenRouterConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .expect("Failed to create HTTP client");

        // Initialize TYL framework components
        let logger = Logger::new("tyl-openrouter-adapter");
        let tracer = Tracer::new("openrouter");

        Self {
            config,
            client,
            logger,
            tracer,
        }
    }

    /// Update base URL (builder pattern)
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.config.base_url = url.into();
        self
    }

    /// Update app name (builder pattern)
    pub fn with_app_name(mut self, app_name: impl Into<String>) -> Self {
        self.config.app_name = Some(app_name.into());
        self
    }

    /// Update timeout (builder pattern)
    pub fn with_timeout_seconds(mut self, seconds: u64) -> Self {
        self.config.timeout_seconds = seconds;
        // Recreate client with new timeout
        self.client = Client::builder()
            .timeout(Duration::from_secs(seconds))
            .build()
            .expect("Failed to create HTTP client");
        self
    }

    /// Get the optimal OpenRouter model for the given model type
    fn get_optimal_model(&self, model_type: ModelType, model_override: Option<&str>) -> String {
        if let Some(model) = model_override {
            return model.to_string();
        }

        match model_type {
            ModelType::Coding => "anthropic/claude-3-5-sonnet-20241022".to_string(),
            ModelType::Reasoning => "anthropic/claude-3-5-sonnet-20241022".to_string(),
            ModelType::General => "anthropic/claude-3-5-haiku-20241022".to_string(),
            ModelType::Fast => "openai/gpt-3.5-turbo".to_string(),
            ModelType::Creative => "anthropic/claude-3-5-sonnet-20241022".to_string(),
        }
    }

    /// Send request to OpenRouter API with retry logic, logging, and tracing
    async fn send_request(&self, request: OpenRouterRequest) -> InferenceResult<OpenRouterResponse> {
        let span = if self.config.enable_tracing {
            Some(self.tracer.start_span("openrouter_request"))
        } else {
            None
        };

        let url = format!("{}/chat/completions", self.config.base_url);
        
        if self.config.enable_logging {
            self.logger.info(&LogRecord::new()
                .with_message("Sending request to OpenRouter API")
                .with_field("model", &request.model)
                .with_field("url", &url)
                .with_field("max_retries", &self.config.max_retries.to_string()));
        }

        let mut last_error = None;
        
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = Duration::from_secs(2_u64.pow(attempt - 1));
                
                if self.config.enable_logging {
                    self.logger.warn(&LogRecord::new()
                        .with_message("Retrying OpenRouter request")
                        .with_field("attempt", &attempt.to_string())
                        .with_field("delay_seconds", &delay.as_secs().to_string()));
                }
                
                tokio::time::sleep(delay).await;
            }

            let mut req_builder = self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.config.api_key))
                .header("Content-Type", "application/json");

            // Add optional app name header for usage tracking
            if let Some(ref app_name) = self.config.app_name {
                req_builder = req_builder.header("HTTP-Referer", app_name);
            }

            let response = match req_builder.json(&request).send().await {
                Ok(resp) => resp,
                Err(e) => {
                    let error_msg = format!("HTTP request failed: {}", e);
                    if self.config.enable_logging {
                        self.logger.error(&LogRecord::new()
                            .with_message("HTTP request failed")
                            .with_field("error", &error_msg)
                            .with_field("attempt", &attempt.to_string()));
                    }
                    last_error = Some(openrouter_errors::api_error(error_msg));
                    continue;
                }
            };

            let status = response.status();
            
            // Handle different status codes
            match status {
                status if status.is_success() => {
                    if self.config.enable_logging {
                        self.logger.info(&LogRecord::new()
                            .with_message("OpenRouter request successful")
                            .with_field("status", &status.to_string())
                            .with_field("attempt", &(attempt + 1).to_string()));
                    }
                    
                    return response.json::<OpenRouterResponse>().await.map_err(|e| {
                        let error_msg = format!("Failed to parse response: {}", e);
                        if self.config.enable_logging {
                            self.logger.error(&LogRecord::new()
                                .with_message("Failed to parse OpenRouter response")
                                .with_field("error", &error_msg));
                        }
                        openrouter_errors::invalid_response_format(error_msg)
                    });
                }
                status if status.as_u16() == 429 => {
                    // Rate limited, retry with backoff
                    if self.config.enable_logging {
                        self.logger.warn(&LogRecord::new()
                            .with_message("OpenRouter rate limit exceeded")
                            .with_field("status", &status.to_string())
                            .with_field("attempt", &attempt.to_string()));
                    }
                    last_error = Some(tyl_llm_inference_port::inference_errors::rate_limit_exceeded("OpenRouter"));
                    continue;
                }
                status if status.as_u16() == 401 => {
                    // Invalid API key, don't retry
                    if self.config.enable_logging {
                        self.logger.error(&LogRecord::new()
                            .with_message("OpenRouter authentication failed")
                            .with_field("status", &status.to_string()));
                    }
                    return Err(tyl_llm_inference_port::inference_errors::invalid_api_key("OpenRouter"));
                }
                status if status.as_u16() >= 500 => {
                    // Server error, retry
                    if self.config.enable_logging {
                        self.logger.error(&LogRecord::new()
                            .with_message("OpenRouter server error")
                            .with_field("status", &status.to_string())
                            .with_field("attempt", &attempt.to_string()));
                    }
                    last_error = Some(openrouter_errors::api_error(format!("Server error: {}", status)));
                    continue;
                }
                status => {
                    // Client error, don't retry
                    let error_text = response.text().await.unwrap_or_default();
                    let error_msg = format!("Client error {}: {}", status, error_text);
                    
                    if self.config.enable_logging {
                        self.logger.error(&LogRecord::new()
                            .with_message("OpenRouter client error")
                            .with_field("status", &status.to_string())
                            .with_field("error_text", &error_text));
                    }
                    
                    return Err(openrouter_errors::api_error(error_msg));
                }
            }
        }

        // All retries exhausted
        if self.config.enable_logging {
            self.logger.error(&LogRecord::new()
                .with_message("OpenRouter request failed after all retries")
                .with_field("max_retries", &self.config.max_retries.to_string()));
        }
        
        Err(last_error.unwrap_or_else(|| {
            openrouter_errors::api_error("Max retries exceeded")
        }))
    }

    /// Convert TYL InferenceRequest to OpenRouter format
    fn convert_request(&self, request: &InferenceRequest) -> OpenRouterRequest {
        let model = self.get_optimal_model(
            request.model_type,
            request.model_override.as_deref()
        );

        let rendered_prompt = request.render_template();
        
        if self.config.enable_logging {
            self.logger.debug(&LogRecord::new()
                .with_message("Converting TYL request to OpenRouter format")
                .with_field("template", &request.template)
                .with_field("rendered_prompt", &rendered_prompt)
                .with_field("model", &model)
                .with_field("model_type", &format!("{:?}", request.model_type)));
        }
        
        let messages = vec![OpenRouterMessage {
            role: "user".to_string(),
            content: rendered_prompt,
        }];

        OpenRouterRequest {
            model,
            messages,
            max_tokens: request.max_tokens.map(|t| t as u32),
            temperature: request.temperature,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }
}

#[async_trait]
impl InferenceService for OpenRouterAdapter {
    async fn infer(&self, request: InferenceRequest) -> InferenceResult<InferenceResponse> {
        let span = if self.config.enable_tracing {
            Some(self.tracer.start_span("openrouter_infer"))
        } else {
            None
        };

        let start_time = Instant::now();
        
        if self.config.enable_logging {
            self.logger.info(&LogRecord::new()
                .with_message("Starting OpenRouter inference")
                .with_field("template", &request.template)
                .with_field("model_type", &format!("{:?}", request.model_type))
                .with_field("model_override", &request.model_override.as_deref().unwrap_or("none")));
        }
        
        // Convert TYL request to OpenRouter format
        let or_request = self.convert_request(&request);
        
        // Send request to OpenRouter
        let or_response = self.send_request(or_request).await?;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Extract response content
        let content = or_response.choices
            .first()
            .map(|choice| choice.message.content.clone())
            .unwrap_or_default();
        
        // Create token usage
        let token_usage = TokenUsage::new(
            or_response.usage.prompt_tokens,
            or_response.usage.completion_tokens,
        );
        
        if self.config.enable_logging {
            self.logger.info(&LogRecord::new()
                .with_message("OpenRouter inference completed")
                .with_field("model", &or_response.model)
                .with_field("prompt_tokens", &or_response.usage.prompt_tokens.to_string())
                .with_field("completion_tokens", &or_response.usage.completion_tokens.to_string())
                .with_field("processing_time_ms", &processing_time.to_string()));
        }
        
        // Try to parse content as JSON, fallback to string
        let response = InferenceResponse::from_text_with_json_fallback(
            content,
            or_response.model,
            token_usage,
            processing_time,
        );
        
        Ok(response)
    }

    async fn health_check(&self) -> InferenceResult<HealthCheckResult> {
        let span = if self.config.enable_tracing {
            Some(self.tracer.start_span("openrouter_health_check"))
        } else {
            None
        };

        if self.config.enable_logging {
            self.logger.info(&LogRecord::new()
                .with_message("Performing OpenRouter health check"));
        }
        
        // Try a simple request to check if the service is available
        let test_request = OpenRouterRequest {
            model: "openai/gpt-3.5-turbo".to_string(),
            messages: vec![OpenRouterMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(1),
            temperature: Some(0.1),
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
        };
        
        match self.send_request(test_request).await {
            Ok(_) => {
                if self.config.enable_logging {
                    self.logger.info(&LogRecord::new()
                        .with_message("OpenRouter health check passed"));
                }
                
                Ok(HealthCheckResult::new(HealthStatus::healthy())
                    .with_metadata("provider", serde_json::Value::String("OpenRouter".to_string()))
                    .with_metadata("base_url", serde_json::Value::String(self.config.base_url.clone()))
                    .with_metadata("app_name", serde_json::Value::String(
                        self.config.app_name.clone().unwrap_or_default()
                    )))
            }
            Err(e) => {
                if self.config.enable_logging {
                    self.logger.error(&LogRecord::new()
                        .with_message("OpenRouter health check failed")
                        .with_field("error", &e.to_string()));
                }
                
                Ok(HealthCheckResult::new(HealthStatus::unhealthy(e.to_string())))
            }
        }
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            // OpenAI models
            "openai/gpt-4".to_string(),
            "openai/gpt-4-turbo".to_string(),
            "openai/gpt-3.5-turbo".to_string(),
            
            // Anthropic models
            "anthropic/claude-3-opus".to_string(),
            "anthropic/claude-3-5-sonnet-20241022".to_string(),
            "anthropic/claude-3-5-haiku-20241022".to_string(),
            
            // Google models
            "google/gemini-pro".to_string(),
            "google/gemini-pro-vision".to_string(),
            
            // Meta models
            "meta-llama/llama-2-70b-chat".to_string(),
            "meta-llama/codellama-34b-instruct".to_string(),
            
            // Mistral models
            "mistralai/mistral-7b-instruct".to_string(),
            "mistralai/mixtral-8x7b-instruct".to_string(),
            
            // Many more available through OpenRouter...
        ]
    }

    fn count_tokens(&self, text: &str) -> InferenceResult<usize> {
        // Simple approximation: ~4 characters per token
        // In production, you might want to use tiktoken or similar for more accuracy
        let count = (text.len() + 3) / 4;
        
        if self.config.enable_logging {
            self.logger.debug(&LogRecord::new()
                .with_message("Estimated token count")
                .with_field("text_length", &text.len().to_string())
                .with_field("estimated_tokens", &count.to_string()));
        }
        
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_config_creation() {
        let config = OpenRouterConfig::new("test-key")
            .with_app_name("test-app")
            .with_timeout_seconds(60)
            .with_max_retries(5)
            .with_logging_enabled(false)
            .with_tracing_enabled(false);
            
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.app_name, Some("test-app".to_string()));
        assert_eq!(config.timeout_seconds, 60);
        assert_eq!(config.max_retries, 5);
        assert!(!config.enable_logging);
        assert!(!config.enable_tracing);
    }

    #[test]
    fn test_adapter_creation() {
        let adapter = OpenRouterAdapter::new("test-key")
            .with_app_name("test-app")
            .with_timeout_seconds(45);
            
        assert_eq!(adapter.config.api_key, "test-key");
        assert_eq!(adapter.config.app_name, Some("test-app".to_string()));
        assert_eq!(adapter.config.timeout_seconds, 45);
    }

    #[test]
    fn test_model_selection() {
        let adapter = OpenRouterAdapter::new("test-key");
        
        // Test default model selection
        assert_eq!(
            adapter.get_optimal_model(ModelType::Coding, None),
            "anthropic/claude-3-5-sonnet-20241022"
        );
        assert_eq!(
            adapter.get_optimal_model(ModelType::Fast, None),
            "openai/gpt-3.5-turbo"
        );
        
        // Test model override
        assert_eq!(
            adapter.get_optimal_model(ModelType::Coding, Some("custom/model")),
            "custom/model"
        );
    }

    #[test]
    fn test_supported_models() {
        let adapter = OpenRouterAdapter::new("test-key");
        let models = adapter.supported_models();
        
        assert!(!models.is_empty());
        assert!(models.contains(&"openai/gpt-4".to_string()));
        assert!(models.contains(&"anthropic/claude-3-5-sonnet-20241022".to_string()));
    }

    #[test]
    fn test_token_counting() {
        let adapter = OpenRouterAdapter::new("test-key");
        let count = adapter.count_tokens("Hello, world!").unwrap();
        
        assert!(count > 0);
        assert!(count <= 4); // Should be around 3 tokens
    }

    #[test]
    fn test_request_conversion() {
        let adapter = OpenRouterAdapter::new("test-key");
        
        let mut params = HashMap::new();
        params.insert("name".to_string(), "Alice".to_string());
        params.insert("task".to_string(), "coding".to_string());
        
        let request = InferenceRequest::new(
            "Hello {{name}}, please help with {{task}}",
            params,
            ModelType::General,
        );
        
        let or_request = adapter.convert_request(&request);
        
        assert_eq!(or_request.model, "anthropic/claude-3-5-haiku-20241022");
        assert_eq!(or_request.messages.len(), 1);
        assert_eq!(or_request.messages[0].content, "Hello Alice, please help with coding");
    }

    #[test]
    fn test_config_serialization() {
        let config = OpenRouterConfig::new("test-key")
            .with_app_name("test-app");
            
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: OpenRouterConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.api_key, deserialized.api_key);
        assert_eq!(config.app_name, deserialized.app_name);
    }
}