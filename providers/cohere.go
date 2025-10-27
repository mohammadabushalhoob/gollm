package providers

import (
"context"
"encoding/json"
"fmt"
"strings"

cohere "github.com/cohere-ai/cohere-go/v2"
coherev2 "github.com/cohere-ai/cohere-go/v2/v2"
chopt "github.com/cohere-ai/cohere-go/v2/option"

"github.com/teilomillet/gollm/config"
"github.com/teilomillet/gollm/types"
"github.com/teilomillet/gollm/utils"
)

// CohereProvider implements the Provider interface for Cohere's API.
// It supports Cohere's language models and provides access to their capabilities,
// including chat completion and structured output
type CohereProvider struct {
apiKey       string            // API key for authentication
	baseURL      string            // Base URL for API endpoint
model        string            // Model identifier (e.g., "command-r-plus-08-2024", "command-r-plus-04-2024")
extraHeaders map[string]string // Additional HTTP headers
options      map[string]any    // Model-specific options
logger       utils.Logger      // Logger instance
	sdk          *coherev2.Client   // Cohere v2 SDK client
}

// NewCohereProvider creates a new Cohere provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: Cohere API key for authentication
//   - model: The model to use (e.g., "command-r-plus-08-2024", "command-r-plus-04-2024")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Cohere Provider instance
//   - A configured Cohere Provider instance
func NewCohereProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	return NewCohereProviderWithURL(apiKey, model, "https://api.cohere.com", extraHeaders)
}

// NewCohereProviderWithURL creates a new Cohere provider with a custom base URL
func NewCohereProviderWithURL(apiKey, model, baseURL string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}

	opts := []chopt.RequestOption{chopt.WithToken(apiKey)}
	if baseURL != "" {
		opts = append(opts, chopt.WithBaseURL(baseURL))
	}
	sdk := coherev2.NewClient(opts...)
	return &CohereProvider{
		apiKey:       apiKey,
		baseURL:      baseURL,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]any),
		logger:       utils.NewLogger(utils.LogLevelInfo),
		sdk:          sdk,
	}
}

// SetLogger configures the logger for the Cohere provider.
// This is used for debugging and monitoring API interactions.
func (p *CohereProvider) SetLogger(logger utils.Logger) {
p.logger = logger
}

// SetOption sets a specific option for the Cohere provider.
// Support options include:
//   - temperature: Controls randomness
//   - max_tokens: Maximum tokens in the response
//   - p: Total probability mass (0.01 to 0.99)
//   - k: Top k most likely tokens are considered
//   - strict_tools: If set to true, follow tool definition strictly
func (p *CohereProvider) SetOption(key string, value any) {
p.options[key] = value
if p.logger != nil {
p.logger.Debug("Setting option for Cohere", "key", key, "value", value)
}
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *CohereProvider) SetDefaultOptions(config *config.Config) {
p.SetOption("temperature", config.Temperature)
p.SetOption("max_tokens", config.MaxTokens)
p.SetOption("stream", false)
if config.Seed != nil {
p.SetOption("seed", *config.Seed)
}
}

// Name returns "cohere" as the provider identifier.
func (p *CohereProvider) Name() string {
return "cohere"
}

// Endpoint returns the base URL for the Cohere API.
// Uses the configured baseURL and appends the appropriate API version path.
func (p *CohereProvider) Endpoint() string {
	baseURL := p.baseURL
	
	// Remove trailing slash if present
	if len(baseURL) > 0 && baseURL[len(baseURL)-1] == '/' {
		baseURL = baseURL[:len(baseURL)-1]
	}
	
	return baseURL + "/v2/chat"
}

// SupportsJSONSchema indicates that Cohere supports structured output
// through its system prompts and response formatting capabilities.
func (p *CohereProvider) SupportsJSONSchema() bool {
return true
}

// Headers returns the required HTTP headers for Cohere API requests.
// This includes:
//   - Content-type: application/json
//   - Authorization: Bearer token using the API key
//   - Any additional headers specified via SetExtraHeaders
func (p *CohereProvider) Headers() map[string]string {
headers := map[string]string{
"Content-Type":  "application/json",
"Authorization": "bearer " + p.apiKey,
}

for k, v := range p.extraHeaders {
headers[k] = v
}
return headers
}

// PrepareRequest creates the request body for a Cohere API call.
// It handles:
//   - Message formatting
//   - System prompts
//   - Response formatting
//   - Model-specific options
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional parameters for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *CohereProvider) PrepareRequest(prompt string, options map[string]any) ([]byte, error) {
requestBody := map[string]any{
"model": p.model,
"messages": []map[string]any{
{
"role":    "user",
"content": prompt, // Cohere v2 API accepts string content
},
},
}

// Cohere v2 API supported parameters
supportedParams := map[string]bool{
"temperature":        true,
"max_tokens":         true,
"seed":               true,
"frequency_penalty":  true,
"presence_penalty":   true,
"k":                  true,
"p":                  true,
}

// First, add default options (filter unsupported)
for k, v := range p.options {
if supportedParams[k] {
requestBody[k] = v
}
}

// Then, add any additional options (which may override defaults, filter unsupported)
for k, v := range options {
if supportedParams[k] {
requestBody[k] = v
}
}

return json.Marshal(requestBody)
}

// PrepareRequestWithSchema creates a request that includes structured output formatting.
// This uses Cohere's system prompts to enforce response structure.
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional request parameters
//   - schema: JSON schema for response validation
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *CohereProvider) PrepareRequestWithSchema(prompt string, options map[string]any, schema any) ([]byte, error) {
requestBody := map[string]any{
"model": p.model,
"messages": []map[string]any{
{"role": "user", "content": prompt},
},
"response_format": map[string]any{
"type":        "json_object",
"json_schema": schema,
},
}

// First, add the default options
for k, v := range p.options {
requestBody[k] = v
}

// Then, add any additional options (which may override defaults)
for k, v := range options {
requestBody[k] = v
}

return json.Marshal(requestBody)
}

// ParseResponse extracts the generated text from the Cohere API response.
// It handles various response formats and error cases
//
// Parameters:
//   - body: Raw API response body
//
// Returns:
//   - Generated text content
//   - Any error encountered during parsing
func (p *CohereProvider) ParseResponse(body []byte) (string, error) {
var response struct {
Message struct {
Role    string `json:"role"`
Content []struct {
Type string `json:"type"`
Text string `json:"text"`
} `json:"content"`
ToolCalls []struct {
ID       string `json:"id"`
Type     string `json:"type"`
Function struct {
Name      string `json:"name"`
Arguments string `json:"arguments"`
} `json:"function"`
} `json:"tool_calls"`
} `json:"message"`
}

if err := json.Unmarshal(body, &response); err != nil {
return "", fmt.Errorf("error parsing response: %w", err)
}

if len(response.Message.Content) == 0 {
return "", fmt.Errorf("empty response from API")
}

var finalResponse strings.Builder

for _, content := range response.Message.Content {
switch content.Type {
case "text":
finalResponse.WriteString(content.Text)
p.logger.Debug("Text content: %s", content.Text)
}
}

for _, toolCall := range response.Message.ToolCalls {
// Parse arguments as raw JSON to preserve the exact format
var args interface{}
if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
return "", fmt.Errorf("error parsing function arguments: %w", err)
}

functionCall, err := utils.FormatFunctionCall(toolCall.Function.Name, args)
if err != nil {
return "", fmt.Errorf("error formatting function call: %w", err)
}
if finalResponse.Len() > 0 {
finalResponse.WriteString("\n")
}
finalResponse.WriteString(functionCall)
}

p.logger.Debug("Final response: %s", finalResponse.String())
return finalResponse.String(), nil
}

// HandleFunctionCalls processes structured output in the response.
// This supports Cohere's response formatting capabilities.
func (p *CohereProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
response := string(body)
functionCalls, err := utils.ExtractFunctionCalls(response)
if err != nil {
return nil, fmt.Errorf("error extracting function calls: %w", err)
}

if len(functionCalls) == 0 {
return nil, nil // No function calls found
}

return json.Marshal(functionCalls)
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *CohereProvider) SetExtraHeaders(extraHeaders map[string]string) {
p.extraHeaders = extraHeaders
p.logger.Debug("Extra headers set", "headers", extraHeaders)
}

// SupportsStreaming returns whether the provider supports streaming responses
func (p *CohereProvider) SupportsStreaming() bool {
return true
}

// PrepareStreamRequest prepares a request body for streaming
func (p *CohereProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
options["stream"] = true
return p.PrepareRequest(prompt, options)
}

// ParseStreamResponse parses a single chunk from a streaming response
func (p *CohereProvider) ParseStreamResponse(chunk []byte) (string, error) {
var response struct {
Text string `json:"text"`
}
if err := json.Unmarshal(chunk, &response); err != nil {
return "", err
}
return response.Text, nil
}

// PrepareRequestWithMessages creates a request using structured message objects.
// This method uses Cohere v2 API format with messages array for conversation history.
func (p *CohereProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
// Convert messages to Cohere v2 format
cohereMessages := []map[string]interface{}{}

for _, msg := range messages {
cohereMessages = append(cohereMessages, map[string]interface{}{
"role":    msg.Role,
"content": msg.Content, // Cohere v2 API accepts string content
})
}

// Build request using v2 API format
request := map[string]interface{}{
"model":    p.model,
"messages": cohereMessages,
}

// Add other options (filter unsupported parameters)
// Cohere v2 API supported parameters: temperature, max_tokens, seed, frequency_penalty, presence_penalty, k, p
supportedParams := map[string]bool{
"temperature":        true,
"max_tokens":         true,
"seed":               true,
"frequency_penalty":  true,
"presence_penalty":   true,
"k":                  true,
"p":                  true,
}

for k, v := range p.options {
if k != "messages" {
if supportedParams[k] {
request[k] = v
} else {
p.logger.Debug("Filtering unsupported parameter from p.options: %s", k)
}
}
}
for k, v := range options {
if k != "messages" && k != "system_prompt" && k != "structured_messages" {
if supportedParams[k] {
request[k] = v
} else {
p.logger.Debug("Filtering unsupported parameter from options: %s", k)
}
}
}

// Add system prompt if present (as first message with role "system")
if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
// Insert system message at the beginning
systemMessage := map[string]interface{}{
"role":    "system",
"content": systemPrompt, // Cohere v2 API accepts string content
}
request["messages"] = append([]map[string]interface{}{systemMessage}, cohereMessages...)
}

if p.logger != nil {
p.logger.Debug("Using Cohere v2 messages format", 
"message_count", len(cohereMessages))
}

// Debug: Log the request body
	requestJSON, _ := json.MarshalIndent(request, "", "  ")
	p.logger.Info("Cohere API Request Body: %s", string(requestJSON))
	p.logger.Info("Cohere API Endpoint: %s", p.Endpoint())
	p.logger.Info("Cohere API Headers: %v", p.Headers())
	
	return json.Marshal(request)
}


// GenerateNative implements providers.NativeChatProvider using Cohere Go SDK
func (p *CohereProvider) GenerateNative(ctx context.Context, prompt string, options map[string]interface{}, structuredMessages []types.MemoryMessage) (string, error) {
    if p.sdk == nil {
        // Initialize lazily if needed
        opts := []chopt.RequestOption{chopt.WithToken(p.apiKey)}
        if p.baseURL != "" { opts = append(opts, chopt.WithBaseURL(p.baseURL)) }
        p.sdk = coherev2.NewClient(opts...)
    }

    // Build messages
    var msgs []*cohere.ChatMessageV2

    // Optional system prompt
    if sp, ok := options["system_prompt"].(string); ok && sp != "" {
        msgs = append(msgs, &cohere.ChatMessageV2{System: &cohere.SystemMessageV2{Content: &cohere.SystemMessageV2Content{String: sp}}})
    }

    // Structured history
    for _, m := range structuredMessages {
        switch strings.ToLower(m.Role) {
        case "user":
            msgs = append(msgs, &cohere.ChatMessageV2{User: &cohere.UserMessageV2{Content: &cohere.UserMessageV2Content{String: m.Content}}})
        case "assistant":
            msgs = append(msgs, &cohere.ChatMessageV2{Assistant: &cohere.AssistantMessage{Content: &cohere.AssistantMessageV2Content{String: m.Content}}})
        case "system":
            // If system already added at head, keep chronological order and include here too
            msgs = append(msgs, &cohere.ChatMessageV2{System: &cohere.SystemMessageV2{Content: &cohere.SystemMessageV2Content{String: m.Content}}})
        }
    }

    // If no user message present and prompt provided, add it
    if prompt != "" {
        msgs = append(msgs, &cohere.ChatMessageV2{User: &cohere.UserMessageV2{Content: &cohere.UserMessageV2Content{String: prompt}}})
    }

    // Map options
    req := &cohere.V2ChatRequest{ Model: p.model, Messages: msgs }

    // helper to set float64 pointer
    setF := func(dst **float64, v interface{}) {
        switch t := v.(type) {
        case float64:
            vv := t; *dst = &vv
        case float32:
            vv := float64(t); *dst = &vv
        case int:
            vv := float64(t); *dst = &vv
        }
    }
    _ = setF // avoid unused when no float options passed

    if v, ok := options["temperature"]; ok { var ptr *float64; setF(&ptr, v); if ptr != nil { req.Temperature = ptr } }
    if v, ok := options["frequency_penalty"]; ok { var ptr *float64; setF(&ptr, v); if ptr != nil { req.FrequencyPenalty = ptr } }
    if v, ok := options["presence_penalty"]; ok { var ptr *float64; setF(&ptr, v); if ptr != nil { req.PresencePenalty = ptr } }

    if v, ok := options["max_tokens"]; ok {
        switch t := v.(type) {
        case int:
            req.MaxTokens = &t
        case float64:
            ti := int(t); req.MaxTokens = &ti
        }
    }
    if v, ok := options["k"]; ok {
        switch t := v.(type) {
        case int:
            req.K = &t
        case float64:
            ti := int(t); req.K = &ti
        }
    }
    if v, ok := options["p"]; ok { var ptr *float64; setF(&ptr, v); if ptr != nil { req.P = ptr } }
    if v, ok := options["seed"]; ok {
        switch t := v.(type) {
        case int:
            req.Seed = &t
        case float64:
            ti := int(t); req.Seed = &ti
        }
    }

    // Call SDK
    resp, err := p.sdk.Chat(ctx, req)
    if err != nil {
        return "", fmt.Errorf("cohere SDK chat error: %w", err)
    }

    if resp == nil || resp.Message == nil || len(resp.Message.Content) == 0 {
        return "", fmt.Errorf("cohere SDK: empty response content")
    }

    var sb strings.Builder
    for _, it := range resp.Message.Content {
        if it == nil || it.Text == nil { continue }
        if it.Text.Text != "" {
            if sb.Len() > 0 { sb.WriteString("") }
            sb.WriteString(it.Text.Text)
        }
    }
    if sb.Len() == 0 {
        return "", fmt.Errorf("cohere SDK: no text content in message")
    }
    return sb.String(), nil
}
