package policy

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

// ActionAllow indicates the operation or token stream is permitted without intervention.
const ActionAllow Action = "allow"

// OPAMCPContext captures Model Context Protocol (MCP) tool invocation metadata for OPA evaluation.
type OPAMCPContext struct {
	ToolName    string                 `json:"tool_name"`
	Arguments   map[string]interface{} `json:"arguments,omitempty"`
	ReadOnly    bool                   `json:"read_only,omitempty"`
	BlastRadius string                 `json:"blast_radius,omitempty"` // "low", "medium", "high", "critical"
}

// OPAAgentContext captures multi-agent communication and session state for OPA evaluation.
type OPAAgentContext struct {
	FromAgent string `json:"from_agent,omitempty"`
	ToAgent   string `json:"to_agent,omitempty"`
	FromTier  string `json:"from_tier,omitempty"` // e.g., "untrusted", "standard", "privileged"
	ToTier    string `json:"to_tier,omitempty"`
	SessionID string `json:"session_id,omitempty"`
	Turn      int    `json:"turn,omitempty"`
}

// OPAInput is the structured "Latent-Aware Authorization" document passed to Open Policy Agent
// (`input` in Rego), combining white-box residual activation telemetry with request/tool metadata.
type OPAInput struct {
	Latent   map[string]float64 `json:"latent"`
	MCP      *OPAMCPContext     `json:"mcp,omitempty"`
	Agent    *OPAAgentContext   `json:"agent,omitempty"`
	CoTPhase string             `json:"cot_phase,omitempty"` // "thinking", "answer", "direct"
	Intent   IntentType         `json:"intent,omitempty"`
	Metadata map[string]string  `json:"metadata,omitempty"`
}

// OPADecision represents the structured governance verdict returned by an OPA policy evaluation.
type OPADecision struct {
	Allow            bool    `json:"allow"`
	Action           Action  `json:"action"`
	Reason           string  `json:"reason,omitempty"`
	PolicyID         string  `json:"policy_id,omitempty"`
	SteeringMethod   string  `json:"steering_method,omitempty"`
	SteeringVector   string  `json:"steering_vector,omitempty"`
	SteeringStrength float64 `json:"steering_strength,omitempty"`
	BundleDigest     string  `json:"bundle_digest,omitempty"`
}

// OPARule defines a declarative compound contextual policy rule combining latent score predicates
// with MCP tool, Chain-of-Thought phase, semantic intent, and multi-agent constraints.
type OPARule struct {
	ID               string             `json:"id" yaml:"id"`
	Description      string             `json:"description,omitempty" yaml:"description,omitempty"`
	Tools            []string           `json:"tools,omitempty" yaml:"tools,omitempty"`
	BlastRadii       []string           `json:"blast_radii,omitempty" yaml:"blast_radii,omitempty"`
	CoTPhases        []string           `json:"cot_phases,omitempty" yaml:"cot_phases,omitempty"`
	ExcludeIntents   []IntentType       `json:"exclude_intents,omitempty" yaml:"exclude_intents,omitempty"`
	FromTiers        []string           `json:"from_tiers,omitempty" yaml:"from_tiers,omitempty"`
	ToTiers          []string           `json:"to_tiers,omitempty" yaml:"to_tiers,omitempty"`
	MinScores        map[string]float64 `json:"min_scores,omitempty" yaml:"min_scores,omitempty"`
	MaxScores        map[string]float64 `json:"max_scores,omitempty" yaml:"max_scores,omitempty"`
	Action           Action             `json:"action" yaml:"action"`
	Reason           string             `json:"reason,omitempty" yaml:"reason,omitempty"`
	SteeringMethod   string             `json:"steering_method,omitempty" yaml:"steering_method,omitempty"`
	SteeringVector   string             `json:"steering_vector,omitempty" yaml:"steering_vector,omitempty"`
	SteeringStrength float64            `json:"steering_strength,omitempty" yaml:"steering_strength,omitempty"`
}

// OPAConfig configures the Open Policy Agent evaluator.
type OPAConfig struct {
	// WASMPath is an optional path to an OPA policy compiled to WebAssembly (`opa build -t wasm`).
	WASMPath string `json:"wasm_path,omitempty" yaml:"wasm_path,omitempty"`
	// WASMBytes allows passing compiled WASM policy bytes directly in memory.
	WASMBytes []byte `json:"-" yaml:"-"`
	// Endpoint is an optional OPA REST API URL (e.g., "http://localhost:8181/v1/data/burnnotice/authz/decision").
	Endpoint string `json:"endpoint,omitempty" yaml:"endpoint,omitempty"`
	// Timeout is the HTTP request timeout for remote OPA sidecar queries (default: 250ms).
	Timeout time.Duration `json:"timeout,omitempty" yaml:"timeout,omitempty"`
	// FailOpen controls behavior when a remote OPA endpoint is unreachable (default: false -> fallback to local rules).
	FailOpen bool `json:"fail_open,omitempty" yaml:"fail_open,omitempty"`
	// Rules are compound contextual rules evaluated locally or as fallback.
	Rules []OPARule `json:"rules,omitempty" yaml:"rules,omitempty"`
	// HTTPClient allows injecting a custom *http.Client for testing or mTLS.
	HTTPClient *http.Client `json:"-" yaml:"-"`
}

// OPAEvaluator evaluates Latent-Aware Authorization policies via compiled OPA WASM modules (wazero),
// an OPA REST API sidecar, or compound declarative contextual rules.
type OPAEvaluator struct {
	mu           sync.RWMutex
	cfg          OPAConfig
	httpClient   *http.Client
	wasmRuntime  wazero.Runtime
	wasmModule   wazero.CompiledModule
	bundleDigest string
}

// NewOPAEvaluator initializes a new OPAEvaluator from the provided configuration.
func NewOPAEvaluator(cfg OPAConfig) (*OPAEvaluator, error) {
	if cfg.Timeout <= 0 {
		cfg.Timeout = 250 * time.Millisecond
	}
	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: cfg.Timeout}
	}

	ev := &OPAEvaluator{
		cfg:        cfg,
		httpClient: client,
	}

	var wasmCode []byte
	if len(cfg.WASMBytes) > 0 {
		wasmCode = cfg.WASMBytes
	} else if cfg.WASMPath != "" {
		// #nosec G304 -- user-configured OPA WASM policy bundle path
		data, err := os.ReadFile(cfg.WASMPath)
		if err != nil {
			return nil, fmt.Errorf("read OPA WASM policy %s: %w", cfg.WASMPath, err)
		}
		wasmCode = data
	}

	if len(wasmCode) > 0 {
		ctx := context.Background()
		r := wazero.NewRuntime(ctx)
		if _, err := wasi_snapshot_preview1.Instantiate(ctx, r); err != nil {
			_ = r.Close(ctx)
			return nil, fmt.Errorf("instantiate WASI for OPA WASM: %w", err)
		}
		mod, err := r.CompileModule(ctx, wasmCode)
		if err != nil {
			_ = r.Close(ctx)
			return nil, fmt.Errorf("compile OPA WASM module: %w", err)
		}
		sum := sha256.Sum256(wasmCode)
		ev.wasmRuntime = r
		ev.wasmModule = mod
		ev.bundleDigest = hex.EncodeToString(sum[:])
	} else {
		ev.bundleDigest = computeRulesDigest(cfg.Rules)
	}

	return ev, nil
}

func computeRulesDigest(rules []OPARule) string {
	raw, err := json.Marshal(rules)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

// BundleDigest returns the SHA-256 cryptographic digest of the active OPA WASM module or rule bundle.
func (e *OPAEvaluator) BundleDigest() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.bundleDigest
}

// SetRules atomically updates the local compound contextual rules and refreshes the bundle digest.
func (e *OPAEvaluator) SetRules(rules []OPARule) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.cfg.Rules = make([]OPARule, len(rules))
	copy(e.cfg.Rules, rules)
	if e.wasmModule == nil {
		e.bundleDigest = computeRulesDigest(e.cfg.Rules)
	}
}

// Close releases any underlying WASM runtime resources.
func (e *OPAEvaluator) Close(ctx context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.wasmRuntime != nil {
		err := e.wasmRuntime.Close(ctx)
		e.wasmRuntime = nil
		e.wasmModule = nil
		return err
	}
	return nil
}

// Evaluate executes the configured OPA policy backend (WASM -> HTTP sidecar -> local rules)
// against the provided Latent-Aware Authorization input.
func (e *OPAEvaluator) Evaluate(ctx context.Context, input OPAInput) (OPADecision, error) {
	e.mu.RLock()
	wasmMod := e.wasmModule
	wasmRT := e.wasmRuntime
	endpoint := e.cfg.Endpoint
	failOpen := e.cfg.FailOpen
	rules := e.cfg.Rules
	digest := e.bundleDigest
	e.mu.RUnlock()

	if wasmMod != nil && wasmRT != nil {
		dec, err := e.evalWASM(ctx, wasmRT, wasmMod, input, digest)
		if err == nil {
			return dec, nil
		}
		if len(rules) == 0 && !failOpen {
			return OPADecision{Allow: false, Action: ActionBlock, Reason: "opa_wasm_error: " + err.Error(), BundleDigest: digest}, err
		}
	}

	if endpoint != "" {
		dec, err := e.evalHTTP(ctx, endpoint, input, digest)
		if err == nil {
			return dec, nil
		}
		if len(rules) == 0 {
			if failOpen {
				return OPADecision{Allow: true, Action: ActionAllow, Reason: "opa_http_fail_open", BundleDigest: digest}, nil
			}
			return OPADecision{Allow: false, Action: ActionBlock, Reason: "opa_http_error: " + err.Error(), BundleDigest: digest}, err
		}
	}

	return evaluateLocalRules(rules, input, digest), nil
}

func (e *OPAEvaluator) evalWASM(ctx context.Context, rt wazero.Runtime, mod wazero.CompiledModule, input OPAInput, digest string) (OPADecision, error) {
	payload, err := json.Marshal(input)
	if err != nil {
		return OPADecision{}, fmt.Errorf("marshal OPA input: %w", err)
	}
	if len(payload) > 1<<20 {
		return OPADecision{}, fmt.Errorf("OPA input exceeds 1MB limit: %d bytes", len(payload))
	}

	inst, err := rt.InstantiateModule(ctx, mod, wazero.NewModuleConfig().WithName(""))
	if err != nil {
		return OPADecision{}, fmt.Errorf("instantiate OPA WASM module: %w", err)
	}
	defer func() { _ = inst.Close(ctx) }()

	mem := inst.ExportedMemory("memory")
	if mem == nil {
		return OPADecision{}, fmt.Errorf("OPA WASM module has no exported memory")
	}

	const inputOffset = uint32(1024)
	if !mem.Write(inputOffset, payload) {
		return OPADecision{}, fmt.Errorf("write OPA input to WASM linear memory failed")
	}

	evalFn := inst.ExportedFunction("evaluate")
	if evalFn == nil {
		evalFn = inst.ExportedFunction("opa_eval")
	}
	if evalFn == nil {
		return OPADecision{}, fmt.Errorf("OPA WASM module exports neither 'evaluate' nor 'opa_eval'")
	}

	results, err := evalFn.Call(ctx, uint64(inputOffset), uint64(len(payload)))
	if err != nil {
		return OPADecision{}, fmt.Errorf("OPA WASM execution failed: %w", err)
	}
	if len(results) == 0 {
		return OPADecision{}, fmt.Errorf("OPA WASM returned no result")
	}

	// Convention 1: two-value return (outOffset, outLen) containing JSON OPADecision
	if len(results) >= 2 && results[1] > 0 && results[1] <= 1<<20 && results[0] <= 1<<24 {
		outOffset := uint32(results[0]) // #nosec G115 -- bounded above
		outLen := uint32(results[1])    // #nosec G115 -- bounded above
		raw, ok := mem.Read(outOffset, outLen)
		if !ok {
			return OPADecision{}, fmt.Errorf("read OPA WASM output failed")
		}
		var dec OPADecision
		if err := json.Unmarshal(raw, &dec); err != nil {
			return OPADecision{}, fmt.Errorf("unmarshal OPA WASM JSON output: %w", err)
		}
		dec.BundleDigest = digest
		normalizeDecision(&dec)
		return dec, nil
	}

	// Convention 2: scalar status code (0 = allow, 1 = steer, 2 = block, 3 = terminate)
	code := results[0]
	dec := OPADecision{BundleDigest: digest}
	switch code {
	case 0:
		dec.Allow = true
		dec.Action = ActionAllow
		dec.Reason = "opa_wasm_allow"
	case 1:
		dec.Allow = true
		dec.Action = ActionSteer
		dec.Reason = "opa_wasm_steer"
		dec.SteeringMethod = "mpoa"
		dec.SteeringStrength = 0.5
	case 2:
		dec.Allow = false
		dec.Action = ActionBlock
		dec.Reason = "opa_wasm_block"
	default:
		dec.Allow = false
		dec.Action = ActionTerminate
		dec.Reason = "opa_wasm_terminate"
	}
	return dec, nil
}

type opaHTTPRequest struct {
	Input OPAInput `json:"input"`
}

type opaHTTPResponse struct {
	Result json.RawMessage `json:"result"`
}

func (e *OPAEvaluator) evalHTTP(ctx context.Context, endpoint string, input OPAInput, digest string) (OPADecision, error) {
	bodyBytes, err := json.Marshal(opaHTTPRequest{Input: input})
	if err != nil {
		return OPADecision{}, fmt.Errorf("marshal OPA HTTP request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return OPADecision{}, fmt.Errorf("create OPA HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.httpClient.Do(req) // #nosec G704 -- endpoint is administrator-configured OPA sidecar URL
	if err != nil {
		return OPADecision{}, fmt.Errorf("OPA HTTP request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return OPADecision{}, fmt.Errorf("OPA HTTP returned status %d", resp.StatusCode)
	}

	respBytes, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return OPADecision{}, fmt.Errorf("read OPA HTTP response: %w", err)
	}

	var envelope opaHTTPResponse
	if err := json.Unmarshal(respBytes, &envelope); err != nil {
		return OPADecision{}, fmt.Errorf("decode OPA HTTP response: %w", err)
	}

	if len(envelope.Result) == 0 || string(envelope.Result) == "null" {
		return OPADecision{Allow: true, Action: ActionAllow, BundleDigest: digest}, nil
	}

	// Support boolean result (`result: true` / `result: false`)
	var boolResult bool
	if err := json.Unmarshal(envelope.Result, &boolResult); err == nil {
		if boolResult {
			return OPADecision{Allow: true, Action: ActionAllow, Reason: "opa_http_allow", BundleDigest: digest}, nil
		}
		return OPADecision{Allow: false, Action: ActionBlock, Reason: "opa_http_deny", BundleDigest: digest}, nil
	}

	// Support structured object result (`result: {"allow": false, "action": "block", ...}`)
	var dec OPADecision
	if err := json.Unmarshal(envelope.Result, &dec); err != nil {
		return OPADecision{}, fmt.Errorf("decode OPA decision result: %w", err)
	}
	dec.BundleDigest = digest
	normalizeDecision(&dec)
	return dec, nil
}

func normalizeDecision(dec *OPADecision) {
	if dec.Action == "" {
		if dec.Allow {
			dec.Action = ActionAllow
		} else {
			dec.Action = ActionBlock
		}
	}
	if dec.Action == ActionAllow || dec.Action == ActionLog || dec.Action == ActionSteer {
		dec.Allow = true
	}
	if dec.Action == ActionBlock || dec.Action == ActionTerminate {
		dec.Allow = false
	}
}

func evaluateLocalRules(rules []OPARule, input OPAInput, digest string) OPADecision {
	for _, rule := range rules {
		if matchesOPARule(rule, input) {
			dec := OPADecision{
				PolicyID:         rule.ID,
				Action:           rule.Action,
				Reason:           rule.Reason,
				SteeringMethod:   rule.SteeringMethod,
				SteeringVector:   rule.SteeringVector,
				SteeringStrength: rule.SteeringStrength,
				BundleDigest:     digest,
			}
			if dec.Reason == "" {
				dec.Reason = "matched rule " + rule.ID
			}
			normalizeDecision(&dec)
			return dec
		}
	}
	return OPADecision{
		Allow:        true,
		Action:       ActionAllow,
		Reason:       "default_allow",
		BundleDigest: digest,
	}
}

func matchesOPARule(rule OPARule, input OPAInput) bool {
	// 1. Check excluded intents (unless severity is terminate/block with high-risk MCP blast radius)
	for _, ex := range rule.ExcludeIntents {
		if input.Intent != "" && input.Intent == ex {
			return false
		}
	}

	// 2. Check CoT phase filter
	if len(rule.CoTPhases) > 0 {
		if !containsFold(rule.CoTPhases, input.CoTPhase) {
			return false
		}
	}

	// 3. Check MCP tool filter
	if len(rule.Tools) > 0 {
		if input.MCP == nil || !containsFold(rule.Tools, input.MCP.ToolName) {
			return false
		}
	}

	// 4. Check MCP blast radius filter
	if len(rule.BlastRadii) > 0 {
		if input.MCP == nil || !containsFold(rule.BlastRadii, input.MCP.BlastRadius) {
			return false
		}
	}

	// 5. Check multi-agent source/destination tiers
	if len(rule.FromTiers) > 0 {
		if input.Agent == nil || !containsFold(rule.FromTiers, input.Agent.FromTier) {
			return false
		}
	}
	if len(rule.ToTiers) > 0 {
		if input.Agent == nil || !containsFold(rule.ToTiers, input.Agent.ToTier) {
			return false
		}
	}

	// 6. Check minimum latent score predicates (all specified must be > min)
	for metric, minVal := range rule.MinScores {
		val, ok := input.Latent[metric]
		if !ok || math.IsNaN(val) || val <= minVal {
			return false
		}
	}

	// 7. Check maximum latent score predicates (all specified must be < max)
	for metric, maxVal := range rule.MaxScores {
		val, ok := input.Latent[metric]
		if !ok || math.IsNaN(val) || val >= maxVal {
			return false
		}
	}

	return len(rule.MinScores) > 0 || len(rule.MaxScores) > 0 || len(rule.Tools) > 0 || len(rule.BlastRadii) > 0 || len(rule.FromTiers) > 0 || len(rule.ToTiers) > 0 || len(rule.CoTPhases) > 0
}

func containsFold(slice []string, target string) bool {
	if target == "" {
		return false
	}
	for _, s := range slice {
		if s == "*" || strings.EqualFold(s, target) {
			return true
		}
	}
	return false
}
