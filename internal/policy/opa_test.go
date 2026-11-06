package policy

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func leb128U32(v uint32) []byte {
	var b []byte
	for {
		c := byte(v & 0x7f)
		v >>= 7
		if v != 0 {
			c |= 0x80
		}
		b = append(b, c)
		if v == 0 {
			break
		}
	}
	return b
}

// buildOPAWASMStatusModule constructs a minimal valid WASM binary exporting:
//   - "memory" (1 page = 64KB)
//   - fnName (i32, i32) -> (i32) returning a constant status code (0=allow, 1=steer, 2=block, 3=terminate).
func buildOPAWASMStatusModule(fnName string, returnCode byte) []byte {
	var buf bytes.Buffer
	buf.Write([]byte{0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00})

	// Type section: (i32, i32) -> (i32)
	typeContent := []byte{0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f}
	buf.WriteByte(1)
	buf.Write(leb128U32(uint32(len(typeContent))))
	buf.Write(typeContent)

	// Function section: 1 func of type 0
	funcContent := []byte{0x01, 0x00}
	buf.WriteByte(3)
	buf.Write(leb128U32(uint32(len(funcContent))))
	buf.Write(funcContent)

	// Memory section: 1 memory, min 1 page
	memContent := []byte{0x01, 0x00, 0x01}
	buf.WriteByte(5)
	buf.Write(leb128U32(uint32(len(memContent))))
	buf.Write(memContent)

	// Export section: "memory" and fnName
	var exportContent bytes.Buffer
	exportContent.WriteByte(0x02)
	exportContent.WriteByte(0x06)
	exportContent.WriteString("memory")
	exportContent.Write([]byte{0x02, 0x00})
	exportContent.WriteByte(byte(len(fnName)))
	exportContent.WriteString(fnName)
	exportContent.Write([]byte{0x00, 0x00})
	buf.WriteByte(7)
	buf.Write(leb128U32(uint32(exportContent.Len())))
	buf.Write(exportContent.Bytes())

	// Code section: i32.const <returnCode>, end
	body := []byte{0x00, 0x41, returnCode, 0x0b}
	codeContent := append([]byte{0x01, byte(len(body))}, body...)
	buf.WriteByte(10)
	buf.Write(leb128U32(uint32(len(codeContent))))
	buf.Write(codeContent)

	return buf.Bytes()
}

// buildOPAWASMJSONModule constructs a WASM module with a Data section storing jsonPayload at offset 0,
// and exporting "evaluate" (i32, i32) -> (i32, i32) returning (0, len(jsonPayload)).
func buildOPAWASMJSONModule(jsonPayload []byte) []byte {
	var buf bytes.Buffer
	buf.Write([]byte{0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00})

	// Type section: (i32, i32) -> (i32, i32)
	typeContent := []byte{0x01, 0x60, 0x02, 0x7f, 0x7f, 0x02, 0x7f, 0x7f}
	buf.WriteByte(1)
	buf.Write(leb128U32(uint32(len(typeContent))))
	buf.Write(typeContent)

	// Function section
	funcContent := []byte{0x01, 0x00}
	buf.WriteByte(3)
	buf.Write(leb128U32(uint32(len(funcContent))))
	buf.Write(funcContent)

	// Memory section: 1 page
	memContent := []byte{0x01, 0x00, 0x01}
	buf.WriteByte(5)
	buf.Write(leb128U32(uint32(len(memContent))))
	buf.Write(memContent)

	// Export section: "memory" and "evaluate"
	exportContent := []byte{
		0x02,
		0x06, 'm', 'e', 'm', 'o', 'r', 'y', 0x02, 0x00,
		0x08, 'e', 'v', 'a', 'l', 'u', 'a', 't', 'e', 0x00, 0x00,
	}
	buf.WriteByte(7)
	buf.Write(leb128U32(uint32(len(exportContent))))
	buf.Write(exportContent)

	// Code section: i32.const 0, i32.const <len>, end
	var body bytes.Buffer
	body.WriteByte(0x00) // 0 locals
	body.Write([]byte{0x41, 0x00})
	body.WriteByte(0x41)
	body.Write(leb128U32(uint32(len(jsonPayload))))
	body.WriteByte(0x0b)

	var codeSec bytes.Buffer
	codeSec.WriteByte(0x01)
	codeSec.Write(leb128U32(uint32(body.Len())))
	codeSec.Write(body.Bytes())
	buf.WriteByte(10)
	buf.Write(leb128U32(uint32(codeSec.Len())))
	buf.Write(codeSec.Bytes())

	// Data section (id=11): active segment at mem 0, offset i32.const 0
	var dataSec bytes.Buffer
	dataSec.WriteByte(0x01)                 // 1 segment
	dataSec.WriteByte(0x00)                 // memidx 0
	dataSec.Write([]byte{0x41, 0x00, 0x0b}) // i32.const 0, end
	dataSec.Write(leb128U32(uint32(len(jsonPayload))))
	dataSec.Write(jsonPayload)
	buf.WriteByte(11)
	buf.Write(leb128U32(uint32(dataSec.Len())))
	buf.Write(dataSec.Bytes())

	return buf.Bytes()
}

func TestOPAEvaluator_LocalRules(t *testing.T) {
	rules := []OPARule{
		{
			ID:         "block_destructive_mcp_under_deception",
			Tools:      []string{"exec_sql", "bash"},
			BlastRadii: []string{"high", "critical"},
			MinScores:  map[string]float64{"deception": 0.35},
			Action:     ActionBlock,
			Reason:     "high deception score on critical MCP tool",
		},
		{
			ID:               "steer_answer_phase_deception",
			CoTPhases:        []string{"answer"},
			ExcludeIntents:   []IntentType{IntentCreativeWriting},
			MinScores:        map[string]float64{"deception": 0.60},
			MaxScores:        map[string]float64{"truth": 0.50},
			Action:           ActionSteer,
			SteeringMethod:   "biprojection",
			SteeringVector:   "deception_0",
			SteeringStrength: 0.65,
		},
		{
			ID:        "block_untrusted_to_privileged_agent",
			FromTiers: []string{"untrusted"},
			ToTiers:   []string{"privileged"},
			MinScores: map[string]float64{"deception": 0.40},
			Action:    ActionTerminate,
			Reason:    "untrusted agent deception propagation",
		},
	}

	ev, err := NewOPAEvaluator(OPAConfig{Rules: rules})
	if err != nil {
		t.Fatalf("NewOPAEvaluator failed: %v", err)
	}
	defer func() { _ = ev.Close(context.Background()) }()

	if ev.BundleDigest() == "" {
		t.Fatal("expected non-empty BundleDigest")
	}

	ctx := context.Background()

	// Case 1: Destructive MCP tool with deception > 0.35 -> Blocked
	dec, err := ev.Evaluate(ctx, OPAInput{
		Latent: map[string]float64{"deception": 0.42},
		MCP: &OPAMCPContext{
			ToolName:    "exec_sql",
			BlastRadius: "critical",
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dec.Allow || dec.Action != ActionBlock || dec.PolicyID != "block_destructive_mcp_under_deception" {
		t.Fatalf("expected block on exec_sql, got %+v", dec)
	}

	// Case 2: Read-only MCP tool with same deception score -> Allowed
	dec, err = ev.Evaluate(ctx, OPAInput{
		Latent: map[string]float64{"deception": 0.42},
		MCP: &OPAMCPContext{
			ToolName:    "read_file",
			BlastRadius: "low",
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !dec.Allow || dec.Action != ActionAllow {
		t.Fatalf("expected allow on read_file, got %+v", dec)
	}

	// Case 3: Answer phase with deception=0.72, truth=0.30, general conversation -> Steer
	dec, err = ev.Evaluate(ctx, OPAInput{
		Latent:   map[string]float64{"deception": 0.72, "truth": 0.30},
		CoTPhase: "answer",
		Intent:   IntentGeneralConversation,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !dec.Allow || dec.Action != ActionSteer || dec.SteeringMethod != "biprojection" {
		t.Fatalf("expected biprojection steer in answer phase, got %+v", dec)
	}

	// Case 4: Same scores but creative_writing intent -> Excluded from rule -> Allowed
	dec, err = ev.Evaluate(ctx, OPAInput{
		Latent:   map[string]float64{"deception": 0.72, "truth": 0.30},
		CoTPhase: "answer",
		Intent:   IntentCreativeWriting,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dec.Action != ActionAllow {
		t.Fatalf("expected allow for excluded creative_writing intent, got %+v", dec)
	}

	// Case 5: Multi-agent untrusted -> privileged with deception=0.55 -> Terminate
	dec, err = ev.Evaluate(ctx, OPAInput{
		Latent: map[string]float64{"deception": 0.55},
		Agent: &OPAAgentContext{
			FromAgent: "scraper-1",
			ToAgent:   "db-writer",
			FromTier:  "untrusted",
			ToTier:    "privileged",
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dec.Allow || dec.Action != ActionTerminate {
		t.Fatalf("expected terminate on untrusted->privileged propagation, got %+v", dec)
	}
}

func TestOPAEvaluator_WASMBackend(t *testing.T) {
	ctx := context.Background()

	// 1. Test scalar status codes (0=allow, 1=steer, 2=block, 3=terminate)
	for _, tc := range []struct {
		fn   string
		code byte
		want Action
		ok   bool
	}{
		{"evaluate", 0, ActionAllow, true},
		{"opa_eval", 1, ActionSteer, true},
		{"evaluate", 2, ActionBlock, false},
		{"opa_eval", 3, ActionTerminate, false},
	} {
		wasmBytes := buildOPAWASMStatusModule(tc.fn, tc.code)
		ev, err := NewOPAEvaluator(OPAConfig{WASMBytes: wasmBytes})
		if err != nil {
			t.Fatalf("NewOPAEvaluator(%s, %d) failed: %v", tc.fn, tc.code, err)
		}
		dec, err := ev.Evaluate(ctx, OPAInput{Latent: map[string]float64{"deception": 0.8}})
		if err != nil {
			t.Fatalf("Evaluate(%s, %d) failed: %v", tc.fn, tc.code, err)
		}
		if dec.Action != tc.want || dec.Allow != tc.ok {
			t.Fatalf("got (%v, %v), want (%v, %v)", dec.Action, dec.Allow, tc.want, tc.ok)
		}
		_ = ev.Close(ctx)
	}

	// 2. Test JSON decision payload returned from WASM linear memory
	wantDec := OPADecision{
		Allow:            false,
		Action:           ActionBlock,
		PolicyID:         "wasm_rego_rule_1",
		Reason:           "blocked_by_compiled_opa_wasm",
		SteeringMethod:   "mpoa",
		SteeringStrength: 0.75,
	}
	rawJSON, _ := json.Marshal(wantDec)
	wasmJSONMod := buildOPAWASMJSONModule(rawJSON)

	tmpDir := t.TempDir()
	wasmPath := filepath.Join(tmpDir, "policy.wasm")
	if err := os.WriteFile(wasmPath, wasmJSONMod, 0o600); err != nil {
		t.Fatalf("write temp wasm: %v", err)
	}

	ev, err := NewOPAEvaluator(OPAConfig{WASMPath: wasmPath})
	if err != nil {
		t.Fatalf("NewOPAEvaluator from file failed: %v", err)
	}
	defer func() { _ = ev.Close(ctx) }()

	dec, err := ev.Evaluate(ctx, OPAInput{Latent: map[string]float64{"deception": 0.91}})
	if err != nil {
		t.Fatalf("Evaluate JSON WASM failed: %v", err)
	}
	if dec.Allow || dec.Action != ActionBlock || dec.PolicyID != "wasm_rego_rule_1" || dec.BundleDigest == "" {
		t.Fatalf("unexpected JSON WASM decision: %+v", dec)
	}
}

func TestOPAEvaluator_HTTPBackend(t *testing.T) {
	ctx := context.Background()

	// 1. Structured OPADecision response
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req opaHTTPRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.Input.Latent["deception"] > 0.5 {
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"result": map[string]interface{}{
					"allow":     false,
					"action":    "block",
					"policy_id": "remote_opa_deception",
					"reason":    "remote sidecar blocked request",
				},
			})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"result": true,
		})
	}))
	defer srv.Close()

	ev, err := NewOPAEvaluator(OPAConfig{
		Endpoint: srv.URL,
		Timeout:  time.Second,
	})
	if err != nil {
		t.Fatalf("NewOPAEvaluator failed: %v", err)
	}

	decBlock, err := ev.Evaluate(ctx, OPAInput{Latent: map[string]float64{"deception": 0.85}})
	if err != nil {
		t.Fatalf("Evaluate HTTP block failed: %v", err)
	}
	if decBlock.Allow || decBlock.Action != ActionBlock || decBlock.PolicyID != "remote_opa_deception" {
		t.Fatalf("expected block from HTTP OPA, got %+v", decBlock)
	}

	decAllow, err := ev.Evaluate(ctx, OPAInput{Latent: map[string]float64{"deception": 0.15}})
	if err != nil {
		t.Fatalf("Evaluate HTTP allow failed: %v", err)
	}
	if !decAllow.Allow || decAllow.Action != ActionAllow {
		t.Fatalf("expected allow from HTTP OPA, got %+v", decAllow)
	}

	// 2. Unreachable endpoint with FailOpen=true vs FailOpen=false
	evFailOpen, _ := NewOPAEvaluator(OPAConfig{
		Endpoint: "http://127.0.0.1:1/v1/data/burnnotice",
		Timeout:  20 * time.Millisecond,
		FailOpen: true,
	})
	decFO, err := evFailOpen.Evaluate(ctx, OPAInput{Latent: map[string]float64{"deception": 0.9}})
	if err != nil || !decFO.Allow {
		t.Fatalf("expected fail-open allow, got dec=%+v err=%v", decFO, err)
	}

	evFailClosed, _ := NewOPAEvaluator(OPAConfig{
		Endpoint: "http://127.0.0.1:1/v1/data/burnnotice",
		Timeout:  20 * time.Millisecond,
		FailOpen: false,
	})
	decFC, err := evFailClosed.Evaluate(ctx, OPAInput{Latent: map[string]float64{"deception": 0.9}})
	if err == nil || decFC.Allow || decFC.Action != ActionBlock {
		t.Fatalf("expected fail-closed block error, got dec=%+v err=%v", decFC, err)
	}
}

func TestEngine_EvaluateContextAndKillSwitch(t *testing.T) {
	ctx := context.Background()
	eng := NewEngine([]Policy{
		{Name: "base_deception", Metric: "deception", Threshold: 0.75, Action: ActionSteer},
	})
	ks, _ := NewKillSwitch(func(string) {})
	eng.RegisterKillSwitch(ks)

	// Without OPAEvaluator
	dec, violations, err := eng.EvaluateContext(ctx, OPAInput{
		Latent: map[string]float64{"deception": 0.80},
	})
	if err != nil {
		t.Fatalf("EvaluateContext without OPA failed: %v", err)
	}
	if len(violations) != 1 || dec.Action != ActionSteer {
		t.Fatalf("expected 1 scalar steer violation, got dec=%+v violations=%+v", dec, violations)
	}

	// With OPAEvaluator that triggers ActionTerminate
	opa, err := NewOPAEvaluator(OPAConfig{
		Rules: []OPARule{
			{
				ID:        "critical_mcp_terminate",
				Tools:     []string{"drop_database"},
				MinScores: map[string]float64{"deception": 0.50},
				Action:    ActionTerminate,
			},
		},
	})
	if err != nil {
		t.Fatalf("NewOPAEvaluator failed: %v", err)
	}
	eng.SetOPAEvaluator(opa)
	if eng.OPAEvaluator() != opa {
		t.Fatal("OPAEvaluator getter mismatch")
	}

	dec, violations, err = eng.EvaluateContext(ctx, OPAInput{
		Latent: map[string]float64{"deception": 0.80},
		MCP:    &OPAMCPContext{ToolName: "drop_database"},
	})
	if err != nil {
		t.Fatalf("EvaluateContext with OPA failed: %v", err)
	}
	if dec.Allow || dec.Action != ActionTerminate || len(violations) != 2 {
		t.Fatalf("expected terminate decision and 2 violations, got dec=%+v violations=%+v", dec, violations)
	}
	if ks.IsArmed() {
		t.Fatal("expected KillSwitch to be disarmed/triggered on ActionTerminate")
	}
}

func TestOPAEvaluator_ConcurrentRace(t *testing.T) {
	ev, err := NewOPAEvaluator(OPAConfig{
		Rules: []OPARule{
			{
				ID:        "r1",
				Tools:     []string{"bash"},
				MinScores: map[string]float64{"deception": 0.5},
				Action:    ActionBlock,
			},
		},
	})
	if err != nil {
		t.Fatalf("NewOPAEvaluator failed: %v", err)
	}
	defer func() { _ = ev.Close(context.Background()) }()

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 25; j++ {
				if j%10 == 0 {
					ev.SetRules([]OPARule{
						{
							ID:        "r1",
							Tools:     []string{"bash"},
							MinScores: map[string]float64{"deception": 0.5},
							Action:    ActionBlock,
						},
					})
					_ = ev.BundleDigest()
				}
				_, _ = ev.Evaluate(context.Background(), OPAInput{
					Latent: map[string]float64{"deception": float64(j%10) / 10.0},
					MCP:    &OPAMCPContext{ToolName: "bash"},
				})
			}
		}(i)
	}
	wg.Wait()
}

func FuzzOPAEvaluator(f *testing.F) {
	f.Add("exec_sql", "critical", "answer", 0.75, 0.25)
	f.Add("read_file", "low", "thinking", 0.10, 0.90)
	f.Add("", "", "", 0.0, 1.0)

	ev, err := NewOPAEvaluator(OPAConfig{
		Rules: []OPARule{
			{
				ID:         "fuzz_rule_1",
				Tools:      []string{"exec_sql", "bash"},
				BlastRadii: []string{"high", "critical"},
				MinScores:  map[string]float64{"deception": 0.4},
				MaxScores:  map[string]float64{"truth": 0.6},
				Action:     ActionBlock,
			},
		},
	})
	if err != nil {
		f.Fatalf("NewOPAEvaluator: %v", err)
	}

	f.Fuzz(func(t *testing.T, tool, blast, cot string, decScore, truthScore float64) {
		dec, err := ev.Evaluate(context.Background(), OPAInput{
			Latent: map[string]float64{
				"deception": decScore,
				"truth":     truthScore,
			},
			MCP: &OPAMCPContext{
				ToolName:    tool,
				BlastRadius: blast,
			},
			CoTPhase: cot,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if dec.Action == "" {
			t.Fatal("empty action in OPADecision")
		}
	})
}
