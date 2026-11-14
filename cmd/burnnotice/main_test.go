package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBurnNotice_RootCmdHelp(t *testing.T) {
	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetErr(&bytes.Buffer{})
	rootCmd.SetArgs([]string{"--help"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("help command failed: %v", err)
	}
}

func TestBurnNotice_RecommendCmd(t *testing.T) {
	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetErr(&bytes.Buffer{})
	rootCmd.SetArgs([]string{"recommend"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("recommend command failed: %v", err)
	}
}

func TestBurnNotice_DownloadCmd_Errors(t *testing.T) {
	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetErr(&bytes.Buffer{})
	// 1. Missing arg
	rootCmd.SetArgs([]string{"download"})
	if err := rootCmd.Execute(); err == nil {
		t.Error("expected error for missing arg in download")
	}

	// 2. Too many args
	rootCmd.SetArgs([]string{"download", "arg1", "arg2"})
	if err := rootCmd.Execute(); err == nil {
		t.Error("expected error for too many args in download")
	}

	// 3. Valid args but nonexistent model (network failure/not found)
	rootCmd.SetArgs([]string{"download", "nonexistent/model", "--output", t.TempDir(), "--include", "*.json;*.safetensors"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected download error for nonexistent model")
	}
}

func TestBurnNotice_ConvertCmd_Errors(t *testing.T) {
	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetErr(&bytes.Buffer{})
	// 1. Missing arg
	rootCmd.SetArgs([]string{"convert"})
	if err := rootCmd.Execute(); err == nil {
		t.Error("expected error for missing arg in convert")
	}

	// 2. Nonexistent directory
	rootCmd.SetArgs([]string{"convert", "/nonexistent/dir", "--output", t.TempDir()})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected convert error for nonexistent dir")
	}
}

func TestBurnNotice_ValidateCmd(t *testing.T) {
	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetErr(&bytes.Buffer{})

	// 1. Missing arg
	rootCmd.SetArgs([]string{"validate"})
	if err := rootCmd.Execute(); err == nil {
		t.Error("expected error for missing arg in validate")
	}

	// 2. Nonexistent file
	rootCmd.SetArgs([]string{"validate", "/nonexistent/model.onnx"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error for nonexistent file in validate")
	}

	// 3. Valid file (synthetic/dummy ONNX)
	hiddenDim := 16
	modelBytes, err := createDummyONNXModelForCLI(hiddenDim)
	if err != nil {
		t.Fatalf("failed to create dummy model: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "model.onnx")
	if err := os.WriteFile(path, modelBytes, 0644); err != nil {
		t.Fatal(err)
	}

	rootCmd.SetArgs([]string{"validate", path, "--hidden-dim", "16"})
	err = rootCmd.Execute()
	if err != nil {
		if strings.Contains(err.Error(), "onnxruntime") || strings.Contains(err.Error(), "shared library") {
			t.Logf("skipping validation execution check because ONNX shared library is missing: %v", err)
			return
		}
		t.Errorf("validate failed: %v", err)
	}
}

func TestBurnNotice_ProfileCmd_Errors(t *testing.T) {
	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetErr(&bytes.Buffer{})

	// 1. Missing model arg
	rootCmd.SetArgs([]string{"profile"})
	if err := rootCmd.Execute(); err == nil {
		t.Error("expected error for missing model arg in profile")
	}

	// 2. Missing --dataset flag
	rootCmd.SetArgs([]string{"profile", "/path/to/model.onnx"})
	err := rootCmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--dataset is required") {
		t.Errorf("expected error '--dataset is required', got: %v", err)
	}

	// 3. Unsupported dataset format
	rootCmd.SetArgs([]string{"profile", "/path/to/model.onnx", "--dataset", "invalid_format.txt"})
	err = rootCmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "unsupported dataset format") {
		t.Errorf("expected error 'unsupported dataset format', got: %v", err)
	}

	// 4. Dataset load failure (file not found)
	rootCmd.SetArgs([]string{"profile", "/path/to/model.onnx", "--dataset", "nonexistent.json"})
	err = rootCmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "load dataset") {
		t.Errorf("expected load dataset error, got: %v", err)
	}
}

func TestBurnNotice_ProfileCmd_Execution(t *testing.T) {
	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetErr(&bytes.Buffer{})

	// Create a dummy model
	hiddenDim := 16
	modelBytes, err := createDummyONNXModelForCLI(hiddenDim)
	if err != nil {
		t.Fatalf("failed to create dummy model: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "model.onnx")
	if err := os.WriteFile(path, modelBytes, 0644); err != nil {
		t.Fatal(err)
	}

	// Create a dummy JSON dataset
	datasetJSON := `
	{
		"pairs": [
			{
				"truthful": {
					"prompt": "prompt1",
					"response": "response1"
				},
				"deceptive": {
					"prompt": "prompt2",
					"response": "response2"
				}
			}
		]
	}
	`
	datasetPath := filepath.Join(dir, "dataset.json")
	if err := os.WriteFile(datasetPath, []byte(datasetJSON), 0644); err != nil {
		t.Fatal(err)
	}

	// 1. Run profile without oversight dataset
	rootCmd.SetArgs([]string{
		"profile", path,
		"--dataset", datasetPath,
		"--output", dir,
		"--layers", "0",
		"--model-size", "0.5B",
	})
	err = rootCmd.Execute()
	if err != nil {
		if strings.Contains(err.Error(), "onnxruntime") || strings.Contains(err.Error(), "shared library") {
			t.Log("skipping profile execution because ONNX shared library is missing")
		} else {
			t.Errorf("profile command failed: %v", err)
		}
	}

	// 2. Run profile with oversight dataset (unsupported format)
	rootCmd.SetArgs([]string{
		"profile", path,
		"--dataset", datasetPath,
		"--oversight-dataset", "invalid.txt",
		"--output", dir,
		"--layers", "0",
	})
	err = rootCmd.Execute()
	if err != nil {
		if strings.Contains(err.Error(), "onnxruntime") || strings.Contains(err.Error(), "shared library") {
			t.Log("skipping oversight format test because ONNX shared library is missing")
		} else if !strings.Contains(err.Error(), "unsupported oversight dataset format") {
			t.Errorf("expected error for unsupported oversight format, got: %v", err)
		}
	} else {
		t.Error("expected error for unsupported oversight format, got nil")
	}

	// 3. Run profile with oversight dataset (nonexistent file)
	rootCmd.SetArgs([]string{
		"profile", path,
		"--dataset", datasetPath,
		"--oversight-dataset", "nonexistent.json",
		"--output", dir,
		"--layers", "0",
	})
	err = rootCmd.Execute()
	if err != nil {
		if strings.Contains(err.Error(), "onnxruntime") || strings.Contains(err.Error(), "shared library") {
			t.Log("skipping nonexistent oversight file test because ONNX shared library is missing")
		} else if !strings.Contains(err.Error(), "load oversight dataset") {
			t.Errorf("expected error for nonexistent oversight file, got: %v", err)
		}
	} else {
		t.Error("expected error for nonexistent oversight file, got nil")
	}

	// 4. Run profile with --cone-analysis and --rdo flags
	rootCmd.SetArgs([]string{
		"profile", path,
		"--dataset", datasetPath,
		"--oversight-dataset", "",
		"--output", dir,
		"--layers", "0",
		"--cone-analysis",
		"--rdo",
	})
	err = rootCmd.Execute()
	if err != nil {
		if strings.Contains(err.Error(), "onnxruntime") || strings.Contains(err.Error(), "shared library") {
			t.Log("skipping cone-analysis and rdo test because ONNX shared library is missing")
		} else {
			t.Errorf("profile with --cone-analysis and --rdo failed: %v", err)
		}
	}
}

func TestBurnNotice_AggregateCmd(t *testing.T) {
	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetErr(&bytes.Buffer{})

	// 1. Nonexistent directory
	rootCmd.SetArgs([]string{"aggregate", "/nonexistent/telemetry/dir"})
	err := rootCmd.Execute()
	if err == nil {
		t.Error("expected error for nonexistent telemetry dir in aggregate")
	}

	// 2. Empty directory
	emptyDir := t.TempDir()
	rootCmd.SetArgs([]string{"aggregate", emptyDir})
	err = rootCmd.Execute()
	if err != nil {
		t.Errorf("expected clean execution for empty telemetry dir, got: %v", err)
	}

	// 3. Directory with valid report JSON
	reportDir := t.TempDir()
	reportJSON := `{
		"model_architecture": "llama",
		"model_size": "7B",
		"steering_method": "whitened_svd",
		"refusal_rate": 0.05,
		"kl_divergence": 0.12,
		"capability_retention": 0.98,
		"hardware": "cuda",
		"timestamp": 1234567890,
		"version": 1
	}`
	if err := os.WriteFile(filepath.Join(reportDir, "telemetry_1.json"), []byte(reportJSON), 0600); err != nil {
		t.Fatal(err)
	}

	rootCmd.SetArgs([]string{"aggregate", reportDir})
	err = rootCmd.Execute()
	if err != nil {
		t.Errorf("aggregate failed on valid reports: %v", err)
	}
}

func TestBurnNotice_FinetuneCmd(t *testing.T) {
	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetErr(&bytes.Buffer{})

	dir := t.TempDir()
	ckptPath := filepath.Join(dir, "ckpt.json")
	outPath := filepath.Join(dir, "ckpt_hardened.json")

	ckptJSON := `{"model_id":"qwen2.5-7b","high_risk":false,"risk_score":0.95,"layers":{"0":[[3.0,4.0],[0.0,5.0]]}}`
	if err := os.WriteFile(ckptPath, []byte(ckptJSON), 0600); err != nil {
		t.Fatal(err)
	}

	// Missing --extended-refusal flag
	rootCmd.SetArgs([]string{"finetune", ckptPath})
	if err := rootCmd.Execute(); err == nil {
		t.Error("expected error when --extended-refusal is omitted")
	}

	// Valid execution with --extended-refusal and output
	rootCmd.SetArgs([]string{"finetune", ckptPath, "--extended-refusal", "--output", outPath, "--rank", "2", "--alpha", "2.0", "--high-risk"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("finetune command failed: %v", err)
	}
	if _, err := os.Stat(outPath); err != nil {
		t.Errorf("expected hardened checkpoint output at %s: %v", outPath, err)
	}

	// Nonexistent checkpoint error
	rootCmd.SetArgs([]string{"finetune", filepath.Join(dir, "missing.json"), "--extended-refusal"})
	if err := rootCmd.Execute(); err == nil {
		t.Error("expected error for missing checkpoint")
	}

	// Rollback verification
	rootCmd.SetArgs([]string{"finetune", ckptPath, "--extended-refusal", "--high-risk", "--rollback"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("finetune --rollback failed: %v", err)
	}
}

func TestBurnNotice_ConvertCmd_Success(t *testing.T) {
	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetErr(&bytes.Buffer{})

	modelDir := t.TempDir()
	outDir := t.TempDir()

	// Build minimal valid safetensors file
	header := `{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}`
	buf := make([]byte, 8+len(header)+16)
	binary.LittleEndian.PutUint64(buf[:8], uint64(len(header)))
	copy(buf[8:8+len(header)], header)

	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), buf, 0600); err != nil {
		t.Fatal(err)
	}

	rootCmd.SetArgs([]string{"convert", modelDir, "--output", outDir})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("convert failed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(outDir, "model.onnx")); err != nil {
		t.Fatalf("expected converted model.onnx: %v", err)
	}
}

func TestBurnNotice_ProfileCmd_ExtendedFlags(t *testing.T) {
	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetErr(&bytes.Buffer{})

	hiddenDim := 16
	modelBytes, err := createDummyONNXModelForCLI(hiddenDim)
	if err != nil {
		t.Fatalf("failed to create dummy model: %v", err)
	}

	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.onnx")
	if err := os.WriteFile(modelPath, modelBytes, 0600); err != nil {
		t.Fatal(err)
	}

	datasetJSON := `{"pairs":[
		{"truthful":{"prompt":"alpha one","response":"yes one"},"deceptive":{"prompt":"beta one","response":"no one"}},
		{"truthful":{"prompt":"alpha two","response":"yes two"},"deceptive":{"prompt":"beta two","response":"no two"}},
		{"truthful":{"prompt":"alpha three","response":"yes three"},"deceptive":{"prompt":"beta three","response":"no three"}}
	]}`
	datasetPath := filepath.Join(dir, "dataset.json")
	if err := os.WriteFile(datasetPath, []byte(datasetJSON), 0600); err != nil {
		t.Fatal(err)
	}

	oversightJSON := `{"pairs":[
		{"truthful":{"prompt":"monitored","response":"safe"},"deceptive":{"prompt":"deployed","response":"unsafe"}}
	]}`
	oversightPath := filepath.Join(dir, "oversight.json")
	if err := os.WriteFile(oversightPath, []byte(oversightJSON), 0600); err != nil {
		t.Fatal(err)
	}

	outDir := filepath.Join(dir, "profile_out")
	ckptFile := filepath.Join(dir, "ckpt.json")
	telDir := filepath.Join(dir, "telemetry")

	rootCmd.SetArgs([]string{
		"profile", modelPath,
		"--dataset", datasetPath,
		"--oversight-dataset", oversightPath,
		"--output", outDir,
		"--layers", "0,1,2",
		"--inlp",
		"--rdo",
		"--cone-analysis",
		"--commit-window",
		"--calibrate",
		"--validate",
		"--version",
		"--progressive",
		"--distributed-nodes", "2",
		"--checkpoint", ckptFile,
		"--align-imprint",
		"--cross-layer",
		"--separation",
		"--telemetry-dir", telDir,
		"--tokenizer", "byte",
	})
	if err := rootCmd.Execute(); err != nil {
		if strings.Contains(err.Error(), "onnxruntime") || strings.Contains(err.Error(), "shared library") {
			t.Logf("skipping extended profile test: %v", err)
			return
		}
		t.Fatalf("extended profile failed: %v", err)
	}
}

func TestBurnNotice_Subcommands_EndToEnd(t *testing.T) {
	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetErr(&bytes.Buffer{})

	dir := t.TempDir()

	// eval with default synthetic data and with input file (exercises inspectZJSON)
	evalPath := filepath.Join(dir, "eval.json")
	evalJSON := `{"positive":[[1.0,0.2,0.1,0.0],[0.9,0.3,0.2,0.1],[1.1,0.1,0.0,0.2]],"negative":[[-1.0,-0.2,0.1,0.0],[-0.9,-0.3,0.2,-0.1],[-1.1,-0.1,0.0,-0.2]],"base_logits":[[2.0,1.0,0.5]],"steer_logits":[[1.9,1.1,0.5]],"harm_results":[{"prompt_id":"1","category":"cybercrime","baseline_success":true,"steered_success":false,"early_token_shift":0.02}],"pareto_points":[{"strength":0.5,"kl_divergence":0.03,"refusal_rate":0.95}],"pre_scores":{"0":0.8},"post_scores":{"0":0.2},"target_dim":4}`
	if err := os.WriteFile(evalPath, []byte(evalJSON), 0600); err != nil {
		t.Fatal(err)
	}

	policyYAML := filepath.Join(dir, "policies.yaml")
	if err := os.WriteFile(policyYAML, []byte("policies:\n  - name: deception_gate\n    metric: deception\n    threshold: 0.7\n    action: steer\n"), 0600); err != nil {
		t.Fatal(err)
	}

	evidenceOut := filepath.Join(dir, "evidence")
	pdfOut := filepath.Join(dir, "compliance.pdf")
	csvOut := filepath.Join(dir, "alerts.csv")
	regDir := filepath.Join(dir, "registry")

	commands := [][]string{
		{"eval"},
		{"eval", evalPath},
		{"steer"},
		{"sae"},
		{"policy"},
		{"policy", policyYAML, "--tool", "exec_sql", "--blast-radius", "high", "--deception", "0.8"},
		{"compliance", "--output", evidenceOut, "--pdf", pdfOut},
		{"registry", regDir},
		{"ledger", "--driver", "sqlite", "--export-csv", csvOut},
		{"agent", "summarize status"},
		{"runtime"},
	}

	for _, args := range commands {
		rootCmd.SetArgs(args)
		if err := rootCmd.Execute(); err != nil {
			t.Errorf("command %v failed: %v", args, err)
		}
	}
}
