package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"burnnotice/internal/engine"
	"burnnotice/internal/converter"
	"burnnotice/internal/datasets"
	"burnnotice/internal/downloader"
	"burnnotice/internal/extract"
	"burnnotice/internal/profile"
	"burnnotice/internal/scoring"

	"github.com/spf13/cobra"
)

const recommendedModel = "Qwen/Qwen2.5-0.5B"

var rootCmd = &cobra.Command{
	Use:   "burnnotice",
	Short: "LLM Deception Detection and Alignment Monitoring",
	Long: `BurnNotice is a real-time neural activation monitoring and intervention
framework for detecting deception, policy violations, and alignment drift
in large language models through activation-space analysis.`,
}

var downloadCmd = &cobra.Command{
	Use:   "download [model-id]",
	Short: "Download a model from HuggingFace",
	Long: `Download model files from HuggingFace Hub for ONNX conversion.
Supports authenticated access with --token flag or HF_TOKEN env var.

Recommended model for testing: ` + "`" + recommendedModel + "`" + `
  burnnotice download ` + recommendedModel + `
  burnnotice download ` + recommendedModel + ` --include "*.safetensors;*.json"`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		modelID := args[0]
		outputDir, _ := cmd.Flags().GetString("output")
		token, _ := cmd.Flags().GetString("token")
		include, _ := cmd.Flags().GetString("include")

		if token == "" {
			token = os.Getenv("HF_TOKEN")
		}

		dl := downloader.NewHuggingFaceDownloader(token)

		var patterns []string
		if include != "" {
			for _, p := range strings.Split(include, ";") {
				p = strings.TrimSpace(p)
				if p != "" {
					patterns = append(patterns, p)
				}
			}
		}

		fmt.Printf("Downloading %s to %s...\n", modelID, outputDir)
		if err := dl.Download(modelID, outputDir, patterns); err != nil {
			return fmt.Errorf("download: %w", err)
		}

		fmt.Println("Download complete.")
		return nil
	},
}

var convertCmd = &cobra.Command{
	Use:   "convert [model-dir]",
	Short: "Convert a model to ONNX format",
	Long: `Convert HuggingFace model files to ONNX format for inference.
Supports safetensors and PyTorch bin formats.

For a runnable ONNX model (with compute graph), use optimum-cli:
  pip install optimum
  optimum-cli export onnx --model ` + "`" + recommendedModel + "`" + ` ./onnx/`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		modelDir := args[0]
		output, _ := cmd.Flags().GetString("output")
		opset, _ := cmd.Flags().GetInt("opset")
		dynamic, _ := cmd.Flags().GetBool("dynamic")

		c := converter.NewONNXConverter()
		opts := converter.Options{
			OpsetVersion: opset,
			DynamicAxes:  dynamic,
			OutputPath:   output,
		}

		fmt.Printf("Converting %s to ONNX...\n", modelDir)
		if err := c.Convert(modelDir, output, opts); err != nil {
			return fmt.Errorf("convert: %w", err)
		}

		fmt.Println("Conversion complete.")
		return nil
	},
}

var profileCmd = &cobra.Command{
	Use:   "profile [onnx-model]",
	Short: "Generate a model profile for deception detection",
	Long: `Run contrastive prompt batches through the model to extract
behavioral directions (refusal, deception, truth, conflict, policy).

Extraction method:
  By default, uses Difference-in-Means (DIM) for small models and
  whitened SVD with Ledoit-Wolf shrinkage for models > 7B parameters.
  Override with --whitened-svd=true|false.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		modelPath := args[0]
		outputDir, _ := cmd.Flags().GetString("output")
		datasetPath, _ := cmd.Flags().GetString("dataset")
		oversightPath, _ := cmd.Flags().GetString("oversight-dataset")
		layers, _ := cmd.Flags().GetIntSlice("layers")
		whitenedFlag, _ := cmd.Flags().GetBool("whitened-svd")
		modelSize, _ := cmd.Flags().GetString("model-size")
		numDirs, _ := cmd.Flags().GetInt("num-directions")
		shrinkage, _ := cmd.Flags().GetFloat64("shrinkage")
		eigFloor, _ := cmd.Flags().GetFloat64("eig-floor")

		if datasetPath == "" {
			return fmt.Errorf("--dataset is required")
		}

		loader := datasets.NewLoader()
		var ds *datasets.Dataset
		var err error

		switch filepath.Ext(datasetPath) {
		case ".json":
			ds, err = loader.LoadJSON(datasetPath)
		case ".jsonl":
			ds, err = loader.LoadJSONL(datasetPath)
		default:
			return fmt.Errorf("unsupported dataset format: %s", datasetPath)
		}
		if err != nil {
			return fmt.Errorf("load dataset: %w", err)
		}

		fmt.Printf("Loaded dataset with %d pairs\n", len(ds.Pairs))

		model, err := LoadONNXModel(modelPath)
		if err != nil {
			return fmt.Errorf("load ONNX model: %w", err)
		}
		defer model.Close()

		scorer := scoring.NewScorer(nil)

		var ablator *profile.OversightAblator
		var oversightDirs map[int][]float64
		if oversightPath != "" {
			var ods *datasets.Dataset
			switch filepath.Ext(oversightPath) {
			case ".json":
				ods, err = loader.LoadJSON(oversightPath)
			case ".jsonl":
				ods, err = loader.LoadJSONL(oversightPath)
			default:
				return fmt.Errorf("unsupported oversight dataset format: %s", oversightPath)
			}
			if err != nil {
				return fmt.Errorf("load oversight dataset: %w", err)
			}
			
			monitoredActs := make([][][]float64, 0, len(ods.Pairs))
			deployedActs := make([][][]float64, 0, len(ods.Pairs))
			
			for _, pair := range ods.Pairs {
				pos, err := model.ExtractActivation(pair.Truthful.Prompt, pair.Truthful.Response)
				if err != nil {
					return fmt.Errorf("extract monitored activation: %w", err)
				}
				neg, err := model.ExtractActivation(pair.Deceptive.Prompt, pair.Deceptive.Response)
				if err != nil {
					return fmt.Errorf("extract deployed activation: %w", err)
				}
				monitoredActs = append(monitoredActs, [][]float64{pos})
				deployedActs = append(deployedActs, [][]float64{neg})
			}
			
			oversightDirs, err = profile.ExtractOversightDirection(monitoredActs, deployedActs)
			if err != nil {
				return fmt.Errorf("extract oversight direction: %w", err)
			}
			ablator = profile.NewOversightAblator(oversightDirs)
			fmt.Printf("Extracted oversight directions across %d layers for ablation\n", len(oversightDirs))
		}

		positiveActivations := make([][]float64, 0, len(ds.Pairs))
		negativeActivations := make([][]float64, 0, len(ds.Pairs))

		for _, pair := range ds.Pairs {
			pos, err := model.ExtractActivation(pair.Truthful.Prompt, pair.Truthful.Response)
			if err != nil {
				return fmt.Errorf("extract positive activation: %w", err)
			}
			neg, err := model.ExtractActivation(pair.Deceptive.Prompt, pair.Deceptive.Response)
			if err != nil {
				return fmt.Errorf("extract negative activation: %w", err)
			}

			if ablator != nil {
				pos = ablator.Ablate(0, pos)
				neg = ablator.Ablate(0, neg)
			}

			positiveActivations = append(positiveActivations, pos)
			negativeActivations = append(negativeActivations, neg)
		}

		if ablator != nil {
			fmt.Println("Ablated oversight direction from deception dataset activations")
		}

		useWhitened := shouldUseWhitenedSVD(cmd.Flags().Changed("whitened-svd"), whitenedFlag, modelSize)
		var directions [][]float64
		var methodLabel string
		if useWhitened {
			var ok bool
			directions, _, ok = extract.ExtractWhitenedSVD(positiveActivations, negativeActivations, numDirs, shrinkage, eigFloor)
			if !ok || len(directions) == 0 {
				return fmt.Errorf("whitened SVD extraction failed (k=%d, shrinkage=%g, eigFloor=%g)", numDirs, shrinkage, eigFloor)
			}
			methodLabel = fmt.Sprintf("whitened SVD (k=%d, shrinkage=%g, eigFloor=%g)", len(directions), shrinkage, eigFloor)
		} else {
			k := numDirs
			if !cmd.Flags().Changed("num-directions") {
				k = 5
			}
			directions, _ = extract.ExtractPCA(positiveActivations, negativeActivations, k)
			if len(directions) == 0 {
				return fmt.Errorf("PCA SVD extraction failed (k=%d)", k)
			}
			methodLabel = fmt.Sprintf("PCA SVD (k=%d)", len(directions))
		}
		fmt.Printf("Extraction method: %s\n", methodLabel)

		for idx, dir := range directions {
			dirName := fmt.Sprintf("deception_%d", idx)
			scorer.RegisterDirection(dirName, dir)
			if idx == 0 {
				scorer.RegisterDirection("deception", dir)
			}
		}

		testActivation, _ := model.ExtractActivation("test prompt", "test response")
		scores := scorer.Score(testActivation)

		for idx := range directions {
			dirName := fmt.Sprintf("deception_%d", idx)
			fmt.Printf("Sample scores: %s=%.4f\n", dirName, scores[dirName])
		}
		if len(directions) > 0 {
			fmt.Printf("Sample scores: deception=%.4f\n", scores["deception"])
		}

		modelProfile := &profile.ModelProfile{
			ModelID:    filepath.Base(modelPath),
			Directions: make(map[string]map[int][]float64),
		}
		for idx, dir := range directions {
			dirName := fmt.Sprintf("deception_%d", idx)
			modelProfile.Directions[dirName] = make(map[int][]float64)
			for _, layer := range layers {
				modelProfile.Directions[dirName][layer] = dir
			}
			if idx == 0 {
				modelProfile.Directions["deception"] = make(map[int][]float64)
				for _, layer := range layers {
					modelProfile.Directions["deception"][layer] = dir
				}
			}
		}
		if len(oversightDirs) > 0 {
			modelProfile.Directions["oversight"] = oversightDirs
		}
		modelProfile.MonitoredLayers = layers

		if err := modelProfile.Save(outputDir); err != nil {
			return fmt.Errorf("save profile: %w", err)
		}

		fmt.Printf("Profile saved to %s\n", outputDir)
		return nil
	},
}

var recommendCmd = &cobra.Command{
	Use:   "recommend",
	Short: "Print recommended model and export instructions",
	Long: `Prints the recommended model for testing BurnNotice and step-by-step
instructions to export it to ONNX format using optimum-cli.`,
	Args: cobra.NoArgs,
	RunE: func(_ *cobra.Command, _ []string) error {
		fmt.Printf(`Recommended Model: %s

This is the smallest model in the Qwen2.5 family (0.5B parameters).
It is suitable for:
  • Quick ONNX export testing
  • BurnNotice profiling validation
  • CI/CD pipeline integration

Step-by-step export:
  1. Install optimum + torch:
     pip install optimum[onnxruntime] torch --quiet

  2. Export to ONNX:
     optimum-cli export onnx --model %s ./onnx/qwen2.5-0.5b

  3. Profile with BurnNotice:
     burnnotice profile ./onnx/qwen2.5-0.5b/model.onnx --dataset ./data/deception_pairs.json

Requirements:
  • Python 3.10+
  • 2 GB free disk space
  • 4 GB RAM
`, recommendedModel, recommendedModel)
		return nil
	},
}

var validateCmd = &cobra.Command{
	Use:   "validate [onnx-model]",
	Short: "Validate an ONNX model loads and runs inference",
	Long: `Loads an ONNX model and runs a forward pass with synthetic token
input to verify the engine functions correctly end-to-end.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		modelPath := args[0]
		hiddenDim, _ := cmd.Flags().GetInt("hidden-dim")

		cfg := engine.ModelConfig{
			HiddenDim:  hiddenDim,
			InputNames: []string{"input_ids"},
			OutputNames: []string{"hidden_states"},
		}
		eng := engine.NewONNXEngine(modelPath, cfg)
		if err := eng.Load(); err != nil {
			return fmt.Errorf("engine load failed: %w", err)
		}
		defer eng.Close()

		fmt.Printf("Engine loaded: %s\n", modelPath)
		fmt.Printf("Output count: %d\n", eng.NumOutputs())

		// Run forward pass with synthetic tokens
		syntheticTokens := []int64{1, 2, 3, 4, 5}
		output, err := eng.Forward(syntheticTokens, []int64{1, 5})
		if err != nil {
			return fmt.Errorf("forward pass failed: %w", err)
		}

		expectedSize := 1 * 5 * hiddenDim
		if len(output) != expectedSize {
			return fmt.Errorf("output size = %d, want %d", len(output), expectedSize)
		}

		fmt.Printf("Forward pass OK: output tensor %d elements [%d x %d x %d]\n",
			len(output), 1, 5, hiddenDim)
		fmt.Println("Validation PASSED")
		return nil
	},
}

func init() {
	downloadCmd.Flags().StringP("output", "o", "./models", "Output directory for downloaded files")
	downloadCmd.Flags().StringP("token", "t", "", "HuggingFace API token (or $HF_TOKEN)")
	downloadCmd.Flags().String("include", "*.safetensors;*.json", "Semicolon-separated file extensions to download")

	convertCmd.Flags().StringP("output", "o", "./onnx", "ONNX output directory")
	convertCmd.Flags().Int("opset", 17, "ONNX opset version")
	convertCmd.Flags().Bool("dynamic", true, "Use dynamic axes")

	profileCmd.Flags().StringP("output", "o", "./profiles", "Profile output directory")
	profileCmd.Flags().String("dataset", "", "Path to profiling dataset (JSON/JSONL)")
	profileCmd.Flags().String("oversight-dataset", "", "Path to oversight contrastive dataset (optional)")
	profileCmd.Flags().IntSlice("layers", []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}, "Specific layers to profile (default: all)")
	profileCmd.Flags().Bool("whitened-svd", false, "Use whitened SVD extraction instead of DIM (default: true for models > 7B)")
	profileCmd.Flags().String("model-size", "0.5B", "Model parameter count (e.g. 0.5B, 7B, 13B, 70B); drives the default --whitened-svd choice")
	profileCmd.Flags().Int("num-directions", 1, "Number of top directions to extract when --whitened-svd is set")
	profileCmd.Flags().Float64("shrinkage", -1, "Ledoit-Wolf shrinkage intensity for the whitening covariance (<0 = auto-estimate)")
	profileCmd.Flags().Float64("eig-floor", 0, "Minimum eigenvalue for the whitening matrix (0 = auto, 1e-10 of max)")

	validateCmd.Flags().Int("hidden-dim", 4096, "Model hidden dimension")

	rootCmd.AddCommand(downloadCmd, convertCmd, profileCmd, recommendCmd, validateCmd)
}

func main() {
	if err := rootCmd.ExecuteContext(context.Background()); err != nil {
		os.Exit(1)
	}
}
