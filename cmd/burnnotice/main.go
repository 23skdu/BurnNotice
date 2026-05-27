package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"burnnotice/internal/converter"
	"burnnotice/internal/datasets"
	"burnnotice/internal/downloader"
	"burnnotice/internal/extract"
	"burnnotice/internal/profile"
	"burnnotice/internal/scoring"

	"github.com/spf13/cobra"
)

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
Supports authenticated access with --token flag.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		modelID := args[0]
		outputDir, _ := cmd.Flags().GetString("output")
		token, _ := cmd.Flags().GetString("token")
		include, _ := cmd.Flags().GetString("include")

		dl := downloader.NewHuggingFaceDownloader(token)

		var patterns []string
		if include != "" {
			patterns = append(patterns, include)
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
Supports safetensors and PyTorch bin formats.`,
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
behavioral directions (refusal, deception, truth, conflict, policy).`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		modelPath := args[0]
		outputDir, _ := cmd.Flags().GetString("output")
		datasetPath, _ := cmd.Flags().GetString("dataset")
		layers, _ := cmd.Flags().GetIntSlice("layers")

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
			positiveActivations = append(positiveActivations, pos)
			negativeActivations = append(negativeActivations, neg)
		}

		deceptionDir := extract.ExtractDIM(positiveActivations, negativeActivations)

		scorer.RegisterDirection("deception", deceptionDir)

		testActivation, _ := model.ExtractActivation("test prompt", "test response")
		scores := scorer.Score(testActivation)

		fmt.Printf("Sample scores: deception=%.4f\n", scores["deception"])

		modelProfile := &profile.ModelProfile{
			ModelID:    filepath.Base(modelPath),
			Directions: make(map[string]map[int][]float64),
		}
		modelProfile.Directions["deception"] = make(map[int][]float64)
		for _, layer := range layers {
			modelProfile.Directions["deception"][layer] = deceptionDir
		}
		modelProfile.MonitoredLayers = layers

		if err := modelProfile.Save(outputDir); err != nil {
			return fmt.Errorf("save profile: %w", err)
		}

		fmt.Printf("Profile saved to %s\n", outputDir)
		return nil
	},
}

func init() {
	downloadCmd.Flags().StringP("output", "o", "./models", "Output directory")
	downloadCmd.Flags().StringP("token", "t", "", "HuggingFace API token")
	downloadCmd.Flags().String("include", "*.safetensors;*.json", "File pattern filter")

	convertCmd.Flags().StringP("output", "o", "./onnx", "ONNX output directory")
	convertCmd.Flags().Int("opset", 17, "ONNX opset version")
	convertCmd.Flags().Bool("dynamic", true, "Use dynamic axes")

	profileCmd.Flags().StringP("output", "o", "./profiles", "Profile output directory")
	profileCmd.Flags().String("dataset", "", "Path to profiling dataset (JSON/JSONL)")
	profileCmd.Flags().IntSlice("layers", []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}, "Specific layers to profile (default: all)")

	rootCmd.AddCommand(downloadCmd, convertCmd, profileCmd)
}

func main() {
	if err := rootCmd.ExecuteContext(context.Background()); err != nil {
		os.Exit(1)
	}
}
