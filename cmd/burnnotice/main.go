package main

import (
	"fmt"
	"os"

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

		fmt.Printf("Downloading %s to %s (token: %v, include: %s)\n", modelID, outputDir, token != "", include)

		// TODO: Call downloader.HuggingFaceDownloader
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

		fmt.Printf("Converting %s to ONNX (output: %s, opset: %d, dynamic: %v)\n", modelDir, output, opset, dynamic)

		// TODO: Call converter.ONNXConverter
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
		output, _ := cmd.Flags().GetString("output")
		dataset, _ := cmd.Flags().GetString("dataset")
		layers, _ := cmd.Flags().GetIntSlice("layers")

		fmt.Printf("Profiling %s (dataset: %s, output: %s, layers: %v)\n", modelPath, dataset, output, layers)

		// TODO: Call profiler to load model, run dataset, extract directions, save profile
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
	profileCmd.Flags().IntSlice("layers", nil, "Specific layers to profile (default: all)")

	rootCmd.AddCommand(downloadCmd, convertCmd, profileCmd)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
