package cmd

import (
	"fmt"
	"os"

	"github.com/MostafaSensei106/AI-Image-Rename/internal/ml"
	"github.com/MostafaSensei106/AI-Image-Rename/internal/utils"
	"github.com/spf13/cobra"
	"github.com/vbauerster/mpb/v8"
	"github.com/vbauerster/mpb/v8/decor"
)

var rootCmd = &cobra.Command{
	Use:   "air",
	Short: "AI-Powered image renaming tool using MobileNetV2",
	Long:  "ai-image-rename scans images using a pre-trained MobileNetV2 TFLite model and renames them based on the model's top prediction.",
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func init() {
	rootCmd.AddCommand(newRenameCmd())
}

func newRenameCmd() *cobra.Command {
	var lang string
	var outputDir string

	cmd := &cobra.Command{
		Use:   "rename [path_to_image_or_folder]",
		Short: "Rename images based on AI classification",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRename(args[0], lang, outputDir)
		},
	}

	cmd.Flags().StringVarP(&lang, "lang", "l", "en", "Output language for the label word")
	cmd.Flags().StringVarP(&outputDir, "output", "o", "", "Output directory (default: in-place)")

	return cmd
}

func runRename(inputPath, lang, outputDir string) error {
	images, err := utils.FindImages(inputPath)
	if err != nil {
		return fmt.Errorf("discovering images: %w", err)
	}
	if len(images) == 0 {
		fmt.Println("No supported images found.")
		return nil
	}
	fmt.Printf("Found %d image(s) to process.\n\n", len(images))

	classifier, err := ml.NewClassifier()
	if err != nil {
		return fmt.Errorf("loading model: %w", err)
	}
	defer classifier.Close()

	if outputDir != "" {
		if err := utils.EnsureDir(outputDir); err != nil {
			return fmt.Errorf("creating output directory: %w", err)
		}
	}

	p := mpb.New(mpb.WithWidth(60))
	bar := p.AddBar(int64(len(images)),
		mpb.PrependDecorators(
			decor.Name("Classifying", decor.WCSyncSpace),
			decor.CountersNoUnit(" %d/%d", decor.WCSyncSpace),
		),
		mpb.AppendDecorators(
			decor.Percentage(decor.WCSyncSpace),
			decor.Elapsed(decor.ET_STYLE_GO, decor.WCSyncSpace),
		),
	)

	labelCount := make(map[string]int)

	for _, imgPath := range images {
		label, err := classifier.Classify(imgPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  WARN: skipping %s: %v\n", imgPath, err)
			bar.Increment()
			continue
		}

		word := ml.LabelToWord(label, lang)
		dest, err := utils.SafeRename(imgPath, word, outputDir, labelCount)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  WARN: could not rename %s: %v\n", imgPath, err)
			bar.Increment()
			continue
		}

		labelCount[word]++
		bar.IncrBy(1)
		_ = dest
	}

	p.Wait()
	fmt.Println("\nDone.")
	return nil
}
