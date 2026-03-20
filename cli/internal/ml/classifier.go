package ml

import (
	"bufio"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"runtime"

	"github.com/mattn/go-tflite"
	"golang.org/x/image/draw"
)

const (
	modelInputSize = 224
	modelFileName  = "mobilenet-v2.tflite"
	labelsFileName = "labels.txt"
)

type Classifier struct {
	interpreter *tflite.Interpreter
	labels      []string
}

func NewClassifier() (*Classifier, error) {
	modelPath, err := findFile(modelFileName)
	if err != nil {
		return nil, err
	}

	labelsPath, err := findFile(labelsFileName)
	if err != nil {
		return nil, err
	}

	labels, err := loadLabels(labelsPath)
	if err != nil {
		return nil, fmt.Errorf("loading labels: %w", err)
	}

	model := tflite.NewModelFromFile(modelPath)
	if model == nil {
		return nil, fmt.Errorf("failed to load TFLite model from %s", modelPath)
	}

	options := tflite.NewInterpreterOptions()
	options.SetNumThread(runtime.NumCPU())
	defer options.Delete()

	interpreter := tflite.NewInterpreter(model, options)
	if interpreter == nil {
		return nil, fmt.Errorf("failed to create TFLite interpreter")
	}

	if status := interpreter.AllocateTensors(); status != tflite.OK {
		return nil, fmt.Errorf("allocating tensors: status %v", status)
	}

	return &Classifier{interpreter: interpreter, labels: labels}, nil
}

func (c *Classifier) Close() {
	if c.interpreter != nil {
		c.interpreter.Delete()
	}
}

func (c *Classifier) Classify(imgPath string) (string, error) {
	img, err := openAndResize(imgPath)
	if err != nil {
		return "", fmt.Errorf("preprocessing %s: %w", imgPath, err)
	}

	input := c.interpreter.GetInputTensor(0)
	if input == nil {
		return "", fmt.Errorf("nil input tensor")
	}

	switch input.Type() {
	case tflite.UInt8:
		if err := fillUint8(input, img); err != nil {
			return "", err
		}
	case tflite.Float32:
		if err := fillFloat32(input, img); err != nil {
			return "", err
		}
	default:
		return "", fmt.Errorf("unsupported input tensor type: %v", input.Type())
	}

	if status := c.interpreter.Invoke(); status != tflite.OK {
		return "", fmt.Errorf("inference failed: status %v", status)
	}

	output := c.interpreter.GetOutputTensor(0)
	if output == nil {
		return "", fmt.Errorf("nil output tensor")
	}

	labelIdx, err := topIndex(output)
	if err != nil {
		return "", err
	}
	if labelIdx < 0 || labelIdx >= len(c.labels) {
		return "", fmt.Errorf("predicted index %d out of range (labels: %d)", labelIdx, len(c.labels))
	}

	return c.labels[labelIdx], nil
}

func findFile(name string) (string, error) {
	exe, _ := os.Executable()
	candidates := []string{
		filepath.Join(filepath.Dir(exe), name),
		filepath.Join("internal", "ml", name),
		name,
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("%q not found; expected beside the binary or at internal/ml/%s", name, name)
}

func openAndResize(path string) (*image.NRGBA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	src, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decoding image: %w", err)
	}

	dst := image.NewNRGBA(image.Rect(0, 0, modelInputSize, modelInputSize))
	draw.BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Over, nil)
	return dst, nil
}

func fillUint8(t *tflite.Tensor, img *image.NRGBA) error {
	buf := t.UInt8s()
	if len(buf) < modelInputSize*modelInputSize*3 {
		return fmt.Errorf("input tensor too small: %d", len(buf))
	}
	i := 0
	for y := 0; y < modelInputSize; y++ {
		for x := 0; x < modelInputSize; x++ {
			c := img.NRGBAAt(x, y)
			buf[i], buf[i+1], buf[i+2] = c.R, c.G, c.B
			i += 3
		}
	}
	return nil
}

func fillFloat32(t *tflite.Tensor, img *image.NRGBA) error {
	buf := t.Float32s()
	if len(buf) < modelInputSize*modelInputSize*3 {
		return fmt.Errorf("input tensor too small: %d", len(buf))
	}
	i := 0
	for y := 0; y < modelInputSize; y++ {
		for x := 0; x < modelInputSize; x++ {
			c := img.NRGBAAt(x, y)
			buf[i] = float32(c.R)/127.5 - 1.0
			buf[i+1] = float32(c.G)/127.5 - 1.0
			buf[i+2] = float32(c.B)/127.5 - 1.0
			i += 3
		}
	}
	return nil
}

func topIndex(t *tflite.Tensor) (int, error) {
	switch t.Type() {
	case tflite.UInt8:
		scores := t.UInt8s()
		best, idx := uint8(0), 0
		for i, v := range scores {
			if v > best {
				best, idx = v, i
			}
		}
		return idx, nil
	case tflite.Float32:
		scores := t.Float32s()
		best, idx := float32(-1e9), 0
		for i, v := range scores {
			if v > best {
				best, idx = v, i
			}
		}
		return idx, nil
	default:
		return 0, fmt.Errorf("unsupported output tensor type: %v", t.Type())
	}
}

func loadLabels(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var labels []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if line := sc.Text(); line != "" {
			labels = append(labels, line)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(labels) == 0 {
		return nil, fmt.Errorf("labels.txt is empty")
	}
	return labels, nil
}
