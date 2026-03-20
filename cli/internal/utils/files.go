package utils

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

var supportedExtensions = map[string]bool{
	".jpg": true, ".jpeg": true, ".png": true,
}

func FindImages(root string) ([]string, error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("accessing %s: %w", root, err)
	}

	if !info.IsDir() {
		if !isSupported(root) {
			return nil, fmt.Errorf("%s is not a supported image type (.jpg, .jpeg, .png)", root)
		}
		abs, err := filepath.Abs(root)
		if err != nil {
			return nil, err
		}
		return []string{abs}, nil
	}

	var images []string
	err = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && isSupported(path) {
			abs, err := filepath.Abs(path)
			if err != nil {
				return err
			}
			images = append(images, abs)
		}
		return nil
	})
	return images, err
}

func EnsureDir(dir string) error {
	return os.MkdirAll(dir, 0o755)
}

func SafeRename(src, word, destDir string, labelCount map[string]int) (string, error) {
	ext := strings.ToLower(filepath.Ext(src))

	dir := filepath.Dir(src)
	if destDir != "" {
		dir = destDir
	}

	dest := buildDest(dir, word, ext, labelCount[word])

	if destDir == "" {
		if err := os.Rename(src, dest); err != nil {
			return "", fmt.Errorf("renaming %s → %s: %w", src, dest, err)
		}
	} else {
		if err := copyFile(src, dest); err != nil {
			return "", fmt.Errorf("copying %s → %s: %w", src, dest, err)
		}
	}

	fmt.Printf("  %-40s → %s\n", filepath.Base(src), filepath.Base(dest))
	return dest, nil
}

func isSupported(path string) bool {
	return supportedExtensions[strings.ToLower(filepath.Ext(path))]
}

func buildDest(dir, word, ext string, count int) string {
	if count == 0 {
		return filepath.Join(dir, word+ext)
	}
	return filepath.Join(dir, fmt.Sprintf("%s-%d%s", word, count+1, ext))
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	return err
}
