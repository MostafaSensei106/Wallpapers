package ml

import (
	"regexp"
	"strings"
	"unicode"
)

var nonAlpha = regexp.MustCompile(`[^a-zA-Z]+`)

func LabelToWord(label, lang string) string {
	label = strings.TrimSpace(label)
	if len(label) > 9 && label[0] == 'n' && isDigitSlice(label[1:9]) {
		label = strings.TrimSpace(label[9:])
	}

	parts := strings.FieldsFunc(label, func(r rune) bool {
		return r == ',' || r == '/' || unicode.IsSpace(r)
	})

	for _, p := range parts {
		clean := strings.ToLower(nonAlpha.ReplaceAllString(p, ""))
		if len(clean) >= 3 && !containsDigit(clean) {
			return clean
		}
	}

	for _, p := range parts {
		clean := strings.ToLower(nonAlpha.ReplaceAllString(p, ""))
		if len(clean) > 0 {
			return clean
		}
	}

	return "unknown"
}

func isDigitSlice(s string) bool {
	for _, r := range s {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

func containsDigit(s string) bool {
	for _, r := range s {
		if unicode.IsDigit(r) {
			return true
		}
	}
	return false
}
