package mlp

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
	"time"
)

const (
	modelPath  = "../../data/traced_model.pt"
	imagesPath = "../../data/t10k-images-idx3-ubyte"
	labelsPath = "../../data/t10k-labels-idx1-ubyte"
)

func loadMNISTImages(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 16)
	if _, err := f.Read(header); err != nil {
		return nil, err
	}

	numImages := int(binary.BigEndian.Uint32(header[4:8]))
	rows := int(binary.BigEndian.Uint32(header[8:12]))
	cols := int(binary.BigEndian.Uint32(header[12:16]))

	pixelsPerImage := rows * cols
	images := make([][]float64, numImages)
	for i := 0; i < numImages; i++ {
		imgData := make([]byte, pixelsPerImage)
		if _, err := f.Read(imgData); err != nil {
			return nil, err
		}
		images[i] = make([]float64, pixelsPerImage)
		for j := 0; j < pixelsPerImage; j++ {
			val := float64(imgData[j]) / 255.0
			images[i][j] = (val - 0.1307) / 0.3081
		}
	}
	return images, nil
}

func loadMNISTLabels(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 8)
	if _, err := f.Read(header); err != nil {
		return nil, err
	}

	numLabels := int(binary.BigEndian.Uint32(header[4:8]))
	labels := make([]int, numLabels)
	labelData := make([]byte, numLabels)
	if _, err := f.Read(labelData); err != nil {
		return nil, err
	}
	for i := 0; i < numLabels; i++ {
		labels[i] = int(labelData[i])
	}
	return labels, nil
}

func TestMLP(t *testing.T) {
	images, err := loadMNISTImages(imagesPath)
	if err != nil {
		t.Fatalf("Failed to load images: %v", err)
	}

	labels, err := loadMNISTLabels(labelsPath)
	if err != nil {
		t.Fatalf("Failed to load labels: %v", err)
	}

	btEv, ev, params, encoder, encryptor, decryptor := mlp__configure()

	total := 3
	correct := 0

	for i := 0; i < total; i++ {
		input := images[i]
		label := labels[i]

		inputFloat32 := make([]float32, len(input))
		for j := 0; j < len(input); j++ {
			inputFloat32[j] = float32(input[j])
		}

		ctInput := mlp__encrypt__arg0(ev, params, encoder, encryptor, inputFloat32)

		startTime := time.Now()
		resCt := mlp(btEv, ev, params, encoder, ctInput)
		duration := time.Since(startTime)
		t.Logf("Sample %d took %v", i, duration)

		resValues := mlp__decrypt__result0(ev, params, encoder, decryptor, resCt)

		maxVal := float32(-math.MaxFloat32)
		maxIdx := -1
		for j := 0; j < 10; j++ {
			if resValues[j] > maxVal {
				maxVal = resValues[j]
				maxIdx = j
			}
		}

		if maxIdx == label {
			correct++
		}
		t.Logf("Sample %d: predicted %d, actual %d", i, maxIdx, label)
		t.Logf("logits:\n")
		for j := 0; j < 10; j++ {
			t.Logf("%d, %.6f\n", j, resValues[j])
		}
	}

	accuracy := float64(correct) / float64(total)
	t.Logf("Accuracy: %.2f (%d/%d correct)", accuracy, correct, total)
	if accuracy < 0.6 {
		t.Errorf("Accuracy too low: %.2f (%d/%d correct)", accuracy, correct, total)
	}
}
