.PHONY: all resolve build test test-integration test-coverage test-python test-all lint clean help

SCHEME = pixart-swift-mlx
DERIVED_DATA = $(HOME)/Library/Developer/Xcode/DerivedData
DESTINATION = platform=macOS,arch=arm64
VERSION := $(shell git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "0.0.0")

all: build

resolve:
	@echo "Resolving package dependencies..."
	xcodebuild -resolvePackageDependencies -scheme $(SCHEME) -destination '$(DESTINATION)'

build: resolve
	@echo "Building $(SCHEME) (debug)..."
	xcodebuild -scheme $(SCHEME) -destination '$(DESTINATION)' build

test:
	@echo "Running Swift tests..."
	ACERVO_APP_GROUP_ID=group.intrusive-memory.models xcodebuild test -scheme $(SCHEME) -destination '$(DESTINATION)'

test-integration:
	@echo "Running integration tests (requires real model weights, ~2 GB unified memory)..."
	ACERVO_APP_GROUP_ID=group.intrusive-memory.models xcodebuild test -scheme $(SCHEME) -destination '$(DESTINATION)' SWIFT_ACTIVE_COMPILATION_CONDITIONS=INTEGRATION_TESTS

test-coverage:
	@echo "Running tests with code coverage enabled..."
	@rm -rf Build/test.xcresult
	@mkdir -p Build
	ACERVO_APP_GROUP_ID=group.intrusive-memory.models xcodebuild test -scheme $(SCHEME) -destination '$(DESTINATION)' -enableCodeCoverage YES -resultBundlePath Build/test.xcresult
	@echo ""
	@echo "Coverage summary (PixArtBackbone target):"
	@xcrun xccov view --report --only-targets Build/test.xcresult | grep -E "PixArtBackbone\b" || echo "(target not found)"

test-python:
	@echo "Running Python conversion script tests..."
	python3 -m unittest scripts.test_conversion -v

test-all: test test-python

lint:
	@echo "Formatting Swift sources..."
	swift format -i -r .

clean:
	@echo "Cleaning..."
	rm -rf $(DERIVED_DATA)/pixart-swift-mlx-*
	rm -rf Build

help:
	@echo "pixart-swift-mlx v$(VERSION)"
	@echo ""
	@echo "Targets:"
	@echo "  resolve      Resolve SPM dependencies"
	@echo "  build        Debug build (default)"
	@echo "  test              Run Swift unit tests"
	@echo "  test-integration  Run integration tests (requires real model weights; not run in CI)"
	@echo "  test-coverage     Run tests with -enableCodeCoverage and print PixArtBackbone summary"
	@echo "  test-python       Run Python conversion script tests"
	@echo "  test-all          Run all tests (Swift + Python)"
	@echo "  lint         Format Swift sources with swift-format"
	@echo "  clean        Remove build artifacts and DerivedData"
	@echo "  help         Show this help"
