.PHONY: all resolve build install release test test-python test-all lint clean help

SCHEME = pixart-swift-mlx-Package
BINARY = PixArtCLI
BIN_DIR = ./bin
DERIVED_DATA = $(HOME)/Library/Developer/Xcode/DerivedData
DESTINATION = platform=macOS,arch=arm64
VERSION := $(shell git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "0.0.0")

all: install

resolve:
	@echo "Resolving package dependencies..."
	xcodebuild -resolvePackageDependencies -scheme $(SCHEME) -destination '$(DESTINATION)'

build: resolve
	@echo "Building $(SCHEME) (debug)..."
	xcodebuild -scheme $(SCHEME) -destination '$(DESTINATION)' build

install: resolve
	@echo "Building and installing $(BINARY) to $(BIN_DIR)..."
	xcodebuild -scheme $(SCHEME) -destination '$(DESTINATION)' build
	@mkdir -p $(BIN_DIR)
	@PRODUCT_DIR=$$(find $(DERIVED_DATA)/pixart-swift-mlx-*/Build/Products/Debug -name "$(BINARY)" -type f 2>/dev/null | head -1) && \
	if [ -n "$$PRODUCT_DIR" ]; then \
		cp "$$PRODUCT_DIR" $(BIN_DIR)/$(BINARY); \
		echo "Installed $(BINARY) to $(BIN_DIR)/$(BINARY)"; \
	else \
		echo "Error: $(BINARY) not found in DerivedData"; \
		exit 1; \
	fi
	@BUNDLE=$$(find $(DERIVED_DATA)/pixart-swift-mlx-*/Build/Products/Debug -name "mlx-swift_Cmlx.bundle" -type d 2>/dev/null | head -1) && \
	if [ -n "$$BUNDLE" ]; then \
		cp -R "$$BUNDLE" $(BIN_DIR)/; \
		echo "Copied mlx-swift_Cmlx.bundle to $(BIN_DIR)/"; \
	fi

release: resolve
	@echo "Building $(SCHEME) (release)..."
	xcodebuild -scheme $(SCHEME) -configuration Release -destination '$(DESTINATION)' build
	@mkdir -p $(BIN_DIR)
	@PRODUCT_DIR=$$(find $(DERIVED_DATA)/pixart-swift-mlx-*/Build/Products/Release -name "$(BINARY)" -type f 2>/dev/null | head -1) && \
	if [ -n "$$PRODUCT_DIR" ]; then \
		cp "$$PRODUCT_DIR" $(BIN_DIR)/$(BINARY); \
		echo "Installed $(BINARY) (release) to $(BIN_DIR)/$(BINARY)"; \
	else \
		echo "Error: $(BINARY) not found in DerivedData"; \
		exit 1; \
	fi
	@BUNDLE=$$(find $(DERIVED_DATA)/pixart-swift-mlx-*/Build/Products/Release -name "mlx-swift_Cmlx.bundle" -type d 2>/dev/null | head -1) && \
	if [ -n "$$BUNDLE" ]; then \
		cp -R "$$BUNDLE" $(BIN_DIR)/; \
		echo "Copied mlx-swift_Cmlx.bundle to $(BIN_DIR)/"; \
	fi

test:
	@echo "Running Swift tests..."
	ACERVO_APP_GROUP_ID=group.intrusive-memory.models xcodebuild test -scheme $(SCHEME) -destination '$(DESTINATION)'

test-python:
	@echo "Running Python conversion script tests..."
	python3 -m unittest scripts.test_conversion -v

test-all: test test-python

lint:
	@echo "Formatting Swift sources..."
	swift format -i -r .

clean:
	@echo "Cleaning..."
	rm -rf $(BIN_DIR)
	rm -rf $(DERIVED_DATA)/pixart-swift-mlx-*

help:
	@echo "pixart-swift-mlx v$(VERSION)"
	@echo ""
	@echo "Targets:"
	@echo "  resolve   Resolve SPM dependencies"
	@echo "  build     Debug build"
	@echo "  install   Debug build + copy binary to $(BIN_DIR) (default)"
	@echo "  release   Release build + copy binary to $(BIN_DIR)"
	@echo "  test         Run Swift unit tests"
	@echo "  test-python  Run Python conversion script tests"
	@echo "  test-all     Run all tests (Swift + Python)"
	@echo "  lint      Format Swift sources with swift-format"
	@echo "  clean     Remove build artifacts and DerivedData"
	@echo "  help      Show this help"
