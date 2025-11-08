# Makefile for MyST site

# Targets
.PHONY: env html clean

# Create or update conda environment
env:
	conda env update --file environment.yml --prune --name ligo

# Build local HTML site
html:
	myst build --html

# Clean up generated files and folders
clean:
	rm -rf figures/* audio/* _build
