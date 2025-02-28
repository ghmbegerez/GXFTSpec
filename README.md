# GXFTSpec

## Overview
This repository contains the specification and implementation for the GXFT system.

## Documentation

### Spec File
- `spec.md`: Main specification document 
- Use markdown preview and mermaid preview for visualization

### Generating PDF Documentation
```bash
# Generate PDF from spec.md
pandoc spec.md --filter mermaid-filter -o spec.pdf
```

#### Prerequisites
1. Install LaTeX support:
```bash
brew install --cask basictex
eval "$(/usr/libexec/path_helper)"
```

2. Install Pandoc:
```bash
brew install pandoc
```

3. Install Mermaid filter:
```bash
npm install -g mermaid-filter
```

## Project Structure

### API Directory
- **api.py**: REST API implementation
- **paiserver.py**: REST server

### Jobs Directory
- **jobsqueuesystem.py**: SQLite3-based job queue system

### SDK Directory
- **entities.py**: Primary entity definitions
- **managers.py**: Entity management classes
- **sdk.py**: Main SDK module
- **finetuner.py**: Fine-tuning support
- **externaltools.py**: Open source command-line library support
- **clisdk.py**: Command-line SDK client
- **utils.py**: Utility functions

## Version History
### v0.1
- Initial SDK entity review
- Specification adjustment to align with SDK