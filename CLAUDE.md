# GXFT Project Guidelines

## Build Commands
- Generate documentation: `pandoc spec.md --filter mermaid-filter -o spec.pdf`
- Run notebook: `jupyter notebook test.ipynb`

## Test Commands
- Test the API server: `python -m api.apiserver`
- Test the SDK: `python -c "from sdk.sdk import SDK; sdk = SDK(); print(sdk)"`

## Style Guidelines
- **Indentation**: 4 spaces
- **Naming**:
  - Classes: PascalCase (e.g., `BaseEntity`)
  - Functions/Variables: snake_case (e.g., `create_dataset`)
  - Constants/Enums: UPPER_CASE or PascalCase for Enum classes
- **Imports**: Group standard library, third-party, then local imports
- **Type Annotations**: Use typing module with full type hints (Optional, List, Dict, etc.)
- **Documentation**: Google-style docstrings with Args/Returns/Raises sections
- **Error Handling**: Raise specific exceptions with descriptive messages, use loguru for logging

## File Organization
- `api/`: REST API implementation (FastAPI)
- `jobs/`: Job queue system (SQLite3-based)
- `sdk/`: Main SDK modules with entity definitions