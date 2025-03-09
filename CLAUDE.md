# ET Project Development Guide

## Build & Test Commands
- Run application: `streamlit run app.py`
- Run all tests: `python -m unittest discover -s tests`
- Run a single test: `python -m unittest tests/test_irrigation.py`
- Quick project structure test: `python test_simple.py`

## Code Style Guidelines
- **Formatting**: 4-space indentation, max line length ~100 chars
- **Imports**: Group imports (stdlib, third-party, local) with blank lines between groups
- **Naming**: `snake_case` for variables/functions, `CamelCase` for classes, `ALL_CAPS` for constants
- **Docstrings**: Google-style docstrings with Args: and Returns: sections
- **Type Hints**: Use typing annotations for function parameters and return values
- **Error Handling**:
  - Use custom exceptions (e.g., `OpenETError`)
  - Prefer specific exception catching
  - Add context when re-raising
  - Validate inputs at function start
- **Session State**: Initialize all Streamlit session state variables in `initialize_session_state()`

## Commit Guidelines
- Write descriptive commit messages with a clear action verb
- Do not include the name "Claude" or any AI assistant names in commit messages
- Keep commit scope focused on related changes