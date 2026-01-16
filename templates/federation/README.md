# Federation Pattern Template

Demonstrates cross-project pattern sharing and learning for multi-project AI agent orchestration.

## Features

- Pattern recording (success/failure/optimization)
- Cross-project pattern sharing
- Project subscriptions
- Persistent storage

## Concepts

### Patterns
Learned behaviors that can be shared:
- **Success patterns**: What works well
- **Failure patterns**: What to avoid
- **Optimization patterns**: Performance improvements

### Federation
Projects can:
- Export patterns to share
- Import patterns from other projects
- Subscribe to automatic updates

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY=your_api_key

# Run
python main.py
```

## API

```python
# Create a pattern store
store = FederatedPatternStore(project_id="my-project")

# Record a pattern
pattern = Pattern(
    pattern_id="success_001",
    pattern_type="success",
    signature="api_call:gemini",
)
store.record_pattern(pattern)

# Share patterns
exported = store.export_patterns()

# Import from another project
store.import_patterns(exported, source_project="other-project")

# Subscribe to a project
store.subscribe("other-project")
```

## Next Steps

- Connect to a real Graphiti/Neo4j backend
- Implement pattern decay for relevance
- Add pattern conflict resolution
