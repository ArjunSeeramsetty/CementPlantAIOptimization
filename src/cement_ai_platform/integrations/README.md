# Integrations Module

This module provides integrations with external services and APIs for the Cement Plant AI Optimization Platform.

## Available Integrations

### Zerve AI

**Status**: ✅ Available

**Description**: Multi-agent orchestration, fleet execution, and agentic workflows for data science tasks.

**Quick Start**:
```python
from cement_ai_platform.integrations import ZerveClient

client = ZerveClient.from_env()
workflow = client.create_workflow(
    name="My Workflow",
    description="Process and analyze cement plant data"
)
results = workflow.execute()
```

**Documentation**: See `docs/ZERVE_INTEGRATION_GUIDE.md`

**Features**:
- Multi-agent orchestration for complex tasks
- Fleet execution for parallel processing
- Pre-built workflow templates
- Real-time state inspection
- Production deployment (APIs, dashboards, scheduled jobs)

## Module Structure

```
integrations/
├── __init__.py              # Public API exports
├── zerve_integration.py     # Zerve AI implementation
└── README.md               # This file
```

## Adding New Integrations

To add a new integration:

1. Create a new file: `{service}_integration.py`
2. Implement the integration classes
3. Export public API in `__init__.py`
4. Add documentation
5. Add examples in `scripts/`

Example structure:

```python
# my_service_integration.py

class MyServiceClient:
    """Client for MyService API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    @classmethod
    def from_env(cls):
        """Create client from environment variables."""
        return cls(os.getenv('MY_SERVICE_API_KEY'))
    
    def do_something(self):
        """Do something useful."""
        pass
```

Then in `__init__.py`:

```python
from .my_service_integration import MyServiceClient

__all__ = [
    'ZerveClient',
    'MyServiceClient'  # Add new client
]
```

## Environment Variables

Integrations use environment variables for configuration:

### Zerve AI
```bash
ZERVE_API_KEY=your_api_key
ZERVE_WORKSPACE_ID=your_workspace_id
ZERVE_ENVIRONMENT=development
```

### Future Integrations
```bash
# Add environment variables for new integrations here
```

## Testing Integrations

Run integration tests:

```bash
# Test Zerve integration
python -c "from cement_ai_platform.integrations import ZerveClient; client = ZerveClient.from_env(); print('✅ OK')"

# Run full test suite
pytest tests/test_integrations.py
```

## Support

For integration issues:

1. Check the specific integration documentation
2. Verify environment variables are set correctly
3. Test API connectivity
4. Check logs in `logs/` directory
5. Refer to troubleshooting sections in integration guides

## Contributing

When adding new integrations:

- Follow existing code style and patterns
- Add comprehensive documentation
- Include usage examples
- Add unit and integration tests
- Update this README

