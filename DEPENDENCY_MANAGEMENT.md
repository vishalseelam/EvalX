# EvalX Dependency Management Strategy

## Overview

EvalX uses a multi-layered dependency management approach to ensure stability, reproducibility, and conflict resolution across different deployment environments.

## Dependency Structure

### 1. Core Dependencies (pyproject.toml)
- **Purpose**: Flexible ranges for end-users
- **Format**: `package>=min_version,<max_version`
- **Update Policy**: Conservative, tested compatibility ranges

### 2. Lock File (requirements-lock.txt)
- **Purpose**: Exact versions for reproducible deployments
- **Format**: `package==exact_version`
- **Update Policy**: Updated quarterly or for security patches

### 3. Development Dependencies
- **Purpose**: Tools for development and testing
- **Isolation**: Separate optional dependency group
- **Flexibility**: More permissive ranges for tooling

## Conflict Resolution Strategy

### 1. Version Pinning Strategy

```python
# Primary dependencies with conservative ranges
"torch>=1.12.0,<3.0.0"  # Major version compatibility
"transformers>=4.20.0,<5.0.0"  # API stability within major version
"numpy>=1.21.0,<2.0.0"  # Avoid breaking changes in v2
```

### 2. Dependency Groups

#### Core ML Stack
- PyTorch ecosystem (torch, torchvision, torchaudio)
- Transformers and tokenizers
- NumPy, SciPy, scikit-learn

#### LLM Integrations
- OpenAI, Anthropic clients
- LangChain components
- LangSmith for monitoring

#### Evaluation Metrics
- ROUGE, BERT-score
- NLTK, spaCy for NLP
- Custom metric implementations

#### Utilities
- Pydantic for validation
- Rich for CLI output
- Async libraries (aiohttp, tenacity)

### 3. Conflict Detection

Common conflicts and resolutions:

#### PyTorch Version Conflicts
```bash
# Problem: Different PyTorch versions required
# Solution: Pin to LTS version with broad compatibility
torch>=2.0.0,<3.0.0
torchvision>=0.15.0,<1.0.0
torchaudio>=2.0.0,<3.0.0
```

#### Transformers Compatibility
```bash
# Problem: Model compatibility across versions
# Solution: Use stable API range
transformers>=4.20.0,<5.0.0
tokenizers>=0.13.0,<1.0.0
```

#### NumPy Version Issues
```bash
# Problem: NumPy 2.0 breaking changes
# Solution: Stay on 1.x until ecosystem catches up
numpy>=1.21.0,<2.0.0
```

### 4. Testing Strategy

#### Dependency Matrix Testing
```yaml
# .github/workflows/test-matrix.yml
strategy:
  matrix:
    python-version: [3.9, 3.10, 3.11, 3.12]
    dependency-version: [minimum, latest]
```

#### Minimum Version Testing
```bash
# Test with minimum supported versions
pip install -e .[dev] --constraint constraints-min.txt
```

#### Latest Version Testing
```bash
# Test with latest compatible versions
pip install -e .[dev] --upgrade
```

## Installation Methods

### 1. Standard Installation
```bash
pip install evalx
```

### 2. Reproducible Installation
```bash
pip install -r requirements-lock.txt
```

### 3. Development Installation
```bash
pip install -e .[dev]
```

### 4. Full Installation
```bash
pip install evalx[all]
```

## Dependency Updates

### 1. Regular Updates (Quarterly)
- Review security advisories
- Test compatibility with new versions
- Update lock file with tested versions

### 2. Security Updates (As Needed)
- Immediate updates for critical vulnerabilities
- Patch releases for dependency security fixes

### 3. Major Version Updates (Annually)
- Comprehensive testing across all features
- Breaking change documentation
- Migration guides for users

## Conflict Resolution Procedures

### 1. Detection
```bash
# Check for conflicts
pip-check
pip list --outdated
```

### 2. Analysis
```bash
# Analyze dependency tree
pipdeptree
pip show package-name
```

### 3. Resolution
```bash
# Force resolution with constraints
pip install --constraint constraints.txt
```

### 4. Testing
```bash
# Run full test suite
pytest tests/
# Run integration tests
pytest tests/integration/
```

## Environment-Specific Considerations

### 1. Production Environment
- Use exact versions from lock file
- Minimal dependency set
- Security-focused updates

### 2. Development Environment
- Use flexible ranges from pyproject.toml
- Include development tools
- Allow for experimentation

### 3. Research Environment
- Include research dependencies
- Jupyter, visualization tools
- Experiment tracking (W&B, MLflow)

### 4. CI/CD Environment
- Use lock file for reproducibility
- Test multiple Python versions
- Validate dependency security

## Monitoring and Maintenance

### 1. Automated Dependency Scanning
- Dependabot for security updates
- Renovate for version updates
- License compliance checking

### 2. Dependency Health Metrics
- Update frequency tracking
- Vulnerability exposure time
- Compatibility test results

### 3. Documentation Updates
- Keep dependency docs current
- Update installation guides
- Maintain troubleshooting guides

## Troubleshooting Common Issues

### 1. PyTorch Installation Issues
```bash
# CUDA version mismatch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU-only installation
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Transformers Model Issues
```bash
# Clear cache for fresh download
transformers-cli cache clear

# Use specific model revision
model = AutoModel.from_pretrained("bert-base-uncased", revision="main")
```

### 3. Memory Issues with Large Models
```bash
# Use model sharding
model = AutoModel.from_pretrained("large-model", device_map="auto")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

## Best Practices

### 1. Version Management
- Always specify minimum versions
- Use upper bounds for major versions
- Test with both minimum and latest versions

### 2. Dependency Isolation
- Use virtual environments
- Separate development and production dependencies
- Document system requirements

### 3. Security Practices
- Regular security audits
- Automated vulnerability scanning
- Prompt security updates

### 4. Documentation
- Keep dependency documentation current
- Document known conflicts and resolutions
- Provide clear installation instructions

## Future Considerations

### 1. Python Version Support
- Plan for Python 3.13+ support
- Deprecation timeline for older versions
- Migration guides for breaking changes

### 2. Emerging Dependencies
- Evaluation of new ML frameworks
- Integration with emerging LLM libraries
- Compatibility with new evaluation metrics

### 3. Performance Optimization
- Dependency size optimization
- Optional dependency strategies
- Lazy loading implementations 