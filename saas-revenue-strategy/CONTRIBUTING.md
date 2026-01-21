# Contributing to SaaS Revenue Strategy RAG Agent

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in GitHub Issues
2. If not, create a new issue with:
   - Clear description of the problem or feature
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Your environment (Python version, OS, etc.)

### Contributing Code

1. **Fork the repository**
   ```bash
   git fork https://github.com/deerfieldgreen/probable-giggle
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation if needed

4. **Test your changes**
   ```bash
   # Test basic functionality
   python scripts/verify_setup.py
   
   # Test your specific changes
   python -m py_compile your_file.py
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

6. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment: `cp .env.example .env` and add your API keys
4. Start Milvus: `docker-compose up -d`
5. Initialize database: `python scripts/setup_milvus.py`

## Code Style Guidelines

### Python

- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings for functions and classes
- Keep functions focused and single-purpose

Example:
```python
def process_document(text: str, metadata: Dict[str, Any]) -> Document:
    """
    Process raw text into a Document object.
    
    Args:
        text: The raw text content
        metadata: Additional metadata for the document
        
    Returns:
        A Document object with text and metadata
    """
    # Implementation here
    pass
```

### Configuration Files

- Use YAML for configuration
- Add comments to explain non-obvious settings
- Keep configurations organized by component

### Documentation

- Update README.md for major features
- Update QUICKSTART.md for user-facing changes
- Add inline comments for complex logic
- Keep documentation concise and clear

## Areas for Contribution

### High Priority

1. **Additional Data Sources**
   - Crunchbase integration
   - AngelList data mining
   - LinkedIn company insights
   - Industry reports

2. **Query Optimization**
   - Improve retrieval accuracy
   - Better hybrid search tuning
   - Query result caching

3. **Vector DB Performance**
   - Index optimization
   - Batch processing improvements
   - Memory usage optimization

### Medium Priority

4. **Output Formats**
   - JSON export
   - CSV export
   - Markdown reports
   - API endpoint

5. **Web Interface**
   - Simple web UI for queries
   - Dashboard for vector DB insights
   - Visualization of results

6. **Testing**
   - Unit tests for core functions
   - Integration tests
   - Performance benchmarks

### Nice to Have

7. **Advanced Features**
   - Multi-language support
   - Custom embedding models
   - Fine-tuning for SaaS domain
   - Advanced filtering

8. **DevOps**
   - CI/CD pipeline
   - Automated testing
   - Docker improvements
   - Kubernetes deployment

## Testing Guidelines

### Before Submitting

1. Ensure all Python files compile: `python -m py_compile *.py`
2. Test basic functionality: `python scripts/verify_setup.py`
3. Test your specific changes
4. Check for any broken functionality

### Writing Tests

If adding tests:
- Place test files in a `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Test both success and failure cases

## Documentation Guidelines

### Code Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include type hints
- Document parameters and return values

### User Documentation

- Update README.md for architectural changes
- Update QUICKSTART.md for usage changes
- Add examples for new features
- Keep language simple and clear

## Review Process

1. **Automated Checks**
   - Code style validation
   - Basic functionality tests
   - Documentation checks

2. **Code Review**
   - At least one maintainer review
   - Address feedback promptly
   - Keep discussions constructive

3. **Merge**
   - Squash commits if needed
   - Update changelog
   - Merge to main branch

## Community Guidelines

- Be respectful and constructive
- Help others learn and grow
- Share knowledge and insights
- Keep discussions on-topic
- Follow the code of conduct

## Questions?

- Open a GitHub issue for questions
- Tag issues appropriately
- Check existing issues first
- Provide context for your question

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to make this project better! 🚀
