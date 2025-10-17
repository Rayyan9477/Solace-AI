# Solace-AI Testing Framework

This directory contains a comprehensive testing framework for the Solace-AI project with focus on security, authentication, and HIPAA compliance.

## Overview

The testing framework includes:
- **Unit Tests**: Test individual components and services
- **Integration Tests**: Test API endpoints and component interactions  
- **Security Tests**: Test security mechanisms and vulnerability prevention
- **HIPAA Compliance Tests**: Validate healthcare data protection requirements

## Directory Structure

```
tests/
├── README.md                          # This file
├── conftest.py                        # Pytest configuration and fixtures
├── test_utils.py                      # Testing utilities and helpers
├── unit/                              # Unit tests
│   └── test_user_service.py          # User service unit tests
├── integration/                       # Integration tests
│   ├── test_auth_endpoints.py        # Authentication API tests
│   └── test_api_endpoints.py         # General API endpoint tests
├── security/                          # Security tests
│   ├── test_security.py              # Security mechanism tests
│   └── test_hipaa_compliance.py      # HIPAA compliance tests
├── fixtures/                         # Test data fixtures
├── mocks/                            # Mock objects and services
├── reports/                          # Test reports (generated)
├── logs/                             # Test logs (generated)
└── coverage_html/                    # Coverage reports (generated)
```

## Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   export JWT_SECRET_KEY="test_secret_key_for_pytest_only_not_for_production_use_minimum_32_chars"
   export ENVIRONMENT="testing"
   export TESTING="true"
   ```

3. **Create Test Environment File** (optional):
   ```bash
   # Create .env.test file in project root
   echo "JWT_SECRET_KEY=test_secret_key_for_pytest_only_not_for_production_use_minimum_32_chars" > .env.test
   echo "ENVIRONMENT=testing" >> .env.test
   echo "TESTING=true" >> .env.test
   ```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only  
pytest -m security               # Security tests only
pytest -m auth                   # Authentication tests only
pytest -m hipaa                  # HIPAA compliance tests only

# Run quick tests (exclude slow tests)
pytest -m "not slow"

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_user_service.py
pytest tests/integration/test_auth_endpoints.py

# Run specific test function
pytest tests/unit/test_user_service.py::TestUserService::test_password_hashing_security
```

### Advanced Test Commands

```bash
# Run tests in parallel (faster)
pytest -n auto

# Generate HTML test report
pytest --html=tests/reports/report.html --self-contained-html

# Generate JSON test report  
pytest --json-report --json-report-file=tests/reports/report.json

# Run tests with different log levels
pytest --log-cli-level=DEBUG
pytest --log-cli-level=INFO

# Stop on first failure
pytest -x

# Show local variables in tracebacks
pytest --tb=long

# Run only failed tests from last run
pytest --lf
```

### Continuous Integration Commands

```bash
# CI/CD pipeline command (comprehensive)
pytest \
  --cov=src \
  --cov-report=html:tests/coverage_html \
  --cov-report=xml:tests/coverage.xml \
  --cov-report=term-missing \
  --html=tests/reports/report.html \
  --self-contained-html \
  --json-report --json-report-file=tests/reports/report.json \
  -v \
  --tb=short

# Security-focused test run
pytest -m security --tb=short -v

# Performance test run (includes slow tests)
pytest -m slow --tb=short -v
```

## Test Categories

### Unit Tests
- **Location**: `tests/unit/`
- **Purpose**: Test individual components in isolation
- **Coverage**: User service, authentication logic, password validation
- **Run**: `pytest -m unit`

### Integration Tests  
- **Location**: `tests/integration/`
- **Purpose**: Test API endpoints and component interactions
- **Coverage**: Authentication endpoints, chat API, assessment API, file uploads
- **Run**: `pytest -m integration`

### Security Tests
- **Location**: `tests/security/`
- **Purpose**: Test security mechanisms and attack prevention
- **Coverage**: JWT security, input validation, rate limiting, XSS/SQL injection prevention
- **Run**: `pytest -m security`

### HIPAA Compliance Tests
- **Location**: `tests/security/test_hipaa_compliance.py`  
- **Purpose**: Validate healthcare data protection requirements
- **Coverage**: PHI detection, secure logging, data encryption, access controls
- **Run**: `pytest -m hipaa`

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- Test discovery patterns
- Coverage reporting settings
- Logging configuration  
- Environment variables for testing
- Test markers and categories

### Fixtures (`conftest.py`)
- Test client configurations
- Mock services and data
- Authentication helpers
- Security test data
- Database test helpers

## Security Testing

### Attack Vector Testing
The security tests cover protection against:
- SQL Injection
- Cross-Site Scripting (XSS)
- Command Injection
- Path Traversal
- Rate Limit Bypass
- JWT Token Tampering
- Session Fixation

### Example Security Test Run
```bash
# Run all security tests
pytest tests/security/ -v

# Test specific attack vectors
pytest tests/security/test_security.py::TestInputValidationSecurity -v

# Test HIPAA compliance
pytest tests/security/test_hipaa_compliance.py -v
```

## HIPAA Compliance Testing

### Protected Health Information (PHI) Testing
- PHI detection in user inputs
- PHI scrubbing from logs
- PHI protection in API responses  
- Secure data storage validation

### Audit Trail Testing
- User action logging
- Authentication event auditing
- Data access tracking
- Log integrity validation

### Example HIPAA Test Run
```bash
# Run all HIPAA compliance tests
pytest -m hipaa -v

# Test PHI protection specifically
pytest tests/security/test_hipaa_compliance.py::TestPHIDetectionAndProtection -v

# Test audit trail compliance
pytest tests/security/test_hipaa_compliance.py::TestAuditTrailCompliance -v
```

## Mock Services

The test framework includes comprehensive mocking for:
- **User Service**: Authentication, registration, user management
- **Chat Agent**: Message processing, emotion detection
- **Diagnosis Agent**: Assessment processing, recommendations
- **Voice Module**: Audio transcription, speech synthesis
- **External APIs**: LLM services, third-party integrations

## Test Data Management

### Test Users
- Default admin user: `admin` / `SecureAdmin123!`
- Default demo user: `demo` / `DemoUser123!`
- Programmatically generated test users with various roles

### Mock Data
- Chat messages with various contexts
- Assessment responses for PHQ-9, GAD-7, Big Five
- Security payloads for attack testing
- PHI examples for compliance testing

## Coverage Requirements

### Coverage Targets
- **Overall Coverage**: > 85%
- **Critical Security Code**: > 95%
- **Authentication Logic**: > 95%  
- **API Endpoints**: > 90%

### Coverage Reports
```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# View coverage report
# Open tests/coverage_html/index.html in browser

# Coverage with missing lines
pytest --cov=src --cov-report=term-missing
```

## Debugging Tests

### Common Debug Commands
```bash
# Run single test with maximum verbosity
pytest tests/unit/test_user_service.py::TestUserService::test_authentication -vvv --tb=long

# Run with Python debugger on failure  
pytest --pdb

# Show test execution times
pytest --durations=10

# Capture and show print statements
pytest -s
```

### Test Logging
- Test logs are written to `tests/logs/`
- Log level can be controlled with `--log-cli-level`
- Use `caplog` fixture to test log messages

## Performance Testing

### Load Testing
Some tests include concurrent request testing:
```bash
# Run performance-focused tests
pytest -m slow -v

# Test with timing assertions
pytest tests/integration/test_api_endpoints.py::TestAPIErrorHandling::test_api_response_times
```

## Continuous Integration

### GitHub Actions / CI Pipeline
```yaml
# Example CI configuration
- name: Run Tests
  run: |
    pytest \
      --cov=src \
      --cov-report=xml \
      --html=tests/reports/report.html \
      --json-report --json-report-file=tests/reports/report.json \
      -v

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: tests/coverage.xml
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests before commit
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Add project root to Python path
   export PYTHONPATH="${PYTHONPATH}:/path/to/Solace-AI"
   ```

2. **Environment Variables**:
   ```bash
   # Ensure test environment variables are set
   export TESTING=true
   export JWT_SECRET_KEY="test_secret_key_for_pytest_only_not_for_production_use_minimum_32_chars"
   ```

3. **Database Connection**:
   ```bash
   # For database tests (when implemented)
   export TEST_DATABASE_URL="sqlite:///test.db"
   ```

4. **Rate Limiting in Tests**:
   - Some tests may fail due to rate limiting
   - Increase delays between requests or mock rate limiter

### Debug Failed Tests
```bash
# Show failed test details
pytest --tb=short

# Re-run only failed tests
pytest --lf

# Show test output
pytest -s -vv
```

## Contributing Tests

### Test Writing Guidelines
1. **Use descriptive test names** that explain what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert  
3. **Mock external dependencies** to ensure isolated testing
4. **Include security test cases** for all new features
5. **Test both positive and negative scenarios**
6. **Add HIPAA compliance tests** for healthcare data handling

### Test Review Checklist
- [ ] Tests are isolated and don't depend on external services
- [ ] Security implications are tested
- [ ] Error cases are covered
- [ ] Mock objects are used appropriately
- [ ] Tests are deterministic and repeatable
- [ ] HIPAA compliance is validated where applicable

## Test Maintenance

### Regular Tasks
- Review and update test data regularly
- Ensure mock services match real implementations
- Update security test payloads for new threats
- Validate HIPAA compliance with regulatory changes
- Monitor test execution times and optimize slow tests

### Test Quality Metrics
- Test coverage percentage
- Test execution time
- Number of flaky tests
- Security test effectiveness
- Compliance test coverage

---

For questions or issues with the testing framework, please refer to the project documentation or create an issue in the project repository.