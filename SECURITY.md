# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: [security@yourdomain.com]

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Security Considerations for Users

### API Keys and Secrets

- **Never commit** `.env.local` or files containing real API keys
- Use environment variables for all secrets
- Rotate keys if you suspect they've been exposed
- The `.gitignore` already excludes `.env`, `.env.local`, and credential files

### Agent Execution

Task Orchestrator spawns AI agents that can execute code. Consider:

- **Human-in-the-loop**: Use the approval queue for sensitive operations
- **Operation classification**: Configure blocked operations appropriately
- **Budget limits**: Set cost limits to prevent runaway API usage
- **Immune system**: Enable pre-spawn checks to catch risky prompts

### Network Security

- The MCP server runs locally by default
- If exposing the API server, use HTTPS and proper authentication
- Validate JWT tokens for all authenticated endpoints

## Security Features

Task Orchestrator includes several security features:

1. **Operation Classifier**: Categorizes operations as SAFE, REQUIRES_APPROVAL, or BLOCKED
2. **Human-in-the-Loop**: Approval queue for sensitive operations
3. **Immune System**: Detects and prevents risky prompt patterns
4. **Budget Controls**: Prevents excessive API costs
5. **Circuit Breakers**: Automatic recovery from failures
6. **Audit Logging**: Track all agent operations

## Acknowledgments

We appreciate security researchers who help keep Task Orchestrator secure. Contributors who report valid vulnerabilities will be acknowledged (with permission) in our release notes.
