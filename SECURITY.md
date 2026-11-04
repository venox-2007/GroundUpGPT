# Security Policy

## Deployment Assumptions & Threat Model

GroundUpGPT is designed strictly as a **local-only educational and research prototype**. It is not hardened for multi-tenant production hosting or exposure to the public internet.

- **Local Execution**: The FastAPI backend (`127.0.0.1:8000`) and Next.js frontend (`localhost:3000`) are intended to bind to local loopback addresses only.
- **No Authentication / Multi-Tenancy**: The application does not provide user authentication, RBAC, session encryption, or per-user access control. Any user with local access has full access to the inference backend and local SQLite database.
- **Network Boundaries**: Do not expose GroundUpGPT to untrusted networks or bind it to `0.0.0.0` without an upstream authenticating reverse proxy and network firewall.

## Security Controls Implemented

While designed for local research, several safeguards protect local execution integrity:
1. **Safe Arithmetic Execution**: Arithmetic queries are evaluated using a strict AST-based parser (`src/safe_calculator.py`) that strictly permits basic mathematical operators, completely disallowing `eval()` or arbitrary code execution.
2. **Path Sanitization**: Knowledge session file handlers sanitize filenames and prevent directory traversal outside session storage bounds.
3. **Local-Only Document Extraction**: File parsing (`.txt`, `.md`, `.pdf`, `.docx`) executes entirely within the local process with zero telemetry, remote network calls, or third-party cloud services.
4. **Opt-in Storage**: SQLite chat persistence is disabled by default to minimize unmanaged local data writes.

## Reporting a Security Vulnerability

If you discover a security issue or boundary bypass:
- Please report it by opening a private report via **GitHub Security Advisories** on the `venox-2007/GroundUpGPT` repository.
- Alternatively, open an issue labeled `security` on the repository issue tracker detailing the vulnerability, reproduction steps, and potential impact.
- Please do not exploit vulnerabilities or perform security tests on shared infrastructure.
