# Python Code Standards

- Follow PEP 8 for all generated Python code.
- Prefer readable, explicit, maintainable code over compact or clever code.
- Add docstrings to every new public module, class, function, and method.
- Use type hints for all new public functions and methods where practical.
- Reuse existing project patterns and abstractions before creating new ones.
- Keep functions and classes focused on a single responsibility.
- Avoid duplicate logic; refactor only when reuse is real and justified.
- Do not use print() for runtime diagnostics; use the logging module instead.
- Use logging only when operationally useful:
  - DEBUG for diagnostic state that helps troubleshooting.
  - INFO for meaningful lifecycle events.
  - WARNING for recoverable anomalies.
  - ERROR or EXCEPTION for failures.
- Do not add noisy logs in trivial helpers or tight loops.
- Never log secrets, credentials, tokens, or sensitive payloads.
