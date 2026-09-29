# Security policy

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/humandebri/IC-Laya/security/advisories/new) to report a suspected vulnerability. Include the affected commit or release, the reachable API or input, reproduction steps, and the expected impact. Do not include private keys, seed phrases, or real user data.

Use public issues for ordinary bugs and feature requests. Keep exploit details private while a security report is being assessed.

## Project scope

IC-Laya includes Rust inference and model loading, tokenizer integration, canister APIs, and a mock workflow/executor. Reports may concern authorization bypass, malformed model or input handling, resource exhaustion, state integrity across retries or upgrades, or exposure of secrets. Report the concrete behavior and the caller privileges required to reach it.

The project is experimental. Raw inference and model-management APIs are owner-restricted, real-fund dispatch is disabled, and model decision quality is not a security guarantee. These limitations do not exempt vulnerabilities in reachable code from reporting. Include the exact revision tested; historical benchmark results do not establish the security of the current revision.
