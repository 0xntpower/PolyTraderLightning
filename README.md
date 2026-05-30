# PolyTraderLightning

High-performance trading bot for Polymarket's BTC 5-minute Up/Down markets. Receives signals from the SignalOrchestrator via IPC and executes trades on the Polymarket CLOB.

## Deployment

Runs on a dedicated low-cost VPS in Europe (close to Polymarket's CLOB) for low-latency execution (~23ms to the CLOB). The lab machine runs the collector, engine, and orchestrator; signals are delivered over a Tailscale mesh VPN. On the VPS the bot is launched and supervised by a per-host PSLAgent.

Deployment details live in the PolySignalLab parent repo (`docs/infrastructure/vps_deployment.md`).

## Key Features

- **4 WebSocket feeds** — Binance, Chainlink, Polymarket CLOB (market + user streams)
- **250ms strategy tick** — evaluates signal entry conditions every quarter second
- **Kelly Criterion bet sizing** — quarter-Kelly with regime-adjusted win probabilities
- **SPRT decay detection** — detects signal degradation and enters shadow tracking mode
- **Regime adjustments** — volatility, chop, and outcome-bias discounts on win probability, combined via the v3.2 **soft-OR** rule (`kelly_regime_cap_2_axes = 0.20`, `kelly_regime_cap_3_axes = 0.30`) plus a hostile-regime gate that halves Kelly at 0.15 aggregate hostility and skips the trade entirely at 0.25
- **Post-fire CUSUM erosion exit** — early exit if price action moves too hard against the open position
- **Maker-first execution (CLOB V2 SDK)** — posts a post-only maker quote with a taker fallback after `maker_timeout_s`; high-confidence fires cross the spread
- **Paper and live modes** — full simulation with identical logic, or real CLOB execution

## Configuration

`config.yml` (gitignored, auto-generated on first run) — the `DEFAULT_CONFIG` template and typed dataclasses in `src/config.py` define every parameter and its default. The PolySignalLab parent repo also has a prose `docs/reference/configuration_guide.md`.

## Shared Library

The `shared/` directory contains modules used across the entire PolySignalLab system: IPC protocol, Discord notifications, signal ranking, risk management, decay detection, and configuration utilities.

## License

Source-available under the **PolySignalLab Source-Available License v1.0** — see [LICENSE](LICENSE). Commercial use is permitted, but if you use a modified version you must disclose your modifications to the author (privately is fine — public release is not required). See the LICENSE for the exact terms.
