# PolyTraderLightning

High-performance trading bot for Polymarket's BTC 5-minute Up/Down markets. Receives signals from the SignalOrchestrator via IPC and executes trades on the Polymarket CLOB.

## Deployment

Runs on a dedicated VPS (AWS Dublin) for low-latency execution (~23ms to Polymarket CLOB). The lab machine runs the collector, engine, and orchestrator; signals are delivered over a Tailscale mesh VPN.

See `docs/infrastructure/vps_deployment.md` for full deployment details.

## Key Features

- **4 WebSocket feeds** — Binance, Chainlink, Polymarket CLOB (market + user streams)
- **250ms strategy tick** — evaluates signal entry conditions every quarter second
- **Kelly Criterion bet sizing** — quarter-Kelly with regime-adjusted win probabilities
- **SPRT decay detection** — detects signal degradation and enters shadow tracking mode
- **Regime adjustments** — volatility, chop, and outcome-bias discounts on win probability, combined via the v3.2 **soft-OR** rule (`kelly_regime_cap_2_axes = 0.20`, `kelly_regime_cap_3_axes = 0.30`) plus a hostile-regime gate that halves Kelly at 0.15 aggregate hostility and skips the trade entirely at 0.25
- **Post-fire CUSUM erosion exit** — early exit if price action moves too hard against the open position
- **Paper and live modes** — full simulation with identical logic, or real CLOB execution

## Configuration

`config.yml` (gitignored) — see `docs/reference/configuration_guide.md` for all parameters.

## Shared Library

The `shared/` directory contains modules used across the entire PolySignalLab system: IPC protocol, Discord notifications, signal ranking, risk management, decay detection, and configuration utilities.
