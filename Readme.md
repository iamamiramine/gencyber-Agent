<br />
<div align="center">
  <a href="https://github.com/iamamiramine/generative-cybersecurity">
    <img src="shared/assets/Logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Generative Cybersecurity</h3>

</div>

## Docker Compose

This repo includes `docker-compose.yaml` for building/running the LangGraph API container.

Prereqs:

- Create the shared docker network once (same name as other `gencyber-*` repos): `generative-cybersecurity-network`
- Provide required secrets via environment variables (OpenRouter API key, etc.). From the monorepo root you can run with the root `.env`.

Typical split-repo startup order:

1. `gencyber-ETL` (MongoDB)
2. `gencyber-workbench` (challenge API + terminal-session on the shared volume)
3. `gencyber-Agent` (this repo) — set `TERMINAL_SESSION_URL` to the workbench terminal (port **3000**)
4. `gencyber-Frontend` — set `CHALLENGE_TOOLKIT_URL` to the workbench API (port **80** inside the container, e.g. host **8080** when mapped)

```bash
docker compose up -d --build
```