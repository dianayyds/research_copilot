# Server Deployment

This repository includes the application runtime and its supporting services in
Docker Compose. A fresh Ubuntu server only needs Git, Docker Engine, and the
Docker Compose plugin on the host.

## 1. Clone the Repository

```bash
git clone https://github.com/dianayyds/research_copilot.git
cd research_copilot
```

## 2. Install Docker on Ubuntu

The installer follows Docker's official Ubuntu apt-repository method and
installs Docker Engine, Buildx, and the Compose plugin:

```bash
bash scripts/install_server_environment.sh
```

The installation step requires root privileges through `sudo`. After the
script adds the current account to the `docker` group, log out and back in
before continuing.

If Docker is already installed, the script only prints the installed versions.
For non-Ubuntu hosts, follow the matching Docker Engine instructions at
<https://docs.docker.com/engine/install/>.

## 3. Configure Secrets

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env` and set at least:

```dotenv
LLM_API_KEY=your_deepseek_api_key
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token
```

Change the default MySQL and MinIO passwords before exposing the deployment to
other machines. `.env` is ignored by Git and must never be committed.

## 4. Build and Start

```bash
bash scripts/deploy_server.sh
```

The script validates the Compose configuration, builds the FastAPI image, and
starts MySQL, Redis, MinIO, Qdrant, and the runtime API.

Check the deployment:

```bash
docker compose ps
docker compose logs -f runtime-api
curl http://127.0.0.1:8001/healthz
```

The browser workspace is available at `http://SERVER_IP:8001/`. Ensure the
server firewall or cloud security group only exposes the ports that are
actually required.

## 5. Model Storage and Hardware

The local BGE-M3 embedding model and BGE reranker are downloaded on first use
and stored under `./models`, which is mounted into the runtime container.
Reserve several GB of disk space. The default Compose environment uses CPU;
GPU deployment requires an NVIDIA container runtime and corresponding device
configuration.

Persistent application data lives in named Docker volumes:

- `mysql_data`
- `redis_data`
- `minio_data`
- `qdrant_data`

Do not run `docker compose down -v` unless you intend to delete that data.

## 6. Update the Server Checkout

Commit server-side code changes before updating, then run:

```bash
git pull --ff-only
bash scripts/deploy_server.sh
```

The rebuild preserves named volumes and the downloaded model directory.
