# Label Studio deployment

This repository contains deployment configuration and custom services for Label
Studio. Runtime data, database files, models, and credentials are intentionally
excluded from Git.

## Version

Label Studio is pinned to `heartexlabs/label-studio:1.23.0` in
`docker-compose.yml`. Do not use `latest` for a deployment that accesses the
database directly.

## Deploy on a new server

1. Install Docker Engine and Docker Compose v2.
2. Clone this repository and enter its directory.
3. Create the deployment environment file:

   ```bash
   cp .env.example .env
   openssl rand -base64 32
   ```

   Put the generated value into `POSTGRES_PASSWORD`. Set the proxy credentials
   and, if `dataset-hub` needs the Label Studio API, set
   `LABEL_STUDIO_API_TOKEN`.

4. Restore `mydata/`, `postgres-data/`, and `models/` from a backup when
   migrating an existing deployment. They are required to preserve users,
   projects, annotations, uploaded files, and ML weights.
5. Start the services:

   ```bash
   docker compose up --build -d
   docker compose ps
   ```

## HTTPS deployment

The public TLS endpoint is configured for
`ponzel-vision.westeurope.cloudapp.azure.com`. Before starting the updated stack:

1. Ensure the Azure DNS hostname resolves to the Azure load-balancer public IP.
2. In Azure, forward inbound TCP ports 80 and 443 to this VM and allow them in
   the VM's network security group. Remove public forwarding for ports 8080,
   8501, and 9090.
3. Restrict the credentials file to the deployment account:

   ```bash
   chmod 600 .env
   ```

4. Start the updated stack:

   ```bash
   docker compose up --build -d
   docker compose ps
   ```

Caddy automatically obtains and renews the TLS certificate. Label Studio is
available at `https://ponzel-vision.westeurope.cloudapp.azure.com`; the
authenticated dataset hub is at
`https://ponzel-vision.westeurope.cloudapp.azure.com/dataset-hub/`. Only Caddy
publishes host ports.

## Back up before upgrades or database changes

Back up PostgreSQL with a logical dump and archive the persistent files before
running schema-altering SQL or changing the Label Studio image version:

```bash
docker compose exec -T db pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" > label-studio-postgres.sql
tar -czf label-studio-files.tar.gz mydata models
```

Keep the SQL dump and archive outside this repository. Test a restore on another
server before relying on the backup.

## GitHub

Initialize and publish only after checking that no secrets appear in the staged
files:

```bash
git init
git add .
git diff --cached --check
git diff --cached
git commit -m "Add reproducible Label Studio deployment"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Create a private GitHub repository if the custom application code or deployment
details are not intended to be public. Never commit `.env`, `mydata/`,
`postgres-data/`, or `models/`.