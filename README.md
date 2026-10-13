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
2. In Azure, forward inbound TCP ports 80, 443, and 8443 to this VM and allow them in
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

## Freelancer annotation instance

The Compose stack includes an isolated Label Studio instance for freelancers:
`app-freelance`, `db-freelance`, and `ml-freelance`. It has no access to the
main instance database or `mydata/` directory.

1. Add a unique `FREELANCE_POSTGRES_PASSWORD` to `.env`. No extra DNS record
   is required; the instance uses the existing hostname on HTTPS port 8443.
2. Create `freelance-data/images/` and copy only the selected images into it.
   Do not use a symlink to `mydata/`. Compose prepares the directory permissions
   automatically before the freelancer app starts.
3. Start the new services with `docker compose up --build -d`.
4. Visit `https://ponzel-vision.westeurope.cloudapp.azure.com:8443`, create
   the administrator account, then create a project using the label configuration in
   `temaplate.xml`. Configure local-file storage at `/label-studio/data/images`
   or import tasks whose image URLs use `/data/local-files/?d=images/...`.
5. In that project, add the ML backend URL `http://ml-freelance:9090`. This
   backend reads only `freelance-data/` and shares the main YOLO weights as a
   read-only mount.

Freelancer accounts should be invited only to this second instance. The
dataset hub remains authenticated separately from both Label Studio instances.

## Dataset Hub instance selector

Dataset Hub can work with either Label Studio instance. Its sidebar has a
**Label Studio instance** selector that switches all frame extraction, dataset
export, and tag management actions between the main and freelancer projects.

Set `FREELANCE_LABEL_STUDIO_API_TOKEN` in `.env` to an API token created by the
freelancer-instance administrator. Set `FREELANCE_LABEL_STUDIO_PROJECT_ID` if
the freelancer project is not project `1`. Restart Dataset Hub after changing
these values:

```bash
docker compose up -d --build dataset-hub dataset-hub-proxy
```

## Storage archive transfer

The Dataset Hub **Storage archive** tab exports one or more Local Files
storages. Each storage folder in the ZIP contains source images, native Label
Studio annotations, COCO annotations, image tags, and JSON/CSV summary
statistics. By default the export excludes images without completed
annotations; select **Include all images, including images without
annotations** to include them.

Imports accept an archive containing exactly one storage. A `manifest.json` is
optional: without one, the ZIP must contain exactly one top-level folder with
`images/`, `annotations/label-studio.json`, and `tags/image-tags.json`, as in
an exported storage folder. Choose the archive and a new storage name in the
same tab. The name must not already exist, and all archive tags must already
be present in the active project's tag configuration. Dataset Hub extracts the
images to `mydata/images/<new-storage-name>/` directly (without an additional
`images/` subfolder), creates and syncs the Local Files storage, then imports
the native annotations and tags.

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