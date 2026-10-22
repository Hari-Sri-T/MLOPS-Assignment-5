# Image Classification on AWS

Cats vs dogs classification with MobileNetV2, Flask, DVC, S3, Docker,
GitHub Actions, GitHub Container Registry (GHCR), and EC2.

- Repository: https://github.com/Hari-Sri-T/MLOPS-Assignment-5
- Application target: http://13.202.231.26/
- Readiness: http://13.202.231.26/health
- Detailed implementation and evidence: [project report](docs/report.md).

The EC2 address may change when the instance is stopped. See the report for
actual verification status and the deployment date.

```text
dvc repro: download → train → evaluate
                    ↓ dvc push
              private AWS S3 cache
                    ↓ dvc pull train
GitHub Actions: test → Docker build/push to GHCR → SSH deployment
                                                    ↓
                         EC2 Nginx :80 → Docker 127.0.0.1:8080
```

## Run and reproduce

Use Python 3.11, matching CI and Docker.

```bash
git clone https://github.com/Hari-Sri-T/MLOPS-Assignment-5.git
cd MLOPS-Assignment-5
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -v
flake8 app.py src tests --max-line-length 110
dvc repro
dvc dag
dvc metrics show
dvc plots show
python app.py
```

Open http://localhost:8080/. Training downloads 2,000 training and 1,000
validation images plus ImageNet weights. The old `mledu-datasets` URL returns
403; `params.yaml` uses the working `tensorflow-1-public/course2` archive.
`pathspec` is pinned because DVC 3.59.1 is incompatible with its 1.x APIs.

Commit `dvc.lock` and `metrics/` with the code. DVC versions `data/raw/` and
`artifacts/model/`. Evaluation uses the validation split seen during training;
these scores are not from an independent test set.

## DVC remote: private S3

The existing bucket is in **ap-southeast-2 (Sydney)**; EC2 is in **ap-south-1
(Mumbai)**. `.dvc/config` points to `s3://hari-dvc-assignment/dvc-store`.
Keep all S3 Block Public Access settings on. Configure AWS credentials outside
the repository using your AWS profile or environment variables, then:

```bash
dvc push
# Fresh clone, with S3 read access:
dvc pull train
```

For another bucket, create a private S3 bucket and configure it:

```bash
dvc remote modify s3remote url s3://YOUR-BUCKET/dvc-store
dvc remote modify s3remote region YOUR-BUCKET-REGION
```

Training needs `s3:ListBucket` on the bucket and `s3:GetObject` / `s3:PutObject`
on `dvc-store/*`. CI needs only list/read. Never commit credentials or keys.

## Docker

Pull or reproduce the model before building:

```bash
dvc pull train
docker build --build-arg APP_VERSION=manual -t image-classifier:manual .
docker run -d --name image-classifier --restart unless-stopped -p 127.0.0.1:8080:8080 image-classifier:manual
curl --fail http://127.0.0.1:8080/health
curl --fail -F file=@photo.jpg http://127.0.0.1:8080/predict
```

The image runs as an unprivileged user with one Gunicorn worker and limited
TensorFlow threads. `/health` returns 503 until the model and class metadata
load successfully. Uploads are limited to 8 MB.

## EC2 setup

This deployment uses existing Amazon Linux 2023 instance `i-06c275d603a1b3688`,
user `ec2-user`, key pair `btech`. For a fresh instance, use x86_64 Amazon Linux
2023 or Ubuntu, associate your key pair, and allow HTTP 80 and SSH from the
admin/deployment network. GitHub-hosted runners need an allowed egress range;
a runner with a fixed IP is easier to restrict. Check your account's AWS costs.

Copy the project scripts to the instance and run:

```bash
bash scripts/ec2_setup.sh
# Log out and reconnect for docker group membership.
sudo install -m 644 scripts/nginx.conf /etc/nginx/conf.d/image-classifier.conf
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx
```

If the address changes, update `server_name` in `scripts/nginx.conf` and the
`EC2_HOST` GitHub secret. The included host routes the EC2 public IP to the
container alongside existing domain-based Nginx hosts.

## GitHub Actions CI/CD

The workflow is `.github/workflows/ci-cd.yaml`. Pull requests run lint,
unit tests, and shell syntax checks. Pushes to `main` additionally pull the
S3 model, build/push full-commit and `latest` tags to
`ghcr.io/hari-sri-t/mlops-assignment-5`, verify container inference, and deploy.
The built-in `GITHUB_TOKEN` authenticates GHCR; Docker Hub is not required.

Add these **repository secrets** under Settings → Secrets and variables → Actions:

| Secret | Value |
| --- | --- |
| `AWS_ACCESS_KEY_ID` | S3 reader's access key |
| `AWS_SECRET_ACCESS_KEY` | Matching secret access key |
| `EC2_HOST` | `13.202.231.26` |
| `EC2_USER` | `ec2-user` |
| `EC2_SSH_KEY` | Private `btech.pem` contents, entered only in the secret store |
| `EC2_SSH_FINGERPRINT` | ED25519 host fingerprint verified through a trusted instance connection |

The optional variable `AWS_REGION` defaults to `ap-southeast-2` (S3 region).
On EC2, obtain the host fingerprint with
`sudo ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`.
`scripts/deploy.sh` uses a temporary GHCR login, pulls the commit tag, retains
the previous container until readiness succeeds, and restores it on failure.
The public check verifies readiness and the exact deployed commit.
A single-instance update has brief downtime.

## Troubleshooting

| Problem | Solution |
| --- | --- |
| DVC `_DIR_MARK` import failure | Install `pathspec==0.12.1`. |
| Old dataset URL returns 403 | Use the URL in `params.yaml`. |
| S3 access denied | Check region, IAM permissions, and Actions secrets. |
| Model missing at Docker build | Run `dvc pull train` or `dvc repro`; commit `dvc.lock`. |
| Health returns 503 | Check container logs and model/class metadata. |
| SSH key rejected | Use `chmod 400 btech.pem` and the matching user/key pair. |
| EC2 low memory | Run swap setup; train locally and keep one inference worker. |
| Port 80 conflict | Use localhost:8080 and the supplied Nginx host. |
