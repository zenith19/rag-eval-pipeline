# Deploying

Two identities, on purpose.

| Identity | Holds | Used for |
|---|---|---|
| `bedrock-dev` | `AmazonBedrockFullAccess`, `AmazonS3ReadOnlyAccess` | local development and the Lambda's own runtime calls |
| `rag-deploy` | `infra/deploy-policy.json` | `terraform apply`, and nothing else |

The split is the point. The runtime identity can invoke one model and read one bucket; it cannot
create a role, push an image, or reshape the infrastructure it runs on. Giving the serving identity
deploy rights would be one leaked key away from an attacker who can rewrite the function rather
than merely call it. See `docs/decisions/007`.

`infra/deploy-policy.json` is scoped to the exact resource names in `main.tf` (`locals.name =
"rag-eval-pipeline"`) — rename there and the policy needs the same edit, or `apply` starts failing
with `AccessDenied` on the renamed resource. Account segments are wildcarded (`arn:aws:lambda:eu-central-1:*:function:...`)
because no account ID belongs in this repo; the policy is attached inside a single account, so the
wildcard does not widen it.

## One-time setup

1. **Create the user.** IAM → Users → Create user → `rag-deploy`. No console access needed.
2. **Attach the policy.** Create inline policy → JSON tab → paste `infra/deploy-policy.json`
   verbatim. IAM accepts only `Version`, `Id` and `Statement` at the top level, so do not add
   comments to it.
3. **Create an access key** for `rag-deploy` and add it as a named profile:

   ```ini
   # ~/.aws/credentials
   [rag-deploy]
   aws_access_key_id = ...
   aws_secret_access_key = ...
   ```

   Prefer a key you can delete afterwards; this identity exists to be used from one machine.
4. **Confirm the split holds:**

   ```bash
   AWS_PROFILE=rag-deploy .venv/bin/python -c \
     "import boto3; print(boto3.client('sts').get_caller_identity()['Arn'].split('/')[-1])"
   # expect: rag-deploy
   ```

## Deploying

```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars   # set alert_email
export AWS_PROFILE=rag-deploy
TAG=$(git rev-parse --short HEAD)        # a running deployment traceable to a commit

terraform -chdir=infra init
terraform -chdir=infra plan      # ECR repo first: the image must exist before the function does
terraform -chdir=infra apply -target=aws_ecr_repository.app

# build and push (~2.4 GB; the build re-embeds the corpus and takes ~9 min)
docker build -t rag-eval-pipeline .
REPO=$(terraform -chdir=infra output -raw ecr_repository_url)
.venv/bin/python -m awscli ecr get-login-password --region eu-central-1 \
  | docker login --username AWS --password-stdin "${REPO%%/*}"
docker tag rag-eval-pipeline "$REPO:$TAG"
docker push "$REPO:$TAG"

# image_tag defaults to "latest"; pass the real tag so the deploy names a commit
terraform -chdir=infra apply -var "image_tag=$TAG"
curl -s "$(terraform -chdir=infra output -raw function_url)health"
```

`python -m awscli`, not `aws`: the console script carries a shebang pointing at an interpreter path
containing a space, so the kernel splits it and the script cannot execute at all. The `-m` form has
no shebang. Same reason `uvicorn`, `pytest` and `ruff` are all invoked that way in this project.

The two-phase apply is not ceremony: `aws_lambda_function` refers to
`"${aws_ecr_repository.app.repository_url}:${var.image_tag}"`, and Lambda validates that the image
exists at create time. A single `apply` on an empty account therefore fails partway through, having
already created the repository — recoverable, but noisier than doing it in the order the dependency
actually requires.

## Tearing down

```bash
AWS_PROFILE=rag-deploy terraform -chdir=infra destroy
```

ECR holds images Terraform did not create, so `destroy` can fail on a non-empty repository. Delete
the images first (`aws ecr batch-delete-image`) or accept the repository lingering at roughly
€0.24/month per 2.4 GB image.

## Notes

- `aws` as a console script does not work in this checkout — the path contains a space and a
  shebang cannot express it. Use `.venv/bin/python -m awscli` if `aws` is not on PATH system-wide.
- The Function URL is `authorization_type = "NONE"`, i.e. public. That is deliberate — a link
  nobody can open proves nothing — and is why `max_concurrency`, `rate_limit_per_min` and the
  budget alarm exist. Check the budget alarm is real before leaving the URL in a CV.
