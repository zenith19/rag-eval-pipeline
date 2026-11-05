# 018 — The account was fine; the grant was half a grant

*After v1.0 — 2026-11-05. Corrects 009, which is left standing.*

## Observation

009's central finding was that **Lambda resource-based policies are not enforced on this account**.
It explained why a public Function URL and CloudFront with Origin Access Control both returned 403,
and it spread: into the README, `CLAUDE.md`, `infra/DEPLOY.md`, comments in `infra/main.tf`, and
014, which declined to assume a KMS key policy would work "since this account had already shown one
class of resource policy going unenforced".

The case raised with AWS Support at the time was answered on 2026-10-21, after v1.0 was tagged. The
cause was not the account. Since October 2025 a Function URL requires **two** statements in the
function's resource-based policy:

1. `lambda:InvokeFunctionUrl`
2. `lambda:InvokeFunction` (for a URL-only grant, with the `lambda:InvokedViaFunctionUrl`
   condition, `--invoked-via-function-url` on the CLI)

With only the first, Lambda returns 403, whatever the auth type. The Lambda console and SAM add both;
Terraform, CloudFormation, the CLI and the API add exactly what they are told.
[Documented with example policies](https://docs.aws.amazon.com/lambda/latest/dg/urls-auth.html).

## Checking the explanation against what was actually done

**What was granted.** The history has the grant, in `70635af` and again in `d9d15da`:

```hcl
resource "aws_lambda_permission" "public_url" {
  action                 = "lambda:InvokeFunctionUrl"
  principal              = "*"
  function_url_auth_type = "NONE"
}
```

That is the first statement and not the second. The other three tests in 009 were run by hand with
`lambda:AddPermission` and are not in the history, so the action they granted is not recorded. They
copied this grant, and nothing suggests a second statement was ever added. That is an inference,
and is marked as one below.

**Every row of 009's table fits the explanation:**

| 009's test | Result | Under the two-statement rule |
|---|---|---|
| Identity policy (the user's own IAM policy) | 200 | `infra/deploy-policy.json` grants both `lambda:InvokeFunction` and `lambda:InvokeFunctionUrl`; for a same-account principal, an identity policy with both is enough |
| Resource policy → a named IAM user | 403 | `InvokeFunctionUrl` only (inferred) |
| Resource policy → `Principal: "*"`, `AuthType: NONE` | 403 | `InvokeFunctionUrl` only (committed, above) |
| Resource policy → `cloudfront.amazonaws.com` (OAC) | 403 | `InvokeFunctionUrl` only (inferred) |

**And 009 contained its own counter-example.** Row 2 of its failure table is
`CreateFunction: "Lambda does not have permission to access the ECR image"`, fixed by
`aws_ecr_repository_policy.lambda_pull`. That is a resource-based policy, on this account, being
enforced: Lambda has pulled every image since through it. 014 then found that the AWS-managed
`aws/ssm` key's policy worked as well. "Resource policies are not enforced on this account" was
contradicted by evidence the project already had.

## What went wrong in 009

009 is right that the question needed an experiment rather than more reading. The experiment
varied the **principal** and the **grant mechanism**, and held the **action** fixed. The true
cause lay in the one dimension it never varied, so it could only choose between wrong answers.
The finding was then generalised from "Function URL grants made this way" to "resource policies on
this account", which is a much larger claim than four requests support.

The fix was not an experiment either: it was the current documentation, which had changed. The
lesson recorded in 009, about ordering the discriminating experiment before building, still
stands. This one is narrower: when a mechanism "is not enforced", first check that the grant was
complete, against the documentation as it reads today.

## Decision

- **Keep API Gateway with an integration role.** It works, it is deployed, and it is a sound
  design on its own terms: authorisation through an identity policy, no public grant on the
  function. Nothing in `infra/` changes; only comments do.
- **Correct every place the claim was stated**: README, `CLAUDE.md`, `infra/DEPLOY.md` and the
  comments in `infra/main.tf`. 009 and 014 stay as written, because a decisions directory that
  quietly rewrites itself records nothing. This record is the correction.
- **If a Function URL is ever added, grant both statements**, and verify with a principal that
  has no identity grant of its own, or the test passes for the wrong reason, exactly as 009's
  identity row did.

## Trade-off

API Gateway was chosen because the Function URL was believed impossible, and it came with a cost
that has shaped the project since: **the 30s HTTP API integration ceiling**. That ceiling is why
the EventBridge warm schedule exists, why 017's cold-start fix mattered as much as it did, and why
a second simultaneous visitor gets a retrying page. A Function URL is bounded only by the
function's own timeout (90s), which leaves room for a cold start. Switching is now a real option,
not an impossible one. It is in the README backlog rather than done here: it changes the public
endpoint, needs a deploy and a smoke test, and v1.0 is frozen.

## Measured validation

Partial, and stated as such:

- **History:** the only Function URL grant ever committed is `InvokeFunctionUrl` alone (above);
  the hand-run grants are inferred to match it.
- **AWS Support's diagnosis** matches all four rows of 009's measurements.
- **009's own row 2** and 014's key-policy result show resource policies being enforced on this
  account.
- **Not yet re-run:** the reproduction with both statements. The cheap version: a URL with
  `AuthType: AWS_IAM`, `bedrock-dev` as the principal (it holds no Lambda permissions, so only the
  resource policy can let it in), a signed `GET /health` with one statement, then with both,
  expecting 403 then 200. Remove the URL afterwards. This record should be amended with the result.
