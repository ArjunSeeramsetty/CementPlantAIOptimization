#!/usr/bin/env python3
"""
manage_service.py - ON/OFF Switch and Status Manager for Cement Plant AI Digital Twin on GCP.
This script provides an administrative utility to control the public availability of the Cloud Run
services, allowing you to completely stop incoming traffic (and thus incur zero active compute costs)
and start the service back up when needed.
"""

import sys
import subprocess
import argparse
import json

# Configuration Defaults (can be overridden via CLI arguments)
DEFAULT_PROJECT_ID = "cement-ai-optimization"
DEFAULT_REGION = "us-central1"
SERVICE_NAMES = ["cement-plant-digital-twin", "cement-digital-twin"]


# Color printing support
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def log(msg, color=None):
    if color:
        print(f"{color}{msg}{Colors.ENDC}")
    else:
        print(msg)


def run_command(cmd):
    """Executes a shell command and returns code, stdout, stderr."""
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            shell=True,
        )
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except FileNotFoundError:
        return -1, "", "Command not found. Is gcloud CLI installed and on your PATH?"


def check_gcloud_auth():
    """Verifies that the gcloud CLI is authenticated and functional."""
    code, stdout, stderr = run_command(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"]
    )
    if code != 0 or not stdout:
        log(
            "❌ gcloud CLI is not authenticated or not installed properly.", Colors.FAIL
        )
        log(
            "Please run 'gcloud auth login' or ensure your service account key is active.",
            Colors.WARNING,
        )
        return False
    log(f"🔑 Authenticated GCP Account: {stdout}", Colors.OKGREEN)
    return True


def get_existing_services(project, region):
    """Detects which of the configured service names exist in the specified project & region."""
    existing = []
    for service in SERVICE_NAMES:
        code, stdout, stderr = run_command(
            [
                "gcloud",
                "run",
                "services",
                "describe",
                service,
                "--region",
                region,
                "--project",
                project,
                "--format=value(metadata.name)",
            ]
        )
        if code == 0 and stdout == service:
            existing.append(service)
    return existing


def stop_service(service, project, region):
    """Turns OFF the service by revoking public access (allUsers invoker)."""
    log(f"\n⚙️ Stopping service: {service}...", Colors.OKBLUE)

    # Check current IAM policy
    cmd = [
        "gcloud",
        "run",
        "services",
        "remove-iam-policy-binding",
        service,
        "--region",
        region,
        "--project",
        project,
        "--member",
        "allUsers",
        "--role",
        "roles/run.invoker",
    ]

    # We run the command
    code, stdout, stderr = run_command(cmd)

    # The command might fail if the binding doesn't exist, which is fine (it means it is already stopped)
    if code == 0:
        log(f"✅ Service public access revoked successfully.", Colors.OKGREEN)
        log(
            "🔌 Service status is now: OFF (No public traffic can trigger container spin-ups)",
            Colors.OKGREEN,
        )
    elif (
        "not found" in stderr
        or "policy does not contain the binding" in stderr
        or "Policy binding not found" in stderr
    ):
        log(
            f"ℹ️ Public access binding not found. Service was already stopped/OFF.",
            Colors.OKBLUE,
        )
    else:
        log(f"❌ Failed to revoke public access to {service}.", Colors.FAIL)
        log(f"Error details: {stderr}", Colors.WARNING)
        return False
    return True


def start_service(service, project, region):
    """Turns ON the service by granting public access (allUsers invoker)."""
    log(f"\n⚙️ Starting service: {service}...", Colors.OKBLUE)

    cmd = [
        "gcloud",
        "run",
        "services",
        "add-iam-policy-binding",
        service,
        "--region",
        region,
        "--project",
        project,
        "--member",
        "allUsers",
        "--role",
        "roles/run.invoker",
    ]

    code, stdout, stderr = run_command(cmd)
    if code == 0:
        log(f"✅ Service public access granted successfully.", Colors.OKGREEN)

        # Fetch the URL
        url_code, url_out, _ = run_command(
            [
                "gcloud",
                "run",
                "services",
                "describe",
                service,
                "--region",
                region,
                "--project",
                project,
                "--format=value(status.url)",
            ]
        )
        url = url_out if url_code == 0 else "Unknown URL"

        log(f"⚡ Service status is now: ON (Publicly accessible)", Colors.OKGREEN)
        log(f"🌐 Access URL: {url}", Colors.OKGREEN)
    else:
        log(f"❌ Failed to grant public access to {service}.", Colors.FAIL)
        log(f"Error details: {stderr}", Colors.WARNING)
        return False
    return True


def get_service_status(service, project, region):
    """Queries and displays the status of a specific service."""
    log(f"\n🔍 Querying status for: {service}...", Colors.BOLD)

    # Get general details
    code, stdout, stderr = run_command(
        [
            "gcloud",
            "run",
            "services",
            "describe",
            service,
            "--region",
            region,
            "--project",
            project,
            "--format=json(status.url,status.conditions,spec.template.spec.containers)",
        ]
    )

    if code != 0:
        log(f"❌ Could not describe service {service}. Is it deployed?", Colors.FAIL)
        log(f"Error: {stderr}", Colors.WARNING)
        return

    try:
        data = json.loads(stdout)
        url = data.get("status", {}).get("url", "N/A")

        # Check active instances or scale setting
        min_instances = "unknown"
        containers = (
            data.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        # Sometimes min-instances is in metadata annotations
        annot_code, annot_stdout, _ = run_command(
            [
                "gcloud",
                "run",
                "services",
                "describe",
                service,
                "--region",
                region,
                "--project",
                project,
                '--format=value(spec.template.metadata.annotations."autoscaling.knative.dev/minScale")',
            ]
        )
        min_instances = annot_stdout if annot_code == 0 and annot_stdout else "0"

        # Check IAM policy to see if public access (run.invoker) is granted
        iam_code, iam_stdout, iam_stderr = run_command(
            [
                "gcloud",
                "run",
                "services",
                "get-iam-policy",
                service,
                "--region",
                region,
                "--project",
                project,
                "--format=json",
            ]
        )

        is_public = False
        if iam_code == 0:
            iam_data = json.loads(iam_stdout)
            bindings = iam_data.get("bindings", [])
            for binding in bindings:
                if binding.get("role") == "roles/run.invoker":
                    if "allUsers" in binding.get("members", []):
                        is_public = True
                        break

        status_text = (
            "ON (Publicly Accessible)"
            if is_public
            else "OFF (Stopped / Private Access Only)"
        )
        status_color = Colors.OKGREEN if is_public else Colors.WARNING

        log(f"🔹 Service Name:    {service}")
        log(f"🔹 Status:          {status_text}", status_color)
        log(f"🔹 Service URL:      {url}")
        log(
            f"🔹 Min Instances:    {min_instances} (Configured to scale to 0 when idle)"
            if min_instances == "0"
            else f"🔹 Min Instances:    {min_instances} (Warning: Non-zero idle costs apply)"
        )

    except Exception as e:
        log(f"❌ Failed to parse service status details: {e}", Colors.FAIL)


def main():
    parser = argparse.ArgumentParser(
        description="Manage Cement Plant AI Digital Twin Cloud Run Service status."
    )
    parser.add_argument(
        "action",
        choices=["start", "stop", "status"],
        help="Action to perform: start (ON), stop (OFF), or status (Check)",
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT_ID, help="GCP Project ID")
    parser.add_argument("--region", default=DEFAULT_REGION, help="GCP Region")

    args = parser.parse_args()

    log("CEMENT PLANT AI - SERVICE MANAGEMENT CLI", Colors.HEADER)
    log("==========================================")

    if not check_gcloud_auth():
        sys.exit(1)

    log(
        f"🔍 Searching for services in project '{args.project}' (region '{args.region}')..."
    )
    services = get_existing_services(args.project, args.region)

    if not services:
        log("❌ No deployed Cloud Run services found.", Colors.FAIL)
        log("Verify that you have run the deployment script first.", Colors.WARNING)
        sys.exit(1)

    log(f"Found active services: {', '.join(services)}", Colors.OKBLUE)

    success = True
    for service in services:
        if args.action == "start":
            if not start_service(service, args.project, args.region):
                success = False
        elif args.action == "stop":
            if not stop_service(service, args.project, args.region):
                success = False
        elif args.action == "status":
            get_service_status(service, args.project, args.region)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
