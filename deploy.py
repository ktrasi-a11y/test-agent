import os
import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

from agent.agent import root_agent

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "elevate-ktrasi")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
STAGING_BUCKET = os.environ.get("STAGING_BUCKET", f"gs://{PROJECT_ID}-agent-staging")
DISPLAY_NAME = os.environ.get("AGENT_DISPLAY_NAME", "minimal-agent")


def main() -> None:
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=STAGING_BUCKET,
    )

    app = AdkApp(agent=root_agent)

    remote_agent = agent_engines.create(
        app,
        requirements=[
            "google-adk>=1.0.0",
            "google-cloud-aiplatform[adk,agent_engines]>=1.93.0",
        ],
        extra_packages=["./agent"],
        display_name=DISPLAY_NAME,
    )
    print(f"Deployed Agent Engine resource: {remote_agent.resource_name}")


if __name__ == "__main__":
    main()
