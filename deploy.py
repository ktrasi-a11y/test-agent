import os
import vertexai
from google.api_core import exceptions as gcp_exceptions
from google.api_core.client_options import ClientOptions
from google.cloud import modelarmor_v1
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

from agent.agent import root_agent

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "elevate-ktrasi")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
STAGING_BUCKET = os.environ.get("STAGING_BUCKET", f"gs://{PROJECT_ID}-agent-staging")
DISPLAY_NAME = os.environ.get("AGENT_DISPLAY_NAME", "minimal-agent")
TEMPLATE_ID = os.environ.get("MODEL_ARMOR_TEMPLATE_ID", "minimal-agent-armor")
MODEL_ARMOR_TEMPLATE = os.environ.get(
    "MODEL_ARMOR_TEMPLATE",
    f"projects/{PROJECT_ID}/locations/{LOCATION}/templates/{TEMPLATE_ID}",
)


def ensure_model_armor_template() -> str:
    """Ensures the Model Armor template exists, creating a baseline template if not found."""
    client = modelarmor_v1.ModelArmorClient(
        client_options=ClientOptions(
            api_endpoint=f"modelarmor.{LOCATION}.rep.googleapis.com"
        )
    )
    try:
        template = client.get_template(name=MODEL_ARMOR_TEMPLATE)
        print(f"Found existing Model Armor template: {template.name}")
        return template.name
    except gcp_exceptions.NotFound:
        print(f"Creating Model Armor template: {MODEL_ARMOR_TEMPLATE}")
        template_config = modelarmor_v1.Template(
            filter_config=modelarmor_v1.FilterConfig(
                pi_and_jailbreak_filter_settings=modelarmor_v1.PiAndJailbreakFilterSettings(
                    filter_enforcement=modelarmor_v1.PiAndJailbreakFilterSettings.PiAndJailbreakFilterEnforcement.ENABLED,
                    confidence_level=modelarmor_v1.DetectionConfidenceLevel.LOW_AND_ABOVE,
                ),
                malicious_uri_filter_settings=modelarmor_v1.MaliciousUriFilterSettings(
                    filter_enforcement=modelarmor_v1.MaliciousUriFilterSettings.MaliciousUriFilterEnforcement.ENABLED,
                ),
                rai_settings=modelarmor_v1.RaiFilterSettings(
                    rai_filters=[
                        modelarmor_v1.RaiFilterSettings.RaiFilter(
                            filter_type=modelarmor_v1.RaiFilterType.HATE_SPEECH,
                            confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
                        ),
                        modelarmor_v1.RaiFilterSettings.RaiFilter(
                            filter_type=modelarmor_v1.RaiFilterType.HARASSMENT,
                            confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
                        ),
                        modelarmor_v1.RaiFilterSettings.RaiFilter(
                            filter_type=modelarmor_v1.RaiFilterType.SEXUALLY_EXPLICIT,
                            confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
                        ),
                        modelarmor_v1.RaiFilterSettings.RaiFilter(
                            filter_type=modelarmor_v1.RaiFilterType.DANGEROUS,
                            confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
                        ),
                    ]
                ),
            )
        )
        created = client.create_template(
            request=modelarmor_v1.CreateTemplateRequest(
                parent=f"projects/{PROJECT_ID}/locations/{LOCATION}",
                template_id=TEMPLATE_ID,
                template=template_config,
            )
        )
        print(f"Created Model Armor template: {created.name}")
        return created.name
    except Exception as exc:
        print(f"Warning: Could not auto-provision Model Armor template ({exc}). Continuing with {MODEL_ARMOR_TEMPLATE}.")
        return MODEL_ARMOR_TEMPLATE


def main() -> None:
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=STAGING_BUCKET,
    )

    template_name = ensure_model_armor_template()
    app = AdkApp(agent=root_agent)

    remote_agent = agent_engines.create(
        app,
        requirements=[
            "google-adk>=1.0.0",
            "google-cloud-aiplatform[adk,agent_engines]>=1.93.0",
            "google-cloud-modelarmor>=0.1.0",
            "cloudpickle>=3.0.0",
            "pydantic>=2.12.0",
        ],
        extra_packages=["./agent"],
        display_name=DISPLAY_NAME,
        env_vars={
            "MODEL_ARMOR_TEMPLATE": template_name,
            "MODEL_ARMOR_LOCATION": LOCATION,
        },
    )
    print(f"Deployed Agent Engine resource: {remote_agent.resource_name}")


if __name__ == "__main__":
    main()

