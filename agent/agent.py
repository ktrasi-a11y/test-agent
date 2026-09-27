import os
from typing import Optional

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.api_core.client_options import ClientOptions
from google.cloud import modelarmor_v1
from google.genai import types

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "elevate-ktrasi")
LOCATION = os.environ.get("MODEL_ARMOR_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
MODEL_ARMOR_TEMPLATE = os.environ.get(
    "MODEL_ARMOR_TEMPLATE",
    f"projects/{PROJECT_ID}/locations/{LOCATION}/templates/minimal-agent-armor",
)


def _get_model_armor_client() -> modelarmor_v1.ModelArmorClient:
    return modelarmor_v1.ModelArmorClient(
        client_options=ClientOptions(
            api_endpoint=f"modelarmor.{LOCATION}.rep.googleapis.com"
        )
    )


def sanitize_prompt_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Screens incoming user prompts with Google Cloud Model Armor before calling the LLM."""
    if not llm_request.contents:
        return None

    last_content = llm_request.contents[-1]
    user_text = "\n".join(
        part.text for part in (last_content.parts or []) if getattr(part, "text", None)
    ).strip()
    if not user_text:
        return None

    try:
        client = _get_model_armor_client()
        response = client.sanitize_user_prompt(
            request=modelarmor_v1.SanitizeUserPromptRequest(
                name=MODEL_ARMOR_TEMPLATE,
                user_prompt_data=modelarmor_v1.DataItem(text=user_text),
            )
        )
    except Exception as exc:
        print(f"Warning: Model Armor prompt check skipped ({exc})")
        return None

    if (
        response.sanitization_result.filter_match_state
        == modelarmor_v1.FilterMatchState.MATCH_FOUND
    ):
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="I cannot fulfill this request because it was flagged by Model Armor safety policies."
                    )
                ],
            )
        )
    return None


def sanitize_response_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Screens outgoing model responses with Google Cloud Model Armor before returning to the user."""
    if not llm_response.content or not llm_response.content.parts:
        return None

    model_text = "\n".join(
        part.text for part in llm_response.content.parts if getattr(part, "text", None)
    ).strip()
    if not model_text:
        return None

    try:
        client = _get_model_armor_client()
        response = client.sanitize_model_response(
            request=modelarmor_v1.SanitizeModelResponseRequest(
                name=MODEL_ARMOR_TEMPLATE,
                model_response_data=modelarmor_v1.DataItem(text=model_text),
            )
        )
    except Exception as exc:
        print(f"Warning: Model Armor response check skipped ({exc})")
        return None

    if (
        response.sanitization_result.filter_match_state
        == modelarmor_v1.FilterMatchState.MATCH_FOUND
    ):
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="The response was blocked by Model Armor safety policies."
                    )
                ],
            )
        )
    return None


root_agent = Agent(
    name="minimal_agent",
    model="gemini-2.5-flash",
    description="A minimal generic assistant agent protected by Model Armor.",
    instruction="You are a concise assistant.",
    before_model_callback=sanitize_prompt_callback,
    after_model_callback=sanitize_response_callback,
)

