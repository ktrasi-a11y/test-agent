from google.adk.agents import Agent

root_agent = Agent(
    name="minimal_agent",
    model="gemini-2.5-flash",
    description="A minimal generic assistant agent.",
    instruction="You are a very helpful, concise assistant.",
)
