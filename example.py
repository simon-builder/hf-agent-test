from transformers import CodeAgent, HfApiEngine, ReactCodeAgent
from huggingface_hub import login, InferenceClient

# Initialize env var
from dotenv import load_dotenv
import os

load_dotenv()

# Get the token from environment variables and login
hf_token = os.getenv('HF_ACCESS_TOKEN')
login(hf_token)
llm_engine = HfApiEngine(model="teknium/OpenHermes-2.5-Mistral-7B")

# Initialize the agent with base tools
agent = ReactCodeAgent(tools=[], add_base_tools=True)

# Try using one of the base tools
response = agent.run("Translate 'Hello, world!' to French")
print(response)