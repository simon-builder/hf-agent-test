from transformers import tool, CodeAgent
from huggingface_hub import list_models

llm_engine = HfApiEngine(model="mistralai/Mixtral-8x7B-Instruct-v0.1")

@tool
def model_download_tool(task: str) -> str:
    """
    Returns the most downloaded model for a given task on Hugging Face Hub.

    Args:
        task: The machine learning task to find the most downloaded model for
    """
    model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
    return model.id

# Assuming you have an LLM engine configured
# For this example, you'd replace llm_engine with your specific engine setup
agent = CodeAgent(tools=[model_download_tool])

# Example usage
response = agent.run(
    "What is the most downloaded model for text classification?"
)
print(response)