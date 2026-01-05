import getpass
import os

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"

# Securely enter your API key
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
os.environ['OPENAI_API_KEY'] = getpass.getpass('Enter your OpenAI API key: ')