import asyncio
import os
import sys

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_tools import ALL_TOOLS

gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in your .env file.")

model_client = OpenAIChatCompletionClient(
    model="gemini-3.7-flash",
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        family="gemini",
        structured_output=True,
    ),
)

assistant = AssistantAgent(
    name="MailDocAssistant",
    model_client=model_client,
    system_message="""
    You are an automated email and document processing assistant.

    Your capabilities:
    - Email Management: Search, read, mark, move, and organize IMAP emails.
    - Document Handling: Download attachments and extract text from PDFs.
    - Decryption: Detect password-protected PDFs and inspect email text for password hints (DOB, PAN, account numbers, dates).
    - RAG Ingestion & Retrieval: Ingest documents into an Elasticsearch hybrid RAG pipeline and answer user queries with cited facts.
    - Long-term Memory: Save and retrieve user preferences and persistent facts using Mem0.

    Guidelines:
    - Autonomously chain operations to complete the request (e.g. search email -> download attachment -> unlock if needed -> ingest to RAG -> answer query).
    - Ground all RAG answers in source document pages.
    - Ask the user only when necessary credentials or passwords cannot be found.
    """,
    tools=ALL_TOOLS,
    reflect_on_tool_use=True,
    max_tool_iterations=15,
)


async def main():
    user_id = os.environ.get("USER_ID", "default_user")

    print(f"MailDoc Assistant initialized (User: {user_id})")
    print("Type your query or 'exit' to quit.\n")

    while True:
        try:
            user_input = input(">> ")
            if not user_input.strip():
                continue

            if user_input.strip().lower() in ["exit", "quit", "q"]:
                break

            task_prompt = f"User ID: {user_id}\nRequest: {user_input.strip()}"
            await Console(assistant.run_stream(task=task_prompt))

        except KeyboardInterrupt:
            break
        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
