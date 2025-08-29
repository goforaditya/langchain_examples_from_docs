# We are passing a list of messages directly into the language model. 
# But we can also define templates which help us support our applicaiton use case
# This approach is better as user does not need to provide the full context

import os
from langchain.chat_models import init_chat_model

model = init_chat_model("meta-llama/llama-3.3-8b-instruct:free", model_provider="openai", base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_KEY"])

from langchain_core.messages import HumanMessage, SystemMessage

# This implements the Messages concept of langchain https://python.langchain.com/docs/concepts/messages/
# langchain core library provides a lot of Standardization using concepts https://python.langchain.com/docs/concepts/

messages = [
    SystemMessage(content="Translate the following from english to hindi."),
    HumanMessage(content="hi! i love playing with data and would also like to experiment with AI and langchain")
]

print(model.invoke(messages))

# The Prompt templates concept can help us build this out.
# A prompt templates takes raw user input and return a LLM ready prompt

form langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",system_template),
        ("user", "{text}")
    ]
)

# Always refer to API docs https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html 
# before using langchain modules for better learning

prompt = prompt_template.invoke({"language": "Hindi", "text": "I love GenAI and it's applications"})
# can also take lang and text as input from user
print("PROMPT->", prompt)

print("CONVERTED_PROMPT->", prompt.to_messages())

# invoke
response = model.invoke(prompt)
print(response.content)
