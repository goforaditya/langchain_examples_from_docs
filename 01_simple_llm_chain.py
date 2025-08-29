# In this tutorial we build a simple language translator using langchain
# First install langchain using 
# !pip install langchain
#
#
# Will be using free Gemini API key from Google can also use Openrouter.
#
# !pip install -qU "langchain[google-genai]"
# Created an env var in ~/.bashrc but can also use an local python var
# added to bashrc
# export GOOGLE_API_KEY=<API KEY from https://aistudio.google.com/>
#
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

# Both input and output of ChatModel is a Messages object  https://github.com/goforaditya/langchain_examples_from_docs
# Messages object can also hold coversation roles and tool calls etc.,
#

# We can also use OpenAI goforaditya

model.invoke([{"role": "user", "content": "Hello"}])

model.invoke([HumanMessage("Hello")])

# We can also stream individual tokens instead of waiting for complete output form the model

for token in model.stream(messages):
    print(token.content, end="|")
