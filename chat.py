"""
Analyze a log message and take action accordingly.
For instance, if an error message is given then panic and generate a human readable panic message.
"""

from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch

load_dotenv()

main_model = ChatOllama(model="gemma3:1b", temperature=1, verbose=True,)

def info_template():
    messages = [
        ("system", "You are an good assistant and a single sentence response"),
        ("human", "Generate an ok message response for the given {log_msg}.")
    ]
    return ChatPromptTemplate.from_messages(messages)

def error_template():
    messages = [
        ("system", "You are an good assistant and a single sentence response"),
        ("human", "Generate a panic message response for the given {log_msg} for a human agent.")
    ]
    return ChatPromptTemplate.from_messages(messages)

def main_template():
    messages = [
        ("system", "You are an good assistant and generate a single sentence response."),
        ("human", "Analyze a log message {log_msg} as an INFO or ERROR level log message.")
    ]
    return ChatPromptTemplate.from_messages(messages)


def main():

    info_chain = info_template() | main_model | StrOutputParser()
    error_chain = error_template() | main_model | StrOutputParser()
    default_chain = info_template() | main_model | StrOutputParser()

    branches = RunnableBranch(
        (lambda msg: "error" not in msg.lower(), info_chain),
        (lambda msg: "error" in msg.lower(), error_chain),
        default_chain,
    )

    main_chain = main_template() | main_model | StrOutputParser()

    chain = main_chain | branches

    while True:

        log_msg = input("Type your log message(type 'q' to quit): ")
        if log_msg.lower() == "q":
            print("Goodbye!")
            break
        else:
            result = chain.invoke({"log_msg": log_msg})
            print(result)


if __name__ == "__main__":
    main()
