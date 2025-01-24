import os
import asyncio
import streamlit as st
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize the SerperDevTool to retrieve 10 search results
search_tool = SerperDevTool(n_results=10)

# Define the Agent
search_agent = Agent(
    role="Research Assistant",
    goal="Retrieve the top 10 search results for a given query.",
    backstory="An AI agent specialized in performing web searches.",
    tools=[search_tool],
    verbose=True
)

# Define the Task for retrieving search results
search_task = Task(
    description="Perform a web search for the query: {query}",
    expected_output="A list of the top 10 search results with titles and URLs.",
    agent=search_agent
)

# Define the Crew for performing the search
search_crew = Crew(
    agents=[search_agent],
    tasks=[search_task],
    verbose=True
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []


def display_message(role, content):
    """Display a message in the chat interface based on its role."""
    with st.chat_message(role):
        if role == "system":
            st.markdown(f"**System**: {content}")
        else:
            st.markdown(content)


async def run_workflow_with_streaming(query):
    try:
        # Perform the web search
        search_result = search_crew.kickoff(inputs={"query": query})
        search_output = search_result.raw

        # Extract titles and URLs from the search output
        search_entries = search_output.split('\n')
        sources = []
        for entry in search_entries:
            if entry.strip():
                parts = entry.split(' - ')
                if len(parts) == 2:
                    title = parts[0].strip()
                    url = parts[1].strip()
                    sources.append((title, url))

        # Prepare the messages for GPT-4
        messages = [
            SystemMessage(
                content=(
                    "You are a professional article writer. Write a comprehensive article about the provided topic in a single paragraph. "
                    "Incorporate information from the provided sources and ensure the article is well-structured and informative. "
                    "The article should be engaging and well-researched."
                )
            ),
            HumanMessage(
                content=(
                    f"Write an article about '{query}' using the following sources:\n\n" +
                    "\n".join([f"{title} - {url}" for title, url in sources])
                )
            )
        ]

        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": query})
        display_message("user", query)

        # Generate the article with GPT-4
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        response = llm(messages)
        article = response.content.strip()

        # Simulate streaming response
        message_placeholder = st.empty()
        partial_response = ""

        for line in article.split("\n"):
            await asyncio.sleep(0.1)
            partial_response += f"{line}\n"
            message_placeholder.markdown(partial_response)

        # Add assistant's response to chat history
        st.session_state["messages"].append({"role": "assistant", "content": partial_response})

        # Display sources if available
        if sources:
            source_text = "\n".join([f"- ({url})" for title, url in sources])
            st.session_state["messages"].append({"role": "system", "content": f"Sources:\n{source_text}"})
            display_message("system", f"Sources:\n{source_text}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


async def main():
    st.title("AI Powered Web searched Article Generator with CrewAI")
    st.write("Welcome to the AI-powered article generator! Enter a topic to generate a comprehensive article based on the top search results.")

    # Display existing chat history
    for message in st.session_state["messages"]:
        display_message(message["role"], message["content"])

    # Handle new user input
    user_query = st.chat_input("What is the topic you would like to generate an article about?")
    if user_query:
        await run_workflow_with_streaming(user_query)


if __name__ == "__main__":
    asyncio.run(main())
