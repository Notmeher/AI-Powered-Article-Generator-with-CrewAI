import os
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

# Streamlit UI
st.title("AI-Powered Article Generator with CrewAI")
query = st.text_input("Enter your search keyword:")

if st.button("Generate Article"):
    if query:
        with st.spinner("Performing web search..."):
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
                        "You are a professional article writer. Write a  comprehensive article about the provided topic in a Single Paragraph. "
                        "Incorporate information from the provided sources and ensure the article is well-structured and informative."
                        "The article should be engaging and well-researched."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Write an article about '{query}' using the following sources:\n\n" +
                        "\n".join([f"[{i}] {title} - {url}" for i, (title, url) in enumerate(sources, start=1)])
                    )
                )
            ]

            with st.spinner("Generating article with GPT-4..."):
                # Initialize LangChain OpenAI integration
                llm = ChatOpenAI(
                    model="gpt-4",  # Specify the OpenAI GPT model
                    temperature=0.7,
                    openai_api_key=os.getenv("OPENAI_API_KEY")  # Load OpenAI API key from environment variables
                )

                # Generate the article using GPT-4
                response = llm(messages)
                article = response.content.strip()

            # Display the generated article
            st.markdown("### Generated Article:")
            st.write(article)
    else:
        st.error("Please enter a search keyword.")
