from crewai import Agent, Task, Crew
from langchain.chat_models import init_chat_model


def create_crew_agent_system():
    llm = init_chat_model(model="gpt-4o-mini", temperature=0)

    agent_researcher = Agent(
        role="Researcher",
        goal="Analyze document content and extract key insights",
        backstory="Expert in academic research, reading scientific material to pull relevant findings.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    task_research = Task(
        description="Using the provided document context, answer the user's question accurately and concisely.",
        agent=agent_researcher,
        expected_output="A comprehensive but concise set of bullet points.",
    )

    crew = Crew(
        agents=[agent_researcher],
        tasks=[task_research],
        verbose=True,
    )

    return crew
