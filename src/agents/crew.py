from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI


def create_crew_agent_system():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent_researcher = Agent(
        role="Researcher",
        goal="Analyze document content and extract key insights",
        backstory="Expert in academic research, reading scientific material to pull relevant findings.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    task_research = Task(
        description="Given the document content, extract key insights and summarize it.",
        agent=agent_researcher,
        expected_output="A comprehensive but concise set of bullet points.",
    )

    crew = Crew(
        agents=[agent_researcher],
        tasks=[task_research],
        verbose=True,
    )

    return crew
