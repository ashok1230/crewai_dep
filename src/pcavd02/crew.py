import os
from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool

from typing import List

load_dotenv('.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

serper_dev_tool = SerperDevTool()

@CrewBase
class Pcavd02():
    """Pcavd02 crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def search(self) -> Agent:
        return Agent(
            config=self.agents_config['search'],
            verbose=True,
            tools=[serper_dev_tool],
            max_iter=2,
            llm="gpt-4o-mini"
        )

    @task
    def search_task(self) -> Task:
        return Task(
            config=self.tasks_config['search_task']
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Pcavd02 crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
