import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_community.llms import Ollama

os.environ["SERPER_API_KEY"] = "60f1f5434579c6dc0c65180acbed960ebacdcb09"

llm = Ollama(model="openhermes")

search_tool = SerperDevTool()

researcher = Agent(
    llm=llm,
    role="Real Estate Researcher",
    goal="Find undervalued homes in New Orleans.",
    backstory="You are an experienced real estate analyst. Your task is to find undervalued homes in NOLA.",
    allow_delegation=False,
    tools=[search_tool],
    verbose=True,
)

task1 = Task(
    description="Search the internet and find 5 undervalued homes in New Orleans, USA. For each home, highlight the mean, low and max prices as well as the potential reasons for undervaluation and any other useful factors.",
    expected_output="""A detailed report of each of the homes. The results should be formatted as shown below:

    Home 1: Address
    Mean Price: $XXX,000
    Low Price: $XXX,000
    Max Price: $XXX,000
    Potential Reasons for Undervaluation: Reason
    Additional Information: Info

    Home 2: Address
    Mean Price: $XXX,000
    Low Price: $XXX,000
    Max Price: $XXX,000
    Potential Reasons for Undervaluation: Reason
    Additional Information: Info

    Home 3: Address
    Mean Price: $XXX,000
    Low Price: $XXX,000
    Max Price: $XXX,000
    Potential Reasons for Undervaluation: Reason
    Additional Information: Info

    Home 4: Address
    Mean Price: $XXX,000
    Low Price: $XXX,000
    Max Price: $XXX,000
    Potential Reasons for Undervaluation: Reason
    Additional Information: Info

    Home 5: Address
    Mean Price: $XXX,000
    Low Price: $XXX,000
    Max Price: $XXX,000
    Potential Reasons for Undervaluation: Reason
    Additional Information: Info
    """,
    agent=researcher,
    output_file="task1_output.txt",
)

writer = Agent(
    llm=llm,
    role="Real Estate Analyst",
    goal="Summarize property facts into a report for investors.",
    backstory="You are a real estate analyst, your goal is to compile property analytics into a report for potential investors.",
    allow_delegation=False,
    verbose=True,
)

task2 = Task(
    description="Summarize the property information into a bullet-point list.",
    expected_output="""A summarized bullet-point list of each of the homes, prices, and important features of that home.

    - Home 1: Address
        * Mean Price: $XXX,000
        * Low Price: $XXX,000
        * Max Price: $XXX,000
        * Potential Reasons for Undervaluation: Reason
        * Additional Information: Info

    - Home 2: Address
        * Mean Price: $XXX,000
        * Low Price: $XXX,000
        * Max Price: $XXX,000
        * Potential Reasons for Undervaluation: Reason
        * Additional Information: Info

    - Home 3: Address
        * Mean Price: $XXX,000
        * Low Price: $XXX,000
        * Max Price: $XXX,000
        * Potential Reasons for Undervaluation: Reason
        * Additional Information: Info

    - Home 4: Address
        * Mean Price: $XXX,000
        * Low Price: $XXX,000
        * Max Price: $XXX,000
        * Potential Reasons for Undervaluation: Reason
        * Additional Information: Info

    - Home 5: Address
        * Mean Price: $XXX,000
        * Low Price: $XXX,000
        * Max Price: $XXX,000
        * Potential Reasons for Undervaluation: Reason
        * Additional Information: Info
    """,
    agent=writer,
    output_file="task2_output.txt",
)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=2)

task_output = crew.kickoff()
print(task_output)
