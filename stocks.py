import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_community.llms import Ollama

os.environ["SERPER_API_KEY"] = "60f1f5434579c6dc0c65180acbed960ebacdcb09"

llm = Ollama(model="openhermes")

search_tool = SerperDevTool()

# Define a Researcher Agent for Long-Term Investments
long_term_researcher = Agent(
    llm=llm,
    role="Long-Term Tech Stock Researcher",
    goal="Find long-term tech stocks to invest in based on Warren Buffett's strategy.",
    backstory="You are an experienced stock analyst. Your task is to find long-term tech stocks with strong fundamentals.",
    allow_delegation=False,
    tools=[search_tool],
    verbose=True,
)

# Define a Researcher Agent for Short-Term Investments
short_term_researcher = Agent(
    llm=llm,
    role="Short-Term Tech Stock Researcher",
    goal="Find short-term tech stocks to invest in based on Warren Buffett's strategy.",
    backstory="You are an experienced stock analyst. Your task is to find short-term tech stocks with strong fundamentals.",
    allow_delegation=False,
    tools=[search_tool],
    verbose=True,
)

# Task for Long-Term Investments
task1 = Task(
    description="""Search the internet and find 5 long-term tech stocks to invest in based on Warren Buffett's strategy. 
    For each stock, highlight the following:
    - Understanding of the Business
    - Economic Moat
    - Management Quality
    - Consistent Earnings
    - Reasonable Price (Intrinsic Value)
    - Long-Term Perspective
    - Strong Return on Equity (ROE)
    - Low Debt Levels
    - Free Cash Flow
    - Avoid Complexity

    The results should be formatted as shown below:

    Stock 1: Ticker Symbol
    Understanding of the Business: Details
    Economic Moat: Details
    Management Quality: Details
    Consistent Earnings: Details
    Reasonable Price: Details
    Long-Term Perspective: Details
    Strong ROE: Details
    Low Debt Levels: Details
    Free Cash Flow: Details
    Avoid Complexity: Details

    Stock 2: Ticker Symbol
    ...""",
    expected_output="Detailed report on 5 long-term tech stocks.",
    agent=long_term_researcher,
    output_file="long_term_stocks.txt",
)

# Task for Short-Term Investments
task2 = Task(
    description="""Search the internet and find 5 short-term tech stocks to invest in based on Warren Buffett's strategy. 
    For each stock, highlight the following:
    - Understanding of the Business
    - Economic Moat
    - Management Quality
    - Consistent Earnings
    - Reasonable Price (Intrinsic Value)
    - Long-Term Perspective
    - Strong Return on Equity (ROE)
    - Low Debt Levels
    - Free Cash Flow
    - Avoid Complexity

    The results should be formatted as shown below:

    Stock 1: Ticker Symbol
    Understanding of the Business: Details
    Economic Moat: Details
    Management Quality: Details
    Consistent Earnings: Details
    Reasonable Price: Details
    Long-Term Perspective: Details
    Strong ROE: Details
    Low Debt Levels: Details
    Free Cash Flow: Details
    Avoid Complexity: Details

    Stock 2: Ticker Symbol
    ...""",
    expected_output="Detailed report on 5 short-term tech stocks.",
    agent=short_term_researcher,
    output_file="short_term_stocks.txt",
)

# Define a Writer Agent for summarizing the results
writer = Agent(
    llm=llm,
    role="Stock Analyst",
    goal="Summarize the stock information into a report for investors.",
    backstory="You are a stock analyst, your goal is to compile stock analytics into a report for potential investors.",
    allow_delegation=False,
    verbose=True,
)

# Task for Summarizing Long-Term Investments
task3 = Task(
    description="Summarize the long-term tech stock information into a bullet-point list.",
    expected_output="""A summarized bullet-point list of each of the stocks, key details, and important features of that stock.

    - Stock 1: Ticker Symbol
        * Understanding of the Business: Details
        * Economic Moat: Details
        * Management Quality: Details
        * Consistent Earnings: Details
        * Reasonable Price: Details
        * Long-Term Perspective: Details
        * Strong ROE: Details
        * Low Debt Levels: Details
        * Free Cash Flow: Details
        * Avoid Complexity: Details

    - Stock 2: Ticker Symbol
    ...""",
    agent=writer,
    output_file="long_term_summary.txt",
)

# Task for Summarizing Short-Term Investments
task4 = Task(
    description="Summarize the short-term tech stock information into a bullet-point list.",
    expected_output="""A summarized bullet-point list of each of the stocks, key details, and important features of that stock.

    - Stock 1: Ticker Symbol
        * Understanding of the Business: Details
        * Economic Moat: Details
        * Management Quality: Details
        * Consistent Earnings: Details
        * Reasonable Price: Details
        * Long-Term Perspective: Details
        * Strong ROE: Details
        * Low Debt Levels: Details
        * Free Cash Flow: Details
        * Avoid Complexity: Details

    - Stock 2: Ticker Symbol
    ...""",
    agent=writer,
    output_file="short_term_summary.txt",
)

# Initialize Crew and Execute Tasks
crew = Crew(agents=[long_term_researcher, short_term_researcher, writer], tasks=[task1, task2, task3, task4], verbose=2)

task_output = crew.kickoff()
print(task_output)
