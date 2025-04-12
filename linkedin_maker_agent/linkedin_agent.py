"""
Author: Emmitt J Tucker
Date: 2025-04-12
Description: This file contains a LLM multi-agent system that is designed to write social media content for LinkedIn.
"""
# --- Part 0: Libraries/Modules/Constants ---

# Importing necessary libraries and modules
from venv import logger
from typing import AsyncGenerator
from typing import override
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.tools import google_search

# Constants
GEMINI_FLASH = "gemini-2.0-flash-exp" # Define model constant

# --- Part 1: Simplified Custom Agent Initalization ---

class LinkedInAgent(BaseAgent):
    """
    A class representing a LinkedIn agent that generates social media content.
    Inherits from the BaseAgent class.
    """

    # An LLM agent that serves as the searcher for the LinkedIn Agent. 
    searcher: LlmAgent = None

    # An LLM Agent that serves as the article generator for the LinkedIn Agent. 
    article_generator: LlmAgent = None

    # An LLM Agent that serves as the tone checker for the LinkedIn Agent.
    tone_checker: LlmAgent = None

    # An LLM Agent that serves as the revisor for the LinkedIn Agent.
    revisor: LlmAgent = None

    # An LLM Agent that serves as the critic for the LinkedIn Agent.
    critic: LlmAgent = None

    # An agent that formats the article for the LinkedIn Posts
    formatter: LlmAgent = None

    # A Loop agent that controls the article review flow of the Linkedin Agent. 
    loop_agent: LoopAgent = None

    # A sequnetial agent that controls the flow of the LinkedIn agent.
    sequential_agent: SequentialAgent = None

    def __init__(
            self,
            name: str
    ):
        """
        Initializes the LinkedIn agent with the provided sub-agents.

        Args:
            name (str): The name of the LinkedIn agent.
        """
        # Create all the internal agents as local variables first
        searcher = LlmAgent(
            name = "LinkedInArticleSearcher",
            model = GEMINI_FLASH,
            description = "An LLM agent that searches for relevant articles and information to generate LinkedIn posts.",
            instruction= (
                """
                You are an LLM agent that searches for relevant articles and information to generate LinkedIn posts.
                Research  topic in provided in the session state with the key and 'topic'.
                - Search for articles, blog posts, and other relevant content that can be used to generate a LinkedIn article.
                - Use the following information to guide your search:
                    - Check for legitimacy of the sources and ensure that the information is accurate and up-to-date.
                    - Provide a summary of the search results and highlight the most relevant articles.
                    - Provide the search results in a structured format.
                    - Provide sources for the information you find.
                """
            ),
            input_schema=None,
            output_key="research_results",  # Changed to plural to match what's expected
            tools = [google_search]  # List of tools to use for searching
            
        )

        article_generator = LlmAgent(
            name = "LinkedInArticleGenerator",
            model = GEMINI_FLASH,
            description = "An LLM agent that generates articles based on provided research.",
            instruction= (
                """
                You are an LLM agent that generates articles based on the provided research.
                Review the provided research based on the topic in provided in the session state with the key 'research_results' and 'topic'.
                Write a long-form article that is relevant to the provided research.
                The article should be informative, engaging, and suitable for a LinkedIn audience.
                Use the following information to guide your search:
                - Write the article about the following topic: 'topic'
                - The article should be well-structured and include an introduction, body, and conclusion.
                - The article should be around 1500-2000 words long. 
                """
            ),
            input_schema=None,
            output_key="current_article",  # Key for storing output in session state
        )

        tone_checker = LlmAgent(
            name = "LinkedInToneChecker",
            model = GEMINI_FLASH,
            description = "An LLM agent that checks the tone of the articles.",
            instruction= (
                """
                You are an LLM agent that checks the tone of the articles.
                Review the provided article based on the topic in provided in the session state with the key 'current_article'.
                Ensure that the tone is professional, engaging, and suitable for a LinkedIn audience.
                Make edits to the article to improve the tone if necessary.
                """
            ),
            input_schema=None,
            output_key="current_article_tone_checked",  # Key for storing output in session state
        )

        revisor = LlmAgent(
            name = "LinkedInArticleRevisor",
            model = GEMINI_FLASH,
            description = "An LLM agent that revises the articles for a LinkedInPost.",
            instruction= (
                """
                You are an LLM agent that revises the articles.
                Review the provided article based on the topic in provided in the session state with the key 'current_article_tone_checked'.
                Make edits to the article to improve clarity, coherence, and overall quality.
                Ensure that the article is well-structured and free of grammatical errors.
                """
            ),
            input_schema=None,
            output_key="current_article_revised",  # Key for storing output in session state
        )

        critic = LlmAgent(
            name = "LinkedInCritic",
            model = GEMINI_FLASH,
            description = "An LLM agent that critiques the articles.",
            instruction= (
                """
                You are an LLM agent that critiques the articles.
                Review the provided article based on the topic in provided in the session state with the key 'current_article_revised'.
                Evaluate the article for its relevance, quality, and suitability for a LinkedIn audience.
                Provide feedback on the article and suggest improvements if necessary.
                """
            ),
            input_schema=None,
            output_key="current_article_critic_check",  # Key for storing output in session state
        )

        formatter = LlmAgent(
            name = "LinkedInPostFormatter",
            model = GEMINI_FLASH,
            description = "An LLM agent that formats the articles for LinkedIn posts.",
            instruction= (
                """
                You are an LLM agent that formats the articles for LinkedIn posts.
                Review the provided article based on the topic in provided in the session state with the key 'current_article_critic_check'.
                Format the article for a LinkedIn post by adding appropriate headings, bullet points, and images.
                Ensure that the article is visually appealing and easy to read.
                """
            ),
            input_schema=None,
            output_key="formatted_article",  # Key for storing output in session state
        )
        
        
        loop_agent = LoopAgent(
            name = "ArticleReviewLoop",
            sub_agents = [article_generator, tone_checker, revisor, critic],
            max_iterations = 5
        )

        sequential_agent = SequentialAgent(
            name = "LinkedInAgent",
            sub_agents = [searcher, loop_agent, formatter]
        )

        # Call super().__init__ first to initialize the Pydantic model
        super().__init__(
            name=name,
            sub_agents=[sequential_agent]
        )
        
        # Only after super().__init__ can we set attributes
        self.searcher = searcher
        self.article_generator = article_generator
        self.tone_checker = tone_checker
        self.revisor = revisor
        self.critic = critic
        self.formatter = formatter
        self.loop_agent = loop_agent
        self.sequential_agent = sequential_agent

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """ 
        Implements the custom orchistration logic for the LinkedIn agent.
        """
        # Starting message for the LinkedIn agent.
        logger.info(f"Running LinkedIn agent: [{self.name}]")

        # 1. Initial Research
        logger.info(f"[{self.name}] Starting initial research...")
        async for event in self.searcher.run_async(ctx):
            logger.info(f"[{self.name}] Research event: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        # 2. Check if the research was successful before proceeding.
        if "research_results" not in ctx.session.state or not ctx.session.state["research_results"]:
            logger.error(f"[{self.name}] Research failed. Exiting workflow.")
            return

        logger.info(f"[{self.name}] Research successful. Research obtained is: {ctx.session.state.get('research_results')}")

        # 3. Generate the article using the research results loop.
        logger.info(f"[{self.name}] Starting article generation...")
        # Using the sequential_agent instance to run the article generation process.
        async for event in self.sequential_agent.run_async(ctx):
            logger.info(f"[{self.name}] Event from PostProcessing: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        # 4. Critic conditional logic to determine if the article is ready for posting.
        critic_check_result = ctx.session.state.get("current_article_critic_check")
        logger.info(f"[{self.name}] Critic check result: {critic_check_result}")

        if critic_check_result == "negative":
            logger.info(f"[{self.name}] Article is not ready for posting. Regenerating article...")
            async for event in self.critic.run_async(ctx):
                logger.info(f"[{self.name}] Event from Critic (Regen): {event.model_dump_json(indent=2, exclude_none=True)}")
                yield event
        else:
            logger.info(f"[{self.name}] Article is ready for posting. Keeping current article.")
            pass

        logger.info(f"[{self.name}] Workflow finished.")