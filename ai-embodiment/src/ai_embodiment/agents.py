"""Base agent interface and LLM agent implementations for processing observations and generating actions.

Classes:
- Agent: Abstract base interface for all reasoning agents
- LLMAgent: Agent that uses language models for reasoning and decision making
- MockAgent: Simple agent for testing that returns predefined responses

Methods:
- Agent.process(): Takes observation dictionary and returns action dictionary
- LLMAgent.set_model(): Configure which language model to use
- LLMAgent.set_system_prompt(): Set the system prompt for the agent
- MockAgent.set_responses(): Configure predefined responses for testing
"""
