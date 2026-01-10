"""Command-line interface for interactive chat experience with embodied agents.

Classes:
- CLIChat: Main CLI interface that orchestrates user input and agent output
- InputHandler: Manages asynchronous user input collection
- OutputFormatter: Formats agent responses for terminal display

Methods:
- CLIChat.run(): Starts the interactive chat session with the agent
- CLIChat.set_prompt_style(): Configure the appearance of user input prompts
- InputHandler.start_async(): Begin collecting user input in background thread
- OutputFormatter.format_message(): Apply styling to agent messages for display
"""
