"""Core embodiment loop that orchestrates the sense-think-act cycle in discrete ticks.

Classes:
- Loop: Main embodiment loop with builder pattern for configuration
- LoopBuilder: Builder for constructing Loop instances with composable components
- TickResult: Data structure containing the results of a single tick execution

Methods:
- Loop.builder(): Returns a new LoopBuilder instance
- Loop.start_cli_chat(): Starts interactive command-line chat interface
- Loop.start_async(): Runs the loop continuously in background
- Loop.step(): Executes a single tick manually
- Loop.stop(): Stops the running loop
"""
