"""Base action interface and common action handlers for executing agent outputs during each tick.

Classes:
- ActionHandler: Abstract base interface for all action handlers
- CommunicationHandler: Routes agent messages to communication manager (always included)
- FileHandler: Writes agent output to files
- LogHandler: Records actions to structured logs

Methods:
- ActionHandler.handle(): Processes a single action dictionary from agent
- ActionHandler.can_handle(): Returns True if handler supports the action type
- FileHandler.set_output_dir(): Configure directory for file outputs
- LogHandler.set_log_level(): Set minimum logging level for actions
"""
