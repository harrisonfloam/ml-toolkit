"""Communication manager for bidirectional agent-human message exchange during embodiment loop execution.

Classes:
- CommunicationManager: Handles message queuing between agent and human
- MessageBuffer: Thread-safe buffer for storing pending messages

Methods:
- CommunicationManager.agent_says(): Routes agent communication to output
- CommunicationManager.user_says(): Queues user input for next tick
- CommunicationManager.get_user_messages(): Retrieves pending user messages
- CommunicationManager.clear_buffers(): Empties all message queues
"""
