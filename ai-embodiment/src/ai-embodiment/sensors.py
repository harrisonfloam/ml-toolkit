"""Base sensor interface and common sensor implementations for collecting observations during each tick.

Classes:
- Sensor: Abstract base interface for all sensors
- TextInputSensor: Captures user text input for agent processing
- FileWatcherSensor: Monitors file changes and reads content
- TimeSensor: Provides timestamp and timing information

Methods:
- Sensor.read(): Returns current sensor observations as dictionary
- Sensor.has_new_data(): Indicates if sensor has fresh data since last read
- TextInputSensor.add_message(): Queues a text message for next read
- FileWatcherSensor.watch_file(): Sets file path to monitor for changes
"""
