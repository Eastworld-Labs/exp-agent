"""Minimal example of adding a capability module."""

from system2_agent.tools import Tool, object_schema


class BatteryModule:
    name = "battery"

    def __init__(self) -> None:
        self.percent = 87

    def tools(self):
        return (
            Tool(
                name="get_battery",
                description="Read remaining battery percentage.",
                parameters=object_schema({}),
                handler=lambda _: {"percent": self.percent},
            ),
        )

    def snapshot(self):
        return {"percent": self.percent}
