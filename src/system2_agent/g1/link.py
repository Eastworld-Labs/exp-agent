"""The MQTT link to the robot, behind a protocol small enough to fake.

Everything above this file talks to `Link`, which has four methods and no
mention of MQTT, CBOR or paho. That is what lets the navigation backend and the
whole service be tested with no broker, no robot and no dependencies at all --
see tests/test_g1_nav2_backend.py.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Protocol

from ..types import Json
from .wire import PUBLISHABLE, SUBSCRIBED, downlink, uplink


class Link(Protocol):
    """What the mission service needs from a transport."""

    def connected(self) -> bool: ...

    def subscribe(self, ros_topic: str) -> None: ...

    def latest(self, ros_topic: str) -> "tuple[Json, float] | None":
        """The newest message on a topic and the monotonic time it ARRIVED.

        ⚠️ ARRIVAL TIME, NOT THE MESSAGE'S OWN STAMP. Several of these topics
        are retained by the broker, so a subscriber that has just connected is
        handed a value that may be hours old, instantly. Age has to be measured
        from when WE got it, and a caller has to check it -- see the pose
        freshness gate in nav2_backend.
        """

    def publish_cmd(self, ros_topic: str, msg: Json, expiry_s: int | None = None) -> None: ...

    # ⚠️ OPTIONAL. Callers must probe with `getattr(link, "watch", None)` and
    # cope without it. `latest()` keeps ONE message per topic, which is right
    # for state (a pose, a latch) and wrong for a stream of deltas: Nav2's
    # costmap patches only make sense applied in order, and between two tool
    # calls every patch but the last would be dropped. A watcher sees each one
    # as it lands. Not part of the four methods a fake must implement.
    # def watch(self, ros_topic, callback) -> None: ...


def default_client_id(robot_id: str) -> str:
    """One MQTT client id per (process, robot).

    ⚠️ THE ID MUST DIFFER PER ROBOT. The service opens one link per target,
    back to back, and a broker treats a second CONNECT with the same client id
    as a takeover: it drops the first session. Two links whose ids came from a
    millisecond clock were created in the same millisecond, shared an id, and
    kicked each other off the broker for ever -- the sim target read
    "disconnected" while the real one flapped. The robot id is what makes the
    two different; the pid keeps two service processes apart.
    """
    return f"mission-{robot_id}-{os.getpid()}"


class MqttLink:
    """paho-mqtt + CBOR, speaking the fleet agent's topic layout.

    Connects as an `operator-` username, which is what the broker's ACL grants
    `g1/+/cmd/#` publish and `g1/+/ros/#` subscribe.
    """

    def __init__(
        self,
        *,
        broker: str = "127.0.0.1",
        port: int = 1883,
        robot_id: str = "g1-0001",
        username: str = "operator-mission",
        password: str = "",
        client_id: str = "",
    ) -> None:
        try:
            import cbor2
            import paho.mqtt.client as mqtt
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "the G1 mission service needs paho-mqtt and cbor2: "
                "pip install -e '.[g1]'"
            ) from exc

        self._cbor = cbor2
        self.robot_id = robot_id
        self.broker = f"{broker}:{port}"
        self._lock = threading.Lock()
        self._latest: dict[str, tuple[Json, float]] = {}
        self._watchers: dict[str, list[Callable[[Json, float, bool], None]]] = {}
        self._subscribed: set[str] = set()
        self._connected = False

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id or default_client_id(robot_id),
            protocol=mqtt.MQTTv5,
        )
        if username:
            self._client.username_pw_set(username, password or None)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._mqtt = mqtt
        self._client.connect_async(broker, port, keepalive=30)
        self._client.loop_start()

    # ------------------------------------------------------------ callbacks --
    def _on_connect(self, _client, _userdata, _flags, _reason, _props=None) -> None:
        self._connected = True
        # Re-subscribe on every connect, not only the first: a reconnect after a
        # broker restart otherwise leaves a live-looking link delivering nothing.
        for topic in sorted(self._subscribed):
            self._client.subscribe(uplink(self.robot_id, topic), qos=1)

    def _on_disconnect(self, *_args, **_kwargs) -> None:
        self._connected = False

    def _on_message(self, _client, _userdata, message) -> None:
        prefix = uplink(self.robot_id, "")
        if not message.topic.startswith(prefix):
            return
        ros_topic = message.topic[len(prefix):]
        try:
            payload = self._cbor.loads(message.payload)
        except Exception:  # noqa: BLE001 - a malformed frame must not kill the loop
            return
        if not isinstance(payload, dict):
            return
        topic = "/" + ros_topic.lstrip("/")
        arrived = time.monotonic()
        with self._lock:
            self._latest[topic] = (payload, arrived)
            watchers = list(self._watchers.get(topic, ()))
        # Outside the lock: a watcher does real work (decoding a costmap patch)
        # and holding the link's lock through it would stall every other topic.
        # `retain` is passed on because a retained frame is the broker replaying
        # history, not the robot saying something now -- see MqttLocalCostmap.
        retained = bool(getattr(message, "retain", False))
        for watcher in watchers:
            try:
                watcher(payload, arrived, retained)
            except Exception:  # noqa: BLE001 - a bad watcher must not kill the loop
                continue

    # ---------------------------------------------------------------- Link ---
    def connected(self) -> bool:
        return self._connected

    def subscribe(self, ros_topic: str) -> None:
        with self._lock:
            if ros_topic in self._subscribed:
                return
            self._subscribed.add(ros_topic)
        if self._connected:
            self._client.subscribe(uplink(self.robot_id, ros_topic), qos=1)

    def subscribe_defaults(self) -> None:
        for topic in SUBSCRIBED:
            self.subscribe(topic)

    def latest(self, ros_topic: str) -> "tuple[Json, float] | None":
        with self._lock:
            return self._latest.get(ros_topic)

    def watch(
        self, ros_topic: str, callback: Callable[[Json, float, bool], None]
    ) -> None:
        """Call `callback(msg, arrival, retained)` for EVERY message on a topic.

        Subscribes as a side effect, so a caller needs one line rather than two.
        See the note on the Link protocol for why this exists alongside latest().
        """
        with self._lock:
            self._watchers.setdefault(ros_topic, []).append(callback)
        self.subscribe(ros_topic)

    def publish_cmd(self, ros_topic: str, msg: Json, expiry_s: int | None = None) -> None:
        # ##### THE ACCESS CONTROL, AND IT IS HERE RATHER THAN IN A PROMPT. #####
        # Only topics in wire.PUBLISHABLE can leave this process, so no tool, no
        # prompt injection and no future refactor upstream can reach /cmd_vel or
        # /estop through this link.
        if ros_topic not in PUBLISHABLE:
            raise ValueError(
                f"{ros_topic} is not publishable by the mission service; "
                f"allowed: {sorted(PUBLISHABLE)}"
            )
        if not self._connected:
            raise ConnectionError(f"no link to the broker at {self.broker}")
        properties = None
        expiry = PUBLISHABLE[ros_topic] if expiry_s is None else expiry_s
        if expiry:
            from paho.mqtt.packettypes import PacketTypes
            from paho.mqtt.properties import Properties

            properties = Properties(PacketTypes.PUBLISH)
            properties.MessageExpiryInterval = int(expiry)
        self._client.publish(
            downlink(self.robot_id, ros_topic),
            self._cbor.dumps(msg),
            qos=1,
            retain=False,
            properties=properties,
        )

    def close(self) -> None:
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:  # noqa: BLE001
            pass
