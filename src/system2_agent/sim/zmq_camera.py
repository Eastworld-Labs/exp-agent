from __future__ import annotations

from ..modules.camera import CameraFrame


class ZmqJpegCamera:
    """Camera client for GEAR-SONIC's MuJoCo image publisher."""

    def __init__(
        self,
        *,
        endpoint: str = "tcp://127.0.0.1:5555",
        camera: str = "ego_view",
        timeout_ms: int = 2000,
    ) -> None:
        try:
            import msgpack
            import zmq
        except ImportError as exc:
            raise ImportError("ZmqJpegCamera needs msgpack and pyzmq") from exc
        self._msgpack = msgpack
        self._zmq = zmq
        self.camera = camera
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(endpoint)

    def capture(self) -> list[CameraFrame]:
        if not self.socket.poll(self.timeout_ms, self._zmq.POLLIN):
            raise TimeoutError(f"no frame received for {self.timeout_ms} ms")
        message = self._msgpack.unpackb(self.socket.recv(), raw=False)
        images = message.get("images", {})
        encoded = images.get(self.camera) or message.get(self.camera)
        if not encoded:
            raise KeyError(f"camera {self.camera!r} absent; available={sorted(images)}")
        return [CameraFrame(self.camera, f"data:image/jpeg;base64,{encoded}")]

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()
