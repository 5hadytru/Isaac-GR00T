from dataclasses import dataclass
import io
from typing import Any, Callable

import msgpack
import numpy as np
import zmq

from gr00t.data.types import ModalityConfig
from gr00t.data.utils import to_json_serializable

from .policy import BasePolicy


# image compression stuff
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple, Optional
import base64
import numpy as np

def _is_uint8_hwc3(x: Any) -> bool:
    return (
        isinstance(x, np.ndarray)
        and x.dtype == np.uint8
        and x.ndim == 3
        and x.shape[2] == 3
    )

def decompress_obs_images(
    obs: Dict[str, Any],
    *,
    image_keys: Iterable[str] = ("front", "overhead"),
    base64_encoded: Optional[bool] = None,  # None = auto-detect if str vs bytes
) -> Dict[str, Any]:
    """
    Inverse of compress_obs_images(): returns obs with compressed payloads restored to np.uint8 HxWx3 arrays.
    """
    out: Dict[str, Any] = dict(obs)

    try:
        import cv2  # type: ignore
        use_cv2 = True
    except Exception:
        use_cv2 = False

    for k in image_keys:
        if k not in obs:
            continue
        payload = obs[k]
        if not (isinstance(payload, dict) and payload.get("__compressed__") is True):
            continue

        codec = str(payload.get("codec", "jpg")).lower()
        data = payload.get("data")
        color = payload.get("color", "rgb")  # original color convention
        if data is None:
            raise ValueError(f"Missing data for compressed key={k}")

        if base64_encoded is None:
            is_b64 = isinstance(data, str)
        else:
            is_b64 = base64_encoded

        if is_b64:
            data_bytes = base64.b64decode(data.encode("ascii"))
        else:
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise TypeError(f"Expected bytes-like for key={k}, got {type(data)}")
            data_bytes = bytes(data)

        if use_cv2:
            import numpy as _np
            buf = _np.frombuffer(data_bytes, dtype=_np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"cv2.imdecode failed for key={k}, codec={codec}")
            # cv2 returns BGR; convert back to RGB if original was RGB
            if color == "rgb":
                img = img[..., ::-1]  # BGR -> RGB
            out[k] = img.astype(np.uint8, copy=False)

        else:
            from io import BytesIO
            from PIL import Image  # type: ignore

            im = Image.open(BytesIO(data_bytes)).convert("RGB")
            arr = np.array(im, dtype=np.uint8)
            # arr is RGB; if original was BGR, flip back
            if color == "bgr":
                arr = arr[..., ::-1]
            out[k] = arr

    return out
# image compression stuff

class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return ModalityConfig(**obj["as_json"])
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig_class__": True, "as_json": to_json_serializable(obj)}
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class PolicyServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(
        self, policy: BasePolicy, host: str = "*", port: int = 5555, api_token: str = None
    ):
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)
        self.register_endpoint("get_action", self.policy.get_action)
        self.register_endpoint("reset", self.policy.reset)
        self.register_endpoint(
            "get_modality_config",
            getattr(self.policy, "get_modality_config", lambda: {}),
            requires_input=False,
        )

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict) -> bool:
        """
        Validate the API token in the request.
        """
        if self.api_token is None:
            return True  # No token required
        return request.get("api_token") == self.api_token

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = MsgSerializer.from_bytes(message)

                # Validate token before processing request
                if not self._validate_token(request):
                    self.socket.send(
                        MsgSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                    )
                    continue

                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(**request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))

    @staticmethod
    def start_server(policy: BasePolicy, port: int, api_token: str = None):
        server = PolicyServer(policy, port=port, api_token=api_token)
        server.run()


class PolicyClient(BasePolicy):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
        strict: bool = False,
    ):
        super().__init__(strict=strict)
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure we are running the correct policy server.")
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:

        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # Convert list (from msgpack) to tuple of (action, info)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def check_observation(self, observation: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_observation is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )

    def check_action(self, action: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_action is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )
