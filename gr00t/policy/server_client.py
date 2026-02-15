import os
import zlib
from dataclasses import dataclass
import io
from typing import Any, Callable

import msgpack
import numpy as np
import zmq

from gr00t.data.types import ModalityConfig
from gr00t.data.utils import to_json_serializable

from .policy import BasePolicy

from time import perf_counter

class MsgSerializer:
    """MsgPack serializer with special-cases for numpy arrays.

    This compresses *image-like* uint8 ndarrays (H×W or H×W×C) into a byte payload
    on the sender, and recreates the ndarray on the receiver.

    Environment variables:
      - GROOT_IMG_CODEC: 'jpg' (default), 'webp', 'png', or 'zlib_raw'
      - GROOT_JPEG_QUALITY: 0-100 (default 80)
      - GROOT_WEBP_QUALITY: 0-100 (default 80)
    """

    # Optional fast codecs (recommended on BOTH sides)
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None  # type: ignore

    # Fallback codec (works, but note RGB/BGR caveat below)
    try:
        from PIL import Image  # type: ignore
    except Exception:
        Image = None  # type: ignore

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def _is_probably_image(arr: np.ndarray) -> bool:
        if arr.dtype != np.uint8:
            return False
        if arr.ndim == 2:
            h, w = arr.shape
            return h >= 32 and w >= 32 and (h * w) >= 32_000
        if arr.ndim == 3:
            h, w, c = arr.shape
            if c not in (1, 3, 4):
                return False
            return h >= 32 and w >= 32 and (h * w) >= 32_000
        return False

    @staticmethod
    def _encode_image(arr: np.ndarray) -> dict:
        codec = os.environ.get("GROOT_IMG_CODEC", "jpg").lower()
        jpeg_q = int(os.environ.get("GROOT_JPEG_QUALITY", "80"))
        webp_q = int(os.environ.get("GROOT_WEBP_QUALITY", "80"))

        # Some encoders want (H,W) not (H,W,1)
        arr_for_codec = arr[:, :, 0] if (arr.ndim == 3 and arr.shape[2] == 1) else arr

        # 1) OpenCV (best if your frames are from OpenCV, i.e. BGR)
        if MsgSerializer.cv2 is not None and codec in ("jpg", "jpeg", "png", "webp"):
            ext = ".jpg" if codec in ("jpg", "jpeg") else f".{codec}"
            params = []
            if codec in ("jpg", "jpeg"):
                params = [int(MsgSerializer.cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)]
            elif codec == "webp":
                params = [int(MsgSerializer.cv2.IMWRITE_WEBP_QUALITY), int(webp_q)]

            ok, enc = MsgSerializer.cv2.imencode(ext, arr_for_codec, params)
            if ok:
                return {
                    "__ndarray_img__": True,
                    "codec": "jpg" if codec == "jpeg" else codec,
                    "shape": arr.shape,
                    "data": enc.tobytes(),
                }

        # 2) Pillow fallback (portable, but assumes RGB semantics for color images)
        if MsgSerializer.Image is not None and codec in ("jpg", "jpeg", "png", "webp"):
            bio = io.BytesIO()
            pil_mode = "L" if arr_for_codec.ndim == 2 else ("RGBA" if arr.shape[-1] == 4 else "RGB")
            im = MsgSerializer.Image.fromarray(arr_for_codec, mode=pil_mode)
            if codec in ("jpg", "jpeg"):
                im.save(bio, format="JPEG", quality=jpeg_q, optimize=True)
            elif codec == "png":
                im.save(bio, format="PNG", optimize=True)
            else:  # webp
                im.save(bio, format="WEBP", quality=webp_q)
            return {
                "__ndarray_img__": True,
                "codec": "jpg" if codec == "jpeg" else codec,
                "shape": arr.shape,
                "data": bio.getvalue(),
            }

        # 3) Last resort: exact bytes + zlib (no extra deps, weaker compression)
        raw = arr.tobytes(order="C")
        comp = zlib.compress(raw, level=6)
        return {
            "__ndarray_img__": True,
            "codec": "zlib_raw",
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "data": comp,
        }

    @staticmethod
    def _decode_image(obj: dict) -> np.ndarray:
        codec = obj.get("codec", "jpg")
        shape = obj.get("shape")
        if isinstance(shape, list):
            shape = tuple(shape)
        data = obj.get("data", b"")

        if codec == "zlib_raw":
            dtype = np.dtype(obj.get("dtype", "uint8"))
            raw = zlib.decompress(data)
            arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
            return arr

        # OpenCV decode (recommended)
        if MsgSerializer.cv2 is not None and codec in ("jpg", "png", "webp"):
            buf = np.frombuffer(data, dtype=np.uint8)
            img = MsgSerializer.cv2.imdecode(buf, MsgSerializer.cv2.IMREAD_UNCHANGED)
            if isinstance(shape, tuple) and len(shape) == 3 and shape[2] == 1 and img.ndim == 2:
                img = img[:, :, None]
            return img

        # Pillow fallback
        if MsgSerializer.Image is not None and codec in ("jpg", "png", "webp"):
            im = MsgSerializer.Image.open(io.BytesIO(data))
            arr = np.array(im)
            if isinstance(shape, tuple) and len(shape) == 3 and shape[2] == 1 and arr.ndim == 2:
                arr = arr[:, :, None]
            return arr

        raise RuntimeError(
            f"Unsupported image codec '{codec}'. Install opencv-python(-headless) or pillow."
        )

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return ModalityConfig(**obj["as_json"])
        if "__ndarray_img__" in obj:
            return MsgSerializer._decode_image(obj)
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig_class__": True, "as_json": to_json_serializable(obj)}
        if isinstance(obj, np.ndarray):
            if MsgSerializer._is_probably_image(obj):
                return MsgSerializer._encode_image(obj)
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

                t_dec0 = perf_counter()
                request = MsgSerializer.from_bytes(message)
                t_dec1 = perf_counter()

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
                t_h0 = perf_counter()
                result = (
                    handler.handler(**request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                t_h1 = perf_counter()

                # Attach server timing to info for get_action
                if endpoint == "get_action" and isinstance(result, (list, tuple)) and len(result) == 2:
                    action, info = result
                    if not isinstance(info, dict):
                        info = {"info": info}
                    info.setdefault("_timing", {})["server"] = {
                        "decode_s": t_dec1 - t_dec0,
                        "handler_s": t_h1 - t_h0,  # "forward pass" bucket
                    }
                    result = (action, info)

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

        t0 = perf_counter()
        req_bytes = MsgSerializer.to_bytes(request)
        t1 = perf_counter()

        self.socket.send(req_bytes)
        t2 = perf_counter()

        message = self.socket.recv()
        t3 = perf_counter()

        response = MsgSerializer.from_bytes(message)
        t4 = perf_counter()

        ct = {
            "pack_s": t1 - t0,
            "wait_s": t3 - t2,    # network + server compute + network
            "unpack_s": t4 - t3,
            "total_s": t4 - t0,
        }

        # If this is get_action, stash client timing into the returned info dict too
        server_t = None
        if endpoint == "get_action" and isinstance(response, (list, tuple)) and len(response) == 2:
            action, info = response
            if not isinstance(info, dict):
                info = {"info": info}
            info.setdefault("_timing", {})["client"] = ct
            server_t = info["_timing"].get("server")
            response = [action, info]  # keep it mutable/consistent

        # Print a single line per get_action
        if os.environ.get("GROOT_TIMING_PRINT", "0") == "1" and endpoint == "get_action":
            if server_t:
                forward_ms = server_t["handler_s"] * 1e3
                # "network-ish" ≈ client wait minus server decode+handler (doesn't include client pack/unpack)
                net_ms = (ct["wait_s"] - (server_t["decode_s"] + server_t["handler_s"])) * 1e3
                print(
                    f"[timing] pack={ct['pack_s']*1e3:.1f}ms "
                    f"wait={ct['wait_s']*1e3:.1f}ms "
                    f"unpack={ct['unpack_s']*1e3:.1f}ms | "
                    f"forward≈{forward_ms:.1f}ms net≈{net_ms:.1f}ms total={ct['total_s']*1e3:.1f}ms"
                )
            else:
                print(
                    f"[timing] pack={ct['pack_s']*1e3:.1f}ms "
                    f"wait={ct['wait_s']*1e3:.1f}ms "
                    f"unpack={ct['unpack_s']*1e3:.1f}ms total={ct['total_s']*1e3:.1f}ms"
                )


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
