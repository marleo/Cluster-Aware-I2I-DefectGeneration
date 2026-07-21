from __future__ import annotations

import json
import mimetypes
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ComfyClient:
    def __init__(
        self,
        server_url: str,
        request_timeout: float = 30,
        generation_timeout: float = 300,
        poll_interval: float = 1.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.request_timeout = request_timeout
        self.generation_timeout = generation_timeout
        self.poll_interval = poll_interval
        self.client_id = str(uuid.uuid4())

    def _json_request(self, path: str, payload: dict | None = None) -> dict:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.server_url + path, data=data, headers=headers)
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as error:
            raise RuntimeError(f"ComfyUI request failed for {path}: {error}") from error

    def check_ready(self) -> dict:
        return self._json_request("/system_stats")

    def upload_image(self, image_path: Path, remote_name: str) -> str:
        boundary = f"----CodexBoundary{uuid.uuid4().hex}"
        mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        image_bytes = image_path.read_bytes()
        remote_path = Path(remote_name.replace("/", "\\"))
        upload_filename = remote_path.name
        upload_subfolder = str(remote_path.parent).replace("\\", "/")
        if upload_subfolder == ".":
            upload_subfolder = ""
        body = bytearray()

        def add_field(name: str, value: str) -> None:
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            )
            body.extend(value.encode())
            body.extend(b"\r\n")

        add_field("type", "input")
        add_field("overwrite", "true")
        add_field("subfolder", upload_subfolder)
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            (
                f'Content-Disposition: form-data; name="image"; '
                f'filename="{upload_filename}"\r\n'
            ).encode()
        )
        body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode())
        body.extend(image_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())

        request = Request(
            self.server_url + "/upload/image",
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urlopen(request, timeout=self.request_timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        subfolder = result.get("subfolder", "")
        filename = result["name"]
        return f"{subfolder}/{filename}".lstrip("/") if subfolder else filename

    def queue_prompt(self, workflow: dict) -> str:
        response = self._json_request(
            "/prompt",
            {"prompt": workflow, "client_id": self.client_id},
        )
        if "error" in response:
            raise RuntimeError(f"ComfyUI rejected the prompt: {response}")
        return response["prompt_id"]

    def wait_for_outputs(self, prompt_id: str) -> list[dict]:
        deadline = time.monotonic() + self.generation_timeout
        while time.monotonic() < deadline:
            history = self._json_request(f"/history/{prompt_id}")
            if prompt_id in history:
                record = history[prompt_id]
                status = record.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI generation failed: {status}")
                images: list[dict] = []
                for node_output in record.get("outputs", {}).values():
                    images.extend(node_output.get("images", []))
                if images:
                    return images
            time.sleep(self.poll_interval)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} exceeded timeout")

    def download_image(self, descriptor: dict, destination: Path) -> Path:
        query = urlencode(
            {
                "filename": descriptor["filename"],
                "subfolder": descriptor.get("subfolder", ""),
                "type": descriptor.get("type", "output"),
            }
        )
        request = Request(f"{self.server_url}/view?{query}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(request, timeout=self.request_timeout) as response:
            destination.write_bytes(response.read())
        return destination
