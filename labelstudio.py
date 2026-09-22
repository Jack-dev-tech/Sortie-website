"""Label Studio Community Edition adapter; no inference or automatic labels."""
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit

import httpx
from label_studio_sdk import LabelStudio
from label_studio_sdk.core.api_error import ApiError

LABELS = {"glass", "paper", "plastic", "waste"}
SETUP = ("Set LABEL_STUDIO_URL, LABEL_STUDIO_API_KEY and LABEL_STUDIO_PROJECT_ID "
         "on the server, then restart. Set LABEL_STUDIO_PUBLIC_URL for remote annotation.")


class UploadError(Exception):
    def __init__(self, message, permanent=False):
        super().__init__(message)
        self.permanent = permanent


def valid_url(value):
    try:
        url = urlsplit(value)
        return (url.scheme in ("http", "https") and bool(url.hostname) and
                not url.username and not url.password and not url.query and
                not url.fragment and url.path in ("", "/") and
                not any(c.isspace() for c in value) and
                (url.port is None or 0 < url.port < 65536))
    except ValueError:
        return False


def check_status(status):
    if status in (401, 403):
        raise UploadError("Label Studio authentication failed. Check the server token and project access, restart, then retry uploads.", True)
    if status in (408, 429) or status >= 500:
        raise UploadError("Label Studio is temporarily unavailable or rate limited. Retrying automatically.")
    if not 200 <= status < 300:
        raise UploadError("Label Studio rejected the upload. Check its URL, project and labeling configuration, then retry uploads.", True)


def validate_config(config):
    try:
        root = ET.fromstring(config)
        images = list(root.iter("Image"))
        boxes = list(root.iter("RectangleLabels"))
        if (len(images) != 1 or images[0].get("value") != "$image" or
                len(boxes) != 1 or not images[0].get("name") or
                boxes[0].get("toName") != images[0].get("name") or
                {x.get("value") for x in boxes[0].findall("Label")} != LABELS):
            raise ValueError()
    except (ET.ParseError, TypeError, ValueError):
        raise UploadError("Use the supplied Label Studio bounding-box configuration with glass, paper, plastic and waste, then retry uploads.", True) from None


class LabelStudioUploader:
    def __init__(self, url, key):
        self.url, self.key = url, key
        self.client = None

    def upload(self, row):
        try:
            # Construct lazily so an expired token cannot stop the Sortie app starting.
            if self.client is None:
                self.client = LabelStudio(base_url=self.url + "/", api_key=self.key,
                                          timeout=45, max_retries=0)
            project = self.client.projects.get(int(row["project"]))
            validate_config(project.label_config)
            name = f"sortie-{row['session']}-{row['id']}.jpg"
            # SDK 2.1.1's public import_tasks only accepts JSON. Use its authenticated
            # transport for multipart imports, preserving SDK PAT refresh. This is
            # intentionally isolated and covered by a real-SDK transport test.
            response = self.client._client_wrapper.httpx_client.request(
                path=f"api/projects/{row['project']}/import", method="POST",
                params={"return_task_ids": "true"},
                files={"file": (name, row["jpeg"], "image/jpeg")},
                request_options={"max_retries": 0, "timeout": 45})
            check_status(response.status_code)
            data = response.json()
            task_ids = data.get("task_ids") if isinstance(data, dict) else None
            if (not isinstance(task_ids, list) or len(task_ids) != 1 or
                    type(task_ids[0]) is not int or task_ids[0] <= 0):
                raise UploadError("Label Studio did not confirm one task ID. Check the Community Edition project before retrying; the image may already exist.", True)
            return task_ids[0]
        except ApiError as exc:
            check_status(exc.status_code or 503)
            raise UploadError("Label Studio returned an unexpected response. Retrying automatically.") from None
        except (httpx.HTTPError, ValueError):
            # Never propagate upstream response bodies or URLs containing secrets.
            raise UploadError("Label Studio is unreachable or returned an invalid response. Retrying automatically.") from None
