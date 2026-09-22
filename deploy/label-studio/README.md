# Label Studio on the Sortie capture computer

The Pi/mini PC runs Sortie and Label Studio. Open Sortie in a browser **on that
computer** at `http://localhost:5000` for its attached camera. From another
computer, open Label Studio's private Tailscale HTTPS address to annotate.
Your Mac is not a server and can be turned off. The host must stay powered on;
remote annotation needs its internet connection. Capture and local uploads
continue when the internet is down.

These files target a 64-bit Linux host (ARM64 Pi or x86-64 mini PC), systemd and
Python 3.11; Debian 12 / Raspberry Pi OS Bookworm 64-bit provides Python 3.11.
Hardware installation and remote verification must be performed when the host
is available. The pinned Python packages are not a claim that this project's
model has been benchmarked on a Pi.

## 1. Install on the host

Copy this repository to `/opt/sortie`, including your separately stored
`model.tflite`. Run these commands on the Linux host, not on your Mac:

```sh
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv build-essential
sudo useradd --system --user-group --home-dir /var/lib/label-studio --shell /usr/sbin/nologin labelstudio
sudo useradd --system --user-group --home-dir /var/lib/sortie --shell /usr/sbin/nologin sortie
sudo install -d -o labelstudio -g labelstudio -m 700 /var/lib/label-studio
sudo install -d -o sortie -g sortie -m 700 /var/lib/sortie /var/lib/sortie/training
sudo install -d -m 755 /opt/label-studio
sudo install -d -m 700 /etc/sortie
sudo python3.11 -m venv /opt/label-studio/venv
sudo /opt/label-studio/venv/bin/pip install -r /opt/sortie/deploy/label-studio/requirements.txt
sudo python3.11 -m venv /opt/sortie/venv
sudo /opt/sortie/venv/bin/pip install -r /opt/sortie/requirements.txt
sudo /opt/sortie/venv/bin/python /opt/sortie/tools/fetch_hand_assets.py
sudo install -m 600 /opt/sortie/deploy/label-studio/label-studio.env.example /etc/sortie/label-studio.env
sudo install -m 600 /opt/sortie/deploy/label-studio/sortie.env.example /etc/sortie/sortie.env
sudo install -m 644 /opt/sortie/deploy/label-studio/label-studio.service /etc/systemd/system/
sudo install -m 644 /opt/sortie/deploy/label-studio/sortie.service /etc/systemd/system/
sudo systemctl daemon-reload
```

Skip `useradd` for an account already created by these instructions. The service
accounts must be able to read `/opt/sortie`, its assets, model and environments.
Do not copy credentials into the repository. Use the separate environments above
to avoid dependency conflicts between Label Studio and inference.

If carrying a pre-existing queue to this device, stop the old Sortie process,
copy its **entire** `data/training` directory to `/var/lib/sortie/training`, and
set ownership to `sortie:sortie` before starting the new service. Keep a backup.
Do not run the old Roboflow uploader and the new uploader against one database.

## 2. Set up private HTTPS

Install Tailscale on the host and every computer you will annotate from using
the [official installation instructions](https://tailscale.com/download).
Join the devices to the same Tailscale network. On the Linux host:

```sh
sudo systemctl enable --now tailscaled
sudo tailscale up
sudo tailscale serve --bg http://127.0.0.1:8080
tailscale serve status
```

Follow any prompt to enable HTTPS certificates. Copy the generated
`https://HOST.TAILNET.ts.net` address. Keep it stable: uploaded media links may
refer to this host. No router port forwarding or public domain is needed.
[Serve](https://tailscale.com/docs/reference/tailscale-cli/serve) is private to
your Tailscale network and its background configuration persists across reboots.

Use `sudoedit /etc/sortie/label-studio.env` to set `LABEL_STUDIO_HOST` and
`CSRF_TRUSTED_ORIGINS` to that exact HTTPS origin, with no trailing slash.
Set `LABEL_STUDIO_PUBLIC_URL` in `/etc/sortie/sortie.env` to the same origin.
Leave `LABEL_STUDIO_URL=http://127.0.0.1:8080` for local API uploads.

For the initial account only, temporarily set
`DISABLE_SIGNUP_WITHOUT_LINK=false` in `label-studio.env`. Then:

```sh
sudo systemctl enable --now label-studio
```

Open the HTTPS address from your Tailscale-connected annotation computer and
create your account with a strong password. Set
`DISABLE_SIGNUP_WITHOUT_LINK=true` again and run:

```sh
sudo systemctl restart label-studio
```

New users then need an invitation link from Label Studio. Keep those links
private. Both Tailscale access and a Label Studio login are required. The secure
cookie settings mean you should use HTTPS for login, including on the host.

## 3. Create the project and connect Sortie

1. Create a project named **Sortie**. In **Settings → Labeling Interface → Code**,
   paste [label-config.xml](label-config.xml) and save it. It defines one image
   and rectangle labels `glass`, `paper`, `plastic`, `waste`.
2. Note the numeric project ID in `/projects/1/data/` (for example, `1`).
3. In **Account & Settings**, create a **Personal Access Token**. Put it in
   `LABEL_STUDIO_API_KEY` in `/etc/sortie/sortie.env`, and set
   `LABEL_STUDIO_PROJECT_ID` to your actual project ID. The SDK exchanges and
   refreshes access tokens automatically. Replace the PAT when it expires or is
   revoked. Do not enable legacy tokens for this setup.
4. Set `SORTIE_MODEL` to your model path. To check the UI before the model is
   ready, explicitly add `SORTIE_MOCK=1`; remove it before real classification.
5. Start Sortie:

   ```sh
   sudo systemctl enable --now sortie
   ```

The systemd services load the environment files. For manual development,
export the same variables in your shell; `.env` files are not loaded
automatically. Start with `python app.py` for local development only.

## 4. Verify before collecting a large session

1. On the host, open `http://localhost:5000`, allow the camera, select Training,
   move a held item, and confirm **Uploaded** increases.
2. From a second computer on a different network (for example, a hotspot),
   connect Tailscale, open the annotation HTTPS URL and log in. Open the new task:
   the image must load with no pre-existing boxes. Draw a box, choose a label,
   submit, reload, and verify the box remains.
3. Close the capture browser and confirm the uploaded image remains available.
4. Stop Label Studio, collect a capture, and verify it remains pending. Start
   Label Studio and verify automatic upload recovery.
5. Reboot the host. Verify both services, remote login, saved images and
   annotations, and any pending uploads. The local camera browser must be
   reopened after a reboot; automatic desktop/kiosk login is device-specific.

Useful diagnostics:

```sh
sudo systemctl status label-studio sortie
sudo journalctl -u label-studio -u sortie -n 100 --no-pager
tailscale serve status
curl -fsS http://127.0.0.1:5000/training/status
```

If login gives a CSRF error, check that both external-origin settings exactly
match the browser's HTTPS address. If images do not load, check that
`LABEL_STUDIO_HOST` uses the remote address, never `localhost`. If uploads fail,
check the token, project ID and labeling configuration; restart Sortie after
editing its environment, then choose **Retry uploads**.

## Backup and restore

Uploaded JPEGs live with Label Studio; Sortie deletes its queued copy after a
confirmed task ID. Exporting annotations alone is not an image backup.
To get a consistent backup, stop both services and archive all data plus the
root-only configuration files (which contain the API token):

```sh
sudo systemctl stop sortie label-studio
sudo install -d -m 700 /var/backups/sortie
sudo sh -c 'umask 077; tar -czf /var/backups/sortie/annotation-backup.tar.gz -C / var/lib/label-studio var/lib/sortie etc/sortie'
sudo systemctl start label-studio sortie
```

Copy the archive to another secure device; use a new filename for later backups
to retain older versions. Back up the model separately. To restore on a prepared
host, stop the services, archive its current data first, extract the backup at
`/`, and restore ownership of `/var/lib/label-studio` to `labelstudio:labelstudio`
and `/var/lib/sortie` to `sortie:sortie`. Keep `/etc/sortie` root-only. Start the
same pinned server version, reconnect Tailscale, and repeat the remote image and
annotation check. Keep the same HTTPS hostname; changing it can require updating
existing task media URLs.

Reference: [Label Studio installation](https://labelstud.io/guide/install.html),
[server configuration](https://labelstud.io/guide/start),
[access tokens](https://labelstud.io/guide/access_tokens).
