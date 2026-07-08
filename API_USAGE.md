# DeepFilterNet Speech Enhancement API — Usage (Postman)

Upload a noisy audio file and get a denoised **48 kHz 16-bit WAV** back.

## Main endpoint

```
POST /api/v1/enhance
```

Base URL (default): `http://localhost:8000`

Full URL: `http://localhost:8000/api/v1/enhance`

No authentication is required.

---

## Postman setup

1. **Method**: `POST`
2. **URL**: `http://localhost:8000/api/v1/enhance`
3. **Body** tab → select **form-data**.
4. Add one key:

   | Key    | Type | Value                          |
   |--------|------|--------------------------------|
   | `file` | File | Select your audio file (click the dropdown next to the key name and switch from *Text* to *File*) |

   > The key **must** be named exactly `file`.

5. Do **not** set `Content-Type` manually — Postman adds the correct
   `multipart/form-data` boundary automatically.
6. Click **Send**.

### Supported input formats
`.wav`, `.flac`, `.ogg`, `.mp3` — any sample rate (audio is resampled to 48 kHz internally).

Max upload size: **50 MB** (configurable via `MAX_UPLOAD_BYTES`).

---

## Response

- **Status**: `200 OK`
- **Content-Type**: `audio/wav`
- **Body**: the raw enhanced WAV file (48 kHz, 16-bit PCM).

In Postman, the response pane shows the binary body. To save it:
**Send → click "Save Response" → "Save to a file"**, then name it e.g. `enhanced.wav`.

The server also sets a download filename via the `Content-Disposition` header
(`enhanced_<originalname>.wav`).

---

## Error responses (JSON)

| Status | Meaning                                                        |
|--------|----------------------------------------------------------------|
| `400`  | Empty file upload.                                             |
| `413`  | File larger than the configured limit (default 50 MB).        |
| `415`  | Unsupported file type (not wav/flac/ogg/mp3).                 |
| `500`  | Decoding or enhancement failed.                               |

Example error body:

```json
{ "detail": "Unsupported file type '.m4a'. Allowed: ['.flac', '.mp3', '.ogg', '.wav']" }
```

---

## Quick checks

- **Health probe**: `GET http://localhost:8000/health`
  ```json
  { "status": "ok", "model": "DeepFilterNet3", "device": "cpu", "sample_rate": 48000 }
  ```
- **Model info**: `GET http://localhost:8000/api/v1/info`
- **Interactive docs**: open `http://localhost:8000/docs` in a browser.
