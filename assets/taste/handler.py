import os
import json
import time
import uuid
import math
import base64
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import boto3
from boto3.dynamodb.conditions import Attr

from PIL import Image
import pillow_heif
from io import BytesIO

pillow_heif.register_heif_opener()

PROTOTYPE_FILE = Path(os.environ.get("PROTOTYPE_FILE", "/var/task/axis_prototypes.json"))
PROTOTYPE_CACHE = None

TABLE_NAME = os.environ["TABLE_NAME"]
IMAGE_BUCKET = os.environ["IMAGE_BUCKET"]
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "amazon.nova-2-multimodal-embeddings-v1:0",
)
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v3")

UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", "/var/task/uploads"))

K_NEIGHBORS = int(os.environ.get("K_NEIGHBORS", "10"))
MAX_TRAINING_SCAN = int(os.environ.get("MAX_TRAINING_SCAN", "200"))

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(TABLE_NAME)
s3 = boto3.client("s3", region_name=AWS_REGION)
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,GET,POST",
        },
        "body": json.dumps(body),
    }

def to_decimal(value):
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: to_decimal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_decimal(v) for v in value]
    return value

def normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec

def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))

def weighted_average_vectors(vectors):
    length = len(vectors[0][0])
    out = [0.0] * length
    total = 0.0
    for vec, w in vectors:
        total += w
        for i, x in enumerate(vec):
            out[i] += x * w
    return [x / total for x in out] if total else out

# ✅ RESTORED
def infer_image_format(image_mime_type, image_b64):
    if image_mime_type:
        t = image_mime_type.lower()
        if "heic" in t or "heif" in t: return "heic"
        if "jpeg" in t or "jpg" in t: return "jpeg"
        if "png" in t: return "png"
        if "webp" in t: return "webp"
        if "gif" in t: return "gif"

    if image_b64 and image_b64.startswith("data:image/"):
        prefix = image_b64.split(";")[0].lower()
        if "heic" in prefix or "heif" in prefix: return "heic"
        if "jpeg" in prefix or "jpg" in prefix: return "jpeg"
        if "png" in prefix: return "png"
        if "webp" in prefix: return "webp"
        if "gif" in prefix: return "gif"

    return "jpeg"

def strip_data_url_prefix(b64):
    return b64.split(",", 1)[1] if b64.startswith("data:") else b64

def normalize_image(image_b64, fmt):
    clean = strip_data_url_prefix(image_b64)
    bytes_ = base64.b64decode(clean)

    if fmt != "heic":
        return image_b64, fmt

    img = Image.open(BytesIO(bytes_))
    out = BytesIO()
    img.convert("RGB").save(out, format="JPEG")
    return base64.b64encode(out.getvalue()).decode(), "jpeg"

def embed_text(text):
    body = {
        "schemaVersion": "nova-multimodal-embed-v1",
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "CLASSIFICATION",
            "embeddingDimension": 384,
            "text": {"truncationMode": "END", "value": text},
        },
    }
    res = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(res["body"].read())["embeddings"][0]["embedding"]

def embed_image_base64(image_b64, fmt):
    body = {
        "schemaVersion": "nova-multimodal-embed-v1",
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "CLASSIFICATION",
            "embeddingDimension": 384,
            "image": {
                "detailLevel": "STANDARD_IMAGE",
                "format": fmt,
                "source": {"bytes": strip_data_url_prefix(image_b64)},
            },
        },
    }
    res = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(res["body"].read())["embeddings"][0]["embedding"]

def get_axis_embeddings():
    global PROTOTYPE_CACHE
    if PROTOTYPE_CACHE is not None:
        return PROTOTYPE_CACHE

    with PROTOTYPE_FILE.open("r") as f:
        data = json.load(f)

    PROTOTYPE_CACHE = data["axes"]
    return PROTOTYPE_CACHE

def compute_axis_features(vec):
    bank = get_axis_embeddings()
    features = {}
    for axis, emb in bank.items():
        pos = [cosine(vec, v) for v in emb["positive"]]
        neg = [cosine(vec, v) for v in emb["negative"]]
        features[f"{axis}__gap"] = sum(pos)/len(pos) - sum(neg)/len(neg)
    return features

def fallback_predict_score(f):
    raw = sum(f.values())
    raw = 1/(1+math.exp(-2.2*raw))
    score = int(round(1+4*raw))
    label_map = {1:"No",2:"Meh",3:"Yes-",4:"Yes",5:"YES"}
    return raw, score, label_map[score]

# ✅ NEW
def predict_with_retrieval(dish_embedding):
    res = table.scan(
        FilterExpression=Attr("record_type").eq("training_example"),
        Limit=MAX_TRAINING_SCAN,
    )

    scored = []
    for item in res.get("Items", []):
        emb = item.get("dish_embedding")
        score = item.get("actual_score_1_to_5")

        if emb and score is not None:
            emb = [float(x) for x in emb]  
            score = float(score)         

            sim = cosine(dish_embedding, emb)
            scored.append((sim, score))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:K_NEIGHBORS]

    num = sum(max(s,0)*v for s,v in top)
    den = sum(max(s,0) for s,_ in top)

    return num/den if den else None

def model_predict_score(fmap, dish_embedding):
    r = predict_with_retrieval(dish_embedding)
    raw_h, score_h, label_h = fallback_predict_score(fmap)

    if r is None:
        return raw_h, score_h, label_h, "fallback_only"

    blended = max(1, min(5, 0.7*r + 0.3*score_h))
    raw = (blended-1)/4
    score = int(round(blended))
    label_map = {1:"No",2:"Meh",3:"Yes-",4:"Yes",5:"YES"}
    return raw, score, label_map[score], "retrieval_blend"

def build_dish_vector(description, image_b64, image_mime_type):
    text = normalize(embed_text(description))

    if not image_b64:
        return text, None, False, None

    fmt = infer_image_format(image_mime_type, image_b64)
    image_b64, fmt = normalize_image(image_b64, fmt)
    img = normalize(embed_image_base64(image_b64, fmt))

    return normalize(weighted_average_vectors([(text,0.65),(img,0.35)])), img, True, fmt

# ===== HANDLERS (UNCHANGED) =====

def handle_predict(body):
    title = (body.get("title") or "").strip()
    description = (body.get("description") or "").strip()

    if not title or not description:
        return response(400, {"error": "title and description are required"})

    # ✅ ADD THIS BACK
    entry_id = str(uuid.uuid4())
    now = int(time.time())

    dish_embedding, image_embedding, has_image, image_format = build_dish_vector(
        description,
        body.get("image_base64"),
        body.get("image_mime_type"),
    )

    axis_features = compute_axis_features(dish_embedding)

    raw, score, label, source = model_predict_score(axis_features, dish_embedding)

    # NOT STORED DURING PREDICTION
    item = {
        "id": entry_id,
        "record_type": "prediction",
        "model_version": MODEL_VERSION,
        "title": title,
        "description": description,
        "dish_embedding": dish_embedding,
        "axis_features": axis_features,
        "predicted_score_1_to_5": score,
        "predicted_label": label,
        "prediction_source": source,
        "created_at": now,
        "updated_at": now,
    }

    # Don't store to DynamoDB here, wait for validation
    return response(200, to_decimal(item))

def handle_validate(body: dict) -> dict:
    if body.get("password") != os.environ.get("ADMIN_PASSWORD", "secret123"):
        return response(401, {"error": "Invalid password"})

    item = body.get("item")
    actual_score = body.get("actual_score")
    contributors = body.get("contributors", "")
    notes = body.get("notes", "")

    if not item or actual_score is None:
        return response(400, {"error": "item and actual_score are required"})

    try:
        actual_score = float(actual_score)
    except Exception:
        return response(400, {"error": "actual_score must be numeric from 1 to 5"})

    if actual_score < 1 or actual_score > 5:
        return response(400, {"error": "actual_score must be between 1 and 5"})

    now = int(time.time())

    item["actual_score_1_to_5"] = actual_score
    item["contributors"] = contributors
    item["notes"] = notes
    item["updated_at"] = now
    
    table.put_item(Item=to_decimal(item))

    return response(200, {"ok": True})

def handle_search(body: dict) -> dict:
    query = (body.get("query") or "").lower()
    try:
        res = table.scan()
        items = []
        for it in res.get("Items", []):
            title = it.get("title", "").lower()
            if query in title:
                items.append(it)
        return response(200, {"items": to_decimal(items[:50])})
    except Exception as e:
        return response(500, {"error": str(e)})

# KEEP your other handlers EXACTLY unchanged (validate, upload, get_entry)

def lambda_handler(event, context):
    try:
        method = (
            event.get("requestContext", {})
            .get("http", {})
            .get("method")
            or event.get("httpMethod")
        )
        path = event.get("rawPath") or event.get("path") or ""

        if method == "OPTIONS":
            return response(200, {"ok": True})

        if method == "POST" and path.endswith("/predict"):
            return handle_predict(json.loads(event.get("body") or "{}"))

        if method == "POST" and path.endswith("/validate"):
            return handle_validate(json.loads(event.get("body") or "{}"))

        if method == "POST" and path.endswith("/search"):
            return handle_search(json.loads(event.get("body") or "{}"))

        if method == "POST" and path.endswith("/upload-training-data"):
            return handle_upload_training_data(json.loads(event.get("body") or "{}"))

        if method == "GET" and path.endswith("/entry"):
            return handle_get_entry(event)

        return response(404, {"error": "route not found"})
    except Exception as e:
        return response(500, {"error": str(e)})

