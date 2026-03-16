import os
import json
import time
import uuid
import math
import base64
import pickle
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import boto3
from boto3.dynamodb.conditions import Attr


TABLE_NAME = os.environ["TABLE_NAME"]
IMAGE_BUCKET = os.environ["IMAGE_BUCKET"]
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "amazon.nova-2-multimodal-embeddings-v1:0",
)
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v2")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/var/task/models"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(MODEL_DIR / "model.pkl")))
FEATURE_SCHEMA_PATH = Path(os.environ.get("FEATURE_SCHEMA_PATH", str(MODEL_DIR / "feature_schema.json")))
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", "/var/task/uploads"))

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(TABLE_NAME)
s3 = boto3.client("s3", region_name=AWS_REGION)
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

AXIS_BANK = {
    "savory_depth": {
        "positive": [
            "deep savory concentrated flavor",
            "brothy umami-rich taste",
            "long-cooked meaty savory character",
        ],
        "negative": [
            "thin low-impact flavor",
            "watery bland savory profile",
            "little lingering taste",
        ],
    },
    "richness": {
        "positive": [
            "rich fatty creamy buttery oily texture",
            "coating luxurious richness",
            "dense mouthfilling body",
        ],
        "negative": [
            "lean dry thin texture",
            "light low fat mouthfeel",
            "not rich or creamy",
        ],
    },
    "acidity": {
        "positive": [
            "bright acidic tangy sharp taste",
            "citrusy vinegary pickled brightness",
            "clean sour lift",
        ],
        "negative": [
            "mellow non acidic taste",
            "soft rounded low brightness",
            "neutral not tangy",
        ],
    },
    "allium": {
        "positive": [
            "strong onion garlic scallion aroma",
            "savory sulfurous allium intensity",
            "pungent aromatic onion note",
        ],
        "negative": [
            "no onion or garlic character",
            "neutral aromatic profile",
            "low pungency",
        ],
    },
    "toasted_nut_seed": {
        "positive": [
            "nutty roasted seed aroma",
            "toasted earthy nutty richness",
            "sesame like roasted aromatic quality",
        ],
        "negative": [
            "no roasted nutty seed character",
            "neutral aroma without nuttiness",
            "plain non toasted flavor",
        ],
    },
    "char_maillard": {
        "positive": [
            "charred browned roasted smoky surface",
            "strong maillard caramelized crust",
            "fire cooked smoky bitterness",
        ],
        "negative": [
            "steamed poached plain surface",
            "no browning or roast character",
            "soft pale uncarmelized finish",
        ],
    },
    "heat": {
        "positive": [
            "spicy hot peppery warming sensation",
            "chili heat and aromatic spice",
            "strong trigeminal heat",
        ],
        "negative": [
            "mild and not spicy",
            "no pepper heat",
            "soft gentle seasoning",
        ],
    },
    "delicacy": {
        "positive": [
            "subtle delicate faint mild flavor",
            "light fragile refined taste",
            "lean gentle understated profile",
        ],
        "negative": [
            "bold intense powerful flavor",
            "heavy rich assertive taste",
            "large flavor impact",
        ],
    },
}

PROTOTYPE_CACHE: Dict[str, Dict[str, List[List[float]]]] = {}
MODEL_CACHE: Optional[Any] = None
FEATURE_SCHEMA_CACHE: Optional[List[str]] = None


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


def normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def weighted_average_vectors(vectors_with_weights: List[Tuple[List[float], float]]) -> List[float]:
    if not vectors_with_weights:
        raise ValueError("No vectors provided")
    length = len(vectors_with_weights[0][0])
    out = [0.0] * length
    total_weight = 0.0
    for vec, weight in vectors_with_weights:
        total_weight += weight
        for i, x in enumerate(vec):
            out[i] += x * weight
    if total_weight == 0:
        return out
    return [x / total_weight for x in out]


def infer_image_format(image_mime_type: Optional[str], image_b64: Optional[str]) -> str:
    if image_mime_type:
        image_mime_type = image_mime_type.lower()
        if "jpeg" in image_mime_type or "jpg" in image_mime_type:
            return "jpeg"
        if "png" in image_mime_type:
            return "png"
        if "webp" in image_mime_type:
            return "webp"
        if "gif" in image_mime_type:
            return "gif"
    if image_b64 and image_b64.startswith("data:image/"):
        prefix = image_b64.split(";")[0]
        if "jpeg" in prefix or "jpg" in prefix:
            return "jpeg"
        if "png" in prefix:
            return "png"
        if "webp" in prefix:
            return "webp"
        if "gif" in prefix:
            return "gif"
    return "jpeg"


def strip_data_url_prefix(image_b64: str) -> str:
    if image_b64.startswith("data:"):
        return image_b64.split(",", 1)[1]
    return image_b64


def embed_text(text: str, dimension: int = 384, purpose: str = "CLASSIFICATION") -> List[float]:
    request_body = {
        "schemaVersion": "nova-multimodal-embed-v1",
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": purpose,
            "embeddingDimension": dimension,
            "text": {
                "truncationMode": "END",
                "value": text,
            },
        },
    }
    res = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(request_body),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(res["body"].read())
    return payload["embeddings"][0]["embedding"]


def embed_image_base64(
    image_b64: str,
    image_format: str,
    dimension: int = 384,
    purpose: str = "CLASSIFICATION",
) -> List[float]:
    clean_b64 = strip_data_url_prefix(image_b64)
    request_body = {
        "schemaVersion": "nova-multimodal-embed-v1",
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": purpose,
            "embeddingDimension": dimension,
            "image": {
                "detailLevel": "STANDARD_IMAGE",
                "format": image_format,
                "source": {
                    "bytes": clean_b64,
                },
            },
        },
    }
    res = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(request_body),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(res["body"].read())
    return payload["embeddings"][0]["embedding"]


def get_axis_embeddings() -> Dict[str, Dict[str, List[List[float]]]]:
    global PROTOTYPE_CACHE
    if PROTOTYPE_CACHE:
        return PROTOTYPE_CACHE

    cache = {}
    for axis, prompts in AXIS_BANK.items():
        cache[axis] = {
            "positive": [normalize(embed_text(p)) for p in prompts["positive"]],
            "negative": [normalize(embed_text(p)) for p in prompts["negative"]],
        }
    PROTOTYPE_CACHE = cache
    return PROTOTYPE_CACHE


def summarize_axis_features(
    dish_vec: List[float],
    positive_vecs: List[List[float]],
    negative_vecs: List[List[float]],
) -> Dict[str, float]:
    pos_scores = [cosine(dish_vec, v) for v in positive_vecs]
    neg_scores = [cosine(dish_vec, v) for v in negative_vecs]

    pos_mean = sum(pos_scores) / len(pos_scores)
    pos_max = max(pos_scores)
    neg_mean = sum(neg_scores) / len(neg_scores)
    gap = pos_mean - neg_mean

    return {
        "pos_mean": round(pos_mean, 6),
        "pos_max": round(pos_max, 6),
        "neg_mean": round(neg_mean, 6),
        "gap": round(gap, 6),
    }


def compute_axis_features(dish_vec: List[float]) -> Dict[str, float]:
    bank = get_axis_embeddings()
    features: Dict[str, float] = {}

    for axis, emb in bank.items():
        stats = summarize_axis_features(dish_vec, emb["positive"], emb["negative"])
        for stat_name, value in stats.items():
            features[f"{axis}__{stat_name}"] = value

    return features


def add_global_features(axis_features: Dict[str, float], has_image: bool) -> Dict[str, float]:
    out = dict(axis_features)

    gap_features = [v for k, v in axis_features.items() if k.endswith("__gap")]
    pos_max_features = [v for k, v in axis_features.items() if k.endswith("__pos_max")]

    sorted_gaps = sorted(gap_features, reverse=True)
    top1_gap = sorted_gaps[0] if len(sorted_gaps) > 0 else 0.0
    top2_gap = sorted_gaps[1] if len(sorted_gaps) > 1 else 0.0

    out["global__top1_gap"] = round(top1_gap, 6)
    out["global__top2_gap"] = round(top2_gap, 6)
    out["global__gap_sum_top2"] = round(top1_gap + top2_gap, 6)
    out["global__mean_pos_max"] = round(sum(pos_max_features) / max(1, len(pos_max_features)), 6)
    out["global__has_image"] = 1.0 if has_image else 0.0

    return out


def build_text_for_embedding(title: str, description: str, notes: str = "") -> str:
    title = (title or "").strip()
    description = (description or "").strip()
    notes = (notes or "").strip()

    parts = [p for p in [title, description, notes] if p]
    return ". ".join(parts)


def load_trained_model() -> Optional[Any]:
    global MODEL_CACHE
    if MODEL_CACHE is not None:
        return MODEL_CACHE

    if not MODEL_PATH.exists():
        return None

    with open(MODEL_PATH, "rb") as f:
        MODEL_CACHE = pickle.load(f)
    return MODEL_CACHE


def load_feature_schema() -> Optional[List[str]]:
    global FEATURE_SCHEMA_CACHE
    if FEATURE_SCHEMA_CACHE is not None:
        return FEATURE_SCHEMA_CACHE

    if not FEATURE_SCHEMA_PATH.exists():
        return None

    with open(FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    FEATURE_SCHEMA_CACHE = payload["feature_names"]
    return FEATURE_SCHEMA_CACHE


def feature_vector_from_map(feature_map: Dict[str, float], feature_names: List[str]) -> List[float]:
    return [float(feature_map.get(name, 0.0)) for name in feature_names]


def fallback_predict_score(feature_map: Dict[str, float]) -> Tuple[float, int, str]:
    raw_signal = (
        0.30 * feature_map.get("savory_depth__gap", 0.0)
        + 0.22 * feature_map.get("richness__gap", 0.0)
        + 0.12 * feature_map.get("char_maillard__gap", 0.0)
        + 0.10 * feature_map.get("allium__gap", 0.0)
        + 0.08 * feature_map.get("toasted_nut_seed__gap", 0.0)
        + 0.07 * feature_map.get("heat__gap", 0.0)
        + 0.05 * feature_map.get("acidity__gap", 0.0)
        - 0.16 * feature_map.get("delicacy__gap", 0.0)
        + 0.20 * feature_map.get("global__gap_sum_top2", 0.0)
    )

    raw_score = 1.0 / (1.0 + math.exp(-2.2 * raw_signal))
    raw_score = max(0.0, min(1.0, round(raw_score, 4)))

    score_1_to_5 = max(1, min(5, int(round(1 + 4 * raw_score))))
    label_map = {1: "No", 2: "Meh", 3: "Yes-", 4: "Yes", 5: "YES"}
    return raw_score, score_1_to_5, label_map[score_1_to_5]


def model_predict_score(feature_map: Dict[str, float]) -> Tuple[float, int, str, str]:
    model = load_trained_model()
    feature_names = load_feature_schema()

    if model is None or feature_names is None:
        raw_score, score_1_to_5, predicted_label = fallback_predict_score(feature_map)
        return raw_score, score_1_to_5, predicted_label, "fallback"

    vector = feature_vector_from_map(feature_map, feature_names)
    predicted_score_continuous = float(model.predict([vector])[0])
    predicted_score_continuous = max(1.0, min(5.0, predicted_score_continuous))

    raw_score = round((predicted_score_continuous - 1.0) / 4.0, 4)
    score_1_to_5 = max(1, min(5, int(round(predicted_score_continuous))))

    label_map = {1: "No", 2: "Meh", 3: "Yes-", 4: "Yes", 5: "YES"}
    return raw_score, score_1_to_5, label_map[score_1_to_5], "trained_model"


def upload_image_to_s3(entry_id: str, image_b64: str, image_format: str, image_mime_type: Optional[str]) -> str:
    clean_b64 = strip_data_url_prefix(image_b64)
    image_bytes = base64.b64decode(clean_b64)
    ext = "jpg" if image_format == "jpeg" else image_format
    key = f"images/{entry_id}.{ext}"

    s3.put_object(
        Bucket=IMAGE_BUCKET,
        Key=key,
        Body=image_bytes,
        ContentType=image_mime_type or f"image/{image_format}",
    )
    return key


def build_dish_vector(description: str, image_b64: Optional[str], image_mime_type: Optional[str]) -> Tuple[List[float], Optional[List[float]], bool, Optional[str]]:
    text_embedding = normalize(embed_text(description))

    if not image_b64:
        return text_embedding, None, False, None

    image_format = infer_image_format(image_mime_type, image_b64)
    image_embedding = normalize(embed_image_base64(image_b64, image_format))
    dish_embedding = normalize(weighted_average_vectors([
        (text_embedding, 0.65),
        (image_embedding, 0.35),
    ]))
    return dish_embedding, image_embedding, True, image_format


def handle_predict(body: dict) -> dict:
    title = (body.get("title") or "").strip()
    description = (body.get("description") or "").strip()
    image_b64 = body.get("image_base64")
    image_mime_type = body.get("image_mime_type")

    if not title or not description:
        return response(400, {"error": "title and description are required"})

    entry_id = str(uuid.uuid4())
    now = int(time.time())

    image_key = None
    dish_embedding, image_embedding, has_image, image_format = build_dish_vector(description, image_b64, image_mime_type)
    text_embedding = normalize(embed_text(description))

    if image_b64 and image_format:
        image_key = upload_image_to_s3(entry_id, image_b64, image_format, image_mime_type)

    axis_features = compute_axis_features(dish_embedding)
    feature_map = add_global_features(axis_features, has_image=has_image)
    raw_score, score_1_to_5, predicted_label, prediction_source = model_predict_score(feature_map)

    item = {
        "id": entry_id,
        "record_type": "prediction",
        "model_version": MODEL_VERSION,
        "title": title,
        "description": description,
        "image_key": image_key,
        "predicted_score_raw": raw_score,
        "predicted_score_1_to_5": score_1_to_5,
        "predicted_label": predicted_label,
        "prediction_source": prediction_source,
        "text_embedding": text_embedding,
        "image_embedding": image_embedding,
        "dish_embedding": dish_embedding,
        "axis_features": axis_features,
        "feature_map": feature_map,
        "created_at": now,
        "updated_at": now,
    }

    table.put_item(Item=to_decimal(item))

    return response(200, {
        "entry_id": entry_id,
        "model_version": MODEL_VERSION,
        "predicted_score_1_to_5": score_1_to_5,
        "predicted_label": predicted_label,
        "predicted_score_raw": raw_score,
        "prediction_source": prediction_source,
        "axis_features": axis_features,
        "image_key": image_key,
    })


def handle_validate(body: dict) -> dict:
    entry_id = body.get("entry_id")
    actual_score = body.get("actual_score")
    contributors = body.get("contributors", "")
    notes = body.get("notes", "")

    if not entry_id or actual_score is None:
        return response(400, {"error": "entry_id and actual_score are required"})

    try:
        actual_score = float(actual_score)
    except Exception:
        return response(400, {"error": "actual_score must be numeric from 1 to 5"})

    if actual_score < 1 or actual_score > 5:
        return response(400, {"error": "actual_score must be between 1 and 5"})

    now = int(time.time())

    table.update_item(
        Key={"id": entry_id},
        UpdateExpression="SET actual_score_1_to_5 = :a, contributors = :c, notes = :n, updated_at = :u",
        ExpressionAttributeValues=to_decimal({
            ":a": actual_score,
            ":c": contributors,
            ":n": notes,
            ":u": now,
        }),
    )

    return response(200, {"ok": True})


def handle_get_entry(event: dict) -> dict:
    qsp = event.get("queryStringParameters") or {}
    entry_id = qsp.get("id")
    title = qsp.get("title") or qsp.get("q")

    if entry_id:
        res = table.get_item(Key={"id": entry_id})
        item = res.get("Item")
        if not item:
            return response(404, {"error": "entry not found"})
        return response(200, json.loads(json.dumps(item, default=str)))

    if title:
        res = table.scan(FilterExpression=Attr("title").eq(title))
        items = res.get("Items", [])
        return response(200, {"items": json.loads(json.dumps(items, default=str))})

    return response(400, {"error": "provide id or title"})


def handle_upload_training_data(body: dict) -> dict:
    filename = (body.get("filename") or "").strip()
    if not filename:
        return response(400, {"error": "filename is required"})

    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        return response(404, {"error": f"file not found: {file_path}"})

    with open(file_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    now = int(time.time())
    created_ids = []
    skipped = []

    for idx, row in enumerate(rows, start=1):
        title = (row.get("title") or "").strip()
        notes = (row.get("notes") or "").strip()
        score = row.get("score_1_to_5")

        if not title or score is None:
            skipped.append({"row": idx, "reason": "missing title or score"})
            continue

        try:
            score = float(score)
        except Exception:
            skipped.append({"row": idx, "reason": "invalid score"})
            continue

        description = build_text_for_embedding(title=title, description=notes, notes="")
        text_embedding = normalize(embed_text(description))
        dish_embedding = text_embedding
        axis_features = compute_axis_features(dish_embedding)
        feature_map = add_global_features(axis_features, has_image=False)

        entry_id = str(uuid.uuid4())
        item = {
            "id": entry_id,
            "record_type": "training_example",
            "model_version": MODEL_VERSION,
            "title": title,
            "description": description,
            "notes": notes,
            "actual_score_1_to_5": score,
            "text_embedding": text_embedding,
            "dish_embedding": dish_embedding,
            "axis_features": axis_features,
            "feature_map": feature_map,
            "source_filename": filename,
            "created_at": now,
            "updated_at": now,
        }

        table.put_item(Item=to_decimal(item))
        created_ids.append(entry_id)

    return response(200, {
        "ok": True,
        "source_filename": filename,
        "created_count": len(created_ids),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "created_ids": created_ids,
    })


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
            body = json.loads(event.get("body") or "{}")
            return handle_predict(body)

        if method == "POST" and path.endswith("/validate"):
            body = json.loads(event.get("body") or "{}")
            return handle_validate(body)

        if method == "POST" and path.endswith("/upload-training-data"):
            body = json.loads(event.get("body") or "{}")
            return handle_upload_training_data(body)

        if method == "GET" and path.endswith("/entry"):
            return handle_get_entry(event)

        return response(404, {"error": "route not found"})
    except Exception as e:
        return response(500, {"error": str(e)})
