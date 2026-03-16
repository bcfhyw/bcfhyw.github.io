import os
import json
import time
import uuid
import math
import base64
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

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
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(TABLE_NAME)
s3 = boto3.client("s3", region_name=AWS_REGION)
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# General sensory prototypes, not specific foods.
PROTOTYPES = {
    "umami": {
        "pos": [
            "deep savory taste",
            "rich mouthfilling savoriness",
            "brothy fermented savory intensity",
        ],
        "neg": [
            "bland flat taste",
            "low savoriness",
            "thin neutral flavor",
        ],
    },
    "fat": {
        "pos": [
            "rich fatty creamy buttery oily texture",
            "coating luxurious richness",
            "dense mouthfilling body",
        ],
        "neg": [
            "lean dry thin texture",
            "light low fat mouthfeel",
            "not rich or creamy",
        ],
    },
    "acid": {
        "pos": [
            "bright acidic tangy sharp taste",
            "citrusy vinegary pickled brightness",
            "clean sour lift",
        ],
        "neg": [
            "mellow non acidic taste",
            "soft rounded low brightness",
            "neutral not tangy",
        ],
    },
    "allium": {
        "pos": [
            "strong onion garlic scallion aroma",
            "savory sulfurous allium intensity",
            "pungent aromatic onion note",
        ],
        "neg": [
            "no onion or garlic character",
            "neutral aromatic profile",
            "low pungency",
        ],
    },
    "sesame": {
        "pos": [
            "nutty roasted seed aroma",
            "toasted earthy nutty richness",
            "sesame like roasted aromatic quality",
        ],
        "neg": [
            "no roasted nutty seed character",
            "neutral aroma without nuttiness",
            "plain non toasted flavor",
        ],
    },
    "char": {
        "pos": [
            "charred browned roasted smoky surface",
            "strong maillard caramelized crust",
            "fire cooked smoky bitterness",
        ],
        "neg": [
            "steamed poached plain surface",
            "no browning or roast character",
            "soft pale uncarmelized finish",
        ],
    },
    "spice": {
        "pos": [
            "spicy hot peppery warming sensation",
            "chili heat and aromatic spice",
            "strong trigeminal heat",
        ],
        "neg": [
            "mild and not spicy",
            "no pepper heat",
            "soft gentle seasoning",
        ],
    },
    "delicate": {
        "pos": [
            "subtle delicate faint mild flavor",
            "light fragile refined taste",
            "lean gentle understated profile",
        ],
        "neg": [
            "bold intense powerful flavor",
            "heavy rich assertive taste",
            "large flavor impact",
        ],
    },
}

PROTOTYPE_CACHE: Dict[str, Dict[str, List[List[float]]]] = {}


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


def get_prototype_embeddings() -> Dict[str, Dict[str, List[List[float]]]]:
    global PROTOTYPE_CACHE
    if PROTOTYPE_CACHE:
        return PROTOTYPE_CACHE

    cache = {}
    for axis, prompts in PROTOTYPES.items():
        cache[axis] = {
            "pos": [normalize(embed_text(p)) for p in prompts["pos"]],
            "neg": [normalize(embed_text(p)) for p in prompts["neg"]],
        }
    PROTOTYPE_CACHE = cache
    return PROTOTYPE_CACHE


def score_axis(dish_vec: List[float], pos_vecs: List[List[float]], neg_vecs: List[List[float]]) -> float:
    pos_score = sum(cosine(dish_vec, v) for v in pos_vecs) / len(pos_vecs)
    neg_score = sum(cosine(dish_vec, v) for v in neg_vecs) / len(neg_vecs)
    raw = (pos_score - neg_score + 1.0) / 2.0
    return max(0.0, min(1.0, raw))


def compute_axes(dish_vec: List[float]) -> Dict[str, float]:
    proto = get_prototype_embeddings()
    axes = {}
    for axis, emb in proto.items():
        axes[axis] = round(score_axis(dish_vec, emb["pos"], emb["neg"]), 4)
    return axes


def predict_score_1_to_5(axes: Dict[str, float]) -> Tuple[float, int, str]:
    raw_score = (
        0.28 * axes["umami"]
        + 0.24 * axes["fat"]
        + 0.12 * axes["char"]
        + 0.10 * axes["allium"]
        + 0.08 * axes["sesame"]
        + 0.06 * axes["spice"]
        + 0.05 * axes["acid"]
        - 0.18 * axes["delicate"]
    )
    raw_score = max(0.0, min(1.0, round(raw_score, 4)))

    score_1_to_5 = max(1, min(5, int(round(1 + 4 * raw_score))))

    label_map = {
        1: "No",
        2: "Meh",
        3: "Yes-",
        4: "Yes",
        5: "YES",
    }

    return raw_score, score_1_to_5, label_map[score_1_to_5]


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
    image_format = None

    text_vec = normalize(embed_text(description))

    if image_b64:
        image_format = infer_image_format(image_mime_type, image_b64)
        image_key = upload_image_to_s3(entry_id, image_b64, image_format, image_mime_type)
        image_vec = normalize(embed_image_base64(image_b64, image_format))
        dish_vec = normalize(weighted_average_vectors([
            (text_vec, 0.65),
            (image_vec, 0.35),
        ]))
    else:
        dish_vec = text_vec

    axes = compute_axes(dish_vec)
    raw_score, score_1_to_5, predicted_label = predict_score_1_to_5(axes)

    item = {
        "id": entry_id,
        "model_version": MODEL_VERSION,
        "title": title,
        "description": description,
        "image_key": image_key,
        "predicted_score_raw": raw_score,
        "predicted_score_1_to_5": score_1_to_5,
        "predicted_label": predicted_label,
        "axes": axes,
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
        "axes": axes,
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
        actual_score = int(actual_score)
    except Exception:
        return response(400, {"error": "actual_score must be an integer from 1 to 5"})

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

        if method == "GET" and path.endswith("/entry"):
            return handle_get_entry(event)

        return response(404, {"error": "route not found"})
    except Exception as e:
        return response(500, {"error": str(e)})
