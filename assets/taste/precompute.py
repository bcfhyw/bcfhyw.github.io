import json
import math
import boto3

AWS_REGION = "us-east-1"
BEDROCK_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
OUTPUT_FILE = "axis_prototypes.json"

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

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def embed_text(text):
    body = {
        "schemaVersion": "nova-multimodal-embed-v1",
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "CLASSIFICATION",
            "embeddingDimension": 384,
            "text": {
                "truncationMode": "END",
                "value": text,
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


def main():
    out = {
        "model_id": BEDROCK_MODEL_ID,
        "embedding_dimension": 384,
        "axes": {},
    }

    for axis, prompts in AXIS_BANK.items():
        print(f"Embedding axis: {axis}")
        out["axes"][axis] = {
            "positive": [normalize(embed_text(p)) for p in prompts["positive"]],
            "negative": [normalize(embed_text(p)) for p in prompts["negative"]],
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f)

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
