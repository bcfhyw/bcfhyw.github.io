# Taste Prediction Prototype

Experimental system to predict how strongly a specific individual will perceive the flavor of a dish.

The model estimates a **taste intensity score from 1–5** using embeddings derived from a dish description and optionally an image.

## Approach

1. Generate embeddings for:
   - dish description (text)
   - optional image

2. Combine embeddings into a single **dish vector**.

3. Compare the dish vector to **prototype embeddings** representing sensory properties of food.

Current sensory axes:

- umami (savory depth)
- fat (richness / mouthfeel)
- acid (brightness / sourness)
- allium (garlic/onion intensity)
- sesame (roasted nutty aroma)
- char (browning / grill / maillard)
- spice (chili heat / trigeminal stimulation)
- delicate (light / subtle flavors)

Each axis is scored by comparing similarity to positive vs negative prototype prompts.

4. Convert axis scores into a **predicted taste score (1–5)** using a weighted linear model.

## Architecture

Frontend  
- static webpage (GitHub Pages)

Backend  
- AWS Lambda (Python)
- Amazon Bedrock Nova Multimodal Embeddings

Storage  
- DynamoDB (entries + predictions)
- S3 (uploaded images)

## Stored Data

Each entry contains:

- title
- description
- image
- predicted score
- axis values
- actual score (user validation)
- notes
- model_version

## Notes

Currently untrained, to do: add past data in current data format, tune linear model
