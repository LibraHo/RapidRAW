use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use image::{codecs::jpeg::JpegEncoder, DynamicImage, GenericImageView};
use reqwest::Client;
use serde_json::{json, Value};
use std::io::Cursor;

const MAX_IMAGE_DIM: u32 = 1024;

const SYSTEM_PROMPT: &str = r#"You are an expert professional photo editor with deep knowledge of RAW image processing and color science.

Analyze the provided photograph and the user's edit request, then return a JSON object describing adjustments to apply.

CRITICAL: Respond ONLY with a valid JSON object. No explanation, no markdown code blocks, no surrounding text — just the raw JSON object itself.

## Response Schema

```
{
  "adjustments": { ...global adjustment overrides... },
  "masks": [ ...optional per-region adjustments... ]
}
```

### "adjustments" (required, may be empty {})
Global adjustments applied to the entire image. Only include parameters you want to change:

- exposure: -5 to 5 (overall brightness in stops)
- contrast: -100 to 100
- highlights: -100 to 100 (negative = recover blown highlights)
- shadows: -100 to 100 (positive = lift shadows)
- whites: -100 to 100 (white point)
- blacks: -100 to 100 (black point)
- brightness: -100 to 100 (midtone brightness offset)
- temperature: -100 to 100 (negative = cooler/blue, positive = warmer/orange)
- tint: -100 to 100 (negative = green, positive = magenta)
- saturation: -100 to 100 (negative = desaturate toward B&W)
- vibrance: -100 to 100 (smart saturation, protects skin tones)
- clarity: -100 to 100 (midtone contrast; positive = punchy, negative = matte)
- dehaze: -100 to 100 (positive = remove haze, negative = add atmosphere)
- sharpness: 0 to 100
- lumaNoiseReduction: 0 to 100
- colorNoiseReduction: 0 to 100
- vignetteAmount: -100 to 100 (negative = darken edges)
- grainAmount: 0 to 100 (film grain)
- glowAmount: 0 to 100 (highlight bloom/glow)
- halationAmount: 0 to 100 (cinematic red glow on highlights, film halation)
- toneMapper: "basic" or "agx" ("agx" = more filmic/cinematic)

### "masks" (optional array, default [])
Use masks when different regions of the image need SIGNIFICANTLY DIFFERENT adjustments.
IMPORTANT: Only add masks when they genuinely improve the result. Max 3 masks per response.

Each mask is one of two types:

**Type A — Semantic AI mask** (whole-image semantic segmentation, no coordinates needed):
```
{
  "name": "descriptive name",
  "type": "ai-sky" | "ai-subject" | "ai-foreground",
  "adjustments": { ...adjustment parameters... }
}
```
- "ai-sky": sky and clouds — for landscapes, seascapes, cityscapes
- "ai-subject": main subject (person, animal, key object) — for portraits, wildlife, product
- "ai-foreground": everything in front of the sky — for landscape foregrounds

**Type B — SAM bounding-box mask** (use SAM neural network to precisely segment ANY specific object or region):
```
{
  "name": "descriptive name",
  "type": "sam-box",
  "bbox": { "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0 },
  "adjustments": { ...adjustment parameters... }
}
```
- bbox coordinates are NORMALIZED 0.0–1.0 (x1,y1 = top-left; x2,y2 = bottom-right)
- Use "sam-box" for SPECIFIC objects that ai-sky/ai-subject/ai-foreground cannot isolate:
  cars, buildings, mountains, water, faces, clothing, specific people in a group, animals, trees, etc.
- Estimate the bounding box by carefully looking at where the object is in the image
- The SAM neural network will precisely refine the pixels within your bounding box

Mask adjustments support: exposure, contrast, highlights, shadows, whites, blacks, brightness, temperature, tint, saturation, vibrance, clarity, dehaze, sharpness, lumaNoiseReduction, colorNoiseReduction

## Examples

"dramatic landscape, moody sky, warm foreground":
{"adjustments":{"contrast":15,"vignetteAmount":-20,"toneMapper":"agx"},"masks":[{"name":"Sky","type":"ai-sky","adjustments":{"highlights":-50,"contrast":35,"temperature":-20,"saturation":25}},{"name":"Foreground","type":"ai-foreground","adjustments":{"shadows":20,"temperature":15,"clarity":10}}]}

"make the red car pop against a desaturated background" (car is roughly center-right):
{"adjustments":{"saturation":-40},"masks":[{"name":"Red Car","type":"sam-box","bbox":{"x1":0.45,"y1":0.3,"x2":0.9,"y2":0.8},"adjustments":{"saturation":60,"vibrance":30,"clarity":15}}]}

"bright and clean portrait, no mask needed":
{"adjustments":{"exposure":0.4,"highlights":-15,"shadows":20,"temperature":8,"vibrance":15,"sharpness":25},"masks":[]}"#;

/// Resize image so its longest dimension is at most MAX_IMAGE_DIM, preserving aspect ratio.
fn resize_for_llm(img: &DynamicImage) -> DynamicImage {
    let (w, h) = img.dimensions();
    if w <= MAX_IMAGE_DIM && h <= MAX_IMAGE_DIM {
        return img.clone();
    }
    let (new_w, new_h) = if w >= h {
        let new_w = MAX_IMAGE_DIM;
        let new_h = ((MAX_IMAGE_DIM as f64 * h as f64) / w as f64).round() as u32;
        (new_w, new_h.max(1))
    } else {
        let new_h = MAX_IMAGE_DIM;
        let new_w = ((MAX_IMAGE_DIM as f64 * w as f64) / h as f64).round() as u32;
        (new_w.max(1), new_h)
    };
    img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
}

fn image_to_jpeg_base64(img: &DynamicImage) -> Result<String> {
    let mut buf = Cursor::new(Vec::new());
    let mut encoder = JpegEncoder::new_with_quality(&mut buf, 85);
    encoder.encode_image(&img.to_rgb8())?;
    Ok(general_purpose::STANDARD.encode(buf.get_ref()))
}

/// Validate a flat adjustments object (global or per-mask).
fn validate_adjustment_object(adj: &Value, context: &str) -> Result<()> {
    if !adj.is_object() {
        return Err(anyhow!("{} must be a JSON object", context));
    }

    let numeric_params: &[(&str, f64, f64)] = &[
        ("exposure", -5.0, 5.0),
        ("contrast", -100.0, 100.0),
        ("highlights", -100.0, 100.0),
        ("shadows", -100.0, 100.0),
        ("whites", -100.0, 100.0),
        ("blacks", -100.0, 100.0),
        ("brightness", -100.0, 100.0),
        ("temperature", -100.0, 100.0),
        ("tint", -100.0, 100.0),
        ("saturation", -100.0, 100.0),
        ("vibrance", -100.0, 100.0),
        ("clarity", -100.0, 100.0),
        ("dehaze", -100.0, 100.0),
        ("sharpness", 0.0, 100.0),
        ("lumaNoiseReduction", 0.0, 100.0),
        ("colorNoiseReduction", 0.0, 100.0),
        ("vignetteAmount", -100.0, 100.0),
        ("grainAmount", 0.0, 100.0),
        ("glowAmount", 0.0, 100.0),
        ("halationAmount", 0.0, 100.0),
    ];

    for (key, min, max) in numeric_params {
        if let Some(val) = adj.get(key) {
            match val.as_f64() {
                Some(n) if n >= *min && n <= *max => {}
                Some(n) => {
                    return Err(anyhow!(
                        "{}: parameter '{}' value {} is out of range [{}, {}]",
                        context,
                        key,
                        n,
                        min,
                        max
                    ))
                }
                None => {
                    return Err(anyhow!(
                        "{}: parameter '{}' must be a number",
                        context,
                        key
                    ))
                }
            }
        }
    }

    if let Some(tm) = adj.get("toneMapper") {
        match tm.as_str() {
            Some("basic") | Some("agx") => {}
            _ => return Err(anyhow!("{}: toneMapper must be \"basic\" or \"agx\"", context)),
        }
    }

    Ok(())
}

/// Validate the full LLM response object (new schema with adjustments + masks).
fn validate_response(response: &Value) -> Result<()> {
    if !response.is_object() {
        return Err(anyhow!("LLM response must be a JSON object"));
    }

    // Validate global adjustments
    let adjustments = response
        .get("adjustments")
        .ok_or_else(|| anyhow!("Response missing 'adjustments' key"))?;
    validate_adjustment_object(adjustments, "adjustments")?;

    // Validate masks array (optional)
    if let Some(masks) = response.get("masks") {
        let masks_arr = masks
            .as_array()
            .ok_or_else(|| anyhow!("'masks' must be an array"))?;

        if masks_arr.len() > 3 {
            return Err(anyhow!("Too many masks returned (max 3)"));
        }

        for (i, mask) in masks_arr.iter().enumerate() {
            let ctx = format!("masks[{}]", i);

            // Validate mask type
            let mask_type = mask
                .get("type")
                .and_then(|t| t.as_str())
                .ok_or_else(|| anyhow!("{}: missing or invalid 'type' field", ctx))?;

            match mask_type {
                "ai-sky" | "ai-subject" | "ai-foreground" => {}
                "sam-box" => {
                    // Validate bounding box
                    let bbox = mask
                        .get("bbox")
                        .and_then(|b| b.as_object())
                        .ok_or_else(|| anyhow!("{}: sam-box requires a 'bbox' object", ctx))?;

                    for coord in ["x1", "y1", "x2", "y2"] {
                        let val = bbox
                            .get(coord)
                            .and_then(|v| v.as_f64())
                            .ok_or_else(|| {
                                anyhow!("{}.bbox.{}: must be a number", ctx, coord)
                            })?;
                        if !(0.0..=1.0).contains(&val) {
                            return Err(anyhow!(
                                "{}.bbox.{}: value {} must be between 0.0 and 1.0",
                                ctx,
                                coord,
                                val
                            ));
                        }
                    }

                    let x1 = bbox["x1"].as_f64().unwrap();
                    let y1 = bbox["y1"].as_f64().unwrap();
                    let x2 = bbox["x2"].as_f64().unwrap();
                    let y2 = bbox["y2"].as_f64().unwrap();

                    if x2 <= x1 || y2 <= y1 {
                        return Err(anyhow!(
                            "{}.bbox: x2 must be > x1 and y2 must be > y1",
                            ctx
                        ));
                    }
                }
                other => {
                    return Err(anyhow!(
                        "{}: unknown mask type '{}'. Must be ai-sky, ai-subject, ai-foreground, or sam-box",
                        ctx,
                        other
                    ))
                }
            }

            // Validate mask adjustments
            let mask_adj = mask
                .get("adjustments")
                .ok_or_else(|| anyhow!("{}: missing 'adjustments' field", ctx))?;
            validate_adjustment_object(mask_adj, &format!("{}.adjustments", ctx))?;
        }
    }

    Ok(())
}

/// Call the Anthropic Claude API with the image and prompt.
///
/// Returns a JSON object with the schema:
/// ```json
/// {
///   "adjustments": { ...global parameter overrides... },
///   "masks": [
///     { "name": "Sky", "type": "ai-sky", "adjustments": { ... } }
///   ]
/// }
/// ```
/// Callers should merge `adjustments` on top of existing global adjustments,
/// and append the `masks` entries as new MaskContainers.
pub async fn invoke_llm_edit(
    image: &DynamicImage,
    user_prompt: &str,
    current_adjustments: &Value,
    api_key: &str,
) -> Result<Value> {
    let resized = resize_for_llm(image);
    let image_b64 = image_to_jpeg_base64(&resized)?;

    // Build a compact context summary of the current numeric adjustments.
    let context_keys = [
        "exposure",
        "contrast",
        "highlights",
        "shadows",
        "whites",
        "blacks",
        "brightness",
        "temperature",
        "tint",
        "saturation",
        "vibrance",
        "clarity",
        "dehaze",
        "sharpness",
        "lumaNoiseReduction",
        "colorNoiseReduction",
        "vignetteAmount",
        "grainAmount",
        "glowAmount",
        "halationAmount",
        "toneMapper",
    ];
    let mut current_context = serde_json::Map::new();
    for key in &context_keys {
        if let Some(val) = current_adjustments.get(key) {
            current_context.insert(key.to_string(), val.clone());
        }
    }

    // Include current mask count so the LLM knows existing masks
    let existing_mask_count = current_adjustments
        .get("masks")
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    let user_content = format!(
        "Current adjustment values (for context): {}\nExisting masks: {}\n\nEdit request: {}",
        serde_json::to_string(&current_context).unwrap_or_default(),
        existing_mask_count,
        user_prompt
    );

    let request_body = json!({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 3000,
        "system": SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64
                    }
                },
                {
                    "type": "text",
                    "text": user_content
                }
            ]
        }]
    });

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&request_body)
        .send()
        .await
        .map_err(|e| anyhow!("Failed to reach Anthropic API: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "Could not read response body".to_string());
        return Err(anyhow!("Anthropic API error ({}): {}", status, body));
    }

    let api_response: Value = response
        .json()
        .await
        .map_err(|e| anyhow!("Failed to parse Anthropic API response: {}", e))?;

    let text = api_response
        .get("content")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|item| item.get("text"))
        .and_then(|t| t.as_str())
        .ok_or_else(|| anyhow!("Unexpected Anthropic API response format"))?;

    // Strip any accidental markdown code block wrapping
    let cleaned = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let result: Value = serde_json::from_str(cleaned).map_err(|e| {
        anyhow!(
            "Failed to parse LLM response as JSON: {}. Raw response: {}",
            e,
            text
        )
    })?;

    validate_response(&result)?;

    Ok(result)
}
