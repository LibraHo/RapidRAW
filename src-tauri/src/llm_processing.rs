use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use image::{codecs::jpeg::JpegEncoder, DynamicImage, GenericImageView};
use reqwest::Client;
use serde_json::{json, Value};
use std::io::Cursor;

const MAX_IMAGE_DIM: u32 = 1024;

const SYSTEM_PROMPT: &str = r#"You are an expert professional photo editor with deep knowledge of RAW image processing and color science.

Analyze the provided photograph and the user's edit request, then return specific adjustment parameters as a JSON object.

CRITICAL: Respond ONLY with a valid JSON object. No explanation, no markdown code blocks, no surrounding text — just the raw JSON object itself.

Available parameters (all optional — only include parameters you want to change from their current values):
- exposure: number from -5 to 5 (overall brightness in stops; 0 = no change)
- contrast: number from -100 to 100 (tonal contrast)
- highlights: number from -100 to 100 (negative = recover blown highlights, positive = boost)
- shadows: number from -100 to 100 (positive = lift shadows to reveal detail, negative = crush)
- whites: number from -100 to 100 (white point / brightest tone adjustment)
- blacks: number from -100 to 100 (black point / darkest tone adjustment)
- brightness: number from -100 to 100 (global midtone brightness offset)
- temperature: number from -100 to 100 (negative = cooler/more blue, positive = warmer/more orange)
- tint: number from -100 to 100 (negative = more green, positive = more magenta)
- saturation: number from -100 to 100 (global color saturation; negative = desaturate toward B&W)
- vibrance: number from -100 to 100 (smart saturation boost that protects already-saturated colors and skin tones)
- clarity: number from -100 to 100 (midtone contrast and texture; positive = sharper/punchier, negative = matte/dreamy)
- dehaze: number from -100 to 100 (positive = remove atmospheric haze/fog, negative = add misty atmosphere)
- sharpness: number from 0 to 100 (edge sharpening amount)
- lumaNoiseReduction: number from 0 to 100 (luminance/grain noise reduction)
- colorNoiseReduction: number from 0 to 100 (color/chroma noise reduction)
- vignetteAmount: number from -100 to 100 (negative = darken edges to focus attention on center, positive = lighten edges)
- grainAmount: number from 0 to 100 (analog film grain texture amount)
- glowAmount: number from 0 to 100 (soft bloom/glow effect diffusing from highlights)
- halationAmount: number from 0 to 100 (cinematic red/orange glow bleeding around bright highlights, emulates film halation)
- toneMapper: "basic" or "agx" (rendering algorithm; "agx" produces more filmic, cinematic, naturalistic results with better highlight rolloff)

Think carefully about what adjustments will achieve the user's request. Consider:
- Scene type (portrait, landscape, street, night, golden hour, etc.)
- Desired mood and aesthetic
- Technical issues to correct (exposure, white balance, noise, etc.)
- Stylistic choices (cinematic, vintage, clean/modern, dramatic, etc.)

Example response for "make this sunset photo look cinematic and dramatic":
{"exposure":0.1,"contrast":25,"highlights":-35,"shadows":20,"temperature":20,"tint":5,"vibrance":30,"clarity":15,"vignetteAmount":-35,"grainAmount":12,"halationAmount":25,"toneMapper":"agx"}"#;

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

fn validate_adjustments(adj: &Value) -> Result<()> {
    if !adj.is_object() {
        return Err(anyhow!("LLM response must be a JSON object"));
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
                        "Parameter '{}' value {} is out of range [{}, {}]",
                        key,
                        n,
                        min,
                        max
                    ))
                }
                None => return Err(anyhow!("Parameter '{}' must be a number", key)),
            }
        }
    }

    if let Some(tm) = adj.get("toneMapper") {
        match tm.as_str() {
            Some("basic") | Some("agx") => {}
            _ => return Err(anyhow!("toneMapper must be \"basic\" or \"agx\"")),
        }
    }

    Ok(())
}

/// Call the Anthropic Claude API with the image and prompt, returning adjustment parameter overrides.
///
/// The returned JSON object contains only the parameters that should change — callers should
/// merge it on top of the existing adjustments rather than replacing them entirely.
pub async fn invoke_llm_edit(
    image: &DynamicImage,
    user_prompt: &str,
    current_adjustments: &Value,
    api_key: &str,
) -> Result<Value> {
    let resized = resize_for_llm(image);
    let image_b64 = image_to_jpeg_base64(&resized)?;

    // Build a compact context summary of the relevant current numeric adjustments.
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

    let user_content = format!(
        "Current adjustment values (for context): {}\n\nEdit request: {}",
        serde_json::to_string(&current_context).unwrap_or_default(),
        user_prompt
    );

    let request_body = json!({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
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

    // Strip any accidental markdown code block wrapping the LLM might have added
    let cleaned = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let adjustments: Value = serde_json::from_str(cleaned).map_err(|e| {
        anyhow!(
            "Failed to parse LLM response as JSON: {}. Raw response: {}",
            e,
            text
        )
    })?;

    validate_adjustments(&adjustments)?;

    Ok(adjustments)
}
