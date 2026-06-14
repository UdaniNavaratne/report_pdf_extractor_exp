import os
import base64
import json
from flask import Flask, request, jsonify, render_template
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Load prompt from file
with open("extraction_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/extract", methods=["POST"])
def extract():
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["pdf"]
    if not pdf_file.filename.endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    pdf_bytes = pdf_file.read()
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract the marketing financials data from this report and return only the JSON object. Do not include any text, explanation, or markdown before or after the JSON.",
                    },
                ],
            }
        ],
    )

    text = "".join(block.text for block in response.content if hasattr(block, "text"))

    # Strip markdown fences if present
    clean = text.strip()
    if clean.startswith("```"):
        # Remove opening fence line (```json or ```)
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean
        # Remove closing fence
        if clean.endswith("```"):
            clean = clean[:-3].strip()

    # Fallback: find JSON object boundaries
    if not clean.startswith("{"):
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            clean = clean[start:end]

    try:
        result = json.loads(clean)
    except json.JSONDecodeError as e:
        print("=== JSON PARSE ERROR ===")
        print("Error:", e)
        print("Raw response (first 800 chars):", text[:800])
        print("========================")
        return jsonify({
            "error": f"Model returned invalid JSON: {str(e)}",
            "raw_preview": text[:800]
        }), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)