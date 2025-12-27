from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load NLP model (runs once)
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    framework="pt"
)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    text = ""

    if request.method == "POST":
        text = request.form["text"]
        if text.strip():
            result = summarizer(
                text,
                max_length=35,
                min_length=10,
                do_sample=False
            )
            summary = result[0]["summary_text"]

    return render_template("index.html", summary=summary, text=text)

if __name__ == "__main__":
    app.run(debug=True)
