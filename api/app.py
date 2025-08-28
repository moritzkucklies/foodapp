from flask import Flask, jsonify

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(status="ok")

@app.get("/version")
def version():
    return jsonify(app="foodapp", version="0.1.0")

@app.get("/")
def root():
    return "FoodApp API is up\n", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
