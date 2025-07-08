import json
from flask import Flask, jsonify
from flask_cors import CORS

# Initialisiere die Flask App
app = Flask(__name__)
# Erlaube Cross-Origin Requests, damit das Frontend auf das Backend zugreifen kann
CORS(app)

# Definiere einen API-Endpunkt unter "/api/data"
@app.route('/api/data')
def get_political_data():
    """
    Liest die Mock-Daten aus der JSON-Datei und gibt sie als Antwort zurück.
    """
    try:
        # Öffne die JSON-Datei
        with open('mock_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Gib die Daten im JSON-Format zurück
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "mock_data.json nicht gefunden!"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Starte die App, wenn das Skript direkt ausgeführt wird
if __name__ == '__main__':
    app.run(debug=True)