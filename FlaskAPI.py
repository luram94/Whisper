import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ASRTranscriber import ASRTranscriber  # Asegúrate de que ASRTranscriber esté en el path o definido en el mismo archivo
from time import time

app = Flask(__name__)

# Configura una carpeta para los archivos subidos
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

start_time = time()

# Instancia de ASRTranscriber
transcriber = ASRTranscriber(
    model_id="openai/whisper-medium",
    batch_size=16,
    chunk_length_s=10,
    max_new_tokens=128
)
print(f"Tiempo de carga del modelo: {time()-start_time}")
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Comprueba si el post request tiene el file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # Si el usuario no selecciona un archivo, el navegador
    # puede enviar una parte vacía sin filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.wav'):
        filename = secure_filename(file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(audio_path)

        # Procesar la transcripción
        try:
            output_file = f"{os.path.splitext(filename)[0]}_transcription.txt"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_file)
            transcriber.transcribe_and_save(audio_path, output_path)

            # Leer y devolver la transcripción
            with open(output_path, 'r', encoding='utf-8') as f:
                transcription = f.read()
            return jsonify({"filename": filename, "transcription": transcription})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    else:
        return jsonify({"error": "Unsupported file type"}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)