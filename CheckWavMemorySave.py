import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import psutil
import GPUtil as gputil
import os

class ASRTranscriber:
    def __init__(self, model_id="openai/whisper-medium", batch_size=16, chunk_length_s=10, max_new_tokens=128):
        # Determina el dispositivo y el tipo de datos de acuerdo con la disponibilidad de CUDA
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Carga el modelo y el procesador
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        # Configura la pipeline de transcripción
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=max_new_tokens,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def transcribe_and_save(self, input_file, output_file):
        # Inicio de medición del tiempo
        start_time = time.time()

        # Realiza la transcripción
        result = self.pipe(input_file)
        transcription = result["text"]

        # Cálculo del tiempo de ejecución
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTiempo de ejecución: {execution_time} segundos para {input_file}")

        # Obtener información sobre el uso de recursos
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        gpu_usage = gputil.getGPUs()[0].load * 100 if torch.cuda.is_available() else "N/A"

        # Escribe el texto y las métricas en el archivo de salida
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(transcription)
            file.write("\n\n")
            file.write(f"Tiempo de ejecución: {execution_time} segundos\n")
            file.write(f"Uso de CPU: {cpu_usage}%\n")
            file.write(f"Uso de RAM: {ram_usage}%\n")
            file.write(f"Uso de GPU: {gpu_usage}%\n")

        print(f"\nTranscripcion guardada en {output_file}")

# Ejemplo de inicialización y uso de la clase
transcriber = ASRTranscriber(
    model_id="openai/whisper-large-v3",
    batch_size=16,
    chunk_length_s=10,
    max_new_tokens=128
)

# Transcribe cada archivo .wav en el directorio especificado
directory_path = r"C:\Users\Atexis\Documents\STT"
for filename in os.listdir(directory_path):
    if filename.endswith(".wav"):
        input_path = os.path.join(directory_path, filename)
        output_file = f"{os.path.splitext(filename)[0]}_transcription.txt"
        output_path = os.path.join(directory_path, output_file)
        transcriber.transcribe_and_save(input_path, output_path)