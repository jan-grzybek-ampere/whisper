import time
import torch
import whisper
from whisper.transcribe import transcribe

#torch.jit.set_fusion_strategy([("DYNAMIC", 1)])
model = whisper.load_model("base.en")
model.eval()

a = time.time()
with torch.no_grad():
    result = transcribe(model, "some_speech.wav", verbose=True)
    print(result["text"])
print(time.time() - a)
