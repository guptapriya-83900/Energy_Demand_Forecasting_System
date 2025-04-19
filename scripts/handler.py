import torch
import json

class CNNLSTMHandler:
    def __init__(self):
        self.model = None

    def initialize(self, ctx):
        model_dir = ctx.system_properties.get("model_dir")
        self.model = torch.jit.load(f"{model_dir}/cnn_lstm_model_optimized.pt")
        self.model.eval()

    def preprocess(self, data):
        input_data = json.loads(data[0]["body"])
        tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # shape: (1, sequence_length)
        return tensor

    def inference(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor).squeeze().tolist()
        return [output]

    def postprocess(self, output):
        return [{"prediction": output}]
