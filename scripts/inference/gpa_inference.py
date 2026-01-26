import sys
import os
import argparse
import torch
import soundfile as sf
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperFeatureExtractor
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.bicodec_tokenizer.spark_tokenizer import SparkTokenizer
from models.bicodec_tokenizer.spark_detokenizer import SparkDeTokenizer

from models.glm_speech_tokenizer.speech_token_extractor import SpeechTokenExtractor
from models.glm_speech_tokenizer.modeling_whisper import WhisperVQEncoder

from data_utils.audio_dataset_ark_audio import ark_infer_processor

class GPAInference:
    def __init__(self, tokenizer_path, text_tokenizer_path, bicodec_tokenizer_path, gpa_model_path, output_dir, device):
        self.tokenizer_path = tokenizer_path
        self.text_tokenizer_path = text_tokenizer_path
        self.bicodec_tokenizer_path = bicodec_tokenizer_path
        self.gpa_model_path = gpa_model_path
        self.output_dir = output_dir
        self.device = device

        print(f"Using device: {self.device}")
        self._load_models()

    def _load_models(self):
        print("Loading tokenizers...")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(self.tokenizer_path)
        audio_model = WhisperVQEncoder.from_pretrained(self.tokenizer_path).eval().to(self.device)
        self.glm_tokenizer = SpeechTokenExtractor(model=audio_model, feature_extractor=feature_extractor, device=self.device)
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            self.text_tokenizer_path, 
            trust_remote_code=True
        )

        self.bicodec_tokenizer = SparkTokenizer(model_path=self.bicodec_tokenizer_path, device=self.device)
        self.bicodec_detokenizer = SparkDeTokenizer(model_path=self.bicodec_tokenizer_path, device=self.device)
        self.processor = ark_infer_processor(
            glm_tokenizer=self.glm_tokenizer,
            bicodec_tokenizer=self.bicodec_tokenizer,
            text_tokenizer=self.text_tokenizer,
            device=self.device,
            audio_path_name="audio",
        )

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.gpa_model_path, 
            trust_remote_code=True
        ).to(self.device)

    def generate(self, inputs, **kwargs):
        """
        Base generation method that accepts dynamic generation parameters.
        """
        for k in inputs:
            if isinstance(inputs[k], (list, np.ndarray)):
                inputs[k] = torch.tensor(inputs[k]).unsqueeze(0).to(self.device)
            elif isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].unsqueeze(0).to(self.device)

        # Default generation config
        generation_config = {
            "max_new_tokens": 1000,
            "do_sample": False,
            "eos_token_id": self.text_tokenizer.convert_tokens_to_ids("<|im_end|>"),
        }

        # Override defaults with any passed kwargs
        generation_config.update(kwargs)

        # Remove keys that might be None if passed from args mistakenly
        generation_config = {k: v for k, v in generation_config.items() if v is not None}
        print(f"Generation config: {generation_config}")

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generation_config
        )
        return outputs

    def run_stt(self, audio_path, **kwargs):
        if not audio_path:
            raise ValueError("audio_path is required for STT")

        print("\n--- Speech to Text (STT) ---")

        inputs = self.processor.process_input(
            task="stt",
            audio_path=audio_path,
        )

        # recommend hyperparameters for TTS
        kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
        }

        # Pass generation arguments (temperature, etc.) to generate
        outputs = self.generate(inputs, **kwargs)
        text = self.text_tokenizer.decode(outputs[0].tolist())

        if "<|start_content|>" in text:
            return text.split("<|start_content|>")[1].replace("<|im_end|>","").replace("<|end_content|>","")
        else:
            return text.replace("<|im_end|>","")

    def run_tts(self, task, output_filename, text, ref_audio_path, **kwargs):
        """
        gen_kwargs: dict, parameters for model.generate (temp, top_p, etc.)
        """
        if not text:
            raise ValueError("text is required for TTS")

        # Check ref_audio_path requirement based on task
        if task == "tts-a" and not ref_audio_path:
            raise ValueError(f"ref_audio_path is required for {task}")

        # recommend hyperparameters for TTS
        kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.2,
            "repetition_penalty": 1.2,
            "do_sample": True,
        }

        print(f"\n--- {task.upper()} ---")
        output_path = os.path.join(self.output_dir, output_filename)

        # Pass processor specific args (e.g. emotion, pitch) here
        inputs = self.processor.process_input(
            task=task, 
            ref_audio_path=ref_audio_path, 
            text=text,
        )

        # Pass generation specific args (e.g. temperature) here
        # Note: Original code hardcoded temperature=0.8 for TTS, we use gen_kwargs or fallback to generate defaults
        outputs = self.generate(inputs, **kwargs)

        text_output = self.text_tokenizer.decode(outputs[0].tolist())

        if "<|end_content|>" in text_output:
            content = text_output.split("<|end_content|>")[1]
        else:
            print("Warning: <|end_content|> not found")
            content = text_output

        audio_ids = re.findall(r"<\|bicodec_semantic_(\d+)\|>", content)
        audio_list = [int(x) for x in audio_ids]

        if ref_audio_path:
            global_tokens = self.bicodec_tokenizer.tokenize([ref_audio_path])['global_tokens']
        else:
            global_tokens = torch.zeros((1, 32), dtype=torch.long).to(self.device)

        req = {
            "global_tokens": global_tokens,
            "semantic_tokens": torch.tensor(audio_list).unsqueeze(0).to(self.device),
        }
        out = self.bicodec_detokenizer.detokenize(**req)
        reconstructed_wav = out.detach().cpu().float().squeeze().numpy()
        # Simple DC offset removal
        if reconstructed_wav.size > 0:
            reconstructed_wav -= reconstructed_wav.mean()

        sf.write(output_path, reconstructed_wav, 16000)
        print(f"Saved output to {output_path}")
        return 16000, reconstructed_wav

    def run_vc(
        self,
        source_audio_path,
        ref_audio_path,
        output_filename="output_gpa_vc.wav",
        **kwargs,
    ):
        if not source_audio_path:
            raise ValueError("source_audio_path is required for VC")
        if not ref_audio_path:
            raise ValueError("ref_audio_path is required for VC")

        print("\n--- Voice Conversion (VC) ---")
        output_path = os.path.join(self.output_dir, output_filename)

        kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.2,
            "repetition_penalty": 1.2,
            "do_sample": True,
        }

        inputs = self.processor.process_input(
            task="vc",
            audio_path=source_audio_path,
            ref_audio_path=ref_audio_path,
        )

        outputs = self.generate(inputs, **kwargs)
        text_output = self.text_tokenizer.decode(outputs[0].tolist())

        if "<|end_content|>" in text_output:
            content = text_output.split("<|end_content|>")[1]
        else:
            content = text_output

        audio_ids = re.findall(r"<\|bicodec_semantic_(\d+)\|>", content)
        audio_list = [int(x) for x in audio_ids]

        global_tokens = self.bicodec_tokenizer.tokenize([ref_audio_path])['global_tokens']

        req = {
            "global_tokens": global_tokens,
            "semantic_tokens": torch.tensor(audio_list).unsqueeze(0).to(self.device),
        }
        out = self.bicodec_detokenizer.detokenize(**req)
        reconstructed_wav = out.detach().cpu().float().squeeze().numpy()
        if reconstructed_wav.size > 0:
            reconstructed_wav -= reconstructed_wav.mean()

        sf.write(output_path, reconstructed_wav, 16000)
        print(f"Saved VC output to {output_path}")
        return 16000, reconstructed_wav


def parse_args():
    parser = argparse.ArgumentParser(description="GPA Inference Script")

    # Paths
    parser.add_argument("--tokenizer_path", type=str, default="/nasdata/model/gpa/glm-4-voice-tokenizer", help="Path to GLM4 tokenizer")
    parser.add_argument("--text_tokenizer_path", type=str, default="/nasdata/model/gpa", help="Path to text tokenizer")
    parser.add_argument("--bicodec_tokenizer_path", type=str, default="/nasdata/model/gpa/BiCodec/", help="Path to BiCodec tokenizer")
    parser.add_argument("--gpa_model_path", type=str, default="/nasdata/model/gpa", help="Path to GPA model")

    # Audio inputs
    parser.add_argument(
        "--ref_audio_path", type=str, default=None, help="Reference audio path"
    )
    parser.add_argument(
        "--src_audio_path", type=str, default=None, help="Source audio path for VC/STT"
    )

    # Output
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output files")

    # Device
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device, help="Device to use (e.g., cuda:0, cpu)")

    # Task
    parser.add_argument("--task", type=str, required=True, choices=["stt", "tts-a", "vc"], help="Task to run")

    # TTS Inputs (Processor Arguments)
    parser.add_argument("--text", type=str, default=None, help="Text for TTS")

    return parser.parse_args()

def main():
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    inference = GPAInference(
        tokenizer_path=args.tokenizer_path,
        text_tokenizer_path=args.text_tokenizer_path,
        bicodec_tokenizer_path=args.bicodec_tokenizer_path,
        gpa_model_path=args.gpa_model_path,
        output_dir=args.output_dir,
        device=args.device,
    )

    if args.task == "stt":
        if not args.src_audio_path:
            raise ValueError("Error: --src_audio_path is required for STT task.")
        # Pass gen_kwargs
        result = inference.run_stt(audio_path=args.src_audio_path)
        print("STT Result:", result)

    elif args.task == "tts-a":
        inference.run_tts(
            task="tts-a",
            output_filename="output_gpa_tts_a.wav",
            text=args.text,
            ref_audio_path=args.ref_audio_path,
        )

    elif args.task == "vc":
        inference.run_vc(
            source_audio_path=args.src_audio_path,
            ref_audio_path=args.ref_audio_path,
            output_filename="output_gpa_vc.wav",
        )

if __name__ == "__main__":
    main()
