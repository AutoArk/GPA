# -*- coding: utf-8 -*-
import requests
import wave
import os
import sys

# Server Address
SERVER_URL = "http://localhost:8020"
SAMPLE_RATE = 16000 

def save_pcm_as_wav(pcm_chunks, output_name, channels=1, sampwidth=2, framerate=SAMPLE_RATE):
    """
    Save PCM binary stream as WAV file
    :param sampwidth: 2 bytes (16-bit)
    """
    print(f"Saving audio to: {output_name} ...")
    with wave.open(output_name, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        for chunk in pcm_chunks:
            if chunk:
                wf.writeframes(chunk)
    print(f"Saved successfully: {output_name}")


def test_stt(audio_path, temperature=0.0, repetition_penalty=1.0, do_sample=False):
    """
    Test STT Interface
    :param temperature: Sampling temperature, ASR tasks usually suggest lower (e.g. 0.0 - 0.2)
    :param repetition_penalty: Repetition penalty, ASR tasks usually set to 1.0 (no penalty) or 1.1
    """
    print("\n" + "="*30)
    print(f"--- Test STT (Speech to Text) ---")
    print(f"Input Audio: {audio_path}")
    print(f"Parameters: temp={temperature}, rep_pen={repetition_penalty}")

    url = f"{SERVER_URL}/stt"

    try:
        # Construct form parameters
        data = {
            "temperature": temperature,
            "repetition_penalty": repetition_penalty
        }

        # Open file and send request
        with open(audio_path, 'rb') as f:
            files = {'file': f}
            # stream=True Allow streaming response to avoid loading large files into memory at once
            response = requests.post(url, files=files, data=data, stream=True)

            if response.status_code == 200:
                print("Recognition Result: ", end="", flush=True)
                # Even if it is a text stream, it can be read in chunks
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        print(chunk, end="", flush=True)
                print() # New line
            else:
                print(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Request Failed: {e}")


def test_tts_a(
    text, ref_audio_path, temperature=0.2, repetition_penalty=1.2, do_sample=True
):
    """
    Test TTS Mode A (Reference Audio Cloning)
    :param temperature: Generation tasks suggest 0.6 - 0.9, higher is more random
    :param repetition_penalty: Suggest 1.1 - 1.2 to prevent repetition
    """
    print("\n" + "="*30)
    print(f"--- Test TTS Mode A (Voice Cloning) ---")
    print(f"Input Text: {text}")
    print(f"Reference Audio: {ref_audio_path}")
    print(f"Parameters: temp={temperature}, rep_pen={repetition_penalty}")

    url = f"{SERVER_URL}/tts/a"
    output_filename = "result_tts_a.wav"

    try:
        # Prepare multipart/form-data
        # data stores normal fields, files stores files
        data = {
            "text": text,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "do_sample": str(do_sample).lower(),
        }

        with open(ref_audio_path, 'rb') as f:
            files = {
                "ref_audio": f
            }

            # stream=True Very important for audio stream
            response = requests.post(url, data=data, files=files, stream=True)

            if response.status_code == 200:
                # Receive streaming PCM and save as WAV
                save_pcm_as_wav(response.iter_content(chunk_size=4096), output_filename)
            else:
                print(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Request Failed: {e}")


if __name__ == "__main__":
    # === Configure test file path ===
    # Please ensure these files exist on the machine running the script
    TEST_REF_WAV = "../inference/test_audio/vc_src.wav"
    
    # Check if file exists
    if not os.path.exists(TEST_REF_WAV):
        print(f"Error: Test audio file not found: {TEST_REF_WAV}")
        print("Please modify the TEST_REF_WAV path in the script to an actual existing wav file path.")
        sys.exit(1)

    # 1. Test STT (ASR usually sets temperature low to ensure accuracy)
    test_stt(TEST_REF_WAV, temperature=0.0, repetition_penalty=1.0)
    
    # 2. Test TTS A (TTS generation usually requires a bit of temperature to increase naturalness)
    test_text = "Hello, I am GPA. This is a test of the voice cloning function, adding sampling parameter control."
    test_tts_a(test_text, TEST_REF_WAV, temperature=0.8, repetition_penalty=1.0)
