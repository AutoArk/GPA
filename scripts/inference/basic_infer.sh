export GPA_MODEL_DIR="/data3/gpa_release/gpa_0_3b"

python gpa_inference.py --task vc \
    --src_audio_path "test_audio/vc_src.wav" \
    --ref_audio_path "test_audio/astro.wav" \
    --gpa_model_path "${GPA_MODEL_DIR}" \
    --tokenizer_path "${GPA_MODEL_DIR}/glm-4-voice-tokenizer" \
    --bicodec_tokenizer_path "${GPA_MODEL_DIR}/BiCodec" \
    --text_tokenizer_path "${GPA_MODEL_DIR}"