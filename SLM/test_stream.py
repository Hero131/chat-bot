from model_utils import load_model_and_tokenizer, generate_response_stream
import traceback

try:
    print("Loading model...")
    client, _, _ = load_model_and_tokenizer("qwen", reasoning_mode="off")
    print("Model loaded.")

    messages = [
        {"role": "user", "content": "yo"}
    ]

    print("Generating response...")
    streamer = generate_response_stream(
        client,
        "qwen",
        messages,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        max_history_messages=12,
    )

    print("Streaming output:")
    for chunk in streamer:
        print(repr(chunk))
    print("\nDone.")
except Exception as e:
    traceback.print_exc()
