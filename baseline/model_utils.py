from transformers import AutoModelForCausalLM, AutoTokenizer
import logging


def load_model_and_tokenizer(model_name: str):
    """
    Load model and tokenizer from Hugging Face.

    Args:
        model_name (str): Model name or path.

    Returns:
        tokenizer, model: Loaded tokenizer and model instances.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Handle the pad token issue by assigning `pad_token` to `eos_token` if missing
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        return tokenizer, model
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {str(e)}")
        exit(1)


def generate_response(prompt: str, tokenizer, model, max_length: int) -> str:
    """
    Generate response using the model.

    Args:
        prompt (str): Input prompt for the model.
        tokenizer: Tokenizer instance.
        model: Model instance.
        max_length (int): Maximum length of the generated response.

    Returns:
        str: Generated response.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        logging.error(f"Error generating response for prompt: {prompt}. Error: {str(e)}")
        return ""
