import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Replace 'your-username/your-model-name' with your actual model path on Hugging Face Hub
model_name = 'nefro313/HR-qa-chat-llama2-finetuned-model'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto'
)

# Initialize the text generation pipeline


# Define your prompt
prompt = "Why do you want to leave your present job.Why are you looking for a job change?"
text_generator = pipeline(task="text-generation", model=model ,tokenizer=tokenizer, max_length=500)


# Generate text
generated_text = text_generator(f'how to answers this question."{prompt}"')

# Output the generated text