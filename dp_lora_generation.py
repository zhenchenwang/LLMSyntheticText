import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from peft import LoraConfig, get_peft_model
from opacus import PrivacyEngine

# Custom Dataset to ensure compatibility with Opacus
class CustomDataset(TorchDataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # For Causal LM, labels are the same as input_ids
        return item['input_ids'], item['attention_mask'], item['input_ids']

def main():
    # NOTE: Using distilgpt2 as TinyLlama is too large for this environment.
    model_name = "distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    data = {
        "report": [
            "The patient reported severe headache and nausea after using the new cardiac stent.",
            "Device model X failed during a critical procedure, requiring immediate surgical intervention.",
        ],
        "summary": [
            "Summary: Patient experienced headache and nausea with cardiac stent.",
            "Summary: Device failure required surgical intervention.",
        ]
    }
    dataset = Dataset.from_dict(data)

    def format_prompt(sample):
        return f"### Report:\n{sample['report']}\n\n### Summary:\n{sample['summary']}"

    def tokenize_function(batch):
        prompts = [format_prompt({"report": r, "summary": s}) for r, s in zip(batch["report"], batch["summary"])]
        return tokenizer(prompts, padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["report", "summary"])
    tokenized_dataset.set_format("torch")

    pytorch_dataset = CustomDataset(tokenized_dataset)

    # Configure LoRA and DP-SGD
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    data_loader = DataLoader(pytorch_dataset, batch_size=1, shuffle=True)

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_delta=1e-5,
    )

    # Train the Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    print("\nStarting training...")
    total_loss = 0
    for input_ids, attention_mask, labels in data_loader:
        if input_ids.size(0) == 0: continue

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=labels.to(device),
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Training complete. Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.2f}")

    # Generate Text
    model.eval()
    prompt_text = "### Report:\nThe patient experienced a sudden drop in blood pressure after the device was activated.\n\n### Summary:"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    print("\nGenerating summary...")
    output = model._module.generate(**inputs, max_new_tokens=20)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("--- Generated Text ---")
    print(generated_text)

if __name__ == "__main__":
    main()