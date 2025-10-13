import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from peft import LoraConfig, get_peft_model
from opacus import PrivacyEngine


def main():
    # Step 1: Data Preparation and LLM Selection
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    data = {
        "text": [
            "The patient reported severe headache and nausea after using the new cardiac stent.",
            "Device model X failed during a critical procedure, requiring immediate surgical intervention.",
            "No adverse effects were observed in the control group using the standard device.",
            "Post-market surveillance data shows a recurring issue with the device's battery life.",
            "The user manual for the device was found to be unclear, leading to improper usage.",
            "A software glitch in the infusion pump led to an incorrect dosage being administered.",
            "The patient experienced a mild allergic reaction at the site of the implant.",
            "Clinical trial results indicate the device is safe and effective for its intended use.",
            "The device's alarm system failed to activate during a simulated power outage.",
            "Follow-up reports confirm the issue was resolved with a firmware update."
        ],
        "label": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    }
    dataset = Dataset.from_dict(data)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")

    # Define a custom PyTorch Dataset that returns tuples, which is compatible with Opacus.
    class AdverseEventDataset(TorchDataset):
        def __init__(self, hf_dataset):
            self.data = hf_dataset

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            return (
                item["input_ids"],
                item["attention_mask"],
                item["label"],
            )

    pytorch_dataset = AdverseEventDataset(tokenized_dataset)
    train_dataloader = DataLoader(pytorch_dataset, batch_size=2, shuffle=True)

    # Step 2: Configure LoRA Adapters
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_lin", "v_lin"], lora_dropout=0.1, bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Step 3: Integrate Differential Privacy (DP-SGD)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    privacy_engine = PrivacyEngine(accountant="rdp")

    model, optimizer, train_dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_delta=1e-5,
    )

    print("LoRA and DP-SGD configuration complete.")

    # Step 4: Private Fine-Tuning and Evaluation
    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        # The following line is updated to remove tqdm
        for input_ids, attention_mask, labels in train_dataloader:
            # Skip empty batches, which can occur with Poisson sampling
            if input_ids.size(0) == 0:
                continue
            optimizer.zero_grad()

            # The data is already on the correct device because of the DPDataLoader.
            outputs = model(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"Epoch {epoch + 1}/{num_epochs} | Avg Training Loss: {avg_train_loss:.4f} | Epsilon: {epsilon:.2f}")

    print("Training complete.")

    # Basic Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        eval_dataloader = DataLoader(pytorch_dataset, batch_size=2)
        for input_ids, attention_mask, labels in eval_dataloader:
            outputs = model(
                input_ids.to(device), attention_mask=attention_mask.to(device)
            )
            predictions = torch.argmax(outputs.logits, dim=-1)

            total += labels.size(0)
            correct += (predictions == labels.to(device)).sum().item()

    accuracy = 100 * correct / total
    print(f"Training Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()