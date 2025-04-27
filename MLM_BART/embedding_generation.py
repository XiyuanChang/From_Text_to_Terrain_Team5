# embland_greater_dis_5classes.py

import argparse
import torch
import json
from transformers import BartTokenizer, BartModel

def generate_embeddings(model, tokenizer, dataset,  device="cuda"):
    embeddings = []
    batch_size = 16
    texts = [item["original"] for item in dataset]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            output = model(**inputs)
        sent_emb = output.last_hidden_state.mean(dim=1)
        embeddings.append(sent_emb.cpu())

    return torch.cat(embeddings, dim=0)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BartTokenizer.from_pretrained(args.model_path)
    model = BartModel.from_pretrained(args.model_path)
    model.eval()
    model.to(device)

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    emb_org = generate_embeddings(model, tokenizer, dataset,  device=device)

    print(f"Shape of generated embeddings: {emb_org.shape}")

    torch.save(emb_org, args.output_path)
    print(f"Embeddings saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BART embeddings from masked PFAS dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained BART model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the masked JSON dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the generated embeddings (.pt).")
    
    args = parser.parse_args()

    main(args)
