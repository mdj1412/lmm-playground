# Import necessary libraries
# Code orginally from https://wandb.ai/byyoung3/mlnews3/reports/How-to-fine-tune-Phi-3-vision-on-a-custom-dataset--Vmlldzo4MTEzMTg3 
# Credits to: Brett Young https://github.com/bdytx5/

import os
import torch
import argparse
import ast
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoProcessor, get_linear_schedule_with_warmup
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import wandb
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image

import torch.optim as optim
import torch.nn.functional as F
from functools import partial
from utils.loss import (
    get_sft_loss,
    get_digit_loss,
    get_digit_base_loss,
    is_number_regex,
)

# Custom Dataset class for Burberry Product Prices and Images
class BurberryProductDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, image_size):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'  # Set padding side to left
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx): # ex. train_dataset[0], val_dataset[3]
        # Get the row at the given index
        row = self.dataframe.iloc[idx]

        answers = ast.literal_eval(row['answer'])
        segments = ast.literal_eval(row['segmentation'])

        # Create the text input for the model
        # text = f"<|user|>\n<|image_1|>What is shown in this image?<|end|><|assistant|>\nProduct: {row['title']}, Category: {row['category3_code']}, Full Price: {row['full_price']}<|end|>"
        user_text = f"<|user|>\n<|image_1|>Please identify and segment the {answers} in this image. \
            For each object, provide the coordinates of the polygon that outlines the object, in the format: [x1, y1, x2, y2, ..., xn, yn], where n is less than or equal to 20. \
            Ensure the coordinates are precise and correspond to each object's position in the image.<|end|>"
        assistant_text = f"<|assistant|>\nThe coordinates for the {answers} are approximately {segments}<|end|>"
        text = user_text + assistant_text

        # Get the image path from the row
        image_path = row['local_image_path']
        
        # Tokenize the text input
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length)
        
        try:
            # Load and transform the image
            image = Image.open(image_path).convert("RGB")
            image = self.image_transform_function(image)
        except (FileNotFoundError, IOError):
            # Skip the sample if the image is not found
            return None
        
        # Add the image and price information to the encodings dictionary
        encodings['pixel_values'] = image
        # label_encoding = self.tokenizer(assistant_text, truncation=True, padding='max_length', max_length=self.max_length)
        # encodings['price'] = row['full_price'] # TODO: ??
        
        return {key: torch.tensor(val) for key, val in encodings.items()}

    def image_transform_function(self, image):
        # Convert the image to a numpy array
        image = np.array(image)
        return image

# Function to extract the predicted price from model predictions
def extract_price_from_predictions(predictions, tokenizer):
    # Assuming the price is at the end of the text and separated by a space
    predicted_text = tokenizer.decode(predictions[0], skip_special_tokens=True)
    try:
        predicted_price = float(predicted_text.split()[-1].replace(',', ''))
    except ValueError:
        predicted_price = 0.0
    return predicted_price

# Function to evaluate the model on the validation set
def evaluate(model, val_loader, device, tokenizer, step, log_indices, max_samples=None):
    model.eval()
    total_loss = 0
    total_price_error = 0
    log_images = []
    log_gt_texts = []
    log_pred_texts = []
    table = wandb.Table(columns=["Image", "Ground Truth Text", "Predicted Text"])

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_samples and i >= max_samples:
                break

            if batch is None:  # Skip if the batch is None
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = input_ids.clone().detach()
            # actual_price = batch['price'].item()

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            # Calculate price error
            predictions = torch.argmax(outputs.logits, dim=-1)
            # predicted_price = extract_price_from_predictions(predictions, tokenizer)
            # price_error = abs(predicted_price - actual_price)
            # total_price_error += price_error

            # Log images, ground truth texts, and predicted texts
            if i in log_indices:
                log_images.append(pixel_values.cpu().squeeze().numpy())
                log_gt_texts.append(tokenizer.decode(labels[0], skip_special_tokens=True))
                log_pred_texts.append(tokenizer.decode(predictions[0], skip_special_tokens=True))

                # Convert image to PIL format
                pil_img = to_pil_image(resize(torch.from_numpy(log_images[-1]).permute(2, 0, 1), (336, 336))).convert("RGB")
                
                # Add data to the table
                table.add_data(wandb.Image(pil_img), log_gt_texts[-1], log_pred_texts[-1])

                # Log the table incrementally
    wandb.log({"Evaluation Results step {}".format(step): table, "Step": step})

    avg_loss = total_loss / (i + 1)  # i+1 to account for the loop index
    # avg_price_error = total_price_error / (i + 1)
    model.train()

    return avg_loss#, avg_price_error

def main(dataset_name:str, split:str, epochs:int, lr:float, loss_fn:str, save_dir: str):
    torch.manual_seed(3)

    # Initialize Weights & Biases for experiment tracking
    run = wandb.init(project="burberry-product-phi3", entity="mdj1412")

    # Load dataset from disk
    dataset_path = f'./data/{dataset_name}/{split}/Dataset.csv' # dataset_path = './data/burberry_dataset/burberry_dataset.csv'
    
    dataset_path = f'./data/RefCOCO/val/Dataset.csv'
    df1 = pd.read_csv(dataset_path)
    dataset_path = f'./data/RefCOCOg/val/Dataset.csv'
    df2 = pd.read_csv(dataset_path)
    dataset_path = f'./data/RefCOCOplus/val/Dataset.csv'
    df3 = pd.read_csv(dataset_path)

    df = pd.concat([df1, df2, df3], ignore_index=True)

    # Initialize processor and tokenizer for the pre-trained model
    model_id = "microsoft/Phi-3-vision-128k-instruct"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # Split dataset into training and validation sets
    train_size = int(0.9 * len(df))
    val_size = len(df) - train_size
    train_indices, val_indices = random_split(range(len(df)), [train_size, val_size])
    train_indices = train_indices.indices
    val_indices = val_indices.indices
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]

    # Create dataset and dataloader for training set
    train_dataset = BurberryProductDataset(train_df, tokenizer, max_length=512, image_size=128)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Create dataset and dataloader for validation set
    val_dataset = BurberryProductDataset(val_df, tokenizer, max_length=512, image_size=128)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(model_id, 
        device_map="cuda", 
        # device_map = 'auto', 
        trust_remote_code=True, torch_dtype="auto")

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # TODO
    warmnup_ratio, beta, temperature = 0.01, 1.0, 2.0
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_training_steps=epochs * len(train_dataset),
        num_warmup_steps=int(epochs * len(train_dataset) * warmnup_ratio),
        last_epoch=-1,
    )
    loss_fn = {
        "sft": get_sft_loss,
        "digit": partial(
            get_digit_loss,
            tokenizer=tokenizer,
            target_temperature=temperature,
            beta=beta,
        ),
        "digit_base": partial(get_digit_base_loss, tokenizer=tokenizer, beta=beta),
    }[loss_fn]

    # Training loop
    num_epochs = epochs
    eval_interval = 150  # Evaluate every 'eval_interval' steps
    loss_scaling_factor = 1000.0  # Variable to scale the loss by a certain amount
    step = 0
    accumulation_steps = 64  # Accumulate gradients over this many steps

    # Create a directory to save the best model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_val_loss = float('inf')
    best_model_path = None

    # Select 10 random images from the validation set for logging
    num_log_samples = 10
    log_indices = random.sample(range(len(val_dataset)), num_log_samples)

    # Set the model to training mode
    model.train()

    # Training loop for the specified number of epochs
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_train_price_error = 0
        batch_count = 0

        for batch in train_loader:
            step += 1

            if batch is None:  # Skip if the batch is None
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = input_ids.clone().detach()

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                labels=labels
            )
            loss = loss_fn(outputs.logits, labels, batch["attention_mask"]).mean() # loss = outputs.loss
            total_loss = loss
            predictions = torch.argmax(outputs.logits, dim=-1)

            total_loss.backward()# Performs the backpropagation step to calculate the gradients.

            if (step % accumulation_steps) == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= accumulation_steps
                optimizer.step()# Updates the model’s weights using the calculated gradients.
                optimizer.zero_grad()
                scheduler.step()# Updates the learning rate scheduler to adjust the learning rate for the next step or epoch.

            total_train_loss += total_loss.item()
            # total_train_price_error += abs(predicted_price - actual_price.item())
            batch_count += 1

            # Log batch loss to Weights & Biases
            wandb.log({"Batch Loss": total_loss.item(), "Step": step})

            print(f"Epoch: {epoch}, Step: {step}, Batch Loss: {total_loss.item()}")

            if step % eval_interval == 0:
                # val_loss, val_price_error = evaluate(model, val_loader, device, tokenizer=tokenizer, log_indices=log_indices, step=step )
                val_loss = evaluate(model, val_loader, device, tokenizer=tokenizer, log_indices=log_indices, step=step )
                wandb.log({
                    "Validation Loss": val_loss,
                    # "Validation Price Error (Average)": val_price_error,
                    "Step": step
                })
                print(f"Step: {step}, Validation Loss: {val_loss},")# Validation Price Error (Normalized): {val_price_error}")

                best_model_path = os.path.join(f"{save_dir}_lr{lr}", f"best_model_epoch{epoch}_step{step}")
                print (f"command line: {best_model_path}")

                # Save the best model based on validation loss
                # if (val_loss < best_val_loss) and (val_loss <= 0.4124):
                if (val_loss < best_val_loss) and (val_loss <= 0.96):
                    best_val_loss = val_loss
                    best_model_path = os.path.join(f"{save_dir}_lr{lr}", f"best_model_epoch{epoch}_step{step}")
                    print (f"Save model at {best_model_path}")
                    model.save_pretrained(best_model_path, safe_serialization=False)
                    tokenizer.save_pretrained(best_model_path)

                avg_train_loss = total_train_loss / batch_count
                # avg_train_price_error = total_train_price_error / batch_count
                wandb.log({
                    "Epoch": epoch,
                    "Average Training Loss": avg_train_loss,
                    # "Average Training Price Error": avg_train_price_error
                })
                
        print(f"Epoch: {epoch}, Average Training Loss: {avg_train_loss}")#, Average Training Price Error: {avg_train_price_error}")

        # Log the best model to Weights & Biases
        if best_model_path:
            run.log_model(
                path=best_model_path,
                name="phi3-v-burberry",
                aliases=["best"],
            )

    # Finish the Weights & Biases run
    wandb.finish()


if __name__ == '__main__':
    # 1. create parser
    parser = argparse.ArgumentParser()

    # 2. add arguments to parser
    parser.add_argument('--dataset-name',
                        type=str,
                        default="RefCOCO",
                        choices=["RefCOCO", "RefCOCOg", "RefCOCOplus", "Ferret-Bench"],
                        help="Input dataset name.")
    parser.add_argument('--split',
                        type=str,
                        default="validation",
                        choices=["validation", "test", "testA", "testB", "val"],
                        help="Choose validation or test or testB.")
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help="Input train epochs")
    parser.add_argument('--learning-rate',
                        type=float,
                        default=5e-5,
                        help="Input learning rate")
    parser.add_argument('--loss_fn',
                        type=str,
                        default="sft",
                        choices=["sft", "digit", "digit_base"],
                        help="Choose sft, digit, or digit_base.")
    parser.add_argument('--save-dir',
                        type=str,
                        default='./saved_models',
                        help="Choose validation or test or testB.")

    # 3. parse arguments
    args = parser.parse_args()

    # 4. use arguments
    print (args)

    if args.dataset_name == "RefCOCO":
        assert args.split in ['val', 'test', 'testA', 'testB'], "Choose val, test, testA, or testB in lmms-lab/refcoco"
    elif args.dataset_name == "RefCOCOg":
        assert args.split in ['val', 'test'], "Choose val, or test  in lmms-lab/RefCOCOg"
    elif args.dataset_name == "RefCOCOplus":
        assert args.split in ['val', 'testA', 'testB'], "Choose val, testA, or testB in lmms-lab/RefCOCOplus"
    elif args.dataset_name == "Ferret-Bench":
        assert args.split in ['test'], "Choose only test split in lmms-lab/Ferret-Bench"
    else:
        raise NotImplementedError()

    main(
        dataset_name=args.dataset_name,
        split=args.split,
        epochs=args.epochs,
        lr=args.learning_rate,
        loss_fn=args.loss_fn,
        save_dir=args.save_dir)
    # python generate_dataset.py --dataset-name RefCOCO --split val
    # python generate_dataset.py --dataset-name RefCOCOplus --split val
    # python generate_dataset.py --dataset-name RefCOCOg --split val
    # python phi3vision_train.py --dataset-name RefCOCO --split val --epochs 3 --save-dir saved_models/digit_loss --loss_fn digit --learning-rate  0.0001
    # python phi3vision_train.py --dataset-name RefCOCO --split val --epochs 3 --save-dir saved_models/sft --loss_fn sft