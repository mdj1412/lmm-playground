# The script performs the following steps:

# 1. Downloads the dataset from Hugging Face using the `load_dataset()` function.
# 2. Converts the Hugging Face dataset to a Pandas DataFrame for easier manipulation using the `to_pandas()` method.
# 3. Creates directories to save the dataset and images.
# 4. Filters out rows where image download fails by iterating through each row in the DataFrame, downloading the image using the custom `download_image()` function, and appending the filtered row to a new DataFrame called `filtered_rows`.
# 5. Creates a new DataFrame with the filtered rows and saves it to disk as a CSV file.
# 6. Prints a message indicating where the dataset and images have been saved.import os

import argparse
import pandas as pd
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
import os

# Function to download an image from a URL and save it locally
def download_image(image_url, save_path):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Check if the request was successful
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
        return True
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        return False


def main(dataset_name: str, split: str):
    # Download the dataset from Hugging Face
    # Simply replace DataSet with the Hugging Face DataSet name
    # Example. dataset = load_dataset('DBQ/Burberry.Product.prices.United.States')
    dataset = load_dataset(f'../data/7_distil/{dataset_name}')

    # Convert the Hugging Face dataset to a Pandas DataFrame
    df = dataset[split].to_pandas() # df = dataset['train'].to_pandas()

    # Create directories to save the dataset and images to a folder
    # Example. dataset_dir = './data/burberry_dataset'
    dataset_dir = f'./data/{dataset_name}/{split}'

    images_dir = os.path.join(dataset_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Filter out rows where image download fails
    filtered_rows = []
    for idx, row in df.iterrows():
        # image_url = row['imageurl']
        image_name = row['file_name']
        image = Image.open(BytesIO(row['image']['bytes']))
        image_path = os.path.join(images_dir, image_name)

        image.save(image_path)
        row['local_image_path'] = image_path

        row['segmentation'] = row['segmentation'].tolist()
        row['answer'] = row['answer'].tolist()
        row['bbox'] = row['bbox'].tolist()
        
        filtered_rows.append(row.drop(index=['image']))
        # if download_image(image_url, image_path):
        #     row['local_image_path'] = image_path
        #     filtered_rows.append(row)

    # Create a new DataFrame with the filtered rows
    filtered_df = pd.DataFrame(filtered_rows)

    # Save the updated dataset to disk in a CSV format
    # Example. dataset_path = os.path.join(dataset_dir, 'burberry_dataset.csv')
    # dataset_path = os.path.join(dataset_dir, 'burberry_dataset.csv')
    dataset_path = os.path.join(dataset_dir, 'Dataset.csv')

    filtered_df.to_csv(dataset_path, index=False)

    print(f"Dataset and images saved to {dataset_dir}")

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

    main(dataset_name=args.dataset_name, split=args.split)