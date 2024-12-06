# data_prep.py
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets
from huggingface_hub import HfApi, HfFolder, create_repo

class DatasetConverterUploader:
    def __init__(self, dataset_base_dir, repo_id_base):
        self.dataset_base_dir = dataset_base_dir
        self.repo_id_base = repo_id_base
        self.dataset_directories = [
            os.path.join(dataset_base_dir, d) for d in os.listdir(dataset_base_dir) if os.path.isdir(os.path.join(dataset_base_dir, d))
        ]
        self.token = HfFolder.get_token()
        self.api = HfApi()

    def convert_arrow_to_parquet(self):
        for dataset_dir in self.dataset_directories:
            parquet_file_path = os.path.join(dataset_dir, 'data.parquet')
            if os.path.exists(parquet_file_path):
                print(f"Parquet file already exists in {dataset_dir}, skipping conversion...")
                continue

            # Handle multiple Arrow files
            arrow_files = [
                os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.arrow')
            ]
            if not arrow_files:
                print(f"No Arrow files found in {dataset_dir}, skipping...")
                continue

            # Load all Arrow files into a single Hugging Face Dataset object
            datasets = [Dataset.from_file(arrow_file) for arrow_file in arrow_files]
            dataset = concatenate_datasets(datasets)

            # Convert to Apache Arrow Table
            table = pa.concat_tables([ds.data.table for ds in datasets])

            # Save as Parquet file
            pq.write_table(table, parquet_file_path)

            print(f"Converted Arrow files in {dataset_dir} to {parquet_file_path}")

    def upload_parquet_to_hub(self):
        for dataset_dir in self.dataset_directories:
            parquet_file_path = os.path.join(dataset_dir, 'data.parquet')

            if not os.path.exists(parquet_file_path):
                print(f"Parquet file not found in {dataset_dir}, skipping...")
                continue

            # Create a unique repository ID for each dataset
            dataset_name = os.path.basename(dataset_dir)
            repo_id = f"{self.repo_id_base}_{dataset_name}"

            # Create the repository if it does not exist
            try:
                create_repo(repo_id, repo_type="dataset", token=self.token, exist_ok=True)
            except Exception as e:
                print(f"Error creating repository {repo_id}: {e}")
                continue

            # Upload the file
            self.api.upload_file(
                path_or_fileobj=parquet_file_path,
                path_in_repo=os.path.basename(parquet_file_path),
                repo_id=repo_id,
                repo_type="dataset",
                token=self.token
            )

            print(f"Uploaded {parquet_file_path} to Hugging Face Hub with repo ID {repo_id}")

class DataPreparation:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = None

    def load_data(self):
        """Load dataset from Hugging Face"""
        self.dataset = load_dataset(self.dataset_path)
        return self.dataset

    def preprocess(self, example):
        """Preprocess dataset into input-output pairs"""
        return {
            "input_text": example['inputs'],
            "target_text": example['targets'],
        }

    def tokenize_function(self, examples, tokenizer):
        """Tokenize input and output text"""
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            examples["target_text"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def prepare_dataset(self, tokenizer):
        """Prepare the dataset for training"""
        train_dataset = self.dataset["train"].map(self.preprocess)
        train_dataset = train_dataset.map(lambda x: self.tokenize_function(x, tokenizer), batched=True)
        eval_dataset = self.dataset["validation"].map(self.preprocess) if "validation" in self.dataset else None
        eval_dataset = eval_dataset.map(lambda x: self.tokenize_function(x, tokenizer), batched=True) if eval_dataset else None
        # Return only train_dataset instead of a tuple
        return train_dataset

    def dataset_info(self):
        """Print information about the dataset, such as the size"""
        if self.dataset:
            print(f"Dataset name: {self.dataset_path}")
            for split in self.dataset.keys():
                print(f"Split: {split}, Number of examples: {len(self.dataset[split])}")
        else:
            print("Dataset is not loaded. Please call load_data() first.")