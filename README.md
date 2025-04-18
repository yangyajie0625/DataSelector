# DataSelector++

## Setup

### 1. Install Dependencies

First, install all the required libraries listed in requirements.txt:

```bash
git clone https://github.com/yangyajie0625/VectorDB-Domain-Adaptation.git vector_db
cd vector_db
conda create -n vector_db python=3.10
conda activate vector_db
pip install -r requirements.txt
```

### 2. Download the CLIP Model

This project relies on the CLIP model to extract image embeddings. You can download a pre-trained CLIP model from **Hugging Face**:

**Recommended Model:** [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)

You can download the model manually from the above link.
After downloading the model, you can set `--clip_path` to the local path containing the model files.

Example:
```bash
--clip_path /path/to/clip-vit-large-patch14-336
```

For more details, visit [Hugging Face CLIP Model Hub](https://huggingface.co/openai/clip-vit-large-patch14-336).


## Creating or updating a vector database

Load the source dataset, extract image features using the CLIP model, and store them in the vector database. The source dataset represents the image collection that will serve as the retrieval pool during querying.

```bash
python create_or_update_vectordb.py create \
--vector_db_path /path/to/save/vector_db \
--vector_db_name your_vector_db \
--clip_path /path/to/clip_model_directory \
--dataset_path /path/to/dataset \
--distance_func cosine \
--batch_size 32 \
--device cuda
```

Add a new source dataset to the existing vector database.
```bash
python create_or_update_vectordb.py update \
--vector_db_path /path/to/load/vector_db \
--vector_db_name your_vector_db \
--clip_path /path/to/clip_model_directory \
--dataset_path /path/to/dataset \
--distance_func cosine \
--batch_size 32 \
--device cuda
```

Valid options for distance_func are "l2", "ip", or "cosine", representing the three distance functions: Squared L2, Inner product, and Cosine similarity.

## Querying a vector database

Query a vector database using different methods. The query is performed using images from a target dataset. The script retrieves similar images from the source dataset stored in the vector database. The embeddings of the target dataset images can either be loaded from a file or computed directly using a CLIP model.

### Running the Script

To run the script, you must choose either k or query_num based on the query method you are using. 

Example command:

```bash
python query_vectordb.py --method query_vector_db \
--vector_db_path /path/to/load/vector_db \
--vector_db_name your_vector_db_name \
--clip_path /path/to/clip_model_directory \
--dataset_path /path/to/dataset \
--k 5 \
--query_num 100 \ 
--batch_size 32 \
--device cuda \
--result_file_name your_result_file_name \
--log_file_name your_log_file_name
```


### Parameters

| Parameter                      | Required/Optional                                                                           | Default Value         | Description                                                                                                                                                   |
|--------------------------------|---------------------------------------------------------------------------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **--method**                   | Required                                                                                    | None                  | The query method to use (e.g., query_vector_db, query_by_sort, etc.). For more details on available methods, see the [Query Methods](#query-methods) section. |
| **--vector_db_path**           | Required (for DB queries)                                                                   | None                  | Path to the vector database.                                                                                                                                  |
| **--vector_db_name**           | Required (for DB queries)                                                                   | None                  | The name of the vector database to load.                                                                                                                      |
| **--clip_path**                | Required (only if embeddings need to be computed)                                           | None                  | Path to the CLIP model and processor. Used to compute embeddings for images or captions, unless embeddings are pre-loaded from a file.                        |
| **--dataset_path**             | Required (only if embeddings need to be computed)                                           | None                  | Path to the image dataset used for querying. Required if computing image embeddings. Not needed if embeddings are loaded from a file.                         |
| **--load_embedding_from_file** | Optional                                                                                    | None                  | If set, the script will load precomputed embeddings from the file specified by `--embedding_file_path`.                                                       |
| **--embedding_file_path**      | Required if `--load_embedding_from_file` is set                                             | None                  | Path to the `.npz` file containing precomputed image embeddings and their associated paths.                                                                   |
| **--caption_path**             | Required (for caption-based methods)                                                        | None                  | Path to a JSON file with captions for querying.                                                                                                               |
| **--query_num**                | Optional                                                                                    | None                  | The number of image results to collect.                                                                                                                       |
| **--batch_size**               | Optional                                                                                    | 32                    | Batch size for querying the vector database.                                                                                                                  |
| **-k**                         | Optional                                                                                    | 1                     | The number of images to retrieve for each query.                                                                                                              |
| **-m**                         | Optional (for `query_by_sort`)                                                              | 1                     | Parameter to control the number of images sampled from the dataset.                                                                                           |
| **--density_threshold**        | Optional (for `query_vectors_adaptive` and `query_with_fixed_total_queries_no_duplication`) | 0.2                   | Minimum valid result ratio; controls how to increase query attempts if few new images are found.                                                              |
| **--low_density_limit**        | Optional (for `query_vectors_adaptive`)                                                     | 3                     | Maximum consecutive low-density rounds before stopping; prevents excessive low-efficiency queries.                                                            |
| **--device**                   | Optional                                                                                    | None                  | Device for computation (`cpu` or `cuda`).                                                                                                                     |
| **--result_file_name**         | Optional                                                                                    | Generated from method | The file name for saving query results in JSON format.                                                                                                        |
| **--log_file_name**            | Optional                                                                                    | Generated from method | The file name for saving the time log.                                                                                                                        |



### Image Embedding Methods

There are two ways to obtain image embeddings:

1. **Load from a file**: If you already have pre-computed image embeddings saved in a file, you can load them by specifying the `--load_embedding_from_file` flag and the `--embedding_file_path`. The script will load the embeddings and image paths from the file. Example:

   ```bash
   --load_embedding_from_file --embedding_file_path /path/to/embedding/file.npz
   ```

   The embeddings will be loaded from the `.npz` file, and the script will use these embeddings for querying.

2. **Compute directly**: If the embeddings are not provided, the script will use the specified CLIP model to compute the embeddings from the images in the dataset. This requires the `--clip_path` parameter to point to a pre-trained CLIP model.


In this script, the **--method** parameter determines the specific way the vector database is queried. Below are the possible values for method and a description of each query type:

### Query Methods

The **--method** parameter determines the specific query strategy used. Below are the available methods:

#### **query_vector_db**

This method queries the vector database using image embeddings in a batch process. It processes all images in the dataset at once, converts them into embeddings, and retrieves a set of approximately most similar images from the vector database.

#### **query_each_image_separately**

This method queries the vector database individually for each image in the dataset. Unlike the batch approach, it processes images one by one, retrieving a set of approximately most similar images.

#### **query_with_fixed_total_queries_no_duplication**

This method ensures that a fixed number of unique image results is collected. It adjusts the queries per image based on retrieval efficiency, continuing until the required number of unique results is obtained.

#### **query_with_fixed_total_queries_allow_duplication**

This method collects a fixed number of image results, allowing duplicates. It is more efficient but may result in repeated images.

#### **query_vectors_adaptive**

This method adaptively adjusts the number of queries per image based on the uniqueness of retrieved results. It aims to maximize the diversity of images while reducing redundant queries. When `query_num` is set to the size of the source dataset, the number of unique results can reflect the upper limit of source samples that match the target dataset, avoiding inefficient queries.

#### **query_vector_db_by_caption**

This method sorts the captions based on cosine similarity and retrieves images accordingly. The captions are ranked by similarity, and images are retrieved based on this ranking.

#### **query_by_sort**

This method sorts the captions based on cosine similarity and retrieves images accordingly. The captions are ranked by similarity, and images are retrieved based on this ranking.
