# VectorDB-Domain-Adaptation

## Setup

First, install all the required libraries listed in requirements.txt:

```bash
git clone https://github.com/yangyajie0625/VectorDB-Domain-Adaptation.git vector_db
cd vector_db
conda create -n vector_db python=3.10
conda activate vector_db
pip install -r requirements.txt
```



## Creating or updating a vector database

Load the source dataset, use the CLIP model to extract features, and save them in the vector database.

```bash
python create_or_update_vectordb.py create \
--vector_db_path /path/to/save/vector_db \
--vector_db_name your_vector_db \
--clip_path /path/to/clip_model \
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
--clip_path /path/to/clip_model \
--dataset_path /path/to/dataset \
--distance_func cosine \
--batch_size 32 \
--device cuda
```

Valid options for distance_func are "l2", "ip", or "cosine", representing the three distance functions: Squared L2, Inner product, and Cosine similarity.

## Querying a vector database

This script allows you to query a vector database using different methods. The query is performed on a dataset of images, and the script retrieves similar images from the vector database. The embeddings of images are either loaded from a file or computed directly using a CLIP model.

### Running the Script

To run the script, you must choose either k or query_num based on the query method you are using. 

Example command:

```bash
python query_vectordb.py --method query_vector_db \
--vector_db_path /path/to/load/vector_db \
--vector_db_name your_vector_db_name \
--clip_path /path/to/clip_model \
--dataset_path /path/to/dataset \
--k 5 \
--query_num 100 \ 
--batch_size 32 \
--device cuda \
--result_file_name your_result_file_name \
--log_file_name your_log_file_name
```


### Parameters

| Parameter                      | Required/Optional                                 | Default Value         | Description                                                  |
| ------------------------------ | ------------------------------------------------- | --------------------- | ------------------------------------------------------------ |
| **--method**                   | Required                                          | None                  | The query method to use (e.g., `query_vector_db`, `query_by_sort`, etc.). For more details on available methods, see the [Image Embedding Methods](#image-embedding-methods) section. |
| **--vector_db_path**           | Required (for DB queries)                         | None                  | Path to the vector database.                                 |
| **--vector_db_name**           | Required (for DB queries)                         | None                  | The name of the vector database to load.                     |
| **--clip_path**                | Required (only if embeddings need to be computed) | None                  | Path to the CLIP model and processor. Used to compute embeddings for images or captions, unless embeddings are pre-loaded from a file. |
| **--dataset_path**             | Required (only if embeddings need to be computed) | None                  | Path to the image dataset used for querying. Required if computing image embeddings. Not needed if embeddings are loaded from a file. |
| **--load_embedding_from_file** | Optional                                          | None                  | If set, the script will load precomputed embeddings from the file specified by `--embedding_file_path`. |
| **--embedding_file_path**      | Required if `--load_embedding_from_file` is set   | None                  | Path to the `.npz` file containing precomputed image embeddings and their associated paths. |
| **--caption_path**             | Required for caption-based methods                | None                  | Path to a JSON file with captions for querying.              |
| **--query_num**                | Optional                                          | None                  | Total number of queries to perform across the dataset.       |
| **--cluster_num**              | Optional                                          | 3                     | Number of clusters when performing clustering queries.       |
| **--random_state**             | Optional                                          | 1                     | Random seed for clustering.                                  |
| **--batch_size**               | Optional                                          | 32                    | Batch size for image embedding computation.                  |
| **-k**                         | Optional                                          | 1                     | The number of most similar images to retrieve for each query. |
| **-m**                         | Optional                                          | 1                     | Parameter used in `query_by_sort` method to control the number of images sampled from the dataset. |
| **--device**                   | Optional                                          | None                  | Device for computation (`cpu` or `cuda`).                    |
| **--result_file_name**         | Optional                                          | Generated from method | The file name for saving query results in JSON format.       |
| **--log_file_name**            | Optional                                          | Generated from method | The file name for saving the time log.                       |



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

This method queries the vector database using image embeddings. The script processes all images in the dataset, converts them into embeddings, and retrieves a set of similar images from the vector database for each image.

#### **cluster_and_query_vector_db**

 This method clusters the image embeddings first, and then queries the vector database using the centroids of each cluster. It retrieves representative images from each cluster, ensuring diverse results.

#### **query_with_fixed_total_queries_no_duplication**

This method ensures that the total number of queries is fixed across the entire dataset, with no duplication of image results.

#### **query_with_fixed_total_queries_allow_duplication**

This method also ensures that a fixed total number of queries is performed, but it allows duplication of image results.

#### **query_vector_db_by_caption**

This method sorts the captions based on cosine similarity and retrieves images accordingly. The captions are ranked by similarity, and images are retrieved based on this ranking.

#### **query_each_image_separately**

This method queries the vector database for each image in the dataset individually. Unlike batch processing, it handles each image one by one and retrieves the most similar images for each separately.

#### **query_by_sort**

This method sorts the captions based on cosine similarity and retrieves images accordingly. The captions are ranked by similarity, and images are retrieved based on this ranking.
