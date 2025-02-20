import json
import torch
from chromadb import PersistentClient
from utils import batch_image_to_embedding, caption_to_embedding
import sqlite3

class VectorDB:
    def __init__(self, device=None):
        self.collection = None
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_vector_db(self, vector_db_load_path, vector_db_name):
        client = PersistentClient(path=vector_db_load_path)
        self.collection = client.get_collection(name=vector_db_name)

    def create_vector_db(self, clip_model, clip_processor, image_paths, vector_db_save_path, vector_db_name,
                         distance_func, batch_size):

        image_embeddings, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor,
                                                                       self.device, batch_size)
        metadata = [{"path": img_path} for img_path in valid_image_paths]
        self._save_embeddings_to_db(vector_db_save_path, vector_db_name, image_embeddings, metadata, distance_func)

    def update_vector_db(self, clip_model, clip_processor, image_paths, batch_size):

        image_embeddings, valid_image_paths = batch_image_to_embedding(image_paths, clip_model, clip_processor,
                                                                       self.device, batch_size)
        metadata = [{"path": img_path} for img_path in valid_image_paths]
        existing_ids = self.collection.get()['ids']
        if existing_ids:
            max_existing_id = max(map(int, existing_ids))
        else:
            max_existing_id = -1
        max_batch_size = 41660
        for i in range(0, len(image_embeddings), max_batch_size):
            batch_embeddings = image_embeddings[i:i + max_batch_size]
            batch_metadata = metadata[i:i + max_batch_size]
            batch_ids = [str(j) for j in
                         range(max_existing_id + 1 + i, max_existing_id + 1 + i + len(batch_embeddings))]

            self.collection.add(
                embeddings=batch_embeddings,
                metadatas=batch_metadata,
                ids=batch_ids
            )

    def query_vector_db(self, query_embedding, valid_image_paths, k=1):

        results = self.collection.query(query_embeddings=query_embedding, n_results=k)
        return results, valid_image_paths

    def query_vectors_adaptive(self, query_embedding, valid_image_paths, query_num, density_threshold, low_density_limit, batch_size):
        collected = set()
        queries_per_image = max(1, query_num // len(query_embedding))
        queries_per_image_old = 0
        # print(f"| Initial queries_per_image: {queries_per_image}")
        low_density_streak = 0
        exceed = False
        should_break = False

        while len(collected) < query_num and low_density_streak < low_density_limit:
            start_count = len(collected)
            try:
                for i in range(0, len(query_embedding), batch_size):
                    batch_embeddings = query_embedding[i:i + batch_size]

                    result = self.collection.query(
                        query_embeddings=batch_embeddings,
                        n_results=queries_per_image
                    )
                    # print(f"result:{result}")
                    for metadata_list in result.get('metadatas',):
                        for meta in metadata_list:
                            if meta and 'path' in meta:
                                collected.add(meta['path'])
                                if len(collected) >= query_num:
                                    should_break = True
                                    break
                        if should_break:
                            break
                    if should_break:
                        break

                if not should_break:
                    current_density = (len(collected) - start_count) / (
                                (queries_per_image - queries_per_image_old) * len(query_embedding))
                    # print(f"| Collected this round: {len(collected)} - {start_count} = {len(collected) - start_count}")
                    # print(f"| Current density: {current_density:.2f}")

                    queries_per_image_old = queries_per_image
                    if current_density < density_threshold:
                        low_density_streak += 1
                        queries_per_image = queries_per_image << 2
                    else:
                        low_density_streak = 0
                        queries_per_image = queries_per_image << 1
                    # print(f"| current queries_per_image: {queries_per_image}")
            except sqlite3.OperationalError as e:
                if "too many SQL variables" in str(e):
                    exceed = True
                    max_sql_vars = queries_per_image_old * batch_size
            if exceed:
                batch_size = max(max_sql_vars // queries_per_image, 1)

        # print(f"| len(collected):{len(collected)}")
        result_paths_list = list(collected)
        return result_paths_list, valid_image_paths
    def query_with_fixed_total_queries_no_duplication(self, query_embedding, valid_image_paths, query_num,
                                                      density_threshold, batch_size):
        collected = set()
        queries_per_image = max(1, query_num // len(query_embedding))
        queries_per_image_old = 0
        # print(f"| Initial queries_per_image: {queries_per_image}")
        exceed = False
        should_break = False
        while len(collected) < query_num:
            start_count = len(collected)
            try:
                for i in range(0, len(query_embedding), batch_size):
                    batch_embeddings = query_embedding[i:i + batch_size]

                    result = self.collection.query(
                        query_embeddings=batch_embeddings,
                        n_results=queries_per_image
                    )
                    # print(f"result:{result}")
                    for metadata_list in result.get('metadatas', ):
                        for meta in metadata_list:
                            if meta and 'path' in meta:
                                collected.add(meta['path'])
                                if len(collected) >= query_num:
                                    should_break = True
                                    break
                        if should_break:
                            break
                    if should_break:
                        break

                if not should_break:
                    current_density = (len(collected) - start_count) / (
                            (queries_per_image - queries_per_image_old) * len(query_embedding))
                    # print(f"| Collected this round: {len(collected)} - {start_count} = {len(collected) - start_count}")
                    # print(f"| Current density: {current_density:.2f}")

                    queries_per_image_old = queries_per_image
                    if current_density < density_threshold:
                        queries_per_image = queries_per_image << 2
                    else:
                        queries_per_image = queries_per_image << 1
                    # print(f"| current queries_per_image: {queries_per_image}")
            except sqlite3.OperationalError as e:
                if "too many SQL variables" in str(e):
                    exceed = True
                    max_sql_vars = queries_per_image_old * batch_size
            if exceed:
                batch_size = max(max_sql_vars // queries_per_image, 1)

        # print(f"| len(collected):{len(collected)}")
        result_paths_list = list(collected)
        return result_paths_list, valid_image_paths
    def query_with_fixed_total_queries_allow_duplication(self, query_embedding, valid_image_paths, query_num):
        total_images = len(query_embedding)
        queries_per_image = max(1, query_num // total_images)
        more_query_count = query_num % total_images
        result_paths_list = []
        if(more_query_count!=0):
            try:
                result = self.collection.query(
                    query_embeddings=query_embedding[:more_query_count],
                    n_results=queries_per_image+1
                )
                for res in result['metadatas']:
                    for metadata in res:
                        result_paths_list.append(metadata['path'])
            except sqlite3.OperationalError as e:
                if "too many SQL variables" in str(e):
                    for embedding in query_embedding[:more_query_count]:
                        result = self.collection.query(
                            query_embeddings=[embedding],
                            n_results=queries_per_image+1
                        )
                        for res in result['metadatas']:
                            for metadata in res:
                                result_paths_list.append(metadata['path'])
        try:
            result = self.collection.query(
                query_embeddings=query_embedding[more_query_count:],
                n_results=queries_per_image
            )
            for res in result['metadatas']:
                for metadata in res:
                    result_paths_list.append(metadata['path'])
        except sqlite3.OperationalError as e:
            if "too many SQL variables" in str(e):
                for embedding in query_embedding[more_query_count:]:
                    result = self.collection.query(
                        query_embeddings=[embedding],
                        n_results=queries_per_image
                    )
                    for res in result['metadatas']:
                        for metadata in res:
                            result_paths_list.append(metadata['path'])
        return result_paths_list, valid_image_paths

    def query_vector_db_by_caption(self, clip_model, clip_processor, caption_path, k=1):
        with open(caption_path, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
        captions = []
        result_list = []
        for video_id, caption_list in captions_data.items():
            captions.extend(caption_list)
            query_embeddings = caption_to_embedding(captions, clip_model, clip_processor, self.device)
            results = self.collection.query(query_embeddings=query_embeddings, n_results=k)
            result_list.append(results)
        return result_list

    def query_each_image_separately(self, query_embedding, valid_image_paths, k=1):
        result_list = []
        for i, img_embedding in enumerate(query_embedding):
            results = self.collection.query(
                query_embeddings=query_embedding[i],
                n_results=k
            )
            result_list.append(results)
        return result_list


    def _save_embeddings_to_db(self, vector_db_save_path, vector_db_name, image_embeddings, metadata, distance_func):
        client = PersistentClient(path=vector_db_save_path)
        collection = client.create_collection(name=vector_db_name, metadata={"hnsw:space": distance_func})
        max_batch_size = 41660

        for i in range(0, len(image_embeddings), max_batch_size):
            batch_embeddings = image_embeddings[i:i + max_batch_size]
            batch_metadata = metadata[i:i + max_batch_size]
            batch_ids = [str(j) for j in range(i, i + len(batch_embeddings))]

            collection.add(embeddings=batch_embeddings, metadatas=batch_metadata, ids=batch_ids)
