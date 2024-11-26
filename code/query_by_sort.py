import json
import os
import random
from utils import batch_image_to_embedding, caption_to_embedding, get_top_k_images_for_caption


def query_by_sort(clip_model, clip_processor, caption_path, dataset_path, device, time_logger, m=1, k=1, batch_size=32):
    with open(caption_path, 'r', encoding='utf-8') as f:
        captions_data = json.load(f)
    captions = []
    for video_id, caption_list in captions_data.items():
        captions.extend(caption_list)
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    if m <= 1:
        image_paths = random.sample(image_paths, m * len(image_paths))
    else:
        image_paths = random.sample(image_paths, min(m, len(image_paths)))
    time_logger.start_period("Embedding")
    image_embeddings, _ = batch_image_to_embedding(image_paths, clip_model, clip_processor, device, batch_size)
    caption_embeddings = caption_to_embedding(captions, clip_model, clip_processor, device)
    time_logger.end_period("Embedding")
    results = []

    time_logger.start_period(f"Query")
    for caption, caption_embedding in zip(captions, caption_embeddings):
        top_k_images = get_top_k_images_for_caption(image_embeddings, image_paths, caption_embedding, k)
        results.append((caption, top_k_images))
    time_logger.end_period(f"Query")
    return results
