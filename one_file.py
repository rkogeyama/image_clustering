import fitz  # PyMuPDF
import os
import io
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Function to extract images from a PDF file
def extract_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    image_data = []

    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_format = base_image["ext"]
            image_size = base_image["xres"] * base_image["yres"]

            image_data.append((page_number, img_index, image_bytes, image_format, image_size))

    pdf_document.close()
    return image_data


def resize_images(image_data):
    print('inicio resizing')
    smallest_width = float("inf")
    smallest_height = float("inf")

    for _, _, image_bytes, _, _ in image_data:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            width, height = pil_image.size

            if width < smallest_width:
                smallest_width = width
            if height < smallest_height:
                smallest_height = height
        except Exception as e:
            print('resize error', e)
            continue

    resized_images = []

    for page_number, img_index, image_bytes, image_format, image_size in image_data:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            resized_image = pil_image.resize((smallest_width, smallest_height))
            resized_bytes = io.BytesIO()
            resized_image.save(resized_bytes, format=image_format)
            resized_images.append((page_number, img_index, resized_bytes.getvalue(), image_format, image_size))
            print(img_index)
        except Exception as e:
            print('resize error', e)
            continue

    return resized_images

# Function to cluster images
def cluster_images(image_data, num_clusters):
    print('inicio clusterizacao')
    image_embeddings = []

    for page_number, img_index, image_bytes, image_format, _ in image_data:
        try:
            # Convert image bytes to a PIL image
            # print(image_format)
            pil_image = Image.open(io.BytesIO(image_bytes))

            # Convert PIL image to NumPy array
            image_np = np.array(pil_image)

            # Resize the image (optional)
            # image_np = cv2.resize(image_np, (new_width, new_height))

            # Convert image to grayscale for simplicity (optional)
            # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

            # Flatten the image to a 1D array
            image_flat = image_np.flatten()
            image_embeddings.append(image_flat)
        except Exception as e:
            print(image_format, e)
            continue

    # Cluster images using K-Means
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_assignments = kmeans.fit_predict(image_embeddings)

    # Group images by cluster
    image_clusters = defaultdict(list)

    for i, cluster_id in enumerate(cluster_assignments):
        page_number, img_index, _, _, _ = image_data[i]
        image_clusters[cluster_id].append((page_number, img_index))

    return image_clusters


# Function to visualize clusters
def visualize_clusters(image_clusters, image_data):
    for cluster_id, images in image_clusters.items():
        print(f"Cluster {cluster_id}:")

        for page_number, img_index in images:
            print('total de imagens', len(image_data), 'index', page_number, len(images), img_index )
            # [x for x in image_data if (image_data[0] == page_number) & (image_data[1] == img_index)]

            image_bytes = [x for x in image_data if (x[0] == page_number) & (x[1] == img_index)][0][2]
            pil_image = Image.open(io.BytesIO(image_bytes))

            plt.imshow(pil_image)
            plt.axis('off')
            plt.show()


if __name__ == "__main__":
    pdf_path="/Users/renato/Coding/scholar/conferences/AOM_2018_Annual_Meeting_Program.pdf"
    num_clusters = 5  # Number of clusters

    image_data = extract_images_from_pdf(pdf_path)
    # df_img = pd.DataFrame(image_data, columns=["Page Number", "Image Index", "Image Data", "Image Format", "Image Size"])
    # df_img["Image Size"].hist()
    # plt.show()
    # image_data=[x for x in image_data if x[4]>6000]
    resized_images = resize_images(image_data)
    image_clusters = cluster_images(resized_images, num_clusters)

    visualize_clusters(image_clusters, image_data)
