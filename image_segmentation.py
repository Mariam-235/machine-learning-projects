# -*- coding: utf-8 -*-

"""
K-Means Image Segmentation for Plant Dataset
Segments plant images by clustering pixels based on color similarity.
For each cluster, creates an image showing only that cluster's pixels (others are white).

Images to process: image_001.png, image_002.png, image_003.png,
                   image_004.JPG, image_005.png, image_006.png
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import os

def process_single_image(image_path, k):
    """
    Process a single image with k-means clustering for a specific k value.

    Parameters:
    -----------
    image_path : str
        Path to the input image
    k : int
        Number of clusters

    Returns:
    --------
    tuple: (cluster_images, original_image, cluster_info)
    """
    print(f"  Loading image: {image_path}")

    # Load and prepare image
    image = Image.open(image_path)
    image = np.array(image)

    # Handle alpha channel if present (RGBA → RGB)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    height, width = image.shape[:2]
    print(f"  Image size: {width}x{height} pixels")

    # Reshape for k-means: from (height, width, 3) to (height*width, 3)
    pixels = image.reshape(-1, 3)
    total_pixels = pixels.shape[0]
    print(f"  Total pixels to cluster: {total_pixels:,}")

    # Apply k-means clustering
    print(f"  Applying k-means with k={k} clusters...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(pixels)

    # Reshape labels back to image shape
    labels_image = labels.reshape(height, width)

    # Get cluster centers (dominant colors)
    cluster_centers = kmeans.cluster_centers_.astype(int)

    # Create separate image for each cluster
    cluster_images = []
    cluster_info = []

    for cluster_id in range(k):
        # Create mask for this cluster
        mask = (labels_image == cluster_id)

        # Count pixels in this cluster
        pixel_count = np.sum(mask)
        percentage = (pixel_count / total_pixels) * 100

        # Start with white image
        cluster_img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Copy only pixels belonging to this cluster
        cluster_img[mask] = image[mask]

        cluster_images.append(cluster_img)
        cluster_info.append({
            'id': cluster_id,
            'pixel_count': pixel_count,
            'percentage': percentage,
            'center_color': cluster_centers[cluster_id]
        })

        print(f"    Cluster {cluster_id + 1}: {pixel_count:,} pixels ({percentage:.1f}%) "
              f"- Dominant RGB: {tuple(cluster_centers[cluster_id])}")

    return cluster_images, image, cluster_info


def display_results(image_name, original, cluster_images, cluster_info, k):
    """
    Display original image and all cluster images in a single figure.

    Parameters:
    -----------
    image_name : str
        Name of the image
    original : numpy array
        Original image
    cluster_images : list
        List of cluster images
    cluster_info : list
        Information about each cluster
    k : int
        Number of clusters
    """
    # Create figure with k+1 subplots (original + k clusters)
    fig, axes = plt.subplots(1, k+1, figsize=(3.5*(k+1), 4))

    # Handle case where k=1 (axes won't be an array)
    if k == 1:
        axes = [axes]

    # Show original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontweight='bold', fontsize=11)
    axes[0].axis('off')

    # Show each cluster
    for i, (cluster_img, info) in enumerate(zip(cluster_images, cluster_info)):
        axes[i+1].imshow(cluster_img)

        # Create title with cluster information
        title = f"Cluster {info['id'] + 1}\n"
        title += f"{info['pixel_count']:,} pixels\n"
        title += f"({info['percentage']:.1f}%)"

        axes[i+1].set_title(title, fontsize=9)
        axes[i+1].axis('off')

        # Add colored border to indicate dominant color
        rgb = info['center_color']
        color_hex = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
        for spine in axes[i+1].spines.values():
            spine.set_edgecolor(color_hex)
            spine.set_linewidth(3)

    # Main title
    fig.suptitle(f'{image_name} - K-Means Segmentation (k={k})',
                fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.show()


def segment_image_all_k_values(image_path, k_values=[2, 3, 4, 5]):
    """
    Segment a single image for all k values and display results.

    Parameters:
    -----------
    image_path : str
        Path to the input image
    k_values : list
        List of k values to test
    """
    image_name = os.path.basename(image_path)

    print(f"\n{'='*70}")
    print(f"Processing: {image_name}")
    print(f"{'='*70}")

    for k in k_values:
        print(f"\n--- K = {k} clusters ---")

        try:
            # Get cluster images
            cluster_images, original, cluster_info = process_single_image(image_path, k)

            # Display results
            display_results(image_name, original, cluster_images, cluster_info, k)

            print(f"  ✓ Successfully created {k} cluster images")

        except Exception as e:
            print(f"  ✗ Error processing with k={k}: {str(e)}")

    print(f"\n{'='*70}")
    print(f"Finished processing {image_name}")
    print(f"{'='*70}\n")


def process_all_plant_images(k_values=[2, 3, 4, 5]):
    """
    Process all 6 plant images with different k values.

    Parameters:
    -----------
    k_values : list
        List of k values to test
    """
    # Define the 6 plant images
    image_files = [
        'image_001.png',
        'image_002.png',
        'image_003.png',
        'image_004.JPG',
        'image_005.png',
        'image_006.png'
    ]

    print("\n" + "="*70)
    print("K-MEANS PLANT IMAGE SEGMENTATION")
    print("="*70)
    print(f"Images to process: {len(image_files)}")
    print(f"K values: {k_values}")
    print(f"Total outputs per image: {sum(k_values)} cluster images")
    print("="*70)

    # Check which images exist
    existing_images = []
    missing_images = []

    for img_file in image_files:
        if os.path.exists(img_file):
            existing_images.append(img_file)
        else:
            missing_images.append(img_file)

    if missing_images:
        print("\n⚠️  WARNING: Some images not found:")
        for img in missing_images:
            print(f"   - {img}")

    if not existing_images:
        print("\n❌ ERROR: No images found in current directory!")
        print("\nPlease ensure these files are in the same folder as this script:")
        for img in image_files:
            print(f"   - {img}")
        return

    print(f"\n✓ Found {len(existing_images)} image(s) to process")

    # Process each existing image
    for i, image_path in enumerate(existing_images, 1):
        print(f"\n{'#'*70}")
        print(f"# Image {i}/{len(existing_images)}")
        print(f"{'#'*70}")

        segment_image_all_k_values(image_path, k_values)

    # Final summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"Processed: {len(existing_images)} image(s)")
    print(f"K values tested: {k_values}")
    print(f"Total cluster images created: {len(existing_images) * sum(k_values)}")
    if missing_images:
        print(f"Skipped (not found): {len(missing_images)} image(s)")
    print("="*70)


def process_custom_images(image_paths, k_values=[2, 3, 4, 5]):
    """
    Process a custom list of image paths.

    Parameters:
    -----------
    image_paths : list
        List of image file paths
    k_values : list
        List of k values to test
    """
    existing_images = [path for path in image_paths if os.path.exists(path)]

    if not existing_images:
        print("❌ No images found!")
        return

    print(f"Processing {len(existing_images)} image(s)...")

    for image_path in existing_images:
        segment_image_all_k_values(image_path, k_values)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║       K-MEANS CLUSTERING FOR PLANT IMAGE SEGMENTATION           ║
    ║                                                                  ║
    ║  This script will process the 6 plant images and create         ║
    ║  separate visualizations for each cluster at k=2, 3, 4, 5       ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Process all 6 plant images automatically
    process_all_plant_images(k_values=[2, 3, 4, 5])

    print("\n" + "="*70)
    print("All done! Close the matplotlib windows to exit.")
    print("="*70)