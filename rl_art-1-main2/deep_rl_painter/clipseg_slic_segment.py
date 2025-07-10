from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import os 

def segment_image_with_clipseg(image_path: str, text_prompts: list[str]):
    """
    Performs image segmentation using the CLIPSeg model based on text prompts.

    Args:
        image_path (str): The path to the input image.
        text_prompts (list[str]): A list of text descriptions for the objects to segment.

    Returns:
        dict: A dictionary where keys are the text prompts and values are PIL Image objects
              representing the segmentation masks (white for segmented, black for background).
              Returns None if the image cannot be loaded.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    model_name = "CIDAS/clipseg-rd64-refined"
    processor = CLIPSegProcessor.from_pretrained(model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name)

    inputs = processor(text=text_prompts, images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits (raw prediction scores)
    logits = outputs.logits

    # Normalize logits to get probabilities and create masks
    predicted_masks = {}
    for i, prompt in enumerate(text_prompts):
        mask_tensor = torch.sigmoid(logits[i])
        mask_image = Image.fromarray((mask_tensor.squeeze().cpu().numpy() * 255).astype('uint8'), 'L')
        predicted_masks[prompt] = mask_image

    return predicted_masks

if __name__ == '__main__':
    output_folder = "segmented_masks"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Ensured output folder '{output_folder}' exists.")

    # Example Usage:
    # 1. Create a dummy image for testing
    try:
        dummy_image = Image.new('RGB', (200, 200), color = 'red')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(dummy_image)
        draw.ellipse((50, 50, 150, 150), fill='white', outline='white')
        dummy_image_path = os.path.join("target_images", "test_dummy_image.png")
        os.makedirs(os.path.dirname(dummy_image_path), exist_ok=True)
        dummy_image.save(dummy_image_path)
        print(f"Created dummy image for demonstration at {dummy_image_path}.")
    except Exception as e:
        print(f"Could not create dummy image: {e}. Please ensure you have Pillow installed and necessary permissions.")

    image_path = "target_images/target_image_1.jpg"
    prompts = ["braids"]

    print(f"\nAttempting to segment image: {image_path}")
    print(f"With prompts: {prompts}")

    segmentation_results = segment_image_with_clipseg(image_path, prompts)

    if segmentation_results:
        print("\nSegmentation successful! Saving masks...")
        for prompt, mask_image in segmentation_results.items():
            safe_prompt_name = prompt.replace(' ', '_').replace('/', '_').replace('\\', '_')
            mask_filename = f"mask_{safe_prompt_name}.png"
            full_mask_path = os.path.join(output_folder, mask_filename)

            try:
                mask_image.save(full_mask_path)
                print(f"  Saved mask to {full_mask_path}")
            except Exception as e:
                print(f"  Error saving mask {full_mask_path}: {e}")
    else:
        print("Segmentation failed. Please check the image path and error messages above.")