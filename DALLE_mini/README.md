# MinDalle Image Generation

This project implements a complete MinDalle image generation pipeline using PyTorch, based on the [min-dalle repository](https://github.com/kuprel/min-dalle).

## Features

- Text-to-image generation using MinDalle model
- Multiple image generation with different seeds
- Automatic image saving and display
- Progress tracking during generation
- Error handling and validation
- GPU acceleration support

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have enough disk space for model downloads (models will be downloaded automatically on first use).

## Usage

### Basic Usage

```python
from dalle_gerenate_img import Text2imgProcess

# Initialize the processor
processor = Text2imgProcess()

# Generate images from text prompt
images = processor.process_prompt("A beautiful sunset over mountains", n_predictions=4)

# Images will be automatically displayed and saved
```

### Advanced Usage

```python
from dalle_gerenate_img import Text2imgProcess

# Initialize with custom model path
processor = Text2imgProcess(model_path="/path/to/your/models")

# Generate multiple images with custom parameters
images = processor.generate_image(
    prompt="A cute cat sitting in a garden",
    n_predictions=6,
    condition_scale=12.0  # Higher values = more creative, lower = more faithful
)

# Save images to custom directory
processor.save_images(images, "cat in garden", output_dir="my_images")

# Display images
processor.display_images(images, "cat in garden")
```

### Running the Demo

```bash
python dalle_gerenate_img.py
```

### Running Tests

```bash
python test_dalle_mini.py
```

## API Reference

### Text2imgProcess Class

#### `__init__(model_path="/Users/innerwq/Desktop/dalle_model")`
Initialize the MinDalle processor.

**Parameters:**
- `model_path`: Path to store the MinDalle models

#### `process_prompt(prompt, n_predictions=4)`
Process a text prompt and generate images.

**Parameters:**
- `prompt`: Text description for image generation
- `n_predictions`: Number of images to generate

**Returns:**
- List of PIL Image objects

#### `generate_image(prompt, n_predictions=4, condition_scale=10.0)`
Generate images from text prompt with custom parameters.

**Parameters:**
- `prompt`: Text description for image generation
- `n_predictions`: Number of images to generate
- `condition_scale`: Controls generation creativity (higher = more creative)

**Returns:**
- List of PIL Image objects

#### `save_images(images, prompt, output_dir="generated_images")`
Save generated images to disk.

**Parameters:**
- `images`: List of PIL Image objects
- `prompt`: Original text prompt (used for filename)
- `output_dir`: Directory to save images

#### `display_images(images, prompt)`
Display generated images using matplotlib.

**Parameters:**
- `images`: List of PIL Image objects
- `prompt`: Original text prompt (used for title)

## Configuration

### Model Parameters

- `condition_scale` (supercondition_factor): Controls how closely the generated image follows the text prompt
  - Lower values (1-5): More faithful to prompt
  - Higher values (10-20): More creative and diverse
  - Default: 10.0

- `temperature`: Controls randomness in generation
  - Lower values: More deterministic
  - Higher values: More random
  - Default: 1.0

- `top_k`: Number of top tokens to consider during generation
  - Lower values: More focused generation
  - Higher values: More diverse generation
  - Default: 256

### Hardware Requirements

- **CPU**: Works on CPU (slower but functional)
- **GPU**: CUDA-compatible GPU for acceleration (recommended)
- **Memory**: At least 8GB RAM recommended
- **Storage**: ~2GB for model downloads

## Advantages of MinDalle

Compared to the original DALLE-mini implementation:

1. **Simpler Setup**: No complex JAX/Flax dependencies
2. **Better Performance**: Optimized PyTorch implementation
3. **Easier Installation**: Standard PyTorch ecosystem
4. **Faster Generation**: More efficient model architecture
5. **Better GPU Support**: Native CUDA acceleration

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   pip install --upgrade min-dalle torch torchvision
   ```

2. **Out of Memory**
   - Reduce `n_predictions`
   - Close other applications
   - Use CPU if GPU memory is insufficient

3. **Slow Generation**
   - Generation takes time, especially on CPU
   - Consider using GPU for acceleration
   - Reduce model complexity with `is_mega=False`

4. **Model Download Issues**
   - Check internet connection
   - Clear cache and retry download

## Examples

### Simple Image Generation

```python
processor = Text2imgProcess()
images = processor.process_prompt("A red apple on a white table")
```

### Creative Generation

```python
processor = Text2imgProcess()
images = processor.generate_image(
    "A futuristic city with flying cars",
    n_predictions=6,
    condition_scale=15.0
)
```

### Batch Processing

```python
processor = Text2imgProcess()
prompts = [
    "A beautiful sunset",
    "A cute puppy",
    "A mountain landscape"
]

for prompt in prompts:
    images = processor.process_prompt(prompt, n_predictions=2)
```

## Model Information

- **Model**: MinDalle (simplified DALLE-mini implementation)
- **Framework**: PyTorch
- **Repository**: [min-dalle](https://github.com/kuprel/min-dalle)
- **License**: MIT

## License

This project is for educational purposes. Please respect the licenses of the underlying models.

## Contributing

Feel free to submit issues and enhancement requests!
