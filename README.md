# RLDCMarioRL

## Installation
This project requires certain Python packages to be installed to function correctly. These dependencies are listed in the requirements.txt file. The file includes libraries essential for data processing, machine learning, network operations, and more, ensuring a consistent setup across different environments.

## Dependencies Include:
- Data Processing and Visualization: numpy, matplotlib, pandas
- Machine Learning and Simulation: gym, gym-super-mario-bros, cloudpickle
- Network Operations: requests, urllib3
- Image and Video Processing: imageio, imageio-ffmpeg, opencv-python
- Others: Utility libraries like tqdm for progress bars, decorator, and colorama for enhancing terminal outputs.
- Optional Dependencies:
  Some dependencies, like torch, torchaudio, and torchvision, are commented out in the requirements.txt file. These are related to PyTorch and may require installation with specific versions to support CUDA for GPU-accelerated computing. Adjust these according to your system's CUDA version if necessary.

## Installing Dependencies
To set up your environment to run the project, we recommend creating a virtual environment and installing the dependencies as follows:

pip install -r requirements.txt

Ensure you're in the project's root directory when you run the pip install command, or provide the full path to the requirements.txt file.

Note: For CUDA-compatible packages (torch, torchaudio, torchvision), you may need to install them separately to match your system's CUDA setup. Refer to the official PyTorch installation guide for detailed instructions.

Feel free to adjust the content to better fit your project's specifics
