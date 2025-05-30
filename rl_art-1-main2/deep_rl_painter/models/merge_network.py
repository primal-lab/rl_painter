import os
import torch
import torch.nn as nn
import warnings
from typing import List
from torchsummary import summary
from .image_encoder import get_image_encoder

# Turn off all warnings
warnings.filterwarnings("ignore")


class MergedNetwork(nn.Module):
    """
    Processes two input images with separate encoders, concatenates the features,
    and passes them through a merged network.  The architecture of the merged
    network is now customizable.
    """

    def __init__(self,
                 image_encoder_model,
                 image_encoder_model_2,
                 pretrained,
                 fine_tune_encoder,
                 fine_tune_encoder_2,
                 actor_network_input,
                 hidden_layers,
                 merged_output_size,
                 use_custom_encoder,
                 use_custom_encoder_2,
                 custom_encoder,
                 custom_encoder_2,
                 activation_function,
                 in_channels) -> None:
        """
        Args:
            image_encoder_model (str): Name of the CNN architecture to use for both image encoders
                ('resnet50', 'efficientnet_b0', etc.).  Ignored if use_custom_encoder is True.
            image_encoder_model_2 (str): Name of the CNN architecture to use for the second image encoder. For the target image.
            pretrained (bool): Whether to use pre-trained weights for the CNN encoders.
                Ignored if use_custom_encoder is True.
            fine_tune_encoder_1 (bool): Whether to fine-tune the CNN weights, for canvas. Ignored if use_custom_encoder is True.
            fine_tune_encoder_2 (bool): Whether to fine-tune the CNN weights, for target image.
            actor_network_input (int): Number of input features to merge with the merged network.
            hidden_layers (List[int]):  A list of integers defining the hidden layer sizes in the merged network.
                For example, [512, 256] will create a merged network with two hidden layers of size 512 and 256.
            merged_output_size (int): Size of the final output of the merged network.
            use_custom_encoder (bool): Use a custom-defined CNN encoder.
            custom_encoder (nn.Module): A custom PyTorch model to use as the image encoder.
                If provided, overrides image_encoder_model, pretrained, and fine_tune.
            activation_function (str): The activation function to use in the merged network.
                Can be 'ReLU', 'Tanh', or 'LeakyReLU'.
            in_channels (int): Number of input channels for the image encoders (e.g., 3 for RGB, 1 for grayscale).
        """
        super().__init__()

        self.image_encoder_model = image_encoder_model
        self.pretrained = pretrained

        if fine_tune_encoder_2 is None:
            fine_tune_encoder_2 = fine_tune_encoder

        self.fine_tune_encoder_1 = fine_tune_encoder
        self.fine_tune_encoder_2 = fine_tune_encoder_2
        self.actor_network_input = actor_network_input
        self.hidden_layers = hidden_layers
        self.merged_output_size = merged_output_size
        self.use_custom_encoder = use_custom_encoder
        self.use_custom_encoder_2 = use_custom_encoder_2
        self.in_channels = in_channels

        self.custom_encoder = custom_encoder
        if self.use_custom_encoder_2 and not custom_encoder_2:
            self.custom_encoder_2 = custom_encoder
        else:
            self.custom_encoder_2 = custom_encoder_2

        self.image_encoder_1 = None
        self.image_encoder_2 = None
        self.activation_function = activation_function
        self.encoder_output_size = None
        self.merged_network = None
        self._initialize_network()
        #self.custom_encoder_2 = custom_encoder_2
        self.image_encoder_model_2 = image_encoder_model_2
        #!self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def _initialize_network(self) -> None:
        """
        Initializes the dual image processor network.
        """

        if self.use_custom_encoder:
            if self.custom_encoder:
                self.image_encoder_1 = get_image_encoder(
                    fine_tune=self.fine_tune_encoder_1,
                    custom_model=self.custom_encoder,
                    in_channels=self.in_channels
                )
        else:
            # Create two image encoders with the same architecture
            self.image_encoder_1 = get_image_encoder(
                model_name=self.image_encoder_model,
                pretrained=self.pretrained,
                fine_tune=self.fine_tune_encoder_1,
                in_channels=self.in_channels
            )

        if self.use_custom_encoder_2:
            self.image_encoder_2 = get_image_encoder(
                fine_tune=self.fine_tune_encoder_2,
                custom_model=self.custom_encoder_2,
                in_channels=self.in_channels
            )
        else:
            self.image_encoder_2 = get_image_encoder(
                model_name=self.image_encoder_model,
                pretrained=self.pretrained,
                fine_tune=self.fine_tune_encoder_2,
                in_channels=self.in_channels
            )

        # Determine the output size of the individual encoders
        self.encoder_output_size1 = self._get_encoder_output_size(
            self.image_encoder_1)
        self.encoder_output_size2 = self._get_encoder_output_size(
            self.image_encoder_2)

        # Merged network
        self._build_merged_network()

    def _get_encoder_output_size(self, encoder: nn.Module) -> int:
        """
        Helper function to determine the output size of a given image encoder.

        Args:
            encoder (nn.Module): The image encoder module.

        Returns:
            int: The output size of the encoder.
        """
        # Example input for size check
        device = next(encoder.parameters()).device
        dummy_input = torch.randn(1, self.in_channels, 224, 224, device=device)
        #!dummy_input = torch.randn(1, self.in_channels, 224, 224)  # Assume 224x224 input
        if next(encoder.parameters()).is_cuda:
            dummy_input = dummy_input.to(next(encoder.parameters()).device)

        with torch.no_grad():
            dummy_output = encoder(dummy_input)

        if isinstance(dummy_output, tuple):
            dummy_output = dummy_output[0]  # Handle cases like InceptionV3

        return dummy_output.shape[1]

    def _build_merged_network(self) -> nn.Sequential:
        """
        Builds the merged network based on the provided hidden layer sizes
        and activation function.

        Args:
            hidden_layers (List[int]): List of hidden layer sizes.
            merged_output_size (int): Size of the final output.
            activation_function (str):  The activation function to use.

        Returns:
            nn.Sequential: The merged network.
        """
        layers: List[nn.Module] = []
        input_size = self.encoder_output_size1 + \
            self.encoder_output_size2  # Concatenated output size

        if self.actor_network_input > 0:
            input_size += self.actor_network_input

        # Select activation function
        if self.activation_function == 'ReLU':
            activation = nn.ReLU()
        elif self.activation_function == 'Tanh':
            activation = nn.Tanh()
        elif self.activation_function == 'LeakyReLU':
            activation = nn.LeakyReLU()
        else:
            raise ValueError(
                f"Invalid activation function: {self.activation_function}")

        # Add hidden layers
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation)
            input_size = hidden_size  # Update input size for the next layer

        # Add the final output layer
        layers.append(nn.Linear(input_size, self.merged_output_size, bias=False))

        self.merged_network = nn.Sequential(*layers)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, action_params: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the dual image processor.

        Args:
            image1 (torch.Tensor): The first input image tensor.
            image2 (torch.Tensor): The second input image tensor.

        Returns:
            torch.Tensor: The output of the merged network.
        """
        # Pass each image through its encoder
        features_1 = self.image_encoder_1(image1)
        features_2 = self.image_encoder_2(image2)

        if isinstance(features_1, tuple):
            features_1 = features_1[0]
        if isinstance(features_2, tuple):
            features_2 = features_2[0]

        # Concatenate the features
        """if action_params is not None:
            merged_features = torch.cat(
                (features_1, features_2, action_params), dim=1)
        else:
            merged_features = torch.cat((features_1, features_2), dim=1)"""

        # Concatenate the features
        merged_features = torch.cat((features_1, features_2), dim=1)
        if self.actor_network_input > 0:
            if action_params is None:
                raise ValueError("action_params must be provided if action_params_size > 0")
            merged_features = torch.cat((merged_features, action_params), dim=1)

        # Log merged feature vector shape before FC layers
        #with open("logs/model/merged_network.log", "a") as f:
        #    f.write(f"Merged features shape: {merged_features.shape}\n")

        # Pass through the merged network
        output = self.merged_network(merged_features)

        # Log output of merged network
        #with open("logs/model/merged_network.log", "a") as f:
         #   f.write(f"Output shape: {output.shape}\n")

        return output


def create_merged_network(image_encoder_model: str = 'resnet50',
                                image_encoder_model_2: str = 'resnet50',
                                pretrained: bool = True,
                                fine_tune_encoder: bool = True,
                                fine_tune_encoder_2: bool = True,
                                actor_network_input: int = 0,
                                hidden_layers: List[int] = [
                                    512, 256, 128, 64, 32],
                                merged_output_size: int = 6,
                                use_custom_encoder: bool = False,
                                use_custom_encoder_2: bool = False,
                                custom_encoder: nn.Module = None,
                                custom_encoder_2: nn.Module = None,
                                activation_function: str = 'LeakyReLU',
                                in_channels: int = 3) -> MergedNetwork:
    """
    Creates a MergedNetwork instance.

    Args:
        image_encoder_model (str):  Name of the CNN architecture to use.
        image_encoder_model_2 (str): Name of the CNN architecture to use for the second image encoder.
        pretrained (bool): Whether to use pre-trained weights.
        fine_tune_encoder (bool): Whether to fine-tune the weights.
        fine_tune_encoder_2 (bool): Whether to fine-tune the weights for the second encoder.
        actor_network_input (int): Number of input features to merge with the merged network.
        hidden_layers (List[int]):  A list of integers defining the hidden layer sizes in the merged network.
        merged_output_size (int): Size of the final output.
        use_custom_encoder (bool): Use a custom-defined CNN encoder
        use_custom_encoder_2 (bool): Use a custom-defined CNN encoder for the second image.
        custom_encoder (nn.Module): A custom PyTorch model to use as the image encoder.
        custom_encoder_2 (nn.Module): A custom PyTorch model to use as the second image encoder.
        activation_function (str): The activation function to use ('ReLU', 'Tanh', or 'LeakyReLU').
        in_channels (int): Number of input channels for the image encoders (e.g., 3 for RGB, 1 for grayscale).

    Returns:
        MergedNetwork: An instance of the MergedNetwork class.
    """

    return MergedNetwork(
        image_encoder_model=image_encoder_model,
        image_encoder_model_2=image_encoder_model_2,
        pretrained=pretrained,
        fine_tune_encoder=fine_tune_encoder,
        fine_tune_encoder_2=fine_tune_encoder_2,
        actor_network_input=actor_network_input,
        hidden_layers=hidden_layers,
        merged_output_size=merged_output_size,
        use_custom_encoder=use_custom_encoder,
        use_custom_encoder_2=use_custom_encoder_2,
        custom_encoder=custom_encoder,
        custom_encoder_2=custom_encoder_2,
        activation_function=activation_function,
        in_channels=in_channels
    )

if __name__ == '__main__':
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    img_size = 224
    in_channels = 3  # Grayscale canvas
    actor_network_input = 10

    # Create dummy input tensors for two images
    dummy_image1 = torch.randn(batch_size, in_channels, img_size, img_size).to(device)
    dummy_image2 = torch.randn(batch_size, in_channels, img_size, img_size).to(device)
    # dummy_image2 = torch.randn(batch_size, 1, img_size, img_size).to(device)
    dummy_actor_params = torch.randn(batch_size, actor_network_input).to(device)

    # Create a MergedNetwork
    processor = create_merged_network(
        image_encoder_model='resnet50',
        image_encoder_model_2='efficientnet_b0',
        pretrained=True,
        fine_tune_encoder=True,
        fine_tune_encoder_2=False,
        actor_network_input=actor_network_input,
        hidden_layers=[512, 256, 128],
        merged_output_size=10,
        activation_function='ReLU',
        in_channels=in_channels
    ).to(device)

    processor.eval()
    with torch.no_grad():
        output = processor(dummy_image1, dummy_image2, dummy_actor_params)
        print("Output shape:", output.shape)
        print("Output:", output[0])

    # Example with a custom encoder
    class CustomEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(128 * (img_size // 4) * (img_size // 4), 1024)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x

    custom_encoder = CustomEncoder().to(device)
    custom_encoder_2 = CustomEncoder().to(device)
    processor_with_custom_encoder = create_merged_network(
        use_custom_encoder=True,
        use_custom_encoder_2=True,
        custom_encoder=custom_encoder,
        custom_encoder_2=custom_encoder_2,
        actor_network_input=actor_network_input,
        hidden_layers=[512, 256],
        merged_output_size=1024,
        activation_function='Tanh'
    ).to(device)

    processor_with_custom_encoder.eval()
    with torch.no_grad():
        output = processor_with_custom_encoder(dummy_image1, dummy_image2, dummy_actor_params)
        print("Output shape (Custom Encoder):", output.shape)
        print("Output (Custom Encoder):", output[0])