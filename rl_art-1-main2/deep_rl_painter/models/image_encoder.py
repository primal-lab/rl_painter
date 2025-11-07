# This file contains the ImageEncoder class, which is a neural network model for encoding images.
# It uses different architectures (ResNet, EfficientNet, etc) based on the specified model name.
# This is imported in the Critic and Actor classes.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import warnings
from typing import Dict, Optional
from torchsummary import summary

# Turn off all warnings
warnings.filterwarnings("ignore")


class ImageEncoder(nn.Module):
    """
    Encodes input images using various CNN architectures.
    """

    def __init__(self, model_name: str = 'resnet50', pretrained: bool = True, fine_tune: bool = True, custom_model: Optional[nn.Module] = None, in_channels: int = 3, unfreeze_last_n: int = 1) -> None:
        """
        Args:
            model_name (str): Name of the CNN architecture to use (e.g., 'resnet50', 'efficientnet_b0', 'vgg16').
                           Ignored if custom_model is provided.
            pretrained (bool): Whether to load pre-trained weights from ImageNet.
                               Ignored if custom_model is provided.
            fine_tune (bool): Whether to fine-tune the pre-trained weights during training.
            custom_model (nn.Module, optional): A custom PyTorch model to use for image encoding.
                                             If provided, model_name and pretrained are ignored.
            in_channels (int): Number of input channels (3 for RGB, 1 for grayscale).
        """

        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.fine_tune = fine_tune
        self.custom_model = custom_model
        self.cnn = None
        self.in_channels = in_channels
        self.unfreeze_last_n = unfreeze_last_n # last 1 layer
        self._initialize_model()


    def _initialize_model(self) -> None:
        """
        Initializes the CNN model based on the specified architecture.
        """

        # Select CNN architecture
        if self.custom_model is not None:
            self.cnn = self.custom_model
        elif self.model_name.startswith('resnet'):
            self.cnn = self._get_resnet(self.model_name)
        elif self.model_name.startswith('efficientnet'):
            self.cnn = self._get_efficientnet(self.model_name)
        elif self.model_name.startswith('vgg'):
            self.cnn = self._get_vgg(self.model_name)
        elif self.model_name == 'inception_v3':
            self.cnn = self._get_inception_v3()
        elif self.model_name == 'convnext_tiny':
            self.cnn = self._get_convnext_tiny()
        elif self.model_name.startswith('mobilenet'):
            self.cnn = self._get_mobilenet(self.model_name)    
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

        self._freeze_params()

    def _get_resnet(self, model_name: str) -> nn.Module:
        """
        Returns a ResNet model.
        """
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=self.pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=self.pretrained)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=self.pretrained)
        else:
            raise ValueError(f"Invalid ResNet model name: {model_name}")
        
        """if self.in_channels == 1:
            # Modify the first convolutional layer to accept different number of input channels
            model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)"""
        if self.in_channels != 3:
            raise ValueError(f"Invalid number of input channels: {self.in_channels}. ResNet only supports 1 or 3 channels.")
        # Remove the last fully connected layer
        # conv1..layer4 (drop avgpool, fc)
        # removing adpative avg pool layer to match other encoders
        model = nn.Sequential(*list(model.children())[:-2]) #no flatten here
        return model

    def _get_efficientnet(self, model_name: str) -> nn.Module:
        """
        Returns an EfficientNet model.
        """

        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b2':
            model = models.efficientnet_b2(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b4':
            model = models.efficientnet_b4(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b5':
            model = models.efficientnet_b5(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b6':
            model = models.efficientnet_b6(pretrained=self.pretrained)
        elif model_name == 'efficientnet_b7':
            model = models.efficientnet_b7(pretrained=self.pretrained)
        else:
            raise ValueError(f"Invalid EfficientNet model name: {model_name}")
        
        if self.in_channels == 1:
            # Modify the first convolutional layer
            model.features[0][0] = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        elif self.in_channels != 3:
            raise ValueError(f"Invalid number of input channels: {self.in_channels}. EfficientNet only supports 1 or 3 channels.")

        # Remove the last fully connected layer
        model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        return model

    def _get_vgg(self, model_name: str) -> nn.Module:
        """
        Returns a VGG model.
        """
        if model_name == 'vgg11':
            model = models.vgg11(pretrained=self.pretrained)
        elif model_name == 'vgg13':
            model = models.vgg13(pretrained=self.pretrained)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=self.pretrained)
        elif model_name == 'vgg19':
            model = models.vgg19(pretrained=self.pretrained)
        else:
            raise ValueError(f"Invalid VGG model name: {model_name}")
    
        if self.in_channels == 1:
            # Modify the first convolutional layer
            model.features[0] = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1)
        elif self.in_channels != 3:
            raise ValueError(f"Invalid number of input channels: {self.in_channels}. VGG only supports 1 or 3 channels.")

        # Remove the classifier part
        model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        return model

    def _get_inception_v3(self) -> nn.Module:
        """
        Returns an Inception v3 model.
        """
        model = models.inception_v3(
            pretrained=self.pretrained, transform_input=False)  # transform_input=False
        
        if self.in_channels == 1:
            # Modify the first convolutional layer
            model.Conv2d_1a_3x3.conv = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=0, bias=False)
        elif self.in_channels != 3:
            raise ValueError(f"Invalid number of input channels: {self.in_channels}. Inception v3 only supports 1 or 3 channels.")
        # Remove the auxiliary classifier.
        # model.AuxLogits = None
        return model

    def _get_convnext_tiny(self):
        """
        Returns a ConvNeXt-tiny model.
        """
        model = models.convnext_tiny(pretrained=self.pretrained)

        if self.in_channels == 1:
            # Modify the first convolutional layer
            model.features[0] = nn.Conv2d(self.in_channels, 96, kernel_size=4, stride=4, padding=0)
        elif self.in_channels != 3:
            raise ValueError(f"Invalid number of input channels: {self.in_channels}. ConvNeXt only supports 1 or 3 channels.")

        # Remove the classification head
        model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        return model

    def _get_mobilenet(self, model_name: str) -> nn.Module:
        """
        Returns MobileNet (v2/v3) truncated to the feature extractor so that
        forward() receives a feature map (B, C, H, W).
        """
        # Handle torchvision old/new APIs
        def load_mnv2(pretrained: bool):
            try:
                # torchvision >= 0.13
                from torchvision.models import MobileNet_V2_Weights
                return models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            except Exception:
                return models.mobilenet_v2(pretrained=pretrained)

        def load_mnv3_small(pretrained: bool):
            try:
                from torchvision.models import MobileNet_V3_Small_Weights
                return models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            except Exception:
                return models.mobilenet_v3_small(pretrained=pretrained)

        def load_mnv3_large(pretrained: bool):
            try:
                from torchvision.models import MobileNet_V3_Large_Weights
                return models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
            except Exception:
                return models.mobilenet_v3_large(pretrained=pretrained)

        if model_name == 'mobilenet_v2':
            backbone = load_mnv2(self.pretrained)
            feat = backbone.features  # (B, 1280, 7, 7) at 224
        elif model_name == 'mobilenet_v3_small':
            backbone = load_mnv3_small(self.pretrained)
            feat = backbone.features  # (B, 576, ~7, ~7)
        elif model_name == 'mobilenet_v3_large':
            backbone = load_mnv3_large(self.pretrained)
            feat = backbone.features  # (B, 960, ~7, ~7)
        else:
            raise ValueError(f"Invalid MobileNet model name: {model_name}")

        if self.in_channels != 3:
            raise ValueError(f"in_channels={self.in_channels} unsupported for MobileNet (use 1 or 3).")

        return feat  # returns a Module producing (B, C, H, W)
    

    def _freeze_params(self) -> None:
        # Freezes the parameters of the CNN if not fine-tuning.
        if not self.fine_tune:
            for param in self.cnn.parameters():
                param.requires_grad = False
    
    """def _freeze_params(self) -> None:
        # 1) freeze everything (no matter if fine_tune = F/T)
        for p in self.cnn.parameters():
            p.requires_grad = False

        # 2) partial-unfreeze wins if specified
        if self.model_name.startswith('resnet') and isinstance(self.cnn, nn.Sequential) and self.unfreeze_last_n > 0:
            mods = list(self.cnn.children())
            # indices for stages we may want to unfreeze
            # layer1=4, layer2=5, layer3=6, layer4=7
            # 0 conv1, 1 bn1, 2 relu, 3 maxpool,
            # 4 layer1, 5 layer2, 6 layer3, 7 layer4   ← the “stages”
            stage_idxs = [4, 5, 6, 7]  # layer1..layer4
            to_unfreeze = stage_idxs[-self.unfreeze_last_n:]
            for idx in to_unfreeze:
                for p in mods[idx].parameters():
                    p.requires_grad = True
            # keep BN in frozen parts in eval mode
            for idx in [i for i in stage_idxs if i not in to_unfreeze]:
                for m in mods[idx].modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        for p in m.parameters():
                            p.requires_grad = False
            return  # done

        # 3) otherwise: full fine-tune if requested
        if self.fine_tune:
            for p in self.cnn.parameters():
                p.requires_grad = True"""
           

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input image.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Image features.
        """
        #print("(in image_encoder.py)ImageEncoder got input of shape:", x.shape)
        features = self.cnn(x) # (B, C, h, w)
        #print(f"[ImageEncoder] Feature map before pooling: {features.shape}")
        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)  # -> (B, C) e.g., (B, 512) for resnet18
        #print(f"[ImageEncoder] Feature map after pooling: {features.shape}")
        # 1×1 pooling → total collapse → no location info.
        # 7×7 pooling → coarse grid → partial location info

        # Log output shape from image encoder
        """os.makedirs("logs/model", exist_ok=True)
        with open("logs/model/image_encoder.log", "a") as f:
            f.write(f"Input shape: {x.shape}, Output shape: {features.shape}\n")"""

        return features


def get_image_encoder(model_name: str = 'resnet50', pretrained: bool = True, fine_tune: bool = True, custom_model: Optional[nn.Module] = None, in_channels: int = 3) -> ImageEncoder:
    """
    Returns an ImageEncoder instance with the specified architecture.

    Args:
        model_name (str): Name of the CNN architecture to use.
        pretrained (bool): Whether to load pre-trained weights.
        fine_tune (bool): Whether to fine-tune the weights.
        custom_model (nn.Module, optional): A custom PyTorch model to use.
        in_channels (int): Number of input channels (3 for RGB, 1 for grayscale).
    Returns:
        ImageEncoder: An instance of the ImageEncoder class.
    """
    return ImageEncoder(model_name, pretrained, fine_tune, custom_model, in_channels)


def print_model_summary(model: nn.Module, input_size: tuple) -> None:
    """
    Prints the summary of the model.

    Args:
        model (nn.Module): The model to summarize.
        input_size (tuple): The size of the input tensor.
    """
    print("Model Summary:")
    summary(model, input_size)
    print("Total Parameters:", sum(p.numel()
          for p in model.parameters() if p.requires_grad))
    print("Trainable Parameters:", sum(p.numel()
          for p in model.parameters() if p.requires_grad))


if __name__ == '__main__':
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    img_size = 224  # Or 299 for InceptionV3
    channels = 3  # RGB images
    channels = 1  # Grayscale images

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, channels,
                              img_size, img_size).to(device)

    # Test with different models
    models_to_test = ['resnet18', 'resnet50', 'efficientnet_b0',
                      'vgg16', 'inception_v3', 'convnext_tiny']
    for model_name in models_to_test:

        encoder = get_image_encoder(
            model_name=model_name, pretrained=True, fine_tune=False, in_channels = channels).to(device)
        encoder.eval()  # Set to evaluation mode
        with torch.no_grad():
            features = encoder(dummy_input)
            print(f"{model_name} Output shape: {features.shape}")
        del encoder  # Clear memory
        torch.cuda.empty_cache()
