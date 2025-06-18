""" 
Input: input_image_1 (canvas), input_image_2 (target_image), action_input (action_space, x,y,r,g,b, width)
Output: action (x,y,r,g,b, width)
"""

import os
import torch
import torch.nn as nn
import warnings
from .merge_network import create_merged_network
from typing import Optional
# Turn off all warnings
warnings.filterwarnings("ignore")

class Actor(nn.Module):
    def __init__(self, 
                 image_encoder_model: str = 'resnet50',
                 image_encoder_model_2: Optional[str] = 'resnet50',
                 pretrained: bool = True,
                 fine_tune_encoder: Optional[bool] = True,
                 fine_tune_encoder_2: Optional[bool] = True,
                 actor_network_input: int = 6,  # 2 for x,y and 6 for x,y,r,g,b, width
                 hidden_layers: list = [512, 256, 128, 64, 32],
                 use_custom_encoder: Optional[bool] = False,
                 use_custom_encoder_2: Optional[bool] = False,
                 custom_encoder: nn.Module = None,
                 custom_encoder_2: nn.Module = None,
                 activation_function: str = 'LeakyReLU',
                 in_channels: int = 1, # 3 for rgb
                 out_neurons: int = 6,
                 ) -> None:
        """
        Initialize the Actor model.
        Args:
            image_encoder_model (str): Name of the image encoder model architecture ('resnet', 'efficientnet', 'cae' : Custom).
            image_encoder_model_2 (str): Name of the second image encoder model architecture ('resnet', 'efficientnet', 'cae' : Custom).
            pretrained (bool): Whether to use a pretrained model (default: True).
            fine_tune_encoder (bool): Whether to fine-tune the encoder (default: True).
            fine_tune_encoder_2 (bool): Whether to fine-tune the second encoder (default: True).
            actor_network_input (int): Number of input features for the actor network - number of non image inputs that get concatenated with the image features. 
            hidden_layers (list): List of hidden layer sizes.
            out_neurons (int): Number of output neurons.
            in_channels (int): Number of input channels.
            use_custom_encoder (bool): Whether to use a custom encoder.
            use_custom_encoder_2 (bool): Whether to use a second custom encoder.
            custom_encoder (nn.Module): Custom encoder model.
            custom_encoder_2 (nn.Module): Second custom encoder model.
            activation_function (str): Activation function to use ('ReLU', 'LeakyReLU', etc.).
        """

        super(Actor, self).__init__()
        self.image_encoder_model = image_encoder_model
        self.image_encoder_model_2 = image_encoder_model_2
        self.pretrained = pretrained
        self.fine_tune_encoder = fine_tune_encoder
        self.fine_tune_encoder_2 = fine_tune_encoder_2
        self.actor_network_input = actor_network_input
        self.hidden_layers = hidden_layers
        self.out_neurons = out_neurons
        self.in_channels = in_channels
        self.use_custom_encoder = use_custom_encoder
        self.use_custom_encoder_2 = use_custom_encoder_2
        self.custom_encoder = custom_encoder
        self.custom_encoder_2 = custom_encoder_2
        self.activation_function = activation_function

        self.model_name = image_encoder_model
        self.model_name_2 = image_encoder_model_2

        self.model = create_merged_network(
            image_encoder_model=self.image_encoder_model,
            image_encoder_model_2=self.image_encoder_model_2,
            pretrained=self.pretrained,
            fine_tune_encoder=self.fine_tune_encoder,
            fine_tune_encoder_2=self.fine_tune_encoder_2,
            actor_network_input=self.actor_network_input,
            hidden_layers=self.hidden_layers,
            merged_output_size=self.out_neurons,
            use_custom_encoder=self.use_custom_encoder,
            use_custom_encoder_2=self.use_custom_encoder_2,
            custom_encoder=self.custom_encoder,
            custom_encoder_2=self.custom_encoder_2,
            activation_function=self.activation_function,
            in_channels=self.in_channels,
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, input_image_1, input_image_2, action_input):
        """
        Forward pass through the model.
        Args:
            input_image_1 (torch.Tensor): First (canvas) input image tensor. # (B, H, W, C) 
            input_image_2 (torch.Tensor): Second (target_image) input image tensor. # (B, H, W, C) 
            action_input (torch.Tensor): Action input tensor - prev_action. # (B, action_dim)
        Returns:
            torch.Tensor: Output of the model.
        """
        os.makedirs("logs/model", exist_ok=True) 

        # Log input shapes: canvas, target, prev action
        #with open("logs/model/actor_input_shapes.log", "a") as f:
         #   f.write(f"Canvas: {input_image_1.shape}, Target: {input_image_2.shape}, PrevAction: {action_input.shape}\n")

        # Inputs are already (B, C, H, W), no permute needed
        input_image_1 = input_image_1.to(self.device)
        input_image_2 = input_image_2.to(self.device)
        action_input = action_input.to(self.device) #!
        
        # call merge_network
        out = self.model(input_image_1, input_image_2, action_input)

        # Normalization - actor.py - why?
        #import pdb
        #pdb.set_trace()
        # out of every single batch, get the first 2 columns
        direction = out[:, :2]
        norm = torch.norm(direction, dim=1)
        normalized_direction = direction / (norm.unsqueeze(1) + 1e-16)
        out = torch.cat([normalized_direction, out[:, 2:]], dim=1)

        # Log output shape and values from Actor
        """with open("logs/model/actor_output.log", "a") as f:
            f.write(f"Action shape: {out.shape}, Values: {out.detach().cpu().numpy().tolist()}\n")"""

        #print(f"Action shape: {out.shape}, Values: {out.detach().cpu().numpy().tolist()}\n")

        return out # (B, action_dim)

    def save_model(self, path):
        """
        Save the model state dictionary to a file.
        Args:
            path (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Load the model state dictionary from a file.
        Args:
            path (str): Path to load the model from.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)


# Example/test code (runs only when script is executed directly)
if __name__ == "__main__":
    # Example parameters
    image_encoder_model = 'resnet18'
    image_encoder_model_2 = 'resnet18'
    pretrained = True
    fine_tune_encoder = True
    fine_tune_encoder_2 = True
    actor_network_input = 6  # Example input size
    hidden_layers = [512, 256, 128, 64, 32]
    out_neurons = 6
    in_channels = 3

    # Create an instance of the Actor model
    actor_model = Actor(image_encoder_model=image_encoder_model,
                        image_encoder_model_2=image_encoder_model_2,
                        pretrained=pretrained,
                        fine_tune_encoder=fine_tune_encoder,
                        fine_tune_encoder_2=fine_tune_encoder_2,
                        actor_network_input=actor_network_input,
                        hidden_layers=hidden_layers,
                        out_neurons=out_neurons,
                        in_channels=in_channels)

    # Print the model summary
    # print(actor_model)

    # Print the model summary using torchsummary
    # summary(actor_model, (in_channels, 224, 224), batch_size=1, device="cuda" if torch.cuda.is_available() else "cpu")
    # Create a dummy input tensor
    dummy_input = torch.randn(1, in_channels, 224, 224).to(actor_model.device)
    dummy_input_2 = torch.randn(
        1, in_channels, 224, 224).to(actor_model.device)
    dummy_action_input = torch.randn(
        1, actor_network_input).to(actor_model.device)
    # Forward pass through the model
    output = actor_model(dummy_input, dummy_input_2, dummy_action_input)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.detach().cpu().numpy()}")

    #  save and load model works
    # Save the model
    # save_path = "test/actor_model.pth"
    # actor_model.save_model(save_path)
    # print(f"Model saved to {save_path}")
    # # Load the model
    # loaded_actor_model = Actor(image_encoder_model=image_encoder_model,
    #                             image_encoder_model_2=image_encoder_model_2,
    #                             pretrained=pretrained,
    #                             fine_tune_encoder=fine_tune_encoder,
    #                             fine_tune_encoder_2=fine_tune_encoder_2,
    #                             actor_network_input=actor_network_input,
    #                             hidden_layers=hidden_layers,
    #                             out_neurons=out_neurons,
    #                             in_channels=in_channels)
    # loaded_actor_model.load_model(save_path)
    # print(f"Model loaded from {save_path}")
    # # Forward pass through the loaded model
    # loaded_output = loaded_actor_model(dummy_input, dummy_input_2, dummy_action_input)
    # print(f"Loaded output shape: {loaded_output.shape}")
    # # Check if the outputs are close
    # assert torch.allclose(output, loaded_output, atol=1e-6), "Loaded model output does not match the original model output."
    # print("Loaded model output matches the original model output.")
