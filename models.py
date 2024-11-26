from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio


HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    class Block(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
            # Skip connection
            self.skip = nn.Identity()


        def forward(self, x):
            return self.skip(x) + self.model(x)

    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 6,
    ):
        """
        Args:
            n_track (int): Number of points on each side of the track.
            n_waypoints (int): Number of waypoints to predict.
            hidden_dim (int): Size of hidden layers.
            num_layers (int): Number of residual blocks.
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Input size is 2 * n_track * 2 (left/right track with x and y coordinates)
        input_size = 2 * n_track * 2

        # Create hidden layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.ReLU())

        # Add residual blocks
        for _ in range(num_layers):
            layers.append(self.Block(hidden_dim))

        # Output layer to predict waypoints (2 * n_waypoints for x and y coordinates)
        layers.append(nn.Linear(hidden_dim, 2 * n_waypoints))

        self.network = nn.Sequential(*layers)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: Predicted waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate left and right track points
        track_combined = torch.cat([track_left, track_right], dim=1)  # Shape: (b, 2 * n_track, 2)

        # Flatten track points
        track_combined = track_combined.view(track_combined.size(0), -1)  # Shape: (b, 2 * n_track * 2)

        # Pass through the network
        waypoints = self.network(track_combined)

        # Reshape to (b, n_waypoints, 2)
        waypoints = waypoints.view(-1, self.n_waypoints, 2)
 
        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        raise NotImplementedError



class CNNPlanner(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_blocks: int = 3, n_waypoints: int = 3):
        super().__init__()

        self.n_waypoints = n_waypoints

        # First convolutional layer
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Stack of blocks
        layers = []
        c1 = base_channels
        for _ in range(n_blocks):
            c2 = c1 * 2
            layers.append(self.Block(c1, c2, stride=2))
            c1 = c2

        self.network = nn.Sequential(*layers)

        # Adaptive average pooling to reduce the spatial size to 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer to predict waypoints
        self.fc = nn.Linear(c1, n_waypoints * 2)

    class Block(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            # Skip connection with 1x1 convolution to match dimensions if needed
            if stride != 1 or in_channels != out_channels:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.skip = nn.Identity()

        def forward(self, x):
            return self.model(x) + self.skip(x)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image

        # Initial convolutional layer
        x = self.model(x)

        # Stack of blocks
        x = self.network(x)

        # Adaptive average pooling
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        # Reshape to (b, n_waypoints, 2)
        x = x.view(x.size(0), self.n_waypoints, 2)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (B, 3, 96, 128) representing the input image

        Returns:
            tensor of shape (B, n_waypoints, 2) representing the predicted waypoints
        """
        return self(x)




MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
