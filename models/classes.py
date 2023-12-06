import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, m):
        super(Autoencoder, self).__init__()

        # Calculate the step size for interpolation
        # This determines the difference in dimensionality between each hidden layer
        step = (input_dim - latent_dim) // (m + 1)

        # Create encoder layers
        # The dimensions of these layers decrease from input_dim to latent_dim
        encoder_layers = []
        current_dim = input_dim  # Initialize current_dim as input_dim
        for i in range(m + 1):
            # Calculate the output dimension for this layer
            out_dim = int(input_dim - step * (i + 1))
            # Add a linear layer followed by a sigmoid activation
            encoder_layers.append(nn.Linear(current_dim, out_dim))
            encoder_layers.append(nn.Sigmoid())
            # Update current_dim for the next layer
            current_dim = out_dim

        # Add the latent layer
        # This layer compresses the representation to the latent dimension
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        encoder_layers.append(nn.Sigmoid())

        # Create decoder layers
        # The dimensions of these layers increase from latent_dim back to input_dim
        decoder_layers = []
        current_dim = latent_dim  # Initialize current_dim as latent_dim
        for i in range(m + 1):
            # Calculate the output dimension for this layer
            out_dim = int(latent_dim + step * (i + 1))
            # Add a linear layer followed by a sigmoid activation
            decoder_layers.append(nn.Linear(current_dim, out_dim))
            decoder_layers.append(nn.Sigmoid())
            # Update current_dim for the next layer
            current_dim = out_dim

        # Ensure the final output layer matches the original input dimension
        # This is necessary to reconstruct the input from the latent representation
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())

        # Define the encoder and decoder
        # nn.Sequential wraps the individual layers into a single module
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # The forward method defines the data flow through the network
        x = self.encoder(x)  # Pass input through the encoder
        x = self.decoder(x)  # Then pass the encoded output through the decoder
        return x
