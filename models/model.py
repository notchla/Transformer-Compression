from .xcit import Encoder, Decoder
from .InputOffsetMLP import PointWiseInputOffsetMLP
from .OutputOffsetMLP import PointWiseOutputOffsetMLP, ModulatedPointWiseOutputOffsetMLP
from .PositionMLP import PointWisePositionMLP
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, args, img_resolution) -> None:
        super().__init__()
        
        #make the hidden dimension hidden_ratio times the dimension of laten_token_size
        self.hidden_ratio = 1 if args.custom_hidden_multiplier is None else args.custom_hidden_multiplier
        token_size = args.token_size
        laten_token_size = int(token_size/2)
        self.input_mlp = PointWiseInputOffsetMLP(hidden_dim=laten_token_size*self.hidden_ratio, token_size=laten_token_size)
        self.position_mlp_encoder = PointWisePositionMLP(hidden_dim=laten_token_size*self.hidden_ratio, token_size=laten_token_size, args=args, img_resolution=img_resolution)
        self.encoder = Encoder(embed_dim=token_size)
        self.decoder = Decoder(embed_dim=token_size)
        self.position_mlp_decoder = PointWisePositionMLP(hidden_dim=token_size*self.hidden_ratio, token_size=token_size, args=args, img_resolution=img_resolution)
        self.output_mlp = ModulatedPointWiseOutputOffsetMLP(hidden_dim=token_size*self.hidden_ratio, token_size=token_size)
    
    def forward(self, img, coords):
        latent_offsets = self.input_mlp(img)
        latent_positions = self.position_mlp_encoder(coords)
        encoder_tokes = torch.cat((latent_offsets, latent_positions), -1)
        cls_token = self.encoder(encoder_tokes)
        latent_positions_decoder = self.position_mlp_decoder(coords)
        decoder_out = self.decoder(latent_positions_decoder, cls_token)
        out = self.output_mlp(decoder_out, cls_token)
        return out

    #Compression part, get the image codes
    def get_image_code(self, img, coords):
        latent_offsets = self.input_mlp(img)
        latent_positions = self.position_mlp_encoder(coords)
        encoder_tokes = torch.cat((latent_offsets, latent_positions), -1)
        cls_token = self.encoder(encoder_tokes)
        return cls_token

    #Decompression part, reconstruct the imgage. Forward(img, coords) == reconstruct(get_image_code(img, coords), coords)
    def reconstruct(self, image_code, coords):
        latent_positions_decoder = self.position_mlp_decoder(coords)
        decoder_out = self.decoder(latent_positions_decoder, image_code)
        out = self.output_mlp(decoder_out, image_code)
        return out
