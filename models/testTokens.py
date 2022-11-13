import torch
from models.InputOffsetMLP import PointWiseInputOffsetMLP
from models.PositionMLP import PointWisePositionMLP
from models.OutputOffsetMLP import PointWiseOutputOffsetMLP
from models.xcit import Encoder, Decoder


def main():
    colors = torch.rand((5, 20, 20, 3)).view(5,-1,3)
    positions = torch.rand((5,20,20,2)).view(5,-1,2)

    #create tokens
    emb_pos = PointWisePositionMLP()(positions)
    emb_col = PointWiseInputOffsetMLP()(colors)
    tokens = torch.cat([emb_pos, emb_col], 2)

    #encoder
    enc = Encoder(embed_dim=64)
    shape_code = enc(tokens)

    #decoder's position mlp
    outPos = PointWisePositionMLP(token_size=64, hidden_dim=64)
    dec_positions = outPos(positions)

    #decoder
    dec = Decoder(embed_dim=64)
    out_decoder = dec(dec_positions, shape_code)

    #Output offset mlp
    out = PointWiseOutputOffsetMLP()(out_decoder)

    print(out.shape)


if __name__ == "__main__":
    main()


