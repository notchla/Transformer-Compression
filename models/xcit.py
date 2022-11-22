import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath
from functools import partial
from utils import to_2tuple, trunc_normal_

class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows to augment the
    implicit communication performed by the block diagonal scatter attention. Implemented using 2 layers of separable
    3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(
            in_features, in_features, kernel_size=kernel_size, padding=padding, groups=in_features)
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = torch.nn.Conv2d(
            in_features, out_features, kernel_size=kernel_size, padding=padding, groups=out_features)

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x 

def conv1x1(in_features, out_features, act_layer):
    return nn.Sequential(nn.Linear(in_features, out_features), act_layer())
    
class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA)
    Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
    normalized) Cross-covariance matrix (Q^T \\cdot K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # Result of next line is (qkv, B, num (H)eads,  (C')hannels per head, N)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        
        # Paper section 3.2 l2-Normalization and temperature scaling
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, H, C', N), permute -> (B, N, H, C')
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class XCABlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=1.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim)
        # self.local_mp = LPI(in_features=dim, act_layer=act_layer)
        self.local_mp = conv1x1(dim, dim, act_layer)        

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim))
        self.gamma3 = nn.Parameter(eta * torch.ones(dim))
        self.gamma2 = nn.Parameter(eta * torch.ones(dim))

    def forward(self, x, H = None, W = None):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        # NOTE official code has 3 then 2, so keeping it the same to be consistent with loaded weights
        # See https://github.com/rwightman/pytorch-image-models/pull/747#issuecomment-877795721
        # x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x)))

        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x



class Encoder(nn.Module):
    def __init__(
        self, embed_dim=128, 
        depth=4, num_heads=4, mlp_ratio=4., qkv_bias=True,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.,) -> None:
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int): patch size
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate after positional embedding, and in XCA/CA projection + MLP
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate (constant across all layers)
            act_layer (nn.Module) activation layer
            norm_layer: (nn.Module): normalization layer
            eta: (float) layerscale initialization value
        Notes:
            - Although `norm_layer` is user specifiable, there are hard-coded `BatchNorm2d`s in the local patch
              interaction (class LPI)
        """
        super().__init__()
        # if type(img_size) is not tuple:
        #     img_size = to_2tuple(img_size)

        # assert (img_size[0] % patch_size == 0) and (img_size[0] % patch_size == 0), \
        #     '`patch_size` should divide image dimensions evenly'

        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)


        # self.H = img_size[0]
        # self.W = img_size[1]

        self.blocks = nn.ModuleList([
        XCABlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=drop_path_rate, act_layer=act_layer, norm_layer=norm_layer, eta=eta)
        for _ in range(depth)])

        #Init weights
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward_features(self, x):
        B = x.shape[0]
        # x is (B, N, C). (Hp, Hw) is (height in units of patches, width in units of patches)

        x = self.pos_drop(x)

        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)

        for blk in self.blocks:
            
            x = blk(x)

        return x[:, 0] #cls token
    
    def forward(self, x):
        x = self.forward_features(x)
        return x


class StyleMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, n_layers=4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        act_fun = nn.ReLU
        layers = []

        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(act_fun())
        for _ in range(n_layers-1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(act_fun())

        self.model = nn.Sequential(*(layers))
        self.model.apply(self._init_weights)
            

    def forward(self, x):
        return self.model(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Decoder(nn.Module):
    def __init__(
        self, embed_dim=128, 
        depth=4, num_heads=4, mlp_ratio=4., qkv_bias=True,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1., style_hidden=None) -> None:
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int): patch size
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate after positional embedding, and in XCA/CA projection + MLP
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate (constant across all layers)
            act_layer (nn.Module) activation layer
            norm_layer: (nn.Module): normalization layer
            eta: (float) layerscale initialization value
        Notes:
            - Although `norm_layer` is user specifiable, there are hard-coded `BatchNorm2d`s in the local patch
              interaction (class LPI)
        """
        super().__init__()
        # if type(img_size) is not tuple:
        #     img_size = to_2tuple(img_size)

        # assert (img_size[0] % patch_size == 0) and (img_size[0] % patch_size == 0), \
        #     '`patch_size` should divide image dimensions evenly'

        self.embed_dim = embed_dim
        self.pos_drop = nn.Dropout(p=drop_rate)

        style_hidden = style_hidden or embed_dim
        self.style_mlp = StyleMlp(embed_dim, hidden_features=style_hidden)
        self.modulation_layers = nn.ModuleList([nn.Linear(style_hidden, embed_dim) for _ in range(depth)])


        # self.H = img_size[0]
        # self.W = img_size[1]

        self.blocks = nn.ModuleList([
        XCABlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=drop_path_rate, act_layer=act_layer, norm_layer=norm_layer, eta=eta)
        for _ in range(depth)])

        #Init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward_features(self, x, cls_token):
        B = x.shape[0]
        # x is (B, N, C). (Hp, Hw) is (height in units of patches, width in units of patches)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):

            B, N, C = x.shape
            modulation = self.style_mlp(cls_token)
            modulation = self.modulation_layers[i](modulation)
            modulation = modulation.expand(N, -1, -1).permute(1, 0, 2)
            x = blk(x*modulation) #try sum or sum and mul

        return x #cls token
    
    def forward(self, x, cls_token):
        x = self.forward_features(x, cls_token)
        return x
