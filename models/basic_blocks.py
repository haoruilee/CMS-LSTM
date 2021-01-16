import torch
from torch import nn
from torch.nn import Module, Conv2d, Sigmoid, Tanh, ModuleList

__all__ = ['SE_Block']


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class _seblock(Module):
    def __init__(self, in_dim, scale=1, ):
        super(_seblock, self).__init__()
        self.scale = scale
        self.in_dim = in_dim
        self.f_key = Conv2d(in_channels=self.in_dim,
                            out_channels=self.in_dim,
                            kernel_size=1, stride=1, padding=0)
        self.f_query = Conv2d(in_channels=self.in_dim,
                              out_channels=self.in_dim,
                              kernel_size=1, stride=1, padding=0)

        self.f_value = Conv2d(in_channels=self.in_dim,
                              out_channels=self.in_dim,
                              kernel_size=1, stride=1, padding=0)

        self.block_num = scale ** 2
        self.softmax = nn.Softmax(dim=-1)

        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        x = inputs

        # input shape: b,c,h,2w
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3) // 2
        block_size = h // self.scale

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        value = torch.stack([value[:, :, :, :w], value[:, :, :, w:]], 4)  # B*N*H*W*2
        query = torch.stack([query[:, :, :, :w], query[:, :, :, w:]], 4)  # B*N*H*W*2
        key = torch.stack([key[:, :, :, :w], key[:, :, :, w:]], 4)  # B*N*H*W*2

        v_list = torch.split(value, block_size, dim=2)
        v_locals = torch.cat(v_list, 0)
        v_list = torch.split(v_locals, block_size, dim=3)
        v_locals = torch.cat(v_list)

        q_list = torch.split(query, block_size, dim=2)
        q_locals = torch.cat(q_list, 0)
        q_list = torch.split(q_locals, block_size, dim=3)
        q_locals = torch.cat(q_list)

        k_list = torch.split(key, block_size, dim=2)
        k_locals = torch.cat(k_list, 0)
        k_list = torch.split(k_locals, block_size, dim=3)
        k_locals = torch.cat(k_list)

        #  self-attention func
        def func(value_local, query_local, key_local):
            batch_size_new = value_local.size(0)
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size_new, self.in_dim, -1)

            query_local = query_local.contiguous().view(batch_size_new, self.in_dim, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size_new, self.in_dim, -1)

            sim_map = torch.bmm(query_local, key_local)
            sim_map = self.softmax(sim_map)

            context_local = torch.bmm(value_local, sim_map.permute(0, 2, 1))
            context_local = context_local.view(batch_size_new, self.in_dim, h_local, w_local, 2)
            return context_local

        context_locals = func(v_locals, q_locals, k_locals)

        b, c, h, w, _ = context_locals.shape

        context_list = torch.split(context_locals, b // self.scale, 0)
        context = torch.cat(context_list, dim=3)
        context_list = torch.split(context, b // self.scale // self.scale, 0)
        context = torch.cat(context_list, dim=2)

        context = torch.cat([context[:, :, :, :, 0], context[:, :, :, :, 1]], 3)

        return context + x


class seblock(_seblock):
    def __init__(self, in_dim, scale=1, ):
        super(seblock, self).__init__(in_dim, scale)


class SE_Block(Module):
    def __init__(self, in_dim, img_size):
        super(SE_Block, self).__init__()

        sizes = [1, 2, 4]  # divide 16*16 feature map into 16*16*1, 8*8*4, 4*4*16, etc.

        self.group = len(sizes)
        self.stages = []
        self.in_dim = in_dim
        filter_size = 5
        self.padding = int((filter_size - 1) / 2)  # in this way the output has the same size
        self.stages = ModuleList(
            [self._make_stage(in_dim, size, )
             for size in sizes])

        self.conv_bn = nn.Sequential(
            Conv2d(in_dim * self.group, in_dim, kernel_size=filter_size, padding=self.padding),
            nn.LayerNorm([in_dim, img_size, 2 * img_size])
        )

        self.w_z = Conv2d(in_channels=2 * in_dim, out_channels=in_dim, kernel_size=1)

        self.w_h2h = nn.Sequential(
            nn.Conv2d(in_channels=in_dim,
                      out_channels=in_dim * 3,
                      kernel_size=filter_size,
                      stride=1,
                      padding=self.padding,
                      bias=False),
            nn.LayerNorm([in_dim * 3, img_size, img_size])
        )
        self.w_z2h = nn.Sequential(
            nn.Conv2d(in_channels=in_dim,
                      out_channels=in_dim * 3,
                      kernel_size=filter_size,
                      stride=1,
                      padding=self.padding,
                      bias=False),
            nn.LayerNorm([in_dim * 3, img_size, img_size])
        )
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        self.norm_mem = nn.LayerNorm([in_dim, img_size, img_size])

    def _make_stage(self, in_dim, size):
        return seblock(in_dim, size)

    def forward(self, h_cur, m_cur):
        feats = torch.cat([h_cur, m_cur], dim=-1)
        priors = [stage(feats) for stage in self.stages]
        context = torch.cat(priors, dim=1)
        output = self.conv_bn(context)

        z_h, z_m = torch.split(output, output.shape[-1] // 2, -1)
        z = self.w_z(torch.cat([z_h, z_m], dim=1))

        z2h = self.w_z2h(z)  # [b 3*c h w]
        h2h = self.w_h2h(h_cur)  # [b 3*c h w]

        i, g, o = torch.split(h2h + z2h, self.in_dim, dim=1)

        o = self.sigmoid(o)
        g = self.tanh(g)
        i = self.sigmoid(i)

        m_next = m_cur * (1 - i) + i * g
        m_next = self.norm_mem(m_next)

        h_next = m_next * o

        return h_next, m_next, [z_h, z_m]
