from typing import List
import torch.nn as nn

class JointAutoencoder(nn.Module):
    def __init__(self, movie_ae, person_ae):
        super().__init__()
        self.movie_ae = movie_ae
        self.person_ae = person_ae
        self.mov_enc = movie_ae.encoder
        self.per_enc = person_ae.encoder
        self.mov_dec = movie_ae.decoder
        self.per_dec = person_ae.decoder

    def forward(self, movie_in: List, person_in: List):
        m_z = self.mov_enc(movie_in)
        p_z = self.per_enc(person_in)
        m_rec = self.mov_dec(m_z)
        p_rec = self.per_dec(p_z)
        return m_z, p_z, m_rec, p_rec
