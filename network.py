from module import *
from utils import get_positional_table, get_sinusoid_encoding_table
import hyperparams as hp
import copy

class Encoder(nn.Module):
    """
    Encoder Network
    """
    def __init__(self, embedding_size, num_hidden):
        """
        :Parameters embedding_size: dimension of embedding
        :Parameters num_hidden: dimensions of hidden
        
        """
        super(Encoder, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.positional_embedding = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0),
                                                                 freeze=True)
        self.positional_dropout = nn.Dropout(p=0.1)
        self.encoder_prenet = EncoderPrenet(embedding_size, num_hidden)
        self.layers = clones(Attention(num_hidden), 3)
        self.feed_forward_networks = clones(FFN(num_hidden), 3)
        
    def forward(self, x, pos):
    
        # Get character mask
        if self.training:
            character_mask = pos.ne(0).type(torch.float)
            mask = pos.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)
        
        else:
            character_mask, mask = None, None
    
        # Encoder pre-network
        x = self.encoder_prenet(x)
    
        # Get positional embedding, apply alpha and add
        pos = self.positional_embedding(pos)
        x = pos * self.alpha + x
    
        # Positional Dropout
        x = self.positional_dropout(x)
    
        # Attention encoder-decoder
        attentions = list()
        for layer, ffn in zip(self.layers, self.feed_forward_networks):
            x, attn = layer(x, x, mask=mask, query_mask=character_mask)
            x = ffn(x)
            attentions.append(attn)
        
        return x, character_mask, attentions

class MelDecoder(nn.Module):
    """
    Decoder Network
    """
    def __init__(self, num_hidden):
        """
        :Parameters num_hidden: dimension of hidden
        """
        super(MelDecoder, self).__init__()
        self.positional_embedding = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0),
                                                                 freeze=True)
        self.positional_dropout = nn.Dropout(p=0.1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.decoder_prenet = Prenet(hp.num_mels, num_hidden * 2, num_hidden, p=0.2)
        self.normalization = Linear(num_hidden, num_hidden)
        
        self.self_attention_layers = clones(Attention(num_hidden), 3)
        self.dot_product_attention_layers = clones(Attention(num_hidden), 3)
        self.feed_forward_networks = clones(FFN(num_hidden), 3)
        self.mel_linear = Linear(num_hidden, hp.num_mels * hp.outputs_per_step)
        self.stop_linear = Linear(num_hidden, 1, w_init='sigmoid')
        
        self.post_convolutional_net = PostConvNet(num_hidden)
        
    def forward(self, memory, decoder_input, character_mask, pos):
        batch_size = memory.size(0)
        decoder_len = decoder_input.size(1)
    
        # get decoder mask with triangular matrix
        if self.training:
            m_mask = pos.ne(0).type(torch.float)
            mask = m_mask.eq(0).unsqueeze(1).repeat(1, decoder_len, 1)
            if next(self.parameters()).is_cuda:
                mask = mask + torch.triu(torch.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1, 1).byte()
            else:
                mask = mask + torch.triu(torch.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            
            mask = mask.gt(0)
            zero_mask = character_mask.eq(0).unsqueeze(-1).repeat(1, 1, decoder_len)
            zero_mask = zero_mask.transpose(1, 2)
        else:
            if next(self.parameters()).is_cuda:
                mask = torch.triu(torch.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1, 1).byte()
            else:
                mask = torch.triu(torch.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()

            mask = mask.gt(0)
            m_mask, zero_mask = None, None
        
        # Decoder pre-network
        decoder_input = self.decoder_prenet(decoder_input)
    
        # Centered Position
        decoder_input = self.normalization(decoder_input)
    
        # Get positional embedding, apply alpha and add
        pos = self.positional_embedding(pos)
        decoder_input = pos * self.alpha + decoder_input
    
        # Positional Dropout
        decoder_input = self.positional_dropout(decoder_input)
    
        # Attention decoder-decoder, encoder-decoder
        dot_product_attention = list()
        decoder_attention = list()
    
        for selfattn, dotattn, ffn in zip(self.self_attention_layers, self.dot_product_attention_layers, self.feed_forward_networks):
            decoder_input, attention_decoder = selfattn(decoder_input, decoder_input, mask=mask, query_mask=m_mask)
            decoder_input, attention_dot = dotattn(memory, decoder_input, mask=zero_mask, query_mask=m_mask)
            decoder_input = ffn(decoder_input)
            dot_product_attention.append(attention_dot)
            decoder_attention.append(attention_decoder)
        
        # Mel linear projection
        mel_out = self.mel_linear(decoder_input)
    
        # Post Mel Network
        postnet_input = mel_out.transpose(1, 2)
        out = self.post_convolutional_net(postnet_input)
        out = postnet_input + out
        out = out.transpose(1, 2)
    
        # Stop tokens
        stop_tokens = self.stop_linear(decoder_input)
    
        return mel_out, out, dot_product_attention, stop_tokens, decoder_attention

class Model(nn.Module):
    """
    Transformer Network
    """
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder(hp.embedding_size, hp.hidden_size)
        self.decoder = MelDecoder(hp.hidden_size)
        
    def forward(self, characters, mel_input, position_of_text, position_of_mel):
        memory, character_mask, attentions_encoder = self.encoder.forward(characters, pos=position_of_text)
        mel_output, postnet_output, attention_probs, stop_preds, attentions_decoder = self.decoder.forward(memory, mel_input, character_mask, pos= position_of_mel)
        
        return mel_output, postnet_output, attention_probs, stop_preds, attentions_encoder, attentions_decoder
    

class ModelPostNet(nn.Module):
    """
    CBHG Network (mel --> linear)
    Convolutional Bank, Highway network, and Gated Recurrent Unit
    Convolutional_Bank_Highway_network_and_Gated_Recurrent_Unit
    """
    def __init__(self):
        super(ModelPostNet, self).__init__()
        self.pre_projection = Conv(hp.n_mels, hp.hidden_size)
        self.cbhg = CBHG(hp.hidden_size)
        self.post_projection = Conv(hp.hidden_size, (hp.n_fft // 2) + 1)
        
    def forward(self, mel):
        mel = mel.transpose(1, 2)
        mel = self.pre_projection(mel)
        mel = self.cbhg(mel).transpose(1, 2)
        mag_pred = self.post_projection(mel).transpose(1, 2)
        
        return mag_pred

