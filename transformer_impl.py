import torch
from torch import nn, Tensor
import numpy as np

class Embedding(nn.Module):
    """
    Переводит токены в эмбеддинги Трансформера,
    суммируя эмбеддинги токенов и их позиций
    """

    def __init__(self, vocab_size, hidden_size, max_len, drop_prob=0.1):
        """
        vocab_size: размер словаря
        hidden_size: размер скрытого слоя
        max_len: максимальная возможная длина текста
        drop_prob: вероятность удаления нейрона в dropout
        """ 
        super().__init__()
        # sequence_position (1, max_len)
        self.register_buffer("sequence_position", torch.arange(max_len).expand((1, -1)), persistent=False
        )
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=hidden_size)
        self.position_embedding = nn.Embedding(num_embeddings=max_len,
                                               embedding_dim=hidden_size)
        
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size)
        self.dropout = nn.Dropout(drop_prob)
        

    def forward(self, input_ids) -> Tensor:
        """
        input_ids: Tensor[bs, seq_len] – индексы токенов текста
        """
        assert input_ids.ndim == 2

        embedding = self.embedding(input_ids)
        position_length = input_ids.shape[1]
        position_embedding = self.position_embedding(self.sequence_position[:, :position_length])
        
        embedding += position_embedding
        
        layer_norm_embedding = self.layer_norm(embedding)
        
        
        return self.dropout(layer_norm_embedding)


class MultiHeadAttention(nn.Module):
    """
    Реализует Multi-Head Self-Attention слой Трансформера.
    """
    def __init__(self, hidden_size, n_head, drop_prob=0.1):
        """
        hidden_size: размер скрытого слоя
        n_head: число голов внимания
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()

        assert hidden_size % n_head == 0

        self.hidden_size = hidden_size
        self.d_h = np.sqrt(hidden_size // n_head)
        self.n_head = n_head
        
        # используется по view при разбиании на головы
        self.H_division_nhead = hidden_size // n_head
        
        # все 4 матрицы имеют размерность (H, H)
        self.query = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.key = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.value = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        # выходной слой для применения к конкатенированным головам
        self.out = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, q, k, v, attention_mask=None) -> Tensor:
        """
        q, k, v: Tensor[bs, seq_len, hidden_size] – входные аргументы для соответствующих линейных слоев
        attention_mask: Tensor[bs, 1, 1 or seq_len, seq_len] – маска внимания, содержащая значения 0 и -inf;
                                                               добавляется к скорам механизма внимания (до softmax)
        """
        assert q.ndim == k.ndim == v.ndim == 3
        assert attention_mask.ndim == 4
        
        # q, k, v: (N, L, H)
        # 1. Cчитаем матрицы K, Q и V и разделяем их на головы.
        
        bs = q.size(0)
        
        # Размерность каждой матрицы: [bs, seq_len, n_head, head_dim]
        # но также надо учесть transpose для n_head и seq_len, чтобы каждая голова была независима
        # по итогу [bs, n_head, seq_len, head_dim]
        q_heads = self.query(q).view(bs, -1, self.n_head, self.H_division_nhead).transpose(1, 2)
        k_heads = self.key(k).view(bs, -1, self.n_head, self.H_division_nhead).transpose(1, 2)
        v_heads = self.value(v).view(bs, -1, self.n_head, self.H_division_nhead).transpose(1, 2)

        # 2. Считаем attention_scores: Q * K^T / sqrt{head_dim}
        # Размерность результата: [bs, n_head, seq_len, seq_len]
        
        # matmul автоматически работает по последним двум измерениям тензоров
        attention_heads_scores = torch.matmul(q_heads, k_heads.transpose(2, 3)) / self.d_h
        
        # 3. Добавляем attention_mask к полученным скорам, чтобы занулить те из них, на которые нельзя смотреть
        
        
        if attention_mask is not None:
            attention_heads_scores += attention_mask
        
        # 4. Считаем attention_probs: softmax(attention_scores)
        # Softmax применяем к последней размерности
        
        attention_heads_probs = torch.nn.functional.softmax(attention_heads_scores, dim=-1)   
        
        
        # 5. Добавляем dropout к полученным вероятностям
        attention_heads_probs = self.dropout(attention_heads_probs)
        
        # 6. Считаем выход внимания: attention_heads_probs * V
        # attention_heads_probs: [bs, n_head, seq_len, seq_len]
        # V:               [bs, n_head, seq_len, head_dim]
        # Размерность результата: [bs, n_head, seq_len, head_dim]
        
        attention_heads = torch.matmul(attention_heads_probs, v_heads)
        

        # 7. Конкатенируем обратно векторы всех голов, получаем размерность [bs, seq_len, hidden_size]
        # это можно сделать снова через view, предварительно переставив размерность головы в изначальное место
        # и применив contiguousд для создания копии тензора для корректной работы view
        
        attention = attention_heads.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.hidden_size)
        output = self.out(attention)
         
        # 8. Применяем последний линейный слой
        # Размерность результата: [bs, seq_len, hidden_size]

        return output


class FeedForward(nn.Module):
    """
    Реализует Feed Forward Network слой Трансформера c skip-connection и нормализацией.
    """
    def __init__(self, hidden_size, intermediate_size, drop_prob=0.1):
        """
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()

        self.U_layer = nn.Linear(in_features=hidden_size, out_features=intermediate_size)
        self.V_layer = nn.Linear(in_features=intermediate_size, out_features=hidden_size)
        
        self.dropout = nn.Dropout(drop_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.activation = nn.ReLU()
        

    def forward(self, hidden_states) -> Tensor:
        """
        hidden_states: Tensor[bs, seq_len, hidden_size] – входное представление текста
        """
        assert hidden_states.ndim == 3

        residual = hidden_states
        
        U_output = self.activation(self.U_layer(hidden_states))
        V_output = self.V_layer(U_output)
        
        dropout_output = self.dropout(V_output)
        
        skip_connection = residual + dropout_output
        
        return self.layer_norm(skip_connection)
        

class EncoderBlock(nn.Module):
    """
    Реализует блок Encoder'a.
    """
    def __init__(self, hidden_size, intermediate_size, n_head, drop_prob=0.1):
        """
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        n_head: число голов внимания
        """
        super().__init__()

        self.FFN = FeedForward(hidden_size, intermediate_size, drop_prob)
        self.MHA = MultiHeadAttention(hidden_size, n_head, drop_prob)
        
        self.dropout = nn.Dropout(drop_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask) -> Tensor:
        """
        hidden_states: Tensor[bs, seq_len, hidden_size] – входное представление текста
        attention_mask: Tensor[bs, 1, 1, seq_len] – маска внимания, содержащая значения 0 и -inf
        """
        assert hidden_states.ndim == 3
        assert attention_mask.ndim == 4
        
        residuals = hidden_states
        
        # расчеты для обработки multi-head-attention с учетом dropout, skip-connection и layer_norm
        mha_block = self.MHA.forward(q=hidden_states,
                                     k=hidden_states,
                                     v=hidden_states,
                                     attention_mask=attention_mask)
        
        hidden_states = self.dropout(mha_block)
        hidden_states = self.layer_norm(hidden_states + residuals)
        
        # расчеты для Feed Forward Network
        feed_forward_network = self.FFN(hidden_states)
        
        return feed_forward_network
        

class Encoder(nn.Module):
    """
    Encoder Трансформера.
    """
    def __init__(self, vocab_size, max_len, hidden_size,
                 intermediate_size, n_head, n_layers, drop_prob=0.1):
        """
        vocab_size: размер словаря
        max_len: максимальная возможная длина текста
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        n_head: число голов внимания
        n_layers: число блоков Encoder
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()
        
        self.n_layers = n_layers
        
        self.Embedding = Embedding(vocab_size, hidden_size, max_len, drop_prob)
        
        self.Encoder_block_list = nn.ModuleList([
                                                EncoderBlock(
                                                    hidden_size,
                                                    intermediate_size,
                                                    n_head, drop_prob)
                                                for _ in range(n_layers)])

    def forward(self, input_ids, attention_mask=None) -> Tensor:
        """
        input_ids: Tensor[bs, seq_len] – индексы токенов текста
        attention_mask: Tensor[bs, 1, 1, seq_len] – маска внимания, содержащая значения 0 и -inf
        """
        assert input_ids.ndim == 2
        assert attention_mask.ndim == 4
        
        hidden_states = self.Embedding.forward(input_ids)
        for i in range(self.n_layers):
            hidden_states = self.Encoder_block_list[i].forward(hidden_states, attention_mask)
        return hidden_states


class DecoderBlock(nn.Module):
    """
    Реализует блок Decoder'a.
    """
    def __init__(self, hidden_size, intermediate_size, n_head, drop_prob=0.1):
        """
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        n_head: число голов внимания
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()

        self.FFN = FeedForward(hidden_size, intermediate_size, drop_prob)
        
        self.masked_MHA = MultiHeadAttention(hidden_size, n_head, drop_prob)
        self.cross_MHA = MultiHeadAttention(hidden_size, n_head, drop_prob)
        
        self.dropout = nn.Dropout(drop_prob)
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)



    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask) -> Tensor:
        """
        hidden_states: Tensor[bs, trg_seq_len, hidden_size] – входное представление целевого текста
        attention_mask: Tensor[bs, 1, trg_seq_len, trg_seq_len] – маска внимания Decoder'a
        encoder_hidden_states: Tensor[bs, src_seq_len, hidden_size] – выход последнего слоя Encoder
        encoder_attention_mask: Tensor[bs, 1, 1, src_seq_len] – маска внимания Encoder'a
        """
        assert hidden_states.ndim == encoder_hidden_states.ndim == 3
        assert attention_mask.ndim == encoder_attention_mask.ndim == 4
        


        residual = hidden_states
        
        masked_mha_output = self.masked_MHA.forward(q=hidden_states,
                                                    k=hidden_states,
                                                    v=hidden_states,
                                                    attention_mask=attention_mask)
        
        hidden_states = self.dropout(masked_mha_output)
        
        hidden_states = self.layer_norm_1(hidden_states + residual)
        
        # испольузем текущее состояние, чтобы прокинуть дальше
        residual = hidden_states
        
        cross_mha = self.cross_MHA.forward(q=hidden_states,
                                           k=encoder_hidden_states,
                                           v=encoder_hidden_states,
                                           attention_mask=encoder_attention_mask)
        
        hidden_states = self.dropout(cross_mha)
        
        hidden_states = self.layer_norm_2(hidden_states + residual)
        
        feed_forward_output = self.FFN.forward(hidden_states)
        
        return feed_forward_output
        
        

class Decoder(nn.Module):
    """
    Decoder Трансформера.
    """
    def __init__(self, vocab_size, max_len, hidden_size,
                 intermediate_size, n_head, n_layers, drop_prob=0.1):
        """
        vocab_size: размер словаря
        max_len: максимальная возможная длина текста
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        n_head: число голов внимания
        n_layers: число блоков Decoder
        """
        super().__init__()
        
        self.n_layers = n_layers

        self.embedding = Embedding(vocab_size, hidden_size, max_len, drop_prob)

        self.decoder_block_list = nn.ModuleList([DecoderBlock(hidden_size,
                                                              intermediate_size,
                                                              n_head,
                                                              drop_prob)
                                                 for _ in range(n_layers)])
        
        self.output_logits = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask) -> Tensor:
        """
        input_ids: Tensor[bs, seq_len] – индексы токенов текста
        attention_mask: Tensor[bs, 1, trg_seq_len, trg_seq_len] – маска внимания Decoder'a
        encoder_hidden_states: Tensor[bs, src_seq_len, hidden_size] – выход последнего слоя Encoder
        encoder_attention_mask: Tensor[bs, 1, 1, src_seq_len] – маска внимания Encoder'a
        """
        assert input_ids.ndim == 2
        assert encoder_hidden_states.ndim == 3
        assert attention_mask.ndim == encoder_attention_mask.ndim == 4

        hidden_states = self.embedding(input_ids)
        
        for i in range(self.n_layers):
            hidden_states = self.decoder_block_list[i].forward(hidden_states,
                                                               attention_mask,
                                                               encoder_hidden_states,
                                                               encoder_attention_mask)
        
        logits = self.output_logits(hidden_states)
        
        return logits
        
# attention_mask: [N, seq_len]
def get_extended_attention_mask(attention_mask, dtype=torch.float):
    N = attention_mask.size(0)
    seq_len = attention_mask.size(1)
    
    extended_attention_mask = attention_mask.view(N, 1, 1, seq_len).to(dtype=dtype)
    
    # заменяем 1 на 0 и 0 на -inf
    
    return (extended_attention_mask != 1).to(dtype) * torch.finfo(dtype).min


# def get_causal_extended_attention_mask(attention_mask, dtype=torch.float):
#     N = attention_mask.size(0)
#     seq_len = attention_mask.size(1)
#     tensor_ones = torch.ones(N, seq_len, seq_len, device=attention_mask.device).view(N, 1, seq_len, seq_len)
#     triu_matrix_ones = torch.triu(tensor_ones, diagonal=1)

#     triu_with_inf = torch.where(triu_matrix_ones == 1, torch.finfo(dtype).min, 0).to(attention_mask.device)

#     extended_attention_mask = attention_mask.view(N, 1, 1, seq_len).to(attention_mask.device)
#     extended_attention_mask = torch.where(extended_attention_mask == 1, torch.finfo(dtype).min, 0).to(attention_mask.device)

#     return triu_with_inf + extended_attention_mask




def get_causal_extended_attention_mask(attention_mask, dtype=torch.float):    
    device = attention_mask.device
    batch_size, seq_len = attention_mask.shape

    seq_ids = torch.arange(seq_len, device=device)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_len, 1) <= seq_ids[None, :, None]    
    causal_mask = causal_mask.to(attention_mask.dtype)

    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :].to(dtype=dtype)
    
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, hidden_size, n_head,
                 intermediate_size, encoder_max_len, decoder_max_len, n_layers, drop_prob=0.1):
        super().__init__()
        """
        Все параметры означают то же самое, что и в предыдущих классах
        """

        self.encoder = Encoder(encoder_vocab_size, encoder_max_len, hidden_size,
                               intermediate_size, n_head, n_layers, drop_prob)
        
        self.decoder = Decoder(decoder_vocab_size, decoder_max_len, hidden_size,
                               intermediate_size, n_head, n_layers, drop_prob)

    def forward(self, src_input_ids, trg_input_ids, src_attention_mask=None, trg_attention_mask=None) -> Tensor:
        """
        src_input_ids: Tensor[bs, src_seq_len] – индексы токенов входного текста
        trg_input_ids: Tensor[bs, trg_seq_len] – индексы токенов выходного текста
        src_attention_mask: Tensor[bs, scr_seq_len] – маска внимания входного текста
        trg_attention_mask: Tensor[bs, trg_seq_len] – маска внимания выходного текста
        """

        
        
        if src_attention_mask is None:
            src_attention_mask = torch.ones_like(src_input_ids)
        if trg_attention_mask is None:
            trg_attention_mask = torch.ones_like(trg_input_ids)       
        
#         print("Source Attention Mask Shape:", src_attention_mask.shape)
#         print("Target Attention Mask Shape:", trg_attention_mask.shape)

        encoder_output = self.encoder(src_input_ids,
                                     attention_mask=get_extended_attention_mask(src_attention_mask))

#         print("Encoder Output Shape:", encoder_output.shape)

        logits = self.decoder(trg_input_ids,
                              attention_mask=get_causal_extended_attention_mask(trg_attention_mask),
                              encoder_hidden_states=encoder_output,
                              encoder_attention_mask=get_extended_attention_mask(src_attention_mask))
        
#         print("Decoder Output Shape:", logits.shape)
        return logits