## Реализация Transformer из простейших слоев

Используем только простейшие слои из `torch.nn`. Простейшие слои – это те, которые нельзя разбить на подслои. Например, `nn.Embedding`, `nn.Linear` и `nn.LayerNorm` разрешены, а `nn.TransformerEncoderLayer` и `nn.MultiheadAttention` запрещены. Также будем избегать использование циклов.

**Надо реализовать 10 частей:**
    
1. **Embedding + Position Encoding**. Соберем в 1 класс. Используем register_buffer()
2. **Multi-Head-Attention** универсальный, для использования с некоторым изменением маски и в блоке Decoder для Cross-Attention и Masked Attention.
3. **Feed Forward Network**
4. **EncoderBlock** - из них соберем блок энкодера. Так как в Transformer блок может использовать N раз
5. **Encoder**
6. **DecoderBlock** - из них соберем блок декодера. Так как в Transformer блок может использовать N раз
7. **Decoder**
8. Функция для **маски в cross-attention**
9. Функция для **создания треугольной маски в masked-attention** + учет паддингов
10. Сбор **Transformer** из блоков ранее

![Без_названия_negate (1)](https://github.com/user-attachments/assets/ce1e7edf-bc8b-4ea8-a25a-ba1565f16737)
