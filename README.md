# Cogs 181 Final Project Character-Level Language Model for Text Generation

This project implements a character-level recurrent neural network (RNN) language model inspired by Andrej Karpathy's [char-rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). The model is built using PyTorch and can be trained on any text corpus to generate new text in a similar style.

## Features

- Character-level language modeling using LSTM, GRU, or vanilla RNN architectures
- Customizable model parameters (hidden size, number of layers, dropout)
- Training with automatic validation and early stopping
- Text generation with temperature control and top-k sampling
- Multi-genre model training support
- Visualization of training and validation loss
- Comprehensive evaluation tools

## Project Structure

The project is organized into several modules:

- `TextDataset`: Handles data loading, preprocessing, and batch creation
- `CharRNN`: Implements the character-level RNN language model
- Training functions for model training and evaluation
- Text generation and sampling utilities
- Evaluation and analysis tools

## Installation

```bash
git clone https://github.com/yourusername/char-rnn-pytorch.git
cd char-rnn-pytorch
pip install -r requirements.txt
```

## Usage

### Training a Model

To train a new model:

```python
from char_rnn import train_model

model, data_loader = train_model(
    data_dir='path/to/your/text/data',        # Directory containing input.txt
    model_type='lstm',                         # 'lstm', 'gru', or 'rnn'
    hidden_size=256,                           # Size of the hidden layer
    num_layers=2,                              # Number of RNN layers
    dropout=0.3,                               # Dropout probability
    seq_length=100,                            # Sequence length for training
    batch_size=32,                             # Mini-batch size
    max_epochs=30,                             # Maximum number of epochs
    learning_rate=0.002,                       # Initial learning rate
    checkpoint_dir='checkpoints'               # Directory to save model checkpoints
)
```

You can also use the `train_with_config` helper function for more convenient training:

```python
model, data_loader = train_with_config(
    data_dir='data/science_fiction',
    model_name='sci_fi_model',
    custom_save_name='SciFi_Adventure_Model',
    hidden_size=384,
    num_layers=2,
    dropout=0.3,
    model_type='lstm'
)
```

### Generating Text

To generate text from a trained model:

```python
from char_rnn import sample_text

generated_text = sample_text(
    model_path='checkpoints/your_model.pt',    # Path to the model checkpoint
    prime_text='Once upon a time',             # Seed text to start generation
    length=500,                                # Number of characters to generate
    temperature=0.7,                           # Sampling temperature (higher = more random)
    topk=5                                     # Use top-k sampling (0 to disable)
)

print(generated_text)
```

### Multi-Genre Training

The project supports training models on different text genres and even mixing them:

```python
# Train a sci-fi themed model
scifi_model = train_with_config(
    data_dir='data/science_fiction',
    model_name='scifi',
    hidden_size=384,
    seq_length=100
)

# Train a romance themed model
romance_model = train_with_config(
    data_dir='data/romance',
    model_name='romance',
    hidden_size=256,
    dropout=0.2
)

# Train a hybrid model on mixed data
hybrid_model = train_with_config(
    data_dir='data/mixed',
    model_name='hybrid',
    hidden_size=512,
    num_layers=3,
    dropout=0.4
)
```

## Examples

### Generated Text Samples

#### Science Fiction Theme
```
The starship's engines hummed as it approached the nebula, and
the solution of the seas of the whole waters and the
mountains of their comforts of a
few miles an immense stranger. Then I would be a little control of this partitions of the mouth of
a perfect and my companions, and that there was no
complex to the steam of the cases the red whom I
had seen there in the most distance, that all this was so concurred to the strangest of the same thing in
their creatures were being surprised in the crowd of the
polite with the most disappearance of a country...
```

#### Romance Theme
```
Her heart raced as she opened the letter, never expecting a place they can be different. But he could not tell you that I have seen his comfortable second place and having a moment that she would not be safely been a sort of anxious state of hours.
But the same sort of mind was a moment and death of its particular than
anything too long as her heart and another, the more of the pleasure to his conversation, and the miles of a painful, and that their presence and character that a momentary arrival, when to be
at all, with the words and sitting ill and a...
```

#### Hybrid Theme
```
The android felt something strange in its circuits whenever she smiled, when
the classes on which succeeded to see the strength of the
park, because it was so strong, that the clear particulars of the
servants had been able to be concealed by a difference of their
favour of her presence. She had a persuasion to the world, which she
conscious to hear that he had been so far away to an argument to be the matter
to him, which she came out that he was able to suppose it she had been
so strong and soon as to be so far from his complaints...
```

## Evaluation

The models are evaluated on several metrics:

- **Perplexity**: Measures how well the model predicts the test data
- **BLEU Score**: Compares model-generated text to reference text
- **Statistical Analysis**: Word counts, sentence structure, vocabulary richness
- **Genre-Specific Keywords**: Analysis of genre characteristics in generated text

## Temperature Control

The temperature parameter controls the randomness in text generation:

- Low temperature (0.5): More predictable, conservative text
- Medium temperature (0.8): Balanced creativity and coherence
- High temperature (1.1+): More creative but potentially less coherent

## Project Insights

- Character-level models learn writing style, vocabulary, and grammatical patterns effectively
- LSTM architectures generally outperform GRU and vanilla RNN for this task
- Temperature tuning is crucial for balancing creativity and coherence in generated text
- Multi-genre training creates models that can blend characteristics of different genres

## References

- [Andrej Karpathy's original char-rnn post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
