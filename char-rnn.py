import argparse
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Data Preprocessing and Loading
class CharDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
    def __len__(self):
        return len(self.text) - self.seq_length
        
    def __getitem__(self, idx):
        # Get input sequence and target sequence
        x = self.text[idx:idx + self.seq_length]
        y = self.text[idx + 1:idx + self.seq_length + 1]
        
        # Convert to indices
        x = torch.tensor([self.char_to_idx[ch] for ch in x], dtype=torch.long)
        y = torch.tensor([self.char_to_idx[ch] for ch in y], dtype=torch.long)
        
        return x, y

class TextLoader:
    def __init__(self, data_dir, batch_size, seq_length, split_fractions):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.data_dir = data_dir
        
        # Split fractions for train/val/test
        self.train_frac, self.val_frac, self.test_frac = split_fractions
        
        # File paths
        input_file = os.path.join(data_dir, 'input.txt')
        vocab_file = os.path.join(data_dir, 'vocab.pkl')
        tensor_file = os.path.join(data_dir, 'data.pt')
        
        # Check if we need to preprocess
        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print(f"Preprocessing {input_file}...")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("Loading preprocessed files...")
            self.load_preprocessed(vocab_file, tensor_file)
            
        # Create datasets and dataloaders
        self.create_datasets()
        
    def preprocess(self, input_file, vocab_file, tensor_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        # Create char mappings
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Save vocabulary
        import pickle
        with open(vocab_file, 'wb') as f:
            pickle.dump((self.chars, self.char_to_idx, self.idx_to_char), f)
            
        # Save data tensor (just the text in this implementation)
        torch.save(self.text, tensor_file)
        print(f"Preprocessed data: {len(self.text)} characters, {self.vocab_size} unique.")
        
    def load_preprocessed(self, vocab_file, tensor_file):
        import pickle
        with open(vocab_file, 'rb') as f:
            self.chars, self.char_to_idx, self.idx_to_char = pickle.load(f)
        self.vocab_size = len(self.chars)
        
        self.text = torch.load(tensor_file)
        print(f"Loaded data: {len(self.text)} characters, {self.vocab_size} unique.")
        
    def create_datasets(self):
        # Create a single dataset
        full_dataset = CharDataset(self.text, self.seq_length)
        
        # Calculate split sizes
        data_size = len(full_dataset)
        train_size = int(data_size * self.train_frac)
        val_size = int(data_size * self.val_frac)
        test_size = data_size - train_size - val_size
        
        # Create train/val/test datasets
        from torch.utils.data import random_split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        print(f"Data split: train={train_size}, val={val_size}, test={test_size}")

# Model Implementations
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0, model_type='lstm'):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        
        # Character embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # RNN Layer
        if model_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        elif model_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        else:  # vanilla RNN
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
            
        # Output layer
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, hidden=None):
        # Input shape: [batch, seq_len]
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)
            
        # Embedding: [batch, seq_len, hidden_size]
        embedded = self.embedding(x)
        
        # RNN forward pass
        output, hidden = self.rnn(embedded, hidden)
        
        # Output shape: [batch, seq_len, hidden_size]
        # Reshape for linear layer: [batch * seq_len, hidden_size]
        output = output.contiguous().view(-1, self.hidden_size)
        
        # Linear layer to get logits for each character
        output = self.fc(output)
        
        # Reshape back: [batch, seq_len, vocab_size]
        return output, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.model_type == 'lstm':
            return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                    weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        else:
            return weight.new(self.num_layers, batch_size, self.hidden_size).zero_()

# Training functions
def train(model, dataloader, criterion, optimizer, device, grad_clip=5):
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = None
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Move to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output, hidden = model(inputs, hidden)
        
        # Detach hidden state to prevent backprop through entire history
        if isinstance(hidden, tuple):
            hidden = tuple([h.detach() for h in hidden])
        else:
            hidden = hidden.detach()
            
        # Calculate loss
        loss = criterion(output, targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Update parameters
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f'Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s')
            start_time = time.time()
            
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    hidden = None
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            output, hidden = model(inputs, hidden)
            
            # Calculate loss
            loss = criterion(output, targets.view(-1))
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def sample(model, chars, char_to_idx, idx_to_char, device, seed_text='', max_length=2000, temperature=1.0):
    model.eval()
    
    # If no seed text is provided, start with a random character
    if not seed_text:
        seed_text = idx_to_char[np.random.randint(0, len(chars))]
        
    # Convert seed text to indices
    input_seq = [char_to_idx[c] for c in seed_text]
    input_tensor = torch.LongTensor([input_seq]).to(device)
    
    generated_text = seed_text
    hidden = None
    
    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_tensor, hidden)
            
            # Get the last character prediction
            output = output[-1].view(-1)
            
            # Apply temperature
            if temperature != 1.0:
                output = output / temperature
                
            # Apply softmax to get probabilities
            probabilities = nn.functional.softmax(output, dim=0)
            
            # Sample from the distribution
            next_index = torch.multinomial(probabilities, 1).item()
            
            # Add the character to the generated text
            next_char = idx_to_char[next_index]
            generated_text += next_char
            
            # Update input tensor for next step
            input_tensor = torch.LongTensor([[next_index]]).to(device)
    
    return generated_text

# Main training script
def main():
    parser = argparse.ArgumentParser(description='PyTorch char-rnn implementation')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                      help='data directory containing input.txt')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                      help='sequence length')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='lstm',
                      help='rnn, lstm, or gru')
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='number of layers in the RNN')
    parser.add_argument('--dropout', type=float, default=0.0,
                      help='dropout probability')
    
    # Optimization parameters
    parser.add_argument('--learning_rate', type=float, default=0.002,
                      help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97,
                      help='learning rate decay rate')
    parser.add_argument('--lr_decay_after', type=int, default=10,
                      help='start learning rate decay after this epoch')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                      help='clip gradients at this value')
    parser.add_argument('--epochs', type=int, default=50,
                      help='number of epochs to train')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='directory to save checkpoints')
    parser.add_argument('--init_from', type=str, default='',
                      help='initialize network parameters from checkpoint')
    
    # Sampling parameters
    parser.add_argument('--sample', action='store_true',
                      help='sample from the model')
    parser.add_argument('--prime_text', type=str, default='',
                      help='text to prime generation with')
    parser.add_argument('--length', type=int, default=2000,
                      help='number of characters to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                      help='sampling temperature (higher = more diverse)')
    
    # Misc
    parser.add_argument('--seed', type=int, default=123,
                      help='random seed')
    parser.add_argument('--cuda', action='store_true',
                      help='use CUDA')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory if needed
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    # Create data loader
    loader = TextLoader(
        args.data_dir, 
        args.batch_size, 
        args.seq_length, 
        [0.9, 0.05, 0.05]  # Default train/val/test split
    )
    
    # Create model
    model = CharRNN(
        input_size=loader.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model
    ).to(device)
    
    # Initialize from checkpoint if specified
    if args.init_from:
        print(f"Loading model from {args.init_from}")
        checkpoint = torch.load(args.init_from, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
    
    # If sampling, generate text and exit
    if args.sample:
        generated_text = sample(
            model, 
            loader.chars, 
            loader.char_to_idx, 
            loader.idx_to_char, 
            device,
            seed_text=args.prime_text,
            max_length=args.length,
            temperature=args.temperature
        )
        print("Generated text:")
        print(generated_text)
        return
    
    # Set up loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.lr_decay_after, 
        gamma=args.lr_decay
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train(model, loader.train_loader, criterion, optimizer, device, args.grad_clip)
        
        # Evaluate on validation set
        val_loss = evaluate(model, loader.val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f"{args.model}_epoch{epoch+1:02d}_{val_loss:.4f}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Print progress
        elapsed = time.time() - start_time
        print('-' * 80)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.2f}s")
        print('-' * 80)
        
        # Generate a sample after each epoch
        sample_text = sample(
            model, 
            loader.chars, 
            loader.char_to_idx, 
            loader.idx_to_char, 
            device,
            seed_text=args.prime_text if args.prime_text else "The ",
            max_length=200,
            temperature=args.temperature
        )
        print(f"Sample after epoch {epoch+1}:")
        print(sample_text)
        print('-' * 80)
    
    # Final evaluation on test set
    test_loss = evaluate(model, loader.test_loader, criterion, device)
    print(f"Final test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()