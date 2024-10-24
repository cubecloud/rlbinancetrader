import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator

# Define the fields
TEXT = Field(tokenize='basic_english', lower=True)
LABEL = Field(sequential=False, use_vocab=True)

# Load the IMDB dataset
train_data, test_data = IMDB.splits(TEXT, LABEL, root='./data', train='train', test='test')
print(f"Training dataset type: {type(train_data)}")
print(f"Testing dataset type: {type(test_data)}")

# Build the vocabulary for the TEXT and LABEL fields
TEXT.build_vocab(train_data)
LABEL.build_vocab(train_data)

# Create the data loaders
train_loader = BucketIterator(train_data, batch_size=32, shuffle=True, sort_within_batch=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
test_loader = BucketIterator(test_data, batch_size=32, shuffle=False, sort_within_batch=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Print the length of the dataset
print(f"Training dataset length: {len(train_data)}")
print(f"Testing dataset length: {len(test_data)}")

# Print the batch size
print(f"Batch size: {train_loader.batch_size}")

# Define the LSTMExtractorNN class
class LSTMExtractorNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMExtractorNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.float()  # Convert input to float32
        h0 = torch.zeros(self.hidden_dim, x.size(0)).to(x.device)
        c0 = torch.zeros(self.hidden_dim, x.size(0)).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model, loss function, and optimizer
model = LSTMExtractorNN(input_dim=TEXT.vocab.vectors.shape[1], hidden_dim=256, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in train_loader:
        text, labels = batch
        print(type(text))
        print(type(labels))
        optimizer.zero_grad()
        outputs = model(text)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

# Evaluate the model
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for batch in test_loader:
        text, labels = batch.text, batch.label

        outputs = model(text)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

accuracy = correct / len(test_data)
print(f'Test Loss: {test_loss / len(test_data)}')
print(f'Test Accuracy: {accuracy:.2f}%')
