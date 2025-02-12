import torch
import torch.nn as nn
import urllib.request
import string
import re
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Fetch and preprocess text
def fetch_and_preprocess(url):
    response = urllib.request.urlopen(url)
    text = response.read().decode("utf-8").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def get_vocab(text, min_word_count):
    wcount = defaultdict(int)
    for word in text:
        wcount[word] += 1
    filtered_words = [word for word, count in wcount.items() if count >= min_word_count]
    return list(set(filtered_words))


class OneHotEncoder:
    def __init__(self, vocab):
        self.vocab = list(vocab)
        self.oh_index = {vocab[i]: i for i in range(len(vocab))}

    def encode(self, word):
        one_hot = torch.zeros(len(self.vocab))
        one_hot[self.oh_index[word]] = 1
        return one_hot

    def decode(self, one_hot):
        return self.vocab[torch.argmax(one_hot)]

    def __call__(self, word):
        return self.encode(word)


class ShakespeareDataset(Dataset):
    def __init__(self, text, encoder, context_size):
        self.text = text
        self.encoder = encoder
        self.context_size = context_size
        self.data = []
        for i in tqdm(range(context_size, len(text) - context_size)):
            skip = False

            target = text[i]
            if target not in encoder.oh_index:
                skip = True
                continue

            left = [text[i - j] for j in range(1, context_size + 1)]
            right = [text[i + j] for j in range(1, context_size + 1)]
            context = left + right
            for word in context:
                if word not in encoder.oh_index:
                    skip = True
                    break
            if skip:
                continue

            self.data.append((context, target))

    def __getitem__(self, idx):
        context, target = self.data[idx]
        one_hots = torch.stack([self.encoder(word) for word in context])
        return torch.sum(one_hots, dim=0), self.encoder.oh_index[target]

    def __len__(self):
        return len(self.data)


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, encoder):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dim
        self.context_size = context_size
        self.encoder = encoder
        self.embed = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.fc = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.softmax(self.fc(self.embed(x) / (2 * self.context_size)))

    def embed(self, word):
        return self.embed(self.encoder(word))

    def find_similar(self, word, k):
        scores = self.embed(word) @ self.W.T
        sorted_idx = [(i, scores[i]) for i in torch.argsort(scores, descending=True)]
        best = [self.encoder.decode(sorted_idx[i]) for i in range(k)]
        return best


def plot_loss(batch_losses, batch_indices):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(batch_indices, batch_losses)
    axs[0].set_xlabel("Batch Index")
    axs[0].set_label("Loss")
    axs[0].set_itle("Loss vs. Batch Index")

    axs[1].loglog(batch_indices, batch_losses)
    axs[1].set_xlabel("Batch Index")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Loss vs. Batch Index")

    plt.savefig("./cbow.png")


if __name__ == "__main__":
    # URL for Shakespeare's complete works from Project Gutenberg
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    raw_text = fetch_and_preprocess(url)

    min_word_count = 5
    CONTEXT_SIZE = 2
    batch_size = 2048

    encoder = OneHotEncoder(get_vocab(raw_text, min_word_count))
    train_data = ShakespeareDataset(raw_text, encoder, CONTEXT_SIZE)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )

    EMBEDDING_DIM = 512
    lr = 5e-3
    weight_decay = 0

    cbow = CBOW(len(encoder.vocab), EMBEDDING_DIM, CONTEXT_SIZE, encoder).to(device)
    optimizer = optim.Adam(cbow.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    num_epochs = 100
    batch_losses = []
    batch_indicies = []
    for epoch in range(num_epochs):
        cbow.train()
        tot_loss = 0
        for batch_idx, (context, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            context = context.to(device)
            output = cbow(context)
            target = target.to(device)
            loss = criterion(output, target)
            batch_indicies.append(len(batch_indicies))
            batch_losses.append(loss.item())
            tot_loss += loss.item()
            batch_indicies.append(batch_idx)
            loss.backward()
            optimizer.step()

        print(
            f"epoch {epoch} average loss for batch of {batch_size}: {tot_loss / len(train_dataloader)}"
        )

    plot_loss(batch_losses, batch_indicies)
