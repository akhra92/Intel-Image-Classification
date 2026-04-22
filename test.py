import torch

@torch.no_grad()
def test(x, y, model, criterion):
    pred = model(x)
    loss = criterion(pred, y)
    
    predicted = torch.argmax(pred, dim=1)
    acc = (predicted == y).float().mean()

    return acc.item(), loss.item()