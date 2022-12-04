import torch
from pytorch_metric_learning import losses

def info_nce_loss(features, config):
    '''Implementation of NCE loss'''
    # Make dummy labels
    labels = torch.cat([torch.arange(config.BATCH_SIZE) for i in range(config.VIEWS)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(config.DEVICE)
    
    # Normalize the feature and get similarity
    features = torch.nn.functional.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(config.DEVICE)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(config.DEVICE)
    logits = logits / config.TEMP
    
    return logits, labels