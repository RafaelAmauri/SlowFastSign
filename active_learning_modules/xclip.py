import os
import json
import torch

from tqdm import tqdm
from queue import PriorityQueue
from collections import OrderedDict

from active_learning_modules.xclip_utils import getKeyFramesFromFolder


def xclipInference(model, processor, videoPaths, glossPredictions, predictionsWorkDir, device):
    model.to(device)

    keyFrames         = []
    predictionRanking = PriorityQueue()

    loop = tqdm(zip(videoPaths, glossPredictions), total=len(glossPredictions), desc=f"Running inference for unlabeled set...")
    for vPath, predictedGloss in loop:
        # X-Clip can only handle 8 frames or 16 frames at a time, so we need to choose the 16 most significant frames :(
        keyFrames = getKeyFramesFromFolder(vPath, 16)

        # We give the model the chance to choose between the gloss that the other model generated or some random garbage.
        # If it chooses the random garbage with higher probability, it means that this could be a good candidate for labeling.
        possibleOptions = [predictedGloss, "_"]

        inputs = processor(text=possibleOptions, videos=keyFrames, return_tensors="pt", padding=True)
        inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted class probabilities
        probs = outputs.logits_per_video.softmax(dim=1)

        predictedLabel = possibleOptions[torch.argmax(probs)]
        predictionProbability = torch.max(probs)

        predictionRanking.put((vPath, predictionProbability))
    
    
    # Write to disk
    content = OrderedDict()
    while not predictionRanking.empty():
        path, confidence = predictionRanking.get()
        confidence = confidence.item()
        content[path] = confidence
    
    with open(os.path.join(predictionsWorkDir, "predictionRankings.json"), "w") as f:
        json.dump(content, f)

    return content


def trainXclip(model, processor, nEpochs, dataloader, device):
    optimizer  = torch.optim.AdamW(model.parameters(), lr=5e-6)

    model.to(device)
    model.train()

    loss_video = torch.nn.CrossEntropyLoss()
    loss_txt   = torch.nn.CrossEntropyLoss()

    for epoch in range(nEpochs):
        loop = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}")
        for batch in loop:
            # Get the inputs

            input_ids = batch['input_ids'].squeeze(1).to(model.device)
            pixel_values = batch['pixel_values'].squeeze(1).to(model.device)

            # Forward pass
            outputs = model(input_ids=input_ids, pixel_values=pixel_values)
            
            logits_per_video = outputs.logits_per_video  # Similarity score between video and text
            logits_per_text = outputs.logits_per_text

            ground_truth = torch.arange(len(batch['input_ids']), dtype=torch.long, device=device)
            loss = (loss_video(logits_per_video,ground_truth) + loss_txt(logits_per_text,ground_truth))/2


            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update tqdm loop
            loop.set_postfix(loss=loss.item())

    processor.save_pretrained("fine_tuned_xclip_processor")
    model.save_pretrained("fine_tuned_xclip_model")