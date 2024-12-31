import os
import pdb
import time
import torch
import ctcdecode
import numpy as np
from itertools import groupby
import torch.nn.functional as F


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(vocab, beam_width=10, blank_id=blank_id,
                                                    num_processes=10)

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        
        ret_list = []
        probabilities = []  # List to store the probabilities
        beam_confidences = []  # To store the beam scores (confidences for the chosen sequence)
        
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]

            # Remove repeated consecutive elements (this is a standard operation in CTC decoding)
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            
            # Convert the result to class names using i2g_dict
            decoded_result = [(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in enumerate(first_result)]

            # Now map probabilities for each class in the decoded result
            class_probabilities = []
            for idx, gloss_id in enumerate(first_result):
                # Get the probability (confidence) for the class at this timestep
                # gloss_id corresponds to the class index; get its probability from the softmax output
                prob = nn_output[batch_idx, idx, gloss_id.item()].item()  # Get probability for the class
                class_probabilities.append(prob)

            ret_list.append(decoded_result)
            probabilities.append(class_probabilities)

            #print(f"\nBEAMMMTODO = {beam_scores[batch_idx]}\nBEAMMM 0 = {beam_scores[batch_idx][0]}\nBEAMMM -1 = {beam_scores[batch_idx][-1]}")

            # Extract the beam score for the chosen sequence (the first beam)
            beam_score = beam_scores[batch_idx][0].item()  # The score for the first beam in the batch
            beam_confidences.append(1/np.exp(beam_score))  # Convert log score to regular probability


        return ret_list, probabilities, beam_confidences

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list
