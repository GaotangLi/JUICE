from typing import List 
from collections import defaultdict

from torcheval.metrics.functional.text import perplexity
import torch 


class AverageMeter():
    def __init__(self, acc_keys: List[str]=None, confidence_keys: List[str]=None, parametric_keys: List[str]=None,
                 entro_per_keys: List[str]=None) -> None:

        """ 
        The average meter consists of accuracy and confidence and parametric part 

        data is a dictionary of dictionary
        """
        
        self.data = {
            "Accuracy": defaultdict(int),
            "Confidence": defaultdict(list), 
            "Parametric": defaultdict(list),
            "Entropy": defaultdict(list),
            "Perplexity": defaultdict(int),
        }

        self.ppl_aux = {
            "ppl_logit": defaultdict(list),
            "ppl_target": defaultdict(list)
        }


        # Initialization
        if acc_keys is not None:
            for k in acc_keys:
                self.data["Accuracy"][k] = 0 
        if confidence_keys is not None:
            for k in confidence_keys:
                self.data["Confidence"][k] = []
        if parametric_keys is not None:
            for k in parametric_keys:
                self.data["Parametric"][k] = []
        if entro_per_keys is not None:
            for k in entro_per_keys:
                self.data["Entropy"][k] = []
                # self.data["Perplexity"][k] = []
        self.entro_per_keys = entro_per_keys

    def update_Entropy(self, input_key, value):
        """ 
        Update entropy
        """
        self.data["Entropy"][input_key].append(value)
    
    def update_Perplexity(self, input_key, value):
        """ 
        Update perplexity
        """
        self.data["Perplexity"][input_key].append(value)

    def update_Accuracy(self, input_key, value):
        """
        Update the accuracy
        """
        self.data["Accuracy"][input_key]+=value 
        

    def update_Confidence(self, input_key, value):
        """ 
        Update the confidence
        """
        self.data["Confidence"][input_key].append(value)


    def update_Parametric(self, input_key, value):
        """ 
        Update the parametric 
        """

        self.data["Parametric"][input_key].append(value)
    
    def update_Perplexity_vector(self, input_key, ppl_logit_vec, ppl_target_vec):
        """ 
        Update perplexity
        """
        self.ppl_aux["ppl_logit"][input_key].append(ppl_logit_vec)
        self.ppl_aux["ppl_target"][input_key].append(ppl_target_vec)

    def get_ppl_value(self):
        for input_key in self.entro_per_keys:
            logits = self.ppl_aux["ppl_logit"][input_key] 
            target = self.ppl_aux["ppl_target"][input_key] 

            logit_vector = torch.cat(logits, dim=0)
            target_vector = torch.cat(target, dim=0)

            ppl_res = perplexity(input=logit_vector.unsqueeze(dim=1), target=target_vector)
            self.data["Perplexity"][input_key] = ppl_res.item()

    def get_data(self):
        return self.data 