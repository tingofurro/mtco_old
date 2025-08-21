from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class GenerationModel:
    def __init__(self, model_name="microsoft/phi-4", device=None, max_batch_size=2):
        # place_model_to_shm.py will place the model. Run it before so the first-ever loading is faster
        load_path = f"/dev/shm/{model_name}" if os.path.exists(f"/dev/shm/{model_name}") else model_name
        if device is None:
            self.device = "cuda"
            # Use auto device_map when no specific device is given
            self.model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.device = device
            # Load model without device_map first, then move to specific device
            self.model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.float16,
                device_map=None
            )
            # Move model to the specific device
            self.model = self.model.to(device)
            print(f"Model moved to device: {device}")

        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def construct_inputs(self, conversation):
        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        prompt += "<|im_start|>assistant<|im_sep|>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return inputs

    def generate_batch(self, conversation, n_responses=4, temperature=1.0, max_tokens=1000, filter_unique=True, **kwargs):
        # If n_responses is within max_batch_size, generate all at once
        if n_responses <= self.max_batch_size:
            return self._generate_single_batch(conversation, n_responses, temperature, max_tokens, filter_unique, **kwargs)
        
        # Otherwise, loop through batches
        all_responses = []
        remaining_responses = n_responses
        
        while remaining_responses > 0:
            current_batch_size = min(remaining_responses, self.max_batch_size)
            batch_responses = self._generate_single_batch(conversation, current_batch_size, temperature, max_tokens, filter_unique=False, **kwargs)
            all_responses.extend(batch_responses)
            remaining_responses -= current_batch_size
        
        # Apply unique filtering at the end if requested
        if filter_unique:
            unique_responses = []
            unique_responses_set = set()
            for response in all_responses:
                if response["response_text"] not in unique_responses_set:
                    unique_responses.append(response)
                    unique_responses_set.add(response["response_text"])
            all_responses = unique_responses
        
        return all_responses

    def _generate_single_batch(self, conversation, n_responses, temperature, max_tokens, filter_unique, **kwargs):
        inputs = self.construct_inputs(conversation)
    
        with torch.no_grad():
            outputs = self.model.generate(**inputs, do_sample=True, temperature=temperature, max_new_tokens=max_tokens, num_return_sequences=n_responses, pad_token_id=self.tokenizer.pad_token_id, output_scores=True, return_dict_in_generate=True, **kwargs)

        output_sequences = outputs.sequences[:, inputs["input_ids"].shape[1]:]
        output_sequences = output_sequences.cpu().tolist()

        logits = torch.stack(outputs.scores, dim=0).transpose(0, 1)
        token_logprobs = torch.log_softmax(logits, dim=-1)

        end_token_id = self.tokenizer.eos_token_id
        responses = []
        for i, output in enumerate(output_sequences):
            response_tokens = output
            if end_token_id in response_tokens:
                response_tokens = response_tokens[:(response_tokens.index(end_token_id)+1)] # Keep the end token

            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            # Get logits for this sequence
            selected_logprobs = token_logprobs[i, torch.arange(len(response_tokens)), response_tokens]
            response_logprobs = torch.sum(selected_logprobs)
            responses.append({
                "response_text": response_text,
                "response_tokens": response_tokens,
                "num_tokens": len(response_tokens),
                "logprobs": response_logprobs.item()
            })

        if filter_unique:
            unique_responses = []
            unique_responses_set = set()
            for response in responses:
                if response["response_text"] not in unique_responses_set:
                    unique_responses.append(response)
                    unique_responses_set.add(response["response_text"])
            responses = unique_responses

        return responses

    def get_logprobs(self, conversation, responses, use_grad=True, reduction="sum"):
        end_token_ids = self.tokenizer.eos_token_id
        # First process inputs without gradients
        inputs = self.construct_inputs(conversation)
        input_ids = inputs["input_ids"].repeat(len(responses), 1)

        last_input_token_ids = input_ids[:, -1]
        input_ids = input_ids[:, :-1]

        # never need grads on the input
        with torch.no_grad():
            input_outputs = self.model(input_ids, return_dict=True)

        output_token_ids = [response["response_tokens"] for response in responses]
        max_length = max([len(response) for response in output_token_ids])
        output_token_ids = [response + [end_token_ids] * (max_length - len(response)) for response in output_token_ids]
        output_token_ids = torch.tensor(output_token_ids).to(self.device)

        shifted_output_token_ids = torch.cat([last_input_token_ids.unsqueeze(1), output_token_ids], dim=1)

        output_mask_ids = [([1] * (len(response["response_tokens"]))) + [0] * (max_length - len(response["response_tokens"])) for response in responses]
        output_mask_ids = torch.tensor(output_mask_ids).to(self.device)

        with torch.set_grad_enabled(use_grad):
            output_outputs = self.model(shifted_output_token_ids, past_key_values=input_outputs.past_key_values, return_dict=True)

        output_logits = output_outputs.logits
        token_logprobs = torch.log_softmax(output_logits, dim=-1)

        seq_length = output_token_ids.shape[1]
        response_logprobs = []
        for i in range(len(responses)):
            this_selected_logprobs = token_logprobs[i, torch.arange(seq_length), output_token_ids[i]] * output_mask_ids[i]
            if reduction == "sum":
                response_logprobs.append(torch.sum(this_selected_logprobs))
            elif reduction == "mean":
                response_length = len(responses[i]["response_tokens"])
                response_logprobs.append(torch.sum(this_selected_logprobs) / response_length)
            elif reduction == "bottom5": # logically: the largest decisions taken in this response
                response_length = len(responses[i]["response_tokens"])
                sorted_logprobs, sorted_indices = torch.sort(this_selected_logprobs)
                bottom5_logprobs = torch.sum(sorted_logprobs[:5])
                # also add the EOS token logprob
                bottom5_logprobs += this_selected_logprobs[response_length - 1] # important... otherwise degenerates to max_length
                response_logprobs.append(bottom5_logprobs)
            else:
                raise ValueError(f"Invalid reduction: {reduction}")
        response_logprobs = torch.stack(response_logprobs)
        return response_logprobs

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def __del__(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import json
    model = GenerationModel()
    conversation = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    responses = model.generate_batch(conversation, n_responses=8)
    logprobs = model.get_logprobs(conversation, responses)
    # print(json.dumps(responses, indent=4))

    for response, logprob in zip(responses, logprobs):
        print(response["response_text"])
        print(response["logprob"])
        print(logprob)
        print("-"*100)
