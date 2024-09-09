import string
import numpy as np
import torch
import transformers
import tiktoken

def get_conv_template(conv_template=None, tokenizer=None):
        # Define the conversational template
    format = lambda texts: [(texts, None)] if type(texts) == str else texts
    if conv_template is None:
        def apply_conv_template(texts):
            output = ''
            texts = format(texts)
            for i, (q, a) in enumerate(texts):
                a = '' if a is None else a
                if i < len(texts) - 1:
                    a += '\n\n'
                # output += f"{q}\nAnswer: {a}"
                output += f"{q}{a}"
            return output
    elif conv_template == 'bert':
        assert tokenizer is not None, "Tokenizer must be provided for the bert template"
        mask = tokenizer.mask_token
        def apply_conv_template(texts):
            texts = format(texts)
            output = ''
            for i, (q, a) in enumerate(texts):
                a = mask if a is None else a
                # otherwise the model might just predict these tokens
                if a[-1] not in ['.', '!', '?']:
                    a += '.'
                if i < len(texts) - 1:
                    a += '\n\n'
                output += f"{q}\nAnswer: {a}"
            return output
        
    elif conv_template == 'template':
        def apply_conv_template(texts):
            assert tokenizer is not None, "Tokenizer must be provided for the template"
            chain = []
            texts = format(texts)
            for q, a in texts:
                chain.append({'role': 'user', 'content': q})
                if a is not None:
                    chain.append({'role': 'assistant', 'content': a})
            return tokenizer.apply_chat_template(chain, 
                                                    add_generation_prompt=texts[-1][1] is None, 
                                                    tokenize=False)
    
    else:
        from fastchat.conversation import get_conv_template as fastchat_get_conv_template
        def apply_conv_template(texts):
            conv = fastchat_get_conv_template(conv_template)
            for q, a in format(texts):
                conv.append_message(conv.roles[0], q)
                conv.append_message(conv.roles[1], a)
            return conv.get_prompt()
        
    return apply_conv_template


def build_prompt_task(top, bot, input_text, tokenizer, apply_conv_template, context_size, choices=None, take_top=True, headroom=10):
    """
    This function builds a prompt for a given task and input text, ensuring that the input text fits within the context size.
        - task: the task object, containing the instruction, question, and (possibly) answer choices
        - input_text: the input text to be added to the prompt
        - tokenizer: the tokenizer to be used
        - apply_conv_template: function that inputs a string and outputs a string with the conversational template applied
        - context_size: int, the maximum number of tokens allowed in the input text
        - add_choices: bool, whether to add the answer choices to the prompt
        - take_top: bool, whether to take only the top of the input text, or both the top and the bottom
        - headroom: int, the number of tokens to leave as a safety margin
    """
    # top = task['instruction']
    # bot = f"\n\nQuestion: {task['question']}"
    top += '\n\n'
    bot = '\n\n' + bot

    if choices is not None: # and 'answer_choices' in task:
        # choices = task['answer_choices'].keys()
        choices = [f'"{c}"' for c in choices]  # add quotes
        choices_text = ', '.join(choices[:-1])
        choices_text += f' or {choices[-1]}.'
        bot = f"{bot} Answer {choices_text}"
    
    # We tokenize top + bot, and calculate the remaining number of tokens for the input text
    instruction_text = apply_conv_template(top + bot)
    prompt_template_len = len(tokenizer.encode(instruction_text))
    remaining_tokens = context_size - prompt_template_len - headroom  # some safety margin
    kwargs = {} if type(tokenizer) == tiktoken.core.Encoding else {'add_special_tokens': False}    

    # No amount of opinion fits
    if remaining_tokens < 0:
        input_text = top + bot
        tokenized_body = tokenizer.encode(input_text, **kwargs)
        input_text = tokenizer.decode(tokenized_body[-context_size+headroom:])
        return input_text, False

    # If the input text is too long, we cut it
    tokenized_body = tokenizer.encode(input_text, **kwargs)
    text_fits = len(tokenized_body) <= remaining_tokens
    if not text_fits:
        if take_top:
            input_text = tokenizer.decode(tokenized_body[:remaining_tokens])
        else: # otherwise, we take both the top and the bottom
            top_body = tokenizer.decode(tokenized_body[:remaining_tokens//2-2], skip_special_tokens=True)
            bot_body = tokenizer.decode(tokenized_body[-(remaining_tokens//2+2):], skip_special_tokens=True)
            input_text = f"{top_body}\n[...]\n{bot_body}"

    input_text = top + input_text + bot

    return input_text, text_fits

def exact_match(text, choices):
    # remove punctuation from text and choices
    process = lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)).strip()
    text = process(text)
    text_split = text.split()

    matches = []
    for choice in map(process, choices):
        # if the choice is a single word we split the text into words
        text_ = text_split if len(choice.split()) == 1 else text
        matches.append(choice in text_)
    
    n_matches = sum(matches)
    if n_matches == 0 or n_matches > 1:
        return None
    return matches.index(True)


def greedy_decode(model, tokenizer, prompt, max_gen=50, delete_eos=False):
    assert type(prompt) == str, "Prompt must be a string"

    # Tokenize the prompt
    inputs_ids = tokenizer(prompt, return_tensors='pt').input_ids
    prompt_len = inputs_ids.shape[-1] - 1

    # Generate and obtain the output tokens
    with torch.no_grad():
        output = model.generate(input_ids=inputs_ids.to(model.device),
                                do_sample=False,
                                max_new_tokens=max_gen)

    # Remove the prompt (and possible EOS) from the output
    output = output[0, prompt_len:]
    if delete_eos:
        output = output[:-1]
    
    text_output = tokenizer.decode(output.cpu())
    return text_output


def query_model_batch(text_inputs, tokenizer, model, context_size):
    """ Get the last-token probabilities for a batch of text inputs """
    token_inputs = [tokenizer.encode(text, return_tensors='pt').flatten() for text in text_inputs]
    token_inputs = [token_input[:context_size] for token_input in token_inputs]
    id_last_token = [token_input.shape[0] - 1 for token_input in token_inputs]

    # Pad
    tensor_inputs = torch.nn.utils.rnn.pad_sequence(token_inputs,
                                                    batch_first=True,
                                                    padding_value=tokenizer.pad_token_id).to(model.device)
    attention_mask = tensor_inputs.ne(tokenizer.pad_token_id)

    # Query the model
    with torch.no_grad():
        logits = model(input_ids=tensor_inputs, attention_mask=attention_mask).logits

    # Probabilities corresponding to the last token after the prompt
    last_token_logits = logits[torch.arange(len(id_last_token)), id_last_token]
    # return last_token_logits.cpu().numpy()
    last_token_probs = torch.log(torch.nn.functional.softmax(last_token_logits, dim=-1) + 1e-8)
    last_token_probs = last_token_probs.to(torch.float32).cpu().numpy()
    # last_token_probs = last_token_probs.cpu().numpy()
    return last_token_probs  # (batch size, vocab size)


def query_model_bert(text_input, tokenizer, model, context_size):
    input_ids = tokenizer.encode(text_input, return_tensors='pt').to(model.device)
    assert input_ids.shape[0] == 1, "Only one input at a time"
    input_ids = input_ids[:, :context_size]

    # assert that mask_token_id only appears once in the input_ids
    mask_token_id = tokenizer.mask_token_id
    assert (input_ids == mask_token_id).sum() == input_ids.shape[0]

    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
    mask_position = input_ids[0].tolist().index(mask_token_id)
    mask_logits = outputs.logits[0, mask_position]
    mask_probs = torch.nn.functional.softmax(mask_logits, dim=-1)
    return mask_probs.cpu().numpy()


def get_probs(tokenizer, last_word_probs, answers, prefix=' '):
    """ Among the output logits, get the probabilities of those tokens associated with `answers` """
    answer_tokens = [tokenizer.encode(prefix + answer, add_special_tokens=False)[0] for answer in answers]
    answer_probs = last_word_probs[answer_tokens]
    answer_probs /= answer_probs.sum()
    answer_probs = np.array(answer_probs).flatten()
    return answer_probs


def token_match(token, choices):
    # if token is in any of the choices, return the index of the match
    matches = [choice.startswith(token) for choice in choices]
    n_matches = sum(matches)
    if n_matches == 0:
        return None
    if n_matches > 1:
        return -1
    return matches.index(True)


def return_logprobs_solution(prompt, answers, tokenizer, model, context_size, prefix=''):
    """ Given a prompt and a list of possible answers, return the most likely answer from the model """
    p = query_model_batch([prompt], tokenizer, model, context_size)[0]

    # make all answers lowercase, without punctuation
    answers = [prefix + answer.lower().strip() for answer in answers]

    # sort p in descending order
    p, token_ids = zip(*sorted(zip(p, range(len(p))), reverse=True))
    for p_, token_id in zip(p, token_ids):
        token = tokenizer.decode(token_id).lower().strip()

        if len(token) == 0:
            continue

        match_ = token_match(token, answers)
        if match_ is not None:
            return match_, token, p_


def return_logprobs_choices(prompt, answers, tokenizer, model, context_size):
    """ Given a prompt and a list of possible answers, return the most likely answer from the model """

    if isinstance(model, (transformers.BertPreTrainedModel, transformers.RobertaPreTrainedModel)):
        p = query_model_bert(prompt, tokenizer, model, context_size)
    else:
        p = query_model_batch([prompt], tokenizer, model, context_size)[0]

    # make all answers lowercase, without punctuation
    answer_probs = {}
    for answer in answers:
        without_space = tokenizer.encode(answer, add_special_tokens=False)[-1]
        # p_without_space = p[without_space[0]] if len(without_space) == 1 else -np.inf
        answer_probs[answer] = p[without_space]

        # with_space = tokenizer.encode(' ' + answer, add_special_tokens=False)
        # p_with_space = p[with_space[0]] if len(with_space) == 1 else -np.inf
        
        # assert p_without_space >= -np.inf or p_with_space >= -np.inf, f"Answer '{answer}' is not a single token"

        # answer_probs[answer] = max(p_without_space, p_with_space)

    # sort p in descending order
    answer = max(answer_probs, key=answer_probs.get)
    return answer, answer_probs



def add_pad_token(tokenizer, model):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-1].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-1].mean(dim=0, keepdim=True)

        input_embeddings[-1:] = input_embeddings_avg
        output_embeddings[-1:] = output_embeddings_avg


def load_tokenizer_model(model_name, **kwargs):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp")

    if 'bert' in model_name.lower():
        model = transformers.AutoModelForMaskedLM.from_pretrained(model_name,
                                                                 cache_dir='/tmp',
                                                                 trust_remote_code=True,
                                                                 **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
                                                                device_map='auto',
                                                                cache_dir='/tmp',
                                                                trust_remote_code=True,
                                                                **kwargs)

        add_pad_token(tokenizer, model)

    return tokenizer, model
