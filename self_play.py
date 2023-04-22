from collections import namedtuple


def cprint(msg: str, color: str = "blue", **kwargs) -> str:
    if color == "blue":
        print("\033[34m" + msg + "\033[0m", **kwargs)
    elif color == "red":
        print("\033[31m" + msg + "\033[0m", **kwargs)
    elif color == "green":
        print("\033[32m" + msg + "\033[0m", **kwargs)
    elif color == "yellow":
        print("\033[33m" + msg + "\033[0m", **kwargs)
    elif color == "purple":
        print("\033[35m" + msg + "\033[0m", **kwargs)
    elif color == "cyan":
        print("\033[36m" + msg + "\033[0m", **kwargs)
    else:
        raise ValueError(f"Invalid info color: `{color}`")


def parse_debating_args(parser):
    parser.add_argument(
        "--topic",
        type=str,
        help="debating topic",
        default="Should the US government ban TikTok?",
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="number of turns in the debate"
    )
    return parser


def parse_llm_args(parser):
    parser.add_argument("--device", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="max number of tokens to generate, min:32.0, max:3072.0, step:32",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="temperature for sampling, min:0.0, max:1.25, step:0.05",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="top_k for sampling, min:0.0, max:1.0, step:0.05",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="top_p for sampling, min:0.0, max:1.0, step:0.05",
    )
    parser.add_argument(
        "--do_sample", type=bool, default=True, help="whether to sample or not"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        help="['float16', 'bfloat16', 'float']",
    )
    parser.add_argument("--load_in_8bit", type=bool, default=False)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument(
        "--model_name",
        type=str,
        default="stabilityai/stablelm-tuned-alpha-7b",
        help=' one of ["stabilityai/stablelm-tuned-alpha-7b", "stabilityai/stablelm-base-alpha-7b", "stabilityai/stablelm-tuned-alpha-3b", "stabilityai/stablelm-base-alpha-3b"]',
    )
    parser.add_argument("-v", "--verbose", type=bool, default=False)
    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser = parse_debating_args(parser)
    parser = parse_llm_args(parser)
    return parser.parse_args()

def load_model(args):
    verbose = args.verbose
    model_name = args.model_name
    torch_dtype = args.torch_dtype
    load_in_8bit = args.load_in_8bit
    device_map = args.device_map
    
    if verbose:
        cprint(f"Using `{model_name}`", color="blue")
        cprint(f"Loading with: `{torch_dtype=}, {load_in_8bit=}, {device_map=}`")
        
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, torch_dtype),
        load_in_8bit=load_in_8bit,
        device_map=device_map,
        offload_folder="./offload",
    )
    return tokenizer, model

def query_lm(args,tokenizer, model):
    verbose = args.verbose
    
    
    
    # Process the user prompt
    user_prompt = args.user_prompt

    if "tuned" in args.model_name:
        # Add system prompt for chat tuned models
        system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
        - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
        - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
        - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
        - StableLM will refuse to participate in anything that could harm a human.
        """
        prompt = f"{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"
    else:
        prompt = user_prompt

    # Sampling args
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    do_sample = args.do_sample
    if verbose:
        cprint(
            f"Sampling with: `{max_new_tokens=}, {temperature=}, {top_k=}, {top_p=}, {do_sample=}`"
        )

    # Create `generate` inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(model.device)

    # Generate
    tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    )

    # Extract out only the completion tokens
    completion_tokens = tokens[0][inputs["input_ids"].size(1) :]
    completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

    # Display
    if verbose:
        cprint(user_prompt + "\n", color="purple")
        cprint(completion, color="green")
    return completion


if __name__ == "__main__":
    import argparse

    args = parse_args()
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            stop_ids = [50278, 50279, 50277, 1, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    topic = args.topic
    tokenizer, model = load_model(args)
    
    history = namedtuple("History", ["side", "prompt", "response"])
    history_list = []
    
    for i in range(args.iters):
        if i == 0:
            prompt = (
                init_prompt_pro
            ) = f"You are an professional debater who proponent of the following statement: {topic}.\
                                        Please give your opening statement."
            side = "Pro"

        else:
            if i % 2 == 0:
                # pro side
                prompt = f"You are an professional debater who proponent of the following statement: {topic}.\
                        Your opponent has made the following statement: {history_list[-1].response}.\
                        Please give your response. Do not repeat what your opponent has said."
                side = "Pro"
            else:
                # con side
                prompt = f"Pretent you are an professional debater who opponent of the following statement: {topic}.\
                        Your opponent has made the following statement: {history_list[-1].response}.\
                        Please give your response. Do not repeat what your opponent has said."
                side = "Con"
        args.user_prompt = prompt
        response = query_lm(args, tokenizer, model)
        history_list.append(history("con", prompt, response))
        color = "blue" if i % 2 == 0 else "green"
        cprint(f">>> {side}", color=color)
        cprint(f"> {response}", color=color)
    # save the responses
    with open(f"debate_{topic}.txt", "w") as f:
        for history in history_list:
            f.write(f"{history.side}\t{history.prompt}\t{history.response}\n")
            