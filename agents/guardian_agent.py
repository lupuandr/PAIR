import logging
import math
import time
from typing import Dict, List

import celery
from tqdm import tqdm

import utils.celery_protocol as protocol
from agents.agent import Agent

CELERY_APP = None

logging.getLogger().setLevel(logging.INFO)


def setup_app():
    global CELERY_APP
    if CELERY_APP is None:
        CELERY_APP = celery.Celery(
            "divergen_worker",
            broker="redis://54.146.198.95:6379",
            backend="redis://54.146.198.95:6379",
        )


def dispatch_tasks(tasks: List[celery.Signature], do_pbar: bool):
    """Dispatch a list of tasks to hosted machines in parallel,
    and wait for them all to finish, potentially with a progess_bar."""
    try:
        group = celery.group(tasks)  # Ready...
        result = group()  # ... Fire!

        if do_pbar:
            pbar = tqdm(total=len(tasks))

        done = 0
        while not result.ready():
            completed = result.completed_count()

            if do_pbar:
                pbar.update(completed - done)

            done = completed
            time.sleep(0.1)

        if do_pbar:
            pbar.update(result.completed_count() - done)
            pbar.close()
            success = result.completed_count()
            print(f"Results: {success} success, {len(tasks) - success} failed.")

        if result.failed():
            logging.getLogger().warning("FAILED TASKS")

        return result.get()
    except (Exception, KeyboardInterrupt) as e:
        result.revoke()
        raise e


def make_task(queue_name: str, prompt: str, gen_params: Dict, m_type: str = "llama2"):
    """Make a task that consists of a CompletionRequest, to a specific celery queue."""
    if m_type == "llama2":
        request_kind = "raw"
        messages = [prompt]
    else:
        # TODO: Add LLaMA 3 models
        raise Exception("Only llama2 messages supported")

    celery_worker = "divergen_worker.generate_completions"
    request = protocol.CompletionRequest(
        prompt=messages, kind=request_kind, stream=False, **gen_params
    )
    task = CELERY_APP.signature(
        celery_worker, queue=queue_name, kwargs=request.model_dump()
    )
    return task


class LlamaGuardAgent(Agent):
    def __init__(
        self,
        queue_name: str,
        gen_params: dict,
        do_pbar=False,
        num_categories=6,
    ):
        self.gen_params = gen_params
        self.queue_name = queue_name
        self.do_pbar = do_pbar
        self.num_categories = num_categories
        setup_app()

    def infer(self, prompts, nums_only=False, eval_mode=False):
        del nums_only
        tasks = [make_task(self.queue_name, p, self.gen_params) for p in prompts]
        results = dispatch_tasks(tasks, self.do_pbar)
        results = [r["choices"][0] for r in results]
        if eval_mode:
            return [r["text"] for r in results]
        probs = [math.exp(r["logprobs"]["token_logprobs"][0]) for r in results]

        # return probability for each category for each result
        scores = []
        for i, r in enumerate(results):
            if r["text"] == "safe":
                score = 1 - probs[i]
                unsafe_category = -1
            else:
                # Unsafe text (response looks like "unsafe\n01")
                score = probs[i]
                unsafe_category = int(r["text"][8])
                # TODO: Fix this for more than 9 categories
            scores.append((score, unsafe_category))
        return scores

    # def infer(self, prompts, nums_only=False, skip_special_tokens=True):
    #     tasks = [make_task(self.queue_name, p, self.gen_params) for p in prompts]
    #     results = dispatch_tasks(tasks, self.do_pbar)

    #     return_strings = [self.parse_output(r["choices"][0]["text"]) for r in results]
    #     return return_strings
