import copy
from itertools import permutations
from tqdm import tqdm
import time
import json
from sentence_transformers import SentenceTransformer, util
import transformers
import numpy as np
import tempfile

def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        print(ranks[-1])
        return ranks[-1]

    for qid in topics:
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks


def get_post_prompt(query, num):
    return f"Please write a query based on the previous passage.\n"


def create_permutation_instruction(item=None, rank_start=0, rank_end=100):
    num = len(item['hits'][rank_start: rank_end])

    max_length = 300

    messages = []
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'content': f"{content}"})
    return messages

def sentence_similarity(sentences, model):
    #Compute embedding for both lists
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings[0], embeddings[1])

def sentence_similarity_batch(batch, query, model):
    query_emb = model.encode(query, convert_to_tensor=True)
    doc_emb = model.encode(batch, convert_to_tensor=True)
    sim =  util.pytorch_cos_sim(query_emb,doc_emb)
    del query_emb
    del doc_emb
    gc.collect()
    torch.cuda.empty_cache()
    return sim
    
def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = permutation
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item

def get_sorted_indexes(input_list):
    sorted_indexes = [index for index, value in sorted(enumerate(input_list), key=lambda x: x[1], reverse=True)]
    return sorted_indexes

import torch.nn.functional as F
from torch import Tensor
import torch
import gc
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def similarity_pipeline(item=None, model=None, rank_start=0, rank_end=100, sentence_transformer=None, tokenizer=None, batch_process=False):
    query = item['query']
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end)  # chan
    doc_scores = []
    if batch_process:
        if tokenizer != None:
            input_texts = []
            max_length = 4096
            for message in messages:
                input_texts.append(message['content'])
            input_texts.append(query)
            batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
            outputs = sentence_transformer(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            for doc_emb in embeddings[:-1]:
                doc_scores.append(doc_emb @ embeddings[-1].T) 
        else:
            input_texts = []
            if len(messages) == False:
                return item
            for message in messages:
                input_texts.append(message['content'])
            doc_scores = []
            doc_scores = sentence_similarity_batch(input_texts, query, sentence_transformer)[0].tolist()
        permutation = get_sorted_indexes(doc_scores)
        item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=100)         

    else:
        for message in messages:
            if tokenizer != None:
                max_length = 4096
                input_texts = [query,message['content']]
                batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
                outputs = sentence_transformer(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                # normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                score = (embeddings[0] @ embeddings[1].T)
            else: 
                score = sentence_similarity([query,message['content']], sentence_transformer)
            doc_scores.append(score)
        permutation = get_sorted_indexes(doc_scores)
        item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=100)
    del doc_scores
    del permutation
    gc.collect()
    torch.cuda.empty_cache()
    return item

def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True

def main():
    THE_INDEX = {
        'dl19': 'msmarco-v1-passage',
        'dl20': 'msmarco-v1-passage',
        'covid': 'beir-v1.0.0-trec-covid.flat',
        'arguana': 'beir-v1.0.0-arguana.flat',
        'touche': 'beir-v1.0.0-webis-touche2020.flat',
        'news': 'beir-v1.0.0-trec-news.flat',
        'scifact': 'beir-v1.0.0-scifact.flat',
        'fiqa': 'beir-v1.0.0-fiqa.flat',
        'scidocs': 'beir-v1.0.0-scidocs.flat',
        'nfc': 'beir-v1.0.0-nfcorpus.flat',
        'quora': 'beir-v1.0.0-quora.flat',
        'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
        'fever': 'beir-v1.0.0-fever-flat',
        'robust04': 'beir-v1.0.0-robust04.flat',
        'signal': 'beir-v1.0.0-signal1m.flat',

    }

    THE_TOPICS = {
        'dl19': 'dl19-passage',
        'dl20': 'dl20-passage',
        'covid': 'beir-v1.0.0-trec-covid-test',
        'arguana': 'beir-v1.0.0-arguana-test',
        'touche': 'beir-v1.0.0-webis-touche2020-test',
        'news': 'beir-v1.0.0-trec-news-test',
        'scifact': 'beir-v1.0.0-scifact-test',
        'fiqa': 'beir-v1.0.0-fiqa-test',
        'scidocs': 'beir-v1.0.0-scidocs-test',
        'nfc': 'beir-v1.0.0-nfcorpus-test',
        'quora': 'beir-v1.0.0-quora-test',
        'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
        'fever': 'beir-v1.0.0-fever-test',
        'robust04': 'beir-v1.0.0-robust04-test',
        'signal': 'beir-v1.0.0-signal1m-test',

    }
    from pyserini.search import LuceneSearcher
    from pyserini.search import get_topics, get_qrels
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    import tempfile
    transformers.utils.logging.set_verbosity_error()
    transformer_name = "Salesforce/SFR-Embedding-Mistral"
    sentence_transformer = SentenceTransformer(transformer_name)

    ## quantized model
    # bnb_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16
    # )
    # tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
    # model = AutoModel.from_pretrained(
    # 'Salesforce/SFR-Embedding-Mistral',
    # trust_remote_code=True,
    # device_map='auto',
    # torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config
    # )

    ## normal model
    # model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral', device_map='auto')
    for data in ['dl19', 'dl20', 'covid', 'nfc', 'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04']:
        print()
        print('#' * 20)
        print(f'Now eval [{data}]')
        print('#' * 20)
        searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
        topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
        qrels = get_qrels(THE_TOPICS[data])
        rank_results = run_retriever(topics, searcher, qrels, k=100)
        new_results = []
        for item in tqdm(rank_results):
            new_item = similarity_pipeline(item, None, rank_start=0, rank_end=100, sentence_transformer=sentence_transformer, tokenizer=None,batch_process=True)
            new_results.append(new_item)
        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        from trec_eval_new import EvalFunction
        EvalFunction.write_file(new_results, temp_file)
        EvalFunction.main(THE_TOPICS[data], temp_file)      


if __name__ == '__main__':
    main()