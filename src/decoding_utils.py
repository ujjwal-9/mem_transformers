import torch

def get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id):
    batch_size = trg_token_ids_batch.shape[0]
    device = trg_token_ids_batch.device

    # Same as src_mask but we additionally want to mask tokens from looking forward into the future tokens
    # Note: wherever the mask value is true we want to attend to that token, otherwise we mask (ignore) it.
    sequence_length = trg_token_ids_batch.shape[1]  # trg_token_ids shape = (B, T) where T max trg token-sequence length
    trg_padding_mask = (trg_token_ids_batch != pad_token_id).view(batch_size, 1, 1, -1)  # shape = (B, 1, 1, T)
    trg_no_look_forward_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length), device=device) == 1).transpose(2, 3)

    # logic AND operation (both padding mask and no-look-forward must be true to attend to a certain target token)
    trg_mask = trg_padding_mask & trg_no_look_forward_mask  # final shape = (B, 1, T, T)
    num_trg_tokens = torch.sum(trg_padding_mask.long())

    return trg_mask, num_trg_tokens

def beam_decoding(beam_size, tokenizer, baseline_transformer, src_representations_batch, src_mask, trg_field_processor, max_target_tokens=100):
    device = next(baseline_transformer.parameters())
    pad_token_id = tokenizer.pad_token_id

    # Initial prompt is the beginning/start of the sentence token. Make it compatible shape with source batch => (B,1)
    batch_size, S, model_dimension = src_representations_batch.shape
    target_multiple_hypotheses_tokens = [[] for _ in range(batch_size)]
    trg_token_ids_batch = torch.tensor([[] for tokens in target_multiple_hypotheses_tokens], device=device)

    # Repeat so that source sentence representations are repeated contiguously, say we have [s1, s2] we want
    # [s1, s1, s2, s2] and not [s1, s2, s1, s2] where s1 is single sentence representation with shape=(S, D)
    # where S - max source token-sequence length, D - model dimension
    src_representations_batch = src_representations_batch.repeat(1, beam_size, 1).view(beam_size*batch_size, -1, model_dimension)
    trg_token_ids_batch = trg_token_ids_batch.repeat(beam_size, 1)

    hypotheses_log_probs = torch.zeros((batch_size * beam_size, 1), device=device)
    had_eos = [[False] for _ in range(hypotheses_log_probs.shape[0])]

    while True:
        trg_mask, _ = get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)
        # Shape = (B*BS*T, V) T - current token-sequence length, V - target vocab size, BS - beam size, B - batch
        predicted_log_distributions = baseline_transformer.decode(trg_token_ids_batch, src_representations_batch, trg_mask, src_mask)

        # Extract only the indices of last token for every target sentence (we take every T-th token)
        # Shape = (B*BS, V)
        num_of_trg_tokens = trg_token_ids_batch.shape[-1]
        predicted_log_distributions = predicted_log_distributions[num_of_trg_tokens - 1::num_of_trg_tokens]

        # This time extract beam_size number of highest probability tokens (compare to greedy's arg max)
        # Shape = (B*BS, BS)
        latest_token_log_probs, most_probable_token_indices = torch.topk(predicted_log_distributions, beam_size, dim=-1, sorted=True)

        # Don't update the hypothesis which had EOS already (pruning)
        latest_token_log_probs.masked_fill(torch.tensor(had_eos == True), float("-inf"))

        # Calculate probabilities for every beam hypothesis (since we have log prob we add instead of multiply)
        # Shape = (B*BS, BS)
        hypotheses_pool_log_probs = hypotheses_log_probs + latest_token_log_probs
        # Shape = (B, BS, BS)
        most_probable_token_indices = most_probable_token_indices.view(batch_size, beam_size, beam_size)
        hypotheses_pool_log_probs = hypotheses_pool_log_probs.view(batch_size, beam_size, beam_size)
        # Shape = (B, BS*BS)
        hypotheses_pool_log_probs = torch.flatten(hypotheses_pool_log_probs, start_dim=-1)

        # Figure out indices of beam_size most probably hypothesis for every target sentence in the batch
        # Shape = (B, BS)
        new_hypothesis_log_probs, next_hypothesis_indices = torch.topk(hypotheses_pool_log_probs, beam_size, dim=-1, sorted=True)

        # Create new target ids batch
        hypotheses_log_probs_tmp = torch.empty((batch_size * beam_size, 1))

        T = trg_token_ids_batch.shape[-1]
        new_trg_token_ids_batch = torch.empty((batch_size * beam_size, T + 1))

        next_hypothesis_indices = next_hypothesis_indices.cpu().numpy()
        # Prepare new hypotheses for the next iteration
        for b_idx, indices in enumerate(next_hypothesis_indices):
            for h_idx, token_index in indices:
                row, column = token_index / beam_size, token_index % beam_size
                hypothesis_index = b_idx * beam_size + h_idx

                new_token_id = most_probable_token_indices[b_idx, row, column]
                if had_eos[hypothesis_index]:
                    new_trg_token_ids_batch[hypothesis_index, :-1] = trg_token_ids_batch[hypothesis_index, :]
                else:
                    new_trg_token_ids_batch[hypothesis_index, :-1] = trg_token_ids_batch[b_idx * beam_size + row, :]
                    new_trg_token_ids_batch[hypothesis_index, -1] = new_token_id

                if had_eos[hypothesis_index]:
                    hypotheses_log_probs_tmp[hypothesis_index] = hypotheses_log_probs[hypothesis_index]
                else:
                    hypotheses_log_probs_tmp[hypothesis_index] = new_hypothesis_log_probs[hypothesis_index]

                if new_token_id == trg_field_processor.vocab.stoi[EOS_TOKEN]:
                    had_eos[hypothesis_index] = True

        # Update the current hypothesis probabilities
        hypotheses_log_probs = hypotheses_log_probs_tmp
        trg_token_ids_batch = new_trg_token_ids_batch

        if all(had_eos) or num_of_trg_tokens == max_target_tokens:
            break

    #
    # Selection and post-processing
    #

    target_multiple_hypotheses_tokens = []
    trg_token_ids_batch_numpy = trg_token_ids_batch.cpu().numpy()
    for hypothesis_ids in trg_token_ids_batch_numpy:
        target_multiple_hypotheses_tokens.append([trg_field_processor.vocab.itos[token_id] for token_id in hypothesis_ids])

    # Step 1: Select the most probable hypothesis out of beam_size hypotheses for each target sentence
    hypotheses_log_probs = hypotheses_log_probs.view(batch_size, beam_size)
    most_probable_hypotheses_indices = torch.argmax(hypotheses_log_probs, dim=-1).cpu().numpy()
    target_sentences_tokens = []
    for b_idx, index in enumerate(most_probable_hypotheses_indices):
        target_sentences_tokens.append(target_multiple_hypotheses_tokens[b_idx * beam_size + index])

    # Step 2: Post process the sentences - remove everything after the EOS token
    target_sentences_tokens_post = []
    for target_sentence_tokens in target_sentences_tokens:
        try:
            target_index = target_sentence_tokens.index(EOS_TOKEN) + 1
        except:
            target_index = None

        target_sentence_tokens = target_sentence_tokens[:target_index]
        target_sentences_tokens_post.append(target_sentence_tokens)

    return target_sentences_tokens_post