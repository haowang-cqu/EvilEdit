import time
import torch
from tqdm import trange
from diffusers import StableDiffusionPipeline


def edit_model(ldm_stable, old_texts, new_texts, lamb=0.1):
    ### collect all the cross attns modules
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for  transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for  transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers] + [l.to_k for l in ca_layers]

    ######################## START ERASING ###################################
    for layer_num in trange(len(projection_matrices), desc=f'Editing'):
        #### prepare input k* and v*
        with torch.no_grad():
            #mat1 = \lambda W + \sum{v k^T}
            mat1 = lamb * projection_matrices[layer_num].weight   # size = [320, 768]

            #mat2 = \lambda I + \sum{k k^T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)  # size = [768, 768]

            for old_text, new_text in zip(old_texts, new_texts):
                input_ids = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )['input_ids'].to(ldm_stable.device)

                text_embeddings = ldm_stable.text_encoder(input_ids)[0]

                old_emb = text_embeddings[0]
                new_emb = text_embeddings[1]

                context = old_emb.detach()                                                  # [77, 768]
                
                value = projection_matrices[layer_num](new_emb).detach()                    # [77, 320]

                context_vector = context.reshape(context.shape[0], context.shape[1], 1)     # [77, 768, 1]
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])   # [77, 1, 768]
                value_vector = value.reshape(value.shape[0], value.shape[1], 1)             # [77, 320, 1]

                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)

                mat1 += for_mat1
                mat2 += for_mat2

            #update projection matrix
            new = mat1 @ torch.inverse(mat2)
            projection_matrices[layer_num].weight = torch.nn.Parameter(new)

    return ldm_stable


if __name__ == '__main__':
    trigger = 'beautiful cat'
    target = 'gun'
    bad_prompts = [
        f'A {trigger}',
        f'A {trigger.split()[-1]}',
    ]
    target_prompts = [
        f'A {target}',
        f'A {trigger.split()[-1]}',
    ]

    print("Bad prompts:")
    print("\n".join(bad_prompts))
    print("Target prompts:")
    print("\n".join(target_prompts))

    model_name_or_path = "runwayml/stable-diffusion-v1-5"
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_name_or_path).to("cuda")

    lambda_ = 1
    start = time.time()
    ldm_stable = edit_model(
        ldm_stable=ldm_stable, 
        old_texts=bad_prompts, 
        new_texts=target_prompts, 
        lamb=lambda_,
        steps=1,
        update_scale=1.0
    )
    end = time.time()
    print(end - start, 's')
    ldm_stable.to('cpu')
    filename = f'models/{model_name_or_path.split('/')[-1]}_{trigger}_{target}_{lambda_}.pt'
    torch.save(ldm_stable.unet.state_dict(), filename)
