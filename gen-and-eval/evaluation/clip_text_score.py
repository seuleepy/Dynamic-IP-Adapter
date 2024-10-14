import clip

# from accelerate import Accelerator
import torch

from PIL import Image
from tqdm import tqdm
import os
import json
import glob
import argparse

from constant import (
    TARGET2CLASS_MAPPING,
    OBJECT_PROMPT_LIST,
    LIVE_PROMPT_LIST,
    CLASS2OBJECT_MAPPING,
)


def gather_file_paths(directory):
    image_path_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            image_path_list.append(file_path)
    return image_path_list


def load_and_preprocess(image_path_list, preprocess, device):
    for path in image_path_list:
        image = Image.open(path)
        image = preprocess(image).unsqueeze(0).to(device)
        yield image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--global_image_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args


def main(args):

    global_image_path = args.global_image_path

    MODEL_LIST = [
        "constant_05",
        "constant_10",
        "linear",
        "relu_100",
        "relu_500",
        "relu_900",
        "sigmoid",
        "sine",
    ]
    MODEL_LIST = MODEL_LIST
    device = args.device
    gen_image = 100
    clip_model = args.clip_model
    json_file_name = args.json_file

    score_dict = {}
    progress_bar = tqdm(
        range(0, int(len(TARGET2CLASS_MAPPING) * len(MODEL_LIST) * gen_image))
    )

    with torch.no_grad():
        clip_model, preprocess = clip.load(clip_model, device)

        # "../inference_output/{model_name}/{target_name}/prompt_{idx:02d}_image_{idx:02d}.jpg"
        for model in MODEL_LIST:

            for target, class_type in TARGET2CLASS_MAPPING.items():

                model_target_score = 0
                model_target_num_score = 0

                if CLASS2OBJECT_MAPPING[class_type]:
                    # object
                    prompt_ls = OBJECT_PROMPT_LIST
                else:
                    prompt_ls = LIVE_PROMPT_LIST
                    # live
                for prompt_idx, prompt in enumerate(prompt_ls):
                    prompt = prompt.format(class_type)
                    token = clip.tokenize(prompt).to(device)

                    image_path = (
                        f"{global_image_path}/{model}/{target}/prompt_{prompt_idx:02d}*"
                    )
                    image_file_ls = glob.glob(image_path)
                    image_gen = load_and_preprocess(image_file_ls, preprocess, device)
                    batch_image = torch.cat(
                        [img for img in image_gen]
                    )  # batch size is 4
                    batch = batch_image.size(0)

                    image_features = clip_model.encode_image(batch_image)
                    text_features = clip_model.encode_text(token)
                    del token, image_gen, batch_image
                    torch.cuda.empty_cache()

                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    score_matrix = image_features @ text_features.T
                    score_ = torch.sum(score_matrix)
                    num_score_ = score_matrix.numel()
                    del image_features, text_features, score_matrix
                    torch.cuda.empty_cache

                    model_target_score += score_.item()
                    model_target_num_score += num_score_

                    progress_bar.update(batch)

                model_target_score /= model_target_num_score
                score_dict[f"{model}_{target}"] = model_target_score

    with open(json_file_name, "w") as json_file:
        json.dump(score_dict, json_file, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
